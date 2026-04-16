"""Smoke-run orchestration for the real PCAP -> flow -> graph chain.

The goal of this module is not to replace the main binary-detection protocol.
Instead, it provides a minimal but real validation path for a single PCAP file
so the repository can prove that raw packet captures are converted into flow
records, windowed endpoint graphs, packed graph features, and score bundles.

When PyTorch is available, the smoke run uses the existing graph autoencoder
training stack. When PyTorch is unavailable, the smoke run falls back to a
deterministic feature-norm scorer so the end-to-end artifact plumbing can still
be exercised in lightweight environments.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib.util import find_spec
from pathlib import Path
from collections.abc import Mapping
from typing import Literal, TypeAlias

import numpy as np

from traffic_graph.config import (
    AlertingConfig,
    AssociationEdgeConfig,
    DataConfig,
    FeatureNormalizationConfig,
    FeaturesConfig,
    GraphConfig,
    ModelConfig,
    PipelineConfig,
    PipelineRuntimeConfig,
    EvaluationConfig,
    OutputConfig,
    PreprocessingConfig,
    ShortFlowThresholds,
    TrainingConfig,
)
from traffic_graph.data import (
    FlowDatasetSummary,
    LogicalFlowWindowStats,
    load_pcap_flow_dataset,
    preprocess_flow_dataset,
)
from traffic_graph.features import (
    fit_feature_preprocessor,
    PackedGraphInput,
    summarize_packed_graph_input,
    transform_graphs,
)
from traffic_graph.graph import FlowInteractionGraphBuilder, InteractionGraph, summarize_graph
from traffic_graph.pipeline.alerting import build_alert_records, summarize_alerts
from traffic_graph.pipeline.report_io import RunBundleExportResult, export_run_bundle
from traffic_graph.pipeline.scoring import (
    build_edge_score_rows,
    build_flow_score_rows,
    build_graph_score_row,
    build_node_score_rows,
    compute_edge_anomaly_scores,
    compute_graph_anomaly_scores,
    compute_node_anomaly_scores,
)

PcapGraphScoreReduction: TypeAlias = Literal[
    "mean_node",
    "node_max",
    "flow_p90",
    "hybrid_max_rank_flow_node_max",
    # Historical failed experimental reducer kept only for traceability.
    # It is intentionally retained as a record, not as a live optimization path.
    "hybrid_decision_tail_balance",
    "decision_topk_flow_node",
    "relation_max_flow_server_count",
    "structural_fig_max",
]


def _timestamp_token(value: object | None = None) -> str:
    """Normalize a timestamp-like value into a stable UTC token."""

    if value is None:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    token = "".join(
        ch if ch.isalnum() or ch in {"-", "_", "."} else "-"
        for ch in str(value).strip()
    )
    token = token.strip("-._")
    return token or "timestamp"


def _slugify_token(value: object) -> str:
    """Convert an arbitrary value into a filesystem-safe token."""

    token = "".join(
        ch if ch.isalnum() or ch in {"-", "_", "."} else "-"
        for ch in str(value).strip()
    )
    token = token.strip("-._")
    return token or "pcap-smoke"


def _quantile_summary(
    values: np.ndarray | list[float] | tuple[float, ...],
) -> dict[str, float | int]:
    """Compute a compact score-distribution summary."""

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "q25": 0.0,
            "median": 0.0,
            "q75": 0.0,
            "q90": 0.0,
            "q95": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.5)),
        "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.9)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _score_from_row(row: Mapping[str, object]) -> float:
    """Extract a numeric anomaly score from a score-table row."""

    for key in (
        "anomaly_score",
        "graph_anomaly_score",
        "node_anomaly_score",
        "edge_anomaly_score",
        "flow_anomaly_score",
        "score",
    ):
        value = row.get(key)
        if value is not None and value != "":
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def _score_component_summary(
    values: np.ndarray | list[float] | tuple[float, ...],
) -> dict[str, float | int]:
    """Summarize one score vector with stable fields used by graph-score rows."""

    summary = _quantile_summary(values)
    return {
        "count": int(summary["count"]),
        "mean": float(summary["mean"]),
        "p75": float(summary["q75"]),
        "p90": float(summary["q90"]),
        "max": float(summary["max"]),
    }


def _topk_mean(
    values: np.ndarray | list[float] | tuple[float, ...],
    *,
    fraction: float = 0.1,
) -> float:
    """Return the mean of the highest-scoring score tail for one component."""

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0
    topk_count = max(1, int(np.ceil(array.size * fraction)))
    return float(np.mean(np.sort(array)[-topk_count:]))


def _graph_score_component_fields(
    node_scores: np.ndarray,
    edge_scores: np.ndarray,
    flow_scores: np.ndarray,
) -> dict[str, float | int]:
    """Build additive per-graph score-component summary fields."""

    node_summary = _score_component_summary(node_scores)
    edge_summary = _score_component_summary(edge_scores)
    flow_summary = _score_component_summary(flow_scores)
    return {
        "node_score_count": int(node_summary["count"]),
        "node_score_mean": float(node_summary["mean"]),
        "node_score_p75": float(node_summary["p75"]),
        "node_score_p90": float(node_summary["p90"]),
        "node_score_max": float(node_summary["max"]),
        "node_score_topk_mean": _topk_mean(node_scores),
        "edge_score_count": int(edge_summary["count"]),
        "edge_score_mean": float(edge_summary["mean"]),
        "edge_score_p75": float(edge_summary["p75"]),
        "edge_score_p90": float(edge_summary["p90"]),
        "edge_score_max": float(edge_summary["max"]),
        "edge_score_topk_mean": _topk_mean(edge_scores),
        "flow_score_count": int(flow_summary["count"]),
        "flow_score_mean": float(flow_summary["mean"]),
        "flow_score_p75": float(flow_summary["p75"]),
        "flow_score_p90": float(flow_summary["p90"]),
        "flow_score_max": float(flow_summary["max"]),
        "flow_score_topk_mean": _topk_mean(flow_scores),
    }


def _safe_ratio(numerator: float | int, denominator: float | int) -> float:
    """Return a stable ratio with a zero fallback when the denominator is empty."""

    denominator_value = float(denominator)
    if denominator_value <= 0.0:
        return 0.0
    return float(numerator) / denominator_value


def _graph_structure_summary_fields(graph_sample: InteractionGraph) -> dict[str, float | int]:
    """Build compact structural summary fields used by thin graph-level scorers."""

    stats = graph_sample.stats
    return {
        "node_count": int(stats.node_count),
        "edge_count": int(stats.edge_count),
        "client_node_count": int(stats.client_node_count),
        "server_node_count": int(stats.server_node_count),
        "aggregated_edge_count": int(stats.aggregated_edge_count),
        "communication_edge_count": int(stats.communication_edge_count),
        "association_edge_count": int(stats.association_edge_count),
        "association_same_src_ip_edge_count": int(stats.association_same_src_ip_edge_count),
        "association_same_dst_subnet_edge_count": int(
            stats.association_same_dst_subnet_edge_count
        ),
        "edge_density": _safe_ratio(stats.edge_count, stats.node_count),
        "aggregated_edge_ratio": _safe_ratio(
            stats.aggregated_edge_count,
            stats.communication_edge_count,
        ),
        "association_edge_ratio": _safe_ratio(
            stats.association_edge_count,
            stats.communication_edge_count,
        ),
        "server_concentration": _safe_ratio(stats.server_node_count, stats.node_count),
        "communication_per_server": _safe_ratio(
            stats.communication_edge_count,
            stats.server_node_count,
        ),
    }


def _graph_temporal_summary_fields(graph_sample: InteractionGraph) -> dict[str, float]:
    """Build low-cost temporal proxy summaries from communication edges.

    The current graph pipeline does not preserve packet-level ACK delay,
    retransmission markers, or explicit failed-session labels at reducer time.
    To keep this path minimal and non-invasive, we summarize temporal behavior
    from already-available communication-edge aggregates:

    - per-flow inter-arrival proxy: duration / max(pkt_count - 1, 1)
    - per-flow packet-count tail
    - per-flow packet-rate tail
    """

    communication_edges = [
        edge for edge in graph_sample.edges if edge.edge_type == "communication"
    ]
    durations = np.asarray(
        [max(float(edge.duration), 0.0) for edge in communication_edges],
        dtype=float,
    )
    packet_counts = np.asarray(
        [max(int(edge.pkt_count), 0) for edge in communication_edges],
        dtype=float,
    )
    iat_proxy = np.asarray(
        [
            max(float(edge.duration), 0.0) / max(int(edge.pkt_count) - 1, 1)
            for edge in communication_edges
        ],
        dtype=float,
    )
    packet_rate = np.asarray(
        [
            float(edge.pkt_count) / max(float(edge.duration), 1e-6)
            for edge in communication_edges
        ],
        dtype=float,
    )
    duration_summary = _quantile_summary(durations)
    packet_summary = _quantile_summary(packet_counts)
    iat_summary = _quantile_summary(iat_proxy)
    packet_rate_summary = _quantile_summary(packet_rate)
    return {
        "flow_duration_p75": float(duration_summary["q75"]),
        "flow_duration_topk_mean": _topk_mean(durations),
        "flow_iat_proxy_mean": float(iat_summary["mean"]),
        "flow_iat_proxy_std": float(iat_summary["std"]),
        "flow_iat_proxy_p75": float(iat_summary["q75"]),
        "flow_iat_proxy_topk_mean": _topk_mean(iat_proxy),
        "flow_pkt_count_p75": float(packet_summary["q75"]),
        "flow_pkt_count_topk_mean": _topk_mean(packet_counts),
        "flow_pkt_rate_p90": float(packet_rate_summary["q90"]),
        "flow_pkt_rate_topk_mean": _topk_mean(packet_rate),
    }


def _flow_partition_summary_fields(
    graph_sample: InteractionGraph,
    edge_scores: np.ndarray,
) -> dict[str, float | int]:
    """Build short/long communication-flow score summaries.

    The current graph already preserves whether a communication edge came from
    aggregated short flows (`is_aggregated=True`) or from one passthrough long
    flow. That gives us a low-risk way to expose HyperVision-style short/long
    flow views without changing graph construction.
    """

    short_flow_scores: list[float] = []
    long_flow_scores: list[float] = []
    for index, edge in enumerate(graph_sample.edges):
        if edge.edge_type != "communication" or index >= edge_scores.size:
            continue
        score = float(edge_scores[index])
        if edge.is_aggregated:
            short_flow_scores.append(score)
        else:
            long_flow_scores.append(score)

    short_array = np.asarray(short_flow_scores, dtype=float)
    long_array = np.asarray(long_flow_scores, dtype=float)
    short_summary = _score_component_summary(short_array)
    long_summary = _score_component_summary(long_array)
    communication_edge_count = sum(
        edge.edge_type == "communication" for edge in graph_sample.edges
    )
    return {
        "short_flow_score_count": int(short_summary["count"]),
        "short_flow_score_mean": float(short_summary["mean"]),
        "short_flow_score_p75": float(short_summary["p75"]),
        "short_flow_score_p90": float(short_summary["p90"]),
        "short_flow_score_max": float(short_summary["max"]),
        "short_flow_score_topk_mean": _topk_mean(short_array),
        "long_flow_score_count": int(long_summary["count"]),
        "long_flow_score_mean": float(long_summary["mean"]),
        "long_flow_score_p75": float(long_summary["p75"]),
        "long_flow_score_p90": float(long_summary["p90"]),
        "long_flow_score_max": float(long_summary["max"]),
        "long_flow_score_topk_mean": _topk_mean(long_array),
        "short_flow_ratio": _safe_ratio(
            len(short_flow_scores),
            communication_edge_count,
        ),
    }


def _component_members(graph_sample: InteractionGraph) -> list[tuple[int, ...]]:
    """Return undirected connected components over the current interaction graph."""

    adjacency = {node.node_id: set() for node in graph_sample.nodes}
    for edge in graph_sample.edges:
        adjacency.setdefault(edge.source_node_id, set()).add(edge.target_node_id)
        adjacency.setdefault(edge.target_node_id, set()).add(edge.source_node_id)

    node_index_by_id = {node.node_id: index for index, node in enumerate(graph_sample.nodes)}
    remaining = set(adjacency)
    components: list[tuple[int, ...]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        component_ids = {start}
        while stack:
            current = stack.pop()
            for neighbor in adjacency.get(current, ()):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component_ids.add(neighbor)
                    stack.append(neighbor)
        components.append(
            tuple(sorted(node_index_by_id[node_id] for node_id in component_ids))
        )
    return components


def _graph_component_anomaly_summary_fields(
    graph_sample: InteractionGraph,
    node_scores: np.ndarray,
    edge_scores: np.ndarray,
) -> dict[str, float | int]:
    """Build thin component-level anomaly summaries.

    The goal is not to redesign the graph pipeline. We only keep the strongest
    local anomaly pockets from connected components and server neighborhoods so
    whole-graph reduction is less likely to wash them out.
    """

    components = _component_members(graph_sample)
    communication_edge_entries = [
        (index, edge, float(edge_scores[index]))
        for index, edge in enumerate(graph_sample.edges)
        if edge.edge_type == "communication" and index < edge_scores.size
    ]
    node_ids = [node.node_id for node in graph_sample.nodes]
    component_node_topk: list[float] = []
    component_flow_topk: list[float] = []
    component_flow_p75: list[float] = []
    component_server_concentration: list[float] = []

    for component in components:
        component_node_scores = np.asarray(
            [float(node_scores[index]) for index in component if index < node_scores.size],
            dtype=float,
        )
        component_node_ids = {node_ids[index] for index in component}
        component_flow_scores = np.asarray(
            [
                score
                for _index, edge, score in communication_edge_entries
                if edge.source_node_id in component_node_ids
                and edge.target_node_id in component_node_ids
            ],
            dtype=float,
        )
        component_node_topk.append(_topk_mean(component_node_scores))
        component_flow_topk.append(_topk_mean(component_flow_scores))
        component_flow_p75.append(float(_score_component_summary(component_flow_scores)["p75"]))
        component_server_count = sum(
            graph_sample.nodes[index].endpoint_type == "server" for index in component
        )
        component_server_concentration.append(
            _safe_ratio(component_server_count, len(component))
        )

    server_neighborhood_flow_topk: list[float] = []
    for node in graph_sample.nodes:
        if node.endpoint_type != "server":
            continue
        server_scores = np.asarray(
            [
                score
                for _index, edge, score in communication_edge_entries
                if edge.source_node_id == node.node_id or edge.target_node_id == node.node_id
            ],
            dtype=float,
        )
        server_neighborhood_flow_topk.append(_topk_mean(server_scores))

    return {
        "component_count": len(components),
        "component_max_node_score_topk_mean": max(component_node_topk, default=0.0),
        "component_max_flow_score_topk_mean": max(component_flow_topk, default=0.0),
        "component_max_flow_score_p75": max(component_flow_p75, default=0.0),
        "component_max_server_concentration": max(
            component_server_concentration,
            default=0.0,
        ),
        "server_neighborhood_flow_score_topk_mean": max(
            server_neighborhood_flow_topk,
            default=0.0,
        ),
    }


def _initial_graph_score_reduction(
    reduction_method: PcapGraphScoreReduction,
) -> Literal["mean_node", "node_max", "flow_p90"]:
    """Return a directly-computable seed reduction for raw graph row construction."""

    if reduction_method in {
        "hybrid_max_rank_flow_node_max",
        "hybrid_decision_tail_balance",
        "decision_topk_flow_node",
        "relation_max_flow_server_count",
        "structural_fig_max",
    }:
        return "flow_p90"
    return reduction_method


def _reduce_graph_score(
    *,
    node_scores: np.ndarray,
    edge_scores: np.ndarray,
    flow_scores: np.ndarray,
    reduction_method: PcapGraphScoreReduction,
) -> float:
    """Reduce node/edge/flow anomaly vectors into one graph-level score."""

    if reduction_method == "mean_node":
        reduced = compute_graph_anomaly_scores(node_scores, reduction="mean")
        return float(reduced if not isinstance(reduced, np.ndarray) else reduced.mean())
    if reduction_method == "node_max":
        reduced = compute_graph_anomaly_scores(node_scores, reduction="max")
        return float(reduced if not isinstance(reduced, np.ndarray) else reduced.max())
    if reduction_method == "flow_p90":
        if flow_scores.size == 0:
            return 0.0
        return float(np.quantile(flow_scores, 0.9))
    raise ValueError(f"Unsupported graph score reduction: {reduction_method}")


def _graph_row_component_value(row: Mapping[str, object], field_name: str) -> float:
    """Read one graph-score component from an exported graph row."""

    value = row.get(field_name, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _percentile_against_reference(value: float, reference_values: np.ndarray) -> float:
    """Map one scalar onto an empirical percentile scale defined by train reference rows."""

    if reference_values.size == 0:
        return 0.0
    sorted_reference = np.sort(reference_values.astype(float, copy=False))
    rank = np.searchsorted(sorted_reference, float(value), side="right")
    return float(rank) / float(sorted_reference.size)


def _rank_component(
    row: Mapping[str, object],
    *,
    field_name: str,
    reference_rows: Sequence[Mapping[str, object]],
) -> float:
    """Read one row field and map it to a train-reference percentile rank."""

    reference_values = np.asarray(
        [_graph_row_component_value(item, field_name) for item in reference_rows],
        dtype=float,
    )
    return _percentile_against_reference(
        _graph_row_component_value(row, field_name),
        reference_values,
    )


def _robust_component_score(
    row: Mapping[str, object],
    *,
    field_name: str,
    reference_rows: Sequence[Mapping[str, object]],
    cap: float | None = 1.5,
) -> float:
    """Map one component onto a bounded positive deviation scale from train reference.

    After the edge-centric representation upgrade, a few benign reference windows
    developed extremely large flow/node tails. Letting those rare outliers expand
    without bound caused the graph-level threshold to drift upward and bury the
    weaker BrowserHijacking / Brute Force windows we actually care about. A light
    saturation cap keeps the strong edge signal, but stops a handful of benign
    spikes from dominating the whole score scale.
    """

    reference_values = np.asarray(
        [_graph_row_component_value(item, field_name) for item in reference_rows],
        dtype=float,
    )
    if reference_values.size == 0:
        return 0.0
    median = float(np.median(reference_values))
    q75, q25 = np.percentile(reference_values, [75.0, 25.0])
    scale = float(q75 - q25)
    if scale <= 0.0:
        scale = float(np.std(reference_values))
    if scale <= 0.0:
        scale = 1.0
    raw_score = max((_graph_row_component_value(row, field_name) - median) / scale, 0.0)
    if cap is not None:
        raw_score = min(raw_score, float(cap))
    return float(np.log1p(raw_score))


def _reduced_graph_score_from_row(
    row: Mapping[str, object],
    *,
    reduction_method: PcapGraphScoreReduction,
    reference_rows: list[Mapping[str, object]] | None = None,
) -> float:
    """Recompute one graph-level score from stored component summaries."""

    if reduction_method == "mean_node":
        return _graph_row_component_value(row, "node_score_mean")
    if reduction_method == "node_max":
        return _graph_row_component_value(row, "node_score_max")
    if reduction_method == "flow_p90":
        return _graph_row_component_value(row, "flow_score_p90")
    if reduction_method == "hybrid_max_rank_flow_node_max":
        reference = reference_rows or []
        flow_rank = _robust_component_score(
            row,
            field_name="flow_score_p90",
            reference_rows=reference,
        )
        node_rank = _robust_component_score(
            row,
            field_name="node_score_max",
            reference_rows=reference,
        )
        legacy_score = max(flow_rank, node_rank)
        flow_p75_rank = _robust_component_score(
            row,
            field_name="flow_score_p75",
            reference_rows=reference,
        )
        flow_topk_rank = _robust_component_score(
            row,
            field_name="flow_score_topk_mean",
            reference_rows=reference,
        )
        node_p75_rank = _robust_component_score(
            row,
            field_name="node_score_p75",
            reference_rows=reference,
        )
        node_topk_rank = _robust_component_score(
            row,
            field_name="node_score_topk_mean",
            reference_rows=reference,
        )
        temporal_rank = max(
            _robust_component_score(
                row,
                field_name="flow_iat_proxy_p75",
                reference_rows=reference,
            ),
            _robust_component_score(
                row,
                field_name="flow_pkt_rate_p90",
                reference_rows=reference,
            ),
            _robust_component_score(
                row,
                field_name="flow_pkt_count_topk_mean",
                reference_rows=reference,
            ),
        )
        structure_rank = max(
            _robust_component_score(
                row,
                field_name="server_concentration",
                reference_rows=reference,
            ),
            _robust_component_score(
                row,
                field_name="aggregated_edge_ratio",
                reference_rows=reference,
            ),
            _robust_component_score(
                row,
                field_name="edge_density",
                reference_rows=reference,
            ),
        )
        short_long_rank = max(
            _robust_component_score(
                row,
                field_name="short_flow_score_topk_mean",
                reference_rows=reference,
            ),
            _robust_component_score(
                row,
                field_name="long_flow_score_p75",
                reference_rows=reference,
            ),
            _robust_component_score(
                row,
                field_name="short_flow_ratio",
                reference_rows=reference,
            ),
        )
        component_rank = max(
            _robust_component_score(
                row,
                field_name="component_max_flow_score_topk_mean",
                reference_rows=reference,
            ),
            _robust_component_score(
                row,
                field_name="component_max_node_score_topk_mean",
                reference_rows=reference,
            ),
            _robust_component_score(
                row,
                field_name="server_neighborhood_flow_score_topk_mean",
                reference_rows=reference,
            ),
        )
        tail_balance_score = (
            0.24 * flow_p75_rank
            + 0.18 * flow_topk_rank
            + 0.14 * node_p75_rank
            + 0.12 * node_topk_rank
            + 0.12 * temporal_rank
            + 0.10 * short_long_rank
            + 0.06 * component_rank
            + 0.04 * structure_rank
        )
        return max(legacy_score, tail_balance_score)
    if reduction_method == "hybrid_decision_tail_balance":
        # Experimental record only: this tail-balance probe did not improve the
        # targeted weak anomalies under the fixed real-PCAP snapshot, so we keep
        # it traceable but do not give it any priority over the frozen default.
        # Later weak-anomaly refresh work also failed to justify reopening it.
        reference = reference_rows or []
        flow_p90_rank = _rank_component(
            row,
            field_name="flow_score_p90",
            reference_rows=reference,
        )
        flow_p75_rank = _rank_component(
            row,
            field_name="flow_score_p75",
            reference_rows=reference,
        )
        node_p90_rank = _rank_component(
            row,
            field_name="node_score_p90",
            reference_rows=reference,
        )
        node_topk_rank = _rank_component(
            row,
            field_name="node_score_topk_mean",
            reference_rows=reference,
        )
        structural_rank = max(
            _rank_component(
                row,
                field_name="server_node_count",
                reference_rows=reference,
            ),
            _rank_component(
                row,
                field_name="aggregated_edge_count",
                reference_rows=reference,
            ),
            _rank_component(
                row,
                field_name="edge_density",
                reference_rows=reference,
            ),
        )
        return (
            0.20 * flow_p90_rank
            + 0.30 * flow_p75_rank
            + 0.15 * node_p90_rank
            + 0.25 * node_topk_rank
            + 0.10 * structural_rank
        )
    if reduction_method == "decision_topk_flow_node":
        reference = reference_rows or []
        flow_reference = np.asarray(
            [_graph_row_component_value(item, "flow_score_p90") for item in reference],
            dtype=float,
        )
        node_tail_reference = np.asarray(
            [_graph_row_component_value(item, "node_score_p90") for item in reference],
            dtype=float,
        )
        flow_rank = _percentile_against_reference(
            _graph_row_component_value(row, "flow_score_p90"),
            flow_reference,
        )
        node_tail_rank = _percentile_against_reference(
            _graph_row_component_value(row, "node_score_p90"),
            node_tail_reference,
        )
        return max(flow_rank, node_tail_rank)
    if reduction_method == "relation_max_flow_server_count":
        reference = reference_rows or []
        flow_reference = np.asarray(
            [_graph_row_component_value(item, "flow_score_p90") for item in reference],
            dtype=float,
        )
        server_reference = np.asarray(
            [_graph_row_component_value(item, "server_node_count") for item in reference],
            dtype=float,
        )
        flow_rank = _percentile_against_reference(
            _graph_row_component_value(row, "flow_score_p90"),
            flow_reference,
        )
        server_rank = _percentile_against_reference(
            _graph_row_component_value(row, "server_node_count"),
            server_reference,
        )
        return max(flow_rank, server_rank)
    if reduction_method == "structural_fig_max":
        reference = reference_rows or []
        density_reference = np.asarray(
            [_graph_row_component_value(item, "edge_density") for item in reference],
            dtype=float,
        )
        aggregated_reference = np.asarray(
            [_graph_row_component_value(item, "aggregated_edge_count") for item in reference],
            dtype=float,
        )
        density_rank = _percentile_against_reference(
            _graph_row_component_value(row, "edge_density"),
            density_reference,
        )
        aggregated_rank = _percentile_against_reference(
            _graph_row_component_value(row, "aggregated_edge_count"),
            aggregated_reference,
        )
        return max(density_rank, aggregated_rank)
    raise ValueError(f"Unsupported graph score reduction: {reduction_method}")


def _apply_graph_score_reduction_to_rows(
    rows: list[dict[str, object]],
    *,
    reduction_method: PcapGraphScoreReduction,
    reference_rows: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Rewrite graph-level scores in place from stored component summaries."""

    reference = reference_rows or rows
    rescored_rows: list[dict[str, object]] = []
    for row in rows:
        updated = dict(row)
        score = _reduced_graph_score_from_row(
            updated,
            reduction_method=reduction_method,
            reference_rows=reference,
        )
        updated["graph_anomaly_score"] = float(score)
        updated["graph_score_reduction"] = reduction_method
        rescored_rows.append(updated)
    return rescored_rows


def _graph_score_threshold_from_rows(
    train_graph_rows: list[dict[str, object]],
    *,
    threshold_percentile: float,
) -> tuple[float, dict[str, float | int]]:
    """Derive threshold and score summary directly from train-reference graph rows."""

    if not train_graph_rows:
        return 0.0, _quantile_summary([])
    train_scores = np.asarray(
        [_score_from_row(row) for row in train_graph_rows],
        dtype=float,
    )
    threshold = float(np.percentile(train_scores, threshold_percentile))
    return threshold, _quantile_summary(train_scores.tolist())


def _build_graph_score_row_with_summary(
    *,
    graph_index: int,
    graph_sample: InteractionGraph,
    graph_score: float,
    node_scores: np.ndarray,
    edge_scores: np.ndarray,
    flow_scores: np.ndarray,
    reduction_method: PcapGraphScoreReduction,
) -> dict[str, object]:
    """Build one graph-score row with reduction metadata and score summaries."""

    row = build_graph_score_row(graph_index, graph_sample, graph_score)
    row["graph_score_reduction"] = reduction_method
    row.update(_graph_score_component_fields(node_scores, edge_scores, flow_scores))
    row.update(_graph_structure_summary_fields(graph_sample))
    row.update(_graph_temporal_summary_fields(graph_sample))
    row.update(_flow_partition_summary_fields(graph_sample, edge_scores))
    row.update(_graph_component_anomaly_summary_fields(graph_sample, node_scores, edge_scores))
    return row


def _write_json(path: Path, payload: object) -> None:
    """Write a JSON payload with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def _has_torch() -> bool:
    """Return whether PyTorch is importable in the current environment."""

    return find_spec("torch") is not None


def _score_graph_rows_with_gae_checkpoint(
    *,
    graphs: list[InteractionGraph],
    packed_graphs: list[PackedGraphInput],
    checkpoint_path: str,
    reduction_method: PcapGraphScoreReduction,
) -> list[dict[str, object]]:
    """Score graphs with one saved GAE checkpoint and return raw graph rows."""

    import torch

    from traffic_graph.pipeline.checkpoint import load_checkpoint

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    checkpoint.model.eval()
    seed_reduction = _initial_graph_score_reduction(reduction_method)
    graph_rows: list[dict[str, object]] = []
    with torch.no_grad():
        for graph_index, (graph_sample, packed_graph) in enumerate(
            zip(graphs, packed_graphs, strict=True)
        ):
            output = checkpoint.model(packed_graph)
            node_scores = compute_node_anomaly_scores(
                packed_graph.node_features,
                output.reconstructed_node_features.detach().cpu().numpy(),
                discrete_mask=packed_graph.node_discrete_mask,
            )
            edge_scores = compute_edge_anomaly_scores(
                packed_graph.edge_features,
                (
                    output.reconstructed_edge_features.detach().cpu().numpy()
                    if output.reconstructed_edge_features is not None
                    else None
                ),
                discrete_mask=packed_graph.edge_discrete_mask,
            )
            flow_scores = edge_scores[np.asarray(packed_graph.edge_types, dtype=int) == 0]
            graph_rows.append(
                _build_graph_score_row_with_summary(
                    graph_index=graph_index,
                    graph_sample=graph_sample,
                    graph_score=_reduce_graph_score(
                        node_scores=node_scores,
                        edge_scores=edge_scores,
                        flow_scores=flow_scores,
                        reduction_method=seed_reduction,
                    ),
                    node_scores=node_scores,
                    edge_scores=edge_scores,
                    flow_scores=flow_scores,
                    reduction_method=reduction_method,
                )
            )
    return graph_rows


@dataclass(slots=True)
class PcapGraphSmokeConfig:
    """Configuration for the real-PCAP smoke validation path."""

    packet_limit: int | None = 5000
    idle_timeout_seconds: float = 60.0
    window_size: int = 60
    short_flow_thresholds: ShortFlowThresholds = field(
        default_factory=ShortFlowThresholds
    )
    use_association_edges: bool = True
    use_graph_structural_features: bool = True
    smoke_graph_limit: int = 16
    train_validation_ratio: float = 0.25
    graph_score_reduction: PcapGraphScoreReduction = "hybrid_max_rank_flow_node_max"
    epochs: int = 2
    batch_size: int = 2
    random_seed: int = 42
    threshold_percentile: float = 95.0
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    hidden_dim: int = 64
    latent_dim: int = 32
    num_layers: int = 2
    dropout: float = 0.1
    use_edge_features: bool = True
    reconstruct_edge_features: bool = True
    use_edge_categorical_embeddings: bool = True
    edge_categorical_embedding_dim: int = 8
    edge_categorical_bucket_size: int = 128
    checkpoint_dir: str = "artifacts/checkpoints"

    def to_pipeline_config(
        self,
        *,
        input_path: str,
        run_name: str,
        output_directory: str,
        graph_association_prefix: int = 24,
    ) -> PipelineConfig:
        """Build a pipeline config suitable for the existing GAE trainer."""

        return PipelineConfig(
            pipeline=PipelineRuntimeConfig(run_name=run_name, seed=self.random_seed),
            data=DataConfig(input_path=input_path, format="pcap"),
            preprocessing=PreprocessingConfig(
                window_size=self.window_size,
                short_flow_thresholds=self.short_flow_thresholds,
            ),
            graph=GraphConfig(
                time_window_seconds=self.window_size,
                directed=True,
                association_edges=AssociationEdgeConfig(
                    enable_same_src_ip=self.use_association_edges,
                    enable_same_dst_subnet=self.use_association_edges,
                    dst_subnet_prefix=graph_association_prefix,
                ),
            ),
            features=FeaturesConfig(
                normalization=FeatureNormalizationConfig(),
                use_graph_structural_features=self.use_graph_structural_features,
            ),
            model=ModelConfig(
                name="graph-autoencoder",
                device="cpu",
                hidden_dim=self.hidden_dim,
                latent_dim=self.latent_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
                use_edge_features=self.use_edge_features,
                reconstruct_edge_features=self.reconstruct_edge_features,
                use_edge_categorical_embeddings=self.use_edge_categorical_embeddings,
                edge_categorical_embedding_dim=self.edge_categorical_embedding_dim,
                edge_categorical_bucket_size=self.edge_categorical_bucket_size,
            ),
            training=TrainingConfig(
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                batch_size=self.batch_size,
                validation_split_ratio=self.train_validation_ratio,
                early_stopping_patience=max(1, self.epochs),
                checkpoint_dir=self.checkpoint_dir,
                shuffle=True,
                seed=self.random_seed,
                smoke_graph_limit=self.smoke_graph_limit,
            ),
            evaluation=EvaluationConfig(),
            alerting=AlertingConfig(),
            output=OutputConfig(directory=output_directory, save_intermediate=True),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the smoke configuration into a plain mapping."""

        return {
            "packet_limit": self.packet_limit,
            "idle_timeout_seconds": self.idle_timeout_seconds,
            "window_size": self.window_size,
            "short_flow_thresholds": {
                "packet_count_lt": self.short_flow_thresholds.packet_count_lt,
                "byte_count_lt": self.short_flow_thresholds.byte_count_lt,
                "duration_seconds_lt": self.short_flow_thresholds.duration_seconds_lt,
            },
            "use_association_edges": self.use_association_edges,
            "use_graph_structural_features": self.use_graph_structural_features,
            "smoke_graph_limit": self.smoke_graph_limit,
            "graph_score_reduction": self.graph_score_reduction,
            "train_validation_ratio": self.train_validation_ratio,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "random_seed": self.random_seed,
            "threshold_percentile": self.threshold_percentile,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "use_edge_features": self.use_edge_features,
            "reconstruct_edge_features": self.reconstruct_edge_features,
            "use_edge_categorical_embeddings": self.use_edge_categorical_embeddings,
            "edge_categorical_embedding_dim": self.edge_categorical_embedding_dim,
            "edge_categorical_bucket_size": self.edge_categorical_bucket_size,
            "checkpoint_dir": self.checkpoint_dir,
        }


@dataclass(slots=True)
class PcapGraphSmokeResult:
    """Structured output returned by the real-PCAP smoke experiment."""

    run_id: str
    timestamp: str
    source_path: str
    backend: str
    config: PcapGraphSmokeConfig
    parse_summary: dict[str, object]
    flow_dataset_summary: dict[str, object]
    window_statistics: list[dict[str, object]] = field(default_factory=list)
    graph_summaries: list[dict[str, int | float]] = field(default_factory=list)
    feature_summaries: list[dict[str, int]] = field(default_factory=list)
    train_graph_count: int = 0
    val_graph_count: int = 0
    checkpoint_paths: dict[str, str] = field(default_factory=dict)
    training_history: list[dict[str, float | int]] = field(default_factory=list)
    score_summary: dict[str, dict[str, float | int]] = field(default_factory=dict)
    alert_summary: dict[str, object] = field(default_factory=dict)
    export_result: RunBundleExportResult | None = None
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Render a concise text summary for CLI output."""

        lines = [
            "PCAP graph smoke experiment summary:",
            f"Run id: {self.run_id}",
            f"Timestamp: {self.timestamp}",
            f"Source: {self.source_path}",
            f"Backend: {self.backend}",
            (
                "PCAP summary: "
                f"total_packets={int(self.parse_summary.get('total_packets', 0))}, "
                f"parsed_packets={int(self.parse_summary.get('parsed_packets', 0))}, "
                f"skipped_packets={int(self.parse_summary.get('skipped_packets', 0))}, "
                f"total_flows={int(self.parse_summary.get('total_flows', 0))}"
            ),
            (
                "Flow dataset: "
                f"flows={int(self.flow_dataset_summary.get('flow_count', 0))}, "
                f"protocols={self.flow_dataset_summary.get('protocols', [])}, "
                f"avg_duration={float(self.flow_dataset_summary.get('average_duration_seconds', 0.0)):.6f}"
            ),
            (
                "Graph config: "
                f"window_size={self.config.window_size}, "
                f"use_association_edges={self.config.use_association_edges}, "
                f"use_graph_structural_features={self.config.use_graph_structural_features}"
            ),
            f"Graphs built: {len(self.graph_summaries)}",
            f"Train/val graphs: {self.train_graph_count}/{self.val_graph_count}",
        ]
        if self.feature_summaries:
            first = self.feature_summaries[0]
            lines.append(
                "First packed graph: "
                f"nodes={first.get('node_count', 0)}, "
                f"edges={first.get('edge_count', 0)}, "
                f"node_feature_dim={first.get('node_feature_dim', 0)}, "
                f"edge_feature_dim={first.get('edge_feature_dim', 0)}"
            )
        if self.training_history:
            lines.append("Training history:")
            for entry in self.training_history:
                lines.append(
                    "  - "
                    f"epoch={int(entry.get('epoch', 0))} "
                    f"train_loss={float(entry.get('train_loss', 0.0)):.6f} "
                    f"val_loss={float(entry.get('val_loss', 0.0)):.6f}"
                )
        if self.score_summary:
            lines.append("Score summaries:")
            for scope in ("graph", "node", "edge", "flow"):
                summary = self.score_summary.get(scope)
                if summary is None:
                    continue
                lines.append(
                    "  - "
                    f"{scope}: count={int(summary.get('count', 0))}, "
                    f"mean={float(summary.get('mean', 0.0)):.6f}, "
                    f"q95={float(summary.get('q95', 0.0)):.6f}"
                )
        if self.alert_summary:
            lines.append(
                "Alerts: "
                f"total={int(self.alert_summary.get('total_count', 0))}, "
                f"positive={int(self.alert_summary.get('positive_count', 0))}"
            )
        if self.export_result is not None:
            lines.append(f"Exported bundle: {self.export_result.run_directory}")
        if self.checkpoint_paths:
            lines.append("Checkpoint paths:")
            for name, path in self.checkpoint_paths.items():
                lines.append(f"  - {name}: {path}")
        if self.notes:
            lines.append("Notes:")
            lines.extend(f"  - {note}" for note in self.notes)
        return "\n".join(lines)


def _split_graphs(
    graphs: list[InteractionGraph],
    *,
    validation_ratio: float,
) -> tuple[list[InteractionGraph], list[InteractionGraph]]:
    """Split graphs into train and validation partitions deterministically."""

    if not graphs:
        raise ValueError("PCAP smoke runs require at least one graph.")
    if len(graphs) == 1:
        return graphs[:], []

    validation_count = int(round(len(graphs) * validation_ratio))
    if validation_count <= 0:
        validation_count = 1
    if validation_count >= len(graphs):
        validation_count = len(graphs) - 1
    return graphs[validation_count:], graphs[:validation_count]


def _score_graph_with_fallback(
    packed_graph: object,
    *,
    reduction_method: PcapGraphScoreReduction,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Score a packed graph with a deterministic feature-norm fallback."""

    node_features = np.asarray(getattr(packed_graph, "node_features"), dtype=float)
    edge_features = np.asarray(getattr(packed_graph, "edge_features"), dtype=float)
    node_discrete_mask = np.asarray(
        getattr(packed_graph, "node_discrete_mask", (False,) * node_features.shape[1]),
        dtype=bool,
    )
    edge_discrete_mask = np.asarray(
        getattr(packed_graph, "edge_discrete_mask", (False,) * edge_features.shape[1]),
        dtype=bool,
    )
    if node_features.size > 0 and node_discrete_mask.size == node_features.shape[1]:
        node_features = node_features.copy()
        node_features[:, node_discrete_mask] = 0.0
    if edge_features.size > 0 and edge_discrete_mask.size == edge_features.shape[1]:
        edge_features = edge_features.copy()
        edge_features[:, edge_discrete_mask] = 0.0
    node_scores = (
        np.linalg.norm(node_features, axis=1)
        if node_features.size > 0
        else np.zeros((0,), dtype=float)
    )
    edge_scores = (
        np.linalg.norm(edge_features, axis=1)
        if edge_features.size > 0
        else np.zeros((0,), dtype=float)
    )
    flow_scores = np.asarray(
        [edge_scores[index] for index, edge_type in enumerate(getattr(packed_graph, "edge_types")) if int(edge_type) == 0],
        dtype=float,
    )
    graph_score = _reduce_graph_score(
        node_scores=node_scores,
        edge_scores=edge_scores,
        flow_scores=flow_scores,
        reduction_method=reduction_method,
    )
    return graph_score, node_scores, edge_scores, flow_scores


def _train_and_score_with_gae(
    *,
    train_graphs: list[InteractionGraph],
    val_graphs: list[InteractionGraph],
    all_graphs: list[InteractionGraph],
    all_packed_graphs: list[PackedGraphInput],
    config: PcapGraphSmokeConfig,
    pipeline_config: PipelineConfig,
) -> tuple[
    list[dict[str, float | int]],
    dict[str, str],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    float,
    dict[str, float | int],
    list[str],
]:
    """Train the graph autoencoder and score graphs using the best checkpoint."""

    import torch

    from traffic_graph.models import (
        GraphAutoEncoder,
        GraphAutoEncoderConfig,
        ReconstructionLossWeights,
    )
    from traffic_graph.pipeline.checkpoint import load_checkpoint
    from traffic_graph.pipeline.trainer import GraphAETrainer

    preprocessor = fit_feature_preprocessor(
        train_graphs,
        normalization_config=pipeline_config.features.normalization,
        include_graph_structural_features=(
            pipeline_config.features.use_graph_structural_features
        ),
    )
    train_packed = transform_graphs(
        train_graphs,
        preprocessor,
        include_graph_structural_features=(
            pipeline_config.features.use_graph_structural_features
        ),
    )
    val_packed = transform_graphs(
        val_graphs,
        preprocessor,
        include_graph_structural_features=(
            pipeline_config.features.use_graph_structural_features
        ),
    )
    if not train_packed:
        raise ValueError("At least one packed graph is required for smoke training.")

    model = GraphAutoEncoder(
        node_input_dim=train_packed[0].node_feature_dim,
        edge_input_dim=train_packed[0].edge_feature_dim,
        config=GraphAutoEncoderConfig(
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_edge_features=config.use_edge_features,
            reconstruct_edge_features=config.reconstruct_edge_features,
            use_edge_categorical_embeddings=config.use_edge_categorical_embeddings,
            edge_categorical_embedding_dim=config.edge_categorical_embedding_dim,
            edge_categorical_bucket_size=config.edge_categorical_bucket_size,
        ),
        loss_weights=ReconstructionLossWeights(),
    )
    trainer = GraphAETrainer(
        model=model,
        config=pipeline_config,
        feature_preprocessor=preprocessor,
        device=pipeline_config.model.device,
    )
    fit_result = trainer.fit(train_packed, val_packed, smoke_run=True)
    checkpoint_paths = {
        "latest_checkpoint": fit_result.latest_checkpoint_path,
        "best_checkpoint": fit_result.best_checkpoint_path,
    }
    checkpoint = load_checkpoint(
        fit_result.best_checkpoint_path or fit_result.latest_checkpoint_path,
        map_location="cpu",
    )
    checkpoint.model.eval()
    seed_reduction = _initial_graph_score_reduction(config.graph_score_reduction)

    score_rows_by_graph: list[dict[str, object]] = []
    node_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    flow_rows: list[dict[str, object]] = []
    graph_scores: list[float] = []
    node_scores_flat: list[float] = []
    edge_scores_flat: list[float] = []
    flow_scores_flat: list[float] = []
    threshold = 0.0

    with torch.no_grad():
        for graph_index, (graph_sample, packed_graph) in enumerate(
            zip(all_graphs, all_packed_graphs, strict=True)
        ):
            output = checkpoint.model(packed_graph)
            node_scores = compute_node_anomaly_scores(
                packed_graph.node_features,
                output.reconstructed_node_features.detach().cpu().numpy(),
                discrete_mask=packed_graph.node_discrete_mask,
            )
            edge_scores = compute_edge_anomaly_scores(
                packed_graph.edge_features,
                (
                    output.reconstructed_edge_features.detach().cpu().numpy()
                    if output.reconstructed_edge_features is not None
                    else None
                ),
                discrete_mask=packed_graph.edge_discrete_mask,
            )
            flow_scores = edge_scores[
                np.asarray(packed_graph.edge_types, dtype=int) == 0
            ]
            graph_score_value = _reduce_graph_score(
                node_scores=node_scores,
                edge_scores=edge_scores,
                flow_scores=flow_scores,
                reduction_method=seed_reduction,
            )
            graph_scores.append(graph_score_value)
            node_scores_flat.extend(float(value) for value in node_scores.tolist())
            edge_scores_flat.extend(float(value) for value in edge_scores.tolist())
            flow_scores_flat.extend(float(value) for value in flow_scores.tolist())
            score_rows_by_graph.append(
                _build_graph_score_row_with_summary(
                    graph_index=graph_index,
                    graph_sample=graph_sample,
                    graph_score=graph_score_value,
                    node_scores=node_scores,
                    edge_scores=edge_scores,
                    flow_scores=flow_scores,
                    reduction_method=config.graph_score_reduction,
                )
            )
            node_rows.extend(build_node_score_rows(graph_index, graph_sample, node_scores))
            edge_rows.extend(build_edge_score_rows(graph_index, graph_sample, edge_scores))
            flow_rows.extend(build_flow_score_rows(graph_index, graph_sample, edge_scores))

        if train_packed:
            train_scores = []
            for graph_sample in train_packed:
                output = checkpoint.model(graph_sample)
                node_scores = compute_node_anomaly_scores(
                    graph_sample.node_features,
                    output.reconstructed_node_features.detach().cpu().numpy(),
                    discrete_mask=graph_sample.node_discrete_mask,
                )
                edge_scores = compute_edge_anomaly_scores(
                    graph_sample.edge_features,
                    (
                        output.reconstructed_edge_features.detach().cpu().numpy()
                        if output.reconstructed_edge_features is not None
                        else None
                    ),
                    discrete_mask=graph_sample.edge_discrete_mask,
                )
                flow_scores = edge_scores[
                    np.asarray(graph_sample.edge_types, dtype=int) == 0
                ]
                train_scores.append(
                    _reduce_graph_score(
                        node_scores=node_scores,
                        edge_scores=edge_scores,
                        flow_scores=flow_scores,
                        reduction_method=seed_reduction,
                    )
                )
        else:
            train_scores = []
    threshold = (
        float(np.percentile(np.asarray(train_scores, dtype=float), config.threshold_percentile))
        if train_scores
        else 0.0
    )

    return (
        [entry.to_dict() for entry in fit_result.history],
        checkpoint_paths,
        score_rows_by_graph,
        node_rows,
        edge_rows,
        flow_rows,
        threshold,
        _quantile_summary(train_scores),
        [
            "PyTorch available; used the graph autoencoder smoke path.",
            f"Best checkpoint: {fit_result.best_checkpoint_path or fit_result.latest_checkpoint_path}",
            f"Validation graphs: {len(val_packed)}",
            f"Train graphs: {len(train_packed)}",
            f"Graph score reduction: {config.graph_score_reduction}",
        ],
    )


def _score_with_fallback(
    *,
    graph_samples: list[InteractionGraph],
    packed_graphs: list[PackedGraphInput],
    threshold_percentile: float,
    graph_score_reduction: PcapGraphScoreReduction = "hybrid_max_rank_flow_node_max",
    threshold_reference_packed_graphs: Sequence[PackedGraphInput] | None = None,
) -> tuple[
    list[dict[str, float | int]],
    dict[str, str],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    float,
    dict[str, float | int],
    list[str],
]:
    """Score packed graphs deterministically when PyTorch is unavailable."""

    graph_rows: list[dict[str, object]] = []
    node_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    flow_rows: list[dict[str, object]] = []
    graph_scores: list[float] = []
    node_scores_flat: list[float] = []
    edge_scores_flat: list[float] = []
    flow_scores_flat: list[float] = []
    reference_graph_scores: list[float] = []
    seed_reduction = _initial_graph_score_reduction(graph_score_reduction)

    for graph_index, (graph_sample, packed_graph) in enumerate(
        zip(graph_samples, packed_graphs, strict=True)
    ):
        graph_score, node_scores, edge_scores, flow_scores = _score_graph_with_fallback(
            packed_graph,
            reduction_method=seed_reduction,
        )
        graph_scores.append(graph_score)
        node_scores_flat.extend(float(value) for value in node_scores.tolist())
        edge_scores_flat.extend(float(value) for value in edge_scores.tolist())
        flow_scores_flat.extend(float(value) for value in flow_scores.tolist())
        graph_rows.append(
            _build_graph_score_row_with_summary(
                graph_index=graph_index,
                graph_sample=graph_sample,
                graph_score=graph_score,
                node_scores=node_scores,
                edge_scores=edge_scores,
                flow_scores=flow_scores,
                reduction_method=graph_score_reduction,
            )
        )
        node_rows.extend(
            build_node_score_rows(graph_index, graph_sample, node_scores)
        )
        edge_rows.extend(
            build_edge_score_rows(graph_index, graph_sample, edge_scores)
        )
        flow_rows.extend(
            build_flow_score_rows(graph_index, graph_sample, edge_scores)
        )

    threshold_reference = (
        list(threshold_reference_packed_graphs)
        if threshold_reference_packed_graphs is not None
        else list(packed_graphs)
    )
    for packed_graph in threshold_reference:
        reference_graph_score, _, _, _ = _score_graph_with_fallback(
            packed_graph,
            reduction_method=seed_reduction,
        )
        reference_graph_scores.append(float(reference_graph_score))

    threshold = (
        float(
            np.percentile(
                np.asarray(reference_graph_scores, dtype=float),
                threshold_percentile,
            )
        )
        if reference_graph_scores
        else 0.0
    )
    return (
        [],
        {},
        graph_rows,
        node_rows,
        edge_rows,
        flow_rows,
        threshold,
        _quantile_summary(reference_graph_scores),
        [
            "PyTorch is unavailable; used the deterministic feature-norm smoke fallback.",
            f"Graph score reduction: {graph_score_reduction}",
        ],
    )


def run_pcap_graph_smoke_experiment(
    source_path: str | Path,
    export_dir: str | Path,
    *,
    run_name: str | None = None,
    config: PcapGraphSmokeConfig | None = None,
) -> PcapGraphSmokeResult:
    """Run the real-PCAP graph smoke experiment end to end."""

    smoke_config = config or PcapGraphSmokeConfig()
    pcap_path = Path(source_path)
    if not pcap_path.exists():
        raise FileNotFoundError(pcap_path)

    load_result = load_pcap_flow_dataset(
        pcap_path,
        max_packets=smoke_config.packet_limit,
        idle_timeout_seconds=smoke_config.idle_timeout_seconds,
    )
    batches = preprocess_flow_dataset(
        load_result.dataset,
        window_size=smoke_config.window_size,
        rules=smoke_config.short_flow_thresholds,
    )
    window_statistics = [batch.stats for batch in batches]
    graph_builder = FlowInteractionGraphBuilder(
        GraphConfig(
            time_window_seconds=smoke_config.window_size,
            directed=True,
            association_edges=AssociationEdgeConfig(
                enable_same_src_ip=smoke_config.use_association_edges,
                enable_same_dst_subnet=smoke_config.use_association_edges,
                dst_subnet_prefix=24,
            ),
        )
    )
    graphs = graph_builder.build_many(batches)
    if smoke_config.smoke_graph_limit > 0:
        graphs = graphs[: smoke_config.smoke_graph_limit]
    if not graphs:
        raise ValueError("The pcap smoke experiment did not yield any graphs.")

    graph_summaries = [summarize_graph(graph) for graph in graphs]
    train_graphs, val_graphs = _split_graphs(
        graphs,
        validation_ratio=smoke_config.train_validation_ratio,
    )
    feature_preprocessor = fit_feature_preprocessor(
        train_graphs,
        normalization_config=FeatureNormalizationConfig(),
        include_graph_structural_features=smoke_config.use_graph_structural_features,
    )
    train_packed_graphs = transform_graphs(
        train_graphs,
        feature_preprocessor,
        include_graph_structural_features=smoke_config.use_graph_structural_features,
    )
    packed_graphs = transform_graphs(
        graphs,
        feature_preprocessor,
        include_graph_structural_features=smoke_config.use_graph_structural_features,
    )
    feature_summaries = [
        summarize_packed_graph_input(packed_graph) for packed_graph in packed_graphs
    ]

    timestamp = _timestamp_token()
    run_id = _slugify_token(run_name or f"{pcap_path.stem}-pcap-smoke")
    export_base_dir = Path(export_dir)
    run_directory = export_base_dir / run_id / timestamp
    checkpoint_dir = run_directory / "checkpoints"
    smoke_config = PcapGraphSmokeConfig(
        packet_limit=smoke_config.packet_limit,
        idle_timeout_seconds=smoke_config.idle_timeout_seconds,
        window_size=smoke_config.window_size,
        short_flow_thresholds=smoke_config.short_flow_thresholds,
        use_association_edges=smoke_config.use_association_edges,
        use_graph_structural_features=smoke_config.use_graph_structural_features,
        smoke_graph_limit=smoke_config.smoke_graph_limit,
        train_validation_ratio=smoke_config.train_validation_ratio,
        graph_score_reduction=smoke_config.graph_score_reduction,
        epochs=smoke_config.epochs,
        batch_size=smoke_config.batch_size,
        random_seed=smoke_config.random_seed,
        threshold_percentile=smoke_config.threshold_percentile,
        learning_rate=smoke_config.learning_rate,
        weight_decay=smoke_config.weight_decay,
        hidden_dim=smoke_config.hidden_dim,
        latent_dim=smoke_config.latent_dim,
        num_layers=smoke_config.num_layers,
        dropout=smoke_config.dropout,
        use_edge_features=smoke_config.use_edge_features,
        reconstruct_edge_features=smoke_config.reconstruct_edge_features,
        checkpoint_dir=checkpoint_dir.as_posix(),
    )
    pipeline_config = smoke_config.to_pipeline_config(
        input_path=pcap_path.as_posix(),
        run_name=run_id,
        output_directory=run_directory.as_posix(),
    )

    notes: list[str] = []
    if _has_torch():
        (
            training_history,
            checkpoint_paths,
            graph_rows,
            node_rows,
            edge_rows,
            flow_rows,
            threshold,
            train_score_summary,
            backend_notes,
        ) = _train_and_score_with_gae(
            train_graphs=train_graphs,
            val_graphs=val_graphs,
            all_graphs=graphs,
            all_packed_graphs=packed_graphs,
            config=smoke_config,
            pipeline_config=pipeline_config,
        )
        backend = "gae"
        notes.extend(backend_notes)
    else:
        (
            training_history,
            checkpoint_paths,
            graph_rows,
            node_rows,
            edge_rows,
            flow_rows,
            threshold,
            train_score_summary,
            backend_notes,
        ) = _score_with_fallback(
            graph_samples=graphs,
            packed_graphs=packed_graphs,
            threshold_percentile=smoke_config.threshold_percentile,
            graph_score_reduction=smoke_config.graph_score_reduction,
        )
        backend = "fallback"
        notes.extend(backend_notes)

    if train_graphs:
        if backend == "gae":
            checkpoint_path = (
                checkpoint_paths.get("best_checkpoint")
                or checkpoint_paths.get("latest_checkpoint")
            )
            if not checkpoint_path:
                raise ValueError("The GAE smoke path did not return a usable checkpoint path.")
            train_graph_rows_raw = _score_graph_rows_with_gae_checkpoint(
                graphs=list(train_graphs),
                packed_graphs=list(train_packed_graphs),
                checkpoint_path=checkpoint_path,
                reduction_method=smoke_config.graph_score_reduction,
            )
        else:
            (
                _ignored_history,
                _ignored_checkpoints,
                train_graph_rows_raw,
                _ignored_node_rows,
                _ignored_edge_rows,
                _ignored_flow_rows,
                _ignored_threshold,
                _ignored_train_summary,
                _ignored_notes,
            ) = _score_with_fallback(
                graph_samples=list(train_graphs),
                packed_graphs=list(train_packed_graphs),
                threshold_percentile=smoke_config.threshold_percentile,
                graph_score_reduction=smoke_config.graph_score_reduction,
                threshold_reference_packed_graphs=train_packed_graphs,
            )
    else:
        train_graph_rows_raw = []

    train_graph_rows = _apply_graph_score_reduction_to_rows(
        train_graph_rows_raw,
        reduction_method=smoke_config.graph_score_reduction,
        reference_rows=train_graph_rows_raw,
    )
    graph_rows = _apply_graph_score_reduction_to_rows(
        graph_rows,
        reduction_method=smoke_config.graph_score_reduction,
        reference_rows=train_graph_rows_raw,
    )
    threshold, train_score_summary = _graph_score_threshold_from_rows(
        train_graph_rows,
        threshold_percentile=smoke_config.threshold_percentile,
    )

    score_tables = {
        "graph_scores": graph_rows,
        "node_scores": node_rows,
        "edge_scores": edge_rows,
        "flow_scores": flow_rows,
    }
    alert_records = build_alert_records(
        score_tables,
        AlertingConfig(anomaly_threshold=threshold),
    )
    alert_summary = summarize_alerts(alert_records)
    score_summary = {
        "graph": _quantile_summary([_score_from_row(row) for row in graph_rows]),
        "node": _quantile_summary([_score_from_row(row) for row in node_rows]),
        "edge": _quantile_summary([_score_from_row(row) for row in edge_rows]),
        "flow": _quantile_summary([_score_from_row(row) for row in flow_rows]),
    }
    flow_dataset_summary_dict = {
        "flow_count": load_result.summary.flow_dataset_summary.flow_count,
        "protocols": list(load_result.summary.flow_dataset_summary.protocols),
        "earliest_start": (
            load_result.summary.flow_dataset_summary.earliest_start.isoformat()
            if load_result.summary.flow_dataset_summary.earliest_start is not None
            else None
        ),
        "latest_end": (
            load_result.summary.flow_dataset_summary.latest_end.isoformat()
            if load_result.summary.flow_dataset_summary.latest_end is not None
            else None
        ),
        "average_duration_seconds": load_result.summary.flow_dataset_summary.average_duration_seconds,
    }
    metrics_summary = {
        "pcap_smoke": {
            "backend": backend,
            "source_path": pcap_path.as_posix(),
            "run_id": run_id,
            "packet_limit": smoke_config.packet_limit,
            "idle_timeout_seconds": smoke_config.idle_timeout_seconds,
            "window_size": smoke_config.window_size,
            "use_association_edges": smoke_config.use_association_edges,
            "use_graph_structural_features": smoke_config.use_graph_structural_features,
            "graph_score_reduction": smoke_config.graph_score_reduction,
            "total_packets": load_result.summary.total_packets,
            "parsed_packets": load_result.summary.parsed_packets,
            "skipped_packets": load_result.summary.skipped_packets,
            "total_flows": load_result.summary.total_flows,
            "flow_dataset_summary": flow_dataset_summary_dict,
            "graph_count": len(graphs),
            "train_graph_count": len(train_graphs),
            "val_graph_count": len(val_graphs),
            "threshold_percentile": smoke_config.threshold_percentile,
            "anomaly_threshold": threshold,
            "train_graph_score_summary": train_score_summary,
            "graph_score_summary": score_summary["graph"],
            "node_score_summary": score_summary["node"],
            "edge_score_summary": score_summary["edge"],
            "flow_score_summary": score_summary["flow"],
            "training_history": training_history,
            "notes": notes,
        }
    }

    export_result = export_run_bundle(
        score_tables,
        alert_records,
        metrics_summary,
        export_dir,
        run_id=run_id,
        split="smoke",
        timestamp=timestamp,
        anomaly_threshold=threshold,
        score_formats=("jsonl", "csv"),
        alert_formats=("jsonl", "csv"),
        metrics_formats=("json", "jsonl", "csv"),
    )

    smoke_summary_payload = {
        "run_id": run_id,
        "timestamp": timestamp,
        "source_path": pcap_path.as_posix(),
        "backend": backend,
        "config": smoke_config.to_dict(),
        "parse_summary": load_result.summary.to_dict(),
        "window_statistics": [asdict(stats) for stats in window_statistics],
        "graph_summaries": graph_summaries,
        "feature_summaries": feature_summaries,
        "train_graph_count": len(train_graphs),
        "val_graph_count": len(val_graphs),
        "score_threshold": threshold,
        "score_summary": score_summary,
        "alert_summary": alert_summary,
        "training_history": training_history,
        "notes": notes,
    }
    config_snapshot_path = run_directory / "pcap_config.json"
    smoke_summary_path = run_directory / "pcap_smoke_summary.json"
    _write_json(config_snapshot_path, smoke_config.to_dict())
    _write_json(smoke_summary_path, smoke_summary_payload)

    manifest_path = Path(export_result.manifest_path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest_payload = json.load(handle)
    if isinstance(manifest_payload, dict):
        artifact_paths = manifest_payload.setdefault("artifact_paths", {})
        if isinstance(artifact_paths, dict):
            artifact_paths["pcap_config_json"] = config_snapshot_path.as_posix()
            artifact_paths["pcap_smoke_summary_json"] = smoke_summary_path.as_posix()
        notes_list = manifest_payload.setdefault("notes", [])
        if isinstance(notes_list, list):
            for note in notes:
                if note not in notes_list:
                    notes_list.append(note)
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest_payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        export_result.artifact_paths["pcap_config_json"] = config_snapshot_path.as_posix()
        export_result.artifact_paths["pcap_smoke_summary_json"] = smoke_summary_path.as_posix()

    return PcapGraphSmokeResult(
        run_id=run_id,
        timestamp=timestamp,
        source_path=pcap_path.as_posix(),
        backend=backend,
        config=smoke_config,
        parse_summary=load_result.summary.to_dict(),
        flow_dataset_summary=flow_dataset_summary_dict,
        window_statistics=[asdict(stats) for stats in window_statistics],
        graph_summaries=graph_summaries,
        feature_summaries=feature_summaries,
        train_graph_count=len(train_graphs),
        val_graph_count=len(val_graphs),
        checkpoint_paths=checkpoint_paths,
        training_history=training_history,
        score_summary=score_summary,
        alert_summary=alert_summary,
        export_result=export_result,
        notes=notes,
    )


def summarize_pcap_graph_smoke_result(result: PcapGraphSmokeResult) -> str:
    """Render a concise human-readable summary of the smoke experiment."""

    return result.render()


__all__ = [
    "PcapGraphSmokeConfig",
    "PcapGraphSmokeResult",
    "PcapGraphScoreReduction",
    "_apply_graph_score_reduction_to_rows",
    "_graph_score_threshold_from_rows",
    "_score_graph_rows_with_gae_checkpoint",
    "_score_with_fallback",
    "run_pcap_graph_smoke_experiment",
    "summarize_pcap_graph_smoke_result",
]
