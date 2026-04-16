#!/usr/bin/env python3
"""Run a mixed-PCAP CTU-13 binary benchmark with node- and edge-centric variants."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_graph.config import AssociationEdgeConfig, FeatureNormalizationConfig, GraphConfig  # noqa: E402
from traffic_graph.data import (  # noqa: E402
    FlowDataset,
    LogicalFlowBatch,
    LogicalFlowWindowStats,
    ShortFlowThresholds,
    load_pcap_flow_dataset,
    preprocess_flow_dataset,
)
from traffic_graph.datasets import (  # noqa: E402
    align_flow_dataset_to_ctu13_labels,
    load_ctu13_manifest,
    parse_ctu13_label_file,
    write_alignment_summary_csv,
    write_alignment_summary_markdown,
)
from traffic_graph.features import fit_feature_preprocessor, transform_graphs  # noqa: E402
from traffic_graph.graph import FlowInteractionGraphBuilder  # noqa: E402
from traffic_graph.graph.nx_compat import HAS_NETWORKX, nx  # noqa: E402
from traffic_graph.models import GraphAutoEncoder, GraphTensorBatch, ReconstructionLossWeights, reconstruction_loss  # noqa: E402
from traffic_graph.models.model_types import GraphAutoEncoderConfig  # noqa: E402
from traffic_graph.pipeline.scoring import compute_edge_anomaly_scores, compute_node_anomaly_scores  # noqa: E402


MANIFEST_PATH = REPO_ROOT / "data" / "ctu13" / "ctu13_manifest.json"
ALIGNMENT_CSV = REPO_ROOT / "results" / "ctu13_flow_label_alignment_summary.csv"
ALIGNMENT_MD = REPO_ROOT / "results" / "ctu13_flow_label_alignment_summary.md"
BENCHMARK_CSV = REPO_ROOT / "results" / "ctu13_binary_benchmark.csv"
BENCHMARK_MD = REPO_ROOT / "results" / "ctu13_binary_benchmark.md"

WINDOW_SIZE_SECONDS = 15
PREFIX_PACKET_COUNT = 16
TRAIN_BENIGN_FRACTION = 0.7
MAX_PACKETS_PER_SCENARIO = 1_300_000
TRAIN_EPOCHS = 1

EDGE_TEMPORAL_FIELD_NAMES: tuple[str, ...] = (
    "coarse_ack_delay_mean",
    "coarse_ack_delay_p75",
    "ack_delay_large_gap_ratio",
    "seq_ack_match_ratio",
    "unmatched_seq_ratio",
    "unmatched_ack_ratio",
    "retry_burst_count",
    "retry_burst_max_len",
    "retry_like_dense_ratio",
    "small_pkt_burst_count",
    "small_pkt_burst_ratio",
    "flow_internal_emb_0",
    "flow_internal_emb_1",
    "flow_internal_emb_2",
    "flow_internal_emb_3",
    "flow_internal_emb_4",
    "flow_internal_emb_5",
    "flow_internal_emb_6",
    "flow_internal_emb_7",
    "flow_internal_emb_8",
    "flow_internal_emb_9",
    "flow_internal_emb_10",
    "flow_internal_emb_11",
    "flow_internal_emb_12",
    "flow_internal_emb_13",
    "flow_internal_emb_14",
    "flow_internal_emb_15",
)

FLOW_EMBEDDING_FIELDS = tuple(field for field in EDGE_TEMPORAL_FIELD_NAMES if field.startswith("flow_internal_emb_"))
RELATION_FIELDS: tuple[str, ...] = (
    "pkt_count",
    "byte_count",
    "duration",
    "flow_count",
    "retry_like_count",
    "retry_like_ratio",
    "seq_ack_match_ratio",
    "unmatched_seq_ratio",
    "unmatched_ack_ratio",
    "retry_burst_count",
    "retry_burst_max_len",
    "retry_like_dense_ratio",
    "small_pkt_burst_count",
    "small_pkt_burst_ratio",
    "flow_internal_packet_count",
    "flow_internal_sequential_edge_count",
    "flow_internal_window_edge_count",
    "flow_internal_ack_edge_count",
    "flow_internal_opposite_direction_edge_count",
)


@dataclass(slots=True)
class ScenarioWindowSample:
    scenario_id: str
    window_index: int
    label: str
    graph: object


@dataclass(slots=True)
class PreparedScenario:
    scenario_id: str
    train_graphs: list[object]
    test_samples: list[ScenarioWindowSample]
    unknown_samples: list[ScenarioWindowSample]


def _graph_window_label(flow_labels: Iterable[str]) -> str:
    labels = set(flow_labels)
    if "malicious" in labels:
        return "malicious"
    if "unknown" in labels or not labels:
        return "unknown"
    if labels == {"benign"}:
        return "benign"
    return "unknown"


def _logical_flow_binary_label(logical_flow, flow_label_by_id: dict[str, str]) -> str:
    labels = {
        flow_label_by_id.get(source_flow_id, "unknown")
        for source_flow_id in logical_flow.source_flow_ids
    }
    if "malicious" in labels:
        return "malicious"
    if labels == {"benign"}:
        return "benign"
    return "unknown"


def _subset_logical_flow_batch(
    batch: LogicalFlowBatch,
    logical_flows,
) -> LogicalFlowBatch:
    subset = tuple(logical_flows)
    short_flow_count = sum(flow.is_aggregated_short_flow for flow in subset)
    long_flow_count = len(subset) - short_flow_count
    stats = LogicalFlowWindowStats(
        index=batch.index,
        window_start=batch.window_start,
        window_end=batch.window_end,
        raw_flow_count=batch.stats.raw_flow_count,
        short_flow_count=short_flow_count,
        long_flow_count=long_flow_count,
        logical_flow_count=len(subset),
    )
    return LogicalFlowBatch(
        index=batch.index,
        window_start=batch.window_start,
        window_end=batch.window_end,
        logical_flows=subset,
        stats=stats,
    )


def _pack_with_labels(
    dataset: FlowDataset,
    *,
    scenario_id: str,
) -> tuple[list[ScenarioWindowSample], list[ScenarioWindowSample]]:
    window_batches = preprocess_flow_dataset(
        dataset,
        window_size=WINDOW_SIZE_SECONDS,
        rules=ShortFlowThresholds(packet_count_lt=5, byte_count_lt=1024),
    )
    graph_builder = FlowInteractionGraphBuilder(
        GraphConfig(
            time_window_seconds=WINDOW_SIZE_SECONDS,
            directed=True,
            association_edges=AssociationEdgeConfig(
                enable_same_src_ip=True,
                enable_same_dst_subnet=True,
                enable_same_dst_ip=True,
                enable_same_prefix_signature=True,
                enable_prefix_similarity=False,
                prefix_similarity_threshold=0.97,
                prefix_similarity_top_k=1,
            ),
        )
    )
    flow_label_by_id = {
        record.flow_id: str(record.metadata.get("ctu13_binary_label", "unknown"))
        for record in dataset.records
    }

    primary_samples: list[ScenarioWindowSample] = []
    unknown_samples: list[ScenarioWindowSample] = []
    for batch in window_batches:
        labeled_flows = [
            (logical_flow, _logical_flow_binary_label(logical_flow, flow_label_by_id))
            for logical_flow in batch.logical_flows
        ]
        primary_flows = [
            logical_flow
            for logical_flow, label in labeled_flows
            if label in {"benign", "malicious"}
        ]
        unknown_flows = [
            logical_flow
            for logical_flow, label in labeled_flows
            if label == "unknown"
        ]
        primary_labels = [
            label
            for _logical_flow, label in labeled_flows
            if label in {"benign", "malicious"}
        ]

        if primary_flows:
            primary_batch = _subset_logical_flow_batch(batch, primary_flows)
            graph = graph_builder.build(primary_batch)
            label = "malicious" if "malicious" in primary_labels else "benign"
            primary_samples.append(
                ScenarioWindowSample(
                    scenario_id=scenario_id,
                    window_index=graph.window_index,
                    label=label,
                    graph=graph,
                )
            )

        if unknown_flows:
            unknown_batch = _subset_logical_flow_batch(batch, unknown_flows)
            unknown_graph = graph_builder.build(unknown_batch)
            unknown_samples.append(
                ScenarioWindowSample(
                    scenario_id=scenario_id,
                    window_index=unknown_graph.window_index,
                    label="unknown",
                    graph=unknown_graph,
                )
            )
    return primary_samples, unknown_samples


def _build_labeled_dataset(entry) -> tuple[FlowDataset, object]:
    load_result = load_pcap_flow_dataset(
        entry.pcap_path,
        max_packets=MAX_PACKETS_PER_SCENARIO,
        idle_timeout_seconds=60.0,
        prefix_packet_count=PREFIX_PACKET_COUNT,
    )
    label_flows = parse_ctu13_label_file(
        entry.label_file_path,
        scenario_id=entry.scenario_id,
    )
    aligned_rows, summary = align_flow_dataset_to_ctu13_labels(
        load_result.dataset,
        label_flows,
        scenario_id=entry.scenario_id,
        time_tolerance_seconds=5.0,
    )
    aligned_by_flow_id = {row.flow_id: row for row in aligned_rows}
    labeled_mappings = []
    for record in load_result.dataset.records:
        mapping = record.to_mapping()
        metadata = dict(mapping.get("metadata", {}))
        aligned = aligned_by_flow_id.get(record.flow_id)
        if aligned is None:
            metadata["ctu13_binary_label"] = "unknown"
            metadata["ctu13_label_text"] = ""
            metadata["ctu13_alignment_status"] = "unaligned"
        else:
            metadata["ctu13_binary_label"] = aligned.aligned_label
            metadata["ctu13_label_text"] = aligned.label_text
            metadata["ctu13_alignment_status"] = aligned.alignment_status
        mapping["metadata"] = metadata
        labeled_mappings.append(mapping)
    return FlowDataset.from_mappings(labeled_mappings), summary
def _group_scenario_samples(scenario_id: str, primary_samples: list[ScenarioWindowSample], unknown_samples: list[ScenarioWindowSample]) -> PreparedScenario:
    benign_samples = [sample for sample in primary_samples if sample.label == "benign"]
    malicious_samples = [sample for sample in primary_samples if sample.label == "malicious"]
    cut = max(1, int(len(benign_samples) * TRAIN_BENIGN_FRACTION)) if benign_samples else 0
    test_samples = benign_samples[cut:] + malicious_samples
    return PreparedScenario(
        scenario_id=scenario_id,
        train_graphs=[sample.graph for sample in benign_samples[:cut]],
        test_samples=test_samples,
        unknown_samples=unknown_samples,
    )


def _train_model(
    train_graphs,
    *,
    variant: str,
):
    if not train_graphs:
        raise ValueError("Need at least one benign training graph for CTU-13 benchmark.")
    normalization_config = FeatureNormalizationConfig(method="robust")
    preprocessor = fit_feature_preprocessor(
        train_graphs,
        normalization_config,
        include_graph_structural_features=True,
    )
    packed_train_graphs = transform_graphs(
        train_graphs,
        preprocessor,
        include_graph_structural_features=True,
    )
    sample_graph = packed_train_graphs[0]
    if variant == "node_recon_baseline":
        config = GraphAutoEncoderConfig(
            hidden_dim=32,
            latent_dim=16,
            num_layers=2,
            dropout=0.1,
            use_edge_features=True,
            reconstruct_edge_features=True,
            use_temporal_edge_projector=False,
            use_edge_categorical_embeddings=True,
        )
        loss_weights = ReconstructionLossWeights(node_weight=1.0, edge_weight=0.5)
    else:
        config = GraphAutoEncoderConfig(
            hidden_dim=48,
            latent_dim=24,
            num_layers=2,
            dropout=0.1,
            use_edge_features=True,
            reconstruct_edge_features=True,
            use_temporal_edge_projector=True,
            temporal_edge_hidden_dim=48,
            temporal_edge_field_names=EDGE_TEMPORAL_FIELD_NAMES,
            use_edge_categorical_embeddings=True,
            edge_categorical_embedding_dim=12,
            edge_categorical_bucket_size=256,
        )
        loss_weights = ReconstructionLossWeights(node_weight=0.25, edge_weight=2.0)

    model = GraphAutoEncoder(
        node_input_dim=sample_graph.node_feature_dim,
        edge_input_dim=sample_graph.edge_feature_dim,
        config=config,
        loss_weights=loss_weights,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    model.train()
    for _epoch in range(TRAIN_EPOCHS):
        for start in range(0, len(packed_train_graphs), 8):
            batch_graphs = packed_train_graphs[start : start + 8]
            tensor_batch = GraphTensorBatch.from_packed_graphs(batch_graphs)
            output = model(tensor_batch)
            loss_output = reconstruction_loss(output, weights=model.loss_weights)
            optimizer.zero_grad(set_to_none=True)
            loss_output.total_loss.backward()
            optimizer.step()
    model.eval()
    return model, preprocessor, packed_train_graphs


def _continuous_field_indices(field_names: tuple[str, ...], selected_fields: tuple[str, ...]) -> list[int]:
    selected = set(selected_fields)
    return [index for index, field_name in enumerate(field_names) if field_name in selected]


def _subset_rowwise_mse(
    reference: np.ndarray,
    reconstruction: np.ndarray,
    indices: list[int],
) -> np.ndarray:
    if not indices:
        return np.zeros((reference.shape[0],), dtype=float)
    residual = reference[:, indices] - reconstruction[:, indices]
    return np.mean(residual * residual, axis=1)


def _node_baseline_graph_score(node_scores: np.ndarray) -> float:
    if node_scores.size == 0:
        return 0.0
    return float(np.mean(node_scores))


def _edge_temporal_binary_v2_graph_score(sample: ScenarioWindowSample, edge_scores: np.ndarray) -> float:
    if edge_scores.size == 0:
        return 0.0
    communication_indices = [
        index
        for index, edge in enumerate(sample.graph.edges)
        if edge.edge_type == "communication"
    ]
    if not communication_indices:
        return float(np.mean(edge_scores))
    communication_scores = np.asarray([edge_scores[index] for index in communication_indices], dtype=float)
    topk_size = max(1, int(np.ceil(len(communication_scores) * 0.2)))
    topk_mean = float(np.mean(np.sort(communication_scores)[-topk_size:]))

    server_neighborhood_scores: list[float] = []
    for node in sample.graph.nodes:
        if node.endpoint_type != "server":
            continue
        incident_scores = [
            edge_scores[index]
            for index, edge in enumerate(sample.graph.edges)
            if edge.edge_type == "communication"
            and (edge.source_node_id == node.node_id or edge.target_node_id == node.node_id)
        ]
        if incident_scores:
            server_neighborhood_scores.append(float(np.mean(sorted(incident_scores)[-topk_size:])))
    server_max = max(server_neighborhood_scores, default=0.0)

    graph_backend = sample.graph.graph
    components: list[set[str]] = []
    if HAS_NETWORKX:
        try:
            undirected = graph_backend.to_undirected()
            components = [set(component) for component in nx.connected_components(undirected)]
        except Exception:
            components = []
    component_scores: list[float] = []
    for component_nodes in components:
        component_edge_scores = [
            edge_scores[index]
            for index, edge in enumerate(sample.graph.edges)
            if edge.edge_type == "communication"
            and edge.source_node_id in component_nodes
            and edge.target_node_id in component_nodes
        ]
        if component_edge_scores:
            component_scores.append(float(np.mean(component_edge_scores)))
    component_max = max(component_scores, default=0.0)

    short_scores = [
        edge_scores[index]
        for index, edge in enumerate(sample.graph.edges)
        if edge.edge_type == "communication" and edge.flow_length_type == "short"
    ]
    long_scores = [
        edge_scores[index]
        for index, edge in enumerate(sample.graph.edges)
        if edge.edge_type == "communication" and edge.flow_length_type == "long"
    ]
    short_mean = float(np.mean(sorted(short_scores)[-topk_size:])) if short_scores else 0.0
    long_mean = float(np.mean(sorted(long_scores)[-topk_size:])) if long_scores else 0.0
    return float(
        0.45 * topk_mean
        + 0.2 * server_max
        + 0.15 * component_max
        + 0.1 * short_mean
        + 0.1 * long_mean
    )


def _score_samples(model, samples: list[ScenarioWindowSample], preprocessor, *, variant: str) -> list[float]:
    scores: list[float] = []
    for sample in samples:
        packed_graph = transform_graphs(
            [sample.graph],
            preprocessor,
            include_graph_structural_features=True,
        )[0]
        output = model(packed_graph)
        node_scores = compute_node_anomaly_scores(
            packed_graph.node_features,
            output.reconstructed_node_features.detach().cpu().numpy(),
            discrete_mask=packed_graph.node_discrete_mask,
        )
        edge_recon = output.reconstructed_edge_features.detach().cpu().numpy()
        all_edge_scores = compute_edge_anomaly_scores(
            packed_graph.edge_features,
            edge_recon,
            discrete_mask=packed_graph.edge_discrete_mask,
        )
        if variant == "node_recon_baseline":
            scores.append(_node_baseline_graph_score(node_scores))
            continue
        flow_embedding_indices = _continuous_field_indices(
            packed_graph.edge_feature_fields,
            FLOW_EMBEDDING_FIELDS,
        )
        relation_indices = _continuous_field_indices(
            packed_graph.edge_feature_fields,
            RELATION_FIELDS,
        )
        flow_embedding_scores = _subset_rowwise_mse(
            np.asarray(packed_graph.edge_features, dtype=float),
            np.asarray(edge_recon, dtype=float),
            flow_embedding_indices,
        )
        relation_scores = _subset_rowwise_mse(
            np.asarray(packed_graph.edge_features, dtype=float),
            np.asarray(edge_recon, dtype=float),
            relation_indices,
        )
        edge_scores = (
            0.55 * all_edge_scores
            + 0.30 * flow_embedding_scores
            + 0.15 * relation_scores
        )
        scores.append(_edge_temporal_binary_v2_graph_score(sample, edge_scores))
    return scores


def _binary_metrics(y_true: list[int], y_score: list[float], threshold: float) -> dict[str, float | None]:
    y_pred = [1 if score >= threshold else 0 for score in y_score]
    negative_total = sum(label == 0 for label in y_true)
    false_positive = sum(label == 0 and pred == 1 for label, pred in zip(y_true, y_pred, strict=True))
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(false_positive / negative_total) if negative_total else 0.0,
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, y_score)) if len(set(y_true)) > 1 else None,
    }


def _evaluate_variant(
    prepared_scenarios: list[PreparedScenario],
    *,
    variant: str,
    evaluation_mode: str,
) -> list[dict[str, object]]:
    if evaluation_mode == "merged":
        train_graphs = [graph for scenario in prepared_scenarios for graph in scenario.train_graphs]
        model, preprocessor, _packed_train_graphs = _train_model(train_graphs, variant=variant)
        merged_samples = [sample for scenario in prepared_scenarios for sample in scenario.test_samples]
        merged_unknown = [sample for scenario in prepared_scenarios for sample in scenario.unknown_samples]
        train_scores = _score_samples(
            model,
            [ScenarioWindowSample("merged", -1, "benign", graph) for graph in train_graphs],
            preprocessor,
            variant=variant,
        )
        threshold = float(np.percentile(np.asarray(train_scores, dtype=float), 95.0))
        test_scores = _score_samples(model, merged_samples, preprocessor, variant=variant)
        unknown_scores = _score_samples(model, merged_unknown, preprocessor, variant=variant) if merged_unknown else []
        y_true = [1 if sample.label == "malicious" else 0 for sample in merged_samples]
        metrics = _binary_metrics(y_true, test_scores, threshold)
        return [
            {
                "evaluation_mode": "merged",
                "scenario_id": "merged",
                "variant": variant,
                "train_benign_graphs": len(train_graphs),
                "test_benign_graphs": sum(sample.label == "benign" for sample in merged_samples),
                "test_malicious_graphs": sum(sample.label == "malicious" for sample in merged_samples),
                "test_unknown_graphs": len(merged_unknown),
                "threshold": threshold,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "fpr": metrics["fpr"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "background_hit_ratio": (
                    float(sum(score >= threshold for score in unknown_scores) / len(unknown_scores))
                    if unknown_scores
                    else 0.0
                ),
                "background_score_mean": (float(np.mean(unknown_scores)) if unknown_scores else 0.0),
                "background_score_median": (
                    float(np.median(unknown_scores)) if unknown_scores else 0.0
                ),
            }
        ]

    rows: list[dict[str, object]] = []
    for scenario in prepared_scenarios:
        if not scenario.train_graphs or not scenario.test_samples:
            continue
        model, preprocessor, _packed_train_graphs = _train_model(scenario.train_graphs, variant=variant)
        train_scores = _score_samples(
            model,
            [ScenarioWindowSample(scenario.scenario_id, -1, "benign", graph) for graph in scenario.train_graphs],
            preprocessor,
            variant=variant,
        )
        threshold = float(np.percentile(np.asarray(train_scores, dtype=float), 95.0))
        test_scores = _score_samples(model, scenario.test_samples, preprocessor, variant=variant)
        unknown_scores = _score_samples(model, scenario.unknown_samples, preprocessor, variant=variant) if scenario.unknown_samples else []
        y_true = [1 if sample.label == "malicious" else 0 for sample in scenario.test_samples]
        metrics = _binary_metrics(y_true, test_scores, threshold)
        rows.append(
            {
                "evaluation_mode": "scenario_wise",
                "scenario_id": scenario.scenario_id,
                "variant": variant,
                "train_benign_graphs": len(scenario.train_graphs),
                "test_benign_graphs": sum(sample.label == "benign" for sample in scenario.test_samples),
                "test_malicious_graphs": sum(sample.label == "malicious" for sample in scenario.test_samples),
                "test_unknown_graphs": len(scenario.unknown_samples),
                "threshold": threshold,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "fpr": metrics["fpr"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "background_hit_ratio": (
                    float(sum(score >= threshold for score in unknown_scores) / len(unknown_scores))
                    if unknown_scores
                    else 0.0
                ),
                "background_score_mean": (float(np.mean(unknown_scores)) if unknown_scores else 0.0),
                "background_score_median": (
                    float(np.median(unknown_scores)) if unknown_scores else 0.0
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# CTU-13 Binary Benchmark",
        "",
        "Primary metrics only use `benign + malicious` windows.",
        "Background-derived windows are reported only as secondary analysis.",
        "",
        "| evaluation_mode | scenario_id | variant | precision | recall | f1 | fpr | roc_auc | pr_auc | background_hit_ratio |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['evaluation_mode']} | {row['scenario_id']} | {row['variant']} | "
            f"{float(row['precision'] or 0.0):.4f} | {float(row['recall'] or 0.0):.4f} | "
            f"{float(row['f1'] or 0.0):.4f} | {float(row['fpr'] or 0.0):.4f} | "
            f"{'n/a' if row['roc_auc'] is None else f'{float(row['roc_auc']):.4f}'} | "
            f"{'n/a' if row['pr_auc'] is None else f'{float(row['pr_auc']):.4f}'} | "
            f"{float(row['background_hit_ratio'] or 0.0):.4f} |"
        )

    merged_rows = [row for row in rows if row["evaluation_mode"] == "merged"]
    if merged_rows:
        baseline = next((row for row in merged_rows if row["variant"] == "node_recon_baseline"), None)
        edge_v2 = next((row for row in merged_rows if row["variant"] == "edge_temporal_binary_v2"), None)
        lines.extend(["", "## Observations", ""])
        if baseline is not None and edge_v2 is not None:
            lines.append(
                f"- Merged F1: baseline `{float(baseline['f1'] or 0.0):.4f}` vs edge-centric `{float(edge_v2['f1'] or 0.0):.4f}`."
            )
            lines.append(
                f"- Merged recall: baseline `{float(baseline['recall'] or 0.0):.4f}` vs edge-centric `{float(edge_v2['recall'] or 0.0):.4f}`."
            )
            lines.append(
                f"- Merged FPR: baseline `{float(baseline['fpr'] or 0.0):.4f}` vs edge-centric `{float(edge_v2['fpr'] or 0.0):.4f}`."
            )
            lines.append(
                f"- Background secondary hit ratio: baseline `{float(baseline['background_hit_ratio'] or 0.0):.4f}` vs edge-centric `{float(edge_v2['background_hit_ratio'] or 0.0):.4f}`."
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    manifest_entries = [
        entry
        for entry in load_ctu13_manifest(MANIFEST_PATH)
        if entry.download_status != "failed" and entry.pcap_path and entry.label_file_path
    ]
    if not manifest_entries:
        raise SystemExit(
            "CTU-13 manifest is empty or missing successful scenarios. Run scripts/download_ctu13.py first."
        )

    alignment_summaries = []
    prepared_scenarios: list[PreparedScenario] = []
    for entry in manifest_entries:
        print(f"[ctu13] loading scenario {entry.scenario_id}", flush=True)
        dataset, alignment_summary = _build_labeled_dataset(entry)
        alignment_summaries.append(alignment_summary)
        primary_samples, unknown_samples = _pack_with_labels(dataset, scenario_id=entry.scenario_id)
        print(
            f"[ctu13] scenario {entry.scenario_id}: primary={len(primary_samples)} unknown={len(unknown_samples)}",
            flush=True,
        )
        prepared_scenarios.append(
            _group_scenario_samples(entry.scenario_id, primary_samples, unknown_samples)
        )

    write_alignment_summary_csv(alignment_summaries, ALIGNMENT_CSV)
    write_alignment_summary_markdown(alignment_summaries, ALIGNMENT_MD)

    rows: list[dict[str, object]] = []
    for variant in ("node_recon_baseline", "edge_temporal_binary_v2"):
        print(f"[ctu13] evaluating {variant}", flush=True)
        rows.extend(_evaluate_variant(prepared_scenarios, variant=variant, evaluation_mode="scenario_wise"))
        rows.extend(_evaluate_variant(prepared_scenarios, variant=variant, evaluation_mode="merged"))

    _write_csv(BENCHMARK_CSV, rows)
    _write_markdown(BENCHMARK_MD, rows)
    print(f"Wrote {BENCHMARK_CSV}")
    print(f"Wrote {BENCHMARK_MD}")


if __name__ == "__main__":
    main()
