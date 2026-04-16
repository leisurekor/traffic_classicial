"""Compare evaluation-only graph score reductions on one existing real-PCAP run.

This helper does not retrain the model and does not alter the main pipeline.
It reuses one exported graph run, reloads the saved checkpoint, reconstructs the
benign train-reference graphs, and compares several graph-level scoring
definitions under the same benign-train q95 threshold rule.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from traffic_graph.config import AssociationEdgeConfig, GraphConfig
from traffic_graph.data import ShortFlowThresholds, load_pcap_flow_dataset, preprocess_flow_dataset
from traffic_graph.features import transform_graphs
from traffic_graph.graph import FlowInteractionGraphBuilder, InteractionGraph
from traffic_graph.pipeline.checkpoint import load_checkpoint
from traffic_graph.pipeline.metrics import BinaryScoreMetrics, evaluate_scores
from traffic_graph.pipeline.scoring import (
    compute_edge_anomaly_scores,
    compute_node_anomaly_scores,
)


@dataclass(frozen=True, slots=True)
class GraphReductionRecord:
    """One graph sample plus its reduction-ready score components."""

    graph_id: str
    source_id: str
    source_name: str
    source_role: str
    split_assignment: str
    raw_label: str
    binary_label: int
    graph_score: float
    node_max: float
    node_topk_mean: float
    flow_p90: float
    aggregated_edge_count: float


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the graph reduction analysis helper."""

    parser = argparse.ArgumentParser(
        description=(
            "Read one representative graph run and compare several evaluation-only "
            "graph score reductions using the same benign-train q95 threshold rule."
        )
    )
    parser.add_argument(
        "--graph-run-dir",
        required=True,
        help="Directory of the representative graph binary-evaluation run.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where reduction comparison artifacts will be written.",
    )
    parser.add_argument(
        "--node-topk-fraction",
        type=float,
        default=0.05,
        help="Fraction of highest-scoring nodes to average for node_topk_mean.",
    )
    return parser


def _require_file(path: Path) -> Path:
    """Return a path when it exists and raise a clear error otherwise."""

    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _load_json(path: Path) -> dict[str, object]:
    """Load a JSON object from disk."""

    payload = json.loads(_require_file(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file with existence validation."""

    return pd.read_csv(_require_file(path))


def _load_score_table(path: Path) -> pd.DataFrame:
    """Load one score table and expand its metadata column when present."""

    frame = _load_csv(path).copy()
    metadata_series = frame.get("metadata", pd.Series(["{}"] * len(frame))).fillna("{}")
    metadata_rows: list[dict[str, object]] = []
    for raw in metadata_series:
        if isinstance(raw, dict):
            metadata_rows.append(dict(raw))
            continue
        try:
            metadata_rows.append(dict(json.loads(str(raw))))
        except json.JSONDecodeError:
            metadata_rows.append({})
    metadata_frame = pd.DataFrame(metadata_rows)
    if not metadata_frame.empty:
        metadata_frame.columns = [f"meta_{column}" for column in metadata_frame.columns]
        frame = pd.concat([frame.reset_index(drop=True), metadata_frame], axis=1)
    for column in ("anomaly_score", "threshold", "label"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "meta_binary_label" in frame.columns:
        frame["meta_binary_label"] = pd.to_numeric(
            frame["meta_binary_label"],
            errors="coerce",
        ).fillna(0).astype(int)
    return frame


def _graph_id(source_role: str, source_id: str, graph: InteractionGraph, entry_index: int) -> str:
    """Rebuild the stable graph identifier used by the PCAP experiment runner."""

    return f"{source_role}:{source_id}:{graph.window_index}:{entry_index}"


def _topk_mean(values: np.ndarray, fraction: float) -> float:
    """Return the mean of the highest-scoring node subset."""

    if values.size == 0:
        return 0.0
    count = max(1, int(math.ceil(values.size * fraction)))
    top_values = np.sort(values)[-count:]
    return float(top_values.mean())


def _p90(values: np.ndarray) -> float:
    """Return the p90 of a score vector with a stable empty fallback."""

    if values.size == 0:
        return 0.0
    return float(np.quantile(values, 0.90))


def _percentile_against_reference(
    values: pd.Series | np.ndarray | list[float],
    reference_values: pd.Series | np.ndarray | list[float],
) -> np.ndarray:
    """Map values onto an empirical percentile scale defined by reference values."""

    reference = np.asarray(reference_values, dtype=float)
    candidates = np.asarray(values, dtype=float)
    if reference.size == 0:
        return np.zeros_like(candidates, dtype=float)
    sorted_reference = np.sort(reference)
    ranks = np.searchsorted(sorted_reference, candidates, side="right")
    return ranks.astype(float) / float(sorted_reference.size)


def _rank_percentile(scores: pd.Series | np.ndarray) -> np.ndarray:
    """Convert anomaly scores into descending anomaly percentiles in [0, 1]."""

    series = pd.Series(np.asarray(scores, dtype=float))
    if series.empty:
        return np.zeros((0,), dtype=float)
    descending_rank = series.rank(method="average", ascending=False)
    if len(series) == 1:
        return np.asarray([1.0], dtype=float)
    return (1.0 - ((descending_rank - 1.0) / float(len(series) - 1.0))).to_numpy(
        dtype=float
    )


def _q95(values: pd.Series | np.ndarray | list[float]) -> float:
    """Return the q95 threshold for one score vector."""

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0
    return float(np.quantile(array, 0.95))


def _false_positive_rate(metrics: BinaryScoreMetrics) -> float:
    """Compute the false-positive rate from one metric bundle."""

    if metrics.negative_count <= 0:
        return 0.0
    return float(metrics.false_positive) / float(metrics.negative_count)


def _quantile_summary(values: pd.Series | np.ndarray | list[float]) -> dict[str, float | int]:
    """Build a compact score summary."""

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _train_source_rows(graph_summary: pd.DataFrame) -> pd.DataFrame:
    """Return unique benign source rows in stable source-id order."""

    benign_rows = graph_summary.loc[
        graph_summary["source_role"].astype(str).eq("benign"),
        ["source_id", "source_name", "source_path", "source_role"],
    ].drop_duplicates()
    return benign_rows.sort_values(by="source_id", ascending=True).reset_index(drop=True)


def _rebuild_train_reference_records(
    *,
    graph_run_dir: Path,
    node_topk_fraction: float,
) -> pd.DataFrame:
    """Rebuild train-reference graphs and score them with the saved checkpoint."""

    config_payload = _load_json(graph_run_dir / "pcap_experiment_config.json")
    summary_payload = _load_json(graph_run_dir / "pcap_experiment_summary.json")
    checkpoint_dir = graph_run_dir / "checkpoints" / "best"
    loaded_checkpoint = load_checkpoint(checkpoint_dir, map_location="cpu")
    loaded_checkpoint.model.eval()

    graph_summary = _load_csv(graph_run_dir / "graph_summary.csv")
    train_ids = set(summary_payload.get("split_graph_ids", {}).get("train", ()))
    if not train_ids:
        raise ValueError("The run summary does not contain any train graph ids.")

    short_flow_thresholds = ShortFlowThresholds.from_mapping(
        config_payload.get("short_flow_thresholds", {})
        if isinstance(config_payload.get("short_flow_thresholds", {}), dict)
        else {}
    )
    graph_builder = FlowInteractionGraphBuilder(
        GraphConfig(
            time_window_seconds=int(config_payload.get("window_size", 10)),
            directed=True,
            association_edges=AssociationEdgeConfig(
                enable_same_src_ip=bool(config_payload.get("use_association_edges", True)),
                enable_same_dst_subnet=bool(config_payload.get("use_association_edges", True)),
                dst_subnet_prefix=24,
            ),
        )
    )

    records: list[GraphReductionRecord] = []
    for source_row in _train_source_rows(graph_summary).itertuples(index=False):
        source_id = str(source_row.source_id)
        source_name = str(source_row.source_name)
        source_path = Path(str(source_row.source_path))
        load_result = load_pcap_flow_dataset(
            source_path,
            max_packets=(
                None
                if config_payload.get("packet_limit") in {None, ""}
                else int(config_payload.get("packet_limit"))
            ),
            idle_timeout_seconds=float(config_payload.get("idle_timeout_seconds", 60.0)),
        )
        window_batches = preprocess_flow_dataset(
            load_result.dataset,
            window_size=int(config_payload.get("window_size", 10)),
            rules=short_flow_thresholds,
        )
        graphs = graph_builder.build_many(window_batches)
        smoke_graph_limit = int(config_payload.get("smoke_graph_limit", 0) or 0)
        if smoke_graph_limit > 0:
            graphs = graphs[:smoke_graph_limit]

        selected_graphs: list[InteractionGraph] = []
        selected_ids: list[str] = []
        for entry_index, graph in enumerate(graphs):
            graph_id = _graph_id("benign", source_id, graph, entry_index)
            if graph_id in train_ids:
                selected_graphs.append(graph)
                selected_ids.append(graph_id)

        if not selected_graphs:
            continue

        packed_graphs = transform_graphs(
            selected_graphs,
            loaded_checkpoint.feature_preprocessor,
            include_graph_structural_features=bool(
                config_payload.get("use_graph_structural_features", True)
            ),
        )
        summary_lookup = graph_summary.set_index("graph_id")
        with torch.no_grad():
            for graph_id, graph_sample, packed_graph in zip(
                selected_ids,
                selected_graphs,
                packed_graphs,
                strict=True,
            ):
                output = loaded_checkpoint.model(packed_graph)
                node_scores = compute_node_anomaly_scores(
                    packed_graph.node_features,
                    output.reconstructed_node_features.detach().cpu().numpy(),
                )
                edge_scores = compute_edge_anomaly_scores(
                    packed_graph.edge_features,
                    (
                        output.reconstructed_edge_features.detach().cpu().numpy()
                        if output.reconstructed_edge_features is not None
                        else None
                    ),
                )
                flow_scores = np.asarray(
                    [
                        float(edge_scores[index])
                        for index, edge in enumerate(graph_sample.edges)
                        if edge.edge_type == "communication"
                    ],
                    dtype=float,
                )
                summary_row = summary_lookup.loc[graph_id]
                records.append(
                    GraphReductionRecord(
                        graph_id=graph_id,
                        source_id=source_id,
                        source_name=source_name,
                        source_role="benign",
                        split_assignment="train",
                        raw_label="BENIGN",
                        binary_label=0,
                        graph_score=float(np.mean(node_scores)) if node_scores.size else 0.0,
                        node_max=float(node_scores.max()) if node_scores.size else 0.0,
                        node_topk_mean=_topk_mean(node_scores, node_topk_fraction),
                        flow_p90=_p90(flow_scores),
                        aggregated_edge_count=float(summary_row["aggregated_edge_count"]),
                    )
                )

    train_frame = pd.DataFrame([asdict(record) for record in records]).sort_values(
        by=["source_name", "graph_id"],
        ascending=[True, True],
    )
    if len(train_frame) != len(train_ids):
        raise ValueError(
            "Failed to reconstruct the full benign train reference set: "
            f"expected {len(train_ids)} graphs but recovered {len(train_frame)}."
        )
    return train_frame.reset_index(drop=True)


def _evaluation_reduction_records(
    *,
    graph_run_dir: Path,
    node_topk_fraction: float,
) -> pd.DataFrame:
    """Aggregate exported evaluation score tables into per-graph reduction rows."""

    graph_scores = _load_score_table(graph_run_dir / "scores" / "graph_scores.binary.csv")
    node_scores = _load_score_table(graph_run_dir / "scores" / "node_scores.binary.csv")
    flow_scores = _load_score_table(graph_run_dir / "scores" / "flow_scores.binary.csv")
    graph_summary = _load_csv(graph_run_dir / "graph_summary.csv")

    node_summary = (
        node_scores.loc[:, ["graph_id", "anomaly_score"]]
        .assign(anomaly_score=lambda frame: pd.to_numeric(frame["anomaly_score"], errors="coerce").fillna(0.0))
        .groupby("graph_id")["anomaly_score"]
        .agg(
            node_max="max",
            node_topk_mean=lambda series: _topk_mean(series.to_numpy(dtype=float), node_topk_fraction),
        )
        .reset_index()
    )
    flow_summary = (
        flow_scores.loc[:, ["graph_id", "anomaly_score"]]
        .assign(anomaly_score=lambda frame: pd.to_numeric(frame["anomaly_score"], errors="coerce").fillna(0.0))
        .groupby("graph_id")["anomaly_score"]
        .agg(flow_p90=lambda series: _p90(series.to_numpy(dtype=float)))
        .reset_index()
    )
    base = graph_scores.loc[
        :,
        [
            "graph_id",
            "anomaly_score",
            "meta_source_id",
            "meta_source_name",
            "meta_source_role",
            "meta_split_assignment",
            "meta_raw_label",
            "meta_binary_label",
        ],
    ].rename(
        columns={
            "anomaly_score": "graph_score",
            "meta_source_id": "source_id",
            "meta_source_name": "source_name",
            "meta_source_role": "source_role",
            "meta_split_assignment": "split_assignment",
            "meta_raw_label": "raw_label",
            "meta_binary_label": "binary_label",
        }
    )
    base["graph_score"] = pd.to_numeric(base["graph_score"], errors="coerce").fillna(0.0)
    base["binary_label"] = pd.to_numeric(base["binary_label"], errors="coerce").fillna(0).astype(int)
    summary_columns = graph_summary.loc[
        :,
        ["graph_id", "aggregated_edge_count"],
    ].copy()
    summary_columns["aggregated_edge_count"] = pd.to_numeric(
        summary_columns["aggregated_edge_count"],
        errors="coerce",
    ).fillna(0.0)

    evaluation = base.merge(node_summary, on="graph_id", how="left").merge(
        flow_summary,
        on="graph_id",
        how="left",
    ).merge(
        summary_columns,
        on="graph_id",
        how="left",
    )
    for column in ("node_max", "node_topk_mean", "flow_p90", "aggregated_edge_count"):
        evaluation[column] = pd.to_numeric(evaluation[column], errors="coerce").fillna(0.0)
    return evaluation.sort_values(by=["split_assignment", "source_name", "graph_id"]).reset_index(drop=True)


def _evaluate_source_against_benign(
    evaluation_frame: pd.DataFrame,
    *,
    score_column: str,
    threshold: float,
    source_name: str,
) -> BinaryScoreMetrics:
    """Evaluate one malicious source against the benign holdout using one score column."""

    mask = evaluation_frame["binary_label"].eq(0) | evaluation_frame["source_name"].eq(source_name)
    subset = evaluation_frame.loc[mask].copy()
    return evaluate_scores(
        subset["binary_label"].tolist(),
        subset[score_column].tolist(),
        threshold=threshold,
    )


def _comparison_rows(
    *,
    train_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare several evaluation-only graph score reductions."""

    train_working = train_frame.copy()
    eval_working = evaluation_frame.copy()

    train_working["baseline_mean_node"] = train_working["graph_score"]
    eval_working["baseline_mean_node"] = eval_working["graph_score"]

    train_working["node_max_method"] = train_working["node_max"]
    eval_working["node_max_method"] = eval_working["node_max"]

    train_working["node_topk_mean_5pct"] = train_working["node_topk_mean"]
    eval_working["node_topk_mean_5pct"] = eval_working["node_topk_mean"]

    train_working["flow_p90_method"] = train_working["flow_p90"]
    eval_working["flow_p90_method"] = eval_working["flow_p90"]

    train_working["hybrid_minimal"] = (
        _percentile_against_reference(
            train_working["node_topk_mean"],
            train_working["node_topk_mean"],
        )
        + _percentile_against_reference(
            train_working["aggregated_edge_count"],
            train_working["aggregated_edge_count"],
        )
    ) / 2.0
    eval_working["hybrid_minimal"] = (
        _percentile_against_reference(
            eval_working["node_topk_mean"],
            train_working["node_topk_mean"],
        )
        + _percentile_against_reference(
            eval_working["aggregated_edge_count"],
            train_working["aggregated_edge_count"],
        )
    ) / 2.0

    method_definitions = [
        (
            "baseline_mean_node",
            "mean(node_scores)",
            "Current production graph score.",
        ),
        (
            "node_max_method",
            "max(node_scores)",
            "Pure tail-sensitive node reduction.",
        ),
        (
            "node_topk_mean_5pct",
            "mean(top max(1, ceil(0.05 * node_count)) node_scores)",
            "Tail-sensitive node reduction with a small averaging buffer.",
        ),
        (
            "flow_p90_method",
            "p90(flow_scores)",
            "Flow-level communication edge tail reduction.",
        ),
        (
            "hybrid_minimal",
            "0.5 * node_topk_mean_pct + 0.5 * aggregated_edge_count_pct",
            "Minimal hybrid: one reduction tail signal plus one proven coarse feature.",
        ),
    ]

    malicious_sources = [
        str(source_name)
        for source_name in sorted(
            eval_working.loc[
                eval_working["binary_label"].eq(1),
                "source_name",
            ].dropna().unique()
        )
    ]

    comparison_rows: list[dict[str, object]] = []
    baseline_percentile = _rank_percentile(eval_working["baseline_mean_node"])
    rank_shift_rows: list[dict[str, object]] = []

    for method_name, score_definition, notes in method_definitions:
        train_scores = pd.to_numeric(train_working[method_name], errors="coerce").fillna(0.0)
        eval_scores = pd.to_numeric(eval_working[method_name], errors="coerce").fillna(0.0)
        threshold = _q95(train_scores)
        overall_metrics = evaluate_scores(
            eval_working["binary_label"].tolist(),
            eval_scores.tolist(),
            threshold=threshold,
        )
        benign_test = eval_working.loc[eval_working["binary_label"].eq(0), method_name]
        benign_summary = _quantile_summary(benign_test)
        source_metrics = {
            source_name: _evaluate_source_against_benign(
                eval_working,
                score_column=method_name,
                threshold=threshold,
                source_name=source_name,
            )
            for source_name in malicious_sources
        }
        worst_source_name = ""
        if source_metrics:
            worst_source_name = sorted(
                source_metrics.items(),
                key=lambda item: (
                    item[1].f1 if item[1].f1 is not None else -1.0,
                    item[1].recall if item[1].recall is not None else -1.0,
                    item[0],
                ),
            )[0][0]

        comparison_rows.append(
            {
                "method_name": method_name,
                "score_definition": score_definition,
                "threshold_policy": "q95 benign train reference",
                "train_reference_count": int(len(train_scores)),
                "threshold": float(threshold),
                "benign_test_count": int(eval_working["binary_label"].eq(0).sum()),
                "malicious_test_count": int(eval_working["binary_label"].eq(1).sum()),
                "overall_fpr": _false_positive_rate(overall_metrics),
                "overall_recall": float(overall_metrics.recall or 0.0),
                "overall_f1": float(overall_metrics.f1 or 0.0),
                "recon_recall": float(
                    source_metrics.get("Recon-HostDiscovery", BinaryScoreMetrics(
                        threshold=threshold,
                        support=0,
                        positive_count=0,
                        negative_count=0,
                        roc_auc=None,
                        pr_auc=None,
                        precision=0.0,
                        recall=0.0,
                        f1=0.0,
                        true_positive=0,
                        false_positive=0,
                        true_negative=0,
                        false_negative=0,
                    )).recall or 0.0
                ),
                "recon_f1": float(
                    source_metrics.get("Recon-HostDiscovery", BinaryScoreMetrics(
                        threshold=threshold,
                        support=0,
                        positive_count=0,
                        negative_count=0,
                        roc_auc=None,
                        pr_auc=None,
                        precision=0.0,
                        recall=0.0,
                        f1=0.0,
                        true_positive=0,
                        false_positive=0,
                        true_negative=0,
                        false_negative=0,
                    )).f1 or 0.0
                ),
                "browserhijacking_recall": float(
                    source_metrics.get("BrowserHijacking", BinaryScoreMetrics(
                        threshold=threshold,
                        support=0,
                        positive_count=0,
                        negative_count=0,
                        roc_auc=None,
                        pr_auc=None,
                        precision=0.0,
                        recall=0.0,
                        f1=0.0,
                        true_positive=0,
                        false_positive=0,
                        true_negative=0,
                        false_negative=0,
                    )).recall or 0.0
                ),
                "browserhijacking_f1": float(
                    source_metrics.get("BrowserHijacking", BinaryScoreMetrics(
                        threshold=threshold,
                        support=0,
                        positive_count=0,
                        negative_count=0,
                        roc_auc=None,
                        pr_auc=None,
                        precision=0.0,
                        recall=0.0,
                        f1=0.0,
                        true_positive=0,
                        false_positive=0,
                        true_negative=0,
                        false_negative=0,
                    )).f1 or 0.0
                ),
                "ddos_recall": float(
                    source_metrics.get("DDoS-ICMP_Flood", BinaryScoreMetrics(
                        threshold=threshold,
                        support=0,
                        positive_count=0,
                        negative_count=0,
                        roc_auc=None,
                        pr_auc=None,
                        precision=0.0,
                        recall=0.0,
                        f1=0.0,
                        true_positive=0,
                        false_positive=0,
                        true_negative=0,
                        false_negative=0,
                    )).recall or 0.0
                ),
                "ddos_f1": float(
                    source_metrics.get("DDoS-ICMP_Flood", BinaryScoreMetrics(
                        threshold=threshold,
                        support=0,
                        positive_count=0,
                        negative_count=0,
                        roc_auc=None,
                        pr_auc=None,
                        precision=0.0,
                        recall=0.0,
                        f1=0.0,
                        true_positive=0,
                        false_positive=0,
                        true_negative=0,
                        false_negative=0,
                    )).f1 or 0.0
                ),
                "worst_malicious_source_name": worst_source_name,
                "benign_test_mean_score": float(benign_summary["mean"]),
                "benign_test_p95": float(benign_summary["p95"]),
                "threshold_minus_benign_test_p95": float(threshold - benign_summary["p95"]),
                "notes": notes,
            }
        )

        method_percentile = _rank_percentile(eval_scores)
        for row, base_pct, method_pct, raw_score in zip(
            eval_working.itertuples(index=False),
            baseline_percentile,
            method_percentile,
            eval_scores,
            strict=True,
        ):
            rank_shift_rows.append(
                {
                    "graph_id": row.graph_id,
                    "source_name": row.source_name,
                    "source_role": row.source_role,
                    "split_assignment": row.split_assignment,
                    "binary_label": int(row.binary_label),
                    "method_name": method_name,
                    "baseline_percentile": float(base_pct),
                    "method_percentile": float(method_pct),
                    "method_score": float(raw_score),
                    "percentile_gap_signed": float(method_pct - base_pct),
                    "percentile_gap_abs": float(abs(method_pct - base_pct)),
                }
            )

    comparison = pd.DataFrame(comparison_rows).sort_values(
        by=["recon_recall", "overall_f1", "method_name"],
        ascending=[False, False, True],
    )
    rank_shift = pd.DataFrame(rank_shift_rows)
    rank_shift = rank_shift.loc[
        rank_shift["method_name"] != "baseline_mean_node"
    ].sort_values(
        by=["percentile_gap_abs", "method_name"],
        ascending=[False, True],
    )
    return comparison.reset_index(drop=True), rank_shift.reset_index(drop=True)


def _write_markdown(
    path: Path,
    *,
    comparison: pd.DataFrame,
    rank_shift: pd.DataFrame,
) -> None:
    """Write a compact Markdown summary of the reduction comparison."""

    baseline = comparison.loc[
        comparison["method_name"].eq("baseline_mean_node")
    ].iloc[0]
    best_recon = comparison.sort_values(
        by=["recon_recall", "recon_f1", "overall_f1"],
        ascending=[False, False, False],
    ).iloc[0]
    browser_best = comparison.sort_values(
        by=["browserhijacking_recall", "browserhijacking_f1", "overall_f1"],
        ascending=[False, False, False],
    ).iloc[0]
    source_gap = (
        rank_shift.groupby(["method_name", "source_name"])["percentile_gap_signed"]
        .mean()
        .reset_index()
    )
    recon_gap_rows = source_gap.loc[
        source_gap["source_name"].eq("Recon-HostDiscovery")
    ].sort_values(by="percentile_gap_signed", ascending=False)
    browser_gap_rows = source_gap.loc[
        source_gap["source_name"].eq("BrowserHijacking")
    ].sort_values(by="percentile_gap_signed", ascending=False)

    lines = [
        "# Graph Reduction Diagnosis",
        "",
        "## Scope",
        "",
        "- Representative graph run: real PCAP, `packet_limit=20000`, confirmed benign set, GAE backend.",
        "- No retraining was performed. Train-reference reductions were rebuilt by replaying the saved checkpoint on the original benign train graphs.",
        "- All candidate methods use the same threshold rule: q95 benign train reference for that method's score definition.",
        "",
        "## Compared Methods",
        "",
        "- `baseline_mean_node`: `mean(node_scores)`",
        "- `node_max`: `max(node_scores)`",
        "- `node_topk_mean_5pct`: mean of the top `max(1, ceil(0.05 * node_count))` node scores",
        "- `flow_p90`: p90 of communication flow scores",
        "- `hybrid_minimal`: 0.5 * node_topk percentile + 0.5 * aggregated_edge_count percentile",
        "",
        "## Headline Result",
        "",
        (
            f"- Baseline mean(node): overall_fpr={float(baseline['overall_fpr']):.6f}, "
            f"overall_recall={float(baseline['overall_recall']):.6f}, "
            f"recon_recall={float(baseline['recon_recall']):.6f}, "
            f"browser_recall={float(baseline['browserhijacking_recall']):.6f}."
        ),
        (
            f"- Best Recon recovery: {best_recon['method_name']} "
            f"(recon_recall={float(best_recon['recon_recall']):.6f}, "
            f"recon_f1={float(best_recon['recon_f1']):.6f}, "
            f"overall_fpr={float(best_recon['overall_fpr']):.6f})."
        ),
        (
            f"- Best Browser result: {browser_best['method_name']} "
            f"(browser_recall={float(browser_best['browserhijacking_recall']):.6f}, "
            f"browser_f1={float(browser_best['browserhijacking_f1']):.6f})."
        ),
        "",
        "## Interpretation",
        "",
        (
            "- If node_max / node_topk / flow_p90 improves Recon under the same q95 train threshold, "
            "that is direct evidence that the current failure sits in final reduction rather than in the encoder alone."
        ),
        (
            "- If BrowserHijacking remains near zero across these alternative reductions, "
            "that supports the earlier diagnosis that Browser is primarily a weak-signal / feature-coverage problem."
        ),
        "",
        "## Mean Rank Shift vs Baseline",
        "",
    ]

    if not recon_gap_rows.empty:
        lines.append(
            "- Recon mean percentile shift by method: "
            + ", ".join(
                f"{row.method_name}={float(row.percentile_gap_signed):.6f}"
                for row in recon_gap_rows.itertuples(index=False)
            )
        )
    if not browser_gap_rows.empty:
        lines.append(
            "- BrowserHijacking mean percentile shift by method: "
            + ", ".join(
                f"{row.method_name}={float(row.percentile_gap_signed):.6f}"
                for row in browser_gap_rows.itertuples(index=False)
            )
        )

    lines.extend(
        [
            "",
            "## Bottom Line",
            "",
            (
                "- This comparison is evaluation-only, but it is enough to tell us whether "
                "changing graph-level reduction is likely to recover Recon without touching training."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(
    *,
    graph_run_dir: Path,
    output_dir: Path,
    node_topk_fraction: float,
) -> dict[str, Path]:
    """Run the reduction comparison and export compact CSV/Markdown artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    train_frame = _rebuild_train_reference_records(
        graph_run_dir=graph_run_dir,
        node_topk_fraction=node_topk_fraction,
    )
    evaluation_frame = _evaluation_reduction_records(
        graph_run_dir=graph_run_dir,
        node_topk_fraction=node_topk_fraction,
    )
    comparison, rank_shift = _comparison_rows(
        train_frame=train_frame,
        evaluation_frame=evaluation_frame,
    )

    train_reference_path = output_dir / "graph_reduction_train_reference.csv"
    train_frame.to_csv(train_reference_path, index=False)
    comparison_path = output_dir / "graph_reduction_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    rank_shift_path = output_dir / "graph_reduction_rank_shift.csv"
    rank_shift.to_csv(rank_shift_path, index=False)
    markdown_path = output_dir / "graph_reduction_diagnosis.md"
    _write_markdown(
        markdown_path,
        comparison=comparison,
        rank_shift=rank_shift,
    )
    return {
        "graph_reduction_train_reference": train_reference_path,
        "graph_reduction_comparison": comparison_path,
        "graph_reduction_rank_shift": rank_shift_path,
        "graph_reduction_diagnosis": markdown_path,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the CLI entrypoint and print the exported artifact paths."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    outputs = run_analysis(
        graph_run_dir=Path(args.graph_run_dir),
        output_dir=Path(args.output_dir),
        node_topk_fraction=float(args.node_topk_fraction),
    )
    print(json.dumps({key: value.as_posix() for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
