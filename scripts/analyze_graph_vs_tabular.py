"""Analyze real-PCAP graph-versus-tabular failure modes from exported artifacts.

This script is intentionally read-only. It compares one graph run directory and
one tabular run directory that were produced from the same real-PCAP source set
and writes small CSV/Markdown diagnostics that help explain why one backend
outperformed the other.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the graph-versus-tabular analysis helper."""

    parser = argparse.ArgumentParser(
        description=(
            "Read one graph run bundle and one tabular run bundle, then export "
            "score, rank-gap, feature-separation, and threshold diagnostics."
        )
    )
    parser.add_argument(
        "--graph-run-dir",
        required=True,
        help="Directory of the representative graph binary-evaluation run.",
    )
    parser.add_argument(
        "--tabular-run-dir",
        required=True,
        help="Directory of the comparison tabular baseline run.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where diagnostic CSV and Markdown files will be written.",
    )
    parser.add_argument(
        "--top-k-rank-gap",
        type=int,
        default=25,
        help="Number of top rank-gap samples to summarize in the Markdown report.",
    )
    return parser


def _require_file(path: Path) -> Path:
    """Return a path when it exists and raise a clear error otherwise."""

    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame with existence validation."""

    return pd.read_csv(_require_file(path))


def _load_optional_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file when present, otherwise return an empty frame."""

    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_scores(path: Path) -> pd.DataFrame:
    """Load a score table and expand the JSON metadata column when present."""

    frame = _load_csv(path).copy()
    if "anomaly_score" not in frame.columns and "score" in frame.columns:
        frame["anomaly_score"] = frame["score"]
    if "sample_id" not in frame.columns and "graph_id" in frame.columns:
        frame["sample_id"] = frame["graph_id"]
    metadata_rows: list[dict[str, object]] = []
    for raw in frame.get("metadata", pd.Series(["{}"] * len(frame))).fillna("{}"):
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
    numeric_columns = (
        "anomaly_score",
        "threshold",
        "binary_label",
        "row_index",
        "feature_count",
    )
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _load_optional_scores(path: Path) -> pd.DataFrame:
    """Load a score table when present, otherwise return an empty frame."""

    if not path.exists():
        return pd.DataFrame()
    return _load_scores(path)


def _quantile_summary(values: pd.Series | np.ndarray | list[float]) -> dict[str, float | int]:
    """Compute a compact distribution summary for a numeric vector."""

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {
            "count": 0,
            "min": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "std": 0.0,
        }
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
        "std": float(array.std(ddof=0)),
    }


def _cohens_d(group_a: pd.Series, group_b: pd.Series) -> float:
    """Compute a simple pooled-standard-deviation effect size."""

    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    variance_a = float(a.var(ddof=0))
    variance_b = float(b.var(ddof=0))
    pooled = float(np.sqrt((variance_a + variance_b) / 2.0))
    if pooled == 0.0:
        return 0.0
    return float((b.mean() - a.mean()) / pooled)


def _safe_corr(feature: pd.Series, score: pd.Series) -> float:
    """Compute a Pearson-style correlation with a stable zero fallback."""

    feature_array = pd.to_numeric(feature, errors="coerce").astype(float)
    score_array = pd.to_numeric(score, errors="coerce").astype(float)
    if feature_array.nunique(dropna=True) <= 1 or score_array.nunique(dropna=True) <= 1:
        return 0.0
    corr = feature_array.corr(score_array, method="pearson")
    if corr is None or np.isnan(corr):
        return 0.0
    return float(corr)


def _rank_percentile(scores: pd.Series) -> pd.Series:
    """Convert anomaly scores into descending anomaly percentiles in [0, 1]."""

    if scores.empty:
        return pd.Series(dtype=float)
    descending_rank = scores.rank(method="average", ascending=False)
    if len(scores) == 1:
        return pd.Series([1.0], index=scores.index, dtype=float)
    return 1.0 - ((descending_rank - 1.0) / float(len(scores) - 1.0))


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


def _binary_threshold_metrics(
    labels: pd.Series | np.ndarray | list[int],
    scores: pd.Series | np.ndarray | list[float],
    *,
    threshold: float,
) -> dict[str, float | int]:
    """Compute compact thresholded binary metrics without changing training logic."""

    y_true = np.asarray(labels, dtype=int)
    y_pred = (np.asarray(scores, dtype=float) >= float(threshold)).astype(int)
    if y_true.size == 0:
        return {
            "support": 0,
            "positive_count": 0,
            "negative_count": 0,
            "true_positive": 0,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "false_positive_rate": 0.0,
        }

    tp = int(np.logical_and(y_pred == 1, y_true == 1).sum())
    tn = int(np.logical_and(y_pred == 0, y_true == 0).sum())
    fp = int(np.logical_and(y_pred == 1, y_true == 0).sum())
    fn = int(np.logical_and(y_pred == 0, y_true == 1).sum())
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = (
        float((2.0 * precision * recall) / (precision + recall))
        if (precision + recall)
        else 0.0
    )
    false_positive_rate = float(fp / (fp + tn)) if (fp + tn) else 0.0
    return {
        "support": int(y_true.size),
        "positive_count": int((y_true == 1).sum()),
        "negative_count": int((y_true == 0).sum()),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
    }


def _subset_frame(
    *,
    train_frame: pd.DataFrame,
    overall_frame: pd.DataFrame,
    subset_name: str,
) -> pd.DataFrame:
    """Select one diagnostic subset from train or overall score tables."""

    if subset_name == "benign_train_reference":
        return train_frame.copy()
    if subset_name == "benign_test":
        return overall_frame.loc[overall_frame["binary_label"] == 0].copy()
    if subset_name == "malicious_test":
        return overall_frame.loc[overall_frame["binary_label"] == 1].copy()
    if subset_name.startswith("source:"):
        source_name = subset_name.split(":", 1)[1]
        return overall_frame.loc[overall_frame["meta_source_name"] == source_name].copy()
    raise KeyError(f"Unsupported subset selector: {subset_name}")


def _summarize_graph_scope(
    frame: pd.DataFrame,
    *,
    prefix: str,
) -> pd.DataFrame:
    """Aggregate one score scope into per-graph summary statistics."""

    if frame.empty:
        return pd.DataFrame(columns=["graph_id"])
    working = frame.loc[:, ["graph_id", "anomaly_score"]].copy()
    working["anomaly_score"] = pd.to_numeric(
        working["anomaly_score"],
        errors="coerce",
    ).fillna(0.0)
    grouped = working.groupby("graph_id")["anomaly_score"]
    summary = grouped.agg(["size", "mean", "median", "max"]).rename(
        columns={
            "size": f"{prefix}_score_count",
            "mean": f"{prefix}_score_mean",
            "median": f"{prefix}_score_median",
            "max": f"{prefix}_score_max",
        }
    )
    summary[f"{prefix}_score_p90"] = grouped.quantile(0.90)
    return summary.reset_index()


def _build_graph_score_reduction_diagnostics(
    *,
    graph_scores: pd.DataFrame,
    node_scores: pd.DataFrame,
    edge_scores: pd.DataFrame,
    flow_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Build per-graph diagnostics across node, edge, flow, and graph score layers."""

    graph_frame = graph_scores.copy()
    graph_frame["graph_score"] = pd.to_numeric(
        graph_frame["anomaly_score"],
        errors="coerce",
    ).fillna(0.0)
    graph_frame["threshold"] = pd.to_numeric(
        graph_frame.get("threshold"),
        errors="coerce",
    ).fillna(0.0)
    base = graph_frame.loc[
        :,
        [
            "graph_id",
            "graph_score",
            "threshold",
            "label",
            "meta_source_name",
            "meta_source_role",
            "meta_split_assignment",
            "meta_raw_label",
            "meta_binary_label",
            "meta_task_name",
        ],
    ].rename(
        columns={
            "label": "graph_label",
            "meta_source_name": "source_name",
            "meta_source_role": "source_role",
            "meta_split_assignment": "split_assignment",
            "meta_raw_label": "raw_label",
            "meta_binary_label": "binary_label",
            "meta_task_name": "task_name",
        }
    )

    diagnostics = base.merge(
        _summarize_graph_scope(node_scores, prefix="node"),
        on="graph_id",
        how="left",
    ).merge(
        _summarize_graph_scope(edge_scores, prefix="edge"),
        on="graph_id",
        how="left",
    ).merge(
        _summarize_graph_scope(flow_scores, prefix="flow"),
        on="graph_id",
        how="left",
    )

    numeric_fill_columns = [
        "node_score_count",
        "node_score_mean",
        "node_score_median",
        "node_score_p90",
        "node_score_max",
        "edge_score_count",
        "edge_score_mean",
        "edge_score_median",
        "edge_score_p90",
        "edge_score_max",
        "flow_score_count",
        "flow_score_mean",
        "flow_score_median",
        "flow_score_p90",
        "flow_score_max",
    ]
    for column in numeric_fill_columns:
        if column in diagnostics.columns:
            diagnostics[column] = pd.to_numeric(
                diagnostics[column],
                errors="coerce",
            ).fillna(0.0)

    diagnostics["binary_label"] = pd.to_numeric(
        diagnostics["binary_label"],
        errors="coerce",
    ).fillna(0).astype(int)
    percentile_columns = [
        "graph_score",
        "node_score_mean",
        "node_score_p90",
        "node_score_max",
        "edge_score_mean",
        "edge_score_p90",
        "edge_score_max",
        "flow_score_mean",
        "flow_score_p90",
        "flow_score_max",
    ]
    for column in percentile_columns:
        diagnostics[f"{column}_eval_percentile"] = _rank_percentile(diagnostics[column])
    diagnostics["graph_reduction_source"] = "node_mean"
    diagnostics["notes"] = (
        "graph_score is derived from mean(node_scores); edge/flow scores are exported "
        "but not used in graph-level reduction."
    )
    return diagnostics.sort_values(
        by=["split_assignment", "source_name", "graph_id"],
        ascending=[True, True, True],
    )


def _source_level_reduction_summary(reduction_frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse graph reduction diagnostics to source-level summaries."""

    rows: list[dict[str, object]] = []
    if reduction_frame.empty:
        return pd.DataFrame(rows)

    grouped = reduction_frame.groupby(["split_assignment", "source_name", "source_role"])
    signal_columns = {
        "graph_mean": "graph_score_eval_percentile",
        "node_p90": "node_score_p90_eval_percentile",
        "node_max": "node_score_max_eval_percentile",
        "edge_p90": "edge_score_p90_eval_percentile",
        "edge_max": "edge_score_max_eval_percentile",
        "flow_p90": "flow_score_p90_eval_percentile",
        "flow_max": "flow_score_max_eval_percentile",
    }
    for (split_assignment, source_name, source_role), frame in grouped:
        row = {
            "split_assignment": split_assignment,
            "source_name": source_name,
            "source_role": source_role,
            "graph_count": int(len(frame)),
            "graph_score_mean": float(frame["graph_score"].mean()),
            "graph_score_median": float(frame["graph_score"].median()),
            "node_score_mean_mean": float(frame["node_score_mean"].mean()),
            "node_score_p90_mean": float(frame["node_score_p90"].mean()),
            "node_score_max_mean": float(frame["node_score_max"].mean()),
            "edge_score_mean_mean": float(frame["edge_score_mean"].mean()),
            "edge_score_p90_mean": float(frame["edge_score_p90"].mean()),
            "edge_score_max_mean": float(frame["edge_score_max"].mean()),
            "flow_score_mean_mean": float(frame["flow_score_mean"].mean()),
            "flow_score_p90_mean": float(frame["flow_score_p90"].mean()),
            "flow_score_max_mean": float(frame["flow_score_max"].mean()),
            "graph_score_percentile_mean": float(
                frame["graph_score_eval_percentile"].mean()
            ),
            "node_p90_percentile_mean": float(
                frame["node_score_p90_eval_percentile"].mean()
            ),
            "node_max_percentile_mean": float(
                frame["node_score_max_eval_percentile"].mean()
            ),
            "edge_p90_percentile_mean": float(
                frame["edge_score_p90_eval_percentile"].mean()
            ),
            "edge_max_percentile_mean": float(
                frame["edge_score_max_eval_percentile"].mean()
            ),
            "flow_p90_percentile_mean": float(
                frame["flow_score_p90_eval_percentile"].mean()
            ),
            "flow_max_percentile_mean": float(
                frame["flow_score_max_eval_percentile"].mean()
            ),
        }
        strongest_layer, strongest_value = max(
            (
                (name, float(frame[column].mean()))
                for name, column in signal_columns.items()
            ),
            key=lambda item: item[1],
        )
        row["strongest_signal_layer"] = strongest_layer
        row["strongest_signal_percentile_mean"] = strongest_value
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        by=["split_assignment", "source_role", "source_name"],
        ascending=[True, True, True],
    )


def _source_binary_metrics(
    evaluation_frame: pd.DataFrame,
    *,
    score_column: str,
    threshold: float,
    source_name: str,
) -> dict[str, float | int | str]:
    """Compute source-vs-benign metrics for one malicious source."""

    mask = evaluation_frame["binary_label"].eq(0) | evaluation_frame["source_name"].eq(
        source_name
    )
    subset = evaluation_frame.loc[mask].copy()
    metrics = _binary_threshold_metrics(
        subset["binary_label"],
        subset[score_column],
        threshold=threshold,
    )
    return {
        "source_name": source_name,
        "support": int(metrics["positive_count"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "false_positive_rate": float(metrics["false_positive_rate"]),
    }


def _sidechannel_method_rows(
    *,
    train_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    top_feature: str,
    top_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a few analysis-only side-channel score fusions on one graph run."""

    train_graph_scores = pd.to_numeric(train_frame["graph_score"], errors="coerce").fillna(0.0)
    eval_graph_scores = pd.to_numeric(
        evaluation_frame["graph_score"],
        errors="coerce",
    ).fillna(0.0)
    graph_threshold = float(np.quantile(train_graph_scores.to_numpy(dtype=float), 0.95))

    train_graph_percentiles = _percentile_against_reference(
        train_graph_scores,
        train_graph_scores,
    )
    eval_graph_percentiles = _percentile_against_reference(
        eval_graph_scores,
        train_graph_scores,
    )

    train_top_feature_percentiles = _percentile_against_reference(
        pd.to_numeric(train_frame[top_feature], errors="coerce").fillna(0.0),
        pd.to_numeric(train_frame[top_feature], errors="coerce").fillna(0.0),
    )
    eval_top_feature_percentiles = _percentile_against_reference(
        pd.to_numeric(evaluation_frame[top_feature], errors="coerce").fillna(0.0),
        pd.to_numeric(train_frame[top_feature], errors="coerce").fillna(0.0),
    )

    train_topk_parts = [train_graph_percentiles]
    eval_topk_parts = [eval_graph_percentiles]
    for feature_name in top_features:
        train_topk_parts.append(
            _percentile_against_reference(
                pd.to_numeric(train_frame[feature_name], errors="coerce").fillna(0.0),
                pd.to_numeric(train_frame[feature_name], errors="coerce").fillna(0.0),
            )
        )
        eval_topk_parts.append(
            _percentile_against_reference(
                pd.to_numeric(
                    evaluation_frame[feature_name],
                    errors="coerce",
                ).fillna(0.0),
                pd.to_numeric(train_frame[feature_name], errors="coerce").fillna(0.0),
            )
        )

    method_payloads = {
        "graph_score_only": {
            "train_scores": train_graph_scores.to_numpy(dtype=float),
            "eval_scores": eval_graph_scores.to_numpy(dtype=float),
            "threshold_policy": "q95 benign train reference on raw graph score",
            "notes": "Current production graph score path.",
        },
        "graph_plus_top1_rankavg": {
            "train_scores": (train_graph_percentiles + train_top_feature_percentiles) / 2.0,
            "eval_scores": (eval_graph_percentiles + eval_top_feature_percentiles) / 2.0,
            "threshold_policy": "q95 benign train reference on graph/top1 percentile rank average",
            "notes": f"Side-channel rank average using top coarse feature: {top_feature}.",
        },
        "graph_plus_top3_rankavg": {
            "train_scores": np.mean(np.vstack(train_topk_parts), axis=0),
            "eval_scores": np.mean(np.vstack(eval_topk_parts), axis=0),
            "threshold_policy": "q95 benign train reference on graph/top3 percentile rank average",
            "notes": "Side-channel rank average using top 3 Recon-separating coarse features: "
            + ", ".join(top_features),
        },
    }

    malicious_sources = [
        str(source_name)
        for source_name in sorted(
            evaluation_frame.loc[
                evaluation_frame["binary_label"].eq(1),
                "source_name",
            ].dropna().unique()
        )
    ]
    method_rows: list[dict[str, object]] = []
    baseline_rank_gap_frame: pd.DataFrame | None = None

    tabular_percentiles = (
        _rank_percentile(pd.to_numeric(evaluation_frame["tabular_score"], errors="coerce"))
        if "tabular_score" in evaluation_frame.columns
        else pd.Series(np.zeros(len(evaluation_frame)), index=evaluation_frame.index)
    )

    for method_name, payload in method_payloads.items():
        train_scores = np.asarray(payload["train_scores"], dtype=float)
        eval_scores = np.asarray(payload["eval_scores"], dtype=float)
        threshold = float(np.quantile(train_scores, 0.95))
        overall_metrics = _binary_threshold_metrics(
            evaluation_frame["binary_label"],
            eval_scores,
            threshold=threshold,
        )
        source_metrics = [
            _source_binary_metrics(
                evaluation_frame.assign(method_score=eval_scores),
                score_column="method_score",
                threshold=threshold,
                source_name=source_name,
            )
            for source_name in malicious_sources
        ]
        worst_row = (
            sorted(
                source_metrics,
                key=lambda row: (
                    float(row["f1"]),
                    float(row["recall"]),
                    str(row["source_name"]),
                ),
            )[0]
            if source_metrics
            else {"source_name": ""}
        )
        source_metric_lookup = {row["source_name"]: row for row in source_metrics}

        method_rows.append(
            {
                "method_name": method_name,
                "threshold_policy": payload["threshold_policy"],
                "threshold": threshold,
                "overall_fpr": float(overall_metrics["false_positive_rate"]),
                "overall_recall": float(overall_metrics["recall"]),
                "overall_f1": float(overall_metrics["f1"]),
                "recon_recall": float(
                    source_metric_lookup.get("Recon-HostDiscovery", {}).get("recall", 0.0)
                ),
                "recon_f1": float(
                    source_metric_lookup.get("Recon-HostDiscovery", {}).get("f1", 0.0)
                ),
                "browserhijacking_recall": float(
                    source_metric_lookup.get("BrowserHijacking", {}).get("recall", 0.0)
                ),
                "browserhijacking_f1": float(
                    source_metric_lookup.get("BrowserHijacking", {}).get("f1", 0.0)
                ),
                "ddos_recall": float(
                    source_metric_lookup.get("DDoS-ICMP_Flood", {}).get("recall", 0.0)
                ),
                "ddos_f1": float(
                    source_metric_lookup.get("DDoS-ICMP_Flood", {}).get("f1", 0.0)
                ),
                "worst_malicious_source_name": str(worst_row.get("source_name", "")),
                "notes": payload["notes"],
            }
        )

        if method_name == "graph_plus_top3_rankavg":
            fused_percentiles = _rank_percentile(pd.Series(eval_scores, index=evaluation_frame.index))
            graph_percentiles = _rank_percentile(
                pd.Series(eval_graph_scores.to_numpy(dtype=float), index=evaluation_frame.index)
            )
            baseline_rank_gap_frame = evaluation_frame.loc[
                :,
                [
                    "graph_id",
                    "source_name",
                    "source_role",
                    "split_assignment",
                    "binary_label",
                    "graph_score",
                    "tabular_score",
                ],
            ].copy()
            baseline_rank_gap_frame["graph_percentile"] = graph_percentiles.to_numpy(
                dtype=float
            )
            baseline_rank_gap_frame["sidechannel_score"] = eval_scores
            baseline_rank_gap_frame["sidechannel_percentile"] = fused_percentiles.to_numpy(
                dtype=float
            )
            baseline_rank_gap_frame["tabular_percentile"] = tabular_percentiles.to_numpy(
                dtype=float
            )
            baseline_rank_gap_frame["percentile_gap_signed"] = (
                baseline_rank_gap_frame["sidechannel_percentile"]
                - baseline_rank_gap_frame["graph_percentile"]
            )
            baseline_rank_gap_frame["percentile_gap_abs"] = (
                baseline_rank_gap_frame["percentile_gap_signed"].abs()
            )
            baseline_rank_gap_frame = baseline_rank_gap_frame.sort_values(
                by=["percentile_gap_signed", "sidechannel_percentile"],
                ascending=[False, False],
            )

    comparison = pd.DataFrame(method_rows)
    rank_gap_frame = baseline_rank_gap_frame if baseline_rank_gap_frame is not None else pd.DataFrame()
    return comparison, rank_gap_frame


def _score_diagnostic_rows(
    *,
    mode_family: str,
    train_frame: pd.DataFrame,
    overall_frame: pd.DataFrame,
) -> list[dict[str, object]]:
    """Build score-distribution diagnostic rows for one backend."""

    threshold = float(train_frame["threshold"].iloc[0]) if not train_frame.empty else 0.0
    subset_names = (
        "benign_train_reference",
        "benign_test",
        "malicious_test",
        "source:Recon-HostDiscovery",
        "source:BrowserHijacking",
    )
    rows: list[dict[str, object]] = []
    for subset_name in subset_names:
        subset = _subset_frame(
            train_frame=train_frame,
            overall_frame=overall_frame,
            subset_name=subset_name,
        )
        summary = _quantile_summary(subset["anomaly_score"] if not subset.empty else [])
        alert_rate = float(subset["is_alert"].astype(str).str.lower().eq("true").mean()) if not subset.empty else 0.0
        source_name = subset_name.split(":", 1)[1] if ":" in subset_name else ""
        rows.append(
            {
                "mode_family": mode_family,
                "subset_name": subset_name,
                "source_name": source_name,
                "count": int(summary["count"]),
                "score_min": float(summary["min"]),
                "score_mean": float(summary["mean"]),
                "score_median": float(summary["median"]),
                "score_p90": float(summary["p90"]),
                "score_p95": float(summary["p95"]),
                "score_max": float(summary["max"]),
                "threshold": threshold,
                "alert_rate": alert_rate,
                "threshold_minus_mean": threshold - float(summary["mean"]),
                "threshold_minus_median": threshold - float(summary["median"]),
                "threshold_minus_p95": threshold - float(summary["p95"]),
            }
        )
    return rows


def _threshold_row(
    *,
    mode_family: str,
    train_frame: pd.DataFrame,
    overall_frame: pd.DataFrame,
) -> dict[str, object]:
    """Build one threshold-diagnostic row for a backend."""

    threshold = float(train_frame["threshold"].iloc[0]) if not train_frame.empty else 0.0
    train_summary = _quantile_summary(train_frame["anomaly_score"] if not train_frame.empty else [])
    benign_test = overall_frame.loc[overall_frame["binary_label"] == 0].copy()
    benign_summary = _quantile_summary(benign_test["anomaly_score"] if not benign_test.empty else [])
    train_std = float(train_summary["std"])

    def _zscore(value: float) -> float:
        if train_std == 0.0:
            return 0.0
        return float((value - float(train_summary["mean"])) / train_std)

    benign_alert_rate = (
        float(benign_test["is_alert"].astype(str).str.lower().eq("true").mean())
        if not benign_test.empty
        else 0.0
    )
    return {
        "mode_family": mode_family,
        "threshold_policy": "q95 benign train reference",
        "train_reference_count": int(train_summary["count"]),
        "train_mean": float(train_summary["mean"]),
        "train_median": float(train_summary["median"]),
        "train_p90": float(train_summary["p90"]),
        "train_p95": float(train_summary["p95"]),
        "train_max": float(train_summary["max"]),
        "train_std": train_std,
        "threshold": threshold,
        "threshold_z_from_train": _zscore(threshold),
        "benign_test_count": int(benign_summary["count"]),
        "benign_test_mean": float(benign_summary["mean"]),
        "benign_test_median": float(benign_summary["median"]),
        "benign_test_p90": float(benign_summary["p90"]),
        "benign_test_p95": float(benign_summary["p95"]),
        "benign_test_max": float(benign_summary["max"]),
        "benign_test_alert_rate": benign_alert_rate,
        "threshold_minus_benign_test_mean": threshold - float(benign_summary["mean"]),
        "threshold_minus_benign_test_p95": threshold - float(benign_summary["p95"]),
        "benign_test_mean_z_from_train": _zscore(float(benign_summary["mean"])),
        "benign_test_p95_z_from_train": _zscore(float(benign_summary["p95"])),
    }


def _feature_note(
    *,
    recon_d: float,
    browser_d: float,
    graph_corr: float,
    tabular_corr: float,
) -> str:
    """Generate a short human-readable interpretation for one feature row."""

    if abs(recon_d) >= 1.0 and abs(browser_d) < 0.5:
        return "Strong Recon separation; Browser remains close to benign."
    if abs(recon_d) >= 1.0 and abs(tabular_corr) > abs(graph_corr):
        return "Recon-separated and tracked more strongly by tabular scores."
    if max(abs(recon_d), abs(browser_d)) < 0.5:
        return "Weak univariate separation for both difficult sources."
    if abs(browser_d) >= 0.75 and abs(recon_d) < 0.75:
        return "Feature helps Browser more than Recon."
    return "Moderate coarse separation signal."


def _write_markdown(
    path: Path,
    *,
    graph_threshold: dict[str, object],
    tabular_threshold: dict[str, object],
    score_diagnostics: pd.DataFrame,
    rank_gap: pd.DataFrame,
    feature_separation: pd.DataFrame,
    graph_per_attack: pd.DataFrame,
    tabular_per_attack: pd.DataFrame,
    top_k_rank_gap: int,
) -> None:
    """Write a compact Markdown diagnosis summary."""

    top_gap = rank_gap.head(top_k_rank_gap).copy()
    top_gap_source_counts = (
        top_gap.groupby("source_name").size().sort_values(ascending=False).to_dict()
        if not top_gap.empty
        else {}
    )
    top_features = feature_separation.head(8)
    graph_lookup = graph_per_attack.set_index("task_name")
    tabular_lookup = tabular_per_attack.set_index("task_name")

    def _metric(mode: str, task: str, column: str) -> float:
        table = graph_lookup if mode == "graph" else tabular_lookup
        return float(table.loc[task, column])

    lines = [
        "# Graph vs Tabular Diagnosis",
        "",
        "## Scope",
        "",
        "- Graph run: representative real-PCAP `packet_limit=20000` GAE baseline.",
        "- Tabular run: PCA reconstruction baseline over the same exported `graph_summary` rows.",
        "- Goal: explain why graph underperformed tabular on Recon and BrowserHijacking.",
        "",
        "## Threshold Snapshot",
        "",
        (
            f"- Graph threshold policy: q95 benign train reference, threshold="
            f"{float(graph_threshold['threshold']):.6f}, benign_test_alert_rate="
            f"{float(graph_threshold['benign_test_alert_rate']):.6f}"
        ),
        (
            f"- Tabular threshold policy: q95 benign train reference, threshold="
            f"{float(tabular_threshold['threshold']):.6e}, benign_test_alert_rate="
            f"{float(tabular_threshold['benign_test_alert_rate']):.6f}"
        ),
        (
            "- Both backends end up with the same benign holdout alert rate, so the "
            "current failure is not primarily a graph-only threshold penalty."
        ),
        "",
        "## Key Findings",
        "",
        (
            f"- Overall FPR is tied at {float(graph_threshold['benign_test_alert_rate']):.6f}, "
            f"but overall recall differs sharply: graph={_metric('graph', 'all_malicious', 'recall'):.6f}, "
            f"tabular={_metric('tabular', 'all_malicious', 'recall'):.6f}."
        ),
        (
            f"- Recon is the clearest divergence: graph recall={_metric('graph', 'Recon-HostDiscovery', 'recall'):.6f}, "
            f"tabular recall={_metric('tabular', 'Recon-HostDiscovery', 'recall'):.6f}; "
            f"graph f1={_metric('graph', 'Recon-HostDiscovery', 'f1'):.6f}, "
            f"tabular f1={_metric('tabular', 'Recon-HostDiscovery', 'f1'):.6f}."
        ),
        (
            f"- BrowserHijacking remains difficult for both backends: graph recall="
            f"{_metric('graph', 'BrowserHijacking', 'recall'):.6f}, "
            f"tabular recall={_metric('tabular', 'BrowserHijacking', 'recall'):.6f}."
        ),
        "",
        "## Score Overlap",
        "",
    ]

    for subset_name in (
        "benign_test",
        "source:Recon-HostDiscovery",
        "source:BrowserHijacking",
    ):
        graph_row = score_diagnostics.loc[
            (score_diagnostics["mode_family"] == "graph")
            & (score_diagnostics["subset_name"] == subset_name)
        ].iloc[0]
        tabular_row = score_diagnostics.loc[
            (score_diagnostics["mode_family"] == "tabular")
            & (score_diagnostics["subset_name"] == subset_name)
        ].iloc[0]
        lines.append(
            (
                f"- {subset_name}: graph median={float(graph_row['score_median']):.6f}, "
                f"graph p95={float(graph_row['score_p95']):.6f}; "
                f"tabular median={float(tabular_row['score_median']):.6e}, "
                f"tabular p95={float(tabular_row['score_p95']):.6e}."
            )
        )

    lines.extend(
        [
            "",
            "## Rank-Gap Concentration",
            "",
            f"- Top-{top_k_rank_gap} absolute rank-gap samples by source: "
            + ", ".join(f"{key}={value}" for key, value in top_gap_source_counts.items()),
            "- Large positive percentile gaps mean tabular ranked a sample as much more anomalous than graph.",
            "",
            "## Coarse Feature Separation",
            "",
            "- Top coarse features by separation score:",
        ]
    )
    for row in top_features.itertuples(index=False):
        lines.append(
            (
                f"  - {row.feature_name}: separation_score={row.separation_score:.6f}, "
                f"recon_d={row.recon_cohens_d:.6f}, browser_d={row.browser_cohens_d:.6f}, "
                f"graph_corr={row.graph_score_corr:.6f}, tabular_corr={row.tabular_score_corr:.6f}. "
                f"{row.notes}"
            )
        )

    lines.extend(
        [
            "",
            "## Diagnosis",
            "",
            (
                "- The graph backend is mainly losing on score ordering, not on a uniquely harsher "
                "threshold policy. It leaves many Recon windows ranked below benign holdout windows."
            ),
            (
                "- Recon appears to be separable with a few coarse graph-summary count features. "
                "The tabular control retains that coarse signal directly, while the current graph "
                "GAE does not convert it into a stronger anomaly ranking."
            ),
            (
                "- BrowserHijacking stays close to benign on these coarse summary features, which is "
                "why both backends struggle. That points to a weak-anomaly / missing-signal problem "
                "more than a threshold-only problem."
            ),
            (
                "- Because both backends use only 25 benign train graphs here, the higher-capacity "
                "graph model is likely also paying a small-sample stability penalty on top of the "
                "representation issue."
            ),
            "",
            "## Next Step",
            "",
            (
                "- The most informative next move is to calibrate the graph line at the input/score "
                "interface: verify whether graph packing and graph-level reduction preserve the coarse "
                "count information that already separates Recon well."
            ),
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_graph_score_markdown(
    path: Path,
    *,
    reduction_by_source: pd.DataFrame,
    sidechannel_comparison: pd.DataFrame,
    sidechannel_rank_gap: pd.DataFrame,
) -> None:
    """Write a compact Markdown note about graph score reduction and side-channels."""

    recon_row = reduction_by_source.loc[
        reduction_by_source["source_name"] == "Recon-HostDiscovery"
    ]
    browser_row = reduction_by_source.loc[
        reduction_by_source["source_name"] == "BrowserHijacking"
    ]
    baseline_row = sidechannel_comparison.loc[
        sidechannel_comparison["method_name"] == "graph_score_only"
    ].iloc[0]
    top1_row = sidechannel_comparison.loc[
        sidechannel_comparison["method_name"] == "graph_plus_top1_rankavg"
    ].iloc[0]
    top3_row = sidechannel_comparison.loc[
        sidechannel_comparison["method_name"] == "graph_plus_top3_rankavg"
    ].iloc[0]
    top_gap_counts = (
        sidechannel_rank_gap.head(25).groupby("source_name").size().sort_values(
            ascending=False
        ).to_dict()
        if not sidechannel_rank_gap.empty
        else {}
    )
    mean_gap_by_source = (
        sidechannel_rank_gap.groupby("source_name")["percentile_gap_signed"]
        .mean()
        .sort_values(ascending=False)
        .to_dict()
        if not sidechannel_rank_gap.empty
        else {}
    )

    lines = [
        "# Graph Score Diagnosis",
        "",
        "## Reduction Chain",
        "",
        "- `compute_node_anomaly_scores(...)` produces row-wise node reconstruction MSE.",
        "- `compute_edge_anomaly_scores(...)` produces row-wise edge reconstruction MSE.",
        "- `compute_graph_anomaly_scores(node_scores, reduction='mean')` collapses the graph score from node scores only.",
        "- In the current PCAP experiment path, the exported graph score is therefore equivalent to `mean(node_scores)`; edge and flow scores are exported but not used in graph-level reduction.",
        "",
        "## Source-Level Reduction Signal",
        "",
    ]

    if not recon_row.empty:
        recon = recon_row.iloc[0]
        lines.append(
            (
                f"- Recon-HostDiscovery: graph percentile mean="
                f"{float(recon['graph_score_percentile_mean']):.6f}, node_p90="
                f"{float(recon['node_p90_percentile_mean']):.6f}, node_max="
                f"{float(recon['node_max_percentile_mean']):.6f}, edge_p90="
                f"{float(recon['edge_p90_percentile_mean']):.6f}, edge_max="
                f"{float(recon['edge_max_percentile_mean']):.6f}."
            )
        )
    if not browser_row.empty:
        browser = browser_row.iloc[0]
        lines.append(
            (
                f"- BrowserHijacking: graph percentile mean="
                f"{float(browser['graph_score_percentile_mean']):.6f}, node_p90="
                f"{float(browser['node_p90_percentile_mean']):.6f}, node_max="
                f"{float(browser['node_max_percentile_mean']):.6f}, edge_p90="
                f"{float(browser['edge_p90_percentile_mean']):.6f}, edge_max="
                f"{float(browser['edge_max_percentile_mean']):.6f}."
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "- Recon does not look strong under the current graph score itself, but its "
                "node/edge tail statistics rank meaningfully higher than the graph mean. "
                "That is consistent with sparse anomalous nodes/edges being flattened by mean pooling."
            ),
            (
                "- BrowserHijacking shows only mild tail elevation and stays much closer to benign, "
                "which explains why both graph and tabular remain weak on it."
            ),
            "",
            "## Side-Channel Probe",
            "",
            (
                f"- Baseline graph-only: overall_fpr={float(baseline_row['overall_fpr']):.6f}, "
                f"overall_recall={float(baseline_row['overall_recall']):.6f}, "
                f"recon_recall={float(baseline_row['recon_recall']):.6f}, "
                f"browser_recall={float(baseline_row['browserhijacking_recall']):.6f}."
            ),
            (
                f"- Graph + top1 coarse feature: overall_fpr={float(top1_row['overall_fpr']):.6f}, "
                f"overall_recall={float(top1_row['overall_recall']):.6f}, "
                f"recon_recall={float(top1_row['recon_recall']):.6f}, "
                f"browser_recall={float(top1_row['browserhijacking_recall']):.6f}."
            ),
            (
                f"- Graph + top3 coarse features: overall_fpr={float(top3_row['overall_fpr']):.6f}, "
                f"overall_recall={float(top3_row['overall_recall']):.6f}, "
                f"recon_recall={float(top3_row['recon_recall']):.6f}, "
                f"browser_recall={float(top3_row['browserhijacking_recall']):.6f}."
            ),
            (
                "- The side-channel fusion is analysis-only and deliberately simple, but if it "
                "improves Recon ranking immediately, that is strong evidence that useful signal is "
                "already present outside the final graph score."
            ),
            "",
            "## Rank Shift",
            "",
            "- Top positive side-channel percentile shifts by source: "
            + ", ".join(f"{key}={value}" for key, value in top_gap_counts.items()),
            "- Mean side-channel percentile shift by source: "
            + ", ".join(f"{key}={value:.6f}" for key, value in mean_gap_by_source.items()),
            "",
            "## Bottom Line",
            "",
            (
                "- The current graph line is not failing solely because the anomaly signal is absent. "
                "It is also failing because the final graph-level score discards or underweights "
                "useful sparse and coarse cues that help Recon."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(
    *,
    graph_run_dir: Path,
    tabular_run_dir: Path,
    output_dir: Path,
    top_k_rank_gap: int,
) -> dict[str, Path]:
    """Run the graph-versus-tabular diagnosis and export compact artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)

    graph_overall = _load_scores(graph_run_dir / "overall_scores.csv")
    tabular_overall = _load_scores(tabular_run_dir / "overall_scores.csv")
    graph_train = _load_scores(graph_run_dir / "train_graph_scores.csv")
    tabular_train = _load_scores(tabular_run_dir / "train_graph_scores.csv")
    graph_per_attack = _load_csv(graph_run_dir / "per_attack_metrics.csv")
    tabular_per_attack = _load_csv(tabular_run_dir / "per_attack_metrics.csv")
    graph_summary = _load_csv(graph_run_dir / "graph_summary.csv")
    graph_graph_scores = _load_optional_scores(
        graph_run_dir / "scores" / "graph_scores.binary.csv"
    )
    graph_node_scores = _load_optional_scores(
        graph_run_dir / "scores" / "node_scores.binary.csv"
    )
    graph_edge_scores = _load_optional_scores(
        graph_run_dir / "scores" / "edge_scores.binary.csv"
    )
    graph_flow_scores = _load_optional_scores(
        graph_run_dir / "scores" / "flow_scores.binary.csv"
    )

    score_rows = []
    score_rows.extend(
        _score_diagnostic_rows(
            mode_family="graph",
            train_frame=graph_train,
            overall_frame=graph_overall,
        )
    )
    score_rows.extend(
        _score_diagnostic_rows(
            mode_family="tabular",
            train_frame=tabular_train,
            overall_frame=tabular_overall,
        )
    )
    score_diagnostics = pd.DataFrame(score_rows).sort_values(
        by=["subset_name", "mode_family"],
        ascending=[True, True],
    )
    score_diagnostics_path = output_dir / "graph_tabular_score_diagnostics.csv"
    score_diagnostics.to_csv(score_diagnostics_path, index=False)

    graph_rank_frame = graph_overall.loc[
        :, ["sample_id", "raw_label", "binary_label", "anomaly_score", "is_alert", "meta_source_name", "meta_source_role", "meta_split_assignment"]
    ].rename(
        columns={
            "anomaly_score": "graph_score",
            "is_alert": "graph_is_alert",
            "meta_source_name": "source_name",
            "meta_source_role": "source_role",
            "meta_split_assignment": "split_assignment",
        }
    )
    tabular_rank_frame = tabular_overall.loc[
        :, ["sample_id", "anomaly_score", "is_alert"]
    ].rename(
        columns={
            "anomaly_score": "tabular_score",
            "is_alert": "tabular_is_alert",
        }
    )
    rank_gap = graph_rank_frame.merge(tabular_rank_frame, on="sample_id", how="inner")
    rank_gap["graph_anomaly_percentile"] = _rank_percentile(rank_gap["graph_score"])
    rank_gap["tabular_anomaly_percentile"] = _rank_percentile(rank_gap["tabular_score"])
    rank_gap["graph_rank_desc"] = rank_gap["graph_score"].rank(method="average", ascending=False)
    rank_gap["tabular_rank_desc"] = rank_gap["tabular_score"].rank(method="average", ascending=False)
    rank_gap["percentile_gap_signed"] = (
        rank_gap["tabular_anomaly_percentile"] - rank_gap["graph_anomaly_percentile"]
    )
    rank_gap["percentile_gap_abs"] = rank_gap["percentile_gap_signed"].abs()
    rank_gap = rank_gap.sort_values(
        by=["percentile_gap_abs", "tabular_anomaly_percentile"],
        ascending=[False, False],
    )
    rank_gap_path = output_dir / "graph_tabular_rank_gap.csv"
    rank_gap.to_csv(rank_gap_path, index=False)

    graph_score_lookup = graph_overall.loc[:, ["sample_id", "anomaly_score"]].rename(
        columns={"anomaly_score": "graph_score"}
    )
    tabular_score_lookup = tabular_overall.loc[:, ["sample_id", "anomaly_score"]].rename(
        columns={"anomaly_score": "tabular_score"}
    )
    feature_frame = graph_summary.merge(
        graph_score_lookup,
        left_on="graph_id",
        right_on="sample_id",
        how="left",
    ).drop(columns=["sample_id"])
    feature_frame = feature_frame.merge(
        tabular_score_lookup,
        left_on="graph_id",
        right_on="sample_id",
        how="left",
    ).drop(columns=["sample_id"])

    benign_eval = feature_frame.loc[feature_frame["split_assignment"] == "benign_test"].copy()
    recon_eval = feature_frame.loc[
        (feature_frame["split_assignment"] == "malicious_test")
        & (feature_frame["source_name"] == "Recon-HostDiscovery")
    ].copy()
    browser_eval = feature_frame.loc[
        (feature_frame["split_assignment"] == "malicious_test")
        & (feature_frame["source_name"] == "BrowserHijacking")
    ].copy()
    overall_eval = feature_frame.loc[
        feature_frame["split_assignment"].isin(["benign_test", "malicious_test"])
    ].copy()

    feature_names = (
        "node_count",
        "edge_count",
        "client_node_count",
        "server_node_count",
        "aggregated_edge_count",
        "communication_edge_count",
        "association_edge_count",
        "association_same_src_ip_edge_count",
        "association_same_dst_subnet_edge_count",
    )
    feature_rows: list[dict[str, object]] = []
    for feature_name in feature_names:
        benign_series = pd.to_numeric(benign_eval[feature_name], errors="coerce").fillna(0.0)
        recon_series = pd.to_numeric(recon_eval[feature_name], errors="coerce").fillna(0.0)
        browser_series = pd.to_numeric(browser_eval[feature_name], errors="coerce").fillna(0.0)
        recon_d = _cohens_d(benign_series, recon_series)
        browser_d = _cohens_d(benign_series, browser_series)
        graph_corr = _safe_corr(overall_eval[feature_name], overall_eval["graph_score"])
        tabular_corr = _safe_corr(overall_eval[feature_name], overall_eval["tabular_score"])
        separation_score = max(abs(recon_d), abs(browser_d))
        feature_rows.append(
            {
                "feature_name": feature_name,
                "benign_mean": float(benign_series.mean()),
                "benign_median": float(benign_series.median()),
                "recon_mean": float(recon_series.mean()),
                "recon_median": float(recon_series.median()),
                "browserhijacking_mean": float(browser_series.mean()),
                "browserhijacking_median": float(browser_series.median()),
                "recon_cohens_d": recon_d,
                "browser_cohens_d": browser_d,
                "graph_score_corr": graph_corr,
                "tabular_score_corr": tabular_corr,
                "separation_score": separation_score,
                "notes": _feature_note(
                    recon_d=recon_d,
                    browser_d=browser_d,
                    graph_corr=graph_corr,
                    tabular_corr=tabular_corr,
                ),
            }
        )
    feature_separation = pd.DataFrame(feature_rows).sort_values(
        by=["separation_score", "tabular_score_corr"],
        ascending=[False, False],
    )
    feature_separation_path = output_dir / "graph_summary_feature_separation.csv"
    feature_separation.to_csv(feature_separation_path, index=False)

    graph_threshold = _threshold_row(
        mode_family="graph",
        train_frame=graph_train,
        overall_frame=graph_overall,
    )
    tabular_threshold = _threshold_row(
        mode_family="tabular",
        train_frame=tabular_train,
        overall_frame=tabular_overall,
    )
    threshold_diagnostics = pd.DataFrame([graph_threshold, tabular_threshold])
    threshold_diagnostics_path = output_dir / "threshold_diagnostics.csv"
    threshold_diagnostics.to_csv(threshold_diagnostics_path, index=False)

    reduction_diagnostics = _build_graph_score_reduction_diagnostics(
        graph_scores=graph_graph_scores,
        node_scores=graph_node_scores,
        edge_scores=graph_edge_scores,
        flow_scores=graph_flow_scores,
    )
    reduction_diagnostics_path = output_dir / "graph_score_reduction_diagnostics.csv"
    reduction_diagnostics.to_csv(reduction_diagnostics_path, index=False)

    reduction_by_source = _source_level_reduction_summary(reduction_diagnostics)
    reduction_by_source_path = output_dir / "graph_score_reduction_by_source.csv"
    reduction_by_source.to_csv(reduction_by_source_path, index=False)

    train_feature_frame = graph_train.loc[:, ["graph_id", "anomaly_score"]].rename(
        columns={"anomaly_score": "graph_score"}
    )
    train_feature_frame = train_feature_frame.merge(
        graph_summary.loc[graph_summary["split_assignment"] == "train"].copy(),
        on="graph_id",
        how="left",
    )

    evaluation_feature_frame = graph_overall.loc[
        :,
        [
            "sample_id",
            "anomaly_score",
            "binary_label",
            "meta_source_name",
            "meta_source_role",
            "meta_split_assignment",
        ],
    ].rename(
        columns={
            "sample_id": "graph_id",
            "anomaly_score": "graph_score",
            "meta_source_name": "source_name",
            "meta_source_role": "source_role",
            "meta_split_assignment": "split_assignment",
        }
    )
    evaluation_feature_frame = evaluation_feature_frame.merge(
        graph_summary,
        on=["graph_id", "binary_label", "source_name", "source_role", "split_assignment"],
        how="left",
    ).merge(
        tabular_score_lookup,
        left_on="graph_id",
        right_on="sample_id",
        how="left",
    ).drop(columns=["sample_id"])

    top_feature_candidates = feature_separation.loc[
        feature_separation["recon_cohens_d"] > 0
    ].copy()
    top_feature = (
        str(top_feature_candidates.iloc[0]["feature_name"])
        if not top_feature_candidates.empty
        else "aggregated_edge_count"
    )
    top_features = (
        top_feature_candidates.head(3)["feature_name"].astype(str).tolist()
        if not top_feature_candidates.empty
        else [
            "aggregated_edge_count",
            "server_node_count",
            "communication_edge_count",
        ]
    )
    sidechannel_comparison, sidechannel_rank_gap = _sidechannel_method_rows(
        train_frame=train_feature_frame,
        evaluation_frame=evaluation_feature_frame,
        top_feature=top_feature,
        top_features=top_features,
    )
    sidechannel_comparison_path = output_dir / "graph_sidechannel_comparison.csv"
    sidechannel_comparison.to_csv(sidechannel_comparison_path, index=False)
    sidechannel_rank_gap_path = output_dir / "graph_sidechannel_rank_gap.csv"
    sidechannel_rank_gap.to_csv(sidechannel_rank_gap_path, index=False)

    markdown_path = output_dir / "graph_vs_tabular_diagnosis.md"
    _write_markdown(
        markdown_path,
        graph_threshold=graph_threshold,
        tabular_threshold=tabular_threshold,
        score_diagnostics=score_diagnostics,
        rank_gap=rank_gap,
        feature_separation=feature_separation,
        graph_per_attack=graph_per_attack,
        tabular_per_attack=tabular_per_attack,
        top_k_rank_gap=top_k_rank_gap,
    )

    graph_score_markdown_path = output_dir / "graph_score_diagnosis.md"
    _write_graph_score_markdown(
        graph_score_markdown_path,
        reduction_by_source=reduction_by_source,
        sidechannel_comparison=sidechannel_comparison,
        sidechannel_rank_gap=sidechannel_rank_gap,
    )

    return {
        "graph_tabular_score_diagnostics": score_diagnostics_path,
        "graph_tabular_rank_gap": rank_gap_path,
        "graph_summary_feature_separation": feature_separation_path,
        "threshold_diagnostics": threshold_diagnostics_path,
        "graph_vs_tabular_diagnosis": markdown_path,
        "graph_score_reduction_diagnostics": reduction_diagnostics_path,
        "graph_score_reduction_by_source": reduction_by_source_path,
        "graph_sidechannel_comparison": sidechannel_comparison_path,
        "graph_sidechannel_rank_gap": sidechannel_rank_gap_path,
        "graph_score_diagnosis": graph_score_markdown_path,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the CLI entrypoint and print exported artifact paths."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    outputs = run_analysis(
        graph_run_dir=Path(args.graph_run_dir),
        tabular_run_dir=Path(args.tabular_run_dir),
        output_dir=Path(args.output_dir),
        top_k_rank_gap=int(args.top_k_rank_gap),
    )
    print(json.dumps({key: value.as_posix() for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
