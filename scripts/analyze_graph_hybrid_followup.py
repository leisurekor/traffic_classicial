"""Compare thin graph-score hybrids on one representative real-PCAP graph run.

This helper stays evaluation-only: it does not retrain the graph model and it
does not modify the primary experiment pipeline. It reads one exported graph
run, recomputes several graph-level score definitions from the saved train and
evaluation score tables, and compares them under the same benign-train q95
threshold rule. An optional tabular control run can be loaded as an external
reference row.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from traffic_graph.pipeline.compare_binary_detection_runs import (
    load_binary_detection_run_summary,
)
from traffic_graph.pipeline.metrics import BinaryScoreMetrics, evaluate_scores


FOLLOWUP_COLUMNS: tuple[str, ...] = (
    "method_name",
    "score_definition",
    "threshold_policy",
    "train_reference_count",
    "threshold",
    "benign_test_count",
    "malicious_test_count",
    "overall_fpr",
    "overall_recall",
    "overall_f1",
    "recon_recall",
    "recon_f1",
    "browserhijacking_recall",
    "browserhijacking_f1",
    "ddos_recall",
    "ddos_f1",
    "worst_malicious_source_name",
    "benign_test_mean_score",
    "benign_test_median_score",
    "benign_threshold_margin",
    "notes",
)


RANK_SHIFT_COLUMNS: tuple[str, ...] = (
    "graph_id",
    "source_name",
    "source_role",
    "split_assignment",
    "binary_label",
    "flow_p90_eval_percentile",
    "hybrid_max_rank_flow_node_max_eval_percentile",
    "hybrid_rank_avg_flow_aggedge_eval_percentile",
    "hybrid_max_rank_shift_vs_flow_p90",
    "hybrid_aggedge_rank_shift_vs_flow_p90",
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the hybrid follow-up analysis."""

    parser = argparse.ArgumentParser(
        description=(
            "Read one representative graph run and compare thin graph-score "
            "hybrids under the same benign-train q95 threshold rule."
        )
    )
    parser.add_argument(
        "--graph-run-dir",
        required=True,
        help="Directory of the representative graph binary-evaluation run.",
    )
    parser.add_argument(
        "--tabular-run-dir",
        help="Optional graph-summary PCA control run used as an external reference row.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where follow-up comparison artifacts will be written.",
    )
    return parser


def _require_file(path: Path) -> Path:
    """Return a file path when it exists and fail clearly otherwise."""

    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _load_csv(path: Path) -> pd.DataFrame:
    """Load one CSV file with stable existence validation."""

    return pd.read_csv(_require_file(path))


def _false_positive_rate(metrics: BinaryScoreMetrics) -> float:
    """Return false-positive rate with a stable empty fallback."""

    if metrics.negative_count <= 0:
        return 0.0
    return float(metrics.false_positive) / float(metrics.negative_count)


def _q95(values: pd.Series | np.ndarray | list[float]) -> float:
    """Return the q95 of one score vector."""

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0
    return float(np.quantile(array, 0.95))


def _p90(values: pd.Series | np.ndarray | list[float]) -> float:
    """Return the p90 of one score vector."""

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0
    return float(np.quantile(array, 0.90))


def _percentile_against_reference(
    values: pd.Series | np.ndarray | list[float],
    reference_values: pd.Series | np.ndarray | list[float],
) -> np.ndarray:
    """Map values onto an empirical percentile scale defined by reference values."""

    candidates = np.asarray(values, dtype=float)
    reference = np.asarray(reference_values, dtype=float)
    if reference.size == 0:
        return np.zeros_like(candidates, dtype=float)
    sorted_reference = np.sort(reference)
    ranks = np.searchsorted(sorted_reference, candidates, side="right")
    return ranks.astype(float) / float(sorted_reference.size)


def _eval_rank_percentiles(values: pd.Series | np.ndarray) -> np.ndarray:
    """Return descending anomaly percentiles within one evaluation vector."""

    series = pd.Series(np.asarray(values, dtype=float))
    if series.empty:
        return np.zeros((0,), dtype=float)
    descending_rank = series.rank(method="average", ascending=False)
    if len(series) == 1:
        return np.asarray([1.0], dtype=float)
    return (1.0 - ((descending_rank - 1.0) / float(len(series) - 1.0))).to_numpy(
        dtype=float
    )


def _extract_attack_metrics(
    *,
    eval_frame: pd.DataFrame,
    score_column: str,
    threshold: float,
) -> dict[str, BinaryScoreMetrics]:
    """Compute per-source binary metrics using benign test rows as negatives."""

    benign_rows = eval_frame.loc[eval_frame["source_role"].astype(str).eq("benign")].copy()
    malicious_rows = eval_frame.loc[
        eval_frame["source_role"].astype(str).eq("malicious")
    ].copy()
    metrics_by_source: dict[str, BinaryScoreMetrics] = {}
    for source_name, source_frame in malicious_rows.groupby("source_name", dropna=False):
        candidate = pd.concat([benign_rows, source_frame], ignore_index=True)
        metrics_by_source[str(source_name)] = evaluate_scores(
            candidate["binary_label"].to_numpy(dtype=int),
            candidate[score_column].to_numpy(dtype=float),
            threshold=threshold,
        )
    return metrics_by_source


def _lookup_attack_metric(
    metrics_by_source: dict[str, BinaryScoreMetrics],
    source_name: str,
) -> BinaryScoreMetrics | None:
    """Look up one source metric by exact source name."""

    return metrics_by_source.get(source_name)


def _select_worst_source(metrics_by_source: dict[str, BinaryScoreMetrics]) -> str:
    """Return the worst malicious source using the same low-F1-first rule."""

    if not metrics_by_source:
        return "unavailable"
    return min(
        metrics_by_source.items(),
        key=lambda item: (
            float("inf") if item[1].f1 is None else float(item[1].f1),
            float("inf") if item[1].pr_auc is None else float(item[1].pr_auc),
            float("inf") if item[1].recall is None else float(item[1].recall),
            item[0],
        ),
    )[0]


def _build_base_frames(graph_run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and align train/evaluation score ingredients for one graph run."""

    train_scores = _load_csv(graph_run_dir / "train_graph_scores.csv").copy()
    graph_summary = _load_csv(graph_run_dir / "graph_summary.csv").copy()
    node_scores = _load_csv(graph_run_dir / "scores" / "node_scores.binary.csv").copy()
    flow_scores = _load_csv(graph_run_dir / "scores" / "flow_scores.binary.csv").copy()

    for frame in (train_scores, graph_summary, node_scores, flow_scores):
        for column in ("binary_label", "anomaly_score", "score"):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

    train_summary = graph_summary.loc[
        graph_summary["graph_id"].isin(train_scores["graph_id"].astype(str)),
        ["graph_id", "aggregated_edge_count", "source_name", "source_role", "split_assignment"],
    ].drop_duplicates(subset=["graph_id"])
    train_frame = train_scores.merge(train_summary, on="graph_id", how="left")
    train_frame["mean_node"] = pd.to_numeric(train_frame["node_score_mean"], errors="coerce").fillna(0.0)
    train_frame["node_max"] = pd.to_numeric(train_frame["node_score_max"], errors="coerce").fillna(0.0)
    train_frame["flow_p90"] = pd.to_numeric(train_frame["flow_score_p90"], errors="coerce").fillna(0.0)
    train_frame["aggregated_edge_count"] = pd.to_numeric(
        train_frame["aggregated_edge_count"],
        errors="coerce",
    ).fillna(0.0)

    node_group = (
        node_scores.groupby("graph_id", dropna=False)["anomaly_score"]
        .agg(node_score_mean="mean", node_score_max="max")
        .reset_index()
    )
    flow_group = (
        flow_scores.groupby("graph_id", dropna=False)["anomaly_score"]
        .agg(flow_score_p90=lambda series: _p90(series))
        .reset_index()
    )
    eval_frame = graph_summary.loc[
        graph_summary["split_assignment"].astype(str).isin(("benign_test", "malicious_test"))
    ].copy()
    eval_frame = eval_frame.merge(node_group, on="graph_id", how="left")
    eval_frame = eval_frame.merge(flow_group, on="graph_id", how="left")
    eval_frame["binary_label"] = pd.to_numeric(eval_frame["binary_label"], errors="coerce").fillna(0).astype(int)
    eval_frame["aggregated_edge_count"] = pd.to_numeric(
        eval_frame["aggregated_edge_count"], errors="coerce"
    ).fillna(0.0)
    eval_frame["mean_node"] = pd.to_numeric(eval_frame["node_score_mean"], errors="coerce").fillna(0.0)
    eval_frame["node_max"] = pd.to_numeric(eval_frame["node_score_max"], errors="coerce").fillna(0.0)
    eval_frame["flow_p90"] = pd.to_numeric(eval_frame["flow_score_p90"], errors="coerce").fillna(0.0)
    return train_frame, eval_frame


def _add_method_columns(train_frame: pd.DataFrame, eval_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create candidate thin-hybrid score columns on top of base summaries."""

    train_working = train_frame.copy()
    eval_working = eval_frame.copy()

    for column in ("mean_node", "node_max", "flow_p90", "aggregated_edge_count"):
        train_working[column] = pd.to_numeric(train_working[column], errors="coerce").fillna(0.0)
        eval_working[column] = pd.to_numeric(eval_working[column], errors="coerce").fillna(0.0)

    train_flow_pct = _percentile_against_reference(
        train_working["flow_p90"], train_working["flow_p90"]
    )
    eval_flow_pct = _percentile_against_reference(
        eval_working["flow_p90"], train_working["flow_p90"]
    )
    train_node_max_pct = _percentile_against_reference(
        train_working["node_max"], train_working["node_max"]
    )
    eval_node_max_pct = _percentile_against_reference(
        eval_working["node_max"], train_working["node_max"]
    )
    train_aggedge_pct = _percentile_against_reference(
        train_working["aggregated_edge_count"],
        train_working["aggregated_edge_count"],
    )
    eval_aggedge_pct = _percentile_against_reference(
        eval_working["aggregated_edge_count"],
        train_working["aggregated_edge_count"],
    )

    train_working["hybrid_max_rank_flow_node_max"] = np.maximum(
        train_flow_pct,
        train_node_max_pct,
    )
    eval_working["hybrid_max_rank_flow_node_max"] = np.maximum(
        eval_flow_pct,
        eval_node_max_pct,
    )

    train_working["hybrid_rank_avg_flow_node_max"] = 0.5 * (
        train_flow_pct + train_node_max_pct
    )
    eval_working["hybrid_rank_avg_flow_node_max"] = 0.5 * (
        eval_flow_pct + eval_node_max_pct
    )

    train_working["hybrid_rank_avg_flow_aggedge"] = 0.5 * (
        train_flow_pct + train_aggedge_pct
    )
    eval_working["hybrid_rank_avg_flow_aggedge"] = 0.5 * (
        eval_flow_pct + eval_aggedge_pct
    )
    return train_working, eval_working


def _evaluate_method(
    *,
    method_name: str,
    score_definition: str,
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    score_column: str,
    notes: str,
) -> dict[str, object]:
    """Evaluate one graph-level score definition under q95 train thresholding."""

    train_scores = train_frame[score_column].to_numpy(dtype=float)
    eval_scores = eval_frame[score_column].to_numpy(dtype=float)
    threshold = _q95(train_scores)
    overall_metrics = evaluate_scores(
        eval_frame["binary_label"].to_numpy(dtype=int),
        eval_scores,
        threshold=threshold,
    )
    attack_metrics = _extract_attack_metrics(
        eval_frame=eval_frame,
        score_column=score_column,
        threshold=threshold,
    )
    benign_test = eval_frame.loc[
        eval_frame["source_role"].astype(str).eq("benign"),
        score_column,
    ].to_numpy(dtype=float)
    benign_mean = float(np.mean(benign_test)) if benign_test.size else 0.0
    benign_median = float(np.median(benign_test)) if benign_test.size else 0.0
    worst_source = _select_worst_source(attack_metrics)

    recon_metric = _lookup_attack_metric(attack_metrics, "Recon-HostDiscovery")
    browser_metric = _lookup_attack_metric(attack_metrics, "BrowserHijacking")
    ddos_metric = _lookup_attack_metric(attack_metrics, "DDoS-ICMP_Flood")
    return {
        "method_name": method_name,
        "score_definition": score_definition,
        "threshold_policy": "q95 benign train reference",
        "train_reference_count": int(train_scores.size),
        "threshold": float(threshold),
        "benign_test_count": int(
            eval_frame["source_role"].astype(str).eq("benign").sum()
        ),
        "malicious_test_count": int(
            eval_frame["source_role"].astype(str).eq("malicious").sum()
        ),
        "overall_fpr": _false_positive_rate(overall_metrics),
        "overall_recall": overall_metrics.recall,
        "overall_f1": overall_metrics.f1,
        "recon_recall": None if recon_metric is None else recon_metric.recall,
        "recon_f1": None if recon_metric is None else recon_metric.f1,
        "browserhijacking_recall": None
        if browser_metric is None
        else browser_metric.recall,
        "browserhijacking_f1": None if browser_metric is None else browser_metric.f1,
        "ddos_recall": None if ddos_metric is None else ddos_metric.recall,
        "ddos_f1": None if ddos_metric is None else ddos_metric.f1,
        "worst_malicious_source_name": worst_source,
        "benign_test_mean_score": benign_mean,
        "benign_test_median_score": benign_median,
        "benign_threshold_margin": float(threshold - benign_mean),
        "notes": notes,
    }


def _tabular_reference_row(tabular_run_dir: Path) -> dict[str, object]:
    """Load one tabular control run into the follow-up table schema."""

    summary = load_binary_detection_run_summary(tabular_run_dir, backend_name="tabular")
    attack_lookup = {metric.task_name: metric for metric in summary.per_attack_metrics}
    overall_scores_path = Path(summary.overall_scores_path) if summary.overall_scores_path else None
    benign_test_count = 0
    malicious_test_count = 0
    if overall_scores_path is not None and overall_scores_path.exists():
        overall_frame = _load_csv(overall_scores_path)
        if "binary_label" in overall_frame.columns:
            labels = pd.to_numeric(overall_frame["binary_label"], errors="coerce").fillna(0).astype(int)
            benign_test_count = int((labels == 0).sum())
            malicious_test_count = int((labels == 1).sum())
    worst_name = "unavailable"
    attack_metrics = {
        key: value
        for key, value in attack_lookup.items()
        if key != "all_malicious"
    }
    if attack_metrics:
        worst_name = min(
            attack_metrics.items(),
            key=lambda item: (
                float("inf") if item[1].f1 is None else float(item[1].f1),
                float("inf") if item[1].pr_auc is None else float(item[1].pr_auc),
                float("inf") if item[1].recall is None else float(item[1].recall),
                item[0],
            ),
        )[0]
    recon_metric = attack_lookup.get("Recon-HostDiscovery")
    browser_metric = attack_lookup.get("BrowserHijacking")
    ddos_metric = attack_lookup.get("DDoS-ICMP_Flood")
    return {
        "method_name": "tabular_graphsummary",
        "score_definition": "graph_summary PCA reconstruction control",
        "threshold_policy": "q95 benign train reference",
        "train_reference_count": int(summary.train_score_summary.get("count", 0)),
        "threshold": float(summary.threshold),
        "benign_test_count": benign_test_count,
        "malicious_test_count": malicious_test_count,
        "overall_fpr": summary.overall_metrics.get("false_positive_rate"),
        "overall_recall": summary.overall_metrics.get("recall"),
        "overall_f1": summary.overall_metrics.get("f1"),
        "recon_recall": None if recon_metric is None else recon_metric.recall,
        "recon_f1": None if recon_metric is None else recon_metric.f1,
        "browserhijacking_recall": None
        if browser_metric is None
        else browser_metric.recall,
        "browserhijacking_f1": None if browser_metric is None else browser_metric.f1,
        "ddos_recall": None if ddos_metric is None else ddos_metric.recall,
        "ddos_f1": None if ddos_metric is None else ddos_metric.f1,
        "worst_malicious_source_name": worst_name,
        "benign_test_mean_score": summary.overall_score_summary.get("benign_mean"),
        "benign_test_median_score": summary.overall_score_summary.get("benign_median"),
        "benign_threshold_margin": None,
        "notes": "External control row loaded from the existing graph-summary PCA baseline run.",
    }


def _build_rank_shift_frame(eval_frame: pd.DataFrame) -> pd.DataFrame:
    """Build a small rank-shift table for the strongest hybrid candidates."""

    working = eval_frame[
        [
            "graph_id",
            "source_name",
            "source_role",
            "split_assignment",
            "binary_label",
            "flow_p90",
            "hybrid_max_rank_flow_node_max",
            "hybrid_rank_avg_flow_aggedge",
        ]
    ].copy()
    working["flow_p90_eval_percentile"] = _eval_rank_percentiles(working["flow_p90"])
    working["hybrid_max_rank_flow_node_max_eval_percentile"] = _eval_rank_percentiles(
        working["hybrid_max_rank_flow_node_max"]
    )
    working["hybrid_rank_avg_flow_aggedge_eval_percentile"] = _eval_rank_percentiles(
        working["hybrid_rank_avg_flow_aggedge"]
    )
    working["hybrid_max_rank_shift_vs_flow_p90"] = (
        working["hybrid_max_rank_flow_node_max_eval_percentile"]
        - working["flow_p90_eval_percentile"]
    )
    working["hybrid_aggedge_rank_shift_vs_flow_p90"] = (
        working["hybrid_rank_avg_flow_aggedge_eval_percentile"]
        - working["flow_p90_eval_percentile"]
    )
    return working.loc[:, list(RANK_SHIFT_COLUMNS)].sort_values(
        by=["hybrid_max_rank_shift_vs_flow_p90", "source_name", "graph_id"],
        ascending=[False, True, True],
    )


def _write_markdown(
    *,
    output_path: Path,
    followup_frame: pd.DataFrame,
) -> None:
    """Write a concise markdown summary for the thin-hybrid follow-up."""

    lookup = {
        str(row["method_name"]): row
        for row in followup_frame.to_dict(orient="records")
    }

    def _fmt(value: object) -> str:
        if value is None or value == "":
            return "unavailable"
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    lines = [
        "# Graph Hybrid Follow-up",
        "",
        "- All graph rows reuse one fixed representative graph run and apply the same q95 benign-train threshold rule.",
        "- `flow_p90` is the current strongest single reduction candidate.",
        "- `hybrid_max_rank_flow_node_max` tests whether node-max tail information adds signal without changing the model.",
        "- `hybrid_rank_avg_flow_aggedge` is one coarse-feature side-channel check using `aggregated_edge_count` only.",
        "",
        "## Key Rows",
        "",
    ]
    for method_name in (
        "mean_node",
        "flow_p90",
        "hybrid_max_rank_flow_node_max",
        "hybrid_rank_avg_flow_aggedge",
        "tabular_graphsummary",
    ):
        row = lookup.get(method_name)
        if row is None:
            continue
        lines.extend(
            [
                f"### {method_name}",
                "",
                f"- Overall FPR: `{_fmt(row.get('overall_fpr'))}`",
                f"- Overall Recall / F1: `{_fmt(row.get('overall_recall'))}` / `{_fmt(row.get('overall_f1'))}`",
                f"- Recon Recall / F1: `{_fmt(row.get('recon_recall'))}` / `{_fmt(row.get('recon_f1'))}`",
                f"- BrowserHijacking Recall / F1: `{_fmt(row.get('browserhijacking_recall'))}` / `{_fmt(row.get('browserhijacking_f1'))}`",
                f"- Worst malicious source: `{_fmt(row.get('worst_malicious_source_name'))}`",
                f"- Notes: {_fmt(row.get('notes'))}",
                "",
            ]
        )

    flow_row = lookup.get("flow_p90")
    hybrid_row = lookup.get("hybrid_max_rank_flow_node_max")
    aggedge_row = lookup.get("hybrid_rank_avg_flow_aggedge")
    tabular_row = lookup.get("tabular_graphsummary")
    lines.extend(
        [
            "## Diagnosis",
            "",
            (
                "- If `hybrid_max_rank_flow_node_max` keeps the same FPR as `flow_p90` while pushing Recon upward, "
                "that supports the current diagnosis that a thin tail-aware fusion is more promising than encoder changes."
            ),
            (
                "- If `hybrid_rank_avg_flow_aggedge` raises FPR or hurts BrowserHijacking, that suggests coarse-feature "
                "fusion is not yet worth promoting into the mainline."
            ),
            "",
            "## Direct Comparison",
            "",
            f"- `flow_p90` overall F1: `{_fmt(None if flow_row is None else flow_row.get('overall_f1'))}`",
            f"- best thin hybrid overall F1: `{_fmt(None if hybrid_row is None else hybrid_row.get('overall_f1'))}`",
            f"- tabular overall F1: `{_fmt(None if tabular_row is None else tabular_row.get('overall_f1'))}`",
            f"- `flow_p90` Recon F1: `{_fmt(None if flow_row is None else flow_row.get('recon_f1'))}`",
            f"- best thin hybrid Recon F1: `{_fmt(None if hybrid_row is None else hybrid_row.get('recon_f1'))}`",
            f"- tabular Recon F1: `{_fmt(None if tabular_row is None else tabular_row.get('recon_f1'))}`",
            f"- coarse-feature hybrid BrowserHijacking F1: `{_fmt(None if aggedge_row is None else aggedge_row.get('browserhijacking_f1'))}`",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def run_followup_analysis(
    *,
    graph_run_dir: Path,
    output_dir: Path,
    tabular_run_dir: Path | None = None,
) -> dict[str, str]:
    """Run the thin-hybrid follow-up analysis and export comparison artifacts."""

    train_frame, eval_frame = _build_base_frames(graph_run_dir)
    train_working, eval_working = _add_method_columns(train_frame, eval_frame)

    rows = [
        _evaluate_method(
            method_name="mean_node",
            score_definition="mean(node_scores)",
            train_frame=train_working,
            eval_frame=eval_working,
            score_column="mean_node",
            notes="Evaluation-only reference on the fixed representative graph run.",
        ),
        _evaluate_method(
            method_name="flow_p90",
            score_definition="p90(flow_scores)",
            train_frame=train_working,
            eval_frame=eval_working,
            score_column="flow_p90",
            notes="Current strongest single reduction candidate.",
        ),
        _evaluate_method(
            method_name="node_max",
            score_definition="max(node_scores)",
            train_frame=train_working,
            eval_frame=eval_working,
            score_column="node_max",
            notes="Node-only tail check used as a conservative control.",
        ),
        _evaluate_method(
            method_name="hybrid_max_rank_flow_node_max",
            score_definition="max(train-CDF(flow_p90), train-CDF(node_max))",
            train_frame=train_working,
            eval_frame=eval_working,
            score_column="hybrid_max_rank_flow_node_max",
            notes=(
                "Thin hybrid candidate: preserve flow tail signal and only add node-max "
                "when it is more extreme on the train-reference percentile scale."
            ),
        ),
        _evaluate_method(
            method_name="hybrid_rank_avg_flow_aggedge",
            score_definition="avg(train-CDF(flow_p90), train-CDF(aggregated_edge_count))",
            train_frame=train_working,
            eval_frame=eval_working,
            score_column="hybrid_rank_avg_flow_aggedge",
            notes="One-coarse-feature side-channel check using aggregated_edge_count only.",
        ),
    ]
    if tabular_run_dir is not None:
        rows.append(_tabular_reference_row(tabular_run_dir))

    followup_frame = pd.DataFrame(rows, columns=list(FOLLOWUP_COLUMNS))
    method_order = [
        "mean_node",
        "flow_p90",
        "node_max",
        "hybrid_max_rank_flow_node_max",
        "hybrid_rank_avg_flow_aggedge",
        "tabular_graphsummary",
    ]
    followup_frame["method_name"] = pd.Categorical(
        followup_frame["method_name"],
        categories=method_order,
        ordered=True,
    )
    followup_frame = followup_frame.sort_values(
        by=["method_name"],
        kind="stable",
    ).reset_index(drop=True)
    rank_shift_frame = _build_rank_shift_frame(eval_working)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "graph_hybrid_followup.csv"
    markdown_path = output_dir / "graph_hybrid_followup.md"
    rank_shift_path = output_dir / "graph_hybrid_rank_shift.csv"
    followup_frame.to_csv(csv_path, index=False)
    rank_shift_frame.to_csv(rank_shift_path, index=False)
    _write_markdown(output_path=markdown_path, followup_frame=followup_frame)
    return {
        "graph_hybrid_followup_csv": csv_path.as_posix(),
        "graph_hybrid_followup_md": markdown_path.as_posix(),
        "graph_hybrid_rank_shift_csv": rank_shift_path.as_posix(),
    }


def main() -> int:
    """Run the thin-hybrid follow-up CLI."""

    args = _build_parser().parse_args()
    artifact_paths = run_followup_analysis(
        graph_run_dir=Path(args.graph_run_dir),
        output_dir=Path(args.output_dir),
        tabular_run_dir=None if args.tabular_run_dir is None else Path(args.tabular_run_dir),
    )
    print("Saved graph hybrid follow-up artifacts:")
    for key, value in artifact_paths.items():
        print(f"  - {key}: {value}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
