"""Graph-mode ablation suite for the merged-CSV binary detection experiment.

The suite keeps the binary/held-out protocol identical to the existing
tabular and graph backends while sweeping graph-specific configuration knobs.
Each configuration is exported into its own directory, and a stable summary
table is written at the suite level for downstream comparison.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from traffic_graph.data import (
    BinaryExperimentConfig,
    HeldOutAttackProtocolConfig,
)
from traffic_graph.pipeline.binary_detection import (
    BinaryAttackMetricRecord,
    BinaryDetectionExportResult,
    BinaryDetectionReport,
)
from traffic_graph.pipeline.graph_binary_detection import (
    GraphBinaryDetectionConfig,
    run_graph_binary_detection_experiment,
)

GRAPH_ABLATION_SUMMARY_FIELDS: tuple[str, ...] = (
    "config_id",
    "window_size",
    "use_association_edges",
    "use_graph_structural_features",
    "run_id",
    "dataset_name",
    "created_at",
    "output_directory",
    "manifest_path",
    "overall_roc_auc",
    "overall_pr_auc",
    "overall_precision",
    "overall_recall",
    "overall_f1",
    "overall_false_positive_rate",
    "recon_recall",
    "recon_pr_auc",
    "recon_f1",
    "web_based_recall",
    "web_based_pr_auc",
    "web_based_f1",
    "all_malicious_recall",
    "all_malicious_pr_auc",
    "all_malicious_f1",
    "train_score_mean",
    "train_score_median",
    "train_score_q95",
    "overall_score_mean",
    "overall_score_median",
    "overall_score_q95",
    "notes",
)
"""Stable field order for graph ablation summaries."""

_DEFAULT_WINDOW_SIZES: tuple[int, ...] = (30, 60, 120, 300)
_DEFAULT_BOOL_SWEEP: tuple[bool, ...] = (False, True)
_DEFAULT_TIMESTAMP = "timestamp"


def _timestamp_token(value: object | None = None) -> str:
    """Normalize a timestamp-like value into a stable UTC token."""

    if value is None:
        moment = datetime.now(timezone.utc)
        return moment.strftime("%Y%m%dT%H%M%SZ")
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    token = "".join(
        ch if ch.isalnum() or ch in {"-", "_", "."} else "-"
        for ch in str(value).strip()
    )
    token = token.strip("-._")
    return token or _DEFAULT_TIMESTAMP


def _bool_label(value: bool) -> str:
    """Render a boolean toggle as a stable label."""

    return "on" if value else "off"


def _config_id(
    *,
    window_size: int,
    use_association_edges: bool,
    use_graph_structural_features: bool,
) -> str:
    """Build a stable configuration identifier for one ablation run."""

    return (
        f"assoc-{_bool_label(use_association_edges)}"
        f"__win-{window_size}s"
        f"__struct-{_bool_label(use_graph_structural_features)}"
    )


def _metric_lookup(
    metrics: Sequence[BinaryAttackMetricRecord],
    task_name: str,
) -> BinaryAttackMetricRecord | None:
    """Return the first metric record matching a task name."""

    for metric in metrics:
        if metric.task_name == task_name:
            return metric
    return None


@dataclass(frozen=True, slots=True)
class GraphAblationSummaryRecord:
    """One row in the graph ablation summary table."""

    config_id: str
    window_size: int
    use_association_edges: bool
    use_graph_structural_features: bool
    run_id: str
    dataset_name: str
    created_at: str
    output_directory: str
    manifest_path: str
    overall_roc_auc: float | None
    overall_pr_auc: float | None
    overall_precision: float
    overall_recall: float
    overall_f1: float
    overall_false_positive_rate: float | None
    recon_recall: float | None
    recon_pr_auc: float | None
    recon_f1: float | None
    web_based_recall: float | None
    web_based_pr_auc: float | None
    web_based_f1: float | None
    all_malicious_recall: float | None
    all_malicious_pr_auc: float | None
    all_malicious_f1: float | None
    train_score_mean: float
    train_score_median: float
    train_score_q95: float
    overall_score_mean: float
    overall_score_median: float
    overall_score_q95: float
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize the summary record into a JSON-friendly mapping."""

        return {
            "config_id": self.config_id,
            "window_size": self.window_size,
            "use_association_edges": self.use_association_edges,
            "use_graph_structural_features": self.use_graph_structural_features,
            "run_id": self.run_id,
            "dataset_name": self.dataset_name,
            "created_at": self.created_at,
            "output_directory": self.output_directory,
            "manifest_path": self.manifest_path,
            "overall_roc_auc": self.overall_roc_auc,
            "overall_pr_auc": self.overall_pr_auc,
            "overall_precision": self.overall_precision,
            "overall_recall": self.overall_recall,
            "overall_f1": self.overall_f1,
            "overall_false_positive_rate": self.overall_false_positive_rate,
            "recon_recall": self.recon_recall,
            "recon_pr_auc": self.recon_pr_auc,
            "recon_f1": self.recon_f1,
            "web_based_recall": self.web_based_recall,
            "web_based_pr_auc": self.web_based_pr_auc,
            "web_based_f1": self.web_based_f1,
            "all_malicious_recall": self.all_malicious_recall,
            "all_malicious_pr_auc": self.all_malicious_pr_auc,
            "all_malicious_f1": self.all_malicious_f1,
            "train_score_mean": self.train_score_mean,
            "train_score_median": self.train_score_median,
            "train_score_q95": self.train_score_q95,
            "overall_score_mean": self.overall_score_mean,
            "overall_score_median": self.overall_score_median,
            "overall_score_q95": self.overall_score_q95,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class GraphAblationRunResult:
    """Structured result for one ablation configuration."""

    config_id: str
    window_size: int
    use_association_edges: bool
    use_graph_structural_features: bool
    report: BinaryDetectionReport
    export_result: BinaryDetectionExportResult


@dataclass(frozen=True, slots=True)
class GraphAblationSuiteResult:
    """Structured result for an entire ablation sweep."""

    suite_id: str
    dataset_name: str
    created_at: str
    output_directory: str
    run_results: tuple[GraphAblationRunResult, ...]
    summary_records: tuple[GraphAblationSummaryRecord, ...]
    summary_csv_path: str
    summary_json_path: str
    manifest_path: str
    artifact_paths: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def render(self) -> str:
        """Render a compact human-readable summary of the ablation sweep."""

        lines = [
            f"Graph ablation suite: id={self.suite_id}, dataset={self.dataset_name}",
            f"Created at: {self.created_at}",
            f"Output directory: {self.output_directory}",
            f"Configurations evaluated: {len(self.summary_records)}",
            f"Summary CSV: {self.summary_csv_path}",
            f"Summary JSON: {self.summary_json_path}",
        ]
        if self.summary_records:
            top_recon = max(
                self.summary_records,
                key=lambda record: (
                    -1.0 if record.recon_recall is None else record.recon_recall
                ),
            )
            top_web = max(
                self.summary_records,
                key=lambda record: (
                    -1.0 if record.web_based_recall is None else record.web_based_recall
                ),
            )
            lines.append(
                "Best recon recall: "
                f"{top_recon.recon_recall if top_recon.recon_recall is not None else 'n/a'} "
                f"at {top_recon.config_id}"
            )
            lines.append(
                "Best web-based recall: "
                f"{top_web.web_based_recall if top_web.web_based_recall is not None else 'n/a'} "
                f"at {top_web.config_id}"
            )
        if self.notes:
            lines.append("Notes:")
            lines.extend(f"  - {note}" for note in self.notes)
        return "\n".join(lines)


def _metric_value(metric: BinaryAttackMetricRecord | None, field_name: str) -> float | None:
    """Return a float metric from a metric record when available."""

    if metric is None:
        return None
    value = getattr(metric, field_name, None)
    if value is None:
        return None
    return float(value)


def collect_ablation_summary_record(
    *,
    config_id: str,
    window_size: int,
    use_association_edges: bool,
    use_graph_structural_features: bool,
    report: BinaryDetectionReport,
    export_result: BinaryDetectionExportResult,
) -> GraphAblationSummaryRecord:
    """Collect a per-configuration summary from a graph binary detection report."""

    recon_metric = _metric_lookup(report.per_attack_metrics, "recon")
    web_metric = _metric_lookup(report.per_attack_metrics, "web-based")
    all_malicious_metric = _metric_lookup(report.per_attack_metrics, "all_malicious")
    return GraphAblationSummaryRecord(
        config_id=config_id,
        window_size=window_size,
        use_association_edges=use_association_edges,
        use_graph_structural_features=use_graph_structural_features,
        run_id=report.run_id,
        dataset_name=report.dataset_name,
        created_at=report.created_at,
        output_directory=export_result.output_directory,
        manifest_path=export_result.manifest_path,
        overall_roc_auc=report.overall_metrics.get("roc_auc"),
        overall_pr_auc=report.overall_metrics.get("pr_auc"),
        overall_precision=float(report.overall_metrics.get("precision") or 0.0),
        overall_recall=float(report.overall_metrics.get("recall") or 0.0),
        overall_f1=float(report.overall_metrics.get("f1") or 0.0),
        overall_false_positive_rate=report.overall_metrics.get("false_positive_rate"),
        recon_recall=_metric_value(recon_metric, "recall"),
        recon_pr_auc=_metric_value(recon_metric, "pr_auc"),
        recon_f1=_metric_value(recon_metric, "f1"),
        web_based_recall=_metric_value(web_metric, "recall"),
        web_based_pr_auc=_metric_value(web_metric, "pr_auc"),
        web_based_f1=_metric_value(web_metric, "f1"),
        all_malicious_recall=_metric_value(all_malicious_metric, "recall"),
        all_malicious_pr_auc=_metric_value(all_malicious_metric, "pr_auc"),
        all_malicious_f1=_metric_value(all_malicious_metric, "f1"),
        train_score_mean=float(report.train_score_summary.get("mean", 0.0)),
        train_score_median=float(report.train_score_summary.get("median", 0.0)),
        train_score_q95=float(report.train_score_summary.get("q95", 0.0)),
        overall_score_mean=float(report.overall_score_summary.get("mean", 0.0)),
        overall_score_median=float(report.overall_score_summary.get("median", 0.0)),
        overall_score_q95=float(report.overall_score_summary.get("q95", 0.0)),
        notes=tuple(report.notes),
    )


def _write_json(path: Path, payload: object) -> None:
    """Write a stable JSON file with a trailing newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def export_ablation_summary(
    records: Sequence[GraphAblationSummaryRecord],
    output_directory: str | Path,
    *,
    suite_id: str,
    dataset_name: str,
    created_at: str,
) -> tuple[str, str, str, dict[str, str]]:
    """Export the suite-level summary artifacts."""

    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = output_dir / "ablation_summary.csv"
    summary_json_path = output_dir / "ablation_summary.json"
    manifest_path = output_dir / "manifest.json"

    frame = pd.DataFrame(
        [record.to_dict() for record in records],
        columns=list(GRAPH_ABLATION_SUMMARY_FIELDS),
    )
    frame.to_csv(summary_csv_path, index=False)
    _write_json(
        summary_json_path,
        {
            "suite_id": suite_id,
            "dataset_name": dataset_name,
            "created_at": created_at,
            "record_count": len(records),
            "records": [record.to_dict() for record in records],
        },
    )
    artifact_paths = {
        "ablation_summary_csv": summary_csv_path.as_posix(),
        "ablation_summary_json": summary_json_path.as_posix(),
    }
    _write_json(
        manifest_path,
        {
            "suite_id": suite_id,
            "dataset_name": dataset_name,
            "created_at": created_at,
            "artifact_paths": artifact_paths,
            "record_count": len(records),
            "config_ids": [record.config_id for record in records],
        },
    )
    artifact_paths["manifest_json"] = manifest_path.as_posix()
    return (
        summary_csv_path.as_posix(),
        summary_json_path.as_posix(),
        manifest_path.as_posix(),
        artifact_paths,
    )


def summarize_graph_ablation_suite(
    suite: GraphAblationSuiteResult,
) -> str:
    """Render a compact human-readable summary for one ablation suite."""

    return suite.render()


def run_graph_ablation_suite(
    source: str | Path | pd.DataFrame,
    output_dir: str | Path,
    *,
    binary_experiment_config: BinaryExperimentConfig | None = None,
    heldout_protocol_config: HeldOutAttackProtocolConfig | None = None,
    graph_model_config: GraphBinaryDetectionConfig | None = None,
    window_sizes: Sequence[int] = _DEFAULT_WINDOW_SIZES,
    use_association_edges_options: Sequence[bool] = _DEFAULT_BOOL_SWEEP,
    use_graph_structural_features_options: Sequence[bool] = _DEFAULT_BOOL_SWEEP,
    threshold_percentile: float = 95.0,
    random_seed: int = 42,
    export_formats: Sequence[str] = ("jsonl", "csv"),
    run_experiment_fn: Callable[..., tuple[BinaryDetectionReport, BinaryDetectionExportResult]] = run_graph_binary_detection_experiment,
    timestamp: object | None = None,
) -> GraphAblationSuiteResult:
    """Run a Cartesian-product ablation sweep over graph configuration knobs."""

    suite_timestamp = _timestamp_token(timestamp)
    suite_id = f"graph-ablation:{suite_timestamp}"
    base_graph_config = graph_model_config or GraphBinaryDetectionConfig()
    dataset_name = "graph_ablation"
    run_results: list[GraphAblationRunResult] = []
    summary_records: list[GraphAblationSummaryRecord] = []
    notes: list[str] = []

    for window_size in window_sizes:
        for use_association_edges in use_association_edges_options:
            for use_graph_structural_features in use_graph_structural_features_options:
                config_id = _config_id(
                    window_size=int(window_size),
                    use_association_edges=bool(use_association_edges),
                    use_graph_structural_features=bool(use_graph_structural_features),
                )
                per_run_output_dir = Path(output_dir) / config_id
                per_run_timestamp = f"{suite_timestamp}__{config_id}"
                run_graph_config = GraphBinaryDetectionConfig(
                    hidden_dim=base_graph_config.hidden_dim,
                    latent_dim=base_graph_config.latent_dim,
                    num_layers=base_graph_config.num_layers,
                    dropout=base_graph_config.dropout,
                    learning_rate=base_graph_config.learning_rate,
                    weight_decay=base_graph_config.weight_decay,
                    batch_size=base_graph_config.batch_size,
                    max_epochs=base_graph_config.max_epochs,
                    window_size=int(window_size),
                    use_association_edges=bool(use_association_edges),
                    use_graph_structural_features=bool(use_graph_structural_features),
                )
                report, export_result = run_experiment_fn(
                    source,
                    per_run_output_dir,
                    binary_experiment_config=binary_experiment_config,
                    heldout_protocol_config=heldout_protocol_config,
                    threshold_percentile=threshold_percentile,
                    graph_config=run_graph_config,
                    random_seed=random_seed,
                    timestamp=per_run_timestamp,
                    export_formats=export_formats,
                )
                if dataset_name == "graph_ablation":
                    dataset_name = report.dataset_name
                run_results.append(
                    GraphAblationRunResult(
                        config_id=config_id,
                        window_size=int(window_size),
                        use_association_edges=bool(use_association_edges),
                        use_graph_structural_features=bool(use_graph_structural_features),
                        report=report,
                        export_result=export_result,
                    )
                )
                summary_records.append(
                    collect_ablation_summary_record(
                        config_id=config_id,
                        window_size=int(window_size),
                        use_association_edges=bool(use_association_edges),
                        use_graph_structural_features=bool(use_graph_structural_features),
                        report=report,
                        export_result=export_result,
                    )
                )

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    summary_csv_path, summary_json_path, manifest_path, artifact_paths = (
        export_ablation_summary(
            summary_records,
            output_directory,
            suite_id=suite_id,
            dataset_name=dataset_name,
            created_at=suite_timestamp,
        )
    )

    if not summary_records:
        notes.append("No ablation runs were produced.")
    elif any(record.recon_recall is None for record in summary_records):
        notes.append("Some runs did not contain recon metrics.")
    elif any(record.web_based_recall is None for record in summary_records):
        notes.append("Some runs did not contain web-based metrics.")

    return GraphAblationSuiteResult(
        suite_id=suite_id,
        dataset_name=dataset_name,
        created_at=suite_timestamp,
        output_directory=output_directory.as_posix(),
        run_results=tuple(run_results),
        summary_records=tuple(summary_records),
        summary_csv_path=summary_csv_path,
        summary_json_path=summary_json_path,
        manifest_path=manifest_path,
        artifact_paths=artifact_paths,
        notes=tuple(notes),
    )


__all__ = [
    "GRAPH_ABLATION_SUMMARY_FIELDS",
    "GraphAblationRunResult",
    "GraphAblationSuiteResult",
    "GraphAblationSummaryRecord",
    "collect_ablation_summary_record",
    "export_ablation_summary",
    "run_graph_ablation_suite",
    "summarize_graph_ablation_suite",
]
