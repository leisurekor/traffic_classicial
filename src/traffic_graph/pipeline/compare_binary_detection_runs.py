"""Filesystem helpers for comparing tabular and graph binary detection runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence

from traffic_graph.pipeline.binary_detection import BinaryAttackMetricRecord
from traffic_graph.pipeline.comparison_report import (
    BinaryDetectionComparisonExportResult,
    BinaryDetectionComparisonReport,
    ComparisonRunSummary,
    build_comparison_run_summary,
    compare_binary_detection_run_summaries,
    export_comparison_report,
    _load_metric_row_from_mapping,
    summarize_comparison,
)
from traffic_graph.pipeline.scorer_roles import normalize_run_scorer_role


@dataclass(frozen=True, slots=True)
class BinaryDetectionRunMetadata:
    """Normalized metadata shared by comparison and analysis consumers."""

    experiment_label: str
    reduction_method: str
    threshold: float | None
    benign_train_graph_count: int
    benign_test_graph_count: int
    malicious_test_graph_count: int
    worst_malicious_source_name: str
    scorer_role: str = ""
    benign_inputs: tuple[str, ...] = ()
    malicious_inputs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BinaryDetectionRunBundle:
    """Normalized run view with raw payloads and consumer-friendly metadata."""

    summary: ComparisonRunSummary
    comparison_summary: dict[str, object]
    pcap_experiment_summary: dict[str, object] | None
    per_attack_metrics_by_task: dict[str, BinaryAttackMetricRecord]
    metadata: BinaryDetectionRunMetadata


def _resolve_manifest_path(path: str | Path) -> Path:
    """Resolve a run directory or manifest path into a concrete manifest file."""

    candidate = Path(path)
    if candidate.is_file():
        return candidate
    manifest_path = candidate / "manifest.json"
    if manifest_path.exists():
        return manifest_path
    manifests = sorted(candidate.rglob("manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No manifest.json found under {candidate.as_posix()}")
    if len(manifests) > 1:
        raise ValueError(
            "Multiple manifest.json files were found. Pass a specific run directory or manifest path."
        )
    return manifests[0]


def _resolve_artifact_path(raw_path: str | None, *, manifest_path: Path) -> Path | None:
    """Resolve a recorded artifact path relative to a manifest."""

    if raw_path is None or str(raw_path).strip() == "":
        return None
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate
    relative_to_manifest = manifest_path.parent / candidate
    if relative_to_manifest.exists():
        return relative_to_manifest
    return candidate


def _read_json_mapping(path: Path) -> dict[str, object]:
    """Read a JSON file and validate that the root payload is a mapping."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a JSON object in {path.as_posix()}.")
    return dict(payload)


def _read_optional_json_mapping(path: Path) -> dict[str, object] | None:
    """Read one optional JSON mapping when it exists."""

    if not path.exists():
        return None
    return _read_json_mapping(path)


def _normalize_report_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Normalize run metrics payloads into the shared comparison schema."""

    if "overall_metrics" in payload:
        return dict(payload)

    pcap_section = payload.get("pcap_experiment")
    if not isinstance(pcap_section, Mapping):
        return dict(payload)

    normalized = dict(pcap_section)
    normalized.setdefault("dataset_name", "pcap_experiment")
    normalized.setdefault("created_at", pcap_section.get("timestamp"))
    normalized.setdefault("threshold", pcap_section.get("anomaly_threshold"))
    normalized.setdefault(
        "train_score_summary",
        pcap_section.get("train_graph_score_summary", {}),
    )
    normalized.setdefault(
        "overall_score_summary",
        pcap_section.get("graph_score_summary", {}),
    )
    if "input_artifacts" not in normalized:
        normalized["input_artifacts"] = {
            "experiment_mode": pcap_section.get("mode"),
            "graph_score_reduction": pcap_section.get("graph_score_reduction"),
            "packet_limit": pcap_section.get("packet_limit"),
            "window_size": pcap_section.get("window_size"),
        }

    benign_inputs = pcap_section.get("benign_inputs")
    malicious_inputs = pcap_section.get("malicious_inputs")
    source_parts: list[str] = []
    if isinstance(benign_inputs, Sequence) and not isinstance(
        benign_inputs, (str, bytes, bytearray)
    ):
        source_parts.extend(str(value) for value in benign_inputs if str(value).strip())
    if isinstance(malicious_inputs, Sequence) and not isinstance(
        malicious_inputs, (str, bytes, bytearray)
    ):
        source_parts.extend(str(value) for value in malicious_inputs if str(value).strip())
    if source_parts:
        normalized.setdefault("source_path", " | ".join(source_parts))

    return normalized


def _sequence_to_strings(value: object | None) -> tuple[str, ...]:
    """Normalize a sequence-like payload into a tuple of non-empty strings."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(str(item) for item in value if str(item).strip())


def _load_per_attack_metrics_path(
    *,
    report_payload: Mapping[str, object],
    per_attack_metrics_path: Path,
) -> tuple[BinaryAttackMetricRecord, ...]:
    """Load per-attack metric rows from the CSV file or embedded report payload."""

    if per_attack_metrics_path.exists():
        rows: list[BinaryAttackMetricRecord] = []
        import csv

        with per_attack_metrics_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                attack_labels_raw = str(row.get("attack_labels") or "")
                attack_labels = tuple(
                    token for token in (part.strip() for part in attack_labels_raw.split("|")) if token
                )
                notes_raw = row.get("notes")
                parsed_notes: tuple[str, ...]
                if notes_raw is None:
                    parsed_notes = ()
                else:
                    try:
                        decoded = json.loads(str(notes_raw))
                    except json.JSONDecodeError:
                        decoded = [str(notes_raw)]
                    if isinstance(decoded, Sequence) and not isinstance(
                        decoded, (str, bytes, bytearray)
                    ):
                        parsed_notes = tuple(str(item) for item in decoded if str(item).strip())
                    else:
                        parsed_notes = (str(decoded),)
                rows.append(
                    BinaryAttackMetricRecord(
                        task_name=str(row.get("task_name", "")),
                        requested_attack_type=str(row.get("requested_attack_type", "")),
                        attack_labels=attack_labels,
                        sample_count=int(float(row.get("sample_count") or 0)),
                        benign_count=int(float(row.get("benign_count") or 0)),
                        attack_count=int(float(row.get("attack_count") or 0)),
                        roc_auc=(
                            float(row["roc_auc"])
                            if row.get("roc_auc") not in {None, "", "None"}
                            else None
                        ),
                        pr_auc=(
                            float(row["pr_auc"])
                            if row.get("pr_auc") not in {None, "", "None"}
                            else None
                        ),
                        precision=float(row.get("precision") or 0.0),
                        recall=float(row.get("recall") or 0.0),
                        f1=float(row.get("f1") or 0.0),
                        false_positive_rate=(
                            float(row["false_positive_rate"])
                            if row.get("false_positive_rate") not in {None, "", "None"}
                            else None
                        ),
                        threshold=float(row.get("threshold") or 0.0),
                        score_min=float(row.get("score_min") or 0.0),
                        score_q25=float(row.get("score_q25") or 0.0),
                        score_median=float(row.get("score_median") or 0.0),
                        score_q75=float(row.get("score_q75") or 0.0),
                        score_q95=float(row.get("score_q95") or 0.0),
                        score_max=float(row.get("score_max") or 0.0),
                        score_mean=float(row.get("score_mean") or 0.0),
                        score_std=float(row.get("score_std") or 0.0),
                        benign_score_mean=float(row.get("benign_score_mean") or 0.0),
                        benign_score_median=float(row.get("benign_score_median") or 0.0),
                        attack_score_mean=float(row.get("attack_score_mean") or 0.0),
                        attack_score_median=float(row.get("attack_score_median") or 0.0),
                        notes=parsed_notes,
                    )
                )
        return tuple(rows)

    payload_metrics = report_payload.get("per_attack_metrics", ())
    if isinstance(payload_metrics, Sequence) and not isinstance(
        payload_metrics, (str, bytes, bytearray)
    ):
        rows = tuple(
            _load_metric_row_from_mapping(item)
            for item in payload_metrics
            if isinstance(item, Mapping)
        )
        return rows
    return ()


def _per_attack_metric_lookup(
    rows: Sequence[BinaryAttackMetricRecord],
) -> dict[str, BinaryAttackMetricRecord]:
    """Build a task-name keyed lookup over per-attack metric rows."""

    return {
        str(row.task_name): row
        for row in rows
        if str(row.task_name).strip()
    }


def load_binary_detection_run_summary(
    run_dir_or_manifest: str | Path,
    *,
    backend_name: str,
) -> ComparisonRunSummary:
    """Load one binary detection experiment directory into a comparable summary."""

    manifest_path = _resolve_manifest_path(run_dir_or_manifest)
    manifest_payload = _read_json_mapping(manifest_path)
    artifact_paths_raw = manifest_payload.get("artifact_paths", {})
    artifact_paths = (
        {str(key): str(value) for key, value in artifact_paths_raw.items()}
        if isinstance(artifact_paths_raw, Mapping)
        else {}
    )
    row_counts_raw = manifest_payload.get("row_counts", {})
    row_counts = (
        {str(key): int(value) for key, value in row_counts_raw.items()}
        if isinstance(row_counts_raw, Mapping)
        else {}
    )
    metrics_summary_raw = artifact_paths.get("metrics_summary_json")
    metrics_summary_path = _resolve_artifact_path(metrics_summary_raw, manifest_path=manifest_path)
    if metrics_summary_path is None:
        metrics_summary_path = manifest_path.parent / "metrics_summary.json"
    if not metrics_summary_path.exists():
        raise FileNotFoundError(
            f"Metrics summary file not found for comparison: {metrics_summary_path.as_posix()}"
        )

    report_payload = _normalize_report_payload(_read_json_mapping(metrics_summary_path))
    per_attack_metrics_raw = artifact_paths.get("per_attack_metrics_csv")
    per_attack_metrics_path = _resolve_artifact_path(
        per_attack_metrics_raw,
        manifest_path=manifest_path,
    )
    if per_attack_metrics_path is None:
        per_attack_metrics_path = manifest_path.parent / "per_attack_metrics.csv"

    per_attack_metrics = _load_per_attack_metrics_path(
        report_payload=report_payload,
        per_attack_metrics_path=per_attack_metrics_path,
    )

    report = build_comparison_run_summary(
        backend_name=backend_name,
        run_directory=manifest_payload.get("comparison_directory")
        or manifest_payload.get("layout_directory")
        or manifest_path.parent.as_posix(),
        manifest_path=manifest_path,
        metrics_summary_path=metrics_summary_path,
        report_payload=report_payload,
        artifact_paths=artifact_paths,
        row_counts=row_counts,
        per_attack_metrics_path=per_attack_metrics_path,
        overall_scores_path=_resolve_artifact_path(
            artifact_paths.get("overall_scores_csv")
            or artifact_paths.get("overall_scores_jsonl")
            or artifact_paths.get("overall_scores_parquet"),
            manifest_path=manifest_path,
        ),
        attack_scores_path=_resolve_artifact_path(
            artifact_paths.get("attack_scores_csv")
            or artifact_paths.get("attack_scores_jsonl")
            or artifact_paths.get("attack_scores_parquet"),
            manifest_path=manifest_path,
        ),
        per_attack_metrics=per_attack_metrics,
    )
    return report


def load_binary_detection_run_bundle(
    run_dir_or_manifest: str | Path,
    *,
    backend_name: str,
) -> BinaryDetectionRunBundle:
    """Load a run plus normalized metadata for comparison and analysis scripts."""

    manifest_path = _resolve_manifest_path(run_dir_or_manifest)
    run_directory = manifest_path.parent
    summary = load_binary_detection_run_summary(run_dir_or_manifest, backend_name=backend_name)
    comparison_summary = _read_optional_json_mapping(run_directory / "comparison_summary.json") or {}
    pcap_experiment_summary = _read_optional_json_mapping(run_directory / "pcap_experiment_summary.json")

    reduction_method = (
        comparison_summary.get("graph_score_reduction")
        or summary.input_artifacts.get("graph_score_reduction")
        or comparison_summary.get("scorer_type")
        or backend_name
    )
    experiment_label = (
        comparison_summary.get("experiment_label")
        or (pcap_experiment_summary or {}).get("experiment_label")
        or summary.run_id
    )
    threshold = comparison_summary.get("threshold", summary.threshold)
    benign_inputs = _sequence_to_strings((pcap_experiment_summary or {}).get("benign_inputs"))
    malicious_inputs = _sequence_to_strings((pcap_experiment_summary or {}).get("malicious_inputs"))
    metadata = BinaryDetectionRunMetadata(
        experiment_label=str(experiment_label),
        reduction_method=str(reduction_method),
        threshold=float(threshold) if threshold not in (None, "", "None") else None,
        benign_train_graph_count=int(comparison_summary.get("benign_train_graph_count") or 0),
        benign_test_graph_count=int(comparison_summary.get("benign_test_graph_count") or 0),
        malicious_test_graph_count=int(comparison_summary.get("malicious_test_graph_count") or 0),
        worst_malicious_source_name=str(
            comparison_summary.get("worst_malicious_source_name") or "unavailable"
        ),
        scorer_role=normalize_run_scorer_role(
            backend_name=backend_name,
            scorer_name=str(reduction_method),
        ),
        benign_inputs=benign_inputs,
        malicious_inputs=malicious_inputs,
    )
    return BinaryDetectionRunBundle(
        summary=summary,
        comparison_summary=dict(comparison_summary),
        pcap_experiment_summary=dict(pcap_experiment_summary)
        if pcap_experiment_summary is not None
        else None,
        per_attack_metrics_by_task=_per_attack_metric_lookup(summary.per_attack_metrics),
        metadata=metadata,
    )


def compare_binary_detection_runs(
    tabular_run_dir: str | Path,
    graph_run_dir: str | Path,
    *,
    highlighted_attacks: Sequence[str] = ("recon", "web-based", "all_malicious"),
) -> BinaryDetectionComparisonReport:
    """Compare two binary detection outputs using the shared report schema."""

    tabular_summary = load_binary_detection_run_summary(tabular_run_dir, backend_name="tabular")
    graph_summary = load_binary_detection_run_summary(graph_run_dir, backend_name="graph")
    return compare_binary_detection_run_summaries(
        tabular_summary,
        graph_summary,
        highlighted_attacks=highlighted_attacks,
    )


def compare_and_export_binary_detection_runs(
    tabular_run_dir: str | Path,
    graph_run_dir: str | Path,
    output_dir: str | Path,
    *,
    highlighted_attacks: Sequence[str] = ("recon", "web-based", "all_malicious"),
    export_markdown: bool = False,
    timestamp: object | None = None,
) -> tuple[BinaryDetectionComparisonReport, BinaryDetectionComparisonExportResult]:
    """Compare two runs and export the resulting comparison report bundle."""

    report = compare_binary_detection_runs(
        tabular_run_dir,
        graph_run_dir,
        highlighted_attacks=highlighted_attacks,
    )
    export_result = export_comparison_report(
        report,
        output_dir,
        export_markdown=export_markdown,
        timestamp=timestamp,
    )
    return report, export_result


__all__ = [
    "BinaryDetectionRunBundle",
    "BinaryDetectionRunMetadata",
    "compare_and_export_binary_detection_runs",
    "compare_binary_detection_runs",
    "load_binary_detection_run_bundle",
    "load_binary_detection_run_summary",
    "summarize_comparison",
]
