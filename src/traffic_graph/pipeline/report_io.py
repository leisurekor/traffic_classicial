"""High-level run bundle export helpers for score and alert persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterable, Mapping, Sequence
import re

from traffic_graph.pipeline.alert_types import AlertRecord
from traffic_graph.pipeline.persistence import (
    ExportResult,
    export_alert_records,
    export_metrics_summary,
    export_score_tables,
)


@dataclass(frozen=True, slots=True)
class RunBundleLayout:
    """Filesystem layout used to persist a single pipeline run."""

    base_directory: str
    run_directory: str
    timestamp: str
    scores_directory: str
    alerts_directory: str
    metrics_directory: str


@dataclass(slots=True)
class RunBundleExportResult:
    """Summary returned after exporting a complete run bundle."""

    run_id: str
    timestamp: str
    run_directory: str
    manifest_path: str
    artifact_paths: dict[str, str] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _timestamp_token(value: object | None) -> str:
    """Normalize a timestamp into the run directory token format."""

    if value is None:
        moment = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        moment = value.astimezone(timezone.utc)
    else:
        token = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
        token = token.strip("-._")
        return token or "timestamp"
    return moment.strftime("%Y%m%dT%H%M%SZ")


def _slugify_run_id(run_id: object) -> str:
    """Normalize the run id so it can safely be used in paths."""

    token = str(run_id).strip()
    token = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in token)
    token = token.strip("-._")
    return token or "run"


def build_run_bundle_layout(
    export_dir: str | Path,
    *,
    run_id: str,
    timestamp: object | None = None,
) -> RunBundleLayout:
    """Construct the directory layout for one persisted run."""

    base_directory = Path(export_dir)
    run_token = _slugify_run_id(run_id)
    timestamp_token = _timestamp_token(timestamp)
    run_directory = base_directory / run_token / timestamp_token
    scores_directory = run_directory / "scores"
    alerts_directory = run_directory / "alerts"
    metrics_directory = run_directory / "metrics"
    return RunBundleLayout(
        base_directory=base_directory.as_posix(),
        run_directory=run_directory.as_posix(),
        timestamp=timestamp_token,
        scores_directory=scores_directory.as_posix(),
        alerts_directory=alerts_directory.as_posix(),
        metrics_directory=metrics_directory.as_posix(),
    )


def _ensure_directory(path: str | Path) -> Path:
    """Create a directory if necessary and return it as a :class:`Path`."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _merge_artifact_paths(*results: ExportResult) -> dict[str, str]:
    """Combine artifact path dictionaries from multiple export results."""

    artifact_paths: dict[str, str] = {}
    for result in results:
        artifact_paths.update(result.artifact_paths)
    return artifact_paths


def _merge_row_counts(*results: ExportResult) -> dict[str, int]:
    """Combine row-count dictionaries from multiple export results."""

    row_counts: dict[str, int] = {}
    for result in results:
        row_counts.update(result.row_counts)
    return row_counts


def _collect_notes(*results: ExportResult) -> list[str]:
    """Collect non-empty notes from multiple export results."""

    notes: list[str] = []
    seen: set[str] = set()
    for result in results:
        for note in result.notes:
            if note and note not in seen:
                seen.add(note)
                notes.append(note)
    return notes


def export_run_bundle(
    score_tables: object,
    alert_records: Iterable[AlertRecord | Mapping[str, object] | object],
    metrics_summary: object,
    export_dir: str | Path,
    *,
    run_id: str,
    split: str,
    timestamp: object | None = None,
    anomaly_threshold: float | None = None,
    score_formats: Sequence[str] = ("jsonl", "csv", "parquet"),
    alert_formats: Sequence[str] = ("jsonl", "csv", "parquet"),
    metrics_formats: Sequence[str] = ("json", "jsonl", "csv", "parquet"),
) -> RunBundleExportResult:
    """Export all run artifacts into a timestamped bundle directory."""

    layout = build_run_bundle_layout(export_dir, run_id=run_id, timestamp=timestamp)
    run_directory = _ensure_directory(layout.run_directory)
    _ensure_directory(layout.scores_directory)
    _ensure_directory(layout.alerts_directory)
    _ensure_directory(layout.metrics_directory)

    score_result = export_score_tables(
        score_tables,
        layout.scores_directory,
        run_id=run_id,
        split=split,
        timestamp=layout.timestamp,
        formats=score_formats,
        anomaly_threshold=anomaly_threshold,
    )
    alert_result = export_alert_records(
        alert_records,
        layout.alerts_directory,
        run_id=run_id,
        split=split,
        timestamp=layout.timestamp,
        formats=alert_formats,
    )
    metric_result = export_metrics_summary(
        metrics_summary,
        layout.metrics_directory,
        run_id=run_id,
        split=split,
        timestamp=layout.timestamp,
        formats=metrics_formats,
    )

    artifact_paths = _merge_artifact_paths(score_result, alert_result, metric_result)
    row_counts = _merge_row_counts(score_result, alert_result, metric_result)
    notes = _collect_notes(score_result, alert_result, metric_result)
    manifest_payload = {
        "run_id": run_id,
        "timestamp": layout.timestamp,
        "split": split,
        "base_directory": layout.base_directory,
        "run_directory": layout.run_directory,
        "score_formats": list(score_formats),
        "alert_formats": list(alert_formats),
        "metrics_formats": list(metrics_formats),
        "artifact_paths": artifact_paths,
        "row_counts": row_counts,
        "notes": notes,
    }
    manifest_path = run_directory / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    artifact_paths["manifest_json"] = manifest_path.as_posix()

    return RunBundleExportResult(
        run_id=run_id,
        timestamp=layout.timestamp,
        run_directory=layout.run_directory,
        manifest_path=manifest_path.as_posix(),
        artifact_paths=artifact_paths,
        row_counts=row_counts,
        notes=notes,
    )


__all__ = [
    "RunBundleExportResult",
    "RunBundleLayout",
    "build_run_bundle_layout",
    "export_run_bundle",
]
