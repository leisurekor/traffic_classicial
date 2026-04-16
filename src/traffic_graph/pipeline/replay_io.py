"""Read back exported run bundles into typed analysis-friendly objects."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from collections.abc import Iterable, Mapping, Sequence

from traffic_graph.pipeline.persistence import (
    is_parquet_export_available,
)
from traffic_graph.pipeline.replay_types import (
    ReplayAlertRecord,
    ReplayBundle,
    ReplayManifestInfo,
    ReplayScoreRecord,
    ScoreScope,
)

_SCORE_SCOPE_ORDER: tuple[ScoreScope, ...] = ("graph", "node", "edge", "flow")
_PREFERRED_TABULAR_FORMATS: tuple[str, ...] = ("parquet", "jsonl", "csv")
_PREFERRED_METRICS_FORMATS: tuple[str, ...] = ("json", "parquet", "jsonl", "csv")


def _json_loads_maybe(value: object) -> object:
    """Best-effort conversion from a scalar or JSON string into a Python value."""

    if value is None:
        return None
    if isinstance(value, (bool, int, float, dict, list)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            lowered = stripped.lower()
            if lowered == "true":
                return True
            if lowered == "false":
                return False
            return stripped
    return value


def _as_str_tuple(value: object) -> tuple[str, ...]:
    """Normalize manifest format sections into tuples of strings."""

    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _ensure_mapping(payload: object, *, message: str) -> Mapping[str, object]:
    """Validate that a loaded JSON payload is a mapping."""

    if not isinstance(payload, Mapping):
        raise ValueError(message)
    return payload


def _resolve_manifest_path(path: str | Path) -> Path:
    """Resolve a bundle directory or manifest path into a concrete manifest file."""

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
            "Multiple manifest.json files were found. Pass a specific bundle "
            "directory or manifest path."
        )
    return manifests[0]


def _resolve_artifact_path(raw_path: str, *, manifest_path: Path) -> Path:
    """Resolve an artifact path recorded in the manifest."""

    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    cwd_candidate = candidate
    if cwd_candidate.exists():
        return cwd_candidate
    manifest_relative = manifest_path.parent / candidate
    if manifest_relative.exists():
        return manifest_relative
    return cwd_candidate


def _read_jsonl_rows(path: Path) -> list[dict[str, object]]:
    """Read JSON Lines records into a list of dictionaries."""

    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, Mapping):
                rows.append(dict(payload))
    return rows


def _read_csv_rows(path: Path) -> list[dict[str, object]]:
    """Read CSV records into a list of dictionaries."""

    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({key: _json_loads_maybe(value) for key, value in row.items()})
    return rows


def _read_parquet_rows(path: Path) -> list[dict[str, object]]:
    """Read Parquet records into a list of dictionaries when supported."""

    if not is_parquet_export_available():
        raise RuntimeError("Parquet support is unavailable in this environment.")
    import pandas as pd  # type: ignore[import-not-found]

    frame = pd.read_parquet(path)
    return [dict(row) for row in frame.to_dict(orient="records")]


def _read_tabular_rows(path: Path) -> list[dict[str, object]]:
    """Read a table file by inferring the format from the path suffix."""

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl_rows(path)
    if suffix == ".csv":
        return _read_csv_rows(path)
    if suffix == ".parquet":
        return _read_parquet_rows(path)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(row) for row in payload if isinstance(row, Mapping)]
        if isinstance(payload, Mapping):
            return [dict(payload)]
        return []
    raise ValueError(f"Unsupported replay table format: {path.suffix}")


def _candidate_artifact_paths(
    manifest: ReplayManifestInfo,
    *,
    artifact_prefix: str,
    preferred_formats: Sequence[str],
) -> list[tuple[str, Path]]:
    """Collect existing candidate files for one logical artifact."""

    candidates: list[tuple[str, Path]] = []
    for fmt in preferred_formats:
        artifact_key = f"{artifact_prefix}_{fmt}"
        raw_path = manifest.artifact_paths.get(artifact_key)
        if raw_path is None:
            continue
        resolved_path = _resolve_artifact_path(raw_path, manifest_path=Path(manifest.manifest_path))
        if resolved_path.exists():
            candidates.append((fmt, resolved_path))
    return candidates


def _choose_artifact_path(
    manifest: ReplayManifestInfo,
    *,
    artifact_prefix: str,
    preferred_formats: Sequence[str],
) -> tuple[str, Path] | None:
    """Choose the best available file for one logical artifact."""

    candidates = _candidate_artifact_paths(
        manifest,
        artifact_prefix=artifact_prefix,
        preferred_formats=preferred_formats,
    )
    if not candidates:
        return None
    for fmt, resolved_path in candidates:
        if fmt == "parquet" and not is_parquet_export_available():
            continue
        return fmt, resolved_path
    return candidates[0]


def _parsed_metadata(value: object) -> dict[str, object]:
    """Restore the exported metadata field back into a dictionary."""

    parsed = _json_loads_maybe(value)
    if isinstance(parsed, Mapping):
        return dict(parsed)
    return {}


def _coerce_bool(value: object) -> bool | None:
    """Convert a scalar into an optional boolean."""

    parsed = _json_loads_maybe(value)
    if parsed is None:
        return None
    if isinstance(parsed, bool):
        return parsed
    if isinstance(parsed, (int, float)):
        return bool(parsed)
    return None


def _coerce_float(value: object) -> float | None:
    """Convert a scalar into an optional float."""

    parsed = _json_loads_maybe(value)
    if parsed is None:
        return None
    try:
        return float(parsed)
    except (TypeError, ValueError):
        return None


def _coerce_str(value: object) -> str:
    """Convert a scalar into a stable string."""

    if value is None:
        return ""
    return str(value)


def _coerce_scope(value: object) -> ScoreScope:
    """Normalize a score scope to the supported set."""

    text = _coerce_str(value).lower()
    if text in _SCORE_SCOPE_ORDER:
        return text  # type: ignore[return-value]
    raise ValueError(f"Unsupported score scope: {value}")


def _normalize_score_row(row: Mapping[str, object]) -> ReplayScoreRecord:
    """Convert a raw tabular row back into a typed replay score record."""

    return ReplayScoreRecord(
        run_id=_coerce_str(row.get("run_id")),
        timestamp=_coerce_str(row.get("timestamp")),
        split=_coerce_str(row.get("split")),
        score_scope=_coerce_scope(row.get("score_scope")),
        graph_id=_json_loads_maybe(row.get("graph_id")),
        window_id=_json_loads_maybe(row.get("window_id")),
        node_id=_json_loads_maybe(row.get("node_id")),
        edge_id=_json_loads_maybe(row.get("edge_id")),
        flow_id=_json_loads_maybe(row.get("flow_id")),
        anomaly_score=_coerce_float(row.get("anomaly_score")) or 0.0,
        threshold=_coerce_float(row.get("threshold")),
        is_alert=_coerce_bool(row.get("is_alert")),
        label=_json_loads_maybe(row.get("label")),
        metadata=_parsed_metadata(row.get("metadata")),
    )


def _normalize_alert_row(row: Mapping[str, object]) -> ReplayAlertRecord:
    """Convert a raw tabular row back into a typed replay alert record."""

    return ReplayAlertRecord(
        alert_id=_coerce_str(row.get("alert_id")),
        alert_level=_coerce_str(row.get("alert_level")),
        alert_scope=_coerce_str(row.get("alert_scope")),
        run_id=_coerce_str(row.get("run_id")),
        timestamp=_coerce_str(row.get("timestamp")),
        split=_coerce_str(row.get("split")),
        graph_id=_json_loads_maybe(row.get("graph_id")),
        window_id=_json_loads_maybe(row.get("window_id")),
        node_id=_json_loads_maybe(row.get("node_id")),
        edge_id=_json_loads_maybe(row.get("edge_id")),
        flow_id=_json_loads_maybe(row.get("flow_id")),
        anomaly_score=_coerce_float(row.get("anomaly_score")) or 0.0,
        threshold=_coerce_float(row.get("threshold")),
        is_alert=_coerce_bool(row.get("is_alert")),
        label=_json_loads_maybe(row.get("label")),
        metadata=_parsed_metadata(row.get("metadata")),
    )


def _assign_nested_value(root: dict[str, object], path: Sequence[str], value: object) -> None:
    """Assign a nested metric value inside a dictionary using a dotted path."""

    current: dict[str, object] = root
    for segment in path[:-1]:
        nested = current.get(segment)
        if not isinstance(nested, dict):
            nested = {}
            current[segment] = nested
        current = nested
    current[path[-1]] = value


def _rebuild_metrics_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Reconstruct the nested metrics summary from flattened metric rows."""

    summary: dict[str, object] = {}
    for row in rows:
        metric_path_value = row.get("metric_path")
        metric_value = _json_loads_maybe(row.get("metric_value"))
        if metric_path_value in {None, ""}:
            continue
        metric_path = str(metric_path_value).split(".")
        _assign_nested_value(summary, metric_path, metric_value)
    return summary


def load_score_table(
    path: str | Path,
    *,
    preferred_format: str | Sequence[str] | None = None,
) -> tuple[ReplayScoreRecord, ...]:
    """Load one persisted score-table file into typed replay records."""

    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Score table file not found: {resolved_path.as_posix()}")
    rows = _read_tabular_rows(resolved_path)
    return tuple(_normalize_score_row(row) for row in rows)


def load_alert_records(
    path: str | Path,
    *,
    preferred_format: str | Sequence[str] | None = None,
) -> tuple[ReplayAlertRecord, ...]:
    """Load one persisted alert-record file into typed replay records."""

    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Alert record file not found: {resolved_path.as_posix()}")
    rows = _read_tabular_rows(resolved_path)
    return tuple(_normalize_alert_row(row) for row in rows)


def load_metrics_summary(
    path: str | Path,
    *,
    preferred_format: str | Sequence[str] | None = None,
) -> dict[str, object]:
    """Load a persisted metrics summary from JSON or a flattened table."""

    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Metrics summary file not found: {resolved_path.as_posix()}")
    if resolved_path.suffix.lower() == ".json":
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
        mapping = _ensure_mapping(payload, message="Metrics summary JSON must be a mapping.")
        return dict(mapping)
    rows = _read_tabular_rows(resolved_path)
    return _rebuild_metrics_summary(rows)


def _load_manifest_info(manifest_path: Path) -> ReplayManifestInfo:
    """Read and validate a manifest file into a typed replay manifest."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    mapping = _ensure_mapping(payload, message="Manifest JSON root must be a mapping.")
    artifact_paths_raw = mapping.get("artifact_paths", {})
    artifact_paths = (
        dict(artifact_paths_raw)
        if isinstance(artifact_paths_raw, Mapping)
        else {}
    )
    row_counts_raw = mapping.get("row_counts", {})
    row_counts = (
        {str(key): int(value) for key, value in row_counts_raw.items()}
        if isinstance(row_counts_raw, Mapping)
        else {}
    )
    notes_value = mapping.get("notes", ())
    notes = _as_str_tuple(notes_value)
    return ReplayManifestInfo(
        run_id=_coerce_str(mapping.get("run_id")),
        timestamp=_coerce_str(mapping.get("timestamp")),
        split=_coerce_str(mapping.get("split")),
        manifest_path=manifest_path.as_posix(),
        base_directory=_coerce_str(mapping.get("base_directory")),
        run_directory=_coerce_str(mapping.get("run_directory")),
        score_formats=_as_str_tuple(mapping.get("score_formats")),
        alert_formats=_as_str_tuple(mapping.get("alert_formats")),
        metrics_formats=_as_str_tuple(mapping.get("metrics_formats")),
        artifact_paths={str(key): str(value) for key, value in artifact_paths.items()},
        row_counts=row_counts,
        notes=notes,
        raw_manifest=dict(mapping),
    )


def _load_scope_table(
    manifest: ReplayManifestInfo,
    *,
    scope: ScoreScope,
) -> tuple[tuple[ReplayScoreRecord, ...], str | None, list[str]]:
    """Load one score table by scope using the manifest's preferred format order."""

    notes: list[str] = []
    artifact_prefix = f"{scope}_scores"
    chosen = _choose_artifact_path(
        manifest,
        artifact_prefix=artifact_prefix,
        preferred_formats=_PREFERRED_TABULAR_FORMATS,
    )
    if chosen is None:
        notes.append(f"No exported {scope} score table was found in the manifest.")
        return (), None, notes
    _, resolved_path = chosen
    records = load_score_table(resolved_path)
    return records, resolved_path.as_posix(), notes


def _load_alert_table(
    manifest: ReplayManifestInfo,
) -> tuple[tuple[ReplayAlertRecord, ...], str | None, list[str]]:
    """Load alert records using the manifest's preferred format order."""

    notes: list[str] = []
    chosen = _choose_artifact_path(
        manifest,
        artifact_prefix="alert_records",
        preferred_formats=_PREFERRED_TABULAR_FORMATS,
    )
    if chosen is None:
        notes.append("No exported alert record table was found in the manifest.")
        return (), None, notes
    _, resolved_path = chosen
    records = load_alert_records(resolved_path)
    return records, resolved_path.as_posix(), notes


def _load_metrics_table(
    manifest: ReplayManifestInfo,
) -> tuple[dict[str, object], str | None, list[str]]:
    """Load metrics summary data using the manifest's preferred format order."""

    notes: list[str] = []
    chosen = _choose_artifact_path(
        manifest,
        artifact_prefix="metrics_summary",
        preferred_formats=_PREFERRED_METRICS_FORMATS,
    )
    if chosen is None:
        notes.append("No exported metrics summary was found in the manifest.")
        return {}, None, notes
    _, resolved_path = chosen
    summary = load_metrics_summary(resolved_path)
    return summary, resolved_path.as_posix(), notes


def load_export_bundle(export_dir: str | Path) -> ReplayBundle:
    """Load a persisted run bundle or manifest into a typed replay object."""

    manifest_path = _resolve_manifest_path(export_dir)
    manifest = _load_manifest_info(manifest_path)
    notes = list(manifest.notes)
    loaded_files: dict[str, str] = {}

    graph_scores, graph_path, graph_notes = _load_scope_table(manifest, scope="graph")
    node_scores, node_path, node_notes = _load_scope_table(manifest, scope="node")
    edge_scores, edge_path, edge_notes = _load_scope_table(manifest, scope="edge")
    flow_scores, flow_path, flow_notes = _load_scope_table(manifest, scope="flow")
    alert_records, alert_path, alert_notes = _load_alert_table(manifest)
    metrics_summary, metrics_path, metrics_notes = _load_metrics_table(manifest)

    notes.extend(graph_notes)
    notes.extend(node_notes)
    notes.extend(edge_notes)
    notes.extend(flow_notes)
    notes.extend(alert_notes)
    notes.extend(metrics_notes)

    if graph_path is not None:
        loaded_files["graph_scores"] = graph_path
    if node_path is not None:
        loaded_files["node_scores"] = node_path
    if edge_path is not None:
        loaded_files["edge_scores"] = edge_path
    if flow_path is not None:
        loaded_files["flow_scores"] = flow_path
    if alert_path is not None:
        loaded_files["alert_records"] = alert_path
    if metrics_path is not None:
        loaded_files["metrics_summary"] = metrics_path

    return ReplayBundle(
        run_id=manifest.run_id,
        timestamp=manifest.timestamp,
        split=manifest.split,
        manifest=manifest,
        file_index=dict(manifest.artifact_paths),
        loaded_files=loaded_files,
        graph_scores=graph_scores,
        node_scores=node_scores,
        edge_scores=edge_scores,
        flow_scores=flow_scores,
        alert_records=alert_records,
        metrics_summary=metrics_summary,
        notes=tuple(notes),
    )


def list_available_tables(bundle: ReplayBundle) -> tuple[str, ...]:
    """List the logical tables that are available in a replay bundle."""

    available: list[str] = []
    if "graph_scores" in bundle.loaded_files:
        available.append("graph")
    if "node_scores" in bundle.loaded_files:
        available.append("node")
    if "edge_scores" in bundle.loaded_files:
        available.append("edge")
    if "flow_scores" in bundle.loaded_files:
        available.append("flow")
    if "alert_records" in bundle.loaded_files:
        available.append("alerts")
    if "metrics_summary" in bundle.loaded_files:
        available.append("metrics")
    return tuple(available)


def get_score_table(
    bundle: ReplayBundle,
    scope: ScoreScope,
) -> tuple[ReplayScoreRecord, ...]:
    """Return one score table by scope from a replay bundle."""

    if scope == "graph":
        return bundle.graph_scores
    if scope == "node":
        return bundle.node_scores
    if scope == "edge":
        return bundle.edge_scores
    if scope == "flow":
        return bundle.flow_scores
    raise ValueError(f"Unsupported score scope: {scope}")


def get_alert_records(
    bundle: ReplayBundle,
    *,
    only_positive: bool = True,
) -> tuple[ReplayAlertRecord, ...]:
    """Return replay alert records, optionally filtering to positives only."""

    if not only_positive:
        return bundle.alert_records
    return tuple(record for record in bundle.alert_records if record.is_alert)


def get_metrics_summary(bundle: ReplayBundle) -> dict[str, object]:
    """Return the nested metrics summary from a replay bundle."""

    return dict(bundle.metrics_summary)


def summarize_replay_bundle(bundle: ReplayBundle) -> str:
    """Render a compact human-readable summary for one replay bundle."""

    lines = [
        f"Replay run id: {bundle.run_id}",
        f"Timestamp: {bundle.timestamp}",
        f"Split: {bundle.split}",
        f"Manifest: {bundle.manifest.manifest_path}",
        "Available tables:",
        "  - " + ", ".join(list_available_tables(bundle)) if list_available_tables(bundle) else "  - none",
        "Counts:",
        f"  - graph_scores={len(bundle.graph_scores)}",
        f"  - node_scores={len(bundle.node_scores)}",
        f"  - edge_scores={len(bundle.edge_scores)}",
        f"  - flow_scores={len(bundle.flow_scores)}",
        f"  - alert_records={len(bundle.alert_records)}",
        f"  - metric_scopes={len(bundle.metrics_summary)}",
    ]
    if bundle.loaded_files:
        lines.append("Loaded files:")
        lines.extend(f"  - {name}: {path}" for name, path in bundle.loaded_files.items())
    if bundle.notes:
        lines.append("Notes:")
        lines.extend(f"  - {note}" for note in bundle.notes)
    return "\n".join(lines)


__all__ = [
    "ReplayAlertRecord",
    "ReplayBundle",
    "ReplayManifestInfo",
    "ReplayScoreRecord",
    "get_alert_records",
    "get_metrics_summary",
    "get_score_table",
    "list_available_tables",
    "load_alert_records",
    "load_export_bundle",
    "load_metrics_summary",
    "load_score_table",
    "summarize_replay_bundle",
]
