"""Local persistence helpers for score tables, alerts, and metric summaries."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.util import find_spec
from pathlib import Path
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

from traffic_graph.pipeline.alert_types import AlertRecord, AlertScoreTables

ExportFormat = Literal["jsonl", "csv", "parquet"]
"""Tabular export formats supported by the persistence layer."""

METRICS_JSON_FORMAT = "json"
"""Dedicated JSON file used for the nested metrics summary."""

_SUPPORTED_TABLE_FORMATS: tuple[ExportFormat, ...] = ("jsonl", "csv", "parquet")
_DEFAULT_TABLE_FORMATS: tuple[ExportFormat, ...] = ("jsonl", "csv", "parquet")
_DEFAULT_METRICS_FORMATS: tuple[str, ...] = ("json", "jsonl", "csv", "parquet")

_SCORE_SCOPE_ORDER: tuple[str, ...] = ("graph", "node", "edge", "flow")
_METRIC_SCOPE_ORDER: tuple[str, ...] = ("graph", "node", "edge", "flow", "overall")

SCORE_EXPORT_FIELDS: tuple[str, ...] = (
    "run_id",
    "timestamp",
    "split",
    "score_scope",
    "graph_id",
    "window_id",
    "node_id",
    "edge_id",
    "flow_id",
    "anomaly_score",
    "threshold",
    "is_alert",
    "label",
    "metadata",
)
"""Stable column order for score table exports."""

ALERT_EXPORT_FIELDS: tuple[str, ...] = (
    "alert_id",
    "alert_level",
    "alert_scope",
    "run_id",
    "timestamp",
    "split",
    "graph_id",
    "window_id",
    "node_id",
    "edge_id",
    "flow_id",
    "anomaly_score",
    "threshold",
    "is_alert",
    "label",
    "metadata",
)
"""Stable column order for alert exports."""

METRICS_EXPORT_FIELDS: tuple[str, ...] = (
    "run_id",
    "timestamp",
    "split",
    "scope",
    "metric_path",
    "metric_value",
)
"""Stable column order for flattened metric exports."""


@dataclass(slots=True)
class ExportResult:
    """Result object returned by the low-level export helpers."""

    output_directory: str
    artifact_paths: dict[str, str] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _slugify_token(value: object, default: str) -> str:
    """Convert an arbitrary value into a filesystem-safe token."""

    token = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    token = token.strip("-._")
    return token or default


def _timestamp_token(value: object | None) -> str:
    """Normalize timestamps into the `YYYYmmddTHHMMSSZ` token format."""

    if value is None:
        moment = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        moment = value.astimezone(timezone.utc)
    else:
        return _slugify_token(value, "timestamp")
    return moment.strftime("%Y%m%dT%H%M%SZ")


def _ensure_directory(path: Path) -> None:
    """Create a directory and its parents when missing."""

    path.mkdir(parents=True, exist_ok=True)


def _json_default(value: object) -> object:
    """Serialize values that are not JSON-native by default."""

    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()  # type: ignore[call-arg]
        except TypeError:
            pass
    return str(value)


def _json_string(value: object) -> str:
    """Serialize a value into a stable JSON string."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)


def is_parquet_export_available() -> bool:
    """Return ``True`` when pandas and pyarrow are importable."""

    return find_spec("pandas") is not None and find_spec("pyarrow") is not None


def _first_present(row: Mapping[str, object], keys: Sequence[str]) -> object | None:
    """Return the first non-empty value from a mapping for the requested keys."""

    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def _score_value(scope: str, row: Mapping[str, object]) -> float:
    """Extract the anomaly score from a raw score-table row."""

    key_map: dict[str, tuple[str, ...]] = {
        "graph": ("graph_anomaly_score", "anomaly_score", "score"),
        "node": ("node_anomaly_score", "anomaly_score", "score"),
        "edge": ("edge_anomaly_score", "anomaly_score", "score"),
        "flow": ("flow_anomaly_score", "anomaly_score", "score"),
    }
    raw_value = _first_present(row, key_map.get(scope, ("anomaly_score", "score")))
    try:
        return float(raw_value) if raw_value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _row_label(row: Mapping[str, object]) -> object | None:
    """Return a stable label field from a raw score row when present."""

    return _first_present(
        row,
        (
            "label",
            "graph_label",
            "edge_label",
            "flow_label",
            "node_label",
        ),
    )


def _row_metadata(
    row: Mapping[str, object],
    *,
    excluded_keys: set[str],
) -> str:
    """Keep the remaining row fields as a compact JSON metadata payload."""

    metadata: dict[str, object] = {}
    for key, value in row.items():
        if key in excluded_keys:
            continue
        metadata[key] = value
    return _json_string(metadata)


def _normalize_score_row(
    scope: str,
    row: Mapping[str, object],
    *,
    run_id: str,
    timestamp: str,
    split: str,
    threshold: float | None,
) -> dict[str, object]:
    """Convert a raw score row into the canonical export schema."""

    graph_id = _first_present(row, ("graph_id", "graph_index"))
    window_id = _first_present(row, ("window_id", "window_index"))
    node_id = _first_present(row, ("node_id",))
    edge_id = _first_present(row, ("edge_id",))
    flow_id = _first_present(row, ("flow_id", "logical_flow_id"))
    anomaly_score = _score_value(scope, row)
    is_alert = None if threshold is None else anomaly_score >= threshold
    excluded_keys = {
        "graph_id",
        "graph_index",
        "window_id",
        "window_index",
        "node_id",
        "edge_id",
        "flow_id",
        "logical_flow_id",
        "graph_anomaly_score",
        "node_anomaly_score",
        "edge_anomaly_score",
        "flow_anomaly_score",
        "anomaly_score",
        "score",
        "label",
        "graph_label",
        "edge_label",
        "flow_label",
        "node_label",
    }
    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "split": split,
        "score_scope": scope,
        "graph_id": graph_id,
        "window_id": window_id,
        "node_id": node_id,
        "edge_id": edge_id,
        "flow_id": flow_id,
        "anomaly_score": anomaly_score,
        "threshold": threshold,
        "is_alert": is_alert,
        "label": _row_label(row),
        "metadata": _row_metadata(row, excluded_keys=excluded_keys),
    }


def _normalize_alert_row(
    record: AlertRecord | Mapping[str, object] | object,
    *,
    run_id: str,
    timestamp: str,
    split: str,
) -> dict[str, object]:
    """Convert an alert record into the canonical export schema."""

    if isinstance(record, AlertRecord):
        payload = record.to_dict()
    elif isinstance(record, Mapping):
        payload = dict(record)
    elif hasattr(record, "to_dict"):
        maybe_payload = getattr(record, "to_dict")()
        payload = dict(maybe_payload) if isinstance(maybe_payload, Mapping) else {}
    else:
        payload = dict(getattr(record, "__dict__", {}))

    metadata_value = payload.get("metadata", {})
    metadata_json = _json_string(metadata_value if isinstance(metadata_value, Mapping) else metadata_value)
    return {
        "alert_id": payload.get("alert_id"),
        "alert_level": payload.get("alert_level"),
        "alert_scope": payload.get("alert_scope"),
        "run_id": run_id,
        "timestamp": timestamp,
        "split": split,
        "graph_id": payload.get("graph_id"),
        "window_id": payload.get("window_id"),
        "node_id": payload.get("node_id"),
        "edge_id": payload.get("edge_id"),
        "flow_id": payload.get("flow_id"),
        "anomaly_score": payload.get("anomaly_score"),
        "threshold": payload.get("threshold"),
        "is_alert": payload.get("is_alert"),
        "label": payload.get("label"),
        "metadata": metadata_json,
    }


def _flatten_metrics_summary(
    summary: object,
    *,
    timestamp: str,
    run_id: str,
    split: str,
    scope: str = "overall",
    prefix: tuple[str, ...] = (),
) -> list[dict[str, object]]:
    """Flatten nested metric dictionaries into a row-oriented table."""

    rows: list[dict[str, object]] = []
    if isinstance(summary, Mapping):
        for key in sorted(summary.keys(), key=str):
            value = summary[key]
            next_prefix = prefix + (str(key),)
            next_scope = scope
            if not prefix and str(key) in _METRIC_SCOPE_ORDER:
                next_scope = str(key)
            if isinstance(value, Mapping):
                rows.extend(
                    _flatten_metrics_summary(
                        value,
                        timestamp=timestamp,
                        run_id=run_id,
                        split=split,
                        scope=next_scope,
                        prefix=next_prefix,
                    )
                )
            else:
                rows.append(
                    {
                        "run_id": run_id,
                        "timestamp": timestamp,
                        "split": split,
                        "scope": next_scope,
                        "metric_path": ".".join(next_prefix),
                        "metric_value": value,
                    }
                )
        return rows

    rows.append(
        {
            "run_id": run_id,
            "timestamp": timestamp,
            "split": split,
            "scope": scope,
            "metric_path": ".".join(prefix) if prefix else "value",
            "metric_value": summary,
        }
    )
    return rows


def _write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    field_order: Sequence[str],
) -> None:
    """Write a stable JSON Lines file."""

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            ordered = {field: row.get(field) for field in field_order}
            handle.write(json.dumps(ordered, ensure_ascii=False, default=_json_default))
            handle.write("\n")


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    field_order: Sequence[str],
) -> None:
    """Write a stable CSV file."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(field_order), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in field_order})


def _write_parquet(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    field_order: Sequence[str],
) -> tuple[bool, str | None]:
    """Write a Parquet file when pandas and pyarrow are available."""

    if not rows:
        if not is_parquet_export_available():
            return False, "Parquet export skipped because pandas or pyarrow is unavailable."
        try:
            import pandas as pd  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - optional dependency path
            return False, f"Parquet export skipped because pandas could not be imported: {exc}"
        frame = pd.DataFrame(columns=list(field_order))
        frame.to_parquet(path, index=False)
        return True, None

    if not is_parquet_export_available():
        return False, "Parquet export skipped because pandas or pyarrow is unavailable."

    try:
        import pandas as pd  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency path
        return False, f"Parquet export skipped because pandas could not be imported: {exc}"

    frame = pd.DataFrame([{field: row.get(field) for field in field_order} for row in rows])
    frame = frame.loc[:, list(field_order)]
    try:
        frame.to_parquet(path, index=False)
    except Exception as exc:  # pragma: no cover - optional dependency path
        return False, f"Parquet export skipped because writing failed: {exc}"
    return True, None


def _write_table_files(
    rows: Sequence[Mapping[str, object]],
    *,
    output_directory: Path,
    base_name: str,
    artifact_prefix: str,
    field_order: Sequence[str],
    formats: Sequence[str],
    result: ExportResult,
) -> None:
    """Write one logical table to the requested export formats."""

    _ensure_directory(output_directory)
    normalized_formats = tuple(fmt.lower().strip() for fmt in formats if str(fmt).strip())
    for fmt in normalized_formats:
        file_path = output_directory / f"{base_name}.{fmt}"
        if fmt == "jsonl":
            _write_jsonl(file_path, rows, field_order)
            result.artifact_paths[f"{artifact_prefix}_jsonl"] = file_path.as_posix()
        elif fmt == "csv":
            _write_csv(file_path, rows, field_order)
            result.artifact_paths[f"{artifact_prefix}_csv"] = file_path.as_posix()
        elif fmt == "parquet":
            written, note = _write_parquet(file_path, rows, field_order)
            if written:
                result.artifact_paths[f"{artifact_prefix}_parquet"] = file_path.as_posix()
            elif note is not None:
                result.notes.append(note)
        else:
            result.notes.append(
                f"Skipped unsupported export format '{fmt}' for '{base_name}'."
            )


def export_score_tables(
    score_tables: object,
    output_dir: str | Path,
    *,
    run_id: str,
    split: str,
    timestamp: object | None = None,
    formats: Sequence[str] = _DEFAULT_TABLE_FORMATS,
    anomaly_threshold: float | None = None,
) -> ExportResult:
    """Export graph, node, edge, and flow score tables to local files."""

    timestamp_token = _timestamp_token(timestamp)
    output_directory = Path(output_dir)
    _ensure_directory(output_directory)
    tables = AlertScoreTables.from_value(score_tables)
    result = ExportResult(output_directory=output_directory.as_posix())
    for scope, rows in (
        ("graph", tables.graph_scores),
        ("node", tables.node_scores),
        ("edge", tables.edge_scores),
        ("flow", tables.flow_scores),
    ):
        normalized_rows = [
            _normalize_score_row(
                scope,
                row,
                run_id=run_id,
                timestamp=timestamp_token,
                split=split,
                threshold=anomaly_threshold,
            )
            for row in rows
        ]
        result.row_counts[f"{scope}_scores"] = len(normalized_rows)
        base_name = f"{scope}_scores.{split}"
        _write_table_files(
            normalized_rows,
            output_directory=output_directory,
            base_name=base_name,
            artifact_prefix=f"{scope}_scores",
            field_order=SCORE_EXPORT_FIELDS,
            formats=formats,
            result=result,
        )
    return result


def export_alert_records(
    alert_records: Iterable[AlertRecord | Mapping[str, object] | object],
    output_dir: str | Path,
    *,
    run_id: str,
    split: str,
    timestamp: object | None = None,
    formats: Sequence[str] = _DEFAULT_TABLE_FORMATS,
) -> ExportResult:
    """Export structured alert records to local files."""

    timestamp_token = _timestamp_token(timestamp)
    output_directory = Path(output_dir)
    _ensure_directory(output_directory)
    rows = [
        _normalize_alert_row(
            record,
            run_id=run_id,
            timestamp=timestamp_token,
            split=split,
        )
        for record in alert_records
    ]
    result = ExportResult(output_directory=output_directory.as_posix())
    result.row_counts["alert_records"] = len(rows)
    base_name = f"alert_records.{split}"
    _write_table_files(
        rows,
        output_directory=output_directory,
        base_name=base_name,
        artifact_prefix="alert_records",
        field_order=ALERT_EXPORT_FIELDS,
        formats=formats,
        result=result,
    )
    return result


def export_metrics_summary(
    metrics_summary: object,
    output_dir: str | Path,
    *,
    run_id: str,
    split: str,
    timestamp: object | None = None,
    formats: Sequence[str] = _DEFAULT_METRICS_FORMATS,
) -> ExportResult:
    """Export nested metric summaries as JSON plus a flattened table view."""

    timestamp_token = _timestamp_token(timestamp)
    output_directory = Path(output_dir)
    _ensure_directory(output_directory)
    result = ExportResult(output_directory=output_directory.as_posix())
    nested_payload = metrics_summary if metrics_summary is not None else {}
    json_path = output_directory / f"metrics_summary.{split}.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(nested_payload, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")
    result.artifact_paths["metrics_summary_json"] = json_path.as_posix()

    normalized_rows = _flatten_metrics_summary(
        nested_payload,
        timestamp=timestamp_token,
        run_id=run_id,
        split=split,
    )
    result.row_counts["metrics_rows"] = len(normalized_rows)

    table_formats = tuple(
        fmt for fmt in formats if fmt != METRICS_JSON_FORMAT and str(fmt).strip()
    )
    if table_formats:
        base_name = f"metrics_summary.{split}"
        _write_table_files(
            normalized_rows,
            output_directory=output_directory,
            base_name=base_name,
            artifact_prefix="metrics_summary",
            field_order=METRICS_EXPORT_FIELDS,
            formats=table_formats,
            result=result,
        )
    return result


__all__ = [
    "ALERT_EXPORT_FIELDS",
    "ExportFormat",
    "ExportResult",
    "METRICS_EXPORT_FIELDS",
    "METRICS_JSON_FORMAT",
    "SCORE_EXPORT_FIELDS",
    "export_alert_records",
    "export_metrics_summary",
    "export_score_tables",
    "is_parquet_export_available",
]
