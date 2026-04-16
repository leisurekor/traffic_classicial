"""Persistence helpers for exported LLM result artifacts."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Sequence

from traffic_graph.explain.llm_results import (
    LLM_RESULT_ARTIFACT_FIELDS,
    LLM_RESULT_FIELDS,
    LLMResultArtifact,
    LLMResultRecord,
    LLMResultSummary,
)


@dataclass(frozen=True, slots=True)
class LLMResultLayout:
    """Filesystem layout used to persist a batch of LLM results."""

    base_directory: str
    run_directory: str
    timestamp: str


@dataclass(slots=True)
class LLMResultExportResult:
    """Summary returned after exporting an LLM result artifact."""

    result_id: str
    run_id: str
    model_name: str
    timestamp: str
    output_directory: str
    manifest_path: str
    summary_path: str
    artifact_paths: dict[str, str] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _timestamp_token(value: object | None) -> str:
    """Normalize a timestamp into a stable UTC token."""

    if value is None:
        moment = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        moment = value.astimezone(timezone.utc)
    else:
        token = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
        token = token.strip("-._")
        return token or "timestamp"
    return moment.strftime("%Y%m%dT%H%M%SZ")


def _slugify_token(value: object) -> str:
    """Convert an arbitrary token into a filesystem-safe path component."""

    token = str(value).strip()
    token = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in token)
    token = token.strip("-._")
    return token or "llm_results"


def build_llm_result_layout(
    output_dir: str | Path,
    *,
    run_id: str,
    model_name: str,
    timestamp: object | None = None,
) -> LLMResultLayout:
    """Construct the directory layout for one LLM result batch."""

    base_directory = Path(output_dir)
    timestamp_token = _timestamp_token(timestamp)
    run_directory = base_directory / _slugify_token(run_id) / timestamp_token / _slugify_token(model_name)
    return LLMResultLayout(
        base_directory=base_directory.as_posix(),
        run_directory=run_directory.as_posix(),
        timestamp=timestamp_token,
    )


def _json_scalar(value: object) -> object:
    """Convert common scalar-like values into JSON-friendly primitives."""

    if hasattr(value, "item"):
        try:
            return value.item()  # type: ignore[call-arg]
        except Exception:
            return value
    return value


def _csv_cell(value: object) -> object:
    """Convert a result field into a CSV-friendly cell value."""

    value = _json_scalar(value)
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)


def _export_llm_results_jsonl(
    result_records: Sequence[LLMResultRecord],
    path: Path,
) -> int:
    """Write result records to a JSON Lines file and return the row count."""

    with path.open("w", encoding="utf-8") as handle:
        for record in result_records:
            payload = record.to_dict()
            ordered = {field: payload.get(field) for field in LLM_RESULT_FIELDS}
            handle.write(json.dumps(ordered, ensure_ascii=False, default=str))
            handle.write("\n")
    return len(result_records)


def _export_llm_results_csv(
    result_records: Sequence[LLMResultRecord],
    path: Path,
) -> int:
    """Write result records to a CSV file and return the row count."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LLM_RESULT_FIELDS))
        writer.writeheader()
        for record in result_records:
            payload = record.to_dict()
            ordered = {field: _csv_cell(payload.get(field)) for field in LLM_RESULT_FIELDS}
            writer.writerow(ordered)
    return len(result_records)


def export_llm_results(
    results: LLMResultArtifact,
    output_dir: str | Path,
    *,
    formats: Sequence[str] = ("jsonl", "csv"),
    timestamp: object | None = None,
) -> LLMResultExportResult:
    """Export LLM results to JSONL, CSV, and lightweight manifest files."""

    layout = build_llm_result_layout(
        output_dir,
        run_id=results.run_id,
        model_name=results.model_name,
        timestamp=timestamp or results.created_at,
    )
    run_directory = Path(layout.run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)
    artifact_paths: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    notes = list(results.notes)
    normalized_formats = {format_name.lower() for format_name in formats}

    if "jsonl" in normalized_formats:
        jsonl_path = run_directory / "results.jsonl"
        row_counts["jsonl"] = _export_llm_results_jsonl(results.result_records, jsonl_path)
        artifact_paths["jsonl"] = jsonl_path.as_posix()
    if "csv" in normalized_formats:
        csv_path = run_directory / "results.csv"
        row_counts["csv"] = _export_llm_results_csv(results.result_records, csv_path)
        artifact_paths["csv"] = csv_path.as_posix()
    unsupported_formats = sorted(normalized_formats.difference({"jsonl", "csv"}))
    if unsupported_formats:
        notes.append(
            "Skipped unsupported LLM result export formats: "
            + ", ".join(unsupported_formats)
        )

    summary_payload = results.summary.to_dict()
    summary_path = run_directory / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    artifact_paths["summary_json"] = summary_path.as_posix()

    manifest_payload = {
        "result_id": results.result_id,
        "run_id": results.run_id,
        "model_name": results.model_name,
        "created_at": results.created_at,
        "timestamp": layout.timestamp,
        "base_directory": layout.base_directory,
        "run_directory": layout.run_directory,
        "source_prompt_dataset": {
            "dataset_id": results.source_prompt_dataset_id,
            "run_id": results.source_prompt_dataset_run_id,
            "timestamp": results.source_prompt_dataset_timestamp,
            "scope": results.source_prompt_dataset_scope,
            "selection_mode": results.source_prompt_dataset_selection_mode,
        },
        "result_count": results.result_count,
        "summary": summary_payload,
        "artifact_paths": artifact_paths,
        "row_counts": row_counts,
        "notes": notes,
        "formats": sorted(normalized_formats),
        "result_fields": list(LLM_RESULT_FIELDS),
        "artifact_fields": list(LLM_RESULT_ARTIFACT_FIELDS),
        "metadata": dict(results.metadata),
    }
    manifest_path = run_directory / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    artifact_paths["manifest_json"] = manifest_path.as_posix()

    return LLMResultExportResult(
        result_id=results.result_id,
        run_id=results.run_id,
        model_name=results.model_name,
        timestamp=layout.timestamp,
        output_directory=layout.run_directory,
        manifest_path=manifest_path.as_posix(),
        summary_path=summary_path.as_posix(),
        artifact_paths=artifact_paths,
        row_counts=row_counts,
        notes=notes,
    )


def summarize_llm_results_text(results: LLMResultArtifact | LLMResultSummary) -> str:
    """Render a compact human-readable summary for LLM result artifacts."""

    summary = results.summary if isinstance(results, LLMResultArtifact) else results
    lines = [
        f"LLM results: total={summary.total_count}",
        "Statuses: "
        + (
            ", ".join(
                f"{status}={summary.status_counts.get(status, 0)}"
                for status in ("success", "failed", "skipped")
            )
            if summary.status_counts
            else "none"
        ),
    ]
    if summary.model_name:
        lines.append(f"Model name: {summary.model_name}")
    if summary.created_at:
        lines.append(f"Created at: {summary.created_at}")
    if summary.source_prompt_dataset_id:
        lines.append(f"Source prompt dataset: {summary.source_prompt_dataset_id}")
    if summary.preview_prompt_ids:
        lines.append("Preview prompt ids: " + ", ".join(summary.preview_prompt_ids))
    if summary.preview_response_ids:
        lines.append("Preview response ids: " + ", ".join(summary.preview_response_ids))
    return "\n".join(lines)


__all__ = [
    "LLMResultExportResult",
    "LLMResultLayout",
    "build_llm_result_layout",
    "export_llm_results",
    "summarize_llm_results_text",
]
