"""Persistence helpers for prompt dataset artifacts."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Sequence

from traffic_graph.explain.prompt_types import (
    PROMPT_INPUT_FIELDS,
    PromptDatasetArtifact,
    PromptInput,
)


@dataclass(frozen=True, slots=True)
class PromptDatasetLayout:
    """Filesystem layout used to persist a prompt dataset."""

    base_directory: str
    run_directory: str
    timestamp: str


@dataclass(slots=True)
class PromptDatasetExportResult:
    """Summary returned after exporting a prompt dataset."""

    dataset_id: str
    run_id: str
    timestamp: str
    output_directory: str
    manifest_path: str
    artifact_paths: dict[str, str] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _timestamp_token(value: object | None) -> str:
    """Normalize a timestamp into the prompt dataset directory token format."""

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
    """Convert an arbitrary object into a filesystem-safe token."""

    token = str(value).strip()
    token = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in token)
    token = token.strip("-._")
    return token or "prompt_dataset"


def build_prompt_dataset_layout(
    output_dir: str | Path,
    *,
    run_id: str,
    timestamp: object | None = None,
) -> PromptDatasetLayout:
    """Construct the directory layout for one exported prompt dataset."""

    base_directory = Path(output_dir)
    timestamp_token = _timestamp_token(timestamp)
    run_directory = base_directory / _slugify_token(run_id) / timestamp_token
    return PromptDatasetLayout(
        base_directory=base_directory.as_posix(),
        run_directory=run_directory.as_posix(),
        timestamp=timestamp_token,
    )


def _json_scalar(value: object) -> object:
    """Convert common scalar values into JSON-friendly primitives."""

    if hasattr(value, "item"):
        try:
            return value.item()  # type: ignore[call-arg]
        except Exception:
            return value
    return value


def _csv_cell(value: object) -> object:
    """Convert a prompt field into a CSV-friendly cell value."""

    value = _json_scalar(value)
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _export_prompt_inputs_jsonl(
    prompt_inputs: Sequence[PromptInput],
    path: Path,
) -> int:
    """Write prompt inputs to a JSON Lines file and return the row count."""

    with path.open("w", encoding="utf-8") as handle:
        for prompt_input in prompt_inputs:
            payload = prompt_input.to_dict()
            ordered = {field: payload.get(field) for field in PROMPT_INPUT_FIELDS}
            handle.write(json.dumps(ordered, ensure_ascii=False, default=str))
            handle.write("\n")
    return len(prompt_inputs)


def _export_prompt_inputs_csv(
    prompt_inputs: Sequence[PromptInput],
    path: Path,
) -> int:
    """Write prompt inputs to a CSV file and return the row count."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PROMPT_INPUT_FIELDS))
        writer.writeheader()
        for prompt_input in prompt_inputs:
            payload = prompt_input.to_dict()
            ordered = {
                field: _csv_cell(payload.get(field))
                for field in PROMPT_INPUT_FIELDS
            }
            writer.writerow(ordered)
    return len(prompt_inputs)


def export_prompt_dataset(
    dataset: PromptDatasetArtifact,
    output_dir: str | Path,
    *,
    formats: Sequence[str] = ("jsonl", "csv"),
    timestamp: object | None = None,
) -> PromptDatasetExportResult:
    """Export a prompt dataset to JSONL, CSV, and a lightweight manifest."""

    layout = build_prompt_dataset_layout(
        output_dir,
        run_id=dataset.run_id,
        timestamp=timestamp,
    )
    run_directory = Path(layout.run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)
    artifact_paths: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    notes = list(dataset.notes)
    normalized_formats = {format_name.lower() for format_name in formats}

    if "jsonl" in normalized_formats:
        jsonl_path = run_directory / "prompt_inputs.jsonl"
        row_counts["jsonl"] = _export_prompt_inputs_jsonl(dataset.prompt_inputs, jsonl_path)
        artifact_paths["jsonl"] = jsonl_path.as_posix()
    if "csv" in normalized_formats:
        csv_path = run_directory / "prompt_inputs.csv"
        row_counts["csv"] = _export_prompt_inputs_csv(dataset.prompt_inputs, csv_path)
        artifact_paths["csv"] = csv_path.as_posix()
    unsupported_formats = sorted(normalized_formats.difference({"jsonl", "csv"}))
    if unsupported_formats:
        notes.append(
            "Skipped unsupported prompt dataset export formats: "
            + ", ".join(unsupported_formats)
        )

    manifest_payload = {
        "dataset_id": dataset.dataset_id,
        "run_id": dataset.run_id,
        "timestamp": layout.timestamp,
        "base_directory": layout.base_directory,
        "run_directory": layout.run_directory,
        "scope": dataset.scope,
        "selection_mode": dataset.selection_mode,
        "only_alerts": dataset.only_alerts,
        "balanced": dataset.balanced,
        "top_k": dataset.top_k,
        "max_samples": dataset.max_samples,
        "source_sample_count": dataset.source_sample_count,
        "selected_sample_count": dataset.selected_sample_count,
        "summary": dataset.summary.to_dict(),
        "artifact_paths": artifact_paths,
        "row_counts": row_counts,
        "notes": notes,
        "formats": sorted(normalized_formats),
        "prompt_input_fields": list(PROMPT_INPUT_FIELDS),
        "metadata": dict(dataset.metadata),
    }
    manifest_path = run_directory / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    artifact_paths["manifest_json"] = manifest_path.as_posix()

    return PromptDatasetExportResult(
        dataset_id=dataset.dataset_id,
        run_id=dataset.run_id,
        timestamp=layout.timestamp,
        output_directory=layout.run_directory,
        manifest_path=manifest_path.as_posix(),
        artifact_paths=artifact_paths,
        row_counts=row_counts,
        notes=notes,
    )


__all__ = [
    "PromptDatasetExportResult",
    "PromptDatasetLayout",
    "build_prompt_dataset_layout",
    "export_prompt_dataset",
]
