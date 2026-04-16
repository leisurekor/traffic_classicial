"""Readback helpers for exported prompt dataset bundles."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import cast

from traffic_graph.explain.prompt_replay_types import (
    PromptDatasetManifestInfo,
    PromptDatasetReplay,
)
from traffic_graph.explain.prompt_types import (
    PROMPT_INPUT_FIELDS,
    PromptDatasetSelectionMode,
    PromptDatasetSummary,
    PromptInput,
    PromptScope,
)


def _json_scalar(value: object) -> object:
    """Convert common scalar-like values into JSON-friendly primitives."""

    if hasattr(value, "item"):
        try:
            return value.item()  # type: ignore[call-arg]
        except Exception:
            return value
    return value


def _normalize_string_value(value: str) -> object:
    """Parse a string field into a more precise typed value when possible."""

    text = value.strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return json.loads(text)
    except Exception:
        return text


def _coerce_scalar(value: object) -> object:
    """Normalize scalar values from JSON or CSV input rows."""

    value = _json_scalar(value)
    if isinstance(value, str):
        return _normalize_string_value(value)
    return value


def _coerce_optional_float(value: object) -> float | None:
    """Convert a value into an optional float."""

    value = _coerce_scalar(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_int(value: object) -> int | None:
    """Convert a value into an optional integer."""

    value = _coerce_scalar(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_bool(value: object) -> bool | None:
    """Convert a value into an optional boolean."""

    value = _coerce_scalar(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def _coerce_mapping(value: object) -> dict[str, object]:
    """Convert a value into a JSON-like mapping."""

    value = _coerce_scalar(value)
    if isinstance(value, Mapping):
        return {str(key): _json_scalar(item) for key, item in value.items()}
    return {}


def _coerce_label(value: object) -> object | None:
    """Convert a label field while preserving stable numeric and string values."""

    value = _coerce_scalar(value)
    if value is None:
        return None
    return _json_scalar(value)


def _coerce_scope(value: object) -> PromptScope:
    """Validate and normalize a prompt scope value."""

    scope = str(_coerce_scalar(value))
    if scope not in {"graph", "flow", "node"}:
        raise ValueError(f"Unsupported prompt scope: {scope}")
    return cast(PromptScope, scope)


def _coerce_selection_mode(value: object) -> PromptDatasetSelectionMode:
    """Validate and normalize a prompt selection mode value."""

    mode = str(_coerce_scalar(value))
    if mode not in {"all", "alerts", "balanced"}:
        raise ValueError(f"Unsupported prompt dataset selection mode: {mode}")
    return cast(PromptDatasetSelectionMode, mode)


def _summary_from_payload(payload: object) -> PromptDatasetSummary:
    """Convert a manifest summary payload into a typed prompt summary."""

    if not isinstance(payload, Mapping):
        return PromptDatasetSummary(total_count=0)
    preview_prompt_ids = payload.get("preview_prompt_ids", ())
    if isinstance(preview_prompt_ids, Sequence) and not isinstance(preview_prompt_ids, (str, bytes)):
        preview_ids = tuple(str(item) for item in preview_prompt_ids)
    else:
        preview_ids = ()
    scope_counts_payload = payload.get("scope_counts", {})
    scope_counts: dict[str, int] = {}
    if isinstance(scope_counts_payload, Mapping):
        for key, value in scope_counts_payload.items():
            try:
                scope_counts[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
    try:
        total_count = int(payload.get("total_count", 0))
    except (TypeError, ValueError):
        total_count = 0
    try:
        alert_count = int(payload.get("alert_count", 0))
    except (TypeError, ValueError):
        alert_count = 0
    try:
        labeled_count = int(payload.get("labeled_count", 0))
    except (TypeError, ValueError):
        labeled_count = 0
    max_anomaly_score_payload = payload.get("max_anomaly_score")
    try:
        max_anomaly_score = (
            None
            if max_anomaly_score_payload is None or max_anomaly_score_payload == ""
            else float(max_anomaly_score_payload)
        )
    except (TypeError, ValueError):
        max_anomaly_score = None
    return PromptDatasetSummary(
        total_count=total_count,
        scope_counts=scope_counts,
        alert_count=alert_count,
        labeled_count=labeled_count,
        max_anomaly_score=max_anomaly_score,
        preview_prompt_ids=preview_ids,
    )


def _summarize_prompt_records(prompt_records: Sequence[PromptInput]) -> PromptDatasetSummary:
    """Derive a prompt dataset summary from a loaded record sequence."""

    scope_counts: dict[str, int] = {}
    alert_count = 0
    labeled_count = 0
    max_score: float | None = None
    for record in prompt_records:
        scope_counts[record.scope] = scope_counts.get(record.scope, 0) + 1
        if record.is_alert:
            alert_count += 1
        if record.label is not None:
            labeled_count += 1
        if max_score is None or record.anomaly_score > max_score:
            max_score = float(record.anomaly_score)
    preview_prompt_ids = tuple(
        record.prompt_id
        for record in sorted(
            prompt_records,
            key=lambda item: (-float(item.anomaly_score), item.prompt_id),
        )[:3]
    )
    return PromptDatasetSummary(
        total_count=len(prompt_records),
        scope_counts=scope_counts,
        alert_count=alert_count,
        labeled_count=labeled_count,
        max_anomaly_score=max_score,
        preview_prompt_ids=preview_prompt_ids,
    )


def _resolve_manifest_path(path: Path) -> Path:
    """Resolve a prompt dataset manifest path from a path or bundle directory."""

    if path.is_file():
        if path.name == "manifest.json":
            return path
        if path.name in {"prompt_inputs.jsonl", "prompt_inputs.csv"}:
            manifest_path = path.parent / "manifest.json"
            if manifest_path.exists():
                return manifest_path
            raise FileNotFoundError(
                f"Prompt dataset manifest not found next to record file: {path}"
            )
        raise FileNotFoundError(
            f"Expected a prompt dataset directory or manifest.json, got file: {path}"
        )
    direct_manifest = path / "manifest.json"
    if direct_manifest.exists():
        return direct_manifest
    manifests = sorted(path.rglob("manifest.json"))
    if len(manifests) == 1:
        return manifests[0]
    if len(manifests) > 1:
        raise ValueError(
            "Multiple prompt dataset manifests were found. Please pass a specific "
            f"run directory or manifest path: {', '.join(str(item) for item in manifests)}"
        )
    raise FileNotFoundError(
        f"Prompt dataset manifest.json was not found under: {path}"
    )


def _resolve_artifact_path(manifest_dir: Path, artifact_path: object) -> Path | None:
    """Resolve an artifact path from the manifest into an absolute path."""

    if not artifact_path:
        return None
    candidate = Path(str(artifact_path))
    if not candidate.is_absolute():
        candidate = manifest_dir / candidate
    return candidate


def _candidate_record_paths(
    manifest_path: Path,
    manifest_payload: Mapping[str, object],
    source_path: Path,
) -> list[Path]:
    """Return candidate record files in the preferred load order."""

    manifest_dir = manifest_path.parent
    candidates: list[Path] = []
    if source_path.is_file() and source_path.name in {"prompt_inputs.jsonl", "prompt_inputs.csv"}:
        candidates.append(source_path)
    artifact_paths = manifest_payload.get("artifact_paths", {})
    if isinstance(artifact_paths, Mapping):
        for key in ("jsonl", "csv"):
            candidate = _resolve_artifact_path(manifest_dir, artifact_paths.get(key))
            if candidate is not None:
                candidates.append(candidate)
    candidates.extend(
        [
            manifest_dir / "prompt_inputs.jsonl",
            manifest_dir / "prompt_inputs.csv",
        ]
    )
    deduplicated: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        token = candidate.as_posix()
        if token not in seen:
            seen.add(token)
            deduplicated.append(candidate)
    return deduplicated


def _parse_prompt_record_row(row: Mapping[str, object]) -> PromptInput:
    """Convert a row from JSONL or CSV into a typed prompt input."""

    payload = {str(key): row.get(key) for key in PROMPT_INPUT_FIELDS}
    structured_context = _coerce_mapping(payload.get("structured_context"))
    rule_summary = _coerce_mapping(payload.get("rule_summary"))
    return PromptInput(
        prompt_id=str(_coerce_scalar(payload.get("prompt_id")) or ""),
        run_id=str(_coerce_scalar(payload.get("run_id")) or ""),
        sample_id=str(_coerce_scalar(payload.get("sample_id")) or ""),
        scope=_coerce_scope(payload.get("scope")),
        anomaly_score=float(_coerce_optional_float(payload.get("anomaly_score")) or 0.0),
        threshold=_coerce_optional_float(payload.get("threshold")),
        is_alert=_coerce_optional_bool(payload.get("is_alert")),
        alert_level=(
            None
            if _coerce_scalar(payload.get("alert_level")) in {None, ""}
            else str(_coerce_scalar(payload.get("alert_level")))
        ),
        label=_coerce_label(payload.get("label")),
        structured_context=structured_context,
        rule_summary=rule_summary,
        prompt_text=str(_coerce_scalar(payload.get("prompt_text")) or ""),
    )


def _load_prompt_records_jsonl(path: Path) -> list[PromptInput]:
    """Load prompt records from a JSON Lines file."""

    prompt_records: list[PromptInput] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"Expected JSON object on line {line_number} of {path}, got {type(payload).__name__}"
                )
            prompt_records.append(_parse_prompt_record_row(payload))
    return prompt_records


def _load_prompt_records_csv(path: Path) -> list[PromptInput]:
    """Load prompt records from a CSV file."""

    prompt_records: list[PromptInput] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV prompt dataset has no header row: {path}")
        for row in reader:
            prompt_records.append(_parse_prompt_record_row(row))
    return prompt_records


def _load_prompt_records_with_fallback(
    candidate_paths: Sequence[Path],
) -> tuple[list[PromptInput], dict[str, str], list[str], str]:
    """Load prompt records using JSONL first and CSV as a fallback."""

    notes: list[str] = []
    loaded_files: dict[str, str] = {}
    failures: list[str] = []
    for candidate in candidate_paths:
        if not candidate.exists():
            continue
        try:
            if candidate.suffix.lower() == ".jsonl":
                prompt_records = _load_prompt_records_jsonl(candidate)
            elif candidate.suffix.lower() == ".csv":
                prompt_records = _load_prompt_records_csv(candidate)
            else:
                continue
        except Exception as exc:  # pragma: no cover - the fallback path is what matters in tests
            failures.append(f"{candidate.as_posix()}: {exc}")
            continue
        loaded_files["prompt_inputs"] = candidate.as_posix()
        if candidate.suffix.lower() == ".csv":
            notes.append("Loaded prompt records from CSV fallback.")
        return prompt_records, loaded_files, notes, candidate.suffix.lower().lstrip(".")
    if failures:
        raise ValueError(
            "Failed to load prompt records from the available files: "
            + "; ".join(failures)
        )
    checked = ", ".join(path.as_posix() for path in candidate_paths)
    raise FileNotFoundError(
        f"No prompt record file was found. Checked: {checked}"
    )


def _build_manifest_info(
    *,
    manifest_path: Path,
    manifest_payload: Mapping[str, object],
    selected_summary: PromptDatasetSummary,
) -> PromptDatasetManifestInfo:
    """Convert a manifest payload into a typed prompt dataset manifest view."""

    prompt_input_fields_payload = manifest_payload.get("prompt_input_fields", PROMPT_INPUT_FIELDS)
    if isinstance(prompt_input_fields_payload, Sequence) and not isinstance(prompt_input_fields_payload, (str, bytes)):
        prompt_input_fields = tuple(str(item) for item in prompt_input_fields_payload)
    else:
        prompt_input_fields = tuple(PROMPT_INPUT_FIELDS)
    artifact_paths_payload = manifest_payload.get("artifact_paths", {})
    artifact_paths: dict[str, str] = {}
    if isinstance(artifact_paths_payload, Mapping):
        for key, value in artifact_paths_payload.items():
            resolved = _resolve_artifact_path(manifest_path.parent, value)
            if resolved is not None:
                artifact_paths[str(key)] = resolved.as_posix()
    row_counts_payload = manifest_payload.get("row_counts", {})
    row_counts: dict[str, int] = {}
    if isinstance(row_counts_payload, Mapping):
        for key, value in row_counts_payload.items():
            try:
                row_counts[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
    notes_payload = manifest_payload.get("notes", ())
    notes = tuple(
        str(item)
        for item in notes_payload
        if str(item).strip()
    ) if isinstance(notes_payload, Sequence) and not isinstance(notes_payload, (str, bytes)) else ()
    formats_payload = manifest_payload.get("formats", ())
    formats = (
        tuple(str(item) for item in formats_payload)
        if isinstance(formats_payload, Sequence) and not isinstance(formats_payload, (str, bytes))
        else ()
    )
    metadata_payload = manifest_payload.get("metadata", {})
    metadata = dict(metadata_payload) if isinstance(metadata_payload, Mapping) else {}
    return PromptDatasetManifestInfo(
        dataset_id=str(manifest_payload.get("dataset_id", "")),
        run_id=str(manifest_payload.get("run_id", "")),
        timestamp=str(manifest_payload.get("timestamp", "")),
        manifest_path=manifest_path.as_posix(),
        base_directory=str(manifest_payload.get("base_directory", manifest_path.parent.parent.as_posix())),
        run_directory=str(manifest_payload.get("run_directory", manifest_path.parent.as_posix())),
        scope=_coerce_scope(manifest_payload.get("scope", "flow")),
        selection_mode=_coerce_selection_mode(manifest_payload.get("selection_mode", "all")),
        only_alerts=_coerce_optional_bool(manifest_payload.get("only_alerts")) is True,
        balanced=_coerce_optional_bool(manifest_payload.get("balanced")) is True,
        top_k=_coerce_optional_int(manifest_payload.get("top_k")),
        max_samples=_coerce_optional_int(manifest_payload.get("max_samples")),
        source_sample_count=_coerce_optional_int(manifest_payload.get("source_sample_count")) or 0,
        selected_sample_count=_coerce_optional_int(manifest_payload.get("selected_sample_count"))
        or selected_summary.total_count,
        prompt_input_fields=prompt_input_fields,
        formats=formats,
        artifact_paths=artifact_paths,
        row_counts=row_counts,
        notes=notes,
        metadata=metadata,
        raw_manifest=dict(manifest_payload),
    )


def load_prompt_dataset(path: str | Path) -> PromptDatasetReplay:
    """Load a replayable prompt dataset from a directory or manifest path."""

    source_path = Path(path)
    manifest_path = _resolve_manifest_path(source_path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest_payload = json.load(handle)
    if not isinstance(manifest_payload, Mapping):
        raise ValueError(
            f"Prompt dataset manifest must contain a JSON object: {manifest_path}"
        )

    candidate_paths = _candidate_record_paths(
        manifest_path,
        manifest_payload,
        source_path,
    )
    prompt_records, loaded_files, record_notes, loaded_format = (
        _load_prompt_records_with_fallback(candidate_paths)
    )
    manifest_summary = _summary_from_payload(manifest_payload.get("summary"))
    computed_summary = _summarize_prompt_records(prompt_records)
    notes = list(record_notes)
    manifest_notes = manifest_payload.get("notes", ())
    if isinstance(manifest_notes, Sequence) and not isinstance(manifest_notes, (str, bytes)):
        for note in manifest_notes:
            text = str(note).strip()
            if text and text not in notes:
                notes.append(text)
    if manifest_summary.total_count != computed_summary.total_count:
        notes.append(
            "Manifest summary count did not match loaded prompt records; using computed counts."
        )
        selection_summary = computed_summary
    else:
        selection_summary = manifest_summary

    manifest_info = _build_manifest_info(
        manifest_path=manifest_path,
        manifest_payload=manifest_payload,
        selected_summary=selection_summary,
    )
    loaded_files["manifest_json"] = manifest_path.as_posix()
    loaded_files["prompt_records_format"] = loaded_format
    if selection_summary.preview_prompt_ids:
        notes.append(
            f"Loaded {selection_summary.total_count} prompt records from {loaded_format.upper()}."
        )
    return PromptDatasetReplay(
        dataset_id=manifest_info.dataset_id,
        run_id=manifest_info.run_id,
        timestamp=manifest_info.timestamp,
        manifest=manifest_info,
        selection_summary=selection_summary,
        prompt_records=tuple(prompt_records),
        prompt_index={prompt_record.prompt_id: prompt_record for prompt_record in prompt_records},
        loaded_files=loaded_files,
        notes=tuple(notes),
    )


def list_prompt_records(dataset: PromptDatasetReplay) -> list[PromptInput]:
    """Return the prompt records in file order."""

    return list(dataset.prompt_records)


def get_prompt_record(dataset: PromptDatasetReplay, prompt_id: str) -> PromptInput:
    """Return one prompt record by id."""

    try:
        return dataset.prompt_index[prompt_id]
    except KeyError as exc:
        raise KeyError(f"Prompt record not found for prompt_id={prompt_id}") from exc


def filter_prompt_records(
    dataset: PromptDatasetReplay,
    *,
    scope: PromptScope | None = None,
    only_alerts: bool | None = None,
    top_k: int | None = None,
) -> list[PromptInput]:
    """Filter prompt records by scope, alert status, and optional top-k score ranking."""

    records = list(dataset.prompt_records)
    if scope is not None:
        records = [record for record in records if record.scope == scope]
    if only_alerts is True:
        records = [record for record in records if record.is_alert is True]
    elif only_alerts is False:
        records = [record for record in records if record.is_alert is False]
    if top_k is not None:
        if top_k <= 0:
            return []
        records = sorted(
            records,
            key=lambda item: (-float(item.anomaly_score), item.prompt_id),
        )[:top_k]
    return records


def summarize_prompt_dataset(dataset: PromptDatasetReplay) -> PromptDatasetSummary:
    """Return the prompt dataset selection summary."""

    return dataset.selection_summary


def summarize_prompt_dataset_text(dataset: PromptDatasetReplay) -> str:
    """Render a compact human-readable prompt dataset summary."""

    summary = dataset.selection_summary
    lines = [
        f"Prompt dataset: id={dataset.dataset_id}, run_id={dataset.run_id}, timestamp={dataset.timestamp}",
        (
            "Manifest: "
            f"scope={dataset.manifest.scope}, "
            f"selection_mode={dataset.manifest.selection_mode}, "
            f"only_alerts={dataset.manifest.only_alerts}, "
            f"balanced={dataset.manifest.balanced}"
        ),
        (
            "Counts: "
            f"total={summary.total_count}, "
            f"alerts={summary.alert_count}, "
            f"labeled={summary.labeled_count}"
        ),
    ]
    if summary.max_anomaly_score is not None:
        lines.append(f"Max anomaly score: {summary.max_anomaly_score:.6f}")
    if summary.scope_counts:
        lines.append(
            "Scope counts: "
            + ", ".join(
                f"{scope}={count}" for scope, count in sorted(summary.scope_counts.items())
            )
        )
    if summary.preview_prompt_ids:
        lines.append("Preview prompt ids: " + ", ".join(summary.preview_prompt_ids))
    if dataset.loaded_files:
        lines.append(
            "Loaded files: "
            + ", ".join(
                f"{name}={path}" for name, path in sorted(dataset.loaded_files.items())
            )
        )
    if dataset.notes:
        lines.append("Notes:")
        lines.extend(f"  - {note}" for note in dataset.notes)
    return "\n".join(lines)


__all__ = [
    "filter_prompt_records",
    "get_prompt_record",
    "list_prompt_records",
    "load_prompt_dataset",
    "summarize_prompt_dataset",
    "summarize_prompt_dataset_text",
]
