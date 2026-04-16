"""Typed readback views for exported prompt datasets."""

from __future__ import annotations

from dataclasses import dataclass, field

from traffic_graph.explain.prompt_types import (
    PromptDatasetSelectionMode,
    PromptDatasetSummary,
    PromptInput,
    PromptScope,
)


@dataclass(frozen=True, slots=True)
class PromptDatasetManifestInfo:
    """Structured manifest metadata for one exported prompt dataset."""

    dataset_id: str
    run_id: str
    timestamp: str
    manifest_path: str
    base_directory: str
    run_directory: str
    scope: PromptScope
    selection_mode: PromptDatasetSelectionMode
    only_alerts: bool
    balanced: bool
    top_k: int | None
    max_samples: int | None
    source_sample_count: int
    selected_sample_count: int
    prompt_input_fields: tuple[str, ...]
    formats: tuple[str, ...] = ()
    artifact_paths: dict[str, str] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)
    raw_manifest: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize the manifest info into a JSON-friendly dictionary."""

        return {
            "dataset_id": self.dataset_id,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "manifest_path": self.manifest_path,
            "base_directory": self.base_directory,
            "run_directory": self.run_directory,
            "scope": self.scope,
            "selection_mode": self.selection_mode,
            "only_alerts": self.only_alerts,
            "balanced": self.balanced,
            "top_k": self.top_k,
            "max_samples": self.max_samples,
            "source_sample_count": self.source_sample_count,
            "selected_sample_count": self.selected_sample_count,
            "prompt_input_fields": list(self.prompt_input_fields),
            "formats": list(self.formats),
            "artifact_paths": dict(self.artifact_paths),
            "row_counts": dict(self.row_counts),
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
            "raw_manifest": dict(self.raw_manifest),
        }


@dataclass(frozen=True, slots=True)
class PromptDatasetReplay:
    """Fully loaded readback view over one exported prompt dataset."""

    dataset_id: str
    run_id: str
    timestamp: str
    manifest: PromptDatasetManifestInfo
    selection_summary: PromptDatasetSummary
    prompt_records: tuple[PromptInput, ...] = ()
    prompt_index: dict[str, PromptInput] = field(default_factory=dict)
    loaded_files: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self, *, include_prompt_records: bool = False) -> dict[str, object]:
        """Serialize the replay view into a compact dictionary representation."""

        payload: dict[str, object] = {
            "dataset_id": self.dataset_id,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "manifest": self.manifest.to_dict(),
            "selection_summary": self.selection_summary.to_dict(),
            "loaded_files": dict(self.loaded_files),
            "notes": list(self.notes),
        }
        if include_prompt_records:
            payload["prompt_records"] = [prompt_record.to_dict() for prompt_record in self.prompt_records]
        return payload


PromptDatasetView = PromptDatasetReplay
"""Alias retained for callers that prefer a view-style name."""


__all__ = [
    "PromptDatasetManifestInfo",
    "PromptDatasetReplay",
    "PromptDatasetView",
]
