"""Typed prompt-input structures for downstream LLM explanation workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


PromptScope = Literal["graph", "flow", "node"]
"""Supported prompt scopes for LLM-ready explanation inputs."""

PromptDatasetSelectionMode = Literal["all", "alerts", "balanced"]
"""Supported selection strategies when batching prompt inputs."""

PROMPT_BASIC_INFO_FIELDS: tuple[str, ...] = (
    "graph_id",
    "window_id",
    "flow_id",
    "node_id",
)
"""Stable field order for sample identity context."""

PROMPT_CONTEXT_FIELDS: tuple[str, ...] = (
    "sample_basic_info",
    "statistics_summary",
    "graph_summary",
    "score_summary",
    "alert_summary",
    "feature_summary",
)
"""Stable field order for structured prompt context sections."""

PROMPT_SCORE_SUMMARY_FIELDS: tuple[str, ...] = (
    "score_scope",
    "anomaly_score",
    "threshold",
    "is_alert",
    "alert_level",
    "label",
)
"""Stable field order for score summaries inside the prompt context."""

PROMPT_ALERT_SUMMARY_FIELDS: tuple[str, ...] = (
    "alert_id",
    "alert_scope",
    "alert_level",
    "anomaly_score",
    "threshold",
    "is_alert",
    "label",
)
"""Stable field order for alert summaries inside the prompt context."""

PROMPT_RULE_SUMMARY_FIELDS: tuple[str, ...] = (
    "rule_id",
    "sample_id",
    "scope",
    "tree_mode",
    "predicted_score_or_class",
    "leaf_id",
    "path_conditions",
    "feature_names_used",
)
"""Stable field order for the surrogate-tree rule summary."""

PROMPT_INPUT_FIELDS: tuple[str, ...] = (
    "prompt_id",
    "run_id",
    "sample_id",
    "scope",
    "anomaly_score",
    "threshold",
    "is_alert",
    "alert_level",
    "label",
    "structured_context",
    "rule_summary",
    "prompt_text",
)
"""Stable field order for serialized prompt inputs."""


@dataclass(frozen=True, slots=True)
class PromptDatasetSummary:
    """Compact summary for a batch of prompt inputs."""

    total_count: int
    scope_counts: dict[str, int] = field(default_factory=dict)
    alert_count: int = 0
    labeled_count: int = 0
    max_anomaly_score: float | None = None
    preview_prompt_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize the prompt dataset summary into a stable dictionary."""

        return {
            "total_count": self.total_count,
            "scope_counts": dict(self.scope_counts),
            "alert_count": self.alert_count,
            "labeled_count": self.labeled_count,
            "max_anomaly_score": self.max_anomaly_score,
            "preview_prompt_ids": list(self.preview_prompt_ids),
        }


@dataclass(frozen=True, slots=True)
class PromptDatasetArtifact:
    """Structured collection of prompt inputs ready for export."""

    dataset_id: str
    run_id: str
    scope: PromptScope
    selection_mode: PromptDatasetSelectionMode
    only_alerts: bool
    balanced: bool
    top_k: int | None
    max_samples: int | None
    source_sample_count: int
    selected_sample_count: int
    prompt_inputs: tuple[PromptInput, ...] = ()
    summary: PromptDatasetSummary = field(
        default_factory=lambda: PromptDatasetSummary(total_count=0)
    )
    notes: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self, *, include_prompt_inputs: bool = False) -> dict[str, object]:
        """Serialize the prompt dataset artifact into a JSON-friendly dictionary."""

        payload: dict[str, object] = {
            "dataset_id": self.dataset_id,
            "run_id": self.run_id,
            "scope": self.scope,
            "selection_mode": self.selection_mode,
            "only_alerts": self.only_alerts,
            "balanced": self.balanced,
            "top_k": self.top_k,
            "max_samples": self.max_samples,
            "source_sample_count": self.source_sample_count,
            "selected_sample_count": self.selected_sample_count,
            "summary": self.summary.to_dict(),
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
        }
        if include_prompt_inputs:
            payload["prompt_inputs"] = [prompt_input.to_dict() for prompt_input in self.prompt_inputs]
        return payload


@dataclass(frozen=True, slots=True)
class PromptInput:
    """LLM-ready explanation prompt input derived from structured artifacts."""

    prompt_id: str
    run_id: str
    sample_id: str
    scope: PromptScope
    anomaly_score: float
    threshold: float | None
    is_alert: bool | None
    alert_level: str | None = None
    label: object | None = None
    structured_context: dict[str, object] = field(default_factory=dict)
    rule_summary: dict[str, object] = field(default_factory=dict)
    prompt_text: str = ""

    def to_dict(self) -> dict[str, object]:
        """Serialize the prompt input into a JSON-friendly dictionary."""

        return {
            "prompt_id": self.prompt_id,
            "run_id": self.run_id,
            "sample_id": self.sample_id,
            "scope": self.scope,
            "anomaly_score": self.anomaly_score,
            "threshold": self.threshold,
            "is_alert": self.is_alert,
            "alert_level": self.alert_level,
            "label": self.label,
            "structured_context": dict(self.structured_context),
            "rule_summary": dict(self.rule_summary),
            "prompt_text": self.prompt_text,
        }


__all__ = [
    "PROMPT_ALERT_SUMMARY_FIELDS",
    "PROMPT_BASIC_INFO_FIELDS",
    "PROMPT_CONTEXT_FIELDS",
    "PROMPT_INPUT_FIELDS",
    "PROMPT_RULE_SUMMARY_FIELDS",
    "PROMPT_SCORE_SUMMARY_FIELDS",
    "PromptDatasetArtifact",
    "PromptDatasetSelectionMode",
    "PromptDatasetSummary",
    "PromptInput",
    "PromptScope",
]
