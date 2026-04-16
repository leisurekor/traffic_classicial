"""Typed result structures for LLM-style batch explanation outputs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

from traffic_graph.explain.prompt_types import PromptScope


LLMResultStatus = Literal["success", "failed", "skipped"]
"""Supported statuses for a batch LLM result record."""

LLM_RESULT_STATUS_VALUES: tuple[LLMResultStatus, ...] = (
    "success",
    "failed",
    "skipped",
)
"""Stable status ordering used in summaries and exports."""

LLM_RESULT_FIELDS: tuple[str, ...] = (
    "response_id",
    "prompt_id",
    "run_id",
    "model_name",
    "response_text",
    "raw_response",
    "status",
    "error_message",
    "created_at",
)
"""Stable field order for serialized LLM result records."""

LLM_RESULT_SUMMARY_FIELDS: tuple[str, ...] = (
    "total_count",
    "status_counts",
    "success_count",
    "failed_count",
    "skipped_count",
    "model_name",
    "created_at",
    "source_prompt_dataset_id",
    "source_prompt_dataset_scope",
    "preview_prompt_ids",
    "preview_response_ids",
)
"""Stable field order for serialized LLM result summaries."""

LLM_RESULT_ARTIFACT_FIELDS: tuple[str, ...] = (
    "result_id",
    "run_id",
    "model_name",
    "created_at",
    "source_prompt_dataset_id",
    "source_prompt_dataset_run_id",
    "source_prompt_dataset_timestamp",
    "source_prompt_dataset_scope",
    "source_prompt_dataset_selection_mode",
    "result_count",
    "summary",
    "notes",
    "metadata",
)
"""Stable field order for serialized LLM result artifacts."""


def _json_scalar(value: object) -> object:
    """Convert common scalar-like values into JSON-friendly primitives."""

    if hasattr(value, "item"):
        try:
            return value.item()  # type: ignore[call-arg]
        except Exception:
            return value
    return value


@dataclass(frozen=True, slots=True)
class LLMResultRecord:
    """Structured response generated for one prompt input."""

    response_id: str
    prompt_id: str
    run_id: str
    model_name: str
    response_text: str
    raw_response: dict[str, object] | None = None
    status: LLMResultStatus = "success"
    error_message: str | None = None
    created_at: str = ""

    def to_dict(self) -> dict[str, object]:
        """Serialize the result record into a JSON-friendly dictionary."""

        return {
            "response_id": self.response_id,
            "prompt_id": self.prompt_id,
            "run_id": self.run_id,
            "model_name": self.model_name,
            "response_text": self.response_text,
            "raw_response": (
                dict(self.raw_response) if self.raw_response is not None else None
            ),
            "status": self.status,
            "error_message": self.error_message,
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class LLMResultSummary:
    """Compact summary for a batch of LLM result records."""

    total_count: int
    status_counts: dict[str, int] = field(default_factory=dict)
    success_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    model_name: str = ""
    created_at: str = ""
    source_prompt_dataset_id: str = ""
    source_prompt_dataset_scope: PromptScope = cast(PromptScope, "flow")
    preview_prompt_ids: tuple[str, ...] = ()
    preview_response_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize the summary into a stable dictionary."""

        return {
            "total_count": self.total_count,
            "status_counts": dict(self.status_counts),
            "success_count": self.success_count,
            "failed_count": self.failed_count,
            "skipped_count": self.skipped_count,
            "model_name": self.model_name,
            "created_at": self.created_at,
            "source_prompt_dataset_id": self.source_prompt_dataset_id,
            "source_prompt_dataset_scope": self.source_prompt_dataset_scope,
            "preview_prompt_ids": list(self.preview_prompt_ids),
            "preview_response_ids": list(self.preview_response_ids),
        }


@dataclass(frozen=True, slots=True)
class LLMResultArtifact:
    """Structured collection of LLM result records ready for export."""

    result_id: str
    run_id: str
    model_name: str
    created_at: str
    source_prompt_dataset_id: str
    source_prompt_dataset_run_id: str
    source_prompt_dataset_timestamp: str
    source_prompt_dataset_scope: PromptScope
    source_prompt_dataset_selection_mode: str
    result_count: int
    result_records: tuple[LLMResultRecord, ...] = ()
    summary: LLMResultSummary = field(
        default_factory=lambda: LLMResultSummary(
            total_count=0,
            model_name="",
            created_at="",
            source_prompt_dataset_id="",
            source_prompt_dataset_scope=cast(PromptScope, "flow"),
        )
    )
    notes: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self, *, include_result_records: bool = False) -> dict[str, object]:
        """Serialize the artifact into a JSON-friendly dictionary."""

        payload: dict[str, object] = {
            "result_id": self.result_id,
            "run_id": self.run_id,
            "model_name": self.model_name,
            "created_at": self.created_at,
            "source_prompt_dataset_id": self.source_prompt_dataset_id,
            "source_prompt_dataset_run_id": self.source_prompt_dataset_run_id,
            "source_prompt_dataset_timestamp": self.source_prompt_dataset_timestamp,
            "source_prompt_dataset_scope": self.source_prompt_dataset_scope,
            "source_prompt_dataset_selection_mode": self.source_prompt_dataset_selection_mode,
            "result_count": self.result_count,
            "summary": self.summary.to_dict(),
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
        }
        if include_result_records:
            payload["result_records"] = [
                result_record.to_dict() for result_record in self.result_records
            ]
        return payload


def summarize_llm_results(records_or_artifact: Sequence[LLMResultRecord] | LLMResultArtifact) -> LLMResultSummary:
    """Summarize a batch of LLM results or an artifact containing them."""

    if isinstance(records_or_artifact, LLMResultArtifact):
        return records_or_artifact.summary

    records = list(records_or_artifact)
    status_counts: dict[str, int] = {status: 0 for status in LLM_RESULT_STATUS_VALUES}
    for record in records:
        status_counts[record.status] = status_counts.get(record.status, 0) + 1
    preview_prompt_ids = tuple(record.prompt_id for record in records[:3])
    preview_response_ids = tuple(record.response_id for record in records[:3])
    success_count = status_counts.get("success", 0)
    failed_count = status_counts.get("failed", 0)
    skipped_count = status_counts.get("skipped", 0)
    model_name = records[0].model_name if records else ""
    created_at = records[0].created_at if records else ""
    return LLMResultSummary(
        total_count=len(records),
        status_counts=status_counts,
        success_count=success_count,
        failed_count=failed_count,
        skipped_count=skipped_count,
        model_name=model_name,
        created_at=created_at,
        preview_prompt_ids=preview_prompt_ids,
        preview_response_ids=preview_response_ids,
    )


def summarize_llm_results_text(records_or_artifact: Sequence[LLMResultRecord] | LLMResultArtifact) -> str:
    """Render a compact human-readable summary of LLM results."""

    summary = summarize_llm_results(records_or_artifact)
    lines = [
        f"LLM results: total={summary.total_count}",
        "Statuses: "
        + (
            ", ".join(
                f"{status}={summary.status_counts.get(status, 0)}"
                for status in LLM_RESULT_STATUS_VALUES
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
    "LLMResultArtifact",
    "LLMResultRecord",
    "LLMResultStatus",
    "LLMResultSummary",
    "LLM_RESULT_ARTIFACT_FIELDS",
    "LLM_RESULT_FIELDS",
    "LLM_RESULT_STATUS_VALUES",
    "LLM_RESULT_SUMMARY_FIELDS",
    "summarize_llm_results",
    "summarize_llm_results_text",
]
