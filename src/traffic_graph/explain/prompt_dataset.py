"""Batch construction helpers for LLM-ready prompt datasets."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from traffic_graph.explain.explanation_samples import (
    select_balanced_samples_for_explanation,
    sort_samples_by_score,
)
from traffic_graph.explain.explanation_types import ExplanationSample
from traffic_graph.explain.prompt_builder import build_prompt_inputs
from traffic_graph.explain.prompt_types import (
    PromptDatasetArtifact,
    PromptDatasetSelectionMode,
    PromptDatasetSummary,
    PromptInput,
    PromptScope,
)
from traffic_graph.explain.rule_records import RuleRecord
from traffic_graph.pipeline.replay_types import ReplayAlertRecord, ReplayScoreRecord


def _derive_run_id(samples: Sequence[ExplanationSample]) -> str:
    """Derive a stable run id from a prompt dataset sample collection."""

    run_ids = sorted({sample.run_id for sample in samples if sample.run_id})
    if len(run_ids) == 1:
        return run_ids[0]
    if len(run_ids) > 1:
        return "mixed"
    return "prompt_dataset"


def _selection_mode(only_alerts: bool, balanced: bool) -> PromptDatasetSelectionMode:
    """Map selection flags to the stable dataset selection mode."""

    if balanced:
        return "balanced"
    if only_alerts:
        return "alerts"
    return "all"


def _filter_samples_by_scope(
    samples: Sequence[ExplanationSample],
    scope: PromptScope,
) -> list[ExplanationSample]:
    """Keep only explanation samples from one scope."""

    return [sample for sample in samples if sample.scope == scope]


def select_prompt_samples(
    samples: Sequence[ExplanationSample],
    *,
    scope: PromptScope,
    only_alerts: bool = False,
    top_k: int | None = None,
    balanced: bool = False,
    max_samples: int = 50,
) -> list[ExplanationSample]:
    """Select a stable subset of samples for prompt generation."""

    scoped_samples = _filter_samples_by_scope(samples, scope)
    if balanced:
        selected_samples = select_balanced_samples_for_explanation(
            scoped_samples,
            max_samples=max_samples,
        )
    else:
        if only_alerts:
            selected_samples = [sample for sample in scoped_samples if sample.is_alert]
        else:
            selected_samples = list(scoped_samples)
        selected_samples = sort_samples_by_score(selected_samples)
        if top_k is not None and top_k >= 0:
            selected_samples = selected_samples[:top_k]
    if balanced and top_k is not None and top_k >= 0:
        selected_samples = sort_samples_by_score(selected_samples)[:top_k]
    return selected_samples


def summarize_prompt_dataset(
    prompt_inputs: Sequence[PromptInput],
) -> PromptDatasetSummary:
    """Summarize a prompt dataset for reporting and manifest generation."""

    scope_counts = Counter(prompt.scope for prompt in prompt_inputs)
    alert_count = sum(1 for prompt in prompt_inputs if prompt.is_alert)
    labeled_count = sum(1 for prompt in prompt_inputs if prompt.label is not None)
    max_anomaly_score = max(
        (prompt.anomaly_score for prompt in prompt_inputs),
        default=None,
    )
    preview_prompt_ids = tuple(
        prompt.prompt_id
        for prompt in sort_prompt_inputs_by_score(prompt_inputs)[:3]
    )
    return PromptDatasetSummary(
        total_count=len(prompt_inputs),
        scope_counts={str(scope): int(count) for scope, count in scope_counts.items()},
        alert_count=alert_count,
        labeled_count=labeled_count,
        max_anomaly_score=max_anomaly_score,
        preview_prompt_ids=preview_prompt_ids,
    )


def sort_prompt_inputs_by_score(
    prompt_inputs: Sequence[PromptInput],
    *,
    descending: bool = True,
) -> list[PromptInput]:
    """Sort prompt inputs by anomaly score in a deterministic order."""

    return sorted(
        prompt_inputs,
        key=lambda prompt: (prompt.anomaly_score, prompt.prompt_id),
        reverse=descending,
    )


def summarize_prompt_dataset_text(
    prompt_dataset: PromptDatasetArtifact | Sequence[PromptInput],
) -> str:
    """Render a compact human-readable prompt dataset summary."""

    if isinstance(prompt_dataset, PromptDatasetArtifact):
        artifact = prompt_dataset
        prompt_inputs = artifact.prompt_inputs
        summary = artifact.summary
        lines = [
            f"Prompt dataset: id={artifact.dataset_id}, run_id={artifact.run_id}, scope={artifact.scope}",
            (
                "Selection: "
                f"mode={artifact.selection_mode}, "
                f"only_alerts={artifact.only_alerts}, "
                f"balanced={artifact.balanced}, "
                f"top_k={artifact.top_k if artifact.top_k is not None else 'n/a'}, "
                f"max_samples={artifact.max_samples if artifact.max_samples is not None else 'n/a'}"
            ),
        ]
    else:
        prompt_inputs = list(prompt_dataset)
        summary = summarize_prompt_dataset(prompt_inputs)
        lines = [
            "Prompt dataset:",
        ]
    lines.extend(
        [
            f"Total prompts: {summary.total_count}",
            f"Alert prompts: {summary.alert_count}",
            f"Labeled prompts: {summary.labeled_count}",
        ]
    )
    if summary.max_anomaly_score is not None:
        lines.append(f"Max anomaly score: {summary.max_anomaly_score:.6f}")
    if summary.scope_counts:
        lines.append(
            "Scope counts: "
            + ", ".join(
                f"{scope}={count}" for scope, count in sorted(summary.scope_counts.items())
            )
        )
    preview_prompt_ids = list(summary.preview_prompt_ids)
    if preview_prompt_ids:
        lines.append("Preview prompt ids: " + ", ".join(preview_prompt_ids))
    if prompt_inputs:
        top_prompt = sort_prompt_inputs_by_score(prompt_inputs)[0]
        lines.append(
            "Top prompt: "
            f"{top_prompt.prompt_id} score={top_prompt.anomaly_score:.6f} "
            f"alert={top_prompt.is_alert}"
        )
    return "\n".join(lines)


def build_prompt_dataset(
    samples: Sequence[ExplanationSample],
    rule_records: Sequence[RuleRecord] | Mapping[str, RuleRecord],
    *,
    scope: PromptScope,
    only_alerts: bool = False,
    top_k: int | None = None,
    balanced: bool = False,
    max_samples: int = 50,
    alert_records: Sequence[ReplayAlertRecord]
    | Mapping[tuple[str, object, object, object], ReplayAlertRecord]
    | None = None,
    score_records: Sequence[ReplayScoreRecord]
    | Mapping[tuple[str, object, object, object], ReplayScoreRecord]
    | None = None,
) -> PromptDatasetArtifact:
    """Build a prompt dataset artifact from explanation samples and rules."""

    selected_samples = select_prompt_samples(
        samples,
        scope=scope,
        only_alerts=only_alerts,
        top_k=top_k,
        balanced=balanced,
        max_samples=max_samples,
    )
    prompt_inputs = tuple(
        build_prompt_inputs(
            selected_samples,
            rule_records,
            alert_records=alert_records,
            score_records=score_records,
        )
    )
    summary = summarize_prompt_dataset(prompt_inputs)
    selection_mode = _selection_mode(only_alerts=only_alerts, balanced=balanced)
    run_id = _derive_run_id(selected_samples or samples)
    dataset_id = f"prompt-dataset:{run_id}:{scope}:{selection_mode}"
    notes: list[str] = []
    if not prompt_inputs:
        notes.append("no prompt inputs were selected for export")
    if len({sample.run_id for sample in samples if sample.run_id}) > 1:
        notes.append("multiple run ids were detected in the source samples")
    return PromptDatasetArtifact(
        dataset_id=dataset_id,
        run_id=run_id,
        scope=scope,
        selection_mode=selection_mode,
        only_alerts=only_alerts,
        balanced=balanced,
        top_k=top_k,
        max_samples=max_samples if balanced else None,
        source_sample_count=len(samples),
        selected_sample_count=len(selected_samples),
        prompt_inputs=prompt_inputs,
        summary=summary,
        notes=tuple(notes),
        metadata={
            "scope": scope,
            "selection_mode": selection_mode,
            "only_alerts": only_alerts,
            "balanced": balanced,
            "top_k": top_k,
            "max_samples": max_samples,
            "source_sample_count": len(samples),
            "selected_sample_count": len(selected_samples),
        },
    )


__all__ = [
    "build_prompt_dataset",
    "select_prompt_samples",
    "sort_prompt_inputs_by_score",
    "summarize_prompt_dataset",
    "summarize_prompt_dataset_text",
]
