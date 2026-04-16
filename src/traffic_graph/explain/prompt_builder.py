"""Build stable LLM-ready prompt inputs from explanation artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import cast

from traffic_graph.explain.explanation_types import ExplanationSample
from traffic_graph.explain.prompt_types import (
    PROMPT_ALERT_SUMMARY_FIELDS,
    PROMPT_BASIC_INFO_FIELDS,
    PROMPT_CONTEXT_FIELDS,
    PROMPT_INPUT_FIELDS,
    PROMPT_RULE_SUMMARY_FIELDS,
    PROMPT_SCORE_SUMMARY_FIELDS,
    PromptInput,
    PromptScope,
)
from traffic_graph.explain.rule_records import RuleRecord
from traffic_graph.pipeline.replay_types import ReplayAlertRecord, ReplayScoreRecord

PromptLookupKey = tuple[str, object, object, object]
"""Stable lookup key for aligning prompt context records by scope and ids."""


def _json_scalar(value: object) -> object:
    """Convert common scalar values into JSON-safe primitives."""

    if hasattr(value, "item"):
        try:
            return value.item()  # type: ignore[call-arg]
        except Exception:
            return value
    return value


def _sample_basic_info(sample: ExplanationSample) -> dict[str, object]:
    """Build the stable sample identity section for a prompt context."""

    return {
        "graph_id": sample.graph_id,
        "window_id": sample.window_id,
        "flow_id": sample.flow_id,
        "node_id": sample.node_id,
    }


def _score_summary_from_sample(
    sample: ExplanationSample,
    *,
    score_record: ReplayScoreRecord | None,
    alert_record: ReplayAlertRecord | None,
) -> dict[str, object]:
    """Construct a stable score summary section for the prompt context."""

    score_scope = score_record.score_scope if score_record is not None else sample.scope
    anomaly_score = (
        score_record.anomaly_score if score_record is not None else sample.anomaly_score
    )
    threshold = (
        score_record.threshold
        if score_record is not None and score_record.threshold is not None
        else sample.threshold
    )
    is_alert = (
        score_record.is_alert
        if score_record is not None and score_record.is_alert is not None
        else sample.is_alert
    )
    alert_level = alert_record.alert_level if alert_record is not None else sample.alert_level
    label = (
        score_record.label
        if score_record is not None and score_record.label is not None
        else sample.label
    )
    payload = {
        "score_scope": score_scope,
        "anomaly_score": anomaly_score,
        "threshold": threshold,
        "is_alert": is_alert,
        "alert_level": alert_level,
        "label": label,
    }
    return {
        key: _json_scalar(value) for key, value in payload.items()
    }


def _alert_summary(alert_record: ReplayAlertRecord | None) -> dict[str, object]:
    """Construct a stable alert summary section for the prompt context."""

    if alert_record is None:
        return {}
    payload = {
        "alert_id": alert_record.alert_id,
        "alert_scope": alert_record.alert_scope,
        "alert_level": alert_record.alert_level,
        "anomaly_score": alert_record.anomaly_score,
        "threshold": alert_record.threshold,
        "is_alert": alert_record.is_alert,
        "label": alert_record.label,
    }
    return {key: _json_scalar(value) for key, value in payload.items()}


def _summarize_mapping(mapping: Mapping[str, object]) -> dict[str, object]:
    """Convert a free-form mapping into a stable JSON-friendly summary."""

    return {
        str(key): _json_scalar(value) for key, value in sorted(mapping.items(), key=lambda item: str(item[0]))
    }


def build_structured_context(
    sample: ExplanationSample,
    _rule_record: RuleRecord,
    *,
    alert_record: ReplayAlertRecord | None = None,
    score_record: ReplayScoreRecord | None = None,
) -> dict[str, object]:
    """Build the structured evidence section used inside an LLM prompt."""

    context: dict[str, object] = {
        "sample_basic_info": {
            field: getattr(sample, field)
            for field in PROMPT_BASIC_INFO_FIELDS
        },
        "statistics_summary": _summarize_mapping(sample.stats_summary),
        "graph_summary": _summarize_mapping(sample.graph_summary),
        "score_summary": _score_summary_from_sample(
            sample,
            score_record=score_record,
            alert_record=alert_record,
        ),
        "alert_summary": _alert_summary(alert_record),
        "feature_summary": _summarize_mapping(sample.feature_summary),
    }
    return {key: context[key] for key in PROMPT_CONTEXT_FIELDS}


def build_rule_summary(rule_record: RuleRecord) -> dict[str, object]:
    """Build the structured surrogate-tree summary section for a prompt."""

    path_conditions = [
        {
            "feature_name": condition.feature_name,
            "operator": condition.operator,
            "threshold": condition.threshold,
            "sample_value": condition.sample_value,
            "tree_node_index": condition.tree_node_index,
        }
        for condition in rule_record.path_conditions
    ]
    summary = {
        "rule_id": rule_record.rule_id,
        "sample_id": rule_record.sample_id,
        "scope": rule_record.scope,
        "tree_mode": rule_record.tree_mode,
        "predicted_score_or_class": _json_scalar(
            rule_record.predicted_score_or_class
        ),
        "leaf_id": rule_record.leaf_id,
        "path_conditions": path_conditions,
        "feature_names_used": list(rule_record.feature_names_used),
    }
    return {key: summary[key] for key in PROMPT_RULE_SUMMARY_FIELDS}


def _format_json_block(payload: Mapping[str, object]) -> str:
    """Render a mapping as a deterministic indented JSON block."""

    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def build_prompt_text(
    *,
    prompt_id: str,
    run_id: str,
    sample_id: str,
    scope: PromptScope,
    anomaly_score: float,
    threshold: float | None,
    is_alert: bool | None,
    alert_level: str | None,
    label: object | None,
    structured_context: Mapping[str, object],
    rule_summary: Mapping[str, object],
) -> str:
    """Render the stable prompt text used by downstream LLM modules."""

    label_text = "n/a" if label is None else json.dumps(_json_scalar(label), ensure_ascii=False, default=str)
    lines = [
        "You are an analyst for an unsupervised malicious traffic detection system.",
        "Use only the structured evidence below to explain why this sample is anomalous.",
        "",
        "Task:",
        "1. Explain why the sample is considered anomalous.",
        "2. Identify the most important rule evidence from the surrogate tree.",
        "3. Suggest the most plausible attack behavior, or explain why it may be benign.",
        "4. Recommend concrete follow-up checks.",
        "",
        "Reference metadata:",
        f"- prompt_id: {prompt_id}",
        f"- run_id: {run_id}",
        f"- sample_id: {sample_id}",
        f"- scope: {scope}",
        f"- anomaly_score: {anomaly_score}",
        f"- threshold: {threshold if threshold is not None else 'n/a'}",
        f"- is_alert: {is_alert if is_alert is not None else 'n/a'}",
        f"- alert_level: {alert_level if alert_level is not None else 'n/a'}",
        f"- label: {label_text} (reference only; do not use as evidence)",
        "",
        "Structured context (JSON):",
        _format_json_block(structured_context),
        "",
        "Rule summary (JSON):",
        _format_json_block(rule_summary),
        "",
        "Response format:",
        "1. Anomaly rationale:",
        "2. Key rule evidence:",
        "3. Possible attack behavior:",
        "4. Follow-up checks:",
    ]
    return "\n".join(lines)


def build_prompt_input(
    sample: ExplanationSample,
    rule_record: RuleRecord,
    *,
    alert_record: ReplayAlertRecord | None = None,
    score_record: ReplayScoreRecord | None = None,
) -> PromptInput:
    """Build one LLM-ready prompt input from a sample, rule, and optional records."""

    structured_context = build_structured_context(
        sample,
        rule_record,
        alert_record=alert_record,
        score_record=score_record,
    )
    rule_summary = build_rule_summary(rule_record)
    anomaly_score = (
        score_record.anomaly_score
        if score_record is not None
        else sample.anomaly_score
    )
    threshold = (
        score_record.threshold
        if score_record is not None and score_record.threshold is not None
        else sample.threshold
    )
    is_alert = (
        score_record.is_alert
        if score_record is not None and score_record.is_alert is not None
        else sample.is_alert
    )
    alert_level = alert_record.alert_level if alert_record is not None else sample.alert_level
    label = (
        score_record.label
        if score_record is not None and score_record.label is not None
        else sample.label
    )
    prompt_id = f"prompt:{sample.sample_id}:{rule_record.tree_mode}:leaf-{rule_record.leaf_id}"
    prompt_text = build_prompt_text(
        prompt_id=prompt_id,
        run_id=sample.run_id,
        sample_id=sample.sample_id,
        scope=sample.scope,
        anomaly_score=anomaly_score,
        threshold=threshold,
        is_alert=is_alert,
        alert_level=alert_level,
        label=label,
        structured_context=structured_context,
        rule_summary=rule_summary,
    )
    return PromptInput(
        prompt_id=prompt_id,
        run_id=sample.run_id,
        sample_id=sample.sample_id,
        scope=sample.scope,
        anomaly_score=float(anomaly_score),
        threshold=threshold,
        is_alert=is_alert,
        alert_level=alert_level,
        label=label,
        structured_context=structured_context,
        rule_summary=rule_summary,
        prompt_text=prompt_text,
    )


def _sample_lookup_key(sample: ExplanationSample) -> PromptLookupKey:
    """Build a stable lookup key for score and alert records."""

    if sample.scope == "graph":
        entity_id: object = sample.graph_id
    elif sample.scope == "flow":
        entity_id = sample.flow_id
    else:
        entity_id = sample.node_id
    return (sample.scope, sample.graph_id, sample.window_id, entity_id)


def _record_lookup_key(
    *,
    scope: str,
    graph_id: object,
    window_id: object,
    entity_id: object,
) -> PromptLookupKey:
    """Build a stable lookup key from a replay score or alert record."""

    return (scope, graph_id, window_id, entity_id)


def _index_rules(
    rule_records: Sequence[RuleRecord] | Mapping[str, RuleRecord],
) -> dict[str, RuleRecord]:
    """Normalize rule-record inputs into a sample-id keyed mapping."""

    if isinstance(rule_records, Mapping):
        return {str(key): value for key, value in rule_records.items()}
    return {rule_record.sample_id: rule_record for rule_record in rule_records}


def _index_score_records(
    score_records: Sequence[ReplayScoreRecord] | Mapping[PromptLookupKey, ReplayScoreRecord] | None,
) -> dict[PromptLookupKey, ReplayScoreRecord]:
    """Normalize score-record inputs into a stable lookup mapping."""

    if score_records is None:
        return {}
    if isinstance(score_records, Mapping):
        return {
            cast(PromptLookupKey, key): value
            for key, value in score_records.items()
        }
    indexed: dict[PromptLookupKey, ReplayScoreRecord] = {}
    for record in score_records:
        if record.score_scope == "graph":
            entity_id: object = record.graph_id
        elif record.score_scope == "flow":
            entity_id = record.flow_id
        elif record.score_scope == "node":
            entity_id = record.node_id
        else:
            entity_id = record.edge_id
        indexed[_record_lookup_key(
            scope=record.score_scope,
            graph_id=record.graph_id,
            window_id=record.window_id,
            entity_id=entity_id,
        )] = record
    return indexed


def _index_alert_records(
    alert_records: Sequence[ReplayAlertRecord] | Mapping[PromptLookupKey, ReplayAlertRecord] | None,
) -> dict[PromptLookupKey, ReplayAlertRecord]:
    """Normalize alert-record inputs into a stable lookup mapping."""

    if alert_records is None:
        return {}
    if isinstance(alert_records, Mapping):
        return {
            cast(PromptLookupKey, key): value
            for key, value in alert_records.items()
        }
    indexed: dict[PromptLookupKey, ReplayAlertRecord] = {}
    for record in alert_records:
        if record.alert_scope == "graph":
            entity_id: object = record.graph_id
        elif record.alert_scope == "flow":
            entity_id = record.flow_id
        elif record.alert_scope == "node":
            entity_id = record.node_id
        else:
            entity_id = record.edge_id
        indexed[_record_lookup_key(
            scope=record.alert_scope,
            graph_id=record.graph_id,
            window_id=record.window_id,
            entity_id=entity_id,
        )] = record
    return indexed


def build_prompt_inputs(
    samples: Sequence[ExplanationSample],
    rule_records: Sequence[RuleRecord] | Mapping[str, RuleRecord],
    *,
    alert_records: Sequence[ReplayAlertRecord] | Mapping[PromptLookupKey, ReplayAlertRecord] | None = None,
    score_records: Sequence[ReplayScoreRecord] | Mapping[PromptLookupKey, ReplayScoreRecord] | None = None,
) -> list[PromptInput]:
    """Build prompt inputs for a batch of explanation samples."""

    rule_lookup = _index_rules(rule_records)
    alert_lookup = _index_alert_records(alert_records)
    score_lookup = _index_score_records(score_records)
    prompt_inputs: list[PromptInput] = []
    for sample in samples:
        rule_record = rule_lookup.get(sample.sample_id)
        if rule_record is None:
            raise KeyError(f"No rule record found for sample_id={sample.sample_id}")
        score_record = score_lookup.get(_sample_lookup_key(sample))
        alert_record = alert_lookup.get(_sample_lookup_key(sample))
        prompt_inputs.append(
            build_prompt_input(
                sample,
                rule_record,
                alert_record=alert_record,
                score_record=score_record,
            )
        )
    return prompt_inputs


def export_prompt_inputs(
    prompt_inputs: Iterable[PromptInput],
    path: str | Path,
) -> str:
    """Export prompt inputs to a JSON Lines file with stable field ordering."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for prompt_input in prompt_inputs:
            payload = prompt_input.to_dict()
            ordered = {field: payload.get(field) for field in PROMPT_INPUT_FIELDS}
            handle.write(json.dumps(ordered, ensure_ascii=False, default=str))
            handle.write("\n")
    return output_path.as_posix()


__all__ = [
    "PromptLookupKey",
    "build_prompt_input",
    "build_prompt_inputs",
    "build_prompt_text",
    "build_rule_summary",
    "build_structured_context",
    "export_prompt_inputs",
]
