"""Build explanation-ready samples from replayed export bundles."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from collections.abc import Iterable, Sequence

from traffic_graph.explain.explanation_types import (
    ExplanationSample,
    ExplanationSampleSummary,
    ExplanationScope,
)
from traffic_graph.pipeline.replay_types import (
    ReplayAlertRecord,
    ReplayBundle,
    ReplayScoreRecord,
)

_GRAPH_SUMMARY_FIELDS: tuple[str, ...] = (
    "node_count",
    "edge_count",
    "client_node_count",
    "server_node_count",
    "communication_edge_count",
    "association_edge_count",
    "aggregated_edge_count",
)
_FLOW_STATS_FIELDS: tuple[str, ...] = (
    "pkt_count",
    "byte_count",
    "duration",
    "flow_count",
    "is_aggregated",
    "edge_type",
    "src_ip",
    "src_port",
    "dst_ip",
    "dst_port",
    "proto",
)
_NODE_STATS_FIELDS: tuple[str, ...] = (
    "endpoint_type",
    "port",
    "proto",
    "total_pkt_count",
    "total_byte_count",
    "total_flow_count",
    "avg_pkt_count",
    "avg_byte_count",
    "avg_duration",
    "communication_edge_count",
    "association_edge_count",
    "total_degree",
    "communication_in_degree",
    "communication_out_degree",
    "unique_neighbor_count",
)


def _sample_identifier(
    *,
    scope: ExplanationScope,
    graph_id: object,
    window_id: object,
    entity_id: object,
) -> str:
    """Construct a stable explanation sample id."""

    graph_token = "none" if graph_id in {None, ""} else str(graph_id)
    window_token = "none" if window_id in {None, ""} else str(window_id)
    entity_token = "none" if entity_id in {None, ""} else str(entity_id)
    return f"{scope}:{graph_token}:{window_token}:{entity_token}"


def _graph_key(record: ReplayScoreRecord | ExplanationSample) -> tuple[object, object]:
    """Return the graph/window key used to align graph summaries."""

    return record.graph_id, record.window_id


def _alert_index(
    alert_records: Sequence[ReplayAlertRecord],
) -> dict[tuple[str, object, object, object], ReplayAlertRecord]:
    """Index alert records by scope and entity ids for quick lookup."""

    indexed: dict[tuple[str, object, object, object], ReplayAlertRecord] = {}
    for record in alert_records:
        entity_id: object
        if record.alert_scope == "graph":
            entity_id = record.graph_id
        elif record.alert_scope == "flow":
            entity_id = record.flow_id
        elif record.alert_scope == "node":
            entity_id = record.node_id
        else:
            entity_id = record.edge_id
        indexed[(record.alert_scope, record.graph_id, record.window_id, entity_id)] = record
    return indexed


def _graph_summary_index(
    graph_scores: Sequence[ReplayScoreRecord],
) -> dict[tuple[object, object], dict[str, object]]:
    """Index graph summary metadata by graph/window id."""

    indexed: dict[tuple[object, object], dict[str, object]] = {}
    for record in graph_scores:
        summary = {
            field: record.metadata[field]
            for field in _GRAPH_SUMMARY_FIELDS
            if field in record.metadata
        }
        summary["graph_anomaly_score"] = record.anomaly_score
        if record.threshold is not None:
            summary["graph_threshold"] = record.threshold
        if record.is_alert is not None:
            summary["graph_is_alert"] = record.is_alert
        if record.label is not None:
            summary["graph_label"] = record.label
        indexed[_graph_key(record)] = summary
    return indexed


def _feature_summary(metadata: dict[str, object]) -> dict[str, object]:
    """Extract an optional feature summary from metadata when available."""

    candidate_fields = (
        "feature_fields",
        "feature_field_names",
        "node_feature_fields",
        "edge_feature_fields",
        "feature_names",
    )
    field_names: list[str] = []
    for key in candidate_fields:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            field_names.append(value.strip())
        elif isinstance(value, (list, tuple)):
            field_names.extend(str(item) for item in value if str(item).strip())
    deduplicated_field_names = tuple(dict.fromkeys(field_names))
    if deduplicated_field_names:
        return {
            "available": True,
            "field_names": list(deduplicated_field_names),
            "field_count": len(deduplicated_field_names),
        }
    return {
        "available": False,
        "field_names": [],
        "field_count": 0,
        "note": "feature_fields_not_available_in_bundle",
    }


def _summary_subset(
    metadata: dict[str, object],
    fields: Sequence[str],
) -> dict[str, object]:
    """Select a stable subset of summary fields from metadata."""

    return {field: metadata[field] for field in fields if field in metadata}


def _build_graph_sample(
    record: ReplayScoreRecord,
    *,
    alert_record: ReplayAlertRecord | None,
    graph_summary: dict[str, object],
) -> ExplanationSample:
    """Create a graph-level explanation sample."""

    return ExplanationSample(
        sample_id=_sample_identifier(
            scope="graph",
            graph_id=record.graph_id,
            window_id=record.window_id,
            entity_id=record.graph_id,
        ),
        scope="graph",
        run_id=record.run_id,
        graph_id=record.graph_id,
        window_id=record.window_id,
        anomaly_score=record.anomaly_score,
        threshold=record.threshold,
        is_alert=alert_record.is_alert if alert_record is not None else record.is_alert,
        alert_level=alert_record.alert_level if alert_record is not None else None,
        label=record.label if record.label is not None else (alert_record.label if alert_record is not None else None),
        stats_summary=_summary_subset(record.metadata, _GRAPH_SUMMARY_FIELDS),
        graph_summary=graph_summary,
        feature_summary=_feature_summary(record.metadata),
        metadata=dict(record.metadata),
    )


def _build_flow_sample(
    record: ReplayScoreRecord,
    *,
    alert_record: ReplayAlertRecord | None,
    graph_summary: dict[str, object],
) -> ExplanationSample:
    """Create a flow-level explanation sample."""

    return ExplanationSample(
        sample_id=_sample_identifier(
            scope="flow",
            graph_id=record.graph_id,
            window_id=record.window_id,
            entity_id=record.flow_id,
        ),
        scope="flow",
        run_id=record.run_id,
        graph_id=record.graph_id,
        window_id=record.window_id,
        flow_id=record.flow_id,
        anomaly_score=record.anomaly_score,
        threshold=record.threshold,
        is_alert=alert_record.is_alert if alert_record is not None else record.is_alert,
        alert_level=alert_record.alert_level if alert_record is not None else None,
        label=record.label if record.label is not None else (alert_record.label if alert_record is not None else None),
        stats_summary=_summary_subset(record.metadata, _FLOW_STATS_FIELDS),
        graph_summary=graph_summary,
        feature_summary=_feature_summary(record.metadata),
        metadata=dict(record.metadata),
    )


def _build_node_sample(
    record: ReplayScoreRecord,
    *,
    alert_record: ReplayAlertRecord | None,
    graph_summary: dict[str, object],
) -> ExplanationSample:
    """Create a node-level explanation sample."""

    return ExplanationSample(
        sample_id=_sample_identifier(
            scope="node",
            graph_id=record.graph_id,
            window_id=record.window_id,
            entity_id=record.node_id,
        ),
        scope="node",
        run_id=record.run_id,
        graph_id=record.graph_id,
        window_id=record.window_id,
        node_id=record.node_id,
        anomaly_score=record.anomaly_score,
        threshold=record.threshold,
        is_alert=alert_record.is_alert if alert_record is not None else record.is_alert,
        alert_level=alert_record.alert_level if alert_record is not None else None,
        label=record.label if record.label is not None else (alert_record.label if alert_record is not None else None),
        stats_summary=_summary_subset(record.metadata, _NODE_STATS_FIELDS),
        graph_summary=graph_summary,
        feature_summary=_feature_summary(record.metadata),
        metadata=dict(record.metadata),
    )


def sort_samples_by_score(
    samples: Sequence[ExplanationSample],
    *,
    descending: bool = True,
) -> list[ExplanationSample]:
    """Return samples sorted by anomaly score."""

    return sorted(
        samples,
        key=lambda sample: (sample.anomaly_score, sample.sample_id),
        reverse=descending,
    )


def select_top_alert_samples(
    samples: Sequence[ExplanationSample],
    *,
    k: int = 50,
) -> list[ExplanationSample]:
    """Return the top-k alert samples by anomaly score."""

    if k <= 0:
        return []
    alert_samples = [sample for sample in samples if sample.is_alert]
    return sort_samples_by_score(alert_samples)[:k]


def select_balanced_samples_for_explanation(
    samples: Sequence[ExplanationSample],
    *,
    max_samples: int = 50,
) -> list[ExplanationSample]:
    """Select a roughly balanced alert/non-alert subset for explanation workflows."""

    if max_samples <= 0:
        return []
    positive_samples = sort_samples_by_score(
        [sample for sample in samples if sample.is_alert]
    )
    negative_samples = sort_samples_by_score(
        [sample for sample in samples if sample.is_alert is False]
    )
    if not positive_samples or not negative_samples:
        return sort_samples_by_score(samples)[:max_samples]

    positive_target = max_samples // 2 + (max_samples % 2)
    negative_target = max_samples // 2
    selected = positive_samples[:positive_target] + negative_samples[:negative_target]
    selected_ids = {candidate.sample_id for candidate in selected}
    if len(selected) < max_samples:
        remaining = [
            sample
            for sample in sort_samples_by_score(samples)
            if sample.sample_id not in selected_ids
        ]
        selected.extend(remaining[: max_samples - len(selected)])
    return sort_samples_by_score(selected)


def build_explanation_samples(
    replay_bundle: ReplayBundle,
    *,
    scope: ExplanationScope = "flow",
    only_alerts: bool = True,
    top_k: int | None = None,
) -> list[ExplanationSample]:
    """Build explanation-ready samples from a replay bundle."""

    alert_records = _alert_index(replay_bundle.alert_records)
    graph_summary_by_key = _graph_summary_index(replay_bundle.graph_scores)
    score_records_by_scope: dict[ExplanationScope, Sequence[ReplayScoreRecord]] = {
        "graph": replay_bundle.graph_scores,
        "flow": replay_bundle.flow_scores,
        "node": replay_bundle.node_scores,
    }
    score_records = score_records_by_scope[scope]
    builders = {
        "graph": _build_graph_sample,
        "flow": _build_flow_sample,
        "node": _build_node_sample,
    }
    samples: list[ExplanationSample] = []
    for record in score_records:
        entity_id: object
        if scope == "graph":
            entity_id = record.graph_id
        elif scope == "flow":
            entity_id = record.flow_id
        else:
            entity_id = record.node_id
        alert_record = alert_records.get((scope, record.graph_id, record.window_id, entity_id))
        graph_summary = graph_summary_by_key.get(_graph_key(record), {})
        sample = builders[scope](
            record,
            alert_record=alert_record,
            graph_summary=graph_summary,
        )
        samples.append(sample)

    if only_alerts:
        samples = [sample for sample in samples if sample.is_alert is True]
    samples = sort_samples_by_score(samples)
    if top_k is not None and top_k >= 0:
        samples = samples[:top_k]
    return samples


def export_explanation_candidates(
    samples: Sequence[ExplanationSample],
    path: str | Path,
) -> str:
    """Export explanation-ready samples to a JSON Lines file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_dict(), ensure_ascii=False, default=str))
            handle.write("\n")
    return output_path.as_posix()


def summarize_explanation_samples(
    samples: Sequence[ExplanationSample],
) -> ExplanationSampleSummary:
    """Return a compact summary over explanation-ready samples."""

    scope_counts = Counter(sample.scope for sample in samples)
    alert_count = sum(1 for sample in samples if sample.is_alert)
    labeled_count = sum(1 for sample in samples if sample.label is not None)
    max_score = max((sample.anomaly_score for sample in samples), default=None)
    return ExplanationSampleSummary(
        total_count=len(samples),
        scope_counts={str(scope): int(count) for scope, count in scope_counts.items()},
        alert_count=alert_count,
        labeled_count=labeled_count,
        max_anomaly_score=max_score,
    )


def summarize_explanation_samples_text(
    samples: Sequence[ExplanationSample],
) -> str:
    """Render a short human-readable explanation sample summary."""

    summary = summarize_explanation_samples(samples)
    lines = [
        f"Explanation samples: total={summary.total_count}, alerts={summary.alert_count}, labeled={summary.labeled_count}",
        "Scope counts:",
    ]
    if summary.scope_counts:
        lines.extend(
            f"  - {scope}: {count}" for scope, count in sorted(summary.scope_counts.items())
        )
    else:
        lines.append("  - none")
    if summary.max_anomaly_score is not None:
        lines.append(f"Max anomaly score: {summary.max_anomaly_score:.6f}")
    top_samples = sort_samples_by_score(samples)[:3]
    if top_samples:
        lines.append("Top samples:")
        lines.extend(
            "  - "
            f"{sample.sample_id} score={sample.anomaly_score:.6f} "
            f"alert={sample.is_alert}"
            for sample in top_samples
        )
    return "\n".join(lines)


__all__ = [
    "ExplanationSample",
    "ExplanationSampleSummary",
    "ExplanationScope",
    "build_explanation_samples",
    "export_explanation_candidates",
    "select_balanced_samples_for_explanation",
    "select_top_alert_samples",
    "sort_samples_by_score",
    "summarize_explanation_samples",
    "summarize_explanation_samples_text",
]
