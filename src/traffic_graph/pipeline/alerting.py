"""Threshold-based alert conversion helpers for anomaly score tables."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

from traffic_graph.config import AlertingConfig
from traffic_graph.pipeline.alert_types import (
    AlertLevel,
    AlertRecord,
    AlertScope,
    AlertScoreTables,
)

_SCOPE_ORDER: tuple[AlertScope, ...] = ("graph", "node", "edge", "flow")
_LEVEL_ORDER: tuple[AlertLevel, ...] = ("low", "medium", "high")


def _coerce_alerting_config(config: object) -> AlertingConfig:
    """Convert a mapping or dataclass-like object into :class:`AlertingConfig`."""

    if isinstance(config, AlertingConfig):
        return config
    if isinstance(config, Mapping):
        return AlertingConfig.from_mapping(config)
    threshold = getattr(config, "anomaly_threshold", 0.5)
    medium_multiplier = getattr(config, "medium_multiplier", 1.5)
    high_multiplier = getattr(config, "high_multiplier", 2.0)
    return AlertingConfig(
        anomaly_threshold=float(threshold),
        medium_multiplier=float(medium_multiplier),
        high_multiplier=float(high_multiplier),
    )


def _row_value(row: Mapping[str, object], keys: Iterable[str], default: object = None) -> object:
    """Return the first non-empty value found in ``row`` for ``keys``."""

    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return default


def _row_score(scope: AlertScope, row: Mapping[str, object]) -> float:
    """Extract a numeric anomaly score from a score-table row."""

    keys_by_scope: dict[AlertScope, tuple[str, ...]] = {
        "graph": ("graph_anomaly_score", "anomaly_score", "score"),
        "node": ("node_anomaly_score", "anomaly_score", "score"),
        "edge": ("edge_anomaly_score", "anomaly_score", "score"),
        "flow": ("flow_anomaly_score", "anomaly_score", "score"),
    }
    raw_score = _row_value(row, keys_by_scope[scope], default=0.0)
    try:
        return float(raw_score)
    except (TypeError, ValueError):
        return 0.0


def _severity_level(
    anomaly_score: float,
    threshold: float,
    medium_multiplier: float,
    high_multiplier: float,
) -> AlertLevel:
    """Map a score and threshold into a simple low/medium/high severity label."""

    safe_threshold = threshold if threshold > 0.0 else 1e-12
    ratio = anomaly_score / safe_threshold
    if ratio < medium_multiplier:
        return "low"
    if ratio < high_multiplier:
        return "medium"
    return "high"


def _alert_identifier(
    *,
    scope: AlertScope,
    graph_id: object,
    window_id: object,
    entity_id: object,
    index: int,
) -> str:
    """Build a stable alert identifier from the graph scope and row identity."""

    graph_token = "none" if graph_id in {None, ""} else str(graph_id)
    window_token = "none" if window_id in {None, ""} else str(window_id)
    entity_token = "none" if entity_id in {None, ""} else str(entity_id)
    return f"{scope}:{graph_token}:{window_token}:{entity_token}:{index}"


def _alert_metadata(
    row: Mapping[str, object],
    *,
    excluded_keys: set[str],
) -> dict[str, object]:
    """Preserve a compact copy of the source row for downstream consumers."""

    metadata: dict[str, object] = {}
    for key, value in row.items():
        if key in excluded_keys:
            continue
        metadata[key] = value
    return metadata


def _row_to_alert_record(
    *,
    scope: AlertScope,
    row: Mapping[str, object],
    config: AlertingConfig,
    index: int,
) -> AlertRecord:
    """Convert one score-table row into an :class:`AlertRecord`."""

    anomaly_score = _row_score(scope, row)
    threshold = float(config.anomaly_threshold)
    is_alert = anomaly_score >= threshold
    alert_level = _severity_level(
        anomaly_score=anomaly_score,
        threshold=threshold,
        medium_multiplier=config.medium_multiplier,
        high_multiplier=config.high_multiplier,
    )
    graph_id = _row_value(row, ("graph_id", "graph_index"))
    window_id = _row_value(row, ("window_id", "window_index"))
    node_id = _row_value(row, ("node_id",))
    edge_id = _row_value(row, ("edge_id",))
    flow_id = _row_value(row, ("flow_id", "logical_flow_id"))
    entity_id = {
        "graph": graph_id,
        "node": node_id,
        "edge": edge_id,
        "flow": flow_id,
    }[scope]
    label = _row_value(
        row,
        ("label", "graph_label", "edge_label", "flow_label", "node_label"),
    )
    metadata = _alert_metadata(
        row,
        excluded_keys={
            "graph_id",
            "graph_index",
            "window_id",
            "window_index",
            "node_id",
            "edge_id",
            "flow_id",
            "logical_flow_id",
            "label",
            "graph_label",
            "edge_label",
            "flow_label",
            "node_label",
            "graph_anomaly_score",
            "node_anomaly_score",
            "edge_anomaly_score",
            "flow_anomaly_score",
            "anomaly_score",
            "score",
        },
    )
    return AlertRecord(
        alert_id=_alert_identifier(
            scope=scope,
            graph_id=graph_id,
            window_id=window_id,
            entity_id=entity_id,
            index=index,
        ),
        alert_level=alert_level,
        alert_scope=scope,
        graph_id=graph_id,
        window_id=window_id,
        node_id=node_id,
        edge_id=edge_id,
        flow_id=flow_id,
        anomaly_score=anomaly_score,
        threshold=threshold,
        is_alert=is_alert,
        label=label,
        metadata=metadata,
    )


def build_alert_records(
    score_tables: object,
    config: object,
) -> list[AlertRecord]:
    """Convert graph, node, edge, and flow score tables into alert records."""

    tables = AlertScoreTables.from_value(score_tables)
    alert_config = _coerce_alerting_config(config)
    alert_records: list[AlertRecord] = []
    for scope, rows in (
        ("graph", tables.graph_scores),
        ("node", tables.node_scores),
        ("edge", tables.edge_scores),
        ("flow", tables.flow_scores),
    ):
        alert_records.extend(
            _row_to_alert_record(
                scope=scope,
                row=row,
                config=alert_config,
                index=index,
            )
            for index, row in enumerate(rows)
        )
    return alert_records


def filter_alerts(
    alert_records: Iterable[AlertRecord],
    scope: AlertScope | None = None,
    only_positive: bool = True,
) -> list[AlertRecord]:
    """Filter alert records by scope and positivity."""

    filtered: list[AlertRecord] = []
    for record in alert_records:
        if scope is not None and record.alert_scope != scope:
            continue
        if only_positive and not record.is_alert:
            continue
        filtered.append(record)
    return filtered


def summarize_alerts(alert_records: Iterable[AlertRecord]) -> dict[str, object]:
    """Summarize alert records by scope, positivity, and severity level."""

    records = list(alert_records)
    scope_counts = Counter(record.alert_scope for record in records)
    level_counts = Counter(record.alert_level for record in records)
    positive_scope_counts = Counter(
        record.alert_scope for record in records if record.is_alert
    )
    positive_level_counts = Counter(
        record.alert_level for record in records if record.is_alert
    )
    summary: dict[str, object] = {
        "total_count": len(records),
        "positive_count": sum(1 for record in records if record.is_alert),
        "scope_counts": {scope: int(scope_counts.get(scope, 0)) for scope in _SCOPE_ORDER},
        "positive_scope_counts": {
            scope: int(positive_scope_counts.get(scope, 0)) for scope in _SCOPE_ORDER
        },
        "level_counts": {level: int(level_counts.get(level, 0)) for level in _LEVEL_ORDER},
        "positive_level_counts": {
            level: int(positive_level_counts.get(level, 0)) for level in _LEVEL_ORDER
        },
    }
    summary["scope_level_counts"] = {
        scope: {
            level: sum(
                1
                for record in records
                if record.alert_scope == scope and record.alert_level == level
            )
            for level in _LEVEL_ORDER
        }
        for scope in _SCOPE_ORDER
    }
    summary["positive_rate"] = (
        float(summary["positive_count"]) / float(summary["total_count"])
        if summary["total_count"]
        else 0.0
    )
    return summary


__all__ = [
    "build_alert_records",
    "filter_alerts",
    "summarize_alerts",
]
