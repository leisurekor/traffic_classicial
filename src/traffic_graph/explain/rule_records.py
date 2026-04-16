"""Typed rule-record structures for surrogate-tree path extraction."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from traffic_graph.explain.explanation_types import ExplanationScope
from traffic_graph.explain.surrogate_types import SurrogateTreeMode

RuleOperator = Literal["<=", ">"]
"""Operators used in surrogate-tree path conditions."""

RULE_PATH_CONDITION_FIELDS: tuple[str, ...] = (
    "feature_name",
    "operator",
    "threshold",
    "sample_value",
    "tree_node_index",
)
"""Stable field order for serialized path conditions."""

RULE_RECORD_FIELDS: tuple[str, ...] = (
    "rule_id",
    "sample_id",
    "scope",
    "tree_mode",
    "predicted_score_or_class",
    "leaf_id",
    "path_conditions",
    "feature_names_used",
)
"""Stable field order for serialized rule records."""


def _jsonable_scalar(value: object) -> object:
    """Convert common scalar types into JSON-friendly values."""

    if hasattr(value, "item"):
        try:
            return value.item()  # type: ignore[call-arg]
        except Exception:
            return value
    return value


@dataclass(frozen=True, slots=True)
class RulePathCondition:
    """One ordered condition along the path from root to leaf."""

    feature_name: str
    operator: RuleOperator
    threshold: float
    sample_value: float | None = None
    tree_node_index: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize the path condition into a stable dictionary."""

        return {
            "feature_name": self.feature_name,
            "operator": self.operator,
            "threshold": self.threshold,
            "sample_value": self.sample_value,
            "tree_node_index": self.tree_node_index,
        }


@dataclass(frozen=True, slots=True)
class RuleRecord:
    """Structured surrogate-tree rule extracted for a single explanation sample."""

    rule_id: str
    sample_id: str
    scope: ExplanationScope
    tree_mode: SurrogateTreeMode
    predicted_score_or_class: object
    leaf_id: int
    path_conditions: tuple[RulePathCondition, ...] = ()
    feature_names_used: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize the rule record into a JSON-friendly dictionary."""

        return {
            "rule_id": self.rule_id,
            "sample_id": self.sample_id,
            "scope": self.scope,
            "tree_mode": self.tree_mode,
            "predicted_score_or_class": _jsonable_scalar(
                self.predicted_score_or_class
            ),
            "leaf_id": self.leaf_id,
            "path_conditions": [condition.to_dict() for condition in self.path_conditions],
            "feature_names_used": list(self.feature_names_used),
        }


@dataclass(frozen=True, slots=True)
class RuleRecordSummary:
    """Compact summary for a batch of extracted rule records."""

    total_count: int
    scope_counts: dict[str, int] = field(default_factory=dict)
    mode_counts: dict[str, int] = field(default_factory=dict)
    max_path_length: int = 0
    max_leaf_id: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize the summary into a stable dictionary."""

        return {
            "total_count": self.total_count,
            "scope_counts": dict(self.scope_counts),
            "mode_counts": dict(self.mode_counts),
            "max_path_length": self.max_path_length,
            "max_leaf_id": self.max_leaf_id,
        }


def export_rule_records(
    rule_records: Iterable[RuleRecord],
    path: str | Path,
) -> str:
    """Export rule records to a JSON Lines file with stable field order."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for rule_record in rule_records:
            payload = rule_record.to_dict()
            ordered = {field: payload.get(field) for field in RULE_RECORD_FIELDS}
            handle.write(json.dumps(ordered, ensure_ascii=False, default=str))
            handle.write("\n")
    return output_path.as_posix()


def summarize_rule(rule_record: RuleRecord) -> str:
    """Render a compact human-readable description of a single rule."""

    if rule_record.path_conditions:
        path_text = " AND ".join(
            f"{condition.feature_name} {condition.operator} {condition.threshold:.6g}"
            for condition in rule_record.path_conditions
        )
    else:
        path_text = "leaf-only"
    prediction = _jsonable_scalar(rule_record.predicted_score_or_class)
    if isinstance(prediction, float):
        prediction_text = f"{prediction:.6f}"
    else:
        prediction_text = str(prediction)
    return (
        f"{rule_record.rule_id}: "
        f"scope={rule_record.scope}, "
        f"mode={rule_record.tree_mode}, "
        f"leaf={rule_record.leaf_id}, "
        f"prediction={prediction_text}, "
        f"path={path_text}"
    )


def summarize_rules(rule_records: Sequence[RuleRecord]) -> str:
    """Render a compact summary over a sequence of rule records."""

    summary = RuleRecordSummary(
        total_count=len(rule_records),
        scope_counts={},
        mode_counts={},
        max_path_length=0,
        max_leaf_id=None,
    )
    scope_counts: dict[str, int] = {}
    mode_counts: dict[str, int] = {}
    max_path_length = 0
    max_leaf_id: int | None = None
    for rule_record in rule_records:
        scope_counts[rule_record.scope] = scope_counts.get(rule_record.scope, 0) + 1
        mode_counts[rule_record.tree_mode] = mode_counts.get(rule_record.tree_mode, 0) + 1
        max_path_length = max(max_path_length, len(rule_record.path_conditions))
        if max_leaf_id is None or rule_record.leaf_id > max_leaf_id:
            max_leaf_id = rule_record.leaf_id
    summary = RuleRecordSummary(
        total_count=len(rule_records),
        scope_counts=scope_counts,
        mode_counts=mode_counts,
        max_path_length=max_path_length,
        max_leaf_id=max_leaf_id,
    )
    lines = [
        f"Rule records: total={summary.total_count}",
        "Scopes: "
        + (
            ", ".join(
                f"{scope}={count}" for scope, count in sorted(summary.scope_counts.items())
            )
            if summary.scope_counts
            else "none"
        ),
        "Modes: "
        + (
            ", ".join(
                f"{mode}={count}" for mode, count in sorted(summary.mode_counts.items())
            )
            if summary.mode_counts
            else "none"
        ),
        f"Max path length: {summary.max_path_length}",
    ]
    if summary.max_leaf_id is not None:
        lines.append(f"Max leaf id: {summary.max_leaf_id}")
    preview_rules = list(rule_records[:3])
    if preview_rules:
        lines.append("Preview:")
        lines.extend(f"  - {summarize_rule(rule_record)}" for rule_record in preview_rules)
    return "\n".join(lines)


__all__ = [
    "RULE_PATH_CONDITION_FIELDS",
    "RULE_RECORD_FIELDS",
    "RuleOperator",
    "RulePathCondition",
    "RuleRecord",
    "RuleRecordSummary",
    "export_rule_records",
    "summarize_rule",
    "summarize_rules",
]
