"""Typed explanation-ready sample structures for downstream analysis modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ExplanationScope = Literal["graph", "flow", "node"]
"""Supported explanation sample granularities."""


@dataclass(frozen=True, slots=True)
class ExplanationSample:
    """Structured explanation-ready sample derived from replayed bundles."""

    sample_id: str
    scope: ExplanationScope
    run_id: str
    graph_id: int | str | None
    window_id: int | str | None
    flow_id: int | str | None = None
    node_id: int | str | None = None
    anomaly_score: float = 0.0
    threshold: float | None = None
    is_alert: bool | None = None
    alert_level: str | None = None
    label: object | None = None
    stats_summary: dict[str, object] = field(default_factory=dict)
    graph_summary: dict[str, object] = field(default_factory=dict)
    feature_summary: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize the explanation sample into a JSON-friendly dictionary."""

        return {
            "sample_id": self.sample_id,
            "scope": self.scope,
            "run_id": self.run_id,
            "graph_id": self.graph_id,
            "window_id": self.window_id,
            "flow_id": self.flow_id,
            "node_id": self.node_id,
            "anomaly_score": self.anomaly_score,
            "threshold": self.threshold,
            "is_alert": self.is_alert,
            "alert_level": self.alert_level,
            "label": self.label,
            "stats_summary": dict(self.stats_summary),
            "graph_summary": dict(self.graph_summary),
            "feature_summary": dict(self.feature_summary),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ExplanationSampleSummary:
    """Compact summary over a set of explanation-ready samples."""

    total_count: int
    scope_counts: dict[str, int] = field(default_factory=dict)
    alert_count: int = 0
    labeled_count: int = 0
    max_anomaly_score: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize the summary into a stable dictionary."""

        return {
            "total_count": self.total_count,
            "scope_counts": dict(self.scope_counts),
            "alert_count": self.alert_count,
            "labeled_count": self.labeled_count,
            "max_anomaly_score": self.max_anomaly_score,
        }


__all__ = [
    "ExplanationSample",
    "ExplanationSampleSummary",
    "ExplanationScope",
]
