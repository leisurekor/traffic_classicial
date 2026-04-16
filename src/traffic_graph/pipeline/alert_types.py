"""Typed alert record structures for threshold-based anomaly output."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

AlertScope = Literal["graph", "node", "edge", "flow"]
"""Supported alert granularities."""

AlertLevel = Literal["low", "medium", "high"]
"""Simple severity labels derived from anomaly scores."""


def _as_row_sequence(
    value: object,
) -> tuple[Mapping[str, object], ...]:
    """Normalize an arbitrary input into a tuple of row mappings."""

    if value is None:
        return ()
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, Iterable) and not isinstance(
        value,
        (str, bytes, bytearray, Mapping),
    ):
        rows: list[Mapping[str, object]] = []
        for item in value:
            if isinstance(item, Mapping):
                rows.append(item)
            elif hasattr(item, "to_dict"):
                maybe_row = getattr(item, "to_dict")()
                if isinstance(maybe_row, Mapping):
                    rows.append(maybe_row)
            elif hasattr(item, "__dict__"):
                rows.append(dict(getattr(item, "__dict__")))
        return tuple(rows)
    return ()


@dataclass(frozen=True, slots=True)
class AlertScoreTables:
    """Container for graph, node, edge, and flow score tables."""

    graph_scores: Sequence[Mapping[str, object]] = field(default_factory=tuple)
    node_scores: Sequence[Mapping[str, object]] = field(default_factory=tuple)
    edge_scores: Sequence[Mapping[str, object]] = field(default_factory=tuple)
    flow_scores: Sequence[Mapping[str, object]] = field(default_factory=tuple)

    @classmethod
    def from_value(cls, value: object) -> "AlertScoreTables":
        """Build score tables from a mapping or an object with score attributes."""

        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(
                graph_scores=_as_row_sequence(value.get("graph_scores")),
                node_scores=_as_row_sequence(value.get("node_scores")),
                edge_scores=_as_row_sequence(value.get("edge_scores")),
                flow_scores=_as_row_sequence(value.get("flow_scores")),
            )
        return cls(
            graph_scores=_as_row_sequence(getattr(value, "graph_scores", None)),
            node_scores=_as_row_sequence(getattr(value, "node_scores", None)),
            edge_scores=_as_row_sequence(getattr(value, "edge_scores", None)),
            flow_scores=_as_row_sequence(getattr(value, "flow_scores", None)),
        )


@dataclass(frozen=True, slots=True)
class AlertRecord:
    """Structured alert output produced from anomaly score tables."""

    alert_id: str
    alert_level: AlertLevel
    alert_scope: AlertScope
    graph_id: int | str | None = None
    window_id: int | str | None = None
    node_id: int | str | None = None
    edge_id: int | str | None = None
    flow_id: int | str | None = None
    anomaly_score: float = 0.0
    threshold: float = 0.0
    is_alert: bool = False
    label: object | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize the alert record into a JSON-friendly dictionary."""

        return {
            "alert_id": self.alert_id,
            "alert_level": self.alert_level,
            "alert_scope": self.alert_scope,
            "graph_id": self.graph_id,
            "window_id": self.window_id,
            "node_id": self.node_id,
            "edge_id": self.edge_id,
            "flow_id": self.flow_id,
            "anomaly_score": self.anomaly_score,
            "threshold": self.threshold,
            "is_alert": self.is_alert,
            "label": self.label,
            "metadata": dict(self.metadata),
        }
