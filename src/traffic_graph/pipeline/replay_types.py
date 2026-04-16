"""Typed views for replaying persisted run bundles back into analysis objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ScoreScope = Literal["graph", "node", "edge", "flow"]
"""Supported anomaly-score table scopes."""


@dataclass(frozen=True, slots=True)
class ReplayManifestInfo:
    """Structured manifest metadata for one exported run bundle."""

    run_id: str
    timestamp: str
    split: str
    manifest_path: str
    base_directory: str
    run_directory: str
    score_formats: tuple[str, ...]
    alert_formats: tuple[str, ...]
    metrics_formats: tuple[str, ...]
    artifact_paths: dict[str, str] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    raw_manifest: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize the manifest info into a JSON-friendly dictionary."""

        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "split": self.split,
            "manifest_path": self.manifest_path,
            "base_directory": self.base_directory,
            "run_directory": self.run_directory,
            "score_formats": list(self.score_formats),
            "alert_formats": list(self.alert_formats),
            "metrics_formats": list(self.metrics_formats),
            "artifact_paths": dict(self.artifact_paths),
            "row_counts": dict(self.row_counts),
            "notes": list(self.notes),
            "raw_manifest": dict(self.raw_manifest),
        }


@dataclass(frozen=True, slots=True)
class ReplayScoreRecord:
    """Typed score-table row restored from a persisted export bundle."""

    run_id: str
    timestamp: str
    split: str
    score_scope: ScoreScope
    graph_id: int | str | None
    window_id: int | str | None
    node_id: int | str | None
    edge_id: int | str | None
    flow_id: int | str | None
    anomaly_score: float
    threshold: float | None
    is_alert: bool | None
    label: object | None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize the replay score record into a stable dictionary."""

        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "split": self.split,
            "score_scope": self.score_scope,
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


@dataclass(frozen=True, slots=True)
class ReplayAlertRecord:
    """Typed alert row restored from a persisted export bundle."""

    alert_id: str
    alert_level: str
    alert_scope: str
    run_id: str
    timestamp: str
    split: str
    graph_id: int | str | None
    window_id: int | str | None
    node_id: int | str | None
    edge_id: int | str | None
    flow_id: int | str | None
    anomaly_score: float
    threshold: float | None
    is_alert: bool | None
    label: object | None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize the replay alert record into a stable dictionary."""

        return {
            "alert_id": self.alert_id,
            "alert_level": self.alert_level,
            "alert_scope": self.alert_scope,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "split": self.split,
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


@dataclass(frozen=True, slots=True)
class ReplayBundle:
    """Fully loaded readback view over one exported run bundle."""

    run_id: str
    timestamp: str
    split: str
    manifest: ReplayManifestInfo
    file_index: dict[str, str] = field(default_factory=dict)
    loaded_files: dict[str, str] = field(default_factory=dict)
    graph_scores: tuple[ReplayScoreRecord, ...] = ()
    node_scores: tuple[ReplayScoreRecord, ...] = ()
    edge_scores: tuple[ReplayScoreRecord, ...] = ()
    flow_scores: tuple[ReplayScoreRecord, ...] = ()
    alert_records: tuple[ReplayAlertRecord, ...] = ()
    metrics_summary: dict[str, object] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize the replay bundle into a compact dictionary view."""

        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "split": self.split,
            "manifest": self.manifest.to_dict(),
            "file_index": dict(self.file_index),
            "loaded_files": dict(self.loaded_files),
            "graph_scores": [record.to_dict() for record in self.graph_scores],
            "node_scores": [record.to_dict() for record in self.node_scores],
            "edge_scores": [record.to_dict() for record in self.edge_scores],
            "flow_scores": [record.to_dict() for record in self.flow_scores],
            "alert_records": [record.to_dict() for record in self.alert_records],
            "metrics_summary": dict(self.metrics_summary),
            "notes": list(self.notes),
        }

