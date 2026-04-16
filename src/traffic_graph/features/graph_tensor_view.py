"""Packed numpy-based graph inputs for model-facing preprocessing outputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass(frozen=True, slots=True)
class PackedGraphMetadata:
    """Serializable metadata describing one packed graph sample."""

    window_index: int
    window_start: datetime
    window_end: datetime
    node_count: int
    edge_count: int
    communication_edge_count: int
    association_edge_count: int


@dataclass(frozen=True, slots=True)
class PackedGraphInput:
    """Packed graph input with numpy matrices and index mappings."""

    node_features: np.ndarray
    edge_features: np.ndarray
    edge_index: np.ndarray
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    node_id_to_index: dict[str, int]
    edge_id_to_index: dict[str, int]
    edge_types: tuple[int, ...]
    node_feature_fields: tuple[str, ...]
    edge_feature_fields: tuple[str, ...]
    node_discrete_mask: tuple[bool, ...]
    edge_discrete_mask: tuple[bool, ...]
    metadata: PackedGraphMetadata

    @property
    def node_feature_dim(self) -> int:
        """Return the number of node feature columns."""

        return int(self.node_features.shape[1]) if self.node_features.ndim == 2 else 0

    @property
    def edge_feature_dim(self) -> int:
        """Return the number of edge feature columns."""

        return int(self.edge_features.shape[1]) if self.edge_features.ndim == 2 else 0

    def to_serializable(self) -> dict[str, object]:
        """Convert the packed graph input into JSON-friendly Python objects."""

        return {
            "node_features": self.node_features.tolist(),
            "edge_features": self.edge_features.tolist(),
            "edge_index": self.edge_index.tolist(),
            "node_ids": list(self.node_ids),
            "edge_ids": list(self.edge_ids),
            "node_id_to_index": dict(self.node_id_to_index),
            "edge_id_to_index": dict(self.edge_id_to_index),
            "edge_types": list(self.edge_types),
            "node_feature_fields": list(self.node_feature_fields),
            "edge_feature_fields": list(self.edge_feature_fields),
            "node_discrete_mask": list(self.node_discrete_mask),
            "edge_discrete_mask": list(self.edge_discrete_mask),
            "metadata": {
                "window_index": self.metadata.window_index,
                "window_start": self.metadata.window_start.isoformat(),
                "window_end": self.metadata.window_end.isoformat(),
                "node_count": self.metadata.node_count,
                "edge_count": self.metadata.edge_count,
                "communication_edge_count": self.metadata.communication_edge_count,
                "association_edge_count": self.metadata.association_edge_count,
            },
        }


def summarize_packed_graph_input(
    packed_graph: PackedGraphInput,
) -> dict[str, int]:
    """Return a compact summary of one packed graph input."""

    return {
        "window_index": packed_graph.metadata.window_index,
        "node_count": packed_graph.metadata.node_count,
        "node_feature_dim": packed_graph.node_feature_dim,
        "edge_count": packed_graph.metadata.edge_count,
        "edge_feature_dim": packed_graph.edge_feature_dim,
    }


__all__ = [
    "PackedGraphInput",
    "PackedGraphMetadata",
    "summarize_packed_graph_input",
]
