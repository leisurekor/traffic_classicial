"""Abstract detector interfaces for future graph-based anomaly models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from traffic_graph.graph import GraphSnapshot


@dataclass(frozen=True, slots=True)
class DetectorSpec:
    """Configuration placeholder describing a future detector backend."""

    name: str
    device: str = "cpu"
    score_threshold: float | None = None


class UnsupervisedDetector(ABC):
    """Abstract contract that all unsupervised detector implementations must follow."""

    def __init__(self, spec: DetectorSpec) -> None:
        """Store the shared detector specification."""

        self.spec = spec

    @abstractmethod
    def fit(self, graph: GraphSnapshot) -> None:
        """Fit the detector on a graph snapshot."""

    @abstractmethod
    def score(self, graph: GraphSnapshot) -> Sequence[float]:
        """Produce anomaly scores for graph structures or associated flows."""

