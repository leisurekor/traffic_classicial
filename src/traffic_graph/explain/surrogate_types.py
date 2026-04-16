"""Typed helpers for surrogate decision-tree training artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from traffic_graph.explain.explanation_types import ExplanationScope

SurrogateTreeMode = Literal["regression", "classification"]
"""Supported surrogate-tree training modes."""


@dataclass(frozen=True, slots=True)
class SurrogateTreeConfig:
    """Configuration used to fit a surrogate decision tree."""

    mode: SurrogateTreeMode = "regression"
    max_depth: int | None = 4
    min_samples_leaf: int = 5
    random_state: int = 42

    def to_dict(self) -> dict[str, object]:
        """Serialize the training configuration into a JSON-friendly mapping."""

        return {
            "mode": self.mode,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
        }


@dataclass(frozen=True, slots=True)
class SurrogateTrainingMatrix:
    """Matrix view extracted from explanation-ready samples for tree fitting."""

    features: np.ndarray
    targets: np.ndarray
    feature_names: tuple[str, ...]
    sample_ids: tuple[str, ...]
    sample_scope: ExplanationScope
    target_name: str

    def to_dict(self) -> dict[str, object]:
        """Serialize the matrix metadata without embedding the raw arrays."""

        return {
            "shape": tuple(int(dimension) for dimension in self.features.shape),
            "target_shape": tuple(int(dimension) for dimension in self.targets.shape),
            "feature_names": list(self.feature_names),
            "sample_ids": list(self.sample_ids),
            "sample_scope": self.sample_scope,
            "target_name": self.target_name,
        }


@dataclass(frozen=True, slots=True)
class SurrogateTreeSummary:
    """Compact training summary for a fitted surrogate tree."""

    sample_count: int
    sample_scope: ExplanationScope
    mode: SurrogateTreeMode
    feature_count: int
    tree_depth: int
    leaf_count: int
    target_name: str

    def to_dict(self) -> dict[str, object]:
        """Serialize the tree summary into a stable dictionary."""

        return {
            "sample_count": self.sample_count,
            "sample_scope": self.sample_scope,
            "mode": self.mode,
            "feature_count": self.feature_count,
            "tree_depth": self.tree_depth,
            "leaf_count": self.leaf_count,
            "target_name": self.target_name,
        }


@dataclass(slots=True)
class SurrogateTreeArtifact:
    """Fitted surrogate model plus metadata needed for persistence and reuse."""

    model: object
    feature_names: tuple[str, ...]
    config: SurrogateTreeConfig
    summary: SurrogateTreeSummary

    def to_dict(self) -> dict[str, object]:
        """Serialize artifact metadata while omitting the binary model payload."""

        return {
            "feature_names": list(self.feature_names),
            "config": self.config.to_dict(),
            "summary": self.summary.to_dict(),
            "model_class": type(self.model).__name__,
            "model_module": type(self.model).__module__,
        }


@dataclass(frozen=True, slots=True)
class SurrogateTreeSaveResult:
    """Filesystem paths produced when a surrogate tree artifact is persisted."""

    output_directory: str
    model_path: str
    metadata_path: str
    artifact_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize the save result for reporting or JSON export."""

        return {
            "output_directory": self.output_directory,
            "model_path": self.model_path,
            "metadata_path": self.metadata_path,
            "artifact_paths": dict(self.artifact_paths),
        }


__all__ = [
    "SurrogateTreeArtifact",
    "SurrogateTreeConfig",
    "SurrogateTreeMode",
    "SurrogateTreeSaveResult",
    "SurrogateTreeSummary",
    "SurrogateTrainingMatrix",
]
