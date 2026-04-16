"""Normalization helpers for packed graph feature matrices."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np


def _mask_from_excluded_fields(
    field_names: tuple[str, ...],
    excluded_fields: tuple[str, ...],
) -> tuple[bool, ...]:
    """Build a boolean mask for fields excluded from continuous normalization."""

    excluded_lookup = set(excluded_fields)
    return tuple(field_name in excluded_lookup for field_name in field_names)


@dataclass(slots=True)
class MatrixNormalizer:
    """Field-aware matrix normalizer supporting standard and robust scaling."""

    field_names: tuple[str, ...]
    excluded_fields: tuple[str, ...] = ()
    method: str = "standard"
    enabled: bool = True
    center_: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    scale_: np.ndarray = field(default_factory=lambda: np.ones(0, dtype=float))
    discrete_mask: tuple[bool, ...] = field(default_factory=tuple)
    is_fitted: bool = False

    def to_dict(self) -> dict[str, object]:
        """Serialize the normalizer state into JSON-friendly data."""

        return {
            "field_names": list(self.field_names),
            "excluded_fields": list(self.excluded_fields),
            "method": self.method,
            "enabled": self.enabled,
            "center": self.center_.tolist(),
            "scale": self.scale_.tolist(),
            "discrete_mask": list(self.discrete_mask),
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "MatrixNormalizer":
        """Rehydrate a normalizer from serialized metadata."""

        field_names = tuple(str(item) for item in data.get("field_names", ()))
        excluded_fields = tuple(str(item) for item in data.get("excluded_fields", ()))
        normalizer = cls(
            field_names=field_names,
            excluded_fields=excluded_fields,
            method=str(data.get("method", "standard")),
            enabled=bool(data.get("enabled", True)),
        )
        normalizer.center_ = np.asarray(data.get("center", ()), dtype=float)
        normalizer.scale_ = np.asarray(data.get("scale", ()), dtype=float)
        normalizer.discrete_mask = tuple(bool(item) for item in data.get("discrete_mask", ()))
        normalizer.is_fitted = bool(data.get("is_fitted", False))
        return normalizer

    def fit(self, matrix: np.ndarray) -> "MatrixNormalizer":
        """Fit scaling parameters on a 2D feature matrix."""

        field_count = len(self.field_names)
        self.discrete_mask = _mask_from_excluded_fields(
            self.field_names,
            self.excluded_fields,
        )
        self.center_ = np.zeros(field_count, dtype=float)
        self.scale_ = np.ones(field_count, dtype=float)

        if not self.enabled or self.method == "none":
            self.is_fitted = True
            return self

        array = np.asarray(matrix, dtype=float)
        if array.size == 0:
            self.is_fitted = True
            return self

        for column_index, is_discrete in enumerate(self.discrete_mask):
            if is_discrete:
                continue

            column = array[:, column_index]
            if self.method == "robust":
                center = float(np.median(column))
                q75, q25 = np.percentile(column, [75.0, 25.0])
                scale = float(q75 - q25)
            else:
                center = float(np.mean(column))
                scale = float(np.std(column))

            if scale == 0.0:
                scale = 1.0
            self.center_[column_index] = center
            self.scale_[column_index] = scale

        self.is_fitted = True
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        """Transform a 2D feature matrix using the fitted normalization state."""

        if not self.is_fitted:
            raise RuntimeError("MatrixNormalizer must be fitted before transform().")

        array = np.asarray(matrix, dtype=float).copy()
        if not self.enabled or self.method == "none" or array.size == 0:
            return array

        for column_index, is_discrete in enumerate(self.discrete_mask):
            if is_discrete:
                continue
            array[:, column_index] = (
                array[:, column_index] - self.center_[column_index]
            ) / self.scale_[column_index]
        return array


@dataclass(slots=True)
class FeaturePreprocessor:
    """Normalizer bundle for packed node and edge feature matrices."""

    node_field_names: tuple[str, ...]
    edge_field_names: tuple[str, ...]
    node_excluded_fields: tuple[str, ...]
    edge_excluded_fields: tuple[str, ...]
    method: str = "standard"
    enabled: bool = True
    node_normalizer: MatrixNormalizer = field(init=False)
    edge_normalizer: MatrixNormalizer = field(init=False)

    def __post_init__(self) -> None:
        """Create node and edge matrix normalizers from the shared config."""

        self.node_normalizer = MatrixNormalizer(
            field_names=self.node_field_names,
            excluded_fields=self.node_excluded_fields,
            method=self.method,
            enabled=self.enabled,
        )
        self.edge_normalizer = MatrixNormalizer(
            field_names=self.edge_field_names,
            excluded_fields=self.edge_excluded_fields,
            method=self.method,
            enabled=self.enabled,
        )

    @property
    def node_discrete_mask(self) -> tuple[bool, ...]:
        """Return the node feature mask for fields excluded from normalization."""

        return self.node_normalizer.discrete_mask

    @property
    def edge_discrete_mask(self) -> tuple[bool, ...]:
        """Return the edge feature mask for fields excluded from normalization."""

        return self.edge_normalizer.discrete_mask

    def fit(self, node_matrix: np.ndarray, edge_matrix: np.ndarray) -> "FeaturePreprocessor":
        """Fit node and edge normalizers on stacked feature matrices."""

        self.node_normalizer.fit(node_matrix)
        self.edge_normalizer.fit(edge_matrix)
        return self

    def transform_node_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Transform a node feature matrix using the fitted node normalizer."""

        return self.node_normalizer.transform(matrix)

    def transform_edge_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Transform an edge feature matrix using the fitted edge normalizer."""

        return self.edge_normalizer.transform(matrix)

    def to_dict(self) -> dict[str, object]:
        """Serialize the preprocessor state for checkpointing."""

        return {
            "node_field_names": list(self.node_field_names),
            "edge_field_names": list(self.edge_field_names),
            "node_excluded_fields": list(self.node_excluded_fields),
            "edge_excluded_fields": list(self.edge_excluded_fields),
            "method": self.method,
            "enabled": self.enabled,
            "node_normalizer": self.node_normalizer.to_dict(),
            "edge_normalizer": self.edge_normalizer.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "FeaturePreprocessor":
        """Restore a feature preprocessor from serialized metadata."""

        preprocessor = cls(
            node_field_names=tuple(str(item) for item in data.get("node_field_names", ())),
            edge_field_names=tuple(str(item) for item in data.get("edge_field_names", ())),
            node_excluded_fields=tuple(
                str(item) for item in data.get("node_excluded_fields", ())
            ),
            edge_excluded_fields=tuple(
                str(item) for item in data.get("edge_excluded_fields", ())
            ),
            method=str(data.get("method", "standard")),
            enabled=bool(data.get("enabled", True)),
        )
        node_normalizer_data = data.get("node_normalizer", {})
        edge_normalizer_data = data.get("edge_normalizer", {})
        if isinstance(node_normalizer_data, Mapping):
            preprocessor.node_normalizer = MatrixNormalizer.from_dict(node_normalizer_data)
        if isinstance(edge_normalizer_data, Mapping):
            preprocessor.edge_normalizer = MatrixNormalizer.from_dict(edge_normalizer_data)
        return preprocessor


__all__ = ["FeaturePreprocessor", "MatrixNormalizer"]
