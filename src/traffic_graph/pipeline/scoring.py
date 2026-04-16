"""Anomaly scoring helpers for graph autoencoder reconstructions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import numpy as np

from traffic_graph.graph.graph_types import InteractionGraph

ScoreReduction: TypeAlias = Literal["mean", "max"]


def _as_float_array(values: np.ndarray | Sequence[float]) -> np.ndarray:
    """Convert score-like values into a one-dimensional float array."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("Score arrays must be one-dimensional.")
    return array


def _rowwise_mse(
    reference: np.ndarray | Sequence[Sequence[float]],
    reconstruction: np.ndarray | Sequence[Sequence[float]] | None,
    *,
    discrete_mask: np.ndarray | Sequence[bool] | None = None,
) -> np.ndarray:
    """Compute row-wise mean-squared reconstruction error."""

    reference_array = np.asarray(reference, dtype=float)
    if reference_array.ndim != 2:
        raise ValueError("Feature matrices must be two-dimensional.")
    if reference_array.shape[0] == 0:
        return np.zeros((0,), dtype=float)

    if discrete_mask is not None:
        mask_array = np.asarray(discrete_mask, dtype=bool)
        if mask_array.ndim != 1 or mask_array.shape[0] != reference_array.shape[1]:
            raise ValueError("discrete_mask must align with the feature width.")
        continuous_mask = ~mask_array
        if not np.any(continuous_mask):
            return np.zeros((reference_array.shape[0],), dtype=float)
        reference_array = reference_array[:, continuous_mask]

    if reconstruction is None:
        return np.zeros((reference_array.shape[0],), dtype=float)

    reconstruction_array = np.asarray(reconstruction, dtype=float)
    if discrete_mask is not None:
        reconstruction_array = reconstruction_array[:, continuous_mask]
    if reconstruction_array.shape != reference_array.shape:
        raise ValueError("Reference and reconstructed feature matrices must match.")

    residual = reference_array - reconstruction_array
    return np.mean(residual * residual, axis=1)


def _graph_score_from_node_scores(
    node_scores: np.ndarray,
    reduction: ScoreReduction = "mean",
) -> float:
    """Reduce a node score vector into a single graph score."""

    if node_scores.size == 0:
        return 0.0
    if reduction == "max":
        return float(np.max(node_scores))
    if reduction == "mean":
        return float(np.mean(node_scores))
    raise ValueError("reduction must be either 'mean' or 'max'.")


def compute_node_anomaly_scores(
    node_features: np.ndarray | Sequence[Sequence[float]],
    reconstructed_node_features: np.ndarray | Sequence[Sequence[float]],
    *,
    discrete_mask: np.ndarray | Sequence[bool] | None = None,
) -> np.ndarray:
    """Compute per-node anomaly scores from node reconstruction error."""

    return _rowwise_mse(
        node_features,
        reconstructed_node_features,
        discrete_mask=discrete_mask,
    )


def compute_edge_anomaly_scores(
    edge_features: np.ndarray | Sequence[Sequence[float]],
    reconstructed_edge_features: np.ndarray | Sequence[Sequence[float]] | None,
    *,
    discrete_mask: np.ndarray | Sequence[bool] | None = None,
) -> np.ndarray:
    """Compute per-edge anomaly scores from edge reconstruction error.

    When edge reconstruction is unavailable, a zero-filled placeholder vector is
    returned so that downstream consumers can still export a consistent report.
    """

    return _rowwise_mse(
        edge_features,
        reconstructed_edge_features,
        discrete_mask=discrete_mask,
    )


def compute_graph_anomaly_scores(
    node_scores: np.ndarray | Sequence[float],
    graph_ptr: Sequence[int] | np.ndarray | None = None,
    *,
    reduction: ScoreReduction = "mean",
) -> float | np.ndarray:
    """Reduce node anomaly scores into graph-level anomaly scores.

    When `graph_ptr` is omitted, the function treats `node_scores` as belonging to
    a single graph and returns one scalar. When `graph_ptr` is present, it is
    interpreted as a prefix-sum vector over concatenated graph nodes.
    """

    scores = _as_float_array(node_scores)
    if graph_ptr is None:
        return _graph_score_from_node_scores(scores, reduction=reduction)

    ptr = np.asarray(graph_ptr, dtype=int)
    if ptr.ndim != 1 or ptr.size < 2:
        raise ValueError("graph_ptr must be a 1D prefix-sum vector.")

    graph_scores: list[float] = []
    for start, end in zip(ptr[:-1], ptr[1:]):
        graph_scores.append(_graph_score_from_node_scores(scores[start:end], reduction))
    return np.asarray(graph_scores, dtype=float)


def build_node_score_row(
    graph_index: int,
    graph_sample: InteractionGraph,
    node_index: int,
    score: float,
) -> dict[str, object]:
    """Build one serialized node-score row."""

    node = graph_sample.nodes[node_index]
    return {
        "graph_index": graph_index,
        "window_index": graph_sample.window_index,
        "window_start": graph_sample.window_start.isoformat(),
        "window_end": graph_sample.window_end.isoformat(),
        "node_index": node_index,
        "node_id": node.node_id,
        "endpoint_type": node.endpoint_type,
        "ip": node.ip,
        "port": node.port if not isinstance(node.port, tuple) else list(node.port),
        "proto": node.proto,
        "node_anomaly_score": float(score),
    }


def build_edge_score_row(
    graph_index: int,
    graph_sample: InteractionGraph,
    edge_index: int,
    score: float,
) -> dict[str, object]:
    """Build one serialized edge-score row."""

    edge = graph_sample.edges[edge_index]
    return {
        "graph_index": graph_index,
        "window_index": graph_sample.window_index,
        "window_start": graph_sample.window_start.isoformat(),
        "window_end": graph_sample.window_end.isoformat(),
        "edge_index": edge_index,
        "edge_id": edge.edge_id,
        "edge_type": edge.edge_type,
        "logical_flow_id": edge.logical_flow_id,
        "source_node_id": edge.source_node_id,
        "target_node_id": edge.target_node_id,
        "pkt_count": edge.pkt_count,
        "byte_count": edge.byte_count,
        "duration": edge.duration,
        "flow_count": edge.flow_count,
        "is_aggregated": edge.is_aggregated,
        "source_flow_ids": list(edge.source_flow_ids),
        "edge_anomaly_score": float(score),
    }


def build_flow_score_row(
    graph_index: int,
    graph_sample: InteractionGraph,
    edge_index: int,
    score: float,
) -> dict[str, object]:
    """Build one serialized flow-score row for a communication edge."""

    edge = graph_sample.edges[edge_index]
    if edge.edge_type != "communication":
        raise ValueError("flow scores can only be built from communication edges.")
    row = build_edge_score_row(graph_index, graph_sample, edge_index, score)
    row["flow_anomaly_score"] = row.pop("edge_anomaly_score")
    return row


def build_graph_score_row(
    graph_index: int,
    graph_sample: InteractionGraph,
    score: float,
) -> dict[str, object]:
    """Build one serialized graph-score row."""

    return {
        "graph_index": graph_index,
        "window_index": graph_sample.window_index,
        "window_start": graph_sample.window_start.isoformat(),
        "window_end": graph_sample.window_end.isoformat(),
        "graph_anomaly_score": float(score),
    }


def build_node_score_rows(
    graph_index: int,
    graph_sample: InteractionGraph,
    node_scores: Sequence[float] | np.ndarray,
) -> list[dict[str, object]]:
    """Build per-node anomaly-score rows for one graph sample."""

    scores = _as_float_array(node_scores)
    if scores.size != len(graph_sample.nodes):
        raise ValueError("Node score count must match the graph node count.")
    return [
        build_node_score_row(graph_index, graph_sample, node_index, float(score))
        for node_index, score in enumerate(scores)
    ]


def build_edge_score_rows(
    graph_index: int,
    graph_sample: InteractionGraph,
    edge_scores: Sequence[float] | np.ndarray,
) -> list[dict[str, object]]:
    """Build per-edge anomaly-score rows for one graph sample."""

    scores = _as_float_array(edge_scores)
    if scores.size != len(graph_sample.edges):
        raise ValueError("Edge score count must match the graph edge count.")
    return [
        build_edge_score_row(graph_index, graph_sample, edge_index, float(score))
        for edge_index, score in enumerate(scores)
    ]


def build_flow_score_rows(
    graph_index: int,
    graph_sample: InteractionGraph,
    edge_scores: Sequence[float] | np.ndarray,
) -> list[dict[str, object]]:
    """Build flow-level score rows from communication edges only."""

    scores = _as_float_array(edge_scores)
    if scores.size != len(graph_sample.edges):
        raise ValueError("Edge score count must match the graph edge count.")

    rows: list[dict[str, object]] = []
    for edge_index, edge in enumerate(graph_sample.edges):
        if edge.edge_type != "communication":
            continue
        rows.append(
            build_flow_score_row(
                graph_index,
                graph_sample,
                edge_index,
                float(scores[edge_index]),
            )
        )
    return rows


__all__ = [
    "build_edge_score_row",
    "build_edge_score_rows",
    "build_flow_score_row",
    "build_flow_score_rows",
    "build_graph_score_row",
    "build_node_score_row",
    "build_node_score_rows",
    "compute_edge_anomaly_scores",
    "compute_graph_anomaly_scores",
    "compute_node_anomaly_scores",
]
