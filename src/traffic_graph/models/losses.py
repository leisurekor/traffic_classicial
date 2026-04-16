"""Reconstruction losses for the minimal graph autoencoder."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from collections.abc import Mapping
from typing import Literal

import torch
from torch import Tensor, nn

from traffic_graph.features.graph_tensor_view import PackedGraphInput
from traffic_graph.models.model_types import (
    GraphAutoEncoderOutput,
    GraphTensorBatch,
    packed_graph_to_tensors,
)


@dataclass(frozen=True, slots=True)
class ReconstructionLossWeights:
    """Scalar weights used to combine node and edge reconstruction losses."""

    node_weight: float = 1.0
    edge_weight: float = 1.0

    def to_dict(self) -> dict[str, float]:
        """Serialize the loss weights into a plain mapping."""

        return {
            "node_weight": self.node_weight,
            "edge_weight": self.edge_weight,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "ReconstructionLossWeights":
        """Restore loss weights from a mapping-like payload."""

        return cls(
            node_weight=float(data.get("node_weight", 1.0)),
            edge_weight=float(data.get("edge_weight", 1.0)),
        )


@dataclass(slots=True)
class GraphAutoEncoderLossOutput:
    """Named loss components returned by the reconstruction objective."""

    total_loss: Tensor
    node_loss: Tensor
    edge_loss: Tensor
    weights: ReconstructionLossWeights

    def as_dict(self) -> dict[str, Tensor]:
        """Return a log-friendly mapping of the scalar loss components."""

        return {
            "total_loss": self.total_loss,
            "node_loss": self.node_loss,
            "edge_loss": self.edge_loss,
        }


def _safe_mse_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    discrete_mask: Tensor | None = None,
) -> Tensor:
    """Compute a masked mean-squared reconstruction loss while tolerating empty tensors."""

    if prediction.shape != target.shape:
        raise ValueError("Prediction and target tensors must have the same shape.")
    if prediction.numel() == 0:
        return prediction.new_zeros(())
    if discrete_mask is not None:
        if discrete_mask.ndim != 1 or int(discrete_mask.shape[0]) != int(prediction.shape[1]):
            raise ValueError("discrete_mask must align with the feature width.")
        continuous_mask = ~discrete_mask.to(device=prediction.device)
        if not bool(torch.any(continuous_mask).item()):
            return prediction.new_zeros(())
        prediction = prediction[:, continuous_mask]
        target = target[:, continuous_mask]
    return nn.functional.mse_loss(prediction, target, reduction="mean")


def reconstruction_loss(
    output: GraphAutoEncoderOutput,
    target: GraphTensorBatch
    | PackedGraphInput
    | Sequence[PackedGraphInput]
    | None = None,
    *,
    weights: ReconstructionLossWeights | None = None,
    edge_reduction: Literal["mean"] = "mean",
) -> GraphAutoEncoderLossOutput:
    """Compute weighted node and edge reconstruction losses.

    The `edge_reduction` parameter is reserved for future extension and currently
    accepts only the mean reduction used by the first model revision.
    """

    del edge_reduction
    loss_weights = weights or ReconstructionLossWeights()
    target_batch = (
        output.tensor_batch
        if target is None
        else packed_graph_to_tensors(target, device=output.reconstructed_node_features.device)
    )

    node_loss = _safe_mse_loss(
        output.reconstructed_node_features,
        target_batch.node_features,
        discrete_mask=target_batch.node_discrete_mask,
    )
    if output.reconstructed_edge_features is None:
        edge_loss = target_batch.edge_features.new_zeros(())
    else:
        edge_loss = _safe_mse_loss(
            output.reconstructed_edge_features,
            target_batch.edge_features,
            discrete_mask=target_batch.edge_discrete_mask,
        )

    total_loss = (
        loss_weights.node_weight * node_loss
        + loss_weights.edge_weight * edge_loss
    )
    return GraphAutoEncoderLossOutput(
        total_loss=total_loss,
        node_loss=node_loss,
        edge_loss=edge_loss,
        weights=loss_weights,
    )


__all__ = [
    "GraphAutoEncoderLossOutput",
    "ReconstructionLossWeights",
    "reconstruction_loss",
]
