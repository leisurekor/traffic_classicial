"""Minimal graph autoencoder built on top of the lightweight encoder stack."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from traffic_graph.features.graph_tensor_view import PackedGraphInput
from traffic_graph.models.encoders.graphsage_gat import GraphEncoder
from traffic_graph.models.losses import ReconstructionLossWeights, reconstruction_loss
from traffic_graph.models.model_types import (
    GraphAutoEncoderConfig,
    GraphAutoEncoderOutput,
    GraphTensorBatch,
    packed_graph_to_tensors,
)


def _build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float,
) -> nn.Sequential:
    """Build a small two-layer MLP used by the decoders."""

    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


class GraphAutoEncoder(nn.Module):
    """A minimal unsupervised graph autoencoder for packed interaction graphs."""

    def __init__(
        self,
        *,
        node_input_dim: int,
        edge_input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_edge_features: bool = True,
        reconstruct_edge_features: bool = True,
        use_temporal_edge_projector: bool = False,
        temporal_edge_hidden_dim: int = 32,
        temporal_edge_field_names: tuple[str, ...] = (),
        use_edge_categorical_embeddings: bool = False,
        edge_categorical_embedding_dim: int = 8,
        edge_categorical_bucket_size: int = 128,
        loss_weights: ReconstructionLossWeights | None = None,
        config: GraphAutoEncoderConfig | None = None,
    ) -> None:
        """Construct the encoder, decoders, and reconstruction loss weights."""

        super().__init__()
        if config is not None:
            hidden_dim = config.hidden_dim
            latent_dim = config.latent_dim
            num_layers = config.num_layers
            dropout = config.dropout
            use_edge_features = config.use_edge_features
            reconstruct_edge_features = config.reconstruct_edge_features
            use_temporal_edge_projector = config.use_temporal_edge_projector
            temporal_edge_hidden_dim = config.temporal_edge_hidden_dim
            temporal_edge_field_names = config.temporal_edge_field_names
            use_edge_categorical_embeddings = config.use_edge_categorical_embeddings
            edge_categorical_embedding_dim = config.edge_categorical_embedding_dim
            edge_categorical_bucket_size = config.edge_categorical_bucket_size

        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_edge_features = use_edge_features
        self.reconstruct_edge_features = reconstruct_edge_features
        self.use_temporal_edge_projector = use_temporal_edge_projector
        self.temporal_edge_hidden_dim = temporal_edge_hidden_dim
        self.temporal_edge_field_names = temporal_edge_field_names
        self.use_edge_categorical_embeddings = use_edge_categorical_embeddings
        self.edge_categorical_embedding_dim = edge_categorical_embedding_dim
        self.edge_categorical_bucket_size = edge_categorical_bucket_size
        self.loss_weights = loss_weights or ReconstructionLossWeights()

        self.encoder = GraphEncoder(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_edge_features=use_edge_features,
            use_temporal_edge_projector=use_temporal_edge_projector,
            temporal_edge_hidden_dim=temporal_edge_hidden_dim,
            temporal_edge_field_names=temporal_edge_field_names,
            use_edge_categorical_embeddings=use_edge_categorical_embeddings,
            edge_categorical_embedding_dim=edge_categorical_embedding_dim,
            edge_categorical_bucket_size=edge_categorical_bucket_size,
        )
        self.node_decoder = _build_mlp(latent_dim, hidden_dim, node_input_dim, dropout)
        self.edge_decoder = (
            _build_mlp(latent_dim * 2, hidden_dim, edge_input_dim, dropout)
            if reconstruct_edge_features and edge_input_dim > 0
            else None
        )

    def _resolve_tensor_batch(
        self,
        inputs: GraphTensorBatch
        | PackedGraphInput
        | Sequence[PackedGraphInput],
    ) -> GraphTensorBatch:
        """Convert packed graph inputs into a tensor batch on the model device."""

        parameter = next(self.parameters())
        return packed_graph_to_tensors(
            inputs,
            device=parameter.device,
            dtype=parameter.dtype,
        )

    def encode(
        self,
        inputs: GraphTensorBatch
        | PackedGraphInput
        | Sequence[PackedGraphInput],
    ) -> tuple[GraphTensorBatch, Tensor, Tensor | None]:
        """Encode one packed graph or a batch of packed graphs."""

        tensor_batch = self._resolve_tensor_batch(inputs)
        encoder_output = self.encoder(tensor_batch)
        return tensor_batch, encoder_output.node_embeddings, encoder_output.graph_embeddings

    def decode_nodes(self, node_embeddings: Tensor) -> Tensor:
        """Reconstruct node-level input features from latent node embeddings."""

        return self.node_decoder(node_embeddings)

    def decode_edges(
        self,
        node_embeddings: Tensor,
        edge_index: Tensor,
    ) -> Tensor | None:
        """Reconstruct edge-level features from latent endpoint embeddings."""

        if self.edge_decoder is None:
            return None
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, edge_count].")
        if edge_index.shape[1] == 0:
            return node_embeddings.new_zeros((0, self.edge_input_dim))
        source_embeddings = node_embeddings[edge_index[0]]
        target_embeddings = node_embeddings[edge_index[1]]
        edge_embeddings = torch.cat([source_embeddings, target_embeddings], dim=-1)
        return self.edge_decoder(edge_embeddings)

    def forward(
        self,
        inputs: GraphTensorBatch
        | PackedGraphInput
        | Sequence[PackedGraphInput],
        *,
        targets: GraphTensorBatch
        | PackedGraphInput
        | Sequence[PackedGraphInput]
        | None = None,
    ) -> GraphAutoEncoderOutput:
        """Run a forward pass and optionally attach reconstruction losses."""

        tensor_batch, node_embeddings, graph_embeddings = self.encode(inputs)
        reconstructed_node_features = self.decode_nodes(node_embeddings)
        reconstructed_edge_features = self.decode_edges(
            node_embeddings,
            tensor_batch.edge_index,
        )
        output = GraphAutoEncoderOutput(
            node_embeddings=node_embeddings,
            graph_embeddings=graph_embeddings,
            reconstructed_node_features=reconstructed_node_features,
            reconstructed_edge_features=reconstructed_edge_features,
            tensor_batch=tensor_batch,
        )

        if targets is not None:
            target_batch = self._resolve_tensor_batch(targets)
            loss_output = reconstruction_loss(
                output,
                target_batch,
                weights=self.loss_weights,
            )
            output.loss_components = loss_output.as_dict()

        return output


__all__ = [
    "GraphAutoEncoder",
]
