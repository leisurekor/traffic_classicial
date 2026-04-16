"""Lightweight GraphSAGE-style encoder for packed interaction graphs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from traffic_graph.models.model_types import GraphTensorBatch


def _mean_pool(x: Tensor, batch: Tensor, num_graphs: int) -> Tensor:
    """Mean-pool node embeddings into graph-level embeddings."""

    if x.numel() == 0:
        return x.new_zeros((num_graphs, x.shape[1]))
    pooled = x.new_zeros((num_graphs, x.shape[1]))
    pooled.index_add_(0, batch, x)
    counts = x.new_zeros((num_graphs,), dtype=x.dtype)
    counts.index_add_(0, batch, torch.ones_like(batch, dtype=x.dtype))
    return pooled / counts.clamp_min(1.0).unsqueeze(-1)


class GraphSAGELayer(nn.Module):
    """A minimal GraphSAGE-style message passing layer."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        edge_input_dim: int = 0,
        *,
        use_edge_features: bool = False,
        dropout: float = 0.0,
        activate: bool = True,
    ) -> None:
        """Construct the message passing layer."""

        super().__init__()
        self.use_edge_features = use_edge_features and edge_input_dim > 0
        message_input_dim = input_dim + edge_input_dim if self.use_edge_features else input_dim
        self.self_projection = nn.Linear(input_dim, output_dim)
        self.message_projection = nn.Linear(message_input_dim, output_dim)
        self.normalization = nn.LayerNorm(output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.activate = activate

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Tensor | None = None,
    ) -> Tensor:
        """Propagate information from source nodes to target nodes."""

        if node_features.ndim != 2:
            raise ValueError("node_features must be a 2D tensor.")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, edge_count].")

        num_nodes = int(node_features.shape[0])
        if num_nodes == 0:
            return node_features.new_zeros((0, self.self_projection.out_features))

        source_index = edge_index[0]
        target_index = edge_index[1]

        if source_index.numel() == 0:
            aggregated_messages = node_features.new_zeros(
                (num_nodes, self.message_projection.out_features)
            )
        else:
            messages = node_features[source_index]
            if self.use_edge_features:
                if edge_features is None:
                    raise ValueError(
                        "edge_features are required when use_edge_features=True."
                    )
                if edge_features.ndim != 2:
                    raise ValueError("edge_features must be a 2D tensor.")
                if edge_features.shape[0] != source_index.shape[0]:
                    raise ValueError("edge_features must align with edge_index columns.")
                messages = torch.cat([messages, edge_features], dim=-1)
            messages = self.message_projection(messages)
            aggregated_messages = node_features.new_zeros(
                (num_nodes, messages.shape[1])
            )
            aggregated_messages.index_add_(0, target_index, messages)
            degree = node_features.new_zeros((num_nodes,), dtype=node_features.dtype)
            degree.index_add_(
                0,
                target_index,
                torch.ones_like(target_index, dtype=node_features.dtype),
            )
            aggregated_messages = aggregated_messages / degree.clamp_min(1.0).unsqueeze(-1)

        combined = self.self_projection(node_features) + aggregated_messages
        combined = self.normalization(combined)
        if self.activate:
            combined = self.activation(combined)
        combined = self.dropout(combined)
        return combined


class TemporalEdgeProjector(nn.Module):
    """Project selected temporal edge fields back into the edge feature space."""

    def __init__(
        self,
        *,
        edge_input_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        """Construct a lightweight residual projector for edge features."""

        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, edge_input_dim),
        )

    def forward(
        self,
        edge_features: Tensor,
        field_mask: Tensor,
    ) -> Tensor:
        """Project only the masked temporal edge fields."""

        if edge_features.ndim != 2:
            raise ValueError("edge_features must be a 2D tensor.")
        if edge_features.shape[1] != int(field_mask.shape[0]):
            raise ValueError("field_mask must align with edge feature width.")
        if edge_features.shape[0] == 0:
            return edge_features
        if torch.count_nonzero(field_mask).item() == 0:
            return edge_features.new_zeros(edge_features.shape)
        masked_edge_features = edge_features * field_mask.unsqueeze(0)
        return self.network(masked_edge_features)


class CategoricalEdgeProjector(nn.Module):
    """Embed hashed categorical edge fields back into the numeric edge space."""

    def __init__(
        self,
        *,
        edge_input_dim: int,
        embedding_dim: int,
        bucket_size: int,
        dropout: float,
    ) -> None:
        """Construct a compact hashed embedding projector for discrete edge fields."""

        super().__init__()
        self.bucket_size = max(16, bucket_size)
        self.embedding = nn.Embedding(self.bucket_size, max(1, embedding_dim))
        self.projection = nn.Sequential(
            nn.Linear(max(1, embedding_dim), edge_input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_input_dim, edge_input_dim),
        )

    def forward(
        self,
        edge_features: Tensor,
        discrete_mask: Tensor,
    ) -> Tensor:
        """Project hashed categorical edge values into the edge feature space."""

        if edge_features.ndim != 2:
            raise ValueError("edge_features must be a 2D tensor.")
        if edge_features.shape[1] != int(discrete_mask.shape[0]):
            raise ValueError("discrete_mask must align with edge feature width.")
        if edge_features.shape[0] == 0:
            return edge_features.new_zeros(edge_features.shape)

        discrete_indices = torch.nonzero(discrete_mask, as_tuple=False).flatten()
        if discrete_indices.numel() == 0:
            return edge_features.new_zeros(edge_features.shape)

        raw_codes = torch.round(edge_features[:, discrete_indices]).to(torch.long).abs()
        field_offsets = (discrete_indices.to(torch.long) + 1).unsqueeze(0) * 131
        hashed_codes = (raw_codes + field_offsets) % self.bucket_size
        embedded = self.embedding(hashed_codes)
        pooled = embedded.mean(dim=1)
        return self.projection(pooled)


@dataclass(slots=True)
class GraphEncoderOutput:
    """Convenience wrapper for encoder outputs."""

    node_embeddings: Tensor
    graph_embeddings: Tensor | None


class GraphEncoder(nn.Module):
    """Stacked GraphSAGE-style encoder for node and graph embeddings."""

    def __init__(
        self,
        *,
        node_input_dim: int,
        edge_input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_edge_features: bool = True,
        use_temporal_edge_projector: bool = False,
        temporal_edge_hidden_dim: int = 32,
        temporal_edge_field_names: tuple[str, ...] = (),
        use_edge_categorical_embeddings: bool = False,
        edge_categorical_embedding_dim: int = 8,
        edge_categorical_bucket_size: int = 128,
    ) -> None:
        """Construct the encoder stack."""

        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be greater than zero.")

        layer_dims = [node_input_dim]
        if num_layers > 1:
            layer_dims.extend([hidden_dim] * (num_layers - 1))
        layer_dims.append(latent_dim)

        layers: list[GraphSAGELayer] = []
        for layer_index in range(num_layers):
            input_dim = layer_dims[layer_index]
            output_dim = layer_dims[layer_index + 1]
            layers.append(
                GraphSAGELayer(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    edge_input_dim=edge_input_dim,
                    use_edge_features=use_edge_features,
                    dropout=dropout,
                    activate=layer_index < num_layers - 1,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.use_edge_features = use_edge_features
        self.latent_dim = latent_dim
        self.temporal_edge_field_names = temporal_edge_field_names
        self.use_edge_categorical_embeddings = (
            use_edge_categorical_embeddings and self.use_edge_features and edge_input_dim > 0
        )
        self.use_temporal_edge_projector = (
            use_temporal_edge_projector and self.use_edge_features and edge_input_dim > 0
        )
        self.categorical_edge_projector = (
            CategoricalEdgeProjector(
                edge_input_dim=edge_input_dim,
                embedding_dim=max(1, edge_categorical_embedding_dim),
                bucket_size=max(16, edge_categorical_bucket_size),
                dropout=dropout,
            )
            if self.use_edge_categorical_embeddings
            else None
        )
        self.temporal_edge_projector = (
            TemporalEdgeProjector(
                edge_input_dim=edge_input_dim,
                hidden_dim=max(1, temporal_edge_hidden_dim),
                dropout=dropout,
            )
            if self.use_temporal_edge_projector
            else None
        )

    def _temporal_field_mask(
        self,
        edge_feature_fields: tuple[str, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Build a 1D mask over edge feature columns used by the temporal branch."""

        if not edge_feature_fields:
            return torch.zeros((0,), device=device, dtype=dtype)
        selected = set(self.temporal_edge_field_names)
        mask_values = [
            1.0 if field_name in selected else 0.0 for field_name in edge_feature_fields
        ]
        return torch.tensor(mask_values, device=device, dtype=dtype)

    def forward(self, tensor_batch: GraphTensorBatch) -> GraphEncoderOutput:
        """Encode packed graph tensors into node and graph embeddings."""

        node_embeddings = tensor_batch.node_features
        edge_features = tensor_batch.edge_features if self.use_edge_features else None
        if (
            edge_features is not None
            and tensor_batch.edge_feature_fields
            and self.categorical_edge_projector is not None
        ):
            raw_edge_features = edge_features
            discrete_mask = tensor_batch.edge_discrete_mask.to(
                device=edge_features.device
            )
            edge_features = edge_features.masked_fill(discrete_mask.unsqueeze(0), 0.0)
            if self.categorical_edge_projector is not None:
                edge_features = edge_features + self.categorical_edge_projector(
                    raw_edge_features,
                    discrete_mask,
                )
        if (
            edge_features is not None
            and self.temporal_edge_projector is not None
            and tensor_batch.edge_feature_fields
        ):
            temporal_mask = self._temporal_field_mask(
                tensor_batch.edge_feature_fields,
                device=edge_features.device,
                dtype=edge_features.dtype,
            )
            edge_features = edge_features + self.temporal_edge_projector(
                edge_features,
                temporal_mask,
            )

        for layer in self.layers:
            node_embeddings = layer(
                node_embeddings,
                tensor_batch.edge_index,
                edge_features=edge_features,
            )

        graph_embeddings: Tensor | None
        if tensor_batch.num_graphs > 0:
            graph_embeddings = _mean_pool(
                node_embeddings,
                tensor_batch.node_batch,
                tensor_batch.num_graphs,
            )
        else:  # pragma: no cover - defensive fallback
            graph_embeddings = None

        return GraphEncoderOutput(
            node_embeddings=node_embeddings,
            graph_embeddings=graph_embeddings,
        )


__all__ = [
    "GraphEncoder",
    "GraphEncoderOutput",
    "GraphSAGELayer",
    "CategoricalEdgeProjector",
    "TemporalEdgeProjector",
]
