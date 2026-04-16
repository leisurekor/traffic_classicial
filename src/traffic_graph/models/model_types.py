"""Typed data structures and tensor conversion helpers for graph autoencoders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from traffic_graph.features.graph_tensor_view import (
    PackedGraphInput,
    PackedGraphMetadata,
)


DEFAULT_TEMPORAL_EDGE_FIELD_NAMES: tuple[str, ...] = (
    "coarse_ack_delay_mean",
    "coarse_ack_delay_p75",
    "ack_delay_large_gap_ratio",
    "seq_ack_match_ratio",
    "unmatched_seq_ratio",
    "unmatched_ack_ratio",
    "retry_burst_count",
    "retry_burst_max_len",
    "retry_like_dense_ratio",
    "small_pkt_burst_count",
    "small_pkt_burst_ratio",
)
"""Default edge feature names routed through the optional temporal branch."""


@dataclass(frozen=True, slots=True)
class GraphAutoEncoderConfig:
    """Minimal configuration for the first graph autoencoder revision."""

    hidden_dim: int = 64
    latent_dim: int = 32
    num_layers: int = 2
    dropout: float = 0.1
    use_edge_features: bool = True
    reconstruct_edge_features: bool = True
    use_temporal_edge_projector: bool = False
    temporal_edge_hidden_dim: int = 32
    temporal_edge_field_names: tuple[str, ...] = DEFAULT_TEMPORAL_EDGE_FIELD_NAMES
    use_edge_categorical_embeddings: bool = False
    edge_categorical_embedding_dim: int = 8
    edge_categorical_bucket_size: int = 128

    def to_dict(self) -> dict[str, object]:
        """Serialize the configuration into a plain mapping."""

        return {
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "use_edge_features": self.use_edge_features,
            "reconstruct_edge_features": self.reconstruct_edge_features,
            "use_temporal_edge_projector": self.use_temporal_edge_projector,
            "temporal_edge_hidden_dim": self.temporal_edge_hidden_dim,
            "temporal_edge_field_names": list(self.temporal_edge_field_names),
            "use_edge_categorical_embeddings": self.use_edge_categorical_embeddings,
            "edge_categorical_embedding_dim": self.edge_categorical_embedding_dim,
            "edge_categorical_bucket_size": self.edge_categorical_bucket_size,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "GraphAutoEncoderConfig":
        """Construct a configuration from a mapping-like payload."""

        hidden_dim = int(data.get("hidden_dim", 64))
        latent_dim = int(data.get("latent_dim", 32))
        num_layers = int(data.get("num_layers", 2))
        dropout = float(data.get("dropout", 0.1))
        use_edge_features = bool(data.get("use_edge_features", True))
        reconstruct_edge_features = bool(
            data.get("reconstruct_edge_features", True)
        )
        use_temporal_edge_projector = bool(
            data.get("use_temporal_edge_projector", False)
        )
        temporal_edge_hidden_dim = int(data.get("temporal_edge_hidden_dim", 32))
        use_edge_categorical_embeddings = bool(
            data.get("use_edge_categorical_embeddings", False)
        )
        edge_categorical_embedding_dim = int(
            data.get("edge_categorical_embedding_dim", 8)
        )
        edge_categorical_bucket_size = int(
            data.get("edge_categorical_bucket_size", 128)
        )
        temporal_edge_field_names_raw = data.get(
            "temporal_edge_field_names",
            DEFAULT_TEMPORAL_EDGE_FIELD_NAMES,
        )
        if isinstance(temporal_edge_field_names_raw, str):
            temporal_edge_field_names = (
                (temporal_edge_field_names_raw.strip(),)
                if temporal_edge_field_names_raw.strip()
                else DEFAULT_TEMPORAL_EDGE_FIELD_NAMES
            )
        elif isinstance(temporal_edge_field_names_raw, Sequence):
            temporal_edge_field_names = tuple(
                str(item).strip()
                for item in temporal_edge_field_names_raw
                if str(item).strip()
            ) or DEFAULT_TEMPORAL_EDGE_FIELD_NAMES
        else:
            temporal_edge_field_names = DEFAULT_TEMPORAL_EDGE_FIELD_NAMES
        return cls(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_edge_features=use_edge_features,
            reconstruct_edge_features=reconstruct_edge_features,
            use_temporal_edge_projector=use_temporal_edge_projector,
            temporal_edge_hidden_dim=temporal_edge_hidden_dim,
            temporal_edge_field_names=temporal_edge_field_names,
            use_edge_categorical_embeddings=use_edge_categorical_embeddings,
            edge_categorical_embedding_dim=max(1, edge_categorical_embedding_dim),
            edge_categorical_bucket_size=max(16, edge_categorical_bucket_size),
        )


@dataclass(slots=True)
class GraphTensorBatch:
    """Torch tensor view of one or more packed graph samples."""

    node_features: Tensor
    edge_features: Tensor
    edge_index: Tensor
    node_batch: Tensor
    edge_batch: Tensor
    graph_ptr: Tensor
    edge_ptr: Tensor
    node_counts: tuple[int, ...]
    edge_counts: tuple[int, ...]
    node_ids: tuple[tuple[str, ...], ...]
    edge_ids: tuple[tuple[str, ...], ...]
    node_id_to_index: tuple[dict[str, int], ...]
    edge_id_to_index: tuple[dict[str, int], ...]
    edge_types: Tensor
    node_feature_fields: tuple[str, ...]
    edge_feature_fields: tuple[str, ...]
    node_discrete_mask: Tensor
    edge_discrete_mask: Tensor
    graph_metadata: tuple[PackedGraphMetadata, ...]

    def __post_init__(self) -> None:
        """Validate the most important tensor shapes eagerly."""

        if self.node_features.ndim != 2:
            raise ValueError("node_features must be a 2D tensor.")
        if self.edge_features.ndim != 2:
            raise ValueError("edge_features must be a 2D tensor.")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, edge_count].")
        if self.node_batch.ndim != 1:
            raise ValueError("node_batch must be a 1D tensor.")
        if self.edge_batch.ndim != 1:
            raise ValueError("edge_batch must be a 1D tensor.")
        if self.graph_ptr.ndim != 1 or self.graph_ptr.shape[0] != self.num_graphs + 1:
            raise ValueError("graph_ptr must have length num_graphs + 1.")
        if self.edge_ptr.ndim != 1 or self.edge_ptr.shape[0] != self.num_graphs + 1:
            raise ValueError("edge_ptr must have length num_graphs + 1.")
        if self.node_discrete_mask.ndim != 1:
            raise ValueError("node_discrete_mask must be a 1D tensor.")
        if self.edge_discrete_mask.ndim != 1:
            raise ValueError("edge_discrete_mask must be a 1D tensor.")
        if self.edge_types.ndim != 1:
            raise ValueError("edge_types must be a 1D tensor.")

    @property
    def num_graphs(self) -> int:
        """Return the number of packed graphs in the batch."""

        return len(self.graph_metadata)

    @property
    def num_nodes(self) -> int:
        """Return the total node count across the packed graphs."""

        return int(self.node_features.shape[0])

    @property
    def num_edges(self) -> int:
        """Return the total edge count across the packed graphs."""

        return int(self.edge_features.shape[0])

    @property
    def node_feature_dim(self) -> int:
        """Return the width of the node feature matrix."""

        return int(self.node_features.shape[1]) if self.node_features.ndim == 2 else 0

    @property
    def edge_feature_dim(self) -> int:
        """Return the width of the edge feature matrix."""

        return int(self.edge_features.shape[1]) if self.edge_features.ndim == 2 else 0

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "GraphTensorBatch":
        """Move the tensor batch to a device, optionally changing float dtype."""

        if dtype is None:
            node_features = self.node_features.to(device=device)
            edge_features = self.edge_features.to(device=device)
        else:
            node_features = self.node_features.to(device=device, dtype=dtype)
            edge_features = self.edge_features.to(device=device, dtype=dtype)
        edge_index = self.edge_index.to(device=device)
        node_batch = self.node_batch.to(device=device)
        edge_batch = self.edge_batch.to(device=device)
        graph_ptr = self.graph_ptr.to(device=device)
        edge_ptr = self.edge_ptr.to(device=device)
        edge_types = self.edge_types.to(device=device)
        node_discrete_mask = self.node_discrete_mask.to(device=device)
        edge_discrete_mask = self.edge_discrete_mask.to(device=device)
        return GraphTensorBatch(
            node_features=node_features,
            edge_features=edge_features,
            edge_index=edge_index,
            node_batch=node_batch,
            edge_batch=edge_batch,
            graph_ptr=graph_ptr,
            edge_ptr=edge_ptr,
            node_counts=self.node_counts,
            edge_counts=self.edge_counts,
            node_ids=self.node_ids,
            edge_ids=self.edge_ids,
            node_id_to_index=self.node_id_to_index,
            edge_id_to_index=self.edge_id_to_index,
            edge_types=edge_types,
            node_feature_fields=self.node_feature_fields,
            edge_feature_fields=self.edge_feature_fields,
            node_discrete_mask=node_discrete_mask,
            edge_discrete_mask=edge_discrete_mask,
            graph_metadata=self.graph_metadata,
        )

    @classmethod
    def from_packed_graphs(
        cls,
        graphs: PackedGraphInput | Sequence[PackedGraphInput],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "GraphTensorBatch":
        """Construct a tensor batch from one packed graph or a graph sequence."""

        return packed_graph_to_tensors(graphs, device=device, dtype=dtype)


@dataclass(slots=True)
class GraphAutoEncoderOutput:
    """Forward pass output produced by the graph autoencoder."""

    node_embeddings: Tensor
    graph_embeddings: Tensor | None
    reconstructed_node_features: Tensor
    reconstructed_edge_features: Tensor | None
    tensor_batch: GraphTensorBatch
    loss_components: dict[str, Tensor] | None = None


def _as_float_tensor(
    value: np.ndarray | Sequence[float] | Tensor,
    *,
    device: torch.device | str | None,
    dtype: torch.dtype,
) -> Tensor:
    """Convert numpy or tensor input into a floating-point tensor."""

    return torch.as_tensor(value, dtype=dtype, device=device)


def _as_long_tensor(
    value: Sequence[int] | np.ndarray | Tensor,
    *,
    device: torch.device | str | None,
) -> Tensor:
    """Convert index-like data into a long tensor."""

    return torch.as_tensor(value, dtype=torch.long, device=device)


def _graph_metadata(packed_graph: PackedGraphInput) -> PackedGraphMetadata:
    """Return the packed metadata object associated with one graph."""

    return packed_graph.metadata


def _validate_feature_fields(
    graphs: Sequence[PackedGraphInput],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Ensure that all packed graphs share the same feature field ordering."""

    node_field_names = graphs[0].node_feature_fields
    edge_field_names = graphs[0].edge_feature_fields
    for graph in graphs[1:]:
        if graph.node_feature_fields != node_field_names:
            raise ValueError("All graphs must share the same node feature fields.")
        if graph.edge_feature_fields != edge_field_names:
            raise ValueError("All graphs must share the same edge feature fields.")
    return node_field_names, edge_field_names


def _edge_type_tensor(
    packed_graph: PackedGraphInput,
    *,
    device: torch.device | str | None,
    edge_feature_fields: tuple[str, ...],
) -> Tensor:
    """Derive stable edge-type encodings from packed graph data."""

    if packed_graph.edge_types:
        return _as_long_tensor(packed_graph.edge_types, device=device)

    try:
        edge_type_index = edge_feature_fields.index("edge_type")
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError("edge_type field is required when edge_types are absent.") from exc
    return _as_long_tensor(packed_graph.edge_features[:, edge_type_index], device=device)


def packed_graph_to_tensors(
    graphs: PackedGraphInput | Sequence[PackedGraphInput] | GraphTensorBatch,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> GraphTensorBatch:
    """Convert packed graph inputs into a batched torch tensor view."""

    if isinstance(graphs, GraphTensorBatch):
        return graphs.to(device=device, dtype=dtype)

    if isinstance(graphs, PackedGraphInput):
        graph_list = (graphs,)
    else:
        graph_list = tuple(graphs)

    if not graph_list:
        raise ValueError("At least one packed graph is required.")

    node_field_names, edge_field_names = _validate_feature_fields(graph_list)

    node_feature_tensors: list[Tensor] = []
    edge_feature_tensors: list[Tensor] = []
    edge_index_tensors: list[Tensor] = []
    node_batch_tensors: list[Tensor] = []
    edge_batch_tensors: list[Tensor] = []
    edge_type_tensors: list[Tensor] = []
    node_counts: list[int] = []
    edge_counts: list[int] = []
    node_ids: list[tuple[str, ...]] = []
    edge_ids: list[tuple[str, ...]] = []
    node_id_maps: list[dict[str, int]] = []
    edge_id_maps: list[dict[str, int]] = []
    graph_metadata: list[PackedGraphMetadata] = []
    graph_ptr: list[int] = [0]
    edge_ptr: list[int] = [0]

    node_offset = 0
    for graph_index, packed_graph in enumerate(graph_list):
        node_features = _as_float_tensor(
            packed_graph.node_features,
            device=device,
            dtype=dtype,
        )
        edge_features = _as_float_tensor(
            packed_graph.edge_features,
            device=device,
            dtype=dtype,
        )
        edge_index = _as_long_tensor(packed_graph.edge_index, device=device)

        if node_features.ndim != 2:
            raise ValueError("Packed node features must be two-dimensional.")
        if edge_features.ndim != 2:
            raise ValueError("Packed edge features must be two-dimensional.")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("Packed edge index must have shape [2, edge_count].")

        node_count = int(node_features.shape[0])
        edge_count = int(edge_features.shape[0])

        node_feature_tensors.append(node_features)
        edge_feature_tensors.append(edge_features)
        node_batch_tensors.append(
            torch.full((node_count,), graph_index, dtype=torch.long, device=device)
        )
        edge_batch_tensors.append(
            torch.full((edge_count,), graph_index, dtype=torch.long, device=device)
        )
        if edge_count > 0:
            edge_index_tensors.append(edge_index + node_offset)
        else:
            edge_index_tensors.append(edge_index)
        edge_type_tensors.append(
            _edge_type_tensor(
                packed_graph,
                device=device,
                edge_feature_fields=edge_field_names,
            )
        )

        node_counts.append(node_count)
        edge_counts.append(edge_count)
        node_ids.append(tuple(packed_graph.node_ids))
        edge_ids.append(tuple(packed_graph.edge_ids))
        node_id_maps.append(dict(packed_graph.node_id_to_index))
        edge_id_maps.append(dict(packed_graph.edge_id_to_index))
        graph_metadata.append(_graph_metadata(packed_graph))
        graph_ptr.append(graph_ptr[-1] + node_count)
        edge_ptr.append(edge_ptr[-1] + edge_count)
        node_offset += node_count

    node_features = torch.cat(node_feature_tensors, dim=0)
    edge_features = torch.cat(edge_feature_tensors, dim=0)
    edge_index = torch.cat(edge_index_tensors, dim=1)
    node_batch = torch.cat(node_batch_tensors, dim=0)
    edge_batch = torch.cat(edge_batch_tensors, dim=0)
    edge_types = torch.cat(edge_type_tensors, dim=0)

    node_discrete_mask = torch.as_tensor(
        graph_list[0].node_discrete_mask,
        dtype=torch.bool,
        device=device,
    )
    edge_discrete_mask = torch.as_tensor(
        graph_list[0].edge_discrete_mask,
        dtype=torch.bool,
        device=device,
    )

    return GraphTensorBatch(
        node_features=node_features,
        edge_features=edge_features,
        edge_index=edge_index,
        node_batch=node_batch,
        edge_batch=edge_batch,
        graph_ptr=torch.as_tensor(graph_ptr, dtype=torch.long, device=device),
        edge_ptr=torch.as_tensor(edge_ptr, dtype=torch.long, device=device),
        node_counts=tuple(node_counts),
        edge_counts=tuple(edge_counts),
        node_ids=tuple(node_ids),
        edge_ids=tuple(edge_ids),
        node_id_to_index=tuple(node_id_maps),
        edge_id_to_index=tuple(edge_id_maps),
        edge_types=edge_types,
        node_feature_fields=node_field_names,
        edge_feature_fields=edge_field_names,
        node_discrete_mask=node_discrete_mask,
        edge_discrete_mask=edge_discrete_mask,
        graph_metadata=tuple(graph_metadata),
    )


__all__ = [
    "DEFAULT_TEMPORAL_EDGE_FIELD_NAMES",
    "GraphAutoEncoderConfig",
    "GraphAutoEncoderOutput",
    "GraphTensorBatch",
    "packed_graph_to_tensors",
]
