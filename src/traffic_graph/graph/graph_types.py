"""Shared graph data structures for endpoint interaction graphs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, TypeAlias

EndpointType: TypeAlias = Literal["client", "server"]
EndpointPort: TypeAlias = int | tuple[int, ...]
EdgeType: TypeAlias = Literal[
    "communication",
    "association_same_src_ip",
    "association_same_dst_subnet",
    "association_same_dst_ip",
    "association_same_prefix_signature",
    "association_prefix_similarity",
]


@dataclass(frozen=True, slots=True)
class EndpointNode:
    """Endpoint node representing either a client or server communication endpoint."""

    node_id: str
    endpoint_type: EndpointType
    ip: str
    port: EndpointPort
    proto: str


@dataclass(frozen=True, slots=True)
class CommunicationEdge:
    """Graph edge record used for communication and lightweight associations."""

    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: EdgeType
    logical_flow_id: str | None
    pkt_count: int
    byte_count: int
    duration: float
    flow_count: int
    is_aggregated: bool
    source_flow_ids: tuple[str, ...]
    retry_like_count: int = 0
    retry_like_ratio: float = 0.0
    iat_hist: tuple[float, ...] = ()
    pkt_len_hist: tuple[float, ...] = ()
    flag_syn_ratio: float = 0.0
    flag_ack_ratio: float = 0.0
    flag_rst_ratio: float = 0.0
    flag_pattern_code: int = 0
    first_packet_size_pattern: int = 0
    coarse_ack_delay_mean: float = 0.0
    coarse_ack_delay_p75: float = 0.0
    ack_delay_large_gap_ratio: float = 0.0
    seq_ack_match_ratio: float = 0.0
    unmatched_seq_ratio: float = 0.0
    unmatched_ack_ratio: float = 0.0
    retry_burst_count: int = 0
    retry_burst_max_len: int = 0
    retry_like_dense_ratio: float = 0.0
    first_packet_dir_size_pattern: int = 0
    first_4_packet_pattern_code: int = 0
    small_pkt_burst_count: int = 0
    small_pkt_burst_ratio: float = 0.0
    rst_after_small_burst_indicator: int = 0
    flow_internal_embedding: tuple[float, ...] = ()
    flow_internal_packet_count: int = 0
    flow_internal_sequential_edge_count: int = 0
    flow_internal_window_edge_count: int = 0
    flow_internal_ack_edge_count: int = 0
    flow_internal_opposite_direction_edge_count: int = 0
    prefix_behavior_signature: int = 0
    flow_length_type: str = "long"


@dataclass(frozen=True, slots=True)
class InteractionGraphStats:
    """Summary statistics describing one endpoint interaction graph."""

    node_count: int
    edge_count: int
    client_node_count: int
    server_node_count: int
    aggregated_edge_count: int
    communication_edge_count: int
    association_edge_count: int
    association_same_src_ip_edge_count: int
    association_same_dst_subnet_edge_count: int
    association_same_dst_ip_edge_count: int
    association_same_prefix_signature_edge_count: int
    association_prefix_similarity_edge_count: int


def build_interaction_graph_stats(
    nodes: tuple[EndpointNode, ...],
    edges: tuple[CommunicationEdge, ...],
) -> InteractionGraphStats:
    """Compute graph-level counts from endpoint nodes and typed edges."""

    communication_edge_count = sum(edge.edge_type == "communication" for edge in edges)
    association_same_src_ip_edge_count = sum(
        edge.edge_type == "association_same_src_ip" for edge in edges
    )
    association_same_dst_subnet_edge_count = sum(
        edge.edge_type == "association_same_dst_subnet" for edge in edges
    )
    association_same_dst_ip_edge_count = sum(
        edge.edge_type == "association_same_dst_ip" for edge in edges
    )
    association_same_prefix_signature_edge_count = sum(
        edge.edge_type == "association_same_prefix_signature" for edge in edges
    )
    association_prefix_similarity_edge_count = sum(
        edge.edge_type == "association_prefix_similarity" for edge in edges
    )
    association_edge_count = (
        association_same_src_ip_edge_count + association_same_dst_subnet_edge_count
        + association_same_dst_ip_edge_count
        + association_same_prefix_signature_edge_count
        + association_prefix_similarity_edge_count
    )
    return InteractionGraphStats(
        node_count=len(nodes),
        edge_count=len(edges),
        client_node_count=sum(node.endpoint_type == "client" for node in nodes),
        server_node_count=sum(node.endpoint_type == "server" for node in nodes),
        aggregated_edge_count=sum(edge.is_aggregated for edge in edges),
        communication_edge_count=communication_edge_count,
        association_edge_count=association_edge_count,
        association_same_src_ip_edge_count=association_same_src_ip_edge_count,
        association_same_dst_subnet_edge_count=association_same_dst_subnet_edge_count,
        association_same_dst_ip_edge_count=association_same_dst_ip_edge_count,
        association_same_prefix_signature_edge_count=association_same_prefix_signature_edge_count,
        association_prefix_similarity_edge_count=association_prefix_similarity_edge_count,
    )


@dataclass(frozen=True, slots=True)
class InteractionGraph:
    """Unified graph sample wrapper around a backend graph object and metadata."""

    window_index: int
    window_start: datetime
    window_end: datetime
    graph: Any
    nodes: tuple[EndpointNode, ...]
    edges: tuple[CommunicationEdge, ...]
    stats: InteractionGraphStats

    @property
    def node_count(self) -> int:
        """Return the number of endpoint nodes."""

        return self.stats.node_count

    @property
    def edge_count(self) -> int:
        """Return the number of total graph edges."""

        return self.stats.edge_count


GraphSample = InteractionGraph


__all__ = [
    "CommunicationEdge",
    "EdgeType",
    "EndpointNode",
    "EndpointPort",
    "EndpointType",
    "GraphSample",
    "InteractionGraph",
    "InteractionGraphStats",
    "build_interaction_graph_stats",
]
