"""Graph-building primitives for the flow interaction layer."""

from traffic_graph.graph.association_edges import (
    add_association_edges,
    build_association_edges_same_dst_subnet,
    build_association_edges_same_src_ip,
)
from traffic_graph.graph.builder import FlowInteractionGraphBuilder, GraphSnapshot
from traffic_graph.graph.endpoint_graph import (
    EndpointGraphBuilder,
    build_endpoint_graph,
    build_endpoint_graphs,
    summarize_graph,
)
from traffic_graph.graph.graph_types import (
    CommunicationEdge,
    EdgeType,
    EndpointNode,
    GraphSample,
    InteractionGraph,
    InteractionGraphStats,
)

__all__ = [
    "CommunicationEdge",
    "EdgeType",
    "EndpointGraphBuilder",
    "EndpointNode",
    "FlowInteractionGraphBuilder",
    "GraphSample",
    "GraphSnapshot",
    "InteractionGraph",
    "InteractionGraphStats",
    "add_association_edges",
    "build_association_edges_same_dst_subnet",
    "build_association_edges_same_src_ip",
    "build_endpoint_graph",
    "build_endpoint_graphs",
    "summarize_graph",
]
