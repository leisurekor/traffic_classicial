"""Endpoint interaction graph builders for logical flow batches."""

from __future__ import annotations

from collections.abc import Iterable

from traffic_graph.config import GraphConfig
from traffic_graph.data.preprocessing import LogicalFlowBatch, LogicalFlowRecord
from traffic_graph.graph.association_edges import add_association_edges
from traffic_graph.graph.graph_types import (
    CommunicationEdge,
    EndpointNode,
    EndpointPort,
    InteractionGraph,
    build_interaction_graph_stats,
)
from traffic_graph.graph.nx_compat import MultiDiGraph


def _render_port_label(port: EndpointPort) -> str:
    """Render a stable port label for node identifiers."""

    if isinstance(port, tuple):
        return ",".join(str(value) for value in port)
    return str(port)


def _make_node_id(endpoint_type: str, ip: str, port: EndpointPort, proto: str) -> str:
    """Build a stable node identifier for the endpoint graph."""

    return f"{endpoint_type}:{ip}:{_render_port_label(port)}:{proto}"


def _client_port(logical_flow: LogicalFlowRecord) -> EndpointPort:
    """Return the client port representation for a logical flow."""

    if len(logical_flow.src_ports) == 1:
        return logical_flow.src_ports[0]
    return logical_flow.src_ports


def _build_client_node(logical_flow: LogicalFlowRecord) -> EndpointNode:
    """Create a client endpoint node from a logical flow."""

    port = _client_port(logical_flow)
    return EndpointNode(
        node_id=_make_node_id("client", logical_flow.src_ip, port, logical_flow.protocol),
        endpoint_type="client",
        ip=logical_flow.src_ip,
        port=port,
        proto=logical_flow.protocol,
    )


def _build_server_node(logical_flow: LogicalFlowRecord) -> EndpointNode:
    """Create a server endpoint node from a logical flow."""

    return EndpointNode(
        node_id=_make_node_id(
            "server",
            logical_flow.dst_ip,
            logical_flow.dst_port,
            logical_flow.protocol,
        ),
        endpoint_type="server",
        ip=logical_flow.dst_ip,
        port=logical_flow.dst_port,
        proto=logical_flow.protocol,
    )


def _build_edge(
    logical_flow: LogicalFlowRecord,
    client_node: EndpointNode,
    server_node: EndpointNode,
) -> CommunicationEdge:
    """Create a communication edge from a logical flow record."""

    return CommunicationEdge(
        edge_id=logical_flow.logical_flow_id,
        source_node_id=client_node.node_id,
        target_node_id=server_node.node_id,
        edge_type="communication",
        logical_flow_id=logical_flow.logical_flow_id,
        pkt_count=logical_flow.total_pkt_count,
        byte_count=logical_flow.total_byte_count,
        duration=logical_flow.avg_duration,
        flow_count=logical_flow.flow_count,
        is_aggregated=logical_flow.is_aggregated_short_flow,
        source_flow_ids=logical_flow.source_flow_ids,
        retry_like_count=logical_flow.retry_like_count,
        retry_like_ratio=logical_flow.retry_like_ratio,
        iat_hist=logical_flow.iat_hist,
        pkt_len_hist=logical_flow.pkt_len_hist,
        flag_syn_ratio=logical_flow.flag_syn_ratio,
        flag_ack_ratio=logical_flow.flag_ack_ratio,
        flag_rst_ratio=logical_flow.flag_rst_ratio,
        flag_pattern_code=logical_flow.flag_pattern_code,
        first_packet_size_pattern=logical_flow.first_packet_size_pattern,
        coarse_ack_delay_mean=logical_flow.coarse_ack_delay_mean,
        coarse_ack_delay_p75=logical_flow.coarse_ack_delay_p75,
        ack_delay_large_gap_ratio=logical_flow.ack_delay_large_gap_ratio,
        seq_ack_match_ratio=logical_flow.seq_ack_match_ratio,
        unmatched_seq_ratio=logical_flow.unmatched_seq_ratio,
        unmatched_ack_ratio=logical_flow.unmatched_ack_ratio,
        retry_burst_count=logical_flow.retry_burst_count,
        retry_burst_max_len=logical_flow.retry_burst_max_len,
        retry_like_dense_ratio=logical_flow.retry_like_dense_ratio,
        first_packet_dir_size_pattern=logical_flow.first_packet_dir_size_pattern,
        first_4_packet_pattern_code=logical_flow.first_4_packet_pattern_code,
        small_pkt_burst_count=logical_flow.small_pkt_burst_count,
        small_pkt_burst_ratio=logical_flow.small_pkt_burst_ratio,
        rst_after_small_burst_indicator=logical_flow.rst_after_small_burst_indicator,
        flow_internal_embedding=logical_flow.flow_internal_embedding,
        flow_internal_packet_count=logical_flow.flow_internal_packet_count,
        flow_internal_sequential_edge_count=logical_flow.flow_internal_sequential_edge_count,
        flow_internal_window_edge_count=logical_flow.flow_internal_window_edge_count,
        flow_internal_ack_edge_count=logical_flow.flow_internal_ack_edge_count,
        flow_internal_opposite_direction_edge_count=(
            logical_flow.flow_internal_opposite_direction_edge_count
        ),
        prefix_behavior_signature=logical_flow.prefix_behavior_signature,
        flow_length_type=logical_flow.flow_length_type,
    )


def _node_sort_key(node: EndpointNode) -> tuple[str, str, str]:
    """Return a deterministic ordering key for endpoint nodes."""

    return (node.endpoint_type, node.ip, _render_port_label(node.port))


def build_endpoint_graph(
    window_batch: LogicalFlowBatch,
    graph_config: GraphConfig | None = None,
) -> InteractionGraph:
    """Build one endpoint interaction graph from a logical flow batch."""

    config = graph_config or GraphConfig()
    graph = MultiDiGraph()
    node_index: dict[str, EndpointNode] = {}
    edge_records: list[CommunicationEdge] = []

    for logical_flow in window_batch.logical_flows:
        client_node = _build_client_node(logical_flow)
        server_node = _build_server_node(logical_flow)

        for node in (client_node, server_node):
            if node.node_id not in node_index:
                node_index[node.node_id] = node
                graph.add_node(
                    node.node_id,
                    endpoint_type=node.endpoint_type,
                    ip=node.ip,
                    port=node.port,
                    proto=node.proto,
                )

        edge = _build_edge(logical_flow, client_node, server_node)
        graph.add_edge(
            edge.source_node_id,
            edge.target_node_id,
            key=edge.edge_id,
            edge_type=edge.edge_type,
            logical_flow_id=edge.logical_flow_id,
            pkt_count=edge.pkt_count,
            byte_count=edge.byte_count,
            duration=edge.duration,
            flow_count=edge.flow_count,
            is_aggregated=edge.is_aggregated,
            source_flow_ids=edge.source_flow_ids,
            retry_like_count=edge.retry_like_count,
            retry_like_ratio=edge.retry_like_ratio,
            iat_hist=edge.iat_hist,
            pkt_len_hist=edge.pkt_len_hist,
            flag_syn_ratio=edge.flag_syn_ratio,
            flag_ack_ratio=edge.flag_ack_ratio,
            flag_rst_ratio=edge.flag_rst_ratio,
            flag_pattern_code=edge.flag_pattern_code,
            first_packet_size_pattern=edge.first_packet_size_pattern,
            coarse_ack_delay_mean=edge.coarse_ack_delay_mean,
            coarse_ack_delay_p75=edge.coarse_ack_delay_p75,
            ack_delay_large_gap_ratio=edge.ack_delay_large_gap_ratio,
            seq_ack_match_ratio=edge.seq_ack_match_ratio,
            unmatched_seq_ratio=edge.unmatched_seq_ratio,
            unmatched_ack_ratio=edge.unmatched_ack_ratio,
            retry_burst_count=edge.retry_burst_count,
            retry_burst_max_len=edge.retry_burst_max_len,
            retry_like_dense_ratio=edge.retry_like_dense_ratio,
            first_packet_dir_size_pattern=edge.first_packet_dir_size_pattern,
            first_4_packet_pattern_code=edge.first_4_packet_pattern_code,
            small_pkt_burst_count=edge.small_pkt_burst_count,
            small_pkt_burst_ratio=edge.small_pkt_burst_ratio,
            rst_after_small_burst_indicator=edge.rst_after_small_burst_indicator,
            flow_internal_embedding=edge.flow_internal_embedding,
            flow_internal_packet_count=edge.flow_internal_packet_count,
            flow_internal_sequential_edge_count=edge.flow_internal_sequential_edge_count,
            flow_internal_window_edge_count=edge.flow_internal_window_edge_count,
            flow_internal_ack_edge_count=edge.flow_internal_ack_edge_count,
            flow_internal_opposite_direction_edge_count=(
                edge.flow_internal_opposite_direction_edge_count
            ),
            prefix_behavior_signature=edge.prefix_behavior_signature,
            flow_length_type=edge.flow_length_type,
        )
        edge_records.append(edge)

    nodes = tuple(sorted(node_index.values(), key=_node_sort_key))
    edges = tuple(sorted(edge_records, key=lambda edge: edge.edge_id))
    graph_sample = InteractionGraph(
        window_index=window_batch.index,
        window_start=window_batch.window_start,
        window_end=window_batch.window_end,
        graph=graph,
        nodes=nodes,
        edges=edges,
        stats=build_interaction_graph_stats(nodes, edges),
    )
    return add_association_edges(graph_sample, config.association_edges)


def build_endpoint_graphs(
    window_batches: Iterable[LogicalFlowBatch],
    graph_config: GraphConfig | None = None,
) -> list[InteractionGraph]:
    """Build endpoint interaction graphs for multiple window batches."""

    return [
        build_endpoint_graph(window_batch, graph_config=graph_config)
        for window_batch in window_batches
    ]


def summarize_graph(graph_sample: InteractionGraph) -> dict[str, int | float]:
    """Return a compact graph summary suitable for logs or CLI previews."""

    return {
        "window_index": graph_sample.window_index,
        "node_count": graph_sample.stats.node_count,
        "edge_count": graph_sample.stats.edge_count,
        "client_node_count": graph_sample.stats.client_node_count,
        "server_node_count": graph_sample.stats.server_node_count,
        "aggregated_edge_count": graph_sample.stats.aggregated_edge_count,
        "communication_edge_count": graph_sample.stats.communication_edge_count,
        "association_edge_count": graph_sample.stats.association_edge_count,
        "association_same_src_ip_edge_count": (
            graph_sample.stats.association_same_src_ip_edge_count
        ),
        "association_same_dst_subnet_edge_count": (
            graph_sample.stats.association_same_dst_subnet_edge_count
        ),
        "association_same_dst_ip_edge_count": (
            graph_sample.stats.association_same_dst_ip_edge_count
        ),
        "association_same_prefix_signature_edge_count": (
            graph_sample.stats.association_same_prefix_signature_edge_count
        ),
        "association_prefix_similarity_edge_count": (
            graph_sample.stats.association_prefix_similarity_edge_count
        ),
    }


class EndpointGraphBuilder:
    """State-free builder wrapper for endpoint interaction graphs."""

    def __init__(self, graph_config: GraphConfig | None = None) -> None:
        """Store optional graph config for communication and association edges."""

        self.graph_config = graph_config or GraphConfig()

    def build(self, window_batch: LogicalFlowBatch) -> InteractionGraph:
        """Build a graph for one logical-flow window batch."""

        return build_endpoint_graph(window_batch, graph_config=self.graph_config)

    def build_many(
        self,
        window_batches: Iterable[LogicalFlowBatch],
    ) -> list[InteractionGraph]:
        """Build graphs for multiple logical-flow window batches."""

        return build_endpoint_graphs(window_batches, graph_config=self.graph_config)


__all__ = [
    "EndpointGraphBuilder",
    "build_endpoint_graph",
    "build_endpoint_graphs",
    "summarize_graph",
]
