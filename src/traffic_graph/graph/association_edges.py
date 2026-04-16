"""Lightweight association-edge builders for endpoint interaction graphs."""

from __future__ import annotations

import ipaddress
import logging
import math
from collections import defaultdict
from collections.abc import Iterable
from itertools import combinations

from traffic_graph.config import AssociationEdgeConfig
from traffic_graph.graph.graph_types import (
    CommunicationEdge,
    EdgeType,
    EndpointNode,
    InteractionGraph,
    build_interaction_graph_stats,
)
from traffic_graph.graph.nx_compat import MultiDiGraph

logger = logging.getLogger(__name__)


def _association_pair_key(
    edge_type: EdgeType,
    source_node_id: str,
    target_node_id: str,
) -> tuple[str, str, str]:
    """Return a canonical key for an unordered association edge."""

    ordered_source, ordered_target = sorted((source_node_id, target_node_id))
    return (edge_type, ordered_source, ordered_target)


def _existing_association_keys(
    edges: Iterable[CommunicationEdge],
) -> set[tuple[str, str, str]]:
    """Collect already-present association edges to avoid duplication."""

    keys: set[tuple[str, str, str]] = set()
    for edge in edges:
        if edge.edge_type == "communication":
            continue
        keys.add(
            _association_pair_key(
                edge.edge_type,
                edge.source_node_id,
                edge.target_node_id,
            )
        )
    return keys


def _materialize_graph(
    nodes: tuple[EndpointNode, ...],
    edges: tuple[CommunicationEdge, ...],
) -> MultiDiGraph:
    """Build a backend graph object from stored nodes and typed edges."""

    graph = MultiDiGraph()
    for node in nodes:
        graph.add_node(
            node.node_id,
            endpoint_type=node.endpoint_type,
            ip=node.ip,
            port=node.port,
            proto=node.proto,
        )
    for edge in edges:
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
    return graph


def _make_association_edge(
    edge_type: EdgeType,
    source_node: EndpointNode,
    target_node: EndpointNode,
) -> CommunicationEdge:
    """Create a deterministic association edge between two endpoint nodes."""

    ordered_source, ordered_target = sorted((source_node.node_id, target_node.node_id))
    return CommunicationEdge(
        edge_id=f"{edge_type}:{ordered_source}->{ordered_target}",
        source_node_id=ordered_source,
        target_node_id=ordered_target,
        edge_type=edge_type,
        logical_flow_id=None,
        pkt_count=0,
        byte_count=0,
        duration=0.0,
        flow_count=0,
        is_aggregated=False,
        source_flow_ids=(),
        retry_like_count=0,
        retry_like_ratio=0.0,
        iat_hist=(),
        pkt_len_hist=(),
        flag_syn_ratio=0.0,
        flag_ack_ratio=0.0,
        flag_rst_ratio=0.0,
        flag_pattern_code=0,
        first_packet_size_pattern=0,
        coarse_ack_delay_mean=0.0,
        coarse_ack_delay_p75=0.0,
        ack_delay_large_gap_ratio=0.0,
        seq_ack_match_ratio=0.0,
        unmatched_seq_ratio=0.0,
        unmatched_ack_ratio=0.0,
        retry_burst_count=0,
        retry_burst_max_len=0,
        retry_like_dense_ratio=0.0,
        first_packet_dir_size_pattern=0,
        first_4_packet_pattern_code=0,
        small_pkt_burst_count=0,
        small_pkt_burst_ratio=0.0,
        rst_after_small_burst_indicator=0,
        flow_internal_embedding=(),
        flow_internal_packet_count=0,
        flow_internal_sequential_edge_count=0,
        flow_internal_window_edge_count=0,
        flow_internal_ack_edge_count=0,
        flow_internal_opposite_direction_edge_count=0,
        prefix_behavior_signature=0,
        flow_length_type="long",
    )


def _communication_edges_for_node(
    graph_sample: InteractionGraph,
    node: EndpointNode,
) -> list[CommunicationEdge]:
    """Return communication edges incident to one node."""

    return [
        edge
        for edge in graph_sample.edges
        if edge.edge_type == "communication"
        and (edge.source_node_id == node.node_id or edge.target_node_id == node.node_id)
    ]


def build_association_edges_same_src_ip(
    graph_sample: InteractionGraph,
    existing_keys: set[tuple[str, str, str]] | None = None,
) -> list[CommunicationEdge]:
    """Build association edges between client nodes that share the same source IP."""

    known_keys = set(existing_keys or set())
    client_groups: dict[str, list[EndpointNode]] = defaultdict(list)
    for node in graph_sample.nodes:
        if node.endpoint_type == "client":
            client_groups[node.ip].append(node)

    association_edges: list[CommunicationEdge] = []
    for grouped_nodes in client_groups.values():
        ordered_nodes = sorted(grouped_nodes, key=lambda node: node.node_id)
        for source_node, target_node in combinations(ordered_nodes, 2):
            edge_key = _association_pair_key(
                "association_same_src_ip",
                source_node.node_id,
                target_node.node_id,
            )
            if edge_key in known_keys:
                continue
            known_keys.add(edge_key)
            association_edges.append(
                _make_association_edge(
                    "association_same_src_ip",
                    source_node,
                    target_node,
                )
            )
    return association_edges


def _server_subnet(ip: str, prefix: int) -> str | None:
    """Return the IPv4 subnet label for a server IP or `None` when unsupported."""

    try:
        address = ipaddress.ip_address(ip)
    except ValueError:
        logger.debug("Skipping same_dst_subnet association for invalid IPv4: %s", ip)
        return None

    if not isinstance(address, ipaddress.IPv4Address):
        logger.debug("Skipping same_dst_subnet association for non-IPv4 address: %s", ip)
        return None

    network = ipaddress.IPv4Network(f"{ip}/{prefix}", strict=False)
    return network.with_prefixlen


def build_association_edges_same_dst_subnet(
    graph_sample: InteractionGraph,
    dst_subnet_prefix: int = 24,
    existing_keys: set[tuple[str, str, str]] | None = None,
) -> list[CommunicationEdge]:
    """Build association edges between server nodes inside the same IPv4 subnet."""

    known_keys = set(existing_keys or set())
    server_groups: dict[str, list[EndpointNode]] = defaultdict(list)
    for node in graph_sample.nodes:
        if node.endpoint_type != "server":
            continue
        subnet = _server_subnet(node.ip, dst_subnet_prefix)
        if subnet is None:
            continue
        server_groups[subnet].append(node)

    association_edges: list[CommunicationEdge] = []
    for grouped_nodes in server_groups.values():
        ordered_nodes = sorted(grouped_nodes, key=lambda node: node.node_id)
        for source_node, target_node in combinations(ordered_nodes, 2):
            edge_key = _association_pair_key(
                "association_same_dst_subnet",
                source_node.node_id,
                target_node.node_id,
            )
            if edge_key in known_keys:
                continue
            known_keys.add(edge_key)
            association_edges.append(
                _make_association_edge(
                    "association_same_dst_subnet",
                    source_node,
                    target_node,
                )
            )
    return association_edges


def build_association_edges_same_dst_ip(
    graph_sample: InteractionGraph,
    existing_keys: set[tuple[str, str, str]] | None = None,
) -> list[CommunicationEdge]:
    """Build association edges between server nodes that share the same destination IP."""

    known_keys = set(existing_keys or set())
    server_groups: dict[str, list[EndpointNode]] = defaultdict(list)
    for node in graph_sample.nodes:
        if node.endpoint_type == "server":
            server_groups[node.ip].append(node)

    association_edges: list[CommunicationEdge] = []
    for grouped_nodes in server_groups.values():
        ordered_nodes = sorted(grouped_nodes, key=lambda node: node.node_id)
        for source_node, target_node in combinations(ordered_nodes, 2):
            edge_key = _association_pair_key(
                "association_same_dst_ip",
                source_node.node_id,
                target_node.node_id,
            )
            if edge_key in known_keys:
                continue
            known_keys.add(edge_key)
            association_edges.append(
                _make_association_edge(
                    "association_same_dst_ip",
                    source_node,
                    target_node,
                )
            )
    return association_edges


def build_association_edges_same_prefix_signature(
    graph_sample: InteractionGraph,
    existing_keys: set[tuple[str, str, str]] | None = None,
) -> list[CommunicationEdge]:
    """Connect client nodes whose outgoing communication edges share a prefix signature."""

    known_keys = set(existing_keys or set())
    signature_groups: dict[int, list[EndpointNode]] = defaultdict(list)
    node_lookup = {node.node_id: node for node in graph_sample.nodes}
    for edge in graph_sample.edges:
        if edge.edge_type != "communication" or edge.prefix_behavior_signature <= 0:
            continue
        source_node = node_lookup.get(edge.source_node_id)
        if source_node is None or source_node.endpoint_type != "client":
            continue
        signature_groups[edge.prefix_behavior_signature].append(source_node)

    association_edges: list[CommunicationEdge] = []
    for grouped_nodes in signature_groups.values():
        ordered_nodes = sorted({node.node_id: node for node in grouped_nodes}.values(), key=lambda n: n.node_id)
        for source_node, target_node in combinations(ordered_nodes, 2):
            edge_key = _association_pair_key(
                "association_same_prefix_signature",
                source_node.node_id,
                target_node.node_id,
            )
            if edge_key in known_keys:
                continue
            known_keys.add(edge_key)
            association_edges.append(
                _make_association_edge(
                    "association_same_prefix_signature",
                    source_node,
                    target_node,
                )
            )
    return association_edges


def _mean_embedding(edges: list[CommunicationEdge]) -> tuple[float, ...]:
    embeddings = [edge.flow_internal_embedding for edge in edges if edge.flow_internal_embedding]
    if not embeddings:
        return ()
    width = len(embeddings[0])
    totals = [0.0] * width
    for embedding in embeddings:
        for index in range(min(width, len(embedding))):
            totals[index] += float(embedding[index])
    return tuple(value / len(embeddings) for value in totals)


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or not right:
        return 0.0
    width = min(len(left), len(right))
    dot = sum(float(left[i]) * float(right[i]) for i in range(width))
    left_norm = math.sqrt(sum(float(left[i]) ** 2 for i in range(width)))
    right_norm = math.sqrt(sum(float(right[i]) ** 2 for i in range(width)))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return float(dot / (left_norm * right_norm))


def build_association_edges_prefix_similarity(
    graph_sample: InteractionGraph,
    *,
    threshold: float = 0.95,
    top_k: int = 1,
    existing_keys: set[tuple[str, str, str]] | None = None,
) -> list[CommunicationEdge]:
    """Connect client nodes whose outgoing flow embeddings are highly similar."""

    known_keys = set(existing_keys or set())
    client_embeddings: list[tuple[EndpointNode, tuple[float, ...]]] = []
    for node in graph_sample.nodes:
        if node.endpoint_type != "client":
            continue
        communication_edges = [
            edge
            for edge in _communication_edges_for_node(graph_sample, node)
            if edge.source_node_id == node.node_id
        ]
        embedding = _mean_embedding(communication_edges)
        if embedding:
            client_embeddings.append((node, embedding))

    association_edges: list[CommunicationEdge] = []
    for index, (source_node, source_embedding) in enumerate(client_embeddings):
        ranked: list[tuple[float, EndpointNode]] = []
        for other_index, (target_node, target_embedding) in enumerate(client_embeddings):
            if index == other_index:
                continue
            similarity = _cosine_similarity(source_embedding, target_embedding)
            if similarity >= threshold:
                ranked.append((similarity, target_node))
        ranked.sort(key=lambda item: (-item[0], item[1].node_id))
        for _similarity, target_node in ranked[: max(1, top_k)]:
            edge_key = _association_pair_key(
                "association_prefix_similarity",
                source_node.node_id,
                target_node.node_id,
            )
            if edge_key in known_keys:
                continue
            known_keys.add(edge_key)
            association_edges.append(
                _make_association_edge(
                    "association_prefix_similarity",
                    source_node,
                    target_node,
                )
            )
    return association_edges


def add_association_edges(
    graph_sample: InteractionGraph,
    config: AssociationEdgeConfig,
) -> InteractionGraph:
    """Return a graph sample extended with configured in-window association edges."""

    if not (
        config.enable_same_src_ip
        or config.enable_same_dst_subnet
        or config.enable_same_dst_ip
        or config.enable_same_prefix_signature
        or config.enable_prefix_similarity
    ):
        return graph_sample

    existing_keys = _existing_association_keys(graph_sample.edges)
    added_edges: list[CommunicationEdge] = []

    if config.enable_same_src_ip:
        new_same_src_ip_edges = build_association_edges_same_src_ip(
            graph_sample,
            existing_keys=existing_keys,
        )
        added_edges.extend(new_same_src_ip_edges)
        existing_keys.update(
            _existing_association_keys(new_same_src_ip_edges)
        )

    if config.enable_same_dst_subnet:
        new_same_dst_subnet_edges = build_association_edges_same_dst_subnet(
            graph_sample,
            dst_subnet_prefix=config.dst_subnet_prefix,
            existing_keys=existing_keys,
        )
        added_edges.extend(new_same_dst_subnet_edges)
        existing_keys.update(_existing_association_keys(new_same_dst_subnet_edges))

    if config.enable_same_dst_ip:
        new_same_dst_ip_edges = build_association_edges_same_dst_ip(
            graph_sample,
            existing_keys=existing_keys,
        )
        added_edges.extend(new_same_dst_ip_edges)
        existing_keys.update(_existing_association_keys(new_same_dst_ip_edges))

    if config.enable_same_prefix_signature:
        new_same_prefix_signature_edges = build_association_edges_same_prefix_signature(
            graph_sample,
            existing_keys=existing_keys,
        )
        added_edges.extend(new_same_prefix_signature_edges)
        existing_keys.update(_existing_association_keys(new_same_prefix_signature_edges))

    if config.enable_prefix_similarity:
        added_edges.extend(
            build_association_edges_prefix_similarity(
                graph_sample,
                threshold=config.prefix_similarity_threshold,
                top_k=config.prefix_similarity_top_k,
                existing_keys=existing_keys,
            )
        )

    if not added_edges:
        return graph_sample

    all_edges = tuple(
        list(graph_sample.edges)
        + sorted(added_edges, key=lambda edge: (edge.edge_type, edge.edge_id))
    )
    graph = _materialize_graph(graph_sample.nodes, all_edges)
    return InteractionGraph(
        window_index=graph_sample.window_index,
        window_start=graph_sample.window_start,
        window_end=graph_sample.window_end,
        graph=graph,
        nodes=graph_sample.nodes,
        edges=all_edges,
        stats=build_interaction_graph_stats(graph_sample.nodes, all_edges),
    )


__all__ = [
    "add_association_edges",
    "build_association_edges_same_dst_subnet",
    "build_association_edges_same_src_ip",
]
