"""Base statistical feature extraction for endpoint interaction graphs."""

from __future__ import annotations

from traffic_graph.features.feature_types import (
    EDGE_BASE_FEATURE_FIELDS,
    NODE_BASE_FEATURE_FIELDS,
    EdgeFeatureSet,
    FeatureRow,
    GraphFeatureView,
    NodeFeatureSet,
)
from traffic_graph.graph.graph_types import CommunicationEdge, EndpointNode, InteractionGraph

ENDPOINT_TYPE_ENCODING: dict[str, int] = {"client": 0, "server": 1}
PROTO_ENCODING: dict[str, int] = {
    "unknown": 0,
    "tcp": 1,
    "udp": 2,
    "icmp": 3,
    "icmpv6": 4,
}
EDGE_TYPE_ENCODING: dict[str, int] = {
    "communication": 0,
    "association_same_src_ip": 1,
    "association_same_dst_subnet": 2,
    "association_same_dst_ip": 3,
    "association_same_prefix_signature": 4,
    "association_prefix_similarity": 5,
}
FLOW_LENGTH_TYPE_ENCODING: dict[str, int] = {"unknown": 0, "short": 1, "long": 2}


def _histogram_bins(histogram: tuple[float, ...], *, width: int = 6) -> list[float]:
    """Return one fixed-width histogram vector with zero padding."""

    values = list(float(value) for value in histogram[:width])
    if len(values) < width:
        values.extend(0.0 for _ in range(width - len(values)))
    return values


def _embedding_bins(values: tuple[float, ...], *, width: int = 16) -> list[float]:
    """Return a fixed-width embedding vector with zero padding."""

    embedding = list(float(value) for value in values[:width])
    if len(embedding) < width:
        embedding.extend(0.0 for _ in range(width - len(embedding)))
    return embedding


def _encode_port(port: int | tuple[int, ...]) -> int:
    """Encode a node port value into one stable scalar feature."""

    if isinstance(port, tuple):
        return min(port) if port else 0
    return port


def _encode_proto(proto: str) -> int:
    """Encode a protocol name with a stable integer mapping."""

    return PROTO_ENCODING.get(proto.lower(), PROTO_ENCODING["unknown"])


def _incident_edges(
    graph_sample: InteractionGraph,
    node: EndpointNode,
) -> list[CommunicationEdge]:
    """Return all edges incident to a node regardless of direction."""

    return [
        edge
        for edge in graph_sample.edges
        if edge.source_node_id == node.node_id or edge.target_node_id == node.node_id
    ]


def _communication_edges(
    graph_sample: InteractionGraph,
    node: EndpointNode,
) -> list[CommunicationEdge]:
    """Return incident communication edges for one node."""

    return [
        edge
        for edge in _incident_edges(graph_sample, node)
        if edge.edge_type == "communication"
    ]


def _association_edges(
    graph_sample: InteractionGraph,
    node: EndpointNode,
) -> list[CommunicationEdge]:
    """Return incident association edges for one node."""

    return [
        edge
        for edge in _incident_edges(graph_sample, node)
        if edge.edge_type != "communication"
    ]


def _node_feature_row(graph_sample: InteractionGraph, node: EndpointNode) -> FeatureRow:
    """Build one node feature dictionary from incident graph statistics."""

    communication_edges = _communication_edges(graph_sample, node)
    association_edges = _association_edges(graph_sample, node)
    communication_edge_count = len(communication_edges)
    total_pkt_count = sum(edge.pkt_count for edge in communication_edges)
    total_byte_count = sum(edge.byte_count for edge in communication_edges)
    total_flow_count = sum(edge.flow_count for edge in communication_edges)
    total_duration = sum(edge.duration for edge in communication_edges)

    avg_pkt_count = (
        total_pkt_count / communication_edge_count if communication_edge_count else 0.0
    )
    avg_byte_count = (
        total_byte_count / communication_edge_count if communication_edge_count else 0.0
    )
    avg_duration = (
        total_duration / communication_edge_count if communication_edge_count else 0.0
    )

    return {
        "endpoint_type": ENDPOINT_TYPE_ENCODING[node.endpoint_type],
        "port": _encode_port(node.port),
        "proto": _encode_proto(node.proto),
        "degree_like_placeholder": 0.0,
        "total_pkt_count": total_pkt_count,
        "total_byte_count": total_byte_count,
        "total_flow_count": total_flow_count,
        "avg_pkt_count": avg_pkt_count,
        "avg_byte_count": avg_byte_count,
        "avg_duration": avg_duration,
        "communication_edge_count": communication_edge_count,
        "association_edge_count": len(association_edges),
    }


def _edge_feature_row(edge: CommunicationEdge) -> FeatureRow:
    """Build one edge feature dictionary with stable defaults for associations."""

    iat_hist = _histogram_bins(edge.iat_hist)
    pkt_len_hist = _histogram_bins(edge.pkt_len_hist)
    flow_internal_embedding = _embedding_bins(edge.flow_internal_embedding)
    return {
        "edge_type": EDGE_TYPE_ENCODING[edge.edge_type],
        "pkt_count": edge.pkt_count,
        "byte_count": edge.byte_count,
        "duration": edge.duration,
        "flow_count": edge.flow_count,
        "is_aggregated": int(edge.is_aggregated),
        "retry_like_count": edge.retry_like_count,
        "retry_like_ratio": edge.retry_like_ratio,
        "flag_syn_ratio": edge.flag_syn_ratio,
        "flag_ack_ratio": edge.flag_ack_ratio,
        "flag_rst_ratio": edge.flag_rst_ratio,
        "flag_pattern_code": edge.flag_pattern_code,
        "first_packet_size_pattern": edge.first_packet_size_pattern,
        "coarse_ack_delay_mean": edge.coarse_ack_delay_mean,
        "coarse_ack_delay_p75": edge.coarse_ack_delay_p75,
        "ack_delay_large_gap_ratio": edge.ack_delay_large_gap_ratio,
        "seq_ack_match_ratio": edge.seq_ack_match_ratio,
        "unmatched_seq_ratio": edge.unmatched_seq_ratio,
        "unmatched_ack_ratio": edge.unmatched_ack_ratio,
        "retry_burst_count": edge.retry_burst_count,
        "retry_burst_max_len": edge.retry_burst_max_len,
        "retry_like_dense_ratio": edge.retry_like_dense_ratio,
        "first_packet_dir_size_pattern": edge.first_packet_dir_size_pattern,
        "first_4_packet_pattern_code": edge.first_4_packet_pattern_code,
        "small_pkt_burst_count": edge.small_pkt_burst_count,
        "small_pkt_burst_ratio": edge.small_pkt_burst_ratio,
        "rst_after_small_burst_indicator": edge.rst_after_small_burst_indicator,
        "flow_length_type_code": FLOW_LENGTH_TYPE_ENCODING.get(
            edge.flow_length_type.lower(),
            FLOW_LENGTH_TYPE_ENCODING["unknown"],
        ),
        "flow_internal_packet_count": edge.flow_internal_packet_count,
        "flow_internal_sequential_edge_count": edge.flow_internal_sequential_edge_count,
        "flow_internal_window_edge_count": edge.flow_internal_window_edge_count,
        "flow_internal_ack_edge_count": edge.flow_internal_ack_edge_count,
        "flow_internal_opposite_direction_edge_count": (
            edge.flow_internal_opposite_direction_edge_count
        ),
        "prefix_behavior_signature": edge.prefix_behavior_signature,
        "iat_hist_bin_0": iat_hist[0],
        "iat_hist_bin_1": iat_hist[1],
        "iat_hist_bin_2": iat_hist[2],
        "iat_hist_bin_3": iat_hist[3],
        "iat_hist_bin_4": iat_hist[4],
        "iat_hist_bin_5": iat_hist[5],
        "pkt_len_hist_bin_0": pkt_len_hist[0],
        "pkt_len_hist_bin_1": pkt_len_hist[1],
        "pkt_len_hist_bin_2": pkt_len_hist[2],
        "pkt_len_hist_bin_3": pkt_len_hist[3],
        "pkt_len_hist_bin_4": pkt_len_hist[4],
        "pkt_len_hist_bin_5": pkt_len_hist[5],
        "flow_internal_emb_0": flow_internal_embedding[0],
        "flow_internal_emb_1": flow_internal_embedding[1],
        "flow_internal_emb_2": flow_internal_embedding[2],
        "flow_internal_emb_3": flow_internal_embedding[3],
        "flow_internal_emb_4": flow_internal_embedding[4],
        "flow_internal_emb_5": flow_internal_embedding[5],
        "flow_internal_emb_6": flow_internal_embedding[6],
        "flow_internal_emb_7": flow_internal_embedding[7],
        "flow_internal_emb_8": flow_internal_embedding[8],
        "flow_internal_emb_9": flow_internal_embedding[9],
        "flow_internal_emb_10": flow_internal_embedding[10],
        "flow_internal_emb_11": flow_internal_embedding[11],
        "flow_internal_emb_12": flow_internal_embedding[12],
        "flow_internal_emb_13": flow_internal_embedding[13],
        "flow_internal_emb_14": flow_internal_embedding[14],
        "flow_internal_emb_15": flow_internal_embedding[15],
    }


def _row_to_vector(row: FeatureRow, field_names: tuple[str, ...]) -> tuple[float, ...]:
    """Convert an ordered feature dictionary into a numeric feature vector."""

    return tuple(float(row[field_name]) for field_name in field_names)


def extract_node_base_features(graph_sample: InteractionGraph) -> NodeFeatureSet:
    """Extract fixed-order base features for every node in an interaction graph."""

    ordered_node_ids = tuple(node.node_id for node in graph_sample.nodes)
    feature_rows = tuple(_node_feature_row(graph_sample, node) for node in graph_sample.nodes)
    feature_by_node_id = {
        node_id: feature_row
        for node_id, feature_row in zip(ordered_node_ids, feature_rows, strict=True)
    }
    feature_matrix = tuple(
        _row_to_vector(feature_row, NODE_BASE_FEATURE_FIELDS)
        for feature_row in feature_rows
    )
    return NodeFeatureSet(
        field_names=NODE_BASE_FEATURE_FIELDS,
        ordered_node_ids=ordered_node_ids,
        feature_rows=feature_rows,
        feature_by_node_id=feature_by_node_id,
        feature_matrix=feature_matrix,
    )


def extract_edge_base_features(graph_sample: InteractionGraph) -> EdgeFeatureSet:
    """Extract fixed-order base features for every edge in an interaction graph."""

    ordered_edge_ids = tuple(edge.edge_id for edge in graph_sample.edges)
    feature_rows = tuple(_edge_feature_row(edge) for edge in graph_sample.edges)
    feature_by_edge_id = {
        edge_id: feature_row
        for edge_id, feature_row in zip(ordered_edge_ids, feature_rows, strict=True)
    }
    feature_matrix = tuple(
        _row_to_vector(feature_row, EDGE_BASE_FEATURE_FIELDS)
        for feature_row in feature_rows
    )
    return EdgeFeatureSet(
        field_names=EDGE_BASE_FEATURE_FIELDS,
        ordered_edge_ids=ordered_edge_ids,
        feature_rows=feature_rows,
        feature_by_edge_id=feature_by_edge_id,
        feature_matrix=feature_matrix,
    )


def build_base_feature_views(graph_sample: InteractionGraph) -> GraphFeatureView:
    """Build node and edge base features for one interaction graph."""

    return GraphFeatureView(
        node_features=extract_node_base_features(graph_sample),
        edge_features=extract_edge_base_features(graph_sample),
    )


def summarize_feature_view(graph_feature_view: GraphFeatureView) -> dict[str, int]:
    """Return a compact feature summary for logs or CLI previews."""

    return {
        "node_count": len(graph_feature_view.node_features.ordered_node_ids),
        "node_feature_dim": graph_feature_view.node_features.feature_dim,
        "edge_count": len(graph_feature_view.edge_features.ordered_edge_ids),
        "edge_feature_dim": graph_feature_view.edge_features.feature_dim,
    }


__all__ = [
    "EDGE_TYPE_ENCODING",
    "FLOW_LENGTH_TYPE_ENCODING",
    "ENDPOINT_TYPE_ENCODING",
    "PROTO_ENCODING",
    "build_base_feature_views",
    "extract_edge_base_features",
    "extract_node_base_features",
    "summarize_feature_view",
]
