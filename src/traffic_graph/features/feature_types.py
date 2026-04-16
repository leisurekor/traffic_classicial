"""Typed containers for graph feature extraction outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

FeatureValue: TypeAlias = int | float
FeatureRow: TypeAlias = dict[str, FeatureValue]
FeatureMatrix: TypeAlias = tuple[tuple[float, ...], ...]

EDGE_PATTERN_CODE_FIELDS: tuple[str, ...] = (
    "flag_pattern_code",
    "first_packet_size_pattern",
    "first_packet_dir_size_pattern",
    "first_4_packet_pattern_code",
    "prefix_behavior_signature",
)
"""Edge fields that encode categorical traffic-pattern IDs, not ordered magnitudes."""

EDGE_DISCRETE_FEATURE_FIELDS: tuple[str, ...] = (
    "edge_type",
    "flow_length_type_code",
    "is_aggregated",
    "rst_after_small_burst_indicator",
) + EDGE_PATTERN_CODE_FIELDS
"""Edge fields that should stay discrete throughout normalization and scoring."""

NODE_BASE_FEATURE_FIELDS: tuple[str, ...] = (
    "endpoint_type",
    "port",
    "proto",
    "degree_like_placeholder",
    "total_pkt_count",
    "total_byte_count",
    "total_flow_count",
    "avg_pkt_count",
    "avg_byte_count",
    "avg_duration",
    "communication_edge_count",
    "association_edge_count",
)

NODE_STRUCTURE_FEATURE_FIELDS: tuple[str, ...] = (
    "total_degree",
    "communication_in_degree",
    "communication_out_degree",
    "unique_neighbor_count",
)

NODE_PACKED_FEATURE_FIELDS: tuple[str, ...] = (
    NODE_BASE_FEATURE_FIELDS + NODE_STRUCTURE_FEATURE_FIELDS
)

EDGE_BASE_FEATURE_FIELDS: tuple[str, ...] = (
    "edge_type",
    "pkt_count",
    "byte_count",
    "duration",
    "flow_count",
    "is_aggregated",
    "retry_like_count",
    "retry_like_ratio",
    "flag_syn_ratio",
    "flag_ack_ratio",
    "flag_rst_ratio",
    "flag_pattern_code",
    "first_packet_size_pattern",
    "coarse_ack_delay_mean",
    "coarse_ack_delay_p75",
    "ack_delay_large_gap_ratio",
    "seq_ack_match_ratio",
    "unmatched_seq_ratio",
    "unmatched_ack_ratio",
    "retry_burst_count",
    "retry_burst_max_len",
    "retry_like_dense_ratio",
    "first_packet_dir_size_pattern",
    "first_4_packet_pattern_code",
    "small_pkt_burst_count",
    "small_pkt_burst_ratio",
    "rst_after_small_burst_indicator",
    "flow_length_type_code",
    "flow_internal_packet_count",
    "flow_internal_sequential_edge_count",
    "flow_internal_window_edge_count",
    "flow_internal_ack_edge_count",
    "flow_internal_opposite_direction_edge_count",
    "prefix_behavior_signature",
    "iat_hist_bin_0",
    "iat_hist_bin_1",
    "iat_hist_bin_2",
    "iat_hist_bin_3",
    "iat_hist_bin_4",
    "iat_hist_bin_5",
    "pkt_len_hist_bin_0",
    "pkt_len_hist_bin_1",
    "pkt_len_hist_bin_2",
    "pkt_len_hist_bin_3",
    "pkt_len_hist_bin_4",
    "pkt_len_hist_bin_5",
    "flow_internal_emb_0",
    "flow_internal_emb_1",
    "flow_internal_emb_2",
    "flow_internal_emb_3",
    "flow_internal_emb_4",
    "flow_internal_emb_5",
    "flow_internal_emb_6",
    "flow_internal_emb_7",
    "flow_internal_emb_8",
    "flow_internal_emb_9",
    "flow_internal_emb_10",
    "flow_internal_emb_11",
    "flow_internal_emb_12",
    "flow_internal_emb_13",
    "flow_internal_emb_14",
    "flow_internal_emb_15",
)

EDGE_PACKED_FEATURE_FIELDS: tuple[str, ...] = EDGE_BASE_FEATURE_FIELDS


@dataclass(frozen=True, slots=True)
class NodeFeatureSet:
    """Node feature dictionaries and their fixed-order matrix view."""

    field_names: tuple[str, ...]
    ordered_node_ids: tuple[str, ...]
    feature_rows: tuple[FeatureRow, ...]
    feature_by_node_id: dict[str, FeatureRow]
    feature_matrix: FeatureMatrix

    @property
    def feature_dim(self) -> int:
        """Return the number of feature columns in each node row."""

        return len(self.field_names)


@dataclass(frozen=True, slots=True)
class EdgeFeatureSet:
    """Edge feature dictionaries and their fixed-order matrix view."""

    field_names: tuple[str, ...]
    ordered_edge_ids: tuple[str, ...]
    feature_rows: tuple[FeatureRow, ...]
    feature_by_edge_id: dict[str, FeatureRow]
    feature_matrix: FeatureMatrix

    @property
    def feature_dim(self) -> int:
        """Return the number of feature columns in each edge row."""

        return len(self.field_names)


@dataclass(frozen=True, slots=True)
class GraphFeatureView:
    """Combined node and edge feature views for one interaction graph."""

    node_features: NodeFeatureSet
    edge_features: EdgeFeatureSet


__all__ = [
    "EDGE_BASE_FEATURE_FIELDS",
    "EDGE_DISCRETE_FEATURE_FIELDS",
    "EDGE_PACKED_FEATURE_FIELDS",
    "EDGE_PATTERN_CODE_FIELDS",
    "NODE_BASE_FEATURE_FIELDS",
    "NODE_PACKED_FEATURE_FIELDS",
    "NODE_STRUCTURE_FEATURE_FIELDS",
    "EdgeFeatureSet",
    "FeatureMatrix",
    "FeatureRow",
    "FeatureValue",
    "GraphFeatureView",
    "NodeFeatureSet",
]
