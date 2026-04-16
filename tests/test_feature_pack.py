"""Unit tests for graph feature packing and normalization."""

from __future__ import annotations

import numpy as np
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_ROOT: Path = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_graph.config import AssociationEdgeConfig, FeatureNormalizationConfig, GraphConfig
from traffic_graph.data import LogicalFlowBatch, LogicalFlowRecord, LogicalFlowWindowStats
from traffic_graph.features import (
    EDGE_PACKED_FEATURE_FIELDS,
    NODE_PACKED_FEATURE_FIELDS,
    build_model_feature_view,
    fit_feature_preprocessor,
    transform_graph,
    transform_graphs,
)
from traffic_graph.graph import build_endpoint_graph


def _graph_config() -> GraphConfig:
    """Return a graph config with both association edge types enabled."""

    return GraphConfig(
        time_window_seconds=60,
        directed=True,
        association_edges=AssociationEdgeConfig(
            enable_same_src_ip=True,
            enable_same_dst_subnet=True,
            dst_subnet_prefix=24,
        ),
    )


def _build_batch(
    *,
    pkt_scale: int = 1,
    byte_scale: int = 1,
    start_offset_seconds: int = 0,
) -> LogicalFlowBatch:
    """Create a reusable logical-flow batch with configurable flow magnitudes."""

    window_start = datetime(2026, 4, 8, 12, 0, 0) + timedelta(
        seconds=start_offset_seconds
    )
    window_end = window_start + timedelta(seconds=60)
    return LogicalFlowBatch(
        index=start_offset_seconds // 60,
        window_start=window_start,
        window_end=window_end,
        logical_flows=(
            LogicalFlowRecord(
                logical_flow_id=f"logical-a-{start_offset_seconds}",
                src_ip="10.0.0.1",
                dst_ip="10.0.0.2",
                dst_port=80,
                protocol="tcp",
                start_time=window_start + timedelta(seconds=5),
                end_time=window_start + timedelta(seconds=10),
                flow_count=1,
                total_pkt_count=11 * pkt_scale,
                total_byte_count=4096 * byte_scale,
                avg_duration=5.0,
                avg_pkt_count=float(11 * pkt_scale),
                avg_byte_count=float(4096 * byte_scale),
                source_flow_ids=("flow-a",),
                src_ports=(11111,),
                directions=("outbound",),
                tcp_flags=("ACK",),
                is_aggregated_short_flow=False,
            ),
            LogicalFlowRecord(
                logical_flow_id=f"logical-b-{start_offset_seconds}",
                src_ip="10.0.0.1",
                dst_ip="10.0.0.3",
                dst_port=443,
                protocol="tcp",
                start_time=window_start + timedelta(seconds=15),
                end_time=window_start + timedelta(seconds=21),
                flow_count=1,
                total_pkt_count=9 * pkt_scale,
                total_byte_count=3500 * byte_scale,
                avg_duration=6.0,
                avg_pkt_count=float(9 * pkt_scale),
                avg_byte_count=float(3500 * byte_scale),
                source_flow_ids=("flow-b",),
                src_ports=(22222,),
                directions=("outbound",),
                tcp_flags=("ACK",),
                is_aggregated_short_flow=False,
            ),
            LogicalFlowRecord(
                logical_flow_id=f"logical-c-{start_offset_seconds}",
                src_ip="172.16.0.8",
                dst_ip="192.168.1.10",
                dst_port=53,
                protocol="udp",
                start_time=window_start + timedelta(seconds=30),
                end_time=window_start + timedelta(seconds=33),
                flow_count=2,
                total_pkt_count=6 * pkt_scale,
                total_byte_count=480 * byte_scale,
                avg_duration=2.5,
                avg_pkt_count=float(3 * pkt_scale),
                avg_byte_count=float(240 * byte_scale),
                source_flow_ids=("flow-c", "flow-d"),
                src_ports=(33333, 33334),
                directions=("outbound",),
                tcp_flags=(),
                is_aggregated_short_flow=True,
            ),
        ),
        stats=LogicalFlowWindowStats(
            index=start_offset_seconds // 60,
            window_start=window_start,
            window_end=window_end,
            raw_flow_count=4,
            short_flow_count=2,
            long_flow_count=2,
            logical_flow_count=3,
        ),
    )


class FeaturePackTest(unittest.TestCase):
    """Validate concatenation, normalization, and packed graph stability."""

    def setUp(self) -> None:
        """Create two graphs with different numeric scales for fitting and transform."""

        self.graphs = [
            build_endpoint_graph(_build_batch(), graph_config=_graph_config()),
            build_endpoint_graph(
                _build_batch(pkt_scale=2, byte_scale=3, start_offset_seconds=60),
                graph_config=_graph_config(),
            ),
        ]

    def test_node_feature_concatenation_order_is_stable(self) -> None:
        """Packed node features should concatenate base and structure fields deterministically."""

        feature_view = build_model_feature_view(self.graphs[0])

        self.assertEqual(feature_view.node_features.field_names, NODE_PACKED_FEATURE_FIELDS)
        self.assertEqual(feature_view.edge_features.field_names, EDGE_PACKED_FEATURE_FIELDS)

    def test_normalization_fit_and_transform_produce_scaled_continuous_fields(self) -> None:
        """Standard normalization should center varying continuous node fields."""

        preprocessor = fit_feature_preprocessor(
            self.graphs,
            normalization_config=FeatureNormalizationConfig(
                enabled=True,
                method="standard",
                exclude_node_fields=("endpoint_type", "port", "proto"),
                exclude_edge_fields=("edge_type", "is_aggregated"),
            ),
        )
        packed_graphs = transform_graphs(self.graphs, preprocessor)
        pkt_column_index = packed_graphs[0].node_feature_fields.index("total_pkt_count")
        combined_pkt_values = np.concatenate(
            [packed_graph.node_features[:, pkt_column_index] for packed_graph in packed_graphs]
        )

        self.assertAlmostEqual(float(np.mean(combined_pkt_values)), 0.0, places=7)
        self.assertAlmostEqual(float(np.std(combined_pkt_values)), 1.0, places=7)

    def test_discrete_fields_are_not_normalized(self) -> None:
        """Discrete node and edge fields should remain unchanged after transform."""

        preprocessor = fit_feature_preprocessor(self.graphs)
        packed_graph = transform_graph(self.graphs[0], preprocessor)
        raw_feature_view = build_model_feature_view(self.graphs[0])

        endpoint_type_index = packed_graph.node_feature_fields.index("endpoint_type")
        port_index = packed_graph.node_feature_fields.index("port")
        edge_type_index = packed_graph.edge_feature_fields.index("edge_type")
        is_aggregated_index = packed_graph.edge_feature_fields.index("is_aggregated")

        raw_node_matrix = np.asarray(raw_feature_view.node_features.feature_matrix, dtype=float)
        raw_edge_matrix = np.asarray(raw_feature_view.edge_features.feature_matrix, dtype=float)

        self.assertTrue(
            np.array_equal(
                packed_graph.node_features[:, endpoint_type_index],
                raw_node_matrix[:, endpoint_type_index],
            )
        )
        self.assertTrue(
            np.array_equal(
                packed_graph.node_features[:, port_index],
                raw_node_matrix[:, port_index],
            )
        )
        self.assertTrue(
            np.array_equal(
                packed_graph.edge_features[:, edge_type_index],
                raw_edge_matrix[:, edge_type_index],
            )
        )
        self.assertTrue(
            np.array_equal(
                packed_graph.edge_features[:, is_aggregated_index],
                raw_edge_matrix[:, is_aggregated_index],
            )
        )

    def test_packed_graph_dimensions_are_correct(self) -> None:
        """Packed graph matrices and index tensors should have the expected shapes."""

        preprocessor = fit_feature_preprocessor(self.graphs)
        packed_graph = transform_graph(self.graphs[0], preprocessor)

        self.assertEqual(packed_graph.node_features.shape, (6, 16))
        self.assertEqual(packed_graph.edge_features.shape, (5, 6))
        self.assertEqual(packed_graph.edge_index.shape, (2, 5))
        self.assertEqual(len(packed_graph.node_id_to_index), 6)
        self.assertEqual(len(packed_graph.edge_types), 5)

    def test_repeated_packing_is_stable(self) -> None:
        """Transforming the same graph twice with the same preprocessor should be deterministic."""

        preprocessor = fit_feature_preprocessor(self.graphs)
        first_packed = transform_graph(self.graphs[0], preprocessor)
        second_packed = transform_graph(self.graphs[0], preprocessor)

        self.assertTrue(np.array_equal(first_packed.node_features, second_packed.node_features))
        self.assertTrue(np.array_equal(first_packed.edge_features, second_packed.edge_features))
        self.assertTrue(np.array_equal(first_packed.edge_index, second_packed.edge_index))
        self.assertEqual(first_packed.node_id_to_index, second_packed.node_id_to_index)
        self.assertEqual(first_packed.edge_ids, second_packed.edge_ids)
