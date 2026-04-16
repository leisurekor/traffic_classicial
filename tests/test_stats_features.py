"""Unit tests for base statistical graph feature extraction."""

from __future__ import annotations

import csv
import io
import sys
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_ROOT: Path = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_graph.cli import main
from traffic_graph.config import AssociationEdgeConfig, GraphConfig
from traffic_graph.data import LogicalFlowBatch, LogicalFlowRecord, LogicalFlowWindowStats
from traffic_graph.features import (
    EDGE_BASE_FEATURE_FIELDS,
    EDGE_TYPE_ENCODING,
    ENDPOINT_TYPE_ENCODING,
    NODE_BASE_FEATURE_FIELDS,
    PROTO_ENCODING,
    build_base_feature_views,
    extract_edge_base_features,
    extract_node_base_features,
    summarize_feature_view,
)
from traffic_graph.graph import build_endpoint_graph


def _build_graph_config() -> GraphConfig:
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


def _build_feature_batch() -> LogicalFlowBatch:
    """Create a logical-flow batch that exercises communication and association edges."""

    window_start = datetime(2026, 4, 8, 11, 0, 0)
    window_end = window_start + timedelta(seconds=60)
    return LogicalFlowBatch(
        index=0,
        window_start=window_start,
        window_end=window_end,
        logical_flows=(
            LogicalFlowRecord(
                logical_flow_id="logical-long-001",
                src_ip="10.0.0.1",
                dst_ip="10.0.0.2",
                dst_port=80,
                protocol="tcp",
                start_time=datetime(2026, 4, 8, 11, 0, 5),
                end_time=datetime(2026, 4, 8, 11, 0, 10),
                flow_count=1,
                total_pkt_count=11,
                total_byte_count=4096,
                avg_duration=5.0,
                avg_pkt_count=11.0,
                avg_byte_count=4096.0,
                source_flow_ids=("flow-a",),
                src_ports=(11111,),
                directions=("outbound",),
                tcp_flags=("ACK",),
                is_aggregated_short_flow=False,
            ),
            LogicalFlowRecord(
                logical_flow_id="logical-long-002",
                src_ip="10.0.0.1",
                dst_ip="10.0.0.3",
                dst_port=443,
                protocol="tcp",
                start_time=datetime(2026, 4, 8, 11, 0, 15),
                end_time=datetime(2026, 4, 8, 11, 0, 21),
                flow_count=1,
                total_pkt_count=9,
                total_byte_count=3500,
                avg_duration=6.0,
                avg_pkt_count=9.0,
                avg_byte_count=3500.0,
                source_flow_ids=("flow-b",),
                src_ports=(22222,),
                directions=("outbound",),
                tcp_flags=("ACK",),
                is_aggregated_short_flow=False,
            ),
            LogicalFlowRecord(
                logical_flow_id="logical-short-agg-001",
                src_ip="172.16.0.8",
                dst_ip="192.168.1.10",
                dst_port=53,
                protocol="udp",
                start_time=datetime(2026, 4, 8, 11, 0, 30),
                end_time=datetime(2026, 4, 8, 11, 0, 33),
                flow_count=2,
                total_pkt_count=6,
                total_byte_count=480,
                avg_duration=2.5,
                avg_pkt_count=3.0,
                avg_byte_count=240.0,
                source_flow_ids=("flow-c", "flow-d"),
                src_ports=(33333, 33334),
                directions=("outbound",),
                tcp_flags=(),
                is_aggregated_short_flow=True,
            ),
        ),
        stats=LogicalFlowWindowStats(
            index=0,
            window_start=window_start,
            window_end=window_end,
            raw_flow_count=4,
            short_flow_count=2,
            long_flow_count=2,
            logical_flow_count=3,
        ),
    )


class StatsFeatureExtractionTest(unittest.TestCase):
    """Validate base node and edge feature extraction behavior."""

    def setUp(self) -> None:
        """Build a graph sample with both communication and association edges."""

        self.graph_sample = build_endpoint_graph(
            _build_feature_batch(),
            graph_config=_build_graph_config(),
        )

    def test_node_communication_statistics_are_correct(self) -> None:
        """Node features should aggregate incident communication edges only."""

        node_features = extract_node_base_features(self.graph_sample)
        client_row = node_features.feature_by_node_id["client:10.0.0.1:11111:tcp"]
        aggregated_server_row = node_features.feature_by_node_id["server:192.168.1.10:53:udp"]

        self.assertEqual(client_row["endpoint_type"], ENDPOINT_TYPE_ENCODING["client"])
        self.assertEqual(client_row["port"], 11111)
        self.assertEqual(client_row["proto"], PROTO_ENCODING["tcp"])
        self.assertEqual(client_row["total_pkt_count"], 11)
        self.assertEqual(client_row["total_byte_count"], 4096)
        self.assertEqual(client_row["total_flow_count"], 1)
        self.assertEqual(client_row["avg_pkt_count"], 11.0)
        self.assertEqual(client_row["avg_byte_count"], 4096.0)
        self.assertEqual(client_row["avg_duration"], 5.0)
        self.assertEqual(client_row["communication_edge_count"], 1)

        self.assertEqual(aggregated_server_row["endpoint_type"], ENDPOINT_TYPE_ENCODING["server"])
        self.assertEqual(aggregated_server_row["proto"], PROTO_ENCODING["udp"])
        self.assertEqual(aggregated_server_row["total_pkt_count"], 6)
        self.assertEqual(aggregated_server_row["total_byte_count"], 480)
        self.assertEqual(aggregated_server_row["total_flow_count"], 2)
        self.assertEqual(aggregated_server_row["avg_pkt_count"], 6.0)
        self.assertEqual(aggregated_server_row["avg_byte_count"], 480.0)
        self.assertEqual(aggregated_server_row["avg_duration"], 2.5)

    def test_node_association_edge_counts_are_correct(self) -> None:
        """Node features should count incident association edges separately."""

        node_features = extract_node_base_features(self.graph_sample)

        self.assertEqual(
            node_features.feature_by_node_id["client:10.0.0.1:11111:tcp"][
                "association_edge_count"
            ],
            1,
        )
        self.assertEqual(
            node_features.feature_by_node_id["server:10.0.0.2:80:tcp"][
                "association_edge_count"
            ],
            1,
        )
        self.assertEqual(
            node_features.feature_by_node_id["server:192.168.1.10:53:udp"][
                "association_edge_count"
            ],
            0,
        )

    def test_edge_features_are_generated_with_stable_encodings(self) -> None:
        """Edge features should preserve communication statistics and edge encodings."""

        edge_features = extract_edge_base_features(self.graph_sample)
        communication_row = edge_features.feature_by_edge_id["logical-short-agg-001"]

        self.assertEqual(communication_row["edge_type"], EDGE_TYPE_ENCODING["communication"])
        self.assertEqual(communication_row["pkt_count"], 6)
        self.assertEqual(communication_row["byte_count"], 480)
        self.assertEqual(communication_row["duration"], 2.5)
        self.assertEqual(communication_row["flow_count"], 2)
        self.assertEqual(communication_row["is_aggregated"], 1)

    def test_association_edge_default_values_are_zeroed(self) -> None:
        """Association edges should expose structural defaults for traffic statistics."""

        edge_features = extract_edge_base_features(self.graph_sample)
        association_edge_id = (
            "association_same_src_ip:client:10.0.0.1:11111:tcp->"
            "client:10.0.0.1:22222:tcp"
        )
        association_row = edge_features.feature_by_edge_id[association_edge_id]

        self.assertEqual(
            association_row["edge_type"],
            EDGE_TYPE_ENCODING["association_same_src_ip"],
        )
        self.assertEqual(association_row["pkt_count"], 0)
        self.assertEqual(association_row["byte_count"], 0)
        self.assertEqual(association_row["duration"], 0.0)
        self.assertEqual(association_row["flow_count"], 0)
        self.assertEqual(association_row["is_aggregated"], 0)

    def test_feature_order_and_dimensions_are_stable(self) -> None:
        """Feature field order and matrix shapes should be deterministic."""

        feature_view = build_base_feature_views(self.graph_sample)

        self.assertEqual(feature_view.node_features.field_names, NODE_BASE_FEATURE_FIELDS)
        self.assertEqual(feature_view.edge_features.field_names, EDGE_BASE_FEATURE_FIELDS)
        self.assertEqual(
            feature_view.node_features.ordered_node_ids,
            tuple(node.node_id for node in self.graph_sample.nodes),
        )
        self.assertEqual(
            feature_view.edge_features.ordered_edge_ids,
            tuple(edge.edge_id for edge in self.graph_sample.edges),
        )
        self.assertEqual(len(feature_view.node_features.feature_matrix), 6)
        self.assertEqual(len(feature_view.node_features.feature_matrix[0]), 12)
        self.assertEqual(len(feature_view.edge_features.feature_matrix), 5)
        self.assertEqual(len(feature_view.edge_features.feature_matrix[0]), 6)

        summary = summarize_feature_view(feature_view)
        self.assertEqual(
            summary,
            {
                "node_count": 6,
                "node_feature_dim": 12,
                "edge_count": 5,
                "edge_feature_dim": 6,
            },
        )

    def test_cli_can_render_feature_summaries(self) -> None:
        """The CLI should print per-window feature dimensions for a small sample."""

        with TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "flows.csv"
            config_path = Path(temp_dir) / "pipeline.yaml"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "flow_id",
                        "src_ip",
                        "src_port",
                        "dst_ip",
                        "dst_port",
                        "protocol",
                        "start_time",
                        "end_time",
                        "packet_count",
                        "byte_count",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "flow_id": "flow-001",
                        "src_ip": "10.0.0.1",
                        "src_port": 11111,
                        "dst_ip": "10.0.0.2",
                        "dst_port": 80,
                        "protocol": "tcp",
                        "start_time": "2026-04-08T11:00:05",
                        "end_time": "2026-04-08T11:00:10",
                        "packet_count": 11,
                        "byte_count": 4096,
                    }
                )
                writer.writerow(
                    {
                        "flow_id": "flow-002",
                        "src_ip": "10.0.0.1",
                        "src_port": 22222,
                        "dst_ip": "10.0.0.3",
                        "dst_port": 443,
                        "protocol": "tcp",
                        "start_time": "2026-04-08T11:00:15",
                        "end_time": "2026-04-08T11:00:21",
                        "packet_count": 9,
                        "byte_count": 3500,
                    }
                )

            config_path.write_text(
                "\n".join(
                    [
                        "preprocessing:",
                        "  window_size: 60",
                        "graph:",
                        "  association_edges:",
                        "    enable_same_src_ip: true",
                        "    enable_same_dst_subnet: true",
                        "    dst_subnet_prefix: 24",
                    ]
                ),
                encoding="utf-8",
            )

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main(
                    [
                        "--config",
                        str(config_path),
                        "--input",
                        str(csv_path),
                        "--dry-run",
                        "--show-feature-summary",
                    ]
                )

            rendered = buffer.getvalue()

        self.assertEqual(exit_code, 0)
        self.assertIn("Feature summaries:", rendered)
        self.assertIn("node_features=16x4, edge_features=6x4", rendered)
        self.assertIn("Node feature fields:", rendered)
        self.assertIn("Edge feature fields:", rendered)
