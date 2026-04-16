"""Unit tests for endpoint interaction graph construction."""

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
from traffic_graph.graph import (
    InteractionGraph,
    build_endpoint_graph,
    build_endpoint_graphs,
    summarize_graph,
)


def _build_logical_flow_batch() -> LogicalFlowBatch:
    """Create a deterministic logical-flow batch for graph construction tests."""

    window_start = datetime(2026, 4, 8, 9, 0, 0)
    window_end = window_start + timedelta(seconds=60)
    logical_flows = (
        LogicalFlowRecord(
            logical_flow_id="logical-long-001",
            src_ip="10.0.0.1",
            dst_ip="10.0.0.2",
            dst_port=80,
            protocol="tcp",
            start_time=datetime(2026, 4, 8, 9, 0, 5),
            end_time=datetime(2026, 4, 8, 9, 0, 12),
            flow_count=1,
            total_pkt_count=12,
            total_byte_count=4096,
            avg_duration=7.0,
            avg_pkt_count=12.0,
            avg_byte_count=4096.0,
            source_flow_ids=("flow-001",),
            src_ports=(12345,),
            directions=("outbound",),
            tcp_flags=("SYN", "ACK"),
            is_aggregated_short_flow=False,
        ),
        LogicalFlowRecord(
            logical_flow_id="logical-long-002",
            src_ip="10.0.0.1",
            dst_ip="10.0.0.3",
            dst_port=443,
            protocol="tcp",
            start_time=datetime(2026, 4, 8, 9, 0, 15),
            end_time=datetime(2026, 4, 8, 9, 0, 19),
            flow_count=1,
            total_pkt_count=9,
            total_byte_count=2048,
            avg_duration=4.0,
            avg_pkt_count=9.0,
            avg_byte_count=2048.0,
            source_flow_ids=("flow-002",),
            src_ports=(12345,),
            directions=("outbound",),
            tcp_flags=("PSH", "ACK"),
            is_aggregated_short_flow=False,
        ),
        LogicalFlowRecord(
            logical_flow_id="logical-short-agg-001",
            src_ip="10.0.0.4",
            dst_ip="10.0.0.5",
            dst_port=53,
            protocol="udp",
            start_time=datetime(2026, 4, 8, 9, 0, 25),
            end_time=datetime(2026, 4, 8, 9, 0, 30),
            flow_count=2,
            total_pkt_count=6,
            total_byte_count=480,
            avg_duration=2.5,
            avg_pkt_count=3.0,
            avg_byte_count=240.0,
            source_flow_ids=("flow-003", "flow-004"),
            src_ports=(20001, 20002),
            directions=("outbound",),
            tcp_flags=(),
            is_aggregated_short_flow=True,
        ),
    )
    return LogicalFlowBatch(
        index=0,
        window_start=window_start,
        window_end=window_end,
        logical_flows=logical_flows,
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


def _build_association_logical_flow_batch() -> LogicalFlowBatch:
    """Create a logical-flow batch that can trigger both association edge types."""

    window_start = datetime(2026, 4, 8, 10, 0, 0)
    window_end = window_start + timedelta(seconds=60)
    logical_flows = (
        LogicalFlowRecord(
            logical_flow_id="logical-assoc-001",
            src_ip="10.0.0.1",
            dst_ip="10.0.0.2",
            dst_port=80,
            protocol="tcp",
            start_time=datetime(2026, 4, 8, 10, 0, 5),
            end_time=datetime(2026, 4, 8, 10, 0, 10),
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
            logical_flow_id="logical-assoc-002",
            src_ip="10.0.0.1",
            dst_ip="10.0.0.3",
            dst_port=443,
            protocol="tcp",
            start_time=datetime(2026, 4, 8, 10, 0, 15),
            end_time=datetime(2026, 4, 8, 10, 0, 21),
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
            logical_flow_id="logical-assoc-003",
            src_ip="172.16.0.8",
            dst_ip="192.168.1.10",
            dst_port=53,
            protocol="udp",
            start_time=datetime(2026, 4, 8, 10, 0, 30),
            end_time=datetime(2026, 4, 8, 10, 0, 31),
            flow_count=1,
            total_pkt_count=6,
            total_byte_count=1500,
            avg_duration=1.0,
            avg_pkt_count=6.0,
            avg_byte_count=1500.0,
            source_flow_ids=("flow-c",),
            src_ports=(33333,),
            directions=("outbound",),
            tcp_flags=(),
            is_aggregated_short_flow=False,
        ),
    )
    return LogicalFlowBatch(
        index=0,
        window_start=window_start,
        window_end=window_end,
        logical_flows=logical_flows,
        stats=LogicalFlowWindowStats(
            index=0,
            window_start=window_start,
            window_end=window_end,
            raw_flow_count=3,
            short_flow_count=0,
            long_flow_count=3,
            logical_flow_count=3,
        ),
    )


def _association_graph_config(
    *,
    enable_same_src_ip: bool = True,
    enable_same_dst_subnet: bool = True,
) -> GraphConfig:
    """Build a graph config with explicit association-edge settings."""

    return GraphConfig(
        time_window_seconds=60,
        directed=True,
        association_edges=AssociationEdgeConfig(
            enable_same_src_ip=enable_same_src_ip,
            enable_same_dst_subnet=enable_same_dst_subnet,
            dst_subnet_prefix=24,
        ),
    )


class EndpointGraphTest(unittest.TestCase):
    """Validate communication-edge graph construction and summary behavior."""

    def setUp(self) -> None:
        """Build a reusable logical-flow batch and graph sample for each test."""

        self.batch = _build_logical_flow_batch()
        self.graph_sample = build_endpoint_graph(self.batch)

    def test_same_endpoint_is_deduplicated(self) -> None:
        """Repeated use of the same endpoint should not create duplicate nodes."""

        self.assertIsInstance(self.graph_sample, InteractionGraph)
        self.assertEqual(self.graph_sample.node_count, 5)
        self.assertEqual(self.graph_sample.graph.number_of_nodes(), 5)

        client_nodes = [
            node for node in self.graph_sample.nodes if node.endpoint_type == "client"
        ]
        self.assertEqual(len(client_nodes), 2)
        self.assertIn(
            ("10.0.0.1", 12345, "tcp"),
            [(node.ip, node.port, node.proto) for node in client_nodes],
        )
        self.assertIn(
            ("10.0.0.4", (20001, 20002), "udp"),
            [(node.ip, node.port, node.proto) for node in client_nodes],
        )

    def test_edges_are_built_for_each_logical_flow(self) -> None:
        """Each logical flow should map to exactly one communication edge."""

        self.assertEqual(self.graph_sample.edge_count, 3)
        self.assertEqual(self.graph_sample.graph.number_of_edges(), 3)
        self.assertEqual(
            {edge.logical_flow_id for edge in self.graph_sample.edges},
            {
                "logical-long-001",
                "logical-long-002",
                "logical-short-agg-001",
            },
        )
        self.assertTrue(
            all(edge.edge_type == "communication" for edge in self.graph_sample.edges)
        )

    def test_aggregated_short_flow_edge_attributes_are_preserved(self) -> None:
        """Aggregated logical flows should retain their communication statistics."""

        aggregated_edge = next(
            edge for edge in self.graph_sample.edges if edge.is_aggregated
        )

        self.assertEqual(aggregated_edge.logical_flow_id, "logical-short-agg-001")
        self.assertEqual(aggregated_edge.pkt_count, 6)
        self.assertEqual(aggregated_edge.byte_count, 480)
        self.assertEqual(aggregated_edge.duration, 2.5)
        self.assertEqual(aggregated_edge.flow_count, 2)
        self.assertEqual(aggregated_edge.source_flow_ids, ("flow-003", "flow-004"))
        self.assertEqual(aggregated_edge.edge_type, "communication")

        backend_edges = self.graph_sample.graph.edges(data=True, keys=True)
        matching_backend_edge = next(
            edge_data for edge_data in backend_edges if edge_data[2] == aggregated_edge.edge_id
        )
        self.assertEqual(matching_backend_edge[3]["pkt_count"], 6)
        self.assertEqual(matching_backend_edge[3]["byte_count"], 480)
        self.assertEqual(matching_backend_edge[3]["duration"], 2.5)
        self.assertTrue(matching_backend_edge[3]["is_aggregated"])
        self.assertEqual(matching_backend_edge[3]["edge_type"], "communication")

    def test_graph_summary_statistics_are_correct(self) -> None:
        """Graph summary should report the expected node and edge counts."""

        summary = summarize_graph(self.graph_sample)

        self.assertEqual(
            summary,
            {
                "window_index": 0,
                "node_count": 5,
                "edge_count": 3,
                "client_node_count": 2,
                "server_node_count": 3,
                "aggregated_edge_count": 1,
                "communication_edge_count": 3,
                "association_edge_count": 0,
                "association_same_src_ip_edge_count": 0,
                "association_same_dst_subnet_edge_count": 0,
            },
        )

    def test_multiple_windows_can_be_built_in_order(self) -> None:
        """The graph builder should support ordered per-window construction."""

        second_batch = LogicalFlowBatch(
            index=1,
            window_start=self.batch.window_start + timedelta(seconds=60),
            window_end=self.batch.window_end + timedelta(seconds=60),
            logical_flows=self.batch.logical_flows[:1],
            stats=LogicalFlowWindowStats(
                index=1,
                window_start=self.batch.window_start + timedelta(seconds=60),
                window_end=self.batch.window_end + timedelta(seconds=60),
                raw_flow_count=1,
                short_flow_count=0,
                long_flow_count=1,
                logical_flow_count=1,
            ),
        )

        graphs = build_endpoint_graphs([self.batch, second_batch])

        self.assertEqual(len(graphs), 2)
        self.assertEqual(graphs[0].window_index, 0)
        self.assertEqual(graphs[1].window_index, 1)


class AssociationEdgeTest(unittest.TestCase):
    """Validate same-source-IP and same-destination-subnet association edges."""

    def setUp(self) -> None:
        """Create a reusable batch that can trigger both association edge rules."""

        self.batch = _build_association_logical_flow_batch()

    def test_same_src_ip_association_edges_are_added(self) -> None:
        """Client endpoints that share the same source IP should be linked."""

        graph_sample = build_endpoint_graph(
            self.batch,
            graph_config=_association_graph_config(
                enable_same_src_ip=True,
                enable_same_dst_subnet=False,
            ),
        )
        same_src_edges = [
            edge
            for edge in graph_sample.edges
            if edge.edge_type == "association_same_src_ip"
        ]

        self.assertEqual(len(same_src_edges), 1)
        self.assertEqual(graph_sample.stats.association_same_src_ip_edge_count, 1)
        self.assertEqual(
            {same_src_edges[0].source_node_id, same_src_edges[0].target_node_id},
            {
                "client:10.0.0.1:11111:tcp",
                "client:10.0.0.1:22222:tcp",
            },
        )

    def test_same_dst_subnet_association_edges_are_added(self) -> None:
        """Server endpoints inside the same /24 should be linked."""

        graph_sample = build_endpoint_graph(
            self.batch,
            graph_config=_association_graph_config(
                enable_same_src_ip=False,
                enable_same_dst_subnet=True,
            ),
        )
        same_subnet_edges = [
            edge
            for edge in graph_sample.edges
            if edge.edge_type == "association_same_dst_subnet"
        ]

        self.assertEqual(len(same_subnet_edges), 1)
        self.assertEqual(graph_sample.stats.association_same_dst_subnet_edge_count, 1)
        self.assertEqual(
            {same_subnet_edges[0].source_node_id, same_subnet_edges[0].target_node_id},
            {
                "server:10.0.0.2:80:tcp",
                "server:10.0.0.3:443:tcp",
            },
        )

    def test_config_disabled_keeps_only_communication_edges(self) -> None:
        """Association edges should not be added when all rules are disabled."""

        graph_sample = build_endpoint_graph(
            self.batch,
            graph_config=_association_graph_config(
                enable_same_src_ip=False,
                enable_same_dst_subnet=False,
            ),
        )

        self.assertEqual(graph_sample.edge_count, 3)
        self.assertEqual(graph_sample.stats.communication_edge_count, 3)
        self.assertEqual(graph_sample.stats.association_edge_count, 0)
        self.assertTrue(
            all(edge.edge_type == "communication" for edge in graph_sample.edges)
        )

    def test_edge_type_summary_statistics_are_correct(self) -> None:
        """Graph summary should distinguish communication and association edges."""

        graph_sample = build_endpoint_graph(
            self.batch,
            graph_config=_association_graph_config(),
        )
        summary = summarize_graph(graph_sample)

        self.assertEqual(graph_sample.edge_count, 5)
        self.assertEqual(graph_sample.stats.communication_edge_count, 3)
        self.assertEqual(graph_sample.stats.association_edge_count, 2)
        self.assertEqual(summary["association_same_src_ip_edge_count"], 1)
        self.assertEqual(summary["association_same_dst_subnet_edge_count"], 1)

    def test_cli_can_render_graph_summaries_with_association_edges(self) -> None:
        """The CLI should print communication and association edge counts."""

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
                        "start_time": "2026-04-08T10:00:05",
                        "end_time": "2026-04-08T10:00:10",
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
                        "start_time": "2026-04-08T10:00:15",
                        "end_time": "2026-04-08T10:00:21",
                        "packet_count": 9,
                        "byte_count": 3500,
                    }
                )
                writer.writerow(
                    {
                        "flow_id": "flow-003",
                        "src_ip": "172.16.0.8",
                        "src_port": 33333,
                        "dst_ip": "192.168.1.10",
                        "dst_port": 53,
                        "protocol": "udp",
                        "start_time": "2026-04-08T10:00:30",
                        "end_time": "2026-04-08T10:00:31",
                        "packet_count": 6,
                        "byte_count": 1500,
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
                        "--show-graph-summary",
                    ]
                )

            rendered = buffer.getvalue()

        self.assertEqual(exit_code, 0)
        self.assertIn("Graph summaries:", rendered)
        self.assertIn("communication_edges=3, association_edges=2", rendered)
        self.assertIn("same_src_ip_edges=1, same_dst_subnet_edges=1", rendered)
