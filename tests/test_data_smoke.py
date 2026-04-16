"""Smoke tests for the normalized flow data layer."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_ROOT: Path = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_graph.data import FlowDataset, FlowRecord


class DataLayerSmokeTest(unittest.TestCase):
    """Validate that the unified flow schema can be instantiated and summarized."""

    def test_flow_record_and_dataset_smoke(self) -> None:
        """Build a small dataset and verify normalization helpers work end to end."""

        first_record = FlowRecord.from_mapping(
            {
                "flow_id": "flow-001",
                "src_ip": "10.0.0.1",
                "src_port": 12345,
                "dst_ip": "10.0.0.2",
                "dst_port": 443,
                "protocol": "TCP",
                "start_time": "2026-04-08T09:00:00",
                "end_time": "2026-04-08T09:00:05",
                "packet_count": 12,
                "byte_count": 4096,
                "direction": "outbound",
                "tcp_flags": ["syn", "ack"],
                "metadata": {"capture": "pcap-a"},
            }
        )
        dataset = FlowDataset.from_mappings(
            [
                first_record.to_mapping(),
                {
                    "flow_id": "flow-002",
                    "src_ip": "10.0.0.2",
                    "src_port": 443,
                    "dst_ip": "10.0.0.3",
                    "dst_port": 53000,
                    "protocol": "tcp",
                    "start_time": "2026-04-08T09:00:02",
                    "end_time": "2026-04-08T09:00:07",
                    "packet_count": 8,
                    "byte_count": 2048,
                },
            ]
        )

        summary = dataset.summary()
        serialized_rows = dataset.to_mappings()

        self.assertEqual(first_record.protocol, "tcp")
        self.assertEqual(first_record.tcp_flags, ("SYN", "ACK"))
        self.assertEqual(summary.flow_count, 2)
        self.assertEqual(summary.protocols, ("tcp",))
        self.assertEqual(summary.average_duration_seconds, 5.0)
        self.assertEqual(serialized_rows[0]["flow_id"], "flow-001")
        self.assertEqual(serialized_rows[1]["dst_ip"], "10.0.0.3")
