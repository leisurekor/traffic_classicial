"""Unit tests for time-window preprocessing and short-flow aggregation."""

from __future__ import annotations

import csv
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_ROOT: Path = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_graph.cli import main
from traffic_graph.data import (
    FlowDataset,
    LogicalFlowRecord,
    ShortFlowThresholds,
    merge_short_flows,
    preprocess_flow_dataset,
    split_into_windows,
)


def _build_sample_dataset() -> FlowDataset:
    """Create a deterministic dataset spanning two one-minute windows."""

    return FlowDataset.from_mappings(
        [
            {
                "flow_id": "flow-001",
                "src_ip": "10.0.0.1",
                "src_port": 11111,
                "dst_ip": "10.0.0.2",
                "dst_port": 80,
                "protocol": "tcp",
                "start_time": "2026-04-08T09:00:05",
                "end_time": "2026-04-08T09:00:07",
                "packet_count": 2,
                "byte_count": 100,
            },
            {
                "flow_id": "flow-002",
                "src_ip": "10.0.0.1",
                "src_port": 22222,
                "dst_ip": "10.0.0.2",
                "dst_port": 80,
                "protocol": "tcp",
                "start_time": "2026-04-08T09:00:40",
                "end_time": "2026-04-08T09:00:46",
                "packet_count": 4,
                "byte_count": 400,
            },
            {
                "flow_id": "flow-003",
                "src_ip": "10.0.0.3",
                "src_port": 34567,
                "dst_ip": "10.0.0.4",
                "dst_port": 443,
                "protocol": "tcp",
                "start_time": "2026-04-08T09:00:50",
                "end_time": "2026-04-08T09:00:58",
                "packet_count": 10,
                "byte_count": 5000,
            },
            {
                "flow_id": "flow-004",
                "src_ip": "10.0.0.1",
                "src_port": 11111,
                "dst_ip": "10.0.0.2",
                "dst_port": 80,
                "protocol": "tcp",
                "start_time": "2026-04-08T09:01:10",
                "end_time": "2026-04-08T09:01:12",
                "packet_count": 1,
                "byte_count": 200,
            },
            {
                "flow_id": "flow-005",
                "src_ip": "10.0.0.1",
                "src_port": 44444,
                "dst_ip": "10.0.0.2",
                "dst_port": 80,
                "protocol": "tcp",
                "start_time": "2026-04-08T09:01:59",
                "end_time": "2026-04-08T09:02:03",
                "packet_count": 8,
                "byte_count": 512,
            },
        ]
    )


def _build_rules() -> ShortFlowThresholds:
    """Return the reusable short-flow rule set for tests."""

    return ShortFlowThresholds(packet_count_lt=5, byte_count_lt=1024)


class PreprocessingTest(unittest.TestCase):
    """Validate window splitting, short-flow rules, and aggregation semantics."""

    def setUp(self) -> None:
        """Create a shared sample dataset for each test case."""

        self.dataset = _build_sample_dataset()
        self.rules = _build_rules()

    def test_split_into_windows_groups_by_start_time(self) -> None:
        """Flows should be assigned to windows using `start_time` flooring."""

        windows = split_into_windows(self.dataset, window_size=60)

        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0].window_start.isoformat(), "2026-04-08T09:00:00")
        self.assertEqual(windows[1].window_start.isoformat(), "2026-04-08T09:01:00")
        self.assertEqual(
            tuple(record.flow_id for record in windows[0].records),
            ("flow-001", "flow-002", "flow-003"),
        )
        self.assertEqual(
            tuple(record.flow_id for record in windows[1].records),
            ("flow-004", "flow-005"),
        )

    def test_short_flow_rules_cover_packet_and_byte_thresholds(self) -> None:
        """Short-flow classification should trigger on either configured threshold."""

        flows_by_id = {record.flow_id: record for record in self.dataset.records}

        self.assertTrue(self.rules.matches(flows_by_id["flow-001"]))
        self.assertTrue(self.rules.matches(flows_by_id["flow-002"]))
        self.assertTrue(self.rules.matches(flows_by_id["flow-004"]))
        self.assertTrue(self.rules.matches(flows_by_id["flow-005"]))
        self.assertFalse(self.rules.matches(flows_by_id["flow-003"]))

    def test_merge_short_flows_aggregates_within_window(self) -> None:
        """Short flows with the same aggregation key should merge into one logical flow."""

        first_window = split_into_windows(self.dataset, window_size=60)[0]
        logical_batch = merge_short_flows(first_window, self.rules)
        aggregated_records = [
            record
            for record in logical_batch.logical_flows
            if record.is_aggregated_short_flow
        ]

        self.assertEqual(logical_batch.stats.raw_flow_count, 3)
        self.assertEqual(logical_batch.stats.short_flow_count, 2)
        self.assertEqual(logical_batch.stats.long_flow_count, 1)
        self.assertEqual(logical_batch.stats.logical_flow_count, 2)
        self.assertEqual(len(aggregated_records), 1)

        aggregated = aggregated_records[0]
        self.assertEqual(aggregated.flow_count, 2)
        self.assertEqual(aggregated.total_pkt_count, 6)
        self.assertEqual(aggregated.total_byte_count, 500)
        self.assertEqual(aggregated.avg_duration, 4.0)
        self.assertEqual(aggregated.avg_pkt_count, 3.0)
        self.assertEqual(aggregated.avg_byte_count, 250.0)
        self.assertEqual(aggregated.source_flow_ids, ("flow-001", "flow-002"))
        self.assertEqual(aggregated.src_ports, (11111, 22222))

    def test_long_flows_are_preserved_as_passthrough_logical_flows(self) -> None:
        """Long flows should remain single logical flows after preprocessing."""

        batches = preprocess_flow_dataset(self.dataset, window_size=60, rules=self.rules)
        first_window_long_flows = [
            record
            for record in batches[0].logical_flows
            if not record.is_aggregated_short_flow
        ]

        self.assertEqual(len(first_window_long_flows), 1)
        passthrough = first_window_long_flows[0]
        self.assertIsInstance(passthrough, LogicalFlowRecord)
        self.assertEqual(passthrough.logical_flow_id, "flow-003")
        self.assertEqual(passthrough.flow_count, 1)
        self.assertEqual(passthrough.total_pkt_count, 10)
        self.assertEqual(passthrough.total_byte_count, 5000)
        self.assertEqual(passthrough.source_flow_ids, ("flow-003",))

    def test_cli_can_render_window_level_statistics(self) -> None:
        """The CLI should print window statistics for a small CSV sample."""

        with TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "flows.csv"
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
                for row in self.dataset.to_mappings():
                    writer.writerow(
                        {
                            "flow_id": row["flow_id"],
                            "src_ip": row["src_ip"],
                            "src_port": row["src_port"],
                            "dst_ip": row["dst_ip"],
                            "dst_port": row["dst_port"],
                            "protocol": row["protocol"],
                            "start_time": row["start_time"],
                            "end_time": row["end_time"],
                            "packet_count": row["packet_count"],
                            "byte_count": row["byte_count"],
                        }
                    )

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main(
                    [
                        "--input",
                        str(csv_path),
                        "--dry-run",
                        "--show-window-stats",
                    ]
                )

            rendered = buffer.getvalue()

        self.assertEqual(exit_code, 0)
        self.assertIn("Window statistics:", rendered)
        self.assertIn("raw=3, short=2, long=1, logical=2", rendered)
        self.assertIn("raw=2, short=2, long=0, logical=1", rendered)
