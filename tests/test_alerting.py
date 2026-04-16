"""Tests for threshold-based anomaly alert generation."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_ROOT: Path = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_graph.cli import build_parser
from traffic_graph.config import AlertingConfig
from traffic_graph.pipeline.alerting import (
    build_alert_records,
    filter_alerts,
    summarize_alerts,
)
from traffic_graph.pipeline.alert_types import AlertRecord, AlertScoreTables
from traffic_graph.pipeline.runner import PipelineReport, PipelineStage


class AlertingTests(unittest.TestCase):
    """Exercise alert thresholding and structured alert output."""

    def setUp(self) -> None:
        """Prepare a compact set of synthetic score tables."""

        self.config = AlertingConfig(
            anomaly_threshold=0.5,
            medium_multiplier=1.5,
            high_multiplier=2.0,
        )
        self.score_tables = AlertScoreTables(
            graph_scores=(
                {
                    "graph_index": 0,
                    "window_index": 7,
                    "graph_anomaly_score": 0.30,
                    "graph_label": 0,
                    "node_count": 4,
                },
                {
                    "graph_index": 1,
                    "window_index": 8,
                    "graph_anomaly_score": 0.75,
                    "graph_label": 1,
                    "node_count": 5,
                },
                {
                    "graph_index": 2,
                    "window_index": 9,
                    "graph_anomaly_score": 1.20,
                    "graph_label": 1,
                    "node_count": 6,
                },
            ),
            node_scores=(
                {
                    "graph_index": 0,
                    "window_index": 7,
                    "node_id": "node-0",
                    "node_anomaly_score": 0.49,
                    "endpoint_type": 0,
                },
                {
                    "graph_index": 0,
                    "window_index": 7,
                    "node_id": "node-1",
                    "node_anomaly_score": 0.90,
                    "endpoint_type": 1,
                },
            ),
            edge_scores=(
                {
                    "graph_index": 0,
                    "window_index": 7,
                    "edge_id": "edge-0",
                    "edge_anomaly_score": 0.70,
                    "edge_type": 0,
                    "is_aggregated": 1,
                },
            ),
            flow_scores=(
                {
                    "graph_index": 0,
                    "window_index": 7,
                    "logical_flow_id": "flow-0",
                    "flow_anomaly_score": 1.05,
                    "flow_label": 1,
                },
            ),
        )

    def test_threshold_and_levels_are_applied(self) -> None:
        """Alert booleans and severity levels should follow the configured threshold."""

        records = build_alert_records(self.score_tables, self.config)
        graph_records = [record for record in records if record.alert_scope == "graph"]
        self.assertEqual([record.is_alert for record in graph_records], [False, True, True])
        self.assertEqual([record.alert_level for record in graph_records], ["low", "medium", "high"])

    def test_alert_records_keep_expected_fields(self) -> None:
        """Structured alert records should expose the expected ids and metadata."""

        records = build_alert_records(self.score_tables, self.config)
        graph_record = next(record for record in records if record.alert_scope == "graph" and record.graph_id == 1)
        node_record = next(record for record in records if record.alert_scope == "node" and record.node_id == "node-1")
        edge_record = next(record for record in records if record.alert_scope == "edge")
        flow_record = next(record for record in records if record.alert_scope == "flow")

        self.assertEqual(graph_record.window_id, 8)
        self.assertEqual(graph_record.label, 1)
        self.assertEqual(node_record.label, None)
        self.assertEqual(edge_record.edge_id, "edge-0")
        self.assertEqual(edge_record.metadata["edge_type"], 0)
        self.assertEqual(flow_record.flow_id, "flow-0")
        self.assertEqual(flow_record.label, 1)
        self.assertEqual(graph_record.metadata["node_count"], 5)
        self.assertIn("endpoint_type", node_record.metadata)

    def test_filtering_by_scope_and_positive_only(self) -> None:
        """Filtering should support scope restriction and negative-record retention."""

        records = build_alert_records(self.score_tables, self.config)
        node_alerts = filter_alerts(records, scope="node", only_positive=True)
        self.assertEqual(len(node_alerts), 1)
        self.assertEqual(node_alerts[0].node_id, "node-1")

        all_graph_records = filter_alerts(records, scope="graph", only_positive=False)
        self.assertEqual(len(all_graph_records), 3)

    def test_summary_counts_are_stable(self) -> None:
        """The alert summary should report per-scope and per-level counts."""

        records = build_alert_records(self.score_tables, self.config)
        summary = summarize_alerts(records)
        self.assertEqual(summary["total_count"], 7)
        self.assertEqual(summary["positive_count"], 5)
        self.assertEqual(summary["scope_counts"]["graph"], 3)
        self.assertEqual(summary["scope_counts"]["node"], 2)
        self.assertEqual(summary["scope_counts"]["edge"], 1)
        self.assertEqual(summary["scope_counts"]["flow"], 1)
        self.assertEqual(summary["level_counts"]["low"], 3)
        self.assertEqual(summary["level_counts"]["medium"], 2)
        self.assertEqual(summary["level_counts"]["high"], 2)

    def test_report_render_includes_alert_summary(self) -> None:
        """Pipeline reports should print a compact alert summary section."""

        records = build_alert_records(self.score_tables, self.config)
        summary = summarize_alerts(records)
        report = PipelineReport(
            run_name="unit-test",
            input_path="data/flows.csv",
            output_directory="artifacts",
            stages=[PipelineStage(name="eval", status="completed", detail="done")],
            dry_run=False,
            alert_summary=summary,
        )
        rendered = report.render()
        self.assertIn("Alert summary:", rendered)
        self.assertIn("graph: total=3, positive=2", rendered)
        self.assertIn("levels: low=3, medium=2, high=2", rendered)

    def test_cli_parser_exposes_alert_summary_flag(self) -> None:
        """The CLI should expose a switch for alert summary output."""

        parser = build_parser()
        args = parser.parse_args(["--show-alert-summary"])
        self.assertTrue(args.show_alert_summary)

    def test_alert_record_to_dict_is_serializable(self) -> None:
        """Alert records should provide a JSON-friendly dictionary view."""

        record = AlertRecord(
            alert_id="graph:0:7:0:0",
            alert_level="medium",
            alert_scope="graph",
            graph_id=0,
            window_id=7,
            anomaly_score=0.75,
            threshold=0.5,
            is_alert=True,
            label=1,
            metadata={"node_count": 4},
        )
        payload = record.to_dict()
        self.assertEqual(payload["alert_id"], "graph:0:7:0:0")
        self.assertEqual(payload["metadata"]["node_count"], 4)


if __name__ == "__main__":
    unittest.main()
