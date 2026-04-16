"""Tests for explanation-ready sample organization built on replay bundles."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_ROOT: Path = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_graph.cli import build_parser, main
from traffic_graph.config import AlertingConfig
from traffic_graph.explain import (
    build_explanation_samples,
    export_explanation_candidates,
    select_balanced_samples_for_explanation,
    select_top_alert_samples,
)
from traffic_graph.pipeline.alert_types import AlertScoreTables
from traffic_graph.pipeline.alerting import build_alert_records
from traffic_graph.pipeline.replay_io import load_export_bundle
from traffic_graph.pipeline.report_io import export_run_bundle


class ExplanationSampleTests(unittest.TestCase):
    """Exercise explanation-ready sample construction and replay-side helpers."""

    def setUp(self) -> None:
        """Prepare a small deterministic bundle fixture for explanation tests."""

        self.score_tables = AlertScoreTables(
            graph_scores=(
                {
                    "graph_index": 0,
                    "window_index": 11,
                    "graph_anomaly_score": 0.35,
                    "node_count": 3,
                    "edge_count": 2,
                    "client_node_count": 1,
                    "server_node_count": 2,
                    "communication_edge_count": 1,
                    "association_edge_count": 1,
                    "aggregated_edge_count": 0,
                },
                {
                    "graph_index": 1,
                    "window_index": 12,
                    "graph_anomaly_score": 1.20,
                    "graph_label": 1,
                    "node_count": 5,
                    "edge_count": 4,
                    "client_node_count": 2,
                    "server_node_count": 3,
                    "communication_edge_count": 2,
                    "association_edge_count": 2,
                    "aggregated_edge_count": 1,
                },
            ),
            node_scores=(
                {
                    "graph_index": 1,
                    "window_index": 12,
                    "node_id": "client-1",
                    "node_anomaly_score": 0.91,
                    "node_label": 1,
                    "endpoint_type": 0,
                    "port": 51515,
                    "proto": 6,
                    "total_pkt_count": 15,
                    "total_byte_count": 4096,
                    "total_flow_count": 4,
                    "avg_pkt_count": 7.5,
                    "avg_byte_count": 2048.0,
                    "avg_duration": 0.20,
                    "communication_edge_count": 2,
                    "association_edge_count": 1,
                    "total_degree": 3,
                    "communication_in_degree": 0,
                    "communication_out_degree": 2,
                    "unique_neighbor_count": 2,
                },
                {
                    "graph_index": 0,
                    "window_index": 11,
                    "node_id": "server-1",
                    "node_anomaly_score": 0.11,
                    "endpoint_type": 1,
                    "port": 443,
                    "proto": 6,
                },
            ),
            edge_scores=(
                {
                    "graph_index": 1,
                    "window_index": 12,
                    "edge_id": "edge-1",
                    "edge_anomaly_score": 0.66,
                    "edge_type": 0,
                    "pkt_count": 8,
                },
            ),
            flow_scores=(
                {
                    "graph_index": 1,
                    "window_index": 12,
                    "logical_flow_id": "flow-a",
                    "flow_anomaly_score": 1.05,
                    "flow_label": 1,
                    "pkt_count": 8,
                    "byte_count": 2048,
                    "duration": 0.40,
                    "flow_count": 3,
                    "is_aggregated": 1,
                    "src_ip": "10.0.0.5",
                    "src_port": 51515,
                    "dst_ip": "172.16.1.20",
                    "dst_port": 443,
                    "proto": 6,
                },
                {
                    "graph_index": 0,
                    "window_index": 11,
                    "logical_flow_id": "flow-b",
                    "flow_anomaly_score": 0.20,
                    "pkt_count": 2,
                    "byte_count": 128,
                    "duration": 0.05,
                    "flow_count": 1,
                    "is_aggregated": 0,
                    "dst_port": 53,
                    "proto": 17,
                },
            ),
        )
        self.alert_records = build_alert_records(
            self.score_tables,
            AlertingConfig(
                anomaly_threshold=0.5,
                medium_multiplier=1.5,
                high_multiplier=2.0,
            ),
        )
        self.metrics_summary = {
            "graph": {"roc_auc": 0.91, "pr_auc": 0.87},
            "flow": {"roc_auc": 0.89, "pr_auc": 0.83},
        }

    def _export_bundle(self, temp_dir: str) -> Path:
        """Export a deterministic run bundle used by explanation tests."""

        result = export_run_bundle(
            self.score_tables,
            self.alert_records,
            self.metrics_summary,
            temp_dir,
            run_id="explain-run",
            split="eval",
            timestamp="20260409T020304Z",
            score_formats=("jsonl", "csv"),
            alert_formats=("jsonl", "csv"),
            metrics_formats=("json", "jsonl", "csv"),
            anomaly_threshold=0.5,
        )
        return Path(result.run_directory)

    def test_graph_level_samples_include_graph_context(self) -> None:
        """Graph-level explanation samples should preserve graph stats and labels."""

        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = load_export_bundle(self._export_bundle(temp_dir))
            samples = build_explanation_samples(
                bundle,
                scope="graph",
                only_alerts=False,
            )

            self.assertEqual(len(samples), 2)
            self.assertEqual(samples[0].graph_id, 1)
            self.assertEqual(samples[0].alert_level, "high")
            self.assertEqual(samples[0].stats_summary["node_count"], 5)
            self.assertEqual(samples[0].graph_summary["graph_anomaly_score"], 1.20)
            self.assertEqual(samples[0].label, 1)
            self.assertIsNone(samples[1].label)

    def test_flow_level_samples_include_stats_and_filters(self) -> None:
        """Flow-level sample organization should preserve flow stats and alert filters."""

        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = load_export_bundle(self._export_bundle(temp_dir))
            samples = build_explanation_samples(
                bundle,
                scope="flow",
                only_alerts=False,
            )

            self.assertEqual(len(samples), 2)
            self.assertEqual(samples[0].flow_id, "flow-a")
            self.assertEqual(samples[0].stats_summary["pkt_count"], 8)
            self.assertEqual(samples[0].stats_summary["dst_port"], 443)
            self.assertEqual(samples[0].graph_summary["node_count"], 5)
            self.assertFalse(samples[0].feature_summary["available"])
            self.assertEqual(samples[0].label, 1)

            positive_samples = build_explanation_samples(
                bundle,
                scope="flow",
                only_alerts=True,
            )
            self.assertEqual([sample.flow_id for sample in positive_samples], ["flow-a"])

            top_sample = build_explanation_samples(
                bundle,
                scope="flow",
                only_alerts=False,
                top_k=1,
            )
            self.assertEqual(len(top_sample), 1)
            self.assertEqual(top_sample[0].flow_id, "flow-a")

    def test_selection_helpers_balance_and_rank_samples(self) -> None:
        """Selection helpers should rank alerts and produce a balanced subset."""

        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = load_export_bundle(self._export_bundle(temp_dir))
            samples = build_explanation_samples(
                bundle,
                scope="node",
                only_alerts=False,
            )

            top_alerts = select_top_alert_samples(samples, k=5)
            balanced = select_balanced_samples_for_explanation(samples, max_samples=2)

            self.assertEqual([sample.node_id for sample in top_alerts], ["client-1"])
            self.assertEqual(len(balanced), 2)
            self.assertEqual(sum(sample.is_alert is True for sample in balanced), 1)
            self.assertEqual(sum(sample.is_alert is False for sample in balanced), 1)

    def test_explanation_candidate_export_uses_stable_field_order(self) -> None:
        """Exported explanation candidate rows should be JSONL and preserve field order."""

        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = load_export_bundle(self._export_bundle(temp_dir))
            samples = build_explanation_samples(
                bundle,
                scope="flow",
                only_alerts=False,
            )
            output_path = Path(temp_dir) / "explanations" / "candidates.jsonl"
            exported_path = export_explanation_candidates(samples, output_path)
            lines = Path(exported_path).read_text(encoding="utf-8").strip().splitlines()
            payload = json.loads(lines[0])

            self.assertEqual(
                list(payload.keys()),
                [
                    "sample_id",
                    "scope",
                    "run_id",
                    "graph_id",
                    "window_id",
                    "flow_id",
                    "node_id",
                    "anomaly_score",
                    "threshold",
                    "is_alert",
                    "alert_level",
                    "label",
                    "stats_summary",
                    "graph_summary",
                    "feature_summary",
                    "metadata",
                ],
            )
            self.assertEqual(len(lines), 2)

    def test_cli_replay_mode_can_render_explanation_summary(self) -> None:
        """The replay CLI should expose the explanation candidate summary view."""

        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_dir = self._export_bundle(temp_dir)
            parser = build_parser()
            parsed_args = parser.parse_args(
                [
                    "--replay-bundle",
                    bundle_dir.as_posix(),
                    "--show-explanation-summary",
                    "--explanation-scope",
                    "flow",
                    "--explanation-top-k",
                    "1",
                ]
            )
            self.assertTrue(parsed_args.show_explanation_summary)
            self.assertEqual(parsed_args.explanation_scope, "flow")
            self.assertEqual(parsed_args.explanation_top_k, 1)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--replay-bundle",
                        bundle_dir.as_posix(),
                        "--show-explanation-summary",
                        "--explanation-scope",
                        "flow",
                        "--explanation-top-k",
                        "1",
                    ]
                )

            rendered = stdout.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertIn("Replay run id: explain-run", rendered)
            self.assertIn("Explanation candidate summary:", rendered)
            self.assertIn("Selection: scope=flow, only_alerts=True, top_k=1", rendered)
            self.assertIn("Explanation samples: total=1", rendered)


if __name__ == "__main__":
    unittest.main()
