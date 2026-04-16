"""Tests for surrogate decision-tree training and persistence."""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_ROOT: Path = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_graph.cli import build_parser, main
from traffic_graph.config import AlertingConfig
from traffic_graph.explain import (
    SurrogateTreeConfig,
    build_explanation_samples,
    extract_training_matrix,
    load_surrogate_tree_artifact,
    save_surrogate_tree_artifact,
    summarize_surrogate_tree_artifact,
    train_surrogate_tree,
)
from traffic_graph.pipeline.alert_types import AlertScoreTables
from traffic_graph.pipeline.alerting import build_alert_records
from traffic_graph.pipeline.replay_io import load_export_bundle
from traffic_graph.pipeline.report_io import export_run_bundle


class SurrogateTreeTests(unittest.TestCase):
    """Exercise surrogate tree training, persistence, and CLI plumbing."""

    def setUp(self) -> None:
        """Prepare a deterministic replay bundle fixture for surrogate training."""

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
        """Export a deterministic run bundle used by surrogate-tree tests."""

        result = export_run_bundle(
            self.score_tables,
            self.alert_records,
            self.metrics_summary,
            temp_dir,
            run_id="surrogate-run",
            split="eval",
            timestamp="20260409T030405Z",
            score_formats=("jsonl", "csv"),
            alert_formats=("jsonl", "csv"),
            metrics_formats=("json", "jsonl", "csv"),
            anomaly_threshold=0.5,
        )
        return Path(result.run_directory)

    def test_extract_training_matrix_has_stable_feature_order(self) -> None:
        """The training matrix should preserve deterministic feature ordering."""

        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = load_export_bundle(self._export_bundle(temp_dir))
            samples = build_explanation_samples(
                bundle,
                scope="flow",
                only_alerts=False,
            )
            matrix = extract_training_matrix(samples)

            self.assertEqual(matrix.sample_scope, "flow")
            self.assertEqual(matrix.target_name, "anomaly_score")
            self.assertEqual(
                matrix.feature_names,
                (
                    "stats_summary.byte_count",
                    "stats_summary.dst_port",
                    "stats_summary.duration",
                    "stats_summary.flow_count",
                    "stats_summary.is_aggregated",
                    "stats_summary.pkt_count",
                    "stats_summary.proto",
                    "stats_summary.src_port",
                    "graph_summary.aggregated_edge_count",
                    "graph_summary.association_edge_count",
                    "graph_summary.client_node_count",
                    "graph_summary.communication_edge_count",
                    "graph_summary.edge_count",
                    "graph_summary.graph_anomaly_score",
                    "graph_summary.graph_is_alert",
                    "graph_summary.graph_label",
                    "graph_summary.graph_threshold",
                    "graph_summary.node_count",
                    "graph_summary.server_node_count",
                    "feature_summary.available",
                    "feature_summary.field_count",
                ),
            )
            self.assertEqual(matrix.features.shape[0], 2)
            self.assertEqual(matrix.features.shape[1], len(matrix.feature_names))

    def test_regression_mode_trains_and_persists(self) -> None:
        """Regression mode should fit a tree that approximates anomaly scores."""

        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = load_export_bundle(self._export_bundle(temp_dir))
            samples = build_explanation_samples(
                bundle,
                scope="flow",
                only_alerts=False,
            )
            artifact = train_surrogate_tree(
                samples,
                SurrogateTreeConfig(
                    mode="regression",
                    max_depth=4,
                    min_samples_leaf=1,
                    random_state=7,
                ),
            )
            save_result = save_surrogate_tree_artifact(
                artifact,
                Path(temp_dir) / "surrogate_tree",
            )
            loaded = load_surrogate_tree_artifact(save_result.output_directory)

            self.assertEqual(artifact.summary.mode, "regression")
            self.assertEqual(artifact.summary.target_name, "anomaly_score")
            self.assertGreaterEqual(artifact.summary.leaf_count, 1)
            self.assertEqual(artifact.feature_names, loaded.feature_names)
            self.assertEqual(loaded.summary.mode, "regression")
            self.assertIn("Surrogate tree:", summarize_surrogate_tree_artifact(artifact))
            self.assertTrue(Path(save_result.model_path).exists())
            self.assertTrue(Path(save_result.metadata_path).exists())

    def test_classification_mode_uses_pseudo_alert_labels(self) -> None:
        """Classification mode should derive targets from the alert decision."""

        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = load_export_bundle(self._export_bundle(temp_dir))
            samples = build_explanation_samples(
                bundle,
                scope="flow",
                only_alerts=False,
            )
            artifact = train_surrogate_tree(
                samples,
                SurrogateTreeConfig(
                    mode="classification",
                    max_depth=3,
                    min_samples_leaf=1,
                    random_state=11,
                ),
            )

            self.assertEqual(artifact.summary.mode, "classification")
            self.assertEqual(artifact.summary.target_name, "is_alert")
            self.assertGreaterEqual(artifact.summary.tree_depth, 0)
            self.assertGreaterEqual(artifact.summary.leaf_count, 1)

    def test_empty_samples_raise_a_clear_error(self) -> None:
        """The training helpers should reject empty sample collections."""

        with self.assertRaises(ValueError):
            train_surrogate_tree([], SurrogateTreeConfig())

    def test_constant_targets_still_train_a_tree(self) -> None:
        """A constant anomaly score should still produce a valid surrogate tree."""

        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = load_export_bundle(self._export_bundle(temp_dir))
            samples = build_explanation_samples(
                bundle,
                scope="flow",
                only_alerts=False,
            )
            constant_samples = [
                replace(samples[0], anomaly_score=1.0),
                replace(samples[1], anomaly_score=1.0),
            ]
            artifact = train_surrogate_tree(
                constant_samples,
                SurrogateTreeConfig(mode="regression", max_depth=4, min_samples_leaf=1),
            )

            self.assertEqual(artifact.summary.sample_count, 2)
            self.assertEqual(artifact.summary.tree_depth, 0)
            self.assertEqual(artifact.summary.leaf_count, 1)

    def test_cli_can_train_surrogate_tree_from_replay_bundle(self) -> None:
        """The CLI should train and persist a surrogate tree from a replay bundle."""

        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_dir = self._export_bundle(temp_dir)
            parser = build_parser()
            parsed_args = parser.parse_args(
                [
                    "--replay-bundle",
                    bundle_dir.as_posix(),
                    "--train-surrogate-tree",
                    "--surrogate-scope",
                    "flow",
                    "--surrogate-mode",
                    "regression",
                    "--surrogate-max-depth",
                    "3",
                    "--surrogate-min-samples-leaf",
                    "1",
                ]
            )
            self.assertTrue(parsed_args.train_surrogate_tree)
            self.assertEqual(parsed_args.surrogate_scope, "flow")
            self.assertEqual(parsed_args.surrogate_mode, "regression")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--replay-bundle",
                        bundle_dir.as_posix(),
                        "--train-surrogate-tree",
                        "--surrogate-scope",
                        "flow",
                        "--surrogate-mode",
                        "regression",
                        "--surrogate-max-depth",
                        "3",
                        "--surrogate-min-samples-leaf",
                        "1",
                    ]
                )

            rendered = stdout.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertIn("Replay run id: surrogate-run", rendered)
            self.assertIn("Surrogate tree summary:", rendered)
            self.assertIn("mode=regression", rendered)
            self.assertIn("Saved surrogate tree artifact", rendered)


if __name__ == "__main__":
    unittest.main()
