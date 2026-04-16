"""Tests for reading back persisted export bundles."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_ROOT: Path = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_graph.cli import build_parser, main
from traffic_graph.config import AlertingConfig
from traffic_graph.pipeline.alert_types import AlertScoreTables
from traffic_graph.pipeline.alerting import build_alert_records
from traffic_graph.pipeline.persistence import ALERT_EXPORT_FIELDS, SCORE_EXPORT_FIELDS
from traffic_graph.pipeline.replay_io import (
    get_alert_records,
    get_metrics_summary,
    get_score_table,
    list_available_tables,
    load_alert_records,
    load_export_bundle,
    load_metrics_summary,
    load_score_table,
    summarize_replay_bundle,
)
from traffic_graph.pipeline.report_io import export_run_bundle


class ReplayBundleTests(unittest.TestCase):
    """Exercise run-bundle readback behavior and format fallback logic."""

    def setUp(self) -> None:
        """Prepare synthetic score tables, alerts, and metrics for replay tests."""

        self.score_tables = AlertScoreTables(
            graph_scores=(
                {
                    "graph_index": 0,
                    "window_index": 11,
                    "graph_anomaly_score": 0.35,
                    "graph_label": 0,
                    "node_count": 3,
                },
                {
                    "graph_index": 1,
                    "window_index": 12,
                    "graph_anomaly_score": 0.92,
                    "graph_label": 1,
                    "node_count": 5,
                },
            ),
            node_scores=(
                {
                    "graph_index": 0,
                    "window_index": 11,
                    "node_id": "node-1",
                    "node_anomaly_score": 0.61,
                    "endpoint_type": 0,
                },
            ),
            edge_scores=(
                {
                    "graph_index": 0,
                    "window_index": 11,
                    "edge_id": "edge-1",
                    "edge_anomaly_score": 0.72,
                    "edge_type": 0,
                    "is_aggregated": 1,
                },
            ),
            flow_scores=(
                {
                    "graph_index": 0,
                    "window_index": 11,
                    "logical_flow_id": "flow-1",
                    "flow_anomaly_score": 1.05,
                    "flow_label": 1,
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
            "graph": {
                "roc_auc": 0.95,
                "pr_auc": 0.91,
            },
            "flow": {
                "roc_auc": 0.89,
                "pr_auc": 0.84,
            },
        }

    def _export_bundle(self, temp_dir: str) -> Path:
        """Export a deterministic run bundle for replay tests."""

        result = export_run_bundle(
            self.score_tables,
            self.alert_records,
            self.metrics_summary,
            temp_dir,
            run_id="replay-run",
            split="eval",
            timestamp="20260409T010203Z",
            score_formats=("jsonl", "csv"),
            alert_formats=("jsonl", "csv"),
            metrics_formats=("json", "jsonl", "csv"),
            anomaly_threshold=0.5,
        )
        return Path(result.run_directory)

    def test_manifest_and_tables_load_from_bundle_directory(self) -> None:
        """The bundle loader should recover all exported tables from a run directory."""

        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_dir = self._export_bundle(temp_dir)
            bundle = load_export_bundle(bundle_dir)

            self.assertEqual(bundle.run_id, "replay-run")
            self.assertEqual(bundle.timestamp, "20260409T010203Z")
            self.assertEqual(bundle.split, "eval")
            self.assertTrue(bundle.manifest.manifest_path.endswith("manifest.json"))
            self.assertEqual(
                list_available_tables(bundle),
                ("graph", "node", "edge", "flow", "alerts", "metrics"),
            )
            self.assertEqual(len(get_score_table(bundle, "graph")), 2)
            self.assertEqual(len(get_score_table(bundle, "node")), 1)
            self.assertEqual(len(get_score_table(bundle, "edge")), 1)
            self.assertEqual(len(get_score_table(bundle, "flow")), 1)
            self.assertEqual(len(get_alert_records(bundle, only_positive=False)), 5)
            self.assertEqual(len(get_alert_records(bundle)), 4)
            self.assertEqual(get_metrics_summary(bundle)["graph"]["roc_auc"], 0.95)
            self.assertTrue(bundle.loaded_files["graph_scores"].endswith(".jsonl"))

    def test_manifest_path_and_direct_table_loaders_work(self) -> None:
        """The loader should accept a manifest path and direct helper functions should work."""

        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_dir = self._export_bundle(temp_dir)
            manifest_path = bundle_dir / "manifest.json"
            bundle = load_export_bundle(manifest_path)

            graph_scores = load_score_table(bundle.loaded_files["graph_scores"])
            alert_rows = load_alert_records(bundle.loaded_files["alert_records"])
            metrics = load_metrics_summary(bundle.loaded_files["metrics_summary"])

            self.assertEqual(len(graph_scores), 2)
            self.assertEqual(graph_scores[0].score_scope, "graph")
            self.assertEqual(len(alert_rows), 5)
            self.assertEqual(alert_rows[0].alert_scope, "graph")
            self.assertEqual(metrics["flow"]["pr_auc"], 0.84)

    def test_csv_fallback_and_missing_optional_table_are_handled_cleanly(self) -> None:
        """Replay should fall back to CSV and emit a clear note for missing optional files."""

        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_dir = self._export_bundle(temp_dir)
            manifest_path = bundle_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            Path(manifest["artifact_paths"]["graph_scores_jsonl"]).unlink()
            Path(manifest["artifact_paths"]["edge_scores_jsonl"]).unlink()
            Path(manifest["artifact_paths"]["edge_scores_csv"]).unlink()

            bundle = load_export_bundle(bundle_dir)

            self.assertTrue(bundle.loaded_files["graph_scores"].endswith(".csv"))
            self.assertEqual(len(bundle.graph_scores), 2)
            self.assertEqual(bundle.edge_scores, ())
            self.assertNotIn("edge", list_available_tables(bundle))
            self.assertTrue(
                any("No exported edge score table was found" in note for note in bundle.notes)
            )

    def test_parquet_priority_wins_when_available(self) -> None:
        """When parquet is available, the replay loader should prefer it over JSONL/CSV."""

        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_dir = self._export_bundle(temp_dir)
            manifest_path = bundle_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            parquet_path = bundle_dir / "scores" / "graph_scores.eval.parquet"
            parquet_path.write_text("placeholder", encoding="utf-8")
            manifest["artifact_paths"]["graph_scores_parquet"] = parquet_path.as_posix()
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            parquet_rows = [
                {
                    "run_id": "replay-run",
                    "timestamp": "20260409T010203Z",
                    "split": "eval",
                    "score_scope": "graph",
                    "graph_id": 99,
                    "window_id": 88,
                    "node_id": None,
                    "edge_id": None,
                    "flow_id": None,
                    "anomaly_score": 9.0,
                    "threshold": 0.5,
                    "is_alert": True,
                    "label": 1,
                    "metadata": "{\"source\": \"parquet\"}",
                }
            ]

            with patch(
                "traffic_graph.pipeline.replay_io.is_parquet_export_available",
                return_value=True,
            ), patch(
                "traffic_graph.pipeline.replay_io._read_parquet_rows",
                return_value=parquet_rows,
            ):
                bundle = load_export_bundle(bundle_dir)

            self.assertEqual(bundle.loaded_files["graph_scores"], parquet_path.as_posix())
            self.assertEqual(len(bundle.graph_scores), 1)
            self.assertEqual(bundle.graph_scores[0].graph_id, 99)
            self.assertEqual(bundle.graph_scores[0].metadata["source"], "parquet")

    def test_field_order_and_bundle_summary_are_stable(self) -> None:
        """Typed replay records should expose stable field names and a readable summary."""

        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_dir = self._export_bundle(temp_dir)
            bundle = load_export_bundle(bundle_dir)

            graph_payload = bundle.graph_scores[0].to_dict()
            alert_payload = bundle.alert_records[0].to_dict()
            summary_text = summarize_replay_bundle(bundle)

            self.assertEqual(list(graph_payload.keys()), list(SCORE_EXPORT_FIELDS))
            self.assertEqual(list(alert_payload.keys()), list(ALERT_EXPORT_FIELDS))
            self.assertIn("Replay run id: replay-run", summary_text)
            self.assertIn("graph_scores=2", summary_text)

    def test_cli_replay_mode_prints_bundle_summary(self) -> None:
        """The CLI should expose a replay mode for lightweight bundle checks."""

        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_dir = self._export_bundle(temp_dir)
            parser = build_parser()
            parsed_args = parser.parse_args(["--replay-bundle", bundle_dir.as_posix()])
            self.assertEqual(parsed_args.replay_bundle, bundle_dir.as_posix())

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["--replay-bundle", bundle_dir.as_posix()])

            rendered = stdout.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertIn("Replay run id: replay-run", rendered)
            self.assertIn("Available tables:", rendered)


if __name__ == "__main__":
    unittest.main()

