"""Tests for score, alert, and metrics persistence helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_ROOT: Path = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_graph.config import AlertingConfig
from traffic_graph.pipeline.alert_types import AlertScoreTables
from traffic_graph.pipeline.alerting import build_alert_records
from traffic_graph.pipeline.persistence import (
    ALERT_EXPORT_FIELDS,
    METRICS_EXPORT_FIELDS,
    SCORE_EXPORT_FIELDS,
    export_alert_records,
    export_metrics_summary,
    export_score_tables,
)
from traffic_graph.pipeline.report_io import export_run_bundle
from traffic_graph.cli import build_parser


class PersistenceTests(unittest.TestCase):
    """Exercise stable score, alert, and metrics file exports."""

    def setUp(self) -> None:
        """Prepare compact synthetic score tables and alert records."""

        self.score_tables = AlertScoreTables(
            graph_scores=(
                {
                    "graph_index": 0,
                    "window_index": 3,
                    "graph_anomaly_score": 0.25,
                    "graph_label": 0,
                    "node_count": 3,
                    "edge_count": 2,
                },
                {
                    "graph_index": 1,
                    "window_index": 4,
                    "graph_anomaly_score": 0.85,
                    "graph_label": 1,
                    "node_count": 4,
                    "edge_count": 3,
                },
            ),
            node_scores=(
                {
                    "graph_index": 0,
                    "window_index": 3,
                    "node_id": "node-a",
                    "node_anomaly_score": 0.60,
                    "endpoint_type": 0,
                },
            ),
            edge_scores=(
                {
                    "graph_index": 0,
                    "window_index": 3,
                    "edge_id": "edge-a",
                    "edge_anomaly_score": 0.72,
                    "edge_type": 0,
                    "is_aggregated": 1,
                },
            ),
            flow_scores=(
                {
                    "graph_index": 0,
                    "window_index": 3,
                    "logical_flow_id": "flow-a",
                    "flow_anomaly_score": 0.93,
                    "flow_label": 1,
                },
            ),
        )
        self.alerting_config = AlertingConfig(
            anomaly_threshold=0.5,
            medium_multiplier=1.5,
            high_multiplier=2.0,
        )
        self.alert_records = build_alert_records(self.score_tables, self.alerting_config)
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

    def test_score_table_jsonl_and_csv_exports_are_stable(self) -> None:
        """Score exports should keep a deterministic column order."""

        with tempfile.TemporaryDirectory() as temp_dir:
            result = export_score_tables(
                self.score_tables,
                temp_dir,
                run_id="run-1",
                split="eval",
                timestamp="20260408T090000Z",
                formats=("jsonl", "csv"),
                anomaly_threshold=0.5,
            )
            csv_path = Path(result.artifact_paths["graph_scores_csv"])
            jsonl_path = Path(result.artifact_paths["graph_scores_jsonl"])
            self.assertTrue(csv_path.exists())
            self.assertTrue(jsonl_path.exists())
            self.assertEqual(result.row_counts["graph_scores"], 2)

            csv_header = csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")
            self.assertEqual(csv_header, list(SCORE_EXPORT_FIELDS))

            jsonl_row = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(list(jsonl_row.keys()), list(SCORE_EXPORT_FIELDS))
            self.assertEqual(jsonl_row["split"], "eval")
            self.assertEqual(jsonl_row["threshold"], 0.5)

    def test_alert_exports_are_stable(self) -> None:
        """Alert exports should preserve the alert schema across file formats."""

        with tempfile.TemporaryDirectory() as temp_dir:
            result = export_alert_records(
                self.alert_records,
                temp_dir,
                run_id="run-1",
                split="eval",
                timestamp="20260408T090000Z",
                formats=("jsonl", "csv"),
            )
            csv_path = Path(result.artifact_paths["alert_records_csv"])
            jsonl_path = Path(result.artifact_paths["alert_records_jsonl"])
            self.assertTrue(csv_path.exists())
            self.assertTrue(jsonl_path.exists())
            self.assertEqual(result.row_counts["alert_records"], len(self.alert_records))

            csv_header = csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")
            self.assertEqual(csv_header, list(ALERT_EXPORT_FIELDS))

            jsonl_row = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(list(jsonl_row.keys()), list(ALERT_EXPORT_FIELDS))
            self.assertEqual(jsonl_row["alert_scope"], "graph")
            self.assertEqual(jsonl_row["is_alert"], False)

    def test_metrics_summary_exports_include_json_and_flat_tables(self) -> None:
        """Metrics summaries should emit a nested JSON file and a flat table view."""

        with tempfile.TemporaryDirectory() as temp_dir:
            result = export_metrics_summary(
                self.metrics_summary,
                temp_dir,
                run_id="run-1",
                split="eval",
                timestamp="20260408T090000Z",
                formats=("json", "csv", "jsonl"),
            )
            json_path = Path(result.artifact_paths["metrics_summary_json"])
            csv_path = Path(result.artifact_paths["metrics_summary_csv"])
            jsonl_path = Path(result.artifact_paths["metrics_summary_jsonl"])

            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())
            self.assertTrue(jsonl_path.exists())
            self.assertEqual(result.row_counts["metrics_rows"], 4)

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["graph"]["roc_auc"], 0.95)

            csv_header = csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")
            self.assertEqual(csv_header, list(METRICS_EXPORT_FIELDS))

    def test_run_bundle_gracefully_skips_missing_parquet_support(self) -> None:
        """Requesting parquet without support should not raise and should leave a note."""

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "traffic_graph.pipeline.persistence.is_parquet_export_available",
                return_value=False,
            ):
                result = export_run_bundle(
                    self.score_tables,
                    self.alert_records,
                    self.metrics_summary,
                    temp_dir,
                    run_id="run-1",
                    split="eval",
                    timestamp="20260408T090000Z",
                    score_formats=("parquet",),
                    alert_formats=("parquet",),
                    metrics_formats=("json", "parquet"),
                    anomaly_threshold=0.5,
                )
            bundle_dir = Path(result.run_directory)
            parquet_files = list(bundle_dir.rglob("*.parquet"))
            self.assertEqual(parquet_files, [])
            self.assertTrue(Path(result.manifest_path).exists())
            self.assertTrue(any("Parquet export skipped" in note for note in result.notes))

    def test_cli_parser_exposes_export_dir_flag(self) -> None:
        """The CLI should allow overriding the export directory."""

        parser = build_parser()
        args = parser.parse_args(["--export-dir", "exports"])
        self.assertEqual(args.export_dir, "exports")


if __name__ == "__main__":
    unittest.main()
