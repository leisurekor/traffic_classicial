"""Tests for surrogate-tree path extraction and rule record generation."""

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
    ExplanationSample,
    SurrogateTreeConfig,
    RULE_PATH_CONDITION_FIELDS,
    RULE_RECORD_FIELDS,
    extract_rule_for_sample,
    extract_rules_for_samples,
    export_rule_records,
    summarize_rule,
    summarize_rules,
    train_surrogate_tree,
)
from traffic_graph.pipeline.alert_types import AlertScoreTables
from traffic_graph.pipeline.alerting import build_alert_records
from traffic_graph.pipeline.replay_io import load_export_bundle
from traffic_graph.pipeline.report_io import export_run_bundle


class RuleRecordTests(unittest.TestCase):
    """Exercise path extraction, structured rules, and CLI integration."""

    def _build_simple_samples(self) -> list[ExplanationSample]:
        """Construct a deterministic sample set with a simple two-level tree."""

        return [
            ExplanationSample(
                sample_id="flow:0:0:0",
                scope="flow",
                run_id="rule-run",
                graph_id=0,
                window_id=0,
                flow_id="flow-0",
                anomaly_score=0.0,
                threshold=0.5,
                is_alert=False,
                stats_summary={"a": 0.0, "b": 0.0},
                graph_summary={},
                feature_summary={},
            ),
            ExplanationSample(
                sample_id="flow:0:0:1",
                scope="flow",
                run_id="rule-run",
                graph_id=0,
                window_id=0,
                flow_id="flow-1",
                anomaly_score=0.0,
                threshold=0.5,
                is_alert=False,
                stats_summary={"a": 0.0, "b": 1.0},
                graph_summary={},
                feature_summary={},
            ),
            ExplanationSample(
                sample_id="flow:1:0:0",
                scope="flow",
                run_id="rule-run",
                graph_id=1,
                window_id=0,
                flow_id="flow-2",
                anomaly_score=1.0,
                threshold=0.5,
                is_alert=True,
                stats_summary={"a": 1.0, "b": 0.0},
                graph_summary={},
                feature_summary={},
            ),
            ExplanationSample(
                sample_id="flow:1:0:1",
                scope="flow",
                run_id="rule-run",
                graph_id=1,
                window_id=0,
                flow_id="flow-3",
                anomaly_score=2.0,
                threshold=0.5,
                is_alert=True,
                stats_summary={"a": 1.0, "b": 1.0},
                graph_summary={},
                feature_summary={},
            ),
        ]

    def _train_regression_tree(self):
        """Fit a deterministic regression surrogate tree for the simple samples."""

        samples = self._build_simple_samples()
        artifact = train_surrogate_tree(
            samples,
            SurrogateTreeConfig(
                mode="regression",
                max_depth=2,
                min_samples_leaf=1,
                random_state=0,
            ),
        )
        return artifact, samples

    def test_single_sample_path_conditions_are_ordered(self) -> None:
        """A single sample should yield a stable root-to-leaf path order."""

        artifact, samples = self._train_regression_tree()
        rule = extract_rule_for_sample(artifact, samples[-1])

        self.assertEqual(rule.scope, "flow")
        self.assertEqual(rule.tree_mode, "regression")
        self.assertEqual(
            [condition.feature_name for condition in rule.path_conditions],
            ["stats_summary.a", "stats_summary.b"],
        )
        self.assertEqual([condition.tree_node_index for condition in rule.path_conditions], [0, 2])
        self.assertEqual(rule.path_conditions[0].operator, ">")
        self.assertEqual(rule.path_conditions[0].sample_value, 1.0)
        self.assertEqual(rule.feature_names_used, ("stats_summary.a", "stats_summary.b"))
        self.assertAlmostEqual(float(rule.predicted_score_or_class), 2.0, places=6)
        self.assertTrue(rule.rule_id.startswith("flow:1:0:1:regression:leaf-"))
        self.assertIn("scope=flow", summarize_rule(rule))

    def test_classification_mode_is_supported(self) -> None:
        """Classification mode should emit a class label and preserve path order."""

        samples = self._build_simple_samples()
        artifact = train_surrogate_tree(
            samples,
            SurrogateTreeConfig(
                mode="classification",
                max_depth=2,
                min_samples_leaf=1,
                random_state=0,
            ),
        )
        rule = extract_rule_for_sample(artifact, samples[-1])

        self.assertEqual(rule.tree_mode, "classification")
        self.assertEqual(rule.predicted_score_or_class, 1)
        self.assertEqual(
            [condition.feature_name for condition in rule.path_conditions],
            ["stats_summary.a"],
        )
        self.assertEqual(rule.feature_names_used, ("stats_summary.a",))

    def test_batch_extraction_and_jsonl_export_are_stable(self) -> None:
        """Batch extraction should preserve order and stable export field names."""

        artifact, samples = self._train_regression_tree()
        rule_records = extract_rules_for_samples(artifact, samples)

        self.assertEqual(len(rule_records), 4)
        self.assertEqual(rule_records[0].sample_id, samples[0].sample_id)
        self.assertEqual(rule_records[-1].sample_id, samples[-1].sample_id)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "rules.jsonl"
            exported_path = export_rule_records(rule_records, output_path)
            lines = Path(exported_path).read_text(encoding="utf-8").strip().splitlines()
            payload = json.loads(lines[0])
            path_payload = payload["path_conditions"][0]

            self.assertEqual(list(payload.keys()), list(RULE_RECORD_FIELDS))
            self.assertEqual(list(path_payload.keys()), list(RULE_PATH_CONDITION_FIELDS))
            self.assertEqual(len(lines), 4)

        summary_text = summarize_rules(rule_records)
        self.assertIn("Rule records: total=4", summary_text)
        self.assertIn("Modes: regression=4", summary_text)

    def test_cli_can_train_and_emit_rule_summary(self) -> None:
        """The CLI should train a surrogate tree and optionally export rule records."""

        score_tables = AlertScoreTables(
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
            node_scores=(),
            edge_scores=(),
            flow_scores=(
                {
                    "graph_index": 0,
                    "window_index": 11,
                    "logical_flow_id": "flow-a",
                    "flow_anomaly_score": 0.20,
                    "pkt_count": 2,
                    "byte_count": 128,
                    "duration": 0.05,
                    "flow_count": 1,
                    "is_aggregated": 0,
                    "dst_port": 53,
                    "proto": 17,
                },
                {
                    "graph_index": 1,
                    "window_index": 12,
                    "logical_flow_id": "flow-b",
                    "flow_anomaly_score": 1.10,
                    "pkt_count": 8,
                    "byte_count": 2048,
                    "duration": 0.40,
                    "flow_count": 3,
                    "is_aggregated": 1,
                    "dst_port": 443,
                    "proto": 6,
                },
            ),
        )
        alert_records = build_alert_records(
            score_tables,
            AlertingConfig(
                anomaly_threshold=0.5,
                medium_multiplier=1.5,
                high_multiplier=2.0,
            ),
        )
        metrics_summary = {"flow": {"roc_auc": 0.9, "pr_auc": 0.8}}

        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_dir = Path(
                export_run_bundle(
                    score_tables,
                    alert_records,
                    metrics_summary,
                    temp_dir,
                    run_id="rule-cli-run",
                    split="eval",
                    timestamp="20260409T040506Z",
                    score_formats=("jsonl", "csv"),
                    alert_formats=("jsonl", "csv"),
                    metrics_formats=("json", "jsonl", "csv"),
                    anomaly_threshold=0.5,
                ).run_directory
            )

            parser = build_parser()
            parsed_args = parser.parse_args(
                [
                    "--replay-bundle",
                    bundle_dir.as_posix(),
                    "--train-surrogate-tree",
                    "--show-rule-summary",
                    "--surrogate-scope",
                    "flow",
                    "--rule-output-dir",
                    (Path(temp_dir) / "rules").as_posix(),
                ]
            )
            self.assertTrue(parsed_args.show_rule_summary)
            self.assertEqual(parsed_args.rule_output_dir, (Path(temp_dir) / "rules").as_posix())

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--replay-bundle",
                        bundle_dir.as_posix(),
                        "--train-surrogate-tree",
                        "--show-rule-summary",
                        "--surrogate-scope",
                        "flow",
                        "--rule-output-dir",
                        (Path(temp_dir) / "rules").as_posix(),
                    ]
                )

            rendered = stdout.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertIn("Rule record summary:", rendered)
            self.assertIn("Saved rule records to", rendered)
            self.assertTrue((Path(temp_dir) / "rules" / "rule_records.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
