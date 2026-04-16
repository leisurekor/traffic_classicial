"""Tests for prompt dataset batching and export helpers."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_ROOT: Path = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_graph.cli import build_parser
from traffic_graph.explain import (
    PROMPT_INPUT_FIELDS,
    PromptDatasetArtifact,
    build_prompt_dataset,
    export_prompt_dataset,
    summarize_prompt_dataset_text,
)
from traffic_graph.explain.explanation_types import ExplanationSample
from traffic_graph.explain.rule_records import RulePathCondition, RuleRecord
from traffic_graph.pipeline.replay_types import ReplayAlertRecord, ReplayScoreRecord


class PromptDatasetTests(unittest.TestCase):
    """Exercise prompt dataset selection, summarization, and export."""

    def setUp(self) -> None:
        """Prepare a small deterministic dataset for prompt batching tests."""

        self.samples = [
            self._make_sample(
                sample_id="graph:10:100:none",
                scope="graph",
                graph_id=10,
                window_id=100,
                anomaly_score=0.92,
                threshold=0.50,
                is_alert=True,
                alert_level="high",
                label=1,
                stats_summary={"node_count": 6, "edge_count": 8},
                graph_summary={"communication_edge_count": 4, "association_edge_count": 4},
                feature_summary={"available": True, "field_count": 3},
            ),
            self._make_sample(
                sample_id="flow:10:100:flow-1",
                scope="flow",
                graph_id=10,
                window_id=100,
                flow_id="flow-1",
                anomaly_score=0.88,
                threshold=0.50,
                is_alert=True,
                alert_level="medium",
                label=None,
                stats_summary={"pkt_count": 11, "byte_count": 4096, "duration": 0.42},
                graph_summary={"communication_edge_count": 4, "node_count": 6},
                feature_summary={"available": True, "field_count": 4},
            ),
            self._make_sample(
                sample_id="flow:10:100:flow-2",
                scope="flow",
                graph_id=10,
                window_id=100,
                flow_id="flow-2",
                anomaly_score=0.41,
                threshold=0.50,
                is_alert=False,
                alert_level=None,
                label=None,
                stats_summary={"pkt_count": 2, "byte_count": 256, "duration": 0.08},
                graph_summary={"communication_edge_count": 4, "node_count": 6},
                feature_summary={"available": True, "field_count": 4},
            ),
            self._make_sample(
                sample_id="node:10:100:node-1",
                scope="node",
                graph_id=10,
                window_id=100,
                node_id="node-1",
                anomaly_score=0.73,
                threshold=0.50,
                is_alert=True,
                alert_level="medium",
                label=None,
                stats_summary={"port": 443, "proto": 6, "total_pkt_count": 9},
                graph_summary={"node_count": 6, "edge_count": 8},
                feature_summary={"available": True, "field_count": 5},
            ),
        ]
        self.rules = [
            self._make_rule(
                sample_id="graph:10:100:none",
                scope="graph",
                tree_mode="regression",
                predicted_value=0.92,
                leaf_id=4,
                path_conditions=(
                    RulePathCondition(
                        feature_name="stats_summary.node_count",
                        operator=">",
                        threshold=4.5,
                        sample_value=6.0,
                        tree_node_index=0,
                    ),
                ),
                feature_names_used=("stats_summary.node_count",),
            ),
            self._make_rule(
                sample_id="flow:10:100:flow-1",
                scope="flow",
                tree_mode="regression",
                predicted_value=0.88,
                leaf_id=6,
                path_conditions=(
                    RulePathCondition(
                        feature_name="stats_summary.pkt_count",
                        operator=">",
                        threshold=4.5,
                        sample_value=11.0,
                        tree_node_index=0,
                    ),
                ),
                feature_names_used=("stats_summary.pkt_count",),
            ),
            self._make_rule(
                sample_id="flow:10:100:flow-2",
                scope="flow",
                tree_mode="regression",
                predicted_value=0.41,
                leaf_id=2,
                path_conditions=(
                    RulePathCondition(
                        feature_name="stats_summary.pkt_count",
                        operator="<=",
                        threshold=4.5,
                        sample_value=2.0,
                        tree_node_index=0,
                    ),
                ),
                feature_names_used=("stats_summary.pkt_count",),
            ),
            self._make_rule(
                sample_id="node:10:100:node-1",
                scope="node",
                tree_mode="classification",
                predicted_value=1,
                leaf_id=3,
                path_conditions=(
                    RulePathCondition(
                        feature_name="stats_summary.port",
                        operator="<=",
                        threshold=1024.0,
                        sample_value=443.0,
                        tree_node_index=0,
                    ),
                ),
                feature_names_used=("stats_summary.port",),
            ),
        ]
        self.score_records = [
            ReplayScoreRecord(
                run_id="prompt-run",
                timestamp="20260409T050607Z",
                split="eval",
                score_scope="graph",
                graph_id=10,
                window_id=100,
                node_id=None,
                edge_id=None,
                flow_id=None,
                anomaly_score=0.92,
                threshold=0.50,
                is_alert=True,
                label=1,
                metadata={"node_count": 6, "edge_count": 8},
            ),
            ReplayScoreRecord(
                run_id="prompt-run",
                timestamp="20260409T050607Z",
                split="eval",
                score_scope="flow",
                graph_id=10,
                window_id=100,
                node_id=None,
                edge_id=None,
                flow_id="flow-1",
                anomaly_score=0.88,
                threshold=0.50,
                is_alert=True,
                label=None,
                metadata={"pkt_count": 11, "byte_count": 4096},
            ),
            ReplayScoreRecord(
                run_id="prompt-run",
                timestamp="20260409T050607Z",
                split="eval",
                score_scope="flow",
                graph_id=10,
                window_id=100,
                node_id=None,
                edge_id=None,
                flow_id="flow-2",
                anomaly_score=0.41,
                threshold=0.50,
                is_alert=False,
                label=None,
                metadata={"pkt_count": 2, "byte_count": 256},
            ),
            ReplayScoreRecord(
                run_id="prompt-run",
                timestamp="20260409T050607Z",
                split="eval",
                score_scope="node",
                graph_id=10,
                window_id=100,
                node_id="node-1",
                edge_id=None,
                flow_id=None,
                anomaly_score=0.73,
                threshold=0.50,
                is_alert=True,
                label=None,
                metadata={"port": 443, "proto": 6},
            ),
        ]
        self.alert_records = [
            ReplayAlertRecord(
                alert_id="alert-graph",
                alert_level="high",
                alert_scope="graph",
                run_id="prompt-run",
                timestamp="20260409T050607Z",
                split="eval",
                graph_id=10,
                window_id=100,
                node_id=None,
                edge_id=None,
                flow_id=None,
                anomaly_score=0.92,
                threshold=0.50,
                is_alert=True,
                label=1,
                metadata={},
            ),
            ReplayAlertRecord(
                alert_id="alert-flow",
                alert_level="medium",
                alert_scope="flow",
                run_id="prompt-run",
                timestamp="20260409T050607Z",
                split="eval",
                graph_id=10,
                window_id=100,
                node_id=None,
                edge_id=None,
                flow_id="flow-1",
                anomaly_score=0.88,
                threshold=0.50,
                is_alert=True,
                label=None,
                metadata={},
            ),
            ReplayAlertRecord(
                alert_id="alert-node",
                alert_level="medium",
                alert_scope="node",
                run_id="prompt-run",
                timestamp="20260409T050607Z",
                split="eval",
                graph_id=10,
                window_id=100,
                node_id="node-1",
                edge_id=None,
                flow_id=None,
                anomaly_score=0.73,
                threshold=0.50,
                is_alert=True,
                label=None,
                metadata={},
            ),
        ]

    def _make_sample(self, **kwargs: object) -> ExplanationSample:
        """Build a lightweight explanation sample for prompt dataset tests."""

        return ExplanationSample(
            sample_id=str(kwargs["sample_id"]),
            scope=kwargs["scope"],  # type: ignore[arg-type]
            run_id="prompt-run",
            graph_id=kwargs["graph_id"],
            window_id=kwargs["window_id"],
            flow_id=kwargs.get("flow_id"),
            node_id=kwargs.get("node_id"),
            anomaly_score=float(kwargs["anomaly_score"]),
            threshold=kwargs.get("threshold"),
            is_alert=kwargs.get("is_alert"),
            alert_level=kwargs.get("alert_level"),
            label=kwargs.get("label"),
            stats_summary=dict(kwargs.get("stats_summary", {})),
            graph_summary=dict(kwargs.get("graph_summary", {})),
            feature_summary=dict(kwargs.get("feature_summary", {})),
        )

    def _make_rule(
        self,
        *,
        sample_id: str,
        scope: str,
        tree_mode: str,
        predicted_value: object,
        leaf_id: int,
        path_conditions: tuple[RulePathCondition, ...],
        feature_names_used: tuple[str, ...],
    ) -> RuleRecord:
        """Build a synthetic rule record for prompt dataset tests."""

        return RuleRecord(
            rule_id=f"{sample_id}:{tree_mode}:leaf-{leaf_id}",
            sample_id=sample_id,
            scope=scope,  # type: ignore[arg-type]
            tree_mode=tree_mode,  # type: ignore[arg-type]
            predicted_score_or_class=predicted_value,
            leaf_id=leaf_id,
            path_conditions=path_conditions,
            feature_names_used=feature_names_used,
        )

    def test_build_prompt_dataset_filters_by_scope_and_alerts(self) -> None:
        """Prompt datasets should honor scope, alert, and top-k selection rules."""

        artifact = build_prompt_dataset(
            self.samples,
            self.rules,
            scope="flow",
            only_alerts=True,
            top_k=1,
            alert_records=self.alert_records,
            score_records=self.score_records,
        )

        self.assertIsInstance(artifact, PromptDatasetArtifact)
        self.assertEqual(artifact.scope, "flow")
        self.assertEqual(artifact.selected_sample_count, 1)
        self.assertEqual(artifact.summary.total_count, 1)
        self.assertEqual(artifact.summary.alert_count, 1)
        self.assertEqual(len(artifact.prompt_inputs), 1)
        self.assertEqual(artifact.prompt_inputs[0].sample_id, "flow:10:100:flow-1")
        self.assertTrue(artifact.prompt_inputs[0].is_alert)
        self.assertIn("Prompt dataset:", summarize_prompt_dataset_text(artifact))

    def test_balanced_selection_keeps_alert_and_non_alert_samples(self) -> None:
        """Balanced selection should preserve both positive and negative examples."""

        artifact = build_prompt_dataset(
            self.samples,
            self.rules,
            scope="flow",
            balanced=True,
            max_samples=2,
            alert_records=self.alert_records,
            score_records=self.score_records,
        )

        self.assertEqual(artifact.selection_mode, "balanced")
        self.assertEqual(artifact.selected_sample_count, 2)
        self.assertEqual({prompt.is_alert for prompt in artifact.prompt_inputs}, {True, False})

    def test_build_prompt_dataset_supports_each_scope(self) -> None:
        """Prompt datasets should be buildable for graph, flow, and node scopes."""

        graph_artifact = build_prompt_dataset(
            self.samples,
            self.rules,
            scope="graph",
            alert_records=self.alert_records,
            score_records=self.score_records,
        )
        flow_artifact = build_prompt_dataset(
            self.samples,
            self.rules,
            scope="flow",
            alert_records=self.alert_records,
            score_records=self.score_records,
        )
        node_artifact = build_prompt_dataset(
            self.samples,
            self.rules,
            scope="node",
            alert_records=self.alert_records,
            score_records=self.score_records,
        )

        self.assertEqual([prompt.scope for prompt in graph_artifact.prompt_inputs], ["graph"])
        self.assertEqual([prompt.scope for prompt in flow_artifact.prompt_inputs], ["flow", "flow"])
        self.assertEqual([prompt.scope for prompt in node_artifact.prompt_inputs], ["node"])

    def test_jsonl_and_csv_exports_are_stable(self) -> None:
        """Exported prompt dataset files should keep a stable field order."""

        artifact = build_prompt_dataset(
            self.samples,
            self.rules,
            scope="flow",
            only_alerts=False,
            top_k=2,
            alert_records=self.alert_records,
            score_records=self.score_records,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            export_result = export_prompt_dataset(artifact, temp_dir)
            run_directory = Path(export_result.output_directory)
            jsonl_path = run_directory / "prompt_inputs.jsonl"
            csv_path = run_directory / "prompt_inputs.csv"
            manifest_path = Path(export_result.manifest_path)

            self.assertTrue(jsonl_path.exists())
            self.assertTrue(csv_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertEqual(export_result.row_counts.get("jsonl"), 2)
            self.assertEqual(export_result.row_counts.get("csv"), 2)

            jsonl_lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
            jsonl_payload = json.loads(jsonl_lines[0])
            self.assertEqual(list(jsonl_payload.keys()), list(PROMPT_INPUT_FIELDS))

            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader)
                csv_row = next(reader)
            self.assertEqual(header, list(PROMPT_INPUT_FIELDS))
            self.assertEqual(len(csv_row), len(PROMPT_INPUT_FIELDS))

            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest_payload["selected_sample_count"], 2)
            self.assertEqual(manifest_payload["summary"]["total_count"], 2)

    def test_cli_parser_exposes_prompt_dataset_flags(self) -> None:
        """The CLI parser should expose prompt-dataset generation switches."""

        parser = build_parser()
        args = parser.parse_args(
            [
                "--build-prompts",
                "--prompt-scope",
                "flow",
                "--prompt-top-k",
                "4",
                "--prompt-output-dir",
                "artifacts/prompts",
                "--prompt-only-alerts",
                "--prompt-balanced",
                "--prompt-max-samples",
                "12",
            ]
        )

        self.assertTrue(args.build_prompts)
        self.assertEqual(args.prompt_scope, "flow")
        self.assertEqual(args.prompt_top_k, 4)
        self.assertEqual(args.prompt_output_dir, "artifacts/prompts")
        self.assertTrue(args.prompt_only_alerts)
        self.assertTrue(args.prompt_balanced)
        self.assertEqual(args.prompt_max_samples, 12)


if __name__ == "__main__":
    unittest.main()
