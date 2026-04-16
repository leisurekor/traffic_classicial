"""Tests for LLM-ready prompt input construction and export."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_ROOT: Path = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_graph.explain import (
    ExplanationSample,
    PROMPT_ALERT_SUMMARY_FIELDS,
    PROMPT_BASIC_INFO_FIELDS,
    PROMPT_CONTEXT_FIELDS,
    PROMPT_INPUT_FIELDS,
    PROMPT_RULE_SUMMARY_FIELDS,
    PROMPT_SCORE_SUMMARY_FIELDS,
    build_prompt_input,
    build_prompt_inputs,
    build_prompt_text,
    export_prompt_inputs,
)
from traffic_graph.explain.rule_records import RULE_PATH_CONDITION_FIELDS, RulePathCondition, RuleRecord
from traffic_graph.pipeline.replay_types import ReplayAlertRecord, ReplayScoreRecord


class PromptBuilderTests(unittest.TestCase):
    """Exercise prompt input construction across graph, flow, and node scopes."""

    def setUp(self) -> None:
        """Prepare small deterministic explanation artifacts for prompt tests."""

        self.samples = [
            self._make_sample(
                sample_id="graph:10:100:none",
                scope="graph",
                graph_id=10,
                window_id=100,
                anomaly_score=0.95,
                threshold=0.50,
                is_alert=True,
                alert_level="high",
                label=1,
                stats_summary={"edge_count": 4, "node_count": 5},
                graph_summary={"communication_edge_count": 2, "association_edge_count": 2},
                feature_summary={"available": False, "field_count": 0},
            ),
            self._make_sample(
                sample_id="flow:10:100:flow-1",
                scope="flow",
                graph_id=10,
                window_id=100,
                flow_id="flow-1",
                anomaly_score=0.77,
                threshold=0.50,
                is_alert=True,
                alert_level="medium",
                label=None,
                stats_summary={
                    "byte_count": 2048,
                    "duration": 0.40,
                    "flow_count": 3,
                    "pkt_count": 8,
                    "proto": 6,
                },
                graph_summary={"communication_edge_count": 2, "node_count": 5},
                feature_summary={"available": False, "field_count": 0},
            ),
            self._make_sample(
                sample_id="node:10:100:node-1",
                scope="node",
                graph_id=10,
                window_id=100,
                node_id="node-1",
                anomaly_score=0.33,
                threshold=0.50,
                is_alert=False,
                alert_level=None,
                label=None,
                stats_summary={
                    "communication_edge_count": 2,
                    "endpoint_type": 1,
                    "port": 443,
                    "proto": 6,
                    "total_pkt_count": 10,
                },
                graph_summary={"edge_count": 4, "node_count": 5},
                feature_summary={"available": True, "field_count": 3},
            ),
        ]
        self.rules = [
            RuleRecord(
                rule_id="graph:10:100:none:regression:leaf-3",
                sample_id="graph:10:100:none",
                scope="graph",
                tree_mode="regression",
                predicted_score_or_class=0.95,
                leaf_id=3,
                path_conditions=(
                    RulePathCondition(
                        feature_name="stats_summary.node_count",
                        operator=">",
                        threshold=4.5,
                        sample_value=5.0,
                        tree_node_index=0,
                    ),
                ),
                feature_names_used=("stats_summary.node_count",),
            ),
            RuleRecord(
                rule_id="flow:10:100:flow-1:regression:leaf-7",
                sample_id="flow:10:100:flow-1",
                scope="flow",
                tree_mode="regression",
                predicted_score_or_class=0.77,
                leaf_id=7,
                path_conditions=(
                    RulePathCondition(
                        feature_name="stats_summary.pkt_count",
                        operator="<=",
                        threshold=10.0,
                        sample_value=8.0,
                        tree_node_index=0,
                    ),
                    RulePathCondition(
                        feature_name="stats_summary.proto",
                        operator=">",
                        threshold=4.5,
                        sample_value=6.0,
                        tree_node_index=2,
                    ),
                ),
                feature_names_used=("stats_summary.pkt_count", "stats_summary.proto"),
            ),
            RuleRecord(
                rule_id="node:10:100:node-1:classification:leaf-2",
                sample_id="node:10:100:node-1",
                scope="node",
                tree_mode="classification",
                predicted_score_or_class=1,
                leaf_id=2,
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
                anomaly_score=0.95,
                threshold=0.50,
                is_alert=True,
                label=1,
                metadata={"node_count": 5, "edge_count": 4},
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
                anomaly_score=0.77,
                threshold=0.50,
                is_alert=True,
                label=None,
                metadata={"pkt_count": 8, "byte_count": 2048},
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
                anomaly_score=0.33,
                threshold=0.50,
                is_alert=False,
                label=None,
                metadata={"port": 443, "total_pkt_count": 10},
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
                anomaly_score=0.95,
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
                anomaly_score=0.77,
                threshold=0.50,
                is_alert=True,
                label=None,
                metadata={},
            ),
            ReplayAlertRecord(
                alert_id="alert-node",
                alert_level="low",
                alert_scope="node",
                run_id="prompt-run",
                timestamp="20260409T050607Z",
                split="eval",
                graph_id=10,
                window_id=100,
                node_id="node-1",
                edge_id=None,
                flow_id=None,
                anomaly_score=0.33,
                threshold=0.50,
                is_alert=False,
                label=None,
                metadata={},
            ),
        ]

    def _make_sample(self, **kwargs: object) -> ExplanationSample:
        """Build a lightweight explanation sample for prompt-input tests."""

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

    def test_prompt_input_field_order_and_text_are_stable(self) -> None:
        """Prompt inputs should preserve field order and deterministic text."""

        prompt = build_prompt_input(
            self.samples[1],
            self.rules[1],
            alert_record=self.alert_records[1],
            score_record=self.score_records[1],
        )
        prompt_again = build_prompt_input(
            self.samples[1],
            self.rules[1],
            alert_record=self.alert_records[1],
            score_record=self.score_records[1],
        )

        self.assertEqual(prompt.prompt_text, prompt_again.prompt_text)
        self.assertEqual(list(prompt.to_dict().keys()), list(PROMPT_INPUT_FIELDS))
        self.assertEqual(
            list(prompt.structured_context.keys()),
            list(PROMPT_CONTEXT_FIELDS),
        )
        self.assertEqual(
            list(prompt.structured_context["sample_basic_info"].keys()),
            list(PROMPT_BASIC_INFO_FIELDS),
        )
        self.assertEqual(
            list(prompt.structured_context["score_summary"].keys()),
            list(PROMPT_SCORE_SUMMARY_FIELDS),
        )
        self.assertEqual(
            list(prompt.structured_context["alert_summary"].keys()),
            list(PROMPT_ALERT_SUMMARY_FIELDS),
        )
        self.assertEqual(
            list(prompt.rule_summary.keys()),
            list(PROMPT_RULE_SUMMARY_FIELDS),
        )
        self.assertEqual(
            list(prompt.rule_summary["path_conditions"][0].keys()),
            list(RULE_PATH_CONDITION_FIELDS),
        )
        self.assertIn("Structured context (JSON):", prompt.prompt_text)
        self.assertIn("Rule summary (JSON):", prompt.prompt_text)
        self.assertIn("label: n/a (reference only; do not use as evidence)", prompt.prompt_text)
        self.assertIn("Response format:", prompt.prompt_text)

    def test_batch_build_handles_graph_flow_and_node_scopes(self) -> None:
        """Batch construction should preserve ordering across all supported scopes."""

        prompt_inputs = build_prompt_inputs(
            self.samples,
            self.rules,
            alert_records=self.alert_records,
            score_records=self.score_records,
        )

        self.assertEqual([prompt.scope for prompt in prompt_inputs], ["graph", "flow", "node"])
        self.assertEqual(prompt_inputs[0].structured_context["sample_basic_info"]["flow_id"], None)
        self.assertEqual(prompt_inputs[0].structured_context["sample_basic_info"]["node_id"], None)
        self.assertEqual(prompt_inputs[1].structured_context["sample_basic_info"]["flow_id"], "flow-1")
        self.assertEqual(prompt_inputs[2].structured_context["sample_basic_info"]["node_id"], "node-1")
        self.assertEqual(prompt_inputs[2].label, None)
        self.assertIn("label: n/a", prompt_inputs[2].prompt_text)

    def test_prompt_text_can_be_rendered_directly_from_components(self) -> None:
        """The low-level prompt text builder should remain deterministic."""

        sample = self.samples[0]
        rule = self.rules[0]
        prompt_text = build_prompt_text(
            prompt_id="prompt:graph:10:100:none:regression:leaf-3",
            run_id=sample.run_id,
            sample_id=sample.sample_id,
            scope=sample.scope,
            anomaly_score=sample.anomaly_score,
            threshold=sample.threshold,
            is_alert=sample.is_alert,
            alert_level=sample.alert_level,
            label=sample.label,
            structured_context=build_prompt_input(
                sample,
                rule,
                alert_record=self.alert_records[0],
                score_record=self.score_records[0],
            ).structured_context,
            rule_summary=build_prompt_input(
                sample,
                rule,
                alert_record=self.alert_records[0],
                score_record=self.score_records[0],
            ).rule_summary,
        )

        self.assertIn("Explain why the sample is considered anomalous.", prompt_text)
        self.assertIn("graph", prompt_text)
        self.assertIn("prompt:graph:10:100:none:regression:leaf-3", prompt_text)

    def test_prompt_inputs_can_be_exported_as_jsonl(self) -> None:
        """Prompt inputs should export to JSONL with stable top-level field order."""

        prompt_inputs = build_prompt_inputs(
            self.samples,
            self.rules,
            alert_records=self.alert_records,
            score_records=self.score_records,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "prompt_inputs.jsonl"
            saved_path = export_prompt_inputs(prompt_inputs, output_path)
            lines = Path(saved_path).read_text(encoding="utf-8").strip().splitlines()
            payload = json.loads(lines[1])

            self.assertEqual(list(payload.keys()), list(PROMPT_INPUT_FIELDS))
            self.assertEqual(
                list(payload["structured_context"].keys()),
                list(PROMPT_CONTEXT_FIELDS),
            )
            self.assertEqual(
                list(payload["rule_summary"].keys()),
                list(PROMPT_RULE_SUMMARY_FIELDS),
            )
            self.assertEqual(len(lines), 3)

if __name__ == "__main__":
    unittest.main()
