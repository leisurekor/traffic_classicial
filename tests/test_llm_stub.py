"""Tests for the mock LLM result schema, stub runner, and export helpers."""

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
    LLM_RESULT_FIELDS,
    LLMResultArtifact,
    build_prompt_dataset,
    export_llm_results,
    export_prompt_dataset,
    load_prompt_dataset,
    run_llm_stub,
)
from traffic_graph.explain.explanation_types import ExplanationSample
from traffic_graph.explain.rule_records import RulePathCondition, RuleRecord
from traffic_graph.pipeline.replay_types import ReplayAlertRecord, ReplayScoreRecord


class LLMStubTests(unittest.TestCase):
    """Exercise the mock LLM runner and its persistence helpers."""

    def setUp(self) -> None:
        """Prepare a compact prompt dataset for stub execution tests."""

        self.samples = [
            self._make_sample(
                sample_id="flow:10:100:flow-1",
                scope="flow",
                graph_id=10,
                window_id=100,
                flow_id="flow-1",
                anomaly_score=0.93,
                threshold=0.50,
                is_alert=True,
                alert_level="high",
                stats_summary={"pkt_count": 14, "byte_count": 8192, "duration": 0.51},
                graph_summary={"node_count": 8, "edge_count": 11},
                feature_summary={"available": True, "field_count": 4},
            ),
            self._make_sample(
                sample_id="flow:10:100:flow-2",
                scope="flow",
                graph_id=10,
                window_id=100,
                flow_id="flow-2",
                anomaly_score=0.27,
                threshold=0.50,
                is_alert=False,
                alert_level=None,
                stats_summary={"pkt_count": 2, "byte_count": 128, "duration": 0.09},
                graph_summary={"node_count": 8, "edge_count": 11},
                feature_summary={"available": True, "field_count": 4},
            ),
        ]
        self.rules = [
            self._make_rule(
                sample_id="flow:10:100:flow-1",
                predicted_value=0.93,
                leaf_id=4,
                threshold=4.5,
                sample_value=14.0,
            ),
            self._make_rule(
                sample_id="flow:10:100:flow-2",
                predicted_value=0.27,
                leaf_id=2,
                threshold=4.5,
                sample_value=2.0,
            ),
        ]
        self.score_records = [
            ReplayScoreRecord(
                run_id="llm-run",
                timestamp="20260409T050607Z",
                split="eval",
                score_scope="flow",
                graph_id=10,
                window_id=100,
                node_id=None,
                edge_id=None,
                flow_id="flow-1",
                anomaly_score=0.93,
                threshold=0.50,
                is_alert=True,
                label=None,
                metadata={"pkt_count": 14, "byte_count": 8192},
            ),
            ReplayScoreRecord(
                run_id="llm-run",
                timestamp="20260409T050607Z",
                split="eval",
                score_scope="flow",
                graph_id=10,
                window_id=100,
                node_id=None,
                edge_id=None,
                flow_id="flow-2",
                anomaly_score=0.27,
                threshold=0.50,
                is_alert=False,
                label=None,
                metadata={"pkt_count": 2, "byte_count": 128},
            ),
        ]
        self.alert_records = [
            ReplayAlertRecord(
                alert_id="alert-flow-1",
                alert_level="high",
                alert_scope="flow",
                run_id="llm-run",
                timestamp="20260409T050607Z",
                split="eval",
                graph_id=10,
                window_id=100,
                node_id=None,
                edge_id=None,
                flow_id="flow-1",
                anomaly_score=0.93,
                threshold=0.50,
                is_alert=True,
                label=None,
                metadata={},
            ),
        ]

    def _make_sample(self, **kwargs: object) -> ExplanationSample:
        """Build a lightweight explanation sample."""

        return ExplanationSample(
            sample_id=str(kwargs["sample_id"]),
            scope=kwargs["scope"],  # type: ignore[arg-type]
            run_id="llm-run",
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
        predicted_value: object,
        leaf_id: int,
        threshold: float,
        sample_value: float,
    ) -> RuleRecord:
        """Build a synthetic surrogate-tree rule record."""

        return RuleRecord(
            rule_id=f"{sample_id}:regression:leaf-{leaf_id}",
            sample_id=sample_id,
            scope="flow",
            tree_mode="regression",  # type: ignore[arg-type]
            predicted_score_or_class=predicted_value,
            leaf_id=leaf_id,
            path_conditions=(
                RulePathCondition(
                    feature_name="stats_summary.pkt_count",
                    operator=">",
                    threshold=threshold,
                    sample_value=sample_value,
                    tree_node_index=0,
                ),
            ),
            feature_names_used=("stats_summary.pkt_count",),
        )

    def _build_prompt_replay(self, temp_dir: str):
        """Build, export, and reload a prompt dataset for stub tests."""

        artifact = build_prompt_dataset(
            self.samples,
            self.rules,
            scope="flow",
            alert_records=self.alert_records,
            score_records=self.score_records,
        )
        export_result = export_prompt_dataset(artifact, temp_dir)
        return load_prompt_dataset(export_result.output_directory)

    def test_stub_runner_generates_results_with_stable_schema(self) -> None:
        """The stub runner should emit one result per prompt with stable fields."""

        with tempfile.TemporaryDirectory() as temp_dir:
            prompt_dataset = self._build_prompt_replay(temp_dir)
            results = run_llm_stub(
                prompt_dataset,
                model_name="mock-llm-unit-test",
                created_at="20260409T050607Z",
            )

            self.assertIsInstance(results, LLMResultArtifact)
            self.assertEqual(results.result_count, len(prompt_dataset.prompt_records))
            self.assertEqual(results.summary.total_count, len(prompt_dataset.prompt_records))
            self.assertEqual(results.summary.model_name, "mock-llm-unit-test")
            self.assertEqual(results.result_records[0].prompt_id, prompt_dataset.prompt_records[0].prompt_id)
            self.assertEqual(
                list(results.result_records[0].to_dict().keys()),
                list(LLM_RESULT_FIELDS),
            )
            self.assertEqual(results.result_records[0].status, "success")
            self.assertTrue(results.result_records[0].raw_response["stub"])
            self.assertIn(prompt_dataset.prompt_records[0].prompt_id, results.result_records[0].response_text)
            self.assertIn(prompt_dataset.prompt_records[0].run_id, results.result_records[0].response_text)
            self.assertIn("Prompt excerpt:", results.result_records[0].response_text)

    def test_export_llm_results_writes_expected_files(self) -> None:
        """Exported LLM results should write JSONL, CSV, manifest, and summary files."""

        with tempfile.TemporaryDirectory() as temp_dir:
            prompt_dataset = self._build_prompt_replay(temp_dir)
            results = run_llm_stub(
                prompt_dataset,
                model_name="mock-llm-unit-test",
                created_at="20260409T050607Z",
            )
            export_result = export_llm_results(results, temp_dir)
            run_directory = Path(export_result.output_directory)
            jsonl_path = run_directory / "results.jsonl"
            csv_path = run_directory / "results.csv"
            summary_path = Path(export_result.summary_path)
            manifest_path = Path(export_result.manifest_path)

            self.assertTrue(jsonl_path.exists())
            self.assertTrue(csv_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertEqual(export_result.row_counts.get("jsonl"), results.result_count)
            self.assertEqual(export_result.row_counts.get("csv"), results.result_count)

            jsonl_payload = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(list(jsonl_payload.keys()), list(LLM_RESULT_FIELDS))

            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader)
                row = next(reader)
            self.assertEqual(header, list(LLM_RESULT_FIELDS))
            self.assertEqual(len(row), len(LLM_RESULT_FIELDS))

            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest_payload["result_count"], results.result_count)
            self.assertEqual(
                manifest_payload["source_prompt_dataset"]["dataset_id"],
                prompt_dataset.dataset_id,
            )
            self.assertEqual(manifest_payload["summary"]["total_count"], results.result_count)

    def test_cli_parser_exposes_llm_stub_flags(self) -> None:
        """The CLI parser should expose the mock LLM execution switches."""

        parser = build_parser()
        args = parser.parse_args(
            [
                "--replay-prompts",
                "path/to/prompts",
                "--run-llm-stub",
                "--llm-model-name",
                "mock-llm-unit-test",
                "--llm-output-dir",
                "artifacts/llm",
            ]
        )

        self.assertEqual(args.replay_prompts, "path/to/prompts")
        self.assertTrue(args.run_llm_stub)
        self.assertEqual(args.llm_model_name, "mock-llm-unit-test")
        self.assertEqual(args.llm_output_dir, "artifacts/llm")


if __name__ == "__main__":
    unittest.main()
