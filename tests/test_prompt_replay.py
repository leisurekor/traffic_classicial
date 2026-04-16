"""Tests for replaying exported prompt datasets."""

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

from traffic_graph.cli import build_parser
from traffic_graph.explain import (
    PROMPT_INPUT_FIELDS,
    PromptDatasetReplay,
    build_prompt_dataset,
    export_prompt_dataset,
    filter_prompt_records,
    get_prompt_record,
    list_prompt_records,
    load_prompt_dataset,
    summarize_prompt_dataset_replay,
)
from traffic_graph.explain.explanation_types import ExplanationSample
from traffic_graph.explain.rule_records import RulePathCondition, RuleRecord
from traffic_graph.pipeline.replay_types import ReplayAlertRecord, ReplayScoreRecord


class PromptReplayTests(unittest.TestCase):
    """Exercise prompt dataset replay, filtering, and lookup helpers."""

    def setUp(self) -> None:
        """Prepare a deterministic prompt dataset bundle for replay tests."""

        self.samples = [
            self._make_sample(
                sample_id="flow:10:100:flow-1",
                scope="flow",
                graph_id=10,
                window_id=100,
                flow_id="flow-1",
                anomaly_score=0.90,
                threshold=0.50,
                is_alert=True,
                alert_level="high",
                stats_summary={"pkt_count": 12, "byte_count": 4096, "duration": 0.42},
                graph_summary={"node_count": 6, "edge_count": 8},
                feature_summary={"available": True, "field_count": 4},
            ),
            self._make_sample(
                sample_id="flow:10:100:flow-2",
                scope="flow",
                graph_id=10,
                window_id=100,
                flow_id="flow-2",
                anomaly_score=0.64,
                threshold=0.50,
                is_alert=True,
                alert_level="medium",
                stats_summary={"pkt_count": 7, "byte_count": 1536, "duration": 0.18},
                graph_summary={"node_count": 6, "edge_count": 8},
                feature_summary={"available": True, "field_count": 4},
            ),
            self._make_sample(
                sample_id="flow:10:100:flow-3",
                scope="flow",
                graph_id=10,
                window_id=100,
                flow_id="flow-3",
                anomaly_score=0.21,
                threshold=0.50,
                is_alert=False,
                alert_level=None,
                stats_summary={"pkt_count": 1, "byte_count": 64, "duration": 0.03},
                graph_summary={"node_count": 6, "edge_count": 8},
                feature_summary={"available": True, "field_count": 4},
            ),
        ]
        self.rules = [
            self._make_rule(
                sample_id="flow:10:100:flow-1",
                tree_mode="regression",
                predicted_value=0.90,
                leaf_id=4,
                threshold=4.5,
                sample_value=12.0,
            ),
            self._make_rule(
                sample_id="flow:10:100:flow-2",
                tree_mode="regression",
                predicted_value=0.64,
                leaf_id=5,
                threshold=4.5,
                sample_value=7.0,
            ),
            self._make_rule(
                sample_id="flow:10:100:flow-3",
                tree_mode="regression",
                predicted_value=0.21,
                leaf_id=2,
                threshold=4.5,
                sample_value=1.0,
            ),
        ]
        self.score_records = [
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
                anomaly_score=0.90,
                threshold=0.50,
                is_alert=True,
                label=None,
                metadata={"pkt_count": 12, "byte_count": 4096},
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
                anomaly_score=0.64,
                threshold=0.50,
                is_alert=True,
                label=None,
                metadata={"pkt_count": 7, "byte_count": 1536},
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
                flow_id="flow-3",
                anomaly_score=0.21,
                threshold=0.50,
                is_alert=False,
                label=None,
                metadata={"pkt_count": 1, "byte_count": 64},
            ),
        ]
        self.alert_records = [
            ReplayAlertRecord(
                alert_id="alert-1",
                alert_level="high",
                alert_scope="flow",
                run_id="prompt-run",
                timestamp="20260409T050607Z",
                split="eval",
                graph_id=10,
                window_id=100,
                node_id=None,
                edge_id=None,
                flow_id="flow-1",
                anomaly_score=0.90,
                threshold=0.50,
                is_alert=True,
                label=None,
                metadata={},
            ),
            ReplayAlertRecord(
                alert_id="alert-2",
                alert_level="medium",
                alert_scope="flow",
                run_id="prompt-run",
                timestamp="20260409T050607Z",
                split="eval",
                graph_id=10,
                window_id=100,
                node_id=None,
                edge_id=None,
                flow_id="flow-2",
                anomaly_score=0.64,
                threshold=0.50,
                is_alert=True,
                label=None,
                metadata={},
            ),
        ]

    def _make_sample(self, **kwargs: object) -> ExplanationSample:
        """Build a small explanation sample for replay tests."""

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
        tree_mode: str,
        predicted_value: object,
        leaf_id: int,
        threshold: float,
        sample_value: float,
    ) -> RuleRecord:
        """Build a synthetic rule record for replay tests."""

        return RuleRecord(
            rule_id=f"{sample_id}:{tree_mode}:leaf-{leaf_id}",
            sample_id=sample_id,
            scope="flow",
            tree_mode=tree_mode,  # type: ignore[arg-type]
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

    def _export_dataset(self, temp_dir: str) -> tuple[Path, Path]:
        """Build and export a prompt dataset bundle for replay tests."""

        artifact = build_prompt_dataset(
            self.samples,
            self.rules,
            scope="flow",
            alert_records=self.alert_records,
            score_records=self.score_records,
        )
        export_result = export_prompt_dataset(artifact, temp_dir)
        return Path(export_result.output_directory), Path(export_result.manifest_path)

    def test_load_prompt_dataset_from_run_directory_and_manifest(self) -> None:
        """Prompt datasets should load from both the run directory and manifest path."""

        with tempfile.TemporaryDirectory() as temp_dir:
            run_directory, manifest_path = self._export_dataset(temp_dir)
            replay_from_dir = load_prompt_dataset(run_directory)
            replay_from_manifest = load_prompt_dataset(manifest_path)

            self.assertIsInstance(replay_from_dir, PromptDatasetReplay)
            self.assertEqual(replay_from_dir.dataset_id, replay_from_manifest.dataset_id)
            self.assertEqual(replay_from_dir.selection_summary.total_count, 3)
            self.assertEqual(replay_from_dir.selection_summary.alert_count, 2)
            self.assertEqual(
                list(replay_from_dir.prompt_records[0].to_dict().keys()),
                list(PROMPT_INPUT_FIELDS),
            )
            self.assertEqual(
                summarize_prompt_dataset_replay(replay_from_dir).total_count,
                3,
            )
            self.assertEqual(list_prompt_records(replay_from_dir)[0].sample_id, "flow:10:100:flow-1")
            second_prompt_id = list_prompt_records(replay_from_dir)[1].prompt_id
            self.assertEqual(
                get_prompt_record(replay_from_dir, second_prompt_id).prompt_id,
                second_prompt_id,
            )

    def test_csv_fallback_and_filters_are_handled_cleanly(self) -> None:
        """Replay should fall back to CSV when JSONL is unavailable."""

        with tempfile.TemporaryDirectory() as temp_dir:
            run_directory, _ = self._export_dataset(temp_dir)
            jsonl_path = run_directory / "prompt_inputs.jsonl"
            jsonl_path.unlink()

            replay = load_prompt_dataset(run_directory)
            filtered_alerts = filter_prompt_records(
                replay,
                scope="flow",
                only_alerts=True,
                top_k=1,
            )
            filtered_negative = filter_prompt_records(
                replay,
                scope="flow",
                only_alerts=False,
            )

            self.assertEqual(replay.loaded_files.get("prompt_records_format"), "csv")
            self.assertIn("prompt_inputs.csv", replay.loaded_files.get("prompt_inputs", ""))
            self.assertEqual(len(filtered_alerts), 1)
            self.assertTrue(filtered_alerts[0].is_alert)
            self.assertEqual(len(filtered_negative), 1)
            self.assertFalse(filtered_negative[0].is_alert)
            self.assertEqual(
                filter_prompt_records(replay, scope="graph"),
                [],
            )

    def test_cli_parser_exposes_prompt_replay_flag(self) -> None:
        """The CLI parser should expose a prompt replay switch."""

        parser = build_parser()
        args = parser.parse_args(["--replay-prompts", "path/to/prompts"])

        self.assertEqual(args.replay_prompts, "path/to/prompts")


if __name__ == "__main__":
    unittest.main()
