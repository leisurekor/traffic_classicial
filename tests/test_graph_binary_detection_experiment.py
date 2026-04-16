"""Smoke tests for the graph-backed merged CSV binary detection experiment."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from typing import Final

import pandas as pd
import sys

ROOT: Final = Path(__file__).resolve().parents[1]
SRC: Final = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from traffic_graph.cli import build_parser, main
from traffic_graph.data import HeldOutAttackProtocolConfig
from traffic_graph.pipeline.graph_binary_detection import (
    GraphModeScoreInput,
    build_graph_mode_score_inputs,
    compute_graph_mode_binary_scores,
    export_per_attack_metrics,
    run_graph_binary_detection_experiment,
    reduce_graph_scores_to_flow_or_sample_level,
    summarize_graph_score_distribution,
)


def _build_toy_merged_csv() -> pd.DataFrame:
    """Create a tiny merged-style dataset with benign and held-out attack labels."""

    rows: list[dict[str, object]] = []
    for index in range(8):
        rows.append(
            {
                "FeatureA": float(index),
                "FeatureB": float(index % 3),
                "FeatureC": float(index * 2),
                "Label": "BENIGN",
            }
        )
    for index in range(6):
        rows.append(
            {
                "FeatureA": float(100 + index),
                "FeatureB": float(50 + index),
                "FeatureC": float(25 + index),
                "Label": "RECON-HOSTDISCOVERY",
            }
        )
    for index in range(6):
        rows.append(
            {
                "FeatureA": float(200 + index),
                "FeatureB": float(75 + index),
                "FeatureC": float(30 + index),
                "Label": "DDOS-ICMP_FLOOD",
            }
        )
    for index in range(6):
        rows.append(
            {
                "FeatureA": float(300 + index),
                "FeatureB": float(90 + index),
                "FeatureC": float(40 + index),
                "Label": "MIRAI-GREETH_FLOOD",
            }
        )
    return pd.DataFrame(rows)


class GraphBinaryDetectionExperimentTests(unittest.TestCase):
    """Graph binary detection smoke tests."""

    def test_cli_parser_exposes_graph_model_mode(self) -> None:
        """The CLI parser should expose the graph backend switch."""

        parser = build_parser()
        model_mode_action = next(
            action
            for action in parser._actions  # type: ignore[attr-defined]
            if action.dest == "binary_detection_model_mode"
        )
        self.assertIsInstance(model_mode_action, argparse._StoreAction)
        self.assertEqual(tuple(model_mode_action.choices or ()), ("tabular", "graph"))

    def test_graph_score_reduction_prefers_flow_then_edge_then_node_then_graph(self) -> None:
        """Graph scores should collapse with a stable priority order."""

        score_inputs = (
            GraphModeScoreInput(
                sample_id="sample-flow",
                task_name="task",
                split="test",
                row_index=0,
                raw_label="BENIGN",
                binary_label=0,
                attack_group="BENIGN",
                graph_score=9.0,
                node_scores=(4.0, 6.0),
                edge_scores=(3.0, 5.0),
                flow_scores=(1.0, 3.0),
            ),
            GraphModeScoreInput(
                sample_id="sample-edge",
                task_name="task",
                split="test",
                row_index=1,
                raw_label="RECON-HOSTDISCOVERY",
                binary_label=1,
                attack_group="RECON-HOSTDISCOVERY",
                graph_score=8.0,
                node_scores=(4.0, 6.0),
                edge_scores=(2.0, 4.0),
                flow_scores=(),
            ),
            GraphModeScoreInput(
                sample_id="sample-node",
                task_name="task",
                split="test",
                row_index=2,
                raw_label="RECON-HOSTDISCOVERY",
                binary_label=1,
                attack_group="RECON-HOSTDISCOVERY",
                graph_score=7.0,
                node_scores=(2.0, 4.0),
                edge_scores=(),
                flow_scores=(),
            ),
            GraphModeScoreInput(
                sample_id="sample-graph",
                task_name="task",
                split="test",
                row_index=3,
                raw_label="BENIGN",
                binary_label=0,
                attack_group="BENIGN",
                graph_score=0.75,
            ),
        )

        reduced = reduce_graph_scores_to_flow_or_sample_level(score_inputs)
        self.assertEqual(reduced[0].reduction_source, "flow")
        self.assertAlmostEqual(reduced[0].anomaly_score, 2.0)
        self.assertEqual(reduced[1].reduction_source, "edge")
        self.assertAlmostEqual(reduced[1].anomaly_score, 3.0)
        self.assertEqual(reduced[2].reduction_source, "node")
        self.assertAlmostEqual(reduced[2].anomaly_score, 3.0)
        self.assertEqual(reduced[3].reduction_source, "graph")
        self.assertAlmostEqual(reduced[3].anomaly_score, 0.75)

    def test_graph_mode_report_bundle_keeps_per_attack_metrics_schema(self) -> None:
        """Graph-mode score bundles should produce tabular-compatible metrics."""

        train_frame = pd.DataFrame(
            {
                "Label": ["BENIGN", "BENIGN", "BENIGN", "BENIGN"],
                "binary_label": [0, 0, 0, 0],
            }
        )
        overall_frame = pd.DataFrame(
            {
                "Label": ["BENIGN", "RECON-HOSTDISCOVERY", "BENIGN", "XSS"],
                "binary_label": [0, 1, 0, 1],
            }
        )
        recon_task_frame = pd.DataFrame(
            {
                "Label": ["BENIGN", "RECON-HOSTDISCOVERY", "BENIGN"],
                "binary_label": [0, 1, 0],
            }
        )
        web_task_frame = pd.DataFrame(
            {
                "Label": ["BENIGN", "XSS", "BENIGN"],
                "binary_label": [0, 1, 0],
            }
        )
        train_inputs = build_graph_mode_score_inputs(
            train_frame,
            [0.1, 0.2, 0.15, 0.25],
            task_name="train",
            split="train",
            label_column="Label",
        )
        overall_inputs = build_graph_mode_score_inputs(
            overall_frame,
            [0.1, 0.95, 0.2, 0.85],
            task_name="overall",
            split="test",
            label_column="Label",
        )
        recon_inputs = build_graph_mode_score_inputs(
            recon_task_frame,
            [0.05, 0.9, 0.1],
            task_name="recon",
            split="test",
            label_column="Label",
        )
        web_inputs = build_graph_mode_score_inputs(
            web_task_frame,
            [0.05, 0.8, 0.1],
            task_name="web-based",
            split="test",
            label_column="Label",
        )
        bundle = compute_graph_mode_binary_scores(
            run_id="graph-run-1",
            timestamp="20260410T000000Z",
            threshold=0.5,
            feature_count=2,
            train_score_inputs=train_inputs,
            overall_score_inputs=overall_inputs,
            task_score_inputs=(
                ("recon", "Recon", ("RECON-HOSTDISCOVERY",), recon_inputs),
                ("web-based", "Web-based", ("XSS",), web_inputs),
            ),
        )

        self.assertEqual(len(bundle.per_attack_metrics), 2)
        self.assertEqual(len(bundle.overall_score_records), len(overall_inputs))
        self.assertEqual(len(bundle.attack_score_records), len(recon_inputs) + len(web_inputs))
        self.assertIn("recon", bundle.attack_score_summaries)
        self.assertIn("web-based", bundle.attack_score_summaries)
        self.assertGreaterEqual(bundle.overall_metrics["roc_auc"] or 0.0, 0.0)
        self.assertEqual(bundle.train_reduced_scores[0].reduction_source, "graph")
        self.assertEqual(bundle.overall_reduced_scores[1].raw_label, "RECON-HOSTDISCOVERY")

        summary = summarize_graph_score_distribution(
            type(
                "DummyReport",
                (),
                {
                    "train_score_summary": bundle.train_score_summary,
                    "overall_score_summary": bundle.overall_score_summary,
                    "per_attack_metrics": bundle.per_attack_metrics,
                    "attack_score_summaries": bundle.attack_score_summaries,
                },
            )()
        )
        self.assertIn("Graph score distribution", summary)
        self.assertIn("recon", summary)
        self.assertIn("web-based", summary)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "per_attack_metrics.csv"
            saved_path = export_per_attack_metrics(bundle.per_attack_metrics, output_path)
            self.assertEqual(saved_path, output_path)
            exported = pd.read_csv(output_path)
            self.assertEqual(
                list(exported.columns),
                [
                    "task_name",
                    "requested_attack_type",
                    "attack_labels",
                    "sample_count",
                    "benign_count",
                    "attack_count",
                    "roc_auc",
                    "pr_auc",
                    "precision",
                    "recall",
                    "f1",
                    "false_positive_rate",
                    "threshold",
                    "score_min",
                    "score_q25",
                    "score_median",
                    "score_q75",
                    "score_q95",
                    "score_max",
                    "score_mean",
                    "score_std",
                    "benign_score_mean",
                    "benign_score_median",
                    "attack_score_mean",
                    "attack_score_median",
                    "notes",
                ],
            )

    def test_graph_mode_smoke_from_csv(self) -> None:
        """Graph mode should run end to end when PyTorch is available."""

        if find_spec("torch") is None:
            self.skipTest("PyTorch is not available in the current environment.")

        toy_frame = _build_toy_merged_csv()
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / "Merged01_toy.csv"
            toy_frame.to_csv(source_path, index=False)
            report, export_result = run_graph_binary_detection_experiment(
                source_path,
                Path(tmp_dir) / "graph-report",
                heldout_protocol_config=HeldOutAttackProtocolConfig(
                    held_out_attack_types=("Recon", "DDoS", "Mirai", "Web-based"),
                    min_samples_per_attack=2,
                    benign_train_ratio=0.7,
                    random_seed=7,
                ),
            )
            self.assertEqual(report.feature_columns, ("FeatureA", "FeatureB", "FeatureC"))
            self.assertEqual(report.input_artifacts["model_mode"], "graph")
            self.assertTrue(Path(export_result.manifest_path).exists())
            self.assertTrue(Path(export_result.metrics_summary_path).exists())
            self.assertTrue(Path(export_result.per_attack_metrics_path).exists())
            self.assertTrue(Path(export_result.overall_scores_path).exists())
            self.assertTrue(Path(export_result.attack_scores_path).exists())

    def test_cli_can_run_graph_mode(self) -> None:
        """The CLI should accept graph mode and produce a report bundle."""

        if find_spec("torch") is None:
            self.skipTest("PyTorch is not available in the current environment.")

        toy_frame = _build_toy_merged_csv()
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / "Merged01_toy.csv"
            toy_frame.to_csv(source_path, index=False)
            exit_code = main(
                [
                    "--run-binary-detection-experiment",
                    "--binary-detection-input",
                    source_path.as_posix(),
                    "--binary-detection-output-dir",
                    (Path(tmp_dir) / "graph-report").as_posix(),
                    "--binary-detection-model-mode",
                    "graph",
                    "--heldout-min-samples-per-attack",
                    "2",
                    "--binary-detection-threshold-percentile",
                    "80",
                ]
            )
            self.assertEqual(exit_code, 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
