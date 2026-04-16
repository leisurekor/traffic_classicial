"""Tests for merged CSV binary detection experiment reporting."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest import TestCase

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from traffic_graph.cli import build_parser, main
from traffic_graph.data import BinaryExperimentConfig, HeldOutAttackProtocolConfig
from traffic_graph.pipeline.binary_detection import (
    _compute_binary_metrics,
    _quantile_summary,
    run_binary_detection_experiment,
)


def _make_merged_frame() -> pd.DataFrame:
    """Build a small merged-style frame for deterministic smoke tests."""

    rows: list[dict[str, object]] = []
    for index in range(18):
        rows.append(
            {
                "FlowID": f"b-{index}",
                "FeatureA": float(index),
                "FeatureB": float(index + 2),
                "FeatureC": float(index % 3),
                "Label": "BENIGN",
            }
        )
    attack_specs = {
        "RECON-HOSTDISCOVERY": 4,
        "DDOS-ICMP_FLOOD": 4,
        "MIRAI-GREETH_FLOOD": 4,
        "XSS": 3,
        "SQLINJECTION": 3,
        "COMMANDINJECTION": 3,
        "UPLOADING_ATTACK": 3,
        "BROWSERHIJACKING": 3,
    }
    offset = 100
    for label, count in attack_specs.items():
        for index in range(count):
            rows.append(
                {
                    "FlowID": f"{label}-{index}",
                    "FeatureA": float(offset + index),
                    "FeatureB": float(offset + index + 2),
                    "FeatureC": float((offset + index) % 5),
                    "Label": label,
                }
            )
        offset += 50
    return pd.DataFrame(rows)


class BinaryDetectionExperimentTests(TestCase):
    """Smoke tests for the binary detection experiment runner."""

    def test_metric_helpers_are_correct_on_simple_arrays(self) -> None:
        """Binary metrics and quantiles should match a trivial separable example."""

        metrics = _compute_binary_metrics(
            np.asarray([0, 0, 1, 1], dtype=int),
            np.asarray([0.1, 0.2, 0.9, 0.8], dtype=float),
            threshold=0.5,
        )
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["pr_auc"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["false_positive_rate"], 0.0)

        summary = _quantile_summary([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["max"], 4.0)
        self.assertAlmostEqual(float(summary["median"]), 2.5)

    def test_binary_detection_pipeline_runs_and_exports(self) -> None:
        """The binary detection runner should fit, score, and export a report."""

        frame = _make_merged_frame()
        with tempfile.TemporaryDirectory() as tmpdir:
            report, export_result = run_binary_detection_experiment(
                frame,
                tmpdir,
                binary_experiment_config=BinaryExperimentConfig(
                    label_column="Label",
                    train_ratio=0.5,
                    val_ratio=0.25,
                    test_ratio=0.25,
                    train_normal_only=True,
                    split_mode="stratified",
                    random_seed=13,
                ),
                heldout_protocol_config=HeldOutAttackProtocolConfig(
                    label_column="Label",
                    held_out_attack_types=("Recon", "DDoS", "Mirai", "Web-based"),
                    min_samples_per_attack=1,
                    random_seed=13,
                    benign_train_ratio=0.5,
                ),
                threshold_percentile=80.0,
                max_components=2,
                random_seed=13,
            )

            self.assertIn("roc_auc", report.overall_metrics)
            self.assertGreater(len(report.per_attack_metrics), 0)
            self.assertIn("count", report.train_score_summary)
            self.assertGreater(len(report.attack_score_summaries), 0)
            self.assertTrue(Path(export_result.manifest_path).exists())
            self.assertTrue(Path(export_result.metrics_summary_path).exists())
            self.assertTrue(Path(export_result.per_attack_metrics_path).exists())
            self.assertTrue(Path(export_result.overall_scores_path).exists())
            self.assertTrue(Path(export_result.attack_scores_path).exists())
            self.assertIn("Overall metrics:", report.render())
            self.assertIn("Per-attack metrics:", report.render())

    def test_cli_parser_exposes_binary_detection_flags(self) -> None:
        """The CLI should expose the binary detection experiment flags."""

        parser = build_parser()
        args = parser.parse_args(
            [
                "--run-binary-detection-experiment",
                "--binary-detection-input",
                "Merged01.csv",
                "--binary-detection-output-dir",
                "artifacts/out",
                "--binary-detection-threshold-percentile",
                "90",
            ]
        )
        self.assertTrue(args.run_binary_detection_experiment)
        self.assertEqual(args.binary_detection_input, "Merged01.csv")
        self.assertEqual(args.binary_detection_output_dir, "artifacts/out")
        self.assertEqual(args.binary_detection_threshold_percentile, 90.0)

    def test_cli_can_run_binary_detection_experiment_from_csv(self) -> None:
        """The CLI should run the binary detection experiment from a CSV file."""

        frame = _make_merged_frame()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "Merged01.csv"
            output_dir = Path(tmpdir) / "binary-detection"
            frame.to_csv(input_path, index=False)

            exit_code = main(
                [
                    "--run-binary-detection-experiment",
                    "--binary-detection-input",
                    str(input_path),
                    "--binary-detection-output-dir",
                    str(output_dir),
                    "--heldout-attack-types",
                    "Recon",
                    "DDoS",
                    "Mirai",
                    "Web-based",
                    "--heldout-min-samples-per-attack",
                    "1",
                    "--binary-detection-threshold-percentile",
                    "80",
                    "--binary-detection-max-components",
                    "2",
                    "--binary-detection-random-seed",
                    "13",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(any(output_dir.iterdir()))
            self.assertTrue((output_dir / "Merged01").exists())

    def test_real_merged_csv_smoke(self) -> None:
        """A small sample from the verified merged CSV should run end to end."""

        merged_path = Path("artifacts/cic_iot2023/Merged01.csv")
        if not merged_path.exists():
            self.skipTest("Merged01.csv is not available in the workspace.")

        frame = pd.read_csv(merged_path, nrows=5000)
        with tempfile.TemporaryDirectory() as tmpdir:
            report, export_result = run_binary_detection_experiment(
                frame,
                tmpdir,
                binary_experiment_config=BinaryExperimentConfig(
                    train_ratio=0.5,
                    val_ratio=0.25,
                    test_ratio=0.25,
                    train_normal_only=True,
                    split_mode="stratified",
                    random_seed=7,
                ),
                heldout_protocol_config=HeldOutAttackProtocolConfig(
                    held_out_attack_types=("Recon", "DDoS", "Mirai", "Web-based"),
                    min_samples_per_attack=1,
                    random_seed=7,
                    benign_train_ratio=0.5,
                ),
                threshold_percentile=95.0,
                max_components=10,
                random_seed=7,
            )

            self.assertGreaterEqual(len(report.per_attack_metrics), 1)
            self.assertGreaterEqual(report.train_sample_count, 1)
            self.assertTrue(Path(export_result.manifest_path).exists())
