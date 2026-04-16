"""Tests for merged CSV binary experiment input construction."""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path
from unittest import TestCase
from unittest import mock

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from traffic_graph.cli import build_parser, main
from traffic_graph.data import (
    BinaryExperimentConfig,
    export_binary_experiment,
    prepare_binary_experiment,
    summarize_binary_experiment_text,
)


def _make_binary_frame(*, benign_count: int, malicious_count: int) -> pd.DataFrame:
    """Build a small synthetic merged-style frame for smoke tests."""

    rows: list[dict[str, object]] = []
    for index in range(benign_count):
        rows.append(
            {
                "FlowID": f"b-{index}",
                "FeatureA": float(index),
                "FeatureB": float(index + 1),
                "Label": "BENIGN",
            }
        )
    for index in range(malicious_count):
        rows.append(
            {
                "FlowID": f"m-{index}",
                "FeatureA": float(index + benign_count),
                "FeatureB": float(index + benign_count + 1),
                "Label": "DoS",
            }
        )
    return pd.DataFrame(rows)


class BinaryExperimentTests(TestCase):
    """Smoke tests for the binary experiment builder and exporter."""

    def test_label_mapping_and_cleaning(self) -> None:
        """Label detection, binary mapping, and inf/NaN cleanup should work."""

        frame = pd.DataFrame(
            {
                "FlowID": ["a", "b", "c", "d"],
                "FeatureA": [1.0, np.inf, -np.inf, np.nan],
                "FeatureB": [5.0, 6.0, 7.0, 8.0],
                "Label": ["BENIGN", "DoS", "BENIGN", "Bot"],
            }
        )
        artifact = prepare_binary_experiment(
            frame,
            BinaryExperimentConfig(label_column=None, train_normal_only=False, split_mode="random"),
        )

        self.assertEqual(artifact.label_column, "Label")
        self.assertEqual(artifact.summary.binary_label_counts, {"0": 2, "1": 2})
        self.assertEqual(artifact.clean_frame["binary_label"].tolist(), [0, 1, 0, 1])
        clean_features = artifact.clean_frame.drop(columns=["Label", "binary_label"])
        self.assertFalse(np.isinf(clean_features.select_dtypes(include=[np.number]).to_numpy()).any())
        self.assertFalse(pd.isna(clean_features.to_numpy()).any())

    def test_train_normal_only_and_stratified_splits(self) -> None:
        """Training should stay benign-only when requested."""

        frame = _make_binary_frame(benign_count=20, malicious_count=20)
        artifact = prepare_binary_experiment(
            frame,
            BinaryExperimentConfig(
                label_column="Label",
                train_ratio=0.5,
                val_ratio=0.25,
                test_ratio=0.25,
                train_normal_only=True,
                split_mode="stratified",
                random_seed=7,
            ),
        )

        self.assertTrue((artifact.train_frame["binary_label"] == 0).all())
        self.assertEqual(set(artifact.val_frame["binary_label"].unique().tolist()), {0, 1})
        self.assertEqual(set(artifact.test_frame["binary_label"].unique().tolist()), {0, 1})
        self.assertTrue(
            any(
                "Training split is restricted to benign samples only" in note
                for note in artifact.summary.notes
            )
        )

    def test_split_is_reproducible(self) -> None:
        """Repeated preparation with the same seed should be stable."""

        frame = _make_binary_frame(benign_count=12, malicious_count=12)
        config = BinaryExperimentConfig(
            label_column="Label",
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
            train_normal_only=False,
            split_mode="stratified",
            random_seed=13,
        )
        first = prepare_binary_experiment(frame, config)
        second = prepare_binary_experiment(frame, config)

        pd.testing.assert_frame_equal(first.train_frame, second.train_frame)
        pd.testing.assert_frame_equal(first.val_frame, second.val_frame)
        pd.testing.assert_frame_equal(first.test_frame, second.test_frame)

    def test_export_and_parquet_fallback(self) -> None:
        """CSV export should work and parquet failures should degrade gracefully."""

        frame = _make_binary_frame(benign_count=10, malicious_count=10)
        artifact = prepare_binary_experiment(
            frame,
            BinaryExperimentConfig(
                label_column="Label",
                train_ratio=0.5,
                val_ratio=0.25,
                test_ratio=0.25,
                train_normal_only=True,
                split_mode="stratified",
                random_seed=21,
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch(
                "traffic_graph.data.binary_experiment._is_parquet_available",
                return_value=False,
            ):
                export_result = export_binary_experiment(
                    artifact,
                    tmpdir,
                    formats=("csv", "parquet"),
                )

            output_dir = Path(export_result.output_directory)
            self.assertTrue((output_dir / "manifest.json").exists())
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "label_mapping.json").exists())
            self.assertTrue((output_dir / "clean.csv").exists())
            self.assertTrue((output_dir / "train.csv").exists())
            self.assertTrue((output_dir / "val.csv").exists())
            self.assertTrue((output_dir / "test.csv").exists())
            self.assertTrue(
                any("parquet support is unavailable" in note for note in export_result.notes)
            )

    def test_cli_end_to_end_prepare_binary_experiment(self) -> None:
        """The CLI should prepare a binary experiment from a CSV file."""

        frame = _make_binary_frame(benign_count=8, malicious_count=8)
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "Merged01.csv"
            output_dir = Path(tmpdir) / "binary-output"
            frame.to_csv(input_path, index=False)

            exit_code = main(
                [
                    "--prepare-binary-experiment",
                    "--binary-input",
                    str(input_path),
                    "--binary-output-dir",
                    str(output_dir),
                    "--binary-split-mode",
                    "stratified",
                    "--binary-train-ratio",
                    "0.5",
                    "--binary-val-ratio",
                    "0.25",
                    "--binary-test-ratio",
                    "0.25",
                    "--train-normal-only",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(any(output_dir.iterdir()))
            self.assertTrue((output_dir / "Merged01").exists())

    def test_real_merged_csv_smoke(self) -> None:
        """A small slice of the verified merged CSV should still prepare cleanly."""

        merged_path = Path("artifacts/cic_iot2023/Merged01.csv")
        if not merged_path.exists():
            self.skipTest("Merged01.csv is not available in the workspace.")

        frame = pd.read_csv(merged_path, nrows=3000)
        artifact = prepare_binary_experiment(
            frame,
            BinaryExperimentConfig(
                train_normal_only=True,
                split_mode="stratified",
                random_seed=5,
            ),
        )

        self.assertGreater(len(artifact.clean_frame), 0)
        self.assertIn("Label", artifact.clean_frame.columns)
        self.assertTrue(set(artifact.clean_frame["binary_label"].unique()).issubset({0, 1}))
        self.assertIn("Binary experiment:", summarize_binary_experiment_text(artifact))


class BinaryExperimentCliParserTests(TestCase):
    """Parser wiring tests for the binary experiment CLI."""

    def test_parser_exposes_binary_flags(self) -> None:
        """The CLI parser should expose the new binary experiment flags."""

        parser = build_parser()
        args = parser.parse_args(
            [
                "--prepare-binary-experiment",
                "--binary-input",
                "Merged01.csv",
                "--binary-output-dir",
                "artifacts/out",
                "--binary-split-mode",
                "stratified",
            ]
        )
        self.assertTrue(args.prepare_binary_experiment)
        self.assertEqual(args.binary_input, "Merged01.csv")
        self.assertEqual(args.binary_output_dir, "artifacts/out")
        self.assertEqual(args.binary_split_mode, "stratified")
