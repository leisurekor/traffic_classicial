"""Tests for held-out attack evaluation task construction."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest import mock

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from traffic_graph.cli import build_parser, main
from traffic_graph.data import (
    DEFAULT_HELD_OUT_ATTACK_TYPES,
    HeldOutAttackProtocolConfig,
    export_heldout_attack_protocol,
    prepare_heldout_attack_protocol,
    summarize_heldout_attack_protocol_text,
)


def _make_heldout_frame() -> pd.DataFrame:
    """Build a synthetic merged-style frame with multiple attack families."""

    rows: list[dict[str, object]] = []
    for index in range(20):
        rows.append(
            {
                "FlowID": f"b-{index}",
                "FeatureA": float(index),
                "FeatureB": float(index + 1),
                "Label": "BENIGN",
            }
        )
    attack_specs = {
        "RECON-HOSTDISCOVERY": 4,
        "RECON-OSSCAN": 3,
        "DDOS-ICMP_FLOOD": 6,
        "MIRAI-GREETH_FLOOD": 5,
        "XSS": 2,
        "SQLINJECTION": 2,
        "COMMANDINJECTION": 2,
        "UPLOADING_ATTACK": 2,
        "BROWSERHIJACKING": 2,
    }
    offset = 100
    for label, count in attack_specs.items():
        for index in range(count):
            rows.append(
                {
                    "FlowID": f"{label}-{index}",
                    "FeatureA": float(offset + index),
                    "FeatureB": float(offset + index + 1),
                    "Label": label,
                }
            )
        offset += 100
    return pd.DataFrame(rows)


class HeldOutAttackProtocolTests(TestCase):
    """Smoke tests for the held-out attack protocol builder and exporter."""

    def test_protocol_builds_expected_family_tasks(self) -> None:
        """Attack-family aliases should expand to deterministic task names."""

        artifact = prepare_heldout_attack_protocol(
            _make_heldout_frame(),
            HeldOutAttackProtocolConfig(
                held_out_attack_types=("Recon", "DDoS", "Mirai", "Web-based"),
                min_samples_per_attack=1,
                benign_train_ratio=0.5,
                random_seed=11,
            ),
        )

        task_names = {task.task_name for task in artifact.task_artifacts}
        self.assertTrue({"recon", "ddos", "mirai", "web-based", "all_malicious"}.issubset(task_names))

        recon_task = next(task for task in artifact.task_artifacts if task.task_name == "recon")
        self.assertEqual(recon_task.requested_attack_type, "Recon")
        self.assertTrue(all(label.startswith("RECON-") for label in recon_task.attack_labels))
        self.assertTrue((recon_task.train_frame["binary_label"] == 0).all())
        self.assertEqual(set(recon_task.test_frame["binary_label"].unique().tolist()), {0, 1})
        self.assertIn("recon", artifact.summary.task_summaries)

    def test_min_samples_filters_out_small_attack_tasks(self) -> None:
        """Tasks below the sample threshold should be skipped cleanly."""

        artifact = prepare_heldout_attack_protocol(
            _make_heldout_frame(),
            HeldOutAttackProtocolConfig(
                held_out_attack_types=("Recon",),
                min_samples_per_attack=20,
                benign_train_ratio=0.5,
                random_seed=11,
            ),
        )

        task_names = {task.task_name for task in artifact.task_artifacts}
        self.assertNotIn("recon", task_names)
        self.assertIn("all_malicious", task_names)
        self.assertTrue(any("Skipped held-out task" in note for note in artifact.notes))

    def test_export_writes_task_manifests_and_summaries(self) -> None:
        """Export should write a root manifest and per-task task manifests."""

        artifact = prepare_heldout_attack_protocol(
            _make_heldout_frame(),
            HeldOutAttackProtocolConfig(
                held_out_attack_types=("Recon", "DDoS"),
                min_samples_per_attack=1,
                benign_train_ratio=0.5,
                random_seed=7,
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch(
                "traffic_graph.data.binary_experiment._is_parquet_available",
                return_value=False,
            ):
                export_result = export_heldout_attack_protocol(
                    artifact,
                    tmpdir,
                    formats=("csv", "parquet"),
                )

            output_dir = Path(export_result.output_directory)
            self.assertTrue((output_dir / "manifest.json").exists())
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "label_mapping.json").exists())
            self.assertTrue((output_dir / "clean.csv").exists())
            self.assertTrue((output_dir / "tasks" / "recon" / "manifest.json").exists())
            self.assertTrue((output_dir / "tasks" / "ddos" / "manifest.json").exists())
            self.assertTrue((output_dir / "tasks" / "all_malicious" / "manifest.json").exists())
            self.assertTrue(
                any("parquet support is unavailable" in note for note in export_result.notes)
            )
            self.assertIn("recon", export_result.task_manifest_paths)

    def test_cli_parser_exposes_heldout_flags(self) -> None:
        """The CLI parser should expose the new held-out protocol flags."""

        parser = build_parser()
        args = parser.parse_args(
            [
                "--build-heldout-tasks",
                "--heldout-input",
                "Merged01.csv",
                "--heldout-output-dir",
                "artifacts/out",
                "--heldout-attack-types",
                "Recon",
                "DDoS",
            ]
        )
        self.assertTrue(args.build_heldout_tasks)
        self.assertEqual(args.heldout_input, "Merged01.csv")
        self.assertEqual(args.heldout_output_dir, "artifacts/out")
        self.assertEqual(args.heldout_attack_types, ["Recon", "DDoS"])

    def test_cli_can_build_heldout_protocol_from_csv(self) -> None:
        """The CLI should build held-out tasks from a CSV file."""

        frame = _make_heldout_frame()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "Merged01.csv"
            output_dir = Path(tmpdir) / "heldout-output"
            frame.to_csv(input_path, index=False)

            exit_code = main(
                [
                    "--build-heldout-tasks",
                    "--heldout-input",
                    str(input_path),
                    "--heldout-output-dir",
                    str(output_dir),
                    "--heldout-attack-types",
                    "Recon",
                    "DDoS",
                    "Mirai",
                    "Web-based",
                    "--heldout-min-samples-per-attack",
                    "1",
                    "--heldout-benign-train-ratio",
                    "0.5",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(any(output_dir.iterdir()))
            self.assertTrue((output_dir / "Merged01").exists())

    def test_real_merged_csv_smoke(self) -> None:
        """The verified merged CSV should build a non-empty held-out protocol."""

        merged_path = Path("artifacts/cic_iot2023/Merged01.csv")
        if not merged_path.exists():
            self.skipTest("Merged01.csv is not available in the workspace.")

        frame = pd.read_csv(merged_path)
        artifact = prepare_heldout_attack_protocol(
            frame,
            HeldOutAttackProtocolConfig(
                held_out_attack_types=DEFAULT_HELD_OUT_ATTACK_TYPES,
                min_samples_per_attack=10,
                benign_train_ratio=0.7,
                random_seed=5,
            ),
        )

        self.assertGreater(len(artifact.task_artifacts), 0)
        self.assertIn("all_malicious", {task.task_name for task in artifact.task_artifacts})
        self.assertTrue((artifact.benign_train_frame["binary_label"] == 0).all())
        self.assertIn("Held-out attack protocol:", summarize_heldout_attack_protocol_text(artifact))
