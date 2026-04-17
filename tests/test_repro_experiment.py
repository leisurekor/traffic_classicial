"""Tests for the repeatable CSV/PCAP wrapper entrypoints."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from traffic_graph.data import BinaryExperimentConfig, prepare_binary_experiment
from traffic_graph.pipeline.repro_experiment import (
    ReproExperimentConfig,
    run_csv_repro_experiment,
    run_pcap_repro_experiment,
)


def _make_csv_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(8):
        rows.append({"FlowID": f"b-{index}", "FeatureA": float(index), "FeatureB": float(index + 1), "Label": "BENIGN"})
    for index in range(4):
        rows.append({"FlowID": f"a-{index}", "FeatureA": float(index + 100), "FeatureB": float(index + 101), "Label": "MALICIOUS"})
    return pd.DataFrame(rows)


class ReproExperimentTests(TestCase):
    def test_explicit_binary_label_mapping_is_applied(self) -> None:
        frame = pd.DataFrame(
            [
                {"FeatureA": 1.0, "Label": "Normal Traffic"},
                {"FeatureA": 9.0, "Label": "Botnet Attack"},
            ]
        )
        artifact = prepare_binary_experiment(
            frame,
            BinaryExperimentConfig(
                label_column="Label",
                label_mapping={"Normal Traffic": 0, "Botnet Attack": 1},
            ),
        )
        self.assertEqual(artifact.summary.binary_label_counts["0"], 1)
        self.assertEqual(artifact.summary.binary_label_counts["1"], 1)

    def test_csv_wrapper_exports_metrics_logs_and_figure(self) -> None:
        frame = _make_csv_frame()
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "flows.csv"
            frame.to_csv(input_path, index=False)
            config = ReproExperimentConfig(
                dataset_name="csv-smoke",
                input_mode="csv",
                input_path=input_path.as_posix(),
                binary_label_mapping={"BENIGN": 0},
                output_dir=(Path(temp_dir) / "outputs").as_posix(),
                threshold_percentile=80.0,
                max_components=2,
            )
            result = run_csv_repro_experiment(config)
            self.assertTrue(Path(result["metrics_path"]).exists())
            self.assertTrue(Path(result["figure_path"]).exists())
            self.assertTrue(Path(result["log_path"]).exists())
            payload = json.loads(Path(result["metrics_path"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["config"]["use_nuisance_aware"], False)
            self.assertIn("report", payload)

    def test_pcap_wrapper_creates_standard_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_directory = Path(temp_dir) / "run"
            checkpoints_dir = run_directory / "checkpoints" / "best"
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            (checkpoints_dir / "model.pt").write_text("stub", encoding="utf-8")
            fake_result = SimpleNamespace(
                summary={
                    "total_packets": 100,
                    "total_flows": 10,
                    "total_graphs": 4,
                    "benign_graph_count": 2,
                    "malicious_graph_count": 2,
                    "overall_metrics": {"roc_auc": 0.9, "pr_auc": 0.8, "precision": 0.7, "recall": 0.6, "f1": 0.65},
                },
                backend="fallback",
                notes=["stub note"],
                export_result=SimpleNamespace(run_directory=run_directory.as_posix(), artifact_paths={}, row_counts={}),
            )
            config = ReproExperimentConfig(
                dataset_name="pcap-smoke",
                input_mode="pcap",
                malicious_inputs=["/tmp/fake.pcap"],
                output_dir=(Path(temp_dir) / "outputs").as_posix(),
            )
            with mock.patch("traffic_graph.pipeline.repro_experiment.run_pcap_graph_experiment", return_value=fake_result), mock.patch(
                "traffic_graph.pipeline.repro_experiment.summarize_pcap_graph_experiment_result",
                return_value="summary",
            ):
                result = run_pcap_repro_experiment(config)
            self.assertTrue(Path(result["metrics_path"]).exists())
            self.assertTrue(Path(result["figure_path"]).exists())
            self.assertTrue(Path(result["log_path"]).exists())
            self.assertTrue((Path(config.output_dir) / "checkpoints").exists())
