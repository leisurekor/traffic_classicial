"""Smoke tests for checkpoint-based anomaly scoring and evaluation."""

from __future__ import annotations

import json
import csv
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local runtime
    raise unittest.SkipTest(
        "PyTorch is not installed in this environment."
    ) from exc

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_ROOT: Path = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_graph.config import PipelineConfig  # noqa: E402
from traffic_graph.pipeline.eval_pipeline import EvaluationPipeline  # noqa: E402
from traffic_graph.pipeline.training_pipeline import TrainingPipeline  # noqa: E402


class EvaluationPipelineSmokeTest(unittest.TestCase):
    """Run a tiny train/evaluate cycle on a synthetic labeled CSV dataset."""

    def test_training_checkpoint_can_be_scored_and_evaluated(self) -> None:
        """The pipeline should train, score, and export evaluation artifacts."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / "flows.csv"
            checkpoint_dir = root / "checkpoints"
            output_dir = root / "artifacts"
            with input_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "flow_id",
                        "src_ip",
                        "src_port",
                        "dst_ip",
                        "dst_port",
                        "protocol",
                        "start_time",
                        "end_time",
                        "packet_count",
                        "byte_count",
                        "metadata",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "flow_id": "flow-1",
                            "src_ip": "10.0.0.1",
                            "src_port": 1234,
                            "dst_ip": "10.0.0.2",
                            "dst_port": 80,
                            "protocol": "tcp",
                            "start_time": "2026-04-08T09:00:00",
                            "end_time": "2026-04-08T09:00:02",
                            "packet_count": 2,
                            "byte_count": 512,
                            "metadata": json.dumps({"label": 0}),
                        },
                        {
                            "flow_id": "flow-2",
                            "src_ip": "10.0.0.1",
                            "src_port": 1235,
                            "dst_ip": "10.0.0.2",
                            "dst_port": 80,
                            "protocol": "tcp",
                            "start_time": "2026-04-08T09:00:04",
                            "end_time": "2026-04-08T09:00:06",
                            "packet_count": 3,
                            "byte_count": 768,
                            "metadata": json.dumps({"label": 1}),
                        },
                        {
                            "flow_id": "flow-3",
                            "src_ip": "10.0.0.3",
                            "src_port": 2345,
                            "dst_ip": "10.0.0.4",
                            "dst_port": 443,
                            "protocol": "tcp",
                            "start_time": "2026-04-08T09:01:00",
                            "end_time": "2026-04-08T09:01:03",
                            "packet_count": 4,
                            "byte_count": 1024,
                            "metadata": json.dumps({"label": 0}),
                        },
                        {
                            "flow_id": "flow-4",
                            "src_ip": "10.0.0.3",
                            "src_port": 2346,
                            "dst_ip": "10.0.0.4",
                            "dst_port": 443,
                            "protocol": "tcp",
                            "start_time": "2026-04-08T09:01:05",
                            "end_time": "2026-04-08T09:01:08",
                            "packet_count": 5,
                            "byte_count": 2048,
                            "metadata": json.dumps({"label": 1}),
                        },
                    ]
                )

            config = PipelineConfig.from_mapping(
                {
                    "data": {
                        "input_path": str(input_path),
                        "format": "csv",
                    },
                    "preprocessing": {
                        "window_size": 60,
                        "short_flow_thresholds": {
                            "packet_count_lt": 5,
                            "byte_count_lt": 1024,
                        },
                    },
                    "graph": {
                        "time_window_seconds": 60,
                        "directed": True,
                        "association_edges": {
                            "enable_same_src_ip": True,
                            "enable_same_dst_subnet": True,
                        },
                    },
                    "model": {
                        "name": "gae",
                        "device": "cpu",
                        "hidden_dim": 8,
                        "latent_dim": 4,
                        "num_layers": 1,
                        "dropout": 0.0,
                        "use_edge_features": True,
                        "reconstruct_edge_features": True,
                    },
                    "training": {
                        "epochs": 1,
                        "learning_rate": 0.01,
                        "weight_decay": 0.0,
                        "batch_size": 1,
                        "validation_split_ratio": 0.5,
                        "early_stopping_patience": 1,
                        "checkpoint_dir": str(checkpoint_dir),
                        "shuffle": True,
                        "seed": 7,
                        "smoke_graph_limit": 4,
                    },
                    "evaluation": {
                        "score_reduction": "mean",
                        "anomaly_threshold": 0.5,
                        "evaluation_label_field": "label",
                        "checkpoint_dir": str(checkpoint_dir),
                        "checkpoint_tag": "best",
                    },
                    "output": {
                        "directory": str(output_dir),
                        "save_intermediate": False,
                    },
                }
            )

            training_result = TrainingPipeline(config).run(smoke_run=True)
            self.assertTrue(training_result.best_checkpoint_path)
            self.assertTrue(Path(training_result.best_checkpoint_path).exists())

            evaluation_result = EvaluationPipeline(config).run(
                checkpoint_path=training_result.best_checkpoint_path
            )
            self.assertGreater(len(evaluation_result.graph_scores), 0)
            self.assertGreater(len(evaluation_result.flow_scores), 0)
            self.assertGreaterEqual(evaluation_result.graph_metrics.support, 1)
            self.assertTrue(Path(evaluation_result.artifact_paths["metrics_json"]).exists())
            self.assertTrue(Path(evaluation_result.artifact_paths["graph_scores_csv"]).exists())


if __name__ == "__main__":
    unittest.main()
