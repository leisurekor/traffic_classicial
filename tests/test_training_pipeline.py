"""Smoke tests for graph autoencoder training and checkpointing."""

from __future__ import annotations

import csv
import io
import sys
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import torch  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    raise unittest.SkipTest("PyTorch is not installed in this environment.") from exc

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_ROOT: Path = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_graph.cli import main
from traffic_graph.config import (
    DataConfig,
    FeaturesConfig,
    GraphConfig,
    ModelConfig,
    OutputConfig,
    PipelineConfig,
    PipelineRuntimeConfig,
    PreprocessingConfig,
    TrainingConfig,
)
from traffic_graph.pipeline.checkpoint import load_checkpoint
from traffic_graph.pipeline.training_pipeline import TrainingPipeline


def _write_sample_csv(csv_path: Path) -> None:
    """Write a tiny CSV dataset with two windows of flow records."""

    rows = (
        {
            "flow_id": "flow-001",
            "src_ip": "10.0.0.1",
            "src_port": 11111,
            "dst_ip": "10.0.0.2",
            "dst_port": 80,
            "protocol": "tcp",
            "start_time": "2026-04-08T09:00:05",
            "end_time": "2026-04-08T09:00:10",
            "packet_count": 12,
            "byte_count": 4096,
            "direction": "outbound",
        },
        {
            "flow_id": "flow-002",
            "src_ip": "10.0.0.3",
            "src_port": 22222,
            "dst_ip": "10.0.0.4",
            "dst_port": 443,
            "protocol": "tcp",
            "start_time": "2026-04-08T09:00:20",
            "end_time": "2026-04-08T09:00:25",
            "packet_count": 11,
            "byte_count": 3500,
            "direction": "outbound",
        },
        {
            "flow_id": "flow-003",
            "src_ip": "10.0.0.5",
            "src_port": 33333,
            "dst_ip": "10.0.0.6",
            "dst_port": 53,
            "protocol": "udp",
            "start_time": "2026-04-08T09:01:05",
            "end_time": "2026-04-08T09:01:08",
            "packet_count": 9,
            "byte_count": 1500,
            "direction": "outbound",
        },
        {
            "flow_id": "flow-004",
            "src_ip": "10.0.0.7",
            "src_port": 44444,
            "dst_ip": "10.0.0.8",
            "dst_port": 22,
            "protocol": "tcp",
            "start_time": "2026-04-08T09:01:20",
            "end_time": "2026-04-08T09:01:26",
            "packet_count": 13,
            "byte_count": 5200,
            "direction": "outbound",
        },
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _build_config(temp_root: Path, csv_path: Path) -> PipelineConfig:
    """Create a compact pipeline config for training smoke tests."""

    return PipelineConfig(
        pipeline=PipelineRuntimeConfig(run_name="training-smoke", seed=7),
        data=DataConfig(input_path=str(csv_path), format="csv"),
        preprocessing=PreprocessingConfig(window_size=60),
        graph=GraphConfig(),
        features=FeaturesConfig(),
        model=ModelConfig(
            name="gae",
            device="cpu",
            hidden_dim=16,
            latent_dim=8,
            num_layers=2,
            dropout=0.0,
            use_edge_features=True,
            reconstruct_edge_features=True,
        ),
        training=TrainingConfig(
            epochs=2,
            learning_rate=0.001,
            weight_decay=0.0,
            batch_size=1,
            validation_split_ratio=0.5,
            early_stopping_patience=2,
            checkpoint_dir=str(temp_root / "checkpoints"),
            shuffle=True,
            seed=7,
            smoke_graph_limit=2,
        ),
        output=OutputConfig(directory=str(temp_root / "artifacts"), save_intermediate=False),
    )


def _write_yaml_config(config: PipelineConfig, config_path: Path) -> None:
    """Write a hand-crafted YAML config used by the CLI smoke test."""

    config_path.write_text(
        "\n".join(
            [
                "pipeline:",
                f"  run_name: {config.pipeline.run_name}",
                f"  seed: {config.pipeline.seed}",
                "data:",
                f"  input_path: {config.data.input_path}",
                f"  format: {config.data.format}",
                "preprocessing:",
                f"  window_size: {config.preprocessing.window_size}",
                "graph:",
                f"  time_window_seconds: {config.graph.time_window_seconds}",
                f"  directed: {str(config.graph.directed).lower()}",
                "  association_edges:",
                f"    enable_same_src_ip: {str(config.graph.association_edges.enable_same_src_ip).lower()}",
                f"    enable_same_dst_subnet: {str(config.graph.association_edges.enable_same_dst_subnet).lower()}",
                f"    dst_subnet_prefix: {config.graph.association_edges.dst_subnet_prefix}",
                "features:",
                "  normalization:",
                f"    enabled: {str(config.features.normalization.enabled).lower()}",
                f"    method: {config.features.normalization.method}",
                "    exclude_node_fields:",
                *[
                    f"      - {field}"
                    for field in config.features.normalization.exclude_node_fields
                ],
                "    exclude_edge_fields:",
                *[
                    f"      - {field}"
                    for field in config.features.normalization.exclude_edge_fields
                ],
                "model:",
                f"  name: {config.model.name}",
                f"  device: {config.model.device}",
                f"  hidden_dim: {config.model.hidden_dim}",
                f"  latent_dim: {config.model.latent_dim}",
                f"  num_layers: {config.model.num_layers}",
                f"  dropout: {config.model.dropout}",
                f"  use_edge_features: {str(config.model.use_edge_features).lower()}",
                f"  reconstruct_edge_features: {str(config.model.reconstruct_edge_features).lower()}",
                "training:",
                f"  epochs: {config.training.epochs}",
                f"  learning_rate: {config.training.learning_rate}",
                f"  weight_decay: {config.training.weight_decay}",
                f"  batch_size: {config.training.batch_size}",
                f"  validation_split_ratio: {config.training.validation_split_ratio}",
                f"  early_stopping_patience: {config.training.early_stopping_patience}",
                f"  checkpoint_dir: {config.training.checkpoint_dir}",
                f"  shuffle: {str(config.training.shuffle).lower()}",
                f"  seed: {config.training.seed}",
                f"  smoke_graph_limit: {config.training.smoke_graph_limit}",
                "output:",
                f"  directory: {config.output.directory}",
                f"  save_intermediate: {str(config.output.save_intermediate).lower()}",
            ]
        ),
        encoding="utf-8",
    )


class TrainingPipelineSmokeTest(unittest.TestCase):
    """Validate the training pipeline, checkpointing, and CLI training stage."""

    def test_training_pipeline_runs_and_checkpoint_loads(self) -> None:
        """A tiny two-window dataset should train and reload cleanly."""

        with TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            csv_path = temp_dir / "flows.csv"
            _write_sample_csv(csv_path)
            config = _build_config(temp_dir, csv_path)

            result = TrainingPipeline(config).run(smoke_run=True)
            loaded = load_checkpoint(result.best_checkpoint_path)

            self.assertGreaterEqual(len(result.training_history), 1)
            self.assertEqual(result.train_graph_count, 1)
            self.assertEqual(result.val_graph_count, 1)
            self.assertTrue(Path(result.best_checkpoint_path).exists())
            self.assertTrue(Path(result.latest_checkpoint_path).exists())
            self.assertTrue(Path(result.feature_preprocessor_path).exists())
            self.assertEqual(len(loaded.history), len(result.training_history))
            self.assertEqual(
                loaded.feature_preprocessor.node_field_names,
                result.node_feature_fields,
            )
            self.assertEqual(loaded.model.hidden_dim, config.model.hidden_dim)
            self.assertEqual(loaded.model.latent_dim, config.model.latent_dim)
            self.assertEqual(loaded.config.model.name, config.model.name)

    def test_cli_training_stage_prints_history_and_checkpoint(self) -> None:
        """The repository CLI should expose the training stage and print its summary."""

        with TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            csv_path = temp_dir / "flows.csv"
            config_path = temp_dir / "pipeline.yaml"
            _write_sample_csv(csv_path)
            config = _build_config(temp_dir, csv_path)
            _write_yaml_config(config, config_path)

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = main(
                    [
                        "--config",
                        str(config_path),
                        "--input",
                        str(csv_path),
                        "--train",
                        "--smoke-train",
                    ]
                )

            rendered = buffer.getvalue()

            self.assertEqual(exit_code, 0)
            self.assertIn("train_graph_autoencoder", rendered)
            self.assertIn("Training history:", rendered)
            self.assertIn("Best checkpoint:", rendered)
            self.assertIn("Train/val split:", rendered)


if __name__ == "__main__":
    unittest.main()
