"""Focused tests for checkpoint JSON shape validation."""

from __future__ import annotations

import json
import sys
import unittest
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

from tests.test_training_pipeline import _build_config, _write_sample_csv
from traffic_graph.pipeline.checkpoint import load_checkpoint
from traffic_graph.pipeline.training_pipeline import TrainingPipeline


class CheckpointIoTests(unittest.TestCase):
    """Validate checkpoint file-shape expectations for history and metadata."""

    def _build_checkpoint(self, temp_dir: Path) -> Path:
        """Train a tiny model and return the best checkpoint path."""

        csv_path = temp_dir / "flows.csv"
        _write_sample_csv(csv_path)
        config = _build_config(temp_dir, csv_path)
        result = TrainingPipeline(config).run(smoke_run=True)
        return Path(result.best_checkpoint_path)

    def test_history_json_list_loads_successfully(self) -> None:
        """A list-backed history.json should load without shape errors."""

        with TemporaryDirectory() as temp_dir_name:
            checkpoint_dir = self._build_checkpoint(Path(temp_dir_name))
            loaded = load_checkpoint(checkpoint_dir)

            self.assertIsInstance(loaded.history, list)
            self.assertGreaterEqual(len(loaded.history), 1)
            self.assertTrue(all(isinstance(entry, dict) for entry in loaded.history))

    def test_history_json_type_error_is_clear(self) -> None:
        """A non-list history.json should raise a list-specific error."""

        with TemporaryDirectory() as temp_dir_name:
            checkpoint_dir = self._build_checkpoint(Path(temp_dir_name))
            history_path = checkpoint_dir / "history.json"
            history_path.write_text(
                json.dumps({"epoch": 1}, indent=2) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "must be a JSON list"):
                load_checkpoint(checkpoint_dir)

    def test_dict_only_checkpoint_files_remain_strict(self) -> None:
        """Metadata-like files should still reject non-object JSON payloads."""

        with TemporaryDirectory() as temp_dir_name:
            checkpoint_dir = self._build_checkpoint(Path(temp_dir_name))
            metadata_path = checkpoint_dir / "metadata.json"
            metadata_path.write_text(json.dumps([], indent=2) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must contain a JSON object"):
                load_checkpoint(checkpoint_dir)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
