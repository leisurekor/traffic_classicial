"""Checkpoint helpers for unsupervised graph autoencoder training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence

import torch

from traffic_graph.config import PipelineConfig
from traffic_graph.features.normalization import FeaturePreprocessor
from traffic_graph.models import GraphAutoEncoder, GraphAutoEncoderConfig
from traffic_graph.models.losses import ReconstructionLossWeights


@dataclass(frozen=True, slots=True)
class CheckpointFiles:
    """Paths that make up one checkpoint directory."""

    checkpoint_dir: Path
    state_path: Path
    config_path: Path
    preprocessor_path: Path
    history_path: Path
    metadata_path: Path


@dataclass(slots=True)
class LoadedCheckpoint:
    """Structured checkpoint contents returned by :func:`load_checkpoint`."""

    checkpoint_dir: Path
    model: GraphAutoEncoder
    config: PipelineConfig
    feature_preprocessor: FeaturePreprocessor
    optimizer_state_dict: dict[str, object]
    history: list[dict[str, float | int]]
    metadata: dict[str, object]


def _checkpoint_files(checkpoint_dir: Path) -> CheckpointFiles:
    """Return the canonical file layout for a checkpoint directory."""

    return CheckpointFiles(
        checkpoint_dir=checkpoint_dir,
        state_path=checkpoint_dir / "state.pt",
        config_path=checkpoint_dir / "config.json",
        preprocessor_path=checkpoint_dir / "preprocessor.json",
        history_path=checkpoint_dir / "history.json",
        metadata_path=checkpoint_dir / "metadata.json",
    )


def _write_json(path: Path, payload: object) -> None:
    """Write a JSON payload with a stable, human-readable formatting."""

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _read_json_value(path: Path) -> object:
    """Read any JSON value from disk without imposing a top-level shape."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_json(path: Path) -> dict[str, object]:
    """Read a JSON object from disk."""

    payload = _read_json_value(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint file {path} must contain a JSON object.")
    return payload


def _history_to_json(
    history: Sequence[Mapping[str, float | int]],
) -> list[dict[str, float | int]]:
    """Normalize history entries into JSON-friendly dictionaries."""

    return [dict(entry) for entry in history]


def save_checkpoint(
    checkpoint_dir: str | Path,
    *,
    tag: str,
    model: GraphAutoEncoder,
    optimizer: torch.optim.Optimizer,
    config: PipelineConfig,
    feature_preprocessor: FeaturePreprocessor,
    history: Sequence[Mapping[str, float | int]],
    epoch: int,
    best_epoch: int,
    best_val_loss: float,
) -> Path:
    """Save a complete training checkpoint into a tag-named subdirectory."""

    root_dir = Path(checkpoint_dir)
    target_dir = root_dir / tag
    target_dir.mkdir(parents=True, exist_ok=True)
    files = _checkpoint_files(target_dir)

    model_config = GraphAutoEncoderConfig(
        hidden_dim=model.hidden_dim,
        latent_dim=model.latent_dim,
        num_layers=model.num_layers,
        dropout=model.dropout,
        use_edge_features=model.use_edge_features,
        reconstruct_edge_features=model.reconstruct_edge_features,
        use_temporal_edge_projector=model.use_temporal_edge_projector,
        temporal_edge_hidden_dim=model.temporal_edge_hidden_dim,
        temporal_edge_field_names=model.temporal_edge_field_names,
        use_edge_categorical_embeddings=model.use_edge_categorical_embeddings,
        edge_categorical_embedding_dim=model.edge_categorical_embedding_dim,
        edge_categorical_bucket_size=model.edge_categorical_bucket_size,
    )
    state_payload = {
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "node_input_dim": model.node_input_dim,
        "edge_input_dim": model.edge_input_dim,
        "model_config": model_config.to_dict(),
        "loss_weights": model.loss_weights.to_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state_payload, files.state_path)

    config_payload = asdict(config)
    preprocessor_payload = feature_preprocessor.to_dict()
    history_payload = _history_to_json(history)
    metadata_payload = {
        "tag": tag,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "state_path": files.state_path.name,
        "config_path": files.config_path.name,
        "preprocessor_path": files.preprocessor_path.name,
        "history_path": files.history_path.name,
        "node_input_dim": model.node_input_dim,
        "edge_input_dim": model.edge_input_dim,
        "model_name": config.model.name,
    }

    _write_json(files.config_path, config_payload)
    _write_json(files.preprocessor_path, preprocessor_payload)
    _write_json(files.history_path, history_payload)
    _write_json(files.metadata_path, metadata_payload)
    return target_dir


def load_checkpoint(
    checkpoint_dir: str | Path,
    *,
    map_location: str | torch.device | None = "cpu",
) -> LoadedCheckpoint:
    """Load a checkpoint directory and restore the model, config, and preprocessor."""

    directory = Path(checkpoint_dir)
    files = _checkpoint_files(directory)
    metadata = _read_json(files.metadata_path)
    config = PipelineConfig.from_mapping(_read_json(files.config_path))
    preprocessor = FeaturePreprocessor.from_dict(_read_json(files.preprocessor_path))
    history_payload = _read_json_value(files.history_path)
    if not isinstance(history_payload, list):
        raise ValueError(
            f"Checkpoint history in {files.history_path} must be a JSON list."
        )
    history: list[dict[str, float | int]] = []
    for entry in history_payload:
        if not isinstance(entry, dict):
            raise ValueError("Checkpoint history entries must be JSON objects.")
        history.append(entry)

    state = torch.load(files.state_path, map_location=map_location)
    if not isinstance(state, dict):
        raise ValueError("Checkpoint state must be a dictionary.")

    model_config_payload = state.get("model_config", {})
    if not isinstance(model_config_payload, dict):
        raise ValueError("Checkpoint model_config must be a dictionary.")
    model_config = GraphAutoEncoderConfig.from_mapping(model_config_payload)
    loss_weights_payload = state.get("loss_weights", {})
    if not isinstance(loss_weights_payload, dict):
        raise ValueError("Checkpoint loss_weights must be a dictionary.")
    loss_weights = ReconstructionLossWeights.from_mapping(loss_weights_payload)

    model = GraphAutoEncoder(
        node_input_dim=int(state["node_input_dim"]),
        edge_input_dim=int(state["edge_input_dim"]),
        config=model_config,
        loss_weights=loss_weights,
    )
    model.load_state_dict(state["model_state_dict"])
    if map_location is not None:
        model.to(map_location)

    optimizer_state_dict = state.get("optimizer_state_dict", {})
    if not isinstance(optimizer_state_dict, dict):
        raise ValueError("Checkpoint optimizer_state_dict must be a dictionary.")

    return LoadedCheckpoint(
        checkpoint_dir=directory,
        model=model,
        config=config,
        feature_preprocessor=preprocessor,
        optimizer_state_dict=optimizer_state_dict,
        history=history,
        metadata=metadata,
    )


__all__ = [
    "CheckpointFiles",
    "LoadedCheckpoint",
    "load_checkpoint",
    "save_checkpoint",
]
