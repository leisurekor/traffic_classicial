"""Training loop utilities for the minimal graph autoencoder."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from traffic_graph.config import PipelineConfig
from traffic_graph.features import FeaturePreprocessor, PackedGraphInput
from traffic_graph.models import GraphAutoEncoder, GraphTensorBatch, reconstruction_loss
from traffic_graph.pipeline.checkpoint import save_checkpoint


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """Loss metrics for one train or validation pass."""

    loss: float
    node_loss: float
    edge_loss: float
    graph_count: int
    batch_count: int

    def to_dict(self) -> dict[str, float | int]:
        """Convert the metrics into a flattened log-friendly mapping."""

        return {
            "loss": self.loss,
            "node_loss": self.node_loss,
            "edge_loss": self.edge_loss,
            "graph_count": self.graph_count,
            "batch_count": self.batch_count,
        }


@dataclass(frozen=True, slots=True)
class TrainingEpochMetrics:
    """Flattened per-epoch training and validation summary."""

    epoch: int
    train_loss: float
    train_node_loss: float
    train_edge_loss: float
    train_graph_count: int
    train_batch_count: int
    val_loss: float
    val_node_loss: float
    val_edge_loss: float
    val_graph_count: int
    val_batch_count: int

    def to_dict(self) -> dict[str, float | int]:
        """Convert the epoch summary into a JSON-friendly dictionary."""

        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "train_node_loss": self.train_node_loss,
            "train_edge_loss": self.train_edge_loss,
            "train_graph_count": self.train_graph_count,
            "train_batch_count": self.train_batch_count,
            "val_loss": self.val_loss,
            "val_node_loss": self.val_node_loss,
            "val_edge_loss": self.val_edge_loss,
            "val_graph_count": self.val_graph_count,
            "val_batch_count": self.val_batch_count,
        }


@dataclass(slots=True)
class TrainerFitResult:
    """High-level result object returned by :meth:`GraphAETrainer.fit`."""

    history: list[TrainingEpochMetrics]
    best_epoch: int
    best_val_loss: float
    stopped_early: bool
    latest_checkpoint_path: str
    best_checkpoint_path: str
    notes: list[str]


def _chunk_graphs(
    graphs: Sequence[PackedGraphInput],
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> list[list[PackedGraphInput]]:
    """Split a graph list into deterministic mini-batches."""

    ordered_graphs = list(graphs)
    if shuffle:
        random.Random(seed).shuffle(ordered_graphs)
    return [
        ordered_graphs[start : start + batch_size]
        for start in range(0, len(ordered_graphs), batch_size)
    ]


class GraphAETrainer:
    """Minimal trainer for the first unsupervised graph autoencoder."""

    def __init__(
        self,
        model: GraphAutoEncoder,
        config: PipelineConfig,
        feature_preprocessor: FeaturePreprocessor,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        """Create a trainer with optimizer, config, and checkpoint context."""

        self.model = model
        self.config = config
        self.feature_preprocessor = feature_preprocessor
        self.training_config = config.training
        self.device = torch.device(device or config.model.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        self.checkpoint_dir = Path(self.training_config.checkpoint_dir)

    def _run_epoch(
        self,
        graphs: Sequence[PackedGraphInput],
        *,
        train: bool,
        batch_size: int,
        seed: int,
    ) -> EpochMetrics:
        """Run one training or validation epoch over a graph sequence."""

        if not graphs:
            raise ValueError("At least one graph is required for an epoch pass.")

        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_node_loss = 0.0
        total_edge_loss = 0.0
        total_graph_count = 0
        total_batch_count = 0
        graph_batches = _chunk_graphs(
            graphs,
            batch_size,
            shuffle=train and self.training_config.shuffle,
            seed=seed,
        )

        with torch.set_grad_enabled(train):
            for batch_graphs in graph_batches:
                tensor_batch = GraphTensorBatch.from_packed_graphs(
                    batch_graphs,
                    device=self.device,
                )
                output = self.model(tensor_batch)
                loss_output = reconstruction_loss(
                    output,
                    weights=self.model.loss_weights,
                )
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss_output.total_loss.backward()
                    self.optimizer.step()

                batch_weight = len(batch_graphs)
                total_loss += float(loss_output.total_loss.detach()) * batch_weight
                total_node_loss += float(loss_output.node_loss.detach()) * batch_weight
                total_edge_loss += float(loss_output.edge_loss.detach()) * batch_weight
                total_graph_count += batch_weight
                total_batch_count += 1

        return EpochMetrics(
            loss=total_loss / total_graph_count,
            node_loss=total_node_loss / total_graph_count,
            edge_loss=total_edge_loss / total_graph_count,
            graph_count=total_graph_count,
            batch_count=total_batch_count,
        )

    def train_epoch(
        self,
        graphs: Sequence[PackedGraphInput],
        *,
        epoch: int,
        batch_size: int | None = None,
        seed: int | None = None,
    ) -> EpochMetrics:
        """Run one training epoch and return averaged reconstruction metrics."""

        return self._run_epoch(
            graphs,
            train=True,
            batch_size=batch_size or self.training_config.batch_size,
            seed=(self.training_config.seed + epoch) if seed is None else seed,
        )

    def validate_epoch(
        self,
        graphs: Sequence[PackedGraphInput],
        *,
        epoch: int,
        batch_size: int | None = None,
    ) -> EpochMetrics:
        """Run one validation epoch and return averaged reconstruction metrics."""

        return self._run_epoch(
            graphs,
            train=False,
            batch_size=batch_size or self.training_config.batch_size,
            seed=self.training_config.seed + epoch,
        )

    def fit(
        self,
        train_graphs: Sequence[PackedGraphInput],
        val_graphs: Sequence[PackedGraphInput],
        *,
        smoke_run: bool = False,
    ) -> TrainerFitResult:
        """Train the graph autoencoder with early stopping and checkpointing."""

        if not train_graphs:
            raise ValueError("Training requires at least one graph.")

        epochs = self.training_config.epochs
        batch_size = self.training_config.batch_size
        notes: list[str] = []
        if smoke_run:
            epochs = min(epochs, 2)
            batch_size = min(batch_size, 2)
            max_graphs = self.training_config.smoke_graph_limit
            train_graphs = train_graphs[:max_graphs]
            val_graphs = val_graphs[:max_graphs]
            notes.append(
                f"Smoke run enabled: using up to {max_graphs} graphs and {epochs} epochs."
            )

        if not val_graphs:
            notes.append(
                "Validation split is empty; validation metrics fall back to training metrics."
            )

        history: list[TrainingEpochMetrics] = []
        best_epoch = 0
        best_val_loss = float("inf")
        stopped_early = False
        latest_checkpoint_path = ""
        best_checkpoint_path = ""
        patience = self.training_config.early_stopping_patience
        stalled_epochs = 0

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(
                train_graphs,
                epoch=epoch,
                batch_size=batch_size,
                seed=self.training_config.seed + epoch,
            )
            if val_graphs:
                val_metrics = self.validate_epoch(
                    val_graphs,
                    epoch=epoch,
                    batch_size=batch_size,
                )
            else:
                val_metrics = train_metrics

            epoch_metrics = TrainingEpochMetrics(
                epoch=epoch,
                train_loss=train_metrics.loss,
                train_node_loss=train_metrics.node_loss,
                train_edge_loss=train_metrics.edge_loss,
                train_graph_count=train_metrics.graph_count,
                train_batch_count=train_metrics.batch_count,
                val_loss=val_metrics.loss,
                val_node_loss=val_metrics.node_loss,
                val_edge_loss=val_metrics.edge_loss,
                val_graph_count=val_metrics.graph_count,
                val_batch_count=val_metrics.batch_count,
            )
            history.append(epoch_metrics)
            print(
                "Epoch "
                f"{epoch}/{epochs} "
                f"train_loss={epoch_metrics.train_loss:.6f} "
                f"val_loss={epoch_metrics.val_loss:.6f} "
                f"train_node_loss={epoch_metrics.train_node_loss:.6f} "
                f"train_edge_loss={epoch_metrics.train_edge_loss:.6f}",
                flush=True,
            )

            latest_checkpoint_path = str(
                save_checkpoint(
                    self.checkpoint_dir,
                    tag="latest",
                    model=self.model,
                    optimizer=self.optimizer,
                    config=self.config,
                    feature_preprocessor=self.feature_preprocessor,
                    history=[entry.to_dict() for entry in history],
                    epoch=epoch,
                    best_epoch=best_epoch or epoch,
                    best_val_loss=best_val_loss
                    if best_val_loss != float("inf")
                    else epoch_metrics.val_loss,
                )
            )

            improved = epoch_metrics.val_loss < best_val_loss
            if improved:
                best_val_loss = epoch_metrics.val_loss
                best_epoch = epoch
                best_checkpoint_path = str(
                    save_checkpoint(
                        self.checkpoint_dir,
                        tag="best",
                        model=self.model,
                        optimizer=self.optimizer,
                        config=self.config,
                        feature_preprocessor=self.feature_preprocessor,
                        history=[entry.to_dict() for entry in history],
                        epoch=epoch,
                        best_epoch=best_epoch,
                        best_val_loss=best_val_loss,
                    )
                )
                stalled_epochs = 0
            elif val_graphs and patience > 0:
                stalled_epochs += 1
                if stalled_epochs >= patience:
                    stopped_early = True
                    notes.append(
                        f"Early stopping triggered at epoch {epoch} after {stalled_epochs} stalled epochs."
                    )
                    break

        if not best_checkpoint_path:
            best_checkpoint_path = latest_checkpoint_path

        return TrainerFitResult(
            history=history,
            best_epoch=best_epoch or len(history),
            best_val_loss=best_val_loss if best_val_loss != float("inf") else history[-1].val_loss,
            stopped_early=stopped_early,
            latest_checkpoint_path=latest_checkpoint_path,
            best_checkpoint_path=best_checkpoint_path,
            notes=notes,
        )


__all__ = [
    "EpochMetrics",
    "GraphAETrainer",
    "TrainerFitResult",
    "TrainingEpochMetrics",
]
