"""End-to-end unsupervised training pipeline for the graph autoencoder."""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from traffic_graph.config import PipelineConfig
from traffic_graph.data import (
    LogicalFlowBatch,
    LogicalFlowWindowStats,
    load_flow_dataset,
    preprocess_flow_dataset,
)
from traffic_graph.features import (
    fit_feature_preprocessor,
    summarize_packed_graph_input,
    transform_graphs,
)
from traffic_graph.graph import FlowInteractionGraphBuilder, InteractionGraph, summarize_graph
from traffic_graph.models import GraphAutoEncoder, GraphAutoEncoderConfig
from traffic_graph.pipeline.trainer import GraphAETrainer, TrainerFitResult


@dataclass(slots=True)
class TrainingPipelineResult:
    """Structured output returned by the training pipeline."""

    window_statistics: list[LogicalFlowWindowStats] = field(default_factory=list)
    graph_summaries: list[dict[str, int | float]] = field(default_factory=list)
    feature_summaries: list[dict[str, int]] = field(default_factory=list)
    node_feature_fields: tuple[str, ...] = ()
    edge_feature_fields: tuple[str, ...] = ()
    training_history: list[dict[str, float | int]] = field(default_factory=list)
    train_graph_count: int = 0
    val_graph_count: int = 0
    best_epoch: int = 0
    best_val_loss: float | None = None
    best_checkpoint_path: str = ""
    latest_checkpoint_path: str = ""
    feature_preprocessor_path: str = ""
    notes: list[str] = field(default_factory=list)


class TrainingPipeline:
    """Load data, build packed graphs, and train the first GAE model."""

    def __init__(self, config: PipelineConfig) -> None:
        """Store the typed pipeline configuration."""

        self.config = config

    def _load_and_preprocess(self) -> list[LogicalFlowBatch]:
        """Load flow data and preprocess it into logical-flow batches."""

        input_path = Path(self.config.data.input_path)
        dataset = load_flow_dataset(
            input_path,
            data_format=self.config.data.format,
        )
        return preprocess_flow_dataset(
            dataset,
            window_size=self.config.preprocessing.window_size,
            rules=self.config.preprocessing.short_flow_thresholds,
        )

    def _build_graphs(
        self,
        batches: Iterable[LogicalFlowBatch],
    ) -> list[InteractionGraph]:
        """Build endpoint interaction graphs from logical-flow batches."""

        graph_builder = FlowInteractionGraphBuilder(self.config.graph)
        return graph_builder.build_many(batches)

    def _split_graphs(
        self,
        graphs: list[InteractionGraph],
    ) -> tuple[list[InteractionGraph], list[InteractionGraph]]:
        """Split graphs into train and validation partitions."""

        if not graphs:
            raise ValueError("Training requires at least one graph.")
        if len(graphs) == 1:
            return graphs[:], []

        ordered_graphs = list(graphs)
        if self.config.training.shuffle:
            random.Random(self.config.training.seed).shuffle(ordered_graphs)

        validation_count = int(
            round(len(ordered_graphs) * self.config.training.validation_split_ratio)
        )
        if validation_count <= 0:
            validation_count = 1
        if validation_count >= len(ordered_graphs):
            validation_count = len(ordered_graphs) - 1

        validation_graphs = ordered_graphs[:validation_count]
        train_graphs = ordered_graphs[validation_count:]
        return train_graphs, validation_graphs

    def run(self, *, smoke_run: bool = False) -> TrainingPipelineResult:
        """Execute the full training pipeline and return structured results."""

        batches = self._load_and_preprocess()
        window_statistics = [batch.stats for batch in batches]
        graphs = self._build_graphs(batches)
        notes: list[str] = []

        if smoke_run:
            max_graphs = min(len(graphs), self.config.training.smoke_graph_limit)
            graphs = graphs[:max_graphs]
            notes.append(
                f"Smoke run limited the training set to {max_graphs} graphs."
            )

        graph_summaries = [summarize_graph(graph) for graph in graphs]
        train_graphs, val_graphs = self._split_graphs(graphs)
        feature_preprocessor = fit_feature_preprocessor(
            train_graphs,
            normalization_config=self.config.features.normalization,
            include_graph_structural_features=(
                self.config.features.use_graph_structural_features
            ),
        )
        packed_graphs = transform_graphs(
            graphs,
            feature_preprocessor,
            include_graph_structural_features=(
                self.config.features.use_graph_structural_features
            ),
        )
        feature_summaries = [
            summarize_packed_graph_input(packed_graph)
            for packed_graph in packed_graphs
        ]

        node_feature_dim = packed_graphs[0].node_feature_dim
        edge_feature_dim = packed_graphs[0].edge_feature_dim
        model = GraphAutoEncoder(
            node_input_dim=node_feature_dim,
            edge_input_dim=edge_feature_dim,
            config=GraphAutoEncoderConfig(
                hidden_dim=self.config.model.hidden_dim,
                latent_dim=self.config.model.latent_dim,
                num_layers=self.config.model.num_layers,
                dropout=self.config.model.dropout,
                use_edge_features=self.config.model.use_edge_features,
                reconstruct_edge_features=self.config.model.reconstruct_edge_features,
                use_temporal_edge_projector=(
                    self.config.model.use_temporal_edge_projector
                ),
                temporal_edge_hidden_dim=self.config.model.temporal_edge_hidden_dim,
                temporal_edge_field_names=self.config.model.temporal_edge_field_names,
                use_edge_categorical_embeddings=(
                    self.config.model.use_edge_categorical_embeddings
                ),
                edge_categorical_embedding_dim=(
                    self.config.model.edge_categorical_embedding_dim
                ),
                edge_categorical_bucket_size=(
                    self.config.model.edge_categorical_bucket_size
                ),
            ),
        )
        trainer = GraphAETrainer(
            model=model,
            config=self.config,
            feature_preprocessor=feature_preprocessor,
            device=self.config.model.device,
        )
        fit_result: TrainerFitResult = trainer.fit(
            transform_graphs(
                train_graphs,
                feature_preprocessor,
                include_graph_structural_features=(
                    self.config.features.use_graph_structural_features
                ),
            ),
            transform_graphs(
                val_graphs,
                feature_preprocessor,
                include_graph_structural_features=(
                    self.config.features.use_graph_structural_features
                ),
            ),
            smoke_run=smoke_run,
        )

        checkpoint_dir = fit_result.best_checkpoint_path or fit_result.latest_checkpoint_path
        feature_preprocessor_path = (
            str(Path(checkpoint_dir) / "preprocessor.json") if checkpoint_dir else ""
        )
        return TrainingPipelineResult(
            window_statistics=window_statistics,
            graph_summaries=graph_summaries,
            feature_summaries=feature_summaries,
            node_feature_fields=packed_graphs[0].node_feature_fields,
            edge_feature_fields=packed_graphs[0].edge_feature_fields,
            training_history=[entry.to_dict() for entry in fit_result.history],
            train_graph_count=len(train_graphs),
            val_graph_count=len(val_graphs),
            best_epoch=fit_result.best_epoch,
            best_val_loss=fit_result.best_val_loss,
            best_checkpoint_path=fit_result.best_checkpoint_path,
            latest_checkpoint_path=fit_result.latest_checkpoint_path,
            feature_preprocessor_path=feature_preprocessor_path,
            notes=notes + fit_result.notes,
        )


__all__ = [
    "TrainingPipeline",
    "TrainingPipelineResult",
]
