"""Evaluation pipeline for anomaly scoring on labeled graph data."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from traffic_graph.config import PipelineConfig
from traffic_graph.data import (
    LogicalFlowBatch,
    FlowDataset,
    load_flow_dataset,
    preprocess_flow_dataset,
)
from traffic_graph.features import transform_graphs
from traffic_graph.graph import FlowInteractionGraphBuilder, InteractionGraph
from traffic_graph.pipeline.scoring import (
    build_edge_score_rows,
    build_flow_score_rows,
    build_graph_score_row,
    build_node_score_rows,
    compute_edge_anomaly_scores,
    compute_graph_anomaly_scores,
    compute_node_anomaly_scores,
)
from traffic_graph.pipeline.metrics import (
    BinaryScoreMetrics,
    coerce_binary_label,
    evaluate_scores,
    summarize_metric_sets,
)


def _extract_label_value(record: object, label_field: str) -> int | None:
    """Extract a binary label from a flow record or its metadata."""

    direct_value = getattr(record, label_field, None)
    if direct_value is not None:
        return coerce_binary_label(direct_value)

    metadata = getattr(record, "metadata", None)
    if isinstance(metadata, Mapping) and label_field in metadata:
        return coerce_binary_label(metadata[label_field])

    mapping_fn = getattr(record, "to_mapping", None)
    if callable(mapping_fn):
        payload = mapping_fn()
        if isinstance(payload, Mapping):
            if label_field in payload:
                return coerce_binary_label(payload[label_field])
            nested_metadata = payload.get("metadata")
            if isinstance(nested_metadata, Mapping) and label_field in nested_metadata:
                return coerce_binary_label(nested_metadata[label_field])

    return None


def _derive_logical_flow_label(
    logical_flow_ids: tuple[str, ...],
    raw_label_by_flow_id: Mapping[str, int | None],
) -> int | None:
    """Derive a logical-flow label by OR-ing labels from the source raw flows."""

    labels = [raw_label_by_flow_id.get(flow_id) for flow_id in logical_flow_ids]
    labeled_values = [label for label in labels if label is not None]
    if not labeled_values:
        return None
    return int(any(labeled_values))


def _derive_graph_label(logical_flow_labels: Mapping[str, int | None]) -> int | None:
    """Derive a window-level label from logical-flow labels."""

    labeled_values = [label for label in logical_flow_labels.values() if label is not None]
    if not labeled_values:
        return None
    return int(any(labeled_values))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a list of row mappings to CSV with stable headers."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    """Write a JSON payload using stable, human-readable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


@dataclass(slots=True)
class EvaluationPipelineResult:
    """Structured output from anomaly-score evaluation."""

    checkpoint_path: str
    output_directory: str
    score_reduction: str
    anomaly_threshold: float
    evaluation_label_field: str
    graph_scores: list[dict[str, object]] = field(default_factory=list)
    flow_scores: list[dict[str, object]] = field(default_factory=list)
    node_scores: list[dict[str, object]] = field(default_factory=list)
    edge_scores: list[dict[str, object]] = field(default_factory=list)
    graph_metrics: BinaryScoreMetrics = field(
        default_factory=lambda: BinaryScoreMetrics(
            threshold=0.5,
            support=0,
            positive_count=0,
            negative_count=0,
            roc_auc=None,
            pr_auc=None,
            precision=None,
            recall=None,
            f1=None,
            true_positive=0,
            false_positive=0,
            true_negative=0,
            false_negative=0,
        )
    )
    flow_metrics: BinaryScoreMetrics = field(
        default_factory=lambda: BinaryScoreMetrics(
            threshold=0.5,
            support=0,
            positive_count=0,
            negative_count=0,
            roc_auc=None,
            pr_auc=None,
            precision=None,
            recall=None,
            f1=None,
            true_positive=0,
            false_positive=0,
            true_negative=0,
            false_negative=0,
        )
    )
    artifact_paths: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def metrics_summary(self) -> dict[str, dict[str, float | int | None]]:
        """Return the nested metric summary used by CLI and JSON exports."""

        return summarize_metric_sets(
            {
                "graph": self.graph_metrics,
                "flow": self.flow_metrics,
            }
        )


class EvaluationPipeline:
    """Load a checkpoint and compute anomaly scores on labeled graph data."""

    def __init__(self, config: PipelineConfig) -> None:
        """Store the configuration used for loading and scoring."""

        self.config = config

    def _load_batches(
        self,
        dataset: FlowDataset,
        *,
        window_size: int,
        thresholds: object,
    ) -> list[LogicalFlowBatch]:
        """Load the configured flow dataset and preprocess it into logical batches."""

        return preprocess_flow_dataset(
            dataset,
            window_size=window_size,
            rules=thresholds,
        )

    def _build_graphs(
        self,
        batches: list[LogicalFlowBatch],
        *,
        graph_config: object,
    ) -> list[InteractionGraph]:
        """Build endpoint interaction graphs for every logical-flow batch."""

        graph_builder = FlowInteractionGraphBuilder(graph_config)
        return graph_builder.build_many(batches)

    def _build_raw_label_map(self, dataset: FlowDataset) -> dict[str, int | None]:
        """Create a flow-id to label mapping from raw input records."""

        raw_label_by_flow_id: dict[str, int | None] = {}
        for record in dataset.records:
            raw_label_by_flow_id[record.flow_id] = _extract_label_value(
                record,
                self.config.evaluation.evaluation_label_field,
            )
        return raw_label_by_flow_id

    def run(
        self,
        *,
        checkpoint_path: str | None = None,
        export: bool = True,
    ) -> EvaluationPipelineResult:
        """Run the evaluation pipeline and return scores plus metric summaries."""

        from traffic_graph.pipeline.checkpoint import load_checkpoint

        resolved_checkpoint_path = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else Path(self.config.evaluation.checkpoint_dir)
            / self.config.evaluation.checkpoint_tag
        )
        loaded_checkpoint = load_checkpoint(resolved_checkpoint_path)
        model = loaded_checkpoint.model
        model.eval()

        dataset = load_flow_dataset(
            self.config.data.input_path,
            data_format=self.config.data.format,
        )
        batches = self._load_batches(
            dataset,
            window_size=loaded_checkpoint.config.preprocessing.window_size,
            thresholds=loaded_checkpoint.config.preprocessing.short_flow_thresholds,
        )
        graphs = self._build_graphs(
            batches,
            graph_config=loaded_checkpoint.config.graph,
        )
        raw_label_by_flow_id = self._build_raw_label_map(dataset)
        packed_graphs = transform_graphs(
            graphs,
            loaded_checkpoint.feature_preprocessor,
            include_graph_structural_features=(
                loaded_checkpoint.config.features.use_graph_structural_features
            ),
        )

        graph_scores: list[dict[str, object]] = []
        flow_scores: list[dict[str, object]] = []
        node_scores: list[dict[str, object]] = []
        edge_scores: list[dict[str, object]] = []
        graph_labels: list[int | None] = []
        graph_values: list[float] = []
        flow_labels: list[int | None] = []
        flow_values: list[float] = []
        notes: list[str] = []

        edge_reconstruction_missing = False
        for graph_index, (batch, graph_sample, packed_graph) in enumerate(
            zip(batches, graphs, packed_graphs)
        ):
            output = model(packed_graph)
            node_scores_array = compute_node_anomaly_scores(
                packed_graph.node_features,
                output.reconstructed_node_features.detach().cpu().numpy(),
                discrete_mask=packed_graph.node_discrete_mask,
            )
            edge_scores_array = compute_edge_anomaly_scores(
                packed_graph.edge_features,
                (
                    output.reconstructed_edge_features.detach().cpu().numpy()
                    if output.reconstructed_edge_features is not None
                    else None
                ),
                discrete_mask=packed_graph.edge_discrete_mask,
            )
            if output.reconstructed_edge_features is None:
                edge_reconstruction_missing = True

            graph_score = float(
                compute_graph_anomaly_scores(
                    node_scores_array,
                    reduction=self.config.evaluation.score_reduction,
                )
            )
            logical_flow_labels = {
                logical_flow.logical_flow_id: _derive_logical_flow_label(
                    logical_flow.source_flow_ids,
                    raw_label_by_flow_id,
                )
                for logical_flow in batch.logical_flows
            }
            graph_label = _derive_graph_label(logical_flow_labels)

            graph_scores.append(
                {
                    **build_graph_score_row(graph_index, graph_sample, graph_score),
                    "graph_label": graph_label,
                    "graph_predicted_label": int(
                        graph_score >= self.config.evaluation.anomaly_threshold
                    ),
                }
            )
            graph_values.append(graph_score)
            graph_labels.append(graph_label)

            node_rows = build_node_score_rows(
                graph_index,
                graph_sample,
                node_scores_array,
            )
            edge_rows = build_edge_score_rows(
                graph_index,
                graph_sample,
                edge_scores_array,
            )
            flow_rows = build_flow_score_rows(
                graph_index,
                graph_sample,
                edge_scores_array,
            )
            node_scores.extend(node_rows)
            edge_scores.extend(edge_rows)
            flow_scores.extend(
                {
                    **row,
                    "flow_label": logical_flow_labels.get(row["logical_flow_id"]),
                    "flow_predicted_label": int(
                        row["flow_anomaly_score"] >= self.config.evaluation.anomaly_threshold
                    ),
                }
                for row in flow_rows
            )
            flow_values.extend(float(row["flow_anomaly_score"]) for row in flow_rows)
            flow_labels.extend(logical_flow_labels.get(row["logical_flow_id"]) for row in flow_rows)

            if not edge_rows:
                notes.append(
                    f"Window {graph_sample.window_index} produced no edge score rows."
                )

        graph_metrics = evaluate_scores(
            graph_labels,
            graph_values,
            threshold=self.config.evaluation.anomaly_threshold,
        )
        flow_metrics = evaluate_scores(
            flow_labels,
            flow_values,
            threshold=self.config.evaluation.anomaly_threshold,
        )

        if edge_reconstruction_missing:
            notes.append(
                "Edge reconstruction was disabled or unavailable; flow scores are based on placeholder edge reconstruction errors."
            )
        if graph_metrics.support == 0:
            notes.append(
                f"No graph labels were available for '{self.config.evaluation.evaluation_label_field}'."
            )
        if flow_metrics.support == 0:
            notes.append(
                f"No flow labels were available for '{self.config.evaluation.evaluation_label_field}'."
            )

        artifact_paths: dict[str, str] = {}
        output_directory = Path(self.config.output.directory) / "evaluation"
        if export:
            result_for_export = EvaluationPipelineResult(
                checkpoint_path=str(resolved_checkpoint_path),
                output_directory=str(output_directory),
                score_reduction=self.config.evaluation.score_reduction,
                anomaly_threshold=self.config.evaluation.anomaly_threshold,
                evaluation_label_field=self.config.evaluation.evaluation_label_field,
                graph_scores=graph_scores,
                flow_scores=flow_scores,
                node_scores=node_scores,
                graph_metrics=graph_metrics,
                flow_metrics=flow_metrics,
                artifact_paths={},
                notes=notes,
            )
            artifact_paths = export_score_report(output_directory, result_for_export)

        return EvaluationPipelineResult(
            checkpoint_path=str(resolved_checkpoint_path),
            output_directory=str(output_directory),
            score_reduction=self.config.evaluation.score_reduction,
            anomaly_threshold=self.config.evaluation.anomaly_threshold,
            evaluation_label_field=self.config.evaluation.evaluation_label_field,
            graph_scores=graph_scores,
            flow_scores=flow_scores,
            node_scores=node_scores,
            edge_scores=edge_scores,
            graph_metrics=graph_metrics,
            flow_metrics=flow_metrics,
            artifact_paths=artifact_paths,
            notes=notes,
        )


def export_score_report(
    output_directory: str | Path,
    result: EvaluationPipelineResult,
) -> dict[str, str]:
    """Persist score tables and metric summaries to disk."""

    directory = Path(output_directory)
    metrics_path = directory / "metrics.json"
    summary_path = directory / "summary.json"
    graph_scores_path = directory / "graph_scores.csv"
    edge_scores_path = directory / "edge_scores.csv"
    flow_scores_path = directory / "flow_scores.csv"
    node_scores_path = directory / "node_scores.csv"

    _write_csv(graph_scores_path, result.graph_scores)
    _write_csv(edge_scores_path, result.edge_scores)
    _write_csv(flow_scores_path, result.flow_scores)
    _write_csv(node_scores_path, result.node_scores)
    _write_json(
        metrics_path,
        {
            "score_reduction": result.score_reduction,
            "anomaly_threshold": result.anomaly_threshold,
            "evaluation_label_field": result.evaluation_label_field,
            "metrics": result.metrics_summary,
        },
    )
    _write_json(
        summary_path,
        {
            "graph_score_count": len(result.graph_scores),
            "edge_score_count": len(result.edge_scores),
            "flow_score_count": len(result.flow_scores),
            "node_score_count": len(result.node_scores),
            "metrics_path": metrics_path.name,
            "graph_scores_path": graph_scores_path.name,
            "edge_scores_path": edge_scores_path.name,
            "flow_scores_path": flow_scores_path.name,
            "node_scores_path": node_scores_path.name,
            "notes": result.notes,
        },
    )
    return {
        "metrics_json": str(metrics_path),
        "summary_json": str(summary_path),
        "graph_scores_csv": str(graph_scores_path),
        "edge_scores_csv": str(edge_scores_path),
        "flow_scores_csv": str(flow_scores_path),
        "node_scores_csv": str(node_scores_path),
    }


__all__ = [
    "EvaluationPipeline",
    "EvaluationPipelineResult",
    "export_score_report",
]
