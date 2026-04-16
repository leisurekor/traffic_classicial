"""Pipeline orchestration skeleton for the traffic graph project."""

from __future__ import annotations

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
from traffic_graph.graph import FlowInteractionGraphBuilder, summarize_graph


@dataclass(frozen=True, slots=True)
class PipelineStage:
    """Single pipeline stage description used for reporting and orchestration."""

    name: str
    status: str
    detail: str


@dataclass(slots=True)
class PipelineReport:
    """Rendered view of a pipeline run or dry-run preview."""

    run_name: str
    input_path: str
    output_directory: str
    stages: list[PipelineStage]
    dry_run: bool
    window_statistics: list[LogicalFlowWindowStats] = field(default_factory=list)
    graph_summaries: list[dict[str, int | float]] = field(default_factory=list)
    feature_summaries: list[dict[str, int]] = field(default_factory=list)
    training_history: list[dict[str, float | int]] = field(default_factory=list)
    node_feature_fields: tuple[str, ...] = ()
    edge_feature_fields: tuple[str, ...] = ()
    train_graph_count: int = 0
    val_graph_count: int = 0
    best_checkpoint_path: str = ""
    latest_checkpoint_path: str = ""
    best_val_loss: float | None = None
    evaluation_metrics: dict[str, dict[str, float | int | None]] = field(
        default_factory=dict
    )
    evaluation_artifacts: dict[str, str] = field(default_factory=dict)
    export_artifacts: dict[str, str] = field(default_factory=dict)
    evaluation_label_field: str = ""
    evaluation_score_reduction: str = ""
    evaluation_threshold: float | None = None
    alert_summary: dict[str, object] = field(default_factory=dict)
    explanation_summary: dict[str, object] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Render the pipeline report as a human-readable text block."""

        mode = "dry-run" if self.dry_run else "skeleton-run"
        lines = [
            f"Run name: {self.run_name}",
            f"Mode: {mode}",
            f"Input: {self.input_path}",
            f"Output: {self.output_directory}",
            "Stages:",
        ]
        lines.extend(
            f"  - {stage.name} [{stage.status}]: {stage.detail}" for stage in self.stages
        )
        if self.window_statistics:
            lines.append("Window statistics:")
            lines.extend(
                "  - "
                f"window[{stats.index}] {stats.window_start.isoformat()} -> "
                f"{stats.window_end.isoformat()}: "
                f"raw={stats.raw_flow_count}, short={stats.short_flow_count}, "
                f"long={stats.long_flow_count}, logical={stats.logical_flow_count}"
                for stats in self.window_statistics
            )
        if self.graph_summaries:
            lines.append("Graph summaries:")
            lines.extend(
                "  - "
                f"window[{int(summary['window_index'])}] "
                f"nodes={int(summary['node_count'])}, "
                f"edges={int(summary['edge_count'])}, "
                f"clients={int(summary['client_node_count'])}, "
                f"servers={int(summary['server_node_count'])}, "
                f"communication_edges={int(summary['communication_edge_count'])}, "
                f"association_edges={int(summary['association_edge_count'])}, "
                f"same_src_ip_edges={int(summary['association_same_src_ip_edge_count'])}, "
                f"same_dst_subnet_edges={int(summary['association_same_dst_subnet_edge_count'])}, "
                f"aggregated_edges={int(summary['aggregated_edge_count'])}"
                for summary in self.graph_summaries
            )
        if self.feature_summaries:
            lines.append("Feature summaries:")
            lines.extend(
                "  - "
                f"window[{int(summary['window_index'])}] "
                f"node_features={summary['node_feature_dim']}x{summary['node_count']}, "
                f"edge_features={summary['edge_feature_dim']}x{summary['edge_count']}"
                for summary in self.feature_summaries
            )
        if self.training_history:
            lines.append("Training history:")
            lines.extend(
                "  - "
                f"epoch={int(entry['epoch'])} "
                f"train_loss={float(entry['train_loss']):.6f} "
                f"val_loss={float(entry['val_loss']):.6f} "
                f"train_node_loss={float(entry['train_node_loss']):.6f} "
                f"train_edge_loss={float(entry['train_edge_loss']):.6f}"
                for entry in self.training_history
            )
        if self.evaluation_metrics:
            lines.append("Evaluation metrics:")
            for scope, metrics in self.evaluation_metrics.items():
                def _fmt(value: object) -> str:
                    if value is None:
                        return "n/a"
                    if isinstance(value, (int, float)):
                        return f"{float(value):.6f}"
                    return str(value)

                lines.append(
                    "  - "
                    f"{scope}: "
                    f"roc_auc={_fmt(metrics.get('roc_auc'))}, "
                    f"pr_auc={_fmt(metrics.get('pr_auc'))}, "
                    f"precision={_fmt(metrics.get('precision'))}, "
                    f"recall={_fmt(metrics.get('recall'))}, "
                    f"f1={_fmt(metrics.get('f1'))}, "
                    f"threshold={_fmt(metrics.get('threshold'))}"
                )
        if self.evaluation_artifacts:
            lines.append("Evaluation artifacts:")
            lines.extend(
                f"  - {name}: {path}" for name, path in self.evaluation_artifacts.items()
            )
        if self.export_artifacts:
            lines.append("Export artifacts:")
            lines.extend(f"  - {name}: {path}" for name, path in self.export_artifacts.items())
        if self.evaluation_label_field:
            lines.append(f"Evaluation label field: {self.evaluation_label_field}")
        if self.evaluation_score_reduction:
            lines.append(
                f"Graph score reduction: {self.evaluation_score_reduction}"
            )
        if self.evaluation_threshold is not None:
            lines.append(f"Anomaly threshold: {self.evaluation_threshold:.6f}")
        if self.alert_summary:
            lines.append("Alert summary:")
            total_count = int(self.alert_summary.get("total_count", 0))
            positive_count = int(self.alert_summary.get("positive_count", 0))
            positive_rate = float(self.alert_summary.get("positive_rate", 0.0))
            lines.append(
                f"  - total={total_count}, positive={positive_count}, "
                f"positive_rate={positive_rate:.6f}"
            )
            scope_counts = self.alert_summary.get("scope_counts", {})
            positive_scope_counts = self.alert_summary.get(
                "positive_scope_counts",
                {},
            )
            for scope_name in ("graph", "node", "edge", "flow"):
                lines.append(
                    f"  - {scope_name}: total={int(scope_counts.get(scope_name, 0))}, "
                    f"positive={int(positive_scope_counts.get(scope_name, 0))}"
                )
            level_counts = self.alert_summary.get("level_counts", {})
            lines.append(
                "  - levels: "
                f"low={int(level_counts.get('low', 0))}, "
                f"medium={int(level_counts.get('medium', 0))}, "
                f"high={int(level_counts.get('high', 0))}"
            )
        if self.explanation_summary:
            lines.append("Explanation candidates:")
            lines.append(
                "  - "
                f"scope={self.explanation_summary.get('scope', 'flow')}, "
                f"only_alerts={self.explanation_summary.get('only_alerts', True)}, "
                f"top_k={self.explanation_summary.get('top_k', 'n/a')}, "
                f"total={int(self.explanation_summary.get('total_count', 0))}"
            )
            lines.append(
                "  - "
                f"alerts={int(self.explanation_summary.get('alert_count', 0))}, "
                f"labeled={int(self.explanation_summary.get('labeled_count', 0))}, "
                f"max_score={self.explanation_summary.get('max_anomaly_score', 'n/a')}"
            )
            scope_counts = self.explanation_summary.get("scope_counts", {})
            if isinstance(scope_counts, dict):
                for scope_name in ("graph", "flow", "node"):
                    if scope_name in scope_counts:
                        lines.append(
                            f"  - {scope_name}: {int(scope_counts.get(scope_name, 0))}"
                        )
            preview_ids = self.explanation_summary.get("preview_sample_ids", [])
            if isinstance(preview_ids, list) and preview_ids:
                lines.append("  - preview: " + ", ".join(str(item) for item in preview_ids))
        if self.train_graph_count or self.val_graph_count:
            lines.append(
                f"Train/val split: train={self.train_graph_count}, val={self.val_graph_count}"
            )
        if self.best_checkpoint_path:
            lines.append(f"Best checkpoint: {self.best_checkpoint_path}")
        if self.latest_checkpoint_path:
            lines.append(f"Latest checkpoint: {self.latest_checkpoint_path}")
        if self.best_val_loss is not None:
            lines.append(f"Best validation loss: {self.best_val_loss:.6f}")
        if self.node_feature_fields:
            lines.append("Node feature fields:")
            lines.append("  - " + ", ".join(self.node_feature_fields))
        if self.edge_feature_fields:
            lines.append("Edge feature fields:")
            lines.append("  - " + ", ".join(self.edge_feature_fields))
        if self.notes:
            lines.append("Notes:")
            lines.extend(f"  - {note}" for note in self.notes)
        lines.append(
            "Note: feature extraction, graph autoencoder training, alerting, and "
            "persistence are now wired into the pipeline; explanation-ready sample "
            "organization is available, while tree distillation, LLM explanation, and "
            "visualization remain extension points."
        )
        return "\n".join(lines)


class PipelineRunner:
    """Assemble configuration and report the planned graph-based pipeline stages."""

    def __init__(self, config: PipelineConfig) -> None:
        """Store the typed configuration used by the pipeline."""

        self.config = config

    def _load_and_preprocess(self) -> tuple[list[LogicalFlowBatch], str, str]:
        """Load raw flows from disk and preprocess them into logical-flow batches."""

        input_path = Path(self.config.data.input_path)
        dataset = load_flow_dataset(
            input_path,
            data_format=self.config.data.format,
        )
        batches = preprocess_flow_dataset(
            dataset,
            window_size=self.config.preprocessing.window_size,
            rules=self.config.preprocessing.short_flow_thresholds,
        )
        load_detail = (
            f"Loaded {len(dataset.records)} raw flows from {input_path.as_posix()}."
        )
        preprocess_detail = (
            f"Prepared {sum(len(batch.logical_flows) for batch in batches)} logical "
            f"flows across {len(batches)} windows using "
            f"{self.config.preprocessing.window_size}s windows."
        )
        return batches, load_detail, preprocess_detail

    def run(
        self,
        *,
        dry_run: bool = False,
        show_window_stats: bool = False,
        show_graph_summary: bool = False,
        show_feature_summary: bool = False,
        show_alert_summary: bool = False,
        show_explanation_summary: bool = False,
        explanation_scope: str = "flow",
        explanation_top_k: int = 20,
        export_dir: str | None = None,
        train: bool = False,
        smoke_train: bool = False,
        evaluate: bool = False,
    ) -> PipelineReport:
        """Run the current pipeline skeleton and return a textual report."""

        stage_status = "planned" if dry_run else "ready"
        train_requested = train or smoke_train
        evaluate_requested = evaluate
        requested_preview = (
            show_window_stats
            or show_graph_summary
            or show_feature_summary
            or (train_requested and dry_run)
        )
        load_status = stage_status
        preprocess_status = stage_status
        graph_status = stage_status
        feature_status = stage_status
        train_status = "planned" if dry_run else ("ready" if train_requested else stage_status)
        evaluation_status = "planned" if dry_run else ("ready" if evaluate_requested else stage_status)
        load_detail = "Normalize raw records into the unified FlowRecord schema."
        preprocess_detail = (
            "Split flows into fixed windows and aggregate short flows into logical "
            "records for graph construction."
        )
        graph_detail = "Construct endpoint interaction graphs for downstream features."
        feature_detail = (
            "Prepare packed node and edge feature matrices for a later unsupervised model."
        )
        train_detail = (
            "Train the minimal Graph AutoEncoder with reconstruction loss, "
            "train/validation splitting, and checkpointing."
        )
        evaluation_detail = (
            "Compute anomaly scores from reconstruction error and evaluate them "
            "with binary classification metrics."
        )
        window_statistics: list[LogicalFlowWindowStats] = []
        graph_summaries: list[dict[str, int | float]] = []
        feature_summaries: list[dict[str, int]] = []
        training_history: list[dict[str, float | int]] = []
        node_feature_fields: tuple[str, ...] = ()
        edge_feature_fields: tuple[str, ...] = ()
        train_graph_count = 0
        val_graph_count = 0
        best_checkpoint_path = ""
        latest_checkpoint_path = ""
        best_val_loss: float | None = None
        evaluation_metrics: dict[str, dict[str, float | int | None]] = {}
        evaluation_artifacts: dict[str, str] = {}
        export_artifacts: dict[str, str] = {}
        evaluation_label_field = ""
        evaluation_score_reduction = ""
        evaluation_threshold: float | None = None
        alert_summary: dict[str, object] = {}
        explanation_summary: dict[str, object] = {}
        notes: list[str] = []
        checkpoint_path_for_eval = ""

        if train_requested and not dry_run:
            try:
                from traffic_graph.pipeline.training_pipeline import TrainingPipeline
            except ModuleNotFoundError as exc:
                train_status = "skipped"
                notes.append(
                    "Training was requested but the PyTorch dependency is not available "
                    f"in this environment: {exc}"
                )
            else:
                try:
                    training_result = TrainingPipeline(self.config).run(
                        smoke_run=smoke_train,
                    )
                except FileNotFoundError:
                    train_status = "failed"
                    notes.append(
                        "Training failed because the input file was not found: "
                        f"{Path(self.config.data.input_path)}"
                    )
                except ValueError as exc:
                    train_status = "failed"
                    notes.append(f"Training failed: {exc}")
                else:
                    completed_status = "completed"
                    load_status = completed_status
                    preprocess_status = completed_status
                    graph_status = completed_status
                    feature_status = completed_status
                    train_status = completed_status
                    window_statistics = training_result.window_statistics
                    graph_summaries = training_result.graph_summaries
                    feature_summaries = training_result.feature_summaries
                    training_history = training_result.training_history
                    node_feature_fields = training_result.node_feature_fields
                    edge_feature_fields = training_result.edge_feature_fields
                    train_graph_count = training_result.train_graph_count
                    val_graph_count = training_result.val_graph_count
                    best_checkpoint_path = training_result.best_checkpoint_path
                    latest_checkpoint_path = training_result.latest_checkpoint_path
                    best_val_loss = training_result.best_val_loss
                    notes.extend(training_result.notes)
                    checkpoint_path_for_eval = (
                        best_checkpoint_path or latest_checkpoint_path
                    )
                    if not graph_summaries:
                        notes.append(
                            "Training completed but no endpoint graphs were produced."
                        )
                    if not feature_summaries:
                        notes.append(
                            "Training completed but no packed feature views were produced."
                        )
                    if training_history:
                        final_epoch = training_history[-1]
                        train_detail = (
                            f"Trained for {len(training_history)} epochs; "
                            f"final train loss={float(final_epoch['train_loss']):.6f}, "
                            f"best val loss={best_val_loss:.6f}."
                        )
                    else:
                        train_detail = "Training completed without recorded epochs."

        if evaluate_requested and not dry_run:
            try:
                from traffic_graph.pipeline.eval_pipeline import EvaluationPipeline
            except ModuleNotFoundError as exc:
                evaluation_status = "skipped"
                notes.append(
                    "Evaluation was requested but the PyTorch dependency is not available "
                    f"in this environment: {exc}"
                )
            else:
                try:
                    evaluation_result = EvaluationPipeline(self.config).run(
                        checkpoint_path=checkpoint_path_for_eval or None,
                    )
                except ModuleNotFoundError as exc:
                    evaluation_status = "skipped"
                    notes.append(
                        "Evaluation was requested but the PyTorch dependency is not available "
                        f"in this environment: {exc}"
                    )
                except FileNotFoundError as exc:
                    evaluation_status = "failed"
                    notes.append(f"Evaluation failed because a required file was missing: {exc}")
                except ValueError as exc:
                    evaluation_status = "failed"
                    notes.append(f"Evaluation failed: {exc}")
                else:
                    evaluation_status = "completed"
                    evaluation_metrics = evaluation_result.metrics_summary
                    evaluation_artifacts = evaluation_result.artifact_paths
                    evaluation_label_field = evaluation_result.evaluation_label_field
                    evaluation_score_reduction = evaluation_result.score_reduction
                    evaluation_threshold = evaluation_result.anomaly_threshold
                    notes.extend(evaluation_result.notes)
                    if not evaluation_result.graph_scores:
                        notes.append("Evaluation completed but no graph scores were produced.")
                    try:
                        from traffic_graph.pipeline.alerting import (
                            build_alert_records,
                            summarize_alerts,
                        )
                        from traffic_graph.pipeline.report_io import export_run_bundle
                    except ModuleNotFoundError as exc:
                        notes.append(
                            "Alerting or persistence modules could not be imported: "
                            f"{exc}"
                        )
                    else:
                        alert_records = build_alert_records(
                            evaluation_result,
                            self.config.alerting,
                        )
                        if show_alert_summary:
                            alert_summary = summarize_alerts(alert_records)
                            if alert_summary:
                                notes.append(
                                    "Alert summary computed for "
                                    f"{int(alert_summary.get('positive_count', 0))} "
                                    "positive alerts."
                                )
                        export_base_dir = (
                            Path(export_dir)
                            if export_dir is not None
                            else Path(self.config.output.directory) / "exports"
                        )
                        try:
                            export_result = export_run_bundle(
                                evaluation_result,
                                alert_records,
                                evaluation_result.metrics_summary,
                                export_base_dir,
                                run_id=self.config.pipeline.run_name,
                                split="eval",
                                anomaly_threshold=self.config.alerting.anomaly_threshold,
                            )
                        except (OSError, ValueError) as exc:
                            notes.append(f"Export bundle generation failed: {exc}")
                        else:
                            export_artifacts = export_result.artifact_paths
                            notes.extend(export_result.notes)
                            notes.append(
                                "Exported score, alert, and metric bundles to "
                                f"{export_result.run_directory}."
                            )
                            if show_explanation_summary:
                                try:
                                    from traffic_graph.explain.explanation_samples import (
                                        build_explanation_samples,
                                        summarize_explanation_samples,
                                    )
                                    from traffic_graph.pipeline.replay_io import (
                                        load_export_bundle,
                                    )
                                except ModuleNotFoundError as exc:
                                    notes.append(
                                        "Explanation sample modules could not be imported: "
                                        f"{exc}"
                                    )
                                else:
                                    try:
                                        replay_bundle = load_export_bundle(
                                            export_result.run_directory
                                        )
                                        explanation_samples = build_explanation_samples(
                                            replay_bundle,
                                            scope=explanation_scope,  # type: ignore[arg-type]
                                            only_alerts=True,
                                            top_k=explanation_top_k,
                                        )
                                    except (FileNotFoundError, OSError, ValueError) as exc:
                                        notes.append(
                                            "Explanation candidate generation failed: "
                                            f"{exc}"
                                        )
                                    else:
                                        explanation_summary = (
                                            summarize_explanation_samples(
                                                explanation_samples
                                            ).to_dict()
                                        )
                                        explanation_summary["scope"] = explanation_scope
                                        explanation_summary["only_alerts"] = True
                                        explanation_summary["top_k"] = explanation_top_k
                                        explanation_summary["preview_sample_ids"] = [
                                            sample.sample_id
                                            for sample in explanation_samples[:3]
                                        ]
                                        notes.append(
                                            "Prepared "
                                            f"{len(explanation_samples)} explanation-ready "
                                            f"{explanation_scope} samples from the exported "
                                            "evaluation bundle."
                                        )
                    evaluation_detail = (
                        f"Scored {len(evaluation_result.graph_scores)} graphs and "
                        f"{len(evaluation_result.flow_scores)} flows using "
                        f"{evaluation_score_reduction} pooling at threshold "
                        f"{evaluation_threshold:.6f}."
                    )
                    if evaluation_metrics:
                        graph_metrics = evaluation_metrics.get("graph", {})
                        flow_metrics = evaluation_metrics.get("flow", {})
                        evaluation_detail = (
                            f"Graph ROC-AUC={graph_metrics.get('roc_auc', 'n/a')}, "
                            f"flow ROC-AUC={flow_metrics.get('roc_auc', 'n/a')}; "
                            f"threshold={evaluation_threshold:.6f}."
                        )

        if requested_preview:
            if train_requested and not dry_run and training_history:
                pass
            else:
                try:
                    batches, load_detail, preprocess_detail = self._load_and_preprocess()
                except FileNotFoundError:
                    load_status = "skipped"
                    preprocess_status = "skipped"
                    graph_status = (
                        "skipped" if show_graph_summary or show_feature_summary else stage_status
                    )
                    feature_status = "skipped" if show_feature_summary else stage_status
                    notes.append(
                        "Preview unavailable because the input file was not found: "
                        f"{Path(self.config.data.input_path)}"
                    )
                except ValueError as exc:
                    load_status = "failed"
                    preprocess_status = "failed"
                    graph_status = (
                        "failed" if show_graph_summary or show_feature_summary else stage_status
                    )
                    feature_status = "failed" if show_feature_summary else stage_status
                    notes.append(f"Preview unavailable: {exc}")
                else:
                    completed_status = "previewed" if dry_run else "completed"
                    load_status = completed_status
                    preprocess_status = completed_status
                    graphs = []
                    if show_window_stats:
                        window_statistics = [batch.stats for batch in batches]
                        if not window_statistics:
                            notes.append(
                                "Input dataset loaded successfully but no windows were produced."
                            )
                    if show_graph_summary or show_feature_summary:
                        graph_builder = FlowInteractionGraphBuilder(self.config.graph)
                        graphs = graph_builder.build_many(batches)
                        graph_status = completed_status
                        total_communication_edges = sum(
                            graph.stats.communication_edge_count for graph in graphs
                        )
                        total_association_edges = sum(
                            graph.stats.association_edge_count for graph in graphs
                        )
                        graph_detail = (
                            f"Built {len(graphs)} endpoint graphs with "
                            f"{total_communication_edges} communication edges and "
                            f"{total_association_edges} association edges."
                        )
                        if show_graph_summary:
                            graph_summaries = [summarize_graph(graph) for graph in graphs]
                        if not graphs:
                            notes.append(
                                "Logical flows were prepared successfully but no endpoint graphs were produced."
                            )
                    if show_feature_summary:
                        preprocessor = fit_feature_preprocessor(
                            graphs,
                            normalization_config=self.config.features.normalization,
                            include_graph_structural_features=(
                                self.config.features.use_graph_structural_features
                            ),
                        )
                        packed_graphs = transform_graphs(
                            graphs,
                            preprocessor,
                            include_graph_structural_features=(
                                self.config.features.use_graph_structural_features
                            ),
                        )
                        feature_summaries = [
                            summarize_packed_graph_input(packed_graph)
                            for packed_graph in packed_graphs
                        ]
                        feature_status = completed_status
                        node_feature_fields = preprocessor.node_field_names
                        edge_feature_fields = preprocessor.edge_field_names
                        total_node_rows = sum(
                            summary["node_count"] for summary in feature_summaries
                        )
                        total_edge_rows = sum(
                            summary["edge_count"] for summary in feature_summaries
                        )
                        normalization_label = (
                            self.config.features.normalization.method
                            if self.config.features.normalization.enabled
                            else "disabled"
                        )
                        feature_detail = (
                            f"Packed features for {total_node_rows} nodes and "
                            f"{total_edge_rows} edges across {len(feature_summaries)} graphs "
                            f"using {normalization_label} normalization."
                        )
                        if not feature_summaries:
                            notes.append(
                                "Endpoint graphs were built successfully but no packed feature views were produced."
                            )

        stages = [
            PipelineStage(
                name="load_flows",
                status=load_status,
                detail=load_detail,
            ),
            PipelineStage(
                name="preprocess_logical_flows",
                status=preprocess_status,
                detail=preprocess_detail,
            ),
            PipelineStage(
                name="build_interaction_graph",
                status=graph_status,
                detail=graph_detail,
            ),
            PipelineStage(
                name="extract_graph_features",
                status=feature_status,
                detail=feature_detail,
            ),
            PipelineStage(
                name="train_graph_autoencoder",
                status=train_status,
                detail=train_detail,
            ),
            PipelineStage(
                name="evaluate_anomaly_scores",
                status=evaluation_status,
                detail=evaluation_detail,
            ),
        ]
        return PipelineReport(
            run_name=self.config.pipeline.run_name,
            input_path=self.config.data.input_path,
            output_directory=self.config.output.directory,
            stages=stages,
            dry_run=dry_run,
            window_statistics=window_statistics,
            graph_summaries=graph_summaries,
            feature_summaries=feature_summaries,
            training_history=training_history,
            node_feature_fields=node_feature_fields,
            edge_feature_fields=edge_feature_fields,
            train_graph_count=train_graph_count,
            val_graph_count=val_graph_count,
            best_checkpoint_path=best_checkpoint_path,
            latest_checkpoint_path=latest_checkpoint_path,
            best_val_loss=best_val_loss,
            evaluation_metrics=evaluation_metrics,
            evaluation_artifacts=evaluation_artifacts,
            export_artifacts=export_artifacts,
            evaluation_label_field=evaluation_label_field,
            evaluation_score_reduction=evaluation_score_reduction,
            evaluation_threshold=evaluation_threshold,
            alert_summary=alert_summary,
            explanation_summary=explanation_summary,
            notes=notes,
        )
