"""Reproducible mini experiment runner for real PCAP -> graph validation.

This module turns the existing real-PCAP smoke path into a small, repeatable
experiment entrypoint. It supports two modes:

- ``smoke``: validate the real packet-to-flow-to-graph chain without labels.
- ``binary_evaluation``: fit on benign graphs only and evaluate benign versus
  malicious graphs using the same anomaly-scoring backend.

The implementation intentionally stays small and reuses the repository's
existing graph, feature, scoring, alerting, and artifact-export plumbing.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd

from traffic_graph.config import (
    AlertingConfig,
    AssociationEdgeConfig,
    DataConfig,
    EvaluationConfig,
    FeatureNormalizationConfig,
    FeaturesConfig,
    GraphConfig,
    ModelConfig,
    OutputConfig,
    PipelineConfig,
    PipelineRuntimeConfig,
    PreprocessingConfig,
    ShortFlowThresholds,
    TrainingConfig,
)
from traffic_graph.data import (
    LogicalFlowWindowStats,
    PcapFlowLoadSummary,
    inspect_classic_pcap,
    load_pcap_flow_dataset,
    preprocess_flow_dataset,
)
from traffic_graph.features import (
    PackedGraphInput,
    fit_feature_preprocessor,
    summarize_packed_graph_input,
    transform_graphs,
)
from traffic_graph.graph import FlowInteractionGraphBuilder, InteractionGraph, summarize_graph
from traffic_graph.pipeline.alerting import build_alert_records, summarize_alerts
from traffic_graph.pipeline.binary_detection import (
    BINARY_ATTACK_METRIC_FIELDS,
    BINARY_DETECTION_SCORE_FIELDS,
    BinaryAttackMetricRecord,
    BinaryDetectionScoreRecord,
)
from traffic_graph.pipeline.metrics import BinaryScoreMetrics, evaluate_scores
from traffic_graph.pipeline.pcap_graph_smoke import (
    PcapGraphScoreReduction,
    _apply_graph_score_reduction_to_rows,
    _graph_score_threshold_from_rows,
    _has_torch,
    _quantile_summary,
    _score_graph_rows_with_gae_checkpoint,
    _score_from_row,
    _score_with_fallback,
    _slugify_token,
    _split_graphs,
    _timestamp_token,
    _train_and_score_with_gae,
)
from traffic_graph.pipeline.report_io import (
    RunBundleExportResult,
    build_run_bundle_layout,
    export_run_bundle,
)
from traffic_graph.pipeline.scorer_roles import normalize_run_scorer_role

PcapGraphExperimentMode: TypeAlias = Literal["smoke", "binary_evaluation"]
PcapSourceRole: TypeAlias = Literal["benign", "malicious"]
PcapPacketSamplingMode: TypeAlias = Literal["prefix", "random_window"]

PCAP_GRAPH_SUMMARY_FIELDS: tuple[str, ...] = (
    "graph_id",
    "source_id",
    "split_assignment",
    "source_role",
    "source_name",
    "source_path",
    "raw_label",
    "binary_label",
    "window_index",
    "window_start",
    "window_end",
    "node_count",
    "edge_count",
    "client_node_count",
    "server_node_count",
    "aggregated_edge_count",
    "communication_edge_count",
    "association_edge_count",
    "association_same_src_ip_edge_count",
    "association_same_dst_subnet_edge_count",
    "node_score_p75",
    "node_score_topk_mean",
    "flow_score_p75",
    "flow_score_topk_mean",
    "short_flow_score_p75",
    "short_flow_score_topk_mean",
    "long_flow_score_p75",
    "long_flow_score_topk_mean",
    "component_max_flow_score_topk_mean",
    "component_max_node_score_topk_mean",
    "server_neighborhood_flow_score_topk_mean",
)
"""Stable column order for exported graph summaries."""

PCAP_SOURCE_SCORE_SUMMARY_FIELDS: tuple[str, ...] = (
    "source_id",
    "source_name",
    "source_path",
    "source_role",
    "raw_label",
    "binary_label",
    "score_scope",
    "availability",
    "graph_count",
    "scored_graph_count",
    "train_graph_count",
    "val_graph_count",
    "benign_test_graph_count",
    "malicious_test_graph_count",
    "smoke_graph_count",
    "score_count",
    "score_mean",
    "score_median",
    "score_p90",
    "score_p95",
    "score_max",
)
"""Stable column order for per-source score summaries."""

PCAP_SPLIT_SCORE_SUMMARY_FIELDS: tuple[str, ...] = (
    "split_name",
    "score_scope",
    "availability",
    "graph_count",
    "score_count",
    "score_mean",
    "score_median",
    "score_p90",
    "score_p95",
    "score_max",
)
"""Stable column order for split-level score summaries."""

PCAP_MALICIOUS_SOURCE_METRIC_FIELDS: tuple[str, ...] = (
    "source_id",
    "source_name",
    "source_path",
    "raw_label",
    "benign_graph_count",
    "malicious_graph_count",
    "support",
    "positive_count",
    "negative_count",
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "threshold",
    "score_mean",
    "score_median",
    "score_p90",
    "score_p95",
    "score_max",
)
"""Stable column order for malicious-source binary metrics."""

PCAP_TRAIN_GRAPH_SCORE_FIELDS: tuple[str, ...] = (
    "run_id",
    "timestamp",
    "graph_id",
    "source_id",
    "source_name",
    "source_path",
    "source_role",
    "split_name",
    "label",
    "binary_label",
    "score",
    "threshold",
    "is_alert",
    "backend",
    "graph_score_reduction",
    "node_score_count",
    "node_score_mean",
    "node_score_p75",
    "node_score_p90",
    "node_score_max",
    "node_score_topk_mean",
    "edge_score_count",
    "edge_score_mean",
    "edge_score_p75",
    "edge_score_p90",
    "edge_score_max",
    "edge_score_topk_mean",
    "flow_score_count",
    "flow_score_mean",
    "flow_score_p75",
    "flow_score_p90",
    "flow_score_max",
    "flow_score_topk_mean",
    "flow_duration_p75",
    "flow_duration_topk_mean",
    "flow_iat_proxy_mean",
    "flow_iat_proxy_std",
    "flow_iat_proxy_p75",
    "flow_iat_proxy_topk_mean",
    "flow_pkt_count_p75",
    "flow_pkt_count_topk_mean",
    "flow_pkt_rate_p90",
    "flow_pkt_rate_topk_mean",
    "short_flow_score_count",
    "short_flow_score_mean",
    "short_flow_score_p75",
    "short_flow_score_p90",
    "short_flow_score_max",
    "short_flow_score_topk_mean",
    "long_flow_score_count",
    "long_flow_score_mean",
    "long_flow_score_p75",
    "long_flow_score_p90",
    "long_flow_score_max",
    "long_flow_score_topk_mean",
    "short_flow_ratio",
    "component_count",
    "component_max_node_score_topk_mean",
    "component_max_flow_score_topk_mean",
    "component_max_flow_score_p75",
    "component_max_server_concentration",
    "server_neighborhood_flow_score_topk_mean",
    "node_count",
    "edge_count",
    "client_node_count",
    "server_node_count",
    "aggregated_edge_count",
    "communication_edge_count",
    "association_edge_count",
    "association_same_src_ip_edge_count",
    "association_same_dst_subnet_edge_count",
    "edge_density",
    "aggregated_edge_ratio",
    "association_edge_ratio",
    "server_concentration",
    "communication_per_server",
    "window_index",
    "window_start",
    "window_end",
)
"""Stable column order for training-reference graph score exports."""

PCAP_COMPARISON_SUMMARY_FIELDS: tuple[str, ...] = (
    "run_id",
    "experiment_label",
    "timestamp",
    "mode",
    "backend",
    "scorer_type",
    "benign_source_count",
    "malicious_source_count",
    "benign_graph_total",
    "benign_train_graph_count",
    "benign_test_graph_count",
    "malicious_test_graph_count",
    "threshold_percentile",
    "threshold",
    "graph_score_reduction",
    "scorer_role",
    "overall_metrics_availability",
    "overall_roc_auc",
    "overall_pr_auc",
    "overall_precision",
    "overall_recall",
    "overall_f1",
    "overall_false_positive_rate",
    "worst_malicious_source_id",
    "worst_malicious_source_name",
    "worst_malicious_metric_basis",
    "worst_malicious_recall",
    "worst_malicious_pr_auc",
    "worst_malicious_f1",
)
"""Stable column order for single-run comparison-ready summaries."""


def _json_default(value: object) -> object:
    """Serialize non-JSON-native values into JSON-friendly payloads."""

    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()  # type: ignore[call-arg]
        except TypeError:
            pass
    return str(value)


def _write_json(path: Path, payload: object) -> None:
    """Write a stable JSON file with a trailing newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> int:
    """Write rows to JSONL with stable field order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(dict(row), handle, ensure_ascii=False, sort_keys=False, default=_json_default)
            handle.write("\n")
            count += 1
    return count


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> int:
    """Write rows to CSV with a stable column order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=list(columns))
    frame.to_csv(path, index=False)
    return len(frame)


def _safe_metric_value(value: float | None) -> float | None:
    """Normalize metric values for JSON and CSV export."""

    return None if value is None else float(value)


def _false_positive_rate(metrics: BinaryScoreMetrics) -> float | None:
    """Compute the false-positive rate from a binary metric summary."""

    if metrics.negative_count <= 0:
        return None
    return float(metrics.false_positive) / float(metrics.negative_count)


def _metric_payload(metrics: BinaryScoreMetrics) -> dict[str, float | int | None]:
    """Convert binary metrics into the repository's report schema."""

    return {
        "support": metrics.support,
        "positive_count": metrics.positive_count,
        "negative_count": metrics.negative_count,
        "roc_auc": _safe_metric_value(metrics.roc_auc),
        "pr_auc": _safe_metric_value(metrics.pr_auc),
        "precision": _safe_metric_value(metrics.precision),
        "recall": _safe_metric_value(metrics.recall),
        "f1": _safe_metric_value(metrics.f1),
        "false_positive_rate": _false_positive_rate(metrics),
        "true_positive": metrics.true_positive,
        "false_positive": metrics.false_positive,
        "true_negative": metrics.true_negative,
        "false_negative": metrics.false_negative,
        "threshold": float(metrics.threshold),
    }


def _normalize_paths(paths: Sequence[str | Path] | None) -> tuple[Path, ...]:
    """Normalize input paths into a stable tuple of :class:`Path` objects."""

    if not paths:
        return ()
    normalized: list[Path] = []
    for raw_path in paths:
        candidate = Path(raw_path)
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        normalized.append(candidate)
    return tuple(normalized)


def _build_source_id(
    path: Path,
    *,
    source_role: PcapSourceRole,
    source_index: int,
) -> str:
    """Build a stable source identifier for one input invocation."""

    return f"{source_role}-{source_index:02d}-{_slugify_token(path.stem)}"


def _empty_score_summary() -> dict[str, float | int]:
    """Return a stable empty quantile payload."""

    return {
        "count": 0,
        "mean": 0.0,
        "median": 0.0,
        "q90": 0.0,
        "q95": 0.0,
        "max": 0.0,
    }


def _score_summary_from_values(values: Sequence[float]) -> dict[str, float | int]:
    """Summarize score values with a stable empty fallback."""

    return _empty_score_summary() if not values else _quantile_summary(values)


def _scorer_type_from_backend(backend: str) -> str:
    """Map the internal backend token to a comparison-friendly scorer name."""

    return "deterministic_fallback" if backend == "fallback" else backend


def _comparison_metric_csv_value(value: float | int | None) -> float | int | str:
    """Render comparison metrics with explicit unavailable markers for CSV export."""

    return "unavailable" if value is None else value


@dataclass(slots=True)
class PcapGraphExperimentConfig:
    """Configuration for the reproducible real-PCAP experiment runner."""

    packet_limit: int | None = 5000
    packet_sampling_mode: PcapPacketSamplingMode = "random_window"
    idle_timeout_seconds: float = 60.0
    window_size: int = 60
    short_flow_thresholds: ShortFlowThresholds = field(
        default_factory=ShortFlowThresholds
    )
    use_association_edges: bool = True
    use_graph_structural_features: bool = True
    smoke_graph_limit: int = 16
    benign_train_ratio: float = 0.7
    train_validation_ratio: float = 0.25
    graph_score_reduction: PcapGraphScoreReduction = "hybrid_max_rank_flow_node_max"
    epochs: int = 2
    batch_size: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    threshold_percentile: float = 95.0
    random_seed: int = 42
    hidden_dim: int = 64
    latent_dim: int = 32
    num_layers: int = 2
    dropout: float = 0.1
    use_edge_features: bool = True
    reconstruct_edge_features: bool = True
    use_edge_categorical_embeddings: bool = True
    edge_categorical_embedding_dim: int = 8
    edge_categorical_bucket_size: int = 128
    checkpoint_dir: str = "artifacts/checkpoints"

    def with_checkpoint_directory(self, checkpoint_dir: str | Path) -> "PcapGraphExperimentConfig":
        """Return a copy of the config with an updated checkpoint directory."""

        return replace(self, checkpoint_dir=Path(checkpoint_dir).as_posix())

    def to_pipeline_config(
        self,
        *,
        input_path: str,
        run_name: str,
        output_directory: str,
    ) -> PipelineConfig:
        """Build a pipeline config for the shared GAE trainer."""

        return PipelineConfig(
            pipeline=PipelineRuntimeConfig(run_name=run_name, seed=self.random_seed),
            data=DataConfig(input_path=input_path, format="pcap"),
            preprocessing=PreprocessingConfig(
                window_size=self.window_size,
                short_flow_thresholds=self.short_flow_thresholds,
            ),
            graph=GraphConfig(
                time_window_seconds=self.window_size,
                directed=True,
                association_edges=AssociationEdgeConfig(
                    enable_same_src_ip=self.use_association_edges,
                    enable_same_dst_subnet=self.use_association_edges,
                    dst_subnet_prefix=24,
                ),
            ),
            features=FeaturesConfig(
                normalization=FeatureNormalizationConfig(),
                use_graph_structural_features=self.use_graph_structural_features,
            ),
            model=ModelConfig(
                name="graph-autoencoder",
                device="cpu",
                hidden_dim=self.hidden_dim,
                latent_dim=self.latent_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
                use_edge_features=self.use_edge_features,
                reconstruct_edge_features=self.reconstruct_edge_features,
                use_edge_categorical_embeddings=self.use_edge_categorical_embeddings,
                edge_categorical_embedding_dim=self.edge_categorical_embedding_dim,
                edge_categorical_bucket_size=self.edge_categorical_bucket_size,
            ),
            training=TrainingConfig(
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                batch_size=self.batch_size,
                validation_split_ratio=self.train_validation_ratio,
                early_stopping_patience=max(1, self.epochs),
                checkpoint_dir=self.checkpoint_dir,
                shuffle=True,
                seed=self.random_seed,
                smoke_graph_limit=self.smoke_graph_limit,
            ),
            evaluation=EvaluationConfig(),
            alerting=AlertingConfig(),
            output=OutputConfig(directory=output_directory, save_intermediate=True),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the config into a plain JSON-friendly dictionary."""

        return {
            "packet_limit": self.packet_limit,
            "packet_sampling_mode": self.packet_sampling_mode,
            "idle_timeout_seconds": self.idle_timeout_seconds,
            "window_size": self.window_size,
            "short_flow_thresholds": {
                "packet_count_lt": self.short_flow_thresholds.packet_count_lt,
                "byte_count_lt": self.short_flow_thresholds.byte_count_lt,
                "duration_seconds_lt": self.short_flow_thresholds.duration_seconds_lt,
            },
            "use_association_edges": self.use_association_edges,
            "use_graph_structural_features": self.use_graph_structural_features,
            "smoke_graph_limit": self.smoke_graph_limit,
            "benign_train_ratio": self.benign_train_ratio,
            "train_validation_ratio": self.train_validation_ratio,
            "graph_score_reduction": self.graph_score_reduction,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "threshold_percentile": self.threshold_percentile,
            "random_seed": self.random_seed,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "use_edge_features": self.use_edge_features,
            "reconstruct_edge_features": self.reconstruct_edge_features,
            "use_edge_categorical_embeddings": self.use_edge_categorical_embeddings,
            "edge_categorical_embedding_dim": self.edge_categorical_embedding_dim,
            "edge_categorical_bucket_size": self.edge_categorical_bucket_size,
            "checkpoint_dir": self.checkpoint_dir,
        }


@dataclass(slots=True)
class PcapGraphSourceArtifact:
    """In-memory representation of one parsed PCAP source."""

    source_id: str
    source_path: str
    source_name: str
    source_role: PcapSourceRole
    raw_label: str
    binary_label: int
    parse_summary: PcapFlowLoadSummary
    window_statistics: list[LogicalFlowWindowStats]
    graphs: list[InteractionGraph]


@dataclass(slots=True)
class PcapGraphEntry:
    """One graph sample plus the source metadata needed for evaluation."""

    graph: InteractionGraph
    source_id: str
    source_path: str
    source_name: str
    source_role: PcapSourceRole
    raw_label: str
    binary_label: int
    entry_index: int
    split_assignment: str = "unassigned"

    @property
    def graph_id(self) -> str:
        """Return a stable graph identifier for artifacts and summaries."""

        return f"{self.source_role}:{self.source_id}:{self.graph.window_index}:{self.entry_index}"

    @property
    def attack_group(self) -> str:
        """Return the attack-group token used by binary score reports."""

        if self.binary_label == 0:
            return "BENIGN"
        return self.source_name


@dataclass(slots=True)
class PcapGraphExperimentResult:
    """Structured result returned by the mini real-PCAP experiment runner."""

    run_id: str
    timestamp: str
    mode: PcapGraphExperimentMode
    backend: str
    config: PcapGraphExperimentConfig
    summary: dict[str, object]
    export_result: RunBundleExportResult
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Render a compact text summary for CLI output."""

        lines = [
            "PCAP graph experiment summary:",
            f"Run id: {self.run_id}",
            f"Timestamp: {self.timestamp}",
            f"Mode: {self.mode}",
            f"Backend: {self.backend}",
            (
                "Inputs: "
                f"benign_files={len(self.summary.get('benign_inputs', []))}, "
                f"malicious_files={len(self.summary.get('malicious_inputs', []))}"
            ),
            (
                "Counts: "
                f"packets={int(self.summary.get('total_packets', 0))}, "
                f"flows={int(self.summary.get('total_flows', 0))}, "
                f"graphs={int(self.summary.get('total_graphs', 0))}, "
                f"benign_graphs={int(self.summary.get('benign_graph_count', 0))}, "
                f"malicious_graphs={int(self.summary.get('malicious_graph_count', 0))}"
            ),
            (
                "Graph config: "
                f"window_size={int(self.summary.get('window_size', 0))}, "
                f"use_association_edges={bool(self.summary.get('use_association_edges', False))}, "
                "use_graph_structural_features="
                f"{bool(self.summary.get('use_graph_structural_features', False))}, "
                "graph_score_reduction="
                f"{self.summary.get('graph_score_reduction', 'mean_node')}"
            ),
            (
                "Graph score quantiles: "
                f"count={int(self.summary.get('graph_score_quantiles', {}).get('count', 0))}, "
                f"median={float(self.summary.get('graph_score_quantiles', {}).get('median', 0.0)):.6f}, "
                f"q95={float(self.summary.get('graph_score_quantiles', {}).get('q95', 0.0)):.6f}"
            ),
            f"Exported directory: {self.export_result.run_directory}",
        ]
        experiment_label = self.summary.get("experiment_label")
        if experiment_label:
            lines.insert(5, f"Experiment label: {experiment_label}")
        split_graph_counts = self.summary.get("split_graph_counts", {})
        if isinstance(split_graph_counts, Mapping):
            lines.append(
                "Split counts: "
                f"train={split_graph_counts.get('train', 0)}, "
                f"val={split_graph_counts.get('val', 0)}, "
                f"benign_test={split_graph_counts.get('benign_test', 0)}, "
                f"malicious_test={split_graph_counts.get('malicious_test', 0)}, "
                f"smoke={split_graph_counts.get('smoke', 0)}"
            )
        if self.mode == "binary_evaluation":
            overall_metrics = self.summary.get("overall_metrics", {})
            if isinstance(overall_metrics, Mapping):
                lines.append(
                    "Overall metrics: "
                    f"roc_auc={overall_metrics.get('roc_auc')}, "
                    f"pr_auc={overall_metrics.get('pr_auc')}, "
                    f"precision={overall_metrics.get('precision')}, "
                    f"recall={overall_metrics.get('recall')}, "
                    f"f1={overall_metrics.get('f1')}, "
                    "false_positive_rate="
                    f"{overall_metrics.get('false_positive_rate')}"
                )
        if self.notes:
            lines.append("Notes:")
            lines.extend(f"  - {note}" for note in self.notes)
        return "\n".join(lines)


def _load_source_artifact(
    path: Path,
    *,
    source_id: str,
    source_role: PcapSourceRole,
    config: PcapGraphExperimentConfig,
) -> PcapGraphSourceArtifact:
    """Parse one PCAP source and convert it into windowed endpoint graphs."""

    start_packet_offset = 0
    if config.packet_sampling_mode == "random_window" and config.packet_limit is not None:
        total_packet_records, _truncated = inspect_classic_pcap(path)
        if total_packet_records > config.packet_limit:
            max_start = total_packet_records - config.packet_limit
            seed_value = sum(ord(char) for char in f"{config.random_seed}:{source_id}:{path.name}")
            rng = np.random.default_rng(seed_value)
            start_packet_offset = int(rng.integers(0, max_start + 1))

    load_result = load_pcap_flow_dataset(
        path,
        max_packets=config.packet_limit,
        start_packet_offset=start_packet_offset,
        idle_timeout_seconds=config.idle_timeout_seconds,
    )
    window_batches = preprocess_flow_dataset(
        load_result.dataset,
        window_size=config.window_size,
        rules=config.short_flow_thresholds,
    )
    graph_builder = FlowInteractionGraphBuilder(
        GraphConfig(
            time_window_seconds=config.window_size,
            directed=True,
            association_edges=AssociationEdgeConfig(
                enable_same_src_ip=config.use_association_edges,
                enable_same_dst_subnet=config.use_association_edges,
                dst_subnet_prefix=24,
            ),
        )
    )
    graphs = graph_builder.build_many(window_batches)
    if config.smoke_graph_limit > 0:
        graphs = graphs[: config.smoke_graph_limit]
    if not graphs:
        raise ValueError(
            f"The PCAP source {path.as_posix()} did not yield any endpoint graphs."
        )
    return PcapGraphSourceArtifact(
        source_id=source_id,
        source_path=path.as_posix(),
        source_name=path.stem,
        source_role=source_role,
        raw_label="BENIGN" if source_role == "benign" else path.stem,
        binary_label=0 if source_role == "benign" else 1,
        parse_summary=load_result.summary,
        window_statistics=[batch.stats for batch in window_batches],
        graphs=graphs,
    )


def _flatten_source_artifacts(
    artifacts: Sequence[PcapGraphSourceArtifact],
) -> list[PcapGraphEntry]:
    """Flatten parsed source artifacts into per-graph entries."""

    entries: list[PcapGraphEntry] = []
    for artifact in artifacts:
        for entry_index, graph in enumerate(artifact.graphs):
            entries.append(
                PcapGraphEntry(
                    graph=graph,
                    source_id=artifact.source_id,
                    source_path=artifact.source_path,
                    source_name=artifact.source_name,
                    source_role=artifact.source_role,
                    raw_label=artifact.raw_label,
                    binary_label=artifact.binary_label,
                    entry_index=entry_index,
                )
            )
    return entries


def _partition_benign_entries(
    benign_entries: Sequence[PcapGraphEntry],
    *,
    benign_train_ratio: float,
    random_seed: int,
) -> tuple[list[PcapGraphEntry], list[PcapGraphEntry]]:
    """Split benign graph entries into train-pool and benign-test partitions."""

    if not benign_entries:
        return [], []
    if len(benign_entries) == 1:
        train_only = list(benign_entries)
        train_only[0].split_assignment = "train"
        return train_only, []

    rng = np.random.default_rng(random_seed)
    permutation = list(rng.permutation(len(benign_entries)))
    train_count = int(round(len(benign_entries) * benign_train_ratio))
    train_count = max(1, min(len(benign_entries) - 1, train_count))
    train_indices = set(permutation[:train_count])
    train_entries: list[PcapGraphEntry] = []
    test_entries: list[PcapGraphEntry] = []
    for index, entry in enumerate(benign_entries):
        if index in train_indices:
            train_entries.append(entry)
        else:
            test_entries.append(entry)
    return train_entries, test_entries


def _mark_assignments(entries: Sequence[PcapGraphEntry], assignment: str) -> None:
    """Assign a stable split label to each graph entry."""

    for entry in entries:
        entry.split_assignment = assignment


def _feature_count(packed_graph: PackedGraphInput) -> int:
    """Return the node feature dimensionality used by the scorer."""

    return int(packed_graph.node_feature_dim)


def _contextualize_score_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    scope: str,
    entries: Sequence[PcapGraphEntry],
    packed_graphs: Sequence[PackedGraphInput],
    mode: PcapGraphExperimentMode,
    backend: str,
    task_name: str,
) -> list[dict[str, object]]:
    """Attach source metadata and labels to raw score-table rows."""

    contextual_rows: list[dict[str, object]] = []
    for raw_row in rows:
        graph_index = int(raw_row.get("graph_index", 0))
        entry = entries[graph_index]
        packed_graph = packed_graphs[graph_index]
        row = dict(raw_row)
        row.update(
            {
                "graph_id": entry.graph_id,
                "sample_id": entry.graph_id,
                "source_id": entry.source_id,
                "task_name": task_name,
                "evaluation_scope": scope,
                "source_name": entry.source_name,
                "source_path": entry.source_path,
                "source_role": entry.source_role,
                "raw_label": entry.raw_label,
                "binary_label": entry.binary_label,
                "label": entry.binary_label if entry.binary_label in {0, 1} else None,
                "attack_group": entry.attack_group,
                "split_assignment": entry.split_assignment,
                "feature_count": _feature_count(packed_graph),
                "mode": mode,
                "backend": backend,
            }
        )
        contextual_rows.append(row)
    return contextual_rows


def _graph_summary_row(entry: PcapGraphEntry) -> dict[str, object]:
    """Serialize one graph entry into a stable summary row."""

    summary = summarize_graph(entry.graph)
    return {
        "graph_id": entry.graph_id,
        "source_id": entry.source_id,
        "split_assignment": entry.split_assignment,
        "source_role": entry.source_role,
        "source_name": entry.source_name,
        "source_path": entry.source_path,
        "raw_label": entry.raw_label,
        "binary_label": entry.binary_label,
        "window_index": entry.graph.window_index,
        "window_start": entry.graph.window_start.isoformat(),
        "window_end": entry.graph.window_end.isoformat(),
        "node_count": int(summary["node_count"]),
        "edge_count": int(summary["edge_count"]),
        "client_node_count": int(summary["client_node_count"]),
        "server_node_count": int(summary["server_node_count"]),
        "aggregated_edge_count": int(summary["aggregated_edge_count"]),
        "communication_edge_count": int(summary["communication_edge_count"]),
        "association_edge_count": int(summary["association_edge_count"]),
        "association_same_src_ip_edge_count": int(
            summary["association_same_src_ip_edge_count"]
        ),
        "association_same_dst_subnet_edge_count": int(
            summary["association_same_dst_subnet_edge_count"]
        ),
    }


def _score_record_from_row(
    row: Mapping[str, object],
    *,
    run_id: str,
    timestamp: str,
    threshold: float,
    split: str,
    task_name: str,
) -> BinaryDetectionScoreRecord:
    """Convert one contextualized graph-score row into a score record."""

    graph_id = str(row.get("graph_id") or row.get("sample_id") or "graph")
    graph_index = int(row.get("graph_index", 0))
    anomaly_score = float(row.get("graph_anomaly_score", row.get("anomaly_score", 0.0)))
    binary_label_value = row.get("binary_label")
    binary_label = int(binary_label_value) if binary_label_value is not None else -1
    metadata = {
        "graph_id": graph_id,
        "source_name": row.get("source_name"),
        "source_path": row.get("source_path"),
        "source_role": row.get("source_role"),
        "split_assignment": row.get("split_assignment"),
        "window_index": row.get("window_index"),
        "window_start": row.get("window_start"),
        "window_end": row.get("window_end"),
        "node_count": row.get("node_count"),
        "edge_count": row.get("edge_count"),
        "association_edge_count": row.get("association_edge_count"),
        "backend": row.get("backend"),
        "mode": row.get("mode"),
    }
    return BinaryDetectionScoreRecord(
        score_id=f"{task_name}:{graph_id}",
        run_id=run_id,
        timestamp=timestamp,
        split=split,  # type: ignore[arg-type]
        evaluation_scope="graph",
        task_name=task_name,
        sample_id=graph_id,
        row_index=graph_index,
        raw_label=str(row.get("raw_label", "")),
        binary_label=binary_label,
        attack_group=str(row.get("attack_group", "")),
        anomaly_score=anomaly_score,
        threshold=float(threshold),
        is_alert=anomaly_score >= float(threshold),
        feature_count=int(row.get("feature_count", 0)),
        metadata=metadata,
    )


def _build_attack_metric_record(
    *,
    task_name: str,
    records: Sequence[BinaryDetectionScoreRecord],
    threshold: float,
    notes: Sequence[str] = (),
) -> BinaryAttackMetricRecord:
    """Compute one per-attack metric record from score records."""

    labels = [record.binary_label if record.binary_label in {0, 1} else None for record in records]
    scores = [record.anomaly_score for record in records]
    metrics = evaluate_scores(labels, scores, threshold=threshold)
    score_summary = _quantile_summary(scores)
    benign_scores = [record.anomaly_score for record in records if record.binary_label == 0]
    malicious_scores = [record.anomaly_score for record in records if record.binary_label == 1]
    benign_summary = _quantile_summary(benign_scores)
    malicious_summary = _quantile_summary(malicious_scores)
    attack_labels = sorted(
        {
            record.attack_group
            for record in records
            if record.binary_label == 1 and record.attack_group
        }
    )
    return BinaryAttackMetricRecord(
        task_name=task_name,
        requested_attack_type=task_name,
        attack_labels=tuple(attack_labels or [task_name]),
        sample_count=len(records),
        benign_count=sum(record.binary_label == 0 for record in records),
        attack_count=sum(record.binary_label == 1 for record in records),
        roc_auc=_safe_metric_value(metrics.roc_auc),
        pr_auc=_safe_metric_value(metrics.pr_auc),
        precision=float(metrics.precision or 0.0),
        recall=float(metrics.recall or 0.0),
        f1=float(metrics.f1 or 0.0),
        false_positive_rate=_false_positive_rate(metrics),
        threshold=float(threshold),
        score_min=float(score_summary["min"]),
        score_q25=float(score_summary["q25"]),
        score_median=float(score_summary["median"]),
        score_q75=float(score_summary["q75"]),
        score_q95=float(score_summary["q95"]),
        score_max=float(score_summary["max"]),
        score_mean=float(score_summary["mean"]),
        score_std=float(score_summary["std"]),
        benign_score_mean=float(benign_summary["mean"]),
        benign_score_median=float(benign_summary["median"]),
        attack_score_mean=float(malicious_summary["mean"]),
        attack_score_median=float(malicious_summary["median"]),
        notes=tuple(notes),
    )


def _export_score_records(
    *,
    records: Sequence[BinaryDetectionScoreRecord],
    base_path: Path,
    artifact_paths: dict[str, str],
    row_counts: dict[str, int],
    artifact_name: str,
) -> None:
    """Export score records to root-level JSONL and CSV files."""

    jsonl_path = base_path.with_suffix(".jsonl")
    csv_path = base_path.with_suffix(".csv")
    row_counts[f"{artifact_name}_jsonl"] = _write_jsonl(
        jsonl_path,
        [record.to_dict() for record in records],
    )
    artifact_paths[f"{artifact_name}_jsonl"] = jsonl_path.as_posix()
    row_counts[f"{artifact_name}_csv"] = _write_csv(
        csv_path,
        [record.to_csv_dict() for record in records],
        BINARY_DETECTION_SCORE_FIELDS,
    )
    artifact_paths[f"{artifact_name}_csv"] = csv_path.as_posix()


def _update_manifest(
    manifest_path: Path,
    *,
    extra_artifact_paths: Mapping[str, str],
    extra_row_counts: Mapping[str, int],
    extra_notes: Sequence[str],
    extra_payload: Mapping[str, object],
) -> None:
    """Merge additional artifact metadata into an existing run manifest."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("The exported manifest payload is not a mapping.")
    artifact_paths = payload.setdefault("artifact_paths", {})
    if isinstance(artifact_paths, dict):
        artifact_paths.update(extra_artifact_paths)
    row_counts = payload.setdefault("row_counts", {})
    if isinstance(row_counts, dict):
        row_counts.update(extra_row_counts)
    notes = payload.setdefault("notes", [])
    if isinstance(notes, list):
        for note in extra_notes:
            if note not in notes:
                notes.append(note)
    for key, value in extra_payload.items():
        payload[key] = value
    _write_json(manifest_path, payload)


def _aggregate_source_counts(
    source_artifacts: Sequence[PcapGraphSourceArtifact],
) -> dict[str, object]:
    """Aggregate packet, flow, and graph counts across all parsed sources."""

    skip_reason_counts: Counter[str] = Counter()
    protocol_counts: Counter[str] = Counter()
    for artifact in source_artifacts:
        skip_reason_counts.update(artifact.parse_summary.skipped_reason_counts)
        protocol_counts.update(artifact.parse_summary.flow_dataset_summary.protocols)
    return {
        "source_count": len(source_artifacts),
        "total_packets": sum(artifact.parse_summary.total_packets for artifact in source_artifacts),
        "parsed_packets": sum(artifact.parse_summary.parsed_packets for artifact in source_artifacts),
        "skipped_packets": sum(artifact.parse_summary.skipped_packets for artifact in source_artifacts),
        "total_flows": sum(artifact.parse_summary.total_flows for artifact in source_artifacts),
        "total_graphs": sum(len(artifact.graphs) for artifact in source_artifacts),
        "skip_reason_counts": dict(skip_reason_counts),
        "protocols": sorted(protocol_counts.keys()),
    }


def _group_entries_by_source_id(
    entries: Sequence[PcapGraphEntry],
) -> dict[str, list[PcapGraphEntry]]:
    """Group graph entries by their unique source identifier."""

    grouped: dict[str, list[PcapGraphEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.source_id, []).append(entry)
    return grouped


def _group_rows_by_source_id(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, list[Mapping[str, object]]]:
    """Group contextual score rows by source identifier."""

    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        source_id = str(row.get("source_id", ""))
        grouped.setdefault(source_id, []).append(row)
    return grouped


def _assignment_counts(entries: Sequence[PcapGraphEntry]) -> dict[str, int]:
    """Count graph assignments per split for one source or experiment."""

    counts = Counter(entry.split_assignment for entry in entries)
    return {split_name: int(counts.get(split_name, 0)) for split_name in sorted(counts)}


def _graph_ids_by_split(entries: Sequence[PcapGraphEntry]) -> dict[str, list[str]]:
    """Return graph identifiers grouped by split assignment."""

    grouped: dict[str, list[str]] = {}
    for entry in entries:
        grouped.setdefault(entry.split_assignment, []).append(entry.graph_id)
    for graph_ids in grouped.values():
        graph_ids.sort()
    return grouped


def _split_score_row(
    *,
    split_name: str,
    graph_count: int,
    score_summary: Mapping[str, float | int],
    availability: str,
) -> dict[str, object]:
    """Build one flat split-level score summary row."""

    return {
        "split_name": split_name,
        "score_scope": "graph",
        "availability": availability,
        "graph_count": int(graph_count),
        "score_count": int(score_summary.get("count", 0)),
        "score_mean": float(score_summary.get("mean", 0.0)),
        "score_median": float(score_summary.get("median", 0.0)),
        "score_p90": float(score_summary.get("q90", 0.0)),
        "score_p95": float(score_summary.get("q95", 0.0)),
        "score_max": float(score_summary.get("max", 0.0)),
    }


def _build_source_summary_rows(
    *,
    source_artifacts: Sequence[PcapGraphSourceArtifact],
    all_entries: Sequence[PcapGraphEntry],
    contextual_graph_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Build rich per-source summaries for JSON artifacts and reports."""

    entries_by_source = _group_entries_by_source_id(all_entries)
    rows_by_source = _group_rows_by_source_id(contextual_graph_rows)
    summary_rows: list[dict[str, object]] = []
    for artifact in source_artifacts:
        source_entries = entries_by_source.get(artifact.source_id, [])
        source_rows = rows_by_source.get(artifact.source_id, [])
        score_summary = _score_summary_from_values(
            [_score_from_row(row) for row in source_rows]
        )
        availability = (
            "available"
            if source_rows
            else "unavailable_no_scored_graphs_for_source"
        )
        summary_rows.append(
            {
                "source_id": artifact.source_id,
                "source_path": artifact.source_path,
                "source_name": artifact.source_name,
                "source_role": artifact.source_role,
                "raw_label": artifact.raw_label,
                "binary_label": artifact.binary_label,
                "parse_summary": artifact.parse_summary.to_dict(),
                "window_statistics": [asdict(stats) for stats in artifact.window_statistics],
                "graph_count": len(artifact.graphs),
                "assignment_counts": _assignment_counts(source_entries),
                "graph_ids_by_split": _graph_ids_by_split(source_entries),
                "scored_graph_count": len(source_rows),
                "score_summary_availability": availability,
                "score_summary": score_summary,
            }
        )
    return summary_rows


def _build_source_score_summary_rows(
    source_summary_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Flatten rich source summaries into a CSV-friendly score summary table."""

    flat_rows: list[dict[str, object]] = []
    for row in source_summary_rows:
        assignment_counts = row.get("assignment_counts", {})
        if not isinstance(assignment_counts, Mapping):
            assignment_counts = {}
        score_summary = row.get("score_summary", {})
        if not isinstance(score_summary, Mapping):
            score_summary = _empty_score_summary()
        flat_rows.append(
            {
                "source_id": row.get("source_id"),
                "source_name": row.get("source_name"),
                "source_path": row.get("source_path"),
                "source_role": row.get("source_role"),
                "raw_label": row.get("raw_label"),
                "binary_label": row.get("binary_label"),
                "score_scope": "graph",
                "availability": row.get("score_summary_availability"),
                "graph_count": int(row.get("graph_count", 0)),
                "scored_graph_count": int(row.get("scored_graph_count", 0)),
                "train_graph_count": int(assignment_counts.get("train", 0)),
                "val_graph_count": int(assignment_counts.get("val", 0)),
                "benign_test_graph_count": int(assignment_counts.get("benign_test", 0)),
                "malicious_test_graph_count": int(
                    assignment_counts.get("malicious_test", 0)
                ),
                "smoke_graph_count": int(assignment_counts.get("smoke", 0)),
                "score_count": int(score_summary.get("count", 0)),
                "score_mean": float(score_summary.get("mean", 0.0)),
                "score_median": float(score_summary.get("median", 0.0)),
                "score_p90": float(score_summary.get("q90", 0.0)),
                "score_p95": float(score_summary.get("q95", 0.0)),
                "score_max": float(score_summary.get("max", 0.0)),
            }
        )
    return flat_rows


def _build_split_score_summary_rows(
    *,
    mode: PcapGraphExperimentMode,
    train_graph_count: int,
    train_score_summary: Mapping[str, float | int],
    contextual_graph_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Build split-level graph-score summaries for analysis-friendly exports."""

    split_rows: list[dict[str, object]] = [
        _split_score_row(
            split_name=(
                "benign_train_reference" if mode == "binary_evaluation" else "smoke_train_reference"
            ),
            graph_count=train_graph_count,
            score_summary=train_score_summary,
            availability="available" if train_graph_count > 0 else "unavailable_no_train_graphs",
        )
    ]

    if mode == "binary_evaluation":
        benign_test_rows = [
            row for row in contextual_graph_rows if int(row.get("binary_label", -1)) == 0
        ]
        malicious_test_rows = [
            row for row in contextual_graph_rows if int(row.get("binary_label", -1)) == 1
        ]
        split_rows.extend(
            [
                _split_score_row(
                    split_name="benign_test",
                    graph_count=len(benign_test_rows),
                    score_summary=_score_summary_from_values(
                        [_score_from_row(row) for row in benign_test_rows]
                    ),
                    availability=(
                        "available"
                        if benign_test_rows
                        else "unavailable_no_benign_test_graphs"
                    ),
                ),
                _split_score_row(
                    split_name="malicious_test",
                    graph_count=len(malicious_test_rows),
                    score_summary=_score_summary_from_values(
                        [_score_from_row(row) for row in malicious_test_rows]
                    ),
                    availability=(
                        "available"
                        if malicious_test_rows
                        else "unavailable_no_malicious_test_graphs"
                    ),
                ),
                _split_score_row(
                    split_name="overall_test",
                    graph_count=len(contextual_graph_rows),
                    score_summary=_score_summary_from_values(
                        [_score_from_row(row) for row in contextual_graph_rows]
                    ),
                    availability=(
                        "available"
                        if contextual_graph_rows
                        else "unavailable_no_evaluation_graphs"
                    ),
                ),
            ]
        )
    else:
        split_rows.append(
            _split_score_row(
                split_name="smoke_evaluation",
                graph_count=len(contextual_graph_rows),
                score_summary=_score_summary_from_values(
                    [_score_from_row(row) for row in contextual_graph_rows]
                ),
                availability=(
                    "available" if contextual_graph_rows else "unavailable_no_smoke_graphs"
                ),
            )
        )
    return split_rows


def _build_malicious_source_metric_rows(
    *,
    contextual_graph_rows: Sequence[Mapping[str, object]],
    threshold: float,
) -> list[dict[str, object]]:
    """Build per-malicious-source binary metrics against benign holdout graphs."""

    benign_rows = [
        row for row in contextual_graph_rows if int(row.get("binary_label", -1)) == 0
    ]
    malicious_rows_by_source = _group_rows_by_source_id(
        [
            row
            for row in contextual_graph_rows
            if int(row.get("binary_label", -1)) == 1
        ]
    )
    metric_rows: list[dict[str, object]] = []
    for source_id in sorted(malicious_rows_by_source):
        malicious_rows = malicious_rows_by_source[source_id]
        task_rows = benign_rows + malicious_rows
        labels = [int(row.get("binary_label", 0)) for row in task_rows]
        scores = [_score_from_row(row) for row in task_rows]
        metrics = evaluate_scores(labels, scores, threshold=threshold)
        malicious_scores = [_score_from_row(row) for row in malicious_rows]
        first_row = malicious_rows[0]
        malicious_score_summary = _score_summary_from_values(malicious_scores)
        metric_rows.append(
            {
                "source_id": source_id,
                "source_name": first_row.get("source_name"),
                "source_path": first_row.get("source_path"),
                "raw_label": first_row.get("raw_label"),
                "benign_graph_count": len(benign_rows),
                "malicious_graph_count": len(malicious_rows),
                "support": metrics.support,
                "positive_count": metrics.positive_count,
                "negative_count": metrics.negative_count,
                "roc_auc": _safe_metric_value(metrics.roc_auc),
                "pr_auc": _safe_metric_value(metrics.pr_auc),
                "precision": _safe_metric_value(metrics.precision),
                "recall": _safe_metric_value(metrics.recall),
                "f1": _safe_metric_value(metrics.f1),
                "false_positive_rate": _false_positive_rate(metrics),
                "threshold": float(threshold),
                "score_mean": float(malicious_score_summary.get("mean", 0.0)),
                "score_median": float(malicious_score_summary.get("median", 0.0)),
                "score_p90": float(malicious_score_summary.get("q90", 0.0)),
                "score_p95": float(malicious_score_summary.get("q95", 0.0)),
                "score_max": float(malicious_score_summary.get("max", 0.0)),
            }
        )
    return metric_rows


def _build_train_graph_score_rows(
    *,
    contextual_train_rows: Sequence[Mapping[str, object]],
    run_id: str,
    timestamp: str,
    threshold: float,
    backend: str,
    split_name: str,
) -> list[dict[str, object]]:
    """Serialize train-reference graph scores into a stable export schema."""

    export_rows: list[dict[str, object]] = []
    for row in contextual_train_rows:
        score = float(row.get("graph_anomaly_score", row.get("anomaly_score", 0.0)))
        export_rows.append(
            {
                "run_id": run_id,
                "timestamp": timestamp,
                "graph_id": str(row.get("graph_id") or row.get("sample_id") or "graph"),
                "source_id": str(row.get("source_id", "")),
                "source_name": str(row.get("source_name", "")),
                "source_path": str(row.get("source_path", "")),
                "source_role": str(row.get("source_role", "")),
                "split_name": split_name,
                "label": "benign_reference",
                "binary_label": int(row.get("binary_label", 0)),
                "score": score,
                "threshold": float(threshold),
                "is_alert": score >= float(threshold),
                "backend": backend,
                "graph_score_reduction": str(row.get("graph_score_reduction", "mean_node")),
                "node_score_count": int(row.get("node_score_count", 0)),
                "node_score_mean": float(row.get("node_score_mean", 0.0)),
                "node_score_p75": float(row.get("node_score_p75", 0.0)),
                "node_score_p90": float(row.get("node_score_p90", 0.0)),
                "node_score_max": float(row.get("node_score_max", 0.0)),
                "node_score_topk_mean": float(row.get("node_score_topk_mean", 0.0)),
                "edge_score_count": int(row.get("edge_score_count", 0)),
                "edge_score_mean": float(row.get("edge_score_mean", 0.0)),
                "edge_score_p75": float(row.get("edge_score_p75", 0.0)),
                "edge_score_p90": float(row.get("edge_score_p90", 0.0)),
                "edge_score_max": float(row.get("edge_score_max", 0.0)),
                "edge_score_topk_mean": float(row.get("edge_score_topk_mean", 0.0)),
                "flow_score_count": int(row.get("flow_score_count", 0)),
                "flow_score_mean": float(row.get("flow_score_mean", 0.0)),
                "flow_score_p75": float(row.get("flow_score_p75", 0.0)),
                "flow_score_p90": float(row.get("flow_score_p90", 0.0)),
                "flow_score_max": float(row.get("flow_score_max", 0.0)),
                "flow_score_topk_mean": float(row.get("flow_score_topk_mean", 0.0)),
                "flow_duration_p75": float(row.get("flow_duration_p75", 0.0)),
                "flow_duration_topk_mean": float(row.get("flow_duration_topk_mean", 0.0)),
                "flow_iat_proxy_mean": float(row.get("flow_iat_proxy_mean", 0.0)),
                "flow_iat_proxy_std": float(row.get("flow_iat_proxy_std", 0.0)),
                "flow_iat_proxy_p75": float(row.get("flow_iat_proxy_p75", 0.0)),
                "flow_iat_proxy_topk_mean": float(
                    row.get("flow_iat_proxy_topk_mean", 0.0)
                ),
                "flow_pkt_count_p75": float(row.get("flow_pkt_count_p75", 0.0)),
                "flow_pkt_count_topk_mean": float(
                    row.get("flow_pkt_count_topk_mean", 0.0)
                ),
                "flow_pkt_rate_p90": float(row.get("flow_pkt_rate_p90", 0.0)),
                "flow_pkt_rate_topk_mean": float(row.get("flow_pkt_rate_topk_mean", 0.0)),
                "short_flow_score_count": int(row.get("short_flow_score_count", 0)),
                "short_flow_score_mean": float(row.get("short_flow_score_mean", 0.0)),
                "short_flow_score_p75": float(row.get("short_flow_score_p75", 0.0)),
                "short_flow_score_p90": float(row.get("short_flow_score_p90", 0.0)),
                "short_flow_score_max": float(row.get("short_flow_score_max", 0.0)),
                "short_flow_score_topk_mean": float(
                    row.get("short_flow_score_topk_mean", 0.0)
                ),
                "long_flow_score_count": int(row.get("long_flow_score_count", 0)),
                "long_flow_score_mean": float(row.get("long_flow_score_mean", 0.0)),
                "long_flow_score_p75": float(row.get("long_flow_score_p75", 0.0)),
                "long_flow_score_p90": float(row.get("long_flow_score_p90", 0.0)),
                "long_flow_score_max": float(row.get("long_flow_score_max", 0.0)),
                "long_flow_score_topk_mean": float(
                    row.get("long_flow_score_topk_mean", 0.0)
                ),
                "short_flow_ratio": float(row.get("short_flow_ratio", 0.0)),
                "component_count": int(row.get("component_count", 0)),
                "component_max_node_score_topk_mean": float(
                    row.get("component_max_node_score_topk_mean", 0.0)
                ),
                "component_max_flow_score_topk_mean": float(
                    row.get("component_max_flow_score_topk_mean", 0.0)
                ),
                "component_max_flow_score_p75": float(
                    row.get("component_max_flow_score_p75", 0.0)
                ),
                "component_max_server_concentration": float(
                    row.get("component_max_server_concentration", 0.0)
                ),
                "server_neighborhood_flow_score_topk_mean": float(
                    row.get("server_neighborhood_flow_score_topk_mean", 0.0)
                ),
                "node_count": int(row.get("node_count", 0)),
                "edge_count": int(row.get("edge_count", 0)),
                "client_node_count": int(row.get("client_node_count", 0)),
                "server_node_count": int(row.get("server_node_count", 0)),
                "aggregated_edge_count": int(row.get("aggregated_edge_count", 0)),
                "communication_edge_count": int(row.get("communication_edge_count", 0)),
                "association_edge_count": int(row.get("association_edge_count", 0)),
                "association_same_src_ip_edge_count": int(
                    row.get("association_same_src_ip_edge_count", 0)
                ),
                "association_same_dst_subnet_edge_count": int(
                    row.get("association_same_dst_subnet_edge_count", 0)
                ),
                "edge_density": float(row.get("edge_density", 0.0)),
                "aggregated_edge_ratio": float(row.get("aggregated_edge_ratio", 0.0)),
                "association_edge_ratio": float(row.get("association_edge_ratio", 0.0)),
                "server_concentration": float(row.get("server_concentration", 0.0)),
                "communication_per_server": float(row.get("communication_per_server", 0.0)),
                "window_index": int(row.get("window_index", 0)),
                "window_start": row.get("window_start"),
                "window_end": row.get("window_end"),
            }
        )
    return export_rows


def _select_worst_malicious_source(
    metric_rows: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    """Select the worst-detected malicious source using a stable ranking rule."""

    if not metric_rows:
        return None

    def _metric_key(row: Mapping[str, object]) -> tuple[float, float, float, str]:
        f1 = row.get("f1")
        pr_auc = row.get("pr_auc")
        recall = row.get("recall")
        return (
            float("inf") if f1 is None else float(f1),
            float("inf") if pr_auc is None else float(pr_auc),
            float("inf") if recall is None else float(recall),
            str(row.get("source_id", "")),
        )

    return min(metric_rows, key=_metric_key)


def _build_comparison_summary_payload(
    *,
    run_id: str,
    experiment_label: str | None,
    timestamp: str,
    mode: PcapGraphExperimentMode,
    backend: str,
    threshold_percentile: float,
    threshold: float,
    graph_score_reduction: PcapGraphScoreReduction,
    benign_paths: Sequence[Path],
    malicious_paths: Sequence[Path],
    split_graph_counts: Mapping[str, int],
    overall_metrics_payload: Mapping[str, float | int | None],
    malicious_source_metric_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    """Build JSON- and CSV-friendly comparison summary payloads for one run."""

    worst_source = _select_worst_malicious_source(malicious_source_metric_rows)
    metrics_available = mode == "binary_evaluation"
    scorer_role = normalize_run_scorer_role(
        backend_name=backend,
        scorer_name=str(graph_score_reduction),
    )
    json_payload: dict[str, object] = {
        "run_id": run_id,
        "experiment_label": experiment_label,
        "timestamp": timestamp,
        "mode": mode,
        "backend": backend,
        "scorer_type": _scorer_type_from_backend(backend),
        "benign_source_count": len(benign_paths),
        "malicious_source_count": len(malicious_paths),
        "benign_graph_total": int(split_graph_counts.get("benign_total", 0)),
        "benign_train_graph_count": int(split_graph_counts.get("train", 0)),
        "benign_test_graph_count": int(split_graph_counts.get("benign_test", 0)),
        "malicious_test_graph_count": int(split_graph_counts.get("malicious_test", 0)),
        "threshold_percentile": float(threshold_percentile),
        "threshold": float(threshold),
        "graph_score_reduction": graph_score_reduction,
        "scorer_role": scorer_role,
        "overall_metrics_availability": (
            "available" if metrics_available else "unavailable_smoke_mode"
        ),
        "overall_roc_auc": overall_metrics_payload.get("roc_auc"),
        "overall_pr_auc": overall_metrics_payload.get("pr_auc"),
        "overall_precision": overall_metrics_payload.get("precision"),
        "overall_recall": overall_metrics_payload.get("recall"),
        "overall_f1": overall_metrics_payload.get("f1"),
        "overall_false_positive_rate": overall_metrics_payload.get("false_positive_rate"),
        "worst_malicious_source_id": None if worst_source is None else worst_source.get("source_id"),
        "worst_malicious_source_name": None
        if worst_source is None
        else worst_source.get("source_name"),
        "worst_malicious_metric_basis": None if worst_source is None else "lowest_f1",
        "worst_malicious_recall": None if worst_source is None else worst_source.get("recall"),
        "worst_malicious_pr_auc": None if worst_source is None else worst_source.get("pr_auc"),
        "worst_malicious_f1": None if worst_source is None else worst_source.get("f1"),
    }
    csv_payload = dict(json_payload)
    for metric_name in (
        "overall_roc_auc",
        "overall_pr_auc",
        "overall_precision",
        "overall_recall",
        "overall_f1",
        "overall_false_positive_rate",
        "worst_malicious_recall",
        "worst_malicious_pr_auc",
        "worst_malicious_f1",
    ):
        csv_payload[metric_name] = _comparison_metric_csv_value(
            csv_payload.get(metric_name)  # type: ignore[arg-type]
        )
    csv_payload["worst_malicious_source_id"] = csv_payload.get(
        "worst_malicious_source_id"
    ) or "unavailable"
    csv_payload["worst_malicious_source_name"] = csv_payload.get(
        "worst_malicious_source_name"
    ) or "unavailable"
    csv_payload["worst_malicious_metric_basis"] = csv_payload.get(
        "worst_malicious_metric_basis"
    ) or "unavailable"
    return json_payload, csv_payload


def _experiment_input_token(
    benign_inputs: Sequence[Path],
    malicious_inputs: Sequence[Path],
) -> str:
    """Build a concise, reproducible source token for the run id."""

    if malicious_inputs:
        return malicious_inputs[0].stem
    if benign_inputs:
        return benign_inputs[0].stem
    return "pcap"


def run_pcap_graph_experiment(
    *,
    export_dir: str | Path,
    benign_inputs: Sequence[str | Path] | None = None,
    malicious_inputs: Sequence[str | Path] | None = None,
    run_name: str | None = None,
    experiment_label: str | None = None,
    config: PcapGraphExperimentConfig | None = None,
    timestamp: object | None = None,
) -> PcapGraphExperimentResult:
    """Run a reproducible mini experiment over one or more real PCAP sources."""

    experiment_config = config or PcapGraphExperimentConfig()
    normalized_experiment_label = experiment_label.strip() if experiment_label else None
    benign_paths = _normalize_paths(benign_inputs)
    malicious_paths = _normalize_paths(malicious_inputs)
    if not benign_paths and not malicious_paths:
        raise ValueError("At least one benign or malicious PCAP input is required.")

    mode: PcapGraphExperimentMode = (
        "binary_evaluation" if benign_paths and malicious_paths else "smoke"
    )
    run_id = _slugify_token(
        run_name or f"{_experiment_input_token(benign_paths, malicious_paths)}-pcap-graph-experiment"
    )
    timestamp_token = _timestamp_token(timestamp)
    layout = build_run_bundle_layout(export_dir, run_id=run_id, timestamp=timestamp_token)
    runtime_config = experiment_config.with_checkpoint_directory(
        Path(layout.run_directory) / "checkpoints"
    )
    pipeline_config = runtime_config.to_pipeline_config(
        input_path=",".join([path.as_posix() for path in benign_paths + malicious_paths]),
        run_name=run_id,
        output_directory=layout.run_directory,
    )

    benign_artifacts = [
        _load_source_artifact(
            path,
            source_id=_build_source_id(path, source_role="benign", source_index=index),
            source_role="benign",
            config=runtime_config,
        )
        for index, path in enumerate(benign_paths)
    ]
    malicious_artifacts = [
        _load_source_artifact(
            path,
            source_id=_build_source_id(path, source_role="malicious", source_index=index),
            source_role="malicious",
            config=runtime_config,
        )
        for index, path in enumerate(malicious_paths)
    ]
    source_artifacts = benign_artifacts + malicious_artifacts
    source_counts = _aggregate_source_counts(source_artifacts)

    benign_entries = _flatten_source_artifacts(benign_artifacts)
    malicious_entries = _flatten_source_artifacts(malicious_artifacts)
    all_entries = benign_entries + malicious_entries
    notes: list[str] = []

    if mode == "binary_evaluation":
        benign_train_pool, benign_test_entries = _partition_benign_entries(
            benign_entries,
            benign_train_ratio=runtime_config.benign_train_ratio,
            random_seed=runtime_config.random_seed,
        )
        train_graphs, val_graphs = _split_graphs(
            [entry.graph for entry in benign_train_pool],
            validation_ratio=runtime_config.train_validation_ratio,
        )
        train_graph_ids = {id(graph) for graph in train_graphs}
        val_graph_ids = {id(graph) for graph in val_graphs}
        _mark_assignments(benign_test_entries, "benign_test")
        _mark_assignments(malicious_entries, "malicious_test")
        for entry in benign_train_pool:
            entry.split_assignment = (
                "train" if id(entry.graph) in train_graph_ids else "val"
            )
        evaluation_entries = benign_test_entries + malicious_entries
        if not evaluation_entries:
            raise ValueError(
                "The binary PCAP experiment did not produce any evaluation graphs."
            )
        notes.extend(
            [
                "Binary evaluation mode: train uses benign graphs only.",
                "Evaluation uses benign holdout graphs plus malicious graphs.",
            ]
        )
        train_reference_entries = [
            entry for entry in benign_entries if id(entry.graph) in train_graph_ids
        ]
        train_reference_split_name = "benign_train_reference"
    else:
        if not all_entries:
            raise ValueError("The PCAP experiment did not yield any graphs.")
        _mark_assignments(all_entries, "smoke")
        train_graphs, val_graphs = _split_graphs(
            [entry.graph for entry in all_entries],
            validation_ratio=runtime_config.train_validation_ratio,
        )
        train_graph_ids = {id(graph) for graph in train_graphs}
        evaluation_entries = all_entries
        notes.append(
            "Smoke validation mode: labels are not required and the chain is validated end to end."
        )
        train_reference_entries = [
            entry for entry in all_entries if id(entry.graph) in train_graph_ids
        ]
        train_reference_split_name = "smoke_train_reference"

    feature_preprocessor = fit_feature_preprocessor(
        train_graphs,
        normalization_config=FeatureNormalizationConfig(),
        include_graph_structural_features=runtime_config.use_graph_structural_features,
    )
    evaluation_graphs = [entry.graph for entry in evaluation_entries]
    evaluation_packed_graphs = transform_graphs(
        evaluation_graphs,
        feature_preprocessor,
        include_graph_structural_features=runtime_config.use_graph_structural_features,
    )
    train_packed_graphs = transform_graphs(
        train_graphs,
        feature_preprocessor,
        include_graph_structural_features=runtime_config.use_graph_structural_features,
    )
    feature_summaries = [
        summarize_packed_graph_input(packed_graph)
        for packed_graph in evaluation_packed_graphs
    ]

    if _has_torch():
        (
            training_history,
            checkpoint_paths,
            graph_rows,
            node_rows,
            edge_rows,
            flow_rows,
            threshold,
            train_score_summary,
            backend_notes,
        ) = _train_and_score_with_gae(
            train_graphs=train_graphs,
            val_graphs=val_graphs,
            all_graphs=evaluation_graphs,
            all_packed_graphs=evaluation_packed_graphs,
            config=runtime_config,  # type: ignore[arg-type]
            pipeline_config=pipeline_config,
        )
        backend = "gae"
    else:
        (
            training_history,
            checkpoint_paths,
            graph_rows,
            node_rows,
            edge_rows,
            flow_rows,
            threshold,
            train_score_summary,
            backend_notes,
        ) = _score_with_fallback(
            graph_samples=evaluation_graphs,
            packed_graphs=evaluation_packed_graphs,
            threshold_percentile=runtime_config.threshold_percentile,
            graph_score_reduction=runtime_config.graph_score_reduction,
            threshold_reference_packed_graphs=train_packed_graphs,
        )
        backend = "fallback"
    notes.extend(backend_notes)

    if train_graphs:
        if backend == "gae":
            checkpoint_path = (
                checkpoint_paths.get("best_checkpoint")
                or checkpoint_paths.get("latest_checkpoint")
            )
            if not checkpoint_path:
                raise ValueError("The GAE path did not return a usable checkpoint path.")
            train_graph_rows_raw = _score_graph_rows_with_gae_checkpoint(
                graphs=list(train_graphs),
                packed_graphs=list(train_packed_graphs),
                checkpoint_path=checkpoint_path,
                reduction_method=runtime_config.graph_score_reduction,
            )
        else:
            (
                _ignored_history,
                _ignored_checkpoints,
                train_graph_rows_raw,
                _ignored_node_rows,
                _ignored_edge_rows,
                _ignored_flow_rows,
                _ignored_threshold,
                _ignored_train_summary,
                _ignored_notes,
            ) = _score_with_fallback(
                graph_samples=list(train_graphs),
                packed_graphs=list(train_packed_graphs),
                threshold_percentile=runtime_config.threshold_percentile,
                graph_score_reduction=runtime_config.graph_score_reduction,
                threshold_reference_packed_graphs=train_packed_graphs,
            )
    else:
        train_graph_rows_raw = []

    train_graph_rows_raw = _apply_graph_score_reduction_to_rows(
        train_graph_rows_raw,
        reduction_method=runtime_config.graph_score_reduction,
        reference_rows=train_graph_rows_raw,
    )
    graph_rows = _apply_graph_score_reduction_to_rows(
        graph_rows,
        reduction_method=runtime_config.graph_score_reduction,
        reference_rows=train_graph_rows_raw,
    )
    threshold, train_score_summary = _graph_score_threshold_from_rows(
        train_graph_rows_raw,
        threshold_percentile=runtime_config.threshold_percentile,
    )

    contextual_train_graph_rows = _contextualize_score_rows(
        train_graph_rows_raw,
        scope="graph",
        entries=train_reference_entries,
        packed_graphs=train_packed_graphs,
        mode=mode,
        backend=backend,
        task_name="train_reference",
    )
    train_graph_score_rows = _build_train_graph_score_rows(
        contextual_train_rows=contextual_train_graph_rows,
        run_id=run_id,
        timestamp=timestamp_token,
        threshold=threshold,
        backend=backend,
        split_name=train_reference_split_name,
    )

    contextual_graph_rows = _contextualize_score_rows(
        graph_rows,
        scope="graph",
        entries=evaluation_entries,
        packed_graphs=evaluation_packed_graphs,
        mode=mode,
        backend=backend,
        task_name="overall",
    )
    contextual_node_rows = _contextualize_score_rows(
        node_rows,
        scope="node",
        entries=evaluation_entries,
        packed_graphs=evaluation_packed_graphs,
        mode=mode,
        backend=backend,
        task_name="overall",
    )
    contextual_edge_rows = _contextualize_score_rows(
        edge_rows,
        scope="edge",
        entries=evaluation_entries,
        packed_graphs=evaluation_packed_graphs,
        mode=mode,
        backend=backend,
        task_name="overall",
    )
    contextual_flow_rows = _contextualize_score_rows(
        flow_rows,
        scope="flow",
        entries=evaluation_entries,
        packed_graphs=evaluation_packed_graphs,
        mode=mode,
        backend=backend,
        task_name="overall",
    )
    score_tables = {
        "graph_scores": contextual_graph_rows,
        "node_scores": contextual_node_rows,
        "edge_scores": contextual_edge_rows,
        "flow_scores": contextual_flow_rows,
    }
    alert_records = build_alert_records(
        score_tables,
        AlertingConfig(anomaly_threshold=threshold),
    )
    alert_summary = summarize_alerts(alert_records)
    score_summary = {
        "graph": _quantile_summary([_score_from_row(row) for row in contextual_graph_rows]),
        "node": _quantile_summary([_score_from_row(row) for row in contextual_node_rows]),
        "edge": _quantile_summary([_score_from_row(row) for row in contextual_edge_rows]),
        "flow": _quantile_summary([_score_from_row(row) for row in contextual_flow_rows]),
    }
    source_summary_rows = _build_source_summary_rows(
        source_artifacts=source_artifacts,
        all_entries=all_entries,
        contextual_graph_rows=contextual_graph_rows,
    )
    source_score_summary_rows = _build_source_score_summary_rows(source_summary_rows)
    split_score_summary_rows = _build_split_score_summary_rows(
        mode=mode,
        train_graph_count=len(train_graphs),
        train_score_summary=train_score_summary,
        contextual_graph_rows=contextual_graph_rows,
    )
    split_graph_counts = {
        "train": sum(1 for entry in all_entries if entry.split_assignment == "train"),
        "val": sum(1 for entry in all_entries if entry.split_assignment == "val"),
        "benign_test": sum(
            1 for entry in all_entries if entry.split_assignment == "benign_test"
        ),
        "malicious_test": sum(
            1 for entry in all_entries if entry.split_assignment == "malicious_test"
        ),
        "smoke": sum(1 for entry in all_entries if entry.split_assignment == "smoke"),
    }
    split_graph_counts["evaluation_total"] = len(evaluation_entries)
    split_graph_counts["benign_total"] = len(benign_entries)
    split_graph_counts["malicious_total"] = len(malicious_entries)
    split_graph_ids = _graph_ids_by_split(all_entries)

    overall_score_records = [
        _score_record_from_row(
            row,
            run_id=run_id,
            timestamp=timestamp_token,
            threshold=threshold,
            split="smoke" if mode == "smoke" else "overall_test",
            task_name="overall",
        )
        for row in contextual_graph_rows
    ]

    per_attack_metrics: list[BinaryAttackMetricRecord] = []
    attack_score_records: list[BinaryDetectionScoreRecord] = []
    attack_score_summaries: dict[str, dict[str, float | int]] = {}
    overall_metrics_payload: dict[str, float | int | None]
    if mode == "binary_evaluation":
        overall_metrics = evaluate_scores(
            [record.binary_label for record in overall_score_records],
            [record.anomaly_score for record in overall_score_records],
            threshold=threshold,
        )
        overall_metrics_payload = _metric_payload(overall_metrics)
        benign_overall_records = [
            record for record in overall_score_records if record.binary_label == 0
        ]
        malicious_by_group: dict[str, list[BinaryDetectionScoreRecord]] = {}
        for record in overall_score_records:
            if record.binary_label != 1:
                continue
            malicious_by_group.setdefault(record.attack_group, []).append(record)

        for task_name in sorted(malicious_by_group):
            task_records = benign_overall_records + malicious_by_group[task_name]
            for record in task_records:
                attack_score_records.append(
                    BinaryDetectionScoreRecord(
                        score_id=f"{task_name}:{record.sample_id}",
                        run_id=record.run_id,
                        timestamp=record.timestamp,
                        split="task_test",
                        evaluation_scope=record.evaluation_scope,
                        task_name=task_name,
                        sample_id=record.sample_id,
                        row_index=record.row_index,
                        raw_label=record.raw_label,
                        binary_label=record.binary_label,
                        attack_group=record.attack_group,
                        anomaly_score=record.anomaly_score,
                        threshold=record.threshold,
                        is_alert=record.is_alert,
                        feature_count=record.feature_count,
                        metadata=dict(record.metadata),
                    )
                )
            metric_record = _build_attack_metric_record(
                task_name=task_name,
                records=task_records,
                threshold=threshold,
                notes=(f"Attack source {task_name} evaluated against benign holdout graphs.",),
            )
            per_attack_metrics.append(metric_record)
            attack_score_summaries[task_name] = _quantile_summary(
                [record.anomaly_score for record in task_records]
            )

        all_malicious_records = benign_overall_records + [
            record for record in overall_score_records if record.binary_label == 1
        ]
        if all_malicious_records:
            for record in all_malicious_records:
                attack_score_records.append(
                    BinaryDetectionScoreRecord(
                        score_id=f"all_malicious:{record.sample_id}",
                        run_id=record.run_id,
                        timestamp=record.timestamp,
                        split="task_test",
                        evaluation_scope=record.evaluation_scope,
                        task_name="all_malicious",
                        sample_id=record.sample_id,
                        row_index=record.row_index,
                        raw_label=record.raw_label,
                        binary_label=record.binary_label,
                        attack_group=record.attack_group,
                        anomaly_score=record.anomaly_score,
                        threshold=record.threshold,
                        is_alert=record.is_alert,
                        feature_count=record.feature_count,
                        metadata=dict(record.metadata),
                    )
                )
            per_attack_metrics.append(
                _build_attack_metric_record(
                    task_name="all_malicious",
                    records=all_malicious_records,
                    threshold=threshold,
                    notes=("All malicious graphs pooled into one evaluation task.",),
                )
            )
            attack_score_summaries["all_malicious"] = _quantile_summary(
                [record.anomaly_score for record in all_malicious_records]
            )
        malicious_source_metric_rows = _build_malicious_source_metric_rows(
            contextual_graph_rows=contextual_graph_rows,
            threshold=threshold,
        )
    else:
        overall_metrics_payload = {
            "support": 0,
            "positive_count": 0,
            "negative_count": 0,
            "roc_auc": None,
            "pr_auc": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "false_positive_rate": None,
            "true_positive": 0,
            "false_positive": 0,
            "true_negative": 0,
            "false_negative": 0,
            "threshold": float(threshold),
        }
        malicious_source_metric_rows = []

    comparison_summary_json, comparison_summary_csv = _build_comparison_summary_payload(
        run_id=run_id,
        experiment_label=normalized_experiment_label,
        timestamp=timestamp_token,
        mode=mode,
        backend=backend,
        threshold_percentile=runtime_config.threshold_percentile,
        threshold=threshold,
        graph_score_reduction=runtime_config.graph_score_reduction,
        benign_paths=benign_paths,
        malicious_paths=malicious_paths,
        split_graph_counts=split_graph_counts,
        overall_metrics_payload=overall_metrics_payload,
        malicious_source_metric_rows=malicious_source_metric_rows,
    )

    metrics_summary = {
        "pcap_experiment": {
            "run_id": run_id,
            "experiment_label": normalized_experiment_label,
            "timestamp": timestamp_token,
            "mode": mode,
            "backend": backend,
            "benign_inputs": [path.as_posix() for path in benign_paths],
            "malicious_inputs": [path.as_posix() for path in malicious_paths],
            "packet_limit": runtime_config.packet_limit,
            "packet_sampling_mode": runtime_config.packet_sampling_mode,
            "idle_timeout_seconds": runtime_config.idle_timeout_seconds,
            "window_size": runtime_config.window_size,
            "use_association_edges": runtime_config.use_association_edges,
            "use_graph_structural_features": runtime_config.use_graph_structural_features,
            "graph_score_reduction": runtime_config.graph_score_reduction,
            "smoke_graph_limit": runtime_config.smoke_graph_limit,
            "threshold_percentile": runtime_config.threshold_percentile,
            "anomaly_threshold": threshold,
            "train_graph_count": len(train_graphs),
            "val_graph_count": len(val_graphs),
            "evaluation_graph_count": len(evaluation_graphs),
            "split_graph_counts": split_graph_counts,
            "split_graph_ids": split_graph_ids,
            "source_counts": source_counts,
            "source_summaries": source_summary_rows,
            "source_score_summaries": source_score_summary_rows,
            "split_score_summaries": split_score_summary_rows,
            "train_graph_score_summary": train_score_summary,
            "graph_score_summary": score_summary["graph"],
            "node_score_summary": score_summary["node"],
            "edge_score_summary": score_summary["edge"],
            "flow_score_summary": score_summary["flow"],
            "overall_metrics": overall_metrics_payload,
            "per_attack_metrics": [metric.to_dict() for metric in per_attack_metrics],
            "malicious_source_metrics": malicious_source_metric_rows,
            "attack_score_summaries": attack_score_summaries,
            "feature_summaries": feature_summaries,
            "training_history": training_history,
            "checkpoint_paths": checkpoint_paths,
            "comparison_summary": comparison_summary_json,
            "alert_summary": alert_summary,
            "notes": notes,
        }
    }

    export_result = export_run_bundle(
        score_tables,
        alert_records,
        metrics_summary,
        export_dir,
        run_id=run_id,
        split="smoke" if mode == "smoke" else "binary",
        timestamp=timestamp_token,
        anomaly_threshold=threshold,
        score_formats=("jsonl", "csv"),
        alert_formats=("jsonl", "csv"),
        metrics_formats=("json", "jsonl", "csv"),
    )

    run_directory = Path(export_result.run_directory)
    graph_summary_rows = [
        _graph_summary_row(entry)
        for entry in benign_entries + malicious_entries
    ]
    graph_summary_json_path = run_directory / "graph_summary.json"
    graph_summary_csv_path = run_directory / "graph_summary.csv"
    _write_json(graph_summary_json_path, graph_summary_rows)
    graph_summary_row_count = _write_csv(
        graph_summary_csv_path,
        graph_summary_rows,
        PCAP_GRAPH_SUMMARY_FIELDS,
    )

    config_snapshot_path = run_directory / "pcap_experiment_config.json"
    _write_json(config_snapshot_path, runtime_config.to_dict())

    summary_payload = {
        "run_id": run_id,
        "experiment_label": normalized_experiment_label,
        "timestamp": timestamp_token,
        "mode": mode,
        "backend": backend,
        "benign_inputs": [path.as_posix() for path in benign_paths],
        "malicious_inputs": [path.as_posix() for path in malicious_paths],
        "total_packets": int(source_counts["total_packets"]),
        "parsed_packets": int(source_counts["parsed_packets"]),
        "skipped_packets": int(source_counts["skipped_packets"]),
        "total_flows": int(source_counts["total_flows"]),
        "total_graphs": int(source_counts["total_graphs"]),
        "benign_graph_count": len(benign_entries),
        "malicious_graph_count": len(malicious_entries),
        "train_graph_count": len(train_graphs),
        "val_graph_count": len(val_graphs),
        "evaluation_graph_count": len(evaluation_graphs),
        "split_graph_counts": split_graph_counts,
        "split_graph_ids": split_graph_ids,
        "source_counts": source_counts,
        "window_size": runtime_config.window_size,
        "use_association_edges": runtime_config.use_association_edges,
        "use_graph_structural_features": runtime_config.use_graph_structural_features,
        "graph_score_reduction": runtime_config.graph_score_reduction,
        "packet_limit": runtime_config.packet_limit,
        "packet_sampling_mode": runtime_config.packet_sampling_mode,
        "idle_timeout_seconds": runtime_config.idle_timeout_seconds,
        "smoke_graph_limit": runtime_config.smoke_graph_limit,
        "threshold_percentile": runtime_config.threshold_percentile,
        "anomaly_threshold": threshold,
        "train_graph_score_quantiles": train_score_summary,
        "train_graph_reference_split_name": train_reference_split_name,
        "train_graph_reference_count": len(train_graph_score_rows),
        "graph_score_quantiles": score_summary["graph"],
        "node_score_quantiles": score_summary["node"],
        "edge_score_quantiles": score_summary["edge"],
        "flow_score_quantiles": score_summary["flow"],
        "overall_metrics": overall_metrics_payload,
        "comparison_summary": comparison_summary_json,
        "alert_summary": alert_summary,
        "source_summaries": source_summary_rows,
        "source_score_summaries": source_score_summary_rows,
        "split_score_summaries": split_score_summary_rows,
        "malicious_source_metrics": malicious_source_metric_rows,
        "notes": notes,
    }
    experiment_summary_path = run_directory / "pcap_experiment_summary.json"
    _write_json(experiment_summary_path, summary_payload)

    metrics_summary_path = run_directory / "metrics_summary.json"
    _write_json(metrics_summary_path, metrics_summary)
    score_quantiles_path = run_directory / "score_quantiles.json"
    _write_json(
        score_quantiles_path,
        {
            "train": train_score_summary,
            "overall": score_summary["graph"],
            "per_attack": attack_score_summaries,
            "by_source": source_score_summary_rows,
            "by_split": split_score_summary_rows,
        },
    )

    artifact_paths = dict(export_result.artifact_paths)
    row_counts = dict(export_result.row_counts)
    extra_notes = list(notes)
    artifact_paths["pcap_experiment_config_json"] = config_snapshot_path.as_posix()
    artifact_paths["pcap_experiment_summary_json"] = experiment_summary_path.as_posix()
    artifact_paths["graph_summary_json"] = graph_summary_json_path.as_posix()
    artifact_paths["graph_summary_csv"] = graph_summary_csv_path.as_posix()
    artifact_paths["metrics_summary_root_json"] = metrics_summary_path.as_posix()
    artifact_paths["score_quantiles_root_json"] = score_quantiles_path.as_posix()
    row_counts["graph_summary_csv"] = graph_summary_row_count

    source_score_summary_json_path = run_directory / "source_score_summary.json"
    source_score_summary_csv_path = run_directory / "source_score_summary.csv"
    _write_json(source_score_summary_json_path, source_score_summary_rows)
    row_counts["source_score_summary_csv"] = _write_csv(
        source_score_summary_csv_path,
        source_score_summary_rows,
        PCAP_SOURCE_SCORE_SUMMARY_FIELDS,
    )
    artifact_paths["source_score_summary_json"] = source_score_summary_json_path.as_posix()
    artifact_paths["source_score_summary_csv"] = source_score_summary_csv_path.as_posix()

    split_score_summary_json_path = run_directory / "split_score_summary.json"
    split_score_summary_csv_path = run_directory / "split_score_summary.csv"
    _write_json(split_score_summary_json_path, split_score_summary_rows)
    row_counts["split_score_summary_csv"] = _write_csv(
        split_score_summary_csv_path,
        split_score_summary_rows,
        PCAP_SPLIT_SCORE_SUMMARY_FIELDS,
    )
    artifact_paths["split_score_summary_json"] = split_score_summary_json_path.as_posix()
    artifact_paths["split_score_summary_csv"] = split_score_summary_csv_path.as_posix()

    train_graph_scores_jsonl_path = run_directory / "train_graph_scores.jsonl"
    train_graph_scores_csv_path = run_directory / "train_graph_scores.csv"
    row_counts["train_graph_scores_jsonl"] = _write_jsonl(
        train_graph_scores_jsonl_path,
        train_graph_score_rows,
    )
    row_counts["train_graph_scores_csv"] = _write_csv(
        train_graph_scores_csv_path,
        train_graph_score_rows,
        PCAP_TRAIN_GRAPH_SCORE_FIELDS,
    )
    artifact_paths["train_graph_scores_jsonl"] = train_graph_scores_jsonl_path.as_posix()
    artifact_paths["train_graph_scores_csv"] = train_graph_scores_csv_path.as_posix()

    comparison_summary_json_path = run_directory / "comparison_summary.json"
    comparison_summary_csv_path = run_directory / "comparison_summary.csv"
    _write_json(comparison_summary_json_path, comparison_summary_json)
    row_counts["comparison_summary_csv"] = _write_csv(
        comparison_summary_csv_path,
        [comparison_summary_csv],
        PCAP_COMPARISON_SUMMARY_FIELDS,
    )
    artifact_paths["comparison_summary_json"] = comparison_summary_json_path.as_posix()
    artifact_paths["comparison_summary_csv"] = comparison_summary_csv_path.as_posix()

    _export_score_records(
        records=overall_score_records,
        base_path=run_directory / "overall_scores",
        artifact_paths=artifact_paths,
        row_counts=row_counts,
        artifact_name="overall_scores",
    )
    _export_score_records(
        records=attack_score_records,
        base_path=run_directory / "attack_scores",
        artifact_paths=artifact_paths,
        row_counts=row_counts,
        artifact_name="attack_scores",
    )
    per_attack_metrics_path = run_directory / "per_attack_metrics.csv"
    per_attack_metric_rows = [metric.to_csv_row() for metric in per_attack_metrics]
    row_counts["per_attack_metrics_csv"] = _write_csv(
        per_attack_metrics_path,
        per_attack_metric_rows,
        BINARY_ATTACK_METRIC_FIELDS,
    )
    artifact_paths["per_attack_metrics_csv"] = per_attack_metrics_path.as_posix()

    malicious_source_metrics_json_path = run_directory / "malicious_source_metrics.json"
    malicious_source_metrics_csv_path = run_directory / "malicious_source_metrics.csv"
    _write_json(malicious_source_metrics_json_path, malicious_source_metric_rows)
    row_counts["malicious_source_metrics_csv"] = _write_csv(
        malicious_source_metrics_csv_path,
        malicious_source_metric_rows,
        PCAP_MALICIOUS_SOURCE_METRIC_FIELDS,
    )
    artifact_paths["malicious_source_metrics_json"] = (
        malicious_source_metrics_json_path.as_posix()
    )
    artifact_paths["malicious_source_metrics_csv"] = (
        malicious_source_metrics_csv_path.as_posix()
    )

    _update_manifest(
        Path(export_result.manifest_path),
        extra_artifact_paths=artifact_paths,
        extra_row_counts=row_counts,
        extra_notes=extra_notes,
        extra_payload={
            "experiment_mode": mode,
            "backend": backend,
            "experiment_label": normalized_experiment_label,
            "benign_inputs": [path.as_posix() for path in benign_paths],
            "malicious_inputs": [path.as_posix() for path in malicious_paths],
        },
    )
    export_result.artifact_paths = artifact_paths
    export_result.row_counts = row_counts
    for note in extra_notes:
        if note not in export_result.notes:
            export_result.notes.append(note)

    return PcapGraphExperimentResult(
        run_id=run_id,
        timestamp=timestamp_token,
        mode=mode,
        backend=backend,
        config=runtime_config,
        summary=summary_payload,
        export_result=export_result,
        notes=notes,
    )


def summarize_pcap_graph_experiment_result(result: PcapGraphExperimentResult) -> str:
    """Render a human-readable summary for the real-PCAP experiment."""

    return result.render()


__all__ = [
    "PCAP_GRAPH_SUMMARY_FIELDS",
    "PcapGraphExperimentConfig",
    "PcapGraphExperimentMode",
    "PcapGraphExperimentResult",
    "run_pcap_graph_experiment",
    "summarize_pcap_graph_experiment_result",
]
