"""Typed configuration objects for the traffic graph pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from traffic_graph.data.preprocessing import ShortFlowThresholds

_DEFAULT_DISCRETE_EDGE_FIELDS: tuple[str, ...] = (
    "edge_type",
    "flow_length_type_code",
    "is_aggregated",
    "rst_after_small_burst_indicator",
    "flag_pattern_code",
    "first_packet_size_pattern",
    "first_packet_dir_size_pattern",
    "first_4_packet_pattern_code",
    "prefix_behavior_signature",
)

_DEFAULT_TEMPORAL_EDGE_FIELD_NAMES: tuple[str, ...] = (
    "coarse_ack_delay_mean",
    "coarse_ack_delay_p75",
    "ack_delay_large_gap_ratio",
    "seq_ack_match_ratio",
    "unmatched_seq_ratio",
    "unmatched_ack_ratio",
    "retry_burst_count",
    "retry_burst_max_len",
    "retry_like_dense_ratio",
    "small_pkt_burst_count",
    "small_pkt_burst_ratio",
)


def _mapping_section(data: Mapping[str, object], key: str) -> Mapping[str, object]:
    """Return a nested mapping section or an empty mapping when absent."""

    value = data.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _as_str(value: object, default: str) -> str:
    """Convert a value to string while preserving a default for missing values."""

    if value is None:
        return default
    return str(value)


def _as_int(value: object, default: int) -> int:
    """Convert a value to integer and fall back to a default on invalid input."""

    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: object, default: bool) -> bool:
    """Convert a scalar value into a boolean configuration flag."""

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    return bool(value)


def _as_optional_float(value: object) -> float | None:
    """Convert an optional scalar into float when present and valid."""

    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_str_tuple(value: object, default: tuple[str, ...]) -> tuple[str, ...]:
    """Convert a scalar or iterable value into a tuple of strings."""

    if value is None:
        return default
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else default
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        normalized = tuple(str(item).strip() for item in value if str(item).strip())
        return normalized if normalized else default
    return default


@dataclass(slots=True)
class PipelineRuntimeConfig:
    """Execution-level options shared across the pipeline."""

    run_name: str = "bootstrap-run"
    seed: int = 42

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "PipelineRuntimeConfig":
        """Build a runtime configuration from a loose mapping."""

        return cls(
            run_name=_as_str(data.get("run_name"), "bootstrap-run"),
            seed=_as_int(data.get("seed"), 42),
        )


@dataclass(slots=True)
class DataConfig:
    """Input data settings for flow ingestion."""

    input_path: str = "data/flows.csv"
    format: str = "csv"

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "DataConfig":
        """Build a data configuration from a loose mapping."""

        return cls(
            input_path=_as_str(data.get("input_path"), "data/flows.csv"),
            format=_as_str(data.get("format"), "csv").lower(),
        )


@dataclass(slots=True)
class PreprocessingConfig:
    """Configuration for window-based preprocessing before graph construction."""

    window_size: int = 60
    short_flow_thresholds: ShortFlowThresholds = field(
        default_factory=ShortFlowThresholds
    )

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "PreprocessingConfig":
        """Build a preprocessing configuration from a loose mapping."""

        thresholds_raw = data.get("short_flow_thresholds", {})
        thresholds_mapping = thresholds_raw if isinstance(thresholds_raw, Mapping) else {}
        window_size = _as_int(
            data.get("window_size", data.get("window_size_seconds")),
            60,
        )
        if window_size <= 0:
            window_size = 60
        return cls(
            window_size=window_size,
            short_flow_thresholds=ShortFlowThresholds.from_mapping(
                thresholds_mapping
            ),
        )


@dataclass(slots=True)
class AssociationEdgeConfig:
    """Configuration for lightweight association edges inside one time window."""

    enable_same_src_ip: bool = False
    enable_same_dst_subnet: bool = False
    enable_same_dst_ip: bool = False
    enable_same_prefix_signature: bool = False
    enable_prefix_similarity: bool = False
    dst_subnet_prefix: int = 24
    prefix_similarity_threshold: float = 0.95
    prefix_similarity_top_k: int = 1

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "AssociationEdgeConfig":
        """Build association-edge settings from a loose mapping."""

        prefix = _as_int(data.get("dst_subnet_prefix"), 24)
        if prefix < 0 or prefix > 32:
            prefix = 24
        return cls(
            enable_same_src_ip=_as_bool(data.get("enable_same_src_ip"), False),
            enable_same_dst_subnet=_as_bool(
                data.get("enable_same_dst_subnet"),
                False,
            ),
            enable_same_dst_ip=_as_bool(data.get("enable_same_dst_ip"), False),
            enable_same_prefix_signature=_as_bool(
                data.get("enable_same_prefix_signature"),
                False,
            ),
            enable_prefix_similarity=_as_bool(
                data.get("enable_prefix_similarity"),
                False,
            ),
            dst_subnet_prefix=prefix,
            prefix_similarity_threshold=(
                _as_optional_float(data.get("prefix_similarity_threshold")) or 0.95
            ),
            prefix_similarity_top_k=max(
                1,
                _as_int(data.get("prefix_similarity_top_k"), 1),
            ),
        )


@dataclass(slots=True)
class GraphConfig:
    """Graph construction settings for the interaction graph layer."""

    time_window_seconds: int = 60
    directed: bool = True
    association_edges: AssociationEdgeConfig = field(
        default_factory=AssociationEdgeConfig
    )

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "GraphConfig":
        """Build a graph configuration from a loose mapping."""

        return cls(
            time_window_seconds=_as_int(data.get("time_window_seconds"), 60),
            directed=_as_bool(data.get("directed"), True),
            association_edges=AssociationEdgeConfig.from_mapping(
                _mapping_section(data, "association_edges")
            ),
        )


@dataclass(slots=True)
class FeatureNormalizationConfig:
    """Normalization options for packed node and edge features."""

    enabled: bool = True
    method: str = "standard"
    exclude_node_fields: tuple[str, ...] = ("endpoint_type", "port", "proto")
    exclude_edge_fields: tuple[str, ...] = _DEFAULT_DISCRETE_EDGE_FIELDS

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "FeatureNormalizationConfig":
        """Build a feature-normalization configuration from a loose mapping."""

        method = _as_str(data.get("method"), "standard").lower()
        if method not in {"standard", "robust", "none"}:
            method = "standard"
        return cls(
            enabled=_as_bool(data.get("enabled"), True),
            method=method,
            exclude_node_fields=_as_str_tuple(
                data.get("exclude_node_fields"),
                ("endpoint_type", "port", "proto"),
            ),
            exclude_edge_fields=_as_str_tuple(
                data.get("exclude_edge_fields"),
                _DEFAULT_DISCRETE_EDGE_FIELDS,
            ),
        )


@dataclass(slots=True)
class FeaturesConfig:
    """Feature preparation settings used before model-facing graph packing."""

    normalization: FeatureNormalizationConfig = field(
        default_factory=FeatureNormalizationConfig
    )
    use_graph_structural_features: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "FeaturesConfig":
        """Build feature configuration from a loose mapping."""

        return cls(
            normalization=FeatureNormalizationConfig.from_mapping(
                _mapping_section(data, "normalization")
            ),
            use_graph_structural_features=_as_bool(
                data.get("use_graph_structural_features"),
                True,
            ),
        )


@dataclass(slots=True)
class ModelConfig:
    """Model settings reserved for future detector backends and GAE training."""

    name: str = "placeholder"
    device: str = "cpu"
    score_threshold: float | None = None
    hidden_dim: int = 64
    latent_dim: int = 32
    num_layers: int = 2
    dropout: float = 0.1
    use_edge_features: bool = True
    reconstruct_edge_features: bool = True
    use_temporal_edge_projector: bool = False
    temporal_edge_hidden_dim: int = 32
    temporal_edge_field_names: tuple[str, ...] = _DEFAULT_TEMPORAL_EDGE_FIELD_NAMES
    use_edge_categorical_embeddings: bool = True
    edge_categorical_embedding_dim: int = 8
    edge_categorical_bucket_size: int = 128

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "ModelConfig":
        """Build a model configuration from a loose mapping."""

        dropout = _as_optional_float(data.get("dropout"))
        return cls(
            name=_as_str(data.get("name"), "placeholder"),
            device=_as_str(data.get("device"), "cpu"),
            score_threshold=_as_optional_float(data.get("score_threshold")),
            hidden_dim=_as_int(data.get("hidden_dim"), 64),
            latent_dim=_as_int(data.get("latent_dim"), 32),
            num_layers=_as_int(data.get("num_layers"), 2),
            dropout=0.1 if dropout is None else dropout,
            use_edge_features=_as_bool(data.get("use_edge_features"), True),
            reconstruct_edge_features=_as_bool(
                data.get("reconstruct_edge_features"),
                True,
            ),
            use_temporal_edge_projector=_as_bool(
                data.get("use_temporal_edge_projector"),
                False,
            ),
            temporal_edge_hidden_dim=max(
                1,
                _as_int(data.get("temporal_edge_hidden_dim"), 32),
            ),
            temporal_edge_field_names=_as_str_tuple(
                data.get("temporal_edge_field_names"),
                _DEFAULT_TEMPORAL_EDGE_FIELD_NAMES,
            ),
            use_edge_categorical_embeddings=_as_bool(
                data.get("use_edge_categorical_embeddings"),
                True,
            ),
            edge_categorical_embedding_dim=max(
                1,
                _as_int(data.get("edge_categorical_embedding_dim"), 8),
            ),
            edge_categorical_bucket_size=max(
                16,
                _as_int(data.get("edge_categorical_bucket_size"), 128),
            ),
        )


@dataclass(slots=True)
class TrainingConfig:
    """Training settings for the first unsupervised graph autoencoder."""

    epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    batch_size: int = 1
    validation_split_ratio: float = 0.2
    early_stopping_patience: int = 5
    checkpoint_dir: str = "artifacts/checkpoints"
    shuffle: bool = True
    seed: int = 42
    smoke_graph_limit: int = 4

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "TrainingConfig":
        """Build a training configuration from a loose mapping."""

        validation_split_ratio = _as_optional_float(
            data.get("validation_split_ratio")
        )
        if validation_split_ratio is None or not 0.0 <= validation_split_ratio <= 1.0:
            validation_split_ratio = 0.2

        return cls(
            epochs=max(1, _as_int(data.get("epochs"), 10)),
            learning_rate=_as_optional_float(data.get("learning_rate")) or 0.001,
            weight_decay=_as_optional_float(data.get("weight_decay")) or 0.0,
            batch_size=max(1, _as_int(data.get("batch_size"), 1)),
            validation_split_ratio=validation_split_ratio,
            early_stopping_patience=max(
                0,
                _as_int(data.get("early_stopping_patience"), 5),
            ),
            checkpoint_dir=_as_str(
                data.get("checkpoint_dir"),
                "artifacts/checkpoints",
            ),
            shuffle=_as_bool(data.get("shuffle"), True),
            seed=_as_int(data.get("seed"), 42),
            smoke_graph_limit=max(1, _as_int(data.get("smoke_graph_limit"), 4)),
        )


@dataclass(slots=True)
class EvaluationConfig:
    """Evaluation settings for anomaly scoring and metric computation."""

    score_reduction: str = "mean"
    anomaly_threshold: float = 0.5
    evaluation_label_field: str = "label"
    checkpoint_dir: str = "artifacts/checkpoints"
    checkpoint_tag: str = "best"

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "EvaluationConfig":
        """Build an evaluation configuration from a loose mapping."""

        score_reduction = _as_str(data.get("score_reduction"), "mean").lower()
        if score_reduction not in {"mean", "max"}:
            score_reduction = "mean"
        threshold = _as_optional_float(data.get("anomaly_threshold"))
        if threshold is None:
            threshold = 0.5
        return cls(
            score_reduction=score_reduction,
            anomaly_threshold=threshold,
            evaluation_label_field=_as_str(
                data.get("evaluation_label_field"),
                "label",
            ),
            checkpoint_dir=_as_str(data.get("checkpoint_dir"), "artifacts/checkpoints"),
            checkpoint_tag=_as_str(data.get("checkpoint_tag"), "best"),
        )


@dataclass(slots=True)
class AlertingConfig:
    """Threshold-to-alert conversion settings for structured alert output."""

    anomaly_threshold: float = 0.5
    medium_multiplier: float = 1.5
    high_multiplier: float = 2.0

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, object],
        *,
        default_threshold: float = 0.5,
    ) -> "AlertingConfig":
        """Build alerting settings from a loose mapping."""

        anomaly_threshold = _as_optional_float(data.get("anomaly_threshold"))
        if anomaly_threshold is None:
            anomaly_threshold = default_threshold
        medium_multiplier = _as_optional_float(data.get("medium_multiplier"))
        if medium_multiplier is None or medium_multiplier <= 1.0:
            medium_multiplier = 1.5
        high_multiplier = _as_optional_float(data.get("high_multiplier"))
        if high_multiplier is None or high_multiplier <= medium_multiplier:
            high_multiplier = 2.0
        return cls(
            anomaly_threshold=anomaly_threshold,
            medium_multiplier=medium_multiplier,
            high_multiplier=high_multiplier,
        )


@dataclass(slots=True)
class OutputConfig:
    """Output settings for generated artifacts."""

    directory: str = "artifacts"
    save_intermediate: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "OutputConfig":
        """Build an output configuration from a loose mapping."""

        return cls(
            directory=_as_str(data.get("directory"), "artifacts"),
            save_intermediate=_as_bool(data.get("save_intermediate"), False),
        )


@dataclass(slots=True)
class PipelineConfig:
    """Top-level typed configuration passed to the pipeline runner."""

    pipeline: PipelineRuntimeConfig = field(default_factory=PipelineRuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    alerting: AlertingConfig = field(default_factory=AlertingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "PipelineConfig":
        """Build a pipeline configuration from a nested mapping."""

        return cls(
            pipeline=PipelineRuntimeConfig.from_mapping(_mapping_section(data, "pipeline")),
            data=DataConfig.from_mapping(_mapping_section(data, "data")),
            preprocessing=PreprocessingConfig.from_mapping(
                _mapping_section(data, "preprocessing")
            ),
            graph=GraphConfig.from_mapping(_mapping_section(data, "graph")),
            features=FeaturesConfig.from_mapping(_mapping_section(data, "features")),
            model=ModelConfig.from_mapping(_mapping_section(data, "model")),
            training=TrainingConfig.from_mapping(_mapping_section(data, "training")),
            evaluation=EvaluationConfig.from_mapping(
                _mapping_section(data, "evaluation")
            ),
            alerting=AlertingConfig.from_mapping(
                _mapping_section(data, "alerting"),
                default_threshold=_as_optional_float(
                    _mapping_section(data, "evaluation").get("anomaly_threshold")
                )
                or 0.5,
            ),
            output=OutputConfig.from_mapping(_mapping_section(data, "output")),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load a pipeline configuration from a YAML file."""

        import yaml

        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, Mapping):
            raise ValueError("Pipeline configuration root must be a mapping.")
        return cls.from_mapping(payload)

    def with_overrides(
        self,
        *,
        input_path: str | None = None,
        output_directory: str | None = None,
        run_name: str | None = None,
    ) -> "PipelineConfig":
        """Return a copy of the config with selected CLI-level overrides applied."""

        return PipelineConfig(
            pipeline=PipelineRuntimeConfig(
                run_name=run_name or self.pipeline.run_name,
                seed=self.pipeline.seed,
            ),
            data=DataConfig(
                input_path=input_path or self.data.input_path,
                format=self.data.format,
            ),
            preprocessing=PreprocessingConfig(
                window_size=self.preprocessing.window_size,
                short_flow_thresholds=ShortFlowThresholds(
                    packet_count_lt=self.preprocessing.short_flow_thresholds.packet_count_lt,
                    byte_count_lt=self.preprocessing.short_flow_thresholds.byte_count_lt,
                    duration_seconds_lt=self.preprocessing.short_flow_thresholds.duration_seconds_lt,
                ),
            ),
            graph=GraphConfig(
                time_window_seconds=self.graph.time_window_seconds,
                directed=self.graph.directed,
                association_edges=AssociationEdgeConfig(
                    enable_same_src_ip=self.graph.association_edges.enable_same_src_ip,
                    enable_same_dst_subnet=self.graph.association_edges.enable_same_dst_subnet,
                    dst_subnet_prefix=self.graph.association_edges.dst_subnet_prefix,
                ),
            ),
            features=FeaturesConfig(
                normalization=FeatureNormalizationConfig(
                    enabled=self.features.normalization.enabled,
                    method=self.features.normalization.method,
                    exclude_node_fields=self.features.normalization.exclude_node_fields,
                    exclude_edge_fields=self.features.normalization.exclude_edge_fields,
                ),
                use_graph_structural_features=self.features.use_graph_structural_features,
            ),
            model=ModelConfig(
                name=self.model.name,
                device=self.model.device,
                score_threshold=self.model.score_threshold,
                hidden_dim=self.model.hidden_dim,
                latent_dim=self.model.latent_dim,
                num_layers=self.model.num_layers,
                dropout=self.model.dropout,
                use_edge_features=self.model.use_edge_features,
                reconstruct_edge_features=self.model.reconstruct_edge_features,
                use_temporal_edge_projector=self.model.use_temporal_edge_projector,
                temporal_edge_hidden_dim=self.model.temporal_edge_hidden_dim,
                temporal_edge_field_names=self.model.temporal_edge_field_names,
                use_edge_categorical_embeddings=self.model.use_edge_categorical_embeddings,
                edge_categorical_embedding_dim=self.model.edge_categorical_embedding_dim,
                edge_categorical_bucket_size=self.model.edge_categorical_bucket_size,
            ),
            training=TrainingConfig(
                epochs=self.training.epochs,
                learning_rate=self.training.learning_rate,
                weight_decay=self.training.weight_decay,
                batch_size=self.training.batch_size,
                validation_split_ratio=self.training.validation_split_ratio,
                early_stopping_patience=self.training.early_stopping_patience,
                checkpoint_dir=self.training.checkpoint_dir,
                shuffle=self.training.shuffle,
                seed=self.training.seed,
                smoke_graph_limit=self.training.smoke_graph_limit,
            ),
            evaluation=EvaluationConfig(
                score_reduction=self.evaluation.score_reduction,
                anomaly_threshold=self.evaluation.anomaly_threshold,
                evaluation_label_field=self.evaluation.evaluation_label_field,
                checkpoint_dir=self.evaluation.checkpoint_dir,
                checkpoint_tag=self.evaluation.checkpoint_tag,
            ),
            alerting=AlertingConfig(
                anomaly_threshold=self.alerting.anomaly_threshold,
                medium_multiplier=self.alerting.medium_multiplier,
                high_multiplier=self.alerting.high_multiplier,
            ),
            output=OutputConfig(
                directory=output_directory or self.output.directory,
                save_intermediate=self.output.save_intermediate,
            ),
        )
