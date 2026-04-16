"""Graph-backed binary detection experiment for merged CSV inputs.

This module keeps the merged CSV held-out protocol identical to the tabular
baseline, but swaps the scoring backend for a minimal Graph AutoEncoder. Since
the public CIC merged CSV does not expose raw packet/endpoint/window structure,
each cleaned CSV row is adapted into a tiny one-node graph so the graph model
can still run end-to-end on the same protocol and export the same report schema.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from traffic_graph.data import (
    BinaryExperimentConfig,
    HeldOutAttackProtocolConfig,
    prepare_binary_experiment,
    prepare_heldout_attack_protocol,
)
from traffic_graph.features.graph_tensor_view import PackedGraphInput, PackedGraphMetadata
from traffic_graph.pipeline.binary_detection import (
    BinaryAttackMetricRecord,
    BinaryDetectionExportResult,
    BinaryDetectionReport,
    BinaryDetectionScoreRecord,
    export_binary_detection_report,
    summarize_binary_detection_report,
)

BinaryDetectionModelMode = Literal["tabular", "graph"]
"""The supported backend modes for the binary detection experiment."""

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


@dataclass(frozen=True, slots=True)
class GraphBinaryDetectionConfig:
    """Configuration for the graph-backed merged CSV experiment."""

    hidden_dim: int = 64
    latent_dim: int = 32
    num_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 128
    max_epochs: int = 5
    window_size: int = 60
    use_association_edges: bool = False
    use_graph_structural_features: bool = False
    use_temporal_edge_projector: bool = False
    temporal_edge_hidden_dim: int = 32
    temporal_edge_field_names: tuple[str, ...] = _DEFAULT_TEMPORAL_EDGE_FIELD_NAMES

    def to_dict(self) -> dict[str, object]:
        """Serialize the graph configuration into a JSON-friendly dictionary."""

        return {
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "window_size": self.window_size,
            "use_association_edges": self.use_association_edges,
            "use_graph_structural_features": self.use_graph_structural_features,
            "use_temporal_edge_projector": self.use_temporal_edge_projector,
            "temporal_edge_hidden_dim": self.temporal_edge_hidden_dim,
            "temporal_edge_field_names": list(self.temporal_edge_field_names),
        }


GRAPH_BINARY_DETECTION_SCORE_FIELDS: tuple[str, ...] = (
    "score_id",
    "run_id",
    "timestamp",
    "split",
    "evaluation_scope",
    "task_name",
    "sample_id",
    "row_index",
    "raw_label",
    "binary_label",
    "attack_group",
    "anomaly_score",
    "threshold",
    "is_alert",
    "feature_count",
    "metadata",
)
"""Stable field ordering for graph-mode score records."""


GRAPH_BINARY_DETECTION_REPORT_FIELDS: tuple[str, ...] = (
    "run_id",
    "dataset_name",
    "source_path",
    "created_at",
    "threshold_percentile",
    "threshold",
    "feature_columns",
    "model_n_components",
    "train_sample_count",
    "train_benign_count",
    "overall_metrics",
    "train_score_summary",
    "overall_score_summary",
    "per_attack_metrics",
    "attack_score_summaries",
    "input_artifacts",
    "artifact_paths",
    "notes",
)
"""Stable field ordering for graph-mode report serialization."""

GRAPH_SCORE_REDUCTION_PRIORITY: tuple[str, ...] = ("flow", "edge", "node", "graph")
"""Priority order used when collapsing graph outputs to sample-level scores."""

GRAPH_STRUCTURAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "graph_node_count",
    "graph_edge_count",
    "graph_client_node_count",
    "graph_server_node_count",
    "graph_aggregated_edge_count",
    "graph_communication_edge_count",
    "graph_association_edge_count",
)
"""Synthetic graph-level structural features for the merged-CSV adapter."""


@dataclass(frozen=True, slots=True)
class GraphModeScoreInput:
    """Raw graph-mode score payload for one sample.

    The payload can hold graph-level, node-level, edge-level, or flow-level
    anomaly scores. The reduction helper always prefers the most specific
    available score source in the order flow -> edge -> node -> graph.
    """

    sample_id: str
    task_name: str
    split: str
    row_index: int
    raw_label: str
    binary_label: int
    attack_group: str
    graph_score: float | None = None
    node_scores: tuple[float, ...] = ()
    edge_scores: tuple[float, ...] = ()
    flow_scores: tuple[float, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GraphModeReducedScore:
    """Sample-level graph score after applying score reduction rules."""

    sample_id: str
    task_name: str
    split: str
    row_index: int
    raw_label: str
    binary_label: int
    attack_group: str
    anomaly_score: float
    reduction_source: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GraphModeBinaryScoreBundle:
    """Structured score outputs and summaries for the graph backend."""

    run_id: str
    timestamp: str
    threshold: float
    feature_count: int
    train_reduced_scores: tuple[GraphModeReducedScore, ...]
    overall_reduced_scores: tuple[GraphModeReducedScore, ...]
    attack_reduced_scores: dict[str, tuple[GraphModeReducedScore, ...]]
    train_score_summary: dict[str, float | int]
    overall_score_summary: dict[str, float | int]
    attack_score_summaries: dict[str, dict[str, object]]
    overall_metrics: dict[str, float | None]
    per_attack_metrics: tuple[BinaryAttackMetricRecord, ...]
    overall_score_records: tuple[BinaryDetectionScoreRecord, ...]
    attack_score_records: tuple[BinaryDetectionScoreRecord, ...]
    notes: tuple[str, ...] = ()


_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _timestamp_token(value: object | None = None) -> str:
    """Normalize a timestamp-like value into a stable UTC token."""

    if value is None:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    token = token.strip("-._")
    return token or "timestamp"


def _slugify_token(value: object) -> str:
    """Convert an arbitrary token into a filesystem-safe path component."""

    token = str(value).strip()
    token = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in token)
    token = token.strip("-._")
    return token or "binary_experiment"


def _coerce_score_values(value: object | None) -> tuple[float, ...]:
    """Convert a scalar or sequence of raw scores into a tuple of floats."""

    if value is None:
        return ()
    if isinstance(value, (float, int, np.floating, np.integer)):
        return (float(value),)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        scores: list[float] = []
        for item in value:
            if item is None:
                continue
            try:
                scores.append(float(item))
            except (TypeError, ValueError):
                continue
        return tuple(scores)
    try:
        return (float(value),)
    except (TypeError, ValueError):
        return ()


def build_graph_mode_score_inputs(
    frame: pd.DataFrame,
    scores: Sequence[float],
    *,
    task_name: str,
    split: str,
    label_column: str,
    score_source: str = "graph",
) -> tuple[GraphModeScoreInput, ...]:
    """Create graph-mode score payloads from a scored frame.

    The current merged CSV graph backend produces one score per row. We encode
    that as a graph-level score for each sample. The reduction helper remains
    general so future node/edge/flow scores can be collapsed with the same API.
    """

    if len(frame) != len(scores):
        raise ValueError("Frame length and score length must match.")
    payloads: list[GraphModeScoreInput] = []
    for row_index, (score, row) in enumerate(zip(scores, frame.itertuples(index=False))):
        raw_label = str(getattr(row, label_column))
        binary_label = int(getattr(row, "binary_label"))
        attack_group = raw_label if raw_label else ("BENIGN" if binary_label == 0 else "malicious")
        sample_id = f"{task_name}:{split}:{row_index}"
        metadata = {
            "score_source": score_source,
            "score_value": float(score),
        }
        payloads.append(
            GraphModeScoreInput(
                sample_id=sample_id,
                task_name=task_name,
                split=split,
                row_index=row_index,
                raw_label=raw_label,
                binary_label=binary_label,
                attack_group=attack_group,
                graph_score=float(score),
                metadata=metadata,
            )
        )
    return tuple(payloads)


def reduce_graph_scores_to_flow_or_sample_level(
    score_inputs: Sequence[GraphModeScoreInput],
) -> tuple[GraphModeReducedScore, ...]:
    """Collapse graph outputs to the sample granularity used by evaluation.

    Score priority:
    1. flow scores
    2. edge scores
    3. node scores
    4. graph score

    The current merged CSV graph backend only supplies graph scores, so the
    reduction is an identity map today while staying future-proof for raw flow
    graphs.
    """

    reduced: list[GraphModeReducedScore] = []
    for input_record in score_inputs:
        source_name = "missing"
        score_value = 0.0
        for candidate_name, candidate_value in (
            ("flow", input_record.flow_scores),
            ("edge", input_record.edge_scores),
            ("node", input_record.node_scores),
        ):
            candidate_scores = _coerce_score_values(candidate_value)
            if candidate_scores:
                source_name = candidate_name
                score_value = float(np.mean(candidate_scores))
                break
        else:
            if input_record.graph_score is not None:
                source_name = "graph"
                score_value = float(input_record.graph_score)
        metadata = dict(input_record.metadata)
        metadata["reduction_source"] = source_name
        reduced.append(
            GraphModeReducedScore(
                sample_id=input_record.sample_id,
                task_name=input_record.task_name,
                split=input_record.split,
                row_index=input_record.row_index,
                raw_label=input_record.raw_label,
                binary_label=input_record.binary_label,
                attack_group=input_record.attack_group,
                anomaly_score=score_value,
                reduction_source=source_name,
                metadata=metadata,
            )
        )
    return tuple(reduced)


def _reduced_scores_to_binary_records(
    *,
    run_id: str,
    timestamp: str,
    evaluation_scope: str,
    task_name: str,
    threshold: float,
    feature_count: int,
    reduced_scores: Sequence[GraphModeReducedScore],
) -> tuple[BinaryDetectionScoreRecord, ...]:
    """Convert reduced graph-mode scores into binary detection score records."""

    records: list[BinaryDetectionScoreRecord] = []
    for reduced in reduced_scores:
        metadata = {
            "model_mode": "graph",
            "evaluation_scope": evaluation_scope,
            "task_name": task_name,
            "split": reduced.split,
            "feature_count": feature_count,
            "row_index": reduced.row_index,
            "attack_group": reduced.attack_group,
            "reduction_source": reduced.reduction_source,
        }
        metadata.update(reduced.metadata)
        records.append(
            BinaryDetectionScoreRecord(
                score_id=f"{run_id}:{evaluation_scope}:{task_name}:{reduced.split}:{reduced.row_index}",
                run_id=run_id,
                timestamp=timestamp,
                split=reduced.split,  # type: ignore[arg-type]
                evaluation_scope=evaluation_scope,
                task_name=task_name,
                sample_id=reduced.sample_id,
                row_index=reduced.row_index,
                raw_label=reduced.raw_label,
                binary_label=reduced.binary_label,
                attack_group=reduced.attack_group,
                anomaly_score=float(reduced.anomaly_score),
                threshold=float(threshold),
                is_alert=bool(reduced.anomaly_score >= threshold),
                feature_count=feature_count,
                metadata=metadata,
            )
        )
    return tuple(records)


def export_per_attack_metrics(
    records: Sequence[BinaryAttackMetricRecord],
    path: str | Path,
) -> Path:
    """Export per-attack metrics to CSV with a stable field order."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(record) for record in records]
    if rows:
        field_order = list(rows[0].keys())
        frame = pd.DataFrame(rows, columns=field_order)
    else:
        frame = pd.DataFrame(columns=[field for field in BinaryAttackMetricRecord.__annotations__])
    frame.to_csv(output_path, index=False)
    return output_path


def summarize_graph_score_distribution(
    report: BinaryDetectionReport,
) -> str:
    """Render a graph-specific score distribution summary."""

    lines = [
        "Graph score distribution:",
        (
            "  Train: "
            f"count={int(report.train_score_summary.get('count', 0))}, "
            f"mean={float(report.train_score_summary.get('mean', 0.0)):.6f}, "
            f"median={float(report.train_score_summary.get('median', 0.0)):.6f}, "
            f"q95={float(report.train_score_summary.get('q95', 0.0)):.6f}"
        ),
        (
            "  Overall test: "
            f"count={int(report.overall_score_summary.get('count', 0))}, "
            f"mean={float(report.overall_score_summary.get('mean', 0.0)):.6f}, "
            f"median={float(report.overall_score_summary.get('median', 0.0)):.6f}, "
            f"q95={float(report.overall_score_summary.get('q95', 0.0)):.6f}"
        ),
        "  Per-attack summaries:",
    ]
    for metric in report.per_attack_metrics:
        attack_summary = report.attack_score_summaries.get(metric.task_name, {})
        lines.append(
            (
                f"    - {metric.task_name}: "
                f"sample_count={metric.sample_count}, "
                f"attack_count={metric.attack_count}, "
                f"score_median={float(attack_summary.get('median', metric.score_median)):.6f}, "
                f"score_q95={float(attack_summary.get('q95', metric.score_q95)):.6f}, "
                f"attack_labels={', '.join(metric.attack_labels)}"
            )
        )
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class GraphModeEvaluationResult:
    """Structured graph-mode evaluation outputs before persistence."""

    train_reduced_scores: tuple[GraphModeReducedScore, ...]
    overall_reduced_scores: tuple[GraphModeReducedScore, ...]
    attack_reduced_scores: dict[str, tuple[GraphModeReducedScore, ...]]
    train_score_summary: dict[str, float | int]
    overall_score_summary: dict[str, float | int]
    attack_score_summaries: dict[str, dict[str, object]]
    overall_metrics: dict[str, float | None]
    per_attack_metrics: tuple[BinaryAttackMetricRecord, ...]


def compute_graph_mode_binary_scores(
    *,
    run_id: str,
    timestamp: str,
    threshold: float,
    feature_count: int,
    train_score_inputs: Sequence[GraphModeScoreInput],
    overall_score_inputs: Sequence[GraphModeScoreInput],
    task_score_inputs: Sequence[
        tuple[str, str, Sequence[str], Sequence[GraphModeScoreInput]]
    ],
) -> GraphModeBinaryScoreBundle:
    """Compute graph-mode scores, metrics, and quantile summaries.

    This helper keeps the graph-mode reduction rules explicit while producing the
    same per-attack metric structure as the tabular baseline.
    """

    train_reduced_scores = reduce_graph_scores_to_flow_or_sample_level(train_score_inputs)
    overall_reduced_scores = reduce_graph_scores_to_flow_or_sample_level(overall_score_inputs)
    train_score_summary = _score_quantile_summary(
        [record.anomaly_score for record in train_reduced_scores]
    )
    overall_score_summary = _score_quantile_summary(
        [record.anomaly_score for record in overall_reduced_scores]
    )
    overall_metrics = _compute_binary_metrics(
        [record.binary_label for record in overall_reduced_scores],
        [record.anomaly_score for record in overall_reduced_scores],
        threshold,
    )
    per_attack_metrics: list[BinaryAttackMetricRecord] = []
    attack_score_summaries: dict[str, dict[str, object]] = {}
    attack_reduced_scores: dict[str, tuple[GraphModeReducedScore, ...]] = {}
    attack_score_records: list[BinaryDetectionScoreRecord] = []
    for task_name, requested_attack_type, attack_labels, score_inputs in task_score_inputs:
        reduced_scores = reduce_graph_scores_to_flow_or_sample_level(score_inputs)
        attack_reduced_scores[task_name] = reduced_scores
        attack_score_summaries[task_name] = _score_quantile_summary(
            [record.anomaly_score for record in reduced_scores]
        )
        per_attack_metrics.append(
            _build_task_metric_record(
                task_name=task_name,
                requested_attack_type=requested_attack_type,
                attack_labels=attack_labels,
                frame=pd.DataFrame(
                    {
                        "binary_label": [record.binary_label for record in reduced_scores]
                    }
                ),
                scores=[record.anomaly_score for record in reduced_scores],
                threshold=threshold,
            )
        )
        attack_score_records.extend(
            _reduced_scores_to_binary_records(
                run_id=run_id,
                timestamp=timestamp,
                evaluation_scope="heldout_attack",
                task_name=task_name,
                threshold=threshold,
                feature_count=feature_count,
                reduced_scores=reduced_scores,
            )
        )
    overall_score_records = _reduced_scores_to_binary_records(
        run_id=run_id,
        timestamp=timestamp,
        evaluation_scope="overall",
        task_name="overall_test",
        threshold=threshold,
        feature_count=feature_count,
        reduced_scores=overall_reduced_scores,
    )
    return GraphModeBinaryScoreBundle(
        run_id=run_id,
        timestamp=timestamp,
        threshold=threshold,
        feature_count=feature_count,
        train_reduced_scores=train_reduced_scores,
        overall_reduced_scores=overall_reduced_scores,
        attack_reduced_scores=attack_reduced_scores,
        train_score_summary=train_score_summary,
        overall_score_summary=overall_score_summary,
        attack_score_summaries=attack_score_summaries,
        overall_metrics=overall_metrics,
        per_attack_metrics=tuple(per_attack_metrics),
        overall_score_records=tuple(overall_score_records),
        attack_score_records=tuple(attack_score_records),
        notes=(
            "Graph score reduction priority is flow -> edge -> node -> graph.",
            "Current merged-CSV graph mode uses graph-level scores per sample, so reduction is an identity map.",
        ),
    )


def _is_torch_available() -> bool:
    """Return whether PyTorch can be imported."""

    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def _require_graph_backend() -> tuple[object, object]:
    """Import the graph model stack lazily."""

    try:
        import torch
        from traffic_graph.models.gae import GraphAutoEncoder
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Graph binary detection requires PyTorch to be installed."
        ) from exc
    return torch, GraphAutoEncoder


def _select_graph_feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Select stable numeric feature columns for graph-mode training and scoring."""

    feature_columns: list[str] = []
    for column in frame.columns:
        if column in {"binary_label"}:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            feature_columns.append(str(column))
    return tuple(feature_columns)


def _frame_to_numeric_matrix(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> np.ndarray:
    """Convert a frame into a dense float32 matrix over the selected columns."""

    if not feature_columns:
        raise ValueError("No numeric feature columns were available for graph mode.")
    columns: list[np.ndarray] = []
    for column in feature_columns:
        if column not in frame.columns:
            columns.append(np.zeros(len(frame), dtype=np.float32))
            continue
        series = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        columns.append(series.to_numpy(dtype=np.float32, copy=False))
    if not columns:
        return np.zeros((len(frame), 0), dtype=np.float32)
    matrix = np.column_stack(columns).astype(np.float32, copy=False)
    return matrix


def _augment_graph_feature_frame(
    frame: pd.DataFrame,
    *,
    graph_config: GraphBinaryDetectionConfig,
) -> pd.DataFrame:
    """Add stable graph-summary columns when structural features are enabled."""

    if not graph_config.use_graph_structural_features:
        return frame.copy(deep=True)

    augmented = frame.copy(deep=True)
    row_count = len(augmented)
    if row_count == 0:
        for column in GRAPH_STRUCTURAL_FEATURE_COLUMNS:
            augmented[column] = []
        return augmented

    augmented["graph_node_count"] = 1
    augmented["graph_edge_count"] = 1
    augmented["graph_client_node_count"] = 1
    augmented["graph_server_node_count"] = 0
    augmented["graph_aggregated_edge_count"] = 0
    augmented["graph_communication_edge_count"] = 1
    augmented["graph_association_edge_count"] = 0
    return augmented


def _build_row_graph(
    feature_vector: np.ndarray,
    *,
    sample_id: str,
    row_index: int,
    raw_label: str,
    binary_label: int,
    task_name: str,
    split_name: str,
    feature_columns: Sequence[str],
    attack_group: str,
) -> PackedGraphInput:
    """Build a one-node graph representation for one merged CSV row."""

    node_features = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
    edge_features = np.zeros((1, 1), dtype=np.float32)
    edge_index = np.asarray([[0], [0]], dtype=np.int64)
    metadata = PackedGraphMetadata(
        window_index=row_index,
        window_start=_EPOCH_UTC,
        window_end=_EPOCH_UTC,
        node_count=1,
        edge_count=1,
        communication_edge_count=1,
        association_edge_count=0,
    )
    node_feature_fields = tuple(str(column) for column in feature_columns)
    return PackedGraphInput(
        node_features=node_features,
        edge_features=edge_features,
        edge_index=edge_index,
        node_ids=(sample_id,),
        edge_ids=(f"{sample_id}:self_loop",),
        node_id_to_index={sample_id: 0},
        edge_id_to_index={f"{sample_id}:self_loop": 0},
        edge_types=(0,),
        node_feature_fields=node_feature_fields,
        edge_feature_fields=("self_loop_weight",),
        node_discrete_mask=tuple(False for _ in node_feature_fields),
        edge_discrete_mask=(False,),
        metadata=metadata,
    )


def _iter_row_graphs(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    split_name: str,
    task_name: str,
    label_column: str,
    batch_size: int,
) -> Sequence[list[tuple[PackedGraphInput, dict[str, object]]]]:
    """Yield graph batches and row metadata for a cleaned frame."""

    if frame.empty:
        return []
    numeric_matrix = _frame_to_numeric_matrix(frame, feature_columns)
    row_count = len(frame)
    batches: list[list[tuple[PackedGraphInput, dict[str, object]]]] = []
    for start in range(0, row_count, batch_size):
        stop = min(start + batch_size, row_count)
        batch: list[tuple[PackedGraphInput, dict[str, object]]] = []
        for row_index in range(start, stop):
            row = frame.iloc[row_index]
            raw_label = str(row[label_column])
            binary_label = int(row["binary_label"])
            sample_id = f"{task_name}:{split_name}:{row_index}"
            attack_group = raw_label if raw_label else ("BENIGN" if binary_label == 0 else "malicious")
            graph = _build_row_graph(
                numeric_matrix[row_index],
                sample_id=sample_id,
                row_index=row_index,
                raw_label=raw_label,
                binary_label=binary_label,
                task_name=task_name,
                split_name=split_name,
                feature_columns=feature_columns,
                attack_group=attack_group,
            )
            batch.append(
                (
                    graph,
                    {
                        "sample_id": sample_id,
                        "row_index": row_index,
                        "raw_label": raw_label,
                        "binary_label": binary_label,
                        "attack_group": attack_group,
                        "task_name": task_name,
                        "split_name": split_name,
                    },
                )
            )
        batches.append(batch)
    return batches


def _batch_graphs(
    batch: Sequence[tuple[PackedGraphInput, dict[str, object]]],
) -> tuple[list[PackedGraphInput], list[dict[str, object]]]:
    """Split graph rows and row metadata from an internal batch structure."""

    graphs = [item[0] for item in batch]
    metadata = [item[1] for item in batch]
    return graphs, metadata


def _graph_reconstruction_scores(
    model: object,
    graph_batches: Sequence[list[tuple[PackedGraphInput, dict[str, object]]]],
) -> list[float]:
    """Score one or more batches of one-node graphs with the trained model."""

    torch, _ = _require_graph_backend()
    scores: list[float] = []
    model.eval()  # type: ignore[attr-defined]
    with torch.no_grad():
        for batch in graph_batches:
            if not batch:
                continue
            graphs, _ = _batch_graphs(batch)
            output = model(graphs)  # type: ignore[call-arg]
            continuous_mask = ~output.tensor_batch.node_discrete_mask
            if bool(torch.any(continuous_mask).item()):
                residual = (
                    output.reconstructed_node_features[:, continuous_mask]
                    - output.tensor_batch.node_features[:, continuous_mask]
                )
            else:
                residual = (
                    output.reconstructed_node_features
                    - output.tensor_batch.node_features
                )
            node_error = torch.mean(residual**2, dim=1)
            scores.extend(node_error.detach().cpu().tolist())
    return scores


def _train_graph_autoencoder(
    train_batches: Sequence[list[tuple[PackedGraphInput, dict[str, object]]]],
    *,
    node_input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    random_seed: int,
    use_temporal_edge_projector: bool = False,
    temporal_edge_hidden_dim: int = 32,
    temporal_edge_field_names: tuple[str, ...] = _DEFAULT_TEMPORAL_EDGE_FIELD_NAMES,
) -> object:
    """Fit a minimal graph autoencoder on the benign training batches."""

    if not train_batches:
        raise ValueError("Graph training requires at least one benign training batch.")
    torch, GraphAutoEncoder = _require_graph_backend()
    torch.manual_seed(random_seed)
    model = GraphAutoEncoder(
        node_input_dim=node_input_dim,
        edge_input_dim=1,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_edge_features=False,
        reconstruct_edge_features=False,
        use_temporal_edge_projector=use_temporal_edge_projector,
        temporal_edge_hidden_dim=temporal_edge_hidden_dim,
        temporal_edge_field_names=temporal_edge_field_names,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_fn = torch.nn.MSELoss()
    rng = np.random.default_rng(random_seed)
    graph_count = sum(len(batch) for batch in train_batches)
    if graph_count == 0:
        raise ValueError("Graph training requires at least one benign graph.")
    train_graphs: list[tuple[PackedGraphInput, dict[str, object]]] = [
        item for batch in train_batches for item in batch
    ]
    batch_size = max(1, min(batch_size, len(train_graphs)))
    for _epoch in range(max_epochs):
        order = rng.permutation(len(train_graphs))
        model.train()
        for start in range(0, len(order), batch_size):
            batch_indices = order[start : start + batch_size]
            batch = [train_graphs[index] for index in batch_indices]
            graphs, _ = _batch_graphs(batch)
            output = model(graphs)  # type: ignore[call-arg]
            continuous_mask = ~output.tensor_batch.node_discrete_mask
            if bool(torch.any(continuous_mask).item()):
                reconstructed = output.reconstructed_node_features[:, continuous_mask]
                target = output.tensor_batch.node_features[:, continuous_mask]
            else:
                reconstructed = output.reconstructed_node_features
                target = output.tensor_batch.node_features
            loss = loss_fn(
                reconstructed,
                target,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def _score_quantile_summary(scores: Sequence[float]) -> dict[str, float | int]:
    """Compute a compact score distribution summary."""

    if not scores:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "q25": 0.0,
            "median": 0.0,
            "q75": 0.0,
            "q95": 0.0,
            "max": 0.0,
        }
    array = np.asarray(scores, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.5)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _compute_binary_metrics(
    y_true: Sequence[int],
    scores: Sequence[float],
    threshold: float,
) -> dict[str, float | None]:
    """Compute standard binary anomaly-detection metrics."""

    y_true_array = np.asarray(y_true, dtype=int)
    score_array = np.asarray(scores, dtype=float)
    predicted = (score_array >= threshold).astype(int)
    precision = float(precision_score(y_true_array, predicted, zero_division=0))
    recall = float(recall_score(y_true_array, predicted, zero_division=0))
    f1 = float(f1_score(y_true_array, predicted, zero_division=0))
    false_positive_rate: float | None
    negatives = y_true_array == 0
    positives = y_true_array == 1
    tn = int(np.sum((predicted == 0) & negatives))
    fp = int(np.sum((predicted == 1) & negatives))
    if (tn + fp) > 0:
        false_positive_rate = float(fp / (tn + fp))
    else:
        false_positive_rate = None
    roc_auc: float | None
    pr_auc: float | None
    if len(np.unique(y_true_array)) > 1:
        from sklearn.metrics import average_precision_score, roc_auc_score

        roc_auc = float(roc_auc_score(y_true_array, score_array))
        pr_auc = float(average_precision_score(y_true_array, score_array))
    else:
        roc_auc = None
        pr_auc = None
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
    }


def _build_score_records(
    *,
    run_id: str,
    timestamp: str,
    split: str,
    evaluation_scope: str,
    task_name: str,
    frame: pd.DataFrame,
    scores: Sequence[float],
    threshold: float,
    label_column: str,
    feature_count: int,
) -> list[BinaryDetectionScoreRecord]:
    """Convert scored rows into stable score records."""

    records: list[BinaryDetectionScoreRecord] = []
    for row_index, (score, row) in enumerate(zip(scores, frame.itertuples(index=False))):
        raw_label = str(getattr(row, label_column))
        binary_label = int(getattr(row, "binary_label"))
        attack_group = raw_label if raw_label else ("BENIGN" if binary_label == 0 else "malicious")
        sample_id = f"{task_name}:{split}:{row_index}"
        metadata = {
            "model_mode": "graph",
            "evaluation_scope": evaluation_scope,
            "task_name": task_name,
            "split": split,
            "feature_count": feature_count,
            "row_index": row_index,
            "attack_group": attack_group,
        }
        records.append(
            BinaryDetectionScoreRecord(
                score_id=f"{run_id}:{evaluation_scope}:{task_name}:{split}:{row_index}",
                run_id=run_id,
                timestamp=timestamp,
                split=split,  # type: ignore[arg-type]
                evaluation_scope=evaluation_scope,
                task_name=task_name,
                sample_id=sample_id,
                row_index=row_index,
                raw_label=raw_label,
                binary_label=binary_label,
                attack_group=attack_group,
                anomaly_score=float(score),
                threshold=float(threshold),
                is_alert=bool(score >= threshold),
                feature_count=feature_count,
                metadata=metadata,
            )
        )
    return records


def _build_task_metric_record(
    *,
    task_name: str,
    requested_attack_type: str,
    attack_labels: Sequence[str],
    frame: pd.DataFrame,
    scores: Sequence[float],
    threshold: float,
) -> BinaryAttackMetricRecord:
    """Build a per-attack metric record for one held-out task."""

    y_true = frame["binary_label"].astype(int).tolist()
    metrics = _compute_binary_metrics(y_true, scores, threshold)
    score_summary = _score_quantile_summary(scores)
    benign_scores = [score for score, label in zip(scores, y_true) if int(label) == 0]
    attack_scores = [score for score, label in zip(scores, y_true) if int(label) == 1]
    attack_label_tuple = tuple(sorted({str(label) for label in attack_labels}))
    return BinaryAttackMetricRecord(
        task_name=task_name,
        requested_attack_type=requested_attack_type,
        attack_labels=attack_label_tuple,
        sample_count=len(frame),
        benign_count=int(sum(1 for value in y_true if int(value) == 0)),
        attack_count=int(sum(1 for value in y_true if int(value) == 1)),
        roc_auc=metrics["roc_auc"],
        pr_auc=metrics["pr_auc"],
        precision=float(metrics["precision"] or 0.0),
        recall=float(metrics["recall"] or 0.0),
        f1=float(metrics["f1"] or 0.0),
        false_positive_rate=metrics["false_positive_rate"],
        threshold=float(threshold),
        score_min=float(score_summary["min"]),
        score_q25=float(score_summary["q25"]),
        score_median=float(score_summary["median"]),
        score_q75=float(score_summary["q75"]),
        score_q95=float(score_summary["q95"]),
        score_max=float(score_summary["max"]),
        score_mean=float(score_summary["mean"]),
        score_std=float(score_summary["std"]),
        benign_score_mean=float(np.mean(benign_scores)) if benign_scores else 0.0,
        benign_score_median=float(np.median(benign_scores)) if benign_scores else 0.0,
        attack_score_mean=float(np.mean(attack_scores)) if attack_scores else 0.0,
        attack_score_median=float(np.median(attack_scores)) if attack_scores else 0.0,
        notes=(),
    )


def summarize_graph_binary_detection_report(report: BinaryDetectionReport) -> str:
    """Render a concise human-readable summary for graph-mode binary detection."""

    base_summary = summarize_binary_detection_report(report)
    graph_distribution = summarize_graph_score_distribution(report)
    return "\n\n".join((base_summary, graph_distribution))


def run_graph_binary_detection_experiment(
    source: str | Path | pd.DataFrame,
    output_dir: str | Path,
    *,
    binary_experiment_config: BinaryExperimentConfig | None = None,
    heldout_protocol_config: HeldOutAttackProtocolConfig | None = None,
    threshold_percentile: float = 95.0,
    graph_config: GraphBinaryDetectionConfig | None = None,
    random_seed: int = 42,
    timestamp: object | None = None,
    export_formats: Sequence[str] = ("jsonl", "csv"),
) -> tuple[BinaryDetectionReport, BinaryDetectionExportResult]:
    """Run the graph-backed binary detection experiment and export a report bundle."""

    if not _is_torch_available():  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Graph binary detection requires PyTorch. Install torch to enable model_mode='graph'."
        )

    graph_runtime_config = graph_config or GraphBinaryDetectionConfig()
    binary_artifact = prepare_binary_experiment(
        source,
        binary_experiment_config or BinaryExperimentConfig(random_seed=random_seed),
    )
    heldout_artifact = prepare_heldout_attack_protocol(
        source,
        heldout_protocol_config
        or HeldOutAttackProtocolConfig(
            random_seed=random_seed,
            benign_train_ratio=0.7,
        ),
    )

    augmented_clean_frame = _augment_graph_feature_frame(
        binary_artifact.clean_frame,
        graph_config=graph_runtime_config,
    )
    feature_columns = _select_graph_feature_columns(augmented_clean_frame)
    if not feature_columns:
        raise ValueError("Graph binary detection requires at least one numeric feature column.")

    train_frame = binary_artifact.train_frame.copy(deep=True)
    if "binary_label" in train_frame.columns:
        train_frame = train_frame.loc[train_frame["binary_label"].astype(int) == 0].copy()
    if train_frame.empty:
        raise ValueError("Graph binary detection requires at least one benign training sample.")
    train_frame = _augment_graph_feature_frame(
        train_frame,
        graph_config=graph_runtime_config,
    )

    benign_scaler = StandardScaler()
    train_matrix = _frame_to_numeric_matrix(train_frame, feature_columns)
    benign_scaler.fit(train_matrix)
    train_scaled = benign_scaler.transform(train_matrix).astype(np.float32, copy=False)
    train_frame = train_frame.reset_index(drop=True)
    train_frame_scaled = train_frame.copy(deep=True)
    train_frame_scaled.loc[:, feature_columns] = train_scaled
    train_batches = _iter_row_graphs(
        train_frame_scaled,
        feature_columns=feature_columns,
        split_name="train",
        task_name="overall_train",
        label_column=binary_artifact.label_column,
        batch_size=graph_runtime_config.batch_size,
    )
    train_model = _train_graph_autoencoder(
        train_batches,
        node_input_dim=len(feature_columns),
        hidden_dim=graph_runtime_config.hidden_dim,
        latent_dim=graph_runtime_config.latent_dim,
        num_layers=graph_runtime_config.num_layers,
        dropout=graph_runtime_config.dropout,
        learning_rate=graph_runtime_config.learning_rate,
        weight_decay=graph_runtime_config.weight_decay,
        batch_size=graph_runtime_config.batch_size,
        max_epochs=graph_runtime_config.max_epochs,
        random_seed=random_seed,
        use_temporal_edge_projector=graph_runtime_config.use_temporal_edge_projector,
        temporal_edge_hidden_dim=graph_runtime_config.temporal_edge_hidden_dim,
        temporal_edge_field_names=graph_runtime_config.temporal_edge_field_names,
    )

    # Rebuild the train batches with scaled values for scoring and threshold calibration.
    train_batches_for_scoring = _iter_row_graphs(
        train_frame_scaled,
        feature_columns=feature_columns,
        split_name="train",
        task_name="overall_train",
        label_column=binary_artifact.label_column,
        batch_size=graph_runtime_config.batch_size,
    )
    train_scores = _graph_reconstruction_scores(train_model, train_batches_for_scoring)
    threshold = float(np.quantile(np.asarray(train_scores, dtype=float), threshold_percentile / 100.0))

    overall_test_frame = binary_artifact.test_frame.reset_index(drop=True).copy(deep=True)
    overall_test_frame = _augment_graph_feature_frame(
        overall_test_frame,
        graph_config=graph_runtime_config,
    )
    overall_test_matrix = benign_scaler.transform(
        _frame_to_numeric_matrix(overall_test_frame, feature_columns)
    ).astype(np.float32, copy=False)
    overall_test_frame.loc[:, feature_columns] = overall_test_matrix
    overall_test_batches = _iter_row_graphs(
        overall_test_frame,
        feature_columns=feature_columns,
        split_name="test",
        task_name="overall_test",
        label_column=binary_artifact.label_column,
        batch_size=graph_runtime_config.batch_size,
    )
    overall_scores = _graph_reconstruction_scores(train_model, overall_test_batches)
    overall_metrics = _compute_binary_metrics(
        overall_test_frame["binary_label"].astype(int).tolist(),
        overall_scores,
        threshold,
    )

    per_attack_metrics: list[BinaryAttackMetricRecord] = []
    attack_score_records: list[BinaryDetectionScoreRecord] = []
    attack_score_summaries: dict[str, dict[str, object]] = {}
    for task_artifact in heldout_artifact.task_artifacts:
        task_frame = task_artifact.test_frame.reset_index(drop=True).copy(deep=True)
        if task_frame.empty:
            continue
        task_frame = _augment_graph_feature_frame(
            task_frame,
            graph_config=graph_runtime_config,
        )
        task_matrix = benign_scaler.transform(
            _frame_to_numeric_matrix(task_frame, feature_columns)
        ).astype(np.float32, copy=False)
        task_frame.loc[:, feature_columns] = task_matrix
        task_batches = _iter_row_graphs(
            task_frame,
            feature_columns=feature_columns,
            split_name="test",
            task_name=task_artifact.task_name,
            label_column=binary_artifact.label_column,
            batch_size=graph_runtime_config.batch_size,
        )
        task_scores = _graph_reconstruction_scores(train_model, task_batches)
        attack_score_summaries[task_artifact.task_name] = _score_quantile_summary(task_scores)
        per_attack_metrics.append(
            _build_task_metric_record(
                task_name=task_artifact.task_name,
                requested_attack_type=task_artifact.requested_attack_type,
                attack_labels=task_artifact.attack_labels,
                frame=task_frame,
                scores=task_scores,
                threshold=threshold,
            )
        )
        attack_score_records.extend(
            _build_score_records(
                run_id=f"graph-binary-experiment:{binary_artifact.dataset_name}:{_timestamp_token(timestamp or binary_artifact.created_at)}",
                timestamp=_timestamp_token(timestamp or binary_artifact.created_at),
                split="test",
                evaluation_scope="heldout_attack",
                task_name=task_artifact.task_name,
                frame=task_frame,
                scores=task_scores,
                threshold=threshold,
                label_column=binary_artifact.label_column,
                feature_count=len(feature_columns),
            )
        )

    run_timestamp = _timestamp_token(timestamp or binary_artifact.created_at)
    run_id = f"graph-binary-experiment:{binary_artifact.dataset_name}:{run_timestamp}"
    overall_score_records = _build_score_records(
        run_id=run_id,
        timestamp=run_timestamp,
        split="test",
        evaluation_scope="overall",
        task_name="overall_test",
        frame=overall_test_frame,
        scores=overall_scores,
        threshold=threshold,
        label_column=binary_artifact.label_column,
        feature_count=len(feature_columns),
    )
    train_score_summary = _score_quantile_summary(train_scores)
    overall_score_summary = _score_quantile_summary(overall_scores)
    report = BinaryDetectionReport(
        run_id=run_id,
        dataset_name=binary_artifact.dataset_name,
        source_path=Path(source).as_posix() if isinstance(source, (str, Path)) else "in-memory-frame",
        created_at=run_timestamp,
        threshold_percentile=float(threshold_percentile),
        threshold=threshold,
        feature_columns=tuple(feature_columns),
        model_n_components=graph_runtime_config.latent_dim,
        train_sample_count=int(len(train_frame)),
        train_benign_count=int(len(train_frame)),
        overall_metrics=overall_metrics,
        train_score_summary=train_score_summary,
        overall_score_summary=overall_score_summary,
        per_attack_metrics=tuple(per_attack_metrics),
        attack_score_summaries=attack_score_summaries,
        input_artifacts={
            "model_mode": "graph",
            "graph_model_config": graph_runtime_config.to_dict(),
            "binary_experiment": binary_artifact.summary.to_dict(),
            "heldout_protocol": heldout_artifact.summary.to_dict(),
        },
        artifact_paths={},
        notes=(
            "Graph mode uses a one-node graph adapter for each cleaned merged CSV row because "
            "the public merged dataset does not contain raw endpoint/window structure.",
            "Graph score reduction priority is flow -> edge -> node -> graph; the current merged CSV adapter is graph-only, so reduction is an identity map.",
            (
                "Graph structural feature columns were "
                + (
                    "included"
                    if graph_runtime_config.use_graph_structural_features
                    else "disabled"
                )
                + " in the merged-CSV adapter."
            ),
            (
                "Graph ablation settings: "
                f"window_size={graph_runtime_config.window_size}, "
                f"use_association_edges={graph_runtime_config.use_association_edges}, "
                f"use_graph_structural_features={graph_runtime_config.use_graph_structural_features}."
            ),
            "Training is restricted to benign rows only and labels are used only for evaluation.",
            "The graph backend uses the minimal Graph AutoEncoder mainline.",
        ),
    )
    export_result = export_binary_detection_report(
        report,
        output_dir,
        overall_scores=overall_score_records,
        attack_scores=attack_score_records,
        export_formats=export_formats,
        binary_input_manifest=None,
        heldout_input_manifest=None,
    )
    return report, export_result


__all__ = [
    "BinaryDetectionModelMode",
    "GraphModeBinaryScoreBundle",
    "GraphModeEvaluationResult",
    "GraphModeReducedScore",
    "GraphModeScoreInput",
    "GRAPH_BINARY_DETECTION_REPORT_FIELDS",
    "GRAPH_BINARY_DETECTION_SCORE_FIELDS",
    "GRAPH_SCORE_REDUCTION_PRIORITY",
    "GraphBinaryDetectionConfig",
    "build_graph_mode_score_inputs",
    "compute_graph_mode_binary_scores",
    "export_per_attack_metrics",
    "reduce_graph_scores_to_flow_or_sample_level",
    "run_graph_binary_detection_experiment",
    "summarize_graph_score_distribution",
    "summarize_graph_binary_detection_report",
]
