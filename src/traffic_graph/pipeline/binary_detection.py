"""Binary detection experiment runner for merged CSV datasets.

This module stitches together the merged-CSV binary experiment input builder,
the held-out attack evaluation protocol, and a lightweight unsupervised
tabular reconstruction scorer. It intentionally keeps labels out of training
and only uses them for evaluation and reporting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from traffic_graph.data import (
    BinaryExperimentArtifact,
    BinaryExperimentConfig,
    DEFAULT_HELD_OUT_ATTACK_TYPES,
    HeldOutAttackProtocolArtifact,
    HeldOutAttackProtocolConfig,
    export_binary_experiment,
    export_heldout_attack_protocol,
    prepare_binary_experiment,
    prepare_heldout_attack_protocol,
    summarize_binary_experiment_text,
    summarize_heldout_attack_protocol_text,
)

BinaryDetectionSplit = Literal["train", "overall_test", "task_test"]
"""Split labels used by the binary detection experiment."""

BINARY_DETECTION_SCORE_FIELDS: tuple[str, ...] = (
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
"""Stable field order used for exported score tables."""

BINARY_ATTACK_METRIC_FIELDS: tuple[str, ...] = (
    "task_name",
    "requested_attack_type",
    "attack_labels",
    "sample_count",
    "benign_count",
    "attack_count",
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "threshold",
    "score_min",
    "score_q25",
    "score_median",
    "score_q75",
    "score_q95",
    "score_max",
    "score_mean",
    "score_std",
    "benign_score_mean",
    "benign_score_median",
    "attack_score_mean",
    "attack_score_median",
    "notes",
)
"""Stable field order used for per-attack metric tables."""

BINARY_DETECTION_REPORT_FIELDS: tuple[str, ...] = (
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
"""Stable field order used in the high-level report payload."""

_DEFAULT_THRESHOLD_PERCENTILE = 95.0
_DEFAULT_MAX_COMPONENTS = 10
_DEFAULT_RANDOM_SEED = 42
_DEFAULT_OUTPUT_NAME = "binary_detection"


def _timestamp_token(value: object | None = None) -> str:
    """Normalize a timestamp into a stable UTC token."""

    if value is None:
        moment = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        moment = value.astimezone(timezone.utc)
    else:
        token = str(value).strip()
        token = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in token)
        token = token.strip("-._")
        return token or "timestamp"
    return moment.strftime("%Y%m%dT%H%M%SZ")


def _safe_float(value: object | None, default: float) -> float:
    """Convert a value to float while preserving a fallback."""

    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object | None, default: int) -> int:
    """Convert a value to int while preserving a fallback."""

    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _json_scalar(value: object) -> object:
    """Convert numpy scalar-like values to Python primitives when possible."""

    if hasattr(value, "item"):
        try:
            return value.item()  # type: ignore[call-arg]
        except Exception:
            return value
    return value


def _json_dumps(value: object) -> str:
    """Serialize a Python object into stable JSON."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return it."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _feature_columns_from_frame(frame: pd.DataFrame, label_column: str) -> tuple[str, ...]:
    """Infer numeric feature columns from a merged CSV frame."""

    numeric_columns = list(frame.select_dtypes(include=[np.number]).columns)
    filtered = [
        column
        for column in numeric_columns
        if column not in {label_column, "binary_label"}
    ]
    return tuple(filtered)


def _build_feature_matrix(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> np.ndarray:
    """Build a numeric feature matrix aligned to the requested columns."""

    if not feature_columns:
        return np.zeros((len(frame), 1), dtype=float)
    feature_frame = frame.reindex(columns=list(feature_columns), fill_value=0.0)
    numeric_frame = feature_frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return numeric_frame.to_numpy(dtype=float, copy=False)


def _quantile_summary(
    values: np.ndarray | Sequence[float],
    *,
    quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
) -> dict[str, float | int]:
    """Compute a compact quantile summary for a score array."""

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "q05": 0.0,
            "q25": 0.0,
            "median": 0.0,
            "q75": 0.0,
            "q95": 0.0,
            "max": 0.0,
        }
    summary: dict[str, float | int] = {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }
    names = ("q05", "q25", "median", "q75", "q95")
    for name, quantile in zip(names, quantiles, strict=False):
        summary[name] = float(np.quantile(array, quantile))
    return summary


def _metric_or_none(values: Sequence[object]) -> float | None:
    """Return a float metric or ``None`` when not computable."""

    if not values:
        return None
    first = values[0]
    if first is None:
        return None
    return float(first)  # type: ignore[arg-type]


def _compute_binary_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | None]:
    """Compute standard binary detection metrics from labels and scores."""

    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)
    try:
        roc_auc = float(roc_auc_score(y_true, scores))
    except ValueError:
        roc_auc = None
    try:
        pr_auc = float(average_precision_score(y_true, scores))
    except ValueError:
        pr_auc = None
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    false_positive_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else None
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "threshold": float(threshold),
    }


def _make_score_record(
    *,
    run_id: str,
    timestamp: str,
    split: BinaryDetectionSplit,
    evaluation_scope: str,
    task_name: str,
    row_index: int,
    sample_id: str,
    raw_label: object,
    binary_label: int,
    attack_group: str,
    anomaly_score: float,
    threshold: float,
    feature_count: int,
    metadata: Mapping[str, object],
) -> "BinaryDetectionScoreRecord":
    """Build one score record with stable identity fields."""

    is_alert = bool(anomaly_score >= threshold)
    return BinaryDetectionScoreRecord(
        score_id=f"{run_id}:{evaluation_scope}:{task_name}:{sample_id}",
        run_id=run_id,
        timestamp=timestamp,
        split=split,
        evaluation_scope=evaluation_scope,
        task_name=task_name,
        sample_id=sample_id,
        row_index=row_index,
        raw_label=str(raw_label),
        binary_label=int(binary_label),
        attack_group=attack_group,
        anomaly_score=float(anomaly_score),
        threshold=float(threshold),
        is_alert=is_alert,
        feature_count=feature_count,
        metadata=dict(metadata),
    )


def _score_table_to_frame(records: Sequence["BinaryDetectionScoreRecord"]) -> pd.DataFrame:
    """Convert score records into a stable tabular representation."""

    return pd.DataFrame([record.to_csv_dict() for record in records])


def _attack_label_breakdown(
    records: Sequence["BinaryDetectionScoreRecord"],
    *,
    task_name: str,
    requested_attack_type: str,
    attack_labels: Sequence[str],
) -> list[dict[str, object]]:
    """Summarize score distributions per raw attack label for one task."""

    if not records:
        return []
    frame = _score_table_to_frame(records)
    breakdown_rows: list[dict[str, object]] = []
    for raw_label, group in frame.groupby("raw_label", dropna=False):
        scores = group["anomaly_score"].astype(float).to_numpy()
        label_value = str(raw_label)
        row = {
            "task_name": task_name,
            "requested_attack_type": requested_attack_type,
            "attack_labels": list(attack_labels),
            "raw_label": label_value,
            "count": int(len(group)),
            "binary_label": int(group["binary_label"].iloc[0]) if len(group) else 0,
            "score_mean": float(scores.mean()) if len(scores) else 0.0,
            "score_median": float(np.median(scores)) if len(scores) else 0.0,
            "score_q25": float(np.quantile(scores, 0.25)) if len(scores) else 0.0,
            "score_q75": float(np.quantile(scores, 0.75)) if len(scores) else 0.0,
            "score_q95": float(np.quantile(scores, 0.95)) if len(scores) else 0.0,
            "threshold": float(group["threshold"].iloc[0]) if len(group) else 0.0,
            "alert_rate": float(group["is_alert"].mean()) if len(group) else 0.0,
        }
        breakdown_rows.append(row)
    return breakdown_rows


@dataclass(slots=True)
class BinaryDetectionModel:
    """Lightweight PCA reconstruction scorer for tabular merged CSV data."""

    scaler: StandardScaler
    pca: PCA
    feature_columns: tuple[str, ...]
    threshold_percentile: float
    threshold: float = 0.0

    def fit_threshold(self, frame: pd.DataFrame) -> np.ndarray:
        """Fit the threshold on a benign training frame and return train scores."""

        matrix = _build_feature_matrix(frame, self.feature_columns)
        if matrix.size == 0 or matrix.shape[0] == 0:
            raise ValueError("The binary detection model cannot be fitted on an empty frame.")
        scaled = self.scaler.fit_transform(matrix)
        transformed = self.pca.fit_transform(scaled)
        reconstructed_scaled = self.pca.inverse_transform(transformed)
        reconstructed = self.scaler.inverse_transform(reconstructed_scaled)
        scores = np.mean((matrix - reconstructed) ** 2, axis=1)
        self.threshold = float(np.quantile(scores, self.threshold_percentile / 100.0))
        return scores

    def score_frame(self, frame: pd.DataFrame) -> np.ndarray:
        """Score a frame using the fitted reconstruction model."""

        matrix = _build_feature_matrix(frame, self.feature_columns)
        scaled = self.scaler.transform(matrix)
        transformed = self.pca.transform(scaled)
        reconstructed_scaled = self.pca.inverse_transform(transformed)
        reconstructed = self.scaler.inverse_transform(reconstructed_scaled)
        scores = np.mean((matrix - reconstructed) ** 2, axis=1)
        return scores.astype(float, copy=False)


@dataclass(frozen=True, slots=True)
class BinaryAttackMetricRecord:
    """Structured per-attack evaluation summary."""

    task_name: str
    requested_attack_type: str
    attack_labels: tuple[str, ...]
    sample_count: int
    benign_count: int
    attack_count: int
    roc_auc: float | None
    pr_auc: float | None
    precision: float
    recall: float
    f1: float
    false_positive_rate: float | None
    threshold: float
    score_min: float
    score_q25: float
    score_median: float
    score_q75: float
    score_q95: float
    score_max: float
    score_mean: float
    score_std: float
    benign_score_mean: float
    benign_score_median: float
    attack_score_mean: float
    attack_score_median: float
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize the metric record into a JSON-friendly dictionary."""

        return {
            "task_name": self.task_name,
            "requested_attack_type": self.requested_attack_type,
            "attack_labels": list(self.attack_labels),
            "sample_count": self.sample_count,
            "benign_count": self.benign_count,
            "attack_count": self.attack_count,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "false_positive_rate": self.false_positive_rate,
            "threshold": self.threshold,
            "score_min": self.score_min,
            "score_q25": self.score_q25,
            "score_median": self.score_median,
            "score_q75": self.score_q75,
            "score_q95": self.score_q95,
            "score_max": self.score_max,
            "score_mean": self.score_mean,
            "score_std": self.score_std,
            "benign_score_mean": self.benign_score_mean,
            "benign_score_median": self.benign_score_median,
            "attack_score_mean": self.attack_score_mean,
            "attack_score_median": self.attack_score_median,
            "notes": list(self.notes),
        }

    def to_csv_row(self) -> dict[str, object]:
        """Serialize the metric record into a CSV-friendly row."""

        row = self.to_dict().copy()
        row["attack_labels"] = "|".join(self.attack_labels)
        row["notes"] = _json_dumps(list(self.notes))
        return row


@dataclass(slots=True)
class BinaryDetectionScoreRecord:
    """Row-level anomaly score record for the merged CSV experiment."""

    score_id: str
    run_id: str
    timestamp: str
    split: BinaryDetectionSplit
    evaluation_scope: str
    task_name: str
    sample_id: str
    row_index: int
    raw_label: str
    binary_label: int
    attack_group: str
    anomaly_score: float
    threshold: float
    is_alert: bool
    feature_count: int
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize the score record into a JSON-friendly dictionary."""

        return {
            "score_id": self.score_id,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "split": self.split,
            "evaluation_scope": self.evaluation_scope,
            "task_name": self.task_name,
            "sample_id": self.sample_id,
            "row_index": self.row_index,
            "raw_label": self.raw_label,
            "binary_label": self.binary_label,
            "attack_group": self.attack_group,
            "anomaly_score": self.anomaly_score,
            "threshold": self.threshold,
            "is_alert": self.is_alert,
            "feature_count": self.feature_count,
            "metadata": dict(self.metadata),
        }

    def to_csv_dict(self) -> dict[str, object]:
        """Serialize the score record into a CSV-friendly dictionary."""

        row = self.to_dict().copy()
        row["metadata"] = _json_dumps(row["metadata"])
        return row


@dataclass(frozen=True, slots=True)
class BinaryDetectionReport:
    """Structured report for the merged CSV binary detection experiment."""

    run_id: str
    dataset_name: str
    source_path: str
    created_at: str
    threshold_percentile: float
    threshold: float
    feature_columns: tuple[str, ...]
    model_n_components: int
    train_sample_count: int
    train_benign_count: int
    overall_metrics: dict[str, float | None]
    train_score_summary: dict[str, float | int]
    overall_score_summary: dict[str, float | int]
    per_attack_metrics: tuple[BinaryAttackMetricRecord, ...]
    attack_score_summaries: dict[str, dict[str, object]]
    input_artifacts: dict[str, object] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize the report into a JSON-friendly dictionary."""

        return {
            "run_id": self.run_id,
            "dataset_name": self.dataset_name,
            "source_path": self.source_path,
            "created_at": self.created_at,
            "threshold_percentile": self.threshold_percentile,
            "threshold": self.threshold,
            "feature_columns": list(self.feature_columns),
            "model_n_components": self.model_n_components,
            "train_sample_count": self.train_sample_count,
            "train_benign_count": self.train_benign_count,
            "overall_metrics": dict(self.overall_metrics),
            "train_score_summary": dict(self.train_score_summary),
            "overall_score_summary": dict(self.overall_score_summary),
            "per_attack_metrics": [metric.to_dict() for metric in self.per_attack_metrics],
            "attack_score_summaries": dict(self.attack_score_summaries),
            "input_artifacts": dict(self.input_artifacts),
            "artifact_paths": dict(self.artifact_paths),
            "notes": list(self.notes),
        }

    def render(self) -> str:
        """Render a compact human-readable experiment report."""

        metrics = self.overall_metrics

        def _fmt(value: object) -> str:
            if value is None:
                return "n/a"
            if isinstance(value, (int, float)):
                return f"{float(value):.6f}"
            return str(value)

        lines = [
            f"Binary detection experiment: run_id={self.run_id}, dataset={self.dataset_name}",
            f"Source: {self.source_path}",
            (
                "Config: "
                f"threshold_percentile={self.threshold_percentile:.1f}, "
                f"threshold={self.threshold:.6f}, "
                f"model_n_components={self.model_n_components}"
            ),
            (
                "Training data: "
                f"samples={self.train_sample_count}, "
                f"benign={self.train_benign_count}"
            ),
            f"Feature columns: {', '.join(self.feature_columns)}",
            (
                "Overall metrics: "
                f"roc_auc={_fmt(metrics.get('roc_auc'))}, "
                f"pr_auc={_fmt(metrics.get('pr_auc'))}, "
                f"precision={_fmt(metrics.get('precision'))}, "
                f"recall={_fmt(metrics.get('recall'))}, "
                f"f1={_fmt(metrics.get('f1'))}, "
                f"false_positive_rate={_fmt(metrics.get('false_positive_rate'))}"
            ),
            (
                "Train score summary: "
                f"count={int(self.train_score_summary.get('count', 0))}, "
                f"mean={float(self.train_score_summary.get('mean', 0.0)):.6f}, "
                f"median={float(self.train_score_summary.get('median', 0.0)):.6f}, "
                f"q95={float(self.train_score_summary.get('q95', 0.0)):.6f}"
            ),
            (
                "Overall test score summary: "
                f"count={int(self.overall_score_summary.get('count', 0))}, "
                f"mean={float(self.overall_score_summary.get('mean', 0.0)):.6f}, "
                f"median={float(self.overall_score_summary.get('median', 0.0)):.6f}, "
                f"q95={float(self.overall_score_summary.get('q95', 0.0)):.6f}"
            ),
            "Per-attack metrics:",
        ]
        for metric in self.per_attack_metrics:
            lines.append(
                "  - "
                f"{metric.task_name}: attack_labels={', '.join(metric.attack_labels)}, "
                f"recall={_fmt(metric.recall)}, pr_auc={_fmt(metric.pr_auc)}, "
                f"precision={_fmt(metric.precision)}, f1={_fmt(metric.f1)}, "
                f"false_positive_rate={_fmt(metric.false_positive_rate)}, "
                f"score_median={metric.score_median:.6f}, score_q95={metric.score_q95:.6f}"
            )
        if self.notes:
            lines.append("Notes:")
            lines.extend(f"  - {note}" for note in self.notes)
        return "\n".join(lines)


@dataclass(slots=True)
class BinaryDetectionExportResult:
    """Summary returned after exporting a binary detection report."""

    run_id: str
    dataset_name: str
    created_at: str
    output_directory: str
    manifest_path: str
    metrics_summary_path: str
    per_attack_metrics_path: str
    overall_scores_path: str
    attack_scores_path: str
    artifact_paths: dict[str, str] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _summarize_detection_scores(
    records: Sequence[BinaryDetectionScoreRecord],
) -> dict[str, float | int]:
    """Summarize a flat sequence of anomaly scores."""

    if not records:
        return _quantile_summary([])
    scores = np.asarray([record.anomaly_score for record in records], dtype=float)
    return _quantile_summary(scores)


def _create_detection_model(
    *,
    feature_columns: Sequence[str],
    threshold_percentile: float,
    max_components: int,
    random_seed: int,
) -> BinaryDetectionModel:
    """Build a PCA reconstruction model with deterministic configuration."""

    feature_count = len(feature_columns)
    if feature_count <= 1:
        n_components = 1
    else:
        n_components = max(1, min(max_components, feature_count - 1))
    return BinaryDetectionModel(
        scaler=StandardScaler(),
        pca=PCA(
            n_components=n_components,
            random_state=random_seed,
            svd_solver="full",
        ),
        feature_columns=tuple(feature_columns),
        threshold_percentile=threshold_percentile,
    )


def _detect_task_metrics(
    *,
    task_name: str,
    requested_attack_type: str,
    attack_labels: Sequence[str],
    frame: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
) -> BinaryAttackMetricRecord:
    """Compute per-task metrics and score summary statistics."""

    y_true = frame["binary_label"].astype(int).to_numpy()
    metrics = _compute_binary_metrics(y_true, scores, threshold)
    benign_scores = scores[y_true == 0]
    attack_scores = scores[y_true == 1]
    score_summary = _quantile_summary(scores)
    benign_summary = _quantile_summary(benign_scores)
    attack_summary = _quantile_summary(attack_scores)
    return BinaryAttackMetricRecord(
        task_name=task_name,
        requested_attack_type=requested_attack_type,
        attack_labels=tuple(attack_labels),
        sample_count=int(len(frame)),
        benign_count=int((y_true == 0).sum()),
        attack_count=int((y_true == 1).sum()),
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
        benign_score_mean=float(benign_summary["mean"]),
        benign_score_median=float(benign_summary["median"]),
        attack_score_mean=float(attack_summary["mean"]),
        attack_score_median=float(attack_summary["median"]),
        notes=(
            f"Task '{task_name}' evaluated against {len(attack_labels)} attack labels.",
        ),
    )


def run_binary_detection_experiment(
    source: str | Path | pd.DataFrame,
    output_dir: str | Path,
    *,
    binary_experiment_config: BinaryExperimentConfig | None = None,
    heldout_protocol_config: HeldOutAttackProtocolConfig | None = None,
    threshold_percentile: float = _DEFAULT_THRESHOLD_PERCENTILE,
    max_components: int = _DEFAULT_MAX_COMPONENTS,
    random_seed: int = _DEFAULT_RANDOM_SEED,
    timestamp: object | None = None,
    export_formats: Sequence[str] = ("jsonl", "csv"),
) -> tuple[BinaryDetectionReport, BinaryDetectionExportResult]:
    """Run the merged CSV binary detection experiment end to end."""

    binary_config = binary_experiment_config or BinaryExperimentConfig()
    heldout_config = heldout_protocol_config or HeldOutAttackProtocolConfig(
        held_out_attack_types=tuple(DEFAULT_HELD_OUT_ATTACK_TYPES),
    )
    binary_artifact = prepare_binary_experiment(source, binary_config, created_at=timestamp)
    heldout_artifact = prepare_heldout_attack_protocol(source, heldout_config, created_at=timestamp)

    if binary_artifact.train_frame.empty:
        raise ValueError("The binary detection experiment requires a non-empty training split.")
    if binary_artifact.test_frame.empty:
        raise ValueError("The binary detection experiment requires a non-empty test split.")

    run_id = binary_artifact.experiment_id
    created_at = binary_artifact.created_at
    dataset_name = binary_artifact.dataset_name
    source_path = binary_artifact.source_path

    input_directory = _ensure_directory(Path(output_dir) / "input_artifacts")
    binary_input_dir = input_directory / "binary_experiment"
    heldout_input_dir = input_directory / "heldout_protocol"
    binary_input_result = export_binary_experiment(binary_artifact, binary_input_dir)
    heldout_input_result = export_heldout_attack_protocol(heldout_artifact, heldout_input_dir)

    feature_columns = _feature_columns_from_frame(
        binary_artifact.train_frame,
        binary_artifact.label_column,
    )
    model = _create_detection_model(
        feature_columns=feature_columns,
        threshold_percentile=threshold_percentile,
        max_components=max_components,
        random_seed=random_seed,
    )
    train_scores = model.fit_threshold(binary_artifact.train_frame)
    overall_test_scores = model.score_frame(binary_artifact.test_frame)
    overall_threshold = model.threshold

    overall_records: list[BinaryDetectionScoreRecord] = []
    for row_index, (index, row, score) in enumerate(
        zip(
            binary_artifact.test_frame.index,
            binary_artifact.test_frame.to_dict(orient="records"),
            overall_test_scores,
            strict=False,
        )
    ):
        sample_id = str(row.get("FlowID", f"row-{index}"))
        raw_label = row.get("Label", row.get("label", "unknown"))
        binary_label = int(row.get("binary_label", 0))
        attack_group = str(raw_label)
        overall_records.append(
            _make_score_record(
                run_id=run_id,
                timestamp=created_at,
                split="overall_test",
                evaluation_scope="overall",
                task_name="overall",
                row_index=int(index),
                sample_id=sample_id,
                raw_label=raw_label,
                binary_label=binary_label,
                attack_group=attack_group,
                anomaly_score=float(score),
                threshold=overall_threshold,
                feature_count=len(feature_columns),
                metadata=row,
            )
        )

    overall_metrics = _compute_binary_metrics(
        binary_artifact.test_frame["binary_label"].astype(int).to_numpy(),
        overall_test_scores,
        overall_threshold,
    )
    train_score_summary = _quantile_summary(train_scores)
    overall_score_summary = _quantile_summary(overall_test_scores)

    per_attack_metrics: list[BinaryAttackMetricRecord] = []
    attack_score_records: list[BinaryDetectionScoreRecord] = []
    attack_score_summaries: dict[str, dict[str, object]] = {}
    for task in heldout_artifact.task_artifacts:
        task_scores = model.score_frame(task.test_frame)
        task_records: list[BinaryDetectionScoreRecord] = []
        for index, (row_index, row, score) in enumerate(
            zip(
                task.test_frame.index,
                task.test_frame.to_dict(orient="records"),
                task_scores,
                strict=False,
            )
        ):
            sample_id = str(row.get("FlowID", f"row-{row_index}"))
            raw_label = row.get("Label", row.get("label", "unknown"))
            binary_label = int(row.get("binary_label", 0))
            task_records.append(
                _make_score_record(
                    run_id=run_id,
                    timestamp=created_at,
                    split="task_test",
                    evaluation_scope=task.task_name,
                    task_name=task.task_name,
                    row_index=int(row_index),
                    sample_id=sample_id,
                    raw_label=raw_label,
                    binary_label=binary_label,
                    attack_group=str(raw_label),
                    anomaly_score=float(score),
                    threshold=overall_threshold,
                    feature_count=len(feature_columns),
                    metadata={
                        **row,
                        "requested_attack_type": task.requested_attack_type,
                        "attack_labels": list(task.attack_labels),
                    },
                )
            )
        attack_score_records.extend(task_records)
        per_attack_metrics.append(
            _detect_task_metrics(
                task_name=task.task_name,
                requested_attack_type=task.requested_attack_type,
                attack_labels=task.attack_labels,
                frame=task.test_frame,
                scores=task_scores,
                threshold=overall_threshold,
            )
        )
        attack_score_summaries[task.task_name] = {
            "task_name": task.task_name,
            "requested_attack_type": task.requested_attack_type,
            "attack_labels": list(task.attack_labels),
            "score_summary": _quantile_summary(task_scores),
            "benign_score_summary": _quantile_summary(
                task_scores[task.test_frame["binary_label"].astype(int).to_numpy() == 0]
            ),
            "attack_score_summary": _quantile_summary(
                task_scores[task.test_frame["binary_label"].astype(int).to_numpy() == 1]
            ),
            "per_attack_label_breakdown": _attack_label_breakdown(
                task_records,
                task_name=task.task_name,
                requested_attack_type=task.requested_attack_type,
                attack_labels=task.attack_labels,
            ),
        }

    report = BinaryDetectionReport(
        run_id=run_id,
        dataset_name=dataset_name,
        source_path=source_path,
        created_at=created_at,
        threshold_percentile=threshold_percentile,
        threshold=overall_threshold,
        feature_columns=tuple(feature_columns),
        model_n_components=int(model.pca.n_components_),
        train_sample_count=len(binary_artifact.train_frame),
        train_benign_count=int((binary_artifact.train_frame["binary_label"] == 0).sum()),
        overall_metrics=overall_metrics,
        train_score_summary=train_score_summary,
        overall_score_summary=overall_score_summary,
        per_attack_metrics=tuple(per_attack_metrics),
        attack_score_summaries=attack_score_summaries,
        input_artifacts={
            "binary_experiment_manifest": binary_input_result.manifest_path,
            "heldout_protocol_manifest": heldout_input_result.manifest_path,
            "binary_experiment_task_manifests": dict(binary_input_result.artifact_paths),
            "heldout_protocol_task_manifests": dict(heldout_input_result.task_manifest_paths),
        },
        notes=tuple(
            list(binary_artifact.notes)
            + list(heldout_artifact.notes)
            + [
                "Binary detection uses a PCA reconstruction baseline over numeric columns only.",
                "Labels are only used when evaluating metrics and never during model fitting.",
            ]
        ),
    )

    export_result = export_binary_detection_report(
        report,
        output_dir,
        overall_scores=overall_records,
        attack_scores=attack_score_records,
        export_formats=export_formats,
        binary_input_manifest=binary_input_result.manifest_path,
        heldout_input_manifest=heldout_input_result.manifest_path,
    )
    return report, export_result


def _write_json(path: Path, payload: object) -> None:
    """Write a stable JSON file with a trailing newline."""

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> int:
    """Write rows to JSONL and return the row count."""

    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False, sort_keys=False)
            handle.write("\n")
            count += 1
    return count


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> int:
    """Write rows to CSV with a stable column order."""

    frame = pd.DataFrame(rows, columns=list(columns))
    frame.to_csv(path, index=False)
    return len(frame)


def _export_score_records(
    *,
    records: Sequence[BinaryDetectionScoreRecord],
    base_path: Path,
    formats: Sequence[str],
    artifact_paths: dict[str, str],
    row_counts: dict[str, int],
    notes: list[str],
    artifact_name: str,
) -> None:
    """Export a sequence of score records."""

    normalized_formats = {format_name.lower() for format_name in formats}
    if "jsonl" in normalized_formats:
        jsonl_path = base_path.with_suffix(".jsonl")
        row_counts[f"{artifact_name}_jsonl"] = _write_jsonl(
            jsonl_path,
            [record.to_dict() for record in records],
        )
        artifact_paths[f"{artifact_name}_jsonl"] = jsonl_path.as_posix()
    if "csv" in normalized_formats:
        csv_path = base_path.with_suffix(".csv")
        row_counts[f"{artifact_name}_csv"] = _write_csv(
            csv_path,
            [record.to_csv_dict() for record in records],
            BINARY_DETECTION_SCORE_FIELDS,
        )
        artifact_paths[f"{artifact_name}_csv"] = csv_path.as_posix()


def export_binary_detection_report(
    report: BinaryDetectionReport,
    output_dir: str | Path,
    *,
    overall_scores: Sequence[BinaryDetectionScoreRecord],
    attack_scores: Sequence[BinaryDetectionScoreRecord],
    export_formats: Sequence[str] = ("jsonl", "csv"),
    binary_input_manifest: str | None = None,
    heldout_input_manifest: str | None = None,
) -> BinaryDetectionExportResult:
    """Export a binary detection report to stable local files."""

    layout_directory = (
        Path(output_dir) / _slugify_report_component(report.dataset_name) / report.created_at
    )
    layout_directory.mkdir(parents=True, exist_ok=True)

    artifact_paths: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    notes = list(report.notes)
    normalized_formats = {format_name.lower() for format_name in export_formats}

    _export_score_records(
        records=overall_scores,
        base_path=layout_directory / "overall_scores",
        formats=normalized_formats,
        artifact_paths=artifact_paths,
        row_counts=row_counts,
        notes=notes,
        artifact_name="overall_scores",
    )
    _export_score_records(
        records=attack_scores,
        base_path=layout_directory / "attack_scores",
        formats=normalized_formats,
        artifact_paths=artifact_paths,
        row_counts=row_counts,
        notes=notes,
        artifact_name="attack_scores",
    )

    per_attack_metrics_path = layout_directory / "per_attack_metrics.csv"
    pd.DataFrame(
        [metric.to_csv_row() for metric in report.per_attack_metrics],
        columns=list(BINARY_ATTACK_METRIC_FIELDS),
    ).to_csv(per_attack_metrics_path, index=False)
    artifact_paths["per_attack_metrics_csv"] = per_attack_metrics_path.as_posix()
    row_counts["per_attack_metrics_csv"] = len(report.per_attack_metrics)

    metrics_summary_path = layout_directory / "metrics_summary.json"
    _write_json(metrics_summary_path, report.to_dict())
    artifact_paths["metrics_summary_json"] = metrics_summary_path.as_posix()

    score_quantiles_path = layout_directory / "score_quantiles.json"
    _write_json(
        score_quantiles_path,
        {
            "train": report.train_score_summary,
            "overall_test": report.overall_score_summary,
            "per_attack": report.attack_score_summaries,
        },
    )
    artifact_paths["score_quantiles_json"] = score_quantiles_path.as_posix()

    manifest_payload = {
        "run_id": report.run_id,
        "dataset_name": report.dataset_name,
        "source_path": report.source_path,
        "created_at": report.created_at,
        "threshold_percentile": report.threshold_percentile,
        "threshold": report.threshold,
        "layout_directory": layout_directory.as_posix(),
        "input_manifests": {
            "binary_experiment_manifest": binary_input_manifest,
            "heldout_protocol_manifest": heldout_input_manifest,
        },
        "artifact_paths": artifact_paths,
        "row_counts": row_counts,
        "notes": notes,
        "formats": sorted(normalized_formats),
    }
    manifest_path = layout_directory / "manifest.json"
    _write_json(manifest_path, manifest_payload)
    artifact_paths["manifest_json"] = manifest_path.as_posix()

    return BinaryDetectionExportResult(
        run_id=report.run_id,
        dataset_name=report.dataset_name,
        created_at=report.created_at,
        output_directory=layout_directory.as_posix(),
        manifest_path=manifest_path.as_posix(),
        metrics_summary_path=metrics_summary_path.as_posix(),
        per_attack_metrics_path=per_attack_metrics_path.as_posix(),
        overall_scores_path=artifact_paths.get("overall_scores_csv", ""),
        attack_scores_path=artifact_paths.get("attack_scores_csv", ""),
        artifact_paths=artifact_paths,
        row_counts=row_counts,
        notes=notes,
    )


def _slugify_report_component(value: object) -> str:
    """Convert a report component into a filesystem-safe slug."""

    token = str(value).strip()
    token = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in token)
    token = token.strip("-._")
    return token or _DEFAULT_OUTPUT_NAME


def summarize_binary_detection_report(report: BinaryDetectionReport) -> str:
    """Render a compact human-readable binary detection report."""

    return report.render()


__all__ = [
    "BinaryAttackMetricRecord",
    "BinaryDetectionExportResult",
    "BinaryDetectionModel",
    "BinaryDetectionReport",
    "BinaryDetectionScoreRecord",
    "BINARY_ATTACK_METRIC_FIELDS",
    "BINARY_DETECTION_REPORT_FIELDS",
    "BINARY_DETECTION_SCORE_FIELDS",
    "BinaryDetectionSplit",
    "export_binary_detection_report",
    "run_binary_detection_experiment",
    "summarize_binary_detection_report",
]
