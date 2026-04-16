"""Binary classification metrics for anomaly-score evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np


_POSITIVE_LABELS = {
    "1",
    "true",
    "yes",
    "y",
    "positive",
    "attack",
    "malicious",
    "anomaly",
    "intrusion",
}
_NEGATIVE_LABELS = {
    "0",
    "false",
    "no",
    "n",
    "negative",
    "benign",
    "normal",
}


def coerce_binary_label(value: object | None) -> int | None:
    """Normalize heterogeneous label values into binary integers.

    The function returns `1` for positive labels, `0` for negative labels, and
    `None` when the input cannot be interpreted as a binary label.
    """

    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return 1 if int(value) > 0 else 0
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        return 1 if float(value) > 0.0 else 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return None
        if normalized in _POSITIVE_LABELS:
            return 1
        if normalized in _NEGATIVE_LABELS:
            return 0
        try:
            return 1 if float(normalized) > 0.0 else 0
        except ValueError:
            return None
    return 1 if bool(value) else 0


def _filter_labeled_pairs(
    labels: Sequence[object | None],
    scores: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Filter out unlabeled pairs and coerce the remaining labels to binary ints."""

    filtered_labels: list[int] = []
    filtered_scores: list[float] = []
    if len(labels) != len(scores):
        raise ValueError("Labels and scores must have the same length.")
    for label, score in zip(labels, scores, strict=False):
        binary_label = coerce_binary_label(label)
        if binary_label is None:
            continue
        filtered_labels.append(binary_label)
        filtered_scores.append(float(score))
    return np.asarray(filtered_labels, dtype=int), np.asarray(filtered_scores, dtype=float)


def _binary_roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Compute ROC-AUC for a binary score vector without external dependencies."""

    if y_true.size == 0 or np.unique(y_true).size < 2:
        return None

    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    y_score = y_score[order]
    distinct_threshold_indices = np.where(np.diff(y_score))[0]
    threshold_indices = np.r_[
        distinct_threshold_indices,
        y_true.size - 1,
    ]
    true_positives = np.cumsum(y_true)[threshold_indices]
    false_positives = 1 + threshold_indices - true_positives
    true_positives = np.r_[0, true_positives]
    false_positives = np.r_[0, false_positives]
    positives = true_positives[-1]
    negatives = false_positives[-1]
    if positives == 0 or negatives == 0:
        return None

    tpr = true_positives / positives
    fpr = false_positives / negatives
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:  # pragma: no cover - compatibility with older numpy
        integrate = np.trapz
    return float(integrate(tpr, fpr))


def _binary_average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Compute average precision for binary labels."""

    positive_count = int(np.sum(y_true))
    if y_true.size == 0 or positive_count == 0:
        return None

    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    y_score = y_score[order]
    distinct_threshold_indices = np.where(np.diff(y_score))[0]
    threshold_indices = np.r_[
        distinct_threshold_indices,
        y_true.size - 1,
    ]
    true_positives = np.cumsum(y_true)[threshold_indices]
    false_positives = 1 + threshold_indices - true_positives
    precision = true_positives / np.maximum(true_positives + false_positives, 1)
    recall = true_positives / positive_count
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def _threshold_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> tuple[float | None, float | None, float | None, int, int, int, int]:
    """Compute precision, recall, F1, and confusion counts at a threshold."""

    if y_true.size == 0:
        return None, None, None, 0, 0, 0, 0

    predictions = (y_score >= threshold).astype(int)
    true_positive = int(np.sum((predictions == 1) & (y_true == 1)))
    false_positive = int(np.sum((predictions == 1) & (y_true == 0)))
    true_negative = int(np.sum((predictions == 0) & (y_true == 0)))
    false_negative = int(np.sum((predictions == 0) & (y_true == 1)))

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = (
        true_positive / precision_denominator if precision_denominator > 0 else 0.0
    )
    recall = true_positive / recall_denominator if recall_denominator > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return precision, recall, f1, true_positive, false_positive, true_negative, false_negative


@dataclass(frozen=True, slots=True)
class BinaryScoreMetrics:
    """Binary evaluation metrics for a single anomaly score series."""

    threshold: float
    support: int
    positive_count: int
    negative_count: int
    roc_auc: float | None
    pr_auc: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    def to_dict(self) -> dict[str, float | int | None]:
        """Serialize the metric summary into a JSON-friendly mapping."""

        return {
            "threshold": self.threshold,
            "support": self.support,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "true_negative": self.true_negative,
            "false_negative": self.false_negative,
        }


def evaluate_scores(
    labels: Sequence[object | None],
    scores: Sequence[float],
    *,
    threshold: float = 0.5,
) -> BinaryScoreMetrics:
    """Evaluate a binary anomaly score series against optional labels."""

    y_true, y_score = _filter_labeled_pairs(labels, scores)
    support = int(y_true.size)
    positive_count = int(np.sum(y_true))
    negative_count = support - positive_count

    roc_auc = _binary_roc_auc_score(y_true, y_score)
    pr_auc = _binary_average_precision_score(y_true, y_score)
    precision, recall, f1, true_positive, false_positive, true_negative, false_negative = (
        _threshold_metrics(y_true, y_score, threshold)
    )

    return BinaryScoreMetrics(
        threshold=threshold,
        support=support,
        positive_count=positive_count,
        negative_count=negative_count,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        precision=precision,
        recall=recall,
        f1=f1,
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
    )


def summarize_metric_sets(
    metrics_by_scope: dict[str, BinaryScoreMetrics],
) -> dict[str, dict[str, float | int | None]]:
    """Convert multiple metric summaries into a nested dictionary."""

    return {scope: metrics.to_dict() for scope, metrics in metrics_by_scope.items()}


__all__ = [
    "BinaryScoreMetrics",
    "coerce_binary_label",
    "evaluate_scores",
    "summarize_metric_sets",
]
