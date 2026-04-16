"""Structured comparison reports for tabular versus graph binary detection runs."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping, Sequence

from traffic_graph.pipeline.binary_detection import BinaryAttackMetricRecord
from traffic_graph.pipeline.scorer_roles import normalize_run_scorer_role

COMPARISON_OVERALL_FIELDS: tuple[str, ...] = (
    "metric_name",
    "higher_is_better",
    "tabular_value",
    "graph_value",
    "delta",
    "absolute_delta",
    "relative_delta",
    "winner",
)
"""Stable column order for overall comparison CSV exports."""

COMPARISON_PER_ATTACK_FIELDS: tuple[str, ...] = (
    "task_name",
    "requested_attack_type",
    "attack_labels",
    "highlighted",
    "sample_count_tabular",
    "sample_count_graph",
    "benign_count_tabular",
    "benign_count_graph",
    "attack_count_tabular",
    "attack_count_graph",
    "tabular_roc_auc",
    "graph_roc_auc",
    "delta_roc_auc",
    "winner_roc_auc",
    "tabular_pr_auc",
    "graph_pr_auc",
    "delta_pr_auc",
    "winner_pr_auc",
    "tabular_precision",
    "graph_precision",
    "delta_precision",
    "winner_precision",
    "tabular_recall",
    "graph_recall",
    "delta_recall",
    "winner_recall",
    "tabular_f1",
    "graph_f1",
    "delta_f1",
    "winner_f1",
    "tabular_false_positive_rate",
    "graph_false_positive_rate",
    "delta_false_positive_rate",
    "winner_false_positive_rate",
    "tabular_score_median",
    "graph_score_median",
    "delta_score_median",
    "tabular_score_q95",
    "graph_score_q95",
    "delta_score_q95",
    "notes",
)
"""Stable column order for per-attack comparison CSV exports."""

_DEFAULT_HIGHLIGHTED_TASKS: tuple[str, ...] = ("recon", "web-based", "all_malicious")
_METRIC_ORDER: tuple[str, ...] = (
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
)


def _timestamp_token(value: object | None = None) -> str:
    """Normalize a timestamp-like value into a UTC token."""

    if value is None:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    token = token.strip("-._")
    return token or "timestamp"


def _slugify_token(value: object, default: str) -> str:
    """Convert an arbitrary value into a filesystem-safe token."""

    token = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    token = token.strip("-._")
    return token or default


def _json_default(value: object) -> object:
    """Serialize non-native JSON values into a readable string."""

    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()  # type: ignore[call-arg]
        except TypeError:
            pass
    return str(value)


def _json_dumps(value: object) -> str:
    """Render a stable JSON string."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)


def _coerce_str(value: object | None) -> str:
    """Normalize a scalar into a stable string."""

    if value is None:
        return ""
    return str(value)


def _coerce_int(value: object | None, default: int = 0) -> int:
    """Normalize a scalar into an integer with fallback support."""

    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: object | None) -> float | None:
    """Normalize a scalar into a float when possible."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: object | None) -> bool:
    """Normalize a scalar into a boolean value."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def _metric_direction(metric_name: str) -> bool:
    """Return whether a higher value is better for the given metric."""

    return metric_name != "false_positive_rate"


def _metric_winner(
    tabular_value: float | None,
    graph_value: float | None,
    *,
    higher_is_better: bool,
    tolerance: float = 1e-12,
) -> str:
    """Determine which backend wins for a comparison metric."""

    if tabular_value is None and graph_value is None:
        return "n/a"
    if tabular_value is None:
        return "graph"
    if graph_value is None:
        return "tabular"
    delta = graph_value - tabular_value
    if abs(delta) <= tolerance:
        return "tie"
    if higher_is_better:
        return "graph" if delta > 0 else "tabular"
    return "graph" if delta < 0 else "tabular"


def _relative_delta(
    tabular_value: float | None,
    graph_value: float | None,
) -> float | None:
    """Compute a relative difference when the baseline value is available."""

    if tabular_value is None or graph_value is None:
        return None
    if abs(tabular_value) <= 1e-12:
        return None
    return (graph_value - tabular_value) / abs(tabular_value)


def _task_sort_key(task_name: str) -> tuple[int, str]:
    """Keep highlighted attack names near the top of rendered summaries."""

    normalized = task_name.lower()
    if normalized == "recon":
        return 0, normalized
    if normalized == "web-based":
        return 1, normalized
    if normalized == "all_malicious":
        return 2, normalized
    return 10, normalized


def _attack_labels_text(attack_labels: Sequence[str]) -> str:
    """Serialize attack labels into a compact pipe-separated string."""

    labels = [str(label) for label in attack_labels if str(label).strip()]
    return "|".join(labels)


def _parse_attack_labels(value: object | None) -> tuple[str, ...]:
    """Restore attack labels from the exported CSV or JSON representation."""

    if value is None:
        return ()
    if isinstance(value, str):
        if "|" in value:
            return tuple(token for token in (part.strip() for part in value.split("|")) if token)
        stripped = value.strip()
        return (stripped,) if stripped else ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return (str(value),)


def _parse_notes(value: object | None) -> tuple[str, ...]:
    """Restore note tuples from CSV/JSON payloads."""

    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ()
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return (stripped,)
        if isinstance(parsed, Sequence) and not isinstance(parsed, (str, bytes, bytearray)):
            return tuple(str(item) for item in parsed if str(item).strip())
        return (str(parsed),)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(str(item) for item in value if str(item).strip())
    return (str(value),)


def _parse_str_sequence(value: object | None) -> tuple[str, ...]:
    """Normalize an arbitrary value into a tuple of non-empty strings."""

    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return (str(value).strip(),)


def _resolve_manifest_path(path: str | Path) -> Path:
    """Resolve a run directory or manifest path into a concrete manifest file."""

    candidate = Path(path)
    if candidate.is_file():
        return candidate
    manifest_path = candidate / "manifest.json"
    if manifest_path.exists():
        return manifest_path
    manifests = sorted(candidate.rglob("manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No manifest.json found under {candidate.as_posix()}")
    if len(manifests) > 1:
        raise ValueError(
            "Multiple manifest.json files were found. Pass a specific run directory or manifest path."
        )
    return manifests[0]


def _resolve_artifact_path(raw_path: str, *, manifest_path: Path) -> Path:
    """Resolve an artifact path recorded in a manifest."""

    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate
    relative_to_manifest = manifest_path.parent / candidate
    if relative_to_manifest.exists():
        return relative_to_manifest
    return candidate


def _read_json_mapping(path: Path) -> dict[str, object]:
    """Read a JSON file and ensure that the root payload is a mapping."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a JSON object in {path.as_posix()}.")
    return dict(payload)


def _read_csv_rows(path: Path) -> list[dict[str, object]]:
    """Read a CSV file into a list of dictionaries."""

    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    return rows


def _load_metric_row_from_mapping(payload: Mapping[str, object]) -> BinaryAttackMetricRecord:
    """Convert a JSON-like mapping into a typed binary attack metric record."""

    return BinaryAttackMetricRecord(
        task_name=_coerce_str(payload.get("task_name")),
        requested_attack_type=_coerce_str(payload.get("requested_attack_type")),
        attack_labels=_parse_attack_labels(payload.get("attack_labels")),
        sample_count=_coerce_int(payload.get("sample_count")),
        benign_count=_coerce_int(payload.get("benign_count")),
        attack_count=_coerce_int(payload.get("attack_count")),
        roc_auc=_coerce_float(payload.get("roc_auc")),
        pr_auc=_coerce_float(payload.get("pr_auc")),
        precision=float(_coerce_float(payload.get("precision")) or 0.0),
        recall=float(_coerce_float(payload.get("recall")) or 0.0),
        f1=float(_coerce_float(payload.get("f1")) or 0.0),
        false_positive_rate=_coerce_float(payload.get("false_positive_rate")),
        threshold=float(_coerce_float(payload.get("threshold")) or 0.0),
        score_min=float(_coerce_float(payload.get("score_min")) or 0.0),
        score_q25=float(_coerce_float(payload.get("score_q25")) or 0.0),
        score_median=float(_coerce_float(payload.get("score_median")) or 0.0),
        score_q75=float(_coerce_float(payload.get("score_q75")) or 0.0),
        score_q95=float(_coerce_float(payload.get("score_q95")) or 0.0),
        score_max=float(_coerce_float(payload.get("score_max")) or 0.0),
        score_mean=float(_coerce_float(payload.get("score_mean")) or 0.0),
        score_std=float(_coerce_float(payload.get("score_std")) or 0.0),
        benign_score_mean=float(_coerce_float(payload.get("benign_score_mean")) or 0.0),
        benign_score_median=float(_coerce_float(payload.get("benign_score_median")) or 0.0),
        attack_score_mean=float(_coerce_float(payload.get("attack_score_mean")) or 0.0),
        attack_score_median=float(_coerce_float(payload.get("attack_score_median")) or 0.0),
        notes=_parse_notes(payload.get("notes")),
    )


def _load_metric_rows_from_csv(path: Path) -> tuple[BinaryAttackMetricRecord, ...]:
    """Load per-attack metrics from a CSV file."""

    if not path.exists():
        return ()
    rows = _read_csv_rows(path)
    return tuple(_load_metric_row_from_mapping(row) for row in rows)


def _summary_metric_rows(
    rows: Sequence[BinaryAttackMetricRecord],
) -> tuple[dict[str, object], ...]:
    """Convert metric rows to stable dictionaries."""

    return tuple(row.to_dict() for row in rows)


@dataclass(frozen=True, slots=True)
class ComparisonRunSummary:
    """Normalized binary-detection run metadata and metrics for comparison."""

    backend_name: str
    run_id: str
    dataset_name: str
    source_path: str
    created_at: str
    run_directory: str
    manifest_path: str
    metrics_summary_path: str
    per_attack_metrics_path: str
    overall_scores_path: str | None = None
    attack_scores_path: str | None = None
    threshold_percentile: float = 0.0
    threshold: float = 0.0
    model_n_components: int = 0
    train_sample_count: int = 0
    train_benign_count: int = 0
    scorer_role: str = ""
    feature_columns: tuple[str, ...] = ()
    overall_metrics: dict[str, float | None] = field(default_factory=dict)
    train_score_summary: dict[str, object] = field(default_factory=dict)
    overall_score_summary: dict[str, object] = field(default_factory=dict)
    per_attack_metrics: tuple[BinaryAttackMetricRecord, ...] = ()
    attack_score_summaries: dict[str, dict[str, object]] = field(default_factory=dict)
    input_artifacts: dict[str, object] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize the run summary into a JSON-friendly dictionary."""

        return {
            "backend_name": self.backend_name,
            "run_id": self.run_id,
            "dataset_name": self.dataset_name,
            "source_path": self.source_path,
            "created_at": self.created_at,
            "run_directory": self.run_directory,
            "manifest_path": self.manifest_path,
            "metrics_summary_path": self.metrics_summary_path,
            "per_attack_metrics_path": self.per_attack_metrics_path,
            "overall_scores_path": self.overall_scores_path,
            "attack_scores_path": self.attack_scores_path,
            "threshold_percentile": self.threshold_percentile,
            "threshold": self.threshold,
            "model_n_components": self.model_n_components,
            "train_sample_count": self.train_sample_count,
            "train_benign_count": self.train_benign_count,
            "scorer_role": self.scorer_role,
            "feature_columns": list(self.feature_columns),
            "overall_metrics": dict(self.overall_metrics),
            "train_score_summary": dict(self.train_score_summary),
            "overall_score_summary": dict(self.overall_score_summary),
            "per_attack_metrics": _summary_metric_rows(self.per_attack_metrics),
            "attack_score_summaries": dict(self.attack_score_summaries),
            "input_artifacts": dict(self.input_artifacts),
            "artifact_paths": dict(self.artifact_paths),
            "row_counts": dict(self.row_counts),
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class ComparisonMetricRow:
    """Row in the overall metric comparison table."""

    metric_name: str
    higher_is_better: bool
    tabular_value: float | None
    graph_value: float | None
    delta: float | None
    absolute_delta: float | None
    relative_delta: float | None
    winner: str
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize the comparison row into a JSON-friendly dictionary."""

        return {
            "metric_name": self.metric_name,
            "higher_is_better": self.higher_is_better,
            "tabular_value": self.tabular_value,
            "graph_value": self.graph_value,
            "delta": self.delta,
            "absolute_delta": self.absolute_delta,
            "relative_delta": self.relative_delta,
            "winner": self.winner,
            "notes": list(self.notes),
        }

    def to_csv_row(self) -> dict[str, object]:
        """Serialize the comparison row into the CSV export schema."""

        row = self.to_dict().copy()
        row["notes"] = _json_dumps(list(self.notes))
        return row


@dataclass(frozen=True, slots=True)
class ComparisonAttackRow:
    """Row in the per-attack comparison table."""

    task_name: str
    requested_attack_type: str
    attack_labels: tuple[str, ...]
    highlighted: bool
    sample_count_tabular: int | None
    sample_count_graph: int | None
    benign_count_tabular: int | None
    benign_count_graph: int | None
    attack_count_tabular: int | None
    attack_count_graph: int | None
    tabular_roc_auc: float | None
    graph_roc_auc: float | None
    delta_roc_auc: float | None
    winner_roc_auc: str
    tabular_pr_auc: float | None
    graph_pr_auc: float | None
    delta_pr_auc: float | None
    winner_pr_auc: str
    tabular_precision: float | None
    graph_precision: float | None
    delta_precision: float | None
    winner_precision: str
    tabular_recall: float | None
    graph_recall: float | None
    delta_recall: float | None
    winner_recall: str
    tabular_f1: float | None
    graph_f1: float | None
    delta_f1: float | None
    winner_f1: str
    tabular_false_positive_rate: float | None
    graph_false_positive_rate: float | None
    delta_false_positive_rate: float | None
    winner_false_positive_rate: str
    tabular_score_median: float | None
    graph_score_median: float | None
    delta_score_median: float | None
    tabular_score_q95: float | None
    graph_score_q95: float | None
    delta_score_q95: float | None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize the per-attack comparison row into a JSON-friendly dictionary."""

        return {
            "task_name": self.task_name,
            "requested_attack_type": self.requested_attack_type,
            "attack_labels": list(self.attack_labels),
            "highlighted": self.highlighted,
            "sample_count_tabular": self.sample_count_tabular,
            "sample_count_graph": self.sample_count_graph,
            "benign_count_tabular": self.benign_count_tabular,
            "benign_count_graph": self.benign_count_graph,
            "attack_count_tabular": self.attack_count_tabular,
            "attack_count_graph": self.attack_count_graph,
            "tabular_roc_auc": self.tabular_roc_auc,
            "graph_roc_auc": self.graph_roc_auc,
            "delta_roc_auc": self.delta_roc_auc,
            "winner_roc_auc": self.winner_roc_auc,
            "tabular_pr_auc": self.tabular_pr_auc,
            "graph_pr_auc": self.graph_pr_auc,
            "delta_pr_auc": self.delta_pr_auc,
            "winner_pr_auc": self.winner_pr_auc,
            "tabular_precision": self.tabular_precision,
            "graph_precision": self.graph_precision,
            "delta_precision": self.delta_precision,
            "winner_precision": self.winner_precision,
            "tabular_recall": self.tabular_recall,
            "graph_recall": self.graph_recall,
            "delta_recall": self.delta_recall,
            "winner_recall": self.winner_recall,
            "tabular_f1": self.tabular_f1,
            "graph_f1": self.graph_f1,
            "delta_f1": self.delta_f1,
            "winner_f1": self.winner_f1,
            "tabular_false_positive_rate": self.tabular_false_positive_rate,
            "graph_false_positive_rate": self.graph_false_positive_rate,
            "delta_false_positive_rate": self.delta_false_positive_rate,
            "winner_false_positive_rate": self.winner_false_positive_rate,
            "tabular_score_median": self.tabular_score_median,
            "graph_score_median": self.graph_score_median,
            "delta_score_median": self.delta_score_median,
            "tabular_score_q95": self.tabular_score_q95,
            "graph_score_q95": self.graph_score_q95,
            "delta_score_q95": self.delta_score_q95,
            "notes": list(self.notes),
        }

    def to_csv_row(self) -> dict[str, object]:
        """Serialize the per-attack comparison row into the CSV export schema."""

        row = self.to_dict().copy()
        row["attack_labels"] = _attack_labels_text(self.attack_labels)
        row["notes"] = _json_dumps(list(self.notes))
        return row


@dataclass(frozen=True, slots=True)
class BinaryDetectionComparisonReport:
    """Structured comparison report for tabular and graph binary runs."""

    tabular: ComparisonRunSummary
    graph: ComparisonRunSummary
    overall_metrics: tuple[ComparisonMetricRow, ...]
    per_attack_metrics: tuple[ComparisonAttackRow, ...]
    highlighted_attacks: tuple[str, ...]
    task_alignment: dict[str, tuple[str, ...]]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize the comparison report into a JSON-friendly dictionary."""

        highlighted_payload: dict[str, object] = {}
        tabular_lookup = {metric.task_name: metric for metric in self.tabular.per_attack_metrics}
        graph_lookup = {metric.task_name: metric for metric in self.graph.per_attack_metrics}
        for task_name in self.highlighted_attacks:
            tabular_metric = tabular_lookup.get(task_name)
            graph_metric = graph_lookup.get(task_name)
            if tabular_metric is None and graph_metric is None:
                continue
            highlighted_payload[task_name] = {
                "task_name": task_name,
                "requested_attack_type": (
                    tabular_metric.requested_attack_type
                    if tabular_metric is not None
                    else graph_metric.requested_attack_type
                ),
                "attack_labels": list(
                    dict.fromkeys(
                        list(tabular_metric.attack_labels if tabular_metric else ())
                        + list(graph_metric.attack_labels if graph_metric else ())
                    )
                ),
                "tabular": tabular_metric.to_dict() if tabular_metric is not None else None,
                "graph": graph_metric.to_dict() if graph_metric is not None else None,
                "deltas": {
                    "roc_auc": _value_delta(
                        tabular_metric.roc_auc if tabular_metric else None,
                        graph_metric.roc_auc if graph_metric else None,
                    ),
                    "pr_auc": _value_delta(
                        tabular_metric.pr_auc if tabular_metric else None,
                        graph_metric.pr_auc if graph_metric else None,
                    ),
                    "precision": _value_delta(
                        tabular_metric.precision if tabular_metric else None,
                        graph_metric.precision if graph_metric else None,
                    ),
                    "recall": _value_delta(
                        tabular_metric.recall if tabular_metric else None,
                        graph_metric.recall if graph_metric else None,
                    ),
                    "f1": _value_delta(
                        tabular_metric.f1 if tabular_metric else None,
                        graph_metric.f1 if graph_metric else None,
                    ),
                    "false_positive_rate": _value_delta(
                        tabular_metric.false_positive_rate if tabular_metric else None,
                        graph_metric.false_positive_rate if graph_metric else None,
                    ),
                    "tabular_score_median": _value_delta(
                        tabular_metric.score_median if tabular_metric else None,
                        graph_metric.score_median if graph_metric else None,
                    ),
                    "tabular_score_q95": _value_delta(
                        tabular_metric.score_q95 if tabular_metric else None,
                        graph_metric.score_q95 if graph_metric else None,
                    ),
                },
                "tabular_score_summary": self.tabular.attack_score_summaries.get(task_name, {}),
                "graph_score_summary": self.graph.attack_score_summaries.get(task_name, {}),
            }
        return {
            "tabular_run": self.tabular.to_dict(),
            "graph_run": self.graph.to_dict(),
            "overall_metrics": [row.to_dict() for row in self.overall_metrics],
            "per_attack_metrics": [row.to_dict() for row in self.per_attack_metrics],
            "highlighted_attacks": highlighted_payload,
            "task_alignment": {
                "shared_tasks": list(self.task_alignment.get("shared_tasks", ())),
                "tabular_only_tasks": list(self.task_alignment.get("tabular_only_tasks", ())),
                "graph_only_tasks": list(self.task_alignment.get("graph_only_tasks", ())),
            },
            "notes": list(self.notes),
        }

    def render(self) -> str:
        """Render a compact human-readable comparison summary."""

        def _fmt(value: object | None) -> str:
            if value is None:
                return "n/a"
            if isinstance(value, bool):
                return "yes" if value else "no"
            if isinstance(value, (int, float)):
                return f"{float(value):.6f}"
            return str(value)

        lines = [
            "Binary detection comparison: tabular vs graph",
            f"Tabular run: {self.tabular.run_id} ({self.tabular.dataset_name}, role={self.tabular.scorer_role or 'n/a'})",
            f"Graph run: {self.graph.run_id} ({self.graph.dataset_name}, role={self.graph.scorer_role or 'n/a'})",
            (
                "Shared attack tasks: "
                f"{len(self.task_alignment.get('shared_tasks', ()))}, "
                f"tabular-only: {len(self.task_alignment.get('tabular_only_tasks', ()))}, "
                f"graph-only: {len(self.task_alignment.get('graph_only_tasks', ()))}"
            ),
            "Overall metric comparison:",
        ]
        for row in self.overall_metrics:
            lines.append(
                "  - "
                f"{row.metric_name}: tabular={_fmt(row.tabular_value)}, "
                f"graph={_fmt(row.graph_value)}, delta={_fmt(row.delta)}, "
                f"winner={row.winner}"
            )
        if self.highlighted_attacks:
            lines.append("Highlighted attack tasks:")
            highlighted_lookup = {row.task_name: row for row in self.per_attack_metrics}
            for task_name in self.highlighted_attacks:
                row = highlighted_lookup.get(task_name)
                if row is None:
                    continue
                lines.append(
                    "  - "
                    f"{task_name}: recall={_fmt(row.delta_recall)} (tabular={_fmt(row.tabular_recall)}, "
                    f"graph={_fmt(row.graph_recall)}), "
                    f"pr_auc={_fmt(row.delta_pr_auc)} (tabular={_fmt(row.tabular_pr_auc)}, "
                    f"graph={_fmt(row.graph_pr_auc)}), "
                    f"f1={_fmt(row.delta_f1)} (tabular={_fmt(row.tabular_f1)}, "
                    f"graph={_fmt(row.graph_f1)}), "
                    f"score_median_delta={_fmt(row.delta_score_median)}, "
                    f"score_q95_delta={_fmt(row.delta_score_q95)}, "
                    f"winner_recall={row.winner_recall}, winner_pr_auc={row.winner_pr_auc}, "
                    f"winner_f1={row.winner_f1}"
                )
        if self.notes:
            lines.append("Notes:")
            lines.extend(f"  - {note}" for note in self.notes)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class BinaryDetectionComparisonExportResult:
    """Summary returned after exporting a comparison report bundle."""

    comparison_directory: str
    manifest_path: str
    summary_path: str
    overall_metrics_path: str
    per_attack_metrics_path: str
    markdown_path: str | None = None
    artifact_paths: dict[str, str] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _value_delta(tabular_value: float | None, graph_value: float | None) -> float | None:
    """Compute graph minus tabular when both values are available."""

    if tabular_value is None or graph_value is None:
        return None
    return float(graph_value - tabular_value)


def build_comparison_run_summary(
    *,
    backend_name: str,
    run_directory: str | Path,
    manifest_path: str | Path,
    metrics_summary_path: str | Path,
    report_payload: Mapping[str, object],
    artifact_paths: Mapping[str, str],
    row_counts: Mapping[str, int],
    per_attack_metrics_path: str | Path,
    overall_scores_path: str | Path | None = None,
    attack_scores_path: str | Path | None = None,
    per_attack_metrics: Sequence[BinaryAttackMetricRecord] | None = None,
) -> ComparisonRunSummary:
    """Build a normalized run summary from a binary detection report payload."""

    parsed_per_attack_metrics: tuple[BinaryAttackMetricRecord, ...]
    if per_attack_metrics is not None:
        parsed_per_attack_metrics = tuple(per_attack_metrics)
    else:
        payload_metrics = report_payload.get("per_attack_metrics", ())
        if isinstance(payload_metrics, Sequence) and not isinstance(
            payload_metrics, (str, bytes, bytearray)
        ):
            parsed_per_attack_metrics = tuple(
                _load_metric_row_from_mapping(item)
                for item in payload_metrics
                if isinstance(item, Mapping)
            )
        else:
            parsed_per_attack_metrics = ()

    if not parsed_per_attack_metrics and Path(per_attack_metrics_path).exists():
        parsed_per_attack_metrics = _load_metric_rows_from_csv(Path(per_attack_metrics_path))

    overall_metrics = report_payload.get("overall_metrics", {})
    train_score_summary = report_payload.get("train_score_summary", {})
    overall_score_summary = report_payload.get("overall_score_summary", {})
    attack_score_summaries = report_payload.get("attack_score_summaries", {})
    input_artifacts = report_payload.get("input_artifacts", {})
    inferred_scorer_name = None
    if isinstance(input_artifacts, Mapping):
        inferred_scorer_name = (
            input_artifacts.get("graph_score_reduction")
            or input_artifacts.get("scorer_type")
            or input_artifacts.get("model_mode")
        )

    return ComparisonRunSummary(
        backend_name=backend_name,
        run_id=_coerce_str(report_payload.get("run_id")),
        dataset_name=_coerce_str(report_payload.get("dataset_name")),
        source_path=_coerce_str(report_payload.get("source_path")),
        created_at=_coerce_str(report_payload.get("created_at")),
        run_directory=Path(run_directory).as_posix(),
        manifest_path=Path(manifest_path).as_posix(),
        metrics_summary_path=Path(metrics_summary_path).as_posix(),
        per_attack_metrics_path=Path(per_attack_metrics_path).as_posix(),
        overall_scores_path=Path(overall_scores_path).as_posix()
        if overall_scores_path is not None
        else None,
        attack_scores_path=Path(attack_scores_path).as_posix()
        if attack_scores_path is not None
        else None,
        threshold_percentile=float(_coerce_float(report_payload.get("threshold_percentile")) or 0.0),
        threshold=float(_coerce_float(report_payload.get("threshold")) or 0.0),
        model_n_components=_coerce_int(report_payload.get("model_n_components")),
        train_sample_count=_coerce_int(report_payload.get("train_sample_count")),
        train_benign_count=_coerce_int(report_payload.get("train_benign_count")),
        scorer_role=_coerce_str(
            report_payload.get("scorer_role")
            or (
                input_artifacts.get("scorer_role")
                if isinstance(input_artifacts, Mapping)
                else None
            )
            or normalize_run_scorer_role(
                backend_name=backend_name,
                scorer_name=None if inferred_scorer_name is None else str(inferred_scorer_name),
            )
        ),
        feature_columns=_parse_str_sequence(report_payload.get("feature_columns")),
        overall_metrics={
            str(key): (
                float(value)
                if isinstance(value, (int, float))
                else None
            )
            for key, value in dict(overall_metrics).items()
        },
        train_score_summary=dict(train_score_summary)
        if isinstance(train_score_summary, Mapping)
        else {},
        overall_score_summary=dict(overall_score_summary)
        if isinstance(overall_score_summary, Mapping)
        else {},
        per_attack_metrics=parsed_per_attack_metrics,
        attack_score_summaries=dict(attack_score_summaries)
        if isinstance(attack_score_summaries, Mapping)
        else {},
        input_artifacts=dict(input_artifacts) if isinstance(input_artifacts, Mapping) else {},
        artifact_paths={str(key): str(value) for key, value in dict(artifact_paths).items()},
        row_counts={str(key): int(value) for key, value in dict(row_counts).items()},
        notes=_parse_notes(report_payload.get("notes")),
    )


def _comparison_task_row(
    tabular: BinaryAttackMetricRecord | None,
    graph: BinaryAttackMetricRecord | None,
    *,
    highlighted: bool,
) -> ComparisonAttackRow:
    """Build one joined per-attack comparison row."""

    def _metric_value(metric: BinaryAttackMetricRecord | None, name: str) -> float | None:
        if metric is None:
            return None
        value = getattr(metric, name)
        if value is None:
            return None
        return float(value)

    def _metric_winner_row(name: str) -> str:
        return _metric_winner(
            _metric_value(tabular, name),
            _metric_value(graph, name),
            higher_is_better=_metric_direction(name),
        )

    tabular_labels = tabular.attack_labels if tabular is not None else ()
    graph_labels = graph.attack_labels if graph is not None else ()
    attack_labels = tuple(
        dict.fromkeys([*tabular_labels, *graph_labels])
    )
    task_name = tabular.task_name if tabular is not None else (graph.task_name if graph else "")
    requested_attack_type = (
        tabular.requested_attack_type
        if tabular is not None
        else (graph.requested_attack_type if graph is not None else "")
    )
    notes: list[str] = []
    if tabular is None:
        notes.append("Task missing from tabular baseline.")
    if graph is None:
        notes.append("Task missing from graph mode.")

    return ComparisonAttackRow(
        task_name=task_name,
        requested_attack_type=requested_attack_type,
        attack_labels=attack_labels,
        highlighted=highlighted,
        sample_count_tabular=tabular.sample_count if tabular is not None else None,
        sample_count_graph=graph.sample_count if graph is not None else None,
        benign_count_tabular=tabular.benign_count if tabular is not None else None,
        benign_count_graph=graph.benign_count if graph is not None else None,
        attack_count_tabular=tabular.attack_count if tabular is not None else None,
        attack_count_graph=graph.attack_count if graph is not None else None,
        tabular_roc_auc=_metric_value(tabular, "roc_auc"),
        graph_roc_auc=_metric_value(graph, "roc_auc"),
        delta_roc_auc=_value_delta(
            _metric_value(tabular, "roc_auc"), _metric_value(graph, "roc_auc")
        ),
        winner_roc_auc=_metric_winner_row("roc_auc"),
        tabular_pr_auc=_metric_value(tabular, "pr_auc"),
        graph_pr_auc=_metric_value(graph, "pr_auc"),
        delta_pr_auc=_value_delta(
            _metric_value(tabular, "pr_auc"), _metric_value(graph, "pr_auc")
        ),
        winner_pr_auc=_metric_winner_row("pr_auc"),
        tabular_precision=_metric_value(tabular, "precision"),
        graph_precision=_metric_value(graph, "precision"),
        delta_precision=_value_delta(
            _metric_value(tabular, "precision"), _metric_value(graph, "precision")
        ),
        winner_precision=_metric_winner_row("precision"),
        tabular_recall=_metric_value(tabular, "recall"),
        graph_recall=_metric_value(graph, "recall"),
        delta_recall=_value_delta(
            _metric_value(tabular, "recall"), _metric_value(graph, "recall")
        ),
        winner_recall=_metric_winner_row("recall"),
        tabular_f1=_metric_value(tabular, "f1"),
        graph_f1=_metric_value(graph, "f1"),
        delta_f1=_value_delta(_metric_value(tabular, "f1"), _metric_value(graph, "f1")),
        winner_f1=_metric_winner_row("f1"),
        tabular_false_positive_rate=_metric_value(tabular, "false_positive_rate"),
        graph_false_positive_rate=_metric_value(graph, "false_positive_rate"),
        delta_false_positive_rate=_value_delta(
            _metric_value(tabular, "false_positive_rate"),
            _metric_value(graph, "false_positive_rate"),
        ),
        winner_false_positive_rate=_metric_winner_row("false_positive_rate"),
        tabular_score_median=(
            float(tabular.score_median) if tabular is not None else None
        ),
        graph_score_median=float(graph.score_median) if graph is not None else None,
        delta_score_median=_value_delta(
            float(tabular.score_median) if tabular is not None else None,
            float(graph.score_median) if graph is not None else None,
        ),
        tabular_score_q95=float(tabular.score_q95) if tabular is not None else None,
        graph_score_q95=float(graph.score_q95) if graph is not None else None,
        delta_score_q95=_value_delta(
            float(tabular.score_q95) if tabular is not None else None,
            float(graph.score_q95) if graph is not None else None,
        ),
        notes=tuple(notes),
    )


def compare_binary_detection_run_summaries(
    tabular: ComparisonRunSummary,
    graph: ComparisonRunSummary,
    *,
    highlighted_attacks: Sequence[str] = _DEFAULT_HIGHLIGHTED_TASKS,
) -> BinaryDetectionComparisonReport:
    """Compare two normalized binary detection run summaries."""

    tabular_lookup = {metric.task_name: metric for metric in tabular.per_attack_metrics}
    graph_lookup = {metric.task_name: metric for metric in graph.per_attack_metrics}
    shared_tasks = tuple(
        sorted(set(tabular_lookup) & set(graph_lookup), key=_task_sort_key)
    )
    tabular_only_tasks = tuple(
        sorted(set(tabular_lookup) - set(graph_lookup), key=_task_sort_key)
    )
    graph_only_tasks = tuple(
        sorted(set(graph_lookup) - set(tabular_lookup), key=_task_sort_key)
    )
    all_tasks = tuple(
        sorted(set(tabular_lookup) | set(graph_lookup), key=_task_sort_key)
    )
    overall_rows: list[ComparisonMetricRow] = []
    for metric_name in _METRIC_ORDER:
        tabular_value = tabular.overall_metrics.get(metric_name)
        graph_value = graph.overall_metrics.get(metric_name)
        if tabular_value is not None:
            tabular_value = float(tabular_value)
        if graph_value is not None:
            graph_value = float(graph_value)
        higher_is_better = _metric_direction(metric_name)
        delta = _value_delta(tabular_value, graph_value)
        overall_rows.append(
            ComparisonMetricRow(
                metric_name=metric_name,
                higher_is_better=higher_is_better,
                tabular_value=tabular_value,
                graph_value=graph_value,
                delta=delta,
                absolute_delta=abs(delta) if delta is not None else None,
                relative_delta=_relative_delta(tabular_value, graph_value),
                winner=_metric_winner(
                    tabular_value,
                    graph_value,
                    higher_is_better=higher_is_better,
                ),
            )
        )
    per_attack_rows = tuple(
        _comparison_task_row(
            tabular_lookup.get(task_name),
            graph_lookup.get(task_name),
            highlighted=task_name in {item.lower() for item in highlighted_attacks},
        )
        for task_name in all_tasks
    )
    notes: list[str] = []
    if tabular.dataset_name and graph.dataset_name and tabular.dataset_name != graph.dataset_name:
        notes.append(
            "The compared runs report different dataset names; interpretation should be cautious."
        )
    if tabular.source_path and graph.source_path and tabular.source_path != graph.source_path:
        notes.append(
            "The compared runs originate from different source paths; protocol equivalence should be checked."
        )
    if tabular_only_tasks:
        notes.append(
            "Tabular baseline contains tasks not present in the graph run: "
            + ", ".join(tabular_only_tasks)
        )
    if graph_only_tasks:
        notes.append(
            "Graph run contains tasks not present in the tabular baseline: "
            + ", ".join(graph_only_tasks)
        )
    return BinaryDetectionComparisonReport(
        tabular=tabular,
        graph=graph,
        overall_metrics=tuple(overall_rows),
        per_attack_metrics=per_attack_rows,
        highlighted_attacks=tuple(str(task) for task in highlighted_attacks),
        task_alignment={
            "shared_tasks": shared_tasks,
            "tabular_only_tasks": tabular_only_tasks,
            "graph_only_tasks": graph_only_tasks,
        },
        notes=tuple(notes),
    )


def _write_json(path: Path, payload: object) -> None:
    """Write a stable JSON file to disk."""

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], field_order: Sequence[str]) -> None:
    """Write rows to CSV with a stable column order."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(field_order), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in field_order})


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render a Markdown table from header and row sequences."""

    rendered_headers = [str(header) for header in headers]
    rendered_rows = [["" if cell is None else str(cell) for cell in row] for row in rows]
    separator = ["---" for _ in rendered_headers]
    lines = [
        "| " + " | ".join(rendered_headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rendered_rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_markdown_comparison_report(report: BinaryDetectionComparisonReport) -> str:
    """Render the comparison report as a Markdown document."""

    def _fmt(value: object | None) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, (int, float)):
            return f"{float(value):.6f}"
        return str(value)

    overall_headers = (
        "metric",
        "higher_is_better",
        "tabular",
        "graph",
        "delta",
        "winner",
    )
    overall_rows = [
        (
            row.metric_name,
            "yes" if row.higher_is_better else "no",
            _fmt(row.tabular_value),
            _fmt(row.graph_value),
            _fmt(row.delta),
            row.winner,
        )
        for row in report.overall_metrics
    ]

    per_attack_headers = (
        "task",
        "requested_attack_type",
        "highlighted",
        "tabular_recall",
        "graph_recall",
        "delta_recall",
        "tabular_pr_auc",
        "graph_pr_auc",
        "delta_pr_auc",
        "tabular_f1",
        "graph_f1",
        "delta_f1",
        "winner_recall",
        "winner_pr_auc",
        "winner_f1",
    )
    per_attack_rows = [
        (
            row.task_name,
            row.requested_attack_type,
            "yes" if row.highlighted else "no",
            _fmt(row.tabular_recall),
            _fmt(row.graph_recall),
            _fmt(row.delta_recall),
            _fmt(row.tabular_pr_auc),
            _fmt(row.graph_pr_auc),
            _fmt(row.delta_pr_auc),
            _fmt(row.tabular_f1),
            _fmt(row.graph_f1),
            _fmt(row.delta_f1),
            row.winner_recall,
            row.winner_pr_auc,
            row.winner_f1,
        )
        for row in report.per_attack_metrics
    ]

    lines = [
        "# Binary Detection Comparison",
        "",
        f"- Tabular run: `{report.tabular.run_id}` (`{report.tabular.scorer_role or 'n/a'}`)",
        f"- Graph run: `{report.graph.run_id}` (`{report.graph.scorer_role or 'n/a'}`)",
        f"- Shared tasks: {len(report.task_alignment.get('shared_tasks', ())) }",
        f"- Tabular-only tasks: {len(report.task_alignment.get('tabular_only_tasks', ())) }",
        f"- Graph-only tasks: {len(report.task_alignment.get('graph_only_tasks', ())) }",
        "",
        "## Overall Metrics",
        _markdown_table(overall_headers, overall_rows),
        "",
        "## Per-Attack Metrics",
        _markdown_table(per_attack_headers, per_attack_rows),
    ]
    if report.highlighted_attacks:
        lines.extend(
            [
                "",
                "## Highlighted Attacks",
            ]
        )
        highlighted_lookup = {row.task_name: row for row in report.per_attack_metrics}
        for task_name in report.highlighted_attacks:
            row = highlighted_lookup.get(task_name)
            if row is None:
                continue
            lines.extend(
                [
                    f"### {task_name}",
                    f"- Requested attack type: `{row.requested_attack_type}`",
                    f"- Attack labels: `{_attack_labels_text(row.attack_labels)}`",
                    f"- Recall delta: `{_fmt(row.delta_recall)}`",
                    f"- PR-AUC delta: `{_fmt(row.delta_pr_auc)}`",
                    f"- F1 delta: `{_fmt(row.delta_f1)}`",
                    f"- Score median delta: `{_fmt(row.delta_score_median)}`",
                    f"- Score q95 delta: `{_fmt(row.delta_score_q95)}`",
                    f"- Recall winner: `{row.winner_recall}`",
                    f"- PR-AUC winner: `{row.winner_pr_auc}`",
                    f"- F1 winner: `{row.winner_f1}`",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def summarize_comparison(report: BinaryDetectionComparisonReport) -> str:
    """Render a compact plain-text summary for a comparison report."""

    return report.render()


def export_comparison_report(
    report: BinaryDetectionComparisonReport,
    output_dir: str | Path,
    *,
    export_markdown: bool = False,
    timestamp: object | None = None,
) -> BinaryDetectionComparisonExportResult:
    """Persist a comparison report bundle to local files."""

    layout_directory = (
        Path(output_dir)
        / _slugify_token(f"{report.tabular.run_id}-vs-{report.graph.run_id}", "comparison")
        / _timestamp_token(timestamp)
    )
    layout_directory.mkdir(parents=True, exist_ok=True)

    artifact_paths: dict[str, str] = {}
    row_counts: dict[str, int] = {}

    summary_path = layout_directory / "comparison_summary.json"
    _write_json(summary_path, report.to_dict())
    artifact_paths["comparison_summary_json"] = summary_path.as_posix()
    row_counts["comparison_summary_json"] = 1

    overall_metrics_path = layout_directory / "comparison_overall.csv"
    _write_csv(
        overall_metrics_path,
        [row.to_csv_row() for row in report.overall_metrics],
        COMPARISON_OVERALL_FIELDS,
    )
    artifact_paths["comparison_overall_csv"] = overall_metrics_path.as_posix()
    row_counts["comparison_overall_csv"] = len(report.overall_metrics)

    per_attack_metrics_path = layout_directory / "comparison_per_attack.csv"
    _write_csv(
        per_attack_metrics_path,
        [row.to_csv_row() for row in report.per_attack_metrics],
        COMPARISON_PER_ATTACK_FIELDS,
    )
    artifact_paths["comparison_per_attack_csv"] = per_attack_metrics_path.as_posix()
    row_counts["comparison_per_attack_csv"] = len(report.per_attack_metrics)

    markdown_path: str | None = None
    if export_markdown:
        markdown_file = layout_directory / "comparison_report.md"
        markdown_file.write_text(render_markdown_comparison_report(report), encoding="utf-8")
        markdown_path = markdown_file.as_posix()
        artifact_paths["comparison_report_md"] = markdown_path

    manifest_path = layout_directory / "manifest.json"
    manifest_payload = {
        "comparison_id": f"{report.tabular.run_id}-vs-{report.graph.run_id}",
        "tabular_run_id": report.tabular.run_id,
        "graph_run_id": report.graph.run_id,
        "created_at": _timestamp_token(timestamp),
        "comparison_directory": layout_directory.as_posix(),
        "artifact_paths": artifact_paths,
        "row_counts": row_counts,
        "notes": list(report.notes),
        "highlighted_attacks": list(report.highlighted_attacks),
    }
    _write_json(manifest_path, manifest_payload)
    artifact_paths["manifest_json"] = manifest_path.as_posix()

    return BinaryDetectionComparisonExportResult(
        comparison_directory=layout_directory.as_posix(),
        manifest_path=manifest_path.as_posix(),
        summary_path=summary_path.as_posix(),
        overall_metrics_path=overall_metrics_path.as_posix(),
        per_attack_metrics_path=per_attack_metrics_path.as_posix(),
        markdown_path=markdown_path,
        artifact_paths=artifact_paths,
        row_counts=row_counts,
        notes=list(report.notes),
    )


__all__ = [
    "BinaryDetectionComparisonExportResult",
    "BinaryDetectionComparisonReport",
    "ComparisonAttackRow",
    "ComparisonMetricRow",
    "ComparisonRunSummary",
    "COMPARISON_OVERALL_FIELDS",
    "COMPARISON_PER_ATTACK_FIELDS",
    "build_comparison_run_summary",
    "compare_binary_detection_run_summaries",
    "export_comparison_report",
    "render_markdown_comparison_report",
    "summarize_comparison",
]
