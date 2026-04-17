#!/usr/bin/env python3
"""Run the reproducible CSV/PCAP experiment suite requested for binary evaluation."""

from __future__ import annotations

import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from traffic_graph.pipeline.repro_experiment import (
    ReproExperimentConfig,
    run_csv_repro_experiment,
    run_pcap_repro_experiment,
)
from traffic_graph.pipeline.ctu13_public_mixedflow import (
    CTU13ScenarioAsset,
    run_ctu13_public_mixedflow_experiment,
)
from traffic_graph.pipeline.pcap_graph_experiment import PcapGraphExperimentConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOTS = (REPO_ROOT / "data", REPO_ROOT / "artifacts")
OUTPUT_ROOT = REPO_ROOT / "outputs"
SUMMARY_DIR = OUTPUT_ROOT / "summary"
FIGURES_DIR = OUTPUT_ROOT / "figures"
LOGS_DIR = OUTPUT_ROOT / "logs"
METRICS_DIR = OUTPUT_ROOT / "metrics"
DATA_INVENTORY_DIR = OUTPUT_ROOT / "data_inventory"
SEEDS = (42, 43, 44)

CSV_BENIGN_TOKENS = {
    "benign": 0,
    "BENIGN": 0,
    "Benign": 0,
    "normal": 0,
    "Normal": 0,
    "Normal Activity": 0,
}


@dataclass(slots=True)
class ExperimentSeedResult:
    experiment_name: str
    seed: int
    status: str
    run_token: str | None = None
    metrics_path: str | None = None
    export_directory: str | None = None
    figure_path: str | None = None
    log_path: str | None = None
    metrics: dict[str, float | None] = field(default_factory=dict)
    pcap_stats: dict[str, float | None] = field(default_factory=dict)
    source_paths: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "status": self.status,
            "run_token": self.run_token,
            "metrics_path": self.metrics_path,
            "export_directory": self.export_directory,
            "figure_path": self.figure_path,
            "log_path": self.log_path,
            "metrics": dict(self.metrics),
            "pcap_stats": dict(self.pcap_stats),
            "source_paths": list(self.source_paths),
            "error": self.error,
        }


def _setup_logger() -> logging.Logger:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "repro_dataset_suite.log"
    logger = logging.getLogger("repro-dataset-suite")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _normalize_name_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value.strip())
    return token.strip("._-") or "run"


def _discover_files(*patterns: str) -> list[Path]:
    matches: list[Path] = []
    seen: set[str] = set()
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.rglob(pattern):
                resolved = path.resolve().as_posix()
                if resolved in seen or not path.is_file():
                    continue
                seen.add(resolved)
                matches.append(path)
    return sorted(matches)


def _looks_like_label_complete_csv(path: Path) -> bool:
    try:
        preview = pd.read_csv(path, nrows=32)
    except Exception:
        return False
    columns = {column.lower().strip() for column in preview.columns}
    return "label" in columns or "attack" in columns


def _discover_ciciot_csv_candidates() -> list[Path]:
    matches = _discover_files("merge*.csv", "Merged*.csv")
    matches = [path for path in matches if _looks_like_label_complete_csv(path)]
    priority: list[tuple[int, int, str, Path]] = []
    for path in matches:
        lower_name = path.name.lower()
        if "merge01" in lower_name or "merged01" in lower_name:
            rank = 0
        elif "merge02" in lower_name or "merged02" in lower_name:
            rank = 1
        elif "merge03" in lower_name or "merged03" in lower_name:
            rank = 2
        else:
            rank = 10
        priority.append((rank, -int(path.stat().st_size), path.name.lower(), path))
    priority.sort()
    return [item[-1] for item in priority]


def _discover_cicids_day_files(extension: str) -> dict[str, Path]:
    patterns = [f"*Wednesday*.{extension}", f"*Thursday*.{extension}", f"*Friday*.{extension}"]
    matches = _discover_files(*patterns)
    by_day: dict[str, Path] = {}
    for day in ("wednesday", "thursday", "friday"):
        candidates = [path for path in matches if day in path.name.lower()]
        if candidates:
            candidates.sort(key=lambda item: (-int(item.stat().st_size), item.name.lower()))
            by_day[day] = candidates[0]
    return by_day


def _default_csv_label_mapping() -> dict[str, int]:
    return dict(CSV_BENIGN_TOKENS)


def _write_markdown_table(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_data_inventory() -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []

    for path in _discover_ciciot_csv_candidates():
        labels = []
        try:
            preview = pd.read_csv(path, usecols=["Label"], nrows=2048)
            labels = sorted({str(value) for value in preview["Label"].dropna().unique()})
        except Exception:
            labels = []
        inventory.append(
            {
                "dataset_name": "CICIoT2023",
                "raw_source_url": "https://www.unb.ca/cic/datasets/iotdataset-2023.html",
                "local_path": path.as_posix(),
                "file_type": "csv",
                "file_size": int(path.stat().st_size),
                "available_labels": labels,
                "whether_public_complete_mixed_pcap_available": "unknown",
            }
        )

    for path in _discover_files("*.pcap", "*.pcapng"):
        lower = path.as_posix().lower()
        if "ciciot2023" in lower or "cic_iot_2023" in lower:
            dataset_name = "CICIoT2023"
            source_url = "https://www.unb.ca/cic/datasets/iotdataset-2023.html"
            public_mixed = "n/a"
        elif "ctu13" in lower:
            dataset_name = "CTU13"
            source_url = "https://www.stratosphereips.org/datasets-ctu13"
            public_mixed = False
        else:
            dataset_name = "unknown"
            source_url = ""
            public_mixed = "unknown"
        inventory.append(
            {
                "dataset_name": dataset_name,
                "raw_source_url": source_url,
                "local_path": path.as_posix(),
                "file_type": "pcap",
                "file_size": int(path.stat().st_size),
                "available_labels": [],
                "whether_public_complete_mixed_pcap_available": public_mixed,
            }
        )

    for path in _discover_files("*.binetflow.2format", "*.binetflow", "*.biargus"):
        labels = []
        try:
            preview = pd.read_csv(path, usecols=["Label"], nrows=2048)
            labels = sorted({str(value) for value in preview["Label"].dropna().unique()})
        except Exception:
            labels = []
        inventory.append(
            {
                "dataset_name": "CTU13",
                "raw_source_url": "https://www.stratosphereips.org/datasets-ctu13",
                "local_path": path.as_posix(),
                "file_type": "mixed_flow_label_file",
                "file_size": int(path.stat().st_size),
                "available_labels": labels,
                "whether_public_complete_mixed_pcap_available": False,
            }
        )
    inventory.sort(key=lambda row: (str(row["dataset_name"]), str(row["local_path"])))
    return inventory


def _write_data_inventory_files(inventory_rows: list[dict[str, object]]) -> None:
    DATA_INVENTORY_DIR.mkdir(parents=True, exist_ok=True)
    inventory_json = DATA_INVENTORY_DIR / "data_inventory.json"
    inventory_md = DATA_INVENTORY_DIR / "data_inventory.md"
    inventory_json.write_text(
        json.dumps(inventory_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    headers = [
        "dataset_name",
        "file_type",
        "local_path",
        "file_size",
        "available_labels",
        "raw_source_url",
        "whether_public_complete_mixed_pcap_available",
    ]
    rows = [
        [
            row.get("dataset_name", ""),
            row.get("file_type", ""),
            row.get("local_path", ""),
            row.get("file_size", ""),
            ", ".join(row.get("available_labels", [])) if isinstance(row.get("available_labels"), list) else row.get("available_labels", ""),
            row.get("raw_source_url", ""),
            row.get("whether_public_complete_mixed_pcap_available", ""),
        ]
        for row in inventory_rows
    ]
    _write_markdown_table(inventory_md, headers, rows)


def _combine_csv_files(paths: list[Path], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_csv(path, low_memory=False) for path in paths]
    merged = pd.concat(frames, axis=0, ignore_index=True)
    merged.to_csv(output_path, index=False)
    return output_path


def _safe_float(value: object | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_score_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for required in ("binary_label", "anomaly_score"):
        if required not in frame.columns:
            raise ValueError(f"Missing required score column: {required}")
    if "is_alert" not in frame.columns:
        threshold = float(frame["threshold"].iloc[0]) if "threshold" in frame.columns and not frame.empty else 0.0
        frame["is_alert"] = (pd.to_numeric(frame["anomaly_score"], errors="coerce").fillna(0.0) >= threshold).astype(int)
    return frame


def _compute_binary_metrics_from_scores(frame: pd.DataFrame) -> dict[str, float | None]:
    y_true = pd.to_numeric(frame["binary_label"], errors="coerce").fillna(0).astype(int).to_numpy()
    scores = pd.to_numeric(frame["anomaly_score"], errors="coerce").fillna(0.0).astype(float).to_numpy()
    y_pred = pd.to_numeric(frame["is_alert"], errors="coerce").fillna(0).astype(int).to_numpy()
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else None,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)) if len(y_true) else None,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)) if len(y_true) else None,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) else None,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if len(y_true) else None,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else None,
    }
    if len(np.unique(y_true)) >= 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        metrics["pr_auc"] = float(average_precision_score(y_true, scores))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["true_negative"] = float(tn)
    metrics["false_positive"] = float(fp)
    metrics["false_negative"] = float(fn)
    metrics["true_positive"] = float(tp)
    return metrics


def _plot_confusion_matrix(frame: pd.DataFrame, output_path: Path, *, title: str) -> None:
    y_true = pd.to_numeric(frame["binary_label"], errors="coerce").fillna(0).astype(int).to_numpy()
    y_pred = pd.to_numeric(frame["is_alert"], errors="coerce").fillna(0).astype(int).to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(4.8, 4.2))
    image = axis.imshow(cm, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks([0, 1], labels=["benign", "malicious"])
    axis.set_yticks([0, 1], labels=["benign", "malicious"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title)
    for row_index in range(cm.shape[0]):
        for col_index in range(cm.shape[1]):
            axis.text(col_index, row_index, str(int(cm[row_index, col_index])), ha="center", va="center", color="black")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_roc_curve(frame: pd.DataFrame, output_path: Path, *, title: str) -> None:
    y_true = pd.to_numeric(frame["binary_label"], errors="coerce").fillna(0).astype(int).to_numpy()
    scores = pd.to_numeric(frame["anomaly_score"], errors="coerce").fillna(0.0).astype(float).to_numpy()
    figure, axis = plt.subplots(figsize=(5.0, 4.0))
    if len(np.unique(y_true)) >= 2:
        fpr, tpr, _thresholds = roc_curve(y_true, scores)
        roc_auc = roc_auc_score(y_true, scores)
        axis.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}", color="#3478f6")
        axis.plot([0, 1], [0, 1], linestyle="--", color="#888888")
        axis.legend(loc="lower right")
    else:
        axis.text(0.5, 0.5, "ROC unavailable\n(single class)", ha="center", va="center")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_pr_curve(frame: pd.DataFrame, output_path: Path, *, title: str) -> None:
    y_true = pd.to_numeric(frame["binary_label"], errors="coerce").fillna(0).astype(int).to_numpy()
    scores = pd.to_numeric(frame["anomaly_score"], errors="coerce").fillna(0.0).astype(float).to_numpy()
    figure, axis = plt.subplots(figsize=(5.0, 4.0))
    if len(np.unique(y_true)) >= 2:
        precision, recall, _thresholds = precision_recall_curve(y_true, scores)
        pr_auc = average_precision_score(y_true, scores)
        axis.plot(recall, precision, label=f"AP={pr_auc:.4f}", color="#2eb872")
        axis.legend(loc="lower left")
    else:
        axis.text(0.5, 0.5, "PR unavailable\n(single class)", ha="center", va="center")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _feature_missing_ratio_from_csv(path: Path) -> float | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty:
        return 0.0
    return float(frame.isna().sum().sum() / max(frame.shape[0] * frame.shape[1], 1))


def _aggregate_metric(seed_results: list[ExperimentSeedResult], metric_name: str) -> tuple[float | None, float | None]:
    values = [
        float(value)
        for result in seed_results
        for value in [result.metrics.get(metric_name)]
        if value is not None
    ]
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return float(np.mean(values)), float(np.std(values, ddof=0))


def _aggregate_pcap_metric(seed_results: list[ExperimentSeedResult], metric_name: str) -> tuple[float | None, float | None]:
    values = [
        float(value)
        for result in seed_results
        for value in [result.pcap_stats.get(metric_name)]
        if value is not None
    ]
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return float(np.mean(values)), float(np.std(values, ddof=0))


def _summarize_successful_run(
    *,
    experiment_name: str,
    dataset_name: str,
    input_mode: str,
    seed_results: list[ExperimentSeedResult],
    pooled_frame: pd.DataFrame | None,
    source_paths: list[str],
    notes: list[str],
) -> dict[str, object]:
    summary: dict[str, object] = {
        "experiment_name": experiment_name,
        "dataset_name": dataset_name,
        "input_mode": input_mode,
        "status": "success",
        "use_nuisance_aware": False,
        "binary_label_mapping": "benign=0 malicious=1",
        "split_strategy": "row-level stratified random split (current stable code path; file/time-block split unavailable)",
        "source_paths": list(source_paths),
        "seed_count_completed": len(seed_results),
        "seed_count_requested": len(SEEDS),
        "seed_runs": [result.to_dict() for result in seed_results],
        "notes": list(notes),
    }
    for metric_name in ("accuracy", "precision", "recall", "f1", "macro_f1", "balanced_accuracy", "roc_auc", "pr_auc"):
        mean_value, std_value = _aggregate_metric(seed_results, metric_name)
        summary[f"{metric_name}_mean"] = mean_value
        summary[f"{metric_name}_std"] = std_value
    if input_mode in {"pcap", "mixed_flow"}:
        for metric_name in (
            "packet_parse_success_rate",
            "flow_construction_success_rate",
            "graph_construction_success_rate",
            "feature_extraction_missing_ratio",
            "avg_preprocessing_time_per_sample",
            "avg_training_time",
            "avg_inference_time",
        ):
            mean_value, std_value = _aggregate_pcap_metric(seed_results, metric_name)
            summary[f"{metric_name}_mean"] = mean_value
            summary[f"{metric_name}_std"] = std_value
    if pooled_frame is not None and not pooled_frame.empty:
        base_name = _normalize_name_token(experiment_name)
        confusion_path = FIGURES_DIR / f"{base_name}_confusion.png"
        roc_path = FIGURES_DIR / f"{base_name}_roc.png"
        pr_path = FIGURES_DIR / f"{base_name}_pr.png"
        _plot_confusion_matrix(pooled_frame, confusion_path, title=f"{experiment_name} pooled confusion")
        _plot_roc_curve(pooled_frame, roc_path, title=f"{experiment_name} pooled ROC")
        _plot_pr_curve(pooled_frame, pr_path, title=f"{experiment_name} pooled PR")
        summary["confusion_matrix_png"] = confusion_path.as_posix()
        summary["roc_curve_png"] = roc_path.as_posix()
        summary["pr_curve_png"] = pr_path.as_posix()
    return summary


def _build_failure_record(
    experiment_name: str,
    *,
    dataset_name: str,
    input_mode: str,
    source_paths: list[str],
    reason: str,
    seed: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "experiment_name": experiment_name,
        "dataset_name": dataset_name,
        "input_mode": input_mode,
        "status": "failed",
        "use_nuisance_aware": False,
        "source_paths": list(source_paths),
        "reason": reason,
    }
    if seed is not None:
        payload["seed"] = seed
    return payload


def _run_csv_seed(
    *,
    experiment_name: str,
    dataset_name: str,
    csv_path: Path,
    seed: int,
) -> ExperimentSeedResult:
    result = ExperimentSeedResult(
        experiment_name=experiment_name,
        seed=seed,
        status="failed",
        source_paths=[csv_path.as_posix()],
    )
    wrapper_result = run_csv_repro_experiment(
        ReproExperimentConfig(
            dataset_name=dataset_name,
            input_mode="csv",
            use_nuisance_aware=False,
            binary_label_mapping=_default_csv_label_mapping(),
            random_seed=seed,
            output_dir=OUTPUT_ROOT.as_posix(),
            run_name=f"{experiment_name}_seed{seed}",
            input_path=csv_path.as_posix(),
            label_column="Label",
        )
    )
    metrics_payload = json.loads(Path(wrapper_result["metrics_path"]).read_text(encoding="utf-8"))
    overall_scores_path = Path(metrics_payload["export_result"]["artifact_paths"]["overall_scores_csv"])
    score_frame = _load_score_table(overall_scores_path)
    result.status = "success"
    result.run_token = str(wrapper_result["run_token"])
    result.metrics_path = str(wrapper_result["metrics_path"])
    result.export_directory = str(wrapper_result["export_directory"])
    result.figure_path = str(wrapper_result["figure_path"])
    result.log_path = str(wrapper_result["log_path"])
    result.metrics = _compute_binary_metrics_from_scores(score_frame)
    return result


def _run_pcap_seed(
    *,
    experiment_name: str,
    dataset_name: str,
    benign_inputs: list[Path],
    malicious_inputs: list[Path],
    seed: int,
) -> ExperimentSeedResult:
    result = ExperimentSeedResult(
        experiment_name=experiment_name,
        seed=seed,
        status="failed",
        source_paths=[path.as_posix() for path in benign_inputs + malicious_inputs],
    )
    start_time = time.perf_counter()
    wrapper_result = run_pcap_repro_experiment(
        ReproExperimentConfig(
            dataset_name=dataset_name,
            input_mode="pcap",
            use_nuisance_aware=False,
            random_seed=seed,
            output_dir=OUTPUT_ROOT.as_posix(),
            run_name=f"{experiment_name}_seed{seed}",
            benign_inputs=[path.as_posix() for path in benign_inputs],
            malicious_inputs=[path.as_posix() for path in malicious_inputs],
        )
    )
    elapsed_seconds = time.perf_counter() - start_time
    metrics_payload = json.loads(Path(wrapper_result["metrics_path"]).read_text(encoding="utf-8"))
    summary = metrics_payload.get("summary", {})
    if not isinstance(summary, dict):
        raise ValueError("PCAP metrics payload does not contain a summary mapping.")
    overall_scores_path = Path(metrics_payload["artifact_paths"]["overall_scores_csv"])
    score_frame = _load_score_table(overall_scores_path)
    result.status = "success"
    result.run_token = str(wrapper_result["run_token"])
    result.metrics_path = str(wrapper_result["metrics_path"])
    result.export_directory = str(wrapper_result["export_directory"])
    result.figure_path = str(wrapper_result["figure_path"])
    result.log_path = str(wrapper_result["log_path"])
    result.metrics = _compute_binary_metrics_from_scores(score_frame)
    total_packets = float(summary.get("total_packets", 0) or 0.0)
    parsed_packets = float(summary.get("parsed_packets", 0) or 0.0)
    total_flows = float(summary.get("total_flows", 0) or 0.0)
    total_graphs = float(summary.get("total_graphs", 0) or 0.0)
    evaluation_graph_count = float(summary.get("evaluation_graph_count", 0) or 0.0)
    train_graph_count = float(summary.get("train_graph_count", 0) or 0.0)
    feature_missing_ratio = _feature_missing_ratio_from_csv(
        Path(metrics_payload["artifact_paths"].get("train_graph_scores_csv", ""))
    )
    result.pcap_stats = {
        "packet_parse_success_rate": parsed_packets / total_packets if total_packets else None,
        "flow_construction_success_rate": total_flows / parsed_packets if parsed_packets else None,
        "graph_construction_success_rate": total_graphs / total_flows if total_flows else None,
        "feature_extraction_missing_ratio": feature_missing_ratio,
        "avg_preprocessing_time_per_sample": elapsed_seconds / total_graphs if total_graphs else None,
        "avg_training_time": None,
        "avg_inference_time": elapsed_seconds / evaluation_graph_count if evaluation_graph_count else None,
        "total_runtime_seconds": elapsed_seconds,
        "train_graph_count": train_graph_count,
    }
    return result


def _run_ctu13_mixedflow_seed(
    *,
    experiment_name: str,
    mode: str,
    scenario_assets: list[CTU13ScenarioAsset],
    seed: int,
) -> ExperimentSeedResult:
    result = ExperimentSeedResult(
        experiment_name=experiment_name,
        seed=seed,
        status="failed",
        source_paths=[
            asset.label_file_path
            for asset in scenario_assets
        ] + [asset.pcap_path for asset in scenario_assets if asset.pcap_path],
    )
    start_time = time.perf_counter()
    mixedflow_result = run_ctu13_public_mixedflow_experiment(
        export_dir=OUTPUT_ROOT / "runs" / "ctu13_public_mixedflow",
        scenario_assets=scenario_assets,
        run_name=f"{experiment_name}_seed{seed}",
        mode="clean" if mode == "clean" else "full",
        config=PcapGraphExperimentConfig(
            window_size=300,
            benign_train_ratio=0.7,
            train_validation_ratio=0.25,
            graph_score_reduction="hybrid_max_rank_flow_node_max",
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            threshold_percentile=95.0,
            random_seed=seed,
        ),
    )
    elapsed_seconds = time.perf_counter() - start_time
    summary = mixedflow_result.summary
    overall_scores_path = Path(mixedflow_result.artifact_paths["overall_scores_csv"])
    score_frame = _load_score_table(overall_scores_path)
    metrics_path = METRICS_DIR / f"{_normalize_name_token(experiment_name)}_seed{seed}_mixedflow.json"
    metrics_payload = {
        "mode": "mixed_flow",
        "dataset_name": "CTU13",
        "config": {
            "use_nuisance_aware": False,
            "random_seed": seed,
            "mode": mode,
            "scenario_ids": [asset.scenario_id for asset in scenario_assets],
        },
        "summary": summary,
        "artifact_paths": mixedflow_result.artifact_paths,
        "notes": mixedflow_result.notes,
    }
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result.status = "success"
    result.run_token = mixedflow_result.run_id
    result.metrics_path = metrics_path.as_posix()
    result.export_directory = mixedflow_result.run_directory
    result.log_path = None
    result.metrics = _compute_binary_metrics_from_scores(score_frame)
    result.pcap_stats = {
        "packet_parse_success_rate": None,
        "flow_construction_success_rate": _safe_float(summary.get("flow_construction_success_rate")),
        "graph_construction_success_rate": _safe_float(summary.get("graph_construction_success_rate")),
        "feature_extraction_missing_ratio": _safe_float(summary.get("feature_extraction_missing_ratio")),
        "avg_preprocessing_time_per_sample": _safe_float(summary.get("avg_preprocessing_time_per_sample")),
        "avg_training_time": _safe_float(summary.get("avg_training_time")),
        "avg_inference_time": _safe_float(summary.get("avg_inference_time")),
        "total_runtime_seconds": elapsed_seconds,
        "train_graph_count": _safe_float(summary.get("train_graph_count")),
    }
    return result


def _run_csv_experiment_group(
    *,
    experiment_name: str,
    dataset_name: str,
    csv_paths: list[Path],
    logger: logging.Logger,
    failed_runs: list[dict[str, object]],
) -> dict[str, object] | None:
    if not csv_paths:
        failed_runs.append(
            _build_failure_record(
                experiment_name,
                dataset_name=dataset_name,
                input_mode="csv",
                source_paths=[],
                reason="No matching CSV files were discovered.",
            )
        )
        return None
    if len(csv_paths) == 1:
        run_path = csv_paths[0]
        notes = []
    else:
        run_path = _combine_csv_files(
            csv_paths,
            SUMMARY_DIR / "derived_inputs" / f"{_normalize_name_token(experiment_name)}.csv",
        )
        notes = [f"Combined {len(csv_paths)} CSV files into {run_path.as_posix()} before running."]
    seed_results: list[ExperimentSeedResult] = []
    pooled_frames: list[pd.DataFrame] = []
    for seed in SEEDS:
        logger.info("Running CSV experiment %s seed=%s path=%s", experiment_name, seed, run_path)
        try:
            seed_result = _run_csv_seed(
                experiment_name=experiment_name,
                dataset_name=dataset_name,
                csv_path=run_path,
                seed=seed,
            )
            seed_results.append(seed_result)
            metrics_payload = json.loads(Path(seed_result.metrics_path or "").read_text(encoding="utf-8"))
            score_path = Path(metrics_payload["export_result"]["artifact_paths"]["overall_scores_csv"])
            pooled_frame = _load_score_table(score_path)
            pooled_frame["seed"] = seed
            pooled_frames.append(pooled_frame)
        except Exception as exc:
            logger.exception("CSV experiment %s seed=%s failed", experiment_name, seed)
            failed_runs.append(
                _build_failure_record(
                    experiment_name,
                    dataset_name=dataset_name,
                    input_mode="csv",
                    source_paths=[path.as_posix() for path in csv_paths],
                    reason=f"{type(exc).__name__}: {exc}",
                    seed=seed,
                )
            )
    if not seed_results:
        return None
    pooled = pd.concat(pooled_frames, ignore_index=True) if pooled_frames else None
    return _summarize_successful_run(
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        input_mode="csv",
        seed_results=seed_results,
        pooled_frame=pooled,
        source_paths=[path.as_posix() for path in csv_paths],
        notes=notes,
    )


def _run_pcap_experiment_group(
    *,
    experiment_name: str,
    dataset_name: str,
    benign_inputs: list[Path],
    malicious_inputs: list[Path],
    logger: logging.Logger,
    failed_runs: list[dict[str, object]],
    notes: list[str] | None = None,
) -> dict[str, object] | None:
    if not benign_inputs or not malicious_inputs:
        failed_runs.append(
            _build_failure_record(
                experiment_name,
                dataset_name=dataset_name,
                input_mode="pcap",
                source_paths=[path.as_posix() for path in benign_inputs + malicious_inputs],
                reason="Binary PCAP evaluation requires both benign_inputs and malicious_inputs.",
            )
        )
        return None
    seed_results: list[ExperimentSeedResult] = []
    pooled_frames: list[pd.DataFrame] = []
    for seed in SEEDS:
        logger.info(
            "Running PCAP experiment %s seed=%s benign=%s malicious=%s",
            experiment_name,
            seed,
            len(benign_inputs),
            len(malicious_inputs),
        )
        try:
            seed_result = _run_pcap_seed(
                experiment_name=experiment_name,
                dataset_name=dataset_name,
                benign_inputs=benign_inputs,
                malicious_inputs=malicious_inputs,
                seed=seed,
            )
            seed_results.append(seed_result)
            metrics_payload = json.loads(Path(seed_result.metrics_path or "").read_text(encoding="utf-8"))
            score_path = Path(metrics_payload["artifact_paths"]["overall_scores_csv"])
            pooled_frame = _load_score_table(score_path)
            pooled_frame["seed"] = seed
            pooled_frames.append(pooled_frame)
        except Exception as exc:
            logger.exception("PCAP experiment %s seed=%s failed", experiment_name, seed)
            failed_runs.append(
                _build_failure_record(
                    experiment_name,
                    dataset_name=dataset_name,
                    input_mode="pcap",
                    source_paths=[path.as_posix() for path in benign_inputs + malicious_inputs],
                    reason=f"{type(exc).__name__}: {exc}",
                    seed=seed,
                )
            )
    if not seed_results:
        return None
    pooled = pd.concat(pooled_frames, ignore_index=True) if pooled_frames else None
    return _summarize_successful_run(
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        input_mode="pcap",
        seed_results=seed_results,
        pooled_frame=pooled,
        source_paths=[path.as_posix() for path in benign_inputs + malicious_inputs],
        notes=list(notes or []),
    )


def _run_ctu13_mixedflow_experiment_group(
    *,
    experiment_name: str,
    mode: str,
    scenario_assets: list[CTU13ScenarioAsset],
    logger: logging.Logger,
    failed_runs: list[dict[str, object]],
) -> dict[str, object] | None:
    if not scenario_assets:
        failed_runs.append(
            _build_failure_record(
                experiment_name,
                dataset_name="CTU13",
                input_mode="mixed_flow",
                source_paths=[],
                reason="No CTU13 public mixed-flow label files were available.",
            )
        )
        return None
    seed_results: list[ExperimentSeedResult] = []
    pooled_frames: list[pd.DataFrame] = []
    for seed in SEEDS:
        logger.info(
            "Running CTU13 mixed-flow experiment %s seed=%s mode=%s scenarios=%s",
            experiment_name,
            seed,
            mode,
            ",".join(asset.scenario_id for asset in scenario_assets),
        )
        try:
            seed_result = _run_ctu13_mixedflow_seed(
                experiment_name=experiment_name,
                mode=mode,
                scenario_assets=scenario_assets,
                seed=seed,
            )
            seed_results.append(seed_result)
            metrics_payload = json.loads(Path(seed_result.metrics_path or "").read_text(encoding="utf-8"))
            score_path = Path(metrics_payload["artifact_paths"]["overall_scores_csv"])
            pooled_frame = _load_score_table(score_path)
            pooled_frame["seed"] = seed
            pooled_frames.append(pooled_frame)
        except Exception as exc:
            logger.exception("CTU13 mixed-flow experiment %s seed=%s failed", experiment_name, seed)
            failed_runs.append(
                _build_failure_record(
                    experiment_name,
                    dataset_name="CTU13",
                    input_mode="mixed_flow",
                    source_paths=[asset.label_file_path for asset in scenario_assets],
                    reason=f"{type(exc).__name__}: {exc}",
                    seed=seed,
                )
            )
    if not seed_results:
        return None
    pooled = pd.concat(pooled_frames, ignore_index=True) if pooled_frames else None
    notes = [
        "CTU13 public mixed-flow evaluation uses official labeled bidirectional NetFlows as the authoritative mixed-label input.",
        "Clean mode drops background before graph construction.",
        "Full mode keeps background and merges it into the negative class as a pressure test.",
        "Training still uses benign-only graphs.",
    ]
    return _summarize_successful_run(
        experiment_name=experiment_name,
        dataset_name="CTU13",
        input_mode="mixed_flow",
        seed_results=seed_results,
        pooled_frame=pooled,
        source_paths=[asset.label_file_path for asset in scenario_assets],
        notes=notes,
    )


def _discover_ctu13_public_mixedflow_assets() -> list[CTU13ScenarioAsset]:
    scenario_specs = [
        ("48", "capture20110816-2.binetflow.2format", "capture20110816-2.truncated.pcap"),
        ("49", "capture20110816-3.binetflow.2format", "capture20110816-3.truncated.pcap"),
        ("52", "capture20110818-2.binetflow.2format", "capture20110818-2.truncated.pcap"),
    ]
    assets: list[CTU13ScenarioAsset] = []
    for scenario_id, label_name, pcap_name in scenario_specs:
        scenario_dir = REPO_ROOT / "data" / "ctu13" / "raw" / f"scenario_{scenario_id}"
        label_path = scenario_dir / label_name
        pcap_path = scenario_dir / pcap_name
        if not label_path.exists():
            continue
        assets.append(
            CTU13ScenarioAsset(
                scenario_id=scenario_id,
                label_file_path=label_path.as_posix(),
                pcap_path=pcap_path.as_posix() if pcap_path.exists() else None,
                label_source_url=f"https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-{scenario_id}/{label_name}",
                pcap_source_url=f"https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-{scenario_id}/{pcap_name}.bz2",
            )
        )
    return assets


def _write_summary_files(success_rows: list[dict[str, object]], failed_runs: list[dict[str, object]]) -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics_json = SUMMARY_DIR / "all_metrics.json"
    all_metrics_csv = SUMMARY_DIR / "all_metrics.csv"
    failed_runs_json = SUMMARY_DIR / "failed_runs.json"
    data_provenance_json = SUMMARY_DIR / "data_provenance.json"
    all_metrics_json.write_text(json.dumps(success_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failed_runs_json.write_text(json.dumps(failed_runs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    provenance_rows: list[dict[str, object]] = []
    for row in success_rows:
        provenance_rows.append(
            {
                "experiment_name": row.get("experiment_name"),
                "dataset_name": row.get("dataset_name"),
                "input_mode": row.get("input_mode"),
                "source_paths": row.get("source_paths", []),
                "label_mapping": row.get("binary_label_mapping"),
                "ctu13_mode": "public_mixedflow"
                if str(row.get("experiment_name", "")).startswith("CTU13_public_mixedflow")
                else None,
                "use_nuisance_aware": False,
            }
        )
    for row in failed_runs:
        provenance_rows.append(
            {
                "experiment_name": row.get("experiment_name"),
                "dataset_name": row.get("dataset_name"),
                "input_mode": row.get("input_mode"),
                "source_paths": row.get("source_paths", []),
                "label_mapping": "benign=0 malicious=1",
                "ctu13_mode": "public_mixedflow"
                if str(row.get("experiment_name", "")).startswith("CTU13_public_mixedflow")
                else None,
                "use_nuisance_aware": False,
                "status": row.get("status"),
                "reason": row.get("reason"),
            }
        )
    data_provenance_json.write_text(
        json.dumps(provenance_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if success_rows:
        pd.DataFrame(success_rows).to_csv(all_metrics_csv, index=False)
    else:
        pd.DataFrame(
            columns=[
                "experiment_name",
                "dataset_name",
                "input_mode",
                "status",
            ]
        ).to_csv(all_metrics_csv, index=False)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_INVENTORY_DIR.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger()
    success_rows: list[dict[str, object]] = []
    failed_runs: list[dict[str, object]] = []
    inventory_rows = _build_data_inventory()
    _write_data_inventory_files(inventory_rows)

    ciciot_candidates = _discover_ciciot_csv_candidates()
    selected_ciciot = ciciot_candidates[:3]
    ciciot_labels = (
        "CICIoT_csv_merge_a",
        "CICIoT_csv_merge_b",
        "CICIoT_csv_merge_c",
    )
    for experiment_name, path in zip(ciciot_labels, selected_ciciot, strict=False):
        row = _run_csv_experiment_group(
            experiment_name=experiment_name,
            dataset_name="CICIoT2023",
            csv_paths=[path],
            logger=logger,
            failed_runs=failed_runs,
        )
        if row is not None:
            success_rows.append(row)
    if len(selected_ciciot) < 3:
        for experiment_name in ciciot_labels[len(selected_ciciot):]:
            failed_runs.append(
                _build_failure_record(
                    experiment_name,
                    dataset_name="CICIoT2023",
                    input_mode="csv",
                    source_paths=[path.as_posix() for path in selected_ciciot],
                    reason=(
                        "Fewer than three merge CSV files were discovered. "
                        f"Available: {[path.name for path in selected_ciciot]}"
                    ),
                )
            )
    if selected_ciciot:
        merge_all_row = _run_csv_experiment_group(
            experiment_name="CICIoT_csv_merge_all",
            dataset_name="CICIoT2023",
            csv_paths=selected_ciciot,
            logger=logger,
            failed_runs=failed_runs,
        )
        if merge_all_row is not None:
            if len(selected_ciciot) < 3:
                merge_all_row.setdefault("notes", []).append(
                    f"Only {len(selected_ciciot)} merge CSV files were available, so merge_all used the available subset."
                )
            success_rows.append(merge_all_row)
    else:
        failed_runs.append(
            _build_failure_record(
                "CICIoT_csv_merge_all",
                dataset_name="CICIoT2023",
                input_mode="csv",
                source_paths=[],
                reason="No merge CSV files were discovered for CICIoT2023.",
            )
        )

    cicids_csv_days = _discover_cicids_day_files("csv")
    day_specs = [
        ("CICIDS2017_csv_wed", "Wednesday", cicids_csv_days.get("wednesday")),
        ("CICIDS2017_csv_thu", "Thursday", cicids_csv_days.get("thursday")),
        ("CICIDS2017_csv_fri", "Friday", cicids_csv_days.get("friday")),
    ]
    merge_day_paths: list[Path] = []
    for experiment_name, day_label, path in day_specs:
        if path is None:
            failed_runs.append(
                _build_failure_record(
                    experiment_name,
                    dataset_name="CICIDS2017",
                    input_mode="csv",
                    source_paths=[],
                    reason=f"No CSV file matching {day_label} was discovered under data/ or artifacts/.",
                )
            )
            continue
        merge_day_paths.append(path)
        row = _run_csv_experiment_group(
            experiment_name=experiment_name,
            dataset_name="CICIDS2017",
            csv_paths=[path],
            logger=logger,
            failed_runs=failed_runs,
        )
        if row is not None:
            success_rows.append(row)
    if len(merge_day_paths) == 3:
        merge_row = _run_csv_experiment_group(
            experiment_name="CICIDS2017_csv_merge",
            dataset_name="CICIDS2017",
            csv_paths=merge_day_paths,
            logger=logger,
            failed_runs=failed_runs,
        )
        if merge_row is not None:
            success_rows.append(merge_row)
    else:
        failed_runs.append(
            _build_failure_record(
                "CICIDS2017_csv_merge",
                dataset_name="CICIDS2017",
                input_mode="csv",
                source_paths=[path.as_posix() for path in merge_day_paths],
                reason="Wednesday/Thursday/Friday CSV inputs were not all available, so merged CSV run was skipped.",
            )
        )

    cicids_pcap_days = _discover_cicids_day_files("pcap")
    pcap_specs = [
        ("CICIDS2017_pcap_wed", "Wednesday", cicids_pcap_days.get("wednesday")),
        ("CICIDS2017_pcap_thu", "Thursday", cicids_pcap_days.get("thursday")),
        ("CICIDS2017_pcap_fri", "Friday", cicids_pcap_days.get("friday")),
    ]
    merge_pcap_paths: list[Path] = []
    for experiment_name, day_label, path in pcap_specs:
        if path is None:
            failed_runs.append(
                _build_failure_record(
                    experiment_name,
                    dataset_name="CICIDS2017",
                    input_mode="pcap",
                    source_paths=[],
                    reason=f"No PCAP file matching {day_label} was discovered under data/ or artifacts/.",
                )
            )
            continue
        merge_pcap_paths.append(path)
        failed_runs.append(
            _build_failure_record(
                experiment_name,
                dataset_name="CICIDS2017",
                input_mode="pcap",
                source_paths=[path.as_posix()],
                reason=(
                    "Discovered only mixed-label day-level PCAP input without separate benign/malicious file assignment. "
                    "The stable runner requires explicit benign_inputs and malicious_inputs."
                ),
            )
        )
    failed_runs.append(
        _build_failure_record(
            "CICIDS2017_pcap_merge",
            dataset_name="CICIDS2017",
            input_mode="pcap",
            source_paths=[path.as_posix() for path in merge_pcap_paths],
            reason=(
                "Wednesday/Thursday/Friday PCAP runs require separate benign/malicious inputs, "
                "but the discovered CICIDS2017 PCAP assets were unavailable."
            ),
        )
    )

    ctu13_assets = _discover_ctu13_public_mixedflow_assets()
    if ctu13_assets:
        clean_row = _run_ctu13_mixedflow_experiment_group(
            experiment_name="CTU13_public_mixedflow_clean_48_49_52_merge",
            mode="clean",
            scenario_assets=ctu13_assets,
            logger=logger,
            failed_runs=failed_runs,
        )
        if clean_row is not None:
            success_rows.append(clean_row)
        full_row = _run_ctu13_mixedflow_experiment_group(
            experiment_name="CTU13_public_mixedflow_full_48_49_52_merge",
            mode="full",
            scenario_assets=ctu13_assets,
            logger=logger,
            failed_runs=failed_runs,
        )
        if full_row is not None:
            full_row.setdefault("notes", []).append(
                "Background was merged into the negative class for this full pressure-test run."
            )
            success_rows.append(full_row)
    else:
        for experiment_name in (
            "CTU13_public_mixedflow_clean_48_49_52_merge",
            "CTU13_public_mixedflow_full_48_49_52_merge",
        ):
            failed_runs.append(
                _build_failure_record(
                    experiment_name,
                    dataset_name="CTU13",
                    input_mode="mixed_flow",
                    source_paths=[],
                    reason="Required CTU13 public labeled bidirectional flow files were not discovered.",
                )
            )

    ciciot_benign = [
        path
        for path in _discover_files("BenignTraffic*.pcap", "benign*.pcap")
        if "ciciot2023" in path.as_posix().lower() or "cic_iot_2023" in path.as_posix().lower()
    ]
    ciciot_malicious = [
        path
        for path in _discover_files("*.pcap")
        if (
            "ciciot2023" in path.as_posix().lower() or "cic_iot_2023" in path.as_posix().lower()
        ) and "benign" not in path.as_posix().lower()
    ]
    if ciciot_benign and ciciot_malicious:
        ciciot_pcap_row = _run_pcap_experiment_group(
            experiment_name="CICIoT2023_pcap_core_merge",
            dataset_name="CICIoT2023",
            benign_inputs=sorted(ciciot_benign)[:3],
            malicious_inputs=sorted(ciciot_malicious)[:3],
            logger=logger,
            failed_runs=failed_runs,
            notes=[
                "Supplementary PCAP run using locally available CICIoT2023 benign/malicious captures.",
                "This run is supplementary and does not replace the requested CICIDS2017 PCAP line.",
            ],
        )
        if ciciot_pcap_row is not None:
            success_rows.append(ciciot_pcap_row)

    _write_summary_files(success_rows, failed_runs)
    print(json.dumps(
        {
            "successful_experiments": [row["experiment_name"] for row in success_rows],
            "failed_experiments": [row["experiment_name"] for row in failed_runs],
            "all_metrics_csv": (SUMMARY_DIR / "all_metrics.csv").as_posix(),
            "all_metrics_json": (SUMMARY_DIR / "all_metrics.json").as_posix(),
            "failed_runs_json": (SUMMARY_DIR / "failed_runs.json").as_posix(),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":  # pragma: no cover
    main()
