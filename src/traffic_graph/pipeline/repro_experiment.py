"""Repeatable CSV/PCAP experiment wrappers with a unified config surface."""

from __future__ import annotations

import json
import logging
import shutil
import struct
import zlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml

from traffic_graph.data import BinaryExperimentConfig, HeldOutAttackProtocolConfig, prepare_binary_experiment
from traffic_graph.pipeline.binary_detection import (
    BinaryDetectionExportResult,
    BinaryDetectionReport,
    run_binary_detection_experiment,
)
from traffic_graph.pipeline.pcap_graph_experiment import (
    PcapGraphExperimentConfig,
    PcapGraphExperimentResult,
    run_pcap_graph_experiment,
    summarize_pcap_graph_experiment_result,
)

InputMode = Literal["csv", "pcap"]


@dataclass(slots=True)
class ReproExperimentConfig:
    """Unified repeatable experiment config used by CSV and PCAP wrappers."""

    dataset_name: str
    input_mode: InputMode
    use_nuisance_aware: bool = False
    binary_label_mapping: dict[str, int] = field(default_factory=dict)
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    random_seed: int = 42
    output_dir: str = "outputs"
    run_name: str | None = None
    input_path: str | None = None
    benign_inputs: list[str] = field(default_factory=list)
    malicious_inputs: list[str] = field(default_factory=list)
    label_column: str | None = None
    heldout_attack_types: list[str] = field(
        default_factory=lambda: ["Recon", "DDoS", "Mirai", "Web-based"]
    )
    threshold_percentile: float = 95.0
    max_components: int = 10
    packet_limit: int | None = 5000
    window_size: int = 60
    packet_sampling_mode: str = "random_window"
    benign_train_ratio: float = 0.7
    train_validation_ratio: float = 0.25
    epochs: int = 2
    batch_size: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    graph_score_reduction: str = "hybrid_max_rank_flow_node_max"

    @classmethod
    def from_path(cls, path: str | Path) -> "ReproExperimentConfig":
        config_path = Path(path)
        text = config_path.read_text(encoding="utf-8")
        if config_path.suffix.lower() == ".json":
            payload = json.loads(text)
        else:
            payload = yaml.safe_load(text) or {}
        if not isinstance(payload, dict):
            raise ValueError("Experiment config root must be a mapping.")
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, data: dict[str, object]) -> "ReproExperimentConfig":
        input_mode = str(data.get("input_mode", "csv")).strip().lower()
        if input_mode not in {"csv", "pcap"}:
            raise ValueError(f"Unsupported input_mode: {input_mode}")
        mapping = data.get("binary_label_mapping", {})
        binary_label_mapping = {
            str(key): 0 if int(value) == 0 else 1
            for key, value in mapping.items()
        } if isinstance(mapping, dict) else {}
        return cls(
            dataset_name=str(data.get("dataset_name", "traffic_experiment")),
            input_mode=input_mode,  # type: ignore[arg-type]
            use_nuisance_aware=bool(data.get("use_nuisance_aware", False)),
            binary_label_mapping=binary_label_mapping,
            train_ratio=float(data.get("train_ratio", 0.6)),
            val_ratio=float(data.get("val_ratio", 0.2)),
            test_ratio=float(data.get("test_ratio", 0.2)),
            random_seed=int(data.get("random_seed", 42)),
            output_dir=str(data.get("output_dir", "outputs")),
            run_name=None if data.get("run_name") in {None, ""} else str(data.get("run_name")),
            input_path=None if data.get("input_path") in {None, ""} else str(data.get("input_path")),
            benign_inputs=[str(item) for item in data.get("benign_inputs", [])] if isinstance(data.get("benign_inputs"), list) else [],
            malicious_inputs=[str(item) for item in data.get("malicious_inputs", [])] if isinstance(data.get("malicious_inputs"), list) else [],
            label_column=None if data.get("label_column") in {None, ""} else str(data.get("label_column")),
            heldout_attack_types=[str(item) for item in data.get("heldout_attack_types", ["Recon", "DDoS", "Mirai", "Web-based"])],
            threshold_percentile=float(data.get("threshold_percentile", 95.0)),
            max_components=int(data.get("max_components", 10)),
            packet_limit=None if data.get("packet_limit") in {None, ""} else int(data.get("packet_limit")),
            window_size=int(data.get("window_size", 60)),
            packet_sampling_mode=str(data.get("packet_sampling_mode", "random_window")),
            benign_train_ratio=float(data.get("benign_train_ratio", 0.7)),
            train_validation_ratio=float(data.get("train_validation_ratio", 0.25)),
            epochs=int(data.get("epochs", 2)),
            batch_size=int(data.get("batch_size", 2)),
            learning_rate=float(data.get("learning_rate", 1e-3)),
            weight_decay=float(data.get("weight_decay", 0.0)),
            graph_score_reduction=str(data.get("graph_score_reduction", "hybrid_max_rank_flow_node_max")),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def validate(self) -> None:
        if self.input_mode == "csv" and not self.input_path:
            raise ValueError("CSV experiments require input_path.")
        if self.input_mode == "pcap" and not (self.benign_inputs or self.malicious_inputs):
            raise ValueError("PCAP experiments require benign_inputs or malicious_inputs.")


@dataclass(frozen=True, slots=True)
class ReproOutputLayout:
    base_dir: Path
    metrics_dir: Path
    figures_dir: Path
    logs_dir: Path
    checkpoints_dir: Path
    runs_dir: Path


def _timestamp_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _slugify(value: object) -> str:
    token = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in str(value).strip())
    token = token.strip("-._")
    return token or "run"


def _prepare_output_layout(base_dir: str | Path) -> ReproOutputLayout:
    root = Path(base_dir)
    metrics_dir = root / "metrics"
    figures_dir = root / "figures"
    logs_dir = root / "logs"
    checkpoints_dir = root / "checkpoints"
    runs_dir = root / "runs"
    for directory in (metrics_dir, figures_dir, logs_dir, checkpoints_dir, runs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return ReproOutputLayout(
        base_dir=root,
        metrics_dir=metrics_dir,
        figures_dir=figures_dir,
        logs_dir=logs_dir,
        checkpoints_dir=checkpoints_dir,
        runs_dir=runs_dir,
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_text_log(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


def _write_metric_bar_png(path: Path, metrics: dict[str, float], *, title: str) -> None:
    width = 480
    height = 320
    margin = 32
    pixels = bytearray([255] * (width * height * 3))

    def _set_pixel(x: int, y: int, rgb: tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            offset = (y * width + x) * 3
            pixels[offset : offset + 3] = bytes(rgb)

    for x in range(margin, width - margin):
        _set_pixel(x, height - margin, (60, 60, 60))
    for y in range(margin, height - margin):
        _set_pixel(margin, y, (60, 60, 60))
    items = list(metrics.items())[:5]
    if items:
        bar_width = max(18, (width - 2 * margin) // (2 * len(items)))
        for index, (_name, value) in enumerate(items):
            clipped = max(0.0, min(float(value), 1.0))
            bar_height = int((height - 2 * margin) * clipped)
            left = margin + 20 + index * bar_width * 2
            right = left + bar_width
            top = height - margin - bar_height
            color = (52, 120, 246) if index % 2 == 0 else (46, 184, 114)
            for x in range(left, min(right, width - margin)):
                for y in range(max(top, margin), height - margin):
                    _set_pixel(x, y, color)
    for index, _char in enumerate(title[:40]):
        x = margin + index * 2
        y = 12
        for dy in range(4):
            for dx in range(1):
                _set_pixel(x + dx, y + dy, (30, 30, 30))
    raw = bytearray()
    for row_index in range(height):
        raw.append(0)
        start = row_index * width * 3
        raw.extend(pixels[start : start + width * 3])
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += _png_chunk(b"IDAT", zlib.compress(bytes(raw), level=9))
    png += _png_chunk(b"IEND", b"")
    path.write_bytes(png)


def _copy_checkpoint_tree(source_dir: Path, target_dir: Path) -> None:
    if not source_dir.exists():
        return
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"repro-experiment:{log_path}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def run_csv_repro_experiment(config: ReproExperimentConfig) -> dict[str, object]:
    config.validate()
    layout = _prepare_output_layout(config.output_dir)
    run_token = f"{_slugify(config.run_name or config.dataset_name)}-{_timestamp_token()}"
    log_path = layout.logs_dir / f"{run_token}.log"
    logger = _setup_logger(log_path)
    logger.info("Starting CSV experiment for dataset=%s", config.dataset_name)
    logger.info("use_nuisance_aware=%s", config.use_nuisance_aware)
    if config.use_nuisance_aware:
        logger.info(
            "Nuisance-aware modules remain available in the repository but are not attached to the stable CSV runner."
        )
    experiment_config = BinaryExperimentConfig(
        label_column=config.label_column,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        random_seed=config.random_seed,
        train_normal_only=True,
        split_mode="stratified",
        label_mapping=config.binary_label_mapping,
    )
    prepared = prepare_binary_experiment(config.input_path or "", experiment_config)
    logger.info("Rows: original=%s clean=%s dropped=%s", prepared.summary.original_row_count, prepared.summary.clean_row_count, prepared.summary.dropped_row_count)
    logger.info("NaN/Inf cleaning: nan_replaced=%s inf_replaced=%s", prepared.summary.nan_replacement_count, prepared.summary.inf_replacement_count)
    for split_name, split_summary in prepared.summary.split_summaries.items():
        logger.info(
            "Split %s: rows=%s benign=%s malicious=%s",
            split_name,
            split_summary.row_count,
            split_summary.benign_count,
            split_summary.malicious_count,
        )
    report, export_result = run_binary_detection_experiment(
        config.input_path or "",
        layout.runs_dir / "csv",
        binary_experiment_config=experiment_config,
        heldout_protocol_config=HeldOutAttackProtocolConfig(
            label_column=config.label_column,
            held_out_attack_types=tuple(config.heldout_attack_types),
            min_samples_per_attack=1,
            random_seed=config.random_seed,
            benign_train_ratio=config.train_ratio,
        ),
        threshold_percentile=config.threshold_percentile,
        max_components=config.max_components,
        random_seed=config.random_seed,
    )
    logger.info("Training epochs=%s", 0)
    logger.info(
        "Best metrics: roc_auc=%s pr_auc=%s precision=%s recall=%s f1=%s",
        report.overall_metrics.get("roc_auc"),
        report.overall_metrics.get("pr_auc"),
        report.overall_metrics.get("precision"),
        report.overall_metrics.get("recall"),
        report.overall_metrics.get("f1"),
    )
    metrics_payload = {
        "mode": "csv",
        "config": config.to_dict(),
        "prepared_summary": prepared.summary.to_dict(),
        "report": report.to_dict(),
        "export_result": {
            "run_id": export_result.run_id,
            "output_directory": export_result.output_directory,
            "manifest_path": export_result.manifest_path,
            "artifact_paths": export_result.artifact_paths,
            "row_counts": export_result.row_counts,
        },
    }
    metrics_path = layout.metrics_dir / f"{run_token}.json"
    _write_json(metrics_path, metrics_payload)
    figure_path = layout.figures_dir / f"{run_token}.png"
    _write_metric_bar_png(
        figure_path,
        {
            "roc_auc": float(report.overall_metrics.get("roc_auc") or 0.0),
            "pr_auc": float(report.overall_metrics.get("pr_auc") or 0.0),
            "precision": float(report.overall_metrics.get("precision") or 0.0),
            "recall": float(report.overall_metrics.get("recall") or 0.0),
            "f1": float(report.overall_metrics.get("f1") or 0.0),
        },
        title=f"{config.dataset_name} csv metrics",
    )
    return {
        "run_token": run_token,
        "metrics_path": metrics_path.as_posix(),
        "figure_path": figure_path.as_posix(),
        "log_path": log_path.as_posix(),
        "export_directory": export_result.output_directory,
    }


def run_pcap_repro_experiment(config: ReproExperimentConfig) -> dict[str, object]:
    config.validate()
    layout = _prepare_output_layout(config.output_dir)
    run_token = f"{_slugify(config.run_name or config.dataset_name)}-{_timestamp_token()}"
    log_path = layout.logs_dir / f"{run_token}.log"
    logger = _setup_logger(log_path)
    logger.info("Starting PCAP experiment for dataset=%s", config.dataset_name)
    logger.info("use_nuisance_aware=%s", config.use_nuisance_aware)
    if config.use_nuisance_aware:
        logger.info(
            "Nuisance-aware modules remain available in the repository but are not attached to the stable PCAP runner."
        )
    pcap_config = PcapGraphExperimentConfig(
        packet_limit=config.packet_limit,
        packet_sampling_mode=config.packet_sampling_mode,  # type: ignore[arg-type]
        window_size=config.window_size,
        benign_train_ratio=config.benign_train_ratio,
        train_validation_ratio=config.train_validation_ratio,
        graph_score_reduction=config.graph_score_reduction,  # type: ignore[arg-type]
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        threshold_percentile=config.threshold_percentile,
        random_seed=config.random_seed,
    )
    result = run_pcap_graph_experiment(
        export_dir=layout.runs_dir / "pcap",
        benign_inputs=config.benign_inputs,
        malicious_inputs=config.malicious_inputs,
        run_name=config.run_name or config.dataset_name,
        config=pcap_config,
    )
    summary = result.summary
    logger.info(
        "Counts: packets=%s flows=%s graphs=%s benign_graphs=%s malicious_graphs=%s",
        summary.get("total_packets", 0),
        summary.get("total_flows", 0),
        summary.get("total_graphs", 0),
        summary.get("benign_graph_count", 0),
        summary.get("malicious_graph_count", 0),
    )
    logger.info("Training epochs=%s", config.epochs)
    logger.info("%s", summarize_pcap_graph_experiment_result(result))
    metrics_payload = {
        "mode": "pcap",
        "config": config.to_dict(),
        "summary": result.summary,
        "backend": result.backend,
        "notes": result.notes,
        "artifact_paths": result.export_result.artifact_paths,
        "row_counts": result.export_result.row_counts,
    }
    metrics_path = layout.metrics_dir / f"{run_token}.json"
    _write_json(metrics_path, metrics_payload)
    overall_metrics = summary.get("overall_metrics", {})
    if isinstance(overall_metrics, dict):
        figure_values = {
            "roc_auc": float(overall_metrics.get("roc_auc") or 0.0),
            "pr_auc": float(overall_metrics.get("pr_auc") or 0.0),
            "precision": float(overall_metrics.get("precision") or 0.0),
            "recall": float(overall_metrics.get("recall") or 0.0),
            "f1": float(overall_metrics.get("f1") or 0.0),
        }
    else:
        quantiles = summary.get("graph_score_quantiles", {}) if isinstance(summary.get("graph_score_quantiles"), dict) else {}
        figure_values = {
            "median": float(quantiles.get("median") or 0.0),
            "q95": float(quantiles.get("q95") or 0.0),
            "max": float(quantiles.get("max") or 0.0),
        }
    figure_path = layout.figures_dir / f"{run_token}.png"
    _write_metric_bar_png(figure_path, figure_values, title=f"{config.dataset_name} pcap metrics")
    checkpoint_root = Path(result.export_result.run_directory) / "checkpoints"
    _copy_checkpoint_tree(checkpoint_root, layout.checkpoints_dir / run_token)
    return {
        "run_token": run_token,
        "metrics_path": metrics_path.as_posix(),
        "figure_path": figure_path.as_posix(),
        "log_path": log_path.as_posix(),
        "checkpoint_dir": (layout.checkpoints_dir / run_token).as_posix(),
        "export_directory": result.export_result.run_directory,
    }


__all__ = [
    "ReproExperimentConfig",
    "run_csv_repro_experiment",
    "run_pcap_repro_experiment",
]
