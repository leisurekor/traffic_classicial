"""CTU-13 public mixed-flow binary experiment utilities.

This module reuses the repository's existing graph construction, feature
packing, and graph-autoencoder scoring stack, but starts from the official
public CTU-13 labeled bidirectional NetFlow files instead of assuming one
PCAP file has one label. The goal is to support the correct public mixed-label
evaluation path for scenarios such as 48 / 49 / 52.
"""

from __future__ import annotations

import csv
import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np

from traffic_graph.config import (
    AssociationEdgeConfig,
    FeatureNormalizationConfig,
    GraphConfig,
)
from traffic_graph.data import (
    FlowDataset,
    FlowDatasetSummary,
    LogicalFlowWindowStats,
    PcapFlowLoadSummary,
    ShortFlowThresholds,
    preprocess_flow_dataset,
)
from traffic_graph.datasets import CTU13LabeledFlow, parse_ctu13_label_file
from traffic_graph.features import PackedGraphInput, fit_feature_preprocessor, transform_graphs
from traffic_graph.graph import FlowInteractionGraphBuilder
from traffic_graph.pipeline.binary_detection import (
    BINARY_DETECTION_SCORE_FIELDS,
    BinaryDetectionScoreRecord,
)
from traffic_graph.pipeline.metrics import evaluate_scores
from traffic_graph.pipeline.pcap_graph_experiment import (
    PcapGraphEntry,
    PcapGraphExperimentConfig,
    _contextualize_score_rows,
    _graph_summary_row,
    _mark_assignments,
    _score_record_from_row,
    _write_csv,
    _write_json,
    _write_jsonl,
)
from traffic_graph.pipeline.pcap_graph_smoke import (
    _apply_graph_score_reduction_to_rows,
    _graph_score_threshold_from_rows,
    _has_torch,
    _score_graph_rows_with_gae_checkpoint,
    _score_with_fallback,
    _slugify_token,
    _split_graphs,
    _timestamp_token,
    _train_and_score_with_gae,
)
from traffic_graph.pipeline.report_io import build_run_bundle_layout

CTU13MixedFlowMode = Literal["clean", "full"]


@dataclass(frozen=True, slots=True)
class CTU13ScenarioAsset:
    """One public CTU-13 scenario asset pair used for mixed-flow evaluation."""

    scenario_id: str
    label_file_path: str
    pcap_path: str | None = None
    label_source_url: str | None = None
    pcap_source_url: str | None = None


@dataclass(slots=True)
class CTU13PublicMixedFlowResult:
    """Structured result emitted by the public mixed-flow runner."""

    run_id: str
    timestamp: str
    mode: CTU13MixedFlowMode
    summary: dict[str, object]
    artifact_paths: dict[str, str]
    run_directory: str
    checkpoint_dir: str | None = None
    notes: list[str] = field(default_factory=list)


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _count_csv_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for _row in reader:
            count += 1
    return count


def _flow_mapping_from_label_flow(
    labeled_flow: CTU13LabeledFlow,
    *,
    flow_id: str,
) -> dict[str, object]:
    raw_row = labeled_flow.raw_row
    src_pkts = _safe_int(raw_row.get("SrcPkts"), 0)
    dst_pkts = _safe_int(raw_row.get("DstPkts"), 0)
    src_bytes = _safe_int(raw_row.get("SrcBytes"), 0)
    dst_bytes = _safe_int(raw_row.get("DstBytes"), 0)
    tot_pkts = _safe_int(raw_row.get("TotPkts"), src_pkts + dst_pkts)
    tot_bytes = _safe_int(raw_row.get("TotBytes"), src_bytes + dst_bytes)
    end_time = labeled_flow.end_time
    if end_time < labeled_flow.start_time:
        end_time = labeled_flow.start_time
    return {
        "flow_id": flow_id,
        "src_ip": labeled_flow.src_ip,
        "src_port": labeled_flow.src_port,
        "dst_ip": labeled_flow.dst_ip,
        "dst_port": labeled_flow.dst_port,
        "protocol": labeled_flow.protocol,
        "start_time": labeled_flow.start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "packet_count": max(tot_pkts, 0),
        "byte_count": max(tot_bytes, 0),
        "direction": str(raw_row.get("State", "unknown")).lower(),
        "fwd_pkt_count": max(src_pkts, 0),
        "bwd_pkt_count": max(dst_pkts, 0),
        "fwd_bytes": max(src_bytes, 0),
        "bwd_bytes": max(dst_bytes, 0),
        "metadata": {
            "ctu13_binary_label": labeled_flow.binary_label,
            "ctu13_label_text": labeled_flow.label_text,
            "scenario_id": labeled_flow.scenario_id,
        },
    }


def _load_public_ctu13_flow_dataset(
    scenario_assets: list[CTU13ScenarioAsset],
    *,
    mode: CTU13MixedFlowMode,
) -> tuple[FlowDataset, dict[str, str], dict[str, object]]:
    flow_mappings: list[dict[str, object]] = []
    flow_label_by_id: dict[str, str] = {}
    per_scenario_stats: list[dict[str, object]] = []
    total_label_rows = 0
    kept_label_rows = 0

    for asset in scenario_assets:
        label_path = Path(asset.label_file_path)
        raw_row_count = _count_csv_rows(label_path)
        label_flows = parse_ctu13_label_file(label_path, scenario_id=asset.scenario_id)
        total_label_rows += raw_row_count
        scenario_kept = 0
        benign_count = 0
        malicious_count = 0
        unknown_count = 0
        for index, labeled_flow in enumerate(label_flows):
            if mode == "clean" and labeled_flow.binary_label == "unknown":
                continue
            flow_id = f"ctu13:{asset.scenario_id}:{index}"
            flow_mappings.append(_flow_mapping_from_label_flow(labeled_flow, flow_id=flow_id))
            flow_label_by_id[flow_id] = labeled_flow.binary_label
            scenario_kept += 1
            if labeled_flow.binary_label == "benign":
                benign_count += 1
            elif labeled_flow.binary_label == "malicious":
                malicious_count += 1
            else:
                unknown_count += 1
        kept_label_rows += scenario_kept
        per_scenario_stats.append(
            {
                "scenario_id": asset.scenario_id,
                "label_file_path": label_path.as_posix(),
                "raw_label_row_count": raw_row_count,
                "parsed_label_flow_count": len(label_flows),
                "kept_flow_count": scenario_kept,
                "benign_flow_count": benign_count,
                "malicious_flow_count": malicious_count,
                "unknown_flow_count": unknown_count,
            }
        )

    dataset = FlowDataset.from_mappings(flow_mappings)
    dataset_summary = dataset.summary()
    loader_summary = {
        "label_row_count": total_label_rows,
        "kept_flow_count": kept_label_rows,
        "per_scenario": per_scenario_stats,
        "flow_dataset_summary": {
            "flow_count": dataset_summary.flow_count,
            "protocols": list(dataset_summary.protocols),
            "earliest_start": (
                dataset_summary.earliest_start.isoformat()
                if dataset_summary.earliest_start is not None
                else None
            ),
            "latest_end": (
                dataset_summary.latest_end.isoformat()
                if dataset_summary.latest_end is not None
                else None
            ),
            "average_duration_seconds": dataset_summary.average_duration_seconds,
        },
    }
    return dataset, flow_label_by_id, loader_summary


def _mixedflow_load_summary(
    *,
    source_name: str,
    source_path: str,
    label_row_count: int,
    kept_flow_count: int,
    dataset_summary: FlowDatasetSummary,
    notes: list[str],
) -> PcapFlowLoadSummary:
    return PcapFlowLoadSummary(
        source_path=source_path,
        capture_name=source_name,
        packet_limit=None,
        idle_timeout_seconds=60.0,
        byte_order="mixedflow",
        timestamp_resolution="flow",
        version_major=0,
        version_minor=0,
        snaplen=0,
        linktype=0,
        packet_start_offset=0,
        total_packets=0,
        parsed_packets=0,
        skipped_packets=0,
        skipped_reason_counts={},
        total_flows=kept_flow_count,
        flow_dataset_summary=dataset_summary,
        notes=notes,
    )


def _graph_label_for_batch(batch, flow_label_by_id: dict[str, str]) -> str:
    labels: set[str] = set()
    for logical_flow in batch.logical_flows:
        for flow_id in logical_flow.source_flow_ids:
            labels.add(flow_label_by_id.get(flow_id, "unknown"))
    if "malicious" in labels:
        return "malicious"
    if "unknown" in labels:
        return "unknown"
    return "benign"


def _packed_graph_missing_ratio(packed_graphs: list[PackedGraphInput]) -> float | None:
    total_values = 0
    missing_values = 0
    for packed_graph in packed_graphs:
        for matrix in (packed_graph.node_features, packed_graph.edge_features):
            if matrix.size == 0:
                continue
            total_values += int(matrix.size)
            missing_values += int(np.isnan(matrix).sum())
            missing_values += int(np.isinf(matrix).sum())
    if total_values == 0:
        return 0.0
    return float(missing_values / total_values)


def run_ctu13_public_mixedflow_experiment(
    *,
    export_dir: str | Path,
    scenario_assets: list[CTU13ScenarioAsset],
    run_name: str,
    mode: CTU13MixedFlowMode,
    config: PcapGraphExperimentConfig | None = None,
    timestamp: object | None = None,
) -> CTU13PublicMixedFlowResult:
    runtime_config = (config or PcapGraphExperimentConfig()).with_checkpoint_directory(
        Path(export_dir).joinpath("_tmp_checkpoints")
    )
    run_id = _slugify_token(run_name)
    timestamp_token = _timestamp_token(timestamp)
    layout = build_run_bundle_layout(export_dir, run_id=run_id, timestamp=timestamp_token)
    run_directory = Path(layout.run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)

    load_started = time.perf_counter()
    dataset, flow_label_by_id, loader_summary = _load_public_ctu13_flow_dataset(
        scenario_assets,
        mode=mode,
    )
    dataset_summary = dataset.summary()
    preprocessing_notes = [
        "Source records come from official public labeled bidirectional NetFlows, not file-level PCAP labels.",
        "Background flows are dropped in clean mode and retained as negative-class pressure-test flows in full mode.",
    ]
    parse_summary = _mixedflow_load_summary(
        source_name=f"ctu13_public_mixedflow_{mode}",
        source_path=",".join(asset.label_file_path for asset in scenario_assets),
        label_row_count=int(loader_summary["label_row_count"]),
        kept_flow_count=int(loader_summary["kept_flow_count"]),
        dataset_summary=dataset_summary,
        notes=preprocessing_notes,
    )
    window_batches = preprocess_flow_dataset(
        dataset,
        window_size=runtime_config.window_size,
        rules=runtime_config.short_flow_thresholds,
    )
    graph_builder = FlowInteractionGraphBuilder(
        GraphConfig(
            time_window_seconds=runtime_config.window_size,
            directed=True,
            association_edges=AssociationEdgeConfig(
                enable_same_src_ip=runtime_config.use_association_edges,
                enable_same_dst_subnet=runtime_config.use_association_edges,
                dst_subnet_prefix=24,
            ),
        )
    )
    graphs = graph_builder.build_many(window_batches)
    preprocessing_elapsed = time.perf_counter() - load_started
    if not graphs:
        raise ValueError("The CTU13 public mixed-flow dataset did not yield any graphs.")

    all_entries: list[PcapGraphEntry] = []
    benign_entries: list[PcapGraphEntry] = []
    malicious_entries: list[PcapGraphEntry] = []
    unknown_negative_entries: list[PcapGraphEntry] = []
    for batch, graph in zip(window_batches, graphs, strict=False):
        label_type = _graph_label_for_batch(batch, flow_label_by_id)
        binary_label = 1 if label_type == "malicious" else 0
        source_role = "malicious" if binary_label == 1 else "benign"
        entry = PcapGraphEntry(
            graph=graph,
            source_id=f"scenario_window:{graph.window_index}",
            source_path=",".join(asset.label_file_path for asset in scenario_assets),
            source_name=f"CTU13_{mode}_{label_type}",
            source_role=source_role,
            raw_label=label_type,
            binary_label=binary_label,
            entry_index=len(all_entries),
        )
        all_entries.append(entry)
        if label_type == "malicious":
            malicious_entries.append(entry)
        elif label_type == "unknown":
            unknown_negative_entries.append(entry)
        else:
            benign_entries.append(entry)

    benign_train_pool, benign_test_entries = _partition_graph_entries(
        benign_entries,
        benign_train_ratio=runtime_config.benign_train_ratio,
        random_seed=runtime_config.random_seed,
    )
    train_graphs, val_graphs = _split_graphs(
        [entry.graph for entry in benign_train_pool],
        validation_ratio=runtime_config.train_validation_ratio,
    )
    train_graph_ids = {id(graph) for graph in train_graphs}
    _mark_assignments(benign_test_entries, "benign_test")
    _mark_assignments(unknown_negative_entries, "unknown_test")
    _mark_assignments(malicious_entries, "malicious_test")
    for entry in benign_train_pool:
        entry.split_assignment = "train" if id(entry.graph) in train_graph_ids else "val"
    evaluation_entries = benign_test_entries + unknown_negative_entries + malicious_entries
    if not train_graphs:
        raise ValueError("The CTU13 mixed-flow run did not produce any benign training graphs.")
    if not evaluation_entries:
        raise ValueError("The CTU13 mixed-flow run did not produce any evaluation graphs.")

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
    feature_missing_ratio = _packed_graph_missing_ratio(train_packed_graphs + evaluation_packed_graphs)

    pipeline_config = runtime_config.to_pipeline_config(
        input_path=",".join(asset.label_file_path for asset in scenario_assets),
        run_name=run_id,
        output_directory=run_directory.as_posix(),
    )

    training_started = time.perf_counter()
    if _has_torch():
        (
            training_history,
            checkpoint_paths,
            graph_rows,
            _node_rows,
            _edge_rows,
            _flow_rows,
            _threshold,
            _train_score_summary,
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
            _node_rows,
            _edge_rows,
            _flow_rows,
            _threshold,
            _train_score_summary,
            backend_notes,
        ) = _score_with_fallback(
            graph_samples=evaluation_graphs,
            packed_graphs=evaluation_packed_graphs,
            threshold_percentile=runtime_config.threshold_percentile,
            graph_score_reduction=runtime_config.graph_score_reduction,
            threshold_reference_packed_graphs=train_packed_graphs,
        )
        backend = "fallback"
    training_elapsed = time.perf_counter() - training_started

    if train_graphs:
        if backend == "gae":
            checkpoint_path = checkpoint_paths.get("best_checkpoint") or checkpoint_paths.get("latest_checkpoint")
            if not checkpoint_path:
                raise ValueError("CTU13 mixed-flow GAE training did not return a checkpoint.")
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

    inference_started = time.perf_counter()
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
    contextual_graph_rows = _contextualize_score_rows(
        graph_rows,
        scope="graph",
        entries=evaluation_entries,
        packed_graphs=evaluation_packed_graphs,
        mode="binary_evaluation",
        backend=backend,
        task_name="overall",
    )
    overall_score_records = [
        _score_record_from_row(
            row,
            run_id=run_id,
            timestamp=timestamp_token,
            threshold=threshold,
            split="overall_test",
            task_name="overall",
        )
        for row in contextual_graph_rows
    ]
    inference_elapsed = time.perf_counter() - inference_started

    labels = [record.binary_label for record in overall_score_records]
    scores = [record.anomaly_score for record in overall_score_records]
    overall_metrics = evaluate_scores(labels, scores, threshold=threshold)
    negative_count = int(overall_metrics.negative_count)
    false_positive_rate = (
        float(overall_metrics.false_positive / negative_count)
        if negative_count > 0
        else None
    )

    artifact_paths: dict[str, str] = {}
    row_counts: dict[str, int] = {}

    overall_scores_csv = run_directory / "overall_scores.csv"
    overall_scores_jsonl = run_directory / "overall_scores.jsonl"
    graph_summary_csv = run_directory / "graph_summary.csv"
    graph_summary_json = run_directory / "graph_summary.json"
    summary_json = run_directory / "ctu13_public_mixedflow_summary.json"

    row_counts["overall_scores_csv"] = _write_csv(
        overall_scores_csv,
        [record.to_csv_dict() for record in overall_score_records],
        BINARY_DETECTION_SCORE_FIELDS,
    )
    row_counts["overall_scores_jsonl"] = _write_jsonl(
        overall_scores_jsonl,
        [record.to_dict() for record in overall_score_records],
    )
    graph_summary_rows = [_graph_summary_row(entry) for entry in all_entries]
    row_counts["graph_summary_csv"] = _write_csv(
        graph_summary_csv,
        graph_summary_rows,
        tuple(graph_summary_rows[0].keys()) if graph_summary_rows else (),
    )
    _write_json(graph_summary_json, graph_summary_rows)

    artifact_paths["overall_scores_csv"] = overall_scores_csv.as_posix()
    artifact_paths["overall_scores_jsonl"] = overall_scores_jsonl.as_posix()
    artifact_paths["graph_summary_csv"] = graph_summary_csv.as_posix()
    artifact_paths["graph_summary_json"] = graph_summary_json.as_posix()

    checkpoint_dir: str | None = None
    if checkpoint_paths:
        checkpoint_root = run_directory / "checkpoints"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        for name, path in checkpoint_paths.items():
            if not path:
                continue
            source = Path(path)
            if not source.exists():
                continue
            if source.is_dir():
                target_dir = checkpoint_root / name
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                if source.resolve() != target_dir.resolve():
                    shutil.copytree(source, target_dir)
                continue
            target = checkpoint_root / f"{name}{source.suffix or '.pt'}"
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
        checkpoint_dir = checkpoint_root.as_posix()
        artifact_paths["checkpoint_dir"] = checkpoint_dir

    notes = list(backend_notes)
    if mode == "clean":
        notes.append("Background flows were dropped before graph construction.")
    else:
        notes.append("Background flows were retained and mapped into the negative class for pressure testing.")
    notes.append("Training still uses benign-only graphs; background graphs are evaluation-only negatives.")

    summary_payload = {
        "run_id": run_id,
        "timestamp": timestamp_token,
        "dataset_name": "CTU13",
        "input_mode": "public_mixedflow",
        "mode": mode,
        "backend": backend,
        "scenario_ids": [asset.scenario_id for asset in scenario_assets],
        "label_paths": [asset.label_file_path for asset in scenario_assets],
        "threshold": float(threshold),
        "threshold_percentile": runtime_config.threshold_percentile,
        "graph_score_reduction": runtime_config.graph_score_reduction,
        "label_loader_summary": loader_summary,
        "total_packets": 0,
        "parsed_packets": 0,
        "total_flows": len(dataset.records),
        "total_graphs": len(all_entries),
        "benign_graph_count": len(benign_entries),
        "unknown_graph_count": len(unknown_negative_entries),
        "malicious_graph_count": len(malicious_entries),
        "train_graph_count": len(train_graphs),
        "val_graph_count": len(val_graphs),
        "evaluation_graph_count": len(evaluation_entries),
        "flow_construction_success_rate": (
            float(len(dataset.records) / max(int(loader_summary["kept_flow_count"]), 1))
            if int(loader_summary["kept_flow_count"]) > 0
            else None
        ),
        "graph_construction_success_rate": (
            float(len(all_entries) / max(len(window_batches), 1)) if window_batches else None
        ),
        "feature_extraction_missing_ratio": feature_missing_ratio,
        "avg_preprocessing_time_per_sample": (
            float(preprocessing_elapsed / max(len(all_entries), 1))
        ),
        "avg_training_time": training_elapsed,
        "avg_inference_time": float(inference_elapsed / max(len(evaluation_entries), 1)),
        "overall_metrics": {
            "precision": float(overall_metrics.precision or 0.0),
            "recall": float(overall_metrics.recall or 0.0),
            "f1": float(overall_metrics.f1 or 0.0),
            "roc_auc": overall_metrics.roc_auc,
            "pr_auc": overall_metrics.pr_auc,
            "false_positive_rate": false_positive_rate,
            "true_positive": overall_metrics.true_positive,
            "false_positive": overall_metrics.false_positive,
            "true_negative": overall_metrics.true_negative,
            "false_negative": overall_metrics.false_negative,
            "threshold": float(threshold),
        },
        "train_graph_score_summary": train_score_summary,
        "training_history": training_history,
        "notes": notes,
        "artifact_paths": artifact_paths,
        "row_counts": row_counts,
    }
    _write_json(summary_json, summary_payload)
    artifact_paths["summary_json"] = summary_json.as_posix()

    return CTU13PublicMixedFlowResult(
        run_id=run_id,
        timestamp=timestamp_token,
        mode=mode,
        summary=summary_payload,
        artifact_paths=artifact_paths,
        run_directory=run_directory.as_posix(),
        checkpoint_dir=checkpoint_dir,
        notes=notes,
    )


def _partition_graph_entries(
    benign_entries: list[PcapGraphEntry],
    *,
    benign_train_ratio: float,
    random_seed: int,
) -> tuple[list[PcapGraphEntry], list[PcapGraphEntry]]:
    if not benign_entries:
        return [], []
    if len(benign_entries) == 1:
        benign_entries[0].split_assignment = "train"
        return list(benign_entries), []
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


__all__ = [
    "CTU13MixedFlowMode",
    "CTU13PublicMixedFlowResult",
    "CTU13ScenarioAsset",
    "run_ctu13_public_mixedflow_experiment",
]
