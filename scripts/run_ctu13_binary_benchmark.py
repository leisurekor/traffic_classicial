#!/usr/bin/env python3
"""Run the CTU-13 mixed-PCAP binary benchmark on scenarios 48 / 49 / 52."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

try:
    import torch
except ImportError:  # pragma: no cover - benchmark runtime handles this explicitly
    torch = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_graph.config import AssociationEdgeConfig, FeatureNormalizationConfig, GraphConfig  # noqa: E402
from traffic_graph.data import (  # noqa: E402
    FlowDataset,
    LogicalFlowBatch,
    LogicalFlowRecord,
    LogicalFlowWindowStats,
    ShortFlowThresholds,
    inspect_classic_pcap,
    load_pcap_flow_dataset,
    preprocess_flow_dataset,
)
from traffic_graph.datasets import (  # noqa: E402
    CTU13LabeledFlow,
    align_flow_dataset_to_ctu13_labels,
    load_ctu13_manifest,
    merge_ctu13_manifest_with_local_raw,
    parse_ctu13_label_file,
    save_ctu13_manifest,
    write_alignment_summary_csv,
    write_alignment_summary_markdown,
)
from traffic_graph.features import fit_feature_preprocessor, transform_graphs  # noqa: E402
from traffic_graph.graph import FlowInteractionGraphBuilder  # noqa: E402
from traffic_graph.graph.nx_compat import HAS_NETWORKX, nx  # noqa: E402
from traffic_graph.pipeline.edge_calibration import (  # noqa: E402
    EdgeGraphScoreBreakdown,
    LocalSupportSummary,
    NeighborhoodPersistenceSummary,
    TemporalConsistencySummary,
    apply_edge_calibration,
    build_support_summary_aware_decision,
    calibrate_edge_profile,
    default_edge_calibration_profiles,
    suppressed_graph_score,
)
from traffic_graph.pipeline.candidate_region_proposal import (  # noqa: E402
    CandidateRegion,
    CandidateRegionProposalMode,
    propose_candidate_regions,
)
from traffic_graph.pipeline.graph_extraction_modes import (  # noqa: E402
    GraphExtractionMode,
    extract_flow_groups,
)
from traffic_graph.pipeline.episode_graph_builder import build_episode_graph  # noqa: E402
from traffic_graph.pipeline.episode_proposal import Episode, EpisodeProposalMode, propose_episodes  # noqa: E402
from traffic_graph.pipeline.episode_sessionization import EpisodeStitchingMode, sessionize_episodes  # noqa: E402
from traffic_graph.pipeline.micrograph_verifier import (  # noqa: E402
    FinalDecisionMode,
    MicrographVerificationResult,
    aggregate_micrograph_decisions,
    verify_candidate_region,
)
from traffic_graph.pipeline.nuisance_boundary import (  # noqa: E402
    NuisanceAwareGraphScore,
    calibrate_nuisance_boundary,
    default_nuisance_boundary_profiles,
    score_graph_nuisance_aware,
)
from traffic_graph.pipeline.nuisance_aware_scoring import (  # noqa: E402
    EpisodeDecisionMode,
    NuisanceAwareEpisodeScore,
    aggregate_episode_graph_decision,
    calibrate_nuisance_aware_scores,
    relabel_episode_scores,
    score_episode_nuisance_aware,
)
from traffic_graph.pipeline.scoring import compute_edge_anomaly_scores, compute_node_anomaly_scores  # noqa: E402


TARGET_SCENARIO_IDS = ("48", "49", "52")
RAW_ROOT = REPO_ROOT / "data" / "ctu13" / "raw"
MANIFEST_PATH = REPO_ROOT / "data" / "ctu13" / "ctu13_manifest.json"
ALIGNMENT_CSV = REPO_ROOT / "results" / "ctu13_flow_label_alignment_summary.csv"
ALIGNMENT_MD = REPO_ROOT / "results" / "ctu13_flow_label_alignment_summary.md"
BENCHMARK_CSV = REPO_ROOT / "results" / "ctu13_binary_benchmark.csv"
BENCHMARK_MD = REPO_ROOT / "results" / "ctu13_binary_benchmark.md"
SUPPORT_SUMMARY_DIAGNOSIS_CSV = REPO_ROOT / "results" / "ctu13_support_summary_diagnosis.csv"
SUPPORT_SUMMARY_DIAGNOSIS_MD = REPO_ROOT / "results" / "ctu13_support_summary_diagnosis.md"
CANDIDATE_REGION_CSV = REPO_ROOT / "results" / "ctu13_candidate_region_proposal.csv"
CANDIDATE_REGION_MD = REPO_ROOT / "results" / "ctu13_candidate_region_proposal.md"
MICROGRAPH_VERIFIER_CSV = REPO_ROOT / "results" / "ctu13_micrograph_verifier.csv"
MICROGRAPH_VERIFIER_MD = REPO_ROOT / "results" / "ctu13_micrograph_verifier.md"
GRAPH_EXTRACTION_MODES_CSV = REPO_ROOT / "results" / "ctu13_graph_extraction_modes.csv"
GRAPH_EXTRACTION_MODES_MD = REPO_ROOT / "results" / "ctu13_graph_extraction_modes.md"
PROPOSAL_QUALITY_DIAGNOSIS_CSV = REPO_ROOT / "results" / "ctu13_proposal_quality_diagnosis.csv"
PROPOSAL_QUALITY_DIAGNOSIS_MD = REPO_ROOT / "results" / "ctu13_proposal_quality_diagnosis.md"
S52_DIAGNOSIS_CSV = REPO_ROOT / "results" / "ctu13_s52_coverage_diagnosis.csv"
S52_DIAGNOSIS_MD = REPO_ROOT / "results" / "ctu13_s52_coverage_diagnosis.md"
PRIMARY_EXTRACTION_CSV = REPO_ROOT / "results" / "ctu13_primary_graph_extraction_summary.csv"
PRIMARY_EXTRACTION_MD = REPO_ROOT / "results" / "ctu13_primary_graph_extraction_summary.md"
UNKNOWN_SUPPRESSION_CSV = REPO_ROOT / "results" / "ctu13_unknown_suppression_diagnosis.csv"
UNKNOWN_SUPPRESSION_MD = REPO_ROOT / "results" / "ctu13_unknown_suppression_diagnosis.md"
EPISODE_GRAPH_CONSTRUCTION_CSV = REPO_ROOT / "results" / "ctu13_episode_graph_construction.csv"
EPISODE_GRAPH_CONSTRUCTION_MD = REPO_ROOT / "results" / "ctu13_episode_graph_construction.md"
EPISODE_QUALITY_DIAGNOSIS_CSV = REPO_ROOT / "results" / "ctu13_episode_quality_diagnosis.csv"
EPISODE_QUALITY_DIAGNOSIS_MD = REPO_ROOT / "results" / "ctu13_episode_quality_diagnosis.md"
EPISODE_GRAPH_BENCHMARK_CSV = REPO_ROOT / "results" / "ctu13_episode_graph_benchmark.csv"
EPISODE_GRAPH_BENCHMARK_MD = REPO_ROOT / "results" / "ctu13_episode_graph_benchmark.md"
NUISANCE_AWARE_DIAGNOSIS_CSV = REPO_ROOT / "results" / "ctu13_nuisance_aware_diagnosis.csv"
NUISANCE_AWARE_DIAGNOSIS_MD = REPO_ROOT / "results" / "ctu13_nuisance_aware_diagnosis.md"
EPISODE_SESSIONIZATION_CSV = REPO_ROOT / "results" / "ctu13_episode_sessionization.csv"
EPISODE_SESSIONIZATION_MD = REPO_ROOT / "results" / "ctu13_episode_sessionization.md"
NUISANCE_BOUNDARY_DIAGNOSIS_CSV = REPO_ROOT / "results" / "ctu13_nuisance_boundary_diagnosis.csv"
NUISANCE_BOUNDARY_DIAGNOSIS_MD = REPO_ROOT / "results" / "ctu13_nuisance_boundary_diagnosis.md"
NUISANCE_AWARE_BENCHMARK_CSV = REPO_ROOT / "results" / "ctu13_nuisance_aware_benchmark.csv"
NUISANCE_AWARE_BENCHMARK_MD = REPO_ROOT / "results" / "ctu13_nuisance_aware_benchmark.md"

WINDOW_SIZE_SECONDS = 5
PREFIX_PACKET_COUNT = 16
TRAIN_EPOCHS = 1
MAX_PACKETS_BY_SCENARIO: dict[str, int] = {
    "48": 1_300_000,
    "49": 1_100_000,
    "52": 1_300_000,
}
SCENARIO_WINDOW_SIZE_SECONDS: dict[str, int] = {
    "48": 5,
    "49": 5,
    "52": 2,
}
SCENARIO_START_OFFSET_SECONDS_PADDING: dict[str, float] = {
    "52": 30.0,
}

BASELINE_PERCENTILES = (95.0, 97.0, 99.0)
EDGE_TEMPORAL_FIELD_NAMES: tuple[str, ...] = (
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
    "flow_internal_emb_0",
    "flow_internal_emb_1",
    "flow_internal_emb_2",
    "flow_internal_emb_3",
    "flow_internal_emb_4",
    "flow_internal_emb_5",
    "flow_internal_emb_6",
    "flow_internal_emb_7",
    "flow_internal_emb_8",
    "flow_internal_emb_9",
    "flow_internal_emb_10",
    "flow_internal_emb_11",
    "flow_internal_emb_12",
    "flow_internal_emb_13",
    "flow_internal_emb_14",
    "flow_internal_emb_15",
)
FLOW_EMBEDDING_FIELDS = tuple(field for field in EDGE_TEMPORAL_FIELD_NAMES if field.startswith("flow_internal_emb_"))
RELATION_FIELDS: tuple[str, ...] = (
    "pkt_count",
    "byte_count",
    "duration",
    "flow_count",
    "retry_like_count",
    "retry_like_ratio",
    "seq_ack_match_ratio",
    "unmatched_seq_ratio",
    "unmatched_ack_ratio",
    "retry_burst_count",
    "retry_burst_max_len",
    "retry_like_dense_ratio",
    "small_pkt_burst_count",
    "small_pkt_burst_ratio",
    "flow_internal_packet_count",
    "flow_internal_sequential_edge_count",
    "flow_internal_window_edge_count",
    "flow_internal_ack_edge_count",
    "flow_internal_opposite_direction_edge_count",
)
LOCAL_SUPPORT_ANALYSIS_TOP_K = 5
TEMPORAL_SLICE_COUNT = 3
TWO_STAGE_PROPOSAL_MODES: tuple[CandidateRegionProposalMode, ...] = (
    "edge_seed_region_v2",
    "temporal_burst_region_v2",
    "support_cluster_region",
)
TWO_STAGE_EXTRACTION_MODES: tuple[GraphExtractionMode, ...] = (
    "per_src_ip_within_window",
    "short_temporal_slice_src_pair",
    "neighborhood_local_burst",
)
TWO_STAGE_FINAL_DECISION_MODES: tuple[FinalDecisionMode, ...] = (
    "max_micrograph_score",
    "top2_micrograph_mean",
    "consistency_aware_aggregation",
)
TWO_STAGE_PROFILE_NAMES: tuple[str, ...] = (
    "heldout_q95_top1",
    "heldout_q99_top5",
)
EPISODE_PROPOSAL_MODES: tuple[EpisodeProposalMode, ...] = (
    "repeated_pair_episode",
    "local_burst_episode",
    "support_cluster_episode",
)
EPISODE_FINAL_DECISION_MODES: tuple[EpisodeDecisionMode, ...] = (
    "max_episode_score",
    "top2_episode_mean",
    "consistency_aware_episode",
)
EPISODE_STITCHING_MODES: tuple[EpisodeStitchingMode, ...] = (
    "repeated_pair_temporal_continuity",
    "protocol_consistent_interaction_chain",
    "repeated_local_burst_stitching",
)


@dataclass(slots=True)
class ScenarioWindowSample:
    scenario_id: str
    window_index: int
    group_key: str
    label: str
    graph: object
    logical_flows: tuple[LogicalFlowRecord, ...] = ()
    extraction_mode: str = "per_src_ip_within_window"


@dataclass(slots=True)
class PreparedScenario:
    scenario_id: str
    train_graphs: list[object]
    calib_graphs: list[object]
    test_samples: list[ScenarioWindowSample]
    unknown_samples: list[ScenarioWindowSample]


@dataclass(slots=True)
class ScenarioCoverageDiagnosis:
    scenario_id: str
    packet_start_offset: int
    packet_limit: int
    total_label_flows: int
    label_malicious_flows: int
    loaded_flow_count: int
    aligned_malicious_flows: int
    malicious_flows_after_primary_filter: int
    malicious_flows_after_windowing: int
    malicious_candidate_graphs: int
    final_test_malicious_graphs: int
    major_loss_stage: str


@dataclass(slots=True)
class PrimaryGraphExtractionSummary:
    scenario_id: str
    window_size: int
    graph_grouping_policy: str
    candidate_graph_count: int
    benign_graph_count: int
    malicious_graph_count: int
    unknown_heavy_graph_count: int
    filtered_out_reason: str


@dataclass(slots=True)
class ScoredSample:
    sample: ScenarioWindowSample
    graph_score: float
    edge_breakdown: EdgeGraphScoreBreakdown | None = None
    edge_scores: np.ndarray | None = None
    communication_indices: list[int] | None = None


def _subset_logical_flow_batch(
    batch: LogicalFlowBatch,
    logical_flows: list[LogicalFlowRecord],
) -> LogicalFlowBatch:
    subset = tuple(logical_flows)
    short_flow_count = sum(flow.is_aggregated_short_flow for flow in subset)
    long_flow_count = len(subset) - short_flow_count
    stats = LogicalFlowWindowStats(
        index=batch.index,
        window_start=batch.window_start,
        window_end=batch.window_end,
        raw_flow_count=batch.stats.raw_flow_count,
        short_flow_count=short_flow_count,
        long_flow_count=long_flow_count,
        logical_flow_count=len(subset),
    )
    return LogicalFlowBatch(
        index=batch.index,
        window_start=batch.window_start,
        window_end=batch.window_end,
        logical_flows=subset,
        stats=stats,
    )


def _logical_flow_binary_label(logical_flow: LogicalFlowRecord, flow_label_by_id: dict[str, str]) -> str:
    labels = {
        flow_label_by_id.get(source_flow_id, "unknown")
        for source_flow_id in logical_flow.source_flow_ids
    }
    if "malicious" in labels:
        return "malicious"
    if labels == {"benign"}:
        return "benign"
    return "unknown"


def _pack_with_labels(
    dataset: FlowDataset,
    *,
    scenario_id: str,
    extraction_mode: GraphExtractionMode = "per_src_ip_within_window",
) -> tuple[list[ScenarioWindowSample], list[ScenarioWindowSample]]:
    window_size_seconds = SCENARIO_WINDOW_SIZE_SECONDS.get(scenario_id, WINDOW_SIZE_SECONDS)
    window_batches = preprocess_flow_dataset(
        dataset,
        window_size=window_size_seconds,
        rules=ShortFlowThresholds(packet_count_lt=5, byte_count_lt=1024),
    )
    graph_builder = FlowInteractionGraphBuilder(
        GraphConfig(
            time_window_seconds=window_size_seconds,
            directed=True,
            association_edges=AssociationEdgeConfig(
                enable_same_src_ip=True,
                enable_same_dst_subnet=True,
                enable_same_dst_ip=True,
                enable_same_prefix_signature=True,
                enable_prefix_similarity=False,
                prefix_similarity_threshold=0.97,
                prefix_similarity_top_k=1,
            ),
        )
    )
    flow_label_by_id = {
        record.flow_id: str(record.metadata.get("ctu13_binary_label", "unknown"))
        for record in dataset.records
    }

    primary_samples: list[ScenarioWindowSample] = []
    unknown_samples: list[ScenarioWindowSample] = []
    for batch in window_batches:
        labeled_flows = [
            (logical_flow, _logical_flow_binary_label(logical_flow, flow_label_by_id))
            for logical_flow in batch.logical_flows
        ]
        primary_flows = [
            logical_flow
            for logical_flow, label in labeled_flows
            if label in {"benign", "malicious"}
        ]
        unknown_flows = [
            logical_flow
            for logical_flow, label in labeled_flows
            if label == "unknown"
        ]
        if primary_flows:
            for extracted_group in extract_flow_groups(
                batch,
                primary_flows,
                extraction_mode=extraction_mode,
                slice_count=TEMPORAL_SLICE_COUNT,
            ):
                group_key = extracted_group.group_key
                grouped_flows = list(extracted_group.logical_flows)
                primary_labels = {
                    _logical_flow_binary_label(logical_flow, flow_label_by_id)
                    for logical_flow in grouped_flows
                }
                primary_batch = _subset_logical_flow_batch(batch, grouped_flows)
                graph = graph_builder.build(primary_batch)
                label = "malicious" if "malicious" in primary_labels else "benign"
                primary_samples.append(
                    ScenarioWindowSample(
                        scenario_id=scenario_id,
                        window_index=graph.window_index,
                        group_key=group_key,
                        label=label,
                        graph=graph,
                        logical_flows=tuple(grouped_flows),
                        extraction_mode=extraction_mode,
                    )
                )
        if unknown_flows:
            for extracted_group in extract_flow_groups(
                batch,
                unknown_flows,
                extraction_mode=extraction_mode,
                slice_count=TEMPORAL_SLICE_COUNT,
            ):
                group_key = extracted_group.group_key
                grouped_flows = list(extracted_group.logical_flows)
                unknown_batch = _subset_logical_flow_batch(batch, grouped_flows)
                unknown_graph = graph_builder.build(unknown_batch)
                unknown_samples.append(
                    ScenarioWindowSample(
                        scenario_id=scenario_id,
                        window_index=unknown_graph.window_index,
                        group_key=group_key,
                        label="unknown",
                        graph=unknown_graph,
                        logical_flows=tuple(grouped_flows),
                        extraction_mode=extraction_mode,
                    )
                )
    return primary_samples, unknown_samples


def _split_three_way(
    benign_samples: list[ScenarioWindowSample],
) -> tuple[list[ScenarioWindowSample], list[ScenarioWindowSample], list[ScenarioWindowSample]]:
    if not benign_samples:
        return [], [], []
    ordered = sorted(benign_samples, key=lambda sample: (sample.window_index, sample.group_key))
    count = len(ordered)
    if count == 1:
        return ordered, [], []
    if count == 2:
        return ordered[:1], ordered[1:], []
    if count == 3:
        return ordered[:1], ordered[1:2], ordered[2:]

    train_count = max(1, int(np.floor(count * 0.5)))
    calib_count = max(1, int(np.floor(count * 0.2)))
    test_count = count - train_count - calib_count
    while test_count < 1 and train_count > 1:
        train_count -= 1
        test_count = count - train_count - calib_count
    while test_count < 1 and calib_count > 1:
        calib_count -= 1
        test_count = count - train_count - calib_count
    train = ordered[:train_count]
    calib = ordered[train_count : train_count + calib_count]
    test = ordered[train_count + calib_count :]
    return train, calib, test


def _group_scenario_samples(
    scenario_id: str,
    primary_samples: list[ScenarioWindowSample],
    unknown_samples: list[ScenarioWindowSample],
) -> PreparedScenario:
    benign_samples = [sample for sample in primary_samples if sample.label == "benign"]
    malicious_samples = [sample for sample in primary_samples if sample.label == "malicious"]
    train_benign, calib_benign, test_benign = _split_three_way(benign_samples)
    return PreparedScenario(
        scenario_id=scenario_id,
        train_graphs=[sample.graph for sample in train_benign],
        calib_graphs=[sample.graph for sample in calib_benign],
        test_samples=test_benign + malicious_samples,
        unknown_samples=unknown_samples,
    )


def _bootstrap_manifest() -> list:
    manifest_entries = load_ctu13_manifest(MANIFEST_PATH)
    manifest_entries = merge_ctu13_manifest_with_local_raw(
        manifest_entries,
        raw_root=RAW_ROOT,
        scenario_ids=TARGET_SCENARIO_IDS,
    )
    save_ctu13_manifest(manifest_entries, MANIFEST_PATH)
    selected_entries = [
        entry
        for entry in manifest_entries
        if entry.scenario_id in TARGET_SCENARIO_IDS
        and entry.download_status != "failed"
        and entry.pcap_path
        and entry.label_file_path
    ]
    return sorted(selected_entries, key=lambda entry: entry.scenario_id)


def _estimate_start_packet_offset(
    *,
    entry,
    labeled_flows: list[CTU13LabeledFlow],
) -> int:
    if entry.scenario_id != "52":
        return 0
    malicious_starts = sorted(
        flow.start_time
        for flow in labeled_flows
        if flow.binary_label == "malicious"
    )
    all_starts = sorted(flow.start_time for flow in labeled_flows)
    if not malicious_starts or not all_starts:
        return 0
    malicious_offset_seconds = (
        malicious_starts[0] - all_starts[0]
    ).total_seconds() - SCENARIO_START_OFFSET_SECONDS_PADDING.get(entry.scenario_id, 0.0)
    malicious_offset_seconds = max(malicious_offset_seconds, 0.0)
    quick_probe = load_pcap_flow_dataset(
        entry.pcap_path,
        max_packets=20_000,
        idle_timeout_seconds=60.0,
        prefix_packet_count=PREFIX_PACKET_COUNT,
    )
    probe_start = quick_probe.summary.flow_dataset_summary.earliest_start
    probe_end = quick_probe.summary.flow_dataset_summary.latest_end
    if probe_start is None or probe_end is None:
        return 0
    probe_start_dt = (
        datetime.fromisoformat(probe_start)
        if isinstance(probe_start, str)
        else probe_start
    )
    probe_end_dt = (
        datetime.fromisoformat(probe_end)
        if isinstance(probe_end, str)
        else probe_end
    )
    observed_seconds = max((probe_end_dt - probe_start_dt).total_seconds(), 1e-6)
    packets_per_second = quick_probe.summary.total_packets / observed_seconds
    estimated_offset = int(max(0.0, packets_per_second * malicious_offset_seconds))
    total_packets, _truncated = inspect_classic_pcap(entry.pcap_path)
    packet_limit = MAX_PACKETS_BY_SCENARIO.get(entry.scenario_id, 1_300_000)
    if total_packets > packet_limit:
        estimated_offset = min(estimated_offset, max(total_packets - packet_limit, 0))
    return max(estimated_offset, 0)


def _build_labeled_dataset(entry) -> tuple[FlowDataset, object, int, int, list[CTU13LabeledFlow]]:
    label_flows = parse_ctu13_label_file(
        entry.label_file_path,
        scenario_id=entry.scenario_id,
    )
    start_packet_offset = _estimate_start_packet_offset(
        entry=entry,
        labeled_flows=label_flows,
    )
    packet_limit = MAX_PACKETS_BY_SCENARIO.get(entry.scenario_id, 1_300_000)
    load_result = load_pcap_flow_dataset(
        entry.pcap_path,
        max_packets=packet_limit,
        start_packet_offset=start_packet_offset,
        idle_timeout_seconds=60.0,
        prefix_packet_count=PREFIX_PACKET_COUNT,
    )
    aligned_rows, summary = align_flow_dataset_to_ctu13_labels(
        load_result.dataset,
        label_flows,
        scenario_id=entry.scenario_id,
        time_tolerance_seconds=5.0,
    )
    aligned_by_flow_id = {row.flow_id: row for row in aligned_rows}
    labeled_mappings = []
    for record in load_result.dataset.records:
        mapping = record.to_mapping()
        metadata = dict(mapping.get("metadata", {}))
        aligned = aligned_by_flow_id.get(record.flow_id)
        if aligned is None:
            metadata["ctu13_binary_label"] = "unknown"
            metadata["ctu13_label_text"] = ""
            metadata["ctu13_alignment_status"] = "unaligned"
        else:
            metadata["ctu13_binary_label"] = aligned.aligned_label
            metadata["ctu13_label_text"] = aligned.label_text
            metadata["ctu13_alignment_status"] = aligned.alignment_status
        mapping["metadata"] = metadata
        labeled_mappings.append(mapping)
    return (
        FlowDataset.from_mappings(labeled_mappings),
        summary,
        start_packet_offset,
        packet_limit,
        label_flows,
    )


def _train_model(
    train_graphs: list[object],
    *,
    model_name: str,
):
    if torch is None:
        raise RuntimeError(
            "CTU-13 benchmark requires torch. Run this script in the project conda environment."
        )
    from traffic_graph.models import GraphAutoEncoder, GraphTensorBatch, ReconstructionLossWeights, reconstruction_loss
    from traffic_graph.models.model_types import GraphAutoEncoderConfig
    if not train_graphs:
        raise ValueError("Need at least one benign training graph for CTU-13 benchmark.")
    normalization_config = FeatureNormalizationConfig(method="robust")
    preprocessor = fit_feature_preprocessor(
        train_graphs,
        normalization_config,
        include_graph_structural_features=True,
    )
    packed_train_graphs = transform_graphs(
        train_graphs,
        preprocessor,
        include_graph_structural_features=True,
    )
    sample_graph = packed_train_graphs[0]
    if model_name == "node_recon_baseline":
        config = GraphAutoEncoderConfig(
            hidden_dim=32,
            latent_dim=16,
            num_layers=2,
            dropout=0.1,
            use_edge_features=True,
            reconstruct_edge_features=True,
            use_temporal_edge_projector=False,
            use_edge_categorical_embeddings=True,
        )
        loss_weights = ReconstructionLossWeights(node_weight=1.0, edge_weight=0.5)
    else:
        config = GraphAutoEncoderConfig(
            hidden_dim=48,
            latent_dim=24,
            num_layers=2,
            dropout=0.1,
            use_edge_features=True,
            reconstruct_edge_features=True,
            use_temporal_edge_projector=True,
            temporal_edge_hidden_dim=48,
            temporal_edge_field_names=EDGE_TEMPORAL_FIELD_NAMES,
            use_edge_categorical_embeddings=True,
            edge_categorical_embedding_dim=12,
            edge_categorical_bucket_size=256,
        )
        loss_weights = ReconstructionLossWeights(node_weight=0.25, edge_weight=2.0)

    model = GraphAutoEncoder(
        node_input_dim=sample_graph.node_feature_dim,
        edge_input_dim=sample_graph.edge_feature_dim,
        config=config,
        loss_weights=loss_weights,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    model.train()
    for _epoch in range(TRAIN_EPOCHS):
        for start in range(0, len(packed_train_graphs), 8):
            batch_graphs = packed_train_graphs[start : start + 8]
            tensor_batch = GraphTensorBatch.from_packed_graphs(batch_graphs)
            output = model(tensor_batch)
            loss_output = reconstruction_loss(output, weights=model.loss_weights)
            optimizer.zero_grad(set_to_none=True)
            loss_output.total_loss.backward()
            optimizer.step()
    model.eval()
    return model, preprocessor


def _continuous_field_indices(field_names: tuple[str, ...], selected_fields: tuple[str, ...]) -> list[int]:
    selected = set(selected_fields)
    return [index for index, field_name in enumerate(field_names) if field_name in selected]


def _subset_rowwise_mse(reference: np.ndarray, reconstruction: np.ndarray, indices: list[int]) -> np.ndarray:
    if not indices:
        return np.zeros((reference.shape[0],), dtype=float)
    residual = reference[:, indices] - reconstruction[:, indices]
    return np.mean(residual * residual, axis=1)


def _dst_subnet(ip: str) -> str:
    parts = ip.split(".")
    if len(parts) >= 3:
        return ".".join(parts[:3])
    return ip


def _normalized_entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0 or len(counts) <= 1:
        return 0.0
    probabilities = np.asarray([count / total for count in counts if count > 0], dtype=float)
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    return float(entropy / np.log(len(probabilities)))


def _mean_pairwise_jaccard(endpoint_sets: list[set[str]]) -> float:
    if len(endpoint_sets) < 2:
        return 0.0
    scores: list[float] = []
    for left_index in range(len(endpoint_sets)):
        for right_index in range(left_index + 1, len(endpoint_sets)):
            left = endpoint_sets[left_index]
            right = endpoint_sets[right_index]
            union = left | right
            if not union:
                continue
            scores.append(len(left & right) / len(union))
    return float(np.mean(scores)) if scores else 0.0


def _edge_slice_index(
    sample: ScenarioWindowSample,
    flow_by_id: dict[str, LogicalFlowRecord],
    edge,
    relative_index: int,
    total_edges: int,
    *,
    slice_count: int,
) -> int:
    logical_flow = flow_by_id.get(str(edge.logical_flow_id))
    if logical_flow is None:
        for candidate_id in edge.source_flow_ids:
            logical_flow = flow_by_id.get(candidate_id)
            if logical_flow is not None:
                break
    if logical_flow is None:
        return min(int(relative_index * slice_count / max(total_edges, 1)), slice_count - 1)
    window_seconds = max((sample.graph.window_end - sample.graph.window_start).total_seconds(), 1e-6)
    midpoint = logical_flow.start_time + (logical_flow.end_time - logical_flow.start_time) / 2
    elapsed = (midpoint - sample.graph.window_start).total_seconds()
    ratio = min(max(elapsed / window_seconds, 0.0), 0.999999)
    return min(int(ratio * slice_count), slice_count - 1)


def _local_support_summary(
    sample: ScenarioWindowSample,
    communication_indices: list[int],
    communication_scores: np.ndarray,
    abnormal_relative_indices: list[int],
) -> LocalSupportSummary:
    total_edges = len(communication_indices)
    total_nodes = max(sample.graph.stats.node_count, 1)
    if not abnormal_relative_indices or total_edges == 0:
        return LocalSupportSummary(0, 0.0, 0.0, 0.0, 0, 0.0)
    support_set = set(abnormal_relative_indices)
    source_ip_by_node = {node.node_id: node.ip for node in sample.graph.nodes if node.endpoint_type == "client"}
    dst_ip_by_node = {node.node_id: node.ip for node in sample.graph.nodes if node.endpoint_type == "server"}
    support_clusters: list[tuple[float, int, float, float, int]] = []
    for center_relative_index in abnormal_relative_indices:
        center_edge = sample.graph.edges[communication_indices[center_relative_index]]
        center_src_ip = source_ip_by_node.get(center_edge.source_node_id, "")
        center_dst_ip = dst_ip_by_node.get(center_edge.target_node_id, "")
        center_dst_subnet = _dst_subnet(center_dst_ip)
        local_relative_indices: list[int] = []
        local_nodes: set[str] = set()
        for relative_index, edge_index in enumerate(communication_indices):
            edge = sample.graph.edges[edge_index]
            edge_src_ip = source_ip_by_node.get(edge.source_node_id, "")
            edge_dst_ip = dst_ip_by_node.get(edge.target_node_id, "")
            edge_dst_subnet = _dst_subnet(edge_dst_ip)
            shares_local_context = (
                edge.source_node_id in {center_edge.source_node_id, center_edge.target_node_id}
                or edge.target_node_id in {center_edge.source_node_id, center_edge.target_node_id}
                or (center_src_ip and edge_src_ip == center_src_ip)
                or (center_dst_ip and edge_dst_ip == center_dst_ip)
                or (center_dst_subnet and edge_dst_subnet == center_dst_subnet)
            )
            if not shares_local_context:
                continue
            local_relative_indices.append(relative_index)
            local_nodes.add(edge.source_node_id)
            local_nodes.add(edge.target_node_id)
        if not local_relative_indices:
            continue
        support_edge_count = sum(index in support_set for index in local_relative_indices)
        support_density = support_edge_count / len(local_relative_indices)
        node_coverage = len(local_nodes) / total_nodes
        cluster_score = (
            0.6 * support_density
            + 0.25 * min(support_edge_count / max(len(abnormal_relative_indices), 1), 1.0)
            + 0.15 * node_coverage
        )
        support_clusters.append(
            (
                float(cluster_score),
                int(support_edge_count),
                float(support_density),
                float(node_coverage),
                int(len(local_relative_indices)),
            )
        )
    if not support_clusters:
        return LocalSupportSummary(0, 0.0, 0.0, 0.0, 0, 0.0)
    support_clusters.sort(key=lambda item: (item[0], item[1], item[2], item[4]), reverse=True)
    best_score, best_edge_count, best_density, best_coverage, best_size = support_clusters[0]
    return LocalSupportSummary(
        local_support_edge_count=best_edge_count,
        local_support_edge_density=best_density,
        local_support_node_coverage=best_coverage,
        max_local_support_density=max(item[2] for item in support_clusters),
        top_local_support_component_size=best_size,
        local_support_score=float(best_score),
    )


def _neighborhood_persistence_summary(
    sample: ScenarioWindowSample,
    communication_indices: list[int],
    abnormal_relative_indices: list[int],
) -> NeighborhoodPersistenceSummary:
    if not abnormal_relative_indices or not communication_indices:
        return NeighborhoodPersistenceSummary(0, 0.0, 0.0, 0, 0.0)
    support_set = set(abnormal_relative_indices)
    neighborhood_counts: dict[str, int] = {}
    endpoint_counts: dict[str, int] = {}
    all_neighborhoods = {
        sample.graph.edges[edge_index].target_node_id
        for edge_index in communication_indices
    }
    for relative_index in support_set:
        edge = sample.graph.edges[communication_indices[relative_index]]
        neighborhood_counts[edge.target_node_id] = neighborhood_counts.get(edge.target_node_id, 0) + 1
        endpoint_counts[edge.source_node_id] = endpoint_counts.get(edge.source_node_id, 0) + 1
        endpoint_counts[edge.target_node_id] = endpoint_counts.get(edge.target_node_id, 0) + 1
    abnormal_neighborhood_count = len(neighborhood_counts)
    abnormal_neighborhood_entropy = _normalized_entropy(list(neighborhood_counts.values()))
    cross_neighborhood_support_ratio = abnormal_neighborhood_count / max(len(all_neighborhoods), 1)
    repeated_abnormal_endpoints = sum(count >= 2 for count in endpoint_counts.values())
    persistence_score = (
        0.5 * cross_neighborhood_support_ratio
        + 0.3 * abnormal_neighborhood_entropy
        + 0.2 * min(repeated_abnormal_endpoints / 3.0, 1.0)
    )
    return NeighborhoodPersistenceSummary(
        abnormal_neighborhood_count=abnormal_neighborhood_count,
        abnormal_neighborhood_entropy=float(abnormal_neighborhood_entropy),
        cross_neighborhood_support_ratio=float(cross_neighborhood_support_ratio),
        repeated_abnormal_endpoints=int(repeated_abnormal_endpoints),
        neighborhood_persistence_score=float(persistence_score),
    )


def _temporal_consistency_summary(
    sample: ScenarioWindowSample,
    communication_indices: list[int],
    abnormal_relative_indices: list[int],
) -> TemporalConsistencySummary:
    if not abnormal_relative_indices or not communication_indices:
        return TemporalConsistencySummary(0, 0.0, 0.0, 0, 0.0)
    slice_count = max(2, min(TEMPORAL_SLICE_COUNT, len(communication_indices)))
    flow_by_id = {flow.logical_flow_id: flow for flow in sample.logical_flows}
    slice_support_indices: dict[int, set[int]] = {index: set() for index in range(slice_count)}
    slice_endpoint_sets: dict[int, set[str]] = {index: set() for index in range(slice_count)}
    endpoint_slice_hits: dict[str, set[int]] = {}
    for relative_index in abnormal_relative_indices:
        edge = sample.graph.edges[communication_indices[relative_index]]
        slice_index = _edge_slice_index(
            sample,
            flow_by_id,
            edge,
            relative_index,
            len(communication_indices),
            slice_count=slice_count,
        )
        slice_support_indices[slice_index].add(relative_index)
        slice_endpoint_sets[slice_index].update({edge.source_node_id, edge.target_node_id})
        endpoint_slice_hits.setdefault(edge.source_node_id, set()).add(slice_index)
        endpoint_slice_hits.setdefault(edge.target_node_id, set()).add(slice_index)
    active_slices = [index for index, values in slice_support_indices.items() if values]
    slice_abnormal_presence_count = len(active_slices)
    slice_abnormal_consistency_ratio = slice_abnormal_presence_count / slice_count
    slice_topk_overlap_ratio = _mean_pairwise_jaccard(
        [slice_endpoint_sets[index] for index in active_slices]
    )
    slice_repeated_support_endpoints = sum(len(indices) >= 2 for indices in endpoint_slice_hits.values())
    temporal_score = (
        0.45 * slice_abnormal_consistency_ratio
        + 0.35 * slice_topk_overlap_ratio
        + 0.2 * min(slice_repeated_support_endpoints / 3.0, 1.0)
    )
    return TemporalConsistencySummary(
        slice_abnormal_presence_count=slice_abnormal_presence_count,
        slice_abnormal_consistency_ratio=float(slice_abnormal_consistency_ratio),
        slice_topk_overlap_ratio=float(slice_topk_overlap_ratio),
        slice_repeated_support_endpoints=int(slice_repeated_support_endpoints),
        temporal_consistency_score=float(temporal_score),
    )


def _edge_breakdown(sample: ScenarioWindowSample, edge_scores: np.ndarray, *, top_k: int) -> EdgeGraphScoreBreakdown:
    if edge_scores.size == 0:
        return EdgeGraphScoreBreakdown(
            graph_score=0.0,
            top1_edge_score=0.0,
            topk_mean=0.0,
            abnormal_edge_count=0,
            abnormal_edge_density=0.0,
            abnormal_edge_concentration=0.0,
            max_component_abnormal_ratio=0.0,
            max_server_neighborhood_abnormal_ratio=0.0,
            component_peak=0.0,
            neighborhood_peak=0.0,
            concentration_score=0.0,
        )
    communication_indices = [
        index
        for index, edge in enumerate(sample.graph.edges)
        if edge.edge_type == "communication"
    ]
    if not communication_indices:
        mean_score = float(np.mean(edge_scores))
        return EdgeGraphScoreBreakdown(
            graph_score=mean_score,
            top1_edge_score=mean_score,
            topk_mean=mean_score,
            abnormal_edge_count=0 if mean_score <= 0.0 else 1,
            abnormal_edge_density=1.0 if mean_score > 0.0 else 0.0,
            abnormal_edge_concentration=1.0 if mean_score > 0.0 else 0.0,
            max_component_abnormal_ratio=1.0 if mean_score > 0.0 else 0.0,
            max_server_neighborhood_abnormal_ratio=1.0 if mean_score > 0.0 else 0.0,
            component_peak=mean_score,
            neighborhood_peak=mean_score,
            concentration_score=1.0 if mean_score > 0.0 else 0.0,
            local_support_edge_count=1 if mean_score > 0.0 else 0,
            local_support_edge_density=1.0 if mean_score > 0.0 else 0.0,
            local_support_node_coverage=1.0 if mean_score > 0.0 else 0.0,
            max_local_support_density=1.0 if mean_score > 0.0 else 0.0,
            top_local_support_component_size=1 if mean_score > 0.0 else 0,
            abnormal_neighborhood_count=1 if mean_score > 0.0 else 0,
            abnormal_neighborhood_entropy=0.0,
            cross_neighborhood_support_ratio=1.0 if mean_score > 0.0 else 0.0,
            repeated_abnormal_endpoints=0,
            slice_abnormal_presence_count=1 if mean_score > 0.0 else 0,
            slice_abnormal_consistency_ratio=1.0 if mean_score > 0.0 else 0.0,
            slice_topk_overlap_ratio=1.0 if mean_score > 0.0 else 0.0,
            slice_repeated_support_endpoints=0,
            local_support_score=1.0 if mean_score > 0.0 else 0.0,
            neighborhood_persistence_score=1.0 if mean_score > 0.0 else 0.0,
            temporal_consistency_score=1.0 if mean_score > 0.0 else 0.0,
        )

    communication_scores = np.asarray(
        [edge_scores[index] for index in communication_indices],
        dtype=float,
    )
    effective_top_k = max(1, min(int(top_k), len(communication_scores)))
    analysis_top_k = max(effective_top_k, min(5, len(communication_scores)))
    sorted_indices = np.argsort(communication_scores)
    topk_relative_indices = sorted_indices[-effective_top_k:]
    topk_mean = float(np.mean(communication_scores[topk_relative_indices]))
    top1_edge_score = float(np.max(communication_scores))
    abnormal_relative_indices = list(sorted_indices[-analysis_top_k:])
    abnormal_edge_count = len(abnormal_relative_indices)
    abnormal_edge_density = float(abnormal_edge_count / max(len(communication_scores), 1))

    graph_backend = sample.graph.graph
    component_peaks: list[float] = []
    component_ratios: list[float] = []
    if HAS_NETWORKX:
        try:
            undirected = graph_backend.to_undirected()
            components = [set(component) for component in nx.connected_components(undirected)]
        except Exception:
            components = []
    else:
        components = []
    for component_nodes in components:
        component_relative_indices = [
            relative_index
            for relative_index, edge_index in enumerate(communication_indices)
            if sample.graph.edges[edge_index].source_node_id in component_nodes
            and sample.graph.edges[edge_index].target_node_id in component_nodes
        ]
        component_edge_scores = [
            communication_scores[relative_index]
            for relative_index in component_relative_indices
        ]
        if component_edge_scores:
            k = max(1, min(effective_top_k, len(component_edge_scores)))
            component_peaks.append(float(np.mean(sorted(component_edge_scores)[-k:])))
            if abnormal_relative_indices:
                abnormal_in_component = sum(
                    relative_index in abnormal_relative_indices
                    for relative_index in component_relative_indices
                )
                component_ratios.append(
                    float(abnormal_in_component / max(len(abnormal_relative_indices), 1))
                )
    component_peak = max(component_peaks, default=0.0)
    max_component_abnormal_ratio = max(component_ratios, default=0.0)

    neighborhood_peaks: list[float] = []
    neighborhood_ratios: list[float] = []
    for node in sample.graph.nodes:
        if node.endpoint_type != "server":
            continue
        incident_relative_indices = [
            relative_index
            for relative_index, edge_index in enumerate(communication_indices)
            if (
                sample.graph.edges[edge_index].source_node_id == node.node_id
                or sample.graph.edges[edge_index].target_node_id == node.node_id
            )
        ]
        incident_scores = [
            communication_scores[relative_index]
            for relative_index in incident_relative_indices
        ]
        if incident_scores:
            k = max(1, min(effective_top_k, len(incident_scores)))
            neighborhood_peaks.append(float(np.mean(sorted(incident_scores)[-k:])))
            if abnormal_relative_indices:
                abnormal_in_neighborhood = sum(
                    relative_index in abnormal_relative_indices
                    for relative_index in incident_relative_indices
                )
                neighborhood_ratios.append(
                    float(abnormal_in_neighborhood / max(len(abnormal_relative_indices), 1))
                )
    neighborhood_peak = max(neighborhood_peaks, default=0.0)
    max_server_neighborhood_abnormal_ratio = max(neighborhood_ratios, default=0.0)

    local_support_summary = _local_support_summary(
        sample,
        communication_indices,
        communication_scores,
        abnormal_relative_indices,
    )
    neighborhood_summary = _neighborhood_persistence_summary(
        sample,
        communication_indices,
        abnormal_relative_indices,
    )
    temporal_summary = _temporal_consistency_summary(
        sample,
        communication_indices,
        abnormal_relative_indices,
    )
    concentration_score = max(component_peak, neighborhood_peak) / max(topk_mean, 1e-12)
    abnormal_edge_concentration = max(
        max_component_abnormal_ratio,
        max_server_neighborhood_abnormal_ratio,
    )
    graph_score = float(
        0.55 * topk_mean
        + 0.15 * component_peak
        + 0.10 * neighborhood_peak
        + 0.10 * local_support_summary.local_support_score
        + 0.05 * neighborhood_summary.neighborhood_persistence_score
        + 0.05 * temporal_summary.temporal_consistency_score
    )
    return EdgeGraphScoreBreakdown(
        graph_score=graph_score,
        top1_edge_score=top1_edge_score,
        topk_mean=topk_mean,
        abnormal_edge_count=abnormal_edge_count,
        abnormal_edge_density=abnormal_edge_density,
        abnormal_edge_concentration=float(abnormal_edge_concentration),
        max_component_abnormal_ratio=float(max_component_abnormal_ratio),
        max_server_neighborhood_abnormal_ratio=float(max_server_neighborhood_abnormal_ratio),
        component_peak=component_peak,
        neighborhood_peak=neighborhood_peak,
        concentration_score=float(concentration_score),
        local_support_edge_count=local_support_summary.local_support_edge_count,
        local_support_edge_density=local_support_summary.local_support_edge_density,
        local_support_node_coverage=local_support_summary.local_support_node_coverage,
        max_local_support_density=local_support_summary.max_local_support_density,
        top_local_support_component_size=local_support_summary.top_local_support_component_size,
        abnormal_neighborhood_count=neighborhood_summary.abnormal_neighborhood_count,
        abnormal_neighborhood_entropy=neighborhood_summary.abnormal_neighborhood_entropy,
        cross_neighborhood_support_ratio=neighborhood_summary.cross_neighborhood_support_ratio,
        repeated_abnormal_endpoints=neighborhood_summary.repeated_abnormal_endpoints,
        slice_abnormal_presence_count=temporal_summary.slice_abnormal_presence_count,
        slice_abnormal_consistency_ratio=temporal_summary.slice_abnormal_consistency_ratio,
        slice_topk_overlap_ratio=temporal_summary.slice_topk_overlap_ratio,
        slice_repeated_support_endpoints=temporal_summary.slice_repeated_support_endpoints,
        local_support_score=local_support_summary.local_support_score,
        neighborhood_persistence_score=neighborhood_summary.neighborhood_persistence_score,
        temporal_consistency_score=temporal_summary.temporal_consistency_score,
    )


def _score_samples(
    model,
    samples: list[ScenarioWindowSample],
    preprocessor,
    *,
    model_name: str,
    top_k: int,
) -> list[ScoredSample]:
    scored: list[ScoredSample] = []
    for sample in samples:
        packed_graph = transform_graphs(
            [sample.graph],
            preprocessor,
            include_graph_structural_features=True,
        )[0]
        output = model(packed_graph)
        node_scores = compute_node_anomaly_scores(
            packed_graph.node_features,
            output.reconstructed_node_features.detach().cpu().numpy(),
            discrete_mask=packed_graph.node_discrete_mask,
        )
        if model_name == "node_recon_baseline":
            graph_score = float(np.mean(node_scores)) if node_scores.size else 0.0
            scored.append(ScoredSample(sample=sample, graph_score=graph_score))
            continue

        edge_recon = output.reconstructed_edge_features.detach().cpu().numpy()
        all_edge_scores = compute_edge_anomaly_scores(
            packed_graph.edge_features,
            edge_recon,
            discrete_mask=packed_graph.edge_discrete_mask,
        )
        flow_embedding_indices = _continuous_field_indices(
            packed_graph.edge_feature_fields,
            FLOW_EMBEDDING_FIELDS,
        )
        relation_indices = _continuous_field_indices(
            packed_graph.edge_feature_fields,
            RELATION_FIELDS,
        )
        flow_embedding_scores = _subset_rowwise_mse(
            np.asarray(packed_graph.edge_features, dtype=float),
            np.asarray(edge_recon, dtype=float),
            flow_embedding_indices,
        )
        relation_scores = _subset_rowwise_mse(
            np.asarray(packed_graph.edge_features, dtype=float),
            np.asarray(edge_recon, dtype=float),
            relation_indices,
        )
        edge_scores = 0.55 * all_edge_scores + 0.30 * flow_embedding_scores + 0.15 * relation_scores
        communication_indices = [
            index
            for index, edge in enumerate(sample.graph.edges)
            if edge.edge_type == "communication"
        ]
        breakdown = _edge_breakdown(sample, edge_scores, top_k=top_k)
        scored.append(
            ScoredSample(
                sample=sample,
                graph_score=breakdown.graph_score,
                edge_breakdown=breakdown,
                edge_scores=np.asarray(edge_scores, dtype=float),
                communication_indices=communication_indices,
            )
        )
    return scored


def _binary_metrics(y_true: list[int], y_score: list[float], y_pred: list[int]) -> dict[str, float | None]:
    negative_total = sum(label == 0 for label in y_true)
    false_positive = sum(label == 0 and pred == 1 for label, pred in zip(y_true, y_pred, strict=True))
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(false_positive / negative_total) if negative_total else 0.0,
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, y_score)) if len(set(y_true)) > 1 else None,
    }


def _suspected_failure_mode(
    breakdown: EdgeGraphScoreBreakdown,
    decision_old_positive: bool,
    decision_new_positive: bool,
    decision_new,
) -> str:
    if not decision_old_positive:
        return "none"
    if decision_new_positive:
        return "none"
    if breakdown.abnormal_edge_count < decision_new.min_abnormal_edge_count:
        return "isolated_spike"
    if (
        decision_new.local_density_threshold is not None
        and breakdown.local_support_score < decision_new.local_density_threshold
    ):
        return "weak_local_support_density"
    if (
        decision_new.neighborhood_persistence_threshold is not None
        and breakdown.neighborhood_persistence_score < decision_new.neighborhood_persistence_threshold
    ):
        return "single_neighborhood_spike"
    if (
        decision_new.temporal_consistency_threshold is not None
        and breakdown.temporal_consistency_score < decision_new.temporal_consistency_threshold
    ):
        return "short_lived_slice_spike"
    return "support_summary_gate"


def _build_support_summary_diagnosis_rows(
    scored_samples: list[ScoredSample],
    decision_old,
    decision_new,
) -> list[dict[str, object]]:
    filtered_samples = [item for item in scored_samples if item.edge_breakdown is not None]
    if not filtered_samples:
        return []
    breakdowns = [item.edge_breakdown for item in filtered_samples if item.edge_breakdown is not None]
    predictions_old = apply_edge_calibration(breakdowns, decision_old)
    predictions_new = apply_edge_calibration(breakdowns, decision_new)
    rows: list[dict[str, object]] = []
    for item, old_positive, new_positive in zip(
        filtered_samples,
        predictions_old,
        predictions_new,
        strict=True,
    ):
        breakdown = item.edge_breakdown
        if breakdown is None:
            continue
        rows.append(
            {
                "scenario_id": item.sample.scenario_id,
                "graph_id": f"{item.sample.window_index}:{item.sample.group_key}",
                "label_type": item.sample.label,
                "top1_edge_score": breakdown.top1_edge_score,
                "topk_edge_mean": breakdown.topk_mean,
                "local_support_edge_count": breakdown.local_support_edge_count,
                "local_support_edge_density": breakdown.local_support_edge_density,
                "local_support_node_coverage": breakdown.local_support_node_coverage,
                "max_local_support_density": breakdown.max_local_support_density,
                "abnormal_neighborhood_count": breakdown.abnormal_neighborhood_count,
                "cross_neighborhood_support_ratio": breakdown.cross_neighborhood_support_ratio,
                "slice_abnormal_presence_count": breakdown.slice_abnormal_presence_count,
                "slice_abnormal_consistency_ratio": breakdown.slice_abnormal_consistency_ratio,
                "decision_old": old_positive,
                "decision_new": new_positive,
                "suspected_failure_mode": _suspected_failure_mode(
                    breakdown,
                    bool(old_positive),
                    bool(new_positive),
                    decision_new,
                ),
            }
        )
    return rows


def _nuisance_failure_mode(
    score: NuisanceAwareGraphScore,
    label: str,
) -> str:
    if label == "unknown" and score.final_internal_state != "nuisance_like":
        return "nuisance_boundary_miss"
    if label == "malicious" and score.final_internal_state == "nuisance_like":
        return "malicious_blocked_by_nuisance"
    if label == "malicious" and score.final_internal_state == "benign_like":
        return "anomaly_boundary_miss"
    if label == "benign" and score.final_internal_state == "nuisance_like":
        return "benign_misrejected_as_nuisance"
    return "none"


def _build_nuisance_boundary_diagnosis_rows(
    scored_samples: list[ScoredSample],
    nuisance_scores: list[NuisanceAwareGraphScore],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item, score in zip(scored_samples, nuisance_scores, strict=True):
        breakdown = item.edge_breakdown
        if breakdown is None:
            continue
        rows.append(
            {
                "scenario_id": item.sample.scenario_id,
                "graph_id": f"{item.sample.window_index}:{item.sample.group_key}",
                "label_type": item.sample.label,
                "anomaly_score": score.anomaly_score,
                "nuisance_score": score.nuisance_score,
                "malicious_support_score": score.malicious_support_score,
                "benign_boundary_value": score.benign_boundary_value,
                "nuisance_boundary_value": score.nuisance_boundary_value,
                "final_internal_state": score.final_internal_state,
                "final_binary_decision": score.final_binary_decision,
                "top1_edge_score": breakdown.top1_edge_score,
                "topk_edge_mean": breakdown.topk_mean,
                "local_support_score": breakdown.local_support_score,
                "neighborhood_persistence_score": breakdown.neighborhood_persistence_score,
                "temporal_consistency_score": breakdown.temporal_consistency_score,
                "suspected_failure_mode": _nuisance_failure_mode(score, item.sample.label),
            }
        )
    return rows


def _evaluate_baseline_profiles(
    *,
    evaluation_mode: str,
    scenario_id: str,
    train_graphs: list[object],
    calib_graphs: list[object],
    test_samples: list[ScenarioWindowSample],
    unknown_samples: list[ScenarioWindowSample],
) -> list[dict[str, object]]:
    model, preprocessor = _train_model(train_graphs, model_name="node_recon_baseline")
    calib_scored = _score_samples(
        model,
        [ScenarioWindowSample(scenario_id, -1, "calib", "benign", graph) for graph in calib_graphs],
        preprocessor,
        model_name="node_recon_baseline",
        top_k=1,
    )
    if not calib_scored:
        calib_scored = _score_samples(
            model,
            [ScenarioWindowSample(scenario_id, -1, "train", "benign", graph) for graph in train_graphs],
            preprocessor,
            model_name="node_recon_baseline",
            top_k=1,
        )
    test_scored = _score_samples(
        model,
        test_samples,
        preprocessor,
        model_name="node_recon_baseline",
        top_k=1,
    )
    unknown_scored = _score_samples(
        model,
        unknown_samples,
        preprocessor,
        model_name="node_recon_baseline",
        top_k=1,
    )

    calib_scores = np.asarray([item.graph_score for item in calib_scored], dtype=float)
    y_true = [1 if item.sample.label == "malicious" else 0 for item in test_scored]
    y_score = [item.graph_score for item in test_scored]
    rows: list[dict[str, object]] = []
    for percentile in BASELINE_PERCENTILES:
        threshold = float(np.percentile(calib_scores, percentile)) if calib_scores.size else 0.0
        y_pred = [1 if score >= threshold else 0 for score in y_score]
        metrics = _binary_metrics(y_true, y_score, y_pred)
        unknown_scores = [item.graph_score for item in unknown_scored]
        rows.append(
            {
                "evaluation_mode": evaluation_mode,
                "scenario_id": scenario_id,
                "model_name": "node_recon_baseline",
                "calibration_profile": f"heldout_q{int(percentile)}",
                "percentile_setting": percentile,
                "top_k_setting": 1,
                "suppression_enabled": False,
                "support_summary_mode": "score_only_baseline",
                "extraction_mode": "per_src_ip_within_window",
                "region_proposal_mode": None,
                "verifier_mode": None,
                "final_decision_mode": None,
                "concentration_threshold_setting": None,
                "component_ratio_setting": None,
                "neighborhood_ratio_setting": None,
                "local_density_threshold": None,
                "neighborhood_persistence_threshold": None,
                "temporal_consistency_threshold": None,
                "candidate_region_count_mean": 0.0,
                "selected_region_count_mean": 0.0,
                "selected_region_coverage_mean": 0.0,
                "single_edge_region_ratio": 0.0,
                "mean_candidate_time_span": 0.0,
                "train_benign_graphs": len(train_graphs),
                "calib_benign_graphs": len(calib_graphs),
                "test_benign_graphs": sum(item.sample.label == "benign" for item in test_scored),
                "test_malicious_graphs": sum(item.sample.label == "malicious" for item in test_scored),
                "test_unknown_graphs": len(unknown_scored),
                "threshold": threshold,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "fpr": metrics["fpr"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "background_hit_ratio": (
                    float(sum(score >= threshold for score in unknown_scores) / len(unknown_scores))
                    if unknown_scores
                    else 0.0
                ),
                "unknown_count": len(unknown_scores),
                "unknown_score_mean": float(np.mean(unknown_scores)) if unknown_scores else 0.0,
                "unknown_score_median": float(np.median(unknown_scores)) if unknown_scores else 0.0,
            }
        )
    return rows


def _evaluate_edge_profiles(
    *,
    evaluation_mode: str,
    scenario_id: str,
    train_graphs: list[object],
    calib_graphs: list[object],
    test_samples: list[ScenarioWindowSample],
    unknown_samples: list[ScenarioWindowSample],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    model, preprocessor = _train_model(train_graphs, model_name="edge_temporal_binary_v2")
    rows: list[dict[str, object]] = []
    diagnosis_rows: list[dict[str, object]] = []
    for profile in default_edge_calibration_profiles():
        if profile.name not in TWO_STAGE_PROFILE_NAMES:
            continue
        calib_scored = _score_samples(
            model,
            [ScenarioWindowSample(scenario_id, -1, "calib", "benign", graph) for graph in calib_graphs],
            preprocessor,
            model_name="edge_temporal_binary_v2",
            top_k=profile.top_k,
        )
        if not calib_scored:
            calib_scored = _score_samples(
                model,
                [ScenarioWindowSample(scenario_id, -1, "train", "benign", graph) for graph in train_graphs],
                preprocessor,
                model_name="edge_temporal_binary_v2",
                top_k=profile.top_k,
            )
        test_scored = _score_samples(
            model,
            test_samples,
            preprocessor,
            model_name="edge_temporal_binary_v2",
            top_k=profile.top_k,
        )
        unknown_scored = _score_samples(
            model,
            unknown_samples,
            preprocessor,
            model_name="edge_temporal_binary_v2",
            top_k=profile.top_k,
        )
        benign_breakdowns = [item.edge_breakdown for item in calib_scored if item.edge_breakdown is not None]
        decision_old = calibrate_edge_profile(
            profile,
            benign_breakdowns,
            suppression_enabled=True,
        )
        decision_local_density = build_support_summary_aware_decision(
            profile,
            benign_breakdowns,
            support_summary_mode="local_support_density",
        )
        decision_combined = build_support_summary_aware_decision(
            profile,
            benign_breakdowns,
            support_summary_mode="combined_support_summary",
        )
        test_breakdowns = [item.edge_breakdown for item in test_scored if item.edge_breakdown is not None]
        y_true = [1 if item.sample.label == "malicious" else 0 for item in test_scored]
        unknown_breakdowns = [item.edge_breakdown for item in unknown_scored if item.edge_breakdown is not None]
        for decision in (decision_old, decision_local_density, decision_combined):
            y_score = [
                suppressed_graph_score(item.edge_breakdown, decision) if item.edge_breakdown is not None else item.graph_score
                for item in test_scored
            ]
            y_pred = apply_edge_calibration(test_breakdowns, decision)
            metrics = _binary_metrics(y_true, y_score, y_pred)
            unknown_scores = [
                suppressed_graph_score(item.edge_breakdown, decision) if item.edge_breakdown is not None else item.graph_score
                for item in unknown_scored
            ]
            unknown_predictions = apply_edge_calibration(unknown_breakdowns, decision)
            rows.append(
                {
                    "evaluation_mode": evaluation_mode,
                    "scenario_id": scenario_id,
                    "model_name": "edge_temporal_binary_v2",
                    "calibration_profile": profile.name,
                    "percentile_setting": profile.percentile,
                    "top_k_setting": profile.top_k,
                    "suppression_enabled": decision.suppression_enabled,
                    "support_summary_mode": decision.support_summary_mode,
                    "extraction_mode": "per_src_ip_within_window",
                    "region_proposal_mode": None,
                    "verifier_mode": None,
                    "final_decision_mode": None,
                    "concentration_threshold_setting": decision.concentration_threshold,
                    "component_ratio_setting": decision.component_ratio_threshold,
                    "neighborhood_ratio_setting": decision.neighborhood_ratio_threshold,
                    "local_density_threshold": decision.local_density_threshold,
                    "neighborhood_persistence_threshold": decision.neighborhood_persistence_threshold,
                    "temporal_consistency_threshold": decision.temporal_consistency_threshold,
                    "candidate_region_count_mean": 0.0,
                    "selected_region_count_mean": 0.0,
                    "selected_region_coverage_mean": 0.0,
                    "single_edge_region_ratio": 0.0,
                    "mean_candidate_time_span": 0.0,
                    "train_benign_graphs": len(train_graphs),
                    "calib_benign_graphs": len(calib_graphs),
                    "test_benign_graphs": sum(item.sample.label == "benign" for item in test_scored),
                    "test_malicious_graphs": sum(item.sample.label == "malicious" for item in test_scored),
                    "test_unknown_graphs": len(unknown_scored),
                    "threshold": decision.score_threshold,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "fpr": metrics["fpr"],
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                    "background_hit_ratio": (
                        float(sum(unknown_predictions) / len(unknown_predictions))
                        if unknown_predictions
                        else 0.0
                    ),
                    "unknown_count": len(unknown_scores),
                    "unknown_score_mean": float(np.mean(unknown_scores)) if unknown_scores else 0.0,
                    "unknown_score_median": float(np.median(unknown_scores)) if unknown_scores else 0.0,
                }
            )
        if profile.name == "heldout_q95_top1":
            diagnosis_rows.extend(
                _build_support_summary_diagnosis_rows(
                    test_scored + unknown_scored,
                    decision_old,
                    decision_combined,
                )
            )
    return rows, diagnosis_rows


def _evaluate_nuisance_aware_edge_profiles(
    *,
    evaluation_mode: str,
    scenario_id: str,
    train_graphs: list[object],
    calib_graphs: list[object],
    test_samples: list[ScenarioWindowSample],
    unknown_samples: list[ScenarioWindowSample],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    model, preprocessor = _train_model(train_graphs, model_name="edge_temporal_binary_v2")
    rows: list[dict[str, object]] = []
    diagnosis_rows: list[dict[str, object]] = []
    benchmark_rows: list[dict[str, object]] = []
    for profile in default_edge_calibration_profiles():
        if profile.name not in TWO_STAGE_PROFILE_NAMES:
            continue
        calib_scored = _score_samples(
            model,
            [ScenarioWindowSample(scenario_id, -1, "calib", "benign", graph) for graph in calib_graphs],
            preprocessor,
            model_name="edge_temporal_binary_v2",
            top_k=profile.top_k,
        )
        if not calib_scored:
            calib_scored = _score_samples(
                model,
                [ScenarioWindowSample(scenario_id, -1, "train", "benign", graph) for graph in train_graphs],
                preprocessor,
                model_name="edge_temporal_binary_v2",
                top_k=profile.top_k,
            )
        test_scored = _score_samples(
            model,
            test_samples,
            preprocessor,
            model_name="edge_temporal_binary_v2",
            top_k=profile.top_k,
        )
        unknown_scored = _score_samples(
            model,
            unknown_samples,
            preprocessor,
            model_name="edge_temporal_binary_v2",
            top_k=profile.top_k,
        )
        benign_breakdowns = [item.edge_breakdown for item in calib_scored if item.edge_breakdown is not None]
        unknown_breakdowns = [item.edge_breakdown for item in unknown_scored if item.edge_breakdown is not None]
        base_decisions = (
            calibrate_edge_profile(profile, benign_breakdowns, suppression_enabled=True),
            build_support_summary_aware_decision(
                profile,
                benign_breakdowns,
                support_summary_mode="local_support_density",
            ),
            build_support_summary_aware_decision(
                profile,
                benign_breakdowns,
                support_summary_mode="combined_support_summary",
            ),
        )
        for base_decision in base_decisions:
            for nuisance_profile in default_nuisance_boundary_profiles():
                nuisance_decision = calibrate_nuisance_boundary(
                    base_decision,
                    benign_breakdowns,
                    unknown_breakdowns,
                    profile=nuisance_profile,
                )
                test_scores = [
                    score_graph_nuisance_aware(item.edge_breakdown, nuisance_decision)
                    for item in test_scored
                    if item.edge_breakdown is not None
                ]
                scored_test_samples = [
                    item for item in test_scored if item.edge_breakdown is not None
                ]
                unknown_scores = [
                    score_graph_nuisance_aware(item.edge_breakdown, nuisance_decision)
                    for item in unknown_scored
                    if item.edge_breakdown is not None
                ]
                scored_unknown_samples = [
                    item for item in unknown_scored if item.edge_breakdown is not None
                ]
                y_true = [1 if item.sample.label == "malicious" else 0 for item in scored_test_samples]
                y_score = [item.anomaly_score - item.nuisance_score for item in test_scores]
                y_pred = [item.final_binary_decision for item in test_scores]
                metrics = _binary_metrics(y_true, y_score, y_pred)
                background_hit_ratio = (
                    float(sum(item.final_binary_decision for item in unknown_scores) / len(unknown_scores))
                    if unknown_scores
                    else 0.0
                )
                nuisance_rejection_rate = (
                    float(sum(item.final_internal_state == "nuisance_like" for item in unknown_scores) / len(unknown_scores))
                    if unknown_scores
                    else 0.0
                )
                benign_scores = [
                    item for sample, item in zip(scored_test_samples, test_scores, strict=True)
                    if sample.sample.label == "benign"
                ]
                malicious_scores = [
                    item for sample, item in zip(scored_test_samples, test_scores, strict=True)
                    if sample.sample.label == "malicious"
                ]
                benign_misrejected_as_nuisance_rate = (
                    float(sum(item.final_internal_state == "nuisance_like" for item in benign_scores) / len(benign_scores))
                    if benign_scores
                    else 0.0
                )
                malicious_blocked_by_nuisance_rate = (
                    float(sum(item.final_internal_state == "nuisance_like" for item in malicious_scores) / len(malicious_scores))
                    if malicious_scores
                    else 0.0
                )
                nuisance_like_false_positive_rate = benign_misrejected_as_nuisance_rate
                malicious_like_episode_precision = (
                    float(
                        sum(
                            sample.sample.label == "malicious" and score.final_internal_state == "malicious_like"
                            for sample, score in zip(scored_test_samples, test_scores, strict=True)
                        )
                        / max(sum(score.final_internal_state == "malicious_like" for score in test_scores), 1)
                    )
                    if test_scores
                    else 0.0
                )
                graph_consistency = (
                    float(
                        sum(score.final_binary_decision == (1 if score.final_internal_state == "malicious_like" else 0) for score in test_scores + unknown_scores)
                        / max(len(test_scores) + len(unknown_scores), 1)
                    )
                )
                row = {
                    "evaluation_mode": evaluation_mode,
                    "scenario_id": scenario_id,
                    "model_name": "edge_temporal_binary_v2_nuisance_aware",
                    "calibration_profile": profile.name,
                    "percentile_setting": profile.percentile,
                    "top_k_setting": profile.top_k,
                    "suppression_enabled": base_decision.suppression_enabled,
                    "support_summary_mode": base_decision.support_summary_mode,
                    "nuisance_boundary_mode": nuisance_profile.name,
                    "extraction_mode": "per_src_ip_within_window",
                    "region_proposal_mode": None,
                    "verifier_mode": None,
                    "final_decision_mode": "nuisance_aware_boundary",
                    "concentration_threshold_setting": base_decision.concentration_threshold,
                    "component_ratio_setting": base_decision.component_ratio_threshold,
                    "neighborhood_ratio_setting": base_decision.neighborhood_ratio_threshold,
                    "local_density_threshold": base_decision.local_density_threshold,
                    "neighborhood_persistence_threshold": base_decision.neighborhood_persistence_threshold,
                    "temporal_consistency_threshold": base_decision.temporal_consistency_threshold,
                    "benign_boundary_value": nuisance_decision.benign_boundary_value,
                    "nuisance_boundary_value": nuisance_decision.nuisance_boundary_value,
                    "candidate_region_count_mean": 0.0,
                    "selected_region_count_mean": 0.0,
                    "selected_region_coverage_mean": 0.0,
                    "single_edge_region_ratio": 0.0,
                    "mean_candidate_time_span": 0.0,
                    "train_benign_graphs": len(train_graphs),
                    "calib_benign_graphs": len(calib_graphs),
                    "test_benign_graphs": sum(item.sample.label == "benign" for item in scored_test_samples),
                    "test_malicious_graphs": sum(item.sample.label == "malicious" for item in scored_test_samples),
                    "test_unknown_graphs": len(scored_unknown_samples),
                    "threshold": nuisance_decision.malicious_support_threshold,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "fpr": metrics["fpr"],
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                    "background_hit_ratio": background_hit_ratio,
                    "unknown_count": len(unknown_scores),
                    "unknown_score_mean": float(np.mean([item.anomaly_score - item.nuisance_score for item in unknown_scores])) if unknown_scores else 0.0,
                    "unknown_score_median": float(np.median([item.anomaly_score - item.nuisance_score for item in unknown_scores])) if unknown_scores else 0.0,
                    "nuisance_rejection_rate": nuisance_rejection_rate,
                    "nuisance_like_false_positive_rate": nuisance_like_false_positive_rate,
                    "benign_misrejected_as_nuisance_rate": benign_misrejected_as_nuisance_rate,
                    "malicious_blocked_by_nuisance_rate": malicious_blocked_by_nuisance_rate,
                    "malicious_like_episode_precision": malicious_like_episode_precision,
                    "episode_to_graph_decision_consistency": graph_consistency,
                }
                rows.append(row)
                benchmark_rows.append(dict(row))
                if (
                    profile.name == "heldout_q95_top1"
                    and base_decision.support_summary_mode == "local_support_density"
                    and nuisance_profile.name == "nuisance_q95_margin025"
                ):
                    diagnosis_rows.extend(
                        _build_nuisance_boundary_diagnosis_rows(
                            scored_test_samples + scored_unknown_samples,
                            test_scores + unknown_scores,
                        )
                    )
    return rows, diagnosis_rows, benchmark_rows


def _proposal_top_k(profile) -> int:
    return max(2, min(profile.top_k, 4))


def _verify_sample_regions(
    item: ScoredSample,
    *,
    proposal_mode: CandidateRegionProposalMode,
    top_k: int,
) -> tuple[list[CandidateRegion], list[MicrographVerificationResult]]:
    if item.edge_scores is None or item.communication_indices is None:
        return [], []
    communication_scores = np.asarray(
        [item.edge_scores[index] for index in item.communication_indices],
        dtype=float,
    )
    if communication_scores.size == 0:
        return [], []
    candidates = propose_candidate_regions(
        item.sample,
        item.communication_indices,
        communication_scores,
        proposal_mode=proposal_mode,
        top_k=top_k,
        slice_count=TEMPORAL_SLICE_COUNT,
    )
    verifier_results = [
        verify_candidate_region(
            item.sample,
            candidate,
            item.communication_indices,
            communication_scores,
            slice_count=TEMPORAL_SLICE_COUNT,
        )
        for candidate in candidates
    ]
    return candidates, verifier_results


def _relabel_verification_results(
    verification_results: list[MicrographVerificationResult],
    *,
    score_threshold: float,
) -> list[MicrographVerificationResult]:
    return [
        MicrographVerificationResult(
            candidate_region_id=item.candidate_region_id,
            proposal_mode=item.proposal_mode,
            verifier_mode=item.verifier_mode,
            micrograph_edge_count=item.micrograph_edge_count,
            micrograph_node_count=item.micrograph_node_count,
            micrograph_slice_count=item.micrograph_slice_count,
            micrograph_score=item.micrograph_score,
            micrograph_consistency_score=item.micrograph_consistency_score,
            micrograph_density_score=item.micrograph_density_score,
            micrograph_temporal_persistence_score=item.micrograph_temporal_persistence_score,
            micrograph_decision=(item.micrograph_score >= score_threshold),
        )
        for item in verification_results
    ]


def _aggregate_two_stage_samples(
    scored_samples: list[ScoredSample],
    *,
    proposal_mode: CandidateRegionProposalMode,
    final_decision_mode: FinalDecisionMode,
    top_k: int,
    verifier_threshold: float,
    graph_threshold: float | None = None,
) -> tuple[list[dict[str, object]], list[float]]:
    graph_rows: list[dict[str, object]] = []
    graph_scores: list[float] = []
    for item in scored_samples:
        candidates, verification_results = _verify_sample_regions(
            item,
            proposal_mode=proposal_mode,
            top_k=top_k,
        )
        relabeled_results = _relabel_verification_results(
            verification_results,
            score_threshold=verifier_threshold,
        )
        selected_regions = [
            candidate
            for candidate, result in zip(candidates, relabeled_results, strict=True)
            if result.micrograph_decision
        ]
        communication_edge_count = len(item.communication_indices or [])
        final_decision = aggregate_micrograph_decisions(
            relabeled_results,
            selected_regions,
            final_decision_mode=final_decision_mode,
            total_edge_count=communication_edge_count,
            score_threshold=verifier_threshold if graph_threshold is None else graph_threshold,
            evidence_threshold=verifier_threshold,
        )
        graph_scores.append(final_decision.decision_score)
        graph_rows.append(
            {
                "graph_score": final_decision.decision_score,
                "prediction": (
                    int(final_decision.decision_score >= graph_threshold and final_decision.positive_evidence_count >= 1)
                    if graph_threshold is not None
                    else 0
                ),
                "candidate_region_count": len(candidates),
                "selected_region_count": final_decision.selected_region_count,
                "selected_region_coverage": final_decision.selected_region_coverage,
                "positive_evidence_count": final_decision.positive_evidence_count,
                "single_edge_region_ratio": (
                    float(sum(candidate.candidate_edge_count == 1 for candidate in candidates) / len(candidates))
                    if candidates
                    else 0.0
                ),
                "mean_candidate_time_span": (
                    float(np.mean([candidate.candidate_time_span for candidate in candidates]))
                    if candidates
                    else 0.0
                ),
            }
        )
    return graph_rows, graph_scores


def _build_candidate_region_diagnosis_rows(
    scored_samples: list[ScoredSample],
    *,
    proposal_mode: CandidateRegionProposalMode,
    top_k: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in scored_samples:
        candidates, _verification_results = _verify_sample_regions(
            item,
            proposal_mode=proposal_mode,
            top_k=top_k,
        )
        for candidate in candidates:
            rows.append(
                {
                    "scenario_id": item.sample.scenario_id,
                    "extraction_mode": item.sample.extraction_mode,
                    "graph_id": f"{item.sample.window_index}:{item.sample.group_key}",
                    "candidate_region_id": candidate.candidate_region_id,
                    "seed_edge_count": candidate.seed_edge_count,
                    "candidate_edge_count": candidate.candidate_edge_count,
                    "candidate_node_count": candidate.candidate_node_count,
                    "candidate_time_span": candidate.candidate_time_span,
                    "candidate_src_count": candidate.candidate_src_count,
                    "candidate_dst_count": candidate.candidate_dst_count,
                    "candidate_score_seed": candidate.candidate_score_seed,
                    "candidate_score_mean": candidate.candidate_score_mean,
                    "candidate_score_max": candidate.candidate_score_max,
                    "repeated_endpoint_count": candidate.repeated_endpoint_count,
                    "repeated_slice_support_count": candidate.repeated_slice_support_count,
                    "support_cluster_density": candidate.support_cluster_density,
                    "label_type": item.sample.label,
                }
            )
    return rows


def _build_micrograph_diagnosis_rows(
    scored_samples: list[ScoredSample],
    *,
    proposal_mode: CandidateRegionProposalMode,
    top_k: int,
    verifier_threshold: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in scored_samples:
        candidates, verification_results = _verify_sample_regions(
            item,
            proposal_mode=proposal_mode,
            top_k=top_k,
        )
        relabeled_results = _relabel_verification_results(
            verification_results,
            score_threshold=verifier_threshold,
        )
        for candidate, result in zip(candidates, relabeled_results, strict=True):
            rows.append(
                {
                    "scenario_id": item.sample.scenario_id,
                    "extraction_mode": item.sample.extraction_mode,
                    "graph_id": f"{item.sample.window_index}:{item.sample.group_key}",
                    "candidate_region_id": candidate.candidate_region_id,
                    "micrograph_edge_count": result.micrograph_edge_count,
                    "micrograph_node_count": result.micrograph_node_count,
                    "micrograph_slice_count": result.micrograph_slice_count,
                    "micrograph_score": result.micrograph_score,
                    "micrograph_consistency_score": result.micrograph_consistency_score,
                    "micrograph_density_score": result.micrograph_density_score,
                    "micrograph_temporal_persistence_score": result.micrograph_temporal_persistence_score,
                    "micrograph_decision": result.micrograph_decision,
                    "label_type": item.sample.label,
                }
            )
    return rows


def _evaluate_two_stage_profiles(
    *,
    evaluation_mode: str,
    scenario_id: str,
    train_graphs: list[object],
    calib_graphs: list[object],
    test_samples: list[ScenarioWindowSample],
    unknown_samples: list[ScenarioWindowSample],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    model, preprocessor = _train_model(train_graphs, model_name="edge_temporal_binary_v2")
    rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    micrograph_rows: list[dict[str, object]] = []
    for profile in default_edge_calibration_profiles():
        calib_scored = _score_samples(
            model,
            [ScenarioWindowSample(scenario_id, -1, "calib", "benign", graph) for graph in calib_graphs],
            preprocessor,
            model_name="edge_temporal_binary_v2",
            top_k=profile.top_k,
        )
        if not calib_scored:
            calib_scored = _score_samples(
                model,
                [ScenarioWindowSample(scenario_id, -1, "train", "benign", graph) for graph in train_graphs],
                preprocessor,
                model_name="edge_temporal_binary_v2",
                top_k=profile.top_k,
            )
        test_scored = _score_samples(
            model,
            test_samples,
            preprocessor,
            model_name="edge_temporal_binary_v2",
            top_k=profile.top_k,
        )
        unknown_scored = _score_samples(
            model,
            unknown_samples,
            preprocessor,
            model_name="edge_temporal_binary_v2",
            top_k=profile.top_k,
        )
        for proposal_mode in TWO_STAGE_PROPOSAL_MODES:
            benign_verifier_scores: list[float] = []
            for item in calib_scored:
                _candidates, verification_results = _verify_sample_regions(
                    item,
                    proposal_mode=proposal_mode,
                    top_k=_proposal_top_k(profile),
                )
                benign_verifier_scores.extend(result.micrograph_score for result in verification_results)
            verifier_threshold = float(np.percentile(benign_verifier_scores, profile.percentile)) if benign_verifier_scores else 1.0
            for final_decision_mode in TWO_STAGE_FINAL_DECISION_MODES:
                calib_graph_rows, calib_graph_scores = _aggregate_two_stage_samples(
                    calib_scored,
                    proposal_mode=proposal_mode,
                    final_decision_mode=final_decision_mode,
                    top_k=_proposal_top_k(profile),
                    verifier_threshold=verifier_threshold,
                    graph_threshold=None,
                )
                graph_threshold = float(np.percentile(np.asarray(calib_graph_scores, dtype=float), profile.percentile)) if calib_graph_scores else 0.0
                test_graph_rows, test_graph_scores = _aggregate_two_stage_samples(
                    test_scored,
                    proposal_mode=proposal_mode,
                    final_decision_mode=final_decision_mode,
                    top_k=_proposal_top_k(profile),
                    verifier_threshold=verifier_threshold,
                    graph_threshold=graph_threshold,
                )
                unknown_graph_rows, unknown_graph_scores = _aggregate_two_stage_samples(
                    unknown_scored,
                    proposal_mode=proposal_mode,
                    final_decision_mode=final_decision_mode,
                    top_k=_proposal_top_k(profile),
                    verifier_threshold=verifier_threshold,
                    graph_threshold=graph_threshold,
                )
                y_true = [1 if item.sample.label == "malicious" else 0 for item in test_scored]
                y_score = [row["graph_score"] for row in test_graph_rows]
                y_pred = [int(row["prediction"]) for row in test_graph_rows]
                metrics = _binary_metrics(y_true, y_score, y_pred)
                rows.append(
                    {
                        "evaluation_mode": evaluation_mode,
                        "scenario_id": scenario_id,
                        "model_name": "edge_temporal_micrograph_v1",
                        "calibration_profile": profile.name,
                        "percentile_setting": profile.percentile,
                        "top_k_setting": profile.top_k,
                        "suppression_enabled": False,
                        "support_summary_mode": "two_stage_micrograph",
                        "extraction_mode": test_samples[0].extraction_mode if test_samples else "per_src_ip_within_window",
                        "region_proposal_mode": proposal_mode,
                        "verifier_mode": "micrograph_consistency_v1",
                        "final_decision_mode": final_decision_mode,
                        "concentration_threshold_setting": None,
                        "component_ratio_setting": None,
                        "neighborhood_ratio_setting": None,
                        "local_density_threshold": verifier_threshold,
                        "neighborhood_persistence_threshold": None,
                        "temporal_consistency_threshold": None,
                        "candidate_region_count_mean": float(np.mean([row["candidate_region_count"] for row in test_graph_rows])) if test_graph_rows else 0.0,
                        "selected_region_count_mean": float(np.mean([row["selected_region_count"] for row in test_graph_rows])) if test_graph_rows else 0.0,
                        "selected_region_coverage_mean": float(np.mean([row["selected_region_coverage"] for row in test_graph_rows])) if test_graph_rows else 0.0,
                        "single_edge_region_ratio": float(np.mean([row["single_edge_region_ratio"] for row in test_graph_rows])) if test_graph_rows else 0.0,
                        "mean_candidate_time_span": float(np.mean([row["mean_candidate_time_span"] for row in test_graph_rows])) if test_graph_rows else 0.0,
                        "train_benign_graphs": len(train_graphs),
                        "calib_benign_graphs": len(calib_graphs),
                        "test_benign_graphs": sum(item.sample.label == "benign" for item in test_scored),
                        "test_malicious_graphs": sum(item.sample.label == "malicious" for item in test_scored),
                        "test_unknown_graphs": len(unknown_scored),
                        "threshold": graph_threshold,
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "fpr": metrics["fpr"],
                        "roc_auc": metrics["roc_auc"],
                        "pr_auc": metrics["pr_auc"],
                        "background_hit_ratio": (
                            float(sum(int(row["prediction"]) for row in unknown_graph_rows) / len(unknown_graph_rows))
                            if unknown_graph_rows
                            else 0.0
                        ),
                        "unknown_count": len(unknown_graph_scores),
                        "unknown_score_mean": float(np.mean(unknown_graph_scores)) if unknown_graph_scores else 0.0,
                        "unknown_score_median": float(np.median(unknown_graph_scores)) if unknown_graph_scores else 0.0,
                    }
                )
            if profile.name == "heldout_q95_top1":
                candidate_rows.extend(
                    _build_candidate_region_diagnosis_rows(
                        test_scored + unknown_scored,
                        proposal_mode=proposal_mode,
                        top_k=_proposal_top_k(profile),
                    )
                )
                micrograph_rows.extend(
                    _build_micrograph_diagnosis_rows(
                        test_scored + unknown_scored,
                        proposal_mode=proposal_mode,
                        top_k=_proposal_top_k(profile),
                        verifier_threshold=verifier_threshold,
                    )
                )
    return rows, candidate_rows, micrograph_rows


def _evaluate_scenario(
    prepared: PreparedScenario,
    *,
    evaluation_mode: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    if not prepared.train_graphs or not prepared.test_samples:
        return [], [], [], []
    rows = _evaluate_baseline_profiles(
        evaluation_mode=evaluation_mode,
        scenario_id=prepared.scenario_id,
        train_graphs=prepared.train_graphs,
        calib_graphs=prepared.calib_graphs,
        test_samples=prepared.test_samples,
        unknown_samples=prepared.unknown_samples,
    )
    edge_rows, diagnosis_rows = _evaluate_edge_profiles(
        evaluation_mode=evaluation_mode,
        scenario_id=prepared.scenario_id,
        train_graphs=prepared.train_graphs,
        calib_graphs=prepared.calib_graphs,
        test_samples=prepared.test_samples,
        unknown_samples=prepared.unknown_samples,
    )
    rows.extend(edge_rows)
    nuisance_rows, nuisance_diagnosis_rows, nuisance_benchmark_rows = _evaluate_nuisance_aware_edge_profiles(
        evaluation_mode=evaluation_mode,
        scenario_id=prepared.scenario_id,
        train_graphs=prepared.train_graphs,
        calib_graphs=prepared.calib_graphs,
        test_samples=prepared.test_samples,
        unknown_samples=prepared.unknown_samples,
    )
    rows.extend(nuisance_rows)
    return rows, diagnosis_rows, nuisance_diagnosis_rows, nuisance_benchmark_rows


def _merge_prepared_scenarios(prepared_scenarios: list[PreparedScenario]) -> PreparedScenario:
    return PreparedScenario(
        scenario_id="merged_48_49_52",
        train_graphs=[graph for item in prepared_scenarios for graph in item.train_graphs],
        calib_graphs=[graph for item in prepared_scenarios for graph in item.calib_graphs],
        test_samples=[sample for item in prepared_scenarios for sample in item.test_samples],
        unknown_samples=[sample for item in prepared_scenarios for sample in item.unknown_samples],
    )


def _graph_extraction_summary_row(
    scenario_id: str,
    extraction_mode: GraphExtractionMode,
    primary_samples: list[ScenarioWindowSample],
    unknown_samples: list[ScenarioWindowSample],
) -> dict[str, object]:
    all_samples = primary_samples + unknown_samples
    return {
        "scenario_id": scenario_id,
        "extraction_mode": extraction_mode,
        "graph_count": len(all_samples),
        "avg_graph_size": float(np.mean([sample.graph.edge_count for sample in all_samples])) if all_samples else 0.0,
        "avg_edge_count": float(np.mean([sample.graph.edge_count for sample in all_samples])) if all_samples else 0.0,
        "avg_node_count": float(np.mean([sample.graph.node_count for sample in all_samples])) if all_samples else 0.0,
        "benign_graph_count": sum(sample.label == "benign" for sample in primary_samples),
        "malicious_graph_count": sum(sample.label == "malicious" for sample in primary_samples),
        "unknown_graph_count": len(unknown_samples),
        "candidate_ready_graph_count": sum(sample.graph.stats.communication_edge_count >= 2 for sample in all_samples),
    }


def _proposal_quality_diagnosis_rows(
    candidate_rows: list[dict[str, object]],
    micrograph_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    keys = sorted(
        {
            (
                str(row["scenario_id"]),
                str(row.get("extraction_mode", "per_src_ip_within_window")),
                str(row["candidate_region_id"]).split(":")[0],
            )
            for row in candidate_rows
        }
    )
    rows: list[dict[str, object]] = []
    for scenario_id, extraction_mode, proposal_mode in keys:
        candidate_subset = [
            row for row in candidate_rows
            if str(row["scenario_id"]) == scenario_id
            and str(row.get("extraction_mode", "per_src_ip_within_window")) == extraction_mode
            and str(row["candidate_region_id"]).startswith(proposal_mode)
        ]
        micro_subset = [
            row for row in micrograph_rows
            if str(row["scenario_id"]) == scenario_id
            and str(row.get("extraction_mode", "per_src_ip_within_window")) == extraction_mode
            and str(row["candidate_region_id"]).startswith(proposal_mode)
        ]
        mal_candidate = [row for row in candidate_subset if row["label_type"] == "malicious"]
        unk_candidate = [row for row in candidate_subset if row["label_type"] == "unknown"]
        mal_micro = [row for row in micro_subset if row["label_type"] == "malicious"]
        unk_micro = [row for row in micro_subset if row["label_type"] == "unknown"]
        single_edge_region_ratio = (
            float(sum(float(row["candidate_edge_count"]) == 1.0 for row in candidate_subset) / len(candidate_subset))
            if candidate_subset
            else 0.0
        )
        def _mean(subset: list[dict[str, object]], field: str) -> float:
            return float(np.mean([float(row[field]) for row in subset])) if subset else 0.0
        rows.append(
            {
                "scenario_id": scenario_id,
                "extraction_mode": extraction_mode,
                "region_proposal_mode": proposal_mode,
                "malicious_candidate_edge_mean": _mean(mal_candidate, "candidate_edge_count"),
                "unknown_candidate_edge_mean": _mean(unk_candidate, "candidate_edge_count"),
                "malicious_candidate_time_span_mean": _mean(mal_candidate, "candidate_time_span"),
                "unknown_candidate_time_span_mean": _mean(unk_candidate, "candidate_time_span"),
                "malicious_repeated_endpoint_mean": _mean(mal_candidate, "repeated_endpoint_count"),
                "unknown_repeated_endpoint_mean": _mean(unk_candidate, "repeated_endpoint_count"),
                "malicious_support_cluster_density_mean": _mean(mal_candidate, "support_cluster_density"),
                "unknown_support_cluster_density_mean": _mean(unk_candidate, "support_cluster_density"),
                "malicious_mean_proposal_score": _mean(mal_candidate, "candidate_score_mean"),
                "unknown_mean_proposal_score": _mean(unk_candidate, "candidate_score_mean"),
                "proposal_score_gap": _mean(mal_candidate, "candidate_score_mean") - _mean(unk_candidate, "candidate_score_mean"),
                "single_edge_region_ratio": single_edge_region_ratio,
                "malicious_micrograph_mean_score": _mean(mal_micro, "micrograph_score"),
                "unknown_micrograph_mean_score": _mean(unk_micro, "micrograph_score"),
                "micrograph_score_gap": _mean(mal_micro, "micrograph_score") - _mean(unk_micro, "micrograph_score"),
            }
        )
    return rows


def _score_sample_episodes(
    item: ScoredSample,
    *,
    proposal_mode: EpisodeProposalMode,
    top_k: int,
) -> tuple[list[Episode], object | None, list[NuisanceAwareEpisodeScore]]:
    if item.edge_scores is None or item.communication_indices is None:
        return [], None, []
    communication_scores = np.asarray(
        [item.edge_scores[index] for index in item.communication_indices],
        dtype=float,
    )
    if communication_scores.size == 0:
        return [], None, []
    episodes = propose_episodes(
        item.sample,
        item.communication_indices,
        communication_scores,
        proposal_mode=proposal_mode,
        top_k=top_k,
        slice_count=TEMPORAL_SLICE_COUNT,
    )
    episode_graph = build_episode_graph(item.sample, episodes)
    episode_scores = [score_episode_nuisance_aware(episode, episode_graph) for episode in episodes]
    return episodes, episode_graph, episode_scores


def _score_sample_sessionized_episodes(
    item: ScoredSample,
    *,
    stitching_mode: EpisodeStitchingMode,
) -> tuple[list[Episode], object | None, list[NuisanceAwareEpisodeScore]]:
    if item.edge_scores is None or item.communication_indices is None:
        return [], None, []
    communication_scores = np.asarray(
        [item.edge_scores[index] for index in item.communication_indices],
        dtype=float,
    )
    if communication_scores.size == 0:
        return [], None, []
    episodes = sessionize_episodes(
        item.sample,
        item.communication_indices,
        communication_scores,
        stitching_mode=stitching_mode,
        slice_count=TEMPORAL_SLICE_COUNT,
    )
    episode_graph = build_episode_graph(item.sample, episodes)
    episode_scores = [score_episode_nuisance_aware(episode, episode_graph) for episode in episodes]
    return episodes, episode_graph, episode_scores


def _aggregate_episode_samples(
    scored_samples: list[ScoredSample],
    *,
    proposal_mode: EpisodeProposalMode,
    final_decision_mode: EpisodeDecisionMode,
    top_k: int,
    calibration,
    graph_threshold: float | None = None,
) -> tuple[list[dict[str, object]], list[float]]:
    graph_rows: list[dict[str, object]] = []
    graph_scores: list[float] = []
    for item in scored_samples:
        episodes, _episode_graph, episode_scores = _score_sample_episodes(
            item,
            proposal_mode=proposal_mode,
            top_k=top_k,
        )
        relabeled_scores = relabel_episode_scores(episode_scores, calibration)
        final_decision = aggregate_episode_graph_decision(
            relabeled_scores,
            episodes,
            final_decision_mode=final_decision_mode,
            total_edge_count=len(item.communication_indices or []),
            graph_threshold=calibration.consistency_threshold if graph_threshold is None else graph_threshold,
        )
        graph_scores.append(final_decision.decision_score)
        graph_rows.append(
            {
                "graph_score": final_decision.decision_score,
                "prediction": (
                    int(final_decision.decision_score >= graph_threshold and final_decision.positive_evidence_count >= 1)
                    if graph_threshold is not None
                    else 0
                ),
                "episode_count": len(episodes),
                "selected_episode_count": final_decision.selected_episode_count,
                "selected_episode_coverage": final_decision.selected_episode_coverage,
                "positive_evidence_count": final_decision.positive_evidence_count,
                "single_edge_region_ratio": (
                    float(sum(episode.edge_count == 1 for episode in episodes) / len(episodes))
                    if episodes
                    else 0.0
                ),
                "single_flow_episode_ratio": (
                    float(sum(episode.flow_count == 1 for episode in episodes) / len(episodes))
                    if episodes
                    else 0.0
                ),
                "short_episode_ratio": (
                    float(sum(episode.duration <= 1.0 for episode in episodes) / len(episodes))
                    if episodes
                    else 0.0
                ),
                "mean_candidate_time_span": (
                    float(np.mean([episode.episode_time_span for episode in episodes]))
                    if episodes
                    else 0.0
                ),
                "mean_episode_flow_count": (
                    float(np.mean([episode.flow_count for episode in episodes]))
                    if episodes
                    else 0.0
                ),
                "nuisance_rejection_rate": final_decision.nuisance_rejection_rate,
            }
        )
    return graph_rows, graph_scores


def _aggregate_sessionized_episode_samples(
    scored_samples: list[ScoredSample],
    *,
    stitching_mode: EpisodeStitchingMode,
    final_decision_mode: EpisodeDecisionMode,
    calibration,
    graph_threshold: float | None = None,
) -> tuple[list[dict[str, object]], list[float]]:
    graph_rows: list[dict[str, object]] = []
    graph_scores: list[float] = []
    for item in scored_samples:
        episodes, _episode_graph, episode_scores = _score_sample_sessionized_episodes(
            item,
            stitching_mode=stitching_mode,
        )
        relabeled_scores = relabel_episode_scores(episode_scores, calibration)
        final_decision = aggregate_episode_graph_decision(
            relabeled_scores,
            episodes,
            final_decision_mode=final_decision_mode,
            total_edge_count=len(item.communication_indices or []),
            graph_threshold=calibration.consistency_threshold if graph_threshold is None else graph_threshold,
        )
        graph_scores.append(final_decision.decision_score)
        graph_rows.append(
            {
                "graph_score": final_decision.decision_score,
                "prediction": (
                    int(final_decision.decision_score >= graph_threshold and final_decision.positive_evidence_count >= 1)
                    if graph_threshold is not None
                    else 0
                ),
                "episode_count": len(episodes),
                "selected_episode_count": final_decision.selected_episode_count,
                "selected_episode_coverage": final_decision.selected_episode_coverage,
                "positive_evidence_count": final_decision.positive_evidence_count,
                "single_edge_region_ratio": (
                    float(sum(episode.edge_count == 1 for episode in episodes) / len(episodes))
                    if episodes
                    else 0.0
                ),
                "single_flow_episode_ratio": (
                    float(sum(episode.flow_count == 1 for episode in episodes) / len(episodes))
                    if episodes
                    else 0.0
                ),
                "short_episode_ratio": (
                    float(sum(episode.duration <= 1.0 for episode in episodes) / len(episodes))
                    if episodes
                    else 0.0
                ),
                "mean_candidate_time_span": (
                    float(np.mean([episode.episode_time_span for episode in episodes]))
                    if episodes
                    else 0.0
                ),
                "mean_episode_flow_count": (
                    float(np.mean([episode.flow_count for episode in episodes]))
                    if episodes
                    else 0.0
                ),
                "nuisance_rejection_rate": final_decision.nuisance_rejection_rate,
            }
        )
    return graph_rows, graph_scores


def _build_episode_construction_rows(
    scored_samples: list[ScoredSample],
    *,
    proposal_mode: EpisodeProposalMode,
    top_k: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in scored_samples:
        episodes, episode_graph, _episode_scores = _score_sample_episodes(
            item,
            proposal_mode=proposal_mode,
            top_k=top_k,
        )
        if episode_graph is None:
            continue
        for episode in episodes:
            rows.append(
                {
                    "scenario_id": item.sample.scenario_id,
                    "graph_id": f"{item.sample.window_index}:{item.sample.group_key}",
                    "episode_id": episode.episode_id,
                    "proposal_mode": proposal_mode,
                    "episode_route_version": "proposal_v1",
                    "stitching_mode": proposal_mode,
                    "start_time": episode.start_time.isoformat() if episode.start_time is not None else "",
                    "end_time": episode.end_time.isoformat() if episode.end_time is not None else "",
                    "involved_flows": len(episode.involved_flow_ids),
                    "involved_endpoints": len(episode.involved_endpoints),
                    "repeated_pair_count": episode.repeated_pair_count,
                    "burst_persistence": episode.burst_persistence,
                    "support_cluster_density": episode.support_cluster_density,
                    "nuisance_likelihood": episode.nuisance_likelihood,
                    "episode_time_span": episode.episode_time_span,
                    "episode_edge_count": episode.edge_count,
                    "gap_count": episode.gap_count,
                    "continuity_span": episode.continuity_span,
                    "protocol_consistency_score": episode.protocol_consistency_score,
                    "direction_pattern_consistency": episode.direction_pattern_consistency,
                    "merged_flow_count": episode.merged_flow_count,
                    "episode_graph_episode_count": episode_graph.episode_count,
                    "episode_graph_endpoint_count": episode_graph.endpoint_count,
                    "episode_graph_temporal_adjacency_count": episode_graph.temporal_adjacency_edge_count,
                    "episode_graph_similarity_count": (
                        episode_graph.repeated_pair_similarity_edge_count + episode_graph.support_cluster_similarity_edge_count
                    ),
                    "label_type": item.sample.label,
                }
            )
    return rows


def _build_nuisance_diagnosis_rows(
    scored_samples: list[ScoredSample],
    *,
    proposal_mode: EpisodeProposalMode,
    top_k: int,
    calibration,
    final_decision_mode: EpisodeDecisionMode,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in scored_samples:
        episodes, _episode_graph, episode_scores = _score_sample_episodes(
            item,
            proposal_mode=proposal_mode,
            top_k=top_k,
        )
        relabeled_scores = relabel_episode_scores(episode_scores, calibration)
        final_decision = aggregate_episode_graph_decision(
            relabeled_scores,
            episodes,
            final_decision_mode=final_decision_mode,
            total_edge_count=len(item.communication_indices or []),
            graph_threshold=calibration.consistency_threshold,
        )
        for episode, score in zip(episodes, relabeled_scores, strict=True):
            suspected_failure_mode = "none"
            if item.sample.label == "unknown" and score.episode_decision:
                suspected_failure_mode = "nuisance_boundary_overlap"
            elif item.sample.label == "malicious" and not score.episode_decision:
                if score.nuisance_like:
                    suspected_failure_mode = "malicious_episode_rejected_as_nuisance"
                elif score.anomaly_score < calibration.anomaly_threshold:
                    suspected_failure_mode = "weak_episode_anomaly"
                else:
                    suspected_failure_mode = "weak_episode_consistency"
            rows.append(
                {
                    "scenario_id": item.sample.scenario_id,
                    "graph_id": f"{item.sample.window_index}:{item.sample.group_key}",
                    "episode_id": episode.episode_id,
                    "episode_route_version": "proposal_v1",
                    "stitching_mode": proposal_mode,
                    "label_type": item.sample.label,
                    "episode_score": score.malicious_consistency_score,
                    "nuisance_score": score.nuisance_score,
                    "anomaly_score": score.anomaly_score,
                    "repeated_pair_count": episode.repeated_pair_count,
                    "burst_persistence": episode.burst_persistence,
                    "support_cluster_density": episode.support_cluster_density,
                    "episode_time_span": episode.episode_time_span,
                    "episode_endpoint_span": episode.endpoint_span,
                    "protocol_consistency_score": episode.protocol_consistency_score,
                    "direction_pattern_consistency": episode.direction_pattern_consistency,
                    "final_episode_decision": score.episode_decision,
                    "final_graph_decision": final_decision.is_positive,
                    "suspected_failure_mode": suspected_failure_mode,
                }
            )
    return rows


def _build_sessionized_episode_construction_rows(
    scored_samples: list[ScoredSample],
    *,
    stitching_mode: EpisodeStitchingMode,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in scored_samples:
        episodes, episode_graph, _episode_scores = _score_sample_sessionized_episodes(
            item,
            stitching_mode=stitching_mode,
        )
        if episode_graph is None:
            continue
        for episode in episodes:
            rows.append(
                {
                    "scenario_id": item.sample.scenario_id,
                    "graph_id": f"{item.sample.window_index}:{item.sample.group_key}",
                    "episode_id": episode.episode_id,
                    "proposal_mode": episode.proposal_mode,
                    "episode_route_version": "sessionized_v2",
                    "stitching_mode": stitching_mode,
                    "start_time": episode.start_time.isoformat() if episode.start_time is not None else "",
                    "end_time": episode.end_time.isoformat() if episode.end_time is not None else "",
                    "involved_flows": len(episode.involved_flow_ids),
                    "involved_endpoints": len(episode.involved_endpoints),
                    "repeated_pair_count": episode.repeated_pair_count,
                    "burst_persistence": episode.burst_persistence,
                    "support_cluster_density": episode.support_cluster_density,
                    "nuisance_likelihood": episode.nuisance_likelihood,
                    "episode_time_span": episode.episode_time_span,
                    "episode_edge_count": episode.edge_count,
                    "gap_count": episode.gap_count,
                    "continuity_span": episode.continuity_span,
                    "protocol_consistency_score": episode.protocol_consistency_score,
                    "direction_pattern_consistency": episode.direction_pattern_consistency,
                    "merged_flow_count": episode.merged_flow_count,
                    "episode_graph_episode_count": episode_graph.episode_count,
                    "episode_graph_endpoint_count": episode_graph.endpoint_count,
                    "episode_graph_temporal_adjacency_count": episode_graph.temporal_adjacency_edge_count,
                    "episode_graph_similarity_count": (
                        episode_graph.repeated_pair_similarity_edge_count + episode_graph.support_cluster_similarity_edge_count
                    ),
                    "label_type": item.sample.label,
                }
            )
    return rows


def _build_sessionized_nuisance_diagnosis_rows(
    scored_samples: list[ScoredSample],
    *,
    stitching_mode: EpisodeStitchingMode,
    calibration,
    final_decision_mode: EpisodeDecisionMode,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in scored_samples:
        episodes, _episode_graph, episode_scores = _score_sample_sessionized_episodes(
            item,
            stitching_mode=stitching_mode,
        )
        relabeled_scores = relabel_episode_scores(episode_scores, calibration)
        final_decision = aggregate_episode_graph_decision(
            relabeled_scores,
            episodes,
            final_decision_mode=final_decision_mode,
            total_edge_count=len(item.communication_indices or []),
            graph_threshold=calibration.consistency_threshold,
        )
        for episode, score in zip(episodes, relabeled_scores, strict=True):
            suspected_failure_mode = "none"
            if item.sample.label == "unknown" and score.episode_decision:
                suspected_failure_mode = "nuisance_boundary_overlap"
            elif item.sample.label == "malicious" and not score.episode_decision:
                if score.nuisance_like:
                    suspected_failure_mode = "malicious_episode_rejected_as_nuisance"
                elif score.anomaly_score < calibration.anomaly_threshold:
                    suspected_failure_mode = "weak_episode_anomaly"
                else:
                    suspected_failure_mode = "weak_episode_consistency"
            rows.append(
                {
                    "scenario_id": item.sample.scenario_id,
                    "graph_id": f"{item.sample.window_index}:{item.sample.group_key}",
                    "episode_id": episode.episode_id,
                    "episode_route_version": "sessionized_v2",
                    "stitching_mode": stitching_mode,
                    "label_type": item.sample.label,
                    "episode_score": score.malicious_consistency_score,
                    "nuisance_score": score.nuisance_score,
                    "anomaly_score": score.anomaly_score,
                    "repeated_pair_count": episode.repeated_pair_count,
                    "burst_persistence": episode.burst_persistence,
                    "support_cluster_density": episode.support_cluster_density,
                    "episode_time_span": episode.episode_time_span,
                    "episode_endpoint_span": episode.endpoint_span,
                    "protocol_consistency_score": episode.protocol_consistency_score,
                    "direction_pattern_consistency": episode.direction_pattern_consistency,
                    "final_episode_decision": score.episode_decision,
                    "final_graph_decision": final_decision.is_positive,
                    "suspected_failure_mode": suspected_failure_mode,
                }
            )
    return rows


def _episode_quality_diagnosis_rows(
    construction_rows: list[dict[str, object]],
    nuisance_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    keys = sorted(
        {
            (
                str(row["scenario_id"]),
                str(row.get("episode_route_version", "proposal_v1")),
                str(row.get("stitching_mode", row.get("proposal_mode", "unknown"))),
            )
            for row in construction_rows
        }
    )
    rows: list[dict[str, object]] = []
    for scenario_id, episode_route_version, stitching_mode in keys:
        construction_subset = [
            row for row in construction_rows
            if str(row["scenario_id"]) == scenario_id
            and str(row.get("episode_route_version", "proposal_v1")) == episode_route_version
            and str(row.get("stitching_mode", row.get("proposal_mode", "unknown"))) == stitching_mode
        ]
        nuisance_subset = [
            row for row in nuisance_rows
            if str(row["scenario_id"]) == scenario_id
            and str(row.get("episode_route_version", "proposal_v1")) == episode_route_version
            and str(row.get("stitching_mode", "unknown")) == stitching_mode
        ]
        mal_construction = [row for row in construction_subset if row["label_type"] == "malicious"]
        unk_construction = [row for row in construction_subset if row["label_type"] == "unknown"]
        benign_construction = [row for row in construction_subset if row["label_type"] == "benign"]
        mal_nuisance = [row for row in nuisance_subset if row["label_type"] == "malicious"]
        unk_nuisance = [row for row in nuisance_subset if row["label_type"] == "unknown"]

        def _mean(subset: list[dict[str, object]], field: str) -> float:
            return float(np.mean([float(row[field]) for row in subset])) if subset else 0.0

        all_episode_scores = [float(row["episode_score"]) for row in nuisance_subset]
        mal_scores = [float(row["episode_score"]) for row in mal_nuisance]
        unk_scores = [float(row["episode_score"]) for row in unk_nuisance]
        overlap_ratio = 0.0
        if mal_scores and unk_scores:
            score_floor = min(np.mean(mal_scores), np.mean(unk_scores))
            score_ceiling = max(np.mean(mal_scores), np.mean(unk_scores))
            overlap_ratio = float(
                sum(score_floor <= score <= score_ceiling for score in all_episode_scores)
                / max(len(all_episode_scores), 1)
            )
        rows.append(
            {
                "scenario_id": scenario_id,
                "episode_route_version": episode_route_version,
                "proposal_mode": construction_subset[0].get("proposal_mode", "") if construction_subset else "",
                "stitching_mode": stitching_mode,
                "total_episode_count": len(construction_subset),
                "benign_episode_count": len(benign_construction),
                "malicious_episode_count": len(mal_construction),
                "unknown_episode_count": len(unk_construction),
                "malicious_episode_coverage": 0.0,
                "malicious_episode_edge_mean": _mean(mal_construction, "episode_edge_count"),
                "unknown_episode_edge_mean": _mean(unk_construction, "episode_edge_count"),
                "malicious_episode_time_span_mean": _mean(mal_construction, "episode_time_span"),
                "unknown_episode_time_span_mean": _mean(unk_construction, "episode_time_span"),
                "mean_episode_duration": _mean(construction_subset, "episode_time_span"),
                "mean_episode_flow_count": _mean(construction_subset, "involved_flows"),
                "malicious_repeated_pair_mean": _mean(mal_construction, "repeated_pair_count"),
                "unknown_repeated_pair_mean": _mean(unk_construction, "repeated_pair_count"),
                "mean_repeated_pair_count": _mean(construction_subset, "repeated_pair_count"),
                "mean_burst_persistence": _mean(construction_subset, "burst_persistence"),
                "malicious_support_cluster_density_mean": _mean(mal_construction, "support_cluster_density"),
                "unknown_support_cluster_density_mean": _mean(unk_construction, "support_cluster_density"),
                "mean_protocol_consistency_score": _mean(construction_subset, "protocol_consistency_score"),
                "single_edge_region_ratio": (
                    float(sum(float(row["episode_edge_count"]) == 1.0 for row in construction_subset) / len(construction_subset))
                    if construction_subset
                    else 0.0
                ),
                "single_flow_episode_ratio": (
                    float(sum(float(row["involved_flows"]) == 1.0 for row in construction_subset) / len(construction_subset))
                    if construction_subset
                    else 0.0
                ),
                "short_episode_ratio": (
                    float(sum(float(row["episode_time_span"]) <= 1.0 for row in construction_subset) / len(construction_subset))
                    if construction_subset
                    else 0.0
                ),
                "malicious_mean_episode_score": _mean(mal_nuisance, "episode_score"),
                "unknown_mean_episode_score": _mean(unk_nuisance, "episode_score"),
                "episode_score_gap": _mean(mal_nuisance, "episode_score") - _mean(unk_nuisance, "episode_score"),
                "malicious_mean_nuisance_score": _mean(mal_nuisance, "nuisance_score"),
                "unknown_mean_nuisance_score": _mean(unk_nuisance, "nuisance_score"),
                "nuisance_score_gap": _mean(unk_nuisance, "nuisance_score") - _mean(mal_nuisance, "nuisance_score"),
                "malicious_unknown_overlap_ratio": overlap_ratio,
            }
        )
    return rows


def _episode_sessionization_summary_rows(
    construction_rows: list[dict[str, object]],
    benchmark_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    keys = sorted(
        {
            (str(row["scenario_id"]), str(row.get("episode_route_version", "proposal_v1")), str(row.get("stitching_mode", "unknown")))
            for row in construction_rows
        }
    )
    rows: list[dict[str, object]] = []
    for scenario_id, episode_route_version, stitching_mode in keys:
        construction_subset = [
            row for row in construction_rows
            if str(row["scenario_id"]) == scenario_id
            and str(row.get("episode_route_version", "proposal_v1")) == episode_route_version
            and str(row.get("stitching_mode", "unknown")) == stitching_mode
        ]
        benchmark_subset = [
            row for row in benchmark_rows
            if str(row["scenario_id"]) == scenario_id
            and str(row.get("episode_route_version", "")) == episode_route_version
            and str(row.get("episode_stitching_mode", "")) == stitching_mode
        ]
        if not construction_subset:
            continue
        def _mean(rows_: list[dict[str, object]], field: str) -> float:
            return float(np.mean([float(row[field]) for row in rows_])) if rows_ else 0.0
        rows.append(
            {
                "scenario_id": scenario_id,
                "episode_route_version": episode_route_version,
                "stitching_mode": stitching_mode,
                "total_episode_count": len(construction_subset),
                "benign_episode_count": sum(row["label_type"] == "benign" for row in construction_subset),
                "malicious_episode_count": sum(row["label_type"] == "malicious" for row in construction_subset),
                "unknown_episode_count": sum(row["label_type"] == "unknown" for row in construction_subset),
                "malicious_episode_coverage": _mean(benchmark_subset, "malicious_episode_coverage"),
                "single_flow_episode_ratio": (
                    float(sum(float(row["involved_flows"]) == 1.0 for row in construction_subset) / len(construction_subset))
                    if construction_subset else 0.0
                ),
                "short_episode_ratio": (
                    float(sum(float(row["episode_time_span"]) <= 1.0 for row in construction_subset) / len(construction_subset))
                    if construction_subset else 0.0
                ),
                "mean_episode_duration": _mean(construction_subset, "episode_time_span"),
                "mean_episode_flow_count": _mean(construction_subset, "involved_flows"),
                "mean_repeated_pair_count": _mean(construction_subset, "repeated_pair_count"),
                "mean_burst_persistence": _mean(construction_subset, "burst_persistence"),
                "mean_protocol_consistency_score": _mean(construction_subset, "protocol_consistency_score"),
                "nuisance_rejection_rate": _mean(benchmark_subset, "nuisance_rejection_rate"),
                "nuisance_like_false_positive_rate": _mean(benchmark_subset, "nuisance_like_false_positive_rate"),
                "malicious_like_episode_precision": _mean(benchmark_subset, "malicious_like_episode_precision"),
                "episode_to_graph_decision_consistency": _mean(benchmark_subset, "episode_to_graph_decision_consistency"),
            }
        )
    return rows


def _evaluate_episode_graph_profiles(
    *,
    evaluation_mode: str,
    scenario_id: str,
    train_graphs: list[object],
    calib_graphs: list[object],
    test_samples: list[ScenarioWindowSample],
    unknown_samples: list[ScenarioWindowSample],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    model, preprocessor = _train_model(train_graphs, model_name="edge_temporal_binary_v2")
    rows: list[dict[str, object]] = []
    construction_rows: list[dict[str, object]] = []
    nuisance_rows: list[dict[str, object]] = []
    benchmark_rows: list[dict[str, object]] = []
    for profile in default_edge_calibration_profiles():
        if profile.name not in TWO_STAGE_PROFILE_NAMES:
            continue
        calib_scored = _score_samples(
            model,
            [ScenarioWindowSample(scenario_id, -1, "calib", "benign", graph) for graph in calib_graphs],
            preprocessor,
            model_name="edge_temporal_binary_v2",
            top_k=profile.top_k,
        )
        if not calib_scored:
            calib_scored = _score_samples(
                model,
                [ScenarioWindowSample(scenario_id, -1, "train", "benign", graph) for graph in train_graphs],
                preprocessor,
                model_name="edge_temporal_binary_v2",
                top_k=profile.top_k,
            )
        test_scored = _score_samples(
            model,
            test_samples,
            preprocessor,
            model_name="edge_temporal_binary_v2",
            top_k=profile.top_k,
        )
        unknown_scored = _score_samples(
            model,
            unknown_samples,
            preprocessor,
            model_name="edge_temporal_binary_v2",
            top_k=profile.top_k,
        )
        for proposal_mode in EPISODE_PROPOSAL_MODES:
            benign_episode_scores: list[NuisanceAwareEpisodeScore] = []
            nuisance_episode_scores: list[NuisanceAwareEpisodeScore] = []
            for item in calib_scored:
                _episodes, _episode_graph, episode_scores = _score_sample_episodes(
                    item,
                    proposal_mode=proposal_mode,
                    top_k=_proposal_top_k(profile),
                )
                benign_episode_scores.extend(episode_scores)
            for item in unknown_scored:
                _episodes, _episode_graph, episode_scores = _score_sample_episodes(
                    item,
                    proposal_mode=proposal_mode,
                    top_k=_proposal_top_k(profile),
                )
                nuisance_episode_scores.extend(episode_scores)
            calibration = calibrate_nuisance_aware_scores(
                benign_episode_scores,
                nuisance_episode_scores,
                percentile=profile.percentile,
            )
            for final_decision_mode in EPISODE_FINAL_DECISION_MODES:
                calib_graph_rows, calib_graph_scores = _aggregate_episode_samples(
                    calib_scored,
                    proposal_mode=proposal_mode,
                    final_decision_mode=final_decision_mode,
                    top_k=_proposal_top_k(profile),
                    calibration=calibration,
                    graph_threshold=None,
                )
                graph_threshold = float(np.percentile(np.asarray(calib_graph_scores, dtype=float), profile.percentile)) if calib_graph_scores else calibration.consistency_threshold
                test_graph_rows, test_graph_scores = _aggregate_episode_samples(
                    test_scored,
                    proposal_mode=proposal_mode,
                    final_decision_mode=final_decision_mode,
                    top_k=_proposal_top_k(profile),
                    calibration=calibration,
                    graph_threshold=graph_threshold,
                )
                unknown_graph_rows, unknown_graph_scores = _aggregate_episode_samples(
                    unknown_scored,
                    proposal_mode=proposal_mode,
                    final_decision_mode=final_decision_mode,
                    top_k=_proposal_top_k(profile),
                    calibration=calibration,
                    graph_threshold=graph_threshold,
                )
                y_true = [1 if item.sample.label == "malicious" else 0 for item in test_scored]
                y_score = [row["graph_score"] for row in test_graph_rows]
                y_pred = [int(row["prediction"]) for row in test_graph_rows]
                metrics = _binary_metrics(y_true, y_score, y_pred)

                unknown_episode_rows = _build_nuisance_diagnosis_rows(
                    unknown_scored,
                    proposal_mode=proposal_mode,
                    top_k=_proposal_top_k(profile),
                    calibration=calibration,
                    final_decision_mode=final_decision_mode,
                )
                malicious_episode_rows = _build_nuisance_diagnosis_rows(
                    test_scored,
                    proposal_mode=proposal_mode,
                    top_k=_proposal_top_k(profile),
                    calibration=calibration,
                    final_decision_mode=final_decision_mode,
                )
                combined_episode_rows = malicious_episode_rows + unknown_episode_rows
                malicious_like_precision_den = sum(bool(row["final_episode_decision"]) for row in combined_episode_rows)
                malicious_like_precision_num = sum(
                    bool(row["final_episode_decision"]) and row["label_type"] == "malicious"
                    for row in combined_episode_rows
                )
                nuisance_rejection_rate = (
                    float(sum(bool(row["nuisance_score"] >= calibration.nuisance_threshold) for row in unknown_episode_rows) / len(unknown_episode_rows))
                    if unknown_episode_rows
                    else 0.0
                )
                nuisance_like_false_positive_rate = (
                    float(sum(bool(row["nuisance_score"] >= calibration.nuisance_threshold) for row in malicious_episode_rows) / len(malicious_episode_rows))
                    if malicious_episode_rows
                    else 0.0
                )
                malicious_episode_coverage = (
                    float(sum(row["episode_count"] > 0 for row, item in zip(test_graph_rows, test_scored, strict=True) if item.sample.label == "malicious")
                    / max(sum(item.sample.label == "malicious" for item in test_scored), 1))
                )
                episode_to_graph_consistency = (
                    float(np.mean([
                        (row["selected_episode_count"] > 0) == bool(row["prediction"])
                        for row in test_graph_rows + unknown_graph_rows
                    ]))
                    if test_graph_rows or unknown_graph_rows
                    else 0.0
                )
                row = {
                    "evaluation_mode": evaluation_mode,
                    "scenario_id": scenario_id,
                    "model_name": "episode_graph_nuisance_aware_v1",
                    "calibration_profile": profile.name,
                    "percentile_setting": profile.percentile,
                    "top_k_setting": profile.top_k,
                    "suppression_enabled": False,
                    "support_summary_mode": "episode_graph_nuisance_aware",
                    "extraction_mode": "episode_graph",
                    "region_proposal_mode": proposal_mode,
                    "episode_stitching_mode": proposal_mode,
                    "episode_route_version": "proposal_v1",
                    "verifier_mode": "nuisance_rejection_v1",
                    "final_decision_mode": final_decision_mode,
                    "concentration_threshold_setting": None,
                    "component_ratio_setting": None,
                    "neighborhood_ratio_setting": None,
                    "local_density_threshold": calibration.anomaly_threshold,
                    "neighborhood_persistence_threshold": calibration.nuisance_threshold,
                    "temporal_consistency_threshold": calibration.consistency_threshold,
                    "candidate_region_count_mean": float(np.mean([row["episode_count"] for row in test_graph_rows])) if test_graph_rows else 0.0,
                    "selected_region_count_mean": float(np.mean([row["selected_episode_count"] for row in test_graph_rows])) if test_graph_rows else 0.0,
                    "selected_region_coverage_mean": float(np.mean([row["selected_episode_coverage"] for row in test_graph_rows])) if test_graph_rows else 0.0,
                    "single_edge_region_ratio": float(np.mean([row["single_edge_region_ratio"] for row in test_graph_rows])) if test_graph_rows else 0.0,
                    "single_flow_episode_ratio": float(np.mean([row["single_flow_episode_ratio"] for row in test_graph_rows])) if test_graph_rows else 0.0,
                    "short_episode_ratio": float(np.mean([row["short_episode_ratio"] for row in test_graph_rows])) if test_graph_rows else 0.0,
                    "mean_candidate_time_span": float(np.mean([row["mean_candidate_time_span"] for row in test_graph_rows])) if test_graph_rows else 0.0,
                    "mean_episode_flow_count": float(np.mean([row["mean_episode_flow_count"] for row in test_graph_rows])) if test_graph_rows else 0.0,
                    "malicious_episode_coverage": malicious_episode_coverage,
                    "train_benign_graphs": len(train_graphs),
                    "calib_benign_graphs": len(calib_graphs),
                    "test_benign_graphs": sum(item.sample.label == "benign" for item in test_scored),
                    "test_malicious_graphs": sum(item.sample.label == "malicious" for item in test_scored),
                    "test_unknown_graphs": len(unknown_scored),
                    "threshold": graph_threshold,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "fpr": metrics["fpr"],
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                    "background_hit_ratio": (
                        float(sum(int(row["prediction"]) for row in unknown_graph_rows) / len(unknown_graph_rows))
                        if unknown_graph_rows
                        else 0.0
                    ),
                    "unknown_count": len(unknown_graph_scores),
                    "unknown_score_mean": float(np.mean(unknown_graph_scores)) if unknown_graph_scores else 0.0,
                    "unknown_score_median": float(np.median(unknown_graph_scores)) if unknown_graph_scores else 0.0,
                    "nuisance_rejection_rate": nuisance_rejection_rate,
                    "nuisance_like_false_positive_rate": nuisance_like_false_positive_rate,
                    "malicious_like_episode_precision": (
                        float(malicious_like_precision_num / malicious_like_precision_den)
                        if malicious_like_precision_den
                        else 0.0
                    ),
                    "episode_to_graph_decision_consistency": episode_to_graph_consistency,
                }
                rows.append(row)
                benchmark_rows.append(dict(row))
            if profile.name == "heldout_q95_top1":
                construction_rows.extend(
                    _build_episode_construction_rows(
                        test_scored + unknown_scored,
                        proposal_mode=proposal_mode,
                        top_k=_proposal_top_k(profile),
                    )
                )
                nuisance_rows.extend(
                    _build_nuisance_diagnosis_rows(
                        test_scored + unknown_scored,
                        proposal_mode=proposal_mode,
                        top_k=_proposal_top_k(profile),
                        calibration=calibration,
                        final_decision_mode="consistency_aware_episode",
                    )
                )
    return rows, construction_rows, nuisance_rows, benchmark_rows


def _evaluate_sessionized_episode_graph_profiles(
    *,
    evaluation_mode: str,
    scenario_id: str,
    train_graphs: list[object],
    calib_graphs: list[object],
    test_samples: list[ScenarioWindowSample],
    unknown_samples: list[ScenarioWindowSample],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    model, preprocessor = _train_model(train_graphs, model_name="edge_temporal_binary_v2")
    rows: list[dict[str, object]] = []
    construction_rows: list[dict[str, object]] = []
    nuisance_rows: list[dict[str, object]] = []
    benchmark_rows: list[dict[str, object]] = []
    for profile in default_edge_calibration_profiles():
        if profile.name not in TWO_STAGE_PROFILE_NAMES:
            continue
        calib_scored = _score_samples(
            model,
            [ScenarioWindowSample(scenario_id, -1, "calib", "benign", graph) for graph in calib_graphs],
            preprocessor,
            model_name="edge_temporal_binary_v2",
            top_k=profile.top_k,
        )
        if not calib_scored:
            calib_scored = _score_samples(
                model,
                [ScenarioWindowSample(scenario_id, -1, "train", "benign", graph) for graph in train_graphs],
                preprocessor,
                model_name="edge_temporal_binary_v2",
                top_k=profile.top_k,
            )
        test_scored = _score_samples(
            model,
            test_samples,
            preprocessor,
            model_name="edge_temporal_binary_v2",
            top_k=profile.top_k,
        )
        unknown_scored = _score_samples(
            model,
            unknown_samples,
            preprocessor,
            model_name="edge_temporal_binary_v2",
            top_k=profile.top_k,
        )
        for stitching_mode in EPISODE_STITCHING_MODES:
            benign_episode_scores: list[NuisanceAwareEpisodeScore] = []
            nuisance_episode_scores: list[NuisanceAwareEpisodeScore] = []
            for item in calib_scored:
                _episodes, _episode_graph, episode_scores = _score_sample_sessionized_episodes(
                    item,
                    stitching_mode=stitching_mode,
                )
                benign_episode_scores.extend(episode_scores)
            for item in unknown_scored:
                _episodes, _episode_graph, episode_scores = _score_sample_sessionized_episodes(
                    item,
                    stitching_mode=stitching_mode,
                )
                nuisance_episode_scores.extend(episode_scores)
            calibration = calibrate_nuisance_aware_scores(
                benign_episode_scores,
                nuisance_episode_scores,
                percentile=profile.percentile,
            )
            for final_decision_mode in EPISODE_FINAL_DECISION_MODES:
                calib_graph_rows, calib_graph_scores = _aggregate_sessionized_episode_samples(
                    calib_scored,
                    stitching_mode=stitching_mode,
                    final_decision_mode=final_decision_mode,
                    calibration=calibration,
                    graph_threshold=None,
                )
                graph_threshold = float(np.percentile(np.asarray(calib_graph_scores, dtype=float), profile.percentile)) if calib_graph_scores else calibration.consistency_threshold
                test_graph_rows, test_graph_scores = _aggregate_sessionized_episode_samples(
                    test_scored,
                    stitching_mode=stitching_mode,
                    final_decision_mode=final_decision_mode,
                    calibration=calibration,
                    graph_threshold=graph_threshold,
                )
                unknown_graph_rows, unknown_graph_scores = _aggregate_sessionized_episode_samples(
                    unknown_scored,
                    stitching_mode=stitching_mode,
                    final_decision_mode=final_decision_mode,
                    calibration=calibration,
                    graph_threshold=graph_threshold,
                )
                y_true = [1 if item.sample.label == "malicious" else 0 for item in test_scored]
                y_score = [row["graph_score"] for row in test_graph_rows]
                y_pred = [int(row["prediction"]) for row in test_graph_rows]
                metrics = _binary_metrics(y_true, y_score, y_pred)
                unknown_episode_rows = _build_sessionized_nuisance_diagnosis_rows(
                    unknown_scored,
                    stitching_mode=stitching_mode,
                    calibration=calibration,
                    final_decision_mode=final_decision_mode,
                )
                malicious_episode_rows = _build_sessionized_nuisance_diagnosis_rows(
                    test_scored,
                    stitching_mode=stitching_mode,
                    calibration=calibration,
                    final_decision_mode=final_decision_mode,
                )
                combined_episode_rows = malicious_episode_rows + unknown_episode_rows
                malicious_like_precision_den = sum(bool(row["final_episode_decision"]) for row in combined_episode_rows)
                malicious_like_precision_num = sum(
                    bool(row["final_episode_decision"]) and row["label_type"] == "malicious"
                    for row in combined_episode_rows
                )
                nuisance_rejection_rate = (
                    float(sum(bool(row["nuisance_score"] >= calibration.nuisance_threshold) for row in unknown_episode_rows) / len(unknown_episode_rows))
                    if unknown_episode_rows
                    else 0.0
                )
                nuisance_like_false_positive_rate = (
                    float(sum(bool(row["nuisance_score"] >= calibration.nuisance_threshold) for row in malicious_episode_rows) / len(malicious_episode_rows))
                    if malicious_episode_rows
                    else 0.0
                )
                malicious_episode_coverage = (
                    float(sum(row["episode_count"] > 0 for row, item in zip(test_graph_rows, test_scored, strict=True) if item.sample.label == "malicious")
                    / max(sum(item.sample.label == "malicious" for item in test_scored), 1))
                )
                episode_to_graph_consistency = (
                    float(np.mean([
                        (row["selected_episode_count"] > 0) == bool(row["prediction"])
                        for row in test_graph_rows + unknown_graph_rows
                    ]))
                    if test_graph_rows or unknown_graph_rows
                    else 0.0
                )
                row = {
                    "evaluation_mode": evaluation_mode,
                    "scenario_id": scenario_id,
                    "model_name": "episode_graph_nuisance_aware_v2",
                    "calibration_profile": profile.name,
                    "percentile_setting": profile.percentile,
                    "top_k_setting": profile.top_k,
                    "suppression_enabled": False,
                    "support_summary_mode": "episode_graph_nuisance_aware",
                    "extraction_mode": "episode_graph",
                    "region_proposal_mode": "sessionized_episode",
                    "episode_stitching_mode": stitching_mode,
                    "episode_route_version": "sessionized_v2",
                    "verifier_mode": "nuisance_rejection_v1",
                    "final_decision_mode": final_decision_mode,
                    "concentration_threshold_setting": None,
                    "component_ratio_setting": None,
                    "neighborhood_ratio_setting": None,
                    "local_density_threshold": calibration.anomaly_threshold,
                    "neighborhood_persistence_threshold": calibration.nuisance_threshold,
                    "temporal_consistency_threshold": calibration.consistency_threshold,
                    "candidate_region_count_mean": float(np.mean([row["episode_count"] for row in test_graph_rows])) if test_graph_rows else 0.0,
                    "selected_region_count_mean": float(np.mean([row["selected_episode_count"] for row in test_graph_rows])) if test_graph_rows else 0.0,
                    "selected_region_coverage_mean": float(np.mean([row["selected_episode_coverage"] for row in test_graph_rows])) if test_graph_rows else 0.0,
                    "single_edge_region_ratio": float(np.mean([row["single_edge_region_ratio"] for row in test_graph_rows])) if test_graph_rows else 0.0,
                    "single_flow_episode_ratio": float(np.mean([row["single_flow_episode_ratio"] for row in test_graph_rows])) if test_graph_rows else 0.0,
                    "short_episode_ratio": float(np.mean([row["short_episode_ratio"] for row in test_graph_rows])) if test_graph_rows else 0.0,
                    "mean_candidate_time_span": float(np.mean([row["mean_candidate_time_span"] for row in test_graph_rows])) if test_graph_rows else 0.0,
                    "mean_episode_flow_count": float(np.mean([row["mean_episode_flow_count"] for row in test_graph_rows])) if test_graph_rows else 0.0,
                    "malicious_episode_coverage": malicious_episode_coverage,
                    "train_benign_graphs": len(train_graphs),
                    "calib_benign_graphs": len(calib_graphs),
                    "test_benign_graphs": sum(item.sample.label == "benign" for item in test_scored),
                    "test_malicious_graphs": sum(item.sample.label == "malicious" for item in test_scored),
                    "test_unknown_graphs": len(unknown_scored),
                    "threshold": graph_threshold,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "fpr": metrics["fpr"],
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                    "background_hit_ratio": (
                        float(sum(int(row["prediction"]) for row in unknown_graph_rows) / len(unknown_graph_rows))
                        if unknown_graph_rows
                        else 0.0
                    ),
                    "unknown_count": len(unknown_graph_scores),
                    "unknown_score_mean": float(np.mean(unknown_graph_scores)) if unknown_graph_scores else 0.0,
                    "unknown_score_median": float(np.median(unknown_graph_scores)) if unknown_graph_scores else 0.0,
                    "nuisance_rejection_rate": nuisance_rejection_rate,
                    "nuisance_like_false_positive_rate": nuisance_like_false_positive_rate,
                    "malicious_like_episode_precision": (
                        float(malicious_like_precision_num / malicious_like_precision_den)
                        if malicious_like_precision_den
                        else 0.0
                    ),
                    "episode_to_graph_decision_consistency": episode_to_graph_consistency,
                }
                rows.append(row)
                benchmark_rows.append(dict(row))
            if profile.name == "heldout_q95_top1":
                construction_rows.extend(
                    _build_sessionized_episode_construction_rows(
                        test_scored + unknown_scored,
                        stitching_mode=stitching_mode,
                    )
                )
                nuisance_rows.extend(
                    _build_sessionized_nuisance_diagnosis_rows(
                        test_scored + unknown_scored,
                        stitching_mode=stitching_mode,
                        calibration=calibration,
                        final_decision_mode="consistency_aware_episode",
                    )
                )
    return rows, construction_rows, nuisance_rows, benchmark_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            handle.write("")
            return
        fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _float_text(value: object | None) -> str:
    if value in {None, "", "n/a"}:
        return "n/a"
    return f"{float(value):.4f}"


def _write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# CTU-13 Binary Benchmark",
        "",
        "Primary metrics use only benign + malicious graphs.",
        "Background / unknown graphs are reported only as secondary analysis.",
        "",
        "| evaluation_mode | scenario_id | model_name | calibration_profile | extraction_mode | support_summary_mode | nuisance_boundary_mode | final_decision_mode | precision | recall | f1 | fpr | background_hit_ratio | nuisance_rejection_rate | malicious_blocked_by_nuisance_rate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['evaluation_mode']} | {row['scenario_id']} | {row['model_name']} | {row['calibration_profile']} | "
            f"{row.get('extraction_mode', 'per_src_ip_within_window')} | {row.get('support_summary_mode', 'old_concentration')} | {row.get('nuisance_boundary_mode', 'n/a')} | "
            f"{row.get('final_decision_mode', 'n/a')} | "
            f"{_float_text(row['precision'])} | {_float_text(row['recall'])} | {_float_text(row['f1'])} | "
            f"{_float_text(row['fpr'])} | {_float_text(row['background_hit_ratio'])} | {_float_text(row.get('nuisance_rejection_rate'))} | {_float_text(row.get('malicious_blocked_by_nuisance_rate'))} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_simple_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            handle.write("")
            return
        fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_s52_diagnosis_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# CTU-13 Scenario 52 Coverage Diagnosis",
        "",
        "| scenario_id | aligned_malicious_flows | malicious_flows_after_primary_filter | malicious_flows_after_windowing | malicious_candidate_graphs | final_test_malicious_graphs | major_loss_stage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario_id']} | {row['aligned_malicious_flows']} | {row['malicious_flows_after_primary_filter']} | "
            f"{row['malicious_flows_after_windowing']} | {row['malicious_candidate_graphs']} | "
            f"{row['final_test_malicious_graphs']} | {row['major_loss_stage']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_primary_extraction_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# CTU-13 Primary Graph Extraction Summary",
        "",
        "| scenario_id | window_size | graph_grouping_policy | candidate_graph_count | benign_graph_count | malicious_graph_count | unknown_heavy_graph_count | filtered_out_reason |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario_id']} | {row['window_size']} | {row['graph_grouping_policy']} | "
            f"{row['candidate_graph_count']} | {row['benign_graph_count']} | {row['malicious_graph_count']} | "
            f"{row['unknown_heavy_graph_count']} | {row['filtered_out_reason']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_unknown_suppression_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# CTU-13 Support Summary Diagnosis",
        "",
        "| scenario_id | graph_id | label_type | top1_edge_score | topk_edge_mean | local_support_edge_count | local_support_edge_density | local_support_node_coverage | max_local_support_density | abnormal_neighborhood_count | cross_neighborhood_support_ratio | slice_abnormal_presence_count | slice_abnormal_consistency_ratio | decision_old | decision_new | suspected_failure_mode |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario_id']} | {row['graph_id']} | {row['label_type']} | "
            f"{_float_text(row['top1_edge_score'])} | {_float_text(row['topk_edge_mean'])} | "
            f"{row['local_support_edge_count']} | {_float_text(row['local_support_edge_density'])} | "
            f"{_float_text(row['local_support_node_coverage'])} | {_float_text(row['max_local_support_density'])} | "
            f"{row['abnormal_neighborhood_count']} | {_float_text(row['cross_neighborhood_support_ratio'])} | "
            f"{row['slice_abnormal_presence_count']} | {_float_text(row['slice_abnormal_consistency_ratio'])} | "
            f"{row['decision_old']} | {row['decision_new']} | "
            f"{row['suspected_failure_mode']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_candidate_region_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# CTU-13 Candidate Region Proposal",
        "",
        "| scenario_id | extraction_mode | graph_id | candidate_region_id | seed_edge_count | candidate_edge_count | candidate_node_count | candidate_time_span | candidate_src_count | candidate_dst_count | repeated_endpoint_count | repeated_slice_support_count | support_cluster_density | candidate_score_seed | candidate_score_mean | candidate_score_max | label_type |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario_id']} | {row.get('extraction_mode', 'per_src_ip_within_window')} | {row['graph_id']} | {row['candidate_region_id']} | "
            f"{row['seed_edge_count']} | {row['candidate_edge_count']} | {row['candidate_node_count']} | "
            f"{_float_text(row['candidate_time_span'])} | {row['candidate_src_count']} | {row['candidate_dst_count']} | "
            f"{row.get('repeated_endpoint_count', 0)} | {row.get('repeated_slice_support_count', 0)} | {_float_text(row.get('support_cluster_density'))} | "
            f"{_float_text(row['candidate_score_seed'])} | {_float_text(row['candidate_score_mean'])} | "
            f"{_float_text(row['candidate_score_max'])} | {row['label_type']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_micrograph_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# CTU-13 Micrograph Verifier",
        "",
        "| scenario_id | extraction_mode | graph_id | candidate_region_id | micrograph_edge_count | micrograph_node_count | micrograph_slice_count | micrograph_score | micrograph_consistency_score | micrograph_density_score | micrograph_temporal_persistence_score | micrograph_decision | label_type |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario_id']} | {row.get('extraction_mode', 'per_src_ip_within_window')} | {row['graph_id']} | {row['candidate_region_id']} | "
            f"{row['micrograph_edge_count']} | {row['micrograph_node_count']} | {row['micrograph_slice_count']} | "
            f"{_float_text(row['micrograph_score'])} | {_float_text(row['micrograph_consistency_score'])} | "
            f"{_float_text(row['micrograph_density_score'])} | {_float_text(row['micrograph_temporal_persistence_score'])} | "
            f"{row['micrograph_decision']} | {row['label_type']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_graph_extraction_modes_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# CTU-13 Graph Extraction Modes",
        "",
        "| scenario_id | extraction_mode | graph_count | avg_graph_size | avg_edge_count | avg_node_count | benign_graph_count | malicious_graph_count | unknown_graph_count | candidate_ready_graph_count |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario_id']} | {row['extraction_mode']} | {row['graph_count']} | "
            f"{_float_text(row['avg_graph_size'])} | {_float_text(row['avg_edge_count'])} | {_float_text(row['avg_node_count'])} | "
            f"{row['benign_graph_count']} | {row['malicious_graph_count']} | {row['unknown_graph_count']} | {row['candidate_ready_graph_count']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_proposal_quality_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# CTU-13 Proposal Quality Diagnosis",
        "",
        "| scenario_id | extraction_mode | region_proposal_mode | malicious_candidate_edge_mean | unknown_candidate_edge_mean | malicious_candidate_time_span_mean | unknown_candidate_time_span_mean | malicious_repeated_endpoint_mean | unknown_repeated_endpoint_mean | malicious_support_cluster_density_mean | unknown_support_cluster_density_mean | malicious_mean_proposal_score | unknown_mean_proposal_score | proposal_score_gap | single_edge_region_ratio | malicious_micrograph_mean_score | unknown_micrograph_mean_score | micrograph_score_gap |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario_id']} | {row['extraction_mode']} | {row['region_proposal_mode']} | "
            f"{_float_text(row['malicious_candidate_edge_mean'])} | {_float_text(row['unknown_candidate_edge_mean'])} | "
            f"{_float_text(row['malicious_candidate_time_span_mean'])} | {_float_text(row['unknown_candidate_time_span_mean'])} | "
            f"{_float_text(row['malicious_repeated_endpoint_mean'])} | {_float_text(row['unknown_repeated_endpoint_mean'])} | "
            f"{_float_text(row['malicious_support_cluster_density_mean'])} | {_float_text(row['unknown_support_cluster_density_mean'])} | "
            f"{_float_text(row['malicious_mean_proposal_score'])} | {_float_text(row['unknown_mean_proposal_score'])} | "
            f"{_float_text(row['proposal_score_gap'])} | {_float_text(row['single_edge_region_ratio'])} | "
            f"{_float_text(row['malicious_micrograph_mean_score'])} | {_float_text(row['unknown_micrograph_mean_score'])} | {_float_text(row['micrograph_score_gap'])} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_episode_graph_construction_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# CTU-13 Episode Graph Construction",
        "",
        "| scenario_id | graph_id | episode_id | episode_route_version | stitching_mode | involved_flows | involved_endpoints | repeated_pair_count | burst_persistence | support_cluster_density | protocol_consistency_score | nuisance_likelihood | episode_time_span | episode_edge_count | label_type |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario_id']} | {row['graph_id']} | {row['episode_id']} | {row.get('episode_route_version', 'proposal_v1')} | {row.get('stitching_mode', row.get('proposal_mode', 'unknown'))} | "
            f"{row['involved_flows']} | {row['involved_endpoints']} | {row['repeated_pair_count']} | "
            f"{_float_text(row['burst_persistence'])} | {_float_text(row['support_cluster_density'])} | {_float_text(row.get('protocol_consistency_score'))} | "
            f"{_float_text(row['nuisance_likelihood'])} | {_float_text(row['episode_time_span'])} | "
            f"{row['episode_edge_count']} | {row['label_type']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_episode_quality_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# CTU-13 Episode Quality Diagnosis",
        "",
        "| scenario_id | episode_route_version | stitching_mode | malicious_episode_count | unknown_episode_count | malicious_episode_edge_mean | unknown_episode_edge_mean | mean_episode_duration | mean_episode_flow_count | mean_repeated_pair_count | mean_burst_persistence | mean_protocol_consistency_score | single_flow_episode_ratio | short_episode_ratio | malicious_mean_episode_score | unknown_mean_episode_score | episode_score_gap | malicious_mean_nuisance_score | unknown_mean_nuisance_score | nuisance_score_gap | malicious_unknown_overlap_ratio |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario_id']} | {row.get('episode_route_version', 'proposal_v1')} | {row.get('stitching_mode', row.get('proposal_mode', 'unknown'))} | "
            f"{row.get('malicious_episode_count', 0)} | {row.get('unknown_episode_count', 0)} | "
            f"{_float_text(row['malicious_episode_edge_mean'])} | {_float_text(row['unknown_episode_edge_mean'])} | {_float_text(row.get('mean_episode_duration'))} | "
            f"{_float_text(row.get('mean_episode_flow_count'))} | {_float_text(row.get('mean_repeated_pair_count'))} | {_float_text(row.get('mean_burst_persistence'))} | "
            f"{_float_text(row.get('mean_protocol_consistency_score'))} | {_float_text(row.get('single_flow_episode_ratio'))} | {_float_text(row.get('short_episode_ratio'))} | "
            f"{_float_text(row['malicious_mean_episode_score'])} | {_float_text(row['unknown_mean_episode_score'])} | "
            f"{_float_text(row['episode_score_gap'])} | {_float_text(row['malicious_mean_nuisance_score'])} | "
            f"{_float_text(row['unknown_mean_nuisance_score'])} | {_float_text(row['nuisance_score_gap'])} | "
            f"{_float_text(row['malicious_unknown_overlap_ratio'])} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_episode_graph_benchmark_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# CTU-13 Episode Graph Benchmark",
        "",
        "| evaluation_mode | scenario_id | model_name | calibration_profile | region_proposal_mode | final_decision_mode | precision | recall | f1 | fpr | background_hit_ratio | nuisance_rejection_rate | nuisance_like_false_positive_rate | malicious_like_episode_precision | episode_to_graph_decision_consistency |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['evaluation_mode']} | {row['scenario_id']} | {row['model_name']} | {row['calibration_profile']} | "
            f"{row['region_proposal_mode']} | {row['final_decision_mode']} | {_float_text(row['precision'])} | "
            f"{_float_text(row['recall'])} | {_float_text(row['f1'])} | {_float_text(row['fpr'])} | "
            f"{_float_text(row['background_hit_ratio'])} | {_float_text(row['nuisance_rejection_rate'])} | "
            f"{_float_text(row['nuisance_like_false_positive_rate'])} | {_float_text(row['malicious_like_episode_precision'])} | "
            f"{_float_text(row['episode_to_graph_decision_consistency'])} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_nuisance_aware_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# CTU-13 Nuisance-Aware Diagnosis",
        "",
        "| scenario_id | graph_id | episode_id | episode_route_version | stitching_mode | label_type | episode_score | nuisance_score | anomaly_score | repeated_pair_count | burst_persistence | support_cluster_density | protocol_consistency_score | direction_pattern_consistency | episode_time_span | episode_endpoint_span | final_episode_decision | final_graph_decision | suspected_failure_mode |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario_id']} | {row['graph_id']} | {row['episode_id']} | {row.get('episode_route_version', 'proposal_v1')} | {row.get('stitching_mode', 'unknown')} | {row['label_type']} | "
            f"{_float_text(row['episode_score'])} | {_float_text(row['nuisance_score'])} | {_float_text(row['anomaly_score'])} | "
            f"{row['repeated_pair_count']} | {_float_text(row['burst_persistence'])} | {_float_text(row['support_cluster_density'])} | {_float_text(row.get('protocol_consistency_score'))} | {_float_text(row.get('direction_pattern_consistency'))} | "
            f"{_float_text(row['episode_time_span'])} | {row['episode_endpoint_span']} | {row['final_episode_decision']} | "
            f"{row['final_graph_decision']} | {row['suspected_failure_mode']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_nuisance_boundary_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# CTU-13 Nuisance Boundary Diagnosis",
        "",
        "| scenario_id | graph_id | label_type | anomaly_score | nuisance_score | malicious_support_score | benign_boundary_value | nuisance_boundary_value | final_internal_state | final_binary_decision | top1_edge_score | topk_edge_mean | local_support_score | neighborhood_persistence_score | temporal_consistency_score | suspected_failure_mode |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario_id']} | {row['graph_id']} | {row['label_type']} | "
            f"{_float_text(row['anomaly_score'])} | {_float_text(row['nuisance_score'])} | {_float_text(row['malicious_support_score'])} | "
            f"{_float_text(row['benign_boundary_value'])} | {_float_text(row['nuisance_boundary_value'])} | {row['final_internal_state']} | {row['final_binary_decision']} | "
            f"{_float_text(row['top1_edge_score'])} | {_float_text(row['topk_edge_mean'])} | {_float_text(row['local_support_score'])} | "
            f"{_float_text(row['neighborhood_persistence_score'])} | {_float_text(row['temporal_consistency_score'])} | {row['suspected_failure_mode']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_nuisance_aware_benchmark_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# CTU-13 Nuisance-Aware Benchmark",
        "",
        "| evaluation_mode | scenario_id | model_name | calibration_profile | support_summary_mode | nuisance_boundary_mode | precision | recall | f1 | fpr | background_hit_ratio | nuisance_rejection_rate | nuisance_like_false_positive_rate | benign_misrejected_as_nuisance_rate | malicious_blocked_by_nuisance_rate |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['evaluation_mode']} | {row['scenario_id']} | {row['model_name']} | {row['calibration_profile']} | "
            f"{row.get('support_summary_mode', 'old_concentration')} | {row.get('nuisance_boundary_mode', 'n/a')} | "
            f"{_float_text(row['precision'])} | {_float_text(row['recall'])} | {_float_text(row['f1'])} | {_float_text(row['fpr'])} | "
            f"{_float_text(row['background_hit_ratio'])} | {_float_text(row['nuisance_rejection_rate'])} | {_float_text(row['nuisance_like_false_positive_rate'])} | "
            f"{_float_text(row['benign_misrejected_as_nuisance_rate'])} | {_float_text(row['malicious_blocked_by_nuisance_rate'])} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_episode_sessionization_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# CTU-13 Episode Sessionization",
        "",
        "| scenario_id | episode_route_version | stitching_mode | total_episode_count | benign_episode_count | malicious_episode_count | unknown_episode_count | malicious_episode_coverage | single_flow_episode_ratio | short_episode_ratio | mean_episode_duration | mean_episode_flow_count | mean_repeated_pair_count | mean_burst_persistence | mean_protocol_consistency_score | nuisance_rejection_rate | nuisance_like_false_positive_rate | malicious_like_episode_precision | episode_to_graph_decision_consistency |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario_id']} | {row['episode_route_version']} | {row['stitching_mode']} | {row['total_episode_count']} | "
            f"{row['benign_episode_count']} | {row['malicious_episode_count']} | {row['unknown_episode_count']} | "
            f"{_float_text(row['malicious_episode_coverage'])} | {_float_text(row['single_flow_episode_ratio'])} | {_float_text(row['short_episode_ratio'])} | "
            f"{_float_text(row['mean_episode_duration'])} | {_float_text(row['mean_episode_flow_count'])} | {_float_text(row['mean_repeated_pair_count'])} | "
            f"{_float_text(row['mean_burst_persistence'])} | {_float_text(row['mean_protocol_consistency_score'])} | {_float_text(row['nuisance_rejection_rate'])} | "
            f"{_float_text(row['nuisance_like_false_positive_rate'])} | {_float_text(row['malicious_like_episode_precision'])} | {_float_text(row['episode_to_graph_decision_consistency'])} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    manifest_entries = _bootstrap_manifest()
    if len(manifest_entries) < len(TARGET_SCENARIO_IDS):
        missing = sorted(set(TARGET_SCENARIO_IDS) - {entry.scenario_id for entry in manifest_entries})
        raise SystemExit(f"Missing CTU-13 scenarios for benchmark: {', '.join(missing)}")

    alignment_summaries = []
    prepared_scenarios: list[PreparedScenario] = []
    two_stage_prepared_by_mode: dict[GraphExtractionMode, list[PreparedScenario]] = {
        mode: [] for mode in TWO_STAGE_EXTRACTION_MODES
    }
    diagnosis_rows: list[dict[str, object]] = []
    primary_extraction_rows: list[dict[str, object]] = []
    graph_extraction_mode_rows: list[dict[str, object]] = []
    for entry in manifest_entries:
        print(f"[ctu13] loading scenario {entry.scenario_id}", flush=True)
        dataset, alignment_summary, start_packet_offset, packet_limit, label_flows = _build_labeled_dataset(entry)
        alignment_summaries.append(alignment_summary)
        primary_samples, unknown_samples = _pack_with_labels(
            dataset,
            scenario_id=entry.scenario_id,
            extraction_mode="per_src_ip_within_window",
        )
        prepared = _group_scenario_samples(entry.scenario_id, primary_samples, unknown_samples)
        for extraction_mode in TWO_STAGE_EXTRACTION_MODES:
            mode_primary_samples, mode_unknown_samples = _pack_with_labels(
                dataset,
                scenario_id=entry.scenario_id,
                extraction_mode=extraction_mode,
            )
            two_stage_prepared_by_mode[extraction_mode].append(
                _group_scenario_samples(entry.scenario_id, mode_primary_samples, mode_unknown_samples)
            )
            graph_extraction_mode_rows.append(
                _graph_extraction_summary_row(
                    entry.scenario_id,
                    extraction_mode,
                    mode_primary_samples,
                    mode_unknown_samples,
                )
            )
        aligned_malicious_flows = alignment_summary.malicious_count
        malicious_flows_after_primary_filter = sum(
            1
            for record in dataset.records
            if str(record.metadata.get("ctu13_binary_label", "unknown")) == "malicious"
        )
        malicious_flows_after_windowing = sum(sample.label == "malicious" for sample in primary_samples)
        diagnosis_stage = "none"
        if aligned_malicious_flows == 0:
            diagnosis_stage = "alignment_or_packet_coverage"
        elif malicious_flows_after_primary_filter == 0:
            diagnosis_stage = "primary_filter"
        elif malicious_flows_after_windowing == 0:
            diagnosis_stage = "windowing_or_grouping"
        elif sum(sample.label == "malicious" for sample in prepared.test_samples) == 0:
            diagnosis_stage = "split_policy"
        diagnosis_rows.append(
            {
                "scenario_id": entry.scenario_id,
                "packet_start_offset": start_packet_offset,
                "packet_limit": packet_limit,
                "total_label_flows": len(label_flows),
                "label_malicious_flows": sum(flow.binary_label == "malicious" for flow in label_flows),
                "loaded_flow_count": len(dataset.records),
                "aligned_malicious_flows": aligned_malicious_flows,
                "malicious_flows_after_primary_filter": malicious_flows_after_primary_filter,
                "malicious_flows_after_windowing": malicious_flows_after_windowing,
                "malicious_candidate_graphs": malicious_flows_after_windowing,
                "final_test_malicious_graphs": sum(sample.label == "malicious" for sample in prepared.test_samples),
                "major_loss_stage": diagnosis_stage,
            }
        )
        primary_extraction_rows.append(
            {
                "scenario_id": entry.scenario_id,
                "window_size": SCENARIO_WINDOW_SIZE_SECONDS.get(entry.scenario_id, WINDOW_SIZE_SECONDS),
                "graph_grouping_policy": "per_src_ip_within_window",
                "candidate_graph_count": len(primary_samples) + len(unknown_samples),
                "benign_graph_count": sum(sample.label == "benign" for sample in primary_samples),
                "malicious_graph_count": sum(sample.label == "malicious" for sample in primary_samples),
                "unknown_heavy_graph_count": len(unknown_samples),
                "filtered_out_reason": (
                    "packet_prefix_coverage_before_malicious_onset"
                    if entry.scenario_id == "52" and alignment_summary.malicious_count == 0
                    else "none"
                ),
            }
        )
        print(
            "[ctu13] "
            f"{entry.scenario_id}: train={len(prepared.train_graphs)} calib={len(prepared.calib_graphs)} "
            f"test_benign={sum(sample.label == 'benign' for sample in prepared.test_samples)} "
            f"test_malicious={sum(sample.label == 'malicious' for sample in prepared.test_samples)} "
            f"unknown={len(prepared.unknown_samples)}",
            flush=True,
        )
        prepared_scenarios.append(prepared)

    write_alignment_summary_csv(alignment_summaries, ALIGNMENT_CSV)
    write_alignment_summary_markdown(alignment_summaries, ALIGNMENT_MD)
    _write_simple_csv(S52_DIAGNOSIS_CSV, diagnosis_rows)
    _write_s52_diagnosis_markdown(S52_DIAGNOSIS_MD, diagnosis_rows)
    _write_simple_csv(PRIMARY_EXTRACTION_CSV, primary_extraction_rows)
    _write_primary_extraction_markdown(PRIMARY_EXTRACTION_MD, primary_extraction_rows)
    _write_simple_csv(GRAPH_EXTRACTION_MODES_CSV, graph_extraction_mode_rows)
    _write_graph_extraction_modes_markdown(GRAPH_EXTRACTION_MODES_MD, graph_extraction_mode_rows)

    rows: list[dict[str, object]] = []
    support_summary_rows: list[dict[str, object]] = []
    nuisance_boundary_rows: list[dict[str, object]] = []
    nuisance_aware_benchmark_rows: list[dict[str, object]] = []
    candidate_region_rows: list[dict[str, object]] = []
    micrograph_rows: list[dict[str, object]] = []
    episode_construction_rows: list[dict[str, object]] = []
    nuisance_aware_rows: list[dict[str, object]] = []
    episode_graph_benchmark_rows: list[dict[str, object]] = []
    episode_sessionization_rows: list[dict[str, object]] = []
    for prepared in prepared_scenarios:
        scenario_rows, scenario_diagnosis_rows, scenario_nuisance_rows, scenario_nuisance_benchmark_rows = _evaluate_scenario(
            prepared,
            evaluation_mode="scenario_wise",
        )
        rows.extend(scenario_rows)
        support_summary_rows.extend(scenario_diagnosis_rows)
        nuisance_boundary_rows.extend(scenario_nuisance_rows)
        nuisance_aware_benchmark_rows.extend(scenario_nuisance_benchmark_rows)
    merged = _merge_prepared_scenarios(prepared_scenarios)
    merged_rows, merged_diagnosis_rows, merged_nuisance_rows, merged_nuisance_benchmark_rows = _evaluate_scenario(
        merged,
        evaluation_mode="merged",
    )
    rows.extend(merged_rows)
    support_summary_rows.extend(merged_diagnosis_rows)
    nuisance_boundary_rows.extend(merged_nuisance_rows)
    nuisance_aware_benchmark_rows.extend(merged_nuisance_benchmark_rows)
    proposal_quality_rows = _proposal_quality_diagnosis_rows(candidate_region_rows, micrograph_rows)
    episode_quality_rows = _episode_quality_diagnosis_rows(episode_construction_rows, nuisance_aware_rows)
    episode_sessionization_rows = _episode_sessionization_summary_rows(
        episode_construction_rows,
        episode_graph_benchmark_rows,
    )

    _write_csv(BENCHMARK_CSV, rows)
    _write_markdown(BENCHMARK_MD, rows)
    _write_simple_csv(CANDIDATE_REGION_CSV, candidate_region_rows)
    _write_candidate_region_markdown(CANDIDATE_REGION_MD, candidate_region_rows)
    _write_simple_csv(MICROGRAPH_VERIFIER_CSV, micrograph_rows)
    _write_micrograph_markdown(MICROGRAPH_VERIFIER_MD, micrograph_rows)
    _write_simple_csv(PROPOSAL_QUALITY_DIAGNOSIS_CSV, proposal_quality_rows)
    _write_proposal_quality_markdown(PROPOSAL_QUALITY_DIAGNOSIS_MD, proposal_quality_rows)
    _write_simple_csv(SUPPORT_SUMMARY_DIAGNOSIS_CSV, support_summary_rows)
    _write_unknown_suppression_markdown(SUPPORT_SUMMARY_DIAGNOSIS_MD, support_summary_rows)
    _write_simple_csv(UNKNOWN_SUPPRESSION_CSV, support_summary_rows)
    _write_unknown_suppression_markdown(UNKNOWN_SUPPRESSION_MD, support_summary_rows)
    _write_simple_csv(EPISODE_GRAPH_CONSTRUCTION_CSV, episode_construction_rows)
    _write_episode_graph_construction_markdown(EPISODE_GRAPH_CONSTRUCTION_MD, episode_construction_rows)
    _write_simple_csv(EPISODE_QUALITY_DIAGNOSIS_CSV, episode_quality_rows)
    _write_episode_quality_markdown(EPISODE_QUALITY_DIAGNOSIS_MD, episode_quality_rows)
    _write_simple_csv(EPISODE_GRAPH_BENCHMARK_CSV, episode_graph_benchmark_rows)
    _write_episode_graph_benchmark_markdown(EPISODE_GRAPH_BENCHMARK_MD, episode_graph_benchmark_rows)
    _write_simple_csv(NUISANCE_AWARE_DIAGNOSIS_CSV, nuisance_aware_rows)
    _write_nuisance_aware_markdown(NUISANCE_AWARE_DIAGNOSIS_MD, nuisance_aware_rows)
    _write_simple_csv(EPISODE_SESSIONIZATION_CSV, episode_sessionization_rows)
    _write_episode_sessionization_markdown(EPISODE_SESSIONIZATION_MD, episode_sessionization_rows)
    _write_simple_csv(NUISANCE_BOUNDARY_DIAGNOSIS_CSV, nuisance_boundary_rows)
    _write_nuisance_boundary_markdown(NUISANCE_BOUNDARY_DIAGNOSIS_MD, nuisance_boundary_rows)
    _write_simple_csv(NUISANCE_AWARE_BENCHMARK_CSV, nuisance_aware_benchmark_rows)
    _write_nuisance_aware_benchmark_markdown(NUISANCE_AWARE_BENCHMARK_MD, nuisance_aware_benchmark_rows)
    print(f"Wrote {BENCHMARK_CSV}")
    print(f"Wrote {BENCHMARK_MD}")
    print(f"Wrote {CANDIDATE_REGION_CSV}")
    print(f"Wrote {CANDIDATE_REGION_MD}")
    print(f"Wrote {MICROGRAPH_VERIFIER_CSV}")
    print(f"Wrote {MICROGRAPH_VERIFIER_MD}")
    print(f"Wrote {GRAPH_EXTRACTION_MODES_CSV}")
    print(f"Wrote {GRAPH_EXTRACTION_MODES_MD}")
    print(f"Wrote {PROPOSAL_QUALITY_DIAGNOSIS_CSV}")
    print(f"Wrote {PROPOSAL_QUALITY_DIAGNOSIS_MD}")
    print(f"Wrote {SUPPORT_SUMMARY_DIAGNOSIS_CSV}")
    print(f"Wrote {SUPPORT_SUMMARY_DIAGNOSIS_MD}")
    print(f"Wrote {UNKNOWN_SUPPRESSION_CSV}")
    print(f"Wrote {UNKNOWN_SUPPRESSION_MD}")
    print(f"Wrote {EPISODE_GRAPH_CONSTRUCTION_CSV}")
    print(f"Wrote {EPISODE_GRAPH_CONSTRUCTION_MD}")
    print(f"Wrote {EPISODE_QUALITY_DIAGNOSIS_CSV}")
    print(f"Wrote {EPISODE_QUALITY_DIAGNOSIS_MD}")
    print(f"Wrote {EPISODE_GRAPH_BENCHMARK_CSV}")
    print(f"Wrote {EPISODE_GRAPH_BENCHMARK_MD}")
    print(f"Wrote {NUISANCE_AWARE_DIAGNOSIS_CSV}")
    print(f"Wrote {NUISANCE_AWARE_DIAGNOSIS_MD}")
    print(f"Wrote {EPISODE_SESSIONIZATION_CSV}")
    print(f"Wrote {EPISODE_SESSIONIZATION_MD}")
    print(f"Wrote {NUISANCE_BOUNDARY_DIAGNOSIS_CSV}")
    print(f"Wrote {NUISANCE_BOUNDARY_DIAGNOSIS_MD}")
    print(f"Wrote {NUISANCE_AWARE_BENCHMARK_CSV}")
    print(f"Wrote {NUISANCE_AWARE_BENCHMARK_MD}")


if __name__ == "__main__":
    main()
