"""Local micrograph verifier for two-stage edge-centric detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from traffic_graph.pipeline.candidate_region_proposal import CandidateRegion

MicrographVerifierMode = Literal["micrograph_consistency_v1"]
FinalDecisionMode = Literal[
    "max_micrograph_score",
    "top2_micrograph_mean",
    "consistency_aware_aggregation",
]


@dataclass(frozen=True, slots=True)
class MicrographVerificationResult:
    """Verifier output for one candidate region."""

    candidate_region_id: str
    proposal_mode: str
    verifier_mode: MicrographVerifierMode
    micrograph_edge_count: int
    micrograph_node_count: int
    micrograph_slice_count: int
    micrograph_score: float
    micrograph_consistency_score: float
    micrograph_density_score: float
    micrograph_temporal_persistence_score: float
    micrograph_decision: bool = False


@dataclass(frozen=True, slots=True)
class MicrographFinalDecision:
    """Aggregated graph-level decision from multiple micrographs."""

    final_decision_mode: FinalDecisionMode
    decision_score: float
    positive_evidence_count: int
    selected_region_count: int
    selected_region_coverage: float
    is_positive: bool = False


def _edge_slice_index(sample: Any, edge: Any, flow_by_id: dict[str, Any], *, slice_count: int) -> int:
    logical_flow = flow_by_id.get(str(edge.logical_flow_id))
    if logical_flow is None:
        for source_flow_id in getattr(edge, "source_flow_ids", ()):
            logical_flow = flow_by_id.get(str(source_flow_id))
            if logical_flow is not None:
                break
    if logical_flow is None:
        return 0
    window_seconds = max((sample.graph.window_end - sample.graph.window_start).total_seconds(), 1e-6)
    midpoint = logical_flow.start_time + (logical_flow.end_time - logical_flow.start_time) / 2
    ratio = min(max((midpoint - sample.graph.window_start).total_seconds() / window_seconds, 0.0), 0.999999)
    return min(int(ratio * slice_count), slice_count - 1)


def verify_candidate_region(
    sample: Any,
    candidate_region: CandidateRegion,
    communication_indices: list[int],
    communication_scores: np.ndarray,
    *,
    verifier_mode: MicrographVerifierMode = "micrograph_consistency_v1",
    slice_count: int = 3,
    score_threshold: float | None = None,
) -> MicrographVerificationResult:
    """Verify one candidate region using only local unsupervised consistency."""

    region_scores = np.asarray(
        [communication_scores[index] for index in candidate_region.relative_edge_indices],
        dtype=float,
    )
    if region_scores.size == 0:
        return MicrographVerificationResult(
            candidate_region_id=candidate_region.candidate_region_id,
            proposal_mode=candidate_region.proposal_mode,
            verifier_mode=verifier_mode,
            micrograph_edge_count=0,
            micrograph_node_count=0,
            micrograph_slice_count=0,
            micrograph_score=0.0,
            micrograph_consistency_score=0.0,
            micrograph_density_score=0.0,
            micrograph_temporal_persistence_score=0.0,
            micrograph_decision=False,
        )
    threshold = float(score_threshold) if score_threshold is not None else float(np.median(region_scores))
    high_score_mask = region_scores >= threshold
    high_score_count = int(np.sum(high_score_mask))
    micrograph_edge_count = len(candidate_region.relative_edge_indices)
    micrograph_node_count = len(candidate_region.node_ids)
    slice_count = max(2, slice_count)
    window_seconds = max((sample.graph.window_end - sample.graph.window_start).total_seconds(), 1e-6)
    flow_by_id = {str(flow.logical_flow_id): flow for flow in getattr(sample, "logical_flows", ())}
    slice_hits: dict[int, int] = {}
    endpoint_counts: dict[str, int] = {}
    for relative_index in candidate_region.relative_edge_indices:
        edge = sample.graph.edges[communication_indices[relative_index]]
        slice_index = _edge_slice_index(sample, edge, flow_by_id, slice_count=slice_count)
        slice_hits[slice_index] = slice_hits.get(slice_index, 0) + 1
        endpoint_counts[edge.source_node_id] = endpoint_counts.get(edge.source_node_id, 0) + 1
        endpoint_counts[edge.target_node_id] = endpoint_counts.get(edge.target_node_id, 0) + 1
    repeated_support_endpoints = sum(count >= 2 for count in endpoint_counts.values())
    dominance_penalty = float(np.max(region_scores) / max(np.sum(region_scores), 1e-12))
    compactness_score = float(
        0.5 * min(2.0 / max(micrograph_node_count, 1), 1.0)
        + 0.5 * min(1.0 / max(candidate_region.candidate_dst_count, 1), 1.0)
    )
    time_span_ratio = float(min(candidate_region.candidate_time_span / window_seconds, 1.0))
    micrograph_density_score = float(
        0.45 * (high_score_count / max(micrograph_edge_count, 1))
        + 0.30 * compactness_score
        + 0.25 * min(micrograph_edge_count / max(micrograph_node_count, 1), 1.0)
    )
    micrograph_consistency_score = float(
        0.40 * min(high_score_count / 3.0, 1.0)
        + 0.25 * min(repeated_support_endpoints / 3.0, 1.0)
        + 0.20 * max(0.0, 1.0 - dominance_penalty)
        + 0.15 * compactness_score
    )
    micrograph_temporal_persistence_score = float(
        0.35 * (len(slice_hits) / slice_count)
        + 0.40 * time_span_ratio
        + 0.25 * min(repeated_support_endpoints / 3.0, 1.0)
    )
    micrograph_score = float(
        0.35 * micrograph_density_score
        + 0.25 * micrograph_consistency_score
        + 0.40 * micrograph_temporal_persistence_score
    )
    return MicrographVerificationResult(
        candidate_region_id=candidate_region.candidate_region_id,
        proposal_mode=candidate_region.proposal_mode,
        verifier_mode=verifier_mode,
        micrograph_edge_count=micrograph_edge_count,
        micrograph_node_count=micrograph_node_count,
        micrograph_slice_count=len(slice_hits),
        micrograph_score=micrograph_score,
        micrograph_consistency_score=micrograph_consistency_score,
        micrograph_density_score=micrograph_density_score,
        micrograph_temporal_persistence_score=micrograph_temporal_persistence_score,
        micrograph_decision=False,
    )


def aggregate_micrograph_decisions(
    verification_results: list[MicrographVerificationResult],
    selected_regions: list[CandidateRegion],
    *,
    final_decision_mode: FinalDecisionMode,
    total_edge_count: int,
    score_threshold: float,
    evidence_threshold: float | None = None,
) -> MicrographFinalDecision:
    """Aggregate candidate-level verifier outputs into one graph decision."""

    if not verification_results:
        return MicrographFinalDecision(
            final_decision_mode=final_decision_mode,
            decision_score=0.0,
            positive_evidence_count=0,
            selected_region_count=0,
            selected_region_coverage=0.0,
            is_positive=False,
        )
    region_scores = sorted((item.micrograph_score for item in verification_results), reverse=True)
    if final_decision_mode == "max_micrograph_score":
        decision_score = float(region_scores[0])
    elif final_decision_mode == "top2_micrograph_mean":
        decision_score = float(np.mean(region_scores[: min(2, len(region_scores))]))
    else:
        top2_mean = float(np.mean(region_scores[: min(2, len(region_scores))]))
        selected_edge_indices = {
            relative_index
            for region in selected_regions
            for relative_index in region.relative_edge_indices
        }
        coverage = len(selected_edge_indices) / max(total_edge_count, 1)
        decision_score = float(0.6 * region_scores[0] + 0.25 * top2_mean + 0.15 * coverage)
    effective_evidence_threshold = score_threshold if evidence_threshold is None else evidence_threshold
    positive_evidence_count = sum(item.micrograph_score >= effective_evidence_threshold for item in verification_results)
    selected_edge_indices = {
        relative_index
        for region in selected_regions
        for relative_index in region.relative_edge_indices
    }
    selected_region_coverage = len(selected_edge_indices) / max(total_edge_count, 1)
    is_positive = decision_score >= score_threshold and positive_evidence_count >= 1
    return MicrographFinalDecision(
        final_decision_mode=final_decision_mode,
        decision_score=decision_score,
        positive_evidence_count=positive_evidence_count,
        selected_region_count=len(selected_regions),
        selected_region_coverage=float(selected_region_coverage),
        is_positive=is_positive,
    )


__all__ = [
    "FinalDecisionMode",
    "MicrographFinalDecision",
    "MicrographVerificationResult",
    "MicrographVerifierMode",
    "aggregate_micrograph_decisions",
    "verify_candidate_region",
]
