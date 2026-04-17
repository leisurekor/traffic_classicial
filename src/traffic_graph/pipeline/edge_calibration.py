"""Benign-side calibration helpers for edge-centric binary decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

SupportSummaryMode = Literal[
    "old_concentration",
    "local_support_density",
    "combined_support_summary",
]


@dataclass(frozen=True, slots=True)
class EdgeCalibrationProfile:
    """One explicit edge-centric calibration profile."""

    name: str
    percentile: float
    top_k: int
    concentration_percentile: float | None = None
    require_dual_threshold: bool = False


@dataclass(frozen=True, slots=True)
class EdgeGraphScoreBreakdown:
    """Graph-level edge anomaly summary before final thresholding."""

    graph_score: float
    top1_edge_score: float
    topk_mean: float
    abnormal_edge_count: int
    abnormal_edge_density: float
    abnormal_edge_concentration: float
    max_component_abnormal_ratio: float
    max_server_neighborhood_abnormal_ratio: float
    component_peak: float
    neighborhood_peak: float
    concentration_score: float
    local_support_edge_count: int = 0
    local_support_edge_density: float = 0.0
    local_support_node_coverage: float = 0.0
    max_local_support_density: float = 0.0
    top_local_support_component_size: int = 0
    abnormal_neighborhood_count: int = 0
    abnormal_neighborhood_entropy: float = 0.0
    cross_neighborhood_support_ratio: float = 0.0
    repeated_abnormal_endpoints: int = 0
    slice_abnormal_presence_count: int = 0
    slice_abnormal_consistency_ratio: float = 0.0
    slice_topk_overlap_ratio: float = 0.0
    slice_repeated_support_endpoints: int = 0
    local_support_score: float = 0.0
    neighborhood_persistence_score: float = 0.0
    temporal_consistency_score: float = 0.0


@dataclass(frozen=True, slots=True)
class EdgeCalibrationDecision:
    """Thresholds estimated from held-out benign graph scores."""

    profile_name: str
    score_threshold: float
    concentration_threshold: float | None
    suppression_enabled: bool = False
    component_ratio_threshold: float | None = None
    neighborhood_ratio_threshold: float | None = None
    density_threshold: float | None = None
    min_abnormal_edge_count: int = 1
    support_summary_mode: SupportSummaryMode = "old_concentration"
    local_density_threshold: float | None = None
    neighborhood_persistence_threshold: float | None = None
    temporal_consistency_threshold: float | None = None
    require_any_support_signal: bool = False


@dataclass(frozen=True, slots=True)
class LocalSupportSummary:
    """Compact local-subgraph support summary for graph-level decisions."""

    local_support_edge_count: int
    local_support_edge_density: float
    local_support_node_coverage: float
    max_local_support_density: float
    top_local_support_component_size: int
    local_support_score: float


@dataclass(frozen=True, slots=True)
class NeighborhoodPersistenceSummary:
    """Cross-neighborhood persistence summary for high-score support edges."""

    abnormal_neighborhood_count: int
    abnormal_neighborhood_entropy: float
    cross_neighborhood_support_ratio: float
    repeated_abnormal_endpoints: int
    neighborhood_persistence_score: float


@dataclass(frozen=True, slots=True)
class TemporalConsistencySummary:
    """Short-slice temporal consistency summary for graph-level support."""

    slice_abnormal_presence_count: int
    slice_abnormal_consistency_ratio: float
    slice_topk_overlap_ratio: float
    slice_repeated_support_endpoints: int
    temporal_consistency_score: float


def _suppression_reference_percentile(profile: EdgeCalibrationProfile) -> float:
    if profile.concentration_percentile is not None:
        return float(profile.concentration_percentile)
    return min(max(profile.percentile - 10.0, 80.0), 95.0)


def suppressed_graph_score(
    breakdown: EdgeGraphScoreBreakdown,
    decision: EdgeCalibrationDecision,
) -> float:
    """Downweight sparse, weakly concentrated anomaly spikes."""

    if not decision.suppression_enabled:
        return float(breakdown.graph_score)
    min_support = max(decision.min_abnormal_edge_count, 1)
    support_ratio = min(float(breakdown.abnormal_edge_count) / float(min_support), 1.0)
    locality_terms: list[float] = []
    if decision.concentration_threshold not in {None, 0.0}:
        locality_terms.append(
            min(
                float(breakdown.abnormal_edge_concentration) / float(decision.concentration_threshold),
                1.0,
            )
        )
    if decision.component_ratio_threshold not in {None, 0.0}:
        locality_terms.append(
            min(
                float(breakdown.max_component_abnormal_ratio) / float(decision.component_ratio_threshold),
                1.0,
            )
        )
    if decision.neighborhood_ratio_threshold not in {None, 0.0}:
        locality_terms.append(
            min(
                float(breakdown.max_server_neighborhood_abnormal_ratio)
                / float(decision.neighborhood_ratio_threshold),
                1.0,
            )
        )
    if decision.density_threshold not in {None, 0.0}:
        locality_terms.append(
            min(
                float(breakdown.abnormal_edge_density) / float(decision.density_threshold),
                1.0,
            )
        )
    locality_ratio = max(locality_terms, default=1.0)
    suppression_multiplier = max(0.2, support_ratio * locality_ratio)
    return float(breakdown.graph_score * suppression_multiplier)


def default_edge_calibration_profiles() -> tuple[EdgeCalibrationProfile, ...]:
    """Return a compact, fixed set of benign-only calibration profiles."""

    return (
        EdgeCalibrationProfile(
            name="heldout_q95_top1",
            percentile=95.0,
            top_k=1,
        ),
        EdgeCalibrationProfile(
            name="heldout_q97_top3",
            percentile=97.0,
            top_k=3,
        ),
        EdgeCalibrationProfile(
            name="heldout_q99_top5",
            percentile=99.0,
            top_k=5,
        ),
        EdgeCalibrationProfile(
            name="dual_q97_top5_conc90",
            percentile=97.0,
            top_k=5,
            concentration_percentile=90.0,
            require_dual_threshold=True,
        ),
        EdgeCalibrationProfile(
            name="dual_q99_top10_conc95",
            percentile=99.0,
            top_k=10,
            concentration_percentile=95.0,
            require_dual_threshold=True,
        ),
    )


def _with_support_mode(
    decision: EdgeCalibrationDecision,
    *,
    support_summary_mode: SupportSummaryMode,
    local_density_threshold: float | None,
    neighborhood_persistence_threshold: float | None,
    temporal_consistency_threshold: float | None,
    require_any_support_signal: bool,
) -> EdgeCalibrationDecision:
    return EdgeCalibrationDecision(
        profile_name=decision.profile_name,
        score_threshold=decision.score_threshold,
        concentration_threshold=decision.concentration_threshold,
        suppression_enabled=decision.suppression_enabled,
        component_ratio_threshold=decision.component_ratio_threshold,
        neighborhood_ratio_threshold=decision.neighborhood_ratio_threshold,
        density_threshold=decision.density_threshold,
        min_abnormal_edge_count=decision.min_abnormal_edge_count,
        support_summary_mode=support_summary_mode,
        local_density_threshold=local_density_threshold,
        neighborhood_persistence_threshold=neighborhood_persistence_threshold,
        temporal_consistency_threshold=temporal_consistency_threshold,
        require_any_support_signal=require_any_support_signal,
    )


def calibrate_edge_profile(
    profile: EdgeCalibrationProfile,
    benign_breakdowns: list[EdgeGraphScoreBreakdown],
    *,
    suppression_enabled: bool = False,
) -> EdgeCalibrationDecision:
    """Estimate thresholds from held-out benign graph breakdowns only."""

    min_abnormal_edge_count = 2 if suppression_enabled else 1
    if not benign_breakdowns:
        return EdgeCalibrationDecision(
            profile_name=profile.name,
            score_threshold=0.0,
            concentration_threshold=None if not (profile.require_dual_threshold or suppression_enabled) else 0.0,
            suppression_enabled=suppression_enabled,
            component_ratio_threshold=None,
            neighborhood_ratio_threshold=None,
            density_threshold=None,
            min_abnormal_edge_count=min_abnormal_edge_count,
        )
    provisional_decision = EdgeCalibrationDecision(
        profile_name=profile.name,
        score_threshold=0.0,
        concentration_threshold=1.0,
        suppression_enabled=suppression_enabled,
        component_ratio_threshold=1.0,
        neighborhood_ratio_threshold=1.0,
        density_threshold=1.0,
        min_abnormal_edge_count=min_abnormal_edge_count,
    )
    score_values = np.asarray(
        [
            suppressed_graph_score(item, provisional_decision)
            if suppression_enabled
            else item.graph_score
            for item in benign_breakdowns
        ],
        dtype=float,
    )
    score_threshold = float(np.percentile(score_values, profile.percentile))
    reference_percentile = _suppression_reference_percentile(profile)
    local_density_threshold = float(
        np.percentile(
            np.asarray([item.local_support_score for item in benign_breakdowns], dtype=float),
            reference_percentile,
        )
    )
    neighborhood_persistence_threshold = float(
        np.percentile(
            np.asarray([item.neighborhood_persistence_score for item in benign_breakdowns], dtype=float),
            reference_percentile,
        )
    )
    temporal_consistency_threshold = float(
        np.percentile(
            np.asarray([item.temporal_consistency_score for item in benign_breakdowns], dtype=float),
            reference_percentile,
        )
    )
    if not suppression_enabled and (
        not profile.require_dual_threshold or profile.concentration_percentile is None
    ):
        return EdgeCalibrationDecision(
            profile_name=profile.name,
            score_threshold=score_threshold,
            concentration_threshold=None,
            suppression_enabled=False,
            component_ratio_threshold=None,
            neighborhood_ratio_threshold=None,
            density_threshold=None,
            min_abnormal_edge_count=min_abnormal_edge_count,
            local_density_threshold=local_density_threshold,
            neighborhood_persistence_threshold=neighborhood_persistence_threshold,
            temporal_consistency_threshold=temporal_consistency_threshold,
        )
    concentration_values = np.asarray(
        [
            item.abnormal_edge_concentration
            if suppression_enabled
            else item.concentration_score
            for item in benign_breakdowns
        ],
        dtype=float,
    )
    concentration_threshold = float(
        np.percentile(
            concentration_values,
            profile.concentration_percentile
            if profile.concentration_percentile is not None
            else reference_percentile,
        )
    )
    component_ratio_threshold = float(
        np.percentile(
            np.asarray([item.max_component_abnormal_ratio for item in benign_breakdowns], dtype=float),
            reference_percentile,
        )
    )
    neighborhood_ratio_threshold = float(
        np.percentile(
            np.asarray(
                [item.max_server_neighborhood_abnormal_ratio for item in benign_breakdowns],
                dtype=float,
            ),
            reference_percentile,
        )
    )
    density_threshold = float(
        np.percentile(
            np.asarray([item.abnormal_edge_density for item in benign_breakdowns], dtype=float),
            reference_percentile,
        )
    )
    return EdgeCalibrationDecision(
        profile_name=profile.name,
        score_threshold=score_threshold,
        concentration_threshold=concentration_threshold,
        suppression_enabled=suppression_enabled,
        component_ratio_threshold=component_ratio_threshold,
        neighborhood_ratio_threshold=neighborhood_ratio_threshold,
        density_threshold=density_threshold,
        min_abnormal_edge_count=min_abnormal_edge_count,
        local_density_threshold=local_density_threshold,
        neighborhood_persistence_threshold=neighborhood_persistence_threshold,
        temporal_consistency_threshold=temporal_consistency_threshold,
    )


def build_support_summary_aware_decision(
    profile: EdgeCalibrationProfile,
    benign_breakdowns: list[EdgeGraphScoreBreakdown],
    *,
    support_summary_mode: SupportSummaryMode,
) -> EdgeCalibrationDecision:
    """Estimate a benign-only decision that gates on support-summary evidence."""

    base_decision = calibrate_edge_profile(profile, benign_breakdowns, suppression_enabled=False)
    if support_summary_mode == "local_support_density":
        return _with_support_mode(
            base_decision,
            support_summary_mode=support_summary_mode,
            local_density_threshold=base_decision.local_density_threshold,
            neighborhood_persistence_threshold=None,
            temporal_consistency_threshold=None,
            require_any_support_signal=False,
        )
    return _with_support_mode(
        base_decision,
        support_summary_mode=support_summary_mode,
        local_density_threshold=base_decision.local_density_threshold,
        neighborhood_persistence_threshold=base_decision.neighborhood_persistence_threshold,
        temporal_consistency_threshold=base_decision.temporal_consistency_threshold,
        require_any_support_signal=(support_summary_mode == "combined_support_summary"),
    )


def _has_support_signal(
    breakdown: EdgeGraphScoreBreakdown,
    decision: EdgeCalibrationDecision,
) -> bool:
    signal_hits = 0
    if (
        decision.local_density_threshold is not None
        and breakdown.local_support_score >= decision.local_density_threshold
    ):
        signal_hits += 1
    if (
        decision.neighborhood_persistence_threshold is not None
        and breakdown.neighborhood_persistence_score >= decision.neighborhood_persistence_threshold
    ):
        signal_hits += 1
    if (
        decision.temporal_consistency_threshold is not None
        and breakdown.temporal_consistency_score >= decision.temporal_consistency_threshold
    ):
        signal_hits += 1
    if decision.support_summary_mode == "local_support_density":
        return signal_hits >= 1
    if decision.require_any_support_signal:
        return signal_hits >= 1
    return True


def apply_edge_calibration(
    breakdowns: list[EdgeGraphScoreBreakdown],
    decision: EdgeCalibrationDecision,
) -> list[int]:
    """Convert graph breakdowns into binary anomaly decisions."""

    predictions: list[int] = []
    for breakdown in breakdowns:
        calibrated_score = suppressed_graph_score(breakdown, decision)
        is_positive = calibrated_score >= decision.score_threshold
        if is_positive and decision.support_summary_mode != "old_concentration":
            is_positive = (
                breakdown.abnormal_edge_count >= decision.min_abnormal_edge_count
                and _has_support_signal(breakdown, decision)
            )
        elif (
            is_positive
            and decision.concentration_threshold is not None
            and (
                (
                    decision.suppression_enabled
                    and breakdown.abnormal_edge_concentration < decision.concentration_threshold
                )
                or (
                    not decision.suppression_enabled
                    and breakdown.concentration_score < decision.concentration_threshold
                )
            )
        ):
            is_positive = False
        predictions.append(1 if is_positive else 0)
    return predictions


__all__ = [
    "LocalSupportSummary",
    "NeighborhoodPersistenceSummary",
    "TemporalConsistencySummary",
    "EdgeCalibrationDecision",
    "EdgeCalibrationProfile",
    "EdgeGraphScoreBreakdown",
    "apply_edge_calibration",
    "build_support_summary_aware_decision",
    "calibrate_edge_profile",
    "default_edge_calibration_profiles",
    "suppressed_graph_score",
]
