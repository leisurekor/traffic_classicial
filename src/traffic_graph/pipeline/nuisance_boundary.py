"""Nuisance-aware graph-level boundary for single-stage edge-centric detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .edge_calibration import EdgeCalibrationDecision, EdgeGraphScoreBreakdown, suppressed_graph_score

InternalState = Literal["benign_like", "nuisance_like", "malicious_like"]


@dataclass(frozen=True, slots=True)
class NuisanceBoundaryProfile:
    """Compact nuisance-aware decision profile."""

    name: str
    nuisance_percentile: float
    nuisance_penalty: float


@dataclass(frozen=True, slots=True)
class NuisanceBoundaryDecision:
    """Benign and nuisance boundaries estimated without malicious supervision."""

    profile_name: str
    base_decision: EdgeCalibrationDecision
    benign_boundary_value: float
    nuisance_boundary_value: float
    malicious_support_threshold: float
    nuisance_penalty: float
    nuisance_prototype_mean: tuple[float, ...]
    nuisance_prototype_scale: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class NuisanceAwareGraphScore:
    """Three-state internal score decomposition for one graph."""

    anomaly_score: float
    nuisance_score: float
    malicious_support_score: float
    benign_boundary_value: float
    nuisance_boundary_value: float
    final_internal_state: InternalState
    final_binary_decision: int


def default_nuisance_boundary_profiles() -> tuple[NuisanceBoundaryProfile, ...]:
    """Return a small fixed profile set for nuisance-aware calibration."""

    return (
        NuisanceBoundaryProfile(
            name="nuisance_q90_margin025",
            nuisance_percentile=90.0,
            nuisance_penalty=0.25,
        ),
        NuisanceBoundaryProfile(
            name="nuisance_q95_margin025",
            nuisance_percentile=95.0,
            nuisance_penalty=0.25,
        ),
        NuisanceBoundaryProfile(
            name="nuisance_q95_margin050",
            nuisance_percentile=95.0,
            nuisance_penalty=0.50,
        ),
    )


def _feature_vector(breakdown: EdgeGraphScoreBreakdown, anomaly_score: float) -> np.ndarray:
    return np.asarray(
        [
            float(anomaly_score),
            float(breakdown.topk_mean),
            float(breakdown.local_support_score),
            float(breakdown.neighborhood_persistence_score),
            float(breakdown.temporal_consistency_score),
            float(breakdown.abnormal_edge_concentration),
            float(breakdown.local_support_edge_density),
            float(breakdown.cross_neighborhood_support_ratio),
        ],
        dtype=float,
    )


def nuisance_affinity_score(
    breakdown: EdgeGraphScoreBreakdown,
    decision: NuisanceBoundaryDecision,
) -> float:
    """Return a prototype-similarity score; higher means more nuisance-like."""

    vector = _feature_vector(breakdown, suppressed_graph_score(breakdown, decision.base_decision))
    mean = np.asarray(decision.nuisance_prototype_mean, dtype=float)
    scale = np.asarray(decision.nuisance_prototype_scale, dtype=float)
    normalized_gap = np.abs(vector - mean) / scale
    return float(np.mean(np.exp(-normalized_gap)))


def malicious_support_score(
    breakdown: EdgeGraphScoreBreakdown,
    decision: NuisanceBoundaryDecision,
) -> float:
    """Return a malicious-support score that still uses only benign/nuisance calibration."""

    anomaly_score = suppressed_graph_score(breakdown, decision.base_decision)
    anomaly_margin = max(anomaly_score - decision.benign_boundary_value, 0.0) / max(
        abs(decision.benign_boundary_value),
        1e-12,
    )
    support_strength = (
        0.40 * float(breakdown.local_support_score)
        + 0.30 * float(breakdown.temporal_consistency_score)
        + 0.20 * float(breakdown.neighborhood_persistence_score)
        + 0.10 * float(breakdown.topk_mean / max(breakdown.top1_edge_score, 1e-12))
    )
    return float(0.55 * min(anomaly_margin, 2.0) / 2.0 + 0.45 * support_strength)


def calibrate_nuisance_boundary(
    base_decision: EdgeCalibrationDecision,
    benign_breakdowns: list[EdgeGraphScoreBreakdown],
    nuisance_breakdowns: list[EdgeGraphScoreBreakdown],
    *,
    profile: NuisanceBoundaryProfile,
) -> NuisanceBoundaryDecision:
    """Estimate a nuisance-aware decision from benign + nuisance graphs only."""

    benign_scores = np.asarray(
        [suppressed_graph_score(item, base_decision) for item in benign_breakdowns],
        dtype=float,
    )
    benign_boundary = (
        float(np.percentile(benign_scores, 95.0))
        if benign_scores.size
        else float(base_decision.score_threshold)
    )
    if nuisance_breakdowns:
        nuisance_vectors = np.asarray(
            [
                _feature_vector(item, suppressed_graph_score(item, base_decision))
                for item in nuisance_breakdowns
            ],
            dtype=float,
        )
        nuisance_mean = nuisance_vectors.mean(axis=0)
        nuisance_scale = np.maximum(nuisance_vectors.std(axis=0), 1e-6)
        provisional_decision = NuisanceBoundaryDecision(
            profile_name=profile.name,
            base_decision=base_decision,
            benign_boundary_value=benign_boundary,
            nuisance_boundary_value=0.0,
            malicious_support_threshold=0.0,
            nuisance_penalty=profile.nuisance_penalty,
            nuisance_prototype_mean=tuple(float(value) for value in nuisance_mean.tolist()),
            nuisance_prototype_scale=tuple(float(value) for value in nuisance_scale.tolist()),
        )
        nuisance_scores = np.asarray(
            [nuisance_affinity_score(item, provisional_decision) for item in nuisance_breakdowns],
            dtype=float,
        )
    else:
        nuisance_mean = np.zeros((8,), dtype=float)
        nuisance_scale = np.ones((8,), dtype=float)
        nuisance_scores = np.zeros((0,), dtype=float)
    nuisance_boundary_value = (
        float(np.percentile(nuisance_scores, profile.nuisance_percentile))
        if nuisance_scores.size
        else 1.0
    )
    benign_support_scores = np.asarray(
        [
            malicious_support_score(
                item,
                NuisanceBoundaryDecision(
                    profile_name=profile.name,
                    base_decision=base_decision,
                    benign_boundary_value=benign_boundary,
                    nuisance_boundary_value=nuisance_boundary_value,
                    malicious_support_threshold=0.0,
                    nuisance_penalty=profile.nuisance_penalty,
                    nuisance_prototype_mean=tuple(float(value) for value in nuisance_mean.tolist()),
                    nuisance_prototype_scale=tuple(float(value) for value in nuisance_scale.tolist()),
                ),
            )
            for item in benign_breakdowns
        ],
        dtype=float,
    )
    malicious_support_threshold = (
        float(np.percentile(benign_support_scores, 95.0))
        if benign_support_scores.size
        else 0.0
    )
    return NuisanceBoundaryDecision(
        profile_name=profile.name,
        base_decision=base_decision,
        benign_boundary_value=benign_boundary,
        nuisance_boundary_value=nuisance_boundary_value,
        malicious_support_threshold=malicious_support_threshold,
        nuisance_penalty=profile.nuisance_penalty,
        nuisance_prototype_mean=tuple(float(value) for value in nuisance_mean.tolist()),
        nuisance_prototype_scale=tuple(float(value) for value in nuisance_scale.tolist()),
    )


def score_graph_nuisance_aware(
    breakdown: EdgeGraphScoreBreakdown,
    decision: NuisanceBoundaryDecision,
) -> NuisanceAwareGraphScore:
    """Score one graph under the nuisance-aware three-state decision."""

    anomaly_score = suppressed_graph_score(breakdown, decision.base_decision)
    nuisance_score = nuisance_affinity_score(breakdown, decision)
    support_score = malicious_support_score(breakdown, decision)
    effective_support = support_score - decision.nuisance_penalty * nuisance_score
    if anomaly_score < decision.benign_boundary_value:
        final_internal_state: InternalState = "benign_like"
    elif nuisance_score >= decision.nuisance_boundary_value and (
        effective_support < decision.malicious_support_threshold
    ):
        final_internal_state = "nuisance_like"
    elif effective_support >= decision.malicious_support_threshold:
        final_internal_state = "malicious_like"
    else:
        final_internal_state = "nuisance_like"
    return NuisanceAwareGraphScore(
        anomaly_score=float(anomaly_score),
        nuisance_score=float(nuisance_score),
        malicious_support_score=float(support_score),
        benign_boundary_value=float(decision.benign_boundary_value),
        nuisance_boundary_value=float(decision.nuisance_boundary_value),
        final_internal_state=final_internal_state,
        final_binary_decision=1 if final_internal_state == "malicious_like" else 0,
    )

