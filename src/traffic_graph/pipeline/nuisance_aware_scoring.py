"""Nuisance-aware episode scoring and graph-level aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from traffic_graph.pipeline.episode_graph_builder import EpisodeGraph
from traffic_graph.pipeline.episode_proposal import Episode

NuisanceAwareMode = Literal["double_boundary_v1"]
EpisodeDecisionMode = Literal[
    "max_episode_score",
    "top2_episode_mean",
    "consistency_aware_episode",
]


@dataclass(frozen=True, slots=True)
class NuisanceAwareEpisodeScore:
    """Episode-level nuisance-aware score bundle."""

    episode_id: str
    proposal_mode: str
    anomaly_score: float
    nuisance_score: float
    malicious_consistency_score: float
    episode_decision: bool
    nuisance_like: bool


@dataclass(frozen=True, slots=True)
class NuisanceAwareCalibration:
    """Unsupervised calibration with explicit nuisance rejection boundary."""

    nuisance_mode: NuisanceAwareMode
    anomaly_threshold: float
    nuisance_threshold: float
    consistency_threshold: float


@dataclass(frozen=True, slots=True)
class NuisanceAwareGraphDecision:
    """Graph-level decision aggregated from episode evidence."""

    final_decision_mode: EpisodeDecisionMode
    decision_score: float
    positive_evidence_count: int
    selected_episode_count: int
    selected_episode_coverage: float
    nuisance_rejection_rate: float
    is_positive: bool


def score_episode_nuisance_aware(
    episode: Episode,
    episode_graph: EpisodeGraph,
    *,
    nuisance_mode: NuisanceAwareMode = "double_boundary_v1",
) -> NuisanceAwareEpisodeScore:
    """Score one episode with explicit anomaly and nuisance boundaries."""

    anomaly_score = float(
        0.35 * episode.proposal_score
        + 0.20 * min(episode.repeated_pair_count / 2.0, 1.0)
        + 0.20 * episode.burst_persistence
        + 0.15 * episode.support_cluster_density
        + 0.10 * min(episode.edge_count / 4.0, 1.0)
    )
    graph_dispersion_penalty = float(
        0.5 * min(episode_graph.nuisance_heavy_episode_count / max(episode_graph.episode_count, 1), 1.0)
        + 0.5 * min(episode.endpoint_span / max(episode.edge_count, 1), 1.0)
    )
    nuisance_score = float(
        0.55 * episode.nuisance_likelihood
        + 0.20 * graph_dispersion_penalty
        + 0.15 * min(episode.target_count / max(episode.edge_count, 1), 1.0)
        + 0.10 * max(0.0, 1.0 - min(episode.repeated_pair_count / 2.0, 1.0))
    )
    malicious_consistency_score = float(
        0.60 * anomaly_score
        + 0.25 * max(0.0, 1.0 - nuisance_score)
        + 0.15 * min(
            (episode_graph.temporal_adjacency_edge_count + episode_graph.repeated_pair_similarity_edge_count)
            / max(episode_graph.episode_count, 1),
            1.0,
        )
    )
    return NuisanceAwareEpisodeScore(
        episode_id=episode.episode_id,
        proposal_mode=episode.proposal_mode,
        anomaly_score=anomaly_score,
        nuisance_score=nuisance_score,
        malicious_consistency_score=malicious_consistency_score,
        episode_decision=False,
        nuisance_like=False,
    )


def calibrate_nuisance_aware_scores(
    benign_episode_scores: list[NuisanceAwareEpisodeScore],
    nuisance_episode_scores: list[NuisanceAwareEpisodeScore],
    *,
    percentile: float,
    nuisance_mode: NuisanceAwareMode = "double_boundary_v1",
) -> NuisanceAwareCalibration:
    """Calibrate a double-boundary detector from benign and nuisance episodes."""

    benign_anomaly = np.asarray([item.anomaly_score for item in benign_episode_scores], dtype=float)
    benign_nuisance = np.asarray([item.nuisance_score for item in benign_episode_scores], dtype=float)
    benign_consistency = np.asarray([item.malicious_consistency_score for item in benign_episode_scores], dtype=float)
    nuisance_nuisance = np.asarray([item.nuisance_score for item in nuisance_episode_scores], dtype=float)

    anomaly_threshold = float(np.percentile(benign_anomaly, percentile)) if benign_anomaly.size else 1.0
    consistency_threshold = float(np.percentile(benign_consistency, percentile)) if benign_consistency.size else 1.0
    benign_nuisance_ref = float(np.percentile(benign_nuisance, 80.0)) if benign_nuisance.size else 0.5
    nuisance_nuisance_ref = float(np.percentile(nuisance_nuisance, 35.0)) if nuisance_nuisance.size else benign_nuisance_ref
    nuisance_threshold = float((benign_nuisance_ref + max(benign_nuisance_ref, nuisance_nuisance_ref)) / 2.0)
    return NuisanceAwareCalibration(
        nuisance_mode=nuisance_mode,
        anomaly_threshold=anomaly_threshold,
        nuisance_threshold=nuisance_threshold,
        consistency_threshold=consistency_threshold,
    )


def relabel_episode_scores(
    episode_scores: list[NuisanceAwareEpisodeScore],
    calibration: NuisanceAwareCalibration,
) -> list[NuisanceAwareEpisodeScore]:
    """Apply calibrated double-boundary labels to episode scores."""

    relabeled: list[NuisanceAwareEpisodeScore] = []
    for item in episode_scores:
        nuisance_like = item.nuisance_score >= calibration.nuisance_threshold
        episode_decision = (
            item.anomaly_score >= calibration.anomaly_threshold
            and item.malicious_consistency_score >= calibration.consistency_threshold
            and not nuisance_like
        )
        relabeled.append(
            NuisanceAwareEpisodeScore(
                episode_id=item.episode_id,
                proposal_mode=item.proposal_mode,
                anomaly_score=item.anomaly_score,
                nuisance_score=item.nuisance_score,
                malicious_consistency_score=item.malicious_consistency_score,
                episode_decision=episode_decision,
                nuisance_like=nuisance_like,
            )
        )
    return relabeled


def aggregate_episode_graph_decision(
    episode_scores: list[NuisanceAwareEpisodeScore],
    episodes: list[Episode],
    *,
    final_decision_mode: EpisodeDecisionMode,
    total_edge_count: int,
    graph_threshold: float,
) -> NuisanceAwareGraphDecision:
    """Aggregate nuisance-aware episode scores into one graph decision."""

    if not episode_scores:
        return NuisanceAwareGraphDecision(
            final_decision_mode=final_decision_mode,
            decision_score=0.0,
            positive_evidence_count=0,
            selected_episode_count=0,
            selected_episode_coverage=0.0,
            nuisance_rejection_rate=0.0,
            is_positive=False,
        )
    ranked_scores = sorted((item.malicious_consistency_score for item in episode_scores), reverse=True)
    if final_decision_mode == "max_episode_score":
        decision_score = float(ranked_scores[0])
    elif final_decision_mode == "top2_episode_mean":
        decision_score = float(np.mean(ranked_scores[: min(2, len(ranked_scores))]))
    else:
        selected = [episode for episode, score in zip(episodes, episode_scores, strict=True) if score.episode_decision]
        coverage = len({index for episode in selected for index in episode.relative_edge_indices}) / max(total_edge_count, 1)
        decision_score = float(0.55 * ranked_scores[0] + 0.30 * np.mean(ranked_scores[: min(2, len(ranked_scores))]) + 0.15 * coverage)
    positive_evidence_count = sum(item.episode_decision for item in episode_scores)
    selected_episodes = [episode for episode, score in zip(episodes, episode_scores, strict=True) if score.episode_decision]
    selected_coverage = len({index for episode in selected_episodes for index in episode.relative_edge_indices}) / max(total_edge_count, 1)
    nuisance_rejection_rate = float(sum(item.nuisance_like for item in episode_scores) / len(episode_scores))
    return NuisanceAwareGraphDecision(
        final_decision_mode=final_decision_mode,
        decision_score=decision_score,
        positive_evidence_count=positive_evidence_count,
        selected_episode_count=len(selected_episodes),
        selected_episode_coverage=float(selected_coverage),
        nuisance_rejection_rate=nuisance_rejection_rate,
        is_positive=decision_score >= graph_threshold and positive_evidence_count >= 1,
    )


__all__ = [
    "EpisodeDecisionMode",
    "NuisanceAwareCalibration",
    "NuisanceAwareEpisodeScore",
    "NuisanceAwareGraphDecision",
    "NuisanceAwareMode",
    "aggregate_episode_graph_decision",
    "calibrate_nuisance_aware_scores",
    "relabel_episode_scores",
    "score_episode_nuisance_aware",
]
