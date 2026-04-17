"""Episode-centric graph builder for nuisance-aware detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from traffic_graph.pipeline.episode_proposal import Episode


@dataclass(frozen=True, slots=True)
class EpisodeGraph:
    """Lightweight episode graph summary with episode nodes as the center object."""

    graph_id: str
    episode_ids: tuple[str, ...]
    endpoint_ids: tuple[str, ...]
    participation_edge_count: int
    temporal_adjacency_edge_count: int
    repeated_pair_similarity_edge_count: int
    support_cluster_similarity_edge_count: int
    nuisance_heavy_episode_count: int
    covered_edge_count: int

    @property
    def episode_count(self) -> int:
        return len(self.episode_ids)

    @property
    def endpoint_count(self) -> int:
        return len(self.endpoint_ids)


def _time_gap_seconds(episode_a: Episode, episode_b: Episode) -> float | None:
    if episode_a.end_time is None or episode_b.start_time is None:
        return None
    if episode_b.start_time >= episode_a.end_time:
        return float((episode_b.start_time - episode_a.end_time).total_seconds())
    if episode_a.start_time is None or episode_b.end_time is None:
        return 0.0
    if episode_a.start_time >= episode_b.end_time:
        return float((episode_a.start_time - episode_b.end_time).total_seconds())
    return 0.0


def build_episode_graph(
    sample: Any,
    episodes: list[Episode],
    *,
    nuisance_threshold: float = 0.6,
) -> EpisodeGraph:
    """Build a lightweight episode graph from proposed episodes."""

    endpoint_ids = sorted({endpoint_id for episode in episodes for endpoint_id in episode.involved_endpoints})
    window_seconds = max((sample.graph.window_end - sample.graph.window_start).total_seconds(), 1e-6)
    temporal_adjacency_edge_count = 0
    repeated_pair_similarity_edge_count = 0
    support_cluster_similarity_edge_count = 0
    for left_index, episode_a in enumerate(episodes):
        for episode_b in episodes[left_index + 1 :]:
            shared_endpoints = set(episode_a.involved_endpoints).intersection(episode_b.involved_endpoints)
            if shared_endpoints:
                repeated_pair_similarity_edge_count += 1
            if (
                episode_a.support_cluster_density >= 0.4
                and episode_b.support_cluster_density >= 0.4
                and shared_endpoints
            ):
                support_cluster_similarity_edge_count += 1
            gap_seconds = _time_gap_seconds(episode_a, episode_b)
            if gap_seconds is not None and gap_seconds <= max(window_seconds / 4.0, 1.0):
                temporal_adjacency_edge_count += 1
    covered_edge_count = len({edge_index for episode in episodes for edge_index in episode.relative_edge_indices})
    return EpisodeGraph(
        graph_id=f"{getattr(sample, 'scenario_id', 'unknown')}:{getattr(sample, 'window_index', -1)}:{getattr(sample, 'group_key', 'graph')}",
        episode_ids=tuple(episode.episode_id for episode in episodes),
        endpoint_ids=tuple(endpoint_ids),
        participation_edge_count=sum(len(episode.involved_endpoints) for episode in episodes),
        temporal_adjacency_edge_count=temporal_adjacency_edge_count,
        repeated_pair_similarity_edge_count=repeated_pair_similarity_edge_count,
        support_cluster_similarity_edge_count=support_cluster_similarity_edge_count,
        nuisance_heavy_episode_count=sum(episode.nuisance_likelihood >= nuisance_threshold for episode in episodes),
        covered_edge_count=covered_edge_count,
    )


__all__ = [
    "EpisodeGraph",
    "build_episode_graph",
]
