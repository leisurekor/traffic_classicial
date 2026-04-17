"""Episode proposal helpers for episode-first nuisance-aware detection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import numpy as np

EpisodeProposalMode = Literal[
    "repeated_pair_episode",
    "local_burst_episode",
    "support_cluster_episode",
]


@dataclass(frozen=True, slots=True)
class Episode:
    """One local, temporally coherent behavior fragment."""

    episode_id: str
    proposal_mode: EpisodeProposalMode
    relative_edge_indices: tuple[int, ...]
    involved_flow_ids: tuple[str, ...]
    involved_endpoints: tuple[str, ...]
    start_time: datetime | None
    end_time: datetime | None
    repeated_pair_count: int
    burst_persistence: float
    support_cluster_density: float
    nuisance_likelihood: float
    proposal_score: float
    repeated_slice_support_count: int
    endpoint_span: int
    source_count: int
    target_count: int
    stitching_mode: str = "proposal_seed"
    duration: float = 0.0
    gap_count: int = 0
    continuity_span: float = 0.0
    protocol_consistency_score: float = 0.0
    direction_pattern_consistency: float = 0.0
    merged_flow_count: int = 0
    service_chain_length: int = 0
    burst_repeat_count: int = 0
    burst_slice_span: int = 0
    burst_endpoint_overlap: float = 0.0

    @property
    def edge_count(self) -> int:
        return len(self.relative_edge_indices)

    @property
    def flow_count(self) -> int:
        return len(self.involved_flow_ids)

    @property
    def episode_time_span(self) -> float:
        if self.duration > 0.0:
            return float(self.duration)
        if self.start_time is None or self.end_time is None:
            return 0.0
        return float(max((self.end_time - self.start_time).total_seconds(), 0.0))


def _dst_subnet(ip: str) -> str:
    parts = ip.split(".")
    if len(parts) >= 3:
        return ".".join(parts[:3])
    return ip


def _flow_lookup(sample: Any) -> dict[str, Any]:
    return {str(flow.logical_flow_id): flow for flow in getattr(sample, "logical_flows", ())}


def _logical_flow_for_edge(edge: Any, flow_by_id: dict[str, Any]) -> Any | None:
    logical_flow = flow_by_id.get(str(edge.logical_flow_id))
    if logical_flow is not None:
        return logical_flow
    for source_flow_id in getattr(edge, "source_flow_ids", ()):
        logical_flow = flow_by_id.get(str(source_flow_id))
        if logical_flow is not None:
            return logical_flow
    return None


def _edge_midpoint_seconds(sample: Any, edge: Any, flow_by_id: dict[str, Any]) -> float:
    logical_flow = _logical_flow_for_edge(edge, flow_by_id)
    if logical_flow is None:
        return 0.0
    midpoint = logical_flow.start_time + (logical_flow.end_time - logical_flow.start_time) / 2
    return float((midpoint - sample.graph.window_start).total_seconds())


def _edge_slice_index(sample: Any, edge: Any, flow_by_id: dict[str, Any], *, slice_count: int) -> int:
    window_seconds = max((sample.graph.window_end - sample.graph.window_start).total_seconds(), 1e-6)
    midpoint_seconds = _edge_midpoint_seconds(sample, edge, flow_by_id)
    ratio = min(max(midpoint_seconds / window_seconds, 0.0), 0.999999)
    return min(int(ratio * slice_count), slice_count - 1)


def _episode_from_relative_indices(
    sample: Any,
    communication_indices: list[int],
    communication_scores: np.ndarray,
    relative_indices: set[int],
    *,
    proposal_mode: EpisodeProposalMode,
    episode_id: str,
    slice_count: int,
    flow_by_id: dict[str, Any],
) -> Episode | None:
    if len(relative_indices) < 2:
        return None
    ordered_relative_indices = tuple(sorted(relative_indices))
    involved_endpoints: set[str] = set()
    involved_flows: set[str] = set()
    source_ids: set[str] = set()
    target_ids: set[str] = set()
    pair_counts: dict[tuple[str, str], int] = {}
    slice_counts: dict[int, int] = {}
    start_time: datetime | None = None
    end_time: datetime | None = None
    for relative_index in ordered_relative_indices:
        edge = sample.graph.edges[communication_indices[relative_index]]
        involved_endpoints.update((edge.source_node_id, edge.target_node_id))
        source_ids.add(edge.source_node_id)
        target_ids.add(edge.target_node_id)
        pair_key = (edge.source_node_id, edge.target_node_id)
        pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
        slice_index = _edge_slice_index(sample, edge, flow_by_id, slice_count=slice_count)
        slice_counts[slice_index] = slice_counts.get(slice_index, 0) + 1
        logical_flow = _logical_flow_for_edge(edge, flow_by_id)
        if logical_flow is not None:
            involved_flows.add(str(logical_flow.logical_flow_id))
            start_time = logical_flow.start_time if start_time is None else min(start_time, logical_flow.start_time)
            end_time = logical_flow.end_time if end_time is None else max(end_time, logical_flow.end_time)
    scores = np.asarray([communication_scores[index] for index in ordered_relative_indices], dtype=float)
    repeated_pair_count = sum(count >= 2 for count in pair_counts.values())
    repeated_slice_support_count = sum(count >= 2 for count in slice_counts.values())
    burst_persistence = float(
        0.55 * (len(slice_counts) / max(slice_count, 1))
        + 0.45 * min(repeated_slice_support_count / max(len(slice_counts), 1), 1.0)
    )
    support_cluster_density = float(
        0.45 * min(len(ordered_relative_indices) / max(len(involved_endpoints), 1), 2.0) / 2.0
        + 0.30 * min(repeated_pair_count / 2.0, 1.0)
        + 0.25 * min(repeated_slice_support_count / 2.0, 1.0)
    )
    dominance_ratio = float(np.max(scores) / max(np.sum(scores), 1e-12))
    nuisance_likelihood = float(
        0.30 * max(0.0, 1.0 - min(repeated_pair_count / 2.0, 1.0))
        + 0.20 * max(0.0, 1.0 - burst_persistence)
        + 0.20 * max(0.0, 1.0 - support_cluster_density)
        + 0.15 * min(len(target_ids) / max(len(ordered_relative_indices), 1), 1.0)
        + 0.15 * dominance_ratio
    )
    proposal_score = float(
        0.45 * float(np.mean(scores))
        + 0.20 * float(np.max(scores))
        + 0.15 * min(repeated_pair_count / 2.0, 1.0)
        + 0.10 * burst_persistence
        + 0.10 * support_cluster_density
    )
    return Episode(
        episode_id=episode_id,
        proposal_mode=proposal_mode,
        stitching_mode="proposal_seed",
        relative_edge_indices=ordered_relative_indices,
        involved_flow_ids=tuple(sorted(involved_flows)),
        involved_endpoints=tuple(sorted(involved_endpoints)),
        start_time=start_time,
        end_time=end_time,
        duration=float(max((end_time - start_time).total_seconds(), 0.0)) if start_time is not None and end_time is not None else 0.0,
        repeated_pair_count=int(repeated_pair_count),
        burst_persistence=burst_persistence,
        support_cluster_density=support_cluster_density,
        gap_count=0,
        continuity_span=float(max((end_time - start_time).total_seconds(), 0.0)) if start_time is not None and end_time is not None else 0.0,
        protocol_consistency_score=1.0,
        direction_pattern_consistency=1.0,
        nuisance_likelihood=nuisance_likelihood,
        proposal_score=proposal_score,
        repeated_slice_support_count=int(repeated_slice_support_count),
        endpoint_span=len(involved_endpoints),
        source_count=len(source_ids),
        target_count=len(target_ids),
        merged_flow_count=len(involved_flows),
        service_chain_length=max(len(target_ids), 1),
        burst_repeat_count=int(repeated_slice_support_count),
        burst_slice_span=len(slice_counts),
        burst_endpoint_overlap=float(repeated_pair_count / max(len(involved_endpoints), 1)),
    )


def _dedupe_episodes(episodes: list[Episode]) -> list[Episode]:
    unique: dict[tuple[int, ...], Episode] = {}
    for episode in episodes:
        key = episode.relative_edge_indices
        existing = unique.get(key)
        if existing is None or (
            episode.proposal_score,
            episode.support_cluster_density,
            episode.burst_persistence,
        ) > (
            existing.proposal_score,
            existing.support_cluster_density,
            existing.burst_persistence,
        ):
            unique[key] = episode
    return sorted(
        unique.values(),
        key=lambda item: (item.proposal_score, item.support_cluster_density, item.edge_count),
        reverse=True,
    )


def _repeated_pair_episodes(
    sample: Any,
    communication_indices: list[int],
    communication_scores: np.ndarray,
    *,
    top_k: int,
    slice_count: int,
    flow_by_id: dict[str, Any],
) -> list[Episode]:
    pair_groups: dict[tuple[str, str], set[int]] = {}
    high_cutoff = float(np.percentile(communication_scores, max(60.0, 100.0 - top_k * 10.0)))
    for relative_index, edge_index in enumerate(communication_indices):
        if float(communication_scores[relative_index]) < high_cutoff:
            continue
        edge = sample.graph.edges[edge_index]
        pair_groups.setdefault((edge.source_node_id, edge.target_node_id), set()).add(relative_index)
    ranked_pairs = sorted(
        pair_groups.items(),
        key=lambda item: (
            len(item[1]),
            float(np.mean([communication_scores[index] for index in item[1]])),
        ),
        reverse=True,
    )[: max(1, top_k)]
    episodes: list[Episode] = []
    for rank, ((source_id, target_id), indices) in enumerate(ranked_pairs):
        expanded = set(indices)
        for relative_index, edge_index in enumerate(communication_indices):
            edge = sample.graph.edges[edge_index]
            if edge.source_node_id == source_id and edge.target_node_id == target_id:
                expanded.add(relative_index)
        episode = _episode_from_relative_indices(
            sample,
            communication_indices,
            communication_scores,
            expanded,
            proposal_mode="repeated_pair_episode",
            episode_id=f"repeated_pair_episode:{rank}",
            slice_count=slice_count,
            flow_by_id=flow_by_id,
        )
        if episode is not None:
            episodes.append(episode)
    return _dedupe_episodes(episodes)


def _local_burst_episodes(
    sample: Any,
    communication_indices: list[int],
    communication_scores: np.ndarray,
    *,
    top_k: int,
    slice_count: int,
    flow_by_id: dict[str, Any],
) -> list[Episode]:
    dst_ip_by_node = {node.node_id: node.ip for node in sample.graph.nodes if getattr(node, "endpoint_type", "") == "server"}
    burst_groups: dict[tuple[str, str], set[int]] = {}
    high_cutoff = float(np.percentile(communication_scores, max(65.0, 100.0 - top_k * 8.0)))
    for relative_index, edge_index in enumerate(communication_indices):
        if float(communication_scores[relative_index]) < high_cutoff:
            continue
        edge = sample.graph.edges[edge_index]
        slice_index = _edge_slice_index(sample, edge, flow_by_id, slice_count=slice_count)
        dst_subnet = _dst_subnet(dst_ip_by_node.get(edge.target_node_id, edge.target_node_id))
        burst_groups.setdefault((f"{edge.source_node_id}|{dst_subnet}", str(slice_index)), set()).add(relative_index)
    ranked_groups = sorted(
        burst_groups.items(),
        key=lambda item: (
            len(item[1]),
            float(np.mean([communication_scores[index] for index in item[1]])),
        ),
        reverse=True,
    )[: max(1, top_k)]
    episodes: list[Episode] = []
    for rank, (((source_cluster, slice_text), indices)) in enumerate(ranked_groups):
        source_id, dst_subnet = source_cluster.split("|", 1)
        target_slice = int(slice_text)
        expanded = set(indices)
        for relative_index, edge_index in enumerate(communication_indices):
            edge = sample.graph.edges[edge_index]
            edge_slice = _edge_slice_index(sample, edge, flow_by_id, slice_count=slice_count)
            edge_dst_subnet = _dst_subnet(dst_ip_by_node.get(edge.target_node_id, edge.target_node_id))
            if (
                edge.source_node_id == source_id
                and edge_dst_subnet == dst_subnet
                and abs(edge_slice - target_slice) <= 1
            ):
                expanded.add(relative_index)
        episode = _episode_from_relative_indices(
            sample,
            communication_indices,
            communication_scores,
            expanded,
            proposal_mode="local_burst_episode",
            episode_id=f"local_burst_episode:{rank}",
            slice_count=slice_count,
            flow_by_id=flow_by_id,
        )
        if episode is not None:
            episodes.append(episode)
    return _dedupe_episodes(episodes)


def _support_cluster_episodes(
    sample: Any,
    communication_indices: list[int],
    communication_scores: np.ndarray,
    *,
    top_k: int,
    slice_count: int,
    flow_by_id: dict[str, Any],
) -> list[Episode]:
    if not communication_indices:
        return []
    sorted_relative = list(np.argsort(communication_scores)[::-1])
    dst_ip_by_node = {node.node_id: node.ip for node in sample.graph.nodes if getattr(node, "endpoint_type", "") == "server"}
    episodes: list[Episode] = []
    for rank, seed_relative_index in enumerate(sorted_relative[: max(1, top_k)]):
        seed_edge = sample.graph.edges[communication_indices[seed_relative_index]]
        seed_slice = _edge_slice_index(sample, seed_edge, flow_by_id, slice_count=slice_count)
        seed_subnet = _dst_subnet(dst_ip_by_node.get(seed_edge.target_node_id, seed_edge.target_node_id))
        expanded = {seed_relative_index}
        for relative_index, edge_index in enumerate(communication_indices):
            edge = sample.graph.edges[edge_index]
            edge_slice = _edge_slice_index(sample, edge, flow_by_id, slice_count=slice_count)
            shares_support = (
                edge.source_node_id in {seed_edge.source_node_id, seed_edge.target_node_id}
                or edge.target_node_id in {seed_edge.source_node_id, seed_edge.target_node_id}
                or (edge.source_node_id == seed_edge.source_node_id and abs(edge_slice - seed_slice) <= 1)
                or (_dst_subnet(dst_ip_by_node.get(edge.target_node_id, edge.target_node_id)) == seed_subnet and abs(edge_slice - seed_slice) <= 1)
            )
            if shares_support and float(communication_scores[relative_index]) >= np.median(communication_scores):
                expanded.add(relative_index)
        episode = _episode_from_relative_indices(
            sample,
            communication_indices,
            communication_scores,
            expanded,
            proposal_mode="support_cluster_episode",
            episode_id=f"support_cluster_episode:{rank}",
            slice_count=slice_count,
            flow_by_id=flow_by_id,
        )
        if episode is not None:
            episodes.append(episode)
    return _dedupe_episodes(episodes)


def propose_episodes(
    sample: Any,
    communication_indices: list[int],
    communication_scores: np.ndarray,
    *,
    proposal_mode: EpisodeProposalMode,
    top_k: int,
    slice_count: int = 3,
) -> list[Episode]:
    """Propose local behavior episodes from scored communication edges."""

    if not communication_indices:
        return []
    flow_by_id = _flow_lookup(sample)
    if proposal_mode == "repeated_pair_episode":
        return _repeated_pair_episodes(
            sample,
            communication_indices,
            communication_scores,
            top_k=top_k,
            slice_count=slice_count,
            flow_by_id=flow_by_id,
        )
    if proposal_mode == "local_burst_episode":
        return _local_burst_episodes(
            sample,
            communication_indices,
            communication_scores,
            top_k=top_k,
            slice_count=slice_count,
            flow_by_id=flow_by_id,
        )
    if proposal_mode == "support_cluster_episode":
        return _support_cluster_episodes(
            sample,
            communication_indices,
            communication_scores,
            top_k=top_k,
            slice_count=slice_count,
            flow_by_id=flow_by_id,
        )
    raise ValueError(f"Unsupported episode proposal mode: {proposal_mode}")


__all__ = [
    "Episode",
    "EpisodeProposalMode",
    "propose_episodes",
]
