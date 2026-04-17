"""Coverage-preserving episode sessionization / stitching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from traffic_graph.pipeline.episode_proposal import Episode

EpisodeStitchingMode = Literal[
    "repeated_pair_temporal_continuity",
    "protocol_consistent_interaction_chain",
    "repeated_local_burst_stitching",
]


@dataclass(frozen=True, slots=True)
class FlowSupport:
    logical_flow_id: str
    relative_edge_index: int
    src_ip: str
    dst_ip: str
    dst_port: int
    protocol: str
    start_time: object
    end_time: object
    directions: tuple[str, ...]
    prefix_behavior_signature: int
    score: float
    dst_subnet: str
    slice_index: int


def _dst_subnet(ip: str) -> str:
    parts = ip.split(".")
    if len(parts) >= 3:
        return ".".join(parts[:3])
    return ip


def _flow_lookup(sample: Any) -> dict[str, Any]:
    return {str(flow.logical_flow_id): flow for flow in getattr(sample, "logical_flows", ())}


def _slice_index(sample: Any, logical_flow: Any, *, slice_count: int) -> int:
    window_seconds = max((sample.graph.window_end - sample.graph.window_start).total_seconds(), 1e-6)
    midpoint = logical_flow.start_time + (logical_flow.end_time - logical_flow.start_time) / 2
    ratio = min(max((midpoint - sample.graph.window_start).total_seconds() / window_seconds, 0.0), 0.999999)
    return min(int(ratio * slice_count), slice_count - 1)


def _flow_supports(
    sample: Any,
    communication_indices: list[int],
    communication_scores: np.ndarray,
    *,
    slice_count: int,
) -> list[FlowSupport]:
    flow_by_id = _flow_lookup(sample)
    edge_by_flow: dict[str, tuple[int, float]] = {}
    for relative_index, edge_index in enumerate(communication_indices):
        edge = sample.graph.edges[edge_index]
        score = float(communication_scores[relative_index])
        logical_ids = [str(getattr(edge, "logical_flow_id", ""))]
        logical_ids.extend(str(flow_id) for flow_id in getattr(edge, "source_flow_ids", ()))
        for logical_flow_id in logical_ids:
            if not logical_flow_id:
                continue
            existing = edge_by_flow.get(logical_flow_id)
            if existing is None or score > existing[1]:
                edge_by_flow[logical_flow_id] = (relative_index, score)
    supports: list[FlowSupport] = []
    for logical_flow_id, logical_flow in flow_by_id.items():
        match = edge_by_flow.get(logical_flow_id)
        if match is None:
            continue
        relative_index, score = match
        supports.append(
            FlowSupport(
                logical_flow_id=logical_flow_id,
                relative_edge_index=relative_index,
                src_ip=logical_flow.src_ip,
                dst_ip=logical_flow.dst_ip,
                dst_port=logical_flow.dst_port,
                protocol=logical_flow.protocol,
                start_time=logical_flow.start_time,
                end_time=logical_flow.end_time,
                directions=tuple(sorted(getattr(logical_flow, "directions", ()))),
                prefix_behavior_signature=int(getattr(logical_flow, "prefix_behavior_signature", 0)),
                score=score,
                dst_subnet=_dst_subnet(logical_flow.dst_ip),
                slice_index=_slice_index(sample, logical_flow, slice_count=slice_count),
            )
        )
    return sorted(supports, key=lambda item: (item.start_time, item.logical_flow_id))


def _gap_seconds(left: FlowSupport, right: FlowSupport) -> float:
    return float(max((right.start_time - left.end_time).total_seconds(), 0.0))


def _direction_consistency(flows: list[FlowSupport]) -> float:
    patterns = [flow.directions for flow in flows if flow.directions]
    if not patterns:
        return 0.0
    unique_patterns = {pattern for pattern in patterns}
    return float(1.0 / max(len(unique_patterns), 1))


def _episode_from_supports(
    flow_supports: list[FlowSupport],
    *,
    stitching_mode: EpisodeStitchingMode,
    episode_id: str,
    window_seconds: float,
) -> Episode | None:
    if len(flow_supports) < 2:
        return None
    ordered = sorted(flow_supports, key=lambda item: (item.start_time, item.logical_flow_id))
    relative_edge_indices = tuple(sorted({flow.relative_edge_index for flow in ordered}))
    if len(relative_edge_indices) < 2:
        return None
    start_time = ordered[0].start_time
    end_time = max(flow.end_time for flow in ordered)
    duration = float(max((end_time - start_time).total_seconds(), 0.0))
    source_ids = sorted({flow.src_ip for flow in ordered})
    target_ids = sorted({flow.dst_ip for flow in ordered})
    involved_endpoints = tuple(sorted({*source_ids, *target_ids}))
    pair_counts: dict[tuple[str, str, int, str], int] = {}
    slice_counts: dict[int, int] = {}
    gap_count = 0
    for idx, flow in enumerate(ordered):
        pair_key = (flow.src_ip, flow.dst_ip, flow.dst_port, flow.protocol)
        pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
        slice_counts[flow.slice_index] = slice_counts.get(flow.slice_index, 0) + 1
        if idx > 0 and _gap_seconds(ordered[idx - 1], flow) > 0.0:
            gap_count += 1
    repeated_pair_count = sum(count >= 2 for count in pair_counts.values())
    repeated_slice_support_count = sum(count >= 2 for count in slice_counts.values())
    continuity_span = float(min(duration / max(window_seconds, 1.0), 1.0))
    protocol_consistency_score = float(
        0.6 * (1.0 if len({flow.protocol for flow in ordered}) == 1 else 0.0)
        + 0.4 * (1.0 if len({(flow.dst_port, flow.dst_subnet) for flow in ordered}) <= max(2, len(ordered) // 2) else 0.0)
    )
    direction_pattern_consistency = _direction_consistency(ordered)
    burst_slice_span = max(slice_counts) - min(slice_counts) + 1 if slice_counts else 0
    burst_repeat_count = sum(count >= 2 for count in slice_counts.values())
    burst_endpoint_overlap = float(
        sum(count >= 2 for count in pair_counts.values()) / max(len(pair_counts), 1)
    )
    support_cluster_density = float(
        0.35 * min(len(relative_edge_indices) / max(len(involved_endpoints), 1), 2.0) / 2.0
        + 0.25 * min(repeated_pair_count / 2.0, 1.0)
        + 0.20 * continuity_span
        + 0.20 * protocol_consistency_score
    )
    burst_persistence = float(
        0.45 * min(len(slice_counts) / 3.0, 1.0)
        + 0.35 * continuity_span
        + 0.20 * min(burst_repeat_count / max(len(slice_counts), 1), 1.0)
    )
    mean_score = float(np.mean([flow.score for flow in ordered]))
    max_score = float(np.max([flow.score for flow in ordered]))
    proposal_score = float(
        0.40 * mean_score
        + 0.15 * max_score
        + 0.15 * min(repeated_pair_count / 2.0, 1.0)
        + 0.15 * protocol_consistency_score
        + 0.15 * burst_persistence
    )
    nuisance_likelihood = float(
        0.30 * max(0.0, 1.0 - continuity_span)
        + 0.20 * max(0.0, 1.0 - protocol_consistency_score)
        + 0.20 * max(0.0, 1.0 - direction_pattern_consistency)
        + 0.15 * min(len(target_ids) / max(len(ordered), 1), 1.0)
        + 0.15 * max(0.0, 1.0 - min(repeated_pair_count / 2.0, 1.0))
    )
    return Episode(
        episode_id=episode_id,
        proposal_mode="support_cluster_episode" if stitching_mode == "repeated_local_burst_stitching" else "repeated_pair_episode",
        relative_edge_indices=relative_edge_indices,
        involved_flow_ids=tuple(flow.logical_flow_id for flow in ordered),
        involved_endpoints=involved_endpoints,
        start_time=start_time,
        end_time=end_time,
        repeated_pair_count=int(repeated_pair_count),
        burst_persistence=burst_persistence,
        support_cluster_density=support_cluster_density,
        nuisance_likelihood=nuisance_likelihood,
        proposal_score=proposal_score,
        repeated_slice_support_count=int(repeated_slice_support_count),
        endpoint_span=len(involved_endpoints),
        source_count=len(source_ids),
        target_count=len(target_ids),
        stitching_mode=stitching_mode,
        duration=duration,
        gap_count=gap_count,
        continuity_span=continuity_span,
        protocol_consistency_score=protocol_consistency_score,
        direction_pattern_consistency=direction_pattern_consistency,
        merged_flow_count=len(ordered),
        service_chain_length=len({(flow.dst_port, flow.dst_subnet) for flow in ordered}),
        burst_repeat_count=burst_repeat_count,
        burst_slice_span=burst_slice_span,
        burst_endpoint_overlap=burst_endpoint_overlap,
    )


def _dedupe(episodes: list[Episode]) -> list[Episode]:
    unique: dict[tuple[str, ...], Episode] = {}
    for episode in episodes:
        key = episode.involved_flow_ids
        existing = unique.get(key)
        if existing is None or (
            episode.proposal_score,
            episode.continuity_span,
            episode.protocol_consistency_score,
        ) > (
            existing.proposal_score,
            existing.continuity_span,
            existing.protocol_consistency_score,
        ):
            unique[key] = episode
    return sorted(unique.values(), key=lambda item: (item.proposal_score, item.flow_count, item.duration), reverse=True)


def _repeated_pair_temporal(
    supports: list[FlowSupport],
    *,
    max_gap_seconds: float,
    window_seconds: float,
) -> list[Episode]:
    grouped: dict[tuple[str, str, int, str], list[FlowSupport]] = {}
    for flow in supports:
        grouped.setdefault((flow.src_ip, flow.dst_ip, flow.dst_port, flow.protocol), []).append(flow)
    episodes: list[Episode] = []
    rank = 0
    for group in grouped.values():
        current: list[FlowSupport] = []
        for flow in sorted(group, key=lambda item: (item.start_time, item.logical_flow_id)):
            if not current:
                current = [flow]
                continue
            if _gap_seconds(current[-1], flow) <= max_gap_seconds:
                current.append(flow)
            else:
                episode = _episode_from_supports(
                    current,
                    stitching_mode="repeated_pair_temporal_continuity",
                    episode_id=f"repeated_pair_temporal_continuity:{rank}",
                    window_seconds=window_seconds,
                )
                if episode is not None:
                    episodes.append(episode)
                    rank += 1
                current = [flow]
        if current:
            episode = _episode_from_supports(
                current,
                stitching_mode="repeated_pair_temporal_continuity",
                episode_id=f"repeated_pair_temporal_continuity:{rank}",
                window_seconds=window_seconds,
            )
            if episode is not None:
                episodes.append(episode)
                rank += 1
    return _dedupe(episodes)


def _protocol_consistent_chain(
    supports: list[FlowSupport],
    *,
    max_gap_seconds: float,
    window_seconds: float,
) -> list[Episode]:
    grouped: dict[tuple[str, str, int, tuple[str, ...]], list[FlowSupport]] = {}
    for flow in supports:
        grouped.setdefault((flow.src_ip, flow.protocol, flow.dst_port, flow.directions), []).append(flow)
    episodes: list[Episode] = []
    rank = 0
    for group in grouped.values():
        ordered = sorted(group, key=lambda item: (item.start_time, item.logical_flow_id))
        current = [ordered[0]]
        for flow in ordered[1:]:
            prev = current[-1]
            service_match = flow.dst_subnet == prev.dst_subnet or flow.dst_ip == prev.dst_ip
            if _gap_seconds(prev, flow) <= max_gap_seconds and service_match:
                current.append(flow)
            else:
                episode = _episode_from_supports(
                    current,
                    stitching_mode="protocol_consistent_interaction_chain",
                    episode_id=f"protocol_consistent_interaction_chain:{rank}",
                    window_seconds=window_seconds,
                )
                if episode is not None:
                    episodes.append(episode)
                    rank += 1
                current = [flow]
        episode = _episode_from_supports(
            current,
            stitching_mode="protocol_consistent_interaction_chain",
            episode_id=f"protocol_consistent_interaction_chain:{rank}",
            window_seconds=window_seconds,
        )
        if episode is not None:
            episodes.append(episode)
            rank += 1
    return _dedupe(episodes)


def _repeated_local_burst(
    supports: list[FlowSupport],
    *,
    max_gap_seconds: float,
    window_seconds: float,
) -> list[Episode]:
    grouped: dict[tuple[str, str, int, str], list[FlowSupport]] = {}
    for flow in supports:
        grouped.setdefault((flow.src_ip, flow.dst_subnet, flow.dst_port, flow.protocol), []).append(flow)
    episodes: list[Episode] = []
    rank = 0
    for group in grouped.values():
        by_slice: dict[int, list[FlowSupport]] = {}
        for flow in group:
            by_slice.setdefault(flow.slice_index, []).append(flow)
        burst_slices = sorted(
            slice_index
            for slice_index, slice_flows in by_slice.items()
            if len(slice_flows) >= 2 or np.mean([flow.score for flow in slice_flows]) >= np.median([flow.score for flow in group])
        )
        if not burst_slices:
            continue
        current_slice_chain: list[int] = [burst_slices[0]]
        for slice_index in burst_slices[1:]:
            if slice_index - current_slice_chain[-1] <= 1:
                current_slice_chain.append(slice_index)
            else:
                chain_flows = [flow for idx in current_slice_chain for flow in by_slice[idx]]
                episode = _episode_from_supports(
                    chain_flows,
                    stitching_mode="repeated_local_burst_stitching",
                    episode_id=f"repeated_local_burst_stitching:{rank}",
                    window_seconds=window_seconds,
                )
                if episode is not None:
                    episodes.append(episode)
                    rank += 1
                current_slice_chain = [slice_index]
        chain_flows = [flow for idx in current_slice_chain for flow in by_slice[idx]]
        episode = _episode_from_supports(
            chain_flows,
            stitching_mode="repeated_local_burst_stitching",
            episode_id=f"repeated_local_burst_stitching:{rank}",
            window_seconds=window_seconds,
        )
        if episode is not None:
            episodes.append(episode)
            rank += 1
    return _dedupe(episodes)


def sessionize_episodes(
    sample: Any,
    communication_indices: list[int],
    communication_scores: np.ndarray,
    *,
    stitching_mode: EpisodeStitchingMode,
    slice_count: int = 3,
    max_gap_seconds: float | None = None,
) -> list[Episode]:
    """Build stitched, coverage-preserving episodes from logical-flow chains."""

    supports = _flow_supports(sample, communication_indices, communication_scores, slice_count=slice_count)
    if not supports:
        return []
    window_seconds = max((sample.graph.window_end - sample.graph.window_start).total_seconds(), 1.0)
    effective_gap_seconds = max_gap_seconds if max_gap_seconds is not None else max(window_seconds / 3.0, 1.0)
    if stitching_mode == "repeated_pair_temporal_continuity":
        return _repeated_pair_temporal(
            supports,
            max_gap_seconds=effective_gap_seconds,
            window_seconds=window_seconds,
        )
    if stitching_mode == "protocol_consistent_interaction_chain":
        return _protocol_consistent_chain(
            supports,
            max_gap_seconds=effective_gap_seconds,
            window_seconds=window_seconds,
        )
    if stitching_mode == "repeated_local_burst_stitching":
        return _repeated_local_burst(
            supports,
            max_gap_seconds=effective_gap_seconds,
            window_seconds=window_seconds,
        )
    raise ValueError(f"Unsupported episode stitching mode: {stitching_mode}")


__all__ = [
    "EpisodeStitchingMode",
    "sessionize_episodes",
]
