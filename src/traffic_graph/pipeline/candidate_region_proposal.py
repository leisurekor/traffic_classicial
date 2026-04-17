"""Lightweight candidate-region proposal helpers for edge-centric detection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import numpy as np

CandidateRegionProposalMode = Literal[
    "edge_seed_region",
    "temporal_burst_region",
    "edge_seed_region_v2",
    "temporal_burst_region_v2",
    "support_cluster_region",
]


@dataclass(frozen=True, slots=True)
class CandidateRegion:
    """Auditable local region proposed from one coarse graph."""

    candidate_region_id: str
    proposal_mode: CandidateRegionProposalMode
    seed_relative_indices: tuple[int, ...]
    relative_edge_indices: tuple[int, ...]
    node_ids: tuple[str, ...]
    slice_indices: tuple[int, ...]
    candidate_time_span: float
    candidate_src_count: int
    candidate_dst_count: int
    candidate_score_seed: float
    candidate_score_mean: float
    candidate_score_max: float
    repeated_endpoint_count: int = 0
    repeated_slice_support_count: int = 0
    support_cluster_density: float = 0.0

    @property
    def seed_edge_count(self) -> int:
        return len(self.seed_relative_indices)

    @property
    def candidate_edge_count(self) -> int:
        return len(self.relative_edge_indices)

    @property
    def candidate_node_count(self) -> int:
        return len(self.node_ids)


def _dst_subnet(ip: str) -> str:
    parts = ip.split(".")
    if len(parts) >= 3:
        return ".".join(parts[:3])
    return ip


def _edge_midpoint_seconds(sample: Any, edge: Any, flow_by_id: dict[str, Any]) -> float:
    logical_flow = flow_by_id.get(str(edge.logical_flow_id))
    if logical_flow is None:
        for source_flow_id in getattr(edge, "source_flow_ids", ()):
            logical_flow = flow_by_id.get(str(source_flow_id))
            if logical_flow is not None:
                break
    if logical_flow is None:
        return 0.0
    midpoint = logical_flow.start_time + (logical_flow.end_time - logical_flow.start_time) / 2
    return float((midpoint - sample.graph.window_start).total_seconds())


def _edge_slice_index(
    sample: Any,
    edge: Any,
    flow_by_id: dict[str, Any],
    *,
    slice_count: int,
) -> int:
    window_seconds = max((sample.graph.window_end - sample.graph.window_start).total_seconds(), 1e-6)
    midpoint_seconds = _edge_midpoint_seconds(sample, edge, flow_by_id)
    ratio = min(max(midpoint_seconds / window_seconds, 0.0), 0.999999)
    return min(int(ratio * slice_count), slice_count - 1)


def _region_from_relative_indices(
    sample: Any,
    communication_indices: list[int],
    communication_scores: np.ndarray,
    relative_indices: set[int],
    seed_relative_indices: tuple[int, ...],
    proposal_mode: CandidateRegionProposalMode,
    candidate_region_id: str,
    *,
    slice_count: int,
    flow_by_id: dict[str, Any],
) -> CandidateRegion | None:
    if not relative_indices:
        return None
    ordered_relative_indices = tuple(sorted(relative_indices))
    node_ids: set[str] = set()
    src_ids: set[str] = set()
    dst_ids: set[str] = set()
    slice_indices: set[int] = set()
    midpoint_seconds: list[float] = []
    for relative_index in ordered_relative_indices:
        edge = sample.graph.edges[communication_indices[relative_index]]
        node_ids.update((edge.source_node_id, edge.target_node_id))
        src_ids.add(edge.source_node_id)
        dst_ids.add(edge.target_node_id)
        slice_indices.add(_edge_slice_index(sample, edge, flow_by_id, slice_count=slice_count))
        midpoint_seconds.append(_edge_midpoint_seconds(sample, edge, flow_by_id))
    time_span = max(midpoint_seconds, default=0.0) - min(midpoint_seconds, default=0.0)
    region_scores = np.asarray([communication_scores[index] for index in ordered_relative_indices], dtype=float)
    seed_scores = np.asarray([communication_scores[index] for index in seed_relative_indices], dtype=float)
    endpoint_counts: dict[str, int] = {}
    for relative_index in ordered_relative_indices:
        edge = sample.graph.edges[communication_indices[relative_index]]
        endpoint_counts[edge.source_node_id] = endpoint_counts.get(edge.source_node_id, 0) + 1
        endpoint_counts[edge.target_node_id] = endpoint_counts.get(edge.target_node_id, 0) + 1
    repeated_endpoint_count = sum(count >= 2 for count in endpoint_counts.values())
    repeated_slice_support_count = sum(1 for slice_index in slice_indices if sum(
        _edge_slice_index(sample, sample.graph.edges[communication_indices[index]], flow_by_id, slice_count=slice_count) == slice_index
        for index in ordered_relative_indices
    ) >= 2)
    support_cluster_density = float(
        0.5 * (len(ordered_relative_indices) / max(len(node_ids), 1))
        + 0.5 * (repeated_endpoint_count / max(len(endpoint_counts), 1))
    )
    return CandidateRegion(
        candidate_region_id=candidate_region_id,
        proposal_mode=proposal_mode,
        seed_relative_indices=tuple(sorted(seed_relative_indices)),
        relative_edge_indices=ordered_relative_indices,
        node_ids=tuple(sorted(node_ids)),
        slice_indices=tuple(sorted(slice_indices)),
        candidate_time_span=float(time_span),
        candidate_src_count=len(src_ids),
        candidate_dst_count=len(dst_ids),
        candidate_score_seed=float(np.mean(seed_scores)) if seed_scores.size else 0.0,
        candidate_score_mean=float(np.mean(region_scores)) if region_scores.size else 0.0,
        candidate_score_max=float(np.max(region_scores)) if region_scores.size else 0.0,
        repeated_endpoint_count=int(repeated_endpoint_count),
        repeated_slice_support_count=int(repeated_slice_support_count),
        support_cluster_density=support_cluster_density,
    )


def _dedupe_regions(candidates: list[CandidateRegion]) -> list[CandidateRegion]:
    unique: dict[tuple[int, ...], CandidateRegion] = {}
    for candidate in candidates:
        key = candidate.relative_edge_indices
        existing = unique.get(key)
        if existing is None or (
            candidate.candidate_score_max,
            candidate.candidate_score_mean,
            candidate.seed_edge_count,
        ) > (
            existing.candidate_score_max,
            existing.candidate_score_mean,
            existing.seed_edge_count,
        ):
            unique[key] = candidate
    return sorted(
        unique.values(),
        key=lambda candidate: (
            candidate.candidate_score_max,
            candidate.candidate_score_mean,
            candidate.candidate_edge_count,
        ),
        reverse=True,
    )


def _edge_seed_regions(
    sample: Any,
    communication_indices: list[int],
    communication_scores: np.ndarray,
    *,
    top_k: int,
    slice_count: int,
    flow_by_id: dict[str, Any],
) -> list[CandidateRegion]:
    if not communication_indices:
        return []
    source_ip_by_node = {node.node_id: node.ip for node in sample.graph.nodes if node.endpoint_type == "client"}
    dst_ip_by_node = {node.node_id: node.ip for node in sample.graph.nodes if node.endpoint_type == "server"}
    sorted_indices = list(np.argsort(communication_scores)[::-1])
    candidate_regions: list[CandidateRegion] = []
    seed_budget = max(1, min(top_k, len(sorted_indices)))
    for rank, seed_relative_index in enumerate(sorted_indices[:seed_budget]):
        seed_edge = sample.graph.edges[communication_indices[seed_relative_index]]
        seed_src_ip = source_ip_by_node.get(seed_edge.source_node_id, "")
        seed_dst_ip = dst_ip_by_node.get(seed_edge.target_node_id, "")
        seed_dst_subnet = _dst_subnet(seed_dst_ip)
        relative_indices: set[int] = set()
        for relative_index, edge_index in enumerate(communication_indices):
            edge = sample.graph.edges[edge_index]
            edge_src_ip = source_ip_by_node.get(edge.source_node_id, "")
            edge_dst_ip = dst_ip_by_node.get(edge.target_node_id, "")
            shares_context = (
                edge.source_node_id in {seed_edge.source_node_id, seed_edge.target_node_id}
                or edge.target_node_id in {seed_edge.source_node_id, seed_edge.target_node_id}
                or (seed_src_ip and edge_src_ip == seed_src_ip)
                or (seed_dst_ip and edge_dst_ip == seed_dst_ip)
                or (seed_dst_subnet and _dst_subnet(edge_dst_ip) == seed_dst_subnet)
            )
            if shares_context:
                relative_indices.add(relative_index)
        candidate = _region_from_relative_indices(
            sample,
            communication_indices,
            communication_scores,
            relative_indices,
            (seed_relative_index,),
            "edge_seed_region",
            f"edge_seed_region:{rank}",
            slice_count=slice_count,
            flow_by_id=flow_by_id,
        )
        if candidate is not None:
            candidate_regions.append(candidate)
    return _dedupe_regions(candidate_regions)


def _temporal_burst_regions(
    sample: Any,
    communication_indices: list[int],
    communication_scores: np.ndarray,
    *,
    top_k: int,
    slice_count: int,
    flow_by_id: dict[str, Any],
) -> list[CandidateRegion]:
    if not communication_indices:
        return []
    percentile_cutoff = float(np.percentile(communication_scores, max(70.0, 100.0 - top_k * 5.0)))
    slice_buckets: dict[tuple[int, str], set[int]] = {}
    for relative_index, edge_index in enumerate(communication_indices):
        score = float(communication_scores[relative_index])
        if score < percentile_cutoff:
            continue
        edge = sample.graph.edges[edge_index]
        slice_index = _edge_slice_index(sample, edge, flow_by_id, slice_count=slice_count)
        bucket_key = (slice_index, edge.target_node_id)
        slice_buckets.setdefault(bucket_key, set()).add(relative_index)
    candidate_regions: list[CandidateRegion] = []
    for rank, ((slice_index, _dst_node_id), indices) in enumerate(
        sorted(
            slice_buckets.items(),
            key=lambda item: (
                max(communication_scores[index] for index in item[1]),
                len(item[1]),
            ),
            reverse=True,
        )[: max(1, top_k)]
    ):
        candidate = _region_from_relative_indices(
            sample,
            communication_indices,
            communication_scores,
            indices,
            tuple(sorted(indices)),
            "temporal_burst_region",
            f"temporal_burst_region:{slice_index}:{rank}",
            slice_count=slice_count,
            flow_by_id=flow_by_id,
        )
        if candidate is not None:
            candidate_regions.append(candidate)
    return _dedupe_regions(candidate_regions)


def _edge_seed_regions_v2(
    sample: Any,
    communication_indices: list[int],
    communication_scores: np.ndarray,
    *,
    top_k: int,
    slice_count: int,
    flow_by_id: dict[str, Any],
) -> list[CandidateRegion]:
    if not communication_indices:
        return []
    source_ip_by_node = {node.node_id: node.ip for node in sample.graph.nodes if node.endpoint_type == "client"}
    sorted_indices = list(np.argsort(communication_scores)[::-1])
    candidate_regions: list[CandidateRegion] = []
    for rank, seed_relative_index in enumerate(sorted_indices[: max(1, min(top_k, len(sorted_indices)))]):
        seed_edge = sample.graph.edges[communication_indices[seed_relative_index]]
        seed_slice = _edge_slice_index(sample, seed_edge, flow_by_id, slice_count=slice_count)
        relative_indices: set[int] = {seed_relative_index}
        repeated_pair_indices: set[int] = set()
        for relative_index, edge_index in enumerate(communication_indices):
            edge = sample.graph.edges[edge_index]
            edge_slice = _edge_slice_index(sample, edge, flow_by_id, slice_count=slice_count)
            same_pair = (
                edge.source_node_id == seed_edge.source_node_id
                and edge.target_node_id == seed_edge.target_node_id
            )
            shared_endpoint = (
                edge.source_node_id in {seed_edge.source_node_id, seed_edge.target_node_id}
                or edge.target_node_id in {seed_edge.source_node_id, seed_edge.target_node_id}
            )
            shared_src_ip = source_ip_by_node.get(edge.source_node_id, "") == source_ip_by_node.get(seed_edge.source_node_id, "")
            nearby_slice = abs(edge_slice - seed_slice) <= 1
            if same_pair:
                repeated_pair_indices.add(relative_index)
            if same_pair or (shared_endpoint and nearby_slice) or (shared_src_ip and nearby_slice and communication_scores[relative_index] >= np.median(communication_scores)):
                relative_indices.add(relative_index)
        if len(relative_indices) < 2 and repeated_pair_indices:
            relative_indices.update(repeated_pair_indices)
        candidate = _region_from_relative_indices(
            sample,
            communication_indices,
            communication_scores,
            relative_indices,
            tuple(sorted(relative_indices & set(sorted_indices[:2])) or (seed_relative_index,)),
            "edge_seed_region_v2",
            f"edge_seed_region_v2:{rank}",
            slice_count=slice_count,
            flow_by_id=flow_by_id,
        )
        if candidate is not None and (
            candidate.candidate_edge_count >= 2
            or candidate.repeated_endpoint_count >= 1
            or candidate.repeated_slice_support_count >= 1
        ):
            candidate_regions.append(candidate)
    return _dedupe_regions(candidate_regions)


def _temporal_burst_regions_v2(
    sample: Any,
    communication_indices: list[int],
    communication_scores: np.ndarray,
    *,
    top_k: int,
    slice_count: int,
    flow_by_id: dict[str, Any],
) -> list[CandidateRegion]:
    if not communication_indices:
        return []
    percentile_cutoff = float(np.percentile(communication_scores, max(60.0, 100.0 - top_k * 10.0)))
    buckets: dict[tuple[int, str], set[int]] = {}
    for relative_index, edge_index in enumerate(communication_indices):
        if float(communication_scores[relative_index]) < percentile_cutoff:
            continue
        edge = sample.graph.edges[edge_index]
        slice_index = _edge_slice_index(sample, edge, flow_by_id, slice_count=slice_count)
        key = (slice_index, edge.target_node_id)
        buckets.setdefault(key, set()).add(relative_index)
        if slice_index > 0:
            buckets.setdefault((slice_index - 1, edge.target_node_id), set())
        if slice_index < slice_count - 1:
            buckets.setdefault((slice_index + 1, edge.target_node_id), set())
    candidate_regions: list[CandidateRegion] = []
    ranked_buckets = sorted(
        buckets.items(),
        key=lambda item: (
            len(item[1]),
            max([communication_scores[index] for index in item[1]], default=0.0),
        ),
        reverse=True,
    )
    for rank, ((slice_index, dst_node_id), indices) in enumerate(ranked_buckets[: max(1, top_k)]):
        expanded_indices = set(indices)
        for relative_index, edge_index in enumerate(communication_indices):
            edge = sample.graph.edges[edge_index]
            edge_slice = _edge_slice_index(sample, edge, flow_by_id, slice_count=slice_count)
            if edge.target_node_id == dst_node_id and abs(edge_slice - slice_index) <= 1:
                expanded_indices.add(relative_index)
        candidate = _region_from_relative_indices(
            sample,
            communication_indices,
            communication_scores,
            expanded_indices,
            tuple(sorted(indices)),
            "temporal_burst_region_v2",
            f"temporal_burst_region_v2:{slice_index}:{rank}",
            slice_count=slice_count,
            flow_by_id=flow_by_id,
        )
        if candidate is not None and candidate.candidate_edge_count >= 2:
            candidate_regions.append(candidate)
    return _dedupe_regions(candidate_regions)


def _support_cluster_regions(
    sample: Any,
    communication_indices: list[int],
    communication_scores: np.ndarray,
    *,
    top_k: int,
    slice_count: int,
    flow_by_id: dict[str, Any],
) -> list[CandidateRegion]:
    if not communication_indices:
        return []
    median_score = float(np.median(communication_scores))
    clusters: dict[tuple[str, str, int], set[int]] = {}
    for relative_index, edge_index in enumerate(communication_indices):
        edge = sample.graph.edges[edge_index]
        slice_index = _edge_slice_index(sample, edge, flow_by_id, slice_count=slice_count)
        cluster_key = (edge.source_node_id, edge.target_node_id, slice_index)
        if float(communication_scores[relative_index]) >= median_score:
            clusters.setdefault(cluster_key, set()).add(relative_index)
    candidate_regions: list[CandidateRegion] = []
    ranked_clusters = sorted(
        clusters.items(),
        key=lambda item: (
            len(item[1]),
            max(communication_scores[index] for index in item[1]),
        ),
        reverse=True,
    )
    for rank, ((src_node_id, dst_node_id, slice_index), indices) in enumerate(ranked_clusters[: max(1, top_k)]):
        expanded_indices = set(indices)
        for relative_index, edge_index in enumerate(communication_indices):
            edge = sample.graph.edges[edge_index]
            edge_slice = _edge_slice_index(sample, edge, flow_by_id, slice_count=slice_count)
            if (
                edge.source_node_id == src_node_id
                or edge.target_node_id == dst_node_id
                or (edge.source_node_id == src_node_id and abs(edge_slice - slice_index) <= 1)
            ):
                expanded_indices.add(relative_index)
        candidate = _region_from_relative_indices(
            sample,
            communication_indices,
            communication_scores,
            expanded_indices,
            tuple(sorted(indices)),
            "support_cluster_region",
            f"support_cluster_region:{rank}",
            slice_count=slice_count,
            flow_by_id=flow_by_id,
        )
        if candidate is not None and (
            candidate.candidate_edge_count >= 2
            or candidate.support_cluster_density >= 0.75
        ):
            candidate_regions.append(candidate)
    return _dedupe_regions(candidate_regions)


def propose_candidate_regions(
    sample: Any,
    communication_indices: list[int],
    communication_scores: np.ndarray,
    *,
    proposal_mode: CandidateRegionProposalMode,
    top_k: int = 3,
    slice_count: int = 3,
) -> list[CandidateRegion]:
    """Propose auditable local candidate regions from one coarse graph."""

    flow_by_id = {str(flow.logical_flow_id): flow for flow in getattr(sample, "logical_flows", ())}
    if proposal_mode == "edge_seed_region":
        return _edge_seed_regions(
            sample,
            communication_indices,
            communication_scores,
            top_k=top_k,
            slice_count=slice_count,
            flow_by_id=flow_by_id,
        )
    if proposal_mode == "temporal_burst_region":
        return _temporal_burst_regions(
            sample,
            communication_indices,
            communication_scores,
            top_k=top_k,
            slice_count=slice_count,
            flow_by_id=flow_by_id,
        )
    if proposal_mode == "edge_seed_region_v2":
        return _edge_seed_regions_v2(
            sample,
            communication_indices,
            communication_scores,
            top_k=top_k,
            slice_count=slice_count,
            flow_by_id=flow_by_id,
        )
    if proposal_mode == "temporal_burst_region_v2":
        return _temporal_burst_regions_v2(
            sample,
            communication_indices,
            communication_scores,
            top_k=top_k,
            slice_count=slice_count,
            flow_by_id=flow_by_id,
        )
    if proposal_mode == "support_cluster_region":
        return _support_cluster_regions(
            sample,
            communication_indices,
            communication_scores,
            top_k=top_k,
            slice_count=slice_count,
            flow_by_id=flow_by_id,
        )
    raise ValueError(f"Unsupported proposal mode: {proposal_mode}")


__all__ = [
    "CandidateRegion",
    "CandidateRegionProposalMode",
    "propose_candidate_regions",
]
