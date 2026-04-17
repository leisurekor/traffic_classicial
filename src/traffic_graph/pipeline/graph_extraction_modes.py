"""Graph extraction/grouping modes for CTU-13 edge-centric benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from traffic_graph.data import LogicalFlowBatch, LogicalFlowRecord

GraphExtractionMode = Literal[
    "per_src_ip_within_window",
    "short_temporal_slice_src_pair",
    "neighborhood_local_burst",
]


@dataclass(frozen=True, slots=True)
class ExtractedFlowGroup:
    """One extracted flow group before graph construction."""

    extraction_mode: GraphExtractionMode
    group_key: str
    logical_flows: tuple[LogicalFlowRecord, ...]


def _slice_index(
    batch: LogicalFlowBatch,
    logical_flow: LogicalFlowRecord,
    *,
    slice_count: int,
) -> int:
    window_seconds = max((batch.window_end - batch.window_start).total_seconds(), 1e-6)
    midpoint = logical_flow.start_time + (logical_flow.end_time - logical_flow.start_time) / 2
    elapsed = (midpoint - batch.window_start).total_seconds()
    ratio = min(max(elapsed / window_seconds, 0.0), 0.999999)
    return min(int(ratio * slice_count), slice_count - 1)


def _dst_subnet(ip: str) -> str:
    parts = ip.split(".")
    if len(parts) >= 3:
        return ".".join(parts[:3])
    return ip


def extract_flow_groups(
    batch: LogicalFlowBatch,
    logical_flows: list[LogicalFlowRecord],
    *,
    extraction_mode: GraphExtractionMode,
    slice_count: int = 3,
) -> list[ExtractedFlowGroup]:
    """Group logical flows into graph-construction units."""

    grouped: dict[str, list[LogicalFlowRecord]] = {}
    if extraction_mode == "per_src_ip_within_window":
        for logical_flow in logical_flows:
            grouped.setdefault(logical_flow.src_ip, []).append(logical_flow)
    elif extraction_mode == "short_temporal_slice_src_pair":
        for logical_flow in logical_flows:
            slice_index = _slice_index(batch, logical_flow, slice_count=slice_count)
            group_key = (
                f"{logical_flow.src_ip}|{logical_flow.dst_ip}|{logical_flow.dst_port}|"
                f"{logical_flow.protocol}|slice={slice_index}"
            )
            grouped.setdefault(group_key, []).append(logical_flow)
    elif extraction_mode == "neighborhood_local_burst":
        for logical_flow in logical_flows:
            slice_index = _slice_index(batch, logical_flow, slice_count=slice_count)
            group_key = (
                f"{logical_flow.src_ip}|subnet={_dst_subnet(logical_flow.dst_ip)}|"
                f"port={logical_flow.dst_port}|slice={slice_index}"
            )
            grouped.setdefault(group_key, []).append(logical_flow)
    else:
        raise ValueError(f"Unsupported extraction mode: {extraction_mode}")
    return [
        ExtractedFlowGroup(
            extraction_mode=extraction_mode,
            group_key=group_key,
            logical_flows=tuple(grouped[group_key]),
        )
        for group_key in sorted(grouped)
    ]


__all__ = [
    "ExtractedFlowGroup",
    "GraphExtractionMode",
    "extract_flow_groups",
]
