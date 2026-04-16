"""Alignment between locally extracted PCAP flows and official CTU-13 labels."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any
import csv
from statistics import median

from traffic_graph.data.dataset import FlowDataset
from traffic_graph.data.schema import FlowRecord
from traffic_graph.datasets.ctu13 import CTU13LabeledFlow


@dataclass(frozen=True, slots=True)
class CTU13AlignedFlowLabel:
    """Auditable label-alignment result for one locally extracted flow."""

    scenario_id: str
    flow_id: str
    protocol: str
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    start_time: str
    end_time: str
    aligned_label: str
    label_text: str
    alignment_status: str
    alignment_score: float
    matched_reference_start: str | None = None
    matched_reference_end: str | None = None

    def to_row(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "flow_id": self.flow_id,
            "protocol": self.protocol,
            "src_ip": self.src_ip,
            "src_port": self.src_port,
            "dst_ip": self.dst_ip,
            "dst_port": self.dst_port,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "aligned_label": self.aligned_label,
            "label_text": self.label_text,
            "alignment_status": self.alignment_status,
            "alignment_score": self.alignment_score,
            "matched_reference_start": self.matched_reference_start,
            "matched_reference_end": self.matched_reference_end,
        }


@dataclass(frozen=True, slots=True)
class CTU13AlignmentSummary:
    """Compact scenario-level label alignment summary."""

    scenario_id: str
    total_flows: int
    benign_count: int
    malicious_count: int
    unknown_count: int
    unaligned_count: int
    alignment_rate: float

    def to_row(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "total_flows": self.total_flows,
            "benign_count": self.benign_count,
            "malicious_count": self.malicious_count,
            "unknown_count": self.unknown_count,
            "unaligned_count": self.unaligned_count,
            "alignment_rate": self.alignment_rate,
        }


def _canonical_key(record: FlowRecord | CTU13LabeledFlow) -> tuple[tuple[str, int], tuple[str, int], str]:
    endpoints = sorted(
        ((record.src_ip, int(record.src_port)), (record.dst_ip, int(record.dst_port))),
        key=lambda item: (item[0], item[1]),
    )
    return (endpoints[0], endpoints[1], str(record.protocol).lower())


def _time_overlap_seconds(
    left_start,
    left_end,
    right_start,
    right_end,
) -> float:
    start = max(left_start, right_start)
    end = min(left_end, right_end)
    return max((end - start).total_seconds(), 0.0)


def _estimate_label_time_offset_seconds(
    flow_dataset: FlowDataset,
    labeled_flows: list[CTU13LabeledFlow],
) -> float:
    """Estimate a stable label-to-pcap clock offset from matching 5-tuples."""

    flows_by_key: dict[
        tuple[tuple[str, int], tuple[str, int], str],
        list[FlowRecord],
    ] = defaultdict(list)
    labels_by_key: dict[
        tuple[tuple[str, int], tuple[str, int], str],
        list[CTU13LabeledFlow],
    ] = defaultdict(list)
    for record in flow_dataset.records:
        flows_by_key[_canonical_key(record)].append(record)
    for labeled_flow in labeled_flows:
        labels_by_key[_canonical_key(labeled_flow)].append(labeled_flow)
    for records in flows_by_key.values():
        records.sort(key=lambda item: item.start_time)
    for records in labels_by_key.values():
        records.sort(key=lambda item: item.start_time)

    deltas: list[float] = []
    for key, records in flows_by_key.items():
        labeled_records = labels_by_key.get(key)
        if not labeled_records:
            continue
        deltas.append((records[0].start_time - labeled_records[0].start_time).total_seconds())
    if not deltas:
        return 0.0
    return float(median(deltas))


def align_flow_dataset_to_ctu13_labels(
    flow_dataset: FlowDataset,
    labeled_flows: list[CTU13LabeledFlow],
    *,
    scenario_id: str,
    time_tolerance_seconds: float = 5.0,
) -> tuple[list[CTU13AlignedFlowLabel], CTU13AlignmentSummary]:
    """Align extracted flows to official CTU-13 labels using 5-tuple + time overlap."""

    labels_by_key: dict[
        tuple[tuple[str, int], tuple[str, int], str],
        list[CTU13LabeledFlow],
    ] = defaultdict(list)
    for labeled_flow in labeled_flows:
        labels_by_key[_canonical_key(labeled_flow)].append(labeled_flow)

    label_time_offset_seconds = _estimate_label_time_offset_seconds(flow_dataset, labeled_flows)
    label_time_offset = timedelta(seconds=label_time_offset_seconds)

    results: list[CTU13AlignedFlowLabel] = []
    benign_count = 0
    malicious_count = 0
    unknown_count = 0
    unaligned_count = 0

    for record in flow_dataset.records:
        candidates = labels_by_key.get(_canonical_key(record), [])
        best_label: CTU13LabeledFlow | None = None
        best_score = -1.0
        for candidate in candidates:
            candidate_start = candidate.start_time + label_time_offset
            candidate_end = candidate.end_time + label_time_offset
            overlap = _time_overlap_seconds(
                record.start_time,
                record.end_time,
                candidate_start,
                candidate_end,
            )
            if overlap > 0.0:
                score = overlap + 1.0
            else:
                gap = min(
                    abs((record.start_time - candidate_start).total_seconds()),
                    abs((record.end_time - candidate_end).total_seconds()),
                )
                if gap > time_tolerance_seconds:
                    continue
                score = max(time_tolerance_seconds - gap, 0.0) / max(time_tolerance_seconds, 1e-6)
            if score > best_score:
                best_score = score
                best_label = candidate

        if best_label is None:
            aligned_label = "unknown"
            label_text = ""
            alignment_status = "unaligned"
            score = 0.0
            unaligned_count += 1
            unknown_count += 1
            matched_start = None
            matched_end = None
        else:
            aligned_label = best_label.binary_label
            label_text = best_label.label_text
            alignment_status = "aligned"
            score = float(best_score)
            matched_start = (best_label.start_time + label_time_offset).isoformat()
            matched_end = (best_label.end_time + label_time_offset).isoformat()
            if aligned_label == "benign":
                benign_count += 1
            elif aligned_label == "malicious":
                malicious_count += 1
            else:
                unknown_count += 1

        results.append(
            CTU13AlignedFlowLabel(
                scenario_id=scenario_id,
                flow_id=record.flow_id,
                protocol=record.protocol,
                src_ip=record.src_ip,
                src_port=record.src_port,
                dst_ip=record.dst_ip,
                dst_port=record.dst_port,
                start_time=record.start_time.isoformat(),
                end_time=record.end_time.isoformat(),
                aligned_label=aligned_label,
                label_text=label_text,
                alignment_status=alignment_status,
                alignment_score=score,
                matched_reference_start=matched_start,
                matched_reference_end=matched_end,
            )
        )

    total_flows = len(flow_dataset.records)
    summary = CTU13AlignmentSummary(
        scenario_id=scenario_id,
        total_flows=total_flows,
        benign_count=benign_count,
        malicious_count=malicious_count,
        unknown_count=unknown_count,
        unaligned_count=unaligned_count,
        alignment_rate=(total_flows - unaligned_count) / total_flows if total_flows else 0.0,
    )
    return results, summary


def write_alignment_summary_csv(
    summaries: list[CTU13AlignmentSummary],
    path: str | Path,
) -> None:
    """Write one compact alignment summary CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].to_row().keys()) if summaries else [
            "scenario_id",
            "total_flows",
            "benign_count",
            "malicious_count",
            "unknown_count",
            "unaligned_count",
            "alignment_rate",
        ])
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary.to_row())


def render_alignment_summary_markdown(summaries: list[CTU13AlignmentSummary]) -> str:
    """Render one short markdown summary for auditability."""

    lines = [
        "# CTU-13 Flow Label Alignment Summary",
        "",
        "| scenario_id | total_flows | benign | malicious | unknown | unaligned | alignment_rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary.scenario_id} | {summary.total_flows} | {summary.benign_count} | "
            f"{summary.malicious_count} | {summary.unknown_count} | {summary.unaligned_count} | "
            f"{summary.alignment_rate:.4f} |"
        )
    return "\n".join(lines) + "\n"


def write_alignment_summary_markdown(
    summaries: list[CTU13AlignmentSummary],
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_alignment_summary_markdown(summaries),
        encoding="utf-8",
    )


__all__ = [
    "CTU13AlignedFlowLabel",
    "CTU13AlignmentSummary",
    "align_flow_dataset_to_ctu13_labels",
    "render_alignment_summary_markdown",
    "write_alignment_summary_csv",
    "write_alignment_summary_markdown",
]
