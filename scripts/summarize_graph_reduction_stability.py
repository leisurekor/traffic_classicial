"""Summarize a small set of graph-reduction runs into one stability table.

This helper reads already-exported real-PCAP graph experiment bundles and
flattens their overall plus per-attack metrics into a compact CSV/Markdown
report. It is intentionally analysis-only and does not retrain any model.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from traffic_graph.pipeline.compare_binary_detection_runs import (
    load_binary_detection_run_bundle,
)


STABILITY_FIELDS: tuple[str, ...] = (
    "run_label",
    "packet_limit",
    "benign_source_set",
    "malicious_source_set",
    "reduction_method",
    "threshold",
    "benign_train_graph_count",
    "benign_test_graph_count",
    "malicious_test_graph_count",
    "overall_fpr",
    "overall_recall",
    "overall_f1",
    "recon_recall",
    "recon_f1",
    "browserhijacking_recall",
    "browserhijacking_f1",
    "ddos_recall",
    "ddos_f1",
    "worst_malicious_source_name",
    "notes",
)


@dataclass(frozen=True, slots=True)
class StabilityRunSpec:
    """One run descriptor passed from the CLI."""

    run_label: str
    packet_limit: int
    run_dir: Path


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Flatten a small set of real-PCAP graph runs into one reduction "
            "stability summary table."
        )
    )
    parser.add_argument(
        "--run-spec",
        action="append",
        required=True,
        help=(
            "Run descriptor formatted as "
            "'run_label|packet_limit|/abs/or/relative/run_dir'. "
            "Repeat this flag for each run to summarize."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the stability summary CSV/Markdown files will be written.",
    )
    parser.add_argument(
        "--basename",
        default="graph_reduction_stability_check",
        help="Basename used for the CSV and Markdown outputs.",
    )
    return parser


def _parse_run_spec(raw_value: str) -> StabilityRunSpec:
    """Parse one CLI run descriptor."""

    parts = raw_value.split("|", 2)
    if len(parts) != 3:
        raise ValueError(
            "Each --run-spec must look like "
            "'run_label|packet_limit|/path/to/run_dir'."
        )
    run_label = parts[0].strip()
    packet_limit = int(parts[1].strip())
    run_dir = Path(parts[2].strip())
    return StabilityRunSpec(run_label=run_label, packet_limit=packet_limit, run_dir=run_dir)


def _metric_or_none(value: object | None) -> float | None:
    """Convert a scalar metric into a float when possible."""

    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_per_attack_metrics(path: Path) -> dict[str, dict[str, float | None]]:
    """Load per-attack metrics from a stable CSV file."""

    rows: dict[str, dict[str, float | None]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            task_name = str(row.get("task_name", "")).strip()
            if not task_name:
                continue
            rows[task_name] = {
                "recall": _metric_or_none(row.get("recall")),
                "f1": _metric_or_none(row.get("f1")),
            }
    return rows


def _source_set_text(values: list[str]) -> str:
    """Render a compact source-set summary."""

    return " | ".join(values)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write the stable CSV export."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STABILITY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(value: object | None) -> str:
    """Format a metric for Markdown output."""

    metric = _metric_or_none(value)
    if metric is None:
        return "unavailable"
    return f"{metric:.6f}"


def _write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a concise Markdown summary."""

    lines = [
        "# Graph Reduction Stability Check",
        "",
        "This summary keeps the real-PCAP input set fixed and only compares",
        "`flow_p90` against the new default `hybrid_max_rank_flow_node_max`",
        "across a small set of representative packet limits.",
        "",
    ]
    rows_by_limit: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        rows_by_limit.setdefault(int(row["packet_limit"]), []).append(row)

    for packet_limit in sorted(rows_by_limit):
        lines.append(f"## packet_limit={packet_limit}")
        lines.append("")
        for row in rows_by_limit[packet_limit]:
            lines.append(
                f"- `{row['run_label']}`: reduction `{row['reduction_method']}`, "
                f"FPR `{_fmt(row['overall_fpr'])}`, overall recall `{_fmt(row['overall_recall'])}`, "
                f"overall F1 `{_fmt(row['overall_f1'])}`, Recon recall `{_fmt(row['recon_recall'])}`, "
                f"BrowserHijacking recall `{_fmt(row['browserhijacking_recall'])}`."
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Run the stability summary export."""

    args = _build_parser().parse_args()
    specs = [_parse_run_spec(raw_value) for raw_value in args.run_spec]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for spec in specs:
        bundle = load_binary_detection_run_bundle(spec.run_dir, backend_name="graph")
        per_attack = _load_per_attack_metrics(spec.run_dir / "per_attack_metrics.csv")

        benign_inputs = list(bundle.metadata.benign_inputs)
        malicious_inputs = list(bundle.metadata.malicious_inputs)
        rows.append(
            {
                "run_label": spec.run_label,
                "packet_limit": spec.packet_limit,
                "benign_source_set": _source_set_text(benign_inputs),
                "malicious_source_set": _source_set_text(malicious_inputs),
                "reduction_method": bundle.metadata.reduction_method,
                "threshold": bundle.metadata.threshold,
                "benign_train_graph_count": bundle.metadata.benign_train_graph_count,
                "benign_test_graph_count": bundle.metadata.benign_test_graph_count,
                "malicious_test_graph_count": bundle.metadata.malicious_test_graph_count,
                "overall_fpr": bundle.summary.overall_metrics.get("false_positive_rate"),
                "overall_recall": bundle.summary.overall_metrics.get("recall"),
                "overall_f1": bundle.summary.overall_metrics.get("f1"),
                "recon_recall": per_attack.get("Recon-HostDiscovery", {}).get("recall"),
                "recon_f1": per_attack.get("Recon-HostDiscovery", {}).get("f1"),
                "browserhijacking_recall": per_attack.get("BrowserHijacking", {}).get("recall"),
                "browserhijacking_f1": per_attack.get("BrowserHijacking", {}).get("f1"),
                "ddos_recall": per_attack.get("DDoS-ICMP_Flood", {}).get("recall"),
                "ddos_f1": per_attack.get("DDoS-ICMP_Flood", {}).get("f1"),
                "worst_malicious_source_name": bundle.metadata.worst_malicious_source_name,
                "notes": bundle.metadata.experiment_label or spec.run_label,
            }
        )

    rows.sort(key=lambda row: (int(row["packet_limit"]), str(row["run_label"])))
    csv_path = output_dir / f"{args.basename}.csv"
    md_path = output_dir / f"{args.basename}.md"
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows)
    print(
        json.dumps(
            {
                "graph_reduction_stability_check_csv": csv_path.as_posix(),
                "graph_reduction_stability_check_md": md_path.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
