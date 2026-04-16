"""Summarize the narrow hybrid-vs-decision scorer trade-off across seeds.

This helper intentionally stays thin and read-only. It consumes already
exported PCAP graph experiment bundles and flattens a small set of scorer runs
into one CSV/Markdown comparison focused on Recon vs BrowserHijacking trade-offs.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from traffic_graph.pipeline.compare_binary_detection_runs import load_binary_detection_run_bundle


OUTPUT_FIELDS: tuple[str, ...] = (
    "run_label",
    "random_seed",
    "scorer_name",
    "threshold",
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


@dataclass(frozen=True)
class RunSpec:
    """Describe one exported run that should appear in the trade-off table."""

    run_label: str
    scorer_name: str
    random_seed: int
    run_dir: Path
    backend_name: str = "graph"
    notes: str = ""


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI for the narrow hybrid-vs-decision trade-off summarizer."""

    parser = argparse.ArgumentParser(
        description=(
            "Flatten one flow_p90 anchor plus hybrid/decision multi-seed runs into "
            "a narrow trade-off table."
        )
    )
    parser.add_argument("--flow-run-dir", required=True, help="Reference flow_p90 run directory.")
    parser.add_argument(
        "--hybrid-run-dir",
        action="append",
        default=[],
        help=(
            "Hybrid run specification in the form seed=RUN_DIR. "
            "Repeat for each seed you want to include."
        ),
    )
    parser.add_argument(
        "--decision-run-dir",
        action="append",
        default=[],
        help=(
            "Decision-topk run specification in the form seed=RUN_DIR. "
            "Repeat for each seed you want to include."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the flattened trade-off summary will be written.",
    )
    return parser


def _metric_or_none(value: object | None) -> float | None:
    """Normalize one metric-like scalar into float when available."""

    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _attack_metric(bundle: Any, attack_name: str, metric_name: str) -> float | None:
    """Return one per-attack metric when it exists in the loaded bundle."""

    row = bundle.per_attack_metrics_by_task.get(attack_name)
    if row is None:
        return None
    return _metric_or_none(getattr(row, metric_name))


def _parse_seed_specs(specs: list[str], *, scorer_name: str, label_prefix: str) -> list[RunSpec]:
    """Parse repeated seed=path CLI arguments into stable run specs."""

    parsed: list[RunSpec] = []
    for raw_spec in specs:
        if "=" not in raw_spec:
            raise ValueError(
                f"Run specification '{raw_spec}' must be in the form seed=/path/to/run."
            )
        seed_text, path_text = raw_spec.split("=", maxsplit=1)
        parsed.append(
            RunSpec(
                run_label=f"{label_prefix}_seed{int(seed_text)}",
                scorer_name=scorer_name,
                random_seed=int(seed_text),
                run_dir=Path(path_text),
                notes=f"{scorer_name} run for seed {int(seed_text)}.",
            )
        )
    return parsed


def _row_from_bundle(bundle: Any, spec: RunSpec) -> dict[str, object]:
    """Flatten one normalized run bundle into the narrow trade-off schema."""

    return {
        "run_label": spec.run_label,
        "random_seed": spec.random_seed,
        "scorer_name": spec.scorer_name,
        "threshold": bundle.summary.threshold,
        "overall_fpr": bundle.summary.overall_metrics.get("false_positive_rate"),
        "overall_recall": bundle.summary.overall_metrics.get("recall"),
        "overall_f1": bundle.summary.overall_metrics.get("f1"),
        "recon_recall": _attack_metric(bundle, "Recon-HostDiscovery", "recall"),
        "recon_f1": _attack_metric(bundle, "Recon-HostDiscovery", "f1"),
        "browserhijacking_recall": _attack_metric(bundle, "BrowserHijacking", "recall"),
        "browserhijacking_f1": _attack_metric(bundle, "BrowserHijacking", "f1"),
        "ddos_recall": _attack_metric(bundle, "DDoS-ICMP_Flood", "recall"),
        "ddos_f1": _attack_metric(bundle, "DDoS-ICMP_Flood", "f1"),
        "worst_malicious_source_name": bundle.metadata.worst_malicious_source_name,
        "notes": spec.notes,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write one stable CSV export."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt_metric(value: object | None) -> str:
    """Render one metric for the markdown summary."""

    metric = _metric_or_none(value)
    if metric is None:
        return "unavailable"
    return f"{metric:.6f}"


def _write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a concise Markdown summary of the hybrid-vs-decision trade-off."""

    hybrid_rows = [row for row in rows if row["scorer_name"] == "hybrid_max_rank_flow_node_max"]
    decision_rows = [row for row in rows if row["scorer_name"] == "decision_topk_flow_node"]
    flow_rows = [row for row in rows if row["scorer_name"] == "flow_p90"]

    lines = [
        "# Hybrid vs Decision Trade-off",
        "",
        "This report keeps the real-PCAP protocol fixed and only compares the",
        "current default candidate (`hybrid_max_rank_flow_node_max`) against the",
        "FlowMiner-inspired decision-style pooling candidate (`decision_topk_flow_node`).",
        "",
        "## Included runs",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"- `{row['run_label']}`: scorer `{row['scorer_name']}`, seed `{row['random_seed']}`, "
                f"FPR `{_fmt_metric(row['overall_fpr'])}`, overall F1 `{_fmt_metric(row['overall_f1'])}`, "
                f"Recon F1 `{_fmt_metric(row['recon_f1'])}`, BrowserHijacking F1 `{_fmt_metric(row['browserhijacking_f1'])}`",
            ]
        )

    if flow_rows:
        flow_row = flow_rows[0]
        lines.extend(
            [
                "",
                "## Anchor",
                "",
                f"- `flow_p90` anchor keeps FPR `{_fmt_metric(flow_row['overall_fpr'])}` with "
                f"Recon F1 `{_fmt_metric(flow_row['recon_f1'])}` and BrowserHijacking F1 "
                f"`{_fmt_metric(flow_row['browserhijacking_f1'])}`.",
            ]
        )

    if hybrid_rows and decision_rows:
        lines.extend(
            [
                "",
                "## Readout",
                "",
                "- Compare the hybrid rows against the decision rows at the same seed.",
                "- If decision keeps lifting Recon while BrowserHijacking repeatedly drops, "
                "that is a stable trade-off rather than a one-off fluctuation.",
                "- If hybrid stays at or below the same FPR with stronger BrowserHijacking, "
                "it remains the better default candidate.",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> int:
    """Load existing runs and export one narrow trade-off bundle."""

    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)

    specs: list[RunSpec] = [
        RunSpec(
            run_label="flow_anchor_seed42",
            scorer_name="flow_p90",
            random_seed=42,
            run_dir=Path(args.flow_run_dir),
            backend_name="graph",
            notes="Reference flow_p90 anchor from the representative seed-42 run.",
        )
    ]
    specs.extend(
        _parse_seed_specs(
            list(args.hybrid_run_dir),
            scorer_name="hybrid_max_rank_flow_node_max",
            label_prefix="hybrid",
        )
    )
    specs.extend(
        _parse_seed_specs(
            list(args.decision_run_dir),
            scorer_name="decision_topk_flow_node",
            label_prefix="decision_topk",
        )
    )

    rows: list[dict[str, object]] = []
    for spec in specs:
        bundle = load_binary_detection_run_bundle(spec.run_dir, backend_name=spec.backend_name)
        rows.append(_row_from_bundle(bundle, spec))

    rows.sort(key=lambda row: (int(row["random_seed"]), str(row["scorer_name"])))

    csv_path = output_dir / "hybrid_vs_decision_tradeoff.csv"
    md_path = output_dir / "hybrid_vs_decision_tradeoff.md"
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows)
    print(
        {
            "hybrid_vs_decision_tradeoff_csv": csv_path.as_posix(),
            "hybrid_vs_decision_tradeoff_md": md_path.as_posix(),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
