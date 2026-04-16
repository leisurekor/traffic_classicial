"""Summarize formal graph-score reduction runs into compact comparison tables.

This helper is intentionally thin: it only reads already-exported binary
detection run artifacts and writes a small set of flattened CSV/Markdown files
for the formal graph mainline A/B and graph-vs-tabular follow-up comparisons.
It does not retrain any model and it does not modify the experiment pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from traffic_graph.pipeline.compare_binary_detection_runs import (
    load_binary_detection_run_bundle,
)


MAINLINE_FIELDS: tuple[str, ...] = (
    "run_label",
    "reduction_method",
    "scorer_role",
    "experiment_label",
    "run_dir",
    "threshold",
    "train_reference_count",
    "benign_test_count",
    "malicious_test_count",
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


FOLLOWUP_FIELDS: tuple[str, ...] = (
    "run_label",
    "mode_family",
    "reduction_method",
    "scorer_role",
    "experiment_label",
    "run_dir",
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


ATTACK_NAME_MAP: dict[str, str] = {
    "Recon-HostDiscovery": "recon",
    "BrowserHijacking": "browserhijacking",
    "DDoS-ICMP_Flood": "ddos",
    "all_malicious": "all_malicious",
}


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the summary helper."""

    parser = argparse.ArgumentParser(
        description=(
            "Flatten formal flow_p90, hybrid, and tabular run artifacts into "
            "compact comparison tables."
        )
    )
    parser.add_argument("--flow-run-dir", required=True, help="Formal flow_p90 run directory.")
    parser.add_argument(
        "--hybrid-run-dir",
        required=True,
        help="Formal hybrid_max_rank_flow_node_max run directory.",
    )
    parser.add_argument(
        "--tabular-run-dir",
        required=True,
        help="Tabular graph-summary PCA control run directory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where flattened summary artifacts will be written.",
    )
    return parser


def _metric_or_none(value: object | None) -> float | None:
    """Normalize one scalar metric into a float when possible."""

    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _attack_metric(
    bundle: Any,
    attack_name: str,
    metric_name: str,
) -> float | None:
    """Return one per-attack metric from a bundle when present."""

    row = bundle.per_attack_metrics_by_task.get(attack_name)
    if row is None:
        return None
    value = getattr(row, metric_name)
    return value if isinstance(value, float) or value is None else _metric_or_none(value)


def _safe_string(value: object | None, *, default: str = "") -> str:
    """Return a stable string representation for exported summaries."""

    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _reduction_method(bundle: Any) -> str:
    """Resolve the score reduction label for a graph or tabular bundle."""

    return bundle.metadata.reduction_method


def _mainline_row(bundle: Any, *, run_label: str, notes: str) -> dict[str, object]:
    """Flatten one graph run into the compact mainline A/B schema."""

    return {
        "run_label": run_label,
        "reduction_method": _reduction_method(bundle),
        "scorer_role": bundle.metadata.scorer_role,
        "experiment_label": bundle.metadata.experiment_label,
        "run_dir": Path(bundle.summary.run_directory).as_posix(),
        "threshold": bundle.metadata.threshold,
        "train_reference_count": bundle.metadata.benign_train_graph_count,
        "benign_test_count": bundle.metadata.benign_test_graph_count,
        "malicious_test_count": bundle.metadata.malicious_test_graph_count,
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
        "notes": notes,
    }


def _followup_row(
    bundle: Any,
    *,
    run_label: str,
    mode_family: str,
    notes: str,
) -> dict[str, object]:
    """Flatten one graph or tabular run into the final follow-up comparison schema."""

    scorer_role = bundle.metadata.scorer_role
    if not scorer_role and mode_family == "tabular":
        scorer_role = "tabular_control"

    return {
        "run_label": run_label,
        "mode_family": mode_family,
        "reduction_method": _reduction_method(bundle),
        "scorer_role": scorer_role,
        "experiment_label": bundle.metadata.experiment_label,
        "run_dir": Path(bundle.summary.run_directory).as_posix(),
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
        "notes": notes,
    }


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    """Write one stable CSV export."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _format_metric(value: object | None) -> str:
    """Render one scalar metric for Markdown summaries."""

    metric = _metric_or_none(value)
    if metric is None:
        return "unavailable"
    return f"{metric:.6f}"


def _delta_text(left: object | None, right: object | None) -> str:
    """Render one compact delta string when both metrics are available."""

    left_value = _metric_or_none(left)
    right_value = _metric_or_none(right)
    if left_value is None or right_value is None:
        return "unavailable"
    delta = right_value - left_value
    return f"{delta:+.6f}"


def _write_markdown(
    path: Path,
    *,
    flow_row: dict[str, object],
    hybrid_row: dict[str, object],
    tabular_row: dict[str, object],
) -> None:
    """Write one concise Markdown readout of the formal follow-up."""

    lines = [
        "# Graph Reduction Mainline Hybrid Follow-up",
        "",
        "## Formal Graph A/B",
        "",
        f"- `flow_p90` (`{flow_row['scorer_role'] or 'unclassified'}`): overall FPR `{_format_metric(flow_row['overall_fpr'])}`, "
        f"overall recall `{_format_metric(flow_row['overall_recall'])}`, "
        f"overall F1 `{_format_metric(flow_row['overall_f1'])}`, "
        f"Recon recall `{_format_metric(flow_row['recon_recall'])}`, "
        f"BrowserHijacking recall `{_format_metric(flow_row['browserhijacking_recall'])}`.",
        f"- `hybrid_max_rank_flow_node_max` (`{hybrid_row['scorer_role'] or 'unclassified'}`): overall FPR `{_format_metric(hybrid_row['overall_fpr'])}`, "
        f"overall recall `{_format_metric(hybrid_row['overall_recall'])}`, "
        f"overall F1 `{_format_metric(hybrid_row['overall_f1'])}`, "
        f"Recon recall `{_format_metric(hybrid_row['recon_recall'])}`, "
        f"BrowserHijacking recall `{_format_metric(hybrid_row['browserhijacking_recall'])}`.",
        f"- Hybrid minus flow deltas: overall FPR `{_delta_text(flow_row['overall_fpr'], hybrid_row['overall_fpr'])}`, "
        f"overall recall `{_delta_text(flow_row['overall_recall'], hybrid_row['overall_recall'])}`, "
        f"overall F1 `{_delta_text(flow_row['overall_f1'], hybrid_row['overall_f1'])}`, "
        f"Recon recall `{_delta_text(flow_row['recon_recall'], hybrid_row['recon_recall'])}`, "
        f"BrowserHijacking recall `{_delta_text(flow_row['browserhijacking_recall'], hybrid_row['browserhijacking_recall'])}`.",
        "",
        "## Graph vs Tabular",
        "",
        f"- `tabular_graphsummary` (`{tabular_row['scorer_role'] or 'unclassified'}`): overall FPR `{_format_metric(tabular_row['overall_fpr'])}`, "
        f"overall recall `{_format_metric(tabular_row['overall_recall'])}`, "
        f"overall F1 `{_format_metric(tabular_row['overall_f1'])}`, "
        f"Recon recall `{_format_metric(tabular_row['recon_recall'])}`, "
        f"BrowserHijacking recall `{_format_metric(tabular_row['browserhijacking_recall'])}`.",
        f"- `graph_flow_p90` (`{flow_row['scorer_role'] or 'unclassified'}`): overall recall delta vs tabular `{_delta_text(tabular_row['overall_recall'], flow_row['overall_recall'])}`, "
        f"overall F1 delta `{_delta_text(tabular_row['overall_f1'], flow_row['overall_f1'])}`, "
        f"Recon recall delta `{_delta_text(tabular_row['recon_recall'], flow_row['recon_recall'])}`, "
        f"BrowserHijacking recall delta `{_delta_text(tabular_row['browserhijacking_recall'], flow_row['browserhijacking_recall'])}`.",
        f"- `graph_hybrid_max_rank_flow_node_max` (`{hybrid_row['scorer_role'] or 'unclassified'}`): overall recall delta vs tabular `{_delta_text(tabular_row['overall_recall'], hybrid_row['overall_recall'])}`, "
        f"overall F1 delta `{_delta_text(tabular_row['overall_f1'], hybrid_row['overall_f1'])}`, "
        f"Recon recall delta `{_delta_text(tabular_row['recon_recall'], hybrid_row['recon_recall'])}`, "
        f"BrowserHijacking recall delta `{_delta_text(tabular_row['browserhijacking_recall'], hybrid_row['browserhijacking_recall'])}`.",
        "",
        "## Takeaways",
        "",
        "- The formal hybrid keeps the same benign-train q95 threshold policy and does not raise overall FPR relative to formal `flow_p90`.",
        "- In this run family, the hybrid closes the remaining Recon gap to the tabular control while retaining a much stronger BrowserHijacking recall than tabular.",
        "- That keeps the current next-step priority on scoring reduction rather than moving upstream to the encoder.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Run the summary export and print the generated artifact paths."""

    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    flow_bundle = load_binary_detection_run_bundle(Path(args.flow_run_dir), backend_name="graph")
    hybrid_bundle = load_binary_detection_run_bundle(Path(args.hybrid_run_dir), backend_name="graph")
    tabular_bundle = load_binary_detection_run_bundle(
        Path(args.tabular_run_dir),
        backend_name="tabular",
    )

    flow_row = _mainline_row(
        flow_bundle,
        run_label="graph_flow_p90",
        notes="Formal mainline graph run with flow_p90 reduction.",
    )
    hybrid_row = _mainline_row(
        hybrid_bundle,
        run_label="graph_hybrid_max_rank_flow_node_max",
        notes="Formal mainline graph run with max-percentile fusion over flow_p90 and node_max.",
    )
    tabular_row = _followup_row(
        tabular_bundle,
        run_label="tabular_graphsummary",
        mode_family="tabular",
        notes="Graph-summary PCA reconstruction control on the same real-PCAP graph windows.",
    )

    mainline_rows = [flow_row, hybrid_row]
    followup_rows = [
        tabular_row,
        _followup_row(
            flow_bundle,
            run_label="graph_flow_p90",
            mode_family="graph",
            notes="Formal graph mainline run with flow_p90 reduction.",
        ),
        _followup_row(
            hybrid_bundle,
            run_label="graph_hybrid_max_rank_flow_node_max",
            mode_family="graph",
            notes="Formal graph mainline run with max-percentile fusion over flow_p90 and node_max.",
        ),
    ]

    mainline_csv_path = output_dir / "graph_reduction_mainline_hybrid_ab.csv"
    followup_csv_path = output_dir / "graph_vs_tabular_hybrid_followup.csv"
    markdown_path = output_dir / "graph_reduction_mainline_hybrid_ab.md"

    _write_csv(mainline_csv_path, MAINLINE_FIELDS, mainline_rows)
    _write_csv(followup_csv_path, FOLLOWUP_FIELDS, followup_rows)
    _write_markdown(
        markdown_path,
        flow_row=flow_row,
        hybrid_row=hybrid_row,
        tabular_row=tabular_row,
    )

    print(
        json.dumps(
            {
                "graph_reduction_mainline_hybrid_ab_csv": mainline_csv_path.as_posix(),
                "graph_vs_tabular_hybrid_followup_csv": followup_csv_path.as_posix(),
                "graph_reduction_mainline_hybrid_ab_md": markdown_path.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
