"""Summarize paper-inspired graph scorer runs into one compact comparison table.

This helper stays intentionally thin: it only reads already-exported run bundles
and flattens a handful of graph-scorer candidates plus one tabular control into
one CSV/Markdown report.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from traffic_graph.pipeline.compare_binary_detection_runs import load_binary_detection_run_bundle
from traffic_graph.pipeline.scorer_roles import normalize_graph_scorer_role


OUTPUT_FIELDS: tuple[str, ...] = (
    "scorer_name",
    "scorer_family",
    "score_definition",
    "threshold_policy",
    "train_reference_count",
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

GRAPH_SCORER_SUMMARY_FIELDS: tuple[str, ...] = (
    "scorer_name",
    "scorer_role",
    "score_definition",
    "protocol_scope",
    "overall_fpr",
    "overall_recall",
    "overall_f1",
    "recon_recall",
    "recon_f1",
    "browserhijacking_recall",
    "browserhijacking_f1",
    "notes",
)


SCORER_DETAILS: dict[str, tuple[str, str, str]] = {
    "flow_p90": (
        "baseline_flow_tail",
        "p90(flow_scores)",
        "Existing flow-aware strong baseline.",
    ),
    "hybrid_max_rank_flow_node_max": (
        "hybrid_default_candidate",
        "max(train-CDF(flow_p90), train-CDF(node_max))",
        "Current default graph-mode candidate.",
    ),
    "decision_topk_flow_node": (
        "flowminer_inspired_decision_pooling",
        "max(train-CDF(flow_p90), train-CDF(node_score_p90))",
        "FlowMiner-inspired decision pooling over two sparse anomaly tails.",
    ),
    "relation_max_flow_server_count": (
        "fgsat_inspired_relation_summary",
        "max(train-CDF(flow_p90), train-CDF(server_node_count))",
        "FG-SAT-inspired thin relation summary using server-side concentration.",
    ),
    "structural_fig_max": (
        "hypervision_icad_inspired_structural_summary",
        "max(train-CDF(edge_density), train-CDF(aggregated_edge_count))",
        "HyperVision/ICAD-inspired lightweight FIG-style structural summary.",
    ),
    "tabular_graphsummary": (
        "tabular_control",
        "graph_summary PCA reconstruction control",
        "External coarse-summary control baseline.",
    ),
}

SCORER_ROLE_NOTES: dict[str, str] = {
    "hybrid_max_rank_flow_node_max": (
        "Current default graph-mode scorer because it keeps the strongest balance "
        "between overall performance, Recon recovery, and BrowserHijacking retention."
    ),
    "flow_p90": (
        "Recommended rollback / ablation scorer because it is simpler, stable, and "
        "easy to explain while remaining strong on the same real-PCAP protocol."
    ),
    "decision_topk_flow_node": (
        "Recon-leaning experimental scorer: it tends to recover Recon tails but "
        "gives up some BrowserHijacking performance, so it stays opt-in."
    ),
}

PROTOCOL_SCOPE = (
    "representative real-PCAP protocol: confirmed benign trio + Recon/DDoS/"
    "BrowserHijacking, packet_limit=20000, q95 benign-train threshold"
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the paper-inspired scorer summarizer."""

    parser = argparse.ArgumentParser(
        description=(
            "Flatten paper-inspired graph scorer runs and one tabular control into "
            "a compact comparison table."
        )
    )
    parser.add_argument("--flow-run-dir", required=True, help="Formal flow_p90 run directory.")
    parser.add_argument(
        "--hybrid-run-dir",
        required=True,
        help="Formal hybrid_max_rank_flow_node_max run directory.",
    )
    parser.add_argument(
        "--decision-run-dir",
        required=True,
        help="Formal decision_topk_flow_node run directory.",
    )
    parser.add_argument(
        "--relation-run-dir",
        required=True,
        help="Formal relation_max_flow_server_count run directory.",
    )
    parser.add_argument(
        "--structural-run-dir",
        required=True,
        help="Formal structural_fig_max run directory.",
    )
    parser.add_argument(
        "--tabular-run-dir",
        required=True,
        help="Graph-summary tabular PCA control run directory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the flattened scorer-family summary will be written.",
    )
    return parser


def _metric_or_none(value: object | None) -> float | None:
    """Normalize one metric-like scalar into a float when available."""

    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _attack_metric(bundle: Any, attack_name: str, metric_name: str) -> float | None:
    """Return one per-attack metric from a normalized run bundle when present."""

    row = bundle.per_attack_metrics_by_task.get(attack_name)
    if row is None:
        return None
    return _metric_or_none(getattr(row, metric_name))


def _row_from_bundle(bundle: Any, *, scorer_name: str) -> dict[str, object]:
    """Flatten one normalized run bundle into the scorer-family CSV schema."""

    family, definition, base_notes = SCORER_DETAILS[scorer_name]
    return {
        "scorer_name": scorer_name,
        "scorer_family": family,
        "score_definition": definition,
        "threshold_policy": "q95 benign train reference",
        "train_reference_count": bundle.metadata.benign_train_graph_count,
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
        "notes": base_notes,
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
    """Render one metric for Markdown summaries."""

    metric = _metric_or_none(value)
    if metric is None:
        return "unavailable"
    return f"{metric:.6f}"


def _write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a concise Markdown summary for the scorer-family comparison."""

    lines = [
        "# Paper-Inspired Scorer Family Comparison",
        "",
        "This summary keeps the representative real-PCAP protocol fixed at the",
        "confirmed benign set + Recon / DDoS / BrowserHijacking malicious set and",
        "only compares graph-level scoring families.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['scorer_name']}",
                "",
                f"- Family: `{row['scorer_family']}`",
                f"- Definition: `{row['score_definition']}`",
                f"- Overall FPR: `{_fmt_metric(row['overall_fpr'])}`",
                f"- Overall recall / F1: `{_fmt_metric(row['overall_recall'])}` / `{_fmt_metric(row['overall_f1'])}`",
                f"- Recon recall / F1: `{_fmt_metric(row['recon_recall'])}` / `{_fmt_metric(row['recon_f1'])}`",
                f"- BrowserHijacking recall / F1: `{_fmt_metric(row['browserhijacking_recall'])}` / `{_fmt_metric(row['browserhijacking_f1'])}`",
                f"- DDoS recall / F1: `{_fmt_metric(row['ddos_recall'])}` / `{_fmt_metric(row['ddos_f1'])}`",
                f"- Worst malicious source: `{row['worst_malicious_source_name']}`",
                f"- Notes: {row['notes']}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _graph_scorer_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Collapse the scorer-family comparison into one concise graph scorer summary."""

    lookup = {str(row["scorer_name"]): row for row in rows}
    summary_rows: list[dict[str, object]] = []
    for scorer_name in (
        "hybrid_max_rank_flow_node_max",
        "flow_p90",
        "decision_topk_flow_node",
    ):
        row = lookup.get(scorer_name)
        if row is None:
            continue
        scorer_role = normalize_graph_scorer_role(scorer_name)
        summary_rows.append(
            {
                "scorer_name": scorer_name,
                "scorer_role": scorer_role,
                "score_definition": row["score_definition"],
                "protocol_scope": PROTOCOL_SCOPE,
                "overall_fpr": row["overall_fpr"],
                "overall_recall": row["overall_recall"],
                "overall_f1": row["overall_f1"],
                "recon_recall": row["recon_recall"],
                "recon_f1": row["recon_f1"],
                "browserhijacking_recall": row["browserhijacking_recall"],
                "browserhijacking_f1": row["browserhijacking_f1"],
                "notes": SCORER_ROLE_NOTES[scorer_name],
            }
        )
    return summary_rows


def _write_graph_scorer_summary_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a short scorer-selection note for the current graph scorer family."""

    lines = [
        "# Graph Scorer Family Summary",
        "",
        "This note captures the current scorer selection after the real-PCAP",
        "mainline follow-ups and the narrow hybrid-vs-decision trade-off check.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['scorer_name']}",
                "",
                f"- Role: `{row['scorer_role']}`",
                f"- Definition: `{row['score_definition']}`",
                f"- Protocol scope: {row['protocol_scope']}",
                f"- Overall FPR / recall / F1: `{_fmt_metric(row['overall_fpr'])}` / `{_fmt_metric(row['overall_recall'])}` / `{_fmt_metric(row['overall_f1'])}`",
                f"- Recon recall / F1: `{_fmt_metric(row['recon_recall'])}` / `{_fmt_metric(row['recon_f1'])}`",
                f"- BrowserHijacking recall / F1: `{_fmt_metric(row['browserhijacking_recall'])}` / `{_fmt_metric(row['browserhijacking_f1'])}`",
                f"- Notes: {row['notes']}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> int:
    """Load scorer-family runs and export one compact comparison bundle."""

    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)
    bundles = [
        (
            "flow_p90",
            load_binary_detection_run_bundle(args.flow_run_dir, backend_name="graph"),
        ),
        (
            "hybrid_max_rank_flow_node_max",
            load_binary_detection_run_bundle(args.hybrid_run_dir, backend_name="graph"),
        ),
        (
            "decision_topk_flow_node",
            load_binary_detection_run_bundle(args.decision_run_dir, backend_name="graph"),
        ),
        (
            "relation_max_flow_server_count",
            load_binary_detection_run_bundle(args.relation_run_dir, backend_name="graph"),
        ),
        (
            "structural_fig_max",
            load_binary_detection_run_bundle(args.structural_run_dir, backend_name="graph"),
        ),
        (
            "tabular_graphsummary",
            load_binary_detection_run_bundle(args.tabular_run_dir, backend_name="tabular"),
        ),
    ]
    rows = [_row_from_bundle(bundle, scorer_name=scorer_name) for scorer_name, bundle in bundles]
    csv_path = output_dir / "paper_inspired_scorer_family_comparison.csv"
    md_path = output_dir / "paper_inspired_scorer_family_comparison.md"
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows)
    graph_summary_rows = _graph_scorer_summary_rows(rows)
    graph_summary_csv_path = output_dir / "graph_scorer_family_summary.csv"
    graph_summary_md_path = output_dir / "graph_scorer_family_summary.md"
    graph_summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with graph_summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GRAPH_SCORER_SUMMARY_FIELDS)
        writer.writeheader()
        for row in graph_summary_rows:
            writer.writerow(row)
    _write_graph_scorer_summary_markdown(graph_summary_md_path, graph_summary_rows)
    print(
        {
            "paper_inspired_scorer_family_comparison_csv": csv_path.as_posix(),
            "paper_inspired_scorer_family_comparison_md": md_path.as_posix(),
            "graph_scorer_family_summary_csv": graph_summary_csv_path.as_posix(),
            "graph_scorer_family_summary_md": graph_summary_md_path.as_posix(),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
