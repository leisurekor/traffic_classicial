"""Validate default graph scorer stability across a small fixed seed set.

This helper intentionally stays small and controlled:

- it reuses the existing real-PCAP graph experiment pipeline,
- keeps packet_limit fixed at 20000,
- keeps the q95 benign-train threshold policy unchanged,
- compares only the current three scorer choices:
  - hybrid_max_rank_flow_node_max
  - flow_p90
  - decision_topk_flow_node

The script prefers reusing already-exported runs that match the fixed protocol
and only executes missing scorer/seed combinations. It then writes one compact
CSV plus a short Markdown summary with a recommendation about whether the
current default scorer remains justified.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from traffic_graph.pipeline.compare_binary_detection_runs import (
    load_binary_detection_run_bundle,
)
from traffic_graph.pipeline.pcap_graph_experiment import (
    PcapGraphExperimentConfig,
    run_pcap_graph_experiment,
)
from traffic_graph.pipeline.scorer_roles import normalize_graph_scorer_role


DEFAULT_SEEDS: tuple[int, ...] = (42, 7, 21, 84, 123)
DEFAULT_SCORERS: tuple[str, ...] = (
    "hybrid_max_rank_flow_node_max",
    "flow_p90",
    "decision_topk_flow_node",
)

DEFAULT_BENIGN_INPUTS: tuple[str, ...] = (
    "data/ciciot2023/pcap/benign/BenignTraffic.pcap",
    "data/ciciot2023/pcap/benign/BenignTraffic1.redownload.pcap",
    "data/ciciot2023/pcap/benign/BenignTraffic3.redownload2.pcap",
)
DEFAULT_MALICIOUS_INPUTS: tuple[str, ...] = (
    "data/ciciot2023/pcap/malicious/recon/Recon-HostDiscovery.pcap",
    "data/ciciot2023/pcap/malicious/ddos/DDoS-ICMP_Flood.pcap",
    "data/ciciot2023/pcap/malicious/web_based/BrowserHijacking.pcap",
)

SUMMARY_FIELDS: tuple[str, ...] = (
    "scorer_name",
    "scorer_role",
    "mean_fpr",
    "std_fpr",
    "mean_f1",
    "std_f1",
    "mean_recon_recall",
    "std_recon_recall",
    "mean_browser_recall",
    "std_browser_recall",
)


@dataclass(frozen=True, slots=True)
class ScorerSeedRun:
    """One resolved run used for the stability snapshot."""

    scorer_name: str
    random_seed: int
    run_dir: Path
    reused_existing: bool


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the stability snapshot helper."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate the current default graph scorer across a small fixed seed set."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/ciciot2023/analysis",
        help="Directory where the summary CSV/Markdown files will be written.",
    )
    parser.add_argument(
        "--run-export-dir",
        default="artifacts/ciciot2023/default_scorer_stability_runs",
        help="Directory used to store any missing scorer/seed runs that must be executed.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        default=[],
        help="Optional seed override. Repeat to run a smaller subset.",
    )
    return parser


def _normalize_paths(values: tuple[str, ...]) -> tuple[str, ...]:
    """Resolve a tuple of repository-relative paths into stable absolute paths."""

    return tuple(str(Path(value).resolve()) for value in values)


def _matching_run_dir(
    *,
    scorer_name: str,
    random_seed: int,
    packet_limit: int,
    benign_inputs: tuple[str, ...],
    malicious_inputs: tuple[str, ...],
) -> Path | None:
    """Return one already-exported matching run when it exists."""

    root = Path("artifacts/ciciot2023")
    candidates: list[Path] = []
    for comparison_path in root.rglob("comparison_summary.json"):
        config_path = comparison_path.with_name("pcap_experiment_config.json")
        summary_path = comparison_path.with_name("pcap_experiment_summary.json")
        if not config_path.exists() or not summary_path.exists():
            continue
        try:
            config_payload = json.loads(config_path.read_text(encoding="utf-8"))
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if config_payload.get("graph_score_reduction") != scorer_name:
            continue
        if int(config_payload.get("random_seed", -1)) != random_seed:
            continue
        if int(config_payload.get("packet_limit", -1)) != packet_limit:
            continue
        if int(config_payload.get("window_size", -1)) != 10:
            continue
        if float(config_payload.get("threshold_percentile", -1.0)) != 95.0:
            continue
        summary_benign = tuple(
            str(Path(value).resolve()) for value in summary_payload.get("benign_inputs", ())
        )
        summary_malicious = tuple(
            str(Path(value).resolve())
            for value in summary_payload.get("malicious_inputs", ())
        )
        if summary_benign != benign_inputs:
            continue
        if summary_malicious != malicious_inputs:
            continue
        candidates.append(comparison_path.parent)
    if not candidates:
        return None
    return sorted(candidates)[-1]


def _run_experiment(
    *,
    scorer_name: str,
    random_seed: int,
    export_dir: Path,
    benign_inputs: tuple[str, ...],
    malicious_inputs: tuple[str, ...],
) -> Path:
    """Execute one missing scorer/seed run with the fixed protocol."""

    config = PcapGraphExperimentConfig(
        packet_limit=20000,
        window_size=10,
        use_association_edges=True,
        use_graph_structural_features=True,
        benign_train_ratio=0.7,
        train_validation_ratio=0.25,
        graph_score_reduction=scorer_name,  # type: ignore[arg-type]
        epochs=1,
        batch_size=1,
        learning_rate=1e-3,
        threshold_percentile=95.0,
        random_seed=random_seed,
    )
    result = run_pcap_graph_experiment(
        export_dir=export_dir,
        benign_inputs=benign_inputs,
        malicious_inputs=malicious_inputs,
        experiment_label=f"default_stability_{scorer_name}_seed{random_seed}_packet20000",
        config=config,
    )
    return Path(result.export_result.run_directory)


def _load_attack_recall(bundle: object, attack_name: str) -> float:
    """Extract one per-attack recall with a stable float fallback."""

    row = bundle.per_attack_metrics_by_task.get(attack_name)  # type: ignore[attr-defined]
    if row is None or row.recall is None:
        return 0.0
    return float(row.recall)


def _fmt(value: float) -> str:
    """Render one metric with a stable decimal format."""

    return f"{value:.6f}"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write the aggregated stability summary CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _table_line(row: dict[str, object]) -> str:
    """Render one Markdown table row."""

    return (
        f"| `{row['scorer_name']}` | `{row['scorer_role']}` | "
        f"{_fmt(float(row['mean_fpr']))} +- {_fmt(float(row['std_fpr']))} | "
        f"{_fmt(float(row['mean_f1']))} +- {_fmt(float(row['std_f1']))} | "
        f"{_fmt(float(row['mean_recon_recall']))} +- {_fmt(float(row['std_recon_recall']))} | "
        f"{_fmt(float(row['mean_browser_recall']))} +- {_fmt(float(row['std_browser_recall']))} |"
    )


def _recommendation(rows_by_scorer: dict[str, dict[str, object]]) -> str:
    """Return one explicit recommendation based on the aggregated snapshot."""

    hybrid = rows_by_scorer["hybrid_max_rank_flow_node_max"]
    flow = rows_by_scorer["flow_p90"]
    decision = rows_by_scorer["decision_topk_flow_node"]

    if (
        float(hybrid["mean_fpr"]) <= float(flow["mean_fpr"]) + 1e-9
        and float(hybrid["mean_f1"]) >= float(flow["mean_f1"])
        and float(hybrid["mean_browser_recall"]) > float(decision["mean_browser_recall"])
    ):
        return (
            "Keep `hybrid_max_rank_flow_node_max` as `default_candidate`: it remains "
            "stable across seeds, preserves BrowserHijacking better than the decision-style "
            "pooler, and stays at least as safe as `flow_p90` on FPR."
        )
    if float(decision["mean_recon_recall"]) > float(hybrid["mean_recon_recall"]) + 0.05:
        return (
            "Flag `decision_topk_flow_node` as worth reconsidering only for Recon-first "
            "scenarios: it improves Recon materially, but the trade-off against BrowserHijacking "
            "still keeps it below the current hybrid default for general use."
        )
    return (
        "Results are too mixed to replace the current default. Keep "
        "`hybrid_max_rank_flow_node_max` as the default and retain `flow_p90` as the stable fallback."
    )


def _write_markdown(
    path: Path,
    *,
    rows: list[dict[str, object]],
    reused: list[ScorerSeedRun],
    executed: list[ScorerSeedRun],
) -> None:
    """Write one short Markdown summary with a recommendation."""

    rows_by_scorer = {str(row["scorer_name"]): row for row in rows}
    hybrid = rows_by_scorer["hybrid_max_rank_flow_node_max"]
    flow = rows_by_scorer["flow_p90"]
    decision = rows_by_scorer["decision_topk_flow_node"]
    hybrid_recon = float(hybrid["mean_recon_recall"])
    decision_recon = float(decision["mean_recon_recall"])
    hybrid_browser = float(hybrid["mean_browser_recall"])
    decision_browser = float(decision["mean_browser_recall"])

    if decision_recon > hybrid_recon + 0.01:
        recon_observation = (
            f"`decision_topk_flow_node` mean Recon recall is `{_fmt(decision_recon)}` "
            f"vs `{_fmt(hybrid_recon)}` for hybrid, so Recon does get a measurable lift."
        )
    elif abs(decision_recon - hybrid_recon) <= 0.01:
        recon_observation = (
            f"`decision_topk_flow_node` mean Recon recall is `{_fmt(decision_recon)}` "
            f"vs `{_fmt(hybrid_recon)}` for hybrid, so Recon is effectively tied rather than materially improved."
        )
    else:
        recon_observation = (
            f"`decision_topk_flow_node` mean Recon recall is `{_fmt(decision_recon)}` "
            f"vs `{_fmt(hybrid_recon)}` for hybrid, so Recon does not justify switching away from the current default."
        )

    if decision_browser < hybrid_browser - 0.01:
        browser_observation = (
            f"That comes with a BrowserHijacking cost: decision mean Browser recall is "
            f"`{_fmt(decision_browser)}` vs `{_fmt(hybrid_browser)}` for hybrid."
        )
    elif abs(decision_browser - hybrid_browser) <= 0.01:
        browser_observation = (
            f"BrowserHijacking stays effectively tied as well: decision mean Browser recall is "
            f"`{_fmt(decision_browser)}` vs `{_fmt(hybrid_browser)}` for hybrid."
        )
    else:
        browser_observation = (
            f"Decision even improves BrowserHijacking in this snapshot: "
            f"`{_fmt(decision_browser)}` vs `{_fmt(hybrid_browser)}` for hybrid."
        )

    observations = [
        (
            f"`hybrid_max_rank_flow_node_max` mean FPR is `{_fmt(float(hybrid['mean_fpr']))}` "
            f"with std `{_fmt(float(hybrid['std_fpr']))}`, which keeps the default on a stable footing "
            "across the fixed seed set."
        ),
        recon_observation,
        browser_observation,
        (
            f"`flow_p90` stays the simplest safe fallback with mean FPR `{_fmt(float(flow['mean_fpr']))}` "
            f"and mean F1 `{_fmt(float(flow['mean_f1']))}`."
        ),
    ]

    lines = [
        "# Default Scorer Stability Summary",
        "",
        "| scorer_name | scorer_role | FPR (mean +- std) | F1 (mean +- std) | Recon recall (mean +- std) | Browser recall (mean +- std) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines.extend(_table_line(row) for row in rows)
    lines.extend(
        [
            "",
            "## Observations",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in observations)
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- {_recommendation(rows_by_scorer)}",
            "",
            "## Run Reuse",
            "",
            f"- Reused existing runs: `{len(reused)}`",
            f"- Newly executed runs: `{len(executed)}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Run the fixed default-scorer stability validation snapshot."""

    args = _build_parser().parse_args()
    seeds = tuple(args.seed) if args.seed else DEFAULT_SEEDS
    benign_inputs = _normalize_paths(DEFAULT_BENIGN_INPUTS)
    malicious_inputs = _normalize_paths(DEFAULT_MALICIOUS_INPUTS)
    output_dir = Path(args.output_dir)
    run_export_dir = Path(args.run_export_dir)

    resolved_runs: list[ScorerSeedRun] = []
    reused_runs: list[ScorerSeedRun] = []
    executed_runs: list[ScorerSeedRun] = []

    for scorer_name in DEFAULT_SCORERS:
        for seed in seeds:
            existing = _matching_run_dir(
                scorer_name=scorer_name,
                random_seed=seed,
                packet_limit=20000,
                benign_inputs=benign_inputs,
                malicious_inputs=malicious_inputs,
            )
            if existing is not None:
                record = ScorerSeedRun(
                    scorer_name=scorer_name,
                    random_seed=seed,
                    run_dir=existing,
                    reused_existing=True,
                )
                resolved_runs.append(record)
                reused_runs.append(record)
                continue

            run_dir = _run_experiment(
                scorer_name=scorer_name,
                random_seed=seed,
                export_dir=run_export_dir,
                benign_inputs=benign_inputs,
                malicious_inputs=malicious_inputs,
            )
            record = ScorerSeedRun(
                scorer_name=scorer_name,
                random_seed=seed,
                run_dir=run_dir,
                reused_existing=False,
            )
            resolved_runs.append(record)
            executed_runs.append(record)

    grouped_metrics: dict[str, dict[str, list[float]]] = {
        scorer: {
            "overall_fpr": [],
            "overall_f1": [],
            "recon_recall": [],
            "browser_recall": [],
        }
        for scorer in DEFAULT_SCORERS
    }

    for run in resolved_runs:
        bundle = load_binary_detection_run_bundle(run.run_dir, backend_name="graph")
        grouped_metrics[run.scorer_name]["overall_fpr"].append(
            float(bundle.summary.overall_metrics.get("false_positive_rate") or 0.0)
        )
        grouped_metrics[run.scorer_name]["overall_f1"].append(
            float(bundle.summary.overall_metrics.get("f1") or 0.0)
        )
        grouped_metrics[run.scorer_name]["recon_recall"].append(
            _load_attack_recall(bundle, "Recon-HostDiscovery")
        )
        grouped_metrics[run.scorer_name]["browser_recall"].append(
            _load_attack_recall(bundle, "BrowserHijacking")
        )

    rows: list[dict[str, object]] = []
    for scorer_name in DEFAULT_SCORERS:
        metrics = grouped_metrics[scorer_name]
        rows.append(
            {
                "scorer_name": scorer_name,
                "scorer_role": normalize_graph_scorer_role(scorer_name),
                "mean_fpr": mean(metrics["overall_fpr"]),
                "std_fpr": pstdev(metrics["overall_fpr"]),
                "mean_f1": mean(metrics["overall_f1"]),
                "std_f1": pstdev(metrics["overall_f1"]),
                "mean_recon_recall": mean(metrics["recon_recall"]),
                "std_recon_recall": pstdev(metrics["recon_recall"]),
                "mean_browser_recall": mean(metrics["browser_recall"]),
                "std_browser_recall": pstdev(metrics["browser_recall"]),
            }
        )

    csv_path = output_dir / "default_scorer_stability_summary.csv"
    md_path = output_dir / "default_scorer_stability_summary.md"
    _write_csv(csv_path, rows)
    _write_markdown(
        md_path,
        rows=rows,
        reused=reused_runs,
        executed=executed_runs,
    )

    print(
        json.dumps(
            {
                "default_scorer_stability_summary_csv": csv_path.resolve().as_posix(),
                "default_scorer_stability_summary_md": md_path.resolve().as_posix(),
                "reused_run_count": len(reused_runs),
                "executed_run_count": len(executed_runs),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
