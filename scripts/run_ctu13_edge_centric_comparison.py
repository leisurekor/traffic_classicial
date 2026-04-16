#!/usr/bin/env python3
"""Render a focused baseline-vs-edge-centric comparison from CTU-13 benchmark rows."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_CSV = REPO_ROOT / "results" / "ctu13_binary_benchmark.csv"
COMPARISON_CSV = REPO_ROOT / "results" / "ctu13_edge_centric_comparison.csv"
COMPARISON_MD = REPO_ROOT / "results" / "ctu13_edge_centric_comparison.md"


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _float_or_zero(value: str | None) -> float:
    if value in {None, "", "n/a"}:
        return 0.0
    return float(value)


def main() -> None:
    if not BENCHMARK_CSV.exists():
        raise SystemExit("Missing CTU-13 benchmark CSV. Run scripts/run_ctu13_binary_benchmark.py first.")

    rows = _load_rows(BENCHMARK_CSV)
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["evaluation_mode"], row["scenario_id"]), {})[row["variant"]] = row

    comparison_rows: list[dict[str, object]] = []
    for (evaluation_mode, scenario_id), variants in sorted(grouped.items()):
        baseline = variants.get("node_recon_baseline")
        edge = variants.get("edge_temporal_binary_v2")
        if baseline is None or edge is None:
            continue
        comparison_rows.append(
            {
                "evaluation_mode": evaluation_mode,
                "scenario_id": scenario_id,
                "baseline_f1": _float_or_zero(baseline.get("f1")),
                "edge_v2_f1": _float_or_zero(edge.get("f1")),
                "delta_f1": _float_or_zero(edge.get("f1")) - _float_or_zero(baseline.get("f1")),
                "baseline_recall": _float_or_zero(baseline.get("recall")),
                "edge_v2_recall": _float_or_zero(edge.get("recall")),
                "delta_recall": _float_or_zero(edge.get("recall")) - _float_or_zero(baseline.get("recall")),
                "baseline_fpr": _float_or_zero(baseline.get("fpr")),
                "edge_v2_fpr": _float_or_zero(edge.get("fpr")),
                "delta_fpr": _float_or_zero(edge.get("fpr")) - _float_or_zero(baseline.get("fpr")),
                "baseline_background_hit_ratio": _float_or_zero(baseline.get("background_hit_ratio")),
                "edge_v2_background_hit_ratio": _float_or_zero(edge.get("background_hit_ratio")),
            }
        )

    COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)
    with COMPARISON_CSV.open("w", encoding="utf-8", newline="") as handle:
        if not comparison_rows:
            handle.write("")
        else:
            writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0].keys()))
            writer.writeheader()
            writer.writerows(comparison_rows)

    lines = [
        "# CTU-13 Edge-Centric Comparison",
        "",
        "| evaluation_mode | scenario_id | baseline_f1 | edge_v2_f1 | delta_f1 | baseline_recall | edge_v2_recall | delta_recall | baseline_fpr | edge_v2_fpr | delta_fpr |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['evaluation_mode']} | {row['scenario_id']} | {row['baseline_f1']:.4f} | "
            f"{row['edge_v2_f1']:.4f} | {row['delta_f1']:.4f} | {row['baseline_recall']:.4f} | "
            f"{row['edge_v2_recall']:.4f} | {row['delta_recall']:.4f} | {row['baseline_fpr']:.4f} | "
            f"{row['edge_v2_fpr']:.4f} | {row['delta_fpr']:.4f} |"
        )
    COMPARISON_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {COMPARISON_CSV}")
    print(f"Wrote {COMPARISON_MD}")


if __name__ == "__main__":
    main()
