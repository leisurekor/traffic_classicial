#!/usr/bin/env python3
"""Render a focused baseline-vs-edge-centric comparison from CTU-13 benchmark rows."""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_CSV = REPO_ROOT / "results" / "ctu13_binary_benchmark.csv"
COMPARISON_CSV = REPO_ROOT / "results" / "ctu13_edge_centric_comparison.csv"
COMPARISON_MD = REPO_ROOT / "results" / "ctu13_edge_centric_comparison.md"


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _float_or_zero(value: str | None) -> float:
    if value in {None, "", "n/a"}:
        return 0.0
    return float(value)


def _select_baseline_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    candidates = [row for row in rows if row["model_name"] == "node_recon_baseline"]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda row: (
            -_float_or_zero(row.get("f1")),
            _float_or_zero(row.get("fpr")),
            -_float_or_zero(row.get("recall")),
            _float_or_zero(row.get("background_hit_ratio")),
        ),
    )[0]


def _select_edge_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    candidates = [row for row in rows if row["model_name"] == "edge_temporal_binary_v2"]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda row: (
            -_float_or_zero(row.get("f1")),
            -_float_or_zero(row.get("recall")),
            _float_or_zero(row.get("fpr")),
            _float_or_zero(row.get("background_hit_ratio")),
            0 if row.get("support_summary_mode") == "combined_support_summary" else 1,
        ),
    )[0]


def _select_nuisance_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    candidates = [row for row in rows if row["model_name"] == "edge_temporal_binary_v2_nuisance_aware"]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda row: (
            -_float_or_zero(row.get("f1")),
            -_float_or_zero(row.get("recall")),
            _float_or_zero(row.get("fpr")),
            _float_or_zero(row.get("background_hit_ratio")),
            _float_or_zero(row.get("malicious_blocked_by_nuisance_rate")),
            0 if row.get("support_summary_mode") == "local_support_density" else 1,
        ),
    )[0]


def main() -> None:
    if not BENCHMARK_CSV.exists():
        raise SystemExit("Missing CTU-13 benchmark CSV. Run scripts/run_ctu13_binary_benchmark.py first.")

    rows = _load_rows(BENCHMARK_CSV)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["evaluation_mode"], row["scenario_id"]), []).append(row)

    comparison_rows: list[dict[str, object]] = []
    for (evaluation_mode, scenario_id), grouped_rows in sorted(grouped.items()):
        baseline = _select_baseline_row(grouped_rows)
        edge = _select_edge_row(grouped_rows)
        nuisance = _select_nuisance_row(grouped_rows)
        if baseline is None or edge is None or nuisance is None:
            continue
        comparison_rows.append(
            {
                "evaluation_mode": evaluation_mode,
                "scenario_id": scenario_id,
                "baseline_profile": baseline["calibration_profile"],
                "edge_profile": edge["calibration_profile"],
                "nuisance_profile": nuisance["calibration_profile"],
                "edge_suppression_enabled": edge.get("suppression_enabled", "False"),
                "edge_support_summary_mode": edge.get("support_summary_mode", "old_concentration"),
                "nuisance_support_summary_mode": nuisance.get("support_summary_mode", "old_concentration"),
                "nuisance_boundary_mode": nuisance.get("nuisance_boundary_mode", "n/a"),
                "baseline_f1": _float_or_zero(baseline.get("f1")),
                "edge_v2_f1": _float_or_zero(edge.get("f1")),
                "nuisance_f1": _float_or_zero(nuisance.get("f1")),
                "delta_f1": _float_or_zero(edge.get("f1")) - _float_or_zero(baseline.get("f1")),
                "delta_nuisance_f1": _float_or_zero(nuisance.get("f1")) - _float_or_zero(baseline.get("f1")),
                "baseline_recall": _float_or_zero(baseline.get("recall")),
                "edge_v2_recall": _float_or_zero(edge.get("recall")),
                "nuisance_recall": _float_or_zero(nuisance.get("recall")),
                "delta_recall": _float_or_zero(edge.get("recall")) - _float_or_zero(baseline.get("recall")),
                "delta_nuisance_recall": _float_or_zero(nuisance.get("recall")) - _float_or_zero(baseline.get("recall")),
                "baseline_fpr": _float_or_zero(baseline.get("fpr")),
                "edge_v2_fpr": _float_or_zero(edge.get("fpr")),
                "nuisance_fpr": _float_or_zero(nuisance.get("fpr")),
                "delta_fpr": _float_or_zero(edge.get("fpr")) - _float_or_zero(baseline.get("fpr")),
                "delta_nuisance_fpr": _float_or_zero(nuisance.get("fpr")) - _float_or_zero(baseline.get("fpr")),
                "baseline_background_hit_ratio": _float_or_zero(baseline.get("background_hit_ratio")),
                "edge_v2_background_hit_ratio": _float_or_zero(edge.get("background_hit_ratio")),
                "nuisance_background_hit_ratio": _float_or_zero(nuisance.get("background_hit_ratio")),
                "nuisance_rejection_rate": _float_or_zero(nuisance.get("nuisance_rejection_rate")),
                "malicious_blocked_by_nuisance_rate": _float_or_zero(nuisance.get("malicious_blocked_by_nuisance_rate")),
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
        "| evaluation_mode | scenario_id | baseline_profile | edge_profile | nuisance_profile | baseline_f1 | edge_v2_f1 | nuisance_f1 | baseline_recall | edge_v2_recall | nuisance_recall | baseline_fpr | edge_v2_fpr | nuisance_fpr | edge_background_hit_ratio | nuisance_background_hit_ratio | nuisance_rejection_rate | malicious_blocked_by_nuisance_rate |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['evaluation_mode']} | {row['scenario_id']} | {row['baseline_profile']} | "
            f"{row['edge_profile']}:{row['edge_support_summary_mode']}:{row['edge_suppression_enabled']} | "
            f"{row['nuisance_profile']}:{row['nuisance_support_summary_mode']}:{row['nuisance_boundary_mode']} | "
            f"{row['baseline_f1']:.4f} | {row['edge_v2_f1']:.4f} | {row['nuisance_f1']:.4f} | "
            f"{row['baseline_recall']:.4f} | {row['edge_v2_recall']:.4f} | {row['nuisance_recall']:.4f} | "
            f"{row['baseline_fpr']:.4f} | {row['edge_v2_fpr']:.4f} | {row['nuisance_fpr']:.4f} | "
            f"{row['edge_v2_background_hit_ratio']:.4f} | {row['nuisance_background_hit_ratio']:.4f} | {row['nuisance_rejection_rate']:.4f} | {row['malicious_blocked_by_nuisance_rate']:.4f} |"
        )
    COMPARISON_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {COMPARISON_CSV}")
    print(f"Wrote {COMPARISON_MD}")


if __name__ == "__main__":
    main()
