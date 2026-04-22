#!/usr/bin/env python3
"""Generate tiny local sample inputs and run a quick smoke workflow."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "quickstart"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "quickstart"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create tiny quickstart artifacts and run a repository smoke check."
    )
    parser.add_argument(
        "--skip-csv-run",
        action="store_true",
        help="Only generate the tiny sample files without running the CSV wrapper.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip the lightweight wrapper smoke test suite.",
    )
    return parser


def _write_quickstart_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index in range(12):
        rows.append(
            {
                "FlowID": f"benign-{index}",
                "FeatureA": float(index),
                "FeatureB": float(index + 1),
                "FeatureC": float(index % 3),
                "Label": "BENIGN",
            }
        )
    for index in range(6):
        rows.append(
            {
                "FlowID": f"malicious-{index}",
                "FeatureA": float(index + 100),
                "FeatureB": float(index + 120),
                "FeatureC": float((index % 3) + 5),
                "Label": "MALICIOUS",
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_quickstart_config(path: Path, csv_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config_text = "\n".join(
        [
            "dataset_name: quickstart_csv_smoke",
            "input_mode: csv",
            "use_nuisance_aware: false",
            f"input_path: {csv_path.relative_to(PROJECT_ROOT).as_posix()}",
            "label_column: Label",
            "binary_label_mapping:",
            "  BENIGN: 0",
            "train_ratio: 0.6",
            "val_ratio: 0.2",
            "test_ratio: 0.2",
            "random_seed: 42",
            "threshold_percentile: 80.0",
            "max_components: 3",
            f"output_dir: {OUTPUT_ROOT.relative_to(PROJECT_ROOT).as_posix()}",
            "",
        ]
    )
    path.write_text(config_text, encoding="utf-8")


def _run_command(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    csv_path = ARTIFACT_ROOT / "quickstart_flows.csv"
    config_path = ARTIFACT_ROOT / "quickstart_csv.yaml"
    _write_quickstart_csv(csv_path)
    _write_quickstart_config(config_path, csv_path)

    print("Quickstart artifacts ready:")
    print(f"- sample CSV: {csv_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"- sample config: {config_path.relative_to(PROJECT_ROOT).as_posix()}")
    print("")
    print("This quickstart path guarantees a tiny CSV run.")
    print("For real PCAP experiments, prepare external data as documented in README.")

    if not args.skip_csv_run:
        print("")
        print("Running CSV quickstart smoke...")
        _run_command(
            [
                sys.executable,
                "scripts/run_csv_experiment.py",
                "--config",
                config_path.relative_to(PROJECT_ROOT).as_posix(),
            ]
        )
    if not args.skip_tests:
        print("")
        print("Running wrapper smoke tests...")
        _run_command([sys.executable, "-m", "unittest", "tests.test_repro_experiment", "-v"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
