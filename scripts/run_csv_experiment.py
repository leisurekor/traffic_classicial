#!/usr/bin/env python3
"""Run the repeatable CSV experiment wrapper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_graph.pipeline.repro_experiment import ReproExperimentConfig, run_csv_repro_experiment


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the repeatable CSV experiment pipeline.")
    parser.add_argument("--config", required=True, help="Path to YAML or JSON experiment config.")
    parser.add_argument("--input", default=None, help="Optional CSV path override.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory override.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = ReproExperimentConfig.from_path(args.config)
    if config.input_mode != "csv":
        raise SystemExit(f"Expected input_mode=csv but got {config.input_mode!r}")
    if args.input:
        config.input_path = args.input
    if args.output_dir:
        config.output_dir = args.output_dir
    result = run_csv_repro_experiment(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
