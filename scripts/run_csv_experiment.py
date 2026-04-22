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


def _resolve_project_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _validate_config_path(config_path: str) -> None:
    resolved_config = _resolve_project_path(config_path)
    if resolved_config is None or not resolved_config.exists():
        raise SystemExit(
            "Config file was not found.\n"
            f"- requested: {config_path}\n"
            f"- expected path: {(resolved_config or Path(config_path)).as_posix()}\n"
            "Hint: pass --config configs/repro_csv.example.yaml or your own YAML/JSON file."
        )


def _validate_csv_input_path(input_path: str | None) -> None:
    resolved_input = _resolve_project_path(input_path)
    if resolved_input is None or not resolved_input.exists():
        requested = input_path or "(empty)"
        expected = (resolved_input or Path(requested)).as_posix()
        raise SystemExit(
            "CSV input file was not found.\n"
            f"- requested: {requested}\n"
            f"- expected path: {expected}\n"
            "Hints:\n"
            "- Download CICIoT2023 from https://www.unb.ca/cic/datasets/iotdataset-2023.html\n"
            "- Put Merged01.csv under artifacts/cic_iot2023/\n"
            "- Or override the path with --input /path/to/your.csv"
        )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_config_path(args.config)
    config = ReproExperimentConfig.from_path(args.config)
    if config.input_mode != "csv":
        raise SystemExit(f"Expected input_mode=csv but got {config.input_mode!r}")
    if args.input:
        config.input_path = args.input
    if args.output_dir:
        config.output_dir = args.output_dir
    _validate_csv_input_path(config.input_path)
    result = run_csv_repro_experiment(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
