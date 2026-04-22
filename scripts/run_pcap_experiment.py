#!/usr/bin/env python3
"""Run the repeatable PCAP experiment wrapper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_graph.pipeline.repro_experiment import ReproExperimentConfig, run_pcap_repro_experiment


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the repeatable PCAP experiment pipeline.")
    parser.add_argument("--config", required=True, help="Path to YAML or JSON experiment config.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory override.")
    return parser


def _resolve_project_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _validate_inputs(config_path: str, benign_inputs: list[str], malicious_inputs: list[str]) -> None:
    resolved_config = _resolve_project_path(config_path)
    if resolved_config is None or not resolved_config.exists():
        raise SystemExit(
            "Config file was not found.\n"
            f"- requested: {config_path}\n"
            f"- expected path: {(resolved_config or Path(config_path)).as_posix()}\n"
            "Hint: pass --config configs/repro_pcap.example.yaml or your own YAML/JSON file."
        )
    if not benign_inputs and not malicious_inputs:
        raise SystemExit(
            "PCAP experiment inputs are empty.\n"
            "- configs/repro_pcap.example.yaml currently requires benign_inputs and/or malicious_inputs\n"
            "- Fill those lists before running\n"
            "Hints:\n"
            "- Download CTU-13 with: python3 scripts/download_ctu13.py --scenarios 48 49 52\n"
            "- Or point to your own PCAP files in the config"
        )
    missing_paths: list[str] = []
    for value in benign_inputs + malicious_inputs:
        resolved = _resolve_project_path(value)
        if resolved is None or not resolved.exists():
            missing_paths.append((resolved or Path(value)).as_posix())
    if missing_paths:
        rendered_missing = "\n".join(f"- {item}" for item in missing_paths)
        raise SystemExit(
            "One or more PCAP inputs were not found.\n"
            f"{rendered_missing}\n"
            "Hints:\n"
            "- Edit configs/repro_pcap.example.yaml and fill benign_inputs / malicious_inputs\n"
            "- For CTU-13, use python3 scripts/download_ctu13.py --scenarios 48 49 52\n"
            "- For CICIoT2023, place your PCAP files under artifacts/ or data/ and reference them explicitly"
        )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = ReproExperimentConfig.from_path(args.config)
    if config.input_mode != "pcap":
        raise SystemExit(f"Expected input_mode=pcap but got {config.input_mode!r}")
    if args.output_dir:
        config.output_dir = args.output_dir
    _validate_inputs(args.config, config.benign_inputs, config.malicious_inputs)
    result = run_pcap_repro_experiment(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
