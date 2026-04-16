#!/usr/bin/env python3
"""Download CTU-13 official assets and build a reusable local manifest."""

from __future__ import annotations

import argparse
import bz2
from dataclasses import replace
from pathlib import Path
from typing import Iterable
import sys

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_graph.datasets import (  # noqa: E402
    build_ctu13_manifest_entry,
    discover_ctu13_scenario_urls,
    load_ctu13_manifest,
    save_ctu13_manifest,
)
from traffic_graph.datasets.ctu13 import CTU13ScenarioManifestEntry  # noqa: E402


RAW_ROOT = REPO_ROOT / "data" / "ctu13" / "raw"
MANIFEST_PATH = REPO_ROOT / "data" / "ctu13" / "ctu13_manifest.json"


def _remote_size_bytes(url: str) -> int | None:
    try:
        response = requests.head(url, allow_redirects=True, timeout=30)
        if not response.ok:
            return None
        size = response.headers.get("Content-Length")
        return int(size) if size is not None else None
    except Exception:
        return None


def _download_with_resume(url: str, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing_size = destination.stat().st_size if destination.exists() else 0
    remote_size = _remote_size_bytes(url)
    if remote_size is not None and existing_size == remote_size and existing_size > 0:
        return "skipped_existing"

    headers = {}
    mode = "wb"
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        mode = "ab"

    with requests.get(url, stream=True, headers=headers, timeout=60) as response:
        if response.status_code == 416 and destination.exists():
            return "skipped_existing"
        response.raise_for_status()
        if response.status_code == 200 and "Range" in headers:
            mode = "wb"
        with destination.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
    return "downloaded"


def _decompress_bz2(source_path: Path, destination_path: Path) -> None:
    if destination_path.exists() and destination_path.stat().st_size > 0:
        return
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with bz2.open(source_path, "rb") as source, destination_path.open("wb") as target:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            if not chunk:
                break
            target.write(chunk)


def _pick_default_scenarios(entries: Iterable[CTU13ScenarioManifestEntry]) -> list[CTU13ScenarioManifestEntry]:
    sized_entries: list[tuple[int, CTU13ScenarioManifestEntry]] = []
    fallback_entries: list[CTU13ScenarioManifestEntry] = []
    for entry in entries:
        if entry.pcap_source_url is None:
            continue
        size = _remote_size_bytes(entry.pcap_source_url)
        if size is None:
            fallback_entries.append(entry)
            continue
        sized_entries.append((size, entry))
    sized_entries.sort(key=lambda item: (item[0], item[1].scenario_id))
    selected = [entry for _size, entry in sized_entries[:4]]
    if len(selected) < 4:
        for entry in fallback_entries:
            if entry not in selected:
                selected.append(entry)
            if len(selected) >= 4:
                break
    return selected


def _materialize_entry(entry: CTU13ScenarioManifestEntry) -> CTU13ScenarioManifestEntry:
    scenario_root = RAW_ROOT / f"scenario_{entry.scenario_id}"
    pcap_compressed_path = None
    pcap_path = None
    label_path = None
    readme_path = None
    notes: list[str] = list(entry.notes)

    download_status = "downloaded"
    try:
        print(f"Downloading scenario {entry.scenario_id} from {entry.scenario_url}", flush=True)
        if entry.pcap_source_url:
            compressed_name = Path(entry.pcap_source_url).name
            pcap_compressed = scenario_root / compressed_name
            pcap_compressed_path = pcap_compressed.as_posix()
            print(f"  pcap: {entry.pcap_source_url}", flush=True)
            _download_with_resume(entry.pcap_source_url, pcap_compressed)
            if compressed_name.endswith(".bz2"):
                pcap_target = scenario_root / compressed_name[:-4]
                print(f"  decompress: {pcap_compressed} -> {pcap_target}", flush=True)
                _decompress_bz2(pcap_compressed, pcap_target)
                pcap_path = pcap_target.as_posix()
            else:
                pcap_path = pcap_compressed.as_posix()
        else:
            download_status = "failed"
            notes.append("Missing pcap source URL.")

        if entry.label_source_url:
            label_target = scenario_root / Path(entry.label_source_url).name
            print(f"  labels: {entry.label_source_url}", flush=True)
            _download_with_resume(entry.label_source_url, label_target)
            label_path = label_target.as_posix()
        else:
            download_status = "failed"
            notes.append("Missing label source URL.")

        if entry.readme_source_url:
            readme_target = scenario_root / Path(entry.readme_source_url).name
            print(f"  readme: {entry.readme_source_url}", flush=True)
            _download_with_resume(entry.readme_source_url, readme_target)
            readme_path = readme_target.as_posix()
    except Exception as exc:
        download_status = "failed"
        notes.append(str(exc))

    return replace(
        entry,
        pcap_path=pcap_path,
        pcap_compressed_path=pcap_compressed_path,
        label_file_path=label_path,
        readme_path=readme_path,
        download_status=download_status,
        notes=tuple(notes),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Download all discoverable CTU-13 scenarios.")
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="Scenario ids to download, for example: --scenarios 42 43 52 54",
    )
    args = parser.parse_args()

    scenario_urls = discover_ctu13_scenario_urls()
    print(f"Discovered {len(scenario_urls)} official CTU-13 scenarios.", flush=True)
    discovered_entries = [
        build_ctu13_manifest_entry(scenario_id, scenario_url)
        for scenario_id, scenario_url in scenario_urls.items()
    ]
    if args.all:
        selected_entries = discovered_entries
    elif args.scenarios:
        wanted = {scenario_id.strip() for scenario_id in args.scenarios if scenario_id.strip()}
        selected_entries = [entry for entry in discovered_entries if entry.scenario_id in wanted]
    else:
        selected_entries = _pick_default_scenarios(discovered_entries)
    print(
        "Selected scenarios: "
        + ", ".join(entry.scenario_id for entry in selected_entries),
        flush=True,
    )

    materialized_entries = [_materialize_entry(entry) for entry in selected_entries]

    existing_entries = {
        entry.scenario_id: entry
        for entry in load_ctu13_manifest(MANIFEST_PATH)
        if MANIFEST_PATH.exists()
    }
    for entry in materialized_entries:
        existing_entries[entry.scenario_id] = entry
    merged_entries = [existing_entries[key] for key in sorted(existing_entries)]
    save_ctu13_manifest(merged_entries, MANIFEST_PATH)

    print(f"Saved manifest to {MANIFEST_PATH}")
    for entry in materialized_entries:
        print(
            f"[{entry.download_status}] scenario {entry.scenario_id}: "
            f"pcap={entry.pcap_path} label={entry.label_file_path}"
        )


if __name__ == "__main__":
    main()
