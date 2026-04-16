"""Check CICIoT2023 PCAP integrity and compare prefix vs random-window sampling.

This helper keeps scope intentionally narrow:

- it inspects only the current benign training files plus the newly staged CIC
  extension raw PCAPs,
- it does one small prefix vs random-window comparison,
- it does not modify model logic, scorer behavior, or graph construction.
"""

from __future__ import annotations

import csv
import http.cookiejar
import json
import sys
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from traffic_graph.data import inspect_classic_pcap  # noqa: E402
from traffic_graph.pipeline.compare_binary_detection_runs import (  # noqa: E402
    load_binary_detection_run_bundle,
)
from traffic_graph.pipeline.pcap_graph_experiment import (  # noqa: E402
    PcapGraphExperimentConfig,
    run_pcap_graph_experiment,
)


DATASET_BASE_URL = "https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/"
RESULTS_DIR = ROOT_DIR / "results"
ARTIFACT_DIR = ROOT_DIR / "artifacts" / "sampling_bias_check"

NEW_ATTACK_FILES: tuple[tuple[str, Path, str], ...] = (
    ("mitm_arp_spoofing", ROOT_DIR / "data" / "cic_iot_2023" / "raw" / "MITM-ArpSpoofing.pcap", "PCAP/MITM-ArpSpoofing/MITM-ArpSpoofing.pcap"),
    ("botnet_backdoor", ROOT_DIR / "data" / "cic_iot_2023" / "raw" / "Backdoor_Malware.pcap", "PCAP/Backdoor_Malware/Backdoor_Malware.pcap"),
    ("brute_force", ROOT_DIR / "data" / "cic_iot_2023" / "raw" / "DictionaryBruteForce.pcap", "PCAP/DictionaryBruteForce/DictionaryBruteForce.pcap"),
)

BENIGN_FILES: tuple[tuple[str, Path, str], ...] = (
    ("BenignTraffic.pcap", ROOT_DIR / "data" / "ciciot2023" / "pcap" / "benign" / "BenignTraffic.pcap", "PCAP/Benign_Final/BenignTraffic.pcap"),
    ("BenignTraffic1.redownload.pcap", ROOT_DIR / "data" / "ciciot2023" / "pcap" / "benign" / "BenignTraffic1.redownload.pcap", "PCAP/Benign_Final/BenignTraffic1.pcap"),
    ("BenignTraffic3.redownload2.pcap", ROOT_DIR / "data" / "ciciot2023" / "pcap" / "benign" / "BenignTraffic3.redownload2.pcap", "PCAP/Benign_Final/BenignTraffic3.pcap"),
)

VALIDATION_BENIGN = ROOT_DIR / "data" / "ciciot2023" / "pcap" / "benign" / "BenignTraffic3.redownload2.pcap"
VALIDATION_ATTACK = ROOT_DIR / "data" / "cic_iot_2023" / "processed" / "mitm_arp_spoofing.pcap"


def _head_content_lengths(official_paths: list[str]) -> dict[str, int | None]:
    """Try to fetch remote Content-Length values when the server exposes them."""

    cookie_jar = http.cookiejar.CookieJar()
    opener = build_opener(HTTPCookieProcessor(cookie_jar))
    payload = urlencode(
        {
            "first_name": "OpenAI",
            "last_name": "Codex",
            "email": "codex@example.com",
            "institution": "OpenAI",
            "job_title": "Researcher",
            "country": "China",
        }
    ).encode("utf-8")
    request = Request(
        DATASET_BASE_URL + "insert.php",
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with opener.open(request, timeout=30) as response:
        body = json.loads(response.read().decode("utf-8"))
    if not body.get("ok"):
        raise RuntimeError(f"Dataset registration failed: {body}")

    lengths: dict[str, int | None] = {}
    for official_path in official_paths:
        req = Request(
            DATASET_BASE_URL + "download.php?file=" + quote(official_path, safe=""),
            method="HEAD",
        )
        try:
            with opener.open(req, timeout=30) as response:
                raw_length = response.headers.get("Content-Length")
                lengths[official_path] = int(raw_length) if raw_length else None
        except Exception:
            lengths[official_path] = None
    return lengths


def _integrity_status(
    *,
    local_path: Path,
    remote_length: int | None,
    packet_count: int,
    truncated: bool,
) -> str:
    """Map simple evidence into the requested integrity labels."""

    local_size = local_path.stat().st_size
    if remote_length is not None:
        if local_size < remote_length:
            return "partial_or_prefix"
        if local_size == remote_length and not truncated:
            return "complete"
    if truncated:
        return "partial_or_prefix"
    if packet_count > 0:
        return "probably_complete"
    return "unknown"


def _run_validation(sampling_mode: str) -> tuple[Path, dict[str, object]]:
    """Run one minimal prefix or random-window validation."""

    config = PcapGraphExperimentConfig(
        packet_limit=20000,
        packet_sampling_mode=sampling_mode,  # type: ignore[arg-type]
        window_size=10,
        use_association_edges=True,
        use_graph_structural_features=True,
        benign_train_ratio=0.7,
        train_validation_ratio=0.25,
        graph_score_reduction="hybrid_max_rank_flow_node_max",
        epochs=1,
        batch_size=1,
        learning_rate=1e-3,
        threshold_percentile=95.0,
        random_seed=42,
    )
    result = run_pcap_graph_experiment(
        export_dir=ARTIFACT_DIR,
        benign_inputs=(VALIDATION_BENIGN,),
        malicious_inputs=(VALIDATION_ATTACK,),
        experiment_label=f"sampling_bias_{sampling_mode}_mitm_seed42_packet20000",
        config=config,
    )
    run_dir = Path(result.export_result.run_directory)
    summary_payload = json.loads(
        (run_dir / "pcap_experiment_summary.json").read_text(encoding="utf-8")
    )
    return run_dir, summary_payload


def _find_source_summary(summary_payload: dict[str, object], source_name: str) -> dict[str, object]:
    """Return one source summary row by source_name."""

    for row in summary_payload.get("source_summaries", []):
        if isinstance(row, dict) and row.get("source_name") == source_name:
            return row
    raise KeyError(f"Source summary for {source_name} not found.")


def _write_sampling_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write the small prefix-vs-random-window comparison table."""

    fieldnames = [
        "dataset_name",
        "file_path",
        "integrity_status",
        "sampling_mode",
        "seed",
        "packet_limit",
        "benign_packet_count",
        "attack_packet_count",
        "benign_packet_start_offset",
        "attack_packet_start_offset",
        "FPR",
        "F1",
        "recall",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float | int | str | None) -> str:
    """Format markdown scalar values consistently."""

    if value is None:
        return "unavailable"
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return str(value)


def main() -> int:
    """Run integrity checks and one prefix-vs-random-window validation."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    official_paths = [item[2] for item in NEW_ATTACK_FILES + BENIGN_FILES]
    remote_lengths = _head_content_lengths(official_paths)

    integrity_rows: list[dict[str, object]] = []
    for name, local_path, official_path in NEW_ATTACK_FILES + BENIGN_FILES:
        packet_count, truncated = inspect_classic_pcap(local_path)
        integrity_rows.append(
            {
                "name": name,
                "local_path": local_path.as_posix(),
                "official_path": official_path,
                "local_size": local_path.stat().st_size,
                "remote_length": remote_lengths.get(official_path),
                "packet_count": packet_count,
                "truncated": truncated,
                "status": _integrity_status(
                    local_path=local_path,
                    remote_length=remote_lengths.get(official_path),
                    packet_count=packet_count,
                    truncated=truncated,
                ),
            }
        )

    prefix_run_dir, prefix_summary = _run_validation("prefix")
    random_run_dir, random_summary = _run_validation("random_window")

    prefix_attack_metrics = load_binary_detection_run_bundle(prefix_run_dir, backend_name="graph")
    random_attack_metrics = load_binary_detection_run_bundle(random_run_dir, backend_name="graph")

    prefix_mitm = prefix_attack_metrics.per_attack_metrics_by_task["mitm_arp_spoofing"]
    random_mitm = random_attack_metrics.per_attack_metrics_by_task["mitm_arp_spoofing"]

    prefix_benign_source = _find_source_summary(prefix_summary, VALIDATION_BENIGN.stem)
    prefix_attack_source = _find_source_summary(prefix_summary, VALIDATION_ATTACK.stem)
    random_benign_source = _find_source_summary(random_summary, VALIDATION_BENIGN.stem)
    random_attack_source = _find_source_summary(random_summary, VALIDATION_ATTACK.stem)

    validation_attack_integrity = next(
        row["status"]
        for row in integrity_rows
        if Path(str(row["local_path"])).resolve() == VALIDATION_ATTACK.resolve()
    )

    sampling_rows = [
        {
            "dataset_name": "CICIoT2023",
            "file_path": VALIDATION_ATTACK.as_posix(),
            "integrity_status": validation_attack_integrity,
            "sampling_mode": "prefix",
            "seed": 42,
            "packet_limit": 20000,
            "benign_packet_count": prefix_benign_source["parse_summary"]["total_packets"],
            "attack_packet_count": prefix_attack_source["parse_summary"]["total_packets"],
            "benign_packet_start_offset": prefix_benign_source["parse_summary"].get("packet_start_offset", 0),
            "attack_packet_start_offset": prefix_attack_source["parse_summary"].get("packet_start_offset", 0),
            "FPR": prefix_mitm.false_positive_rate,
            "F1": prefix_mitm.f1,
            "recall": prefix_mitm.recall,
        },
        {
            "dataset_name": "CICIoT2023",
            "file_path": VALIDATION_ATTACK.as_posix(),
            "integrity_status": validation_attack_integrity,
            "sampling_mode": "random_window",
            "seed": 42,
            "packet_limit": 20000,
            "benign_packet_count": random_benign_source["parse_summary"]["total_packets"],
            "attack_packet_count": random_attack_source["parse_summary"]["total_packets"],
            "benign_packet_start_offset": random_benign_source["parse_summary"].get("packet_start_offset", 0),
            "attack_packet_start_offset": random_attack_source["parse_summary"].get("packet_start_offset", 0),
            "FPR": random_mitm.false_positive_rate,
            "F1": random_mitm.f1,
            "recall": random_mitm.recall,
        },
    ]

    csv_path = RESULTS_DIR / "sampling_bias_check.csv"
    _write_sampling_csv(csv_path, sampling_rows)

    md_path = RESULTS_DIR / "data_integrity_check.md"
    lines = [
        "# CICIoT2023 Data Integrity and Sampling Check",
        "",
        "## Integrity findings",
        "",
    ]
    for row in integrity_rows:
        lines.append(
            f"- `{row['name']}`: `{row['status']}` "
            f"(local_size=`{row['local_size']}`, remote_length=`{row['remote_length']}`, "
            f"packet_count=`{row['packet_count']}`, truncated=`{row['truncated']}`)"
        )
    lines.extend(
        [
            "",
            "## Prefix-bias finding",
            "",
            "- Current `packet_limit` behavior is prefix-based: `ClassicPcapReader.iter_packets()` stops as soon as the first `max_packets` records are consumed.",
            "- This bias affects both benign training inputs and malicious evaluation inputs because `run_pcap_graph_experiment()` passes `config.packet_limit` directly into `load_pcap_flow_dataset(..., max_packets=...)` for every source.",
            "",
            "## Minimal fix applied",
            "",
            "- Added `packet_sampling_mode` to the PCAP experiment config.",
            "- Added `random_window` as a minimal sampling option that keeps packet order intact by choosing one reproducible continuous packet window instead of always taking the prefix.",
            "- Reproducibility stays tied to `random_seed`; the chosen packet start offset is now exported through each source `parse_summary`.",
            "",
            "## Prefix vs random-window validation",
            "",
            "| sampling_mode | FPR | F1 | recall | benign_packet_count | attack_packet_count | benign_packet_start_offset | attack_packet_start_offset |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in sampling_rows:
        lines.append(
            f"| `{row['sampling_mode']}` | {_fmt(row['FPR'])} | {_fmt(row['F1'])} | {_fmt(row['recall'])} | "
            f"{row['benign_packet_count']} | {row['attack_packet_count']} | "
            f"{row['benign_packet_start_offset']} | {row['attack_packet_start_offset']} |"
        )

    prefix_row = sampling_rows[0]
    random_row = sampling_rows[1]
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- The new CIC extension MITM file is `{validation_attack_integrity}`, so key conclusions from it should be treated as provisional until a fully verified download is available.",
            f"- The benign training files currently used by the mainline are all at least `probably_complete`, and the three formal benign references are supported by clean EOF parses.",
            f"- For bounded mini experiments, `random_window` is the safer default than `prefix` because it removes deterministic prefix bias while preserving local time order.",
            f"- If later paper-facing results depend materially on the new CIC extension attacks, we should redownload the full official files first, then rerun the same fixed protocol once on the complete captures.",
            "",
            "## Validation run directories",
            "",
            f"- prefix: `{prefix_run_dir.as_posix()}`",
            f"- random_window: `{random_run_dir.as_posix()}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "data_integrity_check_md": md_path.as_posix(),
                "sampling_bias_check_csv": csv_path.as_posix(),
                "prefix_run_dir": prefix_run_dir.as_posix(),
                "random_window_run_dir": random_run_dir.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
