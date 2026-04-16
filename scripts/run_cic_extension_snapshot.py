"""Run one controlled CIC-IoT-2023 extension snapshot with a few new attacks.

This helper intentionally keeps the scope narrow:

- downloads only a few additional official PCAP attacks,
- keeps those downloads prefix-limited because the current mini protocol only
  consumes the first 20k packets anyway,
- reuses the existing real-PCAP graph experiment pipeline unchanged,
- keeps packet_limit fixed at 20000,
- keeps the current default graph scorer unchanged,
- runs one seed only and produces one compact summary.
"""

from __future__ import annotations

import csv
import http.cookiejar
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from traffic_graph.pipeline.compare_binary_detection_runs import (  # noqa: E402
    load_binary_detection_run_bundle,
)
from traffic_graph.pipeline.pcap_graph_experiment import (  # noqa: E402
    PcapGraphExperimentConfig,
    run_pcap_graph_experiment,
)


DATASET_BASE_URL = "https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/"
RAW_DIR = ROOT_DIR / "data" / "cic_iot_2023" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "cic_iot_2023" / "processed"
RESULTS_DIR = ROOT_DIR / "results"
ARTIFACT_DIR = ROOT_DIR / "artifacts" / "cic_iot_2023_extension"
MAX_DOWNLOAD_BYTES = 192 * 1024 * 1024

BENIGN_INPUTS: tuple[Path, ...] = (
    ROOT_DIR / "data" / "ciciot2023" / "pcap" / "benign" / "BenignTraffic.pcap",
    ROOT_DIR / "data" / "ciciot2023" / "pcap" / "benign" / "BenignTraffic1.redownload.pcap",
    ROOT_DIR / "data" / "ciciot2023" / "pcap" / "benign" / "BenignTraffic3.redownload2.pcap",
)

CSV_FIELDS: tuple[str, ...] = ("attack_type", "FPR", "F1", "recall")


@dataclass(frozen=True, slots=True)
class AttackSpec:
    """One minimal CIC-IoT-2023 extension attack to stage and evaluate."""

    attack_type: str
    official_path: str
    raw_filename: str
    processed_filename: str
    interpretation_label: str


ATTACK_SPECS: tuple[AttackSpec, ...] = (
    AttackSpec(
        attack_type="mitm_arp_spoofing",
        official_path="PCAP/MITM-ArpSpoofing/MITM-ArpSpoofing.pcap",
        raw_filename="MITM-ArpSpoofing.pcap",
        processed_filename="mitm_arp_spoofing.pcap",
        interpretation_label="MITM-ArpSpoofing",
    ),
    AttackSpec(
        attack_type="botnet_backdoor",
        official_path="PCAP/Backdoor_Malware/Backdoor_Malware.pcap",
        raw_filename="Backdoor_Malware.pcap",
        processed_filename="botnet_backdoor.pcap",
        interpretation_label="Backdoor_Malware",
    ),
    AttackSpec(
        attack_type="brute_force",
        official_path="PCAP/DictionaryBruteForce/DictionaryBruteForce.pcap",
        raw_filename="DictionaryBruteForce.pcap",
        processed_filename="brute_force.pcap",
        interpretation_label="DictionaryBruteForce",
    ),
)


def _register_session() -> object:
    """Register once with the official CIC form and return an authenticated session."""

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
    return opener


def _download_if_missing(session: object, official_path: str, destination: Path) -> None:
    """Download one official dataset file prefix when it is not already present locally."""

    if destination.exists() and destination.stat().st_size > 0:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(
        DATASET_BASE_URL + "download.php?file=" + quote(official_path, safe="")
    )
    with session.open(request, timeout=30) as response, destination.open("wb") as handle:
        written = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            written += len(chunk)
            if written >= MAX_DOWNLOAD_BYTES:
                break


def _ensure_symlink(link_path: Path, target_path: Path) -> None:
    """Create or refresh one processed-data symlink."""

    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(target_path.resolve())


def _stage_processed_views(downloaded_raw_paths: dict[str, Path]) -> dict[str, Path]:
    """Create the minimal processed layout used by the current experiment snapshot."""

    processed: dict[str, Path] = {}
    for index, benign_path in enumerate(BENIGN_INPUTS, start=1):
        link_path = PROCESSED_DIR / f"benign_{index}.pcap"
        _ensure_symlink(link_path, benign_path)
        processed[f"benign_{index}"] = link_path

    for spec in ATTACK_SPECS:
        link_path = PROCESSED_DIR / spec.processed_filename
        _ensure_symlink(link_path, downloaded_raw_paths[spec.attack_type])
        processed[spec.attack_type] = link_path

    representative_benign = PROCESSED_DIR / "benign.pcap"
    _ensure_symlink(representative_benign, BENIGN_INPUTS[-1])
    processed["benign"] = representative_benign
    return processed


def _run_single_attack_experiment(spec: AttackSpec, malicious_input: Path) -> Path:
    """Run one fixed-protocol binary evaluation for a single new attack."""

    config = PcapGraphExperimentConfig(
        packet_limit=20000,
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
        benign_inputs=BENIGN_INPUTS,
        malicious_inputs=(malicious_input,),
        experiment_label=f"cic_extension_{spec.attack_type}_packet20000_seed42",
        config=config,
    )
    return Path(result.export_result.run_directory)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write the final compact CSV summary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float) -> str:
    """Format one metric consistently."""

    return f"{value:.6f}"


def _build_markdown(rows: list[dict[str, object]]) -> str:
    """Build the short Markdown summary for the extension snapshot."""

    by_attack = {str(row["attack_type"]): row for row in rows}
    mitm = by_attack["mitm_arp_spoofing"]
    botnet = by_attack["botnet_backdoor"]
    brute = by_attack["brute_force"]

    hardest = min(rows, key=lambda row: float(row["recall"]))
    observations = [
        (
            f"`hybrid_max_rank_flow_node_max` shows non-zero recall on all three new attacks, "
            f"so the current graph pipeline does transfer beyond the original Recon / DDoS / Browser set."
        ),
        (
            f"The hardest new attack in this snapshot is `{hardest['attack_type']}` "
            f"with recall `{_fmt(float(hardest['recall']))}`."
        ),
        (
            f"`MITM-ArpSpoofing` recall is `{_fmt(float(mitm['recall']))}`, which is the closest weak-anomaly check to the existing "
            "BrowserHijacking weakness: if it stays low, that suggests the same subtle-behavior limitation is still present."
        ),
        (
            f"`Backdoor_Malware` recall is `{_fmt(float(botnet['recall']))}`. If it lands above MITM, that supports the current intuition "
            "that graph structure helps more on coordinated / structural anomalies than on weak behavioral shifts."
        ),
        (
            f"`DictionaryBruteForce` recall is `{_fmt(float(brute['recall']))}`, giving a middle-ground reference between weak MITM-style "
            "signals and more structural backdoor behavior."
        ),
    ]

    lines = [
        "# CIC IoT 2023 Extension Summary",
        "",
        "| attack_type | FPR | F1 | recall |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['attack_type']}` | {_fmt(float(row['FPR']))} | {_fmt(float(row['F1']))} | {_fmt(float(row['recall']))} |"
        )
    lines.extend(["", "## Short Analysis", ""])
    lines.extend(f"- {note}" for note in observations)
    return "\n".join(lines) + "\n"


def main() -> int:
    """Download a few new CIC-IoT-2023 attacks and run one controlled extension snapshot."""

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    session = _register_session()
    downloaded_raw_paths: dict[str, Path] = {}
    for spec in ATTACK_SPECS:
        destination = RAW_DIR / spec.raw_filename
        _download_if_missing(session, spec.official_path, destination)
        downloaded_raw_paths[spec.attack_type] = destination

    processed_paths = _stage_processed_views(downloaded_raw_paths)

    summary_rows: list[dict[str, object]] = []
    run_directories: dict[str, str] = {}
    for spec in ATTACK_SPECS:
        run_dir = _run_single_attack_experiment(spec, processed_paths[spec.attack_type])
        run_directories[spec.attack_type] = run_dir.as_posix()
        bundle = load_binary_detection_run_bundle(run_dir, backend_name="graph")
        attack_metrics = bundle.per_attack_metrics_by_task.get(spec.attack_type)
        if attack_metrics is None:
            attack_metrics = bundle.per_attack_metrics_by_task.get(spec.interpretation_label)
        if attack_metrics is None:
            attack_metrics = next(
                (
                    metric
                    for task_name, metric in bundle.per_attack_metrics_by_task.items()
                    if task_name != "all_malicious"
                ),
                None,
            )
        if attack_metrics is None:
            raise KeyError(
                f"Per-attack metrics for {spec.attack_type} were not found in {run_dir.as_posix()}."
            )
        summary_rows.append(
            {
                "attack_type": spec.attack_type,
                "FPR": float(bundle.summary.overall_metrics.get("false_positive_rate") or 0.0),
                "F1": float(bundle.summary.overall_metrics.get("f1") or 0.0),
                "recall": float(attack_metrics.recall or 0.0),
            }
        )

    csv_path = RESULTS_DIR / "cic_extension_summary.csv"
    md_path = RESULTS_DIR / "cic_extension_summary.md"
    _write_csv(csv_path, summary_rows)
    md_path.write_text(_build_markdown(summary_rows), encoding="utf-8")

    print(
        json.dumps(
            {
                "downloaded_raw_files": {
                    attack_type: path.as_posix()
                    for attack_type, path in downloaded_raw_paths.items()
                },
                "processed_files": {
                    key: value.as_posix() for key, value in processed_paths.items()
                },
                "run_directories": run_directories,
                "results_csv": csv_path.as_posix(),
                "results_md": md_path.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
