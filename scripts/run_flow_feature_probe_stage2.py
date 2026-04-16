"""Stage-2 flow-feature probe focused on TCP-aware weak-anomaly signals."""

from __future__ import annotations

import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from traffic_graph.data import inspect_classic_pcap, load_pcap_flow_dataset  # noqa: E402


RESULTS_DIR = ROOT_DIR / "results"
CSV_PATH = RESULTS_DIR / "flow_feature_distribution_probe_stage2.csv"
MD_PATH = RESULTS_DIR / "flow_feature_distribution_probe_stage2.md"
PACKET_LIMIT = 20_000
RANDOM_SEED = 42
IDLE_TIMEOUT_SECONDS = 60.0

BENIGN_INPUTS: tuple[Path, ...] = (
    ROOT_DIR / "data" / "ciciot2023" / "pcap" / "benign" / "BenignTraffic.pcap",
    ROOT_DIR / "data" / "ciciot2023" / "pcap" / "benign" / "BenignTraffic1.redownload.pcap",
    ROOT_DIR / "data" / "ciciot2023" / "pcap" / "benign" / "BenignTraffic3.redownload2.pcap",
)


@dataclass(frozen=True, slots=True)
class AttackSpec:
    """One fixed weak-anomaly target for stage-2 flow probing."""

    attack_type: str
    processed_path: Path
    raw_path_for_integrity: Path | None = None


ATTACK_SPECS: tuple[AttackSpec, ...] = (
    AttackSpec(
        attack_type="browser_hijacking",
        processed_path=ROOT_DIR
        / "data"
        / "ciciot2023"
        / "pcap"
        / "malicious"
        / "web_based"
        / "BrowserHijacking.pcap",
    ),
    AttackSpec(
        attack_type="mitm_arp_spoofing",
        processed_path=ROOT_DIR / "data" / "cic_iot_2023" / "processed" / "mitm_arp_spoofing.pcap",
        raw_path_for_integrity=ROOT_DIR / "data" / "cic_iot_2023" / "raw" / "MITM-ArpSpoofing.pcap",
    ),
    AttackSpec(
        attack_type="dictionary_bruteforce",
        processed_path=ROOT_DIR / "data" / "cic_iot_2023" / "processed" / "brute_force.pcap",
        raw_path_for_integrity=ROOT_DIR / "data" / "cic_iot_2023" / "raw" / "DictionaryBruteForce.pcap",
    ),
    AttackSpec(
        attack_type="backdoor_malware",
        processed_path=ROOT_DIR / "data" / "cic_iot_2023" / "processed" / "botnet_backdoor.pcap",
        raw_path_for_integrity=ROOT_DIR / "data" / "cic_iot_2023" / "raw" / "Backdoor_Malware.pcap",
    ),
)

FEATURE_NAMES: tuple[str, ...] = (
    "coarse_ack_delay_mean",
    "coarse_ack_delay_p75",
    "ack_delay_large_gap_ratio",
    "seq_ack_match_ratio",
    "unmatched_seq_ratio",
    "unmatched_ack_ratio",
    "retry_burst_count",
    "retry_burst_max_len",
    "retry_like_dense_ratio",
    "first_packet_dir_size_pattern",
    "first_4_packet_pattern_code",
    "small_pkt_burst_count",
    "small_pkt_burst_ratio",
    "rst_after_small_burst_indicator",
)

CSV_FIELDS: tuple[str, ...] = (
    "attack_type",
    "feature_name",
    "benign_mean",
    "benign_std",
    "attack_mean",
    "attack_std",
    "signal_judgement",
    "abs_delta",
    "note",
)


def _random_window_offset(path: Path, *, source_id: str) -> int:
    """Replicate the fixed random-window sampling policy used by PCAP experiments."""

    total_packet_records, _truncated = inspect_classic_pcap(path)
    if total_packet_records <= PACKET_LIMIT:
        return 0
    max_start = total_packet_records - PACKET_LIMIT
    seed_value = sum(ord(char) for char in f"{RANDOM_SEED}:{source_id}:{path.name}")
    rng = np.random.default_rng(seed_value)
    return int(rng.integers(0, max_start + 1))


def _load_records(path: Path, *, source_id: str):
    """Load one PCAP into normalized flow records using the fixed protocol."""

    offset = _random_window_offset(path, source_id=source_id)
    load_result = load_pcap_flow_dataset(
        path,
        max_packets=PACKET_LIMIT,
        start_packet_offset=offset,
        idle_timeout_seconds=IDLE_TIMEOUT_SECONDS,
    )
    return load_result.dataset.records, offset


def _feature_values(records) -> dict[str, list[float]]:
    """Flatten one flow-record collection into stage-2 feature series."""

    feature_map: dict[str, list[float]] = {feature_name: [] for feature_name in FEATURE_NAMES}
    for record in records:
        for feature_name in FEATURE_NAMES:
            feature_map[feature_name].append(float(getattr(record, feature_name)))
    return feature_map


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Return mean/std with stable empty fallbacks."""

    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.pstdev(values))


def _signal_label(
    benign_mean: float,
    benign_std: float,
    attack_mean: float,
    attack_std: float,
) -> str:
    """Return a coarse signal-strength label from mean/std separation."""

    pooled = math.sqrt((benign_std * benign_std) + (attack_std * attack_std))
    delta = abs(attack_mean - benign_mean)
    standardized = delta if pooled <= 1e-9 else delta / pooled
    if standardized >= 1.0:
        return "high"
    if standardized >= 0.5:
        return "medium"
    return "low"


def _integrity_note(raw_path: Path | None) -> str:
    """Return a thin integrity note for attacks with known truncation."""

    if raw_path is None or not raw_path.exists():
        return ""
    _packet_count, truncated = inspect_classic_pcap(raw_path)
    return "provisional" if truncated else ""


def main() -> int:
    """Run the stage-2 flow-feature distribution probe."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    benign_records = []
    benign_offsets: dict[str, int] = {}
    for index, benign_path in enumerate(BENIGN_INPUTS):
        records, offset = _load_records(benign_path, source_id=f"benign:{index}")
        benign_offsets[benign_path.name] = offset
        benign_records.extend(records)
    benign_features = _feature_values(benign_records)

    csv_rows: list[dict[str, object]] = []
    notes_by_attack: dict[str, str] = {}
    top_rows_by_attack: dict[str, list[dict[str, object]]] = {}
    offsets_by_attack: dict[str, int] = {}

    for spec in ATTACK_SPECS:
        records, offset = _load_records(spec.processed_path, source_id=spec.attack_type)
        offsets_by_attack[spec.attack_type] = offset
        attack_features = _feature_values(records)
        note = _integrity_note(spec.raw_path_for_integrity)
        notes_by_attack[spec.attack_type] = note
        ranked_rows: list[dict[str, object]] = []
        for feature_name in FEATURE_NAMES:
            benign_mean, benign_std = _mean_std(benign_features[feature_name])
            attack_mean, attack_std = _mean_std(attack_features[feature_name])
            signal = _signal_label(benign_mean, benign_std, attack_mean, attack_std)
            delta = abs(attack_mean - benign_mean)
            row = {
                "attack_type": spec.attack_type,
                "feature_name": feature_name,
                "benign_mean": benign_mean,
                "benign_std": benign_std,
                "attack_mean": attack_mean,
                "attack_std": attack_std,
                "signal_judgement": signal,
                "abs_delta": delta,
                "note": note,
            }
            csv_rows.append(row)
            ranked_rows.append(row)
        ranked_rows.sort(
            key=lambda row: (
                {"high": 2, "medium": 1, "low": 0}[str(row["signal_judgement"])],
                float(row["abs_delta"]),
            ),
            reverse=True,
        )
        top_rows_by_attack[spec.attack_type] = ranked_rows[:4]

    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows)

    high_or_medium_rows = [
        row for row in csv_rows if str(row["signal_judgement"]) in {"high", "medium"}
    ]
    lines = [
        "# Stage-2 Flow Feature Distribution Probe",
        "",
        "Fixed protocol:",
        f"- packet_limit = `{PACKET_LIMIT}`",
        "- packet_sampling_mode = `random_window`",
        f"- seed = `{RANDOM_SEED}`",
        "",
        "Benign sources and offsets:",
    ]
    for benign_name, offset in benign_offsets.items():
        lines.append(f"- `{benign_name}` -> offset `{offset}`")
    lines.append("")

    for spec in ATTACK_SPECS:
        attack_type = spec.attack_type
        lines.extend(
            [
                f"## {attack_type}",
                "",
                f"Sampled offset: `{offsets_by_attack[attack_type]}`",
                "",
                "| feature | benign mean | benign std | attack mean | attack std | signal | abs delta |",
                "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
            ]
        )
        for row in top_rows_by_attack[attack_type]:
            lines.append(
                "| "
                f"{row['feature_name']} | "
                f"{float(row['benign_mean']):.6f} | "
                f"{float(row['benign_std']):.6f} | "
                f"{float(row['attack_mean']):.6f} | "
                f"{float(row['attack_std']):.6f} | "
                f"{row['signal_judgement']} | "
                f"{float(row['abs_delta']):.6f} |"
            )
        lines.append("")
        if notes_by_attack[attack_type]:
            lines.append(
                f"- Note: `{attack_type}` remains `{notes_by_attack[attack_type]}` because the raw PCAP is truncated."
            )
            lines.append("")

    lines.extend(
        [
            "## Observations",
            "",
        ]
    )
    if high_or_medium_rows:
        lines.append(
            f"- Stage-2 produced `{len(high_or_medium_rows)}` feature rows with `medium` or `high` signal, so these TCP-aware features look stronger than the all-`low` stage-1 snapshot."
        )
    else:
        lines.append(
            "- Stage-2 still produced no `medium` or `high` signal rows, so the new TCP-aware features do not yet look stronger than the stage-1 probe."
        )
    for attack_type, top_rows in top_rows_by_attack.items():
        labels = ", ".join(
            f"`{row['feature_name']}` ({row['signal_judgement']})"
            for row in top_rows[:3]
        )
        lines.append(
            f"- `{attack_type}` is most differentiated by {labels}."
        )
    worth_continuing = any(
        str(row["feature_name"]).startswith(("coarse_ack_delay", "seq_ack", "retry_burst"))
        and str(row["signal_judgement"]) in {"medium", "high"}
        for row in csv_rows
    )
    if worth_continuing:
        lines.append(
            "- ACK-delay proxy / seq-ack quality / retry-burst features now show enough signal to justify a follow-up that reconnects them to graph summary."
        )
        lines.append(
            "- These features are worth considering for the next graph-summary / default-hybrid follow-up."
        )
    else:
        lines.append(
            "- ACK-delay proxy / seq-ack quality / retry-burst features remain weak in this snapshot, so they are not ready to be pushed back into graph summary yet."
        )
        lines.append(
            "- This is still a useful probe, but it does not yet justify a new default-hybrid experiment."
        )
    lines.append("")

    MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(CSV_PATH.as_posix())
    print(MD_PATH.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
