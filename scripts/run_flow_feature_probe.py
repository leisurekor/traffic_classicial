"""Probe flow-level feature distributions for weak-anomaly separation.

This script intentionally stops before any reducer or benchmark work. It only
asks whether the newly attached flow-level features show measurable separation
between benign traffic and weak-anomaly captures under the current fixed
packet-sampling protocol.
"""

from __future__ import annotations

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
OUTPUT_PATH = RESULTS_DIR / "flow_feature_distribution_probe.md"
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
    """One fixed weak-anomaly target for flow-level feature probing."""

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


def _random_window_offset(path: Path, *, source_id: str) -> int:
    """Replicate the fixed random-window sampling policy used by PCAP experiments."""

    total_packet_records, _truncated = inspect_classic_pcap(path)
    if total_packet_records <= PACKET_LIMIT:
        return 0
    max_start = total_packet_records - PACKET_LIMIT
    seed_value = sum(ord(char) for char in f"{RANDOM_SEED}:{source_id}:{path.name}")
    rng = np.random.default_rng(seed_value)
    return int(rng.integers(0, max_start + 1))


def _feature_values(records) -> dict[str, list[float]]:
    """Flatten one flow-record collection into scalar feature series."""

    feature_map: dict[str, list[float]] = {
        "retry_like_count": [],
        "retry_like_ratio": [],
        "flag_syn_ratio": [],
        "flag_ack_ratio": [],
        "flag_rst_ratio": [],
        "flag_pattern_code": [],
        "first_packet_size_pattern": [],
    }
    for index in range(6):
        feature_map[f"iat_hist_bin_{index}"] = []
        feature_map[f"pkt_len_hist_bin_{index}"] = []

    for record in records:
        feature_map["retry_like_count"].append(float(record.retry_like_count))
        feature_map["retry_like_ratio"].append(float(record.retry_like_ratio))
        feature_map["flag_syn_ratio"].append(float(record.flag_syn_ratio))
        feature_map["flag_ack_ratio"].append(float(record.flag_ack_ratio))
        feature_map["flag_rst_ratio"].append(float(record.flag_rst_ratio))
        feature_map["flag_pattern_code"].append(float(record.flag_pattern_code))
        feature_map["first_packet_size_pattern"].append(float(record.first_packet_size_pattern))
        for index in range(6):
            iat_hist = record.iat_hist if len(record.iat_hist) == 6 else (0.0,) * 6
            pkt_len_hist = record.pkt_len_hist if len(record.pkt_len_hist) == 6 else (0.0,) * 6
            feature_map[f"iat_hist_bin_{index}"].append(float(iat_hist[index]))
            feature_map[f"pkt_len_hist_bin_{index}"].append(float(pkt_len_hist[index]))
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
    """Return a thin integrity note for attack captures with known truncation."""

    if raw_path is None or not raw_path.exists():
        return ""
    _packet_count, truncated = inspect_classic_pcap(raw_path)
    return "provisional" if truncated else ""


def _load_records(path: Path, *, source_id: str):
    """Load one PCAP into normalized flow records using the fixed sampling protocol."""

    offset = _random_window_offset(path, source_id=source_id)
    load_result = load_pcap_flow_dataset(
        path,
        max_packets=PACKET_LIMIT,
        start_packet_offset=offset,
        idle_timeout_seconds=IDLE_TIMEOUT_SECONDS,
    )
    return load_result.dataset.records, offset


def _render_table(
    attack_type: str,
    attack_features: dict[str, list[float]],
    benign_features: dict[str, list[float]],
) -> tuple[list[str], list[tuple[str, float, float, float, float, str, float]]]:
    """Build markdown lines and sortable feature rows for one attack type."""

    lines = [
        f"## {attack_type}",
        "",
        "| feature | benign mean | benign std | attack mean | attack std | signal | abs delta |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    ranked_rows: list[tuple[str, float, float, float, float, str, float]] = []
    for feature_name in sorted(benign_features):
        benign_mean, benign_std = _mean_std(benign_features[feature_name])
        attack_mean, attack_std = _mean_std(attack_features[feature_name])
        signal = _signal_label(benign_mean, benign_std, attack_mean, attack_std)
        delta = abs(attack_mean - benign_mean)
        ranked_rows.append(
            (
                feature_name,
                benign_mean,
                benign_std,
                attack_mean,
                attack_std,
                signal,
                delta,
            )
        )

    ranked_rows.sort(
        key=lambda row: (
            {"high": 2, "medium": 1, "low": 0}[row[5]],
            row[6],
        ),
        reverse=True,
    )
    for feature_name, benign_mean, benign_std, attack_mean, attack_std, signal, delta in ranked_rows:
        lines.append(
            "| "
            f"{feature_name} | {benign_mean:.6f} | {benign_std:.6f} | "
            f"{attack_mean:.6f} | {attack_std:.6f} | {signal} | {delta:.6f} |"
        )
    lines.append("")
    return lines, ranked_rows


def main() -> int:
    """Run the fixed flow-feature distribution probe and write markdown output."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    benign_records = []
    benign_offsets: dict[str, int] = {}
    for index, benign_path in enumerate(BENIGN_INPUTS):
        records, offset = _load_records(benign_path, source_id=f"benign:{index}")
        benign_offsets[benign_path.name] = offset
        benign_records.extend(records)
    benign_features = _feature_values(benign_records)

    lines = [
        "# Flow Feature Distribution Probe",
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

    observations: list[str] = []
    for spec in ATTACK_SPECS:
        records, offset = _load_records(spec.processed_path, source_id=spec.attack_type)
        attack_features = _feature_values(records)
        attack_lines, ranked_rows = _render_table(spec.attack_type, attack_features, benign_features)
        lines.extend(attack_lines)
        note = _integrity_note(spec.raw_path_for_integrity)
        if note:
            lines.append(f"- Note: `{spec.attack_type}` remains `{note}` because the raw PCAP is truncated.")
            lines.append("")
        top_rows = ranked_rows[:3]
        if top_rows:
            summary_bits = ", ".join(
                f"`{feature_name}` ({signal})"
                for feature_name, _bm, _bs, _am, _as, signal, _delta in top_rows
            )
            observations.append(
                f"- `{spec.attack_type}` most clearly separates on {summary_bits}; sampled offset `{offset}`."
            )

    lines.append("## Observations")
    lines.append("")
    if observations:
        lines.extend(observations)
    lines.append(
        "- Features that repeatedly land in `high` or `medium` signal across BrowserHijacking, MITM, and DictionaryBruteForce are the best candidates for the next flow-schema follow-up. This probe stops here on purpose and does not touch reducers or benchmark metrics."
    )
    lines.append("")

    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT_PATH.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
