"""Compare the current default hybrid reducer against the frozen legacy snapshot.

This helper keeps scope intentionally narrow:

- fixed random-window protocol,
- fixed packet_limit=20000,
- fixed seed=42,
- fixed scorer name (`hybrid_max_rank_flow_node_max`),
- comparison only against the previously recorded legacy baseline rows.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from traffic_graph.data import inspect_classic_pcap  # noqa: E402
from traffic_graph.pipeline.pcap_graph_experiment import (  # noqa: E402
    PcapGraphExperimentConfig,
    run_pcap_graph_experiment,
)
from traffic_graph.pipeline.scorer_roles import normalize_graph_scorer_role  # noqa: E402


RESULTS_DIR = ROOT_DIR / "results"
ARTIFACT_DIR = ROOT_DIR / "artifacts" / "weak_anomaly_hybrid_comparison"
CSV_PATH = RESULTS_DIR / "weak_anomaly_hybrid_comparison.csv"
MD_PATH = RESULTS_DIR / "weak_anomaly_hybrid_comparison.md"
LEGACY_CSV_PATH = RESULTS_DIR / "weak_anomaly_reducer_comparison.csv"

BENIGN_INPUTS: tuple[Path, ...] = (
    ROOT_DIR / "data" / "ciciot2023" / "pcap" / "benign" / "BenignTraffic.pcap",
    ROOT_DIR / "data" / "ciciot2023" / "pcap" / "benign" / "BenignTraffic1.redownload.pcap",
    ROOT_DIR / "data" / "ciciot2023" / "pcap" / "benign" / "BenignTraffic3.redownload2.pcap",
)


@dataclass(frozen=True, slots=True)
class AttackSpec:
    """One fixed weak-anomaly target for the default-hybrid refresh check."""

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

CSV_FIELDS: tuple[str, ...] = (
    "scorer_name",
    "scorer_role",
    "attack_type",
    "FPR",
    "F1",
    "recall",
    "note",
)


def _integrity_note(spec: AttackSpec) -> str:
    """Return a thin integrity note for one attack input."""

    raw_path = spec.raw_path_for_integrity
    if raw_path is None or not raw_path.exists():
        return ""
    _packet_count, truncated = inspect_classic_pcap(raw_path)
    return "provisional" if truncated else ""


def _load_legacy_rows() -> dict[str, dict[str, object]]:
    """Load the frozen legacy-hybrid baseline from the previous comparison artifact."""

    if not LEGACY_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Legacy weak-anomaly baseline is missing: {LEGACY_CSV_PATH.as_posix()}"
        )
    rows: dict[str, dict[str, object]] = {}
    with LEGACY_CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("scorer_name") != "hybrid_max_rank_flow_node_max":
                continue
            attack_type = str(row.get("attack_type", "")).strip()
            if attack_type:
                rows[attack_type] = {
                    "scorer_name": "hybrid_legacy_reference",
                    "scorer_role": "reference",
                    "attack_type": attack_type,
                    "FPR": float(row.get("FPR", 0.0) or 0.0),
                    "F1": float(row.get("F1", 0.0) or 0.0),
                    "recall": float(row.get("recall", 0.0) or 0.0),
                    "note": "historical_prefix_fixed_random_window_reference",
                }
    missing = [spec.attack_type for spec in ATTACK_SPECS if spec.attack_type not in rows]
    if missing:
        raise ValueError(f"Legacy baseline missing attacks: {', '.join(missing)}")
    return rows


def _run_current_hybrid(spec: AttackSpec) -> dict[str, object]:
    """Run the fixed current default scorer once and extract compact metrics."""

    config = PcapGraphExperimentConfig(
        packet_limit=20000,
        packet_sampling_mode="random_window",
        window_size=10,
        use_association_edges=True,
        use_graph_structural_features=True,
        epochs=1,
        batch_size=1,
        threshold_percentile=95.0,
        random_seed=42,
        graph_score_reduction="hybrid_max_rank_flow_node_max",
    )
    result = run_pcap_graph_experiment(
        export_dir=ARTIFACT_DIR,
        benign_inputs=BENIGN_INPUTS,
        malicious_inputs=[spec.processed_path],
        experiment_label=f"weak-anomaly-hybrid-refresh-{spec.attack_type}",
        config=config,
    )
    overall = result.summary.get("overall_metrics", {})
    malicious_rows = result.summary.get("malicious_source_metrics", [])
    recall = None
    if isinstance(malicious_rows, list) and malicious_rows:
        recall = malicious_rows[0].get("recall")
    if recall is None and isinstance(overall, dict):
        recall = overall.get("recall")
    return {
        "scorer_name": "hybrid_max_rank_flow_node_max",
        "scorer_role": normalize_graph_scorer_role("hybrid_max_rank_flow_node_max"),
        "attack_type": spec.attack_type,
        "FPR": float(overall.get("false_positive_rate", 0.0)) if isinstance(overall, dict) else 0.0,
        "F1": float(overall.get("f1", 0.0)) if isinstance(overall, dict) and overall.get("f1") is not None else 0.0,
        "recall": float(recall) if recall is not None else 0.0,
        "note": _integrity_note(spec),
    }


def _write_csv(rows: list[dict[str, object]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def _render_markdown(rows: list[dict[str, object]]) -> str:
    headers = ["scorer_name", "scorer_role", "attack_type", "FPR", "F1", "recall", "note"]
    lines = [
        "# Weak Anomaly Hybrid Comparison",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["scorer_name"]),
                    str(row["scorer_role"]),
                    str(row["attack_type"]),
                    f"{float(row['FPR']):.6f}",
                    f"{float(row['F1']):.6f}",
                    f"{float(row['recall']):.6f}",
                    str(row.get("note", "")),
                ]
            )
            + " |"
        )

    def _lookup(scorer_name: str, attack_type: str) -> dict[str, object]:
        for row in rows:
            if row["scorer_name"] == scorer_name and row["attack_type"] == attack_type:
                return row
        raise KeyError((scorer_name, attack_type))

    browser_base = _lookup("hybrid_legacy_reference", "browser_hijacking")
    browser_new = _lookup("hybrid_max_rank_flow_node_max", "browser_hijacking")
    mitm_base = _lookup("hybrid_legacy_reference", "mitm_arp_spoofing")
    mitm_new = _lookup("hybrid_max_rank_flow_node_max", "mitm_arp_spoofing")
    brute_base = _lookup("hybrid_legacy_reference", "dictionary_bruteforce")
    brute_new = _lookup("hybrid_max_rank_flow_node_max", "dictionary_bruteforce")
    backdoor_base = _lookup("hybrid_legacy_reference", "backdoor_malware")
    backdoor_new = _lookup("hybrid_max_rank_flow_node_max", "backdoor_malware")

    recall_gains = {
        "BrowserHijacking": float(browser_new["recall"]) - float(browser_base["recall"]),
        "MITM-ArpSpoofing": float(mitm_new["recall"]) - float(mitm_base["recall"]),
        "DictionaryBruteForce": float(brute_new["recall"]) - float(brute_base["recall"]),
        "Backdoor_Malware": float(backdoor_new["recall"]) - float(backdoor_base["recall"]),
    }
    fpr_delta = float(browser_new["FPR"]) - float(browser_base["FPR"])
    improved_attacks = [
        attack_name for attack_name, delta in recall_gains.items() if delta > 1e-9
    ]
    regressed_attacks = [
        attack_name for attack_name, delta in recall_gains.items() if delta < -1e-9
    ]

    lines.extend(
        [
            "",
            "## Observations",
            (
                f"- BrowserHijacking recall: legacy `{float(browser_base['recall']):.6f}` vs "
                f"current `{float(browser_new['recall']):.6f}`."
            ),
            (
                f"- MITM-ArpSpoofing recall: legacy `{float(mitm_base['recall']):.6f}` vs "
                f"current `{float(mitm_new['recall']):.6f}`."
            )
            + (" This remains provisional because the MITM raw file is truncated." if mitm_new["note"] else ""),
            (
                f"- DictionaryBruteForce recall: legacy `{float(brute_base['recall']):.6f}` vs "
                f"current `{float(brute_new['recall']):.6f}`."
            ),
            (
                f"- Backdoor_Malware recall: legacy `{float(backdoor_base['recall']):.6f}` vs "
                f"current `{float(backdoor_new['recall']):.6f}`."
            ),
            (
                f"- BrowserHijacking FPR: legacy `{float(browser_base['FPR']):.6f}` vs "
                f"current `{float(browser_new['FPR']):.6f}`."
            ),
            "",
            "## Recommendation",
        ]
    )
    if improved_attacks:
        lines.append(
            "- The new graph-level calibration is materially more aggressive: it lifts "
            + ", ".join(improved_attacks)
            + " relative to the frozen legacy reference."
        )
    if regressed_attacks:
        lines.append(
            "- This gain is not free. It regresses "
            + ", ".join(regressed_attacks)
            + " under the same fixed seed-42 protocol."
        )
    if fpr_delta > 1e-9:
        lines.append(
            f"- The main tradeoff is false positives: FPR rises by `{fpr_delta:.6f}` in the fixed weak-anomaly snapshot."
        )
    elif fpr_delta < -1e-9:
        lines.append(
            f"- FPR improves by `{abs(fpr_delta):.6f}` while keeping the new weak-anomaly calibration."
        )
    else:
        lines.append("- FPR stays flat in this fixed weak-anomaly snapshot.")
    lines.extend(
        [
            "- Treat this as an aggressive detection profile built on the new edge-centric representation, not as a quiet no-op refresh.",
            "- It still does not reopen the failed `hybrid_decision_tail_balance` reducer branch; the improvement comes from graph-level calibration of the new representation, not from reviving that old reducer idea.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    legacy_rows = _load_legacy_rows()
    rows: list[dict[str, object]] = []
    for attack_spec in ATTACK_SPECS:
        if not attack_spec.processed_path.exists():
            raise FileNotFoundError(attack_spec.processed_path)
        rows.append(dict(legacy_rows[attack_spec.attack_type]))
        rows.append(_run_current_hybrid(attack_spec))

    _write_csv(rows)
    MD_PATH.write_text(_render_markdown(rows), encoding="utf-8")
    print(CSV_PATH.as_posix())
    print(MD_PATH.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
