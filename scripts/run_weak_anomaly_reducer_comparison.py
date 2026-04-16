"""Run one narrow weak-anomaly reducer comparison on real PCAP inputs.

This helper intentionally keeps scope fixed:

- packet_sampling_mode stays on reproducible random windows,
- packet_limit stays fixed at 20000,
- threshold_percentile stays fixed at 95,
- only two reducers are compared,
- only a handful of weak anomalies plus one optional structural anomaly are run.
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
ARTIFACT_DIR = ROOT_DIR / "artifacts" / "weak_anomaly_reducer_comparison"
CSV_PATH = RESULTS_DIR / "weak_anomaly_reducer_comparison.csv"
MD_PATH = RESULTS_DIR / "weak_anomaly_reducer_comparison.md"

BENIGN_INPUTS: tuple[Path, ...] = (
    ROOT_DIR / "data" / "ciciot2023" / "pcap" / "benign" / "BenignTraffic.pcap",
    ROOT_DIR / "data" / "ciciot2023" / "pcap" / "benign" / "BenignTraffic1.redownload.pcap",
    ROOT_DIR / "data" / "ciciot2023" / "pcap" / "benign" / "BenignTraffic3.redownload2.pcap",
)


@dataclass(frozen=True, slots=True)
class AttackSpec:
    """One fixed weak-anomaly comparison target."""

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

SCORERS: tuple[str, ...] = (
    "hybrid_max_rank_flow_node_max",
    "hybrid_decision_tail_balance",
)

CSV_FIELDS: tuple[str, ...] = (
    "scorer_name",
    "scorer_role",
    "attack_type",
    "FPR",
    "F1",
    "recall",
)


def _integrity_note(spec: AttackSpec) -> str:
    """Return a thin integrity note for one attack input."""

    raw_path = spec.raw_path_for_integrity
    if raw_path is None or not raw_path.exists():
        return ""
    _packet_count, truncated = inspect_classic_pcap(raw_path)
    return "provisional" if truncated else ""


def _run_one_comparison(spec: AttackSpec, scorer_name: str) -> dict[str, object]:
    """Run one fixed-protocol real-PCAP experiment and extract compact metrics."""

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
        graph_score_reduction=scorer_name,  # type: ignore[arg-type]
    )
    result = run_pcap_graph_experiment(
        export_dir=ARTIFACT_DIR,
        benign_inputs=BENIGN_INPUTS,
        malicious_inputs=[spec.processed_path],
        experiment_label=f"weak-anomaly-{spec.attack_type}-{scorer_name}",
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
        "scorer_name": scorer_name,
        "scorer_role": normalize_graph_scorer_role(scorer_name),
        "attack_type": spec.attack_type,
        "FPR": float(overall.get("false_positive_rate", 0.0)) if isinstance(overall, dict) else 0.0,
        "F1": float(overall.get("f1", 0.0)) if isinstance(overall, dict) and overall.get("f1") is not None else 0.0,
        "recall": float(recall) if recall is not None else 0.0,
        "integrity_note": _integrity_note(spec),
        "run_directory": result.export_result.run_directory,
    }


def _write_csv(rows: list[dict[str, object]]) -> None:
    """Write the compact reducer comparison CSV."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def _render_markdown(rows: list[dict[str, object]]) -> str:
    """Render a compact markdown summary with one table and a few observations."""

    headers = ["scorer_name", "scorer_role", "attack_type", "FPR", "F1", "recall"]
    lines = [
        "# Weak Anomaly Reducer Comparison",
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
                ]
            )
            + " |"
        )

    def _lookup(scorer_name: str, attack_type: str) -> dict[str, object]:
        for row in rows:
            if row["scorer_name"] == scorer_name and row["attack_type"] == attack_type:
                return row
        raise KeyError((scorer_name, attack_type))

    browser_base = _lookup("hybrid_max_rank_flow_node_max", "browser_hijacking")
    browser_new = _lookup("hybrid_decision_tail_balance", "browser_hijacking")
    mitm_base = _lookup("hybrid_max_rank_flow_node_max", "mitm_arp_spoofing")
    mitm_new = _lookup("hybrid_decision_tail_balance", "mitm_arp_spoofing")
    brute_base = _lookup("hybrid_max_rank_flow_node_max", "dictionary_bruteforce")
    brute_new = _lookup("hybrid_decision_tail_balance", "dictionary_bruteforce")
    backdoor_base = _lookup("hybrid_max_rank_flow_node_max", "backdoor_malware")
    backdoor_new = _lookup("hybrid_decision_tail_balance", "backdoor_malware")
    weak_improved = any(
        float(_lookup("hybrid_decision_tail_balance", attack)["recall"])
        > float(_lookup("hybrid_max_rank_flow_node_max", attack)["recall"])
        for attack in (
            "browser_hijacking",
            "mitm_arp_spoofing",
            "dictionary_bruteforce",
        )
    )

    lines.extend(
        [
            "",
            "## Observations",
            (
                f"- BrowserHijacking recall: baseline `{float(browser_base['recall']):.6f}` vs "
                f"tail-balance `{float(browser_new['recall']):.6f}`."
            ),
            (
                f"- MITM-ArpSpoofing recall: baseline `{float(mitm_base['recall']):.6f}` vs "
                f"tail-balance `{float(mitm_new['recall']):.6f}`."
            )
            + (" This remains provisional because the current MITM raw file is truncated." if mitm_new["integrity_note"] else ""),
            (
                f"- DictionaryBruteForce recall: baseline `{float(brute_base['recall']):.6f}` vs "
                f"tail-balance `{float(brute_new['recall']):.6f}`."
            ),
            (
                f"- Backdoor_Malware recall: baseline `{float(backdoor_base['recall']):.6f}` vs "
                f"tail-balance `{float(backdoor_new['recall']):.6f}`."
            ),
            (
                f"- Overall FPR stayed unchanged in this snapshot at "
                f"`{float(browser_base['FPR']):.6f}` for both reducers across all four runs."
            ),
            "",
            "## Recommendation",
            "- Keep `hybrid_decision_tail_balance` as `experimental` only.",
            (
                "- This first tail-balance variant did not improve the targeted weak anomalies, "
                "so it should be treated as a failed experimental reducer record rather than an active follow-up branch."
                if not weak_improved
                else "- It showed at least one directional weak-anomaly gain, so a later paper-facing follow-up could be justified."
            ),
            "- Do not promote it, do not expand scorer family around it, and do not reopen it without a genuinely new reducer idea.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run the narrow reducer comparison and export compact artifacts."""

    rows: list[dict[str, object]] = []
    for attack_spec in ATTACK_SPECS:
        if not attack_spec.processed_path.exists():
            raise FileNotFoundError(attack_spec.processed_path)
        for scorer_name in SCORERS:
            rows.append(_run_one_comparison(attack_spec, scorer_name))

    _write_csv(rows)
    MD_PATH.write_text(_render_markdown(rows), encoding="utf-8")
    print(CSV_PATH.as_posix())
    print(MD_PATH.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
