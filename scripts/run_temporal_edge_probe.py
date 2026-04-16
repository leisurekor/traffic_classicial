"""Probe the optional temporal edge branch against weak anomalies.

This script keeps the default mainline untouched and only compares:

1. the current graph autoencoder path
2. the opt-in temporal edge projector branch

under one fixed protocol. It reports score-distribution separation only and
does not compute F1/recall benchmark metrics.
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from traffic_graph.config import (  # noqa: E402
    AlertingConfig,
    AssociationEdgeConfig,
    DataConfig,
    EvaluationConfig,
    FeatureNormalizationConfig,
    FeaturesConfig,
    GraphConfig,
    ModelConfig,
    OutputConfig,
    PipelineConfig,
    PipelineRuntimeConfig,
    PreprocessingConfig,
    ShortFlowThresholds,
    TrainingConfig,
)
from traffic_graph.data import inspect_classic_pcap, load_pcap_flow_dataset, preprocess_flow_dataset  # noqa: E402
from traffic_graph.features import fit_feature_preprocessor, transform_graphs  # noqa: E402
from traffic_graph.graph import FlowInteractionGraphBuilder  # noqa: E402
from traffic_graph.models import GraphAutoEncoder, GraphAutoEncoderConfig  # noqa: E402
from traffic_graph.pipeline.checkpoint import load_checkpoint  # noqa: E402
from traffic_graph.pipeline.scoring import compute_graph_anomaly_scores, compute_node_anomaly_scores  # noqa: E402
from traffic_graph.pipeline.trainer import GraphAETrainer  # noqa: E402


RESULTS_DIR = ROOT_DIR / "results"
CSV_PATH = RESULTS_DIR / "temporal_edge_distribution_probe.csv"
MD_PATH = RESULTS_DIR / "temporal_edge_distribution_probe.md"
PACKET_LIMIT = 20_000
RANDOM_SEED = 42
IDLE_TIMEOUT_SECONDS = 60.0
WINDOW_SIZE = 60
TRAIN_VALIDATION_RATIO = 0.25

BENIGN_INPUTS: tuple[Path, ...] = (
    ROOT_DIR / "data" / "ciciot2023" / "pcap" / "benign" / "BenignTraffic.pcap",
    ROOT_DIR / "data" / "ciciot2023" / "pcap" / "benign" / "BenignTraffic1.redownload.pcap",
    ROOT_DIR / "data" / "ciciot2023" / "pcap" / "benign" / "BenignTraffic3.redownload2.pcap",
)


@dataclass(frozen=True, slots=True)
class AttackSpec:
    """One fixed weak-anomaly target for temporal-edge probing."""

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
    "attack_type",
    "variant",
    "benign_score_mean",
    "benign_score_std",
    "attack_score_mean",
    "attack_score_std",
    "standardized_gap",
    "signal_judgement",
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


def _load_graphs(path: Path, *, source_id: str):
    """Load one PCAP into interaction graphs under the fixed probe protocol."""

    offset = _random_window_offset(path, source_id=source_id)
    load_result = load_pcap_flow_dataset(
        path,
        max_packets=PACKET_LIMIT,
        start_packet_offset=offset,
        idle_timeout_seconds=IDLE_TIMEOUT_SECONDS,
    )
    batches = preprocess_flow_dataset(
        load_result.dataset,
        window_size=WINDOW_SIZE,
        rules=ShortFlowThresholds(),
    )
    builder = FlowInteractionGraphBuilder(
        GraphConfig(
            time_window_seconds=WINDOW_SIZE,
            directed=True,
            association_edges=AssociationEdgeConfig(
                enable_same_src_ip=True,
                enable_same_dst_subnet=True,
                dst_subnet_prefix=24,
            ),
        )
    )
    return builder.build_many(batches), offset


def _split_graphs(graphs):
    """Deterministically split benign graphs into train and validation shards."""

    ordered = list(graphs)
    if not ordered:
        raise ValueError("The temporal edge probe requires at least one benign graph.")
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(ordered)
    if len(ordered) == 1:
        return ordered[:], []
    validation_count = max(1, int(round(len(ordered) * TRAIN_VALIDATION_RATIO)))
    if validation_count >= len(ordered):
        validation_count = len(ordered) - 1
    return ordered[validation_count:], ordered[:validation_count]


def _pipeline_config(
    *,
    checkpoint_dir: str,
    use_temporal_edge_projector: bool,
) -> PipelineConfig:
    """Build a tiny pipeline config for trainer/checkpoint reuse."""

    return PipelineConfig(
        pipeline=PipelineRuntimeConfig(
            run_name="temporal-edge-probe",
            seed=RANDOM_SEED,
        ),
        data=DataConfig(input_path="unused.pcap", format="pcap"),
        preprocessing=PreprocessingConfig(
            window_size=WINDOW_SIZE,
            short_flow_thresholds=ShortFlowThresholds(),
        ),
        graph=GraphConfig(
            time_window_seconds=WINDOW_SIZE,
            directed=True,
            association_edges=AssociationEdgeConfig(
                enable_same_src_ip=True,
                enable_same_dst_subnet=True,
                dst_subnet_prefix=24,
            ),
        ),
        features=FeaturesConfig(
            normalization=FeatureNormalizationConfig(),
            use_graph_structural_features=True,
        ),
        model=ModelConfig(
            name="graph-autoencoder",
            device="cpu",
            hidden_dim=64,
            latent_dim=32,
            num_layers=2,
            dropout=0.1,
            use_edge_features=True,
            reconstruct_edge_features=True,
            use_temporal_edge_projector=use_temporal_edge_projector,
            temporal_edge_hidden_dim=32,
        ),
        training=TrainingConfig(
            epochs=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            batch_size=2,
            validation_split_ratio=TRAIN_VALIDATION_RATIO,
            early_stopping_patience=2,
            checkpoint_dir=checkpoint_dir,
            shuffle=True,
            seed=RANDOM_SEED,
            smoke_graph_limit=16,
        ),
        evaluation=EvaluationConfig(),
        alerting=AlertingConfig(),
        output=OutputConfig(directory=RESULTS_DIR.as_posix(), save_intermediate=True),
    )


def _score_graph(model: GraphAutoEncoder, packed_graph) -> float:
    """Reduce one packed graph to a plain anomaly score without reducer logic."""

    output = model(packed_graph)
    node_scores = compute_node_anomaly_scores(
        packed_graph.node_features,
        output.reconstructed_node_features.detach().cpu().numpy(),
    )
    return float(compute_graph_anomaly_scores(node_scores, reduction="mean"))


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Return mean/std with stable empty fallbacks."""

    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.pstdev(values))


def _signal_label(benign_mean: float, benign_std: float, attack_mean: float, attack_std: float) -> tuple[str, float]:
    """Return a coarse separation label and standardized gap."""

    pooled = math.sqrt((benign_std * benign_std) + (attack_std * attack_std))
    delta = abs(attack_mean - benign_mean)
    standardized = delta if pooled <= 1e-9 else delta / pooled
    if standardized >= 1.0:
        return "high", float(standardized)
    if standardized >= 0.5:
        return "medium", float(standardized)
    return "low", float(standardized)


def _integrity_note(raw_path: Path | None) -> str:
    """Return a thin integrity note for attacks with known truncation."""

    if raw_path is None or not raw_path.exists():
        return ""
    _packet_count, truncated = inspect_classic_pcap(raw_path)
    return "provisional" if truncated else ""


def _fit_and_score_variant(
    *,
    variant_name: str,
    use_temporal_edge_projector: bool,
    benign_train_graphs,
    benign_eval_graphs,
    attack_graphs_by_name,
) -> list[dict[str, object]]:
    """Train one variant and return per-attack distribution rows."""

    with TemporaryDirectory() as temp_dir:
        pipeline_config = _pipeline_config(
            checkpoint_dir=(Path(temp_dir) / variant_name / "checkpoints").as_posix(),
            use_temporal_edge_projector=use_temporal_edge_projector,
        )
        preprocessor = fit_feature_preprocessor(
            benign_train_graphs,
            normalization_config=pipeline_config.features.normalization,
            include_graph_structural_features=True,
        )
        train_packed = transform_graphs(
            benign_train_graphs,
            preprocessor,
            include_graph_structural_features=True,
        )
        val_packed = transform_graphs(
            benign_eval_graphs,
            preprocessor,
            include_graph_structural_features=True,
        )
        model = GraphAutoEncoder(
            node_input_dim=train_packed[0].node_feature_dim,
            edge_input_dim=train_packed[0].edge_feature_dim,
            config=GraphAutoEncoderConfig(
                hidden_dim=64,
                latent_dim=32,
                num_layers=2,
                dropout=0.1,
                use_edge_features=True,
                reconstruct_edge_features=True,
                use_temporal_edge_projector=use_temporal_edge_projector,
                temporal_edge_hidden_dim=32,
            ),
        )
        trainer = GraphAETrainer(
            model=model,
            config=pipeline_config,
            feature_preprocessor=preprocessor,
            device="cpu",
        )
        fit_result = trainer.fit(train_packed, val_packed, smoke_run=True)
        checkpoint = load_checkpoint(
            fit_result.best_checkpoint_path or fit_result.latest_checkpoint_path,
            map_location="cpu",
        )
        checkpoint.model.eval()

        benign_eval_scores = [
            _score_graph(checkpoint.model, packed_graph)
            for packed_graph in transform_graphs(
                benign_eval_graphs,
                preprocessor,
                include_graph_structural_features=True,
            )
        ]

        rows: list[dict[str, object]] = []
        for attack_type, attack_graphs in attack_graphs_by_name.items():
            attack_scores = [
                _score_graph(checkpoint.model, packed_graph)
                for packed_graph in transform_graphs(
                    attack_graphs,
                    preprocessor,
                    include_graph_structural_features=True,
                )
            ]
            benign_mean, benign_std = _mean_std(benign_eval_scores)
            attack_mean, attack_std = _mean_std(attack_scores)
            signal, standardized_gap = _signal_label(
                benign_mean,
                benign_std,
                attack_mean,
                attack_std,
            )
            rows.append(
                {
                    "attack_type": attack_type,
                    "variant": variant_name,
                    "benign_score_mean": benign_mean,
                    "benign_score_std": benign_std,
                    "attack_score_mean": attack_mean,
                    "attack_score_std": attack_std,
                    "standardized_gap": standardized_gap,
                    "signal_judgement": signal,
                }
            )
        return rows


def main() -> int:
    """Run the temporal edge projector probe."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    benign_graphs = []
    benign_offsets: dict[str, int] = {}
    for index, benign_path in enumerate(BENIGN_INPUTS):
        graphs, offset = _load_graphs(benign_path, source_id=f"benign:{index}")
        benign_offsets[benign_path.name] = offset
        benign_graphs.extend(graphs)
    benign_train_graphs, benign_eval_graphs = _split_graphs(benign_graphs)

    attack_graphs_by_name: dict[str, list[object]] = {}
    offsets_by_attack: dict[str, int] = {}
    notes_by_attack: dict[str, str] = {}
    for spec in ATTACK_SPECS:
        attack_graphs, offset = _load_graphs(spec.processed_path, source_id=spec.attack_type)
        attack_graphs_by_name[spec.attack_type] = attack_graphs
        offsets_by_attack[spec.attack_type] = offset
        notes_by_attack[spec.attack_type] = _integrity_note(spec.raw_path_for_integrity)

    baseline_rows = _fit_and_score_variant(
        variant_name="baseline_graphsage",
        use_temporal_edge_projector=False,
        benign_train_graphs=benign_train_graphs,
        benign_eval_graphs=benign_eval_graphs,
        attack_graphs_by_name=attack_graphs_by_name,
    )
    temporal_rows = _fit_and_score_variant(
        variant_name="temporal_edge_branch",
        use_temporal_edge_projector=True,
        benign_train_graphs=benign_train_graphs,
        benign_eval_graphs=benign_eval_graphs,
        attack_graphs_by_name=attack_graphs_by_name,
    )
    rows = baseline_rows + temporal_rows
    row_lookup = {(str(row["attack_type"]), str(row["variant"])): row for row in rows}
    for row in rows:
        note = notes_by_attack.get(str(row["attack_type"]), "")
        row["note"] = note

    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Temporal Edge Distribution Probe",
        "",
        "Fixed protocol:",
        f"- packet_limit = `{PACKET_LIMIT}`",
        "- packet_sampling_mode = `random_window`",
        f"- seed = `{RANDOM_SEED}`",
        f"- benign train graphs = `{len(benign_train_graphs)}`",
        f"- benign eval graphs = `{len(benign_eval_graphs)}`",
        "",
        "Benign source offsets:",
    ]
    for benign_name, offset in benign_offsets.items():
        lines.append(f"- `{benign_name}` -> offset `{offset}`")
    lines.append("")
    lines.extend(
        [
            "| attack | variant | benign mean | benign std | attack mean | attack std | gap | signal |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{row['attack_type']} | "
            f"{row['variant']} | "
            f"{float(row['benign_score_mean']):.6f} | "
            f"{float(row['benign_score_std']):.6f} | "
            f"{float(row['attack_score_mean']):.6f} | "
            f"{float(row['attack_score_std']):.6f} | "
            f"{float(row['standardized_gap']):.6f} | "
            f"{row['signal_judgement']} |"
        )
    lines.extend(
        [
            "",
            "## Observations",
            "",
        ]
    )
    improvement_count = 0
    for spec in ATTACK_SPECS:
        baseline = row_lookup[(spec.attack_type, "baseline_graphsage")]
        temporal = row_lookup[(spec.attack_type, "temporal_edge_branch")]
        baseline_gap = float(baseline["standardized_gap"])
        temporal_gap = float(temporal["standardized_gap"])
        delta = temporal_gap - baseline_gap
        if delta > 0.05:
            improvement_count += 1
            direction = "improves"
        elif delta < -0.05:
            direction = "regresses"
        else:
            direction = "stays close to"
        lines.append(
            f"- `{spec.attack_type}` {direction} the baseline separation: baseline gap `{baseline_gap:.6f}` vs temporal gap `{temporal_gap:.6f}`."
        )
        if notes_by_attack[spec.attack_type]:
            lines.append(
                f"- `{spec.attack_type}` remains `{notes_by_attack[spec.attack_type]}`, so this row is still only directional."
            )
    if improvement_count > 0:
        lines.append(
            f"- The temporal edge branch improves score-distribution separation on `{improvement_count}` of `{len(ATTACK_SPECS)}` attack families in this fixed snapshot."
        )
        lines.append(
            "- That is enough to justify another controlled round if we want to test whether these score-gap gains survive when translated back into the main evaluation stack."
        )
    else:
        lines.append(
            "- The temporal edge branch does not yet improve score-distribution separation in a meaningful way under this fixed protocol."
        )
        lines.append(
            "- That means the branch is runnable and isolated, but it is not yet showing clear weak-anomaly gains."
        )

    MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(CSV_PATH.as_posix())
    print(MD_PATH.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
