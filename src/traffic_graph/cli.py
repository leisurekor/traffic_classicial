"""Command-line interface helpers for the traffic graph repository."""

from __future__ import annotations

import argparse
from pathlib import Path
from collections.abc import Sequence

from traffic_graph.config import PipelineConfig
from traffic_graph.pipeline import PipelineRunner
from traffic_graph.explain.prompt_replay_types import PromptDatasetReplay


def _parse_bool_token(value: str) -> bool:
    """Parse a textual boolean token for CLI sweep options."""

    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean token: {value!r}")


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the repository pipeline."""

    parser = argparse.ArgumentParser(
        description="Preview or run the unsupervised traffic graph pipeline skeleton."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML configuration file path.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Override the configured input flow dataset path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override the configured output directory.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name for logs and artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render the planned stages without touching input data.",
    )
    parser.add_argument(
        "--show-window-stats",
        action="store_true",
        help="Load the input dataset and print per-window preprocessing statistics.",
    )
    parser.add_argument(
        "--show-graph-summary",
        action="store_true",
        help="Build endpoint graphs from logical flows and print per-window graph summaries.",
    )
    parser.add_argument(
        "--show-feature-summary",
        action="store_true",
        help="Pack node and edge features, apply configured preprocessing, and print per-window feature dimensions.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the minimal graph autoencoder after feature preparation.",
    )
    parser.add_argument(
        "--smoke-train",
        action="store_true",
        help="Train on a tiny graph subset with a short epoch budget for smoke tests.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Load a trained checkpoint and evaluate anomaly scores on the input data.",
    )
    parser.add_argument(
        "--show-alert-summary",
        action="store_true",
        help="When evaluating, derive structured alerts and print a per-scope summary.",
    )
    parser.add_argument(
        "--show-explanation-summary",
        action="store_true",
        help="Build explanation-ready samples from a replay bundle or exported eval bundle and print a compact summary.",
    )
    parser.add_argument(
        "--explanation-scope",
        type=str,
        choices=("graph", "flow", "node"),
        default="flow",
        help="Explanation candidate scope used by the lightweight explanation summary helpers.",
    )
    parser.add_argument(
        "--explanation-top-k",
        type=int,
        default=20,
        help="Keep at most the top-k explanation candidates after sorting by anomaly score.",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default=None,
        help="Optional base directory for persisted score, alert, and metric bundles.",
    )
    parser.add_argument(
        "--replay-bundle",
        type=str,
        default=None,
        help="Load a previously exported run bundle or manifest.json and print a replay summary.",
    )
    parser.add_argument(
        "--replay-prompts",
        type=str,
        default=None,
        help="Load a previously exported prompt dataset bundle or manifest.json and print a replay summary.",
    )
    parser.add_argument(
        "--run-llm-stub",
        action="store_true",
        help="Run the mock LLM batch executor on a replayed prompt dataset and export structured results.",
    )
    parser.add_argument(
        "--llm-model-name",
        type=str,
        default="mock-llm-stub",
        help="Model name recorded in mock or future real LLM result artifacts.",
    )
    parser.add_argument(
        "--llm-output-dir",
        type=str,
        default=None,
        help="Optional output directory for exported LLM result artifacts.",
    )
    parser.add_argument(
        "--build-prompts",
        action="store_true",
        help="Build and export an LLM-ready prompt dataset from explanation samples and surrogate-tree rule records.",
    )
    parser.add_argument(
        "--prompt-scope",
        type=str,
        choices=("graph", "flow", "node"),
        default="flow",
        help="Explanation scope used when building prompt datasets.",
    )
    parser.add_argument(
        "--prompt-top-k",
        type=int,
        default=20,
        help="Keep at most the top-k prompts after score sorting.",
    )
    parser.add_argument(
        "--prompt-output-dir",
        type=str,
        default=None,
        help="Optional output directory for the exported prompt dataset.",
    )
    parser.add_argument(
        "--prompt-only-alerts",
        action="store_true",
        help="Restrict prompt datasets to alert samples before applying top-k selection.",
    )
    parser.add_argument(
        "--prompt-balanced",
        action="store_true",
        help="Select a balanced alert/non-alert subset before exporting prompts.",
    )
    parser.add_argument(
        "--prompt-max-samples",
        type=int,
        default=50,
        help="Maximum number of samples to keep when balanced prompt selection is enabled.",
    )
    parser.add_argument(
        "--train-surrogate-tree",
        action="store_true",
        help="Fit a surrogate decision tree from explanation-ready samples derived from a replay bundle or exported eval bundle.",
    )
    parser.add_argument(
        "--show-rule-summary",
        action="store_true",
        help="After training a surrogate tree, extract and print structured rule records.",
    )
    parser.add_argument(
        "--surrogate-mode",
        type=str,
        choices=("regression", "classification"),
        default="regression",
        help="Surrogate tree training mode.",
    )
    parser.add_argument(
        "--surrogate-scope",
        type=str,
        choices=("graph", "flow", "node"),
        default="flow",
        help="Explanation sample scope used to train the surrogate tree.",
    )
    parser.add_argument(
        "--surrogate-max-depth",
        type=int,
        default=4,
        help="Maximum depth of the surrogate decision tree.",
    )
    parser.add_argument(
        "--surrogate-min-samples-leaf",
        type=int,
        default=5,
        help="Minimum samples per leaf for the surrogate decision tree.",
    )
    parser.add_argument(
        "--surrogate-random-state",
        type=int,
        default=42,
        help="Random seed used by the surrogate decision tree.",
    )
    parser.add_argument(
        "--surrogate-output-dir",
        type=str,
        default=None,
        help="Optional directory where the surrogate tree artifact will be persisted.",
    )
    parser.add_argument(
        "--rule-output-dir",
        type=str,
        default=None,
        help="Optional directory or JSONL path where extracted rule records will be written.",
    )
    parser.add_argument(
        "--build-heldout-tasks",
        action="store_true",
        help="Prepare held-out attack evaluation tasks from a merged CIC-style CSV dataset.",
    )
    parser.add_argument(
        "--heldout-input",
        type=str,
        default=None,
        help="Path to a merged CSV file such as Merged01.csv. If omitted, the CLI falls back to artifacts/cic_iot2023/Merged01.csv when present.",
    )
    parser.add_argument(
        "--heldout-output-dir",
        type=str,
        default=None,
        help="Optional output directory for the held-out attack protocol bundle.",
    )
    parser.add_argument(
        "--heldout-attack-types",
        nargs="*",
        default=None,
        help="Held-out attack family selectors such as Recon, DDoS, Mirai, or Web-based. When omitted, the default protocol groups are used.",
    )
    parser.add_argument(
        "--heldout-min-samples-per-attack",
        type=int,
        default=10,
        help="Minimum attack-sample count required before a held-out task is kept.",
    )
    parser.add_argument(
        "--heldout-random-seed",
        type=int,
        default=42,
        help="Random seed used for the benign train/eval split and task shuffling.",
    )
    parser.add_argument(
        "--heldout-benign-train-ratio",
        type=float,
        default=0.7,
        help="Fraction of benign rows reserved for the training split.",
    )
    parser.add_argument(
        "--heldout-label-column",
        type=str,
        default=None,
        help="Override the detected label column name in the merged CSV.",
    )
    parser.add_argument(
        "--heldout-benign-label",
        type=str,
        default="BENIGN",
        help="Raw label token treated as benign when building held-out tasks.",
    )
    parser.add_argument(
        "--prepare-binary-experiment",
        action="store_true",
        help="Prepare unsupervised binary train/val/test splits from a merged CIC-style CSV dataset.",
    )
    parser.add_argument(
        "--binary-input",
        type=str,
        default=None,
        help="Path to a merged CSV file such as Merged01.csv. If omitted, the CLI falls back to artifacts/cic_iot2023/Merged01.csv when present.",
    )
    parser.add_argument(
        "--binary-output-dir",
        type=str,
        default=None,
        help="Optional output directory for the prepared binary experiment bundle.",
    )
    parser.add_argument(
        "--binary-label-column",
        type=str,
        default=None,
        help="Override the detected label column name in the merged CSV.",
    )
    parser.add_argument(
        "--binary-benign-label",
        type=str,
        default="BENIGN",
        help="Raw label token treated as benign when constructing the binary target.",
    )
    parser.add_argument(
        "--binary-split-mode",
        type=str,
        choices=("random", "stratified"),
        default="stratified",
        help="Split mode used for the binary experiment input construction.",
    )
    parser.add_argument(
        "--binary-train-ratio",
        type=float,
        default=0.6,
        help="Training ratio used for the binary experiment split.",
    )
    parser.add_argument(
        "--binary-val-ratio",
        type=float,
        default=0.2,
        help="Validation ratio used for the binary experiment split.",
    )
    parser.add_argument(
        "--binary-test-ratio",
        type=float,
        default=0.2,
        help="Test ratio used for the binary experiment split.",
    )
    parser.add_argument(
        "--binary-random-seed",
        type=int,
        default=42,
        help="Random seed used for the binary experiment split.",
    )
    parser.add_argument(
        "--train-normal-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict the training split to benign samples only.",
    )
    parser.add_argument(
        "--run-binary-detection-experiment",
        action="store_true",
        help="Run the merged CSV binary detection experiment and export overall / per-attack reports.",
    )
    parser.add_argument(
        "--binary-detection-input",
        type=str,
        default=None,
        help="Path to the merged CSV used for the binary detection experiment. Falls back to artifacts/cic_iot2023/Merged01.csv when available.",
    )
    parser.add_argument(
        "--binary-detection-output-dir",
        type=str,
        default=None,
        help="Optional output directory for the binary detection report bundle.",
    )
    parser.add_argument(
        "--binary-detection-threshold-percentile",
        type=float,
        default=95.0,
        help="Percentile of benign training scores used to set the anomaly threshold.",
    )
    parser.add_argument(
        "--binary-detection-max-components",
        type=int,
        default=10,
        help="Maximum PCA components used by the tabular reconstruction scorer.",
    )
    parser.add_argument(
        "--binary-detection-random-seed",
        type=int,
        default=42,
        help="Random seed used by the tabular scorer and protocol preparation.",
    )
    parser.add_argument(
        "--binary-detection-model-mode",
        type=str,
        choices=("tabular", "graph"),
        default="tabular",
        help="Backend used for the binary detection experiment.",
    )
    parser.add_argument(
        "--compare-binary-detection-runs",
        action="store_true",
        help="Compare two exported binary detection runs and emit a unified comparison report.",
    )
    parser.add_argument(
        "--tabular-run-dir",
        type=str,
        default=None,
        help="Path to the exported tabular baseline run directory or manifest.json.",
    )
    parser.add_argument(
        "--graph-run-dir",
        type=str,
        default=None,
        help="Path to the exported graph-mode run directory or manifest.json.",
    )
    parser.add_argument(
        "--comparison-output-dir",
        type=str,
        default=None,
        help="Optional output directory for the comparison report bundle.",
    )
    parser.add_argument(
        "--comparison-markdown",
        action="store_true",
        help="Also export a Markdown version of the comparison report.",
    )
    parser.add_argument(
        "--run-graph-ablation-suite",
        action="store_true",
        help="Run a graph-mode ablation sweep over association edges, window size, and structural features.",
    )
    parser.add_argument(
        "--graph-ablation-output-dir",
        type=str,
        default=None,
        help="Optional output directory for the graph ablation suite bundle.",
    )
    parser.add_argument(
        "--graph-ablation-window-sizes",
        nargs="*",
        type=int,
        default=None,
        help="Window-size sweep values for the graph ablation suite. Defaults to 30 60 120 300 when omitted.",
    )
    parser.add_argument(
        "--graph-ablation-use-association-edges",
        nargs="*",
        type=_parse_bool_token,
        default=None,
        help="Boolean sweep values for association-edge ablations. Defaults to false true when omitted.",
    )
    parser.add_argument(
        "--graph-ablation-use-graph-structural-features",
        nargs="*",
        type=_parse_bool_token,
        default=None,
        help="Boolean sweep values for graph-structure feature ablations. Defaults to false true when omitted.",
    )
    parser.add_argument(
        "--run-pcap-graph-smoke-experiment",
        action="store_true",
        help="Parse a real PCAP, build endpoint graphs, and run a minimal graph smoke experiment.",
    )
    parser.add_argument(
        "--run-pcap-graph-experiment",
        action="store_true",
        help="Run a reproducible mini PCAP graph experiment. Recommended usage is at least one real benign control PCAP plus one or more malicious PCAPs; training always uses merged benign graphs only, and one-sided runs fall back to smoke validation mode.",
    )
    parser.add_argument(
        "--pcap-input",
        type=str,
        default=None,
        help="Path to a classic Ethernet PCAP capture. Falls back to artifacts/cic_iot2023/Recon-HostDiscovery.pcap when present.",
    )
    parser.add_argument(
        "--pcap-output-dir",
        type=str,
        default=None,
        help="Optional output directory for PCAP smoke or mini binary-evaluation bundles.",
    )
    parser.add_argument(
        "--pcap-experiment-label",
        type=str,
        default=None,
        help="Optional short label recorded in PCAP experiment summaries and comparison-ready artifacts, for example benign_vs_recon_small or baseline_window30_assoc_on.",
    )
    parser.add_argument(
        "--pcap-benign-input",
        action="append",
        default=None,
        help="Benign PCAP input used by the reproducible PCAP graph experiment. Repeat the flag to provide multiple files; all benign graphs are merged before benign-only train/holdout splitting.",
    )
    parser.add_argument(
        "--pcap-malicious-input",
        action="append",
        default=None,
        help="Malicious PCAP input used by the reproducible PCAP graph experiment. Repeat the flag to provide multiple files; all malicious graphs are kept for evaluation only and are additionally summarized at source level.",
    )
    parser.add_argument(
        "--pcap-packet-limit",
        type=int,
        default=5000,
        help="Maximum number of packets to parse from the PCAP capture during the smoke run.",
    )
    parser.add_argument(
        "--pcap-idle-timeout-seconds",
        type=float,
        default=60.0,
        help="Idle timeout used when splitting packet streams into flow sessions.",
    )
    parser.add_argument(
        "--pcap-window-size",
        type=int,
        default=60,
        help="Time-window size in seconds for building endpoint interaction graphs.",
    )
    parser.add_argument(
        "--pcap-short-flow-packet-threshold",
        type=int,
        default=5,
        help="Packet-count threshold used to classify short flows in the PCAP smoke run.",
    )
    parser.add_argument(
        "--pcap-short-flow-byte-threshold",
        type=int,
        default=1024,
        help="Byte-count threshold used to classify short flows in the PCAP smoke run.",
    )
    parser.add_argument(
        "--pcap-short-flow-duration-threshold",
        type=float,
        default=None,
        help="Optional duration threshold used to classify short flows in the PCAP smoke run.",
    )
    parser.add_argument(
        "--pcap-use-association-edges",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable same-src-IP and same-destination-subnet association edges for the PCAP smoke run.",
    )
    parser.add_argument(
        "--pcap-use-graph-structural-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable graph structural node features in the PCAP smoke run.",
    )
    parser.add_argument(
        "--pcap-smoke-graph-limit",
        type=int,
        default=16,
        help="Maximum number of windowed graphs kept for the PCAP smoke run.",
    )
    parser.add_argument(
        "--pcap-train-epochs",
        type=int,
        default=2,
        help="Epoch budget used when PyTorch is available for the PCAP graph experiment. Without torch the CLI automatically falls back to the deterministic scorer.",
    )
    parser.add_argument(
        "--pcap-batch-size",
        type=int,
        default=2,
        help="Mini-batch size used when PyTorch is available for the PCAP smoke run.",
    )
    parser.add_argument(
        "--pcap-learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate used when PyTorch is available for the PCAP smoke run.",
    )
    parser.add_argument(
        "--pcap-threshold-percentile",
        type=float,
        default=95.0,
        help="Percentile used to derive the smoke-run anomaly threshold from training scores.",
    )
    parser.add_argument(
        "--pcap-graph-score-reduction",
        type=str,
        choices=(
            "mean_node",
            "flow_p90",
            "node_max",
            "hybrid_max_rank_flow_node_max",
            "hybrid_decision_tail_balance",
            "decision_topk_flow_node",
            "relation_max_flow_server_count",
            "structural_fig_max",
        ),
        default="hybrid_max_rank_flow_node_max",
        help=(
            "Final graph-score reduction used after scoring one graph. "
            "`hybrid_max_rank_flow_node_max` is the current default candidate, "
            "`flow_p90` promotes sparse high-anomaly communication edges, "
            "`node_max` is a conservative node-only alternative, and `mean_node` "
            "retains the older mean-based baseline. "
            "`hybrid_max_rank_flow_node_max` applies "
            "a thin train-reference-aware max-percentile fusion over flow_p90 and node_max. "
            "`hybrid_decision_tail_balance` adds a decision-style tail-balance experimental "
            "reducer that emphasizes p75/top-k score mass for weaker anomalies. "
            "`decision_topk_flow_node` applies FlowMiner-style decision pooling over "
            "flow_p90 and node_score_p90, `relation_max_flow_server_count` adds a thin "
            "FG-SAT-inspired server-side relation summary, and `structural_fig_max` adds "
            "a HyperVision/ICAD-inspired lightweight structural summary."
        ),
    )
    parser.add_argument(
        "--pcap-random-seed",
        type=int,
        default=42,
        help="Random seed used by the PCAP smoke run.",
    )
    parser.add_argument(
        "--pcap-benign-train-ratio",
        type=float,
        default=0.7,
        help="Fraction of merged benign graphs reserved for train/validation in binary PCAP experiments before benign holdout evaluation is formed.",
    )
    return parser


def _prepare_binary_experiment_from_csv(
    input_path: str | Path,
    *,
    output_dir: str | None,
    label_column: str | None,
    benign_label: str,
    split_mode: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
    train_normal_only: bool,
) -> list[str]:
    """Prepare and export a binary experiment bundle from a merged CSV file."""

    from traffic_graph.data import (
        BinaryExperimentConfig,
        export_binary_experiment,
        prepare_binary_experiment,
        summarize_binary_experiment_text,
    )

    source_path = Path(input_path)
    artifact = prepare_binary_experiment(
        source_path,
        BinaryExperimentConfig(
            label_column=label_column,
            benign_label=benign_label,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed,
            train_normal_only=train_normal_only,
            split_mode=split_mode,  # type: ignore[arg-type]
        ),
    )
    export_base_dir = Path(output_dir) if output_dir is not None else Path("artifacts") / "binary_experiments"
    export_result = export_binary_experiment(artifact, export_base_dir)
    rendered_sections = [
        "Binary experiment summary:",
        summarize_binary_experiment_text(artifact),
        f"Saved binary experiment bundle to {export_result.output_directory}.",
        f"Manifest path: {export_result.manifest_path}",
    ]
    if export_result.artifact_paths:
        rendered_sections.append(
            "Binary experiment artifacts:\n"
            + "\n".join(
                f"  - {name}: {path}" for name, path in export_result.artifact_paths.items()
            )
        )
    if export_result.notes:
        rendered_sections.append(
            "Binary experiment notes:\n"
            + "\n".join(f"  - {note}" for note in export_result.notes)
        )
    return rendered_sections


def _build_heldout_attack_protocol_from_csv(
    input_path: str | Path,
    *,
    output_dir: str | None,
    label_column: str | None,
    benign_label: str,
    attack_types: Sequence[str] | None,
    min_samples_per_attack: int,
    random_seed: int,
    benign_train_ratio: float,
) -> list[str]:
    """Prepare and export a held-out attack protocol bundle from a merged CSV file."""

    from traffic_graph.data import (
        DEFAULT_HELD_OUT_ATTACK_TYPES,
        HeldOutAttackProtocolConfig,
        export_heldout_attack_protocol,
        prepare_heldout_attack_protocol,
        summarize_heldout_attack_protocol_text,
    )

    source_path = Path(input_path)
    selected_attack_types = (
        tuple(str(item) for item in attack_types)
        if attack_types
        else DEFAULT_HELD_OUT_ATTACK_TYPES
    )
    artifact = prepare_heldout_attack_protocol(
        source_path,
        HeldOutAttackProtocolConfig(
            label_column=label_column,
            benign_label=benign_label,
            held_out_attack_types=selected_attack_types,
            min_samples_per_attack=min_samples_per_attack,
            random_seed=random_seed,
            benign_train_ratio=benign_train_ratio,
        ),
    )
    export_base_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path("artifacts") / "heldout_attack_protocols"
    )
    export_result = export_heldout_attack_protocol(artifact, export_base_dir)
    rendered_sections = [
        "Held-out attack protocol summary:",
        summarize_heldout_attack_protocol_text(artifact),
        f"Saved held-out protocol bundle to {export_result.output_directory}.",
        f"Manifest path: {export_result.manifest_path}",
    ]
    if export_result.artifact_paths:
        rendered_sections.append(
            "Held-out protocol artifacts:\n"
            + "\n".join(
                f"  - {name}: {path}" for name, path in export_result.artifact_paths.items()
            )
        )
    if export_result.task_manifest_paths:
        rendered_sections.append(
            "Task manifests:\n"
            + "\n".join(
                f"  - {name}: {path}"
                for name, path in export_result.task_manifest_paths.items()
            )
        )
    if export_result.notes:
        rendered_sections.append(
            "Held-out protocol notes:\n"
            + "\n".join(f"  - {note}" for note in export_result.notes)
        )
    return rendered_sections


def _train_surrogate_tree_from_bundle(
    bundle_path: str | Path,
    *,
    scope: str,
    mode: str,
    max_depth: int,
    min_samples_leaf: int,
    random_state: int,
    output_dir: str | None,
    show_rule_summary: bool,
    rule_output_dir: str | None,
) -> list[str]:
    """Train a surrogate tree from a replay bundle and optionally persist it."""

    from traffic_graph.explain import (
        SurrogateTreeConfig,
        build_explanation_samples,
        extract_rules_for_samples,
        export_rule_records,
        save_surrogate_tree_artifact,
        summarize_rules,
        summarize_surrogate_tree_artifact,
        train_surrogate_tree,
    )
    from traffic_graph.pipeline.replay_io import load_export_bundle

    source_path = Path(bundle_path)
    bundle = load_export_bundle(source_path)
    samples = build_explanation_samples(
        bundle,
        scope=scope,  # type: ignore[arg-type]
        only_alerts=False,
    )
    artifact = train_surrogate_tree(
        samples,
        SurrogateTreeConfig(
            mode=mode,  # type: ignore[arg-type]
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        ),
    )
    rendered_sections = [
        "Surrogate tree summary:",
        summarize_surrogate_tree_artifact(artifact),
        (
            "Feature names: "
            + (", ".join(artifact.feature_names) if artifact.feature_names else "none")
        ),
    ]
    if output_dir is not None:
        save_result = save_surrogate_tree_artifact(artifact, output_dir)
        rendered_sections.extend(
            [
                f"Saved surrogate tree artifact to {save_result.output_directory}.",
                f"Model path: {save_result.model_path}",
                f"Metadata path: {save_result.metadata_path}",
            ]
        )
    else:
        base_directory = source_path if source_path.is_dir() else source_path.parent
        save_result = save_surrogate_tree_artifact(
            artifact,
            (base_directory / "surrogate_tree" / scope / mode).as_posix(),
        )
        rendered_sections.extend(
            [
                f"Saved surrogate tree artifact to {save_result.output_directory}.",
                f"Model path: {save_result.model_path}",
                f"Metadata path: {save_result.metadata_path}",
            ]
        )
    if show_rule_summary or rule_output_dir is not None:
        rule_records = extract_rules_for_samples(artifact, samples)
        if show_rule_summary:
            rendered_sections.append(
                "\n".join(
                    [
                        "Rule record summary:",
                        summarize_rules(rule_records),
                    ]
                )
            )
        if rule_output_dir is not None:
            rule_output_path = Path(rule_output_dir)
            if rule_output_path.suffix.lower() == ".jsonl":
                export_path = rule_output_path
            else:
                export_path = rule_output_path / "rule_records.jsonl"
            saved_rule_path = export_rule_records(rule_records, export_path)
            rendered_sections.append(f"Saved rule records to {saved_rule_path}.")
    return rendered_sections


def _build_prompt_dataset_from_bundle(
    bundle_path: str | Path,
    *,
    scope: str,
    only_alerts: bool,
    top_k: int,
    balanced: bool,
    max_samples: int,
    output_dir: str | None,
    surrogate_mode: str,
    max_depth: int,
    min_samples_leaf: int,
    random_state: int,
) -> list[str]:
    """Build and export a prompt dataset from a replay bundle."""

    from traffic_graph.explain import (
        SurrogateTreeConfig,
        build_explanation_samples,
        build_prompt_dataset,
        export_prompt_dataset,
        extract_rules_for_samples,
        summarize_prompt_dataset_text,
        train_surrogate_tree,
    )
    from traffic_graph.pipeline.replay_io import load_export_bundle

    source_path = Path(bundle_path)
    bundle = load_export_bundle(source_path)
    samples = build_explanation_samples(
        bundle,
        scope=scope,  # type: ignore[arg-type]
        only_alerts=False,
    )
    rendered_sections = []
    if not samples:
        return [
            "Prompt dataset could not be built because no explanation samples were available."
        ]
    artifact = train_surrogate_tree(
        samples,
        SurrogateTreeConfig(
            mode=surrogate_mode,  # type: ignore[arg-type]
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        ),
    )
    rule_records = extract_rules_for_samples(artifact, samples)
    prompt_dataset = build_prompt_dataset(
        samples,
        rule_records,
        scope=scope,  # type: ignore[arg-type]
        only_alerts=only_alerts,
        top_k=top_k,
        balanced=balanced,
        max_samples=max_samples,
        alert_records=bundle.alert_records,
        score_records=(
            tuple(bundle.graph_scores)
            + tuple(bundle.flow_scores)
            + tuple(bundle.node_scores)
        ),
    )
    export_base_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(bundle.manifest.base_directory) / "prompt_datasets"
    )
    export_result = export_prompt_dataset(prompt_dataset, export_base_dir)
    rendered_sections.extend(
        [
            "Prompt dataset summary:",
            summarize_prompt_dataset_text(prompt_dataset),
            f"Saved prompt dataset to {export_result.output_directory}.",
            f"Manifest path: {export_result.manifest_path}",
        ]
    )
    if export_result.artifact_paths:
        rendered_sections.append(
            "Prompt dataset artifacts:\n"
            + "\n".join(
                f"  - {name}: {path}" for name, path in export_result.artifact_paths.items()
            )
        )
    if export_result.notes:
        rendered_sections.append(
            "Prompt dataset notes:\n"
            + "\n".join(f"  - {note}" for note in export_result.notes)
        )
    return rendered_sections


def _run_llm_stub_from_prompt_dataset(
    prompt_dataset: PromptDatasetReplay,
    *,
    model_name: str,
    output_dir: str | None,
) -> list[str]:
    """Run the mock LLM executor against a replayed prompt dataset."""

    from traffic_graph.explain import (
        export_llm_results,
        run_llm_stub,
        summarize_llm_results_text,
    )

    llm_results = run_llm_stub(prompt_dataset, model_name=model_name)
    base_directory = Path(prompt_dataset.manifest.base_directory or ".")
    export_base_dir = Path(output_dir) if output_dir is not None else base_directory / "llm_results"
    export_result = export_llm_results(llm_results, export_base_dir)
    rendered_sections = [
        "LLM stub result summary:",
        summarize_llm_results_text(llm_results),
        f"Saved LLM results to {export_result.output_directory}.",
        f"Manifest path: {export_result.manifest_path}",
    ]
    if export_result.artifact_paths:
        rendered_sections.append(
            "LLM result artifacts:\n"
            + "\n".join(
                f"  - {name}: {path}" for name, path in export_result.artifact_paths.items()
            )
        )
    if export_result.notes:
        rendered_sections.append(
            "LLM result notes:\n"
            + "\n".join(f"  - {note}" for note in export_result.notes)
        )
    return rendered_sections


def _run_binary_detection_experiment_from_csv(
    input_path: str | Path,
    *,
    output_dir: str | None,
    label_column: str | None,
    benign_label: str,
    binary_train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    train_normal_only: bool,
    binary_split_mode: str,
    heldout_attack_types: Sequence[str] | None,
    heldout_min_samples_per_attack: int,
    heldout_benign_train_ratio: float,
    heldout_random_seed: int,
    threshold_percentile: float,
    max_components: int,
    random_seed: int,
    model_mode: str,
) -> list[str]:
    """Run the merged CSV binary detection experiment end to end."""

    from traffic_graph.data import (
        BinaryExperimentConfig,
        DEFAULT_HELD_OUT_ATTACK_TYPES,
        HeldOutAttackProtocolConfig,
    )
    from traffic_graph.pipeline.graph_binary_detection import (
        run_graph_binary_detection_experiment,
        summarize_graph_binary_detection_report,
    )
    from traffic_graph.pipeline.binary_detection import (
        run_binary_detection_experiment,
        summarize_binary_detection_report,
    )

    source_path = Path(input_path)
    binary_config = BinaryExperimentConfig(
        label_column=label_column,
        benign_label=benign_label,
        train_ratio=binary_train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
        train_normal_only=train_normal_only,
        split_mode=binary_split_mode,  # type: ignore[arg-type]
    )
    heldout_config = HeldOutAttackProtocolConfig(
        label_column=label_column,
        benign_label=benign_label,
        held_out_attack_types=(
            tuple(heldout_attack_types)
            if heldout_attack_types
            else DEFAULT_HELD_OUT_ATTACK_TYPES
        ),
        min_samples_per_attack=heldout_min_samples_per_attack,
        random_seed=heldout_random_seed,
        benign_train_ratio=heldout_benign_train_ratio,
    )
    report_output_dir = (
        output_dir if output_dir is not None else Path("artifacts") / "binary_detection_reports"
    )
    if model_mode == "graph":
        report, export_result = run_graph_binary_detection_experiment(
            source_path,
            report_output_dir,
            binary_experiment_config=binary_config,
            heldout_protocol_config=heldout_config,
            threshold_percentile=threshold_percentile,
            graph_config=None,
            random_seed=random_seed,
        )
        summarize_report = summarize_graph_binary_detection_report
    else:
        report, export_result = run_binary_detection_experiment(
            source_path,
            report_output_dir,
            binary_experiment_config=binary_config,
            heldout_protocol_config=heldout_config,
            threshold_percentile=threshold_percentile,
            max_components=max_components,
            random_seed=random_seed,
        )
        summarize_report = summarize_binary_detection_report
    rendered_sections = [
        "Binary detection experiment summary:",
        f"Model mode: {model_mode}",
        summarize_report(report),
        f"Saved binary detection report to {export_result.output_directory}.",
        f"Manifest path: {export_result.manifest_path}",
    ]
    if export_result.artifact_paths:
        rendered_sections.append(
            "Binary detection artifacts:\n"
            + "\n".join(
                f"  - {name}: {path}" for name, path in export_result.artifact_paths.items()
            )
        )
    if export_result.notes:
        rendered_sections.append(
            "Binary detection notes:\n"
            + "\n".join(f"  - {note}" for note in export_result.notes)
        )
    return rendered_sections


def _compare_binary_detection_runs_from_dirs(
    tabular_run_dir: str | Path,
    graph_run_dir: str | Path,
    *,
    output_dir: str | None,
    export_markdown: bool,
) -> list[str]:
    """Compare two exported binary detection runs and optionally persist the report."""

    from traffic_graph.pipeline.compare_binary_detection_runs import (
        compare_and_export_binary_detection_runs,
        compare_binary_detection_runs,
        summarize_comparison,
    )

    report = compare_binary_detection_runs(tabular_run_dir, graph_run_dir)
    rendered_sections = [
        "Binary detection comparison summary:",
        summarize_comparison(report),
    ]
    if output_dir is not None or export_markdown:
        export_base_dir = (
            Path(output_dir)
            if output_dir is not None
            else Path("artifacts") / "binary_detection_comparisons"
        )
        _, export_result = compare_and_export_binary_detection_runs(
            tabular_run_dir,
            graph_run_dir,
            export_base_dir,
            export_markdown=export_markdown,
        )
        rendered_sections.extend(
            [
                f"Saved comparison report to {export_result.comparison_directory}.",
                f"Manifest path: {export_result.manifest_path}",
            ]
        )
        if export_result.artifact_paths:
            rendered_sections.append(
                "Comparison artifacts:\n"
                + "\n".join(
                    f"  - {name}: {path}" for name, path in export_result.artifact_paths.items()
                )
            )
        if export_result.notes:
            rendered_sections.append(
                "Comparison notes:\n"
                + "\n".join(f"  - {note}" for note in export_result.notes)
            )
    return rendered_sections


def _run_graph_ablation_suite_from_csv(
    input_path: str | Path,
    *,
    output_dir: str | None,
    label_column: str | None,
    benign_label: str,
    binary_train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    train_normal_only: bool,
    binary_split_mode: str,
    heldout_attack_types: Sequence[str] | None,
    heldout_min_samples_per_attack: int,
    heldout_benign_train_ratio: float,
    heldout_random_seed: int,
    threshold_percentile: float,
    random_seed: int,
    window_sizes: Sequence[int] | None,
    use_association_edges_options: Sequence[bool] | None,
    use_graph_structural_features_options: Sequence[bool] | None,
) -> list[str]:
    """Run the graph ablation suite against a merged CIC-style CSV file."""

    from traffic_graph.data import (
        BinaryExperimentConfig,
        DEFAULT_HELD_OUT_ATTACK_TYPES,
        HeldOutAttackProtocolConfig,
    )
    from traffic_graph.pipeline.graph_ablation import (
        run_graph_ablation_suite,
        summarize_graph_ablation_suite,
    )

    source_path = Path(input_path)
    binary_config = BinaryExperimentConfig(
        label_column=label_column,
        benign_label=benign_label,
        train_ratio=binary_train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
        train_normal_only=train_normal_only,
        split_mode=binary_split_mode,  # type: ignore[arg-type]
    )
    heldout_config = HeldOutAttackProtocolConfig(
        label_column=label_column,
        benign_label=benign_label,
        held_out_attack_types=(
            tuple(heldout_attack_types)
            if heldout_attack_types
            else DEFAULT_HELD_OUT_ATTACK_TYPES
        ),
        min_samples_per_attack=heldout_min_samples_per_attack,
        random_seed=heldout_random_seed,
        benign_train_ratio=heldout_benign_train_ratio,
    )
    sweep_window_sizes = (
        tuple(int(value) for value in window_sizes)
        if window_sizes
        else (30, 60, 120, 300)
    )
    sweep_association_edges = (
        tuple(bool(value) for value in use_association_edges_options)
        if use_association_edges_options
        else (False, True)
    )
    sweep_structural_features = (
        tuple(bool(value) for value in use_graph_structural_features_options)
        if use_graph_structural_features_options
        else (False, True)
    )
    ablation_output_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path("artifacts") / "graph_ablation_reports"
    )
    suite = run_graph_ablation_suite(
        source_path,
        ablation_output_dir,
        binary_experiment_config=binary_config,
        heldout_protocol_config=heldout_config,
        window_sizes=sweep_window_sizes,
        use_association_edges_options=sweep_association_edges,
        use_graph_structural_features_options=sweep_structural_features,
        threshold_percentile=threshold_percentile,
        random_seed=random_seed,
    )
    rendered_sections = [
        "Graph ablation suite summary:",
        summarize_graph_ablation_suite(suite),
    ]
    if suite.artifact_paths:
        rendered_sections.append(
            "Graph ablation artifacts:\n"
            + "\n".join(
                f"  - {name}: {path}" for name, path in suite.artifact_paths.items()
            )
        )
    if suite.notes:
        rendered_sections.append(
            "Graph ablation notes:\n"
            + "\n".join(f"  - {note}" for note in suite.notes)
        )
    return rendered_sections


def _run_pcap_graph_smoke_experiment_from_pcap(
    input_path: str | Path,
    *,
    output_dir: str | None,
    packet_limit: int | None,
    idle_timeout_seconds: float,
    window_size: int,
    short_flow_packet_threshold: int,
    short_flow_byte_threshold: int,
    short_flow_duration_threshold: float | None,
    use_association_edges: bool,
    use_graph_structural_features: bool,
    smoke_graph_limit: int,
    train_epochs: int,
    batch_size: int,
    learning_rate: float,
    threshold_percentile: float,
    graph_score_reduction: str,
    random_seed: int,
) -> list[str]:
    """Run the real-PCAP smoke experiment end to end."""

    from traffic_graph.data import ShortFlowThresholds
    from traffic_graph.pipeline.pcap_graph_smoke import (
        PcapGraphSmokeConfig,
        run_pcap_graph_smoke_experiment,
        summarize_pcap_graph_smoke_result,
    )

    source_path = Path(input_path)
    smoke_output_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path("artifacts") / "pcap_graph_smoke_reports"
    )
    smoke_config = PcapGraphSmokeConfig(
        packet_limit=packet_limit,
        idle_timeout_seconds=idle_timeout_seconds,
        window_size=window_size,
        short_flow_thresholds=ShortFlowThresholds(
            packet_count_lt=short_flow_packet_threshold,
            byte_count_lt=short_flow_byte_threshold,
            duration_seconds_lt=short_flow_duration_threshold,
        ),
        use_association_edges=use_association_edges,
        use_graph_structural_features=use_graph_structural_features,
        smoke_graph_limit=smoke_graph_limit,
        train_validation_ratio=0.25,
        graph_score_reduction=graph_score_reduction,
        epochs=train_epochs,
        batch_size=batch_size,
        random_seed=random_seed,
        threshold_percentile=threshold_percentile,
        learning_rate=learning_rate,
        checkpoint_dir=str(smoke_output_dir / "checkpoints"),
    )
    result = run_pcap_graph_smoke_experiment(
        source_path,
        smoke_output_dir,
        config=smoke_config,
    )
    rendered_sections = [
        summarize_pcap_graph_smoke_result(result),
        f"Saved PCAP smoke bundle to {result.export_result.run_directory if result.export_result else smoke_output_dir}.",
    ]
    if result.export_result is not None:
        rendered_sections.append(f"Manifest path: {result.export_result.manifest_path}")
        if result.export_result.artifact_paths:
            rendered_sections.append(
                "PCAP smoke artifacts:\n"
                + "\n".join(
                    f"  - {name}: {path}"
                    for name, path in result.export_result.artifact_paths.items()
                )
            )
        if result.export_result.notes:
            rendered_sections.append(
                "PCAP smoke export notes:\n"
                + "\n".join(f"  - {note}" for note in result.export_result.notes)
            )
    return rendered_sections


def _run_pcap_graph_experiment_from_pcaps(
    *,
    benign_inputs: Sequence[str] | None,
    malicious_inputs: Sequence[str] | None,
    fallback_input: str | None,
    output_dir: str | None,
    experiment_label: str | None,
    packet_limit: int | None,
    idle_timeout_seconds: float,
    window_size: int,
    short_flow_packet_threshold: int,
    short_flow_byte_threshold: int,
    short_flow_duration_threshold: float | None,
    use_association_edges: bool,
    use_graph_structural_features: bool,
    smoke_graph_limit: int,
    benign_train_ratio: float,
    train_epochs: int,
    batch_size: int,
    learning_rate: float,
    threshold_percentile: float,
    graph_score_reduction: str,
    random_seed: int,
) -> list[str]:
    """Run the reproducible mini PCAP experiment in smoke or binary mode."""

    from traffic_graph.data import ShortFlowThresholds
    from traffic_graph.pipeline.pcap_graph_experiment import (
        PcapGraphExperimentConfig,
        run_pcap_graph_experiment,
        summarize_pcap_graph_experiment_result,
    )

    normalized_benign_inputs = list(benign_inputs or [])
    normalized_malicious_inputs = list(malicious_inputs or [])
    if not normalized_benign_inputs and not normalized_malicious_inputs and fallback_input:
        normalized_malicious_inputs.append(fallback_input)

    if not normalized_benign_inputs and not normalized_malicious_inputs:
        raise ValueError(
            "The PCAP experiment requires at least one --pcap-benign-input, "
            "--pcap-malicious-input, or fallback --pcap-input."
        )

    experiment_output_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path("artifacts") / "pcap_graph_experiments"
    )
    experiment_config = PcapGraphExperimentConfig(
        packet_limit=packet_limit,
        idle_timeout_seconds=idle_timeout_seconds,
        window_size=window_size,
        short_flow_thresholds=ShortFlowThresholds(
            packet_count_lt=short_flow_packet_threshold,
            byte_count_lt=short_flow_byte_threshold,
            duration_seconds_lt=short_flow_duration_threshold,
        ),
        use_association_edges=use_association_edges,
        use_graph_structural_features=use_graph_structural_features,
        smoke_graph_limit=smoke_graph_limit,
        benign_train_ratio=benign_train_ratio,
        train_validation_ratio=0.25,
        graph_score_reduction=graph_score_reduction,
        epochs=train_epochs,
        batch_size=batch_size,
        random_seed=random_seed,
        threshold_percentile=threshold_percentile,
        learning_rate=learning_rate,
        checkpoint_dir=str(experiment_output_dir / "checkpoints"),
    )
    result = run_pcap_graph_experiment(
        export_dir=experiment_output_dir,
        benign_inputs=normalized_benign_inputs,
        malicious_inputs=normalized_malicious_inputs,
        experiment_label=experiment_label,
        config=experiment_config,
    )
    rendered_sections = [
        summarize_pcap_graph_experiment_result(result),
        f"Saved PCAP experiment bundle to {result.export_result.run_directory}.",
        f"Manifest path: {result.export_result.manifest_path}",
    ]
    if result.export_result.artifact_paths:
        rendered_sections.append(
            "PCAP experiment artifacts:\n"
            + "\n".join(
                f"  - {name}: {path}"
                for name, path in result.export_result.artifact_paths.items()
            )
        )
    if result.export_result.notes:
        rendered_sections.append(
            "PCAP experiment notes:\n"
            + "\n".join(f"  - {note}" for note in result.export_result.notes)
        )
    return rendered_sections


def main(argv: Sequence[str] | None = None) -> int:
    """Run the repository pipeline CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.build_heldout_tasks:
        default_input = Path("artifacts") / "cic_iot2023" / "Merged01.csv"
        heldout_input = args.heldout_input or (
            default_input.as_posix() if default_input.exists() else None
        )
        if heldout_input is None:
            parser.error(
                "--build-heldout-tasks requires --heldout-input or an existing artifacts/cic_iot2023/Merged01.csv."
            )
        print(
            "\n\n".join(
                _build_heldout_attack_protocol_from_csv(
                    heldout_input,
                    output_dir=args.heldout_output_dir,
                    label_column=args.heldout_label_column,
                    benign_label=args.heldout_benign_label,
                    attack_types=args.heldout_attack_types,
                    min_samples_per_attack=args.heldout_min_samples_per_attack,
                    random_seed=args.heldout_random_seed,
                    benign_train_ratio=args.heldout_benign_train_ratio,
                )
            )
        )
        return 0

    if args.run_binary_detection_experiment:
        default_input = Path("artifacts") / "cic_iot2023" / "Merged01.csv"
        binary_detection_input = args.binary_detection_input or args.binary_input or (
            default_input.as_posix() if default_input.exists() else None
        )
        if binary_detection_input is None:
            parser.error(
                "--run-binary-detection-experiment requires --binary-detection-input or an existing artifacts/cic_iot2023/Merged01.csv."
            )
        try:
            rendered_sections = _run_binary_detection_experiment_from_csv(
                binary_detection_input,
                output_dir=args.binary_detection_output_dir,
                label_column=args.binary_label_column,
                benign_label=args.binary_benign_label,
                binary_train_ratio=args.binary_train_ratio,
                val_ratio=args.binary_val_ratio,
                test_ratio=args.binary_test_ratio,
                train_normal_only=args.train_normal_only,
                binary_split_mode=args.binary_split_mode,
                heldout_attack_types=args.heldout_attack_types,
                heldout_min_samples_per_attack=args.heldout_min_samples_per_attack,
                heldout_benign_train_ratio=args.heldout_benign_train_ratio,
                heldout_random_seed=args.heldout_random_seed,
                threshold_percentile=args.binary_detection_threshold_percentile,
                max_components=args.binary_detection_max_components,
                random_seed=args.binary_detection_random_seed,
                model_mode=args.binary_detection_model_mode,
            )
        except RuntimeError as exc:
            parser.error(str(exc))
        print("\n\n".join(rendered_sections))
        return 0

    if args.compare_binary_detection_runs:
        if args.tabular_run_dir is None or args.graph_run_dir is None:
            parser.error(
                "--compare-binary-detection-runs requires both --tabular-run-dir and --graph-run-dir."
            )
        print(
            "\n\n".join(
                _compare_binary_detection_runs_from_dirs(
                    args.tabular_run_dir,
                    args.graph_run_dir,
                    output_dir=args.comparison_output_dir,
                    export_markdown=args.comparison_markdown,
                )
            )
        )
        return 0

    if args.run_graph_ablation_suite:
        default_input = Path("artifacts") / "cic_iot2023" / "Merged01.csv"
        graph_ablation_input = args.binary_detection_input or args.binary_input or (
            default_input.as_posix() if default_input.exists() else None
        )
        if graph_ablation_input is None:
            parser.error(
                "--run-graph-ablation-suite requires --binary-detection-input or an existing artifacts/cic_iot2023/Merged01.csv."
            )
        print(
            "\n\n".join(
                _run_graph_ablation_suite_from_csv(
                    graph_ablation_input,
                    output_dir=args.graph_ablation_output_dir,
                    label_column=args.binary_label_column,
                    benign_label=args.binary_benign_label,
                    binary_train_ratio=args.binary_train_ratio,
                    val_ratio=args.binary_val_ratio,
                    test_ratio=args.binary_test_ratio,
                    train_normal_only=args.train_normal_only,
                    binary_split_mode=args.binary_split_mode,
                    heldout_attack_types=args.heldout_attack_types,
                    heldout_min_samples_per_attack=args.heldout_min_samples_per_attack,
                    heldout_benign_train_ratio=args.heldout_benign_train_ratio,
                    heldout_random_seed=args.heldout_random_seed,
                    threshold_percentile=args.binary_detection_threshold_percentile,
                    random_seed=args.binary_detection_random_seed,
                    window_sizes=args.graph_ablation_window_sizes,
                    use_association_edges_options=args.graph_ablation_use_association_edges,
                    use_graph_structural_features_options=(
                        args.graph_ablation_use_graph_structural_features
                    ),
                )
            )
        )
        return 0

    if args.run_pcap_graph_experiment:
        default_input = (
            Path("artifacts") / "cic_iot2023" / "Recon-HostDiscovery.pcap"
        )
        fallback_pcap_input = args.pcap_input or (
            default_input.as_posix() if default_input.exists() else None
        )
        print(
            "\n\n".join(
                _run_pcap_graph_experiment_from_pcaps(
                    benign_inputs=args.pcap_benign_input,
                    malicious_inputs=args.pcap_malicious_input,
                    fallback_input=fallback_pcap_input,
                    output_dir=args.pcap_output_dir,
                    experiment_label=args.pcap_experiment_label,
                    packet_limit=args.pcap_packet_limit,
                    idle_timeout_seconds=args.pcap_idle_timeout_seconds,
                    window_size=args.pcap_window_size,
                    short_flow_packet_threshold=args.pcap_short_flow_packet_threshold,
                    short_flow_byte_threshold=args.pcap_short_flow_byte_threshold,
                    short_flow_duration_threshold=(
                        args.pcap_short_flow_duration_threshold
                    ),
                    use_association_edges=args.pcap_use_association_edges,
                    use_graph_structural_features=(
                        args.pcap_use_graph_structural_features
                    ),
                    smoke_graph_limit=args.pcap_smoke_graph_limit,
                    benign_train_ratio=args.pcap_benign_train_ratio,
                    train_epochs=args.pcap_train_epochs,
                    batch_size=args.pcap_batch_size,
                    learning_rate=args.pcap_learning_rate,
                    threshold_percentile=args.pcap_threshold_percentile,
                    graph_score_reduction=args.pcap_graph_score_reduction,
                    random_seed=args.pcap_random_seed,
                )
            )
        )
        return 0

    if args.run_pcap_graph_smoke_experiment:
        default_input = (
            Path("artifacts")
            / "cic_iot2023"
            / "Recon-HostDiscovery.pcap"
        )
        pcap_input = args.pcap_input or (
            default_input.as_posix() if default_input.exists() else None
        )
        if pcap_input is None:
            parser.error(
                "--run-pcap-graph-smoke-experiment requires --pcap-input or an existing artifacts/cic_iot2023/Recon-HostDiscovery.pcap."
            )
        print(
            "\n\n".join(
                _run_pcap_graph_smoke_experiment_from_pcap(
                    pcap_input,
                    output_dir=args.pcap_output_dir,
                    packet_limit=args.pcap_packet_limit,
                    idle_timeout_seconds=args.pcap_idle_timeout_seconds,
                    window_size=args.pcap_window_size,
                    short_flow_packet_threshold=args.pcap_short_flow_packet_threshold,
                    short_flow_byte_threshold=args.pcap_short_flow_byte_threshold,
                    short_flow_duration_threshold=(
                        args.pcap_short_flow_duration_threshold
                    ),
                    use_association_edges=args.pcap_use_association_edges,
                    use_graph_structural_features=(
                        args.pcap_use_graph_structural_features
                    ),
                    smoke_graph_limit=args.pcap_smoke_graph_limit,
                    train_epochs=args.pcap_train_epochs,
                    batch_size=args.pcap_batch_size,
                    learning_rate=args.pcap_learning_rate,
                    threshold_percentile=args.pcap_threshold_percentile,
                    graph_score_reduction=args.pcap_graph_score_reduction,
                    random_seed=args.pcap_random_seed,
                )
            )
        )
        return 0

    if args.prepare_binary_experiment:
        default_input = Path("artifacts") / "cic_iot2023" / "Merged01.csv"
        binary_input = args.binary_input or (
            default_input.as_posix() if default_input.exists() else None
        )
        if binary_input is None:
            parser.error(
                "--prepare-binary-experiment requires --binary-input or an existing artifacts/cic_iot2023/Merged01.csv."
            )
        print(
            "\n\n".join(
                _prepare_binary_experiment_from_csv(
                    binary_input,
                    output_dir=args.binary_output_dir,
                    label_column=args.binary_label_column,
                    benign_label=args.binary_benign_label,
                    split_mode=args.binary_split_mode,
                    train_ratio=args.binary_train_ratio,
                    val_ratio=args.binary_val_ratio,
                    test_ratio=args.binary_test_ratio,
                    random_seed=args.binary_random_seed,
                    train_normal_only=args.train_normal_only,
                )
            )
        )
        return 0

    if args.run_llm_stub and args.replay_prompts is None:
        parser.error("--run-llm-stub requires --replay-prompts.")

    if args.replay_prompts is not None:
        from traffic_graph.explain.prompt_replay import (
            load_prompt_dataset,
            summarize_prompt_dataset_text,
        )

        prompt_dataset = load_prompt_dataset(args.replay_prompts)
        rendered_sections = [summarize_prompt_dataset_text(prompt_dataset)]
        if args.run_llm_stub:
            rendered_sections.append(
                "\n".join(
                    _run_llm_stub_from_prompt_dataset(
                        prompt_dataset,
                        model_name=args.llm_model_name,
                        output_dir=args.llm_output_dir,
                    )
                )
            )
        print("\n\n".join(rendered_sections))
        return 0

    if args.replay_bundle is not None:
        from traffic_graph.explain.explanation_samples import (
            build_explanation_samples,
            summarize_explanation_samples_text,
        )
        from traffic_graph.pipeline.replay_io import (
            load_export_bundle,
            summarize_replay_bundle,
        )

        bundle = load_export_bundle(args.replay_bundle)
        rendered_sections = [summarize_replay_bundle(bundle)]
        if args.show_explanation_summary:
            samples = build_explanation_samples(
                bundle,
                scope=args.explanation_scope,
                only_alerts=True,
                top_k=args.explanation_top_k,
            )
            rendered_sections.append(
                "\n".join(
                    [
                        "Explanation candidate summary:",
                        (
                            "Selection: "
                            f"scope={args.explanation_scope}, "
                            "only_alerts=True, "
                            f"top_k={args.explanation_top_k}"
                        ),
                        summarize_explanation_samples_text(samples),
                    ]
                )
            )
        if args.train_surrogate_tree:
            rendered_sections.append(
                "\n".join(
                    _train_surrogate_tree_from_bundle(
                        args.replay_bundle,
                        scope=args.surrogate_scope,
                        mode=args.surrogate_mode,
                        max_depth=args.surrogate_max_depth,
                        min_samples_leaf=args.surrogate_min_samples_leaf,
                        random_state=args.surrogate_random_state,
                        output_dir=args.surrogate_output_dir,
                        show_rule_summary=args.show_rule_summary,
                        rule_output_dir=args.rule_output_dir,
                    )
                )
            )
        if args.build_prompts:
            rendered_sections.append(
                "\n".join(
                    _build_prompt_dataset_from_bundle(
                        args.replay_bundle,
                        scope=args.prompt_scope,
                        only_alerts=args.prompt_only_alerts,
                        top_k=args.prompt_top_k,
                        balanced=args.prompt_balanced,
                        max_samples=args.prompt_max_samples,
                        output_dir=args.prompt_output_dir,
                        surrogate_mode=args.surrogate_mode,
                        max_depth=args.surrogate_max_depth,
                        min_samples_leaf=args.surrogate_min_samples_leaf,
                        random_state=args.surrogate_random_state,
                    )
                )
            )
        print("\n\n".join(rendered_sections))
        return 0

    config = (
        PipelineConfig.from_yaml(args.config)
        if args.config is not None
        else PipelineConfig()
    )
    config = config.with_overrides(
        input_path=args.input,
        output_directory=args.output,
        run_name=args.run_name,
    )

    runner = PipelineRunner(config=config)
    report = runner.run(
        dry_run=args.dry_run,
        show_window_stats=args.show_window_stats,
        show_graph_summary=args.show_graph_summary,
        show_feature_summary=args.show_feature_summary,
        show_alert_summary=args.show_alert_summary,
        show_explanation_summary=args.show_explanation_summary,
        explanation_scope=args.explanation_scope,
        explanation_top_k=args.explanation_top_k,
        export_dir=args.export_dir,
        train=args.train or args.smoke_train,
        smoke_train=args.smoke_train,
        evaluate=args.eval,
    )
    rendered_sections = [report.render()]
    if args.train_surrogate_tree:
        export_manifest_path = report.export_artifacts.get("manifest_json")
        if export_manifest_path:
            rendered_sections.append(
                "\n".join(
                    _train_surrogate_tree_from_bundle(
                        export_manifest_path,
                        scope=args.surrogate_scope,
                        mode=args.surrogate_mode,
                        max_depth=args.surrogate_max_depth,
                        min_samples_leaf=args.surrogate_min_samples_leaf,
                        random_state=args.surrogate_random_state,
                        output_dir=args.surrogate_output_dir,
                        show_rule_summary=args.show_rule_summary,
                        rule_output_dir=args.rule_output_dir,
                    )
                )
            )
        else:
            rendered_sections.append(
                "Surrogate tree training was requested, but no exported evaluation bundle "
                "was available from this run."
            )
    if args.build_prompts:
        export_manifest_path = report.export_artifacts.get("manifest_json")
        if export_manifest_path:
            rendered_sections.append(
                "\n".join(
                    _build_prompt_dataset_from_bundle(
                        export_manifest_path,
                        scope=args.prompt_scope,
                        only_alerts=args.prompt_only_alerts,
                        top_k=args.prompt_top_k,
                        balanced=args.prompt_balanced,
                        max_samples=args.prompt_max_samples,
                        output_dir=args.prompt_output_dir,
                        surrogate_mode=args.surrogate_mode,
                        max_depth=args.surrogate_max_depth,
                        min_samples_leaf=args.surrogate_min_samples_leaf,
                        random_state=args.surrogate_random_state,
                    )
                )
            )
        else:
            rendered_sections.append(
                "Prompt dataset generation was requested, but no exported evaluation "
                "bundle was available from this run."
            )
    print("\n\n".join(rendered_sections))
    return 0
