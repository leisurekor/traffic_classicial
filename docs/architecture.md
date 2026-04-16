# Architecture Overview

## Goal

This repository is the Python mainline for an unsupervised malicious traffic detection project built around flow interaction graphs. The current revision only provides the project skeleton, typed domain schema, pipeline entrypoint, and tests needed to support incremental development.

## Module Layout

| Path | Responsibility |
| --- | --- |
| `src/traffic_graph/data/` | Unified flow schema, lightweight dataset container, and data normalization helpers. |
| `src/traffic_graph/data/preprocessing.py` | Time-window splitting, short-flow classification, short-flow aggregation, and per-window preprocessing statistics. |
| `src/traffic_graph/data/io.py` | Lightweight dataset loading helpers for CSV-backed flow samples and CLI previews. |
| `src/traffic_graph/data/pcap_loader.py` | Minimal classic-PCAP reader for Ethernet IPv4 TCP/UDP captures with parse statistics. |
| `src/traffic_graph/data/pcap_flow_builder.py` | Converts parsed PCAP packets into bidirectional `FlowRecord` objects and flow-load summaries. |
| `src/traffic_graph/data/binary_experiment.py` | Merged CSV binary experiment input construction, cleaning, and deterministic train/val/test export helpers. |
| `src/traffic_graph/data/heldout_attack_protocol.py` | Held-out attack family protocol builder for unseen-attack evaluation tasks on merged CSV datasets. |
| `src/traffic_graph/graph/` | Interaction-graph construction boundary. The current builder is intentionally simple and ready to be replaced by richer temporal or heterogeneous graph logic. |
| `src/traffic_graph/graph/graph_types.py` | Unified graph sample, endpoint node, communication edge, and graph-level statistics data structures. |
| `src/traffic_graph/graph/endpoint_graph.py` | Converts logical flow batches into per-window endpoint interaction graphs and graph summaries. |
| `src/traffic_graph/graph/association_edges.py` | Adds lightweight in-window association edges such as same-source-IP and same-destination-subnet relations. |
| `src/traffic_graph/features/` | Pure-function feature extraction for nodes and edges, including stable encodings and matrix-style feature views. |
| `src/traffic_graph/features/stats_features.py` | Base statistical features for nodes and edges derived from communication and association edges. |
| `src/traffic_graph/features/feature_pack.py` | Concatenates base and structure features, fits preprocessors, and transforms graphs into stable packed inputs. |
| `src/traffic_graph/features/normalization.py` | Field-aware standard or robust scaling that preserves configured discrete feature columns. |
| `src/traffic_graph/features/graph_tensor_view.py` | Numpy-based packed graph input objects with `edge_index`, metadata, and serialization helpers. |
| `src/traffic_graph/models/` | Abstract detector interface plus the first minimal graph autoencoder implementation. |
| `src/traffic_graph/models/encoders/graphsage_gat.py` | Lightweight GraphSAGE-style message passing encoder built directly on `edge_index`. |
| `src/traffic_graph/models/gae.py` | Minimal graph autoencoder that reconstructs node features and optionally edge features. |
| `src/traffic_graph/models/losses.py` | Weighted reconstruction loss helpers for graph autoencoder training. |
| `src/traffic_graph/models/model_types.py` | Typed model-input, model-output, and configuration helpers for packed graph tensors. |
| `src/traffic_graph/pipeline/` | Pipeline orchestration layer that assembles config, data loading, graph construction, feature extraction, and detection stages. |
| `src/traffic_graph/pipeline/binary_detection.py` | Merged CSV binary detection experiment runner, tabular reconstruction scoring, and overall / per-attack report export. |
| `src/traffic_graph/pipeline/pcap_graph_smoke.py` | Real-PCAP smoke pipeline that validates packet parsing, flow aggregation, graph construction, packing, scoring, and artifact export. |
| `src/traffic_graph/pipeline/pcap_graph_experiment.py` | Reproducible mini experiment runner for benign/malicious PCAP inputs with smoke and binary-evaluation modes. |
| `src/traffic_graph/pipeline/trainer.py` | Minimal epoch/batch trainer with checkpointing and early stopping. |
| `src/traffic_graph/pipeline/training_pipeline.py` | End-to-end training orchestration from raw flows to trained graph autoencoder checkpoints. |
| `src/traffic_graph/pipeline/checkpoint.py` | Save and load checkpoint directories with config and preprocessor state. |
| `src/traffic_graph/pipeline/persistence.py` | Persist score tables, alert records, and flattened metric summaries to local files. |
| `src/traffic_graph/pipeline/report_io.py` | Assemble timestamped run bundles and manifest files for downstream analysis. |
| `src/traffic_graph/pipeline/replay_types.py` | Typed readback views for manifests, score rows, alert rows, and replay bundles. |
| `src/traffic_graph/pipeline/replay_io.py` | Load exported run bundles back from manifest-managed files with format fallback logic. |
| `data/ciciot2023/` | Recommended user-managed home for manually downloaded raw CICIoT2023 PCAP inputs; keeps source captures separate from exported `artifacts/`. |
| `src/traffic_graph/explain/` | Pure data-organization layer that converts replayed score and alert artifacts into explanation-ready samples. |
| `src/traffic_graph/explain/explanation_types.py` | Stable dataclasses for graph-, flow-, and node-level explanation candidates. |
| `src/traffic_graph/explain/explanation_samples.py` | Sample builders, ranking helpers, balanced selectors, and JSONL export utilities for explanation candidates. |
| `src/traffic_graph/explain/surrogate_types.py` | Typed config, matrix, summary, and artifact structures for surrogate decision-tree training. |
| `src/traffic_graph/explain/surrogate_tree.py` | Scikit-learn surrogate decision-tree training, persistence, and reload helpers for explanation samples. |
| `src/traffic_graph/explain/rule_records.py` | Structured decision-path rule records, summaries, and JSONL export helpers. |
| `src/traffic_graph/explain/path_extractor.py` | Traverses fitted surrogate trees and converts sample paths into structured rule records. |
| `src/traffic_graph/explain/prompt_types.py` | Typed prompt-input structures and stable field-order constants for LLM-ready explanation datasets. |
| `src/traffic_graph/explain/prompt_builder.py` | Converts explanation samples, alerts, scores, and rules into stable prompt inputs and exported datasets. |
| `src/traffic_graph/explain/prompt_dataset.py` | Batches explanation samples into prompt datasets with scope, alert, top-k, and balanced selection controls. |
| `src/traffic_graph/explain/prompt_export.py` | Exports prompt datasets to JSONL, CSV, and a lightweight manifest for downstream consumers. |
| `src/traffic_graph/explain/prompt_replay.py` | Reloads exported prompt datasets and exposes typed replay / filtering helpers for batch LLM or manual review workflows. |
| `src/traffic_graph/explain/llm_results.py` | Typed batch-result schema, summary helpers, and stable field-order constants for prompt-to-LLM outputs. |
| `src/traffic_graph/explain/llm_runner_stub.py` | Deterministic mock runner that turns replayed prompts into placeholder LLM responses for pipeline testing. |
| `src/traffic_graph/explain/llm_result_export.py` | Exports LLM results to JSONL, CSV, summary JSON, and a lightweight manifest. |
| `src/traffic_graph/config.py` | YAML-backed configuration objects shared by scripts and future services. |
| `scripts/run_pipeline.py` | Repository-level CLI entrypoint for running or previewing the pipeline. |
| `tests/` | Fast smoke tests that validate core schema and package wiring. |

## Current Data Flow

1. Raw rows are normalized into `FlowRecord` instances.
2. `FlowDataset` stores normalized records and exposes summary or export helpers.
3. Preprocessing splits flows into fixed windows and merges short flows into logical flow records.
4. The graph builder transforms logical endpoint interactions into a windowed endpoint graph sample.
5. The feature layer extracts fixed-order base node and edge statistics from each graph.
6. The feature packer concatenates base and structure features, applies optional normalization, and emits packed numpy inputs.
7. The model layer converts packed numpy inputs into torch tensors and encodes them with a minimal graph autoencoder.
8. The training layer splits graphs into train and validation partitions, trains the graph autoencoder, and saves checkpoints.
9. Evaluation exports score tables, alert records, and metrics summaries into a manifest-managed run bundle.
10. The replay layer restores exported artifacts into typed analysis objects with stable field access.
11. The explanation layer organizes replayed rows into graph-, flow-, or node-level explanation candidates for later tree distillation or LLM prompting.
12. The real-PCAP smoke layer parses a packet capture, converts packets into flows, windows them into endpoint graphs, packs features, and optionally scores the resulting graphs with the graph autoencoder.

## Merged CSV Binary Experiment Input

The binary-experiment builder is a separate data path for public CSV datasets
such as `Merged01.csv`. It is intentionally supervised only at the evaluation
boundary: the training split remains benign-only by default, while validation
and test keep both benign and malicious rows for measuring detection quality.

### Cleaning And Label Mapping

- The builder detects a label column automatically, with an explicit override
  available when the dataset uses a nonstandard field name.
- Raw labels are mapped to a binary target with `benign=0` and
  `malicious=1`.
- `inf`, `-inf`, and `NaN` values are cleaned before export.
- The export summary records how many values were replaced and how many rows
  were dropped because their labels were missing or empty.

### Split Policy

- `train_normal_only=True` keeps the training split benign-only and applies the
  configured train ratio to the benign subset.
- Validation and test are built from the remaining benign rows plus all
  malicious rows.
- `train_normal_only=False` performs a regular train / validation / test split
  over the cleaned binary dataset.
- Both `random` and `stratified` split modes are supported, with stratified
  splitting falling back to random when a tiny class count would make
  stratification invalid.

### Export Layout

- `clean.csv` / `clean.parquet` contain the cleaned dataset with an appended
  `binary_label` column.
- `train.csv`, `val.csv`, and `test.csv` contain the deterministic splits.
- `summary.json`, `split_summary.csv`, `label_mapping.json`, and
  `manifest.json` preserve the experiment metadata and split statistics.
- The CLI entrypoint is `python3 scripts/run_pipeline.py --prepare-binary-experiment ...`.

## Held-Out Attack Evaluation Protocol

The held-out attack protocol is the second public-dataset experiment path. It
is designed to answer a different question from ordinary binary classification:
can the unsupervised detector still raise a useful anomaly score when the attack
family seen at test time was never present in the training split?

### Why This Protocol Helps

- Training is restricted to benign traffic only.
- Each test task uses a benign holdout set plus one attack family that is
  withheld from the training split.
- The model is therefore evaluated on attack-family generalization, not on
  memorizing a supervised label boundary.
- This matches the intended usage of the project: train unsupervised on normal
  traffic, then detect novel malicious behavior via anomaly scores.

### Task Construction

- Attack families can be selected by family aliases such as `Recon`, `DDoS`,
  `Mirai`, and `Web-based`, or by explicit raw labels when needed.
- A shared benign split is created once and reused across all task test sets.
- Each task exports an independent train/test bundle and a task manifest so the
  task can be consumed without reconstructing the protocol logic.
- An additional `all_malicious` task is always included when enough malicious
  samples are available.

### Export Layout

- `clean.csv` / `clean.parquet` contain the cleaned merged dataset with the
  appended binary label column.
- `tasks/<task_name>/train.csv` contains the benign-only training split.
- `tasks/<task_name>/test.csv` contains benign holdout rows plus the held-out
  attack family.
- `tasks/<task_name>/summary.json` and `tasks/<task_name>/manifest.json`
  capture per-task statistics and file references.
- The root `manifest.json` records the complete protocol configuration and all
  task manifest paths.
- The CLI entrypoint is `python3 scripts/run_pipeline.py --build-heldout-tasks ...`.

## Merged CSV Binary Detection Experiment

The binary-detection experiment is the public-dataset reporting layer that ties
the merged CSV input constructor and held-out attack protocol together with a
purely unsupervised scorer.

### Why It Still Matches The Project Goal

- Training remains benign-only by default.
- No malicious labels are used to fit the scorer.
- Labels are only used when computing ROC-AUC, PR-AUC, precision, recall, F1,
  and false-positive rate on validation or test splits.
- The held-out attack protocol checks whether the detector still flags attack
  families it never saw during training, which is the key novelty-detection
  question for this project.

### Scoring Policy

- The current merged-CSV experiment uses a PCA reconstruction baseline over the
  numeric feature columns extracted from the cleaned CSV.
- Benign training scores define the anomaly threshold by percentile.
- The overall report is computed on the binary experiment test split
  (`benign + malicious`).
- The per-attack report is computed separately for each held-out task and also
  records a per-raw-label breakdown so family-level and label-level behavior can
  be inspected side by side.

### Exported Files

- `metrics_summary.json` stores the complete report payload.
- `per_attack_metrics.csv` contains one row per held-out task.
- `overall_scores.csv` / `overall_scores.jsonl` store row-level scores for the
  binary experiment test split.
- `attack_scores.csv` / `attack_scores.jsonl` store row-level scores for all
  held-out tasks.
- `score_quantiles.json` stores training, overall-test, and per-attack quantile
  summaries that can be used for quick textual inspection.
- The report manifest records the input-artifact manifests produced by the
  binary experiment and held-out protocol builders.
- The CLI entrypoint is `python3 scripts/run_pipeline.py --run-binary-detection-experiment ...`.

## Real PCAP Graph Smoke Path

The real-PCAP smoke path validates the raw packet capture ingestion route on a
single capture such as `Recon-HostDiscovery.pcap`.

### Why It Exists

- It proves that the repository can ingest an actual packet capture rather
  than only merged CSV rows.
- It exercises the true flow-window graph chain, so the graph knobs are not
  just compatibility placeholders:
  - `window_size`
  - `use_association_edges`
  - `use_graph_structural_features`
- It remains a smoke experiment, not a full benchmark protocol.

### Processing Steps

- Parse Ethernet IPv4 TCP/UDP packets from the PCAP file.
- Aggregate packets into bidirectional `FlowRecord` objects.
- Split flows into fixed-size windows and build endpoint interaction graphs.
- Pack graph node and edge features with the same feature stack used by the
  rest of the project.
- If PyTorch is available, score the packed graphs with the graph autoencoder.
  If PyTorch is unavailable, fall back to a deterministic feature-norm scorer
  so the artifact plumbing can still be validated locally.

### Torch Runtime Verification

- Torch is now documented as an optional Graph AutoEncoder dependency rather
  than a requirement for every repository workflow.
- Install the base repository without the GAE backend:
  - `python3 -m pip install -e .`
- Install the optional GAE backend:
  - recommended in a virtual environment:
    - `python3 -m venv .venv`
    - `. .venv/bin/activate`
    - `python -m pip install -U pip`
    - `python -m pip install -e '.[gae]'`
  - if the system Python is externally managed, install `python3-venv` or
    `python3.12-venv` first
- Verify the active runtime with:
  - `python3 scripts/check_torch_env.py`
- The check script reports:
  - torch version
  - CUDA availability
  - whether real-PCAP paths will use `gae` or `deterministic_fallback`
  - whether merged-CSV graph mode is available at all

### Outputs

- A normal manifest-managed run bundle containing graph, node, edge, flow, and
  alert score tables plus metrics.
- Additional smoke metadata files:
  - `pcap_config.json`
  - `pcap_smoke_summary.json`

### Current Reduction Default

- For the real-PCAP graph experiment path, the current default graph-level
  score candidate is `hybrid_max_rank_flow_node_max`.
- `flow_p90` remains the explicit rollback / comparison option and is still the
  simplest strong baseline for the same protocol.
- This default is based on a small real-PCAP validation loop, not on a claim
  that the hybrid is universally optimal across all future protocols.

### Boundary

- The smoke path is intentionally narrow and quick to execute.
- It does not replace the merged-CSV binary detection protocol.
- It is designed to be the smallest real-data verification that the graph
  stack can run end to end.

## Reproducible PCAP Graph Experiment

The reproducible PCAP graph experiment builds on the smoke path and adds a
repeatable mini-experiment wrapper around it.

### What It Adds

- A dedicated experiment config snapshot:
  - `pcap_experiment_config.json`
- A dedicated root summary:
  - `pcap_experiment_summary.json`
- Stable root-level artifacts for quick comparison across runs:
  - `comparison_summary.csv` / `comparison_summary.json`
  - `train_graph_scores.csv` / `train_graph_scores.jsonl`
  - `overall_scores.csv` / `overall_scores.jsonl`
  - `attack_scores.csv` / `attack_scores.jsonl`
  - `per_attack_metrics.csv`
  - `graph_summary.csv` / `graph_summary.json`
  - `source_score_summary.csv` / `source_score_summary.json`
  - `split_score_summary.csv` / `split_score_summary.json`
  - `malicious_source_metrics.csv` / `malicious_source_metrics.json`
  - `score_quantiles.json`
- A single experiment entrypoint that can run in:
  - `smoke` mode
  - `binary_evaluation` mode
- A repository-level raw-data convention for manually downloaded CICIoT2023
  captures:
  - `data/ciciot2023/pcap/benign/`
  - `data/ciciot2023/pcap/malicious/recon/`
  - `data/ciciot2023/pcap/malicious/web_based/`
  - `data/ciciot2023/pcap/malicious/ddos/`
  - `data/ciciot2023/pcap/malicious/mirai/`

### Modes

- `smoke` mode is used when only one side of the protocol is provided, for
  example a malicious capture without a benign counterpart.
- `binary_evaluation` mode is used when both benign and malicious PCAP inputs
  are provided.
- When multiple benign inputs are provided, all benign graphs are merged first
  and only then split into benign-only train/validation plus benign holdout.
- When multiple malicious inputs are provided, all malicious graphs are kept
  for evaluation and also summarized by source file.
- In binary mode, training remains unsupervised:
  - train uses benign graphs only
  - evaluation uses benign holdout graphs plus malicious graphs
- The optional `pcap_experiment_label` is metadata only. It is written into the
  manifest, root summary, and comparison-ready summary but does not affect
  training, splitting, or scoring.

### Difference From Full Benchmarking

- The experiment is intentionally a mini or reproducible-smoke protocol.
- It is designed to validate the real graph chain and make configuration
  comparisons easy.
- It is not yet a full benchmark over many captures, many attack families, or
  broader protocol support.

## Current Base Features

### Node Features

| Field | Type | Meaning |
| --- | --- | --- |
| `endpoint_type` | int | Stable encoding: `client=0`, `server=1`. |
| `port` | int | Scalar port feature. For aggregated multi-port client endpoints, the minimum port is used as the stable representative value. |
| `proto` | int | Stable protocol encoding. Current built-ins: `unknown=0`, `tcp=1`, `udp=2`, `icmp=3`, `icmpv6=4`. |
| `degree_like_placeholder` | float | Placeholder for future structural features. The current value is always `0.0`. |
| `total_pkt_count` | int | Sum of packet counts over incident `communication` edges only. |
| `total_byte_count` | int | Sum of byte counts over incident `communication` edges only. |
| `total_flow_count` | int | Sum of logical `flow_count` over incident `communication` edges only. |
| `avg_pkt_count` | float | Average packet count across incident `communication` edges. |
| `avg_byte_count` | float | Average byte count across incident `communication` edges. |
| `avg_duration` | float | Average duration across incident `communication` edges. |
| `communication_edge_count` | int | Number of incident `communication` edges. |
| `association_edge_count` | int | Number of incident association edges across all enabled association types. |
| `total_degree` | int | Total number of incident graph edges across all edge types. |
| `communication_in_degree` | int | Number of incoming `communication` edges. |
| `communication_out_degree` | int | Number of outgoing `communication` edges. |
| `unique_neighbor_count` | int | Number of unique neighboring nodes across all incident edges. |

### Edge Features

| Field | Type | Meaning |
| --- | --- | --- |
| `edge_type` | int | Stable encoding: `communication=0`, `association_same_src_ip=1`, `association_same_dst_subnet=2`. |
| `pkt_count` | int | Packet count carried by the edge. |
| `byte_count` | int | Byte count carried by the edge. |
| `duration` | float | Edge duration in seconds. |
| `flow_count` | int | Number of logical flows represented by the edge. |
| `is_aggregated` | int | Boolean-like integer feature. `1` for aggregated short-flow communication edges, otherwise `0`. |

### Association Edge Defaults

- Association edges are structural links, not raw traffic observations.
- For `association_same_src_ip` and `association_same_dst_subnet`, the traffic fields `pkt_count`, `byte_count`, `duration`, and `flow_count` are set to `0`.
- `is_aggregated` is set to `0` for association edges.
- This keeps communication-edge traffic statistics distinct from association-edge structure signals.

## Packed Graph Input

- `node_features`: 2D numpy array in fixed field order.
- `edge_features`: 2D numpy array in fixed field order.
- `edge_index`: `[2, edge_count]` numpy array using node indices aligned with `node_id_to_index`.
- `node_id_to_index`: stable mapping from repository graph node ids to packed row indices.
- `edge_types`: integer-encoded edge types aligned with `edge_features`.
- `metadata`: window index, time bounds, and graph-level counts.

## Minimum Graph AutoEncoder

### Inputs

- `GraphTensorBatch` objects produced from one or more `PackedGraphInput` instances.
- Node inputs are the packed node feature matrix plus `edge_index`-driven neighborhood structure.
- Edge inputs are the packed edge feature matrix when `use_edge_features` is enabled.

### Outputs

- `node_embeddings`: latent node representations from the message-passing encoder.
- `graph_embeddings`: optional mean-pooled graph representations for each packed graph.
- `reconstructed_node_features`: decoded node feature matrix aligned with the input nodes.
- `reconstructed_edge_features`: optional decoded edge feature matrix aligned with the input edges.
- `loss_components`: optional node and edge reconstruction terms for logging or training loops.

### Why adjacency reconstruction is not included yet

- The graph topology is already determined by preprocessing and endpoint construction rules.
- Reconstructing adjacency would optimize a fixed structural matrix instead of the feature reconstruction objective that matters most in this first version.
- Keeping the objective feature-centric makes the model simpler, more stable, and easier to extend later with link prediction or auxiliary structural losses.

## Training Flow

1. Raw flow rows are loaded and normalized into `FlowRecord` instances.
2. Flows are split into fixed windows and short flows are aggregated into logical flow records.
3. Endpoint interaction graphs are built and then split into train and validation subsets.
4. The feature preprocessor is fit on the training graphs only.
5. Packed graph tensors are fed into the graph autoencoder trainer in mini-batches.
6. Per-epoch reconstruction losses are printed and stored in the training history.
7. Best and latest checkpoints are written to the configured checkpoint directory.

## Checkpoint Layout

- `state.pt`: torch-saved model state, optimizer state, and model dimensions.
- `config.json`: serialized pipeline configuration snapshot.
- `preprocessor.json`: serialized feature-preprocessor state and normalization statistics.
- `history.json`: flattened epoch history used for plots or later analysis.
- `metadata.json`: directory-local references to the other files and a concise summary.

## Normalization

- The current preprocessor supports `standard`, `robust`, and `none`.
- Continuous fields are normalized per column after fitting on a batch of graphs.
- Discrete fields are excluded from scaling through configuration masks.
- Default excluded node fields: `endpoint_type`, `port`, `proto`.
- Default excluded edge fields: `edge_type`, `is_aggregated`.

## Alert Record Schema

Alert records are derived from anomaly scores after evaluation. They are not used
for training.

| Field | Type | Meaning |
| --- | --- | --- |
| `alert_id` | str | Stable identifier built from scope, graph, window, and entity ids. |
| `alert_level` | str | Simple severity label: `low`, `medium`, or `high`. |
| `alert_scope` | str | Alert granularity: `graph`, `node`, `edge`, or `flow`. |
| `graph_id` | int / str / null | Graph identifier copied from the source score row. |
| `window_id` | int / str / null | Window identifier copied from the source score row. |
| `node_id` | int / str / null | Present for node alerts only. |
| `edge_id` | int / str / null | Present for edge alerts only. |
| `flow_id` | int / str / null | Present for flow alerts only. |
| `anomaly_score` | float | Reconstruction-error-derived anomaly score. |
| `threshold` | float | Alert threshold used for the positive decision. |
| `is_alert` | bool | `true` when `anomaly_score >= threshold`. |
| `label` | any / null | Evaluation-time label copied from the score table when available. |
| `metadata` | dict | Compact source-row metadata retained for downstream explanation or export. |

### Alert Level Rule

- `is_alert` is determined by `anomaly_score >= threshold`.
- `alert_level` is derived from the score-to-threshold ratio.
- The default configuration uses `medium_multiplier=1.5` and `high_multiplier=2.0`.
- The current implementation is intentionally simple and can be replaced by a richer policy later.

## Export Bundle Layout

Persisted analysis outputs are organized under a run-specific directory:

```text
<export_dir>/<run_id>/<timestamp>/
  manifest.json
  scores/
    graph_scores.eval.csv
    graph_scores.eval.jsonl
    graph_scores.eval.parquet
    node_scores.eval.csv
    ...
  alerts/
    alert_records.eval.csv
    alert_records.eval.jsonl
    alert_records.eval.parquet
  metrics/
    metrics_summary.eval.json
    metrics_summary.eval.csv
    metrics_summary.eval.jsonl
    metrics_summary.eval.parquet
```

- `parquet` files are written only when both `pandas` and `pyarrow` are available.
- When parquet support is missing, the exporter keeps the `jsonl` and `csv` files and records a skip note instead of failing.

### Score Table Fields

| Field | Meaning |
| --- | --- |
| `run_id` | Stable run identifier, typically the pipeline run name. |
| `timestamp` | UTC timestamp token used in the run bundle path and row metadata. |
| `split` | Stage label such as `train`, `val`, `eval`, or `test`. |
| `score_scope` | One of `graph`, `node`, `edge`, or `flow`. |
| `graph_id` | Canonical graph identifier. |
| `window_id` | Canonical time-window identifier. |
| `node_id` | Present for node scores. |
| `edge_id` | Present for edge scores. |
| `flow_id` | Present for flow scores. |
| `anomaly_score` | Reconstruction-error-derived score. |
| `threshold` | Threshold used to compute the optional `is_alert` flag. |
| `is_alert` | Boolean alert decision when a threshold is available. |
| `label` | Evaluation-time label when available. |
| `metadata` | JSON string with the remaining source-row fields. |

### Alert Record Fields

Alert exports reuse the same run metadata and primary-key fields, with `alert_id`, `alert_level`, and `alert_scope` added on top of the common score fields. The `metadata` field remains a JSON string so the records can be written consistently to CSV, JSONL, and Parquet.

### Metrics Summary Fields

The flattened metrics table uses `run_id`, `timestamp`, `split`, `scope`, `metric_path`, and `metric_value`. The nested JSON file preserves the original scope-by-scope metrics dictionary for direct inspection or plotting.

## Export Bundle Replay Interface

- `load_export_bundle(path)` accepts either a bundle directory or a direct `manifest.json` path.
- Score tables are read with the preference order `parquet -> jsonl -> csv` when parquet support is available, and `jsonl -> csv` otherwise.
- Metrics summaries prefer the nested JSON file when present, and can also be rebuilt from flattened `jsonl`, `csv`, or `parquet` files.
- The replay layer restores score rows and alert rows into typed dataclasses so downstream explanation code can use stable attribute access instead of raw dictionaries.
- `metadata` is parsed back into dictionaries during replay, so downstream consumers do not need to decode JSON strings manually.
- Helper interfaces include `list_available_tables(bundle)`, `get_score_table(bundle, scope)`, `get_alert_records(bundle, only_positive=True)`, and `get_metrics_summary(bundle)`.
- The CLI supports a lightweight bundle check through `python3 scripts/run_pipeline.py --replay-bundle <bundle_dir_or_manifest>`.

## Explanation Sample Schema

Explanation-ready samples are built only from replayed bundle artifacts. This keeps
the explanation bridge decoupled from training and evaluation internals.

### Explanation Sample Fields

| Field | Type | Meaning |
| --- | --- | --- |
| `sample_id` | str | Stable identifier built from `scope`, `graph_id`, `window_id`, and the scope-specific entity id. |
| `scope` | str | Explanation granularity: `graph`, `flow`, or `node`. |
| `run_id` | str | Source run identifier copied from the replay bundle. |
| `graph_id` | int / str / null | Graph identifier for the source score row. |
| `window_id` | int / str / null | Time-window identifier for the source score row. |
| `flow_id` | int / str / null | Present for flow-level explanation samples. |
| `node_id` | int / str / null | Present for node-level explanation samples. |
| `anomaly_score` | float | Score copied from the replayed score table. |
| `threshold` | float / null | Alert threshold when available. |
| `is_alert` | bool / null | Alert decision copied from the score table or aligned alert record. |
| `alert_level` | str / null | Severity copied from the aligned alert record when one exists. |
| `label` | any / null | Evaluation-time label when present. Training never uses this field. |
| `stats_summary` | dict | Scope-specific summary fields retained from replayed score metadata. |
| `graph_summary` | dict | Graph-level context keyed by `graph_id` and `window_id`, including graph counts and graph anomaly context when available. |
| `feature_summary` | dict | Best-effort feature-field summary. The current bundle usually does not persist feature names, so this may be an explicit placeholder. |
| `metadata` | dict | Full replayed metadata payload preserved for downstream explanation modules. |

### Scope-Specific Summary Behavior

- Graph-level samples keep graph size and edge-count fields in both `stats_summary` and `graph_summary`.
- Flow-level samples keep flow-centric fields such as `pkt_count`, `byte_count`, `duration`, `flow_count`, and endpoint/protocol hints when they are present in the bundle metadata.
- Node-level samples keep endpoint and incident-traffic aggregates such as `endpoint_type`, `port`, `proto`, `total_pkt_count`, `total_flow_count`, and degree-related counters when they are present in the bundle metadata.

### Sample Selection Helpers

- `build_explanation_samples(bundle, scope="flow", only_alerts=True, top_k=...)` restores explanation candidates directly from a replay bundle.
- `sort_samples_by_score(...)` returns samples in descending anomaly-score order.
- `select_top_alert_samples(...)` keeps the top-k positive alert samples only.
- `select_balanced_samples_for_explanation(...)` returns a roughly balanced alert / non-alert subset when both groups are available, and otherwise falls back to highest-score samples.
- `export_explanation_candidates(samples, path)` writes a stable `jsonl` file for downstream tree or LLM pipelines without changing the original run bundle format.

### CLI Support

- Replay-mode summary: `python3 scripts/run_pipeline.py --replay-bundle <bundle_dir_or_manifest> --show-explanation-summary`
- Optional scope override: `--explanation-scope graph|flow|node`
- Optional cap on returned candidates: `--explanation-top-k 50`
- When `--eval --show-explanation-summary` is used, the runner first exports the evaluation bundle and then derives the explanation summary from the replayed bundle.

## Surrogate Decision Tree

The surrogate decision tree is a lightweight student model trained on explanation-ready
samples. It is not part of the main detector and it never replaces the graph autoencoder.

### Role In The Pipeline

- The main detector still produces anomaly scores from reconstruction error.
- The surrogate tree learns a human-readable approximation of those scores or of the derived alert decisions.
- The resulting tree is intended for rule inspection, path extraction, and later explanation-layer tooling.

### Training Modes

- `regression`: fits `anomaly_score` directly and is the default mode.
- `classification`: fits pseudo labels derived from `is_alert` or `threshold`.

### Feature Policy

- Training only consumes structured numeric fields from explanation-ready samples.
- Free text and label fields are excluded from the feature matrix.
- The feature-name order is stable and recorded in the saved artifact metadata.

### Label Boundary

- Real labels are never used to train the surrogate tree.
- If a label is present in the explanation sample, it can still be preserved in the sample metadata for later analysis, but it is not used as a training target.

## Rule Record Schema

Rule records are derived by traversing a fitted surrogate tree for each explanation sample.
They are intended for downstream prompt builders and human-readable rule inspection.

### Rule Record Fields

| Field | Type | Meaning |
| --- | --- | --- |
| `rule_id` | str | Stable rule identifier derived from `sample_id`, tree mode, and leaf id. |
| `sample_id` | str | Source explanation sample identifier. |
| `scope` | str | Explanation scope copied from the sample: `graph`, `flow`, or `node`. |
| `tree_mode` | str | Surrogate-tree mode: `regression` or `classification`. |
| `predicted_score_or_class` | float / int | Tree prediction at the leaf. Regression returns a score; classification returns the pseudo label. |
| `leaf_id` | int | Leaf node index returned by the fitted tree. |
| `path_conditions` | list | Ordered list of root-to-leaf path conditions. |
| `feature_names_used` | list[str] | Ordered unique feature names encountered along the path. |

### Path Condition Fields

| Field | Type | Meaning |
| --- | --- | --- |
| `feature_name` | str | Stable feature key taken from the surrogate-tree feature matrix, such as `stats_summary.pkt_count`. |
| `operator` | str | Branch operator used at the tree node: `<=` for the left child and `>` for the right child. |
| `threshold` | float | Numeric split threshold stored in the tree node. |
| `sample_value` | float / null | The sample's numeric value for the feature at that decision point. |
| `tree_node_index` | int / null | Internal tree node id for the condition. |

### Path-to-Feature Mapping

- `path_extractor.py` resolves each `feature_name` against the explanation sample's structured fields.
- The current feature source order matches surrogate-tree training: `stats_summary`, then `graph_summary`, then `feature_summary`.
- The resulting path conditions are emitted in traversal order from root to leaf, which keeps later prompt building deterministic.

### CLI Support

- The surrogate-tree CLI can now emit rule records with `python3 scripts/run_pipeline.py --replay-bundle <bundle_dir_or_manifest> --train-surrogate-tree --show-rule-summary`.
- Add `--rule-output-dir <dir_or_jsonl>` to persist `rule_records.jsonl` alongside the summary output.

## Prompt Input Schema

Prompt inputs are the last structured step before a future LLM call. The prompt
builder only assembles a stable data object and rendered text. It does not call
any external model or SDK.

### Prompt Input Fields

| Field | Type | Meaning |
| --- | --- | --- |
| `prompt_id` | str | Stable prompt identifier derived from the sample id, tree mode, and leaf id. |
| `run_id` | str | Source run identifier. |
| `sample_id` | str | Explanation sample identifier. |
| `scope` | str | Prompt scope: `graph`, `flow`, or `node`. |
| `anomaly_score` | float | Score copied from the sample or score table. |
| `threshold` | float / null | Threshold used for alerting, when available. |
| `is_alert` | bool / null | Alert decision, when available. |
| `alert_level` | str / null | Alert severity from the aligned alert record or sample. |
| `label` | any / null | Optional evaluation-time label; retained for audit but not required for prompt reasoning. |
| `structured_context` | dict | Sample identity, statistics, graph context, and score/alert summaries. |
| `rule_summary` | dict | Surrogate-tree mode, leaf prediction, and ordered path conditions. |
| `prompt_text` | str | Stable template text that can be handed directly to an LLM prompt runner. |

### Prompt Builder Boundary

- `prompt_builder.py` only transforms structured artifacts into prompt-ready inputs.
- It deliberately does not import or wrap any OpenAI SDK, HTTP client, or execution layer.
- A future LLM service should consume `PromptInput` objects or exported `jsonl` rows and handle model selection, retries, streaming, and response parsing independently.

### Prompt Template Intent

- Ask the model to explain why the sample is anomalous.
- Ask the model to identify the most important rule evidence.
- Ask the model to suggest a plausible malicious behavior or benign explanation.
- Ask the model to recommend follow-up checks.

## Prompt Dataset Export

The prompt dataset layer turns explanation-ready samples and surrogate-tree
rules into batchable prompt inputs for later LLM calls or manual review.

Prompt datasets are written under `<output_dir>/<run_id>/<timestamp>/` so that
multiple runs can coexist without collisions.

### Exported Files

- `manifest.json`: selection parameters, counts, field order, and file paths.
- `prompt_inputs.jsonl`: one JSON object per prompt input, preserving the stable
  field order from `PROMPT_INPUT_FIELDS`.
- `prompt_inputs.csv`: a tabular view with the same field order; nested context
  fields are serialized as JSON strings.

### Selection Logic

- `build_prompt_dataset(..., scope=...)` filters to one explanation scope.
- `only_alerts=True` keeps only alert samples before optional top-k slicing.
- `top_k` keeps the highest-scoring prompts after deterministic sorting.
- `balanced=True` selects a roughly even alert / non-alert subset for manual
  review, then optionally applies `top_k`.

### Downstream Consumption

- LLM workers should read `prompt_inputs.jsonl` or `prompt_inputs.csv` and treat
  each row as a self-contained prompt request.
- The prompt builder remains SDK-agnostic; it only prepares data and rendered
  prompt text.
- The export manifest is intended for discovery and reproducible replay, not for
  execution.

## Prompt Dataset Replay

The replay layer loads exported prompt datasets back into typed Python objects
for batch processing, manual review, or offline prompt analysis.

### Replay Objects

- `PromptDatasetReplay` stores the run metadata, manifest view, loaded prompt
  records, and the selection summary.
- `PromptDatasetManifestInfo` preserves the manifest path, selection settings,
  artifact paths, row counts, and raw manifest payload.
- `PromptInput` remains the record type for downstream consumers.

### Replay Behavior

- `load_prompt_dataset(<run_dir_or_manifest>)` reads `manifest.json` first and
  then loads `prompt_inputs.jsonl` when available.
- If JSONL is missing or fails to parse, the loader falls back to
  `prompt_inputs.csv`.
- `filter_prompt_records(...)` supports scope filtering, alert filtering, and
  score-based top-k selection for downstream prompt batching.
- `--replay-prompts <path>` on the CLI prints a compact summary of a prompt
  dataset without invoking any LLM.
- Replaying from a CSV-only dataset is supported for environments where JSONL
  artifacts were removed or regenerated independently.

## LLM Result Schema And Stub Runner

The repository intentionally ships with a mock LLM execution layer instead of a
real SDK integration. This keeps the prompt-to-response path testable while
preserving a schema that a future OpenAI or other vendor runner can reuse
without changing downstream code.

### Result Schema

| Field | Type | Meaning |
| --- | --- | --- |
| `response_id` | str | Stable response identifier derived from the model name and prompt id. |
| `prompt_id` | str | Source prompt identifier from the replayed prompt dataset. |
| `run_id` | str | Source run identifier copied from the prompt dataset. |
| `model_name` | str | Recorded model name, such as `mock-llm-stub`. |
| `response_text` | str | Human-readable response body emitted by the runner. |
| `raw_response` | dict / null | Structured payload retained for later vendor-specific integrations. |
| `status` | str | Batch status: `success`, `failed`, or `skipped`. |
| `error_message` | str / null | Optional failure reason. |
| `created_at` | str | UTC timestamp token for the batch. |

### Stub Runner Behavior

- `run_llm_stub(prompt_dataset_replay, ...)` consumes a replayed prompt
  dataset and emits one placeholder response per prompt record.
- The stub response text intentionally includes the prompt id, run id, model
  name, anomaly score, and a compact prompt excerpt so tests can assert on
  deterministic content.
- The exported artifact uses the same high-level shape that a future real LLM
  runner should keep: result records, summary, manifest, and run-specific
  directory layout.
- Exported files currently include `results.jsonl`, `results.csv`,
  `summary.json`, and `manifest.json` under
  `<output_dir>/<run_id>/<timestamp>/<model_name>/`.
- No external network calls or SDK integrations are performed in this stage.
- A future vendor-backed runner should reuse `LLMResultRecord`,
  `LLMResultSummary`, and `LLMResultArtifact` so prompt processing, replay, and
  downstream review tools do not need to change.

## Extension Guidance

- Add protocol-specific parsers and dataset readers under `src/traffic_graph/data/`.
- Extend `src/traffic_graph/data/preprocessing.py` when you need richer short-flow rules, session stitching, or bidirectional flow consolidation.
- Add graph enrichment, temporal windows, node/edge features, and message-passing utilities under `src/traffic_graph/graph/`.
- Extend `src/traffic_graph/features/` for structural graph features, normalization, and future export adapters.
- Add self-supervised or unsupervised detectors under `src/traffic_graph/models/` by implementing the abstract detector contract.
- Keep `scripts/` thin and move reusable logic into `src/` modules so tests remain fast and isolated.
## Anomaly Scoring And Evaluation
- `src/traffic_graph/models/scoring.py` converts reconstruction error into
  node, edge, graph, and flow anomaly scores.
- `src/traffic_graph/pipeline/metrics.py` computes ROC-AUC, PR-AUC, precision,
  recall, and F1 from labeled score tables.
- `src/traffic_graph/pipeline/eval_pipeline.py` loads a checkpoint, rebuilds
  graphs with the saved preprocessing metadata, and exports JSON / CSV score
  reports.
- Labels are never used during training. Evaluation reads labels only through
  `evaluation_label_field`, and only for the test / scoring stage.
- Graph-level scores are reduced from node scores with `mean` or `max`.
- Flow-level scores are taken from communication edges, which makes the score
  table easy to align with logical flows.

## Score Tables
| Table | Key Fields | Notes |
| --- | --- | --- |
| Node scores | `graph_index`, `node_id`, `endpoint_type`, `node_anomaly_score` | Row-wise node reconstruction error. |
| Edge scores | `graph_index`, `edge_id`, `edge_type`, `edge_anomaly_score` | Includes communication and association edges. |
| Flow scores | `graph_index`, `logical_flow_id`, `flow_anomaly_score` | Communication edges only. |
| Graph scores | `graph_index`, `window_index`, `graph_anomaly_score`, `graph_label` | Used for ROC-AUC / PR-AUC / threshold metrics. |
## Reproducible PCAP Graph Experiment

The repository now exposes a small, reproducible real-PCAP experiment line on
top of the existing smoke runner. This path is intentionally narrower than a
full benchmark: it focuses on validating the real packet -> flow -> time-window
graph -> graph scoring chain while keeping artifacts stable enough for later
comparison scripts.

Current semantics:

- `--run-pcap-graph-experiment` supports one or more benign PCAP inputs and one
  or more malicious PCAP inputs.
- In `binary_evaluation` mode, all benign graphs from all benign inputs are
  merged first, then split into benign-only train/validation and benign holdout
  evaluation graphs using `pcap_benign_train_ratio`.
- Malicious graphs are never used for training. They are only scored during
  evaluation.
- In `smoke` mode, labels are not required; the experiment only validates that
  the real graph pipeline can run end to end and export replayable artifacts.
- When PyTorch is unavailable, the experiment automatically reuses the existing
  deterministic fallback scorer instead of the Graph AutoEncoder.

Additional analysis-friendly outputs:

- `source_score_summary.json` / `source_score_summary.csv`
  per-input-source graph score summaries with split counts.
- `split_score_summary.json` / `split_score_summary.csv`
  quantile summaries for `benign_train_reference`, `benign_test`,
  `malicious_test`, and `overall_test` when available.
- `malicious_source_metrics.json` / `malicious_source_metrics.csv`
  source-level benign-vs-one-malicious-source evaluation rows so downstream
  scripts can quickly answer which malicious PCAP is hardest to detect.

This mini experiment differs from the older smoke-only path in one important
way: it treats multiple benign and malicious PCAP inputs as named experiment
sources and records their per-source packet/flow/graph counts, split
assignments, and graph-score summaries in the root experiment summary.
