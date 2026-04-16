# Experiments

This repository currently supports two public-dataset experiment tracks for the
merged CIC IoT 2023 CSV release.

## 1. Binary Experiment Input Construction

- Input: `Merged01.csv`.
- Goal: build an unsupervised binary detection dataset with `benign=0` and
  `malicious=1`.
- Training split: benign-only by default.
- Validation / test splits: benign + malicious.
- This track is useful for smoke-testing the detector with a clean binary
  boundary and for verifying that the data cleaning and split logic are stable.

## 2. Held-Out Attack Evaluation

- Input: `Merged01.csv`.
- Goal: verify whether the detector can identify attack families that were not
  present in the training split.
- Training split: benign-only.
- Test split: benign holdout + one selected attack family.
- This track is more informative for novelty detection because it tests whether
  the detector generalizes to unseen malicious behavior rather than memorizing a
  fixed attack label.

## 3. Binary Detection Report

- The merged-CSV binary detection report combines the two tracks above.
- It fits a benign-only unsupervised scorer on the binary experiment training
  split and then evaluates:
  - `overall` binary metrics on the binary experiment test split.
  - `per-attack` metrics on each held-out attack task.
- Why this still matches the project goal:
  - labels are never used in training,
  - only benign traffic is used to fit the anomaly model,
  - evaluation uses labels only to compute metrics,
  - held-out tasks explicitly measure unseen-attack detection.

## 4. Graph Backend

- The graph backend reuses the exact same binary experiment and held-out attack
  protocol as the tabular baseline.
- Only the scoring backend changes:
  - tabular mode uses a PCA reconstruction scorer on numeric columns,
  - graph mode uses the Graph AutoEncoder mainline.
- For `Merged01.csv`, graph mode adapts each cleaned row into a tiny one-node
  graph so the merged-CSV protocol stays identical while the graph model can be
  exercised end to end.
- Graph scores are reduced to the sample level with a fixed priority order:
  `flow -> edge -> node -> graph`.
  - This matters for future raw-flow experiments where a single sample may have
    many node, edge, or flow scores.
  - In the current merged-CSV graph backend the adapter emits one graph score
    per sample, so the reduction is effectively an identity map.
- In graph mode, `model_n_components` in the exported report stores the GAE
  `latent_dim`, so the report schema stays compatible with the tabular
  baseline while still exposing the graph model capacity.
- This means the protocol comparison is fair:
  - the same train / test rows are used,
  - benign-only training is preserved,
  - labels are still used only for evaluation,
  - the exported metrics and score tables stay schema-compatible.
- When the project later moves to raw flow inputs, the graph backend can be
  swapped from the row-as-graph adapter to the real windowed flow graph builder
  without changing the experiment protocol or report format.
- The graph-mode report also prints a score distribution summary so we can see
  whether recon / web-based scores separate cleanly from benign traffic.

### Torch Runtime Selection

- The repository now treats torch as an optional Graph AutoEncoder dependency.
- Base installation can stay lightweight:
  - `python3 -m pip install -e .`
- To enable the current GAE path, install the optional extra:
  - recommended in a virtual environment:
    - `python3 -m venv .venv`
    - `. .venv/bin/activate`
    - `python -m pip install -U pip`
    - `python -m pip install -e '.[gae]'`
  - if your OS Python is externally managed, install the `python3-venv` or
    `python3.12-venv` package first
- Verify the runtime before training or real-PCAP experiments:
  - `python3 scripts/check_torch_env.py`
- Expected behavior:
  - without torch:
    - real-PCAP smoke and reproducible PCAP experiment use the deterministic
      fallback scorer
    - merged-CSV `model_mode=graph` remains unavailable
  - with torch:
    - real-PCAP smoke and reproducible PCAP experiment use the existing Graph
      AutoEncoder stack
    - merged-CSV `model_mode=graph` is enabled
- This is a backend switch only. It does not guarantee better results by
  itself.

## 4.5 Real PCAP Smoke Path

- Input: `Recon-HostDiscovery.pcap`.
- Goal: verify the end-to-end packet-to-flow-to-graph chain on a real capture.
- This is a smoke experiment only; it is not a benchmark protocol.
- The path is useful for checking that the graph knobs actually affect a real
  flow-window topology:
  - `window_size`
  - `use_association_edges`
  - `use_graph_structural_features`
- If PyTorch is unavailable, the smoke path falls back to a deterministic
  feature-norm scorer so the artifact plumbing can still be validated locally.
- The output bundle stays compatible with the existing run-bundle schema and
  simply adds two smoke metadata files:
  - `pcap_config.json`
  - `pcap_smoke_summary.json`

## 4.6 Reproducible PCAP Graph Experiment

- Input:
  - one or more `benign` PCAP files,
  - one or more `malicious` PCAP files,
  - or a single-sided PCAP input for smoke-only validation.
- Goal: turn the real-PCAP graph path into a small, repeatable experiment unit
  that can be rerun with different graph settings and compared by artifacts.

### Modes

- `smoke` mode:
  - used when only one side of the protocol is available,
  - validates that the real graph path runs end to end,
  - still exports stable score tables, graph summaries, config, and summary
    artifacts.
- `binary_evaluation` mode:
  - used when both benign and malicious PCAP inputs are provided,
  - trains on benign graphs only,
  - evaluates on benign holdout graphs plus malicious graphs,
  - exports `overall_scores`, `attack_scores`, `per_attack_metrics`, and
    `metrics_summary` in a comparison-friendly layout.

### Why It Exists

- The single-PCAP smoke entrypoint is useful for validating the chain once.
- The reproducible experiment entrypoint is useful when we want to rerun the
  same real-input graph path under different settings and compare:
  - `window_size`
  - `use_association_edges`
  - `use_graph_structural_features`
  - packet limits and thresholds

### What It Is Not

- It is not a full benchmark over the full CIC IoT PCAP corpus.
- It does not yet claim broad coverage across many benign and malicious
  captures.
- It is intentionally the smallest protocol that keeps:
  - real packet parsing,
  - real flow-window graph construction,
  - unsupervised benign-only training in binary mode,
  - stable artifact export for later comparison.

### Current Graph Score Candidate

- The current graph-mode default candidate is:
  - `hybrid_max_rank_flow_node_max`
- The repository still keeps explicit alternatives for controlled comparison:
  - `flow_p90`
  - `node_max`
  - `mean_node`
- Why the default candidate changed:
  - the older `mean_node` reduction washed out sparse high-anomaly tails,
  - `flow_p90` recovered a large part of the lost signal on real PCAP runs,
  - the thin hybrid over `flow_p90` and `node_max` then improved the
    representative real-PCAP protocol again without increasing overall FPR.
- This should be read as:
  - the current best default candidate after a small real-PCAP validation loop,
  - not a claim that the hybrid is globally optimal for every future protocol.

### How To Switch The Scorer

- Use `--pcap-graph-score-reduction` to select the graph-level reduction:
  - `hybrid_max_rank_flow_node_max`
  - `flow_p90`
  - `node_max`
  - `mean_node`
- If you do not pass the flag, graph mode now defaults to
  `hybrid_max_rank_flow_node_max`.
- `flow_p90` remains the recommended rollback / ablation baseline because it is
  simpler and still strong on the same real-PCAP protocol.

### Current Scorer Roles

- `hybrid_max_rank_flow_node_max`
  - current default candidate for graph mode
  - use it for the mainline real-PCAP graph experiments when you want the best
    current balance between overall detection, `Recon`, and
    `BrowserHijacking`
- `flow_p90`
  - current fallback / rollback scorer
  - use it when you want the simplest strong baseline with the clearest
    interpretation and the smallest scoring-layer complexity
- `decision_topk_flow_node`
  - current experimental scorer
  - use it when the experiment is explicitly `Recon`-first and you are willing
    to trade away some `BrowserHijacking` performance
- Why `decision_topk_flow_node` is not the default:
  - it repeatedly shows a `Recon`-leaning gain,
  - but the same narrow follow-up runs also show a recurring
    `BrowserHijacking` penalty,
  - so it is more useful as an opt-in experimental candidate than as the
    repository default.
- `hybrid_decision_tail_balance`
  - keep only as a recorded experimental reducer
  - the fixed weak-anomaly snapshot did not improve `BrowserHijacking`,
    `MITM-ArpSpoofing`, or `DictionaryBruteForce`
  - it also degraded `Backdoor_Malware`, so it is not suitable for default,
    fallback, or near-term follow-up promotion
  - later default-hybrid refresh work also failed to produce a meaningful weak-
    anomaly gain, so this reducer should remain frozen as a failed exploration
    rather than a branch to reopen

### Unified Scorer Summary Entry

- Use `scripts/summarize_graph_scorers.py` as the unified analysis entry for the
  current graph scorer family.
- It reuses the existing scorer-family summarizer and writes both:
  - the broader paper-inspired scorer comparison,
  - and the smaller `graph_scorer_family_summary.csv` /
    `graph_scorer_family_summary.md` role summary.
- The role labels are intentionally stable:
  - `default_candidate`
  - `fallback`
  - `experimental`
- This keeps the comparison / replay / analysis layer on one shared scorer-role
  vocabulary instead of repeating ad-hoc mappings in multiple scripts.

### What To Read First

- For one formal real-PCAP graph run:
  - `comparison_summary.json`
  - `per_attack_metrics.csv`
  - `train_graph_scores.csv`
  - `source_score_summary.csv`
- For the current reduction follow-up:
  - `artifacts/ciciot2023/analysis/graph_reduction_mainline_hybrid_ab.csv`
  - `artifacts/ciciot2023/analysis/graph_vs_tabular_hybrid_followup.csv`
  - `artifacts/ciciot2023/analysis/graph_reduction_stability_check.csv`
  - `artifacts/ciciot2023/analysis/graph_scorer_family_summary.csv`
- These files are the fastest way to check:
  - which reduction was used,
  - whether benign-train q95 thresholding stayed stable,
  - how `Recon` and `BrowserHijacking` moved under the current scorer choice,
  - and which scorer is currently treated as the default / fallback /
    experimental option.

## Metric Interpretation

- `ROC-AUC` and `PR-AUC` describe ranking quality before thresholding.
- `Precision`, `Recall`, `F1`, and `False Positive Rate` describe the chosen
  alert threshold.
- Per-attack recall shows which attack families are easiest for the detector to
  flag.
- The quantile summaries help explain whether a task's attack scores separate
  cleanly from benign scores even when thresholded metrics are modest.

## Analysis Notes

- For the shortest current guide to scorer roles, summary commands, and the
  most useful analysis artifacts, start with:
  - [analysis_notes.md](/home/xdo/traffic_classicial/docs/analysis_notes.md)
- That note also records the failed `hybrid_decision_tail_balance`
  experimental reducer so later readers do not reopen that branch without a
  genuinely new idea.

## 5. Model Comparison

- The comparison report reads two exported binary-detection runs and compares
  them under the same protocol.
- This is the fairest way to judge the tabular baseline versus the graph
  backend because:
  - both runs use the same `Merged01.csv` split logic,
  - both runs train without malicious labels,
  - both runs export the same metric schema and held-out attack tasks.
- The comparison report highlights:
  - overall metric deltas for `ROC-AUC`, `PR-AUC`, `Precision`, `Recall`,
    `F1`, and `False Positive Rate`,
  - per-attack deltas for `recon`, `web-based`, `all_malicious`, and any other
    shared attack tasks,
  - score-median and score-q95 deltas for the highlighted attack families so it
    is easy to see whether graph scores separate better even when thresholded
    metrics are close.
- The generated files are:
  - `comparison_summary.json`
  - `comparison_overall.csv`
  - `comparison_per_attack.csv`
  - optional `comparison_report.md`
- If graph mode improves recall or PR-AUC on `recon` or `web-based`, that is a
  strong signal that the graph representation is capturing behavior patterns
  missed by the tabular baseline.

## 6. Graph Ablation Suite

- The graph ablation suite sweeps three graph-design knobs:
  - `association edges` on/off,
  - `window_size` over `30s`, `60s`, `120s`, `300s`,
  - `graph structural features` on/off.
- The purpose is not to add a new detector family, but to isolate which graph
  design choices matter most for the difficult attack families:
  - `recon`,
  - `web-based`.
- Each configuration is exported into its own run directory and the suite writes
  a combined `ablation_summary.csv` at the suite level.
- The summary table always includes:
  - overall `ROC-AUC`, `PR-AUC`, `Precision`, `Recall`, `F1`, `False Positive Rate`,
  - `recon` recall / PR-AUC / F1,
  - `web-based` recall / PR-AUC / F1,
  - `all_malicious` recall / PR-AUC / F1.
- Important implementation note:
  - the current public `Merged01.csv` graph backend still uses a compatibility
    adapter because the dataset does not expose raw endpoint/window topology,
    so `window_size` and association-edge settings are recorded and propagated
    through the suite for forward compatibility,
  - graph structural features are genuinely toggleable and are packed into the
    model input when enabled,
  - the same suite can later be pointed at a true flow-window graph source
    without changing the experiment protocol or output schema.
- This makes the ablation useful in two ways:
  - on the current merged CSV, it validates the end-to-end plumbing and the
    effect of structural feature inclusion,
  - on a true raw-flow graph source, it can measure the actual contribution of
    association edges and window size to `recon` / `web-based` detection.
## Reproducible PCAP Mini Experiment

The real-PCAP graph experiment is still a mini experiment, not a full benchmark.
Its job is to make small, repeatable benign-vs-malicious graph runs easy to
launch and compare while preserving the project rule that training remains
fully unsupervised.

Protocol summary:

- benign PCAPs provide the only training data
- malicious PCAPs are evaluation-only
- all benign graphs across all benign inputs are merged before the benign split
- all malicious graphs across all malicious inputs are kept for evaluation
- if only one side is provided, the runner falls back to smoke validation mode

Recommended raw-data layout:

```text
data/ciciot2023/pcap/
  benign/
  malicious/
    recon/
    web_based/
    ddos/
    mirai/
```

The repository-level guide for that layout is:

- `data/ciciot2023/README.md`

Recommended command shape:

```bash
python3 scripts/run_pipeline.py \
  --run-pcap-graph-experiment \
  --pcap-benign-input data/ciciot2023/pcap/benign/home_idle_control.pcap \
  --pcap-malicious-input data/ciciot2023/pcap/malicious/recon/Recon-HostDiscovery.pcap \
  --pcap-malicious-input data/ciciot2023/pcap/malicious/web_based/Web-CommandInjection.pcap \
  --pcap-experiment-label ciciot2023_real_control_small \
  --pcap-output-dir artifacts/pcap_graph_experiments \
  --pcap-packet-limit 5000 \
  --pcap-window-size 60 \
  --pcap-threshold-percentile 95
```

Important caveat:

- using the same PCAP file as both benign and malicious is acceptable only for
  chain validation,
- it should not be interpreted as a real detection result.

Recommended interpretation:

- use this path to validate that graph knobs such as `window_size`,
  `use_association_edges`, and `use_graph_structural_features` are actually
  affecting real PCAP-derived graphs
- run `python3 scripts/check_torch_env.py` first so you know whether the
  experiment will use GAE or the deterministic fallback scorer
- use the new source-level summaries to compare which malicious input file is
  easiest or hardest to detect
- use `comparison_summary.json/csv` when you want one row per run for later
  cross-run aggregation
- use `train_graph_scores.jsonl/csv` when you want to inspect the benign
  reference score distribution that produced the anomaly threshold
- do not overstate these runs as a full public benchmark, especially when the
  environment is using the deterministic fallback scorer because `torch` is not
  installed

New comparison-friendly artifacts:

- `pcap_experiment_summary.json`
- `graph_summary.json` / `graph_summary.csv`
- `comparison_summary.json` / `comparison_summary.csv`
- `train_graph_scores.jsonl` / `train_graph_scores.csv`
- `source_score_summary.json` / `source_score_summary.csv`
- `split_score_summary.json` / `split_score_summary.csv`
- `malicious_source_metrics.json` / `malicious_source_metrics.csv`
- `metrics_summary.json`
- `overall_scores.csv` / `overall_scores.jsonl`
- `attack_scores.csv` / `attack_scores.jsonl`

Next recommended step:

- add a real benign control PCAP and at least two distinct malicious PCAPs so
  the source-level metrics start reflecting meaningful detection differences
  instead of pure pipeline validation.
