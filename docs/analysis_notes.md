# Analysis Notes

Current graph scorer family roles:

- `hybrid_max_rank_flow_node_max` = default candidate
- `flow_p90` = fallback / rollback baseline
- `decision_topk_flow_node` = experimental `Recon`-first candidate
- `hybrid_decision_tail_balance` = experimental failed weak-anomaly probe

Most useful summary commands:

1. Unified graph scorer summary:
   - `python scripts/summarize_graph_scorers.py ...`
2. Formal graph mainline vs tabular follow-up:
   - `python scripts/summarize_graph_mainline_hybrid.py ...`
3. Narrow hybrid vs decision trade-off:
   - `python scripts/summarize_hybrid_vs_decision_tradeoff.py ...`

Most useful output files:

- `artifacts/ciciot2023/analysis/graph_scorer_family_summary.csv`
- `artifacts/ciciot2023/analysis/paper_inspired_scorer_family_comparison.csv`
- `artifacts/ciciot2023/analysis/graph_vs_tabular_hybrid_followup.csv`

Current working rule:

- The scorer family is already converged enough for mainline use.
- Prefer consuming the existing summaries before adding new scorer variants.

Failed experimental reducer:

- `hybrid_decision_tail_balance` was tested as a weak-anomaly-friendly
  decision-style reducer under the fixed real-PCAP protocol
  (`random_window`, `packet_limit=20000`, `seed=42`).
- It did not improve the targeted weak anomalies:
  - `BrowserHijacking` recall fell from `0.125` to `0.0`
  - `MITM-ArpSpoofing` stayed at `0.0` recall and remains provisional because
    the current MITM raw file is truncated
  - `DictionaryBruteForce` recall fell from `0.3125` to `0.0625`
- It also hurt the structural control case:
  - `Backdoor_Malware` recall fell from `0.375` to `0.0`
- `FPR` stayed flat in that snapshot, but the reducer is still a failed
  experimental probe because it reduced recall without compensating gains.
- Keep the name in the repository for traceability only. Do not optimize it
  further, do not expand attacks around it, and do not treat it as a candidate
  for promotion.
- A later low-risk refresh of the default `hybrid_max_rank_flow_node_max`
  reducer also failed to produce a meaningful weak-anomaly gain. That result
  does not reopen `hybrid_decision_tail_balance`; it reinforces the decision to
  keep the failed reducer frozen as historical traceability only.
