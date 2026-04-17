# Weak Anomaly Hybrid Comparison

| scorer_name | scorer_role | attack_type | FPR | F1 | recall | note |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid_legacy_reference | reference | browser_hijacking | 0.071429 | 0.210526 | 0.125000 | historical_prefix_fixed_random_window_reference |
| hybrid_max_rank_flow_node_max | default_candidate | browser_hijacking | 0.357143 | 0.384615 | 0.312500 |  |
| hybrid_legacy_reference | reference | mitm_arp_spoofing | 0.071429 | 0.000000 | 0.000000 | historical_prefix_fixed_random_window_reference |
| hybrid_max_rank_flow_node_max | default_candidate | mitm_arp_spoofing | 0.357143 | 0.250000 | 0.500000 | provisional |
| hybrid_legacy_reference | reference | dictionary_bruteforce | 0.071429 | 0.454545 | 0.312500 | historical_prefix_fixed_random_window_reference |
| hybrid_max_rank_flow_node_max | default_candidate | dictionary_bruteforce | 0.357143 | 0.384615 | 0.312500 |  |
| hybrid_legacy_reference | reference | backdoor_malware | 0.071429 | 0.521739 | 0.375000 | historical_prefix_fixed_random_window_reference |
| hybrid_max_rank_flow_node_max | default_candidate | backdoor_malware | 0.357143 | 0.250000 | 0.187500 |  |

## Observations
- BrowserHijacking recall: legacy `0.125000` vs current `0.312500`.
- MITM-ArpSpoofing recall: legacy `0.000000` vs current `0.500000`. This remains provisional because the MITM raw file is truncated.
- DictionaryBruteForce recall: legacy `0.312500` vs current `0.312500`.
- Backdoor_Malware recall: legacy `0.375000` vs current `0.187500`.
- BrowserHijacking FPR: legacy `0.071429` vs current `0.357143`.

## Recommendation
- The new graph-level calibration is materially more aggressive: it lifts BrowserHijacking, MITM-ArpSpoofing relative to the frozen legacy reference.
- This gain is not free. It regresses Backdoor_Malware under the same fixed seed-42 protocol.
- The main tradeoff is false positives: FPR rises by `0.285714` in the fixed weak-anomaly snapshot.
- Treat this as an aggressive detection profile built on the new edge-centric representation, not as a quiet no-op refresh.
- It still does not reopen the failed `hybrid_decision_tail_balance` reducer branch; the improvement comes from graph-level calibration of the new representation, not from reviving that old reducer idea.
