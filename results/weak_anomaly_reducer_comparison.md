# Weak Anomaly Reducer Comparison

| scorer_name | scorer_role | attack_type | FPR | F1 | recall |
| --- | --- | --- | --- | --- | --- |
| hybrid_max_rank_flow_node_max | default_candidate | browser_hijacking | 0.071429 | 0.210526 | 0.125000 |
| hybrid_decision_tail_balance | experimental | browser_hijacking | 0.071429 | 0.000000 | 0.000000 |
| hybrid_max_rank_flow_node_max | default_candidate | mitm_arp_spoofing | 0.071429 | 0.000000 | 0.000000 |
| hybrid_decision_tail_balance | experimental | mitm_arp_spoofing | 0.071429 | 0.000000 | 0.000000 |
| hybrid_max_rank_flow_node_max | default_candidate | dictionary_bruteforce | 0.071429 | 0.454545 | 0.312500 |
| hybrid_decision_tail_balance | experimental | dictionary_bruteforce | 0.071429 | 0.111111 | 0.062500 |
| hybrid_max_rank_flow_node_max | default_candidate | backdoor_malware | 0.071429 | 0.521739 | 0.375000 |
| hybrid_decision_tail_balance | experimental | backdoor_malware | 0.071429 | 0.000000 | 0.000000 |

## Observations
- BrowserHijacking recall: baseline `0.125000` vs tail-balance `0.000000`.
- MITM-ArpSpoofing recall: baseline `0.000000` vs tail-balance `0.000000`. This remains provisional because the current MITM raw file is truncated.
- DictionaryBruteForce recall: baseline `0.312500` vs tail-balance `0.062500`.
- Backdoor_Malware recall: baseline `0.375000` vs tail-balance `0.000000`.
- Overall FPR stayed unchanged in this snapshot at `0.071429` for both reducers across all four runs.

## Recommendation
- Keep `hybrid_decision_tail_balance` as `experimental` only.
- This first tail-balance variant did not improve the targeted weak anomalies, so it is not ready for promotion or broader validation.
- Treat this as a failed experimental reducer record, not as an active branch for further optimization.
- Do not promote it, do not expand scorer family around it, and do not reopen it unless a genuinely new reducer idea appears.
