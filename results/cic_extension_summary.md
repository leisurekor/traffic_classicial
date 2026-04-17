# CIC IoT 2023 Extension Summary

| attack_type | FPR | F1 | recall |
| --- | --- | --- | --- |
| `mitm_arp_spoofing` | 0.000000 | 0.500000 | 0.333333 |
| `botnet_backdoor` | 0.071429 | 0.521739 | 0.375000 |
| `brute_force` | 0.000000 | 0.400000 | 0.250000 |

## Short Analysis

- `hybrid_max_rank_flow_node_max` shows non-zero recall on all three new attacks, so the current graph pipeline does transfer beyond the original Recon / DDoS / Browser set.
- The hardest new attack in this snapshot is `brute_force` with recall `0.250000`.
- `MITM-ArpSpoofing` recall is `0.333333`, which is the closest weak-anomaly check to the existing BrowserHijacking weakness: if it stays low, that suggests the same subtle-behavior limitation is still present.
- `Backdoor_Malware` recall is `0.375000`. If it lands above MITM, that supports the current intuition that graph structure helps more on coordinated / structural anomalies than on weak behavioral shifts.
- `DictionaryBruteForce` recall is `0.250000`, giving a middle-ground reference between weak MITM-style signals and more structural backdoor behavior.
