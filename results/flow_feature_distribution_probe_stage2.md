# Stage-2 Flow Feature Distribution Probe

Fixed protocol:
- packet_limit = `20000`
- packet_sampling_mode = `random_window`
- seed = `42`

Benign sources and offsets:
- `BenignTraffic.pcap` -> offset `1115990`
- `BenignTraffic1.redownload.pcap` -> offset `1567742`
- `BenignTraffic3.redownload2.pcap` -> offset `537912`

## browser_hijacking

Sampled offset: `11795`

| feature | benign mean | benign std | attack mean | attack std | signal | abs delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| first_4_packet_pattern_code | 4500.217371 | 5110.143016 | 3847.720865 | 4964.822397 | low | 652.496506 |
| retry_burst_max_len | 7.575941 | 102.424945 | 2.502820 | 30.781885 | low | 5.073121 |
| small_pkt_burst_count | 4.094752 | 45.701665 | 2.855263 | 14.848410 | low | 1.239488 |
| retry_burst_count | 1.316303 | 3.353434 | 2.007519 | 13.194751 | low | 0.691216 |

## mitm_arp_spoofing

Sampled offset: `63623`

| feature | benign mean | benign std | attack mean | attack std | signal | abs delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| first_4_packet_pattern_code | 4500.217371 | 5110.143016 | 2543.067164 | 4348.767764 | low | 1957.150207 |
| retry_burst_max_len | 7.575941 | 102.424945 | 78.313433 | 872.003692 | low | 70.737492 |
| coarse_ack_delay_p75 | 2.139817 | 8.852302 | 0.168241 | 0.579196 | low | 1.971576 |
| coarse_ack_delay_mean | 1.435739 | 5.809353 | 0.118807 | 0.380207 | low | 1.316931 |

- Note: `mitm_arp_spoofing` remains `provisional` because the raw PCAP is truncated.

## dictionary_bruteforce

Sampled offset: `43467`

| feature | benign mean | benign std | attack mean | attack std | signal | abs delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| first_4_packet_pattern_code | 4500.217371 | 5110.143016 | 5427.290381 | 5188.628036 | low | 927.073010 |
| retry_burst_max_len | 7.575941 | 102.424945 | 1.191470 | 6.177252 | low | 6.384470 |
| retry_burst_count | 1.316303 | 3.353434 | 2.799456 | 6.042100 | low | 1.483153 |
| small_pkt_burst_count | 4.094752 | 45.701665 | 3.374773 | 22.258020 | low | 0.719978 |

## backdoor_malware

Sampled offset: `8068`

| feature | benign mean | benign std | attack mean | attack std | signal | abs delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| first_4_packet_pattern_code | 4500.217371 | 5110.143016 | 4999.858974 | 5173.839946 | low | 499.641603 |
| retry_burst_max_len | 7.575941 | 102.424945 | 1.668091 | 8.709760 | low | 5.907849 |
| small_pkt_burst_count | 4.094752 | 45.701665 | 2.040598 | 8.903354 | low | 2.054153 |
| coarse_ack_delay_p75 | 2.139817 | 8.852302 | 2.320151 | 10.070174 | low | 0.180333 |

## Observations

- Stage-2 still produced no `medium` or `high` signal rows, so the new TCP-aware features do not yet look stronger than the stage-1 probe.
- `browser_hijacking` is most differentiated by `first_4_packet_pattern_code` (low), `retry_burst_max_len` (low), `small_pkt_burst_count` (low).
- `mitm_arp_spoofing` is most differentiated by `first_4_packet_pattern_code` (low), `retry_burst_max_len` (low), `coarse_ack_delay_p75` (low).
- `dictionary_bruteforce` is most differentiated by `first_4_packet_pattern_code` (low), `retry_burst_max_len` (low), `retry_burst_count` (low).
- `backdoor_malware` is most differentiated by `first_4_packet_pattern_code` (low), `retry_burst_max_len` (low), `small_pkt_burst_count` (low).
- ACK-delay proxy / seq-ack quality / retry-burst features remain weak in this snapshot, so they are not ready to be pushed back into graph summary yet.
- This is still a useful probe, but it does not yet justify a new default-hybrid experiment.
