# Temporal Edge Distribution Probe

Fixed protocol:
- packet_limit = `20000`
- packet_sampling_mode = `random_window`
- seed = `42`
- benign train graphs = `9`
- benign eval graphs = `3`

Benign source offsets:
- `BenignTraffic.pcap` -> offset `1115990`
- `BenignTraffic1.redownload.pcap` -> offset `1567742`
- `BenignTraffic3.redownload2.pcap` -> offset `537912`

| attack | variant | benign mean | benign std | attack mean | attack std | gap | signal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| browser_hijacking | baseline_graphsage | 79289401.947517 | 7084643.936138 | 75022766.346248 | 8033794.157721 | 0.398327 | low |
| mitm_arp_spoofing | baseline_graphsage | 79289401.947517 | 7084643.936138 | 83849200.415755 | 0.000000 | 0.643617 | medium |
| dictionary_bruteforce | baseline_graphsage | 79289401.947517 | 7084643.936138 | 90124285.605687 | 4521086.902456 | 1.289206 | high |
| backdoor_malware | baseline_graphsage | 79289401.947517 | 7084643.936138 | 84845300.366001 | 6658126.123692 | 0.571460 | medium |
| browser_hijacking | temporal_edge_branch | 79288310.522336 | 7084527.028365 | 75021751.881397 | 8033687.097004 | 0.398325 | low |
| mitm_arp_spoofing | temporal_edge_branch | 79288310.522336 | 7084527.028365 | 83848149.212739 | 0.000000 | 0.643633 | medium |
| dictionary_bruteforce | temporal_edge_branch | 79288310.522336 | 7084527.028365 | 90123176.570673 | 4521037.896664 | 1.289223 | high |
| backdoor_malware | temporal_edge_branch | 79288310.522336 | 7084527.028365 | 84844236.749165 | 6658087.607728 | 0.571470 | medium |

## Observations

- `browser_hijacking` stays close to the baseline separation: baseline gap `0.398327` vs temporal gap `0.398325`.
- `mitm_arp_spoofing` stays close to the baseline separation: baseline gap `0.643617` vs temporal gap `0.643633`.
- `mitm_arp_spoofing` remains `provisional`, so this row is still only directional.
- `dictionary_bruteforce` stays close to the baseline separation: baseline gap `1.289206` vs temporal gap `1.289223`.
- `backdoor_malware` stays close to the baseline separation: baseline gap `0.571460` vs temporal gap `0.571470`.
- The temporal edge branch does not yet improve score-distribution separation in a meaningful way under this fixed protocol.
- That means the branch is runnable and isolated, but it is not yet showing clear weak-anomaly gains.