# CICIoT2023 Data Integrity and Sampling Check

## Integrity findings

- `mitm_arp_spoofing`: `partial_or_prefix` (local_size=`193986560`, remote_length=`None`, packet_count=`171608`, truncated=`True`)
- `botnet_backdoor`: `probably_complete` (local_size=`10544806`, remote_length=`None`, packet_count=`33414`, truncated=`False`)
- `brute_force`: `probably_complete` (local_size=`39130622`, remote_length=`None`, packet_count=`133138`, truncated=`False`)
- `BenignTraffic.pcap`: `probably_complete` (local_size=`2048000100`, remote_length=`None`, packet_count=`3664164`, truncated=`False`)
- `BenignTraffic1.redownload.pcap`: `probably_complete` (local_size=`2048002499`, remote_length=`None`, packet_count=`2988642`, truncated=`False`)
- `BenignTraffic3.redownload2.pcap`: `probably_complete` (local_size=`853841091`, remote_length=`None`, packet_count=`1311897`, truncated=`False`)

## Prefix-bias finding

- Current `packet_limit` behavior is prefix-based: `ClassicPcapReader.iter_packets()` stops as soon as the first `max_packets` records are consumed.
- This bias affects both benign training inputs and malicious evaluation inputs because `run_pcap_graph_experiment()` passes `config.packet_limit` directly into `load_pcap_flow_dataset(..., max_packets=...)` for every source.

## Minimal fix applied

- Added `packet_sampling_mode` to the PCAP experiment config.
- Added `random_window` as a minimal sampling option that keeps packet order intact by choosing one reproducible continuous packet window instead of always taking the prefix.
- Reproducibility stays tied to `random_seed`; the chosen packet start offset is now exported through each source `parse_summary`.

## Prefix vs random-window validation

| sampling_mode | FPR | F1 | recall | benign_packet_count | attack_packet_count | benign_packet_start_offset | attack_packet_start_offset |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `prefix` | 0.400000 | 0.571429 | 0.666667 | 20000 | 20000 | 0 | 0 |
| `random_window` | 0.200000 | 0.500000 | 0.500000 | 20000 | 20000 | 234478 | 94066 |

## Recommendation

- The new CIC extension MITM file is `partial_or_prefix`, so key conclusions from it should be treated as provisional until a fully verified download is available.
- The benign training files currently used by the mainline are all at least `probably_complete`, and the three formal benign references are supported by clean EOF parses.
- For bounded mini experiments, `random_window` is the safer default than `prefix` because it removes deterministic prefix bias while preserving local time order.
- If later paper-facing results depend materially on the new CIC extension attacks, we should redownload the full official files first, then rerun the same fixed protocol once on the complete captures.

## Validation run directories

- prefix: `/home/xdo/traffic_classicial/artifacts/sampling_bias_check/mitm_arp_spoofing-pcap-graph-experiment/20260415T112324Z`
- random_window: `/home/xdo/traffic_classicial/artifacts/sampling_bias_check/mitm_arp_spoofing-pcap-graph-experiment/20260415T112326Z`
