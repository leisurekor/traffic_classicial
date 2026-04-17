# Flow Feature Distribution Probe

Fixed protocol:
- packet_limit = `20000`
- packet_sampling_mode = `random_window`
- seed = `42`

Benign sources and offsets:
- `BenignTraffic.pcap` -> offset `1115990`
- `BenignTraffic1.redownload.pcap` -> offset `1567742`
- `BenignTraffic3.redownload2.pcap` -> offset `537912`

## browser_hijacking

| feature | benign mean | benign std | attack mean | attack std | signal | abs delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| first_packet_size_pattern | 143.425453 | 156.932921 | 119.994361 | 146.428933 | low | 23.431092 |
| retry_like_count | 10.338133 | 113.457035 | 5.122180 | 35.925598 | low | 5.215952 |
| iat_hist_bin_3 | 0.034191 | 0.085212 | 0.207139 | 0.363977 | low | 0.172948 |
| iat_hist_bin_5 | 0.452550 | 0.432005 | 0.335564 | 0.421591 | low | 0.116986 |
| flag_pattern_code | 0.614491 | 0.925469 | 0.696429 | 0.996680 | low | 0.081937 |
| pkt_len_hist_bin_1 | 0.630414 | 0.388736 | 0.549661 | 0.393946 | low | 0.080753 |
| pkt_len_hist_bin_0 | 0.129593 | 0.297210 | 0.202245 | 0.347040 | low | 0.072652 |
| pkt_len_hist_bin_3 | 0.101734 | 0.243607 | 0.066632 | 0.199306 | low | 0.035102 |
| pkt_len_hist_bin_2 | 0.119931 | 0.264338 | 0.154901 | 0.307253 | low | 0.034969 |
| iat_hist_bin_4 | 0.357818 | 0.403131 | 0.322967 | 0.390774 | low | 0.034852 |
| flag_syn_ratio | 0.045013 | 0.145970 | 0.078932 | 0.238158 | low | 0.033918 |
| iat_hist_bin_1 | 0.054105 | 0.133201 | 0.035376 | 0.101225 | low | 0.018729 |
| flag_rst_ratio | 0.005555 | 0.049932 | 0.021130 | 0.098258 | low | 0.015575 |
| retry_like_ratio | 0.103525 | 0.185043 | 0.111394 | 0.216842 | low | 0.007869 |
| pkt_len_hist_bin_5 | 0.009152 | 0.048754 | 0.014589 | 0.067015 | low | 0.005437 |
| flag_ack_ratio | 0.278367 | 0.432620 | 0.275165 | 0.430241 | low | 0.003202 |
| pkt_len_hist_bin_4 | 0.009176 | 0.066880 | 0.011972 | 0.071111 | low | 0.002796 |
| iat_hist_bin_0 | 0.004571 | 0.021716 | 0.003040 | 0.017248 | low | 0.001532 |
| iat_hist_bin_2 | 0.027558 | 0.069562 | 0.027306 | 0.099525 | low | 0.000252 |

## mitm_arp_spoofing

| feature | benign mean | benign std | attack mean | attack std | signal | abs delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| retry_like_count | 10.338133 | 113.457035 | 82.268657 | 912.098418 | low | 71.930524 |
| first_packet_size_pattern | 143.425453 | 156.932921 | 86.738806 | 133.609976 | low | 56.686647 |
| pkt_len_hist_bin_1 | 0.630414 | 0.388736 | 0.494816 | 0.435997 | low | 0.135598 |
| pkt_len_hist_bin_2 | 0.119931 | 0.264338 | 0.235084 | 0.366334 | low | 0.115153 |
| iat_hist_bin_4 | 0.357818 | 0.403131 | 0.471209 | 0.443575 | low | 0.113390 |
| pkt_len_hist_bin_0 | 0.129593 | 0.297210 | 0.225890 | 0.356205 | low | 0.096297 |
| iat_hist_bin_5 | 0.452550 | 0.432005 | 0.358016 | 0.437730 | low | 0.094534 |
| pkt_len_hist_bin_3 | 0.101734 | 0.243607 | 0.025555 | 0.115598 | low | 0.076178 |
| flag_pattern_code | 0.614491 | 0.925469 | 0.649254 | 0.986910 | low | 0.034762 |
| retry_like_ratio | 0.103525 | 0.185043 | 0.069254 | 0.169838 | low | 0.034270 |
| flag_syn_ratio | 0.045013 | 0.145970 | 0.021868 | 0.133919 | low | 0.023145 |
| flag_rst_ratio | 0.005555 | 0.049932 | 0.027756 | 0.152633 | low | 0.022201 |
| iat_hist_bin_1 | 0.054105 | 0.133201 | 0.035290 | 0.112886 | low | 0.018816 |
| iat_hist_bin_2 | 0.027558 | 0.069562 | 0.011585 | 0.045373 | low | 0.015973 |
| pkt_len_hist_bin_5 | 0.009152 | 0.048754 | 0.013361 | 0.071970 | low | 0.004209 |
| pkt_len_hist_bin_4 | 0.009176 | 0.066880 | 0.005293 | 0.030955 | low | 0.003883 |
| flag_ack_ratio | 0.278367 | 0.432620 | 0.274743 | 0.441168 | low | 0.003624 |
| iat_hist_bin_3 | 0.034191 | 0.085212 | 0.036878 | 0.125497 | low | 0.002687 |
| iat_hist_bin_0 | 0.004571 | 0.021716 | 0.004933 | 0.032851 | low | 0.000362 |

- Note: `mitm_arp_spoofing` remains `provisional` because the raw PCAP is truncated.

## dictionary_bruteforce

| feature | benign mean | benign std | attack mean | attack std | signal | abs delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| first_packet_size_pattern | 143.425453 | 156.932921 | 162.297641 | 149.354839 | low | 18.872188 |
| retry_like_count | 10.338133 | 113.457035 | 4.907441 | 24.421112 | low | 5.430692 |
| flag_pattern_code | 0.614491 | 0.925469 | 0.924682 | 1.134650 | low | 0.310191 |
| iat_hist_bin_5 | 0.452550 | 0.432005 | 0.331254 | 0.390068 | low | 0.121297 |
| flag_ack_ratio | 0.278367 | 0.432620 | 0.349168 | 0.445759 | low | 0.070802 |
| pkt_len_hist_bin_0 | 0.129593 | 0.297210 | 0.196618 | 0.332428 | low | 0.067025 |
| pkt_len_hist_bin_3 | 0.101734 | 0.243607 | 0.040878 | 0.180082 | low | 0.060856 |
| flag_syn_ratio | 0.045013 | 0.145970 | 0.105078 | 0.249984 | low | 0.060064 |
| iat_hist_bin_4 | 0.357818 | 0.403131 | 0.413350 | 0.388224 | low | 0.055532 |
| iat_hist_bin_3 | 0.034191 | 0.085212 | 0.084339 | 0.180926 | low | 0.050149 |
| retry_like_ratio | 0.103525 | 0.185043 | 0.148959 | 0.211030 | low | 0.045434 |
| flag_rst_ratio | 0.005555 | 0.049932 | 0.036423 | 0.105480 | low | 0.030868 |
| iat_hist_bin_1 | 0.054105 | 0.133201 | 0.040100 | 0.111676 | low | 0.014005 |
| pkt_len_hist_bin_2 | 0.119931 | 0.264338 | 0.111645 | 0.238726 | low | 0.008286 |
| pkt_len_hist_bin_4 | 0.009176 | 0.066880 | 0.014169 | 0.088802 | low | 0.004993 |
| pkt_len_hist_bin_1 | 0.630414 | 0.388736 | 0.626201 | 0.373596 | low | 0.004213 |
| iat_hist_bin_2 | 0.027558 | 0.069562 | 0.023433 | 0.075412 | low | 0.004125 |
| iat_hist_bin_0 | 0.004571 | 0.021716 | 0.001353 | 0.010718 | low | 0.003218 |
| pkt_len_hist_bin_5 | 0.009152 | 0.048754 | 0.010489 | 0.031616 | low | 0.001336 |

## backdoor_malware

| feature | benign mean | benign std | attack mean | attack std | signal | abs delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| first_packet_size_pattern | 143.425453 | 156.932921 | 152.090456 | 154.438468 | low | 8.665003 |
| retry_like_count | 10.338133 | 113.457035 | 3.574074 | 15.439047 | low | 6.764059 |
| iat_hist_bin_5 | 0.452550 | 0.432005 | 0.352615 | 0.408084 | low | 0.099935 |
| pkt_len_hist_bin_1 | 0.630414 | 0.388736 | 0.536515 | 0.402401 | low | 0.093898 |
| pkt_len_hist_bin_0 | 0.129593 | 0.297210 | 0.202381 | 0.349288 | low | 0.072787 |
| pkt_len_hist_bin_2 | 0.119931 | 0.264338 | 0.187458 | 0.330725 | low | 0.067527 |
| pkt_len_hist_bin_3 | 0.101734 | 0.243607 | 0.049188 | 0.185905 | low | 0.052546 |
| iat_hist_bin_4 | 0.357818 | 0.403131 | 0.401041 | 0.396073 | low | 0.043223 |
| iat_hist_bin_3 | 0.034191 | 0.085212 | 0.072130 | 0.180294 | low | 0.037939 |
| iat_hist_bin_1 | 0.054105 | 0.133201 | 0.040343 | 0.112123 | low | 0.013762 |
| pkt_len_hist_bin_5 | 0.009152 | 0.048754 | 0.016946 | 0.077149 | low | 0.007794 |
| retry_like_ratio | 0.103525 | 0.185043 | 0.097651 | 0.194529 | low | 0.005873 |
| flag_ack_ratio | 0.278367 | 0.432620 | 0.273855 | 0.434942 | low | 0.004512 |
| iat_hist_bin_0 | 0.004571 | 0.021716 | 0.000383 | 0.003734 | low | 0.004189 |
| flag_pattern_code | 0.614491 | 0.925469 | 0.618234 | 0.933460 | low | 0.003742 |
| flag_syn_ratio | 0.045013 | 0.145970 | 0.048100 | 0.179885 | low | 0.003087 |
| pkt_len_hist_bin_4 | 0.009176 | 0.066880 | 0.007512 | 0.057488 | low | 0.001664 |
| flag_rst_ratio | 0.005555 | 0.049932 | 0.006565 | 0.058007 | low | 0.001010 |
| iat_hist_bin_2 | 0.027558 | 0.069562 | 0.026650 | 0.091240 | low | 0.000908 |

## Observations

- `browser_hijacking` most clearly separates on `first_packet_size_pattern` (low), `retry_like_count` (low), `iat_hist_bin_3` (low); sampled offset `11795`.
- `mitm_arp_spoofing` most clearly separates on `retry_like_count` (low), `first_packet_size_pattern` (low), `pkt_len_hist_bin_1` (low); sampled offset `63623`.
- `dictionary_bruteforce` most clearly separates on `first_packet_size_pattern` (low), `retry_like_count` (low), `flag_pattern_code` (low); sampled offset `43467`.
- `backdoor_malware` most clearly separates on `first_packet_size_pattern` (low), `retry_like_count` (low), `iat_hist_bin_5` (low); sampled offset `8068`.
- Features that repeatedly land in `high` or `medium` signal across BrowserHijacking, MITM, and DictionaryBruteForce are the best candidates for the next flow-schema follow-up. This probe stops here on purpose and does not touch reducers or benchmark metrics.
