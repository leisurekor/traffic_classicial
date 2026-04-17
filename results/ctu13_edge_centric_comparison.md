# CTU-13 Edge-Centric Comparison

| evaluation_mode | scenario_id | baseline_profile | edge_profile | nuisance_profile | baseline_f1 | edge_v2_f1 | nuisance_f1 | baseline_recall | edge_v2_recall | nuisance_recall | baseline_fpr | edge_v2_fpr | nuisance_fpr | edge_background_hit_ratio | nuisance_background_hit_ratio | nuisance_rejection_rate | malicious_blocked_by_nuisance_rate |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| merged | merged_48_49_52 | heldout_q99 | heldout_q95_top1:local_support_density:False | heldout_q95_top1:old_concentration:nuisance_q95_margin050 | 0.2857 | 0.9412 | 0.9412 | 0.1765 | 0.9412 | 0.9412 | 0.0294 | 0.0294 | 0.0294 | 0.2844 | 0.2979 | 0.0150 | 0.0000 |
| scenario_wise | 48 | heldout_q99 | heldout_q99_top5:local_support_density:False | heldout_q99_top5:local_support_density:nuisance_q95_margin050 | 0.0000 | 0.6667 | 0.6667 | 0.0000 | 0.5000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.1561 | 0.1621 | 0.0173 | 0.0000 |
| scenario_wise | 49 | heldout_q99 | heldout_q95_top1:local_support_density:False | heldout_q99_top5:old_concentration:nuisance_q95_margin050 | 0.0000 | 0.8571 | 0.8571 | 0.0000 | 1.0000 | 1.0000 | 0.0909 | 0.0909 | 0.0909 | 0.3286 | 0.3110 | 0.0517 | 0.0000 |
| scenario_wise | 52 | heldout_q99 | heldout_q99_top5:local_support_density:False | heldout_q99_top5:local_support_density:nuisance_q95_margin050 | 0.4000 | 1.0000 | 1.0000 | 0.2500 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2546 | 0.2388 | 0.0547 | 0.0000 |
