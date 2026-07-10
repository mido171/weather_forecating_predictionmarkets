# T-1 15:00 Station State

Development-period correlation screening for cutoff-safe HKO station-state features:

| feature | split | n | pearson_r_with_target_tmax | role |
| --- | --- | --- | --- | --- |
| hko_temp_at_tminus1_1500_c | development | 1052 | 0.9232 | eda_only_not_model_selection |
| hko_temp_tminus1_1200_c | development | 1050 | 0.9199 | eda_only_not_model_selection |
| hko_tminus1_max_so_far_1500_c | development | 318 | 0.9093 | eda_only_not_model_selection |
| hko_temp_tminus1_0900_c | development | 1053 | 0.9056 | eda_only_not_model_selection |
| hko_tminus1_min_so_far_1500_c | development | 1045 | 0.8969 | eda_only_not_model_selection |
| hko_tminus2_official_tmax_c | development | 48573 | 0.8695 | eda_only_not_model_selection |
| hko_mslp_3h_change_to_cutoff_hpa | development | 726 | 0.4076 | eda_only_not_model_selection |
| hko_rh_at_tminus1_1500_pct | development | 1051 | 0.1072 | eda_only_not_model_selection |
| hko_temp_6h_change_to_cutoff_c | development | 1051 | 0.0609 | eda_only_not_model_selection |
| hko_temp_3h_change_to_cutoff_c | development | 1050 | 0.0489 | eda_only_not_model_selection |

This is EDA only, not model selection. The target-day label is not used as a feature.
