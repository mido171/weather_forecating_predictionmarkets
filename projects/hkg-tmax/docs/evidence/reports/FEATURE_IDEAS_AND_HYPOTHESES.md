# Feature Ideas And Hypotheses

## Candidate Signals From Current Parsed Archive

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
| hko_mslp_at_tminus1_1500_hpa | development | 728 | -0.751 | eda_only_not_model_selection |
| hko_mslp_tminus1_1200_hpa | development | 726 | -0.7542 | eda_only_not_model_selection |

## Next Feature Families To Engineer

- Diurnal shape before cutoff: morning heating slope, noon-to-15:00 acceleration, and overnight minimum recovery.
- Urban versus marine contrast: HKO minus King's Park, HKO minus Waglan Island, Chek Lap Kok minus HKO.
- Moisture/heat-index regime: humidity-conditioned persistence and dew-point depression proxies.
- Synoptic pressure tendency: pressure fall/rise over 3h, 6h, 12h before cutoff.
- Radiation/cloud mechanism proxies from King's Park solar and HKO daily cloud/sunshine labels, with strict timestamp separation.
- TC/advection flags from advisory vintages only after historical/live pair contract is complete.
