# EDA Master Report

This report intentionally stops before model fitting. Correlations are screening evidence, not tuned performance claims.

## Strongest Development-Period Candidate Signals

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

## 2024 Validation Stability Check

| feature | split | n | pearson_r_with_target_tmax | role |
| --- | --- | --- | --- | --- |
| hko_temp_at_tminus1_1500_c | validation_2024 | 364 | 0.917 | eda_only_not_model_selection |
| hko_temp_tminus1_1200_c | validation_2024 | 364 | 0.9082 | eda_only_not_model_selection |
| hko_temp_tminus1_0900_c | validation_2024 | 364 | 0.9057 | eda_only_not_model_selection |
| hko_tminus1_min_so_far_1500_c | validation_2024 | 360 | 0.8959 | eda_only_not_model_selection |
| hko_tminus2_official_tmax_c | validation_2024 | 366 | 0.8565 | eda_only_not_model_selection |
| hko_mslp_3h_change_to_cutoff_hpa | validation_2024 | 364 | 0.4453 | eda_only_not_model_selection |
| hko_rh_at_tminus1_1500_pct | validation_2024 | 364 | 0.153 | eda_only_not_model_selection |
| hko_temp_3h_change_to_cutoff_c | validation_2024 | 364 | 0.0738 | eda_only_not_model_selection |
| hko_temp_6h_change_to_cutoff_c | validation_2024 | 364 | 0.0492 | eda_only_not_model_selection |
| hko_mslp_at_tminus1_1500_hpa | validation_2024 | 364 | -0.7837 | eda_only_not_model_selection |
| hko_mslp_tminus1_1200_hpa | validation_2024 | 364 | -0.7874 | eda_only_not_model_selection |
| hko_tminus1_max_so_far_1500_c | validation_2024 | 0 | nan | eda_only_not_model_selection |

## High-Value Hypotheses

- T-1 afternoon HKO temperature carries direct persistence information for next-day Tmax.
- T-1 since-midnight max/min separates hot-airmass persistence from transient afternoon spikes.
- Pressure tendency and humidity at cutoff may help identify synoptic regime and overnight cooling potential.
- Station-network offsets can expose sea-breeze penetration, urban heat storage, and northwestern New Territories heating regimes.
