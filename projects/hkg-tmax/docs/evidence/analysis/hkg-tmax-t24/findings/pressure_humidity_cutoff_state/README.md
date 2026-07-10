# Pressure And Humidity Cutoff State

Moisture and pressure screening evidence:

| feature | split | n | pearson_r_with_target_tmax | role |
| --- | --- | --- | --- | --- |
| hko_mslp_3h_change_to_cutoff_hpa | development | 726 | 0.4076 | eda_only_not_model_selection |
| hko_rh_at_tminus1_1500_pct | development | 1051 | 0.1072 | eda_only_not_model_selection |
| hko_mslp_at_tminus1_1500_hpa | development | 728 | -0.751 | eda_only_not_model_selection |
| hko_mslp_3h_change_to_cutoff_hpa | validation_2024 | 364 | 0.4453 | eda_only_not_model_selection |
| hko_rh_at_tminus1_1500_pct | validation_2024 | 364 | 0.153 | eda_only_not_model_selection |
| hko_mslp_at_tminus1_1500_hpa | validation_2024 | 364 | -0.7837 | eda_only_not_model_selection |

These variables should be treated as regime modifiers, not standalone answers.
