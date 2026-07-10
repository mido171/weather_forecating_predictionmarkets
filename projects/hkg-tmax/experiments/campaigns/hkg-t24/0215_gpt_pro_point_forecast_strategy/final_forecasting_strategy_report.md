# Final Forecasting Strategy Report

Generated: `2026-07-04T13:19:12Z`

## Context

This experiment implements GPT-Pro's requested HKO/HKG daily Tmax point-forecast strategy. The task is not Polymarket backtesting. The task is to produce the lowest practical MAE/RMSE point forecast for the Hong Kong Observatory daily absolute maximum temperature, using only data that would have been available before the selected T-1 cutoff.

The market-motivating target is the HKO "Absolute Daily Max (deg. C)" from the Daily Extract. That value resolves to one decimal place after publication. Any trading system built on top of this experiment should first consume the selected point forecast and its diagnostics; pricing, order placement, and market microstructure are intentionally out of scope here.

## Selected Strategy

- Selected cutoff: `23:59` HKT on T-1.
- Selected model: `B3_grouped_residual_shrinkage`.
- Selection rule: `default 23:59 candidate retained`.
- Official-row validation window: `2011-01-01` through `2023-12-31`.
- Official-row count: `4747`.
- Selected MAE / RMSE: `0.92161` / `1.18268` C.
- Selected median AE / p90 AE: `0.73722` / `1.96412` C.
- Selected bias: `0.01270` C.
- Raw official baseline MAE / RMSE at same cutoff: `0.92749` / `1.19152` C.
- MAE / RMSE delta versus raw official: `-0.00588` / `-0.00884` C.

Promotion gates:

- improves_mae_vs_raw_by_0_035: fail
- improves_rmse_vs_raw_by_0_035: fail
- abs_bias_lte_0_040: pass
- has_at_least_4500_official_rows: pass

## Data Inputs

| source_id | location | rows | first_date | last_date | null_or_unusable_percent | source_role |
| --- | --- | --- | --- | --- | --- | --- |
| target_history | feature_safe.hko_target_history_pre2024 | 8765 | 2000-01-02 | 2023-12-31 | 0.00000 | official target labels for daily absolute maximum temperature at HKO/HKG station |
| lead1_hko_forecast_archive | public.hko_historical_forecasts_2000_2026 | 80089 | 2000-01-02 | 2023-12-31 | 0.00000 | HKO local lead-1 forecast min/max archive used as official anchor |
| hko_daily_climate | diagnostic_physics.codex_audit_ds_02_hko_daily_climate_all_elements_hko_d_f7bb0017 | 187025 | 1999-01-01 | 2023-12-31 | 3.08568 | daily HKO climate state, lagged to T-2 in production features |
| gpt_pro_strategy_spec | C:\Users\ahmad\.codex\attachments\2f15d411-f901-46b6-9fb4-5bae7b3c26ef\pasted-text.txt | 1722 |  |  | 0.00000 | implementation specification read before coding |

## Baselines

The basic baseline is `B1_raw_official_latest`, the latest HKO lead-1 local forecast max available by the cutoff. `B0_yearsafe_doy_climatology` is also included as a non-forecast climatology sanity baseline. Residual baselines B2/B3 add fold-safe empirical residual correction to the raw official anchor.

| cutoff | model_id | model_family | scope | n | mae | rmse | median_abs_error | bias | p80_abs_error | p90_abs_error | p95_abs_error | max_abs_error | first_date | last_date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 23:59 | B3_grouped_residual_shrinkage | baseline | official_rows_only | 4747 | 0.92161 | 1.18268 | 0.73722 | 0.01270 | 1.47781 | 1.96412 | 2.39546 | 4.40336 | 2011-01-01 | 2023-12-31 |
| 23:59 | B2_monthly_residual_shrinkage | baseline | official_rows_only | 4747 | 0.92380 | 1.18564 | 0.73676 | 0.02158 | 1.49112 | 1.97706 | 2.39705 | 4.47727 | 2011-01-01 | 2023-12-31 |
| 23:59 | B1_raw_official_latest | baseline | official_rows_only | 4747 | 0.92749 | 1.19152 | 0.70000 | -0.12286 | 1.50000 | 2.00000 | 2.40000 | 4.50000 | 2011-01-01 | 2023-12-31 |
| 21:00 | B3_grouped_residual_shrinkage | baseline | official_rows_only | 4321 | 0.95020 | 1.21293 | 0.76545 | 0.00178 | 1.52607 | 2.02145 | 2.44684 | 4.39905 | 2011-01-01 | 2023-12-31 |
| 21:00 | B2_monthly_residual_shrinkage | baseline | official_rows_only | 4321 | 0.95179 | 1.21564 | 0.76522 | 0.01168 | 1.52243 | 2.00879 | 2.45097 | 4.49868 | 2011-01-01 | 2023-12-31 |
| 18:00 | B3_grouped_residual_shrinkage | baseline | official_rows_only | 4286 | 0.95736 | 1.22070 | 0.77704 | -0.01058 | 1.53388 | 2.03184 | 2.46007 | 4.41467 | 2011-01-01 | 2023-12-31 |
| 17:00 | B3_grouped_residual_shrinkage | baseline | official_rows_only | 4274 | 0.95763 | 1.22120 | 0.77765 | -0.01242 | 1.53423 | 2.03103 | 2.46026 | 4.41775 | 2011-01-01 | 2023-12-31 |
| 21:00 | B1_raw_official_latest | baseline | official_rows_only | 4321 | 0.95800 | 1.22295 | 0.80000 | -0.12907 | 1.50000 | 2.00000 | 2.40000 | 4.50000 | 2011-01-01 | 2023-12-31 |
| 18:00 | B2_monthly_residual_shrinkage | baseline | official_rows_only | 4286 | 0.95880 | 1.22332 | 0.77442 | -0.00155 | 1.53855 | 2.03669 | 2.46036 | 4.53889 | 2011-01-01 | 2023-12-31 |
| 17:00 | B2_monthly_residual_shrinkage | baseline | official_rows_only | 4274 | 0.95909 | 1.22387 | 0.77498 | -0.00391 | 1.54070 | 2.03744 | 2.46508 | 4.54593 | 2011-01-01 | 2023-12-31 |
| 18:00 | B1_raw_official_latest | baseline | official_rows_only | 4286 | 0.96472 | 1.23108 | 0.80000 | -0.13047 | 1.50000 | 2.00000 | 2.50000 | 4.50000 | 2011-01-01 | 2023-12-31 |
| 17:00 | B1_raw_official_latest | baseline | official_rows_only | 4274 | 0.96504 | 1.23163 | 0.80000 | -0.13149 | 1.50000 | 2.00000 | 2.50000 | 4.50000 | 2011-01-01 | 2023-12-31 |
| 17:00 | B4_direct_no_official_huber | baseline_direct_fallback | official_rows_only | 4274 | 1.85284 | 2.31302 | 1.58528 | -0.45098 | 2.89903 | 3.77123 | 4.47721 | 8.99212 | 2011-01-01 | 2023-12-31 |
| 21:00 | B4_direct_no_official_huber | baseline_direct_fallback | official_rows_only | 4321 | 1.85416 | 2.31402 | 1.58791 | -0.44826 | 2.90232 | 3.77129 | 4.48652 | 8.99212 | 2011-01-01 | 2023-12-31 |
| 18:00 | B4_direct_no_official_huber | baseline_direct_fallback | official_rows_only | 4286 | 1.85490 | 2.31519 | 1.58725 | -0.44825 | 2.90242 | 3.77267 | 4.48465 | 8.99212 | 2011-01-01 | 2023-12-31 |
| 17:00 | B5_direct_no_official_lgbm_l1 | baseline_direct_fallback | official_rows_only | 4274 | 1.87873 | 2.33457 | 1.62451 | -0.61800 | 2.91347 | 3.77942 | 4.51095 | 8.88982 | 2011-01-01 | 2023-12-31 |
| 21:00 | B5_direct_no_official_lgbm_l1 | baseline_direct_fallback | official_rows_only | 4321 | 1.87917 | 2.33423 | 1.62568 | -0.61417 | 2.91328 | 3.78076 | 4.51464 | 8.88982 | 2011-01-01 | 2023-12-31 |
| 18:00 | B5_direct_no_official_lgbm_l1 | baseline_direct_fallback | official_rows_only | 4286 | 1.88032 | 2.33599 | 1.62613 | -0.61517 | 2.91432 | 3.78146 | 4.51373 | 8.88982 | 2011-01-01 | 2023-12-31 |
| 23:59 | B4_direct_no_official_huber | baseline_direct_fallback | official_rows_only | 4747 | 1.88427 | 2.36180 | 1.60560 | -0.39209 | 2.92480 | 3.81452 | 4.64324 | 10.51305 | 2011-01-01 | 2023-12-31 |
| 23:59 | B5_direct_no_official_lgbm_l1 | baseline_direct_fallback | official_rows_only | 4747 | 1.90225 | 2.37222 | 1.64551 | -0.55182 | 2.93369 | 3.82242 | 4.60956 | 9.97594 | 2011-01-01 | 2023-12-31 |
| 17:00 | B0_yearsafe_doy_climatology | baseline | official_rows_only | 4274 | 2.06021 | 2.56882 | 1.76114 | -0.66499 | 3.22761 | 4.15891 | 5.06649 | 10.31686 | 2011-01-01 | 2023-12-31 |
| 18:00 | B0_yearsafe_doy_climatology | baseline | official_rows_only | 4286 | 2.06241 | 2.57087 | 1.76258 | -0.66296 | 3.23240 | 4.17532 | 5.06683 | 10.31686 | 2011-01-01 | 2023-12-31 |
| 21:00 | B0_yearsafe_doy_climatology | baseline | official_rows_only | 4321 | 2.06384 | 2.57209 | 1.76344 | -0.66416 | 3.23351 | 4.17627 | 5.06705 | 10.31686 | 2011-01-01 | 2023-12-31 |
| 23:59 | B0_yearsafe_doy_climatology | baseline | official_rows_only | 4747 | 2.09679 | 2.62393 | 1.77957 | -0.60413 | 3.27636 | 4.24994 | 5.17824 | 11.33448 | 2011-01-01 | 2023-12-31 |
| 23:59 | B3_grouped_residual_shrinkage | baseline | all_rows | 4747 | 0.92161 | 1.18268 | 0.73722 | 0.01270 | 1.47781 | 1.96412 | 2.39546 | 4.40336 | 2011-01-01 | 2023-12-31 |
| 23:59 | B2_monthly_residual_shrinkage | baseline | all_rows | 4747 | 0.92380 | 1.18564 | 0.73676 | 0.02158 | 1.49112 | 1.97706 | 2.39705 | 4.47727 | 2011-01-01 | 2023-12-31 |
| 23:59 | B1_raw_official_latest | baseline | all_rows | 4747 | 0.92749 | 1.19152 | 0.70000 | -0.12286 | 1.50000 | 2.00000 | 2.40000 | 4.50000 | 2011-01-01 | 2023-12-31 |
| 21:00 | B3_grouped_residual_shrinkage | baseline | all_rows | 4321 | 0.95020 | 1.21293 | 0.76545 | 0.00178 | 1.52607 | 2.02145 | 2.44684 | 4.39905 | 2011-01-01 | 2023-12-31 |
| 21:00 | B2_monthly_residual_shrinkage | baseline | all_rows | 4321 | 0.95179 | 1.21564 | 0.76522 | 0.01168 | 1.52243 | 2.00879 | 2.45097 | 4.49868 | 2011-01-01 | 2023-12-31 |
| 18:00 | B3_grouped_residual_shrinkage | baseline | all_rows | 4286 | 0.95736 | 1.22070 | 0.77704 | -0.01058 | 1.53388 | 2.03184 | 2.46007 | 4.41467 | 2011-01-01 | 2023-12-31 |
| 17:00 | B3_grouped_residual_shrinkage | baseline | all_rows | 4274 | 0.95763 | 1.22120 | 0.77765 | -0.01242 | 1.53423 | 2.03103 | 2.46026 | 4.41775 | 2011-01-01 | 2023-12-31 |
| 21:00 | B1_raw_official_latest | baseline | all_rows | 4321 | 0.95800 | 1.22295 | 0.80000 | -0.12907 | 1.50000 | 2.00000 | 2.40000 | 4.50000 | 2011-01-01 | 2023-12-31 |
| 18:00 | B2_monthly_residual_shrinkage | baseline | all_rows | 4286 | 0.95880 | 1.22332 | 0.77442 | -0.00155 | 1.53855 | 2.03669 | 2.46036 | 4.53889 | 2011-01-01 | 2023-12-31 |
| 17:00 | B2_monthly_residual_shrinkage | baseline | all_rows | 4274 | 0.95909 | 1.22387 | 0.77498 | -0.00391 | 1.54070 | 2.03744 | 2.46508 | 4.54593 | 2011-01-01 | 2023-12-31 |
| 18:00 | B1_raw_official_latest | baseline | all_rows | 4286 | 0.96472 | 1.23108 | 0.80000 | -0.13047 | 1.50000 | 2.00000 | 2.50000 | 4.50000 | 2011-01-01 | 2023-12-31 |
| 17:00 | B1_raw_official_latest | baseline | all_rows | 4274 | 0.96504 | 1.23163 | 0.80000 | -0.13149 | 1.50000 | 2.00000 | 2.50000 | 4.50000 | 2011-01-01 | 2023-12-31 |
| 17:00 | B4_direct_no_official_huber | baseline_direct_fallback | all_rows | 4748 | 1.88396 | 2.36155 | 1.60545 | -0.39208 | 2.92477 | 3.81447 | 4.64320 | 10.51305 | 2011-01-01 | 2023-12-31 |
| 18:00 | B4_direct_no_official_huber | baseline_direct_fallback | all_rows | 4748 | 1.88396 | 2.36155 | 1.60545 | -0.39208 | 2.92477 | 3.81447 | 4.64320 | 10.51305 | 2011-01-01 | 2023-12-31 |
| 21:00 | B4_direct_no_official_huber | baseline_direct_fallback | all_rows | 4748 | 1.88396 | 2.36155 | 1.60545 | -0.39208 | 2.92477 | 3.81447 | 4.64320 | 10.51305 | 2011-01-01 | 2023-12-31 |
| 23:59 | B4_direct_no_official_huber | baseline_direct_fallback | all_rows | 4748 | 1.88396 | 2.36155 | 1.60545 | -0.39208 | 2.92477 | 3.81447 | 4.64320 | 10.51305 | 2011-01-01 | 2023-12-31 |

## Model Scoreboard

| cutoff | model_id | model_family | scope | n | mae | rmse | median_abs_error | bias | p80_abs_error | p90_abs_error | p95_abs_error | max_abs_error | first_date | last_date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 23:59 | M4_analog_residual | analog | official_rows_only | 4747 | 0.92161 | 1.18317 | 0.73769 | -0.03483 | 1.47893 | 1.96207 | 2.40811 | 4.40266 | 2011-01-01 | 2023-12-31 |
| 23:59 | M1_huber_residual | residual_ml | official_rows_only | 4747 | 0.92187 | 1.18337 | 0.73903 | -0.03852 | 1.48289 | 1.97426 | 2.40855 | 4.43519 | 2011-01-01 | 2023-12-31 |
| 23:59 | M3_empirical_bayes_residual | residual_shrinkage | official_rows_only | 4747 | 0.92224 | 1.18347 | 0.73497 | 0.01520 | 1.48275 | 1.96145 | 2.39446 | 4.41334 | 2011-01-01 | 2023-12-31 |
| 23:59 | M8_constrained_nonnegative_stack | stack | official_rows_only | 4747 | 0.92814 | 1.19077 | 0.73776 | -0.01704 | 1.48830 | 1.97389 | 2.42935 | 4.29835 | 2011-01-01 | 2023-12-31 |
| 23:59 | M2_elasticnet_residual | residual_ml | official_rows_only | 4747 | 0.93660 | 1.20170 | 0.75835 | 0.00169 | 1.50640 | 1.98382 | 2.46950 | 4.43324 | 2011-01-01 | 2023-12-31 |
| 23:59 | M6_lgbm_huber_residual | residual_ml | official_rows_only | 4747 | 0.93763 | 1.20282 | 0.75382 | -0.03565 | 1.50109 | 1.97834 | 2.46018 | 4.46437 | 2011-01-01 | 2023-12-31 |
| 23:59 | M5_lgbm_l1_residual | residual_ml | official_rows_only | 4747 | 0.94009 | 1.20591 | 0.75553 | -0.03739 | 1.50491 | 1.98623 | 2.46939 | 4.49858 | 2011-01-01 | 2023-12-31 |
| 23:59 | M7_high_tail_specialist | residual_ml_tail | official_rows_only | 4747 | 0.94301 | 1.20971 | 0.76004 | -0.01623 | 1.51292 | 1.99724 | 2.48279 | 4.49858 | 2011-01-01 | 2023-12-31 |
| 21:00 | M3_empirical_bayes_residual | residual_shrinkage | official_rows_only | 4321 | 0.95097 | 1.21383 | 0.76425 | 0.00680 | 1.53244 | 2.03209 | 2.44801 | 4.41340 | 2011-01-01 | 2023-12-31 |
| 21:00 | M4_analog_residual | analog | official_rows_only | 4321 | 0.95154 | 1.21446 | 0.76734 | -0.04196 | 1.53033 | 2.03142 | 2.45504 | 4.40496 | 2011-01-01 | 2023-12-31 |
| 21:00 | M1_huber_residual | residual_ml | official_rows_only | 4321 | 0.95158 | 1.21420 | 0.77206 | -0.04754 | 1.52172 | 2.01647 | 2.43787 | 4.43585 | 2011-01-01 | 2023-12-31 |
| 21:00 | M8_constrained_nonnegative_stack | stack | official_rows_only | 4321 | 0.95728 | 1.22184 | 0.76879 | -0.02612 | 1.53507 | 2.02308 | 2.47039 | 4.41276 | 2011-01-01 | 2023-12-31 |
| 18:00 | M3_empirical_bayes_residual | residual_shrinkage | official_rows_only | 4286 | 0.95805 | 1.22163 | 0.77170 | -0.00487 | 1.53892 | 2.03336 | 2.46165 | 4.43479 | 2011-01-01 | 2023-12-31 |
| 17:00 | M3_empirical_bayes_residual | residual_shrinkage | official_rows_only | 4274 | 0.95829 | 1.22213 | 0.77264 | -0.00683 | 1.53867 | 2.03488 | 2.46685 | 4.43875 | 2011-01-01 | 2023-12-31 |
| 18:00 | M4_analog_residual | analog | official_rows_only | 4286 | 0.95864 | 1.22260 | 0.78169 | -0.05031 | 1.53432 | 2.03817 | 2.46239 | 4.42401 | 2011-01-01 | 2023-12-31 |
| 18:00 | M1_huber_residual | residual_ml | official_rows_only | 4286 | 0.95869 | 1.22225 | 0.78046 | -0.05592 | 1.53072 | 2.03419 | 2.44616 | 4.45264 | 2011-01-01 | 2023-12-31 |
| 17:00 | M4_analog_residual | analog | official_rows_only | 4274 | 0.95893 | 1.22313 | 0.78052 | -0.05182 | 1.53426 | 2.03647 | 2.46486 | 4.42771 | 2011-01-01 | 2023-12-31 |
| 17:00 | M1_huber_residual | residual_ml | official_rows_only | 4274 | 0.95900 | 1.22281 | 0.78079 | -0.05747 | 1.52929 | 2.03557 | 2.44923 | 4.45596 | 2011-01-01 | 2023-12-31 |
| 18:00 | M8_constrained_nonnegative_stack | stack | official_rows_only | 4286 | 0.96432 | 1.22982 | 0.77341 | -0.03662 | 1.54345 | 2.04065 | 2.48715 | 4.42584 | 2011-01-01 | 2023-12-31 |
| 17:00 | M8_constrained_nonnegative_stack | stack | official_rows_only | 4274 | 0.96465 | 1.23030 | 0.77435 | -0.03844 | 1.54556 | 2.04017 | 2.49035 | 4.42490 | 2011-01-01 | 2023-12-31 |
| 21:00 | M2_elasticnet_residual | residual_ml | official_rows_only | 4321 | 0.96543 | 1.23283 | 0.77818 | -0.00659 | 1.54404 | 2.02707 | 2.50274 | 4.58266 | 2011-01-01 | 2023-12-31 |
| 21:00 | M6_lgbm_huber_residual | residual_ml | official_rows_only | 4321 | 0.96660 | 1.23406 | 0.77765 | -0.04477 | 1.53998 | 2.03147 | 2.50675 | 4.59940 | 2011-01-01 | 2023-12-31 |
| 21:00 | M5_lgbm_l1_residual | residual_ml | official_rows_only | 4321 | 0.96896 | 1.23713 | 0.78034 | -0.04662 | 1.54302 | 2.03338 | 2.51977 | 4.63475 | 2011-01-01 | 2023-12-31 |
| 21:00 | M7_high_tail_specialist | residual_ml_tail | official_rows_only | 4321 | 0.97137 | 1.24071 | 0.78764 | -0.02523 | 1.55577 | 2.03719 | 2.52631 | 4.63475 | 2011-01-01 | 2023-12-31 |
| 18:00 | M2_elasticnet_residual | residual_ml | official_rows_only | 4286 | 0.97219 | 1.24049 | 0.78678 | -0.01528 | 1.54974 | 2.03934 | 2.51944 | 4.59335 | 2011-01-01 | 2023-12-31 |
| 17:00 | M2_elasticnet_residual | residual_ml | official_rows_only | 4274 | 0.97255 | 1.24092 | 0.78751 | -0.01702 | 1.55052 | 2.03993 | 2.51772 | 4.59265 | 2011-01-01 | 2023-12-31 |
| 18:00 | M6_lgbm_huber_residual | residual_ml | official_rows_only | 4286 | 0.97356 | 1.24210 | 0.78744 | -0.05500 | 1.55675 | 2.04518 | 2.52283 | 4.61233 | 2011-01-01 | 2023-12-31 |
| 17:00 | M6_lgbm_huber_residual | residual_ml | official_rows_only | 4274 | 0.97398 | 1.24258 | 0.78675 | -0.05684 | 1.55605 | 2.04407 | 2.52339 | 4.61177 | 2011-01-01 | 2023-12-31 |
| 18:00 | M5_lgbm_l1_residual | residual_ml | official_rows_only | 4286 | 0.97588 | 1.24516 | 0.78484 | -0.05689 | 1.55513 | 2.04381 | 2.53057 | 4.64752 | 2011-01-01 | 2023-12-31 |
| 17:00 | M5_lgbm_l1_residual | residual_ml | official_rows_only | 4274 | 0.97632 | 1.24563 | 0.78564 | -0.05875 | 1.55441 | 2.04406 | 2.53161 | 4.64694 | 2011-01-01 | 2023-12-31 |
| 18:00 | M7_high_tail_specialist | residual_ml_tail | official_rows_only | 4286 | 0.97805 | 1.24850 | 0.79444 | -0.03565 | 1.56618 | 2.04676 | 2.54624 | 4.64752 | 2011-01-01 | 2023-12-31 |
| 17:00 | M7_high_tail_specialist | residual_ml_tail | official_rows_only | 4274 | 0.97849 | 1.24893 | 0.79500 | -0.03751 | 1.56567 | 2.04702 | 2.54392 | 4.64694 | 2011-01-01 | 2023-12-31 |
| 23:59 | M4_analog_residual | analog | all_rows | 4747 | 0.92161 | 1.18317 | 0.73769 | -0.03483 | 1.47893 | 1.96207 | 2.40811 | 4.40266 | 2011-01-01 | 2023-12-31 |
| 23:59 | M1_huber_residual | residual_ml | all_rows | 4747 | 0.92187 | 1.18337 | 0.73903 | -0.03852 | 1.48289 | 1.97426 | 2.40855 | 4.43519 | 2011-01-01 | 2023-12-31 |
| 23:59 | M3_empirical_bayes_residual | residual_shrinkage | all_rows | 4747 | 0.92224 | 1.18347 | 0.73497 | 0.01520 | 1.48275 | 1.96145 | 2.39446 | 4.41334 | 2011-01-01 | 2023-12-31 |
| 23:59 | M8_constrained_nonnegative_stack | stack | all_rows | 4747 | 0.92814 | 1.19077 | 0.73776 | -0.01704 | 1.48830 | 1.97389 | 2.42935 | 4.29835 | 2011-01-01 | 2023-12-31 |
| 23:59 | M2_elasticnet_residual | residual_ml | all_rows | 4747 | 0.93660 | 1.20170 | 0.75835 | 0.00169 | 1.50640 | 1.98382 | 2.46950 | 4.43324 | 2011-01-01 | 2023-12-31 |
| 23:59 | M6_lgbm_huber_residual | residual_ml | all_rows | 4747 | 0.93763 | 1.20282 | 0.75382 | -0.03565 | 1.50109 | 1.97834 | 2.46018 | 4.46437 | 2011-01-01 | 2023-12-31 |
| 23:59 | M5_lgbm_l1_residual | residual_ml | all_rows | 4747 | 0.94009 | 1.20591 | 0.75553 | -0.03739 | 1.50491 | 1.98623 | 2.46939 | 4.49858 | 2011-01-01 | 2023-12-31 |
| 23:59 | M7_high_tail_specialist | residual_ml_tail | all_rows | 4747 | 0.94301 | 1.20971 | 0.76004 | -0.01623 | 1.51292 | 1.99724 | 2.48279 | 4.49858 | 2011-01-01 | 2023-12-31 |
| 21:00 | M3_empirical_bayes_residual | residual_shrinkage | all_rows | 4321 | 0.95097 | 1.21383 | 0.76425 | 0.00680 | 1.53244 | 2.03209 | 2.44801 | 4.41340 | 2011-01-01 | 2023-12-31 |
| 21:00 | M4_analog_residual | analog | all_rows | 4321 | 0.95154 | 1.21446 | 0.76734 | -0.04196 | 1.53033 | 2.03142 | 2.45504 | 4.40496 | 2011-01-01 | 2023-12-31 |
| 21:00 | M1_huber_residual | residual_ml | all_rows | 4321 | 0.95158 | 1.21420 | 0.77206 | -0.04754 | 1.52172 | 2.01647 | 2.43787 | 4.43585 | 2011-01-01 | 2023-12-31 |
| 21:00 | M8_constrained_nonnegative_stack | stack | all_rows | 4321 | 0.95728 | 1.22184 | 0.76879 | -0.02612 | 1.53507 | 2.02308 | 2.47039 | 4.41276 | 2011-01-01 | 2023-12-31 |
| 18:00 | M3_empirical_bayes_residual | residual_shrinkage | all_rows | 4286 | 0.95805 | 1.22163 | 0.77170 | -0.00487 | 1.53892 | 2.03336 | 2.46165 | 4.43479 | 2011-01-01 | 2023-12-31 |
| 17:00 | M3_empirical_bayes_residual | residual_shrinkage | all_rows | 4274 | 0.95829 | 1.22213 | 0.77264 | -0.00683 | 1.53867 | 2.03488 | 2.46685 | 4.43875 | 2011-01-01 | 2023-12-31 |
| 18:00 | M4_analog_residual | analog | all_rows | 4286 | 0.95864 | 1.22260 | 0.78169 | -0.05031 | 1.53432 | 2.03817 | 2.46239 | 4.42401 | 2011-01-01 | 2023-12-31 |
| 18:00 | M1_huber_residual | residual_ml | all_rows | 4286 | 0.95869 | 1.22225 | 0.78046 | -0.05592 | 1.53072 | 2.03419 | 2.44616 | 4.45264 | 2011-01-01 | 2023-12-31 |
| 17:00 | M4_analog_residual | analog | all_rows | 4274 | 0.95893 | 1.22313 | 0.78052 | -0.05182 | 1.53426 | 2.03647 | 2.46486 | 4.42771 | 2011-01-01 | 2023-12-31 |
| 17:00 | M1_huber_residual | residual_ml | all_rows | 4274 | 0.95900 | 1.22281 | 0.78079 | -0.05747 | 1.52929 | 2.03557 | 2.44923 | 4.45596 | 2011-01-01 | 2023-12-31 |
| 18:00 | M8_constrained_nonnegative_stack | stack | all_rows | 4286 | 0.96432 | 1.22982 | 0.77341 | -0.03662 | 1.54345 | 2.04065 | 2.48715 | 4.42584 | 2011-01-01 | 2023-12-31 |
| 17:00 | M8_constrained_nonnegative_stack | stack | all_rows | 4274 | 0.96465 | 1.23030 | 0.77435 | -0.03844 | 1.54556 | 2.04017 | 2.49035 | 4.42490 | 2011-01-01 | 2023-12-31 |
| 21:00 | M2_elasticnet_residual | residual_ml | all_rows | 4321 | 0.96543 | 1.23283 | 0.77818 | -0.00659 | 1.54404 | 2.02707 | 2.50274 | 4.58266 | 2011-01-01 | 2023-12-31 |
| 21:00 | M6_lgbm_huber_residual | residual_ml | all_rows | 4321 | 0.96660 | 1.23406 | 0.77765 | -0.04477 | 1.53998 | 2.03147 | 2.50675 | 4.59940 | 2011-01-01 | 2023-12-31 |
| 21:00 | M5_lgbm_l1_residual | residual_ml | all_rows | 4321 | 0.96896 | 1.23713 | 0.78034 | -0.04662 | 1.54302 | 2.03338 | 2.51977 | 4.63475 | 2011-01-01 | 2023-12-31 |
| 21:00 | M7_high_tail_specialist | residual_ml_tail | all_rows | 4321 | 0.97137 | 1.24071 | 0.78764 | -0.02523 | 1.55577 | 2.03719 | 2.52631 | 4.63475 | 2011-01-01 | 2023-12-31 |
| 18:00 | M2_elasticnet_residual | residual_ml | all_rows | 4286 | 0.97219 | 1.24049 | 0.78678 | -0.01528 | 1.54974 | 2.03934 | 2.51944 | 4.59335 | 2011-01-01 | 2023-12-31 |
| 17:00 | M2_elasticnet_residual | residual_ml | all_rows | 4274 | 0.97255 | 1.24092 | 0.78751 | -0.01702 | 1.55052 | 2.03993 | 2.51772 | 4.59265 | 2011-01-01 | 2023-12-31 |
| 18:00 | M6_lgbm_huber_residual | residual_ml | all_rows | 4286 | 0.97356 | 1.24210 | 0.78744 | -0.05500 | 1.55675 | 2.04518 | 2.52283 | 4.61233 | 2011-01-01 | 2023-12-31 |
| 17:00 | M6_lgbm_huber_residual | residual_ml | all_rows | 4274 | 0.97398 | 1.24258 | 0.78675 | -0.05684 | 1.55605 | 2.04407 | 2.52339 | 4.61177 | 2011-01-01 | 2023-12-31 |

## Cutoff Decision

| cutoff | best_model_id | best_model_family | best_mae | best_rmse | best_bias | raw_official_mae | raw_official_rmse | delta_mae_vs_raw | delta_rmse_vs_raw | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 23:59 | B3_grouped_residual_shrinkage | baseline | 0.92161 | 1.18268 | 0.01270 | 0.92749 | 1.19152 | -0.00588 | -0.00884 | 4747 |
| 21:00 | B3_grouped_residual_shrinkage | baseline | 0.95020 | 1.21293 | 0.00178 | 0.95800 | 1.22295 | -0.00780 | -0.01002 | 4321 |
| 18:00 | B3_grouped_residual_shrinkage | baseline | 0.95736 | 1.22070 | -0.01058 | 0.96472 | 1.23108 | -0.00736 | -0.01037 | 4286 |
| 17:00 | B3_grouped_residual_shrinkage | baseline | 0.95763 | 1.22120 | -0.01242 | 0.96504 | 1.23163 | -0.00741 | -0.01042 | 4274 |

## Ablation Results

| cutoff | model_id | model_family | scope | n | mae | rmse | median_abs_error | bias | p80_abs_error | p90_abs_error | p95_abs_error | max_abs_error | first_date | last_date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 23:59 | A2_residual_shrinkage | ablation | official_rows_only | 4747 | 0.92161 | 1.18268 | 0.73722 | 0.01270 | 1.47781 | 1.96412 | 2.39546 | 4.40336 | 2011-01-01 | 2023-12-31 |
| 23:59 | A9_analog_residual | ablation | official_rows_only | 4747 | 0.92161 | 1.18317 | 0.73769 | -0.03483 | 1.47893 | 1.96207 | 2.40811 | 4.40266 | 2011-01-01 | 2023-12-31 |
| 23:59 | A3_latest_only_residual_lgbm | ablation | official_rows_only | 4747 | 0.92187 | 1.18337 | 0.73903 | -0.03852 | 1.48289 | 1.97426 | 2.40855 | 4.43519 | 2011-01-01 | 2023-12-31 |
| 23:59 | A8_latest_regime_interactions_lgbm | ablation | official_rows_only | 4747 | 0.92506 | 1.18708 | 0.74539 | -0.09204 | 1.48277 | 1.98141 | 2.41198 | 4.45159 | 2011-01-01 | 2023-12-31 |
| 23:59 | A1_raw_official | ablation | official_rows_only | 4747 | 0.92749 | 1.19152 | 0.70000 | -0.12286 | 1.50000 | 2.00000 | 2.40000 | 4.50000 | 2011-01-01 | 2023-12-31 |
| 23:59 | A10_full_stack | ablation | official_rows_only | 4747 | 0.92814 | 1.19077 | 0.73776 | -0.01704 | 1.48830 | 1.97389 | 2.42935 | 4.29835 | 2011-01-01 | 2023-12-31 |
| 23:59 | A4_latest_plus_revision_residual_lgbm | ablation | official_rows_only | 4747 | 0.93660 | 1.20170 | 0.75835 | 0.00169 | 1.50640 | 1.98382 | 2.46950 | 4.43324 | 2011-01-01 | 2023-12-31 |
| 23:59 | A6_latest_plus_target_history_lgbm | ablation | official_rows_only | 4747 | 0.93660 | 1.20170 | 0.75835 | 0.00169 | 1.50640 | 1.98382 | 2.46950 | 4.43324 | 2011-01-01 | 2023-12-31 |
| 23:59 | A7_latest_plus_climate_lgbm | ablation | official_rows_only | 4747 | 0.93763 | 1.20282 | 0.75382 | -0.03565 | 1.50109 | 1.97834 | 2.46018 | 4.46437 | 2011-01-01 | 2023-12-31 |
| 23:59 | A5_latest_plus_residual_history_lgbm | ablation | official_rows_only | 4747 | 0.94009 | 1.20591 | 0.75553 | -0.03739 | 1.50491 | 1.98623 | 2.46939 | 4.49858 | 2011-01-01 | 2023-12-31 |
| 21:00 | A2_residual_shrinkage | ablation | official_rows_only | 4321 | 0.95020 | 1.21293 | 0.76545 | 0.00178 | 1.52607 | 2.02145 | 2.44684 | 4.39905 | 2011-01-01 | 2023-12-31 |
| 21:00 | A9_analog_residual | ablation | official_rows_only | 4321 | 0.95154 | 1.21446 | 0.76734 | -0.04196 | 1.53033 | 2.03142 | 2.45504 | 4.40496 | 2011-01-01 | 2023-12-31 |
| 21:00 | A3_latest_only_residual_lgbm | ablation | official_rows_only | 4321 | 0.95158 | 1.21420 | 0.77206 | -0.04754 | 1.52172 | 2.01647 | 2.43787 | 4.43585 | 2011-01-01 | 2023-12-31 |
| 21:00 | A8_latest_regime_interactions_lgbm | ablation | official_rows_only | 4321 | 0.95527 | 1.21809 | 0.77333 | -0.10186 | 1.53365 | 2.02782 | 2.44749 | 4.45225 | 2011-01-01 | 2023-12-31 |
| 21:00 | A10_full_stack | ablation | official_rows_only | 4321 | 0.95728 | 1.22184 | 0.76879 | -0.02612 | 1.53507 | 2.02308 | 2.47039 | 4.41276 | 2011-01-01 | 2023-12-31 |
| 18:00 | A2_residual_shrinkage | ablation | official_rows_only | 4286 | 0.95736 | 1.22070 | 0.77704 | -0.01058 | 1.53388 | 2.03184 | 2.46007 | 4.41467 | 2011-01-01 | 2023-12-31 |
| 17:00 | A2_residual_shrinkage | ablation | official_rows_only | 4274 | 0.95763 | 1.22120 | 0.77765 | -0.01242 | 1.53423 | 2.03103 | 2.46026 | 4.41775 | 2011-01-01 | 2023-12-31 |
| 21:00 | A1_raw_official | ablation | official_rows_only | 4321 | 0.95800 | 1.22295 | 0.80000 | -0.12907 | 1.50000 | 2.00000 | 2.40000 | 4.50000 | 2011-01-01 | 2023-12-31 |
| 18:00 | A9_analog_residual | ablation | official_rows_only | 4286 | 0.95864 | 1.22260 | 0.78169 | -0.05031 | 1.53432 | 2.03817 | 2.46239 | 4.42401 | 2011-01-01 | 2023-12-31 |
| 18:00 | A3_latest_only_residual_lgbm | ablation | official_rows_only | 4286 | 0.95869 | 1.22225 | 0.78046 | -0.05592 | 1.53072 | 2.03419 | 2.44616 | 4.45264 | 2011-01-01 | 2023-12-31 |
| 17:00 | A9_analog_residual | ablation | official_rows_only | 4274 | 0.95893 | 1.22313 | 0.78052 | -0.05182 | 1.53426 | 2.03647 | 2.46486 | 4.42771 | 2011-01-01 | 2023-12-31 |
| 17:00 | A3_latest_only_residual_lgbm | ablation | official_rows_only | 4274 | 0.95900 | 1.22281 | 0.78079 | -0.05747 | 1.52929 | 2.03557 | 2.44923 | 4.45596 | 2011-01-01 | 2023-12-31 |
| 18:00 | A8_latest_regime_interactions_lgbm | ablation | official_rows_only | 4286 | 0.96255 | 1.22660 | 0.77843 | -0.11067 | 1.54397 | 2.03705 | 2.45630 | 4.46904 | 2011-01-01 | 2023-12-31 |
| 17:00 | A8_latest_regime_interactions_lgbm | ablation | official_rows_only | 4274 | 0.96290 | 1.22722 | 0.77941 | -0.11227 | 1.54336 | 2.03897 | 2.45717 | 4.47236 | 2011-01-01 | 2023-12-31 |
| 18:00 | A10_full_stack | ablation | official_rows_only | 4286 | 0.96432 | 1.22982 | 0.77341 | -0.03662 | 1.54345 | 2.04065 | 2.48715 | 4.42584 | 2011-01-01 | 2023-12-31 |
| 17:00 | A10_full_stack | ablation | official_rows_only | 4274 | 0.96465 | 1.23030 | 0.77435 | -0.03844 | 1.54556 | 2.04017 | 2.49035 | 4.42490 | 2011-01-01 | 2023-12-31 |
| 18:00 | A1_raw_official | ablation | official_rows_only | 4286 | 0.96472 | 1.23108 | 0.80000 | -0.13047 | 1.50000 | 2.00000 | 2.50000 | 4.50000 | 2011-01-01 | 2023-12-31 |
| 17:00 | A1_raw_official | ablation | official_rows_only | 4274 | 0.96504 | 1.23163 | 0.80000 | -0.13149 | 1.50000 | 2.00000 | 2.50000 | 4.50000 | 2011-01-01 | 2023-12-31 |
| 21:00 | A4_latest_plus_revision_residual_lgbm | ablation | official_rows_only | 4321 | 0.96543 | 1.23283 | 0.77818 | -0.00659 | 1.54404 | 2.02707 | 2.50274 | 4.58266 | 2011-01-01 | 2023-12-31 |
| 21:00 | A6_latest_plus_target_history_lgbm | ablation | official_rows_only | 4321 | 0.96543 | 1.23283 | 0.77818 | -0.00659 | 1.54404 | 2.02707 | 2.50274 | 4.58266 | 2011-01-01 | 2023-12-31 |
| 21:00 | A7_latest_plus_climate_lgbm | ablation | official_rows_only | 4321 | 0.96660 | 1.23406 | 0.77765 | -0.04477 | 1.53998 | 2.03147 | 2.50675 | 4.59940 | 2011-01-01 | 2023-12-31 |
| 21:00 | A5_latest_plus_residual_history_lgbm | ablation | official_rows_only | 4321 | 0.96896 | 1.23713 | 0.78034 | -0.04662 | 1.54302 | 2.03338 | 2.51977 | 4.63475 | 2011-01-01 | 2023-12-31 |
| 18:00 | A4_latest_plus_revision_residual_lgbm | ablation | official_rows_only | 4286 | 0.97219 | 1.24049 | 0.78678 | -0.01528 | 1.54974 | 2.03934 | 2.51944 | 4.59335 | 2011-01-01 | 2023-12-31 |
| 18:00 | A6_latest_plus_target_history_lgbm | ablation | official_rows_only | 4286 | 0.97219 | 1.24049 | 0.78678 | -0.01528 | 1.54974 | 2.03934 | 2.51944 | 4.59335 | 2011-01-01 | 2023-12-31 |
| 17:00 | A4_latest_plus_revision_residual_lgbm | ablation | official_rows_only | 4274 | 0.97255 | 1.24092 | 0.78751 | -0.01702 | 1.55052 | 2.03993 | 2.51772 | 4.59265 | 2011-01-01 | 2023-12-31 |
| 17:00 | A6_latest_plus_target_history_lgbm | ablation | official_rows_only | 4274 | 0.97255 | 1.24092 | 0.78751 | -0.01702 | 1.55052 | 2.03993 | 2.51772 | 4.59265 | 2011-01-01 | 2023-12-31 |
| 18:00 | A7_latest_plus_climate_lgbm | ablation | official_rows_only | 4286 | 0.97356 | 1.24210 | 0.78744 | -0.05500 | 1.55675 | 2.04518 | 2.52283 | 4.61233 | 2011-01-01 | 2023-12-31 |
| 17:00 | A7_latest_plus_climate_lgbm | ablation | official_rows_only | 4274 | 0.97398 | 1.24258 | 0.78675 | -0.05684 | 1.55605 | 2.04407 | 2.52339 | 4.61177 | 2011-01-01 | 2023-12-31 |
| 18:00 | A5_latest_plus_residual_history_lgbm | ablation | official_rows_only | 4286 | 0.97588 | 1.24516 | 0.78484 | -0.05689 | 1.55513 | 2.04381 | 2.53057 | 4.64752 | 2011-01-01 | 2023-12-31 |
| 17:00 | A5_latest_plus_residual_history_lgbm | ablation | official_rows_only | 4274 | 0.97632 | 1.24563 | 0.78564 | -0.05875 | 1.55441 | 2.04406 | 2.53161 | 4.64694 | 2011-01-01 | 2023-12-31 |
| 17:00 | A11_no_official_direct_fallback | ablation | official_rows_only | 4274 | 1.87873 | 2.33457 | 1.62451 | -0.61800 | 2.91347 | 3.77942 | 4.51095 | 8.88982 | 2011-01-01 | 2023-12-31 |
| 21:00 | A11_no_official_direct_fallback | ablation | official_rows_only | 4321 | 1.87917 | 2.33423 | 1.62568 | -0.61417 | 2.91328 | 3.78076 | 4.51464 | 8.88982 | 2011-01-01 | 2023-12-31 |
| 18:00 | A11_no_official_direct_fallback | ablation | official_rows_only | 4286 | 1.88032 | 2.33599 | 1.62613 | -0.61517 | 2.91432 | 3.78146 | 4.51373 | 8.88982 | 2011-01-01 | 2023-12-31 |
| 23:59 | A11_no_official_direct_fallback | ablation | official_rows_only | 4747 | 1.90225 | 2.37222 | 1.64551 | -0.55182 | 2.93369 | 3.82242 | 4.60956 | 9.97594 | 2011-01-01 | 2023-12-31 |
| 17:00 | A0_yearsafe_climatology | ablation | official_rows_only | 4274 | 2.06021 | 2.56882 | 1.76114 | -0.66499 | 3.22761 | 4.15891 | 5.06649 | 10.31686 | 2011-01-01 | 2023-12-31 |
| 18:00 | A0_yearsafe_climatology | ablation | official_rows_only | 4286 | 2.06241 | 2.57087 | 1.76258 | -0.66296 | 3.23240 | 4.17532 | 5.06683 | 10.31686 | 2011-01-01 | 2023-12-31 |
| 21:00 | A0_yearsafe_climatology | ablation | official_rows_only | 4321 | 2.06384 | 2.57209 | 1.76344 | -0.66416 | 3.23351 | 4.17627 | 5.06705 | 10.31686 | 2011-01-01 | 2023-12-31 |
| 23:59 | A0_yearsafe_climatology | ablation | official_rows_only | 4747 | 2.09679 | 2.62393 | 1.77957 | -0.60413 | 3.27636 | 4.24994 | 5.17824 | 11.33448 | 2011-01-01 | 2023-12-31 |
| 23:59 | A2_residual_shrinkage | ablation | all_rows | 4747 | 0.92161 | 1.18268 | 0.73722 | 0.01270 | 1.47781 | 1.96412 | 2.39546 | 4.40336 | 2011-01-01 | 2023-12-31 |
| 23:59 | A9_analog_residual | ablation | all_rows | 4747 | 0.92161 | 1.18317 | 0.73769 | -0.03483 | 1.47893 | 1.96207 | 2.40811 | 4.40266 | 2011-01-01 | 2023-12-31 |
| 23:59 | A3_latest_only_residual_lgbm | ablation | all_rows | 4747 | 0.92187 | 1.18337 | 0.73903 | -0.03852 | 1.48289 | 1.97426 | 2.40855 | 4.43519 | 2011-01-01 | 2023-12-31 |
| 23:59 | A8_latest_regime_interactions_lgbm | ablation | all_rows | 4747 | 0.92506 | 1.18708 | 0.74539 | -0.09204 | 1.48277 | 1.98141 | 2.41198 | 4.45159 | 2011-01-01 | 2023-12-31 |
| 23:59 | A1_raw_official | ablation | all_rows | 4747 | 0.92749 | 1.19152 | 0.70000 | -0.12286 | 1.50000 | 2.00000 | 2.40000 | 4.50000 | 2011-01-01 | 2023-12-31 |
| 23:59 | A10_full_stack | ablation | all_rows | 4747 | 0.92814 | 1.19077 | 0.73776 | -0.01704 | 1.48830 | 1.97389 | 2.42935 | 4.29835 | 2011-01-01 | 2023-12-31 |
| 23:59 | A4_latest_plus_revision_residual_lgbm | ablation | all_rows | 4747 | 0.93660 | 1.20170 | 0.75835 | 0.00169 | 1.50640 | 1.98382 | 2.46950 | 4.43324 | 2011-01-01 | 2023-12-31 |
| 23:59 | A6_latest_plus_target_history_lgbm | ablation | all_rows | 4747 | 0.93660 | 1.20170 | 0.75835 | 0.00169 | 1.50640 | 1.98382 | 2.46950 | 4.43324 | 2011-01-01 | 2023-12-31 |
| 23:59 | A7_latest_plus_climate_lgbm | ablation | all_rows | 4747 | 0.93763 | 1.20282 | 0.75382 | -0.03565 | 1.50109 | 1.97834 | 2.46018 | 4.46437 | 2011-01-01 | 2023-12-31 |
| 23:59 | A5_latest_plus_residual_history_lgbm | ablation | all_rows | 4747 | 0.94009 | 1.20591 | 0.75553 | -0.03739 | 1.50491 | 1.98623 | 2.46939 | 4.49858 | 2011-01-01 | 2023-12-31 |
| 21:00 | A2_residual_shrinkage | ablation | all_rows | 4321 | 0.95020 | 1.21293 | 0.76545 | 0.00178 | 1.52607 | 2.02145 | 2.44684 | 4.39905 | 2011-01-01 | 2023-12-31 |
| 21:00 | A9_analog_residual | ablation | all_rows | 4321 | 0.95154 | 1.21446 | 0.76734 | -0.04196 | 1.53033 | 2.03142 | 2.45504 | 4.40496 | 2011-01-01 | 2023-12-31 |

## Leakage Audit

| cutoff | audit_check | status | failed_rows | evidence |
| --- | --- | --- | --- | --- |
| 17:00 | latest_issue_at_or_before_cutoff | pass | 0 | latest_issue_at_hkt <= asof_cutoff_hkt for all official rows |
| 17:00 | no_target_tminus1_or_target_day_features | pass | 0 | target lag list begins at T-2 |
| 17:00 | no_climate_tminus1_features | pass | 0 | climate lag list begins at T-2 |
| 17:00 | development_range_excludes_2024_plus | pass | 0 | maximum target date 2023-12-31 |
| 18:00 | latest_issue_at_or_before_cutoff | pass | 0 | latest_issue_at_hkt <= asof_cutoff_hkt for all official rows |
| 18:00 | no_target_tminus1_or_target_day_features | pass | 0 | target lag list begins at T-2 |
| 18:00 | no_climate_tminus1_features | pass | 0 | climate lag list begins at T-2 |
| 18:00 | development_range_excludes_2024_plus | pass | 0 | maximum target date 2023-12-31 |
| 21:00 | latest_issue_at_or_before_cutoff | pass | 0 | latest_issue_at_hkt <= asof_cutoff_hkt for all official rows |
| 21:00 | no_target_tminus1_or_target_day_features | pass | 0 | target lag list begins at T-2 |
| 21:00 | no_climate_tminus1_features | pass | 0 | climate lag list begins at T-2 |
| 21:00 | development_range_excludes_2024_plus | pass | 0 | maximum target date 2023-12-31 |
| 23:59 | latest_issue_at_or_before_cutoff | pass | 0 | latest_issue_at_hkt <= asof_cutoff_hkt for all official rows |
| 23:59 | no_target_tminus1_or_target_day_features | pass | 0 | target lag list begins at T-2 |
| 23:59 | no_climate_tminus1_features | pass | 0 | climate lag list begins at T-2 |
| 23:59 | development_range_excludes_2024_plus | pass | 0 | maximum target date 2023-12-31 |

## Selected Model Diagnostics

Yearly:

| fold_year | scope | n | mae | rmse | median_abs_error | bias | p80_abs_error | p90_abs_error | p95_abs_error | max_abs_error | first_date | last_date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2019 | yearly | 365 | 0.83553 | 1.07455 | 0.66760 | 0.00447 | 1.38089 | 1.87262 | 2.17398 | 3.52678 | 2019-01-01 | 2019-12-31 |
| 2015 | yearly | 364 | 0.87593 | 1.12417 | 0.74023 | -0.02082 | 1.35895 | 1.73836 | 2.16497 | 4.02438 | 2015-01-01 | 2015-12-31 |
| 2014 | yearly | 365 | 0.90489 | 1.16738 | 0.70147 | -0.01795 | 1.39065 | 1.90676 | 2.45973 | 3.84275 | 2014-01-01 | 2014-12-31 |
| 2013 | yearly | 365 | 0.90606 | 1.17875 | 0.69257 | 0.03602 | 1.50555 | 1.99591 | 2.44404 | 3.82791 | 2013-01-01 | 2013-12-31 |
| 2018 | yearly | 365 | 0.90612 | 1.14788 | 0.76284 | -0.01532 | 1.39162 | 1.81150 | 2.31178 | 3.79178 | 2018-01-01 | 2018-12-31 |
| 2021 | yearly | 365 | 0.90628 | 1.16194 | 0.70075 | 0.00218 | 1.50650 | 2.02898 | 2.35505 | 3.53014 | 2021-01-01 | 2021-12-31 |
| 2023 | yearly | 365 | 0.91157 | 1.15269 | 0.80691 | 0.06190 | 1.44968 | 1.92021 | 2.31404 | 3.10903 | 2023-01-01 | 2023-12-31 |
| 2017 | yearly | 365 | 0.92499 | 1.17641 | 0.76258 | 0.08550 | 1.44047 | 1.87877 | 2.44544 | 3.91904 | 2017-01-01 | 2017-12-31 |
| 2020 | yearly | 366 | 0.93857 | 1.21626 | 0.69502 | -0.08171 | 1.50389 | 2.05000 | 2.46773 | 4.40336 | 2020-01-01 | 2020-12-31 |
| 2012 | yearly | 366 | 0.95595 | 1.22736 | 0.74896 | 0.09114 | 1.55671 | 2.06763 | 2.35431 | 4.21058 | 2012-01-01 | 2012-12-31 |
| 2011 | yearly | 365 | 0.96057 | 1.23086 | 0.76353 | 0.00512 | 1.54823 | 2.11076 | 2.47789 | 3.98917 | 2011-01-01 | 2011-12-31 |
| 2022 | yearly | 365 | 0.97215 | 1.21837 | 0.83364 | 0.01285 | 1.53537 | 1.94640 | 2.38045 | 3.71846 | 2022-01-01 | 2022-12-31 |
| 2016 | yearly | 366 | 0.98185 | 1.28283 | 0.73710 | 0.00177 | 1.68666 | 2.14537 | 2.67323 | 4.07451 | 2016-01-01 | 2016-12-31 |

Seasonal:

| season_hko | scope | n | mae | rmse | median_abs_error | bias | p80_abs_error | p90_abs_error | p95_abs_error | max_abs_error | first_date | last_date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| autumn_transition | seasonal | 403 | 0.76389 | 0.96924 | 0.64664 | -0.06045 | 1.22620 | 1.59489 | 1.89202 | 3.48205 | 2011-10-01 | 2023-10-31 |
| cool_dry | seasonal | 1562 | 0.87020 | 1.10633 | 0.70933 | 0.02377 | 1.37436 | 1.80837 | 2.26431 | 4.07451 | 2011-01-01 | 2023-12-31 |
| hot_wet | seasonal | 1989 | 0.90349 | 1.16228 | 0.72881 | -0.00180 | 1.44005 | 1.94622 | 2.35588 | 4.40336 | 2011-05-01 | 2023-09-30 |
| spring_transition | seasonal | 793 | 1.14844 | 1.44787 | 0.94090 | 0.06448 | 1.89869 | 2.35282 | 2.87637 | 4.21058 | 2011-03-01 | 2023-04-30 |

High-temperature bins:

| target_heat_bin | scope | n | mae | rmse | median_abs_error | bias | p80_abs_error | p90_abs_error | p95_abs_error | max_abs_error | first_date | last_date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32_34 | high_temp | 746 | 0.73845 | 0.96417 | 0.61504 | -0.43332 | 1.13554 | 1.61285 | 1.95821 | 4.40336 | 2011-05-10 | 2023-10-06 |
| 30_32 | high_temp | 714 | 0.80504 | 1.04156 | 0.64523 | -0.09034 | 1.32090 | 1.67995 | 2.08243 | 3.78099 | 2011-05-03 | 2023-11-06 |
| 28_30 | high_temp | 646 | 0.91389 | 1.15684 | 0.76769 | 0.15679 | 1.42985 | 1.92974 | 2.29541 | 3.84275 | 2011-04-08 | 2023-12-12 |
| lt25 | high_temp | 1769 | 0.97558 | 1.24310 | 0.78418 | 0.22393 | 1.57471 | 2.09705 | 2.51102 | 4.07451 | 2011-01-01 | 2023-12-30 |
| 25_28 | high_temp | 738 | 1.05797 | 1.34423 | 0.83535 | 0.11602 | 1.72721 | 2.28127 | 2.68360 | 4.21058 | 2011-02-28 | 2023-12-31 |
| ge34 | high_temp | 134 | 1.13606 | 1.33824 | 1.14409 | -1.00731 | 1.64154 | 2.01848 | 2.34244 | 3.46214 | 2011-06-09 | 2023-10-05 |

Weather regimes:

| dominant_regime | scope | n | mae | rmse | median_abs_error | bias | p80_abs_error | p90_abs_error | p95_abs_error | max_abs_error | first_date | last_date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tropical_cyclone_proxy_regime | weather_regime | 128 | 0.75847 | 0.93740 | 0.64043 | 0.02589 | 1.09870 | 1.46798 | 1.88913 | 2.75466 | 2011-05-09 | 2022-10-19 |
| hot_humid_persistence | weather_regime | 744 | 0.87909 | 1.14130 | 0.67307 | 0.06589 | 1.40552 | 1.90675 | 2.33511 | 4.40336 | 2011-04-19 | 2023-12-17 |
| high_forecast_uncertainty_regime | weather_regime | 936 | 0.88962 | 1.14153 | 0.72479 | -0.05751 | 1.42056 | 1.91662 | 2.37843 | 3.83752 | 2011-01-01 | 2023-12-31 |
| cloud_rain_suppressed | weather_regime | 2789 | 0.94795 | 1.21372 | 0.77083 | 0.03443 | 1.53247 | 2.02259 | 2.46293 | 4.21058 | 2011-01-06 | 2023-12-25 |
| marine_moderation_regime | weather_regime | 150 | 0.98158 | 1.23807 | 0.78122 | -0.22815 | 1.71400 | 2.07079 | 2.32443 | 3.65957 | 2011-01-03 | 2023-03-01 |

Boundary bins:

| boundary_bin | scope | n | mae | rmse | median_abs_error | bias | p80_abs_error | p90_abs_error | p95_abs_error | max_abs_error | first_date | last_date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| within_0.05C_halfdeg | boundary | 4747 | 0.92161 | 1.18268 | 0.73722 | 0.01270 | 1.47781 | 1.96412 | 2.39546 | 4.40336 | 2011-01-01 | 2023-12-31 |

## Implementation Notes

The implemented system is a hybrid official-anchor residual ensemble. It uses the HKO lead-1 forecast max as the core anchor, then learns residual correction from only historical, cutoff-valid data. The stack is constrained to nonnegative weights summing to one with a tiny intercept bound, so it cannot become an unconstrained black-box extrapolator. Direct no-official models are retained as fallbacks for rows without an official forecast before cutoff, but official-row MAE/RMSE is the primary model-selection view because the raw official anchor is the core competitive edge.
