# EXP-0039 / HKG-T24-R07 Long-Form Experiment Report

## Purpose

R07 tests whether pressure tendency, temperature and dew-point decline, and wind-speed changes can identify transition regimes that produce the largest HKG T-24 Tmax forecast errors. The hypothesis is not that pressure level alone predicts temperature. Pressure level is strongly seasonal and can be redundant with calendar. The useful information should come from changes: rising pressure, cooling, drying, wind increase, and combined surface evidence of fronts or cold surges before the T-1 15:00 cutoff.

## Data Used

The feature backbone is the R06 pre-validation feature matrix, which itself is built from R04 cutoff-safe target-station thermal features and immutable high-frequency temperature/humidity snapshots. R07 adds HKO pressure candidates from `C:\hkg_tmax_data\silver\features\t24_cutoff_feature_candidates.parquet` and wind speed/gust summaries from `C:\hkg_tmax_data\bronze\analysis_phase_a\hko_high_frequency_selected_station_observations.parquet`. The target-date feature period is `2020-07-02` through `2023-12-31`, and the OOF prediction period is `2021-07-01` through `2023-12-31`.

## Feature Construction

The experiment constructs HKO MSLP level, 3-hour pressure tendency, 24-hour and 48-hour pressure changes, pressure acceleration, six-hour temperature decline, six-hour dew-point decline, median network wind speed and gust, three-hour wind-speed changes, and four fixed-scale transition scores: cold-surge score, front score, warm-sector score, and post-frontal score. The scores use fixed meteorological scaling constants rather than fit-on-full-sample z-scores, so the feature construction does not import future distributional information. Target-side daily Tmax change is used only as an auxiliary training label and subgroup diagnostic, not as an inference-time predictor.

## Missing Inputs and Blockers

The uploaded specification asks for 12-station pressure gradients, wind-direction shifts, robust plane-fit pressure gradients, and gradient-vector rotation. Those are not fully available in the current parsed T24 tables. The current R07 implementation uses HKO pressure and parsed network wind speed/gust. Wind direction and pressure-network gradients remain explicit blockers for a later richer R07/R08 extension unless the raw parsers are expanded. This is documented rather than silently pretending that speed-only wind captures direction shifts.

## Model Ladder

R07 runs a baseline temperature/calendar model, pressure-only Elastic Net, temperature/dew-point transition Elastic Net, wind-only Elastic Net, combined transition Elastic Net, shallow constrained gradient boosting, a two-stage transition-probability residual specialist, and a month-permuted transition negative control. The two-stage model trains a logistic transition classifier inside each fold using only training rows and target-side transition labels, then feeds its fold-specific transition probability into a ridge residual model. No classifier is fit on validation or locked-test data.

## Leakage Controls

All ordinary predictors are available before T-1 15:00 HKT under their inherited conservative availability rules. The feature matrix and predictions are guarded against dates from 2025-01-01 onward. Validation 2024 is not used. Target-side transition labels are derived only for training labels and diagnostics inside development folds; they are never included as direct predictors. Preprocessing, logistic classification, scaling, imputation, boosting, and regression are all fit inside chronological training folds.

## OOF Gate

The strict four-year OOF check is `BLOCKED`: R07 modern transition-detection pre-validation feature period: 3.50 years available, requires at least 4.0 years. Therefore R07 is a completed transition diagnostic but not promotable under the user's hard four-year OOF rule. Even a positive transition specialist would require longer eligible development history or a revised predeclared evaluation design before promotion.

## Main Result

The best non-control diagnostic model by OOF MAE is `r07_baseline_temp_calendar` with MAE `1.4723` C, RMSE `1.8861` C, bias `0.0298` C, and CRPS `1.0512` over `911` rows. The fold-delta table shows whether any transition model improves across chronological folds or only in isolated periods. The subgroup scorecard separates target-side cold drops, warm jumps, high front-score days, and ordinary days.

## Interpretation

A useful transition experiment should improve the high-error transition cohort without damaging ordinary days. If pressure-only wins, HKO pressure tendency is already carrying meaningful air-mass change information. If temp/dew decline wins, the station thermal/moisture trajectory is sufficient and pressure is redundant. If wind-only wins, network flow speed is a useful proxy even without direction. If the two-stage specialist wins only on target-side transition days but loses ordinary days, it should become a gated expert rather than a universal model. If all transition candidates lose to baseline, then the available parsed transition variables are too sparse or too incomplete, and the correct next step is parser expansion rather than tuning.

## Decision Record

R07 is complete as an auditable diagnostic once its artifacts and tests pass. The result is retained whether positive, conditional, or null. Current blockers are wind direction, pressure-network gradients, and longer than 3.5 years of modern pre-validation OOF coverage. The next planned experiment is R08 surface wind, advection, and sea-breeze regime, where direction-aware parsing should be prioritized if the raw feed preserves direction columns.

## Actual Diagnostic Disposition

The generated scoreboard is intentionally not optimized after the fact. In this run the simple temperature/calendar baseline remains the best non-control model. The month-permuted transition control sits close to the baseline, while the physically motivated transition models generally lose. That combination says the available transition variables are not yet strong enough in their current parsed form. It also says the project should be careful about any pressure or wind improvement that is only a seasonal artifact. The pressure-only model performs especially poorly because the available HKO pressure sample is shorter and patchier than the thermal baseline sample; fold-local filtering prevents all-null columns from breaking the model, but it cannot invent missing pressure history.

## Why The Null Result Matters

This null result directly answers a high-priority failure mechanism question. The transition hypothesis is meteorologically plausible, especially for winter and spring cold surges, but the current operational feature representation is incomplete. HKO pressure tendency, speed-only network wind, and HKO cooling/drying do not yet beat the baseline in broad chronological OOF. This shifts expected information gain toward parser expansion: wind direction, pressure-network gradients, regional ISD pressure/wind, and upper-air coupling should be added before concluding that transition detection itself is unhelpful. The result is therefore `diagnostic null with parser blockers`, not a scientific rejection of front/cold-surge physics.

## Carry-Forward Rules

Later experiments may reuse the fixed-scale `front_score`, `cold_surge_score`, and target-side transition labels only under strict separation. The scores are legal predictors because they use pre-cutoff pressure, wind speed, temperature, and dew-point changes. The target-side labels are not legal predictors; they are labels for fold-local specialists or evaluation subgroups. If R20 or R22 builds a transition specialist, it must train the transition classifier inside each fold exactly as R07 does or with an equally strict fold-local design. If R08 parses wind direction, it should rerun the transition scorecard rather than assuming speed-only wind was an adequate proxy.

## Acceptance Outcome

R07 does not meet the feature-family promotion rule. It does not improve overall OOF MAE by 0.03 C, and it does not yet prove the required top-decile baseline-error improvement without ordinary-day harm. It is also blocked by the strict four-year OOF sample rule. The correct conclusion is to retain all artifacts and move on to R08, not to tune pressure thresholds against these same folds.

# R07 Machine-Readable Summary Tables

Generated: `2026-06-20T09:47:45.800431Z`

## Overall Scoreboard

| model_id | n | first_date | last_date | mae | rmse | median_abs_error | bias | crps_normal | coverage_80 | coverage_90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r07_baseline_temp_calendar | 911 | 2021-07-01 | 2023-12-31 | 1.472338 | 1.886078 | 1.216189 | 0.029801 | 1.051187 | 0.829857 | 0.908891 |
| r07_month_permuted_transition_control | 911 | 2021-07-01 | 2023-12-31 | 1.477745 | 1.889940 | 1.220211 | 0.017329 | 1.054340 | 0.825467 | 0.909989 |
| r07_temp_dew_transition_elastic_net | 911 | 2021-07-01 | 2023-12-31 | 1.515148 | 1.917111 | 1.307642 | -0.051844 | 1.072780 | 0.819978 | 0.901207 |
| r07_combined_transition_elastic_net | 911 | 2021-07-01 | 2023-12-31 | 1.588381 | 2.005563 | 1.333140 | -0.075706 | 1.121162 | 0.788145 | 0.889133 |
| r07_shallow_boosting_transition | 911 | 2021-07-01 | 2023-12-31 | 1.706575 | 2.154620 | 1.471040 | -0.122391 | 1.210219 | 0.725576 | 0.851811 |
| r07_wind_only_elastic_net | 911 | 2021-07-01 | 2023-12-31 | 1.820006 | 2.536944 | 1.359938 | -0.637331 | 1.338757 | 0.756312 | 0.861690 |
| r07_two_stage_transition_probability_residual | 911 | 2021-07-01 | 2023-12-31 | 1.961215 | 2.727297 | 1.403020 | 0.223591 | 1.453277 | 0.732162 | 0.824369 |
| r07_pressure_only_elastic_net | 911 | 2021-07-01 | 2023-12-31 | 2.741089 | 4.492374 | 1.523501 | 1.390313 | 2.210738 | 0.694841 | 0.776070 |

## Fold Deltas

| fold_id | model_id | n | mae | baseline_mae | mae_improvement_vs_baseline | crps_improvement_vs_baseline |
| --- | --- | --- | --- | --- | --- | --- |
| fold_2022_h2 | r07_baseline_temp_calendar | 182 | 1.273130 | 1.273130 | 0.000000 | 0.000000 |
| fold_2022_h2 | r07_month_permuted_transition_control | 182 | 1.274740 | 1.273130 | -0.001610 | 0.000584 |
| fold_2022_h2 | r07_wind_only_elastic_net | 182 | 1.275674 | 1.273130 | -0.002544 | 0.010871 |
| fold_2023_h2 | r07_wind_only_elastic_net | 184 | 1.284144 | 1.326541 | 0.042397 | 0.025482 |
| fold_2023_h2 | r07_month_permuted_transition_control | 184 | 1.325749 | 1.326541 | 0.000791 | -0.000990 |
| fold_2023_h2 | r07_two_stage_transition_probability_residual | 184 | 1.326058 | 1.326541 | 0.000483 | 0.019504 |
| fold_2023_h2 | r07_baseline_temp_calendar | 184 | 1.326541 | 1.326541 | 0.000000 | 0.000000 |
| fold_2023_h2 | r07_pressure_only_elastic_net | 184 | 1.331174 | 1.326541 | -0.004634 | -0.005751 |
| fold_2023_h2 | r07_temp_dew_transition_elastic_net | 184 | 1.340595 | 1.326541 | -0.014054 | -0.009237 |
| fold_2023_h2 | r07_combined_transition_elastic_net | 184 | 1.343010 | 1.326541 | -0.016469 | 0.002920 |
| fold_2022_h2 | r07_pressure_only_elastic_net | 182 | 1.366097 | 1.273130 | -0.092968 | -0.049666 |
| fold_2022_h2 | r07_temp_dew_transition_elastic_net | 182 | 1.368633 | 1.273130 | -0.095503 | -0.034145 |
| fold_2023_h1 | r07_pressure_only_elastic_net | 181 | 1.416634 | 1.482125 | 0.065491 | 0.048190 |
| fold_2023_h1 | r07_temp_dew_transition_elastic_net | 181 | 1.440574 | 1.482125 | 0.041551 | 0.030864 |
| fold_2023_h1 | r07_combined_transition_elastic_net | 181 | 1.443974 | 1.482125 | 0.038151 | 0.038616 |
| fold_2023_h1 | r07_two_stage_transition_probability_residual | 181 | 1.465395 | 1.482125 | 0.016730 | 0.029554 |
| fold_2023_h1 | r07_baseline_temp_calendar | 181 | 1.482125 | 1.482125 | 0.000000 | 0.000000 |
| fold_2023_h2 | r07_shallow_boosting_transition | 184 | 1.484516 | 1.326541 | -0.157975 | -0.083377 |
| fold_2022_h2 | r07_combined_transition_elastic_net | 182 | 1.485652 | 1.273130 | -0.212522 | -0.096965 |
| fold_2023_h1 | r07_month_permuted_transition_control | 181 | 1.487454 | 1.482125 | -0.005329 | -0.003945 |
| fold_2022_h2 | r07_two_stage_transition_probability_residual | 182 | 1.502902 | 1.273130 | -0.229772 | -0.109712 |
| fold_2023_h1 | r07_shallow_boosting_transition | 181 | 1.504237 | 1.482125 | -0.022112 | -0.020055 |
| fold_2023_h1 | r07_wind_only_elastic_net | 181 | 1.509860 | 1.482125 | -0.027735 | -0.014058 |
| fold_2021_h2 | r07_two_stage_transition_probability_residual | 183 | 1.545337 | 1.588463 | 0.043126 | 0.021649 |
| fold_2021_h2 | r07_temp_dew_transition_elastic_net | 183 | 1.553984 | 1.588463 | 0.034479 | 0.017111 |
| fold_2021_h2 | r07_combined_transition_elastic_net | 183 | 1.556969 | 1.588463 | 0.031494 | 0.016321 |
| fold_2021_h2 | r07_baseline_temp_calendar | 183 | 1.588463 | 1.588463 | 0.000000 | 0.000000 |
| fold_2021_h2 | r07_month_permuted_transition_control | 183 | 1.606278 | 1.588463 | -0.017815 | -0.007441 |
| fold_2021_h2 | r07_pressure_only_elastic_net | 183 | 1.618443 | 1.588463 | -0.029980 | -0.014726 |
| fold_2021_h2 | r07_wind_only_elastic_net | 183 | 1.618443 | 1.588463 | -0.029980 | -0.014726 |
| fold_2022_h1 | r07_baseline_temp_calendar | 181 | 1.693665 | 1.693665 | 0.000000 | 0.000000 |
| fold_2022_h1 | r07_month_permuted_transition_control | 181 | 1.696726 | 1.693665 | -0.003061 | -0.003981 |
| fold_2021_h2 | r07_shallow_boosting_transition | 183 | 1.716135 | 1.588463 | -0.127673 | -0.087398 |
| fold_2022_h2 | r07_shallow_boosting_transition | 182 | 1.776774 | 1.273130 | -0.503645 | -0.312849 |
| fold_2022_h1 | r07_temp_dew_transition_elastic_net | 181 | 1.875227 | 1.693665 | -0.181562 | -0.113120 |
| fold_2022_h1 | r07_shallow_boosting_transition | 181 | 2.054398 | 1.693665 | -0.360733 | -0.292676 |
| fold_2022_h1 | r07_combined_transition_elastic_net | 181 | 2.117281 | 1.693665 | -0.423616 | -0.312779 |
| fold_2022_h1 | r07_wind_only_elastic_net | 181 | 3.426025 | 1.693665 | -1.732360 | -1.455275 |
| fold_2022_h1 | r07_two_stage_transition_probability_residual | 181 | 3.984040 | 1.693665 | -2.290375 | -1.984733 |
| fold_2022_h1 | r07_pressure_only_elastic_net | 181 | 8.016467 | 1.693665 | -6.322802 | -5.813707 |

## Transition Subgroups

| model_id | transition_regime | n | mae | rmse | crps_normal |
| --- | --- | --- | --- | --- | --- |
| r07_baseline_temp_calendar | predictor_top_decile_front_score | 66 | 0.976963 | 1.228737 | 0.735659 |
| r07_month_permuted_transition_control | predictor_top_decile_front_score | 66 | 0.985975 | 1.231509 | 0.737356 |
| r07_baseline_temp_calendar | ordinary | 590 | 1.080771 | 1.339629 | 0.786998 |
| r07_month_permuted_transition_control | ordinary | 590 | 1.085297 | 1.345512 | 0.789742 |
| r07_temp_dew_transition_elastic_net | ordinary | 590 | 1.133894 | 1.409203 | 0.815464 |
| r07_combined_transition_elastic_net | ordinary | 590 | 1.207670 | 1.497900 | 0.854217 |
| r07_temp_dew_transition_elastic_net | predictor_top_decile_front_score | 66 | 1.224201 | 1.422513 | 0.833418 |
| r07_combined_transition_elastic_net | predictor_top_decile_front_score | 66 | 1.328072 | 1.599119 | 0.916620 |
| r07_shallow_boosting_transition | predictor_top_decile_front_score | 66 | 1.344448 | 1.698292 | 0.960777 |
| r07_wind_only_elastic_net | ordinary | 590 | 1.397098 | 1.939840 | 1.027223 |
| r07_wind_only_elastic_net | predictor_top_decile_front_score | 66 | 1.443280 | 2.001284 | 1.050585 |
| r07_shallow_boosting_transition | ordinary | 590 | 1.476140 | 1.836394 | 1.036244 |
| r07_two_stage_transition_probability_residual | ordinary | 590 | 1.575213 | 2.286228 | 1.166935 |
| r07_two_stage_transition_probability_residual | predictor_top_decile_front_score | 66 | 1.616057 | 2.171585 | 1.174464 |
| r07_shallow_boosting_transition | target_side_warm_jump | 118 | 2.015582 | 2.401957 | 1.406642 |
| r07_pressure_only_elastic_net | predictor_top_decile_front_score | 66 | 2.106395 | 3.127443 | 1.615971 |
| r07_baseline_temp_calendar | target_side_warm_jump | 118 | 2.206471 | 2.467497 | 1.489376 |
| r07_month_permuted_transition_control | target_side_warm_jump | 118 | 2.225572 | 2.475838 | 1.499485 |
| r07_temp_dew_transition_elastic_net | target_side_warm_jump | 118 | 2.246861 | 2.526188 | 1.532292 |
| r07_combined_transition_elastic_net | target_side_warm_jump | 118 | 2.384140 | 2.749231 | 1.664465 |
| r07_pressure_only_elastic_net | ordinary | 590 | 2.412232 | 4.386421 | 1.990977 |
| r07_wind_only_elastic_net | target_side_cold_drop | 137 | 2.545384 | 2.948396 | 1.802548 |
| r07_shallow_boosting_transition | target_side_cold_drop | 137 | 2.607261 | 3.160416 | 1.910442 |
| r07_temp_dew_transition_elastic_net | target_side_cold_drop | 137 | 2.666975 | 3.068528 | 1.900459 |
| r07_combined_transition_elastic_net | target_side_cold_drop | 137 | 2.667947 | 3.056478 | 1.901360 |
| r07_two_stage_transition_probability_residual | target_side_warm_jump | 118 | 2.673012 | 3.299582 | 1.945240 |
| r07_month_permuted_transition_control | target_side_cold_drop | 137 | 2.760649 | 3.153526 | 1.963149 |
| r07_baseline_temp_calendar | target_side_cold_drop | 137 | 2.764975 | 3.155089 | 1.963524 |
| r07_pressure_only_elastic_net | target_side_warm_jump | 118 | 2.774731 | 3.383529 | 2.010210 |
| r07_two_stage_transition_probability_residual | target_side_cold_drop | 137 | 3.176762 | 3.911787 | 2.397016 |
| r07_wind_only_elastic_net | target_side_warm_jump | 118 | 3.303082 | 4.305917 | 2.519144 |
| r07_pressure_only_elastic_net | target_side_cold_drop | 137 | 4.434123 | 6.063433 | 3.616402 |

## Transition Diagnostics

| feature | n | corr_with_next_tmax_change | mean | p10 | p90 |
| --- | --- | --- | --- | --- | --- |
| hko_mslp_3h_change_to_cutoff_hpa | 725 | -0.081972 | -1.918069 | -2.700000 | -1.000000 |
| hko_mslp_24h_change_hpa | 725 | -0.114621 | -0.006897 | -2.400000 | 2.600000 |
| hko_temp_decline_6h_c | 1050 | 0.080325 | -2.696571 | -5.000000 | -0.400000 |
| hko_dew_point_decline_6h_c | 1034 | -0.130299 | -0.317455 | -1.870142 | 1.081805 |
| network_median_wind_speed_3h_change_kmh | 728 | -0.076183 | 0.217720 | -4.000000 | 4.000000 |
| cold_surge_score | 1272 | -0.028328 | -1.593953 | -3.356218 | 0.000000 |
| front_score | 1272 | -0.073591 | 2.064579 | 0.000000 | 3.725817 |
| warm_sector_score | 1272 | 0.018692 | 1.606413 | 0.000000 | 3.318957 |
| post_frontal_score | 1272 | -0.108614 | 0.798829 | 0.000000 | 1.440917 |
