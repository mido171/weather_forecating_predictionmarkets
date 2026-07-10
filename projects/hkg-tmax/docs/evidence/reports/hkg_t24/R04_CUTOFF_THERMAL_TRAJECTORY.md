# EXP-0036 / HKG-T24-R04 Long-Form Experiment Report

## Purpose

R04 tests whether the shape of the HKO Headquarters temperature trajectory up to the T-1 15:00 HKT operational cutoff contains next-day official Tmax information beyond the latest eligible temperature snapshot and deterministic calendar geometry. It is deliberately limited to HKO Headquarters data and does not include neighboring stations, upper-air data, NWP, radar, satellite, Polymarket data, validation 2024 outcomes, or locked-test rows. The experiment is built as a leakage-safe diagnostic because the available modern high-frequency archive before validation 2024 is too short for the user's strict four-year OOF requirement.

## Cutoff Contract

For target local date T, the origin date is T-1. The cutoff is T-1 15:00:00 Asia/Hong_Kong. The historical HKO high-frequency archive is replayed with the conservative latency rule used elsewhere in the project: `available_at = observed_at + 20 minutes`. Therefore the latest ordinary observation eligible at the cutoff is 14:40 HKT on the origin date. R04 enforces this by building every row only from observations whose observed time is less than or equal to 14:40 on T-1. If a feature row ever attempts to use a later observation, the script raises an error.

## Data Used

The input table is `C:\hkg_tmax_data\bronze\hkg_t24\r03_hko_hq_full_day_high_frequency.parquet`, which was created in R03 by parsing full-day HKO Headquarters rows from immutable raw DATA.GOV ZIP payloads. R04 uses only origin-day observations through the cutoff. It uses official HKO daily Tmax labels from `C:\hkg_tmax_data\silver\targets\hko_daily_tmax.parquet` for target dates from 2020-07-02 through 2023-12-31. The feature matrix has 1275 rows and 58 columns. Validation 2024 is not read. The locked test is not read.

## Features Constructed

The feature matrix includes the latest eligible HKO temperature, observation age at cutoff, deterministic day-of-year sine/cosine, approximate day length, approximate noon solar elevation, snapshots at 00:00, 03:00, 06:00, 09:00, 12:00, 13:00, and 14:00, current-minus-snapshot differences, robust temperature changes and slopes over 30 minutes, 1 hour, 3 hours, 6 hours, and 12 hours, acceleration between 3-hour and 6-hour slopes, max/min/range/std so far, current-minus-max and current-minus-min, time of max/min so far, first sustained positive-slope minute, trailing non-warming duration before cutoff, and since-midnight max/min/range values available by cutoff. All of these are origin-day T-1 values available at or before the cutoff; no target-day T observation is used.

## Models and Ablations

R04 uses a deliberately small predeclared diagnostic ladder. The baseline model uses latest eligible temperature, calendar seasonality, observation age, and deterministic solar geometry. The core trajectory model uses all numeric trajectory features. A no-since-midnight ablation removes the running max/min feed to test whether the raw temperature curve alone carries value. A shape-only model keeps curve-shape variables and calendar controls but removes most level/state variables. Every model is a Ridge regression with fixed alpha 1.0, median imputation, and standardization fitted inside each chronological training fold. There is no random split and no uncontrolled hyperparameter search.

## Chronological OOF Design

The chronological folds are half-year test windows from 2021-H2 through 2023-H2. Each fold trains only on target dates earlier than the fold test window and predicts the fold test dates. The rows are out of fold for those diagnostic windows, but the total pre-validation OOF span is still below the user's four-year requirement. R04 therefore cannot promote a trajectory feature family even if a metric improves. It can only record evidence, ablations, and blockers.

## Four-Year Gate

The strict four-year OOF feasibility check is `BLOCKED`: R04 modern HKO thermal trajectory pre-validation feature period: 3.50 years available, requires at least 4.0 years. This is the controlling status for R04. The available pre-validation high-frequency era is long enough to generate useful diagnostics, but not long enough to satisfy the acceptance criterion for a promotable modern high-frequency experiment. The experiment status is therefore `COMPLETE_DIAGNOSTIC_OOF_BLOCKED`, not `PASS` and not a final challenger.

## Main Result

The best diagnostic model by OOF MAE is `r04_baseline_latest_temp_calendar` with MAE `1.4723` C, RMSE `1.8861` C, bias `0.0298` C, and CRPS `1.0512` over `911` OOF rows. This result is not comparable to the frozen validation-2024 champion as a promotion candidate because it does not touch validation 2024 and it fails the strict four-year OOF gate. It is, however, useful evidence about whether trajectory shape is likely worth carrying forward after the data-length issue is solved.

The direction of the diagnostic result is conservative: the latest-temperature/calendar baseline is the best overall model. The richer trajectory models do not show a stable improvement in this short development sample. In the fold-level table, isolated fold gains are allowed to exist, but they are not enough. R04 requires stable gains in at least three chronological folds for even exploratory promotion, and the available output does not meet that standard. This is a negative or null result for unconstrained target-station trajectory enrichment, not a failure of the leakage-safe framework.

## Interpretation

If a trajectory model beats the latest-temperature baseline inside the blocked development folds, the result should be treated as a conditional research signal rather than accepted skill. If it fails to beat the baseline, that is still informative: it means the latest eligible temperature and calendar geometry may already summarize most of the target-station thermal state for the short modern sample, or that the public snapshot cadence is too sparse to expose meaningful within-day curve shape before cutoff. Either result helps prioritize later work without contaminating validation.

The most important practical lesson is that more features are not automatically better. A flexible trajectory model can become unstable when the modern history is short, when since-midnight running extrema have reset/carryover quirks, or when several slope and snapshot variables are strongly collinear. The R04 output therefore argues for restraint: keep the latest-temperature/calendar baseline as the default target-station thermal expert, carry only narrowly justified trajectory summaries into later experiments, and require stronger evidence before adding high-dimensional curve-shape blocks to a final architecture.

The null result is still useful for model design. It suggests that future effort should focus on conditional interactions rather than adding the entire trajectory block wholesale. Candidate conditional uses include transition days, days with rapid morning heating, days where the latest temperature is well below the max-so-far, and days with long non-warming duration before cutoff. Those are hypotheses for later gated specialists, not accepted predictors here. They must be tested with fold-safe subgroup definitions and enough cases, and they remain blocked under the strict four-year rule until the modern sample length problem is solved.

## Leakage Review

No feature column is target-day T data. The origin-date rows are T-1 only, and the latest ordinary observation is capped at 14:40 because of the +20 minute conservative availability rule. The target label is used only for training labels and OOF scoring. Imputation, scaling, and Ridge coefficients are fitted separately inside each fold using training rows only. R04 does not fit any transformation on validation 2024 or locked-test data. The script calls the locked-test guard on targets, feature rows, and predictions.

## Artifacts

The feature matrix is stored at `C:\hkg_tmax_data\gold\hkg_t24\r04_thermal_trajectory\r04_feature_matrix.parquet`. OOF predictions are stored at `C:\hkg_tmax_data\gold\hkg_t24\r04_thermal_trajectory\r04_oof_predictions.parquet` and copied into the experiment folder. Scoreboards, fold deltas, and feature correlations are written in both parquet/CSV forms. The repo-level report is `reports/hkg_t24/R04_CUTOFF_THERMAL_TRAJECTORY.md`. The reproduction command is in `REPRODUCE.md`.

## Downstream Rule

R04 does not authorize validation access, locked-test access, or model promotion. The only safe downstream use is to add its feature families to the evidence registry as `OOF_BLOCKED_DIAGNOSTIC` unless the modern high-frequency four-year issue is resolved. Later model experiments may reuse the feature builder, but they must preserve the 14:40 latest-observation cap and fold-local preprocessing. R30 cannot use R04 as a promoted component unless a predeclared final architecture is frozen without adaptive validation feedback.

The next experiment, R05, should test multi-day memory with the same level of discipline. It should avoid treating lagged official daily labels as operationally available unless publication timing is proven, and it should keep separate versions with and without lagged official labels. If R05 relies only on HKO high-frequency trajectories, it inherits the same four-year OOF blocker. If it uses long target history, it can satisfy the four-year requirement but must be reported separately as a target-history or silver replay diagnostic rather than a fully operational high-frequency model.

# R04 Machine-Readable Summary Tables

Generated: `2026-06-20T09:15:15.841987Z`

## Overall Scoreboard

| model_id | n | first_date | last_date | mae | rmse | median_abs_error | bias | crps_normal | coverage_80 | coverage_90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r04_baseline_latest_temp_calendar | 911 | 2021-07-01 | 2023-12-31 | 1.472338 | 1.886078 | 1.216189 | 0.029801 | 1.051187 | 0.829857 | 0.908891 |
| r04_shape_only | 911 | 2021-07-01 | 2023-12-31 | 1.520947 | 1.965402 | 1.242431 | 0.167843 | 1.088231 | 0.803513 | 0.897914 |
| r04_trajectory_no_since_midnight | 911 | 2021-07-01 | 2023-12-31 | 1.545664 | 1.998116 | 1.271208 | 0.195818 | 1.106238 | 0.790340 | 0.890231 |
| r04_trajectory_core | 911 | 2021-07-01 | 2023-12-31 | 2.477144 | 3.954577 | 1.426126 | -1.072500 | 1.975576 | 0.709111 | 0.780461 |

## Fold Score Deltas

| fold_id | model_id | n | mae | baseline_mae | mae_improvement_vs_baseline | crps_improvement_vs_baseline |
| --- | --- | --- | --- | --- | --- | --- |
| fold_2022_h2 | r04_trajectory_core | 182 | 1.254977 | 1.273130 | 0.018152 | -0.000290 |
| fold_2022_h2 | r04_baseline_latest_temp_calendar | 182 | 1.273130 | 1.273130 | 0.000000 | 0.000000 |
| fold_2022_h2 | r04_shape_only | 182 | 1.282293 | 1.273130 | -0.009164 | -0.006860 |
| fold_2022_h2 | r04_trajectory_no_since_midnight | 182 | 1.316975 | 1.273130 | -0.043846 | -0.033111 |
| fold_2023_h2 | r04_shape_only | 184 | 1.324096 | 1.326541 | 0.002445 | 0.004809 |
| fold_2023_h2 | r04_baseline_latest_temp_calendar | 184 | 1.326541 | 1.326541 | 0.000000 | 0.000000 |
| fold_2023_h2 | r04_trajectory_no_since_midnight | 184 | 1.328237 | 1.326541 | -0.001696 | 0.001436 |
| fold_2023_h2 | r04_trajectory_core | 184 | 1.353524 | 1.326541 | -0.026984 | -0.011828 |
| fold_2023_h1 | r04_baseline_latest_temp_calendar | 181 | 1.482125 | 1.482125 | 0.000000 | 0.000000 |
| fold_2023_h1 | r04_shape_only | 181 | 1.548219 | 1.482125 | -0.066094 | -0.039376 |
| fold_2023_h1 | r04_trajectory_no_since_midnight | 181 | 1.566394 | 1.482125 | -0.084269 | -0.048542 |
| fold_2021_h2 | r04_baseline_latest_temp_calendar | 183 | 1.588463 | 1.588463 | 0.000000 | 0.000000 |
| fold_2021_h2 | r04_shape_only | 183 | 1.669657 | 1.588463 | -0.081194 | -0.078985 |
| fold_2021_h2 | r04_trajectory_core | 183 | 1.678994 | 1.588463 | -0.090531 | -0.094868 |
| fold_2022_h1 | r04_baseline_latest_temp_calendar | 181 | 1.693665 | 1.693665 | 0.000000 | 0.000000 |
| fold_2021_h2 | r04_trajectory_no_since_midnight | 183 | 1.754272 | 1.588463 | -0.165809 | -0.145162 |
| fold_2022_h1 | r04_trajectory_no_since_midnight | 181 | 1.765005 | 1.693665 | -0.071340 | -0.049936 |
| fold_2022_h1 | r04_shape_only | 181 | 1.783407 | 1.693665 | -0.089742 | -0.065207 |
| fold_2022_h1 | r04_trajectory_core | 181 | 1.827473 | 1.693665 | -0.133808 | -0.088263 |
| fold_2023_h1 | r04_trajectory_core | 181 | 6.304946 | 1.482125 | -4.822821 | -4.456095 |

## Top Feature Correlations

| feature | n | pearson_corr_with_target | feature_min | feature_max | feature_mean |
| --- | --- | --- | --- | --- | --- |
| hko_temp_snapshot_1400_c | 1051 | 0.921511 | 8.400000 | 35.200000 | 25.975357 |
| hko_temp_snapshot_1200_c | 1051 | 0.919239 | 8.800000 | 34.900000 | 25.490961 |
| hko_temp_snapshot_1300_c | 1050 | 0.916637 | 8.600000 | 36.000000 | 25.928095 |
| hko_latest_temp_c | 1275 | 0.913010 | 8.400000 | 35.700000 | 26.166980 |
| hko_since_midnight_max_to_cutoff_c | 318 | 0.908663 | 10.700000 | 35.300000 | 26.776415 |
| hko_temp_snapshot_0900_c | 1052 | 0.905449 | 7.700000 | 32.700000 | 23.501521 |
| hko_temp_max_so_far_c | 1275 | 0.904049 | 9.800000 | 36.000000 | 26.732549 |
| hko_since_midnight_min_to_cutoff_c | 1275 | 0.900162 | 7.500000 | 31.000000 | 22.779922 |
| hko_temp_min_so_far_c | 1275 | 0.892084 | 7.600000 | 31.800000 | 23.132627 |
| hko_temp_snapshot_0600_c | 1050 | 0.887297 | 7.700000 | 31.500000 | 22.740667 |
| hko_temp_snapshot_0300_c | 1051 | 0.884259 | 8.000000 | 31.700000 | 23.098478 |
| hko_temp_snapshot_0000_c | 1045 | 0.880903 | 8.100000 | 32.500000 | 23.554067 |
| doy_cos | 1275 | -0.791114 | -0.999979 | 0.999991 | 0.000448 |
| noon_solar_elevation_deg | 1275 | 0.723223 | 44.258122 | 89.968858 | 67.192168 |
| day_length_hours | 1275 | 0.722408 | 10.634170 | 13.365794 | 11.978348 |
| solar_declination_deg | 1275 | 0.721377 | -23.439878 | 23.439336 | -0.371697 |
| doy_sin | 1275 | -0.340217 | -0.999999 | 0.999986 | -0.089503 |
| hko_heating_onset_minute | 1021 | -0.298801 | 390.000000 | 860.000000 | 515.667973 |
| hko_latest_minus_0000_c | 1045 | 0.224695 | -8.000000 | 8.200000 | 2.483158 |
| hko_temp_change_720m_to_latest_c | 1049 | 0.213489 | -6.800000 | 8.400000 | 2.883222 |
| hko_temp_slope_720m_to_latest_c_per_hour | 1049 | 0.213489 | -0.566667 | 0.700000 | 0.240269 |
| hko_latest_minus_0300_c | 1051 | 0.207741 | -6.600000 | 8.300000 | 2.933873 |
| hko_latest_minus_0600_c | 1050 | 0.172995 | -4.300000 | 8.600000 | 3.295429 |
| hko_first_obs_minute | 1275 | 0.172176 | 0.000000 | 790.000000 | 107.408627 |
| hko_last_obs_minute | 1275 | -0.170093 | 580.000000 | 880.000000 | 832.633725 |
| latest_age_minutes_at_cutoff | 1275 | 0.170093 | 0.000000 | 300.000000 | 47.366275 |
| hko_obs_count_to_cutoff | 1275 | -0.168591 | 1.000000 | 89.000000 | 73.019608 |
| hko_since_midnight_min_obs_count | 1275 | 0.141189 | 1.000000 | 526.000000 | 146.758431 |
| hko_time_of_max_so_far_minute | 1275 | 0.140867 | 0.000000 | 880.000000 | 720.148235 |
| hko_latest_minus_1300_c | 1050 | 0.104230 | -5.800000 | 3.800000 | 0.102286 |
| hko_trailing_nonwarming_minutes | 1275 | -0.101526 | 0.000000 | 250.000000 | 11.521569 |
| hko_since_midnight_range_to_cutoff_c | 318 | 0.089396 | 0.700000 | 9.400000 | 4.777673 |
| hko_current_minus_max_so_far_c | 1275 | 0.058698 | -8.000000 | 0.000000 | -0.565569 |
| hko_latest_minus_0900_c | 1052 | 0.058637 | -4.800000 | 7.400000 | 2.525095 |
| hko_temp_slope_360m_to_latest_c_per_hour | 1051 | 0.057751 | -0.816667 | 1.316667 | 0.449302 |
| hko_temp_change_360m_to_latest_c | 1051 | 0.057751 | -4.900000 | 7.900000 | 2.695814 |
| hko_latest_minus_1200_c | 1051 | 0.046383 | -5.700000 | 5.100000 | 0.534539 |
| hko_temp_change_60m_to_latest_c | 1053 | 0.042547 | -4.900000 | 2.300000 | 0.051472 |
| hko_temp_slope_60m_to_latest_c_per_hour | 1053 | 0.042547 | -4.900000 | 2.300000 | 0.051472 |
| hko_temp_change_180m_to_latest_c | 1053 | 0.037487 | -5.800000 | 4.800000 | 0.735138 |
