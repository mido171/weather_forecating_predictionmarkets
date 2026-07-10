# EXP-0040 / HKG-T24-R08 Long-Form Experiment Report

## Purpose

R08 tests whether surface wind direction, vector flow, persistence, gustiness, and simple onshore/offshore proxies add forecast information for the HKG T-24 official Tmax problem. This is the first wind experiment in this sequence that parses the raw compass-direction column from the DATA.GOV.HK wind archive instead of using speed-only Phase A summaries. The core question is whether vector treatment separates maritime cooling, weak-flow urban heating, and sea-breeze-like flow from the information already contained in HKO temperature, moisture, and pressure-transition features.

## Data Used

The feature backbone is the R07 pre-validation feature matrix. R08 reparses immutable `datagov_hko_historical_latest_10min_wind_archive` ZIP payloads into a bronze wind-vector table. The parser samples snapshots near 02:40, 08:40, 11:40, 13:40, and 14:40 HKT, converts compass directions to meteorological degrees, and converts speed/direction into flow u/v components. The target-date feature period is `2020-07-02` through `2023-12-31`, the OOF prediction period is `2021-07-01` through `2023-12-31`, and the parsed wind observation period is `2021-12-29 03:40:00+00:00` through `2026-06-18 06:50:00+00:00`.

## Feature Construction

For each target date T, every wind feature is selected as of T-1 15:00 HKT using the inherited conservative `observed_at + 20 minutes` replay latency. The matrix includes station count, direction availability fraction, deliberately wrong linear mean direction, median and maximum speed/gust, calm fraction, vector-mean u/v, vector speed, vector-from direction, circular-variance proxy, easterly/southerly/northerly from-components, onshore/offshore proxies, gustiness ratio, 1/3/6/12/24-hour vector changes, vector-turn proxy, sea-breeze proxy score, and weak-flow urban-heating proxy. Month-permuted vector controls are included as negative controls.

## Models

The ladder includes a temperature/calendar baseline, a deliberately wrong linear-direction control, vector basic Elastic Net, vector-change Elastic Net, onshore/sea-breeze Elastic Net, shallow constrained gradient boosting, and a month-permuted vector control. The wrong-direction model is important: wind direction is circular, so averaging degrees linearly should not be trusted. A useful vector result should beat that control, not merely beat a straw baseline.

## Leakage Controls

No target-day wind observations enter the matrix. No validation-2024 rows are used. No locked-test rows are accessed. All imputation, scaling, Elastic Net, Ridge, and boosting fits are performed inside chronological training folds. The raw wind rows can include later years because they are immutable observations, but the generated feature matrix and prediction table are guarded to end before 2024 for R08 development. Dynamic station selection by target outcome is forbidden.

## Missing Inputs and Blockers

R08 still does not complete the full dynamic-upwind analogue requested by the specification because station-level temperature/dew-point gradient fields are scheduled for R09. It also uses simple geographic onshore proxies rather than station-specific coastline normals. These are explicit limitations. The important completed step is direction-aware vector wind parsing and fold-safe OOF testing.

## OOF Gate

The strict four-year OOF check is `BLOCKED`: R08 modern vector-wind pre-validation feature period: 3.50 years available, requires at least 4.0 years. R08 is therefore a completed diagnostic but not promotable under the user's four-year OOF requirement, regardless of whether a wind feature appears positive.

## Main Result

The best non-control model by OOF MAE is `r08_baseline_temp_calendar` with MAE `1.4723` C, RMSE `1.8861` C, bias `0.0298` C, and CRPS `1.0512` over `911` rows. The fold-delta and wind-regime subgroup tables determine whether any wind signal is stable or confined to weak-flow/onshore/sea-breeze-proxy cohorts.

## Interpretation

If vector models beat both baseline and wrong-direction controls, wind direction carries real incremental information. If wrong linear direction is competitive, the signal is probably seasonal or speed-driven rather than directional. If onshore proxies help only in summer or high sea-breeze-score subgroups, they should be retained as conditional specialists. If all wind models lose to baseline, the current network-level wind representation is insufficient and R09 station-temperature gradients plus station-specific coastline geometry become higher priority.

## Decision Record

R08 is complete as a direction-aware wind diagnostic when artifacts and tests pass. It does not authorize validation access or locked-test access. The next planned experiment is R09 all-station temperature gradients, which is required before dynamic-upwind station pools can be implemented honestly.

## Actual Diagnostic Disposition

The generated scoreboard must be read with the negative controls in view. In this run the month-permuted vector control is extremely competitive and can even edge the baseline by a tiny amount, while the real vector wind models do not establish stable incremental skill. That is not a promotion signal. It is a warning that the current wind feature representation is entangled with season, sample coverage, and fold timing more than with a robust physical wind response. Because the wrong linear-direction control and month-permuted control are retained in the scoreboard, the experiment does not hide this weakness.

## Station and Direction Coverage

The parser extracted direction-aware wind snapshots for the public wind network available in the raw archive, but not every nominal wind station has complete usable direction at every sampled time. Direction values such as `Variable` and `N/A` are preserved as missing direction, not forced into a fake angle. Speed and gust can still be used when direction is missing, but vector u/v components require a valid compass point. The feature matrix therefore includes station counts and direction-availability fractions so that downstream models can distinguish real wind regimes from missing direction coverage.

## Why Dynamic Upwind Is Deferred

Dynamic-upwind features require station-level temperature and dew-point fields at the same cutoff, plus a rule mapping wind vectors to candidate upstream stations. R08 intentionally does not fabricate those inputs from network medians. R09 is the all-station temperature-gradient experiment, and it is the correct dependency for dynamic-upwind station pools. Once R09 exists, R08-style vector direction can be joined to station thermal anomalies to test whether, for example, easterly maritime flow or northerly inland flow changes next-day HKO Tmax residuals. Until then, R08 remains a vector-wind-only diagnostic.

## Acceptance Outcome

R08 does not meet the promotion rule. It does not provide stable incremental OOF skill, does not prove a sea-breeze specialist with a 0.15 C cohort improvement, and is also blocked by the strict four-year OOF requirement. The useful completed artifact is the raw direction parser and the vector-feature matrix. The correct next action is to proceed to R09 and then revisit dynamic-upwind wind-temperature interactions with station-level thermal gradients.

## Provenance Note

Every R08 output row can be traced back to the immutable raw wind ZIP hashes through the bronze wind-vector table listed in `DATA_MANIFEST.yaml`. The experiment does not depend on the live current wind feed, does not backfill direction from later observations, and does not average across future timestamps to smooth noisy wind vectors. This keeps the result reproducible even though the statistical finding is null.

# R08 Machine-Readable Summary Tables

Generated: `2026-06-20T09:53:34.935994Z`

## Overall Scoreboard

| model_id | n | first_date | last_date | mae | rmse | median_abs_error | bias | crps_normal | coverage_80 | coverage_90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r08_month_permuted_vector_control | 911 | 2021-07-01 | 2023-12-31 | 1.468769 | 1.890740 | 1.182266 | 0.067131 | 1.050974 | 0.821076 | 0.905598 |
| r08_baseline_temp_calendar | 911 | 2021-07-01 | 2023-12-31 | 1.472338 | 1.886078 | 1.216189 | 0.029801 | 1.051187 | 0.829857 | 0.908891 |
| r08_onshore_seabreeze_elastic_net | 911 | 2021-07-01 | 2023-12-31 | 1.555848 | 1.999512 | 1.265544 | -0.411590 | 1.110609 | 0.802415 | 0.894621 |
| r08_shallow_boosting_vector_wind | 911 | 2021-07-01 | 2023-12-31 | 1.712137 | 2.152342 | 1.469684 | -0.083488 | 1.206686 | 0.739846 | 0.859495 |
| r08_wrong_linear_direction_control | 911 | 2021-07-01 | 2023-12-31 | 1.870910 | 2.713479 | 1.361605 | 0.463107 | 1.396373 | 0.769484 | 0.845225 |
| r08_vector_change_elastic_net | 911 | 2021-07-01 | 2023-12-31 | 1.926461 | 2.940819 | 1.345666 | 0.458188 | 1.452607 | 0.769484 | 0.856202 |
| r08_vector_basic_elastic_net | 911 | 2021-07-01 | 2023-12-31 | 2.271501 | 3.817770 | 1.415771 | 0.830158 | 1.775800 | 0.738749 | 0.829857 |

## Fold Deltas

| fold_id | model_id | n | mae | baseline_mae | mae_improvement_vs_baseline | crps_improvement_vs_baseline |
| --- | --- | --- | --- | --- | --- | --- |
| fold_2023_h2 | r08_vector_change_elastic_net | 184 | 1.242816 | 1.326541 | 0.083724 | 0.053027 |
| fold_2022_h2 | r08_wrong_linear_direction_control | 182 | 1.258945 | 1.273130 | 0.014185 | 0.010316 |
| fold_2023_h2 | r08_onshore_seabreeze_elastic_net | 184 | 1.269057 | 1.326541 | 0.057483 | 0.041400 |
| fold_2023_h2 | r08_vector_basic_elastic_net | 184 | 1.269200 | 1.326541 | 0.057341 | 0.039225 |
| fold_2022_h2 | r08_baseline_temp_calendar | 182 | 1.273130 | 1.273130 | 0.000000 | 0.000000 |
| fold_2022_h2 | r08_month_permuted_vector_control | 182 | 1.282227 | 1.273130 | -0.009098 | -0.003506 |
| fold_2022_h2 | r08_vector_change_elastic_net | 182 | 1.309487 | 1.273130 | -0.036357 | -0.011107 |
| fold_2022_h2 | r08_vector_basic_elastic_net | 182 | 1.312173 | 1.273130 | -0.039043 | -0.008131 |
| fold_2023_h2 | r08_wrong_linear_direction_control | 184 | 1.319646 | 1.326541 | 0.006895 | 0.007566 |
| fold_2023_h2 | r08_baseline_temp_calendar | 184 | 1.326541 | 1.326541 | 0.000000 | 0.000000 |
| fold_2023_h2 | r08_month_permuted_vector_control | 184 | 1.326888 | 1.326541 | -0.000347 | 0.001947 |
| fold_2023_h1 | r08_onshore_seabreeze_elastic_net | 181 | 1.414166 | 1.482125 | 0.067959 | 0.045948 |
| fold_2023_h1 | r08_vector_change_elastic_net | 181 | 1.416171 | 1.482125 | 0.065954 | 0.042030 |
| fold_2022_h2 | r08_onshore_seabreeze_elastic_net | 182 | 1.453294 | 1.273130 | -0.180164 | -0.087831 |
| fold_2023_h1 | r08_vector_basic_elastic_net | 181 | 1.458101 | 1.482125 | 0.024024 | 0.018905 |
| fold_2023_h2 | r08_shallow_boosting_vector_wind | 184 | 1.468374 | 1.326541 | -0.141834 | -0.072101 |
| fold_2023_h1 | r08_baseline_temp_calendar | 181 | 1.482125 | 1.482125 | 0.000000 | 0.000000 |
| fold_2023_h1 | r08_month_permuted_vector_control | 181 | 1.489988 | 1.482125 | -0.007863 | -0.004553 |
| fold_2023_h1 | r08_wrong_linear_direction_control | 181 | 1.494885 | 1.482125 | -0.012760 | -0.003326 |
| fold_2023_h1 | r08_shallow_boosting_vector_wind | 181 | 1.506017 | 1.482125 | -0.023892 | -0.020139 |
| fold_2021_h2 | r08_month_permuted_vector_control | 183 | 1.559054 | 1.588463 | 0.029409 | 0.002530 |
| fold_2021_h2 | r08_baseline_temp_calendar | 183 | 1.588463 | 1.588463 | 0.000000 | 0.000000 |
| fold_2021_h2 | r08_wrong_linear_direction_control | 183 | 1.588463 | 1.588463 | 0.000000 | 0.000000 |
| fold_2021_h2 | r08_vector_basic_elastic_net | 183 | 1.618443 | 1.588463 | -0.029980 | -0.014726 |
| fold_2021_h2 | r08_vector_change_elastic_net | 183 | 1.622394 | 1.588463 | -0.033932 | -0.016737 |
| fold_2021_h2 | r08_onshore_seabreeze_elastic_net | 183 | 1.625947 | 1.588463 | -0.037485 | -0.018596 |
| fold_2021_h2 | r08_shallow_boosting_vector_wind | 183 | 1.685542 | 1.588463 | -0.097079 | -0.058803 |
| fold_2022_h1 | r08_month_permuted_vector_control | 181 | 1.688074 | 1.693665 | 0.005591 | 0.004611 |
| fold_2022_h1 | r08_baseline_temp_calendar | 181 | 1.693665 | 1.693665 | 0.000000 | 0.000000 |
| fold_2022_h2 | r08_shallow_boosting_vector_wind | 182 | 1.818883 | 1.273130 | -0.545753 | -0.331237 |
| fold_2022_h1 | r08_onshore_seabreeze_elastic_net | 181 | 2.021320 | 1.693665 | -0.327655 | -0.279998 |
| fold_2022_h1 | r08_shallow_boosting_vector_wind | 181 | 2.085615 | 1.693665 | -0.391950 | -0.296693 |
| fold_2022_h1 | r08_wrong_linear_direction_control | 181 | 3.708252 | 1.693665 | -2.014587 | -1.752113 |
| fold_2022_h1 | r08_vector_change_elastic_net | 181 | 4.059538 | 1.693665 | -2.365873 | -2.088253 |
| fold_2022_h1 | r08_vector_basic_elastic_net | 181 | 5.728715 | 1.693665 | -4.035050 | -3.682799 |

## Wind-Regime Subgroups

| model_id | wind_regime | n | mae | rmse | crps_normal |
| --- | --- | --- | --- | --- | --- |
| r08_onshore_seabreeze_elastic_net | sea_breeze_proxy_high | 29 | 1.296365 | 1.660635 | 0.932970 |
| r08_month_permuted_vector_control | ordinary_wind | 620 | 1.439977 | 1.864342 | 1.032746 |
| r08_baseline_temp_calendar | ordinary_wind | 620 | 1.447951 | 1.859212 | 1.034179 |
| r08_baseline_temp_calendar | weak_flow | 77 | 1.453774 | 1.763112 | 1.010790 |
| r08_month_permuted_vector_control | weak_flow | 77 | 1.468507 | 1.783338 | 1.020621 |
| r08_month_permuted_vector_control | sea_breeze_proxy_high | 29 | 1.482788 | 1.800185 | 1.035826 |
| r08_baseline_temp_calendar | sea_breeze_proxy_high | 29 | 1.489946 | 1.828773 | 1.047884 |
| r08_onshore_seabreeze_elastic_net | ordinary_wind | 620 | 1.495459 | 1.894273 | 1.061626 |
| r08_baseline_temp_calendar | onshore_flow | 185 | 1.559032 | 2.028456 | 1.125519 |
| r08_month_permuted_vector_control | onshore_flow | 185 | 1.563173 | 2.030700 | 1.127071 |
| r08_vector_basic_elastic_net | weak_flow | 77 | 1.566962 | 1.882039 | 1.075166 |
| r08_vector_change_elastic_net | weak_flow | 77 | 1.572211 | 1.917541 | 1.088968 |
| r08_vector_change_elastic_net | sea_breeze_proxy_high | 29 | 1.593000 | 2.152820 | 1.164221 |
| r08_shallow_boosting_vector_wind | onshore_flow | 185 | 1.646523 | 2.075191 | 1.173141 |
| r08_shallow_boosting_vector_wind | weak_flow | 77 | 1.667019 | 2.036974 | 1.162211 |
| r08_wrong_linear_direction_control | weak_flow | 77 | 1.670150 | 2.058630 | 1.170858 |
| r08_wrong_linear_direction_control | ordinary_wind | 620 | 1.686564 | 2.314120 | 1.235133 |
| r08_onshore_seabreeze_elastic_net | onshore_flow | 185 | 1.698594 | 2.263722 | 1.237531 |
| r08_shallow_boosting_vector_wind | ordinary_wind | 620 | 1.729285 | 2.184392 | 1.217459 |
| r08_vector_change_elastic_net | ordinary_wind | 620 | 1.768328 | 2.570654 | 1.305523 |
| r08_onshore_seabreeze_elastic_net | weak_flow | 77 | 1.796860 | 2.249102 | 1.266985 |
| r08_shallow_boosting_vector_wind | sea_breeze_proxy_high | 29 | 1.883889 | 2.241647 | 1.308428 |
| r08_vector_basic_elastic_net | sea_breeze_proxy_high | 29 | 2.021366 | 2.565937 | 1.448752 |
| r08_vector_basic_elastic_net | ordinary_wind | 620 | 2.022432 | 3.209300 | 1.539378 |
| r08_wrong_linear_direction_control | sea_breeze_proxy_high | 29 | 2.025195 | 2.632882 | 1.471185 |
| r08_wrong_linear_direction_control | onshore_flow | 185 | 2.548095 | 3.931935 | 2.018881 |
| r08_vector_change_elastic_net | onshore_flow | 185 | 2.656139 | 4.264280 | 2.142096 |
| r08_vector_basic_elastic_net | onshore_flow | 185 | 3.438669 | 5.894902 | 2.911013 |

## Wind Diagnostics

| feature | n | pearson_corr_with_target | mean | p10 | p90 |
| --- | --- | --- | --- | --- | --- |
| wind_vector_speed_kmh | 729 | -0.041936 | 11.171298 | 4.969170 | 18.541906 |
| wind_circular_variance_proxy | 729 | 0.000436 | 0.199404 | 0.000000 | 0.448142 |
| wind_onshore_proxy_kmh | 729 | 0.049154 | 8.286021 | 0.077029 | 17.462939 |
| wind_offshore_proxy_kmh | 729 | 0.297637 | 1.374613 | 0.000000 | 5.408379 |
| sea_breeze_proxy_score | 1275 | -0.028874 | 0.497508 | 0.000000 | 1.284105 |
| weak_flow_urban_heating_proxy | 1275 | -0.013092 | -0.121013 | -0.674663 | 0.085241 |
| wind_vector_turn_3h_abs_proxy | 729 | -0.050627 | 4.615907 | 1.608648 | 8.103573 |
