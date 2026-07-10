# EXP-0038 / HKG-T24-R06 Long-Form Experiment Report

## Purpose

R06 tests whether moisture state adds real T-24 information for the official Hong Kong Observatory Headquarters daily Tmax target. The research question is deliberately narrower than "does humidity correlate with temperature." Relative humidity is temperature-dependent, so raw RH can look weak or misleading. The useful physical variables are dew point, dew-point depression, wet-bulb state, vapor pressure, moisture tendencies, and spatial moisture gradients. These variables can identify maritime air, dry-air intrusion, cloud/rain potential, and evaporative constraints that change how much a warm cutoff temperature can translate into next-day maximum temperature.

## Data Used

The experiment uses the R04 pre-validation feature matrix as its target/date/thermal backbone and then reparses immutable DATA.GOV.HK historical `latest_1min_temperature` and `latest_1min_humidity` ZIP payloads from the retrieval ledger. The parser samples snapshot files near the cutoff-relevant clocks 02:40, 08:40, 11:40, 13:40, and 14:40 HKT. For every target date T, the cutoff remains T-1 15:00:00 HKT. A record is eligible only through the conservative replay rule `available_at = observed_at + 20 minutes`; therefore the current cutoff state normally resolves to the latest observation available by 15:00, usually around 14:40. The feature matrix period is `2020-07-02` through `2023-12-31`, and the OOF prediction period is `2021-07-01` through `2023-12-31`.

## Feature Construction

R06 builds HKO humidity at cutoff, HKO humidity changes over 1, 3, 6, 12, and 24 hours, dew point from a Magnus formula, dew-point depression, Stull wet-bulb approximation, vapor pressure, a pressure-conditioned mixing-ratio proxy where HKO pressure is available, dew-point tendencies, sudden drying and moistening flags, sampled high-RH counts, network median dew point, network RH, network dew-point spread, humid-station fraction, HKO-minus-network dew-point anomaly, and a coastal-minus-inland dew-point gradient. It also creates predeclared interaction terms between dew-point depression and temperature trajectory, dew point and solar geometry, and RH and temperature change.

## Model Ladder

The ladder starts from the same temperature/calendar baseline used in the modern high-frequency experiments. It then tests RH-only Elastic Net, dew-point thermodynamic Elastic Net, network-gradient Elastic Net, a GAM-like spline Ridge model over moisture state and interactions, shallow constrained histogram gradient boosting, an R04-trajectory-plus-moisture Elastic Net, and a month-permuted moisture negative control. The negative control is retained to make sure any apparent moisture value is not just seasonal collinearity or a modeling artifact. All imputation, scaling, spline fitting, coefficient fitting, and boosting fitting occurs inside chronological training folds only.

## Leakage Controls

No target-day observations enter the matrix. No validation-2024 rows are used for feature selection, model choice, or scoring. No locked-test dates are accessed. The script calls the locked-date guard on the source R04 matrix, generated feature matrix, and OOF prediction table. The as-of join is performed on `available_at_hkt`, not filename date, retrieval time, or observed time alone. Daily mean dew point, wet bulb, relative humidity, cloud, and rainfall from HKO daily climate are intentionally excluded because they are retrospective daily aggregates and not proven available at T-1 15:00. This is why R06 uses high-frequency station snapshots instead of tempting long-history daily moisture labels.

## OOF Gate

The strict four-year OOF check is `BLOCKED`: R06 modern HKO moisture-state pre-validation feature period: 3.50 years available, requires at least 4.0 years. As with R04 and R05, this modern high-frequency experiment is therefore a completed diagnostic but not a promotable feature-family result under the user's hard four-year rule. Any useful moisture signal is recorded as evidence for future modeling, not as an accepted challenger.

## Main Result

The best diagnostic model by OOF MAE is `r06_baseline_temp_calendar` with MAE `1.4723` C, RMSE `1.8861` C, bias `0.0298` C, and CRPS `1.0512` over `911` rows. The model must be compared against the baseline and the negative control in the generated scoreboards. A moisture model is useful only if it improves the baseline in real chronological folds and does not simply track month-level seasonality.

## Interpretation Discipline

If dew point beats RH-only, that supports the hypothesis that absolute moisture carries more transferable information than raw RH. If dew-point depression improves dry-air or shoulder-season subgroups, it points toward a physically gateable expert rather than a universal predictor. If the coastal-inland gradient helps, that suggests maritime versus inland air-mass contrast matters for next-day heating at the target station. If the negative control is competitive, the experiment must be treated as null because the apparent signal may be seasonality or chance. If the R04-trajectory-plus-moisture model loses to a simpler moisture model, then moisture is interacting badly with already unstable trajectory features and should be brought forward only through constrained features.

## What Was Not Done

R06 does not use HKO daily mean dew point or wet bulb as operational predictors. It does not use target-day humidity, target-day rainfall, target-day cloud, reanalysis, or finalized products. It does not run validation 2024. It does not use Polymarket data or any market outcome. It does not claim production eligibility because the four-year OOF gate is blocked. The experiment also does not pretend that sampled snapshot counts are a dense continuous saturation-duration measurement; they are named sampled counts because only the selected snapshot windows are parsed for this diagnostic.

## Artifacts

The bronze sampled observation table is `C:\hkg_tmax_data\bronze\hkg_t24\r06_moisture_sampled_observations.parquet`. The feature matrix is `C:\hkg_tmax_data\gold\hkg_t24\r06_moisture_state\r06_feature_matrix.parquet`. OOF predictions, scoreboards, fold deltas, subgroup scores, and moisture diagnostics are in the same data-root folder and copied or summarized inside the experiment directory. The repo-level report is `reports/hkg_t24/R06_MOISTURE_STATE.md`.

## Decision Record

R06 is complete as a moisture-state diagnostic. It may produce accepted, conditional, null, or rejected evidence in the research ledger only after reviewing the scoreboard and fold deltas. Under the current strict sample rule, even a positive diagnostic result cannot be promoted by itself. The next research step remains R07 pressure tendency and front/cold-surge transition detection, using the same leakage firewall and without validation access.

## Result Disposition

The generated scoreboard shows whether moisture helps after the strongest simple cutoff-temperature baseline is already present. In this run the operational champion is selected only from non-negative-control models, and the baseline remains the best model if no moisture candidate improves it. That is an important result, not a failed experiment. It means the sampled public high-frequency moisture archive, as currently represented by this feature family and these conservative fold-safe models, does not justify promotion. The month-permuted control is also informative: if it sits close to or ahead of real moisture features, then month/season structure dominates the apparent moisture signal. This protects the project from carrying a physically plausible but statistically weak feature family into later ensembles.

## Practical Next Use

R06 should not be tuned harder against the same development folds. The useful carry-forward artifact is the audited moisture feature matrix and the source parser. Later experiments may reuse specific features only as predeclared inputs: for example dew-point depression as a dry-air gate in R20/R22, coastal-minus-inland dew point as a marine-regime interaction in R24, or sampled saturation count as a cloud/rain suppression proxy in R13. Reuse must preserve the same available-at join and must not import daily aggregate HKO moisture labels. The null result also narrows the research direction: average moisture state alone appears weaker than transition detection, pressure tendency, wind/advection, upper-air thermal potential, and official forecast vintages are expected to be.

# R06 Machine-Readable Summary Tables

Generated: `2026-06-20T09:40:04.912265Z`

## Overall Scoreboard

| model_id | n | first_date | last_date | mae | rmse | median_abs_error | bias | crps_normal | coverage_80 | coverage_90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r06_baseline_temp_calendar | 911 | 2021-07-01 | 2023-12-31 | 1.472338 | 1.886078 | 1.216189 | 0.029801 | 1.051187 | 0.829857 | 0.908891 |
| r06_month_permuted_moisture_control | 911 | 2021-07-01 | 2023-12-31 | 1.536252 | 1.966631 | 1.254459 | 0.197443 | 1.097290 | 0.811196 | 0.903403 |
| r06_rh_only_elastic_net | 911 | 2021-07-01 | 2023-12-31 | 1.577768 | 2.002620 | 1.319445 | 0.143789 | 1.120766 | 0.792536 | 0.894621 |
| r06_gam_like_spline_thermo | 911 | 2021-07-01 | 2023-12-31 | 1.586134 | 2.057644 | 1.322076 | 0.162196 | 1.142869 | 0.762898 | 0.872667 |
| r06_shallow_boosting_moisture | 911 | 2021-07-01 | 2023-12-31 | 1.972777 | 2.599456 | 1.612503 | 0.387880 | 1.441288 | 0.668496 | 0.790340 |
| r06_r04_trajectory_plus_moisture_elastic_net | 911 | 2021-07-01 | 2023-12-31 | 2.464933 | 3.802171 | 1.560117 | -0.183676 | 1.936798 | 0.675082 | 0.767289 |
| r06_network_gradient_elastic_net | 911 | 2021-07-01 | 2023-12-31 | 3.475855 | 6.854681 | 1.505496 | -1.720997 | 2.952405 | 0.678375 | 0.758507 |
| r06_dewpoint_thermo_elastic_net | 911 | 2021-07-01 | 2023-12-31 | 5.402296 | 12.533002 | 1.507017 | -3.409332 | 4.873745 | 0.675082 | 0.763996 |

## Fold Deltas

| fold_id | model_id | n | mae | baseline_mae | mae_improvement_vs_baseline | crps_improvement_vs_baseline |
| --- | --- | --- | --- | --- | --- | --- |
| fold_2022_h2 | r06_baseline_temp_calendar | 182 | 1.273130 | 1.273130 | 0.000000 | 0.000000 |
| fold_2022_h2 | r06_rh_only_elastic_net | 182 | 1.285268 | 1.273130 | -0.012138 | -0.025513 |
| fold_2022_h2 | r06_month_permuted_moisture_control | 182 | 1.295697 | 1.273130 | -0.022567 | -0.036121 |
| fold_2023_h2 | r06_dewpoint_thermo_elastic_net | 184 | 1.299555 | 1.326541 | 0.026986 | 0.013438 |
| fold_2023_h2 | r06_network_gradient_elastic_net | 184 | 1.319141 | 1.326541 | 0.007400 | 0.008255 |
| fold_2023_h2 | r06_baseline_temp_calendar | 184 | 1.326541 | 1.326541 | 0.000000 | 0.000000 |
| fold_2023_h2 | r06_rh_only_elastic_net | 184 | 1.330244 | 1.326541 | -0.003703 | -0.005474 |
| fold_2023_h2 | r06_r04_trajectory_plus_moisture_elastic_net | 184 | 1.335071 | 1.326541 | -0.008530 | 0.009271 |
| fold_2023_h2 | r06_month_permuted_moisture_control | 184 | 1.336419 | 1.326541 | -0.009879 | -0.015728 |
| fold_2023_h2 | r06_gam_like_spline_thermo | 184 | 1.341914 | 1.326541 | -0.015374 | -0.027390 |
| fold_2022_h2 | r06_dewpoint_thermo_elastic_net | 182 | 1.382208 | 1.273130 | -0.109078 | -0.050517 |
| fold_2022_h2 | r06_network_gradient_elastic_net | 182 | 1.441424 | 1.273130 | -0.168294 | -0.103338 |
| fold_2022_h2 | r06_gam_like_spline_thermo | 182 | 1.455327 | 1.273130 | -0.182197 | -0.076601 |
| fold_2023_h1 | r06_network_gradient_elastic_net | 181 | 1.459537 | 1.482125 | 0.022588 | 0.024659 |
| fold_2023_h2 | r06_shallow_boosting_moisture | 184 | 1.464247 | 1.326541 | -0.137706 | -0.061937 |
| fold_2021_h2 | r06_gam_like_spline_thermo | 183 | 1.470104 | 1.588463 | 0.118358 | 0.054962 |
| fold_2023_h1 | r06_dewpoint_thermo_elastic_net | 181 | 1.471895 | 1.482125 | 0.010230 | 0.013645 |
| fold_2023_h1 | r06_baseline_temp_calendar | 181 | 1.482125 | 1.482125 | 0.000000 | 0.000000 |
| fold_2023_h1 | r06_r04_trajectory_plus_moisture_elastic_net | 181 | 1.483892 | 1.482125 | -0.001767 | 0.001129 |
| fold_2023_h1 | r06_rh_only_elastic_net | 181 | 1.501442 | 1.482125 | -0.019317 | -0.001923 |
| fold_2023_h1 | r06_month_permuted_moisture_control | 181 | 1.503446 | 1.482125 | -0.021321 | -0.009272 |
| fold_2022_h2 | r06_r04_trajectory_plus_moisture_elastic_net | 182 | 1.507100 | 1.273130 | -0.233970 | -0.137529 |
| fold_2023_h1 | r06_gam_like_spline_thermo | 181 | 1.549645 | 1.482125 | -0.067520 | -0.058225 |
| fold_2021_h2 | r06_baseline_temp_calendar | 183 | 1.588463 | 1.588463 | 0.000000 | 0.000000 |
| fold_2023_h1 | r06_shallow_boosting_moisture | 181 | 1.600954 | 1.482125 | -0.118829 | -0.085545 |
| fold_2022_h2 | r06_shallow_boosting_moisture | 182 | 1.672639 | 1.273130 | -0.399509 | -0.219623 |
| fold_2021_h2 | r06_dewpoint_thermo_elastic_net | 183 | 1.681504 | 1.588463 | -0.093041 | -0.072140 |
| fold_2022_h1 | r06_baseline_temp_calendar | 181 | 1.693665 | 1.693665 | 0.000000 | 0.000000 |
| fold_2021_h2 | r06_network_gradient_elastic_net | 183 | 1.710931 | 1.588463 | -0.122468 | -0.101658 |
| fold_2021_h2 | r06_month_permuted_moisture_control | 183 | 1.755558 | 1.588463 | -0.167096 | -0.096439 |
| fold_2022_h1 | r06_month_permuted_moisture_control | 181 | 1.792355 | 1.693665 | -0.098690 | -0.072956 |
| fold_2021_h2 | r06_r04_trajectory_plus_moisture_elastic_net | 183 | 1.807842 | 1.588463 | -0.219379 | -0.181993 |
| fold_2021_h2 | r06_rh_only_elastic_net | 183 | 1.821791 | 1.588463 | -0.233328 | -0.139131 |
| fold_2022_h1 | r06_rh_only_elastic_net | 181 | 1.953119 | 1.693665 | -0.259454 | -0.176389 |
| fold_2022_h1 | r06_shallow_boosting_moisture | 181 | 2.069027 | 1.693665 | -0.375362 | -0.312125 |
| fold_2022_h1 | r06_gam_like_spline_thermo | 181 | 2.119732 | 1.693665 | -0.426067 | -0.353924 |
| fold_2021_h2 | r06_shallow_boosting_moisture | 183 | 3.055144 | 1.588463 | -1.466681 | -1.267955 |
| fold_2022_h1 | r06_r04_trajectory_plus_moisture_elastic_net | 181 | 6.222039 | 1.693665 | -4.528374 | -4.145671 |
| fold_2022_h1 | r06_network_gradient_elastic_net | 181 | 11.514730 | 1.693665 | -9.821065 | -9.395476 |
| fold_2022_h1 | r06_dewpoint_thermo_elastic_net | 181 | 21.307642 | 1.693665 | -19.613977 | -19.143079 |

## Moisture-Regime Subgroups

| model_id | moist_regime | n | mae | rmse | crps_normal |
| --- | --- | --- | --- | --- | --- |
| r06_baseline_temp_calendar | sudden_drying | 37 | 1.313023 | 1.770759 | 0.981928 |
| r06_baseline_temp_calendar | dry_air | 300 | 1.330753 | 1.756862 | 0.974025 |
| r06_month_permuted_moisture_control | dry_air | 300 | 1.343077 | 1.818101 | 1.000757 |
| r06_rh_only_elastic_net | sudden_drying | 37 | 1.352074 | 1.815206 | 1.003446 |
| r06_rh_only_elastic_net | dry_air | 300 | 1.364223 | 1.796622 | 0.997546 |
| r06_month_permuted_moisture_control | sudden_drying | 37 | 1.382279 | 1.810080 | 1.009563 |
| r06_gam_like_spline_thermo | ordinary_moisture | 462 | 1.490427 | 1.890037 | 1.060566 |
| r06_gam_like_spline_thermo | dry_air | 300 | 1.496000 | 1.968934 | 1.092702 |
| r06_baseline_temp_calendar | ordinary_moisture | 462 | 1.503284 | 1.848792 | 1.048286 |
| r06_gam_like_spline_thermo | sudden_drying | 37 | 1.528333 | 1.932564 | 1.082773 |
| r06_shallow_boosting_moisture | sudden_drying | 37 | 1.560349 | 1.943279 | 1.093065 |
| r06_month_permuted_moisture_control | ordinary_moisture | 462 | 1.590265 | 1.936880 | 1.102800 |
| r06_shallow_boosting_moisture | dry_air | 300 | 1.603909 | 1.993439 | 1.131469 |
| r06_rh_only_elastic_net | ordinary_moisture | 462 | 1.653294 | 2.007797 | 1.147149 |
| r06_baseline_temp_calendar | sampled_saturated | 112 | 1.776561 | 2.352000 | 1.292719 |
| r06_month_permuted_moisture_control | sampled_saturated | 112 | 1.881742 | 2.459221 | 1.362112 |
| r06_rh_only_elastic_net | sampled_saturated | 112 | 1.912782 | 2.501512 | 1.380743 |
| r06_r04_trajectory_plus_moisture_elastic_net | sudden_drying | 37 | 2.063577 | 2.812999 | 1.555696 |
| r06_shallow_boosting_moisture | ordinary_moisture | 462 | 2.177749 | 2.870108 | 1.614989 |
| r06_r04_trajectory_plus_moisture_elastic_net | dry_air | 300 | 2.210662 | 3.399489 | 1.721360 |
| r06_gam_like_spline_thermo | sampled_saturated | 112 | 2.241448 | 2.843396 | 1.636594 |
| r06_shallow_boosting_moisture | sampled_saturated | 112 | 2.251555 | 3.015118 | 1.669678 |
| r06_r04_trajectory_plus_moisture_elastic_net | ordinary_moisture | 462 | 2.445072 | 3.751416 | 1.896734 |
| r06_network_gradient_elastic_net | sudden_drying | 37 | 2.758774 | 5.586858 | 2.318461 |
| r06_network_gradient_elastic_net | dry_air | 300 | 2.942433 | 5.517234 | 2.452515 |
| r06_r04_trajectory_plus_moisture_elastic_net | sampled_saturated | 112 | 3.360531 | 5.095827 | 2.805028 |
| r06_network_gradient_elastic_net | ordinary_moisture | 462 | 3.369451 | 6.865081 | 2.839481 |
| r06_dewpoint_thermo_elastic_net | sudden_drying | 37 | 4.141412 | 10.230495 | 3.732159 |
| r06_dewpoint_thermo_elastic_net | dry_air | 300 | 4.450628 | 9.924136 | 3.961008 |
| r06_dewpoint_thermo_elastic_net | ordinary_moisture | 462 | 5.161714 | 12.562182 | 4.616864 |
| r06_network_gradient_elastic_net | sampled_saturated | 112 | 5.580474 | 9.794425 | 4.966639 |
| r06_dewpoint_thermo_elastic_net | sampled_saturated | 112 | 9.360347 | 18.119100 | 8.755343 |

## Moisture Diagnostics

| feature | n | pearson_corr_with_target | mean | p10 | p90 |
| --- | --- | --- | --- | --- | --- |
| hko_rh_cutoff_pct | 1050 | 0.108100 | 67.883810 | 51.000000 | 86.000000 |
| hko_dew_point_c | 1050 | 0.825253 | 19.343950 | 10.675603 | 25.800286 |
| hko_dewpoint_depression_c | 1050 | -0.061494 | 6.677859 | 2.422393 | 10.665446 |
| hko_wet_bulb_c | 1050 | 0.895531 | 21.548222 | 14.204573 | 27.365981 |
| hko_rh_change_3h_pct | 1050 | -0.069024 | -1.956190 | -8.100000 | 4.000000 |
| hko_dew_point_change_3h_c | 1050 | -0.095949 | 0.255995 | -0.897311 | 1.463391 |
| network_median_dew_point_c | 1051 | 0.823205 | 19.162570 | 9.960407 | 25.968399 |
| coastal_minus_inland_dew_point_c | 1051 | -0.065949 | 0.203496 | -0.506003 | 0.962606 |
