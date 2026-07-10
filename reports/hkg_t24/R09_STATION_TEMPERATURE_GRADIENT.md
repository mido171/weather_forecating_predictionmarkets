# EXP-0041 / HKG-T24-R09 Long-Form Experiment Report

## Purpose

R09 tests whether the 39-station HKO temperature field adds information beyond HKO Headquarters' own cutoff temperature. It focuses on transparent spatial summaries: station offsets, network spread, inland-coastal contrasts, urban-coastal contrasts, elevated-lowland contrasts, east-west and north-south gradients, local HKO outlier score, and simple flow-conditioned interactions inherited from R08 vector wind. The goal is to learn whether spatial thermal structure identifies mesoscale heating/cooling regimes without using target-day observations.

## Data Used

The feature backbone is the R08 pre-validation feature matrix. R09 reparses immutable `datagov_hko_historical_latest_1min_temperature_archive` ZIP payloads for the full HKO temperature/max-min station list. The parser samples cutoff-relevant snapshot windows near 02:40, 08:40, 11:40, 13:40, and 14:40 HKT and uses the conservative `observed_at + 20 minutes` replay latency. The target-date feature period is `2020-07-02` through `2023-12-31`, the OOF prediction period is `2021-07-01` through `2023-12-31`, and the parsed station-temperature observation period is `2020-06-30 03:40:00+00:00` through `2026-06-18 06:50:00+00:00`.

## Feature Construction

For every target date T, current and lagged station fields are selected as of T-1 15:00 HKT. R09 computes station count, coverage fraction, network median, trimmed mean, extrema, spread, IQR proxy, standard deviation, skew proxy, HKO-minus-network median, physically defined group means, group contrasts, HKO-minus-each-station offset columns, 3/6/12/24-hour changes in selected spatial summaries, a sea-breeze thermal-gradient proxy, HKO local warm-outlier score, and interactions between temperature gradients and R08 wind components. Month-permuted spatial controls are retained as negative controls.

## Blockers

The uploaded specification asks for station seasonal expected offsets learned inside folds, elevation-adjusted residual offsets, and robust planar gradient magnitude/direction. The current station registry does not yet contain coordinates or elevation fields, so true plane fitting and elevation adjustment are blocked. R09 does not fake those fields. It implements transparent group contrasts and raw station offsets, then documents coordinate/elevation metadata as a blocker for a richer R09/R10 continuation.

## Model Ladder

The ladder includes a temperature/calendar baseline, spatial-summary Elastic Net, spatial-change Elastic Net, station-offset Elastic Net, flow-interaction Elastic Net, shallow constrained boosting, and a month-permuted spatial negative control. Each model is trained only on chronological training folds. The station-offset model lets the fold-local regularizer learn compact marginal station value without selecting stations from validation or locked-test outcomes.

## Leakage Controls

No target-day station observations enter the matrix. No validation-2024 outcomes are used. No locked-test dates are accessed. The parser can read immutable raw rows beyond 2023, but the generated feature and OOF prediction tables are guarded to pre-validation development dates. Seasonal expected offsets are not precomputed using full-sample future data; instead raw offsets are supplied to fold-local models.

## OOF Gate

The strict four-year OOF check is `BLOCKED`: R09 modern station-temperature-gradient pre-validation feature period: 3.50 years available, requires at least 4.0 years. R09 is therefore a completed diagnostic but not promotable under the user's hard four-year OOF rule.

## Main Result

The best non-control model by OOF MAE is `r09_baseline_temp_calendar` with MAE `1.4723` C, RMSE `1.8861` C, bias `0.0298` C, and CRPS `1.0512` over `911` rows. The scoreboard, fold deltas, and diagnostics show whether spatial temperature fields add anything beyond the local HKO cutoff state.

## Interpretation

If station offsets or group contrasts beat baseline consistently, station-network thermal structure is a candidate compact feature family. If only shallow boosting improves isolated folds, the signal may be unstable or missingness-driven. If month-permuted controls are competitive, the apparent spatial value is probably seasonal/sample artifact. If all spatial models lose, the current public snapshot station field is not enough without coordinate/elevation metadata, wind-direction-conditioned upwind pools, or latent spatial modes from R10.

## Decision Record

R09 is complete as a transparent all-station temperature-gradient diagnostic once artifacts and tests pass. It does not authorize validation access. The next planned experiment is R10 latent spatial modes, which should use fold-fit PCA/graph modes only after the R09 station matrix exists and leakage tests pass.

## Actual Diagnostic Disposition

The generated scoreboard shows a nuanced null result. The local HKO temperature/calendar baseline remains best by MAE. The station-offset model can reduce RMSE and CRPS slightly in this diagnostic window, but its MAE is still worse than baseline and interval coverage deteriorates. That is not a promotable feature-family result under the predeclared rules. It does, however, suggest that station offsets may reduce some larger errors while increasing ordinary-day absolute error. This is exactly the kind of conditional signal that should be revisited in R10 latent modes and R22 catastrophic-error specialists, not forced into the main model now.

## Multiple-Comparison Discipline

R09 exposes many station offset columns and physical group contrasts. A single station or contrast looking useful in one fold would not be enough. The station-offset Elastic Net regularizes these columns inside folds, but the research conclusion still has to account for the number of stations and contrasts tested. The month-permuted spatial control remains close to the weaker spatial models, which is a warning that sample timing and seasonality explain part of the apparent network value. Therefore the result is kept as evidence, not as feature promotion.

## Coverage And Metadata Implications

The raw parser confirms that the public archive contains all 39 temperature stations in the current station list, and R09 preserves station count and coverage fraction as first-class features. The missing piece is not station membership; it is metadata. Without coordinates, elevation, and validated station-history segments, robust plane-fit gradients and elevation-adjusted residual offsets would be pseudo-precision. The correct engineering step is to enrich the station registry before claiming terrain or centroid effects.

## Carry-Forward Rules

Later experiments may reuse the R09 station-temperature table and raw offset features, but any dimensionality reduction, station selection, seasonal-offset normalization, or outage robustness must be fit inside each training fold. The all-station table is valuable input for R10 latent spatial modes and for future dynamic-upwind features with R08 wind vectors. R09 itself does not prove enough stable incremental skill for promotion.

## Why The Artifact Matters

Before R09, the project had isolated HKO target-station trajectories and some selected network summaries, but not a dedicated, reproducible all-station temperature matrix tied to the T-24 cutoff. R09 creates that matrix with hashes, manifests, and experiment documentation. Even though the first transparent spatial models are not promoted, the matrix is now available for outage simulation, latent-mode fitting, station marginal-value analysis, dynamic upwind pairing, and conditional specialists. This is concrete progress because future work can build on a leakage-checked table instead of reparsing raw ZIPs ad hoc.

# R09 Machine-Readable Summary Tables

Generated: `2026-06-20T10:00:26.197055Z`

## Overall Scoreboard

| model_id | n | first_date | last_date | mae | rmse | median_abs_error | bias | crps_normal | coverage_80 | coverage_90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r09_baseline_temp_calendar | 911 | 2021-07-01 | 2023-12-31 | 1.472338 | 1.886078 | 1.216189 | 0.029801 | 1.051187 | 0.829857 | 0.908891 |
| r09_month_permuted_spatial_control | 911 | 2021-07-01 | 2023-12-31 | 1.476449 | 1.895176 | 1.216505 | 0.032288 | 1.055530 | 0.828760 | 0.903403 |
| r09_spatial_summary_elastic_net | 911 | 2021-07-01 | 2023-12-31 | 1.477948 | 1.902074 | 1.254493 | 0.194094 | 1.056817 | 0.802415 | 0.894621 |
| r09_station_offsets_elastic_net | 911 | 2021-07-01 | 2023-12-31 | 1.485087 | 1.857126 | 1.299921 | -0.152940 | 1.046776 | 0.785950 | 0.891328 |
| r09_spatial_change_elastic_net | 911 | 2021-07-01 | 2023-12-31 | 1.516959 | 1.952740 | 1.255889 | 0.292120 | 1.084418 | 0.778266 | 0.875960 |
| r09_flow_interaction_elastic_net | 911 | 2021-07-01 | 2023-12-31 | 1.522153 | 1.962700 | 1.278194 | 0.283646 | 1.089201 | 0.783754 | 0.877058 |
| r09_shallow_boosting_spatial | 911 | 2021-07-01 | 2023-12-31 | 1.760386 | 2.228058 | 1.560714 | 0.195804 | 1.252705 | 0.693743 | 0.819978 |

## Fold Deltas

| fold_id | model_id | n | mae | baseline_mae | mae_improvement_vs_baseline | crps_improvement_vs_baseline |
| --- | --- | --- | --- | --- | --- | --- |
| fold_2023_h2 | r09_spatial_summary_elastic_net | 184 | 1.258350 | 1.326541 | 0.068191 | 0.036926 |
| fold_2023_h2 | r09_spatial_change_elastic_net | 184 | 1.261991 | 1.326541 | 0.064550 | 0.039864 |
| fold_2022_h2 | r09_spatial_summary_elastic_net | 182 | 1.264246 | 1.273130 | 0.008884 | 0.018339 |
| fold_2023_h2 | r09_flow_interaction_elastic_net | 184 | 1.266114 | 1.326541 | 0.060427 | 0.037836 |
| fold_2022_h2 | r09_spatial_change_elastic_net | 182 | 1.268345 | 1.273130 | 0.004785 | 0.015428 |
| fold_2022_h2 | r09_baseline_temp_calendar | 182 | 1.273130 | 1.273130 | 0.000000 | 0.000000 |
| fold_2022_h2 | r09_flow_interaction_elastic_net | 182 | 1.275901 | 1.273130 | -0.002771 | 0.012009 |
| fold_2022_h2 | r09_month_permuted_spatial_control | 182 | 1.284836 | 1.273130 | -0.011707 | -0.009073 |
| fold_2023_h2 | r09_station_offsets_elastic_net | 184 | 1.297866 | 1.326541 | 0.028675 | 0.028227 |
| fold_2023_h2 | r09_baseline_temp_calendar | 184 | 1.326541 | 1.326541 | 0.000000 | 0.000000 |
| fold_2023_h2 | r09_month_permuted_spatial_control | 184 | 1.337529 | 1.326541 | -0.010989 | -0.004907 |
| fold_2022_h2 | r09_station_offsets_elastic_net | 182 | 1.344337 | 1.273130 | -0.071208 | -0.017017 |
| fold_2023_h2 | r09_shallow_boosting_spatial | 184 | 1.448455 | 1.326541 | -0.121914 | -0.049686 |
| fold_2023_h1 | r09_month_permuted_spatial_control | 181 | 1.481936 | 1.482125 | 0.000189 | -0.000466 |
| fold_2023_h1 | r09_baseline_temp_calendar | 181 | 1.482125 | 1.482125 | 0.000000 | 0.000000 |
| fold_2023_h1 | r09_station_offsets_elastic_net | 181 | 1.487557 | 1.482125 | -0.005432 | 0.020915 |
| fold_2023_h1 | r09_spatial_summary_elastic_net | 181 | 1.490869 | 1.482125 | -0.008744 | 0.018600 |
| fold_2023_h1 | r09_spatial_change_elastic_net | 181 | 1.492880 | 1.482125 | -0.010755 | 0.014749 |
| fold_2023_h1 | r09_flow_interaction_elastic_net | 181 | 1.498511 | 1.482125 | -0.016386 | 0.005171 |
| fold_2021_h2 | r09_station_offsets_elastic_net | 183 | 1.522601 | 1.588463 | 0.065861 | 0.033323 |
| fold_2023_h1 | r09_shallow_boosting_spatial | 181 | 1.574123 | 1.482125 | -0.091998 | -0.074582 |
| fold_2021_h2 | r09_baseline_temp_calendar | 183 | 1.588463 | 1.588463 | 0.000000 | 0.000000 |
| fold_2021_h2 | r09_month_permuted_spatial_control | 183 | 1.595289 | 1.588463 | -0.006826 | -0.009803 |
| fold_2021_h2 | r09_spatial_summary_elastic_net | 183 | 1.658744 | 1.588463 | -0.070282 | -0.084772 |
| fold_2022_h2 | r09_shallow_boosting_spatial | 182 | 1.666443 | 1.273130 | -0.393314 | -0.204407 |
| fold_2022_h1 | r09_month_permuted_spatial_control | 181 | 1.684701 | 1.693665 | 0.008964 | 0.002629 |
| fold_2022_h1 | r09_baseline_temp_calendar | 181 | 1.693665 | 1.693665 | 0.000000 | 0.000000 |
| fold_2022_h1 | r09_spatial_summary_elastic_net | 181 | 1.720352 | 1.693665 | -0.026687 | -0.017205 |
| fold_2021_h2 | r09_spatial_change_elastic_net | 183 | 1.724113 | 1.588463 | -0.135651 | -0.133925 |
| fold_2021_h2 | r09_flow_interaction_elastic_net | 183 | 1.731995 | 1.588463 | -0.143532 | -0.138294 |
| fold_2022_h1 | r09_station_offsets_elastic_net | 181 | 1.776539 | 1.693665 | -0.082874 | -0.043988 |
| fold_2022_h1 | r09_spatial_change_elastic_net | 181 | 1.840779 | 1.693665 | -0.147114 | -0.102637 |
| fold_2022_h1 | r09_flow_interaction_elastic_net | 181 | 1.841531 | 1.693665 | -0.147866 | -0.107215 |
| fold_2022_h1 | r09_shallow_boosting_spatial | 181 | 2.016816 | 1.693665 | -0.323151 | -0.258738 |
| fold_2021_h2 | r09_shallow_boosting_spatial | 183 | 2.098048 | 1.588463 | -0.509585 | -0.420260 |

## Spatial Diagnostics

| feature | n | pearson_corr_with_target | mean | p10 | p90 |
| --- | --- | --- | --- | --- | --- |
| temp_network_spread_c | 1051 | 0.199725 | 9.588582 | 7.400000 | 11.500000 |
| temp_network_hko_minus_median_c | 1051 | 0.154511 | 0.308040 | -0.600000 | 1.300000 |
| temp_network_inland_minus_coastal_c | 1051 | 0.091675 | 1.083988 | -0.088636 | 2.201136 |
| temp_network_urban_minus_coastal_c | 1051 | -0.159372 | 0.562895 | -0.083117 | 1.281818 |
| temp_network_east_minus_west_c | 1051 | 0.035031 | -1.561855 | -3.440000 | 0.335000 |
| temp_network_north_minus_south_c | 1051 | 0.062488 | 1.616901 | -0.420000 | 3.400000 |
| sea_breeze_thermal_gradient_proxy | 1275 | -0.072983 | 1.435483 | 0.000000 | 3.408409 |
| hko_local_warm_outlier_score | 1051 | 0.127207 | 0.167641 | -0.302748 | 0.663951 |
