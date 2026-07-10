# EXP-0044 / HKG-T24-R12 Long-Form Experiment Report

## Purpose

R12 tests whether observed solar radiation at King's Park before the T-1 15:00 HKT cutoff adds next-day HKO Headquarters Tmax information beyond the existing season and HKO target-station temperature trajectory. The experiment is intentionally narrow: it does not use target-day sunshine, target-day cloud, or finalized daily radiation totals. It uses only high-frequency solar rows whose conservative available-at time is before the operational cutoff.

## Data Used

The source table is `C:\hkg_tmax_data\bronze\analysis_phase_a\hko_high_frequency_selected_station_observations.parquet`. Within that table, the eligible solar family is `latest_1min_solar`, station `King's Park`, with variables `global_solar_wm2`, `direct_solar_wm2`, and `diffuse_solar_wm2`. Solar observations begin late enough that the modern feature matrix remains short. The R12 feature target-date period is `2020-07-02` through `2023-12-31`, while OOF predictions run from `2021-07-01` through `2023-12-31`. Validation 2024 is not accessed. Locked 2025+ target rows are not accessed.

## Feature Construction

For each target date T, the driver maps solar observations from local day T-1 to target date T and keeps only rows with `available_at_hkt <= T-1 15:00`. Because the repository's HKO high-frequency replay uses a conservative 20-minute availability lag, the latest ordinary solar row that can enter at a 15:00 cutoff is approximately 14:40, not 14:50 or 15:00. The generated features summarize count, mean, maximum, minimum, standard deviation, sum, last value, a sampled kWh proxy, low-radiation fraction, direct/global ratio, diffuse/global ratio, and a heating-efficiency proxy relative to the HKO temperature rise.

## Model Ladder

R12 scores a baseline temperature/calendar Ridge model, a deterministic solar-geometry control, observed global-solar Ridge, observed direct/diffuse/global Ridge, heating-efficiency Ridge, and a shifted-solar negative control. The shifted control uses the previous day's solar features shifted forward by one target day. It is not promotable; it exists to check whether any apparent gain is merely seasonal persistence or alignment-insensitive leakage.

## Leakage Controls

All model parameters, imputation medians, standardization parameters, and Ridge coefficients are fit inside chronological training folds only. Target dates are guarded against locked-test access. Target-day radiation is not used. Full-day daily climate radiation is not used. The only deterministic solar geometry terms are calendar-derived and already present in R04-style features. The model comparison therefore tests observed pre-cutoff radiation against a deterministic season/solar-position control.

## OOF Gate

The strict four-year OOF check is `BLOCKED`: R12 modern solar-radiation pre-validation feature period: 3.50 years available, requires at least 4.0 years. R12 is a completed diagnostic, but it is not promotable under the user's hard four-year reliability rule unless the evaluation design is explicitly changed or more prospective years accumulate.

## Main Result

The baseline row has MAE `1.4723` C and CRPS `1.0512` over `911` rows. The best non-control R12 row is `r12_baseline_temp_calendar` with MAE `1.4723` C, RMSE `1.8861` C, bias `0.0298` C, and CRPS `1.0512` over `911` rows. The key interpretation is the difference between observed-radiation models and the shifted-solar negative control, not the absolute ranking alone.

## Interpretation

If observed global/direct/diffuse radiation beats both baseline and shifted radiation, then T-1 observed heating conditions contain real incremental information. If shifted radiation performs similarly, the signal is mostly seasonal or persistent and should not be treated as a precise cutoff observation effect. If the deterministic geometry control performs almost as well as observed radiation, the observed solar archive may not be adding much beyond day-of-year and target-station temperature state. If heating-efficiency improves only in some folds, it may be a conditional cloud/suppression indicator for R13 rather than a standalone model feature.

## Limitations

Only King's Park solar rows are available in the parsed phase-A table even though the broader source inventory mentions King’s Park and Kau Sai Chau solar products. This experiment therefore tests the currently parsed station, not a complete two-station radiation network. UV data is not parsed into a cutoff-safe feature table here. Cloud and rain suppression are deferred to R13 because target-day daily climate cloud/rain values are retrospective mechanism labels, not lawful T-24 predictors. The short modern OOF span is still the controlling reliability limitation.

## Stability Finding

The important R12 finding is not merely that the overall observed-radiation rows lose to baseline. The fold deltas show why the family is not ready: some later folds show small improvements for global solar or heating-efficiency terms, while other folds degrade materially. That pattern is consistent with radiation being conditionally useful only in particular cloud-transition or clear-heating regimes, not as an unconditional additive predictor. Because R13 cloud/rain/visibility suppression and R20 regime classification have not yet produced lawful regime probabilities, R12 has no safe gate that can decide when to trust solar terms.

## Minimum-Support Rule

The driver enforces a fold-local minimum-support rule: a numeric feature must have at least 30 non-null training rows and more than one distinct value before it can enter a fold. This matters because the solar archive starts close to the first modern fold boundary. Without this rule, early folds can be destabilized by only a handful of solar rows. The rule is conservative and leakage-safe because it is evaluated inside each training fold only. It does not inspect test loss to decide whether a feature is allowed.

## Negative-Control Interpretation

The shifted-solar negative control is deliberately retained even though it is not promotable. If a shifted previous-day radiation model had matched or beaten observed same-origin-day solar, the experiment would strongly suggest that solar features were acting as seasonal proxies rather than genuine cutoff-specific observations. In the generated scoreboard the shifted control is worse than baseline, while observed solar is also worse overall. The combined interpretation is a null result for unconditional solar predictors, not evidence of leakage.

## What Would Be Needed For A Stronger R12

A stronger R12 would need the second solar station parsed, UV parsed with exact availability semantics, cloud-break event features from a denser image/cloud source, and a pre-cutoff regime gate that separates clear dry heating from humid cloudy suppression. It may also need a longer modern archive so that radiation features are not learned from such a short overlap. Those are follow-up engineering tasks, not grounds for using target-day daily sunshine or full-day radiation totals as predictors.

## Production Decision

No R12 solar feature is admitted into the production candidate feature bank. The retained evidence is still valuable: it prevents overconfident assumptions that solar radiation automatically improves tomorrow's Tmax once the 15:00 target-station temperature is already known. It also identifies the exact path for a lawful future retest: parse the remaining radiation/UV/cloud families, build a cloud-suppression gate, and require at least four years of OOF support before promotion.

## Decision Record

R12 is complete as a leakage-safe solar-radiation diagnostic. No validation access occurred, no locked-test target rows were scored, and no predictive feature is promoted because the strict four-year OOF rule blocks promotion. The experiment still provides useful scientific evidence about whether observed pre-cutoff radiation adds information beyond temperature trajectory and deterministic solar geometry.

## Reproducibility

The experiment folder contains OOF predictions, metrics JSON, subgroup/fold metrics, solar diagnostics, feature specification, negative controls, run config, as-of contract, data manifest with hashes, and the reproduction command. The data-root gold directory contains the canonical R12 feature matrix, predictions, scoreboard, fold deltas, and diagnostics for downstream inspection.

# R12 Machine-Readable Summary Tables

Generated: `2026-06-20T10:22:45.119229Z`

## Scoreboard

| model_id | is_control | n | first_date | last_date | mae | rmse | bias | crps_normal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r12_baseline_temp_calendar | False | 911 | 2021-07-01 | 2023-12-31 | 1.4723378363959654 | 1.8860776262540533 | 0.02980135527340544 | 1.0511869648672139 |
| r12_heating_efficiency_ridge | False | 911 | 2021-07-01 | 2023-12-31 | 1.5989597536577846 | 2.0674882944901958 | -0.18376150381846984 | 1.1501670812940399 |
| r12_shifted_solar_negative_control | True | 911 | 2021-07-01 | 2023-12-31 | 1.6585903488708635 | 2.147406008139092 | -0.27034983793784284 | 1.1922854883334904 |
| r12_observed_global_solar_ridge | False | 911 | 2021-07-01 | 2023-12-31 | 1.7883480958679405 | 2.360794305855463 | -0.2686123643580555 | 1.3050120244046413 |
| r12_deterministic_solar_geometry_control | True | 911 | 2021-07-01 | 2023-12-31 | 2.1302761563485766 | 2.746755899335468 | 0.2673219192216845 | 1.5237496672515778 |
| r12_observed_direct_diffuse_solar_ridge | False | 911 | 2021-07-01 | 2023-12-31 | 2.421702682636355 | 4.304891783096195 | -0.9826711884115327 | 1.9216211452315495 |

## Solar Diagnostics

| feature | n | first_date | last_date | pearson_with_target | spearman_with_target |
| --- | --- | --- | --- | --- | --- |
| r12_diffuse_solar_wm2_count | 817 | 2021-06-29 | 2023-12-31 | 0.1752754137080309 | 0.07754615947343685 |
| r12_diffuse_solar_wm2_last | 817 | 2021-06-29 | 2023-12-31 | 0.15949346081757446 | 0.16255739367901686 |
| r12_diffuse_solar_wm2_max | 817 | 2021-06-29 | 2023-12-31 | 0.19972217265085868 | 0.16888923226278033 |
| r12_diffuse_solar_wm2_mean | 817 | 2021-06-29 | 2023-12-31 | 0.27817596660985805 | 0.26733105530412943 |
| r12_diffuse_solar_wm2_min | 817 | 2021-06-29 | 2023-12-31 | 0.3394406854864744 | 0.33606830202677745 |
| r12_diffuse_solar_wm2_std | 817 | 2021-06-29 | 2023-12-31 | 0.045888604162114224 | 0.031414550463076424 |
| r12_diffuse_solar_wm2_sum | 817 | 2021-06-29 | 2023-12-31 | 0.24607301030768375 | 0.26025119100729033 |
| r12_diffuse_to_global_mean_ratio | 817 | 2021-06-29 | 2023-12-31 | -0.2845875434523359 | -0.29146066386941455 |
| r12_direct_solar_wm2_count | 817 | 2021-06-29 | 2023-12-31 | 0.1752754137080309 | 0.07754615947343685 |
| r12_direct_solar_wm2_last | 817 | 2021-06-29 | 2023-12-31 | 0.17147388823026377 | 0.21153942897859818 |
| r12_direct_solar_wm2_max | 817 | 2021-06-29 | 2023-12-31 | 0.3387156644246447 | 0.3161856589614662 |
| r12_direct_solar_wm2_mean | 817 | 2021-06-29 | 2023-12-31 | 0.2257701112958681 | 0.29998581265624324 |
| r12_direct_solar_wm2_min | 817 | 2021-06-29 | 2023-12-31 | 0.008302129039312326 | 0.06680727512413796 |
| r12_direct_solar_wm2_std | 817 | 2021-06-29 | 2023-12-31 | 0.32586556361617863 | 0.3543190718848367 |
| r12_direct_solar_wm2_sum | 817 | 2021-06-29 | 2023-12-31 | 0.23748334680022584 | 0.3655306058802706 |
| r12_direct_to_global_mean_ratio | 817 | 2021-06-29 | 2023-12-31 | 0.15769955886763018 | 0.2318958855380159 |
| r12_global_solar_low_fraction | 818 | 2021-06-29 | 2023-12-31 | -0.37169307611427527 | -0.3253193069764498 |
| r12_global_solar_per_daylight_hour_proxy | 818 | 2021-06-29 | 2023-12-31 | 0.2926180384999487 | 0.34530312641276184 |
| r12_global_solar_sampled_kwh_proxy | 818 | 2021-06-29 | 2023-12-31 | 0.331359253110643 | 0.4110906098478721 |
| r12_global_solar_wm2_count | 818 | 2021-06-29 | 2023-12-31 | 0.17516138345636525 | 0.07955457821942637 |
| r12_global_solar_wm2_last | 818 | 2021-06-29 | 2023-12-31 | 0.32932889517468633 | 0.3416476322264571 |
| r12_global_solar_wm2_max | 818 | 2021-06-29 | 2023-12-31 | 0.5157032264509472 | 0.5701202696207995 |
| r12_global_solar_wm2_mean | 818 | 2021-06-29 | 2023-12-31 | 0.44153600427329276 | 0.4491835185826026 |
| r12_global_solar_wm2_min | 818 | 2021-06-29 | 2023-12-31 | 0.28647822766789555 | 0.28084326847819757 |
| r12_global_solar_wm2_std | 818 | 2021-06-29 | 2023-12-31 | 0.40439015848113913 | 0.4139864341356569 |
| r12_global_solar_wm2_sum | 818 | 2021-06-29 | 2023-12-31 | 0.33135925311064307 | 0.4110906098478721 |
| r12_hko_heating_per_solar_proxy | 633 | 2021-12-31 | 2023-12-31 | -0.08280456370309634 | -0.20029550624079157 |
| solar_cutoff_feature_rows | 818 | 2021-06-29 | 2023-12-31 | nan | nan |