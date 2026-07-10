# EXP-0048 / HKG-T24-R16 Long-Form Experiment Report

## Purpose

Fifty-Year Regional ISD Surface Core is a robust long-history continuation experiment for the HKG T-24 Tmax research track. The user constraint for this continuation is stricter than the earlier modern high-frequency diagnostics: a dataset must have at least 39 years of usable history, and the out-of-fold period must cover at least four to five years. This experiment therefore ignores RSS forecasts, radar, satellite, lightning, nowcast, and other current-only families. It uses only source families with enough parsed history to support real chronological stress testing before validation 2024.

The specific research question is: Test whether long-history regional surface observations add robust skill across multiple eras. The target remains Hong Kong Observatory Headquarters official daily maximum temperature for local date T. The forecast cutoff remains T-1 15:00 Asia/Hong_Kong. No Polymarket data, trading logic, market replay, or final validation freeze is touched.

## Data Used

The feature target-date period is `1947-01-01` through `2023-12-31`, giving `76.999` years of usable predictor history for this experiment. The OOF prediction period is `1965-01-01` through `2023-12-31`. The OOF gate is `PASS`: HKG-T24-R16 robust rolling-origin OOF span: 59.00 years available. Validation 2024 is not read. Locked-test dates from 2025-01-01 onward are not read. The generated feature matrix has `25202` rows and `120` columns before fold-local model support filtering.

Input tables are read from the normalized non-minute archive: HKO daily Tmax target labels, NOAA IGRA Hong Kong upper-air sounding features, and NOAA ISD regional station-day cutoff summaries. The ISD table is used only through latest-before-15:00 HKT fields. Full-day ISD daily min/max fields are deliberately excluded because they can contain post-cutoff information. The IGRA relative-humidity fields are deliberately excluded because the normalized table still contains scaling/sentinel anomalies; using them would create a false sense of precision.

## As-Of Contract

Every predictor is either calendar-known, target-history lagged by at least two days, a latest eligible IGRA sounding assigned to origin day T-1, or a regional ISD observation summary using only observations at or before 15:00 local time on origin day T-1. The target label for T is never used as a feature. Target-day weather observations are never used. Daily climate variables for T are never used as predictors. The script calls the locked-test guard on feature dates and prediction dates.

The experiment is still marked proxy-limited rather than production-eligible. IGRA and ISD period-of-record archives are parsed and long-history, but they are retrospective quality-controlled archives rather than exact immutable operational vintages. That means the experiment can produce robust scientific evidence about whether the physical signal exists, while still failing closed for production promotion until publication/release-latency contracts are proven.

## Model Ladder

The first row in the model ladder is a lag/calendar baseline using day-of-year sin/cos and HKO target-history lags that are at least two days old. The remaining rows add only the experiment-specific long-history feature family. Each model is a Ridge regression with median imputation and standard scaling fitted separately inside each chronological training fold. There is no random split, no target-aware feature selection, no validation tuning, and no hyperparameter search. Feature columns must have at least `365` non-null training rows inside a fold before entering that fold.

## Chronological OOF Design

The OOF protocol uses rolling-origin five-year blocks starting in 1965. Each fold trains only on dates before the fold's test window, then scores the next four to five calendar years through 2023. This is intentionally much stricter than the earlier 2020-2023 high-frequency diagnostics. The total scored OOF window is more than five decades for the long-history sources, so single-year luck has much less opportunity to dominate the conclusion. The folds also expose whether a feature works in early, middle, and modern eras rather than only in the recent sample.

## Main Result

The baseline `r16_lag_calendar_baseline` scores MAE `1.7803` C, RMSE `2.2987` C, bias `-0.0884` C, and CRPS `1.2793` over `18627` OOF rows. The best non-baseline row is `r16_isd_regional_aggregate` with MAE `1.2585` C, RMSE `1.5986` C, bias `-0.1178` C, and CRPS `0.8987` over `18627` rows. Its MAE improvement versus the baseline is `0.5218` C. This should be interpreted as robust research evidence, not a production release decision.

## Scoreboard

| model_id | n | first_date | last_date | mae | rmse | bias | crps_normal | coverage_80 | coverage_90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r16_isd_regional_aggregate | 18627 | 1965-01-01 | 2023-12-31 | 1.2585350428359678 | 1.5985689059522357 | -0.11784092375688587 | 0.8986698873221821 | 0.8257905191388845 | 0.915928490900306 |
| r16_isd_station_panel | 18627 | 1965-01-01 | 2023-12-31 | 1.2585350428359678 | 1.5985689059522357 | -0.11784092375688587 | 0.8986698873221821 | 0.8257905191388845 | 0.915928490900306 |
| r16_lag_calendar_baseline | 18627 | 1965-01-01 | 2023-12-31 | 1.7802992704002185 | 2.298735287146762 | -0.08840412019768487 | 1.2792790488555388 | 0.8255757770977613 | 0.908519890481559 |

## Fold Evidence

| fold_id | model_id | n | mae | baseline_mae | mae_improvement_vs_baseline | crps_improvement_vs_baseline |
| --- | --- | --- | --- | --- | --- | --- |
| fold_1965_1969 | r16_lag_calendar_baseline | 2 | 1.0387723402602234 | 1.0387723402602234 | 0.0 | 0.0 |
| fold_2000_2004 | r16_isd_regional_aggregate | 1827 | 1.1219368406347496 | 1.6485499617207986 | 0.526613121086049 | 0.39382511149832233 |
| fold_2000_2004 | r16_isd_station_panel | 1827 | 1.1219368406347496 | 1.6485499617207986 | 0.526613121086049 | 0.39382511149832233 |
| fold_1985_1989 | r16_isd_regional_aggregate | 1826 | 1.1410531860491349 | 1.6860798147391374 | 0.5450266286900025 | 0.40210434281773544 |
| fold_1985_1989 | r16_isd_station_panel | 1826 | 1.1410531860491349 | 1.6860798147391374 | 0.5450266286900025 | 0.40210434281773544 |
| fold_1990_1994 | r16_isd_regional_aggregate | 1826 | 1.161260686372279 | 1.6949231234485693 | 0.5336624370762904 | 0.39504804328806353 |
| fold_1990_1994 | r16_isd_station_panel | 1826 | 1.161260686372279 | 1.6949231234485693 | 0.5336624370762904 | 0.39504804328806353 |
| fold_1980_1984 | r16_isd_regional_aggregate | 1827 | 1.2276729502183272 | 1.701071862609885 | 0.47339891239155785 | 0.346258896824006 |
| fold_1980_1984 | r16_isd_station_panel | 1827 | 1.2276729502183272 | 1.701071862609885 | 0.47339891239155785 | 0.346258896824006 |
| fold_1995_1999 | r16_isd_regional_aggregate | 1826 | 1.2285358227013174 | 1.7911876964003708 | 0.5626518736990533 | 0.41232272647985035 |
| fold_1995_1999 | r16_isd_station_panel | 1826 | 1.2285358227013174 | 1.7911876964003708 | 0.5626518736990533 | 0.41232272647985035 |
| fold_1975_1979 | r16_isd_regional_aggregate | 1826 | 1.2632055059785565 | 1.844919327510814 | 0.5817138215322575 | 0.40415579929161494 |
| fold_1975_1979 | r16_isd_station_panel | 1826 | 1.2632055059785565 | 1.844919327510814 | 0.5817138215322575 | 0.40415579929161494 |
| fold_2015_2019 | r16_isd_regional_aggregate | 1826 | 1.3034484191324192 | 1.8289517353985236 | 0.5255033162661045 | 0.37296755775392354 |
| fold_2015_2019 | r16_isd_station_panel | 1826 | 1.3034484191324192 | 1.8289517353985236 | 0.5255033162661045 | 0.37296755775392354 |
| fold_2010_2014 | r16_isd_regional_aggregate | 1826 | 1.3345053058821486 | 1.865783991359795 | 0.5312786854776463 | 0.3942208234990108 |
| fold_2010_2014 | r16_isd_station_panel | 1826 | 1.3345053058821486 | 1.865783991359795 | 0.5312786854776463 | 0.3942208234990108 |
| fold_2020_2023 | r16_isd_regional_aggregate | 1460 | 1.3478166416000643 | 1.8973581512902011 | 0.5495415096901368 | 0.4008709487528114 |
| fold_2020_2023 | r16_isd_station_panel | 1460 | 1.3478166416000643 | 1.8973581512902011 | 0.5495415096901368 | 0.4008709487528114 |
| fold_1970_1974 | r16_isd_regional_aggregate | 729 | 1.3635709775665694 | 1.8569789811455308 | 0.4934080035789614 | 0.3347444552656922 |
| fold_1970_1974 | r16_isd_station_panel | 729 | 1.3635709775665694 | 1.8569789811455308 | 0.4934080035789614 | 0.3347444552656922 |
| fold_2005_2009 | r16_isd_regional_aggregate | 1826 | 1.4301910137384823 | 1.8379447349507887 | 0.40775372121230635 | 0.3088586938957296 |
| fold_2005_2009 | r16_isd_station_panel | 1826 | 1.4301910137384823 | 1.8379447349507887 | 0.40775372121230635 | 0.3088586938957296 |
| fold_2000_2004 | r16_lag_calendar_baseline | 1827 | 1.6485499617207986 | 1.6485499617207986 | 0.0 | 0.0 |
| fold_1985_1989 | r16_lag_calendar_baseline | 1826 | 1.6860798147391374 | 1.6860798147391374 | 0.0 | 0.0 |
| fold_1990_1994 | r16_lag_calendar_baseline | 1826 | 1.6949231234485693 | 1.6949231234485693 | 0.0 | 0.0 |
| fold_1980_1984 | r16_lag_calendar_baseline | 1827 | 1.701071862609885 | 1.701071862609885 | 0.0 | 0.0 |
| fold_1995_1999 | r16_lag_calendar_baseline | 1826 | 1.7911876964003708 | 1.7911876964003708 | 0.0 | 0.0 |
| fold_2015_2019 | r16_lag_calendar_baseline | 1826 | 1.8289517353985236 | 1.8289517353985236 | 0.0 | 0.0 |
| fold_2005_2009 | r16_lag_calendar_baseline | 1826 | 1.8379447349507887 | 1.8379447349507887 | 0.0 | 0.0 |
| fold_1975_1979 | r16_lag_calendar_baseline | 1826 | 1.844919327510814 | 1.844919327510814 | 0.0 | 0.0 |
| fold_1970_1974 | r16_lag_calendar_baseline | 729 | 1.8569789811455308 | 1.8569789811455308 | 0.0 | 0.0 |
| fold_2010_2014 | r16_lag_calendar_baseline | 1826 | 1.865783991359795 | 1.865783991359795 | 0.0 | 0.0 |
| fold_2020_2023 | r16_lag_calendar_baseline | 1460 | 1.8973581512902011 | 1.8973581512902011 | 0.0 | 0.0 |
| fold_1965_1969 | r16_isd_regional_aggregate | 2 | 2.8812627304297775 | 1.0387723402602234 | -1.8424903901695542 | -1.589464350098881 |
| fold_1965_1969 | r16_isd_station_panel | 2 | 2.8812627304297775 | 1.0387723402602234 | -1.8424903901695542 | -1.589464350098881 |

## Caveats

- NOAA ISD annual archive is quality-controlled and not an exact historical operational feed.
- Station-specific panel columns are chosen by predictor availability only, not by target outcome.

## Interpretation

The important question is not only whether one overall row has a lower MAE. A robust feature family should improve or at least not degrade across many chronological folds, should not win only because of a single modern interval, and should not behave like an accidental proxy for year or missingness. The fold-delta artifact in this experiment folder is therefore as important as the headline scoreboard. A small positive overall delta with unstable fold signs is treated as weak evidence. A stable positive delta across old and modern folds is stronger evidence that the source family deserves further engineering.

For upper-air features, a plausible physical signal is lower-tropospheric warmth, inversion structure, and midlevel stability influencing the next day's heating ceiling. For ISD surface features, a plausible signal is regional air mass state before the cutoff: broad warmth, dewpoint spread, pressure regime, and wind exposure across nearby stations. For coupling features, the core idea is surface-to-aloft mismatch: a warm surface under a stable cap, a cool moist surface under warm air aloft, or regional pressure/wind conditions that modulate mixing. For era-transfer features, the key question is whether simple long-history relationships are stable across reporting regimes and urbanization eras.

## Leakage Review

The runner does not inspect validation 2024. It does not score locked-test dates from 2025 onward. It does not use target-day climate elements, retrospective best tracks, target-day full-day aggregates, reanalysis, model-analysis fields, market outcomes, or post-hoc selected validation errors. The feature support filter is evaluated inside each training fold using predictor availability and numeric variation only. Imputation, scaling, and coefficients are also fit inside each training fold only. The output remains marked non-production because archive vintage timing is not yet exact enough for live deployment.

## What Was Deliberately Not Done

The experiment does not try to rescue short-history modern data by mixing it into a long-history score. It does not backfill RSS forecasts before 2020. It does not pretend June 2026 radar/satellite snapshots can support 1965-2023 OOF. It does not use full-day ISD min/max as if they were known by 15:00. It does not promote any final challenger, and it does not authorize R30 validation. This discipline is what makes the result useful rather than just numerically impressive.

## Decision

The experiment status is `COMPLETE_ROBUST_LONG_HISTORY_PROXY_LIMITED`. It passes the user's robust-history and OOF-span requirements, but it is not production-eligible until the remaining point-in-time release/vintage caveats are resolved. The next safe use of this output is to compare fold stability, subgroup behavior, and feature diagnostics across R14-R17, then decide which long-history source family deserves a stricter as-of-contract hardening pass.

## Reproducibility

The experiment folder contains the local OOF predictions, scoreboard, fold deltas, subgroup metrics, feature diagnostics, hashes for the generated prediction and feature tables, run configuration, protocol, as-of contract, data manifest, status file, and this long-form report. The repo-level report mirrors the same content for handoff reading. The reproduction command is stored in `REPRODUCE.md`.


## Feature Diagnostics

| feature | n | first_date | last_date | pearson_with_target | spearman_with_target |
| --- | --- | --- | --- | --- | --- |
| isd_air_temp_mean_c | 25200 | 1947-01-01 | 2023-12-31 | 0.928799071034803 | 0.925051106119111 |
| isd_air_temp_max_c | 25200 | 1947-01-01 | 2023-12-31 | 0.9096243552229764 | 0.9033821811895519 |
| isd_air_temp_min_c | 25200 | 1947-01-01 | 2023-12-31 | 0.8796252415368198 | 0.8804400819909362 |
| target_tminus2_to_8_mean_c | 25202 | 1947-01-01 | 2023-12-31 | 0.8680841600119799 | 0.8735772545452974 |
| target_tminus2_tmax_c | 25202 | 1947-01-01 | 2023-12-31 | 0.8660215050517643 | 0.8699287324505506 |
| target_tminus2_to_31_mean_c | 25202 | 1947-01-01 | 2023-12-31 | 0.8444332300622142 | 0.8454489940119325 |
| isd_dew_point_mean_c | 25200 | 1947-01-01 | 2023-12-31 | 0.8377764249996892 | 0.8554732027048788 |
| target_tminus3_tmax_c | 25202 | 1947-01-01 | 2023-12-31 | 0.8356382842698964 | 0.8419389030294834 |
| target_tminus7_tmax_c | 25202 | 1947-01-01 | 2023-12-31 | 0.7925464454960308 | 0.8087753876228383 |
| doy_cos | 25202 | 1947-01-01 | 2023-12-31 | -0.7850465559285525 | -0.7963278340217878 |
| isd_pressure_max_hpa | 25198 | 1947-01-01 | 2023-12-31 | -0.7831874235515319 | -0.8009683808292415 |
| isd_pressure_mean_hpa | 25198 | 1947-01-01 | 2023-12-31 | -0.7827350977513801 | -0.7976292266999484 |
| target_tminus14_tmax_c | 25202 | 1947-01-01 | 2023-12-31 | 0.7690763725383143 | 0.7835971817231535 |
| isd_pressure_min_hpa | 25198 | 1947-01-01 | 2023-12-31 | -0.7499897257149298 | -0.7824643463177231 |
| doy_sin | 25202 | 1947-01-01 | 2023-12-31 | -0.3798245260024143 | -0.3843972112567039 |
| isd_wind_speed_max_mps | 25199 | 1947-01-01 | 2023-12-31 | -0.2221190540191073 | -0.21698804188018872 |
| isd_pressure_range_hpa | 25198 | 1947-01-01 | 2023-12-31 | -0.17362781220956192 | -0.19259649295378084 |
| isd_air_temp_mean_c_change_1d | 25198 | 1947-01-01 | 2023-12-31 | 0.16090749125098322 | 0.10672035858491008 |
| isd_dew_point_mean_c_change_1d | 25198 | 1947-01-01 | 2023-12-31 | 0.14916068180382694 | 0.023753511150102255 |
| isd_pressure_mean_hpa_change_1d | 25194 | 1947-01-01 | 2023-12-31 | -0.14418475351308632 | -0.08828173193199261 |
| isd_air_temp_std_c | 23471 | 1947-01-01 | 2023-12-31 | -0.14407111414569823 | -0.09737116298562201 |
| target_tminus2_minus_tminus7_c | 25202 | 1947-01-01 | 2023-12-31 | 0.11781074890131525 | 0.09674745194964395 |
| isd_temp_dewpoint_spread_mean_c | 25200 | 1947-01-01 | 2023-12-31 | -0.11126693982575207 | -0.027838746305161897 |
| isd_wind_u_mean_mps | 25199 | 1947-01-01 | 2023-12-31 | 0.10462574038750573 | 0.07386506861557525 |
| isd_wind_v_mean_mps | 25199 | 1947-01-01 | 2023-12-31 | 0.10462574038750572 | 0.07385863232124638 |
| isd_wind_speed_mean_mps | 25199 | 1947-01-01 | 2023-12-31 | -0.10454742907935713 | -0.07377522302411577 |
| isd_air_temp_range_c | 25200 | 1947-01-01 | 2023-12-31 | -0.0951913263356635 | -0.06064607955235891 |
| isd_obs_count_sum | 25202 | 1947-01-01 | 2023-12-31 | 0.05671379355824218 | 0.04679352192756343 |
| isd_station_count | 25202 | 1947-01-01 | 2023-12-31 | 0.03106840419111028 | 0.010181682461292467 |
