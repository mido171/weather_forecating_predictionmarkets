# EXP-0046 / HKG-T24-R14 Long-Form Experiment Report

## Purpose

Eligible Upper-Air Thermal Potential and Inversion Structure is a robust long-history continuation experiment for the HKG T-24 Tmax research track. The user constraint for this continuation is stricter than the earlier modern high-frequency diagnostics: a dataset must have at least 39 years of usable history, and the out-of-fold period must cover at least four to five years. This experiment therefore ignores RSS forecasts, radar, satellite, lightning, nowcast, and other current-only families. It uses only source families with enough parsed history to support real chronological stress testing before validation 2024.

The specific research question is: Test whether long-history IGRA upper-air thermal structure adds robust next-day Tmax skill. The target remains Hong Kong Observatory Headquarters official daily maximum temperature for local date T. The forecast cutoff remains T-1 15:00 Asia/Hong_Kong. No Polymarket data, trading logic, market replay, or final validation freeze is touched.

## Data Used

The feature target-date period is `1949-06-03` through `2023-12-31`, giving `74.579` years of usable predictor history for this experiment. The OOF prediction period is `1965-01-01` through `2023-12-31`. The OOF gate is `PASS`: HKG-T24-R14 robust rolling-origin OOF span: 59.00 years available. Validation 2024 is not read. Locked-test dates from 2025-01-01 onward are not read. The generated feature matrix has `26632` rows and `120` columns before fold-local model support filtering.

Input tables are read from the normalized non-minute archive: HKO daily Tmax target labels, NOAA IGRA Hong Kong upper-air sounding features, and NOAA ISD regional station-day cutoff summaries. The ISD table is used only through latest-before-15:00 HKT fields. Full-day ISD daily min/max fields are deliberately excluded because they can contain post-cutoff information. The IGRA relative-humidity fields are deliberately excluded because the normalized table still contains scaling/sentinel anomalies; using them would create a false sense of precision.

## As-Of Contract

Every predictor is either calendar-known, target-history lagged by at least two days, a latest eligible IGRA sounding assigned to origin day T-1, or a regional ISD observation summary using only observations at or before 15:00 local time on origin day T-1. The target label for T is never used as a feature. Target-day weather observations are never used. Daily climate variables for T are never used as predictors. The script calls the locked-test guard on feature dates and prediction dates.

The experiment is still marked proxy-limited rather than production-eligible. IGRA and ISD period-of-record archives are parsed and long-history, but they are retrospective quality-controlled archives rather than exact immutable operational vintages. That means the experiment can produce robust scientific evidence about whether the physical signal exists, while still failing closed for production promotion until publication/release-latency contracts are proven.

## Model Ladder

The first row in the model ladder is a lag/calendar baseline using day-of-year sin/cos and HKO target-history lags that are at least two days old. The remaining rows add only the experiment-specific long-history feature family. Each model is a Ridge regression with median imputation and standard scaling fitted separately inside each chronological training fold. There is no random split, no target-aware feature selection, no validation tuning, and no hyperparameter search. Feature columns must have at least `365` non-null training rows inside a fold before entering that fold.

## Chronological OOF Design

The OOF protocol uses rolling-origin five-year blocks starting in 1965. Each fold trains only on dates before the fold's test window, then scores the next four to five calendar years through 2023. This is intentionally much stricter than the earlier 2020-2023 high-frequency diagnostics. The total scored OOF window is more than five decades for the long-history sources, so single-year luck has much less opportunity to dominate the conclusion. The folds also expose whether a feature works in early, middle, and modern eras rather than only in the recent sample.

## Main Result

The baseline `r14_lag_calendar_baseline` scores MAE `1.7854` C, RMSE `2.3105` C, bias `-0.0752` C, and CRPS `1.2853` over `21313` OOF rows. The best non-baseline row is `r14_upper_air_core` with MAE `1.4898` C, RMSE `1.8893` C, bias `-0.0816` C, and CRPS `1.0617` over `21313` rows. Its MAE improvement versus the baseline is `0.2956` C. This should be interpreted as robust research evidence, not a production release decision.

## Scoreboard

| model_id | n | first_date | last_date | mae | rmse | bias | crps_normal | coverage_80 | coverage_90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r14_upper_air_core | 21313 | 1965-01-01 | 2023-12-31 | 1.489762992417958 | 1.889321621831952 | -0.08156071394322333 | 1.0616554729290928 | 0.8489185004457374 | 0.9305118941491108 |
| r14_stability_only | 21313 | 1965-01-01 | 2023-12-31 | 1.5781544881076268 | 1.9935595080431867 | -0.09358775857124485 | 1.1208833563803102 | 0.8371885703561207 | 0.925491484070755 |
| r14_lag_calendar_baseline | 21313 | 1965-01-01 | 2023-12-31 | 1.785398255706591 | 2.3104704931456843 | -0.07523674811701482 | 1.2853195221057376 | 0.8245671655796931 | 0.9074273917327452 |

## Fold Evidence

| fold_id | model_id | n | mae | baseline_mae | mae_improvement_vs_baseline | crps_improvement_vs_baseline |
| --- | --- | --- | --- | --- | --- | --- |
| fold_2000_2004 | r14_upper_air_core | 1827 | 1.277425610571369 | 1.6432479681721353 | 0.3658223576007662 | 0.26460817507105516 |
| fold_2000_2004 | r14_stability_only | 1827 | 1.3734078669211818 | 1.6432479681721353 | 0.2698401012509535 | 0.2079169515222833 |
| fold_1995_1999 | r14_upper_air_core | 1825 | 1.4088810554402291 | 1.788019900046268 | 0.37913884460603886 | 0.2716118696390444 |
| fold_2005_2009 | r14_upper_air_core | 1826 | 1.4174993179813773 | 1.8357366556378154 | 0.4182373376564381 | 0.29507049040253186 |
| fold_1985_1989 | r14_upper_air_core | 1825 | 1.4386905735771367 | 1.6828612344392728 | 0.24417066086213612 | 0.19994823233316183 |
| fold_1980_1984 | r14_upper_air_core | 1827 | 1.4426897843685649 | 1.699012702992063 | 0.2563229186234981 | 0.20043615302318996 |
| fold_2015_2019 | r14_upper_air_core | 1826 | 1.4478766214370524 | 1.8282568801339159 | 0.3803802586968634 | 0.2754781743737529 |
| fold_2010_2014 | r14_upper_air_core | 1826 | 1.4662523383266757 | 1.8627596257119332 | 0.39650728738525753 | 0.30476806356956754 |
| fold_1990_1994 | r14_upper_air_core | 1825 | 1.4723502959039045 | 1.6903368329846382 | 0.21798653708073368 | 0.1680207576752104 |
| fold_1995_1999 | r14_stability_only | 1825 | 1.5029452300576167 | 1.788019900046268 | 0.2850746699886513 | 0.21185975958819614 |
| fold_1980_1984 | r14_stability_only | 1827 | 1.5127223478112304 | 1.699012702992063 | 0.1862903551808326 | 0.15253808429989735 |
| fold_2005_2009 | r14_stability_only | 1826 | 1.5148773438665926 | 1.8357366556378154 | 0.32085931177122284 | 0.23274739524566868 |
| fold_1985_1989 | r14_stability_only | 1825 | 1.5192057208759155 | 1.6828612344392728 | 0.16365551356335728 | 0.13930066081839243 |
| fold_2015_2019 | r14_stability_only | 1826 | 1.5308181223153228 | 1.8282568801339159 | 0.2974387578185931 | 0.2133386181861554 |
| fold_1975_1979 | r14_upper_air_core | 1826 | 1.5513030940290076 | 1.838458503059688 | 0.2871554090306805 | 0.21946199860003346 |
| fold_2020_2023 | r14_upper_air_core | 1461 | 1.5528040259000737 | 1.8974165990834853 | 0.34461257318341154 | 0.260960291912274 |
| fold_2010_2014 | r14_stability_only | 1826 | 1.5683264156372538 | 1.8627596257119332 | 0.2944332100746794 | 0.2326288061840993 |
| fold_1990_1994 | r14_stability_only | 1825 | 1.5766022368328916 | 1.6903368329846382 | 0.11373459615174664 | 0.10234642622618795 |
| fold_2000_2004 | r14_lag_calendar_baseline | 1827 | 1.6432479681721353 | 1.6432479681721353 | 0.0 | 0.0 |
| fold_1970_1974 | r14_upper_air_core | 1814 | 1.6623429180123872 | 1.8069737746838173 | 0.14463085667143005 | 0.11638988546384277 |
| fold_1975_1979 | r14_stability_only | 1826 | 1.6708955426198941 | 1.838458503059688 | 0.1675629604397939 | 0.14037177918667565 |
| fold_2020_2023 | r14_stability_only | 1461 | 1.6780892471308821 | 1.8974165990834853 | 0.2193273519526031 | 0.17644837344451392 |
| fold_1985_1989 | r14_lag_calendar_baseline | 1825 | 1.6828612344392728 | 1.6828612344392728 | 0.0 | 0.0 |
| fold_1990_1994 | r14_lag_calendar_baseline | 1825 | 1.6903368329846382 | 1.6903368329846382 | 0.0 | 0.0 |
| fold_1980_1984 | r14_lag_calendar_baseline | 1827 | 1.699012702992063 | 1.699012702992063 | 0.0 | 0.0 |
| fold_1970_1974 | r14_stability_only | 1814 | 1.7293272981575314 | 1.8069737746838173 | 0.07764647652628587 | 0.07097523902239544 |
| fold_1995_1999 | r14_lag_calendar_baseline | 1825 | 1.788019900046268 | 1.788019900046268 | 0.0 | 0.0 |
| fold_1965_1969 | r14_upper_air_core | 1605 | 1.7890597494662626 | 1.8864843669974702 | 0.09742461753120768 | 0.09882905277176635 |
| fold_1970_1974 | r14_lag_calendar_baseline | 1814 | 1.8069737746838173 | 1.8069737746838173 | 0.0 | 0.0 |
| fold_1965_1969 | r14_stability_only | 1605 | 1.8097039309073493 | 1.8864843669974702 | 0.07678043609012097 | 0.08488122613799587 |
| fold_2015_2019 | r14_lag_calendar_baseline | 1826 | 1.8282568801339159 | 1.8282568801339159 | 0.0 | 0.0 |
| fold_2005_2009 | r14_lag_calendar_baseline | 1826 | 1.8357366556378154 | 1.8357366556378154 | 0.0 | 0.0 |
| fold_1975_1979 | r14_lag_calendar_baseline | 1826 | 1.838458503059688 | 1.838458503059688 | 0.0 | 0.0 |
| fold_2010_2014 | r14_lag_calendar_baseline | 1826 | 1.8627596257119332 | 1.8627596257119332 | 0.0 | 0.0 |
| fold_1965_1969 | r14_lag_calendar_baseline | 1605 | 1.8864843669974702 | 1.8864843669974702 | 0.0 | 0.0 |
| fold_2020_2023 | r14_lag_calendar_baseline | 1461 | 1.8974165990834853 | 1.8974165990834853 | 0.0 | 0.0 |

## Caveats

- NOAA IGRA archive is parsed and long-history, but exact operational release latency before the T-1 15:00 HKT cutoff remains unproven.
- IGRA relative-humidity columns are intentionally excluded because the normalized archive shows scaling/sentinel issues.

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
| igra_temperature_c_1000hpa | 20388 | 1949-06-03 | 2023-12-31 | 0.916371195663577 | 0.9209133062784223 |
| igra_temperature_c_925hpa | 11637 | 1949-11-10 | 2023-12-31 | 0.9100549651487074 | 0.9111472736626975 |
| target_tminus2_to_8_mean_c | 26632 | 1949-06-03 | 2023-12-31 | 0.869804997207918 | 0.8750177535558259 |
| target_tminus2_tmax_c | 26632 | 1949-06-03 | 2023-12-31 | 0.8664560703599953 | 0.8703486996393646 |
| target_tminus2_to_31_mean_c | 26632 | 1949-06-03 | 2023-12-31 | 0.8449508716300188 | 0.8458256465521394 |
| igra_temperature_c_850hpa | 25501 | 1949-06-03 | 2023-12-31 | 0.8368461693926672 | 0.8534774705103648 |
| target_tminus3_tmax_c | 26632 | 1949-06-03 | 2023-12-31 | 0.836205146650945 | 0.8425130997981672 |
| igra_lower_troposphere_mean_temp_c | 26092 | 1949-06-03 | 2023-12-31 | 0.8331409357544208 | 0.8189470639149559 |
| target_tminus7_tmax_c | 26632 | 1949-06-03 | 2023-12-31 | 0.7952616900819528 | 0.8109844183891839 |
| doy_cos | 26632 | 1949-06-03 | 2023-12-31 | -0.7852203366121794 | -0.7968295383662304 |
| target_tminus14_tmax_c | 26632 | 1949-06-03 | 2023-12-31 | 0.7689416131024845 | 0.7838362658079154 |
| igra_temperature_c_700hpa | 25721 | 1949-06-03 | 2023-12-31 | 0.7671480636508082 | 0.7848293855281236 |
| igra_wind_speed_mps_300hpa | 24606 | 1950-01-05 | 2023-12-31 | -0.7500460944717318 | -0.7505810091449103 |
| igra_wind_speed_mps_500hpa | 24941 | 1950-01-05 | 2023-12-31 | -0.7266559145598526 | -0.7168064066698033 |
| igra_geopotential_height_m_1000hpa | 20006 | 1949-06-03 | 2023-12-31 | -0.720043907825127 | -0.7342759140515054 |
| igra_wind_speed_mps_200hpa | 23954 | 1950-01-05 | 2023-12-31 | -0.7082975559686358 | -0.703199655856477 |
| igra_geopotential_height_m_200hpa | 25168 | 1949-11-02 | 2023-12-31 | 0.6942598314776535 | 0.7251572427703525 |
| igra_geopotential_height_m_300hpa | 25507 | 1949-11-02 | 2023-12-31 | 0.6823475346723233 | 0.7053252110283215 |
| igra_geopotential_height_m_925hpa | 11579 | 1964-07-22 | 2023-12-31 | -0.6708680209391007 | -0.705436355563901 |
| igra_temp_850_minus_500_c | 25330 | 1949-06-03 | 2023-12-31 | 0.5866757637617613 | 0.5759540519816455 |
| igra_temperature_c_500hpa | 25824 | 1949-06-03 | 2023-12-31 | 0.5557289462467045 | 0.5777176311045658 |
| igra_geopotential_height_m_500hpa | 25792 | 1949-06-03 | 2023-12-31 | 0.5450760485244298 | 0.517344135579941 |
| igra_temp_925_minus_850_c | 11611 | 1949-11-10 | 2023-12-31 | 0.5258491516166983 | 0.4296048744224404 |
| igra_geopotential_height_m_850hpa | 25465 | 1949-06-03 | 2023-12-31 | -0.5213092409502005 | -0.564432063493065 |
| igra_temperature_c_300hpa | 25518 | 1949-11-02 | 2023-12-31 | 0.5203670533000486 | 0.5634622285678084 |
| igra_temperature_c_200hpa | 25193 | 1949-11-02 | 2023-12-31 | 0.4643068199807129 | 0.5118637253627718 |
| igra_dewpoint_depression_c_500hpa | 18689 | 1971-01-02 | 2023-12-31 | -0.4133249032143631 | -0.37236002442645083 |
| igra_temp_700_minus_500_c | 25553 | 1949-06-03 | 2023-12-31 | 0.4082847919470444 | 0.3453477751707557 |
| doy_sin | 26632 | 1949-06-03 | 2023-12-31 | -0.38372058438347295 | -0.38748858018934706 |
| igra_boundary_inversion_925_minus_1000_c | 8969 | 1949-11-10 | 2023-12-31 | -0.3542709938924084 | -0.34660199072449444 |
| igra_dewpoint_depression_c_200hpa | 15264 | 1976-05-10 | 2023-12-31 | -0.3396610517004519 | -0.3428416437349924 |
| igra_dewpoint_depression_c_300hpa | 18780 | 1971-01-02 | 2023-12-31 | -0.33681483582568666 | -0.32705702793853536 |
| igra_wind_speed_mps_700hpa | 25050 | 1950-01-05 | 2023-12-31 | -0.3250081248786989 | -0.32238641148416974 |
| igra_key_level_count | 26632 | 1949-06-03 | 2023-12-31 | -0.2036784299983229 | -0.23482491768673627 |
| igra_dewpoint_depression_c_1000hpa | 14612 | 1971-01-02 | 2023-12-31 | -0.19869630180417427 | -0.12328823633649488 |
| igra_wind_speed_mps_1000hpa | 19444 | 1950-01-05 | 2023-12-31 | -0.17431518789758144 | -0.21634643288610236 |
| igra_lower_troposphere_mean_dewpoint_depression_c | 18995 | 1971-01-02 | 2023-12-31 | -0.15088076246149995 | -0.042363945527118425 |
| igra_dewpoint_depression_c_700hpa | 18592 | 1971-01-02 | 2023-12-31 | -0.1320830550934004 | 0.008741771726001052 |
| igra_dewpoint_depression_c_925hpa | 11572 | 1971-01-17 | 2023-12-31 | -0.12809014907236227 | -0.016480584753133316 |
| target_tminus2_minus_tminus7_c | 26632 | 1949-06-03 | 2023-12-31 | 0.11500368687224048 | 0.09379594431678398 |
| igra_wind_speed_mps_925hpa | 11329 | 1973-02-25 | 2023-12-31 | -0.10058642466723933 | -0.13758003672852312 |
| igra_wind_speed_mps_850hpa | 24848 | 1950-01-05 | 2023-12-31 | 0.0830048641988503 | 0.0741270383607292 |
| igra_dewpoint_depression_c_850hpa | 18415 | 1971-01-02 | 2023-12-31 | -0.07907201580247453 | 0.09440267098997734 |
| igra_geopotential_height_m_700hpa | 25645 | 1949-06-03 | 2023-12-31 | 0.0013159237137929237 | -0.010298558146204654 |
