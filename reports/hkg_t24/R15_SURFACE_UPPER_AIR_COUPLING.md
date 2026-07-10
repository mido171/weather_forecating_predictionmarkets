# EXP-0047 / HKG-T24-R15 Long-Form Experiment Report

## Purpose

Surface-Upper-Air Coupling and Mixing-Potential Experiment is a robust long-history continuation experiment for the HKG T-24 Tmax research track. The user constraint for this continuation is stricter than the earlier modern high-frequency diagnostics: a dataset must have at least 39 years of usable history, and the out-of-fold period must cover at least four to five years. This experiment therefore ignores RSS forecasts, radar, satellite, lightning, nowcast, and other current-only families. It uses only source families with enough parsed history to support real chronological stress testing before validation 2024.

The specific research question is: Test whether regional surface state plus upper-air mismatch explains robust heating potential. The target remains Hong Kong Observatory Headquarters official daily maximum temperature for local date T. The forecast cutoff remains T-1 15:00 Asia/Hong_Kong. No Polymarket data, trading logic, market replay, or final validation freeze is touched.

## Data Used

The feature target-date period is `1949-06-03` through `2023-12-31`, giving `74.579` years of usable predictor history for this experiment. The OOF prediction period is `1965-01-01` through `2023-12-31`. The OOF gate is `PASS`: HKG-T24-R15 robust rolling-origin OOF span: 59.00 years available. Validation 2024 is not read. Locked-test dates from 2025-01-01 onward are not read. The generated feature matrix has `23943` rows and `120` columns before fold-local model support filtering.

Input tables are read from the normalized non-minute archive: HKO daily Tmax target labels, NOAA IGRA Hong Kong upper-air sounding features, and NOAA ISD regional station-day cutoff summaries. The ISD table is used only through latest-before-15:00 HKT fields. Full-day ISD daily min/max fields are deliberately excluded because they can contain post-cutoff information. The IGRA relative-humidity fields are deliberately excluded because the normalized table still contains scaling/sentinel anomalies; using them would create a false sense of precision.

## As-Of Contract

Every predictor is either calendar-known, target-history lagged by at least two days, a latest eligible IGRA sounding assigned to origin day T-1, or a regional ISD observation summary using only observations at or before 15:00 local time on origin day T-1. The target label for T is never used as a feature. Target-day weather observations are never used. Daily climate variables for T are never used as predictors. The script calls the locked-test guard on feature dates and prediction dates.

The experiment is still marked proxy-limited rather than production-eligible. IGRA and ISD period-of-record archives are parsed and long-history, but they are retrospective quality-controlled archives rather than exact immutable operational vintages. That means the experiment can produce robust scientific evidence about whether the physical signal exists, while still failing closed for production promotion until publication/release-latency contracts are proven.

## Model Ladder

The first row in the model ladder is a lag/calendar baseline using day-of-year sin/cos and HKO target-history lags that are at least two days old. The remaining rows add only the experiment-specific long-history feature family. Each model is a Ridge regression with median imputation and standard scaling fitted separately inside each chronological training fold. There is no random split, no target-aware feature selection, no validation tuning, and no hyperparameter search. Feature columns must have at least `365` non-null training rows inside a fold before entering that fold.

## Chronological OOF Design

The OOF protocol uses rolling-origin five-year blocks starting in 1965. Each fold trains only on dates before the fold's test window, then scores the next four to five calendar years through 2023. This is intentionally much stricter than the earlier 2020-2023 high-frequency diagnostics. The total scored OOF window is more than five decades for the long-history sources, so single-year luck has much less opportunity to dominate the conclusion. The folds also expose whether a feature works in early, middle, and modern eras rather than only in the recent sample.

## Main Result

The baseline `r15_lag_calendar_baseline` scores MAE `1.7807` C, RMSE `2.2991` C, bias `-0.0910` C, and CRPS `1.2796` over `18624` OOF rows. The best non-baseline row is `r15_coupling_terms` with MAE `1.2432` C, RMSE `1.5731` C, bias `-0.1558` C, and CRPS `0.8848` over `18624` rows. Its MAE improvement versus the baseline is `0.5374` C. This should be interpreted as robust research evidence, not a production release decision.

## Scoreboard

| model_id | n | first_date | last_date | mae | rmse | bias | crps_normal | coverage_80 | coverage_90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r15_coupling_terms | 18624 | 1965-01-01 | 2023-12-31 | 1.243242790847873 | 1.5731185064235462 | -0.1557729144023351 | 0.884794166132352 | 0.8249033505154639 | 0.9147873711340206 |
| r15_upper_air_plus_isd | 18624 | 1965-01-01 | 2023-12-31 | 1.243573062812344 | 1.5757674054206285 | -0.15100511544363382 | 0.885909053075407 | 0.8249570446735395 | 0.9143041237113402 |
| r15_lag_calendar_baseline | 18624 | 1965-01-01 | 2023-12-31 | 1.7806897868242968 | 2.2990878003147515 | -0.09104445159481038 | 1.2795873857720628 | 0.8255476804123711 | 0.9086662371134021 |

## Fold Evidence

| fold_id | model_id | n | mae | baseline_mae | mae_improvement_vs_baseline | crps_improvement_vs_baseline |
| --- | --- | --- | --- | --- | --- | --- |
| fold_1965_1969 | r15_lag_calendar_baseline | 2 | 1.0114035109043886 | 1.0114035109043886 | 0.0 | 0.0 |
| fold_2000_2004 | r15_coupling_terms | 1827 | 1.0900520062036285 | 1.6483941123227797 | 0.5583421061191511 | 0.42017234402054593 |
| fold_2000_2004 | r15_upper_air_plus_isd | 1827 | 1.1026573061459133 | 1.6483941123227797 | 0.5457368061768664 | 0.40932524376681 |
| fold_1985_1989 | r15_upper_air_plus_isd | 1825 | 1.131368214364821 | 1.6862556755971831 | 0.5548874612323622 | 0.41220717268366724 |
| fold_1985_1989 | r15_coupling_terms | 1825 | 1.1315141971817257 | 1.6862556755971831 | 0.5547414784154574 | 0.4121193988868276 |
| fold_1990_1994 | r15_coupling_terms | 1825 | 1.136921636368153 | 1.6960478236478975 | 0.5591261872797446 | 0.41555853962753186 |
| fold_1990_1994 | r15_upper_air_plus_isd | 1825 | 1.1429935786958456 | 1.6960478236478975 | 0.5530542449520519 | 0.41224337793278687 |
| fold_1980_1984 | r15_coupling_terms | 1827 | 1.2191121578178812 | 1.7013560734126456 | 0.4822439155947644 | 0.3582008714336641 |
| fold_1980_1984 | r15_upper_air_plus_isd | 1827 | 1.2226578587451717 | 1.7013560734126456 | 0.47869821466747386 | 0.35761856839042383 |
| fold_1995_1999 | r15_coupling_terms | 1825 | 1.2275278290179656 | 1.7911535370191796 | 0.5636257080012139 | 0.4184717996097017 |
| fold_1995_1999 | r15_upper_air_plus_isd | 1825 | 1.2300887831322929 | 1.7911535370191796 | 0.5610647538868867 | 0.4118204680811016 |
| fold_2015_2019 | r15_coupling_terms | 1826 | 1.248819607376915 | 1.8291799639527127 | 0.5803603565757975 | 0.4094800068491897 |
| fold_2015_2019 | r15_upper_air_plus_isd | 1826 | 1.257891610120095 | 1.8291799639527127 | 0.5712883538326177 | 0.4021670220110921 |
| fold_1975_1979 | r15_coupling_terms | 1826 | 1.2670673297212536 | 1.8465813410031975 | 0.5795140112819439 | 0.4119221422857947 |
| fold_1975_1979 | r15_upper_air_plus_isd | 1826 | 1.269808151444323 | 1.8465813410031975 | 0.5767731895588746 | 0.41004502022694755 |
| fold_2010_2014 | r15_coupling_terms | 1826 | 1.2890743052365226 | 1.8655627206116114 | 0.5764884153750889 | 0.4264067526090275 |
| fold_2010_2014 | r15_upper_air_plus_isd | 1826 | 1.294919487259328 | 1.8655627206116114 | 0.5706432333522835 | 0.42051615676271326 |
| fold_2020_2023 | r15_upper_air_plus_isd | 1460 | 1.3145979852352803 | 1.896806974340663 | 0.5822089891053828 | 0.42215620085022143 |
| fold_2020_2023 | r15_coupling_terms | 1460 | 1.3158373415959683 | 1.896806974340663 | 0.5809696327446947 | 0.42220720950896473 |
| fold_2005_2009 | r15_upper_air_plus_isd | 1826 | 1.387290142560653 | 1.8383408579401932 | 0.4510507153795402 | 0.33568765956849456 |
| fold_2005_2009 | r15_coupling_terms | 1826 | 1.4087632691326348 | 1.8383408579401932 | 0.42957758880755836 | 0.3224161191270509 |
| fold_1970_1974 | r15_upper_air_plus_isd | 729 | 1.4798851910819926 | 1.859238467010347 | 0.37935327592835444 | 0.26988598072944314 |
| fold_1970_1974 | r15_coupling_terms | 729 | 1.5211008092936622 | 1.859238467010347 | 0.3381376577166848 | 0.2403205065211007 |
| fold_2000_2004 | r15_lag_calendar_baseline | 1827 | 1.6483941123227797 | 1.6483941123227797 | 0.0 | 0.0 |
| fold_1985_1989 | r15_lag_calendar_baseline | 1825 | 1.6862556755971831 | 1.6862556755971831 | 0.0 | 0.0 |
| fold_1990_1994 | r15_lag_calendar_baseline | 1825 | 1.6960478236478975 | 1.6960478236478975 | 0.0 | 0.0 |
| fold_1980_1984 | r15_lag_calendar_baseline | 1827 | 1.7013560734126456 | 1.7013560734126456 | 0.0 | 0.0 |
| fold_1995_1999 | r15_lag_calendar_baseline | 1825 | 1.7911535370191796 | 1.7911535370191796 | 0.0 | 0.0 |
| fold_2015_2019 | r15_lag_calendar_baseline | 1826 | 1.8291799639527127 | 1.8291799639527127 | 0.0 | 0.0 |
| fold_2005_2009 | r15_lag_calendar_baseline | 1826 | 1.8383408579401932 | 1.8383408579401932 | 0.0 | 0.0 |
| fold_1975_1979 | r15_lag_calendar_baseline | 1826 | 1.8465813410031975 | 1.8465813410031975 | 0.0 | 0.0 |
| fold_1970_1974 | r15_lag_calendar_baseline | 729 | 1.859238467010347 | 1.859238467010347 | 0.0 | 0.0 |
| fold_2010_2014 | r15_lag_calendar_baseline | 1826 | 1.8655627206116114 | 1.8655627206116114 | 0.0 | 0.0 |
| fold_2020_2023 | r15_lag_calendar_baseline | 1460 | 1.896806974340663 | 1.896806974340663 | 0.0 | 0.0 |
| fold_1965_1969 | r15_upper_air_plus_isd | 2 | 2.4438512993687027 | 1.0114035109043886 | -1.432447788464314 | -1.2789883661619879 |
| fold_1965_1969 | r15_coupling_terms | 2 | 2.4555791263205826 | 1.0114035109043886 | -1.444175615416194 | -1.2646386944063006 |

## Caveats

- Both IGRA and ISD are proxy-limited archives rather than exact live vintages.
- Only ISD latest-before-15:00 local observations are used; full-day ISD daily min/max fields are excluded.

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
| isd_air_temp_mean_c | 23941 | 1949-06-03 | 2023-12-31 | 0.9297820323915756 | 0.925747660224144 |
| igra_temperature_c_1000hpa | 18339 | 1949-06-03 | 2023-12-31 | 0.9197241480748728 | 0.9237016574612764 |
| isd_air_temp_max_c | 23941 | 1949-06-03 | 2023-12-31 | 0.9127237313680492 | 0.9066127320659021 |
| igra_temperature_c_925hpa | 11590 | 1949-11-10 | 2023-12-31 | 0.9104571607211173 | 0.9115473463380523 |
| isd_air_temp_min_c | 23941 | 1949-06-03 | 2023-12-31 | 0.8834613311312758 | 0.8831299603753138 |
| target_tminus2_to_8_mean_c | 23943 | 1949-06-03 | 2023-12-31 | 0.8687183614176663 | 0.8738531914181485 |
| target_tminus2_tmax_c | 23943 | 1949-06-03 | 2023-12-31 | 0.8667456228288045 | 0.8707312490576559 |
| target_tminus2_to_31_mean_c | 23943 | 1949-06-03 | 2023-12-31 | 0.8451947147391302 | 0.8456284285994357 |
| igra_lower_troposphere_mean_temp_c | 23453 | 1949-06-03 | 2023-12-31 | 0.8446823657558452 | 0.8331808325788489 |
| igra_temperature_c_850hpa | 22904 | 1949-06-03 | 2023-12-31 | 0.8417539993485936 | 0.8585912917861593 |
| isd_dew_point_mean_c | 23941 | 1949-06-03 | 2023-12-31 | 0.8392832919465174 | 0.8559163057333943 |
| target_tminus3_tmax_c | 23943 | 1949-06-03 | 2023-12-31 | 0.8362289179059793 | 0.8424628720585948 |
| target_tminus7_tmax_c | 23943 | 1949-06-03 | 2023-12-31 | 0.7936349751844074 | 0.809264436490835 |
| doy_cos | 23943 | 1949-06-03 | 2023-12-31 | -0.7867768612906115 | -0.7976001333152785 |
| isd_pressure_max_hpa | 23939 | 1949-06-03 | 2023-12-31 | -0.7861570658498515 | -0.8028588455099684 |
| isd_pressure_mean_hpa | 23939 | 1949-06-03 | 2023-12-31 | -0.7848224401090499 | -0.7981382182213712 |
| target_tminus14_tmax_c | 23943 | 1949-06-03 | 2023-12-31 | 0.7700135161740861 | 0.783867760799947 |
| igra_temperature_c_700hpa | 23163 | 1949-06-03 | 2023-12-31 | 0.7681333234299355 | 0.7865334720149051 |
| isd_pressure_min_hpa | 23939 | 1949-06-03 | 2023-12-31 | -0.7511814728297218 | -0.7822263072768825 |
| igra_wind_speed_mps_300hpa | 22339 | 1950-01-05 | 2023-12-31 | -0.7507944240839758 | -0.749590050321274 |
| igra_wind_speed_mps_500hpa | 22565 | 1950-01-05 | 2023-12-31 | -0.7271681627926979 | -0.7181835278994196 |
| igra_geopotential_height_m_1000hpa | 17953 | 1949-06-03 | 2023-12-31 | -0.7242080997516588 | -0.7382640477859371 |
| igra_wind_speed_mps_200hpa | 21735 | 1950-01-05 | 2023-12-31 | -0.7076574257887642 | -0.7022609718667429 |
| igra_geopotential_height_m_200hpa | 22817 | 1949-11-02 | 2023-12-31 | 0.696842694918119 | 0.7285103467432722 |
| igra_geopotential_height_m_300hpa | 23072 | 1949-11-02 | 2023-12-31 | 0.6862671796401048 | 0.711305440112222 |
| igra_geopotential_height_m_925hpa | 11556 | 1964-07-22 | 2023-12-31 | -0.6708545102515496 | -0.7054167095905985 |
| r15_surface_minus_igra_850_temp_c | 22902 | 1949-06-03 | 2023-12-31 | 0.6400560288177769 | 0.5891695494967614 |
| igra_temp_850_minus_500_c | 22811 | 1949-06-03 | 2023-12-31 | 0.5923229271340306 | 0.5813912311121009 |
| igra_temperature_c_500hpa | 23277 | 1949-06-03 | 2023-12-31 | 0.5570716849031693 | 0.5793553801851322 |
| igra_geopotential_height_m_500hpa | 23236 | 1949-06-03 | 2023-12-31 | 0.5458121791910242 | 0.5213706882187176 |
| igra_temp_925_minus_850_c | 11568 | 1949-11-10 | 2023-12-31 | 0.5259706495669842 | 0.4297693417020134 |
| igra_geopotential_height_m_850hpa | 22865 | 1949-06-03 | 2023-12-31 | -0.525233700034604 | -0.5669467385194967 |
| igra_temperature_c_300hpa | 23101 | 1949-11-02 | 2023-12-31 | 0.5233067929218269 | 0.5659480120933457 |
| r15_pressure_x_stability | 11565 | 1949-11-10 | 2023-12-31 | 0.520557085594475 | 0.40800814072801633 |
| r15_surface_minus_igra_925_temp_c | 11588 | 1949-11-10 | 2023-12-31 | 0.46578403261192153 | 0.43044235825829236 |
| igra_temperature_c_200hpa | 22840 | 1949-11-02 | 2023-12-31 | 0.46393467270011823 | 0.5109569633191875 |
| igra_dewpoint_depression_c_500hpa | 18026 | 1973-01-02 | 2023-12-31 | -0.41453725046866047 | -0.3723393457328146 |
| igra_temp_700_minus_500_c | 23041 | 1949-06-03 | 2023-12-31 | 0.407709043694677 | 0.34391062486489354 |
| doy_sin | 23943 | 1949-06-03 | 2023-12-31 | -0.3793734024587745 | -0.38374149120626233 |
| igra_boundary_inversion_925_minus_1000_c | 8928 | 1949-11-10 | 2023-12-31 | -0.3545697829884259 | -0.3467901109811328 |
| igra_dewpoint_depression_c_200hpa | 15263 | 1976-05-10 | 2023-12-31 | -0.33965561461218075 | -0.3428282558013365 |
| igra_dewpoint_depression_c_300hpa | 18124 | 1973-01-02 | 2023-12-31 | -0.33599162530374804 | -0.3258290191489701 |
| igra_wind_speed_mps_700hpa | 22554 | 1950-01-05 | 2023-12-31 | -0.3235970925696555 | -0.3212288870858457 |
| isd_wind_speed_max_mps | 23941 | 1949-06-03 | 2023-12-31 | -0.23289800140090525 | -0.22301976778543847 |
| igra_wind_speed_mps_1000hpa | 17558 | 1950-01-05 | 2023-12-31 | -0.20514706861539378 | -0.2316812030106031 |
| igra_key_level_count | 23943 | 1949-06-03 | 2023-12-31 | -0.20112211691298248 | -0.22811666502839306 |
| igra_dewpoint_depression_c_1000hpa | 14065 | 1973-01-02 | 2023-12-31 | -0.19815368005079198 | -0.1216018763940035 |
| isd_pressure_range_hpa | 23939 | 1949-06-03 | 2023-12-31 | -0.184958041268446 | -0.2151366070587567 |
| isd_air_temp_mean_c_change_1d | 23939 | 1949-06-03 | 2023-12-31 | 0.16232209587644292 | 0.10613554365691517 |
| igra_lower_troposphere_mean_dewpoint_depression_c | 18300 | 1973-01-02 | 2023-12-31 | -0.15252944790160003 | -0.043377153297554635 |
