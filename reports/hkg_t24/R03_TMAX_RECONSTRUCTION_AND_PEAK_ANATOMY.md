# EXP-0035 / HKG-T24-R03 Long-Form Experiment Report

## Purpose

R03 is the target-side reconstruction and time-of-maximum anatomy experiment for the HKG T24 Tmax project. It does not build an operational forecasting model. It asks whether the available HKO Headquarters high-frequency temperature feed can reconstruct the official daily maximum temperature label, and it quantifies when the label is difficult to reconstruct from the archived public high-frequency data. This is important because a forecast distribution has to be mapped to official 0.1 C labels. If the archived high-frequency feed misses short-lived peaks or differs semantically from the official climate label, later rounding adapters and probabilistic calibration need to carry that uncertainty explicitly.

## Data Used

The analysis uses only pre-validation target dates from 2020-07-01 through 2023-12-31. It reads HKO Headquarters target labels, HKO Headquarters target-day high-frequency air-temperature rows, target-day since-midnight max rows, and target-day daily mechanism context such as rainfall, King's Park solar radiation, and Waglan wind where present. Every one of these target-day values is treated as label-side or mechanism-only. None is allowed to enter a T-24 operational predictor. Validation 2024 and the 2025-2026 locked test are not read.

The important implementation detail is that this experiment parses the full local day directly from the immutable downloaded DATA.GOV monthly ZIP payloads. It does not use the Phase A/B selected high-frequency table for the reconstruction calculation, because that table was intentionally windowed around 09:00, 12:00, and 15:00 for cutoff-feature engineering. That earlier selected table is still valid for T-24 candidate-feature work, but it is not a full-day target reconstruction table. R03 writes a dedicated derived bronze table at `C:\hkg_tmax_data\bronze\hkg_t24\r03_hko_hq_full_day_high_frequency.parquet`. Rows retain `source_id`, `content_sha256`, `archive_entry_name`, `archive_payload_timestamp_hkt`, station, variable, observed time, local date, role, and an explicit diagnostic-only availability assumption.

The source date range in this report is deliberately narrower than the raw archive. The raw HKO high-frequency archive extends beyond 2023-12-31, but R03 refuses to read validation 2024 and refuses to read the 2025-2026 locked-test period. This protects later one-shot validation discipline and prevents target-day anatomy knowledge from leaking into feature choices for model experiments.

## Leakage Control

The script applies the locked-test guard to the reconstructed daily target dates. It caps the analysis at 2023-12-31 and records zero validation rows and zero locked-test rows. The experiment uses target-day observations, so its outputs are explicitly marked TARGET_ONLY or MECHANISM_ONLY for future modelling. Peak-time labels, peak duration, and reconstructed-target flags may be considered later only as auxiliary training targets, not as T-24 inference features. The strict four-year OOF gate is also evaluated and recorded. For this modern high-frequency sample it is blocked: R03 modern high-frequency pre-validation anatomy period: 3.50 years available, requires at least 4.0 years.

## Reconstruction Method

For each local date, the script filters the full-day parsed `latest_1min_temperature` family to station `HK Observatory` and variable `air_temperature_c`. It computes the maximum observed value, first and last timestamp at that observed maximum, observed row count, median cadence, and a conservative complete-day flag. A day is treated as complete only when it has at least 140 rows, starts near local midnight, ends near 23:40 or later, and has a median cadence consistent with the ten-minute historical snapshots.

This is a high-frequency public snapshot archive, not a dense every-minute raw archive. The name of the feed includes `1min`, but each archived payload is captured roughly every ten minutes and each payload carries the latest one-minute mean at the station. The best reconstruction from the temperature rows is therefore the maximum of the observed ten-minute snapshots, not the true continuous maximum over every minute of the day. That distinction matters: an official daily maximum can occur between public snapshots, especially during short convective breaks, rapid post-rain recovery, sea-breeze transitions, cold-surge rebounds, or brief late-afternoon heating. This is why R03 does not silently replace the official HKO daily Tmax with reconstructed temperature-feed maxima.

The complete-day flag is intentionally strict. It is not meant to make the result look better. It is a quality gate: if a day does not have near-full local-day coverage, the comparison is interpreted as a source-coverage diagnostic instead of a proof of target parity. The max of incomplete snapshots can only be a lower-bound-like diagnostic for the official target.

## Peak Anatomy

R03 records first and last time at the reconstructed maximum, number of peak rows, distinct peak episodes, total peak-duration proxy, and maximum heating over 10, 30, and 60 minutes before the first peak. Peak timing is classified using fixed clock thresholds: early before 12:00, normal from 12:00 through 16:59, and late at 17:00 or later. These thresholds are deliberately not fitted to validation or future data. The current pre-validation sample has the following peak-time counts: {'normal_1200_1659': 892, 'early_before_1200': 361, 'late_1700_or_after': 23}.

## Main Result

The pre-validation sample contains 1276 reconstructed days, of which 661 satisfy the complete-day rule. The overall within-0.1 C agreement rate is 0.5964. On complete days the within-0.1 C agreement rate is 0.7262. The mean absolute official-minus-reconstructed difference is 0.5454 C, the median absolute difference is 0.1000 C, and the maximum absolute difference is 5.3000 C. Since the complete-day agreement and the strict four-year OOF gate do not jointly pass, R03 is not a promotion artifact. It is a diagnostic and source-semantics warning.

The sign of the bias is also informative. A positive official-minus-reconstructed value means the official daily maximum is warmer than the maximum observed in the public snapshot temperature feed. That is the expected failure mode when the public archive samples every several minutes rather than preserving every one-minute mean. This supports a conservative conclusion: the official target table remains the authoritative label, while the full-day public temperature archive is useful for peak-timing anatomy, data-quality flags, and missing-peak risk analysis.

## Max/Min Feed Cross-Check

The experiment also compares the temperature-feed reconstruction with the HKO since-midnight max feed. The since-midnight feed is handled with special care. The raw maximum over all running-feed values can be contaminated by a midnight carryover behavior: early rows just after local midnight may still show the previous day's maximum before the running statistic resets. Therefore R03 stores separate columns for the raw feed maximum, the after-01:00 maximum, the latest observed value, and the late-day final value. The late-day final value is the cleanest available source-side approximation to the final daily running max, while the raw maximum is retained as a warning signal rather than treated as truth.

The late-day since-midnight final value is available on 318 days. Its within-0.1 C rate against the official daily target is 0.9969. Its mean absolute difference is 0.011635220125786161 C, and its maximum absolute difference is 3.6999999999999993 C. R03 detected 136 days where the raw running-feed maximum exceeded the late-day final value, which is the specific signature expected from midnight carryover. Rows where raw max, late final max, or reconstructed temperature-feed max differ materially are written to `artifacts/maxmin_feed_disagreements.csv`. There are 211 such rows in the pre-validation analysis. These rows are not model failures; they are source-behavior evidence that later target adapters and label-side QC need to inspect.

## Stratification

Discrepancies are stratified by season, month, peak-time class, feed completeness, rainfall state, solar state, wind-speed state, and high-temperature tail status. Rain, solar, and wind are target-day mechanism labels only, not predictors. The stratified table is useful for deciding where source semantics are fragile. For example, larger errors on incomplete days would suggest missing high-frequency observations; larger errors on high-solar or high-tail days would suggest short-lived peaks being missed by ten-minute snapshots.

## Interpretation

R03 makes two points clear. First, the official daily target remains the only authoritative label; reconstructed high-frequency maxima are diagnostic and must not silently replace it. Second, the available public high-frequency history is not long enough to satisfy the strict four-year pre-validation OOF rule for modern experiments. That means trajectory, spatial-field, moisture, wind, and pressure experiments that depend on the same modern feed need either a blocked status, a revised predeclared evaluation design, or additional prospective archive time. This is exactly why R03 is written as a mechanism/label audit rather than a model-skill experiment.

The practical implication is not that the public high-frequency archive is useless. It is highly valuable, but the value is different from a direct target substitute. The archive can support as-of-safe T-1 station-state features, trajectory features up to 15:00 on T-1, station-network gradients, humidity/pressure/wind regime indicators, and operational freshness diagnostics. For target-day anatomy, it can show approximate peak time, peak broadness, short-term heating before the sampled peak, and whether the target is likely to be hard to reconstruct from public snapshots. Those outputs are label-side evidence only.

For later modelling, the safe rule is simple. `target_tmax_c` from the official HKO daily target table remains the label. `reconstructed_tmax_c`, first/last peak time, peak episode count, peak-duration proxy, and target-day rain/solar/wind states are not predictors for T-24 inference. They may be used only inside training-fold diagnostics or as auxiliary labels for later experiments that explicitly model peak timing or suppression mechanisms. If an auxiliary task is built later, its folds must be chronological, its transformations must fit only on training dates, and no validation 2024 or locked-test rows may influence feature choice before the one-shot R30 validation gate.

## Artifacts

The main daily reconstruction table is stored under `C:\hkg_tmax_data\gold\hkg_t24\r03_tmax_anatomy\r03_daily_reconstruction.parquet` and copied into this experiment folder. Stratified diagnostics, peak summaries, and max/min disagreement tables are stored beside it and in the experiment `artifacts` directory. The dedicated full-day parsed HKO Headquarters diagnostic table is stored under `C:\hkg_tmax_data\bronze\hkg_t24\r03_hko_hq_full_day_high_frequency.parquet`. The human report is `reports/hkg_t24/R03_TMAX_RECONSTRUCTION_AND_PEAK_ANATOMY.md`. The reproduction command is in `REPRODUCE.md`. The final status is `COMPLETE_LABEL_DIAGNOSTIC_SOURCE_SEMANTICS_INVESTIGATED_OOF_BLOCKED`, not accepted as a production model or challenger.

The row-level reconstruction table contains the date range used by this experiment, official target values, reconstructed snapshot maxima, row counts, first and last observed timestamps, median cadence, completeness flags, peak timing classes, peak episode counts, heating-rate summaries, source hashes, since-midnight final-value diagnostics, target-day mechanism context, and all discrepancy columns. This is the artifact a future reviewer should inspect before deciding whether a later peak-time auxiliary model is justified.

## Next Use

R04 should only proceed as a cutoff-safe trajectory analysis if the evaluation design explicitly handles the modern high-frequency four-year blocker. If R04 is run before that is solved, it must remain a blocked or exploratory mechanism experiment and must not use validation 2024 or locked-test rows. Peak-time and reconstructed-target diagnostics from R03 can be used to define future auxiliary labels, but only on training folds and never at operational inference time.

The exact next safe task is to update the research ledger and then run the focused tests for the guard, governance, peak anatomy, and R03 parser behavior. If those checks pass, the project can proceed to R04 with a clear limitation: modern high-frequency model-skill claims remain blocked under the user's strict four-year OOF requirement until either more lawful history is acquired or the evaluation design is explicitly revised and documented without touching validation 2024 or the locked test.

# R03 Machine-Readable Summary Tables

Generated: `2026-06-20T09:08:30.639411Z`

- Validation 2024 accessed: `false`
- Locked test accessed: `false`
- Analysis period: `2020-07-01` through `2023-12-31`
- Days: `1276`
- Complete days: `661`
- Within 0.1 C rate: `0.5964`
- Complete-day within 0.1 C rate: `0.7262`
- Mean absolute difference: `0.5454` C
- Four-year OOF status: `BLOCKED`

## Stratified Discrepancy

| dimension | value | n | complete_days | mean_abs_diff_c | max_abs_diff_c | within_0p1_rate |
| --- | --- | --- | --- | --- | --- | --- |
| season | DJF | 300 | 235 | 0.326000 | 4.000000 | 0.706667 |
| season | JJA | 338 | 92 | 0.790237 | 5.300000 | 0.467456 |
| season | MAM | 276 | 158 | 0.228261 | 4.600000 | 0.681159 |
| season | SON | 362 | 176 | 0.740331 | 5.300000 | 0.560773 |
| month | 1 | 93 | 87 | 0.072043 | 0.300000 | 0.838710 |
| month | 2 | 84 | 77 | 0.121429 | 0.400000 | 0.630952 |
| month | 3 | 93 | 73 | 0.101075 | 0.500000 | 0.752688 |
| month | 4 | 90 | 45 | 0.126667 | 0.600000 | 0.644444 |
| month | 5 | 93 | 40 | 0.453763 | 4.600000 | 0.645161 |
| month | 6 | 90 | 30 | 0.865556 | 5.300000 | 0.444444 |
| month | 7 | 124 | 31 | 0.829032 | 4.500000 | 0.451613 |
| month | 8 | 124 | 31 | 0.696774 | 4.800000 | 0.500000 |
| month | 9 | 118 | 55 | 0.876271 | 4.500000 | 0.525424 |
| month | 10 | 124 | 62 | 0.587097 | 4.300000 | 0.540323 |
| month | 11 | 120 | 59 | 0.765000 | 5.300000 | 0.616667 |
| month | 12 | 123 | 71 | 0.657724 | 4.000000 | 0.658537 |
| peak_time_class | early_before_1200 | 361 | 73 | 1.632687 | 5.300000 | 0.340720 |
| peak_time_class | late_1700_or_after | 23 | 12 | 0.047826 | 0.200000 | 0.913043 |
| peak_time_class | normal_1200_1659 | 892 | 576 | 0.118161 | 0.800000 | 0.691704 |
| feed_coverage_band | complete | 661 | 661 | 0.108472 | 0.600000 | 0.726172 |
| feed_coverage_band | incomplete | 615 | 0 | 1.014959 | 5.300000 | 0.456911 |
| rain_state | dry_lt_1mm | 719 | 421 | 0.493324 | 4.500000 | 0.598053 |
| rain_state | unknown | 189 | 95 | 0.565608 | 4.600000 | 0.640212 |
| rain_state | wet_ge_1mm | 368 | 145 | 0.636685 | 5.300000 | 0.570652 |
| solar_state | high_solar | 423 | 210 | 0.686288 | 5.300000 | 0.501182 |
| solar_state | low_solar | 422 | 229 | 0.364692 | 5.300000 | 0.713270 |
| solar_state | middle_solar | 431 | 222 | 0.583991 | 4.800000 | 0.575406 |
| wind_speed_state | high_wind | 421 | 252 | 0.468171 | 5.300000 | 0.667458 |
| wind_speed_state | low_wind | 422 | 193 | 0.593602 | 4.800000 | 0.540284 |
| wind_speed_state | middle_wind | 428 | 216 | 0.578972 | 5.300000 | 0.581776 |
| wind_speed_state | unknown | 5 | 0 | 0.100000 | 0.200000 | 0.600000 |
| high_tail_33c_or_more | False | 1076 | 604 | 0.475186 | 5.300000 | 0.622677 |
| high_tail_33c_or_more | True | 200 | 57 | 0.923000 | 4.600000 | 0.455000 |

## Peak-Time Summary

| peak_time_class | n | mean_target_tmax_c | mean_abs_diff_c |
| --- | --- | --- | --- |
| early_before_1200 | 361 | 27.768698 | 1.632687 |
| late_1700_or_after | 23 | 23.186957 | 0.047826 |
| normal_1200_1659 | 892 | 27.332735 | 0.118161 |
