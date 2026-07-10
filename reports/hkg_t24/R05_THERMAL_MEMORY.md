# EXP-0037 / HKG-T24-R05 Long-Form Experiment Report

## Purpose

R05 tests multi-day thermal memory for the HKO T-24 Tmax problem. R04 asked whether the shape of the current T-1 cutoff-day curve adds value beyond the latest eligible temperature. R05 asks whether the previous two to seven cutoff-safe thermal states carry additional next-day information, and whether persistence becomes dangerous around abrupt transitions. It remains fully independent of Polymarket work, validation 2024, and locked-test rows.

## Data and Eligibility

The experiment uses the R04 cutoff-safe feature matrix as its source. For target day T, `lag1_cutoff_temp_c` is the T-1 HKO latest eligible temperature at the 15:00 cutoff, with the latest ordinary observed timestamp capped at 14:40 under the +20 minute replay latency rule. Lags 2 through 7 are prior cutoff-safe states from earlier origin dates. These are operationally plausible station features. Lagged official daily Tmax labels are not used as operational predictors here, because their publication timing has not been proven for the T-1 15:00 cutoff. That is a deliberate exclusion, not an omission.

## Features

R05 constructs lagged cutoff temperatures for lags 1 through 7, lagged intraday range and standard-deviation summaries, lag1-minus-lagN differences, 3/5/7-day mean, range, standard deviation, and trend of cutoff temperature, exponentially weighted thermal levels with half-lives 1, 2, 3, and 5 days, an absolute lag1-lag2 transition magnitude, a binary large-change candidate at 1.5 C, and a regime-duration counter since the last large cutoff-temperature change. These are simple, interpretable memory features designed to test persistence half-life without fitting a large opaque model.

## Models and Ablations

The diagnostic model ladder is deliberately compact. The baseline uses lag1 cutoff temperature, deterministic calendar seasonality, day length, and noon solar elevation. The 1-3 day memory model adds lag2, lag3, short differences, and 3-day memory summaries. The 1-7 day memory model adds all lag and memory features. The EWMA/gated memory model uses exponentially weighted memory and transition/regime-duration summaries. All models use Ridge regression with alpha 1.0, median imputation, and standardization fitted inside each chronological fold only.

## OOF Design and Gate

The chronological folds are inherited from R04: 2021-H2, 2022-H1, 2022-H2, 2023-H1, and 2023-H2, each trained only on earlier dates. The strict four-year feasibility check is `BLOCKED`: R05 modern HKO thermal memory pre-validation feature period: 3.48 years available, requires at least 4.0 years. R05 therefore cannot promote a memory feature family even if it improves in the blocked diagnostic folds. It is complete as an evidence-generating experiment and blocked as a promotable modern high-frequency model experiment.

## Main Result

The best diagnostic model by OOF MAE is `r05_baseline_lag1_cutoff_temp_calendar` with MAE `1.5373` C, RMSE `1.9642` C, bias `0.2004` C, and CRPS `1.0962` over `911` rows. The result must be read beside the fold-delta table. A true promotion would require stable improvement across at least three chronological folds and enough OOF coverage; the strict OOF coverage criterion is not met.

## Interpretation

The point of R05 is to separate useful thermal memory from dangerous stale persistence. If short lags improve over lag1 alone, that suggests stable warm or cool regimes carry information beyond the current cutoff snapshot. If longer memory hurts, it suggests old thermal state becomes stale around transitions. If EWMA features help only in stable folds, they should become gated expert inputs rather than universal predictors. If no memory model beats lag1, the latest cutoff state remains the best target-station summary and attention should move to moisture, pressure, wind, station-network, upper-air, and forecast-vintage signals.

## Publication-Timing Blocker

Lagged official daily Tmax labels are tempting because they provide a long and smooth memory signal. R05 does not use them as operational predictors. The HKO daily target is a label source, and T-1 daily values are not known at T-1 15:00. T-2/T-3 daily labels might eventually be usable, but only after empirical publication timing proves they were available before cutoff for each historical row. Until then, any lagged official-label memory experiment must be separately marked target-history or mechanism-only. This report keeps that separation explicit.

## Artifacts

The feature matrix is stored at `C:\hkg_tmax_data\gold\hkg_t24\r05_thermal_memory\r05_feature_matrix.parquet`. OOF predictions, scoreboards, fold deltas, and memory-decay diagnostics are stored under the same data-root folder and copied or summarized in the experiment directory. The repo-level report is `reports/hkg_t24/R05_THERMAL_MEMORY.md`. The reproduction command is in `REPRODUCE.md`.

## Date Ranges Used

The feature target-date period is `2020-07-08` through `2023-12-31`. The OOF prediction period is `2021-07-01` through `2023-12-31`. This narrower prediction period starts after the warm-up required by chronological training folds and seven-day lag construction. The experiment does not look at validation year 2024, and it does not inspect, score, transform, or summarize any locked-test row from 2025-01-01 onward. The modern high-frequency archive is valuable, but for this specific feature family the available pre-validation history is still shorter than the user's hard four-year OOF requirement. That is why R05 is preserved as evidence and not treated as an accepted model improvement.

## Leakage Controls Applied

Every row is interpreted as a T-24 forecast origin at the day-before 15:00 local cutoff. R05 inherits the R04 rule that the latest ordinary station observation can be no later than 14:40 when a +20 minute replay latency is assumed. Lag 1 means the previous origin's cutoff-safe state, not the target day's observed maximum. The script calls the locked-date guard on the input feature matrix and the prediction table, then writes explicit `validation_2024_accessed: false` and `locked_test_accessed: false` metadata into the experiment directory. Preprocessing is also leakage-controlled: median imputation and standardization are fit inside the training slice of each chronological fold rather than on the full dataset. This prevents future rows from influencing scale, missing-value defaults, or model coefficients.

## What Was Actually Tested

The direct baseline asks whether the most recent eligible cutoff temperature plus deterministic seasonality is enough. The short-memory candidate asks whether the previous three cutoff states help distinguish persistent warm regimes from one-day noise. The long-memory candidate asks whether a full week of cutoff states improves or dilutes the signal. The EWMA and transition-gated candidate asks whether decayed memory and large-change flags are better than raw lags. The memory-decay diagnostic is separate from the fitted model ladder: it measures how each lag correlates with the target and how bad a naive persistence forecast would be at each lag. This helps identify whether a future model should use memory as a smooth expert, a gated regime feature, or not at all.

## How It Went

The experiment ran cleanly after the direct-execution import path was corrected so that the CLI script and tests can both load the shared R04 helper functions. The generated feature matrix contains only pre-2024 target dates and contains no locked-test rows. The OOF predictions also contain only 2021-07-01 through 2023-12-31, with no validation or locked-test dates. The best OOF diagnostic model was the simple lag1 cutoff-temperature baseline rather than a richer memory candidate. That means the additional memory features did not yet prove reliable incremental value in the available modern high-frequency window. This is a useful negative result: it directs later experiments toward other signal families such as moisture, wind, pressure, station-network gradients, upper-air profiles, NWP guidance, and forecast vintage deltas instead of overfitting stale persistence.

## Why This Is Still Useful

R05 creates a reusable, audited memory-feature construction path. Later experiments can join these features with other strictly as-of predictors and test interaction terms, but they must keep the same lag semantics and fold-local preprocessing. The result also documents an important governance decision: official daily target-history features are not automatically safe just because they refer to past dates. Availability time matters. A daily value for yesterday could still be unknown at today's 15:00 origin depending on publication behavior, so it remains excluded until a separate publication-latency audit proves it available. This prevents the system from quietly learning from future-published labels.

## Decision Record

R05 is accepted as a completed diagnostic experiment and rejected as a promotable feature-family improvement. The rejection is not because the code failed; it is because the strict OOF acceptance gate failed and the best simple memory baseline did not establish robust improvement. The correct next move is not to tune R05 harder against these same folds. The correct next move is to run the next predeclared signal-family experiment, preserve the same leakage controls, and update the research ledger so the accumulated evidence remains auditable.

## Downstream Rule

R05 does not authorize validation access or model promotion. Any useful memory signal is recorded as `OOF_BLOCKED_DIAGNOSTIC` until the modern high-frequency sample reaches at least four pre-validation-equivalent OOF years or the evaluation design is explicitly revised without touching validation 2024 or the locked test. Later experiments may reuse memory features only if the lag construction remains strictly backward-looking and fold-local preprocessing is preserved.

# R05 Machine-Readable Summary Tables

Generated: `2026-06-20T09:22:53.790888Z`

## Overall Scoreboard

| model_id | n | first_date | last_date | mae | rmse | median_abs_error | bias | crps_normal | coverage_80 | coverage_90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r05_baseline_lag1_cutoff_temp_calendar | 911 | 2021-07-01 | 2023-12-31 | 1.537332 | 1.964167 | 1.251854 | 0.200443 | 1.096197 | 0.817783 | 0.906696 |
| r05_memory_lags_1_3 | 911 | 2021-07-01 | 2023-12-31 | 1.541217 | 1.967641 | 1.261515 | 0.210068 | 1.098171 | 0.807903 | 0.896817 |
| r05_memory_lags_1_7 | 911 | 2021-07-01 | 2023-12-31 | 1.546601 | 1.976794 | 1.284134 | 0.068809 | 1.106282 | 0.785950 | 0.881449 |
| r05_ewma_gated_memory | 911 | 2021-07-01 | 2023-12-31 | 1.555375 | 1.989094 | 1.262773 | 0.190096 | 1.111274 | 0.791438 | 0.900110 |

## Fold Deltas

| fold_id | model_id | n | mae | baseline_mae | mae_improvement_vs_baseline | crps_improvement_vs_baseline |
| --- | --- | --- | --- | --- | --- | --- |
| fold_2022_h2 | r05_ewma_gated_memory | 182 | 1.283905 | 1.289665 | 0.005760 | 0.001587 |
| fold_2022_h2 | r05_baseline_lag1_cutoff_temp_calendar | 182 | 1.289665 | 1.289665 | 0.000000 | 0.000000 |
| fold_2022_h2 | r05_memory_lags_1_3 | 182 | 1.299226 | 1.289665 | -0.009561 | -0.002803 |
| fold_2023_h2 | r05_memory_lags_1_7 | 184 | 1.313629 | 1.341230 | 0.027601 | 0.010917 |
| fold_2023_h2 | r05_ewma_gated_memory | 184 | 1.330042 | 1.341230 | 0.011187 | 0.004302 |
| fold_2022_h2 | r05_memory_lags_1_7 | 182 | 1.330459 | 1.289665 | -0.040794 | -0.010970 |
| fold_2023_h2 | r05_memory_lags_1_3 | 184 | 1.340285 | 1.341230 | 0.000945 | 0.005058 |
| fold_2023_h2 | r05_baseline_lag1_cutoff_temp_calendar | 184 | 1.341230 | 1.341230 | 0.000000 | 0.000000 |
| fold_2023_h1 | r05_ewma_gated_memory | 181 | 1.484255 | 1.501090 | 0.016835 | 0.006715 |
| fold_2023_h1 | r05_memory_lags_1_3 | 181 | 1.496471 | 1.501090 | 0.004619 | 0.003068 |
| fold_2023_h1 | r05_baseline_lag1_cutoff_temp_calendar | 181 | 1.501090 | 1.501090 | 0.000000 | 0.000000 |
| fold_2023_h1 | r05_memory_lags_1_7 | 181 | 1.506805 | 1.501090 | -0.005715 | -0.010447 |
| fold_2021_h2 | r05_memory_lags_1_3 | 183 | 1.734688 | 1.760136 | 0.025447 | 0.011764 |
| fold_2021_h2 | r05_memory_lags_1_7 | 183 | 1.757136 | 1.760136 | 0.003000 | -0.020707 |
| fold_2021_h2 | r05_baseline_lag1_cutoff_temp_calendar | 183 | 1.760136 | 1.760136 | 0.000000 | 0.000000 |
| fold_2021_h2 | r05_ewma_gated_memory | 183 | 1.787874 | 1.760136 | -0.027738 | -0.032254 |
| fold_2022_h1 | r05_baseline_lag1_cutoff_temp_calendar | 181 | 1.796695 | 1.796695 | 0.000000 | 0.000000 |
| fold_2022_h1 | r05_memory_lags_1_7 | 181 | 1.827706 | 1.796695 | -0.031011 | -0.019449 |
| fold_2022_h1 | r05_memory_lags_1_3 | 181 | 1.837945 | 1.796695 | -0.041250 | -0.027225 |
| fold_2022_h1 | r05_ewma_gated_memory | 181 | 1.893465 | 1.796695 | -0.096770 | -0.055961 |

## Memory Decay

| lag_days | n | pearson_corr_with_target | mae_if_direct_persistence |
| --- | --- | --- | --- |
| 1 | 1269 | 0.912603 | 1.991411 |
| 2 | 1269 | 0.845742 | 2.495902 |
| 3 | 1269 | 0.809556 | 2.724035 |
| 4 | 1269 | 0.790389 | 2.872498 |
| 5 | 1269 | 0.773996 | 2.949567 |
| 6 | 1269 | 0.761148 | 2.993775 |
| 7 | 1269 | 0.753387 | 3.008589 |
