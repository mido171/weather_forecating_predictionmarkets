# GPT-Pro Request: All-Model KLGA Tmax Half-Life Bias-Correction Weighting Strategy Suite

You are GPT-Pro. I need you to design a serious, leakage-safe strategy suite for forecasting same-day KLGA daily maximum temperature, using all GribStream model data currently available to us.

This is not a generic forecasting request. The goal is to squeeze the maximum possible information gain from all model groups we have, while staying strictly non-forward-looking.

## 1. Core Objective

Forecast settled KLGA Tmax for target date `T`.

Primary forecast setup:

- Station: `KLGA`
- Target: daily settled Tmax in Fahrenheit
- Truth source: `public.wunderground_daily_tmax`, `station_id = 'KLGA'`
- Forecast cutoff: `T_1245UTC`
- Primary source buffer currently used in tests: `1.5 hours`
- All source rows must satisfy:
  - `source_latest_run_time_utc <= cutoff_utc - 1.5 hours`
  - `max_source_available_at_utc <= cutoff_utc`
- WU labels may be used for training/residual/bias history only through `T-2` unless explicit label-availability timestamps prove `T-1` was settled before cutoff. Default assumption: labels only through `T-2`.
- No target-day WU label may enter any feature, correction, model weight, calibration, or residual calculation.

The final output we want from the eventual implemented system is:

```text
final_forecast_tmax_f_T
```

plus diagnostics showing:

- raw model scalar forecasts
- per-model bias corrections
- per-model corrected forecasts
- model weights or residual coefficients
- adjustment terms
- availability mask
- training window used
- leakage checks
- forecast error after settlement

## 2. Data Available

The GribStream feature table is:

```text
gold.feature_values
```

Filtered by:

```text
feature_build_version = 'TMAX_THIN_V1'
cutoff_id = 'T_1245UTC'
target_station_id = 'KLGA'
```

The target/instance table is:

```text
gold.target_instances
```

The settled label table is:

```text
public.wunderground_daily_tmax
```

WU KLGA label coverage:

```text
accepted/manual KLGA labels: 19,419 rows
date range: 1973-01-01 through 2026-07-01
```

WU all-station truth table coverage:

```text
accepted/manual rows: 327,633
stations: 19
date range: 1973-01-01 through 2026-07-01
```

## 3. GribStream Model Coverage Available

This table was queried live from Postgres on 2026-07-02.

| Model | Feature Rows | Feature Names | Target Days | Target Date Range | WU-Overlap Days | WU-Overlap Range | Notes |
|---|---:|---:|---:|---|---:|---|---|
| `hrrr` | 334,719 | 77 | 4,347 | 2014-07-31 to 2026-06-28 | 4,335 | 2014-07-31 to 2026-06-28 | Longest GribStream history; direct Tmax proxy fields available |
| `rtma` | 9,291 | 3 | 3,097 | 2018-01-01 to 2026-06-28 | 3,089 | 2018-01-01 to 2026-06-28 | Current-state features only: temp, dewpoint, wind |
| `gefsatmos` | 406,818 | 194 | 2,097 | 2020-10-01 to 2026-06-28 | 2,089 | 2020-10-01 to 2026-06-28 | Ensemble member/threshold/temp features |
| `gefsatmosmean` | 4,194 | 2 | 2,097 | 2020-10-01 to 2026-06-28 | 2,089 | 2020-10-01 to 2026-06-28 | Mean model valid-time temp features |
| `nbm` | 161,161 | 77 | 2,093 | 2020-09-30 to 2026-06-28 | 2,085 | 2020-09-30 to 2026-06-28 | Direct Tmax proxy fields available; currently strongest model |
| `rap` | 150,381 | 77 | 1,953 | 2021-02-22 to 2026-06-28 | 1,947 | 2021-02-22 to 2026-06-28 | Direct Tmax proxy fields available |
| `gfs` | 148,148 | 77 | 1,924 | 2021-03-23 to 2026-06-28 | 1,918 | 2021-03-23 to 2026-06-28 | Direct Tmax proxy fields available |
| `ifsoper` | 1,702 | 2 | 851 | 2024-02-29 to 2026-06-28 | 848 | 2024-02-29 to 2026-06-28 | Valid-time temp features |
| `ifsenfo` | 198,806 | 234 | 850 | 2024-03-01 to 2026-06-28 | 847 | 2024-03-01 to 2026-06-28 | Ensemble member/threshold/temp features |
| `aifsoper` | 976 | 2 | 488 | 2025-02-26 to 2026-06-28 | 486 | 2025-02-26 to 2026-06-28 | Shorter-history valid-time temp features |
| `aifsenfo` | 84,558 | 234 | 362 | 2025-07-02 to 2026-06-28 | 361 | 2025-07-02 to 2026-06-28 | Shorter-history ensemble features |
| `nbmqmd` | 13,298 | 90 | 149 | 2026-01-31 to 2026-06-28 | 148 | 2026-01-31 to 2026-06-28 | Probabilistic Tmax distribution/bucket features; very short history |
| `aigefssfc` | 14,550 | 194 | 75 | 2025-06-27 to 2026-06-28 | 75 | 2025-06-27 to 2026-06-28 | Very short-history AI-GEFS-style ensemble features |
| `aigfssfc` | 140 | 2 | 70 | 2026-04-16 to 2026-06-28 | 70 | 2026-04-16 to 2026-06-28 | Very short-history AI-GFS-style valid-time temp features |

## 4. Feature Types Available

Direct Tmax proxy models:

```text
hrrr
nbm
rap
gfs
```

Each has 77 features and includes fields such as:

```text
grib_{model}_klga_core_member_0_tmax_proxy_f
grib_{model}_klga_core_tmax_proxy_mean_f
grib_{model}_klga_core_tmax_proxy_median_f
grib_{model}_klga_core_tmax_proxy_p05_f
grib_{model}_klga_core_tmax_proxy_p10_f
grib_{model}_klga_core_tmax_proxy_p25_f
grib_{model}_klga_core_tmax_proxy_p75_f
grib_{model}_klga_core_tmax_proxy_p90_f
grib_{model}_klga_core_tmax_proxy_p95_f
grib_{model}_klga_core_tmax_proxy_std_f
```

Current-state model:

```text
rtma
```

Available fields:

```text
grib_rtma_klga_core_current_tmp_2m_f
grib_rtma_klga_core_current_dewpoint_2m_f
grib_rtma_klga_core_current_wind_speed_10m_mph
```

Valid-time deterministic or mean temperature models:

```text
gefsatmosmean
ifsoper
aifsoper
aigfssfc
```

These generally include:

```text
grib_{model}_klga_core_valid_18z_tmp_2m_f
grib_{model}_klga_core_valid_00z_nextday_tmp_2m_f
```

Ensemble models:

```text
gefsatmos
ifsenfo
aifsenfo
aigefssfc
```

These include per-member valid-time 2m temperature features and probability/threshold-style features, for example:

```text
grib_{model}_klga_core_valid_18z_member_{n}_tmp_2m_f
grib_{model}_klga_core_valid_00z_nextday_member_{n}_tmp_2m_f
grib_{model}_klga_core_valid_18z_prob_tmp_2m_ge_{threshold}f
grib_{model}_klga_core_valid_00z_nextday_prob_tmp_2m_ge_{threshold}f
```

NBMQMD probabilistic model:

```text
nbmqmd
```

Includes:

```text
grib_nbmqmd_klga_core_mean_proxy_f
grib_nbmqmd_klga_core_prob_tmax_ge_{threshold}f
grib_nbmqmd_klga_core_generic_bucket_prob_60_64
grib_nbmqmd_klga_core_generic_bucket_prob_65_69
...
```

## 5. Current Initial Results To Beat

The current strict test used:

- cutoff `T_1245UTC`
- source buffer `1.5h`
- WU labels only through `T-2`
- bias correction:
  - 45-day lookback
  - 15-day half-life
  - minimum 10 prior labeled days
- common core direct models:
  - HRRR
  - NBM
  - RAP
  - GFS

Strict T-2 baselines before stacker warmup:

| Method | N | Date Range | MAE | RMSE | Bias |
|---|---:|---|---:|---:|---:|
| corrected NBM | 1,907 | 2021-04-03 to 2026-06-28 | 1.7527 | 2.4634 | -0.0071 |
| corrected equal-weight core | 1,907 | 2021-04-03 to 2026-06-28 | 1.8209 | 2.5992 | 0.0159 |
| raw equal-weight core | 1,907 | 2021-04-03 to 2026-06-28 | 2.0299 | 2.8731 | -0.9006 |
| corrected GFS | 1,907 | 2021-04-03 to 2026-06-28 | 2.1202 | 2.9610 | 0.0188 |
| raw NBM | 1,907 | 2021-04-03 to 2026-06-28 | 2.1937 | 2.9823 | -1.6442 |
| corrected HRRR | 1,907 | 2021-04-03 to 2026-06-28 | 2.2720 | 3.1509 | 0.0286 |
| corrected RAP | 1,907 | 2021-04-03 to 2026-06-28 | 2.3844 | 3.3064 | 0.0232 |

Same-date comparison on the residual-stacker-eligible window:

| Method | N | Date Range | MAE | RMSE | Bias |
|---|---:|---|---:|---:|---:|
| rolling convex blend, NBM >= 70% | 1,541 | 2022-04-05 to 2026-06-28 | 1.6780 | 2.3692 | -0.0066 |
| rolling convex blend, NBM >= 60% | 1,541 | 2022-04-05 to 2026-06-28 | 1.6800 | 2.3738 | -0.0095 |
| NBM residual stacker, no final bias | 1,541 | 2022-04-05 to 2026-06-28 | 1.6800 | 2.3800 | -0.1409 |
| online performance blend, best tested | 1,541 | 2022-04-05 to 2026-06-28 | 1.6834 | 2.3764 | -0.0029 |
| corrected NBM alone | 1,541 | 2022-04-05 to 2026-06-28 | 1.7157 | 2.3931 | -0.0099 |
| NBM residual stacker + final bias | 1,541 | 2022-04-05 to 2026-06-28 | 1.7278 | 2.4236 | -0.0096 |
| corrected equal-weight core | 1,541 | 2022-04-05 to 2026-06-28 | 1.7882 | 2.5304 | 0.0136 |
| raw equal-weight core | 1,541 | 2022-04-05 to 2026-06-28 | 1.9467 | 2.7326 | -0.7238 |
| raw NBM | 1,541 | 2022-04-05 to 2026-06-28 | 2.0492 | 2.7923 | -1.4442 |

Important interpretation:

- Equal-weighting is bad.
- Corrected NBM is strong.
- Other models contain incremental information, but they need constrained weighting.
- The best current result is `MAE = 1.6780F` from a rolling convex blend with `NBM >= 70%`.
- We have not yet fully used GEFS, GEFS mean, IFS, IFS ensemble, AIFS, AI ensemble models, NBMQMD, or RTMA in a comprehensive all-model strategy.

## 6. What I Need From You

Design a suite of strategies to test that uses all available model groups to their full potential.

The strategy suite must be based around:

- half-life weighted bias correction
- testing multiple lookback windows
- testing multiple half-lives
- availability-aware model inclusion
- final model weighting or residual weighting
- leakage-safe walk-forward evaluation
- final forecast value construction

Do not give vague advice. Give exact algorithms, hyperparameter grids, constraints, evaluation windows, acceptance gates, and expected diagnostic outputs.

## 7. Required Leakage Rules

Every strategy must obey:

1. Source eligibility:

```text
source_latest_run_time_utc <= cutoff_utc - source_buffer
max_source_available_at_utc <= cutoff_utc
```

2. Label eligibility:

```text
labels usable for target T must have label target_date <= T-2
```

3. Corrected historical forecasts used for training meta-weights must be out-of-sample:

For each historical date `s`, its corrected forecast must have been computed using only labels available before `s`, not labels available later.

4. No strategy may recompute old corrected forecasts using labels through the current prediction date `T` and then train on those artificially improved historical forecasts.

5. All evaluation comparisons must use the same eligible target dates for the methods being compared.

6. Short-history models must not be allowed to create fake performance through different test windows. Evaluate them both:

- on their own availability window
- on same-date windows against baselines

## 8. Strategy Design Questions You Must Answer

Please answer all of the following.

### 8.1 Raw Scalar Extraction

For every model group, define exactly how to create a raw scalar Tmax estimate.

For example:

- direct Tmax proxy models: should we use mean, median, member_0, percentile blend, or learned scalar?
- valid-time temp models: should scalar be `max(valid_18z_tmp, valid_00z_nextday_tmp)`, a weighted combination, or learned mapping?
- ensemble models: should scalar be member mean of max valid temps, median, quantile-weighted, probability-implied expected Tmax, or a distribution-to-scalar mapping?
- NBMQMD: should scalar be `mean_proxy_f`, probability-threshold reconstructed expectation, bucket midpoint expectation, or a calibrated blend?
- RTMA: should it be a correction feature only, not a raw Tmax model?

Give exact formulas and fallback behavior.

### 8.2 Per-Model Bias Correction

Design per-model online bias correction using half-life and lookback.

We need to test grids like:

```text
lookback_days in {15, 30, 45, 60, 90, 120, 180, 365}
half_life_days in {7, 10, 15, 21, 30, 45, 60, 90, 180}
min_history_days by model/history bucket
bias_cap_f in {none, 1.5, 2.5, 4.0}
label_lag_days = 2
```

Tell us:

- whether each model should get one global bias correction
- whether bias correction should be monthly/seasonal
- whether bias correction should be regime-specific
- whether direct Tmax proxy, ensemble scalar, and valid-time scalar should have different bias settings
- how to avoid overfitting short-history models

### 8.3 Model Weighting

Design final weighting methods that produce the final forecast value.

We want strategies like:

1. rolling convex blend
2. NBM-anchored residual stacker
3. availability-aware residual stacker with all models
4. model-family hierarchical blend
5. online performance-weighted blend
6. regime-gated blend
7. dynamic inclusion for short-history models
8. RTMA same-day correction layer
9. NBMQMD probability-informed correction layer

For each method, give exact:

- model inputs
- training window
- half-life
- objective loss
- regularization
- constraints
- minimum history requirements
- model caps
- adjustment caps
- fallback behavior
- final forecast formula

### 8.4 How To Use All Models Without Overfitting

I specifically want to squeeze useful signal from all available models, including shorter-history ones, but not let them overfit.

Please propose exact safe handling for:

- `gefsatmos`
- `gefsatmosmean`
- `ifsoper`
- `ifsenfo`
- `aifsoper`
- `aifsenfo`
- `nbmqmd`
- `aigefssfc`
- `aigfssfc`
- `rtma`

For each, say:

- whether it should be a full weighted member, a residual signal, an uncertainty/spread feature, a regime gate, or a diagnostic-only feature
- maximum allowed weight or adjustment
- minimum required training days
- how to evaluate it honestly
- what improvement would count as real

### 8.5 Availability-Aware Training

We do not want to force every model onto the common intersection of all models because that throws away years of data.

Design an availability-aware framework that:

- uses the long-history models over long windows
- uses shorter-history models only when available
- compares same-date baselines honestly
- allows model additions to prove incremental value
- does not train weights on rows where a model is missing unless the method explicitly supports missingness

### 8.6 Hyperparameter Selection

Design a leakage-safe hyperparameter selection process.

Required:

- no random train/test splits
- annual or rolling walk-forward only
- no selecting hyperparameters on the final evaluation window without nested validation
- objective should include MAE and RMSE

Suggested score:

```text
score = MAE + 0.20 * RMSE
```

But you may propose alternatives.

### 8.7 Evaluation Tables

Tell us exactly which result tables to produce.

Minimum required:

- overall MAE
- RMSE
- bias
- median absolute error
- within 1F
- within 2F
- count
- date range
- metrics by month/season
- metrics by actual Tmax bucket
- metrics for actual >= 90F
- metrics for actual >= 95F
- metrics for actual <= 40F
- metrics by high/low model spread
- metrics by source availability pattern
- metrics by each model-inclusion ablation

### 8.8 Diagnostics To Prove Information Gain

We need to know whether each model actually adds information.

Give exact diagnostics:

- incremental MAE/RMSE improvement from adding each model family
- permutation/drop-one-model importance in walk-forward form
- coefficient/weight distributions
- effective NBM weight distribution
- adjustment size distributions
- days where a model helped most
- days where a model hurt most
- high-spread vs low-spread value
- warm/cold/extreme-day value
- correlation matrix of model residuals
- rolling performance stability

### 8.9 Final Recommended Experiment Order

Give a ranked experiment plan.

I want a sequence such as:

```text
Experiment 0: freeze data/evaluation harness
Experiment 1: scalar extraction tests per model family
Experiment 2: per-model half-life bias correction grid
Experiment 3: corrected core blend
Experiment 4: add GEFS family
Experiment 5: add RTMA correction
Experiment 6: add IFS family
Experiment 7: add NBMQMD
Experiment 8: add AI short-history families
Experiment 9: regime-gated all-model blend
Experiment 10: final selected strategy with ablations
```

But make it better and more exact.

### 8.10 Acceptance Gates

Define what counts as success.

Current best result to beat:

```text
MAE = 1.6780F
RMSE = 2.3692F
Bias = -0.0066F
Window = 2022-04-05 to 2026-06-28
Method = rolling convex blend with NBM >= 70%
```

The new suite should try to improve:

- MAE
- RMSE
- extreme-day errors
- high-spread-day errors
- warm-season errors

But it must not introduce leakage or unstable overfit weights.

## 9. Output Format Required From You

Please return a complete implementation-ready strategy document with:

1. Executive summary.
2. Exact strategy suite.
3. Exact scalar construction per model group.
4. Exact bias-correction formulas and grids.
5. Exact weighting formulas and grids.
6. Exact handling of short-history models.
7. Exact leakage-safe walk-forward evaluation design.
8. Exact comparison tables to produce.
9. Exact diagnostics to prove whether each model adds information.
10. Ranked experiment order.
11. Recommended first strategy to implement.
12. Pseudocode precise enough for Codex to implement directly.
13. Warnings about likely leakage traps.

Do not answer with generic machine learning advice.

The target is a practical, robust, leakage-safe strategy suite for extracting maximum possible information from all available GribStream model families and producing one final KLGA Tmax forecast value.
