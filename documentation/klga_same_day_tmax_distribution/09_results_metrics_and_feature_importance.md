# 09 - Results, Metrics, and Feature Importance (LGBM vs TabM)

This document is the "results book" for the KLGA same-day Tmax distribution system.

It records:

- the exact headline metrics for the reference LightGBM system,
- the exact headline metrics for the first TabM (tabular neural network) experiment trained from exported CSVs,
- the detailed time-of-day behavior (cutoff metrics),
- full feature importance lists (no omissions) for the LightGBM peak and delta models.

If you want the system intent/spec, read:

- `00_high_level_overview.md`
- `01_system_spec_and_implementation.md`

If you want to interpret metrics as a beginner, read:

- `02_metrics_and_interpretation_for_beginners.md`

## 1) The Two Runs This Document Covers

### 1.1 Reference LGBM (no analog, fully exported artifacts)

Run id:

- `20260226T081223Z`

Path:

- `artifacts/same_day_res_poly/20260226T081223Z/`

Mode:

- analog disabled (`--skip-analog-blend`)

Key properties (from `run.log`):

- feature store reused (not rebuilt)
- rows loaded: `359,194`
- model feature count: `496`
- split ranges:
  - train: `1992-01-01 .. 2021-12-31`
  - val: `2022-01-01 .. 2023-12-31`
  - test: `2024-01-01 .. 2025-12-31`

### 1.2 TabM (tabular neural network) trained from exported CSVs

Run id:

- `20260226T224250Z`

Where it ran:

- external machine / external data folder (portable export bundle)

Key properties (from the produced `metrics.md`):

- rows_total: `560,425` (1973..2025 daily horizon with 29 cutoffs/day)
- split_rows_peak: train `517,253`, val `21,158`, test `21,199`
- split_rows_delta: train `404,183`, val `18,234`, test `18,552`

Important nuance:

- the TabM run used a longer history (train start 1973) than the LGBM reference run (train start 1992).
- so this is not a perfectly identical data window comparison, but it is the same feature/label contract and the same val/test windows.

## 2) LGBM Reference Run Metrics (20260226T081223Z)

Sources:

- `artifacts/same_day_res_poly/20260226T081223Z/metrics.json`
- `artifacts/same_day_res_poly/20260226T081223Z/reports/*.csv`
- `artifacts/same_day_res_poly/20260226T081223Z/run.log`

### 2.1 Pipeline stage runtimes (so you know what's slow)

From `run.log`:

- load feature store: `00:01` (rows=359194, cols=500)
- prepare splits/features: `00:07` (feature_count=496)
- train peak model: `02:55`
- train delta model: `56:09` (dominant cost)
- predict delta: `04:15`
- evaluate metrics: `00:50`
- write artifacts: `00:04`
- total pipeline: `01:04:34`

Practical implication:

- "stuck at delta training" for ~1 hour is normal for the full dataset.

### 2.2 Peak model (binary) metrics

Peak predicts:

- `P(delta=0)` = probability the day has already peaked at the cutoff.

Metrics (from `metrics.json`, calibrated is what matters):

- validation:
  - logloss raw: `0.23347`
  - logloss calibrated: `0.218998`
  - Brier raw: `0.071579`
  - Brier calibrated: `0.067748`
- test:
  - logloss raw: `0.223435`
  - logloss calibrated: `0.209856`
  - Brier raw: `0.070227`
  - Brier calibrated: `0.065247`

Interpretation:

- these are strong scores for a binary "already peaked" classifier.
- calibration materially improves probability quality (raw -> calibrated).

### 2.3 Delta model (multiclass, conditional) metrics

Delta predicts:

- `P(delta=k | delta>0)` for `k=1..60` (60 is tail bin for `>=60`).

Recorded metrics (from `metrics.json`, validation only):

- val multi_logloss raw: `2.40935`
- val multi_logloss temp-scaled: `2.40755`
- temperature scaler `T`: `1.063616`

Additional test metric (computed from the stored predictions artifacts):

- test multi_logloss (delta>0 rows only): `2.43260` (n_rows=18,552)

Interpretation:

- delta is the harder half of the problem.
- temperature scaling helps slightly (on validation).

### 2.4 Combined distribution metrics (what matters for trading)

Combined distribution is the final PMF after peak+delta composition.

From `metrics.json` (analog disabled so blended == pure LGBM):

- validation:
  - NLL: `2.29395`
  - top1 accuracy: `0.23291`
- test:
  - NLL: `2.33871`
  - top1 accuracy: `0.22992`

Interpretation:

- combined NLL is the best single-number offline proxy for "live pricing quality".
- top1 is useful, but NLL is more important for probability trading.

### 2.5 Time-of-day behavior (combined distribution by cutoff)

Source:

- `artifacts/same_day_res_poly/20260226T081223Z/reports/cutoff_metrics_test.csv`
- `artifacts/same_day_res_poly/20260226T081223Z/reports/cutoff_metrics_val.csv`

Test split (29 cutoffs, NY local minutes since midnight):

| cutoff_minutes | NY time | test NLL | test top1 |
|---:|---:|---:|---:|
| 240 | 04:00 | 2.8052 | 0.1040 |
| 480 | 08:00 | 2.7390 | 0.1354 |
| 720 | 12:00 | 2.3348 | 0.2052 |
| 900 | 15:00 | 1.8251 | 0.3570 |
| 1080 | 18:00 | 1.5636 | 0.4815 |

Full per-cutoff table:

- see `reports/cutoff_metrics_test.csv` (29 rows) in the run folder

Interpretation:

- later in the day the distribution gets much better (lower NLL, higher top1)
- this is expected because uncertainty collapses as the day is observed

### 2.6 Time-block summaries (combined + delta-only)

These blocks are NY local time, aligned to the cutoff grid:

- 04:00–08:00 (cutoffs 240..480)
- 08:30–12:00 (510..720)
- 12:30–15:00 (750..900)
- 15:30–18:00 (930..1080)

Combined distribution (test):

- 04:00–08:00: NLL `2.7732`, top1 `0.1260`
- 08:30–12:00: NLL `2.5544`, top1 `0.1580`
- 12:30–15:00: NLL `2.0968`, top1 `0.2704`
- 15:30–18:00: NLL `1.6412`, top1 `0.4412`

Delta-only multiclass logloss (test, only rows where truth delta>=1):

- 04:00–08:00: `2.7536`
- 08:30–12:00: `2.5374`
- 12:30–15:00: `2.0911`
- 15:30–18:00: `1.8981`

Interpretation:

- delta gets dramatically easier later in the day (very strong pattern).

### 2.7 Bucket calibration summary (test)

Source:

- `artifacts/same_day_res_poly/20260226T081223Z/reports/bucket_calibration_test.csv`

Summary across integer temperature buckets (this is a calibration sanity check):

- bucket rows: `82`
- mean absolute gap: `0.00307`
- 90th percentile absolute gap: `0.00601`
- max absolute gap: `0.01268`

Interpretation:

- on average, predicted vs empirical frequency by integer temperature bucket is fairly close.
- do not over-interpret a single bucket with low counts; always inspect the `count` column.

## 3) TabM Run Metrics (20260226T224250Z)

Source:

- TabM run’s produced `metrics.md` (copied from the external run folder)

### 3.1 Peak (TabM)

- val logloss cal: `0.245539`
- test logloss cal: `0.244821`
- val brier cal: `0.078333`
- test brier cal: `0.078062`

### 3.2 Delta (TabM)

- val multi_logloss temp: `2.523057`
- test multi_logloss temp: `2.589655`
- temperature: `0.787097`

### 3.3 Combined (TabM)

- val NLL: `2.419914`
- test NLL: `2.510935`
- val top1: `0.236128`
- test top1: `0.231568`

## 4) LGBM vs TabM: Direct Comparison (What Actually Got Better/Worse)

Headline:

- On this system, TabM did **not** beat the LightGBM baseline on the metrics that matter for trading (logloss/Brier/NLL).

Concrete comparisons (test):

- peak logloss cal:
  - LGBM: `0.209856`
  - TabM: `0.244821` (worse)
- combined test NLL:
  - LGBM: `2.338706`
  - TabM: `2.510935` (worse)
- delta test multi_logloss:
  - LGBM: `2.43260`
  - TabM: `2.58966` (worse)

Interpretation:

- boosted trees remain the stronger and more reliable choice for this engineered tabular feature space.
- tabular NNs can be competitive, but they typically require careful tuning and sometimes different preprocessing choices.

## 5) Full Feature Importance (No Omissions)

This section provides a full list of feature importances for the reference LGBM run.

Important:

- feature importance is not causality
- correlated features can split importance
- "gain" is the most informative default for LightGBM

### 5.1 Where the full lists live

This file contains the full, merged feature importance export (no omissions):

- `documentation/klga_same_day_tmax_distribution/results/lgbm_20260226T081223Z_feature_importance_ALL.csv`

How to read:

- `importance_gain`: total loss reduction attributed to splits on that feature (most useful ranking)
- `importance_split`: how many times the feature was used in splits (less informative, but still helpful)

How it is structured (so you can filter it easily):

- it contains 4 blocks concatenated (1984 rows total = 4 * 496)
- columns include:
  - `model` in `{peak, delta}`
  - `sorted_by` in `{gain, split}`
  - `rank` (rank within that model/sort block)
  - `feature_index`, `feature`
  - `importance_gain`, `importance_split`
  - `source_file` (the original per-block filename)

Example filters:

- peak, gain ordering: `model == 'peak' AND sorted_by == 'gain'`
- delta, split ordering: `model == 'delta' AND sorted_by == 'split'`

### 5.2 Top features (quick view, gain-importance)

These are the top 20 features by `importance_gain` for the reference LGBM run.

Peak model (top 20 by gain):

| rank | feature | importance_gain |
|---:|---|---:|
| 1 | climo_rem_delta_mean | 1444623.44 |
| 2 | cutoff_minutes | 960398.73 |
| 3 | year | 650382.50 |
| 4 | temp_now_minus_tmax | 401357.34 |
| 5 | tmax_yday | 342087.78 |
| 6 | year_norm | 295016.61 |
| 7 | n_obs_sofar | 285524.98 |
| 8 | temp_range_sofar | 258681.09 |
| 9 | tmax_std_30d | 246283.04 |
| 10 | doy_cos | 208311.44 |
| 11 | time_of_tmax_sofar_min | 199631.75 |
| 12 | tmax_mean_7d | 188853.18 |
| 13 | tmax_mean_30d | 184356.90 |
| 14 | tmax_2day | 177756.92 |
| 15 | tmax_sofar | 170670.75 |
| 16 | doy_sin | 117164.15 |
| 17 | doy | 114718.08 |
| 18 | dew_pt_max_sofar | 114683.66 |
| 19 | pressure_min_sofar | 105451.89 |
| 20 | climo_rem_delta_std | 102475.51 |

Delta model (top 20 by gain):

| rank | feature | importance_gain |
|---:|---|---:|
| 1 | climo_rem_delta_std | 24025.48 |
| 2 | climo_rem_delta_mean | 13760.88 |
| 3 | tmax_std_30d | 11154.47 |
| 4 | tmax_yday | 6845.73 |
| 5 | doy_sin | 5878.09 |
| 6 | temp_range_sofar | 5832.68 |
| 7 | temp_diff_KBDR | 4880.13 |
| 8 | temp_diff_KTEB | 4705.10 |
| 9 | time_of_tmax_sofar_min | 4621.76 |
| 10 | tmax_2day | 4303.58 |
| 11 | tmax_mean_30d | 4123.80 |
| 12 | nbr_pressure_range | 3901.29 |
| 13 | KISP_vis_now | 3830.24 |
| 14 | tmax_sofar | 3545.72 |
| 15 | tmax_mean_7d | 3402.88 |
| 16 | doy_cos | 3385.03 |
| 17 | year | 3086.69 |
| 18 | dew_pt_range_sofar | 2999.91 |
| 19 | pressure_prev_360 | 2955.39 |
| 20 | pressure_min_sofar | 2935.83 |

### 5.2 Why there is no "native feature importance" for TabM here

Tree models expose split/gain importances directly.

TabM is a neural network:

- it does not naturally produce split/gain importance
- to estimate feature importance, you typically use:
  - permutation importance (slow but straightforward)
  - SHAP-style methods (more complex)

If you want TabM feature importance anyway, the recommended approach is:

- permutation importance on validation for:
  - peak: binary logloss
  - delta: multiclass logloss

That can be added as a separate utility if desired.
