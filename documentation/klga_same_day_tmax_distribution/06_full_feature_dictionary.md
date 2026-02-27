# 06 - Full Feature Dictionary (Implementation-Grade)

This file documents the feature space used by the KLGA same-day Tmax distribution pipeline.

Scope:

- feature names and naming patterns,
- formulas and units,
- missing-value behavior,
- leakage constraints,
- how features map to peak and delta training.

Reference run feature count:

- `496` model features (plus bookkeeping columns).

Reference feature list artifact:

- `artifacts/same_day_res_poly/<RUN_ID>/feature_list.json`

Important:

- this file describes the feature contract, not "the best set"
- if you change feature definitions, you must treat it as a new model contract and re-export artifacts

## 1) Feature naming conventions

General conventions:

- `_now` means latest available value at or before cutoff.
- `_prev_<W>` means value at or before `cutoff - W minutes`.
- `_delta_<W>` means `now - prev` over lookback `W`.
- `_slope_<W>` means `delta / W` (units per minute).
- `_std_<W>`, `_min_<W>`, `_max_<W>`, `_range_<W>` are window stats.
- `<STATION>_` prefix means per-neighbor snapshot feature.
- `_diff_<STATION>` means neighbor minus KLGA gradient feature.

Lookback window set:

- `W in {30, 60, 120, 180, 360}` minutes.

## 2) Calendar and cutoff identity features

Purpose:

- encode systematic seasonal and intraday behavior.

Features:

- `cutoff_minutes` = `hour*60 + minute` in NY local time.
- `cutoff_hour` in NY local.
- `cutoff_minute` in NY local (`0` or `30`).
- `cutoff_sin`, `cutoff_cos` = cyclic encoding of `cutoff_minutes`.
- `doy` = day-of-year (`1..366`).
- `doy_sin`, `doy_cos` = cyclic encoding of `doy`.
- `year`, `year_norm`.
- `is_weekend` (`1` Sat/Sun else `0`).
- `is_dst` (`1` daylight-saving offset period else `0`).

Leakage note:

- all are timestamp-derived and leakage-safe.

## 3) KLGA cutoff snapshot features

Purpose:

- capture current thermodynamic and wind state at cutoff.

Core fields from latest valid row (`valid_time_utc <= cutoff_utc`):

- `temp_now` (F)
- `dewpt_now` (F)
- `rh_now` (%)
- `pressure_now` (inHg)
- `vis_now` (miles)
- `wspd_now` (mph)
- `wdir_now` (degrees)
- `gust_now` (mph)
- `precip_hrly_now` (in/hr proxy)

Derived snapshot fields:

- `dewpoint_depression_now = temp_now - dewpt_now`
- `wdir_sin = sin(wdir_now*pi/180)`
- `wdir_cos = cos(wdir_now*pi/180)`
- `gust_factor = gust_now - wspd_now`

Snapshot missingness flags:

- `is_temp_missing_now`
- `is_dew_pt_missing_now`
- `is_rh_missing_now`
- `is_pressure_missing_now`
- `is_vis_missing_now`
- `is_wspd_missing_now`
- `is_wdir_missing_now`
- `is_gust_missing_now`
- `is_precip_hrly_missing_now`

Data recency feature:

- `age_min_temp` = minutes between cutoff and latest temp observation.

## 4) KLGA data coverage and quality features

Purpose:

- model reliability signal from observation completeness.

Features:

- `n_obs_sofar` = rows from midnight to cutoff.
- `n_obs_temp` = rows with non-null temp.
- `n_expected_bins` = expected 30-min slots from midnight to cutoff inclusive.
- `coverage_frac_temp = n_obs_temp / n_expected_bins`.

Interpretation:

- lower coverage often implies weaker confidence.

## 5) KLGA so-far extrema and regime features

Purpose:

- summarize how far the day has progressed toward its eventual max.

Temperature extrema:

- `tmax_sofar`
- `tmin_sofar`
- `temp_range_sofar = tmax_sofar - tmin_sofar`
- `temp_now_minus_tmax = temp_now - tmax_sofar` (typically <= 0)

Peak timing features:

- `time_of_tmax_sofar_min` = minutes after local midnight when latest so-far max occurred.
- `mins_since_tmax = cutoff_minutes - time_of_tmax_sofar_min`.

Other variable extrema:

- `dew_pt_max_sofar`, `dew_pt_min_sofar`, `dew_pt_range_sofar`
- `pressure_max_sofar`, `pressure_min_sofar`, `pressure_range_sofar`
- `wspd_max_sofar`, `wspd_min_sofar`, `wspd_range_sofar`
- `gust_max_sofar`, `gust_min_sofar`, `gust_range_sofar`

Precip regime flags:

- `any_precip_sofar` (`1` if any precip_hrly > 0)
- `precip_frac_sofar` = fraction of non-null precip rows with precip > 0.

## 6) KLGA trajectory window features

Purpose:

- quantify trend, volatility, and short-term dynamics.

For each variable in:

- `temp`, `dew_pt`, `rh`, `pressure`, `wspd`

For each `W in {30,60,120,180,360}`:

- `<var>_prev_<W>`
- `<var>_delta_<W>`
- `<var>_slope_<W>`
- `<var>_std_<W>`
- `<var>_min_<W>`
- `<var>_max_<W>`
- `<var>_range_<W>`
- `<var>_std_diff_<W>` (std of consecutive diffs in window)

Temperature curvature signatures:

- `temp_accel_60_180 = temp_slope_60 - temp_slope_180`
- `temp_accel_30_120 = temp_slope_30 - temp_slope_120`
- `temp_is_falling_60` (`1` if `temp_slope_60 < 0`)
- `temp_drop_from_peak = tmax_sofar - temp_now`

## 7) Neighbor snapshot features (per station)

Neighbor set:

- `KJFK`, `KEWR`, `KTEB`, `KHPN`, `KISP`, `KBDR`, `KMMU`

Each neighbor receives the same snapshot schema as KLGA with station prefix.

Pattern examples:

- `<ST>_temp_now`
- `<ST>_dewpt_now`
- `<ST>_pressure_now`
- `<ST>_wspd_now`
- `<ST>_wdir_now`
- `<ST>_gust_now`
- `<ST>_vis_now`
- `<ST>_precip_hrly_now`
- `<ST>_dewpoint_depression_now`
- `<ST>_wdir_sin`, `<ST>_wdir_cos`
- `<ST>_gust_factor`
- `<ST>_age_min_temp`
- `<ST>_is_*_missing_now` flags

Implementation note:

- `<ST>_cutoff_minutes` appears for consistency in station-snapshot construction.

## 8) Neighbor gradient features (neighbor minus KLGA)

Purpose:

- encode mesoscale spatial gradients that influence KLGA remaining warming.

Patterns by station:

- `temp_diff_<ST>`
- `dewpt_diff_<ST>`
- `pressure_diff_<ST>`
- `wspd_diff_<ST>`
- `dewpoint_depression_diff_<ST>`

Interpretation:

- coastal cap, inland heating lead, pressure gradient, and advection signals.

## 9) Neighbor aggregate and regime-composite features

All-neighbor aggregates:

- `nbr_temp_mean`, `nbr_temp_min`, `nbr_temp_max`, `nbr_temp_range`
- `nbr_dewpt_mean`, `nbr_dewpt_min`, `nbr_dewpt_max`, `nbr_dewpt_range`
- `nbr_pressure_mean`, `nbr_pressure_min`, `nbr_pressure_max`, `nbr_pressure_range`

Coastal vs inland composites:

- `temp_coastal_mean`
- `temp_inland_mean`
- `coastal_minus_inland_temp`
- `dewpt_coastal_minus_inland`
- `pressure_coastal_minus_inland`

Additional contrast helpers:

- `temp_inland_mean_minus_klga`
- `temp_jfk_minus_klga`
- `temp_north_mean`
- `temp_urban_fringe_mean`

## 10) Historical daily-max prior features (strictly date < D)

Purpose:

- provide low-frequency context for baseline temperature regime.

Features:

- `tmax_yday`
- `tmax_2day`
- `tmax_mean_7d`
- `tmax_mean_30d`
- `tmax_std_30d`

Leakage rule:

- these priors must never read date `D` truth.

## 11) Climatology remaining-delta prior features

Purpose:

- provide train-only empirical prior by `(doy, cutoff_minutes)`.

Features:

- `climo_rem_delta_mean`
- `climo_rem_delta_std`

Important:

- lookup computed only on training slice,
- then frozen and applied to val/test/live.

### 11.1 What climo_rem_delta_mean/std represent (intuitively)

These are "empirical priors" learned from your own training data:

- for each `(day_of_year, cutoff_minutes)` the system computes the historical distribution of remaining warming
- then it stores:
  - average remaining warming (`climo_rem_delta_mean`)
  - variability of remaining warming (`climo_rem_delta_std`)

Interpretation:

- high `climo_rem_delta_mean` early in the day means: historically there is still a lot of warming left
- low `climo_rem_delta_mean` late in the day means: historically there is not much left
- high `climo_rem_delta_std` means: remaining warming is historically more variable/uncertain for that time-of-year and time-of-day

Leakage safety:

- these values are computed only on the training slice and then frozen

## 12) Technical merge-index fields

Fields observed in current feature list:

- `index_x`
- `index_y`

These are merge bookkeeping artifacts from dataframe joins.

Practical note:

- they are present in current model feature list and therefore must be preserved for strict feature-order compatibility with exported models.

## 13) Label and metadata columns (not model features)

These columns exist in dataset rows but are excluded from feature matrix:

- `target_date_local`
- `cutoff_utc`
- `tmax_truth`
- `delta`
- `peak`
- split markers and other bookkeeping columns.

## 14) Missing-value handling contract

General strategy:

- keep explicit missing flags,
- fill numeric nulls with train-derived median/imputer values,
- keep feature order fixed via saved `feature_list.json`.

Artifacts:

- `imputer_values.json`
- `feature_list.json`

Inference must use the same fill map and feature order.

## 15) Banned observation inputs

Never allowed as features:

- `max_temp`
- `min_temp`
- `precip_total`

Reason:

- risk of summary-like leakage behavior.

## 16) Leakage-safety checks related to features

Feature builder enforces:

- no observation with `valid_time_utc > cutoff_utc`,
- no same-day truth usage in priors,
- no banned columns in selected feature set,
- no duplicate-source contamination from `KNYC`.

If these guards fail, pipeline should fail hard.

## 18) Feature importance (how to see what mattered)

The LightGBM models can report feature importance.

Important caveats:

- feature importance is not causality
- correlated features can split importance across them
- importance depends on the model family and training window

### 18.1 How to extract feature importance (example snippet)

For a completed run folder containing `models/peak_model.txt` and `models/delta_model.txt`, you can load the model and inspect importance by "gain".

Pseudo-example (conceptual):

1. Load the LightGBM booster.
2. Get importance array for your `feature_list.json`.
3. Sort and print top N.

If you want this automated, add a small utility script that:

- reads `feature_list.json`
- loads the model files
- writes `reports/feature_importance_peak.csv` and `reports/feature_importance_delta.csv`

### 18.2 What features are typically high-signal (non-exhaustive)

Historically, the highest-value families tend to include:

- climo remaining delta mean/std (strong priors by season + time-of-day)
- current state and proximity to so-far max:
  - `temp_now_minus_tmax`, `mins_since_tmax`, `temp_drop_from_peak`
- short-window trajectory:
  - `temp_slope_60`, `temp_slope_180`, `temp_is_falling_60`
- coastal/inland neighbor composites:
  - `coastal_minus_inland_temp`

The exact ordering should be checked in your run artifacts for the specific model you intend to use.

## 17) Fast audit commands

Inspect exact features used by a completed run:

```powershell
Get-Content artifacts/same_day_res_poly/<RUN_ID>/feature_list.json
```

Inspect imputer values:

```powershell
Get-Content artifacts/same_day_res_poly/<RUN_ID>/imputer_values.json
```

Confirm model feature count from run log (`prepare_splits_and_features` stage) and `feature_list.json` length match.
