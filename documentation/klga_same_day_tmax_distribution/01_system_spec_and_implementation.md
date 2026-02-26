# 01 - System Spec and Implementation

This is the implementation-grade reference for the KLGA same-day Tmax distribution system.

## 1) Objective and non-negotiable constraints

### Objective

At any local cutoff on day `D`, estimate:

- `P(Tmax_final(D)=t)` for integer `t` in Fahrenheit.

Then derive bucket probabilities for market labels.

### Non-negotiable constraints

1. Settlement alignment:
   - KLGA source semantics
   - whole-degree Fahrenheit outputs
2. As-of integrity:
   - no data with timestamp after cutoff
3. No label leakage:
   - date `D` daily max cannot be used as feature
4. Duplicate-source guard:
   - `KNYC:9:US` excluded

## 2) Canonical data contracts

## 2.1 Observation source table

Table:

- `wunderground_ml.wunderground_station_observation_30m`

Allowed columns:

- `request_location_id`
- `valid_time_utc`
- `temp`
- `dew_pt`
- `rh`
- `pressure`
- `vis`
- `wspd`
- `wdir`
- `gust`
- `precip_hrly`

Banned columns:

- `max_temp`
- `min_temp`
- `precip_total`

Enforced in:

- `ml/src/weather_ml/klga_daily_tmax_dist/db.py`
- `ml/src/weather_ml/klga_daily_tmax_dist/config.py`

## 2.2 Daily truth table

Table:

- `wunderground_ml.wunderground_station_daily_max_temperature`

Used fields:

- `target_date_local`
- `max_temp_f`
- `station_zoneid`

Target station:

- `KLGA:9:US`

## 2.3 Station universe

Target:

- `KLGA:9:US`

Neighbors:

- `KJFK:9:US`
- `KEWR:9:US`
- `KTEB:9:US`
- `KHPN:9:US`
- `KISP:9:US`
- `KBDR:9:US`
- `KMMU:9:US`

Hard exclusion:

- `KNYC:9:US`

## 3) Time and cutoff semantics

Timezone:

- `America/New_York`

Cutoff grid:

- every 30 minutes
- from `04:00` through `18:00`, inclusive
- 29 cutoffs/day

Implementation:

- `ml/src/weather_ml/klga_daily_tmax_dist/timegrid.py`

As-of window for same-day features:

- from local midnight of date `D` to cutoff `t_c`
- converted to UTC with timezone-aware conversion

## 4) Label definitions

For each `(D, t_c)`:

1. `tmax_sofar = max(temp up to t_c on date D)`
2. `tmax_truth = round(daily max for D)`
3. `delta = tmax_truth - round(tmax_sofar)`
4. clamp: if `delta < 0`, set `delta = 0`
5. `peak = 1 if delta <= 0 else 0`

Meaning:

- `peak=1` means max already reached by cutoff
- `peak=0` means still room to rise

## 5) Feature engineering contracts

Implementation:

- `ml/src/weather_ml/klga_daily_tmax_dist/features.py`

## 5.1 Calendar and cutoff identity

Examples:

- `cutoff_minutes`
- `cutoff_sin`, `cutoff_cos`
- `doy`, `doy_sin`, `doy_cos`
- `year`, `year_norm`
- `is_weekend`
- `is_dst`

## 5.2 Data availability and quality

Examples:

- `n_obs_sofar`
- `n_obs_temp`
- `coverage_frac_temp`
- `age_min_temp`
- missing-now flags per variable

## 5.3 KLGA snapshot features

Examples:

- `temp_now`, `dewpt_now`, `pressure_now`, `wspd_now`
- `wdir_sin`, `wdir_cos`
- `gust_factor`
- `dewpoint_depression_now`

## 5.4 KLGA so-far state

Examples:

- `tmax_sofar`, `tmin_sofar`
- `temp_range_sofar`
- `temp_now_minus_tmax`
- `time_of_tmax_sofar_min`
- `mins_since_tmax`
- precip indicators

## 5.5 KLGA trajectory windows

Windows:

- 30, 60, 120, 180, 360 minutes

For variables `temp`, `dew_pt`, `rh`, `pressure`, `wspd`:

- previous value at/ before cutoff-window
- delta and slope
- window std/min/max/range/std of diffs

Derived temperature dynamics:

- `temp_accel_60_180`
- `temp_accel_30_120`
- `temp_is_falling_60`
- `temp_drop_from_peak`

## 5.6 Neighbor features

Per-neighbor snapshot:

- same snapshot schema as KLGA

Gradients vs KLGA:

- temp, dew point, pressure, wind, dewpoint depression differences

Composites:

- global neighbor min/mean/max/range
- coastal vs inland contrasts

## 5.7 Historical priors from daily max

Strictly dates `< D`:

- `tmax_yday`
- `tmax_2day`
- `tmax_mean_7d`
- `tmax_mean_30d`
- `tmax_std_30d`

## 5.8 Climatology prior features

Train-only lookup by `(doy, cutoff_minutes)`:

- `climo_rem_delta_mean`
- `climo_rem_delta_std`

Applied to val/test/live with no future leakage.

## 6) Dataset build and persistence

Primary builder:

- `ml/src/weather_ml/klga_daily_tmax_dist/make_dataset.py`

Main steps:

1. Ensure required indexes.
2. Load daily truth dates in split range.
3. Build cutoff calendar grid.
4. Pull observation rows for station set and time window.
5. Build feature rows with as-of guard.
6. Persist parquet feature store + integrity JSON.

Feature store:

- `artifacts/same_day_res_poly/feature_store/klga_feature_store.parquet`

Integrity report:

- `artifacts/same_day_res_poly/feature_store/klga_feature_store_integrity.json`

## 7) Train/validation/test split contract

Date-based, strict, no overlap:

- train: `1992-01-01` to `2021-12-31`
- val: `2022-01-01` to `2023-12-31`
- test: `2024-01-01` to `2025-12-31`

Split code:

- `_split_masks` in `pipeline.py`

## 8) Model architecture

## 8.1 Peak model

Type:

- LightGBM binary classifier

Target:

- `peak`

Calibration:

- isotonic regression

Core API:

- `ml/src/weather_ml/klga_daily_tmax_dist/train_peak.py`

## 8.2 Delta model

Type:

- LightGBM multiclass classifier

Target rows:

- only where `delta>=1` and `peak=0`

Class mapping:

- class `0..59` corresponds to delta `1..60`
- class `59` acts as tail for `delta>=60`

Calibration:

- temperature scaling

Core API:

- `ml/src/weather_ml/klga_daily_tmax_dist/train_delta.py`

## 9) Optional analog kNN module

Implementation:

- `ml/src/weather_ml/klga_daily_tmax_dist/analog_knn.py`

Role:

- retrieval-based posterior generator
- blended with LGBM outputs when enabled

Current run-mode control:

- enabled by default
- disable with `--skip-analog-blend`

## 10) Posterior composition

Without analog:

1. `P(delta=0)` from calibrated peak model
2. `P(delta=k|delta>0)` from calibrated delta model
3. full delta PMF:
   - `P(0)=p_peak`
   - `P(k>0)=(1-p_peak)*p_delta_cond(k)`

With analog enabled:

- same as above, but peak and delta conditional probabilities can be blended with analog posteriors before PMF composition.

PMF utilities:

- `ml/src/weather_ml/klga_daily_tmax_dist/infer.py`

## 11) Metrics semantics

Peak metrics:

- binary logloss (raw and calibrated)
- Brier score (raw and calibrated)

Delta metric:

- multiclass logloss on delta-only slice (`delta>=1`)

Combined metric:

- NLL of final PMF over all evaluated rows

Critical distinction:

- delta multiclass logloss is not directly comparable to combined NLL.

## 12) Leakage and safety guards

Implemented guards:

1. as-of timestamp guard:
   - max used observation time cannot exceed cutoff
2. banned-column guard:
   - dangerous summary columns blocked
3. split overlap guard:
   - train/val/test overlap raises error
4. duplicate-source guard:
   - `KNYC` not allowed in station set
5. feature exclusion guard:
   - labels not included in model matrix

## 13) Pipeline stage flow

Entry point:

- `run_training_pipeline` in `pipeline.py`

Current stage plan:

When analog enabled:

1. build feature store
2. load feature store
3. prepare splits and features
4. train peak
5. predict peak
6. train delta
7. predict delta
8. build analog library
9. analog K selection
10. blend
11. evaluate metrics
12. write artifacts

When analog disabled:

1. build feature store
2. load feature store
3. prepare splits and features
4. train peak
5. predict peak
6. train delta
7. predict delta
8. evaluate metrics
9. write artifacts

## 14) Artifact export behavior

Run directory contains:

- `metrics.json`, `metrics.md`
- model files
- prediction parquet files
- report CSV files
- config and imputer/feature metadata
- run log

Important robustness improvement:

- peak and delta model files are checkpoint-saved immediately after training stage completion, so artifacts exist even if a later stage fails.

## 15) Current authoritative no-analog export run

Run id:

- `20260226T081223Z`

Path:

- `artifacts/same_day_res_poly/20260226T081223Z/`

Mode:

- `skip_analog_blend=true`

Result:

- full artifact package present including exported peak and delta models.

## 16) Design tradeoffs and known limits

Known strengths:

- strong peak model
- strict leakage handling
- clear artifact structure

Known limits:

- delta side remains hard multiclass problem
- performance varies materially by cutoff time
- analog stage is computationally expensive when enabled

## 17) Where to go next

- For metric interpretation: read `02`.
- For operations and rerun commands: read `03`.
- For run chronology and current status snapshots: read `04`.
- For feature-level lookup: read `06`.
