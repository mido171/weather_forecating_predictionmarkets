# 01 - System Spec and Implementation

This is the implementation-grade reference for the KLGA same-day Tmax distribution system.

It is intentionally detailed and is the authoritative reference for:

- data contracts (what columns/tables are allowed)
- as-of semantics (what timestamps are legal)
- label definitions (what peak/delta mean)
- feature engineering contract (what is computed and how)
- model training/calibration contract
- artifact contract (what outputs must exist after a successful run)

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

## 1.1 Code map (where things live)

Primary implementation package:

- `ml/src/weather_ml/klga_daily_tmax_dist/`

Key modules:

- `config.py`: station ids, cutoffs, split boundaries, constants, and guardrail configuration
- `db.py`: MySQL fetch helpers with allowed-column enforcement
- `timegrid.py`: NY-local cutoff grid generation with DST-safe UTC conversion
- `features.py`: feature engineering contract (snapshot, so-far, windows, neighbors, priors)
- `make_dataset.py`: builds and writes the parquet feature store
- `train_peak.py`: trains the LightGBM binary peak model + isotonic calibrator
- `train_delta.py`: trains the LightGBM multiclass delta model + temperature scaling
- `analog_knn.py`: optional analog posterior module and blending (disabled in baseline export mode)
- `infer.py`: PMF composition and bucket parsing utilities
- `pipeline.py`: orchestrates stages, writes artifacts, and produces reports/metrics

Operational extensions:

- exporter:
  - runner: `ml/run_klga_data_exporter.py`
  - module: `ml/src/weather_ml/data_exporter/`
- TabM experiment:
  - runner: `ml/run_training_tabm_klga_from_exports.py`
  - module: `ml/src/weather_ml/training/tabm_klga_from_exports.py`

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

### 2.1.1 Why "allowed vs banned" is strict

The observation table often contains a mixture of:

- instantaneous fields (safe for as-of usage)
- summary fields that may represent full-day extrema/totals (unsafe for as-of usage)

Even if a summary field is sparsely populated, it can create catastrophic leakage on the rows where it is present.

Therefore:

- the safe default is "ban unless proven instantaneous"
- this system hard-fails if banned columns are detected in the selected feature set

## 2.2 Daily truth table

Table:

- `wunderground_ml.wunderground_station_daily_max_temperature`

Used fields:

- `target_date_local`
- `max_temp_f`
- `station_zoneid`

Target station:

- `KLGA:9:US`

### 2.2.1 What "truth" means operationally

This table is treated as the market-aligned "truth":

- it is what the system trains to match
- it is what we evaluate against
- it is only used for priors on dates `< D` and for labels on date `D`

If truth rows are revised later in Wunderground:

- your live trading system must decide whether you use the revised value or the "finalized" value used by the market
- the model as implemented is aligned to "truth as stored at training time" and uses the same source semantics as your stored table

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

### 2.3.1 Why neighbors are in the same feature vector (not separate rows)

We are modeling one target station (KLGA).

Neighbors are used as contextual signals, not as separate training targets:

- each training row corresponds to one `(target_date_local, cutoff_minutes)` for KLGA
- neighbor observations are appended as additional features in the same row

This is important for leakage safety and interpretability:

- the label is always KLGA's daily max
- we never accidentally "train on neighbor truth and transfer"

## 2.4 Recommended indexes (speed + determinism)

Large historical windows make feature building expensive unless the DB access pattern is index-friendly.

Recommended indexes:

Observation table:

- `(request_location_id, valid_time_utc)`
- optionally `(request_location_id, valid_time_utc, temp)` for faster "latest temp <= cutoff" scans

Daily max table:

- `(request_location_id, target_date_local)`

If these indexes do not exist, runs can be orders of magnitude slower.

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

### 3.1 DST correctness and why you must not hardcode UTC offsets

NY local time changes offset due to DST.

So "04:00 local" is:

- 09:00 UTC during standard time (UTC-5)
- 08:00 UTC during daylight time (UTC-4)

If you hardcode offsets, you will:

- build incorrect observation windows
- silently introduce leakage or missingness artifacts near DST transitions

Correct approach:

- represent cutoffs as timezone-aware local datetimes in `America/New_York`
- convert to UTC for querying

### 3.2 Expected bins and coverage features

Coverage features depend on how many 30-minute slots should exist since midnight.

Because DST can make a local day have 23 or 25 hours, the expected number of bins must be computed using timezone-aware arithmetic on local datetimes.

Implementation note:

- the pipeline computes expected bins from `(cutoff_local - midnight_local)` in the local timezone and divides by 30 minutes

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

### 4.1 Why delta is clamped at 0

In a perfect world:

- `tmax_truth >= tmax_sofar`

But in reality, mismatches can occur due to:

- rounding and whole-degree truth semantics
- observation gaps
- slight source timing differences

So if a negative delta occurs, it is clamped:

- `delta = 0`
- `peak = 1`

This prevents impossible "negative remaining warming" labels.

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

### 5.9 Missing values and imputation contract

The system uses a two-layer approach:

1. explicit missingness flags (feature-level indicators)
2. numeric imputation for the model matrix

Imputation policy:

- compute per-feature median on the train split only
- fill any non-finite value (NaN/inf) with that train median
- persist the fill map to `imputer_values.json`

This is critical for:

- reproducibility
- standalone inference using exported models

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

### 6.1 Feature store caching and what gets recomputed

The feature store is expensive to build. So runs are designed to reuse it:

- if the feature store exists and `--force-rebuild-dataset` is not set:
  - dataset build is skipped/reused
  - models are retrained and evaluated on the cached feature store

This makes repeated experiments much faster while preserving leakage safety.

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

Training notes:

- peak outputs raw probabilities that are then mapped by isotonic regression
- calibration quality is assessed by both logloss and Brier score

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

Training notes:

- delta is trained only where truth indicates `delta>=1` (equivalently `peak=0`)
- delta metrics are computed only on that filtered subset

## 9) Optional analog kNN module

Implementation:

- `ml/src/weather_ml/klga_daily_tmax_dist/analog_knn.py`

Role:

- retrieval-based posterior generator
- blended with LGBM outputs when enabled

Current run-mode control:

- enabled by default
- disable with `--skip-analog-blend`

### 9.1 Why analog is treated as optional

Analog is valuable conceptually, but operationally:

- it is computationally expensive
- it increases complexity and failure surface area

So the system supports:

- a fast, reliable no-analog mode for exporting peak+delta
- a heavier analog-enabled mode for research runs

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

### 10.1 Concrete composition example

Suppose at cutoff:

- `round(tmax_sofar)=74`
- peak model says `p_peak=0.80`
- delta model (conditional) says:
  - `P(delta=1|>0)=0.50`
  - `P(delta=2|>0)=0.30`
  - `P(delta=3|>0)=0.20`

Then full delta PMF is:

- `P(delta=0)=0.80`
- `P(delta=1)=(1-0.80)*0.50=0.10`
- `P(delta=2)=(1-0.80)*0.30=0.06`
- `P(delta=3)=(1-0.80)*0.20=0.04`

And Tmax PMF is:

- `P(T=74)=0.80`
- `P(T=75)=0.10`
- `P(T=76)=0.06`
- `P(T=77)=0.04`

Bucket probabilities are then sums of these integer outcomes.

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

### 12.1 Guardrail philosophy

Guards are asserts/hard failures because:

- a silent leakage bug is worse than a crash
- metrics without correctness are actively harmful for trading decisions

### 12.2 Feature-generation performance policy (vectorization + numba)

Feature generation must remain leakage-safe while being optimized aggressively for runtime.

Required policy:

- preserve as-of semantics exactly (`valid_time_utc <= cutoff_utc`),
- preserve split isolation exactly (train/val/test by date),
- preserve truth usage exactly (date `D` daily max is label-only, never feature input),
- prefer vectorized NumPy operations over Python loops when mathematically equivalent,
- use Numba (`@njit`) in hot numeric loops (window stats, onset/change/run-length scans),
- keep safe fallback behavior if Numba is unavailable,
- treat any optimization that changes feature values vs the feature contract as a regression unless explicitly versioned.

Operational note:

- optimization is accepted only if leakage guards still pass and feature outputs remain numerically consistent within expected floating-point tolerance.

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

### 14.1 Full artifact contract (what a "complete run" must contain)

A complete run (LGBM pipeline) should contain:

- `run.log` ending in `PIPELINE_DONE`
- `metrics.json` and `metrics.md`
- `models/` with peak+delta + calibration artifacts
- `predictions/` parquet files
- `reports/` cutoff metrics and bucket calibration CSVs
- config snapshots (`config.json`, feature list, imputer values)

This contract allows:

- reproducible offline evaluation
- portable inference with exported models

## 15) Current authoritative no-analog export run

Run id:

- `20260226T081223Z`

Path:

- `artifacts/same_day_res_poly/20260226T081223Z/`

Mode:

- `skip_analog_blend=true`

Result:

- full artifact package present including exported peak and delta models.

## 16) Export bundle + TabM training (portable workflow)

After the initial system, two additional operational modules were implemented:

1. Exporter:
   - materializes the raw inputs into a portable CSV bundle under `exports/`
2. TabM training runner:
   - trains the same peak+delta decomposition from the exported bundle
   - writes a full evaluation artifact set into the same folder tree

Details:

- `07_exporter_and_remote_training_tabm.md`

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
