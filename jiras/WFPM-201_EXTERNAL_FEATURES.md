# WFPM-201 — Feature engineering: turn new external data tables into leakage‑free daily “as‑of T‑1 12Z” KMIA features

**Type:** Story  
**Priority:** P0 (this is where the information gain is realized)  
**Epic:** WFPM-EPIC-EXTDATA  

## Goal

Create a deterministic feature builder that, for each target day **T** (local day in `America/New_York`), produces one KMIA feature row using only information available at **asof_utc = (T−1) 12:00Z**.

The builder must pull from the new tables created in WFPM‑101..105 and output engineered features that measurably improve Tmax MAE and probability calibration.

## Critical design rule (non‑negotiable)

**Additional locations are NOT added as new training rows.**  
We still train a model that predicts **KMIA Tmax** only. Other stations are used only as **context features** injected into KMIA’s single feature vector per day.

Reason: mixing many stations into one supervised dataset changes the target definition and forces the model to learn cross‑station climatology rather than the specific Miami microclimate.

## Inputs (required tables)

- `iem_asos_obs` (WFPM‑101)
- `ndbc_stdmet_obs` (WFPM‑102)
- `fawn_obs_15min` and `fawn_daily_summary` (WFPM‑103)
- `igra2_derived_header` (+ optional `igra2_derived_level`) (WFPM‑104)
- `oisst_box_daily` (WFPM‑105)
- Existing gribstream feature table / dataset builder (already in repo)

## Output

A single daily feature row per target day for KMIA:

- Primary key: `target_date_local` (DATE)  
- Must also include: `asof_utc`, and any existing join keys already used by your training pipeline.

The feature builder should write:
- `features_external.csv` (or merge into existing `features.csv`)
- and/or persist a `daily_features_external` DB table for reuse.

## As‑of windows and conservative latency buffers

Define:

- `asof = target_date_local(T) at 12:00Z on T−1`

Then enforce source-specific cutoffs:

- ASOS (IEM): use obs where `valid_time_utc <= asof − 15 min`
- NDBC: use obs where `valid_time_utc <= asof − 30 min`
- FAWN last96: use obs where `end_time_utc <= asof − 30 min`
- FAWN daily summaries: only use dates `<= T−2` (daily is end-of-day)
- IGRA derived: use latest `sounding_time_utc <= asof − 12h` (fallback to <= asof − 36h)
- OISST: only use dates `<= T−2` (latency), and prefer rolling means ending at T−2

Also create **coverage features** per source/window so the model can learn when inputs are stale:
- `asos_kmia_obs_count_6h`, `ndbc_vakf1_obs_count_12h`, etc.

## Feature set specification (implement exactly)

### 1) ASOS ring features (IEM)

Stations: KMIA, KFLL, KOPF, KTMB, KHST, KPBI (optionally KEYW)

For each station S, compute for each variable group:

**Thermo (tmpf, dwpf, relh):**
- `S_tmpf_last_2h` = last non-null in (asof-2h, asof]
- `S_tmpf_mean_6h` = mean in (asof-6h, asof]
- `S_tmpf_slope_6h` = slope (°F/hour) from linear regression of tmpf vs time in last 6h (require ≥ 6 points else null)
- `S_tmpf_min_6h`, `S_tmpf_max_6h`
Repeat for `dwpf` and `relh`.

**Wind (wind_ms, drct_deg):**
- Convert to vector components using meteorological convention:
  - `dir_rad = drct_deg * pi/180`
  - `u_east = -wind_ms * sin(dir_rad)`  (positive = toward east)
  - `v_north = -wind_ms * cos(dir_rad)` (positive = toward north)
Compute:
- `S_wind_ms_mean_6h`
- `S_u_east_mean_6h`, `S_v_north_mean_6h`
- `S_wind_from_east_index_6h = mean(max(0, sin(dir_rad)) * wind_ms)` (proxy for onshore easterlies on east coast)
- `S_wind_dir_mode_sector_6h` as categorical sector (N/NE/E/SE/S/SW/W/NW) one-hot encoded later

**Pressure (altimeter_hpa, mslp_mb):**
- `S_mslp_last_2h`
- `S_mslp_trend_6h` = last − first within 6h (hPa)
- `S_altimeter_trend_6h`

**Precip (p01_mm):**
- `S_p01_sum_6h`, `S_p01_sum_24h` (sum of hourly precip; treat missing as 0 only if you have an observation; otherwise null)
- `S_rain_flag_24h` = 1 if sum_24h > 0.1mm else 0

**Cloudiness (optional but high value):**
Using `skyc1..4` and `skyl1..4`:
- `S_low_cloud_base_ft_min_6h` (min of skyl where skyc indicates BKN/OVC)
- `S_ceiling_flag_6h` (1 if any BKN/OVC below 5000 ft)

**Gradient features (most important):**
Compute differences against KMIA to encode sea-breeze gradients:

For each station S != KMIA:
- `dT_S_minus_KMIA_2h = S_tmpf_last_2h − KMIA_tmpf_last_2h`
- `dTd_S_minus_KMIA_2h = S_dwpf_last_2h − KMIA_dwpf_last_2h`
- `dMSLP_S_minus_KMIA_2h = S_mslp_last_2h − KMIA_mslp_last_2h`

**Coastal vs inland composites:**
Define:
- coastal set = {KMIA, KFLL}
- inland set = {KTMB, KHST}
Compute:
- `coastal_tmpf_mean_6h`, `inland_tmpf_mean_6h`, and `coastal_minus_inland_tmpf_6h`
- same for dewpoint and wind_from_east_index

These composites are often stronger than individual stations.

### 2) NDBC marine features (VAKF1, PEGF1)

For each marine station M:
- `M_wspd_ms_mean_6h`, `M_wdir_deg_last_2h`
- `M_u_east_mean_6h`, `M_v_north_mean_6h` (same vector method)
- `M_pres_last_2h`, `M_pres_trend_6h`
- if available: `M_atmp_c_mean_6h`, `M_dewp_c_mean_6h`, `M_wtmp_c_mean_6h`

**Marine-to-airport coupling:**
- `vakf1_minus_kmia_u_east_6h` (sea breeze forcing proxy)
- `vakf1_minus_kmia_pres_2h`
- `marine_wind_shift_flag` = 1 if marine wind is easterly while inland is westerly (classic sea-breeze front signature)

### 3) FAWN features (daily + last96)

**Daily summaries (use through T−2):**
For each FAWN station F:
- `F_rain_mm_tminus2`
- `F_rain_mm_sum_3d` (sum over T−2, T−3, T−4)
- `F_rain_mm_sum_7d`
- `F_tsoil_mean_c_tminus2`
- `F_tmean_c_tminus2` (if available)

**Near real-time last96 (optional operational add-on):**
Use last96 only if present (for live ops):
- `F_t2m_c_last`, `F_rh_last`, `F_ws_ms_last`, `F_rfd_last` at asof
- `F_rfd_mean_3h` (cloud proxy)
- `F_t2m_slope_6h`

### 4) IGRA sounding-derived features

From `igra2_derived_header`:
- `igra_pw_mm` (scaled)
- `igra_cape`, `igra_cin`
- `igra_li`, `igra_ki`, `igra_tti`, `igra_si`
- `igra_lclhgt_m`, `igra_lfchgt_m`, `igra_lnbhgt_m`
- `igra_invtempdiff_k`, `igra_invhgt_m`, `igra_mixhgt_m`

**Nonlinear transforms (tree-friendly but informative):**
- `cape_pos = max(cape, 0)`
- `cin_abs = abs(cin)` (CIN is negative; magnitude matters)
- `pw_cape_interaction = pw_mm * log1p(cape_pos)`
- `deep_convection_score = log1p(cape_pos) - log1p(cin_abs) + 0.01*pw_mm`

Also store:
- `igra_age_hours = (asof_utc - sounding_time_utc).total_hours`

### 5) OISST features (use through T−2)

From `oisst_box_daily`:
- `sst_mean_c_tminus2`
- `sst_anom_mean_c_tminus2`
Rolling (compute in feature builder, not in DB):
- `sst_mean_7d` ending T−2
- `sst_anom_7d` ending T−2
- `sst_trend_14d` = slope of sst_mean over last 14 days ending T−2

### 6) Cross-source meta-features (high leverage)

These compress many raw signals into a few powerful predictors:

- `morning_warmth_index = KMIA_tmpf_mean_6h - climatology_mean_tmpf_for_DOY_and_hour`
  - (Compute climatology on the fly from ASOS history; store as static lookup table.)
- `wet_ground_index = zscore(FAWN_rain_mm_sum_7d) - zscore(tsoil_mean_c_tminus2)`
- `sea_breeze_strength_index = vakf1_wind_from_east_index_6h + coastal_minus_inland_tmpf_6h`
- `convective_suppression_index = -igra_li + 0.05*igra_pw_mm + 0.002*igra_cape - 0.002*igra_cin_abs`

## Implementation details (Codex must implement)

### A) New feature builder module

Create `weather_ml/features/external_context.py` with a single public function:

`build_external_features(target_dates_local: list[date], db, station_zoneid='America/New_York') -> pandas.DataFrame`

This function:
- loops over target_dates_local
- computes asof_utc per date
- queries each table with the source-specific cutoffs
- computes the feature dict for that day
- returns a DataFrame indexed by target_date_local

### B) Deterministic aggregation helpers

Implement helpers:
- `last_value(df, col)`
- `window_mean(df, col)`
- `window_min/max`
- `slope_per_hour(df, col)` via least squares
- `wind_components(speed, dir_deg)` with the meteorological convention above
- `sector_onehot(dir_deg)` (8 sectors)

### C) Merge into existing training dataset

Add a step in the existing dataset build pipeline:
1) build base gribstream features
2) build external features
3) left-join on `target_date_local`
4) output merged features CSV used by training

### D) Leak checks (must exist)

For each target day row:
- assert(max_timestamp_used_by_source <= cutoff_for_source)
- store debug columns in a separate audit CSV:
  - `max_asos_time_used`, `max_ndbc_time_used`, etc.

## Validation & acceptance criteria

- Builder produces features for at least 2021‑01‑01 through present with no crashes.
- Coverage features exist and are non-null for at least ASOS and NDBC.
- Leak audit confirms no feature uses data after asof cutoffs.
- A small golden test:
  - pick one day and manually verify times used are <= asof.

## Definition of done

- New merged dataset build works end-to-end.
- Feature list is documented (generated `feature_columns_external.json`).
- Unit tests cover:
  - time cutoffs
  - slope calculation
  - wind components
  - joins and missing handling
