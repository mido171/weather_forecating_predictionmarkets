# HKG T+24 Tmax Forecasting System — Full Strategy Implementation Blueprint

**Prepared for:** HKG daily maximum temperature forecasting research project
**Target:** Hong Kong Observatory daily Tmax for target date `T`
**Primary decision cutoff:** `H24N`, 15:00 HKT on `T-1`
**Core objective:** produce the most accurate, fully point-in-time-safe, leakage-free point forecast of HKO daily Tmax, with the long-run competitive goal of approaching `0.45°C MAE` while preserving auditability.

---

## 0. Executive summary

This document is the core implementation blueprint for the full HKG T+24 Tmax forecasting system. It defines exactly how to combine:

1. the near-continuous HKO official forecast archive from 2000 through 2026;
2. the HKO target Tmax label history from 1884 onward;
3. GribStream tactical NWP model forecasts already fetched into PostgreSQL;
4. surrounding station and HKO station-network observations;
5. long-history target-memory features;
6. station-network microclimate features;
7. official-forecast residual memory;
8. NWP MOS experts;
9. ensemble probability experts;
10. regime specialists;
11. an expected-error router;
12. a distributional calibration layer;
13. final out-of-fold and sealed-holdout validation.

The key design principle is:

```text
Do not train one giant model on every column.
Build separate trustworthy experts, generate genuine OOF predictions, learn when each expert is reliable, route between them, then apply only validated bounded specialist corrections.
```

The final system should operate every day as:

```text
T-1 15:00 HKT cutoff
    ↓
point-in-time snapshot
    ↓
official forecast anchor
    ↓
expert forecasts: official residual, target-memory, station-network, GFS, GEFS, IFS, AI/NWP challengers, live sources
    ↓
expected-error router
    ↓
specialist detectors and benefit gates
    ↓
distributional calibration
    ↓
final conditional-median Tmax point forecast + threshold probabilities
    ↓
post-settlement scoring and online-state update
```

This strategy explicitly corrects an important prior assumption: the current database now contains a much stronger HKO official forecast archive than previously assumed. The clean usable official local min/max archive has **115,795 rows**, **9,667 distinct target dates**, an issue range from **2000-01-01 16:22:00 to 2026-06-20 23:45:00**, a target-date range from **2000-01-02 to 2026-06-21**, and only **one missing target date inside that range: 2003-02-02**. Therefore, official-forecast residual learning can be much stronger than the old 2000–2011 + 2021–2023 discontinuous view suggested.

---

## 1. Non-negotiable forecast contract

### 1.1 Target

The target is:

```text
actual_hko_tmax_c(T)
```

where `T` is the Hong Kong local calendar settlement date and `actual_hko_tmax_c` is the final observed daily maximum temperature at the Hong Kong Observatory target station.

### 1.2 Primary operational cutoff

The primary implemented decision cutoff is:

```text
cutoff_id: H24N
local decision time: T-1 15:00:00 HKT
UTC decision time: T-1 07:00:00 UTC
operational freeze: T-1 14:45:00 HKT / T-1 06:45:00 UTC
```

The formal cutoff is the moment the forecast decision is made. The operational freeze gives the live system time to validate inputs, produce features, run inference, write audit logs, and publish the forecast before the decision deadline.

### 1.3 Absolute leakage rule

For a target date `T`, a feature may be used only if the value, transformation, model run, forecast issue, or online state was available no later than the operational freeze:

```text
available_at_utc <= T-1 06:45:00 UTC
```

A feature is **not** safe merely because:

```text
its meteorological valid time is before T;
it appears in a retrospective archive;
it was later downloaded successfully;
it has a model run timestamp before cutoff;
it can be reconstructed after the fact.
```

A feature is safe only when its operational availability before the cutoff is proven or conservatively assumed by a documented release-latency rule.

### 1.4 Finalized daily target-history caution

At `T-1 15:00 HKT`, the finalized daily Tmax for `T-1` is generally **not guaranteed to be known**, because the local day has not ended. Therefore:

```text
safe finalized daily HKO target history for H24N defaults to T-2 and earlier.
```

The system may use `T-1` intraday observations only if they come from a live or exact-vintage source that includes no post-cutoff observations. Such features must be named explicitly as partial intraday features, for example:

```text
hko_partial_tminus1_max_before_1500_c
hko_partial_tminus1_temp_at_1450_c
hko_partial_tminus1_heating_0900_1450_c
```

Do **not** use finalized `T-1` daily max as a normal lag unless publication timing and cutoff eligibility are proven.

### 1.5 Sealed confirmation rule

Until explicitly opened, target outcomes after 2023 are treated as sealed for model selection and tuning:

```text
pre-2024: development and OOF research
2024: locked validation, opened once after system freeze
2025: final historical test, opened once after validation pass
2026 onward: prospective live replay or production scoring
```

Features for 2024–2026 may be downloaded and stored, but target outcomes from those years must not influence feature selection, hyperparameters, router thresholds, specialist thresholds, or promotion decisions before the relevant sealed stage is opened.

---

## 2. Source facts and external anchors

This strategy is grounded in the following source facts.

### 2.1 HKO ARWF principle

HKO's Automatic Regional Weather Forecast product uses a multi-model consensus system. It integrates multiple global NWP models, corrects model outputs using observations from stations over Hong Kong and the Pearl River Delta, and combines corrected forecasts with weights based on past performance. It is updated around noon and midnight local time.

Implementation implication:

```text
Our system should emulate this principle for the single target market:
multiple forecast sources → local HKO MOS corrections → past-performance weighting → final HKO Tmax forecast.
```

### 2.2 NWP principle

Numerical Weather Prediction simulates atmospheric evolution using physical equations involving wind, temperature, pressure, moisture and other variables. This means NWP data provides forward-looking atmospheric information that target-memory and station histories cannot contain by themselves.

Implementation implication:

```text
100+ years of HKO target history helps climatology, persistence and local thermal regimes.
NWP helps tomorrow's dynamic atmospheric state.
Both are needed.
```

### 2.3 GribStream endpoint principle

GribStream provides both `/timeseries` and `/runs` workflows. `/timeseries` can return best-available values by valid time, while `/runs` returns values per model run. For leakage-safe historical forecasting, exact model run time, lead time and selected run cycle matter. `asOf` is useful as a model-run-time cutoff, but it is not by itself proof of historical wall-clock availability.

Implementation implication:

```text
Primary training uses exact-cycle /runs data, conservative publication buffers, and stored availability grades.
Unrestricted /timeseries is not used for the primary OOF leaderboard.
```

### 2.4 Cross-validated stacking principle

The router/stacker must be trained on genuine out-of-fold predictions. The final combination must not learn from in-sample expert predictions. This follows the Super Learner / cross-validated stacking principle: produce cross-validated predictions from candidate learners, then select or learn combination weights based on out-of-fold risk.

Implementation implication:

```text
Every expert must generate OOF forecasts before the router can train.
No router may train on in-sample expert predictions.
```

---

## 3. Current data inventory and how each dataset is treated

### 3.1 HKO official forecast archive — primary anchor

Current database fact supplied by the project owner:

```text
table: public.hko_historical_forecasts_2000_2026
usable filter: row_quality_status = 'usable_local_minmax'
rows: 115,795
product type: local only
issue range: 2000-01-01 16:22:00 to 2026-06-20 23:45:00
target-date range: 2000-01-02 to 2026-06-21
distinct target dates: 9,667
missing target dates inside range: 1, 2003-02-02
```

Numerical summary:

| Attribute | Non-null rows | Min | Median | Mean | Max |
|---|---:|---:|---:|---:|---:|
| `forecast_min_c` | 115,795 | 1.0 | 24.0 | 22.2375 | 30.0 |
| `forecast_max_c` | 115,795 | 7.0 | 28.0 | 26.6010 | 39.0 |
| `forecast_range_c` | 115,795 | 2.0 | 4.0 | 4.3636 | 16.0 |
| `forecast_midpoint_c` | 115,795 | 6.0 | 26.0 | 24.4193 | 33.0 |
| `target_issue_lead_days` | 115,795 | 0.0 | 1.0 | 0.7643 | 1.0 |

This dataset is the central operational anchor. It should be used to build:

```text
official_raw_anchor_tmax
latest_pre_cutoff_official_tmax
official_min_c
official_range_c
official_midpoint_c
official_revision_features
official_text_features, if available
official_source_era
official_recent_residual_memory
official_expected_error
```

For H24N, the anchor selection must be:

```sql
WHERE row_quality_status = 'usable_local_minmax'
  AND product_type = 'local'
  AND target_issue_lead_days = 1
  AND issue_at_utc <= target_date_hkt - interval '1 day' + time '07:00 UTC equivalent'
```

In implementation, use a robust timezone expression rather than ambiguous string arithmetic. The selected row per target date should be:

```text
latest usable local min/max forecast issued before the H24N cutoff.
```

If multiple usable issue rows exist before cutoff, keep all of them in a revision table and select the latest row as the anchor. Derived revision features should include:

```text
first_pre_cutoff_forecast_max_c
latest_pre_cutoff_forecast_max_c
revision_max_c = latest - first
revision_abs_max_c
revision_count
minutes_from_latest_issue_to_cutoff
max_revision_direction
forecast_range_latest_c
forecast_range_revision_c
forecast_text_changed_flag, if text exists
```

This archive is valuable for two different reasons:

1. It provides the strongest direct forecast anchor.
2. It provides a large historical residual-learning sample: actual HKO Tmax minus official HKO forecast.

Because the archive is nearly continuous, official residual models can now be trained over the full pre-2024 development period, subject to exact H24N anchor selection.

### 3.2 HKO target Tmax labels — target and causal history

Use as:

```text
label: actual_hko_tmax_c(T)
long-history climatology source
long-history target-memory source
post-settlement online-state update source
```

Do not use `actual_hko_tmax_c(T)` or any target-day final daily climate value as an input for the same target date.

For H24N finalized daily lags:

```text
safe daily lags default to T-2 and earlier.
```

Recommended target-memory features:

```text
target_lag2_tmax_c
target_lag3_tmax_c
target_lag7_tmax_c
target_lag14_tmax_c
target_lag30_tmax_c
target_lag60_tmax_c
target_lag365_tmax_c
target_roll7_mean_lag2_c
target_roll14_mean_lag2_c
target_roll30_mean_lag2_c
target_roll7_std_lag2_c
target_roll14_std_lag2_c
target_slope_7_30_lag2_c_per_day
target_volatility_14_lag2_c
target_recent_range_14_lag2_c
target_hot_spell_length_lag2
target_cold_spell_length_lag2
target_anomaly_vs_causal_doy_climatology_lag2_c
target_yoy_analog_residual_lag2_c
```

If exact-vintage intraday HKO observations exist, add a separate intraday feature family. Never mix finalized daily and partial intraday semantics.

### 3.3 GribStream tactical NWP forecast data — modern dynamic atmosphere

The currently fetched GribStream data is stored as raw compressed API responses plus normalized wide forecast rows in PostgreSQL.

Main tables:

```text
nwp_tactical.acquisition_chunk
nwp_tactical.raw_response_object
nwp_tactical.forecast_wide
nwp_tactical.validation_issue
```

Main full-run scope:

```text
full_tactical_backfill_ok_tmax
```

Critical filter:

```sql
JOIN nwp_tactical.raw_response_object r
  ON r.response_object_id = fw.source_response_object_id
WHERE r.object_uri LIKE '%full_tactical_backfill_ok_tmax%'
```

Reason: older smoke rows exist in `forecast_wide`. Do not train or score without filtering to the full tactical backfill scope unless smoke rows are purged.

Current full tactical rows:

```text
full-run normalized rows: 1,964,157
raw objects checked: 1,163
missing raw files: 0
raw byte-size mismatches: 0
API HTTP errors: 0
estimated credits consumed: 1,889,276
```

Current model inventory from the uploaded GribStream file:

| Dataset | Role | Rows | Target-date HKT range | Tmax status |
|---|---:|---:|---|---|
| `gfs` | Core deterministic NWP | 575,004 | 2021-03-23 to 2026-06-23 | Usable |
| `gefsatmosmean` | GEFS ensemble mean | 200,436 | 2020-10-03 to 2026-06-23 | Usable |
| `gefsatmos` | GEFS ensemble members | 516,891 | 2020-10-03 to 2026-06-23 | Usable |
| `ifsoper` | IFS deterministic | 91,260 | 2024-02-29 to 2026-06-23 | Usable |
| `ifsenfo` | IFS ensemble members | 343,616 | 2024-03-03 to 2026-06-23 | Usable with member-0 caveat |
| `cwawrf15` | Rolling/prospective deterministic | 180 | 2026-06-23 to 2026-06-26 | Live/rolling only |
| `aifsoper` | AI deterministic | 28,884 | 2025-02-26 to 2026-06-23 | Usable, short history |
| `aifsenfo` | AI ensemble | 72,270 | 2025-07-04 to 2026-06-23 | Usable, short history |
| `aigfssfc` | AI/GFS surface deterministic | 3,660 | 2026-04-22 to 2026-06-23 | Usable over very short range |
| `aigfspres` | AI/GFS pressure-level support | 3,660 | 2026-04-22 to 2026-06-23 | Support only |
| `aigefssfc` | AI/GEFS surface ensemble | 46,252 | 2025-06-03 to 2026-06-23 | Blocked as Tmax source |
| `graphcast` | AI deterministic | 44,220 | 2024-04-26 to 2026-05-06 | Usable through archive end |
| `fourcastnetgfs` | AI/GFS deterministic | 37,824 | 2024-05-03 to 2026-02-20 | Usable through observed archive end |
| `nbmoc` | Probe-only | 0 | empty | Not usable |

The current GribStream leakage-safe extraction rule from the uploaded inventory is:

```sql
SELECT fw.*
FROM nwp_tactical.forecast_wide fw
JOIN nwp_tactical.raw_response_object r
  ON r.response_object_id = fw.source_response_object_id
WHERE r.object_uri LIKE '%full_tactical_backfill_ok_tmax%'
  AND fw.run_time_utc + interval '6 hours'
      <= ((fw.target_date_hkt - 1) + time '15:00') AT TIME ZONE 'Asia/Hong_Kong'
  AND fw.dataset_code NOT IN ('nbmoc', 'aigfspres', 'aigefssfc');
```

The exact timestamp expression must be implemented in project code and unit-tested for timezone correctness.

Usable as Tmax forecast sources right now, after full-run and leakage filters:

```text
gfs
gefsatmosmean
gefsatmos
ifsoper
ifsenfo
aifsoper
aifsenfo
aigfssfc
graphcast
fourcastnetgfs
cwawrf15 prospectively/live only
```

Not usable as daily Tmax forecast sources right now:

```text
nbmoc
aigfspres
aigefssfc
```

Known caveats:

```text
ifsenfo: member 0 missing in some recent run/valid groups.
fourcastnetgfs: observed archive ends at 2026-02-18 18Z / target dates through 2026-02-20.
nbmoc: probe returned zero rows.
aigefssfc: many target days lack usable Tmax/2m temperature candidate.
aigfspres: pressure-level support only, not surface Tmax.
```

### 3.4 HKO daily climate elements — diagnostic and lagged-cautious source

HKO daily climate elements include physically rich values such as mean temperature, pressure, rainfall, humidity, cloud, sunshine, solar radiation, sea temperature, wind and visibility. These are valuable for mechanism discovery.

But finalized daily tables without first-publication proof are not automatically operational inputs.

Use cases:

```text
label cross-checks;
long-history diagnostic mechanism mining;
causally lagged features only when publication timing is proven or lags are sufficiently conservative;
training physical-proxy ideas for safe data sources.
```

Do not use target-day daily climate values as deployable predictors.

### 3.5 NOAA ISD regional station data — station-network research/proxy source

ISD regional station data is valuable for:

```text
station temperature anomaly;
dewpoint change;
temperature-dewpoint spread;
pressure tendency;
coastal-inland gradients;
station disagreement;
station-group regime detection;
spatial propagation and microclimate state.
```

However, previous audits flagged major issues, especially broken/corrupted wind direction in existing normalized ISD data. Use repaired and validated station data only.

Treat historical ISD as:

```text
research-proxy / diagnostic unless exact operational availability is proven.
```

Prospective station/live feeds with exact retrieval timestamps can become deployable live features.

### 3.6 NOAA IGRA upper-air — diagnostic physics teacher

IGRA upper-air data can explain mechanisms such as:

```text
lower-tropospheric warmth;
1000/925/850 hPa structure;
moisture profile;
700 hPa humidity;
500 hPa ridge height;
inversion and stability;
vertical wind regime.
```

But prior audit found sentinel/scaling issues and release-latency uncertainty. Therefore:

```text
Do not use IGRA as production predictor unless cleaned and release timing is proven.
Use it as diagnostic teacher for proxy design.
```

Safe proxy examples:

```text
IGRA ridge/subsidence signal → GFS/IFS 500 hPa height + station pressure tendency.
IGRA moisture profile → GFS/GEFS/IFS humidity + station dewpoint change.
IGRA low-level warmth → GFS/GEFS 850 hPa temperature + station anomalies.
```

### 3.7 Tropical cyclone best track — retrospective diagnostic labels

Best-track data is retrospective. It should be used only as:

```text
diagnostic regime label;
post-analysis explanation;
teacher signal for designing live-safe TC proximity features from operational forecasts.
```

Do not use finalized best-track values as live predictors unless operational publication timing is proven.

### 3.8 HKO radar, satellite, lightning, nowcast feeds — live/recent layer

These feeds are potentially powerful for:

```text
cloud/rain suppression;
convective cloud state;
short-term solar interruption;
precutoff rain/cloud nowcast;
uncertainty and no-trade flags.
```

But the current history is very short. Use as:

```text
prospective live features;
short-history shadow scoring;
future specialist triggers after enough observations accumulate.
```

### 3.9 HKO marine/tide/coastal waters — live/recent marine context

Use for:

```text
marine suppression proxy;
sea-breeze context;
coastal waters wind regime;
humidity and onshore-flow regime.
```

Currently, use prospectively or diagnostically unless a long exact-vintage archive is established.

### 3.10 ARWF — high-value prospective local model anchor

ARWF is a local HKO station forecast system. Since no strong historical exact-vintage ARWF archive is currently available in the project, treat it as:

```text
high-value prospective collector now;
shadow expert during year 1;
small capped challenger after sufficient history;
full router expert after at least two complete seasonal cycles if OOF/live evidence supports it.
```

Do not insert ARWF into historical pre-2024 router training.

### 3.11 Static geospatial data — deterministic station context

Static features are safe if computed deterministically and versioned:

```text
station latitude/longitude/elevation;
distance and bearing from HKO;
coastal/inland/island/urban/hill labels;
distance to water;
station group membership;
NWP stencil mapping;
station-to-grid mapping.
```

Use static data to structure station and NWP features, not as a blind feature dump.

### 3.12 Experiment outputs — research evidence, not production source

Prior experiment outputs should be stored as evidence:

```text
scoreboards;
OOF predictions;
feature diagnostics;
negative results;
champion registry;
leakage audit outcomes.
```

Do not use old feature matrices as canonical production sources unless their generation code, point-in-time contract and input eligibility are revalidated.

---

## 4. Canonical H24N snapshot

The system must build one canonical snapshot per target date `T`.

### 4.1 Snapshot identity

Each snapshot row must have:

```text
target_date_hkt
cutoff_id = H24N
formal_cutoff_hkt = T-1 15:00:00
formal_cutoff_utc = T-1 07:00:00
operational_freeze_hkt = T-1 14:45:00
operational_freeze_utc = T-1 06:45:00
snapshot_created_at_utc
snapshot_code_version
feature_store_version
```

### 4.2 Snapshot components

The snapshot contains only data available by the operational freeze:

```text
latest eligible official HKO forecast;
all earlier official revisions before cutoff;
HKO target daily history through safe finalized lag, default T-2 and earlier;
optional exact-vintage T-1 intraday station observations up to cutoff;
station-network data available before cutoff;
GribStream NWP rows passing full-run scope and H24N safety filter;
online residual states based only on previously settled target dates;
static geography;
calendar/time features.
```

The snapshot must not contain:

```text
actual_hko_tmax_c(T);
any target-day observation after cutoff;
finalized daily T or T-1 data if not known by cutoff;
later official forecast revisions;
later model runs;
feature scaling fitted on future dates;
outcome-derived flags such as official_overforecast_c.
```

### 4.3 Snapshot validation gates

Every snapshot build must report:

```text
number of target dates;
number with official anchor;
number with GFS features;
number with GEFS features;
number with station features;
number with target-memory features;
number with each optional model family;
missingness by year/month/season;
all rows failing cutoff rules;
all rows with ambiguous timezone mapping;
all duplicate source keys;
all source versions used.
```

If post-cutoff source rows are detected, the feature view must fail closed.

---

## 5. Feature families

### 5.1 Official forecast features

From the selected H24N official anchor:

```text
official_tmax_c
official_tmin_c
official_range_c
official_midpoint_c
official_issue_at_utc
official_minutes_to_cutoff
official_target_issue_lead_days
official_source_era
official_product_type
official_row_quality_status
```

Revision features:

```text
official_revision_count_pre_cutoff
official_first_pre_cutoff_tmax_c
official_latest_pre_cutoff_tmax_c
official_tmax_revision_c
official_tmax_revision_abs_c
official_revision_direction
official_range_revision_c
minutes_since_first_issue
minutes_since_latest_issue
```

If text exists:

```text
forecast_text_tokens_fold_local
weather_text_bucket
wind_text_bucket
rain_or_shower_flag
thunderstorm_flag
sunny_flag
cloudy_flag
very_hot_flag
mist_fog_flag
monsoon_or_easterly_flag
```

These must be fitted fold-locally, not with full-history vocabulary learned from the future.

### 5.2 Official residual-memory features

After each target settles, compute:

```text
official_residual_t = actual_hko_tmax_t - official_tmax_t
```

For live target `T`, residual states may use only settled dates `< T`.

Features:

```text
official_bias_ewma_h5
official_bias_ewma_h10
official_bias_ewma_h20
official_bias_ewma_h40
official_abs_error_ewma_h10
official_abs_error_ewma_h20
official_residual_volatility_h20
official_overforecast_streak
official_underforecast_streak
official_source_era_bias_h20
official_month_bias_h20
official_season_bias_h20
official_forecast_range_bias_h20
```

All half-life, cap, shrinkage and minimum-history settings are selected in walk-forward validation.

### 5.3 Target-memory features

Safe default for H24N finalized daily labels: lags ending at `T-2`.

Features:

```text
target_lag2_tmax_c
target_lag3_tmax_c
target_lag4_tmax_c
target_lag7_tmax_c
target_lag14_tmax_c
target_lag30_tmax_c
target_lag60_tmax_c
target_lag365_tmax_c
target_lag730_tmax_c
target_roll7_mean_lag2_c
target_roll14_mean_lag2_c
target_roll30_mean_lag2_c
target_roll60_mean_lag2_c
target_roll7_std_lag2_c
target_roll14_std_lag2_c
target_roll30_std_lag2_c
target_roll7_min_lag2_c
target_roll7_max_lag2_c
target_roll14_range_lag2_c
target_slope_7_30_lag2_c_per_day
target_slope_14_60_lag2_c_per_day
target_curvature_lag2_c
target_recent_volatility_expanding_state
target_hot_spell_length_lag2
target_cool_spell_length_lag2
target_breakout_above_roll30_lag2_c
target_anomaly_vs_causal_doy_clim_lag2_c
target_climate_trend_adjusted_anomaly_lag2_c
```

Calendar/climatology:

```text
day_of_year_sin1/cos1
day_of_year_sin2/cos2
day_of_year_sin3/cos3
month
season
MAM_flag
JJA_flag
SON_flag
DJF_flag
causal_doy_mean
causal_doy_median
causal_doy_p25/p75
causal_doy_recent_decade_mean
causal_trend_adjusted_climatology
```

Causal climatology must be computed using only dates before the target date within each training fold.

### 5.4 Station-network features

Use repaired and validated station data only. For historical ISD, maintain strict/proxy distinction.

Station-level features:

```text
station_latest_temp_before_cutoff_c
station_latest_dewpoint_before_cutoff_c
station_latest_pressure_before_cutoff_hpa
station_temp_dewpoint_spread_c
station_temp_anomaly_14d_c
station_temp_anomaly_30d_c
station_dewpoint_anomaly_14d_c
station_dewpoint_change_24h_c
station_pressure_tendency_24h_hpa
station_temperature_slope_pre_cutoff
station_missingness_flag
station_obs_count_before_cutoff
```

Network features:

```text
coastal_inland_temp_spread_c
coastal_inland_dewpoint_spread_c
marine_hko_like_temp_spread_c
north_south_temp_gradient_c
east_west_temp_gradient_c
pressure_gradient_proxy_hpa
station_disagreement_index_temp
station_disagreement_index_dewpoint
station_rank_hko_like_among_peers
station_rank_reversal_flag
station_group_mode_1
station_group_mode_2
upwind_station_temp_anomaly_c
upwind_station_dewpoint_change_c
```

If wind direction is repaired and validated:

```text
station_wind_u_component
station_wind_v_component
onshore_wind_component
wind_speed_before_cutoff
wind_shift_24h
wind_direction_sector
```

If wind direction is not validated, do not derive wind-sector or upwind features from that source.

### 5.5 NWP features

All NWP features must be derived from rows passing the strict H24N GribStream retrieval filter.

General model features:

```text
model_hko_center_sampled_tmax_c
model_hko_center_sampled_temp_at_02/05/08/11/14/17/20/23_hkt_c
model_peak_hour_hkt
model_heating_08_to_14_hkt_c
model_heating_11_to_17_hkt_c
model_afternoon_cooling_14_to_20_hkt_c
model_diurnal_range_sampled_c
model_dewpoint_at_peak_c
model_temp_dewpoint_spread_at_peak_c
model_wind_speed_at_peak_mps
model_u10_at_peak_mps
model_v10_at_peak_mps
model_onshore_component_at_peak_mps
model_mslp_at_peak_hpa
model_precip_window_total_mm
model_shortwave_heating_window_sum
model_low_cloud_heating_window_mean_pct
model_850_temp_at_peak_c
model_925_temp_at_peak_c
model_700_rh_at_peak_pct
model_500_height_at_peak_m
```

Spatial features for deterministic/mean 12-point stencil:

```text
model_hko_minus_marine_s_temp_c
model_hko_minus_marine_e_temp_c
model_inland_nw_minus_hko_temp_c
model_inland_nw_minus_marine_s_temp_c
model_local_north_south_gradient_c
model_local_east_west_gradient_c
model_local_temp_spatial_std_c
model_marine_suppression_proxy
model_inland_heat_proxy
model_pressure_gradient_proxy
```

GEFS member features:

```text
gefs_member_tmax_c_0..30
gefs_tmax_mean_c
gefs_tmax_median_c
gefs_tmax_p10_c
gefs_tmax_p25_c
gefs_tmax_p75_c
gefs_tmax_p90_c
gefs_tmax_std_c
gefs_tmax_iqr_c
gefs_tmax_skew
gefs_prob_tmax_ge_30_5
gefs_prob_tmax_ge_31_5
gefs_prob_tmax_ge_32_5
gefs_prob_tmax_ge_33_5
gefs_prob_tmax_ge_34_5
```

IFS ensemble features mirror GEFS where available.

### 5.6 Cross-source contradiction features

These are essential for routing.

```text
official_minus_gfs_mos_c
official_minus_gefs_median_c
official_minus_gefs_mean_c
official_minus_ifs_mos_c
official_minus_ifsenfo_median_c
official_minus_aifs_c
gfs_minus_gefs_median_c
gfs_minus_ifs_c
gefs_minus_ifsenfo_c
gfs_minus_station_microclimate_c
gefs_minus_station_microclimate_c
official_minus_target_memory_c
official_minus_target_roll30_lag2_c
official_rank_among_experts
model_forecast_std_c
model_forecast_range_c
fraction_experts_above_official
fraction_experts_below_official
```

These features help answer whether the official forecast is unsupported by other evidence.

---

## 6. Expert models

Each expert is trained separately. Each expert produces out-of-fold predictions before the router is trained.

### 6.1 Expert E0 — official raw anchor

This is not trained.

```text
forecast = latest eligible official_tmax_c
```

Every model must beat this baseline on identical rows.

### 6.2 Expert E1 — official residual expert

Training target:

```text
official_residual_t = actual_hko_tmax_t - official_tmax_t
```

Model output:

```text
predicted_official_residual_t
corrected_official_tmax_t = official_tmax_t + predicted_official_residual_t
```

Inputs:

```text
official features;
official revision features;
official text features;
official residual-memory features;
target-memory features;
station-network features where eligible;
NWP contradiction features where available;
season/month/source-era features.
```

Development data:

```text
2000-01-02 through 2023-12-31, using H24N eligible official anchors.
```

Because the official archive is now near-continuous, this becomes the most important long-history residual expert.

Recommended model types:

```text
regularized linear residual model;
Huber regression;
small gradient boosting residual model;
quantile residual model for median correction;
source/month/season random-effect-style shrinkage.
```

Corrections must be capped and shrunk:

```text
initial correction cap: ±0.8°C
preferred conservative cap for production candidate: ±0.5°C unless OOF proves larger safe
minimum source history for online correction: 20 scored rows
```

### 6.3 Expert E2 — target-memory expert

Purpose:

```text
Forecast HKO Tmax from long HKO history alone.
```

This expert is not expected to beat official, but it supplies:

```text
thermal persistence;
climatological prior;
mean-reversion state;
transition state;
router disagreement input.
```

Training data:

```text
1884 through 2023, causal expanding windows.
```

Safe finalized daily features end at `T-2` unless intraday data is proven.

Output:

```text
target_memory_tmax_oof
```

### 6.4 Expert E3 — station-network microclimate expert

Purpose:

```text
Detect HKO-local microclimate deviations that official/NWP may miss.
```

Primary output:

```text
small residual correction or station_microclimate_tmax estimate;
station_expected_error;
regime flags.
```

Inputs:

```text
station anomalies;
dewpoint changes;
temp-dewpoint spreads;
pressure tendencies;
coastal/inland gradients;
station disagreement;
station group modes;
static station geography.
```

Training distinction:

```text
strict deployable: only sources with exact pre-cutoff availability proof;
research proxy: ISD/diagnostic station archive until operational availability is proven.
```

Report strict and proxy scoreboards separately.

### 6.5 Expert E4 — GFS MOS expert

Purpose:

```text
Correct GFS target-day trajectory to HKO station Tmax.
```

Training target:

```text
gfs_residual_t = actual_hko_tmax_t - gfs_direct_tmax_t
```

Inputs:

```text
GFS direct Tmax;
GFS hourly/sampled thermal trajectory;
GFS dewpoint/humidity;
GFS wind;
GFS cloud/rain/radiation;
GFS vertical thermal structure;
GFS spatial stencil gradients;
target-memory state;
station microclimate state;
calendar/season.
```

Development data:

```text
GFS target dates through 2023-12-31.
```

Given current fetched range:

```text
2021-03-23 through 2023-12-31 for pre-2024 development.
```

### 6.6 Expert E5 — GEFS ensemble expert

Purpose:

```text
Use ensemble distribution for median Tmax and uncertainty.
```

Outputs:

```text
gefs_calibrated_median_tmax
gefs_probabilities_by_threshold
gefs_expected_abs_error
gefs_uncertainty_features
```

Training targets:

```text
actual_hko_tmax;
GEFS median residual;
threshold indicators;
absolute error.
```

Development data:

```text
GEFS target dates through 2023-12-31.
```

Use member-level HKO center output plus GEFS mean spatial features.

### 6.7 Expert E6 — IFS deterministic expert

Purpose:

```text
Independent ECMWF deterministic MOS challenger.
```

Training availability:

```text
IFS begins in 2024 in current fetched inventory.
```

Under sealed-confirmation policy, IFS does not participate in pre-2024 development. It enters only after the frozen core system is evaluated and 2024/2025 is opened under controlled rules.

Initial integration:

```text
shadow expert;
then capped challenger adapter;
initial router weight cap ≤ 10% until enough OOF evidence exists.
```

### 6.8 Expert E7 — IFS ensemble expert

Purpose:

```text
Independent ECMWF ensemble uncertainty and median forecast.
```

Caveat:

```text
member 0 missing in some recent groups.
```

Training policy mirrors IFS deterministic: sealed short-history challenger until enough validation evidence exists.

### 6.9 Expert E8 — AI model challengers

Candidate sources:

```text
aifsoper
aifsenfo
aigfssfc
graphcast
fourcastnetgfs
```

Roles:

```text
short-history direct forecast challengers;
model-disagreement features;
router diversity features;
AI-vs-physics contrast.
```

Initial integration:

```text
shadow only or capped challenger;
no large weights until stable OOF/sealed validation evidence exists.
```

Blocked for Tmax source:

```text
aigefssfc due insufficient usable Tmax coverage;
aigfspres because upper-air support only;
nbmoc because zero rows.
```

### 6.10 Expert E9 — CWA WRF live/prospective expert

Purpose:

```text
High-resolution regional East Asia model expert.
```

Current history is only a few target days, so use as:

```text
live collector;
shadow expert;
prospective specialist after sufficient history;
not a historical backtest source yet.
```

### 6.11 Expert E10 — diagnostic physics proxy expert

Uses diagnostic-only sources to design safe proxy features.

Examples:

```text
IGRA 850hPa temperature teacher → GFS/IFS 850hPa temperature + station anomalies.
HKO sea temperature teacher → NWP marine stencil + coastal-inland spreads.
Daily climate cloud/sunshine teacher → NWP cloud/radiation + radar/satellite live features.
TC best-track teacher → NWP wind/pressure/rain fields + operational TC bulletins.
```

The teacher variables do not directly enter deployable scoring unless timestamp eligibility is proven.

---

## 7. Out-of-fold expert generation

### 7.1 Why OOF is mandatory

The router cannot train on in-sample expert predictions. For every expert and target date, the prediction must be generated by a model that did not train on that target date.

For target date `t`:

```text
fit expert using only dates < t
predict t
save prediction
later compare to actual_hko_tmax(t)
```

### 7.2 OOF prediction table

Create:

```text
modeling.oof_expert_prediction
```

Required columns:

```text
target_date_hkt
cutoff_id
expert_id
expert_version
training_window_start
training_window_end
prediction_created_at_utc
forecast_tmax_c
predicted_residual_c
predicted_abs_error_c
prediction_quantile_p10_c
prediction_quantile_p50_c
prediction_quantile_p90_c
feature_snapshot_id
model_artifact_id
is_oof
is_strict_deployable
availability_mask
```

### 7.3 OOF folds

For official-only and long-history experts:

```text
Use expanding walk-forward folds from 2000 through 2023.
Minimum initial training: at least 2 full years or a defined initial warmup.
Test windows: monthly, quarterly, semiannual, or annual, but must be chronological.
```

For modern GFS/GEFS experts:

```text
Development overlap: 2021-03-23 through 2023-12-31.
Use expanding folds, for example:
  Train through 2021-12-31, test 2022-H1
  Train through 2022-06-30, test 2022-H2
  Train through 2022-12-31, test 2023-H1
  Train through 2023-06-30, test 2023-H2
```

For IFS and AI models:

```text
Do not enter pre-2024 development router.
Produce shadow OOF only when sealed periods are opened under the validation protocol.
```

---

## 8. Router design

### 8.1 Router purpose

The router decides how much to trust each expert on a target date.

It does not learn rules such as:

```text
marine day = 10% official, 40% GEFS
```

directly. Instead, it learns:

```text
Given facts known at cutoff, what is each expert's expected absolute error today?
```

Then it converts expected errors into weights.

### 8.2 Router training table

One row per target date:

```text
target_date_hkt
actual_hko_tmax_c
expert forecasts: official, official_corrected, target_memory, station, gfs, gefs, ...
expert errors: abs(actual - expert_forecast)
router context features known at cutoff
```

Example context features:

```text
gefs_spread_c
official_minus_gfs_c
official_minus_gefs_c
gfs_minus_gefs_c
model_forecast_std_c
coastal_inland_spread_c
onshore_wind_component
station_disagreement_index
dewpoint_change_24h_c
official_recent_bias_h20_c
season
month
MAM_flag
source_era
missing_expert_mask
```

### 8.3 Expected-error models

For each expert `e`, define:

```text
L_e,t = abs(actual_hko_tmax_t - forecast_e,t)
```

Train:

```text
predicted_expected_error_e,t = h_e(context_t)
```

Use small, regularized models:

```text
ridge regression on error;
Huber/quantile regression;
monotonic GAM for uncertainty variables;
small gradient boosting model with strict depth and shrinkage.
```

Do not use large unconstrained models for the first router.

### 8.4 Weight conversion

Dynamic weights:

```text
w_dyn_e,t = exp(-predicted_error_e,t / tau) / sum_j exp(-predicted_error_j,t / tau)
```

Where:

```text
tau controls how aggressively the router favors the predicted winner.
```

Static prior weights are learned from OOF expert forecasts by minimizing OOF MAE on the training window.

Final weights:

```text
w_e,t = (1 - lambda) * w_static_e + lambda * w_dyn_e,t
```

`lambda` is selected in walk-forward validation. Because the modern NWP overlap is short, `lambda` should start conservative.

Recommended initial constraints:

```text
lambda <= 0.50 for the first modern router;
minimum expert weight floor may be 0 for unavailable experts;
maximum short-history challenger weight <= 0.10;
weights sum to 1 after availability masks;
no expert can receive nonzero weight if its input is unavailable or ineligible.
```

### 8.5 Availability masks

On each date, the router must mask unavailable experts:

```text
if GFS missing → w_gfs = 0
if GEFS missing → w_gefs = 0
if IFS not eligible before sealed opening → w_ifs = 0
if ARWF not yet trained → w_arwf = 0
if station source is proxy-only under strict scoreboard → w_station_strict = 0, but station_proxy can exist in proxy scoreboard
```

Weights are renormalized after masking.

### 8.6 Router versions

#### Router R0 — long-history official router

Experts:

```text
official_raw
official_residual_corrected
target_memory
station_proxy_or_strict
online_residual_memory
```

Training:

```text
2000 through 2023, using the corrected near-continuous official archive.
```

Purpose:

```text
Strong official-forecast residual system before NWP.
```

#### Router R1 — core modern GFS/GEFS router

Experts:

```text
official_residual_corrected
gfs_mos
gefs_calibrated_median
target_memory
station_microclimate
online_residual_memory
```

Training:

```text
2021-03-23 through 2023-12-31 where official + GFS + GEFS + target labels overlap.
```

Purpose:

```text
First true multi-model H24N system.
```

#### Router R2 — IFS challenger adapter

Experts:

```text
core_R1_system
ifs_mos
ifsenfo_median
```

Training:

```text
Only after sealed 2024/2025 protocol allows it.
```

Initial formula:

```text
forecast_new = (1 - rho) * forecast_core_R1 + rho * forecast_ifs_adapter
rho <= 0.10 initially
```

#### Router R3 — AI challenger adapter

Experts:

```text
core_R1_or_R2_system
aifsoper
aifsenfo
aigfssfc
graphcast
fourcastnetgfs
```

Initial integration:

```text
shadow, then capped, then full only if validated.
```

#### Router R4 — live ARWF/CWA adapter

Training:

```text
prospective after enough exact-vintage data accumulates.
```

Initial integration:

```text
Year 1: shadow only
After 365 days: capped at 5–10% if stable
After 730+ days: eligible for normal competition if stable
```

---

## 9. Specialist system

### 9.1 Specialist philosophy

A specialist is not a manual rule. It is a learned, gated, bounded correction module.

Every specialist has three learned components:

```text
1. Regime detector
2. Residual correction model
3. Benefit/abstention gate
```

### 9.2 Specialist training target

For an anchor forecast:

```text
anchor_residual_t = actual_hko_tmax_t - anchor_forecast_t
```

The specialist predicts the residual only inside its regime.

Benefit target:

```text
benefit_t = abs(actual - anchor_forecast) - abs(actual - specialist_forecast)
```

Positive benefit means the specialist helped.

### 9.3 Generic activation rule

A specialist can activate only when:

```text
P(regime) >= learned_threshold
AND expected_benefit >= learned_threshold
AND sample_support >= minimum_support
AND correction_uncertainty acceptable
AND no-harm gate passes
```

Starting promotion gates:

```text
minimum active historical days: 200 where possible
minimum active rows per main fold: 40 where possible
minimum distinct years: 3
stable correction sign
positive fold-local lift in at least 3 folds
no worsening of P90/P95 absolute error
no meaningful harm outside activation slice
```

For rarer regimes, shrink correction heavily toward zero.

### 9.4 Specialist S1 — marine suppression

Hypothesis:

```text
Official or deterministic NWP may overforecast HKO when inland heat is present but marine/onshore/coastal flow suppresses the target station.
```

Inputs:

```text
onshore/easterly wind component
coastal-inland station spread
NWP HKO-minus-marine spread
NWP inland-minus-HKO spread
dewpoint level/change
low cloud
shortwave suppression
rain probability
station disagreement
official-minus-GEFS
official-minus-GFS
official-minus-ARWF when available
```

Target sign:

```text
negative official residual: actual < official
```

Training:

```text
NWP version: 2021–2023 core overlap
station/official proxy version: 2000–2023 if station features eligible/proxy-labelled
ARWF version: prospective only
```

### 9.5 Specialist S2 — weak-wind heat buildup

Hypothesis:

```text
Official may underforecast HKO Tmax when sunshine, low cloud, weak winds, warm low-level NWP structure and warm station anomalies align.
```

Inputs:

```text
low wind speed
weak onshore component
high shortwave radiation
low cloud
low or zero precipitation
warm 850hPa temperature
positive station temperature anomalies
wide temp-dewpoint spread
hot-spell state
positive target slope
```

Target sign:

```text
positive official residual: actual > official
```

Training:

```text
NWP version: 2021–2023
long-history target/official version: 2000–2023
```

### 9.6 Specialist S3 — MAM transition

Hypothesis:

```text
March/April/May transition produces large errors due to shifts between cool monsoon, humid maritime flow, low cloud/fog/rain, and rapid heating breaks.
```

Inputs:

```text
MAM calendar flag
target slope 7 vs 30
target slope reversal
dewpoint change
pressure tendency
wind shift
station warming disagreement
NWP cloud/rain
NWP shortwave
official minus recent target memory
official/NWP disagreement
```

Validation:

```text
leave-one-spring-out where feasible.
```

Training:

```text
official-only MAM: 2000–2023 MAM seasons
NWP-enhanced MAM: 2021–2023 MAM, heavily shrunk
IFS-enhanced MAM: after sealed periods open
```

### 9.7 Specialist S4 — cloud/rain suppression

Hypothesis:

```text
Forecast Tmax may be too high when target-day cloud, rain, high humidity and reduced radiation suppress daytime heating.
```

Inputs:

```text
NWP low cloud
NWP precipitation
NWP shortwave
GEFS spread in cloud/rain proxy
station dewpoint surge
temp-dewpoint spread narrowing
forecast text showers/thunderstorms/rain
```

Target sign:

```text
negative residual against official or raw NWP.
```

### 9.8 Specialist S5 — dry subsidence / ridge heating

Hypothesis:

```text
Forecast Tmax may be too low when subtropical ridge/subsidence, high shortwave, warm 850hPa temperatures and dry lower troposphere align.
```

Inputs:

```text
500hPa height anomaly proxy
850hPa temperature
700hPa RH low/moderate
shortwave high
low cloud low
precipitation near zero
pressure pattern
weak wind
```

### 9.9 Specialist S6 — high-error tail prevention

Hypothesis:

```text
Some days are not clearly up/down but are high uncertainty; corrections should shrink and forecast distribution should widen.
```

Inputs:

```text
model_forecast_std
GEFS spread
IFS ENS spread
station disagreement
official-NWP contradiction
recent official abs error
missingness flags
MAM transition flag
```

Output:

```text
uncertainty widening;
router shrinkage;
no-trade or no-aggressive-correction flag.
```

This specialist may not improve point MAE directly but protects tails and trading decisions.

---

## 10. Concrete decision questions and how training answers them

### 10.1 Is official too hot or too cold?

Training target:

```text
official_residual = actual_hko_tmax - official_tmax
```

Outputs:

```text
predicted_median_residual
P(official_overforecast)
P(official_underforecast)
expected_correction_benefit
```

Decision:

```text
if predicted residual is negative, overforecast probability is high, and benefit gate passes:
    apply bounded negative correction
elif predicted residual is positive, underforecast probability is high, and benefit gate passes:
    apply bounded positive correction
else:
    abstain or trust router blend
```

### 10.2 Do official, GFS, GEFS, IFS, AI models and stations agree?

Construct:

```text
model_forecast_std
model_forecast_range
official_minus_gfs
official_minus_gefs
official_minus_ifs
gfs_minus_gefs
gfs_minus_ifs
fraction_experts_above_threshold
fraction_experts_below_threshold
station_disagreement_index
NWP_station_contradiction_score
```

Train how these affect each expert’s absolute error.

Example learned behavior:

```text
If official is warmest, GEFS spread is high, and coastal stations do not confirm inland heat:
    official expected error rises
    GEFS/station trust rises
    correction is shrunk if uncertainty is too high
```

### 10.3 Is this a marine-suppression day?

Train a detector using marine/onshore features and a correction model for negative residuals. The detector must learn from historical residual behavior, not just meteorological intuition.

### 10.4 Is this a weak-wind heat-buildup day?

Train a detector using weak wind, high shortwave, low cloud, station anomalies, hot-spell state and warm 850hPa structure. The correction model targets positive residuals.

### 10.5 Is this a MAM transition day?

Use calendar as a gate, but train actual transition intensity from target slopes, dewpoint change, pressure tendency, station disagreement, cloud/rain and official/NWP disagreement.

### 10.6 Is ensemble spread high?

Train monotonic relationship between ensemble spread and expected error:

```text
spread ↑ should generally not imply predicted uncertainty ↓
```

Use spread for:

```text
uncertainty;
correction shrinkage;
router confidence;
no-trade warnings.
```

### 10.7 Has this source been biased recently?

Maintain causal online residual states:

```text
EWMA residual half-lives 5/10/20/40
recent abs error
recent error volatility
recent over/underforecast streaks
```

Update only after settlement.

### 10.8 Should we correct or abstain?

Train benefit:

```text
benefit = abs(actual - anchor) - abs(actual - corrected)
```

Apply correction only when expected benefit exceeds threshold. Abstention must be a real option.

---

## 11. Training date ranges and partitions

### 11.1 Current recommended development partitions

| Component | Development data |
|---|---|
| Target-memory expert | 1884–2023, causal expanding windows, safe lag default T-2 |
| Official raw/residual expert | 2000–2023, using corrected near-continuous HKO forecast archive |
| Online official bias state | 2000–2023 walk-forward replay |
| Station-network expert | strict/proxy separated; pre-2024 only for development |
| GFS MOS | 2021-03-23 through 2023-12-31 |
| GEFS MOS | 2020-10-03 through 2023-12-31; common router overlap with GFS/official from 2021-03-23 |
| R0 official/target/station router | 2000–2023 |
| R1 modern GFS/GEFS router | 2021-03-23 through 2023-12-31 |
| MAM official specialist | 2000–2023 MAM seasons |
| MAM NWP specialist | 2021–2023 MAM, heavily shrunk |
| Marine/heat NWP specialists | 2021-03-23 through 2023-12-31 |
| IFS adapter | 2024+ only after sealed protocol opening |
| AI model adapters | 2024/2025+ only after sealed protocol opening or shadow/live scoring |
| ARWF adapter | prospective after enough exact-vintage history accumulates |
| CWA WRF adapter | prospective/live only initially |

### 11.2 Sealed usage

Features can be stored for 2024–2026, but target outcomes remain sealed until the system is frozen.

Recommended sequence:

```text
1. Develop R0/R1 on pre-2024 only.
2. Freeze feature list, model classes, hyperparameters, router method and specialist gates.
3. Open 2024 once for validation.
4. If validation passes, refit according to frozen rules including 2024.
5. Open 2025 once for final historical test.
6. Treat 2026 as prospective/live replay.
```

---

## 12. Model classes and training discipline

### 12.1 Allowed first-generation models

Use interpretable and regularized models before complex models.

```text
ridge / elastic net regression
Huber regression
quantile regression
monotonic GAM or isotonic components for uncertainty
small LightGBM/CatBoost with strict depth and early stopping
random forest only as diagnostic, not default production
```

### 12.2 Hyperparameter discipline

All hyperparameters must be selected inside historical walk-forward folds.

Prohibited:

```text
tuning on 2024/2025 before sealed opening;
choosing correction caps after seeing holdout failure;
fitting scalers on full history;
using full-history target encodings;
using target-date outcome-derived flags.
```

### 12.3 Fold-local preprocessing

Within each training fold:

```text
fit imputation only on training period;
fit scaling only on training period;
fit categorical encoders only on training period;
fit text vocabulary only on training period;
fit PCA/graph modes only on training period;
fit climatology only on prior dates;
fit station normals only on prior dates.
```

Apply the trained transformer to the test fold.

---

## 13. Final system forecast formula

At a high level:

```text
expert_forecasts = {
    official_corrected,
    target_memory,
    station_microclimate,
    gfs_mos,
    gefs_median,
    ifs_mos_when_allowed,
    ifsenfo_median_when_allowed,
    ai_challengers_when_allowed,
    arwf_when_trained,
    cwawrf_when_trained
}
```

The router produces weights:

```text
w_e,t for each available expert
```

Core blend:

```text
core_blend_tmax_t = sum_e w_e,t * expert_forecast_e,t
```

Specialists propose corrections:

```text
marine_correction_t
heat_buildup_correction_t
MAM_transition_correction_t
cloud_suppression_correction_t
ridge_heating_correction_t
```

Each correction is applied only if its benefit gate passes.

Final point forecast:

```text
final_point_tmax_t = conditional_median(
    core_blend_tmax_t
    + sum(active_specialist_corrections)
)
```

The conditional median is preferred for MAE minimization.

---

## 14. Distributional and trading layer

The final system must produce both point and probabilistic forecasts.

Distribution outputs:

```text
p10_tmax_c
p25_tmax_c
p50_tmax_c
p75_tmax_c
p90_tmax_c
expected_abs_error_c
prob_tmax_ge_30_5
prob_tmax_ge_31_5
prob_tmax_ge_32_5
prob_tmax_ge_33_5
prob_tmax_ge_34_5
confidence_state
no_trade_flag
```

Distribution inputs:

```text
expert spread;
GEFS spread;
IFS ENS spread;
model disagreement;
station disagreement;
recent official error volatility;
MAM/high-error regime flags;
missingness.
```

Metrics:

```text
MAE for point forecast;
RMSE;
bias;
median absolute error;
P90/P95 absolute error;
pinball loss;
CRPS if distribution available;
Brier score for thresholds;
log loss for threshold probabilities;
calibration curves;
sharpness vs calibration.
```

For Polymarket-style decisions, trade only when:

```text
model probability - market implied probability > required edge threshold
AND calibration for that threshold/regime is acceptable
AND uncertainty/no-trade gate does not block
AND liquidity/slippage/risk limits pass
```

This document defines forecasting strategy, not bankroll strategy.

---

## 15. Validation and scoreboards

### 15.1 Required baselines

Every candidate must be compared to:

```text
official raw H24N anchor;
target-memory baseline;
official online residual memory;
GFS direct Tmax;
GFS MOS;
GEFS raw median;
GEFS calibrated median;
static expert blend;
dynamic router blend;
router + specialists;
distributional P50 final system.
```

Use identical rows for each comparison.

### 15.2 Required metrics

```text
MAE
RMSE
bias
median absolute error
P75/P90/P95 absolute error
hot underforecast MAE
cold overforecast MAE
large-error frequency >1.0°C, >1.5°C, >2.0°C
seasonal MAE
monthly MAE
yearly MAE
source-era MAE
MAM MAE
JJA MAE
high-uncertainty MAE
marine-suppression slice MAE
weak-wind heat slice MAE
cloud/rain suppression slice MAE
```

### 15.3 Strict versus proxy scoreboards

Maintain separate leaderboards:

```text
STRICT_DEPLOYABLE:
    only sources with proven or conservatively justified T-24 eligibility.

RESEARCH_PROXY:
    includes diagnostic/proxy sources like historical ISD/IGRA if release timing is unproven.

LIVE_SHADOW:
    recent/prospective sources like ARWF/CWA WRF before sufficient history.
```

Never mix strict and proxy scores.

### 15.4 Negative controls

Run:

```text
shuffled target dates;
lag-shifted NWP features;
post-cutoff source injection test;
outcome-derived feature scan;
future-normalization scan;
same-row residual flag scan;
```

If a model still performs suspiciously well under negative controls, quarantine it.

---

## 16. Implementation sequence

### Phase 0 — freeze and audit sources

Deliverables:

```text
source_inventory.md
hko_forecast_anchor_audit.csv
gribs_inventory_audit.csv
station_data_audit.csv
leakage_contract.md
sealed_partition_manifest.json
```

Actions:

```text
verify HKO official forecast table counts;
verify H24N anchor extraction;
verify GribStream full-run source filter;
verify GribStream 6-hour safety filter;
verify target label availability;
verify station data eligibility;
verify no 2024+ target outcomes are visible to development jobs.
```

### Phase 1 — canonical snapshot builder

Build:

```text
modeling.h24n_snapshot
modeling.h24n_source_availability
modeling.h24n_anchor_official
modeling.h24n_nwp_daily_features
modeling.h24n_target_memory_features
modeling.h24n_station_features
```

Output:

```text
snapshot_coverage_report.md
snapshot_leakage_audit.md
```

### Phase 2 — feature engineering

Build feature families:

```text
official features;
target-memory features;
station-network features;
NWP features;
ensemble features;
cross-source contradiction features;
regime candidate features;
online residual-state features.
```

Output:

```text
feature_dictionary.csv
feature_availability_matrix.csv
feature_lineage_manifest.json
feature_quality_report.md
```

### Phase 3 — train individual experts with OOF predictions

Train:

```text
official residual expert;
target-memory expert;
station expert;
GFS MOS expert;
GEFS MOS expert;
IFS/AI shadow experts when allowed;
```

Output:

```text
oof_expert_predictions.parquet
expert_scoreboard.csv
expert_slice_metrics.csv
expert_training_manifests/
```

### Phase 4 — train expected-error router

Build router training table and train expected-error models.

Output:

```text
router_training_table.parquet
expected_error_models/
router_oof_predictions.parquet
router_weight_audit.csv
router_scoreboard.csv
```

### Phase 5 — train specialists and benefit gates

Train specialist detectors, corrections and gates.

Output:

```text
specialist_registry.csv
specialist_oof_predictions.parquet
specialist_activation_audit.csv
specialist_scoreboard.csv
specialist_no_harm_report.md
```

### Phase 6 — distributional layer

Train quantile/uncertainty calibration.

Output:

```text
distribution_oof_predictions.parquet
threshold_probability_scoreboard.csv
calibration_report.md
```

### Phase 7 — championship tournament

Compare all system variants.

Output:

```text
system_scoreboard.csv
ablation_matrix.csv
slice_metrics.csv
tail_metrics.csv
negative_control_report.md
frozen_candidate_manifest.json
```

### Phase 8 — sealed validation

After freeze:

```text
open 2024 once;
score frozen candidate;
if passes, refit according to frozen rules;
open 2025 once;
score final historical test;
record 2026 prospective replay only as live/prospective.
```

Output:

```text
2024_validation_scoreboard.csv
2025_final_test_scoreboard.csv
2026_ytd_prospective_scoreboard.csv
sealed_validation_report.md
contamination_audit.md
```

### Phase 9 — live daily inference

Daily pipeline:

```text
1. Freeze inputs at T-1 14:45 HKT.
2. Extract latest official anchor.
3. Extract eligible NWP features.
4. Extract station/live features.
5. Compute target-memory features.
6. Load online residual states.
7. Generate expert forecasts.
8. Generate router weights.
9. Evaluate specialists.
10. Generate distribution and final point forecast.
11. Save full audit log.
12. After settlement, score forecast and update online states.
```

Output per day:

```text
live_prediction_<target_date>.json
live_feature_snapshot_<target_date>.parquet
live_router_audit_<target_date>.csv
live_specialist_audit_<target_date>.csv
post_settlement_score_<target_date>.json
```

---

## 17. Database design

### 17.1 Core schemas

```text
source_raw
source_normalized
feature_store
modeling
validation
live_predictions
audit
quarantine
research_proxy
sealed
```

### 17.2 Key tables

```text
source_normalized.hko_historical_forecast_anchor
source_normalized.hko_forecast_revision
source_normalized.hko_target_tmax_label
source_normalized.nwp_tactical_forecast_safe_view
feature_store.h24n_snapshot
feature_store.h24n_official_features
feature_store.h24n_target_memory_features
feature_store.h24n_station_features
feature_store.h24n_nwp_features
feature_store.h24n_cross_source_features
modeling.oof_expert_prediction
modeling.router_training_table
modeling.router_weight_oof
modeling.specialist_oof
modeling.final_system_oof
validation.scoreboard
validation.slice_metrics
live_predictions.daily_prediction
live_predictions.post_settlement_score
audit.leakage_audit_event
quarantine.rejected_feature_row
```

### 17.3 Feature provenance

Every feature must carry:

```text
feature_name
source_family
source_table
source_columns
valid_time_basis
available_at_basis
cutoff_rule
lag_rule
fold_fit_requirement
strict_or_proxy_status
```

---

## 18. Hard denylist

Never use these as model inputs:

```text
actual_hko_tmax_c(T)
official_residual(T)
official_overforecast_c(T)
official_underforecast_c(T)
hot_day_underforecast_flag(T)
cold_day_overforecast_flag(T)
absolute_official_error(T)
post-cutoff official revision
post-cutoff NWP run
finalized target-day daily climate value
finalized T-1 daily Tmax unless known before cutoff
full-history fitted scaler/normalizer
future-year climatology
retrospective TC best track as live predictor
IGRA/HKO daily/marine same-day diagnostic values without timestamp proof
smoke-test GribStream rows not in full tactical scope
nbmoc/aigfspres/aigefssfc as Tmax sources under current inventory
```

---

## 19. Promotion ladder

A signal or model can be promoted only when it passes:

```text
1. Diagnostic signal exists.
2. Timestamp eligibility is proven or explicitly proxy-labelled.
3. Feature construction is causal.
4. OOF test improves relevant baseline on identical rows.
5. Improvement is stable across years/seasons/source states.
6. High-error tails do not worsen.
7. Simpler baselines are beaten.
8. Negative controls pass.
9. Artifact documentation is complete.
10. 2024+ confirmation remains sealed until formal opening.
```

---

## 20. Expected first system variants

Build and score these variants in order:

```text
V0 official_raw
V1 official_raw + online_residual_memory
V2 official_residual_model
V3 target_memory + official_residual blend
V4 GFS_MOS only
V5 GEFS_calibrated_median only
V6 official + GFS + GEFS static blend
V7 official + GFS + GEFS expected-error router
V8 V7 + station_microclimate expert
V9 V8 + specialists
V10 V9 + distributional conditional median
V11 IFS/AI challengers after sealed protocol allows
```

At each stage, compare on identical rows and preserve all negative results.

---

## 21. What success looks like

The first credible success is not immediately `0.45°C MAE`. The first credible success is:

```text
A strict, auditable OOF system that beats official raw on identical rows;
shows stable year/season/source lift;
does not rely on blocked or outcome-derived features;
improves high-error tails;
produces calibrated probabilities;
passes sealed 2024 and 2025 evaluation without re-tuning.
```

The path toward `0.45°C MAE` likely requires:

```text
near-continuous official archive residual learning;
GFS/GEFS MOS;
NWP/station disagreement routing;
IFS and AI challenger integration after sealed validation;
prospective ARWF/CWA WRF accumulation;
strong live high-frequency HKO features;
careful probability calibration;
continued error autopsy and specialist refinement.
```

---

## 22. Daily operational checklist

At `T-1 14:45 HKT`:

```text
1. Create immutable snapshot ID.
2. Load latest eligible official local min/max row.
3. Load all pre-cutoff official revisions.
4. Load safe target-memory features.
5. Load safe station/live features.
6. Load GribStream full tactical rows passing source and H24N safety filters.
7. Derive daily NWP features.
8. Load online residual states based only on settled dates.
9. Run each available expert.
10. Predict expected error for each expert.
11. Compute dynamic router weights.
12. Blend with static prior weights.
13. Apply availability masks and caps.
14. Evaluate specialist regime probabilities.
15. Apply only positive-benefit specialist corrections.
16. Produce point forecast and distribution.
17. Write audit log before decision.
18. After settlement, score forecast and update online states.
```

---

## 23. Final implementation command summary

The strategy to implement now is:

```text
Build a strict H24N forecast system centered on the near-continuous HKO official forecast archive, enhanced by GFS/GEFS NWP MOS, target-memory, station-network microclimate features, expected-error routing, validated specialists, and distributional calibration. Use pre-2024 for development, keep 2024/2025 sealed until freeze, and maintain strict/proxy/live scoreboards separately. All feature extraction must enforce exact T-24 availability, GribStream full-run scope filtering, source-specific leakage gates, and immutable audit logs.
```

The system should never trade trust for score. A smaller, strictly verified improvement is more valuable than a large leaky backtest.

---

## 24. Source reference notes

External references used to ground this implementation:

```text
HKO ARWF / Automatic Regional Weather Forecast notes:
- ARWF uses multiple NWP models, observation-based correction, and past-performance weighting.
- Product is updated around noon and midnight.

HKO Numerical Weather Prediction background:
- NWP simulates atmospheric evolution using equations involving wind, temperature, pressure and moisture.

GribStream API documentation and client notes:
- /runs returns model-run-level data.
- /timeseries returns best-available valid-time values and can be used with asOf, but asOf is not historical wall-clock availability proof.

Super Learner / cross-validated stacking literature:
- Candidate learners should generate cross-validated predictions, then combination weights are selected using out-of-fold risk.
```

Project-local references used:

```text
GRIBSTREAM_FETCHED_DATA_INVENTORY_20260626.md
Corrected public.hko_historical_forecasts_2000_2026 summary supplied by project owner
Previously generated dataset and experiment audits
```
