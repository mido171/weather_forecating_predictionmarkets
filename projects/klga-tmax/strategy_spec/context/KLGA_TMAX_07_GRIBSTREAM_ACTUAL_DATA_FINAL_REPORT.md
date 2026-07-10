# KLGA Tmax GribStream Actual Data Final Report

Generated: 2026-06-30  
Project: KLGA / NYC daily maximum temperature forecasting  
Database: `klga_tmax_research`  
Final authoritative job: `klga_t1245utc_tmax_thin_backfill_v1`  
Final feature profile: `TMAX_THIN_V1`  
Final cutoff: `T_1245UTC`

This document consolidates the GribStream planning and implementation notes under:

- `KLGA_TMAX_03_GRIBSTREAM_SINGLE_CUTOFF_BACKFILL_IMPLEMENTATION_DEEP_DIVE.md`
- `KLGA_TMAX_05_GRIBSTREAM_T1245UTC_BACKFILL_IMPLEMENTATION_DEEP_DIVE.md`
- `KLGA_TMAX_06_GRIBSTREAM_T1245UTC_TMAX_THIN_BACKFILL_IMPLEMENTATION_DEEP_DIVE.md`

It is the final operational description of the GribStream data that was actually fetched and persisted for the KLGA Tmax system. Earlier documents remain useful for history, but this report is the source of truth for the completed backfill.

This report was added as:

```text
weather_data_extraction/bootstrap/klga_tmax/strategy_spec/context/KLGA_TMAX_07_GRIBSTREAM_ACTUAL_DATA_FINAL_REPORT.md
```

## Reader Orientation

Read this document as the operational handoff for the data that exists now. It is not a proposal and it is not a forecast-skill result. The central question answered here is:

```text
What did we actually fetch from GribStream for KLGA Tmax, under what cutoff/run-time rules, over what date ranges, and where is it persisted?
```

The answer is based on the final chunk ledger, raw request ledger, availability table, feature table, and the consolidated Markdown context files listed above.

## Scope Boundaries

In scope:

- final GribStream backfill state for `klga_t1245utc_tmax_thin_backfill_v1`
- model-by-model date coverage
- raw GribStream selectors and request shapes
- cutoff and buffer semantics
- persisted run-time and valid-time lineage
- compact feature coverage
- known gaps and empty chunks
- DB tables and raw artifact locations used by the final job

Out of scope:

- proving that `T_1245UTC` is the best trading cutoff
- comparing Polymarket execution quality
- retraining the forecast model
- recalculating realized KLGA / Wunderground labels
- deleting earlier broad-run data
- changing the GribStream runner code

## Source-of-Truth Inputs

This report uses the following source-of-truth inputs:

| Source | Role |
|---|---|
| `audit.gribstream_backfill_chunks` | Final chunk status, model ranges, request JSON, request hashes, estimated credits, rows written |
| `bronze.source_requests` | Authenticated request status, response sizes, raw response hashes |
| `silver.availability_ledger` | Model run time, valid time, member, variable, forecast-hour availability evidence |
| `gold.feature_values` | Final compact feature rows used by modeling |
| `audit.gribstream_source_gaps` | Explicit missing/empty/member/selector gap evidence |
| Raw `.ndjson.gz` files | Compressed GribStream response bodies |
| `KLGA_TMAX_03...`, `KLGA_TMAX_05...`, `KLGA_TMAX_06...` | Historical implementation context consolidated into this final report |

## Requirements-to-Implementation Traceability

| Requirement | Final state |
|---|---|
| Use `T_1245UTC` cutoff | Met |
| Use HKG-style batched `/runs` requests | Met |
| Fetch Tmax-relevant data only | Met |
| Avoid broad full atomic persistence by default | Met |
| Preserve raw request/response evidence | Met |
| Persist model-ready compact features | Met |
| Persist availability metadata | Met |
| Track chunks and resume by request hash | Met |
| Record missing/empty coverage as explicit gaps | Met |
| Complete all configured live-relevant models through 2026-06-28 | Met, with provider-side gaps recorded |
| Exclude URMA from live cutoff backfill | Met |
| Document actual coverage and known issues | Met in this file |

## Change Inventory

This documentation change adds one final consolidated report:

```text
weather_data_extraction/bootstrap/klga_tmax/strategy_spec/context/KLGA_TMAX_07_GRIBSTREAM_ACTUAL_DATA_FINAL_REPORT.md
```

No runner code, database schema, data files, or prior context files were modified by this documentation consolidation.

## Executive Summary

The completed GribStream backfill is a **Tmax-first thin feature backfill**, not the earlier broad eight-variable atomic-row plan.

The final run fetched only the weather fields needed to support KLGA / NYC daily Tmax forecasting at the agreed target-day cutoff:

```text
cutoff_id: T_1245UTC
cutoff wall time: target date T at 12:45:00 UTC
summer New York display: T 08:45 America/New_York
summer Stockholm display: T 14:45 Europe/Stockholm
endpoint: POST /api/v2/{model}/runs
feature profile: TMAX_THIN_V1
persistence mode: gold_only
job_id: klga_t1245utc_tmax_thin_backfill_v1
```

The job finished successfully:

| Metric | Value |
|---|---:|
| Planned chunks | 438 |
| Terminal chunks | 438 |
| Completed chunks | 436 |
| Completed-empty chunks | 2 |
| Failed chunks | 0 |
| Open/running/blocked chunks | 0 |
| Target-date span | 2014-07-30 through 2026-06-28 |
| Estimated GribStream credits | 404,160 |
| Feature rows upserted | 1,528,742 |
| Availability rows upserted | 1,325,836 |
| Gap rows recorded by chunks | 419 |
| Authenticated source requests | 438 |
| HTTP 200 source requests | 438 |
| HTTP non-200 source requests | 0 |
| Raw response bytes | 305,330,424 |
| Runtime | 110.13 minutes |

The final run intentionally **did not** continue the earlier broad job that wrote millions of full atomic rows. The completed thin job persisted raw request/response evidence plus compact model-ready features. The broad atomic `silver.grib_forecast_values` path was not used for the final `TMAX_THIN_V1` run.

## What Superseded What

Three GribStream paths existed during the KLGA work:

1. `T_MINUS_1_2045UTC` `/timeseries` prior-day plan.
   - Purpose: leakage-safe day-ahead snapshot.
   - Status: superseded for the current final data build.
   - It remains a valid research cutoff, but it is not the final completed backfill described here.

2. `T_1245UTC` broad `/timeseries` plan.
   - Purpose: target-day cutoff with broad feature pull.
   - Status: superseded because it wrote too much data and was too slow for the actual modeling need.
   - The broad approach fetched unnecessary non-Tmax fields and produced excessive persistence volume.

3. `T_1245UTC` thin `/runs` job.
   - Purpose: target-day cutoff, HKG-style batched `/runs`, only Tmax-relevant fields.
   - Status: completed and authoritative.
   - Job ID: `klga_t1245utc_tmax_thin_backfill_v1`.

Only the third path is the final backfilled data contract.

## Plain-Language Storage Contract

The earlier notes sometimes used `bronze`, `silver`, and `gold`. The plain meaning is:

| Layer name | Plain meaning | Used by final thin run? |
|---|---|---|
| Raw request evidence | Exact authenticated GribStream request metadata plus raw compressed response files | Yes |
| Parsed audit/availability rows | Row-level evidence of model, run time, valid time, member, variable, and acquisition availability | Yes |
| Model-ready feature rows | Compact derived features for forecasting and market-bucket probability work | Yes |
| Full atomic forecast value table | One row per coordinate, valid time, variable, member, and forecast value | No by default for final thin run |

The final run used `gold_only` persistence. That means the model-ready feature table is the intended direct modeling surface, while raw files and availability rows preserve auditability.

## Final Cutoff And Buffer Strategy

### Canonical Cutoff

For every target date `T`, the final job uses:

```text
cutoff_utc = T at 12:45:00 UTC
cutoff_id = T_1245UTC
```

This is a target-day morning cutoff for New York summer markets:

```text
New York: T 08:45 EDT
Stockholm: T 14:45 CEST
UTC: T 12:45 UTC
```

### What The Buffer Means

The buffer is a **model-run eligibility guard**, not part of the temperature calculation.

The rule is:

```text
latest_allowed_model_run_time = cutoff_utc - model_specific_buffer
selected_model_run_time = latest real model cycle <= latest_allowed_model_run_time
forecast_valid_time = selected_model_run_time + forecast_lead_time
```

Example:

```text
cutoff_utc = 12:45 UTC
buffer = 4 hours
latest_allowed_model_run_time = 08:45 UTC
selected 6-hourly model run = 06:00 UTC
```

The buffer does **not** mean "take model run time plus four hours to calculate Tmax." It means "do not let the backtest use model runs initialized later than the cutoff-safe availability threshold."

Tmax features are then calculated from forecast valid times inside the intended valid-time window.

### Persisted Run-Time Fields

There is no literal `model_run_time` column named exactly that. The same concept is persisted in:

- `audit.gribstream_backfill_chunks.request_json.timesList`
- `silver.availability_ledger.run_time_utc`
- `gold.feature_values.source_latest_run_time_utc`
- `gold.feature_values.source_trace_json`

Every final `TMAX_THIN_V1` feature row has non-null:

- `source_latest_run_time_utc`
- `source_latest_valid_time_utc`
- `source_age_hours`
- `max_source_available_at_utc`
- `source_trace_json`

## Model Run Cycles Actually Used

For target date `T`, the final thin job requested these run cycles:

| Model | Requested model run time for target date T | Reason |
|---|---|---|
| `hrrr` | T 10:00 UTC | Latest safe HRRR cycle before target-day cutoff under HRRR buffer |
| `rap` | T 08:00 UTC | Latest safe RAP cycle under buffer and useful lead coverage |
| `gfs` | T 06:00 UTC | Latest safe 6-hourly GFS cycle before the 08:45 threshold |
| `nbm` | T 11:00 UTC | Latest safe hourly NBM cycle before cutoff |
| `gefsatmosmean` | T 06:00 UTC | Latest safe GEFS mean cycle before the 08:45 threshold |
| `gefsatmos` | T 06:00 UTC | Latest safe GEFS member cycle before the 08:45 threshold |
| `ifsoper` | T 00:00 UTC | Latest safe ECMWF deterministic cycle before target-day cutoff |
| `ifsenfo` | T 00:00 UTC | Latest safe ECMWF ensemble cycle before target-day cutoff |
| `aifsoper` | T 00:00 UTC | Latest safe AIFS deterministic cycle before target-day cutoff |
| `aifsenfo` | T 00:00 UTC | Latest safe AIFS ensemble cycle before target-day cutoff |
| `aigefssfc` | T 06:00 UTC | Latest safe AI-GEFS surface cycle before the 08:45 threshold |
| `aigfssfc` | T 06:00 UTC | Latest safe AI-GFS surface cycle before the 08:45 threshold |
| `nbmqmd` | T 06:00 UTC | QMD max-18h product shape used in final job |
| `rtma` | T 11:00 UTC | Latest safe current-state analysis near target-day cutoff |

All conceptual run times above are UTC. Some database clients display `timestamptz` values in the session time zone; the source requests and this contract use UTC.

## Architecture and Control Flow

The final data flow is:

```text
target date T
  -> compute T_1245UTC cutoff
  -> apply model-specific buffer
  -> choose latest eligible model run
  -> batch exact run times into /runs request
  -> persist source request metadata
  -> persist compressed raw response
  -> parse availability evidence
  -> derive TMAX_THIN_V1 features
  -> record gaps and terminal chunk status
```

The control-flow constraint that matters most is that model run eligibility is decided before the request is sent. The backfill does not ask GribStream for an arbitrary "latest" run at query time. It builds an exact `timesList`, which makes the data reproducible.

## Endpoint Strategy

The final job used HKG-style batched `/runs` calls:

```text
POST https://gribstream.com/api/v2/{model}/runs
```

Request shape:

- exact `timesList` model-run times
- sparse `minLeadTime` / `maxLeadTime`
- only the selectors needed for Tmax or current-state support
- one worker
- large chunks
- request hashing for resume safety
- compressed raw response persistence

This replaced the earlier slow `/timeseries` per-target-date approach. The reason is that GribStream support explicitly indicated that the API is built for fewer larger requests instead of many tiny requests. The final `/runs` shape follows that guidance.

## Coordinate Strategy

Two coordinate tiers were used:

| Tier | Meaning | Used for |
|---|---|---|
| KLGA exact / Tier A | One point centered on KLGA | Ensembles, deterministic global synoptic models, QMD, RTMA |
| Tier B | Ten KLGA pseudo-points | HRRR, RAP, GFS, NBM temperature-curve fallback |

The final job deliberately kept Tier B only where spatial-gradient features matter. Ensembles and global deterministic models use one KLGA point because requesting more points would not improve the first-pass thin Tmax feature contract enough to justify row volume.

## Overall Model Coverage

| Model | Target from | Target through | Chunks | Completed | Completed empty | Failed | Estimated credits | Feature rows | Availability rows | Chunk gaps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `nbmqmd` | 2026-01-31 | 2026-06-28 | 1 | 1 | 0 | 0 | 3,129 | 13,298 | 3,129 | 0 |
| `nbm` | 2020-09-29 | 2026-06-28 | 34 | 34 | 0 | 0 | 21,672 | 161,161 | 209,270 | 20 |
| `gfs` | 2021-03-22 | 2026-06-28 | 32 | 32 | 0 | 0 | 19,870 | 148,148 | 192,220 | 30 |
| `rap` | 2021-02-22 | 2026-06-28 | 32 | 32 | 0 | 0 | 20,212 | 150,381 | 195,280 | 10 |
| `hrrr` | 2014-07-30 | 2026-06-28 | 71 | 71 | 0 | 0 | 44,946 | 334,719 | 432,060 | 70 |
| `aigfssfc` | 2026-04-16 | 2026-06-28 | 2 | 2 | 0 | 0 | 148 | 140 | 140 | 2 |
| `aigefssfc` | 2025-06-01 | 2026-06-28 | 18 | 16 | 2 | 0 | 24,366 | 14,550 | 24,180 | 124 |
| `gefsatmosmean` | 2020-10-01 | 2026-06-28 | 24 | 24 | 0 | 0 | 4,194 | 4,194 | 4,194 | 0 |
| `ifsenfo` | 2024-03-01 | 2026-06-28 | 56 | 56 | 0 | 0 | 86,700 | 198,806 | 86,606 | 6 |
| `aifsenfo` | 2025-07-02 | 2026-06-28 | 24 | 24 | 0 | 0 | 36,924 | 84,558 | 36,774 | 150 |
| `ifsoper` | 2024-02-28 | 2026-06-28 | 20 | 20 | 0 | 0 | 1,704 | 1,702 | 1,702 | 2 |
| `aifsoper` | 2025-02-25 | 2026-06-28 | 12 | 12 | 0 | 0 | 978 | 976 | 976 | 2 |
| `rtma` | 2018-01-01 | 2026-06-28 | 18 | 18 | 0 | 0 | 9,303 | 9,291 | 9,291 | 3 |
| `gefsatmos` | 2020-10-01 | 2026-06-28 | 94 | 94 | 0 | 0 | 130,014 | 406,818 | 130,014 | 0 |

The `aigefssfc` model is the only model with completed-empty chunks in the final job. They are terminal and explicitly recorded, not silent failures.

## Raw Attribute Coverage By Model

This section describes exactly what was requested from GribStream. It is based on the persisted `request_json` in `audit.gribstream_backfill_chunks`.

### HRRR

| Field | Value |
|---|---|
| Model | `hrrr` |
| Endpoint | `/api/v2/hrrr/runs` |
| Target range | 2014-07-30 through 2026-06-28 |
| Run time | T 10:00 UTC |
| Lead window | `6h` through `15h` |
| Coordinates | Tier B, 10 KLGA pseudo-points |
| Members | deterministic / no member list |
| Variables | `TMP`, level `2 m above ground`, alias `temperature_2m` |
| Purpose | Target-day peak-window 2m temperature curve and spatial-gradient features |

The HRRR feature output includes Tmax proxy, mean/median/spread features, integer-Fahrenheit exceedance probabilities, generic Polymarket bucket probabilities, member count, and Tier B spatial-gradient-derived fields.

### RAP

| Field | Value |
|---|---|
| Model | `rap` |
| Endpoint | `/api/v2/rap/runs` |
| Target range | 2021-02-22 through 2026-06-28 |
| Run time | T 08:00 UTC |
| Lead window | `8h` through `18h` |
| Coordinates | Tier B, 10 KLGA pseudo-points |
| Members | deterministic / no member list |
| Variables | `TMP`, level `2 m above ground`, alias `temperature_2m` |
| Purpose | Target-day peak-window 2m temperature curve and spatial-gradient features |

### GFS

| Field | Value |
|---|---|
| Model | `gfs` |
| Endpoint | `/api/v2/gfs/runs` |
| Target range | 2021-03-22 through 2026-06-28 |
| Run time | T 06:00 UTC |
| Lead window | normally `10h` through `19h` in request chunks |
| Coordinates | Tier B, 10 KLGA pseudo-points |
| Members | deterministic / no member list |
| Variables | `TMP`, level `2 m above ground`, alias `temperature_2m` |
| Purpose | Target-day peak-window 2m temperature curve and spatial-gradient features |

For local-day windows, lead-hour coverage can vary around daylight saving transitions. The request contract stores exact `timesList`, `minLeadTime`, and `maxLeadTime` per chunk.

### NBM

| Field | Value |
|---|---|
| Model | `nbm` |
| Endpoint | `/api/v2/nbm/runs` |
| Target range | 2020-09-29 through 2026-06-28 |
| Run time | T 11:00 UTC |
| Lead window | `5h` through `15h` |
| Coordinates | Tier B, 10 KLGA pseudo-points |
| Members | deterministic / no member list |
| Variables actually fetched | `TMP`, level `2 m above ground`, alias `temperature_2m` |
| Purpose | NBM target-day peak-window temperature curve |

The original thin plan preferred native NBM Tmax where possible, but the final persisted request shape uses the 2m-temperature peak-window fallback. That is why NBM has Tier B coordinates and 2m TMP rather than only a one-point native `TMAX` selector.

### RTMA

| Field | Value |
|---|---|
| Model | `rtma` |
| Endpoint | `/api/v2/rtma/runs` |
| Target range | 2018-01-01 through 2026-06-28 |
| Run time | T 11:00 UTC |
| Lead window | `0h` through `0h` |
| Coordinates | KLGA exact / one point |
| Members | analysis / no member list |
| Variables | `TMP` 2m, `DPT` 2m, `WIND` 10m |
| Purpose | Current-state temperature, dewpoint, and wind at the cutoff-safe analysis time |

RTMA is not a future Tmax model in this feature contract. It is current-state support for the target-day morning forecast.

### GEFS Atmos Mean

| Field | Value |
|---|---|
| Model | `gefsatmosmean` |
| Endpoint | `/api/v2/gefsatmosmean/runs` |
| Target range | 2020-10-01 through 2026-06-28 |
| Run time | T 06:00 UTC |
| Lead windows | Synoptic split chunks covering target-day peak support |
| Coordinates | KLGA exact / one point |
| Members | ensemble mean / implicit member `0` in feature trace |
| Variables | `TMP`, level `2 m above ground`, info `ens mean`, alias `temperature_2m` |
| Purpose | GEFS mean temperature at synoptic valid times |

Feature names include valid-time-specific outputs such as `grib_gefsatmosmean_klga_core_valid_18z_tmp_2m_f`.

### GEFS Atmos Members

| Field | Value |
|---|---|
| Model | `gefsatmos` |
| Endpoint | `/api/v2/gefsatmos/runs` |
| Target range | 2020-10-01 through 2026-06-28 |
| Run time | T 06:00 UTC |
| Lead windows | Synoptic split chunks covering target-day peak support |
| Coordinates | KLGA exact / one point |
| Members | 31 members, `0` through `30` |
| Variables | `TMP`, level `2 m above ground`, alias `temperature_2m` |
| Purpose | Ensemble Tmax proxy distribution, bucket probabilities, and exceedance probabilities |

The GEFS member backfill is the largest feature producer in the final run: 406,818 feature rows.

### IFS Deterministic

| Field | Value |
|---|---|
| Model | `ifsoper` |
| Endpoint | `/api/v2/ifsoper/runs` |
| Target range | 2024-02-28 through 2026-06-28 |
| Run time | T 00:00 UTC |
| Lead windows | Synoptic split chunks, including `18h` and `24h` support |
| Coordinates | KLGA exact / one point |
| Members | deterministic / no member list |
| Variables | `2t`, level `sfc`, alias `temperature_2m` |
| Purpose | ECMWF deterministic 2m temperature support at target-day synoptic valid times |

### IFS Ensemble

| Field | Value |
|---|---|
| Model | `ifsenfo` |
| Endpoint | `/api/v2/ifsenfo/runs` |
| Target range | 2024-03-01 through 2026-06-28 |
| Run time | T 00:00 UTC |
| Lead windows | Synoptic split chunks, including `18h` and `24h` support |
| Coordinates | KLGA exact / one point |
| Members | 51 members, `0` through `50` |
| Variables | `2t`, level `sfc`, alias `temperature_2m` |
| Purpose | ECMWF ensemble Tmax proxy distribution, bucket probabilities, and exceedance probabilities |

### AIFS Deterministic

| Field | Value |
|---|---|
| Model | `aifsoper` |
| Endpoint | `/api/v2/aifsoper/runs` |
| Target range | 2025-02-25 through 2026-06-28 |
| Run time | T 00:00 UTC |
| Lead windows | Synoptic split chunks, including `18h` and `24h` support |
| Coordinates | KLGA exact / one point |
| Members | deterministic / no member list |
| Variables | `2t`, level `sfc`, alias `temperature_2m` |
| Purpose | AIFS deterministic temperature support at target-day synoptic valid times |

### AIFS Ensemble

| Field | Value |
|---|---|
| Model | `aifsenfo` |
| Endpoint | `/api/v2/aifsenfo/runs` |
| Target range | 2025-07-02 through 2026-06-28 |
| Run time | T 00:00 UTC |
| Lead windows | Synoptic split chunks, including `18h` and `24h` support |
| Coordinates | KLGA exact / one point |
| Members | 51 members |
| Variables | `2t`, level `sfc`, alias `temperature_2m` |
| Purpose | AIFS ensemble Tmax proxy distribution, bucket probabilities, and exceedance probabilities |

### AI-GEFS Surface

| Field | Value |
|---|---|
| Model | `aigefssfc` |
| Endpoint | `/api/v2/aigefssfc/runs` |
| Target range | 2025-06-01 through 2026-06-28 |
| Run time | T 06:00 UTC |
| Lead windows | Synoptic split chunks covering target-day peak support |
| Coordinates | KLGA exact / one point |
| Members | 31 members |
| Variables | `TMP`, level `2 m above ground`, alias `temperature_2m` |
| Purpose | AI-GEFS ensemble Tmax proxy distribution and probabilities |

Two `aigefssfc` chunks were completed-empty and are preserved as terminal empty evidence.

### AI-GFS Surface

| Field | Value |
|---|---|
| Model | `aigfssfc` |
| Endpoint | `/api/v2/aigfssfc/runs` |
| Target range | 2026-04-16 through 2026-06-28 |
| Run time | T 06:00 UTC |
| Lead windows | Synoptic split chunks covering target-day peak support |
| Coordinates | KLGA exact / one point |
| Members | deterministic / no member list |
| Variables | `TMP`, level `2 m above ground`, alias `temperature_2m` |
| Purpose | AI-GFS deterministic temperature support at target-day synoptic valid times |

### NBMQMD

| Field | Value |
|---|---|
| Model | `nbmqmd` |
| Endpoint | `/api/v2/nbmqmd/runs` |
| Target range | 2026-01-31 through 2026-06-28 |
| Run time | T 06:00 UTC |
| Lead window | `24h` through `24h` |
| Coordinates | KLGA exact / one point |
| Members | deterministic percentile product / no member list |
| Variables | 21 `TMP` max-18h percentile selectors |
| Purpose | Native probabilistic max-18h temperature curve |

The requested percentile aliases are:

```text
tmp_max18_p01
tmp_max18_p05
tmp_max18_p10
tmp_max18_p15
tmp_max18_p20
tmp_max18_p25
tmp_max18_p30
tmp_max18_p35
tmp_max18_p40
tmp_max18_p45
tmp_max18_p50
tmp_max18_p55
tmp_max18_p60
tmp_max18_p65
tmp_max18_p70
tmp_max18_p75
tmp_max18_p80
tmp_max18_p85
tmp_max18_p90
tmp_max18_p95
tmp_max18_p99
```

Important audit note: the persisted request JSON shows `tmp_max18_p05` with `info = "15% level | max-18h"`. That is an anomaly in the selector metadata and should be checked before treating `p05` as a true 5th percentile. The row is preserved as fetched; it was not silently corrected.

## Feature Output Contract

All final features use:

```text
feature_build_version = TMAX_THIN_V1
feature_family = gribstream_tmax_thin
```

Feature names follow the general shape:

```text
grib_{model}_{point_or_scope}_{feature_name}
```

Examples:

```text
grib_hrrr_klga_core_tmax_proxy_mean_f
grib_hrrr_klga_core_tmax_proxy_std_f
grib_hrrr_klga_core_prob_tmax_ge_90f
grib_hrrr_klga_core_generic_bucket_prob_85_89
grib_rtma_klga_core_current_tmp_2m_f
grib_rtma_klga_core_current_dewpoint_2m_f
grib_rtma_klga_core_current_wind_speed_10m_mph
grib_gefsatmosmean_klga_core_valid_18z_tmp_2m_f
```

### Feature Families By Model

| Model | Feature rows | Distinct feature names | Main feature types |
|---|---:|---:|---|
| `gefsatmos` | 406,818 | 194 | member Tmax proxies, ensemble distribution, Fahrenheit exceedance probabilities, bucket probabilities |
| `hrrr` | 334,719 | 77 | deterministic Tmax proxy, spatial-gradient support, bucket probabilities, Fahrenheit exceedance probabilities |
| `ifsenfo` | 198,806 | 234 | ECMWF ensemble member/distribution/probability features |
| `nbm` | 161,161 | 77 | deterministic peak-window temperature curve features and probabilities |
| `rap` | 150,381 | 77 | deterministic peak-window temperature curve features and probabilities |
| `gfs` | 148,148 | 77 | deterministic peak-window temperature curve features and probabilities |
| `aifsenfo` | 84,558 | 234 | AIFS ensemble member/distribution/probability features |
| `aigefssfc` | 14,550 | 194 | AI-GEFS ensemble member/distribution/probability features |
| `nbmqmd` | 13,298 | 90 | percentile curve, probability curve, bucket probabilities |
| `rtma` | 9,291 | 3 | current-state temperature, dewpoint, wind |
| `gefsatmosmean` | 4,194 | 2 | synoptic 2m temperature values |
| `ifsoper` | 1,702 | 2 | synoptic 2m temperature values |
| `aifsoper` | 976 | 2 | synoptic 2m temperature values |
| `aigfssfc` | 140 | 2 | synoptic 2m temperature values |

The final feature table has 1,265 distinct feature names.

## Availability And Lineage Contract

The final job wrote availability evidence to:

```text
silver.availability_ledger
```

For each model row where GribStream returned data, the availability ledger records:

- source/provider name
- station or point identity
- model name
- run time
- valid time
- forecast hour
- member
- variable name
- acquisition time
- effective availability time
- source record link

The final model-ready feature rows store source lineage in:

```text
gold.feature_values.source_trace_json
```

The trace JSON includes fields such as:

- `model_id`
- `endpoint_type`
- `request_sha256`
- `feature_profile`
- `persistence_mode`
- `raw_storage_uri`
- `source_request_id`
- `source_record_id`
- `selector_aliases`
- `members`
- `grid_point_ids`
- `raw_row_hashes_sample`

This is how a model feature can be traced back to the raw GribStream response.

## Raw Files And Request Evidence

Every final chunk has:

- a request hash
- request JSON
- source request ID
- raw storage URI
- HTTP status
- compressed raw response body

Final request status:

| Request metric | Value |
|---|---:|
| Source requests joined through final chunks | 438 |
| HTTP 200 | 438 |
| HTTP non-200 | 0 |
| Raw response bytes | 305,330,424 |

Raw files are stored under the KLGA GribStream artifact tree:

```text
C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\klga_tmax\implementation\artifacts\klga_tmax\gribstream\raw\runs
```

Each raw URI is recorded in both chunk state and feature trace metadata.

## Gaps And Empty Coverage

The final run had no failed chunks and no unresolved running chunks.

It did record gaps, which is expected for historical weather pulls. Gaps are not necessarily code failures; they can represent unavailable members, missing valid-time coverage, empty provider responses, or selector/date combinations not present in GribStream.

Chunk-level gap counts by model:

| Model | Gap rows recorded by chunk |
|---|---:|
| `aifsenfo` | 150 |
| `aigefssfc` | 124 |
| `hrrr` | 70 |
| `gfs` | 30 |
| `nbm` | 20 |
| `rap` | 10 |
| `ifsenfo` | 6 |
| `rtma` | 3 |
| `ifsoper` | 2 |
| `aifsoper` | 2 |
| `aigfssfc` | 2 |
| `gefsatmos` | 0 |
| `gefsatmosmean` | 0 |
| `nbmqmd` | 0 |

Two chunks were completed-empty, both under `aigefssfc`. They are terminal and preserved in the chunk ledger.

## What Was Not Fetched

The final thin job intentionally did not fetch:

- pressure levels
- full 0-72h horizons
- all-run archival pulls
- broad non-Tmax weather fields for deterministic models
- broad full atomic row persistence
- URMA retrospective support
- previous `T_MINUS_1_2045UTC` production-equivalent history

URMA was excluded because it is retrospective-only and is not live evidence at the target-day cutoff. It can still be used in separate label/verification or retrospective diagnostic work if explicitly needed.

## Credit And Runtime Outcome

The final thin plan estimated roughly 387K-404K credits. The completed job ended at the top of that range:

```text
estimated credits = 404,160
```

This is materially smaller than the original broad `T_1245UTC` plan, which was around 1.15M estimated credits before further broad-row persistence costs. The reduction came from:

- only fetching Tmax-relevant fields
- using one point for ensembles and global synoptic models
- using Tier B only where spatial gradients are needed
- batching with `/runs`
- persisting compact features instead of writing full atomic rows

Runtime was about 110 minutes, from:

```text
started_utc: 2026-06-29 21:57:53 UTC
finished_utc: 2026-06-29 23:48:01 UTC
```

## File-by-File Deep Dive

### `KLGA_TMAX_03_GRIBSTREAM_SINGLE_CUTOFF_BACKFILL_IMPLEMENTATION_DEEP_DIVE.md`

This file documents the earlier `T_MINUS_1_2045UTC` `/timeseries` approach. It is useful for understanding the original prior-day cutoff and model-availability buffer reasoning. It is not the final completed data contract.

### `KLGA_TMAX_05_GRIBSTREAM_T1245UTC_BACKFILL_IMPLEMENTATION_DEEP_DIVE.md`

This file documents the intermediate broad `T_1245UTC` `/timeseries` implementation and pilot direction. It is useful for understanding why target-day `T_1245UTC` became the agreed cutoff. It is superseded by the thin `/runs` job for actual final data coverage.

### `KLGA_TMAX_06_GRIBSTREAM_T1245UTC_TMAX_THIN_BACKFILL_IMPLEMENTATION_DEEP_DIVE.md`

This file documents the final `TMAX_THIN_V1` design and completed run. It is the closest predecessor to this report. This report consolidates that implementation evidence with the refreshed DB-backed coverage tables.

### `KLGA_TMAX_07_GRIBSTREAM_ACTUAL_DATA_FINAL_REPORT.md`

This file is the final consolidated report. It exists to prevent future confusion between the earlier prior-day plan, broad target-day plan, and completed thin target-day backfill.

## Public Interfaces and Contracts

The public contract for downstream model builders is:

```sql
SELECT *
FROM gold.feature_values
WHERE feature_build_version = 'TMAX_THIN_V1';
```

The public audit contract is:

```sql
SELECT *
FROM audit.gribstream_backfill_chunks
WHERE job_id = 'klga_t1245utc_tmax_thin_backfill_v1';
```

The public raw-request contract is:

```sql
SELECT sr.*
FROM audit.gribstream_backfill_chunks c
JOIN bronze.source_requests sr
  ON sr.source_request_id = c.source_request_id
WHERE c.job_id = 'klga_t1245utc_tmax_thin_backfill_v1';
```

The public availability contract is:

```sql
SELECT *
FROM silver.availability_ledger
WHERE source_name LIKE 'gribstream%';
```

Use the final chunk table and feature trace JSON to narrow availability rows to a specific final request, raw file, model, member, or date range.

## Database Tables Used

| Table | Purpose |
|---|---|
| `audit.gribstream_backfill_chunks` | One row per planned/fetched chunk; status, request hash, request JSON, estimated credits, raw URI, rows upserted |
| `bronze.source_requests` | Authenticated request metadata and raw response body hash/size |
| `silver.availability_ledger` | Row-level source/run/valid/member/variable availability evidence |
| `audit.gribstream_source_gaps` | Explicit gaps, missing rows, empty chunks, and coverage defects |
| `gold.feature_values` | Final compact model-ready features |

The final feature query surface is:

```sql
SELECT *
FROM gold.feature_values
WHERE feature_build_version = 'TMAX_THIN_V1';
```

The final chunk-status query surface is:

```sql
SELECT *
FROM audit.gribstream_backfill_chunks
WHERE job_id = 'klga_t1245utc_tmax_thin_backfill_v1';
```

## Important Interpretation Rules

1. `T_1245UTC` is the market/forecast production cutoff, not the model run time.
2. The buffer selects the latest allowed model run before the cutoff.
3. The selected model run plus forecast lead gives the valid time.
4. Tmax proxy features are computed from forecast valid times, not from the buffer itself.
5. Current-state RTMA features are analysis features near the cutoff, not future Tmax forecasts.
6. Ensemble feature rows represent distribution summaries and probabilities, not one row per full raw member value in the final modeling surface.
7. Raw member data remains auditable through raw files, availability rows, and trace JSON.
8. Gap rows are part of the data contract and should be reviewed, not ignored.

## Testing and Verification Evidence

The completed job evidence shows:

- all final chunks terminal
- no failed chunks
- no open/running chunks
- 438/438 final source requests returned HTTP 200
- all final chunks have request hashes
- all final chunks have request JSON
- all final chunks have source request IDs
- all final chunks have raw storage URIs
- all `TMAX_THIN_V1` feature rows have non-null source run time
- all `TMAX_THIN_V1` feature rows have non-null source valid time
- all `TMAX_THIN_V1` feature rows have non-null source age
- all `TMAX_THIN_V1` feature rows have non-null source availability time
- all `TMAX_THIN_V1` feature rows have non-null source trace JSON

The earlier implementation work also ran code-level checks for the thin runner, including compile/test/validation commands recorded in the implementation context. This final document is based on the live persisted database state after the completed run.

## Known Limitations and Follow-Up Work

### NBMQMD `p05` Selector Metadata

The final request JSON includes:

```text
alias: tmp_max18_p05
info: 15% level | max-18h
```

This should be audited against the live GribStream selector catalog before `tmp_max18_p05` is used as a true fifth percentile. The raw request was preserved exactly.

### AIGEFSSFC Empty Chunks

`aigefssfc` has two completed-empty chunks. They should remain in the final ledger as evidence of provider-side empty coverage for those date/selector/member ranges unless a later manual catalog audit proves the request should be repaired.

### Gap Rows Need Feature-Level Review Before Modeling

The final job completed, but models with gaps should be reviewed before downstream training. The highest gap counts are in:

- `aifsenfo`
- `aigefssfc`
- `hrrr`
- `gfs`
- `nbm`

This does not invalidate the backfill. It means downstream feature assembly must respect feature availability and avoid assuming every model has every feature on every target date.

### This Is Not A Forecast Skill Backtest

This document describes fetched and persisted GribStream data. It does not prove that `T_1245UTC` is optimal for trading or forecasting skill. That requires a separate weather + market backtest comparing cutoffs, forecast error, bucket calibration, market prices, order books, spread, depth, and realized PnL.

## Final Operational Conclusion

The KLGA GribStream backfill that actually exists is:

```text
job_id: klga_t1245utc_tmax_thin_backfill_v1
cutoff_id: T_1245UTC
feature_profile: TMAX_THIN_V1
endpoint: /runs
status: complete
failed chunks: 0
open chunks: 0
estimated credits: 404,160
feature rows: 1,528,742
```

It covers all configured live-relevant GribStream models through 2026-06-28 using the target-day 12:45 UTC cutoff and model-specific run availability buffers. The data is intentionally thin: Tmax-relevant temperature, ensemble temperature distributions, NBMQMD max-18h percentiles, and RTMA current-state support. It excludes broad non-Tmax variables and full atomic persistence by design.

For downstream modeling, use `gold.feature_values` with `feature_build_version = 'TMAX_THIN_V1'`, and use `audit.gribstream_backfill_chunks`, `silver.availability_ledger`, `bronze.source_requests`, and raw files for audit and debugging.

## Reviewer Checklist

Before using this dataset for modeling, verify:

- `audit.gribstream_backfill_chunks` has no open chunks for `klga_t1245utc_tmax_thin_backfill_v1`.
- `gold.feature_values` is filtered to `feature_build_version = 'TMAX_THIN_V1'`.
- target dates with model gaps are handled explicitly in feature assembly.
- `aigefssfc` completed-empty chunks are not treated as failed runner behavior.
- NBMQMD `tmp_max18_p05` selector metadata is reviewed before percentile modeling.
- model run times are interpreted as UTC.
- `T_1245UTC` is treated as the forecast-production cutoff, not as the model run time.
- RTMA is treated as current-state support, not as a future Tmax forecast.
- raw files remain available for any feature or availability dispute.
- this report is used as the consolidated GribStream data contract instead of the superseded broad-plan documents.
