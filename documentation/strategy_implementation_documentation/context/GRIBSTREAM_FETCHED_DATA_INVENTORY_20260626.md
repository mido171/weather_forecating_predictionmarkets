# GribStream Fetched Data Inventory - 2026-06-26

This document explains the GribStream data currently fetched into the HKG Tmax research project.

It covers:

- what the data is;
- where it is stored;
- which models were fetched;
- date ranges and row counts;
- what weather variables exist;
- which sources are usable for Tmax;
- known gaps and blockers;
- how to use the data leakage-safely.

## One-Sentence Summary

The fetched GribStream data is historical forecast-model output for Hong Kong Tmax research, stored as raw compressed API responses plus normalized wide forecast rows in PostgreSQL.

It is not observation/actual weather data. It is forecast data that models would have produced before each target day.

## Source Of Truth

Database:

```text
postgresql://***:***@127.0.0.1:5432/hkg_tmax_research
```

Main PostgreSQL tables:

```text
nwp_tactical.acquisition_chunk
nwp_tactical.raw_response_object
nwp_tactical.forecast_wide
nwp_tactical.validation_issue
```

Raw GribStream payloads:

```text
data/_pipeline_internal/raw/gribstream_tactical_full_tactical_backfill_ok_tmax/
data/_pipeline_internal/raw/gribstream_tactical_batch_smoke_10w/
data/_pipeline_internal/raw/gribstream_tactical_first_week/
data/_pipeline_internal/raw/gribstream_tactical_smoke/
```

Main audit/report files:

```text
documentation/T07_T12_FULL_TACTICAL_BACKFILL_20260625_RESULT.md
documentation/T07_T12_DEEP_SANITY_AUDIT_20260625.md
documentation/strategy_implementation_documentation/GRIBSTREAM_LEAKAGE_SAFE_DB_RETRIEVAL_LEDGER_20260626.md
experiments/0214_tactical_h24n_gribstream_backfill/full_tactical_backfill_ok_tmax/DEEP_SANITY_AUDIT_20260625.md
experiments/0214_tactical_h24n_gribstream_backfill/full_tactical_backfill_ok_tmax/deep_sanity_audit_20260625.json
```

## What Kind Of Data Is This?

Each normalized row in `nwp_tactical.forecast_wide` is one model forecast point:

```text
dataset/model + model_run_time + forecast_valid_time + lead_time + location + member + weather variables
```

Core time fields:

- `run_time_utc`: the model run/cycle timestamp.
- `valid_time_utc`: the timestamp the forecast value applies to.
- `lead_hours`: `valid_time_utc - run_time_utc`.
- `target_date_hkt`: the Hong Kong local date being predicted.
- `cutoff_id`: currently `H24N`, meaning the T-24 next-day decision framing.

Important: this is forecast data, not observed actuals. Actual realized HKO Tmax must come from the HKO target/label tables, not from GribStream.

## Current Storage Scope

Raw objects currently present in `nwp_tactical.raw_response_object`:

| Source scope | Raw objects | Row count sum | Bytes |
| --- | ---: | ---: | ---: |
| `smoke` | 14 | 1,728 | 50,853 |
| `first_week` | 13 | 11,796 | 298,866 |
| `batch_smoke_10w` | 95 | 123,652 | 3,129,036 |
| `full_tactical_backfill_ok_tmax` | 1,163 | 1,964,157 | 56,488,866 |

Rows currently visible in `nwp_tactical.forecast_wide`:

| Source scope | Dataset | Rows |
| --- | --- | ---: |
| `batch_smoke_10w` | `gefsatmos` | 933 |
| `full_tactical_backfill_ok_tmax` | all full-run datasets | 1,964,157 |

Critical modeling rule:

```sql
JOIN nwp_tactical.raw_response_object r
  ON r.response_object_id = fw.source_response_object_id
WHERE r.object_uri LIKE '%full_tactical_backfill_ok_tmax%'
```

Reason: 933 older `batch_smoke_10w` `gefsatmos` rows are still mixed into `forecast_wide`. Do not train or score from `forecast_wide` without filtering to the full-run source, unless those old smoke rows are purged/moved first.

## Full Tactical Backfill Totals

The full tactical backfill scope is:

```text
full_tactical_backfill_ok_tmax
```

Run totals:

| Metric | Value |
| --- | ---: |
| Planned chunks | 1,163 |
| Completed chunks | 1,163 |
| Full-run normalized rows | 1,964,157 |
| Raw objects checked in deep audit | 1,163 |
| Missing raw files | 0 |
| Raw byte-size mismatches | 0 |
| API HTTP errors | 0 |
| Estimated credits consumed | 1,889,276 |

The runner ended with `status: failed` only because of data-quality flags, not because GribStream blocked, rejected, or rate-limited the run.

## Model Inventory And Date Ranges

Times below use UTC for model run and valid-time ranges.

Target dates are Hong Kong calendar dates because the target market is Hong Kong Tmax.

| Dataset | Role / family | Rows | Run-time UTC range | Valid-time UTC range | Target-date HKT range | Runs | Locations | Members | Lead hours | Tmax status |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `gfs` | Core deterministic NWP | 575,004 | 2021-03-23 00Z to 2026-06-22 00Z | 2021-03-23 15Z to 2026-06-23 15Z | 2021-03-23 to 2026-06-23 | 1,918 | 12 | 1 | 15..39 | Usable |
| `gefsatmosmean` | Core GEFS ensemble mean | 200,436 | 2020-10-01 18Z to 2026-06-21 18Z | 2020-10-02 18Z to 2026-06-23 15Z | 2020-10-03 to 2026-06-23 | 2,088 | 12 | 1 | 24..45 | Usable |
| `gefsatmos` | Core GEFS ensemble members | 516,891 | 2020-10-01 18Z to 2026-06-21 18Z | 2020-10-02 18Z to 2026-06-23 15Z | 2020-10-03 to 2026-06-23 | 2,085 | 1 | 31 | 24..45 | Usable |
| `ifsoper` | Core IFS deterministic | 91,260 | 2024-02-28 18Z to 2026-06-21 18Z | 2024-02-29 15Z to 2026-06-23 15Z | 2024-02-29 to 2026-06-23 | 845 | 12 | 1 | 21..45 | Usable |
| `ifsenfo` | Core IFS ensemble members | 343,616 | 2024-03-01 18Z to 2026-06-21 18Z | 2024-03-02 18Z to 2026-06-23 15Z | 2024-03-03 to 2026-06-23 | 843 | 1 | 51 | 24..45 | Usable with member-0 caveat |
| `cwawrf15` | Rolling/prospective deterministic source | 180 | 2026-06-22 18Z to 2026-06-24 18Z | 2026-06-23 12Z to 2026-06-26 12Z | 2026-06-23 to 2026-06-26 | 3 | 12 | 1 | 18..42 | Usable only as rolling/live source |
| `aifsoper` | Optional AI deterministic | 28,884 | 2025-02-25 18Z to 2026-06-21 18Z | 2025-02-26 12Z to 2026-06-23 12Z | 2025-02-26 to 2026-06-23 | 482 | 12 | 1 | 18..42 | Usable |
| `aifsenfo` | Optional AI ensemble members | 72,270 | 2025-07-02 18Z to 2026-06-21 18Z | 2025-07-03 18Z to 2026-06-23 12Z | 2025-07-04 to 2026-06-23 | 355 | 1 | 51 | 24..42 | Usable |
| `aigfssfc` | Optional AI/GFS surface deterministic | 3,660 | 2026-04-21 18Z to 2026-06-21 18Z | 2026-04-22 12Z to 2026-06-23 12Z | 2026-04-22 to 2026-06-23 | 61 | 12 | 1 | 18..42 | Usable over short range |
| `aigfspres` | Optional AI/GFS pressure/upper-air support | 3,660 | 2026-04-21 18Z to 2026-06-21 18Z | 2026-04-22 12Z to 2026-06-23 12Z | 2026-04-22 to 2026-06-23 | 61 | 12 | 1 | 18..42 | Not a Tmax source |
| `aigefssfc` | Optional AI/GEFS surface ensemble | 46,252 | 2025-06-01 18Z to 2026-06-21 18Z | 2025-06-02 18Z to 2026-06-23 12Z | 2025-06-03 to 2026-06-23 | 373 | 1 | 31 | 24..42 | Blocked as Tmax source |
| `graphcast` | Optional AI deterministic | 44,220 | 2024-04-25 18Z to 2026-05-04 18Z | 2024-04-26 12Z to 2026-05-06 12Z | 2024-04-26 to 2026-05-06 | 737 | 12 | 1 | 18..42 | Usable |
| `fourcastnetgfs` | Optional AI/GFS deterministic | 37,824 | 2024-05-02 18Z to 2026-02-18 18Z | 2024-05-03 12Z to 2026-02-20 12Z | 2024-05-03 to 2026-02-20 | 631 | 12 | 1 | 18..42 | Usable through observed archive end |
| `nbmoc` | Probe-only source | 0 | empty | empty | empty | 0 | 0 | 0 | empty | Not usable |

## Spatial Policy

Two spatial strategies were used:

1. Deterministic and ensemble-mean models use the 12-point Hong Kong stencil.
2. Full-member ensemble models use HKO center only, to avoid exploding row count and quota.

The 12-point stencil gives local spatial context around Hong Kong. HKO-center-only ensemble pulls give member spread at the main target point.

## Member Policy

| Dataset group | Member policy |
| --- | --- |
| Deterministic models | `member_number = 0` only |
| Ensemble mean models | mean member / deterministic mean row |
| `gefsatmos` | members `0..30` at HKO center |
| `ifsenfo` | members `0..50` at HKO center |
| `aifsenfo` | members `0..50` at HKO center |
| `aigefssfc` | members `0..30` at HKO center |

## Variables Stored In `forecast_wide`

Common row-identifying fields:

| Column | Meaning |
| --- | --- |
| `dataset_code` | GribStream dataset/model code |
| `acquisition_version` | Current tactical version, usually `tactical_h24n_v1` |
| `target_date_hkt` | Hong Kong target date |
| `cutoff_id` | Current cutoff family, `H24N` |
| `run_time_utc` | Model cycle/run time |
| `valid_time_utc` | Forecast valid time |
| `lead_hours` | Forecast lead in hours |
| `location_code` | Project location/stencil identifier |
| `requested_latitude`, `requested_longitude` | Requested coordinate |
| `returned_latitude`, `returned_longitude` | Returned coordinate |
| `returned_grid_distance_km` | Distance to returned grid, where present |
| `member_number` | Ensemble member number, or `0` for deterministic |
| `raw_values_jsonb` | Original normalized raw values retained for provenance |
| `source_response_object_id` | Link to raw object ledger |
| `quality_status` | Row quality marker |

Weather-variable fields:

| Short name | Column | Unit / meaning |
| --- | --- | --- |
| `T2m` | `temperature_2m_k` | 2m/surface temperature, Kelvin |
| `Tmax` | `interval_tmax_2m_k` | model-provided interval max 2m temperature, Kelvin |
| `Dew` | `dewpoint_2m_k` | 2m dewpoint, Kelvin |
| `RH2m` | `relative_humidity_2m_pct` | 2m relative humidity, percent |
| `U10` | `u_wind_10m_mps` | 10m U wind component, m/s |
| `V10` | `v_wind_10m_mps` | 10m V wind component, m/s |
| `MSLP` | `mslp_pa` | mean sea-level pressure, Pa |
| `LowCloud` | `low_cloud_pct` | low cloud cover, percent |
| `APCP` | `accumulated_precip_kg_m2` | accumulated precipitation, kg/m2 |
| `DSW` | `downward_shortwave_w_m2` | downward shortwave radiation, W/m2 |
| `NSW` | `net_shortwave_w_m2` | net shortwave radiation, W/m2 |
| `TP` | `total_precip_m` | total precipitation as returned by provider selector |
| `SSRD` | `shortwave_down_j_m2` | surface solar down energy, J/m2 |
| `TCWV` | `total_column_water_vapour_kg_m2` | total column water vapour, kg/m2 |
| `PWAT` | `pwat_kg_m2` | precipitable water, kg/m2 |
| `T925` | `temperature_925_k` | 925 hPa temperature, Kelvin |
| `T850` | `temperature_850_k` | 850 hPa temperature, Kelvin |
| `RH700` | `relative_humidity_700_pct` | 700 hPa relative humidity, percent |
| `Z500` | `geopotential_height_500_m` | 500 hPa geopotential height, meters |

## Variable Coverage By Model

Percentages show non-null coverage within the full-run rows for that dataset.

Blank means no non-null values for that column in the fetched full-run data.

| Dataset | T2m | Tmax | Dew | RH2m | U10 | V10 | MSLP | LowCloud | APCP | DSW | NSW | TP | SSRD | TCWV | PWAT | T925 | T850 | RH700 | Z500 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `aifsenfo` | 100% |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `aifsoper` | 100% |  | 100% |  | 100% | 100% | 100% |  |  |  |  | 100% | 100% |  |  |  | 100% |  |  |
| `aigefssfc` | 18.0% |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `aigfspres` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 100% |  | 100% |
| `aigfssfc` | 100% |  |  |  | 100% | 100% | 100% |  |  |  |  |  |  |  |  |  |  |  |  |
| `cwawrf15` | 100% |  | 100% |  | 100% | 100% | 100% |  | 100% |  | 100% |  |  |  |  |  | 100% | 100% | 100% |
| `fourcastnetgfs` | 100% |  |  |  | 100% | 100% | 100% |  |  |  |  |  |  |  |  |  | 100% |  | 100% |
| `gefsatmos` |  | 100% |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `gefsatmosmean` | 100% | 100% | 100% | 100% | 100% | 100% | 100% |  |  |  |  |  |  |  | 100% |  |  |  |  |
| `gfs` | 100.0% | 100.0% | 100.0% |  | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |  |  |  |  |  | 100.0% | 100.0% | 100.0% | 100.0% |
| `graphcast` | 100% |  |  |  | 100% | 100% | 100% |  |  |  |  |  |  |  |  |  | 100% |  | 100% |
| `ifsenfo` | 100% |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `ifsoper` | 100% |  | 99.3% |  | 100% | 100% | 100% |  |  |  |  | 100% | 99.3% | 100% |  | 100% | 100% | 100% | 100% |

## Tmax Derivability After Leakage Filter

The table below is after applying the conservative H24N leakage-safety rule with a 6-hour publication/indexing buffer.

| Dataset | Target days seen | Days with Tmax candidate | HKO-center days | Safe daily Tmax range C | Usable rows/day | Locations/day | Members/day | Interpretation |
| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| `gfs` | 1,918 | 1,918 | 1,918 | 13.60..37.13 | 144..288 | 12..12 | 1..1 | Core usable |
| `gefsatmosmean` | 2,088 | 2,088 | 2,088 | 14.48..37.35 | 84..96 | 12..12 | 1..1 | Core usable |
| `gefsatmos` | 2,085 | 2,085 | 2,085 | 11.89..34.95 | 59..248 | 1..1 | 31..31 | Core usable ensemble |
| `ifsoper` | 845 | 845 | 845 | 16.25..35.41 | 96..96 | 12..12 | 1..1 | Core usable |
| `ifsenfo` | 843 | 843 | 843 | 15.38..32.60 | 400..408 | 1..1 | 50..51 | Core usable, member-0 caveat |
| `cwawrf15` | 3 | 3 | 3 | 29.72..33.22 | 48..48 | 12..12 | 1..1 | Rolling/prospective only |
| `aifsoper` | 482 | 482 | 482 | 16.27..33.92 | 24..48 | 12..12 | 1..1 | Optional usable |
| `aifsenfo` | 355 | 355 | 355 | 15.27..32.81 | 154..204 | 1..1 | 51..51 | Optional usable |
| `aigfssfc` | 61 | 61 | 61 | 24.15..36.35 | 48..48 | 12..12 | 1..1 | Optional usable over short range |
| `graphcast` | 737 | 737 | 737 | 16.05..34.64 | 48..48 | 12..12 | 1..1 | Optional usable |
| `fourcastnetgfs` | 631 | 631 | 631 | 14.52..34.70 | 12..48 | 12..12 | 1..1 | Optional usable through archive end |
| `aigefssfc` | 373 | 67 | 67 | 18.04..31.85 | 0..124 | 0..1 | 0..31 | Blocked as Tmax source |
| `aigfspres` | 61 | 0 | 0 | none | 0..0 | 0..0 | 0..0 | Upper-air support only |
| `nbmoc` | 0 | 0 | 0 | none | none | none | none | Not usable |

## Leakage-Safe Use Rule

Raw `forecast_wide` rows are not automatically feature-safe.

Feature extraction must enforce:

```text
run_time_utc + publication_buffer <= target_date_hkt - 1 day at 15:00 HKT
```

The current audit uses a conservative 6-hour buffer.

Equivalent UTC example for an H24N decision cutoff:

```text
target_date_hkt: 2021-03-24
decision cutoff: 2021-03-23 07Z
publication buffer: 6 hours
latest allowed run_time_utc: 2021-03-23 01Z
```

So:

- `2021-03-23 00Z` model run is safe.
- `2021-03-23 06Z` model run is not safe.

Do not simply group by `target_date_hkt`. That can include rows from model runs that were too close to or after the decision cutoff.

## Unsafe Rows If Queried Naively

Rows that fail the 6-hour H24N safety filter:

| Dataset | Rows | Safe rows | Unsafe rows |
| --- | ---: | ---: | ---: |
| `gfs` | 575,004 | 552,000 | 23,004 |
| `ifsoper` | 91,260 | 81,120 | 10,140 |
| `aifsoper` | 28,884 | 23,100 | 5,784 |
| `graphcast` | 44,220 | 35,376 | 8,844 |
| `fourcastnetgfs` | 37,824 | 30,252 | 7,572 |
| `aigfssfc` | 3,660 | 2,928 | 732 |
| `aigfspres` | 3,660 | 2,928 | 732 |
| `cwawrf15` | 180 | 144 | 36 |

Ensemble HKO-center families that were fully safe under the 6-hour filter in this pull:

- `gefsatmos`
- `gefsatmosmean`
- `ifsenfo`
- `aifsenfo`
- `aigefssfc`

Being fully safe does not mean fully usable as Tmax. For example, `aigefssfc` is leakage-safe but blocked as a Tmax source because most target days lack usable 2m/Tmax values.

## Target-Date Gaps

Target-date gap scan inside each dataset's min/max range:

| Dataset | Dates present | Missing dates inside range | First missing ranges |
| --- | ---: | ---: | --- |
| `aifsenfo` | 355 | 0 | none |
| `aifsoper` | 483 | 0 | none |
| `aigefssfc` | 373 | 13 | 2025-06-28, 2025-07-26, 2025-08-13, 2025-08-18, 2025-09-04, 2025-10-08, 2025-10-26, 2025-12-18, 2026-01-03, 2026-01-09, 2026-01-14, 2026-02-05, 2026-02-16 |
| `aigfspres` | 63 | 0 | none |
| `aigfssfc` | 63 | 0 | none |
| `cwawrf15` | 4 | 0 | none |
| `fourcastnetgfs` | 648 | 11 | 2025-01-12..2025-01-13, 2025-06-25, 2025-07-17, 2025-08-08..2025-08-14 |
| `gefsatmos` | 2,085 | 5 | 2020-11-24..2020-11-26, 2023-05-24, 2025-06-29 |
| `gefsatmosmean` | 2,088 | 2 | 2023-05-24, 2025-06-29 |
| `gfs` | 1,919 | 0 | none |
| `graphcast` | 741 | 0 | none |
| `ifsenfo` | 843 | 0 | none |
| `ifsoper` | 846 | 0 | none |

## Known Data-Quality Flags

### `ifsenfo`

Eight recent full-run chunks returned HTTP 200 and persisted data, but missed member `0`.

Affected run windows:

```text
2026-05-15 18Z to 2026-05-19 18Z
2026-05-20 18Z to 2026-05-24 18Z
2026-05-25 18Z to 2026-05-29 18Z
2026-05-30 18Z to 2026-06-03 18Z
2026-06-04 18Z to 2026-06-08 18Z
2026-06-09 18Z to 2026-06-13 18Z
2026-06-14 18Z to 2026-06-18 18Z
2026-06-19 18Z to 2026-06-21 18Z
```

Interpretation: this is a provider-content/member-availability issue, not an HTTP/API failure. Other members were persisted.

### `fourcastnetgfs`

The requested tail from `2026-02-19 18Z` through `2026-02-28 18Z` returned HTTP 200 with zero rows.

Persisted rows end at:

```text
2026-02-18 18Z
```

Treat `fourcastnetgfs` as available only through the observed archive end unless a later provider probe proves more rows exist.

### `nbmoc`

Probe returned HTTP 200 with zero rows.

Treat `nbmoc` as not usable for this project unless a later selector/provider probe proves non-empty HKO-domain coverage.

### `aigfspres`

Rows exist, but this is an upper-air support source in this pull. It does not provide surface daily Tmax candidates.

Do not use as a daily Tmax source.

### `aigefssfc`

Rows exist and are leakage-safe under the 6-hour rule, but usable surface temperature/Tmax coverage is poor:

```text
target days seen: 373
target days with Tmax candidate: 67
```

Do not use as a daily Tmax source unless a later selector/provider probe fixes the missing/non-null coverage.

## Member Coverage Anomalies

| Dataset | Affected run/valid groups | Issue |
| --- | ---: | --- |
| `ifsenfo` | 328 | member `0` missing in affected recent run/valid groups |
| `aifsenfo` | 3 | most members `1..50` missing in three groups |
| `gefsatmos` | 2 | partial member gaps in two early groups |

## Physical Range And Cross-Field Flags

No dewpoint-above-temperature anomalies were found.

Physical-range flags:

| Dataset | Column | Issue |
| --- | --- | --- |
| `cwawrf15` | `accumulated_precip_kg_m2` | tiny negative values down to about `-5.22e-7`; likely floating-point/accumulation noise |
| `aifsoper` | `total_precip_m` | max `146.34`; likely selector/unit semantics, do not treat blindly as meters of daily rain |
| `aifsoper` | `shortwave_down_j_m2` | some values above conservative threshold |
| `ifsoper` | `shortwave_down_j_m2` | some values above conservative threshold |
| `ifsoper` | `relative_humidity_700_pct` | 266 rows above 100 percent, max `106.09` |

Cross-field flags:

| Dataset | Issue |
| --- | --- |
| `gefsatmosmean` | 382 rows where `interval_tmax_2m_k < temperature_2m_k` |
| `gfs` | 27,776 rows where `interval_tmax_2m_k < temperature_2m_k` |

Interpretation: `interval_tmax_2m_k` can be used where valid, but do not assume every row-level interval Tmax dominates every instantaneous 2m temperature row without understanding the model/provider accumulation-window semantics.

## What Is Usable Right Now?

Usable for leakage-safe Tmax feature extraction, with correct filters:

- `gfs`
- `gefsatmosmean`
- `gefsatmos`
- `ifsoper`
- `ifsenfo`
- `aifsoper`
- `aifsenfo`
- `aigfssfc`
- `graphcast`
- `fourcastnetgfs` through observed archive end
- `cwawrf15` for rolling/prospective/live collection only

Not usable as Tmax sources right now:

- `nbmoc`
- `aigfspres`
- `aigefssfc`

Support-only or blocked data may still be useful for diagnostics or future research, but it must not be treated as a clean daily Tmax forecast input.

## Correct Modeling Query Shape

The dedicated implementation ledger for this query policy is:

```text
documentation/strategy_implementation_documentation/GRIBSTREAM_LEAKAGE_SAFE_DB_RETRIEVAL_LEDGER_20260626.md
```

Feature extraction should:

1. join `forecast_wide` to `raw_response_object`;
2. filter to `full_tactical_backfill_ok_tmax`;
3. apply the H24N cutoff rule with publication buffer;
4. exclude blocked Tmax sources;
5. derive daily features by model/date/location/member only after the above filters.

Skeleton:

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

The exact timestamp expression should be implemented and tested in the project feature-extraction code before model training.

## How To Refresh This Inventory

Run the deep sanity audit:

```powershell
.\.venv\Scripts\python.exe scripts\audit_tactical_gribstream_deep_sanity.py --skip-file-hash
```

Then update this file using:

```text
experiments/0214_tactical_h24n_gribstream_backfill/full_tactical_backfill_ok_tmax/deep_sanity_audit_20260625.json
```

If full raw payload hash verification is required, rerun the audit without `--skip-file-hash`, but expect it to take longer.

## Completion / Use Gate

Before this data becomes final modeling input, the project must either:

1. purge/move the 933 old `batch_smoke_10w` rows from `forecast_wide`, or enforce the full-run source filter everywhere;
2. implement a tested leakage-safe feature view or extractor;
3. decide whether the `ifsenfo` member-0 gap is acceptable;
4. treat `fourcastnetgfs` as ending at its observed archive end;
5. keep `nbmoc`, `aigfspres`, and `aigefssfc` excluded from daily Tmax source features.
