# 05 — RTMA and URMA Analysis Field Acquisition

## 1. Purpose

RTMA and URMA are analysis products, not ordinary forecasts. They are used for:

```text
current-state gridded analysis near cutoffs,
model initialization/current-state bias diagnostics,
analysis-based historical verification features,
spatial local-gradient features around KLGA,
training labels for gridded residual research,
quality checks against Wunderground/IEM observations.
```

RTMA can be used as a live/current-state input when availability rules permit. URMA is primarily retrospective and must not leak into same-day forecasts unless its availability timestamp is before the cutoff, which is usually not true for T-1 live prediction.

Primary acquisition path is GribStream using the same API contract as `03_gribstream_nwp_forecast_runs.md`.

## 2. Products

| model_id | Source | Role |
|---|---|---|
| `rtma` | GribStream RTMA | Near-real-time surface analysis, live current-state features. |
| `urma` | GribStream URMA | Delayed high-quality analysis, training/verification/research. |

## 3. Required coordinates

Use the same pseudo-point coordinate tiers as GribStream forecasts.

Default:

```text
Tier B pseudo-points from station registry.
```

Minimum fallback:

```text
Tier A pseudo-points.
```

## 4. Required variables

Request wherever available:

```json
[
  {"name": "TMP",  "level": "2 m above ground", "alias": "tmp_2m"},
  {"name": "DPT",  "level": "2 m above ground", "alias": "dpt_2m"},
  {"name": "UGRD", "level": "10 m above ground", "alias": "u10"},
  {"name": "VGRD", "level": "10 m above ground", "alias": "v10"},
  {"name": "GUST", "level": "surface", "alias": "gust_sfc"},
  {"name": "PRES", "level": "surface", "alias": "pres_sfc"},
  {"name": "TCDC", "level": "surface", "alias": "tcdc_sfc"},
  {"name": "VIS",  "level": "surface", "alias": "vis_sfc"},
  {"name": "APCP", "level": "surface", "alias": "apcp_sfc"}
]
```

If a selector is missing for RTMA or URMA, write a source-gap row and continue.

## 5. Time windows

### 5.1 Live/current-state RTMA extraction

For each target date `T` and cutoff, request all valid analyses in:

```text
cutoff_utc - 12 hours through cutoff_utc
```

Use only rows where:

```text
valid_time_utc + 60 minutes <= cutoff_utc
```

### 5.2 Historical URMA extraction

For each target date `T`, request all valid analyses in the market local day:

```text
T 00:00:00 America/New_York through T 23:59:59 America/New_York
```

Use URMA for:

```text
research labels,
model verification,
retrospective bias correction,
spatial analysis of local gradients.
```

Default URMA availability:

```text
valid_time_utc + 8 hours
```

URMA values inside target day must never be used for the target-date forecast unless that availability rule passes, which it normally will not for T-1 or pre-local-day cutoffs.

## 6. Endpoint usage

Use GribStream `/runs` or `/timeseries` depending on catalog behavior for analysis products. For analyses with one value per valid time, `/timeseries` with explicit `timesList` is acceptable, but rows must still be stored with `forecasted_at`/`forecasted_time` metadata when returned.

Request body template:

```json
{
  "timesList": ["{analysis_valid_time_iso}"],
  "coordinates": [
    {"lat": 40.77945, "lon": -73.88027, "name": "GP_KLGA_EXACT"}
  ],
  "variables": [
    {"name": "TMP", "level": "2 m above ground", "alias": "tmp_2m"},
    {"name": "DPT", "level": "2 m above ground", "alias": "dpt_2m"},
    {"name": "UGRD", "level": "10 m above ground", "alias": "u10"},
    {"name": "VGRD", "level": "10 m above ground", "alias": "v10"}
  ]
}
```

## 7. Silver schema

Use the same `gribstream_values` table defined in `03_gribstream_nwp_forecast_runs.md` with:

```text
model_id in ('rtma', 'urma')
source_product = 'analysis'
forecast_hour = 0 or null if model does not expose a forecast lead
```

Also materialize an analysis-specific view:

```text
CREATE VIEW analysis_values AS
SELECT * FROM gribstream_values
WHERE model_id IN ('rtma', 'urma');
```

## 8. Derived features

### 8.1 RTMA live/current-state features

For each cutoff:

```text
rtma_latest_tmp_2m_f_at_KLGA
rtma_latest_dpt_2m_f_at_KLGA
rtma_latest_wind_speed_mph_at_KLGA
rtma_latest_wind_dir_deg_at_KLGA
rtma_latest_cloud_cover_pct_at_KLGA
rtma_latest_age_minutes
rtma_tmp_2m_change_last_3h
rtma_dpt_2m_change_last_3h
rtma_pressure_change_last_3h
rtma_inland_minus_KLGA_tmp
rtma_coastal_minus_KLGA_tmp
rtma_water_proxy_minus_KLGA_tmp
```

### 8.2 URMA retrospective features

For model training diagnostics after the fact:

```text
urma_market_day_max_tmp_2m_f_at_KLGA
urma_market_day_peak_time_local
urma_spatial_peak_neighbor_minus_KLGA
urma_klga_minus_wunderground_high
urma_klga_minus_iem_asos_high
```

URMA retrospective features cannot enter live prediction for that same target date; they are for evaluation, residual analysis, and settlement-reconciliation modeling.

## 9. Quality checks

```text
RTMA latest valid time must be <= cutoff_utc - 60 minutes for live features.
URMA values inside target date must be excluded from target-date features.
RTMA/URMA temperatures must convert cleanly to Fahrenheit.
RTMA station-like KLGA temp should be within 10°F of latest ASOS KLGA temp unless quality_flag="suspect".
```

## 10. Acceptance tests

```text
[ ] RTMA features at each cutoff use only analysis times available by cutoff.
[ ] URMA same-day analysis values never enter target-date forecast features.
[ ] RTMA/URMA can be joined with pseudo-point registry.
[ ] RTMA current-state features can be compared with IEM ASOS observations.
[ ] URMA diagnostic labels can be compared against Wunderground settlement labels without replacing them.
```
