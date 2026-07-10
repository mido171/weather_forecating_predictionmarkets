# 06 — Open-Meteo Auxiliary Forecast-Run Acquisition

## 1. Purpose

Open-Meteo is an auxiliary forecast source, not the primary source. It is used for:

```text
external sanity checks against GribStream,
cheap extra run-history features where available,
fallback if a GribStream model/source has a gap,
independent API-based model-update availability metadata,
benchmark forecasts for model validation.
```

Open-Meteo Single Runs is the only Open-Meteo product suitable for exact historical model-run reconstruction. Historical Forecast and Previous Runs APIs are useful for baselines and skill analysis but must not replace exact run-based data in the final as-of system.

Official source pages:

```text
Single Runs API:       https://open-meteo.com/en/docs/single-runs-api
Historical Forecast:   https://open-meteo.com/en/docs/historical-forecast-api
Previous Runs API:     https://open-meteo.com/en/docs/previous-runs-api
Model Updates API:     https://open-meteo.com/en/docs/model-updates
```

Open-Meteo states that Single Runs preserves individual model runs retrievable by `run=YYYY-MM-DDTHH:00`, but the run parameter is initialization time, not public availability time. Its docs state that global models typically need 4–6 hours after initialization and regional models 1–3 hours before output is distributed. Therefore Open-Meteo rows must follow the same availability discipline as GribStream rows.

## 2. Required API endpoints

### 2.1 Single Runs API

Endpoint:

```text
GET https://single-runs-api.open-meteo.com/v1/forecast
```

Required parameters:

```text
latitude={lat}
longitude={lon}
run={YYYY-MM-DDTHH:00}
timezone=GMT
start_hour={relative start if supported} OR start_date/end_date if supported by model docs
end_date={YYYY-MM-DD}
hourly={comma-separated variables}
temperature_unit=fahrenheit
wind_speed_unit=mph
precipitation_unit=inch
models={model_id if required/available}
```

Codex must consult Open-Meteo docs for exact model parameter names at implementation time, because Open-Meteo model ids can evolve. The required semantic behavior is exact: retrieve a specific model initialization run and its full/needed horizon.

### 2.2 Model Updates metadata API

Use Open-Meteo model-update metadata to capture live availability fields when available:

```text
last_run_initialisation_time
last_run_modification_time
last_run_availability_time
temporal_resolution_seconds
update_interval_seconds
```

Open-Meteo recommends waiting an additional 10 minutes after `last_run_availability_time` due to eventual consistency. Codex must therefore set:

```text
provider_available_at_utc = last_run_availability_time + 10 minutes
availability_method = provider_metadata
```

for live Open-Meteo data when metadata is available.

## 3. Required models

Open-Meteo model names must be mapped to canonical model families. Codex must implement the mapping as config, not hard-coded inside fetch logic.

Initial required canonical families:

| canonical family | Open-Meteo source role | use |
|---|---|---|
| `ecmwf_ifs_hres` | ECMWF IFS HRES / 9 km where available | Independent high-quality global reference. |
| `gfs` | NOAA GFS | Sanity check against GribStream GFS. |
| `hrrr` | NOAA HRRR where available | Sanity check/high-res comparison. |
| `nbm` | NOAA NBM where available | Sanity check against GribStream/NBM. |
| `aifs` | ECMWF AIFS where available | AI model comparison. |
| `aigfs` | NOAA AI GFS where available | AI model comparison. |

Do not assume every model has long history. Open-Meteo Single Runs availability is shorter for many models. Store exact model availability after first catalog/metadata query.

## 4. Required coordinates

Use the same gridded pseudo-point tiers as GribStream.

Default:

```text
Tier A for initial auxiliary acquisition.
```

Reason: Open-Meteo is auxiliary and should not be allowed to create excessive API volume before its incremental value is validated.

Upgrade to Tier B only if Open-Meteo features improve out-of-fold metrics.

## 5. Required hourly variables

Request all variables below when supported by the model:

```text
temperature_2m
dew_point_2m
relative_humidity_2m
apparent_temperature
precipitation
rain
showers
snowfall
cloud_cover
cloud_cover_low
cloud_cover_mid
cloud_cover_high
shortwave_radiation
direct_radiation
diffuse_radiation
surface_pressure
pressure_msl
wind_speed_10m
wind_direction_10m
wind_gusts_10m
cape
```

For pressure levels where supported:

```text
temperature_925hPa
temperature_850hPa
relative_humidity_925hPa
relative_humidity_850hPa
wind_speed_925hPa
wind_direction_925hPa
wind_speed_850hPa
wind_direction_850hPa
geopotential_height_500hPa
```

If a variable is unsupported, record `selector_missing` and continue.

## 6. Required derived daily variables

If Open-Meteo supports daily derived fields for the exact single run, request:

```text
temperature_2m_max
temperature_2m_min
precipitation_sum
rain_sum
shortwave_radiation_sum
wind_speed_10m_max
wind_gusts_10m_max
```

If daily fields are generated from stitched latest forecasts rather than the exact single run, do not use them. Derive daily max from hourly single-run rows instead.

## 7. Time windows and run selection

For each target date `T` and cutoff:

1. Enumerate all standard model run times in the previous 72 hours:

```text
00Z, 06Z, 12Z, 18Z for global models
hourly or model-specific for regional models if supported
```

2. Apply model-specific availability:

```text
provider_available_at_utc <= cutoff_utc
```

3. Fetch runs that can forecast the target market day.

4. Store all run rows, not just the latest run. Run-to-run trends are features.

## 8. Availability defaults

When provider metadata is unavailable:

```text
global models:   provider_available_at_utc = run_time_utc + 6h10m
regional models: provider_available_at_utc = run_time_utc + 3h10m
```

The extra 10 minutes implements Open-Meteo's eventual-consistency warning.

If metadata exists:

```text
provider_available_at_utc = last_run_availability_time + 10 minutes
```

## 9. Silver table schema

```text
CREATE TABLE open_meteo_run_values (
    source_name TEXT NOT NULL DEFAULT 'open_meteo',
    open_meteo_model_id TEXT NOT NULL,
    canonical_model_family TEXT NOT NULL,
    grid_point_id TEXT NOT NULL,
    lat DOUBLE PRECISION NOT NULL,
    lon DOUBLE PRECISION NOT NULL,
    run_time_utc TIMESTAMP NOT NULL,
    valid_time_utc TIMESTAMP NOT NULL,
    forecast_hour DOUBLE PRECISION NOT NULL,
    variable_name TEXT NOT NULL,
    unit_original TEXT,
    value_original DOUBLE PRECISION,
    unit_canonical TEXT,
    value_canonical DOUBLE PRECISION,
    provider_available_at_utc TIMESTAMP,
    our_ingested_at_utc TIMESTAMP NOT NULL,
    availability_method TEXT NOT NULL,
    source_request_id TEXT NOT NULL,
    quality_flag TEXT NOT NULL DEFAULT 'ok',
    quality_note TEXT,
    PRIMARY KEY (open_meteo_model_id, grid_point_id, run_time_utc, valid_time_utc, variable_name, source_request_id)
);
```

## 10. Derived features

Open-Meteo features mirror GribStream features but must be prefixed separately:

```text
om_{model}_KLGA_daily_max_temperature_2m_f
om_{model}_KLGA_peak_window_max_temperature_2m_f
om_{model}_KLGA_peak_window_cloud_cover_mean
om_{model}_KLGA_peak_window_shortwave_mean
om_{model}_KLGA_peak_window_precip_sum
om_{model}_KLGA_peak_window_wind_speed_mean
om_{model}_KLGA_peak_window_wind_dir_circular_mean
om_{model}_inland_minus_KLGA_peak_temp
om_{model}_coastal_minus_KLGA_peak_temp
om_{model}_latest_minus_previous_cycle_tmax
```

Also create disagreement features against GribStream:

```text
open_meteo_gfs_minus_gribstream_gfs_tmax
open_meteo_hrrr_minus_gribstream_hrrr_tmax
open_meteo_ifs_minus_gribstream_ifs_tmax
open_meteo_nbm_minus_gribstream_nbm_tmax
```

Only create these when both sources are eligible at the cutoff.

## 11. Use restrictions

Open-Meteo Historical Forecast API and Previous Runs API may be used for benchmark features, but the final model must treat them as separate source families:

```text
open_meteo_historical_forecast_baseline
open_meteo_previous_runs_baseline
```

They must not be mixed with exact Single Runs features without preserving source identity.

If Historical Forecast data is stitched/latest-run rather than exact run, it cannot be used for strict as-of run reconstruction unless the endpoint's semantics guarantee the historical issue time.

## 12. Acceptance tests

```text
[ ] Single Runs requests include exact `run` initialization timestamp.
[ ] Open-Meteo availability metadata is stored when available.
[ ] If metadata is absent, global/regional conservative lag defaults are applied.
[ ] Open-Meteo rows after cutoff are excluded from cutoff features.
[ ] Open-Meteo features are namespaced separately from GribStream features.
[ ] Disagreement features are generated only when both source rows are cutoff-eligible.
[ ] Historical Forecast/Previous Runs baselines are never treated as exact run reconstruction unless explicitly proven by endpoint semantics.
```
