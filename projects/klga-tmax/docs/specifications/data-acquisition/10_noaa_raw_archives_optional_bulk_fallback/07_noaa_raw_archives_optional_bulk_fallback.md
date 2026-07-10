# 07 — NOAA / ECMWF Raw Archive Optional Bulk Fallbacks

## 1. Purpose

Raw archives are optional fallback and research sources. They are not the first implementation path because GribStream and IEM give faster integration. However, raw archives are valuable when:

```text
GribStream cost is too high for a broad historical pull,
a model/variable/member is unavailable through an API,
we need full-grid neighborhood extraction at scale,
we need independent verification of API outputs,
we want longer history than GribStream for selected products.
```

Codex must implement raw archive acquisition as modular optional jobs, not as dependencies for the first live trading system.

## 2. Raw archive source list

### 2.1 NOAA HRRR on AWS

Official registry:

```text
https://registry.opendata.aws/noaa-hrrr-pds/
```

Use for:

```text
Long HRRR history, local 3-km gridded forecast features, optional cost reduction versus API extraction.
```

### 2.2 NOAA NBM on AWS

Official registry:

```text
https://registry.opendata.aws/noaa-nbm/
```

Use for:

```text
Raw NBM deterministic/probabilistic grids, wider variable access, optional bulk extraction.
```

### 2.3 NOAA GFS on public cloud / NOMADS / NCEI

Use for:

```text
Raw GFS deterministic synoptic features and longer-history reconstruction.
```

### 2.4 NOAA GEFS public cloud / NCEI

Use for:

```text
Raw GEFS member-level features and probabilistic distribution backfills.
```

### 2.5 NCEI Integrated Surface Database / Global Hourly

Official source:

```text
https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database
```

Use for:

```text
Observation redundancy and station-history quality checks.
```

### 2.6 ECMWF Open Data / archive

ECMWF open data is useful for recent rolling runs but is not a full free historical archive. ECMWF historical archive access is separate. Use only if credentials/license permit.

Use for:

```text
ECMWF IFS/ENS raw verification and expanded features.
```

## 3. Priority rule

Raw archive acquisition order:

```text
1. HRRR raw archive if GribStream HRRR cost or variable coverage is a problem.
2. NBM raw archive if GribStream NBM/NBMQMD cost or variable coverage is a problem.
3. GEFS raw archive if member-level GribStream extraction is too expensive.
4. GFS raw archive if longer GFS history is desired.
5. NCEI ISD for observation redundancy.
6. ECMWF raw archive only if licensing and access are solved.
```

## 4. Raw archive extraction design

### 4.1 Do not store whole raw archives locally by default

Codex must not download entire multi-year gridded datasets unless explicitly configured. Instead:

```text
1. List object paths for required model/date/cycle/forecast-hour/variable.
2. Download only required GRIB2 chunks/files.
3. Extract only Tier B pseudo-points or configured grid neighborhood.
4. Store extracted point rows in silver tables.
5. Cache raw files locally only if caching is enabled.
```

### 4.2 Required tools

Codex should implement with:

```text
s3fs or boto3 for public S3 listing/downloading
cfgrib/eccodes or wgrib2 for GRIB extraction
xarray for array handling
pyproj/cartopy only if needed for grid coordinate transforms
```

If `wgrib2` is unavailable, use `cfgrib`/`eccodes`. If both are unavailable, the raw archive job must fail explicitly with dependency instructions rather than silently returning empty data.

## 5. Required raw model variables

Match GribStream variable families:

```text
2m temperature
2m dewpoint
10m u/v wind
surface pressure / MSLP
TMAX/TMIN where available
cloud cover
shortwave radiation
precipitation/QPF
CAPE/CIN/convection proxies
925/850/700/500 mb temperature, wind, humidity, height
boundary-layer height where available
```

Use model-native short names and GRIB discipline/category/parameter metadata. Store exact GRIB metadata in bronze/silver.

## 6. Required run and forecast-hour selection

For each target date/cutoff, enumerate candidate runs exactly as in the GribStream spec. For each run, fetch only forecast hours that overlap:

```text
market local day T
peak-temperature window
pre-target/current-state windows
```

Required forecast hours:

```text
All forecast hours whose valid time is between T 00:00 and T 23:59 America/New_York.
All forecast hours whose valid time is between local 12:00 and 21:00 America/New_York.
Prior/current-state hours needed for cutoff diagnostics.
```

For hourly models, use hourly hours. For 3-hourly/6-hourly models, use native steps and interpolate only as a downstream feature, never by inventing raw rows.

## 7. Availability rules

Raw archive objects have object modification times, but historical object mod times do not necessarily represent live availability in a retrospective backtest. Therefore:

```text
Use actual live ingestion logs once collected.
For historical backfill, use the same conservative lag rules as the corresponding GribStream model.
```

Raw archive rows must still populate:

```text
run_time_utc
valid_time_utc
forecast_hour
provider_available_at_utc
availability_method
our_ingested_at_utc
```

## 8. Silver schema

```text
CREATE TABLE raw_nwp_point_values (
    source_name TEXT NOT NULL,                 -- noaa_hrrr_aws, noaa_nbm_aws, noaa_gfs, noaa_gefs, ecmwf_archive
    model_id TEXT NOT NULL,
    bucket_or_archive TEXT,
    object_uri TEXT,
    object_last_modified_utc TIMESTAMP,
    grib_message_index INTEGER,
    grib_short_name TEXT,
    grib_parameter_name TEXT,
    grib_level_type TEXT,
    grib_level_value TEXT,
    member TEXT,
    grid_point_id TEXT NOT NULL,
    lat DOUBLE PRECISION NOT NULL,
    lon DOUBLE PRECISION NOT NULL,
    interpolation_method TEXT NOT NULL,        -- nearest, bilinear, native_grid_cell
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
    PRIMARY KEY (source_name, model_id, COALESCE(member,''), grid_point_id, run_time_utc, valid_time_utc, variable_name, object_uri)
);
```

## 9. Equivalence checks against GribStream

If both raw archive and GribStream are available for the same model/variable/coordinate/run/valid time, compute:

```text
raw_minus_gribstream_value
absolute_difference
relative_difference
```

Expected differences may arise from interpolation and gridpoint selection. Store:

```text
comparison_method
coordinate_method_raw
coordinate_method_gribstream
```

Flag:

```text
temperature difference >= 1.5°F
wind speed difference >= 5 mph
pressure difference >= 3 hPa
```

These are QA flags, not automatic source rejections.

## 10. Derived features

Raw archive features should mirror GribStream features with `raw_` prefix:

```text
raw_hrrr_KLGA_daily_max_tmp_2m_f
raw_nbm_KLGA_tmax_2m_f
raw_gefs_ensemble_prob_tmax_ge_{k}
raw_gfs_tmp_850_peak_window_mean
raw_hrrr_inland_minus_KLGA_peak_tmp
```

Do not mix raw and GribStream values under the same feature name.

## 11. Acceptance tests

```text
[ ] Raw archive jobs can list and fetch one HRRR date/cycle/hour for KLGA Tier A points.
[ ] Extracted raw rows include exact object URI and GRIB metadata.
[ ] Raw rows follow the same availability rules as API rows.
[ ] Raw archive features are namespaced separately.
[ ] Comparison jobs can detect and report raw-vs-GribStream differences.
[ ] Raw archive jobs can be disabled without breaking the production pipeline.
```
