# 03 — GribStream NWP Forecast Runs Acquisition

## 1. Purpose

GribStream is the primary source for historical and live numerical weather prediction features. It must provide leakage-safe, as-of model guidance for KLGA and nearby gradient/pseudo-points.

Official source pages:

```text
API docs:          https://gribstream.com/docs
FAQ:               https://gribstream.com/faq
Quickstart:        https://gribstream.com/quickstart
Model catalog:     https://gribstream.com/models
OpenAPI contract:  https://gribstream.com/openapi
```

The user has a GribStream account and an installed GribStream skill in Codex. Codex should use the installed skill when available. If the skill is unavailable in a given environment, Codex must use the HTTP API endpoints documented below.

## 2. Critical GribStream semantics

### 2.1 `/timeseries`

Endpoint pattern:

```text
POST https://gribstream.com/api/v2/<model>/timeseries
```

Purpose:

```text
For a coordinate/variable/time range, return the most recent eligible forecast value for each valid time, subject to filters such as asOf and min/max lead time.
```

Important:

```text
asOf is a model-run-time cutoff based on forecasted_at.
It does not prove the data was already available through GribStream at that wall-clock time.
```

Use `/timeseries` only for convenience when the strategy wants "best eligible forecast by model-run-time cutoff" and when the conservative availability buffer has already been applied.

### 2.2 `/runs`

Endpoint pattern:

```text
POST https://gribstream.com/api/v2/<model>/runs
```

Purpose:

```text
Return all model-run / forecast-horizon values matching run-time and valid-time filters.
```

This is the preferred endpoint for serious backtesting, run-to-run trend features, cycle-specific bias estimation, and exact forecast reconstruction.

### 2.3 Required response metadata

Every GribStream response row must retain:

```text
forecasted_at     # model run time / issue time
forecasted_time   # valid time
lat
lon
name              # coordinate name if provided
member            # if ensemble member exposed
variable columns  # one per variable/expression
```

Codex must sort responses client-side by:

```text
model, member, forecasted_at, forecasted_time, coordinate_name, variable
```

Do not assume provider response order.

### 2.4 Credit/cost control

Use `timesList` whenever extracting sparse target valid times or specific cycles. GribStream's FAQ states that `timesList` usually reduces over-fetching and credit usage because only listed times are returned.

Do not request a continuous multi-month hourly range when only target-day peak-window hours are needed.

## 3. Authentication and request defaults

HTTP headers:

```text
Authorization: Bearer <GRIBSTREAM_API_TOKEN>
Content-Type: application/json
Accept: text/csv
Accept-Encoding: gzip
```

Environment variables:

```text
GRIBSTREAM_API_TOKEN
GRIBSTREAM_API_BASE_URL=https://gribstream.com/api/v2
GRIBSTREAM_ACCEPT=text/csv
GRIBSTREAM_TIMEOUT_SECONDS=120
GRIBSTREAM_MAX_RETRIES=5
GRIBSTREAM_DEFAULT_COORDINATE_TIER=B
```

Preferred response format is CSV for large extraction. JSON/NDJSON is acceptable if the installed skill returns those formats.

## 4. Required coordinate extraction tiers

Use coordinate tiers from `10_station_universe_and_coordinates.md`.

Initial production default:

```text
Tier B pseudo-points
```

Minimum cost fallback:

```text
Tier A pseudo-points
```

Expanded research audition:

```text
Tier C 5×5 neighborhood, only after cost approval
```

Never fetch eight or more airport stations from GribStream just because the observation universe has many stations. Use pseudo-points for gridded model gradients and Wunderground/IEM for station observations.

## 5. Target valid-time extraction windows

For each target date `T`, compute:

```text
market_local_start = T 00:00 America/New_York
market_local_end   = T 23:59:59 America/New_York
market_utc_start   = converted UTC timestamp
market_utc_end     = converted UTC timestamp
```

For summer EDT example, T local day maps approximately to:

```text
T 04:00 UTC through T+1 03:59 UTC
```

Codex must request these valid-time groups for each model:

### 5.1 Minimal target-day valid hours

```text
market_utc_start through market_utc_end at model-native step
```

For models with hourly output, request all hourly valid times in the market local day.

### 5.2 Peak-temperature focus hours

Always generate a `timesList` for local peak window:

```text
12:00 through 21:00 America/New_York on target date T, hourly
```

Convert those times to UTC. In summer this is usually 16Z–01Z. This window captures most high-temperature outcomes and reduces cost for models with expensive ensemble pulls.

### 5.3 Pre-target state hours

For model/state comparison and observation correction, request:

```text
T-1 06:00 through T-1 23:00 America/New_York, every 3 hours
T 00:00 through T 06:00 America/New_York, every 1 or 3 hours depending cutoff
```

Only use rows eligible for the cutoff.

## 6. Core model groups

Codex must implement these exact model groups.

### 6.1 Core deterministic/postprocessed models

| model_id | use | priority |
|---|---|---|
| `nbm` | Primary calibrated blend / deterministic postprocessed guidance | mandatory |
| `hrrr` | High-resolution local/mesoscale guidance | mandatory |
| `rap` | Regional/boundary-layer state and mesoscale guidance | mandatory |
| `gfs` | Global synoptic anchor | mandatory |
| `ifsoper` | ECMWF operational global independent synoptic anchor | mandatory if account access supports it |
| `aifsoper` | ECMWF AI deterministic independent model | mandatory if available historically/live |
| `aigfssfc` | NOAA AI GFS surface deterministic | mandatory audition, production if out-of-fold skill positive |

### 6.2 Core probabilistic/ensemble/quantile models

| model_id | use | priority |
|---|---|---|
| `nbmqmd` | NBM quantile/percentile distribution guidance | mandatory audition, production if available |
| `gefsatmos` | GEFS individual member probabilistic guidance | mandatory |
| `gefsatmosmean` | GEFS mean fields, cheaper fallback/feature source | mandatory if member pull is too expensive |
| `ifsenfo` | ECMWF ensemble independent probabilistic guidance | mandatory if account access supports it |
| `aifsenfo` | ECMWF AI ensemble | mandatory audition, production if out-of-fold skill positive |
| `aigefssfc` | NOAA AI ensemble surface | mandatory audition, production if out-of-fold skill positive |

### 6.3 Analysis products

| model_id | use | priority |
|---|---|---|
| `rtma` | Near-real-time analyzed current-state field | mandatory for cutoff-state features |
| `urma` | Retrospective analysis/verification/bias-correction support | mandatory for research, not live T-1 same-day target leakage |

### 6.4 Audition models

These models must be ingested for a shorter research window first. They are not allowed into final production until they improve leakage-free OOF forecast or trading metrics.

```text
rrfs2dfld
rrfsprslev
refsprslev
spctstm1hr
spctstm4hr
spcltg4hr
spcwind4hr
spchail4hr
spctor4hr
uvi
```

## 7. Historical backfill start dates by model

Use GribStream catalog/model pages to confirm exact availability at runtime. Initial expected starts:

| model_id | expected GribStream start | backfill priority |
|---|---:|---|
| `hrrr` | 2014-07-30 | mandatory |
| `rtma` | 2018-01-01 | mandatory |
| `nbm` | 2020-09-29 | mandatory |
| `gefsatmos` | 2020-10-01 | mandatory |
| `gefsatmosmean` | 2020-10-01 | mandatory |
| `rap` | 2021-02-22 | mandatory |
| `gfs` | 2021-03-22 | mandatory |
| `ifsoper` | 2024-03-01 | mandatory if accessible |
| `ifsenfo` | 2024-03-01 | mandatory if accessible |
| `aifsoper` | 2025-02-25 | audition/production if skill-positive |
| `aigefssfc` | 2025-06-01 | audition/production if skill-positive |
| `aifsenfo` | 2025-07-02 | audition/production if skill-positive |
| `nbmqmd` | 2026-01-31 | audition/production if skill-positive |
| `aigfssfc` | 2026-04-16 | live audition; short history |
| `urma` | use all accessible history | analysis/research |

Codex must not force all models to share the shortest history. Each model trains over its own available period and contributes to the final system via out-of-fold expert predictions and missingness-aware stacking.

## 8. Required variable selectors

Codex must first query/consult the GribStream model catalog for each model and validate selectors. If a selector is absent for a model, record a `selector_missing` source-gap row and continue. Mandatory model groups must fail only if all temperature selectors for the model are unavailable.

### 8.1 Universal target-temperature selectors

Request these wherever available:

```json
[
  {"name": "TMP",  "level": "2 m above ground", "alias": "tmp_2m"},
  {"name": "TMAX", "level": "2 m above ground", "alias": "tmax_2m"},
  {"name": "TMIN", "level": "2 m above ground", "alias": "tmin_2m"},
  {"name": "DPT",  "level": "2 m above ground", "alias": "dpt_2m"},
  {"name": "RH",   "level": "2 m above ground", "alias": "rh_2m"}
]
```

### 8.2 Surface wind/pressure/precip/cloud/radiation selectors

Request these wherever available:

```json
[
  {"name": "UGRD",  "level": "10 m above ground", "alias": "u10"},
  {"name": "VGRD",  "level": "10 m above ground", "alias": "v10"},
  {"name": "GUST",  "level": "surface", "alias": "gust_sfc"},
  {"name": "PRMSL", "level": "mean sea level", "alias": "prmsl"},
  {"name": "MSLET", "level": "mean sea level", "alias": "mslet"},
  {"name": "APCP",  "level": "surface", "alias": "apcp_sfc"},
  {"name": "TCDC",  "level": "surface", "alias": "tcdc_sfc"},
  {"name": "TCDC",  "level": "entire atmosphere", "alias": "tcdc_eatm"},
  {"name": "DSWRF", "level": "surface", "alias": "dswrf_sfc"},
  {"name": "PWAT",  "level": "entire atmosphere", "alias": "pwat"}
]
```

### 8.3 Boundary-layer and convection selectors

Request for HRRR/RAP/RRFS where available:

```json
[
  {"name": "HPBL", "level": "surface", "alias": "hpbl"},
  {"name": "CAPE", "level": "surface", "alias": "cape_sfc"},
  {"name": "CIN",  "level": "surface", "alias": "cin_sfc"},
  {"name": "REFC", "level": "entire atmosphere", "alias": "refc"},
  {"name": "VIS",  "level": "surface", "alias": "vis_sfc"}
]
```

### 8.4 Pressure-level selectors for synoptic/regime features

Request these for global and regional pressure-level models where available:

```json
[
  {"name": "TMP",  "level": "925 mb", "alias": "tmp_925"},
  {"name": "TMP",  "level": "850 mb", "alias": "tmp_850"},
  {"name": "TMP",  "level": "700 mb", "alias": "tmp_700"},
  {"name": "HGT",  "level": "925 mb", "alias": "hgt_925"},
  {"name": "HGT",  "level": "850 mb", "alias": "hgt_850"},
  {"name": "HGT",  "level": "500 mb", "alias": "hgt_500"},
  {"name": "RH",   "level": "925 mb", "alias": "rh_925"},
  {"name": "RH",   "level": "850 mb", "alias": "rh_850"},
  {"name": "UGRD", "level": "925 mb", "alias": "u_925"},
  {"name": "VGRD", "level": "925 mb", "alias": "v_925"},
  {"name": "UGRD", "level": "850 mb", "alias": "u_850"},
  {"name": "VGRD", "level": "850 mb", "alias": "v_850"}
]
```

### 8.5 ECMWF/IFS-style selector aliases

For `ifsoper`, `ifsenfo`, `aifsoper`, and `aifsenfo`, GribStream may expose ECMWF-style names. Request these wherever available:

```json
[
  {"name": "2t",   "level": "sfc", "alias": "tmp_2m"},
  {"name": "2d",   "level": "sfc", "alias": "dpt_2m"},
  {"name": "10u",  "level": "sfc", "alias": "u10"},
  {"name": "10v",  "level": "sfc", "alias": "v10"},
  {"name": "tcc",  "level": "sfc", "alias": "tcc"},
  {"name": "tp",   "level": "sfc", "alias": "tp"},
  {"name": "cp",   "level": "sfc", "alias": "cp"},
  {"name": "ssrd", "level": "sfc", "alias": "ssrd"},
  {"name": "msl",  "level": "sfc", "alias": "msl"},
  {"name": "sp",   "level": "sfc", "alias": "sp"},
  {"name": "cape", "level": "sfc", "alias": "cape_sfc"},
  {"name": "t",    "level": "pl 925", "alias": "tmp_925"},
  {"name": "t",    "level": "pl 850", "alias": "tmp_850"},
  {"name": "u",    "level": "pl 925", "alias": "u_925"},
  {"name": "v",    "level": "pl 925", "alias": "v_925"},
  {"name": "u",    "level": "pl 850", "alias": "u_850"},
  {"name": "v",    "level": "pl 850", "alias": "v_850"},
  {"name": "q",    "level": "pl 925", "alias": "q_925"},
  {"name": "q",    "level": "pl 850", "alias": "q_850"}
]
```

### 8.6 NBM-specific selectors

For `nbm`, request at minimum:

```json
[
  {"name": "TMP",   "level": "2 m above ground", "info": "", "alias": "tmp_2m"},
  {"name": "TMP",   "level": "2 m above ground", "info": "ens std dev", "alias": "tmp_2m_ens_std"},
  {"name": "TMAX",  "level": "2 m above ground", "info": "", "alias": "tmax_2m"},
  {"name": "TMAX",  "level": "2 m above ground", "info": "ens std dev", "alias": "tmax_2m_ens_std"},
  {"name": "TMIN",  "level": "2 m above ground", "info": "", "alias": "tmin_2m"},
  {"name": "DPT",   "level": "2 m above ground", "info": "", "alias": "dpt_2m"},
  {"name": "TCDC",  "level": "surface", "info": "", "alias": "tcdc_sfc"},
  {"name": "DSWRF", "level": "surface", "info": "", "alias": "dswrf_sfc"}
]
```

### 8.7 NBMQMD percentile selectors

For `nbmqmd`, request percentile max-temperature distribution fields wherever available. The required percentiles are:

```text
1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
55, 60, 65, 70, 75, 80, 85, 90, 95, 99
```

For each percentile `p`, request:

```json
{"name": "TMP", "level": "2 m above ground", "info": "{p}% level | max-18h", "alias": "tmp_2m_p{p}_max18h"}
```

Also request min-18h percentiles only if they are needed for low/diurnal-range features:

```json
{"name": "TMP", "level": "2 m above ground", "info": "{p}% level | min-18h", "alias": "tmp_2m_p{p}_min18h"}
```

## 9. Ensemble members

For ensemble models, Codex must fetch member-level data when cost allows.

Mandatory member behavior:

```text
gefsatmos: fetch all available members, expected control + 30 perturbations.
ifsenfo: fetch all available members, expected control + 50 perturbed members.
aifsenfo: fetch all available members if account supports it.
aigefssfc: fetch all available members if account supports it.
```

If member-level data is too costly, fetch:

```text
1. ensemble mean where available,
2. spread/standard-deviation fields where available,
3. a deterministic subset of members: control plus members 1,2,3,4,5,10,15,20,25,30.
```

The subset is an emergency cost fallback only. Full members are preferred for probability calibration.

## 10. Request body templates

### 10.1 `/runs` template for deterministic model

```json
{
  "forecastedFrom": "{run_start_utc_iso}",
  "forecastedUntil": "{run_end_utc_iso}",
  "timesList": ["{valid_time_1_iso}", "{valid_time_2_iso}"],
  "minLeadTime": "0h",
  "maxLeadTime": "72h",
  "coordinates": [
    {"lat": 40.77945, "lon": -73.88027, "name": "GP_KLGA_EXACT"}
  ],
  "variables": [
    {"name": "TMP", "level": "2 m above ground", "alias": "tmp_2m"},
    {"name": "TMAX", "level": "2 m above ground", "alias": "tmax_2m"}
  ]
}
```

### 10.2 `/timeseries` template with conservative asOf

Use only after computing:

```text
asOf_for_model = cutoff_utc - source_specific_publication_buffer
```

Example:

```json
{
  "timesList": ["{valid_time_1_iso}", "{valid_time_2_iso}"],
  "asOf": "{cutoff_minus_buffer_utc_iso}",
  "minLeadTime": "0h",
  "maxLeadTime": "72h",
  "coordinates": [
    {"lat": 40.77945, "lon": -73.88027, "name": "GP_KLGA_EXACT"}
  ],
  "variables": [
    {"name": "TMP", "level": "2 m above ground", "alias": "tmp_2m"}
  ]
}
```

### 10.3 Expressions to reduce payload

Use GribStream expressions only for simple reductions that reduce payload without hiding needed raw data. Allowed expressions:

```text
wind_speed = Hypot(u10, v10)
wind_direction = meteorological direction from u/v
fahrenheit = kelvin_to_fahrenheit(tmp_2m)
```

Do not compute stateful features like rolling bias, analog matching, or multi-model composite inside GribStream.

## 11. Required run cycles by cutoff

Codex must not simply grab latest by `asOf` and ignore explicit run/cycle availability. For every target date/cutoff, enumerate eligible runs and fetch them.

### 11.1 Early alpha cut: `T_MINUS_1_STOCKHOLM_1500`

Expected usable runs before 13:00 UTC:

```text
GFS: latest safely completed 06Z or older, not 12Z.
GEFS: latest safely completed 06Z if availability rule passes, otherwise 00Z.
HRRR/RAP: latest hourly run whose availability rule passes; 12Z may not pass at 13Z.
NBM: latest run whose availability rule passes.
IFS/AIFS/global AI: latest run whose availability rule passes; do not assume 06Z/12Z by run time alone.
MOS: latest product whose parsed issue time + 15m <= cutoff.
RTMA: latest valid analysis with valid_time + 60m <= cutoff.
```

### 11.2 Core T-1 cut: `T_MINUS_1_STOCKHOLM_1915`

Expected usable runs before 17:15 UTC:

```text
HRRR/RAP: several 12Z+ hourly cycles likely eligible.
GFS: 12Z likely eligible after conservative GFS lag; verify through availability ledger.
GEFS: 12Z may not be eligible; use availability ledger.
NBM/MOS: later cycles likely eligible.
IFS/AIFS: use availability ledger only.
```

### 11.3 Late T-1 cut: `T_MINUS_1_STOCKHOLM_2230`

Expected usable runs before 20:30 UTC:

```text
HRRR/RAP: 18Z and later cycles likely eligible.
GFS: 18Z early products may be partially eligible only if source-specific availability passes.
GEFS: 12Z likely eligible; 18Z normally not yet fully eligible.
NBM/MOS: later cycles eligible.
IFS/AIFS: use availability ledger only.
```

### 11.4 Pre-local-day cut: `PRE_LOCAL_DAY_NYC_2350`

Expected usable before 23:50 New York / 03:50 UTC next calendar day:

```text
HRRR/RAP: 00Z cycle likely eligible.
GFS: 00Z early fields may be eligible near 03:45–04:00 UTC only if availability passes; use ledger.
GEFS: 00Z usually not fully eligible by 03:50 UTC.
NBM/MOS: latest available cycle.
RTMA: latest analysis before local midnight, subject to 60m lag.
```

## 12. Silver table schema

```text
CREATE TABLE gribstream_values (
    source_name TEXT NOT NULL DEFAULT 'gribstream',
    model_id TEXT NOT NULL,
    endpoint_type TEXT NOT NULL,                 -- runs or timeseries
    source_product TEXT,
    member TEXT,
    coordinate_tier TEXT NOT NULL,
    grid_point_id TEXT NOT NULL,
    lat DOUBLE PRECISION NOT NULL,
    lon DOUBLE PRECISION NOT NULL,
    forecasted_at_utc TIMESTAMP NOT NULL,
    forecasted_time_utc TIMESTAMP NOT NULL,
    forecast_hour DOUBLE PRECISION NOT NULL,
    variable_name TEXT NOT NULL,
    variable_level TEXT,
    variable_info TEXT,
    alias TEXT,
    unit_original TEXT,
    value_original DOUBLE PRECISION,
    unit_canonical TEXT,
    value_canonical DOUBLE PRECISION,
    provider_available_at_utc TIMESTAMP,
    our_ingested_at_utc TIMESTAMP NOT NULL,
    availability_method TEXT NOT NULL,
    source_request_id TEXT NOT NULL,
    selector_status TEXT NOT NULL DEFAULT 'ok',
    quality_flag TEXT NOT NULL DEFAULT 'ok',
    quality_note TEXT,
    PRIMARY KEY (
        model_id, COALESCE(member,''), grid_point_id, forecasted_at_utc,
        forecasted_time_utc, variable_name, COALESCE(variable_level,''), COALESCE(variable_info,''), source_request_id
    )
);
```

## 13. Derived GribStream features

Codex must generate these feature families for every model where inputs exist.

### 13.1 Direct local temperature features

```text
{model}_KLGA_daily_max_tmp_2m_f
{model}_KLGA_tmax_2m_f
{model}_KLGA_tmp_2m_at_local_12_13_14_15_16_17_18_19_20_21
{model}_KLGA_peak_window_max_tmp_2m_f
{model}_KLGA_peak_window_mean_tmp_2m_f
{model}_KLGA_peak_window_time_of_max_local_hour
```

If both `TMAX` and hourly `TMP` exist:

```text
{model}_tmax_minus_hourly_peak_tmp
```

### 13.2 Coordinate-gradient features

For each model:

```text
{model}_inland_nj_minus_klga_tmp_peak
{model}_newark_corridor_minus_klga_tmp_peak
{model}_long_island_minus_klga_tmp_peak
{model}_sound_water_proxy_minus_klga_tmp_peak
{model}_atlantic_proxy_minus_klga_tmp_peak
{model}_max_neighbor_minus_min_neighbor_tmp_peak
{model}_klga_rank_among_grid_points_by_tmp_peak
```

### 13.3 Wind/marine regime features

```text
{model}_KLGA_peak_window_wind_speed_mean
{model}_KLGA_peak_window_wind_dir_circular_mean
{model}_KLGA_peak_window_easterly_component
{model}_KLGA_peak_window_southerly_component
{model}_sea_breeze_risk_proxy = easterly_or_southeasterly_flow + coastal_cool_gradient
{model}_backdoor_front_proxy = northeast/east low-level wind + cool north/east gradient
```

### 13.4 Cloud/solar/precip bust features

```text
{model}_KLGA_peak_window_cloud_cover_mean
{model}_KLGA_peak_window_cloud_cover_max
{model}_KLGA_peak_window_dswrf_mean
{model}_KLGA_peak_window_precip_total
{model}_KLGA_peak_window_precip_any_indicator
{model}_KLGA_peak_window_cape_max
{model}_KLGA_peak_window_cin_min
{model}_convective_bust_risk_proxy
```

### 13.5 Synoptic features

```text
{model}_KLGA_tmp_925_peak_window_mean
{model}_KLGA_tmp_850_peak_window_mean
{model}_KLGA_925_minus_2m_temperature_relation
{model}_KLGA_850_to_925_lapse_proxy
{model}_KLGA_wind_925_dir_speed
{model}_KLGA_wind_850_dir_speed
{model}_regional_mslp_gradient_proxy
{model}_hgt_500_anomaly_proxy_if_available
```

### 13.6 Run-to-run trend features

For each model and variable:

```text
{model}_latest_minus_previous_cycle_tmax
{model}_latest_minus_24h_prior_run_tmax
{model}_run_to_run_std_last_4_cycles_tmax
{model}_run_to_run_slope_last_4_cycles_tmax
{model}_latest_run_age_hours_at_cutoff
```

### 13.7 Ensemble probability features

For ensemble models, compute member-level daily max and threshold probabilities:

```text
{model}_member_tmax_f[member]
{model}_ensemble_mean_tmax_f
{model}_ensemble_median_tmax_f
{model}_ensemble_p05_p10_p25_p75_p90_p95_tmax_f
{model}_ensemble_spread_std_tmax_f
{model}_ensemble_iqr_tmax_f
{model}_prob_tmax_ge_{k}_f for k = 50..115
{model}_prob_market_bucket_{bucket_id}
```

Use all members when available. If a subset is used, feature names must include `_member_subset` and the member list must be recorded in metadata.

### 13.8 NBM/NBMQMD distribution features

For NBM:

```text
nbm_tmax_2m_f
nbm_tmax_ens_std_f
nbm_tmp_2m_peak_f
nbm_dpt_2m_peak_mean_f
nbm_cloud_solar_precip_features
```

For NBMQMD:

```text
nbmqmd_percentile_curve_max18h_p01...p99
nbmqmd_interpolated_cdf_threshold_probs_50..115F
nbmqmd_distribution_mean_estimate
nbmqmd_distribution_spread_estimate
nbmqmd_distribution_skew_estimate
```

## 14. Unit conversions

Codex must convert all temperature-like fields to Fahrenheit for model features:

```python
F = (K - 273.15) * 9/5 + 32        # Kelvin input
F = C * 9/5 + 32                   # Celsius input
```

Store original units and canonical units. Never overwrite raw values.

Wind:

```text
m/s -> mph = mps * 2.2369362920544
knots -> mph = kt * 1.150779448
```

Pressure:

```text
Pa -> hPa = Pa / 100
```

Precipitation:

```text
kg m^-2 accumulated water equivalent = mm, then inches = mm / 25.4
```

## 15. Availability and leakage rules

GribStream `forecasted_at` is model run time. It is not enough to prove live availability. Codex must set:

```text
provider_available_at_utc = forecasted_at_utc + model_specific_lag
availability_method = conservative_lag_rule
```

unless actual live ingestion logs exist.

Default lags are defined in `00_universal_ingestion_contract_and_availability_ledger.md` and must be applied by model family.

For `/timeseries`, when simulating a cutoff:

```text
asOf = cutoff_utc - model_specific_conservative_buffer
```

For `/runs`, use all rows where:

```text
provider_available_at_utc <= cutoff_utc
```

## 16. Backfill job design

### 16.1 Backfill granularity

Run jobs by:

```text
model_id
coordinate_tier
month
variable_group
endpoint_type
```

Do not run one giant all-model job.

### 16.2 Backfill loop

```python
for model_id in model_groups:
    model_start = confirmed_model_start(model_id)
    for month in months_between(model_start, latest_complete_month):
        for coordinate_tier in ["B"]:
            for variable_group in model_supported_variable_groups:
                build timesList for all target-day valid times needed in that month
                call /runs for run range covering cutoffs and lead times
                write bronze
                parse silver
                update availability ledger
```

### 16.3 Live job loop

At every configured cutoff and hourly between 00Z and 23Z:

```text
1. Fetch candidate latest runs for all core models.
2. Store actual ingestion timestamp.
3. Update availability ledger with actual_ingestion_log.
4. Build cutoff-eligible features.
5. Do not overwrite historical estimates; actual logs supersede them from the date live logging starts.
```

## 17. Source gaps and errors

Codex must write a `source_gap` row for:

```text
model not available for date
selector missing for model
member missing for ensemble
coordinate outside domain
API returns no data
API returns partial data
rate limit exhausted
```

Required schema:

```text
CREATE TABLE source_gaps (
    source_name TEXT,
    source_product TEXT,
    model_id TEXT,
    station_id TEXT,
    grid_point_id TEXT,
    variable_name TEXT,
    variable_level TEXT,
    variable_info TEXT,
    member TEXT,
    start_time_utc TIMESTAMP,
    end_time_utc TIMESTAMP,
    gap_type TEXT,
    gap_reason TEXT,
    first_detected_at_utc TIMESTAMP,
    last_checked_at_utc TIMESTAMP,
    raw_evidence_uri TEXT
);
```

## 18. Acceptance tests

```text
[ ] `/runs` ingestion produces forecasted_at and forecasted_time for every row.
[ ] `/timeseries` ingestion is used only with `asOf = cutoff - buffer` when simulating historical cutoffs.
[ ] GribStream rows with provider_available_at_utc after cutoff are excluded from gold features.
[ ] NBM, HRRR, RAP, GFS, GEFS, and at least one ECMWF/AI source are attempted for KLGA Tier B points.
[ ] Ensemble member rows can be converted into threshold and bucket probabilities.
[ ] Run-to-run trend features change when newer eligible runs arrive at later cutoffs.
[ ] Missing selectors create source_gap rows and do not silently disappear.
[ ] Cost-control mode can run Tier A extraction and produce a valid but reduced feature set.
```
