# 04 — ASOS/METAR, HF-ASOS, and One-Minute Observation Acquisition

## 1. Purpose

Observation data is used for:

```text
current-state correction at forecast cutoffs,
recent air-mass and dew-point features,
station-gradient features,
model bias diagnostics,
Wunderground settlement reconciliation,
minute-data historical diagnostics.
```

Observation data is not the primary forecast source for day-ahead Tmax, but it is critical for local KLGA regime recognition and for avoiding obvious model-current-state errors.

Official source pages:

```text
IEM ASOS download portal:        https://mesonet.agron.iastate.edu/request/download.phtml
IEM ASOS API help:               https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help=
IEM ASOS network/minute archive: https://mesonet.agron.iastate.edu/ASOS/
IEM 1-minute download:           https://mesonet.agron.iastate.edu/request/asos/1min.phtml
IEM ASOS wagering note:          https://mesonet.agron.iastate.edu/onsite/news.phtml?id=1469
MADIS one-minute OMO:            https://madis.ncep.noaa.gov/madis_OMO.shtml
Synoptic HF-ASOS docs:           https://docs.synopticdata.com/services/high-frequency-asos
NCEI ASOS/AWOS overview:         https://www.ncei.noaa.gov/products/land-based-station/automated-surface-weather-observing-systems
```

## 2. Source roles

### 2.1 IEM regular ASOS/METAR archive

Use for long-history station observations and live-ish hourly/sub-hourly observation features. IEM exposes `asos.py` with station, time range, data variables, and report-type filters.

This is mandatory.

### 2.2 Synoptic HF-ASOS or MADIS OMO low-latency feed

Use if the user has access. HF-ASOS can provide minute-level current state with low latency, but it is experimental and can have outages.

This is optional for first production, mandatory audition if credentials are available.

### 2.3 IEM/NCEI one-minute delayed archive

Use for historical diagnostics, settlement-source reconciliation, intraday high behavior, and station microstructure. Do not use it as if it were available at T-1 cutoffs because IEM states this dataset is not realtime and is delayed by 18–36 hours or more.

This is mandatory for research/diagnostics but not mandatory for the first live forecasting feature set.

## 3. Required stations

Use all non-pseudo-point stations in `10_station_universe_and_coordinates.md`:

```text
KLGA, KNYC, KJFK, KEWR, KTEB, KHPN, KISP, KFRG, KBDR,
KSWF, KPOU, KMMU, KCDW, KPHL, KBOS, KDCA, KBWI, KALB, KABE
```

IEM uses three-character ASOS ids, so map:

```text
KLGA -> LGA
KNYC -> NYC
KJFK -> JFK
KEWR -> EWR
...
```

## 4. IEM regular ASOS/METAR acquisition

### 4.1 Endpoint

```text
GET https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py
```

### 4.2 Required query parameters

Use explicit time windows and UTC timezone.

```text
station={IEM_ASOS_ID}             # repeated for each station, e.g. station=LGA&station=NYC
data=tmpf                         # repeated data parameter
data=dwpf
data=relh
data=drct
data=sknt
data=gust
data=p01i
data=alti
data=mslp
data=vsby
data=skyc1
data=skyc2
data=skyc3
data=skyc4
data=skyl1
data=skyl2
data=skyl3
data=skyl4
data=wxcodes
data=feel
data=metar
sts={YYYY-MM-DDTHH:MM:SSZ}
ets={YYYY-MM-DDTHH:MM:SSZ}
tz=UTC
format=onlycomma
latlon=yes
elev=yes
```

Request URL example:

```text
https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=LGA&station=NYC&data=tmpf&data=dwpf&data=relh&data=drct&data=sknt&data=gust&data=p01i&data=alti&data=mslp&data=vsby&data=skyc1&data=skyc2&data=skyc3&data=skyc4&data=skyl1&data=skyl2&data=skyl3&data=skyl4&data=wxcodes&data=feel&data=metar&sts=2024-06-01T00:00:00Z&ets=2024-06-02T00:00:00Z&tz=UTC&format=onlycomma&latlon=yes&elev=yes
```

### 4.3 Report types

For historical research, request all reports unless API limits require subdivision. For live/current features, prioritize routine and specials:

```text
report_type=3   # Routine hourly METAR
report_type=4   # Specials
report_type=1   # HFMETAR, when available
```

If the API supports multiple report types in one request, include all three. If not, fetch separate requests and deduplicate by station/time/METAR text.

### 4.4 Rate limits

IEM documentation states the API has an IP-based rate limit and a 1-second per-IP throttle. Codex must enforce:

```text
minimum_delay_between_iem_requests_seconds = 1.1
max_station_years_per_request = 1000
max_retries = 5
retry_backoff = exponential with jitter
```

If HTTP 422 occurs, split by shorter date ranges or fewer stations. If HTTP 503 occurs, retry with backoff.

## 5. IEM one-minute delayed archive acquisition

### 5.1 Use

Use one-minute delayed data for:

```text
historical daily max diagnostics,
comparison to Wunderground highs,
understanding intraday timing of KLGA highs,
settlement-source mismatch probability,
station-specific microclimate and sensor behavior,
research features based on prior days only.
```

Do not use target-date or T-1 one-minute data as live features unless the availability timestamp is before the cutoff. For IEM/NCEI delayed archive, default availability is:

```text
provider_available_at_utc = valid_time_utc + 48 hours
```

### 5.2 Endpoint

Use IEM one-minute download interface or its backend if available in the environment:

```text
https://mesonet.agron.iastate.edu/request/asos/1min.phtml
```

Codex must inspect the form/backend parameters during implementation if the backend URL is not directly visible. The required semantic request is:

```text
station = IEM_ASOS_ID
start = UTC start timestamp
end = UTC end timestamp
format = CSV/plain text if selectable
```

The bronze layer must preserve the raw downloaded file.

### 5.3 Required fields

Parse every available field, but at minimum:

```text
station_id
valid_time_utc
temperature_f or temperature_c if provided
dewpoint_f or dewpoint_c if provided
wind_direction_deg
wind_speed_kt or mph
wind_gust_kt or mph
altimeter
visibility
precipitation fields if available
raw_line
```

### 5.4 Warning about max/min differences

IEM notes that manual max/min calculations from one-minute data may not match official daily max/min because official daily summaries use different processing logic, such as DSM and 6-hour METAR max/min values. Therefore:

```text
one-minute-derived max is never the official settlement label.
one-minute-derived max is a diagnostic feature and reconciliation input.
```

## 6. Synoptic HF-ASOS / MADIS OMO acquisition

### 6.1 Use

If credentials and source access exist, acquire low-latency HF-ASOS/MADIS OMO for real-time cutoff-state features.

### 6.2 Required behavior

Codex must implement this as a separate provider module:

```python
class HighFrequencyAsosClient:
    def fetch_latest(station_ids: list[str], end_time_utc: datetime) -> list[Observation]: ...
    def fetch_range(station_ids: list[str], start_utc: datetime, end_utc: datetime) -> list[Observation]: ...
```

Configuration:

```text
HF_ASOS_PROVIDER=synoptic|madis|disabled
SYNOPTIC_API_TOKEN
SYNOPTIC_API_BASE_URL
MADIS_API_BASE_URL
MADIS_API_TOKEN_OR_CREDENTIALS
HF_ASOS_TIMEOUT_SECONDS=30
HF_ASOS_MAX_RETRIES=5
```

### 6.3 Availability rule

For live HF-ASOS:

```text
provider_available_at_utc = valid_time_utc + 10 minutes
availability_method = conservative_lag_rule unless provider timestamp exists
```

If the provider exposes actual received time, use that.

### 6.4 Data caveats

HF-ASOS precision may differ from traditional METAR and may have outages. Every feature must include:

```text
hf_asos_available_flag
hf_asos_source_age_minutes
hf_asos_provider
```

If HF-ASOS is missing, the model falls back to regular METAR/IEM/Wunderground observations.

## 7. Silver schemas

### 7.1 `asos_metar_observations`

```text
CREATE TABLE asos_metar_observations (
    source_name TEXT NOT NULL,                  -- iem_asos, synoptic_hf_asos, madis_omo
    station_id TEXT NOT NULL,
    provider_station_id TEXT NOT NULL,
    observation_time_utc TIMESTAMP NOT NULL,
    observation_time_local TIMESTAMP,
    report_type TEXT,
    temp_f DOUBLE PRECISION,
    dewpoint_f DOUBLE PRECISION,
    relative_humidity_pct DOUBLE PRECISION,
    wind_direction_deg DOUBLE PRECISION,
    wind_speed_kt DOUBLE PRECISION,
    wind_speed_mph DOUBLE PRECISION,
    wind_gust_kt DOUBLE PRECISION,
    wind_gust_mph DOUBLE PRECISION,
    precip_1h_in DOUBLE PRECISION,
    altimeter_in DOUBLE PRECISION,
    mslp_mb DOUBLE PRECISION,
    visibility_mi DOUBLE PRECISION,
    skyc1 TEXT,
    skyc2 TEXT,
    skyc3 TEXT,
    skyc4 TEXT,
    skyl1_ft INTEGER,
    skyl2_ft INTEGER,
    skyl3_ft INTEGER,
    skyl4_ft INTEGER,
    weather_codes TEXT,
    feels_like_f DOUBLE PRECISION,
    raw_metar TEXT,
    raw_row_json JSON,
    provider_available_at_utc TIMESTAMP,
    our_ingested_at_utc TIMESTAMP NOT NULL,
    availability_method TEXT NOT NULL,
    source_request_id TEXT NOT NULL,
    quality_flag TEXT NOT NULL DEFAULT 'ok',
    quality_note TEXT,
    PRIMARY KEY (source_name, station_id, observation_time_utc, COALESCE(raw_metar,''), source_request_id)
);
```

### 7.2 `asos_one_minute_observations`

```text
CREATE TABLE asos_one_minute_observations (
    source_name TEXT NOT NULL DEFAULT 'iem_one_minute_asos',
    station_id TEXT NOT NULL,
    provider_station_id TEXT NOT NULL,
    observation_time_utc TIMESTAMP NOT NULL,
    temp_f DOUBLE PRECISION,
    dewpoint_f DOUBLE PRECISION,
    wind_direction_deg DOUBLE PRECISION,
    wind_speed_mph DOUBLE PRECISION,
    wind_gust_mph DOUBLE PRECISION,
    pressure_in DOUBLE PRECISION,
    raw_line TEXT,
    provider_available_at_utc TIMESTAMP NOT NULL,
    our_ingested_at_utc TIMESTAMP NOT NULL,
    availability_method TEXT NOT NULL DEFAULT 'delayed_archive_rule',
    source_request_id TEXT NOT NULL,
    quality_flag TEXT NOT NULL DEFAULT 'ok',
    quality_note TEXT,
    PRIMARY KEY (station_id, observation_time_utc, source_request_id)
);
```

## 8. Derived features

### 8.1 Current-state features at cutoff

For each cutoff and station:

```text
latest_obs_temp_f
latest_obs_dewpoint_f
latest_obs_relative_humidity_pct
latest_obs_wind_direction_deg
latest_obs_wind_speed_mph
latest_obs_wind_gust_mph
latest_obs_pressure_mb
latest_obs_cloud_category
latest_obs_lowest_cloud_base_ft
latest_obs_precip_1h_in
latest_obs_age_minutes
```

Use latest eligible observation at or before:

```text
cutoff_utc - source_safety_buffer
```

The buffer is implicit in provider availability timestamps; no future observation may enter.

### 8.2 Warming-rate features

For KLGA and nearby core stations:

```text
temp_change_last_1h
temp_change_last_3h
temp_change_last_6h
dewpoint_change_last_3h
pressure_change_last_3h
wind_direction_change_last_3h_circular
wind_speed_change_last_3h
```

### 8.3 Gradient features

At the latest common eligible observation time or nearest observations within 30 minutes:

```text
KEWR_temp_minus_KLGA_temp
KNYC_temp_minus_KLGA_temp
KJFK_temp_minus_KLGA_temp
KTEB_temp_minus_KLGA_temp
KISP_temp_minus_KLGA_temp
inland_mean_temp_minus_KLGA_temp
coastal_mean_temp_minus_KLGA_temp
inland_mean_dewpoint_minus_KLGA_dewpoint
coastal_mean_wind_dir_regime
```

### 8.4 Marine/sea-breeze diagnostics

```text
klga_wind_from_east_or_southeast
jfk_cooler_than_ewr_by_5f_indicator
klga_cooler_than_ewr_by_3f_indicator
coastal_stations_cooling_while_inland_warming_indicator
sound_backdoor_indicator = northeast/east winds at BDR/HPN/KLGA + falling temps
```

### 8.5 Prior-day minute-data diagnostics

Using only one-minute data whose availability timestamp is before cutoff:

```text
prior_eligible_day_KLGA_one_minute_max_f
prior_eligible_day_KLGA_one_minute_time_of_max_local
prior_eligible_day_KLGA_one_minute_minus_wunderground_high
prior_eligible_day_intraday_high_duration_above_90f_minutes
prior_eligible_day_fast_cooling_after_peak_indicator
```

These are diagnostic/features only; they do not define settlement labels.

## 9. Historical backfill and live schedule

### 9.1 Regular ASOS/METAR backfill

```text
start = 2000-01-01 UTC or earliest IEM availability
end = latest complete UTC day
chunk = 31 days per request for all stations or smaller if API returns 422
rate limit = at least 1.1 seconds between requests
```

### 9.2 One-minute backfill

```text
start = 2000-01-01 for KLGA first
then nearby core stations
then regional context stations if storage budget allows
chunk = 7 days per request initially; adjust based on response size
availability = valid_time + 48h
```

### 9.3 Live observation acquisition

At minimum:

```text
Every 5 minutes: fetch latest KLGA, KNYC, KJFK, KEWR, KTEB if using HF-ASOS.
Every 10 minutes: fetch latest regular ASOS/METAR for all core stations.
At each forecast cutoff: force-refresh all core stations before feature generation.
```

## 10. Quality controls

```text
-60°F <= temp_f <= 120°F
-80°F <= dewpoint_f <= 90°F
dewpoint_f <= temp_f + 5°F unless quality_flag="suspect"
0 <= wind_speed_mph <= 150
0 <= wind_gust_mph <= 200
0 <= visibility_mi <= 100
0 <= relative_humidity_pct <= 100
```

Duplicate observations with identical station/time/METAR may be deduplicated in silver but all bronze raw pulls must remain stored.

## 11. Acceptance tests

```text
[ ] Regular IEM ASOS/METAR data can be fetched for KLGA and nearby core stations.
[ ] The one-minute delayed archive is stored with a 48h default availability lag and cannot enter same-day T-1 live features.
[ ] HF-ASOS features are optional and produce availability/missingness flags.
[ ] Latest-observation features are cutoff-specific and differ by cutoff.
[ ] Gradient and sea-breeze features are generated for core stations.
[ ] Wunderground-vs-ASOS daily high reconciliation features can be computed without overwriting Wunderground labels.
```
