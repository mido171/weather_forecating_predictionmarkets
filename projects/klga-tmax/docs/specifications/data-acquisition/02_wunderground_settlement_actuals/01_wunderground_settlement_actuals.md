# 01 — Wunderground Settlement Actuals and Historical Observations

## 1. Purpose

Wunderground is the primary settlement source for the Polymarket KLGA daily-high-temperature markets described by the user. This source must provide the canonical label:

```text
Y_T = Wunderground highest temperature at KLGA during local date T in America/New_York, rounded/stored as the whole integer °F reported by Wunderground.
```

Wunderground data is also used for prior-day and nearby-station actual-weather features, but target-day values must never be used before the target day is complete and the market-resolution data has been published.

## 2. Source access assumption

The user stated that they have an API to fetch Wunderground historicals for any requested station. Because the exact private API shape is not specified in this corpus, Codex must implement a strict adapter interface rather than guessing undocumented endpoints.

The adapter must support these functions exactly:

```python
class WundergroundHistoricalClient:
    def fetch_station_day(
        self,
        station_id: str,
        local_date: date,
        units: Literal["e", "m"] = "e",
    ) -> WundergroundRawDayResponse: ...

    def fetch_station_range(
        self,
        station_id: str,
        start_local_date: date,
        end_local_date: date,
        units: Literal["e", "m"] = "e",
    ) -> Iterable[WundergroundRawDayResponse]: ...
```

Where:

```text
station_id must be the canonical Wunderground station id from station registry, e.g. KLGA.
local_date is the America/New_York calendar date.
units="e" requests imperial/Fahrenheit when the API supports unit selection.
```

The implementation must be configured through environment variables:

```text
WUNDERGROUND_API_BASE_URL
WUNDERGROUND_API_KEY
WUNDERGROUND_API_AUTH_HEADER_NAME      # optional; default "Authorization"
WUNDERGROUND_API_AUTH_HEADER_PREFIX    # optional; default "Bearer "
WUNDERGROUND_API_TIMEOUT_SECONDS       # default 30
WUNDERGROUND_API_MAX_RETRIES           # default 5
WUNDERGROUND_API_RATE_LIMIT_PER_MINUTE # required in production config
```

If the user's API is a local scraper or private connector rather than HTTP, Codex must still wrap it behind the same adapter functions and must still write bronze raw payloads.

## 3. Required stations

Fetch Wunderground historical actuals for every airport station in the canonical registry:

```text
KLGA, KNYC, KJFK, KEWR, KTEB, KHPN, KISP, KFRG, KBDR,
KSWF, KPOU, KMMU, KCDW, KPHL, KBOS, KDCA, KBWI, KALB, KABE
```

Required minimum:

```text
KLGA must be complete.
Nearby core stations must be complete if available from the API.
Regional stations may have gaps but gaps must be recorded explicitly.
```

## 4. Required historical range

Backfill from the earliest date supported by the user's Wunderground API, with a target start date of:

```text
2000-01-01 America/New_York
```

If the API starts later for a given station, store:

```text
first_available_local_date
last_available_local_date
missing_reason = "provider_no_data_before_start"
```

Update daily after each target date has completed and after Wunderground has published at least the first datapoint for the following date.

## 5. Required raw fields to capture

For each station-day response, persist the complete raw response. In the silver layer, parse and store the following if available:

```text
station_id
local_date
timezone_name
observation_timestamp_local
observation_timestamp_utc
temperature_f
dew_point_f
humidity_pct
wind_speed_mph
wind_gust_mph
wind_direction_deg
pressure_in
precipitation_in
condition_text
cloud_cover_text
uv_index
solar_radiation
raw_observation_row_json
```

For daily summary fields, parse and store:

```text
daily_high_f
daily_low_f
daily_avg_temp_f
daily_high_dewpoint_f
daily_low_dewpoint_f
daily_precipitation_in
daily_max_wind_speed_mph
daily_max_wind_gust_mph
daily_avg_wind_speed_mph
daily_dominant_wind_direction_deg
source_daily_summary_json
```

If Wunderground exposes both hourly rows and daily summary rows, both must be stored. The official settlement label is the daily high shown by Wunderground for KLGA, not a recomputation unless no daily field exists.

## 6. Canonical label extraction

For each target date `T`:

1. Fetch Wunderground KLGA data for local date `T`.
2. Extract the displayed/reported daily highest temperature in °F.
3. Store as integer:

```text
settlement_high_f_whole = int(reported_daily_high_f)
```

Do not round a decimal if Wunderground already reports a whole integer. If only hourly values are available, compute:

```python
settlement_high_f_whole = max(int(round_or_parse_whole_degree(temp_f)) for obs in local_day_obs)
```

But mark:

```text
label_method = "computed_from_wunderground_intraday_rows"
```

If a daily summary high is available, mark:

```text
label_method = "wunderground_daily_summary"
```

The preferred label method is `wunderground_daily_summary`.

## 7. Label availability and leakage rule

The KLGA label for target date `T` is not eligible for any forecast cutoff for `T`.

Default label availability:

```text
settlement_high_available_at_utc = local_day_end_utc + 24 hours
```

If the live production system actually observes Wunderground publishing the next-day first datapoint earlier, store the actual observation time in the availability ledger. Historical backtests must use the conservative default unless actual captured publication times exist.

Prior-day actuals are allowed only if their availability timestamp is before the cutoff. At T-1 cutoffs, completed daily labels are normally available through T-2. T-1 daily high is not known at 09:00/13:15/16:30/23:50 New York time on T-1 unless a future market uses a very late cutoff after T-1's daily summary is published, which is not part of the default design.

## 8. Required silver tables

### 8.1 `wu_daily_actuals`

```text
CREATE TABLE wu_daily_actuals (
    station_id TEXT NOT NULL,
    wunderground_station_id TEXT NOT NULL,
    local_date DATE NOT NULL,
    timezone_name TEXT NOT NULL,
    local_day_start_utc TIMESTAMP NOT NULL,
    local_day_end_utc TIMESTAMP NOT NULL,
    daily_high_f INTEGER,
    daily_low_f INTEGER,
    daily_avg_temp_f DOUBLE PRECISION,
    daily_high_dewpoint_f DOUBLE PRECISION,
    daily_low_dewpoint_f DOUBLE PRECISION,
    daily_precipitation_in DOUBLE PRECISION,
    daily_max_wind_speed_mph DOUBLE PRECISION,
    daily_max_wind_gust_mph DOUBLE PRECISION,
    daily_avg_wind_speed_mph DOUBLE PRECISION,
    daily_dominant_wind_direction_deg DOUBLE PRECISION,
    label_method TEXT,
    provider_available_at_utc TIMESTAMP,
    our_ingested_at_utc TIMESTAMP NOT NULL,
    source_request_id TEXT NOT NULL,
    quality_flag TEXT NOT NULL DEFAULT 'ok',
    quality_note TEXT,
    PRIMARY KEY (station_id, local_date)
);
```

### 8.2 `wu_intraday_observations`

```text
CREATE TABLE wu_intraday_observations (
    station_id TEXT NOT NULL,
    wunderground_station_id TEXT NOT NULL,
    observation_time_local TIMESTAMP NOT NULL,
    observation_time_utc TIMESTAMP NOT NULL,
    local_date DATE NOT NULL,
    temp_f DOUBLE PRECISION,
    dewpoint_f DOUBLE PRECISION,
    humidity_pct DOUBLE PRECISION,
    wind_speed_mph DOUBLE PRECISION,
    wind_gust_mph DOUBLE PRECISION,
    wind_direction_deg DOUBLE PRECISION,
    pressure_in DOUBLE PRECISION,
    precipitation_in DOUBLE PRECISION,
    condition_text TEXT,
    raw_observation_json JSON,
    provider_available_at_utc TIMESTAMP,
    our_ingested_at_utc TIMESTAMP NOT NULL,
    source_request_id TEXT NOT NULL,
    quality_flag TEXT NOT NULL DEFAULT 'ok',
    quality_note TEXT,
    PRIMARY KEY (station_id, observation_time_utc, source_request_id)
);
```

## 9. Derived features from Wunderground actuals

For each `target_date_local` and cutoff, build only features whose source dates are availability-eligible.

### 9.1 Target-station historical features

```text
KLGA daily_high_f for T-2, T-3, T-4, T-5, T-7, T-14
rolling mean daily_high_f over previous 3/7/14/30 eligible days
rolling max daily_high_f over previous 3/7/14/30 eligible days
rolling min daily_high_f over previous 3/7/14/30 eligible days
rolling standard deviation of daily_high_f over previous 7/14/30 days
previous eligible day daily_low_f
previous eligible day daily_high_dewpoint_f
previous eligible day daily_precipitation_in
previous eligible day daily_max_wind_gust_mph
```

### 9.2 Nearby-station gradient features

For each eligible prior day:

```text
KEWR_high_minus_KLGA_high
KNYC_high_minus_KLGA_high
KJFK_high_minus_KLGA_high
KTEB_high_minus_KLGA_high
KISP_high_minus_KLGA_high
KBDR_high_minus_KLGA_high
mean(INLAND_HOT_REFERENCE_STATIONS high) - KLGA high
mean(COASTAL_MARINE_STATIONS high) - KLGA high
mean(UPSTREAM_SOUTHWEST_STATIONS high) - KLGA high
mean(BACKDOOR_FRONT_STATIONS high) - KLGA high
```

### 9.3 Air-mass persistence features

```text
max_high_last_3_days_core_nyc
max_high_last_7_days_core_nyc
mean_dewpoint_last_3_days_core_nyc
precip_any_last_3_days_core_nyc
heatwave_indicator = KLGA high >= 90°F for at least 2 of previous 3 eligible days
marine_cooling_indicator = KLGA high at least 3°F below KEWR and KTEB on previous eligible day
```

### 9.4 Intraday pre-cutoff features

For cutoff on T-1 or just before local day start, use Wunderground observations only through availability-eligible observation timestamps:

```text
latest_KLGA_temp_f
latest_KLGA_dewpoint_f
latest_KLGA_wind_dir_deg
latest_KLGA_wind_speed_mph
latest_KEWR_temp_f
latest_KJFK_temp_f
latest_KNYC_temp_f
KLGA_minus_KEWR_latest_temp_f
KLGA_minus_KJFK_latest_temp_f
KLGA_minus_KNYC_latest_temp_f
warming_rate_KLGA_last_3_hours
warming_rate_core_nyc_last_6_hours
wind_shift_KLGA_last_6_hours
```

If the Wunderground API has delayed intraday observations, these features will be missing for the cutoff. Do not fill them from future observations.

## 10. Quality controls

### 10.1 Range checks

```text
-40 <= daily_low_f <= 110
-30 <= daily_high_f <= 120
daily_high_f >= daily_low_f
0 <= humidity_pct <= 100
0 <= wind_speed_mph <= 150
0 <= precipitation_in <= 20
```

### 10.2 Cross-source checks

After IEM ASOS and Wunderground are both ingested, compute:

```text
wu_KLGA_daily_high_f - iem_KLGA_daily_high_f
```

Flag if absolute difference is >= 2°F. Do not overwrite Wunderground labels with IEM labels. The comparison is used only for settlement-source reconciliation.

### 10.3 Duplicate and revision handling

If Wunderground revises a completed day:

```text
Store new bronze response.
Update silver current record only if the retrieval is before the market's no-revision cutoff when known.
Keep a revision history table with previous value, new value, detected_at_utc, and raw payload hashes.
```

Required revision table:

```text
CREATE TABLE wu_daily_actual_revisions (
    station_id TEXT,
    local_date DATE,
    previous_daily_high_f INTEGER,
    new_daily_high_f INTEGER,
    previous_source_request_id TEXT,
    new_source_request_id TEXT,
    detected_at_utc TIMESTAMP,
    note TEXT
);
```

## 11. Acquisition schedule

### 11.1 Historical backfill

Run:

```text
for each station in station_registry where station_id is not a gridded pseudo-point:
    for each local_date from 2000-01-01 through latest_complete_local_date:
        fetch_station_day(station_id, local_date, units="e")
        write bronze
        parse silver daily and intraday rows
        write availability ledger
```

Batch requests by range if the private API supports it, but bronze records must preserve exact request ranges and response payloads.

### 11.2 Live daily refresh

For each station:

```text
Every 10 minutes from 00:00 to 02:00 America/New_York after a market day ends:
    fetch prior local date
    compare with current silver
    store revisions if changed

Daily at 10:00 America/New_York:
    fetch prior 3 local dates for revision audit
```

## 12. Acceptance tests

```text
[ ] KLGA Wunderground daily highs exist for all dates where the API provides data.
[ ] `settlement_high_f_whole` is an integer and always equals the Wunderground daily high field when present.
[ ] Target-date Wunderground label is never eligible for its own forecast features.
[ ] At T-1 cutoffs, previous daily-high error features use labels only through T-2 by default.
[ ] Nearby-station gradients are null only when one of the required stations is genuinely missing.
[ ] Every Wunderground row has a bronze source_request_id and raw payload hash.
[ ] Re-fetching a date with revised data creates a revision-history row.
```
