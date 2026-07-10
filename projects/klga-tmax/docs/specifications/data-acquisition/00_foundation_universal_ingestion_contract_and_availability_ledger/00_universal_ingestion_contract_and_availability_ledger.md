# 00 — Universal Ingestion Contract and Availability Ledger

This file defines the mandatory acquisition contract that every other data-source ingestion task must obey. Codex must implement this contract first. No source-specific ingestion is complete unless its rows can be joined into this contract without ad-hoc assumptions.

## 0. Purpose

The forecasting system is an as-of, multi-cutoff, leakage-safe trading system for Polymarket KLGA daily-high-temperature markets. The system must recreate exactly what information would have been available at each historical trading cutoff. Therefore, every source row must carry explicit timing metadata.

The central rule is:

```text
A feature is eligible for a forecast cutoff only if the system could have known it by that cutoff.
```

The implementation must never use a model initialization time, a valid time, or a provider archive timestamp as a substitute for actual availability unless the source-specific acquisition spec explicitly defines a conservative lag rule.

## 1. Canonical target and time zones

### 1.1 Canonical market target

```text
station_id: KLGA
target_station_name: LaGuardia Airport Station
settlement_source: Wunderground historical daily page/API for KLGA
target_variable: highest temperature recorded during the New York local calendar day
unit: degrees Fahrenheit
precision: whole integer °F
calendar_day_timezone: America/New_York
```

The market target for date `T` is:

```text
Y_T = Wunderground integer °F daily high for KLGA during 00:00:00 through 23:59:59 America/New_York on date T.
```

### 1.2 Required time-zone handling

Codex must implement all date math using IANA time-zone names, never fixed UTC offsets:

```text
TARGET_TZ = "America/New_York"
TRADER_TZ = "Europe/Stockholm"
UTC_TZ = "UTC"
```

The code must use Python `zoneinfo.ZoneInfo` or an equivalent IANA-aware library. Offsets must not be hard-coded because both Stockholm and New York observe daylight saving time, and the offset difference can change by date.

### 1.3 Canonical forecast cutoffs

For every target date `T`, create four forecast snapshots. The exact UTC timestamp must be computed by converting the Stockholm or New York local time through IANA time zones.

| cutoff_id | Local specification | Meaning | Required use |
|---|---|---|---|
| `T_MINUS_1_STOCKHOLM_1500` | 15:00 Europe/Stockholm on date T-1 | Early alpha cut | Always train and run |
| `T_MINUS_1_STOCKHOLM_1915` | 19:15 Europe/Stockholm on date T-1 | Main T-1 cut | Always train and run |
| `T_MINUS_1_STOCKHOLM_2230` | 22:30 Europe/Stockholm on date T-1 | Late T-1 cut | Always train and run |
| `PRE_LOCAL_DAY_NYC_2350` | 23:50 America/New_York on date T-1 | Best pre-KLGA-day cut | Always train and run if market is still open/tradable |

For the 2026-06-28 KLGA market example, these are approximately:

```text
T_MINUS_1_STOCKHOLM_1500  = 2026-06-27 13:00 UTC = 2026-06-27 09:00 America/New_York
T_MINUS_1_STOCKHOLM_1915  = 2026-06-27 17:15 UTC = 2026-06-27 13:15 America/New_York
T_MINUS_1_STOCKHOLM_2230  = 2026-06-27 20:30 UTC = 2026-06-27 16:30 America/New_York
PRE_LOCAL_DAY_NYC_2350    = 2026-06-28 03:50 UTC = 2026-06-27 23:50 America/New_York
```

These example conversions are only examples. The implementation must compute them dynamically.

## 2. Canonical station and coordinate keys

Every row that can be mapped to a station or coordinate must use the following identity fields:

```text
station_id                # e.g. KLGA, KNYC, KEWR
provider_station_id       # exact station id used by source; may be LGA for MOS, KLGA for ASOS/Wunderground
grid_point_id             # if gridded/pseudo-point rather than station observation
lat                       # decimal degrees WGS84
lon                       # decimal degrees WGS84
station_role              # target, nearby_observation, regional_observation, gridded_pseudo_point
```

Station definitions live in `10_station_universe_and_coordinates.md`. Codex must implement one canonical station registry table and every ingestion job must reference that table rather than defining coordinates independently.

## 3. Bronze / silver / gold storage layers

Codex must implement three storage layers.

### 3.1 Bronze layer: immutable raw retrievals

Every external request must be stored in an immutable bronze table or file before parsing.

Required bronze fields:

```text
source_name                 # e.g. gribstream, iem_mos, wunderground, polymarket_clob
source_endpoint             # endpoint URL or connector/action name
request_method              # GET, POST, skill_call, file_download, websocket
request_params_json         # exact query params or body, sorted JSON
request_headers_redacted    # headers with credentials removed
retrieved_at_utc            # system time when response was received
provider_response_timestamp # if provider exposes one; else null
http_status                 # or connector status
response_content_type
response_body_sha256
response_size_bytes
raw_storage_uri             # local path, object store URI, or database blob id
parser_version              # initially null until parsed
source_request_id           # deterministic id = sha256(source_name + endpoint + params + retrieved_at_utc bucket)
```

Bronze data must never be mutated. If the same request is repeated, write another bronze row.

### 3.2 Silver layer: normalized facts

Silver rows are parsed, typed, normalized observations/forecasts/orderbook states.

Required common silver fields:

```text
source_name
source_product
source_model
source_member
source_cycle
run_time_utc                # model run/init/issue time if applicable
valid_time_utc              # time the weather value applies to, if applicable
forecast_hour               # valid_time_utc - run_time_utc, in hours, if applicable
station_id
provider_station_id
grid_point_id
lat
lon
variable_name
variable_level
variable_info
unit_original
value_original
unit_canonical
value_canonical
retrieved_at_utc
provider_available_at_utc
our_ingested_at_utc
availability_method
source_request_id
raw_row_hash
quality_flag
quality_note
```

### 3.3 Gold layer: cutoff-eligible features

Gold rows are model-ready features for one target date and cutoff.

Required gold identity fields:

```text
target_date_local           # YYYY-MM-DD in America/New_York
cutoff_id
cutoff_utc
target_station_id           # always KLGA for first production market system
feature_family              # e.g. gribstream_hrrr, iem_mos, wunderground_actuals
feature_name
feature_value
feature_unit
feature_available           # boolean
source_latest_valid_time_utc
source_latest_run_time_utc
source_age_hours            # cutoff_utc - source_latest_run_time_utc or latest obs time
source_latency_minutes      # our_ingested_at_utc - valid_time_utc or run_time_utc when meaningful
feature_build_version
```

Gold features must be deterministic from silver data and a specific feature-code version.

## 4. Availability ledger

The availability ledger is a required table used by every acquisition and feature job.

### 4.1 Required schema

```text
CREATE TABLE availability_ledger (
    source_name TEXT NOT NULL,
    source_product TEXT NOT NULL,
    source_model TEXT,
    source_member TEXT,
    station_id TEXT,
    grid_point_id TEXT,
    variable_name TEXT,
    variable_level TEXT,
    variable_info TEXT,
    run_time_utc TIMESTAMP,
    valid_time_utc TIMESTAMP,
    forecast_hour NUMERIC,
    provider_nominal_issue_time_utc TIMESTAMP,
    provider_available_at_utc TIMESTAMP,
    our_first_seen_at_utc TIMESTAMP,
    our_latest_seen_at_utc TIMESTAMP,
    availability_method TEXT NOT NULL,
    conservative_lag_minutes INTEGER,
    is_historical_estimate BOOLEAN NOT NULL,
    evidence_uri TEXT,
    evidence_note TEXT,
    PRIMARY KEY (
        source_name, source_product, COALESCE(source_model,''), COALESCE(source_member,''),
        COALESCE(station_id,''), COALESCE(grid_point_id,''), COALESCE(variable_name,''),
        COALESCE(variable_level,''), COALESCE(variable_info,''), COALESCE(run_time_utc,'1900-01-01'),
        COALESCE(valid_time_utc,'1900-01-01')
    )
);
```

### 4.2 Availability method enum

Allowed values:

```text
actual_ingestion_log         # preferred; row was actually seen by our system at this time
provider_metadata            # provider exposes availability timestamp, e.g. Open-Meteo metadata API
provider_production_status   # source such as NCEP production status page
parsed_product_issue_time    # timestamp parsed from a text product such as MOS
conservative_lag_rule        # deterministic lag from run or valid time
delayed_archive_rule         # e.g. NCEI/IEM one-minute archive with 48h lag
manual_backfill_audit        # manually curated for historical exceptions
```

### 4.3 Cutoff eligibility rule

For any candidate source row and forecast cutoff:

```python
eligible = (
    row.our_ingested_at_utc is not None
    and row.our_ingested_at_utc <= cutoff_utc
)
```

If historical `our_ingested_at_utc` is unavailable:

```python
eligible = (
    row.provider_available_at_utc is not None
    and row.provider_available_at_utc <= cutoff_utc
)
```

If provider availability is also unavailable, use source-specific conservative lag:

```python
eligible = row.run_time_utc + lag <= cutoff_utc      # for forecast-run products
eligible = row.valid_time_utc + lag <= cutoff_utc    # for observations/analyses
```

No feature builder may use a row for which `eligible == False`.

## 5. Default conservative lag rules

These lags are only default historical estimates. Live ingestion logs override them after the system starts collecting real-time availability.

| source/product | Conservative historical availability rule | Rationale |
|---|---:|---|
| GribStream HRRR/RAP intermediate cycles | `run_time_utc + 1h45m` | Regional products are often available roughly 1–3h after initialization; use conservative rule. |
| GribStream HRRR/RAP 00/06/12/18 extended cycles | `run_time_utc + 2h15m` | Extended products can complete later; use conservative rule until live logs prove tighter. |
| GribStream GFS | `run_time_utc + 5h15m` for early forecast hours; `+6h00m` for full run | Global products typically appear several hours after initialization. |
| GribStream GEFS | `run_time_utc + 6h45m` | Ensemble products are later than deterministic GFS. |
| GribStream NBM | `run_time_utc + 1h45m` | NBM has frequent updates; use conservative rule until actual logs exist. |
| GribStream IFS / AIFS / global AI deterministic | `run_time_utc + 6h15m` | Global model output availability must not be assumed at initialization time. |
| GribStream IFS ENS / AIFS ENS / AIGEFS | `run_time_utc + 7h00m` | Ensemble outputs are heavier and may appear later. |
| IEM MOS text | parsed product issue time + 15m; fallback `cycle_time + 2h00m` | IEM states MOS products are processed in realtime; use parsed issue time when possible. |
| IEM regular ASOS/METAR | `valid_time_utc + 15m` | IEM ASOS archive syncs from realtime ingest regularly and has request throttling. |
| Synoptic HF-ASOS | `valid_time_utc + 10m` | Synoptic says typical HF-ASOS latency is low; buffer for safety. |
| MADIS OMO | `valid_time_utc + 10m` if directly ingested | MADIS processes current/previous hour repeatedly; add buffer. |
| IEM/NCEI one-minute ASOS archive | `valid_time_utc + 48h` | IEM says this delayed archive is not realtime and can be delayed 18–36h or more. |
| GribStream RTMA | `valid_time_utc + 60m` | Near-real-time analysis; use conservative buffer. |
| GribStream URMA | `valid_time_utc + 8h` | Analysis-of-record style product; not a real-time forecast feature for T-1 cut. |
| Wunderground actuals | `valid_day_end_utc + 24h` for labels; observations during prior days by their API retrieval time if available | Settlement label is not available until after target day; never use target-day values before settlement. |
| Polymarket CLOB snapshots | actual API retrieval time only | Market data must be stored when seen; do not reconstruct orderbook from later data. |

## 6. Target-date materialization

For every date `T` in the training or live universe, Codex must materialize:

```text
target_date_local
local_day_start_utc
local_day_end_utc
cutoff_id
cutoff_utc
settlement_high_f_whole
settlement_high_available_at_utc
```

`settlement_high_f_whole` is null until the Wunderground settled actual has been ingested after the day is complete. Training examples require non-null labels.

## 7. Data quality invariants

Codex must implement automated checks.

### 7.1 Time invariants

```text
valid_time_utc must be timezone-aware UTC.
run_time_utc must be timezone-aware UTC when present.
forecast_hour must equal valid_time_utc - run_time_utc within 1 minute tolerance.
provider_available_at_utc must not be before run_time_utc for forecasts.
our_ingested_at_utc must not be before retrieved_at_utc.
cutoff_utc must be timezone-aware UTC.
```

### 7.2 Value invariants

For temperature features around KLGA:

```text
-60°F <= value <= 120°F for station observed temperature/highs.
180 K <= raw Kelvin temperatures <= 330 K before conversion.
Wind speed must be >= 0.
Cloud cover must be in [0, 100] if percent.
Probability variables must be in [0, 1] or [0, 100] and converted explicitly.
```

### 7.3 Leakage invariants

For every gold feature row:

```text
source_available == True implies source availability timestamp <= cutoff_utc.
For T-1 cutoffs, daily-high error features may use labels only through T-2.
For PRE_LOCAL_DAY_NYC_2350, daily-high error features may still use labels only through T-2 unless T-1 label is fully settled and explicitly available, which is normally false.
No feature may include target-date Wunderground actuals before target day is complete.
No future Polymarket prices may be used in forecast generation.
```

## 8. Source-request reproducibility

Every acquisition function must be deterministic given:

```text
source_config
start_datetime
end_datetime
station_or_coordinate_list
variables
run_cycles
retrieval_time
```

Every run must produce a manifest JSON with:

```text
job_id
source_name
code_version_git_sha
config_hash
started_at_utc
finished_at_utc
row_counts_bronze
row_counts_silver
row_counts_gold
errors
warnings
```

## 9. Backfill strategy

Backfills must run source by source, never by mixing APIs in one job. Required order:

```text
1. station registry
2. Wunderground settled actuals
3. IEM ASOS/METAR observations
4. IEM MOS text/table products
5. GribStream forecast runs
6. GribStream RTMA/URMA analyses
7. Open-Meteo auxiliary run products
8. NOAA raw archive optional fallback products
9. Polymarket historical and live market snapshots
10. availability-ledger reconciliation and feature materialization
```

## 10. Definition of done for this universal layer

This layer is complete only when:

```text
[ ] All canonical cutoffs are generated correctly for at least 30 DST and non-DST test dates.
[ ] The station registry exists and every downstream source references it.
[ ] Bronze, silver, gold, and availability ledger tables exist.
[ ] A gold feature builder refuses to use data whose availability timestamp is after the cutoff.
[ ] Unit tests prove that T-1 daily-high error features cannot use T-1 or T labels.
[ ] Every source-specific job writes a manifest with row counts and warnings.
[ ] Every source-specific row can be traced back to a bronze source_request_id.
```
