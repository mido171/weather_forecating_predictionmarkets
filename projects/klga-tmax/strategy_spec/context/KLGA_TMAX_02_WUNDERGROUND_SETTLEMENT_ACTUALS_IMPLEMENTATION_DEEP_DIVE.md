# KLGA Tmax Task 02 Wunderground Settlement Actuals Implementation Deep Dive

Last updated: 2026-06-29

## Executive Summary

Task 02 is implemented as a production-shaped Wunderground/Weather.com ingestion package for KLGA Tmax settlement actuals. It adds database schema, provider acquisition, parsing, normalized daily and intraday persistence, bronze provenance, availability ledger writes, target-label refreshes, resumable backfill tracking, CLI commands, validation, tests, bounded smoke verification, and the full live all-station historical backfill.

The full Wunderground/Weather.com backfill has now been run from `1973-01-01` through the latest complete New York local date at execution time, `2026-06-27`, for all 19 task-01 WU-fetchable stations. The post-backfill sanity audit distinguishes persisted daily rows from usable daily Tmax labels: final audited coverage is `371,184 / 371,184` station-days with `354,501` usable saved Tmax station-days, `16,683` no-data station-days, `0` failed station-days, and `0` not-fetched station-days. The implementation evidence includes the earlier KLGA 2021-08-01 dry-run, the persisted KLGA 2021-08-01 smoke, the persisted KLGA August 2021 one-window backfill, the completed all-station full historical run, and the 2026-06-29 sanity correction that reclassified null-`daily_high_f` rows as `no_data`.

## Reader Orientation

Task `02_wunderground_settlement_actuals` is implemented under:

```text
C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\klga_tmax\implementation
```

The implementation uses the existing Weather.com historical observations API shape as the Wunderground acquisition path:

```text
GET /v1/location/{locationId}/observations/historical.json
```

For KLGA, the default provider location is:

```text
KLGA:9:US
```

This task now provides the schema, provider client, parser, persistence path, resumable backfill command, coverage tracking, leakage-safe label availability, validation command, tests, smoke evidence, and completed full historical coverage for Wunderground/Weather.com settlement actuals.

The full all-station historical backfill was launched explicitly after implementation and completed successfully. It used `--start-date 1973-01-01`, default latest-complete end-date behavior, `--chunk-days 31`, `--workers 4`, and `--resume`.

## Scope Boundaries

In scope:

- Wunderground/Weather.com historical observations client.
- KLGA and task-01 non-pseudo station support.
- Raw bronze response storage.
- Daily Tmax settlement label normalization.
- Intraday observation normalization.
- Availability ledger writes.
- Target label refresh for KLGA.
- Fetch-window and station-date coverage audit tables.
- CLI commands for smoke, day fetch, backfill, coverage, and validation.
- Unit tests, fixture tests, live smoke, bounded persisted smoke, and documentation.

Out of scope:

- IEM reconciliation.
- ASOS/METAR ingestion.
- Wunderground official settlement publication timestamp discovery.
- Feature matrix construction.
- Model training.
- Trading logic.

## Source-of-Truth Inputs

This implementation is based on:

- `strategy_spec/data_aquisition/02_wunderground_settlement_actuals/01_wunderground_settlement_actuals.md`
- `strategy_spec/context/KLGA_TMAX_WUNDERGROUND_EXISTING_SCRAPER_DEEP_DIVE.md`
- `strategy_spec/context/KLGA_TMAX_00_FOUNDATION_IMPLEMENTATION_DEEP_DIVE.md`
- `strategy_spec/context/KLGA_TMAX_01_STATION_UNIVERSE_IMPLEMENTATION_DEEP_DIVE.md`
- Task-01 station registry constants in `implementation/src/klga_tmax/registry/station_universe.py`
- The live Weather.com historical observations response returned for `KLGA:9:US` on `2021-08-01`

## Requirements-to-Implementation Traceability

| Requirement | Implementation evidence |
|---|---|
| Use existing Weather.com historical observations logic | `providers/wunderground/client.py`, live smoke with `KLGA:9:US`, fixture parser tests. |
| Use `WUNDERGROUND_API_KEY` first and fallback to `WEATHERCOM_API_KEY` | `providers/wunderground/config.py`, `wunderground inspect-config` output. |
| Preserve raw responses in bronze | `providers/wunderground/persistence.py`, `bronze.source_requests`, `bronze.source_records`. |
| Normalize daily Tmax labels | `silver.wu_daily_actuals`, parser tests, August 2021 saved table. |
| Normalize intraday observations | `silver.wu_intraday_observations`, parser tests, 868 distinct August intraday rows. |
| Keep leakage-safe availability timestamps | parser availability rules, availability ledger writes, target materialization patch. |
| Track saved/failed/no-data/not-fetched | `audit.wu_station_date_coverage`, coverage CLI evidence. `saved` now means `daily_high_f IS NOT NULL`; daily rows that exist but lack a usable Tmax label are classified as `no_data`. |
| Support resumable windows | `audit.wu_fetch_windows`, `backfill.py`, `--resume` command. |
| Add validation | `validate wunderground`, final validation evidence. |
| Execute full all-station historical backfill | Job `wu_backfill_20260628T154604Z`, coverage `371,184 / 371,184`, failed rows `0`. |
| Patch provider edge cases found during live run | Sentinel/physically impossible parser values are nulled; Weather.com `NDF-0001` no-data responses are tracked as `no_data`, not failed; null daily-high rows are not counted as saved Tmax coverage. |
| Patch foundation validation after Task 03 cutoff addition | `validation/foundation.py` now includes `T_MINUS_1_2045UTC`; regression test keeps the validator map aligned with canonical cutoffs. |
| Update contract inspection | `db inspect-contract`, 28 tables and 30 indexes checked. |

## Change Inventory

Code and migrations:

- `implementation/alembic/versions/0003_wunderground_settlement_actuals.py`
- `implementation/alembic/versions/0004_wunderground_intraday_identity.py`
- `implementation/src/klga_tmax/cli.py`
- `implementation/src/klga_tmax/db/migrations_check.py`
- `implementation/src/klga_tmax/db/models.py`
- `implementation/src/klga_tmax/providers/__init__.py`
- `implementation/src/klga_tmax/providers/wunderground/__init__.py`
- `implementation/src/klga_tmax/providers/wunderground/backfill.py`
- `implementation/src/klga_tmax/providers/wunderground/client.py`
- `implementation/src/klga_tmax/providers/wunderground/config.py`
- `implementation/src/klga_tmax/providers/wunderground/models.py`
- `implementation/src/klga_tmax/providers/wunderground/parser.py`
- `implementation/src/klga_tmax/providers/wunderground/persistence.py`
- `implementation/src/klga_tmax/registry/materialize_targets.py`
- `implementation/src/klga_tmax/validation/wunderground.py`
- `implementation/src/klga_tmax/validation/foundation.py`

Tests and fixtures:

- `implementation/tests/fixtures/weathercom_historical_observations_fixture.json`
- `implementation/tests/test_cli_config.py`
- `implementation/tests/test_wunderground_backfill.py`
- `implementation/tests/test_wunderground_client.py`
- `implementation/tests/test_wunderground_parser.py`
- `implementation/tests/test_wunderground_schema_contract.py`
- `implementation/tests/test_timezones_cutoffs.py`

Documentation and tracking:

- `strategy_spec/context/KLGA_TMAX_02_WUNDERGROUND_SETTLEMENT_ACTUALS_IMPLEMENTATION_DEEP_DIVE.md`
- `strategy_spec/context/KLGA_TMAX_TASK_IMPLEMENTATION_QUEUE.md`
- `strategy_spec/context/KLGA_TMAX_TASK_STATUS_MAP.md`

Repository-relative changed-file coverage used by the documentation quality gate:

- `bootstrap/klga_tmax/implementation/src/klga_tmax/providers/wunderground/backfill.py`
- `bootstrap/klga_tmax/implementation/src/klga_tmax/providers/wunderground/models.py`
- `bootstrap/klga_tmax/implementation/src/klga_tmax/providers/wunderground/parser.py`
- `bootstrap/klga_tmax/implementation/src/klga_tmax/providers/wunderground/persistence.py`
- `bootstrap/klga_tmax/implementation/src/klga_tmax/validation/wunderground.py`
- `bootstrap/klga_tmax/implementation/src/klga_tmax/validation/foundation.py`
- `bootstrap/klga_tmax/implementation/tests/test_wunderground_parser.py`
- `bootstrap/klga_tmax/implementation/tests/test_timezones_cutoffs.py`
- `bootstrap/klga_tmax/strategy_spec/context/KLGA_TMAX_02_WUNDERGROUND_SETTLEMENT_ACTUALS_IMPLEMENTATION_DEEP_DIVE.md`
- `bootstrap/klga_tmax/strategy_spec/context/KLGA_TMAX_TASK_STATUS_MAP.md`
- `bootstrap/klga_tmax/strategy_spec/context/KLGA_TMAX_TASK_IMPLEMENTATION_QUEUE.md`

## File-by-File Deep Dive

### `implementation/alembic/versions/0003_wunderground_settlement_actuals.py`

This migration creates the task-02 persistent schema. It adds normalized daily actuals, normalized intraday observations, daily high revision tracking, fetch-window audit rows, and station-date coverage rows. It also declares the daily station-date identity, intraday station-time identity for fresh databases, quality constraints, label-method constraints, and support indexes used by validation and coverage queries.

### `implementation/alembic/versions/0004_wunderground_intraday_identity.py`

This migration is a local correction migration for databases that received the first version of `0003` before the intraday identity was tightened. It deletes duplicate station-time rows if any exist, drops the older primary key, and recreates `wu_intraday_observations_pkey` as `(station_id, observation_time_utc)`. It keeps the repository safe for fresh installs because `0003` now already creates the correct key, and it keeps the current local database safe because Alembic applies `0004` after `0003`.

### `implementation/src/klga_tmax/providers/wunderground/client.py`

This module owns the provider HTTP boundary. It builds the Weather.com historical observations URL, maps canonical station IDs to Weather.com location IDs, redacts API keys, handles gzip decoding, parses JSON, retries retryable provider failures, honors `Retry-After`, and returns a structured response object even when the provider request fails. It deliberately avoids logging or returning raw credentials.

### `implementation/src/klga_tmax/providers/wunderground/parser.py`

This module owns the Weather.com payload interpretation. It converts epoch observation times into timezone-aware UTC and `America/New_York` local times, normalizes intraday weather fields, groups rows by local date, computes the daily high from `max_temp` or `temp`, and assigns conservative availability timestamps. It is intentionally separate from the client so parser tests can run from fixtures without network access.

The full live backfill exposed provider sentinel values such as impossible daily lows and impossible intraday wind-speed values. The parser now treats physically invalid constrained fields as absent rather than persisting them into checked columns. Daily highs are bounded to `-30..120 F`; daily lows are bounded to `-40..110 F`; intraday temperature is bounded to `-40..130 F`; intraday dew point is bounded to `-80..90 F`; humidity is bounded to `0..100`; wind speed is bounded to `0..150 mph`; hourly precipitation is bounded to `0..20 in`. This keeps raw provider rows available in bronze and `raw_observation_json` while preventing provider sentinels from breaking normalized silver constraints.

### `implementation/src/klga_tmax/providers/wunderground/persistence.py`

This module owns all database writes for Wunderground responses. It writes bronze source request and source record provenance, reuses or revises bronze payload rows through the task-00 revision helper, upserts normalized daily and intraday rows, writes availability ledger rows, updates KLGA target labels, records fetch windows, and updates station-date coverage. The normalized intraday upsert now conflicts on `(station_id, observation_time_utc)` so rerunning different chunks does not duplicate current observations.

The live backfill also exposed Weather.com `400` responses whose payloads contained provider no-data code `NDF-0001`. These are now classified as `no_data` fetch windows when the provider explicitly says no data exists, instead of failed operational windows. This distinction is important: a real HTTP/client failure blocks completion, but a provider no-data period is valid coverage evidence for old station-history gaps. The 2026-06-29 sanity pass also tightened station-date semantics: a persisted daily row with `daily_high_f IS NULL` is retained for provenance, but the station-date coverage row is `no_data`, not `saved`.

### `implementation/src/klga_tmax/providers/wunderground/backfill.py`

This module owns fetch orchestration. It selects task-01 WU-fetchable stations, rejects pseudo-points, chunks inclusive local-date windows, marks planned station-dates as `not_fetched`, skips completed windows when `--resume` is enabled, runs bounded threaded provider fetches, and persists every success or failure in an auditable way.

The full all-station run required two operational hardening changes. First, coverage prefill now inserts planned `not_fetched` rows once per station over the whole selected date range instead of once per 31-day window. Second, worker submission is bounded so the executor does not enqueue thousands of futures at once. These changes keep resume behavior unchanged but reduce startup/database overhead and memory pressure for a 19-station, 53-year run.

### `implementation/src/klga_tmax/cli.py`

This file wires task-02 user operations into the canonical `klga-tmax` CLI. It adds `wunderground inspect-config`, `wunderground smoke`, `wunderground fetch-day`, `wunderground backfill`, `wunderground coverage`, and `validate wunderground`. DB-touching commands use the existing audited command wrapper and canonical `KLGA_DB_URL` behavior.

### `implementation/src/klga_tmax/validation/wunderground.py`

This validator is the task-02 health gate. It combines the shared contract inspection with WU-specific checks for required station mappings, bronze provenance, daily and intraday quality ranges, availability ledger rows, coverage accounting, and target-label leakage conditions. It now fails if a coverage row is marked `saved` while `daily_actual_present=false`, or if normalized intraday `temp_f` / `dewpoint_f` values remain outside the accepted physical bounds.

### `implementation/src/klga_tmax/validation/foundation.py`

The final validation pass found that `validate foundation` still used a fixed 2026-06-28 cutoff example map from before the Task 03 single-cutoff GribStream addition. The canonical cutoff registry now includes `T_MINUS_1_2045UTC`, so the map is now a named `EXPECTED_2026_06_28_CUTOFF_UTC` constant containing all current canonical cutoff IDs. This restores `validate foundation` and makes future omissions easier to test.

### `implementation/src/klga_tmax/registry/materialize_targets.py`

This task touched target materialization because settlement labels must not become visible only because an actual row exists. The materializer now marks a label available only when the target actual has a source availability timestamp at or before the cutoff timestamp.

### `implementation/src/klga_tmax/db/migrations_check.py`

The contract inspector now treats task-02 tables, columns, primary-key indexes, unique indexes, and support indexes as required schema objects. This makes `db inspect-contract` fail if a future migration or manual database change removes WU persistence primitives.

## Public Interfaces and Contracts

Database:

```powershell
$env:KLGA_DB_URL = "postgresql+psycopg://postgres:root@127.0.0.1:5432/klga_tmax_research"
```

Provider credentials:

```powershell
$env:WUNDERGROUND_API_KEY = "set privately in the shell environment"
```

If `WUNDERGROUND_API_KEY` is absent, the implementation falls back to the existing local compatibility variable:

```powershell
$env:WEATHERCOM_API_KEY = "set privately in the shell environment"
```

No provider credential is written to the database docs, CLI output, logs, or source code.

Additional provider settings:

| Environment variable | Default | Purpose |
|---|---:|---|
| `WUNDERGROUND_API_BASE_URL` | `https://api.weather.com` | Base URL for the Weather.com historical observations API. |
| `WUNDERGROUND_API_TIMEOUT_SECONDS` | `30` | Per-request timeout. |
| `WUNDERGROUND_API_MAX_RETRIES` | `5` | Retry count after the first attempt. |
| `WUNDERGROUND_API_RATE_LIMIT_PER_MINUTE` | `60` | Shared process-local request limiter. |
| `WUNDERGROUND_MAX_WORKERS` | `4` | Default backfill worker count. |
| `WUNDERGROUND_CHUNK_DAYS` | `31` | Default request window size. |
| `WUNDERGROUND_INTRADAY_AVAILABLE_LAG_MINUTES` | `90` | Conservative availability lag for intraday observations. |
| `WUNDERGROUND_API_USER_AGENT` | `klga-tmax/0.1 weathercom-historical-observations` | Provider request user agent. |

CLI interfaces:

- `klga-tmax wunderground inspect-config`
- `klga-tmax wunderground smoke`
- `klga-tmax wunderground fetch-day`
- `klga-tmax wunderground backfill`
- `klga-tmax wunderground coverage`
- `klga-tmax validate wunderground`

Exit code behavior follows the existing project contract:

- Missing DB/provider configuration exits with config error code `10`.
- Migration failure exits with migration error code `20`.
- Validation or provider smoke failure exits with validation error code `30`.

## Architecture and Control Flow

Provider acquisition flow:

```text
CLI command
  -> load Wunderground settings from environment
  -> select canonical task-01 station rows
  -> build Weather.com location ID and request URL
  -> throttle through shared RateLimiter
  -> fetch historical observations JSON
  -> return WundergroundRawDayResponse
```

Parsing flow:

```text
WundergroundRawDayResponse
  -> parse observations array
  -> convert valid_time_gmt to UTC and America/New_York local time
  -> normalize intraday weather fields
  -> group rows by local_date
  -> compute daily high, low, wind, dew point, precipitation, and quality flags
  -> assign conservative availability timestamps
```

Persistence flow:

```text
ParsedWundergroundResponse
  -> bronze.source_requests
  -> bronze.source_records
  -> silver.wu_intraday_observations
  -> silver.wu_daily_actuals
  -> silver.wu_daily_actual_revisions when daily high changes
  -> silver.availability_ledger
  -> silver.target_daily_actuals for KLGA
  -> gold.target_instances label refresh for KLGA
  -> audit.wu_fetch_windows
  -> audit.wu_station_date_coverage
```

Validation flow:

```text
validate wunderground
  -> inspect shared DB contract
  -> verify task-01 WU station mappings
  -> verify WU table provenance and quality ranges
  -> verify availability ledger rows
  -> verify target label leakage state
  -> report coverage counts and KLGA contract rows
```

## Schema

Migration `0003_wunderground_settlement_actuals.py` adds five task-02 tables.

### `silver.wu_daily_actuals`

Purpose:

- Canonical normalized daily Wunderground actuals by station and local date.
- Stores the settlement label candidate for KLGA.
- Preserves raw daily source JSON and parser provenance.

Identity:

```sql
PRIMARY KEY (station_id, local_date)
```

Important fields:

- `station_id`
- `wunderground_station_id`
- `weathercom_location_id`
- `local_date`
- `timezone_name`
- `local_day_start_utc`
- `local_day_end_utc`
- `daily_high_f`
- `settlement_high_f_whole`
- `daily_low_f`
- `daily_avg_temp_f`
- `daily_high_dewpoint_f`
- `daily_low_dewpoint_f`
- `daily_precipitation_in`
- `daily_max_wind_speed_mph`
- `daily_max_wind_gust_mph`
- `daily_avg_wind_speed_mph`
- `daily_dominant_wind_direction_deg`
- `label_method`
- `daily_high_source_field`
- `provider_available_at_utc`
- `our_ingested_at_utc`
- `source_request_id`
- `source_record_id`
- `source_daily_summary_json`
- `raw_daily_json`
- `observations_count`
- `quality_flag`
- `quality_note`

Indexes:

- `wu_daily_actuals_pkey`
- `ix_wu_daily_actuals_station_date`
- `ix_wu_daily_actuals_provider_available`

### `silver.wu_intraday_observations`

Purpose:

- Normalized intraday Wunderground/Weather.com observations.
- Stores all fields needed for later intraday feature construction.
- Keeps unavailable fields, such as solar radiation when absent from provider rows, as `NULL`.

Fresh migration identity:

```sql
PRIMARY KEY (station_id, observation_time_utc)
```

Migration `0004_wunderground_intraday_identity.py` exists because the first local application of `0003` keyed intraday rows by `(station_id, observation_time_utc, source_request_id)`. That would duplicate normalized observations when a one-day smoke fetch is followed by a month chunk. Migration `0004` deduplicates any existing duplicates and rekeys the table to station-time identity.

Important fields:

- `station_id`
- `wunderground_station_id`
- `weathercom_location_id`
- `observation_time_local`
- `observation_time_utc`
- `local_date`
- `timezone_name`
- `temp_f`
- `dewpoint_f`
- `humidity_pct`
- `wind_speed_mph`
- `wind_gust_mph`
- `wind_direction_deg`
- `pressure_in`
- `precipitation_in`
- `condition_text`
- `cloud_cover_text`
- `uv_index`
- `solar_radiation`
- `raw_observation_json`
- `provider_available_at_utc`
- `our_ingested_at_utc`
- `source_request_id`
- `source_record_id`
- `quality_flag`
- `quality_note`

Indexes:

- `wu_intraday_observations_pkey`
- `ix_wu_intraday_station_time`
- `ix_wu_intraday_local_date`

### `silver.wu_daily_actual_revisions`

Purpose:

- Records daily high changes observed on refetch.
- Keeps previous and new source request/record IDs for auditability.

Indexes:

- `ix_wu_daily_revisions_station_date`

### `audit.wu_fetch_windows`

Purpose:

- Records every station/date-window fetch attempt.
- Supports resume, failure review, and provider-health audits.

Important fields:

- `job_id`
- `station_id`
- `wunderground_station_id`
- `weathercom_location_id`
- `window_start_date`
- `window_end_date`
- `units`
- `status`
- `attempts`
- `http_status`
- `error_type`
- `error_message`
- `source_request_id`
- `source_record_id`
- `observations_count`
- `daily_rows_upserted`
- `intraday_rows_upserted`
- `started_at_utc`
- `finished_at_utc`

Constraint/index:

- `uq_wu_fetch_windows_job_station_window`
- `ix_wu_fetch_windows_status`

### `audit.wu_station_date_coverage`

Purpose:

- Tracks station-date status as `not_fetched`, `saved`, `no_data`, or `failed`.
- Makes interrupted backfills auditable and resumable.

Identity:

```sql
PRIMARY KEY (station_id, local_date)
```

Indexes:

- `wu_station_date_coverage_pkey`
- `ix_wu_station_date_coverage_status`

## Bronze Provenance

Every provider response is recorded through the existing task-00 bronze tables:

- `bronze.source_requests`
- `bronze.source_records`

For Wunderground:

- `source_name = 'wunderground'`
- `provider_name = 'weathercom'`
- `endpoint_name = 'historical_observations'`
- `parser_version = 'weathercom_historical_observations_v1'`

The request metadata stores:

- canonical station ID
- Wunderground station ID
- Weather.com location ID
- local start/end dates
- units
- redacted endpoint URL
- response content type
- HTTP status
- response hash
- response size

Bronze revision behavior uses the existing task-00 helper:

- Same provider record key and same payload hash returns the existing current source record.
- Same provider record key and changed payload hash creates a new revision and marks the prior current source record non-current.

## Provider Client

Implemented in:

```text
implementation/src/klga_tmax/providers/wunderground/client.py
```

The client uses the standard library `urllib.request` to avoid adding new runtime dependencies.

Request shape:

```text
https://api.weather.com/v1/location/{locationId}/observations/historical.json?apiKey=REDACTED&units=e&startDate=YYYYMMDD&endDate=YYYYMMDD
```

Default station location mapping:

```python
weathercom_location_id("KLGA") == "KLGA:9:US"
```

Retry policy:

- Retries `429`, `500`, `502`, `503`, and `504`.
- Stops immediately for permanent `400`, `401`, `403`, and `404`.
- Honors `Retry-After` when present.
- Uses bounded exponential backoff with jitter otherwise.
- Uses a shared process-local rate limiter across worker threads.

The client returns a structured `WundergroundRawDayResponse` even on provider errors. Failed responses are still persisted into bronze and audit tables.

## Parser

Implemented in:

```text
implementation/src/klga_tmax/providers/wunderground/parser.py
```

The parser reads Weather.com historical observation rows from:

```json
{
  "observations": [...]
}
```

Intraday fields normalized:

- `temp`
- `dewPt`
- `rh`
- `wspd`
- `gust`
- `wdir`
- `pressure`
- `precip_hrly`
- `wx_phrase`
- `terse_phrase`
- `clds`
- `uv_index`
- `solar_radiation`

Daily label rule:

1. Group provider observation rows by `America/New_York` local date.
2. Prefer row-level `max_temp` for daily high candidates.
3. If `max_temp` is absent, use row-level `temp`.
4. Store the selected high as both `daily_high_f` and `settlement_high_f_whole`.
5. Set `label_method = 'computed_from_wunderground_intraday_rows'`.

The current Weather.com endpoint path did not expose a separate daily summary object in the live smoke response, so the implemented path computes daily labels from intraday rows. The schema still allows `label_method = 'wunderground_daily_summary'` for a future parser branch if the provider response supplies a daily summary high.

## Availability And Leakage Rules

Intraday observation availability:

```text
observation_time_utc + WUNDERGROUND_INTRADAY_AVAILABLE_LAG_MINUTES
```

Default:

```text
observation_time_utc + 90 minutes
```

Daily settlement label availability:

```text
America/New_York local day end + 24 hours
```

Example from the live KLGA 2021-08-01 smoke:

```text
local_date = 2021-08-01
provider_available_at_utc = 2021-08-03T04:00:00+00:00
```

This is intentionally conservative. It prevents same-day or next-morning forecast cutoffs from seeing target-day settlement labels unless the configured cutoff is actually after the conservative label availability time.

`implementation/src/klga_tmax/registry/materialize_targets.py` was updated so existing target materialization only marks a label available when:

```sql
actual.high_temp_f IS NOT NULL
AND actual.source_available_at_utc <= :cutoff_utc
```

When WU persistence refreshes existing `gold.target_instances`, it also uses:

```sql
label_available = :settlement_high_available_at_utc <= cutoff_utc
```

Availability ledger entries are written for:

- `wu_daily_actual:{station_id}:{local_date}` with `variable_name = 'daily_high_f'`
- `wu_intraday:{station_id}:{observation_time_utc}` with `variable_name = 'intraday_observation'`

## CLI Command Details

Task 02 adds the `wunderground` command group and one validation command.

Inspect config:

```powershell
python -m klga_tmax.cli wunderground inspect-config
```

Dry-run one station-day:

```powershell
python -m klga_tmax.cli wunderground smoke --station-id KLGA --local-date 2021-08-01 --dry-run
```

Persist one station-day:

```powershell
$env:KLGA_DB_URL = "postgresql+psycopg://postgres:root@127.0.0.1:5432/klga_tmax_research"
python -m klga_tmax.cli wunderground fetch-day --station-id KLGA --local-date 2021-08-01 --persist
```

Run a bounded/resumable backfill:

```powershell
$env:KLGA_DB_URL = "postgresql+psycopg://postgres:root@127.0.0.1:5432/klga_tmax_research"
python -m klga_tmax.cli wunderground backfill --stations KLGA --start-date 2021-08-01 --end-date 2021-08-31 --chunk-days 31 --workers 1 --resume
```

Summarize coverage:

```powershell
$env:KLGA_DB_URL = "postgresql+psycopg://postgres:root@127.0.0.1:5432/klga_tmax_research"
python -m klga_tmax.cli wunderground coverage --station-id KLGA --start-date 2021-08-01 --end-date 2021-08-31
```

Validate task 02:

```powershell
$env:KLGA_DB_URL = "postgresql+psycopg://postgres:root@127.0.0.1:5432/klga_tmax_research"
python -m klga_tmax.cli validate wunderground
```

Full historical command shape used for the completed all-station run:

```powershell
$env:KLGA_DB_URL = "postgresql+psycopg://postgres:root@127.0.0.1:5432/klga_tmax_research"
python -m klga_tmax.cli wunderground backfill --start-date 1973-01-01 --chunk-days 31 --workers 4 --resume
```

## Validation

Implemented in:

```text
implementation/src/klga_tmax/validation/wunderground.py
```

Validation checks:

- Task-00/01 contract inspection still passes.
- Task-02 tables and indexes exist.
- Task-01 registry exposes 19 Wunderground-fetchable non-pseudo stations.
- WU daily rows have bronze source request/record provenance.
- WU intraday rows have bronze source request/record provenance.
- Daily highs are in the accepted physical range.
- Intraday humidity, wind speed, and precipitation are in accepted ranges.
- Daily rows have availability ledger entries.
- `gold.target_instances.label_available` is not true when the settlement high availability is after cutoff.
- Coverage counts are queryable.
- KLGA contract coverage since `2000-01-01` is reported.

## Tests Added

`test_wunderground_client.py`:

- KLGA provider location ID becomes `KLGA:9:US`.
- Explicit Weather.com location IDs are preserved.
- Historical observations URL path and date query are correct.

`test_wunderground_parser.py`:

- Weather.com fixture rows parse into intraday rows and one daily Tmax.
- Daily high prefers `max_temp`.
- Daily label availability is local-day-end plus 24 hours.
- Intraday availability uses the configured lag.
- Absent solar radiation is stored as null.
- Provider sentinel values outside accepted physical bounds become `NULL` in normalized rows.
- Weather.com provider no-data payloads with `NDF-0001` are classified as `no_data`, not failed windows.

`test_wunderground_backfill.py`:

- Station selection accepts KLGA.
- Pseudo-points are rejected for WU fetches.
- Inclusive chunking for August 2021 is deterministic.
- Date counting is inclusive.

`test_wunderground_schema_contract.py`:

- Task-02 tables are included in `db inspect-contract`.
- Task-02 migration declares required tables, primary keys, constraints, and availability fields.
- Required indexes include WU primary/unique/support indexes.
- Target materialization requires actual-label source availability before cutoff.

`test_cli_config.py`:

- `wunderground inspect-config` does not require a DB URL or provider key.

`test_timezones_cutoffs.py`:

- The foundation validator's 2026-06-28 expected cutoff map covers every current canonical cutoff, including `T_MINUS_1_2045UTC`.

## Testing and Verification Evidence

All commands were run from:

```text
C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\klga_tmax\implementation
```

Compile:

```powershell
python -m compileall -q src tests
```

Result:

```text
exit code 0
```

Unit tests:

```powershell
python -m pytest -q
```

Result:

```text
55 passed in 1.69s
```

CLI help:

```powershell
python -m klga_tmax.cli --help
python -m klga_tmax.cli validate --help
```

Result:

```text
exit code 0
validate commands include foundation, station-universe, wunderground
```

Config inspection:

```powershell
python -m klga_tmax.cli wunderground inspect-config
```

Result:

```json
{
  "api_key_present": true,
  "api_key_source": "WEATHERCOM_API_KEY",
  "base_url": "https://api.weather.com",
  "chunk_days": 31,
  "intraday_available_lag_minutes": 90,
  "max_retries": 5,
  "max_workers": 4,
  "rate_limit_per_minute": 60,
  "timeout_seconds": 30,
  "user_agent": "klga-tmax/0.1 weathercom-historical-observations"
}
```

Live dry-run smoke:

```powershell
python -m klga_tmax.cli wunderground smoke --station-id KLGA --local-date 2021-08-01 --dry-run
```

Result:

```json
{
  "daily_actuals": [
    {
      "daily_high_f": 79,
      "label_method": "computed_from_wunderground_intraday_rows",
      "local_date": "2021-08-01",
      "provider_available_at_utc": "2021-08-03T04:00:00+00:00",
      "quality_flag": "ok",
      "settlement_high_f_whole": 79
    }
  ],
  "http_status": 200,
  "intraday_rows": 25,
  "observations_count": 25,
  "ok": true,
  "station_id": "KLGA",
  "weathercom_location_id": "KLGA:9:US"
}
```

Migrations:

```powershell
$env:KLGA_DB_URL = "postgresql+psycopg://postgres:root@127.0.0.1:5432/klga_tmax_research"
python -m klga_tmax.cli db migrate
```

Result:

```text
0003_wu_actuals applied
0004_wu_intraday_identity applied
second migration run was idempotent
row_counts: registry.cutoffs=4, registry.feature_versions=1, registry.station_registry=29, registry.stations=29
```

Contract inspection after the full backfill and foundation-validator patch:

```powershell
$env:KLGA_DB_URL = "postgresql+psycopg://postgres:root@127.0.0.1:5432/klga_tmax_research"
python -m klga_tmax.cli db inspect-contract
```

Result:

```json
{
  "ok": true,
  "failures": [],
  "warnings": [],
  "details": {
    "schemas_checked": 8,
    "tables_checked": 28,
    "indexes_checked": 30,
    "cutoff_rows": 5,
    "feature_version_rows": 1,
    "station_registry_rows": 29,
    "station_rows": 29
  }
}
```

Task 00 validation after task 02 and the Task 03 cutoff addition:

```text
ok=true, failures=[], target_instance_rows=24, late_feature_rows=0
```

Task 01 validation after task 02:

```text
ok=true, mandatory_station_rows=19, mandatory_pseudo_point_rows=10, tier_c_points=25
```

Persisted one-day smoke:

```powershell
$env:KLGA_DB_URL = "postgresql+psycopg://postgres:root@127.0.0.1:5432/klga_tmax_research"
python -m klga_tmax.cli wunderground fetch-day --station-id KLGA --local-date 2021-08-01 --persist
```

Result:

```json
{
  "status": "succeeded",
  "daily_rows_upserted": 1,
  "intraday_rows_upserted": 25,
  "coverage_rows_updated": 1,
  "revisions_inserted": 0,
  "observations_count": 25
}
```

Bounded August 2021 KLGA persisted backfill:

```powershell
$env:KLGA_DB_URL = "postgresql+psycopg://postgres:root@127.0.0.1:5432/klga_tmax_research"
python -m klga_tmax.cli wunderground backfill --stations KLGA --start-date 2021-08-01 --end-date 2021-08-31 --chunk-days 31 --workers 1 --resume
```

Result:

```json
{
  "coverage_rows_updated": 31,
  "daily_rows_upserted": 31,
  "intraday_rows_upserted": 868,
  "job_id": "wu_backfill_20260628T130546Z",
  "ok": true,
  "prepared_not_fetched_rows": 30,
  "revisions_inserted": 0,
  "stations": 1,
  "windows_failed": 0,
  "windows_fetched": 1,
  "windows_no_data": 0,
  "windows_planned": 1,
  "windows_skipped": 0,
  "windows_succeeded": 1
}
```

Coverage:

```powershell
$env:KLGA_DB_URL = "postgresql+psycopg://postgres:root@127.0.0.1:5432/klga_tmax_research"
python -m klga_tmax.cli wunderground coverage --station-id KLGA --start-date 2021-08-01 --end-date 2021-08-31
```

Result:

```json
{
  "ok": true,
  "rows": [
    {
      "rows": 31,
      "station_id": "KLGA",
      "status": "saved"
    }
  ]
}
```

Task 02 validation after August smoke:

```json
{
  "ok": true,
  "failures": [],
  "warnings": [],
  "details": {
    "wu_coverage_rows": 31,
    "wu_coverage_saved_rows": 31,
    "wu_coverage_failed_rows": 0,
    "wu_daily_actual_rows": 31,
    "wu_intraday_observation_rows": 868,
    "wu_klga_contract_rows_since_2000": 31,
    "wunderground_fetchable_stations": 19
  }
}
```

Task 02 validation after the full all-station historical backfill:

```json
{
  "ok": true,
  "failures": [],
  "warnings": [],
  "details": {
    "wu_contract_start_date": "2000-01-01",
    "wu_coverage_rows": 371184,
    "wu_coverage_daily_actual_present_rows": 354501,
    "wu_coverage_saved_rows": 354501,
    "wu_coverage_failed_rows": 0,
    "wu_daily_actual_rows": 358773,
    "wu_intraday_observation_rows": 9405115,
    "wu_klga_contract_rows_since_2000": 9642,
    "wunderground_fetchable_stations": 19
  }
}
```

Full all-station coverage CLI:

```powershell
$env:KLGA_DB_URL = "postgresql+psycopg://postgres:root@127.0.0.1:5432/klga_tmax_research"
python -m klga_tmax.cli wunderground coverage --start-date 1973-01-01 --end-date 2026-06-27
```

Result summary:

```text
ok=true
19 stations reported
saved usable-Tmax station-days=354,501
no_data station-days=16,683
failed station-days=0
not_fetched station-days=0
coverage station-days=371,184
```

Intraday identity check:

```text
KLGA August 2021 intraday_rows = 868
KLGA August 2021 distinct_intraday_rows = 868
```

## KLGA August 2021 Saved Daily Settlement Highs

The following rows were queried from `silver.wu_daily_actuals` after the bounded persisted backfill:

| Local date | Settlement high F |
|---|---:|
| 2021-08-01 | 79 |
| 2021-08-02 | 81 |
| 2021-08-03 | 81 |
| 2021-08-04 | 79 |
| 2021-08-05 | 81 |
| 2021-08-06 | 89 |
| 2021-08-07 | 89 |
| 2021-08-08 | 89 |
| 2021-08-09 | 79 |
| 2021-08-10 | 84 |
| 2021-08-11 | 90 |
| 2021-08-12 | 97 |
| 2021-08-13 | 98 |
| 2021-08-14 | 98 |
| 2021-08-15 | 91 |
| 2021-08-16 | 84 |
| 2021-08-17 | 82 |
| 2021-08-18 | 84 |
| 2021-08-19 | 89 |
| 2021-08-20 | 89 |
| 2021-08-21 | 83 |
| 2021-08-22 | 84 |
| 2021-08-23 | 85 |
| 2021-08-24 | 91 |
| 2021-08-25 | 93 |
| 2021-08-26 | 94 |
| 2021-08-27 | 94 |
| 2021-08-28 | 93 |
| 2021-08-29 | 81 |
| 2021-08-30 | 86 |
| 2021-08-31 | 87 |

## Full All-Station Historical Backfill Evidence

The full live historical Wunderground/Weather.com backfill was launched after the implementation and smoke checks were in place.

Command shape:

```powershell
$env:KLGA_DB_URL = "postgresql+psycopg://postgres:root@127.0.0.1:5432/klga_tmax_research"
$env:WUNDERGROUND_API_RATE_LIMIT_PER_MINUTE = "60"
$env:WUNDERGROUND_MAX_WORKERS = "4"
$env:WUNDERGROUND_CHUNK_DAYS = "31"
python -m klga_tmax.cli wunderground backfill --start-date 1973-01-01 --chunk-days 31 --workers 4 --resume
```

Runtime identity:

```text
job_id=wu_backfill_20260628T154604Z
date range=1973-01-01 through 2026-06-27
stations=19
chunk_days=31
workers=4
rate_limit_per_minute=60
```

Final `audit.wu_fetch_windows` summary for the job:

| Metric | Value |
|---|---:|
| Windows | 9,076 |
| Succeeded windows | 8,754 |
| Provider no-data windows | 322 |
| Failed windows | 0 |
| Observations returned in this job | 7,041,105 |
| Daily rows upserted by this job | 268,755 |
| Intraday rows upserted by this job | 7,041,105 |
| Minimum window date | 1973-01-01 |
| Maximum window date | 2026-06-27 |
| Last finished timestamp | 2026-06-28 23:56:21 Europe/Stockholm |

Final coverage table summary:

| Status | Station-days |
|---|---:|
| saved, usable daily Tmax | 354,501 |
| no_data, including provider no-data and null daily high | 16,683 |
| failed | 0 |
| not_fetched | 0 |
| total | 371,184 |

Final persisted table counts after the full run:

| Table | Rows |
|---|---:|
| `bronze.source_requests` | 12,002 |
| `bronze.source_records` | 12,002 |
| `silver.wu_daily_actuals` | 358,773 |
| `silver.wu_intraday_observations` | 9,405,115 |
| `audit.wu_fetch_windows` | 12,002 |
| `audit.wu_station_date_coverage` | 371,184 |
| `silver.availability_ledger` WU rows | 9,763,888 |

Post-backfill sanity correction applied on 2026-06-29:

| Correction | Rows |
|---|---:|
| Impossible normalized intraday `temp_f` / `dewpoint_f` cells nulled | 52 |
| Daily derived temp/dewpoint summaries recomputed from cleaned intraday rows | 47 |
| Coverage rows reclassified from `saved` to `no_data` because `daily_high_f IS NULL` | 4,272 |

After that correction, `saved` in `audit.wu_station_date_coverage` means a usable daily Tmax label exists. The `silver.wu_daily_actuals` table still retains `4,272` suspect daily rows where provider observations existed but no usable daily high could be computed; those rows keep bronze provenance and raw JSON, but they are not counted as usable settlement-label coverage.

The full-run stderr log was empty:

```text
implementation/run_logs/wunderground_full_backfill_20260628_174601.err.log
length=0
```

The full-run stdout log exists and records the completed command result:

```text
implementation/run_logs/wunderground_full_backfill_20260628_174601.out.log
last_write_time=2026-06-28 23:56:21
```

Station-level final coverage:

| Station | Role | WU ID | IEM | MOS | Lat | Lon | Usable Tmax | Persisted daily rows | Missing-high rows | No data | Failed | Not fetched | Total | Usable % | First usable | Last usable | Daily min/max/avg F | Intraday rows | Suspect intraday rows |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|
| KABE | regional_context | KABE | ABE | ABE | 40.65210 | -75.44080 | 19,490 | 19,492 | 2 | 46 | 0 | 0 | 19,536 | 99.76% | 1973-01-01 | 2026-06-27 | 2/106/63.62 | 540,995 | 1 |
| KALB | regional_context | KALB | ALB | ALB | 42.74720 | -73.79910 | 19,504 | 19,504 | 0 | 32 | 0 | 0 | 19,536 | 99.84% | 1973-01-01 | 2026-06-27 | -2/118/59.32 | 532,415 | 6 |
| KBDR | nearby_core | KBDR | BDR | BDR | 41.16350 | -73.12620 | 19,128 | 19,134 | 6 | 408 | 0 | 0 | 19,536 | 97.91% | 1973-01-01 | 2026-06-27 | 8/120/61.52 | 493,088 | 2 |
| KBOS | regional_context | KBOS | BOS | BOS | 42.36560 | -71.00960 | 19,504 | 19,504 | 0 | 32 | 0 | 0 | 19,536 | 99.84% | 1973-01-01 | 2026-06-27 | 7/103/60.61 | 542,178 | 1 |
| KBWI | regional_context | KBWI | BWI | BWI | 39.17540 | -76.66830 | 19,506 | 19,506 | 0 | 30 | 0 | 0 | 19,536 | 99.85% | 1973-01-01 | 2026-06-27 | 12/106/67.47 | 517,379 | 0 |
| KCDW | regional_context | KCDW | CDW | CDW | 40.87520 | -74.28140 | 14,544 | 16,348 | 1,804 | 4,992 | 0 | 0 | 19,536 | 74.45% | 1981-11-08 | 2026-06-27 | 7/117/64.15 | 393,100 | 11 |
| KDCA | regional_context | KDCA | DCA | DCA | 38.85120 | -77.04020 | 19,504 | 19,504 | 0 | 32 | 0 | 0 | 19,536 | 99.84% | 1973-01-01 | 2026-06-27 | 7/118/68.16 | 541,447 | 2 |
| KEWR | nearby_core | KEWR | EWR | EWR | 40.69250 | -74.16870 | 19,504 | 19,504 | 0 | 32 | 0 | 0 | 19,536 | 99.84% | 1973-01-01 | 2026-06-27 | 5/108/64.83 | 518,643 | 1 |
| KFRG | nearby_core | KFRG | FRG | FRG | 40.72880 | -73.41340 | 14,274 | 15,690 | 1,416 | 5,262 | 0 | 0 | 19,536 | 73.07% | 1980-07-06 | 2026-06-27 | -19/102/62.23 | 397,627 | 21 |
| KHPN | nearby_core | KHPN | HPN | HPN | 41.06700 | -73.70760 | 19,503 | 19,504 | 1 | 33 | 0 | 0 | 19,536 | 99.83% | 1973-01-01 | 2026-06-27 | 5/108/60.99 | 483,701 | 24 |
| KISP | nearby_core | KISP | ISP | ISP | 40.79520 | -73.10020 | 19,503 | 19,503 | 0 | 33 | 0 | 0 | 19,536 | 99.83% | 1973-01-01 | 2026-06-27 | 6/111/61.80 | 516,737 | 12 |
| KJFK | nearby_core | KJFK | JFK | JFK | 40.63980 | -73.77890 | 19,504 | 19,504 | 0 | 32 | 0 | 0 | 19,536 | 99.84% | 1973-01-01 | 2026-06-27 | 8/112/62.63 | 513,926 | 2 |
| KLGA | target | KLGA | LGA | LGA | 40.77945 | -73.88027 | 19,503 | 19,503 | 0 | 33 | 0 | 0 | 19,536 | 99.83% | 1973-01-01 | 2026-06-27 | 8/104/63.69 | 520,519 | 1 |
| KMMU | regional_context | KMMU | MMU | MMU | 40.79940 | -74.41490 | 14,637 | 15,608 | 971 | 4,899 | 0 | 0 | 19,536 | 74.92% | 1983-11-30 | 2026-06-27 | 1/109/62.69 | 401,462 | 104 |
| KNYC | nearby_core | KNYC | NYC | NYC | 40.77898 | -73.96925 | 19,503 | 19,503 | 0 | 33 | 0 | 0 | 19,536 | 99.83% | 1973-01-01 | 2026-06-27 | 8/104/63.69 | 520,519 | 1 |
| KPHL | regional_context | KPHL | PHL | PHL | 39.87190 | -75.24110 | 19,505 | 19,505 | 0 | 31 | 0 | 0 | 19,536 | 99.84% | 1973-01-01 | 2026-06-27 | 6/118/65.73 | 551,682 | 1 |
| KPOU | regional_context | KPOU | POU | POU | 41.62660 | -73.88420 | 19,487 | 19,500 | 13 | 49 | 0 | 0 | 19,536 | 99.75% | 1973-01-01 | 2026-06-27 | 2/108/61.87 | 519,789 | 10 |
| KSWF | regional_context | KSWF | SWF | SWF | 41.50410 | -74.10480 | 19,262 | 19,318 | 56 | 274 | 0 | 0 | 19,536 | 98.60% | 1973-01-01 | 2026-06-27 | -2/117/59.24 | 410,423 | 76 |
| KTEB | nearby_core | KTEB | TEB | TEB | 40.85899 | -74.05600 | 19,136 | 19,139 | 3 | 400 | 0 | 0 | 19,536 | 97.95% | 1973-01-01 | 2026-06-27 | 5/104/64.30 | 489,485 | 9 |

`No data` in the table above includes both provider-declared no-data windows and rows where Weather.com returned observations but no usable daily Tmax could be computed. `Persisted daily rows` is intentionally shown separately because those suspect rows remain useful for audit/debugging but are not usable settlement labels.

Station-level year coverage:

| Station | Years with usable Tmax | Fully usable years | Partial usable years | No usable years | Years with no-data days |
|---|---|---|---|---|---|
| KABE | 1973-2026 | 1973-1998, 2001-2010, 2012-2018, 2022-2023, 2026 | 1999-2000, 2011, 2019-2021, 2024-2025 | - | 1999-2000, 2011, 2019-2021, 2024-2025 |
| KALB | 1973-2026 | 1973-1999, 2001-2019, 2021-2026 | 2000, 2020 | - | 2000, 2020 |
| KBDR | 1973-2026 | 1973-1981, 1983-1997, 1999, 2001-2003, 2005, 2008, 2011, 2013-2015, 2017-2019, 2021-2026 | 1982, 1998, 2000, 2004, 2006-2007, 2009-2010, 2012, 2016, 2020 | - | 1982, 1998, 2000, 2004, 2006-2007, 2009-2010, 2012, 2016, 2020 |
| KBOS | 1973-2026 | 1973-1999, 2001-2019, 2021-2026 | 2000, 2020 | - | 2000, 2020 |
| KBWI | 1973-2026 | 1973-1999, 2001-2019, 2021-2026 | 2000, 2020 | - | 2000, 2020 |
| KCDW | 1981-2026 | 1988-1990, 1993, 1995, 1997-1999, 2003-2011, 2013-2019, 2022-2025 | 1981-1987, 1991-1992, 1994, 1996, 2000-2002, 2012, 2020-2021, 2026 | 1973-1980 | 1973-1987, 1991-1992, 1994, 1996, 2000-2002, 2012, 2020-2021, 2026 |
| KDCA | 1973-2026 | 1973-1999, 2001-2019, 2021-2026 | 2000, 2020 | - | 2000, 2020 |
| KEWR | 1973-2026 | 1973-1999, 2001-2019, 2021-2026 | 2000, 2020 | - | 2000, 2020 |
| KFRG | 1980, 1983-2026 | 1994-1995, 1997-1998, 2001, 2004-2011, 2013-2015, 2018-2019, 2022-2026 | 1980, 1983-1993, 1996, 1999-2000, 2002-2003, 2012, 2016-2017, 2020-2021 | 1973-1979, 1981-1982 | 1973-1993, 1996, 1999-2000, 2002-2003, 2012, 2016-2017, 2020-2021 |
| KHPN | 1973-2026 | 1973-1994, 1996-1999, 2001-2019, 2021-2026 | 1995, 2000, 2020 | - | 1995, 2000, 2020 |
| KISP | 1973-2026 | 1973-1999, 2001-2019, 2021-2026 | 2000, 2020 | - | 2000, 2020 |
| KJFK | 1973-2026 | 1973-1999, 2001-2019, 2021-2026 | 2000, 2020 | - | 2000, 2020 |
| KLGA | 1973-2026 | 1973-1999, 2001-2010, 2012-2019, 2021-2026 | 2000, 2011, 2020 | - | 2000, 2011, 2020 |
| KMMU | 1983-2026 | 1987, 1993, 1995, 1997-1999, 2001-2004, 2008-2011, 2013-2015, 2017-2019, 2021-2026 | 1983-1986, 1988-1992, 1994, 1996, 2000, 2005-2007, 2012, 2016, 2020 | 1973-1982 | 1973-1986, 1988-1992, 1994, 1996, 2000, 2005-2007, 2012, 2016, 2020 |
| KNYC | 1973-2026 | 1973-1999, 2001-2010, 2012-2019, 2021-2026 | 2000, 2011, 2020 | - | 2000, 2011, 2020 |
| KPHL | 1973-2026 | 1973-1999, 2001-2019, 2021-2026 | 2000, 2020 | - | 2000, 2020 |
| KPOU | 1973-2026 | 1973-1975, 1977-1980, 1982-1992, 1994, 1997-1999, 2001-2007, 2010-2015, 2017-2019, 2021-2026 | 1976, 1981, 1993, 1995-1996, 2000, 2008-2009, 2016, 2020 | - | 1976, 1981, 1993, 1995-1996, 2000, 2008-2009, 2016, 2020 |
| KSWF | 1973-2026 | 1975, 1977, 1981-1982, 1984-1996, 1998-1999, 2001-2005, 2007-2018, 2021-2026 | 1973-1974, 1976, 1978-1980, 1983, 1997, 2000, 2006, 2019-2020 | - | 1973-1974, 1976, 1978-1980, 1983, 1997, 2000, 2006, 2019-2020 |
| KTEB | 1973-2026 | 1973, 1975-1979, 1981-1998, 2001-2019, 2021-2026 | 1974, 1980, 1999-2000, 2020 | - | 1974, 1980, 1999-2000, 2020 |

Field completeness by station:

| Station | Daily low null | Daily high dewpoint null | Intraday temp rows | Intraday dewpoint rows | RH rows | Wind rows | Pressure rows | Precip rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| KABE | 2 | 5 | 519,588 | 513,623 | 521,124 | 488,323 | 540,605 | 186,873 |
| KALB | 0 | 0 | 517,658 | 510,473 | 518,967 | 463,169 | 532,103 | 190,432 |
| KBDR | 6 | 8 | 479,242 | 472,078 | 479,107 | 468,244 | 491,427 | 181,767 |
| KBOS | 0 | 0 | 516,057 | 509,536 | 517,467 | 534,491 | 541,823 | 185,232 |
| KBWI | 0 | 0 | 507,483 | 501,595 | 508,303 | 467,605 | 517,290 | 175,224 |
| KCDW | 1,804 | 2,191 | 366,479 | 352,093 | 357,251 | 315,535 | 392,338 | 169,230 |
| KDCA | 0 | 1 | 518,552 | 512,868 | 519,421 | 521,212 | 541,115 | 183,407 |
| KEWR | 0 | 0 | 499,954 | 494,199 | 500,959 | 502,610 | 518,306 | 170,288 |
| KFRG | 1,416 | 1,807 | 367,639 | 355,597 | 360,445 | 374,295 | 396,171 | 166,414 |
| KHPN | 1 | 80 | 470,986 | 460,488 | 468,357 | 435,163 | 483,211 | 135,102 |
| KISP | 0 | 1 | 498,073 | 491,565 | 498,955 | 474,957 | 516,005 | 150,554 |
| KJFK | 0 | 0 | 491,895 | 486,269 | 492,912 | 502,629 | 513,734 | 162,279 |
| KLGA | 0 | 0 | 502,722 | 496,646 | 503,792 | 510,048 | 520,242 | 171,724 |
| KMMU | 971 | 1,357 | 383,188 | 368,723 | 374,215 | 328,768 | 399,310 | 191,295 |
| KNYC | 0 | 0 | 502,722 | 496,646 | 503,792 | 510,048 | 520,242 | 171,724 |
| KPHL | 0 | 0 | 530,091 | 524,719 | 531,378 | 531,688 | 551,148 | 192,871 |
| KPOU | 13 | 270 | 508,237 | 494,304 | 502,471 | 377,511 | 518,730 | 163,994 |
| KSWF | 56 | 164 | 400,996 | 391,927 | 398,800 | 337,438 | 408,127 | 105,377 |
| KTEB | 3 | 31 | 480,717 | 473,328 | 479,549 | 440,895 | 486,521 | 157,196 |

The 2026-06-29 sanity query returned the following zero-count checks after correction:

| Check | Result |
|---|---:|
| Duplicate daily station-date rows | 0 |
| Duplicate intraday station-time rows | 0 |
| Duplicate coverage station-date rows | 0 |
| Usable daily Tmax rows outside accepted range or below daily low | 0 |
| Normalized intraday rows outside accepted temp/dewpoint/humidity/wind/precip bounds | 0 |
| Daily rows missing bronze source request/source record linkage | 0 |
| Usable daily rows missing availability ledger entries | 0 |
| Coverage rows marked `saved` without a daily Tmax label | 0 |
| Usable daily rows available before local day end | 0 |

## Backfill Behavior

Default full range:

```text
1973-01-01 through latest complete America/New_York local date
```

The validation contract start date remains:

```text
2000-01-01
```

Station scope:

- Default `--stations all` selects the 19 non-pseudo task-01 stations that have a `wunderground_station_id`.
- Pseudo-points are rejected for WU station fetches.
- Explicit station selection accepts comma-separated task-01 canonical station IDs.

Windowing:

- Default chunk size is 31 local dates per provider request.
- Each planned date is first marked `not_fetched` in `audit.wu_station_date_coverage`.
- Successful parsed daily rows with `daily_high_f IS NOT NULL` change station-date coverage to `saved`.
- Successful parsed daily rows with `daily_high_f IS NULL` keep their raw and normalized provenance but change station-date coverage to `no_data`.
- Successful requests with no parsed rows change station-date coverage to `no_data`.
- Failed requests change station-date coverage to `failed`.

Resume:

- With `--resume`, a window is skipped only when every date in the window already has status `saved` or `no_data`.
- Partial windows are fetched again, with normalized daily and intraday rows upserted idempotently.

Concurrency:

- `ThreadPoolExecutor` is used for provider fetches.
- Default worker count is 4.
- A shared rate limiter throttles all workers.
- Every completed or failed worker result is persisted in its own transaction.

## Design Decisions

### Weather.com endpoint as Wunderground implementation path

The existing codebase audit showed that the working local historical observations path is Weather.com branded but serves Wunderground-style station IDs. This implementation follows that path and preserves the user-facing task terminology as Wunderground settlement actuals.

### `WUNDERGROUND_API_KEY` with `WEATHERCOM_API_KEY` fallback

The new canonical env var is `WUNDERGROUND_API_KEY`. The fallback exists because the existing local scraper setup already used `WEATHERCOM_API_KEY`, and live smoke verified that this key path is available locally.

### Conservative label availability

The provider response does not expose an exact settlement publication timestamp. The implementation therefore uses local day end plus 24 hours for daily label availability. This is deliberately conservative and safer than assuming labels are available at midnight or at the final intraday observation time.

### Station-time identity for intraday facts

Bronze keeps every raw source response and revision. Silver intraday facts represent the current normalized observation at a station/time. Therefore the final identity is `(station_id, observation_time_utc)`, not `(station_id, observation_time_utc, source_request_id)`.

### Full historical all-station backfill run explicitly

The implementation does not launch long-running acquisitions implicitly. After the user explicitly requested the live run, the all-station historical backfill was executed with one process, four workers, 31-day chunks, and the configured 60 requests/minute limiter. The resulting DB state is complete for the requested date range and contains no failed or not-fetched station-days.

## Known Limitations and Follow-Up Work

- The current daily label path is computed from intraday rows, not a separate provider daily summary, because the live endpoint response used in this task exposed observations.
- There is no IEM or ASOS cross-source reconciliation in task 02.
- There is no official publication-time capture from Wunderground/Weather.com; daily labels use the conservative availability rule.
- Intraday revision history is not separately versioned in silver; bronze preserves raw response revisions and silver stores the latest normalized station-time row.
- The provider client uses process-local rate limiting. Multiple independent processes would each need their own external coordination if run concurrently.

Follow-up work:

- Add task-03 IEM MOS ingestion so forecast guidance can be joined against WU labels.
- Add task-04 ASOS/METAR ingestion for independent observed-weather features.
- Decide whether an external provider publication-time signal is available; if it is, replace the conservative daily-label lag for future rows.
- Add optional cross-source daily-high reconciliation after IEM/ASOS sources exist.

## Reviewer Checklist

- Confirm migration head includes `0003_wu_actuals`, `0004_wu_intraday_identity`, `0005_gribstream_single_cutoff`, and `0006_grib_job_chunk_identity`.
- Confirm `python -m pytest -q` reports `55 passed` or more.
- Confirm `db inspect-contract` reports 28 tables and 30 indexes checked.
- Confirm `validate foundation` reports no failures after the `T_MINUS_1_2045UTC` cutoff addition.
- Confirm `validate wunderground` reports no failures and `wu_coverage_rows=371184`.
- Confirm `wunderground inspect-config` redacts provider credentials.
- Confirm no new backfill is launched accidentally; the completed full historical state is already in the local DB.
- Confirm downstream feature builders use availability ledger timestamps, not raw observation dates, for leakage decisions.

## Next Handoff

Task 03 can now rely on:

- `silver.wu_daily_actuals` for canonical daily actual labels.
- `silver.wu_intraday_observations` for WU intraday observations.
- `silver.availability_ledger` WU rows for leakage-safe source availability.
- `audit.wu_station_date_coverage` for saved/failed/no-data/not-fetched coverage.
- `klga-tmax validate wunderground` as the task-02 health gate.

Recommended next implementation task:

```text
03_iem_mos_station_guidance
```

The full task-02 historical backfill has already been run. To check that state before downstream feature work, use:

```powershell
$env:KLGA_DB_URL = "postgresql+psycopg://postgres:root@127.0.0.1:5432/klga_tmax_research"
python -m klga_tmax.cli wunderground coverage --start-date 1973-01-01 --end-date 2026-06-27
python -m klga_tmax.cli validate wunderground
```
