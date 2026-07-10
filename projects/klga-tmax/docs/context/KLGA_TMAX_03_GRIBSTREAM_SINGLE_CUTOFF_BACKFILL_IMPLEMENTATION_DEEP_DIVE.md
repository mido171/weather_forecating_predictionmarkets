# KLGA Tmax Task 03 GribStream Single-Cutoff Backfill Implementation Deep Dive

Last updated: 2026-06-28

## Executive Summary

Task 03 is implemented as a GribStream `/timeseries` single-cutoff backfill planner, runner, persistence layer, validator, and CLI workflow for KLGA Tmax forecast features.

The implemented plan uses the canonical UTC cutoff:

```text
cutoff_id: T_MINUS_1_2045UTC
cutoff_utc_time: 20:45:00 UTC on target_date_ny - 1 calendar day
display_alias: 22:45 Europe/Stockholm / 16:45 America/New_York in summer
```

The full 15-model action plan is persisted in PostgreSQL under job:

```text
klga_single_cutoff_2045utc_full
```

Current persisted full-plan status:

```text
models_planned: 15
chunks_planned: 20,792
estimated_credits: 1,150,158
completed_chunks: 0
rows_upserted: 0
status: planned
selector_gaps: 0
```

Actual data fetching is blocked by GribStream authentication. A fixed one-chunk GFS smoke run reached GribStream and stopped correctly on:

```text
stopped_reason: auth_failed_http_401
http_status: 401
error_message: Unauthorized
attempts: 1
rows_upserted: 0
```

Because the smoke request returned `401`, the full 20,792-call backfill was not launched. The runner is ready and resumable, but it needs a valid GribStream token before it can fetch rows.

## Reader Orientation

Implementation root:

```text
C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\klga_tmax\implementation
```

Target database:

```text
postgresql+psycopg://<user>:<password>@127.0.0.1:5432/klga_tmax_research
```

Provider endpoint shape:

```text
POST https://gribstream.com/api/v2/{model_id}/timeseries
```

The code uses `GRIBSTREAM_API_TOKEN` first, then `GRIBSTREAM_API_KEY`. If the environment value starts with `Bearer `, the loader strips that prefix before sending the standard:

```text
Authorization: Bearer <token>
```

## Scope Boundaries

In scope:

- Canonical `T_MINUS_1_2045UTC` cutoff registry row.
- Tier B 10-point KLGA coordinate requests.
- All 15 requested GribStream model families.
- Live GribStream catalog/shared-parameter selector resolution.
- `/timeseries` request planning with `timesList`, `coordinates`, model-specific `asOf`, variables, expressions, and members.
- One-worker rate-limited runner with 12 seconds between authenticated calls.
- Stop-on-auth and stop-on-rate-limit behavior.
- Raw success-response artifact pathing.
- Bronze source request/source record persistence for successful responses.
- Silver forecast value persistence.
- Availability ledger persistence.
- Audit job, chunk, catalog, and source-gap tracking.
- Resume by completed `request_sha256`.
- CLI commands for inspect, plan, run, smoke, status, and validation.
- Tests, schema contract checks, live catalog planning, and one live auth smoke.

Out of scope or currently blocked:

- The full historical data pull is not complete because GribStream returned `401 Unauthorized`.
- No successful GribStream forecast rows are present yet.
- No GribStream bronze source requests are present yet.
- No GribStream availability ledger rows are present yet.
- No pressure-level archive, full-horizon `/runs` archive, or Tier 3 model pull is implemented.
- URMA is retained as retrospective target-day support only, never as pre-target live evidence.

## Source-of-Truth Inputs

The implementation follows these local source inputs:

- The user-approved single-cutoff action plan with `T_MINUS_1_2045UTC`.
- `strategy_spec/data_aquisition/08_gribstream_nwp_forecast_runs/03_gribstream_nwp_forecast_runs.md`
- `strategy_spec/context/KLGA_TMAX_POSTGRES_PERSISTENCE_CONTEXT.md`
- `strategy_spec/context/KLGA_TMAX_00_FOUNDATION_IMPLEMENTATION_DEEP_DIVE.md`
- `strategy_spec/context/KLGA_TMAX_01_STATION_UNIVERSE_IMPLEMENTATION_DEEP_DIVE.md`
- `strategy_spec/context/KLGA_TMAX_02_WUNDERGROUND_SETTLEMENT_ACTUALS_IMPLEMENTATION_DEEP_DIVE.md`
- Local GribStream API skill rules for `/timeseries`, `timesList`, shared parameters, selectors, `asOf`, auth, quotas, retries, and stop conditions.

## Requirements-to-Implementation Traceability

| Requirement | Implementation evidence | Current status |
|---|---|---|
| One snapshot per target date | `providers/gribstream/plan.py` forces one-day chunks for all `asOf`-backed models. | Implemented |
| Canonical cutoff `T_MINUS_1_2045UTC` | `registry/cutoffs.py`, seeded by `db migrate`, validated by `validate gribstream`. | Implemented |
| Use `/timeseries` | Client posts to `/{model_id}/timeseries`; planner builds `timesList` payloads. | Implemented |
| Tier B 10 KLGA pseudo-points | Planner reads `coordinate_tier("B")` and serializes 10 named coordinates. | Implemented |
| 15 requested models | `MODEL_SPECS` contains all requested T1/T2 models. | Implemented |
| Respect model-specific `asOf` buffers | `as_of_utc()` subtracts the per-model buffer from T-1 20:45 UTC. | Implemented |
| Use live selector resolution, no guessed selectors | `catalog.py` resolves shared parameters/native selectors from live GribStream catalog endpoints. | Implemented; 72 snapshots, 0 selector gaps |
| Persist plan and chunk tracking | `audit.gribstream_backfill_jobs`, `audit.gribstream_backfill_chunks`. | Implemented |
| Persist raw/bronze/silver/availability on success | `persistence.py` inserts bronze requests/records, silver values, and availability ledger rows after successful responses. | Implemented; no successful rows yet |
| Stop on 401/403 | `client.py` treats 401/403 as stop statuses; runner marks `auth_failed`. | Verified by live 401 smoke |
| Stop on 429 and honor `Retry-After` | `client.py` treats 429 as a stop status and sleeps for `Retry-After` if present. | Implemented |
| One worker, 12s between authenticated calls | `OneThreadRateLimiter`, default `spacing_seconds=12.0`. | Implemented |
| Resume by completed `request_sha256` | Runner checks completed/completed-empty request hashes before refetching. | Implemented |
| Full status by model | `model_status()` reports chunks, completed, remaining, blocked, credits, rows. | Implemented |
| Validation of lineage fields | `validation/gribstream.py` checks required identity/lineage columns for persisted values. | Implemented |

## Model Plan Persisted In Postgres

The full persisted job is:

```text
job_id: klga_single_cutoff_2045utc_full
status: planned
cutoff_id: T_MINUS_1_2045UTC
start_date: 2014-07-31
end_date: 2026-06-28
coordinate_tier: B
planned_chunks: 20,792
estimated_credits: 1,150,158
```

Per-model status from `gribstream status --job-id klga_single_cutoff_2045utc_full`:

| Tier | Model | Effective from | Through | Chunks | Completed | Remaining | Blocked | Est. credits |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| T2 | `aifsenfo` | 2025-07-03 | 2026-06-28 | 361 | 0 | 361 | 0 | 36,822 |
| T2 | `aifsoper` | 2025-02-26 | 2026-06-28 | 488 | 0 | 488 | 0 | 6,832 |
| T2 | `aigefssfc` | 2025-06-02 | 2026-06-28 | 392 | 0 | 392 | 0 | 24,304 |
| T2 | `aigfssfc` | 2026-04-17 | 2026-06-28 | 73 | 0 | 73 | 0 | 146 |
| T1 | `gefsatmos` | 2020-10-02 | 2026-06-28 | 2,096 | 0 | 2,096 | 0 | 129,952 |
| T1 | `gefsatmosmean` | 2020-10-02 | 2026-06-28 | 2,096 | 0 | 2,096 | 0 | 4,192 |
| T1 | `gfs` | 2021-03-23 | 2026-06-28 | 1,924 | 0 | 1,924 | 0 | 153,920 |
| T1 | `hrrr` | 2014-07-31 | 2026-06-28 | 4,351 | 0 | 4,351 | 0 | 348,080 |
| T1 | `ifsenfo` | 2024-03-02 | 2026-06-28 | 849 | 0 | 849 | 0 | 86,598 |
| T1 | `ifsoper` | 2024-02-29 | 2026-06-28 | 851 | 0 | 851 | 0 | 11,914 |
| T1 | `nbm` | 2020-09-30 | 2026-06-28 | 2,098 | 0 | 2,098 | 0 | 167,840 |
| T2 | `nbmqmd` | 2026-02-01 | 2026-06-28 | 148 | 0 | 148 | 0 | 3,108 |
| T1 | `rap` | 2021-02-23 | 2026-06-28 | 1,952 | 0 | 1,952 | 0 | 156,160 |
| T1 | `rtma` | 2018-01-02 | 2026-06-28 | 3,100 | 0 | 3,100 | 0 | 12,400 |
| T1 | `urma` | 2024-05-01 | 2026-06-28 | 13 | 0 | 13 | 0 | 7,890 |

## Time And Availability Semantics

The canonical cutoff is computed as:

```text
cutoff_utc = target_date_ny - 1 calendar day at 20:45:00 UTC
```

Model-specific `asOf` is then:

```text
asOf = cutoff_utc - model_buffer
```

Implemented buffers:

| Model family | Buffer | Effective asOf at cutoff | Intended latest safe cycle |
|---|---:|---:|---|
| `gfs` | 4h00m | T-1 16:45 UTC | 12Z T-1 |
| `gefsatmos`, `gefsatmosmean` | 4h00m | T-1 16:45 UTC | 12Z T-1 |
| `ifsoper`, `ifsenfo` | 3h00m | T-1 17:45 UTC | 12Z T-1 |
| `aifsoper`, `aifsenfo` | 3h30m | T-1 17:15 UTC | 12Z T-1 |
| `hrrr` | 2h15m | T-1 18:30 UTC | 18Z T-1 extended |
| `rap` | 1h45m | T-1 19:00 UTC | 19Z if available, else 18Z |
| `nbm`, `nbmqmd` | 1h45m | T-1 19:00 UTC | latest safe hourly |
| `rtma` | 1h00m | T-1 19:45 UTC | latest safe analysis |
| `urma` | n/a | n/a | retrospective support only |

Valid-time selection:

| Fetch shape | Implemented valid times |
|---|---|
| `hourly_peak` | 12:00 through 21:00 America/New_York on target date, converted to UTC. |
| `synoptic` | 18:00 UTC on target date and 00:00 UTC on target date + 1. |
| `nbmqmd_max18` | 00:00 UTC on target date + 1. |
| `rtma_latest` | Latest whole-hour analysis at or before T-1 19:45 UTC, implemented as T-1 19:00 UTC. |
| `urma_peak_temp` | Target-day 12:00 through 21:00 America/New_York, marked retrospective. |

Important leakage note: GribStream `asOf` is treated as a model-run-time cutoff, not a direct proof of public first availability. The buffers are conservative production-schedule guards, and persisted rows carry availability metadata so downstream features can audit or tighten this rule later.

## Selector And Variable Resolution

Selectors are not hard-coded as guessed request fields. The resolver reads GribStream catalog endpoints and then copies the provider-resolved selector objects into the `/timeseries` payload.

Catalog endpoints used:

```text
GET /api/v2/catalog/datasets/{model_id}
GET /api/v2/catalog/datasets/{model_id}/parameters
GET /api/v2/catalog/datasets/{model_id}/parameters/{parameter}
GET /api/v2/catalog/shared-parameters/{code}?dataset={model_id}&alias={alias}
```

Resolved groups:

| Group | Models | Selector intent |
|---|---|---|
| `hourly_8` | `hrrr`, `gfs`, `rap` | temperature, dew point, relative humidity, 10m U/V wind, wind speed, precipitation, cloud cover or sea-level pressure fallback. |
| `rtma_4` | `rtma` | temperature, dew point, relative humidity, wind gust or wind speed fallback. |
| `nbm_8` | `nbm` | 2m temp, 2m dew point, wind speed, wind gust, precipitation, Tmax, temp ensemble stddev, Tmax ensemble stddev. |
| `ecmwf_7` | `ifsoper`, `aifsoper` | `2t`, `2d`, `10u`, `10v`, `msl`, `tp`, `tcc`. |
| `temp_only` | ensemble/mean temperature models plus `urma` | shared `temperature_2m`. |
| `nbmqmd_percentiles` | `nbmqmd` | 21 native TMP max-18h percentile selectors from p01 through p99. |

Live catalog planning evidence:

```text
catalog_snapshots: 72
selector_gaps: 0
models_planned: 15
```

## Persistence Design

Tables added:

```text
audit.gribstream_catalog_snapshots
audit.gribstream_backfill_jobs
audit.gribstream_backfill_chunks
audit.gribstream_source_gaps
silver.grib_forecast_values
```

Existing shared tables used on successful responses:

```text
bronze.source_requests
bronze.source_records
silver.availability_ledger
```

Chunk identity is job-scoped:

```text
unique: audit.gribstream_backfill_chunks(job_id, request_sha256)
index:  audit.gribstream_backfill_chunks(request_sha256)
```

This matters because a smoke job and the full job can legitimately have the same request hash. The runner still uses completed `request_sha256` values globally to avoid refetching already-completed payloads, but the tracking rows themselves remain separate per job.

For every successful row in `silver.grib_forecast_values`, validation requires:

```text
model_id
member
grid_point_id
forecasted_at_utc
forecasted_time_utc
variable_alias
variable_name
source_request_id
source_record_id
request_sha256
availability_method
```

The current DB has no GribStream forecast rows yet, so this lineage check passes vacuously and will become active once rows are fetched.

## Rate Limits, Retries, And Stop Rules

The authenticated request runner uses:

```text
one worker
12.0 seconds between authenticated calls
max_retries: 3
timeout_seconds: 90
retryable HTTP statuses: 500, 502, 503, 504
stop HTTP statuses: 401, 403, 429
```

`Retry-After` is parsed as either seconds or an HTTP date. On 429, the client sleeps for `Retry-After` if present, marks the chunk `rate_limited`, refreshes job status, and stops so state is preserved.

On 401/403, the runner marks the chunk `auth_failed`, refreshes job status, and stops. This was verified by live smoke.

## Change Inventory

Code and migrations:

- `implementation/alembic/versions/0005_gribstream_single_cutoff.py`
- `implementation/alembic/versions/0006_grib_job_chunk_identity.py`
- `implementation/src/klga_tmax/cli.py`
- `implementation/src/klga_tmax/db/migrations_check.py`
- `implementation/src/klga_tmax/providers/gribstream/__init__.py`
- `implementation/src/klga_tmax/providers/gribstream/backfill.py`
- `implementation/src/klga_tmax/providers/gribstream/catalog.py`
- `implementation/src/klga_tmax/providers/gribstream/client.py`
- `implementation/src/klga_tmax/providers/gribstream/config.py`
- `implementation/src/klga_tmax/providers/gribstream/models.py`
- `implementation/src/klga_tmax/providers/gribstream/parser.py`
- `implementation/src/klga_tmax/providers/gribstream/persistence.py`
- `implementation/src/klga_tmax/providers/gribstream/plan.py`
- `implementation/src/klga_tmax/registry/cutoffs.py`
- `implementation/src/klga_tmax/validation/gribstream.py`

Tests:

- `implementation/tests/test_gribstream_plan.py`
- `implementation/tests/test_gribstream_schema_contract.py`
- `implementation/tests/test_timezones_cutoffs.py`

Documentation:

- `strategy_spec/context/KLGA_TMAX_03_GRIBSTREAM_SINGLE_CUTOFF_BACKFILL_IMPLEMENTATION_DEEP_DIVE.md`

## File-by-File Deep Dive

### `implementation/src/klga_tmax/providers/gribstream/plan.py`

Owns the requested action plan. It defines `MODEL_SPECS`, the default end date, cutoff constants, member expansion, valid-time selection, per-model `asOf` buffers, request payload construction, credit estimates, chunk hashing, and job-plan construction.

Important implementation details:

- `CUTOFF_ID = "T_MINUS_1_2045UTC"`.
- `DEFAULT_END_DATE = 2026-06-28`.
- All `asOf`-backed models are forced to one target date per chunk because `/timeseries` accepts one `asOf` timestamp.
- URMA can use wider chunks because it has no live `asOf` and is marked retrospective.
- `request_sha256` includes `model_id`, endpoint, and payload. This prevents collisions between models with identical bodies.
- The planner emits exactly the user-approved model dates and estimated credits.

### `implementation/src/klga_tmax/providers/gribstream/catalog.py`

Owns live selector resolution. It fetches dataset, parameter, parameter-detail, and shared-parameter catalog JSON, stores catalog snapshots, and creates `ResolvedSelector` objects for request payload generation.

It deliberately copies provider-resolved request variables and expressions instead of inventing selector dictionaries. If a model selector cannot be resolved, the model produces a source gap and no chunks are built for unresolved selectors.

### `implementation/src/klga_tmax/providers/gribstream/client.py`

Owns the authenticated HTTP boundary. It posts NDJSON `/timeseries` requests, rate-limits calls, writes successful raw payloads to gzipped artifacts, parses `Retry-After`, retries transient provider errors, and raises structured errors for auth/rate-limit/terminal failures.

No successful GribStream response has been received yet. The 401 smoke failure did not create a raw file or bronze source request; it only updated audit chunk state.

### `implementation/src/klga_tmax/providers/gribstream/parser.py`

Owns NDJSON interpretation for successful responses. It maps provider locations back to Tier B grid points, normalizes forecast run time and valid time, extracts members, variable aliases, raw-row hashes, units, canonical values, provider index update metadata, target dates, and availability flags.

URMA rows are marked:

```text
availability_method: manual_override
quality_note: retrospective_only_not_pre_target_live_evidence
```

### `implementation/src/klga_tmax/providers/gribstream/persistence.py`

Owns database writes for catalog snapshots, plans, chunks, successful source requests, source records, silver values, availability rows, and source gaps.

It also owns job status calculation and per-model reporting. Chunk state is now job-scoped so smoke jobs cannot corrupt full-job chunk ownership.

### `implementation/src/klga_tmax/providers/gribstream/backfill.py`

Owns orchestration. It resolves selectors, builds/persists the plan, runs chunks sequentially, marks running attempts, skips already-completed request hashes, stops on auth/rate-limit, refreshes job status, and returns per-model status.

### `implementation/src/klga_tmax/providers/gribstream/config.py`

Owns provider settings. It reads token, base URL, artifact root, rate spacing, timeout, retry count, and user agent. It redacts token presence in CLI output.

### `implementation/src/klga_tmax/cli.py`

Adds:

```text
python -m klga_tmax.cli gribstream inspect-config
python -m klga_tmax.cli gribstream plan
python -m klga_tmax.cli gribstream run
python -m klga_tmax.cli gribstream smoke
python -m klga_tmax.cli gribstream status
python -m klga_tmax.cli validate gribstream
```

### `implementation/alembic/versions/0005_gribstream_single_cutoff.py`

Creates the GribStream audit and silver schema. It also adds indexes for catalog snapshots, job status, chunk status, source gaps, forecast value identity, request lookups, and coordinate-variable lookups.

### `implementation/alembic/versions/0006_grib_job_chunk_identity.py`

Fixes request identity for multi-job safety. It replaces the global unique `request_sha256` chunk index with:

```text
ix_gribstream_chunks_request_sha
ux_gribstream_chunks_job_request
```

This keeps request hashes reusable across smoke/full jobs while preserving per-job chunk uniqueness.

### `implementation/src/klga_tmax/validation/gribstream.py`

Validates cutoff presence, required tables, lineage fields, availability rows when values exist, completed chunk count, and model coverage.

## CLI Runbook

Set the database URL:

```powershell
$env:KLGA_DB_URL = "postgresql+psycopg://<user>:<password>@127.0.0.1:5432/klga_tmax_research"
```

Inspect redacted configuration and action-plan model specs:

```powershell
python -m klga_tmax.cli gribstream inspect-config
```

Refresh the full persisted plan:

```powershell
python -m klga_tmax.cli gribstream plan --job-id klga_single_cutoff_2045utc_full
```

Run a one-chunk smoke after replacing/fixing the token:

```powershell
python -m klga_tmax.cli gribstream run --job-id klga_single_cutoff_2045utc_smoke_gfs --models gfs --start-date 2026-06-28 --end-date 2026-06-28 --max-chunks 1 --no-resume
```

Run the full backfill after auth is fixed:

```powershell
python -m klga_tmax.cli gribstream run --job-id klga_single_cutoff_2045utc_full
```

Run a progressive capped backfill:

```powershell
python -m klga_tmax.cli gribstream run --job-id klga_single_cutoff_2045utc_full --max-chunks 25
```

Inspect status:

```powershell
python -m klga_tmax.cli gribstream status --job-id klga_single_cutoff_2045utc_full
```

Validate GribStream persistence:

```powershell
python -m klga_tmax.cli validate gribstream
```

Validate the full DB contract:

```powershell
python -m klga_tmax.cli db inspect-contract
```

## Verification Evidence

Commands run successfully:

```text
python -m compileall src tests
python -m pytest
python -m klga_tmax.cli db migrate
python -m klga_tmax.cli db inspect-contract
python -m klga_tmax.cli gribstream plan --job-id klga_single_cutoff_2045utc_full
python -m klga_tmax.cli validate gribstream
python -m klga_tmax.cli gribstream status --job-id klga_single_cutoff_2045utc_full
```

Test result:

```text
53 passed
```

DB contract result:

```text
ok: true
schemas_checked: 8
tables_checked: 28
indexes_checked: 30
cutoff_rows: 5
station_rows: 29
```

GribStream validation result after final checks:

```text
ok: true
cutoff_rows: 1
grib_chunks: 20,794
grib_models_planned: 15
grib_completed_chunks: 0
grib_forecast_values: 0
grib_availability_rows: 0
```

The `20,794` chunk count includes:

```text
20,792 full-job planned chunks
2 one-chunk GFS smoke jobs
```

Final live lineages:

```text
silver.grib_forecast_values rows: 0
silver.availability_ledger rows where source_name='gribstream': 0
bronze.source_requests rows where source_name='gribstream': 0
audit.gribstream_catalog_snapshots rows: 72
```

Live smoke result after attempt tracking patch:

```text
job_id: klga_single_cutoff_2045utc_smoke_gfs_attempt_tracking
model_id: gfs
target_start_date: 2026-06-28
target_end_date: 2026-06-28
status: auth_failed
attempts: 1
http_status: 401
error_type: HTTP_401
error_message: Unauthorized
rows_upserted: 0
stopped_reason: auth_failed_http_401
```

## Current Blocker And Remaining Work

Blocker:

```text
GribStream authenticated /timeseries smoke returns HTTP 401 Unauthorized.
```

Immediate remaining work:

1. Replace or authorize the GribStream token in `GRIBSTREAM_API_TOKEN` or `GRIBSTREAM_API_KEY`.
2. Rerun the one-chunk GFS smoke.
3. Confirm `bronze.source_requests`, raw artifacts, `silver.grib_forecast_values`, and `silver.availability_ledger` receive rows.
4. Run the full job progressively, for example `--max-chunks 25`, before committing to the full 20,792-call pull.
5. Monitor status by model after each batch.
6. Re-run `validate gribstream` and `db inspect-contract` after meaningful batches.

The full job is ready to resume from:

```text
klga_single_cutoff_2045utc_full
```

No GribStream forecast rows have been fetched yet, so all 20,792 full-job chunks remain.

## Risk Register

| Risk | Mitigation |
|---|---|
| Invalid or unauthorized token blocks all fetches. | Smoke job stops on 401 before full backfill; no full run launched while token is bad. |
| `/timeseries asOf` is not first public availability proof. | Conservative model-specific buffers plus availability metadata are persisted for downstream audit. |
| Large credit spend. | Full plan is visible before execution; runner supports `--max-chunks` progressive batches. |
| Smoke and full jobs sharing request hashes. | Migration `0006` makes chunk identity job-scoped and keeps request hash indexed for resume. |
| Selector drift in GribStream catalog. | Selectors are resolved live and catalog snapshots are persisted; unresolved selectors produce gaps instead of guessed requests. |
| 429 rate-limit event mid-run. | Runner honors `Retry-After`, marks the chunk `rate_limited`, stops, and preserves state. |
| URMA leakage misuse. | URMA is explicitly retrospective-only and marked with manual override metadata. |

## Final State

The KLGA GribStream single-cutoff backfill system is implemented and validated. The database contains the complete 15-model full action plan with the requested dates, chunk counts, and credit estimates. The fetch runner is rate-limited, resumable, selector-safe, and capable of persisting successful responses through bronze, silver, and availability layers.

The only active blocker is provider authentication: live GribStream `/timeseries` returned `401 Unauthorized`, so no forecast data was fetched.
