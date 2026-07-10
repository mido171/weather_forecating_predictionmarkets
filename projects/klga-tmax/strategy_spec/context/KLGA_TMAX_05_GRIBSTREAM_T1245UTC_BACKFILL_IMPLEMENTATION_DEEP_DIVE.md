# KLGA Tmax Task 05 GribStream T1245UTC Backfill Implementation Deep Dive

Last updated: 2026-06-29

## Executive Summary

This implementation adds `T_1245UTC` as a first-class GribStream cutoff profile for the KLGA Tmax project while preserving the earlier `T_MINUS_1_2045UTC` profile. The new profile represents one target-day snapshot per New York target date at `12:45:00 UTC`, displayed in summer as `08:45 America/New_York / 14:45 Europe/Stockholm`.

The GribStream planner now computes profile-specific cutoff timestamps, per-model `asOf` values, effective target starts, request hashes, and NBMQMD max-18h valid times. The new target-day profile uses each model catalog archive start as the first target date, not `archive_start + 1 day`, and uses NBMQMD `target_date + 1 at 06:00 UTC` for the max-18h percentile selector set.

The resumable full plan was persisted in PostgreSQL under job `klga_t1245utc_full_backfill_v1`. It contains 15 models, 20,806 planned chunks, zero selector gaps, and 1,150,873 estimated credits. The full pull was not executed in this pass because 20,806 authenticated calls with 12-second spacing is a multi-day run. Two live pilots were executed instead: one day for all 15 models and seven days for `gfs,hrrr,rap,nbm`. Both pilots completed with HTTP 200 for every chunk and no auth or rate-limit stop.

## Reader Orientation and Document Map

Read this document when continuing the KLGA GribStream backfill from `T_1245UTC`, reviewing what changed after the prior-day cutoff implementation, or deciding whether the full 1.15M-credit backfill is ready to launch.

Major sections:

- Scope Boundaries: what was implemented and what was deliberately left unexecuted.
- Requirements-to-Implementation Traceability: each requested behavior mapped to code and verification.
- Change Inventory and File-by-File Deep Dive: every edited source, test, config, and documentation path.
- Architecture and Control Flow: how CLI input becomes catalog selectors, chunks, requests, rows, and gaps.
- Testing and Verification Evidence: exact unit, DB, dry-run, and live pilot commands.
- Operational Runbook: exact commands for status checks and the full backfill launch.

## Scope Boundaries

Included:

- `T_1245UTC` registry seed support.
- `T_1245UTC` GribStream planner profile support.
- CLI `--cutoff-id` support for `gribstream plan`, `gribstream run`, and `gribstream smoke`.
- Profile-specific `asOf`, target-start, and NBMQMD valid-time logic.
- Parser fixes for ensemble member `0`, NBMQMD target-day mapping, and response gap generation.
- Full persisted action plan for `klga_t1245utc_full_backfill_v1`.
- Live one-day all-model pilot and seven-day high-row pilot.

Excluded:

- The full 20,806-call backfill was not launched to completion in this interactive pass. At the enforced 12-second minimum spacing, the authenticated-call portion alone is roughly 69.4 hours before catalog lookups, response parsing, DB writes, retries, and provider latency.
- No schema migration was added. Existing audit, bronze, silver, and registry tables already support the new cutoff through seed data and runtime rows.
- No selector was invented. Selector resolution remains live catalog/shared-parameter based.

## Source-of-Truth Inputs

Inputs used for this implementation:

- User action plan requesting `T_1245UTC`, all 15 model ranges, model-specific buffers, NBMQMD `06:00 UTC`, resumable tracking, pilots, and final documentation.
- `C:\Users\ahmad\.codex\skills\gribstream-api\SKILL.md`.
- `C:\Users\ahmad\.codex\skills\gribstream-api\references\API_REFERENCE.md`.
- `C:\Users\ahmad\.codex\skills\gribstream-api\references\BACKTESTING_TIME_SEMANTICS.md`.
- `C:\Users\ahmad\.codex\skills\gribstream-api\references\QUOTA_PERFORMANCE_ERRORS.md`.
- `C:\Users\ahmad\.codex\skills\gribstream-api\references\CATALOG_AND_SELECTORS.md`.
- Existing Task 03 GribStream implementation under `bootstrap/klga_tmax/implementation/src/klga_tmax/providers/gribstream`.
- Existing context document `bootstrap/klga_tmax/strategy_spec/context/KLGA_TMAX_03_GRIBSTREAM_SINGLE_CUTOFF_BACKFILL_IMPLEMENTATION_DEEP_DIVE.md`.
- Live PostgreSQL database `klga_tmax_research`.
- Live GribStream API pilots using the token stored outside the KLGA tree.

## Requirements-to-Implementation Traceability

| Requirement | Implementation location | Delivered behavior | Verification |
|---|---|---|---|
| Add `T_1245UTC` while preserving `T_MINUS_1_2045UTC`. | `bootstrap/klga_tmax/implementation/src/klga_tmax/providers/gribstream/plan.py`; `bootstrap/klga_tmax/implementation/src/klga_tmax/registry/cutoffs.py` | Two supported GribStream profiles exist. Existing default remains prior-day unless `--cutoff-id T_1245UTC` is passed. | `python -m pytest -q`; `python -m klga_tmax.cli validate gribstream`. |
| Use target-day archive start for `T_1245UTC`. | `effective_target_start` in `bootstrap/klga_tmax/implementation/src/klga_tmax/providers/gribstream/plan.py`; summary rows in `bootstrap/klga_tmax/implementation/src/klga_tmax/providers/gribstream/catalog.py` | For `T_1245UTC`, target start equals catalog archive start; for prior-day cutoff, it remains archive start plus one day. | `tests/test_gribstream_plan.py` asserts `gfs` starts `2021-03-22` and total credits equal `1,150,873`. |
| Use profile-specific `asOf` values. | `cutoff_utc`, `as_of_utc`, and `build_chunk` in `bootstrap/klga_tmax/implementation/src/klga_tmax/providers/gribstream/plan.py` | `gfs` is `T 08:45Z`, `hrrr` is `T 10:30Z`, `rtma` is `T 11:45Z`, and related models follow their buffer. | Unit tests and live DB query of pilot rows. |
| Use NBMQMD `target_date + 1 at 06:00 UTC`. | `nbmqmd_max18_time_utc` in `bootstrap/klga_tmax/implementation/src/klga_tmax/providers/gribstream/plan.py` | NBMQMD payload `timesList` becomes `2026-06-29T06:00:00Z` for target date `2026-06-28`. | Unit test and live pilot: NBMQMD returned 210 normalized rows for 21 percentiles x 10 points. |
| Track missing members/selectors/times. | `bootstrap/klga_tmax/implementation/src/klga_tmax/providers/gribstream/parser.py` | Parser emits `missing_timeseries_value` or `empty_response` gaps into `audit.gribstream_source_gaps`. | One-day pilot inserted 340 gaps; seven-day high-row pilot inserted 2,240 gaps. |
| Preserve ensemble member `0`. | `_member_for_row` in `bootstrap/klga_tmax/implementation/src/klga_tmax/providers/gribstream/parser.py` | Numeric member `0` no longer becomes `"deterministic"`. | `tests/test_gribstream_parser.py` asserts member `"0"` is preserved. |
| Persist a full resumable action plan. | `bootstrap/klga_tmax/implementation/src/klga_tmax/providers/gribstream/backfill.py`; existing persistence layer | Job `klga_t1245utc_full_backfill_v1` is stored with 20,806 chunks and per-model status. | `python -m klga_tmax.cli gribstream status --job-id klga_t1245utc_full_backfill_v1`. |
| Keep token out of source. | `.gitignore`; `bootstrap/klga_tmax/AGENTS.md` | `.secrets` is ignored and the local token file is documented as outside the KLGA tree. | `git status` shows `.secrets` omitted; `AGENTS.md` contains the load command without token content. |

## Change Inventory

| File path | Type | Why it changed | Main symbols changed | Verification |
|---|---|---|---|---|
| `.gitignore` | Config | Ignore the local secrets directory that holds the GribStream token. | `/.secrets/` | `git status --short -- .secrets` did not expose token file. |
| `bootstrap/klga_tmax/AGENTS.md` | Docs/config | Document the local credential file and loading rule. | Provider fetching rules section | Read before commands; token was loaded from documented path. |
| `bootstrap/klga_tmax/implementation/src/klga_tmax/cli.py` | CLI | Add `--cutoff-id` and expose both profiles in inspect output. | `_parse_gribstream_cutoff_id`, `gribstream_plan`, `gribstream_run`, `gribstream_smoke` | CLI help and live plan/run commands. |
| `bootstrap/klga_tmax/implementation/src/klga_tmax/providers/gribstream/plan.py` | Planner | Add profile-aware cutoff math, `asOf`, target starts, NBMQMD `06Z`, and request hashing. | `GribStreamCutoffProfile`, `CUTOFF_PROFILES`, `cutoff_utc`, `as_of_utc`, `effective_target_start`, `valid_times_for_target`, `build_chunk`, `build_job_plan` | Unit tests and dry-run plan. |
| `bootstrap/klga_tmax/implementation/src/klga_tmax/providers/gribstream/catalog.py` | Catalog summary | Make model summary rows profile-aware and recompute estimated totals from profile-specific days. | `spec_summary_rows` | Unit test totals for both profiles. |
| `bootstrap/klga_tmax/implementation/src/klga_tmax/providers/gribstream/backfill.py` | Runner orchestration | Thread `cutoff_id` through plan and run operations. | `prepare_gribstream_plan`, `run_gribstream_backfill` | Dry-run, persisted plan, live pilots. |
| `bootstrap/klga_tmax/implementation/src/klga_tmax/providers/gribstream/parser.py` | Parser | Preserve member `0`, map NBMQMD `06Z` to target date, emit response gaps. | `_member_for_row`, `_target_date_for_row`, `_missing_response_gaps`, `parse_gribstream_response` | Parser unit test and pilot gap rows. |
| `bootstrap/klga_tmax/implementation/src/klga_tmax/registry/cutoffs.py` | Registry seed | Add canonical cutoff row for `T_1245UTC`. | `CANONICAL_CUTOFFS` | `db migrate`; `validate foundation`; `validate gribstream`. |
| `bootstrap/klga_tmax/implementation/src/klga_tmax/validation/foundation.py` | Validation | Add exact `2026-06-28` UTC expectation for `T_1245UTC`. | `EXPECTED_2026_06_28_CUTOFF_UTC` | `validate foundation`. |
| `bootstrap/klga_tmax/implementation/src/klga_tmax/validation/gribstream.py` | Validation | Require all supported GribStream cutoff registry rows. | `validate_gribstream` | `validate gribstream`. |
| `bootstrap/klga_tmax/implementation/tests/test_gribstream_plan.py` | Tests | Cover both cutoff profiles, new credit totals, `asOf`, NBMQMD `06Z`, and request hash split. | GribStream planner tests | `python -m pytest -q`. |
| `bootstrap/klga_tmax/implementation/tests/test_gribstream_parser.py` | Tests | Add parser regression for member `0` and NBMQMD target-date mapping. | `test_parser_preserves_member_zero_and_maps_nbmqmd_06z_to_target_day` | `python -m pytest -q`. |
| `bootstrap/klga_tmax/implementation/tests/test_timezones_cutoffs.py` | Tests | Include `T_1245UTC` in canonical cutoff examples. | `test_2026_06_28_cutoff_examples_match_spec` | `python -m pytest -q`. |
| `bootstrap/klga_tmax/strategy_spec/context/KLGA_TMAX_05_GRIBSTREAM_T1245UTC_BACKFILL_IMPLEMENTATION_DEEP_DIVE.md` | Docs | Record implementation, evidence, gaps, runbook, and remaining work. | This document | Documentation quality gate. |

## Architecture and Control Flow

```mermaid
flowchart TD
    A["CLI: gribstream plan/run --cutoff-id T_1245UTC"] --> B["load settings and optional GribStream token"]
    B --> C["resolve live catalog/shared selectors per model"]
    C --> D["build_job_plan with cutoff profile"]
    D --> E["one chunk per model-date for asOf-backed models"]
    E --> F["request_payload_for_chunk with timesList, asOf, members, Tier B points"]
    F --> G["persist audit.gribstream_backfill_jobs and audit.gribstream_backfill_chunks"]
    G --> H["run_gribstream_backfill"]
    H --> I["POST /api/v2/{model}/timeseries"]
    I --> J["gzip raw NDJSON in bronze lineage"]
    J --> K["parse values and gaps"]
    K --> L["silver.grib_forecast_values"]
    K --> M["silver.availability_ledger"]
    K --> N["audit.gribstream_source_gaps"]
    N --> O["status by job and model"]
```

The planner remains dataset-path scoped. The request body does not include a model field because the endpoint path is `POST /api/v2/{model}/timeseries`. Request hashes include the model ID, endpoint, and payload, so two models with identical payloads do not collide. The profile change also changes the payload through `asOf` and, for NBMQMD, through `timesList`, so old and new cutoffs do not share request hashes.

## File-by-File Deep Dive

### `.gitignore`

The parent repo now ignores `/.secrets/`. This protects `weather_data_extraction/.secrets/gribstream_api_token.txt`, which is used for live GribStream commands but must not enter source control, generated docs, or logs.

### `bootstrap/klga_tmax/AGENTS.md`

The provider rules now state that local GribStream credentials live at `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\.secrets\gribstream_api_token.txt`. The documented command loads the file into `GRIBSTREAM_API_TOKEN` without printing the token. This aligns local operation with the GribStream skill rule that tokens stay out of prompts and artifacts.

### `bootstrap/klga_tmax/implementation/src/klga_tmax/cli.py`

The GribStream CLI accepts `--cutoff-id` for plan, run, and smoke commands. `_parse_gribstream_cutoff_id` rejects unsupported IDs before any catalog or database work. `gribstream inspect-config` now reports supported cutoff IDs and model summaries by cutoff, which makes the action plan auditable without reading Python code.

### `bootstrap/klga_tmax/implementation/src/klga_tmax/providers/gribstream/plan.py`

The previous hard-coded cutoff constants were replaced by `GribStreamCutoffProfile` entries. `DEFAULT_CUTOFF_ID` remains `T_MINUS_1_2045UTC`; `T1245_CUTOFF_ID` is `T_1245UTC`. The profile controls UTC cutoff time, target-day offset, archive-start offset, and NBMQMD max-18h valid hour.

`effective_target_start` is the key range function. It returns `catalog_archive_start + 1 day` for the prior-day profile and `catalog_archive_start` for `T_1245UTC`. `as_of_utc` subtracts each model buffer from the profile cutoff. `build_chunk` stores the selected cutoff ID on the chunk and includes it in chunk identity, while request hashing remains endpoint and model scoped.

### `bootstrap/klga_tmax/implementation/src/klga_tmax/providers/gribstream/catalog.py`

`spec_summary_rows` now takes `cutoff_id`. It computes target days and estimated total credits from the selected profile. For `T_1245UTC` through `2026-06-28`, it emits the requested total of `1,150,873` estimated credits.

### `bootstrap/klga_tmax/implementation/src/klga_tmax/providers/gribstream/backfill.py`

`prepare_gribstream_plan` and `run_gribstream_backfill` accept `cutoff_id`, validate it, and pass it into plan construction and model summaries. Resume behavior remains request-hash based. Completed or completed-empty chunks are skipped when `resume` is enabled; pilot commands used `--no-resume` to force fresh live validation.

### `bootstrap/klga_tmax/implementation/src/klga_tmax/providers/gribstream/parser.py`

The parser now treats member `0` as a real member, not as an omitted value. `_target_date_for_row` maps NBMQMD `target_date + 1 06Z` back to the source target date for single-date chunks. `_missing_response_gaps` compares observed coordinate, valid-time, member, and alias combinations against the chunk plan and emits aggregated gap rows. A fully empty successful response becomes one `empty_response` gap instead of a silent completed-empty chunk with no diagnostic evidence.

### `bootstrap/klga_tmax/implementation/src/klga_tmax/registry/cutoffs.py`

`CANONICAL_CUTOFFS` includes `T_1245UTC` with UTC timezone, local time `12:45`, target-day offset `0`, and a description that records New York and Stockholm summer display aliases.

### `bootstrap/klga_tmax/implementation/src/klga_tmax/validation/foundation.py`

The foundation validator now has a concrete `2026-06-28` expectation for `T_1245UTC`: `2026-06-28T12:45:00+00:00`. This catches timezone or offset regressions in the registry seed.

### `bootstrap/klga_tmax/implementation/src/klga_tmax/validation/gribstream.py`

The GribStream validator now requires every cutoff returned by `supported_cutoff_ids`, currently `T_MINUS_1_2045UTC` and `T_1245UTC`, to exist in `registry.cutoffs`. It still checks required GribStream tables and row lineage fields.

### `bootstrap/klga_tmax/implementation/tests/test_gribstream_plan.py`

The planner tests now assert old and new totals, old and new target starts, exact `T_1245UTC` `asOf` timestamps, NBMQMD `06Z`, and request hash separation by cutoff profile.

### `bootstrap/klga_tmax/implementation/tests/test_gribstream_parser.py`

This new parser regression creates a tiny gzipped NDJSON response with `member: 0` and NBMQMD `forecasted_time: 2026-06-29T06:00:00Z`. It proves the parsed row remains member `"0"`, target date `2026-06-28`, and cutoff `2026-06-28T12:45:00+00:00`.

### `bootstrap/klga_tmax/implementation/tests/test_timezones_cutoffs.py`

The canonical cutoff example set includes `T_1245UTC`, so the timezone test now fails if the new registry row is removed or its UTC conversion changes.

### `bootstrap/klga_tmax/strategy_spec/context/KLGA_TMAX_05_GRIBSTREAM_T1245UTC_BACKFILL_IMPLEMENTATION_DEEP_DIVE.md`

This document is the handoff record for the target-day GribStream backfill implementation, the live pilot evidence, and the remaining full-backfill execution step.

## Public Interfaces and Contracts

CLI examples:

```powershell
python -m klga_tmax.cli gribstream plan --cutoff-id T_1245UTC --job-id klga_t1245utc_full_backfill_v1 --end-date 2026-06-28 --coordinate-tier B --models gfs,hrrr,rap,nbm,gefsatmosmean,gefsatmos,ifsoper,ifsenfo,aifsoper,aifsenfo,aigefssfc,aigfssfc,rtma,urma,nbmqmd --persist
```

```powershell
python -m klga_tmax.cli gribstream run --cutoff-id T_1245UTC --job-id klga_t1245utc_full_backfill_v1 --end-date 2026-06-28 --coordinate-tier B --models gfs,hrrr,rap,nbm,gefsatmosmean,gefsatmos,ifsoper,ifsenfo,aifsoper,aifsenfo,aigefssfc,aigfssfc,rtma,urma,nbmqmd --resume
```

The request contract remains:

- Endpoint: `POST https://gribstream.com/api/v2/{model}/timeseries`.
- Time mode: non-empty `timesList`.
- Coordinates: Tier B, 10 KLGA pseudo-points.
- Metadata: `includeMetadata: ["index_updated_at"]`.
- Members: only sent for ensemble models after live catalog resolution.
- Auth: `Authorization: Bearer REDACTED`, with token loaded from `GRIBSTREAM_API_TOKEN`.

## Data Model, Persistence, and Migration Notes

No Alembic migration was required. The existing persistence model already stores:

- `audit.gribstream_backfill_jobs` for job-level cutoff, range, status, planned chunks, completed chunks, failed chunks, estimated credits, and config.
- `audit.gribstream_backfill_chunks` for model/date/request status, request hash, `asOf`, payload, rows, gaps, raw URI, and HTTP status.
- `audit.gribstream_source_gaps` for selector, member, coordinate, and empty-response diagnostics.
- `bronze.source_requests` and `bronze.source_records` for raw request/response lineage.
- `silver.grib_forecast_values` for normalized rows.
- `silver.availability_ledger` for availability metadata.

The seed path inserts `T_1245UTC` into `registry.cutoffs` through `python -m klga_tmax.cli db migrate` or `registry seed`.

## Error Handling, Edge Cases, and Failure Modes

Auth and rate limiting retain existing behavior: stop immediately on `401` or `403`; stop and preserve state on `429`; honor the GribStream client retry policy for transient server or transport errors. The planner still forces `asOf`-backed models to one target date per chunk. URMA can use multi-day chunks because it has no live `asOf`.

Known pilot gaps are now explicit:

- `ifsenfo` returned 50 distinct members for `2026-06-28`; expected member `0` was recorded as a gap.
- NBMQMD returned all 21 percentile selectors for the `2026-06-28` pilot at `2026-06-29T06:00Z`.
- High-row models produced missing selector/time gaps, especially around one or more peak-window valid times per alias. These are stored in `audit.gribstream_source_gaps` and should be reviewed during full-run monitoring.
- URMA returned 90 values for the one-day pilot where 100 values were expected, and the missing target-day peak temperature time was recorded as a gap.

## Security, Privacy, and Safety Review

The token is stored outside `bootstrap/klga_tmax` under `.secrets`, which is ignored by Git. Generated docs and CLI output do not include token content. `bronze.source_requests.request_headers_redacted` records `Authorization: Bearer REDACTED`.

The full backfill is quota-sensitive. The persisted full job estimates `1,150,873` credits before cache effects and empty-row reductions. A live launch should be monitored continuously because `429` stops the runner and preserves state, but it may still consume partial daily quota before stopping.

## Performance, Scalability, and Concurrency

The runner remains intentionally single-worker. `OneThreadRateLimiter` enforces 12-second spacing between authenticated calls. The full job has 20,806 chunks. At exactly 12 seconds per chunk, the minimum authenticated-call time is about 249,672 seconds, or 69.4 hours, before network latency, catalog lookup, parsing, DB writes, retries, and operator pauses.

The implementation uses `/timeseries` with sparse `timesList` and Tier B points in each request. It does not use `/runs`, all-run archival pulls, pressure levels, grids, or full horizons.

## Configuration and Environment

Required database variable:

```powershell
$env:KLGA_DB_URL = "postgresql+psycopg://postgres:root@127.0.0.1:5432/klga_tmax_research"
```

Required live GribStream variable:

```powershell
$env:GRIBSTREAM_API_TOKEN = (Get-Content "C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\.secrets\gribstream_api_token.txt" -Raw).Trim()
```

The GribStream CLI accepts:

- `--cutoff-id T_MINUS_1_2045UTC`
- `--cutoff-id T_1245UTC`

## Testing and Verification Evidence

All commands were run from:

```text
C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\klga_tmax\implementation
```

Non-DB verification:

```powershell
python -m pytest tests\test_gribstream_plan.py tests\test_gribstream_parser.py tests\test_timezones_cutoffs.py -q
```

Result: `14 passed in 4.71s`.

```powershell
python -m compileall src tests
```

Result: compile succeeded for edited source and tests.

```powershell
python -m pytest -q
```

Result: `63 passed in 16.45s`.

CLI help verification:

```powershell
python -m klga_tmax.cli --help
python -m klga_tmax.cli validate --help
python -m klga_tmax.cli gribstream --help
```

Result: all help commands exited successfully and listed GribStream subcommands.

DB verification:

```powershell
python -m klga_tmax.cli db migrate
```

Result: `registry.cutoffs: 6`, `registry.station_registry: 29`, `registry.stations: 29`.

```powershell
python -m klga_tmax.cli db inspect-contract
```

Result: `ok: true`, `failures: []`, `tables_checked: 28`, `indexes_checked: 30`, `cutoff_rows: 6`.

```powershell
python -m klga_tmax.cli validate foundation
```

Result: `ok: true`, `late_feature_rows: 0`, `target_instance_rows: 24`.

```powershell
python -m klga_tmax.cli validate gribstream
```

Result after live pilots: `ok: true`, `grib_forecast_values: 24980`, `grib_availability_rows: 23340`, `grib_completed_chunks: 45`, and required cutoff IDs `T_MINUS_1_2045UTC`, `T_1245UTC`.

Dry-run full plan:

```powershell
python -m klga_tmax.cli gribstream plan --cutoff-id T_1245UTC --job-id klga_t1245utc_full_backfill_v1 --end-date 2026-06-28 --coordinate-tier B --models all --dry-run
```

Result: 15 models, 20,806 chunks, 72 catalog snapshots, 0 selector gaps, 1,150,873 estimated credits.

Persisted full plan:

```powershell
python -m klga_tmax.cli gribstream plan --cutoff-id T_1245UTC --job-id klga_t1245utc_full_backfill_v1 --end-date 2026-06-28 --coordinate-tier B --models gfs,hrrr,rap,nbm,gefsatmosmean,gefsatmos,ifsoper,ifsenfo,aifsoper,aifsenfo,aigefssfc,aigfssfc,rtma,urma,nbmqmd --persist
```

Result: job `klga_t1245utc_full_backfill_v1` planned with 20,806 chunks, 15 models, 0 selector gaps, 1,150,873 estimated credits.

One-day live pilot:

```powershell
python -m klga_tmax.cli gribstream run --cutoff-id T_1245UTC --job-id klga_t1245utc_pilot_20260628_all_models --start-date 2026-06-28 --end-date 2026-06-28 --coordinate-tier B --models gfs,hrrr,rap,nbm,gefsatmosmean,gefsatmos,ifsoper,ifsenfo,aifsoper,aifsenfo,aigefssfc,aigfssfc,rtma,urma,nbmqmd --max-chunks 15 --no-resume
```

Result: 15 chunks fetched, 0 skipped, 0 failed, 6,800 rows upserted, 6,800 availability rows upserted, 340 gaps upserted.

Seven-day high-row live pilot:

```powershell
python -m klga_tmax.cli gribstream run --cutoff-id T_1245UTC --job-id klga_t1245utc_pilot_20260622_20260628_highrow --start-date 2026-06-22 --end-date 2026-06-28 --coordinate-tier B --models gfs,hrrr,rap,nbm --max-chunks 28 --no-resume
```

Result: 28 chunks fetched, 0 skipped, 0 failed, 20,160 rows upserted, 20,160 availability rows upserted, 2,240 gaps upserted.

Identity check:

```sql
SELECT count(*) AS rows,
       count(*) FILTER (
         WHERE model_id IS NULL
            OR member IS NULL
            OR grid_point_id IS NULL
            OR forecasted_at_utc IS NULL
            OR forecasted_time_utc IS NULL
            OR variable_alias IS NULL
            OR variable_name IS NULL
            OR source_request_id IS NULL
            OR source_record_id IS NULL
            OR request_sha256 IS NULL
            OR availability_method IS NULL
       ) AS missing_identity
FROM silver.grib_forecast_values
WHERE cutoff_id = 'T_1245UTC'
  AND target_date BETWEEN '2026-06-22' AND '2026-06-28';
```

Result: `rows: 24260`, `missing_identity: 0`.

## Operational Runbook

Check full job status:

```powershell
python -m klga_tmax.cli gribstream status --job-id klga_t1245utc_full_backfill_v1
```

Current full job state after this implementation:

- Status: `planned`.
- Planned chunks: `20,806`.
- Completed chunks: `0`.
- Failed chunks: `0`.
- Estimated credits: `1,150,873`.
- Model ranges: `hrrr` from `2014-07-30`, `rtma` from `2018-01-01`, `urma` from `2024-04-30`, `nbm` from `2020-09-29`, `gfs` from `2021-03-22`, `rap` from `2021-02-22`, `gefsatmosmean/gefsatmos` from `2020-10-01`, `ifsoper` from `2024-02-28`, `ifsenfo` from `2024-03-01`, `nbmqmd` from `2026-01-31`, `aifsoper` from `2025-02-25`, `aifsenfo` from `2025-07-02`, `aigefssfc` from `2025-06-01`, `aigfssfc` from `2026-04-16`.

Launch the full backfill:

```powershell
$env:KLGA_DB_URL = "postgresql+psycopg://postgres:root@127.0.0.1:5432/klga_tmax_research"
$env:GRIBSTREAM_API_TOKEN = (Get-Content "C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\.secrets\gribstream_api_token.txt" -Raw).Trim()
python -m klga_tmax.cli gribstream run --cutoff-id T_1245UTC --job-id klga_t1245utc_full_backfill_v1 --end-date 2026-06-28 --coordinate-tier B --models gfs,hrrr,rap,nbm,gefsatmosmean,gefsatmos,ifsoper,ifsenfo,aifsoper,aifsenfo,aigefssfc,aigfssfc,rtma,urma,nbmqmd --resume
```

Resume after a clean stop or transient failure with the same command. Do not change `--cutoff-id`, model list, coordinate tier, or end date unless intentionally creating a different job scope.

## Compatibility, Rollback, and Upgrade Notes

Backward compatibility is preserved. Existing callers that do not pass `--cutoff-id` still use `T_MINUS_1_2045UTC`. Existing database tables remain compatible because no columns or constraints changed.

Rollback path:

- Code rollback removes the new CLI option behavior and profile support.
- Seed rollback is not required for safety; an extra registry cutoff row is inert unless referenced by jobs or target instances.
- Existing `T_1245UTC` pilot rows can remain in bronze/silver/audit tables. They are isolated by `cutoff_id` and request hash.

## Known Limitations and Follow-Up Work

The full historical backfill is ready but not completed. It should run under direct monitoring because the expected duration is multi-day and the credit estimate is material.

The live pilots show selector/time gaps for high-row models. The gaps are now visible and queryable, but the modeling layer still needs a policy for missing feature values by model, variable, and target hour.

The `index_updated_at` field is preserved as diagnostic metadata. It is not treated as a first-availability timestamp. The live-style cutoff still depends on conservative buffer assumptions as required by the GribStream time-semantics guidance.

## Reviewer Checklist

- [x] `T_1245UTC` exists in registry seed and foundation validation.
- [x] Prior-day `T_MINUS_1_2045UTC` remains supported.
- [x] CLI exposes `--cutoff-id` for plan, run, and smoke.
- [x] NBMQMD uses `target_date + 1 at 06:00 UTC`.
- [x] Request hash changes when cutoff profile changes.
- [x] Member `0` is parsed as `"0"`.
- [x] Missing selector/member/time combinations create source gaps.
- [x] Dry-run plan resolves live selectors for all 15 models with zero selector gaps.
- [x] Full job is persisted as `klga_t1245utc_full_backfill_v1`.
- [x] One-day all-model live pilot completed with HTTP 200 on every chunk.
- [x] Seven-day high-row live pilot completed with HTTP 200 on every chunk.
- [x] `validate gribstream` passes after live pilots.
- [x] Full backfill is explicitly marked remaining work, not claimed complete.
