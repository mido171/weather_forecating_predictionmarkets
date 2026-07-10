# T03-T05 GribStream and NWP Foundation Runbook

Last updated: 2026-06-24

> **Relocation note (2026-07-10):** Historical paths below are retained as
> provenance. For current work, translate `code/src` to `src`, `code/tests` to
> `tests`, `migrations/postgres` to `db/migrations/postgres`, and repo-local
> `data`/run artifacts to `HKG_TMAX_DATA_ROOT`/`HKG_TMAX_RUN_ROOT`. Do not run
> an old absolute-path command verbatim.

## Purpose

This runbook documents the T03-T05 foundation runner and its GribStream safety rules.

Tasks covered:

- T03: GribStream catalog, coverage, licence, and quota audit.
- T04: NWP database, object storage, and lineage migrations.
- T05: canonical location, station, and geospatial registry.

## Files

- Runner: `scripts/run_t03_t05_foundation_tasks.py`
- Status checker: `scripts/check_t03_t05_status.py`
- Test: `code/tests/test_t03_t05_foundation_tasks.py`
- T03 migration: `migrations/postgres/20260624_0004_t03_gribstream_catalog_registry.sql`
- T04 migration: `migrations/postgres/20260624_0005_t04_nwp_storage_lineage.sql`
- T05 migration: `migrations/postgres/20260624_0006_t05_location_station_geospatial_registry.sql`

## Evidence Folders

- `experiments/0210_gribstream_catalog_coverage_licence_quota_audit/`
- `experiments/0211_nwp_database_object_storage_migrations/`
- `experiments/0212_canonical_location_station_geospatial_registry/`

## GribStream Safety Policy

- Use one thread only.
- Default authenticated query spacing is 12 seconds between attempts.
- Default max attempts per query is 3.
- `400`, `401`, `403`, and `404` stop immediately.
- `429`, `500`, `502`, `503`, `504`, timeouts, and connection resets are treated as transient, but retries are bounded.
- `Retry-After` is honored up to the configured cap.
- If a real `429` is encountered, remaining probes are marked `blocked_rate_limit_safety_stop` to avoid repeated rate-limit traffic.
- Tokens must stay in `secrets/local/gribstream.env`; logs store only hashes, statuses, row counts, and sanitized snippets.

## Background Run

Use PowerShell `Start-Process` with a hidden window and redirected logs. The process should run from the repo root.

```powershell
$repo = "C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$out = Join-Path $repo "run_logs\t03_t05_background_$stamp.out.log"
$err = Join-Path $repo "run_logs\t03_t05_background_$stamp.err.log"
Start-Process -FilePath "$repo\.venv\Scripts\python.exe" `
  -ArgumentList @("scripts\run_t03_t05_foundation_tasks.py", "--api-min-interval-seconds", "12", "--api-max-attempts", "3") `
  -WorkingDirectory $repo `
  -RedirectStandardOutput $out `
  -RedirectStandardError $err `
  -WindowStyle Hidden `
  -PassThru
```

## Status Check

```powershell
.\.venv\Scripts\python.exe scripts\check_t03_t05_status.py
```

Important status files:

- `experiments/0210_gribstream_catalog_coverage_licence_quota_audit/logs/t03_t05_background_status.json`
- `experiments/0210_gribstream_catalog_coverage_licence_quota_audit/logs/gribstream_api_events.jsonl`
- `experiments/0210_gribstream_catalog_coverage_licence_quota_audit/coverage_probe_results.csv`

## Recovery Modes

Use recovery modes only after the relevant artifacts already exist.

```powershell
.\.venv\Scripts\python.exe scripts\run_t03_t05_foundation_tasks.py --reuse-existing-coverage-probes
```

This reuses `coverage_probe_results.csv` but may still refresh public catalog pages.

```powershell
.\.venv\Scripts\python.exe scripts\run_t03_t05_foundation_tasks.py --reuse-existing-t03-artifacts
```

This is the no-network finalization path. It reuses existing T03 JSON/CSV artifacts, then runs T04/T05 generation, migrations, DB loads, tests/quality records, completion records, and task moves. Use this after authenticated probes have completed and a later non-API step fails.

The runner uses long-path-safe file writes/copies because this checkout path plus completed task names can exceed normal Windows path limits.

## Completed 2026-06-24 Run

- Final status: `complete` / `passed`.
- Completed task folders: `tasks/HKG_T24_A_TO_Z_CODEX_IMPLEMENTATION/tasks/completed/T03_gribstream_catalog_coverage_licence_quota_audit`, `T04_nwp_database_object_storage_migrations`, and `T05_canonical_location_station_geospatial_registry`.
- Authenticated API event count: 52.
- Coverage rows: 23.
- Probe outcome counts: 1 `pass_hkg_rows_returned`, 17 `blocked_probe_exception`, 5 `blocked_no_surface_temperature_selector`.
- HTTP status counts: 1 `200`, 22 blank because those failures did not receive HTTP responses.
- No `429`, `400`, `401`, or `5xx` response was recorded in the completed probe run.
- The one successful authenticated probe was `gfs` `/runs`, returning 147 HKG rows.
- The 17 probe exceptions were `ConnectTimeout` network blockers and were recorded as blockers, not hidden as passes.
- Main PostgreSQL migrations, T03/T05 DB loads, isolated temp DB migration test, focused tests, and secret scan passed.

## Completion Rule

Only move T03-T05 task folders from `tasks/not-completed/` to `tasks/completed/` after the runner has:

- written all required task artifacts;
- applied or explicitly skipped DB migrations with documented status;
- updated `TASK_STATUS_INDEX.csv`;
- written completion records;
- passed the focused test suite;
- passed the secret scan.
