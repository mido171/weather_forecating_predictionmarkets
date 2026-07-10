# T06 GribStream Resumable Runs Client Runbook

Last updated: 2026-06-24

> **Relocation note (2026-07-10):** Historical paths below are retained as
> provenance. Current code and tests live under `src` and `tests`; acquisition
> policy lives under `config/acquisition`; bulk data, ledgers, logs, and run
> artifacts resolve through `HKG_TMAX_DATA_ROOT` and `HKG_TMAX_RUN_ROOT`.
> Do not run an old repo-local data command verbatim.

## Purpose

T06 implements the reusable GribStream `/runs` client and proves the raw landing, resume ledger, normalization, and PostgreSQL lineage path needed by T07+ acquisition tasks.

T06 is not the full historical GFS backfill. T07 is responsible for the real chunked historical acquisition.

## Files

- Client package: `code/src/hkg_tmax/gribstream/`
- Runner: `scripts/run_t06_gribstream_resumable_runs_client.py`
- Status checker: `scripts/check_t06_gribstream_status.py`
- Focused tests: `code/tests/test_t06_gribstream_resumable_runs_client.py`
- Policy entrypoint: `config/acquisition_policy.yaml`

## Evidence and Data

- Evidence folder: `experiments/0213_gribstream_resumable_runs_client/`
- Status file: `experiments/0213_gribstream_resumable_runs_client/logs/t06_status.json`
- Resume ledger: `experiments/0213_gribstream_resumable_runs_client/resume_ledger.jsonl`
- API event log: `experiments/0213_gribstream_resumable_runs_client/logs/gribstream_api_events.jsonl`
- Raw object: `data/_pipeline_internal/raw/gribstream/gfs/runs/run_time_utc=20260623_000000/ecfb27dcebbbfbf058049cf321478c6309cacc9dca381e797697d6a80b3715f4.ndjson.gz`

## Completed Smoke

- Dataset: `gfs`
- Endpoint: `/api/v2/gfs/runs`
- Selector: `TMP` / `2 m above ground` / empty `info`
- Alias: `temperature_2m`
- Run time: `2026-06-23T00:00:00Z`
- Valid range: `2026-06-23T00:00:00Z` through `2026-06-25T00:00:00Z`
- Lead range: 0 through 2880 minutes
- Locations: 132 canonical `catalog.location` rows
- Raw rows: 6,468
- DB point rows: 6,468
- Rejected normalized rows: 0
- Secret scan: passed

## GribStream Runtime Policy

- Use one thread only.
- Default spacing is 12 seconds between authenticated attempts.
- Honor `Retry-After`.
- If `429` has no `Retry-After`, pause for 300 seconds before retry.
- Retry only transient `429`, `500`, `502`, `503`, `504`, timeouts, and connection resets.
- Do not repeatedly retry `400`, `401`, `403`, or `404`.
- Do not log or print the API token.

## Commands

Run focused tests:

```powershell
.\.venv\Scripts\python.exe -m pytest code\tests\test_t06_gribstream_resumable_runs_client.py
```

Run the T06 smoke:

```powershell
.\.venv\Scripts\python.exe scripts\run_t06_gribstream_resumable_runs_client.py --mode smoke
```

Check status without rerunning the API:

```powershell
.\.venv\Scripts\python.exe scripts\check_t06_gribstream_status.py
```

Dry-run request planning without a live authenticated `/runs` query:

```powershell
.\.venv\Scripts\python.exe scripts\run_t06_gribstream_resumable_runs_client.py --mode dry-run
```

## Resume Behavior

The runner derives a canonical request SHA-256 from stable request JSON. If a final raw object already exists for that request, it reuses the object and continues normalization/DB loading without another GribStream call.

If a `.part` file exists, the client discards it and retries the same canonical request. The DB path is idempotent through `raw_audit.acquisition_request.request_sha256` and the `nwp_core.point_value` primary key.

## Known T06 Incident

During completion on 2026-06-24, the first successful HTTP 200 exposed a local Windows long-path write bug before raw persistence. A second HTTP 200 saved the raw object, then the next resume reused that object and completed DB ingest without another API call.

No `429`, `400`, `401`, or `5xx` responses were recorded in the T06 API event log.
