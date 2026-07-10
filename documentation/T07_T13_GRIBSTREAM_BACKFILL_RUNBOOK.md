# T07-T13 GribStream Backfill Runbook

Last updated: 2026-06-24

## Purpose

T07-T12 are the real GribStream acquisition tasks after the T06 client smoke. The runner plans resumable `/runs` chunks across the required GFS, GEFS, IFS, AI-model, CWA WRF, and secondary-model datasets using live shared-parameter selectors.

T13 is tracked here only as a blocker because HKO ARWF exact-vintage collection is not a GribStream dataset. It needs a separate HKO collector.

## Files

- Runner: `scripts/run_t07_t13_gribstream_backfill.py`
- Status checker: `scripts/check_t07_t13_gribstream_status.py`
- Reusable client/storage code: `code/src/hkg_tmax/gribstream/`
- Focused tests: `code/tests/test_t07_t13_gribstream_backfill.py`
- Evidence folder: `experiments/0214_t07_t13_gribstream_backfill/`

## Evidence and Logs

- Status: `experiments/0214_t07_t13_gribstream_backfill/logs/t07_t13_status.json`
- Planned chunks: `experiments/0214_t07_t13_gribstream_backfill/planned_chunks.csv`
- Executed chunks: `experiments/0214_t07_t13_gribstream_backfill/executed_chunks.csv`
- Resume ledger: `experiments/0214_t07_t13_gribstream_backfill/resume_ledger.jsonl`
- API event log: `experiments/0214_t07_t13_gribstream_backfill/logs/gribstream_api_events.jsonl`
- Blockers: `experiments/0214_t07_t13_gribstream_backfill/blockers.csv`
- Raw objects: `data/_pipeline_internal/raw/gribstream/<dataset>/runs/...`

## Current Acquisition Strategy

- Use one authenticated thread.
- Use `full` mode for the ongoing fair backfill wave.
- Use live GribStream shared-parameter catalog resolution instead of the flawed early T03 surface selector map.
- Use 0-84h leads and 00/06/12/18 UTC run windows.
- Use 132 canonical HKG/HKO/station/reference locations from `catalog.location`.
- Chunk ensemble members into groups of 5 by default.
- Cap each run by estimated credits so it can resume safely without exceeding the daily quota.
- CWA WRF is ordered newest-to-oldest because its archive is short-retention.

## Commands

Check progress without making API calls:

```powershell
.\.venv\Scripts\python.exe scripts\check_t07_t13_gribstream_status.py
```

Resume the full fair backfill wave:

```powershell
.\.venv\Scripts\python.exe scripts\run_t07_t13_gribstream_backfill.py --mode full --credit-budget 85000 --api-min-interval-seconds 12 --api-max-attempts 3 --pause-on-429-seconds 300
```

Start the same wave in the background:

```powershell
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
Start-Process -FilePath '.\.venv\Scripts\python.exe' `
  -ArgumentList @('scripts\run_t07_t13_gribstream_backfill.py','--mode','full','--credit-budget','85000','--api-min-interval-seconds','12','--api-max-attempts','3','--pause-on-429-seconds','300') `
  -WorkingDirectory (Get-Location) `
  -RedirectStandardOutput "run_logs\t07_t13_backfill_$stamp.out.log" `
  -RedirectStandardError "run_logs\t07_t13_backfill_$stamp.err.log" `
  -WindowStyle Hidden
```

Run focused verification:

```powershell
.\.venv\Scripts\python.exe -m pytest code\tests\test_t06_gribstream_resumable_runs_client.py code\tests\test_t07_t13_gribstream_backfill.py -q
```

## Credential Note

Use `secrets/local/gribstream.env` as the canonical project credential. The runner now prefers `GRIBSTREAM_API_KEY` and the project credential file before any legacy `GRIBSTREAM_API_TOKEN` environment variable.

On 2026-06-24, a stale `GRIBSTREAM_API_TOKEN` environment variable caused six `401 Unauthorized` smoke attempts before credential precedence was fixed. Those failures are recorded in `executed_chunks.csv`; later requests using the project credential returned HTTP 200.

## Smoke Evidence

After the credential fix, the runner fetched:

- `cwawrf15`, `temperature_2m`, run date `2026-06-23`
- HTTP status: 200
- Raw rows: 7,920
- DB point rows: 7,920
- Rejected rows: 0

Two `cwawrf15` June 21 chunks returned HTTP 200 with zero rows; those are preserved as completed-empty evidence.

## Blockers

- T13 `hko_arwf`: not available through GribStream; needs separate HKO ARWF collector.
- T12 `uvi`: registered as secondary-model decision evidence, but no active Tmax-relevant shared-parameter mapping is being acquired by this runner.
