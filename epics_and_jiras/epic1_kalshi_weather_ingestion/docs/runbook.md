# Runbook (Epic #1)

## 1) Required configuration
- MySQL connection (URL, user, password)
- Kalshi base URL: https://api.elections.kalshi.com/trade-api/v2
- IEM base URL: https://mesonet.agron.iastate.edu
- Station/timezone overrides (optional)

## 2) Backfill command patterns (internal endpoints or CLI)
You will implement one of:
- Spring Boot CommandLineRunner jobs
- Spring Batch jobs
- or an internal REST endpoint restricted to localhost/VPN

Required job inputs:
- series_ticker (e.g., KXHIGHMIA)
- date_start_local (YYYY-MM-DD)
- date_end_local (YYYY-MM-DD)
- asof_policy_id (or default)
- models list (default: GFS, MEX, NAM, NBS, NBE)

## 3) Backfill jobs + checkpoints
Jobs are run via CommandLineRunner with `backfill.*` properties.

Units of work + checkpoint cursor semantics:
- `kalshi_series_sync`: per series_ticker (station). No cursor fields; status only.
- `cli_ingest_year`: per year slice inside the date range. `cursor_date` is the last completed local date.
- `mos_ingest_window`: per UTC window (default 1 day). `cursor_runtime_utc` is the end of the last completed window.
- `mos_asof_materialize_range`: per target day. `cursor_date` is the last completed target_date_local.

Checkpoints are updated after each unit and refreshed by a heartbeat every ~30s while running.

Example (CLI backfill):
`mvn -pl ingestion-service spring-boot:run -Dspring-boot.run.arguments="--backfill.enabled=true --backfill.job=cli_ingest_year --backfill.series-ticker=KXHIGHMIA --backfill.date-start-local=2023-01-01 --backfill.date-end-local=2023-12-31"`

## 4) Operational checks
- Before backfill:
  - confirm station registry resolved and validated
  - confirm `asof_policy` exists
- During:
  - monitor logs for:
    - retry storms
    - 503 backoffs
    - mapping failures
  - monitor DB row counts increment
- After:
  - run "leakage audit query":
    - ensure no `chosen_run_utc > asof_utc`
  - review the mos as-of completeness report in logs (missing % per model + top missing reasons)

## 5) Common failure modes
- Station mapping mismatch (issuedby -> ICAO not present in IEM)
- IEM endpoint 503 (rate limiting / load)
- Partial years missing in CLI (rare gaps)
- Model run cycles differ by model and era (esp. NBS pre/post Feb 2020)
