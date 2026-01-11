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

## 3) Operational checks
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
  - run “leakage audit query”:
    - ensure no `chosen_run_utc > asof_utc`

## 4) Common failure modes
- Station mapping mismatch (issuedby -> ICAO not present in IEM)
- IEM endpoint 503 (rate limiting / load)
- Partial years missing in CLI (rare gaps)
- Model run cycles differ by model and era (esp. NBS pre/post Feb 2020)

