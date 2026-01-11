# WX-106 — MOS raw run ingestion (IEM /cgi-bin/request/mos.py) for GFS/MEX/NAM/NBS/NBE with backfill windows

## Objective
Ingest MOS forecasts (as archived by IEM) for a station and model, including run-time metadata, so later we can derive T-1 as-of features.

## Authoritative IEM sources
1) MOS archive overview (model coverage, archived cycles, and /api/1/mos.* examples): https://mesonet.agron.iastate.edu/mos/
2) MOS download interface (archive back to June 2000; NBS cycle rules; variable definitions): https://mesonet.agron.iastate.edu/mos/fe.phtml
3) Bulk backend documentation: https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py?help=

## Required models (start with these 5)
- GFS
- MEX (GFS-X)
- NAM
- NBS
- NBE

## Required variables (minimum)
From IEM MOS download interface variable definitions:
- `n_x` = Max/Min Temp [F]
- We will parse `n_x` to extract the “max” component for each target day.

(We may also later store tmp/dpt/etc, but Epic #1 minimum is daily max.)

## Backfill strategy (idempotent)
### Recommended ingestion units
Ingest by **runtime window** per station+model:
- Choose a date window in UTC (e.g., per day: sts=YYYY-MM-DDT00:00Z, ets=YYYY-MM-(DD+1)T00:00Z)
- Call:
  `GET /cgi-bin/request/mos.py?station=<ICAO>&model=<MODEL>&sts=<STS>&ets=<ETS>&format=json`
- This returns all runs in that period.

### Critical: model cycle nuances (NBS)
IEM notes:
- After 25 Feb 2020: NBS archived only at 1, 7, 13, 19 UTC
- Before 25 Feb 2020: NBS archived only at 0, 7, 12, 19 UTC

So the ingestion must NOT assume fixed 00/06/12/18 cycles for NBS.

## Storage
For each MOS run returned:
- Upsert `mos_run` (station_id, model, runtime_utc)
- Store raw payload hash + retrieved_at_utc
Optionally store per-variable structures:
- either normalized `mos_value` rows
- or store the raw run payload and defer parsing until WX-107

## Acceptance Criteria
- [ ] Ingest succeeds for each station+model for a small date range (e.g., 7 days).
- [ ] Upsert prevents duplicates: re-running ingests same window does not increase row count.
- [ ] The service handles missing cycles gracefully (no failures if a cycle doesn’t exist).
- [ ] Retries/backoff are used for transient failures (e.g., 503).
- [ ] Raw payload hash stored.

