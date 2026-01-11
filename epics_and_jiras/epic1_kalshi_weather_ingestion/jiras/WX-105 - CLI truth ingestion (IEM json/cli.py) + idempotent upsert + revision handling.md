# WX-105 — CLI truth ingestion (IEM json/cli.py) + idempotent upsert + revision handling

## Objective
Implement ingestion of the settlement truth used by Kalshi weather markets:
- NWS Daily Climate Report (CLI) values, via IEM `json/cli.py` backend.

We use IEM because it provides programmatic access to parsed CLI daily data for stations.

## IEM endpoint (authoritative)
- `GET https://mesonet.agron.iastate.edu/json/cli.py?station=<STATION>&year=<YYYY>&fmt=json`

Arguments are documented in IEM help page:
- station, year, fmt (json/csv)

## Ingestion approach (scalable + resumable)
For each station, for each year in the requested date range:
1) Fetch year payload
2) Parse day-by-day entries
3) Upsert into `cli_daily` keyed by (station_id, target_date_local)
4) Store raw payload hash and retrieved_at_utc

## Revision handling (important for Kalshi)
Kalshi notes that climate report values can be revised; markets settle on the *final* report.
Therefore:
- The ingest must be able to re-run and overwrite prior stored values if the IEM-provided value changes.

Schema strategy:
- Always upsert `tmax_f` and `tmin_f`
- Keep `previous_value` history optional:
  - If implementing history now: store changes into `cli_daily_revision` table.
  - If deferring: at least store `last_updated_at_utc` each upsert.

## Acceptance Criteria
- [ ] For each of the 5 stations, CLI data for at least one full year is ingested successfully.
- [ ] Re-running ingestion does not create duplicates (unique key enforced).
- [ ] If an existing row’s `tmax_f` changes, it is overwritten and `updated_at_utc` changes.
- [ ] Raw payload hash is stored.
- [ ] End-to-end integration test:
  - Pick a station/year (e.g., KMIA 2025)
  - Assert row count > 300 and that tmax is present for most days.

## Source references
- IEM CLI JSON documentation: /json/cli.py?help=
- Kalshi weather markets settlement and DST handling.

