# Epic #1 Architecture (Java Spring Boot + MySQL)

## 1) Modules
- `models`: JPA entities, enums, and Flyway migrations (single source of truth for schema).
- `ingestion-service`: Spring Boot service that runs backfills and audit reports.
- `common`: shared utilities (time, HTTP clients, hashing).

`mvn clean install` must succeed from repo root.

## 2) Data flow (backfill)
For each station and target date range:
1) Resolve Kalshi series metadata and upsert `kalshi_series` + `station_registry`.
2) CLI truth ingestion (IEM `/json/cli.py`): upsert `cli_daily`.
3) MOS run ingestion (IEM `/cgi-bin/request/mos.py`): upsert `mos_run`.
4) Materialize as-of features:
   - Compute `asof_utc` from station ZoneId + policy time.
   - Select the latest MOS run with `runtime_utc <= asof_utc`.
   - Extract `tmax_f` for target date and upsert `mos_asof_feature`.

## 3) Idempotency
Natural uniqueness keys:
- `kalshi_series` unique on `series_ticker`
- `station_registry` unique on `station_id`
- `cli_daily` unique on `(station_id, target_date_local)`
- `mos_run` unique on `(station_id, model, runtime_utc)`
- `mos_asof_feature` unique on `(station_id, target_date_local, asof_policy_id, model)`

All ingestion writes use MySQL upserts to allow restarts without cleanup.

## 4) Auditing and raw payloads
Every ingestion table stores:
- `retrieved_at_utc` (when it was fetched)
- `raw_payload_hash` (or `raw_payload_hash_ref`) for payload dedupe

MOS as-of features also store:
- `asof_utc`, `asof_local`, `station_zoneid`, and `chosen_runtime_utc`

## 5) External dependencies
Kalshi:
- API base: https://api.elections.kalshi.com/trade-api/v2
- Series metadata: https://api.elections.kalshi.com/trade-api/v2/series/{SERIES_TICKER}
- Weather market rules: https://help.kalshi.com/markets/popular-markets/weather-markets

IEM (Iowa State Mesonet):
- Base: https://mesonet.agron.iastate.edu
- MOS request: https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py
- MOS request help: https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py?help=
- MOS archive notes: https://mesonet.agron.iastate.edu/mos/
- CLI JSON: https://mesonet.agron.iastate.edu/json/cli.py
- CLI JSON help: https://mesonet.agron.iastate.edu/json/cli.py?help=
- CLI day window notes: https://mesonet.agron.iastate.edu/nws/clitable.php
