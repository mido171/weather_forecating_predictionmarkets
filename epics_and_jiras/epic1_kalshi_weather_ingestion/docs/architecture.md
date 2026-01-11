# Epic #1 Architecture (Java Spring Boot + MySQL)

## 1) High-level components
1) **kalshi-client**
   - Calls Kalshi public API to resolve:
     - series ticker → settlement source URL(s)
     - contract terms URLs (stored for audit)

2) **station-registry**
   - Maps Kalshi settlement station → IEM station identifier (ICAO)
   - Stores timezone + standard offset
   - Validates station exists in IEM CLI and MOS archives

3) **iem-cli-ingestor**
   - Fetches CLI daily truth via IEM `/json/cli.py`
   - Stores:
     - `cli_daily` (normalized)
     - `raw_payload` (compressed JSON)
     - `retrieved_at_utc`

4) **iem-mos-ingestor**
   - Fetches MOS runs via IEM `/cgi-bin/request/mos.py` (bulk backfill) and optionally `/api/1/mos.json` (latest)
   - Stores:
     - `mos_run` (one row per station/model/runtime)
     - `mos_value` (variable/time series OR extracted daily fields)
     - `raw_payload`

5) **asof-feature-materializer**
   - For each target day T and as-of policy:
     - selects latest MOS run <= asOfUtc
     - fetches the chosen run payload and extracts `tmax` feature for day T for each model
     - stores `mos_asof_feature`
   - MUST enforce no leakage.

6) **backfill-orchestrator (idempotent)**
   - Runs backfills with checkpoints:
     - by station
     - by model
     - by date range
   - Restartable: picks up exactly where it stopped.

## 2) Suggested Maven module layout (Epic #1)
repo-root/
  pom.xml (packaging=pom)
  models/                <-- common module, JPA entities + enums + DB migration definitions
  ingestion-service/     <-- Spring Boot service that runs jobs + exposes internal endpoints
  common/                <-- shared utils: time, http clients, retries, hashing
  docs/                  <-- architecture/time semantics/runbook
  agents.md

`mvn clean install` must succeed at root.

## 3) Data flow (backfill)
For each station + target date range:

A) Ensure station exists in `station_registry`
   - resolve from Kalshi series
   - validate mapping to IEM station id
   - store timezone + standard offset

B) CLI truth ingestion
   - for each year in date range:
     - GET IEM `/json/cli.py?station=<ICAO>&year=<YYYY>&fmt=json`
     - upsert `cli_daily` rows

C) MOS runs ingestion (raw)
   - for each model (GFS, MEX, NAM, NBS, NBE):
     - request MOS runs over a time window using `/cgi-bin/request/mos.py?station=<ICAO>&model=<MODEL>&sts=<...>&ets=<...>&format=json`
     - upsert `mos_run` + store payload

D) Materialize as-of features
   - for each day T:
     - compute asOfUtc from station timezone + policy
     - for each model:
       - choose max runtime <= asOfUtc
       - extract tmax for target date T
     - store `mos_asof_feature`

## 4) Idempotency strategy
- Each table uses natural unique keys:
  - `kalshi_series` unique (series_ticker)
  - `station_registry` unique (station_id)
  - `cli_daily` unique (station_id, target_date_local)
  - `mos_run` unique (station_id, model, runtime_utc)
  - `mos_asof_feature` unique (station_id, target_date_local, asof_policy_id, model)
- Inserts are UPSERT (MySQL `INSERT ... ON DUPLICATE KEY UPDATE`)
- Raw payloads are stored with SHA-256 hash for dedupe and audit.

## 5) Operational safety
- Respect IEM service rate limits (treat 503 as backoff condition)
- Use bounded concurrency (configurable thread pool)
- Use retries with jittered exponential backoff
- Persist checkpoints frequently
- External HTTP calls use the common HardenedWebClient with connect/read timeouts and retry logging
