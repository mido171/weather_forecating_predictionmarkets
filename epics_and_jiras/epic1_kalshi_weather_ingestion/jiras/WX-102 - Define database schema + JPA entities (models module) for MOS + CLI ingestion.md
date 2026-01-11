# WX-102 — Define database schema + JPA entities (models module) for MOS + CLI ingestion

## Objective
Define a normalized, audit-friendly MySQL schema and matching JPA entities for:
- Kalshi series metadata
- Station registry + mapping overrides
- CLI daily truth values
- MOS runs (raw) + extracted as-of features
- Ingestion checkpoints (restartability)

## Must-support series / stations (day 1)
Use Kalshi series endpoints to confirm settlement sources and store them:

| Series ticker | City | Kalshi settlement source (NWS CLI) | issuedby | Derived IEM stationId | Timezone |
|---|---|---|---|---|---|
| KXHIGHNY | NYC (Central Park) | https://forecast.weather.gov/product.php?site=OKX&product=CLI&issuedby=NYC | NYC | KNYC | America/New_York |
| KXHIGHPHIL | Philadelphia | https://forecast.weather.gov/product.php?site=PHI&product=CLI&issuedby=PHL | PHL | KPHL | America/New_York |
| KXHIGHMIA | Miami | https://forecast.weather.gov/product.php?site=MFL&product=CLI&issuedby=MIA | MIA | KMIA | America/New_York |
| KXHIGHCHI | Chicago (Midway) | https://forecast.weather.gov/product.php?site=LOT&product=CLI&issuedby=MDW | MDW | KMDW | America/Chicago |
| KXHIGHLAX | Los Angeles (LAX) | https://forecast.weather.gov/product.php?site=LOX&product=CLI&issuedby=LAX | LAX | KLAX | America/Los_Angeles |


## Proposed tables (minimum)
### `kalshi_series`
- `series_ticker` (PK)
- `title`
- `category`
- `settlement_source_name`
- `settlement_source_url`  (forecast.weather.gov product.php link)
- `contract_terms_url` (PDF)
- `contract_url` (product certification PDF)
- `retrieved_at_utc`
- `raw_json` (optional)

### `station_registry`
- `station_id` (PK) — IEM/MOS/CLI station ID we use (e.g., KMIA)
- `issuedby` — from Kalshi settlement source (e.g., MIA)
- `wfo_site` — from settlement source (e.g., MFL)
- `series_ticker` (FK)
- `zone_id` (IANA string)
- `standard_offset_minutes` (INT; e.g., -300)
- `mapping_status` (ENUM: AUTO_OK, NEEDS_OVERRIDE)
- `created_at_utc`, `updated_at_utc`

### `station_override`
- `issuedby` (PK)
- `station_id_override` (e.g., KNYC)
- `zone_id_override`
- `notes`

### `cli_daily`
- `station_id` (FK)
- `target_date_local` (DATE)
- `tmax_f` (DECIMAL(5,2))
- `tmin_f` (DECIMAL(5,2), nullable)
- `raw_payload_hash` (CHAR(64))
- `retrieved_at_utc` (TIMESTAMP)
- UNIQUE(station_id, target_date_local)

### `mos_run`
- `station_id` (FK)
- `model` (ENUM: GFS, MEX, NAM, NBS, NBE)
- `runtime_utc` (TIMESTAMP)
- `raw_payload_hash`
- `retrieved_at_utc`
- UNIQUE(station_id, model, runtime_utc)

### `mos_asof_feature`
Stores the T-1 “as-of” value that will be used as model input for predicting day T.
- `station_id`
- `target_date_local` (DATE)
- `asof_policy_id` (FK)
- `model`
- `asof_utc` (TIMESTAMP)
- `chosen_runtime_utc` (TIMESTAMP)
- `tmax_f` (DECIMAL(5,2), nullable)
- `missing_reason` (nullable string)
- `raw_payload_hash_ref` (nullable)
- `retrieved_at_utc`
- UNIQUE(station_id, target_date_local, asof_policy_id, model)

### `asof_policy`
- `id` (PK)
- `name` (e.g., DEFAULT_23_LOCAL)
- `asof_local_time` (TIME) (e.g., 23:00:00)
- `enabled` (BOOL)

### `ingest_checkpoint`
- `job_name`
- `station_id`
- `model` (nullable)
- `cursor_date` (DATE or TIMESTAMP)
- `cursor_runtime_utc` (TIMESTAMP nullable)
- `status` (RUNNING, COMPLETE, FAILED)
- `updated_at_utc`
- UNIQUE(job_name, station_id, model)

## Acceptance Criteria
- [ ] Flyway migrations create all tables + indexes + constraints.
- [ ] All entities/enums live in `models` module.
- [ ] Upsert strategy documented (MySQL `ON DUPLICATE KEY UPDATE`).
- [ ] A written “time semantics contract” is referenced from schema docs:
  - `target_date_local` is a date as used by CLI daily climate report.
  - all instants are stored as UTC timestamps.
- [ ] Schema supports idempotent ingest: uniqueness keys prevent duplicates.

## Source references
- Kalshi series provides settlement sources + contract terms: `GET /series/{ticker}` (Kalshi API docs).
- CLI day uses local standard time (DST => 1AM to 1AM local daylight time). See Kalshi weather markets help and IEM CLI table notes.

