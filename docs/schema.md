# Schema (Epic #1)

This project keeps all JPA entities and shared DB models in the `models` module.
Flyway migrations live under `models/src/main/resources/db/migration` and are
loaded from the classpath by the ingestion service.

## Time semantics contract
The canonical time semantics for this schema are documented at:
`docs/time-semantics.md`.

Key contract points reflected in the schema:
- `target_date_local` is a station-local calendar date (CLI climate report day).
- All instants are stored as UTC timestamps (for example, `runtime_utc`,
  `asof_utc`, `retrieved_at_utc`).
- Local wall-clock values (`asof_local`) are stored alongside `station_zoneid`.

## Idempotent upserts
Ingestion jobs must upsert by the natural uniqueness keys defined in the schema,
using MySQL `INSERT ... ON DUPLICATE KEY UPDATE` to keep jobs restartable.

## Audit fields
All ingestion tables include `retrieved_at_utc` and `raw_payload_hash` (or
`raw_payload_hash_ref`) to support traceability and deduping. `cli_daily`
tracks `updated_at_utc` to capture revision updates from the CLI source.
