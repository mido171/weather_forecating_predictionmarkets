---
name: ingest-source
description: Add or backfill a weather or Polymarket source with immutable raw storage, provenance, as-of semantics, schema tests, and monitoring. Use whenever adding data.
---

1. Add a complete source entry to `config/sources/data_sources.yaml`.
2. Verify the endpoint from official provider material; do not guess it.
3. Record terms/license, cadence, coverage, update latency, revision behavior, and point-in-time role.
4. Store raw bytes first with retrieval metadata and SHA-256.
5. Parse into versioned bronze data without modifying raw files.
6. Assign all applicable timestamps:
   `valid_at`, `issued_at`, `published_at`, `available_at`, `retrieved_at`.
7. Add schema, unit, range, duplicate, freshness, and missingness checks.
8. Backfill only with a documented query manifest and rate limits.
9. Start live archival if historical vintages are unavailable.
10. Produce a compact coverage report and a source contract under `docs/specifications/source-contracts/`; keep raw/derived payloads under `HKG_TMAX_DATA_ROOT`.
