# 0012 Public Weather Backfill Optimized Pipeline Validation

This experiment validates whether the optimized GFS, GEFS control, and Himawari B13/S0510
pipeline is production-worthy for the larger historical backfill.

The target is not just faster downloads. The accepted result must prove the full
Postgres-backed path: fetch, normalize, persist source issues and features, preserve leakage
timestamps, delete transient raw payloads, and report repeatable runtime and resource metrics.

Final status: `accepted_with_notes`. The model side is fully complete for the 2026-06-10
through 2026-07-08 validation window; the only remaining missing issues are recorded Himawari
source-side 404s. Significance score: `89/100`.

## Detailed Documentation

The complete engineering and capacity handoff is under [documentation/](documentation/README.md):

- [Implementation and validation deep dive](documentation/PUBLIC_WEATHER_BACKFILL_IMPLEMENTATION_AND_VALIDATION.md)
- [Live Postgres measurement snapshot](documentation/LIVE_POSTGRES_MEASUREMENT_SNAPSHOT_20260710.md)
- [2017-2026 Postgres storage estimate](documentation/POSTGRES_STORAGE_CAPACITY_ESTIMATE_2017_TO_2026.md)

The current-schema retained storage estimate for GFS, GEFS control, and Himawari from
`2017-01-01` through `2026-07-10` is `121.4 GB` decimal (`113.0 GiB`).
