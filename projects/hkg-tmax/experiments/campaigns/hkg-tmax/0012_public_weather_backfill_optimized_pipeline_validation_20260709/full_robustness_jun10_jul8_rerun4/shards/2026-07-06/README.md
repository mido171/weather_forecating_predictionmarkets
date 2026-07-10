# 0012_public_weather_backfill_optimized_pipeline_validation_20260709

Lean DB-backed public weather backfill for HKG Tmax research.

This experiment streams public GFS, GEFS control, Himawari B13/S0510, and radar imagery into
`weather_backfill` Postgres tables while deleting raw payloads immediately after successful
normalization and DB commit.
