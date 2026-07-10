# Results

State: `complete`
Run id: `0010_public_weather_backfill_jun10_jul8_lean_db_test_20260709_20260709T083057Z`
Elapsed seconds: `5.532315100019332`
Date range: `2026-07-08` to `2026-07-08`

## Counts

- Source issues touched: `1`
- Fetch ok: `1`
- Fetch failed: `None`
- Normalize ok: `1`
- Normalize failed: `None`
- Station features upserted: `95`
- Area features upserted: `11`
- Raw bytes deleted: `3196312`
- Max staging bytes observed: `3196312`
- Max raw object bytes observed: `3196312`
- Minimum free disk bytes observed: `254466752512`

## By Source

- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3196312, 'max_staging_bytes_after_fetch': 3196312, 'source_issues_touched': 1, 'fetch_ok': 1, 'normalize_ok': 1, 'station_features_upserted': 95, 'area_features_upserted': 11, 'raw_bytes_deleted': 3196312, 'raw_files_deleted': 1}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
