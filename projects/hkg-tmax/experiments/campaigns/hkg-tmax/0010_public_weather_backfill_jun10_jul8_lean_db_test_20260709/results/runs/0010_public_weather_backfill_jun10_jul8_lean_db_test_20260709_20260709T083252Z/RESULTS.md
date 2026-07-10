# Results

State: `complete`
Run id: `0010_public_weather_backfill_jun10_jul8_lean_db_test_20260709_20260709T083252Z`
Elapsed seconds: `21.555281099979766`
Date range: `2026-06-10` to `2026-06-10`

## Counts

- Source issues touched: `1`
- Fetch ok: `1`
- Fetch failed: `None`
- Normalize ok: `1`
- Normalize failed: `None`
- Station features upserted: `14`
- Area features upserted: `156`
- Raw bytes deleted: `6698962`
- Max staging bytes observed: `6698962`
- Max raw object bytes observed: `6698962`
- Minimum free disk bytes observed: `254473842688`

## By Source

- `gefs_control`: {'max_raw_object_bytes': 6698962, 'max_staging_bytes_after_fetch': 6698962, 'source_issues_touched': 1, 'fetch_ok': 1, 'normalize_ok': 1, 'station_features_upserted': 14, 'area_features_upserted': 156, 'raw_bytes_deleted': 6698962, 'raw_files_deleted': 1}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
