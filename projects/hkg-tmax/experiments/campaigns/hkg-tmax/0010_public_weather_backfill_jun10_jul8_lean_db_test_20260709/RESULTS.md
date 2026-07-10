# Results

State: `complete`
Run id: `0010_public_weather_backfill_jun10_jul8_lean_db_test_20260709_20260610_20260610_20260709T090330Z_119104`
Elapsed seconds: `8.650711400026921`
Date range: `2026-06-10` to `2026-06-10`

## Counts

- Source issues touched: `1`
- Fetch ok: `1`
- Fetch failed: `None`
- Normalize ok: `1`
- Normalize failed: `None`
- Station features upserted: `13`
- Area features upserted: `143`
- Raw bytes deleted: `8419748`
- Max staging bytes observed: `8419748`
- Max raw object bytes observed: `8419748`
- Minimum free disk bytes observed: `254402822144`

## By Source

- `gfs`: {'max_raw_object_bytes': 8419748, 'max_staging_bytes_after_fetch': 8419748, 'source_issues_touched': 1, 'fetch_ok': 1, 'normalize_ok': 1, 'station_features_upserted': 13, 'area_features_upserted': 143, 'raw_bytes_deleted': 8419748, 'raw_files_deleted': 1}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
