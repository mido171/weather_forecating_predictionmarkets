# Results

State: `complete_with_failures`
Run id: `0010_public_weather_backfill_jun10_jul8_lean_db_test_20260709_20260616_20260616_20260709T095235Z_111272`
Elapsed seconds: `3039.492309699999`
Date range: `2026-06-16` to `2026-06-16`

## Counts

- Source issues touched: `400`
- Fetch ok: `397`
- Fetch failed: `3`
- Normalize ok: `397`
- Normalize failed: `None`
- Station features upserted: `17740`
- Area features upserted: `30862`
- Raw bytes deleted: `1815228330`
- Max staging bytes observed: `12446099`
- Max raw object bytes observed: `12446099`
- Minimum free disk bytes observed: `247945416704`

## By Source

- `envf_hkust_hko_radar`: {'manifest_frames': 120, 'source_issues_touched': 120, 'station_features_upserted': 1920, 'area_features_upserted': 2520, 'fetch_ok': 120, 'normalize_ok': 120}
- `gefs_control`: {'max_raw_object_bytes': 9250098, 'max_staging_bytes_after_fetch': 9250098, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1144, 'area_features_upserted': 13104, 'raw_bytes_deleted': 573272357, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 12446099, 'max_staging_bytes_after_fetch': 12446099, 'source_issues_touched': 68, 'fetch_ok': 67, 'normalize_ok': 67, 'station_features_upserted': 1186, 'area_features_upserted': 13676, 'raw_bytes_deleted': 800207491, 'raw_files_deleted': 67, 'fetch_failed': 1}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3181716, 'max_staging_bytes_after_fetch': 3181716, 'source_issues_touched': 144, 'fetch_ok': 142, 'normalize_ok': 142, 'station_features_upserted': 13490, 'area_features_upserted': 1562, 'raw_bytes_deleted': 441748482, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
