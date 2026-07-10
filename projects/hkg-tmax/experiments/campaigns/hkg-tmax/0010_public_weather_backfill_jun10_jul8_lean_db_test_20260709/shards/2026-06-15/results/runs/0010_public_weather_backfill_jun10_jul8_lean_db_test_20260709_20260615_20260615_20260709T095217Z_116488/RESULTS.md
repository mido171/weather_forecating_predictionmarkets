# Results

State: `complete_with_failures`
Run id: `0010_public_weather_backfill_jun10_jul8_lean_db_test_20260709_20260615_20260615_20260709T095217Z_116488`
Elapsed seconds: `3009.8391138999723`
Date range: `2026-06-15` to `2026-06-15`

## Counts

- Source issues touched: `400`
- Fetch ok: `397`
- Fetch failed: `3`
- Normalize ok: `397`
- Normalize failed: `None`
- Station features upserted: `17663`
- Area features upserted: `31059`
- Raw bytes deleted: `1786039424`
- Max staging bytes observed: `12499160`
- Max raw object bytes observed: `12499160`
- Minimum free disk bytes observed: `247945797632`

## By Source

- `envf_hkust_hko_radar`: {'manifest_frames': 120, 'source_issues_touched': 120, 'station_features_upserted': 1920, 'area_features_upserted': 2520, 'fetch_ok': 120, 'normalize_ok': 120}
- `gefs_control`: {'max_raw_object_bytes': 8902811, 'max_staging_bytes_after_fetch': 8902811, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1144, 'area_features_upserted': 13104, 'raw_bytes_deleted': 550249943, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 12499160, 'max_staging_bytes_after_fetch': 12499160, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1204, 'area_features_upserted': 13884, 'raw_bytes_deleted': 812942804, 'raw_files_deleted': 68}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3027972, 'max_staging_bytes_after_fetch': 3027972, 'source_issues_touched': 144, 'fetch_ok': 141, 'normalize_ok': 141, 'station_features_upserted': 13395, 'area_features_upserted': 1551, 'raw_bytes_deleted': 422846677, 'raw_files_deleted': 141, 'fetch_failed': 3}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
