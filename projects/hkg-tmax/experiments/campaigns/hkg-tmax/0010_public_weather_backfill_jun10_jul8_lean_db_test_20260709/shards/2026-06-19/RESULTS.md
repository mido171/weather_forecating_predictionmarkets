# Results

State: `complete_with_failures`
Run id: `0010_public_weather_backfill_jun10_jul8_lean_db_test_20260709_20260619_20260619_20260709T104302Z_93348`
Elapsed seconds: `2919.5301352999522`
Date range: `2026-06-19` to `2026-06-19`

## Counts

- Source issues touched: `400`
- Fetch ok: `397`
- Fetch failed: `3`
- Normalize ok: `397`
- Normalize failed: `None`
- Station features upserted: `17663`
- Area features upserted: `31059`
- Raw bytes deleted: `1801954746`
- Max staging bytes observed: `12334902`
- Max raw object bytes observed: `12334902`
- Minimum free disk bytes observed: `250621681664`

## By Source

- `envf_hkust_hko_radar`: {'manifest_frames': 120, 'source_issues_touched': 120, 'station_features_upserted': 1920, 'area_features_upserted': 2520, 'fetch_ok': 120, 'normalize_ok': 120}
- `gefs_control`: {'max_raw_object_bytes': 8647303, 'max_staging_bytes_after_fetch': 8647303, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1144, 'area_features_upserted': 13104, 'raw_bytes_deleted': 559113291, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 12334902, 'max_staging_bytes_after_fetch': 12334902, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1204, 'area_features_upserted': 13884, 'raw_bytes_deleted': 798837597, 'raw_files_deleted': 68}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3187305, 'max_staging_bytes_after_fetch': 3187305, 'source_issues_touched': 144, 'fetch_ok': 141, 'normalize_ok': 141, 'station_features_upserted': 13395, 'area_features_upserted': 1551, 'raw_bytes_deleted': 444003858, 'raw_files_deleted': 141, 'fetch_failed': 3}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
