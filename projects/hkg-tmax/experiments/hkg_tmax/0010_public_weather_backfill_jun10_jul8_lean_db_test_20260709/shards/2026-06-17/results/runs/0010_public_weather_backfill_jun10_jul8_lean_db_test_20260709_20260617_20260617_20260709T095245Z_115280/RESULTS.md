# Results

State: `complete_with_failures`
Run id: `0010_public_weather_backfill_jun10_jul8_lean_db_test_20260709_20260617_20260617_20260709T095245Z_115280`
Elapsed seconds: `3060.6134797999985`
Date range: `2026-06-17` to `2026-06-17`

## Counts

- Source issues touched: `400`
- Fetch ok: `398`
- Fetch failed: `2`
- Normalize ok: `398`
- Normalize failed: `None`
- Station features upserted: `17758`
- Area features upserted: `31070`
- Raw bytes deleted: `1821368417`
- Max staging bytes observed: `12288255`
- Max raw object bytes observed: `12288255`
- Minimum free disk bytes observed: `247937384448`

## By Source

- `envf_hkust_hko_radar`: {'manifest_frames': 120, 'source_issues_touched': 120, 'station_features_upserted': 1920, 'area_features_upserted': 2520, 'fetch_ok': 120, 'normalize_ok': 120}
- `gefs_control`: {'max_raw_object_bytes': 9231365, 'max_staging_bytes_after_fetch': 9231365, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1144, 'area_features_upserted': 13104, 'raw_bytes_deleted': 568745178, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 12288255, 'max_staging_bytes_after_fetch': 12288255, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1204, 'area_features_upserted': 13884, 'raw_bytes_deleted': 804487488, 'raw_files_deleted': 68}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3230354, 'max_staging_bytes_after_fetch': 3230354, 'source_issues_touched': 144, 'fetch_ok': 142, 'normalize_ok': 142, 'station_features_upserted': 13490, 'area_features_upserted': 1562, 'raw_bytes_deleted': 448135751, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
