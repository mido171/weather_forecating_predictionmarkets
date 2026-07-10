# Results

State: `complete_with_failures`
Run id: `0010_public_weather_backfill_jun10_jul8_lean_db_test_20260709_20260613_20260613_20260709T090402Z_37836`
Elapsed seconds: `2880.269859699998`
Date range: `2026-06-13` to `2026-06-13`

## Counts

- Source issues touched: `400`
- Fetch ok: `398`
- Fetch failed: `2`
- Normalize ok: `398`
- Normalize failed: `None`
- Station features upserted: `17758`
- Area features upserted: `31070`
- Raw bytes deleted: `1753867582`
- Max staging bytes observed: `16219694`
- Max raw object bytes observed: `12240221`
- Minimum free disk bytes observed: `254279696384`

## By Source

- `envf_hkust_hko_radar`: {'manifest_frames': 120, 'source_issues_touched': 120, 'station_features_upserted': 1920, 'area_features_upserted': 2520, 'fetch_ok': 120, 'normalize_ok': 120}
- `gefs_control`: {'max_raw_object_bytes': 8069620, 'max_staging_bytes_after_fetch': 8069620, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1144, 'area_features_upserted': 13104, 'raw_bytes_deleted': 520607608, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 12240221, 'max_staging_bytes_after_fetch': 16219694, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1204, 'area_features_upserted': 13884, 'raw_bytes_deleted': 802755415, 'raw_files_deleted': 68}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3118080, 'max_staging_bytes_after_fetch': 3118080, 'source_issues_touched': 144, 'fetch_ok': 142, 'normalize_ok': 142, 'station_features_upserted': 13490, 'area_features_upserted': 1562, 'raw_bytes_deleted': 430504559, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
