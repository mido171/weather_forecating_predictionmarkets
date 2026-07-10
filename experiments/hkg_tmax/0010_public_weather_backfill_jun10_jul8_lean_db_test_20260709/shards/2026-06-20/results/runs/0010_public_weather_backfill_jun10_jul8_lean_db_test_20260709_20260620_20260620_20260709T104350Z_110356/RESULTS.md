# Results

State: `complete_with_failures`
Run id: `0010_public_weather_backfill_jun10_jul8_lean_db_test_20260709_20260620_20260620_20260709T104350Z_110356`
Elapsed seconds: `2936.9456102999975`
Date range: `2026-06-20` to `2026-06-20`

## Counts

- Source issues touched: `400`
- Fetch ok: `398`
- Fetch failed: `2`
- Normalize ok: `398`
- Normalize failed: `None`
- Station features upserted: `17758`
- Area features upserted: `31070`
- Raw bytes deleted: `1821807900`
- Max staging bytes observed: `12401525`
- Max raw object bytes observed: `12401525`
- Minimum free disk bytes observed: `250616360960`

## By Source

- `envf_hkust_hko_radar`: {'manifest_frames': 120, 'source_issues_touched': 120, 'station_features_upserted': 1920, 'area_features_upserted': 2520, 'fetch_ok': 120, 'normalize_ok': 120}
- `gefs_control`: {'max_raw_object_bytes': 9258668, 'max_staging_bytes_after_fetch': 9258668, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1144, 'area_features_upserted': 13104, 'raw_bytes_deleted': 562565320, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 12401525, 'max_staging_bytes_after_fetch': 12401525, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1204, 'area_features_upserted': 13884, 'raw_bytes_deleted': 803487679, 'raw_files_deleted': 68}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3246186, 'max_staging_bytes_after_fetch': 3246186, 'source_issues_touched': 144, 'fetch_ok': 142, 'normalize_ok': 142, 'station_features_upserted': 13490, 'area_features_upserted': 1562, 'raw_bytes_deleted': 455754901, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
