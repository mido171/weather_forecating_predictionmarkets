# Parallel Day-Sharded Backfill Results

Status: `complete`
Date range: `2026-06-15` to `2026-06-21`
Elapsed seconds: `2894.153261`
Max workers: `2`

## Counts

- Source issues touched: `1960`
- Fetch ok: `1940`
- Fetch failed: `20`
- Normalize ok: `1940`
- Normalize failed: `0`
- Task errors: `0`
- Station features upserted: `110527`
- Area features upserted: `199193`
- Raw files deleted: `1940`
- Raw bytes deleted: `12657237821`

## Disk

- Max aggregate staging bytes: `0`
- Max single-worker staging bytes: `169313139`
- Max raw object bytes: `12499160`
- Minimum free disk bytes observed: `238059347968`

## By Source

- `gefs_control`: {'area_features_upserted': 91728, 'fetch_ok': 476, 'max_raw_object_bytes': 9258668, 'max_staging_bytes_after_fetch': 114174925, 'normalize_ok': 476, 'raw_bytes_deleted': 3934626155, 'raw_files_deleted': 476, 'source_issues_touched': 476, 'station_features_upserted': 8008}
- `gfs`: {'area_features_upserted': 96564, 'fetch_ok': 473, 'max_raw_object_bytes': 12499160, 'max_staging_bytes_after_fetch': 169313139, 'normalize_ok': 473, 'raw_bytes_deleted': 5603823064, 'raw_files_deleted': 473, 'source_issues_touched': 476, 'station_features_upserted': 8374, 'fetch_failed': 3}
- `himawari9_b13_s0510`: {'area_features_upserted': 10901, 'fetch_failed': 17, 'fetch_ok': 991, 'max_raw_object_bytes': 3246186, 'max_staging_bytes_after_fetch': 25947045, 'normalize_ok': 991, 'raw_bytes_deleted': 3118788602, 'raw_files_deleted': 991, 'source_issues_touched': 1008, 'station_features_upserted': 94145}
