# Parallel Day-Sharded Backfill Results

Status: `complete`
Date range: `2026-06-10` to `2026-07-08`
Elapsed seconds: `6637.310574`
Max workers: `2`

## Counts

- Source issues touched: `4334`
- Fetch ok: `4263`
- Fetch failed: `71`
- Normalize ok: `4263`
- Normalize failed: `0`
- Task errors: `0`
- Station features upserted: `260591`
- Area features upserted: `394457`
- Raw files deleted: `4263`
- Raw bytes deleted: `25627868094`

## Disk

- Max aggregate staging bytes: `241375243`
- Max single-worker staging bytes: `156263623`
- Max raw object bytes: `12391452`
- Minimum free disk bytes observed: `232063463424`

## By Source

- `gefs_control`: {'skipped_existing': 990, 'area_features_upserted': 189306, 'fetch_ok': 982, 'max_raw_object_bytes': 9629296, 'max_staging_bytes_after_fetch': 123582427, 'normalize_ok': 982, 'raw_bytes_deleted': 7950912912, 'raw_files_deleted': 982, 'source_issues_touched': 982, 'station_features_upserted': 16526}
- `gfs`: {'skipped_existing': 1091, 'area_features_upserted': 178685, 'fetch_ok': 875, 'max_raw_object_bytes': 12391452, 'max_staging_bytes_after_fetch': 156263623, 'normalize_ok': 875, 'raw_bytes_deleted': 10008525496, 'raw_files_deleted': 875, 'source_issues_touched': 881, 'station_features_upserted': 15495, 'fetch_failed': 6}
- `himawari9_b13_s0510`: {'fetch_failed': 65, 'skipped_existing': 1705, 'source_issues_touched': 2471, 'area_features_upserted': 26466, 'fetch_ok': 2406, 'max_raw_object_bytes': 3316709, 'max_staging_bytes_after_fetch': 25710344, 'normalize_ok': 2406, 'raw_bytes_deleted': 7668429686, 'raw_files_deleted': 2406, 'station_features_upserted': 228570}
