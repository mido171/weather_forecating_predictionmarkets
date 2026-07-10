# Parallel Day-Sharded Backfill Results

Status: `complete`
Date range: `2026-06-10` to `2026-07-08`
Elapsed seconds: `6493.276066`
Max workers: `2`

## Counts

- Source issues touched: `4334`
- Fetch ok: `4264`
- Fetch failed: `70`
- Normalize ok: `0`
- Normalize failed: `0`
- Task errors: `4264`
- Station features upserted: `0`
- Area features upserted: `0`
- Raw files deleted: `4264`
- Raw bytes deleted: `25641727433`

## Disk

- Max aggregate staging bytes: `242470291`
- Max single-worker staging bytes: `157479249`
- Max raw object bytes: `12391452`
- Minimum free disk bytes observed: `237188857856`

## By Source

- `gefs_control`: {'skipped_existing': 990, 'fetch_ok': 982, 'max_raw_object_bytes': 9629296, 'max_staging_bytes_after_fetch': 123582427, 'raw_bytes_deleted': 7950912912, 'raw_files_deleted': 982, 'source_issues_touched': 982, 'task_errors': 982}
- `gfs`: {'skipped_existing': 1091, 'fetch_ok': 876, 'max_raw_object_bytes': 12391452, 'max_staging_bytes_after_fetch': 157479249, 'raw_bytes_deleted': 10022384835, 'raw_files_deleted': 876, 'source_issues_touched': 881, 'task_errors': 876, 'fetch_failed': 5}
- `himawari9_b13_s0510`: {'fetch_failed': 65, 'skipped_existing': 1705, 'source_issues_touched': 2471, 'fetch_ok': 2406, 'max_raw_object_bytes': 3316709, 'max_staging_bytes_after_fetch': 26105618, 'raw_bytes_deleted': 7668429686, 'raw_files_deleted': 2406, 'task_errors': 2406}
