# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260701_20260701_20260709T171456Z_113992`
Execution mode: `optimized`
Elapsed seconds: `644.6438472999725`
Date range: `2026-07-01` to `2026-07-01`

## Counts

- Source issues touched: `280`
- Fetch ok: `276`
- Fetch failed: `4`
- Normalize ok: `None`
- Normalize failed: `None`
- Station features upserted: `None`
- Area features upserted: `None`
- Raw bytes deleted: `1777575588`
- Max staging bytes observed: `155676562`
- Final staging bytes: `0`
- Max raw object bytes observed: `12290484`
- Minimum free disk bytes observed: `237667188736`

## By Source

- `gefs_control`: {'max_raw_object_bytes': 8621034, 'max_staging_bytes_after_fetch': 106301578, 'source_issues_touched': 68, 'fetch_ok': 68, 'task_errors': 68, 'fetch_seconds_total': 1143.7150191999972, 'normalize_seconds_total': 438.1713879000745, 'db_write_seconds_total': 20.323134299949743, 'total_seconds_total': 1602.2095414000214, 'raw_bytes_deleted': 556187958, 'raw_files_deleted': 68}
- `gfs`: {'source_issues_touched': 68, 'fetch_failed': 2, 'fetch_seconds_total': 1802.1931882001227, 'db_write_seconds_total': 27.94596829992952, 'total_seconds_total': 2297.289988500357, 'max_raw_object_bytes': 12290484, 'max_staging_bytes_after_fetch': 155676562, 'fetch_ok': 66, 'task_errors': 66, 'normalize_seconds_total': 467.15083200030494, 'raw_bytes_deleted': 780043968, 'raw_files_deleted': 66}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3183176, 'max_staging_bytes_after_fetch': 24586664, 'source_issues_touched': 144, 'fetch_ok': 142, 'task_errors': 142, 'fetch_seconds_total': 662.4683115999214, 'normalize_seconds_total': 401.51392910041614, 'db_write_seconds_total': 27.13251160015352, 'total_seconds_total': 1091.114752300491, 'raw_bytes_deleted': 441343662, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Phase Runtime

- `db_write`: {'count': 280, 'mean_seconds': 0.26929147928583136, 'p50_seconds': 0.058678100001998246, 'p90_seconds': 0.4538609000155702, 'max_seconds': 5.766587999998592}
- `fetch`: {'count': 280, 'mean_seconds': 12.887058996428719, 'p50_seconds': 8.951401599973906, 'p90_seconds': 29.05895929998951, 'max_seconds': 46.2760760000092}
- `normalize`: {'count': 278, 'mean_seconds': 4.700849456837394, 'p50_seconds': 4.450858399970457, 'p90_seconds': 7.308504100015853, 'max_seconds': 11.401326399995014}
- `total`: {'count': 280, 'mean_seconds': 17.823622436431677, 'p50_seconds': 14.30179679999128, 'p90_seconds': 36.79225050006062, 'max_seconds': 54.29416970000602}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
