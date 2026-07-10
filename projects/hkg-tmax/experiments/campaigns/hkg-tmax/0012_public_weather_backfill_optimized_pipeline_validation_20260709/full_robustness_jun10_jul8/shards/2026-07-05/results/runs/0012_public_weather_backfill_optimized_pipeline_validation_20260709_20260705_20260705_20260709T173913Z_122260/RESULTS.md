# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260705_20260705_20260709T173913Z_122260`
Execution mode: `optimized`
Elapsed seconds: `631.1364994999603`
Date range: `2026-07-05` to `2026-07-05`

## Counts

- Source issues touched: `280`
- Fetch ok: `277`
- Fetch failed: `3`
- Normalize ok: `None`
- Normalize failed: `None`
- Station features upserted: `None`
- Area features upserted: `None`
- Raw bytes deleted: `1692765349`
- Max staging bytes observed: `133135297`
- Final staging bytes: `0`
- Max raw object bytes observed: `11343997`
- Minimum free disk bytes observed: `237275967488`

## By Source

- `gefs_control`: {'max_raw_object_bytes': 8528393, 'max_staging_bytes_after_fetch': 100738547, 'source_issues_touched': 68, 'fetch_ok': 68, 'task_errors': 68, 'fetch_seconds_total': 1046.861483699875, 'normalize_seconds_total': 441.94300980027765, 'db_write_seconds_total': 18.410309199884068, 'total_seconds_total': 1507.2148027000367, 'raw_bytes_deleted': 518959489, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 11343997, 'max_staging_bytes_after_fetch': 133135297, 'source_issues_touched': 68, 'fetch_ok': 67, 'task_errors': 67, 'fetch_seconds_total': 1505.502287699841, 'normalize_seconds_total': 478.4087047003559, 'db_write_seconds_total': 21.92351580003742, 'total_seconds_total': 2005.8345082002343, 'raw_bytes_deleted': 727166384, 'raw_files_deleted': 67, 'fetch_failed': 1}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3230592, 'max_staging_bytes_after_fetch': 22162981, 'source_issues_touched': 144, 'fetch_ok': 142, 'task_errors': 142, 'fetch_seconds_total': 580.1830769006046, 'normalize_seconds_total': 382.35767540015513, 'db_write_seconds_total': 25.639090299839154, 'total_seconds_total': 988.1798426005989, 'raw_bytes_deleted': 446639476, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Phase Runtime

- `db_write`: {'count': 280, 'mean_seconds': 0.2356175546420023, 'p50_seconds': 0.055264399969018996, 'p90_seconds': 0.5089016000274569, 'max_seconds': 4.997922400012612}
- `fetch`: {'count': 280, 'mean_seconds': 11.187667315358288, 'p50_seconds': 8.005211900046561, 'p90_seconds': 24.237838500004727, 'max_seconds': 39.09837710001739}
- `normalize`: {'count': 279, 'mean_seconds': 4.669209282798525, 'p50_seconds': 3.9920611000270583, 'p90_seconds': 7.622276300040539, 'max_seconds': 12.78228109999327}
- `total`: {'count': 280, 'mean_seconds': 16.07581840536025, 'p50_seconds': 12.724869100085925, 'p90_seconds': 32.1222993999836, 'max_seconds': 47.22355240001343}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
