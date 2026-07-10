# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260704_20260704_20260709T173811Z_57496`
Execution mode: `optimized`
Elapsed seconds: `630.6718331000302`
Date range: `2026-07-04` to `2026-07-04`

## Counts

- Source issues touched: `280`
- Fetch ok: `277`
- Fetch failed: `3`
- Normalize ok: `None`
- Normalize failed: `None`
- Station features upserted: `None`
- Area features upserted: `None`
- Raw bytes deleted: `1722544493`
- Max staging bytes observed: `154877326`
- Final staging bytes: `0`
- Max raw object bytes observed: `11323220`
- Minimum free disk bytes observed: `237272350720`

## By Source

- `gefs_control`: {'max_raw_object_bytes': 8663878, 'max_staging_bytes_after_fetch': 106090685, 'source_issues_touched': 68, 'fetch_ok': 68, 'task_errors': 68, 'fetch_seconds_total': 1100.6388311001938, 'normalize_seconds_total': 426.7853597999783, 'db_write_seconds_total': 19.200672699837014, 'total_seconds_total': 1546.6248636000091, 'raw_bytes_deleted': 543285300, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 11323220, 'max_staging_bytes_after_fetch': 154877326, 'source_issues_touched': 68, 'fetch_ok': 67, 'task_errors': 67, 'fetch_seconds_total': 1578.3377924997476, 'normalize_seconds_total': 469.6198705002316, 'db_write_seconds_total': 19.23164780001389, 'total_seconds_total': 2067.189310799993, 'raw_bytes_deleted': 726881184, 'raw_files_deleted': 67, 'fetch_failed': 1}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3242803, 'max_staging_bytes_after_fetch': 25462599, 'source_issues_touched': 144, 'fetch_ok': 142, 'task_errors': 142, 'fetch_seconds_total': 645.7085611997172, 'normalize_seconds_total': 411.6889349997509, 'db_write_seconds_total': 26.625984499813057, 'total_seconds_total': 1084.0234806992812, 'raw_bytes_deleted': 452378009, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Phase Runtime

- `db_write`: {'count': 280, 'mean_seconds': 0.23235108928451415, 'p50_seconds': 0.051166300021577626, 'p90_seconds': 0.472693000047002, 'max_seconds': 4.850097699963953}
- `fetch`: {'count': 280, 'mean_seconds': 11.873875659998781, 'p50_seconds': 7.946137599996291, 'p90_seconds': 26.431456000020262, 'max_seconds': 41.19101230002707}
- `normalize`: {'count': 279, 'mean_seconds': 4.688509553046455, 'p50_seconds': 4.288849400007166, 'p90_seconds': 7.419921200023964, 'max_seconds': 13.486258400022052}
- `total`: {'count': 280, 'mean_seconds': 16.777991625354584, 'p50_seconds': 12.782307300018147, 'p90_seconds': 33.71805050002877, 'max_seconds': 48.96219280001242}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
