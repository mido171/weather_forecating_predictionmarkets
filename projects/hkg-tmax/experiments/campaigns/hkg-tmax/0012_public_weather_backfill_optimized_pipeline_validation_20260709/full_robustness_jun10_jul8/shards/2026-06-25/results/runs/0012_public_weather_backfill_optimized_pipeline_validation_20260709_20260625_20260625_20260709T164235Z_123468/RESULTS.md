# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260625_20260625_20260709T164235Z_123468`
Execution mode: `optimized`
Elapsed seconds: `296.503065900004`
Date range: `2026-06-25` to `2026-06-25`

## Counts

- Source issues touched: `172`
- Fetch ok: `170`
- Fetch failed: `2`
- Normalize ok: `None`
- Normalize failed: `None`
- Station features upserted: `None`
- Area features upserted: `None`
- Raw bytes deleted: `688858321`
- Max staging bytes observed: `123582427`
- Final staging bytes: `0`
- Max raw object bytes observed: `9064031`
- Minimum free disk bytes observed: `238046527488`

## By Source

- `gefs_control`: {'skipped_existing': 40, 'max_raw_object_bytes': 9064031, 'max_staging_bytes_after_fetch': 123582427, 'source_issues_touched': 28, 'fetch_ok': 28, 'task_errors': 28, 'fetch_seconds_total': 588.718975200085, 'normalize_seconds_total': 191.45087870000862, 'db_write_seconds_total': 1.5769632001174614, 'total_seconds_total': 781.746817100211, 'raw_bytes_deleted': 231476032, 'raw_files_deleted': 28}
- `gfs`: {'skipped_existing': 68}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3246142, 'max_staging_bytes_after_fetch': 22410923, 'source_issues_touched': 144, 'fetch_ok': 142, 'task_errors': 142, 'fetch_seconds_total': 813.6412579997559, 'normalize_seconds_total': 384.22067440033425, 'db_write_seconds_total': 30.034988100465853, 'total_seconds_total': 1227.896920500556, 'raw_bytes_deleted': 457382289, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Phase Runtime

- `db_write`: {'count': 172, 'mean_seconds': 0.18379041453827508, 'p50_seconds': 0.19720090000191703, 'p90_seconds': 0.3230725000030361, 'max_seconds': 0.6448019000235945}
- `fetch`: {'count': 172, 'mean_seconds': 8.153257169766517, 'p50_seconds': 5.464867000002414, 'p90_seconds': 16.10460849996889, 'max_seconds': 47.263958800002}
- `normalize`: {'count': 172, 'mean_seconds': 3.346927634304319, 'p50_seconds': 2.6764660999760963, 'p90_seconds': 5.881480699987151, 'max_seconds': 18.108842900022864}
- `total`: {'count': 172, 'mean_seconds': 11.683975218609111, 'p50_seconds': 8.387964599998668, 'p90_seconds': 23.76882029999979, 'max_seconds': 53.141712000011466}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
