# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260622_20260622_20260709T163627Z_117840`
Execution mode: `optimized`
Elapsed seconds: `273.42289039999014`
Date range: `2026-06-22` to `2026-06-22`

## Counts

- Source issues touched: `166`
- Fetch ok: `163`
- Fetch failed: `3`
- Normalize ok: `None`
- Normalize failed: `None`
- Station features upserted: `None`
- Area features upserted: `None`
- Raw bytes deleted: `631608470`
- Max staging bytes observed: `121471396`
- Final staging bytes: `0`
- Max raw object bytes observed: `8509302`
- Minimum free disk bytes observed: `238018740224`

## By Source

- `gefs_control`: {'skipped_existing': 46, 'max_raw_object_bytes': 8509302, 'max_staging_bytes_after_fetch': 121471396, 'source_issues_touched': 22, 'fetch_ok': 22, 'task_errors': 22, 'fetch_seconds_total': 445.72302319988376, 'normalize_seconds_total': 165.2244910999434, 'db_write_seconds_total': 3.2828618002240546, 'total_seconds_total': 614.2303761000512, 'raw_bytes_deleted': 178685088, 'raw_files_deleted': 22}
- `gfs`: {'skipped_existing': 68}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3243971, 'max_staging_bytes_after_fetch': 25318006, 'source_issues_touched': 144, 'fetch_ok': 141, 'task_errors': 141, 'fetch_seconds_total': 738.7217567000771, 'normalize_seconds_total': 424.78928099997574, 'db_write_seconds_total': 30.402666099951603, 'total_seconds_total': 1193.9137038000044, 'raw_bytes_deleted': 452923382, 'raw_files_deleted': 141, 'fetch_failed': 3}

## Phase Runtime

- `db_write`: {'count': 166, 'mean_seconds': 0.20292486686852806, 'p50_seconds': 0.2014113999903202, 'p90_seconds': 0.36808250000467524, 'max_seconds': 2.7619273000163957}
- `fetch`: {'count': 166, 'mean_seconds': 7.135209517469644, 'p50_seconds': 4.620213599991985, 'p90_seconds': 18.329501900007017, 'max_seconds': 31.60004220000701}
- `normalize`: {'count': 166, 'mean_seconds': 3.554299831927224, 'p50_seconds': 2.8826256000320427, 'p90_seconds': 6.329474699974526, 'max_seconds': 13.344597799994517}
- `total`: {'count': 166, 'mean_seconds': 10.892434216265395, 'p50_seconds': 8.028645000013057, 'p90_seconds': 25.30219640006544, 'max_seconds': 35.911839200067334}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
