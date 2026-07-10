# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260630_20260630_20260709T171355Z_112464`
Execution mode: `optimized`
Elapsed seconds: `653.3671093999874`
Date range: `2026-06-30` to `2026-06-30`

## Counts

- Source issues touched: `280`
- Fetch ok: `277`
- Fetch failed: `3`
- Normalize ok: `None`
- Normalize failed: `None`
- Station features upserted: `None`
- Area features upserted: `None`
- Raw bytes deleted: `1798108088`
- Max staging bytes observed: `152157652`
- Final staging bytes: `0`
- Max raw object bytes observed: `12281591`
- Minimum free disk bytes observed: `237677879296`

## By Source

- `gefs_control`: {'max_raw_object_bytes': 8710525, 'max_staging_bytes_after_fetch': 96638134, 'source_issues_touched': 68, 'fetch_ok': 68, 'task_errors': 68, 'fetch_seconds_total': 1238.38419349998, 'normalize_seconds_total': 436.78416699997615, 'db_write_seconds_total': 14.679374100174755, 'total_seconds_total': 1689.847734600131, 'raw_bytes_deleted': 553428219, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 12281591, 'max_staging_bytes_after_fetch': 152157652, 'source_issues_touched': 68, 'fetch_ok': 67, 'task_errors': 67, 'fetch_seconds_total': 1627.1622306997888, 'normalize_seconds_total': 467.2721821003943, 'db_write_seconds_total': 23.795573600102216, 'total_seconds_total': 2118.2299864002853, 'raw_bytes_deleted': 787809849, 'raw_files_deleted': 67, 'fetch_failed': 1}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3269976, 'max_staging_bytes_after_fetch': 26105618, 'source_issues_touched': 144, 'fetch_ok': 142, 'task_errors': 142, 'fetch_seconds_total': 792.1377031999291, 'normalize_seconds_total': 404.98753140028566, 'db_write_seconds_total': 26.87784960027784, 'total_seconds_total': 1224.0030842004926, 'raw_bytes_deleted': 456870020, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Phase Runtime

- `db_write`: {'count': 280, 'mean_seconds': 0.23340284750198148, 'p50_seconds': 0.0933358000474982, 'p90_seconds': 0.39687959995353594, 'max_seconds': 4.760395599994808}
- `fetch`: {'count': 280, 'mean_seconds': 13.063157597856064, 'p50_seconds': 11.318520599976182, 'p90_seconds': 28.30193340004189, 'max_seconds': 37.649727099982556}
- `normalize`: {'count': 279, 'mean_seconds': 4.691913550181563, 'p50_seconds': 4.437010699999519, 'p90_seconds': 7.570698499970604, 'max_seconds': 12.311272700026166}
- `total`: {'count': 280, 'mean_seconds': 17.971717161431815, 'p50_seconds': 16.09637990006013, 'p90_seconds': 35.16634839994367, 'max_seconds': 45.231883400003426}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
