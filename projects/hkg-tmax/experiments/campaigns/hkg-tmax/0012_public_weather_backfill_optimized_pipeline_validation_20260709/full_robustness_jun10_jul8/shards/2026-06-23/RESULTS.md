# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260623_20260623_20260709T163627Z_119736`
Execution mode: `optimized`
Elapsed seconds: `266.90050410002004`
Date range: `2026-06-23` to `2026-06-23`

## Counts

- Source issues touched: `165`
- Fetch ok: `163`
- Fetch failed: `2`
- Normalize ok: `None`
- Normalize failed: `None`
- Station features upserted: `None`
- Area features upserted: `None`
- Raw bytes deleted: `615849097`
- Max staging bytes observed: `120998895`
- Final staging bytes: `0`
- Max raw object bytes observed: `8564365`
- Minimum free disk bytes observed: `238010671104`

## By Source

- `gefs_control`: {'skipped_existing': 47, 'max_raw_object_bytes': 8564365, 'max_staging_bytes_after_fetch': 120998895, 'source_issues_touched': 21, 'fetch_ok': 21, 'task_errors': 21, 'fetch_seconds_total': 400.4767082000035, 'normalize_seconds_total': 162.6619045000407, 'db_write_seconds_total': 4.353877100045793, 'total_seconds_total': 567.49248980009, 'raw_bytes_deleted': 170306763, 'raw_files_deleted': 21}
- `gfs`: {'skipped_existing': 68}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3208092, 'max_staging_bytes_after_fetch': 22141364, 'source_issues_touched': 144, 'fetch_ok': 142, 'task_errors': 142, 'fetch_seconds_total': 700.8503982999828, 'normalize_seconds_total': 425.83473170001525, 'db_write_seconds_total': 30.51986130012665, 'total_seconds_total': 1157.2049913001247, 'raw_bytes_deleted': 445542334, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Phase Runtime

- `db_write`: {'count': 165, 'mean_seconds': 0.2113559903040754, 'p50_seconds': 0.20130439999047667, 'p90_seconds': 0.3537769999820739, 'max_seconds': 3.5034755999804474}
- `fetch`: {'count': 165, 'mean_seconds': 6.674709736363553, 'p50_seconds': 4.417041999986395, 'p90_seconds': 15.7375219000387, 'max_seconds': 29.18947089998983}
- `normalize`: {'count': 165, 'mean_seconds': 3.566646280000339, 'p50_seconds': 2.9463562999735586, 'p90_seconds': 6.270516499993391, 'max_seconds': 13.722067599999718}
- `total`: {'count': 165, 'mean_seconds': 10.452712006667968, 'p50_seconds': 7.63695800001733, 'p90_seconds': 22.99244360002922, 'max_seconds': 37.36064039997291}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
