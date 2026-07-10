# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260707_20260707_20260709T175126Z_67432`
Execution mode: `optimized`
Elapsed seconds: `609.6337385000079`
Date range: `2026-07-07` to `2026-07-07`

## Counts

- Source issues touched: `273`
- Fetch ok: `270`
- Fetch failed: `3`
- Normalize ok: `None`
- Normalize failed: `None`
- Station features upserted: `None`
- Area features upserted: `None`
- Raw bytes deleted: `1682324045`
- Max staging bytes observed: `152516737`
- Final staging bytes: `0`
- Max raw object bytes observed: `12036815`
- Minimum free disk bytes observed: `237145522176`

## By Source

- `gefs_control`: {'skipped_existing': 1, 'max_raw_object_bytes': 8879192, 'max_staging_bytes_after_fetch': 101282248, 'source_issues_touched': 67, 'fetch_ok': 67, 'task_errors': 67, 'fetch_seconds_total': 1141.7055175999412, 'normalize_seconds_total': 401.34907689969987, 'db_write_seconds_total': 8.820605900022201, 'total_seconds_total': 1551.8752003996633, 'raw_bytes_deleted': 522459057, 'raw_files_deleted': 67}
- `gfs`: {'skipped_existing': 3, 'max_raw_object_bytes': 12036815, 'max_staging_bytes_after_fetch': 152516737, 'source_issues_touched': 65, 'fetch_ok': 65, 'task_errors': 65, 'fetch_seconds_total': 1558.3937118998147, 'normalize_seconds_total': 470.17409539961955, 'db_write_seconds_total': 30.076371100149117, 'total_seconds_total': 2058.6441783995833, 'raw_bytes_deleted': 723927360, 'raw_files_deleted': 65}
- `himawari9_b13_s0510`: {'skipped_existing': 3, 'max_raw_object_bytes': 3195603, 'max_staging_bytes_after_fetch': 22225601, 'source_issues_touched': 141, 'fetch_ok': 138, 'task_errors': 138, 'fetch_seconds_total': 618.0475427996716, 'normalize_seconds_total': 387.3026039999677, 'db_write_seconds_total': 29.37222820025636, 'total_seconds_total': 1034.7223749998957, 'raw_bytes_deleted': 435937628, 'raw_files_deleted': 138, 'fetch_failed': 3}

## Phase Runtime

- `db_write`: {'count': 273, 'mean_seconds': 0.2500703487195153, 'p50_seconds': 0.09808979998342693, 'p90_seconds': 0.3981802999624051, 'max_seconds': 8.131674400006887}
- `fetch`: {'count': 273, 'mean_seconds': 12.154383781316584, 'p50_seconds': 8.644541899964679, 'p90_seconds': 26.613498000020627, 'max_seconds': 45.552937000000384}
- `normalize`: {'count': 273, 'mean_seconds': 4.611083429667718, 'p50_seconds': 4.353023000003304, 'p90_seconds': 7.235842099995352, 'max_seconds': 14.863513499964029}
- `total`: {'count': 273, 'mean_seconds': 17.01553755970382, 'p50_seconds': 13.358173200045712, 'p90_seconds': 33.8155747000128, 'max_seconds': 49.961560200026724}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
