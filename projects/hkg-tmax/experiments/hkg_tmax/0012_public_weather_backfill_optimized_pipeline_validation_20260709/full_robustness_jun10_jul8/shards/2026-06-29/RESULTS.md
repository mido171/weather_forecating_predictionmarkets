# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260629_20260629_20260709T170244Z_110772`
Execution mode: `optimized`
Elapsed seconds: `655.1527835000306`
Date range: `2026-06-29` to `2026-06-29`

## Counts

- Source issues touched: `280`
- Fetch ok: `278`
- Fetch failed: `2`
- Normalize ok: `None`
- Normalize failed: `None`
- Station features upserted: `None`
- Area features upserted: `None`
- Raw bytes deleted: `1824222892`
- Max staging bytes observed: `157479249`
- Final staging bytes: `0`
- Max raw object bytes observed: `12322485`
- Minimum free disk bytes observed: `237713387520`

## By Source

- `gefs_control`: {'max_raw_object_bytes': 9179911, 'max_staging_bytes_after_fetch': 115183035, 'source_issues_touched': 68, 'fetch_ok': 68, 'task_errors': 68, 'fetch_seconds_total': 1119.6814723999123, 'normalize_seconds_total': 425.1250640000799, 'db_write_seconds_total': 16.854476100299507, 'total_seconds_total': 1561.6610125002917, 'raw_bytes_deleted': 562042802, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 12322485, 'max_staging_bytes_after_fetch': 157479249, 'source_issues_touched': 68, 'fetch_ok': 68, 'task_errors': 68, 'fetch_seconds_total': 1796.4977740000468, 'normalize_seconds_total': 501.3465917998692, 'db_write_seconds_total': 25.00927819975186, 'total_seconds_total': 2322.853643999668, 'raw_bytes_deleted': 804595394, 'raw_files_deleted': 68}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3251288, 'max_staging_bytes_after_fetch': 25758885, 'source_issues_touched': 144, 'fetch_ok': 142, 'task_errors': 142, 'fetch_seconds_total': 563.3347396999598, 'normalize_seconds_total': 394.1818535006023, 'db_write_seconds_total': 26.676676599599887, 'total_seconds_total': 984.1932698001619, 'raw_bytes_deleted': 457584696, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Phase Runtime

- `db_write`: {'count': 280, 'mean_seconds': 0.2447872532130402, 'p50_seconds': 0.04766650003148243, 'p90_seconds': 0.4213282999699004, 'max_seconds': 7.715238499979023}
- `fetch`: {'count': 280, 'mean_seconds': 12.426835664642567, 'p50_seconds': 7.731927600048948, 'p90_seconds': 29.556211600021925, 'max_seconds': 74.52501580002718}
- `normalize`: {'count': 280, 'mean_seconds': 4.716619676073398, 'p50_seconds': 4.24793850001879, 'p90_seconds': 7.54680000001099, 'max_seconds': 14.30391330004204}
- `total`: {'count': 280, 'mean_seconds': 17.388242593929004, 'p50_seconds': 10.915052200027276, 'p90_seconds': 37.558292900037486, 'max_seconds': 81.54456150002079}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
