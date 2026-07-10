# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260626_20260626_20260709T164939Z_119256`
Execution mode: `optimized`
Elapsed seconds: `655.08934950002`
Date range: `2026-06-26` to `2026-06-26`

## Counts

- Source issues touched: `280`
- Fetch ok: `278`
- Fetch failed: `2`
- Normalize ok: `None`
- Normalize failed: `None`
- Station features upserted: `None`
- Area features upserted: `None`
- Raw bytes deleted: `1831100604`
- Max staging bytes observed: `141666479`
- Final staging bytes: `0`
- Max raw object bytes observed: `12363708`
- Minimum free disk bytes observed: `237787324416`

## By Source

- `gefs_control`: {'max_raw_object_bytes': 8963119, 'max_staging_bytes_after_fetch': 100280118, 'source_issues_touched': 68, 'fetch_ok': 68, 'task_errors': 68, 'fetch_seconds_total': 1161.1356249998207, 'normalize_seconds_total': 418.55492949998006, 'db_write_seconds_total': 14.154138599871658, 'total_seconds_total': 1593.8446930996724, 'raw_bytes_deleted': 562861835, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 12363708, 'max_staging_bytes_after_fetch': 141666479, 'source_issues_touched': 68, 'fetch_ok': 68, 'task_errors': 68, 'fetch_seconds_total': 1822.9147382001393, 'normalize_seconds_total': 486.1958494002465, 'db_write_seconds_total': 27.593146399827674, 'total_seconds_total': 2336.7037340002134, 'raw_bytes_deleted': 807855095, 'raw_files_deleted': 68}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3278365, 'max_staging_bytes_after_fetch': 19644093, 'source_issues_touched': 144, 'fetch_ok': 142, 'task_errors': 142, 'fetch_seconds_total': 820.7049910998903, 'normalize_seconds_total': 379.26151280041086, 'db_write_seconds_total': 28.917539099988062, 'total_seconds_total': 1228.8840430002892, 'raw_bytes_deleted': 460383674, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Phase Runtime

- `db_write`: {'count': 280, 'mean_seconds': 0.2523743717845978, 'p50_seconds': 0.10060609999345616, 'p90_seconds': 0.407479899993632, 'max_seconds': 5.648284599999897}
- `fetch`: {'count': 280, 'mean_seconds': 13.588411979642322, 'p50_seconds': 9.966905099980067, 'p90_seconds': 28.632695200038143, 'max_seconds': 63.94347100000596}
- `normalize`: {'count': 280, 'mean_seconds': 4.585758184645134, 'p50_seconds': 4.216102200036403, 'p90_seconds': 7.36654419999104, 'max_seconds': 11.210395900008734}
- `total`: {'count': 280, 'mean_seconds': 18.426544536072054, 'p50_seconds': 14.754004700051155, 'p90_seconds': 36.467562199919485, 'max_seconds': 71.49524380004732}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
