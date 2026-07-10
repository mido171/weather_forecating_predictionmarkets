# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260707_20260707_20260709T194653Z_96836`
Execution mode: `optimized`
Elapsed seconds: `637.9268644999829`
Date range: `2026-07-07` to `2026-07-07`

## Counts

- Source issues touched: `273`
- Fetch ok: `269`
- Fetch failed: `4`
- Normalize ok: `269`
- Normalize failed: `None`
- Station features upserted: `15369`
- Area features upserted: `27479`
- Raw bytes deleted: `1670366454`
- Max staging bytes observed: `156263623`
- Final staging bytes: `0`
- Max raw object bytes observed: `12036815`
- Minimum free disk bytes observed: `232355164160`

## By Source

- `gefs_control`: {'skipped_existing': 1, 'max_raw_object_bytes': 8879192, 'max_staging_bytes_after_fetch': 105056213, 'source_issues_touched': 67, 'fetch_ok': 67, 'normalize_ok': 67, 'station_features_upserted': 1127, 'area_features_upserted': 12909, 'fetch_seconds_total': 1072.4122555999202, 'normalize_seconds_total': 427.7650428999332, 'db_write_seconds_total': 32.691017199715134, 'total_seconds_total': 1532.8683156995685, 'raw_bytes_deleted': 522459057, 'raw_files_deleted': 67}
- `gfs`: {'skipped_existing': 3, 'max_raw_object_bytes': 12036815, 'max_staging_bytes_after_fetch': 156263623, 'source_issues_touched': 65, 'fetch_ok': 64, 'normalize_ok': 64, 'station_features_upserted': 1132, 'area_features_upserted': 13052, 'fetch_seconds_total': 1592.7772852997878, 'normalize_seconds_total': 489.52866080001695, 'db_write_seconds_total': 56.385355999926105, 'total_seconds_total': 2138.691302099731, 'raw_bytes_deleted': 711969769, 'raw_files_deleted': 64, 'fetch_failed': 1}
- `himawari9_b13_s0510`: {'skipped_existing': 3, 'max_raw_object_bytes': 3195603, 'max_staging_bytes_after_fetch': 22233433, 'source_issues_touched': 141, 'fetch_ok': 138, 'normalize_ok': 138, 'station_features_upserted': 13110, 'area_features_upserted': 1518, 'fetch_seconds_total': 473.6680087998393, 'normalize_seconds_total': 367.7878245001193, 'db_write_seconds_total': 28.86015909985872, 'total_seconds_total': 870.3159923998173, 'raw_bytes_deleted': 435937628, 'raw_files_deleted': 138, 'fetch_failed': 3}

## Phase Runtime

- `db_write`: {'count': 273, 'mean_seconds': 0.43200194981501816, 'p50_seconds': 0.23604240000713617, 'p90_seconds': 0.7101105999900028, 'max_seconds': 8.018367999990005}
- `fetch`: {'count': 273, 'mean_seconds': 11.497646702196144, 'p50_seconds': 6.943339899997227, 'p90_seconds': 25.515916400006972, 'max_seconds': 45.6458029000205}
- `normalize`: {'count': 272, 'mean_seconds': 4.72456444191202, 'p50_seconds': 4.130499400023837, 'p90_seconds': 7.749521300022025, 'max_seconds': 12.549855199991725}
- `total`: {'count': 273, 'mean_seconds': 16.63690699706636, 'p50_seconds': 12.333986400044523, 'p90_seconds': 33.88098350004293, 'max_seconds': 55.86283930006903}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
