# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260704_20260704_20260709T193231Z_127304`
Execution mode: `optimized`
Elapsed seconds: `653.506396199984`
Date range: `2026-07-04` to `2026-07-04`

## Counts

- Source issues touched: `280`
- Fetch ok: `278`
- Fetch failed: `2`
- Normalize ok: `278`
- Normalize failed: `None`
- Station features upserted: `15838`
- Area features upserted: `28550`
- Raw bytes deleted: `1733728286`
- Max staging bytes observed: `143602379`
- Final staging bytes: `0`
- Max raw object bytes observed: `11323220`
- Minimum free disk bytes observed: `231322345472`

## By Source

- `gefs_control`: {'max_raw_object_bytes': 8663878, 'max_staging_bytes_after_fetch': 102872638, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1144, 'area_features_upserted': 13104, 'fetch_seconds_total': 1222.5154353998369, 'normalize_seconds_total': 442.0410713001038, 'db_write_seconds_total': 44.690159899997525, 'total_seconds_total': 1709.2466665999382, 'raw_bytes_deleted': 543285300, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 11323220, 'max_staging_bytes_after_fetch': 143602379, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1204, 'area_features_upserted': 13884, 'fetch_seconds_total': 1403.2888000000967, 'normalize_seconds_total': 499.10214150021784, 'db_write_seconds_total': 64.18006230029278, 'total_seconds_total': 1966.5710038006073, 'raw_bytes_deleted': 738064977, 'raw_files_deleted': 68}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3242803, 'max_staging_bytes_after_fetch': 25656421, 'source_issues_touched': 144, 'fetch_ok': 142, 'normalize_ok': 142, 'station_features_upserted': 13490, 'area_features_upserted': 1562, 'fetch_seconds_total': 698.5297088997904, 'normalize_seconds_total': 432.99837499985006, 'db_write_seconds_total': 34.523376899946015, 'total_seconds_total': 1166.0514607995865, 'raw_bytes_deleted': 452378009, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Phase Runtime

- `db_write`: {'count': 280, 'mean_seconds': 0.5121199967865583, 'p50_seconds': 0.25758769997628406, 'p90_seconds': 0.9397066999808885, 'max_seconds': 8.178116199967917}
- `fetch`: {'count': 280, 'mean_seconds': 11.87262122964187, 'p50_seconds': 9.214561200002208, 'p90_seconds': 25.28351770003792, 'max_seconds': 39.10523089999333}
- `normalize`: {'count': 280, 'mean_seconds': 4.907648527857756, 'p50_seconds': 4.94101429998409, 'p90_seconds': 7.5595286000170745, 'max_seconds': 14.851537699985784}
- `total`: {'count': 280, 'mean_seconds': 17.292389754286187, 'p50_seconds': 14.980469900008757, 'p90_seconds': 33.01612940005725, 'max_seconds': 48.157389000058174}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
