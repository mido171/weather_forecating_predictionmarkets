# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260621_20260621_20260709T145656Z_94808`
Execution mode: `optimized`
Elapsed seconds: `410.010841100011`
Date range: `2026-06-21` to `2026-06-21`

## Counts

- Source issues touched: `280`
- Fetch ok: `278`
- Fetch failed: `2`
- Normalize ok: `278`
- Normalize failed: `None`
- Station features upserted: `15838`
- Area features upserted: `28550`
- Raw bytes deleted: `1824223663`
- Max staging bytes observed: `0`
- Final staging bytes: `0`
- Max raw object bytes observed: `12384055`
- Minimum free disk bytes observed: `238903066624`

## By Source

- `gefs_control`: {'max_raw_object_bytes': 9034067, 'max_staging_bytes_after_fetch': 0, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1144, 'area_features_upserted': 13104, 'fetch_seconds_total': 814.209210599598, 'normalize_seconds_total': 258.54857629997423, 'db_write_seconds_total': 34.90056910010753, 'total_seconds_total': 1107.6583559996798, 'raw_bytes_deleted': 559003347, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 12384055, 'max_staging_bytes_after_fetch': 0, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1204, 'area_features_upserted': 13884, 'fetch_seconds_total': 971.8564221996348, 'normalize_seconds_total': 292.1075748002622, 'db_write_seconds_total': 55.70900849980535, 'total_seconds_total': 1319.6730054997024, 'raw_bytes_deleted': 808312537, 'raw_files_deleted': 68}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3240229, 'max_staging_bytes_after_fetch': 0, 'source_issues_touched': 144, 'fetch_ok': 142, 'normalize_ok': 142, 'station_features_upserted': 13490, 'area_features_upserted': 1562, 'fetch_seconds_total': 507.5821956007276, 'normalize_seconds_total': 308.0742977994378, 'db_write_seconds_total': 26.63783619971946, 'total_seconds_total': 842.2943295998848, 'raw_bytes_deleted': 456907779, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Phase Runtime

- `db_write`: {'count': 280, 'mean_seconds': 0.4187407635701155, 'p50_seconds': 0.11996159999398515, 'p90_seconds': 0.614242399984505, 'max_seconds': 8.562470800010487}
- `fetch`: {'count': 280, 'mean_seconds': 8.191599387142716, 'p50_seconds': 5.680377500015311, 'p90_seconds': 15.868868999998085, 'max_seconds': 24.223564799991436}
- `normalize`: {'count': 280, 'mean_seconds': 3.0668944603559796, 'p50_seconds': 2.9035552000277676, 'p90_seconds': 4.865996299951803, 'max_seconds': 8.199662400002126}
- `total`: {'count': 280, 'mean_seconds': 11.67723461106881, 'p50_seconds': 9.220534199965186, 'p90_seconds': 20.934132800030056, 'max_seconds': 30.802369999990333}

## Resource Telemetry

- `{'cpu_sampler_available': False, 'cpu_mean_percent': None, 'cpu_max_percent': None, 'staging_max_bytes': 0, 'staging_end_bytes': 0}`

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
