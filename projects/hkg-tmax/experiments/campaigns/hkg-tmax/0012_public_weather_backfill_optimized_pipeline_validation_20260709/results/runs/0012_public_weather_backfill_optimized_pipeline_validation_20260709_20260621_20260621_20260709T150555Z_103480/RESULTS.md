# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260621_20260621_20260709T150555Z_103480`
Execution mode: `optimized`
Elapsed seconds: `780.8856843000394`
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
- Max staging bytes observed: `177595054`
- Final staging bytes: `0`
- Max raw object bytes observed: `12384055`
- Minimum free disk bytes observed: `238549925888`

## By Source

- `gefs_control`: {'max_raw_object_bytes': 9034067, 'max_staging_bytes_after_fetch': 91465001, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1144, 'area_features_upserted': 13104, 'fetch_seconds_total': 1056.7556117997156, 'normalize_seconds_total': 482.5230314997025, 'db_write_seconds_total': 65.23619400011376, 'total_seconds_total': 1604.5148372995318, 'raw_bytes_deleted': 559003347, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 12384055, 'max_staging_bytes_after_fetch': 177595054, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1204, 'area_features_upserted': 13884, 'fetch_seconds_total': 1467.6343530000886, 'normalize_seconds_total': 739.6332096999977, 'db_write_seconds_total': 65.25982970005134, 'total_seconds_total': 2272.5273924001376, 'raw_bytes_deleted': 808312537, 'raw_files_deleted': 68}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3240229, 'max_staging_bytes_after_fetch': 25688144, 'source_issues_touched': 144, 'fetch_ok': 142, 'normalize_ok': 142, 'station_features_upserted': 13490, 'area_features_upserted': 1562, 'fetch_seconds_total': 533.6395633994252, 'normalize_seconds_total': 456.0091644997592, 'db_write_seconds_total': 39.73435640003299, 'total_seconds_total': 1029.3830842992174, 'raw_bytes_deleted': 456907779, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Phase Runtime

- `db_write`: {'count': 280, 'mean_seconds': 0.6079656432149931, 'p50_seconds': 0.2679120000102557, 'p90_seconds': 1.0196018000133336, 'max_seconds': 12.42775899998378}
- `fetch`: {'count': 280, 'mean_seconds': 10.921534029282961, 'p50_seconds': 6.394888999988325, 'p90_seconds': 23.39140010002302, 'max_seconds': 32.843538300017826}
- `normalize`: {'count': 280, 'mean_seconds': 5.9934478774980695, 'p50_seconds': 4.4276408000150695, 'p90_seconds': 11.073618300026283, 'max_seconds': 26.353143799991813}
- `total`: {'count': 280, 'mean_seconds': 17.522947549996026, 'p50_seconds': 12.528676700021606, 'p90_seconds': 34.76894649997121, 'max_seconds': 52.5493108999799}

## Resource Telemetry

- `{'cpu_sampler_available': False, 'cpu_mean_percent': None, 'cpu_max_percent': None, 'staging_max_bytes': 177595054, 'staging_end_bytes': 0}`

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
