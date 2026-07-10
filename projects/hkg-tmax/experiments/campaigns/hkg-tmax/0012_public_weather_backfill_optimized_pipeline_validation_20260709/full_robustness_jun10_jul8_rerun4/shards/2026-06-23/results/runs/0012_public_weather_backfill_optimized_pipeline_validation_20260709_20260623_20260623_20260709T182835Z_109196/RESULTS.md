# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260623_20260623_20260709T182835Z_109196`
Execution mode: `optimized`
Elapsed seconds: `252.78597959998297`
Date range: `2026-06-23` to `2026-06-23`

## Counts

- Source issues touched: `165`
- Fetch ok: `163`
- Fetch failed: `2`
- Normalize ok: `163`
- Normalize failed: `None`
- Station features upserted: `13844`
- Area features upserted: `5618`
- Raw bytes deleted: `615849097`
- Max staging bytes observed: `120998895`
- Final staging bytes: `0`
- Max raw object bytes observed: `8564365`
- Minimum free disk bytes observed: `237029994496`

## By Source

- `gefs_control`: {'skipped_existing': 47, 'max_raw_object_bytes': 8564365, 'max_staging_bytes_after_fetch': 120998895, 'source_issues_touched': 21, 'fetch_ok': 21, 'normalize_ok': 21, 'station_features_upserted': 354, 'area_features_upserted': 4056, 'fetch_seconds_total': 410.7054908001446, 'normalize_seconds_total': 137.3653933001915, 'db_write_seconds_total': 6.233942299964838, 'total_seconds_total': 554.304826400301, 'raw_bytes_deleted': 170306763, 'raw_files_deleted': 21}
- `gfs`: {'skipped_existing': 68}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3208092, 'max_staging_bytes_after_fetch': 22006560, 'source_issues_touched': 144, 'fetch_ok': 142, 'normalize_ok': 142, 'station_features_upserted': 13490, 'area_features_upserted': 1562, 'fetch_seconds_total': 722.8668585994164, 'normalize_seconds_total': 408.0368000996532, 'db_write_seconds_total': 31.2882295998279, 'total_seconds_total': 1162.1918882988975, 'raw_bytes_deleted': 445542334, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Phase Runtime

- `db_write`: {'count': 165, 'mean_seconds': 0.22740710242298628, 'p50_seconds': 0.21309689996996894, 'p90_seconds': 0.41450860002078116, 'max_seconds': 1.0126146000111476}
- `fetch`: {'count': 165, 'mean_seconds': 6.870135450906431, 'p50_seconds': 4.458548600028735, 'p90_seconds': 16.74068199994508, 'max_seconds': 37.79723040002864}
- `normalize`: {'count': 165, 'mean_seconds': 3.3054678387869374, 'p50_seconds': 2.807665399974212, 'p90_seconds': 5.454829599999357, 'max_seconds': 15.863515099976212}
- `total`: {'count': 165, 'mean_seconds': 10.403010392116354, 'p50_seconds': 7.507510599971283, 'p90_seconds': 20.245502400037367, 'max_seconds': 45.15474810008891}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
