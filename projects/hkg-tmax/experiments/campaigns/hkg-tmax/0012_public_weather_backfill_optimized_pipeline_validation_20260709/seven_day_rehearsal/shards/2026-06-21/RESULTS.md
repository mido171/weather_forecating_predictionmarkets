# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260621_20260621_20260709T155849Z_57332`
Execution mode: `optimized`
Elapsed seconds: `647.9931631000363`
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
- Max staging bytes observed: `169313139`
- Final staging bytes: `0`
- Max raw object bytes observed: `12384055`
- Minimum free disk bytes observed: `238110330880`

## By Source

- `gefs_control`: {'max_raw_object_bytes': 9034067, 'max_staging_bytes_after_fetch': 107711504, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1144, 'area_features_upserted': 13104, 'fetch_seconds_total': 897.9236511001363, 'normalize_seconds_total': 440.7241339000175, 'db_write_seconds_total': 31.730729599948972, 'total_seconds_total': 1370.3785146001028, 'raw_bytes_deleted': 559003347, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 12384055, 'max_staging_bytes_after_fetch': 169313139, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1204, 'area_features_upserted': 13884, 'fetch_seconds_total': 1257.2379048999283, 'normalize_seconds_total': 539.9727754000924, 'db_write_seconds_total': 65.06066120020114, 'total_seconds_total': 1862.2713415002218, 'raw_bytes_deleted': 808312537, 'raw_files_deleted': 68}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3240229, 'max_staging_bytes_after_fetch': 25884458, 'source_issues_touched': 144, 'fetch_ok': 142, 'normalize_ok': 142, 'station_features_upserted': 13490, 'area_features_upserted': 1562, 'fetch_seconds_total': 510.0236013007816, 'normalize_seconds_total': 433.0573191997246, 'db_write_seconds_total': 39.973080700146966, 'total_seconds_total': 983.0540012006531, 'raw_bytes_deleted': 456907779, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Phase Runtime

- `db_write`: {'count': 280, 'mean_seconds': 0.4884445410724896, 'p50_seconds': 0.2283051999984309, 'p90_seconds': 0.6113648000173271, 'max_seconds': 13.458867300010752}
- `fetch`: {'count': 280, 'mean_seconds': 9.518518418931594, 'p50_seconds': 6.4174981999676675, 'p90_seconds': 19.73692390002543, 'max_seconds': 31.943336999975145}
- `normalize`: {'count': 280, 'mean_seconds': 5.049122244642266, 'p50_seconds': 4.662020300049335, 'p90_seconds': 8.22226149996277, 'max_seconds': 16.174616800039075}
- `total`: {'count': 280, 'mean_seconds': 15.05608520464635, 'p50_seconds': 11.091003800102044, 'p90_seconds': 28.682718799973372, 'max_seconds': 40.90947619994404}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
