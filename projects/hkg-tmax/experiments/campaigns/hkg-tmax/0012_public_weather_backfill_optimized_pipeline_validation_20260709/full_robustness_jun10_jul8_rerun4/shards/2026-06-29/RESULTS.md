# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260629_20260629_20260709T185356Z_40908`
Execution mode: `optimized`
Elapsed seconds: `692.3549331999966`
Date range: `2026-06-29` to `2026-06-29`

## Counts

- Source issues touched: `280`
- Fetch ok: `278`
- Fetch failed: `2`
- Normalize ok: `278`
- Normalize failed: `None`
- Station features upserted: `15838`
- Area features upserted: `28550`
- Raw bytes deleted: `1824222892`
- Max staging bytes observed: `121206666`
- Final staging bytes: `0`
- Max raw object bytes observed: `12322485`
- Minimum free disk bytes observed: `235034578944`

## By Source

- `gefs_control`: {'max_raw_object_bytes': 9179911, 'max_staging_bytes_after_fetch': 98202592, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1144, 'area_features_upserted': 13104, 'fetch_seconds_total': 1257.5874495999888, 'normalize_seconds_total': 429.18875120033044, 'db_write_seconds_total': 35.67107730003772, 'total_seconds_total': 1722.447278100357, 'raw_bytes_deleted': 562042802, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 12322485, 'max_staging_bytes_after_fetch': 121206666, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1204, 'area_features_upserted': 13884, 'fetch_seconds_total': 1663.3029318996123, 'normalize_seconds_total': 497.06828549987404, 'db_write_seconds_total': 61.492753299942706, 'total_seconds_total': 2221.863970699429, 'raw_bytes_deleted': 804595394, 'raw_files_deleted': 68}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3251288, 'max_staging_bytes_after_fetch': 19478938, 'source_issues_touched': 144, 'fetch_ok': 142, 'normalize_ok': 142, 'station_features_upserted': 13490, 'area_features_upserted': 1562, 'fetch_seconds_total': 1072.3894352000789, 'normalize_seconds_total': 382.24108420073753, 'db_write_seconds_total': 31.838751099829096, 'total_seconds_total': 1486.4692705006455, 'raw_bytes_deleted': 457584696, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Phase Runtime

- `db_write`: {'count': 280, 'mean_seconds': 0.4607235060707483, 'p50_seconds': 0.2612648999784142, 'p90_seconds': 0.9230478999670595, 'max_seconds': 7.532336199947167}
- `fetch`: {'count': 280, 'mean_seconds': 14.261713631070286, 'p50_seconds': 12.208331900008488, 'p90_seconds': 27.938289400015492, 'max_seconds': 42.39139880001312}
- `normalize`: {'count': 280, 'mean_seconds': 4.673207574646222, 'p50_seconds': 4.336972700024489, 'p90_seconds': 7.4605804000166245, 'max_seconds': 11.604480400041211}
- `total`: {'count': 280, 'mean_seconds': 19.395644711787256, 'p50_seconds': 17.91345039999578, 'p90_seconds': 35.940818399889395, 'max_seconds': 49.83347449998837}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
