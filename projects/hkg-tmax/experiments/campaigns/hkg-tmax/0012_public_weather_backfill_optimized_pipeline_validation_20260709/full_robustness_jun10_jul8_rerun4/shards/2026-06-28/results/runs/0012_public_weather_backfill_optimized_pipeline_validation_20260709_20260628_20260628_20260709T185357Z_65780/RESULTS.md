# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260628_20260628_20260709T185357Z_65780`
Execution mode: `optimized`
Elapsed seconds: `704.9488757000072`
Date range: `2026-06-28` to `2026-06-28`

## Counts

- Source issues touched: `280`
- Fetch ok: `277`
- Fetch failed: `3`
- Normalize ok: `277`
- Normalize failed: `None`
- Station features upserted: `15820`
- Area features upserted: `28342`
- Raw bytes deleted: `1820796751`
- Max staging bytes observed: `145109856`
- Final staging bytes: `0`
- Max raw object bytes observed: `12371250`
- Minimum free disk bytes observed: `234980716544`

## By Source

- `gefs_control`: {'max_raw_object_bytes': 9618543, 'max_staging_bytes_after_fetch': 112304979, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1144, 'area_features_upserted': 13104, 'fetch_seconds_total': 1282.6414460001397, 'normalize_seconds_total': 439.97010910027893, 'db_write_seconds_total': 51.627345800050534, 'total_seconds_total': 1774.2389009004692, 'raw_bytes_deleted': 573760518, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 12371250, 'max_staging_bytes_after_fetch': 145109856, 'source_issues_touched': 68, 'fetch_ok': 67, 'normalize_ok': 67, 'station_features_upserted': 1186, 'area_features_upserted': 13676, 'fetch_seconds_total': 1791.740392500069, 'normalize_seconds_total': 493.6790517997579, 'db_write_seconds_total': 62.02656600018963, 'total_seconds_total': 2347.4460103000165, 'raw_bytes_deleted': 796437110, 'raw_files_deleted': 67, 'fetch_failed': 1}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3239798, 'max_staging_bytes_after_fetch': 21812842, 'source_issues_touched': 144, 'fetch_ok': 142, 'normalize_ok': 142, 'station_features_upserted': 13490, 'area_features_upserted': 1562, 'fetch_seconds_total': 1103.4413661000435, 'normalize_seconds_total': 391.5604451003019, 'db_write_seconds_total': 33.95274409989361, 'total_seconds_total': 1528.954555300239, 'raw_bytes_deleted': 450599123, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Phase Runtime

- `db_write`: {'count': 280, 'mean_seconds': 0.5271666282147635, 'p50_seconds': 0.26532419997965917, 'p90_seconds': 1.1225174999563023, 'max_seconds': 11.71605869999621}
- `fetch`: {'count': 280, 'mean_seconds': 14.920797159286614, 'p50_seconds': 11.649699500005227, 'p90_seconds': 30.11341240000911, 'max_seconds': 54.24664289999055}
- `normalize`: {'count': 279, 'mean_seconds': 4.7498552186392065, 'p50_seconds': 4.24233620002633, 'p90_seconds': 7.76873249997152, 'max_seconds': 12.12725409999257}
- `total`: {'count': 280, 'mean_seconds': 20.180855237502588, 'p50_seconds': 17.832720300008077, 'p90_seconds': 38.98814390000189, 'max_seconds': 64.21944930002792}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
