# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260706_20260706_20260709T194446Z_111088`
Execution mode: `optimized`
Elapsed seconds: `645.9588132000063`
Date range: `2026-07-06` to `2026-07-06`

## Counts

- Source issues touched: `280`
- Fetch ok: `278`
- Fetch failed: `2`
- Normalize ok: `278`
- Normalize failed: `None`
- Station features upserted: `15838`
- Area features upserted: `28550`
- Raw bytes deleted: `1718751695`
- Max staging bytes observed: `155030540`
- Final staging bytes: `0`
- Max raw object bytes observed: `11685822`
- Minimum free disk bytes observed: `232352919552`

## By Source

- `gefs_control`: {'max_raw_object_bytes': 8681431, 'max_staging_bytes_after_fetch': 109774474, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1144, 'area_features_upserted': 13104, 'fetch_seconds_total': 1233.1206586998305, 'normalize_seconds_total': 454.2445963998907, 'db_write_seconds_total': 33.28601519984659, 'total_seconds_total': 1720.6512702995678, 'raw_bytes_deleted': 534447058, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 11685822, 'max_staging_bytes_after_fetch': 155030540, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1204, 'area_features_upserted': 13884, 'fetch_seconds_total': 1304.013954999973, 'normalize_seconds_total': 501.8216756999609, 'db_write_seconds_total': 61.57698790024733, 'total_seconds_total': 1867.4126186001813, 'raw_bytes_deleted': 741300283, 'raw_files_deleted': 68}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3173102, 'max_staging_bytes_after_fetch': 21913959, 'source_issues_touched': 144, 'fetch_ok': 142, 'normalize_ok': 142, 'station_features_upserted': 13490, 'area_features_upserted': 1562, 'fetch_seconds_total': 672.8972174004884, 'normalize_seconds_total': 378.64635459997226, 'db_write_seconds_total': 31.738228199479636, 'total_seconds_total': 1083.2818001999403, 'raw_bytes_deleted': 443004354, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Phase Runtime

- `db_write`: {'count': 280, 'mean_seconds': 0.45214725464133415, 'p50_seconds': 0.24179100000765175, 'p90_seconds': 0.6105176000273786, 'max_seconds': 12.257474900048692}
- `fetch`: {'count': 280, 'mean_seconds': 11.464399396786757, 'p50_seconds': 9.353677100036293, 'p90_seconds': 24.374521899968386, 'max_seconds': 43.960599199985154}
- `normalize`: {'count': 280, 'mean_seconds': 4.7668308096422285, 'p50_seconds': 4.200860599987209, 'p90_seconds': 7.663849599950481, 'max_seconds': 15.696846599981654}
- `total`: {'count': 280, 'mean_seconds': 16.683377461070318, 'p50_seconds': 15.111290300032124, 'p90_seconds': 32.507308500004, 'max_seconds': 56.14025950001087}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
