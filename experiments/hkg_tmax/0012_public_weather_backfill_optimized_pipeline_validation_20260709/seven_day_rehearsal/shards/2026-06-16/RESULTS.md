# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260616_20260616_20260709T152141Z_117864`
Execution mode: `optimized`
Elapsed seconds: `711.1129507999867`
Date range: `2026-06-16` to `2026-06-16`

## Counts

- Source issues touched: `280`
- Fetch ok: `277`
- Fetch failed: `3`
- Normalize ok: `277`
- Normalize failed: `None`
- Station features upserted: `15820`
- Area features upserted: `28342`
- Raw bytes deleted: `1815169172`
- Max staging bytes observed: `159148462`
- Final staging bytes: `0`
- Max raw object bytes observed: `12446099`
- Minimum free disk bytes observed: `238283382784`

## By Source

- `gefs_control`: {'max_raw_object_bytes': 9250098, 'max_staging_bytes_after_fetch': 102985008, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1144, 'area_features_upserted': 13104, 'fetch_seconds_total': 1139.4774561001104, 'normalize_seconds_total': 473.1477601000224, 'db_write_seconds_total': 34.65007629978936, 'total_seconds_total': 1647.2752924999222, 'raw_bytes_deleted': 573272357, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 12446099, 'max_staging_bytes_after_fetch': 159148462, 'source_issues_touched': 68, 'fetch_ok': 67, 'normalize_ok': 67, 'station_features_upserted': 1186, 'area_features_upserted': 13676, 'fetch_seconds_total': 1797.031752600451, 'normalize_seconds_total': 563.5418427000986, 'db_write_seconds_total': 47.9153112997883, 'total_seconds_total': 2408.488906600338, 'raw_bytes_deleted': 800148333, 'raw_files_deleted': 67, 'fetch_failed': 1}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3181716, 'max_staging_bytes_after_fetch': 22096157, 'source_issues_touched': 144, 'fetch_ok': 142, 'normalize_ok': 142, 'station_features_upserted': 13490, 'area_features_upserted': 1562, 'fetch_seconds_total': 745.5914406998781, 'normalize_seconds_total': 434.30490089952946, 'db_write_seconds_total': 40.89405310002621, 'total_seconds_total': 1220.7903946994338, 'raw_bytes_deleted': 441748482, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Phase Runtime

- `db_write`: {'count': 280, 'mean_seconds': 0.4409265739271567, 'p50_seconds': 0.2729932000511326, 'p90_seconds': 0.7821542999590747, 'max_seconds': 6.779412399977446}
- `fetch`: {'count': 280, 'mean_seconds': 13.150359462144428, 'p50_seconds': 9.256565699994098, 'p90_seconds': 28.55628889997024, 'max_seconds': 43.72848340001656}
- `normalize`: {'count': 279, 'mean_seconds': 5.272381733690503, 'p50_seconds': 4.7866613999940455, 'p90_seconds': 8.955903300025966, 'max_seconds': 14.58917609998025}
- `total`: {'count': 280, 'mean_seconds': 18.844837834998906, 'p50_seconds': 15.278665899997577, 'p90_seconds': 37.86018610006431, 'max_seconds': 50.51831630000379}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
