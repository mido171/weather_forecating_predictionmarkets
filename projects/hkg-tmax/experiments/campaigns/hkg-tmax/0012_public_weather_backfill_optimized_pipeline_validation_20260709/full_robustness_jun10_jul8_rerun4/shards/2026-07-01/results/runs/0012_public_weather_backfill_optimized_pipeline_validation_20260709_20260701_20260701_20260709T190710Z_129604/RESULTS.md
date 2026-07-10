# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260701_20260701_20260709T190710Z_129604`
Execution mode: `optimized`
Elapsed seconds: `684.5140011999756`
Date range: `2026-07-01` to `2026-07-01`

## Counts

- Source issues touched: `280`
- Fetch ok: `276`
- Fetch failed: `4`
- Normalize ok: `276`
- Normalize failed: `None`
- Station features upserted: `15802`
- Area features upserted: `28134`
- Raw bytes deleted: `1774405696`
- Max staging bytes observed: `140559959`
- Final staging bytes: `0`
- Max raw object bytes observed: `12290484`
- Minimum free disk bytes observed: `234097070080`

## By Source

- `gefs_control`: {'max_raw_object_bytes': 8621034, 'max_staging_bytes_after_fetch': 105968590, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1144, 'area_features_upserted': 13104, 'fetch_seconds_total': 1251.7931506001041, 'normalize_seconds_total': 438.3417729999055, 'db_write_seconds_total': 50.085701799951494, 'total_seconds_total': 1740.2206253999611, 'raw_bytes_deleted': 556187958, 'raw_files_deleted': 68}
- `gfs`: {'source_issues_touched': 68, 'fetch_failed': 2, 'fetch_seconds_total': 1701.662590599968, 'db_write_seconds_total': 58.61738120019436, 'total_seconds_total': 2265.6106391001376, 'max_raw_object_bytes': 12290484, 'max_staging_bytes_after_fetch': 140559959, 'fetch_ok': 66, 'normalize_ok': 66, 'station_features_upserted': 1168, 'area_features_upserted': 13468, 'normalize_seconds_total': 505.3306672999752, 'raw_bytes_deleted': 776874076, 'raw_files_deleted': 66}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3183176, 'max_staging_bytes_after_fetch': 25371949, 'source_issues_touched': 144, 'fetch_ok': 142, 'normalize_ok': 142, 'station_features_upserted': 13490, 'area_features_upserted': 1562, 'fetch_seconds_total': 774.189698099508, 'normalize_seconds_total': 430.0833793996135, 'db_write_seconds_total': 33.69988759997068, 'total_seconds_total': 1237.972965099092, 'raw_bytes_deleted': 441343662, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Phase Runtime

- `db_write`: {'count': 280, 'mean_seconds': 0.5085820378575591, 'p50_seconds': 0.2438731999718584, 'p90_seconds': 0.9029093999997713, 'max_seconds': 8.771604500012472}
- `fetch`: {'count': 280, 'mean_seconds': 13.313019426069928, 'p50_seconds': 9.48897029994987, 'p90_seconds': 29.417656400008127, 'max_seconds': 41.78901740000583}
- `normalize`: {'count': 278, 'mean_seconds': 4.94156769676077, 'p50_seconds': 4.861708399956115, 'p90_seconds': 7.7850194000056945, 'max_seconds': 13.662367300014012}
- `total`: {'count': 280, 'mean_seconds': 18.727872248568538, 'p50_seconds': 14.766002999967895, 'p90_seconds': 38.03340489999391, 'max_seconds': 49.870864399999846}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
