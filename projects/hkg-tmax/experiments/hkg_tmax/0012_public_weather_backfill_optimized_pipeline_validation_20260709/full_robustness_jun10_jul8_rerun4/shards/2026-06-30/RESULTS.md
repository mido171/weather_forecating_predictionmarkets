# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260630_20260630_20260709T190710Z_124188`
Execution mode: `optimized`
Elapsed seconds: `683.6046123999986`
Date range: `2026-06-30` to `2026-06-30`

## Counts

- Source issues touched: `280`
- Fetch ok: `278`
- Fetch failed: `2`
- Normalize ok: `278`
- Normalize failed: `None`
- Station features upserted: `15838`
- Area features upserted: `28550`
- Raw bytes deleted: `1810062292`
- Max staging bytes observed: `152123572`
- Final staging bytes: `0`
- Max raw object bytes observed: `12281591`
- Minimum free disk bytes observed: `234097033216`

## By Source

- `gefs_control`: {'max_raw_object_bytes': 8710525, 'max_staging_bytes_after_fetch': 105376741, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1144, 'area_features_upserted': 13104, 'fetch_seconds_total': 1152.1660317005008, 'normalize_seconds_total': 445.5428532997612, 'db_write_seconds_total': 49.13919459999306, 'total_seconds_total': 1646.848079600255, 'raw_bytes_deleted': 553428219, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 12281591, 'max_staging_bytes_after_fetch': 152123572, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1204, 'area_features_upserted': 13884, 'fetch_seconds_total': 1733.7248496999382, 'normalize_seconds_total': 521.0938405000488, 'db_write_seconds_total': 56.514255900110584, 'total_seconds_total': 2311.3329461000976, 'raw_bytes_deleted': 799764053, 'raw_files_deleted': 68}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3269976, 'max_staging_bytes_after_fetch': 22845914, 'source_issues_touched': 144, 'fetch_ok': 142, 'normalize_ok': 142, 'station_features_upserted': 13490, 'area_features_upserted': 1562, 'fetch_seconds_total': 743.1392627996975, 'normalize_seconds_total': 428.80050659982953, 'db_write_seconds_total': 36.00503769965144, 'total_seconds_total': 1207.9448070991784, 'raw_bytes_deleted': 456870020, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Phase Runtime

- `db_write`: {'count': 280, 'mean_seconds': 0.5059231721419825, 'p50_seconds': 0.2820406000246294, 'p90_seconds': 0.8990212999633513, 'max_seconds': 7.170874100003857}
- `fetch`: {'count': 280, 'mean_seconds': 12.960821943571915, 'p50_seconds': 8.73858730000211, 'p90_seconds': 28.44433809997281, 'max_seconds': 48.60318009997718}
- `normalize`: {'count': 280, 'mean_seconds': 4.9837042871415695, 'p50_seconds': 4.929019899980631, 'p90_seconds': 8.061499100003857, 'max_seconds': 13.941337800002657}
- `total`: {'count': 280, 'mean_seconds': 18.45044940285547, 'p50_seconds': 14.012351299927104, 'p90_seconds': 37.71930509991944, 'max_seconds': 57.393082200025674}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
