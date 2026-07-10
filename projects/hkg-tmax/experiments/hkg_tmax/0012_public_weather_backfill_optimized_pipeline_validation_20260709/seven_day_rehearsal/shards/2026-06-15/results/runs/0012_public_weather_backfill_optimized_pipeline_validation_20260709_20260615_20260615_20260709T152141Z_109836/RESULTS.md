# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260615_20260615_20260709T152141Z_109836`
Execution mode: `optimized`
Elapsed seconds: `718.5351940999972`
Date range: `2026-06-15` to `2026-06-15`

## Counts

- Source issues touched: `280`
- Fetch ok: `277`
- Fetch failed: `3`
- Normalize ok: `277`
- Normalize failed: `None`
- Station features upserted: `15743`
- Area features upserted: `28539`
- Raw bytes deleted: `1786039424`
- Max staging bytes observed: `155356055`
- Final staging bytes: `0`
- Max raw object bytes observed: `12499160`
- Minimum free disk bytes observed: `238283309056`

## By Source

- `gefs_control`: {'max_raw_object_bytes': 8902811, 'max_staging_bytes_after_fetch': 101375374, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1144, 'area_features_upserted': 13104, 'fetch_seconds_total': 1141.6637919001514, 'normalize_seconds_total': 458.19507110037375, 'db_write_seconds_total': 32.92379629990319, 'total_seconds_total': 1632.7826593004283, 'raw_bytes_deleted': 550249943, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 12499160, 'max_staging_bytes_after_fetch': 155356055, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1204, 'area_features_upserted': 13884, 'fetch_seconds_total': 1836.2704625998158, 'normalize_seconds_total': 581.1707723999862, 'db_write_seconds_total': 58.09357159974752, 'total_seconds_total': 2475.5348065995495, 'raw_bytes_deleted': 812942804, 'raw_files_deleted': 68}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3027972, 'max_staging_bytes_after_fetch': 24188410, 'source_issues_touched': 144, 'fetch_ok': 141, 'normalize_ok': 141, 'station_features_upserted': 13395, 'area_features_upserted': 1551, 'fetch_seconds_total': 749.5725044000428, 'normalize_seconds_total': 423.9190480996622, 'db_write_seconds_total': 37.431492100527976, 'total_seconds_total': 1210.923044600233, 'raw_bytes_deleted': 422846677, 'raw_files_deleted': 141, 'fetch_failed': 3}

## Phase Runtime

- `db_write`: {'count': 280, 'mean_seconds': 0.45874592857206675, 'p50_seconds': 0.26431130000855774, 'p90_seconds': 0.8059509000158869, 'max_seconds': 8.033841900003608}
- `fetch`: {'count': 280, 'mean_seconds': 13.312524138928607, 'p50_seconds': 9.68981349997921, 'p90_seconds': 30.574529900040943, 'max_seconds': 41.01019619998988}
- `normalize`: {'count': 280, 'mean_seconds': 5.2260174700000785, 'p50_seconds': 4.3791527999565005, 'p90_seconds': 9.027712100010831, 'max_seconds': 14.597531900042668}
- `total`: {'count': 280, 'mean_seconds': 18.997287537500753, 'p50_seconds': 14.379273300000932, 'p90_seconds': 40.17214649997186, 'max_seconds': 50.53310760000022}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
