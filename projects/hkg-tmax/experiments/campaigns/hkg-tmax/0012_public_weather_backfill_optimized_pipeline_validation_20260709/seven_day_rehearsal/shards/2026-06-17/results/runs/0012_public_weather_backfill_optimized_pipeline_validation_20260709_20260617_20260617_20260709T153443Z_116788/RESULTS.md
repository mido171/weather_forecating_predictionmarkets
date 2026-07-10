# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260617_20260617_20260709T153443Z_116788`
Execution mode: `optimized`
Elapsed seconds: `674.4068193000276`
Date range: `2026-06-17` to `2026-06-17`

## Counts

- Source issues touched: `280`
- Fetch ok: `276`
- Fetch failed: `4`
- Normalize ok: `276`
- Normalize failed: `None`
- Station features upserted: `15802`
- Area features upserted: `28134`
- Raw bytes deleted: `1797351895`
- Max staging bytes observed: `132890997`
- Final staging bytes: `0`
- Max raw object bytes observed: `12288255`
- Minimum free disk bytes observed: `238032371712`

## By Source

- `gefs_control`: {'max_raw_object_bytes': 9231365, 'max_staging_bytes_after_fetch': 92950096, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1144, 'area_features_upserted': 13104, 'fetch_seconds_total': 1102.808498599974, 'normalize_seconds_total': 459.24992900004145, 'db_write_seconds_total': 28.808844200044405, 'total_seconds_total': 1590.8672718000598, 'raw_bytes_deleted': 568745178, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 12288255, 'max_staging_bytes_after_fetch': 132890997, 'source_issues_touched': 68, 'fetch_ok': 66, 'normalize_ok': 66, 'station_features_upserted': 1168, 'area_features_upserted': 13468, 'fetch_seconds_total': 1828.2020291001536, 'normalize_seconds_total': 498.7346601000754, 'db_write_seconds_total': 43.294400199723896, 'total_seconds_total': 2370.231089399953, 'raw_bytes_deleted': 780470966, 'raw_files_deleted': 66, 'fetch_failed': 2}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3230354, 'max_staging_bytes_after_fetch': 22597821, 'source_issues_touched': 144, 'fetch_ok': 142, 'normalize_ok': 142, 'station_features_upserted': 13490, 'area_features_upserted': 1562, 'fetch_seconds_total': 680.087045899767, 'normalize_seconds_total': 460.5208570998511, 'db_write_seconds_total': 41.651794499892276, 'total_seconds_total': 1182.2596974995104, 'raw_bytes_deleted': 448135751, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Phase Runtime

- `db_write`: {'count': 280, 'mean_seconds': 0.40626799607021635, 'p50_seconds': 0.2756884999689646, 'p90_seconds': 0.7775777000351809, 'max_seconds': 3.1424942999728955}
- `fetch`: {'count': 280, 'mean_seconds': 12.896777048571051, 'p50_seconds': 9.481911399983801, 'p90_seconds': 29.29346539999824, 'max_seconds': 46.86998019996099}
- `normalize`: {'count': 278, 'mean_seconds': 5.102537576258878, 'p50_seconds': 4.883381400024518, 'p90_seconds': 7.9973563000094146, 'max_seconds': 13.008177200041246}
- `total`: {'count': 280, 'mean_seconds': 18.36913592392687, 'p50_seconds': 15.620068399992306, 'p90_seconds': 37.474239700008184, 'max_seconds': 55.568627499975264}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
