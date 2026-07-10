# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260703_20260703_20260709T192026Z_128404`
Execution mode: `optimized`
Elapsed seconds: `650.7033796000178`
Date range: `2026-07-03` to `2026-07-03`

## Counts

- Source issues touched: `280`
- Fetch ok: `277`
- Fetch failed: `3`
- Normalize ok: `277`
- Normalize failed: `None`
- Station features upserted: `15825`
- Area features upserted: `28407`
- Raw bytes deleted: `1752751663`
- Max staging bytes observed: `139987475`
- Final staging bytes: `0`
- Max raw object bytes observed: `12125582`
- Minimum free disk bytes observed: `233427750912`

## By Source

- `gefs_control`: {'max_raw_object_bytes': 8825231, 'max_staging_bytes_after_fetch': 92513784, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1144, 'area_features_upserted': 13104, 'fetch_seconds_total': 1134.4945508999517, 'normalize_seconds_total': 428.53261569997994, 'db_write_seconds_total': 42.41083969990723, 'total_seconds_total': 1605.438006299839, 'raw_bytes_deleted': 551331510, 'raw_files_deleted': 68}
- `gfs`: {'source_issues_touched': 68, 'fetch_failed': 1, 'fetch_seconds_total': 1694.334839899675, 'db_write_seconds_total': 58.50215759978164, 'total_seconds_total': 2244.5320198991685, 'max_raw_object_bytes': 12125582, 'max_staging_bytes_after_fetch': 139987475, 'fetch_ok': 67, 'normalize_ok': 67, 'station_features_upserted': 1191, 'area_features_upserted': 13741, 'normalize_seconds_total': 491.6950223997119, 'raw_bytes_deleted': 748131702, 'raw_files_deleted': 67}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3230748, 'max_staging_bytes_after_fetch': 22231332, 'source_issues_touched': 144, 'fetch_ok': 142, 'normalize_ok': 142, 'station_features_upserted': 13490, 'area_features_upserted': 1562, 'fetch_seconds_total': 660.5366461994126, 'normalize_seconds_total': 425.73779249942163, 'db_write_seconds_total': 33.650297000363935, 'total_seconds_total': 1119.9247356991982, 'raw_bytes_deleted': 453288451, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Phase Runtime

- `db_write`: {'count': 280, 'mean_seconds': 0.48058319392876003, 'p50_seconds': 0.25880370003869757, 'p90_seconds': 0.8423554000328295, 'max_seconds': 7.409319500031415}
- `fetch`: {'count': 280, 'mean_seconds': 12.462021560710856, 'p50_seconds': 8.802420300024096, 'p90_seconds': 27.6295157999848, 'max_seconds': 39.97140079998644}
- `normalize`: {'count': 279, 'mean_seconds': 4.824248855193955, 'p50_seconds': 4.424738000030629, 'p90_seconds': 7.696753099968191, 'max_seconds': 11.511189399985597}
- `total`: {'count': 280, 'mean_seconds': 17.74962414963645, 'p50_seconds': 13.411276300030295, 'p90_seconds': 34.72769169998355, 'max_seconds': 47.110458899929654}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
