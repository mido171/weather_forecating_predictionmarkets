# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260619_20260619_20260709T154642Z_114816`
Execution mode: `optimized`
Elapsed seconds: `670.6538118000026`
Date range: `2026-06-19` to `2026-06-19`

## Counts

- Source issues touched: `280`
- Fetch ok: `277`
- Fetch failed: `3`
- Normalize ok: `277`
- Normalize failed: `None`
- Station features upserted: `15743`
- Area features upserted: `28539`
- Raw bytes deleted: `1801954746`
- Max staging bytes observed: `166970908`
- Final staging bytes: `0`
- Max raw object bytes observed: `12334902`
- Minimum free disk bytes observed: `238025175040`

## By Source

- `gefs_control`: {'max_raw_object_bytes': 8647303, 'max_staging_bytes_after_fetch': 114174925, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1144, 'area_features_upserted': 13104, 'fetch_seconds_total': 1104.8360842998372, 'normalize_seconds_total': 463.0490764998831, 'db_write_seconds_total': 38.13255590054905, 'total_seconds_total': 1606.0177167002694, 'raw_bytes_deleted': 559113291, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 12334902, 'max_staging_bytes_after_fetch': 166970908, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1204, 'area_features_upserted': 13884, 'fetch_seconds_total': 1441.4218903001165, 'normalize_seconds_total': 522.2895124995848, 'db_write_seconds_total': 49.46811889944365, 'total_seconds_total': 2013.179521699145, 'raw_bytes_deleted': 798837597, 'raw_files_deleted': 68}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3187305, 'max_staging_bytes_after_fetch': 25407671, 'source_issues_touched': 144, 'fetch_ok': 141, 'normalize_ok': 141, 'station_features_upserted': 13395, 'area_features_upserted': 1551, 'fetch_seconds_total': 635.7722321998444, 'normalize_seconds_total': 448.8757066000253, 'db_write_seconds_total': 41.06608970032539, 'total_seconds_total': 1125.7140285001951, 'raw_bytes_deleted': 444003858, 'raw_files_deleted': 141, 'fetch_failed': 3}

## Phase Runtime

- `db_write`: {'count': 280, 'mean_seconds': 0.4595241589297075, 'p50_seconds': 0.25898480002069846, 'p90_seconds': 0.751313199987635, 'max_seconds': 7.5670082999859005}
- `fetch`: {'count': 280, 'mean_seconds': 11.364393595713565, 'p50_seconds': 10.33490780001739, 'p90_seconds': 22.63733759999741, 'max_seconds': 38.13656749995425}
- `normalize`: {'count': 280, 'mean_seconds': 5.122193912855333, 'p50_seconds': 4.84397230000468, 'p90_seconds': 7.945436000009067, 'max_seconds': 14.419420699996408}
- `total`: {'count': 280, 'mean_seconds': 16.946111667498606, 'p50_seconds': 15.451976000098512, 'p90_seconds': 31.865632999979425, 'max_seconds': 49.528343199926894}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
