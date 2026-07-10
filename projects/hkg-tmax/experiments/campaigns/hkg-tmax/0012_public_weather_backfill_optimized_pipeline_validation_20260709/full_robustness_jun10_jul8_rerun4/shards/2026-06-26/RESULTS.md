# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260626_20260626_20260709T184043Z_127460`
Execution mode: `optimized`
Elapsed seconds: `685.3554500999744`
Date range: `2026-06-26` to `2026-06-26`

## Counts

- Source issues touched: `280`
- Fetch ok: `278`
- Fetch failed: `2`
- Normalize ok: `278`
- Normalize failed: `None`
- Station features upserted: `15838`
- Area features upserted: `28550`
- Raw bytes deleted: `1831100604`
- Max staging bytes observed: `153579608`
- Final staging bytes: `0`
- Max raw object bytes observed: `12363708`
- Minimum free disk bytes observed: `236960325632`

## By Source

- `gefs_control`: {'max_raw_object_bytes': 8963119, 'max_staging_bytes_after_fetch': 99233912, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1144, 'area_features_upserted': 13104, 'fetch_seconds_total': 1183.989678700047, 'normalize_seconds_total': 448.2391572998022, 'db_write_seconds_total': 44.14479009999195, 'total_seconds_total': 1676.3736260998412, 'raw_bytes_deleted': 562861835, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 12363708, 'max_staging_bytes_after_fetch': 153579608, 'source_issues_touched': 68, 'fetch_ok': 68, 'normalize_ok': 68, 'station_features_upserted': 1204, 'area_features_upserted': 13884, 'fetch_seconds_total': 1704.88032019994, 'normalize_seconds_total': 529.7598408997292, 'db_write_seconds_total': 53.66352210001787, 'total_seconds_total': 2288.303683199687, 'raw_bytes_deleted': 807855095, 'raw_files_deleted': 68}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3278365, 'max_staging_bytes_after_fetch': 22923348, 'source_issues_touched': 144, 'fetch_ok': 142, 'normalize_ok': 142, 'station_features_upserted': 13490, 'area_features_upserted': 1562, 'fetch_seconds_total': 747.583091299748, 'normalize_seconds_total': 424.30733899999177, 'db_write_seconds_total': 34.534957800235134, 'total_seconds_total': 1206.425388099975, 'raw_bytes_deleted': 460383674, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Phase Runtime

- `db_write`: {'count': 280, 'mean_seconds': 0.4726545357151606, 'p50_seconds': 0.26170830003684387, 'p90_seconds': 0.833621800004039, 'max_seconds': 7.257322700053919}
- `fetch`: {'count': 280, 'mean_seconds': 12.987332464999053, 'p50_seconds': 10.57236519997241, 'p90_seconds': 27.502743199991528, 'max_seconds': 59.42789960000664}
- `normalize`: {'count': 280, 'mean_seconds': 5.008236918569725, 'p50_seconds': 4.704426700016484, 'p90_seconds': 7.898822900024243, 'max_seconds': 13.970599299995229}
- `total`: {'count': 280, 'mean_seconds': 18.46822391928394, 'p50_seconds': 15.310884399921633, 'p90_seconds': 35.9245214999537, 'max_seconds': 68.64920450001955}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
