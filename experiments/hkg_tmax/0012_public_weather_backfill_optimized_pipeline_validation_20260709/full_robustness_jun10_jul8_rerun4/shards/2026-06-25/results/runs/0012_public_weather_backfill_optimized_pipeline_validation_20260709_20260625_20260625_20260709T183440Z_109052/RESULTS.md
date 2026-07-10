# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260625_20260625_20260709T183440Z_109052`
Execution mode: `optimized`
Elapsed seconds: `270.0951600999688`
Date range: `2026-06-25` to `2026-06-25`

## Counts

- Source issues touched: `172`
- Fetch ok: `170`
- Fetch failed: `2`
- Normalize ok: `170`
- Normalize failed: `None`
- Station features upserted: `13963`
- Area features upserted: `6983`
- Raw bytes deleted: `688858321`
- Max staging bytes observed: `123582427`
- Final staging bytes: `0`
- Max raw object bytes observed: `9064031`
- Minimum free disk bytes observed: `236987535360`

## By Source

- `gefs_control`: {'skipped_existing': 40, 'max_raw_object_bytes': 9064031, 'max_staging_bytes_after_fetch': 123582427, 'source_issues_touched': 28, 'fetch_ok': 28, 'normalize_ok': 28, 'station_features_upserted': 473, 'area_features_upserted': 5421, 'fetch_seconds_total': 537.9460975998663, 'normalize_seconds_total': 190.56893639999907, 'db_write_seconds_total': 13.9975989999366, 'total_seconds_total': 742.512632999802, 'raw_bytes_deleted': 231476032, 'raw_files_deleted': 28}
- `gfs`: {'skipped_existing': 68}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3246142, 'max_staging_bytes_after_fetch': 22662374, 'source_issues_touched': 144, 'fetch_ok': 142, 'normalize_ok': 142, 'station_features_upserted': 13490, 'area_features_upserted': 1562, 'fetch_seconds_total': 632.3990125001292, 'normalize_seconds_total': 443.897884899925, 'db_write_seconds_total': 34.57147979992442, 'total_seconds_total': 1110.8683771999786, 'raw_bytes_deleted': 457382289, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Phase Runtime

- `db_write`: {'count': 172, 'mean_seconds': 0.28237836511547104, 'p50_seconds': 0.23831570002948865, 'p90_seconds': 0.449821799993515, 'max_seconds': 5.918910999957006}
- `fetch`: {'count': 172, 'mean_seconds': 6.80433203546509, 'p50_seconds': 4.151101500028744, 'p90_seconds': 13.088038800051436, 'max_seconds': 37.33564110001316}
- `normalize`: {'count': 172, 'mean_seconds': 3.6887605889530466, 'p50_seconds': 3.035818200034555, 'p90_seconds': 6.341422500030603, 'max_seconds': 11.874805899977218}
- `total`: {'count': 172, 'mean_seconds': 10.775470989533607, 'p50_seconds': 7.8808956000139005, 'p90_seconds': 20.043071799969766, 'max_seconds': 44.256205500045326}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
