# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260624_20260624_20260709T164235Z_125360`
Execution mode: `optimized`
Elapsed seconds: `303.3646301999688`
Date range: `2026-06-24` to `2026-06-24`

## Counts

- Source issues touched: `172`
- Fetch ok: `170`
- Fetch failed: `2`
- Normalize ok: `None`
- Normalize failed: `None`
- Station features upserted: `None`
- Area features upserted: `None`
- Raw bytes deleted: `691597332`
- Max staging bytes observed: `121981818`
- Final staging bytes: `0`
- Max raw object bytes observed: `8565584`
- Minimum free disk bytes observed: `238055112704`

## By Source

- `gefs_control`: {'skipped_existing': 40, 'max_raw_object_bytes': 8565584, 'max_staging_bytes_after_fetch': 121981818, 'source_issues_touched': 28, 'fetch_ok': 28, 'task_errors': 28, 'fetch_seconds_total': 586.3841779998038, 'normalize_seconds_total': 193.3532591002877, 'db_write_seconds_total': 1.2240954000735655, 'total_seconds_total': 780.961532500165, 'raw_bytes_deleted': 230921841, 'raw_files_deleted': 28}
- `gfs`: {'skipped_existing': 68}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3265018, 'max_staging_bytes_after_fetch': 22772964, 'source_issues_touched': 144, 'fetch_ok': 142, 'task_errors': 142, 'fetch_seconds_total': 843.2668443997973, 'normalize_seconds_total': 381.55897550005466, 'db_write_seconds_total': 26.810209900257178, 'total_seconds_total': 1251.636029800109, 'raw_bytes_deleted': 460675491, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Phase Runtime

- `db_write`: {'count': 172, 'mean_seconds': 0.1629901470949462, 'p50_seconds': 0.17719110002508387, 'p90_seconds': 0.3126329999649897, 'max_seconds': 0.5146743999794126}
- `fetch`: {'count': 172, 'mean_seconds': 8.31192454883489, 'p50_seconds': 5.58797189994948, 'p90_seconds': 17.023575100000016, 'max_seconds': 41.87288320000516}
- `normalize`: {'count': 172, 'mean_seconds': 3.3425129918624554, 'p50_seconds': 2.6523922000196762, 'p90_seconds': 5.98598930001026, 'max_seconds': 17.308917000016663}
- `total`: {'count': 172, 'mean_seconds': 11.817427687792291, 'p50_seconds': 8.682950799993705, 'p90_seconds': 22.525401399994735, 'max_seconds': 48.4819717000355}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
