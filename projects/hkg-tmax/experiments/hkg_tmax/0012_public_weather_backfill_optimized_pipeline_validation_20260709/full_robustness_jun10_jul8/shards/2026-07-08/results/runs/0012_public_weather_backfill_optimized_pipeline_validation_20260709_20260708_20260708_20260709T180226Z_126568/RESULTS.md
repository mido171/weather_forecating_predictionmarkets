# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260708_20260708_20260709T180226Z_126568`
Execution mode: `optimized`
Elapsed seconds: `560.684030400007`
Date range: `2026-07-08` to `2026-07-08`

## Counts

- Source issues touched: `279`
- Fetch ok: `277`
- Fetch failed: `2`
- Normalize ok: `None`
- Normalize failed: `None`
- Station features upserted: `None`
- Area features upserted: `None`
- Raw bytes deleted: `1730497854`
- Max staging bytes observed: `144520087`
- Final staging bytes: `0`
- Max raw object bytes observed: `12117980`
- Minimum free disk bytes observed: `237233336320`

## By Source

- `gefs_control`: {'max_raw_object_bytes': 8670278, 'max_staging_bytes_after_fetch': 102721641, 'source_issues_touched': 68, 'fetch_ok': 68, 'task_errors': 68, 'fetch_seconds_total': 809.7105157998158, 'normalize_seconds_total': 401.74570010008756, 'db_write_seconds_total': 14.730967599956784, 'total_seconds_total': 1226.18718349986, 'raw_bytes_deleted': 523588212, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 12117980, 'max_staging_bytes_after_fetch': 144520087, 'source_issues_touched': 68, 'fetch_ok': 68, 'task_errors': 68, 'fetch_seconds_total': 1102.1768481003237, 'normalize_seconds_total': 449.6653000000515, 'db_write_seconds_total': 43.33051730011357, 'total_seconds_total': 1595.1726654004888, 'raw_bytes_deleted': 753037405, 'raw_files_deleted': 68}
- `himawari9_b13_s0510`: {'skipped_existing': 1, 'max_raw_object_bytes': 3245434, 'max_staging_bytes_after_fetch': 22678241, 'source_issues_touched': 143, 'fetch_ok': 141, 'task_errors': 141, 'fetch_seconds_total': 422.1670794998645, 'normalize_seconds_total': 379.88951680000173, 'db_write_seconds_total': 25.241223699820694, 'total_seconds_total': 827.2978199996869, 'raw_bytes_deleted': 453872237, 'raw_files_deleted': 141, 'fetch_failed': 2}

## Phase Runtime

- `db_write`: {'count': 279, 'mean_seconds': 0.29857601648706467, 'p50_seconds': 0.08865450002485886, 'p90_seconds': 0.41547199996421114, 'max_seconds': 6.178964100021403}
- `fetch`: {'count': 279, 'mean_seconds': 8.365786535483885, 'p50_seconds': 6.195652300026268, 'p90_seconds': 17.762812300003134, 'max_seconds': 24.798945599992294}
- `normalize`: {'count': 279, 'mean_seconds': 4.413263501434196, 'p50_seconds': 3.853531999979168, 'p90_seconds': 7.00064070004737, 'max_seconds': 11.040365000022575}
- `total`: {'count': 279, 'mean_seconds': 13.077626053405146, 'p50_seconds': 9.486059200018644, 'p90_seconds': 25.31834479997633, 'max_seconds': 34.29413369996473}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
