# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260702_20260702_20260709T172600Z_85252`
Execution mode: `optimized`
Elapsed seconds: `639.3290088999784`
Date range: `2026-07-02` to `2026-07-02`

## Counts

- Source issues touched: `280`
- Fetch ok: `276`
- Fetch failed: `4`
- Normalize ok: `None`
- Normalize failed: `None`
- Station features upserted: `None`
- Area features upserted: `None`
- Raw bytes deleted: `1798056953`
- Max staging bytes observed: `156133797`
- Final staging bytes: `0`
- Max raw object bytes observed: `12209414`
- Minimum free disk bytes observed: `237640998912`

## By Source

- `gefs_control`: {'max_raw_object_bytes': 8929176, 'max_staging_bytes_after_fetch': 107021604, 'source_issues_touched': 68, 'fetch_ok': 68, 'task_errors': 68, 'fetch_seconds_total': 1082.3355558001786, 'normalize_seconds_total': 421.2322466003825, 'db_write_seconds_total': 23.038524100149516, 'total_seconds_total': 1526.6063265007106, 'raw_bytes_deleted': 559614185, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 12209414, 'max_staging_bytes_after_fetch': 156133797, 'source_issues_touched': 68, 'fetch_ok': 68, 'task_errors': 68, 'fetch_seconds_total': 1484.9400396997808, 'normalize_seconds_total': 489.7350417000707, 'db_write_seconds_total': 30.58621100021992, 'total_seconds_total': 2005.2612924000714, 'raw_bytes_deleted': 795549489, 'raw_files_deleted': 68}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3206180, 'max_staging_bytes_after_fetch': 22436962, 'source_issues_touched': 144, 'fetch_ok': 140, 'task_errors': 140, 'fetch_seconds_total': 672.8636821001419, 'normalize_seconds_total': 399.0375057000783, 'db_write_seconds_total': 27.339449700375553, 'total_seconds_total': 1099.2406375005958, 'raw_bytes_deleted': 442893279, 'raw_files_deleted': 140, 'fetch_failed': 4}

## Phase Runtime

- `db_write`: {'count': 280, 'mean_seconds': 0.2891578028598035, 'p50_seconds': 0.08592520002275705, 'p90_seconds': 0.4432461999822408, 'max_seconds': 8.344442200032063}
- `fetch`: {'count': 280, 'mean_seconds': 11.571925991428932, 'p50_seconds': 8.59780079999473, 'p90_seconds': 24.642252999998163, 'max_seconds': 41.03945159999421}
- `normalize`: {'count': 280, 'mean_seconds': 4.678588550001899, 'p50_seconds': 4.305103200022131, 'p90_seconds': 7.431242900027428, 'max_seconds': 13.145679700013716}
- `total`: {'count': 280, 'mean_seconds': 16.539672344290636, 'p50_seconds': 13.664821600017603, 'p90_seconds': 31.95437980000861, 'max_seconds': 47.9629896999686}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
