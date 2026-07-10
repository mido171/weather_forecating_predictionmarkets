# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260703_20260703_20260709T172701Z_121596`
Execution mode: `optimized`
Elapsed seconds: `628.3599409000017`
Date range: `2026-07-03` to `2026-07-03`

## Counts

- Source issues touched: `280`
- Fetch ok: `278`
- Fetch failed: `2`
- Normalize ok: `None`
- Normalize failed: `None`
- Station features upserted: `None`
- Area features upserted: `None`
- Raw bytes deleted: `1761125584`
- Max staging bytes observed: `146381546`
- Final staging bytes: `0`
- Max raw object bytes observed: `12125582`
- Minimum free disk bytes observed: `237625102336`

## By Source

- `gefs_control`: {'max_raw_object_bytes': 8825231, 'max_staging_bytes_after_fetch': 99140258, 'source_issues_touched': 68, 'fetch_ok': 68, 'task_errors': 68, 'fetch_seconds_total': 1108.810659599665, 'normalize_seconds_total': 411.8446105995681, 'db_write_seconds_total': 15.979282999993302, 'total_seconds_total': 1536.6345531992265, 'raw_bytes_deleted': 551331510, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 12125582, 'max_staging_bytes_after_fetch': 146381546, 'source_issues_touched': 68, 'fetch_ok': 68, 'task_errors': 68, 'fetch_seconds_total': 1424.4219418001012, 'normalize_seconds_total': 496.3950878996984, 'db_write_seconds_total': 23.063620700035244, 'total_seconds_total': 1943.8806503998348, 'raw_bytes_deleted': 756505623, 'raw_files_deleted': 68}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3230748, 'max_staging_bytes_after_fetch': 22548768, 'source_issues_touched': 144, 'fetch_ok': 142, 'task_errors': 142, 'fetch_seconds_total': 626.5756194003625, 'normalize_seconds_total': 392.6883141004946, 'db_write_seconds_total': 27.904935099359136, 'total_seconds_total': 1047.1688686002162, 'raw_bytes_deleted': 453288451, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Phase Runtime

- `db_write`: {'count': 280, 'mean_seconds': 0.23909942428352743, 'p50_seconds': 0.055785899981856346, 'p90_seconds': 0.5166130000143312, 'max_seconds': 3.397531100024935}
- `fetch`: {'count': 280, 'mean_seconds': 11.28502936000046, 'p50_seconds': 7.707538199960254, 'p90_seconds': 23.955970699957106, 'max_seconds': 47.41143190005096}
- `normalize`: {'count': 280, 'mean_seconds': 4.6461714735705755, 'p50_seconds': 4.318333700008225, 'p90_seconds': 7.4664741000160575, 'max_seconds': 12.645829799992498}
- `total`: {'count': 280, 'mean_seconds': 16.170300257854564, 'p50_seconds': 12.65878709993558, 'p90_seconds': 31.256340299907606, 'max_seconds': 53.643222800048534}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
