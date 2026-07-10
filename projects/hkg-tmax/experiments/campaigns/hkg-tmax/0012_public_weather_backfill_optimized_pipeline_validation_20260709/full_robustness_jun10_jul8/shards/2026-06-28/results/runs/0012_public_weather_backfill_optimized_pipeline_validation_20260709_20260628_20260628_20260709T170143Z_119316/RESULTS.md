# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260628_20260628_20260709T170143Z_119316`
Execution mode: `optimized`
Elapsed seconds: `645.2636944999686`
Date range: `2026-06-28` to `2026-06-28`

## Counts

- Source issues touched: `280`
- Fetch ok: `278`
- Fetch failed: `2`
- Normalize ok: `None`
- Normalize failed: `None`
- Station features upserted: `None`
- Area features upserted: `None`
- Raw bytes deleted: `1833112857`
- Max staging bytes observed: `144905942`
- Final staging bytes: `0`
- Max raw object bytes observed: `12371250`
- Minimum free disk bytes observed: `237692653568`

## By Source

- `gefs_control`: {'max_raw_object_bytes': 9618543, 'max_staging_bytes_after_fetch': 123320439, 'source_issues_touched': 68, 'fetch_ok': 68, 'task_errors': 68, 'fetch_seconds_total': 1147.930813500483, 'normalize_seconds_total': 438.44068820000393, 'db_write_seconds_total': 20.799236899940297, 'total_seconds_total': 1607.1707386004273, 'raw_bytes_deleted': 573760518, 'raw_files_deleted': 68}
- `gfs`: {'max_raw_object_bytes': 12371250, 'max_staging_bytes_after_fetch': 144905942, 'source_issues_touched': 68, 'fetch_ok': 68, 'task_errors': 68, 'fetch_seconds_total': 1571.0904766999884, 'normalize_seconds_total': 482.7214468998136, 'db_write_seconds_total': 22.63061010016827, 'total_seconds_total': 2076.44253369997, 'raw_bytes_deleted': 808753216, 'raw_files_deleted': 68}
- `himawari9_b13_s0510`: {'max_raw_object_bytes': 3239798, 'max_staging_bytes_after_fetch': 25874878, 'source_issues_touched': 144, 'fetch_ok': 142, 'task_errors': 142, 'fetch_seconds_total': 739.3064601004007, 'normalize_seconds_total': 389.93031940050423, 'db_write_seconds_total': 27.301511800149456, 'total_seconds_total': 1156.5382913010544, 'raw_bytes_deleted': 450599123, 'raw_files_deleted': 142, 'fetch_failed': 2}

## Phase Runtime

- `db_write`: {'count': 280, 'mean_seconds': 0.2526119957152072, 'p50_seconds': 0.09720310004195198, 'p90_seconds': 0.4955595000064932, 'max_seconds': 4.7396853000391275}
- `fetch`: {'count': 280, 'mean_seconds': 12.351170536788828, 'p50_seconds': 10.013057699950878, 'p90_seconds': 26.042149599990807, 'max_seconds': 42.073812999995425}
- `normalize`: {'count': 280, 'mean_seconds': 4.682473051786864, 'p50_seconds': 4.359043299977202, 'p90_seconds': 7.518074400024489, 'max_seconds': 10.993181099998765}
- `total`: {'count': 280, 'mean_seconds': 17.2862555842909, 'p50_seconds': 16.364715000032447, 'p90_seconds': 33.941419999988284, 'max_seconds': 48.48908599995775}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
