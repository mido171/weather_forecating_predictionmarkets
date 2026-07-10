# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260612_20260612_20260709T162629Z_126316`
Execution mode: `optimized`
Elapsed seconds: `1.542582800029777`
Date range: `2026-06-12` to `2026-06-12`

## Counts

- Source issues touched: `2`
- Fetch ok: `None`
- Fetch failed: `2`
- Normalize ok: `None`
- Normalize failed: `None`
- Station features upserted: `None`
- Area features upserted: `None`
- Raw bytes deleted: `None`
- Max staging bytes observed: `None`
- Final staging bytes: `0`
- Max raw object bytes observed: `None`
- Minimum free disk bytes observed: `None`

## By Source

- `gefs_control`: {'skipped_existing': 68}
- `gfs`: {'skipped_existing': 68}
- `himawari9_b13_s0510`: {'skipped_existing': 142, 'source_issues_touched': 2, 'fetch_failed': 2, 'fetch_seconds_total': 1.5794212000328116, 'normalize_seconds_total': 0.00023290002718567848, 'db_write_seconds_total': 0.14305210002930835, 'total_seconds_total': 1.7227062000893056}

## Phase Runtime

- `db_write`: {'count': 2, 'mean_seconds': 0.07152605001465417, 'p50_seconds': 0.027468600019346923, 'p90_seconds': 0.027468600019346923, 'max_seconds': 0.11558350000996143}
- `fetch`: {'count': 2, 'mean_seconds': 0.7897106000164058, 'p50_seconds': 0.7841007000533864, 'p90_seconds': 0.7841007000533864, 'max_seconds': 0.7953204999794252}
- `normalize`: {'count': 2, 'mean_seconds': 0.00011645001359283924, 'p50_seconds': 0.0001162000116892159, 'p90_seconds': 0.0001162000116892159, 'max_seconds': 0.00011670001549646258}
- `total`: {'count': 2, 'mean_seconds': 0.8613531000446528, 'p50_seconds': 0.8229058000142686, 'p90_seconds': 0.8229058000142686, 'max_seconds': 0.899800400075037}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
