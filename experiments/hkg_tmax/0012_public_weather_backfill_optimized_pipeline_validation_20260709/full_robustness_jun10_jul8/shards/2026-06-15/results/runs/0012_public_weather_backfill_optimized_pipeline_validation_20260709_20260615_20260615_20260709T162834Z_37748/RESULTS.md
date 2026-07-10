# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260615_20260615_20260709T162834Z_37748`
Execution mode: `optimized`
Elapsed seconds: `1.5455249999649823`
Date range: `2026-06-15` to `2026-06-15`

## Counts

- Source issues touched: `3`
- Fetch ok: `None`
- Fetch failed: `3`
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
- `himawari9_b13_s0510`: {'skipped_existing': 141, 'source_issues_touched': 3, 'fetch_failed': 3, 'fetch_seconds_total': 1.9190135999233462, 'normalize_seconds_total': 0.0003760000690817833, 'db_write_seconds_total': 0.025616300059482455, 'total_seconds_total': 1.9450059000519104}

## Phase Runtime

- `db_write`: {'count': 3, 'mean_seconds': 0.008538766686494151, 'p50_seconds': 0.006393600022420287, 'p90_seconds': 0.006393600022420287, 'max_seconds': 0.014473600022029132}
- `fetch`: {'count': 3, 'mean_seconds': 0.6396711999744488, 'p50_seconds': 0.6069649999844842, 'p90_seconds': 0.6069649999844842, 'max_seconds': 0.7703921999782324}
- `normalize`: {'count': 3, 'mean_seconds': 0.00012533335636059442, 'p50_seconds': 0.0001273000380024314, 'p90_seconds': 0.0001273000380024314, 'max_seconds': 0.00012940005399286747}
- `total`: {'count': 3, 'mean_seconds': 0.6483353000173034, 'p50_seconds': 0.6134880000608973, 'p90_seconds': 0.6134880000608973, 'max_seconds': 0.784993100038264}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
