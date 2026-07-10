# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260614_20260614_20260709T162836Z_123780`
Execution mode: `optimized`
Elapsed seconds: `1.3221259000129066`
Date range: `2026-06-14` to `2026-06-14`

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
- `himawari9_b13_s0510`: {'skipped_existing': 142, 'source_issues_touched': 2, 'fetch_failed': 2, 'fetch_seconds_total': 1.279842699994333, 'normalize_seconds_total': 0.00025210005696862936, 'db_write_seconds_total': 0.0183752000448294, 'total_seconds_total': 1.2984700000961311}

## Phase Runtime

- `db_write`: {'count': 2, 'mean_seconds': 0.0091876000224147, 'p50_seconds': 0.004626700014341623, 'p90_seconds': 0.004626700014341623, 'max_seconds': 0.013748500030487776}
- `fetch`: {'count': 2, 'mean_seconds': 0.6399213499971665, 'p50_seconds': 0.5449928999878466, 'p90_seconds': 0.5449928999878466, 'max_seconds': 0.7348498000064865}
- `normalize`: {'count': 2, 'mean_seconds': 0.00012605002848431468, 'p50_seconds': 0.00012590002734214067, 'p90_seconds': 0.00012590002734214067, 'max_seconds': 0.00012620002962648869}
- `total`: {'count': 2, 'mean_seconds': 0.6492350000480656, 'p50_seconds': 0.5588673000456765, 'p90_seconds': 0.5588673000456765, 'max_seconds': 0.7396027000504546}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
