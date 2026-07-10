# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260620_20260620_20260709T182634Z_127684`
Execution mode: `optimized`
Elapsed seconds: `1.3801690999534912`
Date range: `2026-06-20` to `2026-06-20`

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
- `himawari9_b13_s0510`: {'skipped_existing': 142, 'source_issues_touched': 2, 'fetch_failed': 2, 'fetch_seconds_total': 1.5188477999763563, 'normalize_seconds_total': 0.00025069998810067773, 'db_write_seconds_total': 0.019737599999643862, 'total_seconds_total': 1.5388360999641009}

## Phase Runtime

- `db_write`: {'count': 2, 'mean_seconds': 0.009868799999821931, 'p50_seconds': 0.004800900002010167, 'p90_seconds': 0.004800900002010167, 'max_seconds': 0.014936699997633696}
- `fetch`: {'count': 2, 'mean_seconds': 0.7594238999881782, 'p50_seconds': 0.7126913999672979, 'p90_seconds': 0.7126913999672979, 'max_seconds': 0.8061564000090584}
- `normalize`: {'count': 2, 'mean_seconds': 0.00012534999405033886, 'p50_seconds': 0.00012340000830590725, 'p90_seconds': 0.00012340000830590725, 'max_seconds': 0.00012729997979477048}
- `total`: {'count': 2, 'mean_seconds': 0.7694180499820504, 'p50_seconds': 0.717615699977614, 'p90_seconds': 0.717615699977614, 'max_seconds': 0.8212203999864869}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
