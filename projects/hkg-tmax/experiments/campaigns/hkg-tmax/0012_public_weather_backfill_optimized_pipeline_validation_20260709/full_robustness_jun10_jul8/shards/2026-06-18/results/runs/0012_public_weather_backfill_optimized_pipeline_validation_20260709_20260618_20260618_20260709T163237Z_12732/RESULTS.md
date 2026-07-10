# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260618_20260618_20260709T163237Z_12732`
Execution mode: `optimized`
Elapsed seconds: `1.5252454999717884`
Date range: `2026-06-18` to `2026-06-18`

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
- `himawari9_b13_s0510`: {'skipped_existing': 141, 'source_issues_touched': 3, 'fetch_failed': 3, 'fetch_seconds_total': 1.9039598999661393, 'normalize_seconds_total': 0.0003885000478476286, 'db_write_seconds_total': 0.019863499968778342, 'total_seconds_total': 1.9242118999827653}

## Phase Runtime

- `db_write`: {'count': 3, 'mean_seconds': 0.006621166656259447, 'p50_seconds': 0.004930600000079721, 'p90_seconds': 0.004930600000079721, 'max_seconds': 0.010162399965338409}
- `fetch`: {'count': 3, 'mean_seconds': 0.6346532999887131, 'p50_seconds': 0.5837410999811254, 'p90_seconds': 0.5837410999811254, 'max_seconds': 0.7797192999860272}
- `normalize`: {'count': 3, 'mean_seconds': 0.00012950001594920954, 'p50_seconds': 0.00012550002429634333, 'p90_seconds': 0.00012550002429634333, 'max_seconds': 0.00014050002209842205}
- `total`: {'count': 3, 'mean_seconds': 0.6414039666609218, 'p50_seconds': 0.5888122000033036, 'p90_seconds': 0.5888122000033036, 'max_seconds': 0.7900071999756619}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
