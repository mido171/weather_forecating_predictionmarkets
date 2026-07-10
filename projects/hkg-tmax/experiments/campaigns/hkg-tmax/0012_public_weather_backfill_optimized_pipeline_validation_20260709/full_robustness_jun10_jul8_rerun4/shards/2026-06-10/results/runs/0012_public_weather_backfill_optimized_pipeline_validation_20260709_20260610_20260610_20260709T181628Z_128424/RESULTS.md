# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260610_20260610_20260709T181628Z_128424`
Execution mode: `optimized`
Elapsed seconds: `1.3333893999806605`
Date range: `2026-06-10` to `2026-06-10`

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
- `himawari9_b13_s0510`: {'skipped_existing': 142, 'source_issues_touched': 2, 'fetch_failed': 2, 'fetch_seconds_total': 1.5398540000314824, 'normalize_seconds_total': 0.0002429999876767397, 'db_write_seconds_total': 0.016771299997344613, 'total_seconds_total': 1.5568683000165038}

## Phase Runtime

- `db_write`: {'count': 2, 'mean_seconds': 0.008385649998672307, 'p50_seconds': 0.0046820000279694796, 'p90_seconds': 0.0046820000279694796, 'max_seconds': 0.012089299969375134}
- `fetch`: {'count': 2, 'mean_seconds': 0.7699270000157412, 'p50_seconds': 0.7316617000033148, 'p90_seconds': 0.7316617000033148, 'max_seconds': 0.8081923000281677}
- `normalize`: {'count': 2, 'mean_seconds': 0.00012149999383836985, 'p50_seconds': 0.00011610001092776656, 'p90_seconds': 0.00011610001092776656, 'max_seconds': 0.00012689997674897313}
- `total`: {'count': 2, 'mean_seconds': 0.7784341500082519, 'p50_seconds': 0.736459800042212, 'p90_seconds': 0.736459800042212, 'max_seconds': 0.8204084999742918}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
