# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260621_20260621_20260709T182634Z_123852`
Execution mode: `optimized`
Elapsed seconds: `1.3478610999882221`
Date range: `2026-06-21` to `2026-06-21`

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
- `himawari9_b13_s0510`: {'skipped_existing': 142, 'source_issues_touched': 2, 'fetch_failed': 2, 'fetch_seconds_total': 1.4561236999579705, 'normalize_seconds_total': 0.000255499966442585, 'db_write_seconds_total': 0.07741550001082942, 'total_seconds_total': 1.5337946999352425}

## Phase Runtime

- `db_write`: {'count': 2, 'mean_seconds': 0.03870775000541471, 'p50_seconds': 0.00448629999300465, 'p90_seconds': 0.00448629999300465, 'max_seconds': 0.07292920001782477}
- `fetch`: {'count': 2, 'mean_seconds': 0.7280618499789853, 'p50_seconds': 0.693191799975466, 'p90_seconds': 0.693191799975466, 'max_seconds': 0.7629318999825045}
- `normalize`: {'count': 2, 'mean_seconds': 0.0001277499832212925, 'p50_seconds': 0.00012429995695129037, 'p90_seconds': 0.00012429995695129037, 'max_seconds': 0.00013120000949129462}
- `total`: {'count': 2, 'mean_seconds': 0.7668973499676213, 'p50_seconds': 0.6978023999254219, 'p90_seconds': 0.6978023999254219, 'max_seconds': 0.8359923000098206}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
