# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260613_20260613_20260709T162630Z_26016`
Execution mode: `optimized`
Elapsed seconds: `1.8201244000229053`
Date range: `2026-06-13` to `2026-06-13`

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
- `himawari9_b13_s0510`: {'skipped_existing': 142, 'source_issues_touched': 2, 'fetch_failed': 2, 'fetch_seconds_total': 1.3020169000374153, 'normalize_seconds_total': 0.0002321999636478722, 'db_write_seconds_total': 0.01882639992982149, 'total_seconds_total': 1.3210754999308847}

## Phase Runtime

- `db_write`: {'count': 2, 'mean_seconds': 0.009413199964910746, 'p50_seconds': 0.003442999965045601, 'p90_seconds': 0.003442999965045601, 'max_seconds': 0.01538339996477589}
- `fetch`: {'count': 2, 'mean_seconds': 0.6510084500187077, 'p50_seconds': 0.5076518999994732, 'p90_seconds': 0.5076518999994732, 'max_seconds': 0.7943650000379421}
- `normalize`: {'count': 2, 'mean_seconds': 0.0001160999818239361, 'p50_seconds': 0.00011049996828660369, 'p90_seconds': 0.00011049996828660369, 'max_seconds': 0.00012169999536126852}
- `total`: {'count': 2, 'mean_seconds': 0.6605377499654423, 'p50_seconds': 0.5231457999325357, 'p90_seconds': 0.5231457999325357, 'max_seconds': 0.797929699998349}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
