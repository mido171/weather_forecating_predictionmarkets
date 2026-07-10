# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260616_20260616_20260709T182231Z_22908`
Execution mode: `optimized`
Elapsed seconds: `1.5479571999749169`
Date range: `2026-06-16` to `2026-06-16`

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
- `himawari9_b13_s0510`: {'skipped_existing': 142, 'source_issues_touched': 2, 'fetch_failed': 2, 'fetch_seconds_total': 1.560419199988246, 'normalize_seconds_total': 0.00022390001686289907, 'db_write_seconds_total': 0.018221899983473122, 'total_seconds_total': 1.578864999988582}

## Phase Runtime

- `db_write`: {'count': 2, 'mean_seconds': 0.009110949991736561, 'p50_seconds': 0.005940200004260987, 'p90_seconds': 0.005940200004260987, 'max_seconds': 0.012281699979212135}
- `fetch`: {'count': 2, 'mean_seconds': 0.780209599994123, 'p50_seconds': 0.6324164000106975, 'p90_seconds': 0.6324164000106975, 'max_seconds': 0.9280027999775484}
- `normalize`: {'count': 2, 'mean_seconds': 0.00011195000843144953, 'p50_seconds': 0.00011080002877861261, 'p90_seconds': 0.00011080002877861261, 'max_seconds': 0.00011309998808428645}
- `total`: {'count': 2, 'mean_seconds': 0.789432499994291, 'p50_seconds': 0.6384697000030428, 'p90_seconds': 0.6384697000030428, 'max_seconds': 0.9403952999855392}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
