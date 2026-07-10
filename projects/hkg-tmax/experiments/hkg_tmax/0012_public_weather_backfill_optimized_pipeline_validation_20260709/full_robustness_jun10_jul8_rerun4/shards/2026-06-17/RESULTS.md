# Results

State: `complete_with_failures`
Run id: `0012_public_weather_backfill_optimized_pipeline_validation_20260709_20260617_20260617_20260709T182231Z_128904`
Execution mode: `optimized`
Elapsed seconds: `1.4159806999959983`
Date range: `2026-06-17` to `2026-06-17`

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
- `himawari9_b13_s0510`: {'skipped_existing': 142, 'source_issues_touched': 2, 'fetch_failed': 2, 'fetch_seconds_total': 1.4237131999689154, 'normalize_seconds_total': 0.0002397999633103609, 'db_write_seconds_total': 0.01481170003535226, 'total_seconds_total': 1.438764699967578}

## Phase Runtime

- `db_write`: {'count': 2, 'mean_seconds': 0.00740585001767613, 'p50_seconds': 0.0046814000234007835, 'p90_seconds': 0.0046814000234007835, 'max_seconds': 0.010130300011951476}
- `fetch`: {'count': 2, 'mean_seconds': 0.7118565999844577, 'p50_seconds': 0.5556950999889523, 'p90_seconds': 0.5556950999889523, 'max_seconds': 0.8680180999799632}
- `normalize`: {'count': 2, 'mean_seconds': 0.00011989998165518045, 'p50_seconds': 0.00011809996794909239, 'p90_seconds': 0.00011809996794909239, 'max_seconds': 0.00012169999536126852}
- `total`: {'count': 2, 'mean_seconds': 0.719382349983789, 'p50_seconds': 0.5659434999688528, 'p90_seconds': 0.5659434999688528, 'max_seconds': 0.8728211999987252}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
