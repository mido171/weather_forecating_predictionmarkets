# Results

State: `complete_with_failures`
Run id: `0010_public_weather_backfill_jun10_jul8_lean_db_test_20260709_20260709T083118Z`
Elapsed seconds: `0.4417532999650575`
Date range: `2026-06-10` to `2026-06-10`

## Counts

- Source issues touched: `1`
- Fetch ok: `None`
- Fetch failed: `1`
- Normalize ok: `None`
- Normalize failed: `None`
- Station features upserted: `None`
- Area features upserted: `None`
- Raw bytes deleted: `None`
- Max staging bytes observed: `0`
- Max raw object bytes observed: `None`
- Minimum free disk bytes observed: `254469783552`

## By Source

- `gfs`: {'source_issues_touched': 1, 'fetch_failed': 1}

## Notes

- Raw payloads are intentionally not retained.
- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.
