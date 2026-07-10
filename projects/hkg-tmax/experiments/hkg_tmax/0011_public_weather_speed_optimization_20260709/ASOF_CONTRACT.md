# As-Of Contract

This experiment measures acquisition speed only. It does not change the leakage contract used by the production DB backfill.

The tested source objects retain the same source metadata fields as the lean DB backfill:
- `issued_at_utc` for GFS/GEFS cycle runs.
- `valid_at_utc` for forecast lead validity.
- `observed_at_utc` for Himawari scans.
- `availability_proxy_utc` for leakage-safe downstream filtering.
- source URL/object identity and fetch status.

Any future production optimization must preserve the rule:

```sql
available_at_utc <= cutoff_utc
```

or its current schema equivalent, by joining every feature row back to its source issue availability timestamp.
