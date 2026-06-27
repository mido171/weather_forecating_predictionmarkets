# HKG-T24-001 GribStream Source Scope Audit

## Status

PASS

## Required Filter

`forecast_wide` rows are joined to `raw_response_object`, filtered to `full_tactical_backfill_ok_tmax`, constrained to `H24N`, guarded by `run_time_utc + interval '6 hours' <= formal_cutoff_utc`, and daily Tmax blocked datasets are excluded from safe rows.

## Dataset Counts

- `aifsenfo`: scoped=72270, safe=72270, excluded=0, days=355
- `aifsoper`: scoped=28884, safe=23100, excluded=5784, days=483
- `aigefssfc`: scoped=46252, safe=0, excluded=46252, days=373
- `aigfspres`: scoped=3660, safe=0, excluded=3660, days=63
- `aigfssfc`: scoped=3660, safe=2928, excluded=732, days=63
- `cwawrf15`: scoped=180, safe=144, excluded=36, days=4
- `fourcastnetgfs`: scoped=37824, safe=30252, excluded=7572, days=648
- `gefsatmos`: scoped=516891, safe=516891, excluded=0, days=2085
- `gefsatmosmean`: scoped=200436, safe=200436, excluded=0, days=2088
- `gfs`: scoped=575004, safe=552000, excluded=23004, days=1919
- `graphcast`: scoped=44220, safe=35376, excluded=8844, days=741
- `ifsenfo`: scoped=343616, safe=343616, excluded=0, days=843
- `ifsoper`: scoped=91260, safe=81120, excluded=10140, days=846

## Publication Buffer

The 6-hour buffer is a conservative project guardrail, not a confirmed GribStream provider availability SLA.
