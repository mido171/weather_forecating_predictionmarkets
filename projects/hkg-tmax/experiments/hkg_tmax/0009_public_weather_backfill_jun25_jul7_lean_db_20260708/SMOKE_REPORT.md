# Smoke Report

## What Was Verified

- Old raw payload directories were removed from experiments `0005`, `0006`, and `0007`.
- Dry-run inventory for `2026-06-25` through `2026-07-07` produced:
  - `1,768` model objects.
  - `1,872` Himawari B13/S0510 scans.
  - About `1,560` radar frames.
- Live GFS NOMADS path worked for a recent issue.
- Live GFS and GEFS control S3 `.idx` byte-range fallback worked for aged-out `2026-06-25` issues.
- Live Himawari B13/S0510 HSD fetch, decode, HKO pixel/window extraction, and raw deletion worked.
- Live ENVF radar manifest/image proxy extraction worked without retaining raw image bytes.
- Combined bounded smoke across GFS, GEFS control, Himawari, and radar completed successfully.

## Latest DB Audit

- `weather_backfill.source_issue` rows touched by this experiment: `11`.
- `weather_backfill.station_feature` rows joined to those issues: `422`.
- `weather_backfill.area_feature` rows joined to those issues: `1,284`.
- Null `available_at_utc` or `availability_proxy_utc`: `0`.
- Duplicate source issue keys: `0`.
- 0009 staging raw files after smoke: `0`.
- Old smoke raw directories still present: `0`.

## Runtime Finding

The current first draft is deliberately serial and low-disk. It is safe, but not fast enough for a full `2026-06-25` through `2026-07-07` all-source run inside a short interactive turn. Measured per-item timings imply a many-hour full run unless task-level parallel fetch/normalize workers are added.

## Current Acceptance

Status: `SCRIPT_AND_SMOKE_PASS`.

This is an acquisition/persistence result, not a predictive model result and not a promotion candidate.
