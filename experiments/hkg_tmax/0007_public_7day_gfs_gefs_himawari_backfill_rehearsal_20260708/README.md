# Public 7-Day GFS/GEFS/Himawari Backfill Rehearsal

Generated: `2026-07-08T12:30:00.225957Z`

Date range: `2026-07-01` through `2026-07-07` UTC.

## Headline

| Metric | Value |
|---|---:|
| fetch items requested | 1,960 |
| fetch items ok | 1,671 |
| invalid/error fetch items | 289 |
| model station normalized rows | 680 |
| model bbox variable rows | 9,856 |
| Himawari scan normalized rows | 991 |
| raw size on disk GB | 9.626 |
| normalized size on disk MB | 9.00 |
| initial download wall seconds | 517.7 |
| cached rerun validation/download wall seconds | 28.2 |
| model idx refresh wall seconds | 153.6 |
| model normalize wall seconds | 793.2 |
| Himawari normalize wall seconds | 287.6 |

## Key Finding

GFS and Himawari were broadly fetchable for the full seven-day window. GEFS control via live NOMADS was only fetchable for the newer part of the seven-day window in this run; older GEFS requests are recorded explicitly as errors/invalid payloads in `normalized/fetch_manifest.csv`.

Model normalization must use process isolation for cfgrib/eccodes. Threaded model normalization produced decoder instability. Himawari normalization parallelized cleanly.

## Leakage Contract

GFS/GEFS rows carry `issued_at_utc`, `valid_at_utc`, and `availability_proxy_utc = issued_at_utc + 6h`.

Himawari rows carry `observed_at_utc`, `file_creation_utc`, and `availability_proxy_utc = max(file_creation_utc, observed_at_utc + 30m)`.

## Main Files

| File | Purpose |
|---|---|
| `normalized/sanity_report.json` | Coverage, attribute counts, leakage fields, timing, bytes. |
| `normalized/fetch_manifest.csv` | One row per requested raw object with URL, status, bytes, hash, timestamps. |
| `normalized/model_idx_catalog.csv` | Full-product index counts for every requested model source/cycle/lead. |
| `normalized/model_cycle_lead_station_features.csv` | HKO nearest-grid model features by source/cycle/lead. |
| `normalized/model_cycle_lead_bbox_summary_features.csv` | HKG bbox min/mean/median/max/std by variable. |
| `normalized/himawari_b13_s0510_scan_features.csv` | One normalized row per B13 HKG-segment scan. |
| `normalized/backfill_size_estimates.json` | Estimated raw/normalized size for 2015+ and 2017+ backfills. |
