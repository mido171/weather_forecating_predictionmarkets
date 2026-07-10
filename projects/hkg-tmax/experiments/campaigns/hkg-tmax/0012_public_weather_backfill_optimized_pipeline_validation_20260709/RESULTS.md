# Results

Final state: `accepted_with_notes`.

This experiment validates the optimized DB-backed GFS, GEFS control, and Himawari B13/S0510
pipeline for leakage-safe public-weather backfills. The full robustness run completed, then a
targeted GFS retry closed the six transient model fetch misses.

## Validation Runs

| Run | Range | Result | Runtime | Key evidence |
| --- | --- | --- | ---: | --- |
| Dry-run inventory | 2026-06-21 | Passed | 0.0 s | Expected `136` model objects and `144` Himawari scans. |
| One-day optimized smoke | 2026-06-21 | Passed with source 404s | 780.9 s | `278/280` successful objects; `2` Himawari 404s; `15,838` station features; `28,550` area features; final staging `0`. |
| Idempotency rerun | 2026-06-21 | Passed | 1.5 s | `278` completed objects skipped; only the two recorded Himawari 404s touched. |
| Seven-day rehearsal | 2026-06-15..2026-06-21 | Passed after targeted GFS retry | 2,894.2 s | `1,940` successful objects; `20` fetch failures during first pass; `17` Himawari 404s; `3` transient GFS disconnects later forced-refetched successfully. |
| Full robustness run | 2026-06-10..2026-07-08 | Passed after targeted GFS retry | 6,637.3 s plus 125.9 s retry | DB coverage now complete for all model issues; only Himawari source 404s remain. |

## Full-Window DB Audit

Expected issue inventory for 2026-06-10 through 2026-07-08:

| Source | Expected issues | Final OK issues | Final source/error issues | Feature-bearing issue keys |
| --- | ---: | ---: | ---: | ---: |
| GFS | 1,972 | 1,972 | 0 | 1,972 |
| GEFS control | 1,972 | 1,972 | 0 | 1,972 |
| Himawari B13/S0510 | 4,176 | 4,111 | 65 | 4,111 |
| Total | 8,120 | 8,055 | 65 | 8,055 |

Feature rows currently persisted for the full window:

| Source | Station feature rows | Area feature rows |
| --- | ---: | ---: |
| GFS | 34,916 | 402,636 |
| GEFS control | 33,175 | 380,003 |
| Himawari B13/S0510 | 390,545 | 45,221 |

Integrity checks:

- Source issue rows present: `8,120/8,120`.
- Model issue success rate after retry: `3,944/3,944`.
- Himawari missing count: `65`, all `HTTPError: HTTP Error 404: Not Found`.
- `available_at_utc` nulls: `0` in source issues, station features, and area features.
- Feature-bearing successful issues without features: `0`.
- Duplicate station-feature primary-key groups: `0`.
- Duplicate area-feature primary-key groups: `0`.
- Raw-like files left in the experiment folder: `0`.
- Repo-local `_weather_backfill_staging` tree: removed after auditing it contained `0` files.

## Runtime And Disk

Seven-day rehearsal day-worker runtimes:

- p50 day runtime: `676.0 s` (`11.3 min`).
- p90 day runtime: `714.1 s` (`11.9 min`).
- min/max day runtime: `648.0 s` / `718.5 s`.
- Wall-clock throughput with two day workers: `48.2 min / 7 days = 6.9 min per day-equivalent`.

Corrected full robustness run:

- Wall-clock runtime: `6,637.3 s` (`110.6 min`) plus `125.9 s` targeted GFS retry.
- Successful raw files deleted in full run: `4,263`; retry deleted another `6`.
- Raw bytes deleted in full run: `25,627,868,094`; retry deleted another `68,556,803`.
- Full-run peak aggregate staging: `241,375,243 bytes` (`230.2 MiB`), below the `1 GB` cap.
- Full-run peak single-worker staging: `156,263,623 bytes` (`149.0 MiB`).
- Targeted retry peak staging: `23,659,243 bytes` (`22.6 MiB`).
- Final staging after all runs: `0` bytes; staging root removed.

## Source-Side Failures

The final remaining failures are Himawari 404s, not parser or pipeline errors. They are recorded
as source issues with non-null `available_at_utc`, `status='error'`, and
`normalized_status='not_run'`.

The six full-run GFS misses were transient `RemoteDisconnected` errors. A targeted optimized
retry with `--skip-existing-complete --max-attempts 7` fetched and persisted all six.

## Limitations

- CPU telemetry was requested, but `psutil` was not available in the current venv, so CPU
  mean/max are null. Disk and staging telemetry are valid, and worker counts were bounded.
- The current optimized stack does not reach a true individual-day runtime of `3-5 min/day`.
  It reaches about `11 min` per full fresh day, or about `7 min/day-equivalent` with two day
  workers.

## Detailed Handoff And Capacity

The complete implementation, operations, and database-capacity documentation is indexed under
[`documentation/README.md`](documentation/README.md).

Using live Postgres relation sizes and the 29-day feature population, the projected retained
database footprint for `2017-01-01..2026-07-10` is `121.4 GB` decimal (`113.0 GiB`), with a
conservative retained-size range of `100-150 GB` decimal.
