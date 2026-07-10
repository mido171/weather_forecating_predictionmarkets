# 0012 Optimized public-weather backfill validation

Status: `accepted_with_notes`. Significance: 89/100.

## Acceptance question

Validate a production-credible, leakage-safe, low-disk path for GFS, GEFS
control, and Himawari B13/S0510: fetch, normalize, persist source issues and
features, preserve availability clocks, recover transient failures, and delete
raw payloads after commit.

Radar was excluded from the accepted 0012 scope.

## Architecture and invariant

The optimized path uses bounded object/range fetchers, bounded model
normalization processes, bounded Himawari workers, serialized per-process DB
writes, idempotent issue keys, and a staging cap. Raw deletion occurs only
after a successful DB commit or after failure metadata is committed.

Availability values are conservative public-archive proxies; they are not
claims of exact provider publication seconds.

## Validation ladder

| Run | Window | Result | Runtime |
|---|---|---|---:|
| Dry inventory | 2026-06-21 | 136 model + 144 Himawari tasks | 0.0 s |
| One-day optimized smoke | 2026-06-21 | 278/280; two Himawari 404s | 780.9 s |
| Idempotency rerun | 2026-06-21 | 278 completed issues skipped | 1.5 s |
| Seven-day rehearsal | 2026-06-15..21 | Passed after targeted GFS recovery | 2,894.2 s |
| Full robustness | 2026-06-10..07-08 | Passed after six-key GFS retry | 6,637.3 s + 125.9 s |

## Final DB audit

| Source | Expected issues | OK | Source errors | Station rows | Area rows |
|---|---:|---:|---:|---:|---:|
| GFS | 1,972 | 1,972 | 0 | 34,916 | 402,636 |
| GEFS control | 1,972 | 1,972 | 0 | 33,175 | 380,003 |
| Himawari B13/S0510 | 4,176 | 4,111 | 65 | 390,545 | 45,221 |
| Total | 8,120 | 8,055 | 65 | 458,636 | 827,860 |

All 3,944 model issues were complete after retry. The 65 remaining failures
were recorded Himawari source-side HTTP 404s. Availability nulls, missing
features for successful issues, and duplicate natural-key groups were all zero.

## Runtime and disk

- Fresh-day p50/p90: 11.3/11.9 minutes.
- Two-day-worker throughput: about 6.9 minutes per day-equivalent.
- Full-run peak aggregate staging: 230.2 MiB, below the 1 GiB run cap.
- Final staging: zero; 4,263 successful raw files plus six retry files were
  deleted after persistence.
- CPU percentages were unavailable because `psutil` was absent.

## Capacity result

Live measurements on 2026-07-10 showed a 0.989 GiB `weather_backfill` table
footprint. Projecting the current schema from 2017-01-01 through 2026-07-10
gives:

| Item | Projection |
|---|---:|
| Station feature rows | 55,004,690 |
| Area feature rows | 99,286,106 |
| Retained total | 121.4 GB decimal / 113.0 GiB |
| Recommended free space | 180-200 GiB |
| Maintenance-safe same-disk space | 230-250 GiB |

The estimate is evidence-based but uncertain across older source eras and
future bloat. Feature tables were not partitioned in this experiment.

## Decision and limitations

Accepted for the three-source historical backfill with monitoring. It did not
meet a true 3-5 minute fresh-day target, did not measure CPU headroom, did not
validate radar, and does not prove every 2017-era provider object has the 2026
shape.

## Current safe reproduction shape

The historical accepted run used worker/retry settings that current safety
guards no longer allow. Do not copy those old commands. Current bounded
one-day validation:

```powershell
$env:HKG_TMAX_DATABASE_URL = '<local PostgreSQL URL>'
.\.venv\Scripts\python.exe scripts\backfill_public_weather_to_postgres.py --start-date 2026-06-21 --end-date 2026-06-21 --sources gfs,gefs_control,himawari_b13_s0510 --execution-mode optimized --model-fetch-workers 2 --model-range-workers 2 --model-normalize-workers 2 --himawari-workers 2 --model-range-coalesce-gap-bytes 0 --max-attempts 3 --max-staging-gb 1 --execute
```

Review provider, DB, disk, and runtime budgets before `--execute`.

## Evidence map

- Root `STATUS.yaml`, `RUN_CONFIG.yaml`, `DATA_MANIFEST.yaml`, and
  `results/metrics.json`.
- `full_robustness_jun10_jul8_rerun4/` and
  `targeted_gfs_retry_after_full_rerun4/` machine evidence.
- Shard `metrics.json` files preserve every daily run.
- Exact retired implementation, live-DB, and capacity documents remain
  recoverable through
  [`DOCUMENT_PROVENANCE.csv`](../../DOCUMENT_PROVENANCE.csv).
