# Results

## Run Integrity

- command: `.\.venv\Scripts\python.exe scripts\poll_daily_extract.py --year 2026 --month 6 --iterations 6 --interval-seconds 30 --fetch-attempts 3 --retry-sleep-seconds 2 --active-polling-start-at 2026-06-18T17:48:59.956593Z --watch-candidate-date 2026-06-18 --metrics experiments\EXP-0007-g1-daily-extract-active-first-publication-watch-2026-06-18-secon\results\metrics.json`
- active polling start: `2026-06-18T17:48:59.956593Z`
- polling iterations completed: 6
- poll snapshot count: 6
- fetch attempts per request: 3
- retry sleep seconds: 2
- leakage validator: target-publication evidence only; no forecast features or model fitting

## Primary Result

| Metric | Result |
|---|---:|
| Ledger rows | 17 |
| Watched date present | 0 |
| Watched date missing | 1 |
| Provider-first candidates | 0 |
| Revisions observed | 0 |
| Evidence class count: `ARCHIVE_FIRST_OBSERVED_NOT_PROVIDER_FIRST` | 17 |

The watched `2026-06-18` row remained absent through the final monthly snapshot
at `2026-06-18T18:09:27.640472Z`.

## Artifacts

- `experiments/EXP-0007-g1-daily-extract-active-first-publication-watch-2026-06-18-secon/results/metrics.json`
- `reports/daily_extract_publication.md`
- `data/gold/target_publication/daily_extract_first_seen.csv` (ignored generated artifact)

## Gate Checks

- `pytest`: PASS
- `hkg_tmax validate all`: PASS with expected G1/G2 gating warnings
- `ruff`: PASS
- `mypy`: PASS
