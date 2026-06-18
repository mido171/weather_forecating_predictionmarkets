# Results

## Run Integrity

- command: `.\.venv\Scripts\python.exe scripts\poll_daily_extract.py --year 2026 --month 6 --iterations 4 --interval-seconds 30 --active-polling-start-at now --watch-candidate-date 2026-06-18 --metrics experiments\EXP-0005-g1-daily-extract-active-first-publication-watch-2026-06-18\results\metrics.json`
- active polling start: `2026-06-18T17:48:59.956593Z`
- polling iterations completed: 4
- bounded loop: exited normally
- watched candidate date: `2026-06-18`
- leakage validator: target-publication evidence only; no forecast features or model fitting

## Primary Result

| Metric | Result |
|---|---:|
| Poll iterations completed | 4 |
| Interval seconds | 30 |
| Ledger rows | 17 |
| Watched date present | 0 |
| Watched date missing | 1 |
| Provider-first candidates | 0 |
| Revisions observed | 0 |
| Evidence class count: `ARCHIVE_FIRST_OBSERVED_NOT_PROVIDER_FIRST` | 17 |

The watched `2026-06-18` row did not appear during this active polling window.
The final monthly payload hash remained
`c50910ab74e2ba8bff1f661fb1ae663d15b128dae0dfb4ed97c0e40c97bcbefc`.

## Code/Guardrail Result

The publication ledger now requires an active absent-before-present raw snapshot
sequence before assigning `PROVIDER_FIRST_PUBLICATION_CANDIDATE`. Focused
publication tests passed after this change. The full gate suite also passed:
`pytest`, `hkg_tmax validate all`, `ruff`, and `mypy`.

## Artifacts

- `experiments/EXP-0005-g1-daily-extract-active-first-publication-watch-2026-06-18/results/metrics.json`
- `reports/daily_extract_publication.md`
- `data/gold/target_publication/daily_extract_first_seen.csv` (ignored generated artifact)
