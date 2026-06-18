# Results

## Run Integrity

- command: `.\.venv\Scripts\python.exe scripts\poll_daily_extract.py --year 2026 --month 6 --iterations 2 --interval-seconds 1 --metrics experiments/EXP-0004-g1-daily-extract-bounded-polling-candidate-gating/results/metrics.json`
- code commit: `fcd541135aabe3391ecc40b3919649530d53f24c` plus EXP-0004 dirty changes
- polling iterations completed: 2
- bounded loop: exited normally
- watched candidate dates: none
- leakage validator: no forecast features or model fitting performed

## Primary Result

| Metric | Result |
|---|---:|
| Poll iterations completed | 2 |
| Interval seconds | 1 |
| Ledger rows | 17 |
| Provider-first candidates | 0 |
| Revisions observed | 0 |
| Evidence class count: `ARCHIVE_FIRST_OBSERVED_NOT_PROVIDER_FIRST` | 17 |

## Safety Result

The bounded loop ran and exited on its own. Because no watched candidate dates
were provided, no row could receive provider-first-publication candidate status.
Existing June 2026 rows remained archive-first-observed only.

## Artifacts

- `experiments/EXP-0004-g1-daily-extract-bounded-polling-candidate-gating/results/metrics.json`
- `reports/daily_extract_publication.md`
- `data/gold/target_publication/daily_extract_first_seen.csv` (ignored generated artifact)
- `scripts/poll_daily_extract.py`
- `src/hkg_tmax/publication.py`
- `tests/test_publication.py`
