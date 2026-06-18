# Reproduce

## Run

```powershell
.\.venv\Scripts\python.exe scripts\poll_daily_extract.py --year 2026 --month 6 --iterations 4 --interval-seconds 30 --active-polling-start-at now --watch-candidate-date 2026-06-18 --metrics experiments\EXP-0005-g1-daily-extract-active-first-publication-watch-2026-06-18\results\metrics.json
```

## Expected Outputs

| File | Expected contents |
|---|---|
| `results/metrics.json` | active polling start, watched date list, row/evidence counts |
| `reports/daily_extract_publication.md` | notes G1 remains blocked unless cadence review passes |
| `data/gold/target_publication/daily_extract_first_seen.csv` | first/last archive evidence by observed local date |

## Validation

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m hkg_tmax validate all
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m mypy src
```
