# Reproduce

## Run

```powershell
.\.venv\Scripts\python.exe scripts\poll_daily_extract.py --year 2026 --month 6 --iterations 2 --interval-seconds 1 --metrics experiments\EXP-0004-g1-daily-extract-bounded-polling-candidate-gating\results\metrics.json
```

## Expected Outputs

| File | Expected contents |
|---|---|
| `results/metrics.json` | `poll_iterations_completed=2`, `row_count=17`, `revision_count=0` |
| `reports/daily_extract_publication.md` | notes G1 remains blocked |
| `data/gold/target_publication/daily_extract_first_seen.csv` | 17 archive-first-observed rows |

## Validation

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m hkg_tmax validate all
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m mypy src
```
