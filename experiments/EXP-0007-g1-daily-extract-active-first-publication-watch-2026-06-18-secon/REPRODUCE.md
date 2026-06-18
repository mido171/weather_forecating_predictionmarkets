# Reproduce

## Run

```powershell
.\.venv\Scripts\python.exe scripts\poll_daily_extract.py --year 2026 --month 6 --iterations 6 --interval-seconds 30 --fetch-attempts 3 --retry-sleep-seconds 2 --active-polling-start-at 2026-06-18T17:48:59.956593Z --watch-candidate-date 2026-06-18 --metrics experiments\EXP-0007-g1-daily-extract-active-first-publication-watch-2026-06-18-secon\results\metrics.json
```

## Validation

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m hkg_tmax validate all
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m mypy src
```
