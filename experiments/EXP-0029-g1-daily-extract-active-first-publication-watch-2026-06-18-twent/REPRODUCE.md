# Reproduce

## Environment

- Python: `Python 3.11.9`
- Platform: `Windows-10-10.0.26200-SP0`
- Dependency freeze SHA-256:
  `1620ada7725901e17d4ce0d580f660161aa83954748df84034e40a8b9bf41123`
- Base commit before reservation:
  `adaf5d9cc4861b5a524e864ba614b2fa2f00a51b`

## Commands

```powershell
.\.venv\Scripts\python.exe -m hkg_tmax validate all
.\.venv\Scripts\python.exe scripts\poll_daily_extract.py --year 2026 --month 6 --iterations 6 --interval-seconds 30 --fetch-attempts 3 --retry-sleep-seconds 2 --active-polling-start-at 2026-06-18T17:48:59.956593Z --watch-candidate-date 2026-06-18 --metrics experiments\EXP-0029-g1-daily-extract-active-first-publication-watch-2026-06-18-twent\results\metrics.json
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m hkg_tmax validate all
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m mypy src
```

## Expected outputs

| File | SHA-256 or tolerance |
|---|---|
| `results/metrics.json` | exact JSON content may vary by retrieval time and provider state |
| `reports/daily_extract_publication.md` | updated to latest poll |

## Expected metric tolerances

No floating-point model metrics are produced.

## External immutable data locations

Raw snapshots are stored under:

- `data/raw/hko_daily_extract_catalog/`
- `data/raw/hko_daily_extract_202606/`

## Known platform differences

The active run is on Windows PowerShell. Paths in metrics may be absolute
Windows paths.

## No undocumented steps

No manual step is expected beyond running the listed commands.
