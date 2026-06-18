# Reproduce

## Clean environment

```bash
git checkout <commit>
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[research,dev]"
```

## Verify inputs

```bash
python -m hkg_tmax manifest
# Verify the exact manifest/hash commands listed below.
```

## Run

```powershell
.\.venv\Scripts\python.exe scripts\poll_daily_extract.py --year 2026 --month 6 --iterations 6 --interval-seconds 30 --fetch-attempts 3 --retry-sleep-seconds 2 --active-polling-start-at 2026-06-18T17:48:59.956593Z --watch-candidate-date 2026-06-18 --metrics experiments\EXP-0010-g1-daily-extract-active-first-publication-watch-2026-06-18-fifth\results\metrics.json
```

## Expected outputs

| File | SHA-256 or tolerance |
|---|---|
| results/metrics.json | Exact rerun will create new raw retrieval timestamps; semantic metrics should match watched-date status unless provider content changed |
| reports/daily_extract_publication.md | Regenerated publication report |

## Validation

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m hkg_tmax validate all
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m mypy src
```

## Expected metric tolerances

Target-publication status can change if HKO publishes the watched row during a
rerun. Raw archive paths and retrieved timestamps are expected to differ.

## External immutable data locations

- `data/raw/hko_daily_extract_catalog/`
- `data/raw/hko_daily_extract_202606/`

## Known platform differences

None expected beyond path separators.

## No undocumented steps

No manual steps are expected.
