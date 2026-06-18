# Reproduce

## Clean Environment

```powershell
git checkout <commit>
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[research,dev]"
```

## Run

```powershell
.\.venv\Scripts\python.exe scripts\poll_daily_extract.py --year 2026 --month 6
```

## Expected Outputs

| File | Expected contents |
|---|---|
| `experiments/EXP-0003-g1-daily-extract-first-publication-polling/results/metrics.json` | `row_count=17`, `revision_count=0`, `provider_first_publication_proven=false` |
| `data/gold/target_publication/daily_extract_first_seen.csv` | 17 dated rows for 2026-06, all archive-first-observed |
| `reports/daily_extract_publication.md` | documents G1 still blocked pending provider first-publication evidence |

## Expected Metric Tolerances

No tolerance for parser output on the same raw inputs. A live re-fetch can create
new sidecars if HKO updates the current month; that becomes a new evidence
snapshot and should be documented.

## External Immutable Data Locations

- `data/raw/hko_daily_extract_catalog/2026/06/18/20260618T173020.025020Z__f80772b68545c56e.xml`
- `data/raw/hko_daily_extract_202606/2026/06/18/20260618T173021.023419Z__c50910ab74e2ba8b.xml`

## No Undocumented Steps

No manual data edits. The command archives raw bytes before parsing and derives
the ledger from raw sidecars.
