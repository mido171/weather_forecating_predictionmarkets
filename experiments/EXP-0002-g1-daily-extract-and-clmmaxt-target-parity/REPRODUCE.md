# Reproduce

## Clean Environment

```powershell
git checkout <commit>
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[research,dev]"
```

## Verify Inputs

```powershell
.\.venv\Scripts\python.exe -m hkg_tmax manifest
```

## Run

```powershell
.\.venv\Scripts\python.exe scripts\build_target_parity.py --year 2026 --month 5 --daily-source-id hko_daily_extract_202605
```

## Expected Outputs

| File | SHA-256 or tolerance |
|---|---|
| `experiments/EXP-0002-g1-daily-extract-and-clmmaxt-target-parity/results/metrics.json` | `row_count=31`, `latest_match_count=31`, `latest_mismatch_count=0`, `first_publication_proven=false` |
| `data/gold/target_parity/target_parity.csv` | 31 dated rows for 2026-05, all `MATCH_LATEST_ONLY` |
| `reports/target_parity.md` | documents G1 blocked pending first-publication evidence |

## Expected Metric Tolerances

No tolerance. Parsing and equality are deterministic.

## External Immutable Data Locations

- `data/raw/hko_daily_extract_202605/2026/06/18/20260618T171245.346959Z__a97230cd78e0a11c.xml`
- `data/raw/hko_clmmaxt_hko/2026/06/18/20260618T160552.249124Z__5a0a646b4d125e40.csv`

## Known Platform Differences

Paths in metrics/report are absolute Windows paths in this run. Values, hashes,
and row counts are platform-independent.

## No Undocumented Steps

The backing payloads must exist under `data/raw/`. Re-fetching current HKO
payloads may produce new sidecars and should be documented as a new run.
