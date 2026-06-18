# Results

## Run Integrity

- command: `.\.venv\Scripts\python.exe scripts\poll_daily_extract.py --year 2026 --month 6`
- code commit: `25da86741a6ac7d770a8bddbde95bbca2a50deba` plus EXP-0003 dirty changes
- rows: 17 June 2026 Daily Extract target rows
- failed rows: 0 parser failures
- leakage validator: no forecast features or model fitting performed
- reproducibility precheck: `pytest`, `validate all`, `ruff`, and `mypy` passed

## Primary Result

| Metric | Result |
|---|---:|
| Catalog payloads archived in run | 1 |
| Monthly payloads archived in run | 1 |
| Ledger rows | 17 |
| Rows with raw hash and retrieved-at provenance | 17 |
| Rows labelled provider first publication | 0 |
| Revisions observed | 0 |

## Guardrails

All 17 rows are labelled `ARCHIVE_FIRST_OBSERVED_NOT_PROVIDER_FIRST`. This is
intentional: the values already existed before near-publication polling was
active, so they cannot prove provider first-publication timing.

No Polymarket backtesting, price history, order-book, trade, liquidity,
execution, replay, predictive modelling, or machine-learning work was run.

## Evidence

- latest catalog hash:
  `f80772b68545c56e6842c34998696fd11b7b9a80c0088bb1f6e4da65102616eb`
- latest monthly hash:
  `c50910ab74e2ba8bff1f661fb1ae663d15b128dae0dfb4ed97c0e40c97bcbefc`
- first archive observation for the 17 rows came from the earlier
  `2026-06-18T17:12:46.511184Z` raw snapshot.
- latest observation came from the EXP-0003 poll at
  `2026-06-18T17:30:21.023419Z`.

## Failure Taxonomy

- `MISSING_FIRST_PUBLICATION`: all 17 rows, because polling was not active
  before the rows appeared publicly.
- `REVISION_OBSERVED`: 0 rows.
- `SOURCE_OUTAGE`: 0 rows.

## Full Artifact List

- `data/gold/target_publication/daily_extract_first_seen.csv` (ignored generated artifact)
- `reports/daily_extract_publication.md`
- `experiments/EXP-0003-g1-daily-extract-first-publication-polling/results/metrics.json`
- `scripts/poll_daily_extract.py`
- `src/hkg_tmax/publication.py`
- `tests/test_publication.py`
