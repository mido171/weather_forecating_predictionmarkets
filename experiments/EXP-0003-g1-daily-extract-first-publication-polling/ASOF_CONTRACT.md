# As-Of Contract

## Forecast Cutoff

Not applicable. EXP-0003 does not build forecast features or predictions.

## Source Timing

| Source ID | Valid timestamp | Availability evidence | Revision behavior | Eligible role |
|---|---|---|---|---|
| `hko_daily_extract_catalog` | coverage metadata current at retrieval | raw sidecar `retrieved_at` and HTTP headers | provider may update coverage | endpoint selection metadata |
| `hko_daily_extract_YYYYMM` | Hong Kong local calendar date per row | raw sidecar `retrieved_at` and raw hash | later payloads may add/revise rows | target-publication evidence |

## Evidence Classes

- `ARCHIVE_FIRST_OBSERVED_NOT_PROVIDER_FIRST`: first time this repository saw
  a row/hash; not proof of provider first publication.
- `PROVIDER_FIRST_PUBLICATION_CANDIDATE`: allowed only when polling was active
  before the row appeared and cadence/latency evidence is documented.
- `REVISION_OBSERVED`: later payload differs from the first archived value for
  the same local date.

EXP-0003 may create only the first class unless the run genuinely observes a new
row appear under active polling.

## Explicitly Forbidden Data

- Predictive features or model fitting.
- CLMMAXT promoted to canonical target.
- Polymarket price history, books, trades, liquidity, execution, or market
  replay.
- Any first-publication claim based solely on latest or post-hoc payloads.

## Automated Checks

- `.venv\Scripts\python.exe -m pytest`
- `.venv\Scripts\python.exe -m hkg_tmax validate all`
- `.venv\Scripts\python.exe -m ruff check src tests scripts`
- `.venv\Scripts\python.exe -m mypy src`

## Residual Uncertainty

A single run can prove polling mechanics and archive-first-observed evidence. It
cannot prove provider first-publication parity for dates that were already
published before polling began.
