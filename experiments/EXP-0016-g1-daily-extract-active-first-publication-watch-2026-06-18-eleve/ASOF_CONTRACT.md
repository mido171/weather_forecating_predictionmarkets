# As-of Contract

## Forecast cutoff

- timezone: Asia/Hong_Kong for target date identity; UTC for archive timestamps
- horizon ID: not applicable
- local expression: watched target date `2026-06-18`
- UTC expression: inherited active polling start `2026-06-18T17:48:59.956593Z`
- grace/latency rule: provider-first candidate requires an active absent
  snapshot before first active present snapshot

## Feature eligibility

No forecast features are produced. Target publication evidence is eligible only
for G1 parity analysis, not model fitting.

## Source-specific timing

| Source ID | Valid/issue timestamp | Availability evidence | Conservative latency | Revision behavior | Eligible? |
|---|---|---|---:|---|---|
| `hko_daily_extract_catalog` | retrieval time only | raw snapshot metadata sidecar | none inferred | mutable catalog | target-publication evidence only |
| `hko_daily_extract_202606` | retrieval time only | raw snapshot metadata sidecar | none inferred | mutable monthly payload | target-publication evidence only |

## Explicitly forbidden data

Predictive weather features, finalized daily labels as canonical truth, revised
CLMMAXT used to infer first publication, reanalysis, future cycles, and market
data or Polymarket backtesting are forbidden.

## Preprocessing timing

No fitting, imputation, scaling, feature selection, calibration, or regime
classification is performed.

## Automated checks

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m hkg_tmax validate all
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m mypy src
```

## Residual uncertainty

HKO does not expose a signed `published_at` timestamp in the fetched Daily
Extract payload. The experiment therefore only proves a bounded archive
observation window, not an exact provider publication instant.
