# As-of Contract

## Forecast cutoff

- timezone: Asia/Hong_Kong
- horizon ID: not applicable
- local expression: not applicable
- UTC expression: not applicable
- grace/latency rule: every provider payload is usable only as publication
  evidence at or after its archived `retrieved_at` timestamp

## Feature eligibility

This experiment creates no predictive features. If any artifact is later used
for modelling, it must satisfy:

```text
available_at <= forecast_cutoff
```

## Source-specific timing

| Source ID | Valid/issue timestamp | Availability evidence | Conservative latency | Revision behavior | Eligible? |
|---|---|---|---:|---|---|
| hko_daily_extract_catalog | Provider catalog response retrieval time | immutable raw snapshot metadata sidecar `retrieved_at` | none beyond archived retrieval time | can change; each poll snapshot is immutable | target-publication evidence only |
| hko_daily_extract_202606 | Provider monthly payload response retrieval time | immutable raw snapshot metadata sidecar `retrieved_at` | none beyond archived retrieval time | can add rows or revise values; each poll snapshot is immutable | target-publication evidence only |

## Explicitly forbidden data

No target-day realized observation may be used as a predictive feature. No
market prices, model outputs, future cycles, corrected files, reanalysis, or
Daily Extract value may be used to select horizons or train/evaluate forecasts
in this experiment.

## Preprocessing timing

Parsing, ledger generation, and report generation are publication-evidence
steps only. There is no fitting, imputation, scaling, feature selection,
calibration, or regime classification.

## Automated checks

Pre-poll:

```powershell
.\.venv\Scripts\python.exe -m hkg_tmax validate all
```

Final:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m hkg_tmax validate all
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m mypy src
```

## Residual uncertainty

HKO does not provide an explicit first-publication timestamp in the observed
payload. The project therefore requires active absent-before-present archive
evidence and later revision review before G1 can pass.
