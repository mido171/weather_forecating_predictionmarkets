# As-of Contract

## Forecast cutoff

- timezone: `Asia/Hong_Kong`
- horizon ID: not applicable
- local expression: not applicable
- UTC expression: not applicable
- grace/latency rule: not applicable

## Feature eligibility

No feature rows are built. This experiment records target-publication evidence
only.

## Source-specific timing

| Source ID | Valid/issue timestamp | Availability evidence | Conservative latency | Revision behavior | Eligible? |
|---|---|---|---:|---|---|
| hko_daily_extract_catalog | provider coverage metadata | archive `retrieved_at` | 0 for archive observation only | provider may update coverage | metadata only |
| hko_daily_extract_202606 | HKO local day rows | archive `retrieved_at`; no row-level provider timestamp observed | 0 for archive observation only | may add/revise rows | target only |

## Explicitly forbidden data

- predictive features;
- model training or scoring;
- CLMMAXT as canonical target;
- market prices or winners;
- revisions substituted for first observed raw payloads.

## Preprocessing timing

No fitting, imputation, scaling, feature selection, calibration, or regime
classification is performed.

## Automated checks

Pre-poll:

```powershell
.\.venv\Scripts\python.exe -m hkg_tmax validate all
```

Post-poll:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m hkg_tmax validate all
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m mypy src
```

## Residual uncertainty

The provider has not exposed row-level `published_at` in the observed payloads.
The experiment therefore proves an archive-observed absence/presence window,
not an exact provider timestamp.
