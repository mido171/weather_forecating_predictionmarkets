# As-of Contract

## Forecast cutoff

- timezone: Asia/Hong_Kong
- horizon ID:
- local expression:
- UTC expression:
- grace/latency rule:

## Feature eligibility

Every feature row must satisfy:

```text
available_at <= forecast_cutoff
```

## Source-specific timing

| Source ID | Valid/issue timestamp | Availability evidence | Conservative latency | Revision behavior | Eligible? |
|---|---|---|---:|---|---|
| TBD | TBD | TBD | TBD | TBD | TBD |

## Explicitly forbidden data

List target-day realized observations, future cycles, corrected files, reanalysis, final best tracks, or any other unavailable information.

## Preprocessing timing

Explain how fitting, imputation, scaling, feature selection, calibration, and regime classification remain training-only.

## Automated checks

List commands/tests proving the contract.

## Residual uncertainty

Document timing assumptions that cannot be proven exactly and sensitivity tests applied.
