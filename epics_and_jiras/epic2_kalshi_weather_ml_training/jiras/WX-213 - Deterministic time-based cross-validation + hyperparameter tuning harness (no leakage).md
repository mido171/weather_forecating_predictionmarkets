# WX-213 — Deterministic time-based cross-validation + hyperparameter tuning harness (no leakage)

## Objective
Add robust model selection without leakage.

## Requirements
- Provide a tuning harness that:
  - uses TimeSeriesSplit or explicit rolling windows
  - evaluates candidates based on:
    - MAE/RMSE
    - log loss / Brier for bins
  - selects champion model per station
- Must be deterministic with fixed seed.

Reference:
- TimeSeriesSplit documentation (sklearn).

## Acceptance Criteria
- [ ] Re-running tuning with same seed and data produces the same champion selection.
- [ ] Tuning results stored in metrics.json and report.md.
- [ ] No random shuffle is used.

