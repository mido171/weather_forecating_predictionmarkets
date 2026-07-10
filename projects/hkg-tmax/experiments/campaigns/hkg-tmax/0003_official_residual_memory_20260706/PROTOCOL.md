# Protocol

- Primary cutoff: `T-1 23:59 HKT`.
- Sensitivity cutoffs: `T-1 21:00 HKT`, `T-1 18:00 HKT`.
- Fold 1-4 are used for model and hyperparameter selection.
- 2022-2023 is presealed holdout after candidate freeze.
- 2024-2026-05 is sealed confirmation and report-only.
- Residual-memory predictors use same-cutoff official residuals from `T-2` or older.
- Lag-1 residuals, target-date residuals, raw audit payloads, helped/worsened labels, raw error bins, and sealed labels are excluded from predictors.
