# Evaluation Design

Primary forecast target: HKO daily maximum temperature for local date T.

Primary cutoff: `T-1 15:00 HKT`.

## Split Policy

| Split | Dates | Purpose |
|---|---|---|
| Development | up to 2023-12-31 | EDA, feature screening, baseline design |
| Validation 2024 | 2024-01-01 to 2024-12-31 | Stability checks and tuning guardrail |
| Locked holdout | 2025-01-01 onward | Do not use for creative iteration until formal experiment protocol is frozen |

## Rules

- Freeze target definition and timestamp contracts before fitting any model.
- Fit baselines before complex model families.
- Report persistence, climatology, seasonal climatology, and simple physically motivated baselines before ML.
- Keep Polymarket/backtesting fully out of scope.
- Treat correlations in `EDA_MASTER_REPORT.md` as hypothesis generation only.
