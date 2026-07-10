# Split Freeze

Primary T-24 common-sample split is frozen before advanced model fitting.

| Split | Start | End | Role |
|---|---|---|---|
| Development | 2021-07-01 | 2023-12-31 | EDA and baseline design |
| Validation 2024 | 2024-01-01 | 2024-12-31 | Champion baseline selection |
| Locked test | 2025-01-01 | 2026-05-31 | Final untouched comparison for this baseline suite |

Common-sample rule: target exists, HKO high-frequency cutoff temperature exists, and the row is within the frozen dates.

The split was not altered after seeing baseline results.
