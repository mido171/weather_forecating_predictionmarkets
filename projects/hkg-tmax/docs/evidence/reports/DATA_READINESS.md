# Data Readiness

## Gate Status

Status: `PARTIAL_PASS_FOR_PHASE_A_B_EDA`.

This repository can now parse the acquired raw archive into reproducible analysis tables and produce leakage-screened candidate features for the primary HKO T-24 cutoff. It is not yet cleared for final production modelling because several source families remain current-only, credential-gated, historically unavailable, or missing empirical publication-lag proof.

## Parsed Evidence

| Item | Value |
|---|---:|
| Retrieval ledger rows | 10,903 |
| Successful retrieval rows | 10,901 |
| Non-success retrieval rows | 2 |
| Parsed HKO daily climate rows | 556,399 |
| Official HKO Tmax target rows | 49,459 |
| HKO target date range | 1884-01-01 to 2026-05-31 |
| Selected high-frequency observation rows parsed | 1,887,741 |
| Selected high-frequency observed range | 2020-06-30 09:00:00+08:00 to 2026-06-18 15:30:00+08:00 |
| Station-network cutoff summary rows | 75,166 |
| T-24 candidate feature rows with HKO cutoff temperature | 1,932 |

## Hard Gates

- No target-day values are used as T-24 predictors.
- Reanalysis/final products remain retrospective-only unless release lag is explicitly proven.
- Current-only NWP is not backtestable and is rejected for retrospective model fitting.
- Official same-day daily climate rows are target/mechanism labels, not operational predictors.
- The main usable operational archive for initial analysis is HKO high-frequency station observations from 2020/2021 onward.
