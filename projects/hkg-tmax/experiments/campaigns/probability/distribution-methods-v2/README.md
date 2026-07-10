# Probability distribution methods V2

Status: `accepted`. Supreme method: `B4_hierarchical_residual_pmf`.

## Question and contract

Test true EMOS-style and other continuous/discrete distribution engines
against B4 using weather data only. Primary historical cutoff: T-1 23:59 HKT;
sensitivity cutoffs: 18:00 and 21:00. Sealed confirmation was not used for
selection.

A challenger had to beat B4 by at least 1.5% RPS on folds 1-4 and 1.0% on the
2022-2023 presealed holdout, while staying within 0.005 NLL and 0.002 Brier.

## Result

| Method | RPS | Decision |
|---|---:|---|
| B5 kernel analog | 0.041287 | Failed fold/presealed/NLL gates |
| H1 linear pool | 0.041470 | Gain too small |
| S1 simplex stack | 0.041472 | Gain too small |
| T1 time-decay B4 | 0.041486 | Fold gate failed |
| B4 hierarchical residual PMF | 0.041524 | Retained reference champion |
| E2 Student-t EMOS | 0.041658 | Worse RPS; NLL gate failed |

Normal, Student-t, two-piece normal EMOS, tree location-scale, quantile CDF,
and threshold CDF challengers did not promote. Leakage, row identity, and live
no-trading audits passed.

## Decision and limitation

B4 remains champion. V2's full model runner was not rerun after a later
Markdown-renderer-only patch; this does not change the frozen scoreboards but
is retained as a reproducibility limitation.

## Reproduce

```powershell
$env:HKG_TMAX_DATABASE_URL = '<local PostgreSQL URL>'
.\.venv\Scripts\python.exe scripts\run_hkg_tmax_probability_distribution_methods_v2.py --config config\experiments\hkg_tmax\probability_distribution_methods_v2.yaml --output-dir experiments\campaigns\probability\distribution-methods-v2\results
.\.venv\Scripts\python.exe -m pytest tests\test_hkg_tmax_probability_distribution_methods_v2.py -q
```

## Evidence map

`results/scoreboard.csv`, split/cutoff scoreboards, distribution-parameter
Parquet, `method_selection_log.json`, `leakage_audit.json`,
`row_identity_gate.json`, and `reproducibility_manifest.json` remain.
`RUN_CONFIG.yaml` uses `HKG_TMAX_DATABASE_URL` rather than storing a DSN.
