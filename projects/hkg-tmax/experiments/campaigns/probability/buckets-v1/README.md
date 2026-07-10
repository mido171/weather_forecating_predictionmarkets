# Probability bucket calibration V1

Status: `complete`. Scope: weather probability only.

## Question and contract

Convert strict HKO Info.gov lead-1 local forecast rows into one-decimal HKG
Tmax bucket distributions. No market prices, EV, order books, sizing, PnL, or
trade recommendations were used.

The target was the HKO Daily Extract one-decimal daily maximum. Leakage and
label first-publication audits were mandatory.

## Result and decision

`B4_hierarchical_residual_pmf` won under the configured simplicity/gating
contract:

| Metric | B4 |
|---|---:|
| Normalized RPS | 0.041524 |
| NLL | 1.037181 |
| Brier | 0.045921 |
| ECE | 0.019859 |

B5 had a lower raw RPS but failed the NLL no-worse gate. The conservative stack
did not clear the minimum RPS gain. Leakage passed with zero violations and
the label audit reported zero bucket changes.

## Reproduce

```powershell
$env:HKG_TMAX_DATABASE_URL = '<local PostgreSQL URL>'
.\.venv\Scripts\python.exe scripts\run_hkg_tmax_probability_bucket_v1.py --config config\experiments\hkg_tmax\probability_bucket_v1.yaml --output-dir experiments\campaigns\probability\buckets-v1\results
```

## Evidence map

`results/scoreboard*.csv`, `bucket_probabilities.parquet`,
`one_decimal_pmfs.parquet`, `leakage_audit.json`,
`label_publication_audit.json`, `model_selection_log.json`, and
`reproducibility_manifest.json` remain. The manifest's historical Markdown
entry is recoverable through the campaign provenance ledger.
