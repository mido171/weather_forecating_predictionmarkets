# HKG Tmax Probability Bucket Calibration V1

Weather-probability-only experiment for converting strict HKO Info.gov lead-1 local forecast max/min rows into HKG Tmax bucket probability distributions.

No market prices, expected value, order books, Kelly sizing, PnL, market-implied blending, or trade recommendations are used.

## Reproduce

```powershell
.\.venv\Scripts\python.exe scripts\run_hkg_tmax_probability_bucket_v1.py --config configs\hkg_tmax\probability_bucket_v1.yaml --output-dir experiments\hkg_tmax_probability_buckets_v1\results
```

## Main Artifacts

```text
results/scoreboard.csv
results/scoreboard_by_split.csv
results/scoreboard_by_cutoff.csv
results/modeling_table.parquet
results/selected_forecast_rows.parquet
results/per_fold_predictions.parquet
results/bucket_probabilities.parquet
results/one_decimal_pmfs.parquet
results/leakage_audit.json
results/label_publication_audit.json
results/model_selection_log.json
results/final_probability_model_card.md
results/reproducibility_manifest.json
```

## Current Result

`B4_hierarchical_residual_pmf` is the champion under the configured simplicity gate. `B5_kernel_analog_pmf` had the lowest normalized RPS but failed the NLL no-worse gate. `S1_conservative_simplex_stack` passed NLL/Brier but its RPS gain versus B4 was below the configured promotion threshold.
