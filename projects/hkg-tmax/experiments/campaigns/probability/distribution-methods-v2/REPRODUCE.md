# Reproduce HKG Tmax Probability Distribution Methods V2

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\run_hkg_tmax_probability_distribution_methods_v2.py --config configs\hkg_tmax\probability_distribution_methods_v2.yaml --output-dir experiments\hkg_tmax_probability_distribution_methods_v2\results
```

Focused test command:

```powershell
.\.venv\Scripts\python.exe -m pytest code\tests\test_hkg_tmax_probability_distribution_methods_v2.py -q
```

The runner expects PostgreSQL `hkg_tmax_research` at the configured DSN unless `--database-url` is supplied.

The run writes scoreboards, prediction parquet files, continuous distribution parameters, leakage and row-identity audits, diagnostics, final model card, supreme-method summary, and a reproducibility manifest under `results/`.
