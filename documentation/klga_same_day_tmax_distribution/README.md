# KLGA Same-Day Tmax Distribution Documentation

This folder is the canonical documentation set for the KLGA same-day probabilistic Tmax system.

System objective:

- At each KLGA local cutoff (`04:00` to `18:00`, every 30 minutes), estimate a full integer-Fahrenheit PMF for final daily max temperature.
- Convert that PMF into bucket probabilities for prediction-market outcomes.

## Code and Artifacts

Implementation package:

- `ml/src/weather_ml/klga_daily_tmax_dist/`

Runner:

- `ml/run_klga_daily_tmax_dist.py`

Default artifact root:

- `artifacts/same_day_res_poly/`

## Reader Profiles

- Beginner or decision-maker: `00` then `02`.
- Engineer implementing changes: `01`, `03`, and `06`.
- Operator monitoring/recovering runs: `03` and `04`.
- Anyone confused by metrics or model interaction: `02` and `05`.

## Mandatory Read Order

1. `00_high_level_overview.md`
2. `01_system_spec_and_implementation.md`
3. `02_metrics_and_interpretation_for_beginners.md`
4. `03_runbook_training_inference_and_artifacts.md`
5. `04_run_history_and_current_status_2026-02-26.md`
6. `05_faq_ultra_clear.md`
7. `06_full_feature_dictionary.md`

## System Invariants

- Leakage safety is mandatory.
- As-of semantics are mandatory.
- Date split isolation is mandatory.
- KLGA canonical source semantics are mandatory.

Treat any violation as a blocker, not a warning.
