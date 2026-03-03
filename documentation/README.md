# Documentation Index

This repository has a dedicated documentation track for the KLGA same-day Tmax distribution system used for Polymarket-style bucket probabilities.

It also includes a KNYC point-forecast/quantile track for the quantile+KNN+gate+rolling-conformal pipeline.

It also includes a KNYC MOS-first backtesting and audit track for Kalshi execution simulation.

Primary documentation folder:

- `documentation/klga_same_day_tmax_distribution/`
- `documentation/knyc_quantile_knn_conformal/`
- `documentation/mos/`

## Required Read Order

1. `documentation/klga_same_day_tmax_distribution/README.md`
2. `documentation/klga_same_day_tmax_distribution/00_high_level_overview.md`
3. `documentation/klga_same_day_tmax_distribution/01_system_spec_and_implementation.md`
4. `documentation/klga_same_day_tmax_distribution/02_metrics_and_interpretation_for_beginners.md`
5. `documentation/klga_same_day_tmax_distribution/03_runbook_training_inference_and_artifacts.md`
6. `documentation/klga_same_day_tmax_distribution/04_run_history_and_current_status_2026-02-26.md`
7. `documentation/klga_same_day_tmax_distribution/06_full_feature_dictionary.md`
8. `documentation/klga_same_day_tmax_distribution/07_exporter_and_remote_training_tabm.md`
9. `documentation/klga_same_day_tmax_distribution/09_results_metrics_and_feature_importance.md`

## Fast Path by Use Case

- Want executive understanding first: read `00` then `02`.
- Implementing or modifying code: read `01`, `03`, and `06` in full.
- Debugging run quality or artifact completeness: read `03` then `04`.
- Resolving user confusion about peak/delta/NLL semantics: read `02` (includes merged FAQ section).
- Exporting data or training on another machine: read `07` (includes troubleshooting section).
- Reviewing final results and full feature importance: read `09`.

## KNYC Track

- Entry: `documentation/knyc_quantile_knn_conformal/README.md`
- Latest run record: `documentation/knyc_quantile_knn_conformal/00_run_record_2026-02-28.md`
- MOS-first run record: `documentation/knyc_quantile_knn_conformal/01_run_record_2026-03-01_mos_first_plan.md`

## MOS Backtesting Track

- Entry: `documentation/mos/README.md`
- Scope and objective: `documentation/mos/00_scope_and_objective.md`
- Data contracts and mapping: `documentation/mos/01_data_contracts_and_file_mapping.md`
- Backtest formulas: `documentation/mos/02_backtest_logic_and_formulas.md`
- Sanity audit framework: `documentation/mos/03_sanity_audit_framework.md`
- Current audited run: `documentation/mos/04_run_record_2026-03-01_entry1530z_cap400.md`
- Leakage-free runtime matrix: `documentation/mos/06_run_record_2026-03-01_leakage_free_runtime_matrix.md`
- Co-joined baseline run (`KNYC` + `KMIA`): `documentation/mos/07_run_record_2026-03-01_knyc_kmia_cojoined_blend12.md`
- Latest strict co-joined run (fractional Kelly + outlier-filtered recalc): `documentation/mos/08_run_record_2026-03-01_knyc_kmia_cojoined_blend12_fractionalkelly_no_outlier_gt2000.md`
- Troubleshooting: `documentation/mos/05_troubleshooting_and_common_failure_modes.md`

When editing any bridge/backtest/trading code, the MOS track is mandatory reading in addition to the KLGA/KNYC docs.

## Non-Negotiable Policy

Any change to the KLGA system must preserve:

- strict as-of leakage safety,
- canonical KLGA source alignment,
- split isolation (train/val/test by date),
- reproducible artifact generation.

If a code change conflicts with these rules, treat that change as a regression.
