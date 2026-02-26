# Documentation Index

This repository has a dedicated documentation track for the KLGA same-day Tmax distribution system used for Polymarket-style bucket probabilities.

Primary documentation folder:

- `documentation/klga_same_day_tmax_distribution/`

## Required Read Order

1. `documentation/klga_same_day_tmax_distribution/README.md`
2. `documentation/klga_same_day_tmax_distribution/00_high_level_overview.md`
3. `documentation/klga_same_day_tmax_distribution/01_system_spec_and_implementation.md`
4. `documentation/klga_same_day_tmax_distribution/02_metrics_and_interpretation_for_beginners.md`
5. `documentation/klga_same_day_tmax_distribution/03_runbook_training_inference_and_artifacts.md`
6. `documentation/klga_same_day_tmax_distribution/04_run_history_and_current_status_2026-02-26.md`
7. `documentation/klga_same_day_tmax_distribution/05_faq_ultra_clear.md`
8. `documentation/klga_same_day_tmax_distribution/06_full_feature_dictionary.md`

## Fast Path by Use Case

- Want executive understanding first: read `00` then `02`.
- Implementing or modifying code: read `01`, `03`, and `06` in full.
- Debugging run quality or artifact completeness: read `03` then `04`.
- Resolving user confusion about peak/delta/NLL semantics: read `02` and `05`.

## Non-Negotiable Policy

Any change to the KLGA system must preserve:

- strict as-of leakage safety,
- canonical KLGA source alignment,
- split isolation (train/val/test by date),
- reproducible artifact generation.

If a code change conflicts with these rules, treat that change as a regression.
