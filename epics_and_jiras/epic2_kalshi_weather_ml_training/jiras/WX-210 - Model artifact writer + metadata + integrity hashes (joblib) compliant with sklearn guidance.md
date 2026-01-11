# WX-210 — Model artifact writer + metadata + integrity hashes (joblib) compliant with sklearn guidance

## Objective
Persist models and make them safely loadable for inference/backtesting.

## Requirements
- Save artifacts under `artifacts/<run_id>/<station>/...`
- Save metadata.json including:
  - run_id
  - created_at_utc
  - station_id
  - asof_policy_id
  - train/val/test date ranges
  - feature list + missing strategy
  - model types and hyperparameters
  - versions: python, sklearn, numpy, pandas
- Compute SHA-256 hashes for:
  - each model file
  - metrics.json
  - report.md

## Security note (must be included)
- joblib is pickle-based; do not load untrusted artifacts.
- Include this note in metadata and documentation.

References:
- scikit-learn model persistence guidance.
- joblib persistence warning.

## Acceptance Criteria
- [ ] Artifacts saved for each station.
- [ ] Hashes computed and verified on load.
- [ ] Loading artifacts reproduces identical predictions on a small test fixture.

