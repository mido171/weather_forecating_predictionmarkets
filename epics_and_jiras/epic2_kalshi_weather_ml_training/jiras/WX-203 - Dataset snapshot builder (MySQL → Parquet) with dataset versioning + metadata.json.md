# WX-203 — Dataset snapshot builder (MySQL → Parquet) with dataset versioning + metadata.json

## Objective
Create a repeatable dataset snapshot system so training runs are auditable and reproducible.

## Requirements
- Given a config (stations, date range, asOf policy, missing strategy), build a dataset and write:
  - `datasets/<dataset_id>/data.parquet`
  - `datasets/<dataset_id>/metadata.json`
- dataset_id must be deterministic from:
  - stations list
  - date range
  - asof_policy_id
  - feature list
  - missing strategy
  - DB snapshot signature (at least: max(retrieved_at_utc) per input table)
- metadata.json must include:
  - created_at_utc
  - dataset_id
  - SQL query template + parameters
  - row count
  - missing fraction by feature
  - leakage check summary
  - library versions (python, pandas, numpy, sklearn)

## Acceptance Criteria
- [ ] Running dataset build twice (without DB changes) produces same dataset_id and identical parquet hash.
- [ ] metadata.json fully documents the dataset.
- [ ] If DB changes (new rows or updated), dataset_id changes OR metadata notes changed snapshot signature.

