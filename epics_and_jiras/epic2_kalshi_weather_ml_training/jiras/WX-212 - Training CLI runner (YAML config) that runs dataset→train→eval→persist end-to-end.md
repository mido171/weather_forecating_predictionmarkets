# WX-212 — Training CLI runner (YAML config) that runs dataset→train→eval→persist end-to-end

## Objective
Provide a single command that runs the full Epic #2 pipeline.

## Requirements
- `python -m weather_ml.train --config <file> --stage <dataset|train|eval|all>`
- YAML config includes:
  - mysql connection params
  - stations list
  - date_start/date_end
  - asof_policy_id
  - missing strategy
  - model hyperparams
  - artifact output directory
  - random seed

## Acceptance Criteria
- [ ] Running with `--stage all` produces:
  - dataset snapshot
  - mean + sigma models
  - calibrated bin probs
  - metrics.json and report.md
- [ ] CLI prints a concise summary at end with key metrics and artifact locations.

