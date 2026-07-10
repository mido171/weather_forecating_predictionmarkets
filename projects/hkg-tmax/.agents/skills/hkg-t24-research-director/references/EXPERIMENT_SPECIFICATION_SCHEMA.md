# Experiment Specification Schema and Authoring Guide

The machine-readable schema is bundled at `references/schemas/experiment_spec.schema.json`. The specification is the scientific contract between the Director and Executor.

## Required top-level fields

### `schema_version`

Must be `1.0`.

### `experiment_id`

Four digits when reserved. May be null in a draft but is assigned by scaffolding.

### `title` and `slug`

Human title and stable lowercase folder slug.

### `mode`

One of:

- `promotion_oriented`
- `exploratory`
- `diagnostic_only`
- `data_quality`
- `timestamp_unlock`
- `frame_harmonization`

### `hypothesis`

One falsifiable sentence. State the response and expected incremental effect.

Bad:

```text
Explore stations.
```

Good:

```text
On pre-2024 exact-vintage anchor rows, an easterly-wind-gated
coastal-minus-inland temperature anomaly predicts negative official residual
and a bounded correction reduces MAE beyond source-aware h20 memory.
```

### `rationale`

Explain mechanism, prior evidence, unresolved question, and why this experiment has high value.

### `expected_sign_and_falsification`

State expected sign, where it should be strongest, and what result would lower belief.

### `novelty`

Include nearest prior experiment IDs, material difference, and novelty-audit artifact.

### `target`

Required:

- station;
- variable `tmax_c`;
- horizon `T-24`;
- timezone `Asia/Hong_Kong`;
- exact cutoff contract path;
- cutoff function and daily boundary where applicable.

### `frame`

Required:

- stable frame ID;
- date boundaries;
- development end exclusive `2024-01-01`;
- confirmation locked;
- row-universe artifact/hash.

### `data_sources`

For every source:

- source ID;
- exact paths or deterministic glob;
- exact attributes;
- timestamp fields;
- eligibility class;
- availability proof;
- timezone;
- latency contract;
- quality fields;
- expected coverage.

A source marked diagnostic/block/prospective cannot enter promotion inference.

### `stations`

Specify:

- `explicit`, `all_inventory`, or role groups;
- exact IDs or deterministic census selection;
- metadata requirement;
- unknown identity policy;
- role definitions;
- station dropout test when relevant.

### `features`

Each feature object requires:

- name;
- role;
- exact formula;
- inputs;
- units;
- lag/window;
- time alignment;
- availability rule;
- fit scope;
- missingness policy;
- expected mechanism;
- ablation group.

### `response`

State exact formula, sign, unit, target availability, and whether it is primary.

### `baseline`

State:

- ID;
- definition or artifact;
- version/hash;
- identical rows required;
- why this is the correct baseline for the claimed role.

### `candidate`

State:

- family;
- complete algorithm/formula;
- correction cap;
- shrinkage;
- minimum support;
- refit cadence;
- hyperparameter selection;
- random seeds;
- output.

### `validation`

State:

- expanding/rolling/prequential method;
- outer fold dates;
- inner selection;
- minimum history;
- refit cadence;
- embargo;
- source-gap handling;
- cold start;
- late-window reserve.

### `metrics`

At minimum:

- MAE;
- RMSE;
- bias;
- median AE;
- p90 AE;
- p95 AE.

Include response-specific metrics such as Brier, log loss, AUC, calibration, sign errors, or activation lift.

### `sample_rules`

Include:

- minimum total rows;
- per-fold rows;
- per-cell/activation rows;
- maximum concentration in one year;
- missingness;
- station coverage;
- behavior when support fails.

### `acceptance_gates`

Separate:

- information-gain support;
- promotion;
- fold stability;
- year/season/source stability;
- late-window behavior;
- tail no-harm;
- correction magnitude;
- complexity justification;
- reproducibility.

### `rejection_conditions`

Include:

- cutoff unknown;
- availability unproven;
- confirmation contamination;
- frame mismatch;
- baseline row mismatch;
- insufficient support;
- data-quality failure;
- inability to reproduce inputs.

### `parameter_grid`

Fix the complete search space before outer scoring. Empty for deterministic candidates.

### `required_outputs`

List exact files. Include README, RESULTS, CONCLUSION, predictions, scores, audit, manifests, and experiment-specific diagnostics.

### `owner_authorized_confirmation`

Must be false in development.

## Example skeleton

```json
{
  "schema_version": "1.0",
  "experiment_id": null,
  "title": "Wind-gated coastal suppression residual correction",
  "slug": "wind_gated_coastal_suppression",
  "mode": "promotion_oriented",
  "hypothesis": "...",
  "rationale": "...",
  "expected_sign_and_falsification": "...",
  "novelty": {
    "prior_experiments": ["0100", "0101", "0103"],
    "difference": "...",
    "similarity_audit_path": "..."
  },
  "target": {
    "station": "Hong Kong Observatory",
    "variable": "tmax_c",
    "horizon": "T-24",
    "timezone": "Asia/Hong_Kong",
    "cutoff_contract_path": "...",
    "cutoff_function": "...",
    "daily_boundary_contract": "..."
  },
  "frame": {
    "frame_id": "...",
    "development_start": "2000-01-01",
    "development_end_exclusive": "2024-01-01",
    "confirmation_locked": true,
    "row_universe_artifact": "..."
  },
  "data_sources": [],
  "stations": {},
  "features": [],
  "response": {},
  "baseline": {},
  "candidate": {},
  "validation": {},
  "metrics": [],
  "sample_rules": {},
  "acceptance_gates": {},
  "rejection_conditions": [],
  "parameter_grid": {},
  "required_outputs": [],
  "owner_authorized_confirmation": false
}
```

## Validation

Run:

```text
python <director-skill>/scripts/validate_experiment_spec.py experiment_spec.json
```

No experiment is handed off until validation passes.
