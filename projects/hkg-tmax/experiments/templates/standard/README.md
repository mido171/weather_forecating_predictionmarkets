# {{EXPERIMENT_ID}}: {{TITLE}}

Created at `{{CREATED_AT_UTC}}`.

This README is the single human-readable experiment record. Keep the sections
below concise and update them in place; store detailed evidence in the linked
YAML, JSON, CSV, or Parquet artifacts.

## Status

State the lifecycle status and whether this record is planned, running,
complete, blocked, rejected, or accepted.

## Question and hypothesis

State the falsifiable claim, expected mechanism, baseline, and minimum useful
effect before running the experiment.

## As-of contract

List every feature's publication timestamp and prove that no observation,
forecast revision, target, or market value after the decision cutoff is used.

## Method

Document the frozen data slice, temporal split, baseline, candidate, metrics,
ablation plan, promotion gates, resource budget, and stop conditions.

## Results

Record frozen results, uncertainty, slices, resource usage, and exact machine
artifact locations. Never overwrite evidence from a completed run.

## Decision

Record whether the hypothesis was supported, the promotion decision, failure
modes, negative evidence, and the next permitted action.

## Reproduce

Provide the exact offline command, environment, input manifest, seed, expected
outputs, and verification checks. Network or backfill commands require a
separate explicit execution acknowledgement.

## Evidence map

- `STATUS.yaml`: lifecycle and gate status.
- `DATA_MANIFEST.yaml`: frozen input provenance.
- `RUN_CONFIG.yaml`: run configuration and resource bounds.
- `results/metrics.json`: primary machine-readable result.
