# Experiment Execution Protocol

## Phase 0 — specification freeze

Before reading the target column, capture a hash of `experiment_spec.json`. Validate that the specification includes experiment name, hypothesis, response variable, source families, exact features or generation rules, frame, cutoff contract, availability class, baseline, folds, tuning grid, metrics, sample rules, no-harm gates, outputs, and rejection conditions. Missing decisions cause `REJECTED_SPECIFICATION`; do not fill them after viewing outcomes.

## Phase 1 — folder reservation

Use the canonical numbered folder convention `<NNNN>_<slug>`. Compute the next unused number from all existing four-digit prefixes. Create atomically; never reuse a number. Write the specification hash and creation time to `run_manifest.json`.

## Phase 2 — pre-target data audit

Read metadata, schemas, timestamps, units, and row counts before target outcomes. Confirm joins and keys. Detect duplicates, timezone ambiguity, mixed units, impossible values, sentinel values, and files containing both development and confirmation. Build `data_manifest.csv` and `leakage_audit.md` before model fitting.

## Phase 3 — causal feature implementation

For every feature, document formula, inputs, lag, window, minimum periods, fit/update rule, missingness rule, and earliest possible target date. Unit-test boundary dates. For target memory, shift before rolling. For station values, choose the latest row whose operational available-at time is no later than cutoff. For daily aggregates, prove the aggregate excludes post-cutoff observations. For fold-fitted transformations, serialize fold-local parameters or sufficient metadata.

## Phase 4 — baseline reproduction

Reproduce the declared baseline on the exact candidate row set. Compare the reproduced baseline score with its canonical artifact. Investigate discrepancies before candidate scoring. If the candidate requires fewer rows, report coverage loss and do not compare with a baseline score from the larger frame.

## Phase 5 — outer walk-forward

Generate predictions chronologically. Each outer fold trains only on earlier target dates. Any hyperparameter or feature-selection decision is made inside outer training history. For online state, predict first, then update after the target becomes available. Persist fold IDs and train-end dates per prediction.

## Phase 6 — metrics and diagnostics

Compute global, fold, year, season, month, source, source-era, late-window, missingness, and declared regime metrics. Include signed residual analysis and high-error tails. Save exact row identifiers and both predictions. Quantify coverage, trigger frequency, and correction distribution. Report uncertainty intervals where declared.

## Phase 7 — conclusion

Classify the experiment based on predeclared gates. Separate information gain from point-MAE lift. Identify whether failure arose from no signal, redundancy, unstable sign, sparse support, timestamp blocking, bad proxy fidelity, over-flexibility, under-flexibility, or data quality. Do not propose a new experiment beyond concise implications; the Research Director owns the next full specification.

## Phase 8 — validation

Run the folder validator. Verify Markdown metrics match CSV/JSON values. Verify all scripts are in `src/`, imports are reproducible, and no temporary path is embedded as an authoritative input. Record code and data hashes.
