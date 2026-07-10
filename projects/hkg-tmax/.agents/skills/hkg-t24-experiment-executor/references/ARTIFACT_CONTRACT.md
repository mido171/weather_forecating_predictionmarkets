# Experiment Folder Artifact Contract

## Mandatory for every status

```text
README.md
RESULTS.md
CONCLUSION.md
experiment_spec.json
leakage_audit.md
data_manifest.csv
feature_definitions.csv
summary.json
run_manifest.json
REPRODUCE.md
src/
```

## Mandatory for scored experiments

```text
scoreboard.csv
slice_metrics.csv
yearly_metrics.csv
fold_metrics.csv
predictions.parquet or predictions.csv.gz
row_coverage.csv
correction_distribution.csv when corrections are produced
```

## Mandatory for rejected or blocked experiments

```text
REJECTION.md
```

The result files may contain zero score rows, but must explain why scoring was prohibited.

## README.md sections

1. Experiment identity and status
2. One-sentence hypothesis
3. Why it is worth doing
4. Prior evidence and novelty
5. Target, horizon, and exact cutoff
6. Datasets, stations, and attributes
7. Feature definitions
8. Response and baseline
9. Walk-forward design
10. Acceptance and rejection criteria
11. Expected failure modes
12. Reproduction command

## RESULTS.md sections

1. Headline result table
2. Coverage and row identity
3. Global metrics
4. Fold stability
5. Yearly and seasonal results
6. Source and source-era results
7. High-error-tail results
8. Signed over/underforecast results
9. Ablations
10. Data-quality and leakage result
11. Comparison limitations

## CONCLUSION.md sections

1. Verdict
2. What was learned
3. Realized point-MAE change
4. Information gain outside point MAE
5. Robustness and uncertainty
6. Failure diagnosis
7. Promotion status
8. Implication for future research

## summary.json required keys

`experiment_id`, `slug`, `status`, `created_at_utc`, `target`, `frame_id`, `date_start`, `date_end`, `n_candidate`, `n_common`, `baseline_id`, `baseline_mae_c`, `candidate_id`, `candidate_mae_c`, `mae_delta_c`, `candidate_rmse_c`, `candidate_bias_c`, `leakage_status`, `confirmation_rows_used`, `promotion_decision`, `spec_sha256`, `code_sha256`, and `data_manifest_sha256`.

Use `null` for unavailable score fields, never fabricated zeroes.
