# Leakage and Timestamp Protocol

## Row-level eligibility predicate

A predictor value is eligible for target T only when all applicable conditions hold:

```text
source_available_at_hkt <= cutoff_hkt(T)
source_issue_at_hkt     <= cutoff_hkt(T), when an issue concept exists
observation_end_hkt     <= cutoff_hkt(T), for aggregates
feature_fit_end_date    <  target_date(T), for learned state
outcome_update_time     <= cutoff_hkt(T), for residual memory
```

If an archive lacks `available_at`, eligibility requires a documented provider release-latency contract that covers every row and is conservative relative to the cutoff. A median or typical latency is insufficient.

## Required checks

- timezone localization and local daily boundary versus UTC date;
- duplicate issue times and revision selection;
- post-cutoff observations inside daily summaries;
- use of target T in rolling windows;
- climatology and scaler fit ranges;
- future station normals or peer-group ranks;
- full-history PCA, imputation, feature selection, or normalization;
- confirmation rows entering aggregate statistics;
- target-derived missingness labels;
- retrospective best-track or quality-control revisions;
- joins that choose nearest time in both directions rather than backward-only;
- backfilled data whose historical publication cannot be proven.

## Required audit evidence

`leakage_audit.md` must include the exact cutoff function, feature-by-feature eligibility, earliest/latest available-at relative to cutoff, maximum observed look-ahead, confirmation exclusion proof, fold-local fit proof, online state update ordering, rejected columns, and code-test results.

A single failed deployable feature rejects the promotion candidate unless the specification predeclared a target-blind ablation removing it.
