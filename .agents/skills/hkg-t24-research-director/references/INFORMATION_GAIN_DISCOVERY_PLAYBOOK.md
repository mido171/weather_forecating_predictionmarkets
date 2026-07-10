# Information-Gain Discovery Playbook

## Objective

Find relationships that are stable, incremental, point-in-time safe, and actionable. The playbook separates broad discovery from promotion so that nonlinear screens do not become unregistered model selection.

## Stage 1: Define the response and frame

Before reading outcome relationships:

- choose the response from the response atlas;
- choose the canonical frame;
- specify source coverage;
- seal 2024+;
- define minimum row support;
- identify the baseline whose residual is being studied.

Create a row-universe artifact and hash it.

## Stage 2: Univariate views

For each eligible feature:

- Pearson correlation where appropriate;
- Spearman rank correlation;
- fold-local standardized effect;
- decile or quantile response spread;
- monotonicity score;
- missing-versus-present response;
- linear and robust slope;
- response enrichment in high/low tails;
- year and season consistency;
- source consistency.

Report row count, date range, feature missingness, and eligibility. Do not rank a feature without these.

## Stage 3: Nonlinear response curves

Use fold-local bins, isotonic diagnostics, splines, or generalized additive smooths to inspect:

- thresholds;
- saturation;
- sign reversals;
- U-shapes;
- tail effects;
- asymmetry.

Curves are diagnostic unless evaluated out of fold. Avoid plotting full-history target means and calling them predictive.

## Stage 4: Conditional effects

Condition each promising feature by plausible state:

- season/month;
- wind sector;
- moisture regime;
- pressure regime;
- target-memory state;
- source;
- official forecast range;
- residual sign;
- station disagreement.

Use support thresholds and shrinkage. The goal is to distinguish genuine interactions from mix-of-regime artifacts.

## Stage 5: Two-feature interaction cells

For a ranked physically annotated queue:

- define bin thresholds from training history;
- require support;
- compute signed response, absolute response, tail enrichment, and confidence;
- test main-effect subtraction;
- compare folds;
- inspect late-window behavior;
- record activation rate.

Do not screen millions of cells without false-discovery control and a confirmatory follow-up.

## Stage 6: Incremental residual tests

Fit simple, regularized, temporally valid models in this order:

1. baseline only;
2. candidate main feature;
3. baseline plus candidate;
4. baseline plus feature family;
5. baseline plus interaction;
6. current champion plus candidate.

Use out-of-fold predictions and identical rows. This reveals whether information is new.

## Stage 7: Information metrics

Possible diagnostics:

- change in out-of-fold MAE/RMSE;
- change in p90/p95 absolute error;
- out-of-fold R² for residual;
- mutual-information-style binned entropy reduction estimated within temporal folds;
- AUC/average precision for high-error flags;
- Brier/log-loss improvement for event responses;
- calibration improvement;
- stability score;
- activation-weighted error contribution;
- conditional lift beyond parent state.

Mutual information is exploratory and bias-prone with small samples. It must not substitute for walk-forward lift.

## Stage 8: Stability score

Construct a transparent score from:

- number of positive outer folds;
- direction consistency;
- year coverage;
- season/source coverage;
- late-window effect;
- support concentration;
- sensitivity to thresholds;
- missingness robustness;
- station dropout robustness.

Save components, not only the aggregate.

## Stage 9: Redundancy and clustering

Compute feature-family redundancy using prior/fold-training data:

- correlation clusters;
- rank-correlation clusters;
- mutual predictability;
- shared source/lineage;
- ablation overlap.

Select representatives or use regularization. A long list of variants of the same signal is not broad information gain.

## Stage 10: Promotion experiment

A discovery signal is promoted only through a new pre-registered experiment that fixes:

- feature construction;
- response;
- model;
- folds;
- parameter budget;
- correction cap;
- baseline;
- acceptance/no-harm gates.

Discovery results cannot be called confirmation.

## Required outputs

- `univariate_signals.csv`
- `nonlinear_response_curves.parquet`
- `conditional_signals.csv`
- `interaction_cells.csv`
- `incremental_ablation.csv`
- `stability_scores.csv`
- `redundancy_clusters.csv`
- `promotion_queue.csv`
- `negative_signals.csv`
- `leakage_audit.md`
- `README.md`, `RESULTS.md`, `CONCLUSION.md`

## Priority responses

Mine each feature family against:

- raw target Tmax;
- official residual;
- official absolute error;
- overforecast and underforecast;
- high-error flag;
- hot-day underforecast;
- cold-day overforecast;
- MAM high-error;
- station-only residual;
- online-memory residual;
- current champion residual;
- forecast trust.

This matrix is mandatory over the life of the research program, not necessarily in one experiment.
