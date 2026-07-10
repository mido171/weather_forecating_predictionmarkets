# Quantitative Method Atlas

Methods are tools for hypotheses, not automatic experiment lanes.

## Transparent baselines

- causal climatology;
- persistence;
- weighted target memory;
- official raw;
- bounded global/source residual mean;
- exponentially weighted residual memory;
- fixed expert blend.

Every complex model must beat the appropriate simple baseline.

## Regularized linear residual models

Use for:

- signed incremental effects;
- stable interactions;
- high interpretability.

Requirements:

- fold-local scaling;
- regularization selected in nested temporal data;
- family ablation;
- correction cap;
- collinearity diagnostics.

## Generalized additive models

Use for:

- smooth nonlinear response;
- saturation;
- season interactions;
- interpretable residual correction.

Fit splines and smoothing parameters within training history. Compare to linear/hinge features.

## Tree boosting

Use for:

- nonlinear interactions among whitelisted features;
- missingness-aware residual correction;
- high-error classification.

Constraints:

- shallow/depth-limited;
- strict temporal nesting;
- fixed search budget;
- no diagnostic-only features;
- feature-family ablations;
- correction shrinkage/caps.

## Random forests and extremely randomized trees

Useful diagnostics for interaction discovery and uncertainty, but can overfit eras and sparse station patterns. Require temporal OOF and compare against boosting/GAM.

## Hierarchical shrinkage

Use for:

- source-season-residual states;
- station groups;
- sparse regimes;
- cell atlases.

Back off specific states to broader priors. Report effective support.

## Online exponential states

Specify half-life, key, shrinkage, cap, update order, and cold start. Compare multiple half-lives only under nested selection.

## Change-point and regime methods

Potential methods:

- cumulative sum;
- Bayesian or penalized change points;
- hidden-state models;
- slope/volatility heuristics.

State inference for target T must use only prior observations. Complex latent-state models must beat transparent transition scores.

## Analog and nearest-neighbor methods

Use physically scaled prior-only distances, season/source restrictions, minimum effective neighbors, and abstention. Compare against residual memory.

## Spatial methods

- robust group contrasts;
- inverse-distance weighting;
- PCA;
- graph Laplacian;
- spatial regression;
- upwind weighting.

Any learned transform is fold-local. Static geometry is versioned.

## Feature selection

Acceptable:

- predeclared physics whitelist;
- fold-local regularization;
- nested forward selection with small budget;
- stability selection within training folds.

Unacceptable:

- full-history target correlation ranking fed into outer OOF without nesting;
- selecting features from confirmation;
- hiding search count.

## Calibration and distributions

- residual empirical distribution;
- quantile regression;
- variance/absolute-error model;
- isotonic or logistic event calibration;
- conformal intervals with temporal calibration;
- ensemble distribution.

Report calibration by season/source and tail coverage.

## Routers and blends

Baselines:

- fixed blend;
- inverse prior error;
- positive-lift gate;
- no-regret online weights;
- abstention.

Router inputs must be pre-cutoff. Router target and hyperparameters use prior folds only. Compare against the best single expert.

## Statistical diagnostics

- paired loss differences;
- block bootstrap respecting time;
- Diebold-Mariano-style diagnostics with caution;
- fold sign consistency;
- effect concentration;
- permutation under temporal blocks;
- multiple-testing correction for broad atlases.

Statistical significance never overrides operational relevance or leakage.

## Research-overfit controls

- pre-registration;
- nested temporal folds;
- attempt registry;
- family-level search budgets;
- untouched late development slice;
- freeze before confirmation;
- all negative results retained.

## Acceptance philosophy

Prefer:

- small stable gains;
- bounded corrections;
- physically coherent features;
- broad coverage;
- tail protection;
- replayability.

Demote:

- giant corrections;
- one-year wins;
- tiny cells;
- complex stacks that dilute;
- improvements dependent on blocked timing;
- models that cannot be reproduced.
