# Evaluation and Validation

## Forecast objects

For each target date and horizon, save:

- continuous predictive CDF or samples/quantiles sufficient to reconstruct it;
- mean, median, mode;
- calibrated intervals;
- probability for every exact market bucket;
- input cutoff and vintage manifest;
- model/version ID;
- prediction creation timestamp.

## Primary metrics

### Continuous

**CRPS** is primary because it rewards both calibration and sharpness over the whole distribution.

Also report:

- MAE;
- RMSE;
- bias;
- median absolute error;
- quantile loss;
- interval coverage and width;
- probability integral transform diagnostics.

### Contract

**Multiclass log loss** is primary because the market consists of mutually exclusive outcomes and overconfidence must be punished.

Also report:

- multiclass Brier score;
- per-bucket reliability;
- top-choice accuracy;
- boundary-day performance;
- tail-bucket performance;
- entropy/sharpness.

## Why MAE is insufficient

Two systems can have similar MAE but radically different:

- uncertainty calibration;
- threshold probabilities;
- tail risk;
- usefulness when the expected value lies near a bucket boundary.

A model is not promoted based on point error alone.

## Baseline comparisons

Use identical dates. Report:

1. common-sample comparison;
2. each model’s full available sample;
3. excluded-date analysis.

Missing hard days can create fake improvement.

## Rolling origin

For each forecast date:

- train only on earlier eligible target dates;
- fit all preprocessing within training;
- calibrate within training or nested recent window;
- predict the next block/date;
- append immutable prediction.

No random K-fold.

## Locked test governance

Create `reports/TEST_ACCESS_LOG.md` with:

```text
timestamp
researcher/agent
experiment IDs
frozen candidates
reason for opening
files accessed
decision made
```

Repeatedly viewing test performance converts it into validation data. Future confirmation must rely on live shadow or a new untouched period.

## Statistical uncertainty

Weather errors are serially correlated. Use paired moving-block bootstrap rather than iid formulas. Explore block lengths and report sensitivity.

Statistical significance is secondary to:

- effect size;
- stability;
- operational feasibility;
- calibration;
- failure behavior.

## Regime robustness

At minimum score:

- cool/warm season;
- each month;
- dry/rainy;
- wind sectors;
- model spread;
- tropical-cyclone context;
- high/low Tmax;
- years;
- data-quality states.

Do not declare a general gain if it is one-regime-only. A specialized model may still be useful if the gating regime is preobservable and reliable.

## Calibration

Inspect:

- PIT histogram;
- empirical coverage;
- reliability diagrams;
- bucket-frequency calibration;
- calibration slope/intercept where appropriate;
- expected calibration error with binning sensitivity.

Calibration methods are trained only on past data and versioned.

## Boundary diagnostics

For every integer boundary \(k\), study dates where official Tmax lies within:

- ±0.1°C;
- ±0.3°C;
- ±0.5°C.

Report predicted crossing probabilities and systematic errors. A 0.1°C bias can dominate market outcomes near boundaries.

## Champion selection

The champion is selected by the predeclared primary metric subject to guardrails. Tie-breaking order:

1. better calibration;
2. simpler/less fragile;
3. greater source reliability;
4. lower latency;
5. lower operational cost;
6. better worst-regime behavior.

## Market evaluation separation

Do not tune meteorological models directly to historical P&L until the forecast is independently validated. Market prices can be used as a benchmark or later ensemble input only under a separately declared experiment, because they may introduce reflexivity and limited sample size.
