# Modelling Roadmap

## Stage 0 — No model

Prove target, time, archive, and split integrity.

## Stage 1 — Deterministic baselines

- seasonal climatology;
- trend-adjusted climatology;
- persistence;
- official forecast;
- each raw NWP model;
- simple mean/median consensus.

Goal: establish the minimum honest bar.

## Stage 2 — Distributional baselines

- empirical climatological distribution;
- model ensemble distribution;
- model residual bootstrap;
- quantile mapping;
- conformal intervals with rolling calibration.

Goal: calibrated forecast distributions and bucket probabilities.

## Stage 3 — Model output statistics

By model/cycle/lead/season:

- additive bias;
- robust regression;
- quantile regression;
- heteroskedastic residual model;
- regime-conditional correction.

Goal: correct grid/station and systematic model error transparently.

## Stage 4 — Physical expert corrections

Separate components for:

- vertical thermal potential;
- wind/sea-breeze;
- radiation/cloud;
- station-network anomaly;
- rain/convection;
- tropical-cyclone/synoptic regime;
- run-to-run changes.

Goal: mechanisms with measurable incremental value.

## Stage 5 — Transparent expert stack

Constrained combination and calibration. Require component attribution and stable weights.

## Stage 6 — Controlled machine learning

Only after G8 entry gate:

- regularized linear and quantile models;
- GAM;
- gradient boosting;
- distributional boosting;
- shallow ensembles.

Feature count, interaction depth, and search space are constrained. Nested temporal tuning is mandatory.

## Stage 7 — Advanced structured ML

Only with sufficient authentic vintages and live archive:

- graph models for station network;
- temporal sequence models for forecast-vintage trajectories;
- spatial CNN/transformer encoders for radar/satellite;
- multimodal mixture-of-experts;
- learned probabilistic postprocessing.

These are challengers, not presumed improvements.

## Distribution representation options

1. quantile grid with monotonic repair;
2. parametric distribution;
3. residual samples;
4. ensemble dressing;
5. mixture distribution;
6. discrete 0.1°C mass function.

The last option maps naturally to one-decimal settlement but must handle tails and calibration.

## Probability mapping

For bucket \([l,u)\):

\[
p_j = F(u^-) - F(l^-)
\]

Implement with a single tested probability adapter. Avoid ad hoc rounding.

## Calibration architecture

Calibration is a separate versioned layer:

- rolling empirical residual;
- isotonic/Platt/beta for threshold events;
- Dirichlet/multinomial calibration for buckets;
- conformal coverage;
- regime-conditional calibration only when sample size supports it.

## Fallback hierarchy

Example:

1. champion with all sources;
2. champion without one missing NWP;
3. official forecast + climatology;
4. climatology only;
5. no forecast/trade when target/rules integrity is compromised.

Fallback performance is evaluated before production.
