# Feature Engineering Catalog

## Feature family separation

The final model must not be a feature soup. Generate separate, versioned families and preserve missingness/availability masks.

## Official HKO forecast features

- latest eligible max/min/range/midpoint at cutoff;
- first eligible forecast for T and revision deltas to latest;
- number of issues, issue intervals, revision velocity and direction;
- issue age, issue hour, product/parser/source era;
- weather, wind, humidity, rain-probability and free-text features fitted fold-locally;
- official max minus target-memory state;
- official max minus every NWP expert forecast;
- recent source-specific signed and absolute residual states.

## Target-memory features

Use only labels available before cutoff. Include lags, causal rolling means/quantiles, slopes, curvature, volatility, MAD/IQR, spell length, reversal, day-of-year climatology, decayed climatology, year-over-year analog state, and long-run climate trend. If daily publication time is unproven, apply the conservative lag contract; do not use T−1 merely because it is present retrospectively.

## Station-network features

For every station and date-effective metadata identity:

- latest-before-cutoff level;
- prior-only 1/3/7/14/30-day changes and anomalies;
- temperature-dewpoint spread;
- pressure tendency;
- u/v wind and circular direction change after parser repair;
- morning/pre-cutoff warming slopes when subdaily data exists;
- station rank and rank reversal;
- coastal-inland, north-south, east-west and upwind-downwind gradients;
- graph modes/PCA fitted inside training folds;
- station disagreement and missingness state.

## Deterministic NWP features

For each model independently:

- target-day hourly 2 m temperature trajectory and direct Tmax;
- peak hour, heating/cooling slopes, plateau duration and diurnal range;
- dew point, T−Td, humidity and moisture-flux features;
- cloud and radiation integrals over the heating window;
- precipitation timing and accumulation;
- wind vectors, onshore component, shear and direction changes;
- MSLP, pressure tendency, 500 hPa height and subsidence indices;
- 925/850/700/500 hPa thermal/moisture profile and stability;
- HKO/inland/coastal/marine gradients;
- run age and revision between eligible cycles.

## Ensemble features

- member-level target-day Tmax;
- mean, median, P05/P10/P25/P75/P90/P95;
- standard deviation, IQR, skewness, kurtosis;
- threshold probabilities;
- member clusters and multimodality;
- spread of cloud/rain/radiation/temperature trajectories;
- control-minus-mean and deterministic-minus-ensemble disagreement.

## Diagnostic teacher features

IGRA, finalized daily climate, marine historical values, and TC best track may label physical mechanisms for teacher/student experiments. Teacher values must never appear in live student inputs unless availability is separately proven.

## Online states

Maintain source/model residual EWMAs at several half-lives, robust median residual, recent MAE, residual volatility, over/underforecast streaks, change-point probability and source-regime support counts. Update only after settlement.

## Feature registry requirement

Every feature must declare formula, inputs, availability proof, fold-fitting behavior, missingness policy, expected role, unit, bounds, and production status.
