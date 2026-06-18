# Research Charter

## Objective

Forecast the first-published contract-authoritative HKO daily maximum temperature as a calibrated continuous distribution and as probabilities over exact Polymarket outcomes at a predeclared cutoff.

The optimization target is not the number of clever features. It is robust improvement in point-in-time, out-of-sample probabilistic accuracy, followed by conservative executable market value.

## Research hierarchy

1. **Target correctness**
2. **Temporal correctness**
3. **Data quality and provenance**
4. **Strong transparent baselines**
5. **Physical and station-specific insight**
6. **Robust probabilistic combination**
7. **Controlled ML**
8. **Market execution and risk**

A downstream layer may not compensate for failure upstream.

## Core deliverables

- exact target adapter and settlement emulator;
- immutable point-in-time archive;
- source and station metadata timelines;
- fixed forecast cutoffs;
- row-level baseline and model predictions;
- calibrated continuous and bucket distributions;
- experiment ledger with full negative-result retention;
- champion/challenger scoreboard;
- live shadow forecaster;
- market replay and risk framework;
- monitoring and fail-closed production controls.

## Scientific standard

Every experiment must distinguish:

- **exploration**: hypothesis generation using development data;
- **confirmation**: frozen protocol on validation data;
- **final verification**: locked test opened under a documented decision;
- **live shadow**: predictions timestamped before outcomes.

Post-hoc insight is allowed, but it is labelled exploratory and receives a new confirmation experiment.

## Novelty standard

The program should search beyond standard model extraction through:

- station-network geometry and flow-conditioned spatial modes;
- run-to-run NWP information;
- vertical thermal-potential diagnostics;
- urban/marine boundary-layer transitions;
- conditional forecast-error analogues;
- dynamic source reliability;
- distributional and threshold-specific calibration;
- regime-conditioned expert mixtures;
- data-quality and publication-process signals that are lawful and not target leakage.

Novelty is never inferred from complexity. A method is useful only when reproducibly better.

## Scope boundaries

In scope:

- official/public/licensed meteorological and market data;
- first-published target reconstruction;
- pre-event and eventually intraday forecasts;
- lawful automated archival and analysis;
- conservative paper trading and execution simulation.

Out of scope:

- sensor interference or manipulation;
- unauthorized access;
- exploiting confidential information;
- fabricating historical vintages;
- claiming certainty or guaranteed profit;
- live trading before production gates pass.

## Success criteria

### Research success

- target parity and as-of semantics proven;
- champion improves locked-test CRPS and contract log loss;
- calibration remains reliable;
- gains appear across multiple years and regimes;
- effect is operationally reproducible.

### Trading research success

- forecast edge survives bid/ask, fees, slippage, latency, and conservative fills;
- capacity is measured;
- risk is bounded;
- live shadow confirms assumptions.

### Failure is informative

A well-run null experiment is successful research because it reduces the search space and prevents repeated effort. Every null or rejected hypothesis must document what was learned.
