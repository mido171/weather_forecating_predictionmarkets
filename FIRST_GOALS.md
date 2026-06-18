# First Goal Program — HKG Official Tmax at Pre-Event Cutoff

This is the ordered execution plan for Codex. Do not skip dependencies. Each numbered goal must produce its specified artifacts and pass its acceptance criteria before downstream claims can be promoted.

---

## Program objective

Produce a calibrated, leakage-free probability distribution for the contract-authoritative Hong Kong daily maximum temperature on local date T, using only information genuinely available at a frozen cutoff approximately 24 hours before the likely daytime maximum. Then map that distribution into exact Polymarket contract buckets and evaluate whether any forecast advantage survives executable market costs.

The first research phase is **not** “train the fanciest model.” It is to build the truth, time, archive, and baseline infrastructure that makes later discoveries trustworthy.

---

# G0 — Prove the repository and archive work end to end

## Purpose

Establish that the environment is reproducible and can archive a source without changing or losing its raw content.

## Actions

1. Run:
   ```bash
   make doctor
   make test
   make validate
   ```
2. Create `.env` from `.env.example`.
3. Fetch each source currently marked `bootstrap_now` in `config/data_sources.yaml`.
4. Confirm every snapshot has:
   - raw bytes;
   - SHA-256;
   - retrieval timestamp;
   - URL and request metadata;
   - HTTP status and headers;
   - sidecar JSON;
   - no in-place overwrite.
5. Run the same fetch twice and verify both retrieval events are traceable.
6. Generate the source inventory report.
7. Create `EXP-0001` documenting this smoke test.

## Deliverables

- archived raw snapshots in `data/raw/`;
- `reports/source_inventory.md`;
- completed experiment folder;
- all tests passing;
- first immutable Git commit.

## Acceptance criteria

- clean bootstrap succeeds on Linux or WSL;
- tests pass without manually editing source files;
- raw hashes can be independently recomputed;
- malformed, empty, or HTTP-error payloads fail loudly;
- no secret is committed.

---

# G1 — Establish exact settlement truth and historical-label parity

## Purpose

Remove the single most dangerous ambiguity: what exact number the contract resolves from and whether the downloadable `CLMMAXT station=HKO` series is a valid historical proxy.

## Hypotheses to test

1. The contract-authoritative field is the first-published HKO Daily Extract `Absolute Daily Max (deg. C)` for Hong Kong local date T.
2. The machine-readable `CLMMAXT` series for station `HKO` equals that first-published value on all ordinary dates.
3. Differences, if any, are attributable to later revisions, missing data, date handling, or publication mechanics.

## Actions

1. Archive the full event metadata and rules for at least:
   - all available resolved Hong Kong Tmax events;
   - the current event;
   - a stratified sample across seasons and tail outcomes.
2. Hash normalized rules text and enumerate:
   - named source;
   - named field;
   - precision;
   - date/time zone;
   - bucket definitions;
   - revision language;
   - fallback language.
3. Archive Daily Extract pages/payloads at first publication whenever possible.
4. Download the official HKO `CLMMAXT` history for station `HKO`.
5. Build a date-level parity table:
   ```text
   local_date
   daily_extract_first_published
   daily_extract_latest
   clmmaxt_value
   polymarket_winner
   exact_match flags
   timestamps
   source hashes
   notes
   ```
6. Validate resolved event winners by applying the rules-derived bucket map to the authoritative decimal value.
7. Investigate every mismatch manually and programmatically.
8. Implement a production target adapter that fails closed when:
   - rules hash is unknown;
   - field cannot be found;
   - date is ambiguous;
   - precision changes;
   - source is unavailable;
   - bucket coverage overlaps or has gaps.
9. Add regression fixtures for real resolved events.

## Deliverables

- `data/gold/target_parity/`;
- `reports/target_parity.md`;
- parsed rules schema;
- real-event settlement fixtures;
- mismatch ledger;
- target adapter tests;
- completed G1 experiment(s).

## Acceptance criteria

- 100% agreement between computed and actual winners for every verifiable resolved event;
- quantified parity between first-published Daily Extract and `CLMMAXT`;
- all mismatches resolved or explicitly quarantined;
- no future training label is called canonical without evidence;
- rules-change monitor is operational.

## Stop condition

If target parity cannot be established, predictive modelling remains blocked.

---

# G2 — Select and freeze the operational forecast cutoff

## Purpose

Define “T-24 hours” in a way that matches actual information arrival and market opportunity rather than an arbitrary clock label.

## Candidate cutoffs

- `H39`: T-1 00:00 HKT
- `H27`: T-1 12:00 HKT
- `H24N`: T-1 15:00 HKT
- `H15`: T 00:00 HKT

Also record exact hours to:
- local midnight;
- sunrise;
- nominal 15:00 peak;
- event close;
- first meaningful market liquidity.

## Actions

1. Archive creation/open/close timestamps for all recoverable HKG events.
2. Reconstruct market price/liquidity formation where historical data permits.
3. Build an information-arrival matrix at each candidate cutoff:
   - available HKO forecasts;
   - NWP cycles and actual delivery delays;
   - ensemble products;
   - prior-day observations;
   - satellite/radar eligibility;
   - market prices and book depth.
4. Compare forecast skill and expected executable opportunity at each horizon without tuning model families separately on the final holdout.
5. Choose:
   - one primary research horizon;
   - one secondary earlier horizon;
   - exact cutoff grace/latency rules.
6. Freeze this choice in `config/asof.yaml`.

## Deliverables

- `reports/horizon_selection.md`;
- point-in-time availability matrix;
- market opening/liquidity summary;
- frozen cutoff contract.

## Acceptance criteria

- the primary horizon is operationally observable and reproducible;
- all source-vintage cutoffs include realistic publication latency;
- the selection is based on a predeclared utility that balances forecast skill, early entry, and market capacity;
- no horizon shopping on the locked test.

---

# G3 — Build the complete lawful data inventory and acquire history

## Purpose

Collect every realistically useful source, with explicit provenance and point-in-time limitations.

## Source families

### A. Canonical target and HKO station observations

Acquire or archive:

- HKO Daily Extract first publication;
- `CLMMAXT` and related daily climate elements;
- one-minute temperature;
- max/min since midnight;
- hourly and sub-hourly temperature where lawfully available;
- humidity, dew point, wet-bulb;
- pressure;
- wind speed, direction, gust;
- rainfall;
- sunshine duration;
- global/direct/diffuse solar radiation;
- cloud amount;
- visibility;
- UV/heat indices where available;
- station metadata, instrumentation, relocation, elevation, and observing-practice changes.

### B. Hong Kong spatial network

Acquire every relevant automatic station and its metadata. Prioritize but do not limit to:

- King’s Park;
- Hong Kong Park;
- Kai Tak;
- Kowloon City;
- Sham Shui Po;
- Wong Tai Sin;
- Happy Valley;
- Shau Kei Wan;
- Central Pier;
- Waglan Island;
- Cheung Chau;
- Sha Tin;
- Tai Po;
- Lau Fau Shan;
- Ta Kwu Ling;
- Sheung Shui;
- Sai Kung;
- Tseung Kwan O;
- airport and outlying-island stations.

Preserve station life cycles and avoid treating relocated or changed instruments as homogeneous.

### C. Radar, rainfall nowcasts, lightning, satellite

Archive:

- all available HKO radar ranges/products at native cadence;
- gridded rainfall nowcasts and issue times;
- lightning strokes/counts;
- geostationary satellite channels and derived cloud products;
- cloud masks, cloud-top temperature, optical depth where licensed;
- image metadata and exact georeferencing.

### D. Upper-air, marine, and synoptic observations

Acquire:

- King’s Park soundings;
- surface synoptic observations over southern China;
- buoys, ships, sea-surface temperature;
- tides and coastal winds where potentially relevant;
- tropical-cyclone advisories, warning signals, forecast tracks, and only point-in-time track data;
- fronts/troughs/subtropical-ridge diagnostics.

### E. Numerical and AI weather forecasts

Archive every run and member available under applicable terms:

- ECMWF IFS and AIFS open data;
- NOAA GFS and GEFS;
- DWD ICON and ICON-EPS;
- other lawfully accessible operational global/regional models;
- HKO official forecasts, automatic regional forecasts, probability guidance, and updates.

For each forecast value preserve:

```text
model
cycle / initialization
member
valid_time
parameter
level
grid
download_time
expected_release_time
actual_available_time
file hash
processing version
```

Never overwrite a newer run over an older vintage.

### F. Reanalysis and retrospective products

Acquire for mechanism discovery and context:

- ERA5;
- ERA5-Land;
- appropriate satellite/reanalysis products.

Mark them `RETROSPECTIVE_ONLY` unless a historically accurate release-lag simulation is implemented. They may explain mechanisms but may not enter operational backtests as real-time predictors by default.

### G. Polymarket

Archive:

- exact event/market metadata;
- rules and rules hash;
- outcome token IDs;
- bucket boundaries;
- prices;
- trades;
- order-book snapshots and deltas;
- tick sizes;
- fee parameters;
- liquidity/volume metadata;
- resolution events.

### H. Calendar and deterministic context

Generate:

- solar geometry, sunrise/sunset/solar elevation;
- day-of-year;
- weekday/holiday only if a defensible urban heat mechanism is proposed;
- terrain, land cover, coastline orientation, station distance/bearing/elevation;
- no future-dependent calendar fields.

## Actions

1. Complete every row of `config/data_sources.yaml`.
2. Verify licensing, terms, access method, cadence, historical range, latency, and revision behavior.
3. Implement source adapters one at a time.
4. Backfill available history.
5. Start continuous live archival immediately for sources lacking historical vintages.
6. Generate coverage heat maps and missingness reports.
7. Tag each source:
   - `OPERATIONAL_POINT_IN_TIME`;
   - `PROXY_WITH_LIMITATIONS`;
   - `RETROSPECTIVE_ONLY`;
   - `TARGET_ONLY`;
   - `MARKET_ONLY`.
8. Document unavailable or paid data and the expected value of acquiring it.

## Deliverables

- populated raw archive;
- source contracts and schemas;
- coverage matrix;
- license/provenance ledger;
- point-in-time eligibility report;
- acquisition scripts and monitors.

## Acceptance criteria

- no feature source lacks a provenance and timestamp contract;
- no retrospective source is silently used operationally;
- all mutable live sources are being archived;
- critical-source gaps are explicit.

---

# G4 — Audit data quality, station history, and temporal integrity

## Purpose

Prevent false signal from sensor changes, missingness, aggregation differences, timestamp errors, or urbanization trends.

## Actions

1. Build station metadata timelines:
   - location;
   - elevation;
   - instrument;
   - screen/exposure;
   - cadence;
   - relocation;
   - maintenance;
   - known breaks.
2. Test:
   - duplicate timestamps;
   - impossible values;
   - stuck sensors;
   - jumps;
   - unit changes;
   - daylight/time-zone errors;
   - daily aggregation consistency;
   - max-of-minute versus hourly max differences;
   - publication delays and revisions.
3. Compare HKO with neighboring stations for structural breaks.
4. Quantify long-term nonstationarity:
   - warming trend;
   - urban heat evolution;
   - seasonality shift;
   - typhoon/rain-regime changes.
5. Define imputation rules that never borrow future data.
6. Create data quality flags, not silent corrections.
7. Determine which historical years are comparable to the current station regime.
8. Establish sample weighting or rolling windows based on evidence.

## Deliverables

- station history dossier;
- quality-control rules;
- anomaly ledger;
- temporal break analysis;
- operational feature eligibility table.

## Acceptance criteria

- every retained observation has a quality state;
- corrections are versioned and reversible;
- no interpolation crosses the forecast cutoff;
- sensitivity to historical window and structural breaks is quantified.

---

# G5 — Establish hard-to-beat leakage-safe baselines

## Purpose

Create honest benchmarks before searching for sophistication.

## Required baselines

1. Calendar climatology with trend and uncertainty.
2. Recent-year climatology.
3. Analog climatology conditioned on prior-day state.
4. Persistence/anomaly persistence.
5. HKO official maximum forecast at the exact vintage.
6. Raw deterministic NWP for the HKO grid/station.
7. Raw ensemble distribution.
8. Bias-corrected model output statistics by model, cycle, lead, month, and regime.
9. Equal-weight and skill-weighted multi-model consensus.
10. Simple transparent blend of climatology + official forecast + ensemble.

## Evaluation

At each frozen horizon report:

- bias;
- MAE;
- median absolute error;
- RMSE;
- CRPS;
- interval coverage and width;
- bucket probabilities;
- multiclass log loss;
- Brier scores;
- calibration/reliability;
- sharpness;
- extreme/boundary-day performance;
- block-bootstrap confidence intervals;
- Diebold–Mariano or appropriate paired comparisons with caveats.

Break down by:

- month/season;
- forecast lead;
- temperature decile;
- rain/cloud regime;
- wind sector;
- tropical-cyclone context;
- model disagreement;
- year;
- station/data-quality state.

## Split policy

Use rolling-origin evaluation. The bootstrap configuration proposes:

- development: 2000-01-01 through 2022-12-31;
- validation: 2023-01-01 through 2024-06-30;
- locked test: 2024-07-01 through 2026-05-31;
- live shadow: 2026-06-18 onward.

This split is provisional because authentic forecast vintages may cover a shorter history. Only dates with valid point-in-time inputs enter operational comparisons. Once the actual vintage coverage is known, freeze revised periods before model selection.

## Acceptance criteria

- baseline code is deterministic;
- all predictions are stored row-by-row;
- no target-period information influences training or calibration;
- baseline scoreboard is populated;
- one champion baseline is frozen for G6.

---

# G6 — Exhaustive classical and physical mechanism experiments

## Purpose

Extract interpretable predictive signal before ML.

Each item below is a family of separate experiments, not one uncontrolled kitchen-sink test.

## Experiment families

### 1. Station-specific model bias

Study residuals as functions of:

- model/cycle/member/lead;
- month and synoptic regime;
- forecast Tmax level;
- cloud/rain bias;
- wind direction/speed;
- dew point and boundary-layer depth;
- recent model error;
- grid elevation/land-sea mismatch;
- urban-versus-grid thermal contrast.

### 2. Spatial station-network fingerprints

Test whether HKO Tmax residual is predicted by:

- contemporaneous and lagged HKO-minus-neighbor anomalies available at cutoff;
- inland/coastal/urban station contrasts;
- north–south and east–west temperature gradients;
- wind-conditioned station weighting;
- principal spatial modes;
- analog days based on network pattern;
- previous-day heating and cooling trajectories;
- nighttime urban heat retention.

### 3. Solar and cloud-heating budget

Investigate:

- forecast and observed prior-day radiation;
- cloud fraction by layer;
- cloud timing rather than daily mean;
- radiation-to-temperature conversion by humidity/wind regime;
- morning cloud breakup analogues;
- aerosol/haze interactions;
- accumulated positive temperature tendency under clear intervals.

At a 24-hour cutoff, use only forecasts and prior observations, never day-T realized radiation.

### 4. Boundary-layer and thermodynamic controls

Investigate:

- 925/950/850 hPa temperature;
- lapse rates and inversion strength;
- mixing depth;
- dry-adiabatic potential temperature translation;
- dew point and wet-bulb constraints;
- entrainment of warm/dry air;
- CAPE/CIN and convective timing;
- nighttime minimum as a conditional anchor only when available at the chosen cutoff.

### 5. Wind, sea breeze, and terrain

Investigate:

- onshore/offshore wind components;
- pressure gradients;
- sea-breeze onset probability;
- channeling around terrain and Victoria Harbour;
- coastal–inland gradients;
- upwind source regions;
- trajectory-based air-mass classification;
- model errors conditional on wind transition.

### 6. Rain and convection

Investigate:

- probability and timing of rainfall;
- rain coverage versus station-specific hit probability;
- convective versus stratiform regimes;
- post-rain recovery;
- model ensemble precipitation disagreement;
- nearby-day soil/wetness proxies;
- radar/satellite only at horizons where they are legitimately available.

### 7. Synoptic and tropical-cyclone regimes

Investigate:

- subtropical-ridge position/strength;
- monsoon trough;
- fronts/surges;
- tropical-cyclone distance, quadrant, intensity, track uncertainty;
- subsidence and föhn-like warming;
- warning/advisory state available at cutoff;
- analogues built from point-in-time forecast tracks, not final best tracks.

### 8. Forecast-vintage dynamics

Investigate:

- run-to-run trend;
- cross-model convergence/divergence;
- ensemble spread changes;
- systematic correction from specific cycles;
- forecast jump signals;
- official-forecast change versus model change;
- stale guidance detection.

### 9. Temporal analogues and nearest-neighbor methods

Build physically constrained analogues using:

- season;
- synoptic fields;
- forecast profiles;
- station-network state;
- air-mass trajectory;
- model consensus and spread.

Ensure the analogue search excludes future dates and target-day realized fields.

### 10. Distribution and calibration

Compare:

- empirical residual distributions;
- regime-conditional residuals;
- quantile mapping;
- Gaussian/non-Gaussian mixtures;
- conformal intervals with time-aware calibration;
- isotonic, beta, and multinomial calibration for buckets;
- tail and threshold calibration.

### 11. Negative controls

Mandatory examples:

- random future-shifted features should trigger leakage tests;
- irrelevant deterministic columns should not improve locked performance;
- shuffled targets should eliminate signal;
- finalized reanalysis used without lag should be flagged;
- future model cycles should fail as-of validation.

## Promotion rule

A feature family is promoted only if:

- mechanism is plausible;
- improvement is out of sample;
- confidence interval is meaningful;
- effect survives nearby specifications;
- ablation confirms incremental value;
- no subgroup suffers unacceptable degradation;
- data can be obtained reliably in production.

---

# G7 — Build the transparent expert probabilistic stack

## Purpose

Combine accepted classical signals into a robust champion before ML.

## Components

- regime classifier derived without target leakage;
- model-specific bias correction;
- station-network correction;
- physical temperature-potential diagnostic;
- official forecast benchmark;
- multi-model ensemble;
- empirical residual distribution;
- calibration layer;
- fallback hierarchy for missing sources.

## Combination candidates

- constrained linear blend;
- Bayesian model averaging;
- inverse-CRPS weights;
- regime-dependent weights estimated with strict rolling windows;
- robust stacking with regularization;
- mixture distribution with transparent component attribution.

## Required diagnostics

- contribution by component;
- weight stability over time;
- sensitivity to missing models;
- calibration drift;
- year-by-year and regime-by-regime performance;
- threshold/bucket reliability;
- worst-case days and failure taxonomy.

## Acceptance criteria

The expert stack must beat the frozen champion baseline on the locked test in the primary metric, while maintaining or improving calibration and avoiding material tail degradation.

---

# G8 — Machine-learning eligibility and controlled ML program

## Entry gate

ML work is blocked until G1–G7 are complete and the data/archive is mature.

## Permitted first models

Start with models whose behavior can be audited:

- regularized linear/quantile regression;
- generalized additive models;
- monotonic gradient boosting where justified;
- shallow tree ensembles;
- distributional boosting;
- calibrated residual models.

Only later consider:

- deep tabular models;
- sequence models;
- graph models for station networks;
- spatial image encoders for radar/satellite;
- learned forecast postprocessing.

## Controls

- nested rolling-origin tuning;
- embargo/gap where overlapping forecast windows create dependence;
- training-only feature selection;
- no global normalization using future periods;
- frozen preprocessing inside pipelines;
- multiple seeds;
- ablation and permutation tests;
- SHAP/importance stability, not just one plot;
- adversarial leakage audit;
- model simplicity penalty;
- compute and latency accounting.

## Acceptance criteria

An ML model must provide stable incremental value over G7, not merely reproduce it. It must pass the same target, leakage, calibration, robustness, and production gates.

---

# G9 — Translate forecast quality into executable Polymarket value

## Purpose

Determine whether forecast improvements can be monetized after actual trading frictions.

## Actions

1. Convert continuous distribution to exact bucket probabilities using rules-derived boundaries.
2. Verify probabilities sum to one and cover tails.
3. Archive fee parameters per market.
4. Reconstruct or prospectively record:
   - bids/asks;
   - depth;
   - trades;
   - latency;
   - fills;
   - cancellations.
5. Compare against executable bid/ask, not midpoint.
6. Model:
   - taker fees;
   - maker rebates if applicable;
   - slippage;
   - partial fills;
   - adverse selection;
   - inventory across mutually exclusive outcomes;
   - order latency and stale quotes.
7. Define conservative decision thresholds from paper trading, not from desired returns.
8. Evaluate calibration-weighted value, P&L distribution, drawdown, turnover, and capacity.
9. Run shadow/paper trading through multiple weather regimes.

## Acceptance criteria

- positive conservative net expectation on held-out/shadow data;
- performance survives wider spreads/slippage and lower fill rates;
- no dependence on unavailable historical depth;
- risk limits and kill switches are tested;
- no live execution is enabled automatically by this bootstrap.

---

# G10 — Production eligibility, monitoring, and continuous research

## Required production controls

- rules-hash kill switch;
- target-source availability monitor;
- source freshness and schema monitors;
- station anomaly detector;
- model-vintage completeness;
- ensemble disagreement alarm;
- calibration/drift dashboard;
- prediction reproducibility snapshot;
- order-book synchronization checks;
- exposure and daily-loss caps;
- manual emergency stop;
- automatic fallback or no-trade state.

## Continuous research discipline

- maintain live shadow forecasts regardless of trading;
- score every forecast after settlement;
- update milestones only after sufficient samples;
- run periodic champion/challenger reviews;
- quarantine regime shifts;
- preserve all predictions made before outcomes;
- never rewrite a historical forecast;
- document model changes with effective dates.

## Production acceptance

Production remains disabled until all gates in `docs/07_PRODUCTION_GATE.md` are signed off by both leakage and reproducibility reviewers.

---

# Definition of a genuine milestone

A milestone is not “we found a correlation.” It is a reproducible, point-in-time, out-of-sample improvement with:

- exact experiment ID;
- baseline and champion versions;
- date range and sample size;
- primary/secondary metrics;
- absolute and relative delta;
- uncertainty interval;
- regime breakdown;
- leakage audit status;
- reproducibility status;
- operational-data availability;
- failure modes;
- next action.

Every genuine milestone must be entered into `MILESTONES.md` immediately.
