# Feature and Mechanism Map

This document is a hypothesis inventory, not a license to create one giant feature matrix. Every family requires separate point-in-time and out-of-sample testing.

## 1. Seasonal and long-term state

Candidate features:

- day-of-year harmonic terms;
- rolling modern climatology;
- climate trend and trend uncertainty;
- prior 7/14/30-day temperature anomaly;
- prior-day and multi-day Tmax/Tmin;
- seasonal transition speed;
- regime-specific climatology;
- structural-break indicators.

Questions:

- What historical window best represents the current urban station?
- Does trend correction improve tails without destabilizing calibration?
- Are leap-day and calendar-edge effects handled correctly?

Failure risks:

- old urban exposure differs from today;
- trend estimated with future data;
- calendar features proxy test period.

## 2. Official HKO forecast information

Candidate features:

- official forecast maximum;
- forecast range and text;
- issue/update time;
- forecast change since previous issue;
- confidence/probability guidance where available;
- automatic regional forecast at HKO and neighbors;
- discrepancy between human/official and model consensus.

Hypotheses:

- official guidance contains local expert corrections unavailable in raw global NWP;
- direction and size of official update contain information;
- disagreement between official forecast and model consensus identifies difficult regimes.

Controls:

- archive authentic vintage;
- do not scrape latest forecast as history;
- text parser versioned and manually validated.

## 3. NWP surface forecast

Per model/cycle/member:

- 2-m temperature trajectory;
- model daily Tmax;
- dew point/RH;
- 10-m wind and gust;
- pressure/MSLP;
- total/low/mid/high cloud;
- shortwave/longwave radiation;
- precipitation and convective precipitation;
- boundary-layer height;
- soil temperature/moisture where relevant;
- surface fluxes;
- visibility/aerosol fields if available.

Derived:

- peak temperature and timing;
- morning heating slope;
- duration above thresholds;
- integrated shortwave before peak;
- cloud-free window probability;
- ensemble crossing probabilities;
- HKO-grid versus surrounding-grid gradients;
- land/sea grid contrast.

## 4. Vertical thermodynamics

Levels/profile:

- temperature and dew point at 1000/975/950/925/900/850 hPa;
- potential temperature;
- wet-bulb potential temperature;
- lapse rates;
- inversion base/depth/strength;
- precipitable water;
- mixing ratio;
- CAPE/CIN;
- lifted index;
- vertical motion;
- tropopause/subsidence proxies.

Physical diagnostics:

- mixed-down temperature potential under assumed boundary-layer depth;
- dry/moist adiabatic surface potential;
- entrainment warming/drying;
- capping inversion persistence;
- convective initiation timing;
- profile analogue distance.

## 5. Synoptic dynamics

- subtropical ridge axis and geopotential anomalies;
- pressure-gradient vectors;
- monsoon trough distance/orientation;
- frontal/surge probability;
- 850-hPa advection;
- vorticity/divergence;
- vertical velocity/subsidence;
- moisture transport;
- air-mass back trajectories;
- southern China inland heat reservoir;
- Pearl River Delta thermal contrast.

The diagnostic must be computable from forecast fields available at cutoff, not final analysis.

## 6. Sea breeze, coast, and terrain

- onshore/offshore wind component relative to local coastline;
- forecast wind direction transitions;
- pressure differences across Pearl River Delta/coast;
- coastal–inland temperature gradient;
- marine layer depth;
- sea-surface minus air-temperature contrast;
- terrain-aligned wind components;
- harbour-channel flow;
- elevation-adjusted upwind temperature;
- sea-breeze onset probability and ensemble spread.

Hypothesis examples:

- offshore/northerly flow allows HKO to realize a larger fraction of 925-hPa thermal potential;
- an early easterly/onshore transition caps Tmax;
- network gradients at T-1 encode persistent air-mass positioning.

## 7. Station-network state

At or before cutoff:

- HKO anomaly relative to each neighbor;
- urban mean versus rural mean;
- harbour/coastal mean versus inland mean;
- north–south/east–west gradients;
- station covariance modes;
- lagged daytime heating curves;
- prior-night cooling rates;
- prior-day peak time and post-peak decay;
- quality/missingness pattern;
- wind-conditioned neighbor weights;
- graph features based on distance, bearing, elevation, land use.

Use station timelines. A station code is not guaranteed homogeneous over decades.

## 8. Radiation, clouds, and aerosols

- forecast cloud by layer and hour;
- cloud timing entropy across members;
- forecast global/direct/diffuse radiation;
- prior-day observed radiation efficiency;
- satellite-derived cloud state available at cutoff;
- upstream cloud fields;
- aerosol optical depth/air-quality proxy;
- visibility/haze;
- cloud-base and cloud-top proxies.

At H24N, target-day observed radiation is unavailable and forbidden. At later live horizons it becomes eligible only up to cutoff.

## 9. Rain and convection

- probability of any rain before nominal peak;
- probability HKO specifically is hit;
- expected rain start/end;
- ensemble precipitation coverage;
- convective versus stratiform classification;
- CAPE/CIN and forcing;
- prior-day wetness/rain history;
- antecedent soil/wet-surface proxies;
- model disagreement and neighborhood rain probability.

For live models later:

- radar-cell distance/motion/growth;
- rainfall nowcast at HKO;
- lightning density;
- cloud-top cooling;
- station rain onset;
- post-rain recovery potential.

## 10. Tropical-cyclone context

Point-in-time only:

- advisory storm center/intensity;
- forecast-track ensemble;
- distance/bearing and quadrant;
- expected closest approach;
- warning signal;
- pressure/wind anomaly;
- subsidence-side probability;
- track uncertainty;
- forecast movement.

Never use final best track as if known beforehand.

## 11. Forecast-vintage dynamics

- model run-to-run Tmax change;
- spatial-field change, not just point value;
- ensemble mean/spread trend;
- member threshold crossings;
- cross-model convergence;
- cycle-specific historical bias;
- actual delivery completeness at cutoff;
- official forecast response to model changes;
- stale or missing run indicators.

A forecast jump may be informative even if the absolute forecast is biased.

## 12. Analogues

Distance spaces:

- forecast vertical profile;
- synoptic fields;
- station-network state;
- wind/pressure/radiation trajectory;
- model consensus/spread;
- season.

Controls:

- analogues selected from prior dates only;
- distance transform fit on training only;
- avoid outcome-informed feature weighting;
- test analogue stability and effective sample size.

## 13. Target-distribution structure

- residual distribution by regime;
- heteroskedasticity by model spread/cloud/rain;
- skew under hot subsidence versus rainy regimes;
- mixture components;
- boundary crossing probability;
- tail thickness;
- calibration window adaptation.

## 14. Market information as a later research feature

Only after meteorological validation:

- implied bucket distribution;
- market entropy;
- spread/depth;
- price movement;
- disagreement between model and market;
- time since event creation.

Use a separate experiment because market information can obscure whether the weather system itself improved.

## 15. Feature promotion checklist

- [ ] lawful source;
- [ ] timestamp semantics proven;
- [ ] available at primary cutoff;
- [ ] stable schema;
- [ ] plausible mechanism;
- [ ] predeclared experiment;
- [ ] incremental OOS gain;
- [ ] robust to alternate windows/specifications;
- [ ] ablation confirms contribution;
- [ ] production latency acceptable;
- [ ] missing-data fallback exists.
