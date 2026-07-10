# Research Backlog — 160 Falsifiable Candidate Experiments

These are hypotheses, not findings. Each promoted item must receive its own `EXP-####` folder, predeclaration, point-in-time audit, and out-of-sample evaluation. Codex should not run them as one kitchen-sink search. The backlog exists to prevent shallow analysis and repeated rediscovery.

## Target, settlement, and measurement truth

**B001.** First-publication parity: quantify whether the first Daily Extract value ever differs from the latest Daily Extract or CLMMAXT value.

**B002.** Publication-lag distribution: model when the authoritative field first appears and whether lag depends on weekends, missing data, or unusual weather.

**B003.** Aggregation convention audit: prove whether the Daily Extract maximum is derived from one-minute means and identify historical convention changes.

**B004.** Decimal-boundary audit: enumerate all historical values ending in .9 or .0 and verify bucket mapping against resolved markets.

**B005.** Missing/trace value handling: identify every nonnumeric completeness code and determine contract behavior without inference.

**B006.** Station identity timeline: prove the physical observing site and metadata effective dates behind station code HKO.

**B007.** Instrument/exposure breaks: test documented metadata changes for step shifts in Tmax residuals versus neighbors.

**B008.** Daily-date boundary: verify local-day semantics around 00:00 HKT and late-night maxima.

**B009.** Rules-template drift: semantic-diff every Hong Kong market’s source, field, precision, revision, and fallback wording.

**B010.** Resolved-winner regression suite: build permanent fixtures for every verifiable historical event and every boundary class.

## Cutoff and information-arrival optimization

**B011.** Compare H39, H27, H24N, and H15 using identical baseline families and point-in-time vintages.

**B012.** Measure actual NWP file availability at each cutoff rather than using initialization time.

**B013.** Estimate market liquidity formation by minutes since event creation and hours before nominal peak.

**B014.** Optimize cutoff using forecast skill minus conservative spread/fee/capacity cost, predeclared before test.

**B015.** Test a two-stage strategy: early H39 prior plus H24N update without using later information in early evaluation.

**B016.** Quantify whether official HKO forecast updates arrive just before or after candidate cutoffs.

**B017.** Measure cutoff sensitivity to a conservative 5/15/30/60-minute data-latency buffer.

**B018.** Test whether daylight-saving changes in the operator’s timezone affect operational reliability, while keeping HKT target fixed.

**B019.** Measure missing-cycle risk by cutoff and define source-completeness eligibility.

**B020.** Select a secondary earlier horizon that preserves most skill while gaining more market lead time.

## Climatology, trend, and temporal structure

**B021.** Compare 5/10/15/20/30-year rolling climatologies against full-history climatology.

**B022.** Fit calendar harmonics with robust local trend using training-only estimation.

**B023.** Test quantile-specific climate trend rather than a single mean trend.

**B024.** Test whether urban heat trend differs by season and wind regime.

**B025.** Estimate day-of-year distribution with adaptive bandwidth near monsoon transitions.

**B026.** Test anomaly persistence from T-1 Tmax after conditioning on forecast synoptic change.

**B027.** Test multi-day heat accumulation and nighttime retention as predictors of next-day model bias.

**B028.** Identify changepoints in HKO-minus-rural-station Tmax differences.

**B029.** Test recency weighting selected only in rolling validation.

**B030.** Build a regime-conditioned climatological distribution as the hardest non-NWP baseline.

## NWP point extraction and station translation

**B031.** Compare nearest grid, bilinear interpolation, land-only grid, and multi-grid learned station translation.

**B032.** Correct model grid elevation with physically constrained lapse-rate alternatives.

**B033.** Test whether daily model Tmax or hourly trajectory yields better station calibration.

**B034.** Estimate model/cycle/lead/month-specific additive bias.

**B035.** Test robust linear MOS using model Tmax, dew point, wind, and cloud.

**B036.** Test quantile MOS for heteroskedastic and skewed errors.

**B037.** Compare raw IFS, AIFS, GFS, GEFS, ICON, and ICON-EPS on common vintages.

**B038.** Test dynamic recent-error correction with strictly trailing windows.

**B039.** Decompose model error into regional field bias and HKO local residual.

**B040.** Test whether model upgrade/version indicators require separate calibration regimes.

## Vertical thermodynamics and boundary layer

**B041.** Translate 925-hPa potential temperature to surface Tmax under regime-specific mixing depth.

**B042.** Compare 950, 925, 900, and 850-hPa temperatures as thermal ceilings.

**B043.** Test inversion strength and base height as modifiers of realized surface heating.

**B044.** Estimate a dry-adiabatic mixed-down temperature potential from full forecast profiles.

**B045.** Test dew-point depression as a constraint on sensible heating and mixing.

**B046.** Use forecast boundary-layer height trajectory to predict model station bias.

**B047.** Test entrainment warming/drying potential above the boundary layer.

**B048.** Classify capped versus freely mixed days and fit separate residual distributions.

**B049.** Test forecast sounding analogues using temperature, humidity, and wind profiles.

**B050.** Combine thermal potential with cloud-free duration and onshore-flow probability.

## Station-network spatial signal

**B051.** Test prior-day HKO-minus-King’s Park anomaly persistence conditional on wind regime.

**B052.** Build urban, harbour, coastal, inland, and northern station composites.

**B053.** Fit training-only PCA of station temperature anomalies and test mode persistence.

**B054.** Construct graph-smoothed network state using distance, bearing, elevation, and land class.

**B055.** Test north–south gradient as an air-mass boundary indicator.

**B056.** Test east–west gradient as a marine-intrusion/Pearl River signal.

**B057.** Use prior-night cooling-rate contrasts to forecast next-day urban heating residual.

**B058.** Use prior-day peak-time differences across stations to classify ventilation regime.

**B059.** Estimate wind-conditioned optimal neighbor weights without future rows.

**B060.** Test source-reliability weights based on trailing station anomalies and missingness.

## Wind, sea breeze, harbour, and terrain

**B061.** Project wind onto local onshore/offshore coastline normals at HKO and upwind coasts.

**B062.** Test forecast sea-breeze onset time as a cap on Tmax.

**B063.** Build a pressure-gradient proxy across Pearl River Delta and eastern coast.

**B064.** Test northerly/offshore flow interaction with 925-hPa thermal potential.

**B065.** Test easterly marine flow interaction with humidity and low-cloud forecast.

**B066.** Estimate trajectory-based air-mass origin and travel time from operational forecast winds.

**B067.** Test terrain-channelled flow regimes through Victoria Harbour.

**B068.** Compare surface and 950/925-hPa wind-direction shear as a mixing/sea-breeze indicator.

**B069.** Test coastal–inland model temperature gradient as a better predictor than point wind alone.

**B070.** Build a probabilistic ventilation index combining wind speed, direction persistence, and marine contrast.

## Cloud, radiation, aerosol, and visibility

**B071.** Use hourly forecast cloud timing instead of daily mean cloud cover.

**B072.** Decompose low/mid/high cloud impacts on heating by solar elevation.

**B073.** Integrate forecast shortwave radiation only over pre-peak hours.

**B074.** Test ensemble probability of an uninterrupted two-hour sunny window.

**B075.** Estimate model-specific radiation-to-temperature conversion efficiency.

**B076.** Use prior-day observed radiation efficiency as a trailing bias state.

**B077.** Test aerosol/visibility proxies as modifiers of realized radiation conditional on cloud.

**B078.** Use satellite cloud field at the pre-event cutoff only where scan availability is proven.

**B079.** Test upstream cloud advection features from point-in-time satellite and forecast winds.

**B080.** Combine cloud timing uncertainty with vertical thermal ceiling to model right-tail Tmax.

## Rain, convection, wetness, and recovery

**B081.** Compare any-rain probability with station-hit probability for HKO.

**B082.** Separate convective and stratiform precipitation regimes.

**B083.** Use ensemble rain-start time distribution before nominal peak.

**B084.** Test antecedent rainfall and wet-surface proxies as next-day heating suppressors.

**B085.** Model post-rain recovery potential from wind, humidity, and radiation forecast.

**B086.** Test CAPE/CIN plus forcing as a predictor of early convective cutoff.

**B087.** Use spatial precipitation coverage rather than gridpoint precipitation alone.

**B088.** Test model disagreement in rain timing as a predictor of Tmax distribution width.

**B089.** Build a no-rain/early-rain/late-rain mixture distribution.

**B090.** At live horizons only, quantify radar/lightning/cloud-top incremental value over max-so-far.

## Synoptic regime and regional context

**B091.** Objectively classify subtropical-ridge, monsoon-trough, front/surge, moist-southerly, and weak-gradient regimes.

**B092.** Measure HKO model bias by ridge-axis position and strength.

**B093.** Test 850-hPa warm advection from inland southern China.

**B094.** Use regional geopotential/pressure field analogues rather than a single point.

**B095.** Test forecast subsidence and vertical velocity as clear-heating indicators.

**B096.** Measure monsoon onset/withdrawal transition effects on error distribution.

**B097.** Test regional moisture-flux convergence as convection/rain timing signal.

**B098.** Use operational southern-China station observations available before cutoff as model initial-state diagnostics.

**B099.** Test synoptic-regime persistence versus forecast regime transition.

**B100.** Build regime-specific expert weights with minimum-sample shrinkage.

## Tropical-cyclone and extreme-weather context

**B101.** Use contemporaneous advisory distance, bearing, quadrant, and intensity—not final best track.

**B102.** Test subsident-side warming conditional on cyclone motion and environmental flow.

**B103.** Test track-ensemble uncertainty as a predictor of Tmax uncertainty.

**B104.** Separate pre-cyclone heat, cyclone rain/wind, and post-cyclone regimes.

**B105.** Measure official forecast correction relative to raw NWP during cyclone contexts.

**B106.** Use warning-signal state only if issued before cutoff and test incremental value.

**B107.** Test pressure tendency and outer-subsidence diagnostics.

**B108.** Compare forecast-track analogues from prior storms using only contemporaneous forecasts.

**B109.** Quantify model-specific track/Tmax joint error.

**B110.** Create a conservative cyclone fallback when historical sample is insufficient.

## Forecast-vintage dynamics and model disagreement

**B111.** Use run-to-run change in station-translated Tmax.

**B112.** Use field-level pattern change rather than point forecast change.

**B113.** Test ensemble-spread trend across successive cycles.

**B114.** Measure threshold-crossing member counts and their change.

**B115.** Decompose cross-model disagreement into amplitude, timing, cloud, rain, and wind components.

**B116.** Test whether convergence predicts smaller error while divergence widens distribution.

**B117.** Build trailing, regime-specific model reliability scores.

**B118.** Test official HKO forecast change relative to model-consensus change.

**B119.** Detect stale guidance when one model cycle is missing or delayed.

**B120.** Weight models by recent analogous-regime skill with shrinkage toward equal weights.

## Analogues, residual structure, and expert combinations

**B121.** Nearest analogues in forecast vertical-profile space.

**B122.** Nearest analogues in regional geopotential/temperature field space.

**B123.** Nearest analogues in station-network state plus forecast wind.

**B124.** Hybrid analogues requiring season and regime compatibility.

**B125.** Local residual analogues conditioned on raw model error from prior day.

**B126.** Test constrained linear blend of official, climatology, and NWP experts.

**B127.** Test Bayesian model averaging with trailing likelihood and shrinkage.

**B128.** Test inverse-CRPS weights versus equal weights.

**B129.** Fit a transparent mixture-of-regimes distribution.

**B130.** Quantify each expert’s marginal contribution with leave-one-out ablation.

## Probabilistic forecasting and calibration

**B131.** Compare Gaussian residual, Student-t, skew-normal, empirical, and mixture residual distributions.

**B132.** Build a discrete 0.1°C probability mass function aligned to source precision.

**B133.** Test rolling empirical residual calibration windows of 90/180/365/730 days.

**B134.** Test regime-conditional calibration with hierarchical shrinkage.

**B135.** Evaluate quantile mapping with monotonic repair.

**B136.** Test conformal interval coverage under seasonal nonexchangeability.

**B137.** Calibrate integer-threshold crossing probabilities directly.

**B138.** Compare isotonic, beta, and multinomial/Dirichlet calibration where sample supports it.

**B139.** Diagnose ensemble underdispersion and test ensemble dressing.

**B140.** Optimize calibration for CRPS while guarding multiclass log loss and tails.

## Reliability, negative controls, and falsification

**B141.** Inject a future model cycle and verify the as-of validator rejects it.

**B142.** Inject finalized ERA5 as an operational feature and verify role gating rejects it.

**B143.** Shuffle target labels and confirm all apparent skill disappears.

**B144.** Shift features by +1 day and confirm any gain is flagged as leakage.

**B145.** Add random noise columns and verify feature selection does not promote them on locked data.

**B146.** Compare common-sample versus all-available metrics to expose missingness bias.

**B147.** Repeat accepted results across alternate historical windows.

**B148.** Run leave-one-year-out influence analysis.

**B149.** Run leave-one-regime-out transfer tests.

**B150.** Stress timestamps with timezone and daylight-boundary fixtures.

## Later-stage machine learning

**B151.** Regularized linear distributional model over accepted features only.

**B152.** Generalized additive model for nonlinear but interpretable physical relationships.

**B153.** Monotonic gradient boosting with physically justified constraints.

**B154.** Quantile gradient boosting under nested rolling validation.

**B155.** Distributional boosting for mean, scale, and skew.

**B156.** Graph model over station network after sufficient sub-daily vintage history.

**B157.** Sequence model over NWP run-to-run evolution after classical dynamics baseline.

**B158.** Spatial encoder for forecast/radar/satellite fields with strict frame cutoffs.

**B159.** Mixture-of-experts gating learned from preobservable regimes.

**B160.** Knowledge-distilled compact production model from a validated advanced challenger.

## Market probability, execution, and risk

**B161.** Verify exact token-to-bucket mapping for every event.

**B162.** Compare model probabilities to executable ask/bid rather than midpoint.

**B163.** Estimate taker fee and slippage by price and size.

**B164.** Estimate maker fill probability conditional on quote age and subsequent book path.

**B165.** Measure adverse selection after maker fills.

**B166.** Model joint state-contingent payoff across all mutually exclusive bucket positions.

**B167.** Test conservative edge thresholds including forecast uncertainty reserve.

**B168.** Build capacity curves by expected edge and depth.

**B169.** Evaluate market-implied distribution as a benchmark and later optional expert.

**B170.** Run immutable live paper orders before any production eligibility decision.

## Prioritization rule

Rank candidates by expected information gain × operational availability × mechanism strength ÷ overfitting risk and implementation cost. Target/as-of/baseline dependencies always dominate.

## Backlog state

When an item is tested, add its experiment ID and outcome beside it in a new commit rather than deleting the hypothesis. Null findings are retained.
