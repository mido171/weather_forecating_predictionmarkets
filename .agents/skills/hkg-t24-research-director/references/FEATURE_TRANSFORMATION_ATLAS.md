# Feature Transformation Atlas

## Governing rule

A transformation is eligible only when every input is available before the canonical T−24 cutoff and every fitted parameter is estimated from prior or fold-training data. Feature names must encode lag or window when omission could conceal leakage.

## Level and lag

For any scalar variable `x`:

- `x_latest_before_cutoff`
- `x_lag1`, `x_lag2`, `x_lag7`, and justified lags
- `x_lag365` for year-over-year state
- previous observation or launch rather than calendar lag when cadence differs.

Do not assume a daily label means the value was published by cutoff.

## Change and rate

- `x_change_1 = x_tminus1 - x_tminus2`
- multi-day change;
- rate per hour/day;
- relative change only where denominator is stable and meaningful;
- signed and absolute change;
- change anomaly versus prior history.

For irregular observations, divide by actual elapsed time.

## Slope, acceleration, and curvature

Use prior-only regression or endpoint slopes:

- short slope;
- medium slope;
- long slope;
- short-minus-long trajectory;
- slope change;
- curvature;
- reversal indicator;
- monotonic-run score.

Fit within each row's past window, never over future targets.

## Rolling location

- mean;
- median;
- trimmed mean;
- exponentially weighted mean;
- minimum and maximum;
- robust center.

All windows are explicitly shifted. For target memory, a feature named `roll14` without the shift contract is rejected.

## Dispersion and volatility

- standard deviation;
- MAD;
- IQR;
- range;
- realized absolute change;
- volatility ratio short/long;
- compression/expansion;
- disagreement across stations or experts.

Volatility often predicts transitions and uncertainty rather than signed residual.

## Seasonal and local anomaly

- value minus station recent baseline;
- value minus causal day-of-year climatology;
- value minus causal month/season baseline;
- percentile versus prior same-season history;
- standardized anomaly using prior variance;
- anomaly drift.

Do not compute climatology using future years. State whether climatology is expanding, rolling, or frozen.

## Rank and relative state

- station rank among peers;
- percentile within own prior history;
- rank change;
- warmest/coolest fraction;
- rank reversal;
- official forecast rank versus recent target distribution.

Ranks can reduce unit/era drift but may discard magnitude; compare both.

## Spell and persistence

- consecutive warm/cool days;
- consecutive residual sign;
- duration above prior-defined percentile;
- dry/wet spell;
- wind-sector persistence;
- pressure-rise duration;
- time since transition or extreme.

Update state only after values become available.

## Threshold and hinge

Use physically justified or prior-derived thresholds:

- weak wind;
- high dew point;
- large coastal-inland spread;
- strong pressure change;
- forecast deviation.

Prefer hinge or smooth sigmoid features over many hard bins when possible. Threshold selection belongs inside training folds.

## Circular wind

For direction θ:

- `wind_u = speed * sin(θ)`
- `wind_v = speed * cos(θ)` under the repository's meteorological convention;
- direction sine/cosine;
- sector;
- circular difference;
- persistence;
- shift;
- onshore/offshore projection using verified coastline/station bearings.

Record calm and missing direction behavior.

## Moisture and heat content

Where inputs and units are safe:

- temperature-dew-point spread;
- dew-point level and change;
- approximate wet-bulb;
- vapor pressure;
- mixing-ratio proxy;
- moist enthalpy proxy;
- network moisture gradient;
- moisture surge.

Document formulas, pressure assumptions, and numerical safeguards.

## Pressure and synoptic state

- pressure level;
- tendency;
- acceleration;
- station gradient;
- pressure anomaly;
- ridge/surge proxy;
- pressure change crossed with wind/moisture/target slope.

Distinguish station pressure and sea-level pressure.

## Spatial features

- pairwise spread;
- distance-normalized gradient;
- bearing projection;
- group robust mean;
- coastal-inland contrast;
- north-south/east-west gradient;
- station disagreement;
- inverse-distance HKO estimate;
- upwind weighted state;
- graph/PCA modes fitted fold-locally;
- propagation and front-arrival score.

## Forecast-anchor features

- official Tmax, Tmin, range;
- deviation from target rolling means/climatology;
- revision change;
- time since issue;
- source/product;
- text ontology;
- numeric-text contradiction;
- parser confidence;
- recent source bias at multiple half-lives;
- recent source absolute error;
- over/underforecast streak;
- source-season state.

All forecast versions must be exact-vintage and pre-cutoff.

## Interaction forms

- multiplicative interaction after centering;
- two-dimensional bins with minimum support;
- smooth tensor surface;
- regime-gated main effect;
- monotonic interaction;
- hierarchical state;
- contradiction score;
- residual correction only when both mechanism components are present.

Always compare against main effects.

## Analog state

- physically scaled distance;
- season restriction;
- source restriction;
- prior-only neighbor pool;
- k-nearest residual;
- kernel-weighted residual;
- nearest-distance and effective-neighbor uncertainty.

## Missingness and quality

- safe missing-at-cutoff indicator;
- quality flag;
- age/staleness;
- station count available;
- group coverage;
- parser null state.

Archive missingness not observable live is not a deployable feature.

## Time-of-year and trend

- harmonic sin/cos terms;
- multiple harmonics;
- causal trend;
- decayed day-of-year climatology;
- phase-specific terms;
- source-era indicator;
- MAM transition phase.

Calendar is safe; target-conditioned climatology must be causal.

## Online state

- exponentially weighted residual mean;
- robust residual median approximation;
- signed streak;
- recent absolute error;
- recent volatility;
- source/season/regime hierarchical state;
- prior-lift score;
- expert weight.

State updates occur after scoring the current target.

## Uncertainty and routing

- expert disagreement;
- station disagreement;
- transition score;
- source recent error;
- analog distance;
- forecast range;
- specialist activation confidence;
- predicted absolute error;
- probability official beats candidate.

These features may change correction shrinkage or expert weights.

## Required feature definition fields

Every feature must record:

- name;
- role;
- formula;
- input columns and sources;
- units;
- lag;
- window;
- time alignment;
- cutoff/availability rule;
- fit scope;
- missingness policy;
- station/group;
- expected mechanism;
- prior evidence;
- ablation group.
