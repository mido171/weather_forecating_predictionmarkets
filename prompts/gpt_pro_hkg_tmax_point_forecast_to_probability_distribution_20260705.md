# GPT-Pro Prompt: HKG Tmax Official Point Forecast to Probability Buckets

You are GPT-Pro. Take full lead as the senior probabilistic forecasting researcher, weather verification scientist, and probability-calibration architect for this HKG Tmax project.

We do not want you to be biased for or against EMOS just because it was mentioned in discussion. Treat EMOS as one candidate family among many. If EMOS is best, prove that with evidence and implementation detail. If it is not best, prove that with evidence and implementation detail. Do not privilege any method because we named it. Your job is to identify the best robust, leakage-safe, empirically defensible way to convert an official HKO point forecast for Hong Kong daily maximum temperature into a calibrated probability distribution over settlement buckets.

## Core Objective

We are now outgoing from the official forecast. In other words, the primary live forecast anchor is the latest same-source official HKO local forecast max temperature available before the decision cutoff. We are no longer asking whether a separate ML point model can materially beat the official forecast point estimate. Prior experiments showed only tiny point-MAE improvements after heavy feature engineering. The current problem is different:

**Given the latest official point forecast for a target date, what is the best way to convert that point forecast into a full calibrated probability distribution over the settlement buckets?**

We need you to design the full next implementation plan, with no missing scientific or engineering decisions, so Codex can implement and test it.

The desired end state is a repeatable pipeline that:

1. Selects the latest eligible official HKO local forecast before a specified cutoff.
2. Uses archived official forecast data and archived settled Tmax data to learn the historical forecast-error distribution.
3. Converts the live point forecast into a full probability distribution over integer settlement buckets.
4. Measures calibration, sharpness, reliability, and proper scoring-rule performance.
5. Provides hard evidence for choosing one probability method over alternatives.

## Current Scope: Probability System Only

This stage is **not** about Polymarket trading, order books, bid/ask spreads, expected value, Kelly sizing, execution, market-price blending, PnL, or trade recommendations. Those are later-stage concerns.

At this stage, the Polymarket event rules matter only because they define the target bucket labels and settlement mapping. The goal now is to build the absolute best methodology for turning a point forecast into a full set of calibrated probability percentages. We care about the quality of the weather probability distribution itself.

Do not optimize for market prices. Do not include a trading backtest as a required deliverable. Do not choose a method because it would have performed well against historical prices. Do not introduce market-implied probabilities into the weather probability model. The selected method should be judged by out-of-sample probabilistic forecast quality against settled weather outcomes.

You must take full lead and full responsibility for the probability methodology. Do not let our wording anchor you to EMOS, empirical residuals, normal residuals, quantile models, distributional ML, conformal methods, or any other named approach. Decide the best approach from evidence.

## Important Non-Bias Instruction

Do not infer that we prefer EMOS. Do not infer that we dislike EMOS. Do not infer that empirical residual histograms are sufficient. Do not infer that a parametric normal residual is sufficient. Do not infer that distributional ML is better just because it is more complex.

You must evaluate the full methodological landscape and choose the best path based on:

- point-in-time validity;
- out-of-sample proper scoring rules;
- bucket-level calibration;
- robustness by season, forecast level, issue hour, and regime;
- support/sample sufficiency;
- implementation risk;
- live deployability.

## Domain and Settlement Target

The settlement target is Hong Kong Observatory daily maximum temperature at the HKO Headquarters / Hong Kong Observatory station in Tsim Sha Tsui.

Known station metadata:

- Station: Hong Kong Observatory
- Location: HKO Headquarters, Tsim Sha Tsui
- Latitude: 22.301944
- Longitude: 114.174167
- Elevation: 32 m
- Settlement source: HKO Daily Extract
- Settlement variable: `Absolute Daily Max (deg. C)`
- Precision: one decimal deg C

## Specific Polymarket Event and Bucket Rules

For the immediate market we care about, the Polymarket event metadata was retrieved from the public Gamma API on 2026-07-05:

- Event id: `664852`
- Event slug: `highest-temperature-in-hong-kong-on-july-6-2026`
- Event title: `Highest temperature in Hong Kong on July 6?`
- Event date: `2026-07-06`
- Event end: `2026-07-06T12:00:00Z`
- Gamma event endpoint: `https://gamma-api.polymarket.com/events/slug/highest-temperature-in-hong-kong-on-july-6-2026`

The market resolution text says the market resolves to the temperature range that contains the highest temperature recorded by the Hong Kong Observatory in degrees Celsius on 6 Jul 2026. The resolution source is the Hong Kong Observatory Daily Extract, specifically `Absolute Daily Max (deg. C)`, available through the HKO climatological data page. The market cannot resolve until data for the date is published. The source measures temperatures in Celsius to one decimal place, and that one-decimal precision is the precision used for resolving the market. Revisions to temperatures after data is initially published for the market's timeframe are not considered.

This means the bucket assignment is **not nearest-integer rounding**. A settlement value of `31.9` belongs to the `31 deg C` bucket, not the `32 deg C` bucket. A settlement value of `32.0` belongs to the `32 deg C` bucket. For middle buckets, this is integer-degree floor/bin membership on a one-decimal settlement value.

Use decimal-safe arithmetic. Do not use binary floats for final bucket classification. Treat the HKO settlement as a one-decimal Decimal value, for example `Decimal("31.9")`.

The event bucket mapping is:

| Polymarket question | Market id | Bucket label | Settlement values resolving Yes |
|---|---:|---|---|
| Will the highest temperature in Hong Kong be 24 deg C or below on July 6? | 2791312 | `24 deg C or below` | `Tmax <= 24.9` |
| Will the highest temperature in Hong Kong be 25 deg C on July 6? | 2791313 | `25 deg C` | `25.0 <= Tmax <= 25.9` |
| Will the highest temperature in Hong Kong be 26 deg C on July 6? | 2791314 | `26 deg C` | `26.0 <= Tmax <= 26.9` |
| Will the highest temperature in Hong Kong be 27 deg C on July 6? | 2791315 | `27 deg C` | `27.0 <= Tmax <= 27.9` |
| Will the highest temperature in Hong Kong be 28 deg C on July 6? | 2791316 | `28 deg C` | `28.0 <= Tmax <= 28.9` |
| Will the highest temperature in Hong Kong be 29 deg C on July 6? | 2791317 | `29 deg C` | `29.0 <= Tmax <= 29.9` |
| Will the highest temperature in Hong Kong be 30 deg C on July 6? | 2791318 | `30 deg C` | `30.0 <= Tmax <= 30.9` |
| Will the highest temperature in Hong Kong be 31 deg C on July 6? | 2791319 | `31 deg C` | `31.0 <= Tmax <= 31.9` |
| Will the highest temperature in Hong Kong be 32 deg C on July 6? | 2791320 | `32 deg C` | `32.0 <= Tmax <= 32.9` |
| Will the highest temperature in Hong Kong be 33 deg C on July 6? | 2791321 | `33 deg C` | `33.0 <= Tmax <= 33.9` |
| Will the highest temperature in Hong Kong be 34 deg C or higher on July 6? | 2791322 | `34 deg C or higher` | `Tmax >= 34.0` |

The bucket classifier should be equivalent to:

```python
from decimal import Decimal

def hkg_high_tmax_bucket(tmax_1dp: Decimal) -> str:
    # tmax_1dp must be the HKO Daily Extract Absolute Daily Max value
    # represented at one decimal precision.
    if tmax_1dp <= Decimal("24.9"):
        return "24_or_below"
    if tmax_1dp >= Decimal("34.0"):
        return "34_or_higher"
    k = int(tmax_1dp)  # floor for positive Celsius values at one-decimal precision
    return f"{k}"
```

For continuous predictive distributions, integrate over these settlement intervals:

- `P(24_or_below) = P(Tmax <= 24.95)` if modeling latent continuous temperature that is rounded/reported to one decimal, or `P(Tmax_reported <= 24.9)` if modeling the reported one-decimal value directly.
- `P(k) = P(k.0 <= Tmax_reported <= k.9)` for `k = 25, ..., 33`.
- `P(34_or_higher) = P(Tmax_reported >= 34.0)`.

You must explicitly decide whether the probabilistic model is over the latent continuous physical maximum or over the reported one-decimal HKO Daily Extract value. If modeling latent continuous temperature, include a measurement/rounding layer before bucket integration. If modeling the reported value directly, use the one-decimal bucket intervals above. The simpler and likely safer initial approach is to model the reported one-decimal settlement value directly, but evaluate this rather than assume it.

Critical initial-publication rule: the market says revisions after data is initially published for the timeframe will not be considered. Our canonical target tables may contain finalized or later-reconciled labels. Your implementation plan must include a check for whether our archived settled Tmax labels represent the first-published Daily Extract value or a later revised value. If first-published and revised values differ historically, the model-selection/evaluation label must match the market's first-publication resolution rule, or the mismatch must be explicitly quantified.

## Data Assets We Have

We have archived forecast data and settled Tmax data in the project DB. The context documents are under:

`C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\hkg_tmax_elite_codex_bootstrap\hkg_tmax_elite_codex\documentation\strategy_implementation_documentation\context\live_trading`

You should assume Codex can read those files and query the DB. Your output should tell Codex exactly how to use these assets.

### Official Forecast Archive

Primary table:

`public.hko_historical_forecasts_2000_2026`

This table is the apples-to-apples official forecast archive matching the live source we should use for probability generation. It is based on Info.gov HKSAR Government weather press-release pages for `LOCAL WEATHER FORECAST`.

Known coverage and quality:

- Total rows: 324,179
- Strict usable local min/max anchors: 115,795 rows
- Strict usable local min/max target dates: 9,667 target dates
- Strict usable local min/max target date range: 2000-01-02 through 2026-06-21
- Historical local product has strict lead 0 and lead 1 usable local min/max rows
- Lead 1 rows: 88,504
- Lead 1 target dates: 9,665
- Lead 1 first target: 2000-01-02
- Lead 1 last target: 2026-06-21
- Forecast count per lead-1 target day varies; mode is about 10 forecasts per target date

Usable official forecast selector:

```sql
source = 'info_gov'
AND product_type = 'local'
AND row_quality_status = 'usable_local_minmax'
AND target_issue_lead_days = 1
AND forecast_max_c IS NOT NULL
AND issue_at_hkt IS NOT NULL
AND target_date IS NOT NULL
```

Important columns include:

- `source`
- `source_url`
- `product_type`
- `title`
- `index_date`
- `snapshot_at_hkt`
- `issue_at_hkt`
- `issue_at_utc`
- `target_date`
- `target_issue_lead_days`
- `forecast_min_c`
- `forecast_max_c`
- `row_quality_status`
- `full_text`
- `raw_sha256`
- `raw_path`

Live-source hierarchy:

1. Primary: Info.gov local forecast press-release pages.
2. Secondary confirmation only: HKO OpenData `flw` local forecast JSON.
3. Diagnostic only: HKO OpenData `fnd` 9-day forecast JSON.

Do not treat the HKO 9-day API as an apples-to-apples historical anchor unless you explicitly justify and segregate it. The historical archive and live point forecast should align to the same local forecast product wherever possible.

### Current Official-Point Baseline From Prior Work

Prior point-forecast experiment:

- Experiment: `0215_gpt_pro_point_forecast_strategy`
- Main script: `scripts/run_hkg_t24_0215_gpt_pro_point_forecast_strategy.py`
- Results folder: `experiments/0215_gpt_pro_point_forecast_strategy/results`
- Selected cutoff: `T-1 23:59 HKT`
- Equivalent Stockholm time: `17:59 CEST` during summer, `16:59 CET` during winter
- Selected model: `B3_grouped_residual_shrinkage`

Official-row-only scoring frame:

- Evaluation period: 2011-01-01 through 2023-12-31
- Rows: 4,747
- Selected residual-shrinkage model MAE: 0.9216066007226759 deg C
- Selected residual-shrinkage model RMSE: 1.1826760995241477 deg C
- Selected residual-shrinkage model bias: 0.012704769322469249 deg C
- Selected residual-shrinkage p90 absolute error: 1.9641225406986251 deg C
- Selected residual-shrinkage p95 absolute error: 2.3954636051070834 deg C

Raw official latest forecast baseline on identical rows:

- Rows: 4,747
- MAE: 0.9274910469770383 deg C
- RMSE: 1.1915190793489194 deg C
- Bias: -0.12285654097324625 deg C

The point-MAE improvement was tiny:

- MAE improvement: about 0.005884446254362463 deg C
- RMSE improvement: about 0.00884297982477178 deg C

The prior point-forecast model failed hard promotion gates. This does not mean probabilistic calibration is impossible. It means the point forecast is already very strong, and the next useful edge is likely in uncertainty, calibration, bucket pricing, market mispricing, and regime-specific forecast-error structure.

Treat the official point forecast as the central forecast anchor unless your evidence proves a better point anchor is needed.

### Hourly Readings Archive

Secondary feature/context table:

`public.hko_info_gov_hourly_readings_1998_2026`

Known coverage:

- Rows: 268,894
- Date range/index HKT: 1998-05-04 through 2026-07-04
- Parsed rows: 268,856
- Partial rows: 38
- Failed rows: 0
- Target-station present rows: 268,861
- Null HKO temp/RH rows: 33
- Discovered URLs failed fetch: 43
- Unique stations: 27

This table contains Info.gov `PRESS WEATHER NO. ### - HOURLY READINGS` dispatches, not forecasts.

Important columns include:

- `dispatch_at_utc`
- `observation_at_utc`
- `hko_air_temp_c`
- `hko_relative_humidity_pct`
- `station_readings_jsonb`
- `station_count`
- `station_missing_count`
- `station_temp_min_c`
- `station_temp_max_c`
- `station_temp_mean_c`
- `station_temp_spread_c`
- warning/rain/lightning/tropical cyclone text fields
- `raw_sha256`
- `parse_status`

Feature-safety requirements:

- Only use hourly rows where `dispatch_at_utc <= decision_cutoff_utc`.
- Observation timestamp is not sufficient by itself; public dispatch availability matters.
- Never use target-day future readings for a pre-target-day decision.
- HKO hourly values are integer deg C and useful as context features, but they are not one-decimal settlement labels.
- This archive currently ends at 2026-07-04 and needs incremental backfill/live ingestion for future dates.

Hourly readings may be useful for uncertainty regimes, residual conditioning, heat carryover, humidity, station spread, warnings, rain state, and storm regimes. They must not introduce post-cutoff leakage.

### Settled Tmax Archive

Canonical target label source:

`data/datasets/01_hko_daily_tmax_target/hko_daily_tmax_target_labels.parquet`

Canonical DB tables:

`label_core.hko_daily_tmax`

- Pre-2024 labels
- Rows: 48,577
- Date range: 1884-01-01 through 2023-12-31
- Null target values: 0
- Min: 3.20 deg C
- Max: 36.60 deg C

`sealed_confirmation.hko_daily_tmax`

- 2024+ confirmation labels
- Rows: 882
- Date range: 2024-01-01 through 2026-05-31
- Null target values: 0
- Min: 10.40 deg C
- Max: 35.70 deg C

Combined canonical target:

- Rows: 49,459
- Date range: 1884-01-01 through 2026-05-31
- Null `target_tmax_c`: 0
- Coverage: 95.084205 percent
- Only missing block: 1940-01-01 through 1946-12-31

Feature-safe target-history view:

`feature_safe.hko_target_history_pre2024`

Raw audit table:

`raw_audit.codex_audit_ds_01_hko_daily_tmax_target_hko_daily_extr_23cb54da`

- Rows: 49,628
- Date range: 1884-01-01 through 2026-06-17
- Includes raw monthly/yearly overlap and audit payload rows
- Not the canonical modeling label table unless explicitly reconciled

Target-label rules:

- Use settled Tmax as supervised outcome and evaluation truth.
- Never use `target_tmax_c[T]` as a predictor for target date T.
- Keep `sealed_confirmation.hko_daily_tmax` out of training/model selection unless you explicitly define a one-time holdout/confirmation protocol.
- Lagged target-history features must respect real publication availability. Lag1 may be unsafe at a T-1 afternoon/evening cutoff unless publication timing proves it was known.
- Use temporal splits and fold-local preprocessing. Do not fit climatology, residual distributions, scalers, calibration maps, or feature encoders on future rows.

July historical canonical distribution, useful only as background:

- Rows: 4,185
- Average Tmax: 31.129 deg C
- 5th percentile: 28.0 deg C
- Median: 31.3 deg C
- 95th percentile: 33.6 deg C
- Minimum: 24.90 deg C
- Maximum: 36.10 deg C

## Decision Setup

For a target date `T`, we need a live probability distribution at a decision cutoff, for example:

- Cutoff: T-1 23:59 HKT
- In Stockholm: 17:59 CEST during summer, 16:59 CET during winter

The training row construction should:

1. Gather all official local forecast rows for target date `T` with `issue_at_utc <= decision_cutoff_utc`.
2. Select the latest eligible official local forecast row.
3. Join to the settled target Tmax for `T`.
4. Compute residuals such as:
   - `residual = settled_tmax_c - official_forecast_max_c`
   - `abs_error = abs(residual)`
   - `bucket_label = market_bucket(settled_tmax_c)`
5. Add only features that would have been known by the cutoff.

The final live engine should output, for the target date and official forecast:

- official forecast source row and source URL;
- selected issue time HKT and UTC;
- point forecast max;
- target date;
- predictive mean of final distribution if applicable;
- full CDF or PMF;
- market bucket probabilities;
- uncertainty diagnostics;
- calibration metadata;
- full bucket probability percentages and diagnostics only, with no trading recommendation.

## Candidate Method Families To Evaluate

This list is intentionally non-exhaustive. You may add, remove, or merge families. You must not simply pick from this list without evidence.

### 1. Empirical Residual Distribution

Learn `settled_tmax_c - official_forecast_max_c` from historical analogs.

Possible conditioning dimensions:

- month or day-of-year seasonality;
- official forecast max level, such as 29, 30, 31, 32, 33 deg C;
- official forecast range width: `forecast_max_c - forecast_min_c`;
- issue hour or forecast cycle;
- target lead/cutoff;
- recent forecast revision direction;
- forecast text signals;
- rainfall/warning/storm regime;
- prior-day and pre-cutoff hourly context;
- station spread and humidity context from hourly readings;
- hot-season vs cool-season.

Required design questions:

- How should sparse cells be shrunk toward broader parent groups?
- Should residuals be modeled as discrete one-decimal empirical values or smoothed continuous densities?
- Should analog weights decay by day-of-year distance, forecast-level distance, issue-hour distance, or recency?
- How do we prevent overfitting sparse buckets such as 33+ deg C?

### 2. Parametric Residual Distribution

Fit a conditional residual distribution around the official point forecast.

Possible distributions:

- Normal;
- Student-t;
- skew-normal;
- asymmetric Laplace;
- mixture of normals;
- zero-inflated or discretized distributions if rounding artifacts matter.

Required design questions:

- Is the residual distribution biased, skewed, heavy-tailed, or seasonally heteroskedastic?
- Does variance depend on official forecast level, month, forecast range width, text regime, or hourly preconditions?
- Should residual mean and scale be modeled separately?
- Should we produce a continuous CDF and integrate bucket ranges, or directly estimate bucket probabilities?

### 3. MOS / EMOS-Style Calibration

Evaluate Model Output Statistics / Ensemble Model Output Statistics ideas neutrally.

Important: we may not have a true ensemble forecast archive in the official local forecast table. If no real ensemble mean/spread/members are available, EMOS may need to be adapted into a MOS-style parametric calibration around an official deterministic point forecast, or use proxy spread features such as:

- forecast min/max range;
- forecast revision volatility;
- HKO forecast text uncertainty;
- hourly station spread;
- recent residual climatology;
- external diagnostic forecast fields only if point-in-time and historically available.

Required design questions:

- Is there enough ensemble-like information to justify EMOS rather than a simpler MOS or conditional residual model?
- What is the exact predictive distribution family?
- What parameters are learned, and with what features?
- What proper scoring rule is optimized?
- Does EMOS improve out-of-sample NLL, CRPS, RPS, and bucket calibration versus simpler baselines?

Do not recommend EMOS because the acronym was mentioned. Recommend it only if it wins on hard evidence and is implementable with our archived inputs.

### 4. Distributional ML

Consider direct probabilistic ML approaches, but only if they can be validated without leakage.

Possible methods:

- quantile regression;
- quantile gradient boosting;
- NGBoost or similar distributional boosting;
- gradient-boosted multiclass bucket classification;
- ordinal regression;
- random forest quantile regression;
- generalized additive models for location/scale/shape;
- neural approaches only if justified by sample size, stability, and validation.

Required design questions:

- Does the method materially improve proper scoring rules or calibration after temporal validation?
- Does it overfit rare buckets?
- Are probabilities smooth and stable around adjacent buckets?
- Is it robust when the official forecast point is already highly informative?

### 5. Conformal and Calibrated Quantile Methods

Consider split conformal, conformalized quantile regression, and calibration layers.

Required design questions:

- Does conformalization improve coverage but harm sharpness?
- Can conformal intervals be converted into bucket probabilities in a principled way?
- Should conformal methods be used as a diagnostic/coverage layer rather than the primary PMF?
- How should temporal dependence be handled?

### 6. Direct Bucket Calibration

Estimate calibrated probabilities for each market bucket directly.

Possible methods:

- multiclass logistic regression;
- ordinal logistic/probit models;
- gradient boosting classification;
- isotonic calibration;
- beta calibration;
- Platt-style calibration;
- Dirichlet calibration;
- one-vs-rest bucket calibration layered on a base distribution.

Required design questions:

- Does direct bucket classification lose useful continuous structure?
- How do we enforce probability mass consistency and smoothness across adjacent buckets?
- How do we calibrate rare buckets without unstable overconfidence?
- Should direct calibration be layered on top of a continuous residual distribution?

### 7. Bayesian / Hierarchical Residual Models

Consider hierarchical residual models where sparse conditions borrow strength from broader groups.

Potential hierarchy:

- global residual distribution;
- hot-season residual distribution;
- month residual distribution;
- forecast-max-level residual distribution;
- issue-hour/cutoff residual distribution;
- regime-conditioned residual distribution.

Required design questions:

- Can hierarchical shrinkage improve rare bucket estimates while preserving local structure?
- Is the implementation practical for live daily use?
- Does it beat simpler shrinkage empirically?

### 8. Ensemble / Stacking of Probability Models

Consider a final stacked probability distribution if multiple families are complementary.

Required design questions:

- How are weights learned?
- Are weights static, season-specific, forecast-level-specific, or time-varying?
- Is stacking tuned inside a nested temporal validation loop?
- Does stacking improve proper scoring rules without degrading calibration?
- Does the stack collapse to a simple method when evidence supports simplicity?

### 9. Explicitly Out of Scope for This Stage: Market-Implied or Trading Layers

Do not include a market-implied probability layer in the current implementation plan.

This stage is only about the weather probability distribution conditional on official forecast information and leakage-safe weather/context features. Polymarket metadata is used only to define bucket labels and settlement intervals.

Out of scope for this prompt:

- market prices;
- order books;
- bid/ask spread;
- fees;
- slippage;
- liquidity;
- fill probability;
- expected value;
- Kelly sizing;
- PnL;
- long-Yes / long-No recommendations;
- blending weather probabilities with market-implied probabilities.

You may include a short note on what outputs should be preserved for a future trading layer, but do not make trading evaluation part of the method-selection objective.

## Required Evaluation Framework

Point MAE and RMSE are secondary diagnostics for this phase. The main task is probabilistic forecasting and bucket calibration. You must specify the full evaluation framework and say exactly which metric decides the winning method.

Use proper scoring rules and calibration diagnostics, including:

1. Negative log likelihood / log loss over mutually exclusive market buckets.
2. Ranked Probability Score (RPS) for ordered temperature buckets.
3. CRPS for continuous predictive distributions, if the model outputs a continuous distribution.
4. Multiclass Brier score.
5. One-vs-rest Brier score per settlement bucket.
6. Reliability/calibration curves per bucket.
7. PIT histogram for continuous distributions.
8. Probability integral transform or randomized PIT for discrete/rounded outcomes.
9. Expected calibration error (ECE) and maximum calibration error (MCE), with binning rules defined before scoring.
10. Coverage of central intervals such as 50, 80, 90, and 95 percent intervals.
11. Sharpness / entropy, reported alongside calibration.
12. Tail calibration for low and high buckets.
13. Conditional calibration by:
    - month;
    - hot season;
    - July specifically;
    - official forecast max level;
    - issue hour/cutoff;
    - rainfall/storm/warning regime;
    - high-humidity or high-station-spread regime;
    - recent forecast revision regime.

Do not add an economic backtest in this stage. Do not use market prices in model comparison. The winning method should be selected from weather-outcome probability quality only.

## Validation Requirements

Use leakage-safe temporal validation. You must define the exact split plan.

Recommended starting structure, unless you propose a better one:

- Use only rows where official forecast, optional features, and target label are available point-in-time.
- Build a development period from the forecast archive, for example 2000-2023 or 2011-2023 depending on row quality and feature availability.
- Keep 2024+ sealed confirmation data out of model selection unless explicitly used as a one-time final confirmation.
- Use walk-forward validation with expanding or rolling windows.
- Tune hyperparameters and calibration choices only inside training/validation folds.
- Report final performance on outer folds only.
- Consider a final untouched confirmation on 2024+ if and only if governance allows it.

You must specify:

- exact training/validation/test split dates;
- whether folds are yearly, multi-year, rolling, or expanding;
- how recency weighting is handled;
- how calibration maps are trained fold-locally;
- how sparse buckets are handled;
- what minimum sample sizes are required for subgroup claims;
- how many methods can be compared without turning the holdout into a tuning set;
- what is the no-harm baseline.

## Baselines That Must Be Included

At minimum, compare against these baselines:

1. Raw climatology by month/day-of-year, without official forecast.
2. Raw official forecast plus unconditional historical residual distribution.
3. Raw official forecast plus month-conditioned residual distribution.
4. Raw official forecast plus forecast-level-conditioned residual distribution.
5. Raw official forecast plus month and forecast-level residual distribution with shrinkage.
6. Simple parametric normal residual model with conditional mean and variance if justified.
7. Best candidate complex model selected by your plan.

You may add more baselines. You may reject a baseline if it is invalid, but explain why.

The winning method must beat simple baselines under proper scoring rules and calibration diagnostics, not just look more sophisticated.

## Row Construction and Leakage Audit Requirements

You must tell Codex exactly how to construct the modeling table.

Include SQL or pseudo-SQL for:

1. Selecting latest eligible official forecast per target date and cutoff.
2. Joining settled Tmax labels.
3. Computing residual and bucket label.
4. Joining optional hourly-reading features with strict `dispatch_at_utc <= cutoff`.
5. Excluding rows with missing target, missing forecast max, invalid target mapping, or post-cutoff data.
6. Producing row-count audits by year, month, official max level, and issue hour.

The implementation plan must include hard failure conditions:

- no eligible official forecast before cutoff;
- target label unavailable;
- post-cutoff forecast selected;
- mismatch between target date and issue lead;
- duplicate latest forecast tie not resolved deterministically;
- market bucket boundaries not verified;
- features using future observations or target-day future readings;
- any calibration step fit using future rows.

## Bucket Probability Construction

You must specify the exact transformation from predictive distribution to bucket probabilities.

For continuous predictive distributions:

`P(bucket k) = F(upper_k) - F(lower_k_minus_epsilon_or_open_boundary)`

But define the exact interval convention from market rules and one-decimal settlement precision.

For discrete empirical distributions:

`P(bucket k) = sum_i weight_i * I(settled_bucket_i = k)`

with defined smoothing and shrinkage rules.

For direct classifiers:

Ensure:

- probabilities sum to 1 over all possible buckets;
- out-of-range mass is handled explicitly;
- buckets outside the listed market set are not silently dropped;
- probabilities for listed buckets are not renormalized incorrectly if an event exposes only a subset of all possible outcomes;
- the final output reports bucket probabilities as weather probabilities only, not trade prices or recommendations.

## Deliverables You Must Specify

Give Codex a complete implementation plan with concrete artifacts. Include recommended file and folder layout.

The implementation should produce:

1. A reproducible modeling table, preferably Parquet.
2. A row-count and leakage-audit report.
3. Per-fold predictions with full probability vectors.
4. Per-fold continuous CDF parameters where applicable.
5. Bucket probability CSV/Parquet outputs.
6. Proper scoring-rule metrics JSON/CSV.
7. Calibration diagnostics tables.
8. Reliability plot data and, if useful, rendered plots.
9. PIT histogram data and plots for continuous methods.
10. Subgroup metrics by month, July, hot season, official forecast level, and issue hour.
11. A model leaderboard with statistical uncertainty.
12. A final model-card style recommendation explaining the chosen probability method.
13. A live inference function that accepts target date, cutoff, selected official forecast row, and optional features, and returns bucket probabilities.
14. A reproducibility manifest with:
    - git commit if available;
    - DB query timestamps;
    - source table row counts;
    - input hashes where available;
    - code version;
    - model parameters;
    - random seeds;
    - exact command lines.

## Acceptance Gates

Define hard gates before implementation. At minimum include:

1. Leakage gate: zero known post-cutoff features or target leakage.
2. Row identity gate: baseline and candidate methods scored on identical rows for each comparison.
3. Proper scoring gate: candidate must improve NLL/RPS/CRPS/Brier versus required baselines on outer folds.
4. Calibration gate: candidate must not create severe miscalibration in key settlement buckets.
5. Robustness gate: candidate must not win only in one tiny subgroup while failing broad hot-season or July checks.
6. Support gate: subgroup claims require minimum sample counts.
7. Simplicity gate: if a complex method only ties a simple method, choose the simpler and more stable method.
8. Live-deployability gate: all required features must be available before the live cutoff.
9. Scope gate: no market prices, trading backtests, EV calculations, order-book assumptions, or market-implied probabilities are part of the current method-selection process.

Define quantitative thresholds where possible. If exact thresholds require exploratory diagnostics, propose a two-stage process:

- Stage 1: pre-registered exploratory comparison with broad metrics.
- Stage 2: locked confirmation with fixed method and thresholds.

## Specific Questions You Must Answer

Answer these directly and in detail:

1. What is the best primary modeling target: residual CDF, bucket PMF, quantiles, distribution parameters, or a stacked hybrid?
2. Is EMOS appropriate here given the archived inputs, or is MOS/conditional residual calibration more appropriate?
3. What minimum set of features should be used in v1 to avoid overfitting?
4. What expanded feature set should be tested in v2?
5. How should we handle sparse high-temperature buckets such as 33+ deg C?
6. How should we convert one-decimal settled Tmax into integer market buckets?
7. How should we measure calibration in a way that is robust and hard to game?
8. Which proper scoring rule should select the model, and why?
9. How should we split train/validation/test periods?
10. Should 2024+ settled confirmation data be used now, kept sealed, or used only for final confirmation?
11. How should we account for official forecast issue time and multiple forecasts per target date?
12. How should prior-day or pre-cutoff hourly readings be used, if at all?
13. What output schema should represent the full probability distribution cleanly for future downstream use?
14. What exact artifacts should Codex produce so the result is auditable and reusable?
15. What would convince you that the selected method is genuinely better than a simple residual baseline?

## Output Format Requested From GPT-Pro

Please return a full research and implementation specification, not a generic discussion.

Use this structure:

1. Executive recommendation.
2. Methodological landscape and neutral comparison, including EMOS but not privileging it.
3. Exact data construction plan.
4. Exact candidate model set.
5. Exact validation and leakage protocol.
6. Exact scoring and calibration protocol.
7. Exact probability-output and live-inference protocol.
8. Exact implementation file/folder plan.
9. Exact commands or pseudo-commands for Codex to implement.
10. Exact acceptance gates.
11. Risks, blockers, and required clarifications.
12. Final recommended first experiment specification.
13. Reserve queue for second and third experiments if the first fails.

Be precise enough that Codex can implement the full plan without needing to infer scientific choices. Where there are multiple valid paths, choose one and explain why. If you need to leave an option open, specify the exact evidence that will decide it.

## Tone and Decision Standard

Be rigorous and competitive. We are trying to produce the best possible calibrated probability distribution from an official point forecast, with bucket rules defined by the market contract but no trading optimization in this stage. Prefer robust, auditable, leakage-safe evidence over cleverness.

The correct answer may be simple. The correct answer may be complex. Your task is to determine which, using hard out-of-sample probabilistic evidence.

Do not optimize for sounding impressive. Optimize for a plan that will survive implementation, validation, calibration review, and future downstream use.
