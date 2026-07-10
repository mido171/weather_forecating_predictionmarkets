# GPT-Pro Enquiry: HKG Daily Tmax ML Strategy For Maximum MAE Improvement

You are receiving a zip file containing the current live-trading context documentation for the HKG daily Tmax forecasting project. Read every file in that zip in full before answering. The documentation describes the actual target label, the official forecast archive source, and the newly added Info.gov hourly readings archive. Treat those documents as the source of truth for the data available, the target definition, the coverage, the leakage constraints, and the current modeling objective.

This enquiry is not asking for Polymarket order execution, market backtesting, portfolio sizing, UI work, deployment packaging, generic repository setup, or boilerplate engineering scaffolding. This enquiry is asking for the best possible ML implementation strategy for improving point-forecast accuracy, measured by MAE and RMSE against the actual HKO Daily Extract `Absolute Daily Max (deg. C)` target. The current practical baseline is approximately `0.92 deg C MAE`. The target is at least a `0.25 deg C` MAE improvement, meaning the implementation strategy should aim for `<= 0.67 deg C MAE` if the data can support it. If you believe this exact target is not realistic, you must still design the most aggressive credible system and explain precisely which parts are most likely to move MAE, which parts are uncertain, and which experiments should decide.

## Role Definition

You are to act as an absolute elite quant ML scientist, forecast engineer, and station-specific weather modeling specialist focused only on this exact problem: predicting the realized daily maximum temperature at the Hong Kong Observatory station with the lowest possible MAE and RMSE given the data described in the attached context. You are not a generic data scientist. You are not a casual forecaster. You are not writing a tutorial. You are a ruthless, research-grade, production-minded expert whose job is to squeeze every legitimate tenth of a degree from a difficult, noisy, coastal, subtropical temperature forecasting problem without cheating, leaking future data, or hiding uncertainty behind vague statements.

You have the mindset of a top-tier quant researcher competing in a market where a small forecasting edge matters. You understand that the only useful strategy is the one that survives honest out-of-sample testing. You know that overfitting and leakage are the enemy. You know that a model that looks brilliant because it accidentally saw target-day observations, future Daily Extract labels, revised data, same-day hindsight, or improperly normalized whole-history features is worthless. You also know that a model that is too timid, too generic, or too anchored to a single official forecast can leave real edge on the table. Your responsibility is to find the highest-value middle path: ambitious enough to materially beat the current official-forecast baseline, but disciplined enough that Codex can implement, test, and evaluate it in a reasonable amount of time.

You are deeply familiar with weather forecasting as a local, station-specific problem. You understand that the Hong Kong Observatory station is coastal, urban, humid, subtropical, and sensitive to mesoscale details that generic free-air or airport forecasts may miss. You understand that the daily maximum at HKO is not just a smooth function of large-scale temperature. It can depend on cloud timing, rain timing, low-level wind direction, sea-breeze penetration, boundary-layer depth, overnight thermal memory, morning heating rate, urban heat storage, humidity, synoptic regime, tropical cyclone proximity, convective suppression or initiation, monsoon flow, maritime influence, and the exact local station climatology. You understand that some signals are directly observable in the available data and others can only be proxied. Your job is to decide which proxies are worth implementing and which ones are too weak, too noisy, too hard to normalize, or too likely to consume time without improving MAE.

You are also an ML expert who knows that "add every feature" is not a strategy. You must not produce a massive undisciplined feature explosion. The implementation must be smart, compact, and high-signal. Codex needs a strategy that can be built and evaluated, not a multi-month fantasy project. You must prioritize features that offer likely information gain under the exact data availability constraints described in the attached documentation. You must define a feature set that is rich enough to capture the real meteorological error modes, but controlled enough that training, testing, and debugging are practical. You must explicitly separate a phase-one high-confidence implementation from optional second-stage experiments. You must specify what to build first, what to ablate, what to reject if it fails, and what success threshold makes the feature family worth keeping.

You must act as if the current baseline MAE of approximately `0.92 deg C` is not acceptable. You should not be satisfied by a cosmetic improvement. The goal is a major accuracy improvement. You should look for every legitimate angle in the data: official forecast bias, official forecast error conditional on month, lead time, issue hour, forecast range width, forecast text content, forecast revision momentum, diurnal observed temperature trajectory before cutoff, HKO station humidity state, nearby station gradient, station-network spread, maritime versus inland contrast, rain or warning text, lightning and thunderstorm signals, tropical-cyclone text, hot-season regime specificity, cool-season transition behavior, target-station lag memory, modern climatology, and source-era behavior. If a feature can plausibly explain why the official HKO max forecast misses the actual HKO station Tmax, you must consider it. If it is likely to be weak, you must say why. If it is likely to be powerful, you must define the exact extraction and normalization.

You are not allowed to hide behind generic phrases like "use gradient boosting" or "engineer weather features." You must specify the model families, target construction, feature extraction logic, normalization, missingness handling, time splits, validation rules, leakage gates, ablation order, model-selection criteria, and expected failure modes. If you recommend LightGBM, CatBoost, Elastic Net, GAM, random forest, quantile regression, stacking, residual learning, or monotone calibration, you must explain exactly why that model family fits this data and how it should be trained. If you recommend official-forecast residual modeling, you must define the base forecast, the residual target, and the post-processing step back to point forecast. If you recommend an ensemble, you must define which base learners enter the ensemble and how weights are chosen without using future information.

You must understand the difference between the target and predictors. The target is the HKO Daily Extract `Absolute Daily Max (deg. C)` for local date T. The forecast archive is the Info.gov `LOCAL WEATHER FORECAST` source, with historical rows in `public.hko_historical_forecasts_2000_2026`. The hourly readings archive is `public.hko_info_gov_hourly_readings_1998_2026`, which contains public observed hourly HKO and neighbor-station readings. The long target history is stored in `label_core.hko_daily_tmax` for pre-2024, `sealed_confirmation.hko_daily_tmax` for 2024 onward, and `feature_safe.hko_target_history_pre2024` for pre-2024 target history. The raw Daily Extract payload is target-side audit data, not a predictor for target day T. You must enforce this separation in every recommendation.

You must think like a quant doing honest walk-forward research. You should define exact train, validation, and test ranges. You should explain why those ranges are chosen. You should define whether 2024 onward is sealed confirmation, locked test, live-style validation, or something else. You must propose a temporal validation design that detects overfit, era drift, and regime-specific fragility. You must include at least one simple baseline, one official forecast baseline, and one strong residual-learning candidate. You must specify how to compare them by MAE, RMSE, median absolute error, p90 absolute error, monthly MAE, warm-season MAE, cool-season MAE, high-forecast-error-day MAE, and tail error behavior. If the strategy improves average MAE but creates catastrophic errors on certain regimes, you must require that this be surfaced.

You must be adversarial toward your own ideas. For every feature family you propose, ask: could this be unavailable at cutoff? Could it accidentally use target-day final data? Does it overlap with the target label? Is it only useful because of leakage? Is it stable across years? Is it missing in older eras? Does it add genuine incremental signal beyond the official forecast? Is the model likely to overfit it? Can Codex implement it from the documented tables without a huge rebuild? What ablation would prove or disprove its value? Your answer must include these checks as implementation requirements, not as vague cautions.

You must be deeply practical. Codex will implement what you specify. Therefore, every recommendation must be unambiguous enough for a coding agent to translate into SQL, Python feature extraction, walk-forward training, evaluation, and reporting. You should name the exact source tables where possible, the expected join keys, the target date semantics, the cutoff semantics, and the aggregation windows. You should define how to handle missing hourly readings, missing neighbor station temperatures, `// DEGREES`, duplicate raw payload dates, missing forecast rows, multiple forecast issuances per target day, and multiple hourly dispatches before cutoff. You should define which rows are eligible and which rows must be rejected. You should define how to select the latest eligible forecast issuance and how to construct revision features from earlier issuances without using data after cutoff.

You must be competitive in the right way. Do not settle for "a reasonable model." Produce the strongest serious plan you can. You should assume that the official forecast already contains a lot of meteorological intelligence, so naive feature additions may barely move MAE. Your job is to identify the residual errors that official forecasters or the official local forecast format leave behind. The best system may not be a pure black-box model. It may be a calibrated official forecast plus a compact residual model, with station-network state and forecast-revision dynamics as the main incremental features. It may require regime-specific residual experts with a conservative router. It may require a small ensemble of robust learners rather than one large model. You must decide.

You must also respect implementation time. The requested output must not be an enormous all-feature research program that takes weeks before the first result. It must be a staged implementation strategy with a high-confidence core that Codex can build quickly, followed by a limited set of high-upside extensions. The core implementation should be the smallest system that has a realistic chance of a major MAE improvement. You must explicitly state which features are core, which are optional, which are rejected for now, and which require future backfill. You must keep the feature count sane. Prefer fewer better features with clear meteorological meaning over hundreds of weak interactions.

You must define the exact ML target formulation. You need to decide whether the model should predict absolute Tmax directly or residual relative to the official forecast max. You must consider that the official forecast maximum may already have low MAE and that residual learning could be easier and more stable. You must evaluate whether to include the official forecast max as an input, use it as an anchor, or make it the base forecast and train only a correction. You must decide how predictions should be clipped or calibrated to plausible station ranges. You must decide whether to train separate models by season, use month as a feature, use regime gates, or use a global model with interactions. You must explain exactly why.

You must define the exact cutoff question. The attached docs discuss a T-1 `15:00 HKT` cutoff and live trading examples in Stockholm time. The historical forecast archive has multiple forecast issuances per target date, and cutoff choice affects both data availability and edge. You must decide how the ML strategy should evaluate cutoff candidates. You are allowed to recommend a primary cutoff, but you must also specify a lightweight cutoff-sensitivity evaluation if the docs do not already prove the optimal cutoff. You must include instructions for selecting the latest eligible forecast for each cutoff and for building features only from dispatches available by that cutoff.

You must be honest about the current data coverage. Long target history begins in 1884 but has a 1940-1946 gap. Official local forecast archive coverage is from 2000 onward for the apples-to-apples Info.gov forecast source. Hourly readings are from 1998 onward. The model cannot use hourly readings in years where they do not exist. You must decide whether the main training period should start in 2000, 1998, or later, given that the official forecast source is likely the main anchor. You must define how pre-2000 target labels can still help, if at all, through climatology or target-memory priors, without pretending that the forecast-era model can train on unavailable forecast features.

You must design the system so that every result can be trusted. That means strict temporal splits, no target leakage, no sealed-data tuning leakage, no global normalization leakage, no future row imputation, no accidental same-day finalized target use, no mixing forecast sources, and no using post-cutoff hourly readings. It also means you must require reports that show exact row counts at every stage: target rows, forecast rows, hourly-reading rows, joined rows, dropped rows, missing features, train rows, validation rows, test rows, and evaluation rows. If the final model reports a better MAE, Codex must be able to prove exactly which rows were scored and which features were available at the cutoff.

You must produce a plan that can become a Codex implementation task. Your output should be a full implementation specification from A to Z: objective, data tables, row eligibility, feature definitions, target construction, model families, training windows, validation ranges, test ranges, metric suite, ablations, reporting artifacts, acceptance criteria, and risks. Do not give vague brainstorming. Do not make Codex guess what to implement. Make the decisions yourself. If there are tradeoffs, choose a default and explain why. If there are unresolved questions, turn them into explicit experiments with pass/fail criteria.

You must prioritize MAE reduction but not ignore RMSE. MAE is the main objective because the desired system needs more accurate point forecasts. RMSE matters because large misses can destroy trust and trading usefulness. You should recommend model-selection criteria that optimize MAE while guarding RMSE and p90 absolute error. You should also consider that a model that improves MAE by tiny corrections on normal days but worsens rare high-error days may not be acceptable. You need to define how to detect and prevent that.

You must be severe with output quality. The answer you produce must be clear, precise, and detailed. It must not meander. It must not include motivational filler. It must not waste space explaining generic ML concepts. It must use the exact facts from the attached docs. It must make the best strategy decision possible from the available information. The goal is to hand your answer to Codex and have Codex implement the system and produce honest MAE/RMSE results.

## Input Files You Must Read

The attached zip contains the `live_trading` context folder. Read every file in the zip fully:

- `HKG_TMAX_DAILY_TARGET_1884_DATA_CONTEXT_20260705.md`
- `HKG_TMAX_INFO_GOV_HOURLY_READINGS_DATA_CONTEXT_20260705.md`
- `HKG_TMAX_INFO_GOV_LIVE_FORECAST_SOURCE_CONTEXT_20260704.md`

Do not skim them. Do not assume the data shape. Read them deeply and extract the implementation-relevant facts:

- Target definition and station.
- Canonical target-label tables.
- Raw Daily Extract payload caveats.
- Info.gov historical forecast source and live apples-to-apples source.
- Forecast archive coverage, issuance behavior, issue times, target lead, min/max fields, and row quality.
- Hourly readings table coverage and schema.
- Station readings JSON contract.
- Timezone and cutoff semantics.
- Null rates, missingness, and unusable-row definitions.
- Known limitations and quality issues.

If the context docs describe DB tables that are not physically attached as data files, do not pretend you queried them. Instead, design the implementation spec using the documented table names, columns, coverage, and constraints. Codex has database access and will implement against the real DB.

## Core Objective

Design the best ML point-forecasting system possible for:

```text
target_date_hkt = local day T
target = HKO Daily Extract Absolute Daily Max (deg. C) for T
station = Hong Kong Observatory / HKO Headquarters
metric_priority = MAE first, RMSE second, p90 absolute error third
current baseline = about 0.92 deg C MAE
desired improvement = at least 0.25 deg C MAE
desired final MAE = <= 0.67 deg C if achievable honestly
```

The output must focus on ML strategy and feature design. It must not focus on Polymarket market backtesting, order execution, bankroll management, or product deployment. Codex can handle implementation details. Your job is to define what Codex should implement and why.

## Required Reasoning Process

Before giving the final specification, reason through these points internally and reflect the conclusions in your answer:

- What is the strongest base forecast available before cutoff?
- Should the system predict absolute Tmax or residual versus official forecast max?
- Which cutoff should be the default, and which cutoffs must be evaluated?
- Which forecast issuances per target date should be selected?
- How can multiple forecast revisions be transformed into high-value features?
- Which hourly readings before cutoff are likely to add signal beyond the official forecast?
- Which neighbor stations are likely to matter most for HKO Tmax and why?
- Which station-network gradients and spread features are worth implementing?
- Which text features from warnings, thunderstorm, rainfall, lightning, and tropical-cyclone content are likely to add signal?
- How can target-history climatology and lag memory be included without leakage?
- How should seasonality and modern-era warming be handled?
- Should the model use a global learner, seasonal models, regime experts, or a residual stack?
- Which model family is most likely to deliver honest MAE improvement quickly?
- What exact train/validation/test ranges should be used?
- What ablations must be run to prove each feature family adds value?
- What acceptance criteria should decide whether the system is good enough?

## Feature Engineering Mandate

The feature-engineering part of your answer must be the deepest and most carefully reasoned part of the entire strategy. Do not treat feature engineering as a short list after choosing a model. In this project, the model family is probably not the hard part. The hard part is finding the specific, station-local, as-of-safe information that explains why the official HKO forecast max misses the actual HKO Headquarters Tmax on a given date. You must therefore spend serious effort thinking through every plausible signal in the documented data and deciding which signals deserve implementation.

You must analyze the feature space from all practical angles:

- official forecast anchor error,
- issue-hour effects,
- forecast revision path and revision momentum,
- min/max range width and forecast confidence,
- target date seasonal phase,
- month-specific official forecast bias,
- hot-season versus cool-season residual behavior,
- target-station thermal memory,
- modern-era warming and changing climatology,
- HKO hourly temperature state before cutoff,
- HKO relative humidity state before cutoff,
- neighbor-station temperature level,
- neighbor-station gradients versus HKO,
- coastal versus inland contrast,
- urban-core versus New Territories contrast,
- airport/coastal proxy behavior,
- station-network spread and spatial heterogeneity,
- thunderstorm warning text,
- rainfall text,
- lightning text,
- tropical cyclone text and tropical-cyclone proximity clues,
- missing station readings and outage patterns,
- warning/no-warning regime differences,
- high official-forecast-error precursors,
- and the interaction between official forecast residuals and observed pre-cutoff state.

You must not merely list these angles. You must decide which exact features should be implemented in phase one, which should be placed in a small phase-two queue, and which should be rejected for now because they are too noisy, too hard to implement, too weak, too sparse, too leakage-prone, or too duplicative. For every proposed feature group, explain the meteorological rationale, expected incremental information beyond the official forecast, exact source columns, cutoff eligibility rule, aggregation window, normalization, missingness handling, and ablation test. The answer should make it obvious that you have thought deeply about how Hong Kong Observatory Tmax actually forms, not just how to train a generic tabular model.

The feature plan must be compact but high-intelligence. Avoid an uncontrolled feature explosion. The goal is not to create 1,000 weak columns and hope a tree model finds something. The goal is to create a targeted feature matrix where each feature family exists because it has a defensible reason to reduce official-forecast residual error. If you propose interaction features, they must be few, named, and justified. Examples of acceptable targeted interactions include official forecast max by month, official forecast residual prior by issue hour, HKO pre-cutoff temperature anomaly by season, station-network spread by warm-season flag, and thunderstorm-warning indicator by forecast range width. Do not propose generic polynomial expansion or arbitrary pairwise interactions.

You must force yourself to think about feature normalization. A feature should not be dumped into the model raw if a better representation exists. For example, a temperature level feature may be better as an anomaly versus recent same-station climatology or versus official forecast max. A station-network feature may be better as HKO minus neighbor median, inland minus coastal mean, max-min spread, percentile rank of HKO among stations, or latest value minus earlier value. A forecast path feature may be better as latest forecast max, previous forecast max, delta from prior issue, number of revisions, issue-hour bucket, and revision direction. A text feature may be better as a small set of hand-built physical flags rather than a bag of words. You must specify these choices.

The final feature spec must let Codex implement a first serious model without guessing. If a feature requires SQL JSONB extraction from `station_readings_jsonb`, name the station groups or selection logic. If a feature requires hourly cutoff aggregation, name the hours or windows to use. If a feature requires forecast revisions, name the eligible issue sequence. If a feature requires target-history lags, name the lag offsets and rolling windows. If a feature requires climatology, specify expanding or training-fold-local computation so it cannot look forward. If a feature is unavailable on a date, specify whether to impute, null-preserve, add a missingness flag, or drop the row.

## Required Output Shape

Your answer must be a single Codex implementation task description. It must be directly actionable. Use clear headings and compact bullets. Avoid generic filler. Include all of the following sections.

### 1. Final Strategy Decision

State the exact strategy you recommend. Decide whether the best approach is:

- direct absolute-Tmax modeling,
- official-forecast residual modeling,
- calibrated official forecast plus residual model,
- stacked residual ensemble,
- regime-specific experts with a router,
- or another design.

Pick one primary design. Explain why it is the best use of the available data and why it is likely to improve over the `0.92 deg C` baseline.

### 2. Data Sources To Use

List each source table or data object to use, with its role:

- target labels,
- official forecast anchor,
- forecast revisions,
- hourly HKO target-station readings,
- neighbor-station readings,
- warning/text/tropical-cyclone fields,
- target-history lags and climatology,
- optional diagnostic sources only if they can be used safely from the documented data.

For each source, state:

- exact table/file name from the docs,
- join key,
- as-of key,
- coverage period,
- required row filters,
- what to do with nulls,
- whether it is core, optional, or rejected for phase one.

### 3. Cutoff And Issuance Policy

Define the primary decision cutoff. You must make a default decision, not only list options.

Also define a cutoff-sensitivity experiment over a small set of candidate cutoffs if needed. For each cutoff, specify:

- local HKT cutoff time,
- equivalent UTC logic,
- which forecast rows are eligible,
- which hourly readings are eligible,
- how to choose latest eligible forecast issue,
- how to construct forecast-revision features without using post-cutoff information,
- how to prevent target-day finalized target leakage.

### 4. Exact Train, Validation, And Test Ranges

Define exact date ranges. You must choose them.

Include:

- primary training range,
- validation range or walk-forward validation folds,
- locked test range,
- sealed confirmation policy for 2024+,
- how to handle 2026 rows,
- whether pre-2000 target history is used for climatology only,
- how to avoid using sealed data for model selection.

Explain why the ranges are optimal for this problem.

### 5. Target And Baselines

Define:

- target variable,
- residual target if using residual modeling,
- baseline 1: simple climatology or persistence,
- baseline 2: official forecast max,
- baseline 3: any current 0.92 MAE baseline recreation,
- scoring rows,
- metrics,
- minimum report outputs.

### 6. Feature Specification

This is the most important section. Provide a precise feature plan that Codex can implement.

This section must receive exceptional effort. It should be the core of the answer, not an afterthought. You must deeply inspect the documented data sources and propose the smartest compact set of features that can realistically drive a large MAE improvement. You must reason from the physics and from the data structure: what information does the official forecast already contain, what does it miss, and what signals in the hourly readings, forecast revisions, station network, warnings, target history, and seasonality could explain the residual? The answer must include enough detail that Codex can implement the feature matrix directly.

Group features into:

- official forecast anchor features,
- forecast revision and issuance-path features,
- hourly target-station state features,
- hourly neighbor-station network features,
- station-gradient and maritime/inland contrast features,
- warnings/rain/lightning/thunderstorm/tropical-cyclone text features,
- target-history lag and climatology features,
- calendar/seasonality/modern-era features,
- missingness and data-quality flags.

For each feature group, define:

- exact source fields,
- aggregation window,
- timestamp eligibility,
- transformation or normalization,
- missingness handling,
- expected meteorological rationale,
- expected information gain,
- implementation priority.

Do not create a huge feature explosion. Keep the phase-one feature set compact and high-signal. If you propose interactions, keep them targeted and explain why each one matters.

### 7. Model Architecture

Define:

- model family or families,
- whether the target is residual or absolute Tmax,
- input normalization,
- categorical handling,
- missing value handling,
- hyperparameter search boundaries,
- fold structure,
- model selection criterion,
- calibration or clipping,
- ensemble/stacking logic if any,
- uncertainty or diagnostic outputs if useful.

If recommending LightGBM or CatBoost, specify exactly why and how. If recommending a linear residual model as a guardrail, specify how it is combined. If recommending regime experts, define the router features and no-harm rule.

### 8. Ablation And Experiment Plan

Define the exact ablation sequence Codex should run. Include:

- baseline official forecast,
- official forecast plus simple residual correction,
- plus forecast revision features,
- plus hourly HKO features,
- plus neighbor network features,
- plus text/warning features,
- plus target-history features,
- final compact ensemble.

For each ablation, define pass/fail thresholds. Require MAE, RMSE, p90 absolute error, monthly slices, warm-season slices, and high-error-day slices.

### 9. Leakage And Data Integrity Tests

Specify tests Codex must implement before trusting scores:

- cutoff eligibility tests,
- no target-day label leakage,
- no post-cutoff hourly reading leakage,
- no sealed-data training leakage,
- no future normalization leakage,
- no raw Daily Extract predictor use,
- no apples-to-oranges forecast source mixing,
- duplicate-row checks,
- joined-row count checks,
- missingness reports.

### 10. Expected Result And Risk Assessment

State your realistic expected MAE improvement range. Be honest. Include:

- best-case MAE,
- realistic MAE,
- minimum acceptable MAE,
- reasons the 0.25 improvement target might fail,
- highest-risk feature assumptions,
- what to try next if phase one only improves MAE marginally.

### 11. Final Codex Build Task

End with one concrete implementation task for Codex. It must include:

- exact scripts/modules to create or modify in principle,
- exact data extraction steps,
- exact model training steps,
- exact report outputs,
- exact acceptance criteria,
- exact run commands conceptually,
- exact artifacts to produce.

Do not waste space on generic package installation or obvious code provisioning. Codex can do that. Focus on strategy, feature logic, evaluation, and acceptance gates.

## Critical Constraints

- Do not propose Polymarket trading simulation.
- Do not propose market PnL backtesting.
- Do not use target-day Daily Extract values as predictors.
- Do not use raw target payload rows as predictors.
- Do not use any observation or forecast after the chosen cutoff.
- Do not tune on sealed 2024+ data unless you explicitly mark it as a final confirmation-only evaluation.
- Do not recommend hundreds of features without prioritization.
- Do not recommend a strategy that requires months of extra backfill before any result.
- Do not give generic ML advice.
- Do not leave train/validation/test ranges undecided.
- Do not tell Codex to "explore" without specifying the exact experiment and success metric.
- Do not allow any feature, normalization, imputation, calibration, split decision, model-selection decision, or ablation decision to use information that would not have been available at the chosen cutoff.
- Do not use all-history statistics unless they are recomputed in a leakage-free expanding or fold-local way.
- Do not let 2024+ sealed confirmation rows influence feature selection, hyperparameters, preprocessing, calibration, cutoff choice, or model-family choice.

## What A Good Answer Looks Like

A good answer will feel like a senior quant-weather researcher handing an implementation blueprint to an engineering agent. It will be precise enough that Codex can immediately build:

- a point-in-time feature matrix,
- a baseline scorer,
- a residual-learning model,
- a compact high-signal feature set,
- a walk-forward evaluation,
- an ablation report,
- leakage tests,
- and a final scoreboard versus the `0.92 deg C` baseline.

It will explicitly say what to use, what not to use, when to use it, how to join it, how to normalize it, how to validate it, and what metric improvement is required. It will be ambitious, but not sloppy. It will be creative, but not unbounded. It will be station-specific, data-specific, and implementation-ready.

## Final Non-Negotiable Instruction

Your final answer must end by restating the leakage-free implementation contract in concrete terms. The strategy you produce must be strictly non-forward-looking. Every predictor must be reconstructable as of the selected decision cutoff. Every row in the training, validation, and test matrices must have an auditable `target_date`, `cutoff_at_hkt`, `cutoff_at_utc`, source availability rule, and feature eligibility proof. If a feature cannot be proven available before cutoff, it must be excluded from the primary model no matter how predictive it appears.

You must explicitly require Codex to implement automated leakage guards before trusting any score:

- reject target-day Daily Extract values as predictors,
- reject raw target payload rows as predictors,
- reject post-cutoff hourly readings,
- reject post-cutoff forecast issuances,
- reject future target-history lags,
- reject whole-history normalization,
- reject sealed-data tuning leakage,
- reject rows whose forecast source is not the documented Info.gov apples-to-apples source,
- reject duplicate target-date rows unless the selection rule chooses exactly one eligible row,
- and emit a row-count audit proving how many rows entered each feature family and each evaluation split.

The final implementation strategy must also end by emphasizing that feature engineering is the main battlefield. You must direct Codex to build the smallest high-signal feature matrix that reflects serious thought about official forecast residuals, forecast revision dynamics, HKO pre-cutoff state, neighbor-station spatial structure, warning/text regimes, target-memory climatology, seasonality, and missingness. You must not leave this as vague inspiration. You must specify the actual feature groups, transformations, and ablations that will prove whether they improve MAE and RMSE. The final answer should make clear that no reported improvement is valid unless it is both leakage-free and produced by features that were deliberately engineered for this exact HKO station problem.
