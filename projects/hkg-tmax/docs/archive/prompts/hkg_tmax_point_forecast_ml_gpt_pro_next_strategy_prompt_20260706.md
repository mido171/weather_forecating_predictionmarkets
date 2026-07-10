# GPT-Pro Prompt: HKG Tmax Point-Forecast ML Strategy Deep Analysis And Next-Round Specification

You are being given a ZIP archive containing the latest HKG Tmax point-forecasting experiment evidence from our local research repository.

Your mission is narrow and extremely important:

**Help us develop the best possible leakage-safe ML strategy to beat the official HKO/HKG Tmax point forecast.**

This is **not** a Polymarket-trading task. This is **not** a probability-bucket calibration task. This is **not** an EV, Kelly, order-book, or market-pricing task. Focus only on the point forecast: predicting the HKO Daily Extract absolute daily maximum temperature for HKG as accurately as possible, before the target date settles.

## Core Objective

We currently use the official HKO local forecast max as the anchor. The question is:

> Can we build a leakage-safe residual ML system that improves the official forecast enough to be worth promoting?

You must analyze the evidence in the ZIP deeply, critically, and competitively. We want a brutally honest and technically excellent answer. If the evidence says the official forecast is hard to beat, say so and explain why. If there is a route to beat it, give us the exact route.

Do not give generic advice like "try XGBoost", "add more features", "use ensembles", or "do feature engineering." We need a concrete scientific plan with exact experiments, exact features, exact validation, exact acceptance gates, exact failure checks, and exact reasons each step could plausibly improve MAE.

## Archive Contents To Read

Read the entire archive before answering. At minimum, inspect:

- `experiments/0215_gpt_pro_point_forecast_strategy/`
- `experiments/hkg_tmax_residual_ml_strategy/`
- `experiments/hkg_tmax_residual_ml_next_round/`
- `experiments/hkg_tmax/0001_residual_ml_strategy_20260705/`
- `experiments/hkg_tmax/0002_selective_no_harm_router_20260705/`
- `source_context/configs/hkg_tmax/`
- `source_context/scripts/`
- `source_context/code/src/hkg_tmax/`
- `source_context/code/tests/`
- `source_context/documentation/strategy_implementation_documentation/context/live_trading/`
- root context files such as `EXPERIMENT_INDEX.md`, `CHANGELOG.md`, and `PROJECT_STRUCTURE_AND_CODE_MAP.md`.

The organized `experiments/hkg_tmax/0002_selective_no_harm_router_20260705/` folder overlaps with `experiments/hkg_tmax_residual_ml_next_round/`, but inspect both because the organized folder contains the canonical experiment packaging.

## Known Headline Evidence

You must verify these numbers from the files rather than trusting this section blindly, but use them as navigation:

### GPT-Pro Lead-1 Point Forecast Strategy, 2026-07-04

Path:

`experiments/0215_gpt_pro_point_forecast_strategy/`

Primary selected cutoff:

`23:59 HKT on T-1`

Validation window:

`2011-01-01` through `2023-12-31`

Official rows:

`4,747`

Raw official baseline at 23:59:

- MAE: `0.92749 C`
- RMSE: `1.19152 C`
- Bias: `-0.12286 C`

Best selected grouped residual shrinkage:

- MAE: `0.92161 C`
- RMSE: `1.18268 C`
- Bias: `0.01270 C`
- MAE delta vs raw official: about `-0.00588 C`

Conclusion:

Tiny gain. Promotion gates failed.

### Broad Residual-ML Strategy, 2026-07-05

Path:

`experiments/hkg_tmax_residual_ml_strategy/`

Primary benchmark cutoff:

`T-1 23:59 HKT`

Primary model:

`A7_final_residual_ensemble`

Primary baseline:

`A0_raw_official`

Primary rows:

`5,629`

Raw official:

- MAE: `0.930858 C`
- RMSE: `1.195757 C`
- Bias: `-0.122935 C`

A7 final residual ensemble:

- MAE: `0.898665 C`
- RMSE: `1.154088 C`
- Bias: `-0.000486 C`

Improvement:

- MAE improvement: about `0.032193 C`

Conclusion in model card:

`no_promote_cosmetic`, because the configured promotion threshold was `0.035 C` MAE improvement and the model came in slightly below it.

Important feature/model context:

- Feature count: `323`
- Leakage audit: pass
- CatBoost status: fit
- Sealed rows were not used for model selection
- Raw Daily Extract payload rows were not used as predictors

### Residual-ML Next Round / Selective Router, 2026-07-05

Paths:

`experiments/hkg_tmax_residual_ml_next_round/`

`experiments/hkg_tmax/0002_selective_no_harm_router_20260705/`

Primary question:

Should residual correction be applied selectively instead of on every row?

Primary cutoff:

`T-1 23:59 HKT`

Raw official:

- MAE: `0.930858 C`

Current A7 reproduction:

- MAE: `0.898665 C`

C1 pruned residual ensemble:

- MAE: `0.901052 C`

C2 selective router:

- MAE: `0.902930 C`

C3 tail overlay router:

- MAE: `0.902893 C`

Conclusion:

`no_promote`

Reason:

`router edge too small: c2_vs_raw=0.027927927881359227, c2_vs_a7=-0.004264699170343111`

Important audit context:

- Leakage audit: pass
- No-harm audit: pass
- Selected raw feature count: `64`
- Sealed confirmation rows were not used for threshold selection, model selection, feature selection, calibration, or hyperparameter tuning
- Raw official-error slices and helped/worsened labels are evaluation-only columns and not model features

## What We Need From You

Produce a deep, structured research memo with the following sections.

### 1. Executive Verdict

State clearly:

- Whether the latest experiments show a genuinely promotable ML point-forecasting edge.
- What the current best model is, on which frame, and with which metric.
- Whether the current "champion" is still raw official, A7, grouped residual shrinkage, or something else.
- Whether the observed gain is meaningful enough to deploy, continue researching, or reject.

Do not blur frames. Separate the 4,747-row `2011-2023` strategy frame from the 5,629-row residual-ML frame if they are not identical.

### 2. Evidence Inventory

Build a concise but complete evidence inventory from the archive:

- Experiment folder.
- Date.
- Primary model.
- Baseline.
- Cutoff.
- Row count.
- Date range.
- MAE/RMSE/bias/p90 AE.
- Claimed conclusion.
- Leakage status.
- Promotion status.

Call out any missing files, contradictory frames, duplicated outputs, or unclear artifacts.

### 3. What Worked

Analyze what actually helped:

- Which residual corrections reduced bias?
- Which feature families had evidence of real signal?
- Which cutoff was strongest and why?
- Which model families were stable?
- Did shrinkage/empirical residual methods beat heavier ML?
- Did official forecast revision features, target memory, station/network gradients, hourly state, text/warnings, or tail specialists contribute real incremental value?

Quantify everything. Use actual scoreboard numbers.

### 4. What Failed

Analyze what did not work:

- Why did broad ML only improve MAE by about `0.032 C`?
- Why did C1/C2/C3 fail to beat A7?
- Why did selective routing underperform?
- Did the router fail because the correction signal is weak, the confidence model is weak, thresholds are wrong, feature pruning removed useful interactions, target labels are noisy, or the official forecast already encodes most predictable information?
- Are tail specialists helping tails but hurting average MAE?
- Are we overfitting the folds?
- Are the folds too forgiving or too harsh?

Be forensic. Identify the most likely failure mechanisms and show what file/metric supports each one.

### 5. Official Forecast Anatomy

Analyze the official forecast system as an anchor:

- How much residual bias remains after official forecast max?
- Is the official forecast error mostly bias, random noise, regime-specific, seasonal, cutoff-specific, or tail-specific?
- Which slices show the official forecast is most beatable?
- Which slices should we avoid correcting?
- Is there evidence that later cutoff forecasts are materially better than earlier cutoffs?
- Are revisions informative, or mostly already reflected in the final official max?

The goal is to find where the official forecast is vulnerable, not to build a generic model.

### 6. The Best Next Strategy To Beat Official

Give us the exact best path forward. Be decisive.

You must propose:

- The next primary experiment.
- Why it dominates the alternatives.
- The exact hypothesis.
- The exact target variable and sign convention.
- The exact baseline and identical-row requirement.
- The exact cutoff(s).
- The exact train/validation/holdout/sealed protocol.
- The exact feature families to include and exclude.
- The exact modeling architecture.
- The exact hyperparameter-selection protocol.
- The exact acceptance gates.
- The exact no-harm gates.
- The exact leakage tests.
- The exact diagnostic slices.
- The exact artifacts to produce.

Do not leave scientific implementation choices vague.

### 7. Candidate Lane Ranking

Rank at least 10 candidate next research lanes.

Each candidate must include:

- Expected information gain.
- Expected deployable MAE lift.
- Physical/weather rationale.
- Prior evidence from the archive.
- Novelty versus experiments already attempted.
- Leakage risk.
- Sample support.
- Implementation cost.
- Reason it might fail.
- Your score from 1-100.

Reject weak candidates explicitly. We want competitive rigor, not a list of "possible ideas."

### 8. Residual Signal Discovery Plan

Give a concrete plan for discovering where residual signal actually exists.

Consider:

- Error-sign response.
- Absolute-error response.
- High-tail / cold-tail response.
- Forecast-revision response.
- Seasonal and submonthly regimes.
- Transition days.
- Heat storage and previous-day thermal memory.
- Moisture / dewpoint / wet-bulb proxies.
- Cloud/rain/sun proxies.
- Wind and marine influence.
- Station-network gradients and upwind selection.
- HKO forecast text signals.
- Regime interaction terms.
- Error autocorrelation and online residual memory.

For every proposed signal, specify how to prove it is point-in-time eligible.

### 9. Exact Experiment Spec For Codex

Write one final implementation-ready specification that we can hand directly to Codex.

It must be executable without guesswork and include:

- Experiment ID and folder path.
- Files to add/edit.
- Config fields.
- Data sources.
- SQL/table dependencies.
- Feature formulas.
- Model choices.
- Validation folds.
- Acceptance gates.
- Audit outputs.
- Scoreboards.
- Required tests.
- Reproduction command.

Make this the sharpest possible next experiment, not a broad exploratory kitchen sink.

### 10. Hard Red-Team

Attack your own recommended plan:

- Where could it leak?
- Where could it overfit?
- Where could it merely exploit frame mismatch?
- Where could it improve historical MAE but fail live?
- What would falsify it?
- What minimum result would make us stop pursuing that lane?

### 11. Final Recommendation

End with a short, decisive recommendation:

- Current champion.
- Whether to deploy or not deploy.
- The exact next experiment to run.
- The expected realistic MAE improvement range.
- The biggest uncertainty.
- The one thing we must not do.

## Non-Negotiable Standards

- No trading analysis.
- No market prices.
- No probability buckets except as irrelevant context if absolutely needed.
- No hidden leakage.
- No sealed tuning.
- No target-derived predictors.
- No raw Daily Extract publication fields as predictors.
- No adaptive thresholding on sealed confirmation.
- No claim of superiority unless scored on identical rows.
- No generic ML advice.
- No invented data source unless you specify exact acquisition, timestamp, availability, and leakage proof.

Your goal is to give us the most competitive, evidence-grounded route to a genuinely sharper HKG Tmax point-forecast model.
