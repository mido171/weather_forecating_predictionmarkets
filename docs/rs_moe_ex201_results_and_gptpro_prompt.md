# RS‑MoE (EX201) Results Post‑Mortem + GPT‑Pro Prompt Pack (≈7,000 words)

Date: 2026‑01‑29  
Repo: `weather_forecating_predictionmarkets`  
Primary focus: the `time_feature_sweep_trees` ML sweep pipeline and the newly implemented RS‑MoE mean model (`EX201`).

This is written as a copy/paste‑ready package you can use to start a new GPT‑Pro conversation about what to do next.

---

## Part 1 — Recent results (≈2,000 words)

### 1) What we just ran, and what “success” means

We recently implemented and ran **EX201: RS‑MoE (Regime‑Switching Mixture of Experts)** inside the existing sweep runner. The intention was straightforward: reduce **test MAE** for next‑day `tmax` (temperature maximum, Fahrenheit) predictions by combining:

1) a 3‑class “bust regime” **gate** (`cool / normal / warm`) that outputs calibrated probabilities, and  
2) three specialized **regression experts** (one per regime),  
3) combined at inference as a probability‑weighted mixture:

`mu_hat = p_cool*mu_cool + p_normal*mu_normal + p_warm*mu_warm`

In this repo, “success” is not a vibe or a story — it’s a number: **test MAE** in `metrics.json`. Everything else (RMSE, bias, corr, medianAE, maxAE) is supporting evidence to diagnose why MAE went up or down.

Critically, in this workflow, the TEST split is treated as sacred: we do not tune on it. So when we say “EX201 worked” or “EX201 failed,” we mean: *it generalized better on the held‑out test period than BASE, using the same dataset and split definition*.

### 2) The dataset and split that produced the EX201 verdict

The “recent results” you’re reacting to are from the KMIA gribstream dataset used by `time_feature_sweep_trees`. For the RS‑MoE run we’re discussing:

- Train `n = 1162`
- Validation `n = 213`
- Test `n = 329`
- Total `n = 1704`

This is small enough that added complexity has to pay rent. Mixture models and gates can be very powerful, but with only ~1.1k train rows, you can easily lose more to variance and instability than you gain from conditional specialization — especially if the regimes are unbalanced or poorly separated.

The sweep run happened on **2026‑01‑29** and the sweep root is:

`artifacts/time_feature_sweep_trees/20260129T154031Z/`

Within that sweep root, the runner produced both an `xgb/` and an `lgbm/` subtree because the CLI included `--models xgb lgbm`. However, **EX201 currently trains the same RS‑MoE internal model regardless of that sweep label** (CatBoost gate + XGBoost experts). That’s why xgb/EX201 and lgbm/EX201 match exactly.

### 3) Baseline vs RS‑MoE: the numbers (core finding)

The clean apples‑to‑apples comparison is **XGB BASE vs XGB EX201** within the same sweep root, because XGB BASE is the strongest baseline for this KMIA dataset at the moment.

**Baseline (XGB BASE)** metrics:
`artifacts/time_feature_sweep_trees/20260129T154031Z/xgb/BASE/metrics.json`

- Train: MAE **0.6538777**, RMSE **0.8499879**, bias **0.0023440**, corr **0.9901021**
- Validation: MAE **1.3903519**, RMSE **1.8910675**, bias **0.8054517**, corr **0.9710287**
- Test: MAE **1.0629299**, RMSE **1.4127988**, bias **0.4760069**, corr **0.9660146**

**RS‑MoE (EX201)** metrics:
`artifacts/time_feature_sweep_trees/20260129T154031Z/xgb/EX201/metrics.json`

- Train: MAE **0.8067911**, RMSE **1.3294711**, bias **0.0647962**, corr **0.9763229**
- Validation: MAE **1.3939534**, RMSE **1.9488609**, bias **0.8055047**, corr **0.9685243**
- Test: MAE **1.1080269**, RMSE **1.5139200**, bias **0.5801146**, corr **0.9625643**

Now the deltas vs baseline (EX201 − BASE):

- Test MAE: **+0.04510** (**+4.24%**) → *worse*
- Validation MAE: **+0.00360** (**+0.26%**) → *basically flat, slightly worse*
- Train MAE: **+0.15291** (**+23.4%**) → *worse*
- Test RMSE: **+0.10112** (**+7.16%**) → *worse*
- Test bias: **+0.10411** → *worse bias*
- Test corr: **−0.00345** → *slightly lower correlation*
- Test medianAE: **+0.03148** → *worse*
- Test maxAE: **+1.82393** → *worse tails / catastrophic errors*

So, directly answering your question: **yes — this new RS‑MoE strategy did not improve the model meaningfully and instead made the main score worse** (and it worsened other metrics too).

### 4) Why the RS‑MoE miss matters: it’s not just “noise”

There are “lossy” experiments where MAE changes by a hair and you can’t tell if it’s noise, and there are failures where the whole error profile shifts in a coherent bad direction. This looks like the second type:

- RMSE got worse by ~7%, meaning the squared‑error tail got heavier.
- maxAE increased by ~1.82°F, meaning the worst errors became significantly worse.
- Bias increased, meaning the model became more systematically offset on test.

These are exactly what you’d see if the model sometimes routes probability mass to a poorly fit expert on days where the baseline was already decent — the mixture introduces extra ways to be wrong.

### 5) “But mixture of experts should add capacity — why is TRAIN worse too?”

At first glance it’s surprising that **train MAE** got worse (0.654 → 0.807). You added more models, so why can’t it at least fit training data?

In this implementation, the experts are trained using **soft sample weights** derived from **out‑of‑fold gate probabilities** (OOF). That has two practical effects:

1) Each expert gets a *weighted* view of the training set. For minority regimes, the effective weighted sample size can be tiny, making it hard for that expert to learn anything stable.

2) If the gate is uncertain/noisy, the weights “smear” across regimes and each expert learns a similar global function. In that case the mixture becomes a complicated average of similar predictors plus routing noise — and it can underfit relative to one direct model.

Also, the baseline XGB model is already quite strong in this small‑tabular setting. MOE helps when there is real, learnable conditional structure (distinct residual regimes) and the gate can reliably detect them. If that structure is weak (or mislabeled), MOE just adds variance.

### 6) Gate diagnostics: class imbalance is the structural killer

The EX201 report includes “Gate Diagnostics” computed on validation. The key fact is class prevalence.

From `artifacts/time_feature_sweep_trees/20260129T154031Z/xgb/EX201/report.md`, the regime counts are approximately:

- Train: cool **2**, normal **697**, warm **463** (n=1162)
- Validation: cool **1**, normal **162**, warm **50** (n=213)
- Test: cool **0**, normal **260**, warm **69** (n=329)

This means the “cool” class is effectively absent, and is **entirely absent from test**. That is devastating for a 3‑class MOE:

- The gate cannot learn a meaningful “cool” boundary from 2 examples.
- The cool expert can’t specialize; it’s basically undertrained noise.
- The evaluation can’t even compute cool‑stratified test MAE because there are no true cool rows in test.

Even if the gate assigns very small `p_cool`, any non‑zero mass routed to a poorly trained cool expert can increase errors, especially in the tails.

### 7) Gate quality: it’s not separating warm vs normal cleanly (yet)

Beyond imbalance, the second gate issue is performance. The report shows gate validation accuracy around **0.643**. But on validation the true class prevalence is roughly:

- normal ≈ 162/213 ≈ **0.761**
- warm ≈ 50/213 ≈ **0.235**
- cool ≈ 1/213 ≈ **0.005**

A trivial classifier that always predicts “normal” would get ~0.761 accuracy — **higher than the gate**. Accuracy isn’t the only metric that matters (logloss matters too, and the mixture ultimately matters most), but this is a major warning sign: the gate isn’t carving the space into “things that should be routed differently” in a way that aligns with the defined labels.

The confusion matrix in the report is consistent with that: a significant fraction of normal days are predicted warm and a significant fraction of warm days are predicted normal. If the gate frequently mixes up warm and normal, the mixture cannot be reliable because the wrong expert gets weight on many days.

There are also calibration/uncertainty hints:

- average max probability is ~0.679 (so it’s not extremely confident),
- average entropy is ~0.608 (moderate uncertainty),
- temperature scaling ended up with **T = 1.0** in this run (effectively no calibration change).

Temperature of 1.0 doesn’t necessarily mean “calibration is perfect.” It can also mean: the calibration step didn’t have enough stable signal to move away from the initialization, or the optimization landscape is flat for this dataset size and label distribution.

### 8) Expert diagnostics: the experts do not look like clean specialists

The “Expert Diagnostics” in the report show two things you should interpret together:

1) which features the experts rely on, and  
2) whether each expert is actually good on the subset of cases it is weighted toward.

In an ideal MOE:

- Each expert should be best on the cases where the gate assigns it high probability, and
- The gate should assign high probability to the expert that’s actually best for those cases.

What we see instead is a messy specialization picture, especially for the cool expert:

- Cool expert (validation): unweighted MAE ≈ **1.93**, weighted MAE ≈ **4.45**
- Normal expert (validation): unweighted MAE ≈ **1.39**, weighted MAE ≈ **1.31**
- Warm expert (validation): unweighted MAE ≈ **1.43**, weighted MAE ≈ **1.53**

The standout is the cool expert: **its weighted MAE is extremely high**. That implies that on the subset of validation rows that the gate leans “cool” toward, the cool expert performs terribly. That creates a toxic feedback loop:

- The gate assigns some probability to cool on some days (even if the true cool class is rare).
- The cool expert’s prediction on those days is poor.
- The mixture prediction is pulled away from the truth.
- Tail errors can spike if the misrouted days are already difficult.

Also notice how similar the top features are across experts: they’re all leaning heavily on the same core forecast predictors (`gefsatmosmean_tmax_f`, `nbm_tmax_f`, `gfs_n_x_max`, `nam_n_x_max`, plus seasonal sin/cos). That suggests the experts are not discovering fundamentally different mappings — they’re variants of the same relationship with different weighting.

### 9) Regime‑stratified test MAE: warm improved, normal worsened (and normal dominates)

The regime‑stratified MAE reported on test is approximately:

- `test_mae_given_true_normal` ≈ **1.1498** (n=260)
- `test_mae_given_true_warm` ≈ **0.9507** (n=69)
- `test_mae_given_true_cool` = null because `test_n_true_cool = 0`

This is a key nuance. RS‑MoE may actually be doing something useful for warm cases. But overall test MAE is worse because:

1) normal is ~79% of test, and  
2) normal‑case MAE is materially worse than baseline.

So the mixture, as configured, is trading off: “better on warm” for “worse on normal,” and the weights of the dataset make that a losing trade.

This is exactly what you get when the gate is noisy: it routes some normal days as warm (and vice versa), and the experts don’t have a strong enough edge to overcome routing errors.

### 10) LGBM vs XGB, and why EX201 looked identical under both

You asked to “try with LGB/XGBOOST models.” We did, and here’s what happened:

- XGB BASE (KMIA): test MAE **1.0629**
- LGBM BASE (KMIA): test MAE **1.1204**
- EX201 (KMIA): test MAE **1.1080**

So EX201 is:

- worse than the best baseline (XGB BASE),
- but slightly better than the LGBM baseline.

The reason EX201 is identical under `xgb/` and `lgbm/` is implementation: RS‑MoE uses CatBoost for gate and XGBoost for experts regardless of sweep “model.” So it’s not a real LGBM‑flavored RS‑MoE yet.

Also notice how LGBM BASE behaves: train MAE is extremely low (0.438) while val/test are much higher (1.566 / 1.120). That looks like a stronger overfit profile in this exact configuration, whereas XGB BASE has a more balanced generalization profile (train 0.654, val 1.390, test 1.063).

### 11) Context from other experiments (qualitative, not directly comparable)

Two other artifacts in your open tabs provide helpful qualitative context:

1) **Multi‑station EX19** (`KMIA/KPHL/KMDW/KNYC`):
   - Test MAE around **1.58** on a much larger, more heterogeneous dataset (~7k rows).
   - Multi‑station learning is harder: the mapping from forecast features to realized tmax changes by geography and climate regime.

2) **MOS pipeline report JSON** (`ml/artifacts/mos_train_runs/.../report.json`):
   - Thousands of engineered features (feature_count ~3682).
   - Model: `HistGradientBoostingRegressor`.
   - Test MAE around **1.256** (but on a different dataset definition and different time split).

The headline: there’s real room in this project for “feature richness + bias correction + robust evaluation,” but every result must be compared within its own dataset/split regime. Use cross‑artifact comparisons only to generate hypotheses, not to declare winners.

### 12) Why you saw “features imputed” and why it exists

Your sweep JSON includes a block like:

- `derived_features.imputation.fill_values: {feature_name: value}`

This means the pipeline has a defined imputation plan and records the actual values used for reproducibility. It exists because missingness can come from:

- upstream data gaps (some forecast sources missing on some dates),
- engineered features (rolling windows/lags at boundaries),
- station/product availability shifts over history.

Even if a dataset summary shows “missing_by_column: 0” for a given run, the sweep definition can still record the imputation contract. If missingness does occur, the model sees filled values rather than NaNs (unless a given model can natively handle NaNs and you deliberately choose to keep them).

This matters because missingness can be **structured**: if certain models are missing more often in certain seasons or extreme events, naive imputation can inject systematic biases and regime‑dependent error. That’s a fertile next‑experiments area (missingness indicators, per‑season fills, model‑native NaN handling, etc.).

### 13) Bottom line: what EX201 taught us (and why that’s useful)

EX201 failed, but it wasn’t a useless failure. It gave us crisp evidence about what needs to change if we want conditional models to help:

1) The current 3‑class regime definition produces a near‑empty class and even absent class in test.
2) The gate isn’t strong enough on warm vs normal to make routing reliable.
3) The experts aren’t behaving like clean specialists; at least one (cool) looks actively harmful when weighted.
4) The baseline is already strong on KMIA — meaning wins likely come from:
   - better features (especially residual‑predictive features),
   - bias correction that respects season/spread,
   - carefully constrained conditional logic that cannot catastrophically misroute,
   - robust ensembling/stacking that improves without blowing up tails.

That is exactly what the next “60 experiments” brainstorming should target: systematically test the biggest plausible sources of residual structure and extract signal without adding noise.

---

## Part 2 — Context pack for the next GPT‑Pro convo (≈1,000 words)

### A) Where the key artifacts are (copy/paste friendly)

Latest sweep root:

`artifacts/time_feature_sweep_trees/20260129T154031Z/`

Key files:

- `artifacts/time_feature_sweep_trees/20260129T154031Z/tree_sweep_results.json`
- `artifacts/time_feature_sweep_trees/20260129T154031Z/xgb/BASE/metrics.json`
- `artifacts/time_feature_sweep_trees/20260129T154031Z/xgb/BASE/report.md`
- `artifacts/time_feature_sweep_trees/20260129T154031Z/xgb/EX201/metrics.json`
- `artifacts/time_feature_sweep_trees/20260129T154031Z/xgb/EX201/report.md`
- `artifacts/time_feature_sweep_trees/20260129T154031Z/xgb/EX201/predictions_test.parquet`
- `artifacts/time_feature_sweep_trees/20260129T154031Z/lgbm/BASE/metrics.json`
- `artifacts/time_feature_sweep_trees/20260129T154031Z/lgbm/EX201/metrics.json`

Other context you may reference (different datasets):

- `artifacts/time_feature_sweep_trees/multi_kmia_kphl_kmdw_knyc/20260128T084152Z/xgb/EX19/report.md`
- `artifacts/time_feature_sweep_trees/kphl/20260128T073148Z/xgb/time_feature_sweep.json`
- `ml/artifacts/mos_train_runs/20260128T184224Z/report.json`

### B) The exact command pattern used to run these sweeps

The latest RS‑MoE run was produced by:

`python scripts/run_time_feature_sweep_trees.py --config ml/configs/train_mean_sigma_gribstream_cli.yaml --models xgb lgbm --experiment-ids EX201`

So when proposing new experiments, it is most useful to express them as:

- “Add experiment id EX### to registry + add config block + run sweep with `--experiment-ids EX###`”

### C) Baseline target to beat (KMIA gribstream)

The current “score to beat” on KMIA is:

- XGB BASE test MAE ≈ **1.06293**

If an experiment doesn’t beat that, it’s not the best next step (unless it unlocks a later ensemble or a new feature set that beats it).

### D) Common pitfalls (so proposals are grounded)

If you propose conditional models (MOE, regimes, gating), you must explicitly guard against:

1) regimes absent in test (as happened with cool),
2) gates that do not beat trivial baselines,
3) experts that are not actually best on their assigned subsets,
4) catastrophic tails (maxAE spikes) due to misrouting.

If you propose any “imputation” or “data cleaning,” you must explicitly state:

- how it avoids leakage (train‑only statistics),
- how it handles time splits (no using future data to fill earlier rows),
- whether missingness indicators are included.

### E) What the core metrics mean (and how to read them quickly)

When GPT‑Pro suggests improvements, you want it to explicitly reason about *which error mode it is attacking*. These metrics map to different failure modes:

- **MAE**: average absolute error. This is your primary objective. MAE improves when you reduce typical errors across the distribution (or reduce a modest number of medium errors). MAE is robust to outliers compared to RMSE.

- **RMSE**: emphasizes large errors due to squaring. If RMSE moves a lot while MAE barely moves, you likely changed tail behavior (catastrophic mistakes) more than typical performance. In EX201, RMSE got materially worse, which is a red flag for tail misrouting or instability.

- **Bias**: mean signed error (`mean(y_hat − y_true)` in many conventions; double‑check sign in your code). Bias tells you whether you are systematically too warm or too cool. In EX201, test bias increased, suggesting the mixture’s mean prediction shifted systematically.

- **Corr**: correlation of predictions with truth. Useful as a sanity check: if corr collapses, something is very wrong. But corr can remain high even if MAE is mediocre; it mostly measures rank/linear alignment, not calibration.

- **medianAE**: median absolute error. This approximates the “typical” case and is less sensitive to tails than MAE. If medianAE improves but MAE worsens, you might be helping typical days while harming hard/extreme days.

- **maxAE**: worst absolute error. This is the “catastrophic failure” metric. You generally want improvements that do not blow up maxAE (or at least explain why and how you’ll control it). EX201 increased maxAE, which is consistent with occasional disastrous routing or expert failure.

If GPT‑Pro proposes an experiment that “improves MAE” but gives no plan for tails (maxAE/RMSE), treat that proposal as incomplete.

### F) What your reports already contain (so the prompt can demand targeted diagnostics)

Most `report.md` files in this repo include:

- **Dataset Summary**: date coverage, missingness by column, row counts, split counts, station counts (in multi‑station settings).
- **Model Summary**: library/params used for the current experiment.
- **Metrics Summary**: metrics for train/validation/test in a JSON block.
- **Feature Importance**: top feature importances when supported by the model.
- **Config Snapshot**: resolved config settings for reproducibility.

EX201 adds an additional RS‑MoE section that includes:

- **RS‑MoE Summary**: gate/expert params, OOF settings, calibrated temperature.
- **Gate Diagnostics**: logloss, accuracy, precision/recall, confusion matrix, class prevalence, entropy/confidence.
- **Expert Diagnostics**: weighted/unweighted MAE on validation per expert and top features.
- **Mixture Metrics Summary**: standard per‑split metrics for the mixture prediction.
- **Regime‑stratified MAE**: MAE on test conditional on the true regime label.

In a new GPT‑Pro brainstorming session, you can ask for additional diagnostics that are already easy to compute and very informative for MAE:

- MAE by month / season (or by sine/cosine bins),
- MAE by predicted temperature quantile (cold vs hot days),
- MAE by forecast spread quantile (high uncertainty vs low uncertainty),
- bias by month and by predicted temperature bins,
- “error vs ensemble mean” scatter with residual trends,
- error breakdown by which forecast source was missing/imputed (if any).

### G) RS‑MoE prediction schema changes (why they matter)

When EX201 runs, `predictions_test.parquet` is extended (in addition to the legacy columns your pipeline already writes). The key added columns are:

- `p_cool`, `p_normal`, `p_warm` (calibrated gate probabilities)
- `mu_cool`, `mu_normal`, `mu_warm` (expert predictions)
- `mu_hat_f` (the mixture prediction used for metrics; mixture identity should match exactly)
- `gate_temperature` (temperature scalar used in calibration, if stored per row)
- `model_type = "rs_moe"` (or equivalent tag)

Even if you move away from MOE, this schema change pattern is useful: many future experiments can write extra per‑row diagnostic columns (e.g., “baseline residual estimate,” “spread bucket,” “season bucket,” “missingness flags”) to make debugging MAE much faster.

### H) How to quickly pull feature lists and compare experiments safely

When GPT‑Pro proposes a feature experiment, make sure it references where feature lists live in artifacts:

- In sweep summaries (`tree_sweep_results.json`), each experiment often includes `final_feature_columns`.
- Inside an experiment directory, many runs also write `experiment_feature_columns.json` (or similar).

This matters because a “win” with extra features is not directly comparable to a baseline with fewer features unless you treat the baseline as “feature‑set‑specific.” A clean workflow is:

1) lock a feature set, iterate on models/losses; then
2) iterate on feature set changes with controlled ablations; then
3) build ensembles across the best feature sets.

---

## Part 3 — GPT‑Pro prompt (≈4,000 words)

Copy/paste everything in the code block below into a new GPT‑Pro chat.

```text
You are GPT‑Pro. Adopt the persona of a genius‑level quant + ML researcher + applied weather forecaster.

This is a high‑stakes research sprint. I need you to be:
- extremely creative and rigorous,
- aggressively practical (implementation‑ready, not hand‑wavy),
- obsessed with identifying what will actually reduce out‑of‑sample MAE,
- ultra clear and unambiguous in every instruction,
- patient enough to think deeply before writing.

Your mission: design a sequence of new experiments that materially reduces test MAE for next‑day maximum temperature (tmax, Fahrenheit) forecasting using my existing pipeline and artifacts.

Primary objective:
- Reduce TEST MAE (same definition used in my `metrics.json` and `report.md`).

Secondary metrics (must still be computed and reported for train/validation/test):
- RMSE, bias, corr, medianAE, maxAE, n

Hard rules (non‑negotiable):
- Treat TEST as sacred: NEVER tune on it. Only evaluate final results on it.
- No forward‑looking leakage: any feature computation must only use information available at `asof_utc` for that row.
- Any cross‑validation must be time‑respecting (blocked folds, expanding windows, or similar).
- No hidden magic defaults: if a parameter matters, it must be explicit.
- Be precise: if you propose an experiment, you must provide enough detail that I can implement it directly.

----------------------------------------------------------------------------------------------------
CONTEXT (you must incorporate this evidence)

1) Pipeline & command pattern
I run experiments through a sweep runner that writes artifacts per experiment:
- `report.md`, `metrics.json`, `predictions_test.parquet`
- `config_resolved.yaml`, `experiment_meta.json`, `hashes.json`

Recent run command (EX201):
python scripts/run_time_feature_sweep_trees.py --config ml/configs/train_mean_sigma_gribstream_cli.yaml --models xgb lgbm --experiment-ids EX201

Latest relevant sweep root (KMIA, small dataset):
artifacts/time_feature_sweep_trees/20260129T154031Z/

2) Dataset and split (KMIA gribstream)
Total rows: 1704
- Train n=1162
- Validation n=213
- Test n=329

3) Baseline results to beat (do not hand‑wave; use these numbers)
XGB BASE metrics:
- Train MAE 0.6538777, RMSE 0.8499879, bias 0.0023440, corr 0.9901021
- Val   MAE 1.3903519, RMSE 1.8910675, bias 0.8054517, corr 0.9710287
- Test  MAE 1.0629299, RMSE 1.4127988, bias 0.4760069, corr 0.9660146

This XGB BASE (test MAE 1.06293) is the primary target to beat.

LGBM BASE metrics (same dataset/split, different model):
- Train MAE 0.4384485, RMSE 0.6870386, bias ~0, corr 0.9935761
- Val   MAE 1.5662975, RMSE 2.0636249, bias 0.9050762, corr 0.9657097
- Test  MAE 1.1204341, RMSE 1.4821803, bias 0.4069771, corr 0.9613310

4) RS‑MoE attempt (EX201) failed (you must learn from this failure)
We implemented EX201: a 3‑class bust regime gate (cool/normal/warm) + 3 expert regressors, combined as:
mu_hat = p_cool*mu_cool + p_normal*mu_normal + p_warm*mu_warm

EX201 metrics:
- Train MAE 0.8067911, RMSE 1.3294711, bias 0.0647962, corr 0.9763229
- Val   MAE 1.3939534, RMSE 1.9488609, bias 0.8055047, corr 0.9685243
- Test  MAE 1.1080269, RMSE 1.5139200, bias 0.5801146, corr 0.9625643

So EX201 is WORSE than XGB BASE on test MAE by +0.04510 (+4.24%).
It also worsened RMSE, bias, and maxAE on test.

5) RS‑MoE gate diagnostics (extreme regime imbalance)
Regime counts:
- Train: cool 2, normal 697, warm 463 (n=1162)
- Val:   cool 1, normal 162, warm 50  (n=213)
- Test:  cool 0, normal 260, warm 69  (n=329)

This means the “cool” regime basically does not exist and is absent in test entirely.
Gate validation accuracy was ~0.643 while “always normal” would be ~0.761, so the gate is weak.
Regime‑stratified test MAE for EX201:
- True normal: ~1.1498 (n=260)
- True warm:   ~0.9507 (n=69)
- True cool:   none in test

So warm improved but normal got worse and dominates test.

6) Feature imputation metadata exists in sweep definitions
My sweep JSON stores `derived_features.imputation.fill_values` (computed from TRAIN) to handle missing feature values reproducibly. Missingness can be structured and could impact performance.

7) Broader context (qualitative only; don’t mix metrics across different datasets)
I have multi‑station runs and a separate MOS feature pipeline with thousands of features; use them only to generate hypotheses, not to claim direct comparability.

----------------------------------------------------------------------------------------------------
YOUR TASK: PROPOSE EXACTLY 60 NEW EXPERIMENTS

I want exactly 60 uniquely distinct, high‑quality experiments that aim to substantially improve TEST MAE on the KMIA gribstream dataset (n=1704).

Each experiment must:
- be truly distinct (not a trivial hyperparameter tweak),
- be grounded in logic/science/math (but still implementable),
- be leakage‑safe,
- include an implementation plan with explicit config changes and concrete code touchpoints,
- state a numerical target improvement range vs XGB BASE (e.g., “−0.02 to −0.05 MAE”),
- include failure modes and diagnostics.

You must explicitly avoid repeating what seems not to work:
- naive 3‑class MOE when a class is near‑empty or absent in test
- weak gate routing that cannot beat trivial baselines

However, you MAY propose conditional/MOE follow‑ups ONLY if you fix the failure modes (balanced regimes, better labeler, better gate features, regularization, strict evaluation).

----------------------------------------------------------------------------------------------------
REQUIRED OUTPUT FORMAT (strict)

0) If you need clarifications, ask up to 10 questions; otherwise skip questions and proceed immediately.

1) One‑page executive summary
Include:
- your diagnosis of the biggest current bottlenecks
- the most promising 5 “fast win” directions
- a prioritized 3‑stage roadmap (Stage 1 fast wins, Stage 2 medium, Stage 3 big bets)

2) Index table for all 60 experiments
For each include:
- Exp ID (use placeholders EX202..EX261)
- Name
- Category
- Expected MAE impact (small/medium/large) + numeric target range
- Risk (low/med/high)
- Implementation time estimate (hours)
- Compute cost (low/med/high)
- Key failure mode to watch

3) Detailed spec for each experiment (repeat this structure 60 times)
For each experiment, include EXACTLY these subsections:

3.1 Hypothesis
Explain why this should reduce test MAE in this specific setting and how it relates to the observed failure of EX201 and observed baseline behavior.

3.2 What Changes (delta vs baseline)
List exactly:
- feature changes
- model/training changes
- loss/objective changes
- calibration/post‑processing changes
- any new data requirements (if any) and how you would ingest it safely

3.3 Implementation Plan (step‑by‑step)
Must include:
- concrete YAML config snippets/diffs
- code touchpoints (file/module names and the kind of changes)
- expected artifacts and how to verify them

3.4 Leakage & Correctness Guardrails
Explicit checks/assertions (examples):
- “asof_utc must be respected”
- “fit only on TRAIN”
- “calibration only on OOF or validation, never test”
- “time‑blocked folds only”
- “assert no future rows leak into training window”

3.5 Evaluation Plan
Must include:
- which metrics and splits to compare
- which stratifications to compute (season, forecast spread quantiles, extremes, etc.)
- what plots/tables you’d add to `report.md`
- success criteria and when to stop

3.6 Expected Outcome + Failure Analysis
State:
- expected improvement range
- likely failure modes
- what diagnostics would confirm each failure mode
- the most direct fix if it fails

----------------------------------------------------------------------------------------------------
NON‑NEGOTIABLE COVERAGE REQUIREMENTS ACROSS THE 60

Across the 60 experiments, you MUST include at least:

(A) 12 feature engineering experiments
(B) 10 loss/objective experiments
(C) 10 ensembling/stacking experiments
(D) 8 calibration/post‑processing experiments
(E) 10 conditional/regime/gating experiments (but only with balanced meaningful regimes)
(F) 10 data & evaluation experiments (leakage‑safe split strategy, purging, robust evaluation, outlier handling, missingness handling, label cleaning)

You may overlap categories, but you must still output 60 distinct experiments.

----------------------------------------------------------------------------------------------------
QUALITY BAR (read carefully; this is where most AI answers fail)

I do NOT want generic advice like “try more features” or “tune hyperparameters.” I want implementation‑ready, testable experiments.

For every single experiment you propose, you must make it concrete enough that I can:
1) add an experiment entry in my registry/config,
2) run it with my existing sweep runner command pattern,
3) get a report.md and metrics.json that lets me compare it fairly vs BASE,
4) understand exactly why it won or lost.

Therefore, each experiment spec MUST also include these micro‑requirements inside your subsections (do not create extra subsections; embed them inside the required ones):

- In “What Changes”: explicitly list the minimal baseline you are comparing to (e.g., “XGB BASE on same feature set”) and whether you are holding the feature set fixed.
- In “Implementation Plan”: include both (a) the smallest MVP implementation and (b) optional “stretch” improvements if MVP works.
- In “Leakage Guardrails”: name the exact potential leakage channels for that experiment and how you block them. Weather ML is full of subtle leakage (future climatology, future target availability, using test data for imputation/scaling, etc.).
- In “Evaluation Plan”: always include a “tail‑risk check”: compare maxAE and RMSE to baseline, not just MAE.
- In “Failure Analysis”: give a specific debug sequence (what to inspect in predictions_test.parquet, what stratifications to compute, what plots to generate).

If you cannot write a crisp implementation plan for an experiment, do not include it.

----------------------------------------------------------------------------------------------------
DATA CONTRACT YOU MUST RESPECT

Assume each row has at least these columns that must be preserved end‑to‑end:
- station_id
- target_date_local (DATE)
- asof_utc (timestamp)
- target: actual_tmax_f

In the KMIA gribstream run discussed above, a representative baseline feature set includes:
- gefsatmosmean_tmax_f
- rap_tmax_f
- hrrr_tmax_f
- nbm_tmax_f
- gfs_n_x_max
- nam_n_x_max
- gefsatmos_tmp_spread_f
- month, day_of_year, sin_doy, cos_doy, is_weekend

You may propose adding features, but you must explain:
- where the feature comes from,
- how it’s computed without leakage at asof_utc,
- how it’s validated (missingness, ranges),
- how it changes the report.

----------------------------------------------------------------------------------------------------
DIAGNOSTICS YOU SHOULD LEVERAGE (build these into your roadmap)

Before proposing “big model changes,” you should exploit cheap diagnostics that often reveal easy MAE wins:

1) Residual structure vs each base forecast
- Plot residual vs (nbm_tmax_f, hrrr_tmax_f, rap_tmax_f, gefs mean).
- Compute MAE of simple bias‑corrected versions of each forecast source.

2) Residual vs forecast disagreement / spread
- When models disagree (large spread), errors usually increase.
- This is a natural place for conditional logic (but it must be balanced and stable).

3) Seasonal stratification
- MAE by month, and also by sine/cosine bins.
- Some sources are seasonally biased; seasonal bias correction is often a big win.

4) Extremes stratification
- MAE for top decile of actual_tmax_f and bottom decile.
- Many models do worse in extremes; targeted corrections can help overall MAE if extremes are frequent enough.

5) Missingness stratification (if any)
- If any forecast inputs are imputed/missing, compute MAE for those rows separately.
- Add missingness indicators; sometimes that’s an immediate win.

Treat these as “experiments” too if needed: add them as report additions or diagnostic runs that don’t change the model but reveal where to focus.

----------------------------------------------------------------------------------------------------
MISSINGNESS / IMPUTATION EXPERIMENT GUIDANCE (you must include multiple ideas here)

When proposing missingness‑related experiments, be explicit about which strategy you use:

- Strategy 1: Model‑native NaN handling (if supported) + missingness indicators.
- Strategy 2: Train‑only imputation (mean/median per feature) + missingness indicators.
- Strategy 3: Time‑aware imputation (e.g., per‑month medians computed on TRAIN only, or rolling medians that do not use future dates).
- Strategy 4: Source‑aware imputation (if a whole model source is missing, degrade gracefully using other sources).

Every missingness experiment must explicitly include a leakage note: “imputation statistics are computed on TRAIN only,” and for any time‑aware method: “computed using only prior dates for each row.”

----------------------------------------------------------------------------------------------------
IDEA POOLS (use these as inspiration; you still must output 60 well‑specified experiments)

The sections below are NOT the 60 experiments. They are a structured idea pool to help you generate 60 truly distinct, high‑quality experiments. You must still do original synthesis and pick the best 60 proposals, with deep implementation plans.

Also: avoid micro‑variants. Changing `max_depth` from 4→5 is not a distinct experiment. A distinct experiment changes a meaningful modeling assumption: feature set, loss, architecture, routing logic, calibration method, training algorithm, or evaluation protocol.

----------------------------------------------------------------------------------------------------
FEATURE ENGINEERING IDEA POOL (you must turn at least 12 of these into experiments)

1) Consensus / ensemble structure features
- Compute multiple consensus predictors, not just raw model values:
  - simple mean, median, trimmed mean, winsorized mean
  - “best‑2 average” where best is determined on TRAIN only by historical MAE per model and season
  - spread features: max‑min, std, IQR, MAD across model forecasts
  - rank features: each model’s rank among the ensemble forecasts for the day
  - disagreement direction: e.g., (nbm − gefs_mean), (hrrr − rap), etc.
- Hypothesis: many residual errors occur when models disagree; capturing disagreement provides conditional correction signal.

2) Seasonality interactions (beyond sin/cos)
- Interactions like forecast_value × sin_doy, forecast_value × cos_doy, and piecewise seasonal bins (e.g., month buckets) to let the model learn season‑dependent bias of each forecast source.
- Add “season regime” indicators: DJF/MAM/JJA/SON, or heating/cooling degree‑day proxies, computed purely from date.
- Hypothesis: each NWP source has seasonally varying bias; letting the model learn season‑specific corrections reduces MAE.

3) Nonlinear transforms of forecast sources
- Include squared terms, absolute deviations from consensus, and saturation functions (e.g., tanh‑scaled versions) of each forecast source.
- Add “clamped” variants: min(max(x, lo), hi) with lo/hi defined from TRAIN percentiles.
- Hypothesis: the mapping from forecast to truth may be nonlinear, especially at extremes; explicit transforms can help tree models generalize in small data.

4) Residual‑predictive features (two‑stage or self‑supervised)
- Create a baseline predictor `y0` (e.g., weighted linear blend of raw forecasts trained on TRAIN).
- Add features that predict baseline error: (x − y0), (x − consensus), spread, seasonal interactions.
- Hypothesis: predicting residuals is easier than predicting the absolute target and can reduce MAE by focusing model capacity on the correction term.

5) Climatology / analog / persistence features (leakage‑safe)
- Station climatology features computed from TRAIN only:
  - day‑of‑year climatological mean/median tmax (smoothed) from historical observations
  - rolling climatology by month/day‑of‑year
- Persistence features if available without leakage (must be as‑of):
  - yesterday’s observed tmax, recent rolling mean, EMA features
- Hypothesis: on some days, NWP forecasts are systematically biased; climatology/persistence anchors can reduce MAE.

6) Event / regime features derived from model physics proxies
- Use forecast spread as uncertainty regime.
- Use forecast dewpoint/humidity proxies if present in MOS vars for that pipeline (but keep leakage‑safe).
- Hypothesis: the error mapping differs in humid vs dry regimes, frontal passages, etc.; proxies capture this.

7) Missingness indicators as first‑class features
- For each forecast source feature, add `is_missing_feature_X` flags and `missing_count` summary.
- Hypothesis: missingness correlates with specific time periods or regimes; indicators prevent imputation values from being misinterpreted as real signals.

----------------------------------------------------------------------------------------------------
LOSS / OBJECTIVE / TRAINING IDEA POOL (turn at least 10 into experiments)

1) Direct L1 optimization
- Use XGBoost `reg:absoluteerror` where available to align training loss with MAE.
- Compare vs default `reg:squarederror` but with evaluation on MAE.

2) Huber / pseudo‑Huber
- Use robust losses to reduce the influence of outliers while still learning smooth corrections.
- Hypothesis: if a few catastrophic cases dominate gradients, robust loss can improve MAE.

3) Quantile regression for median + mean correction
- Train median model (q=0.5) and optionally learn mean‑median adjustment as a second stage.
- Hypothesis: median fits MAE well; then calibrate back to mean‑optimal if needed.

4) Asymmetric loss
- Penalize warm errors vs cool errors differently if the application values one side more (only if you actually want that).
- Or use asymmetric loss only as an intermediate to correct a known sign bias then return to MAE evaluation.

5) Tail‑aware training with MAE‑preserving evaluation
- Weight rows by uncertainty/spread or by difficulty proxies, but still evaluate plain MAE on test.
- Hypothesis: improving hard days without degrading easy days can reduce overall MAE, but must be done carefully to avoid overfitting.

6) Regularization / monotonic constraints (if feasible)
- Constrain the blender so that increasing a forecast generally increases predicted tmax (monotonicity), reducing overfitting artifacts.
- Hypothesis: physical monotonicity reduces variance and improves MAE in small data.

----------------------------------------------------------------------------------------------------
ENSEMBLING / STACKING IDEA POOL (turn at least 10 into experiments)

These are often the highest‑ROI paths in small tabular forecasting problems, especially when the “raw forecasts” already carry most of the signal.

1) Constrained linear blender (strong baseline upgrade candidate)
- Fit a linear model that predicts `actual_tmax_f` from the raw forecast features with constraints:
  - non‑negative weights
  - weights sum to 1 (optional)
  - optional intercept for bias
- Fit on TRAIN only; choose constraints/regularization via validation.
- Hypothesis: a physically sensible convex blend can beat a single tree model in MAE due to lower variance.

2) Regularized linear stacking (ridge / elastic net)
- Stack base forecasts + key engineered features with ridge/elastic net.
- Hypothesis: with small N, regularized linear models can generalize better than deeper trees.

3) Two‑level stacking with OOF predictions
- Train several diverse base learners on TRAIN (e.g., XGB, LGBM, CatBoost, linear, kNN‑like analog model).
- Generate OOF predictions on TRAIN via time‑blocked folds.
- Train a meta‑learner on OOF predictions (simple ridge or constrained regression).
- Hypothesis: diversity + leakage‑safe stacking often yields robust MAE gains.

4) Season‑conditioned blending
- Learn separate blend weights per season bucket or month.
- Use strong regularization to avoid overfitting (e.g., hierarchical shrinkage toward global weights).
- Hypothesis: different models are best in different seasons; season‑aware blends reduce MAE.

5) Spread‑conditioned blending
- Define uncertainty buckets (e.g., quartiles of ensemble spread).
- Learn separate blend weights per bucket with shrinkage.
- Hypothesis: in high‑uncertainty situations, some sources are more reliable; bucketed blending improves MAE.

6) Residual stacking (“predict correction”)
- Stage 1: build a simple consensus predictor y0 (linear blend).
- Stage 2: train a model to predict residual `y − y0` using spread + seasonal interactions.
- Final: y_hat = y0 + residual_hat.
- Hypothesis: residual is smaller amplitude and easier to model; this reduces MAE without increasing tail risk.

7) Model zoo + selection by validation with guardrails
- Train many variants but select the final model by validation MAE subject to “tail guardrails” (maxAE/RMSE not worse than baseline by more than some threshold).
- Hypothesis: ensures we don’t win MAE by creating rare catastrophes.

----------------------------------------------------------------------------------------------------
CALIBRATION / POST‑PROCESSING IDEA POOL (turn at least 8 into experiments)

1) Seasonal bias correction (per month / per sine‑cos bin)
- Fit bias offsets on TRAIN only (or TRAIN+VAL if you treat VAL as calibration set), then apply to val/test.
- Compare additive vs multiplicative corrections (additive likely for temperature).

2) Piecewise linear correction by predicted temperature
- Fit a piecewise linear map `y_hat_corrected = a_k*y_hat + b_k` within bins of predicted y_hat (bins defined from TRAIN only).
- Hypothesis: corrects systematic under/overprediction at extremes.

3) Quantile mapping / distribution alignment (careful)
- Learn a monotone mapping from predicted to observed distribution on TRAIN only.
- Apply mapping to val/test predictions.
- Hypothesis: if model is systematically too narrow or skewed, mapping can reduce MAE.
- Guardrail: never learn mapping on test; ensure time ordering if mapping uses rolling windows.

4) Isotonic regression on residuals vs baseline prediction
- Fit isotonic function on TRAIN residuals as a function of y_hat or of a key forecast feature.
- Hypothesis: monotone correction can remove systematic bias without adding variance.

5) “Uncertainty‑aware shrinkage” toward climatology/consensus
- When spread is high, shrink the prediction toward climatology or the ensemble median.
- When spread is low, trust the primary model more.
- Hypothesis: reduces MAE by reducing overconfident errors on high‑uncertainty days.

----------------------------------------------------------------------------------------------------
CONDITIONAL / REGIME MODEL IDEA POOL (turn at least 10 into experiments, but fix EX201’s issues)

If you propose regimes, you MUST ensure:
- every regime exists in train/val/test with reasonable counts,
- the gate beats trivial baselines,
- the conditional model cannot create catastrophic tail errors (include guardrails).

1) 2‑class regime: “bust vs non‑bust”
- Instead of cool/normal/warm, define bust as |residual| > tau using a leakage‑safe baseline y0 (tau chosen on TRAIN).
- Train two experts: bust expert and normal expert.
- Hypothesis: easier classification problem, balanced enough to learn, and specialization matches “difficult days.”

2) Season regime experts (4 seasons)
- Train one model per season bucket with shrinkage or shared parameters.
- Gate is trivial (known from date) and cannot overfit.
- Hypothesis: removes the gate learning problem entirely; focuses on stable seasonal structure.

3) Spread regime experts (low/med/high uncertainty)
- Gate is based on spread quantiles computed on TRAIN only (then applied to val/test).
- Train experts per uncertainty bucket.
- Hypothesis: error structure differs by forecast disagreement; stable regimes are present in all splits.

4) Mixture without learned gate (soft weights from deterministic rule)
- Use a deterministic soft weighting function of spread/season (e.g., logistic on spread).
- Hypothesis: keeps conditional behavior but avoids weak learned gate.

5) “Expert = baseline correction” approach
- Expert models predict correction terms, not the full target, and are regularized toward zero.
- Hypothesis: reduces risk of catastrophic misrouting; experts can only nudge baseline.

----------------------------------------------------------------------------------------------------
DATA & EVALUATION IDEA POOL (turn at least 10 into experiments)

1) Time‑blocked cross‑validation on TRAIN for hyperparameter selection
- Use expanding windows or blocked folds within TRAIN only.
- Select hyperparameters by average validation‑like fold MAE; then retrain on full TRAIN.

2) Purged splits around boundary dates
- If there is any possibility of near‑duplicate information across boundaries (e.g., rolling features), purge a small gap around split boundaries to avoid bleed.

3) Bootstrap confidence intervals for MAE deltas (reporting only)
- Compute bootstrap distribution of MAE delta on test for reporting; do not tune on it, just quantify uncertainty.

4) Outlier policy experiment (label cleaning vs robust training)
- Define a reproducible outlier identification rule (on TRAIN only) and test whether clipping/removal improves generalization.
- Must be careful: removing real extremes can harm performance; evaluate tail metrics.

----------------------------------------------------------------------------------------------------
EXAMPLE OF THE DETAIL LEVEL I EXPECT (THIS EXAMPLE DOES NOT COUNT AS ONE OF THE 60)

Below is a miniature example of what a single experiment spec might look like. Use it as a style template only; do not treat it as an actual “1 of 60” unless you independently choose it and then include it properly in the 60 list.

Example Experiment (template only):

Hypothesis:
Seasonal bias varies by month and by forecast source. A constrained convex blend with a month‑specific intercept will reduce systematic bias and improve MAE without increasing tail risk.

What Changes:
- Replace the mean model with a constrained linear blender of raw forecast features.
- Add month‑specific intercept (12 offsets) with strong regularization/shrinkage toward 0.

Implementation Plan:
- Config: add mean_model.type = constrained_blend_month_bias and explicit regularization strength.
- Code: implement a new mean model class that fits weights on TRAIN only, writes weights to an artifact JSON, and emits per‑row blend weights and components in predictions_test.parquet for audit.
- Report: add a section showing weights, month biases, and MAE by month.

Leakage Guardrails:
- Fit weights and month bias offsets using TRAIN only.
- Month is derived from target_date_local (known), so no leakage.
- If you use any normalization, compute it on TRAIN only and apply to val/test.

Evaluation Plan:
- Compare MAE/RMSE/bias/corr/medianAE/maxAE on train/val/test vs XGB BASE.
- Stratify MAE by month and by spread quantile.
- Success if test MAE improves by at least 0.02 with no increase in maxAE > 0.5.

Failure Analysis:
- If MAE worsens: likely over‑regularization or under‑regularization; inspect learned weights and month offsets; check if weights collapse to a single model; adjust shrinkage.
- If tails worsen: inspect cases with largest residuals; check whether blend over‑trusted an unreliable source in high spread; add spread‑aware shrinkage.

----------------------------------------------------------------------------------------------------
FINAL DELIVERABLE CHECKLIST (you must satisfy this while writing)

As you generate the 60 experiments, continuously sanity‑check yourself against this checklist:

- Are the 60 experiments genuinely distinct (different assumption/knob), not micro‑tweaks?
- Did you include a practical “fast wins” set that can be implemented in hours (not weeks) and is likely to beat MAE quickly?
- Did you include at least a handful of “low variance” approaches (constrained blends, monotone corrections) that are less likely to blow up tails?
- For every experiment, did you state exactly what files/artifacts you expect and how to verify correctness?
- For every experiment, did you name the most likely leakage channel and how to block it?
- Did you explicitly address EX201’s failure mode (imbalanced regimes + weak gate) when proposing any conditional model?
- Did you incorporate missingness/imputation and propose at least a few experiments that treat missingness as signal (indicators, time‑aware imputation, source‑aware fallbacks)?
- Did you include a clear stop/continue rule for the research loop (e.g., after N experiments with no gain, pivot to new feature sources or new target transformations)?

If you cannot check all of these boxes for a proposed experiment, remove it and replace it with a better one.

----------------------------------------------------------------------------------------------------
OUTPUT LENGTH HANDLING

If you hit a response limit:
- Stop only at an experiment boundary (never mid‑experiment)
- Print “CONTINUE?” and wait for my reply “continue”

----------------------------------------------------------------------------------------------------
FINAL PUSH

Put an obsessive, godlike level of effort into this. Be creative but not random. Use evidence and propose serious experiments a top team would run. I need a large MAE improvement, not a marginal tweak.
```
