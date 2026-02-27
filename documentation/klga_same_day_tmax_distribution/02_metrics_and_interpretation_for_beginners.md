# 02 - Metrics and Interpretation for Beginners

This file is intentionally plain-language and is focused on interpretation, not code.

If you are trading:

- you should care primarily about probability quality (calibration and NLL), not just point accuracy
- you should compare your probabilities vs market-implied probabilities, not just "did we guess right"

## 1) Start here: there are three different score families

You must keep these separate:

1. Peak model scores:
   - binary event: `peaked yet?`
2. Delta model scores:
   - multiclass event: `how many more degrees if not peaked?`
3. Combined system scores:
   - final PMF quality after peak+delta are merged

Most confusion comes from mixing these three into one.

### 1.1 The most common mistake

People often see a number like:

- `delta multi_logloss_temp = 2.40`

and assume it means:

- "the system is 2.40 NLL overall"

That is wrong.

Delta multi-logloss is computed on a filtered subset:

- rows where truth says `delta >= 1` (equivalently `peak=0`)

So it does not include the `delta=0` regime at all.

## 2) Peak model metrics

Peak predicts:

- `P(delta=0)` which means probability that max is already done.

Main scores:

- `logloss_cal`
- `brier_cal`

Lower is better for both.

Typical interpretation:

- low peak logloss means model probabilities for "already peaked" are useful and calibrated.

### 2.1 How to interpret peak logloss in live terms

Peak is a binary event. For one row (one cutoff):

- if truth is "peaked", the loss is `-log(p_peak)`
- if truth is "not peaked", the loss is `-log(1 - p_peak)`

So:

- a well-calibrated peak model reduces your chance of over-trusting a regime decision
- it is not a guarantee that every individual day will be correct

## 3) Delta model metrics

Delta predicts:

- `P(delta=k | delta>0)` over classes `1..60`

Main score:

- `multi_logloss_temp`

Important:

- this metric is computed on `delta>=1` rows only.
- peak model is not part of this metric.

So if delta score is `2.4`, that tells you only about the conditional delta head.

### 3.1 What delta logloss is measuring (plain language)

For each row (cutoff) where the day is not yet peaked:

- the delta model produces a probability for each remaining warming amount
- delta logloss punishes the model when it assigns low probability to the true delta

Delta logloss does not directly say:

- "the distribution is wide"
- "the model output has many buckets"

It says:

- "how much probability mass did you put on what actually happened"

## 4) Combined metrics (live-like)

Combined PMF uses both heads:

- `P(delta=0)` from peak model
- `P(delta>0)` shape from delta model

Main score:

- combined `nll`

This is the closest offline proxy for live usage quality.

### 4.1 Combined NLL is the number you should track for trading

If you want one offline metric that best matches "live trading usage":

- use combined NLL on the test split

Because combined NLL:

- scores the full final PMF you would use to price market buckets
- includes both the `delta=0` and `delta>0` regimes

## 5) Concrete metric mapping

If you see:

- `delta.multi_logloss_temp = 2.40`
- `combined.test.nll = 2.34`

Interpretation:

- delta head quality is moderate
- full system is slightly better because peak head helps assign correct mass to `delta=0` rows

These are different metrics, different row sets, different meanings.

## 6) NLL intuition

Per-row formula:

- `NLL = -log(p_true)`

Where `p_true` is model probability assigned to what actually happened.

Examples:

- `p_true = 0.80` -> `NLL = 0.223` (excellent)
- `p_true = 0.40` -> `NLL = 0.916` (good)
- `p_true = 0.20` -> `NLL = 1.609` (weak)
- `p_true = 0.03` -> `NLL = 3.507` (bad)

Key point:

- NLL is not distribution width.
- NLL is probability assigned to truth.

### 6.1 A useful mental conversion table

This helps you interpret NLL values quickly:

| NLL | implied probability on truth (`exp(-NLL)`) | meaning |
|---|---:|---|
| 0.69 | 0.50 | you assign about 50% to truth |
| 1.10 | 0.33 | you assign about 33% to truth |
| 1.61 | 0.20 | you assign about 20% to truth |
| 2.30 | 0.10 | you assign about 10% to truth |
| 3.00 | 0.05 | you assign about 5% to truth |

So if your combined NLL is ~2.34 on test, the average probability mass on truth is roughly:

- `exp(-2.34) ~= 0.096` (about 9.6%)

This is not "bad" by itself because:

- the outcome space is many integer temperatures (not a 2-class event)
- early in the day the distribution must be broader

But you still compare runs by NLL: lower is better.

## 7) Why later cutoffs usually score better

Later in day:

- more trajectory already observed
- less remaining uncertainty
- distribution naturally narrows

So it is normal that NLL improves from early morning to late afternoon.

Practical implication:

- if you only trade later (e.g., 20:00 Stockholm time), you should focus on the late-cutoff rows and their NLL/calibration, not the morning rows.

## 8) Calibration: what each tool means

Peak calibration:

- isotonic regression
- can improve both logloss and Brier

Delta calibration:

- temperature scaling
- often smaller but still useful improvements

If calibration improves score, probabilities are usually more trustworthy.

### 8.1 "How do I know 32% really means 32%?"

You check calibration outputs, not just a single NLL number.

For this system:

- peak calibration quality is reflected in peak logloss/Brier (raw vs calibrated)
- distribution calibration is assessed by:
  - combined NLL (global probability quality)
  - bucket calibration tables (empirical vs predicted probability for each integer temperature bucket)

The bucket calibration CSVs are the most direct "does 32% mean 32%" artifact in the current pipeline.

## 9) Practical score ranges (heuristic)

Peak binary logloss:

- `<0.22`: very strong
- `0.22-0.30`: good
- `0.30-0.45`: weak/moderate
- `>0.45`: poor

Delta multiclass logloss:

- `<2.0`: very strong
- `2.0-2.4`: decent to good
- `2.4-2.8`: moderate
- `>2.8`: weak

Combined NLL:

- judge by same cutoff and same split
- do not compare early cutoff directly against late cutoff

## 10) Top1 accuracy and why it can look low

Top1 means:

- whether the single most likely exact final integer matched truth.

Because this is a discrete distribution problem with many plausible temperatures, top1 can be modest even when probability quality is useful.

Trading note:

- you do not need top1 to be huge to have edge
- you need probabilities that differ from the market and are calibrated enough that expected value is positive

## 11) The single most important anti-confusion rule

Do not compare these directly as if identical:

- delta multiclass logloss
- combined final NLL

Instead:

- use delta logloss to judge delta head
- use peak logloss/Brier to judge peak head
- use combined NLL for live-like whole-system quality

## 12) Quick checklist before trusting a score

1. Are you reading val or test?
2. Is this peak, delta, or combined metric?
3. Is analog enabled or skipped?
4. Which cutoff time are you looking at?
5. Are you comparing the same run mode?

## 13) How to read the report CSVs (very practical)

### 13.1 cutoff_metrics_test.csv

This answers:

- "how good is the distribution at each cutoff time?"

Columns typically include:

- `cutoff_minutes` (NY local minutes since midnight)
- `n_rows`
- `nll`
- `top1_accuracy`

Read it like:

- earlier cutoffs will have higher NLL (more uncertainty)
- later cutoffs should have lower NLL and higher top1

### 13.2 bucket_calibration_test.csv

This answers:

- "when the model assigns p to a specific integer temperature, does that happen about p fraction of the time?"

It aggregates over rows and compares:

- `pred_mean` vs `empirical` for each integer temperature bucket

If `pred_mean` is systematically higher than empirical:

- model is overconfident for that temperature bucket

If systematically lower:

- model is underconfident

This table is a concrete way to build trust (or find issues) in your probability outputs.

## 14) FAQ (Ultra Clear, No Ambiguity)

This FAQ is intentionally direct. It exists to eliminate repeated confusion around peak, delta, NLL, calibration, runtime, and live-trading interpretation.

### A) Output semantics

#### Q1) Does the model output one delta value or many?

It outputs a full probability distribution, not one value.

Specifically:

- one probability for `delta=0` from peak model,
- sixty conditional probabilities for `delta=1..60` from delta model (where class 60 is tail bin for `delta>=60`),
- then these are combined into one full delta PMF.

So yes, it outputs many values with probabilities.

#### Q2) Does it output probabilities for each value?

Yes.

Every class/bin has a probability.

Example structure (illustrative):

- `P(delta=0)=0.62`
- `P(delta=1)=0.18`
- `P(delta=2)=0.11`
- `P(delta=3)=0.05`
- ...
- `P(delta>=60)=0.00x`

#### Q2.1) What is a PMF?

PMF means "probability mass function".

In plain language:

- it is a list/map of discrete outcomes and their probabilities
- probabilities sum to 1.0

Here:

- outcomes are integer Fahrenheit temperatures (or integer deltas)
- the model outputs a PMF over those integers

#### Q3) How do we get final temperature probabilities from delta probabilities?

Use:

- `Tmax_final = round(tmax_sofar) + delta`

So each delta bin maps to an integer Fahrenheit final temperature bin.

#### Q4) Why are there many Fahrenheit probabilities instead of one predicted Tmax?

Because market outcomes are bucketed and uncertainty matters.

A single-point forecast cannot properly price ranges like:

- `74-75F`
- `71F or below`
- `82F or higher`

### B) Peak model vs delta model

#### Q5) What exactly does the peak model predict?

Peak model predicts:

- `P(delta=0)`

Meaning:

- probability that today's max is already reached by cutoff.

#### Q6) What exactly does the delta model predict?

Delta model predicts:

- `P(delta=k | delta>0)` for `k=1..60`.

Meaning:

- if day is not already peaked, how much additional warming remains.

#### Q7) Is delta model trained on all rows?

No.

Delta model training uses only rows where truth says `delta>=1` (equivalently `peak=0`).

Why:

- it is a conditional model by design.

#### Q8) Is that data filtering leakage?

No.

Using truth labels to define training targets/slices is normal supervised learning.

Leakage would be putting future truth into input features at inference time.

#### Q9) Is the peak prediction value used as an input feature to delta model?

No.

Delta feature matrix does not include peak model output.

Peak and delta are separate heads merged later in posterior composition.

#### Q9.1) Then how does peak help if it isn't an input feature?

Peak helps during posterior composition:

- it assigns probability mass to `delta=0`
- it determines how much total probability mass is left for the `delta>0` tail

So peak improves the combined distribution even though delta training is independent.

#### Q10) Why use two models instead of one multiclass over all deltas including zero?

Two-head decomposition isolates regimes:

- regime 1: `delta=0` (already peaked)
- regime 2: `delta>0` (still rising)

This usually gives better interpretability and better calibration control.

### C) Metrics: what each number actually means

#### Q11) Is delta multi-logloss the same as final system NLL?

No.

- delta multi-logloss: evaluates delta head only on `delta>0` rows.
- combined NLL: evaluates full final PMF on all rows.

They are different row sets and different scoring problems.

#### Q12) Did the reported delta score (`~2.4076`) include peak model effect?

No.

`2.4076` is delta-only metric.

Peak model effect appears only in combined PMF metrics.

#### Q13) Then how do we measure tangible value of peak model?

Compare combined NLL with and without a strong peak component in PMF construction.

Peak improves mass placement on `delta=0` rows, reducing overall NLL.

#### Q14) If peak model logloss is good, can it still be wrong on some days?

Yes.

Logloss is an average probabilistic score, not a guarantee per row.

You can still see high-confidence misses; the question is whether probabilities are good on average over many rows.

#### Q14.1) How can I see many days with high p_peak but truth=not peaked?

Two reasons can both be true:

1. The model is generally good on average (low logloss), but still makes occasional confident mistakes.
2. Your inspected sample may be biased (e.g., only December at a specific cutoff, which can have more complex early-evening behavior).

The correct way to judge probability quality is:

- aggregated scores on the full test set
- plus calibration tables (bucket calibration CSVs)

#### Q15) Is higher NLL equal to “wider distribution”?

Not necessarily.

NLL depends on probability assigned to the true outcome.

- narrow but wrong distribution can have very high NLL,
- wider but truth-covering distribution can have lower NLL.

#### Q16) Why does model usually score better later in the day?

Because uncertainty collapses as more same-day observations are known.

Example from completed run (`20260226T081223Z`, test):

- 04:00 NY (`cutoff=240`) => NLL `2.8052`
- 18:00 NY (`cutoff=1080`) => NLL `1.5636`

### D) Calibration and trust

#### Q17) Is NLL enough to trust probabilities for trading?

NLL is necessary but not sufficient.

Also check calibration quality (reliability of stated percentages).

#### Q18) What calibration methods are used here?

- peak: isotonic regression
- delta: multiclass temperature scaling

#### Q19) What does “well calibrated” mean in plain language?

If model says 30% often, roughly 30% should happen in reality over many similar cases.

#### Q20) Where do calibration outputs live?

In run artifacts:

- peak calibration reflected in calibrated peak metrics,
- delta calibration reflected by temp-scaled multi-logloss,
- bucket calibration CSVs under `reports/` for completed runs.

### E) Leakage and safety

#### Q21) Biggest leakage rule?

At cutoff `t_c`, features must only use data known at or before `t_c`.

#### Q22) Can daily max for current date `D` be used in features?

No.

Date `D` daily max is label only.

#### Q23) Why are `max_temp`, `min_temp`, `precip_total` banned from observation table?

Because they can behave like summary fields and may include post-cutoff information.

#### Q24) Why exclude `KNYC:9:US`?

To avoid duplicate-source ambiguity with KLGA mapping.

Canonical target is `KLGA:9:US`.

### F) Runtime and monitoring

#### Q25) Why can training look stuck for a long time?

Most of runtime is in delta multiclass LightGBM training.

From completed no-analog run:

- peak train: about 3 minutes
- delta train: about 56 minutes

Long delta stage is expected.

#### Q26) How do I confirm process is alive?

Check run log heartbeat/progress lines and CPU usage.

A truly dead run stops writing log progress and process disappears.

#### Q27) Which mode is fastest for reliable exports?

Use no-analog mode:

```powershell
python ml/run_klga_daily_tmax_dist.py --output-root artifacts/same_day_res_poly --skip-analog-blend
```

### G) Trading interpretation examples

#### Q28) If `P(delta=0)=0.82`, should I assume “done” with certainty?

No.

Still 18% mass remains on `delta>=1` outcomes.

That residual probability can still matter if market pricing is tight.

#### Q29) Practical interpretation of `+2,+3,+4` talk?

It means possible additional warming relative to current so-far max.

If `tmax_sofar=74`:

- `delta=2` maps to final `76F`
- `delta=3` maps to final `77F`
- `delta=4` maps to final `78F`

#### Q30) What is the current reference exported run?

Use:

- `artifacts/same_day_res_poly/20260226T081223Z/`

Reason:

- completed,
- full model exports present,
- full metrics and reports present.

### I) Exporting and TabM (tabular neural nets)

#### Q31) Can I train on another machine without DB access?

Yes.

Use the exporter:

- `ml/run_klga_data_exporter.py`

It writes a portable bundle of CSVs that the feature builder and trainer can consume.

Details:

- `07_exporter_and_remote_training_tabm.md`

#### Q32) Are tabular neural networks always better than boosted trees?

No.

On many structured/tabular problems, especially with:

- missingness patterns
- noisy engineered features
- many threshold interactions

GBDTs (LightGBM/CatBoost/XGBoost) often win on:

- robustness
- time-to-good-solution
- calibration quality after simple post-processing

In this project, the first TabM run underperformed the LightGBM reference run on the key probability metrics.

### H) Common misunderstanding checklist

If confused, check these in order:

1. Are you reading peak metric, delta metric, or combined metric?
2. Is run mode analog-enabled or no-analog?
3. Is run complete (`PIPELINE_DONE`) or partial?
4. Are you comparing same split and same cutoff?
5. Are you interpreting probabilities as averages over many rows, not guarantees per row?
