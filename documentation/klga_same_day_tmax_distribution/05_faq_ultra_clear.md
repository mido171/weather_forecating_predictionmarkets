# 05 - FAQ (Ultra Clear, No Ambiguity)

This FAQ is intentionally direct. It exists to eliminate repeated confusion around peak, delta, NLL, calibration, runtime, and live-trading interpretation.

## A) Output semantics

### Q1) Does the model output one delta value or many?

It outputs a full probability distribution, not one value.

Specifically:

- one probability for `delta=0` from peak model,
- sixty conditional probabilities for `delta=1..60` from delta model (where class 60 is tail bin for `delta>=60`),
- then these are combined into one full delta PMF.

So yes, it outputs many values with probabilities.

### Q2) Does it output probabilities for each value?

Yes.

Every class/bin has a probability.

Example structure (illustrative):

- `P(delta=0)=0.62`
- `P(delta=1)=0.18`
- `P(delta=2)=0.11`
- `P(delta=3)=0.05`
- ...
- `P(delta>=60)=0.00x`

### Q3) How do we get final temperature probabilities from delta probabilities?

Use:

- `Tmax_final = round(tmax_sofar) + delta`

So each delta bin maps to an integer Fahrenheit final temperature bin.

### Q4) Why are there many Fahrenheit probabilities instead of one predicted Tmax?

Because market outcomes are bucketed and uncertainty matters.

A single-point forecast cannot properly price ranges like:

- `74-75F`
- `71F or below`
- `82F or higher`

## B) Peak model vs delta model

### Q5) What exactly does the peak model predict?

Peak model predicts:

- `P(delta=0)`

Meaning:

- probability that today's max is already reached by cutoff.

### Q6) What exactly does the delta model predict?

Delta model predicts:

- `P(delta=k | delta>0)` for `k=1..60`.

Meaning:

- if day is not already peaked, how much additional warming remains.

### Q7) Is delta model trained on all rows?

No.

Delta model training uses only rows where truth says `delta>=1` (equivalently `peak=0`).

Why:

- it is a conditional model by design.

### Q8) Is that data filtering leakage?

No.

Using truth labels to define training targets/slices is normal supervised learning.

Leakage would be putting future truth into input features at inference time.

### Q9) Is the peak prediction value used as an input feature to delta model?

No.

Delta feature matrix does not include peak model output.

Peak and delta are separate heads merged later in posterior composition.

### Q10) Why use two models instead of one multiclass over all deltas including zero?

Two-head decomposition isolates regimes:

- regime 1: `delta=0` (already peaked)
- regime 2: `delta>0` (still rising)

This usually gives better interpretability and better calibration control.

## C) Metrics: what each number actually means

### Q11) Is delta multi-logloss the same as final system NLL?

No.

- delta multi-logloss: evaluates delta head only on `delta>0` rows.
- combined NLL: evaluates full final PMF on all rows.

They are different row sets and different scoring problems.

### Q12) Did the reported delta score (`~2.4076`) include peak model effect?

No.

`2.4076` is delta-only metric.

Peak model effect appears only in combined PMF metrics.

### Q13) Then how do we measure tangible value of peak model?

Compare combined NLL with and without a strong peak component in PMF construction.

Peak improves mass placement on `delta=0` rows, reducing overall NLL.

### Q14) If peak model logloss is good, can it still be wrong on some days?

Yes.

Logloss is an average probabilistic score, not a guarantee per row.

You can still see high-confidence misses; the question is whether probabilities are good on average over many rows.

### Q15) Is higher NLL equal to “wider distribution”?

Not necessarily.

NLL depends on probability assigned to the true outcome.

- narrow but wrong distribution can have very high NLL,
- wider but truth-covering distribution can have lower NLL.

### Q16) Why does model usually score better later in the day?

Because uncertainty collapses as more same-day observations are known.

Example from completed run (`20260226T081223Z`, test):

- 04:00 NY (`cutoff=240`) => NLL `2.8052`
- 18:00 NY (`cutoff=1080`) => NLL `1.5636`

## D) Calibration and trust

### Q17) Is NLL enough to trust probabilities for trading?

NLL is necessary but not sufficient.

Also check calibration quality (reliability of stated percentages).

### Q18) What calibration methods are used here?

- peak: isotonic regression
- delta: multiclass temperature scaling

### Q19) What does “well calibrated” mean in plain language?

If model says 30% often, roughly 30% should happen in reality over many similar cases.

### Q20) Where do calibration outputs live?

In run artifacts:

- peak calibration reflected in calibrated peak metrics,
- delta calibration reflected by temp-scaled multi-logloss,
- bucket calibration CSVs under `reports/` for completed runs.

## E) Leakage and safety

### Q21) Biggest leakage rule?

At cutoff `t_c`, features must only use data known at or before `t_c`.

### Q22) Can daily max for current date `D` be used in features?

No.

Date `D` daily max is label only.

### Q23) Why are `max_temp`, `min_temp`, `precip_total` banned from observation table?

Because they can behave like summary fields and may include post-cutoff information.

### Q24) Why exclude `KNYC:9:US`?

To avoid duplicate-source ambiguity with KLGA mapping.

Canonical target is `KLGA:9:US`.

## F) Runtime and monitoring

### Q25) Why can training look stuck for a long time?

Most of runtime is in delta multiclass LightGBM training.

From completed no-analog run:

- peak train: about 3 minutes
- delta train: about 56 minutes

Long delta stage is expected.

### Q26) How do I confirm process is alive?

Check run log heartbeat/progress lines and CPU usage.

A truly dead run stops writing log progress and process disappears.

### Q27) Which mode is fastest for reliable exports?

Use no-analog mode:

```powershell
python ml/run_klga_daily_tmax_dist.py --output-root artifacts/same_day_res_poly --skip-analog-blend
```

## G) Trading interpretation examples

### Q28) If `P(delta=0)=0.82`, should I assume “done” with certainty?

No.

Still 18% mass remains on `delta>=1` outcomes.

That residual probability can still matter if market pricing is tight.

### Q29) Practical interpretation of `+2,+3,+4` talk?

It means possible additional warming relative to current so-far max.

If `tmax_sofar=74`:

- `delta=2` maps to final `76F`
- `delta=3` maps to final `77F`
- `delta=4` maps to final `78F`

### Q30) What is the current reference exported run?

Use:

- `artifacts/same_day_res_poly/20260226T081223Z/`

Reason:

- completed,
- full model exports present,
- full metrics and reports present.

## H) Common misunderstanding checklist

If confused, check these in order:

1. Are you reading peak metric, delta metric, or combined metric?
2. Is run mode analog-enabled or no-analog?
3. Is run complete (`PIPELINE_DONE`) or partial?
4. Are you comparing same split and same cutoff?
5. Are you interpreting probabilities as averages over many rows, not guarantees per row?
