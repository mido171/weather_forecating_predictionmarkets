# 00 - High Level Overview

This document explains the full system at a decision-maker level before going into implementation details.

If you only read one file, read this one and then `02_metrics_and_interpretation_for_beginners.md`.

## 1) Problem in one sentence

Given what KLGA has reported so far today, estimate the full probability distribution of today's final maximum temperature in whole-degree Fahrenheit.

This is explicitly designed for prediction-market pricing, where the outcomes are temperature buckets and you need a calibrated probability distribution, not a single number.

## 2) Why this exists

Prediction market outcomes are buckets. You do not want a single point forecast. You want a calibrated probability distribution so you can compare model probabilities vs market-implied probabilities.

In practical terms:

- you want to know "what is the probability the final max ends up at 72F vs 73F vs 74F"
- then you sum those into the market labels ("74-75F", "71F or below", "82F or higher")

## 3) Input and output in plain terms

Input:

- Historical and as-of weather observations for KLGA and selected nearby airports.
- Historical KLGA daily max truth values.
- A decision cutoff time (for example 13:00 NYC local, which is 19:00 Stockholm in winter).

Output:

- A PMF over final integer temperatures:
  - `P(T=70), P(T=71), P(T=72), ...`
- Bucket probabilities for market labels:
  - `P(74-75F)`
  - `P(71F or below)`
  - `P(82F or higher)`

### 3.1 The single most important alignment constraint

Polymarket resolves using:

- Wunderground station semantics (KLGA)
- whole-degree Fahrenheit

So the system output is intentionally an integer-Fahrenheit PMF, not a continuous distribution.

### 3.2 How this maps to a live trading decision (conceptual)

At a given cutoff time:

1. Compute your model bucket probabilities from the PMF.
2. Convert market prices into implied probabilities (after fees/spread as appropriate).
3. Only trade when your probability differs enough from the market to overcome:
   - fees
   - bid/ask spread
   - model error / calibration error
4. Prefer late-cutoff times if you want higher accuracy, but note:
   - late-cutoff accuracy is higher
   - late-cutoff market efficiency is also usually higher (harder to find mispricings)

This is why time-of-day cutoff reports matter: they tell you where the model is strongest, and you can choose a trading window accordingly.

## 4) Core model design

The system uses a two-part decomposition:

1. Peak model (binary):
   - Predicts `P(delta=0)`.
   - Translation: probability that today's high is already done.

2. Delta model (multiclass):
   - Predicts `P(delta=k | delta>0)` for `k=1..60`.
   - Translation: if not done yet, how much more can temperature rise.

These are then combined into one final PMF.

### 4.1 What "peak" actually means

At a cutoff time `t_c` on local day `D`:

- compute `tmax_sofar(D, t_c)` from same-day observations up to the cutoff
- obtain `tmax_truth(D)` from the daily max truth table (label only)
- define `delta = tmax_truth - round(tmax_sofar)`

Then:

- `peak = 1` means `delta = 0` (the day's final max is already achieved by cutoff)
- `peak = 0` means `delta >= 1` (there is still room to rise)

This is a real "regime split":

- many days do not keep warming after a given hour (front passage, cloud cap, early max)
- other days keep warming into the afternoon

## 5) Why two models instead of one

Because there are two different regimes:

- Regime A: day has already peaked (`delta=0`).
- Regime B: day can still rise (`delta>0`).

Trying to force both into one single head usually hurts calibration and interpretability.

Also: this decomposition maps directly to trading decisions.

- if `P(delta=0)` is high, the distribution should collapse around the current max-so-far
- if `P(delta=0)` is low, the tail outcomes (higher buckets) still matter

## 6) What "delta" means

- `delta = Tmax_final - Tmax_sofar`.
- Example:
  - current max so far is `74F`
  - if final max ends at `76F`, then `delta=2`.

Delta is intentionally discrete and integer-valued, because the market resolves in whole-degree Fahrenheit.

## 7) Leakage safety model

The system is intentionally leakage-paranoid.

At cutoff `t_c`, features may only use data known by `t_c`.

Allowed:

- observations with `valid_time_utc <= cutoff_utc`
- historical daily max values from dates `< D`

Forbidden:

- daily max for date `D` as a feature
- observations after cutoff
- dangerous summary-like observation columns (`max_temp`, `min_temp`, `precip_total`)

If guardrails fail, run should fail.

### 7.1 Why leakage paranoia matters for trading

If you accidentally include post-cutoff information, offline metrics will look amazing, but live trading will fail.

So the system is designed so that the safe behavior is:

- crash early if a leakage guard triggers
- force you to fix the root cause rather than silently producing invalid results

## 8) Time design

- Station timezone: `America/New_York`
- Cutoffs: every 30 minutes from `04:00` to `18:00` local
- Total: 29 cutoffs/day

This matters because performance depends strongly on time of day.

Later cutoffs are naturally easier:

- more of the day's trajectory is already observed
- less uncertainty remains

## 9) Data sources

Two canonical tables:

1. `wunderground_ml.wunderground_station_observation_30m`
2. `wunderground_ml.wunderground_station_daily_max_temperature`

Target station:

- `KLGA:9:US`

Neighbor stations:

- `KJFK:9:US`
- `KEWR:9:US`
- `KTEB:9:US`
- `KHPN:9:US`
- `KISP:9:US`
- `KBDR:9:US`
- `KMMU:9:US`

Hard exclusion:

- `KNYC:9:US` is excluded to avoid duplicate-source ambiguity.

## 10) What features matter at a high level

Feature families:

1. Time and season identity
2. KLGA current state
3. KLGA so-far max trajectory and momentum
4. Neighbor gradients and coastal/inland contrasts
5. Historical priors (yesterday, rolling windows)
6. Train-only climatology priors by `(day-of-year, cutoff)`

### 10.1 Why neighbor stations are valuable for NYC

NYC is strongly affected by:

- coastal influence / sea breeze
- inland heating gradients
- frontal passage directionality (west to east)

Neighbor features let the model infer these regimes without external numerical weather model guidance.

## 11) Calibration

Peak model:

- isotonic calibration

Delta model:

- multiclass temperature scaling

Reason:

- improve probability quality, not just ranking.

Calibration is not optional for trading:

- you need probabilities that mean what they say
- "30%" should happen about 30% of the time across comparable events

## 12) How to interpret main metrics

Peak model metrics:

- binary logloss
- Brier score

Delta model metric:

- multiclass logloss on `delta>0` rows only

Combined/live-like metric:

- NLL of full PMF after combining peak+delta

Important:

- `delta multi_logloss` is not the same metric as final combined NLL.

See `02_metrics_and_interpretation_for_beginners.md` for a much more detailed explanation with examples.

## 13) Current run modes

Two operational modes are available:

1. Full mode:
   - includes analog kNN + blending
2. No-analog mode:
- `--skip-analog-blend`
- uses pure LGBM peak+delta stack

No-analog mode is useful for:

- faster reruns
- clean export of peak/delta artifacts
- isolating LGBM behavior from analog behavior

Additionally, a separate experimental "TabM from exports" training path exists for tabular neural network comparisons.

## 14) What gets exported

When run completes successfully, you get:

- peak model file
- peak isotonic calibrator
- delta model file
- delta temperature scaler
- feature list and imputer values
- predictions and evaluation reports
- run logs and config snapshot

### 14.1 What "standalone reuse" actually requires

To reuse the exported models later, you must also reuse:

- exact feature set and feature order (from `feature_list.json`)
- exact imputer fill values (from `imputer_values.json`)
- calibration artifacts (peak isotonic and delta temperature scaler)

If any of these differ, your probabilities are invalid.

## 15) Practical takeaway

If you are deciding whether this system is useful:

- Peak side is strong and materially improves combined NLL.
- Delta side is harder and is the current limiting factor.
- System quality improves later in day because uncertainty naturally collapses.

If you are deciding where to improve next:

- prioritize delta calibration and class-imbalance handling
- improve late-stage robustness and diagnostics
- keep leakage and as-of guardrails unchanged

## 16) End-to-end flow (visual mental model)

This is the actual computation pipeline, simplified:

```mermaid
flowchart TD
  A[Obs rows up to cutoff] --> B[Feature builder]
  C[Past daily max rows < D] --> B
  B --> D[Peak model: P(delta=0)]
  B --> E[Delta model: P(delta=k | delta>0)]
  D --> F[Compose full delta PMF]
  E --> F
  F --> G[Convert delta PMF -> Tmax integer PMF]
  G --> H[Sum PMF into bucket labels]
```

This matters because it makes the division of responsibilities explicit:

- peak head assigns mass to `delta=0`
- delta head shapes the positive-delta tail
- composition yields the final PMF used for trading decisions

## 17) Current reference results (so you have an anchor)

This section exists because people naturally ask: "Is this good?"

Important:

- these numbers are snapshots tied to specific run ids
- performance depends heavily on cutoff time

Reference LGBM export run (no analog):

- run id: `20260226T081223Z` (see `04_run_history_and_current_status_2026-02-26.md`)
- peak test logloss cal: `~0.2099`
- combined test NLL: `~2.3387`

Time-of-day behavior example (same reference run, test split):

- 04:00 NY cutoff (`cutoff_minutes=240`) => NLL `~2.8052` (harder, more uncertainty)
- 18:00 NY cutoff (`cutoff_minutes=1080`) => NLL `~1.5636` (easier, uncertainty collapses)

TabM experiment run (trained from exported CSV bundle):

- run id: `20260226T224250Z`
- combined test NLL: `~2.5109` (worse than the LGBM reference)

Interpretation:

- the system is materially more confident/accurate later in the day
- LightGBM remains the stronger baseline than TabM on this feature contract as of these runs
