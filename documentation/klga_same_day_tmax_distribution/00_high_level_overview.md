# 00 - High Level Overview

This document explains the full system at a decision-maker level before going into implementation details.

## 1) Problem in one sentence

Given what KLGA has reported so far today, estimate the full probability distribution of today's final maximum temperature in whole-degree Fahrenheit.

## 2) Why this exists

Prediction market outcomes are buckets. You do not want a single point forecast. You want a calibrated probability distribution so you can compare model probabilities vs market-implied probabilities.

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

## 4) Core model design

The system uses a two-part decomposition:

1. Peak model (binary):
   - Predicts `P(delta=0)`.
   - Translation: probability that today's high is already done.

2. Delta model (multiclass):
   - Predicts `P(delta=k | delta>0)` for `k=1..60`.
   - Translation: if not done yet, how much more can temperature rise.

These are then combined into one final PMF.

## 5) Why two models instead of one

Because there are two different regimes:

- Regime A: day has already peaked (`delta=0`).
- Regime B: day can still rise (`delta>0`).

Trying to force both into one single head usually hurts calibration and interpretability.

## 6) What "delta" means

- `delta = Tmax_final - Tmax_sofar`.
- Example:
  - current max so far is `74F`
  - if final max ends at `76F`, then `delta=2`.

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

## 8) Time design

- Station timezone: `America/New_York`
- Cutoffs: every 30 minutes from `04:00` to `18:00` local
- Total: 29 cutoffs/day

This matters because performance depends strongly on time of day.

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

## 11) Calibration

Peak model:

- isotonic calibration

Delta model:

- multiclass temperature scaling

Reason:

- improve probability quality, not just ranking.

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

## 14) What gets exported

When run completes successfully, you get:

- peak model file
- peak isotonic calibrator
- delta model file
- delta temperature scaler
- feature list and imputer values
- predictions and evaluation reports
- run logs and config snapshot

## 15) Practical takeaway

If you are deciding whether this system is useful:

- Peak side is strong and materially improves combined NLL.
- Delta side is harder and is the current limiting factor.
- System quality improves later in day because uncertainty naturally collapses.

If you are deciding where to improve next:

- prioritize delta calibration and class-imbalance handling
- improve late-stage robustness and diagnostics
- keep leakage and as-of guardrails unchanged
