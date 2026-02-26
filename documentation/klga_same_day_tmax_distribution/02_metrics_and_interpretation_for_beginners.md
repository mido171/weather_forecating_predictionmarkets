# 02 - Metrics and Interpretation for Beginners

This file is intentionally plain-language and is focused on interpretation, not code.

## 1) Start here: there are three different score families

You must keep these separate:

1. Peak model scores:
   - binary event: `peaked yet?`
2. Delta model scores:
   - multiclass event: `how many more degrees if not peaked?`
3. Combined system scores:
   - final PMF quality after peak+delta are merged

Most confusion comes from mixing these three into one.

## 2) Peak model metrics

Peak predicts:

- `P(delta=0)` which means probability that max is already done.

Main scores:

- `logloss_cal`
- `brier_cal`

Lower is better for both.

Typical interpretation:

- low peak logloss means model probabilities for "already peaked" are useful and calibrated.

## 3) Delta model metrics

Delta predicts:

- `P(delta=k | delta>0)` over classes `1..60`

Main score:

- `multi_logloss_temp`

Important:

- this metric is computed on `delta>=1` rows only.
- peak model is not part of this metric.

So if delta score is `2.4`, that tells you only about the conditional delta head.

## 4) Combined metrics (live-like)

Combined PMF uses both heads:

- `P(delta=0)` from peak model
- `P(delta>0)` shape from delta model

Main score:

- combined `nll`

This is the closest offline proxy for live usage quality.

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

## 7) Why later cutoffs usually score better

Later in day:

- more trajectory already observed
- less remaining uncertainty
- distribution naturally narrows

So it is normal that NLL improves from early morning to late afternoon.

## 8) Calibration: what each tool means

Peak calibration:

- isotonic regression
- can improve both logloss and Brier

Delta calibration:

- temperature scaling
- often smaller but still useful improvements

If calibration improves score, probabilities are usually more trustworthy.

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
