# 20 - Engineering Spec (2026-03-04, Bucket Probability Interpolation for YES/NO)

## 1) Purpose

Define the exact strategy used by the live MOS inference pipeline to convert model quantiles into per-bucket probabilities:

1. `bucket_yes_prob`
2. `bucket_no_prob`

This is the probability layer used for Kalshi bucket event scoring in live inference outputs.

Primary implementation:

1. `tools/live/mos_quantile_live_inference.py`
   1. `parse_bucket_label(...)`
   2. `cdf_from_quantiles(...)`
   3. `pmf_int_from_quantiles(...)`
   4. `bucket_prob(...)`
   5. `build_bucket_probabilities(...)`
   6. `enforce_non_cross(...)`

## 2) Input and Quantile Set

The live script uses station quantiles produced from the `blend_12` bundle:

1. `q_0.05`
2. `q_0.10`
3. `q_0.25`
4. `q_0.50`
5. `q_0.75`
6. `q_0.90`
7. `q_0.95`

These are values of forecasted Tmax (in Fahrenheit) at the given quantile levels.

## 3) Exact Algorithm

### 3.1 Quantile monotonic guardrail

Before interpolation, quantiles are forced non-decreasing (`enforce_non_cross`):

1. Iterate quantiles from low to high probability.
2. Apply running max:
   `q_fixed[tau_i] = max(q_raw[tau_i], q_fixed[tau_{i-1}])`

This removes quantile crossing artifacts so CDF interpolation remains valid.

### 3.2 Quantile-to-CDF interpolation

The script builds `F(x)` using piecewise linear interpolation in value space:

1. `taus = sorted(qmap.keys())`
2. `qvals = [qmap[t] for t in taus]`
3. `qvals = cumulative_max(qvals)` (same monotonic fix)
4. `F(x) = interp(x, x_points=qvals, y_points=taus, left=0.0, right=1.0)`

Interpretation:

1. Temperatures below the smallest quantile map to probability `0`.
2. Temperatures above the largest quantile map to probability `1`.
3. Between quantiles, CDF is linearly interpolated.

### 3.3 CDF-to-integer PMF conversion

The script converts the continuous CDF into integer Fahrenheit mass on support `[-20, 130]`:

`P(T=t) = F(t+0.5) - F(t-0.5)`

for each integer `t`.

Then:

1. Negative values are clipped to `0` (`max(0, p)`).
2. PMF is renormalized to sum to `1`.
3. If the total mass is `<= 0`, fallback is uniform over `[-20, 130]`.

### 3.4 Bucket parsing

Bucket text is parsed from market column labels:

1. `"75F or below"` -> mode `or_below`, `hi=75`
2. `"84F or above"` -> mode `or_above`, `lo=84`
3. `"80F to 81F"` -> mode `range`, `lo=80`, `hi=81`

Parsing uses integer extraction, then inclusive integer membership.

### 3.5 Bucket YES and NO probabilities

Given PMF:

1. `or_below`: `P_yes = sum_{t <= hi} P(T=t)`
2. `or_above`: `P_yes = sum_{t >= lo} P(T=t)`
3. `range`: `P_yes = sum_{lo <= t <= hi} P(T=t)`

Then always:

1. `bucket_yes_prob = P_yes`
2. `bucket_no_prob = 1 - P_yes`

Important: NO is defined as the complement event for that specific bucket.

## 4) Worked Example A (Synthetic Quantiles)

Quantile map:

1. `q05=77`
2. `q10=78`
3. `q25=79`
4. `q50=80`
5. `q75=81`
6. `q90=82`
7. `q95=83`

Resulting integer PMF (selected values):

1. `P(77)=0.075`
2. `P(78)=0.100`
3. `P(79)=0.200`
4. `P(80)=0.250`
5. `P(81)=0.200`
6. `P(82)=0.100`
7. `P(83)=0.075`

Example bucket probabilities:

1. Bucket `80-81`:
   `P_yes = P(80)+P(81)=0.25+0.20=0.45`
   `P_no = 0.55`
2. Bucket `79 or below`:
   `P_yes = P(T<=79)=0.375`
   `P_no = 0.625`
3. Bucket `82 or above`:
   `P_yes = P(T>=82)=0.175`
   `P_no = 0.825`

## 5) Worked Example B (Real Run Artifact)

Artifact:

1. `D:\Ahmed\data\live\mos_quantile_live_inference\20260302T144741Z_target_20260226\inference_report.json`
2. Station: `KMIA`

Quantiles in that run:

1. `q_0.05=77.91686544871057`
2. `q_0.10=78.43550742964894`
3. `q_0.25=79.02822896427321`
4. `q_0.50=80.09006548334594`
5. `q_0.75=80.38674883715044`
6. `q_0.90=81.20483833997062`
7. `q_0.95=81.59648503295136`

Reported bucket probabilities (exactly consistent with interpolation implementation):

1. `78F to 79F`: YES `0.36107431022874725`, NO `0.6389256897712527`
2. `80F to 81F`: YES `0.5766078233522826`, NO `0.42339217664771744`
3. `82F to 83F`: YES `0.06231786641897019`, NO `0.9376821335810298`
4. `84F or above`: YES `0.0`, NO `1.0`

## 6) Crossing-Quantile Example (Why `enforce_non_cross` matters)

If raw quantiles are:

1. `q25=81`
2. `q50=80`
3. `q75=79.5`

they are invalid (crossing). The script converts them to:

1. `q25=81`
2. `q50=81`
3. `q75=81`

using running max. This prevents negative/illogical local CDF slopes.

## 7) Semantics for Trading Logic

For a specific Kalshi bucket market:

1. YES side model probability uses `bucket_yes_prob`.
2. NO side model probability uses `bucket_no_prob = 1 - bucket_yes_prob`.

This is event-complement logic. It is not derived from exchange bid/ask or orderbook microstructure.

## 8) Known Constraints and Design Choices

1. Temperature support for PMF is truncated to integer range `[-20, 130]`.
2. Bucket parsing currently uses integer extraction from label text.
3. Range buckets are inclusive on both ends.
4. A small interpolation/normalization artifact can exist in extreme tails due to finite support and integerization.
5. PMF is explicitly normalized, so bucket YES + NO is always exactly `1` (subject to floating-point representation).

## 9) Traceability

Implementation references:

1. `tools/live/mos_quantile_live_inference.py`
2. `tools/live/mos_blend12_bundle.py`

Related docs:

1. `documentation/mos/10_run_record_2026-03-02_model_export_and_live_inference.md`
2. `documentation/mos/11_run_record_2026-03-02_cojoined_blend12_live_script_replay.md`
3. `documentation/mos/12_run_record_2026-03-02_ui_toggle_and_2026_live_script_replay.md`
4. `documentation/mos/19_engineering_spec_2026-03-04_live_inference_station_generic_and_shared_bundle_module.md`


