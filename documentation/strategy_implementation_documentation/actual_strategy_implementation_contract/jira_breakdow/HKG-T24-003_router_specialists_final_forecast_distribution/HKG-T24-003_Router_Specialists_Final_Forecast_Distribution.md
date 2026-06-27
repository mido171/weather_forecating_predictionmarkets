# HKG-T24-003 — Expected-Error Router, Specialists, Final Forecast Formula, and Distribution Layer

## Repository Implementation Location

In this repository, the contract path `src/hkg_t24` resolves to `code/src/hkg_t24`.

All implementation code for this Jira must live under `code/src/hkg_t24/`.

Supporting files must use:

```text
code/tests/hkg_t24/          tests
config/hkg_t24/              configuration
sql/hkg_t24/                 reviewed SQL/query assets
migrations/postgres/         durable PostgreSQL migrations
schemas/hkg_t24/             machine-readable schemas
reports/hkg_t24/             report indexes and non-canonical reports
artifacts/hkg_t24/           artifact indexes and small durable metadata
```

Do not put implementation logic in this Jira folder, root files, reports, notebooks, or ad hoc scripts. Scripts may call the package, but the package owns the implementation logic.

## Objective

Implement the trained expert-routing and final forecast system. This ticket consumes OOF expert predictions and feature matrices, trains/demotes/promotes routers, trains/demotes/promotes specialists, computes final strict/proxy/shadow replay predictions, applies fallback rules, trains distributional calibration, produces threshold probabilities, confidence states, and no-trade flags, and writes all router/specialist/distribution artifacts.

## Full Detailed Scope

Implement routers:

```text
R0_OFFICIAL_LONG_HISTORY
R1_CORE_GFS_GEFS
R2_IFS_SHADOW_ADAPTER
R3_AI_SHADOW_ADAPTER
R4_LIVE_SHADOW_ADAPTER
```

Implement specialists:

```text
S1_MARINE_SUPPRESSION
S2_WEAK_WIND_HEAT
S3_MAM_TRANSITION
S4_CLOUD_RAIN_SUPPRESSION
S5_DRY_RIDGE_HEAT
S6_HIGH_ERROR_TAIL
```

Implement final strict formula:

```text
If R1 experts E0/E1/E2/E4/E5 are available and R1 promoted: use R1.
Else if R0 available and promoted: use R0.
Else use best fallback E0, then E2.

base_forecast_c = sum(final_weight_e * expert_prediction_tmax_c_e)
specialist_total_c = clip(sum(active specialist corrections), -0.40, +0.40)
final_pre_distribution_c = base_forecast_c + specialist_total_c

if official exists:
    final_pre_distribution_c = clip(final_pre_distribution_c,
                                    official_forecast_max_c - 1.20,
                                    official_forecast_max_c + 1.20)

final_point_tmax_c = promoted distribution p50; otherwise final_pre_distribution_c
final_point_tmax_rounded_c = round(final_point_tmax_c, 1)
```

The router must train only on OOF predictions and cutoff-safe context features. No router may use in-sample expert predictions.

## Explicit Out of Scope

This ticket does not build source contracts, raw feature tables, or expert OOF predictions.

This ticket does not open sealed 2024 or 2025 labels.

This ticket does not train 2024 adapters.

This ticket does not run final sealed validation or live prediction commands.

## Required Implementation Steps

1. Implement modules:

```text
src/hkg_t24/models/static_weights.py
src/hkg_t24/models/expected_error.py
src/hkg_t24/models/router.py
src/hkg_t24/models/specialists.py
src/hkg_t24/models/distribution.py
src/hkg_t24/models/final_formula.py
src/hkg_t24/validation/metrics.py
src/hkg_t24/validation/slices.py
src/hkg_t24/validation/ablation.py
```

2. Implement CLI commands:

```bash
python -m hkg_t24.cli train-router --router R0
python -m hkg_t24.cli train-router --router R1
python -m hkg_t24.cli train-specialists --scope strict-pre2024
python -m hkg_t24.cli train-distribution --scope strict-pre2024
python -m hkg_t24.cli run-system-replay --scope strict-pre2024
```

3. Train expected-error models per expert:

```text
loss_e_t = abs(target_tmax_c - expert_prediction_tmax_c)
expected_error_e_t = h_e(context_t)
```

4. Clamp expected errors:

```text
0.20 <= expected_error <= 3.00
```

5. Optimize static weights with SLSQP:

```text
minimize mean(abs(y - sum(w_e * f_e)))
subject to:
  w_e >= 0
  sum(w_e) = 1
  w_e <= cap_e
```

6. Use caps:

```text
E0_OFFICIAL_RAW_ANCHOR: 0.80
E1_OFFICIAL_RESIDUAL: 0.80 if promoted, else 0
E2_TARGET_MEMORY: 0.40
E4_GFS_MOS: 0.70 if promoted, else 0
E5_GEFS_ENSEMBLE: 0.70 if promoted, else 0
E3_STATION_PROXY: 0 in strict, 0.40 in proxy
IFS combined: 0 pre-2024 strict, 0.10 after sealed protocol pass
AI combined: 0 pre-2024 strict, 0.05 after promotion
CWA/ARWF combined: 0 until live-shadow promotion
```

7. Implement dynamic weights:

```text
raw_dyn_weight_e = exp(-expected_error_e / tau)
dyn_weight_e = raw_dyn_weight_e / sum(raw_dyn_weight_j)
final_weight_e = (1 - lambda) * static_weight_e + lambda * dyn_weight_e
```

8. Use grids:

```text
tau: [0.25, 0.35, 0.50, 0.75, 1.00]
lambda: [0.00, 0.25, 0.50]
```

9. Tie-breaking:

```text
MAE tie <= 0.005°C: lower lambda
then higher tau
```

10. Apply availability masks:

```text
unavailable expert weight = 0
demoted expert weight = 0
blocked/proxy/shadow expert strict weight = 0
```

11. Apply cap redistribution:

```text
if total_allowed_weight > 0:
    renormalize
else if E0 available:
    E0=1
else if E2 promoted and available:
    E2=1
else:
    no forecast
```

12. Implement R0 with promoted E0/E1/E2 only, plus E3 only in proxy scope.

13. Implement R1 with promoted E0/E1/E2/E4/E5 only.

14. R1 must beat both E0 and R0 on identical rows:

```text
MAE improvement >= 0.01°C versus the better of E0 and R0
P90 AE worsening <= 0.02°C
abs(bias) <= abs(baseline bias) + 0.03°C
```

15. If R1 beats E0 but not R0, R1 is demoted.

16. Implement R2/R3/R4 as shadow/adapters with zero strict pre-2024 impact.

17. Write demoted router artifacts and fallback predictions.

18. Implement specialist deterministic prior formulas exactly below.

19. Use fold-local percentile ranking for specialist score components:

```text
pct_high(x) = (count(training_values <= x) + 0.5 * count(training_values = x)) / count(non_null_training_values)
pct_low(x) = 1 - pct_high(x)
missing component -> 0.50
score unavailable if more than 40% weighted components are missing
top 40% = prior_score >= training fold p60
```

20. Specialist target definitions:

```text
base_residual_c = target_tmax_c - router_base_forecast_c
correction_target_c = base_residual_c
specialist_candidate_forecast_c = router_base_forecast_c + clipped_candidate_correction_c
benefit_target_c = abs(target_tmax_c - router_base_forecast_c) - abs(target_tmax_c - specialist_candidate_forecast_c)
```

21. Shared correction model:

```text
HuberRegressor
epsilon = 1.35
alpha = 0.05
max_iter = 500
```

22. Shared benefit model:

```text
HistGradientBoostingRegressor
loss = absolute_error
max_iter = 100
max_leaf_nodes = 7
learning_rate = 0.05
l2_regularization = 1.0
random_state = 20260626
```

23. If correction model fails to converge, fallback:

```text
raw_correction_prediction_c = median(training correction_target_c among active candidate rows)
```

24. Specialist shrinkage:

```text
support_shrink = min(1.0, active_training_rows / 400.0)
shrunk_correction_c = support_shrink * raw_correction_prediction_c
applied_correction_c = clip(shrunk_correction_c, -0.25, +0.25)
```

25. Specialist activation requires all:

```text
score_available = true
prior_score >= fold_p60
expected_benefit_c >= 0.02
active_training_rows >= 200
active_year_count >= 3
correction_model_available = true
specialist_promotion_status = promoted
```

26. Total specialist cap:

```text
abs(sum specialist corrections) <= 0.40°C
```

27. No-harm gate:

```text
mean active_lift_c across folds >= 0.02
at least 3 folds have active_lift_c >= 0
no fold active_lift_c < -0.03
tail_p90_worsening_c <= 0.03 in every fold with at least 40 activated rows
```

28. Demote failed specialists globally for first strict implementation.

## Exact Specialist Formulas

### S1 — `S1_MARINE_SUPPRESSION`

Prior score:

```text
0.20 * pct_high(gfs__center__onshore_easterly_component_mps)
0.15 * pct_high(gefsmean__center__onshore_east_component_mps_mean)
0.20 * pct_high(gfs__spatial__inland_nw_minus_marine_s_tmax_c)
0.10 * pct_high(gfs__spatial__inland_nw_minus_center_tmax_c)
0.10 * pct_high(gfs__center__dewpoint_change_proxy_c)
0.10 * pct_high(gfs__center__low_cloud_pct_mean)
0.10 * pct_low(gfs__center__shortwave_w_m2_mean)
0.05 * pct_high(official__forecast_max_minus_gefs_median_c)
```

Expected sign: negative.

Demote if active-row median correction target is greater than `-0.03°C`.

### S2 — `S2_WEAK_WIND_HEAT`

Prior score:

```text
0.20 * pct_low(gfs__center__wind_speed_10m_mean_mps)
0.15 * pct_low(gefsmean__center__wind_speed_10m_mps_mean)
0.20 * pct_high(gfs__center__shortwave_w_m2_mean)
0.10 * pct_low(gfs__center__low_cloud_pct_mean)
0.15 * pct_high(gfs__center__t850_c_mean)
0.10 * pct_high(target__slope7_minus_slope30_lag2_c_per_day)
0.05 * pct_high(gfs__spatial__inland_nw_minus_center_tmax_c)
0.05 * pct_high(gfs__center__temp_dewpoint_spread_mean_c)
```

Expected sign: positive.

Demote if active-row median correction target is below `+0.03°C`.

### S3 — `S3_MAM_TRANSITION`

Gate:

```text
month in {3,4,5}
```

If not MAM, score `0`, inactive.

If MAM, prior score:

```text
0.20 * pct_high(abs(target__slope7_minus_slope30_lag2_c_per_day))
0.15 * pct_high(target__roll14_std_lag2_c)
0.15 * pct_high(abs(target__lag2_minus_roll7_c))
0.15 * pct_high(abs(official__forecast_max_minus_target_roll7_c))
0.10 * pct_high(gfs__center__low_cloud_pct_mean)
0.10 * pct_high(gfs__center__precip_mm_sum)
0.10 * pct_high(gefsens__center__tmax_spread_p90_p10_c)
0.05 * pct_high(abs(gfs__center__tmax_c - gefsmean__center__tmax_c))
```

Expected sign: learned, no sign demotion.

Training uses MAM rows only, but writes inactive rows for all months.

### S4 — `S4_CLOUD_RAIN_SUPPRESSION`

Prior score:

```text
0.20 * pct_high(gfs__center__low_cloud_pct_mean)
0.15 * pct_high(gfs__center__precip_mm_sum)
0.15 * pct_low(gfs__center__shortwave_w_m2_mean)
0.15 * pct_high(gfs__center__relative_humidity_700_pct_mean)
0.10 * pct_high(gefsmean__center__pwat_kg_m2_mean)
0.10 * pct_high(gfs__center__dewpoint_2m_c_mean)
0.10 * pct_high(gefsens__center__tmax_spread_p90_p10_c)
0.05 * pct_high(official__psr_numeric_proxy)
```

Expected sign: negative.

Demote if active-row median correction target is greater than `-0.03°C`.

### S5 — `S5_DRY_RIDGE_HEAT`

Prior score:

```text
0.20 * pct_high(gfs__center__z500_m_mean)
0.15 * pct_high(gfs__center__t850_c_mean)
0.15 * pct_low(gfs__center__relative_humidity_700_pct_mean)
0.15 * pct_high(gfs__center__shortwave_w_m2_mean)
0.10 * pct_low(gfs__center__low_cloud_pct_mean)
0.10 * pct_low(gfs__center__precip_mm_sum)
0.10 * pct_low(gfs__center__wind_speed_10m_mean_mps)
0.05 * pct_high(target__hot_spell_length_lag2_days)
```

Expected sign: positive.

Demote if active-row median correction target is below `+0.03°C`.

### S6 — `S6_HIGH_ERROR_TAIL`

Prior score:

```text
0.25 * pct_high(router__expert_prediction_spread_c)
0.20 * pct_high(gefsens__center__tmax_spread_p90_p10_c)
0.15 * pct_high(online__official_raw__global__abs_error_h20_c)
0.10 * pct_high(abs(official__forecast_max_minus_gfs_center_tmax_c))
0.10 * pct_high(abs(official__forecast_max_minus_gefs_median_c))
0.10 * pct_high(target__roll14_std_lag2_c)
0.10 * pct_high(router__missing_expert_count)
```

Additional activation condition:

```text
router__expected_abs_error_c >= training_fold_p60_router_expected_abs_error
```

Expected sign: learned.

No-harm stricter:

```text
P95 AE must improve or remain within +0.01°C.
MAE on non-activated rows must be identical to baseline.
```

## Distributional Layer

Train on OOF final system predictions from pre-2024 development rows.

Residual:

```text
system_residual_c = target_tmax_c - final_point_pre_distribution_c
```

Train LightGBM quantile residual models for:

```text
0.10, 0.25, 0.50, 0.75, 0.90
```

Distribution inputs:

```text
final point forecast
router expected expert errors
GEFS spread
expert disagreement
calendar features
specialist activation flags
source availability masks
```

Output:

```text
p10_c
p25_c
p50_c
p75_c
p90_c
expected_abs_error_c
prob_tmax_ge_20_0 through prob_tmax_ge_40_0 by 0.5
confidence_state
no_trade_flag
```

Quantile monotonicity:

```text
p10 <= p25 <= p50 <= p75 <= p90
```

If violated, sort quantile values ascending and flag `quantile_monotonic_repair=true`.

If quantile models fail or P50 worsens MAE by more than `0.005°C`:

```text
distributional_promotion_status = demoted_empirical_fallback
p50_c = final_pre_distribution_c
```

Empirical fallback intervals:

```text
Use OOF residual quantiles from promoted frozen candidate.
If month has >= 100 training residuals, use month-specific residual quantiles.
Else use global residual quantiles.
```

If degenerate:

```text
p10 = p50 - 1.20
p25 = p50 - 0.60
p75 = p50 + 0.60
p90 = p50 + 1.20
```

Threshold probabilities:

```text
Keys: prob_tmax_ge_20_0 through prob_tmax_ge_40_0 inclusive by 0.5.
Exactly 41 keys.
```

Gaussian fallback:

```text
sigma = max(expected_abs_error_c * sqrt(pi/2), 0.60)
prob_tmax_ge_X = 1 - NormalCDF((X - p50_c)/sigma)
clamp to [0.001, 0.999]
```

Expected absolute error:

```text
LightGBM regression_l1 target = abs(system_residual_c)
Clamp expected_abs_error_c between 0.20 and 3.00
```

Confidence state:

```text
HIGH if expected_abs_error_c <= 0.55 and expert_disagreement <= fold-local p50
MEDIUM if expected_abs_error_c <= 0.85
LOW otherwise
```

No-trade flag:

```text
true if confidence_state = LOW
or expected_abs_error_c > 1.00
or source availability misses official and all core NWP
or leakage_status != passed
```

## Required Database Schemas / Tables / Views / Materializations

Use/create:

```text
model_router.router_prediction
model_router.specialist_prediction
model_oof.system_prediction
model_eval.system_prediction_component
model_validation.scoreboard
model_router.router_scoreboard
```

`model_router.router_scoreboard` must include:

```text
router_version
router_scope
promotion_status
promotion_gate_passed
demotion_reason
included_experts_jsonb
excluded_experts_jsonb
mae
baseline_mae
mae_delta_vs_baseline
mae_delta_vs_r0
p90_abs_error_delta
row_count
first_date
last_date
```

## Required CLI Commands / Scripts / Modules

```bash
python -m hkg_t24.cli train-router --router R0
python -m hkg_t24.cli train-router --router R1
python -m hkg_t24.cli train-specialists --scope strict-pre2024
python -m hkg_t24.cli train-distribution --scope strict-pre2024
python -m hkg_t24.cli run-system-replay --scope strict-pre2024
```

## Required Feature / Model / Artifact Outputs

Router outputs:

```text
reports/router_scoreboard.csv
reports/router_weight_diagnostics.csv
reports/router_promotion_decisions.csv
```

Specialist outputs:

```text
reports/specialist_scoreboard.csv
reports/specialist_activation_report.csv
reports/specialist_no_harm_report.csv
reports/specialist_promotion_decisions.csv
```

Distribution outputs:

```text
reports/distribution_scoreboard.csv
reports/distribution_calibration_report.csv
reports/distribution_calibration_report.md
reports/calibration_report.md
reports/threshold_probability_scoreboard.csv
reports/prediction_interval_coverage_report.csv
```

System outputs:

```text
reports/system_scoreboard_strict.csv
reports/system_scoreboard_proxy.csv
reports/system_ablation_matrix.csv
```

## Required Provenance / Audit / Logging Behavior

Every router prediction must store:

```text
router_version
router_scope
fold_id
base_forecast_c
static_weight_jsonb
dynamic_weight_jsonb
final_weight_jsonb
expected_error_jsonb
availability_mask_jsonb
selected_tau
selected_lambda
promotion_status
demotion_reason
expert_mask_jsonb
cap_trace_jsonb
```

Every specialist prediction must store:

```text
specialist_id
fold_id
prior_score
score_available
fold_p60
regime_probability
raw_correction_c
shrunk_correction_c
applied_correction_c
expected_benefit_c
activated
activation_reason
support_count
no_harm_pass
promotion_status
```

Every final system prediction must store:

```text
base_forecast_c
specialist_total_correction_c
final_point_tmax_c
p10/p25/p50/p75/p90
expected_abs_error_c
confidence_state
no_trade_flag
component_jsonb
```

## Required Fail-Closed / Error Behavior

Fail closed when:

```text
router trains on non-OOF expert prediction
expected-error model sees target label outside training fold
static blend row set does not match identical-row comparison
unavailable expert receives nonzero weight
strict router uses proxy or shadow expert
specialist feature missing from dictionaries
specialist uses post-cutoff or outcome-derived feature
distribution probability keys not exactly 41
quantile monotonicity cannot be repaired
final formula produces forecast without available fallback expert
```

Demote rather than fail when:

```text
router fails promotion
specialist fails no-harm gate
distribution quantile model fails
short-history challenger unavailable
```

## Leakage-Free / Non-Forward-Looking Requirements

This Jira must enforce:

- router training uses OOF expert predictions only;
- router context uses only H24N cutoff-safe features;
- no router context contains `target_tmax_c` except as training label for scoring/loss;
- no same-row residual is a router input;
- no specialist detector uses future residual as feature;
- specialist detector targets may use labels only inside training folds;
- specialist activation on validation/live rows uses only prior-trained thresholds and cutoff-safe features;
- distributional layer trains only on OOF residuals;
- no 2024+ labels are used in pre-2024 router/specialist/distribution training;
- no global scaling is fit across train/test;
- no proxy/shadow source enters strict router or strict final formula;
- no post-cutoff GribStream row contributes to router context or specialist features;
- every router/specialist/distribution artifact records train dates and feature schema version.

## Dependencies on Earlier Jiras

Depends on HKG-T24-001 and HKG-T24-002.

## Acceptance Criteria

1. R0 trains on pre-2024 eligible rows using promoted E0/E1/E2, with E3 only in proxy scope.
2. R1 trains on `2021-03-23` through `2023-12-31` common rows using promoted E0/E1/E2/E4/E5.
3. R1 is promoted only if it beats both E0 and R0 on identical rows by required thresholds.
4. Demoted experts receive zero weight.
5. Missing experts receive zero weight.
6. Static and dynamic weights sum to 1 after masks/caps.
7. No strict router includes proxy or shadow prefixes.
8. Every specialist writes rows, even if inactive or demoted.
9. Every promoted specialist passes support, active lift, and P90 no-harm gates.
10. Total specialist correction never exceeds `±0.40°C`.
11. Final forecast is clipped to official `±1.20°C` when official exists.
12. Distribution outputs exactly 41 threshold probabilities.
13. Quantile outputs are monotonic or repaired with audit flag.
14. If distribution demoted, empirical/Gaussian fallback still produces intervals and probabilities.
15. No-trade flag is produced for every final prediction.
16. Reports for router, specialist, distribution, and system replay exist.
17. No leakage audit `ERROR` events exist for accepted strict system predictions.

## Extensive Test Scenarios

Unit tests:

```text
tests/unit/test_router_weight_math.py
tests/unit/test_specialist_prior_scores.py
tests/unit/test_specialist_caps.py
tests/unit/test_distribution_monotonicity.py
tests/unit/test_threshold_probability_keys.py
tests/unit/test_final_formula.py
```

Integration tests:

```text
tests/integration/test_router_training_oof_only.py
tests/integration/test_router_demotion_fallback.py
tests/integration/test_specialist_training_and_demotion.py
tests/integration/test_distribution_fallback.py
tests/integration/test_system_replay_strict_pre2024.py
```

## Required Smoke Tests

Run:

```bash
python -m hkg_t24.cli train-router --router R0 --smoke
python -m hkg_t24.cli train-router --router R1 --smoke
python -m hkg_t24.cli train-specialists --scope strict-pre2024 --smoke
python -m hkg_t24.cli train-distribution --scope strict-pre2024 --smoke
python -m hkg_t24.cli run-system-replay --scope strict-pre2024 --smoke
```

Expected outputs:

```text
router predictions >= 40 rows
specialist prediction rows for all six specialists
distribution rows >= 40
system replay rows >= 40
all weight sums within 1e-8 of 1.0 where forecast exists
zero strict proxy/shadow weights
```

## Required Integration Tests

Integration tests must prove:

- R0/R1 use only OOF expert predictions;
- R1 common-row set is intersection of required experts;
- R1 demotes when it fails to beat E0 and R0;
- cap redistribution follows final patch;
- specialists use fold-local p60 scores;
- specialist missing inputs use neutral 0.50;
- specialist activation requires all gates;
- distribution P50 fallback works;
- threshold probability keys are exact;
- final forecast formula follows fallback ladder.

## Leakage and Temporal Integrity Tests

Required tests:

```text
Router in-sample contamination test: replace OOF flag with false and ensure router refuses.
Router future-label test: make training row include validation target and ensure scanner fails.
Specialist outcome-feature test: inject base_residual as validation feature and ensure rejection.
Distribution future residual test: attempt to train intervals on non-OOF residuals and ensure failure.
Proxy strict-inclusion test: insert station__ feature into strict router context and ensure failure.
Shadow strict-inclusion test: insert ifsoper__ feature into strict R1 context and ensure failure.
GribStream contamination test: router context using unsafe NWP feature fails.
Fold-local percentile test: specialist p60 computed on training fold only.
```

## Required Negative-Control Tests Where Relevant

Component-level controls:

```text
shuffled target control for router
lag-shifted NWP control for R1
specialist randomized activation control
distribution shuffled residual control
```

Pass conditions:

```text
negative-control router must not beat official raw by more than 0.02°C
lag-shifted NWP system must not improve official raw by more than 0.02°C
random specialist activation must not pass promotion gates
shuffled residual distribution must not improve calibration gates
```

## Required Final Artifacts / Reports

```text
reports/router_scoreboard.csv
reports/router_weight_diagnostics.csv
reports/router_promotion_decisions.csv
reports/specialist_scoreboard.csv
reports/specialist_activation_report.csv
reports/specialist_no_harm_report.csv
reports/specialist_promotion_decisions.csv
reports/distribution_scoreboard.csv
reports/distribution_calibration_report.csv
reports/distribution_calibration_report.md
reports/calibration_report.md
reports/threshold_probability_scoreboard.csv
reports/prediction_interval_coverage_report.csv
reports/system_scoreboard_strict.csv
reports/system_scoreboard_proxy.csv
reports/system_ablation_matrix.csv
```

## Definition of Done

This Jira is done when all routers, specialists, final formula, and distributional calibration are implemented, strict/proxy/shadow scoreboards are produced, demotion/fallback behavior is verified, final pre-2024 system replay exists, and every leakage, temporal-integrity, weight, cap, threshold, and distribution test passes.
