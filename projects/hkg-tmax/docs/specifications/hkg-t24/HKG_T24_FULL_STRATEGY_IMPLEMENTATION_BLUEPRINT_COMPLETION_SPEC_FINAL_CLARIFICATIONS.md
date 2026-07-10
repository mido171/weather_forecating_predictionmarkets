# HKG T24 / H24N Full Strategy Implementation Blueprint — Final Clarifications Addendum

**Document path:** `HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT_COMPLETION_SPEC_FINAL_CLARIFICATIONS.md`  
**Status:** binding supplement to `HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT_COMPLETION_SPEC.md`  
**Cutoff:** `H24N` only, meaning `15:00 HKT` on `T-1`, equivalent to `07:00 UTC` on `T-1`  
**Schema version:** `hkg_t24_h24n_full_impl_v1_final_20260626`  
**Feature schema version:** `hkg_t24_h24n_features_v1_final_20260626`  
**Router schema version:** `hkg_t24_h24n_router_v1_final_20260626`

This addendum resolves the remaining implementation ambiguities in the previous completion specification. It **does not replace** the prior completion specification. It overrides the prior specification only where this addendum gives a more specific instruction. Codex must implement the instructions below exactly.

Binding rule: every date, feature, model, router, specialist, validation command, and artifact must fail closed when its point-in-time contract is not proven. No model may infer or backfill missing eligibility from a later run, a later target label, a later forecast revision, or a future-fitted preprocessing object.

---

## 1. Online residual-memory features

### 1.1 Purpose

Online residual-memory features encode recent source-specific error state using only already-settled target dates strictly before the forecast target date. They are used by:

- `E1_OFFICIAL_RESIDUAL` as official-forecast bias context;
- routers as source-trust context;
- live scoring as a causal state updated after settlement only.

They are not labels. They are not allowed to use the current target date outcome.

### 1.2 New table schema

Create this permanent rebuildable table. It may be fully regenerated from OOF/live predictions and labels.

```sql
CREATE TABLE IF NOT EXISTS model_features.online_residual_state (
  state_id text PRIMARY KEY,
  schema_version text NOT NULL DEFAULT 'hkg_t24_h24n_online_state_v1_final_20260626',
  cutoff_id text NOT NULL DEFAULT 'H24N',
  target_date date NOT NULL,
  snapshot_id text NOT NULL,
  source_key text NOT NULL,
  source_family text NOT NULL,
  state_scope text NOT NULL,
  residual_reference text NOT NULL,
  n_prior_rows integer NOT NULL,
  latest_prior_target_date date NULL,
  updated_through_target_date date NULL,
  ewma_bias_h5_c double precision NULL,
  ewma_bias_h10_c double precision NULL,
  ewma_bias_h20_c double precision NULL,
  ewma_bias_h40_c double precision NULL,
  ewma_abs_error_h5_c double precision NULL,
  ewma_abs_error_h10_c double precision NULL,
  ewma_abs_error_h20_c double precision NULL,
  ewma_abs_error_h40_c double precision NULL,
  ewma_sq_error_h5_c2 double precision NULL,
  ewma_sq_error_h10_c2 double precision NULL,
  ewma_sq_error_h20_c2 double precision NULL,
  ewma_sq_error_h40_c2 double precision NULL,
  residual_volatility_h5_c double precision NULL,
  residual_volatility_h10_c double precision NULL,
  residual_volatility_h20_c double precision NULL,
  residual_volatility_h40_c double precision NULL,
  overforecast_streak_count integer NOT NULL DEFAULT 0,
  underforecast_streak_count integer NOT NULL DEFAULT 0,
  neutral_streak_count integer NOT NULL DEFAULT 0,
  correction_bias_h20_shrunk_c double precision NULL,
  correction_bias_h20_capped_c double precision NULL,
  expected_abs_error_h20_shrunk_c double precision NULL,
  warmup_status text NOT NULL,
  state_available boolean NOT NULL DEFAULT false,
  provenance_prediction_table text NOT NULL,
  provenance_label_table text NOT NULL,
  created_at_utc timestamptz NOT NULL DEFAULT now(),
  UNIQUE (cutoff_id, target_date, source_key, state_scope, residual_reference)
);

CREATE INDEX IF NOT EXISTS idx_online_state_target ON model_features.online_residual_state(cutoff_id, target_date);
CREATE INDEX IF NOT EXISTS idx_online_state_source ON model_features.online_residual_state(source_key, state_scope, residual_reference);
```

Allowed `source_key` values for first full implementation:

```text
official_raw
official_residual
hko_target_memory
gfs_mos
gefs_prob_mos
r0_router
r1_router
final_system
```

Allowed `state_scope` values:

```text
global
source_era
month
season
source_era_month
```

Allowed `residual_reference` values:

```text
actual_minus_forecast
actual_minus_router_base
actual_minus_final_system
```

### 1.3 Exact feature names exposed to matrices

For every online state row joined to a snapshot, expose these feature names:

```text
online__{source_key}__{state_scope}__n_prior_rows
online__{source_key}__{state_scope}__bias_h5_c
online__{source_key}__{state_scope}__bias_h10_c
online__{source_key}__{state_scope}__bias_h20_c
online__{source_key}__{state_scope}__bias_h40_c
online__{source_key}__{state_scope}__abs_error_h5_c
online__{source_key}__{state_scope}__abs_error_h10_c
online__{source_key}__{state_scope}__abs_error_h20_c
online__{source_key}__{state_scope}__abs_error_h40_c
online__{source_key}__{state_scope}__volatility_h5_c
online__{source_key}__{state_scope}__volatility_h10_c
online__{source_key}__{state_scope}__volatility_h20_c
online__{source_key}__{state_scope}__volatility_h40_c
online__{source_key}__{state_scope}__overforecast_streak_count
online__{source_key}__{state_scope}__underforecast_streak_count
online__{source_key}__{state_scope}__correction_bias_h20_capped_c
online__{source_key}__{state_scope}__expected_abs_error_h20_shrunk_c
online__{source_key}__{state_scope}__state_available
```

The first implementation must build these scopes:

```text
official_raw/global
official_raw/source_era
official_raw/month
official_raw/season
gfs_mos/global
gefs_prob_mos/global
r0_router/global
r1_router/global
final_system/global
```

For `gfs_mos`, `gefs_prob_mos`, `r1_router`, and `final_system`, the state exists only on dates after their first OOF or live prediction. Before that, emit unavailable states with `state_available=false` and `n_prior_rows=0`.

### 1.4 Residual definition

For a prior target date `d < target_date`:

```text
residual_c(d, source_key) = target_tmax_c(d) - prediction_tmax_c(d, source_key)
abs_error_c(d, source_key) = abs(residual_c(d, source_key))
sq_error_c2(d, source_key) = residual_c(d, source_key)^2
```

Overforecast and underforecast definitions:

```text
overforecast day: residual_c <= -0.05
underforecast day: residual_c >= +0.05
neutral day: -0.05 < residual_c < +0.05
```

The `0.05 C` tolerance prevents floating-point and rounding ties from creating streaks.

### 1.5 EWMA formulas

Use half-lives exactly:

```text
h in {5, 10, 20, 40}
alpha_h = 1 - exp(-ln(2) / h)
```

For a chronological sequence of prior residuals ordered by target date:

```text
ewma_bias_h(d) = alpha_h * residual_c(d) + (1 - alpha_h) * ewma_bias_h(previous_d)
ewma_abs_error_h(d) = alpha_h * abs_error_c(d) + (1 - alpha_h) * ewma_abs_error_h(previous_d)
ewma_sq_error_h(d) = alpha_h * sq_error_c2(d) + (1 - alpha_h) * ewma_sq_error_h(previous_d)
residual_volatility_h(d) = sqrt(max(ewma_sq_error_h(d) - ewma_bias_h(d)^2, 0))
```

Initialization on the first prior row:

```text
ewma_bias_h(first) = residual_c(first)
ewma_abs_error_h(first) = abs_error_c(first)
ewma_sq_error_h(first) = sq_error_c2(first)
residual_volatility_h(first) = 0
```

The online state stored for `target_date=T` is the state after processing the last available prior row with `target_date <= T-1`. It must never process target date `T` before prediction for `T`.

### 1.6 Minimum history and warmup behavior

Use these exact warmup states:

```text
n_prior_rows = 0: warmup_status = NO_HISTORY, state_available = false, all numeric state features NULL except streak counts = 0
1 <= n_prior_rows < 5: warmup_status = COLD_START, state_available = true
5 <= n_prior_rows < 20: warmup_status = WARMING, state_available = true
n_prior_rows >= 20: warmup_status = READY, state_available = true
```

Models may receive COLD_START and WARMING states, but correction features must be shrunk aggressively as defined below.

### 1.7 Shrinkage and capping

For a half-life `h`, define:

```text
shrink_h = n_prior_rows / (n_prior_rows + 2*h)
```

The official residual correction candidate uses `h=20`:

```text
correction_bias_h20_shrunk_c = shrink_20 * ewma_bias_h20_c
correction_bias_h20_capped_c = clip(correction_bias_h20_shrunk_c, -0.40, +0.40)
expected_abs_error_h20_shrunk_c = max(0.20, shrink_20 * ewma_abs_error_h20_c + (1 - shrink_20) * 0.90)
```

The `0.90` fallback is the conservative default expected absolute error before sufficient source history. The `0.20` floor prevents the router from treating a short perfect streak as near-certain accuracy.

### 1.8 Update timing after settlement

Live update sequence for target date `T`:

```text
1. Before 15:00 HKT on T-1, create prediction for T using online states through T-1 only.
2. After HKO target Tmax for T is available in the label table, run settlement scoring.
3. Compute residuals for all experts/systems that predicted T.
4. Update online states for target dates >= T+1.
5. Never update the state for T itself after prediction creation.
```

### 1.9 Fold-local replay logic for OOF

OOF online states must be generated by chronological replay inside each outer fold.

For each outer fold:

```text
1. Sort all dates up to the fold end chronologically.
2. For each date d in order:
   a. Build online state for d using only dates < d.
   b. Generate any expert/router prediction for d using that state.
   c. After d is scored inside the replay, make d available to states for dates > d.
3. For validation dates, store the prediction and online state exactly as they existed before scoring that date.
```

Training folds may use predictions generated by earlier chronological training replay, but no validation-row residual may enter its own features.

### 1.10 Responsible module and CLI

Create:

```text
src/hkg_tmax/features/online_state.py
```

Required CLI commands:

```bash
python -m hkg_tmax.cli build-online-states \
  --cutoff-id H24N \
  --scope strict \
  --from-date YYYY-MM-DD \
  --to-date YYYY-MM-DD

python -m hkg_tmax.cli replay-online-states-oof \
  --cutoff-id H24N \
  --fold-spec config/folds/h24n_folds_v1.yaml

python -m hkg_tmax.cli update-online-states-after-settlement \
  --cutoff-id H24N \
  --target-date YYYY-MM-DD
```

The commands must write `model_features.online_residual_state` and `reports/online_state_audit.csv`. The audit must contain row counts by `source_key`, `state_scope`, warmup status, and the maximum target date visible to each state.

---

## 2. Specialist prior scores

### 2.1 General scoring method

Each specialist uses a deterministic fold-local prior score. The detector is not a free-form ML classifier in the first implementation. The detector probability is the deterministic prior score after fold-local percentile scaling.

For every raw component `x`, compute a fold-local percentile on the training fold only:

```text
pct_high(x) = (count(training_values <= x) + 0.5 * count(training_values = x)) / count(non_null_training_values)
pct_low(x)  = 1 - pct_high(x)
```

For a live or validation value outside the training range, clamp to `[0, 1]` after percentile interpolation.

Missing component handling:

```text
component missing -> component_percentile = 0.50 and component_missing_flag = 1
component present -> component_missing_flag = 0
```

A specialist score is unavailable when more than `40%` of its weighted components are missing by total component weight. Unavailable specialist score means:

```text
score_available = false
prior_score = NULL
activation = false
applied_correction_c = 0
```

Top 40% construction:

```text
fold_p60 = 60th percentile of prior_score among score_available training rows
candidate_regime_flag = 1 if prior_score >= fold_p60 else 0
```

If fewer than `200` score-available training rows exist, the specialist is demoted for that fold and emits inactive rows only.

### 2.2 Common targets for all specialists

For each specialist `s`, after the router base forecast is generated OOF:

```text
base_residual_c = target_tmax_c - router_base_forecast_c
correction_target_c = base_residual_c
```

A specialist-specific expected sign is used for reporting and demotion:

```text
S1_MARINE_SUPPRESSION: expected correction sign negative
S2_WEAK_WIND_HEAT: expected correction sign positive
S3_MAM_TRANSITION: sign learned; no fixed sign
S4_CLOUD_RAIN_SUPPRESSION: expected correction sign negative
S5_DRY_RIDGE_HEAT: expected correction sign positive
S6_HIGH_ERROR_TAIL: sign learned; no fixed sign
```

Benefit target is calculated from OOF specialist candidate predictions:

```text
specialist_candidate_forecast_c = router_base_forecast_c + clipped_candidate_correction_c
benefit_target_c = abs(target_tmax_c - router_base_forecast_c) - abs(target_tmax_c - specialist_candidate_forecast_c)
```

Positive benefit means the specialist improved MAE.

### 2.3 Shared correction and benefit model defaults

Correction model:

```text
model_class = sklearn.linear_model.HuberRegressor
epsilon = 1.35
alpha = 0.05
max_iter = 500
```

Benefit model:

```text
model_class = sklearn.ensemble.HistGradientBoostingRegressor
loss = absolute_error
max_iter = 100
max_leaf_nodes = 7
learning_rate = 0.05
l2_regularization = 1.0
random_state = 20260626
```

If the correction model fails to converge, fit this deterministic fallback:

```text
correction_prediction_c = median(training correction_target_c among active candidate rows)
```

Then shrink and cap as normal.

Candidate correction shrinkage:

```text
support_shrink = min(1.0, active_training_rows / 400.0)
shrunk_correction_c = support_shrink * raw_correction_prediction_c
```

Per-specialist cap:

```text
clip(shrunk_correction_c, -0.25, +0.25)
```

Global specialist total cap remains:

```text
clip(sum(active_specialist_corrections), -0.40, +0.40)
```

### 2.4 Activation rule for all specialists

A specialist activates only when all conditions are true:

```text
score_available = true
prior_score >= fold_p60
expected_benefit_c >= 0.02
active_training_rows >= 200
active_year_count >= 3
correction_model_available = true
specialist_promotion_status = promoted
```

Inactive specialists still write prediction rows with:

```text
activated = false
applied_correction_c = 0
activation_reason = explicit reason code
```

### 2.5 No-harm calculation

For every specialist, calculate on each outer validation fold:

```text
mae_base_active = mean(abs(target_tmax_c - router_base_forecast_c)) on activated rows
mae_specialist_active = mean(abs(target_tmax_c - (router_base_forecast_c + applied_correction_c))) on activated rows
active_lift_c = mae_base_active - mae_specialist_active

tail_p90_base_active = percentile_90(abs(base_residual_c))
tail_p90_specialist_active = percentile_90(abs(target_tmax_c - specialist_forecast_c))
tail_p90_worsening_c = tail_p90_specialist_active - tail_p90_base_active
```

Promotion requires:

```text
mean active_lift_c across folds >= 0.02
at least 3 folds have active_lift_c >= 0
no fold has active_lift_c < -0.03
tail_p90_worsening_c <= 0.03 in every fold with at least 40 activated rows
```

Failure demotes the specialist globally for the first strict implementation.

### 2.6 Exact specialist formulas

All feature names below refer to canonical feature keys defined in Section 6 of this addendum. If a feature is unavailable in a fold, use the missing handling rule above.

#### S1 — `S1_MARINE_SUPPRESSION`

Raw components and weights:

```text
0.20 * pct_high(gfs__center__onshore_east_component_mps_mean)
0.15 * pct_high(gefsmean__center__onshore_east_component_mps_mean)
0.20 * pct_high(gfs__spatial__inland_nw_minus_marine_s_tmax_c)
0.10 * pct_high(gfs__spatial__inland_nw_minus_center_tmax_c)
0.10 * pct_high(gfs__center__dewpoint_change_proxy_c)
0.10 * pct_high(gfs__center__low_cloud_pct_mean)
0.10 * pct_low(gfs__center__shortwave_sum_mj_m2)
0.05 * pct_high(official__forecast_max_minus_gefs_median_c)
```

Formula:

```text
marine_prior_score = weighted_sum_above / sum(non_missing_component_weights)
```

Detector target:

```text
marine_detector_target = 1 if marine_prior_score >= training_fold_p60_marine else 0
```

Correction target:

```text
marine_correction_target_c = base_residual_c
```

Expected useful sign:

```text
negative; median correction among candidate active rows must be <= -0.03 C
```

If the active-row median correction target is greater than `-0.03 C`, demote S1.

#### S2 — `S2_WEAK_WIND_HEAT`

Components:

```text
0.20 * pct_low(gfs__center__wind_speed_10m_mps_mean)
0.15 * pct_low(gefsmean__center__wind_speed_10m_mps_mean)
0.20 * pct_high(gfs__center__shortwave_sum_mj_m2)
0.10 * pct_low(gfs__center__low_cloud_pct_mean)
0.15 * pct_high(gfs__center__temperature_850_c_mean)
0.10 * pct_high(target__slope_7_30_c_per_day)
0.05 * pct_high(gfs__spatial__inland_nw_minus_center_tmax_c)
0.05 * pct_high(gfs__center__td_spread_at_tmax_c)
```

```text
heat_prior_score = weighted_sum / available_weight
heat_detector_target = 1 if heat_prior_score >= training_fold_p60_heat else 0
heat_correction_target_c = base_residual_c
expected sign = positive; candidate active median correction target must be >= +0.03 C
```

If the active-row median correction target is below `+0.03 C`, demote S2.

#### S3 — `S3_MAM_TRANSITION`

First apply calendar gate:

```text
mam_calendar_gate = 1 if month in {3,4,5} else 0
```

If `mam_calendar_gate=0`, then:

```text
mam_transition_prior_score = 0
score_available = true
activation = false
```

If `mam_calendar_gate=1`, components:

```text
0.20 * pct_high(abs(target__slope_7_30_c_per_day))
0.15 * pct_high(target__volatility_14_lag1_c)
0.15 * pct_high(abs(target__lag1_minus_roll7_c))
0.15 * pct_high(abs(official__forecast_max_minus_target_roll7_c))
0.10 * pct_high(gfs__center__low_cloud_pct_mean)
0.10 * pct_high(gfs__center__apcp_delta_mm)
0.10 * pct_high(gefsens__center__tmax_spread_p90_p10_c)
0.05 * pct_high(abs(gfs__center__tmax_c - gefsmean__center__tmax_c))
```

```text
mam_transition_prior_score = weighted_sum / available_weight
mam_detector_target = 1 if mam_calendar_gate=1 and prior_score >= training_fold_p60_mam_among_mam_rows else 0
mam_correction_target_c = base_residual_c
expected sign = learned; no sign demotion
```

Training for the MAM specialist uses only MAM rows, but it writes inactive rows for all months.

#### S4 — `S4_CLOUD_RAIN_SUPPRESSION`

Components:

```text
0.20 * pct_high(gfs__center__low_cloud_pct_mean)
0.15 * pct_high(gfs__center__apcp_delta_mm)
0.15 * pct_low(gfs__center__shortwave_sum_mj_m2)
0.15 * pct_high(gfs__center__relative_humidity_700_pct_mean)
0.10 * pct_high(gefsmean__center__pwat_kg_m2_mean)
0.10 * pct_high(gfs__center__dewpoint_2m_c_mean)
0.10 * pct_high(gefsens__center__tmax_spread_p90_p10_c)
0.05 * pct_high(official__psr_numeric_proxy)
```

```text
cloud_rain_prior_score = weighted_sum / available_weight
cloud_rain_detector_target = 1 if prior_score >= training_fold_p60_cloud_rain else 0
cloud_rain_correction_target_c = base_residual_c
expected sign = negative; candidate active median correction target must be <= -0.03 C
```

If `official__psr_numeric_proxy` is unavailable, it contributes neutral `0.50` and a missing flag.

#### S5 — `S5_DRY_RIDGE_HEAT`

Components:

```text
0.20 * pct_high(gfs__center__geopotential_height_500_m_mean)
0.15 * pct_high(gfs__center__temperature_850_c_mean)
0.15 * pct_low(gfs__center__relative_humidity_700_pct_mean)
0.15 * pct_high(gfs__center__shortwave_sum_mj_m2)
0.10 * pct_low(gfs__center__low_cloud_pct_mean)
0.10 * pct_low(gfs__center__apcp_delta_mm)
0.10 * pct_low(gfs__center__wind_speed_10m_mps_mean)
0.05 * pct_high(target__hot_spell_length_lag1_days)
```

```text
ridge_heat_prior_score = weighted_sum / available_weight
ridge_heat_detector_target = 1 if prior_score >= training_fold_p60_ridge else 0
ridge_heat_correction_target_c = base_residual_c
expected sign = positive; candidate active median correction target must be >= +0.03 C
```

#### S6 — `S6_HIGH_ERROR_TAIL`

Components:

```text
0.25 * pct_high(router__expert_prediction_spread_c)
0.20 * pct_high(gefsens__center__tmax_spread_p90_p10_c)
0.15 * pct_high(online__official_raw__global__abs_error_h20_c)
0.10 * pct_high(abs(official__forecast_max_minus_gfs_center_tmax_c))
0.10 * pct_high(abs(official__forecast_max_minus_gefs_median_c))
0.10 * pct_high(target__volatility_14_lag1_c)
0.10 * pct_high(router__missing_expert_count)
```

```text
high_error_tail_prior_score = weighted_sum / available_weight
high_error_tail_detector_target = 1 if prior_score >= training_fold_p60_tail else 0
high_error_tail_correction_target_c = base_residual_c
expected sign = learned; no sign demotion
```

S6 has an additional activation condition:

```text
router__expected_abs_error_c >= training_fold_p60_router_expected_abs_error
```

---

## 3. Shadow expert prediction behavior

### 3.1 Shared rules for all shadow experts

Shadow experts must always write rows to `model_oof.expert_prediction` or `model_live.live_prediction_component` for every snapshot in scope.

Unavailable placeholder row:

```text
is_available = false
prediction_tmax_c = NULL
prediction_type = 'placeholder_unavailable'
router_weight_cap = 0
promotion_status = 'shadow_unavailable'
```

Real shadow row:

```text
is_available = true
prediction_tmax_c = direct_raw_prediction_c
prediction_type = 'direct_untrained_shadow'
router_weight_cap = 0 before promotion
promotion_status = 'shadow_scored' or 'shadow_unscored_sealed'
```

No shadow expert receives nonzero strict router weight before explicit promotion.

Scoring with sealed labels:

```text
Shadow predictions may be stored before sealed labels are opened.
Shadow MAE/RMSE is not calculated until the sealed scoring command is run.
No shadow score may affect pre-2024 model selection.
```

### 3.2 Minimum valid-time requirements

Deterministic 12-point source real shadow prediction is allowed when:

```text
center location exists
at least 4 valid times exist for 6-hourly models or 8 valid times for 3-hourly models or 16 valid times for hourly models
all valid rows pass H24N safety filter
all required temperature fields are non-null for the center location
```

Full ensemble real shadow prediction is allowed when:

```text
center location exists
at least 80% of expected members exist
at least 4 valid times per available member exist
all rows pass H24N safety filter
```

### 3.3 Expert-specific direct prediction formulas

#### `E6_IFS_OPER_SHADOW`

Feature set:

```text
ifsoper__center__tmax_c
ifsoper__center__t2m_mean_day_c
ifsoper__center__dewpoint_2m_c_mean
ifsoper__center__wind_speed_10m_mps_mean
ifsoper__center__mslp_hpa_mean
ifsoper__center__precip_delta_mm
ifsoper__center__shortwave_sum_mj_m2
ifsoper__center__temperature_850_c_mean
ifsoper__center__geopotential_height_500_m_mean
ifsoper__spatial__max_tmax_c
ifsoper__spatial__min_tmax_c
ifsoper__spatial__range_tmax_c
```

Direct prediction:

```text
prediction_tmax_c = ifsoper__center__tmax_c
```

Promotion gate after sealed scoring:

```text
365 scored days minimum
MAE improvement over E0 on identical rows >= 0.03 C
P90 absolute error worsening <= 0.03 C
initial router cap after promotion = 0.10
```

#### `E7_IFS_ENS_SHADOW`

Feature set:

```text
ifsenfo__center__member_tmax_p10_c
ifsenfo__center__member_tmax_p25_c
ifsenfo__center__member_tmax_p50_c
ifsenfo__center__member_tmax_p75_c
ifsenfo__center__member_tmax_p90_c
ifsenfo__center__member_tmax_mean_c
ifsenfo__center__member_tmax_std_c
ifsenfo__center__member_count
```

Direct prediction:

```text
prediction_tmax_c = ifsenfo__center__member_tmax_p50_c
```

`ifsenfo` missing member 0 handling:

```text
If member 0 is missing and member_count >= 45, use all available members and set ifsenfo__center__member0_missing_flag = 1.
If member_count < 45, emit placeholder unavailable.
```

Promotion gate equals E6, with cap `0.10`.

#### `E8_AIFS_OPER_SHADOW`

Direct prediction:

```text
prediction_tmax_c = aifsoper__center__tmax_c
```

Required features are the same deterministic template as `ifsoper` where available. Initial cap after promotion is `0.05` because history starts in 2025.

#### `E8_AIFS_ENS_SHADOW`

Direct prediction:

```text
prediction_tmax_c = aifsenfo__center__member_tmax_p50_c
```

Real prediction requires `member_count >= 45`. Initial cap after promotion is `0.05`.

#### `E8_AIGFS_SFC_SHADOW`

Direct prediction:

```text
prediction_tmax_c = aigfssfc__center__tmax_c
```

`aigfssfc` has very short history. It remains shadow-only until 365 prospective scored days exist. Cap before that is always `0`.

#### `E8_GRAPHCAST_SHADOW`

Direct prediction:

```text
prediction_tmax_c = graphcast__center__tmax_c
```

Archive end rule:

```text
For target_date_hkt > 2026-05-06, emit placeholder unavailable.
```

GraphCast remains diagnostic shadow in the first full implementation. Router cap is `0` until a separate future gate approves historical archive usage.

#### `E8_FOURCASTNET_SHADOW`

Direct prediction:

```text
prediction_tmax_c = fourcastnetgfs__center__tmax_c
```

Archive end rule:

```text
For target_date_hkt > 2026-02-20, emit placeholder unavailable.
```

FourCastNet remains diagnostic shadow in the first full implementation. Router cap is `0`.

#### `E9_CWA_WRF_LIVE_SHADOW`

Direct prediction:

```text
prediction_tmax_c = cwawrf15__center__tmax_c
```

Real prediction is allowed only for prospectively collected rows with:

```text
availability_grade = 'PROSPECTIVE_EXACT_FIRST_SEEN'
first_seen_at_utc <= operational_freeze_utc
```

No historical CWA WRF backfill beyond the available rolling window may enter training. Router cap is `0` until at least 365 live scored days exist.

#### `E11_ARWF_LIVE_SHADOW`

Required source:

```text
public.hko_arwf_station_daily_forecasts or discovered table matching Section 4 source contract in the prior completion spec
```

Direct prediction:

```text
prediction_tmax_c = max(hourly forecast temperature for HKO target station over target date T HKT)
```

When only daily station maximum is stored:

```text
prediction_tmax_c = arwf_daily_forecast_max_c
```

Real prediction requires exact issue/retrieval time proof:

```text
published_at_hkt or retrieved_at_utc must be <= operational_freeze_utc
```

Without proof, emit placeholder unavailable. Router cap is `0` until 365 live scored days exist.

---

## 4. Inner validation and model selection

### 4.1 Outer fold definitions remain binding

Use the fold boundaries in the prior completion spec. This addendum defines only inner validation inside those outer training periods.

### 4.2 R0 inner validation

For every R0 outer fold training block:

```text
If training_rows >= 1095:
    inner_train = all outer-training rows except final 365 chronological days
    inner_validation = final 365 chronological days
Else if 730 <= training_rows < 1095:
    inner_train = first 70% chronological rows
    inner_validation = final 30% chronological rows
Else:
    no inner validation; use default hyperparameters
```

R0 inner validation is always last-block chronological, never random.

### 4.3 R1 inner validation

For every R1 outer fold training block:

```text
If training_rows >= 365:
    inner_train = all outer-training rows except final 90 chronological days
    inner_validation = final 90 chronological days
Else:
    no inner validation; use default hyperparameters
```

R1 Fold 1 normally has insufficient rows and must use defaults.

### 4.4 LightGBM early stopping

Use LightGBM only when the Python environment has `lightgbm` importable. If not importable, use `HistGradientBoostingRegressor` with the specified grid.

When LightGBM is used and an inner validation set exists:

```text
objective = regression_l1
metric = l1
num_boost_round = 2000
early_stopping_rounds = 50
seed = 20260626
```

When no inner validation exists:

```text
n_estimators = 200
learning_rate = 0.03
num_leaves = 15
min_child_samples = 40
subsample = 0.8
colsample_bytree = 0.8
reg_lambda = 5.0
```

### 4.5 Hyperparameter selection tie-breaking

Select the candidate with:

```text
1. lowest inner validation MAE;
2. if tied within 0.002 C, lowest inner validation P90 absolute error;
3. if still tied, lowest absolute bias;
4. if still tied, fewer model features;
5. if still tied, stronger regularization;
6. if still tied, smaller maximum correction cap;
7. if still tied, lexicographically smallest serialized hyperparameter JSON.
```

### 4.6 Required model-selection artifact

For every fitted model, write:

```text
artifacts/models/{expert_or_router_id}/{fold_id}/model_selection.json
```

Required fields:

```json
{
  "schema_version": "hkg_t24_model_selection_v1_final_20260626",
  "model_id": "...",
  "fold_id": "...",
  "outer_train_start": "YYYY-MM-DD",
  "outer_train_end": "YYYY-MM-DD",
  "outer_validation_start": "YYYY-MM-DD",
  "outer_validation_end": "YYYY-MM-DD",
  "inner_validation_used": true,
  "inner_train_start": "YYYY-MM-DD",
  "inner_train_end": "YYYY-MM-DD",
  "inner_validation_start": "YYYY-MM-DD",
  "inner_validation_end": "YYYY-MM-DD",
  "candidate_grid": [],
  "selected_hyperparameters": {},
  "selection_metrics": {
    "mae": 0.0,
    "p90_abs_error": 0.0,
    "bias": 0.0
  },
  "tie_break_trace": [],
  "random_seed": 20260626,
  "feature_schema_version": "hkg_t24_h24n_features_v1_final_20260626",
  "training_row_count": 0,
  "inner_validation_row_count": 0
}
```

---

## 5. Schema consistency fixes and final DDL changes

### 5.1 Dedicated columns required

`unit_semantics_verified` is required in `model_core.source_registry`. It must be a dedicated column, not only JSONB.

Apply this migration after creating schemas and before building features:

```sql
ALTER TABLE model_core.source_registry
  ADD COLUMN IF NOT EXISTS strict_allowed boolean NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS proxy_allowed boolean NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS shadow_allowed boolean NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS blocked boolean NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS blocked_reason text NULL,
  ADD COLUMN IF NOT EXISTS availability_grade text NOT NULL DEFAULT 'UNKNOWN',
  ADD COLUMN IF NOT EXISTS source_time_contract text NOT NULL DEFAULT 'UNKNOWN',
  ADD COLUMN IF NOT EXISTS publication_buffer_hours integer NOT NULL DEFAULT 6,
  ADD COLUMN IF NOT EXISTS unit_semantics_verified boolean NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS source_scope_filter_sql text NULL,
  ADD COLUMN IF NOT EXISTS first_usable_target_date date NULL,
  ADD COLUMN IF NOT EXISTS last_usable_target_date date NULL,
  ADD COLUMN IF NOT EXISTS training_partition text NOT NULL DEFAULT 'unassigned',
  ADD COLUMN IF NOT EXISTS promotion_status text NOT NULL DEFAULT 'unreviewed',
  ADD COLUMN IF NOT EXISTS source_inventory_sha256 text NULL,
  ADD COLUMN IF NOT EXISTS required_filter_jsonb jsonb NOT NULL DEFAULT '{}'::jsonb,
  ADD COLUMN IF NOT EXISTS schema_contract_jsonb jsonb NOT NULL DEFAULT '{}'::jsonb,
  ADD COLUMN IF NOT EXISTS notes text NULL;
```

Also apply:

```sql
ALTER TABLE model_features.feature_matrix
  ADD COLUMN IF NOT EXISTS feature_schema_version text NOT NULL DEFAULT 'hkg_t24_h24n_features_v1_final_20260626',
  ADD COLUMN IF NOT EXISTS feature_names text[] NULL,
  ADD COLUMN IF NOT EXISTS feature_missing_indicators_jsonb jsonb NOT NULL DEFAULT '{}'::jsonb,
  ADD COLUMN IF NOT EXISTS source_availability_jsonb jsonb NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE model_oof.expert_prediction
  ADD COLUMN IF NOT EXISTS promotion_status text NOT NULL DEFAULT 'unreviewed',
  ADD COLUMN IF NOT EXISTS router_weight_cap double precision NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS prediction_type text NOT NULL DEFAULT 'model_prediction',
  ADD COLUMN IF NOT EXISTS is_available boolean NOT NULL DEFAULT true,
  ADD COLUMN IF NOT EXISTS unavailable_reason text NULL,
  ADD COLUMN IF NOT EXISTS training_range_start date NULL,
  ADD COLUMN IF NOT EXISTS training_range_end date NULL,
  ADD COLUMN IF NOT EXISTS model_artifact_uri text NULL;

ALTER TABLE model_router.router_prediction
  ADD COLUMN IF NOT EXISTS promotion_status text NOT NULL DEFAULT 'unreviewed',
  ADD COLUMN IF NOT EXISTS demotion_reason text NULL,
  ADD COLUMN IF NOT EXISTS fallback_router_version text NULL,
  ADD COLUMN IF NOT EXISTS expert_mask_jsonb jsonb NOT NULL DEFAULT '{}'::jsonb,
  ADD COLUMN IF NOT EXISTS cap_trace_jsonb jsonb NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE model_router.specialist_prediction
  ADD COLUMN IF NOT EXISTS prior_score double precision NULL,
  ADD COLUMN IF NOT EXISTS score_available boolean NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS fold_p60 double precision NULL,
  ADD COLUMN IF NOT EXISTS expected_benefit_c double precision NULL,
  ADD COLUMN IF NOT EXISTS raw_correction_c double precision NULL,
  ADD COLUMN IF NOT EXISTS shrunk_correction_c double precision NULL,
  ADD COLUMN IF NOT EXISTS applied_correction_c double precision NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS activated boolean NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS activation_reason text NOT NULL DEFAULT 'not_evaluated',
  ADD COLUMN IF NOT EXISTS promotion_status text NOT NULL DEFAULT 'unreviewed';
```

### 5.2 JSONB policy

Dedicated columns are required for:

```text
source eligibility
source status
unit semantics
publication buffer
feature schema version
promotion status
availability masks
prediction type
```

JSONB is allowed only for:

```text
request provenance
feature dictionaries
cap traces
audit details
source-specific raw metadata
```

Codex must not hide eligibility or promotion status solely inside JSONB.

### 5.3 Compatibility behavior

All migrations must use `ADD COLUMN IF NOT EXISTS`. If a column exists with a conflicting type, stop and write:

```text
reports/schema_conflict_report.md
```

Then fail closed. Do not coerce existing incompatible columns.

---

## 6. Canonical feature matrix schema

### 6.1 Feature schema version

All feature matrices must store:

```text
feature_schema_version = hkg_t24_h24n_features_v1_final_20260626
```

### 6.2 Naming convention

Use lowercase snake-case keys with double underscores separating source groups:

```text
{source_prefix}__{location_or_scope}__{feature_name}_{unit_suffix}
```

Examples:

```text
gfs__center__tmax_c
gefsmean__marine_s_far__dewpoint_2m_c_mean
gefsens__center__member_tmax_p90_c
official__forecast_max_c
target__roll7_mean_lag1_c
online__official_raw__global__bias_h20_c
router__expert_prediction_spread_c
```

### 6.3 Strict feature whitelist

The strict feature whitelist for first full implementation is exactly these groups.

#### Identity columns, not model inputs

```text
snapshot_id
target_date
cutoff_id
season
month
day_of_year
is_mam
is_jja
is_son
is_djf
```

#### Official forecast features

```text
official__available_flag
official__forecast_min_c
official__forecast_max_c
official__forecast_range_c
official__forecast_midpoint_c
official__target_issue_lead_days
official__issue_hour_hkt
official__issue_minutes_before_cutoff
official__revision_count_pre_cutoff
official__first_pre_cutoff_max_c
official__latest_pre_cutoff_max_c
official__max_revision_c
official__abs_max_revision_c
official__forecast_max_minus_target_roll7_c
official__forecast_max_minus_target_roll30_c
official__forecast_max_minus_target_clim30_c
official__forecast_max_minus_gfs_center_tmax_c
official__forecast_max_minus_gefs_median_c
official__psr_numeric_proxy
official__text_hot_flag
official__text_very_hot_flag
official__text_showers_flag
official__text_thunderstorm_flag
official__text_cloudy_flag
official__text_fine_flag
official__text_mist_fog_flag
official__text_easterly_flag
official__text_light_wind_flag
```

#### Target-memory features

```text
target__lag1_tmax_c
target__lag2_tmax_c
target__lag3_tmax_c
target__lag7_tmax_c
target__lag14_tmax_c
target__lag30_tmax_c
target__lag60_tmax_c
target__lag365_tmax_c
target__roll3_mean_lag1_c
target__roll7_mean_lag1_c
target__roll14_mean_lag1_c
target__roll30_mean_lag1_c
target__roll60_mean_lag1_c
target__roll7_std_lag1_c
target__roll14_std_lag1_c
target__roll30_std_lag1_c
target__roll7_min_lag1_c
target__roll7_max_lag1_c
target__roll14_range_lag1_c
target__lag1_minus_roll7_c
target__lag1_minus_roll30_c
target__roll7_minus_roll30_c
target__slope_3_7_c_per_day
target__slope_7_30_c_per_day
target__volatility_14_lag1_c
target__hot_spell_length_lag1_days
target__cool_spell_length_lag1_days
target__clim30_mean_c
target__clim30_std_c
target__anomaly_vs_clim30_lag1_c
target__year_index
target__warming_trend_10y_c_per_year
```

#### Online state features

Use the exact names from Section 1.3 for these sources and scopes:

```text
official_raw/global
official_raw/source_era
official_raw/month
official_raw/season
gfs_mos/global
gefs_prob_mos/global
r0_router/global
r1_router/global
final_system/global
```

#### GFS deterministic NWP features

For each location in:

```text
center
local_n
local_s
local_e
local_w
local_ne
local_nw
local_se
local_sw
inland_nw_far
marine_s_far
marine_e_far
```

include:

```text
gfs__{loc}__available_flag
gfs__{loc}__tmax_c
gfs__{loc}__t2m_mean_day_c
gfs__{loc}__t2m_min_day_c
gfs__{loc}__t2m_max_11_18hkt_c
gfs__{loc}__t2m_08hkt_c
gfs__{loc}__t2m_11hkt_c
gfs__{loc}__t2m_14hkt_c
gfs__{loc}__t2m_17hkt_c
gfs__{loc}__dewpoint_2m_c_mean
gfs__{loc}__dewpoint_at_tmax_c
gfs__{loc}__td_spread_at_tmax_c
gfs__{loc}__u10_mps_mean
gfs__{loc}__v10_mps_mean
gfs__{loc}__wind_speed_10m_mps_mean
gfs__{loc}__onshore_east_component_mps_mean
gfs__{loc}__mslp_hpa_mean
gfs__{loc}__mslp_hpa_min
gfs__{loc}__low_cloud_pct_mean
gfs__{loc}__low_cloud_pct_max
gfs__{loc}__apcp_delta_mm
gfs__{loc}__shortwave_sum_mj_m2
gfs__{loc}__temperature_925_c_mean
gfs__{loc}__temperature_850_c_mean
gfs__{loc}__relative_humidity_700_pct_mean
gfs__{loc}__geopotential_height_500_m_mean
```

GFS spatial features:

```text
gfs__spatial__max_tmax_c
gfs__spatial__min_tmax_c
gfs__spatial__range_tmax_c
gfs__spatial__inland_nw_minus_center_tmax_c
gfs__spatial__inland_nw_minus_marine_s_tmax_c
gfs__spatial__center_minus_marine_s_tmax_c
gfs__spatial__local_n_minus_local_s_tmax_c
gfs__spatial__local_e_minus_local_w_tmax_c
gfs__spatial__max_dewpoint_mean_c
gfs__spatial__min_dewpoint_mean_c
gfs__spatial__dewpoint_range_c
gfs__spatial__mslp_range_hpa
```

#### GEFS ensemble mean features

For the same 12 locations:

```text
gefsmean__{loc}__available_flag
gefsmean__{loc}__tmax_c
gefsmean__{loc}__t2m_mean_day_c
gefsmean__{loc}__dewpoint_2m_c_mean
gefsmean__{loc}__relative_humidity_2m_pct_mean
gefsmean__{loc}__u10_mps_mean
gefsmean__{loc}__v10_mps_mean
gefsmean__{loc}__wind_speed_10m_mps_mean
gefsmean__{loc}__onshore_east_component_mps_mean
gefsmean__{loc}__mslp_hpa_mean
gefsmean__{loc}__pwat_kg_m2_mean
```

GEFS mean spatial features:

```text
gefsmean__spatial__max_tmax_c
gefsmean__spatial__min_tmax_c
gefsmean__spatial__range_tmax_c
gefsmean__spatial__inland_nw_minus_center_tmax_c
gefsmean__spatial__inland_nw_minus_marine_s_tmax_c
gefsmean__spatial__center_minus_marine_s_tmax_c
```

#### GEFS member features

```text
gefsens__center__available_flag
gefsens__center__member_count
gefsens__center__member_tmax_mean_c
gefsens__center__member_tmax_p10_c
gefsens__center__member_tmax_p25_c
gefsens__center__member_tmax_p50_c
gefsens__center__member_tmax_p75_c
gefsens__center__member_tmax_p90_c
gefsens__center__tmax_spread_p90_p10_c
gefsens__center__tmax_iqr_c
gefsens__center__tmax_std_c
gefsens__center__prob_ge_20_0
...
gefsens__center__prob_ge_40_0
```

The threshold sequence is every `0.5 C`; the explicit key format is defined in Section 11.3.

#### Router context features

```text
router__expert_prediction_spread_c
router__expert_prediction_std_c
router__missing_expert_count
router__available_expert_count
router__official_minus_gfs_c
router__official_minus_gefs_c
router__gfs_minus_gefs_c
router__expected_abs_error_c
```

### 6.4 Proxy feature whitelist

Proxy features are excluded from the strict first final formula unless separately promoted under proxy reporting. Allowed proxy feature groups:

```text
station_proxy__*
diagnostic_climate_lagged__*
igra_diagnostic_report__*
tc_diagnostic_report__*
```

`igra_diagnostic_report__*` and `tc_diagnostic_report__*` are report features only and must not enter matrices used for strict or proxy point forecasts in the first implementation.

### 6.5 Feature ordering

Feature export order is:

```text
1. identity columns
2. official__ keys in lexical order
3. target__ keys in lexical order
4. online__ keys in lexical order
5. gfs__ keys in lexical order
6. gefsmean__ keys in lexical order
7. gefsens__ keys in lexical order
8. router__ keys in lexical order
9. proxy keys in lexical order, only for proxy matrices
```

The exact ordered list must be written to:

```text
artifacts/feature_schema/hkg_t24_h24n_features_v1_final_20260626.json
reports/feature_dictionary_strict.csv
reports/feature_dictionary_proxy.csv
```

### 6.6 NULL and missingness rules

Raw feature tables preserve NULL.

Before model fitting, preprocessing is fold-local:

```text
For tree models: pass NaN directly where supported.
For linear models: impute using median from training fold only and add {feature}__is_missing.
For all models: source-level availability flags remain explicit input features.
```

Never fit imputers or scalers on validation, sealed, or future rows.

---

## 7. Static geospatial and station dossier behavior

### 7.1 First implementation decision

Station-network features are included only in the proxy scoreboard and as diagnostic context. They do not enter the strict first final forecast formula.

### 7.2 Metadata source priority

Station metadata must be loaded in this order:

```text
1. config/stations/hkg_station_dossier_v1.csv
2. public.station_metadata if it exists
3. distinct station_id/latitude/longitude/elevation_m from public.noaa_isd_core_observations or discovered ISD source table
```

If no metadata source satisfies the minimum contract, station features are unavailable and `E3_STATION_PROXY` emits placeholder unavailable rows.

### 7.3 Manual station dossier

Manual station dossier is allowed only at:

```text
config/stations/hkg_station_dossier_v1.csv
```

Required columns:

```text
station_id
station_name
latitude
longitude
elevation_m
station_source
station_role
role_confidence
notes
```

Allowed `station_role` values:

```text
target_hko
urban_core
airport_open
coastal
marine_island
inland_heat
hill_exposure
regional_synoptic
unknown_role
```

Rows with missing latitude or longitude are retained in the dossier but excluded from distance/gradient features.

### 7.4 Fallback station roles

When role metadata is missing:

```text
station_role = unknown_role
role_confidence = 0
```

Codex must not infer physical station roles from station IDs alone.

### 7.5 Coast distance

Coast distance is computed only when a static coastline geometry exists in PostGIS or a local vector file explicitly registered in `model_core.source_registry`.

Formula:

```sql
coast_distance_km = ST_Distance(station_geom::geography, nearest_coastline_geom::geography) / 1000.0
```

If coastline geometry is unavailable:

```text
coast_distance_km = NULL
coast_distance_available = false
```

No proxy coast distance is invented.

### 7.6 Minimum station requirements

Station proxy features are built only when:

```text
at least 5 stations have usable temperature before cutoff
at least 3 stations have usable dewpoint before cutoff for dewpoint features
at least 3 stations have usable pressure before cutoff for pressure features
wind-direction features are disabled unless repaired wind direction passes validation
```

If fewer stations exist for a feature family, that family is NULL with an availability flag set to false.

---

## 8. Diagnostic proxy behavior

### 8.1 First implementation decision

`E10_DIAGNOSTIC_PROXY` is not trained for the strict first full implementation. It must emit placeholder unavailable rows in `model_oof.expert_prediction` with:

```text
is_available = false
prediction_tmax_c = NULL
router_weight_cap = 0
promotion_status = diagnostic_report_only
```

Diagnostic feature tables and reports are still required.

### 8.2 Allowed lagged HKO daily climate variables

Allowed diagnostic climate variables from `02_hko_daily_climate_all_elements`:

```text
mean_temperature
mean_sea_level_pressure
daily_rainfall
mean_relative_humidity
mean_wet_bulb_temperature
mean_cloud_amount
bright_sunshine_duration
mean_dew_point_temperature
evaporation
global_solar_radiation
mean_wind_speed
prevailing_wind_direction
sea_temperature
sea_temperature_am
sea_temperature_pm
reduced_visibility_hours
cloud_to_cloud_lightning
cloud_to_ground_lightning
```

Forbidden as predictor:

```text
daily_maximum_temperature for target date T
daily_minimum_temperature for target date T
mean_temperature for target date T
any climate value with local_date >= T-1
```

### 8.3 Safe lag rule for diagnostic climate features

Because first-publication timing is unproven, only values with:

```text
local_date <= T-2
```

may enter diagnostic proxy feature tables. These features remain proxy-only and do not enter strict final forecast.

### 8.4 Climate long-to-wide formulas

For each allowed climate variable `v`, build:

```text
diagnostic_climate_lagged__{v}__lag2
diagnostic_climate_lagged__{v}__lag3
diagnostic_climate_lagged__{v}__lag7
diagnostic_climate_lagged__{v}__roll7_mean_lag2
diagnostic_climate_lagged__{v}__roll14_mean_lag2
diagnostic_climate_lagged__{v}__roll30_mean_lag2
diagnostic_climate_lagged__{v}__roll7_std_lag2
diagnostic_climate_lagged__{v}__lag2_minus_roll14_lag2
```

Roll windows end at `T-2` inclusive.

`Trace` treatment:

```text
rainfall Trace -> value_mm = 0.0 and trace_flag = true
other Trace -> value = NULL and trace_flag = true
```

### 8.5 IGRA and TC best-track diagnostics

IGRA and TC best-track do not feed any first implementation expert.

Required reports:

```text
reports/diagnostic_igra_mechanism_report.md
reports/diagnostic_tc_regime_report.md
```

They must summarize coverage, timestamp blocker, top diagnostic correlations with target/residual on pre-2024 rows, and safe proxy suggestions. No IGRA or TC feature may appear in strict or proxy model matrices.

---

## 9. Official residual model and fallback behavior

### 9.1 E1 failure behavior

If `E1_OFFICIAL_RESIDUAL` fails promotion:

```text
1. E1 predictions remain written for diagnostics.
2. E1 router_weight_cap = 0 in all strict routers.
3. E1 may appear in proxy reports with promotion_status = demoted_diagnostic_only.
4. R0 and R1 are retrained or reweighted without E1 for final strict candidate selection.
```

### 9.2 Router fallback when E1 fails

`R0_OFFICIAL_LONG_HISTORY` candidate set becomes:

```text
E0_OFFICIAL_RAW
E2_TARGET_MEMORY
```

`R1_CORE_GFS_GEFS` candidate set becomes:

```text
E0_OFFICIAL_RAW
E2_TARGET_MEMORY
E4_GFS_MOS if promoted
E5_GEFS_PROB_MOS if promoted
```

### 9.3 Final frozen candidate behavior when core experts fail

Binding fallback ladder:

```text
If R1 promoted: final strict candidate = R1 + promoted strict specialists + distribution layer if promoted.
Else if R0 promoted: final strict candidate = R0 + promoted strict specialists + distribution layer if promoted.
Else if E0_OFFICIAL_RAW available: final strict candidate = E0_OFFICIAL_RAW.
Else if E2_TARGET_MEMORY promoted and E0 unavailable: final strict candidate = E2_TARGET_MEMORY.
Else no strict prediction is produced; fail closed.
```

When E4 or E5 fails promotion:

```text
E4/E5 predictions remain in diagnostics.
Router cap for failed expert = 0.
R1 may still be promoted with the remaining promoted NWP expert, but R1 must beat R0 and E0 on identical rows.
If both E4 and E5 fail, R1 is unavailable.
```

### 9.4 Definition of best safe baseline

For row-level comparison:

```text
best_safe_baseline = E0_OFFICIAL_RAW when official anchor is available
best_safe_baseline = E2_TARGET_MEMORY only on rows where official anchor is unavailable
```

`R0` is a candidate system, not the baseline definition. `E2` is not the baseline when `E0` exists.

---

## 10. Router edge cases

### 10.1 R1 unavailable but R0 available

Use R0 output. Write a final system audit entry:

```text
router_selected = R0_OFFICIAL_LONG_HISTORY
router_selection_reason = R1_unavailable_for_date
```

### 10.2 E1 demoted but R1 otherwise available

R1 is built from promoted experts only. When E1 is demoted, it is removed from:

```text
static weight optimization
dynamic expected-error modelling
weight normalization
final router output
```

The R1 artifact must list:

```text
excluded_experts = ['E1_OFFICIAL_RESIDUAL']
exclusion_reason = 'failed_promotion'
```

### 10.3 Static blend candidate set after demotion

For each router version, static blend candidate set is exactly:

```text
promoted experts with router_weight_cap > 0 plus E0_OFFICIAL_RAW if available
```

E0 is always eligible for static blend when present, even though it is not “promoted” by a model-training gate.

### 10.4 Cap redistribution edge case

After applying caps and availability masks:

```text
If total_allowed_weight > 0:
    renormalize allowed weights to sum to 1.
Else if E0 available:
    E0 weight = 1.
Else if E2 promoted and available:
    E2 weight = 1.
Else:
    no forecast; fail closed.
```

### 10.5 Demoted router artifact schema

Every router attempt writes a row to `model_router.router_scoreboard` with:

```text
router_version
router_scope
promotion_status
promotion_gate_passed boolean
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

Every demoted router also writes prediction rows using the fallback selected by the fallback ladder with:

```text
promotion_status = demoted_fallback_used
fallback_router_version = selected fallback
```

### 10.6 R1 promotion requirement

R1 must beat both:

```text
E0_OFFICIAL_RAW on identical R1 rows
R0_OFFICIAL_LONG_HISTORY on identical R1 rows when R0 is available on those rows
```

Required improvement:

```text
MAE improvement >= 0.01 C versus the better of E0 and R0
P90 absolute error worsening <= 0.02 C
bias absolute value <= baseline bias absolute value + 0.03 C
```

---

## 11. Distributional layer edge cases

### 11.1 Quantile model failure

If any quantile model fails training, monotonicity, or validation:

```text
distributional_promotion_status = demoted_empirical_fallback
p50_c = final_pre_distribution_c
```

### 11.2 Empirical fallback intervals

For each month `m`, compute training residuals from the promoted point system:

```text
residual_final_c = target_tmax_c - final_pre_distribution_c
```

If month `m` has at least `100` training residuals, use month-specific quantiles. Otherwise use all-season global quantiles.

Fallback intervals:

```text
p10_c = p50_c + quantile(residual_final_c, 0.10)
p25_c = p50_c + quantile(residual_final_c, 0.25)
p75_c = p50_c + quantile(residual_final_c, 0.75)
p90_c = p50_c + quantile(residual_final_c, 0.90)
```

Enforce monotonicity:

```text
p10 <= p25 <= p50 <= p75 <= p90
```

If monotonicity fails due to degenerate residuals:

```text
p10 = p50 - 1.20
p25 = p50 - 0.60
p75 = p50 + 0.60
p90 = p50 + 1.20
```

### 11.3 Threshold probability keys

Produce all threshold probability keys from `20.0` through `40.0` inclusive in `0.5 C` increments.

Key format:

```text
prob_ge_20_0
prob_ge_20_5
prob_ge_21_0
...
prob_ge_39_5
prob_ge_40_0
```

There must be exactly `41` threshold keys.

### 11.4 Probability fallback

When quantile models are demoted, probabilities are still produced.

Use Gaussian fallback:

```text
sigma_c = max(expected_abs_error_c * sqrt(pi / 2), 0.60)
prob_ge_x = 1 - normal_cdf((x - p50_c) / sigma_c)
```

Fallback `expected_abs_error_c`:

```text
If router expected_abs_error_c exists: use it.
Else use median(abs(residual_final_c)) from training residuals.
Else use 0.90.
```

Clamp probabilities to `[0.001, 0.999]`.

### 11.5 Calibration report requirements

Write:

```text
reports/distribution_calibration_report.csv
reports/distribution_calibration_report.md
```

Required metrics:

```text
pinball_loss_p10
pinball_loss_p25
pinball_loss_p50
pinball_loss_p75
pinball_loss_p90
empirical_coverage_p10_p90
empirical_coverage_p25_p75
mean_predicted_interval_width_p10_p90
brier_score_prob_ge_30_0
brier_score_prob_ge_31_0
brier_score_prob_ge_32_0
brier_score_prob_ge_33_0
brier_score_prob_ge_34_0
calibration_bin_count
monotonicity_violation_count
```

---

## 12. Sealed validation and shadow sources

### 12.1 2024 strict opening sequence

Before opening 2024 labels, freeze:

```text
feature schema
expert artifacts
router artifacts
specialist artifacts
distribution artifacts
final strict candidate manifest
negative-control report
```

Command:

```bash
python -m hkg_tmax.cli freeze-pre2024-candidate --cutoff-id H24N
```

Then score strict candidate and shadow predictions on 2024 in one command:

```bash
python -m hkg_tmax.cli sealed-score-2024 \
  --cutoff-id H24N \
  --candidate-manifest artifacts/frozen/final_strict_candidate_pre2024.json \
  --score-shadows true
```

This command may compute shadow scores but may not train adapters.

### 12.2 2024 pass condition for opening adapter training

The frozen strict candidate passes 2024 only when:

```text
MAE <= E0_OFFICIAL_RAW_MAE - 0.01 C on identical 2024 rows
P90 absolute error worsening <= 0.03 C versus E0
negative controls already passed pre-2024
no sealed-access violation is detected
```

If this pass condition fails, no IFS/AI adapter training is allowed.

### 12.3 Adapter training after 2024 pass

After strict 2024 pass, run:

```bash
python -m hkg_tmax.cli train-2024-shadow-adapters \
  --cutoff-id H24N \
  --opened-year 2024 \
  --frozen-grid artifacts/config/shadow_adapter_grid_v1.json
```

Allowed training labels:

```text
2024 labels only, plus pre-2024 labels for core experts already frozen/refit under unchanged rules.
```

Adapter hyperparameter grid must be fixed before the command runs. It is:

```json
{
  "adapter_model_class": ["HuberRegressor"],
  "alpha": [0.01, 0.05, 0.10],
  "epsilon": [1.35],
  "max_weight_cap": [0.05, 0.10],
  "correction_cap_c": [0.20, 0.30]
}
```

### 12.4 Conditions for IFS/AI entry into refit-through-2024 candidate

A shadow source can enter the refit-through-2024 candidate only when all are true on 2024:

```text
source has at least 250 scored 2024 rows
shadow direct MAE beats E0 by >= 0.03 C or adapter-corrected MAE beats E0 by >= 0.03 C
P90 absolute error worsening <= 0.03 C
absolute bias <= E0 absolute bias + 0.05 C
negative controls pass for adapter features
source availability status is strict or shadow with documented run-time/proxy grade
```

Initial router cap after 2024 entry:

```text
IFS deterministic: 0.10
IFS ensemble: 0.10
AIFS deterministic: 0.05
AIFS ensemble: 0.05
GraphCast/FourCastNet: 0.00 in strict live candidate, diagnostics only
AIGFS/AIGEFS: 0.00 until 365 prospective scored days
```

### 12.5 2025 final test contamination prevention

After 2024 adapter training, freeze:

```bash
python -m hkg_tmax.cli freeze-refit-through-2024-candidate --cutoff-id H24N
```

Then open 2025 once:

```bash
python -m hkg_tmax.cli sealed-score-2025 \
  --cutoff-id H24N \
  --candidate-manifest artifacts/frozen/final_candidate_refit_through_2024.json \
  --score-shadows true
```

Forbidden after seeing 2025:

```text
changing feature schema
changing hyperparameter grids
changing adapter caps
changing router tau/lambda grid
changing specialist activation thresholds
removing bad 2025 slices from scoreboards
retraining on 2025 labels before publishing final 2025 report
```

2025 shadow scores may be reported only. They do not authorize additional changes in the final historical test report.

---

## 13. Test expectations

### 13.1 Test data modes

There are two test modes:

```text
synthetic_unit: uses generated fixtures and does not require real source tables
real_db_smoke: requires PostgreSQL with project source tables
```

Unit tests must not require live GribStream API access.

### 13.2 Synthetic fixtures

Create:

```text
tests/fixtures/synthetic_h24n/create_synthetic_fixture.py
```

The fixture must generate 120 target dates with:

```text
HKO labels
official forecasts
GFS rows
GEFS mean rows
GEFS member rows
minimal source_registry rows
minimal cutoff calendar
```

Expected synthetic properties:

```text
120 snapshots
119 target-memory lag1 rows available after first label
at least 100 official anchors
at least 90 GFS daily feature rows
at least 90 GEFS daily feature rows
no post-cutoff rows
```

### 13.3 Real database smoke test

`run-full-pre2024 --smoke` must use real DB data, not synthetic fixtures.

Command:

```bash
python -m hkg_tmax.cli run-full-pre2024 \
  --cutoff-id H24N \
  --smoke \
  --from-date 2021-04-14 \
  --to-date 2021-05-31
```

Expected minimum real DB counts:

```text
snapshots >= 45
official anchors >= 45
target labels >= 45
gfs daily feature rows >= 40
gefs mean daily feature rows >= 40
gefs member feature rows >= 40
feature matrix rows >= 40
E0 predictions >= 40
E2 predictions >= 30
```

OOF model training may use default hyperparameters in smoke mode. The smoke command must not open or access 2024+ labels.

### 13.4 PostgreSQL requirements

Tests marked `real_db` require:

```text
HKG_TMAX_DATABASE_URL
```

If the environment variable is absent, real DB tests are skipped with reason:

```text
SKIPPED_REAL_DB_NO_DATABASE_URL
```

Synthetic tests must still run.

### 13.5 Required test files

Create these tests:

```text
tests/unit/test_h24n_cutoff_time.py
tests/unit/test_online_state_formulas.py
tests/unit/test_specialist_prior_scores.py
tests/unit/test_nwp_unit_conversions.py
tests/unit/test_threshold_probability_keys.py
tests/unit/test_router_weight_math.py
tests/unit/test_feature_schema_order.py
tests/integration/test_snapshot_builder_synthetic.py
tests/integration/test_oof_no_current_label_synthetic.py
tests/integration/test_negative_controls_synthetic.py
tests/integration/test_schema_migrations_realdb.py
tests/integration/test_realdb_smoke_h24n.py
```

### 13.6 Exact expected results

The test suite passes only when:

```text
all synthetic tests pass
schema migrations are idempotent
H24N UTC conversion equals 07:00 UTC on T-1
online state for T excludes residual from T
post-cutoff NWP rows are rejected
feature schema order is deterministic
router weights sum to 1 after masks and caps
threshold probability count is exactly 41
negative controls fail to beat real model gates
real DB smoke minimum counts pass when DB is available
```

---

## 14. Final Codex Implementation Readiness

YES — Codex should now implement exactly this specification without making design decisions.

Codex must treat this addendum and the prior completion specification as a complete implementation contract. Where this addendum is more specific, this addendum is binding. The first full implementation is now fully specified with conservative defaults for online residual states, specialist formulas, shadow expert behavior, inner validation, schema migrations, feature naming, station metadata, diagnostic proxies, fallback rules, router edge cases, distributional fallback, sealed validation, and executable tests.
