# Codex Implementation Task: Leakage-Free HKG Daily Tmax ML Strategy for Maximum MAE Improvement

Created: 2026-07-05

This task is for one exact problem: predict the Hong Kong Observatory Headquarters daily absolute maximum temperature for target local date `T`, scored against the HKO Daily Extract field `Absolute Daily Max (deg. C)`. The objective is point-forecast accuracy, with MAE first, RMSE second, and p90 absolute error third. This is not a market-execution, UI, deployment, bankroll, or generic repository task.

The attached context documents are the source of truth for the local database schema, target definition, source coverage, cutoff constraints, and leakage rules. Public web checks were used only to verify the public source families: HKO Climatological Information Services / Daily Extract, Info.gov weather press releases, HKO local forecast update schedule, and HKO OpenData forecast JSON as a secondary cross-check. They do not override the attached DB-context documents.

---

## 1. Final Strategy Decision

### Primary design

Build a **calibrated official-forecast residual ensemble**:

```text
base_forecast(T, C) = latest eligible Info.gov LOCAL WEATHER FORECAST max for target date T available by cutoff C
residual_target(T) = actual_hko_daily_extract_tmax_c(T) - base_forecast(T, C)
model_residual_hat(T, C) = compact leakage-safe ML residual estimate
final_prediction(T, C) = base_forecast(T, C) + shrunk_residual_hat(T, C)
```

The primary production profile is:

```text
target_date = local HKT day T
cutoff_profile = tminus1_2359
cutoff_at_hkt = T - 1 day at 23:59:00 Asia/Hong_Kong
cutoff_at_utc = T - 1 day at 15:59:00 UTC
anchor_source = public.hko_historical_forecasts_2000_2026 / Info.gov LOCAL WEATHER FORECAST
anchor_row = latest eligible usable_local_minmax lead-1 forecast row by issue_at_utc <= cutoff_at_utc
model_target = actual_tmax_c - anchor_forecast_max_c
prediction = anchor_forecast_max_c + no-harm-shrunk ensemble residual
```

### Why residual modeling, not direct Tmax modeling

Use the official local forecast maximum as the anchor because it already compresses a large amount of meteorological analysis into a small, stable signal. The documented prior experiment shows that the raw official latest forecast at `T-1 23:59 HKT` already achieves about `0.9275 C` MAE on the 2011-2023 official-row-only evaluation set, and a grouped residual correction only improves this to about `0.9216 C`. That means the official forecast is hard to beat and naive feature additions are unlikely to move MAE materially.

Direct absolute-Tmax modeling should still be implemented as a diagnostic baseline, but it should not be the main strategy. A direct model can overfit seasonality and the official forecast level; a residual model focuses learning capacity on the only part that matters: when and why the official HKO local forecast misses the HKO Headquarters station daily max.

### Why not a single black-box model

The final system should be an ensemble of deliberately different residual estimators:

1. **Grouped empirical-Bayes residual correction**: the strongest transparent baseline, reproducing and extending the previous `B3_grouped_residual_shrinkage` idea.
2. **LightGBM MAE residual model**: main nonlinear learner for compact tabular interactions among official forecast, revisions, hourly state, station gradients, warning regimes, and seasonality.
3. **CatBoost MAE residual model**: second nonlinear learner with strong handling of categorical buckets, missing values, and small-to-medium tabular data.
4. **Huber or quantile linear residual model**: guardrail learner that captures stable additive biases and avoids tree overreaction.
5. **Zero-correction option**: included explicitly in the ensemble so the optimizer can choose to do almost nothing when residual features do not beat the official anchor.

The blend must be nonnegative, validation-chosen, and shrunk toward zero correction. A feature-rich model is only promoted when it beats the raw official forecast on honest pre-2024 validation without worsening RMSE or p90 tail error.

### Primary expected edge source

The official forecast itself is already strong. The only credible path to a large MAE improvement is not generic model complexity; it is a compact feature matrix that captures station-local residual mechanisms the local forecast text does not fully encode:

- late forecast revision path and forecast-confidence information;
- HKO target-station thermal and humidity state before cutoff;
- coastal versus inland station-network gradients;
- urban-core versus maritime contrast;
- thunderstorm, rain, lightning, and tropical-cyclone regimes;
- target-memory climatology and modern-era anomalies;
- month/season-specific official forecast bias;
- missingness and abnormal-dispatch behavior as weather-regime signals.

### Phase-one versus phase-two decision

Phase one should build the strongest compact residual system quickly and honestly. It should not build a 1,000-column feature warehouse. The phase-one feature budget should be approximately 80-140 columns after one-hot/categorical handling, with every feature family tied to a physical or source-structure rationale.

Phase two should only begin after phase-one ablations prove what actually moves MAE. Phase two may add regime experts, all-station sparse temporal features, cyclone-distance geometry, or alternative cutoffs. It must not add external NWP, NOAA/ISD, IGRA, daily all-elements climate variables, or the HKO 9-day API as primary predictors until point-in-time availability and source comparability are proven.

---

## 2. Data Sources To Use

### Source table policy

Use only documented, source-compatible, cutoff-eligible data. The canonical target label is never a predictor for the target date. Raw Daily Extract payload rows are audit-only. The official forecast anchor must remain the Info.gov `LOCAL WEATHER FORECAST` source family because the historical archive and live source contract are built around that source.

### Required source table map

| Source | Role | Join key | As-of key | Coverage from docs | Required filters | Null handling | Phase |
|---|---:|---|---|---|---|---|---|
| `label_core.hko_daily_tmax` | Pre-2024 supervised labels and pre-2024 evaluation truth | `local_date = target_date` | Not a predictor; label known only after publication | `1884-01-01` to `2023-12-31`; 48,577 rows; no null targets | `local_date < '2024-01-01'`, `quality_status='VALID'`, station/source as documented | Drop if target null; current null count is zero | Core label source |
| `sealed_confirmation.hko_daily_tmax` | 2024+ locked confirmation labels | `local_date = target_date` | Not a predictor except strictly online lag features after freeze | `2024-01-01` to `2026-05-31`; 882 rows; no null targets in current DB | Use only after model, features, cutoff, hyperparameters, imputation, calibration, and weights are frozen | Drop if target null; current null count is zero | Locked final confirmation only |
| `feature_safe.hko_target_history_pre2024` | Target-history lag and climatology features | `local_date <= target_date - 2 days`, unless row-level publication availability proves a shorter lag | Predictor availability is governed by target label availability at cutoff; default safe lag starts at 2 days | `1884-01-01` to `2023-12-31`; excludes sealed rows by definition | Never include `target_date`, never include lag 1 by default, never compute all-history climatology | Null if insufficient past observations; add missing flags | Core feature source |
| `raw_audit.codex_audit_ds_01_hko_daily_tmax_target_hko_daily_extr_23cb54da` | Audit/provenance only | None for modeling | Not eligible | Raw payload through `2026-06-17`, but includes duplicate monthly/yearly payload rows and one failed row | Forbidden as predictor; only use in leakage tests and provenance audits | Not applicable | Rejected for model predictors |
| `public.hko_historical_forecasts_2000_2026` | Official forecast anchor, forecast revisions, latest forecast text | `target_date` | `issue_at_utc <= cutoff_at_utc` | Strict usable local min/max subset covers `2000-01-02` to `2026-06-21`; lead-1 subset has 88,504 rows | `source='info_gov'`, `product_type='local'`, `row_quality_status='usable_local_minmax'`, `target_issue_lead_days=1`, non-null `forecast_max_c`, `forecast_min_c`, `issue_at_hkt`, `target_date` | If no eligible latest row, row is not scorable for official-anchor models; do not fallback to 9-day API | Core anchor and revision source |
| `public.hko_info_gov_hourly_readings_1998_2026` | HKO target-station state, relative humidity, neighbor-station state, warning/rain/lightning/tropical-cyclone context | Windowed by `dispatch_at_utc <= cutoff_at_utc`, not a simple date join | `dispatch_at_utc` and `available_at_utc`; also require `observation_at_utc <= cutoff_at_utc` when present | 268,894 rows from `1998-05-04` to `2026-07-04`; target-station present in 268,861 rows; 27 station names | `parse_status in ('parsed','partial')`; require dispatch before cutoff; station JSON values handled sparsely | Do not drop target rows for hourly missingness; impute/model-missing with flags; invalid station outliers become null + outlier flag | Core feature source |
| HKO OpenData `flw` local weather forecast JSON | Live cross-check only | Live issue/update time | Update time, not primary archive source | Current public endpoint only, not historical primary table | May be logged as secondary confirmation of Info.gov local forecast text | Never substitute silently for Info.gov anchor | Optional diagnostic only |
| HKO OpenData `fnd` 9-day forecast JSON | Diagnostic only | Forecast date | Update time | Current public endpoint only | Do not use as official anchor for this strategy | No model use in phase one | Rejected for phase-one anchor |
| NOAA/ISD and IGRA diagnostic tables mentioned in target context | Future diagnostic research only | Station/time | Release timing unresolved | Long historical diagnostic coverage but not settlement truth | Do not use in phase one | Not applicable | Rejected for phase one |
| HKO daily climate all-elements dataset | Lagged diagnostic research only | Date | Publication timing unresolved | Broad climate data through current target coverage | Do not use as live predictor until publication timing and quality are proven | Not applicable | Rejected for phase one |

### Forecast source eligibility details

A forecast row is eligible only when all of these hold:

```sql
source = 'info_gov'
AND product_type = 'local'
AND row_quality_status = 'usable_local_minmax'
AND target_issue_lead_days = 1
AND target_date = :target_date
AND forecast_max_c IS NOT NULL
AND forecast_min_c IS NOT NULL
AND issue_at_utc IS NOT NULL
AND issue_at_utc <= :cutoff_at_utc
```

Tie handling:

1. Deduplicate exact duplicate source pages by `raw_sha256` where possible.
2. If multiple rows share the same `issue_at_utc`, prefer the row with a canonical Info.gov `source_url` and valid `raw_sha256`.
3. If still tied, choose deterministic lexicographic order by `source_url`; emit `forecast_tie_flag=1`.
4. Never choose a row after cutoff.
5. Never fallback to `fnd`, weather widgets, airport forecasts, or manually interpreted New Territories add-ons.

### Hourly readings eligibility details

An hourly reading row is eligible only when:

```sql
parse_status IN ('parsed', 'partial')
AND dispatch_at_utc <= :cutoff_at_utc
AND available_at_utc <= :cutoff_at_utc
AND (observation_at_utc IS NULL OR observation_at_utc <= :cutoff_at_utc)
```

For target-day `T` under all pre-target-day cutoff profiles, target-day readings must be absent by construction because the cutoff is before `T 00:00 HKT`. The implementation must still test this explicitly by verifying:

```text
max(observation_at_hkt) <= cutoff_at_hkt
max(dispatch_at_hkt) <= cutoff_at_hkt
```

Do not use `retrieved_at_utc` as historical availability; it is a backfill retrieval timestamp.

### Target-history eligibility details

Default safe target-history features may use only:

```text
local_date <= target_date - 2 days
```

This is conservative and intentional. For a `T-1` cutoff, the HKO Daily Extract value for `T-1` is not safe unless row-level first-publication evidence proves it was available before the cutoff. The public HKO climate page says climatological data are updated every working day before 2 p.m. up to the previous day, but that is not enough to treat lag 1 as universally safe across weekends, holidays, and historical periods. Therefore:

```text
default lag floor = lag2
lag1 = excluded from phase-one primary model
lag1_candidate = phase-two only, gated by exact publication-time proof
```

For 2024+ locked confirmation evaluation, produce two target-history modes:

1. **sealed_blind_mode**: target-history features never use `sealed_confirmation` rows. This is the strictest no-sealed-input audit.
2. **online_live_replay_mode**: after the entire model has been frozen, lagged 2024+ labels may be used for later 2024+ predictions only when their dates are strictly before the target date and their publication availability is proven before cutoff. This mimics live operation but must be reported separately from sealed_blind_mode.

The headline sealed confirmation score must state which mode was used.

---

## 3. Cutoff And Issuance Policy

### Default cutoff decision

Use `T-1 23:59:00 HKT` as the primary MAE-maximizing cutoff profile.

Reason: the problem statement asks for maximum point-forecast accuracy, not early-day execution. `T-1 23:59 HKT` is still strictly pre-target-day and therefore avoids target-day observation leakage, but it captures the latest same-source official forecast and the most complete pre-target-day HKO/station-network state. The prior documented baseline was also evaluated at this cutoff, making it the cleanest comparison against the current `~0.92 C` MAE practical baseline.

If a separate trading workflow requires `T-1 15:00 HKT`, implement that as a secondary profile using the same code path, not as the primary accuracy benchmark.

### Cutoff profiles to implement

| Profile | HKT cutoff for target date `T` | UTC cutoff logic | Purpose | Primary? |
|---|---:|---:|---|---:|
| `tminus1_1500` | `T - 1 day 15:00:00 Asia/Hong_Kong` | `T - 1 day 07:00:00 UTC` | Active T-24-style conservative profile; earlier operational decision | Secondary |
| `tminus1_1800` | `T - 1 day 18:00:00 Asia/Hong_Kong` | `T - 1 day 10:00:00 UTC` | Captures late-afternoon forecast and observed afternoon state | Sensitivity |
| `tminus1_2100` | `T - 1 day 21:00:00 Asia/Hong_Kong` | `T - 1 day 13:00:00 UTC` | Captures evening forecast/revision path and early evening cooling | Sensitivity |
| `tminus1_2359` | `T - 1 day 23:59:00 Asia/Hong_Kong` | `T - 1 day 15:59:00 UTC` | Maximal pre-target-day information; default MAE benchmark | Primary |

HKT is UTC+8 with no DST. Do not use Stockholm time inside model code except for human-facing logging. Store all matrix rows with both:

```text
cutoff_at_hkt
cutoff_at_utc
```

### Cutoff sensitivity experiment

Run all four profiles on pre-2024 data only. Do not use 2024+ sealed confirmation to choose a cutoff.

Selection rule:

1. The primary reported model is `tminus1_2359` unless another cutoff beats it by at least `0.020 C` MAE on the pre-sealed 2022-2023 holdout and does not worsen RMSE or p90 absolute error by more than `0.020 C`.
2. If `tminus1_2359` beats earlier cutoffs materially, keep it as default.
3. If `tminus1_1500` is required operationally, report it as a separate scorecard and do not compare it to the `23:59` profile without stating the different information set.

### Latest eligible forecast selection

For each `(target_date, cutoff_profile)`:

1. Fetch all forecast rows satisfying the strict forecast eligibility filter.
2. Keep rows where `issue_at_utc <= cutoff_at_utc`.
3. Sort by `issue_at_utc`, then deterministic tie-breaker.
4. The final row is `anchor_forecast_row`.
5. The previous eligible row is `prev_forecast_row`.
6. All eligible rows form the revision path.

Required forecast selector outputs per scored row:

```text
target_date
cutoff_profile
cutoff_at_hkt
cutoff_at_utc
anchor_source_url
anchor_issue_at_hkt
anchor_issue_at_utc
anchor_forecast_min_c
anchor_forecast_max_c
anchor_forecast_range_c
anchor_raw_sha256
eligible_forecast_count
forecast_selector_status
```

If no eligible row exists, set `forecast_selector_status='no_eligible_anchor'` and exclude the row from official-anchor scoring. Do not impute official forecast max for primary evaluation.

### Forecast-revision construction

Use only rows eligible before cutoff. Construct revision features from the sorted sequence:

```text
R_1, R_2, ..., R_n where R_n = latest eligible anchor row
```

Never include an issue after cutoff. Never include lead-0 forecasts in the primary pre-target-day model. Lead-0 rows may be used only in a separately named intraday experiment with its own cutoff and leakage tests.

### Hourly readings cutoff policy

For each target date and cutoff:

1. Select all hourly readings where `dispatch_at_utc <= cutoff_at_utc` and `observation_at_utc <= cutoff_at_utc` when observation time is present.
2. For trailing-window features, use rolling windows ending at `cutoff_at_hkt`, usually `1h`, `3h`, `6h`, `12h`, `24h`, and selected HKT daypart windows.
3. For target-day pre-open profiles, reject any row with `observation_at_hkt::date = target_date`.
4. For late `T-1` profiles, the latest hourly observation should usually be on `T-1`, not `T`.
5. If a row is dispatched after cutoff but observation time is before cutoff, it is still ineligible. Publication time controls availability.

### Target-day finalized data prevention

The target label for date `T` may only appear in:

```text
y_true
residual_target = y_true - anchor_forecast_max_c
post-hoc evaluation reports
```

It must never appear in:

```text
features
imputation values
normalization statistics
clipping thresholds
model selection
calibration
cutoff selection
preprocessing fitted across all dates
```

---

## 4. Exact Train, Validation, And Test Ranges

### Forecast-era matrix date range

The main supervised matrix starts at `2000-01-02` because the strict Info.gov lead-1 local min/max forecast anchor begins there. Hourly readings start in 1998, so they cover the whole forecast-era matrix after 2000. Long target history from 1884 is used only for lagged climatology and target-memory features, not as standalone supervised rows before the forecast archive exists.

### Development, validation, and test policy

Use these ranges:

| Split role | Date range | Purpose | May influence model choice? |
|---|---:|---|---:|
| Initial training base | `2000-01-02` through `2010-12-31` | First rolling fold training start | Yes, inside pre-2024 development only |
| Rolling validation fold 1 | Train `2000-01-02`-`2010-12-31`, validate `2011-01-01`-`2013-12-31` | Check performance in the same broad era as the prior 2011-2023 baseline | Yes |
| Rolling validation fold 2 | Train `2000-01-02`-`2013-12-31`, validate `2014-01-01`-`2016-12-31` | Check warm modern era and hot-year behavior | Yes |
| Rolling validation fold 3 | Train `2000-01-02`-`2016-12-31`, validate `2017-01-01`-`2019-12-31` | Check recent pre-COVID, high-temperature era | Yes |
| Rolling validation fold 4 | Train `2000-01-02`-`2019-12-31`, validate `2020-01-01`-`2021-12-31` | Check recent operational-era robustness | Yes |
| Pre-sealed holdout | Train/freeze candidate on allowed fold decisions, evaluate `2022-01-01`-`2023-12-31` | Final pre-2024 no-excuses holdout for promotion decision | No hyperparameter search after this, except reject/no-promote |
| Final pre-sealed production training | `2000-01-02` through `2023-12-31` | Retrain final selected model before sealed confirmation | No new choices; use frozen feature/model/cutoff/weights |
| Locked sealed confirmation test | `2024-01-01` through latest canonical sealed label, currently `2026-05-31` | Final confirmation-only evaluation | No |

### Why these ranges

This split design matches the data reality:

- The official forecast anchor exists from 2000 onward, so earlier target labels cannot train the full residual model.
- 2011-2023 is the documented prior evaluation range for the existing practical baseline, so the rolling folds and holdout cover that same benchmark era.
- 2022-2023 gives a recent pre-sealed holdout immediately before the sealed boundary.
- 2024+ remains protected from cutoff choice, feature selection, model-family choice, hyperparameter tuning, imputation fitting, clipping calibration, and ensemble weight selection.
- Retraining the final selected model on all pre-2024 data is allowed only after all choices are frozen.

### 2026 row policy

The current canonical target label union ends at `2026-05-31`. The hourly table currently extends to `2026-07-04`, and the forecast archive strict local min/max rows extend beyond the target canonical end. For scoring:

```text
score only dates with canonical target labels
score sealed dates only after model freeze
exclude target dates after 2026-05-31 until canonical labels are promoted
```

For live inference after the latest labeled date, the system may build features and predictions, but those rows must be marked `unlabeled_live_inference` and must not enter training, tuning, or reported MAE/RMSE.

### Pre-2000 target history use

Pre-2000 target labels may be used only for leakage-safe historical climatology and lag features for forecast-era rows. They must not create supervised rows for the official-anchor residual model because forecast features are unavailable before 2000.

Allowed:

```text
past-only day-of-year climatology using 1884+ data excluding 1940-1946 and excluding any date after target_date - 2
past-only rolling target memory ending at target_date - 2
modern-era climatology features computed only from past rows
```

Forbidden:

```text
training a direct model on 1884-1999 and comparing it against forecast-era residual models as if features were identical
whole-history monthly averages computed with future rows
using 2024+ sealed target labels to choose climatology windows or smoothing parameters
```

---

## 5. Target And Baselines

### Target variable

For every scored target date `T`:

```text
y_true_c(T) = HKO Daily Extract Absolute Daily Max (deg. C) for local date T
source = label_core.hko_daily_tmax for T < 2024-01-01
source = sealed_confirmation.hko_daily_tmax for T >= 2024-01-01, confirmation only
precision = 0.1 C
station = Hong Kong Observatory / HKO Headquarters
```

### Residual target

For each cutoff profile `C` and target date `T` with an eligible anchor forecast:

```text
anchor_max_c(T, C) = latest eligible Info.gov local forecast max by cutoff
residual_y_c(T, C) = y_true_c(T) - anchor_max_c(T, C)
```

The model trains on `residual_y_c`, not on absolute Tmax, for the primary strategy.

### Scoring rows

A row is scorable for official-anchor models only when all are true:

```text
canonical target label exists
strict eligible latest Info.gov local forecast max exists before cutoff
target_date matches forecast target_date
forecast source and row quality match the documented apples-to-apples source
cutoff_at_hkt and cutoff_at_utc are present
feature eligibility audit passes
```

Do not let hourly feature availability change the scoring row set. Missing hourly features should become missing values plus flags. This keeps ablation scores comparable to the official forecast baseline.

### Baseline 1: simple climatology and persistence

Implement as a weak sanity baseline:

```text
B0a = expanding same-day-of-year climatology median using only labels with local_date <= T - 2 days
B0b = target_lag2_tmax_c
B0c = 0.60 * expanding_doy_climatology + 0.40 * lag2, with weights selected only on rolling validation folds
```

Purpose: verify that the target matrix, date handling, and scoring pipeline work. This baseline should not beat the official forecast consistently.

### Baseline 2: raw official latest forecast max

```text
B1_raw_official_latest = anchor_forecast_max_c
```

This is the primary baseline. All model improvements must be measured against this baseline on the exact same scoring rows and cutoff profile.

### Baseline 3: current grouped residual recreation

Recreate the prior practical baseline as closely as possible:

```text
B3_grouped_residual_shrinkage = anchor_forecast_max_c + grouped_shrunk_residual_c
```

Use groupings such as:

```text
month
season
forecast_range_bin
integer forecast max
forecast_max_bin
issue_hour_bucket
month x integer forecast max
month x forecast_range_bin
```

Train group residual means with empirical-Bayes shrinkage toward the global residual mean. Fit shrinkage hyperparameters only on pre-2024 rolling validation. This baseline is not expected to give a large improvement, but it is essential as the transparent reference.

### Diagnostic direct model baseline

Implement one direct absolute-Tmax model only as a diagnostic:

```text
B4_direct_lgbm_absolute = LightGBM trained to predict y_true_c directly, with anchor_forecast_max_c as a feature
```

Do not promote this unless it beats the residual ensemble on pre-2024 validation and sealed confirmation without tail degradation. Its main role is detecting whether residual formulation is leaving signal unused.

### Metrics

Report all of these for every baseline, ablation, cutoff profile, validation fold, pre-sealed holdout, and sealed confirmation evaluation:

```text
n_scored
MAE
RMSE
median_absolute_error
bias = mean(pred - actual)
p80_absolute_error
p90_absolute_error
p95_absolute_error
max_absolute_error
mean_prediction
mean_actual
mean_anchor_forecast
```

Required slices:

```text
month
calendar quarter
warm season: April-October
cool season: November-March
hot season: June-September
typhoon season: June-October
forecast_max_c bins: <=20, 21-24, 25-27, 28-30, 31-32, >=33
forecast_range_c bins: 0, 1, 2, >=3
issue_hour_bucket
raw official absolute-error deciles, computed post-hoc only
raw official top-20-percent error days, computed post-hoc only
rain/thunderstorm/text regimes
high station-spread regimes
large inland-minus-coastal contrast regimes
```

The raw official error decile is evaluation-only. It must not become a feature because it uses the target.

### Minimum report outputs

For every run, produce:

```text
scoreboard.csv
scoreboard_by_split.csv
scoreboard_by_month.csv
scoreboard_by_regime.csv
ablation_scoreboard.csv
cutoff_sensitivity_scoreboard.csv
prediction_rows.parquet
feature_matrix_schema.json
feature_missingness_report.csv
row_count_audit.json
leakage_audit.json
feature_importance_report.csv
residual_error_diagnostics.md
final_model_card.md
```

---

## 6. Feature Specification

This is the main battlefield. The feature set must be compact, high-signal, and designed for HKO Headquarters residual errors. Do not build hundreds of arbitrary interactions. Do not use generic bag-of-words. Do not use post-cutoff readings. Do not drop rows merely because an hourly or station feature is missing.

### Phase-one feature count target

Target approximate feature counts before model-specific encoding:

```text
official anchor features: 12-18
forecast revision features: 12-20
HKO target-station hourly state: 18-25
neighbor network and gradients: 25-40
warning/text/cyclone flags: 18-30
target-history/climatology: 18-25
calendar/era/missingness flags: 10-20
phase-one total: about 100-150 raw columns maximum
```

If implementation exceeds 160 raw columns in phase one, Codex must justify each added family in the feature schema report or move it to phase two.

---

### 6.1 Official forecast anchor features

#### Source fields

From `public.hko_historical_forecasts_2000_2026` latest eligible anchor row:

```text
forecast_min_c
forecast_max_c
issue_at_hkt
issue_at_utc
target_date
target_issue_lead_days
full_text
source_url
raw_sha256
row_quality_status
```

#### Core features

```text
official_max_c = forecast_max_c
official_min_c = forecast_min_c
official_range_c = forecast_max_c - forecast_min_c
official_midpoint_c = (forecast_max_c + forecast_min_c) / 2
official_max_round_c = forecast_max_c as integer/decimal level
official_range_bin = {0, 1, 2, >=3}
official_max_bin = {<=20, 21-24, 25-27, 28-30, 31-32, >=33}
issue_hour_hkt = hour(issue_at_hkt)
issue_minute_hkt = minute(issue_at_hkt)
issue_hour_bucket = {before_16, 16_17, 18_20, 21_22, 23_plus}
issue_age_minutes = (cutoff_at_utc - issue_at_utc) in minutes
lead_seconds_to_target_start = target_date 00:00 HKT - issue_at_hkt
latest_issue_is_after_23_flag
forecast_anchor_tie_flag
```

#### Normalized official features

Compute climatology using only target-history rows available before `target_date - 2`:

```text
official_max_minus_doy_clim30_c = official_max_c - target_clim_doy_30yr_median_c
official_max_minus_month_clim10_c = official_max_c - target_clim_month_10yr_mean_c
official_midpoint_minus_doy_clim30_c
official_range_z_by_month_past = z-score using expanding past-only month-specific range distribution if enough rows; else null
```

The z-score must be fold-local or expanding-past. No whole-history statistics.

#### Meteorological rationale

The official maximum is the best available meteorological summary. Its range, issue time, and anomaly versus HKO climatology encode confidence, seasonality, and whether the forecast is near a local ceiling/floor. Forecast range can proxy uncertainty: narrow ranges may be high-confidence synoptic days; wider or unusual ranges may indicate frontal, rainy, or convective regimes.

#### Expected information gain

High as anchor, but low incremental gain by itself because this is already the baseline. Its primary value is enabling the residual model to learn conditional corrections.

#### Missingness

Rows without `forecast_max_c` or `forecast_min_c` are excluded from the primary matrix. Do not impute the official anchor for model scoring.

#### Ablation

Ablation A0 and A1 test these features. If a model using only official anchor features cannot reproduce the raw official and grouped baseline scores within `0.002 C`, the pipeline is wrong.

#### Priority

Core phase one.

---

### 6.2 Forecast revision and issuance-path features

#### Source fields

All eligible `public.hko_historical_forecasts_2000_2026` rows for target date `T` and cutoff `C`:

```text
issue_at_hkt
issue_at_utc
forecast_min_c
forecast_max_c
full_text
source_url
raw_sha256
```

#### Aggregation window

The full lead-1 eligible path:

```text
issue_at_utc <= cutoff_at_utc
target_issue_lead_days = 1
target_date = T
```

Do not include lead-0 forecasts. Do not include target-day same-day updates in the primary profile.

#### Core revision features

Let `max_i` be `forecast_max_c` for eligible issue `i`, sorted by issue time, and `n` be the count.

```text
rev_count = n
rev_first_max_c = max_1
rev_prev_max_c = max_{n-1} if n >= 2 else null
rev_latest_minus_prev_max_c = max_n - max_{n-1}
rev_latest_minus_first_max_c = max_n - max_1
rev_path_max_c = max(max_i)
rev_path_min_c = min(max_i)
rev_path_range_c = rev_path_max_c - rev_path_min_c
rev_path_std_c = std(max_i) when n >= 2 else 0
rev_num_up_moves = count(max_i > max_{i-1})
rev_num_down_moves = count(max_i < max_{i-1})
rev_num_same_moves = count(max_i = max_{i-1})
rev_last_change_age_hours = hours between cutoff and latest issue where max changed
rev_last3_slope_c_per_hour = robust slope of max over last min(3,n) issue times
rev_latest3_all_same_flag = 1 if latest three max forecasts are identical
rev_first_issue_hour_hkt
rev_last_issue_hour_hkt
rev_issue_span_hours = issue_at_n - issue_at_1
```

Parallel, lower-priority min/range path features:

```text
rev_latest_minus_prev_min_c
rev_latest_minus_first_min_c
rev_range_latest_minus_prev_c
rev_range_path_std_c
```

Limit phase-one revision features to about 20 columns. Do not add every possible statistic.

#### Text revision features

Using the manual forecast text flags defined in section 6.6:

```text
rev_thunderstorm_added_latest = latest_flag_thunderstorm - previous_flag_thunderstorm clipped to {0,1}
rev_showers_added_latest
rev_very_hot_added_latest
rev_cloudy_added_latest
rev_bright_periods_added_latest
rev_text_regime_changed_latest = any core text flag changed from previous eligible forecast
```

These are useful because sometimes the numeric max remains unchanged while the text changes from “bright periods” to “showers” or adds squally thunderstorms.

#### Meteorological rationale

Forecast revisions are not noise. They encode forecaster uncertainty and newly assimilated meteorological information. A late downward revision may indicate cloud/rain timing. A stable max across many bulletins may mean the official max is robust. A late upward revision may indicate stronger daytime heating, weaker cloud, or delayed rainfall.

#### Expected information gain

Moderate. This family should beat simple grouped residual correction if the official forecast path contains useful unresolved information. Expected retention threshold: at least `0.010 C` MAE improvement over official-only residual model across rolling validation with no tail deterioration.

#### Missingness

If `n=1`, set previous/delta features to null and add:

```text
rev_has_prev_flag = 0
rev_single_issue_flag = 1
```

Tree models handle nulls; linear model uses fold-local median imputation plus missing flags.

#### Ablation

A2: add revision features after official anchor/grouped correction. Retain if:

```text
mean rolling-validation MAE improves >= 0.010 C
pre-sealed 2022-2023 MAE improves >= 0.005 C
RMSE and p90 absolute error do not worsen by > 0.020 C
no month with n >= 100 worsens by > 0.060 C unless overall MAE improves >= 0.030 C
```

#### Priority

Core phase one.

---

### 6.3 Hourly target-station HKO state features

#### Source fields

From `public.hko_info_gov_hourly_readings_1998_2026`:

```text
dispatch_at_hkt
dispatch_at_utc
available_at_utc
observation_at_hkt
observation_at_utc
hko_air_temp_c
hko_relative_humidity_pct
target_station_present
parse_status
station_count
station_missing_count
```

#### Eligible rows

```text
dispatch_at_utc <= cutoff_at_utc
available_at_utc <= cutoff_at_utc
observation_at_utc <= cutoff_at_utc when non-null
parse_status in ('parsed','partial')
```

For phase-one features, use the last 36 hours before cutoff for raw extraction and compute final windows ending at cutoff. This gives enough context for late-evening, afternoon, and previous-day thermal memory without accidentally touching target-day data.

#### Core latest-state features

From the latest eligible row with HKO target-station fields present:

```text
hko_latest_temp_c
hko_latest_rh_pct
hko_latest_dewpoint_c
hko_latest_dewpoint_depression_c = hko_latest_temp_c - hko_latest_dewpoint_c
hko_latest_temp_minus_official_max_c
hko_latest_temp_minus_official_min_c
hko_latest_temp_minus_doy_hour_clim_c
hko_latest_rh_minus_doy_hour_clim_pct
hko_latest_age_minutes = cutoff_at_utc - latest_hko_dispatch_at_utc
hko_latest_observation_hour_hkt
hko_target_station_present_latest_flag
```

Dew point should be computed from temperature and RH using a standard Magnus approximation:

```text
a = 17.625
b = 243.04 C
alpha = ln(RH/100) + a*T/(b+T)
dewpoint = b*alpha/(a-alpha)
```

If RH is missing or outside `0-100`, dew point is null and a missing flag is set.

#### Core trend and window features

Use nearest observations at or before each lookback time. Require the older observation to be within a tolerance of `lookback + 45 minutes`; otherwise set the trend null and set a missing flag.

```text
hko_temp_trend_1h_c = latest_temp - temp_at_or_before(cutoff - 1h)
hko_temp_trend_3h_c
hko_temp_trend_6h_c
hko_temp_trend_12h_c
hko_rh_trend_3h_pct
hko_rh_trend_6h_pct
hko_temp_mean_6h_c
hko_temp_mean_12h_c
hko_temp_mean_24h_c
hko_temp_max_24h_c
hko_temp_min_24h_c
hko_temp_range_24h_c
hko_rh_mean_6h_pct
hko_rh_mean_24h_pct
hko_dispatch_count_24h
hko_partial_parse_count_24h
```

#### HKT daypart features

For target `T`, the pre-target day is `D = T - 1 day`. Compute these only from eligible readings on `D` and earlier, never target date `T`.

```text
hko_pre_target_overnight_min_c = min temp on D 00:00-06:59 HKT, if observed by cutoff
hko_pre_target_morning_mean_c = mean temp on D 07:00-11:59 HKT, if observed by cutoff
hko_pre_target_afternoon_max_sofar_c = max temp on D 12:00-cutoff HKT
hko_pre_target_evening_temp_c = latest temp on D >=18:00 HKT, if cutoff after 18:00
hko_evening_cooling_18_to_cutoff_c = latest temp - first temp at/after D 18:00 HKT
hko_afternoon_warmup_09_to_15_c = temp near D 15:00 - temp near D 09:00, if both eligible
```

For `tminus1_1500`, evening features will be null. For `tminus1_2359`, they should usually be present.

#### Normalization

Do not use raw temperature levels alone. Include anomaly features:

```text
hko_latest_temp_minus_doy_hour_clim_c
hko_temp_mean_24h_minus_doy_clim_c
hko_pre_target_afternoon_max_minus_official_max_c
hko_evening_temp_minus_official_min_c
```

Hourly climatologies must be expanding-past and computed only from previous years/dates available before the row’s cutoff. If hourly climatology implementation is too slow, use target-history daily climatology in phase one and mark hourly climatology as phase two. Do not compute all-history hourly means.

#### Meteorological rationale

HKO Tmax depends on heat memory, humidity, cloud/rain timing, and boundary-layer state. A warm humid evening before target day can indicate persistent warm maritime air and urban heat storage. A rapidly cooling evening can indicate clear skies or dry advection. A high pre-target afternoon temperature relative to the official forecast can signal that the air mass is warmer than the official local forecast anchor implies. RH/dewpoint help distinguish hot-dry continental heating from humid cloudy maritime regimes.

#### Expected information gain

High among non-official features. If this new table is going to materially beat the `~0.92 C` baseline, HKO state features are one of the most likely contributors.

#### Missingness

Do not drop target rows. Add missing flags:

```text
hko_latest_missing_flag
hko_trend_1h_missing_flag
hko_trend_3h_missing_flag
hko_trend_6h_missing_flag
hko_daypart_missing_flags
hko_latest_age_gt_90min_flag
```

If the latest HKO target-station reading is missing but a row exists, use null for the values and preserve dispatch/station missingness features.

#### Ablation

A3: add HKO hourly state after forecast-revision features. Retain if:

```text
rolling-validation MAE improves >= 0.020 C
pre-sealed 2022-2023 MAE improves >= 0.010 C
warm-season MAE improves or is neutral
RMSE and p90 absolute error do not worsen by > 0.020 C
```

If HKO hourly features improve only the training folds but not 2022-2023, suspect overfitting to source-era quirks or leakage in hourly climatology.

#### Priority

Core phase one.

---

### 6.4 Hourly neighbor-station network features

#### Source fields

From `public.hko_info_gov_hourly_readings_1998_2026`:

```text
station_readings_jsonb
station_count
station_missing_count
station_temp_min_c
station_temp_max_c
station_temp_mean_c
station_temp_spread_c
dispatch_at_utc
observation_at_utc
```

Each JSONB station object contains:

```text
station_canonical_name
temperature_c
temperature_missing
raw_temperature_text
station_order
```

#### Station groups

Create explicit group definitions in code. Do not infer groups dynamically from names.

```python
URBAN_CORE = [
    "KING'S PARK", "HONG KONG PARK", "HAPPY VALLEY", "SHAM SHUI PO",
    "KOWLOON CITY", "WONG TAI SIN", "KWUN TONG", "KAI TAK RUNWAY PARK",
    "SHAU KEI WAN"
]

COASTAL_MARINE = [
    "CHEK LAP KOK", "CHEUNG CHAU", "SAI KUNG", "STANLEY",
    "WONG CHUK HANG", "TSEUNG KWAN O"
]

INLAND_NT = [
    "SHA TIN", "TA KWU LING", "SHEK KONG", "YUEN LONG PARK",
    "TAI PO", "TAI MEI TUK"
]

WEST_NW_NT = [
    "LAU FAU SHAN", "TUEN MUN", "TSING YI", "TSUEN WAN",
    "TSUEN WAN HO KOON", "TSUEN WAN SHING MUN VALLEY"
]

CORE_STATION_DELTAS = [
    "KING'S PARK", "HONG KONG PARK", "CHEK LAP KOK", "CHEUNG CHAU",
    "SHA TIN", "TA KWU LING", "LAU FAU SHAN", "SAI KUNG",
    "WONG CHUK HANG", "SHEK KONG", "TSING YI", "TSEUNG KWAN O"
]
```

If a station is absent in an older era, leave the feature null and set its missing flag. Do not require a fixed panel.

#### Outlier handling

Source summaries include rare neighbor extremes such as `-9 C` and `50 C`. These can be real source-text anomalies. Apply robust handling before group features:

1. If station temperature is null or `temperature_missing=true`, treat as missing.
2. If temperature is outside hard Hong Kong plausibility range `[0, 42] C`, set it null and set `station_outlier_flag=1`.
3. If HKO latest temp is present and `abs(station_temp - hko_latest_temp) > 12 C`, set station value null and set `station_relative_outlier_flag=1`.
4. In model training, additionally winsorize continuous station features using fold-local training quantiles `[0.001, 0.999]`; apply those cut points to validation/test.
5. Report counts of outlier removals by station and year.

#### Latest-network features

From the latest eligible hourly readings row with station JSON:

```text
network_latest_station_count
network_latest_missing_count
network_latest_valid_count
network_latest_temp_mean_c
network_latest_temp_min_c
network_latest_temp_max_c
network_latest_temp_spread_c
network_latest_hko_percentile = percentile rank of hko_latest_temp among HKO + valid neighbor temps
network_latest_max_minus_hko_c
network_latest_hko_minus_mean_c
network_latest_valid_fraction
```

Do not blindly trust source aggregate min/max if station outlier rules modify individual values; recompute aggregates from cleaned station values for model features.

#### Group latest features

For each of the four station groups:

```text
{group}_latest_mean_c
{group}_latest_median_c
{group}_latest_max_c
{group}_latest_min_c
{group}_latest_count
{group}_latest_missing_fraction
{group}_latest_mean_minus_hko_c
{group}_latest_max_minus_hko_c
```

To keep feature count sane, include mean, max, count, and mean-minus-HKO in phase one for all groups; keep median/min only if feature count remains below budget or if ablation shows they add signal.

#### Core cross-group contrast features

```text
inland_nt_mean_minus_coastal_marine_mean_c
inland_nt_max_minus_urban_core_mean_c
west_nw_nt_mean_minus_coastal_marine_mean_c
urban_core_mean_minus_coastal_marine_mean_c
urban_core_mean_minus_hko_c
coastal_marine_mean_minus_hko_c
nt_hotspot_max_minus_official_max_c = max(INLAND_NT, WEST_NW_NT) - official_max_c
network_spread_minus_month_clim_c = network_latest_spread - expanding past month spread climatology if available
```

#### Core station-specific features

For each station in `CORE_STATION_DELTAS`, using the latest eligible row:

```text
station_{name}_latest_temp_c
station_{name}_latest_minus_hko_c
station_{name}_missing_flag
```

Do not add trends for all stations in phase one. Trends are phase two unless group-level trends prove useful.

#### Windowed network features

Over windows ending at cutoff:

```text
network_spread_mean_6h_c
network_spread_max_6h_c
network_spread_max_24h_c
network_mean_trend_6h_c = latest cleaned network mean - cleaned network mean near cutoff - 6h
inland_minus_coastal_trend_6h_c
nt_hotspot_max_24h_c
nt_hotspot_max_24h_minus_official_max_c
station_missing_count_mean_24h
station_missing_count_max_24h
```

#### Meteorological rationale

HKO Headquarters is coastal and urban. Its daily max can be suppressed by maritime influence, sea breeze, cloud, or convective outflow, while inland New Territories may be much hotter. The official forecast text often gives an urban-area max plus a New Territories caveat, but the settlement station is HKO Headquarters, not the territory maximum. Network gradients can tell whether the official urban max is too high/low for HKO specifically:

- inland much warmer than coastal: HKO may stay moderated despite hot NT wording;
- urban core warmer than coastal: HKO may track urban heat better;
- coastal/marine stations similar to HKO: maritime suppression likely;
- high spatial spread: mesoscale uncertainty and thunderstorm/outflow risk;
- station missingness/outages: sometimes correlated with severe weather operations or source irregularities.

#### Expected information gain

High to moderate. The group-contrast features are more likely to generalize than individual sparse station features. The strongest expected features are inland-minus-coastal, urban-minus-coastal, network spread, HKO percentile, and core station deltas to HKO.

#### Missingness

Use nulls plus flags. For group means, require at least two valid stations for group mean; otherwise null and `{group}_insufficient_count_flag=1`. For single station deltas, null if station missing/outlier or HKO latest missing.

#### Ablation

A4: add network/gradient features after HKO state. Retain if:

```text
rolling-validation MAE improves >= 0.015 C
pre-sealed 2022-2023 MAE improves >= 0.010 C
warm-season or high-spread-regime MAE improves >= 0.020 C
RMSE and p90 do not worsen by > 0.020 C
```

If station-specific features overfit, keep group/contrast features and remove individual station deltas except `KING'S PARK`, `CHEK LAP KOK`, `CHEUNG CHAU`, `SHA TIN`, and `TA KWU LING`.

#### Priority

Core phase one for group features and core station deltas. All-station sparse temporal features are phase two.

---

### 6.5 Station-gradient and maritime/inland contrast features

This is separated from the general network block because these are the most physically important neighbor-derived features for HKO station residuals.

#### Core contrast features

```text
maritime_suppression_index_c = hko_latest_temp_c - coastal_marine_latest_mean_c
urban_heat_index_c = urban_core_latest_mean_c - coastal_marine_latest_mean_c
inland_heat_index_c = inland_nt_latest_mean_c - coastal_marine_latest_mean_c
nt_heat_ceiling_index_c = max(inland_nt_latest_max_c, west_nw_nt_latest_max_c) - official_max_c
hko_station_rank_pct = percentile rank of HKO temp among HKO + valid stations
coastal_hko_alignment_c = abs(hko_latest_temp_c - coastal_marine_latest_mean_c)
urban_hko_alignment_c = abs(hko_latest_temp_c - urban_core_latest_mean_c)
inland_coastal_spread_6h_max_c
network_spatial_heterogeneity_flag = 1 if latest spread >= expanding past month p80
```

#### Targeted interactions

Only include these interactions in phase one:

```text
inland_heat_index_x_warm_season
network_spread_x_thunderstorm_flag
maritime_suppression_x_southerly_forecast_flag
hko_latest_temp_anom_x_warm_season
official_range_x_thunderstorm_or_rainstorm_flag
nt_heat_ceiling_x_forecast_nt_higher_text_flag
```

Do not create arbitrary pairwise interactions. These six are justified by meteorology and source text semantics.

#### Meteorological rationale

HKO Tmax is often a boundary between urban heat, maritime air, and New Territories heating. The official forecast phrase “urban areas” may not perfectly map to the HKO Headquarters sensor under all wind/cloud regimes. These contrast features directly test whether HKO is behaving like an urban-core station, a coastal station, or an intermediate station before cutoff.

#### Expected information gain

Potentially high on warm-season and convective days, lower in stable cool-season regimes. This group should be judged especially on warm-season, high-network-spread, thunderstorm-warning, and high-official-error slices.

#### Missingness

Same as neighbor-network features. If a group has too few stations, set contrast null and add missing flag.

#### Ablation

Included in A4. Additionally report a contrast-only diagnostic:

```text
A4b = HKO hourly + group contrast features, no individual station deltas
A4c = HKO hourly + individual station deltas, no group contrasts
```

Keep the simpler one if scores are statistically similar.

#### Priority

Core phase one.

---

### 6.6 Warnings, rain, lightning, thunderstorm, and tropical-cyclone text features

#### Source fields

From forecast table latest eligible anchor and previous eligible forecast rows:

```text
full_text
temperature_text if present
forecast_period if present
issue_at_hkt
```

From hourly readings table eligible rows:

```text
warning_text
rainfall_text
lightning_text
tropical_cyclone_text
tropical_cyclone_name
tropical_cyclone_lat
tropical_cyclone_lon
full_text
```

#### Text extraction rule

Use a fixed, hand-built, case-insensitive regex dictionary. Do not use bag-of-words, TF-IDF, embeddings, or arbitrary NLP in phase one. Every text feature must correspond to a physical regime.

#### Forecast latest text flags

Parse the latest eligible local forecast `full_text`:

```text
fcst_flag_showers = regex /SHOWERS?/
fcst_flag_heavy_showers = regex /HEAVY|HEAVIER/ within 8 words of /SHOWERS?/
fcst_flag_thunderstorm = regex /THUNDERSTORMS?/
fcst_flag_squally = regex /SQUALLY/
fcst_flag_bright_periods = regex /BRIGHT PERIODS/
fcst_flag_sunny = regex /SUNNY INTERVALS|SUNNY PERIODS|SUNNY/
fcst_flag_cloudy = regex /MAINLY CLOUDY|CLOUDY|OVERCAST/
fcst_flag_fine = regex /FINE/
fcst_flag_very_hot = regex /VERY HOT/
fcst_flag_hot = regex /\bHOT\b|VERY HOT/
fcst_flag_mist_fog = regex /MIST|FOG|HAZE/
fcst_flag_tropical_cyclone = regex /TROPICAL STORM|TYPHOON|TROPICAL CYCLONE|SEVERE TROPICAL STORM/
fcst_flag_monsoon = regex /MONSOON/
fcst_flag_active_southerly = regex /ACTIVE SOUTHERLY AIRSTREAM/
fcst_flag_southerly = regex /SOUTHERLY|SOUTHEASTERLY|SOUTH TO SOUTHEASTERLY/
fcst_flag_easterly_ne = regex /EASTERLY|NORTHEASTERLY|NORTH TO NORTHEASTERLY/
fcst_flag_nt_higher = regex /NEW TERRITORIES/ and /COUPLE OF DEGREES HIGHER|DEGREES HIGHER/
```

#### Hourly warning/context features

For eligible hourly readings in trailing windows ending at cutoff:

```text
hourly_any_warning_text_24h
hourly_warning_text_count_24h
hourly_any_thunderstorm_warning_24h
hourly_any_rainstorm_warning_24h
hourly_any_amber_rainstorm_24h
hourly_any_red_black_rainstorm_24h
hourly_any_very_hot_warning_24h
hourly_any_strong_monsoon_24h
hourly_any_tropical_cyclone_text_24h
hourly_any_lightning_text_24h
hourly_lightning_text_count_24h
hourly_any_rainfall_text_24h
hourly_rainfall_text_count_24h
hours_since_latest_warning_text
hours_since_latest_lightning_text
hours_since_latest_rainfall_text
hours_since_latest_tropical_cyclone_text
```

Use both `6h` and `24h` windows for the most important sparse features:

```text
hourly_any_thunderstorm_warning_6h
hourly_any_rainstorm_warning_6h
hourly_any_lightning_text_6h
hourly_any_tropical_cyclone_text_6h
```

#### Tropical cyclone geometry

Phase-one compact implementation:

```text
tc_name_present_latest_flag
tc_position_present_latest_flag
tc_lat_latest
tc_lon_latest
tc_distance_to_hko_km_latest
tc_bearing_sin_latest
tc_bearing_cos_latest
tc_distance_missing_flag
```

Compute distance/bearing only from hourly table columns when present and eligible before cutoff. Do not attempt complex track modeling in phase one.

If cyclone lat/lon parsing is unreliable or sparse, this feature group may remain nullable; do not drop rows.

#### Meteorological rationale

Cloud, rain, squally thunderstorms, tropical cyclones, and monsoon regimes affect daily maximum mainly through solar suppression, convective cooling, wind direction, and boundary-layer mixing. These text fields can explain residuals not captured by the numeric max. A forecast max of 31 under “bright periods” is not the same as 31 under “heavy showers and squally thunderstorms at first tomorrow.”

#### Expected information gain

Moderate overall, high in sparse regimes. Text may not move full-sample MAE much, but it should reduce tail errors on thunderstorm, rainstorm, and tropical-cyclone days. Keep a text feature if it improves relevant regime slices without harming overall MAE materially.

#### Missingness

`NULL` text means no parsed text block, not necessarily no weather. Encode:

```text
text_block_present_flag
feature_flag = 0 if text block present and regex absent
feature_flag = 0 with text_missing_flag=1 if text block missing
```

For forecast `full_text`, missing should be rare in eligible rows; report any missing full text.

#### Ablation

A5: add text/warning/cyclone features. Retain if either:

```text
overall rolling-validation MAE improves >= 0.005 C with no tail degradation
```

or:

```text
thunderstorm/rain/tropical-cyclone slice MAE improves >= 0.020 C
AND overall MAE worsens by <= 0.005 C
AND RMSE/p90 do not worsen by > 0.020 C
```

If text features worsen normal days, keep them only inside the nonlinear models with regularization or move them to a phase-two regime expert.

#### Priority

Core phase one, but with strict ablation retention. Cyclone geometry is phase-one if easy from existing columns; otherwise phase two.

---

### 6.7 Target-history lag and climatology features

#### Source fields

From `feature_safe.hko_target_history_pre2024` and, for final locked live-style replay only, lagged `sealed_confirmation.hko_daily_tmax` rows after the model is frozen and availability is proven.

```text
local_date
target_tmax_c
target_station
target_source_id
quality_status
```

#### Eligibility rule

Primary phase-one lag cutoff:

```text
history_date <= target_date - 2 days
```

Do not use lag 1 unless a separate publication-availability table proves it is available before the cutoff for every row.

#### Core lag features

```text
target_lag2_tmax_c
target_lag3_tmax_c
target_lag7_tmax_c
target_lag14_tmax_c
target_lag30_tmax_c
target_lag60_tmax_c
target_lag365_tmax_c
```

If a lag falls inside the 1940-1946 gap or any missing label date, set null and add missing flag.

#### Rolling target-memory features

All rolling windows end at `target_date - 2 days`:

```text
target_roll7_mean_lag2_c
target_roll14_mean_lag2_c
target_roll30_mean_lag2_c
target_roll60_mean_lag2_c
target_roll365_mean_lag2_c
target_roll7_max_lag2_c
target_roll14_max_lag2_c
target_roll30_anomaly_lag2_c = target_lag2_tmax_c - target_roll30_mean_lag2_c
target_roll7_minus_roll30_c
target_hot_spell_33_lag2_days = consecutive days ending T-2 with tmax >= 33.0 C
target_very_hot_spell_34_lag2_days = consecutive days ending T-2 with tmax >= 34.0 C
target_cool_spell_16_lag2_days = consecutive days ending T-2 with tmax <= 16.0 C
```

Require minimum observed counts:

```text
roll7 requires >=5 observed days
roll14 requires >=10 observed days
roll30 requires >=20 observed days
roll60 requires >=40 observed days
roll365 requires >=240 observed days
```

If count threshold fails, feature null + missing flag.

#### Past-only climatology features

For each row, compute using only labels with `local_date <= target_date - 2 days`:

```text
target_clim_doy_all_past_median_c = median Tmax for day-of-year ±7 days across all past years
target_clim_doy_30yr_median_c = same, last 30 available years only
target_clim_doy_10yr_median_c = same, last 10 available years only
target_clim_month_30yr_mean_c
target_clim_month_10yr_mean_c
target_clim_month_10yr_std_c
target_modern_warming_signal_c = target_clim_doy_10yr_median_c - target_clim_doy_30yr_median_c
target_lag2_minus_doy30_clim_c
target_roll30_minus_doy30_clim_c
```

Day-of-year windows must handle leap days explicitly. For Feb 29, use Feb 28/Mar 1 neighborhood or numeric day-of-year on a 366-day calendar consistently.

#### Meteorological rationale

Target history captures station-specific climatology, thermal persistence, hot spells, cool-season cold-air persistence, urban heat memory, and modern warming. Official forecasts already contain climatological knowledge, so target history may not add huge average signal, but it can stabilize residual corrections by season and regime.

#### Expected information gain

Low to moderate overall, potentially useful in hot/cool spells and modern-era anomalies. It is also valuable for normalizing official forecast and hourly features.

#### Missingness

Use nulls plus flags. Do not impute 1940-1946 target labels. Do not fill gaps with external data.

#### Ablation

A6: add target-history features after text/network. Retain if:

```text
overall rolling-validation MAE improves >= 0.005 C
OR hot-spell/cool-season slice MAE improves >= 0.015 C
AND RMSE/p90 remain neutral within 0.020 C
```

If target-history features only help because lag computation accidentally used future rows, leakage tests must catch this.

#### Priority

Core phase one for lag2+ and past-only climatology. Lag1 is rejected for phase one.

---

### 6.8 Calendar, seasonality, and modern-era features

#### Source fields

Derived only from `target_date` and past-only climatology.

#### Core calendar features

```text
month
quarter
day_of_year
doy_sin = sin(2*pi*day_of_year/366)
doy_cos = cos(2*pi*day_of_year/366)
warm_season_flag = month in [4,5,6,7,8,9,10]
cool_season_flag = month in [11,12,1,2,3]
hot_season_flag = month in [6,7,8,9]
typhoon_season_flag = month in [6,7,8,9,10]
shoulder_season_flag = month in [3,4,5,10,11]
year
trend_years_since_2000 = year + fractional_day - 2000
post_2010_flag
post_2018_flag
```

For tree models, month and issue bucket can be categorical. For linear models, one-hot encode month/season using training-fold categories only.

#### Targeted interactions

Only these calendar interactions are allowed in phase one:

```text
official_max_c x month categorical handled by tree/catboost; for linear add month-specific intercepts only
hko_latest_temp_minus_doy_clim_c x warm_season_flag
network_spread_c x warm_season_flag
official_range_c x thunderstorm_or_rainstorm_flag
forecast_revision_delta_c x issue_hour_bucket
inland_minus_coastal_c x southerly_forecast_flag
```

Do not add polynomial expansion.

#### Meteorological rationale

HKO residual behavior differs strongly by season. Cool-season Tmax errors often involve cloud, monsoon, and frontal timing. Warm-season errors often involve rain timing, sea breeze, thunderstorms, tropical cyclones, and sunshine duration. Modern warming shifts station climatology, and the model needs a compact way to avoid anchoring too strongly to older climate.

#### Expected information gain

Calendar variables are necessary for conditioning; alone they should not be expected to deliver a large improvement beyond official forecast.

#### Missingness

None.

#### Ablation

Calendar features are included from A1 onward because they are required for fair residual correction. Do not report a model without month/season except as a sanity check.

#### Priority

Core phase one.

---

### 6.9 Missingness and data-quality flags

#### Required flags

```text
forecast_anchor_tie_flag
forecast_revision_missing_prev_flag
forecast_single_issue_flag
forecast_full_text_missing_flag
hourly_latest_missing_flag
hourly_latest_age_gt_90min_flag
hko_target_station_missing_flag
hko_rh_missing_flag
hko_dewpoint_missing_flag
station_json_empty_latest_flag
station_group_{group}_insufficient_count_flag
station_{name}_missing_flag for core station deltas
station_{name}_outlier_flag for core station deltas
network_outlier_any_flag_24h
station_missing_count_latest
station_missing_count_mean_24h
parse_partial_count_24h
text_warning_block_missing_flag
cyclone_position_missing_flag
target_lag_{k}_missing_flag
target_roll_{window}_insufficient_count_flag
```

#### Meteorological and data rationale

Missingness is not only data dirt. In this archive, high-dispatch days, warning periods, tropical cyclone conditions, station missing values, and partial parses can correlate with severe or abnormal weather regimes. Preserve missingness as signal while preventing the model from exploiting parser artifacts in a non-generalizable way.

#### Expected information gain

Low overall but useful for robustness and debugging. Missingness flags are mandatory because they make imputation auditable.

#### Missingness handling by model

```text
LightGBM/CatBoost: preserve nulls where supported; include flags.
Linear model: fold-local median imputation for continuous variables; zero/false for booleans; include flags.
Grouped residual baseline: use explicit 'MISSING' buckets for categorical bins.
```

#### Ablation

Missingness flags should be included with each feature family. Do not run family features without their missingness flags.

#### Priority

Core phase one.

---

### 6.10 Phase-two feature queue

Only implement these after phase-one ablations:

1. **All 27 station sparse latest features**: add latest temp, delta to HKO, and missing flag for every station. Keep only if it improves validation by at least `0.010 C` beyond core groups.
2. **Station group temporal trends by daypart**: morning/afternoon/evening group trends beyond 6h windows.
3. **Cyclone geometry expansion**: distance trend, quadrant, storm movement proxies from consecutive hourly cyclone positions.
4. **Regime-specific residual experts**: separate residual models for warm-wet, hot-sunny, tropical-cyclone/monsoon, and cool-season regimes with a conservative router.
5. **Lag1 target-history experiment**: only after row-level publication availability for Daily Extract proves lag1 is available before each cutoff.
6. **Forecast `usable_local_tmax_only` rows**: optional coverage extension if strict min/max row count is limiting; must report apples-to-apples comparability separately because range/min features are absent.
7. **Intraday target-day cutoffs**: only if a separate task permits target-day observations before a live trade timestamp; not part of the primary pre-target-day system.

### 6.11 Rejected phase-one feature families

Reject these for phase one:

```text
raw Daily Extract target payload values as predictors
same-day target labels
post-cutoff hourly readings
HKO 9-day API as primary anchor
Info.gov 5-day/7-day/9-day bulletin-only rows as anchor
weather website widgets
airport forecasts as anchor
NOAA/ISD direct operational features
IGRA upper-air features
HKO daily all-elements variables as live predictors
arbitrary text embeddings
TF-IDF bag-of-words
large polynomial/pairwise interaction expansions
whole-history climatology or normalization
features requiring months of extra backfill before first result
```

---

## 7. Model Architecture

### Primary residual target

All primary models train on:

```text
residual_y_c = actual_hko_tmax_c - official_anchor_max_c
```

Absolute final prediction is reconstructed only at scoring time:

```text
pred_tmax_c = official_anchor_max_c + residual_hat_c
```

### Model families

#### Model M0: zero residual

```text
residual_hat = 0
prediction = official_anchor_max_c
```

This must be included as an ensemble candidate and fallback.

#### Model M1: grouped empirical-Bayes residual correction

Use grouped historical residuals:

```text
groups = [
  month,
  season,
  forecast_max_bin,
  official_range_bin,
  issue_hour_bucket,
  month x rounded forecast_max_c,
  month x official_range_bin,
  warm_season x forecast_max_bin
]
```

For each group, estimate residual mean with shrinkage:

```text
shrunk_group_mean = (n_group * mean_group + k * global_mean) / (n_group + k)
```

Tune `k` on rolling validation from:

```text
k in [5, 10, 20, 50, 100, 200]
```

If multiple group estimates are available, combine by validation-chosen convex weights or by reliability order. Include a global residual mean fallback.

#### Model M2: LightGBM residual model

Use LightGBM as the main nonlinear residual learner.

Objective candidates:

```text
objective = regression_l1  # primary for MAE
secondary experiment = huber or regression with fair loss if available
```

Search boundaries:

```text
num_leaves: [7, 15, 31]
max_depth: [3, 4, 5, 6, -1 with num_leaves <= 15]
min_data_in_leaf: [40, 80, 120, 200]
learning_rate: [0.015, 0.025, 0.04, 0.06]
n_estimators: up to 3000 with early stopping
feature_fraction: [0.60, 0.75, 0.90]
bagging_fraction: [0.60, 0.80, 1.00]
bagging_freq: [0, 1, 5]
lambda_l1: [0, 0.1, 1.0]
lambda_l2: [1, 5, 20, 50]
min_gain_to_split: [0, 0.01, 0.05]
```

Constraints:

```text
no randomized search using sealed data
early stopping only on the current fold validation range
feature preprocessing fitted only on fold training rows
cap total hyperparameter trials to a practical budget, e.g. <= 80 per cutoff profile
```

Why LightGBM fits this problem:

- It handles nonlinear thresholds and interactions among official forecast, station gradients, and warning regimes.
- It handles missing values natively.
- It performs well on medium-sized tabular data without needing huge feature sets.
- It can learn month/range/gradient interactions that grouped residual models cannot.

#### Model M3: CatBoost residual model

Use CatBoost as a complementary nonlinear learner.

Parameters:

```text
loss_function = MAE
iterations = up to 3000 with early stopping
depth = [3, 4, 5, 6]
learning_rate = [0.015, 0.025, 0.04, 0.06]
l2_leaf_reg = [3, 10, 30, 80]
random_strength = [0, 0.5, 1.0]
bootstrap_type = Bayesian or Bernoulli
subsample = [0.7, 0.9, 1.0] where applicable
```

Categorical fields:

```text
month
quarter
season bucket
issue_hour_bucket
official_max_bin
official_range_bin
cutoff_profile if multi-profile training is later enabled
```

Why CatBoost fits this problem:

- It handles categorical buckets and missing values robustly.
- It is often stable on small-to-medium datasets with mixed numeric/categorical features.
- It provides a useful second view of nonlinear residual structure.

#### Model M4: robust linear residual model

Use one of:

```text
HuberRegressor with ElasticNet-style preprocessing, or
QuantileRegressor(quantile=0.5) if runtime is acceptable
```

Preprocessing:

```text
fold-local median imputation for continuous variables
fold-local robust scaling using median/IQR
one-hot encoding for categorical variables fitted on training fold
include only core features and targeted interactions; no high-cardinality station explosion
```

Search:

```text
Huber epsilon: [1.1, 1.35, 1.7, 2.0]
alpha/l2 regularization: [0.0001, 0.001, 0.01, 0.1, 1.0]
QuantileRegressor alpha: [0.0001, 0.001, 0.01, 0.1]
```

Why include it:

- It is a no-drama guardrail.
- If trees overfit sparse text/station regimes, the linear residual model may preserve stable low-amplitude corrections.
- Its coefficients are diagnostic for official forecast bias and feature direction.

### Ensemble and shrinkage logic

For each validation fold, produce out-of-fold residual predictions from:

```text
M0_zero
M1_grouped
M2_lgbm
M3_catboost
M4_linear
```

Fit ensemble weights only on pre-2024 validation predictions:

```text
minimize MAE(actual, official_anchor + sum(w_i * residual_hat_i))
subject to:
  w_i >= 0
  sum(w_i) = 1
  M0_zero included as a candidate
```

Then fit a global shrinkage scalar:

```text
residual_blend = sum(w_i * residual_hat_i)
final_residual = lambda * residual_blend
lambda in [0.00, 0.05, 0.10, ..., 1.00]
```

Choose `lambda` on validation under no-harm constraints:

```text
MAE must improve vs raw official
RMSE must not worsen by > 0.010 C
p90 absolute error must not worsen by > 0.020 C
absolute bias must not exceed raw official absolute bias by > 0.030 C
```

If no nonzero lambda passes, use `lambda=0` and report no-promote.

### Calibration and clipping

Residual clipping:

```text
clip residual_hat by fold-local month-specific residual quantiles [0.01, 0.99]
if month-specific count insufficient, use global fold residual quantiles
hard emergency residual cap: [-3.0 C, +3.0 C] unless validation proves a wider cap improves p90/RMSE
```

Final prediction clipping:

```text
lower_bound = fold-local month-specific target p0.001 - 0.5 C
upper_bound = fold-local month-specific target p0.999 + 0.5 C
pred = min(max(pred, lower_bound), upper_bound)
```

Do not compute clipping bounds on all history. Do not compute them using sealed data. Do not clip to a range that hides model failures without reporting pre/post clipping metrics.

### Input normalization

Tree models:

```text
no standard scaling required
preserve continuous values and nulls
categoricals as native where supported or integer categories where safe
```

Linear model:

```text
fit imputer and scaler on training fold only
apply to validation/test
store preprocessing artifacts with fold/run metadata
```

Climatology and anomaly features:

```text
computed as-of each target_date using past-only rows
not fitted globally
```

### Fold structure

Use the rolling folds defined in section 4. For each fold:

1. Build train features using only source rows available by each train row cutoff.
2. Fit preprocessing on train only.
3. Train models on train only.
4. Predict validation rows.
5. Store out-of-fold predictions and feature metadata.

For final pre-sealed production model:

1. Freeze all feature definitions, hyperparameters, clipping rules, ensemble weights, and shrinkage based on pre-2024 development.
2. Refit models on `2000-01-02` through `2023-12-31`.
3. Evaluate 2024+ sealed confirmation once.

### Model selection criterion

Primary:

```text
lowest validation MAE among candidates that satisfy no-harm RMSE and p90 constraints
```

Tie-breakers:

```text
1. lower p90 absolute error
2. lower RMSE
3. lower absolute bias
4. smaller feature set / simpler model
5. more stable monthly MAE
```

No model may be selected based on sealed 2024+ performance.

### Diagnostics and uncertainty outputs

Compute diagnostic uncertainty proxies, not for target scoring but for risk reporting:

```text
ensemble_residual_std = standard deviation of residual predictions across M1-M4
model_disagreement_abs_c
predicted_abs_residual_proxy = rolling-validation calibrated absolute residual estimate if implemented
high_risk_day_flag = ensemble_disagreement or station_spread high or warning regime
```

These are useful for identifying days where MAE may be fragile. They are not a substitute for point-forecast scoring.

---

## 8. Ablation And Experiment Plan

### Fixed row-set rule

For a given cutoff profile and date range, all ablations must score the same official-anchor-eligible rows. Adding hourly/network/text features must not drop rows. Missing features are null + flags.

### Required ablation sequence

Run the sequence for `tminus1_2359` first, then repeat the compact final candidates for cutoff sensitivity profiles.

| Ablation | Name | Features/models | Purpose | Retention threshold |
|---|---|---|---|---|
| A0 | Raw official | `prediction = official_max_c` | Primary baseline | Must reproduce official rows exactly |
| A1 | Grouped residual | Official anchor + calendar + grouped residual shrinkage | Recreate current practical baseline | Keep as baseline if MAE improves at all; do not promote if improvement < `0.010 C` |
| A2 | Revision residual | A1 + forecast revision/path features in LightGBM/CatBoost/linear | Test whether issuance path explains residuals | Keep if rolling MAE improves `>=0.010 C`, p90/RMSE safe |
| A3 | HKO hourly state | A2 + HKO temp/RH/dewpoint/trends/dayparts | Test target-station pre-cutoff state | Keep if rolling MAE improves `>=0.020 C`, pre-sealed improves `>=0.010 C` |
| A4 | Network gradients | A3 + group/station/network spread/contrast features | Test spatial structure around HKO | Keep if rolling MAE improves `>=0.015 C`, warm/high-spread slices improve |
| A4b | Group contrasts only | A3 + group contrast features, no individual station deltas | Simpler network diagnostic | Prefer if within `0.005 C` of A4 |
| A4c | Core station deltas only | A3 + core station deltas, no group contrasts | Sparse station diagnostic | Keep station deltas only if clearly helpful |
| A5 | Text/warning regimes | A4 best + forecast/hourly warning/rain/lightning/cyclone flags | Test weather-regime tail correction | Keep if overall `>=0.005 C` MAE gain or sparse-regime `>=0.020 C` gain with no overall harm |
| A6 | Target memory/climatology | A5 + lag2+/rolling/past-only climatology | Test station-specific memory and normalization | Keep if overall `>=0.005 C` gain or hot/cool slice `>=0.015 C` gain |
| A7 | Final residual ensemble | Best retained feature set + M0/M1/M2/M3/M4 convex blend + shrinkage | Final candidate | Must beat raw official on MAE and pass no-harm RMSE/p90 gates |
| A8 | Direct absolute diagnostic | Direct LightGBM absolute-Tmax model with same features | Challenge residual assumption | Promote only if it beats A7 robustly with no tail harm |

### Cutoff sensitivity sequence

After A7 is selected on pre-2024 data for `tminus1_2359`, run:

```text
tminus1_1500 A0, A1, A3, A7 compact
tminus1_1800 A0, A1, A3, A7 compact
tminus1_2100 A0, A1, A3, A7 compact
tminus1_2359 A0-A7 full
```

Do not let sealed data choose cutoff. Report how MAE, RMSE, p90, and row counts change by cutoff. The earlier cutoffs may be useful operationally even if less accurate.

### Slice requirements for every ablation

For every ablation, report:

```text
overall metrics
monthly metrics
warm/cool/hot/typhoon-season metrics
forecast_max_bin metrics
forecast_range_bin metrics
issue_hour_bucket metrics
warning/thunderstorm/rain/cyclone regime metrics
network_spread_high versus normal metrics
inland_minus_coastal_high versus normal metrics
raw official top-20-percent error days, post-hoc only
```

### Pass/fail gates

#### Feature family gate

A feature family is retained only if it meets its threshold and does not break tail robustness:

```text
RMSE degradation <= 0.020 C versus previous retained ablation
p90 degradation <= 0.020 C versus previous retained ablation
no large month slice degradation > 0.060 C for n >= 100 unless overall MAE gain >= 0.030 C
```

#### Final promotion gate

The final model has three possible outcome labels:

```text
stretch_success:
  final MAE <= 0.67 C on pre-sealed holdout and sealed confirmation
  MAE improvement >= 0.25 C vs raw official on same rows
  RMSE and p90 improve vs raw official

credible_major_improvement:
  final MAE improves >= 0.08 C vs raw official on pre-sealed holdout
  sealed confirmation also improves >= 0.06 C after freeze
  RMSE and p90 do not worsen

no_promote_cosmetic:
  MAE improvement < 0.035 C, or RMSE/p90 worsen materially, or monthly/regime fragility appears
```

If the system only reproduces the previous `~0.006 C` residual improvement, it is not a useful ML improvement and should be reported as no-promote.

### Statistical stability checks

Because daily samples are serially correlated, do not rely only on raw point differences. Add:

```text
block bootstrap confidence interval for MAE delta using 30-day blocks
year-by-year MAE delta table
monthly win/loss count versus raw official
residual scatter by official forecast max and month
```

The final report should state whether improvements are broad or concentrated in a few regimes.

---

## 9. Leakage And Data Integrity Tests

Codex must implement these tests before trusting any score.

### 9.1 Cutoff eligibility tests

For every feature row:

```text
assert forecast_anchor_issue_at_utc <= cutoff_at_utc
assert max_forecast_revision_issue_at_utc <= cutoff_at_utc
assert max_hourly_dispatch_at_utc <= cutoff_at_utc
assert max_hourly_available_at_utc <= cutoff_at_utc
assert max_hourly_observation_at_utc <= cutoff_at_utc where non-null
assert cutoff_at_hkt timezone == Asia/Hong_Kong
assert cutoff_at_utc is exact UTC conversion of cutoff_at_hkt
```

Emit violations to `leakage_audit.json` and fail the run if any violation count is nonzero.

### 9.2 No target-day label leakage

Assert:

```text
target_tmax_c for target_date appears only in y_true/residual_y/evaluation columns
no feature column name contains target_tmax_c without lag/roll/clim prefix
no raw Daily Extract absolute_daily_max_c column enters feature matrix
all target lag source dates <= target_date - 2 by default
```

Implement a feature lineage table:

```text
feature_name
source_table
source_time_column
max_source_time_utc_per_row
eligibility_rule
uses_target_label_boolean
minimum_lag_days
```

Fail if any predictor has `uses_target_label_boolean=true` and `minimum_lag_days < 2` in phase one.

### 9.3 No post-cutoff hourly reading leakage

For every row, record:

```text
max_hourly_dispatch_at_hkt
max_hourly_observation_at_hkt
latest_hourly_source_url
hourly_feature_window_start_hkt
hourly_feature_window_end_hkt
```

Fail if:

```text
max_hourly_dispatch_at_hkt > cutoff_at_hkt
max_hourly_observation_at_hkt > cutoff_at_hkt
any target-date T observation is used in pre-target-day profiles
```

### 9.4 No sealed-data training or tuning leakage

Before final sealed confirmation:

```text
assert no training label date >= 2024-01-01
assert no validation/model-selection date >= 2024-01-01
assert no imputer/scaler/clipping/calibration/ensemble-weight fitting uses date >= 2024-01-01
assert no cutoff selection uses date >= 2024-01-01
assert no feature-family retention decision uses sealed scores
```

For online live replay using lagged sealed labels, assert that:

```text
model artifacts were frozen before sealed evaluation
sealed lag rows are strictly before the prediction target date
sealed lag rows pass publication availability rules
sealed lag mode is reported separately
```

### 9.5 No future normalization leakage

Tests:

```text
all climatology features have max_source_local_date <= target_date - 2
all rolling features have window_end_local_date <= target_date - 2 for target labels
all hourly climatologies, if used, are expanding-past or fold-local
all scalers/imputers are fit only on training fold rows
all winsorization/clipping thresholds are fit only on training fold rows
```

Fail if any preprocessing artifact is fit on the full dataset before splitting.

### 9.6 No raw Daily Extract predictor use

Hard reject any feature lineage with:

```text
source_table = raw_audit.codex_audit_ds_01_hko_daily_tmax_target_hko_daily_extr_23cb54da
source_field = absolute_daily_max_c
source_field = local_date for target-date raw payload rows used as predictors
```

The raw audit table may appear only in:

```text
row count audit
canonical-label reconciliation report
leakage forbidden-source test
```

### 9.7 No apples-to-oranges forecast source mixing

Assert for all anchor/revision rows:

```text
source = 'info_gov'
product_type = 'local'
row_quality_status = 'usable_local_minmax'
target_issue_lead_days = 1
source_url contains info.gov.hk/gia/wr
```

Fail if any primary anchor row comes from:

```text
HKO 9-day API
HKO fnd
weather widget
airport forecast
5-day/7-day/9-day bulletin-only rows
general web source
manual New Territories adjustment
```

### 9.8 Duplicate-row checks

Forecast duplicates:

```text
one selected anchor row per target_date/cutoff_profile
no duplicate selected issue rows unless deterministic tie-breaker chooses one
revision path deduplicated by raw_sha256/source_url/issue_at_utc as specified
```

Target duplicates:

```text
one canonical label per target_date
raw payload duplicates never enter y or X
```

Hourly duplicates:

```text
source_url unique
if multiple rows have same dispatch/observation, aggregate deterministically or use latest source_url tie-breaker
```

### 9.9 Joined-row count checks

Emit row counts at every stage:

```text
target_rows_available_by_split
forecast_rows_raw_total
forecast_rows_strict_eligible_total
forecast_target_dates_with_anchor
forecast_target_dates_without_anchor
hourly_rows_raw_total
hourly_rows_eligible_by_cutoff
hourly_rows_used_in_latest_features
hourly_rows_used_in_window_features
joined_rows_before_feature_generation
joined_rows_after_feature_generation
scored_rows_by_split
scored_rows_by_ablation
missing_feature_counts_by_family
outlier_station_values_removed_by_station_year
sealed_rows_scored
sealed_rows_excluded_and_reason
```

The official baseline and all ablations must have identical scored row counts for a given cutoff profile and split.

### 9.10 Missingness reports

Produce:

```text
feature_missingness_report.csv with train/validation/test missing percentages
station_missingness_by_year.csv
hourly_latest_age_distribution.csv
forecast_revision_count_distribution.csv
text_flag_prevalence_by_year.csv
target_history_missingness_by_feature.csv
```

Any feature with >80% missingness in the main 2000-2023 matrix must be justified or moved to phase two, unless it is a sparse but important regime flag such as tropical cyclone position.

---

## 10. Expected Result And Risk Assessment

### Realistic expected MAE range

The `<=0.67 C` target is aggressive. It is possible only if the new hourly readings and station-network features explain a large part of official forecast residual error. The previous official-only residual correction improved MAE by only about `0.006 C`, which proves that simple historical bias correction is almost exhausted.

Expected outcomes:

```text
best_case_mae: 0.64-0.70 C
  Requires strong, stable incremental signal from HKO hourly state, station-network contrasts, and warning regimes.

realistic_good_mae: 0.75-0.83 C
  A meaningful improvement if hourly/network features reduce station-specific and convective residuals without tail harm.

conservative_realistic_mae: 0.84-0.89 C
  Likely if official forecast already incorporates most available weather information and hourly features only help certain regimes.

no_promote_mae: >0.89 C or improvement <0.035 C
  Cosmetic improvement; do not claim a successful ML edge.
```

### Best-case path to `<=0.67 C`

The target could be reached if these are true:

1. Latest pre-target HKO temperature/RH and dewpoint identify systematic official under/overprediction in warm humid regimes.
2. Inland-versus-coastal and urban-versus-marine gradients strongly predict when HKO will undershoot New Territories heat or overshoot under maritime suppression.
3. Thunderstorm/rain/tropical-cyclone text flags reduce tail errors on official high-error days.
4. Forecast revision momentum encodes late forecaster uncertainty not present in the latest integer max alone.
5. Residual corrections remain stable across 2022-2023 and 2024+ rather than overfitting 2000s-era source behavior.

### Why the 0.25 C improvement target might fail

It may fail because:

- The official local forecast already uses the same observed HKO and station-network information manually.
- The local forecast max is integer-like and designed for urban Hong Kong; residual noise around HKO daily extract may be partly irreducible at this horizon.
- Hourly readings before target day may not capture target-day cloud/rain timing, which is a key driver of daily max.
- Thunderstorm and shower timing is noisy and may not be predictable from previous-evening signals.
- Neighbor station readings are integer Celsius, sparse by station era, and sometimes contain source anomalies.
- The 2024+ weather regime may drift relative to 2000-2023.
- A model can improve normal days by tiny corrections but worsen rare convective/tropical-cyclone tails; this must be rejected.

### Highest-risk feature assumptions

| Feature assumption | Risk | Required control |
|---|---|---|
| HKO pre-target hourly state predicts target-day Tmax residual | May be mostly persistence already captured by official forecast | A3 ablation and year-by-year validation |
| Neighbor gradients add station-local information | Can overfit station panel changes or source anomalies | Group features first, sparse station features gated, outlier reports |
| Text flags help tails | Sparse regimes may not improve full-sample MAE | Slice-specific gates and no overall harm rule |
| Target climatology helps residuals | Official forecast already encodes climatology | Keep compact; use as normalization and ablate |
| 23:59 cutoff is best | Later cutoff may not be operationally available for all decisions | Separate cutoff profiles and report information-set differences |
| Sealed online lags are fair | Can look like sealed leakage if not audited | Report sealed_blind and online_live_replay separately |

### What to try next if phase one only marginally improves MAE

If A7 improves MAE by less than `0.035 C`:

1. Do not promote the model.
2. Inspect official top-20-percent error days and manually classify error mechanisms using already-documented sources.
3. Run phase-two station group temporal trends and cyclone geometry only if error analysis points to those regimes.
4. Test regime-specific residual experts with a conservative router:
   - warm wet / thunderstorm regime;
   - hot sunny / very-hot regime;
   - cool monsoon/frontal regime;
   - tropical cyclone / strong monsoon regime.
5. Add all 27 station sparse features only if group features show signal.
6. Consider external NWP/upper-air/ISD only after a separate point-in-time availability contract is documented.
7. Reassess whether the desired `0.25 C` MAE improvement is statistically attainable from the documented data.

---

## 11. Final Codex Build Task

### Task title

Implement a leakage-free HKG Daily Tmax residual ML research pipeline using Info.gov official forecasts, Info.gov hourly readings, target-history climatology, compact station-network features, honest walk-forward validation, and full ablation reporting.

### Modules/scripts to create or modify

Create these modules in principle. Use the repository’s existing conventions if names differ, but preserve responsibilities exactly.

```text
scripts/run_hkg_tmax_residual_ml_strategy.py
src/hkg_tmax/data/forecast_anchor.py
src/hkg_tmax/data/hourly_readings_features.py
src/hkg_tmax/data/target_history_features.py
src/hkg_tmax/features/feature_registry.py
src/hkg_tmax/features/text_regime_flags.py
src/hkg_tmax/features/station_groups.py
src/hkg_tmax/features/leakage_guards.py
src/hkg_tmax/modeling/baselines.py
src/hkg_tmax/modeling/residual_models.py
src/hkg_tmax/modeling/ensemble.py
src/hkg_tmax/evaluation/metrics.py
src/hkg_tmax/evaluation/ablation_runner.py
src/hkg_tmax/evaluation/reporting.py
configs/hkg_tmax/residual_ml_strategy.yaml
```

If the repo already has equivalent modules, modify those rather than duplicating. The implementation must still produce the artifacts listed below.

### Data extraction steps

1. Build a target date table:

```sql
SELECT local_date AS target_date, target_tmax_c AS y_true_c, 'label_core' AS label_source
FROM label_core.hko_daily_tmax
WHERE local_date BETWEEN '2000-01-02' AND '2023-12-31'
UNION ALL
SELECT local_date AS target_date, target_tmax_c AS y_true_c, 'sealed_confirmation' AS label_source
FROM sealed_confirmation.hko_daily_tmax
WHERE local_date BETWEEN '2024-01-01' AND '2026-05-31';
```

2. Generate cutoff rows for each target date and cutoff profile:

```text
cutoff_at_hkt = target_date - 1 day + profile_time_hkt
cutoff_at_utc = cutoff_at_hkt converted from Asia/Hong_Kong to UTC
```

3. Extract strict eligible forecast rows:

```sql
SELECT *
FROM public.hko_historical_forecasts_2000_2026
WHERE source = 'info_gov'
  AND product_type = 'local'
  AND row_quality_status = 'usable_local_minmax'
  AND target_issue_lead_days = 1
  AND forecast_max_c IS NOT NULL
  AND forecast_min_c IS NOT NULL
  AND issue_at_utc IS NOT NULL
  AND target_date IS NOT NULL;
```

4. For every target/cutoff, select latest eligible anchor and revision path as defined above.
5. Extract hourly readings only through cutoff from `public.hko_info_gov_hourly_readings_1998_2026`.
6. Parse station JSONB using `jsonb_array_elements` or a Python extraction layer; preserve missing/outlier flags.
7. Compute target-history features from `feature_safe.hko_target_history_pre2024` for pre-2024 development and from the explicit sealed policy for final confirmation.
8. Join all features to the fixed official-anchor row set.
9. Emit feature lineage metadata.

### Model training steps

For each cutoff profile, with `tminus1_2359` first:

1. Build the feature matrix for `2000-01-02` through `2023-12-31`.
2. Run leakage guards before modeling.
3. Score baselines A0 and A1.
4. Train M1-M4 residual models on the rolling folds.
5. Generate out-of-fold residual predictions.
6. Run ablations A2-A7 in order.
7. Choose retained feature families using only rolling validation and pre-sealed holdout rules.
8. Fit ensemble weights and shrinkage using pre-2024 validation predictions only.
9. Freeze feature set, hyperparameters, preprocessing, clipping, ensemble weights, shrinkage, and cutoff decision.
10. Retrain final models on all pre-2024 data through `2023-12-31`.
11. Evaluate 2024+ sealed confirmation once in both `sealed_blind_mode` and, if implemented, `online_live_replay_mode`.
12. Produce all scoreboards, predictions, and audit artifacts.

### Conceptual run commands

Use commands equivalent to:

```bash
python scripts/run_hkg_tmax_residual_ml_strategy.py \
  --config configs/hkg_tmax/residual_ml_strategy.yaml \
  --cutoff-profile tminus1_2359 \
  --start-date 2000-01-02 \
  --presealed-end-date 2023-12-31 \
  --sealed-start-date 2024-01-01 \
  --sealed-end-date 2026-05-31 \
  --run-ablation full \
  --output-dir experiments/hkg_tmax_residual_ml_strategy/results/tminus1_2359

python scripts/run_hkg_tmax_residual_ml_strategy.py \
  --config configs/hkg_tmax/residual_ml_strategy.yaml \
  --cutoff-profiles tminus1_1500,tminus1_1800,tminus1_2100,tminus1_2359 \
  --run-ablation cutoff_sensitivity \
  --output-dir experiments/hkg_tmax_residual_ml_strategy/results/cutoff_sensitivity
```

The actual repo command format may differ, but the run must accept cutoff profile, date range, ablation mode, and output directory as explicit arguments.

### Required artifacts

Create:

```text
experiments/hkg_tmax_residual_ml_strategy/results/
  README.md
  row_count_audit.json
  leakage_audit.json
  source_eligibility_audit.csv
  feature_matrix_trainval.parquet
  feature_matrix_presealed_holdout.parquet
  feature_matrix_sealed_confirmation.parquet
  feature_matrix_schema.json
  feature_lineage.json
  feature_missingness_report.csv
  station_missingness_by_year.csv
  station_outlier_report.csv
  forecast_revision_count_distribution.csv
  hourly_latest_age_distribution.csv
  text_flag_prevalence_by_year.csv
  target_history_feature_audit.csv
  scoreboard.csv
  scoreboard_by_split.csv
  scoreboard_by_month.csv
  scoreboard_by_regime.csv
  ablation_scoreboard.csv
  cutoff_sensitivity_scoreboard.csv
  prediction_rows.parquet
  prediction_rows.csv
  model_selection_log.json
  ensemble_weights.json
  clipping_and_preprocessing_audit.json
  feature_importance_lgbm.csv
  feature_importance_catboost.csv
  linear_coefficients.csv
  residual_error_diagnostics.md
  final_model_card.md
```

### Acceptance criteria

The implementation is accepted only if all are true:

```text
1. All leakage guards pass with zero violations.
2. The row-count audit reconciles target rows, forecast rows, hourly rows, joined rows, scored rows, and dropped rows.
3. A0 raw official baseline is scored on the exact same rows as all ablations for each cutoff profile.
4. The prior grouped residual baseline is recreated or explicitly explained if row differences prevent exact reproduction.
5. Feature-family ablations are reported in the required order.
6. No 2024+ sealed labels influence feature choice, hyperparameters, preprocessing, calibration, clipping, cutoff selection, or ensemble weights.
7. Final sealed confirmation is run only after model freeze.
8. Final report states whether the result is stretch_success, credible_major_improvement, or no_promote_cosmetic.
9. If MAE improvement is less than 0.035 C, the report refuses to claim a meaningful ML edge.
10. If final MAE is above 0.67 C, the report states honestly that the stretch target was not achieved and identifies which feature families did or did not move MAE.
```

### Final reporting language requirement

The final model card must contain these exact decision statements:

```text
Primary benchmark cutoff: T-1 23:59 HKT.
Primary baseline: latest eligible Info.gov LOCAL WEATHER FORECAST max before cutoff.
Primary target: HKO Daily Extract Absolute Daily Max (deg. C).
Primary model target: residual versus official forecast max.
Primary metric: MAE, with RMSE and p90 absolute error as guardrails.
Sealed confirmation rows were not used for model selection.
No post-cutoff forecast or hourly observation was used.
Raw Daily Extract payload rows were not used as predictors.
```

### Final leakage-free implementation contract

Every predictor in this system must be reconstructable as of the selected decision cutoff. Every row in the training, validation, pre-sealed holdout, sealed confirmation, and live-inference matrices must carry an auditable:

```text
target_date
cutoff_profile
cutoff_at_hkt
cutoff_at_utc
selected_forecast_source_url
selected_forecast_issue_at_hkt
selected_forecast_issue_at_utc
latest_hourly_dispatch_at_hkt_used
latest_hourly_observation_at_hkt_used
feature_source_tables
feature_eligibility_rule
feature_lineage_record
```

Codex must reject target-day Daily Extract values as predictors, reject raw target payload rows as predictors, reject post-cutoff hourly readings, reject post-cutoff forecast issuances, reject future target-history lags, reject whole-history normalization, reject sealed-data tuning leakage, reject rows whose forecast source is not the documented Info.gov apples-to-apples local forecast source, reject duplicate target-date rows unless the deterministic selection rule chooses exactly one eligible row, and emit a row-count audit proving how many rows entered each feature family and each evaluation split.

No reported improvement is valid unless it is both leakage-free and produced by features deliberately engineered for this exact HKO station problem. Feature engineering is the main battlefield: Codex must build the smallest high-signal feature matrix that reflects official forecast residuals, forecast revision dynamics, HKO pre-cutoff thermal and humidity state, neighbor-station spatial structure, maritime/inland and urban/coastal gradients, warning/rain/lightning/thunderstorm/tropical-cyclone regimes, target-memory climatology, seasonality, modern-era drift, and missingness. These are not vague inspirations; they are the required feature groups, transformations, and ablations that decide whether the system genuinely improves MAE and RMSE. If a feature cannot be proven available before cutoff, it must be excluded from the primary model no matter how predictive it appears.
