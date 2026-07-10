# HKG T+24 Full Strategy Implementation Blueprint — Final Consistency Patch

**File name:** `HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT_FINAL_CONSISTENCY_PATCH.md`  
**Status:** Binding implementation addendum.  
**Applies to:**

1. `HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT.md`
2. `HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT_COMPLETION_SPEC.md`
3. `HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT_COMPLETION_SPEC_FINAL_CLARIFICATIONS.md`

This patch is the final consistency layer. Wherever this patch is more specific than earlier documents, this patch wins. Wherever earlier documents conflict with this patch, Codex must implement this patch exactly.

Codex must not make architecture, naming, schema, data-selection, modelling, validation, or workflow decisions beyond what is written here.

---

## 0. Binding precedence and implementation rule

The implementation contract now consists of four documents in this exact precedence order:

1. **This patch** — `HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT_FINAL_CONSISTENCY_PATCH.md`
2. `HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT_COMPLETION_SPEC_FINAL_CLARIFICATIONS.md`
3. `HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT_COMPLETION_SPEC.md`
4. `HKG_T24_FULL_STRATEGY_IMPLEMENTATION_BLUEPRINT.md`

When two documents disagree, Codex must implement the highest-precedence document.

The first full implementation remains a strict H24N historical + replay + live-shadow implementation. H24N means the decision is made at **15:00 HKT on T−1**, with an operational data freeze at **14:45 HKT on T−1 / 06:45 UTC on T−1**.

---

# 1. Target-memory lag naming contradiction — final decision

## 1.1 Final decision

Codex must implement **Approach A**.

All finalized HKO daily target-memory features must use explicit safe-lag naming. The latest finalized daily HKO Tmax value allowed as a model feature for H24N is:

```text
target__lag2_tmax_c
```

This means the finalized daily Tmax from calendar date **T−2**.

The following feature names are forbidden and must not appear in any feature matrix, model artifact, report, or test expectation:

```text
target__lag1_tmax_c
target__roll7_mean_lag1_c
target__roll14_mean_lag1_c
target__roll30_mean_lag1_c
target__volatility_14_lag1_c
target__hot_spell_length_lag1_days
target_lag1_final_tmax_c
```

The string `lag1` must not be used for finalized daily target labels in the first full implementation.

Rationale: at 15:00 HKT on T−1, the finalized daily maximum temperature for T−1 is not allowed by default. It may be physically partly known intraday, but it is not a finalized safe daily label. The finalized daily target-memory feature boundary is T−2.

## 1.2 Canonical target-memory feature whitelist

The strict target-memory feature set is exactly the following. Additional target-memory features are out of scope for the first full implementation.

| Feature name | Formula | Required minimum finalized history | NULL behavior |
|---|---|---:|---|
| `target__lag2_tmax_c` | `target_tmax_c[T-2]` | 2 days | NULL until available |
| `target__lag3_tmax_c` | `target_tmax_c[T-3]` | 3 days | NULL until available |
| `target__lag7_tmax_c` | `target_tmax_c[T-7]` | 7 days | NULL until available |
| `target__lag14_tmax_c` | `target_tmax_c[T-14]` | 14 days | NULL until available |
| `target__lag30_tmax_c` | `target_tmax_c[T-30]` | 30 days | NULL until available |
| `target__lag60_tmax_c` | `target_tmax_c[T-60]` | 60 days | NULL until available |
| `target__lag365_tmax_c` | `target_tmax_c[T-365]` | 365 days | NULL until available |
| `target__roll7_mean_lag2_c` | mean of `target_tmax_c[T-8:T-2]` inclusive | 7 values | NULL until full window |
| `target__roll14_mean_lag2_c` | mean of `target_tmax_c[T-15:T-2]` inclusive | 14 values | NULL until full window |
| `target__roll30_mean_lag2_c` | mean of `target_tmax_c[T-31:T-2]` inclusive | 30 values | NULL until full window |
| `target__roll60_mean_lag2_c` | mean of `target_tmax_c[T-61:T-2]` inclusive | 60 values | NULL until full window |
| `target__roll365_mean_lag2_c` | mean of `target_tmax_c[T-366:T-2]` inclusive | 365 values | NULL until full window |
| `target__roll7_std_lag2_c` | population std, ddof=0, over `T-8:T-2` | 7 values | NULL until full window |
| `target__roll14_std_lag2_c` | population std, ddof=0, over `T-15:T-2` | 14 values | NULL until full window |
| `target__roll30_std_lag2_c` | population std, ddof=0, over `T-31:T-2` | 30 values | NULL until full window |
| `target__range7_lag2_c` | max minus min over `T-8:T-2` | 7 values | NULL until full window |
| `target__range14_lag2_c` | max minus min over `T-15:T-2` | 14 values | NULL until full window |
| `target__slope7_lag2_c_per_day` | ordinary least squares slope over days `T-8:T-2`; x = 0..6 | 7 values | NULL until full window |
| `target__slope30_lag2_c_per_day` | ordinary least squares slope over days `T-31:T-2`; x = 0..29 | 30 values | NULL until full window |
| `target__slope7_minus_slope30_lag2_c_per_day` | `target__slope7_lag2_c_per_day - target__slope30_lag2_c_per_day` | both slopes | NULL until both available |
| `target__lag2_minus_roll7_c` | `target__lag2_tmax_c - target__roll7_mean_lag2_c` | lag2 + roll7 | NULL until both available |
| `target__lag2_minus_roll30_c` | `target__lag2_tmax_c - target__roll30_mean_lag2_c` | lag2 + roll30 | NULL until both available |
| `target__roll7_minus_roll30_c` | `target__roll7_mean_lag2_c - target__roll30_mean_lag2_c` | roll7 + roll30 | NULL until both available |
| `target__hot_spell_length_lag2_days` | count of consecutive finalized days ending at T−2 with `target_tmax_c >= 30.0`; stops at first prior day `< 30.0` or missing | lag2 available | NULL if lag2 unavailable |
| `target__cool_spell_length_lag2_days` | count of consecutive finalized days ending at T−2 with `target_tmax_c <= 15.0`; stops at first prior day `> 15.0` or missing | lag2 available | NULL if lag2 unavailable |
| `target__clim30_mean_c` | causal target-date climatology for target date T; use prior 30 calendar years only, years `< year(T)`, same day-of-year ±15 local-calendar days; leap day maps to Feb 28/Mar 1 window; full details below | at least 10 prior values | NULL until available |
| `target__clim30_std_c` | population std, ddof=0, over the same sample used for `target__clim30_mean_c` | at least 10 prior values | NULL until available |
| `target__lag2_minus_clim30_c` | `target__lag2_tmax_c - causal climatology for local date T-2 using same clim30 method anchored on T-2` | lag2 + clim30(T−2) | NULL until both available |
| `target__warming_trend_10y_c_per_year` | OLS slope of annual mean HKO Tmax over the 10 complete calendar years ending before `year(T)`; x in calendar years; use only finalized labels available by T−2 | at least 8 complete annual means | NULL until available |
| `target__year_index` | `year(target_date_hkt) - 2000` | target date available | never NULL |

## 1.3 Exact clim30 construction

For a row with `target_date_hkt = T`:

1. Let `doy_window(T)` be local calendar dates whose day-of-year is within ±15 of `day_of_year(T)`, with circular year wrap.
2. Candidate climatology rows must satisfy:
   - `local_date < date_trunc('year', T)`; this prevents using the target year.
   - `local_date <= T - interval '2 days'`; this preserves the H24N finalized-label cutoff.
   - `local_date` is within the day-of-year window.
   - `local_date` is in the most recent 30 prior calendar years if more than 30 years exist.
3. If fewer than 10 non-null target labels remain, set `target__clim30_mean_c` and `target__clim30_std_c` to NULL.
4. Otherwise compute mean and population std ddof=0.

## 1.4 Missing indicators for target-memory features

For every target-memory feature above except `target__year_index`, Codex must also generate a strict missing indicator named:

```text
<feature_name>__is_missing
```

Example:

```text
target__lag2_tmax_c__is_missing
```

Value is boolean. True means the feature is NULL before model imputation.

## 1.5 Imputation for model matrices

In the stored `features_jsonb`, missing numeric features remain JSON null and the missing indicator is true. In exported model matrices, numeric NULLs are imputed fold-locally using the training-fold median for that feature. If a feature is all NULL in a fold, it is dropped from that fold and listed in the fold artifact `dropped_features_all_null.csv`.

## 1.6 Synthetic fixture expectations

All tests must be updated as follows.

For a synthetic target-label table with exactly 120 consecutive daily labels and no missing target values:

| Test | Expected valid non-null rows |
|---|---:|
| `target__lag2_tmax_c` | 118 |
| `target__lag3_tmax_c` | 117 |
| `target__lag7_tmax_c` | 113 |
| `target__roll7_mean_lag2_c` | 112 |
| `target__roll14_mean_lag2_c` | 105 |
| `target__roll30_mean_lag2_c` | 89 |
| `target__slope7_lag2_c_per_day` | 112 |
| `target__slope30_lag2_c_per_day` | 89 |
| `target__hot_spell_length_lag2_days` | 118 |

Any test expecting `119 target-memory lag1 rows` is obsolete and must be replaced with the table above.

---

# 2. Feature matrix table naming conflict — final canonical design

## 2.1 Final decision

Codex must implement one canonical feature matrix table:

```text
model_features.feature_matrix
```

The prior tables:

```text
model_features.snapshot_feature_matrix_strict
model_features.snapshot_feature_matrix_proxy
```

are replaced by compatibility views. They must not be physical tables in the final implementation.

## 2.2 Final DDL

```sql
CREATE SCHEMA IF NOT EXISTS model_features;

CREATE TABLE IF NOT EXISTS model_features.feature_matrix (
    target_date_hkt date NOT NULL,
    cutoff_id text NOT NULL,
    feature_scope text NOT NULL,
    schema_version text NOT NULL,
    snapshot_id text NOT NULL,
    features_jsonb jsonb NOT NULL,
    feature_count integer NOT NULL,
    generated_at_utc timestamptz NOT NULL DEFAULT now(),
    source_hash text NOT NULL,
    leakage_status text NOT NULL,
    matrix_status text NOT NULL DEFAULT 'active',
    PRIMARY KEY (target_date_hkt, cutoff_id, feature_scope, schema_version),
    CONSTRAINT feature_matrix_scope_chk CHECK (feature_scope IN ('strict', 'proxy', 'live_shadow')),
    CONSTRAINT feature_matrix_leakage_chk CHECK (leakage_status IN ('passed', 'failed_closed')),
    CONSTRAINT feature_matrix_status_chk CHECK (matrix_status IN ('active', 'superseded', 'failed_closed'))
);

CREATE INDEX IF NOT EXISTS ix_feature_matrix_snapshot_id
    ON model_features.feature_matrix (snapshot_id);

CREATE INDEX IF NOT EXISTS ix_feature_matrix_schema_scope
    ON model_features.feature_matrix (schema_version, feature_scope);

CREATE INDEX IF NOT EXISTS ix_feature_matrix_features_gin
    ON model_features.feature_matrix USING gin (features_jsonb);
```

## 2.3 Compatibility views

```sql
CREATE OR REPLACE VIEW model_features.snapshot_feature_matrix_strict AS
SELECT *
FROM model_features.feature_matrix
WHERE feature_scope = 'strict';

CREATE OR REPLACE VIEW model_features.snapshot_feature_matrix_proxy AS
SELECT *
FROM model_features.feature_matrix
WHERE feature_scope = 'proxy';
```

Earlier references to `snapshot_feature_matrix_strict` and `snapshot_feature_matrix_proxy` must be interpreted as these views.

## 2.4 Migration behavior

If physical tables named `model_features.snapshot_feature_matrix_strict` or `model_features.snapshot_feature_matrix_proxy` already exist, Codex must:

1. Create a backup table with suffix `_backup_YYYYMMDDHH24MISS`.
2. Copy the old rows into `model_features.feature_matrix` with `feature_scope='strict'` or `feature_scope='proxy'` as appropriate.
3. Drop the old physical table.
4. Create the compatibility view.
5. Write a migration report to:

```text
reports/schema_migration_feature_matrix.md
```

---

# 3. Database environment variable conflict — final rule

Codex must support both environment variables, with this priority order:

1. `HKG_TMAX_DATABASE_URL`
2. fallback to `HKG_TMAX_DB_DSN`

If both exist, `HKG_TMAX_DATABASE_URL` wins and Codex must log:

```text
Using HKG_TMAX_DATABASE_URL; HKG_TMAX_DB_DSN is present but ignored.
```

If neither exists, every database command must fail before doing any work with this exact error message:

```text
ERROR: Database DSN not configured. Set HKG_TMAX_DATABASE_URL or HKG_TMAX_DB_DSN. HKG_TMAX_DATABASE_URL has priority when both are present.
```

The database connection resolver module must be:

```text
src/hkg_t24/db/connection.py
```

Required function:

```python
get_database_url() -> str
```

---

# 4. Undefined live component table — final DDL and behavior

## 4.1 Required table

`model_live.live_prediction_component` is required in the first full implementation.

It stores component-level rows for live predictions only. Historical OOF expert predictions remain in `model_oof.expert_prediction`. Historical final-system component replay rows use `model_eval.system_prediction_component`, defined below.

## 4.2 `model_live.prediction` final DDL

```sql
CREATE SCHEMA IF NOT EXISTS model_live;

CREATE TABLE IF NOT EXISTS model_live.prediction (
    prediction_id uuid PRIMARY KEY,
    target_date_hkt date NOT NULL,
    cutoff_id text NOT NULL,
    snapshot_id text NOT NULL,
    schema_version text NOT NULL,
    forecast_tmax_c numeric(6,3),
    forecast_tmax_rounded_c numeric(4,1),
    p10_tmax_c numeric(6,3),
    p25_tmax_c numeric(6,3),
    p50_tmax_c numeric(6,3),
    p75_tmax_c numeric(6,3),
    p90_tmax_c numeric(6,3),
    expected_abs_error_c numeric(6,3),
    confidence_state text NOT NULL,
    no_trade_flag boolean NOT NULL,
    no_trade_reason text,
    produced_at_utc timestamptz NOT NULL DEFAULT now(),
    input_freeze_utc timestamptz NOT NULL,
    model_candidate_id text NOT NULL,
    run_mode text NOT NULL DEFAULT 'live',
    status text NOT NULL DEFAULT 'active',
    CONSTRAINT live_prediction_cutoff_chk CHECK (cutoff_id = 'H24N'),
    CONSTRAINT live_prediction_run_mode_chk CHECK (run_mode IN ('live', 'prospective_replay')),
    CONSTRAINT live_prediction_status_chk CHECK (status IN ('active', 'superseded', 'failed_closed')),
    UNIQUE (target_date_hkt, cutoff_id, model_candidate_id, run_mode)
);
```

## 4.3 `model_live.live_prediction_component` final DDL

```sql
CREATE TABLE IF NOT EXISTS model_live.live_prediction_component (
    prediction_id uuid NOT NULL REFERENCES model_live.prediction(prediction_id) ON DELETE CASCADE,
    component_id text NOT NULL,
    component_type text NOT NULL,
    expert_id text,
    router_id text,
    specialist_id text,
    raw_prediction_tmax_c numeric(6,3),
    corrected_prediction_tmax_c numeric(6,3),
    expected_error_c numeric(6,3),
    weight numeric(10,8),
    correction_c numeric(6,3),
    is_shadow boolean NOT NULL DEFAULT false,
    is_placeholder boolean NOT NULL DEFAULT false,
    availability_status text NOT NULL,
    unavailable_reason text,
    source_scope text NOT NULL,
    metadata_jsonb jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at_utc timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (prediction_id, component_id),
    CONSTRAINT live_component_type_chk CHECK (component_type IN ('expert', 'router', 'specialist', 'distribution', 'fallback')),
    CONSTRAINT live_availability_chk CHECK (availability_status IN ('available', 'unavailable', 'blocked', 'demoted', 'placeholder'))
);

CREATE INDEX IF NOT EXISTS ix_live_component_expert
    ON model_live.live_prediction_component (expert_id);

CREATE INDEX IF NOT EXISTS ix_live_component_status
    ON model_live.live_prediction_component (availability_status);
```

## 4.4 Historical replay component table

```sql
CREATE SCHEMA IF NOT EXISTS model_eval;

CREATE TABLE IF NOT EXISTS model_eval.system_prediction_component (
    target_date_hkt date NOT NULL,
    cutoff_id text NOT NULL,
    evaluation_scope text NOT NULL,
    candidate_id text NOT NULL,
    component_id text NOT NULL,
    component_type text NOT NULL,
    expert_id text,
    router_id text,
    specialist_id text,
    raw_prediction_tmax_c numeric(6,3),
    corrected_prediction_tmax_c numeric(6,3),
    expected_error_c numeric(6,3),
    weight numeric(10,8),
    correction_c numeric(6,3),
    is_shadow boolean NOT NULL DEFAULT false,
    is_placeholder boolean NOT NULL DEFAULT false,
    availability_status text NOT NULL,
    unavailable_reason text,
    metadata_jsonb jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at_utc timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (target_date_hkt, cutoff_id, evaluation_scope, candidate_id, component_id)
);
```

## 4.5 Placeholder rows

For every snapshot, shadow experts must emit a placeholder component row when the source is absent, too short, blocked, or not yet promoted.

Live placeholders go to:

```text
model_live.live_prediction_component
```

Historical replay placeholders go to:

```text
model_oof.expert_prediction
model_eval.system_prediction_component
```

Required unavailable reason codes:

```text
SOURCE_TABLE_ABSENT
SOURCE_TOO_SHORT
NO_ELIGIBLE_ROWS_FOR_DATE
BLOCKED_SOURCE
NOT_PROMOTED
SEALED_LABELS_UNAVAILABLE
LIVE_COLLECTOR_NOT_STARTED
INSUFFICIENT_HISTORY
```

---

# 5. Calendar fields — final model-input policy

Calendar values are both metadata and model inputs, but the metadata columns themselves must not be used directly by models. Models may use only the explicit `calendar__*` feature keys listed here.

## 5.1 Metadata columns

These are identity/reporting fields only:

```text
season
month
day_of_year
is_mam
is_jja
is_son
is_djf
```

## 5.2 Calendar feature whitelist

The following strict features are model inputs:

| Feature | Formula |
|---|---|
| `calendar__month_sin1` | `sin(2*pi*month/12)` |
| `calendar__month_cos1` | `cos(2*pi*month/12)` |
| `calendar__doy_sin1` | `sin(2*pi*day_of_year/365.2425)` |
| `calendar__doy_cos1` | `cos(2*pi*day_of_year/365.2425)` |
| `calendar__is_mam` | 1 if month in 3,4,5 else 0 |
| `calendar__is_jja` | 1 if month in 6,7,8 else 0 |
| `calendar__is_son` | 1 if month in 9,10,11 else 0 |
| `calendar__is_djf` | 1 if month in 12,1,2 else 0 |
| `calendar__year_index` | `year(target_date_hkt) - 2000` |

`target__year_index` from the target-memory whitelist remains present for backward compatibility inside the target feature family. It must equal `calendar__year_index`. If the two values differ, the feature builder must fail closed.

---

# 6. `model_core.source_registry` final schema and required rows

## 6.1 Final DDL

`strict_status` is removed from the final design. Use explicit boolean status fields plus a canonical `source_role`.

```sql
CREATE SCHEMA IF NOT EXISTS model_core;

CREATE TABLE IF NOT EXISTS model_core.source_registry (
    source_code text PRIMARY KEY,
    source_family text NOT NULL,
    source_role text NOT NULL,
    feature_prefix text NOT NULL,
    strict_allowed boolean NOT NULL DEFAULT false,
    proxy_allowed boolean NOT NULL DEFAULT false,
    shadow_allowed boolean NOT NULL DEFAULT false,
    blocked boolean NOT NULL DEFAULT false,
    live_only boolean NOT NULL DEFAULT false,
    support_only boolean NOT NULL DEFAULT false,
    unit_semantics_verified boolean NOT NULL DEFAULT false,
    availability_grade text NOT NULL,
    source_time_policy text NOT NULL,
    min_target_date_hkt date,
    max_target_date_hkt date,
    required_source_scope text,
    blocker_reason text,
    promotion_gate text NOT NULL,
    notes text NOT NULL DEFAULT '',
    updated_at_utc timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT source_role_chk CHECK (source_role IN ('strict_core', 'strict_optional', 'proxy_research', 'shadow_challenger', 'live_shadow', 'support_only', 'blocked')),
    CONSTRAINT availability_grade_chk CHECK (availability_grade IN ('EXACT_VINTAGE', 'CONSERVATIVE_SCHEDULE', 'MODEL_RUN_TIME_PROXY_ONLY', 'DIAGNOSTIC_ONLY', 'LIVE_FIRST_SEEN_ONLY', 'BLOCKED')),
    CONSTRAINT source_status_consistency_chk CHECK (
        (blocked = true AND strict_allowed = false AND proxy_allowed = false AND shadow_allowed = false)
        OR blocked = false
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_source_registry_feature_prefix
    ON model_core.source_registry (feature_prefix);
```

## 6.2 Migration from older registry

If `model_core.source_registry` exists with `strict_status`, Codex must:

1. Add the final columns if missing.
2. Backfill final booleans from the mapping below.
3. Preserve `strict_status` as a deprecated column only if dropping it would break existing views.
4. Never read `strict_status` in new implementation code.
5. Write migration output to:

```text
reports/schema_migration_source_registry.md
```

Dedicated columns are required. JSONB is not acceptable for `unit_semantics_verified`, `strict_allowed`, `proxy_allowed`, `shadow_allowed`, `blocked`, `live_only`, or `support_only`.

## 6.3 Required `source_registry` rows

| source_code | source_family | source_role | feature_prefix | strict_allowed | proxy_allowed | shadow_allowed | blocked | live_only | support_only | unit_semantics_verified | availability_grade | required_source_scope | promotion_gate |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `hko_target_labels` | target | strict_core | `target` | true | false | false | false | false | false | true | EXACT_VINTAGE | NULL | always included as labels and lagged target memory only |
| `hko_official_forecasts` | official | strict_core | `official` | true | false | false | false | false | false | true | EXACT_VINTAGE | NULL | always included when eligible row exists |
| `calendar` | deterministic | strict_core | `calendar` | true | false | false | false | false | false | true | EXACT_VINTAGE | NULL | always included |
| `gfs` | gribstream | strict_core | `gfs` | true | false | false | false | false | false | true | CONSERVATIVE_SCHEDULE | `full_tactical_backfill_ok_tmax` | core strict expert E4 |
| `gefsatmosmean` | gribstream | strict_core | `gefsmean` | true | false | false | false | false | false | true | CONSERVATIVE_SCHEDULE | `full_tactical_backfill_ok_tmax` | core strict expert E5 context |
| `gefsatmos` | gribstream | strict_core | `gefsens` | true | false | false | false | false | false | true | CONSERVATIVE_SCHEDULE | `full_tactical_backfill_ok_tmax` | core strict expert E5 ensemble |
| `ifsoper` | gribstream | shadow_challenger | `ifsoper` | false | false | true | false | false | false | true | CONSERVATIVE_SCHEDULE | `full_tactical_backfill_ok_tmax` | may enter after sealed protocol |
| `ifsenfo` | gribstream | shadow_challenger | `ifsens` | false | false | true | false | false | false | true | CONSERVATIVE_SCHEDULE | `full_tactical_backfill_ok_tmax` | may enter after sealed protocol; member-0 caveat tracked |
| `cwawrf15` | gribstream | live_shadow | `cwawrf15` | false | false | true | false | true | false | true | LIVE_FIRST_SEEN_ONLY | `full_tactical_backfill_ok_tmax` | prospective only until two seasonal cycles |
| `aifsoper` | gribstream | shadow_challenger | `aifsoper` | false | false | true | false | false | false | true | CONSERVATIVE_SCHEDULE | `full_tactical_backfill_ok_tmax` | may enter after sealed protocol; capped |
| `aifsenfo` | gribstream | shadow_challenger | `aifsens` | false | false | true | false | false | false | true | CONSERVATIVE_SCHEDULE | `full_tactical_backfill_ok_tmax` | may enter after sealed protocol; capped |
| `aigfssfc` | gribstream | shadow_challenger | `aigfssfc` | false | false | true | false | false | false | true | CONSERVATIVE_SCHEDULE | `full_tactical_backfill_ok_tmax` | short-history shadow only |
| `aigfspres` | gribstream | support_only | `aigfspres` | false | false | false | false | false | true | true | CONSERVATIVE_SCHEDULE | `full_tactical_backfill_ok_tmax` | support only; no Tmax expert |
| `aigefssfc` | gribstream | blocked | `aigefssfc` | false | false | false | true | false | false | false | BLOCKED | `full_tactical_backfill_ok_tmax` | blocked until Tmax coverage fixed |
| `graphcast` | gribstream | shadow_challenger | `graphcast` | false | false | true | false | false | false | true | MODEL_RUN_TIME_PROXY_ONLY | `full_tactical_backfill_ok_tmax` | diagnostic shadow only |
| `fourcastnetgfs` | gribstream | shadow_challenger | `fourcastnet` | false | false | true | false | false | false | true | MODEL_RUN_TIME_PROXY_ONLY | `full_tactical_backfill_ok_tmax` | diagnostic shadow through observed archive end |
| `nbmoc` | gribstream | blocked | `nbmoc` | false | false | false | true | false | false | false | BLOCKED | `full_tactical_backfill_ok_tmax` | blocked; zero rows |
| `station_network_proxy` | station | proxy_research | `station` | false | true | false | false | false | false | false | DIAGNOSTIC_ONLY | NULL | proxy reports only until operational-vintage proof |
| `hko_daily_climate_proxy` | diagnostic | proxy_research | `climate` | false | true | false | false | false | false | false | DIAGNOSTIC_ONLY | NULL | proxy reports only; lagged diagnostic features |
| `igra_upper_air_proxy` | diagnostic | support_only | `igra` | false | true | false | false | false | true | false | DIAGNOSTIC_ONLY | NULL | diagnostic report only; no first-system expert |
| `tc_best_track_proxy` | diagnostic | support_only | `tc` | false | true | false | false | false | true | false | DIAGNOSTIC_ONLY | NULL | diagnostic report only; no live predictor |
| `arwf_live` | hko_live | live_shadow | `arwf` | false | false | true | false | true | false | true | LIVE_FIRST_SEEN_ONLY | NULL | prospective shadow only until two seasonal cycles |

---

# 7. GribStream source names and feature prefixes

The final mapping is binding:

| dataset_code | feature_prefix | expert_id | status |
|---|---|---|---|
| `gfs` | `gfs` | `E4_GFS_MOS` | strict_core |
| `gefsatmosmean` | `gefsmean` | `E5_GEFS_ENSEMBLE` | strict_core context |
| `gefsatmos` | `gefsens` | `E5_GEFS_ENSEMBLE` | strict_core ensemble |
| `ifsoper` | `ifsoper` | `E6_IFS_OPER_SHADOW` | shadow_challenger |
| `ifsenfo` | `ifsens` | `E7_IFS_ENS_SHADOW` | shadow_challenger |
| `cwawrf15` | `cwawrf15` | `E9_CWA_WRF_LIVE_SHADOW` | live_shadow |
| `aifsoper` | `aifsoper` | `E8_AIFS_OPER_SHADOW` | shadow_challenger |
| `aifsenfo` | `aifsens` | `E8_AIFS_ENS_SHADOW` | shadow_challenger |
| `aigfssfc` | `aigfssfc` | `E8_AIGFS_SFC_SHADOW` | shadow_challenger |
| `aigfspres` | `aigfspres` | none | support_only |
| `aigefssfc` | `aigefssfc` | none | blocked |
| `graphcast` | `graphcast` | `E8_GRAPHCAST_SHADOW` | shadow_challenger |
| `fourcastnetgfs` | `fourcastnet` | `E8_FOURCASTNET_SHADOW` | shadow_challenger |
| `nbmoc` | `nbmoc` | none | blocked |

Feature names must use these prefixes exactly. Example:

```text
gfs__center__tmax_c
gefsmean__center__pwat_kg_m2_mean
gefsens__center__tmax_p50_c
ifsoper__center__tmax_c
fourcastnet__center__tmax_c
```

---

# 8. Features referenced by specialists/router — final formulas

This section closes all feature-name gaps. Every feature below is required in the strict feature whitelist unless explicitly marked shadow/support.

## 8.1 NWP center shorthand

For all 12-point deterministic/mean models, `center` means `location_code='hko_center'`.

For each deterministic model with 2m temperature available, define:

```text
<prefix>__center__tmax_c
```

Formula:

1. Apply the leakage-safe NWP filter.
2. Use all safe rows for the dataset, target date, center location, and member 0.
3. Prefer `interval_tmax_2m_k` when present and valid.
4. Otherwise use max of `temperature_2m_k` over valid times.
5. Convert K to C using `value_k - 273.15`.

## 8.2 Required specialist/router gap features

| Feature | Required? | Formula |
|---|---:|---|
| `gfs__center__dewpoint_change_proxy_c` | yes | `gfs__center__dewpoint_14h_c - gfs__center__dewpoint_02h_c`; use nearest valid HKT times within ±1 hour. If either unavailable, NULL. |
| `official__psr_numeric_proxy` | yes | If official row has numeric PSR percent, divide by 100. Else map `psr_text`: `low=0.20`, `medium low=0.35`, `medium=0.50`, `medium high=0.65`, `high=0.80`. Else text heuristic: contains `thunderstorm` or `heavy rain` -> 0.70; contains `showers` -> 0.55; contains `isolated showers` -> 0.40; contains `rain` -> 0.60; contains `fine` or `sunny` and no rain/showers -> 0.15; else NULL. |
| `target__warming_trend_10y_c_per_year` | yes | Defined in Section 1.2. |
| `target__year_index` | yes | `year(target_date_hkt) - 2000`; must equal `calendar__year_index`. |
| `target__clim30_std_c` | yes | Defined in Section 1.2 and 1.3. |
| `gefsmean__center__pwat_kg_m2_mean` | yes | Mean of `pwat_kg_m2` over safe target-window rows at center for `gefsatmosmean`; if no PWAT rows, NULL. |
| `gfs__center__low_cloud_pct_mean` | yes | Mean `low_cloud_pct` over safe target-window rows at center. |
| `gfs__center__shortwave_w_m2_mean` | yes | Mean `downward_shortwave_w_m2` over safe daylight rows 08:00–18:00 HKT at center. |
| `gfs__center__precip_mm_sum` | yes | Difference accumulated precipitation within same run over target window; convert kg/m2 to mm 1:1; negative diffs below −0.01 are flagged and set NULL; tiny negative diffs in [−0.01,0) set to 0. |
| `gfs__center__wind_speed_10m_mean_mps` | yes | Mean `sqrt(u_wind_10m_mps^2 + v_wind_10m_mps^2)` over safe target-window rows at center. |
| `gfs__center__onshore_easterly_component_mps` | yes | Mean `-u_wind_10m_mps` over safe target-window rows at center, because positive easterly flow corresponds to negative U wind. |
| `gfs__center__temp_dewpoint_spread_mean_c` | yes | Mean `(temperature_2m_k - dewpoint_2m_k)` over safe target-window rows at center; Kelvin difference equals C difference. |
| `gfs__center__t850_c_mean` | yes | Mean `temperature_850_k - 273.15` over safe target-window rows at center. |
| `gfs__center__z500_m_mean` | yes | Mean `geopotential_height_500_m` over safe target-window rows at center. |
| `gefsens__center__tmax_p10_c` | yes | 10th percentile of member daily Tmax candidates. |
| `gefsens__center__tmax_p50_c` | yes | median of member daily Tmax candidates. |
| `gefsens__center__tmax_p90_c` | yes | 90th percentile of member daily Tmax candidates. |
| `gefsens__center__tmax_spread_p90_p10_c` | yes | `p90 - p10`. |
| `gefsens__center__prob_ge_30_0` through `gefsens__center__prob_ge_40_0` | yes | Fraction of ensemble member daily Tmax candidates >= threshold, thresholds 30.0 to 40.0 by 0.5. |

## 8.3 Feature whitelist inclusion rule

Any feature referenced by a router, expert, specialist, or distribution model must exist in exactly one feature dictionary file:

```text
reports/feature_dictionary_strict.csv
reports/feature_dictionary_proxy.csv
reports/feature_dictionary_shadow.csv
```

If code references a feature not found in those dictionaries, Phase 4 feature validation must fail closed.

---

# 9. Canonical date field naming

All new model schemas, feature tables, OOF tables, live tables, scoreboards, and reports must use:

```text
target_date_hkt
```

The following names are source-only aliases and must be mapped at ingestion or view creation:

| Source field | Canonical model field |
|---|---|
| `local_date` | `target_date_hkt` when the row is a target label or finalized daily target-side value |
| `target_date` | `target_date_hkt` |
| `forecast_date` | `target_date_hkt` when the row is forecast for a local date |
| `target_date_hkt` | `target_date_hkt` |

Reports must use `target_date_hkt`. The bare field name `target_date` is forbidden in new model schemas except as a source-column alias inside ingestion SQL comments.

---

# 10. LightGBM requirement — final rule

LightGBM is mandatory for the first full implementation.

Codex must add `lightgbm` to the project dependency manifest if it is missing. Phase 0 preflight must execute:

```python
import lightgbm
```

If import fails, Phase 0 fails with:

```text
ERROR: lightgbm is required for HKG T24 first full implementation. Install project dependencies before running the pipeline.
```

No `HistGradientBoostingRegressor` fallback is allowed in the first full implementation.

Allowed sklearn usage remains:

```text
linear models for simple baselines
isotonic regression for monotonic calibration
standard preprocessing utilities
metrics
```

All LightGBM scoreboards must record:

```text
model_library = lightgbm
model_library_version
```

---

# 11. ARWF and CWA live-shadow behavior

## 11.1 ARWF

ARWF is a required live-shadow component but not a required strict historical source.

If ARWF source tables are absent, Phase 0 records a warning, not a failure:

```text
WARNING: ARWF source table absent. E11_ARWF_LIVE_SHADOW will emit placeholder rows with SOURCE_TABLE_ABSENT.
```

For every snapshot, E11 must emit a prediction row:

| Condition | Row behavior |
|---|---|
| ARWF table absent | placeholder row, `availability_status='placeholder'`, `unavailable_reason='SOURCE_TABLE_ABSENT'`, prediction NULL, weight 0 |
| ARWF present but no eligible row for date | placeholder row, `unavailable_reason='NO_ELIGIBLE_ROWS_FOR_DATE'`, weight 0 |
| ARWF present and eligible but fewer than 365 settled live-shadow days | real direct prediction row, `is_shadow=true`, weight 0 |
| ARWF has at least 365 settled live-shadow days | still shadow in first full implementation; future gate required |

ARWF promotion is out of scope for first full implementation. Future gate: at least 730 settled live-shadow target dates spanning at least two complete MAM/JJA/SON/DJF cycles, then a separate addendum may authorize capped router entry.

## 11.2 CWA WRF

CWA WRF is a required live-shadow component.

If `cwawrf15` GribStream rows are absent, Phase 0 records a warning, not a failure:

```text
WARNING: cwawrf15 source absent or too short. E9_CWA_WRF_LIVE_SHADOW will emit placeholder rows.
```

For every snapshot:

| Condition | Row behavior |
|---|---|
| no `cwawrf15` source table or rows | placeholder, `SOURCE_TABLE_ABSENT`, weight 0 |
| rows exist but no eligible H24N rows | placeholder, `NO_ELIGIBLE_ROWS_FOR_DATE`, weight 0 |
| eligible rows exist | real direct prediction row, `is_shadow=true`, weight 0 |

CWA WRF remains shadow-only in first full implementation. It may not enter strict scoreboards or router weights until a future gate of 730 settled target dates spanning two complete seasonal cycles is satisfied.

## 11.3 Live-shadow reports

Live-shadow reports are always generated, even if all rows are placeholders:

```text
reports/live_shadow_availability_report.csv
reports/live_shadow_availability_report.md
```

---

# 12. Sealed validation command naming

Canonical command:

```bash
python -m hkg_t24.cli sealed-score --year 2024
python -m hkg_t24.cli sealed-score --year 2025
```

Aliases may be implemented for convenience:

```bash
python -m hkg_t24.cli sealed-score-2024
python -m hkg_t24.cli sealed-score-2025
```

All documentation, audit reports, manifests, and CI scripts must use the canonical `sealed-score --year YYYY` form.

If aliases exist, they must call the same function as the canonical command and write the same artifact names.

---

# 13. Freeze command naming

Canonical commands:

```bash
python -m hkg_t24.cli freeze-candidate --stage pre2024
python -m hkg_t24.cli freeze-candidate --stage refit_through_2024
```

Compatibility aliases may be implemented:

| Old/alternate command | Canonical equivalent |
|---|---|
| `phase14-freeze-candidate` | `freeze-candidate --stage pre2024` |
| `freeze-pre2024-candidate` | `freeze-candidate --stage pre2024` |
| `freeze-refit-through-2024-candidate` | `freeze-candidate --stage refit_through_2024` |

All new reports must use canonical stage values:

```text
pre2024
refit_through_2024
```

---

# 14. Final report and artifact naming consistency

## 14.1 Final required canonical artifacts

At the end of first full implementation, Codex must produce all of the following.

### Source and data readiness

```text
reports/source_inventory_report.md
reports/source_registry.csv
reports/schema_migration_source_registry.md
reports/schema_migration_feature_matrix.md
reports/gribstream_source_scope_audit.csv
reports/gribstream_source_scope_audit.md
reports/leakage_audit_report.md
reports/snapshot_coverage_report.csv
reports/snapshot_coverage_report.md
reports/live_shadow_availability_report.csv
reports/live_shadow_availability_report.md
```

### Feature documentation

```text
reports/feature_dictionary_strict.csv
reports/feature_dictionary_proxy.csv
reports/feature_dictionary_shadow.csv
reports/feature_dictionary.md
reports/feature_availability_matrix.csv
reports/feature_availability_matrix.md
reports/feature_null_rate_report.csv
reports/feature_schema_validation_report.md
```

`reports/feature_dictionary.md` is a generated human-readable combined rendering of the three canonical CSV dictionaries. It remains required.

### OOF and expert outputs

```text
reports/oof_integrity_report.md
reports/expert_scoreboard_strict.csv
reports/expert_scoreboard_proxy.csv
reports/expert_scoreboard_shadow.csv
reports/expert_fold_metrics.csv
reports/expert_promotion_decisions.csv
```

### Router and specialist outputs

```text
reports/router_scoreboard.csv
reports/router_weight_diagnostics.csv
reports/router_promotion_decisions.csv
reports/specialist_scoreboard.csv
reports/specialist_activation_report.csv
reports/specialist_no_harm_report.csv
reports/specialist_promotion_decisions.csv
```

### Distributional outputs

```text
reports/distribution_scoreboard.csv
reports/distribution_calibration_report.csv
reports/distribution_calibration_report.md
reports/threshold_probability_scoreboard.csv
reports/prediction_interval_coverage_report.csv
```

`reports/calibration_report.md` is retained as a compatibility copy of `reports/distribution_calibration_report.md` and must contain a one-line header:

```text
This file is a compatibility copy of reports/distribution_calibration_report.md.
```

### Full-system outputs

```text
reports/system_scoreboard_strict.csv
reports/system_scoreboard_proxy.csv
reports/system_ablation_matrix.csv
reports/negative_control_report.md
reports/frozen_candidate_manifest_pre2024.json
reports/frozen_candidate_manifest_refit_through_2024.json
reports/final_candidate_manifest.json
```

### Sealed validation outputs

```text
reports/sealed_2024_scoreboard.csv
reports/sealed_2024_report.md
reports/sealed_2025_scoreboard.csv
reports/sealed_2025_report.md
reports/2026_prospective_replay_scoreboard.csv
reports/2026_prospective_replay_report.md
```

## 14.2 Old artifact names

| Old name | Final status |
|---|---|
| `reports/feature_dictionary.md` | still required, generated from strict/proxy/shadow CSV dictionaries |
| `reports/feature_availability_matrix.md` | still required, generated with CSV counterpart |
| `reports/distribution_scoreboard.csv` | still required |
| `reports/calibration_report.md` | compatibility copy of `distribution_calibration_report.md` |
| `reports/feature_dictionary_strict.csv` | canonical required |
| `reports/feature_dictionary_proxy.csv` | canonical required |
| `reports/distribution_calibration_report.csv` | canonical required |
| `reports/distribution_calibration_report.md` | canonical required |

---

# 15. Final consistency for strict/proxy/shadow feature dictionaries

## 15.1 Schema version

The final strict schema version is:

```text
hkg_t24_h24n_strict_v1_20260626_patch1
```

The final proxy schema version is:

```text
hkg_t24_h24n_proxy_v1_20260626_patch1
```

The final shadow schema version is:

```text
hkg_t24_h24n_shadow_v1_20260626_patch1
```

Every `model_features.feature_matrix` row must use one of these schema versions.

## 15.2 Feature ordering

Feature ordering for exported parquet/model matrices is deterministic:

1. Sort features lexicographically by feature name.
2. Missing indicators appear immediately after their base feature in exports, even if lexicographic order would place them elsewhere.
3. Metadata columns appear before feature columns in this exact order:

```text
target_date_hkt
cutoff_id
snapshot_id
feature_scope
schema_version
```

## 15.3 Strict first-system model features

Strict model features consist of:

```text
calendar__*
official__*
target__*
gfs__*
gefsmean__*
gefsens__*
```

Strict model features do not include:

```text
ifsoper__*
ifsens__*
aifsoper__*
aifsens__*
aigfssfc__*
graphcast__*
fourcastnet__*
cwawrf15__*
arwf__*
station__*
climate__*
igra__*
tc__*
```

Those appear only in shadow/proxy dictionaries unless a future gate promotes them.

---

# 16. Source feature prefixes and exact strict/proxy/shadow handling

## 16.1 Strict feature scope

Strict first implementation includes:

```text
calendar
hko_target_labels as lagged target-memory only
hko_official_forecasts
gfs
gefsatmosmean
gefsatmos
```

## 16.2 Proxy feature scope

Proxy reports include, but strict models exclude:

```text
station_network_proxy
hko_daily_climate_proxy
```

Proxy features are built and scored separately. They are not allowed into strict frozen candidates.

## 16.3 Shadow feature scope

Shadow features and predictions include:

```text
ifsoper
ifsenfo
aifsoper
aifsenfo
aigfssfc
graphcast
fourcastnetgfs
cwawrf15
arwf_live
```

Shadow experts may be scored where labels are available under sealed/live protocol, but their weights are forced to zero in the first strict pre-2024 frozen candidate.

## 16.4 Blocked/support-only

Blocked or support-only sources:

```text
aigfspres       support-only, no Tmax expert
aigefssfc       blocked as Tmax source
nbmoc           blocked
igra            diagnostic report only
tc              diagnostic report only
```

---

# 17. Final handling of feature names in specialists and routers

The following features must be available for strict router/specialist training before those modules run. If any are unavailable because their source has no data for a date, the feature is NULL and its `__is_missing` indicator is true. If the feature definition itself is absent from the feature dictionary, the pipeline fails closed.

```text
official__forecast_max_c
official__forecast_min_c
official__forecast_range_c
official__forecast_midpoint_c
official__issue_hour_hkt
official__lead_days
official__revision_count_pre_cutoff
official__latest_minus_first_forecast_max_c
official__psr_numeric_proxy

target__lag2_tmax_c
target__roll7_mean_lag2_c
target__roll14_std_lag2_c
target__roll30_mean_lag2_c
target__slope7_lag2_c_per_day
target__slope30_lag2_c_per_day
target__slope7_minus_slope30_lag2_c_per_day
target__hot_spell_length_lag2_days
target__clim30_mean_c
target__clim30_std_c
target__lag2_minus_clim30_c
target__warming_trend_10y_c_per_year
target__year_index

calendar__month_sin1
calendar__month_cos1
calendar__doy_sin1
calendar__doy_cos1
calendar__is_mam
calendar__is_jja
calendar__is_son
calendar__is_djf
calendar__year_index

gfs__center__tmax_c
gfs__center__dewpoint_change_proxy_c
gfs__center__low_cloud_pct_mean
gfs__center__shortwave_w_m2_mean
gfs__center__precip_mm_sum
gfs__center__wind_speed_10m_mean_mps
gfs__center__onshore_easterly_component_mps
gfs__center__temp_dewpoint_spread_mean_c
gfs__center__t850_c_mean
gfs__center__z500_m_mean

gefsmean__center__tmax_c
gefsmean__center__pwat_kg_m2_mean

gefsens__center__tmax_p10_c
gefsens__center__tmax_p25_c
gefsens__center__tmax_p50_c
gefsens__center__tmax_p75_c
gefsens__center__tmax_p90_c
gefsens__center__tmax_spread_p90_p10_c
gefsens__center__prob_ge_30_0
gefsens__center__prob_ge_30_5
gefsens__center__prob_ge_31_0
gefsens__center__prob_ge_31_5
gefsens__center__prob_ge_32_0
gefsens__center__prob_ge_32_5
gefsens__center__prob_ge_33_0
gefsens__center__prob_ge_33_5
gefsens__center__prob_ge_34_0
gefsens__center__prob_ge_34_5
gefsens__center__prob_ge_35_0
gefsens__center__prob_ge_35_5
gefsens__center__prob_ge_36_0
gefsens__center__prob_ge_36_5
gefsens__center__prob_ge_37_0
gefsens__center__prob_ge_37_5
gefsens__center__prob_ge_38_0
gefsens__center__prob_ge_38_5
gefsens__center__prob_ge_39_0
gefsens__center__prob_ge_39_5
gefsens__center__prob_ge_40_0
```

---

# 18. Final rules for source absence, demotion, and fallback

## 18.1 E1 official residual demotion

If `E1_OFFICIAL_RESIDUAL` fails promotion:

1. Its OOF predictions remain written.
2. It appears in scoreboards with `promotion_status='demoted'`.
3. Its router cap is zero.
4. It may appear in proxy reports.
5. R0 and R1 automatically fall back to `E0_OFFICIAL_RAW_ANCHOR` for the official component.

## 18.2 E4/E5 demotion

If `E4_GFS_MOS` or `E5_GEFS_ENSEMBLE` fails promotion:

1. The failed expert remains scored and reported.
2. Its router cap is zero.
3. R1 may still run if at least two of the following remain promoted: E0, E1, E4, E5, E2.
4. If fewer than two promoted R1 experts remain, R1 is demoted and R0 is used.

## 18.3 Best safe baseline

`best_safe_baseline` is defined in this order:

1. `E0_OFFICIAL_RAW_ANCHOR` when available.
2. If E0 unavailable, `E2_TARGET_MEMORY`.
3. If both unavailable, no strict forecast is produced and the row is marked `NO_BASELINE_AVAILABLE`.

R0 static blend is not the baseline. It is a candidate system.

## 18.4 E0 unavailable edge case

If E0 is unavailable for a date:

1. The strict router first attempts E2 target-memory-only forecast.
2. If E2 unavailable, no strict prediction is emitted.
3. All router weights are zero.
4. The row is excluded from official-identical-row scoreboards and included in target-memory fallback coverage reports.

---

# 19. Router command and artifact consistency

## 19.1 R1 availability when R0 is available

If R1 is unavailable for a target date but R0 is available, final strict system uses R0.

Reasons R1 may be unavailable:

```text
insufficient common-row NWP features
missing promoted E4 and E5
NWP rows fail leakage filter
R1 demoted in promotion ladder
```

## 19.2 R1 promotion condition

R1 must beat both:

```text
E0_OFFICIAL_RAW_ANCHOR on identical rows
R0_STATIC_BLEND on identical rows
```

Promotion thresholds remain those in the completion spec. If R1 beats E0 but not R0, R1 is demoted.

## 19.3 Static blend after expert demotion

Static blend candidate set is the set of promoted experts with nonzero cap. If no promoted expert is available but E0 is available, static blend is E0 weight 1.0. If E0 unavailable, use E2 if promoted. If neither available, no forecast.

## 19.4 Router demotion artifact schema

Router demotion records must be written to:

```text
reports/router_promotion_decisions.csv
```

Required columns:

```text
router_id
candidate_id
evaluation_scope
identical_row_n
mae_candidate
mae_baseline_e0
mae_baseline_r0
delta_vs_e0
delta_vs_r0
rmse_candidate
bias_candidate
p90_abs_error_candidate
promotion_status
demotion_reason
created_at_utc
```

---

# 20. Distributional edge cases and threshold output keys

## 20.1 Quantile failure behavior

If quantile models fail promotion:

1. Point forecast still uses the promoted strict point system.
2. Distributional layer is marked `demoted`.
3. Prediction intervals fall back to empirical residual quantiles from promoted system OOF residuals.
4. Threshold probabilities are still produced using a Gaussian residual fallback.

## 20.2 Empirical residual fallback

For a target date, let:

```text
point = final strict point forecast
sigma = max(0.30, recent_or_oof_residual_mae * sqrt(pi/2))
```

If recent residual MAE is unavailable, use pre-2024 OOF residual MAE of the frozen candidate.

Fallback quantiles:

```text
p10 = point + empirical_q10_residual
p25 = point + empirical_q25_residual
p50 = point
p75 = point + empirical_q75_residual
p90 = point + empirical_q90_residual
```

Empirical residual quantiles must be computed only from OOF residuals of the same frozen candidate.

## 20.3 Threshold probability keys

The live and replay output JSON must include all keys from 20.0 through 40.0 inclusive in 0.5 increments:

```text
prob_tmax_ge_20_0
prob_tmax_ge_20_5
...
prob_tmax_ge_39_5
prob_tmax_ge_40_0
```

Key formatting rule:

```text
threshold 32.5 -> prob_tmax_ge_32_5
threshold 33.0 -> prob_tmax_ge_33_0
```

## 20.4 Gaussian fallback probabilities

If the calibrated distribution is demoted, compute:

```text
prob_tmax_ge_X = 1 - NormalCDF((X - point) / sigma)
```

Clamp probabilities to `[0.001, 0.999]`.

## 20.5 Calibration reports

Even when distribution is demoted, these reports are required:

```text
reports/distribution_calibration_report.csv
reports/distribution_calibration_report.md
reports/threshold_probability_scoreboard.csv
```

They must include a field:

```text
distribution_status = promoted | demoted_empirical_fallback | failed_closed
```

---

# 21. Sealed validation and shadow-source protocol — final sequence

## 21.1 Before opening 2024

Before opening 2024 labels, Codex must freeze:

```bash
python -m hkg_t24.cli freeze-candidate --stage pre2024
```

The frozen pre-2024 candidate may include only strict pre-2024 sources:

```text
E0, E1 if promoted, E2, E4, E5, R0, R1, promoted strict specialists, distribution if promoted
```

It must not include IFS, AI, CWA WRF, ARWF, station proxy, daily climate proxy, IGRA, or TC.

## 21.2 Opening 2024

Canonical command:

```bash
python -m hkg_t24.cli sealed-score --year 2024
```

This command must do two separate operations in this order:

1. Score the frozen pre-2024 strict candidate on 2024.
2. Score shadow experts on 2024 as diagnostic shadow, without altering the frozen candidate.

Shadow scoring in 2024 is allowed inside the same command, but shadow results must be written only to shadow reports and must not change the frozen strict candidate.

Required output:

```text
reports/sealed_2024_scoreboard.csv
reports/sealed_2024_report.md
reports/sealed_2024_shadow_expert_scoreboard.csv
```

## 21.3 IFS/AI adapter training after 2024

IFS/AI adapters may be trained using 2024 labels only after:

1. The frozen pre-2024 strict candidate has been scored on 2024.
2. `reports/sealed_2024_report.md` exists.
3. The strict candidate has not been modified after scoring.

Adapter command:

```bash
python -m hkg_t24.cli train-adapters --through-year 2024
```

This command may train capped challenger adapters for:

```text
ifsoper
ifsenfo
aifsoper
aifsenfo
graphcast
fourcastnet
```

It may not train ARWF or CWA WRF adapters unless their prospective live history gate has been met.

## 21.4 Refit-through-2024 candidate

After adapter training, Codex may create a refit-through-2024 candidate only with:

```bash
python -m hkg_t24.cli freeze-candidate --stage refit_through_2024
```

This candidate may include IFS/AI adapters only if all of these are true:

```text
adapter has at least 250 settled labelled rows
adapter improves 2024 shadow MAE versus the pre-2024 strict candidate on identical rows by at least 0.015 C
adapter does not worsen p90 absolute error by more than 0.020 C
adapter negative-control tests pass
adapter maximum router weight cap <= 0.10
```

## 21.5 2025 final test

Canonical command:

```bash
python -m hkg_t24.cli sealed-score --year 2025
```

2025 is final test. After seeing 2025 results, Codex must not tune:

```text
features
thresholds
hyperparameters
router caps
specialist thresholds
adapter gates
calibration parameters
```

2025 results may only be reported.

## 21.6 2026 prospective replay

2026 is prospective replay/live scoring. It uses predictions generated before outcome availability. If predictions were not generated before settlement for a date, that date is excluded from prospective live performance and included only in historical replay reports.

---

# 22. Final commands affected by this patch

The following canonical CLI commands are binding:

```bash
python -m hkg_t24.cli phase0-preflight
python -m hkg_t24.cli build-source-registry
python -m hkg_t24.cli build-h24n-snapshots
python -m hkg_t24.cli build-features --scope strict
python -m hkg_t24.cli build-features --scope proxy
python -m hkg_t24.cli build-features --scope live_shadow
python -m hkg_t24.cli train-experts --scope strict-pre2024
python -m hkg_t24.cli generate-oof --scope strict-pre2024
python -m hkg_t24.cli train-router --router R0
python -m hkg_t24.cli train-router --router R1
python -m hkg_t24.cli train-specialists --scope strict-pre2024
python -m hkg_t24.cli train-distribution --scope strict-pre2024
python -m hkg_t24.cli run-system-replay --scope strict-pre2024
python -m hkg_t24.cli run-negative-controls --scope strict-pre2024
python -m hkg_t24.cli freeze-candidate --stage pre2024
python -m hkg_t24.cli sealed-score --year 2024
python -m hkg_t24.cli train-adapters --through-year 2024
python -m hkg_t24.cli freeze-candidate --stage refit_through_2024
python -m hkg_t24.cli sealed-score --year 2025
python -m hkg_t24.cli live-predict --target-date YYYY-MM-DD --cutoff-id H24N
python -m hkg_t24.cli score-live --target-date YYYY-MM-DD
```

Compatibility aliases may exist, but CI and documentation must use only the canonical commands above.

---

# 23. Final Consistency Resolution

**YES — this consistency patch resolves the final contradictions. Codex should now implement exactly the blueprint, completion spec, final clarifications, and this patch without making design decisions.**
