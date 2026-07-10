# supplemental_doc_1_patch_1 - Binding Clarifications

**Applies to:** `supplemental_doc_1.md` and `KLGA_TMAX_TRADING_STRATEGY_SPEC.md`.

**Status:** binding implementation patch. If this patch conflicts with `supplemental_doc_1.md`, this patch wins.

**Purpose:** close the small residual contract gaps that would otherwise require Codex to make implementation decisions while building the KLGA Tmax trading system.

---

## 1. Reports Schema Is Required

`supplemental_doc_1.md` references `reports.metrics` and `reports.backtest_runs`. Codex must create these tables exactly.

### 1.1 `reports.backtest_runs`

```sql
CREATE TABLE reports.backtest_runs (
    backtest_run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_name text NOT NULL,
    run_id_text text NOT NULL UNIQUE,
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    status text NOT NULL,
    start_date date NOT NULL,
    end_date date NOT NULL,
    cutoff_id text,
    market_mode text NOT NULL,
    frozen_config_uri text,
    frozen_config_hash text,
    model_version_id uuid REFERENCES registry.model_versions(model_version_id),
    calibration_version_id uuid REFERENCES registry.model_versions(model_version_id),
    feature_version_id uuid REFERENCES registry.feature_versions(feature_version_id),
    source_code_git_sha text NOT NULL,
    metrics_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    artifact_root_uri text,
    error_message text,
    CHECK (status IN ('started','success','failed','skipped')),
    CHECK (market_mode IN ('synthetic','historical_polymarket'))
);
```

### 1.2 `reports.metrics`

```sql
CREATE TABLE reports.metrics (
    metric_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_group text NOT NULL,
    metric_name text NOT NULL,
    metric_value double precision,
    metric_text text,
    metric_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    target_date date,
    cutoff_id text,
    backtest_run_id uuid REFERENCES reports.backtest_runs(backtest_run_id),
    model_version_id uuid REFERENCES registry.model_versions(model_version_id),
    feature_version_id uuid REFERENCES registry.feature_versions(feature_version_id),
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (
        (metric_value IS NOT NULL)::int +
        (metric_text IS NOT NULL)::int +
        (metric_json <> '{}'::jsonb)::int >= 1
    )
);
```

Indexes:

```sql
CREATE INDEX ix_reports_metrics_group_name ON reports.metrics(metric_group, metric_name);
CREATE INDEX ix_reports_metrics_backtest ON reports.metrics(backtest_run_id);
CREATE INDEX ix_reports_metrics_target_cutoff ON reports.metrics(target_date, cutoff_id);
```

---

## 2. Revision-Capable Label Schema

The original `silver.target_daily_actuals` table used `target_date` as primary key, which prevents storing multiple label revisions. Replace that table contract with this version.

```sql
CREATE TABLE silver.target_daily_actuals (
    target_daily_actual_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    target_date date NOT NULL,
    station_id text NOT NULL REFERENCES registry.stations(station_id),
    source_name text NOT NULL,
    high_temp_f integer NOT NULL,
    low_temp_f integer,
    source_available_at_utc timestamptz NOT NULL,
    source_record_id uuid REFERENCES bronze.source_records(source_record_id),
    revision_number integer NOT NULL DEFAULT 1,
    is_current boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (station_id = 'KLGA'),
    CHECK (high_temp_f BETWEEN -80 AND 140),
    UNIQUE (target_date, station_id, source_name, revision_number)
);
```

Current-label uniqueness must be enforced with a partial unique index:

```sql
CREATE UNIQUE INDEX ux_target_daily_actuals_one_current
ON silver.target_daily_actuals(target_date, station_id, source_name)
WHERE is_current = true;
```

Training/scoring rule:

```text
MVP: use the current row (`is_current=true`) for labels and set `label_revision_sensitive=true` in reports.
If a future acquisition layer provides historical label freeze timestamps, use the latest revision whose `source_available_at_utc <= label_freeze_time_utc`.
Do not invent historical freeze timestamps.
```

`gold.target_instances` must join to `silver.target_daily_actuals` through the current row unless an explicit historical freeze timestamp is available.

---

## 3. PostgreSQL Expression Uniqueness

Any `UNIQUE (...)` clause in `supplemental_doc_1.md` that uses expressions such as `COALESCE(...)` or `md5(...)` must be implemented as a unique expression index, not as an inline table constraint.

### 3.1 `registry.model_versions`

Remove the inline `UNIQUE` with `md5(hyperparams::text)` and create:

```sql
CREATE UNIQUE INDEX ux_model_versions_identity
ON registry.model_versions (
    model_family,
    model_name,
    source_code_git_sha,
    COALESCE(training_data_start, '1900-01-01'::date),
    COALESCE(training_data_end, '1900-01-01'::date),
    md5(hyperparams::text)
);
```

### 3.2 `silver.availability_ledger`

Remove the inline expression `UNIQUE` and create:

```sql
CREATE UNIQUE INDEX ux_availability_ledger_identity
ON silver.availability_ledger (
    source_name,
    provider_name,
    canonical_record_key,
    variable_name,
    COALESCE(member, ''),
    COALESCE(model_name, ''),
    COALESCE(station_id, ''),
    COALESCE(run_time_utc, '1900-01-01'::timestamptz),
    COALESCE(valid_time_utc, '1900-01-01'::timestamptz)
);
```

### 3.3 Other Expression-Based Uniqueness

Apply the same rule for:

```text
silver.station_observations
silver.mos_guidance
silver.grib_forecast_values
```

If the uniqueness expression contains `COALESCE`, implement it as `CREATE UNIQUE INDEX`, not as an inline `UNIQUE` constraint.

SQLAlchemy models must represent these as `Index(..., unique=True)` expression indexes where supported. If SQLAlchemy expression-index rendering is awkward, Alembic migration DDL is authoritative.

---

## 4. Canonical Feature Alias Map

Codex must implement a deterministic alias map in:

```text
src/klga_tmax/features/aliases.py
```

The alias map prevents feature-name ambiguity between shorthand names used in formulas and long materialized names.

```python
FEATURE_ALIASES = {
    "nbm_tmax_f": "grib_nbm_klga_core_tmax_hourly_f",
    "nbm_tmax": "grib_nbm_klga_core_tmax_hourly_f",
    "hrrr_tmax_f": "grib_hrrr_klga_core_tmax_hourly_f",
    "hrrr_tmax": "grib_hrrr_klga_core_tmax_hourly_f",
    "rap_tmax_f": "grib_rap_klga_core_tmax_hourly_f",
    "rap_tmax": "grib_rap_klga_core_tmax_hourly_f",
    "gfs_tmax_f": "grib_gfs_klga_core_tmax_hourly_f",
    "gfs_tmax": "grib_gfs_klga_core_tmax_hourly_f",
    "gefs_mean_tmax_f": "grib_gefsatmosmean_klga_core_tmax_hourly_f",
    "gefs_mean_tmax": "grib_gefsatmosmean_klga_core_tmax_hourly_f",
    "ifsoper_tmax_f": "grib_ifsoper_klga_core_tmax_hourly_f",
    "aifsoper_tmax_f": "grib_aifsoper_klga_core_tmax_hourly_f",
    "aigfssfc_tmax_f": "grib_aigfssfc_klga_core_tmax_hourly_f",
}
```

Rules:

```text
1. Feature materialization stores canonical long names.
2. Modeling code may request shorthand names only through `resolve_feature_alias`.
3. If both shorthand and canonical names exist in a feature row, canonical value wins and a warning is logged.
4. Shorthand names must not be inserted into `gold.feature_values`.
```

---

## 5. Materialized Intermediate Risk Features

Some formulas in `supplemental_doc_1.md` compute intermediate scores and later reuse them. Codex must materialize these intermediate scores as explicit features.

### 5.1 Sea-Breeze Intermediates

Materialize:

```text
risk_sea_breeze_marine_dir_score
risk_sea_breeze_marine_speed_score
risk_sea_breeze_inland_heat_gradient_score
risk_sea_breeze_clear_enough_score
risk_sea_breeze_dry_enough_score
risk_sea_breeze_warm_season_score
risk_sea_breeze_inputs_available_count
risk_sea_breeze_score
```

The HRRR/RAP interaction formula must use:

```python
interaction_hrrr_wind_east_x_gradient = (
    risk_sea_breeze_marine_dir_score *
    grad_grib_hrrr_inland_minus_marine_tmax_f
)
```

Do not use an unmaterialized variable named `marine_dir_score` outside the sea-breeze formula.

### 5.2 Backdoor, Marine, Cloud, And Storm Intermediates

Codex should materialize intermediate components for:

```text
risk_backdoor_front_*
risk_marine_layer_*
risk_cloud_bust_*
risk_storm_outflow_*
```

Exact names use the pattern:

```text
<final_risk_feature_name_without_score>_<component_name>_score
```

Example:

```text
risk_cloud_bust_cloud_amount_score
risk_cloud_bust_cloud_disagreement_score
risk_cloud_bust_precip_score
```

These features are required for auditability and report explanations.

---

## 6. AI Feature Typo Fix

In `supplemental_doc_1.md`, Section 4.9 contains a non-ASCII typo:

```text
grיב_aifsoper_*
```

The binding correct feature pattern is:

```text
grib_aifsoper_*
```

All source files and feature names must be ASCII.

---

## 7. Timedelta Formula Fixes

Where `supplemental_doc_1.md` uses `.hours` on a time delta, Codex must instead use:

```python
hours = (later_timestamp - earlier_timestamp).total_seconds() / 3600.0
```

Apply this to:

```text
warming_rate_3h_f_per_hour
source_age_hours
run_age_hours
previous_run_gap_hours
```

Correct warming-rate formula:

```python
delta_hours = (obs_30m.observed_at_utc - obs_3h.observed_at_utc).total_seconds() / 3600.0
warming_rate_3h_f_per_hour = (obs_30m.temp_f - obs_3h.temp_f) / delta_hours
```

Correct staleness age formula:

```python
source_age_hours = (cutoff_utc - latest_effective_available_at_utc).total_seconds() / 3600.0
```

---

## 8. Stale-Data Cross-Reference Fix

`supplemental_doc_1.md` Section 1.10 says "hard maximum in Section 2.18". The correct reference is:

```text
Section 2.25 Data staleness score
```

The hard maximums are the `Max age hours` values in Section 2.25.

---

## 9. Critical Source Missing Count

Codex must define:

```python
CRITICAL_SOURCE_FAMILIES = [
    "MOS",
    "NBM",
    "HRRR_or_RAP",
    "GFS_or_IFS",
    "observations",
    "Polymarket_orderbook",
]
```

For a target/cutoff:

```python
critical_source_missing_count = count(
    family for family in CRITICAL_SOURCE_FAMILIES
    if family_required_for_context(family) and family_missing_or_stale(family)
)
```

Rules:

```text
1. `Polymarket_orderbook` is required only for `trade decide`, paper/backtest/live trading reports, and market-aware backtests.
2. `HRRR_or_RAP` is present if at least one of HRRR or RAP latest eligible scalar features exists and is not stale.
3. `GFS_or_IFS` is present if at least one of GFS, IFS, or AIFS latest eligible scalar features exists and is not stale.
4. A family is stale when its staleness score from Section 2.25 is >= 1.0.
5. A family is missing when all canonical scalar features for that family are missing.
```

---

## 10. Acquisition Database Normalization Contract

The user may fetch acquisition data before implementation. Codex must not assume the fetched database shape exactly matches this supplement unless verified.

Implementation rule:

```text
Before feature materialization, Codex must run a contract inspection step that verifies required `bronze`, `silver`, and `registry` tables/columns exist.
```

CLI:

```text
klga-tmax db inspect-contract
```

Behavior:

```text
1. If acquisition tables already match this supplement's required schema, exit 0.
2. If acquisition data exists under a different schema/table layout, Codex must create deterministic normalization loaders into this supplement's `bronze.*` and `silver.*` tables.
3. Normalization loaders must be implemented in `src/klga_tmax/db/normalize_acquisition.py`.
4. Every source-to-contract mapping must be declared in `config/acquisition_table_map.yaml`.
5. If a required field cannot be mapped exactly, fail with exit code 30 and print the missing mapping. Do not infer silently.
```

Mapping file minimum structure:

```yaml
sources:
  wunderground_daily_actuals:
    source_table: "existing_schema.existing_table"
    target_table: "silver.target_daily_actuals"
    columns:
      target_date: "..."
      station_id: "..."
      high_temp_f: "..."
      source_available_at_utc: "..."
  grib_forecast_values:
    source_table: "existing_schema.existing_table"
    target_table: "silver.grib_forecast_values"
    columns:
      model_name: "..."
      coord_name: "..."
      run_time_utc: "..."
      valid_time_utc: "..."
      variable_name: "..."
      value: "..."
      effective_available_at_utc: "..."
```

If no mapping file exists and contract tables are missing, `features materialize` must fail before doing any work.

---

## 11. Full-Production Live Trading Persistence

MVP remains paper-only by default. If full-production live execution is implemented, Codex must create the following live persistence tables.

### 11.1 `trading.live_orders`

```sql
CREATE TABLE trading.live_orders (
    live_order_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    trade_decision_leg_id uuid NOT NULL REFERENCES trading.trade_decision_legs(trade_decision_leg_id),
    polymarket_order_id text,
    token_id text NOT NULL,
    side text NOT NULL,
    order_type text NOT NULL,
    limit_price numeric(18,8) NOT NULL,
    requested_shares numeric(18,8) NOT NULL,
    requested_notional_usdc numeric(18,8) NOT NULL,
    status text NOT NULL,
    submitted_at_utc timestamptz,
    last_reconciled_at_utc timestamptz,
    raw_order_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    CHECK (side IN ('buy','sell')),
    CHECK (order_type IN ('limit','marketable_limit')),
    CHECK (status IN ('prepared','submitted','partially_filled','filled','cancelled','rejected','expired','reconcile_failed'))
);
```

### 11.2 `trading.live_order_events`

```sql
CREATE TABLE trading.live_order_events (
    live_order_event_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    live_order_id uuid NOT NULL REFERENCES trading.live_orders(live_order_id) ON DELETE CASCADE,
    event_type text NOT NULL,
    event_at_utc timestamptz NOT NULL DEFAULT now(),
    raw_event_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    message text,
    CHECK (event_type IN ('prepared','submitted','fill','partial_fill','cancel_requested','cancelled','rejected','expired','reconciled','error'))
);
```

### 11.3 `trading.live_fills`

```sql
CREATE TABLE trading.live_fills (
    live_fill_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    live_order_id uuid NOT NULL REFERENCES trading.live_orders(live_order_id) ON DELETE CASCADE,
    polymarket_fill_id text,
    filled_at_utc timestamptz NOT NULL,
    filled_shares numeric(18,8) NOT NULL,
    avg_fill_price numeric(18,8) NOT NULL,
    fee_usdc numeric(18,8) NOT NULL DEFAULT 0,
    raw_fill_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (polymarket_fill_id)
);
```

### 11.4 `trading.risk_kill_switches`

```sql
CREATE TABLE trading.risk_kill_switches (
    kill_switch_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    enabled boolean NOT NULL DEFAULT true,
    reason text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz,
    created_by text NOT NULL DEFAULT 'codex'
);
```

Live execution rule:

```text
Before any live order submission, Codex must verify no active kill switch exists where `enabled=true` and (`expires_at is null` or `expires_at > now()`).
```

---

## 12. Live Execution CLI Boundary

`klga-tmax trade decide --mode live` must only write a live-mode decision. It must not submit orders by itself.

Add this command for actual full-production order submission:

```text
klga-tmax trade execute-live
```

Arguments:

```text
--trade-decision-id UUID required
--allow-live required
--allow-taker false default
--resting-maker false default
--max-clock-drift-seconds 2 default
```

Behavior:

```text
1. Fail with exit code 70 if `KLGA_TRADING_MODE != live`.
2. Fail with exit code 70 if `--allow-live` is absent.
3. Fail with exit code 70 if live credentials are missing.
4. Fail with exit code 70 if an active kill switch exists.
5. Fail with exit code 70 if clock drift check fails.
6. Create `trading.live_orders` rows with status `prepared`.
7. Submit orders only after all preflight checks pass.
8. Reconcile submitted orders into `trading.live_order_events` and `trading.live_fills`.
```

MVP must implement this command either as:

```text
disabled stub that always exits 70 unless full-production live implementation is enabled
```

or omit it while keeping all paper/backtest behavior complete.

---

## 13. Final No-Decision Rule

After applying this patch, Codex should not need to invent design choices for MVP implementation.

Remaining non-MVP empirical choices are already bounded by:

```text
supplemental_doc_1.md Section 13 Closed empirical decision rules
```

If implementation encounters a conflict between the main spec, `supplemental_doc_1.md`, and this patch, precedence is:

```text
1. supplemental_doc_1_patch_1.md
2. supplemental_doc_1.md
3. KLGA_TMAX_TRADING_STRATEGY_SPEC.md
4. data_acquisition specs for acquisition-only behavior
```

