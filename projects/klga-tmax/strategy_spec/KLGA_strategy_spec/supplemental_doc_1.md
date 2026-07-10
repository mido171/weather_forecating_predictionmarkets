# supplemental_doc_1 — KLGA Tmax Trading System Implementation Contract

**Document status:** binding implementation supplement.  
**Primary document supplemented:** `KLGA_TMAX_TRADING_STRATEGY_SPEC.md`.  
**Acquisition specs assumed implemented:** `C:\Users\ahmad\Desktop\generalFiles\git\weather_markets\weather_data_extraction\bootstrap\klga_tmax\strategy_spec\data_aquisition`.  
**Scope of this supplement:** post-acquisition database contract, feature formulas, regime flags, modeling, calibration, validation, trading, orchestration, artifacts, testing, and MVP/full-production boundaries.  
**Do not rewrite the main strategy from this supplement.** Implement this supplement as the decision-complete contract that removes ambiguity left after acquisition.

---

## 0. Global implementation constants

### 0.1 Canonical target

```text
Y_T = Wunderground-reported KLGA whole-degree Fahrenheit daily high for target date T.
T is a New York local calendar date: America/New_York 00:00:00 through 23:59:59.
```

Temperature grid for every PMF:

```python
TEMP_GRID_F = list(range(50, 116))  # inclusive 50..115
TEMP_GRID_MIN_F = 50
TEMP_GRID_MAX_F = 115
```

If any historical target label is outside `[50, 115]`, Codex must fail the feature/materialization run with exit code `31` and message:

```text
TARGET_OUTSIDE_TEMP_GRID: extend TEMP_GRID_F before training.
```

Do not silently clip labels.

### 0.2 Canonical cutoffs

Codex must materialize every target date for these four cutoffs exactly:

| cutoff_id | Definition |
|---|---|
| `T_MINUS_1_STOCKHOLM_1500` | 15:00 Europe/Stockholm on target date T minus one calendar day. |
| `T_MINUS_1_STOCKHOLM_1915` | 19:15 Europe/Stockholm on target date T minus one calendar day. |
| `T_MINUS_1_STOCKHOLM_2230` | 22:30 Europe/Stockholm on target date T minus one calendar day. |
| `PRE_LOCAL_DAY_NYC_2350` | 23:50 America/New_York on target date T minus one local calendar day. |

Implementation must use Python `zoneinfo.ZoneInfo`, never fixed UTC offsets.

### 0.3 Naming conventions

All dates are ISO `YYYY-MM-DD`. All timestamps stored in PostgreSQL are `timestamptz` and must be UTC on insert. All database identifiers use lowercase snake case. All model/expert IDs are lowercase snake case.

### 0.4 Required Python stack

Codex must implement v1 using these libraries:

```text
python >=3.11,<3.13
numpy >=1.26,<3
pandas >=2.2,<3
polars >=1.0,<2              # optional speed path, but pandas API must remain available
pyarrow >=15,<25
scipy >=1.11,<2
scikit-learn >=1.4,<2
lightgbm >=4.3,<5
sqlalchemy >=2.0,<3
alembic >=1.13,<2
psycopg[binary] >=3.1,<4
pydantic >=2.7,<3
typer >=0.12,<1
rich >=13,<15
joblib >=1.3,<2
matplotlib >=3.8,<4
```

If `lightgbm` is unavailable in the execution environment, Codex must use the exact fallback class `sklearn.ensemble.HistGradientBoostingRegressor` with the fallback hyperparameters specified in Section 4.3. Codex must log `LIGHTGBM_UNAVAILABLE_USING_SKLEARN_FALLBACK` and set `used_fallback_model=true` in `registry.model_versions`.

### 0.5 Determinism

Every run must set:

```python
GLOBAL_RANDOM_SEED = 1729
np.random.seed(1729)
random.seed(1729)
PYTHONHASHSEED=1729
```

For deterministic reproducibility, all training commands must default to one worker:

```text
KLGA_N_JOBS=1
```

Parallel execution may be added later, but MVP acceptance requires deterministic output hashes on repeated single-worker runs.

### 0.6 Numeric precision

Store forecast/model values as `double precision`. Store market prices, bankroll, and trade sizes as `numeric(18,8)`. PMF probabilities must be stored as `double precision` and must satisfy:

```text
0 <= p <= 1
abs(sum_k p_k - 1.0) <= 1e-8
```

---

## 1. Exact database contract

### 1.1 Database engine

Codex must use PostgreSQL 16 or newer. SQLite is forbidden except in unit tests where explicitly mocked. The production database engine must support schemas, JSONB, generated UUIDs, and materialized views.

Required PostgreSQL extension:

```sql
CREATE EXTENSION IF NOT EXISTS pgcrypto;
```

### 1.2 Connection configuration

Required environment variables:

| Env var | Required | Meaning |
|---|---:|---|
| `KLGA_DB_URL` | yes | SQLAlchemy URL, e.g. `postgresql+psycopg://user:pass@host:5432/weather_markets`. |
| `KLGA_ARTIFACT_ROOT` | yes | Root path for model/report artifacts. Default in local dev only: `./artifacts/klga_tmax`. |
| `KLGA_ENV` | yes | One of `local`, `paper`, `prod`. |
| `KLGA_TRADING_MODE` | yes | One of `backtest`, `paper`, `live`. MVP default must be `paper`. |
| `KLGA_N_JOBS` | no | Integer; default `1`. |
| `KLGA_LOG_LEVEL` | no | `DEBUG`, `INFO`, `WARNING`, `ERROR`; default `INFO`. |
| `KLGA_POLYMARKET_PRIVATE_KEY` | live only | Required only for authenticated live execution. MVP must not require it. |
| `KLGA_POLYMARKET_FUNDER_ADDRESS` | live only | Required only for live execution. |
| `KLGA_BANKROLL_USDC` | paper/live | Decimal bankroll if not read from account. Required for paper mode. |

If `KLGA_DB_URL` is missing, every CLI command except `--help` must exit with code `10`.

### 1.3 Required schemas

Codex must create and use these schemas exactly:

```sql
CREATE SCHEMA IF NOT EXISTS registry;
CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;
CREATE SCHEMA IF NOT EXISTS predictions;
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS reports;
CREATE SCHEMA IF NOT EXISTS audit;
```

Schema semantics:

| Schema | Purpose |
|---|---|
| `registry` | Stable reference data, version registries, run metadata. |
| `bronze` | Raw post-acquisition source payload records. Immutable except supersession flags. |
| `silver` | Normalized source records with unit conversions and availability metadata. |
| `gold` | Target/cutoff feature matrices and feature lineage. |
| `predictions` | Expert PMFs, final PMFs, calibration artifacts, scoring outputs. |
| `trading` | Market snapshots, orderbook snapshots, decisions, simulated/live fills. |
| `reports` | Backtest/report metadata and metrics. |
| `audit` | Pipeline runs, logs, data-quality failures. |

### 1.4 Migration strategy

Codex must create **both**:

```text
alembic migrations      # authoritative DDL
SQLAlchemy ORM models   # application access layer
```

The Alembic migration files are authoritative. SQLAlchemy models must match the current migration head. CI must include a test that creates an empty database, applies migrations, reflects metadata, and verifies all required tables/indexes exist.

Migration directory:

```text
alembic/
  env.py
  script.py.mako
  versions/
```

Initial migration filename:

```text
0001_klga_tmax_core_schema.py
```

### 1.5 Core table definitions

The following DDL is the binding v1 contract. Codex may add columns only if all required columns and constraints remain unchanged.

#### 1.5.1 `registry.stations`

```sql
CREATE TABLE registry.stations (
    station_id text PRIMARY KEY,
    station_name text NOT NULL,
    provider_primary_id text NOT NULL,
    latitude double precision NOT NULL,
    longitude double precision NOT NULL,
    elevation_m double precision,
    timezone text NOT NULL DEFAULT 'America/New_York',
    station_role text NOT NULL,
    station_group text[] NOT NULL DEFAULT '{}',
    active boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (latitude BETWEEN -90 AND 90),
    CHECK (longitude BETWEEN -180 AND 180),
    CHECK (station_role IN ('target','nearby','pseudo_point','external_context'))
);
```

Required station rows:

| station_id | role | station_group |
|---|---|---|
| `KLGA` | `target` | `{target,nyc_airport,coastal}` |
| `KNYC` | `nearby` | `{urban_core,nearby}` |
| `KJFK` | `nearby` | `{coastal_marine,nyc_airport}` |
| `KEWR` | `nearby` | `{inland_warm,nyc_airport}` |
| `KTEB` | `nearby` | `{inland_warm}` |
| `KHPN` | `nearby` | `{inland_north}` |
| `KISP` | `nearby` | `{coastal_marine,long_island}` |
| `KBDR` | `nearby` | `{backdoor_ne,coastal_marine}` |
| `KSWF` | `nearby` | `{upstream_nw}` |
| `KPOU` | `nearby` | `{upstream_nw}` |
| `KPHL` | `nearby` | `{upstream_sw}` |
| `KBOS` | `nearby` | `{backdoor_ne}` |
| `KDCA` | `nearby` | `{upstream_sw,mid_atlantic}` |
| `KBWI` | `nearby` | `{upstream_sw,mid_atlantic}` |
| `PSEUDO_KLGA_CORE` | `pseudo_point` | `{pseudo_core}` |
| `PSEUDO_INLAND_NJ` | `pseudo_point` | `{pseudo_inland_warm}` |
| `PSEUDO_MARINE_JFK_BAY` | `pseudo_point` | `{pseudo_coastal_marine}` |
| `PSEUDO_UPSTREAM_SW` | `pseudo_point` | `{pseudo_upstream_sw}` |
| `PSEUDO_BACKDOOR_NE` | `pseudo_point` | `{pseudo_backdoor_ne}` |

If acquisition has different exact coordinates for pseudo points, Codex must use acquisition coordinates; otherwise use these defaults:

```text
PSEUDO_KLGA_CORE:       40.77945, -73.88027
PSEUDO_INLAND_NJ:       40.73500, -74.18000
PSEUDO_MARINE_JFK_BAY:  40.64000, -73.78000
PSEUDO_UPSTREAM_SW:     40.20000, -74.90000
PSEUDO_BACKDOOR_NE:     41.10000, -72.90000
```

#### 1.5.2 `registry.cutoffs`

```sql
CREATE TABLE registry.cutoffs (
    cutoff_id text PRIMARY KEY,
    cutoff_order integer NOT NULL UNIQUE,
    timezone_name text NOT NULL,
    local_time time NOT NULL,
    target_day_offset integer NOT NULL,
    description text NOT NULL,
    active boolean NOT NULL DEFAULT true
);
```

Required rows:

```text
('T_MINUS_1_STOCKHOLM_1500', 1, 'Europe/Stockholm', '15:00:00', -1)
('T_MINUS_1_STOCKHOLM_1915', 2, 'Europe/Stockholm', '19:15:00', -1)
('T_MINUS_1_STOCKHOLM_2230', 3, 'Europe/Stockholm', '22:30:00', -1)
('PRE_LOCAL_DAY_NYC_2350',   4, 'America/New_York',  '23:50:00', -1)
```

#### 1.5.3 `registry.model_versions`

```sql
CREATE TABLE registry.model_versions (
    model_version_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    model_family text NOT NULL,
    model_name text NOT NULL,
    model_role text NOT NULL,
    source_code_git_sha text NOT NULL,
    training_data_start date,
    training_data_end date,
    feature_version_id uuid,
    hyperparams jsonb NOT NULL DEFAULT '{}'::jsonb,
    artifact_uri text,
    artifact_hash text,
    used_fallback_model boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (model_family, model_name, source_code_git_sha, training_data_start, training_data_end, md5(hyperparams::text))
);
```

`model_role` must be one of:

```text
expert, meta_combiner, calibrator, simulation, report
```

#### 1.5.4 `registry.feature_versions`

```sql
CREATE TABLE registry.feature_versions (
    feature_version_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    feature_set_name text NOT NULL,
    feature_version text NOT NULL,
    source_code_git_sha text NOT NULL,
    formula_contract_hash text NOT NULL,
    feature_names text[] NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (feature_set_name, feature_version)
);
```

For this supplement, v1 feature version is:

```text
feature_set_name = 'klga_tmax_core'
feature_version = 'supplemental_doc_1_v1'
```

#### 1.5.5 `audit.pipeline_runs`

```sql
CREATE TABLE audit.pipeline_runs (
    pipeline_run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    command_name text NOT NULL,
    command_args jsonb NOT NULL DEFAULT '{}'::jsonb,
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    status text NOT NULL,
    exit_code integer,
    source_code_git_sha text NOT NULL,
    row_counts jsonb NOT NULL DEFAULT '{}'::jsonb,
    error_message text,
    log_uri text,
    CHECK (status IN ('started','success','failed','skipped'))
);
```

#### 1.5.6 `bronze.source_records`

```sql
CREATE TABLE bronze.source_records (
    source_record_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_name text NOT NULL,
    provider_name text NOT NULL,
    endpoint_name text NOT NULL,
    provider_record_key text NOT NULL,
    request_hash text,
    payload_hash text NOT NULL,
    payload_format text NOT NULL,
    payload_json jsonb,
    payload_text text,
    payload_uri text,
    provider_issued_at_utc timestamptz,
    provider_valid_at_utc timestamptz,
    provider_available_at_utc timestamptz,
    acquired_at_utc timestamptz NOT NULL,
    revision_number integer NOT NULL DEFAULT 1,
    supersedes_source_record_id uuid REFERENCES bronze.source_records(source_record_id),
    is_current boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (payload_format IN ('json','csv','ndjson','text','parquet','binary_uri')),
    CHECK ((payload_json IS NOT NULL)::int + (payload_text IS NOT NULL)::int + (payload_uri IS NOT NULL)::int >= 1),
    UNIQUE (source_name, provider_name, endpoint_name, provider_record_key, revision_number)
);
```

Indexes:

```sql
CREATE INDEX ix_bronze_source_records_provider_time ON bronze.source_records(source_name, provider_name, provider_issued_at_utc, provider_valid_at_utc);
CREATE INDEX ix_bronze_source_records_current ON bronze.source_records(source_name, provider_name, is_current);
CREATE INDEX ix_bronze_source_records_payload_hash ON bronze.source_records(payload_hash);
```

Revision handling rule:

```text
If a new payload arrives with same provider_record_key and different payload_hash, insert revision_number = previous max + 1, set previous is_current=false, set supersedes_source_record_id to previous current row. Never update payload content in place.
```

Duplicate handling rule:

```text
If same provider_record_key and same payload_hash already exists, do not insert a second row. Return existing source_record_id.
```

#### 1.5.7 `silver.availability_ledger`

```sql
CREATE TABLE silver.availability_ledger (
    availability_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_record_id uuid REFERENCES bronze.source_records(source_record_id),
    source_name text NOT NULL,
    provider_name text NOT NULL,
    canonical_record_key text NOT NULL,
    station_id text REFERENCES registry.stations(station_id),
    model_name text,
    run_time_utc timestamptz,
    valid_time_utc timestamptz,
    forecast_hour integer,
    member text,
    variable_name text NOT NULL,
    provider_available_at_utc timestamptz NOT NULL,
    acquired_at_utc timestamptz NOT NULL,
    effective_available_at_utc timestamptz NOT NULL,
    availability_method text NOT NULL,
    source_lag_seconds integer,
    is_revision_current boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (availability_method IN ('observed_provider_timestamp','observed_ingest_timestamp','conservative_lag_rule','manual_override')),
    UNIQUE (source_name, provider_name, canonical_record_key, variable_name, COALESCE(member,''), COALESCE(model_name,''), COALESCE(station_id,''), COALESCE(run_time_utc,'1900-01-01'::timestamptz), COALESCE(valid_time_utc,'1900-01-01'::timestamptz))
);
```

Eligibility is always:

```sql
effective_available_at_utc <= cutoff_utc
```

`effective_available_at_utc` must be:

```text
provider_available_at_utc if observed and trusted;
else acquired_at_utc if acquired_at_utc is the earliest observed availability;
else run_time_utc + source-specific conservative lag from acquisition spec.
```

#### 1.5.8 `silver.target_daily_actuals`

```sql
CREATE TABLE silver.target_daily_actuals (
    target_date date PRIMARY KEY,
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
    CHECK (high_temp_f BETWEEN -80 AND 140)
);
```

Only `is_current=true` rows may be used for final scoring. For historical training, if revisions exist, use the latest revision whose `source_available_at_utc <= label_freeze_time_utc`. If no label freeze is known, use current revision but mark `label_revision_sensitive=true` in reports.

#### 1.5.9 `silver.station_daily_actuals`

```sql
CREATE TABLE silver.station_daily_actuals (
    station_id text NOT NULL REFERENCES registry.stations(station_id),
    local_date date NOT NULL,
    source_name text NOT NULL,
    high_temp_f double precision,
    low_temp_f double precision,
    precip_in double precision,
    avg_wind_speed_kt double precision,
    max_wind_gust_kt double precision,
    source_available_at_utc timestamptz NOT NULL,
    source_record_id uuid REFERENCES bronze.source_records(source_record_id),
    is_current boolean NOT NULL DEFAULT true,
    PRIMARY KEY (station_id, local_date, source_name)
);
```

#### 1.5.10 `silver.station_observations`

```sql
CREATE TABLE silver.station_observations (
    observation_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    station_id text NOT NULL REFERENCES registry.stations(station_id),
    source_name text NOT NULL,
    observed_at_utc timestamptz NOT NULL,
    effective_available_at_utc timestamptz NOT NULL,
    temp_f double precision,
    dewpoint_f double precision,
    wind_dir_deg double precision,
    wind_speed_kt double precision,
    wind_gust_kt double precision,
    sea_level_pressure_mb double precision,
    altimeter_inhg double precision,
    visibility_mi double precision,
    cloud_cover_code text,
    precip_1h_in double precision,
    raw_metar text,
    source_record_id uuid REFERENCES bronze.source_records(source_record_id),
    quality_flag text NOT NULL DEFAULT 'ok',
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (station_id, source_name, observed_at_utc, COALESCE(raw_metar,''))
);
```

Allowed `quality_flag`: `ok`, `suspect`, `duplicate`, `revised`, `missing_core_fields`.

#### 1.5.11 `silver.mos_guidance`

```sql
CREATE TABLE silver.mos_guidance (
    mos_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    station_id text NOT NULL REFERENCES registry.stations(station_id),
    product text NOT NULL,
    model_cycle_utc timestamptz NOT NULL,
    forecast_valid_time_utc timestamptz,
    forecast_period_start_utc timestamptz,
    forecast_period_end_utc timestamptz,
    forecast_hour integer,
    variable_code text NOT NULL,
    variable_name text NOT NULL,
    value double precision,
    unit text NOT NULL,
    effective_available_at_utc timestamptz NOT NULL,
    source_record_id uuid REFERENCES bronze.source_records(source_record_id),
    raw_token text,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (station_id, product, model_cycle_utc, COALESCE(forecast_valid_time_utc,'1900-01-01'::timestamptz), COALESCE(forecast_period_start_utc,'1900-01-01'::timestamptz), COALESCE(forecast_period_end_utc,'1900-01-01'::timestamptz), variable_code)
);
```

`product` allowed values for v1:

```text
MAV, MET, MEX, LAV, NBS, NBE
```

#### 1.5.12 `silver.grib_forecast_values`

```sql
CREATE TABLE silver.grib_forecast_values (
    grib_value_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    provider_name text NOT NULL DEFAULT 'gribstream',
    model_name text NOT NULL,
    station_id text REFERENCES registry.stations(station_id),
    coord_name text NOT NULL,
    latitude double precision NOT NULL,
    longitude double precision NOT NULL,
    run_time_utc timestamptz NOT NULL,
    valid_time_utc timestamptz NOT NULL,
    forecast_hour integer NOT NULL,
    member text NOT NULL DEFAULT 'deterministic',
    variable_name text NOT NULL,
    level_name text,
    value double precision,
    unit text NOT NULL,
    effective_available_at_utc timestamptz NOT NULL,
    source_record_id uuid REFERENCES bronze.source_records(source_record_id),
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (provider_name, model_name, coord_name, run_time_utc, valid_time_utc, member, variable_name, COALESCE(level_name,''))
);
```

#### 1.5.13 `silver.polymarket_markets`

```sql
CREATE TABLE silver.polymarket_markets (
    polymarket_market_id text PRIMARY KEY,
    event_id text,
    condition_id text,
    question text NOT NULL,
    title text,
    slug text,
    market_url text,
    target_station_id text,
    target_date date,
    resolution_source_url text,
    enable_order_book boolean,
    active boolean,
    closed boolean,
    archived boolean,
    fees_enabled boolean,
    category text,
    raw_market_json jsonb NOT NULL,
    discovered_at_utc timestamptz NOT NULL,
    updated_at_utc timestamptz NOT NULL
);
```

#### 1.5.14 `silver.polymarket_outcomes`

```sql
CREATE TABLE silver.polymarket_outcomes (
    outcome_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    polymarket_market_id text NOT NULL REFERENCES silver.polymarket_markets(polymarket_market_id),
    outcome_index integer NOT NULL,
    outcome_name text NOT NULL,
    token_id text,
    parsed_bucket_type text NOT NULL,
    lower_temp_f integer,
    upper_temp_f integer,
    lower_inclusive boolean NOT NULL DEFAULT true,
    upper_inclusive boolean NOT NULL DEFAULT true,
    is_yes_token boolean NOT NULL DEFAULT true,
    parse_status text NOT NULL,
    parse_error text,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (polymarket_market_id, outcome_index),
    UNIQUE (polymarket_market_id, token_id),
    CHECK (parsed_bucket_type IN ('below_or_equal','below_strict','above_or_equal','above_strict','closed_range','exact','other','unknown')),
    CHECK (parse_status IN ('parsed','ambiguous','unsupported'))
);
```

#### 1.5.15 `gold.target_instances`

```sql
CREATE TABLE gold.target_instances (
    target_instance_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    target_date date NOT NULL,
    cutoff_id text NOT NULL REFERENCES registry.cutoffs(cutoff_id),
    cutoff_utc timestamptz NOT NULL,
    target_station_id text NOT NULL DEFAULT 'KLGA' REFERENCES registry.stations(station_id),
    label_high_temp_f integer,
    label_available_at_utc timestamptz,
    label_is_available boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (target_date, cutoff_id)
);
```

#### 1.5.16 `gold.feature_values`

Canonical long-form feature table:

```sql
CREATE TABLE gold.feature_values (
    feature_value_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    target_date date NOT NULL,
    cutoff_id text NOT NULL REFERENCES registry.cutoffs(cutoff_id),
    feature_version_id uuid NOT NULL REFERENCES registry.feature_versions(feature_version_id),
    feature_name text NOT NULL,
    value_float double precision,
    value_text text,
    value_bool boolean,
    is_missing boolean NOT NULL DEFAULT false,
    missing_reason text,
    min_history_met boolean NOT NULL DEFAULT true,
    leakage_checked boolean NOT NULL DEFAULT false,
    trace_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (target_date, cutoff_id, feature_version_id, feature_name),
    CHECK ((value_float IS NOT NULL)::int + (value_text IS NOT NULL)::int + (value_bool IS NOT NULL)::int + (is_missing::int) >= 1)
);
```

Traceability requirement:

```json
{
  "formula_id": "mos_scalar_v1",
  "input_tables": ["silver.mos_guidance"],
  "source_record_ids": ["..."],
  "availability_ids": ["..."],
  "max_effective_available_at_utc": "...",
  "cutoff_utc": "...",
  "leakage_rule": "max_effective_available_at_utc <= cutoff_utc",
  "parameters": {"window_days": 30}
}
```

`leakage_checked` must be true for every non-missing feature. Feature materialization must fail if any trace has `max_effective_available_at_utc > cutoff_utc`.

#### 1.5.17 `gold.feature_matrix`

Codex must materialize a wide table for fast model training:

```sql
CREATE TABLE gold.feature_matrix (
    target_date date NOT NULL,
    cutoff_id text NOT NULL,
    feature_version_id uuid NOT NULL REFERENCES registry.feature_versions(feature_version_id),
    features_json jsonb NOT NULL,
    label_high_temp_f integer,
    label_available boolean NOT NULL,
    row_hash text NOT NULL,
    materialized_at_utc timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (target_date, cutoff_id, feature_version_id)
);
```

`features_json` contains all features as key-value pairs. Missing numeric features must be represented as JSON `null`; missing flags must be explicit separate boolean features named `<feature_name>__missing`.

#### 1.5.18 `predictions.expert_predictions`

```sql
CREATE TABLE predictions.expert_predictions (
    expert_prediction_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    target_date date NOT NULL,
    cutoff_id text NOT NULL,
    expert_id text NOT NULL,
    model_version_id uuid NOT NULL REFERENCES registry.model_versions(model_version_id),
    feature_version_id uuid NOT NULL REFERENCES registry.feature_versions(feature_version_id),
    prediction_mode text NOT NULL,
    point_mean_f double precision NOT NULL,
    point_median_f double precision NOT NULL,
    sigma_f double precision NOT NULL,
    source_available boolean NOT NULL DEFAULT true,
    n_training_samples integer,
    diagnostics_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (target_date, cutoff_id, expert_id, model_version_id, prediction_mode),
    CHECK (prediction_mode IN ('oof','live','backtest_refit')),
    CHECK (sigma_f >= 0.25)
);
```

#### 1.5.19 `predictions.expert_prediction_pmf`

```sql
CREATE TABLE predictions.expert_prediction_pmf (
    expert_prediction_id uuid NOT NULL REFERENCES predictions.expert_predictions(expert_prediction_id) ON DELETE CASCADE,
    temp_f integer NOT NULL,
    probability double precision NOT NULL,
    PRIMARY KEY (expert_prediction_id, temp_f),
    CHECK (temp_f BETWEEN 50 AND 115),
    CHECK (probability >= 0 AND probability <= 1)
);
```

#### 1.5.20 `predictions.final_predictions`

```sql
CREATE TABLE predictions.final_predictions (
    final_prediction_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    target_date date NOT NULL,
    cutoff_id text NOT NULL,
    model_version_id uuid NOT NULL REFERENCES registry.model_versions(model_version_id),
    calibration_version_id uuid REFERENCES registry.model_versions(model_version_id),
    feature_version_id uuid NOT NULL REFERENCES registry.feature_versions(feature_version_id),
    prediction_mode text NOT NULL,
    point_mean_f double precision NOT NULL,
    point_median_f double precision NOT NULL,
    entropy double precision NOT NULL,
    model_disagreement_score double precision,
    calibration_se_mean double precision,
    diagnostics_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (target_date, cutoff_id, model_version_id, prediction_mode),
    CHECK (prediction_mode IN ('oof','live','backtest_refit'))
);
```

#### 1.5.21 `predictions.final_prediction_pmf`

```sql
CREATE TABLE predictions.final_prediction_pmf (
    final_prediction_id uuid NOT NULL REFERENCES predictions.final_predictions(final_prediction_id) ON DELETE CASCADE,
    temp_f integer NOT NULL,
    probability_raw double precision NOT NULL,
    probability_calibrated double precision NOT NULL,
    PRIMARY KEY (final_prediction_id, temp_f),
    CHECK (temp_f BETWEEN 50 AND 115),
    CHECK (probability_raw >= 0 AND probability_raw <= 1),
    CHECK (probability_calibrated >= 0 AND probability_calibrated <= 1)
);
```

#### 1.5.22 `predictions.calibration_artifacts`

```sql
CREATE TABLE predictions.calibration_artifacts (
    calibration_artifact_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version_id uuid NOT NULL REFERENCES registry.model_versions(model_version_id),
    cutoff_id text,
    regime_name text,
    calibrator_type text NOT NULL,
    temp_threshold_f integer,
    n_samples integer NOT NULL,
    oof_bucket_logloss_before double precision,
    oof_bucket_logloss_after double precision,
    ece_before double precision,
    ece_after double precision,
    artifact_uri text NOT NULL,
    artifact_hash text NOT NULL,
    selection_reason text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (calibrator_type IN ('identity','logistic_threshold','isotonic_threshold'))
);
```

#### 1.5.23 `trading.market_snapshots`

```sql
CREATE TABLE trading.market_snapshots (
    market_snapshot_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    polymarket_market_id text NOT NULL REFERENCES silver.polymarket_markets(polymarket_market_id),
    snapshot_at_utc timestamptz NOT NULL,
    target_date date,
    cutoff_id text,
    market_title text,
    is_active boolean,
    is_closed boolean,
    enable_order_book boolean,
    fees_enabled boolean,
    category text,
    raw_json jsonb NOT NULL,
    UNIQUE (polymarket_market_id, snapshot_at_utc)
);
```

#### 1.5.24 `trading.orderbook_levels`

```sql
CREATE TABLE trading.orderbook_levels (
    orderbook_level_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    market_snapshot_id uuid NOT NULL REFERENCES trading.market_snapshots(market_snapshot_id) ON DELETE CASCADE,
    token_id text NOT NULL,
    side text NOT NULL,
    price numeric(18,8) NOT NULL,
    size_shares numeric(18,8) NOT NULL,
    level_index integer NOT NULL,
    raw_json jsonb,
    CHECK (side IN ('bid','ask')),
    CHECK (price >= 0 AND price <= 1),
    CHECK (size_shares >= 0),
    UNIQUE (market_snapshot_id, token_id, side, level_index)
);
```

#### 1.5.25 `trading.trade_decisions`

```sql
CREATE TABLE trading.trade_decisions (
    trade_decision_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    final_prediction_id uuid NOT NULL REFERENCES predictions.final_predictions(final_prediction_id),
    market_snapshot_id uuid NOT NULL REFERENCES trading.market_snapshots(market_snapshot_id),
    decision_mode text NOT NULL,
    bankroll_usdc numeric(18,8) NOT NULL,
    max_edge double precision,
    selected_outcome_id uuid REFERENCES silver.polymarket_outcomes(outcome_id),
    action text NOT NULL,
    reason_codes text[] NOT NULL,
    risk_summary_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (decision_mode IN ('backtest','paper','live')),
    CHECK (action IN ('buy_yes','sell_yes','buy_no','sell_no','no_trade')),
    UNIQUE (final_prediction_id, market_snapshot_id, decision_mode)
);
```

#### 1.5.26 `trading.trade_decision_legs`

```sql
CREATE TABLE trading.trade_decision_legs (
    trade_decision_leg_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    trade_decision_id uuid NOT NULL REFERENCES trading.trade_decisions(trade_decision_id) ON DELETE CASCADE,
    outcome_id uuid NOT NULL REFERENCES silver.polymarket_outcomes(outcome_id),
    token_id text NOT NULL,
    side text NOT NULL,
    fair_probability double precision NOT NULL,
    market_price double precision,
    vwap_price_25 double precision,
    vwap_price_100 double precision,
    vwap_price_250 double precision,
    vwap_price_500 double precision,
    edge double precision,
    uncertainty_buffer double precision,
    recommended_notional_usdc numeric(18,8) NOT NULL DEFAULT 0,
    recommended_shares numeric(18,8) NOT NULL DEFAULT 0,
    reason_codes text[] NOT NULL DEFAULT '{}',
    CHECK (side IN ('buy','sell','none'))
);
```

#### 1.5.27 `trading.simulated_fills`

```sql
CREATE TABLE trading.simulated_fills (
    simulated_fill_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    trade_decision_leg_id uuid NOT NULL REFERENCES trading.trade_decision_legs(trade_decision_leg_id),
    fill_model text NOT NULL,
    filled_shares numeric(18,8) NOT NULL,
    avg_fill_price numeric(18,8) NOT NULL,
    fee_usdc numeric(18,8) NOT NULL DEFAULT 0,
    notional_usdc numeric(18,8) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (fill_model IN ('instant_taker_vwap','maker_midpoint_no_fill','maker_queue_conservative'))
);
```

### 1.6 Required views

Codex must create these views or materialized views.

#### 1.6.1 `gold.v_feature_matrix_flat`

One row per target/cutoff/feature version with JSON features plus label. This is a plain view over `gold.feature_matrix`.

#### 1.6.2 `predictions.v_final_prediction_bucket_probs`

Columns:

```text
final_prediction_id, polymarket_market_id, outcome_id, outcome_name,
lower_temp_f, upper_temp_f, bucket_probability_raw, bucket_probability_calibrated
```

Bucket probability formula is defined in Section 8.

#### 1.6.3 `trading.v_latest_market_snapshot`

Latest snapshot per `polymarket_market_id` by `snapshot_at_utc`.

#### 1.6.4 `reports.v_backtest_daily_scores`

One row per target date/cutoff/final model with:

```text
target_date, cutoff_id, y_true, point_mean_f, point_median_f,
absolute_error_mean, absolute_error_median, crps, log_loss_integer,
model_disagreement_score, calibration_se_mean
```

### 1.7 Raw-to-feature traceability contract

Every feature in `gold.feature_values` must trace to raw or silver inputs. Minimum trace fields:

```text
formula_id
formula_version
input_table_names
source_record_ids
availability_ids
input_row_primary_keys
max_input_effective_available_at_utc
cutoff_utc
missing_policy
source_code_git_sha
```

If a feature is derived entirely from calendar constants, set:

```json
{"input_table_names": [], "source_record_ids": [], "availability_ids": [], "calendar_only": true}
```

### 1.8 Bronze/silver/gold/predictions/trading relationships

The required lineage is:

```text
bronze.source_records
    -> silver normalized tables + silver.availability_ledger
        -> gold.feature_values
            -> gold.feature_matrix
                -> predictions.expert_predictions + predictions.expert_prediction_pmf
                    -> predictions.final_predictions + predictions.final_prediction_pmf
                        -> trading.trade_decisions + trading.trade_decision_legs
                            -> trading.simulated_fills or live order records
```

No model may read directly from `bronze` except traceability verification. Models read only `gold.feature_matrix` and prediction tables.

### 1.9 Row uniqueness rules

| Entity | Uniqueness |
|---|---|
| Source record | `(source_name, provider_name, endpoint_name, provider_record_key, revision_number)` |
| Availability ledger | source/model/station/member/run/valid/variable canonical identity defined in DDL |
| Gold feature | `(target_date, cutoff_id, feature_version_id, feature_name)` |
| Feature matrix | `(target_date, cutoff_id, feature_version_id)` |
| Expert prediction | `(target_date, cutoff_id, expert_id, model_version_id, prediction_mode)` |
| Expert PMF row | `(expert_prediction_id, temp_f)` |
| Final prediction | `(target_date, cutoff_id, model_version_id, prediction_mode)` |
| Final PMF row | `(final_prediction_id, temp_f)` |
| Calibration artifact | unique artifact hash per model/cutoff/regime/threshold; duplicates must reuse artifact row |
| Market snapshot | `(polymarket_market_id, snapshot_at_utc)` |
| Orderbook level | `(market_snapshot_id, token_id, side, level_index)` |
| Trade decision | `(final_prediction_id, market_snapshot_id, decision_mode)` |
| Simulated fill | one or more fills per `trade_decision_leg_id`; no unique constraint beyond primary key |

### 1.10 Missing, stale, duplicate, and revision handling

#### Missing source values

A feature with insufficient source values must be inserted with:

```text
is_missing=true
value_float=null
missing_reason=<enum>
<feature_name>__missing=true in feature_matrix
```

Allowed missing reasons:

```text
source_absent
source_not_available_by_cutoff
insufficient_history
insufficient_valid_hours
all_values_null
unsupported_product
parse_failed
quality_filtered
```

#### Stale source values

Stale data is not automatically missing. It must generate both the original feature and a staleness feature. If source age exceeds hard maximum in Section 2.18, set the source-specific feature to missing and set missing reason `source_not_available_by_cutoff` or `insufficient_valid_hours` as appropriate.

#### Revisions

Weather actuals and Wunderground labels may revise. Forecast records may also revise due to provider restatement. Rules:

```text
Training labels: use latest current row unless a historical label freeze timestamp exists.
Trading decisions: always store the exact label/feature/prediction versions used at decision time.
Backtest reports: include label_revision_sensitive=true if labels are current-revision labels rather than frozen historical labels.
```

#### Duplicate source rows

Duplicates with same payload hash are ignored. Duplicates with different payload hash become revisions.

---

## 2. Feature formula contract

### 2.1 General feature materialization rules

All features are computed for one `(target_date, cutoff_id)`.

Define:

```python
target_local_start = datetime(target_date.year, target_date.month, target_date.day, 0, 0, tzinfo=ZoneInfo('America/New_York'))
target_local_end   = datetime(target_date.year, target_date.month, target_date.day, 23, 59, 59, tzinfo=ZoneInfo('America/New_York'))
target_utc_start   = target_local_start.astimezone(UTC)
target_utc_end     = target_local_end.astimezone(UTC)
cutoff_utc         = computed from registry.cutoffs
```

Every source row used by a feature must satisfy:

```text
effective_available_at_utc <= cutoff_utc
```

If multiple source rows are otherwise identical, use the row with latest `effective_available_at_utc`, then latest `created_at`, but only if not after cutoff.

Temperature conversions:

```python
F_from_C = C * 9.0 / 5.0 + 32.0
F_from_K = (K - 273.15) * 9.0 / 5.0 + 32.0
```

Wind vector conversion if `u` and `v` exist:

```python
wind_speed = sqrt(u*u + v*v)
wind_dir_deg_from = (270.0 - atan2(v, u) * 180.0 / pi) % 360.0
```

This matches GribStream's documented vector conversion convention.

### 2.2 Calendar features

Inputs: target date only. No missing behavior.

| Feature name | Formula | Unit |
|---|---|---|
| `cal_day_of_year` | `target_date.timetuple().tm_yday` | day |
| `cal_month` | month number | integer |
| `cal_is_weekend` | `target_date.weekday() in {5,6}` | bool |
| `cal_sin_doy` | `sin(2*pi*day_of_year/366)` | unitless |
| `cal_cos_doy` | `cos(2*pi*day_of_year/366)` | unitless |
| `cal_warm_season` | month in `{5,6,7,8,9}` | bool |
| `cal_jja` | month in `{6,7,8}` | bool |

### 2.3 Climatology features

Source: `silver.target_daily_actuals` for KLGA Wunderground current labels, using dates strictly before target date and labels available by cutoff.

Eligibility:

```text
local_date < target_date
source_available_at_utc <= cutoff_utc
```

For every window `W` in `{15, 31, 61}` calendar days around day-of-year:

```python
center = day_of_year(target_date)
include historical date d if circular_day_distance(day_of_year(d), center) <= floor(W/2)
exclude current target year
```

Minimum history:

```text
climo_window_15: at least 30 historical days
climo_window_31: at least 75 historical days
climo_window_61: at least 150 historical days
```

Features:

| Feature | Formula | Missing behavior |
|---|---|---|
| `climo_doy31_mean_high_f` | mean high over W=31 | missing if min history not met |
| `climo_doy31_median_high_f` | median high over W=31 | missing if min history not met |
| `climo_doy31_p10_high_f` | 10th percentile | missing if min history not met |
| `climo_doy31_p90_high_f` | 90th percentile | missing if min history not met |
| `climo_doy61_mean_high_f` | mean high over W=61 | fallback for W=31 if W=31 missing |
| `climo_doy15_mean_high_f` | mean high over W=15 | missing if min history not met |
| `climo_anomaly_recent_7_f` | `rolling_actual_high_mean_7 - climo_doy31_mean_high_f` | missing if either missing |

### 2.4 Rolling actual-history features

Source: `silver.station_daily_actuals` and `silver.target_daily_actuals`.

Strict leakage rule:

```text
Use local dates <= target_date - 2 days for all finalized daily-high-error and daily-high rolling features.
```

Reason: at T-1 cutoffs, T-1 high may be unknown or not settled; use T-2 as universal conservative cutoff across all cutoffs.

For station `S` and window `N` in `{1, 3, 5, 7, 14, 30}`:

```python
eligible_dates = local_date in [target_date - (N+1) days, target_date - 2 days]
rolling_mean_high_S_N = mean(high_temp_f over eligible_dates)
rolling_max_high_S_N = max(high_temp_f over eligible_dates)
rolling_min_high_S_N = min(high_temp_f over eligible_dates)
```

Minimum history:

```text
N=1: 1 observation required
N=3: 2 observations required
N=5: 3 observations required
N=7: 5 observations required
N=14: 10 observations required
N=30: 20 observations required
```

Feature names:

```text
actual_<station_id_lower>_high_mean_<N>d_f
actual_<station_id_lower>_high_max_<N>d_f
actual_<station_id_lower>_high_min_<N>d_f
```

For KLGA target source specifically:

```text
actual_klga_high_tminus2_f = high on target_date - 2
actual_klga_high_tminus3_f = high on target_date - 3
actual_klga_high_change_tminus2_minus_tminus3_f = tminus2 - tminus3
```

### 2.5 Nearby-station group features

Station groups are fixed:

```python
STATION_GROUPS = {
    'target': ['KLGA'],
    'urban_core': ['KNYC'],
    'inland_warm': ['KEWR', 'KTEB', 'KHPN'],
    'coastal_marine': ['KJFK', 'KISP', 'KBDR'],
    'upstream_sw': ['KPHL', 'KDCA', 'KBWI'],
    'upstream_nw': ['KSWF', 'KPOU'],
    'backdoor_ne': ['KBOS', 'KBDR', 'KISP'],
    'nyc_airports': ['KLGA', 'KJFK', 'KEWR'],
}
```

For group `G`, window `N=7` and `N=14`:

```python
group_high_mean_G_N = mean(actual_<station>_high_mean_N for stations in group with non-missing values)
group_high_max_G_N = max(actual_<station>_high_mean_N)
group_high_min_G_N = min(actual_<station>_high_mean_N)
```

Minimum: at least 50% of group stations available, rounded up.

Feature names:

```text
group_<group>_high_mean_<N>d_f
group_<group>_high_max_<N>d_f
group_<group>_high_min_<N>d_f
```

Gradients:

```python
grad_actual_inland_minus_klga_7d_f = group_inland_warm_high_mean_7d_f - actual_klga_high_mean_7d_f
grad_actual_coastal_minus_klga_7d_f = group_coastal_marine_high_mean_7d_f - actual_klga_high_mean_7d_f
grad_actual_sw_minus_klga_7d_f = group_upstream_sw_high_mean_7d_f - actual_klga_high_mean_7d_f
grad_actual_backdoor_ne_minus_klga_7d_f = group_backdoor_ne_high_mean_7d_f - actual_klga_high_mean_7d_f
grad_actual_inland_minus_coastal_7d_f = group_inland_warm_high_mean_7d_f - group_coastal_marine_high_mean_7d_f
```

### 2.6 MOS scalar extraction

Source: `silver.mos_guidance`.

Products: `MAV`, `MET`, `MEX`, `LAV`, `NBS`, `NBE`.

Eligible rows:

```text
station_id = KLGA or nearby station
model_cycle_utc <= cutoff_utc
effective_available_at_utc <= cutoff_utc
forecast period overlaps target local day by at least 6 hours OR forecast_valid_time_utc inside target day
```

For every product `P`, station `S`, variable code `V`, and target date/cutoff:

1. Keep rows satisfying availability.
2. Choose latest `model_cycle_utc`.
3. For scalar daily max variables, use rows with variable codes in priority order:

```text
txn > n_x > max_temp > xnd-adjusted n_x
```

4. If multiple candidate forecast periods overlap target day, choose the candidate with greatest overlap hours. Tie-breaker: forecast period midpoint closest to target local 15:00.
5. Convert value to Fahrenheit if needed.

Required MOS scalar features for KLGA:

```text
mos_<product_lower>_klga_tmax_f
mos_<product_lower>_klga_temp_peak_window_max_f
mos_<product_lower>_klga_dewpoint_peak_window_mean_f
mos_<product_lower>_klga_cloud_peak_window_mean_pct
mos_<product_lower>_klga_wind_speed_peak_window_mean_kt
mos_<product_lower>_klga_pop_max_pct
mos_<product_lower>_klga_qpf_max_in
mos_<product_lower>_klga_tstorm_prob_max_pct
mos_<product_lower>_klga_temp_std_f
mos_<product_lower>_klga_tmax_std_f
```

Variable mapping:

| MOS code | Meaning | Unit handling |
|---|---|---|
| `n_x` | max/min temp | °F |
| `txn` | max/min temp variant | °F |
| `tmp` | valid-time temp | °F |
| `dpt` | dewpoint | °F |
| `cld`/`sky` | cloud/sky | convert categorical to pct if needed |
| `wsp` | wind speed | kt |
| `wdr` | wind direction | tens of degrees -> deg = value*10 |
| `p06`, `p12`, `p24`, `p01` | precip probability | pct |
| `q06`, `q12`, `q24` | QPF | hundredths inches -> inches = value/100 |
| `t03`, `t06`, `t12`, `t24` | thunderstorm probability | pct |
| `tsd`, `xnd` | standard deviation | °F |

Cloud categorical conversion:

```python
MOS_CLOUD_TO_PCT = {
    'CL': 0,
    'FW': 20,
    'SC': 40,
    'BK': 75,
    'OV': 100,
}
```

If cloud value is numeric 0..100, use as-is. If numeric 0..8 oktas, convert `value/8*100`.

### 2.7 MOS disagreement features

Using available KLGA MOS tmax scalar forecasts:

```python
mos_tmax_values = [mos_mav_klga_tmax_f, mos_met_klga_tmax_f, mos_mex_klga_tmax_f, mos_lav_klga_tmax_f, mos_nbs_klga_tmax_f, mos_nbe_klga_tmax_f]
```

Use non-missing values only. Minimum 2 values.

Features:

```python
mos_tmax_mean_f = mean(values)
mos_tmax_median_f = median(values)
mos_tmax_std_f = sample_std(values) if len(values) >= 2 else missing
mos_tmax_range_f = max(values) - min(values)
mos_tmax_max_minus_min_f = mos_tmax_range_f
mos_nbm_minus_mos_mean_f = nbm_klga_tmax_f - mos_tmax_mean_f
mos_lav_minus_mav_tmax_f = mos_lav_klga_tmax_f - mos_mav_klga_tmax_f
mos_nbe_minus_nbs_tmax_f = mos_nbe_klga_tmax_f - mos_nbs_klga_tmax_f
```

Missing behavior: if only one MOS value is available, mean/median are computed and std/range missing.

### 2.8 GribStream scalar forecast extraction

Source: `silver.grib_forecast_values`.

Models grouped:

```python
GRIB_CORE_MODELS = ['nbm','nbmqmd','hrrr','rap','gfs','gefsatmosmean','ifsoper','aifsoper','aigfssfc']
GRIB_ENSEMBLE_MODELS = ['gefsatmos','ifsenfo','aifsenfo','aigefssfc']
GRIB_AUDITION_MODELS = ['rrfs2dfld','rrfsprslev','refsprslev','spctstm1hr','spctstm4hr','spcltg4hr','spcwind4hr','spchail4hr','spctor4hr','uvi']
```

Valid-time windows:

```python
TARGET_DAY_WINDOW = target local 00:00..23:59 converted to UTC
PEAK_WINDOW_LOCAL = target local hours 12:00..19:00 inclusive
PEAK_WINDOW_UTC = each local peak hour converted to UTC
MORNING_WINDOW_LOCAL = 06:00..11:00
AFTERNOON_WINDOW_LOCAL = 12:00..18:00
EVENING_WINDOW_LOCAL = 18:00..23:00
```

Latest eligible run selection for deterministic model `M`:

```python
eligible_rows = rows where model_name=M and effective_available_at_utc <= cutoff_utc
candidate_runs = unique run_time_utc
For each run, coverage_ratio = count(valid hourly temps in target day) / expected_hour_count
Choose latest run with coverage_ratio >= 0.70.
If none, choose latest run with coverage_ratio >= 0.50 and set feature <model>_coverage_low=true.
If none, model scalar features missing.
```

Expected hourly count is the number of integer local hours inside target local day after timezone conversion. Usually 24, but DST transition days may be 23 or 25. Codex must compute it, not hard-code 24.

Temperature scalar features for every deterministic model `M` and coord group `C`:

```python
model_M_C_hourly_t2m_f = converted hourly 2m temperature values
model_M_C_tmax_hourly_f = max(hourly t2m over TARGET_DAY_WINDOW)
model_M_C_tmax_peak_window_f = max(hourly t2m over PEAK_WINDOW_LOCAL)
model_M_C_temp_15local_f = t2m nearest target local 15:00 valid time
model_M_C_temp_18local_f = t2m nearest target local 18:00 valid time
model_M_C_temp_peak_hour_local = local hour at which hourly t2m max occurs
```

Feature names:

```text
grib_<model>_<coord>_tmax_hourly_f
grib_<model>_<coord>_tmax_peak_window_f
grib_<model>_<coord>_temp_15local_f
grib_<model>_<coord>_temp_18local_f
grib_<model>_<coord>_temp_peak_hour_local
grib_<model>_<coord>_coverage_ratio
```

Pressure-level and regime features for levels 925 and 850 hPa, if available:

```text
grib_<model>_<coord>_temp_925_peak_mean_f
grib_<model>_<coord>_temp_850_peak_mean_f
grib_<model>_<coord>_wind_925_peak_mean_kt
grib_<model>_<coord>_wind_850_peak_mean_kt
grib_<model>_<coord>_wind_925_peak_dir_deg
grib_<model>_<coord>_wind_850_peak_dir_deg
```

All feature names in this subsection must use the ASCII prefix `grib_`.

### 2.9 Run-to-run trend formulas

For each deterministic model `M`, coordinate `C`, and scalar `S`:

```python
latest_run = selected latest eligible run
previous_run = max eligible run_time_utc < latest_run with same model/coord and coverage_ratio >= 0.50
trend = scalar(latest_run) - scalar(previous_run)
run_age_hours = (cutoff_utc - latest_run).total_seconds() / 3600
previous_run_gap_hours = (latest_run - previous_run).total_seconds() / 3600
```

Feature names:

```text
grib_<model>_<coord>_<scalar>_runtrend_f
grib_<model>_<coord>_<scalar>_run_age_hours
grib_<model>_<coord>_<scalar>_previous_run_gap_hours
```


Minimum: previous run must exist within 18 hours for HRRR/RAP/NBM, 30 hours for GFS/IFS/AI global models. Otherwise trend missing.

### 2.10 Pseudo-point gradient formulas

Required pseudo coord names:

```text
klga_core
inland_nj
marine_jfk_bay
upstream_sw
backdoor_ne
```

For every deterministic grib model where at least `klga_core` and another pseudo point exist:

```python
grad_grib_<model>_inland_minus_klga_tmax_f = tmax(inland_nj) - tmax(klga_core)
grad_grib_<model>_klga_minus_marine_tmax_f = tmax(klga_core) - tmax(marine_jfk_bay)
grad_grib_<model>_sw_minus_klga_tmax_f = tmax(upstream_sw) - tmax(klga_core)
grad_grib_<model>_backdoor_ne_minus_klga_tmax_f = tmax(backdoor_ne) - tmax(klga_core)
grad_grib_<model>_inland_minus_marine_tmax_f = tmax(inland_nj) - tmax(marine_jfk_bay)
```

Missing: if either side missing, gradient missing.

### 2.11 Peak-window definition

For any hourly forecast/observation series on target day:

```text
peak_window_local_start = 12:00 America/New_York on target date
peak_window_local_end   = 19:00 America/New_York on target date
```

Hourly timestamps included:

```python
valid local hour in {12,13,14,15,16,17,18,19}
```

If model provides subhourly data, aggregate to hourly by taking the last value whose valid time is within `[hour:00, hour:59:59]`; if multiple values exist exactly at same valid time, use latest available row.

### 2.12 Sea-breeze risk proxy

Inputs, preferred from HRRR latest eligible run at `klga_core`; fallback RAP; fallback NBM; fallback GFS. For each input, use peak-window mean unless otherwise stated.

Required intermediate functions:

```python
def clip01(x): return max(0.0, min(1.0, x))

def angular_band_score(direction_deg, center_deg, half_width_deg):
    # direction is meteorological FROM direction
    diff = abs(((direction_deg - center_deg + 180) % 360) - 180)
    return clip01(1 - diff / half_width_deg)

def speed_band_score(speed_kt, low_kt, high_kt):
    if speed_kt is None: return missing
    if speed_kt < low_kt: return clip01(speed_kt / low_kt)
    if speed_kt <= high_kt: return 1.0
    return clip01(1 - (speed_kt - high_kt) / high_kt)
```

Inputs:

```text
wind_dir_10m_peak_deg
wind_speed_10m_peak_kt
grad_grib_hrrr_inland_minus_marine_tmax_f or fallback analogous gradient
cloud_peak_mean_pct
precip_peak_max_in
cal_warm_season
```

Formula:

```python
marine_dir_score = max(
    angular_band_score(wind_dir_10m_peak_deg, 110, 80),
    angular_band_score(wind_dir_10m_peak_deg, 160, 45)
)
marine_speed_score = speed_band_score(wind_speed_10m_peak_kt, 4, 18)
inland_heat_gradient_score = clip01((inland_minus_marine_tmax_f - 3.0) / 7.0)
clear_enough_score = clip01((85.0 - cloud_peak_mean_pct) / 60.0)
dry_enough_score = clip01((0.10 - precip_peak_max_in) / 0.10)
warm_season_score = 1.0 if cal_warm_season else 0.25

sea_breeze_risk_score = clip01(
    0.30 * marine_dir_score +
    0.20 * marine_speed_score +
    0.20 * inland_heat_gradient_score +
    0.15 * clear_enough_score +
    0.10 * dry_enough_score +
    0.05 * warm_season_score
)
```

Feature names:

```text
risk_sea_breeze_score
risk_sea_breeze_inputs_available_count
```

Missing behavior:

```text
If wind direction or wind speed missing: score missing.
If gradient missing: gradient component = 0.5 and inputs_available_count decremented.
If cloud missing: cloud component = 0.5.
If precip missing: precip component = 0.5.
```

Testable example:

```text
wind_dir=130, wind_speed=10, inland_minus_marine=8, cloud=20, precip=0, warm_season=true -> score >= 0.80.
wind_dir=270, wind_speed=12, inland_minus_marine=8, cloud=20, precip=0, warm_season=true -> score <= 0.55.
```

### 2.13 Backdoor-front risk proxy

Preferred inputs: HRRR/RAP/GFS/IFS latest eligible run and recent observations.

Inputs:

```text
wind_dir_10m_peak_deg
wind_speed_10m_peak_kt
grad_backdoor_ne_minus_klga_tmax_f
pressure_change_3h_mb at KLGA from observations, if cutoff-day obs available
cloud_peak_mean_pct
```

Formula:

```python
ne_e_flow_score = max(
    angular_band_score(wind_dir_10m_peak_deg, 45, 65),
    angular_band_score(wind_dir_10m_peak_deg, 75, 55)
)
wind_score = speed_band_score(wind_speed_10m_peak_kt, 6, 22)
ne_cooler_score = clip01((0.0 - grad_backdoor_ne_minus_klga_tmax_f) / 6.0)
pressure_rise_score = clip01((pressure_change_3h_mb + 0.5) / 4.0)
cloud_score = clip01((cloud_peak_mean_pct - 40.0) / 60.0)

risk_backdoor_front_score = clip01(
    0.35 * ne_e_flow_score +
    0.20 * wind_score +
    0.20 * ne_cooler_score +
    0.15 * pressure_rise_score +
    0.10 * cloud_score
)
```

If pressure change missing, use `pressure_rise_score=0.5`.

### 2.14 Marine-layer risk proxy

Inputs:

```text
risk_sea_breeze_score
cloud_low_peak_mean_pct or cloud_peak_mean_pct
klga_dewpoint_peak_mean_f
klga_temp_dewpoint_spread_peak_mean_f
wind_dir_10m_peak_deg
```

Formula:

```python
marine_flow_score = max(angular_band_score(wind_dir_10m_peak_deg, 100, 75), angular_band_score(wind_dir_10m_peak_deg, 150, 50))
low_cloud_score = clip01((cloud_low_peak_mean_pct - 35.0) / 65.0)
humidity_score = clip01((klga_dewpoint_peak_mean_f - 58.0) / 14.0)
small_spread_score = clip01((10.0 - temp_dewpoint_spread_peak_mean_f) / 10.0)

risk_marine_layer_score = clip01(
    0.25 * risk_sea_breeze_score +
    0.25 * marine_flow_score +
    0.25 * low_cloud_score +
    0.15 * humidity_score +
    0.10 * small_spread_score
)
```

Missing fallback: missing components receive 0.5 except `risk_sea_breeze_score`, which receives 0 if missing.

### 2.15 Cloud-bust risk score

Purpose: probability that cloud forecast uncertainty creates a Tmax bust.

Inputs:

```text
cloud_peak_mean_pct from NBM/HRRR/RAP/GFS
cloud_peak_range_across_models_pct
qpf_peak_max_in
pop_max_pct
shortwave_peak_mean_wm2 if available
model_tmax_disagreement_f
```

Formula:

```python
cloud_amount_score = clip01((cloud_peak_mean_pct - 45.0) / 55.0)
cloud_disagreement_score = clip01(cloud_peak_range_across_models_pct / 60.0)
precip_score = max(clip01(pop_max_pct / 70.0), clip01(qpf_peak_max_in / 0.25))
shortwave_uncertainty_score = 0.5 if shortwave missing else clip01((750.0 - shortwave_peak_mean_wm2) / 450.0)
tmax_disagreement_score = clip01(model_tmax_disagreement_f / 4.0)

risk_cloud_bust_score = clip01(
    0.25 * cloud_amount_score +
    0.25 * cloud_disagreement_score +
    0.20 * precip_score +
    0.15 * shortwave_uncertainty_score +
    0.15 * tmax_disagreement_score
)
```

### 2.16 Storm-outflow risk score

Inputs:

```text
mos_tstorm_prob_max_pct
spc_tstorm_prob_max_pct if available
qpf_peak_max_in
hrrr_precip_peak_max_in
hrrr_reflectivity_proxy if available
cape_peak_jkg if available
wind_gust_peak_kt if available
```

Formula:

```python
tstorm_prob_score = clip01(max(mos_tstorm_prob_max_pct, spc_tstorm_prob_max_pct or 0) / 50.0)
qpf_score = clip01(max(qpf_peak_max_in, hrrr_precip_peak_max_in or 0) / 0.35)
reflectivity_score = 0.5 if reflectivity missing else clip01((reflectivity_proxy_dbz - 25.0) / 25.0)
cape_score = 0.5 if cape missing else clip01(cape_peak_jkg / 1500.0)
gust_score = 0.5 if wind_gust_peak_kt missing else clip01((wind_gust_peak_kt - 20.0) / 25.0)

risk_storm_outflow_score = clip01(
    0.30 * tstorm_prob_score +
    0.25 * qpf_score +
    0.15 * reflectivity_score +
    0.15 * cape_score +
    0.15 * gust_score
)
```

### 2.17 Advection and mixing proxies

Preferred source order: HRRR, RAP, NBM, GFS, IFS.

Warm/cold advection proxy at 925 hPa:

```python
morning_925_temp_f = mean 925 hPa temp over local 06:00..11:00
peak_925_temp_f = mean 925 hPa temp over local 12:00..19:00
advect_925_warming_f = peak_925_temp_f - morning_925_temp_f
```

Feature:

```text
advect_<model>_925_warming_morning_to_peak_f
```

Westerly mixing/downsloping proxy:

```python
westerly_dir_score = angular_band_score(wind_dir_925_peak_deg, 270, 70)
westerly_mixing_proxy = westerly_dir_score * wind_speed_925_peak_kt
```

Feature:

```text
mixing_<model>_westerly_925_proxy_kt
```

Northeasterly cooling proxy:

```python
ne_dir_score = angular_band_score(wind_dir_925_peak_deg, 45, 70)
ne_cooling_proxy = ne_dir_score * wind_speed_925_peak_kt
```

Feature:

```text
cooling_<model>_ne_925_proxy_kt
```

### 2.18 Ensemble features

Source: `silver.grib_forecast_values` for member-level ensemble models.

For each ensemble model `E` and coordinate `klga_core`:

1. For each member, compute member daily Tmax over target local day using same hourly extraction as deterministic models.
2. Require at least:

```text
GEFS/AIGEFS: >= 15 members
IFS ENS/AIFS ENS: >= 25 members
```

3. Compute features:

```python
values = member_tmax_f array
ens_<model>_member_count
ens_<model>_tmax_mean_f
ens_<model>_tmax_median_f
ens_<model>_tmax_std_f = sample std
ens_<model>_tmax_p05_f
ens_<model>_tmax_p10_f
ens_<model>_tmax_p25_f
ens_<model>_tmax_p75_f
ens_<model>_tmax_p90_f
ens_<model>_tmax_p95_f
ens_<model>_tmax_iqr_f = p75 - p25
ens_<model>_tmax_p90_minus_p10_f = p90 - p10
ens_<model>_tmax_skew = scipy.stats.skew(values, bias=False) if n>=8 else missing
```

Threshold probabilities for every integer `k` from 50 to 115:

```python
ens_<model>_prob_ge_<k>f = mean(values >= k)
ens_<model>_prob_eq_<k>f_raw = mean(round(values) == k)
```

Smoothed PMF for ensemble expert:

```python
For each member value x_i, assign Gaussian kernel N(x_i, sigma=1.0°F) over integer grid.
ensemble_pmf[k] = mean_i Integral from k-0.5 to k+0.5 of N(x_i,1.0) dx
Apply probability floor 1e-6 and renormalize.
```

### 2.19 Dynamic recent-error features

For each base scalar source `S` that produces a point tmax forecast:

```text
S examples: mos_mav, mos_lav, mos_nbs, mos_nbe, nbm, hrrr, rap, gfs, gefsmean, ifsoper, aifsoper, aigfssfc
```

Let `f_S,d,c` be the scalar forecast for target date `d` at cutoff `c`, computed by identical as-of feature rules. Let `y_d` be Wunderground KLGA high.

For target date `T`, eligible error dates:

```python
error_dates = dates d where d <= T - 2 days and f_S,d,c was available and y_d is known
```

EWMA weight with half-life `h` days:

```python
age_days = (T - d).days
w_d = 0.5 ** (age_days / h)
```

Bias:

```python
ewma_bias_S_h = sum(w_d * (f_S,d,c - y_d)) / sum(w_d)
```

MAE after bias:

```python
ewma_mae_S_h = sum(w_d * abs((f_S,d,c - ewma_bias_S_h) - y_d)) / sum(w_d)
```

Features for default half-lives:

```text
bias half-lives: 14, 21, 30
mae half-lives: 45, 60, 90
```

Feature names:

```text
recenterr_<source>_<cutoff_id_lower>_bias_h<h>_f
recenterr_<source>_<cutoff_id_lower>_mae_h<h>_f
recenterr_<source>_<cutoff_id_lower>_n_h<h>
```

Minimum history:

```text
bias: at least 10 prior error dates
mae: at least 20 prior error dates
```

### 2.20 Dynamic composite features

Using selected sources that have point forecasts and dynamic MAE:

```python
SOURCE_FAMILY_CAPS = {
    'mos': 0.35,
    'nbm': 0.35,
    'hrrr_rap': 0.30,
    'global_det': 0.25,
    'global_ens': 0.25,
    'ai': 0.15,
}
```

Default hyperparameters:

```python
BIAS_HALF_LIFE_DEFAULT = 21
MAE_HALF_LIFE_DEFAULT = 60
SKILL_WEIGHT_EXPONENT_DEFAULT = 2.0
SKILL_WEIGHT_FLOOR_F_DEFAULT = 1.0
```

For each source `S`:

```python
corrected_forecast_S = raw_forecast_S - recenterr_S_bias_h21_f
raw_weight_S = 1.0 / ((recenterr_S_mae_h60_f + 1.0) ** 2.0)
```

If MAE missing but source forecast exists:

```python
recenterr_S_mae_h60_f = source_family_default_mae
```

Family default MAE:

```text
mos: 2.3°F
nbm: 2.1°F
hrrr_rap: 2.5°F
global_det: 2.7°F
global_ens: 2.4°F
ai: 2.8°F
```

Apply family caps:

```python
weights normalized within all sources, then if sum family weights > cap, scale family weights to cap and redistribute excess proportionally to uncapped families. Iterate until all caps satisfied or 10 iterations.
```

Features:

```text
dyncomp_point_f
dyncomp_weight_<source>
dyncomp_n_sources
dyncomp_weighted_recent_mae_f = sum(weight_S * recenterr_S_mae_h60_f)
dyncomp_raw_model_std_f = sample std(raw_forecast_S)
dyncomp_corrected_model_std_f = sample std(corrected_forecast_S)
```

Minimum sources: at least 3 source forecasts. If fewer, missing.

### 2.21 Observation features at cutoff

Source: `silver.station_observations`.

Observation eligibility:

```text
effective_available_at_utc <= cutoff_utc - 10 minutes
observed_at_utc <= cutoff_utc
quality_flag in ('ok','suspect')
```

Latest observation for station `S`:

```python
latest_obs = max observed_at_utc satisfying eligibility and observed_at_utc >= cutoff_utc - 6 hours
```

Features:

```text
obs_<station>_latest_age_minutes = (cutoff_utc - latest_obs.observed_at_utc)/60
obs_<station>_temp_latest_f
obs_<station>_dewpoint_latest_f
obs_<station>_wind_speed_latest_kt
obs_<station>_wind_dir_latest_deg
obs_<station>_slp_latest_mb
obs_<station>_temp_dewpoint_spread_latest_f = temp - dewpoint
```

Warming rate:

```python
obs_30m = nearest obs to cutoff_utc - 30 minutes within ±20 minutes
obs_3h = nearest obs to cutoff_utc - 3 hours within ±45 minutes
warming_rate_3h_f_per_hour = (obs_30m.temp_f - obs_3h.temp_f) / ((obs_30m.time - obs_3h.time).hours)
```

Feature:

```text
obs_<station>_warming_rate_3h_f_per_hour
```

Wind shift:

```python
wind_shift_3h_deg = abs(((latest_wind_dir - wind_dir_3h + 180) % 360) - 180)
```

Pressure change:

```python
pressure_change_3h_mb = latest_slp_mb - slp_3h_mb
```

Group gradients at cutoff:

```python
obs_grad_inland_minus_klga_temp_latest_f = mean(latest temp KEWR,KTEB,KHPN) - latest temp KLGA
obs_grad_coastal_minus_klga_temp_latest_f = mean(latest temp KJFK,KISP,KBDR) - latest temp KLGA
obs_grad_sw_minus_klga_temp_latest_f = mean(latest temp KPHL,KDCA,KBWI) - latest temp KLGA
obs_grad_backdoor_ne_minus_klga_temp_latest_f = mean(latest temp KBOS,KBDR,KISP) - latest temp KLGA
```

### 2.22 Settlement-discrepancy features

Sources:

```text
silver.target_daily_actuals for Wunderground KLGA
silver.station_daily_actuals for IEM/ASOS/official KLGA, if available
```

For historical date `d <= T-2`:

```python
discrepancy_d = wunderground_klga_high_d - official_or_iem_klga_high_d
```

Features:

```python
settle_disc_mean_365d_f = mean(discrepancy over last 365 eligible days)
settle_disc_abs_ge1_rate_365d = mean(abs(discrepancy) >= 1)
settle_disc_abs_ge2_rate_365d = mean(abs(discrepancy) >= 2)
settle_disc_recent_30d_mean_f = mean(discrepancy over last 30 eligible days)
settle_disc_recent_30d_abs_ge1_rate = mean(abs(discrepancy) >= 1)
```

Minimum history:

```text
365d features: at least 200 paired days
30d features: at least 15 paired days
```

Fallback:

```text
settle_disc_mean_365d_f = 0.0
settle_disc_abs_ge1_rate_365d = 0.03
settle_disc_abs_ge2_rate_365d = 0.005
```

### 2.23 Model disagreement score

Inputs: expert/source point forecasts available for target/cutoff before final meta-combiner.

Use point forecasts from:

```text
mos_tmax_mean_f
nbm_tmax
hrrr_tmax
rap_tmax
gfs_tmax
gefs_mean_tmax
ifsoper_tmax
aifsoper_tmax
ai/aigfs tmax if available
dyncomp_point_f
```

Formula:

```python
values = non_missing point forecasts
robust_center = median(values)
mad = median(abs(values - robust_center))
robust_sigma = 1.4826 * mad
sample_sigma = sample_std(values) if len(values) >= 2 else 0
model_disagreement_f = max(robust_sigma, 0.5 * sample_sigma)
model_disagreement_score = clip01(model_disagreement_f / 4.0)
```

Minimum: at least 3 values. If fewer, set missing and regime `stale_data=true`.

### 2.24 Calibration standard error

For any calibrated bucket probability `p` and effective calibration sample count `n_eff`:

```python
calibration_se = sqrt(max(p * (1 - p), 1e-6) / max(n_eff, 1))
```

For final prediction mean calibration SE:

```python
calibration_se_mean = mean(calibration_se over market buckets if market exists; otherwise mean over integer temp thresholds 60..105)
```

If no calibration artifact exists:

```text
calibration_se_mean = 0.05
calibration_se_source = 'fallback_no_calibrator'
```

### 2.25 Data staleness score

For each source family, define maximum acceptable age at cutoff:

| Source family | Max age hours |
|---|---:|
| MOS | 18 |
| NBM | 12 |
| HRRR | 8 |
| RAP | 8 |
| GFS | 18 |
| GEFS | 24 |
| IFS | 24 |
| AI global | 24 |
| Observations | 6 |
| Polymarket orderbook | 0.25 |

For source family `S`:

```python
source_age_hours = (cutoff_utc - latest_effective_available_at_utc).hours
staleness_S = clip01((source_age_hours - 0.5 * max_age_S) / (0.5 * max_age_S))
```

If source is required for MVP and missing:

```python
staleness_S = 1.0
```

Overall:

```python
data_staleness_score = max(staleness_S over required source families)
data_staleness_mean_score = mean(staleness_S over source families present or required)
```

Required source families for MVP:

```text
Wunderground labels/history, MOS, NBM, HRRR or RAP, GFS, observations, Polymarket market data
```

### 2.26 Market liquidity score

Inputs: parsed outcomes and orderbook levels for market snapshot.

For each outcome token:

```python
best_bid = max bid price or None
best_ask = min ask price or None
spread = best_ask - best_bid if both exist else 1.0
mid = (best_bid + best_ask)/2 if both exist else best_ask or best_bid or None
ask_depth_5c_usdc = sum(price*size for asks with price <= best_ask + 0.05)
bid_depth_5c_usdc = sum(price*size for bids with price >= best_bid - 0.05)
```

Market-level:

```python
median_spread = median(spread across outcomes with both sides)
total_ask_depth_5c_usdc = sum ask_depth_5c_usdc across outcomes
total_bid_depth_5c_usdc = sum bid_depth_5c_usdc across outcomes
freshness_minutes = snapshot age relative to decision time
```

Score:

```python
spread_score = clip01((0.08 - median_spread) / 0.08)
depth_score = clip01(total_ask_depth_5c_usdc / 1000.0)
bid_depth_score = clip01(total_bid_depth_5c_usdc / 1000.0)
freshness_score = clip01((15.0 - freshness_minutes) / 15.0)
outcome_coverage_score = 1.0 if all parsed outcomes cover TEMP_GRID with no overlap gaps inside realistic 40..120 envelope else 0.0

market_liquidity_score = clip01(
    0.30 * spread_score +
    0.30 * depth_score +
    0.15 * bid_depth_score +
    0.15 * freshness_score +
    0.10 * outcome_coverage_score
)
```

---

## 3. Regime classifier contract

Regime features are deterministic boolean flags materialized into `gold.feature_values`. All thresholds are fixed for MVP unless explicitly marked as tunable in Section 5.

### 3.1 Warm season

```python
regime_warm_season = target_date.month in [5,6,7,8,9]
```

Fallback: never missing.

### 3.2 Cool season

```python
regime_cool_season = not regime_warm_season
```

### 3.3 Heat-wave regime

Inputs:

```text
dyncomp_point_f
mos_tmax_mean_f
nbm_tmax_f
actual_klga_high_mean_3d_f
```

Formula:

```python
forecast_center = median(non_missing [dyncomp_point_f, mos_tmax_mean_f, nbm_tmax_f])
regime_heat_wave = (forecast_center >= 90.0) or (forecast_center >= 88.0 and actual_klga_high_mean_3d_f >= 90.0)
```

Fallback: if all forecast center inputs missing, `false` and stale-data regime likely true.

### 3.4 High model-disagreement regime

```python
regime_high_model_disagreement = (model_disagreement_score >= 0.60) or (model_disagreement_f >= 2.4)
```

Threshold tunability: full-production only; allowed `model_disagreement_f` threshold grid `[2.0, 2.4, 2.8, 3.2]`, choose by OOF bucket log loss improvement at least 0.001.

### 3.5 High ensemble-spread regime

Inputs:

```text
ens_gefsatmos_tmax_p90_minus_p10_f
ens_ifsenfo_tmax_p90_minus_p10_f
ens_aifsenfo_tmax_p90_minus_p10_f
ens_aigefssfc_tmax_p90_minus_p10_f
```

Formula:

```python
spread_values = non_missing p90_minus_p10 values
ensemble_spread_best = median(spread_values) if any else missing
regime_high_ensemble_spread = ensemble_spread_best >= 5.0
```

Fallback: if no ensemble spread, `false` and set `regime_high_ensemble_spread__missing=true`.

### 3.6 Sea-breeze-risk regime

```python
regime_sea_breeze_risk = risk_sea_breeze_score >= 0.65
```

Fallback: if score missing, `false` and missing flag true.

### 3.7 Backdoor-front regime

```python
regime_backdoor_front = risk_backdoor_front_score >= 0.60
```

Fallback: if score missing, `false`.

### 3.8 Marine-layer regime

```python
regime_marine_layer = risk_marine_layer_score >= 0.65
```

### 3.9 Cloud/storm-risk regime

```python
regime_cloud_storm_risk = max(risk_cloud_bust_score, risk_storm_outflow_score) >= 0.60
```

Fallback: missing scores treated as 0.0.

### 3.10 Stale-data regime

```python
regime_stale_data = (data_staleness_score >= 0.60) or (critical_source_missing_count >= 2)
```

Critical source families:

```text
MOS, NBM, HRRR_or_RAP, GFS_or_IFS, observations, Polymarket orderbook if trading decision
```

### 3.11 Settlement-discrepancy-risk regime

```python
regime_settlement_discrepancy_risk = (
    settle_disc_abs_ge1_rate_365d >= 0.08 or
    abs(settle_disc_recent_30d_mean_f) >= 0.35 or
    settle_disc_recent_30d_abs_ge1_rate >= 0.12
)
```

Fallback: use fallback discrepancy values from Section 2.22; normally false.

### 3.12 Thin-book / illiquid-book regime

Inputs from market snapshot.

```python
regime_thin_book = (
    market_liquidity_score < 0.35 or
    median_spread >= 0.08 or
    total_ask_depth_5c_usdc < 250 or
    count_outcomes_with_best_ask < count_parsed_outcomes
)
```

If no market snapshot, `regime_thin_book=true` for trading decision and `missing_market_snapshot` reason.

### 3.13 High-entropy forecast regime

Final calibrated PMF entropy:

```python
entropy = -sum(p_k * log(p_k) for p_k > 0)
max_entropy = log(number of grid points with p_k >= 1e-6)
normalized_entropy = entropy / max_entropy
regime_high_entropy_forecast = normalized_entropy >= 0.70
```

Before final PMF exists, use expert/meta raw PMF. After final PMF exists, update regime field in final prediction diagnostics.

### 3.14 Marginal-edge regime

For best candidate trade:

```python
net_edge = fair_probability - vwap_buy_price - fee_probability_equivalent - uncertainty_buffer
regime_marginal_edge = 0 < net_edge < 0.035
```

If no market, false.

### 3.15 Extreme-boundary regime

This regime is added because bucket-boundary errors drive P&L.

For each market bucket boundary `b`:

```python
distance_to_boundary = min(abs(point_median_f - b) for all bucket boundaries)
regime_near_bucket_boundary = distance_to_boundary <= 0.75
```

If no market buckets parsed, false.

---

## 4. Model implementation contract

### 4.1 Global modeling rules

Every expert must output a full integer PMF on `TEMP_GRID_F`. Every expert PMF must be stored in `predictions.expert_prediction_pmf`.

Every expert must have:

```text
expert_id
model_version_id
feature_version_id
training_start_date
training_end_date
cutoff_id or all-cutoff support
artifact_uri
artifact_hash
```

Targets:

```text
Primary target: label_high_temp_f from gold.feature_matrix.
No expert may train on target rows with label_available=false.
```

Missing features:

```text
For tree models: impute missing numeric values to -9999.0 and include explicit __missing flags.
For linear/logistic models: impute missing numeric values to training median and include __missing flags.
For formulas: follow feature-specific fallback rules.
```

PMF construction from point forecast and sigma:

```python
from scipy.stats import norm

def gaussian_integer_pmf(mu, sigma, grid=TEMP_GRID_F):
    sigma = max(float(sigma), 0.85)
    probs = []
    for k in grid:
        lo = k - 0.5
        hi = k + 0.5
        probs.append(norm.cdf((hi - mu)/sigma) - norm.cdf((lo - mu)/sigma))
    probs = np.maximum(probs, 1e-8)
    return probs / probs.sum()
```

Variance/sigma floors:

```text
Any expert sigma floor: 0.85°F
Final meta PMF implicit sigma floor after calibration: no single integer may exceed 0.72 probability unless supported by at least 500 OOF calibration samples for that cutoff and ECE <= 0.03.
```

Artifact formats:

```text
models: joblib .joblib
hyperparameters: JSON
feature lists: JSON
calibrators: joblib .joblib plus JSON metadata
PMF predictions: SQL rows
reports: Markdown + CSV + JSON + PNG plots
```

### 4.2 Exact feature groups

Codex must implement feature groups as JSON lists in `src/klga_tmax/features/feature_groups.py`.

```python
FEATURE_GROUPS = {
    'calendar': ['cal_*'],
    'climatology': ['climo_*'],
    'actual_history': ['actual_*', 'group_*', 'grad_actual_*'],
    'mos': ['mos_*'],
    'grib_nbm': ['grib_nbm_*', 'grib_nbmqmd_*'],
    'grib_hrrr_rap': ['grib_hrrr_*', 'grib_rap_*', 'grad_grib_hrrr_*', 'grad_grib_rap_*'],
    'grib_global': ['grib_gfs_*', 'grib_ifsoper_*'],
    'ensemble': ['ens_*'],
    'ai': ['grib_aifsoper_*', 'grib_aigfssfc_*', 'ens_aifsenfo_*', 'ens_aigefssfc_*'],
    'observations': ['obs_*'],
    'risk_regime': ['risk_*', 'regime_*'],
    'dynamic_error': ['recenterr_*', 'dyncomp_*'],
    'settlement': ['settle_disc_*'],
    'market': ['market_*'],
}
```

Pattern expansion must be deterministic by sorted feature name.

### 4.3 Default LightGBM regressor

Use for point-forecast experts unless expert-specific rules override.

```python
LGBMRegressor(
    objective='regression_l1',
    n_estimators=700,
    learning_rate=0.025,
    num_leaves=15,
    max_depth=4,
    min_child_samples=35,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_alpha=0.10,
    reg_lambda=1.50,
    random_state=1729,
    n_jobs=1,
    deterministic=True,
    force_col_wise=True,
    verbosity=-1,
)
```

Sklearn fallback:

```python
HistGradientBoostingRegressor(
    loss='absolute_error',
    learning_rate=0.035,
    max_iter=500,
    max_leaf_nodes=15,
    min_samples_leaf=35,
    l2_regularization=0.10,
    random_state=1729,
)
```

### 4.4 Expert A: long-history MOS/station expert

`expert_id = 'long_history_mos_station'`

Included in MVP: yes.

Training rows:

```text
All target/cutoff rows with at least one MOS tmax feature or climatology/actual-history features.
Minimum training samples: 1,000. If fewer, use fallback climatology PMF.
```

Allowed features:

```text
calendar, climatology, actual_history, mos, observations limited to previous-day/older daily actuals only, settlement rolling discrepancy rates
```

Forbidden features:

```text
Any GribStream gridded model feature except NBM MOS-like if present in MOS tables.
Any market feature.
Any observation from target date after cutoff.
```

Model:

```text
LightGBM default regressor predicting Y_T directly.
```

Residual sigma:

```python
sigma = max(0.85, rolling_oof_residual_std_by_cutoff_or_global)
```

If cutoff-specific OOF residual count >= 250, use cutoff-specific sigma. Otherwise use global sigma.

PMF:

```text
Gaussian integer PMF centered at predicted point.
```

Fallback climatology PMF:

```python
mu = climo_doy31_mean_high_f if available else global training mean
sigma = max(3.5, climo_doy31_p90_high_f - climo_doy31_p10_high_f)/2.56 if percentiles available else 6.0
```

### 4.5 Expert B: dynamic bias-corrected composite expert

`expert_id = 'dynamic_bias_corrected_composite'`

Included in MVP: yes.

No ML model. Use formulas from Sections 2.19 and 2.20.

Input point sources for MVP:

```text
mos_mav_klga_tmax_f
mos_lav_klga_tmax_f
mos_nbs_klga_tmax_f
mos_nbe_klga_tmax_f
grib_nbm_klga_core_tmax_hourly_f
grib_hrrr_klga_core_tmax_hourly_f
grib_rap_klga_core_tmax_hourly_f
grib_gfs_klga_core_tmax_hourly_f
grib_gefsatmosmean_klga_core_tmax_hourly_f
grib_ifsoper_klga_core_tmax_hourly_f, if available
grib_aifsoper_klga_core_tmax_hourly_f, if available
```

Output point:

```text
dyncomp_point_f
```

Sigma construction:

```python
base_sigma = max(0.85, dyncomp_weighted_recent_mae_f * 1.253)  # convert MAE to normal sigma approx
spread_sigma = max(0.0, dyncomp_corrected_model_std_f)
sigma = max(0.85, sqrt(base_sigma**2 * 0.65 + spread_sigma**2 * 0.35))
```

PMF: Gaussian integer PMF.

If fewer than 3 point sources, expert must output fallback equal to long-history expert PMF if available; otherwise climatology fallback.

### 4.6 Expert C: NBM/NBMQMD specialist

`expert_id = 'nbm_specialist'`

Included in MVP: yes if NBM data exists for at least 500 training samples. Otherwise fallback to dynamic composite.

Allowed features:

```text
grib_nbm_*, grib_nbmqmd_*, calendar, climatology, actual_history limited to rolling means/gradients, risk_regime, dynamic_error for NBM only
```

Target:

```text
residual = Y_T - dyncomp_point_f
```

If `dyncomp_point_f` missing for a row, exclude row from training.

Model:

```text
LightGBM default regressor predicting residual.
```

Prediction:

```python
point = dyncomp_point_f + predicted_residual
```

Sigma:

```python
sigma = max(0.85, std_oof_residual_by_cutoff_or_global)
```

If NBM probabilistic/quantile features exist, Codex must add them as features but v1 PMF still uses residual Gaussian unless `nbmqmd` provides explicit quantiles. If explicit quantiles exist, estimate sigma:

```python
sigma_quantile = (q90 - q10) / 2.563
sigma = max(0.85, 0.5 * sigma_residual + 0.5 * sigma_quantile)
```

### 4.7 Expert D: HRRR/RAP local-regime expert

`expert_id = 'hrrr_rap_local_regime'`

Included in MVP: yes if HRRR or RAP features exist for at least 500 training samples.

Allowed features:

```text
grib_hrrr_*, grib_rap_*, grad_grib_hrrr_*, grad_grib_rap_*, observations, risk_regime, calendar warm-season flags, actual_history gradients, dynamic_error for HRRR/RAP
```

Actual code must use ASCII `grib_` feature names.

Target:

```text
residual = Y_T - dyncomp_point_f
```

Model:

```text
LightGBM default regressor predicting residual.
```

Additional rule:

```text
If risk_sea_breeze_score >= 0.65 or risk_backdoor_front_score >= 0.60, include interaction features generated at materialization:
interaction_hrrr_tmax_x_seabreeze = grib_hrrr_klga_core_tmax_hourly_f * risk_sea_breeze_score
interaction_hrrr_wind_east_x_gradient = marine_dir_score * grad_grib_hrrr_inland_minus_marine_tmax_f
```

If fewer than 500 samples, expert stub outputs dynamic composite PMF and diagnostics `stub_reason='insufficient_hrrr_rap_history'`.

### 4.8 Expert E: global ensemble distribution expert

`expert_id = 'global_ensemble_distribution'`

Included in MVP: yes if at least one ensemble source has member-level data for at least 250 target/cutoff samples. If only ensemble means exist, use ensemble-mean fallback.

Allowed features:

```text
ens_gefsatmos_*, ens_ifsenfo_*, ens_aifsenfo_*, ens_aigefssfc_*, grib_gefsatmosmean_*, grib_gfs_*, grib_ifsoper_*, calendar, risk_regime
```

No LightGBM required for MVP. Construct PMF:

1. For each available ensemble model, build smoothed member PMF using Section 2.18.
2. Bias-correct ensemble member values by recent ensemble mean bias if available:

```python
bias_E = recenterr_<ensemble_source>_bias_h21_f if available else 0
member_corrected = member_raw - bias_E
```

3. Family weights:

```text
GEFS: 0.35
IFS ENS: 0.40
AIFS ENS: 0.15
AIGEFS: 0.10
```

4. Renormalize weights over available ensembles.
5. Output weighted PMF.
6. Point mean/median derived from PMF.

If no member-level ensembles but GEFS mean exists:

```python
mu = grib_gefsatmosmean_klga_core_tmax_hourly_f - recenterr_gefsmean_bias_h21_f
sigma = max(2.2, recenterr_gefsmean_mae_h60_f * 1.253 if available else 2.8)
```

### 4.9 Expert F: AI model expert

`expert_id = 'ai_model_expert'`

Included in MVP: stub unless AI sources have at least 250 OOF samples for the cutoff.

Allowed features:

```text
grיב_aifsoper_* ASCII grib_aifsoper_*, grib_aigfssfc_*, ens_aifsenfo_*, ens_aigefssfc_*, calendar, risk_regime, model_disagreement
```

V1 behavior:

```text
If n_training_samples >= 250: train LightGBM residual model against dyncomp_point_f.
If n_training_samples < 250: output dynamic composite PMF with temperature-inflated sigma = dynamic sigma * 1.10 and diagnostics stub_reason='insufficient_ai_history'.
```

AI expert meta weight cap is 0.10 in MVP regardless of validation.

### 4.10 Expert G: current-state observation correction expert

`expert_id = 'current_state_observation_correction'`

Included in MVP: yes.

Allowed features:

```text
observations, actual_history gradients, risk_regime, calendar, dyncomp_point_f, model_disagreement_score
```

Target:

```text
residual = Y_T - dyncomp_point_f
```

Model:

```text
Ridge regression with robust scaling.
```

Exact pipeline:

```python
Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25, 75))),
    ('ridge', Ridge(alpha=8.0, random_state=1729))
])
```

Prediction:

```python
point = dyncomp_point_f + predicted_residual
```

Sigma:

```python
sigma = max(1.25, std_oof_residual)
```

If observations stale-data regime is true, inflate sigma by 1.25.

### 4.11 Expert H: analog residual expert

`expert_id = 'analog_residual_expert'`

Included in MVP: yes.

Input analog vector features:

```text
dyncomp_point_f
mos_tmax_mean_f
grib_nbm_klga_core_tmax_hourly_f
grib_hrrr_klga_core_tmax_hourly_f
grib_rap_klga_core_tmax_hourly_f
grib_gfs_klga_core_tmax_hourly_f
ens_gefsatmos_tmax_mean_f
ens_gefsatmos_tmax_p90_minus_p10_f
risk_sea_breeze_score
risk_backdoor_front_score
risk_cloud_bust_score
risk_storm_outflow_score
obs_grad_inland_minus_klga_temp_latest_f
cal_sin_doy
cal_cos_doy
```

Distance normalization:

```python
For each feature j, scale by robust IQR computed on training data.
scaled_diff_j = (x_j - x_hist_j) / max(IQR_j, 0.5)
```

Weights by feature family:

```python
ANALOG_WEIGHTS = {
    'dyncomp_point_f': 2.0,
    'mos_tmax_mean_f': 1.0,
    'nbm_tmax': 1.5,
    'hrrr_rap_tmax': 1.5,
    'gfs_global': 1.0,
    'ensemble_spread': 1.0,
    'risk_scores': 1.5,
    'obs_gradients': 1.0,
    'seasonality': 1.0,
}
```

Distance:

```python
distance = sqrt(sum(weight_j * scaled_diff_j**2 over available j) / sum(weight_j over available j))
```

Minimum overlap: at least 8 analog vector features non-missing.

Analog count:

```text
K_ANALOG_DEFAULT = 80
K_ANALOG_MIN = 40
K_ANALOG_MAX = 150
```

Use nearest `K=80` historical rows, excluding dates within 14 calendar days after target date in any OOF fold. For live prediction, use all settled historical dates <= T-2.

Residuals:

```python
residual_hist = y_hist - dyncomp_point_hist
analog_residual_weight_i = exp(-distance_i / 0.75)
```

PMF:

```python
For each analog i, center at dyncomp_point_current + residual_hist_i with Gaussian kernel sigma=0.75°F.
Weighted average kernels by analog_residual_weight_i.
Probability floor 1e-8 and renormalize.
```

If fewer than 40 valid analogs, output dynamic composite PMF with diagnostics `stub_reason='insufficient_analogs'`.

### 4.12 Expert I: settlement-source reconciliation expert

`expert_id = 'settlement_source_reconciliation'`

Included in MVP: yes.

This expert is a PMF adjustment, not a standalone weather forecast. It learns the discrepancy:

```python
delta = wunderground_klga_high - official_or_iem_klga_high
```

Discrete delta support:

```python
DELTA_GRID = [-3, -2, -1, 0, 1, 2, 3]
```

Delta probabilities:

```python
p_delta from last 365 paired days with additive smoothing alpha=1.0
If fewer than 200 paired days, use fallback:
{-3:0.001, -2:0.004, -1:0.025, 0:0.94, 1:0.025, 2:0.004, 3:0.001}
```

If recent 30-day mean discrepancy absolute value >= 0.35, shift delta distribution by multiplying:

```python
p_delta[d] *= exp(0.75 * sign(recent_mean) * d)
renormalize
```

The expert PMF is produced by applying this delta distribution to the dynamic composite PMF:

```python
p_settle[k] = sum_delta p_base[k - delta] * p_delta[delta]
```

Where `p_base` is dynamic composite PMF.

---

## 5. Hyperparameter and validation contract

### 5.1 Training date ranges

Codex must infer available label date range from `silver.target_daily_actuals`. Then:

```python
first_label_date = min target_date with valid high_temp_f
last_label_date = max target_date with valid high_temp_f and target_date <= today - 1 day
```

Minimum required MVP history:

```text
At least 5 full calendar years of KLGA labels.
At least 3 full calendar years of MOS or equivalent station forecast features.
At least 2 full calendar years of NBM/HRRR/RAP/GFS gridded features for gridded experts.
```

If these are not met:

```text
Training command fails with exit code 40 unless --allow-insufficient-history is passed.
```

### 5.2 Final holdout policy

Define:

```python
if at least 8 years of labels:
    final_holdout_start = Jan 1 of the latest complete calendar year before current year
else:
    final_holdout_start = last_label_date - 365 days
```

No hyperparameter selection may use final holdout. Final holdout is used only by `backtest final-report` after all tuning decisions are frozen.

### 5.3 Walk-forward folds

For model selection and OOF generation:

```python
folds = []
for validation_year in years from first_label_year + 3 through final_holdout_start.year - 1:
    train_start = first_label_date
    train_end = Dec 31 of validation_year - 1
    val_start = Jan 1 of validation_year
    val_end = Dec 31 of validation_year
```

For partial current year OOF:

```text
Use train through previous full year, predict current settled dates, but do not use current year for tuning unless it is not final holdout and command is explicitly --include-current-for-tuning.
```

Default: do not include current year for tuning.

### 5.4 Nested validation method

For each expert with tunable hyperparameters:

```text
Outer folds: annual walk-forward folds.
Inner selection: use all completed outer-fold validation scores before final holdout, averaged equally by year.
```

Do not do random K-fold.

### 5.5 Scoring objectives

Primary per-component scoring:

| Component | Primary score | Secondary tie-breakers |
|---|---|---|
| Point expert | OOF MAE | OOF RMSE, calibration CRPS after PMF |
| PMF expert | OOF CRPS | integer log loss, MAE |
| Meta-combiner | OOF market-bucket log loss if market buckets exist; else integer log loss | CRPS, ECE, MAE |
| Calibration | OOF bucket log loss | ECE, Brier, monotonicity violations |
| Trading | Simulated risk-adjusted P&L | max drawdown, hit rate, calibration by traded bucket |

Tie-break rule:

```text
If primary score difference < 0.001 for log loss/CRPS or < 0.03°F for MAE, choose the simpler model/hyperparameter set with fewer effective parameters and lower turnover.
```

### 5.6 Hyperparameter grids

#### Dynamic composite

```python
BIAS_HALF_LIFE_GRID = [7, 14, 21, 30, 45]
MAE_HALF_LIFE_GRID = [30, 45, 60, 90, 120]
SKILL_WEIGHT_EXPONENT_GRID = [1.0, 1.5, 2.0, 2.5]
SKILL_WEIGHT_FLOOR_GRID = [0.75, 1.0, 1.25, 1.5]
```

Selection objective:

```text
OOF CRPS of dynamic composite PMF, tie-break lower MAE.
```

Default if fewer than 300 OOF samples:

```text
bias_half_life=21, mae_half_life=60, exponent=2.0, floor=1.0
```

#### PMF variance floor

```python
SIGMA_FLOOR_GRID = [0.85, 1.00, 1.15, 1.30]
```

Selection: OOF integer log loss. Default `0.85`.

#### Analog

```python
ANALOG_K_GRID = [40, 60, 80, 120, 150]
ANALOG_LENGTH_SCALE_GRID = [0.50, 0.75, 1.00, 1.25]
```

Selection: OOF CRPS. Default `K=80`, length scale `0.75`.

#### Model-family caps

MVP fixed caps from Section 2.20. Full production grid:

```python
MOS_CAP_GRID = [0.30, 0.35, 0.40]
NBM_CAP_GRID = [0.30, 0.35, 0.40]
HRRR_RAP_CAP_GRID = [0.25, 0.30, 0.35]
GLOBAL_CAP_GRID = [0.20, 0.25, 0.30]
AI_CAP_GRID = [0.05, 0.10, 0.15]
```

Full-production selection requires at least 1,000 OOF samples and improvement in CRPS >= 0.002.

#### Calibration method

Candidates:

```text
identity
logistic_threshold
isotonic_threshold
```

Selection rule defined in Section 7.

#### Trading hyperparameters

```python
MIN_EDGE_GRID = [0.025, 0.035, 0.050, 0.075]
KELLY_MULTIPLIER_GRID = [0.05, 0.10, 0.15, 0.25]
MODEL_RISK_BUFFER_GRID = [0.010, 0.015, 0.025, 0.035]
BOUNDARY_BUFFER_GRID = [0.000, 0.010, 0.020]
```

MVP defaults:

```python
MIN_EDGE_DEFAULT = 0.035
KELLY_MULTIPLIER_DEFAULT = 0.10
MODEL_RISK_BUFFER_DEFAULT = 0.015
BOUNDARY_BUFFER_DEFAULT = 0.010
```

Selection objective: simulated OOF net P&L with max drawdown penalty:

```python
score = mean_daily_pnl - 0.25 * max_drawdown_abs - 0.10 * pnl_std
```

Choose a trading configuration only if it has at least 100 simulated trades. Otherwise use defaults.

### 5.7 Expert/source disabling rules

Disable an expert for a cutoff if any condition holds:

```text
OOF samples for that cutoff < 250 and expert is not formula fallback.
OOF CRPS worse than dynamic composite by > 0.15 for two consecutive validation years.
PMF normalization failures > 0.
Calibration ECE > 0.12 after calibration and not improved by fallback.
Source missing rate in last 60 live/materialized target/cutoffs > 40%.
```

Disabled expert behavior:

```text
Do not delete predictions. Set source_available=false and diagnostics_json.disabled=true for live; meta-combiner weight forced to zero.
```

### 5.8 Tuning result storage

All tuning results must be stored as:

```text
reports/hyperparameter_search/<run_id>/<component>.csv
reports/hyperparameter_search/<run_id>/<component>.json
SQL: reports.metrics with metric_group='hyperparameter_search'
```

CSV columns:

```text
component, cutoff_id, params_json, fold_id, train_start, train_end, val_start, val_end, score_name, score_value, n_samples
```

### 5.9 Final-test leakage prevention

Before final report, Codex must write a frozen config:

```text
artifacts/klga_tmax/frozen_configs/frozen_config_<timestamp>_<gitsha>.json
```

This config contains selected hyperparameters and training date cutoffs. The final holdout report command must require `--frozen-config`. It must fail if any model artifact was trained using dates after `final_holdout_start - 1 day` for tuning.

---

## 6. Meta-ensemble contract

### 6.1 V1 combiner algorithm

V1 final combiner is a **static log-opinion pool** with nonnegative weights optimized on OOF expert PMFs.

For expert PMFs `P_j(k)`:

```python
P_j_floor(k) = max(P_j(k), 1e-8)
log_score(k) = intercept_k + sum_j w_j * log(P_j_floor(k))
P_final_raw(k) = softmax(log_score over k)
```

MVP uses `intercept_k = 0` for all k. Full production may add learned seasonal intercepts only after 2,000 OOF samples.

### 6.2 Optimization objective

Use `scipy.optimize.minimize` with SLSQP.

Variables:

```text
w_j for each expert
```

Constraints:

```text
0 <= w_j <= expert_cap_j
sum_j w_j = 1
```

Expert caps:

| Expert | Cap |
|---|---:|
| long_history_mos_station | 0.35 |
| dynamic_bias_corrected_composite | 0.40 |
| nbm_specialist | 0.35 |
| hrrr_rap_local_regime | 0.25 |
| global_ensemble_distribution | 0.25 |
| ai_model_expert | 0.10 |
| current_state_observation_correction | 0.15 |
| analog_residual_expert | 0.20 |
| settlement_source_reconciliation | 0.10 |

Objective:

```python
loss = mean(integer_log_loss on OOF rows) + lambda_l2 * sum((w_j - prior_w_j)**2)
lambda_l2 = 0.05
```

If parsed market buckets and historical market definitions exist for >= 250 OOF rows, objective becomes bucket log loss. Otherwise integer log loss.

### 6.3 Prior weights and initialization

Prior weights:

```python
PRIOR_WEIGHTS = {
    'long_history_mos_station': 0.18,
    'dynamic_bias_corrected_composite': 0.24,
    'nbm_specialist': 0.20,
    'hrrr_rap_local_regime': 0.11,
    'global_ensemble_distribution': 0.10,
    'ai_model_expert': 0.03,
    'current_state_observation_correction': 0.04,
    'analog_residual_expert': 0.07,
    'settlement_source_reconciliation': 0.03,
}
```

Initialize optimizer at prior weights renormalized over available experts.

### 6.4 Static vs regime-aware weights

MVP: static weights by cutoff only.

```text
Train one weight vector per cutoff_id if each cutoff has >= 500 OOF rows.
If cutoff has < 500 OOF rows, use global all-cutoff weight vector.
```

Regime-aware gating is allowed only in full production if all conditions hold:

```text
At least 2,000 OOF rows total.
At least 250 OOF rows in each regime bucket used.
Regime-aware combiner improves OOF bucket log loss by >= 0.003.
Regime-aware combiner does not worsen ECE by > 0.005.
```

Regime-aware v1.1 allowed algorithm:

```python
w_j(x) = softmax(a_j + b_j1*sea_breeze + b_j2*cloud_storm + b_j3*high_disagreement + b_j4*stale_data)
```

But MVP must not implement this as default.

### 6.5 Fallback linear pool

If log-opinion optimizer fails or any optimized PMF has invalid probabilities, use linear pool:

```python
P_final = sum_j prior_weight_j * P_j
```

Fallback weights over available experts are prior weights renormalized. Log reason:

```text
META_COMBINER_FALLBACK_LINEAR_POOL
```

Fallback condition triggers:

```text
optimizer status not success
any weight NaN
any final PMF sum outside 1 ± 1e-8
OOF rows < 250
less than 3 available experts
```

### 6.6 Missing expert PMFs

At prediction time:

```text
If expert PMF missing because source not available: exclude expert and renormalize weights over available experts.
If fewer than 3 experts remain: final PMF = dynamic composite PMF if available, else long-history PMF, else climatology fallback.
```

### 6.7 Probability floor and normalization

Every final raw PMF:

```python
p = np.maximum(p, 1e-8)
p = p / p.sum()
```

### 6.8 Required logging fields

`predictions.final_predictions.diagnostics_json` must include:

```json
{
  "available_experts": [],
  "missing_experts": [],
  "weights_used": {},
  "combiner_type": "log_opinion_pool|linear_pool_fallback",
  "optimizer_success": true,
  "optimizer_message": "...",
  "regimes": {},
  "probability_floor_applied": true,
  "pmf_sum_before_normalization": 1.0
}
```

---

## 7. Calibration contract

### 7.1 V1 calibration method

V1 calibrates threshold probabilities. Default method is `logistic_threshold`. `isotonic_threshold` may replace it only by the exact empirical rule below.

For each threshold `t` in `TEMP_GRID_F[1:]` plus `TEMP_GRID_MAX_F + 1`:

```python
raw_q_t = P_raw(Y >= t)
y_t = 1 if observed Y >= t else 0
x_t = logit(clip(raw_q_t, 1e-6, 1-1e-6))
```

Fit logistic calibrator:

```python
LogisticRegression(C=10.0, solver='lbfgs', random_state=1729)
input X = [[x_t]]
target y_t
```

Fit isotonic candidate:

```python
IsotonicRegression(out_of_bounds='clip', increasing=True)
input raw_q_t
target y_t
```

### 7.2 Global vs cutoff-specific calibration

For each cutoff:

```text
If cutoff-specific OOF samples >= 500, fit cutoff-specific calibrators.
If < 500, use global all-cutoff calibrators.
```

For threshold-specific fitting:

```text
If both classes are not present or n_samples < 250 for threshold t, use identity for that threshold.
```

### 7.3 Isotonic selection rule

For a cutoff or global calibrator family:

```text
Evaluate logistic vs isotonic on OOF folds using bucket log loss if market buckets exist for >=250 rows; otherwise integer log loss reconstructed from thresholds.
Choose isotonic only if:
  1. mean OOF score improves by at least 0.002, and
  2. ECE does not worsen by more than 0.005, and
  3. isotonic produces no more than 2 monotonicity repairs per 100 thresholds after reconstruction.
Otherwise choose logistic.
If logistic unavailable for a threshold due to one-class data, use identity for that threshold.
```

### 7.4 Integer PMF reconstruction

Define calibrated survival probabilities:

```python
q_ge[50] = 1.0
q_ge[t] = calibrated P(Y >= t) for t=51..115
q_ge[116] = 0.0
```

Monotonicity enforcement:

```python
for t from 51 to 116:
    q_ge[t] = min(q_ge[t], q_ge[t-1])
for t from 115 down to 50:
    q_ge[t] = max(q_ge[t], q_ge[t+1])
```

Then:

```python
p[k] = q_ge[k] - q_ge[k+1] for k=50..115
p = maximum(p, 1e-8)
p = p / p.sum()
```

### 7.5 Bucket-level calibration

MVP does not fit separate bucket calibrators unless historical market buckets for the same bucket grammar have at least 500 OOF examples.

If enough examples exist, bucket calibrator is logistic:

```python
x = logit(raw_bucket_probability)
y = 1 if bucket settled true else 0
```

Selection rule:

```text
Use bucket-level calibrator only if it improves OOF bucket Brier score by >=0.003 and bucket log loss by >=0.002 without worsening integer PMF CRPS by >0.002.
```

Otherwise derive bucket probabilities from calibrated integer PMF.

### 7.6 Calibration standard error and effective n

For each threshold or bucket:

```python
n_eff = number of OOF samples used by selected calibrator
se = sqrt(max(p*(1-p), 1e-6) / n_eff)
```

For isotonic, effective n must be reduced by number of bins:

```python
n_eff_isotonic = max(1, n_samples / max(number_of_isotonic_steps, 1))
```

Store in artifact JSON:

```json
{
  "calibrator_type": "logistic_threshold",
  "cutoff_id": "...",
  "thresholds": {
    "80": {"n_samples": 1234, "n_eff": 1234, "coef": ..., "intercept": ..., "se_default": ...}
  },
  "selection_metrics": {...},
  "monotonicity_repairs": 0
}
```

### 7.7 Reliability diagnostics

Every calibration report must include:

```text
ECE with 10 equal-count bins
MCE with 10 equal-count bins
Brier score by threshold group
Bucket reliability table if market buckets exist
Pre/post calibration log loss
Pre/post calibration CRPS
Number of monotonicity repairs
```

ECE formula:

```python
For bins B:
ECE = sum_B (len(B)/N) * abs(mean(pred_prob_B) - mean(outcome_B))
```

---

## 8. Polymarket bucket and trading contract

### 8.1 Market discovery rules

Use Polymarket Gamma API market/event data and CLOB public orderbook endpoints. Market metadata APIs are public; authenticated endpoints are only needed for live order management.

Accepted market if all conditions hold:

```text
1. question/title contains one of: 'NYC', 'New York City', 'LaGuardia', 'LGA', 'KLGA'.
2. question/title contains one of: 'high temp', 'highest temperature', 'temperature range', 'high temperature'.
3. resolution text or URL contains Wunderground daily history for KLGA or station KLGA.
4. target date can be parsed unambiguously.
5. enable_order_book=true for CLOB trading decisions.
6. all outcomes can be parsed into temperature buckets or market has explicit unsupported status.
```

Accepted regex patterns:

```python
TITLE_PATTERNS = [
    r'(?i)(NYC|New York City|LaGuardia|LGA|KLGA).*(high|highest).*temp',
    r'(?i)(high|highest).*temp.*(NYC|New York City|LaGuardia|LGA|KLGA)',
    r'(?i)temperature range.*(LaGuardia|LGA|KLGA|NYC)',
]
```

If more than one market matches same target date, choose market with:

```text
active=true, closed=false, enable_order_book=true, highest liquidity, latest discovered_at_utc.
```

### 8.2 Bucket parser grammar

Normalize outcome text:

```python
s = lowercase(outcome_name)
replace '°f','f'; replace 'degrees',''; replace 'deg',''; remove commas; normalize spaces
```

Supported forms:

| Form | Bucket |
|---|---|
| `85-86`, `85 – 86`, `85 to 86`, `85 through 86` | closed range `[85,86]` |
| `85` or `85f` | exact `[85,85]` |
| `85 or lower`, `85 and below`, `85 or below`, `≤85`, `85 or less` | `(-inf,85]` |
| `below 85`, `less than 85`, `<85` | `(-inf,84]` because settlement is integer |
| `85 or higher`, `85 and above`, `85 or above`, `≥85`, `85+` | `[85,inf)` |
| `above 85`, `more than 85`, `>85` | `[86,inf)` |
| `other` | unsupported unless all other buckets leave exactly one complement interval |

Parser must produce:

```text
parsed_bucket_type
lower_temp_f
upper_temp_f
lower_inclusive
upper_inclusive
parse_status
```

### 8.3 Ambiguous and non-exhaustive markets

Codex must reject trading if:

```text
Any active outcome parse_status != 'parsed'.
Parsed bucket intervals overlap.
Parsed bucket intervals do not cover all temperatures with final PMF mass >= 0.001.
Sum of bucket probabilities differs from 1 by > 0.005.
Resolution station/date cannot be matched to KLGA target date.
```

No-trade reason code:

```text
UNSUPPORTED_OR_AMBIGUOUS_BUCKETS
```

### 8.4 Bucket probability formula

For calibrated PMF `p[k]`:

```python
closed range [a,b]: sum(p[k] for k in grid if a <= k <= b)
below_or_equal b: sum(p[k] for k <= b)
below_strict b: sum(p[k] for k < b)
above_or_equal a: sum(p[k] for k >= a)
above_strict a: sum(p[k] for k > a)
exact a: p[a]
```

If bucket extends outside grid, include all grid temperatures in that direction and require missing tail mass < 1e-6 by construction.

### 8.5 Orderbook snapshot schema and VWAP

Orderbook levels are stored in `trading.orderbook_levels`.

VWAP for buying YES with notional budget `B` USDC:

```python
remaining_usdc = B
filled_shares = 0
spent = 0
for ask in asks sorted price ascending:
    level_cost = ask.price * ask.size_shares
    take_cost = min(remaining_usdc, level_cost)
    take_shares = take_cost / ask.price
    spent += take_cost
    filled_shares += take_shares
    remaining_usdc -= take_cost
    if remaining_usdc <= 1e-9: break
if spent < B:
    vwap = None  # insufficient depth
else:
    vwap = spent / filled_shares
```

Standard order sizes:

```python
STANDARD_NOTIONALS_USDC = [25, 100, 250, 500]
```

Compute and store `vwap_price_25`, `vwap_price_100`, `vwap_price_250`, `vwap_price_500` for every parsed outcome.

For selling YES, use bid side sorted descending. MVP trading decisions only buy YES; sell logic is for full production exposure reduction.

### 8.6 Fee assumptions

Codex must read `fees_enabled` and category if available. If fee metadata is missing, assume fees may apply and use weather default.

Default fee formula for taker buys:

```python
fee_usdc = shares * fee_rate * price * (1 - price)
```

Default fee rates:

```python
FEE_RATE_BY_CATEGORY = {
    'weather': 0.05,
    'crypto': 0.07,
    'sports': 0.03,
    'finance': 0.04,
    'politics': 0.04,
    'economics': 0.05,
    'culture': 0.05,
    'other': 0.05,
    'geopolitics': 0.0,
}
```

If `fees_enabled=false`, use fee 0. If category unknown and fees_enabled=true, use `0.05`.

Probability-equivalent fee buffer:

```python
fee_probability_equivalent = fee_usdc / notional_usdc
```

### 8.7 Slippage assumptions

MVP uses orderbook VWAP as slippage. Additional conservative buffer:

```python
slippage_buffer = max(0.005, 0.25 * spread)
```

For thin book regime, add:

```python
thin_book_extra_buffer = 0.015
```

### 8.8 Live vs paper default

MVP default:

```text
KLGA_TRADING_MODE=paper
Authenticated live execution is out of MVP scope.
```

Codex may implement live scaffolding, but live order placement must be disabled unless all are true:

```text
KLGA_TRADING_MODE=live
CLI flag --allow-live is present
KLGA_POLYMARKET_PRIVATE_KEY is set
KLGA_POLYMARKET_FUNDER_ADDRESS is set
risk preflight passes
```

If live flag absent, command must create paper decisions only and exit 0.

### 8.9 Live order behavior for full production

If full-production live execution is implemented:

```text
Default order type: limit, post-only if supported.
Taker order allowed only with --allow-taker and edge >= 2 * minimum_edge_threshold.
Retry count: 2.
Retry delay: 3 seconds.
Cancel unfilled order after 30 seconds unless --resting-maker is set.
Reconcile by querying open orders and fills after every placement/cancel.
Never place live orders if local system clock drift > 2 seconds from NTP.
```

MVP tests should assert live orders are not placed by default.

### 8.10 Bankroll and Kelly sizing

Bankroll source:

```text
paper/backtest: KLGA_BANKROLL_USDC env var.
live: account balance if available; otherwise min(account balance, KLGA_BANKROLL_USDC).
```

Kelly fraction for binary YES buy:

```python
p = fair_probability
q = 1 - p
price = vwap_price
b = (1 - price) / price
kelly_full = (b*p - q) / b
kelly_full = max(0, kelly_full)
kelly_fraction = KELLY_MULTIPLIER_DEFAULT * kelly_full
```

Default:

```python
KELLY_MULTIPLIER_DEFAULT = 0.10
```

Recommended notional:

```python
raw_notional = bankroll * kelly_fraction
notional = min(raw_notional, cap_per_outcome, cap_per_market, depth_cap)
```

Caps:

```python
CAP_PER_OUTCOME_FRACTION = 0.03
CAP_PER_MARKET_FRACTION = 0.08
CAP_PER_DAY_ALL_WEATHER_FRACTION = 0.12
DEPTH_CAP_FRACTION_OF_5C_ASK_DEPTH = 0.25
MIN_ORDER_NOTIONAL_USDC = 10
MAX_SINGLE_ORDER_NOTIONAL_USDC = 500
```

If `notional < 10`, no trade reason `SIZE_BELOW_MINIMUM`.

### 8.11 Liquid vs illiquid thresholds

Liquid market:

```python
market_liquidity_score >= 0.60 and median_spread <= 0.04 and total_ask_depth_5c_usdc >= 1000
```

Illiquid market:

```python
market_liquidity_score < 0.35 or median_spread >= 0.08 or total_ask_depth_5c_usdc < 250
```

Thin-book rule equals illiquid market condition.

### 8.12 No-trade conditions

Trade decision must be `no_trade` if any condition holds:

```text
NO_MARKET_MATCH
MARKET_CLOSED_OR_INACTIVE
ORDERBOOK_DISABLED
UNSUPPORTED_OR_AMBIGUOUS_BUCKETS
STALE_MARKET_SNAPSHOT age > 15 minutes
STALE_CRITICAL_WEATHER_DATA regime_stale_data=true
PMF_INVALID
BUCKET_PROB_SUM_INVALID
EDGE_BELOW_THRESHOLD
CALIBRATION_UNCERTAINTY_TOO_HIGH calibration_se_mean > 0.08
THIN_BOOK_AND_EDGE_NOT_LARGE_ENOUGH: thin_book and net_edge < 0.075
NEAR_BUCKET_BOUNDARY_AND_EDGE_NOT_LARGE_ENOUGH: near boundary and net_edge < 0.055
SIZE_BELOW_MINIMUM
RISK_CAP_EXCEEDED
LIVE_TRADING_NOT_ENABLED
```

Reason-code enum must be implemented exactly as strings above plus:

```text
OK_TO_TRADE
INSUFFICIENT_ORDERBOOK_DEPTH
FEE_METADATA_MISSING_USING_DEFAULT
EXPERTS_INSUFFICIENT_FALLBACK_USED
```

### 8.13 Edge calculation

For each outcome:

```python
fair = calibrated_bucket_probability
price = vwap_price_100 if available else best_ask
raw_edge = fair - price
uncertainty_buffer = max(
    MIN_EDGE_DEFAULT,
    MODEL_RISK_BUFFER_DEFAULT,
    calibration_se_bucket,
    0.50 * model_disagreement_score * 0.05
)
if regime_near_bucket_boundary: uncertainty_buffer += 0.010
if regime_thin_book: uncertainty_buffer += 0.015
if regime_high_entropy_forecast: uncertainty_buffer += 0.010
net_edge = raw_edge - fee_probability_equivalent - slippage_buffer - uncertainty_buffer
```

Default trade threshold:

```python
trade_allowed = net_edge >= 0.0 and raw_edge >= MIN_EDGE_DEFAULT
MIN_EDGE_DEFAULT = 0.035
```

### 8.14 Exposure-cap enforcement order

Apply caps in this order:

```text
1. Existing open exposure by target date.
2. Per-market cap.
3. Per-outcome cap.
4. Daily all-weather cap.
5. Orderbook depth cap.
6. Absolute max single-order cap.
7. Minimum order threshold.
```

### 8.15 Simulation fill model

MVP backtest fill model:

```text
instant_taker_vwap
```

For a historical market snapshot, simulate buy YES at VWAP for recommended notional using asks visible in that snapshot. If insufficient depth for recommended notional, fill only up to available depth within best ask + 0.05 and record reason `INSUFFICIENT_ORDERBOOK_DEPTH`; if filled notional < 10 USDC, no trade.

Settlement P&L:

```python
if bucket settles true:
    gross_payout = filled_shares * 1.0
else:
    gross_payout = 0
pnl = gross_payout - notional_usdc - fee_usdc
roi = pnl / notional_usdc
```

No maker fill assumption in MVP.

---

## 9. Orchestration and CLI contract

### 9.1 CLI package

Console script:

```text
klga-tmax
```

Typer app module:

```text
src/klga_tmax/cli.py
```

All commands must create an `audit.pipeline_runs` row at start and update it on success/failure.

Exit codes:

| Code | Meaning |
|---:|---|
| 0 | success |
| 10 | missing env/config |
| 20 | database/migration failure |
| 30 | data validation failure |
| 31 | target outside temp grid |
| 40 | insufficient training history |
| 50 | model training failure |
| 60 | prediction/calibration failure |
| 70 | market/trading failure |
| 80 | report generation failure |
| 90 | unhandled exception |

### 9.2 `klga-tmax db migrate`

Arguments:

```text
--revision head     default head
--dry-run           optional
```

Required env: `KLGA_DB_URL`.

Inputs: Alembic migrations.  
Outputs: database schemas/tables.  
Idempotency: running twice must leave DB unchanged and exit 0.  
Failure: exit 20.

### 9.3 `klga-tmax features materialize`

Arguments:

```text
--start-date YYYY-MM-DD required
--end-date YYYY-MM-DD required
--cutoff-id one of registry.cutoffs or 'all' default all
--feature-version supplemental_doc_1_v1 default
--replace / --no-replace default --replace
```

Inputs: silver tables.  
Outputs: `gold.target_instances`, `gold.feature_values`, `gold.feature_matrix`.  
Idempotency: with `--replace`, delete and reinsert target/cutoff/feature-version rows in one transaction. Without replace, skip existing rows.  
Failure: exit 30/31.

### 9.4 `klga-tmax train experts`

Arguments:

```text
--feature-version supplemental_doc_1_v1
--start-date YYYY-MM-DD optional
--end-date YYYY-MM-DD optional
--cutoff-id all default
--expert-id all default
--allow-insufficient-history false default
```

Inputs: `gold.feature_matrix`.  
Outputs: expert artifacts and `registry.model_versions`.  
Failure: exit 40 or 50.

This trains final refit expert artifacts on allowed training period excluding final holdout unless `--mode backtest-refit` is specified.

### 9.5 `klga-tmax predict oof`

Arguments:

```text
--feature-version supplemental_doc_1_v1
--expert-id all default
--fold-config annual_default
--replace
```

Outputs: `predictions.expert_predictions` with `prediction_mode='oof'` and PMF rows.

OOF predictions must be generated by training only on dates before each validation fold.

### 9.6 `klga-tmax train combiner`

Arguments:

```text
--feature-version supplemental_doc_1_v1
--objective auto|integer_logloss|bucket_logloss default auto
--cutoff-id all default
```

Inputs: OOF expert PMFs and labels.  
Outputs: `registry.model_versions` for meta combiner, artifact JSON/joblib.

### 9.7 `klga-tmax calibrate`

Arguments:

```text
--model-version-id UUID required or --latest-combiner
--method auto default
--cutoff-id all default
```

Inputs: OOF final raw PMFs.  
Outputs: calibration artifacts and `predictions.calibration_artifacts`.

### 9.8 `klga-tmax backtest run`

Arguments:

```text
--start-date YYYY-MM-DD
--end-date YYYY-MM-DD
--cutoff-id all default
--market-mode synthetic|historical_polymarket default historical_polymarket
--frozen-config path optional but required for final report
--replace
```

Outputs:

```text
reports.backtest_runs
reports.metrics
trading.trade_decisions with decision_mode='backtest'
trading.simulated_fills
artifacts/klga_tmax/backtests/<run_id>/
```

### 9.9 `klga-tmax live forecast`

Arguments:

```text
--target-date YYYY-MM-DD required
--cutoff-id required
--model-version-id latest default
--calibration-version-id latest default
--write-db true default
```

Inputs: latest materialized features. If features missing, command must materialize the single target/cutoff first.  
Outputs: expert live predictions, final live prediction.  
Failure: exit 60.

### 9.10 `klga-tmax market snapshot pull`

Arguments:

```text
--target-date YYYY-MM-DD required
--market-id optional
--write-db true default
```

Inputs: Polymarket APIs.  
Outputs: silver market metadata, trading market snapshot/orderbook levels.  
Failure: exit 70 if API unreachable; if no market found, write audit warning and exit 0 with no snapshot.

### 9.11 `klga-tmax trade decide`

Arguments:

```text
--target-date YYYY-MM-DD required
--cutoff-id required
--mode paper|backtest|live default from KLGA_TRADING_MODE
--allow-live false default
--bankroll-usdc optional
```

Inputs: latest final prediction and market snapshot.  
Outputs: `trading.trade_decisions`, `trading.trade_decision_legs`; simulated fills for backtest/paper.  
Failure: only DB/API errors exit 70; no-trade conditions exit 0.

### 9.12 `klga-tmax report generate`

Arguments:

```text
--report-type daily|backtest|calibration|trading|full default daily
--target-date optional
--backtest-run-id optional
--output-format markdown,json,csv default markdown
```

Outputs under `artifacts/klga_tmax/reports/...` and `reports.metrics`.

### 9.13 `klga-tmax settlement update`

Arguments:

```text
--target-date YYYY-MM-DD required
--source wunderground default
--score-predictions true default
--score-trades true default
```

Inputs: `silver.target_daily_actuals`.  
Outputs: reports metrics for settled predictions/trades.  
Failure: exit 30 if label missing.

---

## 10. Artifact and directory contract

Required source layout:

```text
weather_markets/
  pyproject.toml
  alembic.ini
  alembic/
  src/
    klga_tmax/
      __init__.py
      cli.py
      config.py
      db/
        engine.py
        models.py
        migrations_check.py
      registry/
        seed_cutoffs.py
        seed_stations.py
      features/
        materialize.py
        formulas.py
        feature_groups.py
        regimes.py
        traceability.py
      modeling/
        datasets.py
        pmf.py
        experts/
          long_history_mos_station.py
          dynamic_bias_composite.py
          nbm_specialist.py
          hrrr_rap_local.py
          global_ensemble.py
          ai_model.py
          obs_correction.py
          analog_residual.py
          settlement_reconciliation.py
        oof.py
        combiner.py
        calibration.py
        validation.py
      trading/
        market_discovery.py
        bucket_parser.py
        orderbook.py
        fees.py
        sizing.py
        decision.py
        simulation.py
        live_execution.py
      reports/
        daily.py
        backtest.py
        calibration.py
        plots.py
      utils/
        timezones.py
        hashing.py
        logging.py
        math.py
  tests/
    unit/
    integration/
    fixtures/
```

Artifact root:

```text
${KLGA_ARTIFACT_ROOT}/
  models/
    experts/<expert_id>/<model_version_id>/model.joblib
    experts/<expert_id>/<model_version_id>/metadata.json
    combiner/<model_version_id>/combiner.json
    calibrators/<model_version_id>/calibrator.joblib
    calibrators/<model_version_id>/metadata.json
  oof_predictions/<run_id>/expert_oof.parquet
  backtests/<run_id>/
    config.json
    metrics.json
    daily_scores.csv
    trade_ledger.csv
    equity_curve.csv
    report.md
    plots/
  reports/
    daily/<target_date>/<cutoff_id>/report.md
    calibration/<model_version_id>/report.md
    trading/<target_date>/<cutoff_id>/decision.md
  frozen_configs/
    frozen_config_<YYYYMMDDTHHMMSSZ>_<gitsha>.json
  logs/
```

Model artifact naming:

```text
<expert_id>__<cutoff_id_or_all>__train_<start>_<end>__git_<sha>__<model_version_id>.joblib
```

Backtest run ID:

```python
run_id = f"bt_{start_date}_{end_date}_{timestamp_utc}_{short_git_sha}"
```

Report naming:

```text
KLGA_TMAX_<report_type>_<target_date_or_range>_<cutoff_id_or_all>_<timestamp>.md
```

---

## 11. Testing and acceptance contract

Codex must implement these tests before declaring done.

### 11.1 Timezone/cutoff tests

File: `tests/unit/test_timezones_cutoffs.py`

Fixtures:

```text
target_date=2026-06-28
target_date=2026-12-15
DST transition examples: 2026-03-08, 2026-11-01
```

Assertions:

```text
Cutoffs are timezone-aware.
Stockholm and New York conversions use zoneinfo.
Target local day UTC window has 23/24/25 hours as appropriate.
PRE_LOCAL_DAY_NYC_2350 is before target local midnight.
```

### 11.2 Availability eligibility tests

File: `tests/unit/test_availability_eligibility.py`

Synthetic example:

```text
cutoff=2026-06-27T13:00Z
row A effective_available=12:59Z -> eligible
row B effective_available=13:00Z -> eligible
row C effective_available=13:00:01Z -> not eligible
```

Assertions: exact boundary behavior.

### 11.3 No-leakage tests

File: `tests/integration/test_no_leakage_feature_materialization.py`

Fixture: create a future observation after cutoff with extreme temp 999°F.

Assertion:

```text
Feature materialization never includes the extreme value.
Trace JSON max availability <= cutoff for every feature.
```

### 11.4 Feature formula tests

File: `tests/unit/test_feature_formulas.py`

Assertions:

```text
EWMA half-life weights: age=half_life -> weight=0.5.
Run trend = latest scalar - previous scalar.
Gradient = station/group A - station/group B.
Sea-breeze high-score example >=0.80.
Sea-breeze west-wind example <=0.55.
Gaussian PMF sums to 1.
```

### 11.5 Missingness handling tests

File: `tests/unit/test_missingness.py`

Assertions:

```text
Missing numeric feature is JSON null plus __missing=true.
Insufficient history produces missing_reason='insufficient_history'.
Tree model dataset transforms missing to -9999 plus missing flag.
```

### 11.6 PMF normalization tests

File: `tests/unit/test_pmf.py`

Assertions:

```text
All expert PMFs have all TEMP_GRID values.
PMF probabilities between 0 and 1.
PMF sum within 1e-8.
Probability floor applied and renormalized.
```

### 11.7 Expert output schema tests

File: `tests/integration/test_expert_predictions_schema.py`

Assertions:

```text
Each expert creates one predictions.expert_predictions row per target/cutoff.
Each expert prediction has exactly len(TEMP_GRID_F) PMF rows.
source_available and diagnostics_json present.
```

### 11.8 OOF-only meta training tests

File: `tests/integration/test_oof_meta_training.py`

Synthetic fold data:

```text
train dates before validation dates.
```

Assertions:

```text
No meta-combiner training row uses expert prediction generated by a model trained on that row's target date or later.
Final holdout rows are absent from tuning input.
```

### 11.9 Calibration monotonicity tests

File: `tests/unit/test_calibration.py`

Assertions:

```text
Calibrated survival q_ge is non-increasing with threshold.
Reconstructed PMF nonnegative and sums to 1.
Identity fallback works for one-class thresholds.
```

### 11.10 Bucket parser tests

File: `tests/unit/test_bucket_parser.py`

Cases:

```text
'85-86' -> [85,86]
'85 to 86' -> [85,86]
'85 or lower' -> (-inf,85]
'below 85' -> (-inf,84]
'85 or above' -> [85,inf)
'above 85' -> [86,inf)
'85°F' -> [85,85]
'Other' with complement -> parsed complement if unique; otherwise ambiguous
```

### 11.11 VWAP/edge/sizing math tests

File: `tests/unit/test_trading_math.py`

Assertions:

```text
VWAP consumes asks ascending.
Insufficient depth returns None or partial fill as specified.
Kelly fraction zero when fair <= price.
Fee formula matches shares * fee_rate * p * (1-p).
Risk caps applied in required order.
```

### 11.12 No-trade rules tests

File: `tests/unit/test_no_trade_rules.py`

Assertions:

```text
Unsupported buckets -> no_trade.
Stale market snapshot -> no_trade.
Thin book with small edge -> no_trade.
Calibration SE >0.08 -> no_trade.
```

### 11.13 Backtest non-leakage tests

File: `tests/integration/test_backtest_non_leakage.py`

Assertions:

```text
Backtest features and market snapshots have availability <= simulated decision time.
OOF predictions only.
No final holdout leakage when frozen config supplied.
```

### 11.14 Report generation tests

File: `tests/integration/test_reports.py`

Assertions:

```text
Daily report generated.
Backtest report includes MAE, CRPS, log loss, ECE, P&L, drawdown.
Calibration report includes reliability table.
All report file paths exist and are non-empty.
```

### 11.15 Minimum command before done

Codex must run at least:

```bash
pytest -q
klga-tmax db migrate
klga-tmax features materialize --start-date 2024-01-01 --end-date 2024-01-10 --cutoff-id all
klga-tmax predict oof --feature-version supplemental_doc_1_v1 --expert-id all
klga-tmax train combiner --feature-version supplemental_doc_1_v1
klga-tmax calibrate --latest-combiner
klga-tmax backtest run --start-date 2024-01-01 --end-date 2024-01-10 --cutoff-id all --market-mode synthetic
klga-tmax report generate --report-type backtest
```

If real historical data for 2024-01-01..2024-01-10 is absent in local dev, tests may use fixtures, but production acceptance requires real data.

---

## 12. MVP vs full production boundary

### 12.1 MVP complete definition

MVP is complete only when all of the following are true.

Included sources:

```text
Wunderground KLGA labels and nearby station actuals
IEM MOS station guidance
GribStream NBM/NBMQMD if available
GribStream HRRR and RAP
GribStream GFS and GEFS mean/member data if fetched
IEM ASOS/METAR observations
Polymarket Gamma/CLOB market snapshots
```

Included experts:

```text
long_history_mos_station
dynamic_bias_corrected_composite
nbm_specialist
hrrr_rap_local_regime
global_ensemble_distribution with fallback if only means exist
current_state_observation_correction
analog_residual_expert
settlement_source_reconciliation
ai_model_expert stub if insufficient AI history
```

Required historical coverage:

```text
At least 5 years of KLGA Wunderground labels.
At least 3 years of MOS features.
At least 2 years of core gridded features for gridded experts, or those experts must run in documented fallback mode.
```

Required cutoffs:

```text
All four cutoffs from Section 0.2.
```

Required markets:

```text
Polymarket KLGA/NYC high-temperature bucket markets resolving from Wunderground KLGA.
Synthetic bucket backtests are acceptable for model validation if historical Polymarket markets are sparse, but real market parser/orderbook tests must pass.
```

Required reports:

```text
Daily forecast report.
Backtest metrics report.
Calibration report.
Trading decision report.
```

Required metrics:

```text
MAE
RMSE
CRPS
integer log loss
bucket log loss if buckets exist
Brier by threshold
ECE
trade count
simulated P&L
max drawdown
average edge traded
```

Required live behavior:

```text
Paper trading only by default.
No live order placement in MVP unless explicitly enabled with full-production live safeguards.
```

### 12.2 Full production complete definition

Full production adds:

```text
Member-level GEFS, IFS ENS, AIFS ENS, AIGEFS.
AI model expert enabled if >=250 OOF rows and passes validation gates.
HF-ASOS low-latency observations if available.
SPC/RRFS/uvi audition sources.
Regime-aware meta-combiner if gates pass.
Live execution with order reconciliation and kill switch.
Monitoring dashboards.
Data freshness alerts.
Source pruning process.
Daily post-settlement automatic scoring.
```

Production readiness gates:

```text
1. All tests pass.
2. OOF backtest over final holdout completed with frozen config.
3. PMF calibration ECE <= 0.06 globally and <= 0.09 on traded buckets.
4. No data-leakage test failures.
5. Paper trading for at least 20 markets or 60 days, whichever comes first.
6. Simulated/live paper P&L positive after conservative fees/slippage.
7. Maximum drawdown in paper mode within configured risk tolerance.
8. Manual review of at least 10 no-trade and 10 trade reports.
```

### 12.3 Source-pruning process

Do not remove sources early. A source can be pruned from production only if:

```text
It has at least 500 OOF comparable rows.
Its inclusion worsens meta-combiner OOF bucket log loss by >=0.002 or CRPS by >=0.002.
It does not improve any predefined regime by >=0.003 log loss.
It has source missing/staleness rate >50% over last 90 live cutoffs.
```

If pruned, keep acquisition and historical tables; set production expert weight cap to zero and document in `reports/source_pruning/`.

---

## 13. Closed empirical decision rules

This section closes remaining decisions that depend on future empirical results.

### 13.1 Calibration choice

```text
Evaluate identity, logistic_threshold, isotonic_threshold on OOF folds.
Choose isotonic only if it improves OOF bucket log loss by >=0.002 and does not worsen ECE by >0.005.
Otherwise use logistic.
If logistic cannot fit because fewer than 250 samples or one class, use identity for that threshold.
```

### 13.2 Expert inclusion

```text
Include an expert in meta-combiner if it has >=250 OOF rows and valid PMFs.
Set weight cap to zero if disabled by Section 5.7.
Formula experts with fallback PMFs are always allowed but can receive optimized weight zero.
```

### 13.3 Cutoff-specific vs global models

```text
Train cutoff-specific expert/model/calibrator only if that cutoff has >=500 training rows.
Otherwise train global all-cutoff model with cutoff_id one-hot feature.
```

### 13.4 Hyperparameter selection

```text
Use the defined grids only.
Select by primary score.
If score difference below tie-break tolerance, choose simpler/default setting.
Never search outside the grid without changing this supplement.
```

### 13.5 Trading threshold selection

```text
Use MVP default trading thresholds until at least 100 OOF simulated trades exist for each candidate.
Then choose candidate maximizing mean_daily_pnl - 0.25*max_drawdown_abs - 0.10*pnl_std.
If best candidate improves score by less than 5% over default, keep default.
```

### 13.6 Live execution activation

```text
Live execution remains disabled until full production gates pass.
Paper mode is the default and required behavior.
```

---

## 14. Implementation notes tied to external API behavior

These notes are binding because they affect correctness.

1. GribStream `/timeseries` returns the shortest eligible forecast horizon for a valid time. For exact run reconstruction, use `/runs` data already acquired into `silver.grib_forecast_values` and always respect `forecasted_at` / `forecasted_time` lineage.
2. GribStream `asOf` is a model-run-time cutoff and does not prove wall-clock availability. Therefore the database contract uses `effective_available_at_utc`, not `run_time_utc`, as the feature eligibility timestamp.
3. IEM ASOS/METAR observations are near-real-time but not operational-grade. Treat them as useful current-state features with explicit staleness scoring.
4. IEM MOS is long-history station-specific guidance; treat MOS as a core expert and never discard it due to shorter modern model histories.
5. Open-Meteo Single Runs preserve individual run structure and are useful as auxiliary/fallback run-based data when acquisition includes them; do not mix stitched historical forecast series into as-of backtests unless the acquisition spec marks them as as-of-safe.
6. Polymarket public market/orderbook data is public, but authenticated endpoints are required for order management. MVP must remain paper by default.

---

## 15. Done definition

Codex may declare this supplement implemented only when:

```text
1. Database migrations create every required schema/table/view.
2. Feature materialization produces traceable features for every target/cutoff.
3. All feature formulas in Section 2 are implemented or explicitly stubbed with the defined fallback behavior.
4. Every regime flag in Section 3 is materialized.
5. Every MVP expert in Section 4 produces valid OOF and live PMFs.
6. Meta-combiner and calibration follow Sections 6 and 7 exactly.
7. Market parser, VWAP, fees, edge, sizing, no-trade rules, and fill simulator follow Section 8 exactly.
8. CLI commands in Section 9 exist and are idempotent.
9. Artifact layout follows Section 10.
10. Tests in Section 11 pass.
11. MVP boundary in Section 12.1 is satisfied.
12. No open implementation choices are left for Codex to invent.
```


---

## 16. Reference documentation consulted for API-behavior assumptions

These references do not replace the acquisition specs. They are included only to make the implementation assumptions traceable.

```text
GribStream Weather Forecast API documentation:
https://gribstream.com/docs

GribStream FAQ / backtesting and asOf guidance:
https://gribstream.com/faq

Iowa Environmental Mesonet MOS archive:
https://mesonet.agron.iastate.edu/mos/
https://mesonet.agron.iastate.edu/mos/fe.phtml

Iowa Environmental Mesonet ASOS/METAR archive:
https://mesonet.agron.iastate.edu/request/download.phtml

Open-Meteo Historical Forecast / Single Runs documentation:
https://open-meteo.com/en/docs/historical-forecast-api

Polymarket API documentation:
https://docs.polymarket.com/api-reference/introduction
https://docs.polymarket.com/market-data/overview
https://docs.polymarket.com/trading/orderbook
https://docs.polymarket.com/trading/fees
```
