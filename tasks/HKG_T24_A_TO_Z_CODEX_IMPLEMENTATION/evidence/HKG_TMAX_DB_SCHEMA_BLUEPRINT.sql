-- HKG T+24 Tmax database governance blueprint
-- PostgreSQL 15+ reference design. Adapt names/types to the actual warehouse.
-- This file deliberately separates storage from predictor eligibility.

BEGIN;

CREATE SCHEMA IF NOT EXISTS catalog;
CREATE SCHEMA IF NOT EXISTS governance;
CREATE SCHEMA IF NOT EXISTS label_core;
CREATE SCHEMA IF NOT EXISTS raw_audit;
CREATE SCHEMA IF NOT EXISTS operational_archive_raw;
CREATE SCHEMA IF NOT EXISTS operational_archive_normalized;
CREATE SCHEMA IF NOT EXISTS operational_anchor;
CREATE SCHEMA IF NOT EXISTS diagnostic_physics;
CREATE SCHEMA IF NOT EXISTS diagnostic_station_network;
CREATE SCHEMA IF NOT EXISTS diagnostic_regime_labels;
CREATE SCHEMA IF NOT EXISTS live_exact_vintage;
CREATE SCHEMA IF NOT EXISTS live_nwp_anchor;
CREATE SCHEMA IF NOT EXISTS object_catalog;
CREATE SCHEMA IF NOT EXISTS research_artifacts;
CREATE SCHEMA IF NOT EXISTS research_metrics;
CREATE SCHEMA IF NOT EXISTS research_oof_predictions;
CREATE SCHEMA IF NOT EXISTS quarantine;

CREATE TABLE IF NOT EXISTS catalog.dataset_registry (
    dataset_id text PRIMARY KEY,
    title text NOT NULL,
    purpose text NOT NULL,
    recommended_layer text NOT NULL,
    db_inclusion_decision text NOT NULL,
    current_operational_value smallint NOT NULL CHECK (current_operational_value BETWEEN 0 AND 100),
    diagnostic_value smallint NOT NULL CHECK (diagnostic_value BETWEEN 0 AND 100),
    future_potential smallint NOT NULL CHECK (future_potential BETWEEN 0 AND 100),
    created_at_utc timestamptz NOT NULL DEFAULT now(),
    updated_at_utc timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS catalog.source_file_registry (
    source_file_id bigserial PRIMARY KEY,
    dataset_id text NOT NULL REFERENCES catalog.dataset_registry(dataset_id),
    portable_uri text NOT NULL,
    original_local_path text,
    file_type text,
    content_sha256 text,
    byte_size bigint,
    row_count bigint,
    attribute_count integer,
    source_valid_start_utc timestamptz,
    source_valid_end_utc timestamptz,
    retrieved_at_utc timestamptz,
    ingestion_version text NOT NULL,
    UNIQUE(dataset_id, portable_uri, content_sha256)
);

CREATE TABLE IF NOT EXISTS catalog.profile_snapshot (
    profile_snapshot_id bigserial PRIMARY KEY,
    profile_name text NOT NULL,
    generated_at_utc timestamptz NOT NULL,
    source_root text,
    content_sha256 text NOT NULL UNIQUE,
    object_uri text NOT NULL,
    dataset_count integer NOT NULL,
    file_count integer NOT NULL,
    row_table_count integer NOT NULL,
    row_count bigint NOT NULL,
    attribute_count integer NOT NULL,
    loaded_at_utc timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS catalog.attribute_contract (
    attribute_contract_id bigserial PRIMARY KEY,
    dataset_id text NOT NULL REFERENCES catalog.dataset_registry(dataset_id),
    source_file text NOT NULL,
    attribute_name text NOT NULL,
    source_dtype text,
    semantic_class text,
    storage_decision text NOT NULL,
    db_layer text NOT NULL,
    model_role text NOT NULL,
    operational_status text NOT NULL,
    quality_action text NOT NULL,
    usefulness_score smallint CHECK (usefulness_score BETWEEN 0 AND 100),
    rationale text NOT NULL,
    contract_version text NOT NULL,
    UNIQUE(dataset_id, source_file, attribute_name, contract_version)
);

CREATE TABLE IF NOT EXISTS governance.availability_contract (
    availability_contract_id bigserial PRIMARY KEY,
    dataset_id text NOT NULL REFERENCES catalog.dataset_registry(dataset_id),
    source_id text,
    contract_version text NOT NULL,
    decision_cutoff_rule text NOT NULL,
    valid_time_rule text,
    issue_time_rule text,
    available_at_rule text NOT NULL,
    conservative_latency interval,
    operationally_eligible boolean NOT NULL DEFAULT false,
    evidence_uri text,
    approved_by text,
    approved_at_utc timestamptz,
    UNIQUE(dataset_id, source_id, contract_version)
);

CREATE TABLE IF NOT EXISTS governance.quality_issue (
    quality_issue_id bigserial PRIMARY KEY,
    severity text NOT NULL,
    dataset_id text NOT NULL,
    source_table text,
    attributes text,
    evidence text NOT NULL,
    required_action text NOT NULL,
    status text NOT NULL DEFAULT 'OPEN',
    detected_at_utc timestamptz NOT NULL DEFAULT now(),
    resolved_at_utc timestamptz
);

CREATE TABLE IF NOT EXISTS catalog.station_dim (
    station_sk bigserial PRIMARY KEY,
    station_id text NOT NULL,
    station_name text,
    country_code text,
    icao text,
    valid_from date NOT NULL,
    valid_to date,
    latitude double precision,
    longitude double precision,
    elevation_m double precision,
    distance_to_hko_km double precision,
    bearing_from_hko_deg double precision,
    station_tier text,
    meteorological_role text,
    source_metadata_version text NOT NULL,
    UNIQUE(station_id, valid_from, source_metadata_version)
);

CREATE TABLE IF NOT EXISTS label_core.hko_daily_tmax (
    local_date date PRIMARY KEY,
    target_tmax_c numeric(5,2) NOT NULL CHECK (target_tmax_c BETWEEN 0 AND 45),
    target_station text NOT NULL,
    target_source_id text NOT NULL,
    content_sha256 text NOT NULL,
    retrieved_at_utc timestamptz NOT NULL,
    quality_status text NOT NULL DEFAULT 'VALID',
    ingestion_version text NOT NULL
);

CREATE TABLE IF NOT EXISTS operational_archive_raw.hko_forecast_item (
    forecast_item_id bigserial PRIMARY KEY,
    source_id text NOT NULL,
    source_product text,
    feed_type text,
    language text,
    guid text,
    title text,
    category text,
    link text,
    description_text text,
    issue_time_utc timestamptz,
    published_at_utc timestamptz,
    available_at_utc timestamptz NOT NULL,
    retrieved_at_utc timestamptz NOT NULL,
    content_sha256 text NOT NULL,
    parser_version text,
    quality_status text NOT NULL,
    ingestion_version text NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_hko_forecast_item_available
ON operational_archive_raw.hko_forecast_item(available_at_utc);

CREATE TABLE IF NOT EXISTS operational_archive_normalized.hko_forecast_day (
    forecast_day_id bigserial PRIMARY KEY,
    forecast_item_id bigint REFERENCES operational_archive_raw.hko_forecast_item(forecast_item_id),
    target_date date NOT NULL,
    forecast_min_c numeric(5,2),
    forecast_max_c numeric(5,2),
    weather_text text,
    wind_text text,
    temperature_text text,
    issue_time_utc timestamptz NOT NULL,
    available_at_utc timestamptz NOT NULL,
    lead_hours integer NOT NULL,
    source_era text NOT NULL,
    scoreable_row_valid boolean NOT NULL,
    parser_version text NOT NULL,
    quality_status text NOT NULL,
    CHECK (forecast_min_c IS NULL OR forecast_min_c BETWEEN 0 AND 40),
    CHECK (forecast_max_c IS NULL OR forecast_max_c BETWEEN 5 AND 45)
);

CREATE INDEX IF NOT EXISTS idx_hko_forecast_day_target_available
ON operational_archive_normalized.hko_forecast_day(target_date, available_at_utc);

CREATE TABLE IF NOT EXISTS operational_anchor.hko_t24_anchor (
    target_date date NOT NULL,
    cutoff_utc timestamptz NOT NULL,
    selected_forecast_day_id bigint NOT NULL REFERENCES operational_archive_normalized.hko_forecast_day(forecast_day_id),
    official_tmin_c numeric(5,2),
    official_tmax_c numeric(5,2) NOT NULL,
    source_era text NOT NULL,
    selection_rule_version text NOT NULL,
    created_at_utc timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY(target_date, cutoff_utc)
);

CREATE TABLE IF NOT EXISTS diagnostic_physics.hko_daily_climate_long (
    station_or_domain text NOT NULL,
    variable text NOT NULL,
    unit text NOT NULL,
    local_date date NOT NULL,
    value double precision,
    trace_flag boolean NOT NULL DEFAULT false,
    completeness text,
    source_id text NOT NULL,
    content_sha256 text NOT NULL,
    retrieved_at_utc timestamptz NOT NULL,
    operationally_eligible boolean NOT NULL DEFAULT false,
    quality_status text NOT NULL,
    ingestion_version text NOT NULL,
    PRIMARY KEY(station_or_domain, variable, local_date, source_id)
);

CREATE TABLE IF NOT EXISTS diagnostic_physics.igra_key_level (
    station_id text NOT NULL,
    valid_time_utc timestamptz NOT NULL,
    pressure_level_hpa numeric(7,2) NOT NULL,
    temperature_c numeric(6,2),
    geopotential_height_m numeric(8,2),
    dewpoint_depression_c numeric(6,2),
    relative_humidity_pct numeric(6,2),
    wind_direction_deg numeric(6,2),
    wind_speed_mps numeric(7,2),
    source_id text NOT NULL,
    quality_status text NOT NULL,
    release_latency_proven boolean NOT NULL DEFAULT false,
    ingestion_version text NOT NULL,
    PRIMARY KEY(station_id, valid_time_utc, pressure_level_hpa, source_id)
);

CREATE TABLE IF NOT EXISTS diagnostic_station_network.isd_observation (
    station_sk bigint NOT NULL REFERENCES catalog.station_dim(station_sk),
    observed_at_utc timestamptz NOT NULL,
    report_type text,
    air_temperature_c numeric(6,2),
    dew_point_c numeric(6,2),
    sea_level_pressure_hpa numeric(7,2),
    wind_direction_deg numeric(6,2),
    wind_speed_mps numeric(7,2),
    temperature_quality_code text,
    dew_point_quality_code text,
    pressure_quality_code text,
    source_id text NOT NULL,
    content_sha256 text NOT NULL,
    retrieved_at_utc timestamptz NOT NULL,
    quality_status text NOT NULL,
    exact_vintage_proven boolean NOT NULL DEFAULT false,
    ingestion_version text NOT NULL,
    PRIMARY KEY(station_sk, observed_at_utc, report_type, source_id)
);

CREATE TABLE IF NOT EXISTS live_nwp_anchor.arwf_station_forecast (
    station_code text NOT NULL,
    cycle_time_utc timestamptz NOT NULL,
    available_at_utc timestamptz NOT NULL,
    valid_date date NOT NULL,
    lead_hours integer NOT NULL,
    forecast_max_temperature_c numeric(5,2),
    forecast_min_temperature_c numeric(5,2),
    latitude double precision,
    longitude double precision,
    source_id text NOT NULL,
    content_sha256 text NOT NULL,
    quality_status text NOT NULL,
    ingestion_version text NOT NULL,
    PRIMARY KEY(station_code, cycle_time_utc, valid_date)
);

CREATE TABLE IF NOT EXISTS object_catalog.asset (
    asset_id bigserial PRIMARY KEY,
    dataset_id text NOT NULL REFERENCES catalog.dataset_registry(dataset_id),
    source_id text,
    object_uri text NOT NULL,
    content_sha256 text NOT NULL,
    byte_size bigint,
    media_type text,
    valid_time_utc timestamptz,
    issue_time_utc timestamptz,
    available_at_utc timestamptz,
    retrieved_at_utc timestamptz NOT NULL,
    extraction_status text NOT NULL DEFAULT 'RAW_ONLY',
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE(object_uri, content_sha256)
);

CREATE TABLE IF NOT EXISTS research_metrics.experiment (
    experiment_id text PRIMARY KEY,
    experiment_folder_uri text NOT NULL,
    hypothesis text NOT NULL,
    frame_id text NOT NULL,
    leakage_status text NOT NULL,
    created_at_utc timestamptz NOT NULL,
    code_commit text,
    data_snapshot_hash text NOT NULL
);

CREATE TABLE IF NOT EXISTS research_metrics.score (
    experiment_id text NOT NULL REFERENCES research_metrics.experiment(experiment_id),
    candidate_id text NOT NULL,
    slice_id text NOT NULL,
    n integer NOT NULL,
    mae double precision,
    rmse double precision,
    bias double precision,
    median_absolute_error double precision,
    p90_absolute_error double precision,
    p95_absolute_error double precision,
    baseline_id text,
    delta_mae double precision,
    PRIMARY KEY(experiment_id, candidate_id, slice_id)
);

CREATE TABLE IF NOT EXISTS research_oof_predictions.prediction (
    experiment_id text NOT NULL REFERENCES research_metrics.experiment(experiment_id),
    model_id text NOT NULL,
    fold_id text NOT NULL,
    target_date date NOT NULL,
    prediction double precision NOT NULL,
    baseline_prediction double precision,
    actual_tmax_c double precision,
    is_strict_oof boolean NOT NULL,
    frame_id text NOT NULL,
    PRIMARY KEY(experiment_id, model_id, fold_id, target_date)
);

-- The production feature role must not receive SELECT on label_core, research labels,
-- diagnostic schemas, or quarantine. Grant only curated views after governance approval.

COMMIT;
