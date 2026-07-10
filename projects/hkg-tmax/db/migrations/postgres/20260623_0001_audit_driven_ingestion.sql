-- HKG T+24 Tmax audit-driven database ingestion foundation.
-- PostgreSQL 15+. This migration is additive/idempotent and creates the
-- governed storage, catalog, role, cutoff, sealing, and safe-view boundary.

BEGIN;

CREATE SCHEMA IF NOT EXISTS catalog;
CREATE SCHEMA IF NOT EXISTS governance;
CREATE SCHEMA IF NOT EXISTS ingestion;
CREATE SCHEMA IF NOT EXISTS label_core;
CREATE SCHEMA IF NOT EXISTS sealed_confirmation;
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
CREATE SCHEMA IF NOT EXISTS research_supervised;
CREATE SCHEMA IF NOT EXISTS acquisition_quality;
CREATE SCHEMA IF NOT EXISTS acquisition_provenance;
CREATE SCHEMA IF NOT EXISTS quality_monitoring;
CREATE SCHEMA IF NOT EXISTS feature_safe;
CREATE SCHEMA IF NOT EXISTS quarantine;

CREATE TABLE IF NOT EXISTS governance.schema_version (
    migration_version text PRIMARY KEY,
    applied_at_utc timestamptz NOT NULL DEFAULT now(),
    description text NOT NULL,
    checksum_sha256 text
);

CREATE OR REPLACE FUNCTION governance.hkg_t24_cutoff_utc(target_date date)
RETURNS timestamptz
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT (((target_date - 1)::timestamp + time '15:00') AT TIME ZONE 'Asia/Hong_Kong');
$$;

CREATE TABLE IF NOT EXISTS catalog.audit_snapshot (
    audit_snapshot_id text PRIMARY KEY,
    bundle_sha256 char(64) NOT NULL UNIQUE,
    bundle_bytes bigint NOT NULL,
    original_local_path text,
    repository_uri text NOT NULL UNIQUE,
    extracted_uri text NOT NULL,
    generated_at_utc timestamptz,
    extracted_at_utc timestamptz NOT NULL,
    git_commit_before text,
    git_commit_after text,
    manifest jsonb NOT NULL
);

CREATE TABLE IF NOT EXISTS catalog.dataset_registry (
    dataset_id text PRIMARY KEY,
    db_inclusion text NOT NULL,
    recommended_layer text NOT NULL,
    current_operational_value smallint NOT NULL CHECK (current_operational_value BETWEEN 0 AND 100),
    diagnostic_research_value smallint NOT NULL CHECK (diagnostic_research_value BETWEEN 0 AND 100),
    future_potential smallint NOT NULL CHECK (future_potential BETWEEN 0 AND 100),
    verdict text NOT NULL,
    audit_snapshot_id text NOT NULL REFERENCES catalog.audit_snapshot(audit_snapshot_id),
    contract_version text NOT NULL,
    loaded_at_utc timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS catalog.source_file_registry (
    source_file_id bigserial PRIMARY KEY,
    dataset_id text NOT NULL REFERENCES catalog.dataset_registry(dataset_id),
    source_file text NOT NULL,
    repository_uri text NOT NULL,
    original_local_path text,
    file_type text NOT NULL,
    physical_sha256 char(64),
    byte_size bigint NOT NULL,
    source_row_count bigint NOT NULL,
    attribute_count integer NOT NULL,
    data_min timestamptz,
    data_max timestamptz,
    metadata_min timestamptz,
    metadata_max timestamptz,
    ingestion_action text NOT NULL,
    target_database_layer text NOT NULL,
    model_status text NOT NULL,
    priority text NOT NULL,
    ingestion_version text NOT NULL,
    audit_snapshot_id text NOT NULL REFERENCES catalog.audit_snapshot(audit_snapshot_id),
    status text NOT NULL DEFAULT 'REGISTERED',
    created_at_utc timestamptz NOT NULL DEFAULT now(),
    UNIQUE (source_file, physical_sha256, ingestion_version)
);

CREATE TABLE IF NOT EXISTS object_catalog.asset (
    asset_id bigserial PRIMARY KEY,
    asset_uri text NOT NULL,
    original_local_path text,
    content_sha256 char(64),
    byte_size bigint NOT NULL,
    media_type text NOT NULL,
    dataset_id text,
    source_file_id bigint REFERENCES catalog.source_file_registry(source_file_id),
    asset_role text NOT NULL,
    extraction_status text NOT NULL DEFAULT 'REGISTERED',
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    registered_at_utc timestamptz NOT NULL DEFAULT now(),
    UNIQUE (asset_uri, content_sha256)
);

CREATE TABLE IF NOT EXISTS catalog.profile_snapshot (
    profile_snapshot_id bigserial PRIMARY KEY,
    profile_name text NOT NULL,
    generated_at_utc timestamptz,
    source_root text,
    content_sha256 char(64) NOT NULL,
    object_uri text NOT NULL,
    dataset_count integer NOT NULL,
    file_count integer NOT NULL,
    row_table_count integer NOT NULL,
    row_count bigint NOT NULL,
    attribute_count integer NOT NULL,
    audit_snapshot_id text REFERENCES catalog.audit_snapshot(audit_snapshot_id),
    loaded_at_utc timestamptz NOT NULL DEFAULT now(),
    UNIQUE (object_uri, content_sha256)
);

CREATE TABLE IF NOT EXISTS governance.table_load_contract (
    dataset_id text NOT NULL,
    source_file text NOT NULL,
    file_type text NOT NULL,
    row_count bigint NOT NULL,
    byte_size bigint NOT NULL,
    attribute_count integer NOT NULL,
    data_min timestamptz,
    data_max timestamptz,
    db_action text NOT NULL,
    db_layer text NOT NULL,
    model_status text NOT NULL,
    priority text NOT NULL,
    notes text NOT NULL,
    audit_snapshot_id text NOT NULL REFERENCES catalog.audit_snapshot(audit_snapshot_id),
    contract_version text NOT NULL,
    PRIMARY KEY (source_file, contract_version)
);

CREATE TABLE IF NOT EXISTS catalog.attribute_contract (
    attribute_contract_id bigserial PRIMARY KEY,
    dataset_id text NOT NULL REFERENCES catalog.dataset_registry(dataset_id),
    source_file text NOT NULL,
    file_type text NOT NULL,
    attribute_name text NOT NULL,
    source_dtype text,
    semantic_class text,
    row_count bigint NOT NULL,
    non_null_count bigint NOT NULL,
    null_count bigint NOT NULL,
    null_pct double precision NOT NULL,
    storage_decision text NOT NULL,
    db_layer text NOT NULL,
    model_role text NOT NULL,
    operational_status text NOT NULL,
    quality_action text NOT NULL,
    usefulness_score smallint CHECK (usefulness_score BETWEEN 0 AND 100),
    rationale text NOT NULL,
    profile_min text,
    profile_max text,
    audit_snapshot_id text NOT NULL REFERENCES catalog.audit_snapshot(audit_snapshot_id),
    contract_version text NOT NULL,
    reconciliation_status text NOT NULL DEFAULT 'PENDING',
    physical_destination text,
    UNIQUE (dataset_id, source_file, attribute_name, contract_version)
);

CREATE TABLE IF NOT EXISTS governance.quality_issue (
    quality_issue_id text PRIMARY KEY,
    severity text NOT NULL,
    dataset_id text REFERENCES catalog.dataset_registry(dataset_id),
    source_table text,
    attributes text,
    evidence text NOT NULL,
    required_action text NOT NULL,
    current_status text NOT NULL DEFAULT 'OPEN'
        CHECK (current_status IN ('OPEN','MITIGATED','RESOLVED','ACCEPTED_DIAGNOSTIC_ONLY')),
    remediation_implementation_path text,
    validation_evidence_uri text,
    resolution_timestamp timestamptz,
    resolution_commit text,
    notes text
);
ALTER TABLE governance.quality_issue ALTER COLUMN dataset_id DROP NOT NULL;

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
    tier text,
    meteorological_role text,
    research_note text,
    dossier_version text NOT NULL,
    audit_snapshot_id text NOT NULL REFERENCES catalog.audit_snapshot(audit_snapshot_id),
    UNIQUE (station_id, valid_from, dossier_version)
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
    UNIQUE (dataset_id, source_id, contract_version)
);

CREATE TABLE IF NOT EXISTS governance.feature_eligibility (
    feature_eligibility_id bigserial PRIMARY KEY,
    dataset_id text NOT NULL,
    source_file text,
    attribute_name text,
    eligibility_status text NOT NULL,
    live_inference_allowed boolean NOT NULL DEFAULT false,
    reason text NOT NULL,
    contract_version text NOT NULL,
    UNIQUE (dataset_id, source_file, attribute_name, contract_version)
);

CREATE TABLE IF NOT EXISTS governance.leakage_test_result (
    leakage_test_result_id bigserial PRIMARY KEY,
    test_name text NOT NULL,
    status text NOT NULL CHECK (status IN ('PASS','FAIL','BLOCKED')),
    evidence_uri text,
    details jsonb NOT NULL DEFAULT '{}'::jsonb,
    executed_at_utc timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS governance.parser_version (
    parser_version_id text PRIMARY KEY,
    parser_name text NOT NULL,
    code_uri text NOT NULL,
    code_sha256 char(64),
    created_at_utc timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS governance.quarantine_reason (
    reason_code text PRIMARY KEY,
    severity text NOT NULL,
    description text NOT NULL
);

CREATE TABLE IF NOT EXISTS ingestion.batch (
    batch_id text PRIMARY KEY,
    started_at_utc timestamptz NOT NULL,
    finished_at_utc timestamptz,
    status text NOT NULL CHECK (status IN ('STARTED','SUCCEEDED','FAILED','BLOCKED','DRY_RUN')),
    code_commit text,
    audit_snapshot_hash char(64) NOT NULL,
    dataset_root_uri text NOT NULL,
    cutoff_rule_version text NOT NULL,
    database_target_redacted text NOT NULL,
    loader_version text NOT NULL,
    command_line text NOT NULL,
    host_metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    files_succeeded integer NOT NULL DEFAULT 0,
    files_failed integer NOT NULL DEFAULT 0,
    files_skipped integer NOT NULL DEFAULT 0,
    files_resumed integer NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS ingestion.file_result (
    file_result_id bigserial PRIMARY KEY,
    batch_id text NOT NULL REFERENCES ingestion.batch(batch_id),
    source_file_id bigint REFERENCES catalog.source_file_registry(source_file_id),
    source_file text NOT NULL,
    expected_hash char(64),
    observed_hash char(64),
    expected_row_count bigint,
    observed_row_count bigint,
    expected_schema jsonb,
    observed_schema jsonb,
    load_action text NOT NULL,
    started_at_utc timestamptz NOT NULL,
    finished_at_utc timestamptz,
    rows_staged bigint NOT NULL DEFAULT 0,
    rows_inserted bigint NOT NULL DEFAULT 0,
    rows_updated_versioned bigint NOT NULL DEFAULT 0,
    rows_quarantined bigint NOT NULL DEFAULT 0,
    rows_skipped_as_duplicate bigint NOT NULL DEFAULT 0,
    status text NOT NULL,
    error_text text,
    reconciliation_artifact_uri text,
    UNIQUE (batch_id, source_file)
);

CREATE TABLE IF NOT EXISTS ingestion.row_rejection (
    rejection_id bigserial PRIMARY KEY,
    batch_id text NOT NULL REFERENCES ingestion.batch(batch_id),
    source_file_id bigint REFERENCES catalog.source_file_registry(source_file_id),
    source_row_number bigint,
    dataset_id text NOT NULL,
    target_table text,
    reason_code text NOT NULL REFERENCES governance.quarantine_reason(reason_code),
    reason_detail text NOT NULL,
    raw_row_payload jsonb NOT NULL,
    raw_content_hash char(64),
    detected_at_utc timestamptz NOT NULL DEFAULT now(),
    repair_status text NOT NULL DEFAULT 'OPEN',
    repair_lineage text
);

CREATE TABLE IF NOT EXISTS ingestion.reconciliation (
    reconciliation_id bigserial PRIMARY KEY,
    batch_id text NOT NULL REFERENCES ingestion.batch(batch_id),
    reconciliation_scope text NOT NULL CHECK (reconciliation_scope IN ('SOURCE_FILE','ATTRIBUTE')),
    dataset_id text,
    source_file text,
    attribute_name text NOT NULL DEFAULT '',
    expected_disposition text NOT NULL,
    actual_disposition text NOT NULL,
    physical_destination text,
    count_hash_evidence jsonb NOT NULL DEFAULT '{}'::jsonb,
    status text NOT NULL CHECK (status IN ('PASS','FAIL','BLOCKED','SKIPPED')),
    exception_explanation text,
    UNIQUE (batch_id, reconciliation_scope, source_file, attribute_name)
);
DELETE FROM ingestion.reconciliation keep
USING ingestion.reconciliation drop_candidate
WHERE keep.reconciliation_id > drop_candidate.reconciliation_id
  AND keep.batch_id = drop_candidate.batch_id
  AND keep.reconciliation_scope = drop_candidate.reconciliation_scope
  AND COALESCE(keep.source_file, '') = COALESCE(drop_candidate.source_file, '')
  AND COALESCE(keep.attribute_name, '') = COALESCE(drop_candidate.attribute_name, '');
UPDATE ingestion.reconciliation SET attribute_name = '' WHERE attribute_name IS NULL;
ALTER TABLE ingestion.reconciliation ALTER COLUMN attribute_name SET DEFAULT '';
ALTER TABLE ingestion.reconciliation ALTER COLUMN attribute_name SET NOT NULL;

CREATE TABLE IF NOT EXISTS label_core.hko_daily_tmax (
    local_date date PRIMARY KEY,
    target_tmax_c numeric(5,2) NOT NULL CHECK (target_tmax_c BETWEEN -20 AND 60),
    target_station text NOT NULL,
    target_source_id text NOT NULL,
    content_sha256 char(64) NOT NULL,
    retrieved_at_utc timestamptz,
    quality_status text NOT NULL DEFAULT 'VALID',
    source_file_id bigint REFERENCES catalog.source_file_registry(source_file_id),
    ingestion_batch_id text REFERENCES ingestion.batch(batch_id),
    CHECK (local_date < date '2024-01-01')
);

CREATE TABLE IF NOT EXISTS sealed_confirmation.hko_daily_tmax (
    local_date date PRIMARY KEY,
    target_tmax_c numeric(5,2) NOT NULL CHECK (target_tmax_c BETWEEN -20 AND 60),
    target_station text NOT NULL,
    target_source_id text NOT NULL,
    content_sha256 char(64) NOT NULL,
    retrieved_at_utc timestamptz,
    quality_status text NOT NULL DEFAULT 'SEALED_CONFIRMATION',
    source_file_id bigint REFERENCES catalog.source_file_registry(source_file_id),
    ingestion_batch_id text REFERENCES ingestion.batch(batch_id),
    CHECK (local_date >= date '2024-01-01')
);

CREATE TABLE IF NOT EXISTS operational_anchor.hko_t24_official_anchor_rows (
    anchor_row_id bigserial PRIMARY KEY,
    target_date date NOT NULL,
    cutoff_utc timestamptz NOT NULL,
    forecast_min_c double precision,
    forecast_max_c double precision,
    forecast_range_c double precision,
    source_era text NOT NULL,
    source_product text,
    issue_time_utc timestamptz,
    published_at_utc timestamptz,
    available_at_utc timestamptz NOT NULL,
    selected_source_row_id text NOT NULL,
    selection_rule_version text NOT NULL,
    quality_status text NOT NULL,
    eligibility_status text NOT NULL,
    source_file_id bigint REFERENCES catalog.source_file_registry(source_file_id),
    ingestion_batch_id text REFERENCES ingestion.batch(batch_id),
    UNIQUE (target_date, selection_rule_version)
);

CREATE INDEX IF NOT EXISTS hko_t24_anchor_target_cutoff_idx
ON operational_anchor.hko_t24_official_anchor_rows (target_date, cutoff_utc, available_at_utc);

CREATE TABLE IF NOT EXISTS live_exact_vintage.catalog (
    live_exact_vintage_id bigserial PRIMARY KEY,
    dataset_id text NOT NULL,
    source_file_id bigint REFERENCES catalog.source_file_registry(source_file_id),
    source_id text,
    valid_time_utc timestamptz,
    issue_time_utc timestamptz,
    available_at_utc timestamptz,
    retrieved_at_utc timestamptz,
    eligibility_status text NOT NULL DEFAULT 'NOT_APPROVED',
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE OR REPLACE VIEW feature_safe.hko_t24_official_anchor AS
SELECT
    target_date,
    cutoff_utc,
    forecast_min_c AS official_tmin_c,
    forecast_max_c AS official_tmax_c,
    forecast_range_c,
    source_era,
    source_product,
    issue_time_utc,
    published_at_utc,
    available_at_utc,
    selected_source_row_id,
    selection_rule_version,
    quality_status,
    eligibility_status
FROM operational_anchor.hko_t24_official_anchor_rows
WHERE available_at_utc <= cutoff_utc
  AND eligibility_status = 'ELIGIBLE';

CREATE OR REPLACE VIEW feature_safe.hko_target_history_pre2024 AS
SELECT local_date, target_tmax_c, target_station, target_source_id, quality_status
FROM label_core.hko_daily_tmax
WHERE local_date < date '2024-01-01';

CREATE OR REPLACE VIEW feature_safe.live_exact_vintage_catalog AS
SELECT live_exact_vintage_id, dataset_id, source_id, valid_time_utc, issue_time_utc,
       available_at_utc, retrieved_at_utc, eligibility_status, metadata
FROM live_exact_vintage.catalog
WHERE available_at_utc IS NOT NULL
  AND eligibility_status = 'ELIGIBLE';

INSERT INTO governance.quarantine_reason (reason_code, severity, description)
VALUES
    ('IMPOSSIBLE_TEMPERATURE', 'CRITICAL', 'Temperature outside physically plausible or contract-approved range.'),
    ('INVALID_LEAD_OR_TARGET_DATE', 'CRITICAL', 'Forecast lead or target date failed plausibility checks.'),
    ('IGRA_SENTINEL_OR_SCALE_BLOCKED', 'CRITICAL', 'IGRA sentinel/scaling contamination requires repair before clean use.'),
    ('ISD_WIND_DIRECTION_BLOCKED', 'CRITICAL', 'ISD wind direction parser defect makes direction fields unusable.'),
    ('MALFORMED_OBJECT_TIMESTAMP', 'HIGH', 'Object timestamp could not be parsed without inventing availability.'),
    ('UNPROVEN_AVAILABILITY', 'HIGH', 'Historical availability cannot be proven before the cutoff.')
ON CONFLICT (reason_code) DO NOTHING;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hkg_tmax_ingest') THEN
        CREATE ROLE hkg_tmax_ingest;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hkg_tmax_audit') THEN
        CREATE ROLE hkg_tmax_audit;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hkg_tmax_research_dev') THEN
        CREATE ROLE hkg_tmax_research_dev;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hkg_tmax_training') THEN
        CREATE ROLE hkg_tmax_training;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hkg_tmax_live_inference') THEN
        CREATE ROLE hkg_tmax_live_inference;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hkg_tmax_confirmation_admin') THEN
        CREATE ROLE hkg_tmax_confirmation_admin;
    END IF;
END $$;

REVOKE ALL ON SCHEMA sealed_confirmation FROM PUBLIC;
REVOKE ALL ON SCHEMA label_core, raw_audit, diagnostic_physics, diagnostic_station_network,
    diagnostic_regime_labels, research_artifacts, research_metrics, research_oof_predictions,
    research_supervised, quarantine FROM hkg_tmax_live_inference;

GRANT USAGE ON SCHEMA catalog, governance, ingestion, label_core, sealed_confirmation, raw_audit,
    operational_archive_raw, operational_archive_normalized, operational_anchor, diagnostic_physics,
    diagnostic_station_network, diagnostic_regime_labels, live_exact_vintage, live_nwp_anchor,
    object_catalog, research_artifacts, research_metrics, research_oof_predictions, research_supervised,
    acquisition_quality, acquisition_provenance, quality_monitoring, feature_safe, quarantine
TO hkg_tmax_audit;

GRANT SELECT ON ALL TABLES IN SCHEMA catalog, governance, ingestion, label_core, sealed_confirmation,
    raw_audit, operational_archive_raw, operational_archive_normalized, operational_anchor,
    diagnostic_physics, diagnostic_station_network, diagnostic_regime_labels, live_exact_vintage,
    live_nwp_anchor, object_catalog, research_artifacts, research_metrics, research_oof_predictions,
    research_supervised, acquisition_quality, acquisition_provenance, quality_monitoring, feature_safe,
    quarantine
TO hkg_tmax_audit;

GRANT USAGE ON SCHEMA feature_safe, operational_anchor, live_exact_vintage TO hkg_tmax_live_inference;
GRANT SELECT ON feature_safe.hko_t24_official_anchor, feature_safe.live_exact_vintage_catalog
TO hkg_tmax_live_inference;

GRANT USAGE ON SCHEMA feature_safe, label_core, diagnostic_physics, diagnostic_station_network,
    diagnostic_regime_labels, research_metrics, research_oof_predictions, research_supervised
TO hkg_tmax_research_dev, hkg_tmax_training;
GRANT SELECT ON feature_safe.hko_target_history_pre2024 TO hkg_tmax_research_dev, hkg_tmax_training;
GRANT USAGE ON SCHEMA sealed_confirmation TO hkg_tmax_confirmation_admin;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA sealed_confirmation TO hkg_tmax_confirmation_admin;

INSERT INTO governance.schema_version (migration_version, description)
VALUES ('20260623_0001_audit_driven_ingestion', 'Audit-driven governed PostgreSQL ingestion foundation')
ON CONFLICT (migration_version) DO NOTHING;

COMMIT;
