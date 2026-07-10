-- T02 full current data census registry compatibility.
-- PostgreSQL 15+. Additive and idempotent.

BEGIN;

CREATE SCHEMA IF NOT EXISTS catalog;
CREATE SCHEMA IF NOT EXISTS governance;

CREATE OR REPLACE VIEW catalog.source_registry AS
SELECT
    source_file_id AS source_registry_id,
    dataset_id,
    source_file AS source_id,
    source_file,
    repository_uri,
    original_local_path,
    file_type,
    physical_sha256,
    byte_size,
    source_row_count,
    attribute_count,
    data_min,
    data_max,
    metadata_min,
    metadata_max,
    ingestion_action AS disposition,
    target_database_layer AS db_layer,
    model_status,
    priority,
    ingestion_version,
    audit_snapshot_id,
    status,
    created_at_utc
FROM catalog.source_file_registry;

COMMENT ON VIEW catalog.source_registry IS
    'T02 compatibility registry over catalog.source_file_registry. One row per audited physical source file with its disposition.';

CREATE OR REPLACE VIEW governance.attribute_contract AS
SELECT
    attribute_contract_id,
    dataset_id,
    source_file,
    file_type,
    attribute_name AS attribute,
    attribute_name,
    source_dtype,
    semantic_class,
    row_count,
    non_null_count,
    null_count,
    null_pct,
    storage_decision,
    db_layer,
    model_role,
    operational_status,
    quality_action,
    usefulness_score,
    rationale,
    profile_min,
    profile_max,
    audit_snapshot_id,
    contract_version,
    reconciliation_status,
    physical_destination
FROM catalog.attribute_contract;

COMMENT ON VIEW governance.attribute_contract IS
    'T02 compatibility contract over catalog.attribute_contract. Preserves the audit attribute contract under the governance namespace required by the A-to-Z task.';

INSERT INTO governance.schema_version (migration_version, description)
VALUES (
    '20260624_0003_t02_census_registry_compatibility',
    'T02 source and attribute registry compatibility views for full census reconciliation'
)
ON CONFLICT (migration_version) DO NOTHING;

COMMIT;
