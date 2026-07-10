-- T03 GribStream catalog, selector, licence, and quota audit registry.
-- PostgreSQL 15+. Additive and idempotent.

BEGIN;

CREATE SCHEMA IF NOT EXISTS catalog;
CREATE SCHEMA IF NOT EXISTS governance;

CREATE TABLE IF NOT EXISTS catalog.catalog_snapshot (
    catalog_snapshot_id text PRIMARY KEY,
    provider text NOT NULL,
    source_url text NOT NULL,
    retrieved_at_utc timestamptz NOT NULL,
    status_code integer NOT NULL,
    content_sha256 char(64) NOT NULL,
    content_bytes bigint NOT NULL,
    content_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    notes text NOT NULL DEFAULT '',
    UNIQUE (provider, source_url, content_sha256)
);

CREATE TABLE IF NOT EXISTS catalog.model_registry (
    model_code text PRIMARY KEY,
    provider text NOT NULL,
    model_name text,
    domain text NOT NULL,
    disposition text NOT NULL,
    archive_or_window text,
    archive_start date,
    model_type text,
    native_resolution text,
    update_cadence text,
    lead_time text,
    page_url text NOT NULL,
    catalog_snapshot_id text NOT NULL REFERENCES catalog.catalog_snapshot(catalog_snapshot_id),
    selector_count integer NOT NULL DEFAULT 0,
    coverage_status text NOT NULL,
    final_status text NOT NULL,
    retrieved_at_utc timestamptz NOT NULL,
    notes text NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS catalog.selector_snapshot (
    selector_snapshot_id bigserial PRIMARY KEY,
    model_code text NOT NULL REFERENCES catalog.model_registry(model_code),
    semantic_variable text NOT NULL,
    semantic_family text NOT NULL,
    semantic_priority text NOT NULL,
    requested_levels text NOT NULL,
    native_name text,
    native_level text,
    native_info text NOT NULL DEFAULT '',
    exact_selector jsonb,
    selector_status text NOT NULL CHECK (selector_status IN ('selected','blocked')),
    blocker text NOT NULL DEFAULT '',
    source_sha256 char(64) NOT NULL,
    retrieved_at_utc timestamptz NOT NULL,
    UNIQUE (model_code, semantic_family, semantic_variable, semantic_priority, requested_levels)
);

CREATE TABLE IF NOT EXISTS catalog.source_license (
    source_code text PRIMARY KEY,
    source_name text NOT NULL,
    provider text NOT NULL,
    terms_url text NOT NULL,
    terms_last_updated date,
    licence_status text NOT NULL,
    commercial_or_bulk_status text NOT NULL,
    asof_availability_status text NOT NULL,
    quota_status text NOT NULL,
    notes text NOT NULL,
    retrieved_at_utc timestamptz NOT NULL
);

CREATE TABLE IF NOT EXISTS governance.gribstream_usage_constraint (
    constraint_id text PRIMARY KEY,
    source_code text NOT NULL REFERENCES catalog.source_license(source_code),
    constraint_kind text NOT NULL,
    constraint_status text NOT NULL,
    evidence_uri text NOT NULL,
    operational_effect text NOT NULL,
    created_at_utc timestamptz NOT NULL DEFAULT now()
);

INSERT INTO governance.schema_version (migration_version, description)
VALUES (
    '20260624_0004_t03_gribstream_catalog_registry',
    'T03 GribStream catalog snapshot, model disposition, selector, licence, quota, and usage constraint registry'
)
ON CONFLICT (migration_version) DO NOTHING;

COMMIT;
