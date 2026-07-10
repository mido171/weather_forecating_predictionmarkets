-- Reference only. Adapt to the repository's migration stack.
CREATE SCHEMA IF NOT EXISTS catalog;
CREATE SCHEMA IF NOT EXISTS governance;
CREATE SCHEMA IF NOT EXISTS raw_audit;
CREATE SCHEMA IF NOT EXISTS nwp_core;
CREATE SCHEMA IF NOT EXISTS feature_store;
CREATE SCHEMA IF NOT EXISTS research;
CREATE SCHEMA IF NOT EXISTS live;
CREATE SCHEMA IF NOT EXISTS quarantine;

CREATE TABLE IF NOT EXISTS catalog.weather_model (
  model_id bigserial PRIMARY KEY,
  provider text NOT NULL,
  model_code text NOT NULL,
  domain text,
  model_type text,
  native_resolution text,
  archive_start date,
  archive_end date,
  disposition text NOT NULL,
  catalog_snapshot_sha256 text NOT NULL,
  UNIQUE(provider, model_code, catalog_snapshot_sha256)
);

CREATE TABLE IF NOT EXISTS catalog.location (
  location_id bigserial PRIMARY KEY,
  location_code text NOT NULL UNIQUE,
  name text NOT NULL,
  latitude double precision NOT NULL CHECK(latitude BETWEEN -90 AND 90),
  longitude double precision NOT NULL CHECK(longitude BETWEEN -180 AND 180),
  elevation_m double precision,
  location_role text NOT NULL,
  valid_from date,
  valid_to date,
  metadata_source text NOT NULL,
  metadata_sha256 text NOT NULL
);

CREATE TABLE IF NOT EXISTS catalog.variable_selector_snapshot (
  selector_id bigserial PRIMARY KEY,
  model_id bigint NOT NULL REFERENCES catalog.weather_model(model_id),
  semantic_variable text NOT NULL,
  native_name text NOT NULL,
  native_level text NOT NULL,
  native_info text NOT NULL DEFAULT '',
  native_unit text,
  introduced_at date,
  retired_at date,
  retrieved_at_utc timestamptz NOT NULL,
  source_sha256 text NOT NULL,
  UNIQUE(model_id,native_name,native_level,native_info,retrieved_at_utc)
);

CREATE TABLE IF NOT EXISTS raw_audit.acquisition_request (
  request_id uuid PRIMARY KEY,
  model_code text NOT NULL,
  endpoint text NOT NULL,
  canonical_request_json jsonb NOT NULL,
  request_sha256 text NOT NULL UNIQUE,
  status text NOT NULL,
  attempt_count integer NOT NULL DEFAULT 0,
  started_at_utc timestamptz,
  completed_at_utc timestamptz,
  error_class text,
  error_message text
);

CREATE TABLE IF NOT EXISTS raw_audit.response_object (
  response_object_id bigserial PRIMARY KEY,
  request_id uuid NOT NULL REFERENCES raw_audit.acquisition_request(request_id),
  object_uri text NOT NULL,
  byte_size bigint NOT NULL,
  sha256 text NOT NULL,
  content_type text NOT NULL,
  retrieved_at_utc timestamptz NOT NULL,
  first_seen_at_utc timestamptz,
  row_count bigint,
  UNIQUE(request_id, sha256)
);

CREATE TABLE IF NOT EXISTS nwp_core.model_run (
  model_run_id bigserial PRIMARY KEY,
  model_id bigint NOT NULL REFERENCES catalog.weather_model(model_id),
  run_time_utc timestamptz NOT NULL,
  first_seen_at_utc timestamptz,
  documented_release_at_utc timestamptz,
  availability_grade text NOT NULL,
  availability_contract_version text,
  model_version text NOT NULL DEFAULT '',
  UNIQUE(model_id, run_time_utc, model_version)
);

CREATE TABLE IF NOT EXISTS nwp_core.point_value (
  model_run_id bigint NOT NULL REFERENCES nwp_core.model_run(model_run_id),
  valid_time_utc timestamptz NOT NULL,
  lead_minutes integer NOT NULL CHECK(lead_minutes >= 0),
  location_id bigint NOT NULL REFERENCES catalog.location(location_id),
  selector_id bigint NOT NULL REFERENCES catalog.variable_selector_snapshot(selector_id),
  member_number integer NOT NULL DEFAULT 0,
  value double precision,
  response_object_id bigint NOT NULL REFERENCES raw_audit.response_object(response_object_id),
  quality_status text NOT NULL DEFAULT 'raw_valid',
  PRIMARY KEY(model_run_id,valid_time_utc,location_id,selector_id,member_number)
) PARTITION BY RANGE (valid_time_utc);

CREATE TABLE IF NOT EXISTS governance.availability_contract (
  contract_id bigserial PRIMARY KEY,
  source_code text NOT NULL,
  contract_version text NOT NULL,
  proof_grade text NOT NULL,
  evidence_uri text NOT NULL,
  base_release_rule jsonb NOT NULL,
  conservative_buffer interval NOT NULL,
  approved boolean NOT NULL DEFAULT false,
  approved_at_utc timestamptz,
  UNIQUE(source_code,contract_version)
);

CREATE TABLE IF NOT EXISTS feature_store.target_snapshot_manifest (
  snapshot_id uuid PRIMARY KEY,
  target_date date NOT NULL,
  cutoff_utc timestamptz NOT NULL,
  cutoff_contract_version text NOT NULL,
  builder_version text NOT NULL,
  source_manifest_sha256 text NOT NULL,
  feature_manifest_sha256 text NOT NULL,
  created_at_utc timestamptz NOT NULL,
  UNIQUE(target_date,cutoff_contract_version,builder_version)
);

CREATE TABLE IF NOT EXISTS research.expert_oof_prediction (
  target_date date NOT NULL,
  frame_id text NOT NULL,
  fold_id text NOT NULL,
  expert_id text NOT NULL,
  point_forecast_c double precision NOT NULL,
  predicted_abs_error_c double precision,
  model_artifact_sha256 text NOT NULL,
  snapshot_id uuid NOT NULL REFERENCES feature_store.target_snapshot_manifest(snapshot_id),
  PRIMARY KEY(target_date,frame_id,expert_id)
);

CREATE TABLE IF NOT EXISTS live.issued_forecast (
  issued_forecast_id uuid PRIMARY KEY,
  target_date date NOT NULL,
  cutoff_utc timestamptz NOT NULL,
  issued_at_utc timestamptz NOT NULL,
  final_point_tmax_c double precision NOT NULL,
  p10_c double precision,
  p50_c double precision,
  p90_c double precision,
  system_version text NOT NULL,
  snapshot_id uuid NOT NULL REFERENCES feature_store.target_snapshot_manifest(snapshot_id),
  decision_log_json jsonb NOT NULL,
  UNIQUE(target_date,system_version)
);
