"""PostgreSQL DDL for the HKG-T24-001 foundation."""

from __future__ import annotations

from dataclasses import dataclass

from hkg_t24.constants import MODEL_SCHEMAS


@dataclass(frozen=True)
class ExpectedColumn:
    schema: str
    table: str
    column: str
    data_types: tuple[str, ...]


SCHEMA_SQL = "\n".join(f"CREATE SCHEMA IF NOT EXISTS {schema};" for schema in MODEL_SCHEMAS)

FOUNDATION_SQL = """
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS model_core.run_manifest (
  run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  run_kind text NOT NULL,
  cutoff_id text NOT NULL,
  started_at_utc timestamptz NOT NULL DEFAULT now(),
  ended_at_utc timestamptz,
  status text NOT NULL CHECK (status IN ('running','passed','failed_closed','blocked')),
  git_commit text NOT NULL,
  code_version text NOT NULL,
  config_sha256 text NOT NULL,
  db_dsn_hash text NOT NULL,
  notes text NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS model_core.source_registry (
  source_code text PRIMARY KEY,
  source_family text NOT NULL,
  source_role text NOT NULL CHECK (
    source_role IN (
      'strict_core','strict_optional','proxy_research','shadow_challenger',
      'live_shadow','support_only','blocked'
    )
  ),
  feature_prefix text NOT NULL,
  strict_allowed boolean NOT NULL DEFAULT false,
  proxy_allowed boolean NOT NULL DEFAULT false,
  shadow_allowed boolean NOT NULL DEFAULT false,
  blocked boolean NOT NULL DEFAULT false,
  live_only boolean NOT NULL DEFAULT false,
  support_only boolean NOT NULL DEFAULT false,
  unit_semantics_verified boolean NOT NULL DEFAULT false,
  availability_grade text NOT NULL CHECK (
    availability_grade IN (
      'EXACT_VINTAGE','CONSERVATIVE_SCHEDULE','MODEL_RUN_TIME_PROXY_ONLY',
      'DIAGNOSTIC_ONLY','LIVE_FIRST_SEEN_ONLY','BLOCKED'
    )
  ),
  source_time_policy text NOT NULL,
  min_target_date_hkt date,
  max_target_date_hkt date,
  required_source_scope text,
  blocker_reason text,
  promotion_gate text NOT NULL,
  notes text NOT NULL DEFAULT '',
  updated_at_utc timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT source_registry_blocked_consistency CHECK (
    (blocked AND availability_grade = 'BLOCKED' AND blocker_reason IS NOT NULL)
    OR (NOT blocked)
  )
);
ALTER TABLE model_core.source_registry ADD COLUMN IF NOT EXISTS source_code text;
ALTER TABLE model_core.source_registry ADD COLUMN IF NOT EXISTS source_family text;
ALTER TABLE model_core.source_registry ADD COLUMN IF NOT EXISTS source_role text;
ALTER TABLE model_core.source_registry ADD COLUMN IF NOT EXISTS feature_prefix text;
ALTER TABLE model_core.source_registry ADD COLUMN IF NOT EXISTS strict_allowed boolean DEFAULT false;
ALTER TABLE model_core.source_registry ADD COLUMN IF NOT EXISTS proxy_allowed boolean DEFAULT false;
ALTER TABLE model_core.source_registry ADD COLUMN IF NOT EXISTS shadow_allowed boolean DEFAULT false;
ALTER TABLE model_core.source_registry ADD COLUMN IF NOT EXISTS blocked boolean DEFAULT false;
ALTER TABLE model_core.source_registry ADD COLUMN IF NOT EXISTS live_only boolean DEFAULT false;
ALTER TABLE model_core.source_registry ADD COLUMN IF NOT EXISTS support_only boolean DEFAULT false;
ALTER TABLE model_core.source_registry ADD COLUMN IF NOT EXISTS unit_semantics_verified boolean DEFAULT false;
ALTER TABLE model_core.source_registry ADD COLUMN IF NOT EXISTS availability_grade text;
ALTER TABLE model_core.source_registry ADD COLUMN IF NOT EXISTS source_time_policy text;
ALTER TABLE model_core.source_registry ADD COLUMN IF NOT EXISTS min_target_date_hkt date;
ALTER TABLE model_core.source_registry ADD COLUMN IF NOT EXISTS max_target_date_hkt date;
ALTER TABLE model_core.source_registry ADD COLUMN IF NOT EXISTS required_source_scope text;
ALTER TABLE model_core.source_registry ADD COLUMN IF NOT EXISTS blocker_reason text;
ALTER TABLE model_core.source_registry ADD COLUMN IF NOT EXISTS promotion_gate text;
ALTER TABLE model_core.source_registry ADD COLUMN IF NOT EXISTS notes text DEFAULT '';
ALTER TABLE model_core.source_registry ADD COLUMN IF NOT EXISTS updated_at_utc timestamptz DEFAULT now();
CREATE UNIQUE INDEX IF NOT EXISTS source_registry_source_code_uidx
  ON model_core.source_registry(source_code);
CREATE UNIQUE INDEX IF NOT EXISTS source_registry_feature_prefix_uidx
  ON model_core.source_registry(feature_prefix);

CREATE TABLE IF NOT EXISTS model_core.cutoff_calendar (
  target_date_hkt date NOT NULL,
  cutoff_id text NOT NULL CHECK (cutoff_id = 'H24N'),
  formal_cutoff_utc timestamptz NOT NULL,
  operational_freeze_utc timestamptz NOT NULL,
  partition_name text NOT NULL CHECK (
    partition_name IN ('pre2024_development','sealed_2024','sealed_2025','prospective_2026')
  ),
  snapshot_id text NOT NULL,
  season text NOT NULL CHECK (season IN ('MAM','JJA','SON','DJF')),
  month integer NOT NULL CHECK (month BETWEEN 1 AND 12),
  day_of_year integer NOT NULL CHECK (day_of_year BETWEEN 1 AND 366),
  is_mam boolean NOT NULL,
  is_jja boolean NOT NULL,
  is_son boolean NOT NULL,
  is_djf boolean NOT NULL,
  calendar__year_index integer NOT NULL,
  created_at_utc timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (target_date_hkt, cutoff_id),
  UNIQUE (snapshot_id)
);

CREATE TABLE IF NOT EXISTS model_core.target_label (
  target_date_hkt date PRIMARY KEY,
  target_tmax_c numeric,
  label_visible_for_development boolean NOT NULL,
  source_table text NOT NULL,
  source_hash text NOT NULL,
  loaded_at_utc timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_features.h24n_snapshot (
  snapshot_id text PRIMARY KEY,
  target_date_hkt date NOT NULL,
  cutoff_id text NOT NULL CHECK (cutoff_id = 'H24N'),
  formal_cutoff_utc timestamptz NOT NULL,
  operational_freeze_utc timestamptz NOT NULL,
  partition_name text NOT NULL,
  official_available boolean NOT NULL DEFAULT false,
  gfs_available boolean NOT NULL DEFAULT false,
  gefs_available boolean NOT NULL DEFAULT false,
  station_proxy_available boolean NOT NULL DEFAULT false,
  ifs_shadow_available boolean NOT NULL DEFAULT false,
  ai_shadow_available boolean NOT NULL DEFAULT false,
  arwf_live_shadow_available boolean NOT NULL DEFAULT false,
  cwa_live_shadow_available boolean NOT NULL DEFAULT false,
  snapshot_status text NOT NULL DEFAULT 'active'
    CHECK (snapshot_status IN ('active','failed_closed','placeholder')),
  placeholder_reason text,
  generated_at_utc timestamptz NOT NULL DEFAULT now(),
  UNIQUE (target_date_hkt, cutoff_id)
);

CREATE TABLE IF NOT EXISTS model_features.nwp_safe_row_ledger (
  ledger_id bigserial PRIMARY KEY,
  target_date_hkt date NOT NULL,
  cutoff_id text NOT NULL CHECK (cutoff_id = 'H24N'),
  dataset_code text NOT NULL,
  run_time_utc timestamptz,
  valid_time_utc timestamptz,
  source_response_object_id bigint,
  object_uri text,
  row_is_safe_h24n boolean NOT NULL,
  exclusion_reason text,
  source_scope text NOT NULL,
  publication_buffer_hours integer NOT NULL DEFAULT 6,
  created_at_utc timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS nwp_safe_row_ledger_target_dataset_idx
  ON model_features.nwp_safe_row_ledger(target_date_hkt, dataset_code);

CREATE TABLE IF NOT EXISTS model_features.feature_matrix (
  target_date_hkt date NOT NULL,
  cutoff_id text NOT NULL CHECK (cutoff_id = 'H24N'),
  feature_scope text NOT NULL CHECK (feature_scope IN ('strict','proxy','live_shadow')),
  schema_version text NOT NULL,
  snapshot_id text NOT NULL,
  features_jsonb jsonb NOT NULL DEFAULT '{}'::jsonb,
  feature_count integer NOT NULL DEFAULT 0,
  generated_at_utc timestamptz NOT NULL DEFAULT now(),
  source_hash text NOT NULL,
  leakage_status text NOT NULL CHECK (leakage_status IN ('passed','failed_closed')),
  matrix_status text NOT NULL DEFAULT 'active' CHECK (matrix_status IN ('active','superseded','failed_closed')),
  PRIMARY KEY (target_date_hkt, cutoff_id, feature_scope, schema_version)
);

CREATE TABLE IF NOT EXISTS model_validation.leakage_audit_event (
  event_id bigserial PRIMARY KEY,
  event_level text NOT NULL CHECK (event_level IN ('INFO','WARNING','ERROR')),
  event_code text NOT NULL,
  event_message text NOT NULL,
  target_date_hkt date,
  source_code text,
  created_at_utc timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_validation.scoreboard (
  scoreboard_id text PRIMARY KEY,
  scoreboard_scope text NOT NULL,
  candidate_id text NOT NULL,
  baseline_id text,
  row_count integer NOT NULL,
  first_target_date_hkt date NOT NULL,
  last_target_date_hkt date NOT NULL,
  mae_c double precision NOT NULL,
  rmse_c double precision NOT NULL,
  bias_c double precision NOT NULL,
  median_abs_error_c double precision NOT NULL,
  p75_abs_error_c double precision NOT NULL,
  p90_abs_error_c double precision NOT NULL,
  p95_abs_error_c double precision NOT NULL,
  large_error_ge_1c_rate double precision NOT NULL,
  large_error_ge_2c_rate double precision NOT NULL,
  delta_mae_vs_baseline_c double precision,
  slice_jsonb jsonb NOT NULL DEFAULT '{}'::jsonb,
  pass_fail_status text NOT NULL CHECK (pass_fail_status IN ('pass','fail','warning','not_run')),
  run_id uuid REFERENCES model_core.run_manifest(run_id),
  created_at_utc timestamptz NOT NULL DEFAULT now(),
  updated_at_utc timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_validation.negative_control_result (
  control_id text PRIMARY KEY,
  control_name text NOT NULL,
  candidate_id text NOT NULL,
  row_count integer NOT NULL,
  mae_c double precision,
  expected_behavior text NOT NULL,
  pass_fail_status text NOT NULL CHECK (pass_fail_status IN ('pass','fail','warning','not_run')),
  details_jsonb jsonb NOT NULL DEFAULT '{}'::jsonb,
  run_id uuid REFERENCES model_core.run_manifest(run_id),
  created_at_utc timestamptz NOT NULL DEFAULT now(),
  updated_at_utc timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_audit.schema_contract_audit (
  audit_id bigserial PRIMARY KEY,
  object_name text NOT NULL,
  contract_status text NOT NULL CHECK (contract_status IN ('PASS','WARNING','FAIL')),
  details_jsonb jsonb NOT NULL DEFAULT '{}'::jsonb,
  checked_at_utc timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_live.prediction (
  prediction_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  target_date_hkt date NOT NULL,
  cutoff_id text NOT NULL CHECK (cutoff_id = 'H24N'),
  snapshot_id text NOT NULL,
  schema_version text NOT NULL,
  forecast_tmax_c numeric,
  quantile_05_tmax_c numeric,
  quantile_25_tmax_c numeric,
  quantile_50_tmax_c numeric,
  quantile_75_tmax_c numeric,
  quantile_95_tmax_c numeric,
  expected_abs_error_c numeric,
  confidence_score numeric,
  no_trade boolean NOT NULL DEFAULT false,
  produced_at_utc timestamptz NOT NULL DEFAULT now(),
  input_freeze_utc timestamptz NOT NULL,
  model_candidate_id text NOT NULL,
  run_mode text NOT NULL DEFAULT 'live' CHECK (run_mode IN ('live','prospective_replay')),
  status text NOT NULL DEFAULT 'active' CHECK (status IN ('active','superseded','failed_closed')),
  UNIQUE (target_date_hkt, cutoff_id, model_candidate_id, run_mode)
);

CREATE TABLE IF NOT EXISTS model_live.live_prediction_component (
  component_id bigserial PRIMARY KEY,
  prediction_id uuid NOT NULL REFERENCES model_live.prediction(prediction_id) ON DELETE CASCADE,
  component_kind text NOT NULL,
  component_name text NOT NULL,
  component_value numeric,
  component_weight numeric,
  component_status text NOT NULL DEFAULT 'active'
    CHECK (component_status IN ('active','placeholder','failed_closed')),
  placeholder_reason text,
  created_at_utc timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_eval.system_prediction_component (
  component_id bigserial PRIMARY KEY,
  target_date_hkt date NOT NULL,
  cutoff_id text NOT NULL CHECK (cutoff_id = 'H24N'),
  model_candidate_id text NOT NULL,
  run_mode text NOT NULL CHECK (run_mode IN ('live','prospective_replay','sealed_replay')),
  component_kind text NOT NULL,
  component_name text NOT NULL,
  component_value numeric,
  component_weight numeric,
  component_status text NOT NULL DEFAULT 'active'
    CHECK (component_status IN ('active','placeholder','failed_closed')),
  created_at_utc timestamptz NOT NULL DEFAULT now()
);
"""

SNAPSHOT_COMPAT_VIEW_SQL = """
CREATE OR REPLACE VIEW model_features.snapshot_feature_matrix_strict AS
SELECT *
FROM model_features.feature_matrix
WHERE feature_scope = 'strict';

CREATE OR REPLACE VIEW model_features.snapshot_feature_matrix_proxy AS
SELECT *
FROM model_features.feature_matrix
WHERE feature_scope = 'proxy';
"""

NWP_COMPAT_VIEW_SQL = """
CREATE OR REPLACE VIEW model_features.v_nwp_forecast_wide_compat AS
SELECT
  fw.*,
  fw.source_response_object_id AS compat_source_response_object_id
FROM nwp_tactical.forecast_wide fw;

CREATE OR REPLACE VIEW model_features.v_raw_response_object_compat AS
SELECT
  response_object_id,
  object_uri,
  row_count,
  byte_size,
  sha256 AS response_sha256,
  retrieved_at_utc AS created_at_utc
FROM nwp_tactical.raw_response_object;
"""

NWP_SAFE_VIEW_SQL = """
CREATE OR REPLACE VIEW model_features.v_nwp_h24n_safe_rows AS
SELECT
  fw.target_date_hkt::date AS target_date_hkt,
  fw.cutoff_id,
  fw.dataset_code,
  fw.run_time_utc,
  fw.valid_time_utc,
  fw.source_response_object_id,
  r.object_uri,
  (fw.run_time_utc + interval '6 hours'
     <= ((fw.target_date_hkt::date - 1) + time '15:00') AT TIME ZONE 'Asia/Hong_Kong')
   AS row_is_safe_h24n,
  CASE
    WHEN r.object_uri NOT LIKE '%full_tactical_backfill_ok_tmax%' THEN 'OUT_OF_SCOPE_OBJECT_URI'
    WHEN fw.cutoff_id <> 'H24N' THEN 'NON_H24N_CUTOFF'
    WHEN fw.dataset_code IN ('nbmoc','aigfspres','aigefssfc') THEN 'BLOCKED_SOURCE'
    WHEN fw.run_time_utc + interval '6 hours'
       > ((fw.target_date_hkt::date - 1) + time '15:00') AT TIME ZONE 'Asia/Hong_Kong'
      THEN 'AFTER_H24N_CUTOFF_WITH_BUFFER'
    ELSE NULL
  END AS exclusion_reason,
  'full_tactical_backfill_ok_tmax'::text AS source_scope
FROM nwp_tactical.forecast_wide fw
JOIN nwp_tactical.raw_response_object r
  ON r.response_object_id = fw.source_response_object_id
WHERE r.object_uri LIKE '%full_tactical_backfill_ok_tmax%'
  AND fw.cutoff_id = 'H24N'
  AND fw.dataset_code NOT IN ('nbmoc','aigfspres','aigefssfc')
  AND fw.run_time_utc + interval '6 hours'
      <= ((fw.target_date_hkt::date - 1) + time '15:00') AT TIME ZONE 'Asia/Hong_Kong';
"""

EXPECTED_COLUMNS = (
    ExpectedColumn("model_core", "source_registry", "source_code", ("text",)),
    ExpectedColumn("model_core", "source_registry", "strict_allowed", ("boolean",)),
    ExpectedColumn("model_core", "source_registry", "blocked", ("boolean",)),
    ExpectedColumn("model_core", "cutoff_calendar", "target_date_hkt", ("date",)),
    ExpectedColumn("model_features", "h24n_snapshot", "target_date_hkt", ("date",)),
    ExpectedColumn("model_features", "feature_matrix", "features_jsonb", ("jsonb",)),
    ExpectedColumn("model_features", "feature_matrix", "feature_count", ("integer",)),
    ExpectedColumn("model_validation", "scoreboard", "scoreboard_id", ("text",)),
    ExpectedColumn("model_validation", "scoreboard", "first_target_date_hkt", ("date",)),
    ExpectedColumn("model_validation", "negative_control_result", "control_id", ("text",)),
    ExpectedColumn("model_live", "prediction", "prediction_id", ("uuid",)),
    ExpectedColumn("model_live", "prediction", "target_date_hkt", ("date",)),
)
