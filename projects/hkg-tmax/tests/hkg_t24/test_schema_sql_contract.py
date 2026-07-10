from __future__ import annotations

from hkg_t24.db.ddl import (
    FOUNDATION_SQL,
    NWP_COMPAT_VIEW_SQL,
    NWP_SAFE_VIEW_SQL,
    SNAPSHOT_COMPAT_VIEW_SQL,
)


def test_feature_matrix_is_physical_and_snapshot_matrices_are_views_only() -> None:
    assert "CREATE TABLE IF NOT EXISTS model_features.feature_matrix" in FOUNDATION_SQL
    assert "CREATE TABLE IF NOT EXISTS model_features.snapshot_feature_matrix_strict" not in FOUNDATION_SQL
    assert "CREATE TABLE IF NOT EXISTS model_features.snapshot_feature_matrix_proxy" not in FOUNDATION_SQL
    assert "CREATE OR REPLACE VIEW model_features.snapshot_feature_matrix_strict" in SNAPSHOT_COMPAT_VIEW_SQL
    assert "CREATE OR REPLACE VIEW model_features.snapshot_feature_matrix_proxy" in SNAPSHOT_COMPAT_VIEW_SQL


def test_nwp_safe_view_contains_mandatory_filters() -> None:
    assert "full_tactical_backfill_ok_tmax" in NWP_SAFE_VIEW_SQL
    assert "fw.run_time_utc + interval '6 hours'" in NWP_SAFE_VIEW_SQL
    assert "fw.cutoff_id = 'H24N'" in NWP_SAFE_VIEW_SQL
    assert "fw.dataset_code NOT IN ('nbmoc','aigfspres','aigefssfc')" in NWP_SAFE_VIEW_SQL
    assert "nwp_tactical.raw_response_object" in NWP_SAFE_VIEW_SQL


def test_raw_response_object_compat_view_uses_final_column_names() -> None:
    assert "CREATE OR REPLACE VIEW model_features.v_raw_response_object_compat" in NWP_COMPAT_VIEW_SQL
    assert "sha256 AS response_sha256" in NWP_COMPAT_VIEW_SQL
    assert "retrieved_at_utc AS created_at_utc" in NWP_COMPAT_VIEW_SQL


def test_live_prediction_scaffold_uses_final_patch_fields() -> None:
    assert "CREATE TABLE IF NOT EXISTS model_live.prediction" in FOUNDATION_SQL
    assert "run_mode text NOT NULL DEFAULT 'live'" in FOUNDATION_SQL
    assert "UNIQUE (target_date_hkt, cutoff_id, model_candidate_id, run_mode)" in FOUNDATION_SQL
    assert "CREATE TABLE IF NOT EXISTS model_live.live_prediction_component" in FOUNDATION_SQL
    assert "CREATE TABLE IF NOT EXISTS model_eval.system_prediction_component" in FOUNDATION_SQL


def test_validation_scaffold_tables_exist_without_model_outputs() -> None:
    assert "CREATE TABLE IF NOT EXISTS model_validation.scoreboard" in FOUNDATION_SQL
    assert "first_target_date_hkt date NOT NULL" in FOUNDATION_SQL
    assert "CREATE TABLE IF NOT EXISTS model_validation.negative_control_result" in FOUNDATION_SQL


def test_jira002_feature_family_tables_exist() -> None:
    for table_name in (
        "model_features.official_features",
        "model_features.official_revision_features",
        "model_features.target_memory_features",
        "model_features.online_residual_state",
        "model_features.nwp_daily_features",
        "model_features.nwp_ensemble_features",
        "model_features.station_proxy_features",
        "model_features.diagnostic_proxy_features",
        "model_features.static_geospatial_features",
    ):
        assert f"CREATE TABLE IF NOT EXISTS {table_name}" in FOUNDATION_SQL
    assert "ALTER TABLE model_features.feature_matrix ADD COLUMN IF NOT EXISTS target_tmax_c" in FOUNDATION_SQL


def test_jira002_oof_tables_enforce_chronology() -> None:
    assert "CREATE TABLE IF NOT EXISTS model_oof.expert_prediction" in FOUNDATION_SQL
    assert "train_end_date < test_start_date" in FOUNDATION_SQL
    assert "CREATE TABLE IF NOT EXISTS model_oof.expert_artifact" in FOUNDATION_SQL
