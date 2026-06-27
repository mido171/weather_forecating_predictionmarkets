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
