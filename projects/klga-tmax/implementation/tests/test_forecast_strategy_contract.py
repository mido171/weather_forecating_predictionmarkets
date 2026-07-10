from __future__ import annotations

from pathlib import Path

from klga_tmax.db.migrations_check import (
    REQUIRED_COLUMNS,
    REQUIRED_INDEXES,
    REQUIRED_TABLES,
    REQUIRED_VIEWS,
)


def test_prediction_and_forecast_evaluation_tables_are_in_contract_list() -> None:
    observed = {
        f"{schema}.{table}"
        for schema, tables in REQUIRED_TABLES.items()
        for table in tables
    }
    expected = {
        "silver.station_daily_actuals",
        "silver.station_observations",
        "silver.mos_guidance",
        "predictions.expert_predictions",
        "predictions.final_predictions",
        "predictions.calibration_versions",
        "predictions.calibrated_predictions",
        "reports.forecast_evaluation_runs",
        "reports.forecast_evaluation_daily_scores",
        "reports.metrics",
    }
    assert expected.issubset(observed)


def test_forecast_report_views_are_required() -> None:
    assert REQUIRED_VIEWS["gold"] == ("v_feature_matrix_flat",)
    assert REQUIRED_VIEWS["predictions"] == ("v_final_prediction_daily",)
    assert REQUIRED_VIEWS["reports"] == ("v_forecast_accuracy_daily_scores",)


def test_forecast_schema_columns_are_contract_checked() -> None:
    assert "pmf_json" in REQUIRED_COLUMNS["predictions.expert_predictions"]
    assert "expert_weights_json" in REQUIRED_COLUMNS["predictions.final_predictions"]
    assert "settled_wu_tmax_f" in REQUIRED_COLUMNS["reports.forecast_evaluation_daily_scores"]
    assert "leakage_checked" in REQUIRED_COLUMNS["reports.forecast_evaluation_daily_scores"]


def test_forecast_eval_migration_declares_required_indexes_and_views() -> None:
    migration = Path("alembic/versions/0009_forecast_eval.py").read_text(encoding="utf-8")
    expected_indexes = {
        "ux_station_daily_actuals_current",
        "ux_station_observations_identity",
        "ux_mos_guidance_identity",
        "ux_expert_predictions_identity",
        "ux_final_predictions_identity",
        "ux_calibration_versions_identity",
        "ux_calibrated_predictions_identity",
        "ux_forecast_eval_daily_identity",
    }
    assert expected_indexes.issubset(REQUIRED_INDEXES)
    for index_name in expected_indexes:
        assert index_name in migration
    assert "CREATE OR REPLACE VIEW gold.v_feature_matrix_flat" in migration
    assert "CREATE OR REPLACE VIEW predictions.v_final_prediction_daily" in migration
    assert "CREATE OR REPLACE VIEW reports.v_forecast_accuracy_daily_scores" in migration


def test_acquisition_table_map_declares_all_normalization_sources() -> None:
    text = Path("config/acquisition_table_map.yaml").read_text(encoding="utf-8")
    for section in (
        "wunderground_daily_actuals",
        "wunderground_intraday_observations",
        "iem_mos_forecast_rows",
        "gribstream_forecast_values",
    ):
        assert f"{section}:" in text
    assert "missing mapped columns fail with exit 30" in text.lower()
