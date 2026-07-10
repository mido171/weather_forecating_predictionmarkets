from pathlib import Path

from klga_tmax.db.migrations_check import REQUIRED_COLUMNS, REQUIRED_INDEXES, REQUIRED_TABLES


def test_iem_mos_migration_declares_backfill_tables_and_indexes() -> None:
    migration = Path("alembic/versions/0008_iem_mos_backfill.py").read_text(
        encoding="utf-8"
    )

    assert "CREATE TABLE IF NOT EXISTS audit.iem_mos_backfill_jobs" in migration
    assert "CREATE TABLE IF NOT EXISTS audit.iem_mos_backfill_chunks" in migration
    assert "CREATE TABLE IF NOT EXISTS silver.iem_mos_forecast_rows" in migration
    assert "CREATE TABLE IF NOT EXISTS gold.iem_mos_daily_features" in migration
    assert "CREATE TABLE IF NOT EXISTS gold.iem_mos_feature_matrix_v1" in migration
    assert "provider_available_at_utc timestamptz NOT NULL" in migration
    assert "effective_available_at_utc timestamptz NOT NULL" in migration
    assert "CONSTRAINT uq_iem_mos_forecast_raw_hash UNIQUE" in migration


def test_iem_mos_tables_are_in_contract_inspection() -> None:
    assert "iem_mos_backfill_jobs" in REQUIRED_TABLES["audit"]
    assert "iem_mos_backfill_chunks" in REQUIRED_TABLES["audit"]
    assert "iem_mos_forecast_rows" in REQUIRED_TABLES["silver"]
    assert "iem_mos_daily_features" in REQUIRED_TABLES["gold"]
    assert "iem_mos_feature_matrix_v1" in REQUIRED_TABLES["gold"]

    assert "raw_row_hash" in REQUIRED_COLUMNS["silver.iem_mos_forecast_rows"]
    assert "source_trace_json" in REQUIRED_COLUMNS["gold.iem_mos_daily_features"]
    assert "uq_iem_mos_forecast_raw_hash" in REQUIRED_INDEXES
