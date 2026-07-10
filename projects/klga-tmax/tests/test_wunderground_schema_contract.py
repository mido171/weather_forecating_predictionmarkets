from __future__ import annotations

from pathlib import Path

from klga_tmax.db.migrations_check import (
    FORBIDDEN_WU_TABLES,
    REQUIRED_COLUMNS,
    REQUIRED_INDEXES,
    REQUIRED_TABLES,
)


def test_wunderground_truth_table_is_in_contract_list() -> None:
    observed = {
        f"{schema}.{table}"
        for schema, tables in REQUIRED_TABLES.items()
        for table in tables
    }
    assert "public.wunderground_daily_tmax" in observed
    assert "silver.wu_daily_actuals" not in observed
    assert "audit.wu_fetch_windows" not in observed


def test_wunderground_truth_migration_declares_table_and_legacy_drop() -> None:
    migration = Path("alembic/versions/0010_wunderground_truth_table.py").read_text(
        encoding="utf-8"
    )
    assert "CREATE TABLE IF NOT EXISTS public.wunderground_daily_tmax" in migration
    assert "PRIMARY KEY (station_id, local_date)" in migration
    assert "daily_high_source text NOT NULL DEFAULT 'hourly_temp_max'" in migration
    assert "provider_max_temp_values_json jsonb" in migration
    assert "CREATE SCHEMA IF NOT EXISTS legacy_drop_pending" in migration
    assert "RENAME TO wu_daily_actuals_legacy_0010" in migration
    assert "RENAME TO wu_fetch_windows_legacy_0010" in migration


def test_contract_requires_truth_columns_and_indexes() -> None:
    assert "public.wunderground_daily_tmax" in REQUIRED_COLUMNS
    required = REQUIRED_COLUMNS["public.wunderground_daily_tmax"]
    assert "tmax_f" in required
    assert "hourly_observations_json" in required
    assert "provider_max_temp_values_json" in required
    assert "daily_high_source" in required
    assert "validation_status" in required
    assert "wunderground_daily_tmax_pkey" in REQUIRED_INDEXES
    assert "ix_wunderground_daily_tmax_station_date" in REQUIRED_INDEXES
    assert "silver" in FORBIDDEN_WU_TABLES
    assert "wu_daily_actuals" in FORBIDDEN_WU_TABLES["silver"]


def test_target_materialization_uses_truth_table_availability() -> None:
    source = Path("src/klga_tmax/registry/materialize_targets.py").read_text(encoding="utf-8")
    assert "public.wunderground_daily_tmax" in source
    assert "actual.tmax_f IS NOT NULL" in source
    assert "actual.settlement_available_at_utc <= :cutoff_utc" in source
