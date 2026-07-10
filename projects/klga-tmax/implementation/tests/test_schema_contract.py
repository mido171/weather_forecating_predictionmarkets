from __future__ import annotations

from pathlib import Path

from klga_tmax.db.migrations_check import REQUIRED_INDEXES, REQUIRED_TABLES


def test_required_task00_tables_are_in_contract_list() -> None:
    expected = {
        "registry.stations",
        "registry.station_registry",
        "registry.cutoffs",
        "registry.feature_versions",
        "audit.pipeline_runs",
        "audit.ingestion_manifests",
        "audit.data_quality_failures",
        "bronze.source_requests",
        "bronze.source_records",
        "silver.normalized_facts",
        "silver.availability_ledger",
        "gold.target_instances",
        "gold.feature_values",
        "gold.feature_matrix",
    }
    observed = {
        f"{schema}.{table}"
        for schema, tables in REQUIRED_TABLES.items()
        for table in tables
    }
    assert expected.issubset(observed)


def test_availability_ledger_expression_unique_index_is_declared() -> None:
    assert "ux_availability_ledger_identity" in REQUIRED_INDEXES
    migration = Path("alembic/versions/0001_klga_tmax_core_schema.py").read_text(encoding="utf-8")
    assert "CREATE UNIQUE INDEX IF NOT EXISTS ux_availability_ledger_identity" in migration
    assert "COALESCE(member, '')" in migration
    assert "COALESCE(run_time_utc, '1900-01-01'::timestamptz)" in migration


def test_station_registry_migration_declares_versioned_registry_table() -> None:
    migration = Path("alembic/versions/0002_klga_station_universe_registry.py").read_text(
        encoding="utf-8"
    )
    assert "CREATE TABLE IF NOT EXISTS registry.station_registry" in migration
    assert "station_registry_version text NOT NULL" in migration
    assert "grid_point_id text NOT NULL DEFAULT ''" in migration
    assert "PRIMARY KEY (station_registry_version, station_id, grid_point_id)" in migration
    assert "nearby_core" in migration
    assert "gridded_pseudo_point" in migration
