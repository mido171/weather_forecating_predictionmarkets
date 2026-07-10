from __future__ import annotations

from pathlib import Path

from klga_tmax.db.migrations_check import REQUIRED_COLUMNS, REQUIRED_INDEXES, REQUIRED_TABLES


def test_gribstream_tables_are_in_contract_list() -> None:
    observed = {
        f"{schema}.{table}"
        for schema, tables in REQUIRED_TABLES.items()
        for table in tables
    }
    assert {
        "audit.gribstream_catalog_snapshots",
        "audit.gribstream_backfill_jobs",
        "audit.gribstream_backfill_chunks",
        "audit.gribstream_source_gaps",
        "silver.grib_forecast_values",
    }.issubset(observed)


def test_gribstream_migration_declares_tracking_and_value_tables() -> None:
    migration = Path("alembic/versions/0005_gribstream_single_cutoff.py").read_text(
        encoding="utf-8"
    )
    assert "CREATE TABLE IF NOT EXISTS audit.gribstream_backfill_jobs" in migration
    assert "CREATE TABLE IF NOT EXISTS audit.gribstream_backfill_chunks" in migration
    assert "CREATE TABLE IF NOT EXISTS silver.grib_forecast_values" in migration
    assert "request_sha256 text NOT NULL" in migration
    assert "CONSTRAINT uq_grib_forecast_values_raw_hash UNIQUE" in migration
    assert "CREATE UNIQUE INDEX IF NOT EXISTS ux_gribstream_chunks_request_sha" in migration


def test_gribstream_job_chunk_identity_migration_keeps_request_hash_reusable() -> None:
    migration = Path("alembic/versions/0006_grib_job_chunk_identity.py").read_text(
        encoding="utf-8"
    )
    assert "DROP INDEX IF EXISTS audit.ux_gribstream_chunks_request_sha" in migration
    assert "CREATE INDEX IF NOT EXISTS ix_gribstream_chunks_request_sha" in migration
    assert "CREATE UNIQUE INDEX IF NOT EXISTS ux_gribstream_chunks_job_request" in migration
    assert "ON audit.gribstream_backfill_chunks(job_id, request_sha256)" in migration


def test_gribstream_runs_endpoint_migration_allows_runs_lineage() -> None:
    migration = Path("alembic/versions/0007_allow_gribstream_runs_endpoint.py").read_text(
        encoding="utf-8"
    )
    assert 'revision = "0007_gribstream_runs_endpoint"' in migration
    assert "endpoint_type IN ('timeseries','runs')" in migration
    assert "audit.gribstream_backfill_chunks" in migration
    assert "silver.grib_forecast_values" in migration


def test_contract_requires_gribstream_lineage_columns_and_indexes() -> None:
    assert "silver.grib_forecast_values" in REQUIRED_COLUMNS
    required = REQUIRED_COLUMNS["silver.grib_forecast_values"]
    assert "model_id" in required
    assert "member" in required
    assert "forecasted_at_utc" in required
    assert "forecasted_time_utc" in required
    assert "variable_alias" in required
    assert "source_request_id" in required
    assert "source_record_id" in required
    assert "request_sha256" in required
    assert "ix_gribstream_chunks_request_sha" in REQUIRED_INDEXES
    assert "ux_gribstream_chunks_job_request" in REQUIRED_INDEXES
    assert "uq_grib_forecast_values_raw_hash" in REQUIRED_INDEXES
