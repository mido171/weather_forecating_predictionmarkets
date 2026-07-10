"""Allow GribStream runs endpoint lineage.

Revision ID: 0007_gribstream_runs_endpoint
Revises: 0006_grib_job_chunk_identity
Create Date: 2026-06-29 18:30:00
"""

from __future__ import annotations

from alembic import op

revision = "0007_gribstream_runs_endpoint"
down_revision = "0006_grib_job_chunk_identity"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE audit.gribstream_backfill_chunks
            DROP CONSTRAINT IF EXISTS ck_gribstream_chunks_endpoint;
        ALTER TABLE audit.gribstream_backfill_chunks
            ADD CONSTRAINT ck_gribstream_chunks_endpoint
            CHECK (endpoint_type IN ('timeseries','runs'));

        ALTER TABLE silver.grib_forecast_values
            DROP CONSTRAINT IF EXISTS ck_grib_values_endpoint;
        ALTER TABLE silver.grib_forecast_values
            ADD CONSTRAINT ck_grib_values_endpoint
            CHECK (endpoint_type IN ('timeseries','runs'));
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE audit.gribstream_backfill_chunks
            DROP CONSTRAINT IF EXISTS ck_gribstream_chunks_endpoint;
        ALTER TABLE audit.gribstream_backfill_chunks
            ADD CONSTRAINT ck_gribstream_chunks_endpoint
            CHECK (endpoint_type = 'timeseries');

        ALTER TABLE silver.grib_forecast_values
            DROP CONSTRAINT IF EXISTS ck_grib_values_endpoint;
        ALTER TABLE silver.grib_forecast_values
            ADD CONSTRAINT ck_grib_values_endpoint
            CHECK (endpoint_type = 'timeseries');
        """
    )
