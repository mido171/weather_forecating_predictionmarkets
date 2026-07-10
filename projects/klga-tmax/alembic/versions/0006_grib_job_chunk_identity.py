"""Make GribStream chunk identity job-scoped.

Revision ID: 0006_grib_job_chunk_identity
Revises: 0005_gribstream_single_cutoff
Create Date: 2026-06-28 23:10:00
"""

from __future__ import annotations

from alembic import op

revision = "0006_grib_job_chunk_identity"
down_revision = "0005_gribstream_single_cutoff"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        DROP INDEX IF EXISTS audit.ux_gribstream_chunks_request_sha;

        CREATE INDEX IF NOT EXISTS ix_gribstream_chunks_request_sha
        ON audit.gribstream_backfill_chunks(request_sha256);

        CREATE UNIQUE INDEX IF NOT EXISTS ux_gribstream_chunks_job_request
        ON audit.gribstream_backfill_chunks(job_id, request_sha256);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP INDEX IF EXISTS audit.ux_gribstream_chunks_job_request;
        DROP INDEX IF EXISTS audit.ix_gribstream_chunks_request_sha;

        CREATE UNIQUE INDEX IF NOT EXISTS ux_gribstream_chunks_request_sha
        ON audit.gribstream_backfill_chunks(request_sha256);
        """
    )
