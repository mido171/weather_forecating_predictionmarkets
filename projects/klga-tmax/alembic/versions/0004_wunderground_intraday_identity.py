"""Use station-time identity for Wunderground intraday facts.

Revision ID: 0004_wu_intraday_identity
Revises: 0003_wu_actuals
Create Date: 2026-06-28 18:20:00
"""

from __future__ import annotations

from alembic import op

revision = "0004_wu_intraday_identity"
down_revision = "0003_wu_actuals"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        DELETE FROM silver.wu_intraday_observations a
        USING silver.wu_intraday_observations b
        WHERE a.ctid < b.ctid
          AND a.station_id = b.station_id
          AND a.observation_time_utc = b.observation_time_utc;

        ALTER TABLE silver.wu_intraday_observations
        DROP CONSTRAINT IF EXISTS wu_intraday_observations_pkey;

        ALTER TABLE silver.wu_intraday_observations
        ADD CONSTRAINT wu_intraday_observations_pkey
        PRIMARY KEY (station_id, observation_time_utc);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE silver.wu_intraday_observations
        DROP CONSTRAINT IF EXISTS wu_intraday_observations_pkey;

        ALTER TABLE silver.wu_intraday_observations
        ADD CONSTRAINT wu_intraday_observations_pkey
        PRIMARY KEY (station_id, observation_time_utc, source_request_id);
        """
    )
