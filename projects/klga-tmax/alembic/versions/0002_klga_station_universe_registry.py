"""Add task-01 versioned station universe registry.

Revision ID: 0002_station_universe
Revises: 0001_klga_tmax_core_schema
Create Date: 2026-06-28 13:30:00
"""

from __future__ import annotations

from alembic import op

revision = "0002_station_universe"
down_revision = "0001_klga_tmax_core_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE registry.stations
        DROP CONSTRAINT IF EXISTS ck_stations_role;

        UPDATE registry.stations
        SET station_role = CASE
            WHEN station_role = 'nearby' THEN 'nearby_core'
            WHEN station_role = 'pseudo_point' THEN 'gridded_pseudo_point'
            WHEN station_role = 'external_context' THEN 'regional_context'
            ELSE station_role
        END;

        ALTER TABLE registry.stations
        ADD CONSTRAINT ck_stations_role CHECK (
            station_role IN (
                'target',
                'nearby_core',
                'regional_context',
                'gridded_pseudo_point'
            )
        );

        CREATE TABLE IF NOT EXISTS registry.station_registry (
            station_registry_version text NOT NULL,
            station_id text NOT NULL,
            iem_asos_id text,
            wunderground_station_id text,
            mos_station_id text,
            grid_point_id text NOT NULL DEFAULT '',
            role text NOT NULL,
            lat double precision NOT NULL,
            lon double precision NOT NULL,
            elevation_m double precision,
            source_native_metadata_json jsonb NOT NULL DEFAULT '{}'::jsonb,
            active_from_date date NOT NULL DEFAULT '1900-01-01',
            active_until_date date,
            notes text,
            created_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (station_registry_version, station_id, grid_point_id),
            CONSTRAINT ck_station_registry_role CHECK (
                role IN (
                    'target',
                    'nearby_core',
                    'regional_context',
                    'gridded_pseudo_point'
                )
            ),
            CONSTRAINT ck_station_registry_lat CHECK (lat BETWEEN -90 AND 90),
            CONSTRAINT ck_station_registry_lon CHECK (lon BETWEEN -180 AND 180),
            CONSTRAINT ck_station_registry_active_dates CHECK (
                active_until_date IS NULL OR active_until_date >= active_from_date
            )
        );

        CREATE INDEX IF NOT EXISTS ix_station_registry_role
        ON registry.station_registry(station_registry_version, role);

        CREATE INDEX IF NOT EXISTS ix_station_registry_grid_point
        ON registry.station_registry(station_registry_version, grid_point_id)
        WHERE grid_point_id <> '';

        CREATE INDEX IF NOT EXISTS ix_station_registry_iem_asos_id
        ON registry.station_registry(station_registry_version, iem_asos_id)
        WHERE iem_asos_id IS NOT NULL;

        CREATE INDEX IF NOT EXISTS ix_station_registry_mos_station_id
        ON registry.station_registry(station_registry_version, mos_station_id)
        WHERE mos_station_id IS NOT NULL;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE IF EXISTS registry.station_registry CASCADE;

        ALTER TABLE registry.stations
        DROP CONSTRAINT IF EXISTS ck_stations_role;

        UPDATE registry.stations
        SET station_role = CASE
            WHEN station_role IN ('nearby_core', 'regional_context') THEN 'nearby'
            WHEN station_role = 'gridded_pseudo_point' THEN 'pseudo_point'
            ELSE station_role
        END;

        ALTER TABLE registry.stations
        ADD CONSTRAINT ck_stations_role CHECK (
            station_role IN ('target','nearby','pseudo_point','external_context')
        );
        """
    )
