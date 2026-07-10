"""Replace legacy WU persistence with plain settled Tmax truth table.

Revision ID: 0010_wu_truth
Revises: 0009_forecast_eval
Create Date: 2026-07-02 18:00:00
"""

from __future__ import annotations

from alembic import op

revision = "0010_wu_truth"
down_revision = "0009_forecast_eval"
branch_labels = None
depends_on = None


def upgrade() -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS public.wunderground_daily_tmax (
            station_id text NOT NULL REFERENCES registry.stations(station_id),
            wunderground_station_id text NOT NULL,
            local_date date NOT NULL,
            timezone_name text NOT NULL DEFAULT 'America/New_York',
            tmax_f integer,
            tmin_f integer,
            observation_count integer NOT NULL DEFAULT 0,
            high_observation_times_local_json jsonb NOT NULL DEFAULT '[]'::jsonb,
            hourly_observations_json jsonb NOT NULL DEFAULT '[]'::jsonb,
            provider_max_temp_values_json jsonb NOT NULL DEFAULT '[]'::jsonb,
            provider_min_temp_values_json jsonb NOT NULL DEFAULT '[]'::jsonb,
            source_url_redacted text,
            wu_page_url text NOT NULL,
            payload_hash text,
            parser_version text NOT NULL,
            fetched_at_utc timestamptz NOT NULL,
            settlement_available_at_utc timestamptz,
            daily_high_source text NOT NULL DEFAULT 'hourly_temp_max',
            validation_status text NOT NULL,
            validation_notes_json jsonb NOT NULL DEFAULT '{}'::jsonb,
            fetch_status text NOT NULL DEFAULT 'saved',
            http_status integer,
            error_type text,
            error_message text,
            attempts integer NOT NULL DEFAULT 1,
            created_at_utc timestamptz NOT NULL DEFAULT now(),
            updated_at_utc timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (station_id, local_date),
            CONSTRAINT ck_wu_truth_high_source CHECK (daily_high_source = 'hourly_temp_max'),
            CONSTRAINT ck_wu_truth_validation_status CHECK (
                validation_status IN ('accepted','manual_confirmed','suspect','no_data','fetch_failed')
            ),
            CONSTRAINT ck_wu_truth_fetch_status CHECK (
                fetch_status IN ('saved','no_data','fetch_failed')
            ),
            CONSTRAINT ck_wu_truth_tmax_range CHECK (tmax_f IS NULL OR tmax_f BETWEEN -30 AND 120),
            CONSTRAINT ck_wu_truth_tmin_range CHECK (tmin_f IS NULL OR tmin_f BETWEEN -40 AND 110),
            CONSTRAINT ck_wu_truth_tmax_ge_tmin CHECK (
                tmax_f IS NULL OR tmin_f IS NULL OR tmax_f >= tmin_f
            ),
            CONSTRAINT ck_wu_truth_observation_count CHECK (observation_count >= 0)
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_wunderground_daily_tmax_station_date
        ON public.wunderground_daily_tmax(station_id, local_date)
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_wunderground_daily_tmax_status
        ON public.wunderground_daily_tmax(validation_status, fetch_status)
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_wunderground_daily_tmax_available
        ON public.wunderground_daily_tmax(settlement_available_at_utc)
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_wunderground_daily_tmax_payload_hash
        ON public.wunderground_daily_tmax(payload_hash)
        """,
        """
        DELETE FROM silver.target_daily_actuals
        WHERE source_name = 'wunderground'
        """,
        """
        DELETE FROM silver.station_daily_actuals
        WHERE source_name = 'wunderground'
        """,
        """
        DELETE FROM silver.station_observations
        WHERE source_name = 'wunderground'
        """,
        """
        CREATE SCHEMA IF NOT EXISTS legacy_drop_pending
        """,
        """
        DO $$
        DECLARE
            constraint_row record;
        BEGIN
            FOR constraint_row IN
                SELECT conrelid::regclass::text AS table_name, conname
                FROM pg_constraint
                WHERE contype = 'f'
                  AND conrelid IN (
                    'silver.wu_daily_actual_revisions'::regclass,
                    'silver.wu_intraday_observations'::regclass,
                    'silver.wu_daily_actuals'::regclass,
                    'audit.wu_station_date_coverage'::regclass,
                    'audit.wu_fetch_windows'::regclass
                  )
            LOOP
                EXECUTE format(
                    'ALTER TABLE %s DROP CONSTRAINT IF EXISTS %I',
                    constraint_row.table_name,
                    constraint_row.conname
                );
            END LOOP;
        END $$
        """,
        """
        ALTER TABLE IF EXISTS silver.wu_daily_actual_revisions
        RENAME TO wu_daily_actual_revisions_legacy_0010
        """,
        """
        ALTER TABLE IF EXISTS silver.wu_daily_actual_revisions_legacy_0010
        SET SCHEMA legacy_drop_pending
        """,
        """
        ALTER TABLE IF EXISTS silver.wu_intraday_observations
        RENAME TO wu_intraday_observations_legacy_0010
        """,
        """
        ALTER TABLE IF EXISTS silver.wu_intraday_observations_legacy_0010
        SET SCHEMA legacy_drop_pending
        """,
        """
        ALTER TABLE IF EXISTS silver.wu_daily_actuals
        RENAME TO wu_daily_actuals_legacy_0010
        """,
        """
        ALTER TABLE IF EXISTS silver.wu_daily_actuals_legacy_0010
        SET SCHEMA legacy_drop_pending
        """,
        """
        ALTER TABLE IF EXISTS audit.wu_station_date_coverage
        RENAME TO wu_station_date_coverage_legacy_0010
        """,
        """
        ALTER TABLE IF EXISTS audit.wu_station_date_coverage_legacy_0010
        SET SCHEMA legacy_drop_pending
        """,
        """
        ALTER TABLE IF EXISTS audit.wu_fetch_windows
        RENAME TO wu_fetch_windows_legacy_0010
        """,
        """
        ALTER TABLE IF EXISTS audit.wu_fetch_windows_legacy_0010
        SET SCHEMA legacy_drop_pending
        """,
        """
        UPDATE reports.forecast_evaluation_daily_scores
        SET label_source_record_id = NULL,
            diagnostics_json = diagnostics_json || jsonb_build_object(
                'legacy_label_source_record_removed',
                true
            )
        WHERE label_source_record_id IN (
            SELECT source_record_id
            FROM bronze.source_records
            WHERE source_name = 'wunderground'
               OR provider_name = 'weathercom'
        )
        """,
        """
        UPDATE bronze.source_records
        SET source_name = 'wunderground_legacy_0010',
            provider_name = 'weathercom_legacy_0010',
            is_current = false
        WHERE source_name = 'wunderground'
           OR provider_name = 'weathercom'
        """,
        """
        UPDATE bronze.source_requests
        SET source_name = 'wunderground_legacy_0010'
        WHERE source_name = 'wunderground'
        """,
    ]
    for statement in statements:
        op.execute(statement)


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE IF EXISTS public.wunderground_daily_tmax CASCADE;
        """
    )
