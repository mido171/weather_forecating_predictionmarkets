"""Add task-02 Wunderground settlement actuals tables.

Revision ID: 0003_wu_actuals
Revises: 0002_station_universe
Create Date: 2026-06-28 18:00:00
"""

from __future__ import annotations

from alembic import op

revision = "0003_wu_actuals"
down_revision = "0002_station_universe"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS silver.wu_daily_actuals (
            station_id text NOT NULL REFERENCES registry.stations(station_id),
            wunderground_station_id text NOT NULL,
            weathercom_location_id text NOT NULL,
            local_date date NOT NULL,
            timezone_name text NOT NULL,
            local_day_start_utc timestamptz NOT NULL,
            local_day_end_utc timestamptz NOT NULL,
            daily_high_f integer,
            settlement_high_f_whole integer,
            daily_low_f integer,
            daily_avg_temp_f double precision,
            daily_high_dewpoint_f double precision,
            daily_low_dewpoint_f double precision,
            daily_precipitation_in double precision,
            daily_max_wind_speed_mph double precision,
            daily_max_wind_gust_mph double precision,
            daily_avg_wind_speed_mph double precision,
            daily_dominant_wind_direction_deg double precision,
            label_method text,
            daily_high_source_field text,
            provider_available_at_utc timestamptz NOT NULL,
            our_ingested_at_utc timestamptz NOT NULL,
            source_request_id text NOT NULL REFERENCES bronze.source_requests(source_request_id),
            source_record_id uuid REFERENCES bronze.source_records(source_record_id),
            source_daily_summary_json jsonb NOT NULL DEFAULT '{}'::jsonb,
            raw_daily_json jsonb NOT NULL DEFAULT '{}'::jsonb,
            observations_count integer NOT NULL DEFAULT 0,
            quality_flag text NOT NULL DEFAULT 'ok',
            quality_note text,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (station_id, local_date),
            CONSTRAINT ck_wu_daily_label_method CHECK (
                label_method IS NULL OR label_method IN (
                    'wunderground_daily_summary',
                    'computed_from_wunderground_intraday_rows'
                )
            ),
            CONSTRAINT ck_wu_daily_quality_flag CHECK (
                quality_flag IN ('ok','suspect','failed','missing','revised')
            ),
            CONSTRAINT ck_wu_daily_high_range CHECK (
                daily_high_f IS NULL OR daily_high_f BETWEEN -30 AND 120
            ),
            CONSTRAINT ck_wu_daily_low_range CHECK (
                daily_low_f IS NULL OR daily_low_f BETWEEN -40 AND 110
            ),
            CONSTRAINT ck_wu_daily_high_ge_low CHECK (
                daily_high_f IS NULL OR daily_low_f IS NULL OR daily_high_f >= daily_low_f
            )
        );

        CREATE INDEX IF NOT EXISTS ix_wu_daily_actuals_station_date
        ON silver.wu_daily_actuals(station_id, local_date);

        CREATE INDEX IF NOT EXISTS ix_wu_daily_actuals_provider_available
        ON silver.wu_daily_actuals(provider_available_at_utc);

        CREATE TABLE IF NOT EXISTS silver.wu_intraday_observations (
            station_id text NOT NULL REFERENCES registry.stations(station_id),
            wunderground_station_id text NOT NULL,
            weathercom_location_id text NOT NULL,
            observation_time_local timestamptz NOT NULL,
            observation_time_utc timestamptz NOT NULL,
            local_date date NOT NULL,
            timezone_name text NOT NULL,
            temp_f double precision,
            dewpoint_f double precision,
            humidity_pct double precision,
            wind_speed_mph double precision,
            wind_gust_mph double precision,
            wind_direction_deg double precision,
            pressure_in double precision,
            precipitation_in double precision,
            condition_text text,
            cloud_cover_text text,
            uv_index double precision,
            solar_radiation double precision,
            raw_observation_json jsonb NOT NULL,
            provider_available_at_utc timestamptz NOT NULL,
            our_ingested_at_utc timestamptz NOT NULL,
            source_request_id text NOT NULL REFERENCES bronze.source_requests(source_request_id),
            source_record_id uuid REFERENCES bronze.source_records(source_record_id),
            quality_flag text NOT NULL DEFAULT 'ok',
            quality_note text,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (station_id, observation_time_utc),
            CONSTRAINT ck_wu_intraday_quality_flag CHECK (
                quality_flag IN ('ok','suspect','failed','missing','duplicate','revised')
            ),
            CONSTRAINT ck_wu_intraday_humidity CHECK (
                humidity_pct IS NULL OR humidity_pct BETWEEN 0 AND 100
            ),
            CONSTRAINT ck_wu_intraday_wind_speed CHECK (
                wind_speed_mph IS NULL OR wind_speed_mph BETWEEN 0 AND 150
            ),
            CONSTRAINT ck_wu_intraday_precip CHECK (
                precipitation_in IS NULL OR precipitation_in BETWEEN 0 AND 20
            )
        );

        CREATE INDEX IF NOT EXISTS ix_wu_intraday_station_time
        ON silver.wu_intraday_observations(station_id, observation_time_utc);

        CREATE INDEX IF NOT EXISTS ix_wu_intraday_local_date
        ON silver.wu_intraday_observations(station_id, local_date);

        CREATE TABLE IF NOT EXISTS silver.wu_daily_actual_revisions (
            wu_daily_actual_revision_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            station_id text NOT NULL REFERENCES registry.stations(station_id),
            local_date date NOT NULL,
            previous_daily_high_f integer,
            new_daily_high_f integer,
            previous_source_request_id text REFERENCES bronze.source_requests(source_request_id),
            new_source_request_id text REFERENCES bronze.source_requests(source_request_id),
            previous_source_record_id uuid REFERENCES bronze.source_records(source_record_id),
            new_source_record_id uuid REFERENCES bronze.source_records(source_record_id),
            detected_at_utc timestamptz NOT NULL,
            note text
        );

        CREATE INDEX IF NOT EXISTS ix_wu_daily_revisions_station_date
        ON silver.wu_daily_actual_revisions(station_id, local_date, detected_at_utc);

        CREATE TABLE IF NOT EXISTS audit.wu_fetch_windows (
            wu_fetch_window_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            job_id text NOT NULL,
            station_id text NOT NULL REFERENCES registry.stations(station_id),
            wunderground_station_id text NOT NULL,
            weathercom_location_id text NOT NULL,
            window_start_date date NOT NULL,
            window_end_date date NOT NULL,
            units text NOT NULL DEFAULT 'e',
            status text NOT NULL,
            attempts integer NOT NULL DEFAULT 0,
            http_status integer,
            error_type text,
            error_message text,
            source_request_id text REFERENCES bronze.source_requests(source_request_id),
            source_record_id uuid REFERENCES bronze.source_records(source_record_id),
            observations_count integer NOT NULL DEFAULT 0,
            daily_rows_upserted integer NOT NULL DEFAULT 0,
            intraday_rows_upserted integer NOT NULL DEFAULT 0,
            started_at_utc timestamptz,
            finished_at_utc timestamptz,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT ck_wu_fetch_windows_status CHECK (
                status IN ('pending','running','succeeded','failed','no_data','skipped')
            ),
            CONSTRAINT ck_wu_fetch_windows_date_order CHECK (
                window_end_date >= window_start_date
            ),
            CONSTRAINT uq_wu_fetch_windows_job_station_window UNIQUE (
                job_id,
                station_id,
                window_start_date,
                window_end_date
            )
        );

        CREATE INDEX IF NOT EXISTS ix_wu_fetch_windows_status
        ON audit.wu_fetch_windows(status, station_id, window_start_date);

        CREATE TABLE IF NOT EXISTS audit.wu_station_date_coverage (
            station_id text NOT NULL REFERENCES registry.stations(station_id),
            local_date date NOT NULL,
            wunderground_station_id text NOT NULL,
            weathercom_location_id text NOT NULL,
            status text NOT NULL,
            source_request_id text REFERENCES bronze.source_requests(source_request_id),
            source_record_id uuid REFERENCES bronze.source_records(source_record_id),
            wu_fetch_window_id uuid REFERENCES audit.wu_fetch_windows(wu_fetch_window_id),
            daily_actual_present boolean NOT NULL DEFAULT false,
            intraday_observation_count integer NOT NULL DEFAULT 0,
            first_attempt_at_utc timestamptz,
            last_attempt_at_utc timestamptz,
            last_success_at_utc timestamptz,
            last_error_type text,
            last_error_message text,
            quality_flag text NOT NULL DEFAULT 'ok',
            updated_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (station_id, local_date),
            CONSTRAINT ck_wu_station_date_coverage_status CHECK (
                status IN ('saved','failed','no_data','not_fetched')
            ),
            CONSTRAINT ck_wu_station_date_coverage_quality CHECK (
                quality_flag IN ('ok','suspect','failed','missing')
            )
        );

        CREATE INDEX IF NOT EXISTS ix_wu_station_date_coverage_status
        ON audit.wu_station_date_coverage(status, station_id, local_date);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE IF EXISTS audit.wu_station_date_coverage CASCADE;
        DROP TABLE IF EXISTS audit.wu_fetch_windows CASCADE;
        DROP TABLE IF EXISTS silver.wu_daily_actual_revisions CASCADE;
        DROP TABLE IF EXISTS silver.wu_intraday_observations CASCADE;
        DROP TABLE IF EXISTS silver.wu_daily_actuals CASCADE;
        """
    )
