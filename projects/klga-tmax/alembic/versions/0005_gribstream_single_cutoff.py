"""Add GribStream single-cutoff backfill tables.

Revision ID: 0005_gribstream_single_cutoff
Revises: 0004_wu_intraday_identity
Create Date: 2026-06-28 21:45:00
"""

from __future__ import annotations

from alembic import op

revision = "0005_gribstream_single_cutoff"
down_revision = "0004_wu_intraday_identity"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS audit.gribstream_catalog_snapshots (
            gribstream_catalog_snapshot_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            model_id text NOT NULL,
            catalog_kind text NOT NULL,
            catalog_url text NOT NULL,
            payload_sha256 text NOT NULL,
            payload_json jsonb NOT NULL,
            retrieved_at_utc timestamptz NOT NULL,
            status text NOT NULL DEFAULT 'ok',
            error_message text,
            created_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT ck_gribstream_catalog_status CHECK (status IN ('ok','failed'))
        );

        CREATE UNIQUE INDEX IF NOT EXISTS ux_gribstream_catalog_snapshot
        ON audit.gribstream_catalog_snapshots(catalog_url, payload_sha256);

        CREATE INDEX IF NOT EXISTS ix_gribstream_catalog_model
        ON audit.gribstream_catalog_snapshots(model_id, retrieved_at_utc);

        CREATE TABLE IF NOT EXISTS audit.gribstream_backfill_jobs (
            job_id text PRIMARY KEY,
            cutoff_id text NOT NULL REFERENCES registry.cutoffs(cutoff_id),
            start_date date NOT NULL,
            end_date date NOT NULL,
            coordinate_tier text NOT NULL,
            status text NOT NULL,
            planned_chunks integer NOT NULL DEFAULT 0,
            completed_chunks integer NOT NULL DEFAULT 0,
            failed_chunks integer NOT NULL DEFAULT 0,
            estimated_credits integer NOT NULL DEFAULT 0,
            row_counts_json jsonb NOT NULL DEFAULT '{}'::jsonb,
            config_json jsonb NOT NULL DEFAULT '{}'::jsonb,
            error_message text,
            started_at_utc timestamptz NOT NULL DEFAULT now(),
            finished_at_utc timestamptz,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT ck_gribstream_jobs_status CHECK (
                status IN ('planned','running','completed','failed','blocked')
            ),
            CONSTRAINT ck_gribstream_jobs_dates CHECK (end_date >= start_date)
        );

        CREATE INDEX IF NOT EXISTS ix_gribstream_jobs_status
        ON audit.gribstream_backfill_jobs(status, started_at_utc);

        CREATE TABLE IF NOT EXISTS audit.gribstream_backfill_chunks (
            chunk_id text PRIMARY KEY,
            job_id text NOT NULL REFERENCES audit.gribstream_backfill_jobs(job_id),
            model_id text NOT NULL,
            target_start_date date NOT NULL,
            target_end_date date NOT NULL,
            cutoff_id text NOT NULL REFERENCES registry.cutoffs(cutoff_id),
            endpoint_type text NOT NULL,
            coordinate_tier text NOT NULL,
            as_of_utc timestamptz,
            valid_time_count integer NOT NULL,
            variable_count integer NOT NULL,
            member_count integer NOT NULL DEFAULT 1,
            estimated_credits integer NOT NULL DEFAULT 0,
            request_sha256 text NOT NULL,
            request_json jsonb NOT NULL,
            status text NOT NULL,
            attempts integer NOT NULL DEFAULT 0,
            http_status integer,
            error_type text,
            error_message text,
            source_request_id text REFERENCES bronze.source_requests(source_request_id),
            source_record_id uuid REFERENCES bronze.source_records(source_record_id),
            raw_storage_uri text,
            rows_upserted integer NOT NULL DEFAULT 0,
            availability_rows_upserted integer NOT NULL DEFAULT 0,
            gaps_upserted integer NOT NULL DEFAULT 0,
            started_at_utc timestamptz,
            finished_at_utc timestamptz,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT ck_gribstream_chunks_endpoint CHECK (endpoint_type = 'timeseries'),
            CONSTRAINT ck_gribstream_chunks_status CHECK (
                status IN (
                    'planned','running','completed','completed_empty','failed',
                    'rate_limited','auth_failed','selector_missing','skipped'
                )
            ),
            CONSTRAINT ck_gribstream_chunks_dates CHECK (target_end_date >= target_start_date)
        );

        CREATE UNIQUE INDEX IF NOT EXISTS ux_gribstream_chunks_request_sha
        ON audit.gribstream_backfill_chunks(request_sha256);

        CREATE INDEX IF NOT EXISTS ix_gribstream_chunks_job_status
        ON audit.gribstream_backfill_chunks(job_id, status, model_id, target_start_date);

        CREATE INDEX IF NOT EXISTS ix_gribstream_chunks_model_dates
        ON audit.gribstream_backfill_chunks(model_id, target_start_date, target_end_date);

        CREATE TABLE IF NOT EXISTS audit.gribstream_source_gaps (
            gribstream_source_gap_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            model_id text NOT NULL,
            target_start_date date,
            target_end_date date,
            cutoff_id text,
            grid_point_id text,
            variable_alias text,
            variable_name text,
            member text,
            gap_type text NOT NULL,
            gap_reason text NOT NULL,
            evidence_json jsonb NOT NULL DEFAULT '{}'::jsonb,
            first_detected_at_utc timestamptz NOT NULL DEFAULT now(),
            last_detected_at_utc timestamptz NOT NULL DEFAULT now(),
            created_at timestamptz NOT NULL DEFAULT now()
        );

        CREATE UNIQUE INDEX IF NOT EXISTS ux_gribstream_source_gap_identity
        ON audit.gribstream_source_gaps(
            model_id,
            COALESCE(target_start_date, '1900-01-01'::date),
            COALESCE(target_end_date, '1900-01-01'::date),
            COALESCE(cutoff_id, ''),
            COALESCE(grid_point_id, ''),
            COALESCE(variable_alias, ''),
            COALESCE(member, ''),
            gap_type
        );

        CREATE INDEX IF NOT EXISTS ix_gribstream_source_gaps_model
        ON audit.gribstream_source_gaps(model_id, gap_type);

        CREATE TABLE IF NOT EXISTS silver.grib_forecast_values (
            grib_value_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            model_id text NOT NULL,
            endpoint_type text NOT NULL,
            target_date date NOT NULL,
            cutoff_id text NOT NULL REFERENCES registry.cutoffs(cutoff_id),
            cutoff_utc timestamptz NOT NULL,
            as_of_utc timestamptz,
            coordinate_tier text NOT NULL,
            grid_point_id text NOT NULL REFERENCES registry.stations(station_id),
            lat double precision NOT NULL,
            lon double precision NOT NULL,
            forecasted_at_utc timestamptz NOT NULL,
            forecasted_time_utc timestamptz NOT NULL,
            forecast_hour double precision NOT NULL,
            member text NOT NULL DEFAULT 'deterministic',
            variable_alias text NOT NULL,
            variable_name text NOT NULL,
            variable_level text,
            variable_info text,
            unit_original text,
            value_original double precision,
            unit_canonical text,
            value_canonical double precision,
            index_updated_at_utc timestamptz,
            provider_available_at_utc timestamptz NOT NULL,
            effective_available_at_utc timestamptz NOT NULL,
            our_ingested_at_utc timestamptz NOT NULL,
            availability_method text NOT NULL,
            source_request_id text NOT NULL REFERENCES bronze.source_requests(source_request_id),
            source_record_id uuid REFERENCES bronze.source_records(source_record_id),
            request_sha256 text NOT NULL,
            raw_row_hash text NOT NULL,
            raw_row_json jsonb NOT NULL,
            quality_flag text NOT NULL DEFAULT 'ok',
            quality_note text,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT ck_grib_values_endpoint CHECK (endpoint_type = 'timeseries'),
            CONSTRAINT ck_grib_values_availability_method CHECK (
                availability_method IN (
                    'observed_provider_timestamp',
                    'observed_ingest_timestamp',
                    'conservative_lag_rule',
                    'manual_override'
                )
            ),
            CONSTRAINT ck_grib_values_quality CHECK (
                quality_flag IN ('ok','suspect','failed','missing','duplicate','revised')
            ),
            CONSTRAINT uq_grib_forecast_values_raw_hash UNIQUE (raw_row_hash)
        );

        CREATE INDEX IF NOT EXISTS ix_grib_values_model_target
        ON silver.grib_forecast_values(model_id, target_date, cutoff_id);

        CREATE INDEX IF NOT EXISTS ix_grib_values_valid_time
        ON silver.grib_forecast_values(forecasted_time_utc, forecasted_at_utc);

        CREATE INDEX IF NOT EXISTS ix_grib_values_request
        ON silver.grib_forecast_values(request_sha256, source_request_id);

        CREATE INDEX IF NOT EXISTS ix_grib_values_coordinate_variable
        ON silver.grib_forecast_values(grid_point_id, variable_alias, target_date);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE IF EXISTS silver.grib_forecast_values CASCADE;
        DROP TABLE IF EXISTS audit.gribstream_source_gaps CASCADE;
        DROP TABLE IF EXISTS audit.gribstream_backfill_chunks CASCADE;
        DROP TABLE IF EXISTS audit.gribstream_backfill_jobs CASCADE;
        DROP TABLE IF EXISTS audit.gribstream_catalog_snapshots CASCADE;
        """
    )
