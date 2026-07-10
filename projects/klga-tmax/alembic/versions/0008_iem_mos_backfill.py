"""Add IEM MOS backfill tables.

Revision ID: 0008_iem_mos_backfill
Revises: 0007_gribstream_runs_endpoint
Create Date: 2026-06-30 12:00:00
"""

from __future__ import annotations

from alembic import op

revision = "0008_iem_mos_backfill"
down_revision = "0007_gribstream_runs_endpoint"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS audit.iem_mos_backfill_jobs (
            job_id text PRIMARY KEY,
            cutoff_id text NOT NULL REFERENCES registry.cutoffs(cutoff_id),
            start_date date NOT NULL,
            end_date date NOT NULL,
            status text NOT NULL,
            planned_chunks integer NOT NULL DEFAULT 0,
            completed_chunks integer NOT NULL DEFAULT 0,
            completed_empty_chunks integer NOT NULL DEFAULT 0,
            failed_chunks integer NOT NULL DEFAULT 0,
            rows_upserted bigint NOT NULL DEFAULT 0,
            feature_rows_upserted bigint NOT NULL DEFAULT 0,
            bytes_fetched bigint NOT NULL DEFAULT 0,
            config_json jsonb NOT NULL DEFAULT '{}'::jsonb,
            error_message text,
            started_at_utc timestamptz NOT NULL DEFAULT now(),
            finished_at_utc timestamptz,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT ck_iem_mos_jobs_status CHECK (
                status IN ('planned','running','completed','failed','blocked')
            ),
            CONSTRAINT ck_iem_mos_jobs_dates CHECK (end_date >= start_date)
        );

        CREATE INDEX IF NOT EXISTS ix_iem_mos_jobs_status
        ON audit.iem_mos_backfill_jobs(status, started_at_utc);

        CREATE TABLE IF NOT EXISTS audit.iem_mos_backfill_chunks (
            chunk_id text PRIMARY KEY,
            job_id text NOT NULL REFERENCES audit.iem_mos_backfill_jobs(job_id),
            station_id text NOT NULL REFERENCES registry.stations(station_id),
            mos_station_id text NOT NULL,
            source_product text NOT NULL,
            endpoint_model text NOT NULL,
            cutoff_id text NOT NULL REFERENCES registry.cutoffs(cutoff_id),
            window_start_utc timestamptz NOT NULL,
            window_end_utc timestamptz NOT NULL,
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
            feature_rows_upserted integer NOT NULL DEFAULT 0,
            gaps_upserted integer NOT NULL DEFAULT 0,
            response_size_bytes bigint NOT NULL DEFAULT 0,
            started_at_utc timestamptz,
            finished_at_utc timestamptz,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT ck_iem_mos_chunks_status CHECK (
                status IN (
                    'planned','running','completed','completed_empty','failed',
                    'rate_limited','skipped'
                )
            ),
            CONSTRAINT ck_iem_mos_chunks_window CHECK (window_end_utc > window_start_utc),
            CONSTRAINT ck_iem_mos_chunks_product CHECK (
                source_product IN ('MAV','MET','MEX','LAV','NBS','NBE')
            )
        );

        CREATE UNIQUE INDEX IF NOT EXISTS ux_iem_mos_chunks_job_request
        ON audit.iem_mos_backfill_chunks(job_id, request_sha256);

        CREATE INDEX IF NOT EXISTS ix_iem_mos_chunks_job_status
        ON audit.iem_mos_backfill_chunks(job_id, status, source_product, station_id);

        CREATE INDEX IF NOT EXISTS ix_iem_mos_chunks_station_product_window
        ON audit.iem_mos_backfill_chunks(station_id, source_product, window_start_utc, window_end_utc);

        CREATE TABLE IF NOT EXISTS audit.iem_mos_source_gaps (
            iem_mos_source_gap_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            job_id text REFERENCES audit.iem_mos_backfill_jobs(job_id),
            chunk_id text REFERENCES audit.iem_mos_backfill_chunks(chunk_id),
            station_id text REFERENCES registry.stations(station_id),
            mos_station_id text,
            source_product text,
            endpoint_model text,
            window_start_utc timestamptz,
            window_end_utc timestamptz,
            target_date date,
            cutoff_id text,
            gap_type text NOT NULL,
            gap_reason text NOT NULL,
            evidence_json jsonb NOT NULL DEFAULT '{}'::jsonb,
            first_detected_at_utc timestamptz NOT NULL DEFAULT now(),
            last_detected_at_utc timestamptz NOT NULL DEFAULT now(),
            created_at timestamptz NOT NULL DEFAULT now()
        );

        CREATE UNIQUE INDEX IF NOT EXISTS ux_iem_mos_source_gap_identity
        ON audit.iem_mos_source_gaps(
            COALESCE(job_id, ''),
            COALESCE(chunk_id, ''),
            COALESCE(station_id, ''),
            COALESCE(source_product, ''),
            COALESCE(window_start_utc, '1900-01-01'::timestamptz),
            COALESCE(window_end_utc, '1900-01-01'::timestamptz),
            COALESCE(target_date, '1900-01-01'::date),
            gap_type
        );

        CREATE INDEX IF NOT EXISTS ix_iem_mos_source_gaps_product
        ON audit.iem_mos_source_gaps(source_product, station_id, gap_type);

        CREATE TABLE IF NOT EXISTS silver.iem_mos_forecast_rows (
            iem_mos_forecast_row_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            station_id text NOT NULL REFERENCES registry.stations(station_id),
            mos_station_id text NOT NULL,
            source_product text NOT NULL,
            endpoint_model text NOT NULL,
            cutoff_id text NOT NULL REFERENCES registry.cutoffs(cutoff_id),
            run_time_utc timestamptz NOT NULL,
            forecast_valid_time_utc timestamptz NOT NULL,
            forecast_hour double precision,
            period_type text NOT NULL DEFAULT 'point',
            n_x_f double precision,
            tmp_f double precision,
            dpt_f double precision,
            wdr double precision,
            wsp_kt double precision,
            gst_kt double precision,
            sky_or_cloud double precision,
            pop double precision,
            qpf double precision,
            tstm_prob double precision,
            raw_values_jsonb jsonb NOT NULL,
            raw_payload_hash text NOT NULL,
            provider_available_at_utc timestamptz NOT NULL,
            effective_available_at_utc timestamptz NOT NULL,
            availability_method text NOT NULL,
            source_request_id text NOT NULL REFERENCES bronze.source_requests(source_request_id),
            source_record_id uuid REFERENCES bronze.source_records(source_record_id),
            request_sha256 text NOT NULL,
            raw_row_hash text NOT NULL,
            parser_version text NOT NULL,
            quality_flag text NOT NULL DEFAULT 'ok',
            quality_note text,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT ck_iem_mos_forecast_product CHECK (
                source_product IN ('MAV','MET','MEX','LAV','NBS','NBE')
            ),
            CONSTRAINT ck_iem_mos_forecast_availability CHECK (
                availability_method IN (
                    'observed_provider_timestamp',
                    'observed_ingest_timestamp',
                    'conservative_lag_rule',
                    'manual_override'
                )
            ),
            CONSTRAINT ck_iem_mos_forecast_quality CHECK (
                quality_flag IN ('ok','suspect','failed','missing','duplicate','revised')
            ),
            CONSTRAINT uq_iem_mos_forecast_raw_hash UNIQUE (raw_row_hash)
        );

        CREATE INDEX IF NOT EXISTS ix_iem_mos_forecast_station_product_runtime
        ON silver.iem_mos_forecast_rows(station_id, source_product, run_time_utc);

        CREATE INDEX IF NOT EXISTS ix_iem_mos_forecast_station_product_valid
        ON silver.iem_mos_forecast_rows(station_id, source_product, forecast_valid_time_utc);

        CREATE INDEX IF NOT EXISTS ix_iem_mos_forecast_valid
        ON silver.iem_mos_forecast_rows(forecast_valid_time_utc, run_time_utc);

        CREATE INDEX IF NOT EXISTS ix_iem_mos_forecast_available
        ON silver.iem_mos_forecast_rows(effective_available_at_utc);

        CREATE INDEX IF NOT EXISTS ix_iem_mos_forecast_request
        ON silver.iem_mos_forecast_rows(request_sha256, source_request_id);

        CREATE TABLE IF NOT EXISTS gold.iem_mos_daily_features (
            iem_mos_daily_feature_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            target_date date NOT NULL,
            cutoff_id text NOT NULL REFERENCES registry.cutoffs(cutoff_id),
            target_instance_id uuid REFERENCES gold.target_instances(target_instance_id)
                ON DELETE CASCADE,
            station_id text NOT NULL REFERENCES registry.stations(station_id),
            source_product text NOT NULL,
            chosen_run_time_utc timestamptz,
            latest_valid_time_utc timestamptz,
            max_source_available_at_utc timestamptz,
            availability_method text,
            tmax_f double precision,
            tmp_peak_window_max_f double precision,
            tmp_peak_window_mean_f double precision,
            dpt_peak_window_mean_f double precision,
            wind_speed_peak_window_mean_kt double precision,
            pop_max double precision,
            qpf_max double precision,
            tstm_prob_max double precision,
            source_row_count integer NOT NULL DEFAULT 0,
            source_trace_json jsonb NOT NULL DEFAULT '{}'::jsonb,
            feature_build_version text NOT NULL DEFAULT 'iem_mos_daily_features_v1',
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT uq_iem_mos_daily_features_identity UNIQUE (
                target_date,
                cutoff_id,
                station_id,
                source_product,
                feature_build_version
            )
        );

        CREATE INDEX IF NOT EXISTS ix_iem_mos_daily_features_target
        ON gold.iem_mos_daily_features(target_date, cutoff_id);

        CREATE INDEX IF NOT EXISTS ix_iem_mos_daily_features_station_product
        ON gold.iem_mos_daily_features(station_id, source_product, target_date);

        CREATE TABLE IF NOT EXISTS gold.iem_mos_feature_matrix_v1 (
            target_instance_id uuid PRIMARY KEY REFERENCES gold.target_instances(target_instance_id)
                ON DELETE CASCADE,
            target_date date NOT NULL,
            cutoff_id text NOT NULL REFERENCES registry.cutoffs(cutoff_id),
            feature_vector_json jsonb NOT NULL DEFAULT '{}'::jsonb,
            feature_trace_json jsonb NOT NULL DEFAULT '{}'::jsonb,
            source_feature_count integer NOT NULL DEFAULT 0,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        );

        CREATE INDEX IF NOT EXISTS ix_iem_mos_feature_matrix_date_cutoff
        ON gold.iem_mos_feature_matrix_v1(target_date, cutoff_id);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE IF EXISTS gold.iem_mos_feature_matrix_v1 CASCADE;
        DROP TABLE IF EXISTS gold.iem_mos_daily_features CASCADE;
        DROP TABLE IF EXISTS silver.iem_mos_forecast_rows CASCADE;
        DROP TABLE IF EXISTS audit.iem_mos_source_gaps CASCADE;
        DROP TABLE IF EXISTS audit.iem_mos_backfill_chunks CASCADE;
        DROP TABLE IF EXISTS audit.iem_mos_backfill_jobs CASCADE;
        """
    )
