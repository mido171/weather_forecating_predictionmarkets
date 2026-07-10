"""KLGA Tmax task-00 foundation schema.

Revision ID: 0001_klga_tmax_core_schema
Revises:
Create Date: 2026-06-28 10:15:00
"""

from __future__ import annotations

from alembic import op

revision = "0001_klga_tmax_core_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE EXTENSION IF NOT EXISTS pgcrypto;

        CREATE SCHEMA IF NOT EXISTS registry;
        CREATE SCHEMA IF NOT EXISTS bronze;
        CREATE SCHEMA IF NOT EXISTS silver;
        CREATE SCHEMA IF NOT EXISTS gold;
        CREATE SCHEMA IF NOT EXISTS predictions;
        CREATE SCHEMA IF NOT EXISTS trading;
        CREATE SCHEMA IF NOT EXISTS reports;
        CREATE SCHEMA IF NOT EXISTS audit;

        CREATE TABLE IF NOT EXISTS registry.stations (
            station_id text PRIMARY KEY,
            station_name text NOT NULL,
            provider_primary_id text NOT NULL,
            latitude double precision NOT NULL,
            longitude double precision NOT NULL,
            elevation_m double precision,
            timezone text NOT NULL DEFAULT 'America/New_York',
            station_role text NOT NULL,
            station_group text[] NOT NULL DEFAULT '{}',
            active boolean NOT NULL DEFAULT true,
            created_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT ck_stations_latitude CHECK (latitude BETWEEN -90 AND 90),
            CONSTRAINT ck_stations_longitude CHECK (longitude BETWEEN -180 AND 180),
            CONSTRAINT ck_stations_role CHECK (
                station_role IN ('target','nearby','pseudo_point','external_context')
            )
        );

        CREATE TABLE IF NOT EXISTS registry.cutoffs (
            cutoff_id text PRIMARY KEY,
            cutoff_order integer NOT NULL UNIQUE,
            timezone_name text NOT NULL,
            local_time time NOT NULL,
            target_day_offset integer NOT NULL,
            description text NOT NULL,
            active boolean NOT NULL DEFAULT true
        );

        CREATE TABLE IF NOT EXISTS registry.feature_versions (
            feature_version_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            feature_set_name text NOT NULL,
            feature_version text NOT NULL,
            source_code_git_sha text NOT NULL,
            formula_contract_hash text NOT NULL,
            feature_names text[] NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT uq_feature_versions_name UNIQUE (feature_set_name, feature_version)
        );

        CREATE TABLE IF NOT EXISTS registry.model_versions (
            model_version_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            model_family text NOT NULL,
            model_name text NOT NULL,
            model_role text NOT NULL,
            source_code_git_sha text NOT NULL,
            training_data_start date,
            training_data_end date,
            feature_version_id uuid REFERENCES registry.feature_versions(feature_version_id),
            hyperparams jsonb NOT NULL DEFAULT '{}'::jsonb,
            artifact_uri text,
            artifact_hash text,
            used_fallback_model boolean NOT NULL DEFAULT false,
            created_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT ck_model_versions_role CHECK (
                model_role IN ('expert','meta_combiner','calibrator','simulation','report')
            )
        );

        CREATE UNIQUE INDEX IF NOT EXISTS ux_model_versions_identity
        ON registry.model_versions (
            model_family,
            model_name,
            source_code_git_sha,
            COALESCE(training_data_start, '1900-01-01'::date),
            COALESCE(training_data_end, '1900-01-01'::date),
            md5(hyperparams::text)
        );

        CREATE TABLE IF NOT EXISTS audit.pipeline_runs (
            pipeline_run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            command_name text NOT NULL,
            command_args jsonb NOT NULL DEFAULT '{}'::jsonb,
            started_at timestamptz NOT NULL DEFAULT now(),
            finished_at timestamptz,
            status text NOT NULL,
            exit_code integer,
            source_code_git_sha text NOT NULL,
            row_counts jsonb NOT NULL DEFAULT '{}'::jsonb,
            error_message text,
            log_uri text,
            CONSTRAINT ck_pipeline_status CHECK (
                status IN ('started','success','failed','skipped')
            )
        );

        CREATE TABLE IF NOT EXISTS audit.ingestion_manifests (
            job_id text PRIMARY KEY,
            source_name text NOT NULL,
            code_version_git_sha text NOT NULL,
            config_hash text NOT NULL,
            started_at_utc timestamptz NOT NULL,
            finished_at_utc timestamptz,
            row_counts_bronze integer NOT NULL DEFAULT 0,
            row_counts_silver integer NOT NULL DEFAULT 0,
            row_counts_gold integer NOT NULL DEFAULT 0,
            errors jsonb NOT NULL DEFAULT '[]'::jsonb,
            warnings jsonb NOT NULL DEFAULT '[]'::jsonb,
            manifest_uri text,
            created_at timestamptz NOT NULL DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS audit.data_quality_failures (
            data_quality_failure_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            pipeline_run_id uuid REFERENCES audit.pipeline_runs(pipeline_run_id),
            table_name text NOT NULL,
            record_key text,
            check_name text NOT NULL,
            severity text NOT NULL,
            message text NOT NULL,
            observed_value_json jsonb NOT NULL DEFAULT '{}'::jsonb,
            created_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT ck_data_quality_severity CHECK (
                severity IN ('warning','error','fatal')
            )
        );

        CREATE INDEX IF NOT EXISTS ix_data_quality_failures_check
        ON audit.data_quality_failures(check_name, severity);

        CREATE TABLE IF NOT EXISTS bronze.source_requests (
            source_request_id text PRIMARY KEY,
            source_name text NOT NULL,
            source_endpoint text NOT NULL,
            request_method text NOT NULL,
            request_params_json jsonb NOT NULL DEFAULT '{}'::jsonb,
            request_headers_redacted jsonb NOT NULL DEFAULT '{}'::jsonb,
            retrieved_at_utc timestamptz NOT NULL,
            provider_response_timestamp timestamptz,
            http_status integer,
            response_content_type text,
            response_body_sha256 text NOT NULL,
            response_size_bytes bigint NOT NULL,
            raw_storage_uri text NOT NULL,
            parser_version text,
            created_at timestamptz NOT NULL DEFAULT now()
        );

        CREATE INDEX IF NOT EXISTS ix_source_requests_source_retrieved
        ON bronze.source_requests(source_name, retrieved_at_utc);

        CREATE TABLE IF NOT EXISTS bronze.source_records (
            source_record_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            source_request_id text REFERENCES bronze.source_requests(source_request_id),
            source_name text NOT NULL,
            provider_name text NOT NULL,
            endpoint_name text NOT NULL,
            provider_record_key text NOT NULL,
            request_hash text,
            payload_hash text NOT NULL,
            payload_format text NOT NULL,
            payload_json jsonb,
            payload_text text,
            payload_uri text,
            provider_issued_at_utc timestamptz,
            provider_valid_at_utc timestamptz,
            provider_available_at_utc timestamptz,
            acquired_at_utc timestamptz NOT NULL,
            revision_number integer NOT NULL DEFAULT 1,
            supersedes_source_record_id uuid REFERENCES bronze.source_records(source_record_id),
            is_current boolean NOT NULL DEFAULT true,
            created_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT ck_source_records_payload_format CHECK (
                payload_format IN ('json','csv','ndjson','text','parquet','binary_uri')
            ),
            CONSTRAINT ck_source_records_payload_present CHECK (
                ((payload_json IS NOT NULL)::int +
                 (payload_text IS NOT NULL)::int +
                 (payload_uri IS NOT NULL)::int) >= 1
            ),
            CONSTRAINT uq_source_records_revision UNIQUE (
                source_name,
                provider_name,
                endpoint_name,
                provider_record_key,
                revision_number
            )
        );

        CREATE INDEX IF NOT EXISTS ix_bronze_source_records_provider_time
        ON bronze.source_records(
            source_name,
            provider_name,
            provider_issued_at_utc,
            provider_valid_at_utc
        );

        CREATE INDEX IF NOT EXISTS ix_bronze_source_records_current
        ON bronze.source_records(source_name, provider_name, is_current);

        CREATE INDEX IF NOT EXISTS ix_bronze_source_records_payload_hash
        ON bronze.source_records(payload_hash);

        CREATE UNIQUE INDEX IF NOT EXISTS ux_bronze_source_records_one_current
        ON bronze.source_records(
            source_name,
            provider_name,
            endpoint_name,
            provider_record_key
        )
        WHERE is_current = true;

        CREATE TABLE IF NOT EXISTS silver.normalized_facts (
            normalized_fact_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            source_name text NOT NULL,
            source_product text,
            source_model text,
            source_member text,
            source_cycle text,
            run_time_utc timestamptz,
            valid_time_utc timestamptz,
            forecast_hour double precision,
            station_id text REFERENCES registry.stations(station_id),
            provider_station_id text,
            grid_point_id text,
            lat double precision,
            lon double precision,
            variable_name text NOT NULL,
            variable_level text,
            variable_info text,
            unit_original text,
            value_original double precision,
            unit_canonical text,
            value_canonical double precision,
            retrieved_at_utc timestamptz NOT NULL,
            provider_available_at_utc timestamptz,
            our_ingested_at_utc timestamptz NOT NULL,
            availability_method text NOT NULL,
            source_request_id text REFERENCES bronze.source_requests(source_request_id),
            source_record_id uuid REFERENCES bronze.source_records(source_record_id),
            raw_row_hash text NOT NULL,
            quality_flag text NOT NULL DEFAULT 'ok',
            quality_note text,
            created_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT uq_normalized_facts_raw_row_hash UNIQUE (raw_row_hash)
        );

        CREATE INDEX IF NOT EXISTS ix_normalized_facts_source_time
        ON silver.normalized_facts(source_name, valid_time_utc, run_time_utc);

        CREATE INDEX IF NOT EXISTS ix_normalized_facts_station_variable
        ON silver.normalized_facts(station_id, variable_name, valid_time_utc);

        CREATE TABLE IF NOT EXISTS silver.availability_ledger (
            availability_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            source_record_id uuid REFERENCES bronze.source_records(source_record_id),
            source_name text NOT NULL,
            provider_name text NOT NULL,
            canonical_record_key text NOT NULL,
            station_id text REFERENCES registry.stations(station_id),
            model_name text,
            run_time_utc timestamptz,
            valid_time_utc timestamptz,
            forecast_hour integer,
            member text,
            variable_name text NOT NULL,
            provider_available_at_utc timestamptz NOT NULL,
            acquired_at_utc timestamptz NOT NULL,
            effective_available_at_utc timestamptz NOT NULL,
            availability_method text NOT NULL,
            source_lag_seconds integer,
            is_revision_current boolean NOT NULL DEFAULT true,
            created_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT ck_availability_method CHECK (
                availability_method IN (
                    'observed_provider_timestamp',
                    'observed_ingest_timestamp',
                    'conservative_lag_rule',
                    'manual_override'
                )
            )
        );

        CREATE UNIQUE INDEX IF NOT EXISTS ux_availability_ledger_identity
        ON silver.availability_ledger (
            source_name,
            provider_name,
            canonical_record_key,
            variable_name,
            COALESCE(member, ''),
            COALESCE(model_name, ''),
            COALESCE(station_id, ''),
            COALESCE(run_time_utc, '1900-01-01'::timestamptz),
            COALESCE(valid_time_utc, '1900-01-01'::timestamptz)
        );

        CREATE INDEX IF NOT EXISTS ix_availability_ledger_effective_available
        ON silver.availability_ledger(effective_available_at_utc);

        CREATE TABLE IF NOT EXISTS silver.target_daily_actuals (
            target_daily_actual_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            target_date date NOT NULL,
            station_id text NOT NULL REFERENCES registry.stations(station_id),
            source_name text NOT NULL,
            high_temp_f integer NOT NULL,
            low_temp_f integer,
            source_available_at_utc timestamptz NOT NULL,
            source_record_id uuid REFERENCES bronze.source_records(source_record_id),
            revision_number integer NOT NULL DEFAULT 1,
            is_current boolean NOT NULL DEFAULT true,
            created_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT ck_target_daily_actuals_station CHECK (station_id = 'KLGA'),
            CONSTRAINT ck_target_daily_actuals_high CHECK (high_temp_f BETWEEN -80 AND 140),
            CONSTRAINT uq_target_daily_actuals_revision UNIQUE (
                target_date,
                station_id,
                source_name,
                revision_number
            )
        );

        CREATE UNIQUE INDEX IF NOT EXISTS ux_target_daily_actuals_one_current
        ON silver.target_daily_actuals(target_date, station_id, source_name)
        WHERE is_current = true;

        CREATE TABLE IF NOT EXISTS gold.target_instances (
            target_instance_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            target_date date NOT NULL,
            cutoff_id text NOT NULL REFERENCES registry.cutoffs(cutoff_id),
            cutoff_utc timestamptz NOT NULL,
            target_station_id text NOT NULL DEFAULT 'KLGA' REFERENCES registry.stations(station_id),
            local_day_start_utc timestamptz NOT NULL,
            local_day_end_utc timestamptz NOT NULL,
            settlement_high_f_whole integer,
            settlement_high_available_at_utc timestamptz,
            label_available boolean NOT NULL DEFAULT false,
            label_revision_sensitive boolean NOT NULL DEFAULT true,
            created_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT uq_target_instances_date_cutoff UNIQUE (target_date, cutoff_id)
        );

        CREATE INDEX IF NOT EXISTS ix_target_instances_cutoff_utc
        ON gold.target_instances(cutoff_utc);

        CREATE TABLE IF NOT EXISTS gold.feature_values (
            feature_value_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            target_instance_id uuid NOT NULL REFERENCES gold.target_instances(target_instance_id)
                ON DELETE CASCADE,
            feature_family text NOT NULL,
            feature_name text NOT NULL,
            feature_value double precision,
            feature_unit text,
            feature_available boolean NOT NULL,
            source_latest_valid_time_utc timestamptz,
            source_latest_run_time_utc timestamptz,
            source_age_hours double precision,
            source_latency_minutes double precision,
            feature_build_version text NOT NULL,
            max_source_available_at_utc timestamptz,
            source_trace_json jsonb NOT NULL DEFAULT '{}'::jsonb,
            created_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT uq_feature_values_identity UNIQUE (
                target_instance_id,
                feature_name,
                feature_build_version
            )
        );

        CREATE INDEX IF NOT EXISTS ix_feature_values_name_available
        ON gold.feature_values(feature_name, feature_available);

        CREATE TABLE IF NOT EXISTS gold.feature_matrix (
            feature_matrix_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            target_instance_id uuid NOT NULL REFERENCES gold.target_instances(target_instance_id)
                ON DELETE CASCADE,
            feature_version_id uuid NOT NULL REFERENCES registry.feature_versions(feature_version_id),
            feature_vector_json jsonb NOT NULL DEFAULT '{}'::jsonb,
            feature_availability_json jsonb NOT NULL DEFAULT '{}'::jsonb,
            label_high_temp_f integer,
            label_available boolean NOT NULL DEFAULT false,
            label_revision_sensitive boolean NOT NULL DEFAULT true,
            created_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT uq_feature_matrix_identity UNIQUE (
                target_instance_id,
                feature_version_id
            )
        );

        CREATE TABLE IF NOT EXISTS reports.backtest_runs (
            backtest_run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            run_name text NOT NULL,
            run_id_text text NOT NULL UNIQUE,
            started_at timestamptz NOT NULL DEFAULT now(),
            finished_at timestamptz,
            status text NOT NULL,
            start_date date NOT NULL,
            end_date date NOT NULL,
            cutoff_id text,
            market_mode text NOT NULL,
            frozen_config_uri text,
            frozen_config_hash text,
            model_version_id uuid REFERENCES registry.model_versions(model_version_id),
            calibration_version_id uuid REFERENCES registry.model_versions(model_version_id),
            feature_version_id uuid REFERENCES registry.feature_versions(feature_version_id),
            source_code_git_sha text NOT NULL,
            metrics_json jsonb NOT NULL DEFAULT '{}'::jsonb,
            artifact_root_uri text,
            error_message text,
            CONSTRAINT ck_backtest_status CHECK (
                status IN ('started','success','failed','skipped')
            ),
            CONSTRAINT ck_backtest_market_mode CHECK (
                market_mode IN ('synthetic','historical_polymarket')
            )
        );

        CREATE TABLE IF NOT EXISTS reports.metrics (
            metric_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            metric_group text NOT NULL,
            metric_name text NOT NULL,
            metric_value double precision,
            metric_text text,
            metric_json jsonb NOT NULL DEFAULT '{}'::jsonb,
            target_date date,
            cutoff_id text,
            backtest_run_id uuid REFERENCES reports.backtest_runs(backtest_run_id),
            model_version_id uuid REFERENCES registry.model_versions(model_version_id),
            feature_version_id uuid REFERENCES registry.feature_versions(feature_version_id),
            created_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT ck_metrics_value_present CHECK (
                ((metric_value IS NOT NULL)::int +
                 (metric_text IS NOT NULL)::int +
                 (metric_json <> '{}'::jsonb)::int) >= 1
            )
        );

        CREATE INDEX IF NOT EXISTS ix_reports_metrics_group_name
        ON reports.metrics(metric_group, metric_name);

        CREATE INDEX IF NOT EXISTS ix_reports_metrics_backtest
        ON reports.metrics(backtest_run_id);

        CREATE INDEX IF NOT EXISTS ix_reports_metrics_target_cutoff
        ON reports.metrics(target_date, cutoff_id);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE IF EXISTS reports.metrics CASCADE;
        DROP TABLE IF EXISTS reports.backtest_runs CASCADE;
        DROP TABLE IF EXISTS gold.feature_matrix CASCADE;
        DROP TABLE IF EXISTS gold.feature_values CASCADE;
        DROP TABLE IF EXISTS gold.target_instances CASCADE;
        DROP TABLE IF EXISTS silver.target_daily_actuals CASCADE;
        DROP TABLE IF EXISTS silver.availability_ledger CASCADE;
        DROP TABLE IF EXISTS silver.normalized_facts CASCADE;
        DROP TABLE IF EXISTS bronze.source_records CASCADE;
        DROP TABLE IF EXISTS bronze.source_requests CASCADE;
        DROP TABLE IF EXISTS audit.data_quality_failures CASCADE;
        DROP TABLE IF EXISTS audit.ingestion_manifests CASCADE;
        DROP TABLE IF EXISTS audit.pipeline_runs CASCADE;
        DROP TABLE IF EXISTS registry.model_versions CASCADE;
        DROP TABLE IF EXISTS registry.feature_versions CASCADE;
        DROP TABLE IF EXISTS registry.cutoffs CASCADE;
        DROP TABLE IF EXISTS registry.stations CASCADE;
        """
    )
