"""Add local Tmax forecast and WU-settlement evaluation tables.

Revision ID: 0009_forecast_eval
Revises: 0008_iem_mos_backfill
Create Date: 2026-07-02 12:00:00
"""

from __future__ import annotations

from alembic import op

revision = "0009_forecast_eval"
down_revision = "0008_iem_mos_backfill"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS silver.station_daily_actuals (
            station_daily_actual_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            target_date date NOT NULL,
            station_id text NOT NULL REFERENCES registry.stations(station_id),
            source_name text NOT NULL,
            high_temp_f integer,
            low_temp_f integer,
            avg_temp_f double precision,
            precip_in double precision,
            max_wind_speed_mph double precision,
            max_wind_gust_mph double precision,
            provider_available_at_utc timestamptz NOT NULL,
            effective_available_at_utc timestamptz NOT NULL,
            source_request_id text REFERENCES bronze.source_requests(source_request_id),
            source_record_id uuid REFERENCES bronze.source_records(source_record_id),
            revision_number integer NOT NULL DEFAULT 1,
            is_current boolean NOT NULL DEFAULT true,
            quality_flag text NOT NULL DEFAULT 'ok',
            source_trace_json jsonb NOT NULL DEFAULT '{}'::jsonb,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT uq_station_daily_actuals_revision UNIQUE (
                target_date,
                station_id,
                source_name,
                revision_number
            )
        );

        CREATE UNIQUE INDEX IF NOT EXISTS ux_station_daily_actuals_current
        ON silver.station_daily_actuals(target_date, station_id, source_name)
        WHERE is_current = true;

        CREATE INDEX IF NOT EXISTS ix_station_daily_actuals_available
        ON silver.station_daily_actuals(effective_available_at_utc);

        CREATE TABLE IF NOT EXISTS silver.station_observations (
            station_observation_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            station_id text NOT NULL REFERENCES registry.stations(station_id),
            source_name text NOT NULL,
            observation_time_utc timestamptz NOT NULL,
            local_date date NOT NULL,
            temp_f double precision,
            dewpoint_f double precision,
            humidity_pct double precision,
            wind_speed_mph double precision,
            wind_gust_mph double precision,
            wind_direction_deg double precision,
            pressure_in double precision,
            precipitation_in double precision,
            condition_text text,
            provider_available_at_utc timestamptz NOT NULL,
            effective_available_at_utc timestamptz NOT NULL,
            source_request_id text REFERENCES bronze.source_requests(source_request_id),
            source_record_id uuid REFERENCES bronze.source_records(source_record_id),
            raw_row_hash text NOT NULL,
            quality_flag text NOT NULL DEFAULT 'ok',
            source_trace_json jsonb NOT NULL DEFAULT '{}'::jsonb,
            created_at timestamptz NOT NULL DEFAULT now()
        );

        CREATE UNIQUE INDEX IF NOT EXISTS ux_station_observations_identity
        ON silver.station_observations (
            station_id,
            source_name,
            observation_time_utc,
            COALESCE(source_record_id, '00000000-0000-0000-0000-000000000000'::uuid),
            md5(raw_row_hash)
        );

        CREATE INDEX IF NOT EXISTS ix_station_observations_station_time
        ON silver.station_observations(station_id, observation_time_utc);

        CREATE INDEX IF NOT EXISTS ix_station_observations_available
        ON silver.station_observations(effective_available_at_utc);

        CREATE TABLE IF NOT EXISTS silver.mos_guidance (
            mos_guidance_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            station_id text NOT NULL REFERENCES registry.stations(station_id),
            mos_station_id text NOT NULL,
            source_product text NOT NULL,
            endpoint_model text NOT NULL,
            cutoff_id text NOT NULL REFERENCES registry.cutoffs(cutoff_id),
            run_time_utc timestamptz NOT NULL,
            forecast_valid_time_utc timestamptz NOT NULL,
            target_date date NOT NULL,
            raw_values_jsonb jsonb NOT NULL DEFAULT '{}'::jsonb,
            tmax_f double precision,
            tmp_f double precision,
            dpt_f double precision,
            wsp_kt double precision,
            pop double precision,
            qpf double precision,
            tstm_prob double precision,
            provider_available_at_utc timestamptz NOT NULL,
            effective_available_at_utc timestamptz NOT NULL,
            availability_method text NOT NULL,
            source_request_id text REFERENCES bronze.source_requests(source_request_id),
            source_record_id uuid REFERENCES bronze.source_records(source_record_id),
            request_sha256 text,
            raw_row_hash text NOT NULL,
            source_trace_json jsonb NOT NULL DEFAULT '{}'::jsonb,
            created_at timestamptz NOT NULL DEFAULT now()
        );

        CREATE UNIQUE INDEX IF NOT EXISTS ux_mos_guidance_identity
        ON silver.mos_guidance (
            station_id,
            source_product,
            endpoint_model,
            cutoff_id,
            run_time_utc,
            forecast_valid_time_utc,
            COALESCE(request_sha256, ''),
            md5(raw_row_hash)
        );

        CREATE INDEX IF NOT EXISTS ix_mos_guidance_target_cutoff
        ON silver.mos_guidance(target_date, cutoff_id);

        CREATE INDEX IF NOT EXISTS ix_mos_guidance_available
        ON silver.mos_guidance(effective_available_at_utc);

        CREATE TABLE IF NOT EXISTS predictions.expert_predictions (
            expert_prediction_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            target_instance_id uuid NOT NULL REFERENCES gold.target_instances(target_instance_id)
                ON DELETE CASCADE,
            expert_name text NOT NULL,
            prediction_kind text NOT NULL,
            model_version_id uuid NOT NULL REFERENCES registry.model_versions(model_version_id),
            feature_version_id uuid NOT NULL REFERENCES registry.feature_versions(feature_version_id),
            fold_id text,
            training_start_date date,
            training_end_date date,
            pmf_json jsonb NOT NULL,
            expected_tmax_f double precision NOT NULL,
            median_tmax_f integer NOT NULL,
            mode_tmax_f integer NOT NULL,
            prediction_interval_low_f integer NOT NULL,
            prediction_interval_high_f integer NOT NULL,
            uncertainty_f double precision NOT NULL,
            feature_names text[] NOT NULL DEFAULT '{}',
            feature_hash text NOT NULL,
            source_availability_json jsonb NOT NULL DEFAULT '{}'::jsonb,
            diagnostics_json jsonb NOT NULL DEFAULT '{}'::jsonb,
            prediction_status text NOT NULL DEFAULT 'ok',
            created_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT ck_expert_prediction_kind CHECK (
                prediction_kind IN ('oof','holdout','forecast','replay')
            ),
            CONSTRAINT ck_expert_prediction_status CHECK (
                prediction_status IN ('ok','fallback','disabled_data_sufficiency')
            )
        );

        CREATE UNIQUE INDEX IF NOT EXISTS ux_expert_predictions_identity
        ON predictions.expert_predictions(
            target_instance_id,
            expert_name,
            prediction_kind,
            model_version_id,
            COALESCE(fold_id, '')
        );

        CREATE TABLE IF NOT EXISTS predictions.final_predictions (
            final_prediction_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            target_instance_id uuid NOT NULL REFERENCES gold.target_instances(target_instance_id)
                ON DELETE CASCADE,
            prediction_kind text NOT NULL,
            model_version_id uuid NOT NULL REFERENCES registry.model_versions(model_version_id),
            feature_version_id uuid NOT NULL REFERENCES registry.feature_versions(feature_version_id),
            expert_prediction_ids uuid[] NOT NULL DEFAULT '{}',
            expert_weights_json jsonb NOT NULL DEFAULT '{}'::jsonb,
            pmf_json jsonb NOT NULL,
            expected_tmax_f double precision NOT NULL,
            median_tmax_f integer NOT NULL,
            mode_tmax_f integer NOT NULL,
            prediction_interval_low_f integer NOT NULL,
            prediction_interval_high_f integer NOT NULL,
            uncertainty_f double precision NOT NULL,
            entropy double precision NOT NULL,
            diagnostics_json jsonb NOT NULL DEFAULT '{}'::jsonb,
            created_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT ck_final_prediction_kind CHECK (
                prediction_kind IN ('oof','holdout','forecast','replay')
            )
        );

        CREATE UNIQUE INDEX IF NOT EXISTS ux_final_predictions_identity
        ON predictions.final_predictions(target_instance_id, prediction_kind, model_version_id);

        CREATE TABLE IF NOT EXISTS predictions.calibration_versions (
            calibration_version_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            calibration_name text NOT NULL,
            prediction_kind text NOT NULL,
            model_version_id uuid NOT NULL REFERENCES registry.model_versions(model_version_id),
            training_start_date date NOT NULL,
            training_end_date date NOT NULL,
            cutoff_id text REFERENCES registry.cutoffs(cutoff_id),
            method text NOT NULL,
            config_json jsonb NOT NULL DEFAULT '{}'::jsonb,
            metrics_json jsonb NOT NULL DEFAULT '{}'::jsonb,
            artifact_uri text,
            artifact_hash text,
            created_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT ck_calibration_kind CHECK (
                prediction_kind IN ('oof','holdout','forecast','replay')
            )
        );

        CREATE UNIQUE INDEX IF NOT EXISTS ux_calibration_versions_identity
        ON predictions.calibration_versions(
            calibration_name,
            prediction_kind,
            model_version_id,
            training_start_date,
            training_end_date,
            COALESCE(cutoff_id, ''),
            method,
            md5(config_json::text)
        );

        CREATE TABLE IF NOT EXISTS predictions.calibrated_predictions (
            calibrated_prediction_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            final_prediction_id uuid NOT NULL REFERENCES predictions.final_predictions(final_prediction_id)
                ON DELETE CASCADE,
            calibration_version_id uuid NOT NULL REFERENCES predictions.calibration_versions(calibration_version_id),
            pmf_json jsonb NOT NULL,
            expected_tmax_f double precision NOT NULL,
            median_tmax_f integer NOT NULL,
            mode_tmax_f integer NOT NULL,
            prediction_interval_low_f integer NOT NULL,
            prediction_interval_high_f integer NOT NULL,
            uncertainty_f double precision NOT NULL,
            diagnostics_json jsonb NOT NULL DEFAULT '{}'::jsonb,
            created_at timestamptz NOT NULL DEFAULT now()
        );

        CREATE UNIQUE INDEX IF NOT EXISTS ux_calibrated_predictions_identity
        ON predictions.calibrated_predictions(final_prediction_id, calibration_version_id);

        CREATE TABLE IF NOT EXISTS reports.forecast_evaluation_runs (
            evaluation_run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            run_id_text text NOT NULL UNIQUE,
            run_name text NOT NULL,
            prediction_kind text NOT NULL,
            started_at timestamptz NOT NULL DEFAULT now(),
            finished_at timestamptz,
            status text NOT NULL,
            start_date date NOT NULL,
            end_date date NOT NULL,
            cutoff_id text REFERENCES registry.cutoffs(cutoff_id),
            model_version_id uuid REFERENCES registry.model_versions(model_version_id),
            calibration_version_id uuid REFERENCES predictions.calibration_versions(calibration_version_id),
            feature_version_id uuid REFERENCES registry.feature_versions(feature_version_id),
            source_code_git_sha text NOT NULL,
            config_json jsonb NOT NULL DEFAULT '{}'::jsonb,
            metrics_json jsonb NOT NULL DEFAULT '{}'::jsonb,
            artifact_root_uri text,
            error_message text,
            CONSTRAINT ck_forecast_eval_status CHECK (
                status IN ('started','success','failed','skipped')
            ),
            CONSTRAINT ck_forecast_eval_prediction_kind CHECK (
                prediction_kind IN ('oof','holdout','forecast','replay')
            )
        );

        CREATE TABLE IF NOT EXISTS reports.forecast_evaluation_daily_scores (
            evaluation_daily_score_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            evaluation_run_id uuid NOT NULL REFERENCES reports.forecast_evaluation_runs(evaluation_run_id)
                ON DELETE CASCADE,
            target_date date NOT NULL,
            cutoff_id text NOT NULL REFERENCES registry.cutoffs(cutoff_id),
            prediction_id uuid NOT NULL,
            calibrated_prediction_id uuid REFERENCES predictions.calibrated_predictions(calibrated_prediction_id),
            settled_wu_tmax_f integer NOT NULL,
            expected_tmax_f double precision NOT NULL,
            median_tmax_f integer NOT NULL,
            mode_tmax_f integer NOT NULL,
            absolute_error_f double precision NOT NULL,
            signed_error_f double precision NOT NULL,
            squared_error_f double precision NOT NULL,
            pmf_probability_at_observed double precision NOT NULL,
            log_score double precision NOT NULL,
            within_1f boolean NOT NULL,
            within_2f boolean NOT NULL,
            prediction_interval_low_f integer NOT NULL,
            prediction_interval_high_f integer NOT NULL,
            prediction_interval_hit boolean NOT NULL,
            label_source_record_id uuid REFERENCES bronze.source_records(source_record_id),
            label_revision_number integer NOT NULL,
            label_available_at_utc timestamptz NOT NULL,
            leakage_checked boolean NOT NULL DEFAULT true,
            diagnostics_json jsonb NOT NULL DEFAULT '{}'::jsonb,
            created_at timestamptz NOT NULL DEFAULT now()
        );

        CREATE UNIQUE INDEX IF NOT EXISTS ux_forecast_eval_daily_identity
        ON reports.forecast_evaluation_daily_scores(evaluation_run_id, target_date, cutoff_id, prediction_id);

        CREATE OR REPLACE VIEW gold.v_feature_matrix_flat AS
        SELECT
            fm.feature_matrix_id,
            fm.target_instance_id,
            ti.target_date,
            ti.cutoff_id,
            ti.cutoff_utc,
            fm.feature_version_id,
            fm.feature_vector_json,
            fm.feature_availability_json,
            fm.label_high_temp_f,
            fm.label_available,
            fm.label_revision_sensitive
        FROM gold.feature_matrix fm
        JOIN gold.target_instances ti
          ON ti.target_instance_id = fm.target_instance_id;

        CREATE OR REPLACE VIEW predictions.v_final_prediction_daily AS
        SELECT
            fp.final_prediction_id,
            fp.target_instance_id,
            ti.target_date,
            ti.cutoff_id,
            ti.cutoff_utc,
            fp.prediction_kind,
            fp.model_version_id,
            fp.feature_version_id,
            fp.expected_tmax_f,
            fp.median_tmax_f,
            fp.mode_tmax_f,
            fp.prediction_interval_low_f,
            fp.prediction_interval_high_f,
            fp.uncertainty_f,
            fp.entropy,
            fp.created_at
        FROM predictions.final_predictions fp
        JOIN gold.target_instances ti
          ON ti.target_instance_id = fp.target_instance_id;

        CREATE OR REPLACE VIEW reports.v_forecast_accuracy_daily_scores AS
        SELECT
            r.run_id_text,
            r.prediction_kind,
            r.status AS evaluation_status,
            s.target_date,
            s.cutoff_id,
            s.settled_wu_tmax_f,
            s.expected_tmax_f,
            s.median_tmax_f,
            s.mode_tmax_f,
            s.absolute_error_f,
            s.signed_error_f,
            s.pmf_probability_at_observed,
            s.log_score,
            s.within_1f,
            s.within_2f,
            s.prediction_interval_hit
        FROM reports.forecast_evaluation_daily_scores s
        JOIN reports.forecast_evaluation_runs r
          ON r.evaluation_run_id = s.evaluation_run_id;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP VIEW IF EXISTS reports.v_forecast_accuracy_daily_scores;
        DROP VIEW IF EXISTS predictions.v_final_prediction_daily;
        DROP VIEW IF EXISTS gold.v_feature_matrix_flat;
        DROP TABLE IF EXISTS reports.forecast_evaluation_daily_scores CASCADE;
        DROP TABLE IF EXISTS reports.forecast_evaluation_runs CASCADE;
        DROP TABLE IF EXISTS predictions.calibrated_predictions CASCADE;
        DROP TABLE IF EXISTS predictions.calibration_versions CASCADE;
        DROP TABLE IF EXISTS predictions.final_predictions CASCADE;
        DROP TABLE IF EXISTS predictions.expert_predictions CASCADE;
        DROP TABLE IF EXISTS silver.mos_guidance CASCADE;
        DROP TABLE IF EXISTS silver.station_observations CASCADE;
        DROP TABLE IF EXISTS silver.station_daily_actuals CASCADE;
        """
    )
