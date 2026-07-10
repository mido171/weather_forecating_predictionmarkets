from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection

from klga_tmax.constants import FEATURE_SET_NAME, FEATURE_VERSION, REQUIRED_SCHEMAS
from klga_tmax.registry.cutoffs import CANONICAL_CUTOFFS
from klga_tmax.registry.station_universe import (
    CANONICAL_STATION_REGISTRY,
    STATION_REGISTRY_VERSION,
)


REQUIRED_TABLES: dict[str, tuple[str, ...]] = {
    "public": (
        "wunderground_daily_tmax",
    ),
    "registry": (
        "stations",
        "station_registry",
        "cutoffs",
        "feature_versions",
        "model_versions",
    ),
    "audit": (
        "pipeline_runs",
        "ingestion_manifests",
        "data_quality_failures",
        "gribstream_catalog_snapshots",
        "gribstream_backfill_jobs",
        "gribstream_backfill_chunks",
        "gribstream_source_gaps",
        "iem_mos_backfill_jobs",
        "iem_mos_backfill_chunks",
        "iem_mos_source_gaps",
    ),
    "bronze": ("source_requests", "source_records"),
    "silver": (
        "normalized_facts",
        "availability_ledger",
        "target_daily_actuals",
        "station_daily_actuals",
        "station_observations",
        "mos_guidance",
        "grib_forecast_values",
        "iem_mos_forecast_rows",
    ),
    "gold": (
        "target_instances",
        "feature_values",
        "feature_matrix",
        "iem_mos_daily_features",
        "iem_mos_feature_matrix_v1",
    ),
    "predictions": (
        "expert_predictions",
        "final_predictions",
        "calibration_versions",
        "calibrated_predictions",
    ),
    "reports": (
        "backtest_runs",
        "forecast_evaluation_runs",
        "forecast_evaluation_daily_scores",
        "metrics",
    ),
}

REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "public.wunderground_daily_tmax": (
        "station_id",
        "wunderground_station_id",
        "local_date",
        "timezone_name",
        "tmax_f",
        "tmin_f",
        "observation_count",
        "high_observation_times_local_json",
        "hourly_observations_json",
        "provider_max_temp_values_json",
        "provider_min_temp_values_json",
        "source_url_redacted",
        "wu_page_url",
        "payload_hash",
        "parser_version",
        "fetched_at_utc",
        "settlement_available_at_utc",
        "daily_high_source",
        "validation_status",
        "validation_notes_json",
    ),
    "registry.stations": (
        "station_id",
        "station_name",
        "provider_primary_id",
        "latitude",
        "longitude",
        "station_role",
        "station_group",
    ),
    "registry.station_registry": (
        "station_registry_version",
        "station_id",
        "iem_asos_id",
        "wunderground_station_id",
        "mos_station_id",
        "grid_point_id",
        "role",
        "lat",
        "lon",
        "active_from_date",
        "active_until_date",
    ),
    "registry.cutoffs": (
        "cutoff_id",
        "cutoff_order",
        "timezone_name",
        "local_time",
        "target_day_offset",
    ),
    "registry.feature_versions": (
        "feature_version_id",
        "feature_set_name",
        "feature_version",
        "formula_contract_hash",
    ),
    "audit.pipeline_runs": (
        "pipeline_run_id",
        "command_name",
        "status",
        "exit_code",
        "source_code_git_sha",
    ),
    "audit.ingestion_manifests": (
        "job_id",
        "source_name",
        "config_hash",
        "row_counts_bronze",
        "row_counts_silver",
        "row_counts_gold",
    ),
    "audit.data_quality_failures": (
        "data_quality_failure_id",
        "table_name",
        "check_name",
        "severity",
        "message",
    ),
    "audit.gribstream_catalog_snapshots": (
        "gribstream_catalog_snapshot_id",
        "model_id",
        "catalog_kind",
        "catalog_url",
        "payload_sha256",
        "payload_json",
        "retrieved_at_utc",
    ),
    "audit.gribstream_backfill_jobs": (
        "job_id",
        "cutoff_id",
        "start_date",
        "end_date",
        "coordinate_tier",
        "status",
        "planned_chunks",
        "estimated_credits",
        "config_json",
    ),
    "audit.gribstream_backfill_chunks": (
        "chunk_id",
        "job_id",
        "model_id",
        "target_start_date",
        "target_end_date",
        "cutoff_id",
        "endpoint_type",
        "coordinate_tier",
        "request_sha256",
        "request_json",
        "status",
    ),
    "audit.gribstream_source_gaps": (
        "gribstream_source_gap_id",
        "model_id",
        "gap_type",
        "gap_reason",
        "evidence_json",
        "last_detected_at_utc",
    ),
    "audit.iem_mos_backfill_jobs": (
        "job_id",
        "cutoff_id",
        "start_date",
        "end_date",
        "status",
        "planned_chunks",
        "completed_chunks",
        "rows_upserted",
        "feature_rows_upserted",
        "config_json",
    ),
    "audit.iem_mos_backfill_chunks": (
        "chunk_id",
        "job_id",
        "station_id",
        "mos_station_id",
        "source_product",
        "endpoint_model",
        "cutoff_id",
        "window_start_utc",
        "window_end_utc",
        "request_sha256",
        "request_json",
        "status",
    ),
    "audit.iem_mos_source_gaps": (
        "iem_mos_source_gap_id",
        "job_id",
        "station_id",
        "source_product",
        "gap_type",
        "gap_reason",
        "evidence_json",
    ),
    "bronze.source_requests": (
        "source_request_id",
        "source_name",
        "source_endpoint",
        "request_method",
        "request_params_json",
        "retrieved_at_utc",
        "response_body_sha256",
    ),
    "bronze.source_records": (
        "source_record_id",
        "source_request_id",
        "source_name",
        "provider_name",
        "provider_record_key",
        "payload_hash",
        "revision_number",
        "is_current",
    ),
    "silver.normalized_facts": (
        "normalized_fact_id",
        "source_name",
        "valid_time_utc",
        "station_id",
        "variable_name",
        "our_ingested_at_utc",
        "raw_row_hash",
    ),
    "silver.availability_ledger": (
        "availability_id",
        "source_name",
        "provider_name",
        "canonical_record_key",
        "effective_available_at_utc",
        "availability_method",
    ),
    "silver.target_daily_actuals": (
        "target_daily_actual_id",
        "target_date",
        "station_id",
        "source_name",
        "high_temp_f",
        "source_available_at_utc",
        "source_record_id",
        "revision_number",
        "is_current",
    ),
    "silver.station_daily_actuals": (
        "station_daily_actual_id",
        "target_date",
        "station_id",
        "source_name",
        "high_temp_f",
        "provider_available_at_utc",
        "effective_available_at_utc",
        "source_record_id",
        "revision_number",
        "is_current",
    ),
    "silver.station_observations": (
        "station_observation_id",
        "station_id",
        "source_name",
        "observation_time_utc",
        "local_date",
        "temp_f",
        "provider_available_at_utc",
        "effective_available_at_utc",
        "raw_row_hash",
    ),
    "silver.mos_guidance": (
        "mos_guidance_id",
        "station_id",
        "source_product",
        "endpoint_model",
        "cutoff_id",
        "run_time_utc",
        "forecast_valid_time_utc",
        "target_date",
        "raw_values_jsonb",
        "provider_available_at_utc",
        "effective_available_at_utc",
        "raw_row_hash",
    ),
    "silver.grib_forecast_values": (
        "grib_value_id",
        "model_id",
        "endpoint_type",
        "target_date",
        "cutoff_id",
        "grid_point_id",
        "forecasted_at_utc",
        "forecasted_time_utc",
        "member",
        "variable_alias",
        "variable_name",
        "source_request_id",
        "source_record_id",
        "request_sha256",
        "raw_row_hash",
        "availability_method",
    ),
    "silver.iem_mos_forecast_rows": (
        "iem_mos_forecast_row_id",
        "station_id",
        "mos_station_id",
        "source_product",
        "endpoint_model",
        "cutoff_id",
        "run_time_utc",
        "forecast_valid_time_utc",
        "raw_values_jsonb",
        "provider_available_at_utc",
        "effective_available_at_utc",
        "availability_method",
        "source_request_id",
        "source_record_id",
        "request_sha256",
        "raw_row_hash",
        "parser_version",
    ),
    "gold.target_instances": (
        "target_instance_id",
        "target_date",
        "cutoff_id",
        "cutoff_utc",
        "local_day_start_utc",
        "local_day_end_utc",
    ),
    "gold.feature_values": (
        "feature_value_id",
        "target_instance_id",
        "feature_name",
        "feature_available",
        "max_source_available_at_utc",
        "source_trace_json",
    ),
    "gold.feature_matrix": (
        "feature_matrix_id",
        "target_instance_id",
        "feature_version_id",
        "feature_vector_json",
        "feature_availability_json",
    ),
    "gold.iem_mos_daily_features": (
        "iem_mos_daily_feature_id",
        "target_date",
        "cutoff_id",
        "target_instance_id",
        "station_id",
        "source_product",
        "chosen_run_time_utc",
        "max_source_available_at_utc",
        "tmax_f",
        "tmp_peak_window_max_f",
        "source_trace_json",
        "feature_build_version",
    ),
    "gold.iem_mos_feature_matrix_v1": (
        "target_instance_id",
        "target_date",
        "cutoff_id",
        "feature_vector_json",
        "feature_trace_json",
        "source_feature_count",
    ),
    "predictions.expert_predictions": (
        "expert_prediction_id",
        "target_instance_id",
        "expert_name",
        "prediction_kind",
        "model_version_id",
        "pmf_json",
        "expected_tmax_f",
        "median_tmax_f",
        "mode_tmax_f",
        "prediction_interval_low_f",
        "prediction_interval_high_f",
        "feature_hash",
        "source_availability_json",
    ),
    "predictions.final_predictions": (
        "final_prediction_id",
        "target_instance_id",
        "prediction_kind",
        "model_version_id",
        "expert_weights_json",
        "pmf_json",
        "expected_tmax_f",
        "median_tmax_f",
        "mode_tmax_f",
        "prediction_interval_low_f",
        "prediction_interval_high_f",
        "entropy",
    ),
    "predictions.calibration_versions": (
        "calibration_version_id",
        "calibration_name",
        "prediction_kind",
        "model_version_id",
        "training_start_date",
        "training_end_date",
        "method",
        "config_json",
        "metrics_json",
    ),
    "predictions.calibrated_predictions": (
        "calibrated_prediction_id",
        "final_prediction_id",
        "calibration_version_id",
        "pmf_json",
        "expected_tmax_f",
        "median_tmax_f",
        "mode_tmax_f",
        "prediction_interval_low_f",
        "prediction_interval_high_f",
    ),
    "reports.forecast_evaluation_runs": (
        "evaluation_run_id",
        "run_id_text",
        "prediction_kind",
        "status",
        "start_date",
        "end_date",
        "cutoff_id",
        "metrics_json",
    ),
    "reports.forecast_evaluation_daily_scores": (
        "evaluation_daily_score_id",
        "evaluation_run_id",
        "target_date",
        "cutoff_id",
        "prediction_id",
        "settled_wu_tmax_f",
        "expected_tmax_f",
        "absolute_error_f",
        "signed_error_f",
        "pmf_probability_at_observed",
        "label_available_at_utc",
        "leakage_checked",
    ),
}

REQUIRED_INDEXES = {
    "ux_availability_ledger_identity",
    "ux_target_daily_actuals_one_current",
    "ux_model_versions_identity",
    "ux_bronze_source_records_one_current",
    "ix_station_registry_role",
    "wunderground_daily_tmax_pkey",
    "ix_wunderground_daily_tmax_station_date",
    "ix_wunderground_daily_tmax_status",
    "ix_wunderground_daily_tmax_available",
    "ux_gribstream_catalog_snapshot",
    "ix_gribstream_catalog_model",
    "ix_gribstream_jobs_status",
    "ix_gribstream_chunks_request_sha",
    "ux_gribstream_chunks_job_request",
    "ix_gribstream_chunks_job_status",
    "ix_gribstream_chunks_model_dates",
    "ux_gribstream_source_gap_identity",
    "ix_gribstream_source_gaps_model",
    "uq_grib_forecast_values_raw_hash",
    "ix_grib_values_model_target",
    "ix_grib_values_valid_time",
    "ix_grib_values_request",
    "ix_grib_values_coordinate_variable",
    "ix_iem_mos_jobs_status",
    "ux_iem_mos_chunks_job_request",
    "ix_iem_mos_chunks_job_status",
    "ix_iem_mos_chunks_station_product_window",
    "ux_iem_mos_source_gap_identity",
    "ix_iem_mos_source_gaps_product",
    "uq_iem_mos_forecast_raw_hash",
    "ix_iem_mos_forecast_station_product_runtime",
    "ix_iem_mos_forecast_station_product_valid",
    "ix_iem_mos_forecast_valid",
    "ix_iem_mos_forecast_available",
    "ix_iem_mos_forecast_request",
    "ix_iem_mos_daily_features_target",
    "ix_iem_mos_daily_features_station_product",
    "ix_iem_mos_feature_matrix_date_cutoff",
    "ux_station_daily_actuals_current",
    "ux_station_observations_identity",
    "ux_mos_guidance_identity",
    "ix_mos_guidance_target_cutoff",
    "ux_expert_predictions_identity",
    "ux_final_predictions_identity",
    "ux_calibration_versions_identity",
    "ux_calibrated_predictions_identity",
    "ux_forecast_eval_daily_identity",
}

FORBIDDEN_WU_TABLES: dict[str, tuple[str, ...]] = {
    "silver": (
        "wu_daily_actuals",
        "wu_intraday_observations",
        "wu_daily_actual_revisions",
    ),
    "audit": (
        "wu_fetch_windows",
        "wu_station_date_coverage",
    ),
}

REQUIRED_VIEWS: dict[str, tuple[str, ...]] = {
    "gold": ("v_feature_matrix_flat",),
    "predictions": ("v_final_prediction_daily",),
    "reports": ("v_forecast_accuracy_daily_scores",),
}


@dataclass
class ContractInspection:
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, int | str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failures


def inspect_contract(connection: Connection) -> ContractInspection:
    result = ContractInspection()
    inspector = inspect(connection)

    extension_count = connection.execute(
        text("SELECT count(*) FROM pg_extension WHERE extname = 'pgcrypto'")
    ).scalar_one()
    if extension_count != 1:
        result.failures.append("missing extension pgcrypto")

    available_schemas = set(inspector.get_schema_names())
    for schema in REQUIRED_SCHEMAS:
        if schema not in available_schemas:
            result.failures.append(f"missing schema {schema}")

    for schema, tables in REQUIRED_TABLES.items():
        schema_tables = set(inspector.get_table_names(schema=schema))
        for table_name in tables:
            if table_name not in schema_tables:
                result.failures.append(f"missing table {schema}.{table_name}")

    for schema, views in REQUIRED_VIEWS.items():
        schema_views = set(inspector.get_view_names(schema=schema))
        for view_name in views:
            if view_name not in schema_views:
                result.failures.append(f"missing view {schema}.{view_name}")

    for qualified_table, required_columns in REQUIRED_COLUMNS.items():
        schema, table_name = qualified_table.split(".", 1)
        if f"missing table {qualified_table}" in result.failures:
            continue
        column_names = {
            column["name"] for column in inspector.get_columns(table_name, schema=schema)
        }
        for column_name in required_columns:
            if column_name not in column_names:
                result.failures.append(f"missing column {qualified_table}.{column_name}")

    index_rows = connection.execute(
        text(
            """
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE schemaname IN ('public','registry','bronze','silver','gold','predictions','reports','audit')
            """
        )
    ).mappings()
    indexes = {row["indexname"]: row["indexdef"] for row in index_rows}
    for index_name in REQUIRED_INDEXES:
        if index_name not in indexes:
            result.failures.append(f"missing index {index_name}")
    availability_index = indexes.get("ux_availability_ledger_identity", "")
    if "COALESCE" not in availability_index.upper():
        result.failures.append("ux_availability_ledger_identity is not an expression index")

    for schema, tables in FORBIDDEN_WU_TABLES.items():
        schema_tables = set(inspector.get_table_names(schema=schema))
        for table_name in tables:
            if table_name in schema_tables:
                result.failures.append(f"legacy WU table still exists: {schema}.{table_name}")

    cutoff_count = connection.execute(text("SELECT count(*) FROM registry.cutoffs")).scalar_one()
    station_count = connection.execute(
        text("SELECT count(*) FROM registry.stations WHERE active = true")
    ).scalar_one()
    station_registry_count = connection.execute(
        text(
            """
            SELECT count(*)
            FROM registry.station_registry
            WHERE station_registry_version = :station_registry_version
            """
        ),
        {"station_registry_version": STATION_REGISTRY_VERSION},
    ).scalar_one()
    feature_count = connection.execute(
        text(
            """
            SELECT count(*)
            FROM registry.feature_versions
            WHERE feature_set_name = :feature_set_name
              AND feature_version = :feature_version
            """
        ),
        {"feature_set_name": FEATURE_SET_NAME, "feature_version": FEATURE_VERSION},
    ).scalar_one()
    if cutoff_count < len(CANONICAL_CUTOFFS):
        result.failures.append(
            f"registry.cutoffs has {cutoff_count} rows; expected at least {len(CANONICAL_CUTOFFS)}"
        )
    if station_registry_count != len(CANONICAL_STATION_REGISTRY):
        result.failures.append(
            f"registry.station_registry has {station_registry_count} rows for "
            f"{STATION_REGISTRY_VERSION}; expected {len(CANONICAL_STATION_REGISTRY)}"
        )
    if station_count != len(CANONICAL_STATION_REGISTRY):
        result.failures.append(
            f"registry.stations has {station_count} active rows; "
            f"expected {len(CANONICAL_STATION_REGISTRY)}"
        )
    if feature_count != 1:
        result.failures.append(
            f"missing feature version {FEATURE_SET_NAME}/{FEATURE_VERSION}"
        )

    result.details.update(
        {
            "schemas_checked": len(REQUIRED_SCHEMAS),
            "tables_checked": sum(len(tables) for tables in REQUIRED_TABLES.values()),
            "views_checked": sum(len(views) for views in REQUIRED_VIEWS.values()),
            "indexes_checked": len(REQUIRED_INDEXES),
            "cutoff_rows": int(cutoff_count),
            "station_rows": int(station_count),
            "station_registry_rows": int(station_registry_count),
            "feature_version_rows": int(feature_count),
        }
    )
    return result
