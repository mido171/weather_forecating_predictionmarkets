from __future__ import annotations

from datetime import date, datetime, time
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Station(Base):
    __tablename__ = "stations"
    __table_args__ = (
        CheckConstraint("latitude BETWEEN -90 AND 90", name="ck_stations_latitude"),
        CheckConstraint("longitude BETWEEN -180 AND 180", name="ck_stations_longitude"),
        CheckConstraint(
            "station_role IN ('target','nearby_core','regional_context','gridded_pseudo_point')",
            name="ck_stations_role",
        ),
        {"schema": "registry"},
    )

    station_id: Mapped[str] = mapped_column(Text, primary_key=True)
    station_name: Mapped[str] = mapped_column(Text, nullable=False)
    provider_primary_id: Mapped[str] = mapped_column(Text, nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    elevation_m: Mapped[float | None] = mapped_column(Float)
    timezone: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'America/New_York'")
    )
    station_role: Mapped[str] = mapped_column(Text, nullable=False)
    station_group: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default=text("'{}'::text[]")
    )
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class StationRegistry(Base):
    __tablename__ = "station_registry"
    __table_args__ = (
        CheckConstraint(
            "role IN ('target','nearby_core','regional_context','gridded_pseudo_point')",
            name="ck_station_registry_role",
        ),
        CheckConstraint("lat BETWEEN -90 AND 90", name="ck_station_registry_lat"),
        CheckConstraint("lon BETWEEN -180 AND 180", name="ck_station_registry_lon"),
        CheckConstraint(
            "active_until_date IS NULL OR active_until_date >= active_from_date",
            name="ck_station_registry_active_dates",
        ),
        {"schema": "registry"},
    )

    station_registry_version: Mapped[str] = mapped_column(Text, primary_key=True)
    station_id: Mapped[str] = mapped_column(Text, primary_key=True)
    iem_asos_id: Mapped[str | None] = mapped_column(Text)
    wunderground_station_id: Mapped[str | None] = mapped_column(Text)
    mos_station_id: Mapped[str | None] = mapped_column(Text)
    grid_point_id: Mapped[str] = mapped_column(Text, primary_key=True, server_default=text("''"))
    role: Mapped[str] = mapped_column(Text, nullable=False)
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    lon: Mapped[float] = mapped_column(Float, nullable=False)
    elevation_m: Mapped[float | None] = mapped_column(Float)
    source_native_metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    active_from_date: Mapped[date] = mapped_column(
        Date, nullable=False, server_default=text("'1900-01-01'")
    )
    active_until_date: Mapped[date | None] = mapped_column(Date)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class Cutoff(Base):
    __tablename__ = "cutoffs"
    __table_args__ = ({"schema": "registry"},)

    cutoff_id: Mapped[str] = mapped_column(Text, primary_key=True)
    cutoff_order: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)
    timezone_name: Mapped[str] = mapped_column(Text, nullable=False)
    local_time: Mapped[time] = mapped_column(nullable=False)
    target_day_offset: Mapped[int] = mapped_column(Integer, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))


class FeatureVersion(Base):
    __tablename__ = "feature_versions"
    __table_args__ = (
        UniqueConstraint("feature_set_name", "feature_version", name="uq_feature_versions_name"),
        {"schema": "registry"},
    )

    feature_version_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    feature_set_name: Mapped[str] = mapped_column(Text, nullable=False)
    feature_version: Mapped[str] = mapped_column(Text, nullable=False)
    source_code_git_sha: Mapped[str] = mapped_column(Text, nullable=False)
    formula_contract_hash: Mapped[str] = mapped_column(Text, nullable=False)
    feature_names: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (
        CheckConstraint(
            "model_role IN ('expert','meta_combiner','calibrator','simulation','report')",
            name="ck_model_versions_role",
        ),
        {"schema": "registry"},
    )

    model_version_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    model_family: Mapped[str] = mapped_column(Text, nullable=False)
    model_name: Mapped[str] = mapped_column(Text, nullable=False)
    model_role: Mapped[str] = mapped_column(Text, nullable=False)
    source_code_git_sha: Mapped[str] = mapped_column(Text, nullable=False)
    training_data_start: Mapped[date | None] = mapped_column(Date)
    training_data_end: Mapped[date | None] = mapped_column(Date)
    feature_version_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True))
    hyperparams: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    artifact_uri: Mapped[str | None] = mapped_column(Text)
    artifact_hash: Mapped[str | None] = mapped_column(Text)
    used_fallback_model: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"
    __table_args__ = (
        CheckConstraint("status IN ('started','success','failed','skipped')", name="ck_pipeline_status"),
        {"schema": "audit"},
    )

    pipeline_run_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    command_name: Mapped[str] = mapped_column(Text, nullable=False)
    command_args: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(Text, nullable=False)
    exit_code: Mapped[int | None] = mapped_column(Integer)
    source_code_git_sha: Mapped[str] = mapped_column(Text, nullable=False)
    row_counts: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    error_message: Mapped[str | None] = mapped_column(Text)
    log_uri: Mapped[str | None] = mapped_column(Text)


class IngestionManifest(Base):
    __tablename__ = "ingestion_manifests"
    __table_args__ = ({"schema": "audit"},)

    job_id: Mapped[str] = mapped_column(Text, primary_key=True)
    source_name: Mapped[str] = mapped_column(Text, nullable=False)
    code_version_git_sha: Mapped[str] = mapped_column(Text, nullable=False)
    config_hash: Mapped[str] = mapped_column(Text, nullable=False)
    started_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    row_counts_bronze: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    row_counts_silver: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    row_counts_gold: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    errors: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    warnings: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    manifest_uri: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class DataQualityFailure(Base):
    __tablename__ = "data_quality_failures"
    __table_args__ = (
        CheckConstraint("severity IN ('warning','error','fatal')", name="ck_data_quality_severity"),
        {"schema": "audit"},
    )

    data_quality_failure_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    pipeline_run_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("audit.pipeline_runs.pipeline_run_id")
    )
    table_name: Mapped[str] = mapped_column(Text, nullable=False)
    record_key: Mapped[str | None] = mapped_column(Text)
    check_name: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str] = mapped_column(Text, nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    observed_value_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class WuFetchWindow(Base):
    __tablename__ = "wu_fetch_windows"
    __table_args__ = (
        UniqueConstraint(
            "job_id",
            "station_id",
            "window_start_date",
            "window_end_date",
            name="uq_wu_fetch_windows_job_station_window",
        ),
        CheckConstraint(
            "status IN ('pending','running','succeeded','failed','no_data','skipped')",
            name="ck_wu_fetch_windows_status",
        ),
        CheckConstraint(
            "window_end_date >= window_start_date",
            name="ck_wu_fetch_windows_date_order",
        ),
        {"schema": "audit"},
    )

    wu_fetch_window_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    job_id: Mapped[str] = mapped_column(Text, nullable=False)
    station_id: Mapped[str] = mapped_column(Text, ForeignKey("registry.stations.station_id"), nullable=False)
    wunderground_station_id: Mapped[str] = mapped_column(Text, nullable=False)
    weathercom_location_id: Mapped[str] = mapped_column(Text, nullable=False)
    window_start_date: Mapped[date] = mapped_column(Date, nullable=False)
    window_end_date: Mapped[date] = mapped_column(Date, nullable=False)
    units: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'e'"))
    status: Mapped[str] = mapped_column(Text, nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    http_status: Mapped[int | None] = mapped_column(Integer)
    error_type: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    source_request_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("bronze.source_requests.source_request_id")
    )
    source_record_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("bronze.source_records.source_record_id")
    )
    observations_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    daily_rows_upserted: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    intraday_rows_upserted: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    started_at_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class WuStationDateCoverage(Base):
    __tablename__ = "wu_station_date_coverage"
    __table_args__ = (
        CheckConstraint(
            "status IN ('saved','failed','no_data','not_fetched')",
            name="ck_wu_station_date_coverage_status",
        ),
        CheckConstraint(
            "quality_flag IN ('ok','suspect','failed','missing')",
            name="ck_wu_station_date_coverage_quality",
        ),
        {"schema": "audit"},
    )

    station_id: Mapped[str] = mapped_column(
        Text, ForeignKey("registry.stations.station_id"), primary_key=True
    )
    local_date: Mapped[date] = mapped_column(Date, primary_key=True)
    wunderground_station_id: Mapped[str] = mapped_column(Text, nullable=False)
    weathercom_location_id: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    source_request_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("bronze.source_requests.source_request_id")
    )
    source_record_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("bronze.source_records.source_record_id")
    )
    wu_fetch_window_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("audit.wu_fetch_windows.wu_fetch_window_id")
    )
    daily_actual_present: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    intraday_observation_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    first_attempt_at_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_attempt_at_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_success_at_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error_type: Mapped[str | None] = mapped_column(Text)
    last_error_message: Mapped[str | None] = mapped_column(Text)
    quality_flag: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'ok'"))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class SourceRequest(Base):
    __tablename__ = "source_requests"
    __table_args__ = ({"schema": "bronze"},)

    source_request_id: Mapped[str] = mapped_column(Text, primary_key=True)
    source_name: Mapped[str] = mapped_column(Text, nullable=False)
    source_endpoint: Mapped[str] = mapped_column(Text, nullable=False)
    request_method: Mapped[str] = mapped_column(Text, nullable=False)
    request_params_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    request_headers_redacted: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    retrieved_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    provider_response_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    http_status: Mapped[int | None] = mapped_column(Integer)
    response_content_type: Mapped[str | None] = mapped_column(Text)
    response_body_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    response_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    raw_storage_uri: Mapped[str] = mapped_column(Text, nullable=False)
    parser_version: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class SourceRecord(Base):
    __tablename__ = "source_records"
    __table_args__ = (
        CheckConstraint(
            "payload_format IN ('json','csv','ndjson','text','parquet','binary_uri')",
            name="ck_source_records_payload_format",
        ),
        CheckConstraint(
            "((payload_json IS NOT NULL)::int + (payload_text IS NOT NULL)::int + "
            "(payload_uri IS NOT NULL)::int) >= 1",
            name="ck_source_records_payload_present",
        ),
        UniqueConstraint(
            "source_name",
            "provider_name",
            "endpoint_name",
            "provider_record_key",
            "revision_number",
            name="uq_source_records_revision",
        ),
        {"schema": "bronze"},
    )

    source_record_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    source_request_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("bronze.source_requests.source_request_id")
    )
    source_name: Mapped[str] = mapped_column(Text, nullable=False)
    provider_name: Mapped[str] = mapped_column(Text, nullable=False)
    endpoint_name: Mapped[str] = mapped_column(Text, nullable=False)
    provider_record_key: Mapped[str] = mapped_column(Text, nullable=False)
    request_hash: Mapped[str | None] = mapped_column(Text)
    payload_hash: Mapped[str] = mapped_column(Text, nullable=False)
    payload_format: Mapped[str] = mapped_column(Text, nullable=False)
    payload_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    payload_text: Mapped[str | None] = mapped_column(Text)
    payload_uri: Mapped[str | None] = mapped_column(Text)
    provider_issued_at_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    provider_valid_at_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    provider_available_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    acquired_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revision_number: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    supersedes_source_record_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("bronze.source_records.source_record_id")
    )
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class NormalizedFact(Base):
    __tablename__ = "normalized_facts"
    __table_args__ = (
        UniqueConstraint("raw_row_hash", name="uq_normalized_facts_raw_row_hash"),
        {"schema": "silver"},
    )

    normalized_fact_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    source_name: Mapped[str] = mapped_column(Text, nullable=False)
    source_product: Mapped[str | None] = mapped_column(Text)
    source_model: Mapped[str | None] = mapped_column(Text)
    source_member: Mapped[str | None] = mapped_column(Text)
    source_cycle: Mapped[str | None] = mapped_column(Text)
    run_time_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    valid_time_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    forecast_hour: Mapped[float | None] = mapped_column(Float)
    station_id: Mapped[str | None] = mapped_column(Text, ForeignKey("registry.stations.station_id"))
    provider_station_id: Mapped[str | None] = mapped_column(Text)
    grid_point_id: Mapped[str | None] = mapped_column(Text)
    lat: Mapped[float | None] = mapped_column(Float)
    lon: Mapped[float | None] = mapped_column(Float)
    variable_name: Mapped[str] = mapped_column(Text, nullable=False)
    variable_level: Mapped[str | None] = mapped_column(Text)
    variable_info: Mapped[str | None] = mapped_column(Text)
    unit_original: Mapped[str | None] = mapped_column(Text)
    value_original: Mapped[float | None] = mapped_column(Float)
    unit_canonical: Mapped[str | None] = mapped_column(Text)
    value_canonical: Mapped[float | None] = mapped_column(Float)
    retrieved_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    provider_available_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    our_ingested_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    availability_method: Mapped[str] = mapped_column(Text, nullable=False)
    source_request_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("bronze.source_requests.source_request_id")
    )
    source_record_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("bronze.source_records.source_record_id")
    )
    raw_row_hash: Mapped[str] = mapped_column(Text, nullable=False)
    quality_flag: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'ok'"))
    quality_note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class AvailabilityLedger(Base):
    __tablename__ = "availability_ledger"
    __table_args__ = (
        CheckConstraint(
            "availability_method IN "
            "('observed_provider_timestamp','observed_ingest_timestamp',"
            "'conservative_lag_rule','manual_override')",
            name="ck_availability_method",
        ),
        {"schema": "silver"},
    )

    availability_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    source_record_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("bronze.source_records.source_record_id")
    )
    source_name: Mapped[str] = mapped_column(Text, nullable=False)
    provider_name: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_record_key: Mapped[str] = mapped_column(Text, nullable=False)
    station_id: Mapped[str | None] = mapped_column(Text, ForeignKey("registry.stations.station_id"))
    model_name: Mapped[str | None] = mapped_column(Text)
    run_time_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    valid_time_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    forecast_hour: Mapped[int | None] = mapped_column(Integer)
    member: Mapped[str | None] = mapped_column(Text)
    variable_name: Mapped[str] = mapped_column(Text, nullable=False)
    provider_available_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    acquired_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    effective_available_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    availability_method: Mapped[str] = mapped_column(Text, nullable=False)
    source_lag_seconds: Mapped[int | None] = mapped_column(Integer)
    is_revision_current: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class WuDailyActual(Base):
    __tablename__ = "wu_daily_actuals"
    __table_args__ = (
        CheckConstraint(
            "label_method IS NULL OR label_method IN "
            "('wunderground_daily_summary','computed_from_wunderground_intraday_rows')",
            name="ck_wu_daily_label_method",
        ),
        CheckConstraint(
            "quality_flag IN ('ok','suspect','failed','missing','revised')",
            name="ck_wu_daily_quality_flag",
        ),
        {"schema": "silver"},
    )

    station_id: Mapped[str] = mapped_column(
        Text, ForeignKey("registry.stations.station_id"), primary_key=True
    )
    wunderground_station_id: Mapped[str] = mapped_column(Text, nullable=False)
    weathercom_location_id: Mapped[str] = mapped_column(Text, nullable=False)
    local_date: Mapped[date] = mapped_column(Date, primary_key=True)
    timezone_name: Mapped[str] = mapped_column(Text, nullable=False)
    local_day_start_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    local_day_end_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    daily_high_f: Mapped[int | None] = mapped_column(Integer)
    settlement_high_f_whole: Mapped[int | None] = mapped_column(Integer)
    daily_low_f: Mapped[int | None] = mapped_column(Integer)
    daily_avg_temp_f: Mapped[float | None] = mapped_column(Float)
    daily_high_dewpoint_f: Mapped[float | None] = mapped_column(Float)
    daily_low_dewpoint_f: Mapped[float | None] = mapped_column(Float)
    daily_precipitation_in: Mapped[float | None] = mapped_column(Float)
    daily_max_wind_speed_mph: Mapped[float | None] = mapped_column(Float)
    daily_max_wind_gust_mph: Mapped[float | None] = mapped_column(Float)
    daily_avg_wind_speed_mph: Mapped[float | None] = mapped_column(Float)
    daily_dominant_wind_direction_deg: Mapped[float | None] = mapped_column(Float)
    label_method: Mapped[str | None] = mapped_column(Text)
    daily_high_source_field: Mapped[str | None] = mapped_column(Text)
    provider_available_at_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    our_ingested_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source_request_id: Mapped[str] = mapped_column(
        Text, ForeignKey("bronze.source_requests.source_request_id"), nullable=False
    )
    source_record_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("bronze.source_records.source_record_id")
    )
    source_daily_summary_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    raw_daily_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    observations_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    quality_flag: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'ok'"))
    quality_note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class WuIntradayObservation(Base):
    __tablename__ = "wu_intraday_observations"
    __table_args__ = (
        CheckConstraint(
            "quality_flag IN ('ok','suspect','failed','missing','duplicate','revised')",
            name="ck_wu_intraday_quality_flag",
        ),
        {"schema": "silver"},
    )

    station_id: Mapped[str] = mapped_column(
        Text, ForeignKey("registry.stations.station_id"), primary_key=True
    )
    wunderground_station_id: Mapped[str] = mapped_column(Text, nullable=False)
    weathercom_location_id: Mapped[str] = mapped_column(Text, nullable=False)
    observation_time_local: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    observation_time_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    local_date: Mapped[date] = mapped_column(Date, nullable=False)
    timezone_name: Mapped[str] = mapped_column(Text, nullable=False)
    temp_f: Mapped[float | None] = mapped_column(Float)
    dewpoint_f: Mapped[float | None] = mapped_column(Float)
    humidity_pct: Mapped[float | None] = mapped_column(Float)
    wind_speed_mph: Mapped[float | None] = mapped_column(Float)
    wind_gust_mph: Mapped[float | None] = mapped_column(Float)
    wind_direction_deg: Mapped[float | None] = mapped_column(Float)
    pressure_in: Mapped[float | None] = mapped_column(Float)
    precipitation_in: Mapped[float | None] = mapped_column(Float)
    condition_text: Mapped[str | None] = mapped_column(Text)
    cloud_cover_text: Mapped[str | None] = mapped_column(Text)
    uv_index: Mapped[float | None] = mapped_column(Float)
    solar_radiation: Mapped[float | None] = mapped_column(Float)
    raw_observation_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    provider_available_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    our_ingested_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source_request_id: Mapped[str] = mapped_column(
        Text, ForeignKey("bronze.source_requests.source_request_id"), nullable=False
    )
    source_record_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("bronze.source_records.source_record_id")
    )
    quality_flag: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'ok'"))
    quality_note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class WuDailyActualRevision(Base):
    __tablename__ = "wu_daily_actual_revisions"
    __table_args__ = ({"schema": "silver"},)

    wu_daily_actual_revision_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    station_id: Mapped[str] = mapped_column(Text, ForeignKey("registry.stations.station_id"), nullable=False)
    local_date: Mapped[date] = mapped_column(Date, nullable=False)
    previous_daily_high_f: Mapped[int | None] = mapped_column(Integer)
    new_daily_high_f: Mapped[int | None] = mapped_column(Integer)
    previous_source_request_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("bronze.source_requests.source_request_id")
    )
    new_source_request_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("bronze.source_requests.source_request_id")
    )
    previous_source_record_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("bronze.source_records.source_record_id")
    )
    new_source_record_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("bronze.source_records.source_record_id")
    )
    detected_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    note: Mapped[str | None] = mapped_column(Text)


class TargetInstance(Base):
    __tablename__ = "target_instances"
    __table_args__ = (
        UniqueConstraint("target_date", "cutoff_id", name="uq_target_instances_date_cutoff"),
        {"schema": "gold"},
    )

    target_instance_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    target_date: Mapped[date] = mapped_column(Date, nullable=False)
    cutoff_id: Mapped[str] = mapped_column(Text, ForeignKey("registry.cutoffs.cutoff_id"), nullable=False)
    cutoff_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    target_station_id: Mapped[str] = mapped_column(
        Text, ForeignKey("registry.stations.station_id"), nullable=False, server_default=text("'KLGA'")
    )
    local_day_start_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    local_day_end_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    settlement_high_f_whole: Mapped[int | None] = mapped_column(Integer)
    settlement_high_available_at_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    label_available: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    label_revision_sensitive: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class FeatureValue(Base):
    __tablename__ = "feature_values"
    __table_args__ = (
        UniqueConstraint(
            "target_instance_id",
            "feature_name",
            "feature_build_version",
            name="uq_feature_values_identity",
        ),
        {"schema": "gold"},
    )

    feature_value_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    target_instance_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("gold.target_instances.target_instance_id", ondelete="CASCADE"),
        nullable=False,
    )
    feature_family: Mapped[str] = mapped_column(Text, nullable=False)
    feature_name: Mapped[str] = mapped_column(Text, nullable=False)
    feature_value: Mapped[float | None] = mapped_column(Float)
    feature_unit: Mapped[str | None] = mapped_column(Text)
    feature_available: Mapped[bool] = mapped_column(Boolean, nullable=False)
    source_latest_valid_time_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_latest_run_time_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_age_hours: Mapped[float | None] = mapped_column(Float)
    source_latency_minutes: Mapped[float | None] = mapped_column(Float)
    feature_build_version: Mapped[str] = mapped_column(Text, nullable=False)
    max_source_available_at_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_trace_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class FeatureMatrix(Base):
    __tablename__ = "feature_matrix"
    __table_args__ = (
        UniqueConstraint("target_instance_id", "feature_version_id", name="uq_feature_matrix_identity"),
        {"schema": "gold"},
    )

    feature_matrix_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    target_instance_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("gold.target_instances.target_instance_id", ondelete="CASCADE"),
        nullable=False,
    )
    feature_version_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("registry.feature_versions.feature_version_id"), nullable=False
    )
    feature_vector_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    feature_availability_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    label_high_temp_f: Mapped[int | None] = mapped_column(Integer)
    label_available: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    label_revision_sensitive: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class BacktestRun(Base):
    __tablename__ = "backtest_runs"
    __table_args__ = (
        CheckConstraint("status IN ('started','success','failed','skipped')", name="ck_backtest_status"),
        CheckConstraint("market_mode IN ('synthetic','historical_polymarket')", name="ck_backtest_market_mode"),
        {"schema": "reports"},
    )

    backtest_run_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    run_name: Mapped[str] = mapped_column(Text, nullable=False)
    run_id_text: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(Text, nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    cutoff_id: Mapped[str | None] = mapped_column(Text)
    market_mode: Mapped[str] = mapped_column(Text, nullable=False)
    frozen_config_uri: Mapped[str | None] = mapped_column(Text)
    frozen_config_hash: Mapped[str | None] = mapped_column(Text)
    model_version_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True))
    calibration_version_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True))
    feature_version_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True))
    source_code_git_sha: Mapped[str] = mapped_column(Text, nullable=False)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    artifact_root_uri: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)


class Metric(Base):
    __tablename__ = "metrics"
    __table_args__ = (
        CheckConstraint(
            "((metric_value IS NOT NULL)::int + (metric_text IS NOT NULL)::int + "
            "(metric_json <> '{}'::jsonb)::int) >= 1",
            name="ck_metrics_value_present",
        ),
        {"schema": "reports"},
    )

    metric_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    metric_group: Mapped[str] = mapped_column(Text, nullable=False)
    metric_name: Mapped[str] = mapped_column(Text, nullable=False)
    metric_value: Mapped[float | None] = mapped_column(Float)
    metric_text: Mapped[str | None] = mapped_column(Text)
    metric_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    target_date: Mapped[date | None] = mapped_column(Date)
    cutoff_id: Mapped[str | None] = mapped_column(Text)
    backtest_run_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True))
    model_version_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True))
    feature_version_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class StationDailyActual(Base):
    __tablename__ = "station_daily_actuals"
    __table_args__ = (
        UniqueConstraint(
            "target_date",
            "station_id",
            "source_name",
            "revision_number",
            name="uq_station_daily_actuals_revision",
        ),
        {"schema": "silver"},
    )

    station_daily_actual_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    target_date: Mapped[date] = mapped_column(Date, nullable=False)
    station_id: Mapped[str] = mapped_column(Text, ForeignKey("registry.stations.station_id"), nullable=False)
    source_name: Mapped[str] = mapped_column(Text, nullable=False)
    high_temp_f: Mapped[int | None] = mapped_column(Integer)
    low_temp_f: Mapped[int | None] = mapped_column(Integer)
    avg_temp_f: Mapped[float | None] = mapped_column(Float)
    precip_in: Mapped[float | None] = mapped_column(Float)
    max_wind_speed_mph: Mapped[float | None] = mapped_column(Float)
    max_wind_gust_mph: Mapped[float | None] = mapped_column(Float)
    provider_available_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    effective_available_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source_request_id: Mapped[str | None] = mapped_column(Text, ForeignKey("bronze.source_requests.source_request_id"))
    source_record_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("bronze.source_records.source_record_id"))
    revision_number: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    quality_flag: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'ok'"))
    source_trace_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=text("now()"))


class StationObservation(Base):
    __tablename__ = "station_observations"
    __table_args__ = ({"schema": "silver"},)

    station_observation_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    station_id: Mapped[str] = mapped_column(Text, ForeignKey("registry.stations.station_id"), nullable=False)
    source_name: Mapped[str] = mapped_column(Text, nullable=False)
    observation_time_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    local_date: Mapped[date] = mapped_column(Date, nullable=False)
    temp_f: Mapped[float | None] = mapped_column(Float)
    dewpoint_f: Mapped[float | None] = mapped_column(Float)
    humidity_pct: Mapped[float | None] = mapped_column(Float)
    wind_speed_mph: Mapped[float | None] = mapped_column(Float)
    wind_gust_mph: Mapped[float | None] = mapped_column(Float)
    wind_direction_deg: Mapped[float | None] = mapped_column(Float)
    pressure_in: Mapped[float | None] = mapped_column(Float)
    precipitation_in: Mapped[float | None] = mapped_column(Float)
    condition_text: Mapped[str | None] = mapped_column(Text)
    provider_available_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    effective_available_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source_request_id: Mapped[str | None] = mapped_column(Text, ForeignKey("bronze.source_requests.source_request_id"))
    source_record_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("bronze.source_records.source_record_id"))
    raw_row_hash: Mapped[str] = mapped_column(Text, nullable=False)
    quality_flag: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'ok'"))
    source_trace_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=text("now()"))


class MosGuidance(Base):
    __tablename__ = "mos_guidance"
    __table_args__ = ({"schema": "silver"},)

    mos_guidance_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    station_id: Mapped[str] = mapped_column(Text, ForeignKey("registry.stations.station_id"), nullable=False)
    mos_station_id: Mapped[str] = mapped_column(Text, nullable=False)
    source_product: Mapped[str] = mapped_column(Text, nullable=False)
    endpoint_model: Mapped[str] = mapped_column(Text, nullable=False)
    cutoff_id: Mapped[str] = mapped_column(Text, ForeignKey("registry.cutoffs.cutoff_id"), nullable=False)
    run_time_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    forecast_valid_time_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    target_date: Mapped[date] = mapped_column(Date, nullable=False)
    raw_values_jsonb: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    tmax_f: Mapped[float | None] = mapped_column(Float)
    tmp_f: Mapped[float | None] = mapped_column(Float)
    dpt_f: Mapped[float | None] = mapped_column(Float)
    wsp_kt: Mapped[float | None] = mapped_column(Float)
    pop: Mapped[float | None] = mapped_column(Float)
    qpf: Mapped[float | None] = mapped_column(Float)
    tstm_prob: Mapped[float | None] = mapped_column(Float)
    provider_available_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    effective_available_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    availability_method: Mapped[str] = mapped_column(Text, nullable=False)
    source_request_id: Mapped[str | None] = mapped_column(Text, ForeignKey("bronze.source_requests.source_request_id"))
    source_record_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("bronze.source_records.source_record_id"))
    request_sha256: Mapped[str | None] = mapped_column(Text)
    raw_row_hash: Mapped[str] = mapped_column(Text, nullable=False)
    source_trace_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=text("now()"))


class ExpertPrediction(Base):
    __tablename__ = "expert_predictions"
    __table_args__ = (
        CheckConstraint("prediction_kind IN ('oof','holdout','forecast','replay')", name="ck_expert_prediction_kind"),
        CheckConstraint("prediction_status IN ('ok','fallback','disabled_data_sufficiency')", name="ck_expert_prediction_status"),
        {"schema": "predictions"},
    )

    expert_prediction_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    target_instance_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("gold.target_instances.target_instance_id"), nullable=False)
    expert_name: Mapped[str] = mapped_column(Text, nullable=False)
    prediction_kind: Mapped[str] = mapped_column(Text, nullable=False)
    model_version_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("registry.model_versions.model_version_id"), nullable=False)
    feature_version_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("registry.feature_versions.feature_version_id"), nullable=False)
    fold_id: Mapped[str | None] = mapped_column(Text)
    training_start_date: Mapped[date | None] = mapped_column(Date)
    training_end_date: Mapped[date | None] = mapped_column(Date)
    pmf_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    expected_tmax_f: Mapped[float] = mapped_column(Float, nullable=False)
    median_tmax_f: Mapped[int] = mapped_column(Integer, nullable=False)
    mode_tmax_f: Mapped[int] = mapped_column(Integer, nullable=False)
    prediction_interval_low_f: Mapped[int] = mapped_column(Integer, nullable=False)
    prediction_interval_high_f: Mapped[int] = mapped_column(Integer, nullable=False)
    uncertainty_f: Mapped[float] = mapped_column(Float, nullable=False)
    feature_names: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, server_default=text("'{}'::text[]"))
    feature_hash: Mapped[str] = mapped_column(Text, nullable=False)
    source_availability_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    diagnostics_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    prediction_status: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'ok'"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=text("now()"))


class FinalPrediction(Base):
    __tablename__ = "final_predictions"
    __table_args__ = (
        CheckConstraint("prediction_kind IN ('oof','holdout','forecast','replay')", name="ck_final_prediction_kind"),
        {"schema": "predictions"},
    )

    final_prediction_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    target_instance_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("gold.target_instances.target_instance_id"), nullable=False)
    prediction_kind: Mapped[str] = mapped_column(Text, nullable=False)
    model_version_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("registry.model_versions.model_version_id"), nullable=False)
    feature_version_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("registry.feature_versions.feature_version_id"), nullable=False)
    expert_prediction_ids: Mapped[list[UUID]] = mapped_column(ARRAY(PG_UUID(as_uuid=True)), nullable=False, server_default=text("'{}'::uuid[]"))
    expert_weights_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    pmf_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    expected_tmax_f: Mapped[float] = mapped_column(Float, nullable=False)
    median_tmax_f: Mapped[int] = mapped_column(Integer, nullable=False)
    mode_tmax_f: Mapped[int] = mapped_column(Integer, nullable=False)
    prediction_interval_low_f: Mapped[int] = mapped_column(Integer, nullable=False)
    prediction_interval_high_f: Mapped[int] = mapped_column(Integer, nullable=False)
    uncertainty_f: Mapped[float] = mapped_column(Float, nullable=False)
    entropy: Mapped[float] = mapped_column(Float, nullable=False)
    diagnostics_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=text("now()"))


class CalibrationVersion(Base):
    __tablename__ = "calibration_versions"
    __table_args__ = (
        CheckConstraint("prediction_kind IN ('oof','holdout','forecast','replay')", name="ck_calibration_kind"),
        {"schema": "predictions"},
    )

    calibration_version_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    calibration_name: Mapped[str] = mapped_column(Text, nullable=False)
    prediction_kind: Mapped[str] = mapped_column(Text, nullable=False)
    model_version_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("registry.model_versions.model_version_id"), nullable=False)
    training_start_date: Mapped[date] = mapped_column(Date, nullable=False)
    training_end_date: Mapped[date] = mapped_column(Date, nullable=False)
    cutoff_id: Mapped[str | None] = mapped_column(Text, ForeignKey("registry.cutoffs.cutoff_id"))
    method: Mapped[str] = mapped_column(Text, nullable=False)
    config_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    artifact_uri: Mapped[str | None] = mapped_column(Text)
    artifact_hash: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=text("now()"))


class CalibratedPrediction(Base):
    __tablename__ = "calibrated_predictions"
    __table_args__ = ({"schema": "predictions"},)

    calibrated_prediction_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    final_prediction_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("predictions.final_predictions.final_prediction_id"), nullable=False)
    calibration_version_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("predictions.calibration_versions.calibration_version_id"), nullable=False)
    pmf_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    expected_tmax_f: Mapped[float] = mapped_column(Float, nullable=False)
    median_tmax_f: Mapped[int] = mapped_column(Integer, nullable=False)
    mode_tmax_f: Mapped[int] = mapped_column(Integer, nullable=False)
    prediction_interval_low_f: Mapped[int] = mapped_column(Integer, nullable=False)
    prediction_interval_high_f: Mapped[int] = mapped_column(Integer, nullable=False)
    uncertainty_f: Mapped[float] = mapped_column(Float, nullable=False)
    diagnostics_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=text("now()"))


class ForecastEvaluationRun(Base):
    __tablename__ = "forecast_evaluation_runs"
    __table_args__ = (
        CheckConstraint("status IN ('started','success','failed','skipped')", name="ck_forecast_eval_status"),
        CheckConstraint("prediction_kind IN ('oof','holdout','forecast','replay')", name="ck_forecast_eval_prediction_kind"),
        {"schema": "reports"},
    )

    evaluation_run_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    run_id_text: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    run_name: Mapped[str] = mapped_column(Text, nullable=False)
    prediction_kind: Mapped[str] = mapped_column(Text, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=text("now()"))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(Text, nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    cutoff_id: Mapped[str | None] = mapped_column(Text, ForeignKey("registry.cutoffs.cutoff_id"))
    model_version_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("registry.model_versions.model_version_id"))
    calibration_version_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("predictions.calibration_versions.calibration_version_id"))
    feature_version_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("registry.feature_versions.feature_version_id"))
    source_code_git_sha: Mapped[str] = mapped_column(Text, nullable=False)
    config_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    artifact_root_uri: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)


class ForecastEvaluationDailyScore(Base):
    __tablename__ = "forecast_evaluation_daily_scores"
    __table_args__ = ({"schema": "reports"},)

    evaluation_daily_score_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    evaluation_run_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("reports.forecast_evaluation_runs.evaluation_run_id"), nullable=False)
    target_date: Mapped[date] = mapped_column(Date, nullable=False)
    cutoff_id: Mapped[str] = mapped_column(Text, ForeignKey("registry.cutoffs.cutoff_id"), nullable=False)
    prediction_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    calibrated_prediction_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("predictions.calibrated_predictions.calibrated_prediction_id"))
    settled_wu_tmax_f: Mapped[int] = mapped_column(Integer, nullable=False)
    expected_tmax_f: Mapped[float] = mapped_column(Float, nullable=False)
    median_tmax_f: Mapped[int] = mapped_column(Integer, nullable=False)
    mode_tmax_f: Mapped[int] = mapped_column(Integer, nullable=False)
    absolute_error_f: Mapped[float] = mapped_column(Float, nullable=False)
    signed_error_f: Mapped[float] = mapped_column(Float, nullable=False)
    squared_error_f: Mapped[float] = mapped_column(Float, nullable=False)
    pmf_probability_at_observed: Mapped[float] = mapped_column(Float, nullable=False)
    log_score: Mapped[float] = mapped_column(Float, nullable=False)
    within_1f: Mapped[bool] = mapped_column(Boolean, nullable=False)
    within_2f: Mapped[bool] = mapped_column(Boolean, nullable=False)
    prediction_interval_low_f: Mapped[int] = mapped_column(Integer, nullable=False)
    prediction_interval_high_f: Mapped[int] = mapped_column(Integer, nullable=False)
    prediction_interval_hit: Mapped[bool] = mapped_column(Boolean, nullable=False)
    label_source_record_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("bronze.source_records.source_record_id"))
    label_revision_number: Mapped[int] = mapped_column(Integer, nullable=False)
    label_available_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    leakage_checked: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    diagnostics_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=text("now()"))
