from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Literal
from uuid import UUID


FetchShape = Literal[
    "hourly_peak",
    "rtma_latest",
    "urma_peak_temp",
    "synoptic",
    "nbmqmd_max18",
    "nbm_tmax_native",
]
MemberMode = Literal["none", "gefs_31", "ecmwf_51"]
ChunkStatus = Literal[
    "planned",
    "running",
    "completed",
    "completed_empty",
    "failed",
    "rate_limited",
    "auth_failed",
    "selector_missing",
    "skipped",
]


@dataclass(frozen=True)
class GribStreamModelSpec:
    tier: str
    model_id: str
    catalog_archive_start: date
    fetch_shape: FetchShape
    expected_credits_per_day: int
    expected_total_credits: int
    variable_group: str
    buffer: timedelta | None
    intended_latest_cycle: str
    member_mode: MemberMode = "none"
    expected_members: int = 1
    default_chunk_days: int = 31

    @property
    def effective_target_start(self) -> date:
        return self.catalog_archive_start + timedelta(days=1)


@dataclass(frozen=True)
class ResolvedSelector:
    alias: str
    request_variables: tuple[dict[str, Any], ...]
    variable_name: str
    variable_level: str | None = None
    variable_info: str | None = None
    shared_parameter: str | None = None
    request_expressions: tuple[dict[str, Any], ...] = ()
    unit_hint: str | None = None

    @property
    def request_alias(self) -> str:
        return self.alias


@dataclass(frozen=True)
class GribStreamChunk:
    model_id: str
    target_start_date: date
    target_end_date: date
    cutoff_id: str
    cutoff_utc_time: str
    coordinate_tier: str
    as_of_utc: datetime | None
    valid_times_utc: tuple[datetime, ...]
    selectors: tuple[ResolvedSelector, ...]
    members: tuple[int, ...]
    request_payload: dict[str, Any]
    request_sha256: str
    estimated_credits: int
    chunk_id: str
    fetch_shape: str = ""
    feature_profile: str = "BROAD_V1"
    persistence_mode: str = "silver_atomic"
    endpoint_type: str = "timeseries"
    expected_run_valid_pairs_utc: tuple[tuple[datetime, datetime], ...] = ()

    @property
    def target_days(self) -> int:
        return (self.target_end_date - self.target_start_date).days + 1


@dataclass(frozen=True)
class GribStreamRawResponse:
    chunk: GribStreamChunk
    endpoint_url_redacted: str
    retrieved_at_utc: datetime
    http_status: int | None
    content_type: str | None
    response_body_sha256: str
    response_size_bytes: int
    raw_storage_uri: str
    attempts: int
    error_type: str | None = None
    error_message: str | None = None

    @property
    def success(self) -> bool:
        return self.http_status is not None and 200 <= self.http_status < 300


@dataclass(frozen=True)
class GribStreamParsedValue:
    model_id: str
    endpoint_type: str
    target_date: date
    cutoff_id: str
    cutoff_utc: datetime
    as_of_utc: datetime | None
    coordinate_tier: str
    grid_point_id: str
    lat: float
    lon: float
    forecasted_at_utc: datetime
    forecasted_time_utc: datetime
    forecast_hour: float
    member: str
    variable_alias: str
    variable_name: str
    variable_level: str | None
    variable_info: str | None
    unit_original: str | None
    value_original: float | None
    unit_canonical: str | None
    value_canonical: float | None
    index_updated_at_utc: datetime | None
    provider_available_at_utc: datetime
    effective_available_at_utc: datetime
    availability_method: str
    raw_row_hash: str
    raw_row_json: dict[str, Any]
    quality_flag: str = "ok"
    quality_note: str | None = None


@dataclass(frozen=True)
class GribStreamGoldFeature:
    target_date: date
    cutoff_id: str
    cutoff_utc: datetime
    model_id: str
    feature_family: str
    feature_name: str
    feature_value: float | None
    feature_unit: str | None
    feature_available: bool
    source_latest_valid_time_utc: datetime | None
    source_latest_run_time_utc: datetime | None
    source_age_hours: float | None
    source_latency_minutes: float | None
    max_source_available_at_utc: datetime | None
    source_trace_json: dict[str, Any]
    feature_build_version: str = "TMAX_THIN_V1"


@dataclass(frozen=True)
class ParsedGribStreamResponse:
    values: tuple[GribStreamParsedValue, ...]
    row_count_raw: int
    gaps: tuple[dict[str, Any], ...] = ()
    gold_features: tuple[GribStreamGoldFeature, ...] = ()


@dataclass(frozen=True)
class PersistedGribStreamChunk:
    chunk_id: str
    source_request_id: str | None
    source_record_id: UUID | None
    status: ChunkStatus
    rows_upserted: int
    availability_rows_upserted: int
    gaps_upserted: int
    http_status: int | None = None
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class GribStreamJobPlan:
    job_id: str
    cutoff_id: str
    start_date: date
    end_date: date
    coordinate_tier: str
    chunks: tuple[GribStreamChunk, ...]
    selector_gaps: tuple[dict[str, Any], ...] = field(default_factory=tuple)
