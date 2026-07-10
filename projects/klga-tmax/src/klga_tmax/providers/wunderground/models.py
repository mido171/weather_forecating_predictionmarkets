from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal
from uuid import UUID


Units = Literal["e", "m"]


@dataclass(frozen=True)
class WundergroundRawDayResponse:
    station_id: str
    wunderground_station_id: str
    weathercom_location_id: str
    start_local_date: date
    end_local_date: date
    units: Units
    endpoint_url_redacted: str
    retrieved_at_utc: datetime
    http_status: int | None
    content_type: str | None
    response_body_text: str
    response_body_sha256: str
    response_size_bytes: int
    payload_json: dict[str, Any] | None
    attempts: int
    error_type: str | None = None
    error_message: str | None = None

    @property
    def success(self) -> bool:
        return self.http_status is not None and 200 <= self.http_status < 300 and self.payload_json is not None

    @property
    def provider_no_data(self) -> bool:
        if self.payload_json is None:
            return False
        errors = self.payload_json.get("errors")
        if not isinstance(errors, list):
            return False
        for item in errors:
            if not isinstance(item, dict):
                continue
            error = item.get("error")
            if not isinstance(error, dict):
                continue
            code = str(error.get("code") or "").strip().upper()
            message = str(error.get("message") or "").strip().lower()
            if code == "NDF-0001" or "no data found" in message:
                return True
        return False


@dataclass(frozen=True)
class WundergroundIntradayObservation:
    station_id: str
    wunderground_station_id: str
    weathercom_location_id: str
    observation_time_local: datetime
    observation_time_utc: datetime
    local_date: date
    timezone_name: str
    temp_f: float | None
    dewpoint_f: float | None
    humidity_pct: float | None
    wind_speed_mph: float | None
    wind_gust_mph: float | None
    wind_direction_deg: float | None
    pressure_in: float | None
    precipitation_in: float | None
    condition_text: str | None
    cloud_cover_text: str | None
    uv_index: float | None
    solar_radiation: float | None
    raw_observation_json: dict[str, Any]
    provider_available_at_utc: datetime
    quality_flag: str = "ok"
    quality_note: str | None = None


@dataclass(frozen=True)
class WundergroundDailyActual:
    station_id: str
    wunderground_station_id: str
    weathercom_location_id: str
    local_date: date
    timezone_name: str
    local_day_start_utc: datetime
    local_day_end_utc: datetime
    daily_high_f: int | None
    settlement_high_f_whole: int | None
    daily_low_f: int | None
    daily_avg_temp_f: float | None
    daily_high_dewpoint_f: float | None
    daily_low_dewpoint_f: float | None
    daily_precipitation_in: float | None
    daily_max_wind_speed_mph: float | None
    daily_max_wind_gust_mph: float | None
    daily_avg_wind_speed_mph: float | None
    daily_dominant_wind_direction_deg: float | None
    label_method: str | None
    daily_high_source_field: str | None
    provider_available_at_utc: datetime
    source_daily_summary_json: dict[str, Any] = field(default_factory=dict)
    raw_daily_json: dict[str, Any] = field(default_factory=dict)
    high_observation_times_local_json: list[str] = field(default_factory=list)
    hourly_observations_json: list[dict[str, Any]] = field(default_factory=list)
    provider_max_temp_values_json: list[dict[str, Any]] = field(default_factory=list)
    provider_min_temp_values_json: list[dict[str, Any]] = field(default_factory=list)
    validation_status: str = "accepted"
    validation_notes_json: dict[str, Any] = field(default_factory=dict)
    observations_count: int = 0
    quality_flag: str = "ok"
    quality_note: str | None = None


@dataclass(frozen=True)
class ParsedWundergroundResponse:
    daily_actuals: tuple[WundergroundDailyActual, ...]
    intraday_observations: tuple[WundergroundIntradayObservation, ...]

    @property
    def observations_count(self) -> int:
        return len(self.intraday_observations)


@dataclass(frozen=True)
class PersistedWundergroundWindow:
    source_request_id: str | None
    source_record_id: UUID | None
    fetch_window_id: UUID | None
    status: str
    daily_rows_upserted: int
    intraday_rows_upserted: int
    coverage_rows_updated: int
    revisions_inserted: int
    observations_count: int
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class WundergroundFetchTask:
    station_id: str
    wunderground_station_id: str
    weathercom_location_id: str
    start_date: date
    end_date: date
    units: Units = "e"
