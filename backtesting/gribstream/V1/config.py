from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Iterator
from zoneinfo import ZoneInfo

UTC = timezone.utc


@dataclass(frozen=True)
class StationConfig:
    station_id: str
    latitude: float
    longitude: float
    timezone_name: str

    @property
    def zoneinfo(self) -> ZoneInfo:
        return ZoneInfo(self.timezone_name)


ROOT_DIR = Path(__file__).resolve().parent
SQLITE_DIR = ROOT_DIR / "sqlite"
DB_PATH = SQLITE_DIR / "knyc_gribstream_v1.sqlite3"
SCHEMA_PATH = ROOT_DIR / "sql" / "schema.sql"

STATION = StationConfig(
    station_id="KNYC",
    latitude=40.78333,
    longitude=-73.96667,
    timezone_name="America/New_York",
)

WARMUP_START_DATE = date(2022, 1, 1)
EVALUATION_START_DATE = date(2023, 1, 1)
EVALUATION_END_DATE = date(2024, 12, 31)
TRUTH_START_DATE = WARMUP_START_DATE
TRUTH_END_DATE = EVALUATION_END_DATE

DECISION_CUTOFF_UTC = time(13, 0, 0)
GRIBSTREAM_API_TOKEN_ENV = "GRIBSTREAM_API_TOKEN"
GRIBSTREAM_BASE_URL = "https://gribstream.com"
GRIBSTREAM_RESPONSE_FORMAT = "text/csv"
GRIBSTREAM_RESPONSE_COMPRESSED = 1
TRUTH_SOURCE_NAME = "iem_cli_json"
IEM_CLI_BASE_URL = "https://mesonet.agron.iastate.edu/json/cli.py"
TEMPERATURE_NATIVE_UNIT = "K"
TRUTH_NATIVE_UNIT = "F"

DEFAULT_FETCH_THREADS = 2
DEFAULT_TRUTH_THREADS = 4
REQUEST_CONNECT_TIMEOUT_SECONDS = 10
REQUEST_READ_TIMEOUT_SECONDS = 120
REQUEST_MAX_RETRIES = 5
REQUEST_BACKOFF_SECONDS = 1.0

ROLLING_WINDOW_DAYS = 180
ROLLING_HALF_LIFE_DAYS = 45.0
MIN_TRAIN_DAYS = 45
MIN_RMSE_FLOOR = 0.75
MODEL_WEIGHT_CAP = 0.35
FAMILY_WEIGHT_CAP = 0.50

FAMILY_ORDER = (
    "regional_noaa_short",
    "noaa_global_blend",
    "ecmwf_physics",
    "ai_global",
    "ecmwf_ai",
)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
COORD_TOLERANCE_DEGREES = 0.001


def ensure_directories() -> None:
    SQLITE_DIR.mkdir(parents=True, exist_ok=True)


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def isoformat_utc(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def parse_utc(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def iter_dates(start_date: date, end_date: date) -> Iterator[date]:
    if end_date < start_date:
        raise ValueError("end_date must be >= start_date")
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def settlement_asof_utc(settlement_date_local: date) -> datetime:
    return datetime.combine(settlement_date_local, DECISION_CUTOFF_UTC, tzinfo=UTC)


def local_day_window_utc(
    settlement_date_local: date,
    timezone_name: str = STATION.timezone_name,
) -> tuple[datetime, datetime]:
    zone = ZoneInfo(timezone_name)
    local_start = datetime.combine(settlement_date_local, time(0, 0), tzinfo=zone)
    local_end = local_start + timedelta(days=1)
    return local_start.astimezone(UTC), local_end.astimezone(UTC)


def localize_forecast_time(
    forecasted_time_utc: datetime,
    timezone_name: str = STATION.timezone_name,
) -> datetime:
    if forecasted_time_utc.tzinfo is None:
        forecasted_time_utc = forecasted_time_utc.replace(tzinfo=UTC)
    return forecasted_time_utc.astimezone(ZoneInfo(timezone_name))


def kelvin_to_f(value_k: float) -> float:
    return (float(value_k) - 273.15) * 9.0 / 5.0 + 32.0


def safe_float(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        value = stripped
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def date_from_iso(value: str) -> date:
    return date.fromisoformat(value)


def days_between(older: date, newer: date) -> int:
    return (newer - older).days
