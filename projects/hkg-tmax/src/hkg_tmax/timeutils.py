from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

HONG_KONG_TZ = ZoneInfo("Asia/Hong_Kong")
UTC = UTC


class TimeContractError(ValueError):
    """Raised when a datetime violates point-in-time requirements."""


def require_aware(value: datetime, field_name: str = "datetime") -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise TimeContractError(f"{field_name} must be timezone-aware: {value!r}")
    return value


def parse_iso_aware(value: str, field_name: str = "datetime") -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise TimeContractError(f"Invalid ISO datetime for {field_name}: {value!r}") from exc
    return require_aware(parsed, field_name)


def to_utc(value: datetime) -> datetime:
    return require_aware(value).astimezone(UTC)


def to_hong_kong(value: datetime) -> datetime:
    return require_aware(value).astimezone(HONG_KONG_TZ)


def cutoff_for_local_date(target_date: date, horizon_id: str) -> datetime:
    """Return an exact Hong Kong-local cutoff for a configured candidate horizon."""
    if horizon_id == "H39":
        local_day = target_date - timedelta(days=1)
        local_time = time(0, 0)
    elif horizon_id == "H27":
        local_day = target_date - timedelta(days=1)
        local_time = time(12, 0)
    elif horizon_id == "H24N":
        local_day = target_date - timedelta(days=1)
        local_time = time(15, 0)
    elif horizon_id == "H15":
        local_day = target_date
        local_time = time(0, 0)
    else:
        raise TimeContractError(f"Unknown horizon_id: {horizon_id!r}")
    return datetime.combine(local_day, local_time, tzinfo=HONG_KONG_TZ)


def asof_eligible(available_at: datetime, cutoff_at: datetime) -> bool:
    return to_utc(available_at) <= to_utc(cutoff_at)


def enforce_asof(available_at: datetime, cutoff_at: datetime, label: str = "record") -> None:
    if not asof_eligible(available_at, cutoff_at):
        raise TimeContractError(
            f"{label} is unavailable at cutoff: "
            f"available_at={available_at.isoformat()} cutoff={cutoff_at.isoformat()}"
        )


def local_target_day_bounds(target_date: date) -> tuple[datetime, datetime]:
    start = datetime.combine(target_date, time.min, tzinfo=HONG_KONG_TZ)
    end = start + timedelta(days=1)
    return start, end
