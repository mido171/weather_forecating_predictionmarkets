from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


LOCAL_STD_OFFSET = timezone(timedelta(hours=-5))


@dataclass(frozen=True)
class RuntimeClock:
    asof_utc: datetime
    target_date_local: date
    from_time_utc: datetime
    until_time_utc: datetime


def resolve_asof_utc(now_utc: datetime | None = None) -> datetime:
    now = now_utc or datetime.now(timezone.utc)
    effective = now - timedelta(minutes=10)
    candidate = datetime(effective.year, effective.month, effective.day, 12, 0, tzinfo=timezone.utc)
    if effective < candidate:
        candidate = candidate - timedelta(days=1)
    return candidate


def parse_target_date(value: str) -> date:
    cleaned = value.strip()
    if len(cleaned) == 8 and cleaned.isdigit():
        return datetime.strptime(cleaned, "%Y%m%d").date()
    return datetime.strptime(cleaned, "%Y-%m-%d").date()


def asof_from_target_date(target_date: date) -> datetime:
    return datetime(target_date.year, target_date.month, target_date.day, 12, 0, tzinfo=timezone.utc) - timedelta(days=1)


def target_date_from_asof(asof_utc: datetime) -> date:
    return (asof_utc + timedelta(days=1)).astimezone(LOCAL_STD_OFFSET).date()


def target_window_utc(target_date: date) -> tuple[datetime, datetime]:
    from_time = datetime(target_date.year, target_date.month, target_date.day, 5, 0, tzinfo=timezone.utc)
    until_time = from_time + timedelta(days=1)
    return from_time, until_time


def standard_time_window_utc(target_date: date, zone_id: str) -> tuple[datetime, datetime]:
    zone = ZoneInfo(zone_id)
    midday = datetime.combine(target_date, time(12, 0), tzinfo=zone)
    dst = zone.dst(midday) or timedelta(0)
    standard_offset = zone.utcoffset(midday) - dst
    if standard_offset is None:
        standard_offset = timedelta(0)
    standard_zone = timezone(standard_offset)
    start_local = datetime.combine(target_date, time(0, 0), tzinfo=standard_zone)
    start_utc = start_local.astimezone(timezone.utc)
    end_utc = start_utc + timedelta(days=1)
    return start_utc, end_utc


def build_clock(asof_utc: datetime) -> RuntimeClock:
    target_date = target_date_from_asof(asof_utc)
    from_time, until_time = target_window_utc(target_date)
    return RuntimeClock(
        asof_utc=asof_utc,
        target_date_local=target_date,
        from_time_utc=from_time,
        until_time_utc=until_time,
    )
