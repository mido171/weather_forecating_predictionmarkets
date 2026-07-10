"""Time and partition rules for the HKG H24N contract."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from .constants import CUTOFF_ID, SNAPSHOT_ID_PREFIX

HONG_KONG_TZ = ZoneInfo("Asia/Hong_Kong")


@dataclass(frozen=True)
class CalendarRow:
    """Deterministic cutoff-calendar row."""

    target_date_hkt: date
    cutoff_id: str
    formal_cutoff_utc: datetime
    operational_freeze_utc: datetime
    partition_name: str
    snapshot_id: str
    season: str
    month: int
    day_of_year: int
    is_mam: bool
    is_jja: bool
    is_son: bool
    is_djf: bool
    year_index: int


def formal_cutoff_utc(target_date_hkt: date) -> datetime:
    """Return 15:00 HKT on T-1 as UTC."""
    local_cutoff = datetime.combine(target_date_hkt - timedelta(days=1), time(15, 0), HONG_KONG_TZ)
    return local_cutoff.astimezone(UTC)


def operational_freeze_utc(target_date_hkt: date) -> datetime:
    """Return 14:45 HKT on T-1 as UTC."""
    local_freeze = datetime.combine(target_date_hkt - timedelta(days=1), time(14, 45), HONG_KONG_TZ)
    return local_freeze.astimezone(UTC)


def partition_for_target_date(target_date_hkt: date) -> str:
    """Return the contract partition for a target date."""
    if target_date_hkt < date(2024, 1, 1):
        return "pre2024_development"
    if target_date_hkt.year == 2024:
        return "sealed_2024"
    if target_date_hkt.year == 2025:
        return "sealed_2025"
    return "prospective_2026"


def season_for_month(month: int) -> str:
    if month in {3, 4, 5}:
        return "MAM"
    if month in {6, 7, 8}:
        return "JJA"
    if month in {9, 10, 11}:
        return "SON"
    return "DJF"


def snapshot_id(target_date_hkt: date) -> str:
    return f"{SNAPSHOT_ID_PREFIX}{target_date_hkt.isoformat()}"


def calendar_row(target_date_hkt: date) -> CalendarRow:
    season = season_for_month(target_date_hkt.month)
    return CalendarRow(
        target_date_hkt=target_date_hkt,
        cutoff_id=CUTOFF_ID,
        formal_cutoff_utc=formal_cutoff_utc(target_date_hkt),
        operational_freeze_utc=operational_freeze_utc(target_date_hkt),
        partition_name=partition_for_target_date(target_date_hkt),
        snapshot_id=snapshot_id(target_date_hkt),
        season=season,
        month=target_date_hkt.month,
        day_of_year=target_date_hkt.timetuple().tm_yday,
        is_mam=season == "MAM",
        is_jja=season == "JJA",
        is_son=season == "SON",
        is_djf=season == "DJF",
        year_index=target_date_hkt.year - 2000,
    )


def iter_target_dates(start_date: date, end_date: date) -> list[date]:
    """Return inclusive target dates."""
    if end_date < start_date:
        raise ValueError("end_date must be >= start_date")
    days = (end_date - start_date).days
    return [start_date + timedelta(days=offset) for offset in range(days + 1)]


def assert_hong_kong_fixed_utc8(sample_years: tuple[int, ...] = (2000, 2024, 2026)) -> None:
    """Fail if Python timezone data no longer treats Hong Kong as fixed UTC+08."""
    for year in sample_years:
        offset = datetime(year, 6, 1, 12, 0, tzinfo=HONG_KONG_TZ).utcoffset()
        if offset != timedelta(hours=8):
            raise AssertionError(f"Asia/Hong_Kong offset changed for {year}: {offset}")
