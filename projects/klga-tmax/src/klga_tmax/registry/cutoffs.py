from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from klga_tmax.constants import TARGET_TZ, TRADER_TZ


@dataclass(frozen=True)
class CutoffSpec:
    cutoff_id: str
    cutoff_order: int
    timezone_name: str
    local_time: time
    target_day_offset: int
    description: str


CANONICAL_CUTOFFS: tuple[CutoffSpec, ...] = (
    CutoffSpec(
        cutoff_id="T_MINUS_1_STOCKHOLM_1500",
        cutoff_order=1,
        timezone_name=TRADER_TZ,
        local_time=time(15, 0),
        target_day_offset=-1,
        description="Early alpha cut at 15:00 Stockholm on target date minus one day.",
    ),
    CutoffSpec(
        cutoff_id="T_MINUS_1_STOCKHOLM_1915",
        cutoff_order=2,
        timezone_name=TRADER_TZ,
        local_time=time(19, 15),
        target_day_offset=-1,
        description="Main T-1 cut at 19:15 Stockholm on target date minus one day.",
    ),
    CutoffSpec(
        cutoff_id="T_MINUS_1_STOCKHOLM_2230",
        cutoff_order=3,
        timezone_name=TRADER_TZ,
        local_time=time(22, 30),
        target_day_offset=-1,
        description="Late T-1 cut at 22:30 Stockholm on target date minus one day.",
    ),
    CutoffSpec(
        cutoff_id="PRE_LOCAL_DAY_NYC_2350",
        cutoff_order=4,
        timezone_name=TARGET_TZ,
        local_time=time(23, 50),
        target_day_offset=-1,
        description="Best pre-local-day cut at 23:50 New York on target date minus one day.",
    ),
    CutoffSpec(
        cutoff_id="T_MINUS_1_2045UTC",
        cutoff_order=5,
        timezone_name="UTC",
        local_time=time(20, 45),
        target_day_offset=-1,
        description=(
            "Canonical single GribStream snapshot cut at 20:45 UTC on target date minus one day; "
            "summer display alias is 22:45 Stockholm / 16:45 New York."
        ),
    ),
    CutoffSpec(
        cutoff_id="T_1245UTC",
        cutoff_order=6,
        timezone_name="UTC",
        local_time=time(12, 45),
        target_day_offset=0,
        description=(
            "Target-day GribStream snapshot cut at 12:45 UTC; "
            "summer display alias is 14:45 Stockholm / 08:45 New York."
        ),
    ),
)


def cutoff_by_id(cutoff_id: str) -> CutoffSpec:
    for cutoff in CANONICAL_CUTOFFS:
        if cutoff.cutoff_id == cutoff_id:
            return cutoff
    raise KeyError(f"unknown cutoff_id: {cutoff_id}")


def cutoff_timestamp_utc(target_date: date, cutoff: CutoffSpec) -> datetime:
    local_date = target_date + timedelta(days=cutoff.target_day_offset)
    local_dt = datetime.combine(local_date, cutoff.local_time, ZoneInfo(cutoff.timezone_name))
    return local_dt.astimezone(timezone.utc)


def target_local_day_window_utc(target_date: date) -> tuple[datetime, datetime]:
    target_zone = ZoneInfo(TARGET_TZ)
    start_local = datetime.combine(target_date, time(0, 0), target_zone)
    end_local = datetime.combine(target_date + timedelta(days=1), time(0, 0), target_zone)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def materialized_cutoff_rows(target_date: date) -> list[dict[str, object]]:
    start_utc, end_utc = target_local_day_window_utc(target_date)
    rows: list[dict[str, object]] = []
    for cutoff in CANONICAL_CUTOFFS:
        rows.append(
            {
                "target_date": target_date,
                "cutoff_id": cutoff.cutoff_id,
                "cutoff_utc": cutoff_timestamp_utc(target_date, cutoff),
                "local_day_start_utc": start_utc,
                "local_day_end_utc": end_utc,
            }
        )
    return rows


def sample_dst_and_non_dst_dates() -> list[date]:
    dates = [
        date(2026, 1, 15),
        date(2026, 2, 15),
        date(2026, 3, 8),
        date(2026, 3, 9),
        date(2026, 4, 15),
        date(2026, 5, 15),
        date(2026, 6, 28),
        date(2026, 7, 15),
        date(2026, 8, 15),
        date(2026, 9, 15),
        date(2026, 10, 15),
        date(2026, 11, 1),
        date(2026, 11, 2),
        date(2026, 12, 15),
    ]
    dates.extend(date(2025, month, 10) for month in range(1, 13))
    dates.extend(date(2027, month, 20) for month in range(1, 5))
    return dates
