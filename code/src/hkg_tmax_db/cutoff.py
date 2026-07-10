"""Canonical T-24 cutoff logic."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from zoneinfo import ZoneInfo

HONG_KONG_TZ = ZoneInfo("Asia/Hong_Kong")
CUTOFF_RULE_VERSION = "hkg_t24_1500hkt_v1"


class AvailabilityGrade(StrEnum):
    """Point-in-time evidence grades for strict T-24 eligibility."""

    A_EXACT_FIRST_SEEN = "A_EXACT_FIRST_SEEN"
    B_PROVIDER_SCHEDULE_PROVEN = "B_PROVIDER_SCHEDULE_PROVEN"
    C_RUN_TIME_ONLY = "C_RUN_TIME_ONLY"
    D_RETROSPECTIVE_ONLY = "D_RETROSPECTIVE_ONLY"
    E_REJECTED = "E_REJECTED"


def hkg_t24_cutoff_utc(target_date: date) -> datetime:
    """Return 15:00 HKT on T-1 as a timezone-aware UTC datetime."""
    local_cutoff = datetime.combine(target_date - timedelta(days=1), time(15, 0), HONG_KONG_TZ)
    return local_cutoff.astimezone(UTC)


def assert_hong_kong_fixed_utc8(sample_years: tuple[int, ...] = (2000, 2024, 2026)) -> None:
    """Fail if Python timezone data ever reports Hong Kong as not fixed UTC+08."""
    for year in sample_years:
        offset = datetime(year, 6, 1, 12, 0, tzinfo=HONG_KONG_TZ).utcoffset()
        if offset != timedelta(hours=8):
            raise AssertionError(f"Asia/Hong_Kong offset changed for {year}: {offset}")


def is_strictly_eligible(
    *,
    available_at_utc: datetime | None,
    cutoff_utc: datetime,
    grade: AvailabilityGrade,
) -> bool:
    """Return whether a row can enter strict production-style scoring."""
    if available_at_utc is None:
        return False
    if available_at_utc.tzinfo is None or cutoff_utc.tzinfo is None:
        raise ValueError("available_at_utc and cutoff_utc must be timezone-aware UTC datetimes")
    if grade not in {
        AvailabilityGrade.A_EXACT_FIRST_SEEN,
        AvailabilityGrade.B_PROVIDER_SCHEDULE_PROVEN,
    }:
        return False
    return available_at_utc.astimezone(UTC) <= cutoff_utc.astimezone(UTC)
