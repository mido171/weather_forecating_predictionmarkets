from __future__ import annotations

from datetime import date, datetime

import pytest

from hkg_tmax.hkg_t24.guard import (
    LOCKED_TEST_START,
    LockedTestAccessError,
    assert_no_locked_dates,
    coerce_local_date,
    locked_test_violations,
)


def test_locked_test_start_is_frozen() -> None:
    assert date(2025, 1, 1) == LOCKED_TEST_START


def test_coerce_local_date_accepts_common_scalars() -> None:
    assert coerce_local_date("2024-12-31") == date(2024, 12, 31)
    assert coerce_local_date(datetime(2024, 12, 31, 15, 0)) == date(2024, 12, 31)
    assert coerce_local_date(date(2024, 12, 31)) == date(2024, 12, 31)


def test_locked_test_violations_detects_2025_and_later() -> None:
    violations = locked_test_violations(["2024-12-31", "2025-01-01", "2026-05-31"])

    assert [item.target_date for item in violations] == [date(2025, 1, 1), date(2026, 5, 31)]


def test_assert_no_locked_dates_rejects_ordinary_research_access() -> None:
    with pytest.raises(LockedTestAccessError, match="ordinary research"):
        assert_no_locked_dates(["2024-12-31", "2025-01-01"], context="ordinary research")


def test_assert_no_locked_dates_allows_development_and_validation() -> None:
    assert_no_locked_dates(["2021-12-30", "2024-12-31"], context="validation reproduction")
