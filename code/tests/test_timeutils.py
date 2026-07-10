from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

import pytest

from hkg_tmax.timeutils import (
    TimeContractError,
    asof_eligible,
    cutoff_for_local_date,
    parse_iso_aware,
)


def test_h24n_cutoff_is_previous_day_15_hkt() -> None:
    cutoff = cutoff_for_local_date(date(2026, 6, 19), "H24N")
    assert cutoff.isoformat() == "2026-06-18T15:00:00+08:00"
    assert cutoff.astimezone(UTC).isoformat() == "2026-06-18T07:00:00+00:00"


def test_asof_eligibility_normalizes_timezones() -> None:
    cutoff = datetime(2026, 6, 18, 15, tzinfo=ZoneInfo("Asia/Hong_Kong"))
    available = datetime(2026, 6, 18, 6, 59, tzinfo=UTC)
    unavailable = datetime(2026, 6, 18, 7, 1, tzinfo=UTC)
    assert asof_eligible(available, cutoff)
    assert not asof_eligible(unavailable, cutoff)


def test_naive_datetime_is_rejected() -> None:
    with pytest.raises(TimeContractError):
        parse_iso_aware("2026-06-18T12:00:00")


def test_unknown_horizon_is_rejected() -> None:
    with pytest.raises(TimeContractError):
        cutoff_for_local_date(date(2026, 6, 19), "T24")
