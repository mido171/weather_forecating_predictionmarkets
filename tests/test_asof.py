from datetime import UTC, datetime

import pytest

from hkg_tmax.asof import (
    AsOfError,
    TemporalRecord,
    assert_no_target_columns,
    latest_available_by_key,
)


def test_latest_available_vintage_is_selected_without_future() -> None:
    valid = datetime(2026, 6, 19, 15, tzinfo=UTC)
    records = [
        TemporalRecord(
            "HKO",
            "t2m_forecast",
            valid,
            datetime(2026, 6, 18, 4, tzinfo=UTC),
            30.0,
        ),
        TemporalRecord(
            "HKO",
            "t2m_forecast",
            valid,
            datetime(2026, 6, 18, 6, tzinfo=UTC),
            31.0,
        ),
        TemporalRecord(
            "HKO",
            "t2m_forecast",
            valid,
            datetime(2026, 6, 18, 8, tzinfo=UTC),
            32.0,
        ),
    ]
    selected = latest_available_by_key(
        records,
        cutoff_at=datetime(2026, 6, 18, 7, tzinfo=UTC),
    )
    assert next(iter(selected.values())).value == 31.0


def test_target_column_guard() -> None:
    assert_no_target_columns([{"wind": 2.0, "cloud": 0.3}])
    with pytest.raises(AsOfError):
        assert_no_target_columns([{"wind": 2.0, "target_value": 31.4}])
