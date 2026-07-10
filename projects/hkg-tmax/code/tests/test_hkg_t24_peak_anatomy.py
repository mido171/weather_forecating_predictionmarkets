from __future__ import annotations

from datetime import datetime, time

import pytest

from hkg_tmax.hkg_t24.peak_anatomy import (
    classify_peak_time,
    count_peak_episodes,
    maximum_heating_in_window,
)


def test_classify_peak_time_uses_fixed_clock_thresholds() -> None:
    assert classify_peak_time(time(11, 59)) == "early_before_1200"
    assert classify_peak_time(time(12, 0)) == "normal_1200_1659"
    assert classify_peak_time(time(16, 59)) == "normal_1200_1659"
    assert classify_peak_time(time(17, 0)) == "late_1700_or_after"


def test_count_peak_episodes_splits_large_gaps() -> None:
    peaks = [
        datetime(2023, 7, 1, 13, 0),
        datetime(2023, 7, 1, 13, 10),
        datetime(2023, 7, 1, 15, 0),
    ]

    assert count_peak_episodes(peaks, gap_minutes=15) == 2


def test_maximum_heating_in_window_returns_largest_prior_rise() -> None:
    times = [
        datetime(2023, 7, 1, 14, 0),
        datetime(2023, 7, 1, 14, 20),
        datetime(2023, 7, 1, 14, 40),
        datetime(2023, 7, 1, 15, 0),
    ]
    values = [29.0, 30.5, 30.1, 31.2]

    assert maximum_heating_in_window(times, values, end_time=times[-1], window_minutes=60) == pytest.approx(2.2)
