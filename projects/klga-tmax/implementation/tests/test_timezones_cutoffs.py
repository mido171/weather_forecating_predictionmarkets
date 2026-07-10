from __future__ import annotations

from datetime import date, timezone

from klga_tmax.registry.cutoffs import (
    CANONICAL_CUTOFFS,
    cutoff_timestamp_utc,
    sample_dst_and_non_dst_dates,
    target_local_day_window_utc,
)
from klga_tmax.validation.foundation import EXPECTED_2026_06_28_CUTOFF_UTC


def test_thirty_dst_and_non_dst_cutoff_conversions_are_aware_utc() -> None:
    dates = sample_dst_and_non_dst_dates()
    assert len(dates) >= 30
    for target_date in dates:
        for cutoff in CANONICAL_CUTOFFS:
            cutoff_utc = cutoff_timestamp_utc(target_date, cutoff)
            assert cutoff_utc.tzinfo is timezone.utc
            assert cutoff_utc.utcoffset().total_seconds() == 0


def test_2026_06_28_cutoff_examples_match_spec() -> None:
    target_date = date(2026, 6, 28)
    observed = {
        cutoff.cutoff_id: cutoff_timestamp_utc(target_date, cutoff).isoformat()
        for cutoff in CANONICAL_CUTOFFS
    }
    assert observed == {
        "T_MINUS_1_STOCKHOLM_1500": "2026-06-27T13:00:00+00:00",
        "T_MINUS_1_STOCKHOLM_1915": "2026-06-27T17:15:00+00:00",
        "T_MINUS_1_STOCKHOLM_2230": "2026-06-27T20:30:00+00:00",
        "PRE_LOCAL_DAY_NYC_2350": "2026-06-28T03:50:00+00:00",
        "T_MINUS_1_2045UTC": "2026-06-27T20:45:00+00:00",
        "T_1245UTC": "2026-06-28T12:45:00+00:00",
    }


def test_foundation_validation_example_covers_all_canonical_cutoffs() -> None:
    assert set(EXPECTED_2026_06_28_CUTOFF_UTC) == {
        cutoff.cutoff_id for cutoff in CANONICAL_CUTOFFS
    }


def test_target_local_day_window_handles_dst_lengths() -> None:
    spring_start, spring_end = target_local_day_window_utc(date(2026, 3, 8))
    fall_start, fall_end = target_local_day_window_utc(date(2026, 11, 1))
    normal_start, normal_end = target_local_day_window_utc(date(2026, 12, 15))

    assert (spring_end - spring_start).total_seconds() / 3600 == 23
    assert (fall_end - fall_start).total_seconds() / 3600 == 25
    assert (normal_end - normal_start).total_seconds() / 3600 == 24


def test_pre_local_day_cutoff_is_before_target_local_midnight() -> None:
    target_date = date(2026, 6, 28)
    start_utc, _ = target_local_day_window_utc(target_date)
    pre_local_cutoff = next(c for c in CANONICAL_CUTOFFS if c.cutoff_id == "PRE_LOCAL_DAY_NYC_2350")
    assert cutoff_timestamp_utc(target_date, pre_local_cutoff) < start_utc
