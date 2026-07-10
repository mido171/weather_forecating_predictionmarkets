from __future__ import annotations

import pandas as pd

from scripts.monitor_hko_official_backfill import gap_rows_from_dates, target_coverage


def test_gap_rows_from_dates_includes_start_middle_and_end_gaps() -> None:
    gaps = gap_rows_from_dates(
        pd.Series(["2000-01-03", "2000-01-04", "2000-01-07"]),
        start=pd.Timestamp("2000-01-01"),
        end=pd.Timestamp("2000-01-08"),
    )

    assert gaps.to_dict("records") == [
        {"gap_id": 1, "missing_start": "2000-01-01", "missing_end": "2000-01-02", "missing_days": 2},
        {"gap_id": 2, "missing_start": "2000-01-05", "missing_end": "2000-01-06", "missing_days": 2},
        {"gap_id": 3, "missing_start": "2000-01-08", "missing_end": "2000-01-08", "missing_days": 1},
    ]


def test_target_coverage_requires_all_dates_and_one_segment() -> None:
    selected = pd.DataFrame({"target_date": ["2000-01-01", "2000-01-02", "2000-01-03"]})

    complete = target_coverage(selected, start=pd.Timestamp("2000-01-01"), end=pd.Timestamp("2000-01-03"))
    incomplete = target_coverage(selected, start=pd.Timestamp("2000-01-01"), end=pd.Timestamp("2000-01-04"))

    assert complete["complete"] is True
    assert complete["missing_target_days"] == 0
    assert complete["segment_count"] == 1
    assert incomplete["complete"] is False
    assert incomplete["missing_target_days"] == 1
