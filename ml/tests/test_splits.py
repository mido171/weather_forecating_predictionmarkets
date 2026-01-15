"""Tests for time-based splits."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from weather_ml import splits


def test_filter_date_ranges_excludes_gap() -> None:
    df = pd.DataFrame(
        {
            "station_id": ["KAAA", "KAAA", "KAAA"],
            "target_date_local": ["2025-01-30", "2025-01-31", "2025-02-01"],
        }
    )
    result = splits.filter_date_ranges(
        df,
        train_start=date(2025, 1, 29),
        train_end=date(2025, 1, 31),
        test_start=date(2025, 2, 1),
        test_end=date(2025, 2, 2),
        gap_dates=[date(2025, 1, 31)],
        val_start=None,
        val_end=None,
        validation_enabled=False,
    )
    train_dates = set(result.train_df["target_date_local"].tolist())
    assert date(2025, 1, 31) not in train_dates
    assert date(2025, 1, 30) in train_dates
    test_dates = set(result.test_df["target_date_local"].tolist())
    assert date(2025, 2, 1) in test_dates


def test_time_cv_gap_respected() -> None:
    start = date(2025, 1, 1)
    dates = [start + timedelta(days=i) for i in range(10)]
    df = pd.DataFrame(
        {
            "station_id": ["KAAA"] * len(dates),
            "target_date_local": dates,
        }
    )
    gap_days = 1
    splits_list = splits.make_time_cv_splits(df, n_splits=3, gap_days=gap_days)
    assert splits_list
    dates_series = pd.to_datetime(df["target_date_local"]).dt.date
    for train_idx, val_idx in splits_list:
        train_max = max(dates_series.iloc[train_idx])
        val_min = min(dates_series.iloc[val_idx])
        assert (val_min - train_max).days >= gap_days
