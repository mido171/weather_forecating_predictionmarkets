"""Time-based splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


@dataclass(frozen=True)
class SplitResult:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


def filter_date_ranges(
    df: pd.DataFrame,
    *,
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date,
    gap_dates: list[date],
    val_start: date | None,
    val_end: date | None,
    validation_enabled: bool,
) -> SplitResult:
    df = df.copy()
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    gap_dates_set = set(gap_dates)
    in_gap = df["target_date_local"].isin(gap_dates_set)

    train_mask = (df["target_date_local"] >= train_start) & (
        df["target_date_local"] <= train_end
    )
    test_mask = (df["target_date_local"] >= test_start) & (
        df["target_date_local"] <= test_end
    )

    if validation_enabled and val_start and val_end:
        val_mask = (df["target_date_local"] >= val_start) & (
            df["target_date_local"] <= val_end
        )
        train_mask = train_mask & ~val_mask
    else:
        val_mask = pd.Series(False, index=df.index)

    train_df = df[train_mask & ~in_gap].copy()
    val_df = df[val_mask & ~in_gap].copy()
    test_df = df[test_mask & ~in_gap].copy()
    return SplitResult(train_df=train_df, val_df=val_df, test_df=test_df)


def make_time_cv_splits(
    df: pd.DataFrame,
    *,
    n_splits: int,
    gap_days: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if df.empty:
        return []
    df = df.copy()
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    unique_dates = np.array(sorted(df["target_date_local"].unique()))
    if unique_dates.size <= n_splits:
        return []
    tss = TimeSeriesSplit(n_splits=n_splits, gap=gap_days)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    date_to_index = {d: idx for idx, d in enumerate(unique_dates)}
    for train_idx, val_idx in tss.split(unique_dates):
        train_dates = set(unique_dates[train_idx])
        val_dates = set(unique_dates[val_idx])
        train_mask = df["target_date_local"].map(lambda d: d in train_dates)
        val_mask = df["target_date_local"].map(lambda d: d in val_dates)
        splits.append(
            (
                df.index[train_mask].to_numpy(),
                df.index[val_mask].to_numpy(),
            )
        )
    return splits
