"""Validation split helpers for HKG Tmax probability experiments."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitWindow:
    name: str
    train_end: str
    validate_start: str
    validate_end: str
    sealed: bool = False


def split_windows_from_config(config: dict) -> list[SplitWindow]:
    governance = config["temporal_governance"]
    windows = [
        SplitWindow(
            name=item["name"],
            train_end=item["train_end"],
            validate_start=item["validate_start"],
            validate_end=item["validate_end"],
        )
        for item in governance["outer_folds"]
    ]
    pre = governance["presealed_holdout"]
    windows.append(SplitWindow(pre["name"], pre["train_end"], pre["validate_start"], pre["validate_end"]))
    sealed = governance["sealed_confirmation"]
    windows.append(SplitWindow(sealed["name"], sealed["train_end"], sealed["validate_start"], sealed["validate_end"], sealed=True))
    return windows


def train_validation_frames(modeling: pd.DataFrame, window: SplitWindow, primary_only: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = modeling.copy()
    if primary_only:
        frame = frame[frame["is_primary_cutoff"]].copy()
    dates = pd.to_datetime(frame["target_date"])
    train = frame[dates <= pd.Timestamp(window.train_end)].copy()
    validation = frame[(dates >= pd.Timestamp(window.validate_start)) & (dates <= pd.Timestamp(window.validate_end))].copy()
    if window.sealed:
        train = train[train["target_table"] != "sealed_confirmation"].copy()
    return train.reset_index(drop=True), validation.reset_index(drop=True)


def sensitivity_validation_frame(modeling: pd.DataFrame, window: SplitWindow) -> pd.DataFrame:
    dates = pd.to_datetime(modeling["target_date"])
    return modeling[(dates >= pd.Timestamp(window.validate_start)) & (dates <= pd.Timestamp(window.validate_end))].copy()
