"""Feature policy helpers for official residual-memory experiments."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from hkg_tmax.data.official_residual_memory_features import RESIDUAL_MEMORY_FEATURES
from hkg_tmax.features.pruned_feature_policy import EVALUATION_ONLY_COLUMNS


RESIDUAL_MEMORY_FORBIDDEN_COLUMNS: set[str] = {
    *EVALUATION_ONLY_COLUMNS,
    "y_true_c",
    "residual_y_c",
    "true_residual_c",
    "raw_abs_error_c",
    "candidate_abs_error_c",
    "benefit_c",
    "apply_label",
    "strong_apply_label",
    "helped_vs_raw_flag",
    "worsened_vs_raw_flag",
    "raw_error_decile",
    "raw_abs_error_decile",
}


def residual_memory_feature_names(frame: pd.DataFrame) -> list[str]:
    return [feature for feature in RESIDUAL_MEMORY_FEATURES if feature in frame.columns]


def assert_no_forbidden_residual_memory_predictors(feature_names: Iterable[str]) -> None:
    selected = set(feature_names)
    forbidden = sorted(selected & RESIDUAL_MEMORY_FORBIDDEN_COLUMNS)
    if forbidden:
        raise ValueError(f"Forbidden evaluation or target columns used as predictors: {forbidden}")
    lag1 = sorted(
        feature
        for feature in selected
        if str(feature) == "residual_lag1_c" or str(feature).startswith("residual_lag1_")
    )
    if lag1:
        raise ValueError(f"Lag1 residual memory columns are not allowed without publication proof: {lag1}")


def assert_residual_memory_features_present(frame: pd.DataFrame, feature_names: Iterable[str]) -> None:
    missing = [feature for feature in feature_names if feature not in frame.columns]
    if missing:
        raise ValueError(f"Residual-memory features missing from frame: {missing}")
