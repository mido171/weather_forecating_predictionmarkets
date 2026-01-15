"""Dataset validation rules for the CSV pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class ValidationRules:
    required_columns: list[str]
    allowed_columns: list[str] | None
    forecast_min_f: float
    forecast_max_f: float
    spread_min_f: float
    require_asof_not_after_target: bool


def validate_schema(df: pd.DataFrame, rules: ValidationRules) -> None:
    missing = [col for col in rules.required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if rules.allowed_columns is not None:
        extra = [col for col in df.columns if col not in rules.allowed_columns]
        if extra:
            raise ValueError(f"Unexpected columns present: {extra}")


def validate_keys_unique(df: pd.DataFrame, key_cols: Iterable[str]) -> None:
    if df.empty:
        return
    duplicates = df.duplicated(subset=list(key_cols), keep=False)
    if duplicates.any():
        sample = df.loc[duplicates, list(key_cols)].head(10).to_dict(orient="records")
        raise ValueError(f"Duplicate key rows found: {sample}")


def validate_missingness(df: pd.DataFrame, required_non_null: Iterable[str]) -> None:
    for column in required_non_null:
        if df[column].isna().any():
            sample = df.loc[df[column].isna(), ["station_id", "target_date_local"]].head(5)
            raise ValueError(
                f"Column {column} has missing values. Sample rows: "
                f"{sample.to_dict(orient='records')}"
            )


def validate_ranges(df: pd.DataFrame, rules: ValidationRules) -> None:
    feature_cols = [
        "gfs_tmax_f",
        "nam_tmax_f",
        "gefsatmosmean_tmax_f",
        "rap_tmax_f",
        "hrrr_tmax_f",
        "nbm_tmax_f",
        "actual_tmax_f",
    ]
    for column in feature_cols:
        if column in df.columns:
            too_low = df[column] < rules.forecast_min_f
            too_high = df[column] > rules.forecast_max_f
            if (too_low | too_high).any():
                sample = df.loc[too_low | too_high, ["station_id", "target_date_local", column]]
                raise ValueError(
                    f"Column {column} has values outside bounds "
                    f"[{rules.forecast_min_f}, {rules.forecast_max_f}]. "
                    f"Sample: {sample.head(5).to_dict(orient='records')}"
                )
    if "gefsatmos_tmp_spread_f" in df.columns:
        if (df["gefsatmos_tmp_spread_f"] < rules.spread_min_f).any():
            sample = df.loc[
                df["gefsatmos_tmp_spread_f"] < rules.spread_min_f,
                ["station_id", "target_date_local", "gefsatmos_tmp_spread_f"],
            ].head(5)
            raise ValueError(
                "gefsatmos_tmp_spread_f has negative values. "
                f"Sample: {sample.to_dict(orient='records')}"
            )


def validate_asof_alignment(df: pd.DataFrame, require_asof_not_after_target: bool) -> None:
    if df.empty:
        return
    if require_asof_not_after_target:
        asof_date = pd.to_datetime(df["asof_utc"], utc=True, errors="coerce").dt.date
        target_date = pd.to_datetime(df["target_date_local"], errors="coerce").dt.date
        invalid = asof_date > target_date
        if invalid.any():
            sample = df.loc[invalid, ["station_id", "target_date_local", "asof_utc"]].head(5)
            raise ValueError(
                "asof_utc occurs after target_date_local. "
                f"Sample: {sample.to_dict(orient='records')}"
            )


def run_all_validations(df: pd.DataFrame, rules: ValidationRules) -> None:
    validate_schema(df, rules)
    validate_keys_unique(df, ["station_id", "target_date_local", "asof_utc"])
    validate_missingness(df, ["station_id", "target_date_local", "asof_utc", "actual_tmax_f"])
    validate_ranges(df, rules)
    validate_asof_alignment(df, rules.require_asof_not_after_target)


def build_rules_from_config(config) -> ValidationRules:
    required = list(config.features.base_features) + [
        "station_id",
        "target_date_local",
        "asof_utc",
        "actual_tmax_f",
    ]
    required = sorted(set(required))
    allowed = required if config.validation.strict_schema else None
    return ValidationRules(
        required_columns=required,
        allowed_columns=allowed,
        forecast_min_f=config.validation.forecast_min_f,
        forecast_max_f=config.validation.forecast_max_f,
        spread_min_f=config.validation.spread_min_f,
        require_asof_not_after_target=config.validation.require_asof_not_after_target,
    )
