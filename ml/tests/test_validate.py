"""Tests for dataset validation."""

from __future__ import annotations

import pandas as pd
import pytest

from weather_ml import validate


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "station_id": "KMIA",
                "target_date_local": "2025-01-02",
                "asof_utc": "2025-01-01T00:00:00Z",
                "gfs_tmax_f": 70.0,
                "nam_tmax_f": 71.0,
                "gefsatmosmean_tmax_f": 69.5,
                "rap_tmax_f": 72.0,
                "hrrr_tmax_f": 71.5,
                "nbm_tmax_f": 70.5,
                "gefsatmos_tmp_spread_f": 2.0,
                "actual_tmax_f": 73.0,
            }
        ]
    )


def _rules() -> validate.ValidationRules:
    required = [
        "station_id",
        "target_date_local",
        "asof_utc",
        "gfs_tmax_f",
        "nam_tmax_f",
        "gefsatmosmean_tmax_f",
        "rap_tmax_f",
        "hrrr_tmax_f",
        "nbm_tmax_f",
        "gefsatmos_tmp_spread_f",
        "actual_tmax_f",
    ]
    return validate.ValidationRules(
        required_columns=required,
        allowed_columns=required,
        forecast_min_f=-80.0,
        forecast_max_f=140.0,
        spread_min_f=0.0,
        require_asof_not_after_target=True,
    )


def test_validation_passes_on_good_data() -> None:
    df = _sample_df()
    validate.run_all_validations(df, _rules())


def test_validation_duplicate_keys_fail() -> None:
    df = pd.concat([_sample_df(), _sample_df()], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate key rows found"):
        validate.run_all_validations(df, _rules())


def test_validation_missing_column_fails() -> None:
    df = _sample_df().drop(columns=["nam_tmax_f"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate.run_all_validations(df, _rules())


def test_validation_out_of_range_fails() -> None:
    df = _sample_df()
    df.loc[0, "gfs_tmax_f"] = 200.0
    with pytest.raises(ValueError, match="outside bounds"):
        validate.run_all_validations(df, _rules())
