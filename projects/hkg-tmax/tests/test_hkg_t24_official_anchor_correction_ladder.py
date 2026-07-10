from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_hkg_t24_official_anchor_correction_ladder import (
    bucket_correction_prediction,
    mean_correction_prediction,
    prediction_score_row,
)


def test_mean_correction_prediction_uses_prior_rows_only() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "target_tmax_c": [11.0, 11.0, 11.0, 100.0, 11.0],
            "forecast_max_c": [10.0, 10.0, 10.0, 10.0, 10.0],
            "forecast_source_family": ["rss"] * 5,
            "season": ["DJF"] * 5,
            "month": [1] * 5,
        }
    )

    out = mean_correction_prediction(
        frame,
        group_cols=["forecast_source_family"],
        fallback_group_cols=[[]],
        min_history=1,
        min_group=1,
    )

    assert pd.isna(out.loc[0, "candidate_prediction_c"])
    assert out.loc[1, "candidate_prediction_c"] == pytest.approx(11.0)
    assert out.loc[3, "candidate_prediction_c"] == pytest.approx(11.0)
    assert out.loc[4, "candidate_prediction_c"] == pytest.approx(33.25)


def test_bucket_correction_prediction_fits_buckets_from_prior_rows_only() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.date_range("2020-01-01", periods=6, freq="D"),
            "target_tmax_c": [10.0, 12.0, 10.0, 12.0, 99.0, 12.0],
            "forecast_max_c": [10.0] * 6,
            "forecast_source_family": ["rss"] * 6,
            "season": ["DJF"] * 6,
            "feature_x": [0.0, 10.0, 0.0, 10.0, 10.0, 10.0],
        }
    )

    out = bucket_correction_prediction(
        frame,
        feature="feature_x",
        bins=2,
        pool_mode="same_source",
        season_conditioned=False,
        min_history=2,
        min_group=1,
    )

    assert pd.isna(out.loc[0, "candidate_prediction_c"])
    assert out.loc[2, "candidate_prediction_c"] == pytest.approx(10.0)
    assert out.loc[4, "candidate_prediction_c"] == pytest.approx(12.0)
    assert out.loc[5, "candidate_prediction_c"] == pytest.approx(41.0)


def test_prediction_score_row_reports_delta_against_same_rows() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.date_range("2020-01-01", periods=2, freq="D"),
            "target_tmax_c": [10.0, 12.0],
            "forecast_max_c": [11.0, 11.0],
            "candidate_prediction_c": [10.0, 12.0],
        }
    )

    row = prediction_score_row(
        frame,
        prediction_col="candidate_prediction_c",
        candidate_id="unit",
        mechanism="unit",
        detail="unit",
    )

    assert row["mae"] == 0.0
    assert row["official_same_rows_mae"] == 1.0
    assert row["delta_vs_official_same_rows"] == -1.0
