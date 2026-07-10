from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_hkg_t24_regime_expert_factory import (
    RegimeExpertSpec,
    add_composite_features,
    past_only_regime_expert_prediction,
)


def test_add_composite_features_creates_forecast_minus_sea_temp() -> None:
    frame = pd.DataFrame(
        {
            "forecast_max_c": [30.0],
            "daily_waglan_island_sea_temperature_lag7_roll7": [27.5],
        }
    )

    out = add_composite_features(frame)

    assert out.loc[0, "forecast_minus_waglan_sea_temp_roll7_c"] == pytest.approx(2.5)


def test_past_only_regime_expert_excludes_current_target_label() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "forecast_source_family": ["press_archive"] * 5,
            "season": ["DJF"] * 5,
            "target_tmax_c": [11.0, 11.0, 100.0, 11.0, 11.0],
            "forecast_max_c": [10.0, 10.0, 10.0, 10.0, 10.0],
            "feature_a": [1.0, 1.1, 1.05, 1.2, 1.3],
        }
    )
    spec = RegimeExpertSpec(
        family="test",
        name="feature_a",
        features=("feature_a",),
        bins=2,
        season_conditioned=False,
        same_source=False,
        statistic="mean",
        min_history=2,
        min_match_rows=1,
    )

    out = past_only_regime_expert_prediction(frame, spec)

    assert out.loc[2, "candidate_prediction_c"] == pytest.approx(11.0)
    assert out.loc[2, "past_rows_used"] == 1


def test_past_only_regime_expert_same_source_uses_only_prior_same_source() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.date_range("2020-01-01", periods=6, freq="D"),
            "forecast_source_family": ["press_archive", "rss_archive", "press_archive", "rss_archive", "press_archive", "rss_archive"],
            "season": ["DJF"] * 6,
            "target_tmax_c": [11.0, 30.0, 11.0, 30.0, 11.0, 30.0],
            "forecast_max_c": [10.0] * 6,
            "feature_a": [1.0, 1.0, 1.1, 1.1, 1.05, 1.05],
        }
    )
    spec = RegimeExpertSpec(
        family="test",
        name="feature_a",
        features=("feature_a",),
        bins=2,
        season_conditioned=False,
        same_source=True,
        statistic="mean",
        min_history=2,
        min_match_rows=1,
    )

    out = past_only_regime_expert_prediction(frame, spec)

    assert out.loc[4, "candidate_prediction_c"] == pytest.approx(11.0)
    assert out.loc[4, "past_rows_used"] == 1
