from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_hkg_t24_pressure_gradient_experts import (
    RidgeExpertSpec,
    add_pressure_gradient_features,
    fit_ridge_residual,
    past_only_ridge_predictions,
)


def test_add_pressure_gradient_features_builds_spreads_and_slope_magnitude() -> None:
    frame = pd.DataFrame(
        {
            "isd_station_sea_level_pressure_hpa_590960_99999": [1010.0],
            "isd_station_sea_level_pressure_hpa_596730_99999": [1008.5],
            "isd_pressure_plane_lat_slope_hpa_per_deg": [3.0],
            "isd_pressure_plane_lon_slope_hpa_per_deg": [4.0],
        }
    )

    out = add_pressure_gradient_features(frame)

    assert out.loc[0, "slp_590960_minus_596730_hpa"] == pytest.approx(1.5)
    assert out.loc[0, "pressure_plane_slope_magnitude_hpa_per_deg"] == pytest.approx(5.0)


def test_fit_ridge_residual_returns_finite_clipped_correction() -> None:
    x_prior = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    y_prior = np.array([0.0, 4.0, 8.0, 12.0])
    x_current = np.array([10.0, 10.0])

    correction = fit_ridge_residual(
        x_prior,
        y_prior,
        x_current,
        alpha=1.0,
        shrinkage=0.0,
        correction_clip_c=1.25,
    )

    assert np.isfinite(correction)
    assert correction == pytest.approx(1.25)


def test_past_only_ridge_predictions_excludes_current_target_date_label() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "forecast_source_family": ["rss"] * 5,
            "target_tmax_c": [10.0, 10.0, 100.0, 10.0, 10.0],
            "forecast_max_c": [10.0] * 5,
            "pressure_feature_a": [0.0, 1.0, 100.0, 1.0, 1.0],
            "pressure_feature_b": [0.0, 1.0, 100.0, 1.0, 1.0],
        }
    )
    spec = RidgeExpertSpec(
        family="test",
        name="leak_guard",
        features=("pressure_feature_a", "pressure_feature_b"),
        alpha=1.0,
        same_source=False,
        min_history=2,
        shrinkage=0.0,
        correction_clip_c=90.0,
    )

    out = past_only_ridge_predictions(frame, spec)

    assert out.loc[2, "past_rows_used"] == 2
    assert out.loc[2, "residual_correction_c"] == pytest.approx(0.0)
    assert out.loc[2, "candidate_prediction_c"] == pytest.approx(10.0)


def test_past_only_ridge_predictions_same_source_isolates_history() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.date_range("2020-01-01", periods=6, freq="D"),
            "forecast_source_family": ["rss", "press", "rss", "press", "rss", "press"],
            "target_tmax_c": [30.0, 11.0, 30.0, 11.0, 30.0, 11.0],
            "forecast_max_c": [10.0] * 6,
            "pressure_feature_a": [1.0] * 6,
            "pressure_feature_b": [2.0] * 6,
        }
    )
    spec = RidgeExpertSpec(
        family="test",
        name="same_source",
        features=("pressure_feature_a", "pressure_feature_b"),
        alpha=10.0,
        same_source=True,
        min_history=2,
        shrinkage=0.0,
        correction_clip_c=25.0,
    )

    out = past_only_ridge_predictions(frame, spec)

    assert out.loc[4, "forecast_source_family"] == "rss"
    assert out.loc[4, "past_rows_used"] == 2
    assert out.loc[4, "residual_correction_c"] == pytest.approx(20.0)
    assert out.loc[4, "candidate_prediction_c"] == pytest.approx(30.0)
