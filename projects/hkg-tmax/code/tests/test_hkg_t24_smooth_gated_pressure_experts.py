from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_hkg_t24_smooth_gated_pressure_experts import (
    SmoothExpertSpec,
    past_only_smooth_predictions,
    smooth_residual_correction,
)


def test_smooth_residual_correction_uses_neighbors_and_clips() -> None:
    x_prior = np.array([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0]])
    residual_prior = np.array([0.0, 10.0, 50.0])
    x_current = np.array([10.0, 10.0])

    correction, used, mean_distance = smooth_residual_correction(
        x_prior,
        residual_prior,
        x_current,
        k_neighbors=1,
        shrinkage=0.0,
        correction_clip_c=2.0,
    )

    assert used == 1
    assert correction == pytest.approx(2.0)
    assert np.isfinite(mean_distance)


def test_past_only_smooth_predictions_excludes_current_target_date_label() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "forecast_source_family": ["rss"] * 5,
            "target_tmax_c": [10.0, 10.0, 100.0, 10.0, 10.0],
            "forecast_max_c": [10.0] * 5,
            "monsoon_phase": ["northeast_monsoon"] * 5,
            "feature_a": [0.0, 1.0, 100.0, 1.0, 1.0],
            "feature_b": [0.0, 1.0, 100.0, 1.0, 1.0],
        }
    )
    spec = SmoothExpertSpec(
        name="leak_guard",
        features=("feature_a", "feature_b"),
        k_neighbors=1,
        same_source=False,
        phase_conditioned=False,
        shrinkage=0.0,
        min_history=2,
    )

    out = past_only_smooth_predictions(frame, spec)

    assert out.loc[2, "past_rows_used"] == 1
    assert out.loc[2, "residual_correction_c"] == pytest.approx(0.0)
    assert out.loc[2, "candidate_prediction_c"] == pytest.approx(10.0)


def test_past_only_smooth_predictions_same_source_isolates_history() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.date_range("2020-01-01", periods=6, freq="D"),
            "forecast_source_family": ["rss", "press", "rss", "press", "rss", "press"],
            "target_tmax_c": [30.0, 11.0, 30.0, 11.0, 30.0, 11.0],
            "forecast_max_c": [10.0] * 6,
            "monsoon_phase": ["northeast_monsoon"] * 6,
            "feature_a": [0.0, 100.0, 1.0, 101.0, 1.0, 101.0],
            "feature_b": [0.0, 100.0, 1.0, 101.0, 1.0, 101.0],
        }
    )
    spec = SmoothExpertSpec(
        name="same_source",
        features=("feature_a", "feature_b"),
        k_neighbors=1,
        same_source=True,
        phase_conditioned=False,
        shrinkage=0.0,
        correction_clip_c=25.0,
        min_history=2,
    )

    out = past_only_smooth_predictions(frame, spec)

    assert out.loc[4, "forecast_source_family"] == "rss"
    assert out.loc[4, "past_rows_used"] == 1
    assert out.loc[4, "residual_correction_c"] == pytest.approx(20.0)
    assert out.loc[4, "candidate_prediction_c"] == pytest.approx(30.0)
