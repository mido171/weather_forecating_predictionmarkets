from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_hkg_t24_official_anchor_analog_screen import (
    analog_correction_prediction,
    weighted_mean,
)


def test_weighted_mean_prefers_closer_analogs() -> None:
    values = np.array([0.0, 10.0])
    distances = np.array([1.0, 9.0])

    assert weighted_mean(values, distances, weighted=False) == pytest.approx(5.0)
    assert weighted_mean(values, distances, weighted=True) < 2.0


def test_analog_correction_prediction_uses_only_prior_rows() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.date_range("2020-01-01", periods=6, freq="D"),
            "target_tmax_c": [10.0, 11.0, 10.0, 11.0, 99.0, 11.0],
            "forecast_max_c": [10.0] * 6,
            "forecast_source_family": ["rss"] * 6,
            "season": ["DJF"] * 6,
            "f1": [0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            "f2": [0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        }
    )

    out = analog_correction_prediction(
        frame,
        features=["f1", "f2"],
        k=3,
        pool_mode="same_source",
        season_conditioned=False,
        weighted=False,
        min_history=2,
        min_analogs=1,
    )

    assert pd.isna(out.loc[0, "candidate_prediction_c"])
    assert out.loc[2, "candidate_prediction_c"] == pytest.approx(10.5)
    assert out.loc[4, "candidate_prediction_c"] == pytest.approx(10.6666666667)
    assert out.loc[5, "candidate_prediction_c"] > 30.0
