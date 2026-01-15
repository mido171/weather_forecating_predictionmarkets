"""Tests for derived feature utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from weather_ml import derived_features


def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "nbm_tmax_f": [1.0],
            "hrrr_tmax_f": [2.0],
            "rap_tmax_f": [3.0],
            "gefsatmosmean_tmax_f": [4.0],
            "gfs_tmax_f": [5.0],
            "nam_tmax_f": [100.0],
        }
    )


def test_trimmed_mean_6() -> None:
    df = _base_df()
    features = derived_features.compute_rowwise_features(
        df,
        list(df.columns),
        include=["ens_trimmed_mean_6_1"],
    )
    assert np.isclose(features["ens_trimmed_mean_6_1"].iloc[0], 3.5)


def test_winsor_mean_6() -> None:
    df = _base_df()
    features = derived_features.compute_rowwise_features(
        df,
        list(df.columns),
        include=["ens_winsor_mean_6_1"],
    )
    assert np.isclose(features["ens_winsor_mean_6_1"].iloc[0], 3.5)


def test_mad_6() -> None:
    df = pd.DataFrame(
        {
            "nbm_tmax_f": [1.0],
            "hrrr_tmax_f": [2.0],
            "rap_tmax_f": [3.0],
            "gefsatmosmean_tmax_f": [4.0],
            "gfs_tmax_f": [5.0],
            "nam_tmax_f": [6.0],
        }
    )
    features = derived_features.compute_rowwise_features(
        df,
        list(df.columns),
        include=["ens_mad_6"],
    )
    assert np.isclose(features["ens_mad_6"].iloc[0], 1.5)


def test_rank_tie_break() -> None:
    df = pd.DataFrame(
        {
            "nbm_tmax_f": [5.0],
            "hrrr_tmax_f": [5.0],
            "rap_tmax_f": [7.0],
        }
    )
    features = derived_features.compute_rowwise_features(
        df,
        list(df.columns),
        include=[
            "rank_nbm_tmax_f_in_ens",
            "rank_hrrr_tmax_f_in_ens",
            "rank_rap_tmax_f_in_ens",
        ],
    )
    assert features["rank_nbm_tmax_f_in_ens"].iloc[0] == 1
    assert features["rank_hrrr_tmax_f_in_ens"].iloc[0] == 2
    assert features["rank_rap_tmax_f_in_ens"].iloc[0] == 3


def test_bias_correction() -> None:
    df = pd.DataFrame(
        {
            "nbm_tmax_f": [1.0, 2.0, 3.0],
            "actual_tmax_f": [2.0, 3.0, 4.0],
        }
    )
    bias = derived_features.fit_bias_correction(df, ["nbm_tmax_f"], y_col="actual_tmax_f")
    corrected = derived_features.apply_bias_correction(df, ["nbm_tmax_f"], bias)
    assert np.allclose(corrected["nbm_tmax_f_bc"].to_numpy(), df["nbm_tmax_f"] + 1.0)


def test_reliability_weights_normalized() -> None:
    df = pd.DataFrame(
        {
            "nbm_tmax_f": [1.0, 2.0, 3.0],
            "gfs_tmax_f": [1.0, 2.0, 3.0],
            "actual_tmax_f": [2.0, 2.0, 2.0],
        }
    )
    weights = derived_features.fit_reliability_weights(
        df, ["nbm_tmax_f", "gfs_tmax_f"], y_col="actual_tmax_f"
    )
    total = sum(weights.weights_norm.values())
    assert np.isclose(total, 1.0)
