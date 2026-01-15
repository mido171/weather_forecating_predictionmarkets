"""Tests for feature engineering."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from weather_ml import features


def _config():
    climatology = SimpleNamespace(enabled=False, label_lag_days=2, rolling_windows_days=[])
    feature_cfg = SimpleNamespace(
        base_features=[
            "gfs_tmax_f",
            "nam_tmax_f",
            "gefsatmosmean_tmax_f",
            "rap_tmax_f",
            "hrrr_tmax_f",
            "nbm_tmax_f",
            "gefsatmos_tmp_spread_f",
        ],
        ensemble_stats=True,
        pairwise_deltas=False,
        model_vs_ens_deltas=False,
        calendar=False,
        station_onehot=False,
        pairwise_pairs=[],
        climatology=climatology,
    )
    return SimpleNamespace(features=feature_cfg)


def test_ensemble_stats_and_missing_indicators() -> None:
    df = pd.DataFrame(
        {
            "station_id": ["KAAA", "KAAA"],
            "target_date_local": ["2025-01-02", "2025-01-03"],
            "asof_utc": ["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"],
            "gfs_tmax_f": [70.0, np.nan],
            "nam_tmax_f": [71.0, 72.0],
            "gefsatmosmean_tmax_f": [69.0, 70.0],
            "rap_tmax_f": [72.0, 73.0],
            "hrrr_tmax_f": [71.0, 74.0],
            "nbm_tmax_f": [70.0, 71.0],
            "gefsatmos_tmp_spread_f": [2.0, 2.5],
            "actual_tmax_f": [75.0, 76.0],
        }
    )
    config = _config()
    X_df, y, state = features.build_features(df, config=config, training=True)
    assert y is not None
    assert "ens_mean" in X_df.columns
    expected_mean = np.mean([70.0, 71.0, 69.0, 72.0, 71.0, 70.0])
    assert np.isclose(X_df.loc[0, "ens_mean"], expected_mean)
    assert X_df.loc[1, "gfs_tmax_f_missing"] == 1.0
    assert X_df.isna().sum().sum() == 0

    X_df_2, _, _ = features.build_features(df, config=config, fit_state=state)
    assert X_df_2.equals(X_df)
