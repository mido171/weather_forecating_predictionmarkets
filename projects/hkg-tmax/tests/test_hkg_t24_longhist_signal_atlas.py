from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_hkg_t24_longhist_signal_atlas import (
    add_analysis_targets,
    build_station_pair_spreads,
    parse_station_feature,
    safe_corr,
    tail_spread,
)


def test_add_analysis_targets_uses_lagged_climatology_and_lagged_target() -> None:
    frame = pd.DataFrame(
        {
            "target_date": ["2020-01-01", "2020-01-02"],
            "target_tmax_c": [25.0, 28.0],
            "clim_constrained_equal_blend_lag7_c": [24.0, 27.5],
            "target_lag7_tmax_c": [23.0, 27.0],
        }
    )

    out = add_analysis_targets(frame)

    assert out["target_anomaly_vs_clim_c"].to_list() == [1.0, 0.5]
    assert out["target_change_vs_lag7_c"].to_list() == [2.0, 1.0]


def test_parse_station_feature_extracts_station_and_metric() -> None:
    parsed = parse_station_feature("isd_station_air_temperature_c_450050_99999")

    assert parsed == ("450050_99999", "air_temperature_c")
    assert parse_station_feature("isd_air_temp_mean_c") is None


def test_safe_corr_requires_enough_rows_and_nonconstant_values() -> None:
    assert np.isnan(safe_corr(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]), min_rows=3))
    assert np.isnan(safe_corr(pd.Series([1.0, 1.0, 1.0]), pd.Series([1.0, 2.0, 3.0]), min_rows=3))
    assert safe_corr(pd.Series([1.0, 2.0, 3.0]), pd.Series([1.0, 2.0, 3.0]), min_rows=3) == pytest.approx(1.0)


def test_tail_spread_uses_high_minus_low_feature_deciles() -> None:
    feature = pd.Series(range(100))
    target = pd.Series([value * 0.5 for value in range(100)])

    n, spread, low_mean, high_mean = tail_spread(feature, target, min_rows=20)

    assert n == 100
    assert spread > 35.0
    assert high_mean > low_mean


def test_build_station_pair_spreads_scores_station_metric_spreads() -> None:
    n = 2100
    frame = pd.DataFrame(
        {
            "target_date": pd.date_range("2010-01-01", periods=n, freq="D"),
            "target_tmax_c": np.linspace(20.0, 30.0, n),
            "target_anomaly_vs_clim_c": np.linspace(-2.0, 2.0, n),
            "target_change_vs_lag7_c": np.sin(np.arange(n) / 30.0),
            "hot_tail_flag": [False] * (n - 210) + [True] * 210,
            "isd_station_air_temperature_c_450050_99999": np.linspace(10.0, 20.0, n),
            "isd_station_air_temperature_c_450070_99999": np.linspace(9.0, 19.0, n) - np.sin(np.arange(n) / 50.0),
        }
    )

    spreads = build_station_pair_spreads(frame)

    assert len(spreads) == 1
    assert spreads.loc[0, "metric"] == "air_temperature_c"
    assert "450050_99999" in spreads.loc[0, "spread_feature"]
