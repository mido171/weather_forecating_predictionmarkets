from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_hkg_t24_remaining_0075_error_feature_mining import (
    feature_family,
    feature_scoreboard,
    numeric_feature_columns,
    quantile_bucket_stats,
)


def test_numeric_feature_columns_excludes_current_target_and_errors() -> None:
    frame = pd.DataFrame(
        {
            "target_tmax_c": [1.0],
            "target_tmax_c_current": [1.0],
            "target_tmax_c_feature": [1.0],
            "current_target_tmax_c": [1.0],
            "abs_error_0075_c": [0.1],
            "residual_0075_c": [0.1],
            "igra_temp_850hpa_c": [10.0],
            "isd_station_air_temperature_c_450110_99999": [20.0],
        }
    )

    columns = numeric_feature_columns(frame)

    assert "target_tmax_c" not in columns
    assert "target_tmax_c_current" not in columns
    assert "target_tmax_c_feature" not in columns
    assert "current_target_tmax_c" not in columns
    assert "abs_error_0075_c" not in columns
    assert "igra_temp_850hpa_c" in columns


def test_feature_family_classifies_core_sources() -> None:
    assert feature_family("igra_temp_850hpa_c") == "upper_air_igra"
    assert feature_family("isd_station_air_temperature_c_450110_99999") == "regional_isd_station"
    assert feature_family("target_lag7_tmax_c") == "target_memory_calendar"
    assert feature_family("daily_waglan_island_sea_temperature_lag7_roll7") == "hko_daily_climate"


def test_quantile_bucket_stats_finds_spread() -> None:
    values = pd.Series(np.arange(400, dtype=float))
    target = pd.Series(np.r_[np.zeros(200), np.ones(200)])

    stats = quantile_bucket_stats(values, target)

    assert stats["bucket_count"] > 0
    assert float(stats["bucket_spread"]) > 0.5


def test_feature_scoreboard_ranks_error_association() -> None:
    values = np.arange(400, dtype=float)
    frame = pd.DataFrame(
        {
            "feature_signal": values,
            "feature_noise": np.ones(400),
            "residual_0075_c": values / 100.0,
            "abs_error_0075_c": values / 100.0,
        }
    )

    scoreboard = feature_scoreboard(frame, ["feature_signal", "feature_noise"], min_rows=100)

    assert scoreboard.iloc[0]["feature"] == "feature_signal"
