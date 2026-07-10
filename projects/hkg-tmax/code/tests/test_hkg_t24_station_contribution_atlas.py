from __future__ import annotations

import math

import numpy as np
import pandas as pd

from scripts.run_hkg_t24_station_contribution_atlas import (
    STATION_ATTRIBUTES,
    bearing_degrees,
    haversine_km,
    tertile_spread,
)


def test_station_attributes_exclude_full_day_extrema() -> None:
    assert "daily_air_temperature_max_c" not in STATION_ATTRIBUTES
    assert "daily_air_temperature_min_c" not in STATION_ATTRIBUTES
    assert all("latest_before_1500" in item or item.endswith("_minus_network_median") for item in STATION_ATTRIBUTES)


def test_haversine_and_bearing_basic_geometry() -> None:
    assert haversine_km(22.301944, 114.174167, 22.301944, 114.174167) == 0.0
    east_bearing = bearing_degrees(22.301944, 114.174167, 22.301944, 115.174167)
    assert 89.0 <= east_bearing <= 91.0


def test_tertile_spread_uses_train_edges_and_eval_outcomes() -> None:
    dates = pd.date_range("1990-01-01", periods=3600, freq="D")
    cyclic_values = np.tile(np.arange(300), 12)
    values = pd.Series(cyclic_values, index=dates, dtype=float)
    outcome = pd.Series(cyclic_values, index=dates, dtype=float)
    train_mask = pd.Series(dates <= pd.Timestamp("1997-12-31"), index=dates)
    eval_mask = pd.Series(dates > pd.Timestamp("1997-12-31"), index=dates)
    result = tertile_spread(values, outcome, train_mask, eval_mask, min_rows=30)
    assert result["valid_cells"] >= 2
    assert isinstance(result["cell_spread"], float)
    assert math.isfinite(result["cell_spread"])
