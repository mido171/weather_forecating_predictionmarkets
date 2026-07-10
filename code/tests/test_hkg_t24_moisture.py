from __future__ import annotations

import pandas as pd

from hkg_tmax.hkg_t24.moisture import (
    dew_point_depression_c,
    magnus_dew_point_c,
    mixing_ratio_g_per_kg,
    saturation_vapor_pressure_hpa,
    stull_wet_bulb_c,
)


def test_magnus_dew_point_is_temperature_at_saturation() -> None:
    temp = pd.Series([25.0, 30.0])
    rh = pd.Series([100.0, 100.0])

    dew = magnus_dew_point_c(temp, rh)

    assert dew.round(6).tolist() == temp.tolist()


def test_dew_point_depression_increases_when_air_is_drier() -> None:
    temp = pd.Series([30.0, 30.0])
    dew = magnus_dew_point_c(temp, pd.Series([80.0, 40.0]))

    depression = dew_point_depression_c(temp, dew)

    assert depression.iloc[1] > depression.iloc[0] > 0


def test_wet_bulb_sits_between_dew_point_and_temperature_for_valid_sample() -> None:
    temp = pd.Series([30.0])
    rh = pd.Series([70.0])
    dew = magnus_dew_point_c(temp, rh)
    wet = stull_wet_bulb_c(temp, rh)

    assert dew.iloc[0] <= wet.iloc[0] <= temp.iloc[0]


def test_mixing_ratio_is_positive_for_hong_kong_like_pressure() -> None:
    vapor = saturation_vapor_pressure_hpa(pd.Series([24.0]))
    ratio = mixing_ratio_g_per_kg(vapor, pd.Series([1008.0]))

    assert 0 < ratio.iloc[0] < 30
