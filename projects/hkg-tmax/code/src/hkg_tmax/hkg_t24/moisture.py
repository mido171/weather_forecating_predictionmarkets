from __future__ import annotations

from typing import Any

import numpy as np

MAGNUS_A = 17.625
MAGNUS_B_C = 243.04


def clamp_relative_humidity(relative_humidity_pct: Any) -> Any:
    return relative_humidity_pct.astype(float).clip(lower=1.0, upper=100.0)


def magnus_dew_point_c(temperature_c: Any, relative_humidity_pct: Any) -> Any:
    temp = temperature_c.astype(float)
    rh = clamp_relative_humidity(relative_humidity_pct)
    gamma = np.log(rh / 100.0) + (MAGNUS_A * temp) / (MAGNUS_B_C + temp)
    return (MAGNUS_B_C * gamma) / (MAGNUS_A - gamma)


def dew_point_depression_c(temperature_c: Any, dew_point_c: Any) -> Any:
    return temperature_c.astype(float) - dew_point_c.astype(float)


def stull_wet_bulb_c(temperature_c: Any, relative_humidity_pct: Any) -> Any:
    temp = temperature_c.astype(float)
    rh = clamp_relative_humidity(relative_humidity_pct)
    return (
        temp * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(temp + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * np.power(rh, 1.5) * np.arctan(0.023101 * rh)
        - 4.686035
    )


def saturation_vapor_pressure_hpa(dew_point_c: Any) -> Any:
    dew = dew_point_c.astype(float)
    return 6.1094 * np.exp((MAGNUS_A * dew) / (MAGNUS_B_C + dew))


def mixing_ratio_g_per_kg(vapor_pressure_hpa: Any, pressure_hpa: Any) -> Any:
    vapor = vapor_pressure_hpa.astype(float)
    pressure = pressure_hpa.astype(float)
    denominator = (pressure - vapor).where((pressure - vapor) > 0)
    return 621.97 * vapor / denominator
