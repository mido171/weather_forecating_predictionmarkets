from __future__ import annotations

from dataclasses import dataclass


BASE_MODELS = [
    "nbm",
    "hrrr",
    "rap",
    "gefsmean",
    "gfsMOS",
    "namMOS",
]

BASE_MODEL_COLS = {
    "nbm": "nbm_tmax_f",
    "hrrr": "hrrr_tmax_f",
    "rap": "rap_tmax_f",
    "gefsmean": "gefsatmosmean_tmax_f",
    "gfsMOS": "gfs_n_x_max",
    "namMOS": "nam_n_x_max",
}

SPREAD_COL = "gefsatmos_tmp_spread_f"
ENSEMBLE_MEAN_COL = "ens_raw_mean"

BIAS_SOURCES = BASE_MODELS + ["ensmean"]

W_BIAS = [3, 5, 7, 14, 21, 30, 60, 120]
W_RESID = [7, 14, 30, 60]
W_COND = [30, 60]

W_SKILL = [7, 14, 30, 60]
W_BMA = [60, 180]
W_RIDGE = [180, 365]
W_AR = [60, 120]

HUBER_K = 3.0
TRUTH_LAG_DAYS = 2


@dataclass(frozen=True)
class RegimeThresholds:
    spread_low: float
    spread_high: float
    disagree_low: float
    disagree_high: float
    anom_low: float
    anom_high: float


@dataclass(frozen=True)
class AnalogScaling:
    means: dict[str, float]
    stds: dict[str, float]


@dataclass(frozen=True)
class E902Metadata:
    thresholds: RegimeThresholds
    analog_scaling: AnalogScaling

