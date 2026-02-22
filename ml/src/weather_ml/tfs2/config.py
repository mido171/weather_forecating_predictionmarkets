"""Configuration for the TFS2 sweep (DB-backed)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path


GUIDANCE_COLS = [
    "gefsatmosmean_tmax_f",
    "rap_tmax_f",
    "hrrr_tmax_f",
    "nbm_tmax_f",
    "gfs_n_x_max",
    "nam_n_x_max",
]

SPREAD_COL = "gefsatmos_tmp_spread_f"

MOS_CODES_BASE = [
    "tmp",
    "dpt",
    "wsp",
    "wdr",
    "cig",
    "vis",
    "p06",
    "p12",
    "q06",
    "q12",
    "pos",
    "poz",
    "t06",
    "t06_1",
    "t06_2",
    "n_x",
]

DEFAULT_OUTPUT_ROOT = Path("artifacts") / "time_feature_sweep_v2"

BASELINE_PARAMS = {
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
}


@dataclass(frozen=True)
class SplitConfig:
    train_start: date = date(2021, 2, 23)
    train_end: date = date(2024, 6, 30)
    val_start: date = date(2024, 7, 1)
    val_end: date = date(2025, 1, 30)
    test_start: date = date(2025, 2, 1)
    test_end: date = date(2025, 12, 31)
    gap_dates: tuple[date, ...] = (date(2025, 1, 31),)


DEFAULT_SPLIT = SplitConfig()

TRUTH_LAG_DAYS = 2
ASOF_HOUR_UTC = 12
