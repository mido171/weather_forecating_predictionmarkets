"""Configuration constants for the 30-experiment sweep."""

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

ENSEMBLE_REFERENCE = "ens_mean_guidance"
SPREAD_COL = "gefsatmos_tmp_spread_f"

MOS_VALUE_COLUMNS = {
    "tmp": ("gfs_tmp_mean", "nam_tmp_mean"),
    "dpt": ("gfs_dpt_mean", "nam_dpt_mean"),
    "wsp": ("gfs_wsp_mean", "nam_wsp_mean"),
    "wdr": ("gfs_wdr_mean", "nam_wdr_mean"),
    "cig": ("gfs_cig_mean", "nam_cig_mean"),
    "vis": ("gfs_vis_mean", "nam_vis_mean"),
    "p06": ("gfs_p06_mean", "nam_p06_mean"),
    "p12": ("gfs_p12_mean", "nam_p12_mean"),
    "q06": ("gfs_q06_mean", "nam_q06_mean"),
    "q12": ("gfs_q12_mean", "nam_q12_mean"),
    "pos": ("gfs_pos_mean", "nam_pos_mean"),
    "poz": ("gfs_poz_mean", "nam_poz_mean"),
    "n_x": ("gfs_n_x_max", "nam_n_x_max"),
    "t06": ("gfs_t06_mean", "nam_t06_mean"),
    "t06_1": ("gfs_t06_1_mean", "nam_t06_1_mean"),
    "t06_2": ("gfs_t06_2_mean", "nam_t06_2_mean"),
    "snw": ("gfs_snw_mean", "nam_snw_mean"),
}

MOS_ALL_CODES = [
    "cig",
    "dpt",
    "n_x",
    "p06",
    "p12",
    "pos",
    "poz",
    "q06",
    "q12",
    "snw",
    "t06",
    "t06_1",
    "t06_2",
    "tmp",
    "vis",
    "wdr",
    "wsp",
]

MOS_SURFACE_CODES = ["tmp", "dpt", "wsp", "wdr", "cig", "vis"]
MOS_THERMO_CODES = ["tmp", "dpt", "t06", "t06_1", "t06_2", "n_x"]
MOS_WIND_CODES = ["wsp", "wdr"]
MOS_CLOUD_CODES = ["cig", "vis"]
MOS_PRECIP_CODES = ["p06", "p12", "q06", "q12", "pos", "poz"]

DEFAULT_GRIB_CSV = Path(
    "ingestion-service/src/main/resources/trainingdata_output/gribstream_training_data.csv"
)
DEFAULT_MOS_CSV = Path(
    "ingestion-service/src/main/resources/trainingdata_output/mos_training_data.csv"
)

DEFAULT_OUTPUT_ROOT = Path("artifacts") / "exp30_sweeps"

DEFAULT_SEEDS = [0, 1, 2]


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
