"""CSV dataset loader for the training pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "station_id",
    "target_date_local",
    "asof_utc",
    "gfs_tmax_f",
    "nam_tmax_f",
    "gefsatmosmean_tmax_f",
    "rap_tmax_f",
    "hrrr_tmax_f",
    "nbm_tmax_f",
    "gefsatmos_tmp_spread_f",
    "actual_tmax_f",
]


def load_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path, dtype={"station_id": "string"})
    if "actual_tmax_f" not in df.columns and "target_tmax_f" in df.columns:
        df = df.rename(columns={"target_tmax_f": "actual_tmax_f"})
    elif "actual_tmax_f" in df.columns and "target_tmax_f" in df.columns:
        df = df.drop(columns=["target_tmax_f"])
    df = _coerce_types(df)
    _log_dataset_stats(df)
    return df


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target_date_local"] = pd.to_datetime(
        df["target_date_local"], errors="coerce"
    ).dt.normalize()
    df["asof_utc"] = pd.to_datetime(df["asof_utc"], errors="coerce", utc=True)
    numeric_cols = [
        "gfs_tmax_f",
        "nam_tmax_f",
        "gefsatmosmean_tmax_f",
        "rap_tmax_f",
        "hrrr_tmax_f",
        "nbm_tmax_f",
        "gefsatmos_tmp_spread_f",
        "actual_tmax_f",
    ]
    for column in numeric_cols:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "station_id" in df.columns:
        df["station_id"] = df["station_id"].astype("string")
    return df


def _log_dataset_stats(df: pd.DataFrame) -> None:
    LOGGER.info("CSV row count: %s", len(df))
    if not df.empty:
        LOGGER.info("Stations: %s", sorted(df["station_id"].dropna().unique()))
        missing = df.isna().sum().to_dict()
        LOGGER.info("Missing counts: %s", missing)
