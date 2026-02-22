"""Dataset loading and merging helpers for exp30 sweeps."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import DEFAULT_SPLIT

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetSplit:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    split_ref: dict


def load_gribstream_csv(path: Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Gribstream CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, dtype={"station_id": "string"})
    if "actual_tmax_f" not in df.columns and "target_tmax_f" in df.columns:
        df = df.rename(columns={"target_tmax_f": "actual_tmax_f"})
    df = _coerce_types(df)
    return df


def load_mos_csv(path: Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"MOS CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, dtype={"station_id": "string"})
    if "actual_tmax_f" not in df.columns and "target_tmax_f" in df.columns:
        df = df.rename(columns={"target_tmax_f": "actual_tmax_f"})
    df = _coerce_types(df)
    return df


def merge_grib_mos(grib_df: pd.DataFrame, mos_df: pd.DataFrame) -> pd.DataFrame:
    grib = grib_df.copy()
    mos = mos_df.copy()

    key_cols = ["station_id", "target_date_local", "asof_utc"]
    grib = _dedupe(grib, key_cols)
    mos = _dedupe(mos, key_cols)

    merged = grib.merge(mos, on=key_cols, how="left", suffixes=("", "_mos"))
    if "actual_tmax_f_mos" in merged.columns:
        merged["actual_tmax_f"] = merged["actual_tmax_f"].fillna(
            merged["actual_tmax_f_mos"]
        )
        merged = merged.drop(columns=["actual_tmax_f_mos"])
    return merged


def filter_station(df: pd.DataFrame, station_id: str | None) -> pd.DataFrame:
    if station_id is None:
        return df
    return df[df["station_id"].astype(str).str.upper() == station_id.upper()].copy()


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    missing = [col for col in columns if col not in df.columns]
    if missing:
        for col in missing:
            df[col] = pd.NA
        LOGGER.warning("Added missing columns with NA: %s", missing)
    return df


def split_dataset(df: pd.DataFrame, split_cfg=DEFAULT_SPLIT) -> DatasetSplit:
    df = df.copy()
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    gap_set = set(split_cfg.gap_dates)
    in_gap = df["target_date_local"].isin(gap_set)

    train_mask = (df["target_date_local"] >= split_cfg.train_start) & (
        df["target_date_local"] <= split_cfg.train_end
    )
    val_mask = (df["target_date_local"] >= split_cfg.val_start) & (
        df["target_date_local"] <= split_cfg.val_end
    )
    test_mask = (df["target_date_local"] >= split_cfg.test_start) & (
        df["target_date_local"] <= split_cfg.test_end
    )

    train_df = df[train_mask & ~val_mask & ~in_gap].copy()
    val_df = df[val_mask & ~in_gap].copy()
    test_df = df[test_mask & ~in_gap].copy()

    split_ref = {
        "train_start": str(split_cfg.train_start),
        "train_end": str(split_cfg.train_end),
        "val_start": str(split_cfg.val_start),
        "val_end": str(split_cfg.val_end),
        "test_start": str(split_cfg.test_start),
        "test_end": str(split_cfg.test_end),
        "gap_dates": [str(d) for d in split_cfg.gap_dates],
    }
    return DatasetSplit(train_df=train_df, val_df=val_df, test_df=test_df, split_ref=split_ref)


def dataset_summary(df: pd.DataFrame, split: DatasetSplit) -> dict:
    return {
        "row_count": int(len(df)),
        "stations": sorted(df["station_id"].dropna().unique().tolist()),
        "date_min": str(pd.to_datetime(df["target_date_local"]).min().date())
        if not df.empty
        else None,
        "date_max": str(pd.to_datetime(df["target_date_local"]).max().date())
        if not df.empty
        else None,
        "split_counts": {
            "train": int(len(split.train_df)),
            "validation": int(len(split.val_df)),
            "test": int(len(split.test_df)),
        },
    }


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "target_date_local" in df.columns:
        df["target_date_local"] = pd.to_datetime(
            df["target_date_local"], errors="coerce"
        ).dt.date
    if "asof_utc" in df.columns:
        df["asof_utc"] = pd.to_datetime(df["asof_utc"], errors="coerce", utc=True)
    for col in df.columns:
        if col in ("station_id", "target_date_local", "asof_utc"):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "station_id" in df.columns:
        df["station_id"] = df["station_id"].astype("string")
    return df


def _dedupe(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df
    before = len(df)
    df = df.sort_values(key_cols)
    df = df.drop_duplicates(subset=key_cols, keep="last")
    dropped = before - len(df)
    if dropped > 0:
        LOGGER.warning("Dropped %s duplicate rows on %s", dropped, key_cols)
    return df
