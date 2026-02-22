"""Calendar/as-of helpers for MOS dataset."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from .mos_config import MosDatasetConfig


def expected_asof_utc(target_date_local: date, cfg: MosDatasetConfig) -> datetime:
    rule = (cfg.asof_rule or "target_minus_one_day_12z").lower()
    if rule.startswith("target_minus_one_day_local"):
        zone = ZoneInfo(cfg.station_zoneid)
        local_dt = datetime(
            target_date_local.year,
            target_date_local.month,
            target_date_local.day,
            cfg.asof_hour_utc,
            cfg.asof_minute_utc,
            tzinfo=zone,
        ) - timedelta(days=1)
        return local_dt.astimezone(timezone.utc)
    if rule.startswith("target_day_local"):
        zone = ZoneInfo(cfg.station_zoneid)
        local_dt = datetime(
            target_date_local.year,
            target_date_local.month,
            target_date_local.day,
            cfg.asof_hour_utc,
            cfg.asof_minute_utc,
            tzinfo=zone,
        )
        return local_dt.astimezone(timezone.utc)
    if rule.startswith("target_day_utc"):
        return datetime(
            target_date_local.year,
            target_date_local.month,
            target_date_local.day,
            cfg.asof_hour_utc,
            cfg.asof_minute_utc,
            tzinfo=timezone.utc,
        )
    base = datetime(
        target_date_local.year,
        target_date_local.month,
        target_date_local.day,
        cfg.asof_hour_utc,
        cfg.asof_minute_utc,
        tzinfo=timezone.utc,
    )
    return base - timedelta(days=1)


def build_calendar(cfg: MosDatasetConfig) -> pd.DataFrame:
    rule = (cfg.asof_rule or "target_minus_one_day_12z").lower()
    if rule.startswith("target_day"):
        start_target = cfg.build_start_asof
        end_target = cfg.end_asof
    else:
        start_target = cfg.build_start_asof + timedelta(days=1)
        end_target = cfg.end_asof + timedelta(days=1)
    dates = pd.date_range(start_target, end_target, freq="D")
    df = pd.DataFrame({"target_date_local": dates.date})
    df["asof_utc"] = df["target_date_local"].apply(lambda d: expected_asof_utc(d, cfg))
    zone = ZoneInfo(cfg.station_zoneid)
    df["asof_date_local"] = df["asof_utc"].apply(lambda dt: dt.astimezone(zone).date())
    df["station_id"] = cfg.station_id
    df["station_zoneid"] = cfg.station_zoneid
    return df


def add_calendar_features(df: pd.DataFrame, cfg: MosDatasetConfig) -> pd.DataFrame:
    df = df.copy()
    df["asof_date_local"] = pd.to_datetime(df["asof_date_local"])
    df["target_date_local"] = pd.to_datetime(df["target_date_local"])
    df["cal_t_year"] = df["asof_date_local"].dt.year
    df["cal_t_month"] = df["asof_date_local"].dt.month
    df["cal_t_day"] = df["asof_date_local"].dt.day
    df["cal_t_doy"] = df["asof_date_local"].dt.dayofyear
    df["cal_t_dow"] = df["asof_date_local"].dt.dayofweek
    df["cal_t_weekofyear"] = df["asof_date_local"].dt.isocalendar().week.astype(int)

    df["cal_d_year"] = df["target_date_local"].dt.year
    df["cal_d_month"] = df["target_date_local"].dt.month
    df["cal_d_day"] = df["target_date_local"].dt.day
    df["cal_d_doy"] = df["target_date_local"].dt.dayofyear
    df["cal_d_dow"] = df["target_date_local"].dt.dayofweek
    df["cal_d_weekofyear"] = df["target_date_local"].dt.isocalendar().week.astype(int)

    df["cal_t_doy_sin"] = np.sin(2 * np.pi * df["cal_t_doy"] / 365.25)
    df["cal_t_doy_cos"] = np.cos(2 * np.pi * df["cal_t_doy"] / 365.25)
    df["cal_d_doy_sin"] = np.sin(2 * np.pi * df["cal_d_doy"] / 365.25)
    df["cal_d_doy_cos"] = np.cos(2 * np.pi * df["cal_d_doy"] / 365.25)

    df["cal_t_dow_sin"] = np.sin(2 * np.pi * df["cal_t_dow"] / 7)
    df["cal_t_dow_cos"] = np.cos(2 * np.pi * df["cal_t_dow"] / 7)
    df["cal_d_dow_sin"] = np.sin(2 * np.pi * df["cal_d_dow"] / 7)
    df["cal_d_dow_cos"] = np.cos(2 * np.pi * df["cal_d_dow"] / 7)

    zone = ZoneInfo(cfg.station_zoneid)
    asof_local_hour = []
    asof_offset = []
    asof_is_dst = []
    for ts in pd.to_datetime(df["asof_utc"], utc=True):
        local = ts.tz_convert(zone)
        asof_local_hour.append(local.hour)
        offset = local.utcoffset().total_seconds() / 3600.0 if local.utcoffset() else 0.0
        asof_offset.append(offset)
        asof_is_dst.append(1.0 if local.dst() and local.dst().total_seconds() != 0 else 0.0)
    df["cal_asof_local_hour"] = asof_local_hour
    df["cal_asof_local_utc_offset_hours"] = asof_offset
    df["cal_asof_is_dst"] = asof_is_dst
    return df
