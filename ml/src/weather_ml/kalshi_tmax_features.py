"""Additional feature engineering blocks for KMIA Tmax Kalshi pipeline."""

from __future__ import annotations

from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd

from .mos_config import MosDatasetConfig


def add_kalshi_extra_features(df: pd.DataFrame, cfg: MosDatasetConfig) -> pd.DataFrame:
    df = df.copy()
    models = [m.lower() for m in (cfg.models or ["GFS", "NAM"])]
    eps = 1e-3

    _add_calendar_harmonics(df)
    _add_wet_season_flag(df)

    for model in models:
        _add_temp_dewpoint_features(df, model, eps)
        _add_wind_direction_features(df, model)
        _add_precip_features(df, model)
        _add_ceiling_vis_flags(df, model)

    return df


def _add_calendar_harmonics(df: pd.DataFrame) -> None:
    if "cal_d_doy" in df.columns:
        doy = pd.to_numeric(df["cal_d_doy"], errors="coerce")
    elif "target_date_local" in df.columns:
        doy = pd.to_datetime(df["target_date_local"]).dt.dayofyear
        df["cal_d_doy"] = doy
    else:
        return
    df["cal_d_doy_sin2"] = np.sin(4 * np.pi * doy / 365.25)
    df["cal_d_doy_cos2"] = np.cos(4 * np.pi * doy / 365.25)
    df["cal_d_doy_sin3"] = np.sin(6 * np.pi * doy / 365.25)
    df["cal_d_doy_cos3"] = np.cos(6 * np.pi * doy / 365.25)


def _add_wet_season_flag(df: pd.DataFrame) -> None:
    if "target_date_local" not in df.columns:
        return
    month = pd.to_datetime(df["target_date_local"]).dt.month
    df["cal_wet_season_flag"] = ((month >= 5) & (month <= 10)).astype(float)


def _add_temp_dewpoint_features(df: pd.DataFrame, model: str, eps: float) -> None:
    tmp_mean = _col(df, f"mos_{model}_tmp_mean")
    tmp_min = _col(df, f"mos_{model}_tmp_min")
    tmp_max = _col(df, f"mos_{model}_tmp_max")
    dpt_mean = _col(df, f"mos_{model}_dpt_mean")
    dpt_min = _col(df, f"mos_{model}_dpt_min")
    dpt_max = _col(df, f"mos_{model}_dpt_max")

    if tmp_mean is not None and dpt_mean is not None:
        df[f"mos_phys_{model}_dd_mean"] = tmp_mean - dpt_mean
        df[f"mos_phys_{model}_rh_mean"] = _relative_humidity(tmp_mean, dpt_mean)
        df[f"mos_phys_{model}_heat_index_mean"] = _heat_index_f(tmp_mean, df[f"mos_phys_{model}_rh_mean"])

    if tmp_max is not None and dpt_min is not None:
        df[f"mos_phys_{model}_dd_max"] = tmp_max - dpt_min
    if tmp_min is not None and dpt_max is not None:
        df[f"mos_phys_{model}_dd_min"] = tmp_min - dpt_max

    if tmp_max is not None and tmp_min is not None:
        diurnal_amp = tmp_max - tmp_min
        df[f"mos_phys_{model}_diurnal_amp"] = diurnal_amp
        if tmp_mean is not None:
            df[f"mos_phys_{model}_temp_curve_flatness"] = (
                (tmp_mean - tmp_min) / (diurnal_amp.abs() + eps)
            )


def _add_wind_direction_features(df: pd.DataFrame, model: str) -> None:
    wdr_mean = _col(df, f"mos_{model}_wdr_mean")
    wsp_mean = _col(df, f"mos_{model}_wsp_mean")
    if wdr_mean is None:
        return
    radians = np.deg2rad(wdr_mean)
    df[f"mos_phys_{model}_wdr_mean_sin"] = np.sin(radians)
    df[f"mos_phys_{model}_wdr_mean_cos"] = np.cos(radians)
    if wsp_mean is None:
        return
    # Onshore component for Miami (east wind ~ 90 degrees).
    df[f"mos_phys_{model}_onshore_component"] = wsp_mean * np.cos(radians - np.deg2rad(90.0))


def _add_precip_features(df: pd.DataFrame, model: str) -> None:
    p06_max = _col(df, f"mos_{model}_p06_max")
    p12_max = _col(df, f"mos_{model}_p12_max")
    if p06_max is not None or p12_max is not None:
        df[f"mos_phys_{model}_pop_max"] = _rowwise_max([p06_max, p12_max])
        df[f"mos_phys_{model}_pop_mean"] = _rowwise_mean([p06_max, p12_max])

    q06_mean = _col(df, f"mos_{model}_q06_mean")
    q12_mean = _col(df, f"mos_{model}_q12_mean")
    if q06_mean is not None:
        df[f"mos_phys_{model}_qpf06_ev"] = _map_qpf_categories(q06_mean, hours=6)
    if q12_mean is not None:
        df[f"mos_phys_{model}_qpf12_ev"] = _map_qpf_categories(q12_mean, hours=12)
    if q06_mean is not None or q12_mean is not None:
        df[f"mos_phys_{model}_qpf_total_ev"] = _rowwise_sum(
            [df.get(f"mos_phys_{model}_qpf06_ev"), df.get(f"mos_phys_{model}_qpf12_ev")]
        )

    t06_mean = _col(df, f"mos_{model}_t06_mean")
    t06_1_mean = _col(df, f"mos_{model}_t06_1_mean")
    t06_2_mean = _col(df, f"mos_{model}_t06_2_mean")
    if t06_mean is not None:
        df[f"mos_phys_{model}_thunder_mean"] = t06_mean
    if t06_1_mean is not None:
        df[f"mos_phys_{model}_thunder_prob"] = t06_1_mean
    if t06_2_mean is not None:
        df[f"mos_phys_{model}_thunder_severe_prob"] = t06_2_mean


def _add_ceiling_vis_flags(df: pd.DataFrame, model: str) -> None:
    cig_min = _col(df, f"mos_{model}_cig_min")
    vis_min = _col(df, f"mos_{model}_vis_min")
    if cig_min is not None:
        df[f"mos_phys_{model}_cig_low_flag"] = _flag_threshold(cig_min, 3.0)
    if vis_min is not None:
        df[f"mos_phys_{model}_vis_low_flag"] = _flag_threshold(vis_min, 3.0)


def _flag_threshold(series: pd.Series, threshold: float) -> pd.Series:
    return np.where(series.isna(), np.nan, (series <= threshold).astype(float))


def _rowwise_max(series_list: Iterable[pd.Series | None]) -> pd.Series:
    series_list = [s for s in series_list if s is not None]
    if not series_list:
        return pd.Series(np.nan, index=[])
    stacked = np.vstack([s.to_numpy(dtype=float) for s in series_list])
    return pd.Series(np.nanmax(stacked, axis=0), index=series_list[0].index)


def _rowwise_mean(series_list: Iterable[pd.Series | None]) -> pd.Series:
    series_list = [s for s in series_list if s is not None]
    if not series_list:
        return pd.Series(np.nan, index=[])
    stacked = np.vstack([s.to_numpy(dtype=float) for s in series_list])
    return pd.Series(np.nanmean(stacked, axis=0), index=series_list[0].index)


def _rowwise_sum(series_list: Iterable[pd.Series | None]) -> pd.Series:
    series_list = [s for s in series_list if s is not None]
    if not series_list:
        return pd.Series(np.nan, index=[])
    stacked = np.vstack([s.to_numpy(dtype=float) for s in series_list])
    return pd.Series(np.nansum(stacked, axis=0), index=series_list[0].index)


def _map_qpf_categories(series: pd.Series, *, hours: int) -> pd.Series:
    if hours == 6:
        mapping = {0: 0.0, 1: 0.05, 2: 0.17, 3: 0.37, 4: 0.75, 5: 1.25}
    else:
        mapping = {0: 0.0, 1: 0.05, 2: 0.17, 3: 0.37, 4: 0.75, 5: 1.25, 6: 2.5}
    rounded = pd.to_numeric(series, errors="coerce").round().astype("Int64")
    mapped = rounded.map(mapping)
    return mapped.where(mapped.notna(), pd.to_numeric(series, errors="coerce"))


def _relative_humidity(temp_f: pd.Series, dewpoint_f: pd.Series) -> pd.Series:
    temp_c = (temp_f - 32.0) * 5.0 / 9.0
    dew_c = (dewpoint_f - 32.0) * 5.0 / 9.0
    es = np.exp((17.625 * temp_c) / (243.04 + temp_c))
    e = np.exp((17.625 * dew_c) / (243.04 + dew_c))
    rh = 100.0 * (e / es)
    return rh.clip(lower=0.0, upper=100.0)


def _heat_index_f(temp_f: pd.Series, rh: pd.Series) -> pd.Series:
    t = temp_f.astype(float)
    r = rh.astype(float)
    hi = (
        -42.379
        + 2.04901523 * t
        + 10.14333127 * r
        - 0.22475541 * t * r
        - 0.00683783 * t * t
        - 0.05481717 * r * r
        + 0.00122874 * t * t * r
        + 0.00085282 * t * r * r
        - 0.00000199 * t * t * r * r
    )
    # Use temperature itself when outside typical heat-index regime.
    return np.where((t < 80.0) | (r < 40.0), t, hi)


def _col(df: pd.DataFrame, name: str) -> pd.Series | None:
    if name not in df.columns:
        return None
    return pd.to_numeric(df[name], errors="coerce")
