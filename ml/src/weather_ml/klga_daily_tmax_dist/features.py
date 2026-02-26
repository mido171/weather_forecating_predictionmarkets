from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any
import logging

import numpy as np
import pandas as pd

from .config import (
    COASTAL_STATIONS,
    INLAND_STATIONS,
    NORTH_INTERIOR_STATIONS,
    PipelineConfig,
    TARGET_STATION_ID,
    URBAN_FRINGE_STATIONS,
)
from .logging_utils import ProgressTracker


@dataclass(frozen=True)
class StationSeries:
    station_id: str
    times_ns: np.ndarray
    values: dict[str, np.ndarray]
    last_valid_idx: dict[str, np.ndarray]


def _station_short(station_id: str) -> str:
    return station_id.split(":", 1)[0]


def _build_last_valid_index(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape[0], -1, dtype=np.int32)
    last = -1
    for i in range(values.shape[0]):
        if np.isfinite(values[i]):
            last = i
        out[i] = last
    return out


def prepare_station_series(
    obs_df: pd.DataFrame,
    *,
    station_ids: tuple[str, ...],
) -> dict[str, StationSeries]:
    required_cols = {
        "request_location_id",
        "valid_time_utc",
        "temp",
        "dew_pt",
        "rh",
        "pressure",
        "vis",
        "wspd",
        "wdir",
        "gust",
        "precip_hrly",
    }
    missing = required_cols.difference(obs_df.columns)
    if missing:
        raise ValueError(f"Observation DataFrame missing columns: {sorted(missing)}")

    out: dict[str, StationSeries] = {}
    for station_id in station_ids:
        sdf = obs_df[obs_df["request_location_id"] == station_id].copy()
        sdf = sdf.sort_values("valid_time_utc").reset_index(drop=True)
        times_ns = sdf["valid_time_utc"].astype("int64").to_numpy(dtype=np.int64)

        values: dict[str, np.ndarray] = {}
        last_valid_idx: dict[str, np.ndarray] = {}
        for col in ["temp", "dew_pt", "rh", "pressure", "vis", "wspd", "wdir", "gust", "precip_hrly"]:
            arr = pd.to_numeric(sdf[col], errors="coerce").to_numpy(dtype=float)
            values[col] = arr
            last_valid_idx[col] = _build_last_valid_index(arr)

        out[station_id] = StationSeries(
            station_id=station_id,
            times_ns=times_ns,
            values=values,
            last_valid_idx=last_valid_idx,
        )
    return out


def build_daily_prior_frame(daily_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame(
            columns=[
                "target_date_local",
                "tmax_yday",
                "tmax_2day",
                "tmax_mean_7d",
                "tmax_mean_30d",
                "tmax_std_30d",
            ]
        )
    df = daily_df[["target_date_local", "max_temp_f"]].copy()
    df = df.sort_values("target_date_local").reset_index(drop=True)
    vals = pd.to_numeric(df["max_temp_f"], errors="coerce").astype(float)
    lag1 = vals.shift(1)
    out = pd.DataFrame({"target_date_local": df["target_date_local"]})
    out["tmax_yday"] = lag1
    out["tmax_2day"] = vals.shift(2)
    out["tmax_mean_7d"] = lag1.rolling(7, min_periods=1).mean()
    out["tmax_mean_30d"] = lag1.rolling(30, min_periods=1).mean()
    out["tmax_std_30d"] = lag1.rolling(30, min_periods=2).std(ddof=0)
    return out


def _update_max_used(max_used_ns: int | None, ts_ns: int | None) -> int | None:
    if ts_ns is None:
        return max_used_ns
    if max_used_ns is None:
        return ts_ns
    return max(max_used_ns, ts_ns)


def _value_at_or_before(
    station: StationSeries,
    *,
    column: str,
    target_ns: int,
) -> tuple[float, int | None, int]:
    idx = int(np.searchsorted(station.times_ns, target_ns, side="right") - 1)
    if idx < 0:
        return np.nan, None, -1
    lv = int(station.last_valid_idx[column][idx])
    if lv < 0:
        return np.nan, None, -1
    return float(station.values[column][lv]), int(station.times_ns[lv]), lv


def _window_slice_indices(
    station: StationSeries,
    *,
    start_ns: int,
    end_ns: int,
) -> tuple[int, int]:
    i0 = int(np.searchsorted(station.times_ns, start_ns, side="left"))
    i1 = int(np.searchsorted(station.times_ns, end_ns, side="right"))
    return i0, i1


def _window_stats(
    station: StationSeries,
    *,
    column: str,
    start_ns: int,
    end_ns: int,
) -> tuple[float, float, float, float, float, int | None]:
    i0, i1 = _window_slice_indices(station, start_ns=start_ns, end_ns=end_ns)
    if i1 <= i0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, None
    vals = station.values[column][i0:i1]
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, int(station.times_ns[i1 - 1])
    std = float(np.std(finite, ddof=0))
    min_v = float(np.min(finite))
    max_v = float(np.max(finite))
    rng = float(max_v - min_v)
    if finite.size >= 2:
        std_diff = float(np.std(np.diff(finite), ddof=0))
    else:
        std_diff = np.nan
    return std, min_v, max_v, rng, std_diff, int(station.times_ns[i1 - 1])


def _local_minute_of_day(ts_ns: int, local_zone) -> int:
    ts = pd.Timestamp(ts_ns, tz="UTC").tz_convert(local_zone)
    return int(ts.hour * 60 + ts.minute)


def _compute_station_snapshot(
    station: StationSeries,
    *,
    cutoff_ns: int,
    cutoff_minutes: int,
) -> tuple[dict[str, float], int | None, int]:
    features: dict[str, float] = {}
    max_used_ns: int | None = None
    cutoff_idx = int(np.searchsorted(station.times_ns, cutoff_ns, side="right") - 1)
    if cutoff_idx >= 0:
        max_used_ns = int(station.times_ns[cutoff_idx])

    temp_now, temp_ts_ns, row_idx = _value_at_or_before(
        station, column="temp", target_ns=cutoff_ns
    )
    max_used_ns = _update_max_used(max_used_ns, temp_ts_ns)
    features["temp_now"] = temp_now
    features["is_temp_missing_now"] = 0.0 if np.isfinite(temp_now) else 1.0
    features["age_min_temp"] = (
        float((cutoff_ns - temp_ts_ns) / 60_000_000_000.0) if temp_ts_ns is not None else np.nan
    )

    row_cols = ["dew_pt", "rh", "pressure", "vis", "wspd", "wdir", "gust", "precip_hrly"]
    for col in row_cols:
        key = "dewpt_now" if col == "dew_pt" else f"{col}_now"
        miss_key = f"is_{col}_missing_now"
        if col == "dew_pt":
            miss_key = "is_dew_pt_missing_now"
        if row_idx >= 0:
            raw = float(station.values[col][row_idx])
        else:
            raw = np.nan

        if col in {"wdir", "gust", "precip_hrly"} and not np.isfinite(raw):
            features[key] = 0.0
            features[miss_key] = 1.0
        else:
            features[key] = raw
            features[miss_key] = 0.0 if np.isfinite(raw) else 1.0

    if np.isfinite(features["temp_now"]) and np.isfinite(features.get("dewpt_now", np.nan)):
        features["dewpoint_depression_now"] = float(features["temp_now"] - features["dewpt_now"])
    else:
        features["dewpoint_depression_now"] = np.nan

    wdir_now = features.get("wdir_now", np.nan)
    if np.isfinite(wdir_now):
        wdir_rad = float(np.deg2rad(wdir_now))
        features["wdir_sin"] = float(np.sin(wdir_rad))
        features["wdir_cos"] = float(np.cos(wdir_rad))
    else:
        features["wdir_sin"] = 0.0
        features["wdir_cos"] = 0.0

    gust_now = features.get("gust_now", np.nan)
    wspd_now = features.get("wspd_now", np.nan)
    if np.isfinite(gust_now) and np.isfinite(wspd_now):
        features["gust_factor"] = float(gust_now - wspd_now)
    else:
        features["gust_factor"] = np.nan

    features["cutoff_minutes"] = float(cutoff_minutes)
    return features, max_used_ns, row_idx


def _compute_station_full_features(
    station: StationSeries,
    *,
    cutoff_ns: int,
    midnight_ns: int,
    cutoff_minutes: int,
    n_expected_bins: int,
    windows_minutes: tuple[int, ...],
    local_zone,
) -> tuple[dict[str, float], int | None]:
    snapshot, max_used_ns, row_idx = _compute_station_snapshot(
        station,
        cutoff_ns=cutoff_ns,
        cutoff_minutes=cutoff_minutes,
    )
    features = dict(snapshot)

    mid_i0 = int(np.searchsorted(station.times_ns, midnight_ns, side="left"))
    cut_i1 = int(np.searchsorted(station.times_ns, cutoff_ns, side="right"))
    features["n_obs_sofar"] = float(max(cut_i1 - mid_i0, 0))

    temp_seg = station.values["temp"][mid_i0:cut_i1]
    valid_temp_mask = np.isfinite(temp_seg)
    n_obs_temp = int(np.sum(valid_temp_mask))
    features["n_obs_temp"] = float(n_obs_temp)
    features["coverage_frac_temp"] = (
        float(n_obs_temp / n_expected_bins) if n_expected_bins > 0 else np.nan
    )
    features["n_expected_bins"] = float(n_expected_bins)

    if valid_temp_mask.any():
        tmax = float(np.nanmax(temp_seg))
        tmin = float(np.nanmin(temp_seg))
        features["tmax_sofar"] = tmax
        features["tmin_sofar"] = tmin
        features["temp_range_sofar"] = float(tmax - tmin)
        features["temp_now_minus_tmax"] = (
            float(features["temp_now"] - tmax) if np.isfinite(features["temp_now"]) else np.nan
        )
        rel_idxs = np.where(temp_seg == tmax)[0]
        if rel_idxs.size:
            tmax_idx = int(mid_i0 + rel_idxs[-1])
            tmax_ts = int(station.times_ns[tmax_idx])
            local_minute = _local_minute_of_day(tmax_ts, local_zone)
            features["time_of_tmax_sofar_min"] = float(local_minute)
            features["mins_since_tmax"] = float(max(cutoff_minutes - local_minute, 0))
            max_used_ns = _update_max_used(max_used_ns, tmax_ts)
        else:
            features["time_of_tmax_sofar_min"] = np.nan
            features["mins_since_tmax"] = np.nan
    else:
        features["tmax_sofar"] = np.nan
        features["tmin_sofar"] = np.nan
        features["temp_range_sofar"] = np.nan
        features["temp_now_minus_tmax"] = np.nan
        features["time_of_tmax_sofar_min"] = np.nan
        features["mins_since_tmax"] = np.nan

    for var in ["dew_pt", "pressure", "wspd", "gust"]:
        seg = station.values[var][mid_i0:cut_i1]
        finite = seg[np.isfinite(seg)]
        if finite.size == 0:
            features[f"{var}_max_sofar"] = np.nan
            features[f"{var}_min_sofar"] = np.nan
            features[f"{var}_range_sofar"] = np.nan
        else:
            max_v = float(np.max(finite))
            min_v = float(np.min(finite))
            features[f"{var}_max_sofar"] = max_v
            features[f"{var}_min_sofar"] = min_v
            features[f"{var}_range_sofar"] = float(max_v - min_v)

    precip_seg = station.values["precip_hrly"][mid_i0:cut_i1]
    finite_precip = precip_seg[np.isfinite(precip_seg)]
    if finite_precip.size == 0:
        features["any_precip_sofar"] = 0.0
        features["precip_frac_sofar"] = 0.0
    else:
        precip_hits = (finite_precip > 0).astype(float)
        features["any_precip_sofar"] = float(np.max(precip_hits))
        features["precip_frac_sofar"] = float(np.mean(precip_hits))

    now_keys = {
        "temp": "temp_now",
        "dew_pt": "dewpt_now",
        "rh": "rh_now",
        "pressure": "pressure_now",
        "wspd": "wspd_now",
    }
    for w in windows_minutes:
        w_ns = int(w * 60 * 1_000_000_000)
        start_ns = cutoff_ns - w_ns
        for var in ["temp", "dew_pt", "rh", "pressure", "wspd"]:
            now_v = features.get(now_keys[var], np.nan)
            prev_v, prev_ts_ns, _ = _value_at_or_before(
                station,
                column=var,
                target_ns=start_ns,
            )
            max_used_ns = _update_max_used(max_used_ns, prev_ts_ns)
            if np.isfinite(now_v) and np.isfinite(prev_v):
                delta = float(now_v - prev_v)
                slope = float(delta / w)
            else:
                delta = np.nan
                slope = np.nan
            features[f"{var}_prev_{w}"] = prev_v
            features[f"{var}_delta_{w}"] = delta
            features[f"{var}_slope_{w}"] = slope

            std, min_v, max_v, rng, std_diff, win_used_ns = _window_stats(
                station,
                column=var,
                start_ns=start_ns,
                end_ns=cutoff_ns,
            )
            features[f"{var}_std_{w}"] = std
            features[f"{var}_min_{w}"] = min_v
            features[f"{var}_max_{w}"] = max_v
            features[f"{var}_range_{w}"] = rng
            features[f"{var}_std_diff_{w}"] = std_diff
            max_used_ns = _update_max_used(max_used_ns, win_used_ns)

    features["temp_accel_60_180"] = (
        features.get("temp_slope_60", np.nan) - features.get("temp_slope_180", np.nan)
    )
    features["temp_accel_30_120"] = (
        features.get("temp_slope_30", np.nan) - features.get("temp_slope_120", np.nan)
    )
    slope_60 = features.get("temp_slope_60", np.nan)
    features["temp_is_falling_60"] = float(1.0 if np.isfinite(slope_60) and slope_60 < 0 else 0.0)
    if np.isfinite(features.get("tmax_sofar", np.nan)) and np.isfinite(features.get("temp_now", np.nan)):
        features["temp_drop_from_peak"] = float(features["tmax_sofar"] - features["temp_now"])
    else:
        features["temp_drop_from_peak"] = np.nan

    if row_idx >= 0:
        max_used_ns = _update_max_used(max_used_ns, int(station.times_ns[row_idx]))
    return features, max_used_ns


def _doy_sin_cos(doy: int) -> tuple[float, float]:
    angle = 2.0 * np.pi * (doy / 366.0)
    return float(np.sin(angle)), float(np.cos(angle))


def _time_sin_cos(cutoff_minutes: int) -> tuple[float, float]:
    angle = 2.0 * np.pi * (cutoff_minutes / 1440.0)
    return float(np.sin(angle)), float(np.cos(angle))


def _safe_mean(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return np.nan
    return float(np.mean(finite))


def _safe_min(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return np.nan
    return float(np.min(finite))


def _safe_max(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return np.nan
    return float(np.max(finite))


def _safe_range(values: list[float]) -> float:
    mn = _safe_min(values)
    mx = _safe_max(values)
    if np.isfinite(mn) and np.isfinite(mx):
        return float(mx - mn)
    return np.nan


def _get_station_value(row: dict[str, Any], station_id: str, key: str) -> float:
    short = _station_short(station_id)
    return float(row.get(f"{short}_{key}", np.nan))


def _add_neighbor_composites(row: dict[str, Any], cfg: PipelineConfig) -> None:
    temp_vals = [_get_station_value(row, sid, "temp_now") for sid in cfg.neighbor_station_ids]
    dew_vals = [_get_station_value(row, sid, "dewpt_now") for sid in cfg.neighbor_station_ids]
    pres_vals = [_get_station_value(row, sid, "pressure_now") for sid in cfg.neighbor_station_ids]

    row["nbr_temp_mean"] = _safe_mean(temp_vals)
    row["nbr_temp_min"] = _safe_min(temp_vals)
    row["nbr_temp_max"] = _safe_max(temp_vals)
    row["nbr_temp_range"] = _safe_range(temp_vals)

    row["nbr_dewpt_mean"] = _safe_mean(dew_vals)
    row["nbr_dewpt_min"] = _safe_min(dew_vals)
    row["nbr_dewpt_max"] = _safe_max(dew_vals)
    row["nbr_dewpt_range"] = _safe_range(dew_vals)

    row["nbr_pressure_mean"] = _safe_mean(pres_vals)
    row["nbr_pressure_min"] = _safe_min(pres_vals)
    row["nbr_pressure_max"] = _safe_max(pres_vals)
    row["nbr_pressure_range"] = _safe_range(pres_vals)

    temp_coastal = _safe_mean([_get_station_value(row, sid, "temp_now") for sid in COASTAL_STATIONS])
    temp_inland = _safe_mean([_get_station_value(row, sid, "temp_now") for sid in INLAND_STATIONS])
    dew_coastal = _safe_mean([_get_station_value(row, sid, "dewpt_now") for sid in COASTAL_STATIONS])
    dew_inland = _safe_mean([_get_station_value(row, sid, "dewpt_now") for sid in INLAND_STATIONS])
    pres_coastal = _safe_mean([_get_station_value(row, sid, "pressure_now") for sid in COASTAL_STATIONS])
    pres_inland = _safe_mean([_get_station_value(row, sid, "pressure_now") for sid in INLAND_STATIONS])

    row["temp_coastal_mean"] = temp_coastal
    row["temp_inland_mean"] = temp_inland
    if np.isfinite(temp_coastal) and np.isfinite(temp_inland):
        row["coastal_minus_inland_temp"] = float(temp_coastal - temp_inland)
    else:
        row["coastal_minus_inland_temp"] = np.nan

    if np.isfinite(dew_coastal) and np.isfinite(dew_inland):
        row["dewpt_coastal_minus_inland"] = float(dew_coastal - dew_inland)
    else:
        row["dewpt_coastal_minus_inland"] = np.nan

    if np.isfinite(pres_coastal) and np.isfinite(pres_inland):
        row["pressure_coastal_minus_inland"] = float(pres_coastal - pres_inland)
    else:
        row["pressure_coastal_minus_inland"] = np.nan

    klga_temp = float(row.get("temp_now", np.nan))
    row["temp_inland_mean_minus_klga"] = (
        float(temp_inland - klga_temp) if np.isfinite(temp_inland) and np.isfinite(klga_temp) else np.nan
    )
    jfk_temp = _get_station_value(row, "KJFK:9:US", "temp_now")
    row["temp_jfk_minus_klga"] = (
        float(jfk_temp - klga_temp) if np.isfinite(jfk_temp) and np.isfinite(klga_temp) else np.nan
    )

    # Keep group means for downstream diagnostics.
    row["temp_north_mean"] = _safe_mean(
        [_get_station_value(row, sid, "temp_now") for sid in NORTH_INTERIOR_STATIONS]
    )
    row["temp_urban_fringe_mean"] = _safe_mean(
        [_get_station_value(row, sid, "temp_now") for sid in URBAN_FRINGE_STATIONS]
    )


def build_feature_rows(
    *,
    calendar_df: pd.DataFrame,
    station_series: dict[str, StationSeries],
    daily_truth_df: pd.DataFrame,
    daily_prior_df: pd.DataFrame,
    cfg: PipelineConfig,
    logger: logging.Logger | None = None,
    log_every_rows: int = 2000,
    log_every_seconds: float = 20.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if TARGET_STATION_ID != cfg.target_station_id:
        raise ValueError("Target station mismatch; KLGA must be canonical.")
    station_set = set(cfg.all_station_ids)
    if "KNYC:9:US" in station_set:
        raise ValueError("Duplicate-source guard failed: KNYC must be excluded.")

    missing_station_series = station_set.difference(station_series.keys())
    if missing_station_series:
        raise ValueError(f"Station series missing for {sorted(missing_station_series)}")

    truth_by_date = (
        daily_truth_df.set_index("target_date_local")["max_temp_f"].to_dict()
        if not daily_truth_df.empty
        else {}
    )
    prior_by_date = (
        daily_prior_df.set_index("target_date_local").to_dict("index")
        if not daily_prior_df.empty
        else {}
    )

    rows: list[dict[str, Any]] = []
    guard_failures = 0
    max_used_overall: pd.Timestamp | None = None
    active_logger = logger or logging.getLogger(__name__)
    tracker = ProgressTracker(
        logger=active_logger,
        name="FEATURE_ROWS",
        total=len(calendar_df),
        log_every_rows=log_every_rows,
        log_every_seconds=log_every_seconds,
    )

    for i, rec in enumerate(calendar_df.itertuples(index=False), start=1):
        target_date = rec.target_date_local
        cutoff_utc = pd.Timestamp(rec.cutoff_utc).tz_convert("UTC")
        midnight_utc = pd.Timestamp(rec.midnight_utc).tz_convert("UTC")
        cutoff_ns = int(cutoff_utc.value)
        midnight_ns = int(midnight_utc.value)
        cutoff_minutes = int(rec.cutoff_minutes)
        n_expected_bins = int(rec.n_expected_bins)
        cutoff_local = pd.Timestamp(rec.cutoff_local)

        doy = int(cutoff_local.dayofyear)
        cutoff_sin, cutoff_cos = _time_sin_cos(cutoff_minutes)
        doy_sin, doy_cos = _doy_sin_cos(doy)
        row: dict[str, Any] = {
            "target_date_local": target_date,
            "cutoff_local": cutoff_local.isoformat(),
            "cutoff_utc": cutoff_utc,
            "midnight_utc": midnight_utc,
            "cutoff_minutes": float(cutoff_minutes),
            "cutoff_hour": float(cutoff_local.hour),
            "cutoff_minute": float(cutoff_local.minute),
            "cutoff_sin": cutoff_sin,
            "cutoff_cos": cutoff_cos,
            "doy": float(doy),
            "doy_sin": doy_sin,
            "doy_cos": doy_cos,
            "year": float(cutoff_local.year),
            "year_norm": float((cutoff_local.year - 1992) / (2025 - 1992)),
            "is_weekend": float(1.0 if cutoff_local.weekday() >= 5 else 0.0),
            "is_dst": float(rec.is_dst),
        }

        klga_features, klga_max_used = _compute_station_full_features(
            station_series[cfg.target_station_id],
            cutoff_ns=cutoff_ns,
            midnight_ns=midnight_ns,
            cutoff_minutes=cutoff_minutes,
            n_expected_bins=n_expected_bins,
            windows_minutes=cfg.windows_minutes,
            local_zone=cfg.local_zone,
        )
        row.update(klga_features)

        station_max_used: list[int | None] = [klga_max_used]
        for sid in cfg.neighbor_station_ids:
            snap, max_used_ns, _ = _compute_station_snapshot(
                station_series[sid],
                cutoff_ns=cutoff_ns,
                cutoff_minutes=cutoff_minutes,
            )
            short = _station_short(sid)
            for k, v in snap.items():
                row[f"{short}_{k}"] = v
            station_max_used.append(max_used_ns)

        for sid in cfg.neighbor_station_ids:
            short = _station_short(sid)
            row[f"temp_diff_{short}"] = float(row.get(f"{short}_temp_now", np.nan) - row.get("temp_now", np.nan))
            row[f"dewpt_diff_{short}"] = float(
                row.get(f"{short}_dewpt_now", np.nan) - row.get("dewpt_now", np.nan)
            )
            row[f"pressure_diff_{short}"] = float(
                row.get(f"{short}_pressure_now", np.nan) - row.get("pressure_now", np.nan)
            )
            row[f"wspd_diff_{short}"] = float(
                row.get(f"{short}_wspd_now", np.nan) - row.get("wspd_now", np.nan)
            )
            row[f"dewpoint_depression_diff_{short}"] = float(
                row.get(f"{short}_dewpoint_depression_now", np.nan) - row.get("dewpoint_depression_now", np.nan)
            )

        _add_neighbor_composites(row, cfg)

        priors = prior_by_date.get(target_date, {})
        row["tmax_yday"] = float(priors.get("tmax_yday", np.nan))
        row["tmax_2day"] = float(priors.get("tmax_2day", np.nan))
        row["tmax_mean_7d"] = float(priors.get("tmax_mean_7d", np.nan))
        row["tmax_mean_30d"] = float(priors.get("tmax_mean_30d", np.nan))
        row["tmax_std_30d"] = float(priors.get("tmax_std_30d", np.nan))

        truth = truth_by_date.get(target_date)
        if truth is None or not np.isfinite(float(truth)):
            row["tmax_truth"] = np.nan
            row["delta"] = np.nan
            row["peak"] = np.nan
        else:
            tmax_truth = int(np.round(float(truth)))
            row["tmax_truth"] = float(tmax_truth)
            tmax_sofar = row.get("tmax_sofar", np.nan)
            if np.isfinite(tmax_sofar):
                delta = int(tmax_truth - int(np.round(float(tmax_sofar))))
                if delta < 0:
                    delta = 0
                row["delta"] = float(delta)
                row["peak"] = float(1.0 if delta <= 0 else 0.0)
            else:
                row["delta"] = np.nan
                row["peak"] = np.nan

        max_used_ns = None
        for ts_ns in station_max_used:
            max_used_ns = _update_max_used(max_used_ns, ts_ns)
        if max_used_ns is None:
            row["max_valid_time_used_utc"] = pd.NaT
        else:
            max_used_ts = pd.Timestamp(max_used_ns, tz="UTC")
            row["max_valid_time_used_utc"] = max_used_ts
            max_used_overall = max_used_ts if max_used_overall is None else max(max_used_overall, max_used_ts)
            if max_used_ts > cutoff_utc:
                guard_failures += 1

        rows.append(row)
        tracker.maybe_log(
            i,
            extra=(
                f"date={target_date} cutoff_min={cutoff_minutes} "
                f"asof_guard_failures={guard_failures}"
            ),
        )

    if guard_failures > 0:
        raise AssertionError(f"As-of guard failed for {guard_failures} rows.")

    feature_df = pd.DataFrame(rows)
    feature_df = feature_df.sort_values(["target_date_local", "cutoff_minutes"]).reset_index(drop=True)
    audit = {
        "rows": int(len(feature_df)),
        "dates": int(feature_df["target_date_local"].nunique() if not feature_df.empty else 0),
        "max_valid_time_used_utc": max_used_overall.isoformat() if max_used_overall is not None else None,
        "asof_guard_failures": int(guard_failures),
    }
    tracker.done(extra=f"dates={feature_df['target_date_local'].nunique()} asof_guard_failures={guard_failures}")
    return feature_df, audit


def model_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "target_date_local",
        "cutoff_local",
        "cutoff_utc",
        "midnight_utc",
        "max_valid_time_used_utc",
        "tmax_truth",
        "delta",
        "peak",
    }
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols
