from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd


STATION_COORDS = {
    "KNYC": (40.7831, -73.9712),
    "KLGA": (40.7794, -73.8740),
    "KJFK": (40.6413, -73.7781),
    "KEWR": (40.6895, -74.1745),
    "KTEB": (40.8501, -74.0608),
    "KHPN": (41.0670, -73.7076),
    "KISP": (40.7952, -73.1002),
    "KBDR": (41.1635, -73.1262),
    "KMMU": (40.7992, -74.4149),
}

COASTAL = ["KJFK", "KISP", "KBDR"]
INLAND = ["KEWR", "KTEB", "KMMU", "KHPN"]

CLDS_TO_FRAC = {
    "CLR": 0.05,
    "SKC": 0.05,
    "FEW": 0.2,
    "SCT": 0.4,
    "BKN": 0.7,
    "OVC": 0.95,
    "VV": 0.95,
    "NSC": 0.05,
}

PHRASE_STATE_ORDER = [
    "CLEAR",
    "MOSTLY_CLEAR",
    "PARTLY_CLOUDY",
    "MOSTLY_CLOUDY",
    "OVERCAST_LOW",
    "OVERCAST_HIGH",
    "FOG_OR_LOW_VIS",
    "HAZE_SMOKE",
    "DRIZZLE_LIGHT_RAIN",
    "STEADY_RAIN",
    "SHOWERS_CONVECTIVE",
    "THUNDER",
    "SNOW_OR_ICE",
    "UNKNOWN_OR_MISSING",
]
PHRASE_STATE_TO_ID = {k: i for i, k in enumerate(PHRASE_STATE_ORDER)}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return r * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlon)
    brng = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    return brng


def _wdir_to_math_rad(wdir_deg: np.ndarray) -> np.ndarray:
    return np.deg2rad((270.0 - wdir_deg) % 360.0)


def _phrase_to_attrs(phrase: str | None) -> dict[str, float | int]:
    if phrase is None or (isinstance(phrase, float) and np.isnan(phrase)) or pd.isna(phrase):
        p = ""
    else:
        p = str(phrase).strip().lower()
    if not p:
        state = "UNKNOWN_OR_MISSING"
    elif re.search(r"thunder|t-storm|tstorm", p):
        state = "THUNDER"
    elif re.search(r"snow|sleet|ice|freezing", p):
        state = "SNOW_OR_ICE"
    elif re.search(r"showers", p):
        state = "SHOWERS_CONVECTIVE"
    elif re.search(r"drizzle|light rain", p):
        state = "DRIZZLE_LIGHT_RAIN"
    elif re.search(r"rain", p):
        state = "STEADY_RAIN"
    elif re.search(r"fog|mist", p):
        state = "FOG_OR_LOW_VIS"
    elif re.search(r"haze|smoke", p):
        state = "HAZE_SMOKE"
    elif re.search(r"overcast", p):
        state = "OVERCAST_HIGH"
    elif re.search(r"mostly cloudy|cloudy", p):
        state = "MOSTLY_CLOUDY"
    elif re.search(r"partly cloudy", p):
        state = "PARTLY_CLOUDY"
    elif re.search(r"mostly sunny|mostly clear", p):
        state = "MOSTLY_CLEAR"
    elif re.search(r"clear|fair|sunny", p):
        state = "CLEAR"
    else:
        state = "UNKNOWN_OR_MISSING"

    cloud_frac = {
        "CLEAR": 0.05,
        "MOSTLY_CLEAR": 0.2,
        "PARTLY_CLOUDY": 0.4,
        "MOSTLY_CLOUDY": 0.75,
        "OVERCAST_LOW": 0.9,
        "OVERCAST_HIGH": 0.95,
        "FOG_OR_LOW_VIS": 0.6,
        "HAZE_SMOKE": 0.25,
        "DRIZZLE_LIGHT_RAIN": 0.8,
        "STEADY_RAIN": 0.95,
        "SHOWERS_CONVECTIVE": 0.85,
        "THUNDER": 0.95,
        "SNOW_OR_ICE": 0.9,
        "UNKNOWN_OR_MISSING": 0.5,
    }[state]

    precip_flag = int(state in {"DRIZZLE_LIGHT_RAIN", "STEADY_RAIN", "SHOWERS_CONVECTIVE", "THUNDER", "SNOW_OR_ICE"})
    precip_rank = 0
    if state in {"DRIZZLE_LIGHT_RAIN"}:
        precip_rank = 1
    elif state in {"STEADY_RAIN", "SNOW_OR_ICE"}:
        precip_rank = 2
    elif state in {"SHOWERS_CONVECTIVE", "THUNDER"}:
        precip_rank = 3
    convective = int(state in {"SHOWERS_CONVECTIVE", "THUNDER"})
    fog_flag = int(state == "FOG_OR_LOW_VIS")
    haze_flag = int(state == "HAZE_SMOKE")
    windy = int(("windy" in p) or ("breezy" in p))
    radiation_killer = float(np.clip(cloud_frac + 0.5 * haze_flag + 0.5 * fog_flag + 0.25 * precip_flag, 0.0, 1.0))

    return {
        "phrase_state_id": PHRASE_STATE_TO_ID[state],
        "cloud_frac_est": cloud_frac,
        "precip_flag": precip_flag,
        "precip_intensity_rank": precip_rank,
        "convective_flag": convective,
        "fog_flag": fog_flag,
        "haze_smoke_flag": haze_flag,
        "wind_modifier_flag": windy,
        "radiation_killer_score": radiation_killer,
    }


def _compute_intraday_state(df: pd.DataFrame, median_step_minutes: int) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("target_date_local", sort=False)
    out["tmax_sofar"] = g["temp"].cummax()
    out["tmin_sofar"] = g["temp"].cummin()

    max_marker = out["valid_time_utc"].where(out["temp"].eq(out["tmax_sofar"]))
    out["last_tmax_time"] = max_marker.groupby(out["target_date_local"]).ffill()
    out["mins_since_tmax"] = (out["valid_time_utc"] - out["last_tmax_time"]).dt.total_seconds().div(60.0)
    out["mins_since_tmax"] = out["mins_since_tmax"].fillna(0.0)

    def _delta(col: str, mins: int, name: str) -> None:
        steps = max(1, int(round(mins / max(median_step_minutes, 1))))
        sh = g[col].shift(steps)
        out[name] = out[col] - sh

    _delta("temp", 60, "temp_delta_60")
    _delta("temp", 180, "temp_delta_180")
    _delta("dew_pt", 180, "dew_pt_delta_180")
    _delta("pressure", 360, "pressure_delta_360")

    out["temp_slope_60"] = out["temp_delta_60"] / 1.0
    out["temp_slope_180"] = out["temp_delta_180"] / 3.0
    out["temp_now_minus_tmax"] = out["temp"] - out["tmax_sofar"]
    out["dewpoint_depression_now"] = out["temp"] - out["dew_pt"]
    return out


def _rolling_transition_features(df: pd.DataFrame, median_step_minutes: int) -> pd.DataFrame:
    out = df.copy()
    steps_180 = max(1, int(round(180 / max(median_step_minutes, 1))))

    g = out.groupby("target_date_local", sort=False)
    changed = out["phrase_state_id"].ne(g["phrase_state_id"].shift(1)).astype(float)
    out["state_change_count_180"] = changed.groupby(out["target_date_local"]).rolling(steps_180, min_periods=1).sum().reset_index(level=0, drop=True)

    last_change_ts = out["valid_time_utc"].where(changed > 0)
    out["last_state_change_time"] = last_change_ts.groupby(out["target_date_local"]).ffill()
    out["mins_since_last_change"] = (out["valid_time_utc"] - out["last_state_change_time"]).dt.total_seconds().div(60.0).fillna(0.0)

    rk_shift = g["radiation_killer_score"].shift(steps_180)
    out["radiation_killer_slope_180"] = (out["radiation_killer_score"] - rk_shift) / 3.0

    vis_shift = g["vis"].shift(steps_180)
    out["vis_slope_180"] = (out["vis"] - vis_shift) / 3.0
    out["vis_shock_180"] = g["vis"].transform(lambda s: s.diff().abs().rolling(steps_180, min_periods=1).max())
    return out


def _add_climo_features(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    out = df.copy()
    train = out.loc[train_mask].copy()
    train["remaining_delta"] = train["y_tmax"] - train["temp"]
    grp = train.groupby(["doy", "cutoff_minutes"]) ["remaining_delta"].agg(["mean", "std"]).reset_index()
    grp = grp.rename(columns={"mean": "climo_rem_delta_mean", "std": "climo_rem_delta_std"})

    out = out.merge(grp, on=["doy", "cutoff_minutes"], how="left")

    doy_grp = train.groupby("doy")["remaining_delta"].agg(["mean", "std"]).reset_index().rename(columns={"mean": "doy_mean", "std": "doy_std"})
    out = out.merge(doy_grp, on="doy", how="left")
    out["climo_rem_delta_mean"] = out["climo_rem_delta_mean"].fillna(out["doy_mean"]).fillna(train["remaining_delta"].mean())
    out["climo_rem_delta_std"] = out["climo_rem_delta_std"].fillna(out["doy_std"]).fillna(train["remaining_delta"].std())
    out = out.drop(columns=["doy_mean", "doy_std"])
    return out


def _add_spatial_features(df: pd.DataFrame, target_station: str) -> pd.DataFrame:
    out = df.copy()

    for sid in set(COASTAL + INLAND):
        for c in ["temp", "dew_pt", "pressure", "vis"]:
            col = f"{sid}_{c}"
            if col not in out.columns:
                out[col] = np.nan

    coastal_temp = out[[f"{s}_temp" for s in COASTAL]].mean(axis=1)
    inland_temp = out[[f"{s}_temp" for s in INLAND]].mean(axis=1)
    coastal_dew = out[[f"{s}_dew_pt" for s in COASTAL]].mean(axis=1)
    inland_dew = out[[f"{s}_dew_pt" for s in INLAND]].mean(axis=1)

    out["coastal_minus_inland_temp"] = coastal_temp - inland_temp
    out["coastal_minus_inland_dewpt"] = coastal_dew - inland_dew
    out["temp_diff_KJFK"] = out.get("KJFK_temp", np.nan) - out["temp"]
    out["temp_diff_KEWR"] = out.get("KEWR_temp", np.nan) - out["temp"]

    target_lat, target_lon = STATION_COORDS.get(target_station, STATION_COORDS["KNYC"])
    wdir = pd.to_numeric(out["wdir"], errors="coerce").to_numpy(dtype=float)
    wdir_math = _wdir_to_math_rad(np.where(np.isfinite(wdir), wdir, 0.0))

    upwind_temp_num = np.zeros(len(out), dtype=float)
    upwind_dew_num = np.zeros(len(out), dtype=float)
    upwind_w = np.zeros(len(out), dtype=float)

    for sid in ["KLGA", "KJFK", "KEWR", "KTEB", "KHPN", "KISP", "KBDR", "KMMU"]:
        if sid == target_station:
            continue
        if sid not in STATION_COORDS:
            continue
        lat, lon = STATION_COORDS[sid]
        bearing = _bearing_deg(target_lat, target_lon, lat, lon)
        dist = _haversine_km(target_lat, target_lon, lat, lon)
        theta = np.deg2rad((90.0 - bearing) % 360.0)
        align = np.maximum(0.0, np.cos(theta - wdir_math))
        dist_w = np.exp(-dist / 30.0)
        weight = align * dist_w

        tcol = f"{sid}_temp"
        dcol = f"{sid}_dew_pt"
        tval = pd.to_numeric(out[tcol], errors="coerce").to_numpy(dtype=float) if tcol in out.columns else np.full(len(out), np.nan)
        dval = pd.to_numeric(out[dcol], errors="coerce").to_numpy(dtype=float) if dcol in out.columns else np.full(len(out), np.nan)

        good_t = np.isfinite(tval)
        good_d = np.isfinite(dval)

        upwind_temp_num[good_t] += weight[good_t] * tval[good_t]
        upwind_dew_num[good_d] += weight[good_d] * dval[good_d]
        upwind_w[good_t | good_d] += weight[good_t | good_d]

    out["upwind_temp"] = np.where(upwind_w > 0, upwind_temp_num / upwind_w, np.nan)
    out["upwind_dew_pt"] = np.where(upwind_w > 0, upwind_dew_num / upwind_w, np.nan)
    return out


def build_features(df: pd.DataFrame, train_end_date: str, target_station: str) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    out = df.copy()

    # Temporal geometry
    out["doy_sin"] = np.sin(2.0 * np.pi * out["doy"] / 366.0)
    out["doy_cos"] = np.cos(2.0 * np.pi * out["doy"] / 366.0)
    out["cutoff_minutes"] = pd.to_numeric(out["cutoff_minutes"], errors="coerce")

    # Cloud proxy
    out["clds_norm"] = out["clds"].astype("string").str.upper().map(CLDS_TO_FRAC).astype(float)

    # Phrase-to-physics
    attrs = out["wx_phrase"].apply(_phrase_to_attrs).apply(pd.Series)
    out = pd.concat([out, attrs], axis=1)

    # Wind geometry
    wdir = pd.to_numeric(out["wdir"], errors="coerce")
    out["wdir_sin"] = np.sin(np.deg2rad(wdir))
    out["wdir_cos"] = np.cos(np.deg2rad(wdir))

    dt_minutes = out.sort_values("valid_time_utc")["valid_time_utc"].diff().dt.total_seconds().div(60.0)
    median_step = int(np.nanmedian(dt_minutes[(dt_minutes > 0) & (dt_minutes < 181)].to_numpy())) if dt_minutes.notna().any() else 60
    median_step = max(30, min(120, median_step if median_step > 0 else 60))

    out = _compute_intraday_state(out, median_step)
    out = _rolling_transition_features(out, median_step)

    train_mask = pd.to_datetime(out["target_date_local"]) <= pd.Timestamp(train_end_date)
    out = _add_climo_features(out, train_mask)

    out = _add_spatial_features(out, target_station)

    # Final feature list
    drop_cols = {
        "request_location_id",
        "station_id",
        "valid_time_ny",
        "valid_time_stockholm",
        "last_tmax_time",
        "last_state_change_time",
        "y_tmax",
        "target_date_local",
        "valid_time_utc",
        "wx_phrase",
        "uv_desc",
        "clds",
        "wdir_cardinal",
    }
    non_feature_prefixes = ("K",)

    feature_cols: list[str] = []
    for c in out.columns:
        if c in drop_cols:
            continue
        if c.endswith("_source_valid_time_utc"):
            continue
        if c in {"doy", "stockholm_minutes"}:
            continue
        if out[c].dtype == "O" or str(out[c].dtype).startswith("string"):
            continue
        if c.startswith(non_feature_prefixes) and c.split("_")[0] in STATION_COORDS:
            # include neighbor numeric columns deliberately
            pass
        feature_cols.append(c)

    feature_cols = sorted(set(feature_cols))

    null_report = {
        "total_rows": int(len(out)),
        "feature_null_fraction": {
            c: float(out[c].isna().mean()) for c in feature_cols
        },
        "median_step_minutes": median_step,
    }
    return out, feature_cols, null_report
