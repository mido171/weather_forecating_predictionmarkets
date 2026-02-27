from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Any
import logging
import re

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


CLDS_CODE_ORDER = ("CLR", "FEW", "SCT", "BKN", "OVC")
CLDS_OKTAS_MIDPOINT = {
    "CLR": 0.0,
    "FEW": 1.5,
    "SCT": 3.5,
    "BKN": 6.0,
    "OVC": 8.0,
}
ALLOWED_CLDS_NORM = set(CLDS_CODE_ORDER) | {"UNK"}

CARDINAL_DEGREES = {
    "N": 0.0,
    "NNE": 22.5,
    "NE": 45.0,
    "ENE": 67.5,
    "E": 90.0,
    "ESE": 112.5,
    "SE": 135.0,
    "SSE": 157.5,
    "S": 180.0,
    "SSW": 202.5,
    "SW": 225.0,
    "WSW": 247.5,
    "W": 270.0,
    "WNW": 292.5,
    "NW": 315.0,
    "NNW": 337.5,
}
ALLOWED_CARDINALS = set(CARDINAL_DEGREES.keys()) | {"CALM", "VAR"}

WX_FLAG_ORDER = (
    "wx_has_light",
    "wx_has_heavy",
    "wx_has_rain",
    "wx_has_drizzle",
    "wx_has_snow",
    "wx_has_sleet",
    "wx_has_freezing",
    "wx_has_hail",
    "wx_has_wintry_mix",
    "wx_has_tstorm",
    "wx_has_thunder",
    "wx_has_funnel",
    "wx_has_tornado",
    "wx_has_squalls",
    "wx_has_fog",
    "wx_has_mist",
    "wx_has_haze",
    "wx_has_smoke",
    "wx_has_dust",
    "wx_has_windy",
)
WX_PRECIP_FLAGS = (
    "wx_has_rain",
    "wx_has_drizzle",
    "wx_has_snow",
    "wx_has_sleet",
    "wx_has_freezing",
    "wx_has_hail",
    "wx_has_wintry_mix",
)


@dataclass(frozen=True)
class StationSeries:
    station_id: str
    times_ns: np.ndarray
    values: dict[str, np.ndarray]
    texts: dict[str, np.ndarray]
    last_valid_idx: dict[str, np.ndarray]
    wx_unseen_counts: dict[str, int]


def _station_short(station_id: str) -> str:
    return station_id.split(":", 1)[0]


def _sanitize_text(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, float) and np.isnan(raw):
        return ""
    return str(raw).strip()


def _build_last_valid_index(valid_mask: np.ndarray) -> np.ndarray:
    out = np.full(valid_mask.shape[0], -1, dtype=np.int32)
    last = -1
    for i, ok in enumerate(valid_mask.tolist()):
        if ok:
            last = i
        out[i] = last
    return out


def _normalize_clds(raw: Any) -> tuple[str, float, float, float]:
    txt = _sanitize_text(raw)
    if txt == "":
        return "UNK", np.nan, 1.0, 0.0
    token = txt.upper().replace("-", " ").split()[0]
    token = {"SKC": "CLR", "OVR": "OVC", "VV": "OVC"}.get(token, token)
    if token in CLDS_OKTAS_MIDPOINT:
        return token, float(CLDS_OKTAS_MIDPOINT[token]), 0.0, 0.0
    return "UNK", np.nan, 0.0, 1.0


def _parse_wx_phrase(raw: Any) -> dict[str, Any]:
    txt_raw = _sanitize_text(raw)
    if txt_raw == "":
        out = {k: 0.0 for k in WX_FLAG_ORDER}
        out["wx_missing"] = 1.0
        out["wx_coarse_code"] = 0.0
        out["wx_phrase_norm"] = ""
        out["is_unseen"] = False
        return out

    phrase = re.sub(r"\s+", " ", txt_raw.strip().lower())
    tokens = [t.strip() for t in re.split(r"\s*/\s*", phrase) if t.strip()]
    joined = " | ".join(tokens) if tokens else phrase

    def has_any(parts: tuple[str, ...]) -> bool:
        return any(p in joined for p in parts)

    flags: dict[str, float] = {
        "wx_has_light": float(has_any(("light",))),
        "wx_has_heavy": float(has_any(("heavy",))),
        "wx_has_rain": float(has_any(("rain", "showers", "showers in the vicinity"))),
        "wx_has_drizzle": float(has_any(("drizzle",))),
        "wx_has_snow": float(has_any(("snow",))),
        "wx_has_sleet": float(has_any(("sleet",))),
        "wx_has_freezing": float(has_any(("freezing", "ice pellets", "freezing rain"))),
        "wx_has_hail": float(has_any(("hail",))),
        "wx_has_wintry_mix": float(has_any(("wintry mix", "wintry", "mix"))),
        "wx_has_tstorm": float(has_any(("tstorm", "thunderstorm", "storms"))),
        "wx_has_thunder": float(has_any(("thunder",))),
        "wx_has_funnel": float(has_any(("funnel",))),
        "wx_has_tornado": float(has_any(("tornado",))),
        "wx_has_squalls": float(has_any(("squall", "squalls"))),
        "wx_has_fog": float(has_any(("fog",))),
        "wx_has_mist": float(has_any(("mist",))),
        "wx_has_haze": float(has_any(("haze",))),
        "wx_has_smoke": float(has_any(("smoke",))),
        "wx_has_dust": float(has_any(("dust",))),
        "wx_has_windy": float(has_any(("windy",))),
    }

    has_obstruction = any(
        flags[k] > 0
        for k in ("wx_has_fog", "wx_has_mist", "wx_has_haze", "wx_has_smoke", "wx_has_dust")
    )
    has_rain = any(flags[k] > 0 for k in ("wx_has_rain", "wx_has_drizzle"))
    has_frozen = any(
        flags[k] > 0
        for k in ("wx_has_snow", "wx_has_sleet", "wx_has_freezing", "wx_has_wintry_mix", "wx_has_hail")
    )
    has_thunder = flags["wx_has_tstorm"] > 0 or flags["wx_has_thunder"] > 0
    has_severe = any(flags[k] > 0 for k in ("wx_has_tornado", "wx_has_funnel", "wx_has_squalls"))
    has_cloud = has_any(("fair", "partly", "mostly", "cloud", "clear", "sunny", "overcast"))

    if has_severe:
        coarse = 6.0
    elif has_thunder:
        coarse = 5.0
    elif has_frozen:
        coarse = 4.0
    elif has_rain:
        coarse = 3.0
    elif has_obstruction:
        coarse = 2.0
    elif has_cloud:
        coarse = 1.0
    else:
        coarse = 0.0

    out = dict(flags)
    out["wx_missing"] = 0.0
    out["wx_coarse_code"] = float(coarse)
    out["wx_phrase_norm"] = phrase
    out["is_unseen"] = bool(coarse == 0.0 and sum(flags.values()) == 0.0)
    return out


def _clean_uv_index(raw: Any) -> tuple[float, float, float]:
    txt = _sanitize_text(raw)
    if txt == "":
        return np.nan, 0.0, 1.0
    val = pd.to_numeric(pd.Series([txt]), errors="coerce").iloc[0]
    if not np.isfinite(val):
        return np.nan, 1.0, 1.0
    val_f = float(val)
    if 0.0 <= val_f <= 20.0:
        return val_f, 0.0, 0.0
    return np.nan, 1.0, 1.0


def _encode_uv_desc(raw: Any) -> tuple[str, dict[str, float]]:
    txt = _sanitize_text(raw)
    if txt == "":
        return "", {
            "uv_desc_low": 0.0,
            "uv_desc_moderate": 0.0,
            "uv_desc_high": 0.0,
            "uv_desc_very_high": 0.0,
            "uv_desc_missing": 1.0,
        }
    norm = txt.strip().lower().replace("_", " ")
    mapped = {
        "low": "Low",
        "moderate": "Moderate",
        "high": "High",
        "very high": "Very High",
    }.get(norm, "")
    out = {
        "uv_desc_low": 1.0 if mapped == "Low" else 0.0,
        "uv_desc_moderate": 1.0 if mapped == "Moderate" else 0.0,
        "uv_desc_high": 1.0 if mapped == "High" else 0.0,
        "uv_desc_very_high": 1.0 if mapped == "Very High" else 0.0,
        "uv_desc_missing": 0.0 if mapped else 1.0,
    }
    return mapped, out


def _normalize_wdir_cardinal(raw: Any) -> tuple[str, float, float, float]:
    txt = _sanitize_text(raw)
    if txt == "":
        return "", 0.0, 0.0, 1.0
    norm = txt.upper().replace("-", "").replace(" ", "")
    if norm not in ALLOWED_CARDINALS:
        return "", 0.0, 0.0, 1.0
    return norm, float(1.0 if norm == "CALM" else 0.0), float(1.0 if norm == "VAR" else 0.0), 0.0


def _clean_feels_like(raw: Any) -> tuple[float, float, float]:
    txt = _sanitize_text(raw)
    if txt == "":
        return np.nan, 0.0, 1.0
    val = pd.to_numeric(pd.Series([txt]), errors="coerce").iloc[0]
    if not np.isfinite(val):
        return np.nan, 1.0, 1.0
    val_f = float(val)
    if -80.0 <= val_f <= 140.0:
        return val_f, 0.0, 0.0
    return np.nan, 1.0, 1.0


def prepare_station_series(
    obs_df: pd.DataFrame,
    *,
    station_ids: tuple[str, ...],
    include_feels_like: bool = False,
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
        "clds",
        "wx_phrase",
        "uv_index",
        "uv_desc",
        "wdir_cardinal",
    }
    if include_feels_like:
        required_cols.add("feels_like")
    missing = required_cols.difference(obs_df.columns)
    if missing:
        raise ValueError(f"Observation DataFrame missing columns: {sorted(missing)}")

    data = obs_df.copy()
    if "feels_like" not in data.columns:
        data["feels_like"] = np.nan

    out: dict[str, StationSeries] = {}
    for station_id in station_ids:
        sdf = data[data["request_location_id"] == station_id].copy()
        sdf = sdf.sort_values("valid_time_utc").reset_index(drop=True)
        times_ns = sdf["valid_time_utc"].astype("int64").to_numpy(dtype=np.int64)

        values: dict[str, np.ndarray] = {}
        texts: dict[str, np.ndarray] = {}
        last_valid_idx: dict[str, np.ndarray] = {}
        for col in ["temp", "dew_pt", "rh", "pressure", "vis", "wspd", "wdir", "gust", "precip_hrly"]:
            arr = pd.to_numeric(sdf[col], errors="coerce").to_numpy(dtype=float)
            values[col] = arr
            last_valid_idx[col] = _build_last_valid_index(np.isfinite(arr))

        clds_norm: list[str] = []
        clds_oktas: list[float] = []
        clds_missing: list[float] = []
        clds_unk: list[float] = []
        wx_payloads: dict[str, list[float]] = {k: [] for k in WX_FLAG_ORDER}
        wx_payloads["wx_missing"] = []
        wx_payloads["wx_coarse_code"] = []
        wx_phrase_norm: list[str] = []
        unseen_counts: Counter[str] = Counter()

        uv_clean: list[float] = []
        uv_invalid: list[float] = []
        uv_missing: list[float] = []
        uv_desc_norm: list[str] = []
        uv_desc_cols = {
            "uv_desc_low": [],
            "uv_desc_moderate": [],
            "uv_desc_high": [],
            "uv_desc_very_high": [],
            "uv_desc_missing": [],
        }

        wdir_cardinal_norm: list[str] = []
        wdir_is_calm: list[float] = []
        wdir_is_var: list[float] = []
        wdir_cardinal_missing: list[float] = []
        wdir_filled: list[float] = []
        wdir_filled_from_cardinal: list[float] = []

        feels_like_clean: list[float] = []
        feels_like_invalid: list[float] = []
        feels_like_missing: list[float] = []

        raw_wdir = values["wdir"]
        raw_clds = sdf["clds"].to_numpy(dtype=object)
        raw_wx = sdf["wx_phrase"].to_numpy(dtype=object)
        raw_uv = sdf["uv_index"].to_numpy(dtype=object)
        raw_uv_desc = sdf["uv_desc"].to_numpy(dtype=object)
        raw_card = sdf["wdir_cardinal"].to_numpy(dtype=object)
        raw_feels = sdf["feels_like"].to_numpy(dtype=object)

        for i in range(len(sdf)):
            clds_n, clds_o, clds_m, clds_u = _normalize_clds(raw_clds[i])
            if clds_n not in ALLOWED_CLDS_NORM:
                raise AssertionError(f"CLDS vocabulary guard failed at parse time: {clds_n}")
            clds_norm.append(clds_n)
            clds_oktas.append(clds_o)
            clds_missing.append(clds_m)
            clds_unk.append(clds_u)

            wx = _parse_wx_phrase(raw_wx[i])
            for key in WX_FLAG_ORDER:
                wx_payloads[key].append(float(wx[key]))
            wx_payloads["wx_missing"].append(float(wx["wx_missing"]))
            wx_payloads["wx_coarse_code"].append(float(wx["wx_coarse_code"]))
            wx_phrase_norm.append(str(wx["wx_phrase_norm"]))
            if wx["is_unseen"]:
                unseen_counts[str(wx["wx_phrase_norm"])] += 1

            uv_v, uv_inv, uv_miss = _clean_uv_index(raw_uv[i])
            uv_clean.append(uv_v)
            uv_invalid.append(uv_inv)
            uv_missing.append(uv_miss)

            uv_desc_v, uv_desc_flags = _encode_uv_desc(raw_uv_desc[i])
            uv_desc_norm.append(uv_desc_v)
            for key, val in uv_desc_flags.items():
                uv_desc_cols[key].append(float(val))

            cardinal, is_calm, is_var, card_miss = _normalize_wdir_cardinal(raw_card[i])
            wdir_cardinal_norm.append(cardinal)
            wdir_is_calm.append(is_calm)
            wdir_is_var.append(is_var)
            wdir_cardinal_missing.append(card_miss)

            wdir_val = float(raw_wdir[i]) if np.isfinite(raw_wdir[i]) else np.nan
            filled = 0.0
            if not np.isfinite(wdir_val) and cardinal in CARDINAL_DEGREES:
                wdir_val = CARDINAL_DEGREES[cardinal]
                filled = 1.0
            wdir_filled.append(wdir_val)
            wdir_filled_from_cardinal.append(filled)

            if include_feels_like:
                fl_v, fl_inv, fl_miss = _clean_feels_like(raw_feels[i])
            else:
                fl_v, fl_inv, fl_miss = np.nan, 0.0, 1.0
            feels_like_clean.append(fl_v)
            feels_like_invalid.append(fl_inv)
            feels_like_missing.append(fl_miss)

        values["clds_oktas"] = np.asarray(clds_oktas, dtype=float)
        values["clds_missing"] = np.asarray(clds_missing, dtype=float)
        values["clds_unk"] = np.asarray(clds_unk, dtype=float)
        for code in CLDS_CODE_ORDER:
            values[f"clds_is_{code}"] = np.asarray([1.0 if c == code else 0.0 for c in clds_norm], dtype=float)

        for key, vals in wx_payloads.items():
            values[key] = np.asarray(vals, dtype=float)

        values["uv_index_clean"] = np.asarray(uv_clean, dtype=float)
        values["uv_invalid"] = np.asarray(uv_invalid, dtype=float)
        values["uv_missing"] = np.asarray(uv_missing, dtype=float)
        for key, vals in uv_desc_cols.items():
            values[key] = np.asarray(vals, dtype=float)

        values["wdir_filled"] = np.asarray(wdir_filled, dtype=float)
        values["wdir_is_calm"] = np.asarray(wdir_is_calm, dtype=float)
        values["wdir_is_var"] = np.asarray(wdir_is_var, dtype=float)
        values["wdir_cardinal_missing"] = np.asarray(wdir_cardinal_missing, dtype=float)
        values["wdir_filled_from_cardinal"] = np.asarray(wdir_filled_from_cardinal, dtype=float)

        values["feels_like_clean"] = np.asarray(feels_like_clean, dtype=float)
        values["feels_like_invalid"] = np.asarray(feels_like_invalid, dtype=float)
        values["feels_like_missing"] = np.asarray(feels_like_missing, dtype=float)

        texts["clds_norm"] = np.asarray(clds_norm, dtype=object)
        texts["wx_phrase_norm"] = np.asarray(wx_phrase_norm, dtype=object)
        texts["uv_desc_norm"] = np.asarray(uv_desc_norm, dtype=object)
        texts["wdir_cardinal_norm"] = np.asarray(wdir_cardinal_norm, dtype=object)

        for key, arr in values.items():
            if key in last_valid_idx:
                continue
            last_valid_idx[key] = _build_last_valid_index(np.isfinite(arr))
        for key, arr in texts.items():
            valid = np.asarray([_sanitize_text(v) != "" for v in arr], dtype=bool)
            last_valid_idx[key] = _build_last_valid_index(valid)

        out[station_id] = StationSeries(
            station_id=station_id,
            times_ns=times_ns,
            values=values,
            texts=texts,
            last_valid_idx=last_valid_idx,
            wx_unseen_counts={k: int(v) for k, v in unseen_counts.items()},
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
    if df["target_date_local"].duplicated().any():
        raise ValueError("Daily truth has duplicate target_date_local rows; cannot build leakage-safe priors.")

    history: list[float] = []
    rows: list[dict[str, Any]] = []
    for rec in df.itertuples(index=False):
        rows.append(
            {
                "target_date_local": rec.target_date_local,
                "tmax_yday": float(history[-1]) if len(history) >= 1 else np.nan,
                "tmax_2day": float(history[-2]) if len(history) >= 2 else np.nan,
                "tmax_mean_7d": float(np.mean(history[-7:])) if len(history) >= 1 else np.nan,
                "tmax_mean_30d": float(np.mean(history[-30:])) if len(history) >= 1 else np.nan,
                "tmax_std_30d": float(np.std(history[-30:], ddof=0)) if len(history) >= 2 else np.nan,
            }
        )
        cur = pd.to_numeric(pd.Series([rec.max_temp_f]), errors="coerce").iloc[0]
        if np.isfinite(cur):
            history.append(float(cur))
    return pd.DataFrame(rows)


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


def _text_at_or_before(
    station: StationSeries,
    *,
    column: str,
    target_ns: int,
) -> tuple[str, int | None, int]:
    idx = int(np.searchsorted(station.times_ns, target_ns, side="right") - 1)
    if idx < 0:
        return "", None, -1
    lv = int(station.last_valid_idx[column][idx])
    if lv < 0:
        return "", None, -1
    return _sanitize_text(station.texts[column][lv]), int(station.times_ns[lv]), lv


def _window_slice_indices(
    station: StationSeries,
    *,
    start_ns: int,
    end_ns: int,
) -> tuple[int, int]:
    i0 = int(np.searchsorted(station.times_ns, start_ns, side="left"))
    i1 = int(np.searchsorted(station.times_ns, end_ns, side="right"))
    return i0, i1


def _window_numeric_values(
    station: StationSeries,
    *,
    column: str,
    start_ns: int,
    end_ns: int,
) -> tuple[np.ndarray, np.ndarray]:
    i0, i1 = _window_slice_indices(station, start_ns=start_ns, end_ns=end_ns)
    if i1 <= i0:
        return np.asarray([], dtype=float), np.asarray([], dtype=np.int64)
    return station.values[column][i0:i1], station.times_ns[i0:i1]


def _window_text_values(
    station: StationSeries,
    *,
    column: str,
    start_ns: int,
    end_ns: int,
) -> tuple[np.ndarray, np.ndarray]:
    i0, i1 = _window_slice_indices(station, start_ns=start_ns, end_ns=end_ns)
    if i1 <= i0:
        return np.asarray([], dtype=object), np.asarray([], dtype=np.int64)
    return station.texts[column][i0:i1], station.times_ns[i0:i1]


def _window_stats(
    station: StationSeries,
    *,
    column: str,
    start_ns: int,
    end_ns: int,
) -> tuple[float, float, float, float, float, int | None]:
    vals, times = _window_numeric_values(station, column=column, start_ns=start_ns, end_ns=end_ns)
    if vals.size == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, None
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, int(times[-1])
    std = float(np.std(finite, ddof=0))
    min_v = float(np.min(finite))
    max_v = float(np.max(finite))
    rng = float(max_v - min_v)
    if finite.size >= 2:
        std_diff = float(np.std(np.diff(finite), ddof=0))
    else:
        std_diff = np.nan
    return std, min_v, max_v, rng, std_diff, int(times[-1])


def _local_minute_of_day(ts_ns: int, local_zone) -> int:
    ts = pd.Timestamp(ts_ns, tz="UTC").tz_convert(local_zone)
    return int(ts.hour * 60 + ts.minute)


def _compute_station_snapshot(
    station: StationSeries,
    *,
    cutoff_ns: int,
    cutoff_minutes: int,
    include_regime: bool = True,
) -> tuple[dict[str, Any], int | None]:
    features: dict[str, Any] = {}
    max_used_ns: int | None = None

    temp_now, temp_ts_ns, _ = _value_at_or_before(
        station, column="temp", target_ns=cutoff_ns
    )
    max_used_ns = _update_max_used(max_used_ns, temp_ts_ns)
    features["temp_now"] = temp_now
    features["is_temp_missing_now"] = 0.0 if np.isfinite(temp_now) else 1.0
    features["age_min_temp"] = (
        float((cutoff_ns - temp_ts_ns) / 60_000_000_000.0) if temp_ts_ns is not None else np.nan
    )

    row_cols = ["dew_pt", "rh", "pressure", "vis", "wspd", "wdir_filled", "gust", "precip_hrly"]
    for col in row_cols:
        key = "dewpt_now" if col == "dew_pt" else f"{col}_now"
        if col == "wdir_filled":
            key = "wdir_now"
        miss_key = f"is_{col}_missing_now"
        if col == "dew_pt":
            miss_key = "is_dew_pt_missing_now"
        if col == "wdir_filled":
            miss_key = "is_wdir_missing_now"
        raw, ts_ns, _ = _value_at_or_before(station, column=col, target_ns=cutoff_ns)
        max_used_ns = _update_max_used(max_used_ns, ts_ns)
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

    if include_regime:
        # Regime sensors.
        clds_norm, clds_ts_ns, _ = _text_at_or_before(station, column="clds_norm", target_ns=cutoff_ns)
        clds_oktas, clds_oktas_ts_ns, _ = _value_at_or_before(station, column="clds_oktas", target_ns=cutoff_ns)
        clds_missing_now, clds_missing_ts_ns, _ = _value_at_or_before(
            station, column="clds_missing", target_ns=cutoff_ns
        )
        clds_unk_now, clds_unk_ts_ns, _ = _value_at_or_before(station, column="clds_unk", target_ns=cutoff_ns)
        max_used_ns = _update_max_used(max_used_ns, clds_ts_ns)
        max_used_ns = _update_max_used(max_used_ns, clds_oktas_ts_ns)
        max_used_ns = _update_max_used(max_used_ns, clds_missing_ts_ns)
        max_used_ns = _update_max_used(max_used_ns, clds_unk_ts_ns)

        clds_norm = clds_norm if clds_norm else "UNK"
        if clds_norm not in ALLOWED_CLDS_NORM:
            raise AssertionError(f"CLDS vocabulary guard failed: {clds_norm}")
        features["clds_norm_now"] = clds_norm
        features["clds_oktas_now"] = clds_oktas
        features["clds_missing_now"] = float(clds_missing_now) if np.isfinite(clds_missing_now) else 1.0
        features["clds_unk_now"] = float(clds_unk_now) if np.isfinite(clds_unk_now) else 0.0
        for code in CLDS_CODE_ORDER:
            features[f"clds_is_{code}"] = 1.0 if clds_norm == code else 0.0

        for key in WX_FLAG_ORDER:
            v, ts_ns, _ = _value_at_or_before(station, column=key, target_ns=cutoff_ns)
            features[key] = float(v) if np.isfinite(v) else 0.0
            max_used_ns = _update_max_used(max_used_ns, ts_ns)
        wx_missing_now, ts_ns, _ = _value_at_or_before(station, column="wx_missing", target_ns=cutoff_ns)
        features["wx_missing_now"] = float(wx_missing_now) if np.isfinite(wx_missing_now) else 1.0
        max_used_ns = _update_max_used(max_used_ns, ts_ns)
        wx_coarse_now, ts_ns, _ = _value_at_or_before(station, column="wx_coarse_code", target_ns=cutoff_ns)
        features["wx_coarse_code"] = float(wx_coarse_now) if np.isfinite(wx_coarse_now) else 0.0
        max_used_ns = _update_max_used(max_used_ns, ts_ns)

        uv_now, uv_ts_ns, _ = _value_at_or_before(station, column="uv_index_clean", target_ns=cutoff_ns)
        uv_invalid_now, uv_invalid_ts_ns, _ = _value_at_or_before(station, column="uv_invalid", target_ns=cutoff_ns)
        uv_missing_now, uv_missing_ts_ns, _ = _value_at_or_before(station, column="uv_missing", target_ns=cutoff_ns)
        max_used_ns = _update_max_used(max_used_ns, uv_ts_ns)
        max_used_ns = _update_max_used(max_used_ns, uv_invalid_ts_ns)
        max_used_ns = _update_max_used(max_used_ns, uv_missing_ts_ns)
        features["uv_index_now"] = uv_now
        features["uv_invalid_flag"] = float(uv_invalid_now) if np.isfinite(uv_invalid_now) else 0.0
        features["uv_missing_now"] = float(uv_missing_now) if np.isfinite(uv_missing_now) else 1.0
        if features["uv_missing_now"] < 0.5 and (not np.isfinite(uv_now) or uv_now < 0.0 or uv_now > 20.0):
            raise AssertionError(f"UV sanity guard failed: uv_index_now={uv_now}")
        for key in ["uv_desc_low", "uv_desc_moderate", "uv_desc_high", "uv_desc_very_high", "uv_desc_missing"]:
            v, ts_ns, _ = _value_at_or_before(station, column=key, target_ns=cutoff_ns)
            features[key] = float(v) if np.isfinite(v) else 0.0
            max_used_ns = _update_max_used(max_used_ns, ts_ns)

        for key in ["wdir_is_calm", "wdir_is_var", "wdir_filled_from_cardinal", "wdir_cardinal_missing"]:
            v, ts_ns, _ = _value_at_or_before(station, column=key, target_ns=cutoff_ns)
            features[key] = float(v) if np.isfinite(v) else 0.0
            max_used_ns = _update_max_used(max_used_ns, ts_ns)

        if station.values["feels_like_missing"].size:
            fl_now, fl_ts_ns, _ = _value_at_or_before(station, column="feels_like_clean", target_ns=cutoff_ns)
            fl_invalid, fl_invalid_ts_ns, _ = _value_at_or_before(
                station, column="feels_like_invalid", target_ns=cutoff_ns
            )
            fl_missing, fl_missing_ts_ns, _ = _value_at_or_before(
                station, column="feels_like_missing", target_ns=cutoff_ns
            )
            max_used_ns = _update_max_used(max_used_ns, fl_ts_ns)
            max_used_ns = _update_max_used(max_used_ns, fl_invalid_ts_ns)
            max_used_ns = _update_max_used(max_used_ns, fl_missing_ts_ns)
            features["feels_like_now"] = fl_now
            features["feels_like_invalid_flag"] = float(fl_invalid) if np.isfinite(fl_invalid) else 0.0
            features["feels_like_missing_now"] = float(fl_missing) if np.isfinite(fl_missing) else 1.0
            features["feels_like_minus_temp_now"] = _safe_diff(fl_now, features.get("temp_now", np.nan))

    return features, max_used_ns


def _compute_station_full_features(
    station: StationSeries,
    *,
    cutoff_ns: int,
    midnight_ns: int,
    cutoff_minutes: int,
    n_expected_bins: int,
    windows_minutes: tuple[int, ...],
    local_zone,
    enable_regime: bool = True,
    enable_v2_dynamics: bool = True,
) -> tuple[dict[str, Any], int | None]:
    snapshot, max_used_ns = _compute_station_snapshot(
        station,
        cutoff_ns=cutoff_ns,
        cutoff_minutes=cutoff_minutes,
        include_regime=enable_regime,
    )
    features: dict[str, Any] = dict(snapshot)

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

    regime_dynamics = enable_regime and enable_v2_dynamics
    if regime_dynamics:
        regime_windows = (60, 180, 360)
        wx_windows = (180, 360)
        vis_now = float(features.get("vis_now", np.nan))
        precip_now = float(features.get("precip_hrly_now", np.nan))
        clds_now = float(features.get("clds_oktas_now", np.nan))
        uv_now = float(features.get("uv_index_now", np.nan))
        wdir_now = float(features.get("wdir_now", np.nan))
        sin_now = float(np.sin(np.deg2rad(wdir_now))) if np.isfinite(wdir_now) else np.nan
        cos_now = float(np.cos(np.deg2rad(wdir_now))) if np.isfinite(wdir_now) else np.nan

        for w in regime_windows:
            w_ns = int(w * 60 * 1_000_000_000)
            start_ns = cutoff_ns - w_ns

            vis_prev, vis_prev_ts, _ = _value_at_or_before(station, column="vis", target_ns=start_ns)
            max_used_ns = _update_max_used(max_used_ns, vis_prev_ts)
            features[f"vis_prev_{w}"] = vis_prev
            features[f"vis_delta_{w}"] = _safe_diff(vis_now, vis_prev)
            features[f"vis_slope_{w}"] = (
                float(features[f"vis_delta_{w}"] / w) if np.isfinite(features[f"vis_delta_{w}"]) else np.nan
            )
            vis_win, vis_times = _window_numeric_values(station, column="vis", start_ns=start_ns, end_ns=cutoff_ns)
            vis_fin = vis_win[np.isfinite(vis_win)]
            features[f"vis_min_{w}"] = float(np.min(vis_fin)) if vis_fin.size else np.nan
            features[f"vis_max_{w}"] = float(np.max(vis_fin)) if vis_fin.size else np.nan
            features[f"vis_range_{w}"] = _safe_diff(features[f"vis_max_{w}"], features[f"vis_min_{w}"])
            features[f"vis_std_{w}"] = float(np.std(vis_fin, ddof=0)) if vis_fin.size else np.nan
            features[f"vis_missing_frac_{w}"] = _frac_missing(vis_win)
            if vis_times.size:
                max_used_ns = _update_max_used(max_used_ns, int(vis_times[-1]))

            precip_prev, precip_prev_ts, _ = _value_at_or_before(station, column="precip_hrly", target_ns=start_ns)
            max_used_ns = _update_max_used(max_used_ns, precip_prev_ts)
            features[f"precip_prev_{w}"] = precip_prev
            features[f"precip_delta_{w}"] = _safe_diff(precip_now, precip_prev)
            precip_win, precip_times = _window_numeric_values(
                station,
                column="precip_hrly",
                start_ns=start_ns,
                end_ns=cutoff_ns,
            )
            precip_fin = precip_win[np.isfinite(precip_win)]
            features[f"precip_mean_{w}"] = float(np.mean(precip_fin)) if precip_fin.size else np.nan
            features[f"precip_max_{w}"] = float(np.max(precip_fin)) if precip_fin.size else np.nan
            features[f"precip_nonzero_frac_{w}"] = float(np.mean(precip_fin > 0.0)) if precip_fin.size else 0.0
            features[f"precip_any_{w}"] = float(1.0 if (precip_fin > 0.0).any() else 0.0)
            if precip_times.size:
                max_used_ns = _update_max_used(max_used_ns, int(precip_times[-1]))

            clds_prev, clds_prev_ts, _ = _value_at_or_before(station, column="clds_oktas", target_ns=start_ns)
            max_used_ns = _update_max_used(max_used_ns, clds_prev_ts)
            features[f"clds_oktas_prev_{w}"] = clds_prev
            features[f"clds_oktas_delta_{w}"] = _safe_diff(clds_now, clds_prev)
            clds_oktas_win, clds_times = _window_numeric_values(
                station,
                column="clds_oktas",
                start_ns=start_ns,
                end_ns=cutoff_ns,
            )
            clds_norm_win, _ = _window_text_values(
                station,
                column="clds_norm",
                start_ns=start_ns,
                end_ns=cutoff_ns,
            )
            clds_fin = clds_oktas_win[np.isfinite(clds_oktas_win)]
            features[f"clds_oktas_mean_{w}"] = float(np.mean(clds_fin)) if clds_fin.size else np.nan
            features[f"clds_oktas_min_{w}"] = float(np.min(clds_fin)) if clds_fin.size else np.nan
            features[f"clds_oktas_max_{w}"] = float(np.max(clds_fin)) if clds_fin.size else np.nan
            features[f"clds_oktas_std_{w}"] = float(np.std(clds_fin, ddof=0)) if clds_fin.size else np.nan
            clds_norm_clean = [_sanitize_text(v) for v in clds_norm_win.tolist()]
            known = [c for c in clds_norm_clean if c in CLDS_CODE_ORDER]
            denom = max(len(known), 1)
            for code in CLDS_CODE_ORDER:
                features[f"clds_frac_{code}_{w}"] = float(sum(1 for c in known if c == code) / denom) if known else 0.0
            trans_seq = [c for c in clds_norm_clean if c != ""]
            features[f"clds_transitions_{w}"] = float(
                sum(1 for j in range(1, len(trans_seq)) if trans_seq[j] != trans_seq[j - 1])
            )
            clds_missing_win, _ = _window_numeric_values(
                station,
                column="clds_missing",
                start_ns=start_ns,
                end_ns=cutoff_ns,
            )
            features[f"clds_missing_frac_{w}"] = float(np.nanmean(clds_missing_win)) if clds_missing_win.size else 1.0
            if clds_times.size:
                max_used_ns = _update_max_used(max_used_ns, int(clds_times[-1]))

            uv_prev, uv_prev_ts, _ = _value_at_or_before(station, column="uv_index_clean", target_ns=start_ns)
            max_used_ns = _update_max_used(max_used_ns, uv_prev_ts)
            features[f"uv_prev_{w}"] = uv_prev
            features[f"uv_delta_{w}"] = _safe_diff(uv_now, uv_prev)
            features[f"uv_slope_{w}"] = (
                float(features[f"uv_delta_{w}"] / w) if np.isfinite(features[f"uv_delta_{w}"]) else np.nan
            )
            uv_win, uv_times = _window_numeric_values(
                station,
                column="uv_index_clean",
                start_ns=start_ns,
                end_ns=cutoff_ns,
            )
            uv_fin = uv_win[np.isfinite(uv_win)]
            features[f"uv_mean_{w}"] = float(np.mean(uv_fin)) if uv_fin.size else np.nan
            features[f"uv_max_{w}"] = float(np.max(uv_fin)) if uv_fin.size else np.nan
            features[f"uv_std_{w}"] = float(np.std(uv_fin, ddof=0)) if uv_fin.size else np.nan
            features[f"uv_missing_frac_{w}"] = _frac_missing(uv_win)
            if uv_times.size:
                max_used_ns = _update_max_used(max_used_ns, int(uv_times[-1]))

            wdir_prev, wdir_prev_ts, _ = _value_at_or_before(station, column="wdir_filled", target_ns=start_ns)
            max_used_ns = _update_max_used(max_used_ns, wdir_prev_ts)
            sin_prev = float(np.sin(np.deg2rad(wdir_prev))) if np.isfinite(wdir_prev) else np.nan
            cos_prev = float(np.cos(np.deg2rad(wdir_prev))) if np.isfinite(wdir_prev) else np.nan
            features[f"wdir_sin_prev_{w}"] = sin_prev
            features[f"wdir_cos_prev_{w}"] = cos_prev
            features[f"wdir_sin_delta_{w}"] = _safe_diff(sin_now, sin_prev)
            features[f"wdir_cos_delta_{w}"] = _safe_diff(cos_now, cos_prev)
            features[f"wdir_angdiff_{w}"] = _signed_angle_diff_deg(wdir_now, wdir_prev)
            wdir_win, wdir_times = _window_numeric_values(
                station,
                column="wdir_filled",
                start_ns=start_ns,
                end_ns=cutoff_ns,
            )
            wdir_fin = wdir_win[np.isfinite(wdir_win)]
            if wdir_fin.size:
                rad = np.deg2rad(wdir_fin)
                r = float(np.sqrt(np.mean(np.sin(rad)) ** 2 + np.mean(np.cos(rad)) ** 2))
                features[f"wdir_var_{w}"] = float(1.0 - r)
            else:
                features[f"wdir_var_{w}"] = np.nan
            if wdir_times.size:
                max_used_ns = _update_max_used(max_used_ns, int(wdir_times[-1]))

    if regime_dynamics:
        sofar_vis, sofar_vis_times = _window_numeric_values(
            station,
            column="vis",
            start_ns=midnight_ns,
            end_ns=cutoff_ns,
        )
        features["vis_last_change_min"] = _minutes_since_last_change_numeric(
            times_ns=sofar_vis_times,
            values=sofar_vis,
            cutoff_ns=cutoff_ns,
        )

        sofar_precip, sofar_precip_times = _window_numeric_values(
            station,
            column="precip_hrly",
            start_ns=midnight_ns,
            end_ns=cutoff_ns,
        )
        features["precip_onset_min_today"] = _onset_minutes_today(
            times_ns=sofar_precip_times,
            event_mask=np.isfinite(sofar_precip) & (sofar_precip > 0.0),
            cutoff_ns=cutoff_ns,
        )

        sofar_clds, sofar_clds_times = _window_text_values(
            station,
            column="clds_norm",
            start_ns=midnight_ns,
            end_ns=cutoff_ns,
        )
        features["clds_last_change_min"] = _minutes_since_last_change_text(
            times_ns=sofar_clds_times,
            values=sofar_clds,
            cutoff_ns=cutoff_ns,
        )
        features["is_persistently_OVC_180"] = (
            1.0 if np.isfinite(features.get("clds_frac_OVC_180", np.nan)) and features.get("clds_frac_OVC_180", 0.0) >= 0.8 else 0.0
        )
        last_180_text, _ = _window_text_values(
            station,
            column="clds_norm",
            start_ns=cutoff_ns - int(180 * 60 * 1_000_000_000),
            end_ns=cutoff_ns,
        )
        prev_180_text, _ = _window_text_values(
            station,
            column="clds_norm",
            start_ns=cutoff_ns - int(360 * 60 * 1_000_000_000),
            end_ns=cutoff_ns - int(180 * 60 * 1_000_000_000),
        )
        last_clean = [_sanitize_text(v) for v in last_180_text.tolist()]
        prev_clean = [_sanitize_text(v) for v in prev_180_text.tolist()]
        has_clear = any(v in {"CLR", "FEW"} for v in last_clean)
        prev_known = [v for v in prev_clean if v in CLDS_CODE_ORDER]
        prev_bkn_ovc = (
            float(sum(1 for v in prev_known if v in {"BKN", "OVC"}) / max(len(prev_known), 1))
            if prev_known
            else 0.0
        )
        features["clear_break_180"] = float(1.0 if has_clear and prev_bkn_ovc >= 0.5 else 0.0)

    if regime_dynamics:
        for w in wx_windows:
            w_ns = int(w * 60 * 1_000_000_000)
            start_ns = cutoff_ns - w_ns
            precip_flags = []
            for flag in WX_PRECIP_FLAGS:
                arr, _ = _window_numeric_values(station, column=flag, start_ns=start_ns, end_ns=cutoff_ns)
                precip_flags.append(arr)
            stacked = np.vstack([np.nan_to_num(a, nan=0.0) for a in precip_flags]) if precip_flags and precip_flags[0].size else np.zeros((len(precip_flags), 0))
            precip_any_row = np.max(stacked, axis=0) if stacked.size else np.asarray([], dtype=float)
            fog_arr, _ = _window_numeric_values(station, column="wx_has_fog", start_ns=start_ns, end_ns=cutoff_ns)
            haze_arr, _ = _window_numeric_values(station, column="wx_has_haze", start_ns=start_ns, end_ns=cutoff_ns)
            thunder_arr, _ = _window_numeric_values(station, column="wx_has_thunder", start_ns=start_ns, end_ns=cutoff_ns)
            tstorm_arr, _ = _window_numeric_values(station, column="wx_has_tstorm", start_ns=start_ns, end_ns=cutoff_ns)
            thunder_any_row = np.maximum(np.nan_to_num(thunder_arr, nan=0.0), np.nan_to_num(tstorm_arr, nan=0.0))

            features[f"wx_precip_any_{w}"] = float(1.0 if precip_any_row.size and np.any(precip_any_row > 0.0) else 0.0)
            features[f"wx_precip_frac_{w}"] = float(np.mean(precip_any_row > 0.0)) if precip_any_row.size else 0.0
            features[f"wx_fog_any_{w}"] = float(1.0 if fog_arr.size and np.any(fog_arr > 0.0) else 0.0)
            features[f"wx_fog_frac_{w}"] = float(np.mean(fog_arr > 0.0)) if fog_arr.size else 0.0
            features[f"wx_haze_any_{w}"] = float(1.0 if haze_arr.size and np.any(haze_arr > 0.0) else 0.0)
            features[f"wx_haze_frac_{w}"] = float(np.mean(haze_arr > 0.0)) if haze_arr.size else 0.0
            features[f"wx_thunder_any_{w}"] = float(1.0 if thunder_any_row.size and np.any(thunder_any_row > 0.0) else 0.0)
            features[f"wx_thunder_frac_{w}"] = float(np.mean(thunder_any_row > 0.0)) if thunder_any_row.size else 0.0

    if regime_dynamics:
        sofar_times = station.times_ns[
            _window_slice_indices(station, start_ns=midnight_ns, end_ns=cutoff_ns)[0] : _window_slice_indices(station, start_ns=midnight_ns, end_ns=cutoff_ns)[1]
        ]
        precip_today = np.zeros_like(sofar_times, dtype=bool)
        for flag in WX_PRECIP_FLAGS:
            arr, _ = _window_numeric_values(station, column=flag, start_ns=midnight_ns, end_ns=cutoff_ns)
            if arr.size:
                precip_today |= np.nan_to_num(arr, nan=0.0) > 0.0
        fog_today, _ = _window_numeric_values(station, column="wx_has_fog", start_ns=midnight_ns, end_ns=cutoff_ns)
        haze_today, _ = _window_numeric_values(station, column="wx_has_haze", start_ns=midnight_ns, end_ns=cutoff_ns)
        thunder_today, _ = _window_numeric_values(station, column="wx_has_thunder", start_ns=midnight_ns, end_ns=cutoff_ns)
        tstorm_today, _ = _window_numeric_values(station, column="wx_has_tstorm", start_ns=midnight_ns, end_ns=cutoff_ns)
        features["wx_precip_onset_min_today"] = _onset_minutes_today(
            times_ns=sofar_times,
            event_mask=precip_today,
            cutoff_ns=cutoff_ns,
        )
        features["wx_fog_onset_min_today"] = _onset_minutes_today(
            times_ns=sofar_times,
            event_mask=np.nan_to_num(fog_today, nan=0.0) > 0.0,
            cutoff_ns=cutoff_ns,
        )
        features["wx_haze_onset_min_today"] = _onset_minutes_today(
            times_ns=sofar_times,
            event_mask=np.nan_to_num(haze_today, nan=0.0) > 0.0,
            cutoff_ns=cutoff_ns,
        )
        features["wx_thunder_onset_min_today"] = _onset_minutes_today(
            times_ns=sofar_times,
            event_mask=np.maximum(np.nan_to_num(thunder_today, nan=0.0), np.nan_to_num(tstorm_today, nan=0.0)) > 0.0,
            cutoff_ns=cutoff_ns,
        )

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


def _safe_diff(a: float, b: float) -> float:
    if np.isfinite(a) and np.isfinite(b):
        return float(a - b)
    return np.nan


def _frac_missing(vals: np.ndarray) -> float:
    if vals.size == 0:
        return 1.0
    return float(np.mean(~np.isfinite(vals)))


def _minutes_since_last_change_numeric(
    *,
    times_ns: np.ndarray,
    values: np.ndarray,
    cutoff_ns: int,
    cap_minutes: float = 360.0,
) -> float:
    finite = np.isfinite(values)
    if not finite.any():
        return np.nan
    vals = values[finite]
    ts = times_ns[finite]
    last_change_ts = int(ts[0])
    prev = float(vals[0])
    for i in range(1, len(vals)):
        cur = float(vals[i])
        if not np.isclose(cur, prev, rtol=0.0, atol=1e-9):
            last_change_ts = int(ts[i])
            prev = cur
    mins = float((cutoff_ns - last_change_ts) / 60_000_000_000.0)
    return float(min(max(mins, 0.0), cap_minutes))


def _minutes_since_last_change_text(
    *,
    times_ns: np.ndarray,
    values: np.ndarray,
    cutoff_ns: int,
    cap_minutes: float = 360.0,
) -> float:
    cleaned = np.asarray([_sanitize_text(v) for v in values], dtype=object)
    valid = cleaned != ""
    if not valid.any():
        return np.nan
    vals = cleaned[valid]
    ts = times_ns[valid]
    last_change_ts = int(ts[0])
    prev = str(vals[0])
    for i in range(1, len(vals)):
        cur = str(vals[i])
        if cur != prev:
            last_change_ts = int(ts[i])
            prev = cur
    mins = float((cutoff_ns - last_change_ts) / 60_000_000_000.0)
    return float(min(max(mins, 0.0), cap_minutes))


def _onset_minutes_today(
    *,
    times_ns: np.ndarray,
    event_mask: np.ndarray,
    cutoff_ns: int,
    sentinel: float = 9999.0,
) -> float:
    idx = np.where(event_mask)[0]
    if idx.size == 0:
        return float(sentinel)
    first_ts = int(times_ns[int(idx[0])])
    return float(max((cutoff_ns - first_ts) / 60_000_000_000.0, 0.0))


def _signed_angle_diff_deg(now_deg: float, prev_deg: float) -> float:
    if not np.isfinite(now_deg) or not np.isfinite(prev_deg):
        return np.nan
    return float((now_deg - prev_deg + 180.0) % 360.0 - 180.0)


def _get_station_value(row: dict[str, Any], station_id: str, key: str) -> float:
    short = _station_short(station_id)
    return float(row.get(f"{short}_{key}", np.nan))


def _compute_group_mean(row: dict[str, Any], *, feature_name: str, stations: tuple[str, ...]) -> float:
    vals = [_get_station_value(row, sid, feature_name) for sid in stations]
    return _safe_mean(vals)


def _compute_group_or(row: dict[str, Any], *, feature_name: str, stations: tuple[str, ...]) -> float:
    vals = [_get_station_value(row, sid, feature_name) for sid in stations]
    finite = [v for v in vals if np.isfinite(v)]
    if not finite:
        return 0.0
    return float(1.0 if any(v > 0.0 for v in finite) else 0.0)


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

    if any(f"{_station_short(sid)}_clds_oktas_now" in row for sid in cfg.neighbor_station_ids):
        clds_coastal = _compute_group_mean(row, feature_name="clds_oktas_now", stations=COASTAL_STATIONS)
        clds_inland = _compute_group_mean(row, feature_name="clds_oktas_now", stations=INLAND_STATIONS)
        row["clds_oktas_coastal_mean"] = clds_coastal
        row["clds_oktas_inland_mean"] = clds_inland
        row["clds_oktas_coastal_minus_inland"] = _safe_diff(clds_coastal, clds_inland)

        uv_coastal = _compute_group_mean(row, feature_name="uv_index_now", stations=COASTAL_STATIONS)
        uv_inland = _compute_group_mean(row, feature_name="uv_index_now", stations=INLAND_STATIONS)
        row["uv_coastal_mean"] = uv_coastal
        row["uv_inland_mean"] = uv_inland
        row["uv_coastal_minus_inland"] = _safe_diff(uv_coastal, uv_inland)

        vis_coastal = _compute_group_mean(row, feature_name="vis_now", stations=COASTAL_STATIONS)
        vis_inland = _compute_group_mean(row, feature_name="vis_now", stations=INLAND_STATIONS)
        row["vis_coastal_mean"] = vis_coastal
        row["vis_inland_mean"] = vis_inland
        row["vis_coastal_minus_inland"] = _safe_diff(vis_coastal, vis_inland)

        precip_features = [
            "wx_has_rain",
            "wx_has_drizzle",
            "wx_has_snow",
            "wx_has_sleet",
            "wx_has_freezing",
            "wx_has_hail",
            "wx_has_wintry_mix",
        ]
        coastal_any_vals: list[float] = []
        inland_any_vals: list[float] = []
        for sid in COASTAL_STATIONS:
            short = _station_short(sid)
            coastal_any_vals.append(
                float(any(float(row.get(f"{short}_{p}", 0.0)) > 0.0 for p in precip_features))
            )
        for sid in INLAND_STATIONS:
            short = _station_short(sid)
            inland_any_vals.append(
                float(any(float(row.get(f"{short}_{p}", 0.0)) > 0.0 for p in precip_features))
            )
        row["wx_precip_any_coastal"] = float(1.0 if any(v > 0.0 for v in coastal_any_vals) else 0.0)
        row["wx_precip_any_inland"] = float(1.0 if any(v > 0.0 for v in inland_any_vals) else 0.0)
        row["precip_any_coastal"] = _compute_group_or(
            row, feature_name="precip_hrly_now", stations=COASTAL_STATIONS
        )
        row["precip_any_inland"] = _compute_group_or(
            row, feature_name="precip_hrly_now", stations=INLAND_STATIONS
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

    for sid in cfg.all_station_ids:
        unseen = station_series[sid].wx_unseen_counts
        if unseen:
            top = sorted(unseen.items(), key=lambda x: x[1], reverse=True)[:10]
            active_logger.warning(
                "WX_UNSEEN_PHRASES station=%s unique=%d top=%s",
                sid,
                len(unseen),
                top,
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
            enable_regime=cfg.enable_v2_regime_features,
            enable_v2_dynamics=cfg.enable_v2_vis_precip_wdir_dynamics,
        )
        row.update(klga_features)

        station_max_used: list[int | None] = [klga_max_used]
        for sid in cfg.neighbor_station_ids:
            if cfg.enable_neighbor_regime_features:
                snap, max_used_ns = _compute_station_full_features(
                    station_series[sid],
                    cutoff_ns=cutoff_ns,
                    midnight_ns=midnight_ns,
                    cutoff_minutes=cutoff_minutes,
                    n_expected_bins=n_expected_bins,
                    windows_minutes=cfg.windows_minutes,
                    local_zone=cfg.local_zone,
                    enable_regime=cfg.enable_v2_regime_features,
                    enable_v2_dynamics=cfg.enable_v2_vis_precip_wdir_dynamics,
                )
            else:
                snap, max_used_ns = _compute_station_snapshot(
                    station_series[sid],
                    cutoff_ns=cutoff_ns,
                    cutoff_minutes=cutoff_minutes,
                    include_regime=cfg.enable_v2_regime_features,
                )
            short = _station_short(sid)
            for k, v in snap.items():
                row[f"{short}_{k}"] = v
            station_max_used.append(max_used_ns)

        for sid in cfg.neighbor_station_ids:
            short = _station_short(sid)
            row[f"temp_diff_{short}"] = _safe_diff(
                float(row.get(f"{short}_temp_now", np.nan)),
                float(row.get("temp_now", np.nan)),
            )
            row[f"dewpt_diff_{short}"] = _safe_diff(
                float(row.get(f"{short}_dewpt_now", np.nan)),
                float(row.get("dewpt_now", np.nan)),
            )
            row[f"pressure_diff_{short}"] = _safe_diff(
                float(row.get(f"{short}_pressure_now", np.nan)),
                float(row.get("pressure_now", np.nan)),
            )
            row[f"wspd_diff_{short}"] = _safe_diff(
                float(row.get(f"{short}_wspd_now", np.nan)),
                float(row.get("wspd_now", np.nan)),
            )
            row[f"dewpoint_depression_diff_{short}"] = _safe_diff(
                float(row.get(f"{short}_dewpoint_depression_now", np.nan)),
                float(row.get("dewpoint_depression_now", np.nan)),
            )

            if cfg.enable_neighbor_regime_features:
                row[f"vis_diff_{short}"] = _safe_diff(
                    float(row.get(f"{short}_vis_now", np.nan)),
                    float(row.get("vis_now", np.nan)),
                )
                row[f"clds_oktas_diff_{short}"] = _safe_diff(
                    float(row.get(f"{short}_clds_oktas_now", np.nan)),
                    float(row.get("clds_oktas_now", np.nan)),
                )
                row[f"uv_diff_{short}"] = _safe_diff(
                    float(row.get(f"{short}_uv_index_now", np.nan)),
                    float(row.get("uv_index_now", np.nan)),
                )

                for w in cfg.neighbor_gradient_windows_minutes:
                    row[f"temp_diff_{short}_prev_{w}"] = _safe_diff(
                        float(row.get(f"{short}_temp_prev_{w}", np.nan)),
                        float(row.get(f"temp_prev_{w}", np.nan)),
                    )
                    row[f"temp_diff_{short}_delta_{w}"] = _safe_diff(
                        float(row.get(f"temp_diff_{short}", np.nan)),
                        float(row.get(f"temp_diff_{short}_prev_{w}", np.nan)),
                    )
                    row[f"pressure_diff_{short}_prev_{w}"] = _safe_diff(
                        float(row.get(f"{short}_pressure_prev_{w}", np.nan)),
                        float(row.get(f"pressure_prev_{w}", np.nan)),
                    )
                    row[f"pressure_diff_{short}_delta_{w}"] = _safe_diff(
                        float(row.get(f"pressure_diff_{short}", np.nan)),
                        float(row.get(f"pressure_diff_{short}_prev_{w}", np.nan)),
                    )
                    row[f"vis_diff_{short}_prev_{w}"] = _safe_diff(
                        float(row.get(f"{short}_vis_prev_{w}", np.nan)),
                        float(row.get(f"vis_prev_{w}", np.nan)),
                    )
                    row[f"vis_diff_{short}_delta_{w}"] = _safe_diff(
                        float(row.get(f"vis_diff_{short}", np.nan)),
                        float(row.get(f"vis_diff_{short}_prev_{w}", np.nan)),
                    )
                    row[f"clds_oktas_diff_{short}_prev_{w}"] = _safe_diff(
                        float(row.get(f"{short}_clds_oktas_prev_{w}", np.nan)),
                        float(row.get(f"clds_oktas_prev_{w}", np.nan)),
                    )
                    row[f"clds_oktas_diff_{short}_delta_{w}"] = _safe_diff(
                        float(row.get(f"clds_oktas_diff_{short}", np.nan)),
                        float(row.get(f"clds_oktas_diff_{short}_prev_{w}", np.nan)),
                    )
                    row[f"uv_diff_{short}_prev_{w}"] = _safe_diff(
                        float(row.get(f"{short}_uv_prev_{w}", np.nan)),
                        float(row.get(f"uv_prev_{w}", np.nan)),
                    )
                    row[f"uv_diff_{short}_delta_{w}"] = _safe_diff(
                        float(row.get(f"uv_diff_{short}", np.nan)),
                        float(row.get(f"uv_diff_{short}_prev_{w}", np.nan)),
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

        stations_to_check = [cfg.target_station_id] + list(cfg.neighbor_station_ids if cfg.enable_neighbor_regime_features else [])
        for sid in stations_to_check:
            prefix = "" if sid == cfg.target_station_id else f"{_station_short(sid)}_"
            clds_now = str(row.get(f"{prefix}clds_norm_now", "UNK"))
            if clds_now not in ALLOWED_CLDS_NORM:
                raise AssertionError(
                    f"CLDS vocabulary guard failed station={sid} date={target_date} cutoff={cutoff_minutes} value={clds_now}"
                )
            uv_now = float(row.get(f"{prefix}uv_index_now", np.nan))
            uv_missing = float(row.get(f"{prefix}uv_missing_now", 1.0))
            if np.isfinite(uv_missing) and uv_missing < 0.5:
                if (not np.isfinite(uv_now)) or uv_now < 0.0 or uv_now > 20.0:
                    raise AssertionError(
                        f"UV sanity guard failed station={sid} date={target_date} cutoff={cutoff_minutes} value={uv_now}"
                    )

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


def model_feature_columns(df: pd.DataFrame, cfg: PipelineConfig | None = None) -> list[str]:
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
    keep_merge = bool(cfg.keep_merge_index_features) if cfg is not None else False
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if (not keep_merge) and c in {"index", "index_x", "index_y"}:
            continue
        if cfg is not None and (not cfg.include_feels_like) and c.startswith("feels_like"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols
