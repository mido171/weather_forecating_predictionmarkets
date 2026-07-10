"""Feature engineering from Info.gov hourly readings."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from hkg_tmax.features.feature_registry import FeatureRegistry
from hkg_tmax.features.station_groups import CORE_STATION_DELTAS, STATION_GROUPS, station_feature_name
from hkg_tmax.features.text_regime_flags import hourly_text_flags

HKO_LAT = 22.301944
HKO_LON = 114.174167


HOURLY_SQL = """
SELECT
  bulletin_id,
  source_url,
  index_date_hkt,
  title,
  dispatch_at_hkt,
  dispatch_at_utc,
  observation_at_hkt,
  observation_at_utc,
  available_at_utc,
  hko_air_temp_c,
  hko_relative_humidity_pct,
  rainfall_text,
  warning_text,
  lightning_text,
  tropical_cyclone_text,
  tropical_cyclone_name,
  tropical_cyclone_lat,
  tropical_cyclone_lon,
  station_readings_jsonb,
  station_count,
  station_missing_count,
  station_temp_min_c,
  station_temp_max_c,
  station_temp_mean_c,
  station_temp_spread_c,
  target_station_present,
  parse_status,
  raw_sha256
FROM public.hko_info_gov_hourly_readings_1998_2026
WHERE parse_status IN ('parsed', 'partial')
  AND dispatch_at_utc >= %(hourly_start_utc)s
  AND dispatch_at_utc <= %(hourly_end_utc)s
ORDER BY dispatch_at_utc, source_url
"""


def load_hourly_readings(connection: Any, params: dict[str, Any]) -> pd.DataFrame:
    with connection.cursor() as cursor:
        cursor.execute(HOURLY_SQL, params)
        rows = cursor.fetchall()
        columns = [desc.name for desc in cursor.description]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    for col in ("dispatch_at_utc", "observation_at_utc", "available_at_utc"):
        frame[col] = pd.to_datetime(frame[col], errors="coerce", utc=True)
    for col in ("dispatch_at_hkt", "observation_at_hkt"):
        frame[col] = pd.to_datetime(frame[col], errors="coerce")
    numeric_cols = [
        "hko_air_temp_c",
        "hko_relative_humidity_pct",
        "tropical_cyclone_lat",
        "tropical_cyclone_lon",
        "station_count",
        "station_missing_count",
        "station_temp_min_c",
        "station_temp_max_c",
        "station_temp_mean_c",
        "station_temp_spread_c",
    ]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["station_objects"] = frame["station_readings_jsonb"].map(_parse_station_json)
    station_summaries = frame.apply(_station_window_summary, axis=1, result_type="expand")
    frame = pd.concat([frame, station_summaries], axis=1)
    frame["dispatch_ns"] = frame["dispatch_at_utc"].astype("int64")
    frame = frame.sort_values(["dispatch_at_utc", "source_url"]).reset_index(drop=True)
    return frame


def _parse_station_json(value: object) -> list[dict[str, Any]]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [dict(item) for item in parsed if isinstance(item, dict)]
    return []


def dewpoint_c(temp_c: float | None, rh_pct: float | None) -> float:
    if temp_c is None or rh_pct is None or pd.isna(temp_c) or pd.isna(rh_pct):
        return math.nan
    if rh_pct <= 0 or rh_pct > 100:
        return math.nan
    a = 17.625
    b = 243.04
    alpha = math.log(float(rh_pct) / 100.0) + a * float(temp_c) / (b + float(temp_c))
    return float(b * alpha / (a - alpha))


def _value_at_or_before(window: pd.DataFrame, cutoff: pd.Timestamp, lookback_hours: float, column: str) -> float:
    target = cutoff - pd.Timedelta(hours=lookback_hours)
    eligible = window[window["dispatch_at_utc"].le(target)].dropna(subset=[column])
    if eligible.empty:
        return math.nan
    row = eligible.iloc[-1]
    age_hours = (target - row["dispatch_at_utc"]).total_seconds() / 3600.0
    if age_hours > lookback_hours + 0.75:
        return math.nan
    return float(row[column])


def _safe_mean(values: list[float]) -> float:
    clean = [float(v) for v in values if v is not None and not pd.isna(v)]
    return float(np.mean(clean)) if clean else math.nan


def _safe_max(values: list[float]) -> float:
    clean = [float(v) for v in values if v is not None and not pd.isna(v)]
    return float(np.max(clean)) if clean else math.nan


def _safe_min(values: list[float]) -> float:
    clean = [float(v) for v in values if v is not None and not pd.isna(v)]
    return float(np.min(clean)) if clean else math.nan


def _safe_median(values: list[float]) -> float:
    clean = [float(v) for v in values if v is not None and not pd.isna(v)]
    return float(np.median(clean)) if clean else math.nan


def clean_station_values(row: pd.Series) -> tuple[dict[str, float], dict[str, int], int]:
    hko_temp = row.get("hko_air_temp_c")
    values: dict[str, float] = {}
    flags: dict[str, int] = {}
    outliers = 0
    for item in row.get("station_objects") or []:
        name = str(item.get("station_canonical_name") or "").strip().upper()
        if not name:
            continue
        temp = item.get("temperature_c")
        missing = bool(item.get("temperature_missing"))
        outlier_flag = 0
        if missing or temp is None or pd.isna(temp):
            flags[f"{name}|missing"] = 1
            continue
        value = float(temp)
        if value < 0.0 or value > 42.0:
            outlier_flag = 1
        if pd.notna(hko_temp) and abs(value - float(hko_temp)) > 12.0:
            outlier_flag = 1
        if outlier_flag:
            outliers += 1
            flags[f"{name}|outlier"] = 1
            continue
        values[name] = value
        flags[f"{name}|missing"] = 0
        flags[f"{name}|outlier"] = 0
    return values, flags, outliers


def _station_window_summary(row: pd.Series) -> dict[str, float]:
    values, _, _ = clean_station_values(row)
    station_values = list(values.values())
    if station_values:
        spread = _safe_max(station_values) - _safe_min(station_values)
        mean = _safe_mean(station_values)
    else:
        spread = math.nan
        mean = math.nan
    group_map = {group.name: _group_stats(values, group.stations) for group in STATION_GROUPS}
    nt_hotspot = _safe_max([group_map["inland_nt"]["max"], group_map["west_nw_nt"]["max"]])
    return {
        "pre_station_network_spread_c": spread,
        "pre_station_network_mean_c": mean,
        "pre_station_inland_minus_coastal_c": _diff(
            group_map["inland_nt"]["mean"], group_map["coastal_marine"]["mean"]
        ),
        "pre_station_nt_hotspot_c": nt_hotspot,
    }


def _group_stats(values: dict[str, float], stations: tuple[str, ...]) -> dict[str, float]:
    group_values = [values[station] for station in stations if station in values]
    return {
        "mean": _safe_mean(group_values),
        "median": _safe_median(group_values),
        "max": _safe_max(group_values),
        "min": _safe_min(group_values),
        "count": float(len(group_values)),
        "missing_fraction": float(1.0 - len(group_values) / max(1, len(stations))),
    }


def _percent_rank(value: float, values: list[float]) -> float:
    clean = sorted(v for v in values if pd.notna(v))
    if not clean or pd.isna(value):
        return math.nan
    return float(sum(v <= value for v in clean) / len(clean))


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return float(2 * radius_km * math.asin(math.sqrt(a)))


def _bearing_rad(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return math.atan2(y, x)


def build_hourly_features(
    rows: pd.DataFrame,
    hourly: pd.DataFrame,
    registry: FeatureRegistry,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if hourly.empty:
        empty = rows[["target_date", "cutoff_profile"]].copy()
        return rows.merge(empty, on=["target_date", "cutoff_profile"]), {}
    dispatch_ns = hourly["dispatch_ns"].to_numpy(dtype=np.int64)
    records: list[dict[str, Any]] = []
    station_outliers: list[dict[str, Any]] = []
    latest_age_rows: list[dict[str, Any]] = []
    for row in rows.itertuples(index=False):
        cutoff_utc = pd.Timestamp(row.cutoff_at_utc)
        target_date = pd.Timestamp(row.target_date).normalize()
        start_utc = cutoff_utc - pd.Timedelta(hours=36)
        start_idx = int(np.searchsorted(dispatch_ns, start_utc.value, side="left"))
        end_idx = int(np.searchsorted(dispatch_ns, cutoff_utc.value, side="right"))
        window = hourly.iloc[start_idx:end_idx].copy()
        if not window.empty:
            window = window[
                window["available_at_utc"].le(cutoff_utc)
                & (window["observation_at_utc"].isna() | window["observation_at_utc"].le(cutoff_utc))
                & (window["observation_at_hkt"].isna() | (window["observation_at_hkt"].dt.normalize() < target_date))
            ].copy()
        feature: dict[str, Any] = {
            "target_date": target_date,
            "cutoff_profile": row.cutoff_profile,
            "latest_hourly_dispatch_at_hkt_used": pd.NaT,
            "latest_hourly_observation_at_hkt_used": pd.NaT,
            "latest_hourly_dispatch_at_utc_used": pd.NaT,
            "latest_hourly_observation_at_utc_used": pd.NaT,
            "max_hourly_dispatch_at_utc_used": pd.NaT,
            "max_hourly_observation_at_utc_used": pd.NaT,
            "hourly_latest_missing_flag": 1,
        }
        if window.empty:
            records.append(feature)
            continue
        target_rows = window[window["hko_air_temp_c"].notna()].copy()
        latest = target_rows.iloc[-1] if not target_rows.empty else window.iloc[-1]
        feature.update(_latest_hko_features(latest, cutoff_utc, row))
        feature.update(_hko_window_features(window, latest, cutoff_utc, target_date))
        network_features, outlier_records = _network_features(latest, window, row)
        feature.update(network_features)
        for outlier_record in outlier_records:
            outlier_record["target_date"] = str(target_date.date())
            outlier_record["cutoff_profile"] = row.cutoff_profile
            station_outliers.append(outlier_record)
        feature.update(_hourly_text_context_features(window, cutoff_utc))
        feature.update(_tropical_cyclone_features(window))
        feature["latest_hourly_dispatch_at_hkt_used"] = latest.get("dispatch_at_hkt")
        feature["latest_hourly_observation_at_hkt_used"] = latest.get("observation_at_hkt")
        feature["latest_hourly_dispatch_at_utc_used"] = latest.get("dispatch_at_utc")
        feature["latest_hourly_observation_at_utc_used"] = latest.get("observation_at_utc")
        feature["max_hourly_dispatch_at_utc_used"] = window["dispatch_at_utc"].max()
        feature["max_hourly_observation_at_utc_used"] = window["observation_at_utc"].max()
        latest_age_rows.append(
            {
                "target_date": target_date.date().isoformat(),
                "cutoff_profile": row.cutoff_profile,
                "latest_age_minutes": feature.get("hko_latest_age_minutes"),
            }
        )
        records.append(feature)
    features = pd.DataFrame(records)
    out = rows.merge(features, on=["target_date", "cutoff_profile"], how="left", validate="one_to_one")
    hko_cols = [col for col in features.columns if col.startswith("hko_") or col.startswith("hourly_latest")]
    network_cols = [
        col
        for col in features.columns
        if col.startswith("network_")
        or col.startswith("urban_")
        or col.startswith("coastal_")
        or col.startswith("inland_")
        or col.startswith("west_")
        or col.startswith("station_")
        or col.endswith("_index_c")
        or col.endswith("_alignment_c")
    ]
    text_cols = [
        col
        for col in features.columns
        if col.startswith("hourly_any_")
        or col.startswith("hourly_")
        or col.startswith("tc_")
        or col.startswith("hours_since_")
    ]
    registry.add(
        [col for col in hko_cols if col in out.columns],
        family="hko_hourly_state",
        source_table="public.hko_info_gov_hourly_readings_1998_2026",
        source_time_column="dispatch_at_utc",
        eligibility_rule="dispatch_at_utc/available_at_utc/observation_at_utc <= cutoff_at_utc",
    )
    registry.add(
        [col for col in network_cols if col in out.columns],
        family="station_network",
        source_table="public.hko_info_gov_hourly_readings_1998_2026",
        source_time_column="dispatch_at_utc",
        eligibility_rule="latest and trailing-window station JSON from eligible hourly readings only",
    )
    registry.add(
        [col for col in text_cols if col in out.columns],
        family="text_warning_regime",
        source_table="public.hko_info_gov_hourly_readings_1998_2026",
        source_time_column="dispatch_at_utc",
        eligibility_rule="warning/rain/lightning/tropical-cyclone text from eligible hourly readings only",
    )
    reports = {
        "station_outlier_report": pd.DataFrame(station_outliers),
        "hourly_latest_age_distribution": pd.DataFrame(latest_age_rows),
        "station_missingness_by_year": _station_missingness_by_year(hourly),
        "text_flag_prevalence_by_year": _text_flag_prevalence_by_year(hourly),
    }
    return out, reports


def _latest_hko_features(latest: pd.Series, cutoff_utc: pd.Timestamp, row: Any) -> dict[str, Any]:
    temp = latest.get("hko_air_temp_c")
    rh = latest.get("hko_relative_humidity_pct")
    dewpoint = dewpoint_c(float(temp) if pd.notna(temp) else None, float(rh) if pd.notna(rh) else None)
    age_minutes = (cutoff_utc - latest["dispatch_at_utc"]).total_seconds() / 60.0
    return {
        "hko_latest_temp_c": temp,
        "hko_latest_rh_pct": rh,
        "hko_latest_dewpoint_c": dewpoint,
        "hko_latest_dewpoint_depression_c": float(temp) - dewpoint if pd.notna(temp) and pd.notna(dewpoint) else math.nan,
        "hko_latest_temp_minus_official_max_c": float(temp) - float(row.official_max_c) if pd.notna(temp) and pd.notna(row.official_max_c) else math.nan,
        "hko_latest_temp_minus_official_min_c": float(temp) - float(row.official_min_c) if pd.notna(temp) and pd.notna(row.official_min_c) else math.nan,
        "hko_latest_age_minutes": float(age_minutes),
        "hko_latest_age_gt_90min_flag": int(age_minutes > 90.0),
        "hko_latest_observation_hour_hkt": int(latest["observation_at_hkt"].hour) if pd.notna(latest.get("observation_at_hkt")) else math.nan,
        "hko_target_station_present_latest_flag": int(bool(latest.get("target_station_present"))),
        "hko_target_station_missing_flag": int(not bool(latest.get("target_station_present"))),
        "hko_rh_missing_flag": int(pd.isna(rh)),
        "hko_dewpoint_missing_flag": int(pd.isna(dewpoint)),
        "hourly_latest_missing_flag": 0,
    }


def _hko_window_features(
    window: pd.DataFrame,
    latest: pd.Series,
    cutoff_utc: pd.Timestamp,
    target_date: pd.Timestamp,
) -> dict[str, Any]:
    temp = latest.get("hko_air_temp_c")
    rh = latest.get("hko_relative_humidity_pct")
    features: dict[str, Any] = {}
    for hours in (1, 3, 6, 12):
        old_temp = _value_at_or_before(window, cutoff_utc, hours, "hko_air_temp_c")
        features[f"hko_temp_trend_{hours}h_c"] = float(temp) - old_temp if pd.notna(temp) and pd.notna(old_temp) else math.nan
        features[f"hko_trend_{hours}h_missing_flag"] = int(pd.isna(features[f"hko_temp_trend_{hours}h_c"]))
    for hours in (3, 6):
        old_rh = _value_at_or_before(window, cutoff_utc, hours, "hko_relative_humidity_pct")
        features[f"hko_rh_trend_{hours}h_pct"] = float(rh) - old_rh if pd.notna(rh) and pd.notna(old_rh) else math.nan
    for hours in (6, 12, 24):
        sub = window[window["dispatch_at_utc"].ge(cutoff_utc - pd.Timedelta(hours=hours))]
        features[f"hko_temp_mean_{hours}h_c"] = _safe_mean(sub["hko_air_temp_c"].dropna().tolist())
        if hours == 24:
            features["hko_temp_max_24h_c"] = _safe_max(sub["hko_air_temp_c"].dropna().tolist())
            features["hko_temp_min_24h_c"] = _safe_min(sub["hko_air_temp_c"].dropna().tolist())
            features["hko_temp_range_24h_c"] = (
                features["hko_temp_max_24h_c"] - features["hko_temp_min_24h_c"]
                if pd.notna(features["hko_temp_max_24h_c"]) and pd.notna(features["hko_temp_min_24h_c"])
                else math.nan
            )
        features[f"hko_rh_mean_{hours}h_pct"] = _safe_mean(sub["hko_relative_humidity_pct"].dropna().tolist())
    features["hko_dispatch_count_24h"] = int((window["dispatch_at_utc"] >= cutoff_utc - pd.Timedelta(hours=24)).sum())
    features["hko_partial_parse_count_24h"] = int(
        window[window["dispatch_at_utc"].ge(cutoff_utc - pd.Timedelta(hours=24))]["parse_status"].eq("partial").sum()
    )
    d = target_date - pd.Timedelta(days=1)
    day_rows = window[window["observation_at_hkt"].dt.normalize().eq(d)]
    features["hko_pre_target_overnight_min_c"] = _safe_min(
        day_rows[day_rows["observation_at_hkt"].dt.hour.between(0, 6)]["hko_air_temp_c"].dropna().tolist()
    )
    features["hko_pre_target_morning_mean_c"] = _safe_mean(
        day_rows[day_rows["observation_at_hkt"].dt.hour.between(7, 11)]["hko_air_temp_c"].dropna().tolist()
    )
    features["hko_pre_target_afternoon_max_sofar_c"] = _safe_max(
        day_rows[day_rows["observation_at_hkt"].dt.hour.ge(12)]["hko_air_temp_c"].dropna().tolist()
    )
    evening = day_rows[day_rows["observation_at_hkt"].dt.hour.ge(18)].dropna(subset=["hko_air_temp_c"])
    features["hko_pre_target_evening_temp_c"] = float(evening["hko_air_temp_c"].iloc[-1]) if not evening.empty else math.nan
    features["hko_evening_cooling_18_to_cutoff_c"] = (
        float(evening["hko_air_temp_c"].iloc[-1]) - float(evening["hko_air_temp_c"].iloc[0])
        if len(evening) >= 2
        else math.nan
    )
    t09 = day_rows[day_rows["observation_at_hkt"].dt.hour.le(9)]["hko_air_temp_c"].dropna()
    t15 = day_rows[day_rows["observation_at_hkt"].dt.hour.le(15)]["hko_air_temp_c"].dropna()
    features["hko_afternoon_warmup_09_to_15_c"] = (
        float(t15.iloc[-1]) - float(t09.iloc[-1]) if not t09.empty and not t15.empty else math.nan
    )
    for name, value in list(features.items()):
        if name.startswith("hko_pre_target") or name.startswith("hko_evening") or name.startswith("hko_afternoon"):
            features[f"{name}_missing_flag"] = int(pd.isna(value))
    return features


def _network_features(latest: pd.Series, window: pd.DataFrame, row: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    values, flags, outliers = clean_station_values(latest)
    hko_temp = latest.get("hko_air_temp_c")
    station_values = list(values.values())
    all_values = ([float(hko_temp)] if pd.notna(hko_temp) else []) + station_values
    feature: dict[str, Any] = {
        "network_latest_station_count": latest.get("station_count"),
        "network_latest_missing_count": latest.get("station_missing_count"),
        "network_latest_valid_count": len(station_values),
        "network_latest_temp_mean_c": _safe_mean(station_values),
        "network_latest_temp_min_c": _safe_min(station_values),
        "network_latest_temp_max_c": _safe_max(station_values),
        "network_latest_temp_spread_c": _safe_max(station_values) - _safe_min(station_values) if station_values else math.nan,
        "network_latest_hko_percentile": _percent_rank(float(hko_temp), all_values) if pd.notna(hko_temp) else math.nan,
        "network_latest_max_minus_hko_c": _safe_max(station_values) - float(hko_temp) if station_values and pd.notna(hko_temp) else math.nan,
        "network_latest_hko_minus_mean_c": float(hko_temp) - _safe_mean(station_values) if station_values and pd.notna(hko_temp) else math.nan,
        "network_latest_valid_fraction": float(len(station_values) / max(1.0, float(latest.get("station_count") or 1))),
        "network_outlier_any_flag_24h": int(outliers > 0),
        "station_json_empty_latest_flag": int(len(values) == 0),
        "station_missing_count_latest": latest.get("station_missing_count"),
    }
    outlier_records = [
        {
            "dispatch_at_hkt": latest.get("dispatch_at_hkt"),
            "outlier_count_latest": outliers,
        }
    ] if outliers else []
    group_stats: dict[str, dict[str, float]] = {}
    for group in STATION_GROUPS:
        stats = _group_stats(values, group.stations)
        group_stats[group.name] = stats
        for key in ("mean", "max", "count", "missing_fraction"):
            feature[f"{group.name}_latest_{key}_c" if key in {"mean", "max"} else f"{group.name}_latest_{key}"] = stats[key]
        feature[f"{group.name}_latest_mean_minus_hko_c"] = (
            stats["mean"] - float(hko_temp) if pd.notna(stats["mean"]) and pd.notna(hko_temp) else math.nan
        )
        feature[f"{group.name}_latest_max_minus_hko_c"] = (
            stats["max"] - float(hko_temp) if pd.notna(stats["max"]) and pd.notna(hko_temp) else math.nan
        )
        feature[f"station_group_{group.name}_insufficient_count_flag"] = int(stats["count"] < 2)
    feature["inland_nt_mean_minus_coastal_marine_mean_c"] = _diff(
        group_stats["inland_nt"]["mean"], group_stats["coastal_marine"]["mean"]
    )
    feature["inland_nt_max_minus_urban_core_mean_c"] = _diff(
        group_stats["inland_nt"]["max"], group_stats["urban_core"]["mean"]
    )
    feature["west_nw_nt_mean_minus_coastal_marine_mean_c"] = _diff(
        group_stats["west_nw_nt"]["mean"], group_stats["coastal_marine"]["mean"]
    )
    feature["urban_core_mean_minus_coastal_marine_mean_c"] = _diff(
        group_stats["urban_core"]["mean"], group_stats["coastal_marine"]["mean"]
    )
    feature["urban_core_mean_minus_hko_c"] = _diff(group_stats["urban_core"]["mean"], hko_temp)
    feature["coastal_marine_mean_minus_hko_c"] = _diff(group_stats["coastal_marine"]["mean"], hko_temp)
    nt_hotspot = _safe_max([group_stats["inland_nt"]["max"], group_stats["west_nw_nt"]["max"]])
    feature["nt_hotspot_max_minus_official_max_c"] = _diff(nt_hotspot, row.official_max_c)
    feature["maritime_suppression_index_c"] = _diff(hko_temp, group_stats["coastal_marine"]["mean"])
    feature["urban_heat_index_c"] = _diff(group_stats["urban_core"]["mean"], group_stats["coastal_marine"]["mean"])
    feature["inland_heat_index_c"] = _diff(group_stats["inland_nt"]["mean"], group_stats["coastal_marine"]["mean"])
    feature["nt_heat_ceiling_index_c"] = _diff(nt_hotspot, row.official_max_c)
    feature["hko_station_rank_pct"] = feature["network_latest_hko_percentile"]
    feature["coastal_hko_alignment_c"] = abs(_diff(hko_temp, group_stats["coastal_marine"]["mean"]))
    feature["urban_hko_alignment_c"] = abs(_diff(hko_temp, group_stats["urban_core"]["mean"]))
    for station in CORE_STATION_DELTAS:
        temp = values.get(station)
        feature[station_feature_name(station, "latest_temp_c")] = temp
        feature[station_feature_name(station, "latest_minus_hko_c")] = _diff(temp, hko_temp)
        feature[station_feature_name(station, "missing_flag")] = int(station not in values)
        feature[station_feature_name(station, "outlier_flag")] = flags.get(f"{station}|outlier", 0)
    feature.update(_window_network_features(window, row))
    feature["network_spread_x_thunderstorm_flag"] = (
        feature["network_latest_temp_spread_c"] * _flag(row, "fcst_flag_thunderstorm")
        if pd.notna(feature["network_latest_temp_spread_c"])
        else math.nan
    )
    feature["inland_heat_index_x_warm_season"] = (
        feature["inland_heat_index_c"] * int(row.warm_season_flag)
        if pd.notna(feature["inland_heat_index_c"])
        else math.nan
    )
    feature["maritime_suppression_x_southerly_forecast_flag"] = (
        feature["maritime_suppression_index_c"] * _flag(row, "fcst_flag_southerly")
        if pd.notna(feature["maritime_suppression_index_c"])
        else math.nan
    )
    feature["official_range_x_thunderstorm_or_rainstorm_flag"] = (
        float(row.official_range_c) * _flag(row, "fcst_flag_thunderstorm")
        if pd.notna(getattr(row, "official_range_c", math.nan))
        else math.nan
    )
    feature["nt_heat_ceiling_x_forecast_nt_higher_text_flag"] = (
        feature["nt_heat_ceiling_index_c"] * _flag(row, "fcst_flag_nt_higher")
        if pd.notna(feature["nt_heat_ceiling_index_c"])
        else math.nan
    )
    return feature, outlier_records


def _diff(left: object, right: object) -> float:
    if left is None or right is None or pd.isna(left) or pd.isna(right):
        return math.nan
    return float(left) - float(right)


def _flag(row: Any, name: str) -> int:
    value = getattr(row, name, 0)
    if value is None or pd.isna(value):
        return 0
    return int(value)


def _window_network_features(window: pd.DataFrame, row: Any) -> dict[str, Any]:
    spreads = pd.to_numeric(window["pre_station_network_spread_c"], errors="coerce").dropna().tolist()
    means = pd.to_numeric(window["pre_station_network_mean_c"], errors="coerce").dropna().tolist()
    inland_minus_coastal = pd.to_numeric(
        window["pre_station_inland_minus_coastal_c"], errors="coerce"
    ).dropna().tolist()
    nt_hotspots = pd.to_numeric(window["pre_station_nt_hotspot_c"], errors="coerce").dropna().tolist()
    missing_counts = pd.to_numeric(window["station_missing_count"], errors="coerce").dropna().tolist()
    last = window.iloc[-1]
    start_6h = last["dispatch_at_utc"] - pd.Timedelta(hours=6)
    window_6h = window[window["dispatch_at_utc"].ge(start_6h)]
    spread_6h = pd.to_numeric(window_6h["pre_station_network_spread_c"], errors="coerce").dropna().tolist()
    latest_mean = means[-1] if means else math.nan
    first_mean = means[0] if means else math.nan
    latest_inland = inland_minus_coastal[-1] if inland_minus_coastal else math.nan
    first_inland = inland_minus_coastal[0] if inland_minus_coastal else math.nan
    return {
        "network_spread_mean_6h_c": _safe_mean(spread_6h),
        "network_spread_max_6h_c": _safe_max(spread_6h),
        "network_spread_max_24h_c": _safe_max(spreads),
        "network_mean_trend_6h_c": _diff(latest_mean, first_mean),
        "inland_minus_coastal_trend_6h_c": _diff(latest_inland, first_inland),
        "nt_hotspot_max_24h_c": _safe_max(nt_hotspots),
        "nt_hotspot_max_24h_minus_official_max_c": _diff(_safe_max(nt_hotspots), row.official_max_c),
        "station_missing_count_mean_24h": _safe_mean(missing_counts),
        "station_missing_count_max_24h": _safe_max(missing_counts),
        "inland_coastal_spread_6h_max_c": _safe_max([v for v in inland_minus_coastal if pd.notna(v)]),
        "network_spatial_heterogeneity_flag": int(_safe_max(spreads) >= 5.0) if spreads else 0,
    }


def _hourly_text_context_features(window: pd.DataFrame, cutoff_utc: pd.Timestamp) -> dict[str, Any]:
    feature: dict[str, Any] = {}
    for hours in (6, 24):
        sub = window[window["dispatch_at_utc"].ge(cutoff_utc - pd.Timedelta(hours=hours))]
        text_parts: list[object] = []
        for col in ("warning_text", "rainfall_text", "lightning_text", "tropical_cyclone_text"):
            text_parts.extend(sub[col].dropna().tolist())
        feature.update(hourly_text_flags(text_parts, f"{hours}h"))
        feature[f"hourly_warning_text_count_{hours}h"] = int(sub["warning_text"].notna().sum())
        feature[f"hourly_lightning_text_count_{hours}h"] = int(sub["lightning_text"].notna().sum())
        feature[f"hourly_rainfall_text_count_{hours}h"] = int(sub["rainfall_text"].notna().sum())
    for col, name in (
        ("warning_text", "warning_text"),
        ("lightning_text", "lightning_text"),
        ("rainfall_text", "rainfall_text"),
        ("tropical_cyclone_text", "tropical_cyclone_text"),
    ):
        present = window[window[col].notna()]
        feature[f"hours_since_latest_{name}"] = (
            float((cutoff_utc - present["dispatch_at_utc"].iloc[-1]).total_seconds() / 3600.0)
            if not present.empty
            else math.nan
        )
    feature["text_warning_block_missing_flag"] = int(window["warning_text"].notna().sum() == 0)
    return feature


def _tropical_cyclone_features(window: pd.DataFrame) -> dict[str, Any]:
    tc = window[window["tropical_cyclone_text"].notna()].copy()
    if tc.empty:
        return {
            "tc_name_present_latest_flag": 0,
            "tc_position_present_latest_flag": 0,
            "tc_lat_latest": math.nan,
            "tc_lon_latest": math.nan,
            "tc_distance_to_hko_km_latest": math.nan,
            "tc_bearing_sin_latest": math.nan,
            "tc_bearing_cos_latest": math.nan,
            "tc_distance_missing_flag": 1,
            "cyclone_position_missing_flag": 1,
        }
    latest = tc.iloc[-1]
    lat = latest.get("tropical_cyclone_lat")
    lon = latest.get("tropical_cyclone_lon")
    present = pd.notna(lat) and pd.notna(lon)
    bearing = _bearing_rad(HKO_LAT, HKO_LON, float(lat), float(lon)) if present else math.nan
    return {
        "tc_name_present_latest_flag": int(pd.notna(latest.get("tropical_cyclone_name"))),
        "tc_position_present_latest_flag": int(present),
        "tc_lat_latest": lat,
        "tc_lon_latest": lon,
        "tc_distance_to_hko_km_latest": _haversine_km(HKO_LAT, HKO_LON, float(lat), float(lon)) if present else math.nan,
        "tc_bearing_sin_latest": math.sin(bearing) if present else math.nan,
        "tc_bearing_cos_latest": math.cos(bearing) if present else math.nan,
        "tc_distance_missing_flag": int(not present),
        "cyclone_position_missing_flag": int(not present),
    }


def _station_missingness_by_year(hourly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year, group in hourly.groupby(hourly["dispatch_at_hkt"].dt.year):
        total = len(group)
        rows.append(
            {
                "year": int(year),
                "dispatch_rows": int(total),
                "station_missing_count_mean": float(group["station_missing_count"].mean()),
                "station_json_empty_rows": int(group["station_count"].fillna(0).eq(0).sum()),
                "target_station_missing_rows": int((~group["target_station_present"].fillna(False).astype(bool)).sum()),
            }
        )
    return pd.DataFrame(rows)


def _text_flag_prevalence_by_year(hourly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year, group in hourly.groupby(hourly["dispatch_at_hkt"].dt.year):
        rows.append(
            {
                "year": int(year),
                "rows": int(len(group)),
                "warning_text_rows": int(group["warning_text"].notna().sum()),
                "rainfall_text_rows": int(group["rainfall_text"].notna().sum()),
                "lightning_text_rows": int(group["lightning_text"].notna().sum()),
                "tropical_cyclone_text_rows": int(group["tropical_cyclone_text"].notna().sum()),
            }
        )
    return pd.DataFrame(rows)
