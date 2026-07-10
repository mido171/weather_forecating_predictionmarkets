from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests


logger = logging.getLogger("ml_live.gribstream")
_MEMBER_KEY_PATTERN = re.compile(r"(?i)tmpk.*?(?:member|mem|ens)?[_-]?(\\d{1,3})$")


def resolve_station_coordinates(station_id: str, base_url: str = "https://forecast.weather.gov") -> tuple[float, float]:
    url = f"{base_url}/xml/current_obs/{station_id.upper()}.xml"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    lat = root.findtext("latitude")
    lon = root.findtext("longitude")
    if lat is None or lon is None:
        raise ValueError(f"Missing latitude/longitude in NWS XML for {station_id}")
    return float(lat), float(lon)


def _build_auth_header(token: str, auth_scheme: str | None) -> str:
    cleaned = token.strip()
    if not cleaned:
        raise ValueError("GribStream token is empty")
    if " " in cleaned:
        return cleaned
    scheme = (auth_scheme or "Bearer").strip() or "Bearer"
    return f"{scheme} {cleaned}"


def _summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "forecastedFrom": payload.get("forecastedFrom"),
        "forecastedUntil": payload.get("forecastedUntil"),
        "fromTime": payload.get("fromTime"),
        "untilTime": payload.get("untilTime"),
        "asOf": payload.get("asOf"),
        "minHorizon": payload.get("minHorizon"),
        "maxHorizon": payload.get("maxHorizon"),
        "variables": len(payload.get("variables", []) or []),
        "coordinates": len(payload.get("coordinates", []) or []),
        "members": len(payload.get("members", []) or []),
    }


def _limit_snippet(text: str, limit: int = 200) -> str:
    if not text:
        return ""
    trimmed = text.replace("\n", "\\n")
    if len(trimmed) <= limit:
        return trimmed
    return trimmed[:limit] + "...(truncated)"


def fetch_history_payload(
    base_url: str,
    model: str,
    token: str,
    payload: dict[str, Any],
    accept: str,
    auth_scheme: str | None = None,
    timeout: int = 60,
) -> tuple[str, str]:
    if "asOf" not in payload:
        raise ValueError("GribStream payload must include asOf")
    url = f"{base_url}/api/v2/{model}/history"
    auth_header = _build_auth_header(token, auth_scheme)
    logger.debug(
        "GribStream request model=%s url=%s accept=%s payload=%s",
        model,
        url,
        accept,
        _summarize_payload(payload),
    )
    headers = {
        "Content-Type": "application/json",
        "Accept": accept,
        "Accept-Encoding": "gzip",
        "Authorization": auth_header,
        "User-Agent": "weather-forecasting-predictionmarkets (ml_live)",
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    logger.debug(
        "GribStream response model=%s status=%s contentType=%s bytes=%s snippet=%s",
        model,
        resp.status_code,
        resp.headers.get("Content-Type", ""),
        len(resp.content or b""),
        _limit_snippet(resp.text),
    )
    resp.raise_for_status()
    return resp.text, resp.headers.get("Content-Type", "")


def fetch_forecast_payload(
    base_url: str,
    model: str,
    token: str,
    payload: dict[str, Any],
    accept: str,
    auth_scheme: str | None = None,
    timeout: int = 60,
) -> tuple[str, str]:
    if "forecastedFrom" not in payload or "forecastedUntil" not in payload:
        raise ValueError("GribStream forecast payload must include forecastedFrom/forecastedUntil")
    url = f"{base_url}/api/v2/{model}/forecasts"
    auth_header = _build_auth_header(token, auth_scheme)
    logger.debug(
        "GribStream request model=%s url=%s accept=%s payload=%s",
        model,
        url,
        accept,
        _summarize_payload(payload),
    )
    headers = {
        "Content-Type": "application/json",
        "Accept": accept,
        "Accept-Encoding": "gzip",
        "Authorization": auth_header,
        "User-Agent": "weather-forecasting-predictionmarkets (ml_live)",
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    logger.debug(
        "GribStream response model=%s status=%s contentType=%s bytes=%s snippet=%s",
        model,
        resp.status_code,
        resp.headers.get("Content-Type", ""),
        len(resp.content or b""),
        _limit_snippet(resp.text),
    )
    resp.raise_for_status()
    return resp.text, resp.headers.get("Content-Type", "")


def fetch_forecast_payload_raw(
    base_url: str,
    model: str,
    token: str,
    payload: dict[str, Any],
    accept: str,
    auth_scheme: str | None = None,
    timeout: int = 60,
) -> tuple[str, str, bytes, datetime]:
    if "forecastedFrom" not in payload or "forecastedUntil" not in payload:
        raise ValueError("GribStream forecast payload must include forecastedFrom/forecastedUntil")
    url = f"{base_url}/api/v2/{model}/forecasts"
    auth_header = _build_auth_header(token, auth_scheme)
    logger.debug(
        "GribStream request model=%s url=%s accept=%s payload=%s",
        model,
        url,
        accept,
        _summarize_payload(payload),
    )
    headers = {
        "Content-Type": "application/json",
        "Accept": accept,
        "Accept-Encoding": "gzip",
        "Authorization": auth_header,
        "User-Agent": "weather-forecasting-predictionmarkets (ml_live)",
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    logger.debug(
        "GribStream response model=%s status=%s contentType=%s bytes=%s snippet=%s",
        model,
        resp.status_code,
        resp.headers.get("Content-Type", ""),
        len(resp.content or b""),
        _limit_snippet(resp.text),
    )
    resp.raise_for_status()
    return resp.text, resp.headers.get("Content-Type", ""), resp.content or b"", datetime.now(timezone.utc)


def _parse_json_rows(text: str) -> list[dict[str, Any]]:
    cleaned = text.strip()
    if not cleaned:
        return []
    if cleaned.startswith("["):
        data = json.loads(cleaned)
        if not isinstance(data, list):
            raise ValueError("Expected JSON array from GribStream response")
        return data
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, list):
            rows.extend(obj)
        else:
            rows.append(obj)
    return rows


def _parse_payload_df(payload_text: str, content_type: str, accept: str) -> pd.DataFrame:
    content_type = (content_type or "").lower()
    if "text/csv" in content_type or accept.lower().startswith("text/csv"):
        df = pd.read_csv(io.StringIO(payload_text))
        logger.debug("GribStream parsed rows=%d columns=%s", len(df), list(df.columns))
        return df
    rows = _parse_json_rows(payload_text)
    df = pd.DataFrame(rows)
    logger.debug("GribStream parsed rows=%d columns=%s", len(df), list(df.columns))
    return df


def fetch_forecast_tmp(
    base_url: str,
    model: str,
    token: str,
    station_id: str,
    lat: float,
    lon: float,
    asof_utc: datetime,
    min_horizon: int,
    max_horizon: int,
    members: list[int] | None = None,
    accept: str = "text/csv",
    auth_scheme: str | None = None,
) -> pd.DataFrame:
    payload = _build_forecast_payload(
        station_id=station_id,
        lat=lat,
        lon=lon,
        asof_utc=asof_utc,
        min_horizon=min_horizon,
        max_horizon=max_horizon,
        members=members,
    )
    logger.info("Fetching GribStream forecasts model=%s asOf=%s", model, payload["forecastedFrom"])
    payload_text, content_type = fetch_forecast_payload(
        base_url,
        model,
        token,
        payload,
        accept=accept,
        auth_scheme=auth_scheme,
    )
    df = _parse_payload_df(payload_text, content_type, accept)
    return _normalize_and_filter(df, model, payload, asof_utc, lat, lon, accept)


def fetch_forecast_tmp_with_raw(
    base_url: str,
    model: str,
    token: str,
    station_id: str,
    lat: float,
    lon: float,
    asof_utc: datetime,
    min_horizon: int,
    max_horizon: int,
    members: list[int] | None = None,
    accept: str = "text/csv",
    auth_scheme: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload = _build_forecast_payload(
        station_id=station_id,
        lat=lat,
        lon=lon,
        asof_utc=asof_utc,
        min_horizon=min_horizon,
        max_horizon=max_horizon,
        members=members,
    )
    logger.info("Fetching GribStream forecasts model=%s asOf=%s", model, payload["forecastedFrom"])
    payload_text, content_type, response_bytes, retrieved_at_utc = fetch_forecast_payload_raw(
        base_url,
        model,
        token,
        payload,
        accept=accept,
        auth_scheme=auth_scheme,
    )
    df = _parse_payload_df(payload_text, content_type, accept)
    df = _normalize_and_filter(df, model, payload, asof_utc, lat, lon, accept)
    request_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    request_sha = _sha256_text(request_json)
    response_sha = _sha256_bytes(response_bytes)
    meta = {
        "payload": payload,
        "request_json": request_json,
        "request_sha256": request_sha,
        "response_sha256": response_sha,
        "retrieved_at_utc": retrieved_at_utc,
    }
    return df, meta


def _build_forecast_payload(
    station_id: str,
    lat: float,
    lon: float,
    asof_utc: datetime,
    min_horizon: int,
    max_horizon: int,
    members: list[int] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "forecastedFrom": asof_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "forecastedUntil": asof_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "minHorizon": int(min_horizon),
        "maxHorizon": int(max_horizon),
        "coordinates": [{"lat": lat, "lon": lon, "name": station_id}],
        "variables": [
            {
                "name": "TMP",
                "level": "2 m above ground",
                "info": "",
                "alias": "tmpK",
            }
        ],
    }
    if members:
        payload["members"] = members
    return payload


def _normalize_and_filter(
    df: pd.DataFrame,
    model: str,
    payload: dict[str, Any],
    asof_utc: datetime,
    lat: float,
    lon: float,
    accept: str,
) -> pd.DataFrame:
    if df.empty:
        raise ValueError(
            "Empty GribStream response for model="
            f"{model} asOf={payload.get('forecastedFrom')} "
            f"minHorizon={payload.get('minHorizon')} maxHorizon={payload.get('maxHorizon')} "
            f"accept={accept}"
        )
    df = _normalize_gribstream_df(df, default_lat=lat, default_lon=lon)
    df = _filter_asof_rows(df, asof_utc)
    if df.empty:
        raise ValueError(
            "Missing forecasted_at rows model="
            f"{model} asOf={payload.get('forecastedFrom')} "
            f"minHorizon={payload.get('minHorizon')} maxHorizon={payload.get('maxHorizon')}"
        )
    logger.debug(
        "GribStream normalized rows=%d forecasted_time=[%s..%s] forecasted_at=[%s..%s]",
        len(df),
        df["forecasted_time"].min(),
        df["forecasted_time"].max(),
        df["forecasted_at"].min(),
        df["forecasted_at"].max(),
    )
    return df


def fetch_hourly_tmp(
    base_url: str,
    model: str,
    token: str,
    station_id: str,
    lat: float,
    lon: float,
    from_time_utc: datetime,
    until_time_utc: datetime,
    asof_utc: datetime,
    members: list[int] | None = None,
    accept: str = "application/ndjson",
    auth_scheme: str | None = None,
) -> pd.DataFrame:
    payload: dict[str, Any] = {
        "fromTime": from_time_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "untilTime": until_time_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "asOf": asof_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "coordinates": [{"lat": lat, "lon": lon, "name": station_id}],
        "variables": [
            {
                "name": "TMP",
                "level": "2 m above ground",
                "info": "",
                "alias": "tmpk",
            }
        ],
    }
    if members:
        payload["members"] = members
    logger.info("Fetching GribStream history model=%s asOf=%s", model, payload["asOf"])
    payload_text, content_type = fetch_history_payload(
        base_url,
        model,
        token,
        payload,
        accept=accept,
        auth_scheme=auth_scheme,
    )
    df = _parse_payload_df(payload_text, content_type, accept)
    if df.empty:
        raise ValueError(
            "Empty GribStream response for model="
            f"{model} asOf={payload.get('asOf')} "
            f"from={payload.get('fromTime')} until={payload.get('untilTime')} "
            f"accept={accept}"
        )
    df = _normalize_gribstream_df(df, default_lat=lat, default_lon=lon)
    logger.debug(
        "GribStream normalized rows=%d forecasted_time=[%s..%s] forecasted_at=[%s..%s]",
        len(df),
        df["forecasted_time"].min(),
        df["forecasted_time"].max(),
        df["forecasted_at"].min(),
        df["forecasted_at"].max(),
    )
    return df


def _filter_asof_rows(df: pd.DataFrame, asof_utc: datetime) -> pd.DataFrame:
    if asof_utc.tzinfo is None:
        asof_utc = asof_utc.replace(tzinfo=timezone.utc)
    asof_ts = pd.Timestamp(asof_utc).tz_convert("UTC")
    return df[df["forecasted_at"] == asof_ts].copy()


def _normalize_gribstream_df(
    df: pd.DataFrame,
    default_lat: float | None = None,
    default_lon: float | None = None,
) -> pd.DataFrame:
    df = df.copy()
    if "member" not in df.columns:
        df = _expand_member_columns(df)
    if "tmp_k" not in df.columns:
        tmp_col = _resolve_column(df, ["tmpK", "tmp_k", "tmpk"])
        df.rename(columns={tmp_col: "tmp_k"}, inplace=True)
    forecasted_time_col = _resolve_column(df, ["forecasted_time", "forecasted_time_utc", "forecast_time"])
    forecasted_at_col = _resolve_column(df, ["forecasted_at", "forecasted_at_utc", "forecasted_from"])
    df.rename(columns={forecasted_time_col: "forecasted_time"}, inplace=True)
    df.rename(columns={forecasted_at_col: "forecasted_at"}, inplace=True)
    if "lat" not in df.columns:
        if default_lat is None:
            raise ValueError("Missing lat in GribStream response")
        df["lat"] = default_lat
    if "lon" not in df.columns:
        if default_lon is None:
            raise ValueError("Missing lon in GribStream response")
        df["lon"] = default_lon
    if "member" not in df.columns:
        df["member"] = 0
    df["member"] = pd.to_numeric(df["member"], errors="coerce").fillna(0).astype(int)
    df["forecasted_time"] = pd.to_datetime(df["forecasted_time"], utc=True)
    df["forecasted_at"] = pd.to_datetime(df["forecasted_at"], utc=True)
    return df[["forecasted_at", "forecasted_time", "lat", "lon", "member", "tmp_k"]]


def _resolve_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(f"Missing required column in GribStream response. Tried: {candidates}")


def _expand_member_columns(df: pd.DataFrame) -> pd.DataFrame:
    member_cols: dict[str, int] = {}
    for col in df.columns:
        match = _MEMBER_KEY_PATTERN.match(str(col))
        if match:
            member_cols[col] = int(match.group(1))
    if not member_cols:
        return df
    base_cols = [col for col in df.columns if col not in member_cols]
    frames: list[pd.DataFrame] = []
    for col, member in member_cols.items():
        temp = df[base_cols].copy()
        temp["member"] = member
        temp["tmp_k"] = df[col]
        frames.append(temp)
    return pd.concat(frames, ignore_index=True)


def to_fahrenheit_from_kelvin(tmp_k: pd.Series) -> pd.Series:
    return (tmp_k - 273.15) * 9.0 / 5.0 + 32.0


def compute_daily_tmax_f(
    df: pd.DataFrame,
    target_date_local: date,
    zone_id: str,
    min_points: int,
) -> float:
    if df.empty:
        raise ValueError("Empty GribStream dataframe for daily tmax")
    local_dates = df["forecasted_time"].dt.tz_convert(ZoneInfo(zone_id)).dt.date
    temps_k = df.loc[local_dates == target_date_local, "tmp_k"].astype(float).dropna()
    points = int(temps_k.shape[0])
    if points < 1:
        raise ValueError(
            f"Missing tmax values target_date_local={target_date_local} zone_id={zone_id}"
        )
    if points < min_points:
        raise ValueError(
            "Incomplete tmax day "
            f"target_date_local={target_date_local} zone_id={zone_id} "
            f"points={points} min_points={min_points}"
        )
    max_k = float(temps_k.max())
    return float((max_k - 273.15) * 9.0 / 5.0 + 32.0)


@dataclass(frozen=True)
class DailyValue:
    value_f: float
    value_k: float
    points: int


def compute_daily_tmax(
    df: pd.DataFrame,
    target_date_local: date,
    zone_id: str,
    min_points: int,
) -> DailyValue:
    if df.empty:
        raise ValueError("Empty GribStream dataframe for daily tmax")
    local_dates = df["forecasted_time"].dt.tz_convert(ZoneInfo(zone_id)).dt.date
    temps_k = df.loc[local_dates == target_date_local, "tmp_k"].astype(float).dropna()
    points = int(temps_k.shape[0])
    if points < 1:
        raise ValueError(
            f"Missing tmax values target_date_local={target_date_local} zone_id={zone_id}"
        )
    if points < min_points:
        raise ValueError(
            "Incomplete tmax day "
            f"target_date_local={target_date_local} zone_id={zone_id} "
            f"points={points} min_points={min_points}"
        )
    max_k = float(temps_k.max())
    value_f = float((max_k - 273.15) * 9.0 / 5.0 + 32.0)
    return DailyValue(value_f=value_f, value_k=max_k, points=points)


def compute_gefs_spread_f(
    df: pd.DataFrame,
    target_date_local: date,
    zone_id: str,
    expected_members: list[int],
    min_points: int,
) -> float:
    if df.empty:
        raise ValueError("Empty GribStream dataframe for GEFS spread")
    local_dates = df["forecasted_time"].dt.tz_convert(ZoneInfo(zone_id)).dt.date
    filtered = df.loc[local_dates == target_date_local].copy()
    expected = len(expected_members)
    stddevs_k: list[float] = []
    for _, group in filtered.groupby("forecasted_time"):
        temps = group.dropna(subset=["member", "tmp_k"]).groupby("member")["tmp_k"].mean()
        if temps.shape[0] < expected:
            continue
        stddevs_k.append(float(np.std(temps.astype(float), ddof=0)))
    points = len(stddevs_k)
    if points < 1:
        raise ValueError(
            "Missing GEFS members "
            f"target_date_local={target_date_local} zone_id={zone_id} "
            f"expected_members={expected}"
        )
    if points < min_points:
        raise ValueError(
            "Incomplete GEFS spread day "
            f"target_date_local={target_date_local} zone_id={zone_id} "
            f"points={points} min_points={min_points}"
        )
    mean_std_k = float(np.mean(stddevs_k))
    return float(mean_std_k * 9.0 / 5.0)


def compute_gefs_spread(
    df: pd.DataFrame,
    target_date_local: date,
    zone_id: str,
    expected_members: list[int],
    min_points: int,
) -> DailyValue:
    if df.empty:
        raise ValueError("Empty GribStream dataframe for GEFS spread")
    local_dates = df["forecasted_time"].dt.tz_convert(ZoneInfo(zone_id)).dt.date
    filtered = df.loc[local_dates == target_date_local].copy()
    expected = len(expected_members)
    stddevs_k: list[float] = []
    for _, group in filtered.groupby("forecasted_time"):
        temps = group.dropna(subset=["member", "tmp_k"]).groupby("member")["tmp_k"].mean()
        if temps.shape[0] < expected:
            continue
        stddevs_k.append(float(np.std(temps.astype(float), ddof=0)))
    points = len(stddevs_k)
    if points < 1:
        raise ValueError(
            "Missing GEFS members "
            f"target_date_local={target_date_local} zone_id={zone_id} "
            f"expected_members={expected}"
        )
    if points < min_points:
        raise ValueError(
            "Incomplete GEFS spread day "
            f"target_date_local={target_date_local} zone_id={zone_id} "
            f"points={points} min_points={min_points}"
        )
    mean_std_k = float(np.mean(stddevs_k))
    mean_std_f = float(mean_std_k * 9.0 / 5.0)
    return DailyValue(value_f=mean_std_f, value_k=mean_std_k, points=points)


def compute_gefs_mean_tmax_f(
    df: pd.DataFrame,
    target_date_local: date,
    zone_id: str,
    expected_members: list[int],
    min_points: int,
) -> float:
    if df.empty:
        raise ValueError("Empty GribStream dataframe for GEFS mean")
    local_dates = df["forecasted_time"].dt.tz_convert(ZoneInfo(zone_id)).dt.date
    filtered = df.loc[local_dates == target_date_local].copy()
    expected = len(expected_members)
    means_k: list[float] = []
    for _, group in filtered.groupby("forecasted_time"):
        temps = group.dropna(subset=["member", "tmp_k"]).groupby("member")["tmp_k"].mean()
        if temps.shape[0] < expected:
            continue
        means_k.append(float(temps.astype(float).mean()))
    points = len(means_k)
    if points < 1:
        raise ValueError(
            "Missing GEFS mean members "
            f"target_date_local={target_date_local} zone_id={zone_id} "
            f"expected_members={expected}"
        )
    if points < min_points:
        raise ValueError(
            "Incomplete GEFS mean day "
            f"target_date_local={target_date_local} zone_id={zone_id} "
            f"points={points} min_points={min_points}"
        )
    max_k = float(max(means_k))
    return float((max_k - 273.15) * 9.0 / 5.0 + 32.0)


def compute_gefs_mean_tmax(
    df: pd.DataFrame,
    target_date_local: date,
    zone_id: str,
    expected_members: list[int],
    min_points: int,
) -> DailyValue:
    if df.empty:
        raise ValueError("Empty GribStream dataframe for GEFS mean")
    local_dates = df["forecasted_time"].dt.tz_convert(ZoneInfo(zone_id)).dt.date
    filtered = df.loc[local_dates == target_date_local].copy()
    expected = len(expected_members)
    means_k: list[float] = []
    for _, group in filtered.groupby("forecasted_time"):
        temps = group.dropna(subset=["member", "tmp_k"]).groupby("member")["tmp_k"].mean()
        if temps.shape[0] < expected:
            continue
        means_k.append(float(temps.astype(float).mean()))
    points = len(means_k)
    if points < 1:
        raise ValueError(
            "Missing GEFS mean members "
            f"target_date_local={target_date_local} zone_id={zone_id} "
            f"expected_members={expected}"
        )
    if points < min_points:
        raise ValueError(
            "Incomplete GEFS mean day "
            f"target_date_local={target_date_local} zone_id={zone_id} "
            f"points={points} min_points={min_points}"
        )
    max_k = float(max(means_k))
    value_f = float((max_k - 273.15) * 9.0 / 5.0 + 32.0)
    return DailyValue(value_f=value_f, value_k=max_k, points=points)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
