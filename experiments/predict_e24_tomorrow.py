"""Fetch MOS + Gribstream inputs, build E24 features, and predict tomorrow's Tmax."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
import gzip
import io

import numpy as np
import pandas as pd
import yaml
import joblib

# Allow importing weather_ml from repo.
REPO_ROOT = Path(__file__).resolve().parents[1]
ML_SRC = REPO_ROOT / "ml" / "src"
if str(ML_SRC) not in os.sys.path:
    os.sys.path.insert(0, str(ML_SRC))

from weather_ml import time_feature_library as tfl  # noqa: E402
from weather_ml import derived_features  # noqa: E402

MODEL_COLS = [
    "nbm_tmax_f",
    "gfs_tmax_f",
    "gefsatmosmean_tmax_f",
    "nam_tmax_f",
    "hrrr_tmax_f",
    "rap_tmax_f",
]
SPREAD_COL = "gefsatmos_tmp_spread_f"


@dataclass(frozen=True)
class StationConfig:
    station_id: str
    zone_id: str
    lat: float
    lon: float
    name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch MOS + Gribstream inputs and predict tomorrow using E24 model."
    )
    parser.add_argument(
        "--app-config",
        default=str(REPO_ROOT / "ingestion-service" / "src" / "main" / "resources" / "application.yml"),
        help="Path to application.yml (for IEM + Gribstream config).",
    )
    parser.add_argument(
        "--train-config",
        default=str(REPO_ROOT / "ml" / "configs" / "train_mean_sigma.yaml"),
        help="Training config used by time_feature_sweep (for fallback training).",
    )
    parser.add_argument(
        "--training-csv",
        default=str(
            REPO_ROOT
            / "ingestion-service"
            / "src"
            / "main"
            / "resources"
            / "trainingdata_output"
            / "gribstream_training_data.csv"
        ),
        help="CSV containing historical features + actuals.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(REPO_ROOT / "artifacts" / "time_feature_sweep" / "sweep_002" / "E24"),
        help="Directory containing E24 mean_model.joblib + feature_state.joblib.",
    )
    parser.add_argument("--station-id", help="Station ID (e.g., KNYC).")
    parser.add_argument(
        "--target-date",
        help="Target date local (YYYY-MM-DD). Default: tomorrow in station zone.",
    )
    parser.add_argument(
        "--asof-local-time",
        default="12:00",
        help="As-of local time (default: 12:00).",
    )
    parser.add_argument(
        "--asof-time-zone",
        default="UTC",
        help="As-of time zone (default: UTC).",
    )
    parser.add_argument(
        "--mos-runtime-window-hours",
        type=int,
        default=48,
        help="Hours to look back for MOS runtimes (default: 48).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app_config = load_yaml(Path(args.app_config))
    station = resolve_station(app_config, args.station_id)

    target_date_local = resolve_target_date(args.target_date, station.zone_id)
    asof_utc = resolve_asof_utc(
        target_date_local, args.asof_local_time, args.asof_time_zone, station.zone_id
    )

    model_dir = Path(args.model_dir)
    ensure_e24_model(model_dir, Path(args.train_config))

    mos = fetch_mos_tmax(
        app_config,
        station,
        target_date_local,
        asof_utc,
        args.mos_runtime_window_hours,
    )
    grib = fetch_gribstream_tmax(
        app_config,
        station,
        target_date_local,
        asof_utc,
    )

    row = {
        "station_id": station.station_id,
        "target_date_local": target_date_local,
        "asof_utc": asof_utc,
        "gfs_tmax_f": mos["gfs_tmax_f"],
        "nam_tmax_f": mos["nam_tmax_f"],
        "gefsatmosmean_tmax_f": grib["gefsatmosmean_tmax_f"],
        "rap_tmax_f": grib["rap_tmax_f"],
        "hrrr_tmax_f": grib["hrrr_tmax_f"],
        "nbm_tmax_f": grib["nbm_tmax_f"],
        "gefsatmos_tmp_spread_f": grib["gefsatmos_tmp_spread_f"],
        "actual_tmax_f": np.nan,
    }

    history = load_history(Path(args.training_csv), station.station_id)
    combined = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    features = build_e24_features(combined)

    feature_state = joblib.load(model_dir / "feature_state.joblib")
    feature_columns = feature_state["feature_columns"]
    impute_values = feature_state["impute_values"]["fill_values"]
    model = joblib.load(model_dir / "mean_model.joblib")

    row_features = features.iloc[-1:].copy()
    row_features = row_features.reindex(columns=feature_columns)
    row_features = apply_imputation(row_features, impute_values)

    prediction = float(model.predict(row_features.to_numpy(dtype=float))[0])
    payload = {
        "station_id": station.station_id,
        "target_date_local": target_date_local.isoformat(),
        "asof_utc": asof_utc.isoformat(),
        "prediction_tmax_f": prediction,
        "inputs": {k: row[k] for k in MODEL_COLS + [SPREAD_COL]},
    }
    print(json.dumps(payload, indent=2))
    return 0


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_station(app_config: dict, station_id: str | None) -> StationConfig:
    stations = ((app_config.get("gribstream") or {}).get("stations") or [])
    if not stations:
        raise ValueError("gribstream.stations is required in application.yml")
    station_map = {s["stationId"].upper(): s for s in stations}

    if station_id is None:
        station_ids_to_run = (
            (app_config.get("pipeline") or {}).get("station-ids-to-run") or ""
        )
        station_id = station_ids_to_run.split(",")[0].strip().upper() if station_ids_to_run else None
    if station_id is None:
        station_id = next(iter(station_map.keys()))
    station_id = station_id.upper()
    if station_id not in station_map:
        raise ValueError(f"Station not found in gribstream.stations: {station_id}")
    cfg = station_map[station_id]
    return StationConfig(
        station_id=cfg["stationId"].upper(),
        zone_id=cfg["zoneId"],
        lat=float(cfg["latitude"]),
        lon=float(cfg["longitude"]),
        name=cfg.get("name") or cfg["stationId"],
    )


def resolve_target_date(target_date: str | None, zone_id: str) -> date:
    if target_date:
        return date.fromisoformat(target_date)
    tz = zoneinfo(zone_id)
    return (datetime.now(tz).date() + timedelta(days=1))


def resolve_asof_utc(
    target_date: date, asof_time: str, asof_tz: str, station_zone: str
) -> datetime:
    hour, minute = [int(part) for part in asof_time.split(":")]
    if asof_tz.upper() == "UTC":
        tz = timezone.utc
    else:
        tz = zoneinfo(asof_tz if asof_tz != "LOCAL" else station_zone)
    asof_local = datetime.combine(target_date, time(hour, minute), tzinfo=tz)
    return asof_local.astimezone(timezone.utc)


def zoneinfo(zone_id: str):
    from zoneinfo import ZoneInfo

    return ZoneInfo(zone_id)


def ensure_e24_model(model_dir: Path, train_config: Path) -> None:
    if (model_dir / "mean_model.joblib").exists():
        return
    cmd = [
        os.sys.executable,
        str(REPO_ROOT / "scripts" / "run_time_feature_sweep.py"),
        "--config",
        str(train_config),
        "--sweep-id",
        model_dir.parent.name,
        "--experiment-ids",
        "E24",
    ]
    subprocess.check_call(cmd, cwd=str(REPO_ROOT))


def load_history(csv_path: Path, station_id: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df[df["station_id"].str.upper() == station_id.upper()].copy()
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    df["asof_utc"] = pd.to_datetime(df["asof_utc"], utc=True)
    return df


def build_e24_features(df: pd.DataFrame) -> pd.DataFrame:
    df = tfl.prepare_frame(df)
    df = tfl.add_calendar_features(df)
    df = tfl.add_ensemble_stats(df, MODEL_COLS)
    if "actual_tmax_f" in df.columns:
        df["resid_ens_mean"] = df["actual_tmax_f"] - df["ens_mean"]
    gk = df["station_id"]
    lag = 2
    min_periods = int(math.ceil(0.7 * 60))

    features = df[MODEL_COLS + [SPREAD_COL, "month", "day_of_year", "sin_doy", "cos_doy", "is_weekend"]].copy()

    mae_cols = []
    for col in MODEL_COLS:
        resid = df["actual_tmax_f"] - df[col]
        mae = tfl.rolling_mean(
            resid.abs(), window=60, min_periods=min_periods, lag=lag, group_key=gk
        )
        name = f"mae_{col}_rm60_l{lag}"
        features[name] = mae
        mae_cols.append(name)
    mae_ens = tfl.rolling_mean(
        df["resid_ens_mean"].abs(),
        window=60,
        min_periods=min_periods,
        lag=lag,
        group_key=gk,
    )
    features[f"mae_ensmean_rm60_l{lag}"] = mae_ens
    for col, mae_name in zip(MODEL_COLS, mae_cols):
        features[f"rel_mae_{col}_vs_ens_rm60_l{lag}"] = features[mae_name] - mae_ens

    mae_frame = pd.DataFrame(
        {model: features[name] for model, name in zip(MODEL_COLS, mae_cols)},
        index=df.index,
    )
    best_id = tfl.argmin_with_tie_break(mae_frame, MODEL_COLS)
    for col in MODEL_COLS:
        features[f"best_is_{col}"] = (best_id == col).astype(int)
    best_forecast = np.full(len(df), np.nan, dtype=float)
    for col in MODEL_COLS:
        mask = best_id == col
        best_forecast[mask.to_numpy()] = df.loc[mask, col].to_numpy(dtype=float)
    features["best_model_forecast_today"] = best_forecast
    return features


def apply_imputation(df: pd.DataFrame, fill_values: dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    for col, value in fill_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(float(value))
    return df.fillna(0.0)


def fetch_mos_tmax(
    app_config: dict,
    station: StationConfig,
    target_date_local: date,
    asof_utc: datetime,
    runtime_window_hours: int,
) -> dict[str, float]:
    base_url = (app_config.get("iem") or {}).get("base-url", "https://mesonet.agron.iastate.edu")
    start = asof_utc - timedelta(hours=runtime_window_hours)
    end = asof_utc + timedelta(hours=1)
    gfs = fetch_mos_model(
        base_url, station, "GFS", start, end, target_date_local, asof_utc
    )
    nam = fetch_mos_model(
        base_url, station, "NAM", start, end, target_date_local, asof_utc
    )
    return {"gfs_tmax_f": gfs, "nam_tmax_f": nam}


def fetch_mos_model(
    base_url: str,
    station: StationConfig,
    model: str,
    start: datetime,
    end: datetime,
    target_date_local: date,
    asof_utc: datetime,
) -> float:
    params = {
        "station": station.station_id,
        "model": model,
        "sts": start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "ets": end.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "format": "json",
    }
    url = f"{base_url}/cgi-bin/request/mos.py?{encode_params(params)}"
    entries = fetch_json(url)
    if not isinstance(entries, list):
        raise ValueError(f"IEM MOS response not a list for model={model}")
    runtime_candidates = []
    for entry in entries:
        runtime_ms = entry.get("runtime")
        if runtime_ms is None:
            continue
        runtime = datetime.fromtimestamp(runtime_ms / 1000, tz=timezone.utc)
        if runtime <= asof_utc:
            runtime_candidates.append(runtime)
    if not runtime_candidates:
        raise ValueError(f"No MOS runtimes <= asOf for model={model}")
    chosen_runtime = max(runtime_candidates)

    tmax_values = []
    for entry in entries:
        runtime_ms = entry.get("runtime")
        ftime_ms = entry.get("ftime")
        value = entry.get("n_x")
        if runtime_ms is None or ftime_ms is None:
            continue
        runtime = datetime.fromtimestamp(runtime_ms / 1000, tz=timezone.utc)
        if runtime != chosen_runtime:
            continue
        forecast_time = datetime.fromtimestamp(ftime_ms / 1000, tz=timezone.utc)
        forecast_local = forecast_time.astimezone(zoneinfo(station.zone_id)).date()
        if forecast_local != target_date_local:
            continue
        if value is None:
            continue
        try:
            tmax_values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not tmax_values:
        raise ValueError(f"No MOS tmax values found for model={model}")
    return float(max(tmax_values))


def fetch_gribstream_tmax(
    app_config: dict,
    station: StationConfig,
    target_date_local: date,
    asof_utc: datetime,
) -> dict[str, float]:
    grib_cfg = app_config.get("gribstream") or {}
    base_url = grib_cfg.get("baseUrl", "https://gribstream.com")
    token = grib_cfg.get("apiToken")
    auth_scheme = grib_cfg.get("authScheme", "Bearer")
    if not token:
        raise ValueError("gribstream.apiToken is required in application.yml")
    auth_header = token if " " in token else f"{auth_scheme} {token}"

    start_utc, end_utc = standard_time_window(station.zone_id, target_date_local)
    min_horizon = int((grib_cfg.get("models") or {}).get("defaultMinHorizonHours", 0))

    max_horizon_map = grib_cfg.get("models") or {}
    def max_horizon(model_code: str) -> int:
        model_cfg = max_horizon_map.get(model_code)
        if not model_cfg or not model_cfg.get("maxHorizonHours"):
            raise ValueError(f"Missing gribstream.models.{model_code}.maxHorizonHours")
        return int(model_cfg["maxHorizonHours"])

    coords = [{"lat": station.lat, "lon": station.lon, "name": station.name}]
    variables = [{"name": "TMP", "level": "2 m above ground", "info": "", "alias": "tmpk"}]

    def fetch_model(model_code: str, members: list[int] | None) -> list[dict]:
        payload = {
            "fromTime": start_utc.isoformat().replace("+00:00", "Z"),
            "untilTime": end_utc.isoformat().replace("+00:00", "Z"),
            "asOf": asof_utc.isoformat().replace("+00:00", "Z"),
            "minHorizon": min_horizon,
            "maxHorizon": max_horizon(model_code),
            "coordinates": coords,
            "variables": variables,
        }
        if members is not None:
            payload["members"] = members
        url = f"{base_url}/api/v2/{model_code}/history"
        return fetch_gribstream_rows(url, auth_header, payload)

    hrrr_rows = fetch_model("hrrr", None)
    rap_rows = fetch_model("rap", None)
    nbm_rows = fetch_model("nbm", None)

    gefs_members = (grib_cfg.get("gefs") or {}).get("members") or []
    gefs_rows = fetch_model("gefsatmos", gefs_members)

    use_mean_endpoint = (grib_cfg.get("gefs") or {}).get("useMeanEndpoint", False)
    if use_mean_endpoint:
        gefs_mean_rows = fetch_model("gefsatmosmean", None)
        gefs_mean_f = tmax_from_rows(gefs_mean_rows)
    else:
        gefs_mean_f = ensemble_mean_tmax(gefs_rows, min_members=10)

    spread_f = ensemble_spread(gefs_rows, min_members=10)

    return {
        "hrrr_tmax_f": tmax_from_rows(hrrr_rows),
        "rap_tmax_f": tmax_from_rows(rap_rows),
        "nbm_tmax_f": tmax_from_rows(nbm_rows),
        "gefsatmosmean_tmax_f": gefs_mean_f,
        "gefsatmos_tmp_spread_f": spread_f,
    }


def standard_time_window(zone_id: str, target_date: date) -> tuple[datetime, datetime]:
    tz = zoneinfo(zone_id)
    noon = datetime.combine(target_date, time(12, 0), tzinfo=tz)
    offset = noon.utcoffset() or timedelta(0)
    dst = noon.dst() or timedelta(0)
    standard = offset - dst
    start = datetime.combine(target_date, time(0, 0), tzinfo=timezone(standard))
    start_utc = start.astimezone(timezone.utc)
    end_utc = start_utc + timedelta(days=1)
    return start_utc, end_utc


def fetch_gribstream_rows(url: str, auth_header: str, payload: dict) -> list[dict]:
    body = json.dumps(payload).encode("utf-8")
    req = Request(url, data=body, method="POST")
    req.add_header("Authorization", auth_header)
    req.add_header("Accept", "application/ndjson")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept-Encoding", "gzip")
    with urlopen(req, timeout=30) as resp:
        raw = resp.read()
        if resp.headers.get("Content-Encoding", "").lower() == "gzip":
            raw = gzip.decompress(raw)
    return parse_ndjson_rows(raw)


def parse_ndjson_rows(raw: bytes) -> list[dict]:
    text = raw.decode("utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    rows: list[dict] = []
    for line in lines:
        obj = json.loads(line)
        if isinstance(obj, list):
            rows.extend(obj)
        else:
            rows.append(obj)
    return rows


def tmax_from_rows(rows: list[dict]) -> float:
    max_k = None
    for row in rows:
        val = row.get("tmpk")
        if val is None:
            continue
        try:
            val = float(val)
        except (TypeError, ValueError):
            continue
        if max_k is None or val > max_k:
            max_k = val
    if max_k is None:
        raise ValueError("No tmpk values in gribstream response")
    return kelvin_to_f(max_k)


def ensemble_spread(rows: list[dict], min_members: int) -> float:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        t = row.get("forecasted_time")
        val = row.get("tmpk")
        if t is None or val is None:
            continue
        grouped.setdefault(t, []).append(float(val))
    max_std = None
    used = 0
    for values in grouped.values():
        if len(values) < min_members:
            continue
        used += 1
        std = float(np.std(values, ddof=0))
        if max_std is None or std > max_std:
            max_std = std
    if max_std is None or used == 0:
        raise ValueError("Insufficient member coverage for GEFS spread")
    return max_std * 9.0 / 5.0


def ensemble_mean_tmax(rows: list[dict], min_members: int) -> float:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        t = row.get("forecasted_time")
        val = row.get("tmpk")
        if t is None or val is None:
            continue
        grouped.setdefault(t, []).append(float(val))
    max_mean = None
    for values in grouped.values():
        if len(values) < min_members:
            continue
        mean = float(np.mean(values))
        if max_mean is None or mean > max_mean:
            max_mean = mean
    if max_mean is None:
        raise ValueError("Insufficient member coverage for GEFS mean")
    return kelvin_to_f(max_mean)


def kelvin_to_f(value_k: float) -> float:
    return (value_k - 273.15) * 9.0 / 5.0 + 32.0


def fetch_json(url: str) -> Any:
    with urlopen(url, timeout=30) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def encode_params(params: dict[str, Any]) -> str:
    from urllib.parse import urlencode

    return urlencode(params)


if __name__ == "__main__":
    raise SystemExit(main())
