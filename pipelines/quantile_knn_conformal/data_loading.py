from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from tools.data_sanitizer.data_sanitizer import load_rules, read_station_universe, sanitize_observations_dataframe


REQUIRED_OBS_COLUMNS = [
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
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_of_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_manifest(stage_dir: str | Path, manifest: dict[str, Any]) -> None:
    out = ensure_dir(stage_dir) / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_station_universe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "request_location_id" not in df.columns or "role" not in df.columns:
        raise ValueError("station_universe must include request_location_id and role")
    df["request_location_id"] = df["request_location_id"].astype(str).str.upper().str.strip()
    df["role"] = df["role"].astype(str).str.lower().str.strip()
    if (df["role"] == "target").sum() != 1:
        raise ValueError("station_universe must include exactly one target row")
    return df


def load_truth(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = set(df.columns)
    if {"station_id", "date", "settled_tmax"}.issubset(cols):
        out = df.rename(columns={"date": "target_date_local", "settled_tmax": "y_tmax"}).copy()
    elif {"station_id", "target_date_local", "max_temp_f"}.issubset(cols):
        out = df.rename(columns={"max_temp_f": "y_tmax"}).copy()
    else:
        raise ValueError(f"Unsupported truth schema in {path}; expected KNYC_settled_tmax or daily_max_truth-like columns")
    out["station_id"] = out["station_id"].astype(str).str.upper().str.strip()
    out["target_date_local"] = pd.to_datetime(out["target_date_local"]).dt.date
    out["y_tmax"] = pd.to_numeric(out["y_tmax"], errors="coerce")
    out = out.dropna(subset=["target_date_local", "y_tmax"])
    out["y_tmax"] = out["y_tmax"].astype(float)
    return out


def load_and_optionally_sanitize_observations(
    obs_csv: str,
    station_universe_csv: str,
    skip_sanitization: bool,
    schema_profile: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(obs_csv, low_memory=False)
    missing = [c for c in REQUIRED_OBS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required observation columns: {missing}")

    sanitization_summary: dict[str, Any] = {
        "enabled": not skip_sanitization,
        "rows_input": int(len(df)),
    }

    if not skip_sanitization:
        rules_path = Path("tools") / "data_sanitizer" / "default_rules.yaml"
        rules = load_rules(str(rules_path))
        station_universe = read_station_universe(station_universe_csv)
        sdf, report = sanitize_observations_dataframe(
            df,
            rules=rules,
            station_universe=station_universe,
            emit_flags=False,
            drop_invalid_timestamps=True,
            fill_wdir_from_cardinal=False,
            enforce_30m_grid=False,
            collect_triggered_rules=False,
        )
        df = sdf
        sanitization_summary["report"] = report
        sanitization_summary["rows_output"] = int(len(df))
    else:
        sanitization_summary["rows_output"] = int(len(df))

    df["request_location_id"] = df["request_location_id"].astype(str).str.upper().str.strip()
    df["valid_time_utc"] = pd.to_datetime(df["valid_time_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["valid_time_utc", "request_location_id"]).copy()

    num_cols = ["temp", "dew_pt", "rh", "pressure", "vis", "wspd", "wdir", "gust", "precip_hrly", "uv_index"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    cat_cols = ["clds", "wx_phrase", "uv_desc", "wdir_cardinal"]
    for c in cat_cols:
        df[c] = df[c].astype("string")

    df = df.sort_values(["request_location_id", "valid_time_utc"]).reset_index(drop=True)
    sanitization_summary["rows_after_timestamp_parse"] = int(len(df))
    return df, sanitization_summary
