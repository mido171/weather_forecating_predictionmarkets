from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import pandas as pd

from download import SnapshotResult

TRUTH_SOURCE = "NCEI_ADS_DAILY_SUMMARIES"


@dataclass(frozen=True)
class AttributeTokens:
    measurement_flag: str
    quality_flag: str
    source_flag: str
    obs_time_hhmm: str
    raw: str


def parse_tmax_attributes(raw_value: Any) -> AttributeTokens:
    raw = "" if raw_value is None else str(raw_value)
    parts = raw.split(",")
    if len(parts) < 4:
        parts = parts + [""] * (4 - len(parts))
    elif len(parts) > 4:
        parts = parts[:4]
    return AttributeTokens(
        measurement_flag=(parts[0] or "").strip(),
        quality_flag=(parts[1] or "").strip(),
        source_flag=(parts[2] or "").strip(),
        obs_time_hhmm=(parts[3] or "").strip(),
        raw=raw,
    )


def _round_half_away_from_zero(v: float) -> int:
    d = Decimal(str(v))
    return int(d.to_integral_value(rounding=ROUND_HALF_UP))


def _coerce_tmax_to_int_f(value: Any) -> int | None:
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    try:
        f = float(s)
    except Exception:
        return None
    if not math.isfinite(f):
        return None
    if abs(f - round(f)) <= 1e-12:
        return int(round(f))
    return _round_half_away_from_zero(f)


def _source_record_id(
    *,
    station_usw: str,
    date_local: str,
    attrs_raw: str,
) -> str:
    return (
        "NCEI_ADS|daily-summaries"
        f"|station={station_usw}"
        f"|date={date_local}"
        "|datatype=TMAX"
        f"|attrs={attrs_raw}"
    )


def normalize_snapshots_to_rows(
    *,
    snapshots: list[SnapshotResult],
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for snap in snapshots:
        if snap.http_status != 200:
            logger.warning(
                "NCEI_NORMALIZE_SKIP_NON200 station=%s usw=%s range=%s..%s status=%d",
                snap.station_id,
                snap.station_usw,
                snap.start_date,
                snap.end_date,
                snap.http_status,
            )
            continue
        try:
            payload = json.loads(snap.response_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(
                "NCEI_NORMALIZE_JSON_ERROR station=%s usw=%s range=%s..%s err=%s",
                snap.station_id,
                snap.station_usw,
                snap.start_date,
                snap.end_date,
                exc,
            )
            continue
        if not isinstance(payload, list):
            logger.warning(
                "NCEI_NORMALIZE_PAYLOAD_NOT_LIST station=%s usw=%s range=%s..%s",
                snap.station_id,
                snap.station_usw,
                snap.start_date,
                snap.end_date,
            )
            continue

        for item in payload:
            if not isinstance(item, dict):
                continue
            tmax_int = _coerce_tmax_to_int_f(item.get("TMAX"))
            if tmax_int is None:
                continue
            date_local = str(item.get("DATE") or "").strip()
            if not date_local:
                continue
            attrs = parse_tmax_attributes(item.get("TMAX_ATTRIBUTES"))
            rows.append(
                {
                    "station_id": snap.station_id,
                    "station_usw": snap.station_usw,
                    "target_date_local": date_local,
                    "tmax_f": int(tmax_int),
                    "truth_source": TRUTH_SOURCE,
                    "source_record_id": _source_record_id(
                        station_usw=snap.station_usw,
                        date_local=date_local,
                        attrs_raw=attrs.raw,
                    ),
                    "retrieved_at_utc": snap.retrieved_at_utc,
                    "attribute_measurement_flag": attrs.measurement_flag,
                    "attribute_quality_flag": attrs.quality_flag,
                    "attribute_source_flag": attrs.source_flag,
                    "attribute_obs_time_hhmm": attrs.obs_time_hhmm,
                    "attribute_raw": attrs.raw,
                    "source_station_field": str(item.get("STATION") or "").strip(),
                }
            )
    return rows


def write_canonical_csv(rows: list[dict[str, Any]], canonical_csv_path: Path) -> pd.DataFrame:
    canonical_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        df = pd.DataFrame(
            columns=[
                "station_id",
                "target_date_local",
                "tmax_f",
                "truth_source",
                "source_record_id",
                "retrieved_at_utc",
            ]
        )
        df.to_csv(canonical_csv_path, index=False, encoding="utf-8")
        return df

    df = pd.DataFrame(rows)
    df["target_date_local"] = pd.to_datetime(df["target_date_local"], errors="coerce").dt.date
    df = df[df["target_date_local"].notna()].copy()
    df.sort_values(["station_id", "target_date_local", "retrieved_at_utc"], inplace=True)
    df = df.drop_duplicates(subset=["station_id", "target_date_local"], keep="last")
    out = df[
        [
            "station_id",
            "target_date_local",
            "tmax_f",
            "truth_source",
            "source_record_id",
            "retrieved_at_utc",
        ]
    ].copy()
    out["target_date_local"] = out["target_date_local"].astype(str)
    out["tmax_f"] = pd.to_numeric(out["tmax_f"], errors="coerce").round().astype("Int64")
    out.to_csv(canonical_csv_path, index=False, encoding="utf-8")
    return out


def write_enriched_rows(rows: list[dict[str, Any]], enriched_csv_path: Path) -> pd.DataFrame:
    enriched_csv_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "station_id",
        "station_usw",
        "target_date_local",
        "tmax_f",
        "truth_source",
        "source_record_id",
        "retrieved_at_utc",
        "attribute_measurement_flag",
        "attribute_quality_flag",
        "attribute_source_flag",
        "attribute_obs_time_hhmm",
        "attribute_raw",
        "source_station_field",
    ]
    if not rows:
        df = pd.DataFrame(columns=cols)
        df.to_csv(enriched_csv_path, index=False, encoding="utf-8")
        return df
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df["target_date_local"] = pd.to_datetime(df["target_date_local"], errors="coerce").dt.date
    df = df[df["target_date_local"].notna()].copy()
    df.sort_values(["station_id", "target_date_local", "retrieved_at_utc"], inplace=True)
    df.to_csv(enriched_csv_path, index=False, encoding="utf-8")
    return df


def write_klga_training_truth(
    *,
    canonical_df: pd.DataFrame,
    station_id: str,
    out_csv_path: Path,
) -> pd.DataFrame:
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    station_norm = station_id.strip().upper()
    df = canonical_df[canonical_df["station_id"].astype(str).str.upper() == station_norm].copy()
    if df.empty:
        empty = pd.DataFrame(columns=["request_location_id", "target_date_local", "max_temp_f", "station_zoneid"])
        empty.to_csv(out_csv_path, index=False, encoding="utf-8")
        return empty
    out = pd.DataFrame(
        {
            "request_location_id": f"{station_norm}:9:US",
            "target_date_local": pd.to_datetime(df["target_date_local"], errors="coerce").dt.date.astype(str),
            "max_temp_f": pd.to_numeric(df["tmax_f"], errors="coerce").round().astype("Int64"),
            "station_zoneid": "America/New_York",
        }
    )
    out = out[out["target_date_local"].notna() & out["max_temp_f"].notna()].copy()
    out.sort_values("target_date_local", inplace=True)
    out.to_csv(out_csv_path, index=False, encoding="utf-8")
    return out

