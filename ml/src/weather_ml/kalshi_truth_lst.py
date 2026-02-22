"""Build LST-aligned daily truth from hourly observations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from .mos_utils import sha256_hex, utc_now


@dataclass(frozen=True)
class LstTruthConfig:
    station_id: str
    station_zoneid: str
    climate_day_utc_offset_hours: int = -5
    source_name: str = "HOURLY_LST"
    source_station: str = "KMIA"


def ensure_lst_truth_table(engine: Engine) -> None:
    sql = """
        CREATE TABLE IF NOT EXISTS weather_predictionmarkets.station_daily_truth_lst (
          station_id           VARCHAR(8)  NOT NULL,
          station_zoneid       VARCHAR(64) NOT NULL,
          date_local           DATE        NOT NULL,
          tmax_f               DECIMAL(6,2) NULL,
          tmin_f               DECIMAL(6,2) NULL,
          source_name          VARCHAR(32) NOT NULL,
          source_station       VARCHAR(16) NOT NULL,
          source_hash          CHAR(64)    NOT NULL,
          retrieved_at_utc     TIMESTAMP   NOT NULL,
          PRIMARY KEY (station_id, date_local),
          KEY idx_station_date (station_id, date_local)
        ) ENGINE=InnoDB
    """
    with engine.begin() as conn:
        conn.execute(text(sql))


def compute_lst_daily_from_hourly(
    hourly: pd.DataFrame,
    *,
    timestamp_col: str,
    temp_col: str,
    cfg: LstTruthConfig,
) -> pd.DataFrame:
    if hourly.empty:
        return pd.DataFrame()
    df = hourly.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")
    offset = int(cfg.climate_day_utc_offset_hours)
    df["lst_date"] = (df[timestamp_col] + pd.to_timedelta(offset, unit="h")).dt.date
    grouped = df.groupby("lst_date")[temp_col]
    out = pd.DataFrame(
        {
            "date_local": grouped.max().index,
            "tmax_f": grouped.max().values,
            "tmin_f": grouped.min().values,
        }
    )
    out["station_id"] = cfg.station_id
    out["station_zoneid"] = cfg.station_zoneid
    out["source_name"] = cfg.source_name
    out["source_station"] = cfg.source_station
    source_hash = sha256_hex(
        f"{cfg.station_id}|{cfg.source_station}|{timestamp_col}|{temp_col}|{offset}"
    )
    out["source_hash"] = source_hash
    out["retrieved_at_utc"] = utc_now()
    return out


def upsert_lst_truth(engine: Engine, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    sql = """
        INSERT INTO weather_predictionmarkets.station_daily_truth_lst
        (station_id, station_zoneid, date_local, tmax_f, tmin_f, source_name, source_station,
         source_hash, retrieved_at_utc)
        VALUES
        (:station_id, :station_zoneid, :date_local, :tmax_f, :tmin_f, :source_name, :source_station,
         :source_hash, :retrieved_at_utc)
        ON DUPLICATE KEY UPDATE
            station_zoneid=VALUES(station_zoneid),
            tmax_f=VALUES(tmax_f),
            tmin_f=VALUES(tmin_f),
            source_name=VALUES(source_name),
            source_station=VALUES(source_station),
            source_hash=VALUES(source_hash),
            retrieved_at_utc=VALUES(retrieved_at_utc)
    """
    with engine.begin() as conn:
        conn.execute(text(sql), rows)
