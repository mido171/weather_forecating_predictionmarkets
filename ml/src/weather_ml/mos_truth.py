"""IEM daily truth ingestion for station_daily_truth."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from io import StringIO
from typing import Iterable

import pandas as pd
import requests
from sqlalchemy import text
from sqlalchemy.engine import Engine

from .mos_utils import sha256_hex, utc_now


IEM_DAILY_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py"


@dataclass(frozen=True)
class IemDailyConfig:
    station_id: str
    station_zoneid: str
    source_network: str
    source_station: str
    source_name: str = "IEM_DAILY"


def ensure_truth_table(engine: Engine) -> None:
    sql = """
        CREATE TABLE IF NOT EXISTS weather_predictionmarkets.station_daily_truth (
          station_id           VARCHAR(8)  NOT NULL,
          station_zoneid       VARCHAR(64) NOT NULL,
          date_local           DATE        NOT NULL,
          tmax_f               DECIMAL(6,2) NULL,
          tmin_f               DECIMAL(6,2) NULL,
          source_name          VARCHAR(32) NOT NULL,
          source_network       VARCHAR(32) NOT NULL,
          source_station       VARCHAR(16) NOT NULL,
          source_query_hash    CHAR(64)    NOT NULL,
          retrieved_at_utc     TIMESTAMP   NOT NULL,
          PRIMARY KEY (station_id, date_local),
          KEY idx_station_date (station_id, date_local)
        ) ENGINE=InnoDB
    """
    with engine.begin() as conn:
        conn.execute(text(sql))


def fetch_iem_daily(
    cfg: IemDailyConfig,
    start_date: date,
    end_date: date,
) -> tuple[pd.DataFrame, str]:
    params = {
        "sts": start_date.isoformat(),
        "ets": end_date.isoformat(),
        "network": cfg.source_network,
        "stations": cfg.source_station,
        "var": "max_temp_f,min_temp_f",
        "format": "csv",
        "na": "M",
    }
    # deterministic query hash
    param_str = "&".join([f"{key}={params[key]}" for key in sorted(params)])
    query_hash = sha256_hex(f"{IEM_DAILY_URL}?{param_str}")
    response = requests.get(IEM_DAILY_URL, params=params, timeout=60)
    response.raise_for_status()
    raw_text = response.text
    df = pd.read_csv(StringIO(raw_text))
    if df.empty:
        return df, query_hash
    # normalize columns
    rename_map = {}
    if "date" in df.columns:
        rename_map["date"] = "date_local"
    if "day" in df.columns:
        rename_map["day"] = "date_local"
    if "max_temp_f" in df.columns:
        rename_map["max_temp_f"] = "tmax_f"
    if "min_temp_f" in df.columns:
        rename_map["min_temp_f"] = "tmin_f"
    df = df.rename(columns=rename_map)
    df["date_local"] = pd.to_datetime(df["date_local"]).dt.date
    df["tmax_f"] = pd.to_numeric(df.get("tmax_f"), errors="coerce")
    df["tmin_f"] = pd.to_numeric(df.get("tmin_f"), errors="coerce")
    df["station_id"] = cfg.station_id
    df["station_zoneid"] = cfg.station_zoneid
    df["source_name"] = cfg.source_name
    df["source_network"] = cfg.source_network
    df["source_station"] = cfg.source_station
    df["source_query_hash"] = query_hash
    df["retrieved_at_utc"] = utc_now()
    return df, query_hash


def upsert_truth(engine: Engine, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    sql = """
        INSERT INTO weather_predictionmarkets.station_daily_truth
        (station_id, station_zoneid, date_local, tmax_f, tmin_f, source_name, source_network,
         source_station, source_query_hash, retrieved_at_utc)
        VALUES
        (:station_id, :station_zoneid, :date_local, :tmax_f, :tmin_f, :source_name, :source_network,
         :source_station, :source_query_hash, :retrieved_at_utc)
        ON DUPLICATE KEY UPDATE
            station_zoneid=VALUES(station_zoneid),
            tmax_f=VALUES(tmax_f),
            tmin_f=VALUES(tmin_f),
            source_name=VALUES(source_name),
            source_network=VALUES(source_network),
            source_station=VALUES(source_station),
            source_query_hash=VALUES(source_query_hash),
            retrieved_at_utc=VALUES(retrieved_at_utc)
    """
    with engine.begin() as conn:
        conn.execute(text(sql), rows)


def ingest_iem_daily(
    engine: Engine,
    cfg: IemDailyConfig,
    start_date: date,
    end_date: date,
) -> dict[str, object]:
    ensure_truth_table(engine)
    df, query_hash = fetch_iem_daily(cfg, start_date, end_date)
    if df.empty:
        return {
            "row_count": 0,
            "query_hash": query_hash,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    upsert_truth(engine, df.to_dict(orient="records"))
    return {
        "row_count": len(df),
        "query_hash": query_hash,
        "retrieved_at_utc": df["retrieved_at_utc"].iloc[0].isoformat(),
    }
