from __future__ import annotations

import os
from datetime import date, datetime
from typing import Iterable

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .config import BANNED_OBS_COLUMNS, OBS_ALLOWED_COLUMNS


def default_mysql_url() -> str:
    host = os.getenv("MYSQL_HOST", "localhost")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "root")
    database = os.getenv("MYSQL_DB", "weather_predictionmarkets")
    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"


def create_engine_from_url(url: str | None = None) -> Engine:
    return create_engine(url or default_mysql_url(), pool_pre_ping=True, pool_recycle=3600)


def _index_exists(
    engine: Engine,
    *,
    schema_name: str,
    table_name: str,
    index_name: str,
) -> bool:
    sql = """
        SELECT 1
        FROM information_schema.statistics
        WHERE table_schema = :schema_name
          AND table_name = :table_name
          AND index_name = :index_name
        LIMIT 1
    """
    with engine.connect() as conn:
        row = conn.execute(
            text(sql),
            {
                "schema_name": schema_name,
                "table_name": table_name,
                "index_name": index_name,
            },
        ).first()
    return row is not None


def ensure_required_indexes(engine: Engine) -> list[str]:
    created: list[str] = []
    specs = [
        (
            "wunderground_ml",
            "wunderground_station_observation_30m",
            "idx_wu_obs30m_request_valid_utc",
            """
            CREATE INDEX idx_wu_obs30m_request_valid_utc
            ON wunderground_ml.wunderground_station_observation_30m (request_location_id, valid_time_utc)
            """,
        ),
        (
            "wunderground_ml",
            "wunderground_station_observation_30m",
            "idx_wu_obs30m_request_valid_utc_temp",
            """
            CREATE INDEX idx_wu_obs30m_request_valid_utc_temp
            ON wunderground_ml.wunderground_station_observation_30m (request_location_id, valid_time_utc, temp)
            """,
        ),
        (
            "wunderground_ml",
            "wunderground_station_daily_max_temperature",
            "idx_wu_daily_request_day",
            """
            CREATE INDEX idx_wu_daily_request_day
            ON wunderground_ml.wunderground_station_daily_max_temperature (request_location_id, target_date_local)
            """,
        ),
    ]
    for schema_name, table_name, index_name, ddl in specs:
        if _index_exists(
            engine,
            schema_name=schema_name,
            table_name=table_name,
            index_name=index_name,
        ):
            continue
        with engine.begin() as conn:
            conn.execute(text(ddl))
        created.append(index_name)
    return created


def fetch_daily_max(
    engine: Engine,
    *,
    request_location_id: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    sql = """
        SELECT
            request_location_id,
            obs_id,
            station_zoneid,
            target_date_local,
            max_temp_f
        FROM wunderground_ml.wunderground_station_daily_max_temperature
        WHERE request_location_id = :request_location_id
          AND target_date_local BETWEEN :start_date AND :end_date
        ORDER BY target_date_local
    """
    df = pd.read_sql(
        text(sql),
        engine,
        params={
            "request_location_id": request_location_id,
            "start_date": start_date,
            "end_date": end_date,
        },
    )
    if df.empty:
        return df
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    df["max_temp_f"] = pd.to_numeric(df["max_temp_f"], errors="coerce").round().astype("Int64")
    return df


def fetch_observations(
    engine: Engine,
    *,
    request_location_ids: Iterable[str],
    start_utc: datetime,
    end_utc: datetime,
    columns: Iterable[str] = OBS_ALLOWED_COLUMNS,
) -> pd.DataFrame:
    colset = tuple(columns)
    banned = BANNED_OBS_COLUMNS.intersection(colset)
    if banned:
        raise ValueError(f"Banned observation columns requested: {sorted(banned)}")
    valid = set(OBS_ALLOWED_COLUMNS)
    unknown = [c for c in colset if c not in valid]
    if unknown:
        raise ValueError(f"Unknown/unsupported observation columns requested: {unknown}")

    location_ids = tuple(request_location_ids)
    if not location_ids:
        return pd.DataFrame(columns=list(colset))

    cols_sql = ", ".join(colset)
    placeholders = ", ".join(f":loc_{i}" for i in range(len(location_ids)))
    sql = f"""
        SELECT {cols_sql}
        FROM wunderground_ml.wunderground_station_observation_30m
        WHERE request_location_id IN ({placeholders})
          AND valid_time_utc >= :start_utc
          AND valid_time_utc <= :end_utc
        ORDER BY request_location_id, valid_time_utc
    """
    params = {"start_utc": start_utc, "end_utc": end_utc}
    params.update({f"loc_{i}": loc for i, loc in enumerate(location_ids)})
    df = pd.read_sql(text(sql), engine, params=params)
    if df.empty:
        return df

    df["valid_time_utc"] = pd.to_datetime(df["valid_time_utc"], utc=True)
    numeric_cols = [
        "temp",
        "dew_pt",
        "rh",
        "pressure",
        "vis",
        "wspd",
        "wdir",
        "gust",
        "precip_hrly",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

