"""DB-backed dataset assembly for TFS2 sweep."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from weather_ml.mos_config import MosDatasetConfig
from weather_ml.mos_mos_features import (
    fetch_mos_rows,
    select_latest_mos,
    build_mos_pivots,
)
from weather_ml.mos_utils import normalize_sql, sha256_hex

from .config import ASOF_HOUR_UTC, MOS_CODES_BASE

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetRef:
    station_id: str
    start_date: date
    end_date: date
    asof_hour_utc: int
    grib_sql_hash: str
    mos_sql_hash: str
    truth_sql_hash: str


@dataclass(frozen=True)
class DatasetBundle:
    df: pd.DataFrame
    dataset_ref: DatasetRef


def default_mysql_url() -> str:
    host = os.getenv("MYSQL_HOST", "localhost")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "")
    db = os.getenv("MYSQL_DB", "weather_predictionmarkets")
    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}"


def create_engine_from_url(url: str | None = None) -> Engine:
    return create_engine(url or default_mysql_url(), pool_pre_ping=True, pool_recycle=3600)


def _fetch_station_zoneid(engine: Engine, station_id: str) -> str:
    sql = "SELECT zone_id FROM weather_predictionmarkets.station_registry WHERE station_id = :station_id"
    df = pd.read_sql(text(sql), engine, params={"station_id": station_id})
    if df.empty:
        return "UNKNOWN"
    return str(df.iloc[0]["zone_id"])


def _fetch_gribstream(engine: Engine, station_id: str, start_date: date, end_date: date) -> tuple[pd.DataFrame, str]:
    sql = """
        SELECT station_id, target_date_local, asof_utc, model_code, metric, value_f, retrieved_at_utc, id
        FROM weather_predictionmarkets.gribstream_daily_feature
        WHERE station_id = :station_id
          AND target_date_local BETWEEN :start_date AND :end_date
          AND HOUR(asof_utc) = :asof_hour
    """
    params = {
        "station_id": station_id,
        "start_date": start_date,
        "end_date": end_date,
        "asof_hour": ASOF_HOUR_UTC,
    }
    sql_hash = sha256_hex(normalize_sql(sql))
    df = pd.read_sql(text(sql), engine, params=params)
    if df.empty:
        return df, sql_hash
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    df["asof_utc"] = pd.to_datetime(df["asof_utc"], utc=True)
    df["retrieved_at_utc"] = pd.to_datetime(df["retrieved_at_utc"], utc=True, errors="coerce")
    df["model_code"] = df["model_code"].astype(str).str.lower()
    df["metric"] = df["metric"].astype(str).str.upper()
    df["value_f"] = pd.to_numeric(df.get("value_f"), errors="coerce")

    # enforce as-of cutoff: T-1 12Z
    cutoff = df["target_date_local"].apply(lambda d: datetime(d.year, d.month, d.day, ASOF_HOUR_UTC, tzinfo=timezone.utc) - timedelta(days=1))
    df = df[df["asof_utc"] == pd.to_datetime(cutoff)]

    if df.empty:
        return df, sql_hash

    # dedupe
    df = df.sort_values(["retrieved_at_utc", "id"], ascending=[True, True])
    before = len(df)
    df = df.drop_duplicates(
        subset=["station_id", "target_date_local", "asof_utc", "model_code", "metric"],
        keep="last",
    )
    dropped = before - len(df)
    if dropped:
        LOGGER.warning("Dropped %d duplicate gribstream rows.", dropped)

    # map to feature names
    def to_feature(row: pd.Series) -> str | None:
        metric = row["metric"]
        model = row["model_code"]
        if metric == "TMAX_F":
            return f"{model}_tmax_f"
        if metric == "TMP_SPREAD_F" and model == "gefsatmos":
            return "gefsatmos_tmp_spread_f"
        return None

    df["feature_name"] = df.apply(to_feature, axis=1)
    df = df[df["feature_name"].notna()].copy()
    if df.empty:
        return df, sql_hash

    wide = df.pivot_table(
        index=["station_id", "target_date_local", "asof_utc"],
        columns="feature_name",
        values="value_f",
        aggfunc="last",
    ).reset_index()
    wide.columns = [str(c) for c in wide.columns]
    return wide, sql_hash


def _fetch_truth(engine: Engine, station_id: str, start_date: date, end_date: date) -> tuple[pd.DataFrame, str]:
    sql_live = """
        SELECT station_id, target_date_local, tmax_f
        FROM weather_predictionmarkets.live_truth_cli
        WHERE station_id = :station_id
          AND target_date_local BETWEEN :start_date AND :end_date
    """
    params = {
        "station_id": station_id,
        "start_date": start_date,
        "end_date": end_date,
    }
    sql_hash = sha256_hex(normalize_sql(sql_live))
    try:
        df = pd.read_sql(text(sql_live), engine, params=params)
        if df.empty:
            return df, sql_hash
    except Exception:
        sql_cli = """
            SELECT station_id, target_date_local, tmax_f
            FROM weather_predictionmarkets.cli_daily
            WHERE station_id = :station_id
              AND target_date_local BETWEEN :start_date AND :end_date
        """
        sql_hash = sha256_hex(normalize_sql(sql_cli))
        df = pd.read_sql(text(sql_cli), engine, params=params)
        if df.empty:
            return df, sql_hash
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    df["tmax_f"] = pd.to_numeric(df.get("tmax_f"), errors="coerce")
    return df, sql_hash


def _fetch_mos(
    engine: Engine,
    station_id: str,
    start_date: date,
    end_date: date,
    cal_df: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    zone_id = _fetch_station_zoneid(engine, station_id)
    cfg = MosDatasetConfig(
        station_id=station_id,
        station_zoneid=zone_id,
        feature_version="tfs2",
        build_start_asof=start_date,
        output_start_asof=start_date,
        end_asof=end_date,
        include_retrieved_at_guard=False,
        models=["GFS", "NAM"],
        variables=MOS_CODES_BASE,
    ).normalized()
    mos_df, sql_hash = fetch_mos_rows(engine, cfg, start_date, end_date)
    if mos_df.empty:
        return pd.DataFrame(), sql_hash
    latest = select_latest_mos(mos_df, cal_df, cfg)
    piv = build_mos_pivots(latest)
    if piv.empty:
        return piv, sql_hash
    return piv, sql_hash


def build_dataset(
    engine: Engine,
    station_id: str,
    start_date: date,
    end_date: date,
) -> DatasetBundle:
    LOGGER.info("DATA_BUILD_START station=%s start=%s end=%s", station_id, start_date, end_date)
    LOGGER.info("DATA_FETCH_GRIBSTREAM_START")
    grib_df, grib_hash = _fetch_gribstream(engine, station_id, start_date, end_date)
    if grib_df.empty:
        raise ValueError("No gribstream rows found for station/date range.")
    LOGGER.info("DATA_FETCH_GRIBSTREAM_DONE rows=%d", len(grib_df))

    grib_df = grib_df.copy()
    grib_df["asof_utc"] = pd.to_datetime(grib_df["asof_utc"], utc=True)
    cal_df = grib_df[["target_date_local", "asof_utc"]].copy()

    LOGGER.info("DATA_FETCH_MOS_START")
    mos_df, mos_hash = _fetch_mos(engine, station_id, start_date, end_date, cal_df)
    if mos_df.empty:
        LOGGER.warning("DATA_FETCH_MOS_EMPTY")
        mos_df = cal_df[["target_date_local"]].drop_duplicates().copy()
    else:
        LOGGER.info("DATA_FETCH_MOS_DONE rows=%d", len(mos_df))
    LOGGER.info("DATA_FETCH_TRUTH_START")
    truth_df, truth_hash = _fetch_truth(engine, station_id, start_date, end_date)
    LOGGER.info("DATA_FETCH_TRUTH_DONE rows=%d", len(truth_df))

    df = grib_df.merge(mos_df, on="target_date_local", how="left")
    df = df.merge(truth_df, on=["station_id", "target_date_local"], how="left")
    if "tmax_f" in df.columns and "actual_tmax_f" not in df.columns:
        df = df.rename(columns={"tmax_f": "actual_tmax_f"})

    # ensure unique key
    before = len(df)
    df = df.drop_duplicates(subset=["station_id", "target_date_local", "asof_utc"], keep="last")
    dropped = before - len(df)
    if dropped:
        LOGGER.warning("Dropped %d duplicate dataset rows.", dropped)

    df = df.sort_values(["target_date_local", "asof_utc"]).reset_index(drop=True)
    df["asof_date_local"] = pd.to_datetime(df["asof_utc"], utc=True).dt.date
    LOGGER.info("DATA_BUILD_DONE rows=%d cols=%d", len(df), df.shape[1])

    dataset_ref = DatasetRef(
        station_id=station_id,
        start_date=start_date,
        end_date=end_date,
        asof_hour_utc=ASOF_HOUR_UTC,
        grib_sql_hash=grib_hash,
        mos_sql_hash=mos_hash,
        truth_sql_hash=truth_hash,
    )
    return DatasetBundle(df=df, dataset_ref=dataset_ref)
