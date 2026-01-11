"""Tests for dataset extraction and leakage checks."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from weather_ml import dataset


def _build_engine():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    _create_schema(engine)
    return engine


def _create_schema(engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE cli_daily (
                    station_id TEXT NOT NULL,
                    target_date_local DATE NOT NULL,
                    tmax_f DECIMAL,
                    retrieved_at_utc TIMESTAMP,
                    raw_payload_hash TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE mos_asof_feature (
                    station_id TEXT NOT NULL,
                    target_date_local DATE NOT NULL,
                    asof_policy_id INTEGER NOT NULL,
                    model TEXT NOT NULL,
                    asof_utc TIMESTAMP,
                    chosen_runtime_utc TIMESTAMP,
                    tmax_f DECIMAL,
                    missing_reason TEXT,
                    retrieved_at_utc TIMESTAMP
                )
                """
            )
        )


def _insert_cli_daily(conn, station_id: str, target_date: str, tmax_f: float) -> None:
    conn.execute(
        text(
            """
            INSERT INTO cli_daily (station_id, target_date_local, tmax_f)
            VALUES (:station_id, :target_date_local, :tmax_f)
            """
        ),
        {
            "station_id": station_id,
            "target_date_local": target_date,
            "tmax_f": tmax_f,
        },
    )


def _insert_mos_feature(
    conn,
    *,
    station_id: str,
    target_date: str,
    model: str,
    tmax_f: float | None,
    asof_utc: str,
    chosen_runtime_utc: str,
    asof_policy_id: int = 1,
) -> None:
    conn.execute(
        text(
            """
            INSERT INTO mos_asof_feature (
                station_id,
                target_date_local,
                asof_policy_id,
                model,
                asof_utc,
                chosen_runtime_utc,
                tmax_f
            )
            VALUES (
                :station_id,
                :target_date_local,
                :asof_policy_id,
                :model,
                :asof_utc,
                :chosen_runtime_utc,
                :tmax_f
            )
            """
        ),
        {
            "station_id": station_id,
            "target_date_local": target_date,
            "asof_policy_id": asof_policy_id,
            "model": model,
            "asof_utc": asof_utc,
            "chosen_runtime_utc": chosen_runtime_utc,
            "tmax_f": tmax_f,
        },
    )


def test_build_dataset_pivot_and_missing() -> None:
    engine = _build_engine()
    with engine.begin() as conn:
        _insert_cli_daily(conn, "KMIA", "2024-01-01", 85.0)
        _insert_cli_daily(conn, "KMIA", "2024-01-02", 86.0)

        for model, value in {
            "GFS": 80.0,
            "MEX": 81.0,
            "NAM": 79.0,
            "NBS": 82.0,
            "NBE": 83.0,
        }.items():
            _insert_mos_feature(
                conn,
                station_id="KMIA",
                target_date="2024-01-01",
                model=model,
                tmax_f=value,
                asof_utc="2024-01-01 23:00:00",
                chosen_runtime_utc="2024-01-01 12:00:00",
            )

        for model, value in {
            "GFS": 81.0,
            "NAM": 80.0,
            "NBS": 83.0,
            "NBE": 84.0,
        }.items():
            _insert_mos_feature(
                conn,
                station_id="KMIA",
                target_date="2024-01-02",
                model=model,
                tmax_f=value,
                asof_utc="2024-01-02 23:00:00",
                chosen_runtime_utc="2024-01-02 12:00:00",
            )

    df = dataset.build_dataset(
        ["KMIA"],
        date(2024, 1, 1),
        date(2024, 1, 2),
        1,
        engine=engine,
    )

    assert len(df) == 2
    assert all(column in df.columns for column in dataset.REQUIRED_COLUMNS)
    assert df["asof_policy_id"].nunique() == 1

    target_dates = pd.to_datetime(df["target_date_local"]).dt.date
    row_day1 = df.loc[target_dates == date(2024, 1, 1)].iloc[0]
    row_day2 = df.loc[target_dates == date(2024, 1, 2)].iloc[0]

    assert float(row_day1["gfs_mos_tmax_f"]) == 80.0
    assert float(row_day1["cli_tmax_f"]) == 85.0
    assert pd.isna(row_day2["mex_tmax_f"])


def test_leakage_check_raises() -> None:
    engine = _build_engine()
    with engine.begin() as conn:
        _insert_cli_daily(conn, "KMIA", "2024-01-03", 87.0)
        _insert_mos_feature(
            conn,
            station_id="KMIA",
            target_date="2024-01-03",
            model="GFS",
            tmax_f=82.0,
            asof_utc="2024-01-03 00:00:00",
            chosen_runtime_utc="2024-01-03 06:00:00",
        )

    with pytest.raises(ValueError, match="Leakage check failed"):
        dataset.build_dataset(
            ["KMIA"],
            date(2024, 1, 3),
            date(2024, 1, 3),
            1,
            engine=engine,
        )
