"""Tests for dataset snapshot builder."""

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


def _insert_cli_daily(
    conn, station_id: str, target_date: str, tmax_f: float, retrieved_at_utc: str
) -> None:
    conn.execute(
        text(
            """
            INSERT INTO cli_daily (
                station_id,
                target_date_local,
                tmax_f,
                retrieved_at_utc
            )
            VALUES (
                :station_id,
                :target_date_local,
                :tmax_f,
                :retrieved_at_utc
            )
            """
        ),
        {
            "station_id": station_id,
            "target_date_local": target_date,
            "tmax_f": tmax_f,
            "retrieved_at_utc": retrieved_at_utc,
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
    retrieved_at_utc: str,
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
                tmax_f,
                retrieved_at_utc
            )
            VALUES (
                :station_id,
                :target_date_local,
                :asof_policy_id,
                :model,
                :asof_utc,
                :chosen_runtime_utc,
                :tmax_f,
                :retrieved_at_utc
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
            "retrieved_at_utc": retrieved_at_utc,
        },
    )


def _seed_data(engine) -> None:
    with engine.begin() as conn:
        _insert_cli_daily(conn, "KMIA", "2024-01-01", 85.0, "2024-01-02 00:00:00")
        _insert_cli_daily(conn, "KMIA", "2024-01-02", 86.0, "2024-01-03 00:00:00")

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
                retrieved_at_utc="2024-01-01 23:10:00",
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
                retrieved_at_utc="2024-01-02 23:10:00",
            )


def test_snapshot_is_deterministic(tmp_path) -> None:
    engine = _build_engine()
    _seed_data(engine)

    result_one = dataset.build_dataset_snapshot(
        ["KMIA"],
        date(2024, 1, 1),
        date(2024, 1, 2),
        1,
        missing_strategy="drop",
        datasets_dir=tmp_path,
        engine=engine,
    )
    result_two = dataset.build_dataset_snapshot(
        ["KMIA"],
        date(2024, 1, 1),
        date(2024, 1, 2),
        1,
        missing_strategy="drop",
        datasets_dir=tmp_path,
        engine=engine,
    )

    assert result_one.dataset_id == result_two.dataset_id
    assert result_one.parquet_sha256 == result_two.parquet_sha256
    assert result_one.metadata["row_count_raw"] == 2
    assert result_one.metadata["row_count"] == 1
    assert pytest.approx(result_one.metadata["missing_fraction_by_feature"]["mex_tmax_f"]) == 0.5


def test_snapshot_id_changes_on_db_update(tmp_path) -> None:
    engine = _build_engine()
    _seed_data(engine)

    first = dataset.build_dataset_snapshot(
        ["KMIA"],
        date(2024, 1, 1),
        date(2024, 1, 2),
        1,
        missing_strategy="drop",
        datasets_dir=tmp_path / "first",
        engine=engine,
    )

    with engine.begin() as conn:
        _insert_mos_feature(
            conn,
            station_id="KMIA",
            target_date="2024-01-02",
            model="MEX",
            tmax_f=85.0,
            asof_utc="2024-01-02 23:00:00",
            chosen_runtime_utc="2024-01-02 12:00:00",
            retrieved_at_utc="2024-01-04 00:00:00",
        )

    second = dataset.build_dataset_snapshot(
        ["KMIA"],
        date(2024, 1, 1),
        date(2024, 1, 2),
        1,
        missing_strategy="drop",
        datasets_dir=tmp_path / "second",
        engine=engine,
    )

    assert first.dataset_id != second.dataset_id


def test_snapshot_imputes_missing_values(tmp_path) -> None:
    engine = _build_engine()
    _seed_data(engine)

    result = dataset.build_dataset_snapshot(
        ["KMIA"],
        date(2024, 1, 1),
        date(2024, 1, 2),
        1,
        missing_strategy="per_model_mean",
        datasets_dir=tmp_path,
        engine=engine,
    )

    assert result.metadata["row_count"] == 2
    df = pd.read_parquet(result.data_path, engine="pyarrow")
    assert df[dataset.FEATURE_COLUMNS].isna().sum().sum() == 0
