"""Dataset extraction and validation for Epic #2."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Iterable, Sequence

import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

from weather_ml import db as db_module

LOGGER = logging.getLogger(__name__)

MODEL_COLUMN_MAP = {
    "GFS": "gfs_mos_tmax_f",
    "MEX": "mex_tmax_f",
    "NAM": "nam_mos_tmax_f",
    "NBS": "nbs_tmax_f",
    "NBE": "nbe_tmax_f",
}

REQUIRED_COLUMNS = [
    "station_id",
    "target_date_local",
    "asof_policy_id",
    "gfs_mos_tmax_f",
    "mex_tmax_f",
    "nam_mos_tmax_f",
    "nbs_tmax_f",
    "nbe_tmax_f",
    "cli_tmax_f",
]

LEAKAGE_FILTER_SQL = """
WHERE asof_policy_id = :asof_policy_id
  AND station_id IN :stations
  AND target_date_local BETWEEN :start_date AND :end_date
  AND chosen_runtime_utc IS NOT NULL
  AND asof_utc IS NOT NULL
  AND chosen_runtime_utc > asof_utc
"""

DATASET_SQL = """
SELECT
  f.station_id,
  f.target_date_local,
  f.asof_policy_id,
  MAX(CASE WHEN f.model = 'GFS' THEN f.tmax_f END) AS gfs_mos_tmax_f,
  MAX(CASE WHEN f.model = 'MEX' THEN f.tmax_f END) AS mex_tmax_f,
  MAX(CASE WHEN f.model = 'NAM' THEN f.tmax_f END) AS nam_mos_tmax_f,
  MAX(CASE WHEN f.model = 'NBS' THEN f.tmax_f END) AS nbs_tmax_f,
  MAX(CASE WHEN f.model = 'NBE' THEN f.tmax_f END) AS nbe_tmax_f,
  c.tmax_f AS cli_tmax_f
FROM mos_asof_feature f
JOIN cli_daily c
  ON c.station_id = f.station_id
 AND c.target_date_local = f.target_date_local
WHERE f.asof_policy_id = :asof_policy_id
  AND f.station_id IN :stations
  AND f.target_date_local BETWEEN :start_date AND :end_date
GROUP BY f.station_id, f.target_date_local, f.asof_policy_id, c.tmax_f
ORDER BY f.station_id, f.target_date_local
"""


def build_dataset(
    stations: Sequence[str],
    start_date: date | datetime | str,
    end_date: date | datetime | str,
    asof_policy_id: int,
    *,
    engine: Engine | None = None,
    db_url: str | None = None,
) -> pd.DataFrame:
    """Build the training dataset for the given station/date range and policy."""
    station_list = _normalize_stations(stations)
    start = _coerce_date(start_date, label="start_date")
    end = _coerce_date(end_date, label="end_date")
    if start > end:
        raise ValueError(f"start_date {start} is after end_date {end}.")
    if asof_policy_id is None:
        raise ValueError("asof_policy_id is required.")

    resolved_engine = engine or db_module.get_engine(db_url)

    _validate_no_leakage(resolved_engine, station_list, start, end, asof_policy_id)
    dataset = _fetch_dataset(resolved_engine, station_list, start, end, asof_policy_id)
    _validate_dataset(dataset, station_list, start, end, asof_policy_id)
    dataset = dataset[REQUIRED_COLUMNS]
    _log_dataset_stats(dataset)
    return dataset


def _normalize_stations(stations: Iterable[str]) -> list[str]:
    if isinstance(stations, str):
        station_iterable = [stations]
    else:
        station_iterable = list(stations)
    station_list = [station for station in station_iterable if station]
    if not station_list:
        raise ValueError("stations must be a non-empty list of station ids.")
    return list(dict.fromkeys(station_list))


def _coerce_date(value: date | datetime | str, *, label: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return pd.to_datetime(value).date()
    except Exception as exc:
        raise ValueError(f"{label} must be a date or ISO date string.") from exc


def _validate_no_leakage(
    engine: Engine,
    stations: Sequence[str],
    start_date: date,
    end_date: date,
    asof_policy_id: int,
) -> None:
    params = {
        "stations": list(stations),
        "start_date": start_date,
        "end_date": end_date,
        "asof_policy_id": asof_policy_id,
    }
    count_sql = text(
        f"SELECT COUNT(*) AS violation_count FROM mos_asof_feature {LEAKAGE_FILTER_SQL}"
    ).bindparams(bindparam("stations", expanding=True))
    count_df = db_module.read_dataframe(engine, count_sql, params=params)
    violation_count = int(count_df["violation_count"].iloc[0]) if not count_df.empty else 0
    if violation_count == 0:
        return

    sample_sql = text(
        "SELECT station_id, target_date_local, model, asof_utc, chosen_runtime_utc "
        f"FROM mos_asof_feature {LEAKAGE_FILTER_SQL} "
        "ORDER BY chosen_runtime_utc DESC LIMIT 10"
    ).bindparams(bindparam("stations", expanding=True))
    sample_df = db_module.read_dataframe(engine, sample_sql, params=params)
    sample_records = sample_df.to_dict(orient="records")
    raise ValueError(
        "Leakage check failed: "
        f"{violation_count} rows have chosen_runtime_utc > asof_utc. "
        f"Sample rows: {sample_records}"
    )


def _fetch_dataset(
    engine: Engine,
    stations: Sequence[str],
    start_date: date,
    end_date: date,
    asof_policy_id: int,
) -> pd.DataFrame:
    params = {
        "stations": list(stations),
        "start_date": start_date,
        "end_date": end_date,
        "asof_policy_id": asof_policy_id,
    }
    statement = text(DATASET_SQL).bindparams(bindparam("stations", expanding=True))
    return db_module.read_dataframe(engine, statement, params=params)


def _validate_dataset(
    dataset: pd.DataFrame,
    stations: Sequence[str],
    start_date: date,
    end_date: date,
    asof_policy_id: int,
) -> None:
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in dataset.columns]
    if missing_columns:
        raise ValueError(f"Dataset missing required columns: {missing_columns}")

    if dataset.empty:
        LOGGER.info("Dataset is empty for station range %s.", stations)
        return

    if dataset["station_id"].isna().any():
        raise ValueError("Dataset contains null station_id values.")

    invalid_stations = set(dataset["station_id"]) - set(stations)
    if invalid_stations:
        raise ValueError(f"Dataset contains unexpected stations: {sorted(invalid_stations)}")

    date_series = pd.to_datetime(dataset["target_date_local"], errors="coerce").dt.date
    if date_series.isna().any():
        raise ValueError("Dataset contains invalid target_date_local values.")
    out_of_range = (date_series < start_date) | (date_series > end_date)
    if out_of_range.any():
        bad_rows = dataset.loc[out_of_range, ["station_id", "target_date_local"]].head(10)
        raise ValueError(
            "Dataset contains rows outside requested date range "
            f"{start_date}..{end_date}: {bad_rows.to_dict(orient='records')}"
        )

    if dataset["asof_policy_id"].isna().any():
        raise ValueError("Dataset contains null asof_policy_id values.")

    mismatched_policy = dataset["asof_policy_id"] != asof_policy_id
    if mismatched_policy.any():
        bad_values = sorted(set(dataset.loc[mismatched_policy, "asof_policy_id"]))
        raise ValueError(f"Dataset contains unexpected asof_policy_id values: {bad_values}")


def _log_dataset_stats(dataset: pd.DataFrame) -> None:
    LOGGER.info("Dataset row count: %s", len(dataset))
    missing_by_model = {
        model: int(dataset[column].isna().sum()) for model, column in MODEL_COLUMN_MAP.items()
    }
    LOGGER.info("Missing counts by model: %s", missing_by_model)
