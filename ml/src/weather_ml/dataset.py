"""Dataset extraction and validation for Epic #2."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import sklearn
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

FEATURE_COLUMNS = list(MODEL_COLUMN_MAP.values())
LABEL_COLUMN = "cli_tmax_f"
MISSING_STRATEGIES = {"drop", "per_model_mean", "station_climatology"}

REQUIRED_COLUMNS = [
    "station_id",
    "target_date_local",
    "asof_policy_id",
    "gfs_mos_tmax_f",
    "mex_tmax_f",
    "nam_mos_tmax_f",
    "nbs_tmax_f",
    "nbe_tmax_f",
    LABEL_COLUMN,
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


@dataclass(frozen=True)
class DatasetSnapshotResult:
    dataset_id: str
    dataset_dir: Path
    data_path: Path
    metadata_path: Path
    parquet_sha256: str
    metadata: dict


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


def build_dataset_snapshot(
    stations: Sequence[str],
    start_date: date | datetime | str,
    end_date: date | datetime | str,
    asof_policy_id: int,
    *,
    missing_strategy: str = "drop",
    datasets_dir: str | Path = "datasets",
    engine: Engine | None = None,
    db_url: str | None = None,
    force: bool = False,
) -> DatasetSnapshotResult:
    """Build a dataset snapshot and write parquet + metadata to a versioned folder."""
    station_list = sorted(_normalize_stations(stations))
    start = _coerce_date(start_date, label="start_date")
    end = _coerce_date(end_date, label="end_date")
    if start > end:
        raise ValueError(f"start_date {start} is after end_date {end}.")
    if asof_policy_id is None:
        raise ValueError("asof_policy_id is required.")
    if missing_strategy not in MISSING_STRATEGIES:
        raise ValueError(
            f"missing_strategy must be one of {sorted(MISSING_STRATEGIES)}."
        )

    resolved_engine = engine or db_module.get_engine(db_url)
    db_snapshot_signature = _get_db_snapshot_signature(
        resolved_engine, station_list, start, end, asof_policy_id
    )
    dataset_id = _compute_dataset_id(
        stations=station_list,
        start_date=start,
        end_date=end,
        asof_policy_id=asof_policy_id,
        feature_columns=FEATURE_COLUMNS,
        missing_strategy=missing_strategy,
        db_snapshot_signature=db_snapshot_signature,
    )

    dataset_dir = Path(datasets_dir) / dataset_id
    data_path = dataset_dir / "data.parquet"
    metadata_path = dataset_dir / "metadata.json"

    if not force and data_path.exists() and metadata_path.exists():
        metadata = _load_metadata(metadata_path)
        parquet_sha256 = metadata.get("parquet_sha256") or _hash_file(data_path)
        return DatasetSnapshotResult(
            dataset_id=dataset_id,
            dataset_dir=dataset_dir,
            data_path=data_path,
            metadata_path=metadata_path,
            parquet_sha256=parquet_sha256,
            metadata=metadata,
        )

    dataset_dir.mkdir(parents=True, exist_ok=True)

    leakage_summary = _validate_no_leakage(
        resolved_engine, station_list, start, end, asof_policy_id
    )
    dataset = _fetch_dataset(resolved_engine, station_list, start, end, asof_policy_id)
    _validate_dataset(dataset, station_list, start, end, asof_policy_id)
    dataset = dataset[REQUIRED_COLUMNS]

    raw_row_count = len(dataset)
    missing_fraction = _missing_fraction(dataset, FEATURE_COLUMNS)
    label_missing_count = int(dataset[LABEL_COLUMN].isna().sum())
    if label_missing_count:
        dataset = dataset.dropna(subset=[LABEL_COLUMN])

    dataset, missing_strategy_details = _apply_missing_strategy(
        dataset, missing_strategy, FEATURE_COLUMNS
    )
    _log_dataset_stats(dataset)

    dataset.to_parquet(data_path, index=False, engine="pyarrow")
    parquet_sha256 = _hash_file(data_path)

    metadata = {
        "created_at_utc": _utc_now_iso(),
        "dataset_id": dataset_id,
        "query": {
            "template": DATASET_SQL.strip(),
            "parameters": {
                "stations": station_list,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "asof_policy_id": asof_policy_id,
            },
        },
        "row_count": len(dataset),
        "row_count_raw": raw_row_count,
        "missing_fraction_by_feature": missing_fraction,
        "missing_strategy": missing_strategy,
        "missing_strategy_details": missing_strategy_details,
        "label_missing_count": label_missing_count,
        "leakage_check": leakage_summary,
        "db_snapshot_signature": db_snapshot_signature,
        "feature_columns": FEATURE_COLUMNS,
        "label_column": LABEL_COLUMN,
        "parquet_sha256": parquet_sha256,
        "library_versions": _get_library_versions(),
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )

    return DatasetSnapshotResult(
        dataset_id=dataset_id,
        dataset_dir=dataset_dir,
        data_path=data_path,
        metadata_path=metadata_path,
        parquet_sha256=parquet_sha256,
        metadata=metadata,
    )


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
) -> dict:
    summary = _get_leakage_summary(engine, stations, start_date, end_date, asof_policy_id)
    violation_count = summary["violation_count"]
    if violation_count == 0:
        return summary

    raise ValueError(
        "Leakage check failed: "
        f"{violation_count} rows have chosen_runtime_utc > asof_utc. "
        f"Sample rows: {summary['sample_records']}"
    )


def _get_leakage_summary(
    engine: Engine,
    stations: Sequence[str],
    start_date: date,
    end_date: date,
    asof_policy_id: int,
) -> dict:
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
    sample_records: list[dict] = []
    if violation_count:
        sample_sql = text(
            "SELECT station_id, target_date_local, model, asof_utc, chosen_runtime_utc "
            f"FROM mos_asof_feature {LEAKAGE_FILTER_SQL} "
            "ORDER BY chosen_runtime_utc DESC LIMIT 10"
        ).bindparams(bindparam("stations", expanding=True))
        sample_df = db_module.read_dataframe(engine, sample_sql, params=params)
        sample_records = sample_df.to_dict(orient="records")

    return {
        "violation_count": violation_count,
        "sample_records": sample_records,
    }


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


def _missing_fraction(dataset: pd.DataFrame, columns: Sequence[str]) -> dict:
    if dataset.empty:
        return {column: 0.0 for column in columns}
    return {
        column: float(dataset[column].isna().mean())
        for column in columns
    }


def _apply_missing_strategy(
    dataset: pd.DataFrame,
    strategy: str,
    feature_columns: Sequence[str],
) -> tuple[pd.DataFrame, dict]:
    if dataset.empty:
        return dataset, {"strategy": strategy, "note": "dataset_empty"}

    if strategy == "drop":
        before = len(dataset)
        dataset = dataset.dropna(subset=feature_columns)
        return dataset, {"strategy": strategy, "dropped_rows": before - len(dataset)}

    if strategy == "per_model_mean":
        means: dict[str, float] = {}
        for column in feature_columns:
            mean_value = dataset[column].mean()
            if pd.isna(mean_value):
                raise ValueError(
                    f"Cannot impute {column}: no non-null values for per_model_mean."
                )
            means[column] = float(mean_value)
        dataset = dataset.copy()
        dataset[feature_columns] = dataset[feature_columns].fillna(value=means)
        return dataset, {"strategy": strategy, "imputed_values": means}

    if strategy == "station_climatology":
        dataset = dataset.copy()
        global_means: dict[str, float] = {}
        for column in feature_columns:
            station_means = dataset.groupby("station_id")[column].transform("mean")
            dataset[column] = dataset[column].fillna(station_means)
            global_mean = dataset[column].mean()
            if pd.isna(global_mean) and dataset[column].isna().any():
                raise ValueError(
                    f"Cannot impute {column}: no non-null values for station_climatology."
                )
            if not pd.isna(global_mean):
                dataset[column] = dataset[column].fillna(global_mean)
                global_means[column] = float(global_mean)
        return dataset, {
            "strategy": strategy,
            "global_means": global_means,
        }

    raise ValueError(f"Unsupported missing_strategy: {strategy}")


def _get_db_snapshot_signature(
    engine: Engine,
    stations: Sequence[str],
    start_date: date,
    end_date: date,
    asof_policy_id: int,
) -> dict:
    params = {
        "stations": list(stations),
        "start_date": start_date,
        "end_date": end_date,
        "asof_policy_id": asof_policy_id,
    }
    mos_statement = text(
        """
        SELECT
            MAX(retrieved_at_utc) AS max_retrieved_at_utc,
            COUNT(*) AS row_count
        FROM mos_asof_feature
        WHERE asof_policy_id = :asof_policy_id
          AND station_id IN :stations
          AND target_date_local BETWEEN :start_date AND :end_date
        """
    ).bindparams(bindparam("stations", expanding=True))
    cli_statement = text(
        """
        SELECT
            MAX(retrieved_at_utc) AS max_retrieved_at_utc,
            COUNT(*) AS row_count
        FROM cli_daily
        WHERE station_id IN :stations
          AND target_date_local BETWEEN :start_date AND :end_date
        """
    ).bindparams(bindparam("stations", expanding=True))

    mos_df = db_module.read_dataframe(engine, mos_statement, params=params)
    cli_df = db_module.read_dataframe(engine, cli_statement, params=params)

    return {
        "mos_asof_feature": _format_snapshot_row(mos_df),
        "cli_daily": _format_snapshot_row(cli_df),
    }


def _format_snapshot_row(snapshot_df: pd.DataFrame) -> dict:
    if snapshot_df.empty:
        return {"max_retrieved_at_utc": None, "row_count": 0}
    max_value = snapshot_df["max_retrieved_at_utc"].iloc[0]
    row_count = int(snapshot_df["row_count"].iloc[0]) if "row_count" in snapshot_df else 0
    return {
        "max_retrieved_at_utc": _format_timestamp(max_value),
        "row_count": row_count,
    }


def _format_timestamp(value: object) -> str | None:
    if value is None:
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone.utc)
    else:
        timestamp = timestamp.tz_convert(timezone.utc)
    return timestamp.isoformat().replace("+00:00", "Z")


def _compute_dataset_id(
    *,
    stations: Sequence[str],
    start_date: date,
    end_date: date,
    asof_policy_id: int,
    feature_columns: Sequence[str],
    missing_strategy: str,
    db_snapshot_signature: dict,
) -> str:
    payload = {
        "stations": list(stations),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "asof_policy_id": asof_policy_id,
        "feature_columns": list(feature_columns),
        "missing_strategy": missing_strategy,
        "db_snapshot_signature": db_snapshot_signature,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _get_library_versions() -> dict:
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
    }


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_metadata(metadata_path: Path) -> dict:
    return json.loads(metadata_path.read_text(encoding="utf-8"))
