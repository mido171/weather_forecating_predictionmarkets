"""Configuration helpers for the ML pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import yaml


@dataclass(frozen=True)
class RunConfig:
    name: str
    seed: int


@dataclass(frozen=True)
class DbConfig:
    url: str


@dataclass(frozen=True)
class DataConfig:
    stations: list[str]
    start_date_local: date
    end_date_local: date
    asof_policy_id: int
    missing_strategy: str


@dataclass(frozen=True)
class OutputConfig:
    artifacts_dir: str
    datasets_dir: str


@dataclass(frozen=True)
class TrainingConfig:
    run: RunConfig
    db: DbConfig
    data: DataConfig
    output: OutputConfig


def load_config(path: str | Path) -> TrainingConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    run_raw = _require_section(raw, "run")
    db_raw = _require_section(raw, "db")
    data_raw = _require_section(raw, "data")
    output_raw = raw.get("output", {}) or {}

    run = RunConfig(
        name=str(run_raw.get("name", "default")),
        seed=int(run_raw.get("seed", 0)),
    )
    db_url = db_raw.get("url")
    if not db_url:
        raise ValueError("db.url is required in the config.")
    db = DbConfig(url=str(db_url))

    stations = data_raw.get("stations")
    if not stations:
        raise ValueError("data.stations must be provided in the config.")
    if isinstance(stations, str):
        stations_list = [stations]
    else:
        stations_list = list(stations)
    stations_list = [station for station in stations_list if station]
    if not stations_list:
        raise ValueError("data.stations must include at least one station id.")

    start_date = _parse_date(data_raw.get("start_date_local"), "data.start_date_local")
    end_date = _parse_date(data_raw.get("end_date_local"), "data.end_date_local")

    asof_policy_value = data_raw.get("asof_policy_id")
    if asof_policy_value is None:
        raise ValueError("data.asof_policy_id must be provided in the config.")

    missing_strategy = data_raw.get("missing_strategy") or "drop"

    data = DataConfig(
        stations=stations_list,
        start_date_local=start_date,
        end_date_local=end_date,
        asof_policy_id=int(asof_policy_value),
        missing_strategy=str(missing_strategy),
    )

    output = OutputConfig(
        artifacts_dir=str(output_raw.get("artifacts_dir", "artifacts")),
        datasets_dir=str(output_raw.get("datasets_dir", "datasets")),
    )

    return TrainingConfig(run=run, db=db, data=data, output=output)


def _require_section(raw: dict, name: str) -> dict:
    section = raw.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Missing or invalid config section: {name}")
    return section


def _parse_date(value: object, label: str) -> date:
    if value is None:
        raise ValueError(f"{label} is required.")
    try:
        return pd.to_datetime(value).date()
    except Exception as exc:
        raise ValueError(f"{label} must be a date or ISO date string.") from exc
