"""Database access helpers for the ML pipeline."""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

LOGGER = logging.getLogger(__name__)
_DB_URL_ENV = "WEATHER_ML_DB_URL"
_ENGINE: Engine | None = None


def configure_engine(db_url: str, *, pool_pre_ping: bool = True) -> Engine:
    """Configure and cache a SQLAlchemy engine for reuse."""
    if not db_url:
        raise ValueError("db_url is required to configure the database engine.")
    global _ENGINE
    _ENGINE = create_engine(db_url, pool_pre_ping=pool_pre_ping)
    return _ENGINE


def get_engine(db_url: str | None = None) -> Engine:
    """Return a cached engine, configuring one from the env or provided URL."""
    if db_url:
        return configure_engine(db_url)
    if _ENGINE is not None:
        return _ENGINE
    env_url = os.getenv(_DB_URL_ENV)
    if not env_url:
        raise ValueError(
            "Database URL not configured. Set WEATHER_ML_DB_URL or call configure_engine."
        )
    return configure_engine(env_url)


def read_dataframe(
    engine: Engine, statement, params: Mapping[str, Any] | None = None
) -> pd.DataFrame:
    """Read a SQL query into a pandas DataFrame."""
    return pd.read_sql(statement, engine, params=params)
