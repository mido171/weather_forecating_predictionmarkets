"""Database helpers for MOS dataset builder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


@dataclass(frozen=True)
class DbConfig:
    host: str
    port: int
    database: str
    user: str
    password: str


def load_db_config(path: str | Path) -> DbConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"DB config not found: {cfg_path}")
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    password = raw.get("password")
    if not password:
        env_key = raw.get("password_env")
        if env_key:
            password = os.environ.get(env_key)
    if not password:
        raise ValueError("Missing database password (password or password_env).")
    return DbConfig(
        host=str(raw["host"]),
        port=int(raw.get("port", 3306)),
        database=str(raw["database"]),
        user=str(raw["user"]),
        password=str(password),
    )


def load_db_config_from_env(prefix: str = "WEATHER_ML_DB_") -> DbConfig:
    host = os.environ.get(f"{prefix}HOST")
    database = os.environ.get(f"{prefix}DATABASE")
    user = os.environ.get(f"{prefix}USER")
    password = os.environ.get(f"{prefix}PASSWORD")
    if not all([host, database, user, password]):
        raise ValueError("Missing required DB env vars (HOST, DATABASE, USER, PASSWORD).")
    port = int(os.environ.get(f"{prefix}PORT", "3306"))
    return DbConfig(host=host, port=port, database=database, user=user, password=password)


def create_engine_from_config(cfg: DbConfig) -> Engine:
    return create_engine(
        f"mysql+pymysql://{cfg.user}:{cfg.password}@{cfg.host}:{cfg.port}/{cfg.database}",
        pool_pre_ping=True,
        pool_recycle=3600,
    )

