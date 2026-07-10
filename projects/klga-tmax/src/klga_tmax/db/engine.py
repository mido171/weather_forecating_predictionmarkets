from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from klga_tmax.config import load_settings


def make_engine(database_url: str | None = None) -> Engine:
    url = database_url or load_settings(require_db=True).database_url
    if url is None:
        raise RuntimeError("database URL was unexpectedly absent")
    return create_engine(url, future=True, pool_pre_ping=True)
