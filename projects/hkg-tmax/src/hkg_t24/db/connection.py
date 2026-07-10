"""Database connection and DSN policy for HKG-T24."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

from hkg_t24.constants import (
    DATABASE_DSN_ENV,
    DATABASE_URL_ENV,
    DUAL_DSN_WARNING,
    MISSING_DSN_ERROR,
)
from hkg_t24.utils.hashing import sha256_text


class DatabaseConfigError(RuntimeError):
    """Raised when required database configuration is missing."""


class DatabaseUnavailable(RuntimeError):
    """Raised when the PostgreSQL runtime dependency is unavailable."""


def get_database_url(
    environ: Mapping[str, str] | None = None,
    *,
    message_sink: Callable[[str], None] | None = None,
) -> str:
    """Return the contract DSN using the final-patch priority order."""
    env = os.environ if environ is None else environ
    database_url = env.get(DATABASE_URL_ENV)
    fallback_dsn = env.get(DATABASE_DSN_ENV)
    if database_url:
        if fallback_dsn and message_sink is not None:
            message_sink(DUAL_DSN_WARNING)
        return database_url
    if fallback_dsn:
        return fallback_dsn
    raise DatabaseConfigError(MISSING_DSN_ERROR)


def import_psycopg() -> Any:
    try:
        import psycopg
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
        raise DatabaseUnavailable("psycopg is required for HKG-T24 database commands.") from exc
    return psycopg


def connect(database_url: str) -> Any:
    psycopg = import_psycopg()
    return psycopg.connect(database_url)


def redact_database_url(database_url: str | None) -> str:
    if not database_url:
        return "NOT_SET"
    if "@" not in database_url:
        return "postgresql://***"
    scheme_and_auth, host = database_url.rsplit("@", 1)
    scheme = scheme_and_auth.split("://", 1)[0] if "://" in scheme_and_auth else "postgresql"
    return f"{scheme}://***:***@{host}"


def database_url_hash(database_url: str) -> str:
    return sha256_text(database_url)[:16]
