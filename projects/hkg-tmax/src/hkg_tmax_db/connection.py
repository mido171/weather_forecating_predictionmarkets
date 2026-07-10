"""Optional PostgreSQL execution helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class DatabaseUnavailable(RuntimeError):
    """Raised when PostgreSQL execution cannot run in this environment."""


def import_psycopg() -> Any:
    try:
        import psycopg
    except ModuleNotFoundError as exc:
        raise DatabaseUnavailable(
            "psycopg is not installed in the active environment. "
            "Install requirements or use the project venv after dependency sync.",
        ) from exc
    return psycopg


def apply_migration(database_url: str, migration_path: Path) -> None:
    psycopg = import_psycopg()
    sql = migration_path.read_text(encoding="utf-8")
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql)
        connection.commit()


def redact_database_url(database_url: str | None) -> str:
    if not database_url:
        return "NOT_SET"
    if "@" not in database_url:
        return database_url
    scheme_and_auth, host = database_url.rsplit("@", 1)
    scheme = scheme_and_auth.split("://", 1)[0] if "://" in scheme_and_auth else "postgresql"
    return f"{scheme}://***:***@{host}"
