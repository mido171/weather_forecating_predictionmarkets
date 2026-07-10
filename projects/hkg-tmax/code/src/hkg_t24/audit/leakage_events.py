"""Leakage audit event helpers."""

from __future__ import annotations

from datetime import date
from typing import Any


def record_leakage_event(
    connection: Any,
    *,
    event_level: str,
    event_code: str,
    event_message: str,
    target_date_hkt: date | None = None,
    source_code: str | None = None,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO model_validation.leakage_audit_event (
              event_level, event_code, event_message, target_date_hkt, source_code
            )
            VALUES (%s, %s, %s, %s, %s)
            """,
            (event_level, event_code, event_message, target_date_hkt, source_code),
        )


def leakage_error_count(connection: Any) -> int:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT count(*) FROM model_validation.leakage_audit_event WHERE event_level = 'ERROR'"
        )
        row = cursor.fetchone()
    return 0 if row is None else int(row[0])
