"""Schema contract primitives for source discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TableRef:
    schema: str
    table: str

    @property
    def qualified(self) -> str:
        return f"{self.schema}.{self.table}"


@dataclass(frozen=True)
class SourceCheck:
    object_name: str
    status: str
    message: str
    row_count: int | None = None


@dataclass(frozen=True)
class DiscoveredTable:
    logical_name: str
    table_ref: TableRef
    date_column: str | None = None
    value_column: str | None = None


def table_exists(connection: Any, table_ref: TableRef) -> bool:
    with connection.cursor() as cursor:
        cursor.execute("SELECT to_regclass(%s)", (table_ref.qualified,))
        row = cursor.fetchone()
    return row is not None and row[0] is not None


def table_columns(connection: Any, table_ref: TableRef) -> set[str]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            """,
            (table_ref.schema, table_ref.table),
        )
        return {str(row[0]) for row in cursor.fetchall()}


def count_rows(connection: Any, table_ref: TableRef) -> int:
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT count(*) FROM {table_ref.qualified}")
        row = cursor.fetchone()
    if row is None:
        return 0
    return int(row[0])


def choose_column(columns: set[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def discover_table(
    connection: Any,
    *,
    logical_name: str,
    primary: TableRef,
    fallbacks: tuple[TableRef, ...],
    date_columns: tuple[str, ...] = (),
    value_columns: tuple[str, ...] = (),
    ordered_fallbacks: bool = False,
) -> tuple[DiscoveredTable | None, list[SourceCheck]]:
    checks: list[SourceCheck] = []
    if table_exists(connection, primary):
        columns = table_columns(connection, primary)
        return (
            DiscoveredTable(
                logical_name=logical_name,
                table_ref=primary,
                date_column=choose_column(columns, date_columns),
                value_column=choose_column(columns, value_columns),
            ),
            [SourceCheck(primary.qualified, "PASS", "Primary table exists.", count_rows(connection, primary))],
        )

    existing_fallbacks = [candidate for candidate in fallbacks if table_exists(connection, candidate)]
    if ordered_fallbacks and existing_fallbacks:
        table_ref = existing_fallbacks[0]
        columns = table_columns(connection, table_ref)
        checks.append(
            SourceCheck(
                primary.qualified,
                "WARNING",
                f"Primary table absent; selected ordered fallback `{table_ref.qualified}`.",
            )
        )
        if len(existing_fallbacks) > 1:
            checks.append(
                SourceCheck(
                    logical_name,
                    "WARNING",
                    "Additional fallback candidates also exist but are lower priority: "
                    + ", ".join(table.qualified for table in existing_fallbacks[1:]),
                )
            )
        checks.append(
            SourceCheck(table_ref.qualified, "PASS", "Fallback table selected.", count_rows(connection, table_ref))
        )
        return (
            DiscoveredTable(
                logical_name=logical_name,
                table_ref=table_ref,
                date_column=choose_column(columns, date_columns),
                value_column=choose_column(columns, value_columns),
            ),
            checks,
        )
    if len(existing_fallbacks) == 1:
        table_ref = existing_fallbacks[0]
        columns = table_columns(connection, table_ref)
        checks.append(
            SourceCheck(
                primary.qualified,
                "WARNING",
                f"Primary table absent; discovered fallback `{table_ref.qualified}`.",
            )
        )
        checks.append(
            SourceCheck(table_ref.qualified, "PASS", "Fallback table selected.", count_rows(connection, table_ref))
        )
        return (
            DiscoveredTable(
                logical_name=logical_name,
                table_ref=table_ref,
                date_column=choose_column(columns, date_columns),
                value_column=choose_column(columns, value_columns),
            ),
            checks,
        )
    if len(existing_fallbacks) > 1:
        checks.append(
            SourceCheck(
                logical_name,
                "FAIL",
                "Multiple fallback candidates exist; source discovery is ambiguous: "
                + ", ".join(table.qualified for table in existing_fallbacks),
            )
        )
    else:
        checks.append(SourceCheck(logical_name, "FAIL", "No primary or fallback table exists."))
    return None, checks
