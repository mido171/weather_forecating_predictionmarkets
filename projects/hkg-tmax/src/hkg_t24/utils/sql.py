"""Narrow SQL string helpers."""

from __future__ import annotations

import re
from collections.abc import Sequence

IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def quote_ident(identifier: str) -> str:
    """Quote an already-validated PostgreSQL identifier."""
    if not IDENTIFIER_RE.match(identifier):
        raise ValueError(f"Unsafe SQL identifier: {identifier!r}")
    return f'"{identifier}"'


def qualified_name(schema: str, table: str) -> str:
    return f"{quote_ident(schema)}.{quote_ident(table)}"


def csv_line(values: Sequence[object | None]) -> str:
    """Render a small RFC4180-compatible CSV line."""
    rendered: list[str] = []
    for value in values:
        text = "" if value is None else str(value)
        if any(ch in text for ch in [",", '"', "\n", "\r"]):
            text = '"' + text.replace('"', '""') + '"'
        rendered.append(text)
    return ",".join(rendered)
