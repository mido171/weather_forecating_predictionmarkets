"""Normalize GribStream `/runs` NDJSON into NWP core rows."""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NormalizedPoint:
    run_time_utc: datetime
    valid_time_utc: datetime
    lead_minutes: int
    location_id: int
    location_code: str
    selector_id: int | None
    member_number: int
    value: float | None


@dataclass(frozen=True)
class RejectedRow:
    row_number: int
    rejection_class: str
    rejection_reason: str
    evidence: dict[str, Any]


@dataclass(frozen=True)
class NormalizationResult:
    points: tuple[NormalizedPoint, ...]
    rejected_rows: tuple[RejectedRow, ...]


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def iter_ndjson_gzip(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(fs_path(path), "rt", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def fs_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name != "nt" or resolved.startswith("\\\\?\\"):
        return resolved
    if resolved.startswith("\\\\"):
        return "\\\\?\\UNC\\" + resolved.lstrip("\\")
    return "\\\\?\\" + resolved


def _member_number(row: dict[str, Any]) -> int:
    for key in ("member", "member_number", "ensemble_member"):
        value = row.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, int):
            return value
        text = str(value).strip().lower()
        if text in {"control", "deterministic", "none"}:
            return 0
        return int(text)
    return 0


def normalize_runs_rows(
    rows: list[dict[str, Any]],
    *,
    value_alias: str,
    location_ids_by_code: dict[str, int],
) -> NormalizationResult:
    points: list[NormalizedPoint] = []
    rejected: list[RejectedRow] = []
    for row_number, row in enumerate(rows, start=1):
        try:
            location_code = str(row["name"])
            if location_code not in location_ids_by_code:
                raise KeyError(f"Unknown location name/code: {location_code}")
            run_time = parse_utc(str(row["forecasted_at"]))
            valid_time = parse_utc(str(row["forecasted_time"]))
            lead_minutes = int((valid_time - run_time).total_seconds() // 60)
            if lead_minutes < 0:
                raise ValueError("valid time precedes run time")
            raw_value = row.get(value_alias)
            value = None if raw_value is None or raw_value == "" else float(raw_value)
            points.append(
                NormalizedPoint(
                    run_time_utc=run_time,
                    valid_time_utc=valid_time,
                    lead_minutes=lead_minutes,
                    location_id=location_ids_by_code[location_code],
                    location_code=location_code,
                    selector_id=None,
                    member_number=_member_number(row),
                    value=value,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - preserve bad-row evidence
            rejected.append(
                RejectedRow(
                    row_number=row_number,
                    rejection_class=type(exc).__name__,
                    rejection_reason=str(exc),
                    evidence={key: row.get(key) for key in ("forecasted_at", "forecasted_time", "name")},
                ),
            )
    points.sort(
        key=lambda item: (
            item.location_code,
            item.member_number,
            item.run_time_utc,
            item.valid_time_utc,
        ),
    )
    return NormalizationResult(points=tuple(points), rejected_rows=tuple(rejected))


def normalize_runs_ndjson_gzip(
    path: Path,
    *,
    value_alias: str,
    location_ids_by_code: dict[str, int],
) -> NormalizationResult:
    return normalize_runs_rows(
        iter_ndjson_gzip(path),
        value_alias=value_alias,
        location_ids_by_code=location_ids_by_code,
    )
