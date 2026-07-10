from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any


class HKOParseError(ValueError):
    """Raised when an HKO climate payload cannot be parsed safely."""


@dataclass(frozen=True)
class DailyClimateRow:
    year: int
    month: int
    day: int
    local_date: date | None
    value: Decimal | None
    value_precision: Decimal | None
    completeness: str | None
    parse_issue: str | None
    raw: dict[str, str]


@dataclass(frozen=True)
class DailyExtractRow:
    year: int
    month: int
    day: int
    local_date: date
    absolute_daily_max_c: Decimal | None
    value_precision: Decimal | None
    completeness: str
    parse_issue: str | None
    raw: dict[str, str]


_REQUIRED_HEADER = ("Year", "Month", "Day", "Value")
_HEADER_ALIASES = {
    "year": "Year",
    "month": "Month",
    "day": "Day",
    "value": "Value",
    "data completeness": "Data Completeness",
}


def _canonical_header_cell(cell: str) -> str:
    clean = " ".join(cell.strip().lstrip("\ufeff").split())
    for part in reversed(clean.split("/")):
        key = " ".join(part.strip().split()).casefold()
        if key in _HEADER_ALIASES:
            return _HEADER_ALIASES[key]
    return clean


def _find_header(lines: list[str]) -> int:
    for index, line in enumerate(lines):
        cells = tuple(_canonical_header_cell(cell) for cell in next(csv.reader([line])))
        if all(field in cells for field in _REQUIRED_HEADER):
            return index
    raise HKOParseError(
        "Could not find a CSV header containing Year, Month, Day, and Value"
    )


def _precision_from_decimal_text(value: str) -> Decimal:
    if "." not in value:
        return Decimal("1")
    places = len(value.split(".", 1)[1])
    return Decimal("1").scaleb(-places)


def _parse_optional_decimal(raw: str) -> tuple[Decimal | None, Decimal | None, str | None]:
    value = raw.strip()
    if not value or value.upper() in {"N/A", "NA", "NULL", "***", "M"}:
        return None, None, "missing_value"
    try:
        return Decimal(value), _precision_from_decimal_text(value), None
    except InvalidOperation:
        return None, None, f"non_numeric_value:{value}"


def _parse_marked_decimal(raw: str) -> tuple[Decimal | None, Decimal | None, str | None]:
    value = raw.strip()
    if not value:
        return None, None, "missing_value"
    if value == "***":
        return None, None, "missing_value"

    issues: list[str] = []
    if "#" in value:
        issues.append("data_incomplete")
    if "*" in value:
        issues.append("data_unavailable_marker")
    cleaned = value.replace("#", "").replace("*", "").strip()
    if not cleaned:
        return None, None, ";".join(issues + ["missing_value"])
    try:
        parsed = Decimal(cleaned)
    except InvalidOperation:
        return None, None, ";".join(issues + [f"non_numeric_value:{value}"])
    return parsed, _precision_from_decimal_text(cleaned), ";".join(issues) or None


def parse_daily_climate_csv(content: bytes | str) -> list[DailyClimateRow]:
    text = content.decode("utf-8-sig") if isinstance(content, bytes) else content
    lines = text.splitlines()
    header_index = _find_header(lines)
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
    if reader.fieldnames is None:
        raise HKOParseError("CSV has no field names")

    rows: list[DailyClimateRow] = []
    for row_number, raw_row in enumerate(reader, start=header_index + 2):
        normalized = {
            _canonical_header_cell(key or ""): (value or "").strip()
            for key, value in raw_row.items()
        }
        if not any(normalized.values()):
            continue
        if (
            normalized.get("Year")
            and not normalized.get("Month")
            and not normalized.get("Day")
            and not normalized.get("Value")
        ):
            continue
        try:
            year = int(normalized["Year"])
            month = int(normalized["Month"])
            day = int(normalized["Day"])
        except (KeyError, ValueError) as exc:
            raise HKOParseError(f"Invalid date fields at CSV row {row_number}") from exc

        local_date: date | None
        date_issue: str | None = None
        try:
            local_date = date(year, month, day)
        except ValueError as exc:
            local_date = None
            date_issue = f"invalid_date:{exc}"

        value, precision, value_issue = _parse_optional_decimal(normalized.get("Value", ""))
        issues = [issue for issue in (date_issue, value_issue) if issue]
        rows.append(
            DailyClimateRow(
                year=year,
                month=month,
                day=day,
                local_date=local_date,
                value=value,
                value_precision=precision,
                completeness=normalized.get("Data Completeness") or None,
                parse_issue=";".join(issues) if issues else None,
                raw=normalized,
            )
        )
    if not rows:
        raise HKOParseError("CSV contained no data rows")
    return rows


def _as_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HKOParseError(f"Daily Extract payload has invalid {context}")
    return value


def _as_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise HKOParseError(f"Daily Extract payload has invalid {context}")
    return value


def parse_daily_extract_json(
    content: bytes | str,
    *,
    year: int,
    month: int | None = None,
) -> list[DailyExtractRow]:
    text = content.decode("utf-8-sig") if isinstance(content, bytes) else content
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HKOParseError("Daily Extract payload is not valid JSON text") from exc

    root = _as_mapping(payload, "root object")
    stn = _as_mapping(root.get("stn"), "stn object")
    month_blocks = _as_list(stn.get("data"), "stn.data array")

    rows: list[DailyExtractRow] = []
    for block_index, raw_block in enumerate(month_blocks):
        block = _as_mapping(raw_block, f"stn.data[{block_index}]")
        try:
            block_month = int(block["month"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HKOParseError(f"Daily Extract month is invalid at block {block_index}") from exc
        if month is not None and block_month != month:
            continue

        day_rows = _as_list(block.get("dayData"), f"stn.data[{block_index}].dayData")
        for row_index, raw_day in enumerate(day_rows):
            day_values = _as_list(raw_day, f"dayData[{row_index}]")
            if len(day_values) < 3:
                raise HKOParseError(
                    f"Daily Extract dayData[{row_index}] has fewer than 3 fields"
                )
            day_token = str(day_values[0]).strip()
            if not day_token.isdigit():
                continue
            day = int(day_token)
            try:
                local_date = date(year, block_month, day)
            except ValueError as exc:
                raise HKOParseError(
                    f"Daily Extract date is invalid: {year:04d}-{block_month:02d}-{day:02d}"
                ) from exc

            raw_max = str(day_values[2])
            value, precision, value_issue = _parse_marked_decimal(raw_max)
            completeness = "C"
            if value is None:
                completeness = "M"
            elif value_issue:
                completeness = "I"
            rows.append(
                DailyExtractRow(
                    year=year,
                    month=block_month,
                    day=day,
                    local_date=local_date,
                    absolute_daily_max_c=value,
                    value_precision=precision,
                    completeness=completeness,
                    parse_issue=value_issue,
                    raw={
                        "day": day_token,
                        "mean_pressure_hpa": str(day_values[1]).strip(),
                        "absolute_daily_max_c": raw_max.strip(),
                        "mean_temperature_c": str(day_values[3]).strip()
                        if len(day_values) > 3
                        else "",
                        "absolute_daily_min_c": str(day_values[4]).strip()
                        if len(day_values) > 4
                        else "",
                    },
                )
            )
    if not rows:
        raise HKOParseError("Daily Extract payload contained no matching daily rows")
    return rows
