from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation


class HKOParseError(ValueError):
    """Raised when an HKO climate payload cannot be parsed safely."""


@dataclass(frozen=True)
class DailyClimateRow:
    year: int
    month: int
    day: int
    local_date: date | None
    value: Decimal | None
    completeness: str | None
    parse_issue: str | None
    raw: dict[str, str]


_REQUIRED_HEADER = ("Year", "Month", "Day", "Value")


def _find_header(lines: list[str]) -> int:
    for index, line in enumerate(lines):
        cells = tuple(cell.strip().lstrip("\ufeff") for cell in next(csv.reader([line])))
        if all(field in cells for field in _REQUIRED_HEADER):
            return index
    raise HKOParseError(
        "Could not find a CSV header containing Year, Month, Day, and Value"
    )


def _parse_optional_decimal(raw: str) -> tuple[Decimal | None, str | None]:
    value = raw.strip()
    if not value or value.upper() in {"N/A", "NA", "NULL", "***", "M"}:
        return None, "missing_value"
    try:
        return Decimal(value), None
    except InvalidOperation:
        return None, f"non_numeric_value:{value}"


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
            (key or "").strip().lstrip("\ufeff"): (value or "").strip()
            for key, value in raw_row.items()
        }
        if not any(normalized.values()):
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

        value, value_issue = _parse_optional_decimal(normalized.get("Value", ""))
        issues = [issue for issue in (date_issue, value_issue) if issue]
        rows.append(
            DailyClimateRow(
                year=year,
                month=month,
                day=day,
                local_date=local_date,
                value=value,
                completeness=normalized.get("Data Completeness") or None,
                parse_issue=";".join(issues) if issues else None,
                raw=normalized,
            )
        )
    if not rows:
        raise HKOParseError("CSV contained no data rows")
    return rows
