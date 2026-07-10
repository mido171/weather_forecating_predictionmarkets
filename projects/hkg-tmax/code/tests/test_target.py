from datetime import date
from decimal import Decimal

import pytest

from hkg_tmax.hko import DailyExtractRow
from hkg_tmax.target import TargetError, require_daily_extract_target


def _row(
    *,
    local_date: date = date(2026, 5, 31),
    value: Decimal | None = Decimal("30.9"),
    precision: Decimal | None = Decimal("0.1"),
    completeness: str = "C",
    parse_issue: str | None = None,
) -> DailyExtractRow:
    return DailyExtractRow(
        year=local_date.year,
        month=local_date.month,
        day=local_date.day,
        local_date=local_date,
        absolute_daily_max_c=value,
        value_precision=precision,
        completeness=completeness,
        parse_issue=parse_issue,
        raw={"absolute_daily_max_c": "" if value is None else str(value)},
    )


def test_require_daily_extract_target_returns_observation() -> None:
    observation = require_daily_extract_target(
        [_row()],
        target_date=date(2026, 5, 31),
        source_id="hko_daily_extract_202605",
        source_sha256="abc123",
    )
    assert observation.station_code == "HKO"
    assert observation.value_c == Decimal("30.9")
    assert observation.precision_c == Decimal("0.1")
    assert observation.source_sha256 == "abc123"


def test_require_daily_extract_target_rejects_missing_source() -> None:
    with pytest.raises(TargetError, match="missing source_id"):
        require_daily_extract_target([_row()], target_date=date(2026, 5, 31), source_id="")


def test_require_daily_extract_target_rejects_missing_field() -> None:
    with pytest.raises(TargetError, match="missing expected field"):
        require_daily_extract_target(
            [_row()],
            target_date=date(2026, 5, 31),
            source_id="hko_daily_extract_202605",
            field_name="Mean (deg. C)",
        )


def test_require_daily_extract_target_rejects_ambiguous_date() -> None:
    with pytest.raises(TargetError, match="ambiguous date match"):
        require_daily_extract_target(
            [_row(), _row()],
            target_date=date(2026, 5, 31),
            source_id="hko_daily_extract_202605",
        )


def test_require_daily_extract_target_rejects_unsupported_precision() -> None:
    with pytest.raises(TargetError, match="unsupported precision"):
        require_daily_extract_target(
            [_row(value=Decimal("30.91"), precision=Decimal("0.01"))],
            target_date=date(2026, 5, 31),
            source_id="hko_daily_extract_202605",
        )


def test_require_daily_extract_target_rejects_station_mismatch() -> None:
    with pytest.raises(TargetError, match="station mismatch"):
        require_daily_extract_target(
            [_row()],
            target_date=date(2026, 5, 31),
            source_id="hko_daily_extract_202605",
            station_code="HKA",
        )


def test_require_daily_extract_target_rejects_missing_value() -> None:
    with pytest.raises(TargetError, match="missing target value"):
        require_daily_extract_target(
            [_row(value=None, precision=None, completeness="M", parse_issue="missing_value")],
            target_date=date(2026, 5, 31),
            source_id="hko_daily_extract_202605",
        )


def test_require_daily_extract_target_rejects_source_failure() -> None:
    with pytest.raises(TargetError, match="source failure"):
        require_daily_extract_target(
            [_row()],
            target_date=date(2026, 5, 31),
            source_id="hko_daily_extract_202605",
            source_error="HTTP 500",
        )
