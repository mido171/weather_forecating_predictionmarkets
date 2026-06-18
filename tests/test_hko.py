from datetime import date
from decimal import Decimal

import pytest

from hkg_tmax.hko import HKOParseError, parse_daily_climate_csv


def test_parse_hko_daily_climate_with_title_line() -> None:
    content = """Daily Maximum Temperature (°C) at the Hong Kong Observatory
Year,Month,Day,Value,Data Completeness
2026,6,17,31.4,C
2026,6,18,***,M
1900,2,29,N/A,M
"""
    rows = parse_daily_climate_csv(content)
    assert rows[0].local_date == date(2026, 6, 17)
    assert rows[0].value == Decimal("31.4")
    assert rows[1].value is None
    assert rows[2].local_date is None
    assert "invalid_date" in (rows[2].parse_issue or "")


def test_parse_hko_daily_climate_requires_expected_header() -> None:
    with pytest.raises(HKOParseError, match="Could not find a CSV header"):
        parse_daily_climate_csv("not,the,expected,columns\n1,2,3,4\n")
