from datetime import date
from decimal import Decimal

import pytest

from hkg_tmax.hko import HKOParseError, parse_daily_climate_csv, parse_daily_extract_json


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
    assert rows[0].value_precision == Decimal("0.1")
    assert rows[1].value is None
    assert rows[2].local_date is None
    assert "invalid_date" in (rows[2].parse_issue or "")


def test_parse_hko_daily_climate_with_bilingual_header() -> None:
    content = """"日最高氣溫(攝氏度) - 天文台"
"Daily Maximum Temperature (°C) at the Hong Kong Observatory"
年/Year,月/Month,日/Day,數值/Value,"數據完整性/data Completeness"
2026,5,30,32.6,C
2026,5,31,30.9,C
"""
    rows = parse_daily_climate_csv(content)
    assert rows[0].local_date == date(2026, 5, 30)
    assert rows[0].value == Decimal("32.6")
    assert rows[0].completeness == "C"


def test_parse_hko_daily_climate_requires_expected_header() -> None:
    with pytest.raises(HKOParseError, match="Could not find a CSV header"):
        parse_daily_climate_csv("not,the,expected,columns\n1,2,3,4\n")


def test_parse_daily_extract_json_extracts_absolute_daily_max() -> None:
    content = """{"stn":{"data":[{"month":5,"dayData":[
["30","1009.6","32.6","29.3","27.3","24.0","73","70","  0.0"],
["31","1009.4","30.9","28.8","27.9","23.8","74","79","Trace"],
["Mean/Total","1010.3","29.2","26.8","25.2","23.3","81","82","227.2"],
["Normal","1009.3","28.8","26.3","24.5","23.0","83","76","290.6"]
]}]}}"""
    rows = parse_daily_extract_json(content, year=2026, month=5)
    assert [row.local_date for row in rows] == [date(2026, 5, 30), date(2026, 5, 31)]
    assert rows[0].absolute_daily_max_c == Decimal("32.6")
    assert rows[0].value_precision == Decimal("0.1")
    assert rows[0].completeness == "C"


def test_parse_daily_extract_json_marks_incomplete_values() -> None:
    content = """{"stn":{"data":[{"month":6,"dayData":[
["01","1008.9","32.3#","29.2","27.6","24.2","74","79","Trace"]
]}]}}"""
    rows = parse_daily_extract_json(content, year=2026, month=6)
    assert rows[0].absolute_daily_max_c == Decimal("32.3")
    assert rows[0].completeness == "I"
    assert rows[0].parse_issue == "data_incomplete"


def test_parse_daily_extract_json_requires_matching_rows() -> None:
    with pytest.raises(HKOParseError, match="no matching daily rows"):
        parse_daily_extract_json('{"stn":{"data":[{"month":5,"dayData":[]}]}}', year=2026, month=6)
