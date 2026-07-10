from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from backfill_hko_info_gov_hourly_readings import parse_dispatch, parse_index_candidates  # noqa: E402


def _parse(text: str, url: str, index_date: date):
    return parse_dispatch(
        content=text.encode("utf-8"),
        source_url=url,
        raw_html_path=r"C:\tmp\raw.html",
        raw_sha256="a" * 64,
        index_date_hkt=index_date,
        title_hint=None,
        press_weather_no_hint=None,
        retrieved_at_utc="2026-07-04T13:00:00Z",
    )


def test_index_parser_accepts_old_and_modern_hourly_links() -> None:
    html = """
    <a href="04/0504088.htm">PRESS WEATHER NO. 065 - HOURLY READINGS</a>
    <a href="/gia/wr/202607/04/P2026070400751.htm">PRESS WEATHER NO. 267 - HOURLY READINGS</a>
    <a href="/gia/wr/202607/04/P2026070400800.htm">PRESS WEATHER NO. 270 - LOCAL WEATHER FORECAST</a>
    """

    candidates = parse_index_candidates(
        html.encode("utf-8"),
        date(1998, 5, 4),
        "https://www.info.gov.hk/gia/wr/199805/04.htm",
    )

    assert [item.press_weather_no for item in candidates] == [65, 267]
    assert candidates[0].source_url == "https://www.info.gov.hk/gia/wr/199805/04/0504088.htm"
    assert candidates[1].source_url == "https://www.info.gov.hk/gia/wr/202607/04/P2026070400751.htm"


def test_old_1998_noon_dispatch_parses_hko_and_station_readings() -> None:
    parsed = _parse(
        """
        PRESS WEATHER NO. 065 - HOURLY READINGS
        AT NOON AT THE HONG KONG OBSERVATORY THE AIR TEMPERATURE WAS
        25 DEGREES CELSIUS AND THE RELATIVE HUMIDITY 90 PER CENT.
        2.8 MILLIMETRES OF RAINFALL WERE RECORDED AT THE HONG KONG
        OBSERVATORY BETWEEN MIDNIGHT LAST NIGHT AND MIDDAY TODAY.
        THE AIR TEMPERATURES AT OTHER PLACES WERE:
        KING'S PARK 25 DEGREES;
        WONG CHUK HANG 25 DEGREES;
        TA KWU LING 24 DEGREES;
        LAU FAU SHAN 26 DEGREES;
        TAI PO 25 DEGREES;
        SHA TIN 25 DEGREES;
        TUEN MUN 27 DEGREES;
        TSEUNG KWAN O 24 DEGREES;
        SAI KUNG 25 DEGREES;
        CHEUNG CHAU 27 DEGREES.
        DISPATCHED BY HONG KONG OBSERVATORY AT 12:03 HKT ON 04.05.1998
        """,
        "https://www.info.gov.hk/gia/wr/199805/04/0504088.htm",
        date(1998, 5, 4),
    )

    assert parsed.parse_status == "parsed"
    assert parsed.dispatch_at_utc == "1998-05-04T04:03:00Z"
    assert parsed.observation_at_hkt == "1998-05-04 12:00:00"
    assert parsed.hko_air_temp_c == 25
    assert parsed.hko_relative_humidity_pct == 90
    assert parsed.rainfall_text is not None
    assert parsed.station_count == 10
    assert parsed.station_temp_min_c == 24
    assert parsed.station_temp_max_c == 27
    assert parsed.station_readings_jsonb[-1]["station_canonical_name"] == "CHEUNG CHAU"


def test_2000_pm_dispatch_preserves_warning_text() -> None:
    parsed = _parse(
        """
        PRESS WEATHER NO. 102 - HOURLY READINGS
        AT 11 P.M. AT THE HONG KONG OBSERVATORY THE AIR TEMPERATURE WAS
        19 DEGREES CELSIUS AND THE RELATIVE HUMIDITY 79 PER CENT.
        PLEASE BE REMINDED THAT:
        THE FIRE DANGER WARNING IS YELLOW AND THE FIRE RISK IS HIGH.
        THE AIR TEMPERATURES AT OTHER PLACES WERE:
        KING'S PARK 17 DEGREES;
        WONG CHUK HANG 15 DEGREES;
        TA KWU LING 15 DEGREES;
        LAU FAU SHAN 16 DEGREES;
        TAI PO 15 DEGREES;
        SHA TIN 14 DEGREES;
        TUEN MUN 18 DEGREES;
        TSEUNG KWAN O 17 DEGREES;
        SAI KUNG 15 DEGREES;
        CHEUNG CHAU 17 DEGREES;
        CHEK LAP KOK 17 DEGREES.
        DISPATCHED BY HONG KONG OBSERVATORY AT 23:02 HKT ON 01.01.2000
        """,
        "https://www.info.gov.hk/gia/wr/200001/01/0101259.htm",
        date(2000, 1, 1),
    )

    assert parsed.observation_at_hkt == "2000-01-01 23:00:00"
    assert parsed.hko_air_temp_c == 19
    assert parsed.station_count == 11
    assert parsed.warning_text is not None
    assert "FIRE DANGER WARNING" in parsed.warning_text


def test_modern_dispatch_preserves_all_station_lines_and_missing_marker() -> None:
    parsed = _parse(
        """
        PRESS WEATHER NO. 267 - HOURLY READINGS
        AT 7 P.M. AT THE HONG KONG OBSERVATORY THE AIR TEMPERATURE
        WAS 28 DEGREES CELSIUS AND THE RELATIVE HUMIDITY 86 PER
        CENT.
        THE AIR TEMPERATURES AT OTHER PLACES WERE:
        KING'S PARK                 27 DEGREES;
        WONG CHUK HANG              27 DEGREES;
        TA KWU LING                 27 DEGREES;
        LAU FAU SHAN                26 DEGREES;
        TAI PO                      27 DEGREES;
        SHA TIN                     28 DEGREES;
        TUEN MUN                    28 DEGREES;
        TSEUNG KWAN O               26 DEGREES;
        SAI KUNG                    27 DEGREES;
        CHEUNG CHAU                 27 DEGREES;
        CHEK LAP KOK                29 DEGREES;
        TSING YI                    28 DEGREES;
        SHEK KONG                   // DEGREES;
        TSUEN WAN HO KOON           26 DEGREES;
        TSUEN WAN SHING MUN VALLEY  26 DEGREES;
        HONG KONG PARK              28 DEGREES;
        SHAU KEI WAN                27 DEGREES;
        KOWLOON CITY                27 DEGREES;
        HAPPY VALLEY                29 DEGREES;
        WONG TAI SIN                27 DEGREES;
        STANLEY                     27 DEGREES;
        KWUN TONG                   27 DEGREES;
        SHAM SHUI PO                28 DEGREES;
        KAI TAK RUNWAY PARK         28 DEGREES;
        YUEN LONG PARK              27 DEGREES;
        TAI MEI TUK                 26 DEGREES.
        BETWEEN 5:45 AND 6:45 P.M., LIGHTNING WAS DETECTED WITHIN
        NEW TERRITORIES EAST, HONG KONG AND KOWLOON.
        DISPATCHED BY HONG KONG OBSERVATORY AT 19:02 HKT ON 04.07.2026
        """,
        "https://www.info.gov.hk/gia/wr/202607/04/P2026070400751.htm",
        date(2026, 7, 4),
    )

    assert parsed.station_count == 26
    assert parsed.station_missing_count == 1
    assert parsed.station_temp_min_c == 26
    assert parsed.station_temp_max_c == 29
    shek_kong = next(
        item for item in parsed.station_readings_jsonb if item["station_canonical_name"] == "SHEK KONG"
    )
    assert shek_kong["temperature_c"] is None
    assert shek_kong["temperature_missing"] is True
    assert shek_kong["raw_temperature_text"] == "// DEGREES"
    assert parsed.lightning_text is not None
    assert "LIGHTNING WAS DETECTED" in parsed.lightning_text


def test_modern_dispatch_parses_tropical_cyclone_fields_and_warning_text() -> None:
    parsed = _parse(
        """
        PRESS WEATHER NO. 284 - HOURLY READINGS
        AT 9 P.M. AT THE HONG KONG OBSERVATORY THE AIR TEMPERATURE
        WAS 29 DEGREES CELSIUS AND THE RELATIVE HUMIDITY 81 PER
        CENT.
        PLEASE BE REMINDED THAT:
        THE THUNDERSTORM WARNING HAS BEEN ISSUED. IT WILL REMAIN
        EFFECTIVE UNTIL 11:00 P.M. TODAY.
        THE AIR TEMPERATURES AT OTHER PLACES WERE:
        KING'S PARK 28 DEGREES;
        SHEK KONG // DEGREES.
        HERE IS THE INFORMATION ON SEVERE TROPICAL STORM MAYSAK AT
        8 P.M.:
        LOCATION: 21.1 DEGREES NORTH, 107.9 DEGREES EAST.
        DISPATCHED BY HONG KONG OBSERVATORY AT 21:02 HKT ON 04.07.2026
        """,
        "https://www.info.gov.hk/gia/wr/202607/04/P2026070400838.htm",
        date(2026, 7, 4),
    )

    assert parsed.warning_text is not None
    assert "THUNDERSTORM WARNING" in parsed.warning_text
    assert parsed.tropical_cyclone_name == "MAYSAK"
    assert parsed.tropical_cyclone_lat == 21.1
    assert parsed.tropical_cyclone_lon == 107.9
    assert parsed.target_station_present is True


def test_midnight_dispatch_accepts_split_hko_temperature_phrase() -> None:
    parsed = _parse(
        """
        PRESS WEATHER NO. 004 - HOURLY READINGS
        AT MIDNIGHT AT THE HONG KONG OBSERVATORY THE AIR
        TEMPERATURE WAS 29 DEGREES CELSIUS AND THE RELATIVE
        HUMIDITY 87 PER CENT.
        THE AIR TEMPERATURES AT OTHER PLACES WERE:
        KING'S PARK 28 DEGREES;
        SHEK KONG // DEGREES.
        DISPATCHED BY HONG KONG OBSERVATORY AT 00:02 HKT ON 04.07.2026
        """,
        "https://www.info.gov.hk/gia/wr/202607/04/P2026070400005.htm",
        date(2026, 7, 4),
    )

    assert parsed.parse_status == "parsed"
    assert parsed.observation_at_hkt == "2026-07-04 00:00:00"
    assert parsed.hko_air_temp_c == 29
    assert parsed.hko_relative_humidity_pct == 87
    assert parsed.target_station_present is True


def test_old_tropical_cyclone_coordinates_do_not_become_station_readings() -> None:
    parsed = _parse(
        """
        PRESS WEATHER NO. 045 - HOURLY READINGS
        AT 10 A.M. AT THE HONG KONG OBSERVATORY THE AIR TEMPERATURE WAS
        24 DEGREES CELSIUS AND THE RELATIVE HUMIDITY 68 PER CENT.
        THE AIR TEMPERATURES AT OTHER PLACES WERE:
        KING'S PARK     23 DEGREES;
        WONG CHUK HANG  24 DEGREES;
        TA KWU LING     23 DEGREES;
        LAU FAU SHAN    24 DEGREES;
        TAI PO          23 DEGREES;
        SHA TIN         23 DEGREES;
        TUEN MUN        23 DEGREES;
        TSEUNG KWAN O   23 DEGREES;
        SAI KUNG        23 DEGREES;
        CHEUNG CHAU     24 DEGREES;
        CHEK LAP KOK    24 DEGREES.
        AT 10 A.M. THE CENTRE OF TROPICAL DEPRESSION WAS NEAR 15.6
        DEGREES NORTH 111.2 DEGREES EAST.
        DISPATCHED BY HONG KONG OBSERVATORY AT 10:02 HKT ON 28.04.1999
        """,
        "https://www.info.gov.hk/gia/wr/199904/28/0428055.htm",
        date(1999, 4, 28),
    )

    assert parsed.station_count == 11
    assert parsed.station_temp_min_c == 23
    assert parsed.station_temp_max_c == 24
    assert parsed.station_readings_jsonb[-1]["station_canonical_name"] == "CHEK LAP KOK"
    assert parsed.tropical_cyclone_lat == 15.6
    assert parsed.tropical_cyclone_lon == 111.2


def test_old_minimum_temperature_summary_does_not_become_station_reading() -> None:
    parsed = _parse(
        """
        PRESS WEATHER NO. 047 - HOURLY READINGS
        AT 9 A.M. AT THE HONG KONG OBSERVATORY THE AIR TEMPERATURE WAS
        15 DEGREES CELSIUS AND THE RELATIVE HUMIDITY 65 PER CENT.
        THE AIR TEMPERATURES AT OTHER PLACES WERE:
        KING'S PARK     15 DEGREES;
        WONG CHUK HANG  15 DEGREES;
        TA KWU LING     13 DEGREES;
        LAU FAU SHAN    12 DEGREES;
        TAI PO          12 DEGREES;
        SHA TIN         13 DEGREES;
        TUEN MUN        13 DEGREES;
        TSEUNG KWAN O   15 DEGREES;
        SAI KUNG        14 DEGREES;
        CHEUNG CHAU     14 DEGREES;
        CHEK LAP KOK    14 DEGREES.
        BETWEEN MIDNIGHT AND 9 A.M. THE MINIMUM TEMPERATURE WAS
        15.4 DEGREES CELSIUS AT THE HONG KONG OBSERVATORY.
        DISPATCHED BY HONG KONG OBSERVATORY AT 09:02 HKT ON 11.03.1999
        """,
        "https://www.info.gov.hk/gia/wr/199903/11/0311047.htm",
        date(1999, 3, 11),
    )

    assert parsed.station_count == 11
    assert parsed.station_temp_min_c == 12
    assert parsed.station_temp_max_c == 15
    assert parsed.station_readings_jsonb[-1]["station_canonical_name"] == "CHEK LAP KOK"
