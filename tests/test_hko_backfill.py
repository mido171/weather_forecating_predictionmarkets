from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from hkg_tmax.hko_backfill import (
    DATAGOV_HISTORICAL_LIVE_FEEDS,
    DATAGOV_HISTORICAL_RSS_FEEDS,
    UPPER_AIR_DOWNLOADS,
    build_daily_climate_downloads,
    build_daily_extract_downloads,
    build_datagov_historical_downloads_from_listing,
    build_ncep_filter_url,
    build_tc_best_track_downloads,
    extract_arwf_forecast_codes,
    extract_noaa_isd_nearby_stations,
)


def test_daily_climate_backfill_uses_hko_d1_all_years() -> None:
    downloads = build_daily_climate_downloads()

    assert len(downloads) == 21
    maximum = next(item for item in downloads if item.source_id.endswith("maximum_temperature_all"))
    assert maximum.url.endswith("stn=HKO&ele=MAXT&yr=ALL")
    assert maximum.extension == "csv"
    assert maximum.metadata["station_code"] == "HKO"


def test_daily_extract_backfill_includes_annual_and_current_month_payloads() -> None:
    now = datetime(2026, 6, 19, 8, 0, tzinfo=ZoneInfo("Asia/Hong_Kong"))
    downloads = build_daily_extract_downloads(now)

    assert len(downloads) == 150
    assert downloads[0].source_id == "hko_daily_extract_catalog"
    assert any(item.url.endswith("dailyExtract_1884.xml") for item in downloads)
    assert any(item.url.endswith("dailyExtract_2026.xml") for item in downloads)
    assert any(item.url.endswith("dailyExtract_202606.xml") for item in downloads)
    assert not any(item.url.endswith("dailyExtract_202607.xml") for item in downloads)


def test_tc_best_track_backfill_covers_public_hko_years() -> None:
    downloads = build_tc_best_track_downloads()

    assert len(downloads) == 41
    assert downloads[0].source_id == "hko_tropical_cyclone_best_track_dictionary"
    assert any(item.url.endswith("HKO1985BST.csv") for item in downloads)
    assert any(item.url.endswith("HKO2024BST.csv") for item in downloads)


def test_upper_air_batch_uses_resolved_igra_station() -> None:
    urls = {item.source_id: item.url for item in UPPER_AIR_DOWNLOADS}

    assert "HKM00045004-data.txt.zip" in urls["noaa_igra_hkm00045004_period_of_record"]
    assert "HKM00045004-data-beg2025.txt.zip" in urls["noaa_igra_hkm00045004_year_to_date"]


def test_datagov_historical_listing_builds_official_get_file_urls() -> None:
    feed = DATAGOV_HISTORICAL_LIVE_FEEDS[0]
    downloads = build_datagov_historical_downloads_from_listing(
        feed,
        {
            "data-files": [
                {
                    "filename": "archive-20260618.zip",
                    "period": "D",
                    "resource_file_count": 143,
                    "size": 116627,
                    "timestamp": "20260618",
                }
            ]
        },
    )

    assert len(downloads) == 1
    item = downloads[0]
    assert item.source_id == "datagov_hko_historical_latest_1min_temperature_archive"
    assert item.extension == "zip"
    assert "historical-archive%2Fget-file" not in item.url
    assert "historical-archive/get-file" in item.url
    assert "time=20260618" in item.url
    assert item.metadata["data_gov_archive_resource_file_count"] == 143


def test_datagov_historical_rss_feeds_are_forecast_vintages() -> None:
    assert len(DATAGOV_HISTORICAL_RSS_FEEDS) == 15
    assert {
        feed.family for feed in DATAGOV_HISTORICAL_RSS_FEEDS
    } == {"D_official_hko_forecast_vintages"}
    assert any("CurrentWeather.xml" in feed.resource_url for feed in DATAGOV_HISTORICAL_RSS_FEEDS)
    assert any(
        "SeveralDaysWeatherForecast_v2.xml" in feed.resource_url
        for feed in DATAGOV_HISTORICAL_RSS_FEEDS
    )


def test_arwf_forecast_code_extraction_uses_station_alias_and_grid_codes() -> None:
    station_config = """
    stationConfigAWS["hko"] = { code: "hko" };
    stationConfigAWS["gi"] = { code: "gi", ARWF_code:"PEN" };
    stationConfigAWS["zcp"] = { code: "zcp", ARWF_code:"SE1", gridXML:"G153" };
    """
    common_js = """
    matchXML["HKO"] = "G152";
    matchXML["SC"] = "G150";
    """

    assert extract_arwf_forecast_codes(station_config, common_js) == (
        "G150",
        "G152",
        "G153",
        "GI",
        "HKO",
        "PEN",
        "SE1",
        "ZCP",
    )


def test_noaa_isd_nearby_station_filter_uses_hong_kong_regional_bounds() -> None:
    history = """USAF,WBAN,STATION NAME,CTRY,STATE,ICAO,LAT,LON,ELEV(M),BEGIN,END
450050,99999,HONG KONG OBSERVATORY,CH,,,+22.300,+114.167,+0062.0,19460901,20180930
592870,99999,BAIYUN INTL,CH,,ZGGG,+23.392,+113.299,+0015.2,19451130,20250824
999999,99999,OUTSIDE,CH,,,+30.000,+120.000,+0000.0,20000101,20010101
"""

    stations = extract_noaa_isd_nearby_stations(history)

    assert [station.usaf for station in stations] == ["450050", "592870"]
    assert stations[0].begin_year == 1946
    assert stations[0].end_year == 2018
    assert stations[1].icao == "ZGGG"


def test_ncep_filter_url_includes_point_in_time_subset_parameters() -> None:
    url = build_ncep_filter_url(
        "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl",
        directory="/gfs.20260619/00/atmos",
        filename="gfs.t00z.pgrb2.0p25.f006",
        level_params=("lev_2_m_above_ground", "lev_925_mb"),
        variable_params=("var_TMP", "var_RH"),
    )

    assert "dir=%2Fgfs.20260619%2F00%2Fatmos" in url
    assert "file=gfs.t00z.pgrb2.0p25.f006" in url
    assert "lev_2_m_above_ground=on" in url
    assert "lev_925_mb=on" in url
    assert "var_TMP=on" in url
    assert "var_RH=on" in url
    assert "leftlon=112" in url
    assert "rightlon=116" in url
    assert "toplat=25" in url
    assert "bottomlat=21" in url
