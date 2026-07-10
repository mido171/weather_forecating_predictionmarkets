from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from hkg_tmax.hko_backfill import (
    DATAGOV_HISTORICAL_LIVE_FEEDS,
    DATAGOV_HISTORICAL_RSS_FEEDS,
    SATELLITE_CURRENT_IMAGE_BASE_URL,
    SATELLITE_CURRENT_IMAGE_SPECS,
    UPPER_AIR_DOWNLOADS,
    _extract_satellite_filenames,
    _satellite_image_extension,
    _successful_archive_urls,
    build_daily_climate_downloads,
    build_daily_extract_downloads,
    build_datagov_historical_downloads_from_listing,
    build_ncep_filter_url,
    build_static_context_downloads,
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


def test_static_context_batch_includes_official_terrain_and_land_use_sources() -> None:
    downloads = build_static_context_downloads()
    urls = {item.url for item in downloads}
    source_ids = {item.source_id for item in downloads}

    assert "landsd_whole_hk_dtm_5m_asc_zip" in source_ids
    assert "landsd_igeocom_geojson_zip" in source_ids
    assert "csdi_landsd_dtm_geotiff_zip" in source_ids
    assert "pland_luhk_2024_statistics_english_csv" in source_ids
    assert any(item.source_id == "csdi_pland_luhk_2018_raster_geotiff_zip" for item in downloads)
    assert any(item.source_id == "csdi_pland_luhk_2024_raster_geotiff_zip" for item in downloads)
    assert len([item for item in downloads if item.source_id.endswith("_raster_geotiff_zip")]) == 7
    assert "https://www.landsd.gov.hk/landsd_psi_data/SMO/data/Whole_HK_DTM_5m.zip" in urls


def test_satellite_current_manifest_specs_cover_live_hko_products() -> None:
    source_ids = {source_id for source_id, _, _ in SATELLITE_CURRENT_IMAGE_SPECS}

    assert SATELLITE_CURRENT_IMAGE_BASE_URL.endswith("/wxinfo/intersat/satellite/image/images/")
    assert source_ids == {
        "hko_satellite_current_infrared_h8_image",
        "hko_satellite_current_deepconvection_fy4b_image",
        "hko_satellite_current_deepconvection_h8_image",
        "hko_satellite_current_infrared_fy4b_image",
        "hko_satellite_current_truecolour_h8_image",
        "hko_satellite_current_truecolour_fy4b_image",
        "hko_satellite_current_alldayvisible_h8_image",
        "hko_satellite_current_aerosolopticaldepth_gk2b_image",
    }


def test_satellite_filename_extraction_handles_current_and_legacy_manifests() -> None:
    js_text = """
    var infrared_h8_img=[["h8_ir_x2M_20260619085000.jpg",new Date(1781830200000)]];
    c_t_colour_img[0] = ["modis_HK_VIS_20260618.png", "MODIS"];
    var duplicate=[["h8_ir_x2M_20260619085000.jpg",new Date(1781830200000)]];
    """

    assert _extract_satellite_filenames(js_text) == [
        "h8_ir_x2M_20260619085000.jpg",
        "modis_HK_VIS_20260618.png",
    ]
    assert _satellite_image_extension("h8_ir_x2M_20260619085000.jpg") == "jpg"
    assert _satellite_image_extension("fy4b_dcred_WA_20260619084500.png") == "png"


def test_successful_archive_urls_include_request_and_final_urls(tmp_path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "retrieval_ledger.csv").write_text(
        "\n".join(
            [
                "retrieval_id,source_id,provider,retrieved_at,status,http_status,request_url,final_url,etag,last_modified,content_sha256,content_length,content_path,sidecar_path,deduplicated,error",
                "ok,source,provider,2026-06-19T00:00:00Z,success,200,https://example.com/request,https://example.com/final,,,,0,,,false,",
                "bad,source,provider,2026-06-19T00:00:00Z,error,,https://example.com/error,,,,,0,,,false,boom",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert _successful_archive_urls(tmp_path) == {
        "https://example.com/request",
        "https://example.com/final",
    }
