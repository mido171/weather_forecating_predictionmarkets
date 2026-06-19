from __future__ import annotations

import csv
import io
import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import httpx

from .acquisition import AcquisitionRecord, ensure_data_root, fetch_http_to_acquisition
from .fetch import FetchPolicy, httpx_verify_context

HKT = ZoneInfo("Asia/Hong_Kong")
HKO_PROVIDER = "Hong Kong Observatory"
LANDSD_PROVIDER = "Lands Department"
PLAND_PROVIDER = "Planning Department"
CSDI_PROVIDER = "Common Spatial Data Infrastructure Portal"
DATA_GOV_PROVIDER = "DATA.GOV.HK"


@dataclass(frozen=True)
class HkoDownload:
    source_id: str
    provider: str
    url: str
    extension: str
    description: str
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class HkoBackfillOutcome:
    requested: int
    succeeded: int
    failed: int
    records: tuple[AcquisitionRecord, ...]
    failures: tuple[str, ...]


@dataclass(frozen=True)
class DailyClimateElement:
    source_suffix: str
    page_code: str
    station_code: str
    download_code: str
    title: str
    start_year: int
    point_in_time_class: str


@dataclass(frozen=True)
class DataGovHistoricalFeed:
    source_suffix: str
    resource_url: str
    description: str
    start_date: str
    extension: str = "zip"
    family: str = "C_high_frequency_hko_regional_observations"
    point_in_time_class: str = "POTENTIAL_POINT_IN_TIME_ARCHIVE"


@dataclass(frozen=True)
class NoaaIsdStation:
    usaf: str
    wban: str
    name: str
    country: str
    icao: str
    latitude: str
    longitude: str
    elevation_m: str
    begin_year: int
    end_year: int


DAILY_CLIMATE_ELEMENTS: tuple[DailyClimateElement, ...] = (
    DailyClimateElement("mslp", "MSLP", "HKO", "MSLP", "Daily mean pressure", 1884, "PROXY_WITH_LIMITATIONS"),
    DailyClimateElement("mean_temperature", "TEMP", "HKO", "TEMP", "Daily mean temperature", 1884, "PROXY_WITH_LIMITATIONS"),
    DailyClimateElement("dew_point", "DEW_PT", "HKO", "DEW", "Daily mean dew point", 1961, "PROXY_WITH_LIMITATIONS"),
    DailyClimateElement("wet_bulb", "WET_BULB", "HKO", "WET", "Daily mean wet-bulb temperature", 1947, "PROXY_WITH_LIMITATIONS"),
    DailyClimateElement("relative_humidity", "RH", "HKO", "RH", "Daily mean relative humidity", 1947, "PROXY_WITH_LIMITATIONS"),
    DailyClimateElement("cloud_amount", "CLD", "HKO", "CLD", "Daily mean cloud amount", 1949, "PROXY_WITH_LIMITATIONS"),
    DailyClimateElement("rainfall", "RF", "HKO", "RF", "Daily total rainfall", 1884, "PROXY_WITH_LIMITATIONS"),
    DailyClimateElement("maximum_temperature", "MAX_TEMP", "HKO", "MAXT", "Daily maximum temperature", 1884, "PROXY_WITH_LIMITATIONS"),
    DailyClimateElement("minimum_temperature", "MIN_TEMP", "HKO", "MINT", "Daily minimum temperature", 1884, "PROXY_WITH_LIMITATIONS"),
    DailyClimateElement("bright_sunshine", "SUNSHINE", "KP", "SUN", "Daily total bright sunshine", 1961, "PROXY_WITH_LIMITATIONS"),
    DailyClimateElement("global_solar_radiation", "GLOBAL", "KP", "GSR", "Daily global solar radiation", 1978, "PROXY_WITH_LIMITATIONS"),
    DailyClimateElement("evaporation", "EVAPO", "KP", "EVAP", "Daily total evaporation", 1968, "PROXY_WITH_LIMITATIONS"),
    DailyClimateElement("lightning_ground", "LIGHT_GROUND", "HK", "LGTG", "Daily cloud-to-ground lightning strokes", 2005, "PROXY_WITH_LIMITATIONS"),
    DailyClimateElement("lightning_cloud", "LIGHT_CLOUD", "HK", "LGTC", "Daily cloud-to-cloud lightning strokes", 2005, "PROXY_WITH_LIMITATIONS"),
    DailyClimateElement("prevailing_wind_direction", "PREV_DIR", "WGL", "PDIR", "Daily prevailing wind direction", 1989, "PROXY_WITH_LIMITATIONS"),
    DailyClimateElement("mean_wind_speed", "MEAN_WIND", "WGL", "WSPD", "Daily mean wind speed", 1989, "PROXY_WITH_LIMITATIONS"),
    DailyClimateElement("grass_min_temperature", "GRASS", "HKO", "GMT", "Daily grass minimum temperature", 1968, "PROXY_WITH_LIMITATIONS"),
    DailyClimateElement("sea_temp_np_am", "SEATEMP_NP_AM", "NPF", "SSTA", "Daily North Point sea temperature AM", 1974, "PROXY_WITH_LIMITATIONS"),
    DailyClimateElement("sea_temp_np_pm", "SEATEMP_NP_PM", "NPF", "SSTP", "Daily North Point sea temperature PM", 1974, "PROXY_WITH_LIMITATIONS"),
    DailyClimateElement("sea_temp_waglan", "SEATEMP_WGL", "WGL", "SST", "Daily Waglan sea temperature", 1990, "PROXY_WITH_LIMITATIONS"),
    DailyClimateElement("reduced_visibility_hka", "VIS_HKA", "HKA", "RVIS", "Daily reduced visibility hours", 1997, "PROXY_WITH_LIMITATIONS"),
)


DATAGOV_HISTORICAL_LIVE_FEEDS: tuple[DataGovHistoricalFeed, ...] = (
    DataGovHistoricalFeed(
        "latest_1min_temperature",
        "https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/latest_1min_temperature.csv",
        "DATA.GOV.HK historical archives for HKO latest one-minute mean temperature",
        "20200601",
    ),
    DataGovHistoricalFeed(
        "latest_since_midnight_maxmin",
        "https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/latest_since_midnight_maxmin.csv",
        "DATA.GOV.HK historical archives for HKO latest since-midnight max/min temperature",
        "20200601",
    ),
    DataGovHistoricalFeed(
        "latest_1min_humidity",
        "https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/latest_1min_humidity.csv",
        "DATA.GOV.HK historical archives for HKO latest one-minute mean relative humidity",
        "20200601",
    ),
    DataGovHistoricalFeed(
        "latest_1min_pressure",
        "https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/latest_1min_pressure.csv",
        "DATA.GOV.HK historical archives for HKO latest one-minute mean sea-level pressure",
        "20210601",
    ),
    DataGovHistoricalFeed(
        "latest_10min_wind",
        "https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/latest_10min_wind.csv",
        "DATA.GOV.HK historical archives for HKO latest ten-minute wind",
        "20210601",
    ),
    DataGovHistoricalFeed(
        "latest_1min_solar",
        "https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/latest_1min_solar.csv",
        "DATA.GOV.HK historical archives for HKO latest one-minute solar radiation",
        "20210601",
    ),
    DataGovHistoricalFeed(
        "latest_15min_uvindex",
        "https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/latest_15min_uvindex.csv",
        "DATA.GOV.HK historical archives for HKO latest fifteen-minute UV index",
        "20200601",
    ),
)


DATAGOV_HISTORICAL_RSS_FEEDS: tuple[DataGovHistoricalFeed, ...] = (
    DataGovHistoricalFeed(
        "rss_current_weather_en",
        "https://rss.weather.gov.hk/rss/CurrentWeather.xml",
        "DATA.GOV.HK historical RSS archives for HKO current weather report English",
        "20200601",
        family="D_official_hko_forecast_vintages",
    ),
    DataGovHistoricalFeed(
        "rss_current_weather_tc",
        "https://rss.weather.gov.hk/rss/CurrentWeather_uc.xml",
        "DATA.GOV.HK historical RSS archives for HKO current weather report Traditional Chinese",
        "20200601",
        family="D_official_hko_forecast_vintages",
    ),
    DataGovHistoricalFeed(
        "rss_current_weather_sc",
        "https://rss.weather.gov.hk/sc/rss/CurrentWeather_uc.xml",
        "DATA.GOV.HK historical RSS archives for HKO current weather report Simplified Chinese",
        "20200601",
        family="D_official_hko_forecast_vintages",
    ),
    DataGovHistoricalFeed(
        "rss_local_forecast_en",
        "https://rss.weather.gov.hk/rss/LocalWeatherForecast.xml",
        "DATA.GOV.HK historical RSS archives for HKO local weather forecast English",
        "20200601",
        family="D_official_hko_forecast_vintages",
    ),
    DataGovHistoricalFeed(
        "rss_local_forecast_tc",
        "https://rss.weather.gov.hk/rss/LocalWeatherForecast_uc.xml",
        "DATA.GOV.HK historical RSS archives for HKO local weather forecast Traditional Chinese",
        "20200601",
        family="D_official_hko_forecast_vintages",
    ),
    DataGovHistoricalFeed(
        "rss_local_forecast_sc",
        "https://rss.weather.gov.hk/sc/rss/LocalWeatherForecast_uc.xml",
        "DATA.GOV.HK historical RSS archives for HKO local weather forecast Simplified Chinese",
        "20200601",
        family="D_official_hko_forecast_vintages",
    ),
    DataGovHistoricalFeed(
        "rss_9day_forecast_en",
        "https://rss.weather.gov.hk/rss/SeveralDaysWeatherForecast_v2.xml",
        "DATA.GOV.HK historical RSS archives for HKO 9-day weather forecast English",
        "20210401",
        family="D_official_hko_forecast_vintages",
    ),
    DataGovHistoricalFeed(
        "rss_9day_forecast_tc",
        "https://rss.weather.gov.hk/rss/SeveralDaysWeatherForecast_v2_uc.xml",
        "DATA.GOV.HK historical RSS archives for HKO 9-day weather forecast Traditional Chinese",
        "20210401",
        family="D_official_hko_forecast_vintages",
    ),
    DataGovHistoricalFeed(
        "rss_9day_forecast_sc",
        "https://rss.weather.gov.hk/sc/rss/SeveralDaysWeatherForecast_v2_uc.xml",
        "DATA.GOV.HK historical RSS archives for HKO 9-day weather forecast Simplified Chinese",
        "20210401",
        family="D_official_hko_forecast_vintages",
    ),
    DataGovHistoricalFeed(
        "rss_warning_bulletin_en",
        "https://rss.weather.gov.hk/rss/WeatherWarningBulletin.xml",
        "DATA.GOV.HK historical RSS archives for HKO weather warning bulletin English",
        "20200601",
        family="D_official_hko_forecast_vintages",
    ),
    DataGovHistoricalFeed(
        "rss_warning_bulletin_tc",
        "https://rss.weather.gov.hk/rss/WeatherWarningBulletin_uc.xml",
        "DATA.GOV.HK historical RSS archives for HKO weather warning bulletin Traditional Chinese",
        "20200601",
        family="D_official_hko_forecast_vintages",
    ),
    DataGovHistoricalFeed(
        "rss_warning_bulletin_sc",
        "https://rss.weather.gov.hk/sc/rss/WeatherWarningBulletin_uc.xml",
        "DATA.GOV.HK historical RSS archives for HKO weather warning bulletin Simplified Chinese",
        "20200601",
        family="D_official_hko_forecast_vintages",
    ),
    DataGovHistoricalFeed(
        "rss_warning_summary_en",
        "https://rss.weather.gov.hk/rss/WeatherWarningSummaryv2.xml",
        "DATA.GOV.HK historical RSS archives for HKO weather warning summary English",
        "20200601",
        family="D_official_hko_forecast_vintages",
    ),
    DataGovHistoricalFeed(
        "rss_warning_summary_tc",
        "https://rss.weather.gov.hk/rss/WeatherWarningSummaryv2_uc.xml",
        "DATA.GOV.HK historical RSS archives for HKO weather warning summary Traditional Chinese",
        "20200601",
        family="D_official_hko_forecast_vintages",
    ),
    DataGovHistoricalFeed(
        "rss_warning_summary_sc",
        "https://rss.weather.gov.hk/sc/rss/WeatherWarningSummaryv2_uc.xml",
        "DATA.GOV.HK historical RSS archives for HKO weather warning summary Simplified Chinese",
        "20200601",
        family="D_official_hko_forecast_vintages",
    ),
)


ARWF_METADATA_DOWNLOADS: tuple[HkoDownload, ...] = (
    HkoDownload(
        "hko_arwf_regional_portal_page",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/en/wxinfo/awsgis/regional_portal.html",
        "html",
        "HKO Automatic Regional Weather Forecast portal page",
        {"family": "D_official_hko_forecast_vintages", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "hko_arwf_remark_notes_page",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/en/wxinfo/awsgis/remark_notes.html",
        "html",
        "HKO Automatic Regional Weather Forecast remark notes",
        {"family": "D_official_hko_forecast_vintages", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "hko_arwf_map_config_js",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/en/wxinfo/awsgis/files/irwip-map-config.js?v=",
        "js",
        "HKO ARWF map configuration script with data endpoint constants",
        {"family": "D_official_hko_forecast_vintages", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "hko_arwf_station_config_aws_js",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/en/wxinfo/awsgis/files/irwip-station-config-aws.js?v=",
        "js",
        "HKO ARWF AWS station configuration script",
        {"family": "B_hko_station_metadata_history", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "hko_arwf_station_config_rmn_js",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/en/wxinfo/awsgis/files/station-config-rmn.js?v=",
        "js",
        "HKO radiation monitoring network station configuration script",
        {"family": "B_hko_station_metadata_history", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "hko_arwf_station_config_api_js",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/en/wxinfo/awsgis/files/irwip-station-config-api.js?v=",
        "js",
        "HKO ARWF station configuration helper script",
        {"family": "B_hko_station_metadata_history", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "hko_arwf_data_parser_js",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/en/wxinfo/awsgis/files/irwip-data-parser.js?v=",
        "js",
        "HKO ARWF source data parser script",
        {"family": "D_official_hko_forecast_vintages", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "hko_arwf_gis_common_js",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/en/wxinfo/awsgis/files/irwip-gis-common.js?v=",
        "js",
        "HKO ARWF GIS common script with forecast XML fallback rules",
        {"family": "D_official_hko_forecast_vintages", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "hko_arwf_portal_js",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/en/wxinfo/awsgis/files/irwip.js?v=",
        "js",
        "HKO ARWF portal script with current data and nowcast bundle references",
        {"family": "D_official_hko_forecast_vintages", "point_in_time_class": "METADATA"},
    ),
)


ARWF_CURRENT_DATA_DOWNLOADS: tuple[HkoDownload, ...] = (
    HkoDownload(
        "hko_arwf_latest_aws_readings",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/wxinfo/awsgis/latestReadings_AWS1_v2.txt",
        "txt",
        "HKO ARWF latest AWS readings used by the regional forecast portal",
        {
            "family": "C_high_frequency_hko_regional_observations",
            "point_in_time_class": "OPERATIONAL_POINT_IN_TIME",
        },
    ),
    HkoDownload(
        "hko_arwf_rmn_hourly_mean",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/wxinfo/awsgis/rmn_hourly_mean_used.txt",
        "txt",
        "HKO radiation monitoring network hourly mean readings used by the regional forecast portal",
        {
            "family": "C_high_frequency_hko_regional_observations",
            "point_in_time_class": "OPERATIONAL_POINT_IN_TIME",
        },
    ),
    HkoDownload(
        "hko_arwf_lightning_gis_latest",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/wxinfo/awsgis/gislatest_portal.txt",
        "txt",
        "HKO ARWF latest GIS lightning stroke data used by the regional forecast portal",
        {
            "family": "G_radar_rainfall_nowcasts_lightning",
            "point_in_time_class": "OPERATIONAL_POINT_IN_TIME",
        },
    ),
    HkoDownload(
        "hko_arwf_server_timestamp",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/wxinfo/awsgis/timestamp.txt",
        "txt",
        "HKO ARWF server timestamp",
        {"family": "D_official_hko_forecast_vintages", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "hko_arwf_alive_internet_portal_6h",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/wxinfo/awsgis/alive_internet_portal_6h.txt",
        "txt",
        "HKO ARWF six-hour portal alive/status series",
        {"family": "D_official_hko_forecast_vintages", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "hko_arwf_rainfall_nowcast_bundle",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/wxinfo/awsgis/forecast/rainfall.tar.gz",
        "tgz",
        "HKO ARWF rainfall and lightning nowcast tarball with index files and images",
        {
            "family": "G_radar_rainfall_nowcasts_lightning",
            "point_in_time_class": "OPERATIONAL_POINT_IN_TIME",
        },
    ),
    HkoDownload(
        "hko_arwf_nowcast_geojson_bundle",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/wxinfo/awsgis/forecast/geojson.tar.gz",
        "tgz",
        "HKO ARWF rainfall and lightning nowcast GeoJSON tarball",
        {
            "family": "G_radar_rainfall_nowcasts_lightning",
            "point_in_time_class": "OPERATIONAL_POINT_IN_TIME",
        },
    ),
    HkoDownload(
        "hko_arwf_radar_kml_64km",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/wxinfo/radars/R4_GIS_rad_064/R4_GIS_server_Radar_064.kml",
        "kml",
        "HKO ARWF 64 km radar KML overlay metadata",
        {
            "family": "G_radar_rainfall_nowcasts_lightning",
            "point_in_time_class": "OPERATIONAL_POINT_IN_TIME",
        },
    ),
    HkoDownload(
        "hko_arwf_radar_kml_128km",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/wxinfo/radars/R4_GIS_rad_128/R4_GIS_server_Radar_128.kml",
        "kml",
        "HKO ARWF 128 km radar KML overlay metadata",
        {
            "family": "G_radar_rainfall_nowcasts_lightning",
            "point_in_time_class": "OPERATIONAL_POINT_IN_TIME",
        },
    ),
    HkoDownload(
        "hko_arwf_radar_kml_256km",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/wxinfo/radars/R4_GIS_rad_256/R4_GIS_server_Radar_256.kml",
        "kml",
        "HKO ARWF 256 km radar KML overlay metadata",
        {
            "family": "G_radar_rainfall_nowcasts_lightning",
            "point_in_time_class": "OPERATIONAL_POINT_IN_TIME",
        },
    ),
)


ARWF_ANIMATION_DOWNLOADS: tuple[HkoDownload, ...] = (
    HkoDownload(
        "hko_arwf_yesterday_max_temperature_animation",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/wxinfo/awsgis/animate_J1+MAXIMID_yesterday.csv",
        "csv",
        "HKO ARWF yesterday maximum temperature animation source",
        {
            "family": "C_high_frequency_hko_regional_observations",
            "point_in_time_class": "OPERATIONAL_POINT_IN_TIME",
        },
    ),
    HkoDownload(
        "hko_arwf_yesterday_min_temperature_animation",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/wxinfo/awsgis/animate_J1+MINUMID_yesterday.csv",
        "csv",
        "HKO ARWF yesterday minimum temperature animation source",
        {
            "family": "C_high_frequency_hko_regional_observations",
            "point_in_time_class": "OPERATIONAL_POINT_IN_TIME",
        },
    ),
    HkoDownload(
        "hko_arwf_past_wind_animation",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/wxinfo/awsgis/animate_hrwind.csv",
        "csv",
        "HKO ARWF past hourly wind animation source",
        {
            "family": "C_high_frequency_hko_regional_observations",
            "point_in_time_class": "OPERATIONAL_POINT_IN_TIME",
        },
    ),
    HkoDownload(
        "hko_arwf_past_relative_humidity_animation",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/wxinfo/awsgis/animate_rh.csv",
        "csv",
        "HKO ARWF past relative humidity animation source",
        {
            "family": "C_high_frequency_hko_regional_observations",
            "point_in_time_class": "OPERATIONAL_POINT_IN_TIME",
        },
    ),
    HkoDownload(
        "hko_arwf_past_visibility_animation",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/wxinfo/awsgis/animate_m1.csv",
        "csv",
        "HKO ARWF past visibility animation source",
        {
            "family": "C_high_frequency_hko_regional_observations",
            "point_in_time_class": "OPERATIONAL_POINT_IN_TIME",
        },
    ),
    HkoDownload(
        "hko_arwf_past_mslp_animation",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/wxinfo/awsgis/animate_S1.csv",
        "csv",
        "HKO ARWF past mean sea-level pressure animation source",
        {
            "family": "C_high_frequency_hko_regional_observations",
            "point_in_time_class": "OPERATIONAL_POINT_IN_TIME",
        },
    ),
    HkoDownload(
        "hko_arwf_past_heat_index_animation",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/wxinfo/awsgis/animate_hi2.csv",
        "csv",
        "HKO ARWF past Hong Kong heat index animation source",
        {
            "family": "C_high_frequency_hko_regional_observations",
            "point_in_time_class": "OPERATIONAL_POINT_IN_TIME",
        },
    ),
    HkoDownload(
        "hko_arwf_past_temperature_animation",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/wxinfo/awsgis/animate_J1.csv",
        "csv",
        "HKO ARWF past temperature animation source",
        {
            "family": "C_high_frequency_hko_regional_observations",
            "point_in_time_class": "OPERATIONAL_POINT_IN_TIME",
        },
    ),
    HkoDownload(
        "hko_arwf_past_wind_gust_animation",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/wxinfo/awsgis/animate_F1.csv",
        "csv",
        "HKO ARWF past wind gust animation source",
        {
            "family": "C_high_frequency_hko_regional_observations",
            "point_in_time_class": "OPERATIONAL_POINT_IN_TIME",
        },
    ),
)


NOAA_ISD_METADATA_DOWNLOADS: tuple[HkoDownload, ...] = (
    HkoDownload(
        "noaa_isd_history",
        "NOAA NCEI",
        "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv",
        "csv",
        "NOAA ISD station history catalog",
        {
            "family": "I_tropical_cyclone_monsoon_synoptic_information",
            "point_in_time_class": "METADATA",
        },
    ),
    HkoDownload(
        "noaa_isd_inventory",
        "NOAA NCEI",
        "https://www.ncei.noaa.gov/pub/data/noaa/isd-inventory.csv",
        "csv",
        "NOAA ISD station-year/month inventory",
        {
            "family": "I_tropical_cyclone_monsoon_synoptic_information",
            "point_in_time_class": "METADATA",
        },
    ),
    HkoDownload(
        "noaa_isd_readme",
        "NOAA NCEI",
        "https://www.ncei.noaa.gov/pub/data/noaa/readme.txt",
        "txt",
        "NOAA ISD readme",
        {
            "family": "I_tropical_cyclone_monsoon_synoptic_information",
            "point_in_time_class": "METADATA",
        },
    ),
    HkoDownload(
        "noaa_isd_format_document",
        "NOAA NCEI",
        "https://www.ncei.noaa.gov/pub/data/noaa/isd-format-document.pdf",
        "pdf",
        "NOAA ISD format document",
        {
            "family": "I_tropical_cyclone_monsoon_synoptic_information",
            "point_in_time_class": "METADATA",
        },
    ),
)


NCEP_GFS_LEVEL_PARAMS: tuple[str, ...] = (
    "lev_2_m_above_ground",
    "lev_10_m_above_ground",
    "lev_mean_sea_level",
    "lev_surface",
    "lev_1000_mb",
    "lev_975_mb",
    "lev_950_mb",
    "lev_925_mb",
    "lev_900_mb",
    "lev_850_mb",
    "lev_700_mb",
    "lev_500_mb",
)

NCEP_GFS_VARIABLE_PARAMS: tuple[str, ...] = (
    "var_TMP",
    "var_DPT",
    "var_RH",
    "var_UGRD",
    "var_VGRD",
    "var_GUST",
    "var_PRMSL",
    "var_APCP",
    "var_TCDC",
    "var_DSWRF",
    "var_DLWRF",
    "var_CAPE",
    "var_HPBL",
    "var_HGT",
    "var_VVEL",
)

NCEP_GEFS_LEVEL_PARAMS: tuple[str, ...] = (
    "lev_2_m_above_ground",
    "lev_10_m_above_ground",
    "lev_mean_sea_level",
    "lev_surface",
    "lev_1000_mb",
    "lev_925_mb",
    "lev_850_mb",
    "lev_700_mb",
    "lev_500_mb",
)

NCEP_GEFS_VARIABLE_PARAMS: tuple[str, ...] = (
    "var_TMP",
    "var_RH",
    "var_UGRD",
    "var_VGRD",
    "var_PRMSL",
    "var_APCP",
    "var_TCDC",
    "var_HGT",
)

NCEP_FORECAST_HOURS_6H_TO_120H: tuple[int, ...] = tuple(range(6, 121, 6))
NCEP_HK_REGIONAL_DOMAIN: Mapping[str, str] = {
    "subregion": "",
    "leftlon": "112",
    "rightlon": "116",
    "toplat": "25",
    "bottomlat": "21",
}
NCEP_GEFS_MEMBERS: tuple[str, ...] = (
    "gec00",
    *(f"gep{member:02d}" for member in range(1, 31)),
    "geavg",
    "gespr",
)

NCEP_METADATA_DOWNLOADS: tuple[HkoDownload, ...] = (
    HkoDownload(
        "ncep_gfs_nomads_filter_page",
        "NOAA NCEP NOMADS",
        "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl",
        "html",
        "NOAA NOMADS GFS 0.25 degree filter page",
        {
            "family": "E_operational_numerical_ai_forecast_archives",
            "point_in_time_class": "METADATA",
        },
    ),
    HkoDownload(
        "ncep_gefs_nomads_filter_page",
        "NOAA NCEP NOMADS",
        "https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p50a.pl",
        "html",
        "NOAA NOMADS GEFS 0.50 degree atmospheric filter page",
        {
            "family": "E_operational_numerical_ai_forecast_archives",
            "point_in_time_class": "METADATA",
        },
    ),
)


DISCOVERED_HKO_FEEDS: tuple[HkoDownload, ...] = (
    HkoDownload(
        "hko_api_documentation_pdf",
        HKO_PROVIDER,
        "https://data.weather.gov.hk/weatherAPI/doc/HKO_Open_Data_API_Documentation.pdf",
        "pdf",
        "Official HKO Open Data API documentation",
        {"family": "B_hko_station_metadata_history", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "hko_latest_relative_humidity",
        HKO_PROVIDER,
        "https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/latest_1min_humidity.csv",
        "csv",
        "Latest one-minute mean relative humidity feed",
        {"family": "C_high_frequency_hko_regional_observations", "cadence": "approximately_10_minutes"},
    ),
    HkoDownload(
        "hko_latest_pressure",
        HKO_PROVIDER,
        "https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/latest_1min_pressure.csv",
        "csv",
        "Latest one-minute mean sea-level pressure feed",
        {"family": "C_high_frequency_hko_regional_observations", "cadence": "approximately_10_minutes"},
    ),
    HkoDownload(
        "hko_latest_wind",
        HKO_PROVIDER,
        "https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/latest_10min_wind.csv",
        "csv",
        "Latest ten-minute wind direction, speed and gust feed",
        {"family": "C_high_frequency_hko_regional_observations", "cadence": "approximately_10_minutes"},
    ),
    HkoDownload(
        "hko_latest_solar_radiation",
        HKO_PROVIDER,
        "https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/latest_1min_solar.csv",
        "csv",
        "Latest one-minute solar radiation feed",
        {"family": "C_high_frequency_hko_regional_observations", "cadence": "approximately_10_minutes"},
    ),
    HkoDownload(
        "hko_latest_uv_index",
        HKO_PROVIDER,
        "https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/latest_15min_uvindex.csv",
        "csv",
        "Latest fifteen-minute mean UV index feed",
        {"family": "C_high_frequency_hko_regional_observations", "cadence": "approximately_15_minutes"},
    ),
    HkoDownload(
        "hko_automatic_rainfall",
        HKO_PROVIDER,
        "https://data.weather.gov.hk/weatherAPI/opendata/hourlyRainfall.php?lang=en",
        "json",
        "Rainfall in the past hour from automatic weather stations",
        {"family": "C_high_frequency_hko_regional_observations", "cadence": "approximately_15_minutes"},
    ),
    HkoDownload(
        "hko_latest_visibility",
        HKO_PROVIDER,
        "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=LTMV&lang=en&rformat=csv",
        "csv",
        "Regional latest ten-minute mean visibility feed",
        {"family": "C_high_frequency_hko_regional_observations", "cadence": "approximately_10_minutes"},
    ),
    HkoDownload(
        "hko_current_weather_report",
        HKO_PROVIDER,
        "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=rhrread&lang=en",
        "json",
        "Current weather report with regional observations",
        {"family": "C_high_frequency_hko_regional_observations", "cadence": "provider_updated"},
    ),
    HkoDownload(
        "hko_weather_warning_summary",
        HKO_PROVIDER,
        "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=warnsum&lang=en",
        "json",
        "Weather warning summary",
        {"family": "D_official_hko_forecast_vintages", "cadence": "warning_update"},
    ),
    HkoDownload(
        "hko_weather_warning_information",
        HKO_PROVIDER,
        "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=warningInfo&lang=en",
        "json",
        "Weather warning information",
        {"family": "D_official_hko_forecast_vintages", "cadence": "warning_update"},
    ),
    HkoDownload(
        "hko_special_weather_tips",
        HKO_PROVIDER,
        "https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=swt&lang=en",
        "json",
        "Special weather tips",
        {"family": "D_official_hko_forecast_vintages", "cadence": "provider_updated"},
    ),
    HkoDownload(
        "hko_gridded_rainfall_nowcast",
        HKO_PROVIDER,
        "https://data.weather.gov.hk/weatherAPI/hko_data/F3/Gridded_rainfall_nowcast.csv",
        "csv",
        "Gridded rainfall nowcast over Hong Kong",
        {"family": "G_radar_rainfall_nowcasts_lightning", "cadence": "approximately_12_minutes"},
    ),
    HkoDownload(
        "hko_tropical_cyclone_track_realtime",
        HKO_PROVIDER,
        "https://www.weather.gov.hk/wxinfo/currwx/tc_list.xml",
        "xml",
        "Realtime tropical cyclone track list",
        {"family": "I_tropical_cyclone_monsoon_synoptic", "cadence": "advisory_specific"},
    ),
    HkoDownload(
        "hko_south_china_coastal_waters_bulletin",
        HKO_PROVIDER,
        "https://data.weather.gov.hk/openData/json/sccw_json_datagov.json",
        "json",
        "South China coastal waters bulletin",
        {"family": "J_marine_ocean_surface_state", "cadence": "provider_updated"},
    ),
    HkoDownload(
        "hko_latest_tidal_information",
        HKO_PROVIDER,
        "https://data.weather.gov.hk/weatherAPI/hko_data/tide/ALL_en.csv",
        "csv",
        "Latest tide information for all listed stations",
        {"family": "J_marine_ocean_surface_state", "cadence": "provider_updated"},
    ),
)


UPPER_AIR_DOWNLOADS: tuple[HkoDownload, ...] = (
    HkoDownload(
        "noaa_igra_station_list",
        "NOAA NCEI",
        "https://www.ncei.noaa.gov/pub/data/igra/igra2-station-list.txt",
        "txt",
        "NOAA IGRA2 station list; HKM00045004 resolves to Kowloon / King's Park",
        {"family": "F_upper_air_vertical_profile_observations", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "noaa_igra_list_format",
        "NOAA NCEI",
        "https://www.ncei.noaa.gov/pub/data/igra/igra2-list-format.txt",
        "txt",
        "NOAA IGRA2 station-list format documentation",
        {"family": "F_upper_air_vertical_profile_observations", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "noaa_igra_product_description",
        "NOAA NCEI",
        "https://www.ncei.noaa.gov/pub/data/igra/igra2-product-description.pdf",
        "pdf",
        "NOAA IGRA2 product description",
        {"family": "F_upper_air_vertical_profile_observations", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "noaa_igra_readme",
        "NOAA NCEI",
        "https://www.ncei.noaa.gov/pub/data/igra/igra2-readme.txt",
        "txt",
        "NOAA IGRA2 readme",
        {"family": "F_upper_air_vertical_profile_observations", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "noaa_igra_hkm00045004_period_of_record",
        "NOAA NCEI",
        "https://www.ncei.noaa.gov/pub/data/igra/data/data-por/HKM00045004-data.txt.zip",
        "zip",
        "NOAA IGRA2 Kowloon / King's Park full period-of-record sounding archive",
        {
            "family": "F_upper_air_vertical_profile_observations",
            "station_id": "HKM00045004",
            "station_name": "KOWLOON (45004-0)",
            "point_in_time_class": "PROXY_WITH_LIMITATIONS",
        },
    ),
    HkoDownload(
        "noaa_igra_hkm00045004_year_to_date",
        "NOAA NCEI",
        "https://www.ncei.noaa.gov/pub/data/igra/data/data-y2d/HKM00045004-data-beg2025.txt.zip",
        "zip",
        "NOAA IGRA2 Kowloon / King's Park year-to-date sounding archive beginning 2025",
        {
            "family": "F_upper_air_vertical_profile_observations",
            "station_id": "HKM00045004",
            "station_name": "KOWLOON (45004-0)",
            "point_in_time_class": "OPERATIONAL_POINT_IN_TIME",
        },
    ),
)


RADAR_LIGHTNING_BASE_DOWNLOADS: tuple[HkoDownload, ...] = (
    HkoDownload(
        "hko_radar_page_64km",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/en/wxinfo/radars/radar_range1.htm",
        "html",
        "HKO radar 64 km range page",
        {"family": "G_radar_rainfall_nowcasts_lightning", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "hko_radar_image_manifest",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/wxinfo/radars/temp_json/nradar_img.json",
        "json",
        "HKO radar current image manifest",
        {"family": "G_radar_rainfall_nowcasts_lightning", "point_in_time_class": "OPERATIONAL_POINT_IN_TIME"},
    ),
    HkoDownload(
        "hko_lightning_page",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/en/wxinfo/llis/gm_index.htm",
        "html",
        "HKO lightning location information page",
        {"family": "G_radar_rainfall_nowcasts_lightning", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "hko_lightning_counts_latest",
        HKO_PROVIDER,
        "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=LHL&lang=en&rformat=csv",
        "csv",
        "HKO cloud-to-ground and cloud-to-cloud lightning count in the past hour",
        {"family": "G_radar_rainfall_nowcasts_lightning", "point_in_time_class": "OPERATIONAL_POINT_IN_TIME"},
    ),
    HkoDownload(
        "hko_lightning_area_counts_page",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/en/wxinfo/llis/gm/llis_area_counts.htm",
        "html",
        "HKO recent cloud-to-ground lightning count distribution page",
        {"family": "G_radar_rainfall_nowcasts_lightning", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "hko_lightning_past_hour_page",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/en/wxinfo/llis/gm/llis_past_hr.htm",
        "html",
        "HKO cloud-to-ground lightning count distribution over the past hour page",
        {"family": "G_radar_rainfall_nowcasts_lightning", "point_in_time_class": "METADATA"},
    ),
)


SATELLITE_MANIFEST_DOWNLOADS: tuple[HkoDownload, ...] = (
    HkoDownload(
        "hko_satellite_page",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/en/wxinfo/intersat/satellite/sate.htm",
        "html",
        "HKO weather satellite imagery page",
        {"family": "H_satellite_cloud_aerosol_observations", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "hko_satellite_himawari_infrared_manifest",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/wxinfo/intersat/satellite/image/data/e_infrared_nh8_I.js",
        "js",
        "HKO Himawari infrared image manifest",
        {"family": "H_satellite_cloud_aerosol_observations", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "hko_satellite_himawari_true_colour_manifest",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/wxinfo/intersat/satellite/image/data/e_true_nh8_I.js",
        "js",
        "HKO Himawari true-colour image manifest",
        {"family": "H_satellite_cloud_aerosol_observations", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "hko_satellite_modis_true_colour_manifest",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/wxinfo/intersat/satellite/image/data/modis_HK_VIS.js",
        "js",
        "HKO MODIS Hong Kong true-colour image manifest",
        {"family": "H_satellite_cloud_aerosol_observations", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "hko_satellite_modis_aod_manifest",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/wxinfo/intersat/satellite/image/data/modis_HK_V2_AOD.js",
        "js",
        "HKO MODIS Hong Kong aerosol optical depth image manifest",
        {"family": "H_satellite_cloud_aerosol_observations", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "hko_satellite_modis_sst_manifest",
        HKO_PROVIDER,
        "https://www.hko.gov.hk/wxinfo/intersat/satellite/image/data/modis_HKS_SST.js",
        "js",
        "HKO MODIS Hong Kong sea-surface-temperature image manifest",
        {"family": "H_satellite_cloud_aerosol_observations", "point_in_time_class": "METADATA"},
    ),
)


STATIC_CONTEXT_BASE_DOWNLOADS: tuple[HkoDownload, ...] = (
    HkoDownload(
        "landsd_open_data_geospatial_page",
        LANDSD_PROVIDER,
        "https://www.landsd.gov.hk/en/spatial-data/open-data.html",
        "html",
        "Lands Department geospatial open-data source page",
        {"family": "L_static_geospatial_deterministic_context", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "data_gov_hk_landsd_dtm_dataset_page",
        DATA_GOV_PROVIDER,
        "https://data.gov.hk/en-data/dataset/hk-landsd-openmap-5m-grid-dtm",
        "html",
        "DATA.GOV.HK Digital Terrain Model dataset page",
        {"family": "L_static_geospatial_deterministic_context", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "csdi_geospatial_services_documentation",
        CSDI_PROVIDER,
        "https://portal.csdi.gov.hk/csdi-webpage/doc/GeoSpatialServices",
        "html",
        "CSDI geospatial service documentation",
        {"family": "L_static_geospatial_deterministic_context", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "csdi_landsd_dtm_dataset_page",
        CSDI_PROVIDER,
        "https://portal.csdi.gov.hk/csdi-webpage/dataset/landsd_rcd_1638158088368_93806",
        "html",
        "CSDI LandsD Digital Terrain Model dataset page",
        {"family": "L_static_geospatial_deterministic_context", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "csdi_landsd_dtm_metadata_xml",
        CSDI_PROVIDER,
        "https://portal.csdi.gov.hk/csdi-webpage/metadata/landsd_rcd_1638158088368_93806",
        "xml",
        "CSDI ISO metadata for LandsD Digital Terrain Model",
        {"family": "L_static_geospatial_deterministic_context", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "csdi_landsd_dtm_geotiff_zip",
        CSDI_PROVIDER,
        "https://static.csdi.gov.hk/csdi-webpage/download/common/e9a836d098a3a9ccf55a7f61f69a93e71923c722fb94ddf7b36be83a67b99140",
        "zip",
        "CSDI GeoTIFF package for LandsD Digital Terrain Model",
        {
            "family": "L_static_geospatial_deterministic_context",
            "point_in_time_class": "STATIC_CONTEXT_VERSIONED",
            "variable": "terrain_elevation",
            "grid": "Hong Kong 5 m DTM",
        },
    ),
    HkoDownload(
        "csdi_landsd_dtm_supporting_document_pdf_1",
        CSDI_PROVIDER,
        "https://static.csdi.gov.hk/csdi-webpage/download/common/fb780f95f32ee58f2d58dd41152e9fb833d35b2f73a3ae5000145092fafd26ce",
        "pdf",
        "CSDI supporting document for LandsD Digital Terrain Model",
        {"family": "L_static_geospatial_deterministic_context", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "csdi_landsd_dtm_supporting_document_pdf_2",
        CSDI_PROVIDER,
        "https://static.csdi.gov.hk/csdi-webpage/download/common/9d38d3b5ab4a73a8abdf42918cd71b38a9bea54c14c17b1234bb444d9b70445d",
        "pdf",
        "CSDI supporting document for LandsD Digital Terrain Model",
        {"family": "L_static_geospatial_deterministic_context", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "csdi_landsd_dtm_supporting_document_pdf_3",
        CSDI_PROVIDER,
        "https://static.csdi.gov.hk/csdi-webpage/download/common/61f8e4893aa81043a8b3e2e3c56eb7782da6dd6aec1e37d33f4bca54955a2067",
        "pdf",
        "CSDI supporting document for LandsD Digital Terrain Model",
        {"family": "L_static_geospatial_deterministic_context", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "landsd_whole_hk_dtm_5m_asc_zip",
        LANDSD_PROVIDER,
        "https://www.landsd.gov.hk/landsd_psi_data/SMO/data/Whole_HK_DTM_5m.zip",
        "zip",
        "LandsD Whole Hong Kong 5 m Digital Terrain Model ASC package",
        {
            "family": "L_static_geospatial_deterministic_context",
            "point_in_time_class": "STATIC_CONTEXT_VERSIONED",
            "variable": "terrain_elevation",
            "grid": "Hong Kong 5 m DTM",
        },
    ),
    HkoDownload(
        "landsd_3d_spatial_data_revision_csv",
        LANDSD_PROVIDER,
        "https://www.landsd.gov.hk/doc/en/mapping/digital-map/common/update/3d_update.csv",
        "csv",
        "LandsD 3D spatial data revision-date table",
        {"family": "L_static_geospatial_deterministic_context", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "landsd_topographic_ib1000_revision_csv",
        LANDSD_PROVIDER,
        "https://www.landsd.gov.hk/doc/en/mapping/digital-map/common/update/ib1_update.csv",
        "csv",
        "LandsD iB1000 digital topographic map revision-date table",
        {"family": "L_static_geospatial_deterministic_context", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "landsd_topographic_ib5000_revision_csv",
        LANDSD_PROVIDER,
        "https://www.landsd.gov.hk/doc/en/mapping/digital-map/common/update/ib5_update.csv",
        "csv",
        "LandsD iB5000 digital topographic map revision-date table",
        {"family": "L_static_geospatial_deterministic_context", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "landsd_topographic_ib10000_revision_csv",
        LANDSD_PROVIDER,
        "https://www.landsd.gov.hk/doc/en/mapping/digital-map/common/update/ib10_update.csv",
        "csv",
        "LandsD iB10000 digital topographic map revision-date table",
        {"family": "L_static_geospatial_deterministic_context", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "landsd_topographic_ib20000_revision_csv",
        LANDSD_PROVIDER,
        "https://www.landsd.gov.hk/doc/en/mapping/digital-map/common/update/ib20_update.csv",
        "csv",
        "LandsD iB20000 digital topographic map revision-date table",
        {"family": "L_static_geospatial_deterministic_context", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "landsd_land_boundary_ic1000_revision_csv",
        LANDSD_PROVIDER,
        "https://www.landsd.gov.hk/doc/en/mapping/digital-map/common/update/ic1_update.csv",
        "csv",
        "LandsD iC1000 digital land boundary map revision-date table",
        {"family": "L_static_geospatial_deterministic_context", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "landsd_georeference_ig1000_revision_csv",
        LANDSD_PROVIDER,
        "https://www.landsd.gov.hk/doc/en/mapping/digital-map/common/update/ig1_update.csv",
        "csv",
        "LandsD iG1000 geo-reference database revision-date table",
        {"family": "L_static_geospatial_deterministic_context", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "landsd_igeocom_revision_csv",
        LANDSD_PROVIDER,
        "https://www.landsd.gov.hk/doc/en/mapping/digital-map/common/update/igeocom_update.csv",
        "csv",
        "LandsD iGeoCommunity database revision-date table",
        {"family": "L_static_geospatial_deterministic_context", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "landsd_topographic_ib50000_gml_zip",
        LANDSD_PROVIDER,
        "https://www.landsd.gov.hk/landsd_psi_data/SMO/data/iB50000GML.zip",
        "zip",
        "LandsD 1:50 000 digital topographic map GML package",
        {
            "family": "L_static_geospatial_deterministic_context",
            "point_in_time_class": "STATIC_CONTEXT_VERSIONED",
            "scale": "1:50000",
        },
    ),
    HkoDownload(
        "landsd_topographic_ib100000_gml_zip",
        LANDSD_PROVIDER,
        "https://www.landsd.gov.hk/landsd_psi_data/SMO/data/iB100000GML.zip",
        "zip",
        "LandsD 1:100 000 digital topographic map GML package",
        {
            "family": "L_static_geospatial_deterministic_context",
            "point_in_time_class": "STATIC_CONTEXT_VERSIONED",
            "scale": "1:100000",
        },
    ),
    HkoDownload(
        "landsd_topographic_ib200000_gml_zip",
        LANDSD_PROVIDER,
        "https://www.landsd.gov.hk/landsd_psi_data/SMO/data/iB200000GML.zip",
        "zip",
        "LandsD 1:200 000 digital topographic map GML package",
        {
            "family": "L_static_geospatial_deterministic_context",
            "point_in_time_class": "STATIC_CONTEXT_VERSIONED",
            "scale": "1:200000",
        },
    ),
    HkoDownload(
        "landsd_topographic_b50000_geotiff",
        LANDSD_PROVIDER,
        "https://www.landsd.gov.hk/landsd_psi_data/SMO/image/B50K_R200index-geo.tif",
        "tif",
        "LandsD 1:50 000 georeferenced topographic map GeoTIFF",
        {
            "family": "L_static_geospatial_deterministic_context",
            "point_in_time_class": "STATIC_CONTEXT_VERSIONED",
            "scale": "1:50000",
        },
    ),
    HkoDownload(
        "landsd_topographic_b100000_2017_geotiff",
        LANDSD_PROVIDER,
        "https://www.landsd.gov.hk/landsd_psi_data/SMO/image/B100K_R200index-geo_Ed2017.tif",
        "tif",
        "LandsD 1:100 000 georeferenced topographic map GeoTIFF 2017 edition",
        {
            "family": "L_static_geospatial_deterministic_context",
            "point_in_time_class": "STATIC_CONTEXT_VERSIONED",
            "scale": "1:100000",
            "edition": "2017",
        },
    ),
    HkoDownload(
        "landsd_topographic_b100000_2018_geotiff",
        LANDSD_PROVIDER,
        "https://www.landsd.gov.hk/landsd_psi_data/SMO/image/B100K_R200index-geo_Ed2018.tif",
        "tif",
        "LandsD 1:100 000 georeferenced topographic map GeoTIFF 2018 edition",
        {
            "family": "L_static_geospatial_deterministic_context",
            "point_in_time_class": "STATIC_CONTEXT_VERSIONED",
            "scale": "1:100000",
            "edition": "2018",
        },
    ),
    HkoDownload(
        "landsd_topographic_b200000_geotiff",
        LANDSD_PROVIDER,
        "https://www.landsd.gov.hk/landsd_psi_data/SMO/image/B200K_R500index-geo.tif",
        "tif",
        "LandsD 1:200 000 georeferenced topographic map GeoTIFF",
        {
            "family": "L_static_geospatial_deterministic_context",
            "point_in_time_class": "STATIC_CONTEXT_VERSIONED",
            "scale": "1:200000",
        },
    ),
    HkoDownload(
        "landsd_igeocom_csv_zip",
        LANDSD_PROVIDER,
        "https://open.hkmapservice.gov.hk/OpenData/directDownload?productName=iGeoCom&sheetName=iGeoCom&productFormat=CSV",
        "zip",
        "LandsD iGeoCommunity database CSV package",
        {
            "family": "L_static_geospatial_deterministic_context",
            "point_in_time_class": "STATIC_CONTEXT_VERSIONED",
            "product": "iGeoCom",
        },
    ),
    HkoDownload(
        "landsd_igeocom_geojson_zip",
        LANDSD_PROVIDER,
        "https://open.hkmapservice.gov.hk/OpenData/directDownload?productName=iGeoCom&sheetName=iGeoCom&productFormat=GEOJSON",
        "zip",
        "LandsD iGeoCommunity database GeoJSON package",
        {
            "family": "L_static_geospatial_deterministic_context",
            "point_in_time_class": "STATIC_CONTEXT_VERSIONED",
            "product": "iGeoCom",
        },
    ),
    HkoDownload(
        "pland_open_data_page",
        PLAND_PROVIDER,
        "https://www.pland.gov.hk/pland_en/resources/info_serv/open_data/index.html",
        "html",
        "Planning Department open-data source page",
        {"family": "L_static_geospatial_deterministic_context", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "pland_land_utilization_page",
        PLAND_PROVIDER,
        "https://www.pland.gov.hk/pland_en/info_serv/open_data/landu/",
        "html",
        "Planning Department Land Utilization in Hong Kong source page",
        {"family": "L_static_geospatial_deterministic_context", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "data_gov_hk_pland_luhk_raster_grid_page",
        DATA_GOV_PROVIDER,
        "https://data.gov.hk/en-data/dataset/hk-pland-pland1-land-utilization-in-hong-kong-raster-grid",
        "html",
        "DATA.GOV.HK LUHK raster-grid dataset page",
        {"family": "L_static_geospatial_deterministic_context", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "data_gov_hk_pland_luhk_statistics_page",
        DATA_GOV_PROVIDER,
        "https://data.gov.hk/en-data/dataset/hk-pland-pland1-land-utilization-in-hong-kong-statistics",
        "html",
        "DATA.GOV.HK LUHK statistics dataset page",
        {"family": "L_static_geospatial_deterministic_context", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "pland_3d_photo_realistic_grid_index_cesium_csv",
        PLAND_PROVIDER,
        "https://pdmap.pland.gov.hk/PLANDWEB/public/3d_photo_realistic_models/Metadata/GridIdx_CESIUM.csv",
        "csv",
        "Planning Department 3D photo-realistic model Cesium tile grid index",
        {"family": "L_static_geospatial_deterministic_context", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "pland_3d_photo_realistic_grid_index_obj_csv",
        PLAND_PROVIDER,
        "https://pdmap.pland.gov.hk/PLANDWEB/public/3d_photo_realistic_models/Metadata/GridIdx_OBJ.csv",
        "csv",
        "Planning Department 3D photo-realistic model OBJ tile grid index",
        {"family": "L_static_geospatial_deterministic_context", "point_in_time_class": "METADATA"},
    ),
    HkoDownload(
        "pland_3d_photo_realistic_grid_index_osgb_csv",
        PLAND_PROVIDER,
        "https://pdmap.pland.gov.hk/PLANDWEB/public/3d_photo_realistic_models/Metadata/GridIdx_OSGB.csv",
        "csv",
        "Planning Department 3D photo-realistic model OSGB tile grid index",
        {"family": "L_static_geospatial_deterministic_context", "point_in_time_class": "METADATA"},
    ),
)


PLAND_LUHK_RASTER_SOURCES: tuple[tuple[int, str, str], ...] = (
    (
        2018,
        "pland_rcd_1634281791890_21963",
        "https://static.csdi.gov.hk/csdi-webpage/download/common/0044b949d744ed5ffa8ed21ac21a033cdf14ac6f1e576f1e9443d8ab0f8991c4",
    ),
    (
        2019,
        "pland_rcd_1634282086807_94936",
        "https://static.csdi.gov.hk/csdi-webpage/download/common/3003f406d1da89bc8ccc6c2478fd33aaa7b9de2d1fda272aef552f91af69158f",
    ),
    (
        2020,
        "pland_rcd_1634282545137_98227",
        "https://static.csdi.gov.hk/csdi-webpage/download/common/0f7706f12b120946ce2a4bafcedcb4dd6f004f4b63dde337410f09214262ffd2",
    ),
    (
        2021,
        "pland_rcd_1634024370851_27769",
        "https://static.csdi.gov.hk/csdi-webpage/download/common/33c4bb4d4a0cac30c5afa16512ae0c474d16ab295bc3a0d48a145bd0d8ed95de",
    ),
    (
        2022,
        "pland_rcd_1665742315124_26227",
        "https://static.csdi.gov.hk/csdi-webpage/download/common/4993f93e28e19fb4dc0dca9a672dfc4254bc0a88a094a5a0464d30e637587d5c",
    ),
    (
        2023,
        "pland_rcd_1696577406166_85973",
        "https://static.csdi.gov.hk/csdi-webpage/download/common/8c59922e3882395ba69a87af722119534d6e6723202476e84d99c56c394a7224",
    ),
    (
        2024,
        "pland_rcd_1725865972233_20687",
        "https://static.csdi.gov.hk/csdi-webpage/download/common/ca6cfc600335247dcf833e558efc09962f1243c2f33e8b32438256cc1eea1388",
    ),
)


PLAND_LUHK_STATISTICS_YEARS: tuple[int, ...] = (2022, 2023, 2024)
PLAND_LUHK_STATISTICS_DESCRIPTION_YEARS: tuple[int, ...] = (2023, 2024)


def _default_policy(max_bytes: int = 512 * 1024 * 1024) -> FetchPolicy:
    return FetchPolicy(max_attempts=2, retry_sleep_seconds=1.0, timeout_seconds=60.0, max_bytes=max_bytes)


def _d1_url(station: str, element: str, year: str) -> str:
    return "https://data.weather.gov.hk/weatherAPI/D1/caller.php?" + urlencode(
        {"stn": station, "ele": element, "yr": year}
    )


def _yyyymmdd(dt: datetime) -> str:
    return dt.astimezone(HKT).strftime("%Y%m%d")


def _default_datagov_history_end(now: datetime | None = None) -> str:
    local_now = (now or datetime.now(UTC)).astimezone(HKT)
    return _yyyymmdd(local_now - timedelta(days=1))


def _datagov_list_file_versions_url(resource_url: str, start: str, end: str) -> str:
    return "https://app.data.gov.hk/v1/historical-archive/list-file-versions?" + urlencode(
        {"url": resource_url, "start": start, "end": end}
    )


def _datagov_get_file_url(resource_url: str, timestamp: str) -> str:
    return "https://app.data.gov.hk/v1/historical-archive/get-file?" + urlencode(
        {"url": resource_url, "time": timestamp}
    )


def build_daily_climate_downloads() -> tuple[HkoDownload, ...]:
    downloads: list[HkoDownload] = []
    for element in DAILY_CLIMATE_ELEMENTS:
        downloads.append(
            HkoDownload(
                source_id=f"hko_daily_climate_{element.source_suffix}_all",
                provider=HKO_PROVIDER,
                url=_d1_url(element.station_code, element.download_code, "ALL"),
                extension="csv",
                description=element.title,
                metadata={
                    "family": "A_hko_target_labels_daily_climate",
                    "page_code": element.page_code,
                    "station_code": element.station_code,
                    "download_code": element.download_code,
                    "year": "ALL",
                    "start_year": element.start_year,
                    "point_in_time_class": element.point_in_time_class,
                },
            )
        )
    return tuple(downloads)


def build_datagov_historical_downloads_from_listing(
    feed: DataGovHistoricalFeed,
    listing: Mapping[str, object],
) -> tuple[HkoDownload, ...]:
    downloads: list[HkoDownload] = []
    data_files = listing.get("data-files", [])
    if not isinstance(data_files, list):
        return ()
    for item in data_files:
        if not isinstance(item, Mapping):
            continue
        timestamp = str(item.get("timestamp", ""))
        if not timestamp:
            continue
        downloads.append(
            HkoDownload(
                source_id=f"datagov_hko_historical_{feed.source_suffix}_archive",
                provider="DATA.GOV.HK / Hong Kong Observatory",
                url=_datagov_get_file_url(feed.resource_url, timestamp),
                extension=feed.extension,
                description=f"{feed.description} ({timestamp})",
                metadata={
                    "family": feed.family,
                    "point_in_time_class": feed.point_in_time_class,
                    "data_gov_historical_resource_url": feed.resource_url,
                    "data_gov_archive_timestamp": timestamp,
                    "data_gov_archive_period": str(item.get("period", "")),
                    "data_gov_archive_filename": str(item.get("filename", "")),
                    "data_gov_archive_resource_file_count": item.get("resource_file_count", ""),
                    "data_gov_archive_expected_size": item.get("size", ""),
                },
            )
        )
    return tuple(downloads)


def build_datagov_historical_live_downloads(
    now: datetime | None = None,
) -> tuple[HkoDownload, ...]:
    end = _default_datagov_history_end(now)
    downloads: list[HkoDownload] = [
        HkoDownload(
            "datagov_historical_api_documentation",
            "DATA.GOV.HK",
            "https://data.gov.hk/en/help/api-spec#historicalAPI",
            "html",
            "DATA.GOV.HK historical archive API documentation",
            {
                "family": "C_high_frequency_hko_regional_observations",
                "point_in_time_class": "METADATA",
            },
        )
    ]
    with httpx.Client(timeout=60.0, follow_redirects=True, verify=httpx_verify_context()) as client:
        for feed in DATAGOV_HISTORICAL_LIVE_FEEDS:
            listing_url = _datagov_list_file_versions_url(feed.resource_url, feed.start_date, end)
            listing = client.get(listing_url).json()
            downloads.append(
                HkoDownload(
                    source_id=f"datagov_hko_historical_{feed.source_suffix}_listing",
                    provider="DATA.GOV.HK / Hong Kong Observatory",
                    url=listing_url,
                    extension="json",
                    description=f"{feed.description} listing {feed.start_date}-{end}",
                    metadata={
                        "family": "C_high_frequency_hko_regional_observations",
                        "point_in_time_class": "METADATA",
                        "data_gov_historical_resource_url": feed.resource_url,
                        "start_date": feed.start_date,
                        "end_date": end,
                    },
                )
            )
            downloads.extend(build_datagov_historical_downloads_from_listing(feed, listing))
    return tuple(downloads)


def build_datagov_historical_rss_downloads(
    now: datetime | None = None,
) -> tuple[HkoDownload, ...]:
    end = _default_datagov_history_end(now)
    downloads: list[HkoDownload] = [
        HkoDownload(
            "datagov_historical_rss_api_documentation",
            "DATA.GOV.HK",
            "https://data.gov.hk/en/help/api-spec#historicalAPI",
            "html",
            "DATA.GOV.HK historical archive API documentation for HKO RSS archives",
            {
                "family": "D_official_hko_forecast_vintages",
                "point_in_time_class": "METADATA",
            },
        )
    ]
    with httpx.Client(timeout=60.0, follow_redirects=True, verify=httpx_verify_context()) as client:
        for feed in DATAGOV_HISTORICAL_RSS_FEEDS:
            listing_url = _datagov_list_file_versions_url(feed.resource_url, feed.start_date, end)
            listing = client.get(listing_url).json()
            downloads.append(
                HkoDownload(
                    source_id=f"datagov_hko_historical_{feed.source_suffix}_listing",
                    provider="DATA.GOV.HK / Hong Kong Observatory",
                    url=listing_url,
                    extension="json",
                    description=f"{feed.description} listing {feed.start_date}-{end}",
                    metadata={
                        "family": feed.family,
                        "point_in_time_class": "METADATA",
                        "data_gov_historical_resource_url": feed.resource_url,
                        "start_date": feed.start_date,
                        "end_date": end,
                    },
                )
            )
            downloads.extend(build_datagov_historical_downloads_from_listing(feed, listing))
    return tuple(downloads)


def extract_arwf_forecast_codes(station_config_text: str, common_js_text: str) -> tuple[str, ...]:
    station_codes = {
        code.upper()
        for code in re.findall(r'stationConfigAWS\["([^"]+)"\]', station_config_text)
    }
    arwf_codes = {
        code.upper()
        for code in re.findall(r'ARWF_code\s*:\s*"([^"]+)"', station_config_text)
    }
    grid_codes = {
        code.upper()
        for code in re.findall(r'gridXML\s*:\s*"([^"]+)"', station_config_text)
    }
    fallback_codes = {
        code.upper()
        for code in re.findall(r'matchXML\["[^"]+"\]\s*=\s*"([^"]+)"', common_js_text)
    }
    return tuple(sorted(station_codes | arwf_codes | grid_codes | fallback_codes))


def build_arwf_current_downloads() -> tuple[HkoDownload, ...]:
    station_config_url = "https://www.hko.gov.hk/en/wxinfo/awsgis/files/irwip-station-config-aws.js?v="
    common_js_url = "https://www.hko.gov.hk/en/wxinfo/awsgis/files/irwip-gis-common.js?v="
    forecast_base_url = "https://www.hko.gov.hk/wxinfo/awsgis/forecast/"
    downloads = list(ARWF_METADATA_DOWNLOADS + ARWF_CURRENT_DATA_DOWNLOADS + ARWF_ANIMATION_DOWNLOADS)
    with httpx.Client(timeout=60.0, follow_redirects=True, verify=httpx_verify_context()) as client:
        station_config_text = client.get(station_config_url).text
        common_js_text = client.get(common_js_url).text
        for code in extract_arwf_forecast_codes(station_config_text, common_js_text):
            url = f"{forecast_base_url}{code}.xml"
            response = client.get(url)
            if response.status_code < 200 or response.status_code >= 300:
                continue
            if not response.content or response.text.lstrip().startswith("<html"):
                continue
            downloads.append(
                HkoDownload(
                    "hko_arwf_station_forecast",
                    HKO_PROVIDER,
                    url,
                    "json",
                    "HKO ARWF station/grid forecast JSON served from forecast XML endpoint",
                    {
                        "family": "D_official_hko_forecast_vintages",
                        "point_in_time_class": "OPERATIONAL_POINT_IN_TIME",
                        "forecast_code": code,
                        "station_config_url": station_config_url,
                        "common_js_url": common_js_url,
                    },
                )
            )
    return tuple(downloads)


def extract_noaa_isd_nearby_stations(
    history_text: str,
    *,
    min_latitude: float = 21.0,
    max_latitude: float = 24.5,
    min_longitude: float = 112.0,
    max_longitude: float = 116.0,
) -> tuple[NoaaIsdStation, ...]:
    stations: list[NoaaIsdStation] = []
    for row in csv.DictReader(io.StringIO(history_text)):
        try:
            latitude = float(row.get("LAT", ""))
            longitude = float(row.get("LON", ""))
            begin_year = int(str(row.get("BEGIN", ""))[:4])
            end_year = int(str(row.get("END", ""))[:4])
        except ValueError:
            continue
        if not (min_latitude <= latitude <= max_latitude):
            continue
        if not (min_longitude <= longitude <= max_longitude):
            continue
        stations.append(
            NoaaIsdStation(
                usaf=str(row.get("USAF", "")),
                wban=str(row.get("WBAN", "")),
                name=str(row.get("STATION NAME", "")),
                country=str(row.get("CTRY", "")),
                icao=str(row.get("ICAO", "")),
                latitude=str(row.get("LAT", "")),
                longitude=str(row.get("LON", "")),
                elevation_m=str(row.get("ELEV(M)", "")),
                begin_year=begin_year,
                end_year=end_year,
            )
        )
    return tuple(sorted(stations, key=lambda station: (station.usaf, station.wban)))


def build_noaa_isd_nearby_downloads(now: datetime | None = None) -> tuple[HkoDownload, ...]:
    history_url = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
    current_year = (now or datetime.now(UTC)).astimezone(UTC).year
    downloads = list(NOAA_ISD_METADATA_DOWNLOADS)
    with httpx.Client(timeout=60.0, follow_redirects=True, verify=httpx_verify_context()) as client:
        history_text = client.get(history_url).text
        for station in extract_noaa_isd_nearby_stations(history_text):
            for year in range(station.begin_year, min(station.end_year, current_year) + 1):
                url = (
                    f"https://www.ncei.noaa.gov/pub/data/noaa/{year}/"
                    f"{station.usaf}-{station.wban}-{year}.gz"
                )
                response = client.head(url)
                if response.status_code < 200 or response.status_code >= 300:
                    continue
                downloads.append(
                    HkoDownload(
                        "noaa_isd_nearby_station_year",
                        "NOAA NCEI",
                        url,
                        "gz",
                        "NOAA ISD annual station archive for Hong Kong/Pearl River Delta bounding box",
                        {
                            "family": "I_tropical_cyclone_monsoon_synoptic_information",
                            "point_in_time_class": "PROXY_WITH_LIMITATIONS",
                            "station_usaf": station.usaf,
                            "station_wban": station.wban,
                            "station_name": station.name,
                            "station_country": station.country,
                            "station_icao": station.icao,
                            "station_latitude": station.latitude,
                            "station_longitude": station.longitude,
                            "station_elevation_m": station.elevation_m,
                            "year": year,
                            "source_catalog_url": history_url,
                            "spatial_filter": "21.0<=lat<=24.5 and 112.0<=lon<=116.0",
                        },
                    )
                )
    return tuple(downloads)


def build_ncep_filter_url(
    script_url: str,
    *,
    directory: str,
    filename: str,
    level_params: Iterable[str],
    variable_params: Iterable[str],
) -> str:
    params: dict[str, str] = {"dir": directory, "file": filename}
    params.update({level: "on" for level in level_params})
    params.update({variable: "on" for variable in variable_params})
    params.update(NCEP_HK_REGIONAL_DOMAIN)
    return script_url + "?" + urlencode(params)


def _ncep_cycle_candidates(now: datetime | None = None) -> tuple[tuple[str, str], ...]:
    utc_now = (now or datetime.now(UTC)).astimezone(UTC)
    cycles: list[tuple[str, str]] = []
    for day_offset in range(0, 4):
        date = (utc_now - timedelta(days=day_offset)).strftime("%Y%m%d")
        for cycle in ("18", "12", "06", "00"):
            cycles.append((date, cycle))
    return tuple(cycles)


def _latest_available_ncep_cycle(
    client: httpx.Client,
    *,
    script_url: str,
    directory_template: str,
    filename_template: str,
    level_param: str,
    variable_param: str,
    now: datetime | None = None,
) -> tuple[str, str] | None:
    for date, cycle in _ncep_cycle_candidates(now):
        url = build_ncep_filter_url(
            script_url,
            directory=directory_template.format(date=date, cycle=cycle),
            filename=filename_template.format(cycle=cycle, forecast_hour=120),
            level_params=(level_param,),
            variable_params=(variable_param,),
        )
        response = client.head(url)
        if response.status_code >= 200 and response.status_code < 300:
            return date, cycle
    return None


def build_ncep_operational_current_downloads(
    now: datetime | None = None,
) -> tuple[HkoDownload, ...]:
    gfs_script_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
    gefs_script_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p50a.pl"
    downloads = list(NCEP_METADATA_DOWNLOADS)
    with httpx.Client(timeout=60.0, follow_redirects=True, verify=httpx_verify_context()) as client:
        gfs_cycle = _latest_available_ncep_cycle(
            client,
            script_url=gfs_script_url,
            directory_template="/gfs.{date}/{cycle}/atmos",
            filename_template="gfs.t{cycle}z.pgrb2.0p25.f{forecast_hour:03d}",
            level_param="lev_2_m_above_ground",
            variable_param="var_TMP",
            now=now,
        )
        if gfs_cycle is not None:
            date, cycle = gfs_cycle
            for forecast_hour in NCEP_FORECAST_HOURS_6H_TO_120H:
                filename = f"gfs.t{cycle}z.pgrb2.0p25.f{forecast_hour:03d}"
                downloads.append(
                    HkoDownload(
                        "ncep_gfs_hk_subset_grib2",
                        "NOAA NCEP NOMADS",
                        build_ncep_filter_url(
                            gfs_script_url,
                            directory=f"/gfs.{date}/{cycle}/atmos",
                            filename=filename,
                            level_params=NCEP_GFS_LEVEL_PARAMS,
                            variable_params=NCEP_GFS_VARIABLE_PARAMS,
                        ),
                        "grib2",
                        "NOAA GFS 0.25 degree Hong Kong regional operational GRIB2 subset",
                        {
                            "family": "E_operational_numerical_ai_forecast_archives",
                            "point_in_time_class": "OPERATIONAL_POINT_IN_TIME",
                            "model": "GFS",
                            "grid": "0p25",
                            "cycle_date": date,
                            "cycle": cycle,
                            "forecast_hour": forecast_hour,
                            "member": "deterministic",
                            "domain": dict(NCEP_HK_REGIONAL_DOMAIN),
                            "levels": list(NCEP_GFS_LEVEL_PARAMS),
                            "variables": list(NCEP_GFS_VARIABLE_PARAMS),
                            "original_filename": filename,
                        },
                    )
                )

        gefs_cycle = _latest_available_ncep_cycle(
            client,
            script_url=gefs_script_url,
            directory_template="/gefs.{date}/{cycle}/atmos/pgrb2ap5",
            filename_template="gep01.t{cycle}z.pgrb2a.0p50.f{forecast_hour:03d}",
            level_param="lev_2_m_above_ground",
            variable_param="var_TMP",
            now=now,
        )
        if gefs_cycle is not None:
            date, cycle = gefs_cycle
            for member in NCEP_GEFS_MEMBERS:
                for forecast_hour in NCEP_FORECAST_HOURS_6H_TO_120H:
                    filename = f"{member}.t{cycle}z.pgrb2a.0p50.f{forecast_hour:03d}"
                    downloads.append(
                        HkoDownload(
                            "ncep_gefs_hk_subset_grib2",
                            "NOAA NCEP NOMADS",
                            build_ncep_filter_url(
                                gefs_script_url,
                                directory=f"/gefs.{date}/{cycle}/atmos/pgrb2ap5",
                                filename=filename,
                                level_params=NCEP_GEFS_LEVEL_PARAMS,
                                variable_params=NCEP_GEFS_VARIABLE_PARAMS,
                            ),
                            "grib2",
                            "NOAA GEFS 0.50 degree Hong Kong regional operational GRIB2 subset",
                            {
                                "family": "E_operational_numerical_ai_forecast_archives",
                                "point_in_time_class": "OPERATIONAL_POINT_IN_TIME",
                                "model": "GEFS",
                                "grid": "0p50",
                                "cycle_date": date,
                                "cycle": cycle,
                                "forecast_hour": forecast_hour,
                                "member": member,
                                "domain": dict(NCEP_HK_REGIONAL_DOMAIN),
                                "levels": list(NCEP_GEFS_LEVEL_PARAMS),
                                "variables": list(NCEP_GEFS_VARIABLE_PARAMS),
                                "original_filename": filename,
                            },
                        )
                    )
    return tuple(downloads)


def build_daily_extract_downloads(now: datetime | None = None) -> tuple[HkoDownload, ...]:
    local_now = (now or datetime.now(UTC)).astimezone(HKT)
    current_year = local_now.year
    current_month = local_now.month
    downloads = [
        HkoDownload(
            "hko_daily_extract_catalog",
            HKO_PROVIDER,
            "https://www.hko.gov.hk/cis/hko.xml",
            "json",
            "Daily Extract coverage catalog",
            {"family": "A_hko_target_labels_daily_climate", "point_in_time_class": "METADATA"},
        )
    ]
    for year in range(1884, current_year + 1):
        downloads.append(
            HkoDownload(
                "hko_daily_extract_year",
                HKO_PROVIDER,
                f"https://www.hko.gov.hk/cis/dailyExtract/dailyExtract_{year}.xml",
                "json",
                f"Daily Extract annual payload {year}",
                {
                    "family": "A_hko_target_labels_daily_climate",
                    "point_in_time_class": "TARGET_ONLY" if year == current_year else "PROXY_WITH_LIMITATIONS",
                    "year": year,
                },
            )
        )
    for month in range(1, current_month + 1):
        yyyymm = f"{current_year}{month:02d}"
        downloads.append(
            HkoDownload(
                "hko_daily_extract_month",
                HKO_PROVIDER,
                f"https://www.hko.gov.hk/cis/dailyExtract/dailyExtract_{yyyymm}.xml",
                "json",
                f"Daily Extract monthly payload {yyyymm}",
                {
                    "family": "A_hko_target_labels_daily_climate",
                    "point_in_time_class": "TARGET_ONLY",
                    "year": current_year,
                    "month": month,
                },
            )
        )
    return tuple(downloads)


def build_tc_best_track_downloads(start_year: int = 1985, end_year: int = 2024) -> tuple[HkoDownload, ...]:
    downloads = [
        HkoDownload(
            "hko_tropical_cyclone_best_track_dictionary",
            HKO_PROVIDER,
            "https://data.weather.gov.hk/weatherAPI/doc/data_dictionary_tropical_cyclone_best_track_data_post_analysis.pdf",
            "pdf",
            "Tropical cyclone best-track data dictionary",
            {"family": "I_tropical_cyclone_monsoon_synoptic", "point_in_time_class": "METADATA"},
        )
    ]
    for year in range(start_year, end_year + 1):
        downloads.append(
            HkoDownload(
                "hko_tropical_cyclone_best_track",
                HKO_PROVIDER,
                f"https://data.weather.gov.hk/weatherAPI/hko_data/tc/HKO{year}BST.csv",
                "csv",
                f"HKO tropical cyclone best-track data {year}",
                {
                    "family": "I_tropical_cyclone_monsoon_synoptic",
                    "point_in_time_class": "RETROSPECTIVE_ONLY",
                    "year": year,
                },
            )
        )
    return tuple(downloads)


def build_radar_lightning_downloads() -> tuple[HkoDownload, ...]:
    downloads = list(RADAR_LIGHTNING_BASE_DOWNLOADS)
    manifest_url = "https://www.hko.gov.hk/wxinfo/radars/temp_json/nradar_img.json"
    with httpx.Client(timeout=60.0, follow_redirects=True, verify=httpx_verify_context()) as client:
        data = client.get(manifest_url).json()
    image_scripts: list[str] = []
    radar = data.get("radar", {})
    if isinstance(radar, dict):
        for range_data in radar.values():
            if not isinstance(range_data, dict):
                continue
            images = range_data.get("image", [])
            if isinstance(images, list):
                for item in images:
                    if isinstance(item, str):
                        image_scripts.append(item)
    for script in image_scripts:
        match = re.search(r'"([^"]+\.jpg)"', script)
        if not match:
            continue
        relative_path = match.group(1)
        downloads.append(
            HkoDownload(
                "hko_radar_current_frame",
                HKO_PROVIDER,
                f"https://www.hko.gov.hk/wxinfo/radars/{relative_path}",
                "jpg",
                "HKO current radar image frame referenced by nradar_img.json",
                {
                    "family": "G_radar_rainfall_nowcasts_lightning",
                    "point_in_time_class": "OPERATIONAL_POINT_IN_TIME",
                    "manifest_url": manifest_url,
                    "relative_path": relative_path,
                },
            )
        )
    return tuple(downloads)


def _extract_satellite_filenames(js_text: str) -> list[str]:
    filenames: list[str] = []
    for match in re.finditer(r'\["([^"]+\.(?:png|jpg|gif))"', js_text):
        filename = match.group(1)
        if filename and filename not in filenames:
            filenames.append(filename)
    return filenames


def build_satellite_downloads() -> tuple[HkoDownload, ...]:
    downloads = list(SATELLITE_MANIFEST_DOWNLOADS)
    manifest_specs = (
        (
            "hko_satellite_modis_true_colour_image",
            "https://www.hko.gov.hk/wxinfo/intersat/satellite/image/data/modis_HK_VIS.js",
            "https://www.hko.gov.hk/wxinfo/intersat/satellite/image/t_colour/",
            "png",
            "HKO MODIS Hong Kong true-colour image",
        ),
        (
            "hko_satellite_modis_aod_image",
            "https://www.hko.gov.hk/wxinfo/intersat/satellite/image/data/modis_HK_V2_AOD.js",
            "https://www.hko.gov.hk/wxinfo/intersat/satellite/image/aod/",
            "png",
            "HKO MODIS Hong Kong aerosol optical depth image",
        ),
        (
            "hko_satellite_modis_sst_image",
            "https://www.hko.gov.hk/wxinfo/intersat/satellite/image/data/modis_HKS_SST.js",
            "https://www.hko.gov.hk/wxinfo/intersat/satellite/image/sst/",
            "png",
            "HKO MODIS Hong Kong sea-surface-temperature image",
        ),
    )
    with httpx.Client(timeout=60.0, follow_redirects=True, verify=httpx_verify_context()) as client:
        for source_id, manifest_url, base_url, extension, description in manifest_specs:
            js_text = client.get(manifest_url).text
            for filename in _extract_satellite_filenames(js_text):
                image_url = base_url + filename
                response = client.head(image_url)
                if response.status_code < 200 or response.status_code >= 300:
                    continue
                downloads.append(
                    HkoDownload(
                        source_id,
                        HKO_PROVIDER,
                        image_url,
                        extension,
                        description,
                        {
                            "family": "H_satellite_cloud_aerosol_observations",
                            "point_in_time_class": "OPERATIONAL_POINT_IN_TIME",
                            "manifest_url": manifest_url,
                            "filename": filename,
                        },
                    )
                )
    return tuple(downloads)


def build_static_context_downloads() -> tuple[HkoDownload, ...]:
    downloads = list(STATIC_CONTEXT_BASE_DOWNLOADS)
    for year, dataset_id, package_url in PLAND_LUHK_RASTER_SOURCES:
        downloads.extend(
            (
                HkoDownload(
                    f"csdi_pland_luhk_{year}_raster_dataset_page",
                    CSDI_PROVIDER,
                    f"https://portal.csdi.gov.hk/csdi-webpage/dataset/{dataset_id}",
                    "html",
                    f"CSDI LUHK {year} raster-grid dataset page",
                    {
                        "family": "L_static_geospatial_deterministic_context",
                        "point_in_time_class": "METADATA",
                        "year": year,
                        "dataset_id": dataset_id,
                    },
                ),
                HkoDownload(
                    f"csdi_pland_luhk_{year}_raster_metadata_xml",
                    CSDI_PROVIDER,
                    f"https://portal.csdi.gov.hk/csdi-webpage/metadata/{dataset_id}",
                    "xml",
                    f"CSDI ISO metadata for LUHK {year} raster grid",
                    {
                        "family": "L_static_geospatial_deterministic_context",
                        "point_in_time_class": "METADATA",
                        "year": year,
                        "dataset_id": dataset_id,
                    },
                ),
                HkoDownload(
                    f"csdi_pland_luhk_{year}_raster_geotiff_zip",
                    CSDI_PROVIDER,
                    package_url,
                    "zip",
                    f"CSDI LUHK {year} 10 m land-utilization raster GeoTIFF package",
                    {
                        "family": "L_static_geospatial_deterministic_context",
                        "point_in_time_class": "STATIC_CONTEXT_VERSIONED",
                        "year": year,
                        "dataset_id": dataset_id,
                        "variable": "land_utilization",
                        "grid": "Hong Kong 10 m raster",
                    },
                ),
            )
        )
    for year in PLAND_LUHK_STATISTICS_YEARS:
        downloads.append(
            HkoDownload(
                f"pland_luhk_{year}_statistics_english_csv",
                PLAND_PROVIDER,
                f"https://www.pland.gov.hk/pland_en/info_serv/statistic/landu/csv/LUHK{year}_English.csv",
                "csv",
                f"Planning Department LUHK {year} English land-use statistics",
                {
                    "family": "L_static_geospatial_deterministic_context",
                    "point_in_time_class": "STATIC_CONTEXT_VERSIONED",
                    "year": year,
                    "variable": "land_utilization_statistics",
                },
            )
        )
    for year in PLAND_LUHK_STATISTICS_DESCRIPTION_YEARS:
        downloads.append(
            HkoDownload(
                f"pland_luhk_{year}_statistics_description_english_csv",
                PLAND_PROVIDER,
                f"https://www.pland.gov.hk/pland_en/info_serv/statistic/landu/csv/LUHK{year}_English_description.csv",
                "csv",
                f"Planning Department LUHK {year} English data-description file",
                {
                    "family": "L_static_geospatial_deterministic_context",
                    "point_in_time_class": "METADATA",
                    "year": year,
                },
            )
        )
    return tuple(downloads)


def iter_batch_downloads(batch: str) -> tuple[HkoDownload, ...]:
    if batch == "daily-climate":
        return build_daily_climate_downloads()
    if batch == "daily-extract":
        return build_daily_extract_downloads()
    if batch == "live-discovered":
        return DISCOVERED_HKO_FEEDS
    if batch == "tc-best-track":
        return build_tc_best_track_downloads()
    if batch == "upper-air":
        return UPPER_AIR_DOWNLOADS
    if batch == "radar-lightning":
        return build_radar_lightning_downloads()
    if batch == "satellite-current":
        return build_satellite_downloads()
    if batch == "datagov-historical-live":
        return build_datagov_historical_live_downloads()
    if batch == "datagov-historical-rss":
        return build_datagov_historical_rss_downloads()
    if batch == "arwf-current":
        return build_arwf_current_downloads()
    if batch == "noaa-isd-nearby":
        return build_noaa_isd_nearby_downloads()
    if batch == "ncep-operational-current":
        return build_ncep_operational_current_downloads()
    if batch == "static-context-current":
        return build_static_context_downloads()
    if batch == "all-small":
        return (
            build_daily_climate_downloads()
            + build_daily_extract_downloads()
            + DISCOVERED_HKO_FEEDS
            + build_tc_best_track_downloads()
            + UPPER_AIR_DOWNLOADS
            + build_radar_lightning_downloads()
            + build_satellite_downloads()
            + build_arwf_current_downloads()
            + build_static_context_downloads()
        )
    raise ValueError(f"Unknown HKO backfill batch: {batch}")


def run_hko_backfill_batch(
    root: Path,
    *,
    batch: str,
    continue_on_error: bool = False,
    delay_seconds: float = 0.2,
) -> HkoBackfillOutcome:
    data_root = ensure_data_root(root)
    records: list[AcquisitionRecord] = []
    failures: list[str] = []
    downloads = iter_batch_downloads(batch)
    policy = _default_policy()
    for index, item in enumerate(downloads, start=1):
        if index > 1 and delay_seconds > 0:
            time.sleep(delay_seconds)
        try:
            record = fetch_http_to_acquisition(
                data_root,
                source_id=item.source_id,
                provider=item.provider,
                url=item.url,
                policy=policy,
                extension_override=item.extension,
                extra_metadata={"description": item.description, **dict(item.metadata)},
            )
            records.append(record)
        except Exception as exc:
            message = f"{item.source_id} {item.url}: {exc}"
            failures.append(message)
            if not continue_on_error:
                raise RuntimeError(message) from exc
    return HkoBackfillOutcome(
        requested=len(downloads),
        succeeded=len(records),
        failed=len(failures),
        records=tuple(records),
        failures=tuple(failures),
    )


def summarize_records(records: Iterable[AcquisitionRecord]) -> list[dict[str, object]]:
    return [
        {
            "source_id": record.source_id,
            "content_sha256": record.content_sha256,
            "content_length": record.content_length,
            "content_path": str(record.content_path),
            "deduplicated": record.deduplicated,
        }
        for record in records
    ]
