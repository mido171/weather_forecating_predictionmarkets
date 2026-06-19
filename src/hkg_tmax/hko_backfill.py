from __future__ import annotations

import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import httpx

from .acquisition import AcquisitionRecord, ensure_data_root, fetch_http_to_acquisition
from .fetch import FetchPolicy

HKT = ZoneInfo("Asia/Hong_Kong")
HKO_PROVIDER = "Hong Kong Observatory"


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


def _default_policy(max_bytes: int = 512 * 1024 * 1024) -> FetchPolicy:
    return FetchPolicy(max_attempts=2, retry_sleep_seconds=1.0, timeout_seconds=60.0, max_bytes=max_bytes)


def _d1_url(station: str, element: str, year: str) -> str:
    return "https://data.weather.gov.hk/weatherAPI/D1/caller.php?" + urlencode(
        {"stn": station, "ele": element, "yr": year}
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
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
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
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
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
    if batch == "all-small":
        return (
            build_daily_climate_downloads()
            + build_daily_extract_downloads()
            + DISCOVERED_HKO_FEEDS
            + build_tc_best_track_downloads()
            + UPPER_AIR_DOWNLOADS
            + build_radar_lightning_downloads()
            + build_satellite_downloads()
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
