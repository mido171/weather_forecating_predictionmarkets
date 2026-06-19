from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from hkg_tmax.hko_backfill import (
    UPPER_AIR_DOWNLOADS,
    build_daily_climate_downloads,
    build_daily_extract_downloads,
    build_tc_best_track_downloads,
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
