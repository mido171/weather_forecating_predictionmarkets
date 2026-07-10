from __future__ import annotations

from datetime import date

from klga_tmax.providers.wunderground.client import build_weathercom_url, weathercom_location_id


def test_weathercom_location_id_adds_station_suffix() -> None:
    assert weathercom_location_id("klga") == "KLGA:9:US"


def test_weathercom_location_id_preserves_explicit_location_id() -> None:
    assert weathercom_location_id("KLGA:9:US") == "KLGA:9:US"


def test_build_weathercom_url_uses_historical_observations_endpoint() -> None:
    url = build_weathercom_url(
        base_url="https://api.weather.com/",
        weathercom_location_id_value="KLGA:9:US",
        api_key="secret",
        units="e",
        start_date=date(2021, 8, 1),
        end_date=date(2021, 8, 31),
    )
    assert url.startswith("https://api.weather.com/v1/location/KLGA:9:US/observations/historical.json?")
    assert "apiKey=secret" in url
    assert "units=e" in url
    assert "startDate=20210801" in url
    assert "endDate=20210831" in url
