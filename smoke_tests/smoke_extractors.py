from __future__ import annotations

import importlib.util
import json
import logging
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ingestion-service" / "scripts"))
sys.path.insert(0, str(ROOT / "ml_live"))
sys.path.insert(0, str(ROOT / "ml" / "src"))
sys.path.insert(0, str(ROOT / "tools" / "ncei_truth"))


def load_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def smoke_weathercom() -> None:
    mod = load_module("weathercom_download_to_csv", "ingestion-service/scripts/weathercom_download_to_csv.py")
    task = mod.WindowTask(datetime(2026, 1, 2), datetime(2026, 1, 2))
    url = mod.build_request_url("https://api.weather.com", "KNYC:9:US", "secret", "e", task)
    require("/v1/location/KNYC:9:US/observations/historical.json" in url, "Weather.com URL path changed")
    require("startDate=20260102" in url and "endDate=20260102" in url, "Weather.com date params missing")
    payload = {
        "observations": [
            {
                "valid_time_gmt": 1767357000,
                "temp": 42,
                "dewPt": 35,
                "rh": 76,
                "pressure": 30.1,
                "vis": 10,
                "wspd": 8,
                "wdir": 270,
                "wx_phrase": "Partly Cloudy",
            }
        ]
    }
    df = mod.normalize_to_30m_rows("KNYC:9:US", payload)
    require(len(df) == 1, "Weather.com normalization did not produce one row")
    require(df.iloc[0]["wdir_cardinal"] == "W", "Weather.com wind cardinal normalization failed")


def smoke_ncei() -> None:
    download = load_module("download", "tools/ncei_truth/download.py")
    normalize = load_module("ncei_normalize", "tools/ncei_truth/normalize.py")
    with tempfile.TemporaryDirectory() as tmp:
        response_path = Path(tmp) / "response.json"
        response_path.write_text(
            json.dumps(
                [
                    {
                        "STATION": "USW00014732",
                        "DATE": "2026-01-02",
                        "TMAX": "42.4",
                        "TMAX_ATTRIBUTES": ",,W,2359",
                    }
                ]
            ),
            encoding="utf-8",
        )
        snap = download.SnapshotResult(
            station_id="KNYC",
            station_usw="USW00014732",
            start_date=date(2026, 1, 2),
            end_date=date(2026, 1, 2),
            url="https://example.test/ncei",
            response_path=response_path,
            headers_path=Path(tmp) / "headers.txt",
            retrieved_at_utc="2026-01-03T00:00:00Z",
            http_status=200,
            body_sha256="unused",
            byte_count=response_path.stat().st_size,
            skipped_existing=False,
        )
        rows = normalize.normalize_snapshots_to_rows(snapshots=[snap], logger=logging.getLogger("smoke"))
    require(len(rows) == 1, "NCEI normalization did not produce one row")
    require(rows[0]["tmax_f"] == 42, "NCEI TMAX rounding changed")


def smoke_iem_mos() -> None:
    from ml_live.python.fetch import iem_mos

    values = iem_mos._parse_values({"n_x": "84", "tmp": "82", "model": "GFS"})
    require(values["n_x"].numeric == 84.0, "IEM MOS numeric parse failed")
    start, end = iem_mos.mos_window_utc(date(2026, 7, 4), "America/New_York")
    require(start.tzinfo is timezone.utc and end.tzinfo is timezone.utc, "IEM MOS window must be UTC")


def smoke_gribstream() -> None:
    from ml_live.python.fetch import gribstream

    csv_text = "forecasted_at,forecasted_time,lat,lon,name,tmpk\n2026-01-01T00:00:00Z,2026-01-01T03:00:00Z,40.7,-73.9,KNYC,280.0\n"
    df = gribstream._parse_payload_df(csv_text, "text/csv", "text/csv")
    require(len(df) == 1 and float(df.iloc[0]["tmpk"]) == 280.0, "Gribstream CSV parse failed")

    v1 = load_module("backtesting.gribstream.V1.gribstream_client", "backtesting/gribstream/V1/gribstream_client.py")
    rows = v1._parse_csv_body(
        "forecasted_at,forecasted_time,lat,lon,name,member,TMP|2 m above ground|\n"
        "2026-01-01T00:00:00Z,2026-01-01T03:00:00Z,40.7,-73.9,KNYC,,280.0\n"
    )
    require(len(rows) == 1 and rows[0].value_native == 280.0, "Gribstream V1 CSV parse failed")


def smoke_kalshi() -> None:
    api = load_module("kalshi_api", "ingestion-service/scripts/kalshi_api.py")
    generic = load_module("kalshi_download_temperature_minute", "ingestion-service/scripts/kalshi_download_temperature_minute.py")
    client = api.KalshiClient(base_url="https://api.elections.kalshi.com/trade-api/v2")
    require(client.base_url.endswith("/trade-api/v2"), "Kalshi base URL normalization failed")
    primary, alternate = generic._event_tickers("KXHIGHMIA", date(2026, 1, 2))
    require(primary.startswith("KXHIGHMIA-") and alternate.startswith("KXHIGHMIA-"), "Kalshi event ticker build failed")


def smoke_polymarket() -> None:
    poly = load_module("polymarket_download_nyc_daily_wide", "ingestion-service/scripts/polymarket_download_nyc_daily_wide.py")
    lo, hi = poly.parse_bucket_bounds("80-81")
    require((lo, hi) == (80.0, 81.0), "Polymarket bucket parse failed")
    points = poly.parse_history_points({"history": [{"t": 1767357000, "p": 0.42}]}, start_ts=1767350000, end_ts=1767360000)
    require(points == [(1767357000, 42.0)], "Polymarket history parse failed")


def main() -> None:
    smoke_weathercom()
    smoke_ncei()
    smoke_iem_mos()
    smoke_gribstream()
    smoke_kalshi()
    smoke_polymarket()
    print("smoke_extractors: ok")


if __name__ == "__main__":
    main()
