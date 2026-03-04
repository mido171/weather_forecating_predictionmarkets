from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import requests


KALSHI_SERIES_URL = "https://api.elections.kalshi.com/trade-api/v2/series/{ticker}"

# Default zones currently used in this repo.
DEFAULT_ZONE_BY_STATION = {
    "KNYC": "America/New_York",
    "KMIA": "America/New_York",
    "KPHL": "America/New_York",
    "KMDW": "America/Chicago",
    "KLAX": "America/Los_Angeles",
}


@dataclass(frozen=True)
class ResolvedStationMetadata:
    series_ticker: str
    station_id: str
    station_zoneid: str
    file_prefix: str
    settlement_url: str
    issuedby: Optional[str]
    source: str


def _normalize_series_ticker(value: str) -> str:
    out = str(value or "").strip().upper()
    if not out:
        raise ValueError("--station-id is required")
    return out


def _normalize_station_id(value: str) -> str:
    out = str(value or "").strip().upper()
    if not out:
        raise ValueError("station id is required")
    return out


def _normalize_zoneid(value: str) -> str:
    out = str(value or "").strip()
    if not out:
        raise ValueError("station zoneid is required")
    return out


def _extract_issuedby_from_settlement_url(url: str) -> Optional[str]:
    parsed = urlparse(url)
    if not parsed.query:
        return None
    params = parse_qs(parsed.query)
    issuedby_list = params.get("issuedby") or params.get("ISSUEDBY")
    if not issuedby_list:
        return None
    issuedby = str(issuedby_list[0]).strip().upper()
    return issuedby or None


def _derive_station_id_from_issuedby(issuedby: str) -> str:
    code = str(issuedby or "").strip().upper()
    if not code:
        raise ValueError("issuedby is empty")
    if re.fullmatch(r"[A-Z]{3}", code):
        return f"K{code}"
    if re.fullmatch(r"K[A-Z0-9]{3}", code):
        return code
    raise ValueError(f"Could not derive station id from issuedby={issuedby}")


def fetch_series_payload(series_ticker: str, timeout_seconds: int = 30) -> dict[str, Any]:
    ticker = _normalize_series_ticker(series_ticker)
    url = KALSHI_SERIES_URL.format(ticker=ticker)
    resp = requests.get(url, timeout=timeout_seconds)
    if resp.status_code != 200:
        raise ValueError(f"Kalshi series lookup failed for ticker={ticker}: HTTP {resp.status_code}")
    payload = resp.json()
    if "series" not in payload or not isinstance(payload["series"], dict):
        raise ValueError(f"Unexpected Kalshi series response for ticker={ticker}")
    return payload


def resolve_station_metadata(
    *,
    station_id_series: str,
    mos_station_id_override: Optional[str] = None,
    station_zoneid_override: Optional[str] = None,
    file_prefix_override: Optional[str] = None,
) -> ResolvedStationMetadata:
    payload = fetch_series_payload(station_id_series)
    series = payload["series"]
    series_ticker = _normalize_series_ticker(series.get("ticker") or station_id_series)
    settlement_sources = series.get("settlement_sources") or []
    settlement_url = ""
    if settlement_sources and isinstance(settlement_sources[0], dict):
        settlement_url = str(settlement_sources[0].get("url") or "").strip()
    issuedby = _extract_issuedby_from_settlement_url(settlement_url) if settlement_url else None

    if mos_station_id_override:
        station_id = _normalize_station_id(mos_station_id_override)
        source = "override"
    else:
        if not issuedby:
            raise ValueError(
                "Could not resolve station id from Kalshi settlement metadata "
                f"for series={series_ticker}. Provide --mos-station-id explicitly."
            )
        station_id = _derive_station_id_from_issuedby(issuedby)
        source = "kalshi_settlement_issuedby"

    if station_zoneid_override:
        station_zoneid = _normalize_zoneid(station_zoneid_override)
    else:
        station_zoneid = DEFAULT_ZONE_BY_STATION.get(station_id, "")
        if not station_zoneid:
            raise ValueError(
                f"Could not resolve timezone for station={station_id}. "
                "Provide --station-zoneid explicitly."
            )

    file_prefix = _normalize_station_id(file_prefix_override) if file_prefix_override else station_id

    return ResolvedStationMetadata(
        series_ticker=series_ticker,
        station_id=station_id,
        station_zoneid=station_zoneid,
        file_prefix=file_prefix,
        settlement_url=settlement_url,
        issuedby=issuedby,
        source=source,
    )
