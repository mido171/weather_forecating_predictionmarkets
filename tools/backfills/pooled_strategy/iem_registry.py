from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

from tools.backfills.pooled_strategy.station_universe import StationSeed


NETWORKS_URL = "https://mesonet.agron.iastate.edu/geojson/networks.geojson"
NETWORK_URL_TEMPLATE = "https://mesonet.agron.iastate.edu/geojson/network/{network}.geojson"
US_AIRPORT_NETWORK_RE = re.compile(r"^[A-Z]{2}_(ASOS|AWOS)$")


@dataclass(frozen=True)
class ResolvedStationMetadata:
    station_id: str
    iem_station_id: str
    iem_network: str
    station_zoneid: str
    latitude: float
    longitude: float
    elevation_m: float | None
    display_name: str
    archive_begin: str | None
    archive_end: str | None
    climate_site: str | None
    wfo: str | None
    nws_usw: str | None
    ncei91: str | None
    ghcnh_id: str | None
    synop_wban: str | None
    metar_reset_minute: int | None
    wu_location_id: str
    raw_feature: dict[str, Any]


def _log(logger: logging.Logger | None) -> logging.Logger:
    return logger or logging.getLogger("pooled_strategy_iem_registry")


def _fetch_json(url: str, session: requests.Session, timeout_seconds: int) -> dict[str, Any]:
    response = session.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    return response.json()


def _canonical_station_id(iem_station_id: str) -> str:
    sid = str(iem_station_id).strip().upper()
    if not sid:
        raise ValueError("iem station id is empty")
    if sid.startswith("K") and len(sid) == 4:
        return sid
    if len(sid) == 3:
        return f"K{sid}"
    return sid


def _extract_metadata(feature: dict[str, Any]) -> ResolvedStationMetadata:
    props = feature.get("properties") or {}
    attrs = props.get("attributes") or {}
    iem_station_id = str(feature.get("id") or props.get("sid") or "").strip().upper()
    station_id = _canonical_station_id(iem_station_id)
    timezone = str(props.get("tzname") or "").strip()
    coordinates = feature.get("geometry", {}).get("coordinates") or [None, None]
    lon = float(coordinates[0])
    lat = float(coordinates[1])
    synop = props.get("synop")
    synop_wban = str(int(synop)) if isinstance(synop, (int, float)) else (str(synop).strip() if synop else None)
    metar_reset_raw = attrs.get("METAR_RESET_MINUTE")
    metar_reset = int(metar_reset_raw) if str(metar_reset_raw or "").strip().isdigit() else None
    nws_usw = str(props.get("ncdc81") or attrs.get("GHCNH_ID") or "").strip().upper() or None
    ncei91 = str(props.get("ncei91") or "").strip().upper() or None
    ghcnh_id = str(attrs.get("GHCNH_ID") or "").strip().upper() or None
    return ResolvedStationMetadata(
        station_id=station_id,
        iem_station_id=iem_station_id,
        iem_network=str(props.get("network") or "").strip().upper(),
        station_zoneid=timezone,
        latitude=lat,
        longitude=lon,
        elevation_m=float(props["elevation"]) if props.get("elevation") is not None else None,
        display_name=str(props.get("sname") or station_id).strip(),
        archive_begin=str(props.get("archive_begin") or "").strip() or None,
        archive_end=str(props.get("archive_end") or "").strip() or None,
        climate_site=str(props.get("climate_site") or "").strip() or None,
        wfo=str(props.get("wfo") or "").strip() or None,
        nws_usw=nws_usw,
        ncei91=ncei91,
        ghcnh_id=ghcnh_id,
        synop_wban=synop_wban,
        metar_reset_minute=metar_reset,
        wu_location_id=f"{station_id}:9:US",
        raw_feature=feature,
    )


def build_iem_airport_registry(
    *,
    cache_dir: Path,
    refresh: bool = False,
    timeout_seconds: int = 60,
    logger: logging.Logger | None = None,
) -> dict[str, ResolvedStationMetadata]:
    log = _log(logger)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "iem_us_airport_registry.json"
    if cache_path.exists() and not refresh:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        entries = payload.get("entries") or {}
        return {key: ResolvedStationMetadata(**value) for key, value in entries.items()}

    session = requests.Session()
    session.headers.update({"User-Agent": "pooled-strategy station registry builder"})

    networks_doc = _fetch_json(NETWORKS_URL, session, timeout_seconds)
    network_ids = sorted(
        str(feature.get("id") or "").strip().upper()
        for feature in (networks_doc.get("features") or [])
        if US_AIRPORT_NETWORK_RE.match(str(feature.get("id") or "").strip().upper())
    )
    log.info("IEM_REGISTRY_NETWORKS count=%d", len(network_ids))

    entries: dict[str, ResolvedStationMetadata] = {}
    for idx, network_id in enumerate(network_ids, start=1):
        url = NETWORK_URL_TEMPLATE.format(network=network_id)
        network_doc = _fetch_json(url, session, timeout_seconds)
        features = network_doc.get("features") or []
        log.debug("IEM_REGISTRY_NETWORK network=%s index=%d/%d features=%d", network_id, idx, len(network_ids), len(features))
        for feature in features:
            props = feature.get("properties") or {}
            if str(props.get("country") or "").strip().upper() != "US":
                continue
            try:
                metadata = _extract_metadata(feature)
            except Exception:
                continue
            entries.setdefault(metadata.station_id, metadata)
            entries.setdefault(metadata.iem_station_id, metadata)

    payload = {
        "source": NETWORKS_URL,
        "network_count": len(network_ids),
        "entry_count": len(entries),
        "entries": {key: asdict(value) for key, value in entries.items()},
    }
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    log.info("IEM_REGISTRY_CACHE_WRITTEN path=%s entries=%d", cache_path, len(entries))
    return entries


def resolve_station_metadata(
    station_id: str,
    registry: dict[str, ResolvedStationMetadata],
) -> ResolvedStationMetadata:
    keys = [str(station_id).strip().upper()]
    if keys[0].startswith("K") and len(keys[0]) == 4:
        keys.append(keys[0][1:])
    for key in keys:
        if key in registry:
            return registry[key]
    raise KeyError(f"Station {station_id} was not found in the IEM airport registry cache")


def build_station_crosswalk_rows(
    seeds: list[StationSeed],
    registry: dict[str, ResolvedStationMetadata],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        lookup_station_id = seed.metadata_lookup_station_id or seed.station_id
        meta = resolve_station_metadata(lookup_station_id, registry)
        rows.append(
            {
                "station_id": seed.station_id,
                "metadata_lookup_station_id": lookup_station_id,
                "tier": seed.tier,
                "group_name": seed.group_name,
                "active_flag": int(seed.active),
                "traded_station_flag": int(seed.traded_station),
                "kalshi_series": seed.kalshi_series or "",
                "market_station_id": seed.station_id if seed.traded_station else "",
                "mos_station_id": meta.station_id,
                "truth_station_id": meta.nws_usw or "",
                "obs_station_id": meta.wu_location_id,
                "display_name": meta.display_name,
                "timezone": meta.station_zoneid,
                "lat": meta.latitude,
                "lon": meta.longitude,
                "elevation_m": meta.elevation_m if meta.elevation_m is not None else "",
                "iem_station_id": meta.iem_station_id,
                "iem_network": meta.iem_network,
                "archive_begin": meta.archive_begin or "",
                "archive_end": meta.archive_end or "",
                "climate_site": meta.climate_site or "",
                "wfo": meta.wfo or "",
                "wu_location_id": meta.wu_location_id,
            }
        )
    return rows
