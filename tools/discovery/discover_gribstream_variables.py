"""Discover Gribstream variable availability via OpenAPI/probes."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import gzip
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover Gribstream variables.")
    parser.add_argument(
        "--app-config",
        default=str(
            REPO_ROOT
            / "ingestion-service"
            / "src"
            / "main"
            / "resources"
            / "application.yml"
        ),
        help="Path to application.yml",
    )
    parser.add_argument("--station-id", help="Station ID (default: first configured).")
    parser.add_argument(
        "--models",
        default="gfs,nam,hrrr,rap,nbm,gefsatmos,gefsatmosmean",
        help="Comma-separated model codes.",
    )
    parser.add_argument(
        "--target-date",
        help="Target date local (YYYY-MM-DD). Default: tomorrow in station zone.",
    )
    parser.add_argument(
        "--asof-local-time",
        default="12:00",
        help="As-of local time (default: 12:00).",
    )
    parser.add_argument(
        "--asof-time-zone",
        default="UTC",
        help="As-of time zone (default: UTC).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "artifacts" / "discovery"),
        help="Output directory.",
    )
    parser.add_argument(
        "--skip-probes",
        action="store_true",
        help="Skip per-variable probes (only OpenAPI/catalog discovery).",
    )
    parser.add_argument(
        "--skip-openapi",
        action="store_true",
        help="Skip OpenAPI discovery endpoints.",
    )
    parser.add_argument(
        "--skip-catalog",
        action="store_true",
        help="Skip catalog/model discovery endpoints.",
    )
    parser.add_argument(
        "--max-probes",
        type=int,
        default=0,
        help="Limit number of variable probes per model (0 = no limit).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(Path(args.app_config))
    grib_cfg = config.get("gribstream") or {}
    base_url = grib_cfg.get("baseUrl", "https://gribstream.com").rstrip("/")
    token = grib_cfg.get("apiToken") or ""
    auth_scheme = grib_cfg.get("authScheme", "Bearer")
    auth_header = token if " " in token else f"{auth_scheme} {token}"

    station = resolve_station(config, args.station_id)
    target_date_local = resolve_target_date(args.target_date, station["zoneId"])
    asof_utc = resolve_asof_utc(
        target_date_local, args.asof_local_time, args.asof_time_zone, station["zoneId"]
    )

    start_utc, end_utc = standard_time_window(station["zoneId"], target_date_local)
    min_horizon = int((grib_cfg.get("models") or {}).get("defaultMinHorizonHours", 0))

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    discovery: dict[str, Any] = {
        "base_url": base_url,
        "station": station,
        "target_date_local": target_date_local.isoformat(),
        "asof_utc": asof_utc.isoformat(),
        "openapi": {},
        "catalog_endpoints": {},
        "probes": {},
    }

    if args.skip_openapi:
        discovery["openapi"] = {"skipped": True}
    else:
        discovery["openapi"] = fetch_openapi(base_url, auth_header)

    if args.skip_catalog:
        discovery["catalog_endpoints"] = {"skipped": True}
    else:
        discovery["catalog_endpoints"] = probe_catalog_endpoints(
            base_url, auth_header, models
        )

    probe_variables = candidate_variables()
    model_cfg = grib_cfg.get("models") or {}
    gefs_members = (grib_cfg.get("gefs") or {}).get("members")

    if not args.skip_probes:
        for model in models:
            max_horizon = resolve_max_horizon(model_cfg, model)
            model_result = {
                "max_horizon": max_horizon,
                "success": [],
                "failed": [],
            }
            for idx, var in enumerate(probe_variables):
                if args.max_probes and idx >= args.max_probes:
                    break
                payload = {
                    "fromTime": start_utc.isoformat().replace("+00:00", "Z"),
                    "untilTime": end_utc.isoformat().replace("+00:00", "Z"),
                    "asOf": asof_utc.isoformat().replace("+00:00", "Z"),
                    "minHorizon": min_horizon,
                    "maxHorizon": max_horizon,
                    "coordinates": [
                        {
                            "lat": station["latitude"],
                            "lon": station["longitude"],
                            "name": station["name"],
                        }
                    ],
                    "variables": [var],
                }
                if model == "gefsatmos" and gefs_members:
                    payload["members"] = gefs_members
                url = f"{base_url}/api/v2/{model}/history"
                ok, info = probe_history(url, auth_header, payload, var)
                if ok:
                    model_result["success"].append(info)
                else:
                    model_result["failed"].append(info)
            discovery["probes"][model] = model_result

    out_path = output_dir / f"gribstream_probe_{station['stationId']}_{target_date_local}.json"
    out_path.write_text(json.dumps(discovery, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[GRIBSTREAM] probe output={out_path}")
    return 0


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_station(config: dict, station_id: str | None) -> dict:
    stations = (config.get("gribstream") or {}).get("stations") or []
    if not stations:
        raise ValueError("gribstream.stations is required")
    if station_id:
        station_id = station_id.strip().upper()
        for station in stations:
            if station.get("stationId", "").upper() == station_id:
                return station
        raise ValueError(f"Station not found: {station_id}")
    return stations[0]


def resolve_target_date(target_date: str | None, zone_id: str) -> date:
    if target_date:
        return date.fromisoformat(target_date)
    tz = zoneinfo(zone_id)
    return (datetime.now(tz).date() + timedelta(days=1))


def resolve_asof_utc(
    target_date: date, asof_time: str, asof_tz: str, station_zone: str
) -> datetime:
    hour, minute = [int(part) for part in asof_time.split(":")]
    if asof_tz.upper() == "UTC":
        tz = timezone.utc
    else:
        tz = zoneinfo(asof_tz if asof_tz != "LOCAL" else station_zone)
    asof_local = datetime.combine(target_date, time(hour, minute), tzinfo=tz)
    return asof_local.astimezone(timezone.utc)


def zoneinfo(zone_id: str):
    from zoneinfo import ZoneInfo

    return ZoneInfo(zone_id)


def standard_time_window(zone_id: str, target_date: date) -> tuple[datetime, datetime]:
    tz = zoneinfo(zone_id)
    noon = datetime.combine(target_date, time(12, 0), tzinfo=tz)
    offset = noon.utcoffset() or timedelta(0)
    dst = noon.dst() or timedelta(0)
    standard = offset - dst
    start = datetime.combine(target_date, time(0, 0), tzinfo=timezone(standard))
    start_utc = start.astimezone(timezone.utc)
    end_utc = start_utc + timedelta(days=1)
    return start_utc, end_utc


def resolve_max_horizon(model_cfg: dict, model: str) -> int:
    cfg = model_cfg.get(model)
    if not cfg or not cfg.get("maxHorizonHours"):
        return 48
    return int(cfg["maxHorizonHours"])


def candidate_variables() -> list[dict]:
    return [
        {"name": "TMP", "level": "2 m above ground", "info": "", "alias": "tmpk"},
        {"name": "RH", "level": "2 m above ground", "info": "", "alias": "rh"},
        {"name": "WIND", "level": "10 m above ground", "info": "", "alias": "wind"},
        {"name": "WDIR", "level": "10 m above ground", "info": "", "alias": "wdir"},
        {"name": "UGRD", "level": "10 m above ground", "info": "", "alias": "ugrd10"},
        {"name": "VGRD", "level": "10 m above ground", "info": "", "alias": "vgrd10"},
        {"name": "DSWRF", "level": "surface", "info": "", "alias": "dswrf"},
        {"name": "SOILW", "level": "0.3-0.3 m below ground", "info": "", "alias": "soilw"},
        {"name": "CAPE", "level": "180-0 mb above ground", "info": "", "alias": "cape"},
        {"name": "CIN", "level": "180-0 mb above ground", "info": "", "alias": "cin"},
        {"name": "APCP", "level": "surface", "info": "", "alias": "apcp"},
        {"name": "PRATE", "level": "surface", "info": "", "alias": "prate"},
        {"name": "TCDC", "level": "entire atmosphere", "info": "", "alias": "tcdc"},
        {"name": "PRES", "level": "surface", "info": "", "alias": "pres"},
        {"name": "PBLH", "level": "surface", "info": "", "alias": "pblh"},
    ]


def fetch_openapi(base_url: str, auth_header: str) -> dict[str, Any]:
    endpoints = ["openapi.json", "openapi.yaml", "openapi", "swagger.json"]
    results: dict[str, Any] = {}
    for suffix in endpoints:
        url = f"{base_url}/{suffix}"
        ok, payload = fetch_endpoint(url, auth_header, accept="application/json")
        results[url] = {"ok": ok, "sample": payload[:500] if isinstance(payload, str) else payload}
        if ok and isinstance(payload, str) and payload.strip().startswith("{"):
            try:
                results[url]["parsed"] = json.loads(payload)
            except json.JSONDecodeError:
                pass
    return results


def probe_catalog_endpoints(base_url: str, auth_header: str, models: list[str]) -> dict[str, Any]:
    endpoints = [
        "api/v2/models",
        "api/v2/model",
        "api/v2/variables",
        "api/v2/catalog",
    ]
    results: dict[str, Any] = {}
    for suffix in endpoints:
        url = f"{base_url}/{suffix}"
        ok, payload = fetch_endpoint(url, auth_header, accept="application/json")
        results[url] = {"ok": ok, "sample": payload[:500] if isinstance(payload, str) else payload}
    for model in models:
        for suffix in ["variables", "inventory", "metadata"]:
            url = f"{base_url}/api/v2/{model}/{suffix}"
            ok, payload = fetch_endpoint(url, auth_header, accept="application/json")
            results[url] = {"ok": ok, "sample": payload[:500] if isinstance(payload, str) else payload}
    return results


def probe_history(
    url: str, auth_header: str, payload: dict, variable: dict
) -> tuple[bool, dict]:
    body = json.dumps(payload).encode("utf-8")
    req = Request(url, data=body, method="POST")
    req.add_header("Authorization", auth_header)
    req.add_header("Accept", "application/ndjson")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept-Encoding", "gzip")
    info = {"variable": variable}
    try:
        with urlopen(req, timeout=30) as resp:
            raw = resp.read()
            if resp.headers.get("Content-Encoding", "").lower() == "gzip":
                raw = gzip.decompress(raw)
        text = raw.decode("utf-8", errors="replace")
        info["rows_sample"] = text.splitlines()[:3]
        return True, info
    except Exception as exc:  # noqa: BLE001
        info["error"] = str(exc)
        return False, info


def fetch_endpoint(url: str, auth_header: str, accept: str) -> tuple[bool, Any]:
    try:
        req = Request(url, method="GET")
        req.add_header("Authorization", auth_header)
        req.add_header("Accept", accept)
        with urlopen(req, timeout=20) as resp:
            raw = resp.read()
        return True, raw.decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


if __name__ == "__main__":
    raise SystemExit(main())
