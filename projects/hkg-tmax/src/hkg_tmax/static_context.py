from __future__ import annotations

import csv
import importlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from .acquisition import ensure_data_root
from .config import load_yaml
from .hko_backfill import extract_noaa_isd_nearby_stations
from .paths import ProjectPaths, resolve_archive_content_path


class StaticContextError(RuntimeError):
    """Raised when deterministic static context cannot be generated."""


@dataclass(frozen=True)
class StaticContextOutputs:
    station_registry_csv: Path
    station_distance_csv: Path
    solar_geometry_csv: Path
    report_path: Path


def _read_ledger(data_root: Path) -> list[dict[str, str]]:
    path = data_root / "manifests" / "retrieval_ledger.csv"
    if not path.exists():
        raise StaticContextError(f"Missing retrieval ledger: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _latest_success(rows: Sequence[Mapping[str, str]], source_id: str) -> Mapping[str, str]:
    matches = [
        row
        for row in rows
        if row.get("source_id") == source_id and row.get("status") == "success"
    ]
    if not matches:
        raise StaticContextError(f"No successful retrieval found for {source_id}")
    return max(matches, key=lambda row: row.get("retrieved_at", ""))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise StaticContextError(f"Refusing to write empty static context: {path}")
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_parquet(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pa: Any = importlib.import_module("pyarrow")
        pq: Any = importlib.import_module("pyarrow.parquet")
    except ModuleNotFoundError:
        return
    table = pa.Table.from_pylist(
        [
            {
                key: (
                    json.dumps(value, sort_keys=True)
                    if isinstance(value, (list, dict))
                    else "" if value is None else str(value)
                )
                for key, value in row.items()
            }
            for row in rows
        ]
    )
    pq.write_table(table, path, compression="zstd")


def _as_float(value: object) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(a))


def _bearing_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def _solar_declination(day_of_year: int) -> float:
    return math.radians(23.44) * math.sin(math.radians((360 / 365.0) * (day_of_year - 81)))


def _solar_elevation(lat: float, day: date, local_hour: float) -> float:
    latitude = math.radians(lat)
    declination = _solar_declination(day.timetuple().tm_yday)
    hour_angle = math.radians(15 * (local_hour - 12))
    sine_elevation = (
        math.sin(latitude) * math.sin(declination)
        + math.cos(latitude) * math.cos(declination) * math.cos(hour_angle)
    )
    return math.degrees(math.asin(max(-1.0, min(1.0, sine_elevation))))


def _day_length_hours(lat: float, day: date) -> float:
    latitude = math.radians(lat)
    declination = _solar_declination(day.timetuple().tm_yday)
    argument = -math.tan(latitude) * math.tan(declination)
    argument = max(-1.0, min(1.0, argument))
    return (2 / 15) * math.degrees(math.acos(argument))


def _build_station_rows(root: Path, data_root: Path) -> list[dict[str, object]]:
    station_config = load_yaml(root / "config" / "sources" / "stations_hko.yaml")
    target = station_config.get("target_station", {})
    if not isinstance(target, dict):
        raise StaticContextError("config/sources/stations_hko.yaml missing target_station mapping")
    target_row = {
        "station_id": str(target.get("code", "HKO")),
        "station_name": str(target.get("name", "Hong Kong Observatory")),
        "network": "HKO",
        "latitude": target.get("latitude"),
        "longitude": target.get("longitude"),
        "elevation_m": target.get("elevation_m"),
        "source": "config/sources/stations_hko.yaml",
        "point_in_time_class": "STATIC_METADATA",
    }

    ledger = _read_ledger(data_root)
    isd_history = _latest_success(ledger, "noaa_isd_history")
    history_text = resolve_archive_content_path(isd_history, data_root=data_root).read_text(
        encoding="utf-8", errors="replace"
    )
    rows = [target_row]
    for station in extract_noaa_isd_nearby_stations(history_text):
        rows.append(
            {
                "station_id": f"{station.usaf}-{station.wban}",
                "station_name": station.name,
                "network": "NOAA_ISD",
                "latitude": station.latitude,
                "longitude": station.longitude,
                "elevation_m": station.elevation_m,
                "source": "noaa_isd_history",
                "point_in_time_class": "PROXY_WITH_LIMITATIONS",
                "begin_year": station.begin_year,
                "end_year": station.end_year,
                "icao": station.icao,
                "country": station.country,
            }
        )
    return rows


def _build_distance_rows(stations: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for origin in stations:
        origin_lat = _as_float(origin.get("latitude"))
        origin_lon = _as_float(origin.get("longitude"))
        origin_elev = _as_float(origin.get("elevation_m"))
        if origin_lat is None or origin_lon is None:
            continue
        for target in stations:
            target_lat = _as_float(target.get("latitude"))
            target_lon = _as_float(target.get("longitude"))
            target_elev = _as_float(target.get("elevation_m"))
            if target_lat is None or target_lon is None:
                continue
            rows.append(
                {
                    "origin_station_id": origin.get("station_id"),
                    "target_station_id": target.get("station_id"),
                    "distance_km": round(
                        _haversine_km(origin_lat, origin_lon, target_lat, target_lon), 3
                    ),
                    "bearing_degrees": round(
                        _bearing_degrees(origin_lat, origin_lon, target_lat, target_lon), 1
                    ),
                    "elevation_delta_m": (
                        round(target_elev - origin_elev, 2)
                        if target_elev is not None and origin_elev is not None
                        else ""
                    ),
                }
            )
    return rows


def _build_solar_rows(stations: Sequence[Mapping[str, object]], *, year: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    day = date(year, 1, 1)
    while day.year == year:
        for station in stations:
            latitude = _as_float(station.get("latitude"))
            longitude = _as_float(station.get("longitude"))
            if latitude is None or longitude is None:
                continue
            day_length = _day_length_hours(latitude, day)
            for local_hour in (6, 9, 12, 15, 18):
                rows.append(
                    {
                        "station_id": station.get("station_id"),
                        "local_date": day.isoformat(),
                        "local_hour_hkt": local_hour,
                        "solar_elevation_degrees": round(
                            _solar_elevation(latitude, day, local_hour), 3
                        ),
                        "day_length_hours": round(day_length, 3),
                        "method": "deterministic_astronomical_approximation_v1",
                    }
                )
        day += timedelta(days=1)
    return rows


def build_static_context(root: Path, *, solar_year: int | None = None) -> StaticContextOutputs:
    data_root = ensure_data_root(root)
    year = solar_year or datetime.now().year
    stations = _build_station_rows(root, data_root)
    distances = _build_distance_rows(stations)
    solar = _build_solar_rows(stations, year=year)

    metadata_dir = data_root / "metadata" / "static_context"
    station_csv = metadata_dir / "station_registry.csv"
    distance_csv = metadata_dir / "station_distance_bearing.csv"
    solar_csv = metadata_dir / f"solar_geometry_{year}.csv"
    _write_csv(station_csv, stations)
    _write_csv(distance_csv, distances)
    _write_csv(solar_csv, solar)
    _write_parquet(metadata_dir / "station_registry.parquet", stations)
    _write_parquet(metadata_dir / "station_distance_bearing.parquet", distances)
    _write_parquet(metadata_dir / f"solar_geometry_{year}.parquet", solar)

    lineage = {
        "schema_version": 1,
        "created_at_utc": datetime.now().astimezone().isoformat(),
        "inputs": [
            "config/sources/stations_hko.yaml",
            "latest successful noaa_isd_history raw object",
        ],
        "outputs": [str(station_csv), str(distance_csv), str(solar_csv)],
        "solar_year": year,
        "notes": "Terrain, coastline, LUHK fractions, horizon obstruction, and upwind terrain summaries remain pending source-specific geospatial parsing.",
    }
    (metadata_dir / "lineage.json").write_text(
        json.dumps(lineage, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = ProjectPaths.from_project_root(root).run_root / "reports" / "static_context_derived.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            [
                "# Static Context Derived Products",
                "",
                "No modelling features were trained or evaluated.",
                "",
                f"- data root: `{data_root}`",
                f"- station registry rows: `{len(stations):,}`",
                f"- station distance/bearing rows: `{len(distances):,}`",
                f"- solar geometry rows for {year}: `{len(solar):,}`",
                f"- station registry: `{station_csv}`",
                f"- distance matrix: `{distance_csv}`",
                f"- solar geometry: `{solar_csv}`",
                "",
                "## Completed Now",
                "",
                "- HKO target station plus NOAA ISD nearby station registry from archived raw metadata.",
                "- Deterministic station-to-station distance, bearing, and elevation-delta matrix.",
                "- Deterministic station solar-elevation/day-length table for fixed local hours.",
                "",
                "## Still Pending",
                "",
                "- Terrain elevation/slope/aspect from archived DTM rasters.",
                "- Coastline distance/bearing and land/water exposure from archived official geospatial packages.",
                "- LUHK land-use fractions and urban/vegetation fractions from archived rasters.",
                "- Horizon obstruction and upwind terrain summaries.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return StaticContextOutputs(station_csv, distance_csv, solar_csv, report)
