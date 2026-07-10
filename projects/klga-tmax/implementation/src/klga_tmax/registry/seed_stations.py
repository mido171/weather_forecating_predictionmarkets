from __future__ import annotations

import json

from sqlalchemy import text
from sqlalchemy.engine import Connection

from klga_tmax.registry.station_universe import (
    CANONICAL_STATION_REGISTRY,
    STATION_GROUPS,
    STATION_REGISTRY_VERSION,
    StationRegistryEntry,
)


ALL_STATION_SEEDS = CANONICAL_STATION_REGISTRY

_STATION_NAMES = {
    "KLGA": "LaGuardia Airport",
    "KNYC": "Central Park Manhattan",
    "KJFK": "John F. Kennedy International Airport",
    "KEWR": "Newark Liberty International Airport",
    "KTEB": "Teterboro Airport",
    "KHPN": "Westchester County Airport",
    "KISP": "Long Island MacArthur Airport",
    "KFRG": "Republic Airport",
    "KBDR": "Igor I. Sikorsky Memorial Airport",
    "KSWF": "Stewart International Airport",
    "KPOU": "Hudson Valley Regional Airport",
    "KMMU": "Morristown Municipal Airport",
    "KCDW": "Essex County Airport",
    "KPHL": "Philadelphia International Airport",
    "KBOS": "Boston Logan International Airport",
    "KDCA": "Ronald Reagan Washington National Airport",
    "KBWI": "Baltimore/Washington International Airport",
    "KALB": "Albany International Airport",
    "KABE": "Lehigh Valley International Airport",
}


def _station_name(entry: StationRegistryEntry) -> str:
    if entry.is_pseudo_point:
        return f"{entry.grid_point_id} KLGA gridded pseudo-point"
    return _STATION_NAMES[entry.station_id]


def _provider_primary_id(entry: StationRegistryEntry) -> str:
    return (
        entry.wunderground_station_id
        or entry.iem_asos_id
        or entry.mos_station_id
        or entry.grid_point_id
        or entry.station_id
    )


def _compatibility_groups(entry: StationRegistryEntry) -> list[str]:
    groups = {entry.role}
    for group_name, station_ids in STATION_GROUPS.items():
        if entry.station_id in station_ids:
            groups.add(group_name.lower())
    if entry.is_pseudo_point:
        groups.add("gridded_pseudo_point")
    return sorted(groups)


def seed_station_registry(connection: Connection) -> int:
    rows_changed = 0
    for entry in CANONICAL_STATION_REGISTRY:
        result = connection.execute(
            text(
                """
                INSERT INTO registry.station_registry (
                    station_registry_version,
                    station_id,
                    iem_asos_id,
                    wunderground_station_id,
                    mos_station_id,
                    grid_point_id,
                    role,
                    lat,
                    lon,
                    elevation_m,
                    source_native_metadata_json,
                    active_from_date,
                    active_until_date,
                    notes
                )
                VALUES (
                    :station_registry_version,
                    :station_id,
                    :iem_asos_id,
                    :wunderground_station_id,
                    :mos_station_id,
                    :grid_point_id,
                    :role,
                    :lat,
                    :lon,
                    :elevation_m,
                    CAST(:source_native_metadata_json AS jsonb),
                    :active_from_date,
                    :active_until_date,
                    :notes
                )
                ON CONFLICT (station_registry_version, station_id, grid_point_id)
                DO UPDATE SET
                    iem_asos_id = EXCLUDED.iem_asos_id,
                    wunderground_station_id = EXCLUDED.wunderground_station_id,
                    mos_station_id = EXCLUDED.mos_station_id,
                    role = EXCLUDED.role,
                    lat = EXCLUDED.lat,
                    lon = EXCLUDED.lon,
                    elevation_m = EXCLUDED.elevation_m,
                    source_native_metadata_json = EXCLUDED.source_native_metadata_json,
                    active_from_date = EXCLUDED.active_from_date,
                    active_until_date = EXCLUDED.active_until_date,
                    notes = EXCLUDED.notes
                """
            ),
            {
                "station_registry_version": STATION_REGISTRY_VERSION,
                "station_id": entry.station_id,
                "iem_asos_id": entry.iem_asos_id,
                "wunderground_station_id": entry.wunderground_station_id,
                "mos_station_id": entry.mos_station_id,
                "grid_point_id": entry.grid_point_id,
                "role": entry.role,
                "lat": entry.lat,
                "lon": entry.lon,
                "elevation_m": entry.elevation_m,
                "source_native_metadata_json": json.dumps(
                    entry.source_native_metadata_json or {},
                    sort_keys=True,
                ),
                "active_from_date": entry.active_from_date,
                "active_until_date": entry.active_until_date,
                "notes": entry.notes,
            },
        )
        rows_changed += result.rowcount or 0
    return rows_changed


def seed_stations(connection: Connection) -> int:
    rows_changed = 0
    active_station_ids = [entry.station_id for entry in CANONICAL_STATION_REGISTRY]
    for entry in CANONICAL_STATION_REGISTRY:
        result = connection.execute(
            text(
                """
                INSERT INTO registry.stations (
                    station_id,
                    station_name,
                    provider_primary_id,
                    latitude,
                    longitude,
                    station_role,
                    station_group,
                    active
                )
                VALUES (
                    :station_id,
                    :station_name,
                    :provider_primary_id,
                    :latitude,
                    :longitude,
                    :station_role,
                    :station_group,
                    true
                )
                ON CONFLICT (station_id) DO UPDATE SET
                    station_name = EXCLUDED.station_name,
                    provider_primary_id = EXCLUDED.provider_primary_id,
                    latitude = EXCLUDED.latitude,
                    longitude = EXCLUDED.longitude,
                    station_role = EXCLUDED.station_role,
                    station_group = EXCLUDED.station_group,
                    active = true
                """
            ),
            {
                "station_id": entry.station_id,
                "station_name": _station_name(entry),
                "provider_primary_id": _provider_primary_id(entry),
                "latitude": entry.lat,
                "longitude": entry.lon,
                "station_role": entry.role,
                "station_group": _compatibility_groups(entry),
            },
        )
        rows_changed += result.rowcount or 0

    stale_result = connection.execute(
        text(
            """
            UPDATE registry.stations
            SET active = false
            WHERE NOT (station_id = ANY(:active_station_ids))
              AND active = true
            """
        ),
        {"active_station_ids": active_station_ids},
    )
    rows_changed += stale_result.rowcount or 0
    return rows_changed
