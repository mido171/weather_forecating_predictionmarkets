from __future__ import annotations

from collections import Counter
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

from klga_tmax.db.migrations_check import ContractInspection
from klga_tmax.registry.station_universe import (
    BACKDOOR_FRONT_STATIONS,
    CANONICAL_STATION_REGISTRY,
    COASTAL_MARINE_STATIONS,
    INLAND_HOT_REFERENCE_STATIONS,
    LONG_ISLAND_SOUND_STATIONS,
    MANDATORY_PSEUDO_POINT_REGISTRY,
    MANDATORY_STATION_REGISTRY,
    NYC_CORE_STATIONS,
    STATION_GROUPS,
    STATION_REGISTRY_VERSION,
    TARGET_STATION,
    TIER_A_POINT_IDS,
    TIER_B_POINT_IDS,
    UPSTREAM_SOUTHWEST_STATIONS,
    coordinate_tier,
    provider_station_id,
    station_group,
    tier_c_points,
)


EXPECTED_STATION_GROUPS = {
    "TARGET_STATION": (TARGET_STATION,),
    "NYC_CORE_STATIONS": NYC_CORE_STATIONS,
    "COASTAL_MARINE_STATIONS": COASTAL_MARINE_STATIONS,
    "INLAND_HOT_REFERENCE_STATIONS": INLAND_HOT_REFERENCE_STATIONS,
    "UPSTREAM_SOUTHWEST_STATIONS": UPSTREAM_SOUTHWEST_STATIONS,
    "BACKDOOR_FRONT_STATIONS": BACKDOOR_FRONT_STATIONS,
    "LONG_ISLAND_SOUND_STATIONS": LONG_ISLAND_SOUND_STATIONS,
}


def _duplicate_values(values: list[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _assert_equal(result: ContractInspection, label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        result.failures.append(f"{label} expected {expected!r}; observed {observed!r}")


def _assert_close(
    result: ContractInspection,
    label: str,
    observed: float | None,
    expected: float,
    tolerance: float = 0.000001,
) -> None:
    if observed is None or abs(float(observed) - expected) > tolerance:
        result.failures.append(f"{label} expected {expected}; observed {observed}")


def _validate_constant_contract(result: ContractInspection) -> None:
    _assert_equal(result, "mandatory station count", len(MANDATORY_STATION_REGISTRY), 19)
    _assert_equal(result, "mandatory pseudo-point count", len(MANDATORY_PSEUDO_POINT_REGISTRY), 10)
    _assert_equal(result, "canonical registry count", len(CANONICAL_STATION_REGISTRY), 29)

    station_ids = [entry.station_id for entry in CANONICAL_STATION_REGISTRY]
    duplicate_station_ids = _duplicate_values(station_ids)
    if duplicate_station_ids:
        result.failures.append(f"duplicate station_id values: {duplicate_station_ids}")

    grid_point_ids = [entry.grid_point_id for entry in MANDATORY_PSEUDO_POINT_REGISTRY]
    duplicate_grid_point_ids = _duplicate_values(grid_point_ids)
    if duplicate_grid_point_ids:
        result.failures.append(f"duplicate grid_point_id values: {duplicate_grid_point_ids}")

    _assert_equal(result, "KLGA IEM ASOS id", provider_station_id("KLGA", "iem_asos"), "LGA")
    _assert_equal(
        result,
        "KLGA Wunderground id",
        provider_station_id("KLGA", "wunderground"),
        "KLGA",
    )
    _assert_equal(result, "KLGA MOS id", provider_station_id("KLGA", "mos"), "LGA")

    _assert_equal(
        result,
        "Tier A point ids",
        tuple(entry.grid_point_id for entry in coordinate_tier("A")),
        TIER_A_POINT_IDS,
    )
    _assert_equal(
        result,
        "Tier B point ids",
        tuple(entry.grid_point_id for entry in coordinate_tier("B")),
        TIER_B_POINT_IDS,
    )

    tier_c = tier_c_points()
    _assert_equal(result, "Tier C point count", len(tier_c), 25)
    tier_c_base = [
        entry
        for entry in tier_c
        if entry.grid_point_id == "GP_KLGA_GRID_DLAT_+0.00_DLON_+0.00"
    ]
    _assert_equal(result, "Tier C KLGA base point count", len(tier_c_base), 1)
    if tier_c_base:
        _assert_close(result, "Tier C base lat", tier_c_base[0].lat, 40.77945)
        _assert_close(result, "Tier C base lon", tier_c_base[0].lon, -73.88027)

    for group_name, expected_station_ids in EXPECTED_STATION_GROUPS.items():
        _assert_equal(result, f"{group_name} constant", STATION_GROUPS[group_name], expected_station_ids)
        _assert_equal(result, f"{group_name} helper", station_group(group_name), expected_station_ids)

    for provider_name, attr_name in (
        ("iem_asos", "iem_asos_id"),
        ("wunderground", "wunderground_station_id"),
        ("mos", "mos_station_id"),
    ):
        provider_values = [
            getattr(entry, attr_name)
            for entry in MANDATORY_STATION_REGISTRY
            if getattr(entry, attr_name)
        ]
        duplicates = _duplicate_values(provider_values)
        if duplicates:
            result.failures.append(f"duplicate {provider_name} ids in constants: {duplicates}")


def _validate_database_rows(connection: Connection, result: ContractInspection) -> None:
    rows = connection.execute(
        text(
            """
            SELECT
                station_registry_version,
                station_id,
                iem_asos_id,
                wunderground_station_id,
                mos_station_id,
                grid_point_id,
                role,
                lat,
                lon,
                active_from_date,
                active_until_date,
                notes
            FROM registry.station_registry
            WHERE station_registry_version = :station_registry_version
            """
        ),
        {"station_registry_version": STATION_REGISTRY_VERSION},
    ).mappings().all()

    rows_by_key = {
        (row["station_registry_version"], row["station_id"], row["grid_point_id"]): row
        for row in rows
    }
    result.details["station_registry_rows_for_version"] = len(rows)

    _assert_equal(
        result,
        f"registry.station_registry rows for {STATION_REGISTRY_VERSION}",
        len(rows),
        len(CANONICAL_STATION_REGISTRY),
    )

    for expected in CANONICAL_STATION_REGISTRY:
        key = (STATION_REGISTRY_VERSION, expected.station_id, expected.grid_point_id)
        observed = rows_by_key.get(key)
        if observed is None:
            result.failures.append(f"missing registry.station_registry row {key}")
            continue
        _assert_equal(result, f"{expected.station_id} role", observed["role"], expected.role)
        _assert_equal(result, f"{expected.station_id} iem_asos_id", observed["iem_asos_id"], expected.iem_asos_id)
        _assert_equal(
            result,
            f"{expected.station_id} wunderground_station_id",
            observed["wunderground_station_id"],
            expected.wunderground_station_id,
        )
        _assert_equal(result, f"{expected.station_id} mos_station_id", observed["mos_station_id"], expected.mos_station_id)
        _assert_close(result, f"{expected.station_id} lat", observed["lat"], expected.lat)
        _assert_close(result, f"{expected.station_id} lon", observed["lon"], expected.lon)

    active_rows = connection.execute(
        text(
            """
            SELECT station_id, station_role, latitude, longitude, active
            FROM registry.stations
            WHERE station_id = ANY(:station_ids)
            """
        ),
        {"station_ids": [entry.station_id for entry in CANONICAL_STATION_REGISTRY]},
    ).mappings().all()
    active_rows_by_id = {row["station_id"]: row for row in active_rows}
    result.details["registry_stations_rows_for_canonical_ids"] = len(active_rows)

    for expected in CANONICAL_STATION_REGISTRY:
        observed = active_rows_by_id.get(expected.station_id)
        if observed is None:
            result.failures.append(f"missing compatibility registry.stations row {expected.station_id}")
            continue
        _assert_equal(result, f"{expected.station_id} compatibility active", observed["active"], True)
        _assert_equal(
            result,
            f"{expected.station_id} compatibility station_role",
            observed["station_role"],
            expected.role,
        )
        _assert_close(result, f"{expected.station_id} compatibility latitude", observed["latitude"], expected.lat)
        _assert_close(result, f"{expected.station_id} compatibility longitude", observed["longitude"], expected.lon)


def validate_station_universe(connection: Connection) -> ContractInspection:
    result = ContractInspection()
    _validate_constant_contract(result)
    _validate_database_rows(connection, result)
    result.details.update(
        {
            "station_registry_version": STATION_REGISTRY_VERSION,
            "mandatory_station_rows": len(MANDATORY_STATION_REGISTRY),
            "mandatory_pseudo_point_rows": len(MANDATORY_PSEUDO_POINT_REGISTRY),
            "tier_a_points": len(TIER_A_POINT_IDS),
            "tier_b_points": len(TIER_B_POINT_IDS),
            "tier_c_points": len(tier_c_points()),
        }
    )
    return result

