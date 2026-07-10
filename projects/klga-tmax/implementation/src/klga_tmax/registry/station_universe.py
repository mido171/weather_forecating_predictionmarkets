from __future__ import annotations

from dataclasses import dataclass
from typing import Any


STATION_REGISTRY_VERSION = "v2026_06_27_klga_core"


@dataclass(frozen=True)
class StationRegistryEntry:
    station_id: str
    role: str
    lat: float
    lon: float
    iem_asos_id: str | None = None
    wunderground_station_id: str | None = None
    mos_station_id: str | None = None
    grid_point_id: str = ""
    elevation_m: float | None = None
    source_native_metadata_json: dict[str, Any] | None = None
    active_from_date: str = "1900-01-01"
    active_until_date: str | None = None
    notes: str | None = None

    @property
    def is_pseudo_point(self) -> bool:
        return self.role == "gridded_pseudo_point"


TARGET_STATION = "KLGA"

NYC_CORE_STATIONS = ("KLGA", "KNYC", "KJFK", "KEWR", "KTEB")
COASTAL_MARINE_STATIONS = ("KJFK", "KISP", "KFRG", "KBDR", "KBOS")
INLAND_HOT_REFERENCE_STATIONS = ("KEWR", "KTEB", "KMMU", "KCDW", "KSWF", "KPOU", "KABE")
UPSTREAM_SOUTHWEST_STATIONS = ("KPHL", "KDCA", "KBWI", "KABE")
BACKDOOR_FRONT_STATIONS = ("KBOS", "KBDR", "KALB", "KHPN")
LONG_ISLAND_SOUND_STATIONS = ("KLGA", "KBDR", "KHPN", "KISP", "KFRG")

STATION_GROUPS: dict[str, tuple[str, ...]] = {
    "TARGET_STATION": (TARGET_STATION,),
    "NYC_CORE_STATIONS": NYC_CORE_STATIONS,
    "COASTAL_MARINE_STATIONS": COASTAL_MARINE_STATIONS,
    "INLAND_HOT_REFERENCE_STATIONS": INLAND_HOT_REFERENCE_STATIONS,
    "UPSTREAM_SOUTHWEST_STATIONS": UPSTREAM_SOUTHWEST_STATIONS,
    "BACKDOOR_FRONT_STATIONS": BACKDOOR_FRONT_STATIONS,
    "LONG_ISLAND_SOUND_STATIONS": LONG_ISLAND_SOUND_STATIONS,
}


MANDATORY_STATION_REGISTRY: tuple[StationRegistryEntry, ...] = (
    StationRegistryEntry("KLGA", "target", 40.77945, -73.88027, "LGA", "KLGA", "LGA", notes="Market settlement station, LaGuardia Airport."),
    StationRegistryEntry("KNYC", "nearby_core", 40.77898, -73.96925, "NYC", "KNYC", "NYC", notes="Central Park / Manhattan urban reference."),
    StationRegistryEntry("KJFK", "nearby_core", 40.63980, -73.77890, "JFK", "KJFK", "JFK", notes="Marine/coastal Queens/Atlantic influence."),
    StationRegistryEntry("KEWR", "nearby_core", 40.69250, -74.16870, "EWR", "KEWR", "EWR", notes="Newark/inland hot corridor reference."),
    StationRegistryEntry("KTEB", "nearby_core", 40.85899, -74.05600, "TEB", "KTEB", "TEB", notes="North Jersey / lower-Hudson inland reference."),
    StationRegistryEntry("KHPN", "nearby_core", 41.06700, -73.70760, "HPN", "KHPN", "HPN", notes="Northern suburban / inland gradient."),
    StationRegistryEntry("KISP", "nearby_core", 40.79520, -73.10020, "ISP", "KISP", "ISP", notes="Long Island inland/coastal moderation reference."),
    StationRegistryEntry("KFRG", "nearby_core", 40.72880, -73.41340, "FRG", "KFRG", "FRG", notes="Western Long Island / sea-breeze transition."),
    StationRegistryEntry("KBDR", "nearby_core", 41.16350, -73.12620, "BDR", "KBDR", "BDR", notes="Connecticut coast / Sound influence."),
    StationRegistryEntry("KSWF", "regional_context", 41.50410, -74.10480, "SWF", "KSWF", "SWF", notes="Hudson Valley hot/cool air-mass source."),
    StationRegistryEntry("KPOU", "regional_context", 41.62660, -73.88420, "POU", "KPOU", "POU", notes="Mid-Hudson Valley air-mass reference."),
    StationRegistryEntry("KMMU", "regional_context", 40.79940, -74.41490, "MMU", "KMMU", "MMU", notes="Inland New Jersey / terrain/heat gradient."),
    StationRegistryEntry("KCDW", "regional_context", 40.87520, -74.28140, "CDW", "KCDW", "CDW", notes="North Jersey local gradient."),
    StationRegistryEntry("KPHL", "regional_context", 39.87190, -75.24110, "PHL", "KPHL", "PHL", notes="Southwest corridor heat source."),
    StationRegistryEntry("KBOS", "regional_context", 42.36560, -71.00960, "BOS", "KBOS", "BOS", notes="Northeast coastal/backdoor-front reference."),
    StationRegistryEntry("KDCA", "regional_context", 38.85120, -77.04020, "DCA", "KDCA", "DCA", notes="Mid-Atlantic warm sector / upstream urban airport."),
    StationRegistryEntry("KBWI", "regional_context", 39.17540, -76.66830, "BWI", "KBWI", "BWI", notes="Mid-Atlantic upstream air mass."),
    StationRegistryEntry("KALB", "regional_context", 42.74720, -73.79910, "ALB", "KALB", "ALB", notes="Interior northeast/backdoor-front context."),
    StationRegistryEntry("KABE", "regional_context", 40.65210, -75.44080, "ABE", "KABE", "ABE", notes="Inland Pennsylvania heat/cool-front context."),
)


MANDATORY_PSEUDO_POINT_REGISTRY: tuple[StationRegistryEntry, ...] = (
    StationRegistryEntry("GP_KLGA_EXACT", "gridded_pseudo_point", 40.77945, -73.88027, grid_point_id="GP_KLGA_EXACT", notes="Exact target coordinate."),
    StationRegistryEntry("GP_KLGA_NORTH", "gridded_pseudo_point", 40.87945, -73.88027, grid_point_id="GP_KLGA_NORTH", notes="North of KLGA; Sound/backdoor/easterly gradient."),
    StationRegistryEntry("GP_KLGA_SOUTH", "gridded_pseudo_point", 40.67945, -73.88027, grid_point_id="GP_KLGA_SOUTH", notes="South of KLGA; Queens/Brooklyn influence."),
    StationRegistryEntry("GP_KLGA_EAST", "gridded_pseudo_point", 40.77945, -73.78027, grid_point_id="GP_KLGA_EAST", notes="East/coastal Queens and marine influence."),
    StationRegistryEntry("GP_KLGA_WEST", "gridded_pseudo_point", 40.77945, -73.98027, grid_point_id="GP_KLGA_WEST", notes="Manhattan/urban west reference."),
    StationRegistryEntry("GP_KLGA_NW_INLAND_NJ", "gridded_pseudo_point", 40.86000, -74.15000, grid_point_id="GP_KLGA_NW_INLAND_NJ", notes="Inland west/northwest warm-source point."),
    StationRegistryEntry("GP_KLGA_SW_NEWARK_CORRIDOR", "gridded_pseudo_point", 40.70000, -74.17000, grid_point_id="GP_KLGA_SW_NEWARK_CORRIDOR", notes="Newark corridor heat-source point."),
    StationRegistryEntry("GP_KLGA_E_LONG_ISLAND", "gridded_pseudo_point", 40.78000, -73.40000, grid_point_id="GP_KLGA_E_LONG_ISLAND", notes="Long Island sea-breeze penetration reference."),
    StationRegistryEntry("GP_KLGA_SOUND_WATER_PROXY", "gridded_pseudo_point", 40.90000, -73.80000, grid_point_id="GP_KLGA_SOUND_WATER_PROXY", notes="Long Island Sound/water-side thermal proxy."),
    StationRegistryEntry("GP_KLGA_ATLANTIC_PROXY", "gridded_pseudo_point", 40.60000, -73.70000, grid_point_id="GP_KLGA_ATLANTIC_PROXY", notes="Atlantic/Jamaica Bay marine proxy."),
)


CANONICAL_STATION_REGISTRY: tuple[StationRegistryEntry, ...] = (
    MANDATORY_STATION_REGISTRY + MANDATORY_PSEUDO_POINT_REGISTRY
)

TIER_A_POINT_IDS = (
    "GP_KLGA_EXACT",
    "GP_KLGA_NW_INLAND_NJ",
    "GP_KLGA_E_LONG_ISLAND",
    "GP_KLGA_ATLANTIC_PROXY",
)

TIER_KLGA_POINT_IDS = ("GP_KLGA_EXACT",)

TIER_B_POINT_IDS = (
    "GP_KLGA_EXACT",
    "GP_KLGA_NORTH",
    "GP_KLGA_SOUTH",
    "GP_KLGA_EAST",
    "GP_KLGA_WEST",
    "GP_KLGA_NW_INLAND_NJ",
    "GP_KLGA_SW_NEWARK_CORRIDOR",
    "GP_KLGA_E_LONG_ISLAND",
    "GP_KLGA_SOUND_WATER_PROXY",
    "GP_KLGA_ATLANTIC_PROXY",
)

_TIER_C_OFFSETS = (-0.10, -0.05, 0.00, 0.05, 0.10)
_TIER_C_BASE_LAT = 40.77945
_TIER_C_BASE_LON = -73.88027


def registry_entry_by_station_id(station_id: str) -> StationRegistryEntry:
    for entry in CANONICAL_STATION_REGISTRY:
        if entry.station_id == station_id:
            return entry
    raise KeyError(f"unknown station_id {station_id}")


def registry_entry_by_grid_point_id(grid_point_id: str) -> StationRegistryEntry:
    for entry in MANDATORY_PSEUDO_POINT_REGISTRY:
        if entry.grid_point_id == grid_point_id:
            return entry
    raise KeyError(f"unknown grid_point_id {grid_point_id}")


def provider_station_id(station_id: str, provider: str) -> str | None:
    entry = registry_entry_by_station_id(station_id)
    provider_key = provider.lower().replace("-", "_")
    if provider_key in {"iem", "iem_asos", "asos"}:
        return entry.iem_asos_id
    if provider_key in {"wunderground", "wu"}:
        return entry.wunderground_station_id
    if provider_key in {"mos", "iem_mos"}:
        return entry.mos_station_id
    raise KeyError(f"unknown provider {provider}")


def station_group(group_name: str) -> tuple[str, ...]:
    try:
        return STATION_GROUPS[group_name]
    except KeyError as exc:
        raise KeyError(f"unknown station group {group_name}") from exc


def tier_c_points() -> tuple[StationRegistryEntry, ...]:
    rows: list[StationRegistryEntry] = []
    for lat_offset in _TIER_C_OFFSETS:
        for lon_offset in _TIER_C_OFFSETS:
            grid_point_id = f"GP_KLGA_GRID_DLAT_{lat_offset:+.2f}_DLON_{lon_offset:+.2f}"
            rows.append(
                StationRegistryEntry(
                    station_id=grid_point_id,
                    role="gridded_pseudo_point",
                    lat=round(_TIER_C_BASE_LAT + lat_offset, 5),
                    lon=round(_TIER_C_BASE_LON + lon_offset, 5),
                    grid_point_id=grid_point_id,
                    notes="Generated Tier C KLGA 5x5 research grid point.",
                )
            )
    return tuple(rows)


def coordinate_tier(tier: str) -> tuple[StationRegistryEntry, ...]:
    tier_key = tier.upper()
    if tier_key in {"KLGA", "A_KLGA", "A1"}:
        return tuple(registry_entry_by_grid_point_id(point_id) for point_id in TIER_KLGA_POINT_IDS)
    if tier_key == "A":
        return tuple(registry_entry_by_grid_point_id(point_id) for point_id in TIER_A_POINT_IDS)
    if tier_key == "B":
        return tuple(registry_entry_by_grid_point_id(point_id) for point_id in TIER_B_POINT_IDS)
    if tier_key == "C":
        return tier_c_points()
    raise KeyError(f"unknown coordinate tier {tier}")
