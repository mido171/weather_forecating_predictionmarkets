"""Fixed Hong Kong station groups for HKO residual modeling.

The groups are intentionally static. Dynamic grouping would make the model
harder to audit across changing station panels.
"""

from __future__ import annotations

from dataclasses import dataclass


URBAN_CORE = [
    "KING'S PARK",
    "HONG KONG PARK",
    "HAPPY VALLEY",
    "SHAM SHUI PO",
    "KOWLOON CITY",
    "WONG TAI SIN",
    "KWUN TONG",
    "KAI TAK RUNWAY PARK",
    "SHAU KEI WAN",
]

COASTAL_MARINE = [
    "CHEK LAP KOK",
    "CHEUNG CHAU",
    "SAI KUNG",
    "STANLEY",
    "WONG CHUK HANG",
    "TSEUNG KWAN O",
]

INLAND_NT = [
    "SHA TIN",
    "TA KWU LING",
    "SHEK KONG",
    "YUEN LONG PARK",
    "TAI PO",
    "TAI MEI TUK",
]

WEST_NW_NT = [
    "LAU FAU SHAN",
    "TUEN MUN",
    "TSING YI",
    "TSUEN WAN",
    "TSUEN WAN HO KOON",
    "TSUEN WAN SHING MUN VALLEY",
]

CORE_STATION_DELTAS = [
    "KING'S PARK",
    "HONG KONG PARK",
    "CHEK LAP KOK",
    "CHEUNG CHAU",
    "SHA TIN",
    "TA KWU LING",
    "LAU FAU SHAN",
    "SAI KUNG",
    "WONG CHUK HANG",
    "SHEK KONG",
    "TSING YI",
    "TSEUNG KWAN O",
]


@dataclass(frozen=True)
class StationGroup:
    name: str
    stations: tuple[str, ...]


STATION_GROUPS = (
    StationGroup("urban_core", tuple(URBAN_CORE)),
    StationGroup("coastal_marine", tuple(COASTAL_MARINE)),
    StationGroup("inland_nt", tuple(INLAND_NT)),
    StationGroup("west_nw_nt", tuple(WEST_NW_NT)),
)

ALL_MODELED_STATIONS = tuple(
    dict.fromkeys(
        [
            *URBAN_CORE,
            *COASTAL_MARINE,
            *INLAND_NT,
            *WEST_NW_NT,
            *CORE_STATION_DELTAS,
        ]
    )
)


def station_feature_name(station: str, suffix: str) -> str:
    safe = (
        station.lower()
        .replace("'", "")
        .replace(" ", "_")
        .replace("-", "_")
    )
    return f"station_{safe}_{suffix}"

