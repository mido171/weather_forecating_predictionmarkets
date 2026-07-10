from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StationSeed:
    station_id: str
    tier: str
    group_name: str
    active: bool
    traded_station: bool
    kalshi_series: str | None = None
    metadata_lookup_station_id: str | None = None


OPTIONAL_KALSHI_SERIES: dict[str, str] = {
    "KATL": "KXHIGHTATL",
    "KPHL": "KXHIGHPHIL",
    "KNYC": "KXHIGHNY",
    "KMIA": "KXHIGHMIA",
    "KMDW": "KXHIGHCHI",
    "KLAX": "KXHIGHLAX",
}


_ACTIVE_GROUPS: tuple[tuple[str, str, list[str], set[str]], ...] = (
    ("A", "mandatory_traded", ["KNYC", "KLAX", "KMIA", "KPHL"], {"KNYC", "KLAX", "KMIA", "KPHL"}),
    (
        "A",
        "nyc_phl_corridor_ring",
        [
            "KJFK",
            "KLGA",
            "KEWR",
            "KTEB",
            "KHPN",
            "KISP",
            "KBDR",
            "KSWF",
            "KHFD",
            "KBDL",
            "KHVN",
            "KGON",
            "KACY",
            "KILG",
            "KTTN",
            "KABE",
            "KRDG",
            "KDOV",
            "KGED",
            "KPVD",
            "KBOS",
            "KDCA",
            "KBWI",
            "KIAD",
            "KORF",
        ],
        set(),
    ),
    (
        "A",
        "south_florida_core",
        ["KFLL", "KPBI", "KAPF", "KRSW", "KMCO", "KMLB", "KTPA", "KSRQ"],
        set(),
    ),
    (
        "A",
        "socal_core",
        ["KSNA", "KLGB", "KBUR", "KVNY", "KONT", "KSAN", "KOXR", "KSBA", "KSMX", "KBFL", "KPSP", "KDAG"],
        set(),
    ),
    (
        "B",
        "great_lakes_upper_midwest",
        [
            "KBUF",
            "KROC",
            "KSYR",
            "KCLE",
            "KERI",
            "KDTW",
            "KMKG",
            "KGRR",
            "KAZO",
            "KLAN",
            "KORD",
            "KMDW",
            "KMKE",
            "KMSN",
            "KSBN",
            "KTVC",
            "KAPN",
            "KPLN",
            "KMQT",
            "KDLH",
            "KRST",
        ],
        set(),
    ),
    (
        "B",
        "midwest_ohio_valley",
        ["KDSM", "KOMA", "KLNK", "KMCI", "KSTL", "KIND", "KCVG", "KSDF", "KLEX"],
        set(),
    ),
    (
        "B",
        "southeast_gulf",
        ["KATL", "KBHM", "KHSV", "KCHA", "KCLT", "KCHS", "KSAV", "KMOB", "KGPT", "KMSY", "KBTR", "KIAH", "KHOU", "KCRP"],
        set(),
    ),
    ("B", "southern_plains_benchmark", ["KDFW", "KOKC", "KTUL"], set()),
    (
        "C",
        "high_plains_front_range",
        [
            "KICT",
            "KDDC",
            "KHYS",
            "KAMA",
            "KLBB",
            "KABI",
            "KSJT",
            "KMAF",
            "KGLD",
            "KLBF",
            "KFSD",
            "KBIS",
            "KDEN",
            "KBJC",
            "KCOS",
            "KPUB",
            "KCYS",
            "KCPR",
            "KRIW",
        ],
        set(),
    ),
    (
        "C",
        "interior_west_desert",
        ["KSLC", "KBOI", "KTWF", "KRNO", "KIDA", "KEKO", "KGJT", "KPHX", "KIWA", "KTUS", "KYUM", "KLAS", "KELP", "KABQ", "KFLG"],
        set(),
    ),
    (
        "C",
        "pnw_california_support",
        ["KSEA", "KPAE", "KPDX", "KEUG", "KMFR", "KAST", "KSMF", "KFAT", "KMOD", "KSFO"],
        set(),
    ),
)

_RESERVE_GROUPS: tuple[tuple[str, str, list[str], set[str]], ...] = (
    (
        "D",
        "reserve",
        ["KJAX", "KDAB", "KGNV", "KTLH", "KPNS", "KCAE", "KGSP", "KRIC", "KALB", "KPWM", "KCMI", "KPIA", "KORH", "KMHT", "KMRY", "KCHO", "KROA", "KSAT", "KGEG", "KSLE"],
        set(),
    ),
)


def _build_seeds(
    groups: tuple[tuple[str, str, list[str], set[str]], ...],
    *,
    active: bool,
) -> list[StationSeed]:
    seeds: list[StationSeed] = []
    for tier, group_name, station_ids, traded in groups:
        for station_id in station_ids:
            seeds.append(
                StationSeed(
                    station_id=station_id,
                    tier=tier,
                    group_name=group_name,
                    active=active,
                    traded_station=station_id in traded,
                    kalshi_series=OPTIONAL_KALSHI_SERIES.get(station_id),
                    metadata_lookup_station_id={"KMQT": "KSAW"}.get(station_id),
                )
            )
    return seeds


ACTIVE_STATION_SEEDS: list[StationSeed] = _build_seeds(_ACTIVE_GROUPS, active=True)
RESERVE_STATION_SEEDS: list[StationSeed] = _build_seeds(_RESERVE_GROUPS, active=False)
ALL_STATION_SEEDS: list[StationSeed] = [*ACTIVE_STATION_SEEDS, *RESERVE_STATION_SEEDS]


def _validate_station_universe() -> None:
    station_ids = [seed.station_id for seed in ALL_STATION_SEEDS]
    if len(station_ids) != len(set(station_ids)):
        dupes = sorted({station_id for station_id in station_ids if station_ids.count(station_id) > 1})
        raise ValueError(f"Duplicate station ids in pooled universe: {dupes}")
    if len(ACTIVE_STATION_SEEDS) != 140:
        raise ValueError(f"Expected 140 active stations, found {len(ACTIVE_STATION_SEEDS)}")
    if len(RESERVE_STATION_SEEDS) != 20:
        raise ValueError(f"Expected 20 reserve stations, found {len(RESERVE_STATION_SEEDS)}")
    if len(ALL_STATION_SEEDS) != 160:
        raise ValueError(f"Expected 160 total stations, found {len(ALL_STATION_SEEDS)}")


_validate_station_universe()


def get_station_seeds(scope: str = "all") -> list[StationSeed]:
    scope_norm = str(scope).strip().lower()
    if scope_norm == "active":
        return list(ACTIVE_STATION_SEEDS)
    if scope_norm == "reserve":
        return list(RESERVE_STATION_SEEDS)
    if scope_norm == "all":
        return list(ALL_STATION_SEEDS)
    station_ids = [token.strip().upper() for token in scope.split(",") if token.strip()]
    lookup = {seed.station_id: seed for seed in ALL_STATION_SEEDS}
    out: list[StationSeed] = []
    for station_id in station_ids:
        if station_id not in lookup:
            raise ValueError(f"Unknown pooled-strategy station id: {station_id}")
        out.append(lookup[station_id])
    return out
