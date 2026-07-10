from __future__ import annotations

import argparse
import csv
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

Row = dict[str, str]


@dataclass(frozen=True)
class FamilyRule:
    family: str
    status: str
    matcher: Callable[[str], bool]
    remaining: str


@dataclass
class FamilyCoverage:
    family: str
    status: str
    remaining: str
    rows: int = 0
    source_ids: set[str] | None = None
    bytes: int = 0
    date_tokens: set[str] | None = None

    def __post_init__(self) -> None:
        if self.source_ids is None:
            self.source_ids = set()
        if self.date_tokens is None:
            self.date_tokens = set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate source-family acquisition coverage from the raw retrieval ledger."
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("HKG_TMAX_DATA_ROOT", r"C:\hkg_tmax_data"),
        help="Canonical acquisition data root.",
    )
    parser.add_argument(
        "--output",
        default="reports/source_family_coverage.md",
        help="Markdown report path to write inside the repository.",
    )
    return parser.parse_args()


def sid_starts(*prefixes: str) -> Callable[[str], bool]:
    return lambda source_id: source_id.startswith(prefixes)


def sid_any(*source_ids: str) -> Callable[[str], bool]:
    allowed = set(source_ids)
    return lambda source_id: source_id in allowed


def sid_contains(*parts: str) -> Callable[[str], bool]:
    return lambda source_id: any(part in source_id for part in parts)


def any_match(*matchers: Callable[[str], bool]) -> Callable[[str], bool]:
    return lambda source_id: any(matcher(source_id) for matcher in matchers)


def target_daily_climate(source_id: str) -> bool:
    return (
        source_id == "hko_clmmaxt_hko"
        or source_id.startswith("hko_daily_extract")
        or (source_id.startswith("hko_daily_climate_") and "sea_temp" not in source_id)
    )


def hko_high_frequency(source_id: str) -> bool:
    return (
        source_id.startswith("datagov_hko_historical_latest_")
        or (
            source_id.startswith("hko_latest_")
            and source_id != "hko_latest_tidal_information"
        )
        or source_id
        in {
            "hko_since_midnight_maxmin",
            "hko_automatic_rainfall",
            "hko_current_weather_report",
        }
    )


FAMILY_RULES: tuple[FamilyRule, ...] = (
    FamilyRule(
        "A target labels / daily climate",
        "DOWNLOADED",
        target_daily_climate,
        "Parse to bronze target/daily climate tables; no additional raw download known.",
    ),
    FamilyRule(
        "B station/catalog metadata",
        "DOWNLOADED_INITIAL",
        sid_any("hko_open_data_catalog", "hko_station_metadata", "hko_api_documentation_pdf"),
        "Parse station metadata and reconcile station aliases/history.",
    ),
    FamilyRule(
        "C high-frequency HKO observations",
        "PARTIAL_WITH_HISTORICAL_BACKFILL",
        hko_high_frequency,
        "Rainfall, visibility, and direct RHR JSON older histories were not found; collect prospectively or find another official archive.",
    ),
    FamilyRule(
        "D forecasts / warnings / ARWF",
        "PARTIAL_WITH_HISTORICAL_RSS",
        any_match(
            sid_starts("datagov_hko_historical_rss_", "hko_arwf_"),
            sid_any(
                "hko_local_weather_forecast",
                "hko_nine_day_forecast",
                "hko_weather_warning_summary",
                "hko_weather_warning_information",
                "hko_special_weather_tips",
            ),
        ),
        "Historical JSON forecast versions still need another official archive; install prospective collectors.",
    ),
    FamilyRule(
        "E operational NWP",
        "PARTIAL_CURRENT_NCEP",
        sid_starts("ncep_"),
        "Full historical/continuous GFS/GEFS/ECMWF/DWD/AI archives need approved byte-budgeted subsets.",
    ),
    FamilyRule(
        "F upper-air",
        "DOWNLOADED",
        sid_starts("noaa_igra_"),
        "Parse IGRA period-of-record and year-to-date archives.",
    ),
    FamilyRule(
        "G radar / rainfall nowcast / lightning",
        "DOWNLOADED_INITIAL",
        any_match(sid_starts("hko_radar_", "hko_lightning_"), sid_any("hko_gridded_rainfall_nowcast")),
        "Historical imagery/backfill limited unless another official archive is found; collect prospectively.",
    ),
    FamilyRule(
        "H satellite / cloud / aerosol",
        "PARTIAL_CURRENT_ARCHIVED",
        sid_starts("hko_satellite_"),
        "Keep current HKO satellite collectors running; historical Himawari/archive-scale satellite acquisition remains byte-budgeted.",
    ),
    FamilyRule(
        "I tropical cyclone / regional surface",
        "PARTIAL_WITH_SURFACE_ARCHIVE",
        any_match(sid_starts("noaa_isd_", "hko_tropical_cyclone_")),
        "Operational advisory vintages before acquisition remain unavailable unless another archive is found.",
    ),
    FamilyRule(
        "J marine / ocean",
        "DOWNLOADED_INITIAL",
        any_match(
            sid_any("hko_south_china_coastal_waters_bulletin", "hko_latest_tidal_information"),
            sid_contains("sea_temp"),
        ),
        "Gridded SST/OISST/OSTIA needs product choice and byte budget.",
    ),
    FamilyRule(
        "L static geospatial context",
        "PARTIAL_DERIVED_CONTEXT",
        any_match(
            sid_starts("landsd_", "csdi_", "pland_"),
            sid_starts("data_gov_hk_landsd_", "data_gov_hk_pland_"),
        ),
        "Station registry, distance/bearing, and solar geometry are derived; terrain/coastline/LUHK context still needs source-specific parsers.",
    ),
)


def read_ledger(data_root: Path) -> list[Row]:
    with (data_root / "manifests" / "retrieval_ledger.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        return list(csv.DictReader(handle))


def as_int(value: str | None) -> int:
    try:
        return int(value or "0")
    except ValueError:
        return 0


def row_url(row: Row) -> str:
    return row.get("request_url") or row.get("final_url") or ""


def date_tokens(url: str) -> set[str]:
    tokens: set[str] = set()
    parsed = urlparse(url)
    for value in parse_qs(parsed.query).get("time", []):
        if re.fullmatch(r"\d{8}", value):
            tokens.add(value)
    for pattern in (
        r"(?:dailyExtract_|HKO)(\d{4})(?:BST)?",
        r"dailyExtract_(\d{6})",
        r"((?:18|19|20)\d{6,10})",
    ):
        tokens.update(re.findall(pattern, url))
    return tokens


def classify(source_id: str) -> FamilyRule | None:
    for rule in FAMILY_RULES:
        if rule.matcher(source_id):
            return rule
    return None


def build_coverage(rows: list[Row]) -> tuple[list[FamilyCoverage], list[str]]:
    coverage = {
        rule.family: FamilyCoverage(rule.family, rule.status, rule.remaining)
        for rule in FAMILY_RULES
    }
    unclassified: set[str] = set()
    for row in rows:
        if row.get("status") != "success":
            continue
        source_id = row.get("source_id", "")
        rule = classify(source_id)
        if rule is None:
            unclassified.add(source_id)
            continue
        family = coverage[rule.family]
        family.rows += 1
        family.bytes += as_int(row.get("content_length"))
        assert family.source_ids is not None
        assert family.date_tokens is not None
        family.source_ids.add(source_id)
        family.date_tokens.update(date_tokens(row_url(row)))
    return list(coverage.values()), sorted(unclassified)


def write_report(output: Path, data_root: Path, coverage: list[FamilyCoverage], unclassified: list[str]) -> None:
    lines = [
        "# Source Family Coverage",
        "",
        "Generated offline from the canonical raw retrieval ledger. This report is",
        "acquisition evidence only; it does not perform modelling or mutate the raw archive.",
        "",
        f"- data root: `{data_root}`",
        "",
        "| Family | Status | Success rows | Source IDs | Bytes | Date/token range | Remaining acquisition work |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for family in coverage:
        source_count = len(family.source_ids or set())
        tokens = sorted(family.date_tokens or set())
        token_range = f"{tokens[0]} to {tokens[-1]}" if tokens else ""
        lines.append(
            f"| {family.family} | `{family.status}` | {family.rows:,} | {source_count:,} | {family.bytes:,} | {token_range} | {family.remaining} |"
        )
    lines.extend(["", "## Unclassified Successful Source IDs", ""])
    if unclassified:
        for source_id in unclassified:
            lines.append(f"- `{source_id}`")
    else:
        lines.append("None.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    rows = read_ledger(data_root)
    coverage, unclassified = build_coverage(rows)
    write_report(Path(args.output), data_root, coverage, unclassified)


if __name__ == "__main__":
    main()
