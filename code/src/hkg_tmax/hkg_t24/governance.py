from __future__ import annotations

import csv
import importlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

from hkg_tmax.config import load_yaml

from .guard import LOCKED_TEST_START

DEFAULT_DATA_ROOT = Path(r"C:\hkg_tmax_data")
HKT_TIMEZONE = "Asia/Hong_Kong"

AVAILABILITY_TIERS: dict[str, dict[str, object]] = {
    "GOLD_EXACT_VINTAGE": {
        "operational_input_allowed": True,
        "description": "Exact historical payload/vintage is preserved and timestamp-proven.",
    },
    "SILVER_OPERATIONAL_REPLAY": {
        "operational_input_allowed": True,
        "description": "Observation/report plausibly existed by cutoff; conservative latency is used.",
    },
    "MECHANISM_ONLY": {
        "operational_input_allowed": False,
        "description": "Retrospectively useful for physical analysis but forbidden as operational input.",
    },
    "PROSPECTIVE_ONLY": {
        "operational_input_allowed": False,
        "description": "Usable only for live-shadow forecasts after collection began.",
    },
    "TARGET_ONLY": {
        "operational_input_allowed": False,
        "description": "Usable only as a label or label-side diagnostic.",
    },
    "FORBIDDEN": {
        "operational_input_allowed": False,
        "description": "Known or unresolved leakage risk; do not use in model inputs.",
    },
}

HKO_TEMPERATURE_AND_MAXMIN_STATIONS = (
    "Chek Lap Kok",
    "Cheung Chau",
    "Clear Water Bay",
    "HK Observatory",
    "HK Park",
    "Happy Valley",
    "Kai Tak Runway Park",
    "Kau Sai Chau",
    "King's Park",
    "Kowloon City",
    "Kwun Tong",
    "Lau Fau Shan",
    "Ngong Ping",
    "Pak Tam Chung",
    "Peng Chau",
    "Sai Kung",
    "Sha Tin",
    "Sham Shui Po",
    "Shau Kei Wan",
    "Shek Kong",
    "Sheung Shui",
    "Stanley",
    "Ta Kwu Ling",
    "Tai Lung",
    "Tai Mei Tuk",
    "Tai Mo Shan",
    "Tai Po",
    "Tate's Cairn",
    "The Peak",
    "Tseung Kwan O",
    "Tsing Yi",
    "Tsuen Wan Ho Koon",
    "Tsuen Wan Shing Mun Valley",
    "Tuen Mun",
    "Waglan Island",
    "Wetland Park",
    "Wong Chuk Hang",
    "Wong Tai Sin",
    "Yuen Long Park",
)

HKO_HUMIDITY_STATIONS = (
    "Chek Lap Kok",
    "Cheung Chau",
    "Clear Water Bay",
    "HK Observatory",
    "HK Park",
    "Kai Tak Runway Park",
    "Kau Sai Chau",
    "King's Park",
    "Kowloon City",
    "Lau Fau Shan",
    "Pak Tam Chung",
    "Peng Chau",
    "Sai Kung",
    "Sha Tin",
    "Shau Kei Wan",
    "Shek Kong",
    "Sheung Shui",
    "Ta Kwu Ling",
    "Tai Lung",
    "Tai Po",
    "Tseung Kwan O",
    "Tsing Yi",
    "Tsuen Wan Ho Koon",
    "Tsuen Wan Shing Mun Valley",
    "Tuen Mun",
    "Waglan Island",
    "Wetland Park",
    "Wong Chuk Hang",
)

HKO_PRESSURE_STATIONS = (
    "Chek Lap Kok",
    "Cheung Chau",
    "HK Observatory",
    "Lau Fau Shan",
    "Peng Chau",
    "Sha Tin",
    "Shek Kong",
    "Sheung Shui",
    "Ta Kwu Ling",
    "Tai Po",
    "Waglan Island",
    "Wetland Park",
)

HKO_SOLAR_STATIONS = ("Kau Sai Chau", "King's Park")

HKO_WIND_STATIONS = (
    "Central Pier",
    "Chek Lap Kok",
    "Cheung Chau",
    "Cheung Chau Beach",
    "Green Island",
    "Hong Kong Sea School",
    "Kai Tak",
    "King's Park",
    "Lamma Island",
    "Lau Fau Shan",
    "Ngong Ping",
    "North Point",
    "Peng Chau",
    "Sai Kung",
    "Sha Chau",
    "Sha Tin",
    "Shek Kong",
    "Stanley",
    "Star Ferry",
    "Ta Kwu Ling",
    "Tai Mei Tuk",
    "Tai Po Kau",
    "Tap Mun",
    "Tate's Cairn",
    "Tseung Kwan O",
    "Tsing Yi",
    "Tuen Mun",
    "Waglan Island",
    "Wetland Park",
    "Wong Chuk Han",
    "Wong Chuk Hang",
)

HKO_DAILY_CLIMATE_GROUPS = {
    "Hong Kong Observatory": (
        "mean_temperature",
        "max_temperature",
        "min_temperature",
        "mean_sea_level_pressure",
        "dew_point_temperature",
        "wet_bulb_temperature",
        "relative_humidity",
        "cloud_amount",
        "rainfall",
        "grass_minimum_temperature",
    ),
    "King's Park": ("bright_sunshine", "global_solar_radiation", "evaporation"),
    "Waglan Island": ("prevailing_wind_direction", "wind_speed", "sea_temperature"),
    "North Point": ("sea_temperature_am", "sea_temperature_pm"),
    "Hong Kong Territory": ("cloud_to_ground_lightning", "cloud_to_cloud_lightning"),
    "Hong Kong International Airport": ("reduced_visibility_hours",),
}


class GovernanceError(RuntimeError):
    """Raised when T24 governance artifacts cannot be built safely."""


@dataclass(frozen=True)
class OOFFeasibility:
    status: str
    available_days: int
    required_days: int
    available_years: float
    start_date: str
    end_date: str
    reason: str


@dataclass(frozen=True)
class GovernanceOutputs:
    station_registry_csv: Path
    station_registry_parquet: Path
    research_ledger_csv: Path
    research_ledger_parquet: Path
    source_contracts_yaml: Path
    feature_catalog_yaml: Path
    availability_tiers_yaml: Path
    asof_contract_yaml: Path
    evaluation_design_yaml: Path
    reports: tuple[Path, ...]


def _pd() -> Any:
    return importlib.import_module("pandas")


def _pa() -> Any:
    return importlib.import_module("pyarrow")


def _pq() -> Any:
    return importlib.import_module("pyarrow.parquet")


def _now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _slug(value: str) -> str:
    normalized = value.lower().replace("&", "and").replace("'", "")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_parquet(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = _pa().Table.from_pylist([dict(row) for row in rows])
    _pq().write_table(table, path, compression="zstd")


def _write_yaml(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _row_string_set(row: dict[str, object], field: str) -> set[str]:
    value = row.get(field)
    if isinstance(value, set):
        parsed = {str(item) for item in value}
        row[field] = parsed
        return parsed
    if isinstance(value, str) and value:
        parsed_from_text = {item for item in value.split(",") if item}
        row[field] = parsed_from_text
        return parsed_from_text
    empty_set: set[str] = set()
    row[field] = empty_set
    return empty_set


def check_four_year_oof_feasibility(
    start_date: date,
    end_date: date,
    *,
    min_years: float = 4.0,
    reason_context: str,
) -> OOFFeasibility:
    if end_date < start_date:
        return OOFFeasibility(
            status="BLOCKED",
            available_days=0,
            required_days=int(round(min_years * 365.25)),
            available_years=0.0,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            reason=f"{reason_context}: end date precedes start date",
        )
    available_days = (end_date - start_date).days + 1
    required_days = int(round(min_years * 365.25))
    available_years = available_days / 365.25
    if available_days >= required_days:
        status = "PASS"
        reason = f"{reason_context}: {available_years:.2f} years available"
    else:
        status = "BLOCKED"
        reason = (
            f"{reason_context}: {available_years:.2f} years available, "
            f"requires at least {min_years:.1f} years"
        )
    return OOFFeasibility(
        status=status,
        available_days=available_days,
        required_days=required_days,
        available_years=round(available_years, 3),
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        reason=reason,
    )


def tier_for_point_in_time_status(status: str, *, role: str = "") -> str:
    normalized = status.upper()
    role_normalized = role.lower()
    if "MARKET" in normalized or "polymarket" in role_normalized:
        return "FORBIDDEN"
    if "TARGET_ONLY" in normalized:
        return "TARGET_ONLY"
    if "RETROSPECTIVE" in normalized:
        return "MECHANISM_ONLY"
    if normalized in {"STATIC_METADATA", "METADATA"}:
        return "SILVER_OPERATIONAL_REPLAY"
    if normalized == "OPERATIONAL_POINT_IN_TIME":
        return "GOLD_EXACT_VINTAGE"
    if normalized in {"PROXY_WITH_LIMITATIONS", "POTENTIAL_POINT_IN_TIME_ARCHIVE"}:
        return "SILVER_OPERATIONAL_REPLAY"
    return "FORBIDDEN"


def operational_allowed_for_tier(tier: str) -> bool:
    data = AVAILABILITY_TIERS.get(tier)
    return bool(data and data.get("operational_input_allowed"))


def _station_rows_from_required_lists() -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}

    def add(station_name: str, feed: str) -> None:
        station_id = f"hko_name:{_slug(station_name)}"
        row = rows.setdefault(
            station_name,
            {
                "canonical_station_id": station_id,
                "station_name": station_name,
                "network": "HKO",
                "official_station_code": "",
                "official_code_status": "pending_official_metadata_resolution",
                "alias_status": "unmerged_name_preserved",
                "feed_membership": set(),
                "source_evidence": set(),
                "target_station": station_name in {"HK Observatory", "Hong Kong Observatory"},
                "notes": "",
            },
        )
        _row_string_set(row, "feed_membership").add(feed)
        _row_string_set(row, "source_evidence").add("goal_required_station_list")

    for station in HKO_TEMPERATURE_AND_MAXMIN_STATIONS:
        add(station, "temperature_1min")
        add(station, "since_midnight_maxmin")
    for station in HKO_HUMIDITY_STATIONS:
        add(station, "humidity_1min")
    for station in HKO_PRESSURE_STATIONS:
        add(station, "pressure_1min")
    for station in HKO_SOLAR_STATIONS:
        add(station, "solar_1min")
    for station in HKO_WIND_STATIONS:
        add(station, "wind_10min")
    for domain, variables in HKO_DAILY_CLIMATE_GROUPS.items():
        add(domain, "daily_climate:" + ",".join(variables))

    if "Wong Chuk Han" in rows:
        rows["Wong Chuk Han"][
            "notes"
        ] = "Potential source typo or separate wind-row alias; not merged with Wong Chuk Hang."
    return rows


def _read_optional_parquet(path: Path) -> Any:
    if not path.exists():
        return None
    return _pd().read_parquet(path)


def _finalize_station_rows(rows: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    finalized: list[dict[str, object]] = []
    for row in rows.values():
        out = dict(row)
        feeds = sorted(_row_string_set(out, "feed_membership"))
        evidence = sorted(_row_string_set(out, "source_evidence"))
        out["feed_membership"] = ",".join(feeds)
        out["source_evidence"] = ",".join(evidence)
        finalized.append(out)
    return sorted(finalized, key=lambda item: (str(item.get("network")), str(item.get("station_name"))))


def build_station_registry(root: Path, data_root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = _station_rows_from_required_lists()
    cutoff_path = data_root / "silver" / "observations" / "hko_station_temperature_cutoff_summary.parquet"
    selected_path = data_root / "bronze" / "analysis_phase_a" / "hko_high_frequency_selected_station_observations.parquet"
    daily_path = data_root / "bronze" / "analysis_phase_a" / "hko_daily_climate_elements.parquet"
    static_path = data_root / "metadata" / "static_context" / "station_registry.parquet"

    cutoff = _read_optional_parquet(cutoff_path)
    if cutoff is not None:
        for station_name, group in cutoff.groupby("station"):
            key = str(station_name)
            row = rows.setdefault(
                key,
                {
                    "canonical_station_id": f"hko_name:{_slug(key)}",
                    "station_name": key,
                    "network": "HKO",
                    "official_station_code": "",
                    "official_code_status": "pending_official_metadata_resolution",
                    "alias_status": "unmerged_name_preserved",
                    "feed_membership": set(),
                    "source_evidence": set(),
                    "target_station": key in {"HK Observatory", "Hong Kong Observatory"},
                    "notes": "",
                },
            )
            _row_string_set(row, "feed_membership").add("silver_cutoff_temperature")
            _row_string_set(row, "source_evidence").add(str(cutoff_path))
            row["temperature_cutoff_start"] = str(group["local_date"].min().date())
            row["temperature_cutoff_end"] = str(group["local_date"].max().date())
            row["temperature_cutoff_rows"] = int(len(group))

    selected = _read_optional_parquet(selected_path)
    if selected is not None:
        for (station_name, family, variable), group in selected.groupby(["station", "family", "variable"]):
            key = str(station_name)
            row = rows.setdefault(
                key,
                {
                    "canonical_station_id": f"hko_name:{_slug(key)}",
                    "station_name": key,
                    "network": "HKO",
                    "official_station_code": "",
                    "official_code_status": "pending_official_metadata_resolution",
                    "alias_status": "unmerged_name_preserved",
                    "feed_membership": set(),
                    "source_evidence": set(),
                    "target_station": key in {"HK Observatory", "Hong Kong Observatory"},
                    "notes": "",
                },
            )
            _row_string_set(row, "feed_membership").add(f"{family}:{variable}")
            _row_string_set(row, "source_evidence").add(str(selected_path))
            row["selected_hf_start"] = str(group["observed_at_hkt"].min())
            row["selected_hf_end"] = str(group["observed_at_hkt"].max())

    daily = _read_optional_parquet(daily_path)
    if daily is not None:
        for (domain, variable), group in daily.groupby(["station_or_domain", "variable"]):
            key = str(domain)
            row = rows.setdefault(
                key,
                {
                    "canonical_station_id": f"hko_daily:{_slug(key)}",
                    "station_name": key,
                    "network": "HKO_DAILY_CLIMATE",
                    "official_station_code": "",
                    "official_code_status": "domain_or_station_pending_official_metadata_resolution",
                    "alias_status": "daily_climate_domain_preserved",
                    "feed_membership": set(),
                    "source_evidence": set(),
                    "target_station": key == "Hong Kong Observatory",
                    "notes": "",
                },
            )
            _row_string_set(row, "feed_membership").add(f"daily_climate:{variable}")
            _row_string_set(row, "source_evidence").add(str(daily_path))
            row[f"daily_{variable}_start"] = str(group["local_date"].min().date())
            row[f"daily_{variable}_end"] = str(group["local_date"].max().date())

    static = _read_optional_parquet(static_path)
    if static is not None:
        for record in static.to_dict(orient="records"):
            name = str(record.get("station_name", ""))
            network = str(record.get("network", ""))
            if network == "HKO":
                key = "Hong Kong Observatory"
                row = rows.setdefault(
                    key,
                    {
                        "canonical_station_id": "hko:HKO",
                        "station_name": key,
                        "network": "HKO",
                        "official_station_code": "HKO",
                        "official_code_status": "configured_target_candidate_pending_g1",
                        "alias_status": "target_alias_family",
                        "feed_membership": set(),
                        "source_evidence": set(),
                        "target_station": True,
                        "notes": "",
                    },
                )
                row["official_station_code"] = "HKO"
                row["official_code_status"] = "configured_target_candidate_pending_g1"
                row["canonical_station_id"] = "hko:HKO"
                row["latitude"] = record.get("latitude", "")
                row["longitude"] = record.get("longitude", "")
                row["elevation_m"] = record.get("elevation_m", "")
                _row_string_set(row, "source_evidence").add(str(static_path))
            else:
                key = str(record.get("station_id", ""))
                rows[key] = {
                    "canonical_station_id": f"noaa_isd:{key}",
                    "station_name": name,
                    "network": "NOAA_ISD",
                    "official_station_code": key,
                    "official_code_status": "noaa_station_history_id",
                    "alias_status": "separate_network_do_not_merge_without_overlap_validation",
                    "feed_membership": {"noaa_isd_history"},
                    "source_evidence": {str(static_path)},
                    "target_station": False,
                    "latitude": record.get("latitude", ""),
                    "longitude": record.get("longitude", ""),
                    "elevation_m": record.get("elevation_m", ""),
                    "notes": "Regional proxy station; final archive availability is not exact operational vintage.",
                }

    finalized = _finalize_station_rows(rows)
    summary = {
        "generated_at_utc": _now_utc(),
        "rows": len(finalized),
        "hko_rows": sum(1 for row in finalized if str(row.get("network", "")).startswith("HKO")),
        "noaa_isd_rows": sum(1 for row in finalized if row.get("network") == "NOAA_ISD"),
        "wong_chuk_han_preserved": any(row.get("station_name") == "Wong Chuk Han" for row in finalized),
    }
    return finalized, summary


def build_source_contracts(root: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    catalog = load_yaml(root / "config" / "data_sources.yaml")
    sources = catalog.get("sources")
    if not isinstance(sources, list):
        raise GovernanceError("config/data_sources.yaml sources must be a list")
    contracts: list[dict[str, object]] = []
    for item in sources:
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("id", ""))
        role = str(item.get("role", ""))
        status = str(item.get("point_in_time_status", ""))
        tier = tier_for_point_in_time_status(status, role=role)
        is_market = tier == "FORBIDDEN" and "polymarket" in json.dumps(item, sort_keys=True).lower()
        contracts.append(
            {
                "source_id": source_id,
                "provider": item.get("provider", ""),
                "role": role,
                "priority": item.get("priority", ""),
                "cadence": item.get("cadence", ""),
                "historical_range": item.get("historical_range", ""),
                "point_in_time_status": status,
                "availability_tier": tier,
                "operational_input_allowed_by_tier": operational_allowed_for_tier(tier),
                "market_source_forbidden_in_this_goal": is_market,
                "availability_rule": item.get("availability_rule", ""),
                "revision_policy": item.get("revision_policy", ""),
                "tags": item.get("tags", []),
            }
        )
    payload = {
        "schema_version": 1,
        "generated_at_utc": _now_utc(),
        "locked_test_start": LOCKED_TEST_START.isoformat(),
        "polymarket_allowed": False,
        "contracts": contracts,
    }
    return payload, contracts


def build_feature_catalog(data_root: Path) -> dict[str, object]:
    feature_path = data_root / "silver" / "features" / "t24_cutoff_feature_candidates.parquet"
    registry_path = data_root / "metadata" / "feature_eligibility_registry.parquet"
    features = _pd().read_parquet(feature_path)
    registry = _pd().read_parquet(registry_path) if registry_path.exists() else None
    registry_by_name: dict[str, Mapping[str, object]] = {}
    if registry is not None:
        for record in registry.to_dict(orient="records"):
            registry_by_name[str(record.get("feature_name", ""))] = record
    entries: list[dict[str, object]] = []
    for column in features.columns:
        if column in {"local_date", "cutoff_hkt", "split_role"} or column.endswith("_observed_at_hkt") or column.endswith("_available_at_hkt"):
            continue
        availability_column = f"{column}_available_at_hkt"
        observed_column = f"{column}_observed_at_hkt"
        record = registry_by_name.get(column, {})
        role = str(record.get("role", "UNREGISTERED_FEATURE"))
        tier = "FORBIDDEN"
        if role == "TARGET_ONLY":
            tier = "TARGET_ONLY"
        elif role in {
            "OPERATIONAL_WITH_CONSERVATIVE_LATENCY",
            "OPERATIONAL_POINT_IN_TIME",
            "PROXY_WITH_LIMITATIONS",
        }:
            tier = "SILVER_OPERATIONAL_REPLAY"
        elif role == "RETROSPECTIVE_MECHANISM_ONLY":
            tier = "MECHANISM_ONLY"
        non_null = features[features[column].notna()]
        entries.append(
            {
                "feature_name": column,
                "role": role,
                "availability_tier": tier,
                "operational_input_allowed_by_tier": operational_allowed_for_tier(tier),
                "target_derived": bool(record.get("target_derived", column == "target_tmax_c")),
                "registered": column in registry_by_name,
                "has_available_at_column": availability_column in features.columns,
                "has_observed_at_column": observed_column in features.columns,
                "non_null_rows": int(len(non_null)),
                "first_non_null_date": None if non_null.empty else str(non_null["local_date"].min().date()),
                "last_non_null_date": None if non_null.empty else str(non_null["local_date"].max().date()),
                "available_at_rule": record.get("available_at_rule", ""),
                "notes": record.get("notes", ""),
            }
        )
    return {
        "schema_version": 1,
        "generated_at_utc": _now_utc(),
        "feature_table": str(feature_path),
        "feature_count": len(entries),
        "features": entries,
    }


def _retrieval_coverage(data_root: Path) -> list[dict[str, object]]:
    ledger_path = data_root / "manifests" / "retrieval_ledger.parquet"
    ledger = _pd().read_parquet(ledger_path)
    rows: list[dict[str, object]] = []
    for source_id, group in ledger.groupby("source_id"):
        success = group[group["status"] == "success"]
        rows.append(
            {
                "source_id": str(source_id),
                "retrieval_rows": int(len(group)),
                "success_rows": int(len(success)),
                "first_success_retrieved_at": None if success.empty else str(success["retrieved_at"].min()),
                "last_success_retrieved_at": None if success.empty else str(success["retrieved_at"].max()),
                "bytes_success": int(success["content_length"].fillna(0).astype(int).sum()) if not success.empty else 0,
                "unique_hashes": int(success["content_sha256"].nunique()) if not success.empty else 0,
            }
        )
    return sorted(rows, key=lambda row: str(row["source_id"]))


def _research_plan_rows(root: Path) -> list[dict[str, object]]:
    titles = {
        "HKG-T24-R01": "Baseline Reproduction, Eligibility Gaps, and Research Firewall",
        "HKG-T24-R02": "Long-History Climatology, Trend, and Training-Window Value",
        "HKG-T24-R03": "Official Tmax Reconstruction and Time-of-Maximum Anatomy",
        "HKG-T24-R04": "HKO Cutoff Thermal-Trajectory Model",
        "HKG-T24-R05": "Multi-Day Thermal Memory and Regime Persistence",
        "HKG-T24-R06": "Moisture-State and Dew-Point Thermodynamics",
        "HKG-T24-R07": "Pressure-Tendency, Front, and Cold-Surge Transition Detection",
        "HKG-T24-R08": "Surface Wind, Advection, and Sea-Breeze Regime",
        "HKG-T24-R09": "All-Station Temperature Gradient and Thermal-Field Experiment",
        "HKG-T24-R10": "Latent Spatial Modes, Graph Structure, and Field Coherence",
        "HKG-T24-R11": "Dynamic Upwind Station Selection and Flow-Relative Advection",
        "HKG-T24-R12": "Solar Radiation, UV, and Heating-Efficiency Experiment",
        "HKG-T24-R13": "Cloud, Rain, Visibility, and Surface-Wetness Suppression",
        "HKG-T24-R14": "Eligible Upper-Air Thermal Potential and Inversion Structure",
        "HKG-T24-R15": "Surface-Upper-Air Coupling and Mixing-Potential Experiment",
        "HKG-T24-R16": "Fifty-Year Regional ISD Surface Core",
        "HKG-T24-R17": "Station Metadata Breaks, Urbanization, and Era Transfer",
        "HKG-T24-R18": "HKO Official Forecast Baseline, Bias Correction, and MOS",
        "HKG-T24-R19": "Analogue-System Redesign and Learned Similarity",
        "HKG-T24-R20": "Physically Defined Regime Classifier and Specialist Experts",
        "HKG-T24-R21": "Data-Driven Regime Discovery and Cluster Stability",
        "HKG-T24-R22": "Abrupt Transition and Catastrophic-Error Specialist",
        "HKG-T24-R23": "Extreme-Heat and Tropical-Cyclone-Adjacent Subsidence Specialist",
        "HKG-T24-R24": "Marine, Sea-Temperature, Coastline, and Terrain Interaction",
        "HKG-T24-R25": "Privileged-Information Teacher/Student Auxiliary Targets",
        "HKG-T24-R26": "Multi-Era Hierarchical Core and Modern Residual Booster",
        "HKG-T24-R27": "Operational Latency, Missing-Station, and Data-Outage Robustness",
        "HKG-T24-R28": "Transparent Nonlinear Model-Family Benchmark",
        "HKG-T24-R29": "Conditional Distribution, Calibration, and OOF Expert Ensemble",
        "HKG-T24-R30": "Predeclared Final Challenger Freeze and One-Shot Validation",
    }
    known_folders = {
        "HKG-T24-R01": root / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0033-HKG-T24-R01",
        "HKG-T24-R02": root / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0034-HKG-T24-R02",
        "HKG-T24-R03": root / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0035-HKG-T24-R03",
        "HKG-T24-R04": root / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0036-HKG-T24-R04",
        "HKG-T24-R05": root / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0037-HKG-T24-R05",
        "HKG-T24-R06": root / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0038-HKG-T24-R06",
        "HKG-T24-R07": root / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0039-HKG-T24-R07",
        "HKG-T24-R08": root / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0040-HKG-T24-R08",
        "HKG-T24-R09": root / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0041-HKG-T24-R09",
        "HKG-T24-R10": root / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0042-HKG-T24-R10",
        "HKG-T24-R11": root / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0043-HKG-T24-R11",
        "HKG-T24-R12": root / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0044-HKG-T24-R12",
        "HKG-T24-R13": root / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0045-HKG-T24-R13",
        "HKG-T24-R14": root / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0046-HKG-T24-R14",
        "HKG-T24-R15": root / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0047-HKG-T24-R15",
        "HKG-T24-R16": root / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0048-HKG-T24-R16",
        "HKG-T24-R17": root / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0049-HKG-T24-R17",
        "HKG-T24-R18": root / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0050-HKG-T24-R18",
        "HKG-T24-R19": root / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0051-HKG-T24-R19",
        "HKG-T24-R20": root / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0052-HKG-T24-R20",
        "HKG-T24-R21": root / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0053-HKG-T24-R21",
        "HKG-T24-R22": root / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0054-HKG-T24-R22",
        "HKG-T24-R23": root / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0055-HKG-T24-R23",
        "HKG-T24-R24": root / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0056-HKG-T24-R24",
        "HKG-T24-R25": root / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0057-HKG-T24-R25",
        "HKG-T24-R26": root / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0058-HKG-T24-R26",
        "HKG-T24-R27": root / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0059-HKG-T24-R27",
        "HKG-T24-R28": root / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0060-HKG-T24-R28",
        "HKG-T24-R29": root / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0061-HKG-T24-R29",
        "HKG-T24-R30": root / "analysis" / "hkg_tmax_t24" / "experiments" / "EXP-0062-HKG-T24-R30",
    }
    rows: list[dict[str, object]] = []
    for number in range(1, 31):
        research_id = f"HKG-T24-R{number:02d}"
        folder = known_folders.get(research_id)
        status = "PENDING"
        evidence = ""
        if folder is not None and folder.exists():
            status_path = folder / "STATUS.yaml"
            if status_path.exists():
                parsed_status = yaml.safe_load(status_path.read_text(encoding="utf-8")) or {}
                status = str(parsed_status.get("status", "COMPLETE_DOCUMENTED"))
            else:
                status = "FOLDER_EXISTS_STATUS_MISSING"
            evidence = str(folder)
        rows.append(
            {
                "research_id": research_id,
                "title": titles.get(research_id, "Pending title from beastmode goal sequence"),
                "status": status,
                "evidence_path": evidence,
                "validation_2024_accessed": research_id == "HKG-T24-R01",
                "locked_test_accessed": False,
                "notes": (
                    "R01 validation access was limited to supplied baseline reproduction."
                    if research_id == "HKG-T24-R01"
                    else "No validation/locked access recorded."
                ),
            }
        )
    return rows


def _research_ledger_report(rows: Sequence[Mapping[str, object]]) -> str:
    counts = Counter(str(row.get("status", "")) for row in rows)
    lines = [
        "# HKG T24 Research Ledger",
        "",
        f"Generated: `{_now_utc()}`",
        "",
        f"- status counts: `{dict(sorted(counts.items()))}`",
        "- locked-test access recorded: `false`",
        "",
        "| Research ID | Title | Status | Evidence |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("research_id", "")),
                    str(row.get("title", "")),
                    str(row.get("status", "")),
                    str(row.get("evidence_path", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _source_coverage_report(rows: Sequence[Mapping[str, object]], contracts: Sequence[Mapping[str, object]]) -> str:
    coverage_by_id = {str(row["source_id"]): row for row in rows}
    tier_counts = Counter(str(row.get("availability_tier", "")) for row in contracts)
    lines = [
        "# HKG T24 Source Coverage",
        "",
        f"Generated: `{_now_utc()}`",
        "",
        f"- source contracts: `{len(contracts)}`",
        f"- sources with retrieval rows: `{len(rows)}`",
        f"- availability tiers: `{dict(sorted(tier_counts.items()))}`",
        "",
        "| Source | Tier | Operational allowed | Success rows | Unique hashes | First success | Last success |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for contract in contracts:
        if contract.get("market_source_forbidden_in_this_goal"):
            continue
        source_id = str(contract.get("source_id", ""))
        cov = coverage_by_id.get(source_id, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    source_id,
                    str(contract.get("availability_tier", "")),
                    str(contract.get("operational_input_allowed_by_tier", "")),
                    str(cov.get("success_rows", 0)),
                    str(cov.get("unique_hashes", 0)),
                    str(cov.get("first_success_retrieved_at", "")),
                    str(cov.get("last_success_retrieved_at", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _station_registry_report(rows: Sequence[Mapping[str, object]], summary: Mapping[str, object]) -> str:
    lines = [
        "# HKG T24 Station Registry",
        "",
        f"Generated: `{summary['generated_at_utc']}`",
        "",
        f"- total rows: `{summary['rows']}`",
        f"- HKO/domain rows: `{summary['hko_rows']}`",
        f"- NOAA ISD rows: `{summary['noaa_isd_rows']}`",
        f"- `Wong Chuk Han` preserved separately: `{summary['wong_chuk_han_preserved']}`",
        "",
        "Official HKO station codes remain pending unless explicitly configured. Names are not merged by string similarity.",
        "",
        "| Station | Network | Code/status | Target | Feed membership | Notes |",
        "|---|---|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("station_name", "")),
                    str(row.get("network", "")),
                    f"{row.get('official_station_code', '')} / {row.get('official_code_status', '')}",
                    str(row.get("target_station", False)),
                    str(row.get("feed_membership", "")),
                    str(row.get("notes", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _point_in_time_report(feature_catalog: Mapping[str, object], oof_payload: Mapping[str, object]) -> str:
    features = feature_catalog["features"]
    assert isinstance(features, list)
    long_history = oof_payload.get("long_history", {})
    modern_high_frequency = oof_payload.get("modern_high_frequency", {})
    assert isinstance(long_history, dict)
    assert isinstance(modern_high_frequency, dict)
    registered = sum(1 for row in features if isinstance(row, dict) and row.get("registered"))
    forbidden = [
        row for row in features if isinstance(row, dict) and not row.get("operational_input_allowed_by_tier")
    ]
    lines = [
        "# HKG T24 Point-In-Time Eligibility",
        "",
        f"Generated: `{feature_catalog['generated_at_utc']}`",
        "",
        "- Cutoff: T-1 15:00:00 Asia/Hong_Kong.",
        "- Governing timestamp: `available_at`.",
        f"- Locked-test ordinary access denied from: `{LOCKED_TEST_START.isoformat()}`.",
        f"- feature entries: `{len(features)}`",
        f"- registered in existing eligibility registry: `{registered}`",
        f"- not operationally allowed by tier: `{len(forbidden)}`",
        "",
        "## Four-Year OOF Gate",
        "",
        f"- strict requirement: `{oof_payload['requirement']}`",
        f"- long-history status: `{long_history['status']}` - {long_history['reason']}",
        f"- modern high-frequency status: `{modern_high_frequency['status']}` - {modern_high_frequency['reason']}",
        "",
        "| Feature | Tier | Registered | Available-at column | Non-null rows | Range | Notes |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in features:
        if not isinstance(row, dict):
            continue
        date_range = f"{row.get('first_non_null_date')} to {row.get('last_non_null_date')}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("feature_name", "")),
                    str(row.get("availability_tier", "")),
                    str(row.get("registered", "")),
                    str(row.get("has_available_at_column", "")),
                    str(row.get("non_null_rows", "")),
                    date_range,
                    str(row.get("notes", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _publication_latency_report(feature_catalog: Mapping[str, object], source_contracts: Sequence[Mapping[str, object]]) -> str:
    hko_operational = [
        row
        for row in source_contracts
        if str(row.get("provider", "")).startswith("Hong Kong Observatory")
        and row.get("operational_input_allowed_by_tier")
    ]
    lines = [
        "# HKG T24 Publication Latency",
        "",
        f"Generated: `{_now_utc()}`",
        "",
        "Current enforceable latency rules are conservative and source-specific. No experiment may select by `observed_at` alone.",
        "",
        "## Active Rules",
        "",
        "- HKO historical high-frequency station observations: `available_at = observed_at + 20 minutes`.",
        "- Exact retrieved live HKO vintages: available no earlier than successful immutable retrieval time.",
        "- HKO daily climate and Daily Extract labels: target/label-side only unless first-publication timing is proven.",
        "- Reanalysis, final IMERG, final TC best track and retrospective archives: mechanism-only unless exact operational vintage and release lag are reconstructed.",
        "",
        f"HKO operational source contracts currently allowed by tier: `{len(hko_operational)}`.",
        "",
        "| Source | Cadence | Availability rule | Revision policy |",
        "|---|---|---|---|",
    ]
    for row in hko_operational:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("source_id", "")),
                    str(row.get("cadence", "")),
                    str(row.get("availability_rule", "")),
                    str(row.get("revision_policy", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _oof_payload(data_root: Path) -> dict[str, object]:
    target = _pd().read_parquet(data_root / "silver" / "targets" / "hko_daily_tmax.parquet")
    features = _pd().read_parquet(
        data_root / "silver" / "features" / "t24_cutoff_feature_candidates.parquet",
        columns=["local_date", "hko_temp_at_tminus1_1500_c"],
    )
    target["local_date"] = _pd().to_datetime(target["local_date"])
    features["local_date"] = _pd().to_datetime(features["local_date"])
    long_history = check_four_year_oof_feasibility(
        target["local_date"].min().date(),
        target[target["local_date"].dt.date < date(2024, 1, 1)]["local_date"].max().date(),
        reason_context="target/daily climate development history before validation 2024",
    )
    modern = features[
        features["hko_temp_at_tminus1_1500_c"].notna()
        & (features["local_date"].dt.date < date(2024, 1, 1))
    ]
    modern_start = modern["local_date"].min().date()
    modern_end = modern["local_date"].max().date()
    modern_feasibility = check_four_year_oof_feasibility(
        modern_start,
        modern_end,
        reason_context="modern HKO high-frequency development history before validation 2024",
    )
    return {
        "requirement": "at least four years of OOF test data for all experiments",
        "long_history": long_history.__dict__,
        "modern_high_frequency": modern_feasibility.__dict__,
        "policy": (
            "Modern high-frequency experiments are blocked under the strict four-year rule unless "
            "the evaluation design is explicitly revised without using locked-test data."
        ),
    }


def write_hkg_t24_governance(root: Path, *, data_root: Path = DEFAULT_DATA_ROOT) -> GovernanceOutputs:
    config_dir = root / "config" / "hkg_t24"
    reports_dir = root / "reports" / "hkg_t24"
    config_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    station_rows, station_summary = build_station_registry(root, data_root)
    station_csv = config_dir / "station_registry.csv"
    station_parquet = config_dir / "station_registry.parquet"
    _write_csv(station_csv, station_rows)
    _write_parquet(station_parquet, station_rows)

    research_rows = _research_plan_rows(root)
    research_dir = root / "research"
    research_csv = research_dir / "HKG_T24_RESEARCH_LEDGER.csv"
    research_parquet = research_dir / "HKG_T24_RESEARCH_LEDGER.parquet"
    _write_csv(research_csv, research_rows)
    _write_parquet(research_parquet, research_rows)

    source_payload, source_rows = build_source_contracts(root)
    source_yaml = config_dir / "source_contracts.yaml"
    _write_yaml(source_yaml, source_payload)

    feature_payload = build_feature_catalog(data_root)
    feature_yaml = config_dir / "feature_catalog.yaml"
    _write_yaml(feature_yaml, feature_payload)

    tiers_yaml = config_dir / "availability_tiers.yaml"
    _write_yaml(
        tiers_yaml,
        {
            "schema_version": 1,
            "generated_at_utc": _now_utc(),
            "tiers": AVAILABILITY_TIERS,
        },
    )

    asof_yaml = config_dir / "asof_t24_1500.yaml"
    _write_yaml(
        asof_yaml,
        {
            "schema_version": 1,
            "timezone": HKT_TIMEZONE,
            "forecast_question": "HKO Headquarters official daily Tmax for local day T",
            "cutoff_hkt": "T-1 15:00:00",
            "cutoff_utc": "T-1 07:00:00",
            "governing_timestamp": "available_at",
            "eligibility_rule": "feature.available_at_hkt <= cutoff_hkt",
            "locked_test_start": LOCKED_TEST_START.isoformat(),
            "ordinary_research_locked_test_policy": "deny",
            "forbidden": [
                "target-day observation or target-derived value as predictor",
                "T-1 full-day aggregate at 15:00 T-1",
                "centered rolling windows crossing cutoff",
                "validation or locked-test rows used to fit preprocessing, feature selection, calibration, or model choice",
                "final reanalysis, final TC best tracks, final retrospective precipitation as operational input",
            ],
        },
    )

    oof = _oof_payload(data_root)
    long_history = oof["long_history"]
    modern_high_frequency = oof["modern_high_frequency"]
    assert isinstance(long_history, dict)
    assert isinstance(modern_high_frequency, dict)
    eval_yaml = config_dir / "evaluation_design.yaml"
    _write_yaml(
        eval_yaml,
        {
            "schema_version": 1,
            "generated_at_utc": _now_utc(),
            "locked_test_start": LOCKED_TEST_START.isoformat(),
            "validation_2024_budget": "R01 reproduction only until R30 preregistration",
            "four_year_oof": oof,
        },
    )
    _write_json(reports_dir / "oof_feasibility.json", oof)

    coverage_rows = _retrieval_coverage(data_root)
    reports = (
        reports_dir / "station_registry.md",
        reports_dir / "source_coverage.md",
        reports_dir / "point_in_time_eligibility.md",
        reports_dir / "publication_latency.md",
        reports_dir / "OOF_FEASIBILITY.md",
        reports_dir / "RESEARCH_LEDGER.md",
    )
    reports[0].write_text(_station_registry_report(station_rows, station_summary), encoding="utf-8")
    reports[1].write_text(_source_coverage_report(coverage_rows, source_rows), encoding="utf-8")
    reports[2].write_text(_point_in_time_report(feature_payload, oof), encoding="utf-8")
    reports[3].write_text(_publication_latency_report(feature_payload, source_rows), encoding="utf-8")
    reports[4].write_text(
        "\n".join(
            [
                "# HKG T24 Four-Year OOF Feasibility",
                "",
                f"Generated: `{_now_utc()}`",
                "",
                f"- requirement: `{oof['requirement']}`",
                f"- long-history status: `{long_history['status']}` - {long_history['reason']}",
                f"- modern high-frequency status: `{modern_high_frequency['status']}` - {modern_high_frequency['reason']}",
                f"- policy: {oof['policy']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    reports[5].write_text(_research_ledger_report(research_rows), encoding="utf-8")

    return GovernanceOutputs(
        station_registry_csv=station_csv,
        station_registry_parquet=station_parquet,
        research_ledger_csv=research_csv,
        research_ledger_parquet=research_parquet,
        source_contracts_yaml=source_yaml,
        feature_catalog_yaml=feature_yaml,
        availability_tiers_yaml=tiers_yaml,
        asof_contract_yaml=asof_yaml,
        evaluation_design_yaml=eval_yaml,
        reports=reports,
    )
