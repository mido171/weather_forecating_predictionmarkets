from __future__ import annotations

import csv
import importlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .acquisition import ensure_data_root, inspect_data_root
from .config import load_yaml

PAIR_REQUIRED_FIELDS = (
    "family_id",
    "feature_family",
    "historical_source_id",
    "historical_start",
    "historical_end",
    "historical_native_cadence",
    "historical_revision_status",
    "live_source_id",
    "live_endpoint_or_bucket",
    "live_native_cadence",
    "expected_live_latency",
    "actual_live_latency_statistics",
    "live_issue_time_field",
    "live_valid_time_field",
    "live_available_at_rule",
    "historical_live_schema_compatibility",
    "historical_live_known_differences",
    "operational_point_in_time_eligibility",
    "license_and_commercial_use_status",
    "backfill_adapter",
    "live_adapter",
    "collector_schedule",
    "last_historical_success",
    "last_live_success",
    "coverage_status",
    "blocker",
)

GRIDDED_FAMILY_REQUIRED_FIELDS = (
    "family_id",
    "provider",
    "product",
    "priority",
    "status",
    "historical_period",
    "domains",
    "variables",
    "pressure_levels_hpa",
    "cycles",
    "members",
    "leads",
    "temporal_cadence",
    "estimated_bytes",
    "actual_bytes",
    "server_side_subset_method",
    "live_retention_policy",
    "credential_status",
    "bulk_download_allowed",
    "blocker",
)


class AcquisitionContractError(RuntimeError):
    """Raised when acquisition contracts are missing required fields."""


def _iso_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
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
    table = pa.Table.from_pylist([dict(row) for row in rows])
    pq.write_table(table, path, compression="zstd")


def _stringify(value: object) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return "" if value is None else str(value)


def _flat_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, str]]:
    return [{key: _stringify(value) for key, value in row.items()} for row in rows]


def load_pair_rows(root: Path) -> list[dict[str, object]]:
    data = load_yaml(root / "config" / "historical_live_pairs.yaml")
    pairs = data.get("pairs")
    if not isinstance(pairs, list):
        raise AcquisitionContractError("config/historical_live_pairs.yaml pairs must be a list")
    rows: list[dict[str, object]] = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise AcquisitionContractError(f"historical_live_pairs pairs[{index}] must be a mapping")
        missing = [field for field in PAIR_REQUIRED_FIELDS if field not in pair]
        if missing:
            family = pair.get("family_id", index)
            raise AcquisitionContractError(f"historical_live_pair {family!r} missing fields: {missing}")
        rows.append(pair)
    return rows


def validate_historical_live_pairs(root: Path) -> list[str]:
    rows = load_pair_rows(root)
    family_ids = [str(row["family_id"]) for row in rows]
    duplicates = sorted(name for name, count in Counter(family_ids).items() if count > 1)
    if duplicates:
        raise AcquisitionContractError(f"Duplicate historical/live family IDs: {duplicates}")
    if any("polymarket" in json.dumps(row, sort_keys=True).lower() for row in rows):
        raise AcquisitionContractError("Historical/live pair contract must exclude Polymarket")
    return [f"historical/live pair contract: {len(rows)} families with required fields"]


def load_gridded_policy(root: Path) -> dict[str, Any]:
    return load_yaml(root / "config" / "gridded_acquisition_policy.yaml")


def validate_gridded_policy(root: Path) -> list[str]:
    data = load_gridded_policy(root)
    domains = data.get("domains")
    if not isinstance(domains, dict):
        raise AcquisitionContractError("gridded_acquisition_policy domains must be a mapping")
    for required_domain in ("local_hk", "regional_schina", "synoptic_asia"):
        if required_domain not in domains:
            raise AcquisitionContractError(f"gridded policy missing domain {required_domain}")
        domain = domains[required_domain]
        if not isinstance(domain, dict):
            raise AcquisitionContractError(f"gridded policy domain {required_domain} must be a mapping")
        for field in ("south", "north", "west", "east"):
            if field not in domain:
                raise AcquisitionContractError(f"gridded policy domain {required_domain} missing {field}")
    families = data.get("families")
    if not isinstance(families, list) or not families:
        raise AcquisitionContractError("gridded_acquisition_policy families must be a non-empty list")
    for index, family in enumerate(families):
        if not isinstance(family, dict):
            raise AcquisitionContractError(f"gridded policy families[{index}] must be a mapping")
        missing = [field for field in GRIDDED_FAMILY_REQUIRED_FIELDS if field not in family]
        if missing:
            raise AcquisitionContractError(
                f"gridded policy family {family.get('family_id', index)!r} missing fields: {missing}"
            )
        if family.get("bulk_download_allowed") is True and str(family.get("blocker", "")):
            raise AcquisitionContractError(
                f"gridded policy family {family.get('family_id')} cannot be allowed and blocked"
            )
    return [f"gridded acquisition policy: {len(families)} families and {len(domains)} domains"]


def _ledger_rows(root: Path) -> list[dict[str, str]]:
    data_root = ensure_data_root(root)
    return _read_csv(data_root / "manifests" / "retrieval_ledger.csv")


def _success_rows(root: Path) -> list[dict[str, str]]:
    return [row for row in _ledger_rows(root) if row.get("status") == "success"]


def _last_success_by_source(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in _success_rows(root):
        source_id = row.get("source_id", "")
        retrieved_at = row.get("retrieved_at", "")
        if source_id and retrieved_at > result.get(source_id, ""):
            result[source_id] = retrieved_at
    return result


def write_historical_live_pair_artifacts(root: Path) -> list[Path]:
    rows = load_pair_rows(root)
    flat = _flat_rows(rows)
    last_success = _last_success_by_source(root)
    enriched: list[dict[str, str]] = []
    for row in flat:
        historical_id = row.get("historical_source_id", "")
        live_id = row.get("live_source_id", "")
        if not row.get("last_historical_success") and historical_id in last_success:
            row["last_historical_success"] = last_success[historical_id]
        if not row.get("last_live_success") and live_id in last_success:
            row["last_live_success"] = last_success[live_id]
        enriched.append(row)

    metadata_dir = root / "metadata"
    csv_path = metadata_dir / "historical_live_pairs.csv"
    parquet_path = metadata_dir / "historical_live_pairs.parquet"
    _write_csv(csv_path, enriched)
    _write_parquet(parquet_path, enriched)

    status_counts = Counter(row.get("coverage_status", "") for row in enriched)
    lines = [
        "# Historical / Live Pairing",
        "",
        "Polymarket is excluded. This report covers meteorological acquisition families only.",
        "",
        f"- generated_at_utc: `{_iso_now()}`",
        f"- pair families: `{len(enriched)}`",
        f"- coverage statuses: `{dict(sorted(status_counts.items()))}`",
        "",
        "| Family | Feature family | Historical source | Historical range | Live source | Live cadence | Eligibility | Coverage status | Blocker |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in enriched:
        historical_range = f"{row.get('historical_start', '')} to {row.get('historical_end', '')}"
        lines.append(
            "| "
            + " | ".join(
                [
                    row.get("family_id", ""),
                    row.get("feature_family", ""),
                    row.get("historical_source_id", ""),
                    historical_range,
                    row.get("live_source_id", ""),
                    row.get("live_native_cadence", ""),
                    row.get("operational_point_in_time_eligibility", ""),
                    row.get("coverage_status", ""),
                    row.get("blocker", ""),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Contract Fields",
            "",
            "Each row in `config/historical_live_pairs.yaml` carries all required fields from the reset goal, including timestamp semantics, latency, compatibility, license status, adapters, collector schedule, last success fields, status, and blocker.",
        ]
    )
    report_path = root / "reports" / "historical_live_pairing.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [csv_path, parquet_path, report_path]


def _count_sources(rows: Sequence[Mapping[str, str]], prefix_or_exact: str) -> int:
    return sum(1 for row in rows if str(row.get("source_id", "")).startswith(prefix_or_exact))


def write_required_acquisition_reports(root: Path) -> list[Path]:
    data_root = ensure_data_root(root)
    success_rows = _success_rows(root)
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in success_rows:
        by_source[row.get("source_id", "")].append(row)
    storage = inspect_data_root(root)
    policy = load_gridded_policy(root)
    families = policy.get("families", [])
    if not isinstance(families, list):
        families = []
    policy_by_id = {
        str(family.get("family_id", "")): family
        for family in families
        if isinstance(family, dict)
    }

    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    nwp_lines = [
        "# NWP Cycle Coverage",
        "",
        f"- data root: `{data_root}`",
        f"- generated_at_utc: `{_iso_now()}`",
        "",
        "| Product | Successful files | Actual bytes | Policy status | Historical period | Blocker |",
        "|---|---:|---:|---|---|---|",
    ]
    nwp_sources = (
        ("GFS", "ncep_gfs_hk_subset_grib2", "gfs_operational"),
        ("GEFS", "ncep_gefs_hk_subset_grib2", "gefs_operational_and_reforecast"),
        ("ECMWF IFS/ENS/AIFS", "ecmwf", "ecmwf_open_and_tigge"),
        ("DWD ICON/ICON-EPS", "dwd", "dwd_icon_icon_eps"),
    )
    for label, source_prefix, family_id in nwp_sources:
        rows = [
            row
            for source_id, source_rows in by_source.items()
            if source_id.startswith(source_prefix)
            for row in source_rows
        ]
        policy_row = policy_by_id.get(family_id, {})
        nwp_lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(len(rows)),
                    str(sum(int(row.get("content_length") or 0) for row in rows)),
                    str(policy_row.get("status", "not_in_policy")),
                    str(policy_row.get("historical_period", "")),
                    str(policy_row.get("blocker", "")),
                ]
            )
            + " |"
        )
    nwp_path = reports_dir / "nwp_cycle_coverage.md"
    nwp_path.write_text("\n".join(nwp_lines) + "\n", encoding="utf-8")
    written.append(nwp_path)

    satellite_lines = [
        "# Satellite Coverage",
        "",
        f"- data root: `{data_root}`",
        f"- generated_at_utc: `{_iso_now()}`",
        "",
        "| Source family | Successful files | Actual bytes | Status | Blocker |",
        "|---|---:|---:|---|---|",
    ]
    for label, source_prefix, family_id in (
        ("HKO current satellite windows", "hko_satellite", "hko_current_satellite"),
        ("Himawari 8/9 AHI", "himawari", "himawari_ahi"),
    ):
        rows = [
            row
            for source_id, source_rows in by_source.items()
            if source_id.startswith(source_prefix)
            for row in source_rows
        ]
        policy_row = policy_by_id.get(family_id, {})
        satellite_lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(len(rows)),
                    str(sum(int(row.get("content_length") or 0) for row in rows)),
                    str(policy_row.get("status", "current_window" if rows else "not_started")),
                    str(policy_row.get("blocker", "")),
                ]
            )
            + " |"
        )
    satellite_path = reports_dir / "satellite_coverage.md"
    satellite_path.write_text("\n".join(satellite_lines) + "\n", encoding="utf-8")
    written.append(satellite_path)

    precip_policy = policy_by_id.get("gpm_imerg", {})
    precip_lines = [
        "# Gridded Precipitation Coverage",
        "",
        f"- data root: `{data_root}`",
        f"- generated_at_utc: `{_iso_now()}`",
        "",
        "| Product | Successful files | Actual bytes | Policy status | Historical period | Blocker |",
        "|---|---:|---:|---|---|---|",
        "| GPM IMERG | "
        + " | ".join(
            [
                str(_count_sources(success_rows, "gpm_imerg")),
                str(
                    sum(
                        int(row.get("content_length") or 0)
                        for row in success_rows
                        if row.get("source_id", "").startswith("gpm_imerg")
                    )
                ),
                str(precip_policy.get("status", "not_started")),
                str(precip_policy.get("historical_period", "")),
                str(precip_policy.get("blocker", "")),
            ]
        )
        + " |",
    ]
    precip_path = reports_dir / "gridded_precipitation_coverage.md"
    precip_path.write_text("\n".join(precip_lines) + "\n", encoding="utf-8")
    written.append(precip_path)

    official_request_lines = [
        "# Official Request Gaps",
        "",
        "These gaps remain after public-source acquisition and bounded official-source discovery. They must not block other acquisition work.",
        "",
        "## HKO Dense Headquarters / Regional Sub-Daily History Request Package",
        "",
        "- requested station: HKO Headquarters / WMO 45005 plus all available regional automatic weather stations",
        "- requested period: 1984-01-01 through 2019-12-31, or earliest available dense archive through start of public DATA.GOV.HK history",
        "- requested cadence: one-, five-, or ten-minute observations, whichever is officially available",
        "- requested variables: temperature, max/min, relative humidity, pressure, wind direction/speed/gust, rainfall, visibility, present weather/RHR fields, solar radiation, UV, station metadata and relocations",
        "- required metadata: issue/publication timestamps, station IDs, units, missing-value codes, QC flags, revision policy, license and commercial-use terms",
        "- current public counterpart: HKO live feeds and DATA.GOV.HK historical ZIP archives where available from 2020/2021 onward",
        "- status: `official_request_required`",
        "",
        "## Forecast / ARWF JSON Vintages",
        "",
        "- requested products: FLW, FND, Warnings, SWT, ARWF station/grid forecasts, RHR/current-weather JSON vintages",
        "- requested period: earliest retained provider archive through present",
        "- status: `historically_unavailable_or_request_required`; RSS historical archives are already acquired where public",
    ]
    request_path = reports_dir / "official_request_gaps.md"
    request_path.write_text("\n".join(official_request_lines) + "\n", encoding="utf-8")
    written.append(request_path)

    storage_lines = [
        "# Storage and Volume",
        "",
        f"- data root: `{storage.path}`",
        f"- generated_at_utc: `{_iso_now()}`",
        f"- path length: `{storage.path_length}`",
        f"- exists: `{storage.exists}`",
        f"- long path risk: `{storage.long_path_risk}`",
        f"- free GB: `{storage.free_bytes / (1024 ** 3):.2f}`",
        f"- total GB: `{storage.total_bytes / (1024 ** 3):.2f}`",
        "",
        "Bulk raw data remains outside Git. The repository stores code, configs, reports, and manifests only.",
    ]
    storage_path = reports_dir / "storage_and_volume.md"
    storage_path.write_text("\n".join(storage_lines) + "\n", encoding="utf-8")
    written.append(storage_path)

    return written


def write_all_acquisition_contract_outputs(root: Path) -> list[Path]:
    written = []
    written.extend(write_historical_live_pair_artifacts(root))
    written.extend(write_required_acquisition_reports(root))
    return written
