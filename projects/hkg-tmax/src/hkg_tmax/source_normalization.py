from __future__ import annotations

import csv
import gzip
import html
import importlib
import io
import json
import math
import re
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree
from zoneinfo import ZoneInfo

from .acquisition import ensure_data_root
from .config import find_repo_root
from .hko import parse_daily_climate_csv, parse_daily_extract_json
from .paths import ProjectPaths, resolve_archive_content_path


class SourceNormalizationError(RuntimeError):
    """Raised when raw-source normalization cannot be completed."""


HKT = ZoneInfo("Asia/Hong_Kong")
KEY_IGRA_PRESSURE_HPA = (1000, 925, 850, 700, 500, 300, 200)

SHORT_MINUTE_ARCHIVE_PREFIXES = (
    "datagov_hko_historical_latest_1min_",
    "datagov_hko_historical_latest_since_midnight_maxmin_archive",
    "datagov_hko_historical_latest_10min_wind_archive",
    "datagov_hko_historical_latest_15min_uvindex_archive",
)

DAILY_CLIMATE_META: dict[str, dict[str, str]] = {
    "hko_daily_climate_mslp_all": {
        "variable": "mean_sea_level_pressure",
        "unit": "hPa",
        "station_or_domain": "Hong Kong Observatory",
    },
    "hko_daily_climate_mean_temperature_all": {
        "variable": "mean_temperature",
        "unit": "degC",
        "station_or_domain": "Hong Kong Observatory",
    },
    "hko_daily_climate_dew_point_all": {
        "variable": "mean_dew_point_temperature",
        "unit": "degC",
        "station_or_domain": "Hong Kong Observatory",
    },
    "hko_daily_climate_wet_bulb_all": {
        "variable": "mean_wet_bulb_temperature",
        "unit": "degC",
        "station_or_domain": "Hong Kong Observatory",
    },
    "hko_daily_climate_relative_humidity_all": {
        "variable": "mean_relative_humidity",
        "unit": "percent",
        "station_or_domain": "Hong Kong Observatory",
    },
    "hko_daily_climate_cloud_amount_all": {
        "variable": "mean_cloud_amount",
        "unit": "percent",
        "station_or_domain": "Hong Kong Observatory",
    },
    "hko_daily_climate_rainfall_all": {
        "variable": "daily_rainfall",
        "unit": "mm",
        "station_or_domain": "Hong Kong Observatory",
    },
    "hko_daily_climate_maximum_temperature_all": {
        "variable": "daily_maximum_temperature",
        "unit": "degC",
        "station_or_domain": "Hong Kong Observatory",
    },
    "hko_daily_climate_minimum_temperature_all": {
        "variable": "daily_minimum_temperature",
        "unit": "degC",
        "station_or_domain": "Hong Kong Observatory",
    },
    "hko_daily_climate_bright_sunshine_all": {
        "variable": "bright_sunshine_duration",
        "unit": "hours",
        "station_or_domain": "King's Park",
    },
    "hko_daily_climate_global_solar_radiation_all": {
        "variable": "global_solar_radiation",
        "unit": "MJ/m2",
        "station_or_domain": "King's Park",
    },
    "hko_daily_climate_evaporation_all": {
        "variable": "evaporation",
        "unit": "mm",
        "station_or_domain": "King's Park",
    },
    "hko_daily_climate_lightning_ground_all": {
        "variable": "cloud_to_ground_lightning",
        "unit": "count",
        "station_or_domain": "Hong Kong Territory",
    },
    "hko_daily_climate_lightning_cloud_all": {
        "variable": "cloud_to_cloud_lightning",
        "unit": "count",
        "station_or_domain": "Hong Kong Territory",
    },
    "hko_daily_climate_grass_min_temperature_all": {
        "variable": "grass_minimum_temperature",
        "unit": "degC",
        "station_or_domain": "Hong Kong Observatory",
    },
    "hko_daily_climate_prevailing_wind_direction_all": {
        "variable": "prevailing_wind_direction",
        "unit": "degree_or_compass",
        "station_or_domain": "Waglan Island",
    },
    "hko_daily_climate_mean_wind_speed_all": {
        "variable": "mean_wind_speed",
        "unit": "km/h",
        "station_or_domain": "Waglan Island",
    },
    "hko_daily_climate_sea_temp_waglan_all": {
        "variable": "sea_temperature",
        "unit": "degC",
        "station_or_domain": "Waglan Island",
    },
    "hko_daily_climate_sea_temp_np_am_all": {
        "variable": "sea_temperature_am",
        "unit": "degC",
        "station_or_domain": "North Point",
    },
    "hko_daily_climate_sea_temp_np_pm_all": {
        "variable": "sea_temperature_pm",
        "unit": "degC",
        "station_or_domain": "North Point",
    },
    "hko_daily_climate_reduced_visibility_hka_all": {
        "variable": "reduced_visibility_hours",
        "unit": "hours",
        "station_or_domain": "Hong Kong International Airport",
    },
}


@dataclass(frozen=True)
class NormalizedTable:
    table_id: str
    path: Path
    row_count: int
    start: str
    end: str
    status: str
    notes: str


@dataclass(frozen=True)
class NormalizationOutputs:
    tables: tuple[NormalizedTable, ...]
    manifest_path: Path
    report_path: Path


def _pd() -> Any:
    return importlib.import_module("pandas")


def _pa() -> Any:
    return importlib.import_module("pyarrow")


def _pq() -> Any:
    return importlib.import_module("pyarrow.parquet")


def _now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def read_ledger(data_root: Path) -> list[dict[str, str]]:
    path = data_root / "manifests" / "retrieval_ledger.csv"
    if not path.exists():
        raise SourceNormalizationError(f"Missing retrieval ledger: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def successful_rows(rows: Sequence[Mapping[str, str]], source_id: str) -> list[dict[str, str]]:
    return [
        dict(row)
        for row in rows
        if row.get("status") == "success" and row.get("source_id") == source_id
    ]


def successful_source_rows(
    rows: Sequence[Mapping[str, str]], source_ids: Iterable[str]
) -> list[dict[str, str]]:
    source_set = set(source_ids)
    return [
        dict(row)
        for row in rows
        if row.get("status") == "success" and row.get("source_id") in source_set
    ]


def latest_unique_successes(
    rows: Sequence[Mapping[str, str]], source_ids: Iterable[str]
) -> list[dict[str, str]]:
    latest: dict[tuple[str, str], dict[str, str]] = {}
    for row in successful_source_rows(rows, source_ids):
        key = (row.get("source_id", ""), row.get("content_sha256", ""))
        previous = latest.get(key)
        if previous is None or row.get("retrieved_at", "") > previous.get("retrieved_at", ""):
            latest[key] = row
    return sorted(latest.values(), key=lambda item: (item.get("source_id", ""), item.get("retrieved_at", "")))


def _write_parquet(path: Path, records: Sequence[Mapping[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = _pd().DataFrame([dict(row) for row in records])
    frame.to_parquet(path, index=False)
    return int(len(frame))


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _safe_float(value: object) -> float | None:
    token = str(value).strip()
    if not token or token.upper() in {"N/A", "NA", "NULL", "***", "M", "----", "-9999", "9999"}:
        return None
    try:
        parsed = float(token)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _safe_int(value: object) -> int | None:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _decimal_to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _iso_or_empty(value: object) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return "" if value is None else str(value)


def _range_from_records(records: Sequence[Mapping[str, object]], field: str) -> tuple[str, str]:
    values = sorted(str(row.get(field, "")) for row in records if row.get(field) not in {None, ""})
    if not values:
        return "", ""
    return values[0], values[-1]


def _source_year_from_url(row: Mapping[str, str]) -> int | None:
    text = row.get("request_url", "") + " " + row.get("final_url", "")
    match = re.search(r"dailyExtract_(\d{4})(?:\d{2})?\.xml", text)
    if match:
        return int(match.group(1))
    match = re.search(r"HKO((?:19|20)\d{2})BST\.csv", text)
    if match:
        return int(match.group(1))
    match = re.search(r"/((?:19|20)\d{2})/[^/]+\.gz", text)
    if match:
        return int(match.group(1))
    return None


def _archive_content(row: Mapping[str, str], output_dir: Path) -> Path:
    return resolve_archive_content_path(row, data_root=output_dir.parent)


def normalize_hko_daily_climate(
    ledger: Sequence[Mapping[str, str]], bronze_dir: Path, silver_dir: Path
) -> NormalizedTable:
    records: list[dict[str, object]] = []
    for row in latest_unique_successes(ledger, DAILY_CLIMATE_META):
        source_id = row["source_id"]
        meta = DAILY_CLIMATE_META[source_id]
        content_path = _archive_content(row, bronze_dir)
        for item in parse_daily_climate_csv(content_path.read_bytes()):
            variable = meta["variable"]
            target_only = variable == "daily_maximum_temperature"
            records.append(
                {
                    "source_id": source_id,
                    "content_sha256": row["content_sha256"],
                    "raw_retrieved_at_utc": row["retrieved_at"],
                    "station_or_domain": meta["station_or_domain"],
                    "variable": variable,
                    "unit": meta["unit"],
                    "local_date": _iso_or_empty(item.local_date),
                    "year": item.year,
                    "month": item.month,
                    "day": item.day,
                    "value": _decimal_to_float(item.value),
                    "value_precision": _decimal_to_float(item.value_precision),
                    "completeness": item.completeness,
                    "parse_issue": item.parse_issue,
                    "availability_tier": "TARGET_ONLY" if target_only else "MECHANISM_ONLY",
                    "operational_input_allowed": False,
                    "source_time_policy": "finalized HKO daily climate table; no first-publication timing is proven",
                }
            )
    if not records:
        raise SourceNormalizationError("No HKO daily climate rows parsed")
    bronze_path = bronze_dir / "hko_daily_climate_elements.parquet"
    _write_parquet(bronze_path, records)

    target_records = [
        {
            "local_date": row["local_date"],
            "target_tmax_c": row["value"],
            "target_station": "Hong Kong Observatory",
            "target_source_id": row["source_id"],
            "content_sha256": row["content_sha256"],
            "raw_retrieved_at_utc": row["raw_retrieved_at_utc"],
            "availability_tier": "TARGET_ONLY",
            "operational_input_allowed": False,
        }
        for row in records
        if row["variable"] == "daily_maximum_temperature" and row["local_date"] and row["value"] is not None
    ]
    _write_parquet(silver_dir / "hko_daily_tmax_target_labels.parquet", target_records)
    start, end = _range_from_records(records, "local_date")
    return NormalizedTable(
        "hko_daily_climate_elements",
        bronze_path,
        len(records),
        start,
        end,
        "parsed",
        "Long-history finalized daily HKO climate rows. Target-day daily elements are not operational predictors.",
    )


def normalize_hko_daily_extract(
    ledger: Sequence[Mapping[str, str]], bronze_dir: Path
) -> NormalizedTable:
    records: list[dict[str, object]] = []
    for row in latest_unique_successes(ledger, ("hko_daily_extract_year", "hko_daily_extract_month")):
        year = _source_year_from_url(row)
        if year is None:
            continue
        month_match = re.search(r"dailyExtract_(\d{6})\.xml", row.get("request_url", ""))
        month = int(month_match.group(1)[4:]) if month_match else None
        content_path = _archive_content(row, bronze_dir)
        try:
            parsed = parse_daily_extract_json(content_path.read_bytes(), year=year, month=month)
        except Exception as exc:
            records.append(
                {
                    "source_id": row["source_id"],
                    "content_sha256": row["content_sha256"],
                    "raw_retrieved_at_utc": row["retrieved_at"],
                    "year": year,
                    "month": month,
                    "parse_issue": f"parse_failed:{exc}",
                }
            )
            continue
        for item in parsed:
            records.append(
                {
                    "source_id": row["source_id"],
                    "content_sha256": row["content_sha256"],
                    "raw_retrieved_at_utc": row["retrieved_at"],
                    "local_date": item.local_date.isoformat(),
                    "year": item.year,
                    "month": item.month,
                    "day": item.day,
                    "absolute_daily_max_c": _decimal_to_float(item.absolute_daily_max_c),
                    "value_precision": _decimal_to_float(item.value_precision),
                    "completeness": item.completeness,
                    "parse_issue": item.parse_issue,
                    "availability_tier": "TARGET_ONLY",
                    "operational_input_allowed": False,
                    "source_time_policy": "Daily Extract payload is target/label side unless first-publication polling proves exact availability",
                }
            )
    if not records:
        raise SourceNormalizationError("No HKO Daily Extract rows parsed")
    path = bronze_dir / "hko_daily_extract_tmax_payload_rows.parquet"
    _write_parquet(path, records)
    start, end = _range_from_records(records, "local_date")
    return NormalizedTable(
        "hko_daily_extract_tmax_payload_rows",
        path,
        len(records),
        start,
        end,
        "parsed",
        "Annual/monthly Daily Extract payload rows normalized as target-side publication evidence.",
    )


def _parse_igra_header(line: str) -> dict[str, object]:
    parts = line[1:].split()
    if len(parts) < 10:
        raise SourceNormalizationError(f"Malformed IGRA header: {line[:80]}")
    year, month, day, hour = (int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
    valid_at = datetime(year, month, day, hour, tzinfo=UTC)
    reltime = _safe_int(parts[5])
    lat_raw = _safe_int(parts[-2])
    lon_raw = _safe_int(parts[-1])
    return {
        "station_id": parts[0],
        "valid_at_utc": valid_at,
        "valid_at_hkt": valid_at.astimezone(HKT),
        "reltime": reltime,
        "num_levels": int(parts[6]),
        "pressure_source": parts[7],
        "non_pressure_source": parts[8] if len(parts) > 10 else "",
        "latitude": None if lat_raw is None else lat_raw / 10000.0,
        "longitude": None if lon_raw is None else lon_raw / 10000.0,
    }


def _field_int(line: str, start: int, end: int) -> int | None:
    token = line[start:end].strip()
    if not token or token in {"-9999", "-99999", "99999"}:
        return None
    try:
        return int(token)
    except ValueError:
        return None


def parse_igra_level_line(line: str) -> dict[str, object]:
    pressure_raw = _field_int(line, 9, 15)
    temperature_raw = _field_int(line, 22, 27)
    dewpoint_depression_raw = _field_int(line, 34, 39)
    wind_speed_raw = _field_int(line, 46, 51)
    return {
        "level_type": _field_int(line, 0, 2),
        "elapsed_time_minutes": _field_int(line, 3, 8),
        "pressure_hpa": None if pressure_raw is None else pressure_raw / 100.0,
        "pressure_flag": line[15:16].strip(),
        "geopotential_height_m": _field_int(line, 16, 21),
        "geopotential_flag": line[21:22].strip(),
        "temperature_c": None if temperature_raw is None else temperature_raw / 10.0,
        "temperature_flag": line[27:28].strip(),
        "relative_humidity_pct": _field_int(line, 28, 33),
        "relative_humidity_flag": line[33:34].strip(),
        "dewpoint_depression_c": None
        if dewpoint_depression_raw is None
        else dewpoint_depression_raw / 10.0,
        "dewpoint_depression_flag": line[39:40].strip(),
        "wind_direction_deg": _field_int(line, 40, 45),
        "wind_speed_mps": None if wind_speed_raw is None else wind_speed_raw / 10.0,
    }


def _iter_igra_text_lines(path: Path) -> Iterable[str]:
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".txt")]
        if not names:
            raise SourceNormalizationError(f"IGRA ZIP has no txt file: {path}")
        with archive.open(names[0]) as handle:
            for raw_line in io.TextIOWrapper(handle, encoding="utf-8", errors="replace"):
                yield raw_line.rstrip("\n")


def normalize_igra_upper_air(
    ledger: Sequence[Mapping[str, str]], bronze_dir: Path, silver_dir: Path
) -> tuple[NormalizedTable, NormalizedTable]:
    source_rows = latest_unique_successes(
        ledger,
        ("noaa_igra_hkm00045004_period_of_record", "noaa_igra_hkm00045004_year_to_date"),
    )
    selected_by_sounding: dict[tuple[str, str], dict[str, object]] = {}
    level_records: list[dict[str, object]] = []
    for row in source_rows:
        path = _archive_content(row, bronze_dir)
        header: dict[str, object] | None = None
        sounding_levels: list[dict[str, object]] = []
        for line in _iter_igra_text_lines(path):
            if not line:
                continue
            if line.startswith("#"):
                if header is not None:
                    _add_igra_sounding(row, header, sounding_levels, selected_by_sounding, level_records)
                header = _parse_igra_header(line)
                sounding_levels = []
            elif header is not None:
                level = parse_igra_level_line(line)
                pressure = level.get("pressure_hpa")
                if isinstance(pressure, float) and int(round(pressure)) in KEY_IGRA_PRESSURE_HPA:
                    sounding_levels.append(level)
        if header is not None:
            _add_igra_sounding(row, header, sounding_levels, selected_by_sounding, level_records)

    sounding_records = sorted(
        selected_by_sounding.values(), key=lambda item: str(item.get("valid_at_utc", ""))
    )
    level_path = bronze_dir / "noaa_igra_hkm00045004_key_pressure_levels.parquet"
    sounding_path = silver_dir / "noaa_igra_hkm00045004_sounding_features.parquet"
    _write_parquet(level_path, level_records)
    _write_parquet(sounding_path, sounding_records)
    start, end = _range_from_records(sounding_records, "valid_at_utc")
    levels_table = NormalizedTable(
        "noaa_igra_hkm00045004_key_pressure_levels",
        level_path,
        len(level_records),
        start,
        end,
        "parsed_key_levels",
        "Key pressure levels only; full level dump omitted to keep derived tables tractable.",
    )
    soundings_table = NormalizedTable(
        "noaa_igra_hkm00045004_sounding_features",
        sounding_path,
        len(sounding_records),
        start,
        end,
        "parsed_proxy_features",
        "Upper-air profile features parsed, but operational release latency remains unproven and fails closed.",
    )
    return levels_table, soundings_table


def _add_igra_sounding(
    row: Mapping[str, str],
    header: Mapping[str, object],
    levels: Sequence[Mapping[str, object]],
    selected_by_sounding: dict[tuple[str, str], dict[str, object]],
    level_records: list[dict[str, object]],
) -> None:
    valid_at = header["valid_at_utc"]
    if not isinstance(valid_at, datetime):
        return
    source_id = row["source_id"]
    base = {
        "source_id": source_id,
        "content_sha256": row["content_sha256"],
        "raw_retrieved_at_utc": row["retrieved_at"],
        "station_id": header["station_id"],
        "valid_at_utc": valid_at.isoformat().replace("+00:00", "Z"),
        "valid_at_hkt": header["valid_at_hkt"].isoformat() if isinstance(header["valid_at_hkt"], datetime) else "",
        "nominal_hour_utc": valid_at.hour,
        "latitude": header.get("latitude"),
        "longitude": header.get("longitude"),
        "availability_tier": "PROXY_WITH_LIMITATIONS",
        "operational_input_allowed": False,
        "release_latency_proven": False,
        "source_time_policy": "NOAA IGRA archive parsed; release latency before T-1 cutoff not proven",
    }
    feature_row: dict[str, object] = dict(base)
    feature_row["key_level_count"] = len(levels)
    for level in levels:
        pressure = level.get("pressure_hpa")
        if not isinstance(pressure, float):
            continue
        level_tag = str(int(round(pressure)))
        level_records.append({**base, **level, "pressure_level_tag": level_tag})
        for source_name, out_name in [
            ("temperature_c", "temperature_c"),
            ("relative_humidity_pct", "relative_humidity_pct"),
            ("dewpoint_depression_c", "dewpoint_depression_c"),
            ("geopotential_height_m", "geopotential_height_m"),
            ("wind_direction_deg", "wind_direction_deg"),
            ("wind_speed_mps", "wind_speed_mps"),
        ]:
            feature_row[f"{out_name}_{level_tag}hpa"] = level.get(source_name)
    if (
        feature_row.get("temperature_c_850hpa") is not None
        and feature_row.get("temperature_c_500hpa") is not None
    ):
        temperature_850 = _safe_float(feature_row["temperature_c_850hpa"])
        temperature_500 = _safe_float(feature_row["temperature_c_500hpa"])
        if temperature_850 is not None and temperature_500 is not None:
            feature_row["temp_850_minus_500_c"] = temperature_850 - temperature_500
    if (
        feature_row.get("temperature_c_925hpa") is not None
        and feature_row.get("temperature_c_850hpa") is not None
    ):
        temperature_925 = _safe_float(feature_row["temperature_c_925hpa"])
        temperature_850 = _safe_float(feature_row["temperature_c_850hpa"])
        if temperature_925 is not None and temperature_850 is not None:
            feature_row["temp_925_minus_850_c"] = temperature_925 - temperature_850

    key = (str(feature_row["station_id"]), str(feature_row["valid_at_utc"]))
    previous = selected_by_sounding.get(key)
    if previous is None or source_id.endswith("year_to_date"):
        selected_by_sounding[key] = feature_row


def parse_isd_line(line: str) -> dict[str, object] | None:
    if len(line) < 105:
        return None
    try:
        observed_at = datetime.strptime(line[15:27], "%Y%m%d%H%M").replace(tzinfo=UTC)
    except ValueError:
        return None
    station_id = f"{line[4:10]}-{line[10:15]}"
    temp_raw = _first_signed_tenths(line, ((87, 92), (84, 89)))
    dew_raw = _first_signed_tenths(line, ((93, 98), (90, 95)))
    slp_raw = _first_unsigned_tenths(line, ((99, 104), (96, 101)))
    wind_speed_raw = _parse_missing_int(line[65:69], {9999})
    return {
        "station_id": station_id,
        "observed_at_utc": observed_at,
        "observed_at_hkt": observed_at.astimezone(HKT),
        "report_type": line[41:46].strip(),
        "latitude": _signed_scaled(line[28:34], 1000.0),
        "longitude": _signed_scaled(line[34:41], 1000.0),
        "elevation_m": _signed_scaled(line[46:51], 1.0),
        "wind_direction_deg": _parse_missing_int(line[57:60], {999}),
        "wind_speed_mps": None if wind_speed_raw is None else wind_speed_raw / 10.0,
        "air_temperature_c": temp_raw,
        "dew_point_c": dew_raw,
        "sea_level_pressure_hpa": slp_raw,
        "temperature_quality_code": line[92:93].strip() if len(line) > 92 else "",
        "dew_point_quality_code": line[98:99].strip() if len(line) > 98 else "",
        "sea_level_pressure_quality_code": line[104:105].strip() if len(line) > 104 else "",
    }


def _parse_missing_int(token: str, missing: set[int]) -> int | None:
    try:
        value = int(token.strip())
    except ValueError:
        return None
    return None if value in missing else value


def _signed_scaled(token: str, scale: float) -> float | None:
    stripped = token.strip()
    if not stripped or set(stripped) <= {"9", "+", "-"}:
        return None
    try:
        return int(stripped) / scale
    except ValueError:
        return None


def _first_signed_tenths(line: str, slices: Sequence[tuple[int, int]]) -> float | None:
    for start, end in slices:
        token = line[start:end]
        if token[:1] in {"+", "-"}:
            try:
                value = int(token)
            except ValueError:
                continue
            if abs(value) >= 9999:
                return None
            return value / 10.0
    return None


def _first_unsigned_tenths(line: str, slices: Sequence[tuple[int, int]]) -> float | None:
    for start, end in slices:
        token = line[start:end].strip()
        if token.isdigit():
            value = int(token)
            if value >= 99999:
                return None
            return value / 10.0
    return None


def normalize_noaa_isd(
    ledger: Sequence[Mapping[str, str]], bronze_dir: Path, silver_dir: Path
) -> tuple[NormalizedTable, NormalizedTable]:
    rows = successful_rows(ledger, "noaa_isd_nearby_station_year")
    if not rows:
        raise SourceNormalizationError("No NOAA ISD station-year files found")
    obs_path = bronze_dir / "noaa_isd_core_observations.parquet"
    obs_path.parent.mkdir(parents=True, exist_ok=True)
    writer: Any | None = None
    written = 0
    chunk: list[dict[str, object]] = []
    daily: dict[tuple[str, str], dict[str, object]] = {}
    try:
        for row in sorted(rows, key=lambda item: item.get("request_url", "")):
            content_path = _archive_content(row, bronze_dir)
            with gzip.open(content_path, "rt", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    parsed = parse_isd_line(line.rstrip("\n"))
                    if parsed is None:
                        continue
                    observed_hkt = parsed["observed_at_hkt"]
                    if not isinstance(observed_hkt, datetime):
                        continue
                    parsed.update(
                        {
                            "source_id": row["source_id"],
                            "content_sha256": row["content_sha256"],
                            "raw_retrieved_at_utc": row["retrieved_at"],
                            "availability_tier": "PROXY_WITH_LIMITATIONS",
                            "operational_input_allowed": False,
                            "source_time_policy": "NOAA ISD quality-controlled annual archive; not exact operational vintage",
                        }
                    )
                    _update_isd_daily_summary(daily, parsed)
                    chunk.append(_parquet_safe(parsed))
                    if len(chunk) >= 100_000:
                        writer = _write_parquet_chunk(obs_path, chunk, writer)
                        written += len(chunk)
                        chunk = []
        if chunk:
            writer = _write_parquet_chunk(obs_path, chunk, writer)
            written += len(chunk)
    finally:
        if writer is not None:
            writer.close()
    summary_records = sorted(daily.values(), key=lambda item: (str(item["station_id"]), str(item["local_date"])))
    summary_path = silver_dir / "noaa_isd_station_day_cutoff_summary.parquet"
    _write_parquet(summary_path, summary_records)
    start, end = _range_from_records(summary_records, "local_date")
    obs_table = NormalizedTable(
        "noaa_isd_core_observations",
        obs_path,
        written,
        start,
        end,
        "parsed_streaming",
        "Core hourly observations parsed from NOAA ISD station-year gzip files.",
    )
    summary_table = NormalizedTable(
        "noaa_isd_station_day_cutoff_summary",
        summary_path,
        len(summary_records),
        start,
        end,
        "parsed_proxy_features",
        "Daily station summaries include latest observation before 15:00 HKT, but archive is proxy-limited.",
    )
    return obs_table, summary_table


def _parquet_safe(row: Mapping[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in row.items():
        if isinstance(value, (datetime, date)):
            out[key] = value.isoformat()
        else:
            out[key] = value
    return out


def _write_parquet_chunk(path: Path, records: Sequence[Mapping[str, object]], writer: Any | None) -> Any:
    table = _pa().Table.from_pylist([dict(row) for row in records])
    if writer is None:
        writer = _pq().ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def _update_isd_daily_summary(daily: dict[tuple[str, str], dict[str, object]], row: Mapping[str, object]) -> None:
    observed_hkt = row["observed_at_hkt"]
    if not isinstance(observed_hkt, datetime):
        return
    local_date = observed_hkt.date().isoformat()
    key = (str(row["station_id"]), local_date)
    item = daily.setdefault(
        key,
        {
            "station_id": row["station_id"],
            "local_date": local_date,
            "obs_count": 0,
            "latest_before_1500_hkt": "",
            "air_temperature_c_latest_before_1500": None,
            "dew_point_c_latest_before_1500": None,
            "sea_level_pressure_hpa_latest_before_1500": None,
            "wind_direction_deg_latest_before_1500": None,
            "wind_speed_mps_latest_before_1500": None,
            "daily_air_temperature_min_c": None,
            "daily_air_temperature_max_c": None,
            "availability_tier": "PROXY_WITH_LIMITATIONS",
            "operational_input_allowed": False,
        },
    )
    item["obs_count"] = (_safe_int(item["obs_count"]) or 0) + 1
    temp = row.get("air_temperature_c")
    if isinstance(temp, (int, float)):
        previous_min = _safe_float(item["daily_air_temperature_min_c"])
        previous_max = _safe_float(item["daily_air_temperature_max_c"])
        item["daily_air_temperature_min_c"] = (
            temp if previous_min is None else min(previous_min, temp)
        )
        item["daily_air_temperature_max_c"] = (
            temp if previous_max is None else max(previous_max, temp)
        )
    cutoff = datetime.combine(observed_hkt.date(), datetime.min.time(), tzinfo=HKT) + timedelta(hours=15)
    previous = item["latest_before_1500_hkt"]
    if observed_hkt <= cutoff and (not previous or observed_hkt.isoformat() > str(previous)):
        item["latest_before_1500_hkt"] = observed_hkt.isoformat()
        item["air_temperature_c_latest_before_1500"] = row.get("air_temperature_c")
        item["dew_point_c_latest_before_1500"] = row.get("dew_point_c")
        item["sea_level_pressure_hpa_latest_before_1500"] = row.get("sea_level_pressure_hpa")
        item["wind_direction_deg_latest_before_1500"] = row.get("wind_direction_deg")
        item["wind_speed_mps_latest_before_1500"] = row.get("wind_speed_mps")


def _entry_timestamp_hkt(name: str) -> str:
    match = re.search(r"/(\d{8})-(\d{4})-[^/]+$", name)
    if not match:
        return ""
    dt = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M").replace(tzinfo=HKT)
    return dt.isoformat()


def _rss_language(source_id: str) -> str:
    for language in ("_en_", "_tc_", "_sc_"):
        if language in source_id:
            return language.strip("_")
    return ""


def _rss_feed_type(source_id: str) -> str:
    text = source_id.replace("datagov_hko_historical_rss_", "").replace("_archive", "")
    return re.sub(r"_(en|tc|sc)$", "", text)


def _clean_html_text(value: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", value, flags=re.IGNORECASE)
    text = re.sub(r"<p\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _element_text(parent: ElementTree.Element, name: str) -> str:
    child = parent.find(name)
    return "" if child is None or child.text is None else child.text.strip()


def _parse_pubdate(value: str) -> tuple[str, str]:
    if not value:
        return "", ""
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return value, ""
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    parsed_utc = parsed.astimezone(UTC)
    return parsed_utc.isoformat().replace("+00:00", "Z"), parsed_utc.astimezone(HKT).isoformat()


def _forecast_rows_from_9day_description(
    base: Mapping[str, object], description_text: str
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    normalized = _clean_html_text(description_text)
    pattern = re.compile(
        r"Date/Month:\s*(?P<day>\d{1,2})/(?P<month>\d{1,2}).*?"
        r"Temp range:\s*(?P<min>-?\d+(?:\.\d+)?)\s*-\s*(?P<max>-?\d+(?:\.\d+)?)\s*C",
        flags=re.IGNORECASE,
    )
    issue_hkt = str(base.get("published_at_hkt", ""))
    issue_year = int(issue_hkt[:4]) if re.match(r"\d{4}", issue_hkt) else None
    previous_month = 0
    for match in pattern.finditer(normalized):
        month = int(match.group("month"))
        day = int(match.group("day"))
        year = issue_year
        if year is None:
            continue
        if previous_month and month < previous_month:
            year += 1
        previous_month = month
        try:
            forecast_date = date(year, month, day).isoformat()
        except ValueError:
            continue
        rows.append(
            {
                **base,
                "forecast_date": forecast_date,
                "forecast_min_temperature_c": float(match.group("min")),
                "forecast_max_temperature_c": float(match.group("max")),
                "parser": "rss_9day_description_regex_v1",
            }
        )
    return rows


def _forecast_rows_from_local_description(
    base: Mapping[str, object], description_text: str
) -> list[dict[str, object]]:
    normalized = _clean_html_text(description_text)
    match = re.search(
        r"Weather forecast for Hong Kong \([^)]*(?P<day>\d{1,2})\s+"
        r"(?P<month>[A-Za-z]{3,9})\s+(?P<year>\d{4})\).*?"
        r"Temperatures will range between (?P<min>-?\d+(?:\.\d+)?) and (?P<max>-?\d+(?:\.\d+)?) degrees",
        normalized,
        flags=re.IGNORECASE,
    )
    if match is None:
        return []
    month_lookup = {
        "jan": 1,
        "january": 1,
        "feb": 2,
        "february": 2,
        "mar": 3,
        "march": 3,
        "apr": 4,
        "april": 4,
        "may": 5,
        "jun": 6,
        "june": 6,
        "jul": 7,
        "july": 7,
        "aug": 8,
        "august": 8,
        "sep": 9,
        "september": 9,
        "oct": 10,
        "october": 10,
        "nov": 11,
        "november": 11,
        "dec": 12,
        "december": 12,
    }
    month = month_lookup.get(match.group("month").lower())
    if month is None:
        return []
    try:
        forecast_date = date(int(match.group("year")), month, int(match.group("day"))).isoformat()
    except ValueError:
        return []
    return [
        {
            **base,
            "forecast_date": forecast_date,
            "forecast_min_temperature_c": float(match.group("min")),
            "forecast_max_temperature_c": float(match.group("max")),
            "parser": "rss_local_description_regex_v1",
        }
    ]


def normalize_hko_rss_archives(
    ledger: Sequence[Mapping[str, str]], bronze_dir: Path, silver_dir: Path
) -> tuple[NormalizedTable, NormalizedTable]:
    source_ids = sorted(
        {
            row.get("source_id", "")
            for row in ledger
            if row.get("source_id", "").startswith("datagov_hko_historical_rss_")
            and row.get("source_id", "").endswith("_archive")
        }
    )
    item_records: list[dict[str, object]] = []
    forecast_records: list[dict[str, object]] = []
    for row in latest_unique_successes(ledger, source_ids):
        content_path = _archive_content(row, bronze_dir)
        source_id = row["source_id"]
        feed_type = _rss_feed_type(source_id)
        language = _rss_language(source_id)
        with zipfile.ZipFile(content_path) as archive:
            for name in archive.namelist():
                if not name.lower().endswith(".xml"):
                    continue
                try:
                    root = ElementTree.fromstring(archive.read(name).decode("utf-8-sig", errors="replace"))
                except ElementTree.ParseError as exc:
                    item_records.append(
                        {
                            "source_id": source_id,
                            "content_sha256": row["content_sha256"],
                            "archive_entry_name": name,
                            "available_at_hkt": _entry_timestamp_hkt(name),
                            "feed_type": feed_type,
                            "language": language,
                            "parse_issue": f"xml_parse_failed:{exc}",
                        }
                    )
                    continue
                for item in root.findall("./channel/item"):
                    pub_utc, pub_hkt = _parse_pubdate(_element_text(item, "pubDate"))
                    description_raw = _element_text(item, "description")
                    base = {
                        "source_id": source_id,
                        "content_sha256": row["content_sha256"],
                        "raw_retrieved_at_utc": row["retrieved_at"],
                        "archive_entry_name": name,
                        "feed_type": feed_type,
                        "language": language,
                        "available_at_hkt": _entry_timestamp_hkt(name),
                        "published_at_utc": pub_utc,
                        "published_at_hkt": pub_hkt,
                        "title": _element_text(item, "title"),
                        "guid": _element_text(item, "guid"),
                        "category": _element_text(item, "category"),
                        "link": _element_text(item, "link"),
                        "description_text": _clean_html_text(description_raw),
                        "availability_tier": "GOLD_EXACT_VINTAGE",
                        "operational_input_allowed": True,
                        "source_time_policy": "DATA.GOV.HK historical archive entry timestamp is treated as exact archive-vintage availability",
                    }
                    item_records.append(base)
                    if feed_type == "9day_forecast" and language == "en":
                        forecast_records.extend(_forecast_rows_from_9day_description(base, description_raw))
                    elif feed_type == "local_forecast" and language == "en":
                        forecast_records.extend(_forecast_rows_from_local_description(base, description_raw))
    item_path = bronze_dir / "hko_historical_rss_items.parquet"
    forecast_path = silver_dir / "hko_historical_rss_temperature_forecasts.parquet"
    _write_parquet(item_path, item_records)
    _write_parquet(forecast_path, forecast_records)
    item_start, item_end = _range_from_records(item_records, "available_at_hkt")
    forecast_start, forecast_end = _range_from_records(forecast_records, "forecast_date")
    return (
        NormalizedTable(
            "hko_historical_rss_items",
            item_path,
            len(item_records),
            item_start,
            item_end,
            "parsed_vintages",
            "All downloaded HKO historical RSS XML archive entries normalized as forecast/warning vintages.",
        ),
        NormalizedTable(
            "hko_historical_rss_temperature_forecasts",
            forecast_path,
            len(forecast_records),
            forecast_start,
            forecast_end,
            "parsed_forecast_temperatures",
            "English local and 9-day RSS forecast temperature ranges extracted for MOS-style later experiments.",
        ),
    )


def normalize_tc_best_track(ledger: Sequence[Mapping[str, str]], bronze_dir: Path) -> NormalizedTable:
    records: list[dict[str, object]] = []
    for row in latest_unique_successes(ledger, ("hko_tropical_cyclone_best_track",)):
        path = _archive_content(row, bronze_dir)
        lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        header_index = next((idx for idx, line in enumerate(lines) if line.startswith("Tropical Cyclone Name")), None)
        if header_index is None:
            continue
        reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
        for raw in reader:
            year = _safe_int(raw.get("Year/年/年"))
            month = _safe_int(raw.get("Month/月/月"))
            day = _safe_int(raw.get("Day/日/日"))
            hour = _safe_int(raw.get("Time (UTC)/時間(協調世界時)/时间(协调世界时)"))
            if year is None or month is None or day is None or hour is None:
                continue
            valid_at = datetime(year, month, day, hour, tzinfo=UTC)
            lat = _safe_float(raw.get("Latitude (0.01 degree N)/北緯(0.01度)/北纬(0.01度)"))
            lon = _safe_float(raw.get("Longitude (0.01 degree E)/東經(0.01度)/东经(0.01度)"))
            records.append(
                {
                    "source_id": row["source_id"],
                    "content_sha256": row["content_sha256"],
                    "raw_retrieved_at_utc": row["retrieved_at"],
                    "cyclone_name": raw.get("Tropical Cyclone Name/熱帶氣旋名稱/热带气旋名称", ""),
                    "valid_at_utc": valid_at.isoformat().replace("+00:00", "Z"),
                    "valid_at_hkt": valid_at.astimezone(HKT).isoformat(),
                    "intensity": raw.get("Intensity/強度/强度", ""),
                    "latitude": None if lat is None else lat / 100.0,
                    "longitude": None if lon is None else lon / 100.0,
                    "minimum_central_pressure_hpa": _safe_float(
                        raw.get("Estimated minimum central pressure (hPa)/估計最低中心氣壓(百帕斯卡)/估计最低中心气压(百帕斯卡)")
                    ),
                    "maximum_surface_wind_kt": _safe_float(
                        raw.get("Estimated maximum surface winds (knot)/估計最高風速(海里)/估计最高风速(海里)")
                    ),
                    "jma_code": raw.get("JMA Code/JMA編號/JMA编号", ""),
                    "hko_code": raw.get("HKO Code/HKO編號/HKO编号", ""),
                    "availability_tier": "MECHANISM_ONLY",
                    "operational_input_allowed": False,
                    "source_time_policy": "HKO best track is retrospective post-analysis and forbidden as operational predictor",
                }
            )
    path = bronze_dir / "hko_tropical_cyclone_best_track.parquet"
    _write_parquet(path, records)
    start, end = _range_from_records(records, "valid_at_utc")
    return NormalizedTable(
        "hko_tropical_cyclone_best_track",
        path,
        len(records),
        start,
        end,
        "parsed_retrospective",
        "Best-track rows are parsed for mechanism/regime labels only, not operational inputs.",
    )


def normalize_json_and_csv_live_feeds(
    ledger: Sequence[Mapping[str, str]], bronze_dir: Path, silver_dir: Path
) -> tuple[NormalizedTable, ...]:
    tables: list[NormalizedTable] = []
    tables.append(_normalize_arwf_station_forecasts(ledger, silver_dir))
    tables.append(_normalize_gridded_nowcast_summary(ledger, silver_dir))
    tables.append(_normalize_lightning_counts(ledger, bronze_dir))
    tables.append(_normalize_tide_rows(ledger, bronze_dir))
    tables.append(_normalize_marine_bulletins(ledger, bronze_dir))
    tables.append(_normalize_radar_manifest(ledger, bronze_dir))
    tables.append(_normalize_satellite_image_inventory(ledger, bronze_dir))
    tables.append(_normalize_nwp_inventory(ledger, bronze_dir))
    return tuple(tables)


def _normalize_arwf_station_forecasts(
    ledger: Sequence[Mapping[str, str]], silver_dir: Path
) -> NormalizedTable:
    records: list[dict[str, object]] = []
    for row in latest_unique_successes(ledger, ("hko_arwf_station_forecast",)):
        payload = json.loads(
            _archive_content(row, silver_dir).read_text(encoding="utf-8-sig", errors="replace")
        )
        last_modified = str(payload.get("LastModified", ""))
        station_code = str(payload.get("StationCode", ""))
        for item in payload.get("DailyForecast", []):
            if not isinstance(item, dict):
                continue
            records.append(
                {
                    "source_id": row["source_id"],
                    "content_sha256": row["content_sha256"],
                    "raw_retrieved_at_utc": row["retrieved_at"],
                    "station_code": station_code,
                    "latitude": payload.get("Latitude"),
                    "longitude": payload.get("Longitude"),
                    "model_time": payload.get("ModelTime"),
                    "last_modified": last_modified,
                    "forecast_date": str(item.get("ForecastDate", "")),
                    "forecast_max_temperature_c": item.get("ForecastMaximumTemperature"),
                    "forecast_min_temperature_c": item.get("ForecastMinimumTemperature"),
                    "availability_tier": "GOLD_EXACT_VINTAGE",
                    "operational_input_allowed": True,
                    "source_time_policy": "Live ARWF payload available no earlier than immutable retrieval time",
                }
            )
    path = silver_dir / "hko_arwf_station_daily_forecasts.parquet"
    _write_parquet(path, records)
    start, end = _range_from_records(records, "forecast_date")
    return NormalizedTable("hko_arwf_station_daily_forecasts", path, len(records), start, end, "parsed_live_vintages", "ARWF station forecast JSON snapshots normalized.")


def _normalize_gridded_nowcast_summary(
    ledger: Sequence[Mapping[str, str]], silver_dir: Path
) -> NormalizedTable:
    records: list[dict[str, object]] = []
    for row in latest_unique_successes(ledger, ("hko_gridded_rainfall_nowcast",)):
        path = _archive_content(row, silver_dir)
        values: list[float] = []
        update_time = ""
        ending_time = ""
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                update_time = raw.get("Updated Date and Time (in Hong Kong Time)", update_time)
                ending_time = raw.get("Ending Date and Time (in Hong Kong Time)", ending_time)
                value = _safe_float(raw.get("Half-hourly Nowcast Accumulated Rainfall (mm)"))
                if value is not None:
                    values.append(value)
        if values:
            sorted_values = sorted(values)
            records.append(
                {
                    "source_id": row["source_id"],
                    "content_sha256": row["content_sha256"],
                    "raw_retrieved_at_utc": row["retrieved_at"],
                    "issue_time_hkt": _yyyymmddhhmm_hkt(update_time),
                    "ending_time_hkt": _yyyymmddhhmm_hkt(ending_time),
                    "grid_cell_count": len(values),
                    "rainfall_mean_mm": sum(values) / len(values),
                    "rainfall_max_mm": max(values),
                    "rainfall_p95_mm": sorted_values[int(0.95 * (len(sorted_values) - 1))],
                    "rain_area_fraction_gt_0mm": sum(1 for value in values if value > 0) / len(values),
                    "rain_area_fraction_ge_1mm": sum(1 for value in values if value >= 1) / len(values),
                    "availability_tier": "GOLD_EXACT_VINTAGE",
                    "operational_input_allowed": True,
                }
            )
    path = silver_dir / "hko_gridded_rainfall_nowcast_summary.parquet"
    _write_parquet(path, records)
    start, end = _range_from_records(records, "issue_time_hkt")
    return NormalizedTable("hko_gridded_rainfall_nowcast_summary", path, len(records), start, end, "parsed_summary", "Nowcast grid summarized by vintage; individual grid cells are not expanded.")


def _yyyymmddhhmm_hkt(value: str) -> str:
    token = str(value).strip()
    if not re.fullmatch(r"\d{12}", token):
        return token
    return datetime.strptime(token, "%Y%m%d%H%M").replace(tzinfo=HKT).isoformat()


def _normalize_lightning_counts(ledger: Sequence[Mapping[str, str]], bronze_dir: Path) -> NormalizedTable:
    records: list[dict[str, object]] = []
    for row in latest_unique_successes(ledger, ("hko_lightning_counts_latest",)):
        with _archive_content(row, bronze_dir).open(
            encoding="utf-8-sig", errors="replace", newline=""
        ) as handle:
            for raw in csv.DictReader(handle):
                records.append(
                    {
                        "source_id": row["source_id"],
                        "content_sha256": row["content_sha256"],
                        "raw_retrieved_at_utc": row["retrieved_at"],
                        "period": raw.get("DateTime", ""),
                        "lightning_type": raw.get("Type", ""),
                        "region": raw.get("Region", ""),
                        "lightning_count": _safe_int(raw.get("lightning count")),
                        "availability_tier": "GOLD_EXACT_VINTAGE",
                        "operational_input_allowed": True,
                    }
                )
    path = bronze_dir / "hko_lightning_counts_latest.parquet"
    _write_parquet(path, records)
    start, end = _range_from_records(records, "period")
    return NormalizedTable("hko_lightning_counts_latest", path, len(records), start, end, "parsed_live_vintages", "Live lightning count snapshots normalized.")


def _normalize_tide_rows(ledger: Sequence[Mapping[str, str]], bronze_dir: Path) -> NormalizedTable:
    records: list[dict[str, object]] = []
    for row in latest_unique_successes(ledger, ("hko_latest_tidal_information",)):
        with _archive_content(row, bronze_dir).open(
            encoding="utf-8-sig", errors="replace", newline=""
        ) as handle:
            for raw in csv.DictReader(handle):
                dt_text = f"{raw.get('Date', '')} {raw.get('Time', '')}".strip()
                observed_at = ""
                if dt_text:
                    try:
                        observed_at = datetime.strptime(dt_text, "%Y-%m-%d %H:%M").replace(tzinfo=HKT).isoformat()
                    except ValueError:
                        observed_at = dt_text
                records.append(
                    {
                        "source_id": row["source_id"],
                        "content_sha256": row["content_sha256"],
                        "raw_retrieved_at_utc": row["retrieved_at"],
                        "tide_station": raw.get("Tide Station", ""),
                        "observed_at_hkt": observed_at,
                        "height_m": _safe_float(raw.get("Height(m)")),
                        "availability_tier": "GOLD_EXACT_VINTAGE",
                        "operational_input_allowed": True,
                    }
                )
    path = bronze_dir / "hko_latest_tidal_information.parquet"
    _write_parquet(path, records)
    start, end = _range_from_records(records, "observed_at_hkt")
    return NormalizedTable("hko_latest_tidal_information", path, len(records), start, end, "parsed_live_vintages", "Latest tidal rows normalized.")


def _normalize_marine_bulletins(ledger: Sequence[Mapping[str, str]], bronze_dir: Path) -> NormalizedTable:
    records: list[dict[str, object]] = []
    for row in latest_unique_successes(ledger, ("hko_south_china_coastal_waters_bulletin",)):
        payload = json.loads(
            _archive_content(row, bronze_dir).read_text(encoding="utf-8-sig", errors="replace")
        )
        update_time = str(payload.get("updateTime", ""))
        forecast = payload.get("weatherForecast", {})
        data = forecast.get("data", []) if isinstance(forecast, dict) else []
        for item in data:
            if not isinstance(item, dict):
                continue
            records.append(
                {
                    "source_id": row["source_id"],
                    "content_sha256": row["content_sha256"],
                    "raw_retrieved_at_utc": row["retrieved_at"],
                    "update_time_hkt": update_time,
                    "location_name": item.get("locationName", ""),
                    "wind_info": item.get("windInfo", ""),
                    "weather_description": item.get("weatherDescription", ""),
                    "sea_situation": item.get("seaSituation", ""),
                    "availability_tier": "GOLD_EXACT_VINTAGE",
                    "operational_input_allowed": True,
                }
            )
    path = bronze_dir / "hko_south_china_coastal_waters_bulletin.parquet"
    _write_parquet(path, records)
    start, end = _range_from_records(records, "update_time_hkt")
    return NormalizedTable("hko_south_china_coastal_waters_bulletin", path, len(records), start, end, "parsed_live_vintages", "Marine bulletin area forecasts normalized.")


def _normalize_radar_manifest(ledger: Sequence[Mapping[str, str]], bronze_dir: Path) -> NormalizedTable:
    records: list[dict[str, object]] = []
    frame_re = re.compile(r'"(?P<path>[^"]*?(?P<stamp>\d{12})[^"]*?\.jpg)"')
    for row in latest_unique_successes(ledger, ("hko_radar_image_manifest",)):
        payload = json.loads(
            _archive_content(row, bronze_dir).read_text(encoding="utf-8-sig", errors="replace")
        )
        radar = payload.get("radar", {})
        if not isinstance(radar, dict):
            continue
        for range_key, range_data in radar.items():
            if not isinstance(range_data, dict):
                continue
            for script in range_data.get("image", []):
                if not isinstance(script, str):
                    continue
                match = frame_re.search(script)
                if not match:
                    continue
                records.append(
                    {
                        "source_id": row["source_id"],
                        "content_sha256": row["content_sha256"],
                        "raw_retrieved_at_utc": row["retrieved_at"],
                        "range_key": range_key,
                        "frame_relative_path": match.group("path"),
                        "frame_time_hkt": _yyyymmddhhmm_hkt(match.group("stamp")),
                        "availability_tier": "GOLD_EXACT_VINTAGE",
                        "operational_input_allowed": True,
                    }
                )
    path = bronze_dir / "hko_radar_manifest_frames.parquet"
    _write_parquet(path, records)
    start, end = _range_from_records(records, "frame_time_hkt")
    return NormalizedTable("hko_radar_manifest_frames", path, len(records), start, end, "parsed_metadata", "Radar manifest frame times normalized; image pixels are not decoded.")


def _normalize_satellite_image_inventory(ledger: Sequence[Mapping[str, str]], bronze_dir: Path) -> NormalizedTable:
    records: list[dict[str, object]] = []
    for row in ledger:
        source_id = row.get("source_id", "")
        if row.get("status") != "success" or not source_id.startswith("hko_satellite_"):
            continue
        filename = Path(row.get("request_url", "")).name or Path(row.get("content_path", "")).name
        stamp_match = re.search(r"((?:20)\d{12})", filename)
        records.append(
            {
                "source_id": source_id,
                "content_sha256": row.get("content_sha256", ""),
                "raw_retrieved_at_utc": row.get("retrieved_at", ""),
                "filename": filename,
                "image_time_hkt": _yyyymmddhhmm_hkt(stamp_match.group(1)[:12]) if stamp_match else "",
                "content_length": _safe_int(row.get("content_length")),
                "content_path": row.get("content_path", ""),
                "availability_tier": "GOLD_EXACT_VINTAGE",
                "operational_input_allowed": source_id.endswith("_image"),
                "source_time_policy": "Satellite image/manifest metadata only; pixel/georeference features still require image parser",
            }
        )
    path = bronze_dir / "hko_satellite_image_inventory.parquet"
    _write_parquet(path, records)
    start, end = _range_from_records(records, "image_time_hkt")
    return NormalizedTable("hko_satellite_image_inventory", path, len(records), start, end, "metadata_only", "Satellite raw imagery normalized to metadata inventory; pixels not decoded in this pass.")


def _normalize_nwp_inventory(ledger: Sequence[Mapping[str, str]], bronze_dir: Path) -> NormalizedTable:
    records: list[dict[str, object]] = []
    for row in ledger:
        source_id = row.get("source_id", "")
        if row.get("status") != "success" or source_id not in {"ncep_gfs_hk_subset_grib2", "ncep_gefs_hk_subset_grib2"}:
            continue
        url = row.get("request_url", "")
        cycle_match = re.search(r"/(?:gfs|gefs)\.(?P<date>\d{8})/(?P<cycle>\d{2})/", url)
        fh_match = re.search(r"\.f(?P<fh>\d{3})", url)
        member_match = re.search(r"file=(?P<member>ge\w+|gfs)\.", url)
        records.append(
            {
                "source_id": source_id,
                "content_sha256": row.get("content_sha256", ""),
                "raw_retrieved_at_utc": row.get("retrieved_at", ""),
                "cycle_date": cycle_match.group("date") if cycle_match else "",
                "cycle_hour_utc": cycle_match.group("cycle") if cycle_match else "",
                "forecast_hour": int(fh_match.group("fh")) if fh_match else None,
                "member": member_match.group("member") if member_match else "",
                "content_length": _safe_int(row.get("content_length")),
                "content_path": row.get("content_path", ""),
                "availability_tier": "GOLD_EXACT_VINTAGE",
                "operational_input_allowed": True,
                "source_time_policy": "GRIB2 cycle inventory only; meteorological fields need GRIB decoding dependency/policy",
            }
        )
    path = bronze_dir / "ncep_operational_grib2_inventory.parquet"
    _write_parquet(path, records)
    start, end = _range_from_records(records, "cycle_date")
    return NormalizedTable("ncep_operational_grib2_inventory", path, len(records), start, end, "metadata_only", "NCEP GFS/GEFS GRIB2 subset inventory normalized; fields not decoded in this pass.")


def normalize_static_metadata_inventory(
    ledger: Sequence[Mapping[str, str]], bronze_dir: Path
) -> NormalizedTable:
    prefixes = ("landsd_", "csdi_", "pland_", "data_gov_hk_landsd_", "data_gov_hk_pland_")
    records: list[dict[str, object]] = []
    for row in ledger:
        source_id = row.get("source_id", "")
        if row.get("status") != "success" or not source_id.startswith(prefixes):
            continue
        path = _archive_content(row, bronze_dir)
        zip_members: list[str] = []
        if path.exists() and path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(path) as archive:
                    zip_members = archive.namelist()[:200]
            except zipfile.BadZipFile:
                zip_members = []
        records.append(
            {
                "source_id": source_id,
                "content_sha256": row.get("content_sha256", ""),
                "raw_retrieved_at_utc": row.get("retrieved_at", ""),
                "content_length": _safe_int(row.get("content_length")),
                "content_path": row.get("content_path", ""),
                "extension": path.suffix.lower().lstrip("."),
                "zip_member_count_sampled": len(zip_members),
                "zip_members_sample_json": json.dumps(zip_members, ensure_ascii=False),
                "availability_tier": "STATIC_METADATA",
                "operational_input_allowed": True,
                "source_time_policy": "Static geospatial package inventory; station raster/vector extraction remains source-specific",
            }
        )
    path = bronze_dir / "static_geospatial_package_inventory.parquet"
    _write_parquet(path, records)
    return NormalizedTable(
        "static_geospatial_package_inventory",
        path,
        len(records),
        "",
        "",
        "metadata_only",
        "Static geospatial raw packages inventoried; terrain/coast/LUHK station buffers still require geospatial decoding.",
    )


def skipped_short_minute_sources(ledger: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    counts: dict[str, int] = defaultdict(int)
    for row in ledger:
        source_id = row.get("source_id", "")
        if row.get("status") == "success" and source_id.startswith(SHORT_MINUTE_ARCHIVE_PREFIXES):
            counts[source_id] += 1
    return [
        {
            "source_id": source_id,
            "success_rows": count,
            "status": "skipped_by_user_instruction",
            "reason": "short 2020/2021-start minute/snapshot historical archive excluded from this normalization pass",
        }
        for source_id, count in sorted(counts.items())
    ]


def write_normalization_report(
    repo_root: Path,
    data_root: Path,
    tables: Sequence[NormalizedTable],
    skipped: Sequence[Mapping[str, object]],
) -> Path:
    path = (
        ProjectPaths.from_project_root(repo_root).run_root
        / "reports"
        / "hkg_t24"
        / "SOURCE_NORMALIZATION_NON_MINUTE.md"
    )
    lines = [
        "# HKG T24 Non-Minute Source Normalization",
        "",
        f"Generated: `{_now_utc()}`",
        "",
        "This pass parsed and normalized raw source families without running predictive experiments, validation scoring, Polymarket logic, backtesting, or ML.",
        "",
        f"- data root: `{data_root}`",
        "- user exclusion honored: short HKO minute/snapshot historical archives starting around 2020/2021 were skipped.",
        "- raw archive objects were not modified.",
        "- operational use still requires each table's `operational_input_allowed` and availability fields to pass the as-of contract.",
        "",
        "## Normalized Tables",
        "",
        "| Table | Rows | Range | Status | Path | Notes |",
        "|---|---:|---|---|---|---|",
    ]
    for table in tables:
        date_range = f"{table.start} to {table.end}".strip()
        lines.append(
            f"| {table.table_id} | {table.row_count:,} | {date_range} | {table.status} | `{table.path}` | {table.notes} |"
        )
    lines.extend(["", "## Explicitly Skipped Short Minute/Snapshot Archives", ""])
    if skipped:
        lines.extend(["| Source | Success rows | Reason |", "|---|---:|---|"])
        for row in skipped:
            lines.append(
                f"| {row.get('source_id', '')} | {row.get('success_rows', 0)} | {row.get('reason', '')} |"
            )
    else:
        lines.append("No skipped short-minute archive rows were found.")
    lines.extend(
        [
            "",
            "## Important Fail-Closed Notes",
            "",
            "- IGRA upper-air profiles are parsed, but release latency is not proven, so those rows are proxy-limited until the R14 availability contract is completed.",
            "- NOAA ISD station observations are parsed from the quality-controlled archive, so they are useful for long-history proxy/mechanism work but not exact real-time vintages by default.",
            "- HKO RSS forecast archives are exact DATA.GOV.HK historical archive entries and are the strongest current official forecast-vintage source.",
            "- Satellite, radar-image, static geospatial, and GRIB2 products are normalized to metadata inventories here; pixel/raster/GRIB field decoding is a separate parser step.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def normalize_non_minute_sources(root: Path | None = None, data_root: Path | None = None) -> NormalizationOutputs:
    repo_root = root or find_repo_root()
    resolved_data_root = data_root or ensure_data_root(repo_root)
    return normalize_non_minute_sources_to_output(
        root=repo_root,
        input_data_root=resolved_data_root,
        output_data_root=resolved_data_root,
    )


def normalize_non_minute_sources_to_output(
    *,
    root: Path | None = None,
    input_data_root: Path,
    output_data_root: Path,
) -> NormalizationOutputs:
    repo_root = root or find_repo_root()
    ledger = read_ledger(input_data_root)
    bronze_dir = output_data_root / "bronze" / "source_normalized_non_minute"
    silver_dir = output_data_root / "silver" / "source_normalized_non_minute"
    metadata_dir = output_data_root / "metadata" / "source_normalization"
    tables: list[NormalizedTable] = []

    tables.append(normalize_hko_daily_climate(ledger, bronze_dir, silver_dir))
    tables.append(normalize_hko_daily_extract(ledger, bronze_dir))
    tables.extend(normalize_igra_upper_air(ledger, bronze_dir, silver_dir))
    tables.extend(normalize_noaa_isd(ledger, bronze_dir, silver_dir))
    tables.extend(normalize_hko_rss_archives(ledger, bronze_dir, silver_dir))
    tables.append(normalize_tc_best_track(ledger, bronze_dir))
    tables.extend(normalize_json_and_csv_live_feeds(ledger, bronze_dir, silver_dir))
    tables.append(normalize_static_metadata_inventory(ledger, bronze_dir))

    skipped = skipped_short_minute_sources(ledger)
    manifest_path = metadata_dir / "non_minute_normalization_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "generated_at_utc": _now_utc(),
            "input_data_root": str(input_data_root),
            "output_data_root": str(output_data_root),
            "user_instruction": "parse raw unusable data, excluding short 2020/2021 HKO minute/snapshot archives; do not run experiments",
            "tables": [table.__dict__ | {"path": str(table.path)} for table in tables],
            "skipped_sources": list(skipped),
        },
    )
    report_path = write_normalization_report(repo_root, output_data_root, tables, skipped)
    return NormalizationOutputs(tuple(tables), manifest_path, report_path)
