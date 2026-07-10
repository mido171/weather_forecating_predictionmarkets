from __future__ import annotations

import csv
import importlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .hko import parse_daily_climate_csv
from .paths import resolve_archive_content_path


class BronzeBuildError(RuntimeError):
    """Raised when a bronze dataset cannot be rebuilt from raw content."""


HKT = ZoneInfo("Asia/Hong_Kong")


@dataclass(frozen=True)
class BronzeDataset:
    source_id: str
    content_sha256: str
    row_count: int
    parquet_path: Path
    metadata_path: Path


def _read_ledger(data_root: Path) -> list[dict[str, str]]:
    path = data_root / "manifests" / "retrieval_ledger.csv"
    if not path.exists():
        raise BronzeBuildError(f"Missing retrieval ledger: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _latest_success(data_root: Path, source_id: str) -> dict[str, str]:
    rows = [
        row
        for row in _read_ledger(data_root)
        if row.get("source_id") == source_id and row.get("status") == "success"
    ]
    if not rows:
        raise BronzeBuildError(f"No successful retrieval for source: {source_id}")
    return max(rows, key=lambda row: row.get("retrieved_at", ""))


def _write_parquet(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pa: Any = importlib.import_module("pyarrow")
    pq: Any = importlib.import_module("pyarrow.parquet")
    table = pa.Table.from_pylist([dict(row) for row in rows])
    pq.write_table(table, path, compression="zstd")


def _decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _parse_hko_datetime(token: str) -> tuple[str, str]:
    local = datetime.strptime(token, "%Y%m%d%H%M").replace(tzinfo=HKT)
    return (
        local.isoformat(),
        local.astimezone(UTC).isoformat().replace("+00:00", "Z"),
    )


def _bronze_clmmaxt(content: bytes, *, source_id: str, sha256: str, retrieved_at: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in parse_daily_climate_csv(content):
        rows.append(
            {
                "source_id": source_id,
                "content_sha256": sha256,
                "retrieved_at": retrieved_at,
                "station_id": "HKO",
                "parameter": "daily_maximum_air_temperature",
                "unit": "degree_celsius",
                "local_date": row.local_date.isoformat() if row.local_date else None,
                "value": _decimal_text(row.value),
                "value_precision": _decimal_text(row.value_precision),
                "completeness": row.completeness,
                "parse_issue": row.parse_issue,
                "source_year": row.year,
                "source_month": row.month,
                "source_day": row.day,
            }
        )
    return rows


def _bronze_latest_1min(content: bytes, *, source_id: str, sha256: str, retrieved_at: str) -> list[dict[str, object]]:
    text = content.decode("utf-8-sig")
    reader = csv.DictReader(text.splitlines())
    rows: list[dict[str, object]] = []
    for raw in reader:
        observed_hkt, observed_utc = _parse_hko_datetime(raw["Date time"])
        rows.append(
            {
                "source_id": source_id,
                "content_sha256": sha256,
                "retrieved_at": retrieved_at,
                "station_name": raw["Automatic Weather Station"],
                "parameter": "one_minute_mean_air_temperature",
                "unit": "degree_celsius",
                "observed_at_hkt": observed_hkt,
                "observed_at_utc": observed_utc,
                "value": raw["Air Temperature(degree Celsius)"],
                "parse_issue": "",
            }
        )
    return rows


def _bronze_since_midnight(content: bytes, *, source_id: str, sha256: str, retrieved_at: str) -> list[dict[str, object]]:
    text = content.decode("utf-8-sig")
    reader = csv.DictReader(text.splitlines())
    rows: list[dict[str, object]] = []
    for raw in reader:
        observed_hkt, observed_utc = _parse_hko_datetime(raw["Date time"])
        rows.append(
            {
                "source_id": source_id,
                "content_sha256": sha256,
                "retrieved_at": retrieved_at,
                "station_name": raw["Automatic Weather Station"],
                "unit": "degree_celsius",
                "observed_at_hkt": observed_hkt,
                "observed_at_utc": observed_utc,
                "maximum_since_midnight": raw[
                    "Maximum Air Temperature Since Midnight(degree Celsius)"
                ],
                "minimum_since_midnight": raw[
                    "Minimum Air Temperature Since Midnight(degree Celsius)"
                ],
                "parse_issue": "",
            }
        )
    return rows


def _bronze_weather_json(content: bytes, *, source_id: str, sha256: str, retrieved_at: str) -> list[dict[str, object]]:
    payload = json.loads(content.decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise BronzeBuildError(f"{source_id} JSON payload is not an object")
    rows: list[dict[str, object]] = []
    if source_id == "hko_local_weather_forecast":
        rows.append(
            {
                "source_id": source_id,
                "content_sha256": sha256,
                "retrieved_at": retrieved_at,
                "issued_at": payload.get("updateTime", ""),
                "forecast_period": payload.get("forecastPeriod", ""),
                "forecast_description": payload.get("forecastDesc", ""),
                "outlook": payload.get("outlook", ""),
                "tc_info": payload.get("tcInfo", ""),
            }
        )
        return rows
    if source_id == "hko_nine_day_forecast":
        forecast_detail = payload.get("weatherForecast")
        if not isinstance(forecast_detail, list):
            raise BronzeBuildError("Nine-day forecast missing weatherForecast list")
        for item in forecast_detail:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "source_id": source_id,
                    "content_sha256": sha256,
                    "retrieved_at": retrieved_at,
                    "issued_at": payload.get("updateTime", ""),
                    "forecast_date": item.get("forecastDate", ""),
                    "week": item.get("week", ""),
                    "forecast_max_temperature": (item.get("forecastMaxtemp") or {}).get("value", ""),
                    "forecast_min_temperature": (item.get("forecastMintemp") or {}).get("value", ""),
                    "forecast_temperature_unit": (item.get("forecastMaxtemp") or {}).get("unit", ""),
                    "forecast_weather": item.get("forecastWeather", ""),
                    "forecast_wind": item.get("forecastWind", ""),
                    "forecast_max_rh": (item.get("forecastMaxrh") or {}).get("value", ""),
                    "forecast_min_rh": (item.get("forecastMinrh") or {}).get("value", ""),
                }
            )
        return rows
    raise BronzeBuildError(f"No JSON bronze adapter for {source_id}")


def build_bronze_latest(data_root: Path, *, source_id: str) -> BronzeDataset:
    latest = _latest_success(data_root, source_id)
    content_path = resolve_archive_content_path(latest, data_root=data_root)
    content = content_path.read_bytes()
    sha256 = latest["content_sha256"]
    retrieved_at = latest["retrieved_at"]
    if source_id == "hko_clmmaxt_hko":
        rows = _bronze_clmmaxt(content, source_id=source_id, sha256=sha256, retrieved_at=retrieved_at)
    elif source_id == "hko_latest_1min_temperature":
        rows = _bronze_latest_1min(content, source_id=source_id, sha256=sha256, retrieved_at=retrieved_at)
    elif source_id == "hko_since_midnight_maxmin":
        rows = _bronze_since_midnight(content, source_id=source_id, sha256=sha256, retrieved_at=retrieved_at)
    elif source_id in {"hko_local_weather_forecast", "hko_nine_day_forecast"}:
        rows = _bronze_weather_json(content, source_id=source_id, sha256=sha256, retrieved_at=retrieved_at)
    else:
        raise BronzeBuildError(f"No bronze adapter for {source_id}")
    if not rows:
        raise BronzeBuildError(f"Bronze adapter produced no rows for {source_id}")

    out_dir = data_root / "bronze" / source_id
    parquet_path = out_dir / f"{sha256}.parquet"
    metadata_path = out_dir / f"{sha256}.metadata.json"
    _write_parquet(parquet_path, rows)
    metadata = {
        "source_id": source_id,
        "content_sha256": sha256,
        "retrieved_at": retrieved_at,
        "row_count": len(rows),
        "schema_version": "bronze_hko_v1",
        "raw_content_path": str(content_path),
        "bronze_path": str(parquet_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return BronzeDataset(
        source_id=source_id,
        content_sha256=sha256,
        row_count=len(rows),
        parquet_path=parquet_path,
        metadata_path=metadata_path,
    )
