from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import math
import os
import re
import struct
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, time as dt_time, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import cfgrib
import httpx
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "0007_public_7day_gfs_gefs_himawari_backfill_rehearsal_20260708"
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "hkg_tmax" / EXPERIMENT_ID
RAW_DIR = EXPERIMENT_DIR / "raw"
NORMALIZED_DIR = EXPERIMENT_DIR / "normalized"
METADATA_DIR = EXPERIMENT_DIR / "metadata"
USER_AGENT = "weather-markets-hkg-public-7day-backfill-rehearsal/1.0"

HKG_BBOX = {"leftlon": "113.0", "rightlon": "115.5", "toplat": "23.5", "bottomlat": "21.5"}
HKO = {
    "station_id": "hko:HKO",
    "station_name": "Hong Kong Observatory",
    "latitude": 22.301944,
    "longitude": 114.174167,
}
HKT = timezone(timedelta(hours=8))

MODEL_CYCLES = [0, 6, 12, 18]
DEFAULT_MODEL_LEADS = list(range(0, 49, 3))
MODEL_AVAILABILITY_BUFFER_HOURS = 6

# Common GFS/GEFS feature pack for HKG Tmax physics. This is deliberately
# broader than a smoke test while staying bounded to the Hong Kong box.
MODEL_FILTER_PARAMS = {
    "lev_2_m_above_ground": "on",
    "var_TMP": "on",
    "var_DPT": "on",
    "var_RH": "on",
    "var_TMAX": "on",
    "var_TMIN": "on",
    "lev_10_m_above_ground": "on",
    "var_UGRD": "on",
    "var_VGRD": "on",
    "lev_mean_sea_level": "on",
    "var_PRMSL": "on",
    "lev_surface": "on",
    "var_GUST": "on",
    "var_APCP": "on",
    "var_DSWRF": "on",
    "var_CAPE": "on",
    "var_CIN": "on",
    "lev_entire_atmosphere": "on",
    "var_PWAT": "on",
    "var_TCDC": "on",
}


@dataclass(frozen=True)
class FetchTask:
    kind: str
    source: str
    item_id: str
    url: str
    path: str
    issue_day_utc: str | None = None
    cycle_hour: int | None = None
    lead_hour: int | None = None
    issued_at_utc: str | None = None
    valid_at_utc: str | None = None
    observed_at_utc: str | None = None
    band: str | None = None
    segment: str | None = None
    availability_proxy_utc: str | None = None
    availability_proxy_method: str | None = None


@dataclass
class FetchResult:
    kind: str
    source: str
    item_id: str
    status: str
    url: str
    path: str | None
    bytes: int
    sha256: str | None
    elapsed_seconds: float
    retrieved_at_utc: str
    fetched_now: bool
    issue_day_utc: str | None = None
    cycle_hour: int | None = None
    lead_hour: int | None = None
    issued_at_utc: str | None = None
    valid_at_utc: str | None = None
    observed_at_utc: str | None = None
    band: str | None = None
    segment: str | None = None
    availability_proxy_utc: str | None = None
    availability_proxy_method: str | None = None
    http_last_modified_utc: str | None = None
    content_length_header: int | None = None
    content_type: str | None = None
    error: str | None = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat().replace("+00:00", "Z")


def iso(value: datetime | pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat().replace("+00:00", "Z")


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def parse_http_datetime(value: str | None) -> str | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except (TypeError, ValueError, IndexError, OverflowError):
        return None


def wp(path: Path) -> str:
    resolved = str(path.resolve())
    if sys.platform.startswith("win") and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def ensure_dir(path: Path) -> None:
    Path(wp(path)).mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    Path(wp(path)).write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def file_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for dirpath, _dirnames, filenames in os.walk(wp(path)):
        for name in filenames:
            try:
                total += os.stat(os.path.join(dirpath, name)).st_size
            except FileNotFoundError:
                continue
    return total


def request_bytes(url: str, timeout: int = 180) -> tuple[bytes, dict[str, str]]:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as response:
        headers = {key.lower(): value for key, value in response.headers.items()}
        return response.read(), headers


def request_bytes_follow_redirects(url: str, timeout: int = 60) -> tuple[bytes, dict[str, str]]:
    with httpx.Client(follow_redirects=True, timeout=timeout, headers={"User-Agent": USER_AGENT}) as client:
        response = client.get(url)
        response.raise_for_status()
        headers = {key.lower(): value for key, value in response.headers.items()}
        return response.content, headers


def payload_validation_error(task: FetchTask, data: bytes) -> str | None:
    if not data:
        return "empty payload"
    if task.kind == "model_grib":
        if not data.startswith(b"GRIB"):
            return f"not a GRIB payload; prefix={data[:80]!r}"
        if not data.rstrip().endswith(b"7777"):
            return "GRIB payload does not end with 7777 terminator"
    if task.kind == "himawari_hsd" and not data.startswith(b"BZh"):
        return f"not a bzip2 Himawari HSD payload; prefix={data[:80]!r}"
    return None


def fetch_to_path(task: FetchTask, timeout: int = 180) -> FetchResult:
    started = time.perf_counter()
    path = EXPERIMENT_DIR / task.path
    retrieved_at = utc_now_iso()
    headers: dict[str, str] = {}
    fetched_now = False
    try:
        if Path(wp(path)).exists():
            data = Path(wp(path)).read_bytes()
        else:
            data, headers = request_bytes(task.url, timeout=timeout)
            invalid = payload_validation_error(task, data)
            if invalid is not None:
                return FetchResult(
                    kind=task.kind,
                    source=task.source,
                    item_id=task.item_id,
                    status="invalid_payload",
                    url=task.url,
                    path=None,
                    bytes=len(data),
                    sha256=hashlib.sha256(data).hexdigest(),
                    elapsed_seconds=time.perf_counter() - started,
                    retrieved_at_utc=retrieved_at,
                    fetched_now=True,
                    issue_day_utc=task.issue_day_utc,
                    cycle_hour=task.cycle_hour,
                    lead_hour=task.lead_hour,
                    issued_at_utc=task.issued_at_utc,
                    valid_at_utc=task.valid_at_utc,
                    observed_at_utc=task.observed_at_utc,
                    band=task.band,
                    segment=task.segment,
                    availability_proxy_utc=task.availability_proxy_utc,
                    availability_proxy_method=task.availability_proxy_method,
                    http_last_modified_utc=parse_http_datetime(headers.get("last-modified")),
                    content_length_header=int(headers["content-length"]) if headers.get("content-length", "").isdigit() else None,
                    content_type=headers.get("content-type"),
                    error=invalid,
                )
            ensure_dir(path.parent)
            Path(wp(path)).write_bytes(data)
            fetched_now = True
        invalid = payload_validation_error(task, data)
        if invalid is not None:
            return FetchResult(
                kind=task.kind,
                source=task.source,
                item_id=task.item_id,
                status="invalid_payload",
                url=task.url,
                path=task.path if Path(wp(path)).exists() else None,
                bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
                elapsed_seconds=time.perf_counter() - started,
                retrieved_at_utc=retrieved_at,
                fetched_now=fetched_now,
                issue_day_utc=task.issue_day_utc,
                cycle_hour=task.cycle_hour,
                lead_hour=task.lead_hour,
                issued_at_utc=task.issued_at_utc,
                valid_at_utc=task.valid_at_utc,
                observed_at_utc=task.observed_at_utc,
                band=task.band,
                segment=task.segment,
                availability_proxy_utc=task.availability_proxy_utc,
                availability_proxy_method=task.availability_proxy_method,
                http_last_modified_utc=parse_http_datetime(headers.get("last-modified")),
                content_length_header=int(headers["content-length"]) if headers.get("content-length", "").isdigit() else None,
                content_type=headers.get("content-type"),
                error=invalid,
            )
        return FetchResult(
            kind=task.kind,
            source=task.source,
            item_id=task.item_id,
            status="ok",
            url=task.url,
            path=task.path,
            bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            elapsed_seconds=time.perf_counter() - started,
            retrieved_at_utc=retrieved_at,
            fetched_now=fetched_now,
            issue_day_utc=task.issue_day_utc,
            cycle_hour=task.cycle_hour,
            lead_hour=task.lead_hour,
            issued_at_utc=task.issued_at_utc,
            valid_at_utc=task.valid_at_utc,
            observed_at_utc=task.observed_at_utc,
            band=task.band,
            segment=task.segment,
            availability_proxy_utc=task.availability_proxy_utc,
            availability_proxy_method=task.availability_proxy_method,
            http_last_modified_utc=parse_http_datetime(headers.get("last-modified")),
            content_length_header=int(headers["content-length"]) if headers.get("content-length", "").isdigit() else None,
            content_type=headers.get("content-type"),
        )
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        return FetchResult(
            kind=task.kind,
            source=task.source,
            item_id=task.item_id,
            status="error",
            url=task.url,
            path=None,
            bytes=0,
            sha256=None,
            elapsed_seconds=time.perf_counter() - started,
            retrieved_at_utc=retrieved_at,
            fetched_now=fetched_now,
            issue_day_utc=task.issue_day_utc,
            cycle_hour=task.cycle_hour,
            lead_hour=task.lead_hour,
            issued_at_utc=task.issued_at_utc,
            valid_at_utc=task.valid_at_utc,
            observed_at_utc=task.observed_at_utc,
            band=task.band,
            segment=task.segment,
            availability_proxy_utc=task.availability_proxy_utc,
            availability_proxy_method=task.availability_proxy_method,
            error=f"{type(exc).__name__}: {exc}",
        )


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def completed_utc_yesterday() -> date:
    return utc_now().date() - timedelta(days=1)


def date_span(end_date: date, days: int) -> list[date]:
    start = end_date - timedelta(days=days - 1)
    return [start + timedelta(days=offset) for offset in range(days)]


def parse_leads(value: str) -> list[int]:
    if value.strip().lower() in {"common_3h_0_48", "default"}:
        return DEFAULT_MODEL_LEADS
    leads = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not leads:
        raise ValueError("At least one lead hour is required.")
    return leads


def model_issue(day: date, cycle_hour: int) -> datetime:
    return datetime.combine(day, dt_time(cycle_hour, 0), tzinfo=timezone.utc)


def model_available_at(issue: datetime) -> datetime:
    return issue + timedelta(hours=MODEL_AVAILABILITY_BUFFER_HOURS)


def build_gfs_url(day: date, cycle_hour: int, lead_hour: int) -> str:
    cc = f"{cycle_hour:02d}"
    params = {
        "dir": f"/gfs.{day:%Y%m%d}/{cc}/atmos",
        "file": f"gfs.t{cc}z.pgrb2.0p25.f{lead_hour:03d}",
        **MODEL_FILTER_PARAMS,
        **HKG_BBOX,
    }
    return "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?" + urlencode(params)


def build_gefs_url(day: date, cycle_hour: int, lead_hour: int) -> str:
    cc = f"{cycle_hour:02d}"
    params = {
        "dir": f"/gefs.{day:%Y%m%d}/{cc}/atmos/pgrb2sp25",
        "file": f"gec00.t{cc}z.pgrb2s.0p25.f{lead_hour:03d}",
        **MODEL_FILTER_PARAMS,
        **HKG_BBOX,
    }
    return "https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p25s.pl?" + urlencode(params)


def gfs_idx_url(day: date, cycle_hour: int, lead_hour: int) -> str:
    cc = f"{cycle_hour:02d}"
    return (
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/"
        f"gfs.{day:%Y%m%d}/{cc}/atmos/gfs.t{cc}z.pgrb2.0p25.f{lead_hour:03d}.idx"
    )


def gefs_idx_url(day: date, cycle_hour: int, lead_hour: int) -> str:
    cc = f"{cycle_hour:02d}"
    return (
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod/"
        f"gefs.{day:%Y%m%d}/{cc}/atmos/pgrb2sp25/gec00.t{cc}z.pgrb2s.0p25.f{lead_hour:03d}.idx"
    )


def himawari_resolution_for_band(band: str) -> str:
    if band == "B03":
        return "R05"
    if band in {"B01", "B02", "B04"}:
        return "R10"
    return "R20"


def himawari_url(scan: datetime, band: str, segment: str) -> str:
    resolution = himawari_resolution_for_band(band)
    key = (
        f"AHI-L1b-FLDK/{scan:%Y/%m/%d/%H%M}/"
        f"HS_H09_{scan:%Y%m%d}_{scan:%H%M}_{band}_FLDK_{resolution}_{segment}.DAT.bz2"
    )
    return f"https://noaa-himawari9.s3.amazonaws.com/{key}"


def himawari_scans(day: date) -> list[datetime]:
    return [
        datetime(day.year, day.month, day.day, hour, minute, tzinfo=timezone.utc)
        for hour in range(24)
        for minute in range(0, 60, 10)
    ]


def build_model_tasks(days: list[date], leads: list[int]) -> list[FetchTask]:
    tasks: list[FetchTask] = []
    for day in days:
        for source in ["gfs", "gefs_control"]:
            for cycle_hour in MODEL_CYCLES:
                issue_at = model_issue(day, cycle_hour)
                available_at = model_available_at(issue_at)
                for lead_hour in leads:
                    valid_at = issue_at + timedelta(hours=lead_hour)
                    if source == "gfs":
                        url = build_gfs_url(day, cycle_hour, lead_hour)
                        raw_path = (
                            RAW_DIR.relative_to(EXPERIMENT_DIR)
                            / "gfs"
                            / f"{day:%Y%m%d}"
                            / f"gfs_{day:%Y%m%d}_{cycle_hour:02d}z_f{lead_hour:03d}.grib2"
                        )
                    else:
                        url = build_gefs_url(day, cycle_hour, lead_hour)
                        raw_path = (
                            RAW_DIR.relative_to(EXPERIMENT_DIR)
                            / "gefs_control"
                            / f"{day:%Y%m%d}"
                            / f"gefs_control_{day:%Y%m%d}_{cycle_hour:02d}z_f{lead_hour:03d}.grib2"
                        )
                    tasks.append(
                        FetchTask(
                            kind="model_grib",
                            source=source,
                            item_id=f"{source}_{day:%Y%m%d}_{cycle_hour:02d}z_f{lead_hour:03d}",
                            url=url,
                            path=str(raw_path),
                            issue_day_utc=day.isoformat(),
                            cycle_hour=cycle_hour,
                            lead_hour=lead_hour,
                            issued_at_utc=iso(issue_at),
                            valid_at_utc=iso(valid_at),
                            availability_proxy_utc=iso(available_at),
                            availability_proxy_method=(
                                f"issued_at_utc + {MODEL_AVAILABILITY_BUFFER_HOURS}h conservative NOMADS buffer"
                            ),
                        )
                    )
    return tasks


def build_himawari_tasks(days: list[date], bands: list[str], segment: str) -> list[FetchTask]:
    tasks: list[FetchTask] = []
    for day in days:
        for scan in himawari_scans(day):
            for band in bands:
                resolution = himawari_resolution_for_band(band)
                file_name = f"HS_H09_{scan:%Y%m%d}_{scan:%H%M}_{band}_FLDK_{resolution}_{segment}.DAT.bz2"
                raw_path = RAW_DIR.relative_to(EXPERIMENT_DIR) / "himawari" / band / segment / f"{day:%Y%m%d}" / file_name
                available_at = scan + timedelta(minutes=30)
                tasks.append(
                    FetchTask(
                        kind="himawari_hsd",
                        source=f"himawari9_{band.lower()}_{segment.lower()}",
                        item_id=f"himawari9_{band}_{segment}_{scan:%Y%m%d_%H%M}",
                        url=himawari_url(scan, band, segment),
                        path=str(raw_path),
                        observed_at_utc=iso(scan),
                        band=band,
                        segment=segment,
                        availability_proxy_utc=iso(available_at),
                        availability_proxy_method="observed_at_utc + 30m conservative live-ingestion buffer",
                    )
                )
    return tasks


def parse_idx(data: bytes) -> dict[str, Any]:
    text = data.decode("utf-8", errors="replace")
    rows = [line for line in text.splitlines() if line.strip()]
    variables: set[str] = set()
    variable_level_pairs: set[tuple[str, str]] = set()
    for line in rows:
        parts = line.split(":")
        if len(parts) >= 5:
            variables.add(parts[3])
            variable_level_pairs.add((parts[3], parts[4]))
    return {
        "message_count": len(rows),
        "unique_variable_count": len(variables),
        "unique_variables": sorted(variables),
        "unique_variable_level_pair_count": len(variable_level_pairs),
    }


def fetch_idx_catalog(days: list[date], leads: list[int], workers: int) -> list[dict[str, Any]]:
    idx_jobs: list[dict[str, Any]] = []
    for day in days:
        for source in ["gfs", "gefs_control"]:
            for cycle_hour in MODEL_CYCLES:
                for lead_hour in leads:
                    idx_jobs.append(
                        {
                            "source": source,
                            "issue_day_utc": day.isoformat(),
                            "cycle_hour": cycle_hour,
                            "lead_hour": lead_hour,
                            "url": gfs_idx_url(day, cycle_hour, lead_hour)
                            if source == "gfs"
                            else gefs_idx_url(day, cycle_hour, lead_hour),
                        }
                    )

    def fetch_one(job: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            data, headers = request_bytes_follow_redirects(job["url"], timeout=60)
            parsed = parse_idx(data)
            return {
                **job,
                **parsed,
                "status": "ok",
                "elapsed_seconds": time.perf_counter() - started,
                "http_last_modified_utc": parse_http_datetime(headers.get("last-modified")),
                "error": None,
            }
        except Exception as exc:
            return {
                **job,
                "message_count": None,
                "unique_variable_count": None,
                "unique_variables": [],
                "unique_variable_level_pair_count": None,
                "status": "error",
                "elapsed_seconds": time.perf_counter() - started,
                "http_last_modified_utc": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    return run_pool("model-idx", idx_jobs, fetch_one, workers, progress_every=100)


def normalize_value(value: float, units: str | None) -> tuple[float, str]:
    if units == "K":
        return float(value) - 273.15, "degC"
    if units == "Pa":
        return float(value) / 100.0, "hPa"
    return float(value), re.sub(r"[^a-zA-Z0-9]+", "_", units or "").strip("_")


def lead_hours_from_coord(value: Any) -> float | None:
    try:
        return float(value / np.timedelta64(1, "h"))
    except Exception:
        return None


def scalar_coord(da: Any, name: str) -> Any:
    if name not in da.coords:
        return None
    value = da.coords[name].values
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0]
    return value


def crop_dataarray(da: Any) -> Any:
    lat_values = np.asarray(da["latitude"].values, dtype=float)
    lat_slice = (
        slice(float(HKG_BBOX["toplat"]), float(HKG_BBOX["bottomlat"]))
        if lat_values[0] > lat_values[-1]
        else slice(float(HKG_BBOX["bottomlat"]), float(HKG_BBOX["toplat"]))
    )
    return da.sel(latitude=lat_slice, longitude=slice(float(HKG_BBOX["leftlon"]), float(HKG_BBOX["rightlon"])))


def nearest_grid(ds: Any) -> tuple[float, float]:
    lat_values = np.asarray(ds["latitude"].values, dtype=float)
    lon_values = np.asarray(ds["longitude"].values, dtype=float)
    lat = float(lat_values[np.argmin(np.abs(lat_values - HKO["latitude"]))])
    lon = float(lon_values[np.argmin(np.abs(lon_values - HKO["longitude"]))])
    return lat, lon


def safe_feature_key(base: str, units: str, existing: dict[str, Any]) -> str:
    key = f"{base}_{units}".strip("_").lower()
    key = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    if key not in existing:
        return key
    suffix = 2
    while f"{key}_{suffix}" in existing:
        suffix += 1
    return f"{key}_{suffix}"


def normalize_model_result(fetch: FetchResult) -> dict[str, Any]:
    started = time.perf_counter()
    if fetch.status != "ok" or not fetch.path:
        return {"status": "skip", "fetch": asdict(fetch), "station_row": None, "bbox_rows": []}
    if fetch.lead_hour == 0:
        return {
            "status": "skipped_f000_analysis_lead",
            "fetch": asdict(fetch),
            "station_row": {
                "source": fetch.source,
                "item_id": fetch.item_id,
                "issue_day_utc": fetch.issue_day_utc,
                "cycle_hour": fetch.cycle_hour,
                "lead_hour_requested": fetch.lead_hour,
                "issued_at_utc": fetch.issued_at_utc,
                "valid_at_utc_expected": fetch.valid_at_utc,
                "availability_proxy_utc": fetch.availability_proxy_utc,
                "availability_proxy_method": fetch.availability_proxy_method,
                "http_last_modified_utc": fetch.http_last_modified_utc,
                "raw_path": fetch.path,
                "raw_bytes": fetch.bytes,
                "raw_sha256": fetch.sha256,
                "normalized_variable_count": 0,
                "normalization_elapsed_seconds": time.perf_counter() - started,
                "normalization_note": "Skipped f000 analysis/initialization lead; fetched and cataloged but not decoded for forward T-24 features.",
            },
            "bbox_rows": [],
        }
    path = EXPERIMENT_DIR / fetch.path
    station_row: dict[str, Any] = {
        "source": fetch.source,
        "item_id": fetch.item_id,
        "issue_day_utc": fetch.issue_day_utc,
        "cycle_hour": fetch.cycle_hour,
        "lead_hour_requested": fetch.lead_hour,
        "issued_at_utc": fetch.issued_at_utc,
        "valid_at_utc_expected": fetch.valid_at_utc,
        "availability_proxy_utc": fetch.availability_proxy_utc,
        "availability_proxy_method": fetch.availability_proxy_method,
        "http_last_modified_utc": fetch.http_last_modified_utc,
        "raw_path": fetch.path,
        "raw_bytes": fetch.bytes,
        "raw_sha256": fetch.sha256,
        "station_id": HKO["station_id"],
        "station_name": HKO["station_name"],
        "station_latitude": HKO["latitude"],
        "station_longitude": HKO["longitude"],
    }
    summary_rows: list[dict[str, Any]] = []
    variable_count = 0
    nearest_lat = None
    nearest_lon = None
    try:
        dsets = cfgrib.open_datasets(wp(path), backend_kwargs={"indexpath": ""})
        for ds in dsets:
            if "latitude" not in ds.coords or "longitude" not in ds.coords:
                continue
            if nearest_lat is None:
                nearest_lat, nearest_lon = nearest_grid(ds)
                station_row["nearest_grid_latitude"] = nearest_lat
                station_row["nearest_grid_longitude"] = nearest_lon
            for var_name in ds.data_vars:
                da = ds[var_name].squeeze(drop=True)
                if "latitude" not in da.coords or "longitude" not in da.coords:
                    continue
                attrs = da.attrs
                units = attrs.get("GRIB_units")
                short_name = attrs.get("GRIB_shortName", var_name)
                type_level = attrs.get("GRIB_typeOfLevel", "")
                level = attrs.get("GRIB_level", "")
                step_type = attrs.get("GRIB_stepType", "")
                canonical = re.sub(
                    r"[^a-zA-Z0-9]+",
                    "_",
                    f"{short_name}_{type_level}_{level}_{step_type}",
                ).strip("_")

                issued_at = iso(pd.Timestamp(scalar_coord(da, "time"))) if scalar_coord(da, "time") is not None else None
                valid_at = (
                    iso(pd.Timestamp(scalar_coord(da, "valid_time")))
                    if scalar_coord(da, "valid_time") is not None
                    else None
                )
                lead = lead_hours_from_coord(scalar_coord(da, "step")) if scalar_coord(da, "step") is not None else None
                if issued_at:
                    station_row["issued_at_utc_grib"] = issued_at
                if valid_at:
                    station_row["valid_at_utc_grib"] = valid_at
                if lead is not None:
                    station_row["lead_hour_grib"] = lead

                point_native = float(da.sel(latitude=nearest_lat, longitude=nearest_lon).values)
                point_value, out_unit = normalize_value(point_native, units)
                station_row[safe_feature_key(canonical, out_unit, station_row)] = point_value

                cropped = crop_dataarray(da)
                values = np.asarray(cropped.values, dtype=float)
                if values.size == 0:
                    continue
                if units in {"K", "Pa"}:
                    normalized = np.vectorize(lambda x: normalize_value(float(x), units)[0])(values)
                else:
                    normalized = values.astype(float)
                finite = np.isfinite(normalized)
                if not finite.any():
                    continue
                summary_rows.append(
                    {
                        "source": fetch.source,
                        "item_id": fetch.item_id,
                        "issue_day_utc": fetch.issue_day_utc,
                        "cycle_hour": fetch.cycle_hour,
                        "lead_hour_requested": fetch.lead_hour,
                        "issued_at_utc": issued_at or fetch.issued_at_utc,
                        "valid_at_utc": valid_at or fetch.valid_at_utc,
                        "lead_hour_grib": lead,
                        "availability_proxy_utc": fetch.availability_proxy_utc,
                        "variable": var_name,
                        "grib_short_name": short_name,
                        "grib_name": attrs.get("GRIB_name"),
                        "grib_type_of_level": type_level,
                        "grib_level": level,
                        "grib_step_type": step_type,
                        "grib_units": units,
                        "normalized_units": out_unit,
                        "grid_point_count": int(finite.sum()),
                        "bbox_min": float(np.nanmin(normalized)),
                        "bbox_mean": float(np.nanmean(normalized)),
                        "bbox_median": float(np.nanmedian(normalized)),
                        "bbox_max": float(np.nanmax(normalized)),
                        "bbox_std": float(np.nanstd(normalized)),
                    }
                )
                variable_count += 1
        station_row["normalized_variable_count"] = variable_count
        station_row["normalization_elapsed_seconds"] = time.perf_counter() - started
        return {"status": "ok", "fetch": asdict(fetch), "station_row": station_row, "bbox_rows": summary_rows}
    except Exception as exc:
        station_row["normalized_variable_count"] = variable_count
        station_row["normalization_elapsed_seconds"] = time.perf_counter() - started
        station_row["normalization_error"] = f"{type(exc).__name__}: {exc}"
        return {"status": "error", "fetch": asdict(fetch), "station_row": station_row, "bbox_rows": summary_rows}


def read_c_string(data: bytes, offset: int, length: int) -> str:
    return data[offset : offset + length].split(b"\0", 1)[0].decode("ascii", errors="replace").strip()


def mjd_to_iso(mjd: float) -> str | None:
    if not np.isfinite(mjd):
        return None
    ts = pd.Timestamp("1858-11-17T00:00:00Z") + pd.to_timedelta(mjd, unit="D")
    return ts.isoformat().replace("+00:00", "Z")


def parse_himawari_file_name(file_name: str) -> dict[str, Any]:
    match = re.match(
        r"HS_(H\d{2})_(\d{8})_(\d{4})_(B\d{2})_(\w+)_(R\d+)_(S(\d{2})(\d{2}))\.DAT",
        file_name.replace(".bz2", ""),
    )
    if not match:
        return {}
    observed = datetime.strptime(match.group(2) + match.group(3), "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    return {
        "satellite_code": match.group(1),
        "observed_at_utc": iso(observed),
        "band": match.group(4),
        "area": match.group(5),
        "resolution_code": match.group(6),
        "segment_code": match.group(7),
        "segment_number": int(match.group(8)),
        "segment_count": int(match.group(9)),
    }


def parse_himawari_header(data: bytes, fetch: FetchResult) -> dict[str, Any]:
    basic = 0
    data_info = 282
    projection = 332
    calibration = 598
    segment = 1004
    observation_time = 1132
    file_name = read_c_string(data, basic + 114, 128)
    file_info = parse_himawari_file_name(file_name)

    number_of_observation_times = struct.unpack_from("<H", data, observation_time + 3)[0]
    observation_times: list[dict[str, Any]] = []
    cursor = observation_time + 5
    for _ in range(number_of_observation_times):
        line_number = struct.unpack_from("<H", data, cursor)[0]
        mjd = struct.unpack_from("<d", data, cursor + 2)[0]
        observation_times.append({"line_number": int(line_number), "mjd": float(mjd), "utc": mjd_to_iso(mjd)})
        cursor += 10

    return {
        "source": fetch.source,
        "source_record": asdict(fetch),
        "hsd_file_name": file_name,
        **file_info,
        "satellite_name": read_c_string(data, basic + 6, 16),
        "processing_center": read_c_string(data, basic + 22, 16),
        "observation_start_utc": mjd_to_iso(struct.unpack_from("<d", data, basic + 46)[0]),
        "observation_end_utc": mjd_to_iso(struct.unpack_from("<d", data, basic + 54)[0]),
        "file_creation_utc": mjd_to_iso(struct.unpack_from("<d", data, basic + 62)[0]),
        "header_total_bytes": struct.unpack_from("<I", data, basic + 70)[0],
        "data_total_bytes": struct.unpack_from("<I", data, basic + 74)[0],
        "format_version": read_c_string(data, basic + 82, 32),
        "bits_per_pixel": struct.unpack_from("<H", data, data_info + 3)[0],
        "columns": struct.unpack_from("<H", data, data_info + 5)[0],
        "lines_in_segment": struct.unpack_from("<H", data, data_info + 7)[0],
        "projection": {
            "sub_satellite_longitude_deg": struct.unpack_from("<d", data, projection + 3)[0],
            "cfac": struct.unpack_from("<i", data, projection + 11)[0],
            "lfac": struct.unpack_from("<i", data, projection + 15)[0],
            "coff": struct.unpack_from("<f", data, projection + 19)[0],
            "loff": struct.unpack_from("<f", data, projection + 23)[0],
            "satellite_distance_km": struct.unpack_from("<d", data, projection + 27)[0],
            "earth_equatorial_radius_km": struct.unpack_from("<d", data, projection + 35)[0],
            "earth_polar_radius_km": struct.unpack_from("<d", data, projection + 43)[0],
        },
        "calibration": {
            "band_number": struct.unpack_from("<H", data, calibration + 3)[0],
            "central_wavelength_um": struct.unpack_from("<d", data, calibration + 5)[0],
            "valid_bits_per_pixel": struct.unpack_from("<H", data, calibration + 13)[0],
            "error_count": struct.unpack_from("<H", data, calibration + 15)[0],
            "outside_scan_count": struct.unpack_from("<H", data, calibration + 17)[0],
            "count_to_radiance_slope": struct.unpack_from("<d", data, calibration + 19)[0],
            "count_to_radiance_intercept": struct.unpack_from("<d", data, calibration + 27)[0],
            "radiance_to_bt_c0": struct.unpack_from("<d", data, calibration + 35)[0],
            "radiance_to_bt_c1": struct.unpack_from("<d", data, calibration + 43)[0],
            "radiance_to_bt_c2": struct.unpack_from("<d", data, calibration + 51)[0],
        },
        "segment": {
            "total_segments": data[segment + 3],
            "segment_sequence_number": data[segment + 4],
            "first_global_line_number": struct.unpack_from("<H", data, segment + 5)[0],
        },
        "observation_times": observation_times,
    }


def hko_pixel(header: dict[str, Any]) -> tuple[int, int, float, float]:
    proj = header["projection"]
    lat = math.radians(HKO["latitude"])
    lon = math.radians(HKO["longitude"])
    lon0 = math.radians(proj["sub_satellite_longitude_deg"])
    req = proj["earth_equatorial_radius_km"]
    rpol = proj["earth_polar_radius_km"]
    rs = proj["satellite_distance_km"]
    phi_c = math.atan((rpol * rpol) / (req * req) * math.tan(lat))
    re_phi = rpol / math.sqrt(1.0 - ((req * req - rpol * rpol) / (req * req)) * math.cos(phi_c) ** 2)
    rel_lon = lon - lon0
    r1 = rs - re_phi * math.cos(phi_c) * math.cos(rel_lon)
    r2 = -re_phi * math.cos(phi_c) * math.sin(rel_lon)
    r3 = re_phi * math.sin(phi_c)
    x = math.atan(r2 / r1)
    y = math.atan(r3 / math.sqrt(r1 * r1 + r2 * r2))
    global_col = proj["coff"] + x * proj["cfac"] / (2**16)
    global_line = proj["loff"] - y * proj["lfac"] / (2**16)
    local_row = int(round(global_line - header["segment"]["first_global_line_number"]))
    local_col = int(round(global_col - 1))
    return local_row, local_col, global_line, global_col


def himawari_bt(data: bytes, header: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    columns = int(header["columns"])
    lines = int(header["lines_in_segment"])
    counts = np.frombuffer(data, dtype="<u2", count=columns * lines, offset=int(header["header_total_bytes"])).reshape(
        lines, columns
    )
    cal = header["calibration"]
    error_count = int(cal["error_count"])
    outside_count = int(cal["outside_scan_count"])
    valid = (counts != outside_count) & (counts != error_count)
    radiance = cal["count_to_radiance_slope"] * counts.astype("float64") + cal["count_to_radiance_intercept"]
    radiance[~valid] = np.nan
    radiance[radiance <= 0] = np.nan
    c1 = 1.191042e8
    c2 = 1.4387752e4
    wavelength = cal["central_wavelength_um"]
    effective_bt = c2 / (wavelength * np.log(c1 / (radiance * (wavelength**5)) + 1.0))
    bt_k = cal["radiance_to_bt_c0"] + cal["radiance_to_bt_c1"] * effective_bt + cal["radiance_to_bt_c2"] * effective_bt**2
    quality_code = np.zeros_like(counts, dtype="uint8")
    quality_code[counts == outside_count] = 1
    quality_code[counts == error_count] = 2
    return counts, radiance.astype("float32"), (bt_k - 273.15).astype("float32"), quality_code


def window_features(prefix: str, matrix: np.ndarray, row: int, col: int, radius: int) -> dict[str, Any]:
    window = matrix[max(0, row - radius) : row + radius + 1, max(0, col - radius) : col + radius + 1]
    vals = window[np.isfinite(window)]
    if vals.size == 0:
        return {f"{prefix}_pixel_count": 0}
    return {
        f"{prefix}_pixel_count": int(vals.size),
        f"{prefix}_mean_bt_c": float(np.mean(vals)),
        f"{prefix}_median_bt_c": float(np.median(vals)),
        f"{prefix}_p10_bt_c": float(np.percentile(vals, 10)),
        f"{prefix}_p90_bt_c": float(np.percentile(vals, 90)),
        f"{prefix}_min_bt_c": float(np.min(vals)),
        f"{prefix}_max_bt_c": float(np.max(vals)),
        f"{prefix}_std_bt_c": float(np.std(vals)),
        f"{prefix}_range_bt_c": float(np.max(vals) - np.min(vals)),
        f"{prefix}_cloud_fraction_lt_0c": float(np.mean(vals < 0)),
        f"{prefix}_cloud_fraction_lt_10c": float(np.mean(vals < 10)),
        f"{prefix}_cloud_fraction_lt_15c": float(np.mean(vals < 15)),
        f"{prefix}_cool_fraction_lt_20c": float(np.mean(vals < 20)),
        f"{prefix}_warm_fraction_gt_20c": float(np.mean(vals > 20)),
        f"{prefix}_warm_fraction_gt_23c": float(np.mean(vals > 23)),
    }


def hkt_date(value_utc: str | None) -> str | None:
    parsed = parse_iso(value_utc)
    if parsed is None:
        return None
    return parsed.astimezone(HKT).date().isoformat()


def t24_next_day_cutoff_for_observed(value_utc: str | None) -> str | None:
    parsed = parse_iso(value_utc)
    if parsed is None:
        return None
    local_date = parsed.astimezone(HKT).date()
    cutoff_hkt = datetime.combine(local_date, dt_time(15, 0), tzinfo=HKT)
    return iso(cutoff_hkt.astimezone(timezone.utc))


def normalize_himawari_result(fetch: FetchResult) -> dict[str, Any]:
    started = time.perf_counter()
    if fetch.status != "ok" or not fetch.path:
        return {"status": "skip", "fetch": asdict(fetch), "row": None}
    path = EXPERIMENT_DIR / fetch.path
    try:
        with bz2.open(wp(path), "rb") as handle:
            data = handle.read()
        header = parse_himawari_header(data, fetch)
        counts, radiance, bt_c, quality_code = himawari_bt(data, header)
        row, col, global_line, global_col = hko_pixel(header)
        if not (0 <= row < bt_c.shape[0] and 0 <= col < bt_c.shape[1]):
            raise ValueError(f"Projected HKO pixel outside segment: row={row}, col={col}")

        file_created = header.get("file_creation_utc")
        available_proxy = fetch.availability_proxy_utc
        if file_created and parse_iso(file_created):
            observed_plus_30 = parse_iso(fetch.availability_proxy_utc)
            created = parse_iso(file_created)
            available_proxy = iso(max(created, observed_plus_30)) if observed_plus_30 and created else file_created

        valid_bt = bt_c[np.isfinite(bt_c)]
        row_data: dict[str, Any] = {
            "source": fetch.source,
            "item_id": fetch.item_id,
            "band": header.get("band") or fetch.band,
            "segment": header.get("segment_code") or fetch.segment,
            "observed_at_utc": header.get("observed_at_utc") or fetch.observed_at_utc,
            "observed_date_hkt": hkt_date(header.get("observed_at_utc") or fetch.observed_at_utc),
            "availability_proxy_utc": available_proxy,
            "availability_proxy_method": "max(hsd_file_creation_utc, observed_at_utc + 30m conservative buffer)",
            "t24_next_target_date_hkt": (
                (parse_iso(header.get("observed_at_utc") or fetch.observed_at_utc).astimezone(HKT).date() + timedelta(days=1)).isoformat()
                if parse_iso(header.get("observed_at_utc") or fetch.observed_at_utc)
                else None
            ),
            "t24_next_day_cutoff_utc": t24_next_day_cutoff_for_observed(
                header.get("observed_at_utc") or fetch.observed_at_utc
            ),
            "eligible_for_next_day_t24_cutoff": (
                parse_iso(available_proxy) <= parse_iso(t24_next_day_cutoff_for_observed(header.get("observed_at_utc") or fetch.observed_at_utc))
                if parse_iso(available_proxy)
                and parse_iso(t24_next_day_cutoff_for_observed(header.get("observed_at_utc") or fetch.observed_at_utc))
                else None
            ),
            "file_creation_utc": file_created,
            "http_last_modified_utc": fetch.http_last_modified_utc,
            "file_name": header["hsd_file_name"],
            "raw_path": fetch.path,
            "raw_bytes": fetch.bytes,
            "raw_sha256": fetch.sha256,
            "columns": int(header["columns"]),
            "lines_in_segment": int(header["lines_in_segment"]),
            "valid_pixel_count": int(np.sum(quality_code == 0)),
            "outside_scan_pixel_count": int(np.sum(quality_code == 1)),
            "error_pixel_count": int(np.sum(quality_code == 2)),
            "hko_global_line": float(global_line),
            "hko_global_col": float(global_col),
            "hko_local_row": int(row),
            "hko_local_col": int(col),
            "hko_count": int(counts[row, col]),
            "hko_quality_code": int(quality_code[row, col]),
            "hko_radiance_w_m2_sr_um": float(radiance[row, col]),
            "hko_bt_c": float(bt_c[row, col]),
            "segment_bt_c_p05": float(np.nanpercentile(valid_bt, 5)) if valid_bt.size else None,
            "segment_bt_c_median": float(np.nanmedian(valid_bt)) if valid_bt.size else None,
            "segment_bt_c_p95": float(np.nanpercentile(valid_bt, 95)) if valid_bt.size else None,
        }
        row_data.update(window_features("w3", bt_c, row, col, 1))
        row_data.update(window_features("w5", bt_c, row, col, 2))
        row_data.update(window_features("w11", bt_c, row, col, 5))
        row_data.update(window_features("w21", bt_c, row, col, 10))
        if "w21_mean_bt_c" in row_data:
            row_data["hko_minus_w21_mean_bt_c"] = float(row_data["hko_bt_c"] - row_data["w21_mean_bt_c"])
        if 5 <= row < bt_c.shape[0] - 5 and 5 <= col < bt_c.shape[1] - 5:
            row_data["east_west_gradient_bt_c"] = float(np.nanmean(bt_c[row - 5 : row + 6, col + 1 : col + 6]) - np.nanmean(bt_c[row - 5 : row + 6, col - 5 : col]))
            row_data["south_north_gradient_bt_c"] = float(np.nanmean(bt_c[row + 1 : row + 6, col - 5 : col + 6]) - np.nanmean(bt_c[row - 5 : row, col - 5 : col + 6]))
        row_data["normalization_elapsed_seconds"] = time.perf_counter() - started
        return {"status": "ok", "fetch": asdict(fetch), "row": row_data}
    except Exception as exc:
        return {
            "status": "error",
            "fetch": asdict(fetch),
            "row": {
                "source": fetch.source,
                "item_id": fetch.item_id,
                "band": fetch.band,
                "segment": fetch.segment,
                "observed_at_utc": fetch.observed_at_utc,
                "availability_proxy_utc": fetch.availability_proxy_utc,
                "http_last_modified_utc": fetch.http_last_modified_utc,
                "raw_path": fetch.path,
                "raw_bytes": fetch.bytes,
                "raw_sha256": fetch.sha256,
                "normalization_elapsed_seconds": time.perf_counter() - started,
                "normalization_error": f"{type(exc).__name__}: {exc}",
            },
        }


def run_pool(
    label: str,
    jobs: list[Any],
    fn: Any,
    workers: int,
    progress_every: int = 25,
    use_processes: bool = False,
) -> list[Any]:
    if not jobs:
        return []
    started = time.perf_counter()
    out: list[Any] = []
    executor_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    with executor_cls(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(fn, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "status": "worker_exception",
                    "error": f"{type(exc).__name__}: {exc}",
                    "job": object_to_record(futures[future]) if not isinstance(futures[future], (str, int, float)) else futures[future],
                }
            out.append(result)
            if index == 1 or index % progress_every == 0 or index == len(jobs):
                print(
                    f"{label}: {index}/{len(jobs)} done in {time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    return out


def s3_list_scan(scan: datetime) -> list[dict[str, Any]]:
    url = f"https://noaa-himawari9.s3.amazonaws.com/?list-type=2&prefix=AHI-L1b-FLDK/{scan:%Y/%m/%d/%H%M}/&max-keys=1000"
    data, _headers = request_bytes(url, timeout=60)
    root = ET.fromstring(data)
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    rows: list[dict[str, Any]] = []
    for item in root.findall("s3:Contents", ns):
        key = item.findtext("s3:Key", default="", namespaces=ns)
        size = int(item.findtext("s3:Size", default="0", namespaces=ns))
        last_modified = item.findtext("s3:LastModified", default="", namespaces=ns)
        rows.append({"key": key, "size": size, "last_modified_utc": last_modified})
    return rows


def probe_himawari_scan_size(scan: datetime, segment: str, band: str) -> dict[str, Any]:
    try:
        rows = s3_list_scan(scan)
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}", "scan": iso(scan)}
    segment_token = f"_{segment}.DAT.bz2"
    band_token = f"_{band}_"
    selected_band_segment = [row for row in rows if band_token in row["key"] and row["key"].endswith(segment_token)]
    selected_all_bands_segment = [row for row in rows if row["key"].endswith(segment_token)]
    return {
        "status": "ok",
        "scan": iso(scan),
        "total_files_in_scan": len(rows),
        "total_bytes_full_disk_all_bands_all_segments": int(sum(row["size"] for row in rows)),
        "selected_band_segment_files": len(selected_band_segment),
        "selected_band_segment_bytes": int(sum(row["size"] for row in selected_band_segment)),
        "all_bands_selected_segment_files": len(selected_all_bands_segment),
        "all_bands_selected_segment_bytes": int(sum(row["size"] for row in selected_all_bands_segment)),
        "all_bands_selected_segment_to_selected_band_ratio": (
            float(sum(row["size"] for row in selected_all_bands_segment) / sum(row["size"] for row in selected_band_segment))
            if selected_band_segment and sum(row["size"] for row in selected_band_segment)
            else None
        ),
        "full_disk_all_bands_to_selected_band_segment_ratio": (
            float(sum(row["size"] for row in rows) / sum(row["size"] for row in selected_band_segment))
            if selected_band_segment and sum(row["size"] for row in selected_band_segment)
            else None
        ),
        "rows": rows,
    }


def dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(records) if records else pd.DataFrame()


def summarize_fetches(fetches: list[FetchResult]) -> dict[str, Any]:
    by_source: dict[str, dict[str, Any]] = {}
    for item in fetches:
        bucket = by_source.setdefault(item.source, {"requested": 0, "ok": 0, "bytes": 0, "sum_elapsed_seconds": 0.0})
        bucket["requested"] += 1
        bucket["ok"] += int(item.status == "ok")
        bucket["bytes"] += item.bytes
        bucket["sum_elapsed_seconds"] += item.elapsed_seconds
    return by_source


def day_coverage(fetch_df: pd.DataFrame) -> list[dict[str, Any]]:
    if fetch_df.empty:
        return []
    rows: list[dict[str, Any]] = []
    for keys, group in fetch_df.groupby(["kind", "source", "issue_day_utc"], dropna=False):
        rows.append(
            {
                "kind": keys[0],
                "source": keys[1],
                "day_utc": keys[2],
                "requested": int(len(group)),
                "ok": int((group["status"] == "ok").sum()),
                "bytes": int(group["bytes"].sum()),
                "missing_or_error": int((group["status"] != "ok").sum()),
            }
        )
    return rows


def object_to_record(item: Any) -> dict[str, Any]:
    return asdict(item) if hasattr(item, "__dataclass_fields__") else dict(item)


def compute_backfill_estimates(
    days: list[date],
    fetches: list[FetchResult],
    normalized_size_bytes: int,
    raw_size_bytes: int,
    probe: dict[str, Any],
) -> dict[str, Any]:
    day_count = len(days)
    fetched_bytes = sum(item.bytes for item in fetches if item.status == "ok")
    per_day_raw = fetched_bytes / day_count if day_count else 0.0
    per_day_normalized = normalized_size_bytes / day_count if day_count else 0.0
    history_2015_start = date(2015, 7, 7)
    history_2017_start = date(2017, 1, 1)
    end = max(days)
    days_2015 = max(0, (end - history_2015_start).days + 1)
    days_2017 = max(0, (end - history_2017_start).days + 1)

    selected_himawari_bytes = sum(item.bytes for item in fetches if item.kind == "himawari_hsd" and item.status == "ok")
    himawari_per_day = selected_himawari_bytes / day_count if day_count else 0.0
    all_bands_ratio = probe.get("all_bands_selected_segment_to_selected_band_ratio") or 16.0
    full_disk_ratio = probe.get("full_disk_all_bands_to_selected_band_segment_ratio") or 160.0
    return {
        "profile": {
            "actual_days": day_count,
            "actual_date_min": min(days).isoformat(),
            "actual_date_max": max(days).isoformat(),
            "raw_size_on_disk_bytes": int(raw_size_bytes),
            "normalized_size_on_disk_bytes": int(normalized_size_bytes),
            "downloaded_bytes_manifest_sum": int(fetched_bytes),
            "raw_bytes_per_day_observed": per_day_raw,
            "normalized_bytes_per_day_observed": per_day_normalized,
        },
        "history_day_counts": {
            "2015_07_07_to_run_end": days_2015,
            "2017_01_01_to_run_end": days_2017,
        },
        "actual_profile_estimates": {
            "raw_gb_2015_07_07_to_run_end": per_day_raw * days_2015 / 1_000_000_000,
            "raw_gb_2017_01_01_to_run_end": per_day_raw * days_2017 / 1_000_000_000,
            "normalized_gb_2015_07_07_to_run_end": per_day_normalized * days_2015 / 1_000_000_000,
            "normalized_gb_2017_01_01_to_run_end": per_day_normalized * days_2017 / 1_000_000_000,
        },
        "himawari_scale_estimates_from_probe": {
            "selected_b13_s0510_raw_gb_2015_07_07_to_run_end": himawari_per_day * days_2015 / 1_000_000_000,
            "selected_b13_s0510_raw_gb_2017_01_01_to_run_end": himawari_per_day * days_2017 / 1_000_000_000,
            "all_bands_s0510_raw_gb_2015_07_07_to_run_end": himawari_per_day * all_bands_ratio * days_2015 / 1_000_000_000,
            "all_bands_s0510_raw_gb_2017_01_01_to_run_end": himawari_per_day * all_bands_ratio * days_2017 / 1_000_000_000,
            "full_disk_all_bands_raw_gb_2015_07_07_to_run_end": himawari_per_day * full_disk_ratio * days_2015 / 1_000_000_000,
            "full_disk_all_bands_raw_gb_2017_01_01_to_run_end": himawari_per_day * full_disk_ratio * days_2017 / 1_000_000_000,
            "all_bands_s0510_to_b13_s0510_ratio": all_bands_ratio,
            "full_disk_all_bands_to_b13_s0510_ratio": full_disk_ratio,
        },
    }


def write_outputs(
    args: argparse.Namespace,
    days: list[date],
    leads: list[int],
    model_tasks: list[FetchTask],
    himawari_tasks: list[FetchTask],
    fetches: list[FetchResult],
    idx_rows: list[dict[str, Any]],
    model_norms: list[dict[str, Any]],
    himawari_norms: list[dict[str, Any]],
    timings: dict[str, Any],
    probe: dict[str, Any],
) -> dict[str, Any]:
    ensure_dir(NORMALIZED_DIR)
    ensure_dir(METADATA_DIR)

    fetch_records = [asdict(item) for item in fetches]
    fetch_df = dataframe(fetch_records)
    idx_df = dataframe(idx_rows)
    model_station_df = dataframe([item["station_row"] for item in model_norms if item.get("station_row")])
    model_bbox_df = dataframe([row for item in model_norms for row in item.get("bbox_rows", [])])
    himawari_df = dataframe([item["row"] for item in himawari_norms if item.get("row")])
    model_norm_status_df = dataframe(
        [
            {
                **item.get("fetch", {}),
                "normalization_status": item.get("status"),
                "normalization_error": (item.get("station_row") or {}).get("normalization_error"),
            }
            for item in model_norms
        ]
    )
    himawari_norm_status_df = dataframe(
        [
            {
                **item.get("fetch", {}),
                "normalization_status": item.get("status"),
                "normalization_error": (item.get("row") or {}).get("normalization_error"),
            }
            for item in himawari_norms
        ]
    )

    fetch_df.to_csv(wp(NORMALIZED_DIR / "fetch_manifest.csv"), index=False)
    idx_df.to_csv(wp(NORMALIZED_DIR / "model_idx_catalog.csv"), index=False)
    model_station_df.to_csv(wp(NORMALIZED_DIR / "model_cycle_lead_station_features.csv"), index=False)
    model_bbox_df.to_csv(wp(NORMALIZED_DIR / "model_cycle_lead_bbox_summary_features.csv"), index=False)
    himawari_df.to_csv(wp(NORMALIZED_DIR / "himawari_b13_s0510_scan_features.csv"), index=False)
    model_norm_status_df.to_csv(wp(NORMALIZED_DIR / "model_normalization_status.csv"), index=False)
    himawari_norm_status_df.to_csv(wp(NORMALIZED_DIR / "himawari_normalization_status.csv"), index=False)

    for frame, name in [
        (fetch_df, "fetch_manifest.parquet"),
        (idx_df, "model_idx_catalog.parquet"),
        (model_station_df, "model_cycle_lead_station_features.parquet"),
        (model_bbox_df, "model_cycle_lead_bbox_summary_features.parquet"),
        (himawari_df, "himawari_b13_s0510_scan_features.parquet"),
    ]:
        if not frame.empty:
            frame.to_parquet(wp(NORMALIZED_DIR / name), index=False)

    write_json(METADATA_DIR / "himawari_representative_scan_size_probe.json", probe)

    raw_size = file_size(RAW_DIR)
    normalized_size = file_size(NORMALIZED_DIR)
    estimates = compute_backfill_estimates(days, fetches, normalized_size, raw_size, probe)
    write_json(NORMALIZED_DIR / "backfill_size_estimates.json", estimates)

    expected = {
        "model_grib_items": len(model_tasks),
        "himawari_hsd_items": len(himawari_tasks),
        "total_fetch_items": len(model_tasks) + len(himawari_tasks),
        "model_idx_items": len(idx_rows),
        "days": len(days),
        "model_cycles_per_day": len(MODEL_CYCLES),
        "model_leads_per_cycle": len(leads),
        "himawari_scans_per_day_per_band_segment": 144,
    }
    fetch_summary = summarize_fetches(fetches)
    sanity = {
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": utc_now_iso(),
        "date_range_utc": {"start": min(days).isoformat(), "end": max(days).isoformat(), "days": len(days)},
        "scope": {
            "gfs": "NOMADS filtered HKG bbox, cycles 00/06/12/18, common leads f000..f048 every 3h",
            "gefs_control": "NOMADS filtered GEFS control HKG bbox, cycles 00/06/12/18, common leads f000..f048 every 3h",
            "himawari9": f"B13 infrared HKG segment {args.himawari_segment}, every 10 minutes",
            "leads": leads,
            "himawari_bands": args.himawari_bands,
        },
        "expected_counts": expected,
        "fetch_summary_by_source": fetch_summary,
        "day_coverage": day_coverage(fetch_df),
        "attribute_counts": {
            "gfs_idx_message_count_min": int(idx_df[idx_df["source"] == "gfs"]["message_count"].dropna().min())
            if not idx_df.empty and not idx_df[idx_df["source"] == "gfs"]["message_count"].dropna().empty
            else None,
            "gfs_idx_message_count_max": int(idx_df[idx_df["source"] == "gfs"]["message_count"].dropna().max())
            if not idx_df.empty and not idx_df[idx_df["source"] == "gfs"]["message_count"].dropna().empty
            else None,
            "gefs_idx_message_count_min": int(idx_df[idx_df["source"] == "gefs_control"]["message_count"].dropna().min())
            if not idx_df.empty and not idx_df[idx_df["source"] == "gefs_control"]["message_count"].dropna().empty
            else None,
            "gefs_idx_message_count_max": int(idx_df[idx_df["source"] == "gefs_control"]["message_count"].dropna().max())
            if not idx_df.empty and not idx_df[idx_df["source"] == "gefs_control"]["message_count"].dropna().empty
            else None,
            "model_station_feature_columns": int(len(model_station_df.columns)),
            "model_bbox_summary_columns": int(len(model_bbox_df.columns)),
            "himawari_scan_feature_columns": int(len(himawari_df.columns)),
            "himawari_raw_files_per_full_disk_scan": 160,
            "himawari_raw_files_per_hko_segment_scan_all_bands": 16,
            "himawari_selected_b13_segment_files_per_day": 144,
        },
        "normalized_rows": {
            "model_station_rows": int(len(model_station_df)),
            "model_bbox_variable_rows": int(len(model_bbox_df)),
            "himawari_scan_rows": int(len(himawari_df)),
            "model_normalization_errors": int((model_norm_status_df.get("normalization_status", pd.Series(dtype=str)) == "error").sum())
            if not model_norm_status_df.empty
            else 0,
            "himawari_normalization_errors": int((himawari_norm_status_df.get("normalization_status", pd.Series(dtype=str)) == "error").sum())
            if not himawari_norm_status_df.empty
            else 0,
        },
        "leakage_safety": {
            "model_availability_contract": f"issued_at_utc + {MODEL_AVAILABILITY_BUFFER_HOURS}h",
            "himawari_availability_contract": "max(hsd_file_creation_utc, observed_at_utc + 30m)",
            "fetch_rows_missing_availability_proxy": int(fetch_df["availability_proxy_utc"].isna().sum())
            if not fetch_df.empty and "availability_proxy_utc" in fetch_df
            else None,
            "model_rows_missing_grib_issued_at": int(model_station_df["issued_at_utc_grib"].isna().sum())
            if not model_station_df.empty and "issued_at_utc_grib" in model_station_df
            else None,
            "himawari_rows_missing_observed_at": int(himawari_df["observed_at_utc"].isna().sum())
            if not himawari_df.empty and "observed_at_utc" in himawari_df
            else None,
            "himawari_rows_missing_file_creation": int(himawari_df["file_creation_utc"].isna().sum())
            if not himawari_df.empty and "file_creation_utc" in himawari_df
            else None,
        },
        "timing_seconds": timings,
        "bytes": {
            "downloaded_manifest_sum": int(fetch_df["bytes"].sum()) if not fetch_df.empty else 0,
            "raw_size_on_disk": int(raw_size),
            "normalized_size_on_disk": int(normalized_size),
        },
        "outputs": {
            "fetch_manifest": "normalized/fetch_manifest.csv",
            "model_idx_catalog": "normalized/model_idx_catalog.csv",
            "model_station_features": "normalized/model_cycle_lead_station_features.csv",
            "model_bbox_summary_features": "normalized/model_cycle_lead_bbox_summary_features.csv",
            "himawari_scan_features": "normalized/himawari_b13_s0510_scan_features.csv",
            "model_normalization_status": "normalized/model_normalization_status.csv",
            "himawari_normalization_status": "normalized/himawari_normalization_status.csv",
            "backfill_size_estimates": "normalized/backfill_size_estimates.json",
        },
    }
    write_json(NORMALIZED_DIR / "sanity_report.json", sanity)

    status = "COMPLETE"
    if any(item.status != "ok" for item in fetches) or sanity["normalized_rows"]["model_normalization_errors"] or sanity["normalized_rows"]["himawari_normalization_errors"]:
        status = "COMPLETE_WITH_GAPS"
    write_text(
        EXPERIMENT_DIR / "STATUS.yaml",
        f"state: {status}\n"
        "gate_result: PUBLIC_7DAY_BACKFILL_REHEARSAL_DONE\n"
        "uses_gribstream: false\n"
        f"date_start_utc: {min(days).isoformat()}\n"
        f"date_end_utc: {max(days).isoformat()}\n",
    )

    readme = f"""# Public 7-Day GFS/GEFS/Himawari Backfill Rehearsal

Generated: `{sanity["generated_at_utc"]}`

Date range: `{min(days).isoformat()}` through `{max(days).isoformat()}` UTC.

This experiment rehearses the future public-source backfill path:

- GFS: 00/06/12/18Z cycles, HKG bbox, selected weather feature pack, common leads `{leads[0]}`..`{leads[-1]}` every 3h.
- GEFS control: same cycle/lead grid as GFS.
- Himawari-9: B13 HKG segment `{args.himawari_segment}`, all 144 ten-minute scans/day.

## Headline

| Metric | Value |
|---|---:|
| fetch items requested | {expected["total_fetch_items"]:,} |
| fetch items ok | {sum(1 for item in fetches if item.status == "ok"):,} |
| model GRIB requested | {expected["model_grib_items"]:,} |
| Himawari HSD requested | {expected["himawari_hsd_items"]:,} |
| model station normalized rows | {len(model_station_df):,} |
| model bbox variable rows | {len(model_bbox_df):,} |
| Himawari scan normalized rows | {len(himawari_df):,} |
| downloaded bytes | {sanity["bytes"]["downloaded_manifest_sum"]:,} |
| raw size on disk GB | {raw_size / 1_000_000_000:.3f} |
| normalized size on disk GB | {normalized_size / 1_000_000_000:.3f} |
| wall seconds total | {timings["total_wall_seconds"]:.2f} |

## Leakage Contract

For GFS/GEFS, `issued_at_utc` is the model cycle time and `availability_proxy_utc`
is `issued_at_utc + {MODEL_AVAILABILITY_BUFFER_HOURS}h`.

For Himawari, `observed_at_utc` comes from the HSD file name/header and
`availability_proxy_utc` is `max(hsd_file_creation_utc, observed_at_utc + 30m)`.

Both raw fetch manifests and normalized rows carry the issue/observed timestamp,
the availability proxy, raw path, URL, byte count, and SHA256.

## Main Files

| File | Purpose |
|---|---|
| `normalized/sanity_report.json` | Coverage, attribute counts, leakage fields, timing, bytes. |
| `normalized/fetch_manifest.csv` | One row per requested raw object with URL, status, bytes, hash, timestamps. |
| `normalized/model_idx_catalog.csv` | Full-product index counts for every requested model source/cycle/lead. |
| `normalized/model_cycle_lead_station_features.csv` | HKO nearest-grid model features by source/cycle/lead. |
| `normalized/model_cycle_lead_bbox_summary_features.csv` | HKG bbox min/mean/median/max/std by variable. |
| `normalized/himawari_b13_s0510_scan_features.csv` | One normalized row per B13 HKG-segment scan. |
| `normalized/backfill_size_estimates.json` | Estimated raw/normalized size for 2015+ and 2017+ backfills. |
"""
    write_text(EXPERIMENT_DIR / "README.md", readme)
    return sanity


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a 7-day public GFS/GEFS/Himawari backfill rehearsal.")
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--end-date", type=parse_date, default=completed_utc_yesterday())
    parser.add_argument("--lead-hours", type=parse_leads, default=DEFAULT_MODEL_LEADS)
    parser.add_argument("--himawari-bands", default="B13")
    parser.add_argument("--himawari-segment", default="S0510")
    parser.add_argument("--download-workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--idx-workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--model-normalize-workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--himawari-normalize-workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--skip-normalize", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if not args.execute:
        print("DRY RUN: no requests made; pass --execute with a reviewed 1..7 day scope.")
        return 2
    if args.days < 1 or args.days > 7:
        parser.error("--days must be between 1 and 7")

    total_started = time.perf_counter()
    days = date_span(args.end_date, args.days)
    leads = args.lead_hours if isinstance(args.lead_hours, list) else parse_leads(args.lead_hours)
    args.himawari_bands = [part.strip().upper() for part in args.himawari_bands.split(",") if part.strip()]
    ensure_dir(RAW_DIR)
    ensure_dir(NORMALIZED_DIR)
    ensure_dir(METADATA_DIR)

    model_tasks = build_model_tasks(days, leads)
    himawari_tasks = build_himawari_tasks(days, args.himawari_bands, args.himawari_segment)
    all_tasks = model_tasks + himawari_tasks
    write_json(
        METADATA_DIR / "run_config.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "generated_at_utc": utc_now_iso(),
            "days": [item.isoformat() for item in days],
            "model_leads": leads,
            "model_cycles": MODEL_CYCLES,
            "himawari_bands": args.himawari_bands,
            "himawari_segment": args.himawari_segment,
            "download_workers": args.download_workers,
            "idx_workers": args.idx_workers,
            "model_normalize_workers": args.model_normalize_workers,
            "himawari_normalize_workers": args.himawari_normalize_workers,
            "model_tasks": len(model_tasks),
            "himawari_tasks": len(himawari_tasks),
        },
    )

    print(f"Experiment: {EXPERIMENT_DIR}", flush=True)
    print(f"Fetching {len(all_tasks)} raw objects for {min(days)}..{max(days)}", flush=True)
    download_started = time.perf_counter()
    fetches = [
        FetchResult(**record)
        if isinstance(record, dict) and "kind" in record and "status" in record
        else record
        for record in run_pool("download", all_tasks, fetch_to_path, args.download_workers, progress_every=100)
    ]
    download_wall = time.perf_counter() - download_started

    idx_started = time.perf_counter()
    idx_rows = fetch_idx_catalog(days, leads, args.idx_workers)
    idx_wall = time.perf_counter() - idx_started

    model_fetches = [item for item in fetches if item.kind == "model_grib"]
    himawari_fetches = [item for item in fetches if item.kind == "himawari_hsd"]
    model_norms: list[dict[str, Any]] = []
    himawari_norms: list[dict[str, Any]] = []
    model_norm_wall = 0.0
    himawari_norm_wall = 0.0
    if not args.skip_normalize:
        model_norm_started = time.perf_counter()
        model_norms = run_pool(
            "model-normalize",
            model_fetches,
            normalize_model_result,
            args.model_normalize_workers,
            progress_every=100,
            use_processes=True,
        )
        model_norm_wall = time.perf_counter() - model_norm_started

        himawari_norm_started = time.perf_counter()
        himawari_norms = run_pool(
            "himawari-normalize",
            himawari_fetches,
            normalize_himawari_result,
            args.himawari_normalize_workers,
            progress_every=100,
        )
        himawari_norm_wall = time.perf_counter() - himawari_norm_started

    representative_scan = datetime.combine(days[-1], dt_time(6, 0), tzinfo=timezone.utc)
    probe = probe_himawari_scan_size(representative_scan, args.himawari_segment, "B13")
    timings = {
        "total_wall_seconds": time.perf_counter() - total_started,
        "download_wall_seconds": download_wall,
        "idx_wall_seconds": idx_wall,
        "model_normalize_wall_seconds": model_norm_wall,
        "himawari_normalize_wall_seconds": himawari_norm_wall,
        "download_sum_elapsed_seconds": float(sum(item.elapsed_seconds for item in fetches)),
        "model_download_sum_elapsed_seconds": float(sum(item.elapsed_seconds for item in model_fetches)),
        "himawari_download_sum_elapsed_seconds": float(sum(item.elapsed_seconds for item in himawari_fetches)),
        "parallel_download_speedup_estimate": float(sum(item.elapsed_seconds for item in fetches) / download_wall)
        if download_wall
        else None,
    }
    sanity = write_outputs(
        args,
        days,
        leads,
        model_tasks,
        himawari_tasks,
        fetches,
        idx_rows,
        model_norms,
        himawari_norms,
        timings,
        probe,
    )
    print(json.dumps(sanity["normalized_rows"], indent=2, sort_keys=True), flush=True)
    print(EXPERIMENT_DIR, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
