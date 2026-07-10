from __future__ import annotations

import argparse
import bz2
import hashlib
import io
import json
import math
import os
import re
import shutil
import statistics
import sys
import threading
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from contextlib import suppress
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta, timezone
from datetime import time as dt_time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urljoin, urlparse
from urllib.request import Request, urlopen

import cfgrib
import httpx
import numpy as np
import pandas as pd
import psycopg
from PIL import Image
from psycopg.types.json import Jsonb

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_public_gfs_gefs_himawari_7day_backfill_rehearsal as public  # noqa: E402

EXPERIMENT_ID = "0009_public_weather_backfill_jun25_jul7_lean_db_20260708"
DEFAULT_EXPERIMENT_DIR = REPO_ROOT / "experiments" / "campaigns" / "hkg-tmax" / EXPERIMENT_ID
USER_AGENT = "weather-markets-hkg-public-weather-backfill-db/1.0"
HKT = timezone(timedelta(hours=8))
S3_RANGE_WORKERS = 4
README_CONTRACT_START = "<!-- BEGIN GENERATED BACKFILL CONTRACT -->"
README_CONTRACT_END = "<!-- END GENERATED BACKFILL CONTRACT -->"
README_RESULTS_START = "<!-- BEGIN GENERATED BACKFILL LATEST RUN -->"
README_RESULTS_END = "<!-- END GENERATED BACKFILL LATEST RUN -->"

MODEL_SELECTOR_LEVELS: dict[str, tuple[str, ...]] = {
    "TMP": ("2 m above ground",),
    "DPT": ("2 m above ground",),
    "RH": ("2 m above ground",),
    "TMAX": ("2 m above ground",),
    "TMIN": ("2 m above ground",),
    "UGRD": ("10 m above ground",),
    "VGRD": ("10 m above ground",),
    "PRMSL": ("mean sea level",),
    "GUST": ("surface",),
    "APCP": ("surface",),
    "DSWRF": ("surface",),
    "CAPE": ("surface",),
    "CIN": ("surface",),
    "PWAT": ("entire atmosphere",),
    "TCDC": ("entire atmosphere",),
}

MODEL_STATION_META_KEYS = {
    "source",
    "item_id",
    "issue_day_utc",
    "cycle_hour",
    "lead_hour_requested",
    "issued_at_utc",
    "issued_at_utc_grib",
    "valid_at_utc_expected",
    "valid_at_utc_grib",
    "lead_hour_grib",
    "availability_proxy_utc",
    "availability_proxy_method",
    "http_last_modified_utc",
    "raw_path",
    "raw_bytes",
    "raw_sha256",
    "station_id",
    "station_name",
    "station_latitude",
    "station_longitude",
    "normalized_variable_count",
    "normalization_elapsed_seconds",
    "normalization_error",
}

HIMAWARI_META_KEYS = {
    "source",
    "item_id",
    "band",
    "segment",
    "observed_at_utc",
    "observed_date_hkt",
    "availability_proxy_utc",
    "availability_proxy_method",
    "t24_next_target_date_hkt",
    "t24_next_day_cutoff_utc",
    "eligible_for_next_day_t24_cutoff",
    "file_creation_utc",
    "http_last_modified_utc",
    "file_name",
    "raw_path",
    "raw_bytes",
    "raw_sha256",
    "normalization_elapsed_seconds",
    "normalization_error",
}

RADAR_META_KEYS = {
    "source",
    "provider",
    "product",
    "frame_time_hkt",
    "observed_at_utc",
    "availability_proxy_utc",
    "availability_proxy_method",
    "native_issue_metadata_status",
    "envf_query_url",
    "display_large_image_url",
    "display_image_url",
    "envf_temp_image_path",
    "frame_index_in_chunk",
    "image_fetch_status",
    "image_fetch_error",
    "normalization_elapsed_seconds",
}


def wp(path: Path) -> str:
    return public.wp(path)


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso(value: datetime | pd.Timestamp | None) -> str | None:
    return public.iso(value)


def parse_iso(value: str | None) -> datetime | None:
    return public.parse_iso(value)


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def parse_csv_ints(value: str) -> list[int]:
    out = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not out:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return out


def parse_leads(value: str) -> list[int]:
    text = value.strip().lower()
    if text in {"default", "0:48:3", "common_3h_0_48"}:
        return list(range(0, 49, 3))
    if ":" in text:
        parts = [int(part) for part in text.split(":")]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("Lead range must be start:end:step.")
        start, end, step = parts
        return list(range(start, end + 1, step))
    return parse_csv_ints(value)


def normalize_sources(value: str) -> set[str]:
    aliases = {
        "gfs": "gfs",
        "gefs": "gefs_control",
        "gefs_control": "gefs_control",
        "himawari": "himawari_b13_s0510",
        "himawari_b13": "himawari_b13_s0510",
        "himawari_b13_s0510": "himawari_b13_s0510",
        "radar": "radar",
        "envf_radar": "radar",
    }
    sources: set[str] = set()
    for part in value.split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise argparse.ArgumentTypeError(f"Unknown source '{part}'.")
        sources.add(aliases[key])
    if not sources:
        raise argparse.ArgumentTypeError("At least one source is required.")
    return sources


def date_span(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("end date must be on or after start date")
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def ensure_dir(path: Path) -> None:
    Path(wp(path)).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    Path(wp(path)).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    Path(wp(path)).write_text(text, encoding="utf-8")


def upsert_readme_section(
    path: Path,
    *,
    title: str,
    start_marker: str,
    end_marker: str,
    content: str,
) -> None:
    """Insert or replace one generated README section without touching manual prose."""
    path_s = Path(wp(path))
    existing = path_s.read_text(encoding="utf-8") if path_s.exists() else f"# {title}\n"
    start_count = existing.count(start_marker)
    end_count = existing.count(end_marker)
    if start_count != end_count or start_count > 1:
        raise RuntimeError(f"Malformed generated README section in {path}")

    block = f"{start_marker}\n{content.strip()}\n{end_marker}"
    if start_count == 1:
        start_index = existing.index(start_marker)
        end_index = existing.index(end_marker, start_index) + len(end_marker)
        prefix = existing[:start_index].rstrip()
        suffix = existing[end_index:].lstrip("\r\n").rstrip()
    else:
        prefix = existing.rstrip()
        suffix = ""

    parts = [part for part in (prefix, block, suffix) if part]
    updated = "\n\n".join(parts) + "\n"
    if updated != existing:
        write_text(path, updated)


def file_size(path: Path) -> int:
    path_s = wp(path)
    if not os.path.exists(path_s):
        return 0
    if os.path.isfile(path_s):
        return os.path.getsize(path_s)
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path_s):
        for name in filenames:
            try:
                total += os.stat(os.path.join(dirpath, name)).st_size
            except FileNotFoundError:
                continue
    return total


def drive_free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(str(path.anchor or path))
    return int(usage.free)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def http_get_bytes(
    url: str,
    *,
    timeout: int,
    headers: dict[str, str] | None = None,
) -> tuple[bytes, dict[str, str], int]:
    req_headers = {"User-Agent": USER_AGENT}
    if headers:
        req_headers.update(headers)
    req = Request(url, headers=req_headers)
    with urlopen(req, timeout=timeout) as response:
        return (
            response.read(),
            {key.lower(): value for key, value in response.headers.items()},
            int(getattr(response, "status", 200)),
        )


def http_head_content_length(url: str, *, timeout: int) -> int:
    req = Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as response:
        length = response.headers.get("content-length")
    if not length:
        raise RuntimeError(f"HEAD did not return content-length for {url}")
    return int(length)


def should_retry_error(exc: BaseException) -> bool:
    if isinstance(exc, HTTPError):
        return exc.code in {408, 429, 500, 502, 503, 504}
    return isinstance(exc, (TimeoutError, URLError, httpx.HTTPError))


def retry_sleep(attempt: int) -> None:
    time.sleep(min(20.0, 1.5 * attempt + 0.25 * (attempt % 3)))


def request_with_retries(
    label: str,
    fetch_fn: Any,
    *,
    max_attempts: int,
) -> tuple[Any, int]:
    attempts = 0
    last_error: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        try:
            return fetch_fn(), attempts
        except BaseException as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= max_attempts or not should_retry_error(exc):
                break
            print(f"[retry] {label} attempt={attempt} error={type(exc).__name__}: {exc}", flush=True)
            retry_sleep(attempt)
    assert last_error is not None
    raise last_error


def model_object_url(source: str, day: date, cycle_hour: int, lead_hour: int) -> str:
    cc = f"{cycle_hour:02d}"
    if source == "gfs":
        return (
            "https://noaa-gfs-bdp-pds.s3.amazonaws.com/"
            f"gfs.{day:%Y%m%d}/{cc}/atmos/gfs.t{cc}z.pgrb2.0p25.f{lead_hour:03d}"
        )
    if source == "gefs_control":
        return (
            "https://noaa-gefs-pds.s3.amazonaws.com/"
            f"gefs.{day:%Y%m%d}/{cc}/atmos/pgrb2sp25/gec00.t{cc}z.pgrb2s.0p25.f{lead_hour:03d}"
        )
    raise ValueError(f"Unsupported model source: {source}")


def model_idx_url(source: str, day: date, cycle_hour: int, lead_hour: int) -> str:
    return model_object_url(source, day, cycle_hour, lead_hour) + ".idx"


def model_message_selected(variable: str, level: str) -> bool:
    wanted_levels = MODEL_SELECTOR_LEVELS.get(variable.upper())
    if not wanted_levels:
        return False
    level_lower = level.lower()
    return any(wanted.lower() in level_lower for wanted in wanted_levels)


def parse_grib_idx_ranges(text: str, object_length: int) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    lines = [line for line in text.splitlines() if line.strip()]
    for line in lines:
        parts = line.split(":")
        if len(parts) < 5:
            continue
        try:
            offset = int(parts[1])
        except ValueError:
            continue
        parsed.append(
            {
                "line": line,
                "message_no": int(parts[0]) if parts[0].isdigit() else None,
                "offset": offset,
                "variable": parts[3],
                "level": parts[4],
                "forecast": parts[5] if len(parts) > 5 else None,
            }
        )
    for index, item in enumerate(parsed):
        if index + 1 < len(parsed):
            item["end_offset"] = int(parsed[index + 1]["offset"]) - 1
        else:
            item["end_offset"] = object_length - 1
    return [item for item in parsed if model_message_selected(item["variable"], item["level"])]


def merge_selected_ranges(ranges: list[dict[str, Any]], max_gap_bytes: int) -> list[dict[str, Any]]:
    """Merge selected GRIB byte ranges without changing their order.

    With max_gap_bytes=0 this only merges directly adjacent selected messages, so the
    downloaded byte stream is identical to the existing per-message path.
    """
    if max_gap_bytes < 0:
        raise ValueError("max_gap_bytes must be >= 0")
    if not ranges:
        return []
    ordered = sorted(ranges, key=lambda item: int(item["offset"]))
    current: dict[str, Any] = {
        "offset": int(ordered[0]["offset"]),
        "end_offset": int(ordered[0]["end_offset"]),
        "message_count": 1,
    }
    merged: list[dict[str, Any]] = []
    for item in ordered[1:]:
        offset = int(item["offset"])
        end_offset = int(item["end_offset"])
        gap = offset - int(current["end_offset"]) - 1
        if gap <= max_gap_bytes:
            current["end_offset"] = max(int(current["end_offset"]), end_offset)
            current["message_count"] = int(current["message_count"]) + 1
            current["downloaded_gap_bytes"] = int(current.get("downloaded_gap_bytes", 0)) + max(gap, 0)
        else:
            merged.append(current)
            current = {"offset": offset, "end_offset": end_offset, "message_count": 1}
    merged.append(current)
    return merged


def write_payload(path: Path, data: bytes) -> None:
    ensure_dir(path.parent)
    Path(wp(path)).write_bytes(data)


def fetch_nomads_model(
    task: public.FetchTask,
    full_path: Path,
    *,
    timeout: int,
) -> tuple[public.FetchResult, dict[str, Any]]:
    data, headers, _status = http_get_bytes(task.url, timeout=timeout)
    invalid = public.payload_validation_error(task, data)
    if invalid:
        return (
            public.FetchResult(
                kind=task.kind,
                source=task.source,
                item_id=task.item_id,
                status="invalid_payload",
                url=task.url,
                path=None,
                bytes=len(data),
                sha256=sha256_bytes(data),
                elapsed_seconds=0.0,
                retrieved_at_utc=iso(utc_now()) or "",
                fetched_now=True,
                issue_day_utc=task.issue_day_utc,
                cycle_hour=task.cycle_hour,
                lead_hour=task.lead_hour,
                issued_at_utc=task.issued_at_utc,
                valid_at_utc=task.valid_at_utc,
                availability_proxy_utc=task.availability_proxy_utc,
                availability_proxy_method=task.availability_proxy_method,
                http_last_modified_utc=public.parse_http_datetime(headers.get("last-modified")),
                content_length_header=int(headers["content-length"]) if headers.get("content-length", "").isdigit() else None,
                content_type=headers.get("content-type"),
                error=invalid,
            ),
            {"provider_mode": "nomads_filter", "validation_error": invalid},
        )
    write_payload(full_path, data)
    return (
        public.FetchResult(
            kind=task.kind,
            source=task.source,
            item_id=task.item_id,
            status="ok",
            url=task.url,
            path=task.path,
            bytes=len(data),
            sha256=sha256_bytes(data),
            elapsed_seconds=0.0,
            retrieved_at_utc=iso(utc_now()) or "",
            fetched_now=True,
            issue_day_utc=task.issue_day_utc,
            cycle_hour=task.cycle_hour,
            lead_hour=task.lead_hour,
            issued_at_utc=task.issued_at_utc,
            valid_at_utc=task.valid_at_utc,
            availability_proxy_utc=task.availability_proxy_utc,
            availability_proxy_method=task.availability_proxy_method,
            http_last_modified_utc=public.parse_http_datetime(headers.get("last-modified")),
            content_length_header=int(headers["content-length"]) if headers.get("content-length", "").isdigit() else None,
            content_type=headers.get("content-type"),
            error=None,
        ),
        {"provider_mode": "nomads_filter"},
    )


def fetch_s3_idx_range_model(
    task: public.FetchTask,
    full_path: Path,
    *,
    timeout: int,
    range_workers: int | None = None,
    coalesce_gap_bytes: int | None = None,
) -> tuple[public.FetchResult, dict[str, Any]]:
    if task.issue_day_utc is None or task.cycle_hour is None or task.lead_hour is None:
        raise ValueError("Model task is missing issue day, cycle, or lead")
    day = parse_date(task.issue_day_utc)
    object_url = model_object_url(task.source, day, task.cycle_hour, task.lead_hour)
    idx_url = model_idx_url(task.source, day, task.cycle_hour, task.lead_hour)
    object_length = http_head_content_length(object_url, timeout=timeout)
    idx_data, idx_headers, _status = http_get_bytes(idx_url, timeout=timeout)
    idx_text = idx_data.decode("utf-8", errors="replace")
    ranges = parse_grib_idx_ranges(idx_text, object_length)
    if not ranges:
        return (
            public.FetchResult(
                kind=task.kind,
                source=task.source,
                item_id=task.item_id,
                status="missing_selected_messages",
                url=object_url,
                path=None,
                bytes=0,
                sha256=None,
                elapsed_seconds=0.0,
                retrieved_at_utc=iso(utc_now()) or "",
                fetched_now=True,
                issue_day_utc=task.issue_day_utc,
                cycle_hour=task.cycle_hour,
                lead_hour=task.lead_hour,
                issued_at_utc=task.issued_at_utc,
                valid_at_utc=task.valid_at_utc,
                availability_proxy_utc=task.availability_proxy_utc,
                availability_proxy_method=task.availability_proxy_method,
                http_last_modified_utc=public.parse_http_datetime(idx_headers.get("last-modified")),
                content_length_header=object_length,
                content_type=None,
                error="No selected variable/level messages found in GRIB index.",
            ),
            {
                "provider_mode": "s3_idx_range",
                "idx_url": idx_url,
                "object_url": object_url,
                "object_length": object_length,
                "selected_message_count": 0,
                "selected_variables": [],
                "model_range_coalesce_gap_bytes": coalesce_gap_bytes,
            },
        )
    ensure_dir(full_path.parent)
    selected_variables: list[str] = [str(item["variable"]) for item in ranges]
    selected_level_pairs: list[str] = [f"{item['variable']}:{item['level']}" for item in ranges]
    selected_bytes = sum(int(item["end_offset"]) - int(item["offset"]) + 1 for item in ranges)
    transfer_ranges = merge_selected_ranges(ranges, coalesce_gap_bytes) if coalesce_gap_bytes is not None else ranges
    downloaded_bytes_expected = sum(
        int(item["end_offset"]) - int(item["offset"]) + 1 for item in transfer_ranges
    )

    def fetch_selected_range(index_and_item: tuple[int, dict[str, Any]]) -> tuple[int, bytes]:
        index, item = index_and_item
        byte_range = f"bytes={item['offset']}-{item['end_offset']}"
        chunk, _headers, _status = http_get_bytes(
            object_url,
            timeout=timeout,
            headers={"Range": byte_range},
        )
        if not chunk.startswith(b"GRIB"):
            raise RuntimeError(f"S3 range did not start with GRIB for {task.item_id}: {byte_range}")
        return index, chunk

    chunks: list[bytes | None] = [None] * len(transfer_ranges)
    effective_range_workers = max(1, min(range_workers or S3_RANGE_WORKERS, len(transfer_ranges)))
    with ThreadPoolExecutor(max_workers=effective_range_workers) as executor:
        futures = [executor.submit(fetch_selected_range, item) for item in enumerate(transfer_ranges)]
        for future in as_completed(futures):
            index, chunk = future.result()
            chunks[index] = chunk

    total_bytes = 0
    with open(wp(full_path), "wb") as handle:
        for chunk in chunks:
            if chunk is None:
                raise RuntimeError(f"Missing selected S3 range for {task.item_id}")
            handle.write(chunk)
            total_bytes += len(chunk)
    data = Path(wp(full_path)).read_bytes()
    invalid = public.payload_validation_error(task, data)
    if invalid:
        return (
            public.FetchResult(
                kind=task.kind,
                source=task.source,
                item_id=task.item_id,
                status="invalid_payload",
                url=object_url,
                path=task.path,
                bytes=len(data),
                sha256=sha256_bytes(data),
                elapsed_seconds=0.0,
                retrieved_at_utc=iso(utc_now()) or "",
                fetched_now=True,
                issue_day_utc=task.issue_day_utc,
                cycle_hour=task.cycle_hour,
                lead_hour=task.lead_hour,
                issued_at_utc=task.issued_at_utc,
                valid_at_utc=task.valid_at_utc,
                availability_proxy_utc=task.availability_proxy_utc,
                availability_proxy_method=task.availability_proxy_method,
                http_last_modified_utc=public.parse_http_datetime(idx_headers.get("last-modified")),
                content_length_header=object_length,
                content_type=None,
                error=invalid,
            ),
            {"provider_mode": "s3_idx_range", "idx_url": idx_url, "validation_error": invalid},
        )
    provider_mode = "s3_idx_range_coalesced" if coalesce_gap_bytes is not None else "s3_idx_range"
    return (
        public.FetchResult(
            kind=task.kind,
            source=task.source,
            item_id=task.item_id,
            status="ok",
            url=object_url,
            path=task.path,
            bytes=total_bytes,
            sha256=sha256_bytes(data),
            elapsed_seconds=0.0,
            retrieved_at_utc=iso(utc_now()) or "",
            fetched_now=True,
            issue_day_utc=task.issue_day_utc,
            cycle_hour=task.cycle_hour,
            lead_hour=task.lead_hour,
            issued_at_utc=task.issued_at_utc,
            valid_at_utc=task.valid_at_utc,
            availability_proxy_utc=task.availability_proxy_utc,
            availability_proxy_method=task.availability_proxy_method,
            http_last_modified_utc=public.parse_http_datetime(idx_headers.get("last-modified")),
            content_length_header=object_length,
            content_type="application/octet-stream",
            error=None,
        ),
        {
            "provider_mode": provider_mode,
            "idx_url": idx_url,
            "object_url": object_url,
            "object_length": object_length,
            "selected_message_count": len(ranges),
            "selected_variables": sorted(set(selected_variables)),
            "selected_variable_level_pairs": sorted(set(selected_level_pairs)),
            "selected_range_count": len(ranges),
            "merged_range_count": len(transfer_ranges),
            "selected_bytes": selected_bytes,
            "downloaded_bytes_expected": downloaded_bytes_expected,
            "downloaded_extra_bytes": max(0, downloaded_bytes_expected - selected_bytes),
            "model_range_coalesce_gap_bytes": coalesce_gap_bytes,
            "selected_range_workers": effective_range_workers,
            "nomads_filter_url": task.url,
        },
    )


def fetch_himawari(
    task: public.FetchTask,
    full_path: Path,
    *,
    timeout: int,
) -> tuple[public.FetchResult, dict[str, Any]]:
    data, headers, _status = http_get_bytes(task.url, timeout=timeout)
    invalid = public.payload_validation_error(task, data)
    if invalid:
        return (
            public.FetchResult(
                kind=task.kind,
                source=task.source,
                item_id=task.item_id,
                status="invalid_payload",
                url=task.url,
                path=None,
                bytes=len(data),
                sha256=sha256_bytes(data),
                elapsed_seconds=0.0,
                retrieved_at_utc=iso(utc_now()) or "",
                fetched_now=True,
                observed_at_utc=task.observed_at_utc,
                band=task.band,
                segment=task.segment,
                availability_proxy_utc=task.availability_proxy_utc,
                availability_proxy_method=task.availability_proxy_method,
                http_last_modified_utc=public.parse_http_datetime(headers.get("last-modified")),
                content_length_header=int(headers["content-length"]) if headers.get("content-length", "").isdigit() else None,
                content_type=headers.get("content-type"),
                error=invalid,
            ),
            {"provider_mode": "s3_himawari_hsd", "validation_error": invalid},
        )
    write_payload(full_path, data)
    return (
        public.FetchResult(
            kind=task.kind,
            source=task.source,
            item_id=task.item_id,
            status="ok",
            url=task.url,
            path=task.path,
            bytes=len(data),
            sha256=sha256_bytes(data),
            elapsed_seconds=0.0,
            retrieved_at_utc=iso(utc_now()) or "",
            fetched_now=True,
            observed_at_utc=task.observed_at_utc,
            band=task.band,
            segment=task.segment,
            availability_proxy_utc=task.availability_proxy_utc,
            availability_proxy_method=task.availability_proxy_method,
            http_last_modified_utc=public.parse_http_datetime(headers.get("last-modified")),
            content_length_header=int(headers["content-length"]) if headers.get("content-length", "").isdigit() else None,
            content_type=headers.get("content-type"),
            error=None,
        ),
        {"provider_mode": "s3_himawari_hsd"},
    )


def fetch_task_to_path(
    task: public.FetchTask,
    experiment_dir: Path,
    *,
    timeout: int,
    max_attempts: int,
    prefer_s3_model: bool = False,
    model_range_workers: int | None = None,
    model_range_coalesce_gap_bytes: int | None = None,
) -> tuple[public.FetchResult, dict[str, Any]]:
    full_path = experiment_dir / task.path
    started = time.perf_counter()
    try:
        if task.kind == "model_grib":
            if prefer_s3_model:
                (fetch, metadata), attempts = request_with_retries(
                    f"{task.item_id}:s3_idx_range",
                    lambda: fetch_s3_idx_range_model(
                        task,
                        full_path,
                        timeout=timeout,
                        range_workers=model_range_workers,
                        coalesce_gap_bytes=model_range_coalesce_gap_bytes,
                    ),
                    max_attempts=max_attempts,
                )
            else:
                try:
                    (fetch, metadata), attempts = request_with_retries(
                        task.item_id,
                        lambda: fetch_nomads_model(task, full_path, timeout=timeout),
                        max_attempts=max_attempts,
                    )
                except HTTPError as exc:
                    if exc.code not in {403, 404}:
                        raise
                    (fetch, metadata), attempts = request_with_retries(
                        f"{task.item_id}:s3_idx_range",
                        lambda: fetch_s3_idx_range_model(
                            task,
                            full_path,
                            timeout=timeout,
                            range_workers=model_range_workers,
                            coalesce_gap_bytes=model_range_coalesce_gap_bytes,
                        ),
                        max_attempts=max_attempts,
                    )
        elif task.kind == "himawari_hsd":
            (fetch, metadata), attempts = request_with_retries(
                task.item_id,
                lambda: fetch_himawari(task, full_path, timeout=timeout),
                max_attempts=max_attempts,
            )
        else:
            raise ValueError(f"Unsupported task kind: {task.kind}")
        fetch.elapsed_seconds = time.perf_counter() - started
        metadata["attempts"] = attempts
        return fetch, metadata
    except Exception as exc:  # noqa: BLE001
        return (
            public.FetchResult(
                kind=task.kind,
                source=task.source,
                item_id=task.item_id,
                status="error",
                url=task.url,
                path=str(Path(task.path)) if Path(wp(full_path)).exists() else None,
                bytes=file_size(full_path),
                sha256=sha256_bytes(Path(wp(full_path)).read_bytes()) if Path(wp(full_path)).is_file() else None,
                elapsed_seconds=time.perf_counter() - started,
                retrieved_at_utc=iso(utc_now()) or "",
                fetched_now=False,
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
            ),
            {"provider_mode": "unresolved", "error": f"{type(exc).__name__}: {exc}"},
        )


def safe_delete_file(path: Path, staging_root: Path) -> int:
    if not os.path.exists(wp(path)):
        return 0
    resolved = path.resolve()
    staging = staging_root.resolve()
    if not str(resolved).lower().startswith(str(staging).lower()):
        raise RuntimeError(f"Refusing to delete outside staging: {resolved}")
    if not os.path.isfile(wp(resolved)):
        return 0
    size = os.path.getsize(wp(resolved))
    os.remove(wp(resolved))
    return size


def safe_remove_staging_day(day_staging: Path, experiment_dir: Path) -> None:
    if not day_staging.exists():
        return
    resolved = day_staging.resolve()
    root = (experiment_dir / "staging").resolve()
    if not str(resolved).lower().startswith(str(root).lower()):
        raise RuntimeError(f"Refusing to remove staging outside experiment: {resolved}")
    if resolved.name == "staging":
        raise RuntimeError("Refusing to remove the whole staging root")
    shutil.rmtree(wp(resolved))


def safe_remove_staging_dir(target: Path, staging_root: Path) -> None:
    if not os.path.exists(wp(target)):
        return
    resolved = target.resolve()
    root = staging_root.resolve()
    if not str(resolved).lower().startswith(str(root).lower()):
        raise RuntimeError(f"Refusing to remove staging outside root: {resolved}")
    if resolved == root:
        raise RuntimeError("Refusing to remove the whole staging root")
    shutil.rmtree(wp(resolved))


def normalize_value_array(values: np.ndarray, units: str | None) -> tuple[np.ndarray, str]:
    if units == "K":
        return values.astype(float) - 273.15, "degC"
    if units == "Pa":
        return values.astype(float) / 100.0, "hPa"
    normalized_unit = re.sub(r"[^a-zA-Z0-9]+", "_", units or "").strip("_")
    return values.astype(float), normalized_unit


def half_gradient(values: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    finite = np.isfinite(values)
    if not finite.any():
        return {"east_west_gradient": None, "south_north_gradient": None}
    lon_grid = np.broadcast_to(lons.reshape(1, -1), values.shape)
    lat_grid = np.broadcast_to(lats.reshape(-1, 1), values.shape)
    east = values[(lon_grid >= public.HKO["longitude"]) & finite]
    west = values[(lon_grid < public.HKO["longitude"]) & finite]
    north = values[(lat_grid >= public.HKO["latitude"]) & finite]
    south = values[(lat_grid < public.HKO["latitude"]) & finite]
    out["east_west_gradient"] = float(np.nanmean(east) - np.nanmean(west)) if east.size and west.size else None
    out["south_north_gradient"] = float(np.nanmean(south) - np.nanmean(north)) if south.size and north.size else None
    return out


def normalize_model_result_full(experiment_dir: Path, fetch_dict: dict[str, Any]) -> dict[str, Any]:
    fetch = public.FetchResult(**fetch_dict)
    started = time.perf_counter()
    if fetch.status != "ok" or not fetch.path:
        return {"status": "skip", "fetch": asdict(fetch), "station_row": None, "bbox_rows": []}
    path = experiment_dir / fetch.path
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
        "station_id": public.HKO["station_id"],
        "station_name": public.HKO["station_name"],
        "station_latitude": public.HKO["latitude"],
        "station_longitude": public.HKO["longitude"],
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
                nearest_lat, nearest_lon = public.nearest_grid(ds)
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

                issued_at = (
                    iso(pd.Timestamp(public.scalar_coord(da, "time")))
                    if public.scalar_coord(da, "time") is not None
                    else None
                )
                valid_at = (
                    iso(pd.Timestamp(public.scalar_coord(da, "valid_time")))
                    if public.scalar_coord(da, "valid_time") is not None
                    else None
                )
                lead = (
                    public.lead_hours_from_coord(public.scalar_coord(da, "step"))
                    if public.scalar_coord(da, "step") is not None
                    else None
                )
                if issued_at:
                    station_row["issued_at_utc_grib"] = issued_at
                if valid_at:
                    station_row["valid_at_utc_grib"] = valid_at
                if lead is not None:
                    station_row["lead_hour_grib"] = lead

                point_native = float(da.sel(latitude=nearest_lat, longitude=nearest_lon).values)
                point_value, out_unit = public.normalize_value(point_native, units)
                station_row[public.safe_feature_key(canonical, out_unit, station_row)] = point_value

                cropped = public.crop_dataarray(da)
                raw_values = np.asarray(cropped.values, dtype=float)
                if raw_values.size == 0:
                    continue
                normalized, out_unit = normalize_value_array(raw_values, units)
                finite = np.isfinite(normalized)
                total_count = int(normalized.size)
                valid_count = int(finite.sum())
                if not valid_count:
                    continue
                vals = normalized[finite]
                lats = np.asarray(cropped["latitude"].values, dtype=float)
                lons = np.asarray(cropped["longitude"].values, dtype=float)
                gradients = half_gradient(normalized, lats, lons)
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
                        "grid_point_count": valid_count,
                        "grid_total_count": total_count,
                        "grid_missing_count": total_count - valid_count,
                        "bbox_min": float(np.nanmin(vals)),
                        "bbox_mean": float(np.nanmean(vals)),
                        "bbox_median": float(np.nanmedian(vals)),
                        "bbox_p10": float(np.nanpercentile(vals, 10)),
                        "bbox_p50": float(np.nanpercentile(vals, 50)),
                        "bbox_p90": float(np.nanpercentile(vals, 90)),
                        "bbox_max": float(np.nanmax(vals)),
                        "bbox_std": float(np.nanstd(vals)),
                        "bbox_east_west_gradient": gradients["east_west_gradient"],
                        "bbox_south_north_gradient": gradients["south_north_gradient"],
                    }
                )
                variable_count += 1
        station_row["normalized_variable_count"] = variable_count
        station_row["normalization_elapsed_seconds"] = time.perf_counter() - started
        return {"status": "ok", "fetch": asdict(fetch), "station_row": station_row, "bbox_rows": summary_rows}
    except Exception as exc:  # noqa: BLE001
        station_row["normalized_variable_count"] = variable_count
        station_row["normalization_elapsed_seconds"] = time.perf_counter() - started
        station_row["normalization_error"] = f"{type(exc).__name__}: {exc}"
        return {"status": "error", "fetch": asdict(fetch), "station_row": station_row, "bbox_rows": summary_rows}


def normalize_himawari_result_full(experiment_dir: Path, fetch: public.FetchResult) -> dict[str, Any]:
    started = time.perf_counter()
    if fetch.status != "ok" or not fetch.path:
        return {"status": "skip", "fetch": asdict(fetch), "row": None}
    path = experiment_dir / fetch.path
    try:
        with bz2.open(wp(path), "rb") as handle:
            data = handle.read()
        header = public.parse_himawari_header(data, fetch)
        counts, radiance, bt_c, quality_code = public.himawari_bt(data, header)
        row, col, global_line, global_col = public.hko_pixel(header)
        if not (0 <= row < bt_c.shape[0] and 0 <= col < bt_c.shape[1]):
            raise ValueError(f"Projected HKO pixel outside segment: row={row}, col={col}")

        file_created = header.get("file_creation_utc")
        available_proxy = fetch.availability_proxy_utc
        if file_created and parse_iso(file_created):
            observed_plus_30 = parse_iso(fetch.availability_proxy_utc)
            created = parse_iso(file_created)
            available_proxy = iso(max(created, observed_plus_30)) if observed_plus_30 and created else file_created

        valid_bt = bt_c[np.isfinite(bt_c)]
        observed = header.get("observed_at_utc") or fetch.observed_at_utc
        parsed_observed = parse_iso(observed)
        row_data: dict[str, Any] = {
            "source": fetch.source,
            "item_id": fetch.item_id,
            "band": header.get("band") or fetch.band,
            "segment": header.get("segment_code") or fetch.segment,
            "observed_at_utc": observed,
            "observed_date_hkt": public.hkt_date(observed),
            "availability_proxy_utc": available_proxy,
            "availability_proxy_method": "max(hsd_file_creation_utc, observed_at_utc + 30m conservative buffer)",
            "t24_next_target_date_hkt": (
                (parsed_observed.astimezone(HKT).date() + timedelta(days=1)).isoformat()
                if parsed_observed
                else None
            ),
            "t24_next_day_cutoff_utc": public.t24_next_day_cutoff_for_observed(observed),
            "eligible_for_next_day_t24_cutoff": (
                parse_iso(available_proxy) <= parse_iso(public.t24_next_day_cutoff_for_observed(observed))
                if parse_iso(available_proxy) and parse_iso(public.t24_next_day_cutoff_for_observed(observed))
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
        row_data.update(public.window_features("w3", bt_c, row, col, 1))
        row_data.update(public.window_features("w5", bt_c, row, col, 2))
        row_data.update(public.window_features("w11", bt_c, row, col, 5))
        row_data.update(public.window_features("w21", bt_c, row, col, 10))
        row_data.update(public.window_features("w41", bt_c, row, col, 20))
        if "w21_mean_bt_c" in row_data:
            row_data["hko_minus_w21_mean_bt_c"] = float(row_data["hko_bt_c"] - row_data["w21_mean_bt_c"])
        if "w41_mean_bt_c" in row_data:
            row_data["hko_minus_w41_mean_bt_c"] = float(row_data["hko_bt_c"] - row_data["w41_mean_bt_c"])
        if 5 <= row < bt_c.shape[0] - 5 and 5 <= col < bt_c.shape[1] - 5:
            row_data["east_west_gradient_bt_c"] = float(
                np.nanmean(bt_c[row - 5 : row + 6, col + 1 : col + 6])
                - np.nanmean(bt_c[row - 5 : row + 6, col - 5 : col])
            )
            row_data["south_north_gradient_bt_c"] = float(
                np.nanmean(bt_c[row + 1 : row + 6, col - 5 : col + 6])
                - np.nanmean(bt_c[row - 5 : row, col - 5 : col + 6])
            )
        row_data["normalization_elapsed_seconds"] = time.perf_counter() - started
        return {"status": "ok", "fetch": asdict(fetch), "row": row_data}
    except Exception as exc:  # noqa: BLE001
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


def task_to_payload(task: public.FetchTask) -> dict[str, Any]:
    return asdict(task)


def task_from_payload(payload: dict[str, Any]) -> public.FetchTask:
    return public.FetchTask(**payload)


def error_fetch_from_task(task: public.FetchTask, error: str, elapsed_seconds: float = 0.0) -> dict[str, Any]:
    return asdict(
        public.FetchResult(
            kind=task.kind,
            source=task.source,
            item_id=task.item_id,
            status="error",
            url=task.url,
            path=None,
            bytes=0,
            sha256=None,
            elapsed_seconds=elapsed_seconds,
            retrieved_at_utc=iso(utc_now()) or "",
            fetched_now=False,
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
            error=error,
        )
    )


def fetch_static_worker(
    task_payload: dict[str, Any],
    task_base_dir: str,
    timeout: int,
    max_attempts: int,
    prefer_s3_model: bool,
    model_range_workers: int,
    model_range_coalesce_gap_bytes: int | None,
) -> dict[str, Any]:
    task = task_from_payload(task_payload)
    started = time.perf_counter()
    fetch, metadata = fetch_task_to_path(
        task,
        Path(task_base_dir),
        timeout=timeout,
        max_attempts=max_attempts,
        prefer_s3_model=prefer_s3_model,
        model_range_workers=model_range_workers,
        model_range_coalesce_gap_bytes=model_range_coalesce_gap_bytes,
    )
    phase_seconds = {"fetch": time.perf_counter() - started}
    raw_full_path = str(Path(task_base_dir) / fetch.path) if fetch.path else None
    return {
        "task": task_payload,
        "fetch": asdict(fetch),
        "metadata": metadata,
        "raw_full_path": raw_full_path,
        "phase_seconds": phase_seconds,
    }


def normalize_model_worker(task_base_dir: str, fetch_dict: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    result = normalize_model_result_full(Path(task_base_dir), fetch_dict)
    return {"normalization": result, "phase_seconds": {"normalize": time.perf_counter() - started}}


def fetch_normalize_himawari_worker(
    task_payload: dict[str, Any],
    task_base_dir: str,
    timeout: int,
    max_attempts: int,
) -> dict[str, Any]:
    fetched = fetch_static_worker(
        task_payload,
        task_base_dir,
        timeout,
        max_attempts,
        False,
        S3_RANGE_WORKERS,
        None,
    )
    fetch = public.FetchResult(**fetched["fetch"])
    started = time.perf_counter()
    fetched["normalization"] = normalize_himawari_result_full(Path(task_base_dir), fetch)
    fetched["phase_seconds"]["normalize"] = time.perf_counter() - started
    return fetched


def envf_url(start_hkt: datetime, npics: int = 72, interval_hours: float = 0.2) -> str:
    return (
        "https://envf.ust.hk/dataview/hko_radar/current/index.py?"
        f"year__int={start_hkt.year}"
        f"&month__int={start_hkt.month}"
        f"&day__int={start_hkt.day}"
        f"&hour__int={start_hkt.hour}"
        f"&npics__int={npics}"
        f"&interval__float={interval_hours}"
        "&display=Search"
    )


def parse_hkt_datetime(value: str) -> datetime:
    return datetime.strptime(value.strip(), "%Y/%m/%d %H:%M").replace(tzinfo=HKT)


def envf_chunk_starts(day: date) -> list[datetime]:
    utc_start = datetime.combine(day, dt_time(0, 0), tzinfo=UTC)
    utc_end = utc_start + timedelta(days=1)
    cursor = utc_start.astimezone(HKT)
    starts: list[datetime] = []
    while cursor.astimezone(UTC) < utc_end:
        starts.append(cursor)
        cursor += timedelta(hours=12)
    return starts


def fetch_envf_manifest_for_day(day: date, timeout: int, max_attempts: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    utc_start = datetime.combine(day, dt_time(0, 0), tzinfo=UTC)
    utc_end = utc_start + timedelta(days=1)
    rows: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    seen: set[str] = set()
    with httpx.Client(follow_redirects=True, timeout=timeout, headers={"User-Agent": USER_AGENT}) as client:
        for start_hkt in envf_chunk_starts(day):
            url = envf_url(start_hkt)
            t0 = time.perf_counter()
            html = ""
            status = "error"
            error = None
            attempts = 0
            for attempt in range(1, max_attempts + 1):
                attempts = attempt
                try:
                    resp = client.get(url)
                    resp.raise_for_status()
                    html = resp.text
                    status = "ok"
                    break
                except Exception as exc:  # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}"
                    if attempt < max_attempts and should_retry_error(exc):
                        retry_sleep(attempt)
                        continue
                    break
            logs.append(
                {
                    "chunk_start_hkt": start_hkt.isoformat(),
                    "url": url,
                    "status": status,
                    "attempts": attempts,
                    "elapsed_seconds": time.perf_counter() - t0,
                    "html_bytes": len(html.encode("utf-8")),
                    "error": error,
                }
            )
            if status != "ok":
                continue
            links = re.findall(r'href="([^"]*display_large_image[^"]*)"', html)
            thumbs = re.findall(r'<img[^>]+src="([^"]+)"', html, flags=re.I)
            thumb_by_index = {i: urljoin(url, link) for i, link in enumerate(thumbs)}
            for idx, href in enumerate(links):
                full_href = urljoin(url, href.replace("&amp;", "&"))
                parsed = urlparse(full_href)
                qs = parse_qs(parsed.query)
                dt_values = qs.get("datetime", [])
                imagef_values = qs.get("imagef", [])
                if not dt_values:
                    continue
                frame_hkt = parse_hkt_datetime(dt_values[0])
                observed_utc = frame_hkt.astimezone(UTC)
                if not (utc_start <= observed_utc < utc_end):
                    continue
                key = iso(observed_utc) or observed_utc.isoformat()
                if key in seen:
                    continue
                seen.add(key)
                availability_utc = observed_utc + timedelta(minutes=30)
                rows.append(
                    {
                        "source": "envf_hkust_hko_radar",
                        "provider": "HKUST ENVF mirror of HKO radar imagery",
                        "product": "hko_radar_image",
                        "frame_time_hkt": frame_hkt.isoformat(),
                        "observed_at_utc": iso(observed_utc),
                        "availability_proxy_utc": iso(availability_utc),
                        "availability_proxy_method": "historical_display_proxy_observed_at_plus_30m",
                        "native_issue_metadata_status": "not_native_exact_vintage",
                        "envf_query_url": url,
                        "display_large_image_url": full_href,
                        "display_image_url": thumb_by_index.get(idx),
                        "envf_temp_image_path": unquote(imagef_values[0]) if imagef_values else None,
                        "frame_index_in_chunk": idx,
                    }
                )
    rows.sort(key=lambda item: str(item.get("observed_at_utc")))
    return rows, logs


def color_fraction(mask: np.ndarray, total: int) -> float:
    return float(mask.sum() / total) if total else 0.0


def rgb_block_features(prefix: str, arr: np.ndarray) -> dict[str, Any]:
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    maxc = np.maximum.reduce([r, g, b])
    minc = np.minimum.reduce([r, g, b])
    sat = maxc - minc
    bright = (r + g + b) / 3.0
    colored = (sat >= 35) & (maxc >= 60)
    dark = bright < 40
    blue = colored & (b > r + 25) & (b > g + 10)
    cyan = colored & (b > r + 20) & (g > r + 20) & (np.abs(g - b) <= 55)
    green = colored & (g > r + 25) & (g > b + 10)
    yellow = colored & (r > b + 35) & (g > b + 35) & (np.abs(r - g) <= 80)
    orange = colored & (r > g + 20) & (g > b + 25)
    red = colored & (r > g + 35) & (r > b + 35)
    purple = colored & (r > g + 20) & (b > g + 20)
    total = int(arr.shape[0] * arr.shape[1])
    return {
        f"{prefix}_pixel_count": total,
        f"{prefix}_rgb_r_mean": float(r.mean()) if total else None,
        f"{prefix}_rgb_g_mean": float(g.mean()) if total else None,
        f"{prefix}_rgb_b_mean": float(b.mean()) if total else None,
        f"{prefix}_rgb_brightness_mean": float(bright.mean()) if total else None,
        f"{prefix}_rgb_saturation_mean": float(sat.mean()) if total else None,
        f"{prefix}_dark_pixel_fraction": color_fraction(dark, total),
        f"{prefix}_rain_colored_pixel_fraction": color_fraction(colored, total),
        f"{prefix}_rain_blue_fraction": color_fraction(blue, total),
        f"{prefix}_rain_cyan_fraction": color_fraction(cyan, total),
        f"{prefix}_rain_green_fraction": color_fraction(green, total),
        f"{prefix}_rain_yellow_fraction": color_fraction(yellow, total),
        f"{prefix}_rain_orange_fraction": color_fraction(orange, total),
        f"{prefix}_rain_red_fraction": color_fraction(red, total),
        f"{prefix}_rain_purple_fraction": color_fraction(purple, total),
        f"{prefix}_rain_intensity_proxy": float(
            color_fraction(blue, total) * 1.0
            + color_fraction(cyan, total) * 1.2
            + color_fraction(green, total) * 1.7
            + color_fraction(yellow, total) * 2.5
            + color_fraction(orange, total) * 3.2
            + color_fraction(red, total) * 4.0
            + color_fraction(purple, total) * 4.5
        ),
    }


def radar_image_features(content: bytes) -> dict[str, Any]:
    with Image.open(io.BytesIO(content)) as img:
        rgb = img.convert("RGB")
        arr = np.asarray(rgb, dtype=np.uint8)
    h, w, _channels = arr.shape
    center_radius = max(10, min(h, w) // 20)
    cy = h // 2
    cx = w // 2
    center = arr[max(0, cy - center_radius) : min(h, cy + center_radius + 1), max(0, cx - center_radius) : min(w, cx + center_radius + 1), :]
    features: dict[str, Any] = {
        "image_sha256": sha256_bytes(content),
        "image_bytes": len(content),
        "image_width": int(w),
        "image_height": int(h),
        "center_proxy_note": "not_georeferenced; center-window proxy only",
    }
    features.update(rgb_block_features("image", arr))
    features.update(rgb_block_features("image_center_proxy", center))
    return features


def fetch_radar_frame(
    record: dict[str, Any],
    *,
    timeout: int,
    max_attempts: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    url = record.get("display_image_url") or record.get("display_large_image_url")
    if not url:
        return {
            **record,
            "image_fetch_status": "error",
            "image_fetch_error": "missing image url",
            "normalization_elapsed_seconds": time.perf_counter() - started,
        }
    error = None
    attempts = 0
    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        try:
            data, _headers, _status = http_get_bytes(str(url), timeout=timeout)
            features = radar_image_features(data)
            return {
                **record,
                "image_fetch_status": "ok",
                "image_fetch_error": None,
                "image_fetch_attempts": attempts,
                "normalization_elapsed_seconds": time.perf_counter() - started,
                **features,
            }
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            if attempt < max_attempts and should_retry_error(exc):
                retry_sleep(attempt)
                continue
            break
    return {
        **record,
        "image_fetch_status": "error",
        "image_fetch_error": error,
        "image_fetch_attempts": attempts,
        "normalization_elapsed_seconds": time.perf_counter() - started,
    }


SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS weather_backfill;

CREATE TABLE IF NOT EXISTS weather_backfill.ingest_run (
    run_id text PRIMARY KEY,
    experiment_id text NOT NULL,
    started_at timestamptz NOT NULL,
    completed_at timestamptz,
    status text NOT NULL,
    start_date_utc date NOT NULL,
    end_date_utc date NOT NULL,
    sources text[] NOT NULL,
    config jsonb NOT NULL,
    summary jsonb,
    error text
);

CREATE TABLE IF NOT EXISTS weather_backfill.source_issue (
    issue_key text PRIMARY KEY,
    source text NOT NULL,
    product text NOT NULL,
    issue_day_utc date,
    issued_at_utc timestamptz,
    observed_at_utc timestamptz,
    valid_at_utc timestamptz,
    availability_proxy_utc timestamptz NOT NULL,
    availability_proxy_method text,
    status text NOT NULL,
    source_url text,
    raw_sha256 text,
    raw_bytes bigint,
    raw_retention_policy text NOT NULL,
    normalized_dataset_id text,
    created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE weather_backfill.source_issue ADD COLUMN IF NOT EXISTS run_id text;
ALTER TABLE weather_backfill.source_issue ADD COLUMN IF NOT EXISTS item_id text;
ALTER TABLE weather_backfill.source_issue ADD COLUMN IF NOT EXISTS cycle_hour integer;
ALTER TABLE weather_backfill.source_issue ADD COLUMN IF NOT EXISTS lead_hour integer;
ALTER TABLE weather_backfill.source_issue ADD COLUMN IF NOT EXISTS band text;
ALTER TABLE weather_backfill.source_issue ADD COLUMN IF NOT EXISTS segment text;
ALTER TABLE weather_backfill.source_issue ADD COLUMN IF NOT EXISTS retrieved_at_utc timestamptz;
ALTER TABLE weather_backfill.source_issue ADD COLUMN IF NOT EXISTS available_at_utc timestamptz;
ALTER TABLE weather_backfill.source_issue ADD COLUMN IF NOT EXISTS normalized_status text;
ALTER TABLE weather_backfill.source_issue ADD COLUMN IF NOT EXISTS error text;
ALTER TABLE weather_backfill.source_issue ADD COLUMN IF NOT EXISTS metadata jsonb NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE weather_backfill.source_issue ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT now();

CREATE TABLE IF NOT EXISTS weather_backfill.station_feature (
    issue_key text NOT NULL REFERENCES weather_backfill.source_issue(issue_key) ON DELETE CASCADE,
    station_id text NOT NULL,
    feature_name text NOT NULL,
    value_double double precision,
    value_text text,
    feature_unit text,
    valid_at_utc timestamptz,
    available_at_utc timestamptz NOT NULL,
    run_id text,
    feature_context jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (issue_key, station_id, feature_name)
);

CREATE TABLE IF NOT EXISTS weather_backfill.area_feature (
    issue_key text NOT NULL REFERENCES weather_backfill.source_issue(issue_key) ON DELETE CASCADE,
    area_key text NOT NULL,
    variable_name text NOT NULL,
    statistic text NOT NULL,
    value_double double precision,
    value_text text,
    feature_unit text,
    valid_at_utc timestamptz,
    available_at_utc timestamptz NOT NULL,
    run_id text,
    feature_context jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (issue_key, area_key, variable_name, statistic)
);

CREATE TABLE IF NOT EXISTS weather_backfill.normalized_artifact (
    artifact_id text PRIMARY KEY,
    dataset_id text NOT NULL,
    source text NOT NULL,
    product text NOT NULL,
    date_start_utc date,
    date_end_utc date,
    uri text NOT NULL,
    format text NOT NULL,
    row_count bigint NOT NULL,
    column_count integer NOT NULL,
    bytes bigint,
    content_sha256 text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS weather_backfill.artifact_column (
    artifact_id text NOT NULL REFERENCES weather_backfill.normalized_artifact(artifact_id) ON DELETE CASCADE,
    column_name text NOT NULL,
    dtype text NOT NULL,
    non_null_count bigint,
    distinct_count bigint,
    min_text text,
    max_text text,
    sample_values_json jsonb,
    PRIMARY KEY (artifact_id, column_name)
);

CREATE TABLE IF NOT EXISTS weather_backfill.ingest_event (
    event_id bigserial PRIMARY KEY,
    run_id text NOT NULL,
    issue_key text,
    source text,
    event_type text NOT NULL,
    status text NOT NULL,
    message text,
    elapsed_seconds double precision,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_weather_source_issue_time
ON weather_backfill.source_issue (source, product, availability_proxy_utc, valid_at_utc);

CREATE INDEX IF NOT EXISTS ix_weather_source_issue_available
ON weather_backfill.source_issue (available_at_utc, source, product);

CREATE INDEX IF NOT EXISTS ix_weather_station_feature_lookup
ON weather_backfill.station_feature (station_id, feature_name, available_at_utc);

CREATE INDEX IF NOT EXISTS ix_weather_area_feature_lookup
ON weather_backfill.area_feature (area_key, variable_name, statistic, available_at_utc);
"""


def apply_schema(conn: psycopg.Connection[Any]) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    conn.commit()


def to_db_ts(value: str | datetime | None) -> datetime | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    return parse_iso(str(value))


def to_db_date(value: str | date | None) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return parse_date(str(value)[:10])


def finite_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return float(int(value))
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def text_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if finite_number(value) is not None:
        return None
    return str(value)


def infer_unit(feature_name: str, explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    lowered = feature_name.lower()
    if lowered.endswith("_degc") or lowered.endswith("_bt_c") or lowered.endswith("_c"):
        return "degC"
    if lowered.endswith("_hpa"):
        return "hPa"
    if lowered.endswith("_fraction"):
        return "fraction"
    if lowered.endswith("_count"):
        return "count"
    if "radiance" in lowered:
        return "W m-2 sr-1 um-1"
    return None


def issue_key_from_fetch(fetch: public.FetchResult) -> str:
    return fetch.item_id


def issue_key_from_radar(record: dict[str, Any]) -> str:
    return f"envf_hko_radar:{record.get('observed_at_utc')}"


def upsert_source_issue(
    conn: psycopg.Connection[Any],
    *,
    run_id: str,
    issue_key: str,
    source: str,
    product: str,
    issue_day_utc: date | None,
    issued_at_utc: datetime | None,
    observed_at_utc: datetime | None,
    valid_at_utc: datetime | None,
    availability_proxy_utc: datetime,
    availability_proxy_method: str | None,
    status: str,
    source_url: str | None,
    raw_sha256: str | None,
    raw_bytes: int | None,
    raw_retention_policy: str,
    item_id: str | None = None,
    cycle_hour: int | None = None,
    lead_hour: int | None = None,
    band: str | None = None,
    segment: str | None = None,
    retrieved_at_utc: datetime | None = None,
    normalized_status: str | None = None,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
    experiment_id: str = EXPERIMENT_ID,
) -> None:
    sql = """
        INSERT INTO weather_backfill.source_issue (
            issue_key, source, product, issue_day_utc, issued_at_utc, observed_at_utc,
            valid_at_utc, availability_proxy_utc, availability_proxy_method, status,
            source_url, raw_sha256, raw_bytes, raw_retention_policy, normalized_dataset_id,
            run_id, item_id, cycle_hour, lead_hour, band, segment, retrieved_at_utc,
            available_at_utc, normalized_status, error, metadata, updated_at
        )
        VALUES (
            %(issue_key)s, %(source)s, %(product)s, %(issue_day_utc)s, %(issued_at_utc)s,
            %(observed_at_utc)s, %(valid_at_utc)s, %(availability_proxy_utc)s,
            %(availability_proxy_method)s, %(status)s, %(source_url)s, %(raw_sha256)s,
            %(raw_bytes)s, %(raw_retention_policy)s, %(normalized_dataset_id)s,
            %(run_id)s, %(item_id)s, %(cycle_hour)s, %(lead_hour)s, %(band)s, %(segment)s,
            %(retrieved_at_utc)s, %(available_at_utc)s, %(normalized_status)s, %(error)s,
            %(metadata)s, now()
        )
        ON CONFLICT (issue_key) DO UPDATE SET
            run_id = EXCLUDED.run_id,
            source = EXCLUDED.source,
            product = EXCLUDED.product,
            issue_day_utc = EXCLUDED.issue_day_utc,
            issued_at_utc = EXCLUDED.issued_at_utc,
            observed_at_utc = EXCLUDED.observed_at_utc,
            valid_at_utc = EXCLUDED.valid_at_utc,
            availability_proxy_utc = EXCLUDED.availability_proxy_utc,
            availability_proxy_method = EXCLUDED.availability_proxy_method,
            status = EXCLUDED.status,
            source_url = EXCLUDED.source_url,
            raw_sha256 = EXCLUDED.raw_sha256,
            raw_bytes = EXCLUDED.raw_bytes,
            raw_retention_policy = EXCLUDED.raw_retention_policy,
            item_id = EXCLUDED.item_id,
            cycle_hour = EXCLUDED.cycle_hour,
            lead_hour = EXCLUDED.lead_hour,
            band = EXCLUDED.band,
            segment = EXCLUDED.segment,
            retrieved_at_utc = EXCLUDED.retrieved_at_utc,
            available_at_utc = EXCLUDED.available_at_utc,
            normalized_status = EXCLUDED.normalized_status,
            error = EXCLUDED.error,
            metadata = COALESCE(weather_backfill.source_issue.metadata, '{}'::jsonb) || EXCLUDED.metadata,
            updated_at = now()
    """
    params = {
        "issue_key": issue_key,
        "source": source,
        "product": product,
        "issue_day_utc": issue_day_utc,
        "issued_at_utc": issued_at_utc,
        "observed_at_utc": observed_at_utc,
        "valid_at_utc": valid_at_utc,
        "availability_proxy_utc": availability_proxy_utc,
        "availability_proxy_method": availability_proxy_method,
        "status": status,
        "source_url": source_url,
        "raw_sha256": raw_sha256,
        "raw_bytes": raw_bytes,
        "raw_retention_policy": raw_retention_policy,
        "normalized_dataset_id": experiment_id,
        "run_id": run_id,
        "item_id": item_id,
        "cycle_hour": cycle_hour,
        "lead_hour": lead_hour,
        "band": band,
        "segment": segment,
        "retrieved_at_utc": retrieved_at_utc,
        "available_at_utc": availability_proxy_utc,
        "normalized_status": normalized_status,
        "error": error,
        "metadata": Jsonb(metadata or {}),
    }
    with conn.cursor() as cur:
        cur.execute(sql, params)


def insert_ingest_event(
    conn: psycopg.Connection[Any],
    *,
    run_id: str,
    issue_key: str | None,
    source: str | None,
    event_type: str,
    status: str,
    message: str | None,
    elapsed_seconds: float | None,
    metadata: dict[str, Any] | None = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO weather_backfill.ingest_event (
                run_id, issue_key, source, event_type, status, message, elapsed_seconds, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (run_id, issue_key, source, event_type, status, message, elapsed_seconds, Jsonb(metadata or {})),
        )


def source_issue_has_features(conn: psycopg.Connection[Any], issue_key: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                EXISTS (SELECT 1 FROM weather_backfill.station_feature WHERE issue_key = %s)
                OR EXISTS (SELECT 1 FROM weather_backfill.area_feature WHERE issue_key = %s)
            """,
            (issue_key, issue_key),
        )
        return bool(cur.fetchone()[0])


def insert_station_features(
    conn: psycopg.Connection[Any],
    *,
    run_id: str,
    issue_key: str,
    station_id: str,
    valid_at_utc: datetime | None,
    available_at_utc: datetime,
    features: dict[str, Any],
    context: dict[str, Any],
    skip_keys: set[str],
) -> int:
    rows: list[tuple[Any, ...]] = []
    for name, value in features.items():
        if name in skip_keys:
            continue
        number = finite_number(value)
        text = text_value(value)
        if number is None and text is None:
            continue
        rows.append(
            (
                issue_key,
                station_id,
                name,
                number,
                text,
                infer_unit(name),
                valid_at_utc,
                available_at_utc,
                run_id,
                Jsonb(context),
            )
        )
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO weather_backfill.station_feature (
                issue_key, station_id, feature_name, value_double, value_text, feature_unit,
                valid_at_utc, available_at_utc, run_id, feature_context, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (issue_key, station_id, feature_name) DO UPDATE SET
                value_double = EXCLUDED.value_double,
                value_text = EXCLUDED.value_text,
                feature_unit = EXCLUDED.feature_unit,
                valid_at_utc = EXCLUDED.valid_at_utc,
                available_at_utc = EXCLUDED.available_at_utc,
                run_id = EXCLUDED.run_id,
                feature_context = EXCLUDED.feature_context,
                updated_at = now()
            """,
            rows,
        )
    return len(rows)


def insert_area_features(
    conn: psycopg.Connection[Any],
    *,
    run_id: str,
    issue_key: str,
    area_key: str,
    variable_name: str,
    valid_at_utc: datetime | None,
    available_at_utc: datetime,
    features: dict[str, Any],
    context: dict[str, Any],
    skip_keys: set[str],
) -> int:
    rows: list[tuple[Any, ...]] = []
    for statistic, value in features.items():
        if statistic in skip_keys:
            continue
        number = finite_number(value)
        text = text_value(value)
        if number is None and text is None:
            continue
        rows.append(
            (
                issue_key,
                area_key,
                variable_name,
                statistic,
                number,
                text,
                infer_unit(statistic, str(context.get("normalized_units")) if context.get("normalized_units") else None),
                valid_at_utc,
                available_at_utc,
                run_id,
                Jsonb(context),
            )
        )
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO weather_backfill.area_feature (
                issue_key, area_key, variable_name, statistic, value_double, value_text,
                feature_unit, valid_at_utc, available_at_utc, run_id, feature_context, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (issue_key, area_key, variable_name, statistic) DO UPDATE SET
                value_double = EXCLUDED.value_double,
                value_text = EXCLUDED.value_text,
                feature_unit = EXCLUDED.feature_unit,
                valid_at_utc = EXCLUDED.valid_at_utc,
                available_at_utc = EXCLUDED.available_at_utc,
                run_id = EXCLUDED.run_id,
                feature_context = EXCLUDED.feature_context,
                updated_at = now()
            """,
            rows,
        )
    return len(rows)


def begin_ingest_run(
    conn: psycopg.Connection[Any],
    *,
    run_id: str,
    experiment_id: str,
    started_at: datetime,
    start_date: date,
    end_date: date,
    sources: set[str],
    config: dict[str, Any],
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO weather_backfill.ingest_run (
                run_id, experiment_id, started_at, status, start_date_utc, end_date_utc, sources, config
            )
            VALUES (%s, %s, %s, 'running', %s, %s, %s, %s)
            ON CONFLICT (run_id) DO UPDATE SET
                started_at = EXCLUDED.started_at,
                status = 'running',
                start_date_utc = EXCLUDED.start_date_utc,
                end_date_utc = EXCLUDED.end_date_utc,
                sources = EXCLUDED.sources,
                config = EXCLUDED.config,
                error = NULL
            """,
            (
                run_id,
                experiment_id,
                started_at,
                start_date,
                end_date,
                sorted(sources),
                Jsonb(config),
            ),
        )
    conn.commit()


def finish_ingest_run(
    conn: psycopg.Connection[Any],
    *,
    run_id: str,
    status: str,
    summary: dict[str, Any],
    error: str | None = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE weather_backfill.ingest_run
            SET completed_at = now(), status = %s, summary = %s, error = %s
            WHERE run_id = %s
            """,
            (status, Jsonb(summary), error, run_id),
        )
    conn.commit()


def build_tasks_for_day(day: date, sources: set[str], cycles: list[int], leads: list[int], experiment_dir: Path) -> list[public.FetchTask]:
    public.EXPERIMENT_DIR = experiment_dir
    public.RAW_DIR = experiment_dir / "staging" / day.isoformat() / "raw"
    public.MODEL_CYCLES = cycles
    tasks: list[public.FetchTask] = []
    if {"gfs", "gefs_control"} & sources:
        tasks.extend([task for task in public.build_model_tasks([day], leads) if task.source in sources])
    if "himawari_b13_s0510" in sources:
        tasks.extend(public.build_himawari_tasks([day], ["B13"], "S0510"))
    return tasks


def fetch_to_issue_record(
    fetch: public.FetchResult,
    metadata: dict[str, Any],
    *,
    normalized_status: str | None,
    retain_failed_raw: bool,
) -> dict[str, Any]:
    status_ok = fetch.status == "ok"
    available = to_db_ts(fetch.availability_proxy_utc)
    if available is None:
        raise ValueError(f"Fetch missing availability timestamp: {fetch.item_id}")
    raw_policy = "raw_deleted_after_db_commit" if status_ok else "raw_not_retained_fetch_failed"
    if not status_ok and retain_failed_raw:
        raw_policy = "failed_raw_retained_for_debug"
    return {
        "issue_key": issue_key_from_fetch(fetch),
        "source": fetch.source,
        "product": fetch.kind,
        "issue_day_utc": to_db_date(fetch.issue_day_utc or (fetch.observed_at_utc or "")[:10]),
        "issued_at_utc": to_db_ts(fetch.issued_at_utc),
        "observed_at_utc": to_db_ts(fetch.observed_at_utc),
        "valid_at_utc": to_db_ts(fetch.valid_at_utc or fetch.observed_at_utc),
        "availability_proxy_utc": available,
        "availability_proxy_method": fetch.availability_proxy_method,
        "status": fetch.status,
        "source_url": fetch.url,
        "raw_sha256": fetch.sha256,
        "raw_bytes": fetch.bytes,
        "raw_retention_policy": raw_policy,
        "item_id": fetch.item_id,
        "cycle_hour": fetch.cycle_hour,
        "lead_hour": fetch.lead_hour,
        "band": fetch.band,
        "segment": fetch.segment,
        "retrieved_at_utc": to_db_ts(fetch.retrieved_at_utc),
        "normalized_status": normalized_status,
        "error": fetch.error,
        "metadata": metadata,
    }


def insert_model_normalization(
    conn: psycopg.Connection[Any],
    *,
    run_id: str,
    issue_key: str,
    result: dict[str, Any],
) -> tuple[int, int]:
    station_count = 0
    area_count = 0
    station_row = result.get("station_row")
    if station_row:
        available = to_db_ts(station_row.get("availability_proxy_utc"))
        if available is None:
            raise ValueError(f"Model row missing availability: {issue_key}")
        valid = to_db_ts(station_row.get("valid_at_utc_grib") or station_row.get("valid_at_utc_expected"))
        context = {
            "source": station_row.get("source"),
            "cycle_hour": station_row.get("cycle_hour"),
            "lead_hour_requested": station_row.get("lead_hour_requested"),
            "raw_sha256": station_row.get("raw_sha256"),
        }
        station_count += insert_station_features(
            conn,
            run_id=run_id,
            issue_key=issue_key,
            station_id=str(station_row.get("station_id") or public.HKO["station_id"]),
            valid_at_utc=valid,
            available_at_utc=available,
            features=station_row,
            context=context,
            skip_keys=MODEL_STATION_META_KEYS,
        )
    for row in result.get("bbox_rows") or []:
        available = to_db_ts(row.get("availability_proxy_utc"))
        if available is None:
            continue
        valid = to_db_ts(row.get("valid_at_utc"))
        variable_name = re.sub(
            r"[^a-zA-Z0-9]+",
            "_",
            f"{row.get('grib_short_name')}_{row.get('grib_type_of_level')}_{row.get('grib_level')}_{row.get('grib_step_type')}",
        ).strip("_").lower()
        context = {
            "source": row.get("source"),
            "cycle_hour": row.get("cycle_hour"),
            "lead_hour_requested": row.get("lead_hour_requested"),
            "grib_short_name": row.get("grib_short_name"),
            "grib_name": row.get("grib_name"),
            "grib_type_of_level": row.get("grib_type_of_level"),
            "grib_level": row.get("grib_level"),
            "grib_step_type": row.get("grib_step_type"),
            "grib_units": row.get("grib_units"),
            "normalized_units": row.get("normalized_units"),
        }
        area_count += insert_area_features(
            conn,
            run_id=run_id,
            issue_key=issue_key,
            area_key="hkg_bbox_113.0_115.5_21.5_23.5",
            variable_name=variable_name or str(row.get("variable")),
            valid_at_utc=valid,
            available_at_utc=available,
            features=row,
            context=context,
            skip_keys={
                "source",
                "item_id",
                "issue_day_utc",
                "cycle_hour",
                "lead_hour_requested",
                "issued_at_utc",
                "valid_at_utc",
                "lead_hour_grib",
                "availability_proxy_utc",
                "variable",
                "grib_short_name",
                "grib_name",
                "grib_type_of_level",
                "grib_level",
                "grib_step_type",
                "grib_units",
                "normalized_units",
            },
        )
    return station_count, area_count


def insert_himawari_normalization(
    conn: psycopg.Connection[Any],
    *,
    run_id: str,
    issue_key: str,
    result: dict[str, Any],
) -> tuple[int, int]:
    row = result.get("row")
    if not row:
        return 0, 0
    available = to_db_ts(row.get("availability_proxy_utc"))
    if available is None:
        raise ValueError(f"Himawari row missing availability: {issue_key}")
    valid = to_db_ts(row.get("observed_at_utc"))
    context = {
        "source": row.get("source"),
        "band": row.get("band"),
        "segment": row.get("segment"),
        "raw_sha256": row.get("raw_sha256"),
    }
    station_count = insert_station_features(
        conn,
        run_id=run_id,
        issue_key=issue_key,
        station_id=public.HKO["station_id"],
        valid_at_utc=valid,
        available_at_utc=available,
        features=row,
        context=context,
        skip_keys=HIMAWARI_META_KEYS,
    )
    segment_features = {key: value for key, value in row.items() if key.startswith("segment_") or key.endswith("_pixel_count")}
    area_count = insert_area_features(
        conn,
        run_id=run_id,
        issue_key=issue_key,
        area_key=f"himawari_{row.get('band')}_{row.get('segment')}",
        variable_name="brightness_temperature",
        valid_at_utc=valid,
        available_at_utc=available,
        features=segment_features,
        context=context,
        skip_keys=set(),
    )
    return station_count, area_count


def upsert_radar_issue_and_features(
    conn: psycopg.Connection[Any],
    *,
    run_id: str,
    experiment_id: str,
    record: dict[str, Any],
) -> tuple[int, int]:
    issue_key = issue_key_from_radar(record)
    available = to_db_ts(record.get("availability_proxy_utc"))
    observed = to_db_ts(record.get("observed_at_utc"))
    if available is None:
        raise ValueError(f"Radar row missing availability: {issue_key}")
    upsert_source_issue(
        conn,
        run_id=run_id,
        issue_key=issue_key,
        source=str(record.get("source")),
        product=str(record.get("product")),
        issue_day_utc=to_db_date(str(record.get("observed_at_utc"))[:10]),
        issued_at_utc=None,
        observed_at_utc=observed,
        valid_at_utc=observed,
        availability_proxy_utc=available,
        availability_proxy_method=record.get("availability_proxy_method"),
        status=str(record.get("image_fetch_status") or "unknown"),
        source_url=record.get("display_image_url") or record.get("display_large_image_url"),
        raw_sha256=record.get("image_sha256"),
        raw_bytes=int(record["image_bytes"]) if record.get("image_bytes") is not None else None,
        raw_retention_policy="image_bytes_loaded_in_memory_then_discarded",
        item_id=issue_key,
        retrieved_at_utc=utc_now(),
        normalized_status=str(record.get("image_fetch_status") or "unknown"),
        error=record.get("image_fetch_error"),
        experiment_id=experiment_id,
        metadata={
            key: value
            for key, value in record.items()
            if key in RADAR_META_KEYS or key in {"center_proxy_note", "image_fetch_attempts"}
        },
    )
    context = {
        "source": record.get("source"),
        "provider": record.get("provider"),
        "native_issue_metadata_status": record.get("native_issue_metadata_status"),
    }
    station_features = {key: value for key, value in record.items() if key.startswith("image_center_proxy_")}
    area_features = {key: value for key, value in record.items() if key.startswith("image_") and not key.startswith("image_center_proxy_")}
    station_count = insert_station_features(
        conn,
        run_id=run_id,
        issue_key=issue_key,
        station_id=public.HKO["station_id"],
        valid_at_utc=observed,
        available_at_utc=available,
        features=station_features,
        context={**context, "proxy_note": "center-window proxy, not georeferenced HKO pixel"},
        skip_keys=set(),
    )
    area_count = insert_area_features(
        conn,
        run_id=run_id,
        issue_key=issue_key,
        area_key="envf_hko_radar_image",
        variable_name="rgb_rain_intensity_proxy",
        valid_at_utc=observed,
        available_at_utc=available,
        features=area_features,
        context=context,
        skip_keys={"image_sha256"},
    )
    return station_count, area_count


def task_counts_for_days(days: list[date], sources: set[str], cycles: list[int], leads: list[int]) -> dict[str, Any]:
    model_sources = len({"gfs", "gefs_control"} & sources)
    return {
        "days": len(days),
        "model_objects": len(days) * model_sources * len(cycles) * len(leads),
        "himawari_scans": len(days) * 144 if "himawari_b13_s0510" in sources else 0,
        "radar_expected_frames_approx": len(days) * 120 if "radar" in sources else 0,
        "total_static_tasks_without_radar_manifest": len(days) * model_sources * len(cycles) * len(leads)
        + (len(days) * 144 if "himawari_b13_s0510" in sources else 0),
    }


def initialize_experiment_docs(experiment_dir: Path, args: argparse.Namespace, run_id: str) -> None:
    experiment_id = args.experiment_id
    ensure_dir(experiment_dir / "logs")
    ensure_dir(experiment_dir / "metadata")
    ensure_dir(experiment_dir / "artifacts")
    ensure_dir(experiment_dir / "normalized")
    upsert_readme_section(
        experiment_dir / "README.md",
        title=experiment_id,
        start_marker=README_CONTRACT_START,
        end_marker=README_CONTRACT_END,
        content=f"""## Acquisition Contract

Lean DB-backed public weather backfill for HKG Tmax research. The pipeline streams public GFS,
GEFS control, Himawari B13/S0510, and radar imagery into `weather_backfill` Postgres tables while
deleting raw payloads immediately after successful normalization and DB commit.

### Hypothesis

Point-in-time public NWP, satellite infrared, and radar scalar features can be acquired with
strict availability timestamps and minimal disk retention, creating a richer leakage-safe
feature store for later HKG Tmax residual experiments.

### Protocol

Process one UTC day at a time. For each source issue, download only the required raw payload,
validate it, normalize scalar station and area features, write source metadata and features to
Postgres, then delete raw bytes before moving to the next item. Every feature must join to a
source issue with `available_at_utc`.

Optimized mode uses bounded fetch/normalize workers, keeps DB writes serialized in the main
process, and deletes each raw payload only after the DB commit or recorded failure handling.

### As-Of Contract

Model issues use `issued_at_utc + 6h` as a conservative availability proxy unless a stronger
provider timestamp is captured. Himawari uses the later of native HSD file creation time and
observed time plus 30 minutes. ENVF radar frames use observed time plus 30 minutes and are
marked as historical display proxy, not native exact radar issue metadata.

### Reproduce

```powershell
$env:HKG_TMAX_DATABASE_URL = '<local postgres url>'
.\\.venv\\Scripts\\python.exe scripts\\backfill_public_weather_to_postgres.py --execute --start-date {args.start_date} --end-date {args.end_date}
```

Review `RUN_CONFIG.yaml` before execution. `DATA_MANIFEST.yaml`, `STATUS.yaml`,
`results/metrics.json`, and `results/runs/<run_id>/metrics.json` are the machine-readable evidence.
""",
    )
    upsert_readme_section(
        experiment_dir / "README.md",
        title=experiment_id,
        start_marker=README_RESULTS_START,
        end_marker=README_RESULTS_END,
        content=f"""## Latest Run

State: `initialized`
Run id: `{run_id}`

Results are pending. See `STATUS.yaml` for the current machine-readable state.
""",
    )
    write_text(
        experiment_dir / "RUN_CONFIG.yaml",
        f"""experiment_id: {experiment_id}
run_id: {run_id}
start_date: {args.start_date}
end_date: {args.end_date}
sources: {sorted(args.sources)}
cycles: {args.cycles}
leads: {args.leads}
execution_mode: {args.execution_mode}
model_fetch_workers: {args.model_fetch_workers}
model_range_workers: {args.model_range_workers}
model_normalize_workers: {args.model_normalize_workers}
himawari_workers: {args.himawari_workers}
model_range_coalesce_gap_bytes: {args.model_range_coalesce_gap_bytes}
cpu_telemetry: {args.cpu_telemetry}
staging_root: {args.staging_root}
max_staging_gb: {args.max_staging_gb}
stop_free_gb: {args.stop_free_gb}
retain_failed_raw: {args.retain_failed_raw}
dry_run: {args.dry_run}
""",
    )
    write_text(
        experiment_dir / "DATA_MANIFEST.yaml",
        f"""sources:
  - noaa_gfs_public_nomads_or_s3_idx_range
  - noaa_gefs_control_public_nomads_or_s3_idx_range
  - noaa_himawari9_ahi_b13_s0510
  - hkust_envf_hko_radar_image_proxy
date_range_utc:
  start: {args.start_date}
  end: {args.end_date}
raw_retention: deleted_after_db_commit
database_schema: weather_backfill
""",
    )


def write_status(experiment_dir: Path, state: str, summary: dict[str, Any] | None = None) -> None:
    payload = {
        "state": state,
        "updated_at_utc": iso(utc_now()),
        "summary": summary or {},
    }
    write_text(experiment_dir / "STATUS.yaml", json.dumps(payload, indent=2, sort_keys=True))


def write_results(experiment_dir: Path, summary: dict[str, Any]) -> None:
    run_id = str(summary.get("run_id") or "unknown_run")
    run_results_dir = experiment_dir / "results" / "runs" / run_id
    write_json(run_results_dir / "metrics.json", summary)
    write_json(experiment_dir / "results" / "metrics.json", summary)
    lines = [
        "## Latest Run",
        "",
        "### Results",
        "",
        f"State: `{summary.get('status')}`",
        f"Run id: `{summary.get('run_id')}`",
        f"Execution mode: `{summary.get('execution_mode')}`",
        f"Elapsed seconds: `{summary.get('elapsed_seconds')}`",
        f"Date range: `{summary.get('start_date')}` to `{summary.get('end_date')}`",
        "",
        "## Counts",
        "",
        f"- Source issues touched: `{summary.get('source_issues_touched')}`",
        f"- Fetch ok: `{summary.get('fetch_ok')}`",
        f"- Fetch failed: `{summary.get('fetch_failed')}`",
        f"- Normalize ok: `{summary.get('normalize_ok')}`",
        f"- Normalize failed: `{summary.get('normalize_failed')}`",
        f"- Station features upserted: `{summary.get('station_features_upserted')}`",
        f"- Area features upserted: `{summary.get('area_features_upserted')}`",
        f"- Raw bytes deleted: `{summary.get('raw_bytes_deleted')}`",
        f"- Max staging bytes observed: `{summary.get('max_staging_bytes')}`",
        f"- Final staging bytes: `{summary.get('final_staging_bytes')}`",
        f"- Max raw object bytes observed: `{summary.get('max_raw_object_bytes')}`",
        f"- Minimum free disk bytes observed: `{summary.get('min_free_disk_bytes')}`",
        "",
        "## By Source",
        "",
    ]
    for source, row in sorted((summary.get("by_source") or {}).items()):
        lines.append(f"- `{source}`: {row}")
    if summary.get("phase_summary"):
        lines.extend(["", "## Phase Runtime", ""])
        for phase, row in sorted(summary["phase_summary"].items()):
            lines.append(f"- `{phase}`: {row}")
    if summary.get("resource_telemetry"):
        lines.extend(["", "## Resource Telemetry", "", f"- `{summary['resource_telemetry']}`"])
    lines.extend(
        [
            "",
            "### Notes",
            "",
            "- Raw payloads are intentionally not retained.",
            "- Radar is sourced from ENVF historical display imagery and is marked as a proxy, not native exact radar issue metadata.",
            "",
            "### Conclusion",
            "",
            f"Status: `{summary.get('status')}`.",
            "",
            "This run is an acquisition and persistence experiment, not a model promotion experiment. Its",
            "main acceptance criteria are leakage-clock completeness, DB feature persistence, and raw",
            "staging cleanup.",
        ]
    )
    upsert_readme_section(
        experiment_dir / "README.md",
        title=str(summary.get("experiment_id") or "Public Weather Backfill"),
        start_marker=README_RESULTS_START,
        end_marker=README_RESULTS_END,
        content="\n".join(lines),
    )


def increment(summary: dict[str, Any], key: str, amount: int = 1) -> None:
    summary[key] = int(summary.get(key, 0)) + amount


def increment_source(summary: dict[str, Any], source: str, key: str, amount: int = 1) -> None:
    by_source = summary.setdefault("by_source", {})
    row = by_source.setdefault(source, {})
    row[key] = int(row.get(key, 0)) + amount


def add_float(summary: dict[str, Any], key: str, amount: float) -> None:
    summary[key] = float(summary.get(key, 0.0) or 0.0) + float(amount)


def add_source_float(summary: dict[str, Any], source: str, key: str, amount: float) -> None:
    by_source = summary.setdefault("by_source", {})
    row = by_source.setdefault(source, {})
    row[key] = float(row.get(key, 0.0) or 0.0) + float(amount)


def set_max(summary: dict[str, Any], key: str, value: int) -> None:
    summary[key] = max(int(summary.get(key, 0)), int(value))


def set_min(summary: dict[str, Any], key: str, value: int) -> None:
    existing = summary.get(key)
    summary[key] = int(value) if existing is None else min(int(existing), int(value))


def set_source_max(summary: dict[str, Any], source: str, key: str, value: int) -> None:
    by_source = summary.setdefault("by_source", {})
    row = by_source.setdefault(source, {})
    row[key] = max(int(row.get(key, 0)), int(value))


def record_phase_observation(
    summary: dict[str, Any],
    *,
    source: str,
    item_id: str,
    phase_seconds: dict[str, float],
) -> None:
    observation = {"source": source, "item_id": item_id, **phase_seconds}
    summary.setdefault("phase_observations", []).append(observation)
    for phase, seconds in phase_seconds.items():
        if seconds is None:
            continue
        key = f"{phase}_seconds_total"
        add_float(summary, key, float(seconds))
        add_source_float(summary, source, key, float(seconds))


def summarize_phase_observations(summary: dict[str, Any]) -> None:
    observations = summary.get("phase_observations") or []
    if not observations:
        return
    phase_summary: dict[str, dict[str, float | int]] = {}
    for phase in ("fetch", "normalize", "db_write", "total"):
        values = [float(row[phase]) for row in observations if row.get(phase) is not None]
        if not values:
            continue
        ordered = sorted(values)
        phase_summary[phase] = {
            "count": len(ordered),
            "mean_seconds": statistics.fmean(ordered),
            "p50_seconds": ordered[int((len(ordered) - 1) * 0.50)],
            "p90_seconds": ordered[int((len(ordered) - 1) * 0.90)],
            "max_seconds": max(ordered),
        }
    summary["phase_summary"] = phase_summary


def observe_storage(summary: dict[str, Any], experiment_dir: Path, staging_root: Path) -> int:
    staging_bytes = file_size(staging_root)
    free_bytes = drive_free_bytes(experiment_dir)
    set_max(summary, "max_staging_bytes", staging_bytes)
    set_min(summary, "min_free_disk_bytes", free_bytes)
    return staging_bytes


def check_disk_limits(experiment_dir: Path, staging_root: Path, *, max_staging_bytes: int, stop_free_bytes: int) -> int:
    staging_bytes = file_size(staging_root)
    free_bytes = drive_free_bytes(experiment_dir)
    if staging_bytes > max_staging_bytes:
        raise RuntimeError(f"Staging exceeds limit: {staging_bytes} > {max_staging_bytes}")
    if free_bytes < stop_free_bytes:
        raise RuntimeError(f"Free disk below limit: {free_bytes} < {stop_free_bytes}")
    return staging_bytes


class ResourceSampler:
    def __init__(self, watch_dir: Path, interval_seconds: float = 1.0) -> None:
        self.watch_dir = watch_dir
        self.interval_seconds = interval_seconds
        self.cpu_samples: list[float] = []
        self.staging_samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            import psutil  # type: ignore

            self._psutil = psutil
        except Exception:  # noqa: BLE001
            self._psutil = None

    def __enter__(self) -> ResourceSampler:
        if self._psutil is not None:
            self._psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._loop, name="public-weather-resource-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._psutil is not None:
                with suppress(Exception):
                    self.cpu_samples.append(float(self._psutil.cpu_percent(interval=None)))
            with suppress(Exception):
                self.staging_samples.append(file_size(self.watch_dir))
            self._stop.wait(self.interval_seconds)

    def summary(self) -> dict[str, Any]:
        return {
            "cpu_sampler_available": self._psutil is not None,
            "cpu_mean_percent": statistics.fmean(self.cpu_samples) if self.cpu_samples else None,
            "cpu_max_percent": max(self.cpu_samples) if self.cpu_samples else None,
            "staging_max_bytes": max(self.staging_samples) if self.staging_samples else None,
            "staging_end_bytes": file_size(self.watch_dir),
        }


def persist_static_task_result(
    conn: psycopg.Connection[Any],
    *,
    args: argparse.Namespace,
    run_id: str,
    task: public.FetchTask,
    fetch_dict: dict[str, Any],
    metadata: dict[str, Any],
    normalization: dict[str, Any] | None,
    raw_full_path: Path | None,
    task_base_dir: Path,
    day_staging: Path,
    staging_root: Path,
    summary: dict[str, Any],
    phase_seconds: dict[str, float],
) -> None:
    fetch = public.FetchResult(**fetch_dict)
    issue_key = task.item_id
    db_started = time.perf_counter()
    committed = False
    normalization_status = str(normalization.get("status") if normalization else "not_run")
    try:
        metadata = {
            **metadata,
            "execution_mode": getattr(args, "execution_mode", "serial"),
            "phase_seconds": phase_seconds,
        }
        if raw_full_path and os.path.exists(wp(raw_full_path)):
            raw_object_bytes = os.path.getsize(wp(raw_full_path))
            set_max(summary, "max_raw_object_bytes", raw_object_bytes)
            set_source_max(summary, fetch.source, "max_raw_object_bytes", raw_object_bytes)
            staging_bytes_after_fetch = observe_storage(summary, task_base_dir, staging_root)
            set_source_max(summary, fetch.source, "max_staging_bytes_after_fetch", staging_bytes_after_fetch)

        increment(summary, "source_issues_touched")
        increment_source(summary, fetch.source, "source_issues_touched")
        if fetch.status == "ok":
            increment(summary, "fetch_ok")
            increment_source(summary, fetch.source, "fetch_ok")
        else:
            increment(summary, "fetch_failed")
            increment_source(summary, fetch.source, "fetch_failed")

        pending_issue_record = fetch_to_issue_record(
            fetch,
            metadata,
            normalized_status="pending" if fetch.status == "ok" else "not_run",
            retain_failed_raw=args.retain_failed_raw,
        )
        upsert_source_issue(conn, run_id=run_id, experiment_id=args.experiment_id, **pending_issue_record)

        station_count = 0
        area_count = 0
        if fetch.status == "ok":
            if fetch.kind == "model_grib" and normalization is not None:
                station_count, area_count = insert_model_normalization(
                    conn,
                    run_id=run_id,
                    issue_key=issue_key,
                    result=normalization,
                )
            elif fetch.kind == "himawari_hsd" and normalization is not None:
                station_count, area_count = insert_himawari_normalization(
                    conn,
                    run_id=run_id,
                    issue_key=issue_key,
                    result=normalization,
                )
            if normalization_status == "ok":
                increment(summary, "normalize_ok")
                increment_source(summary, fetch.source, "normalize_ok")
            else:
                increment(summary, "normalize_failed")
                increment_source(summary, fetch.source, "normalize_failed")
            increment(summary, "station_features_upserted", station_count)
            increment(summary, "area_features_upserted", area_count)
            increment_source(summary, fetch.source, "station_features_upserted", station_count)
            increment_source(summary, fetch.source, "area_features_upserted", area_count)

        final_issue_record = fetch_to_issue_record(
            fetch,
            metadata,
            normalized_status=normalization_status if fetch.status == "ok" else "not_run",
            retain_failed_raw=args.retain_failed_raw,
        )
        upsert_source_issue(conn, run_id=run_id, experiment_id=args.experiment_id, **final_issue_record)
        insert_ingest_event(
            conn,
            run_id=run_id,
            issue_key=issue_key,
            source=fetch.source,
            event_type="source_issue_processed",
            status=f"{fetch.status}/{normalization_status}",
            message=fetch.error or ((normalization or {}).get("row") or {}).get("normalization_error"),
            elapsed_seconds=sum(float(value) for value in phase_seconds.values()),
            metadata=metadata,
        )
        conn.commit()
        committed = True
    except Exception as exc:  # noqa: BLE001
        conn.rollback()
        increment(summary, "task_errors")
        increment_source(summary, task.source, "task_errors")
        available = to_db_ts(task.availability_proxy_utc)
        if available is not None:
            try:
                upsert_source_issue(
                    conn,
                    run_id=run_id,
                    issue_key=issue_key,
                    source=task.source,
                    product=task.kind,
                    issue_day_utc=to_db_date(task.issue_day_utc or (task.observed_at_utc or "")[:10]),
                    issued_at_utc=to_db_ts(task.issued_at_utc),
                    observed_at_utc=to_db_ts(task.observed_at_utc),
                    valid_at_utc=to_db_ts(task.valid_at_utc or task.observed_at_utc),
                    availability_proxy_utc=available,
                    availability_proxy_method=task.availability_proxy_method,
                    status="error",
                    source_url=task.url,
                    raw_sha256=fetch.sha256,
                    raw_bytes=fetch.bytes,
                    raw_retention_policy="raw_not_retained_fetch_failed",
                    item_id=task.item_id,
                    cycle_hour=task.cycle_hour,
                    lead_hour=task.lead_hour,
                    band=task.band,
                    segment=task.segment,
                    retrieved_at_utc=utc_now(),
                    normalized_status="error",
                    error=f"{type(exc).__name__}: {exc}",
                    metadata={"execution_mode": getattr(args, "execution_mode", "serial"), "phase_seconds": phase_seconds},
                    experiment_id=args.experiment_id,
                )
                insert_ingest_event(
                    conn,
                    run_id=run_id,
                    issue_key=issue_key,
                    source=task.source,
                    event_type="source_issue_error",
                    status="error",
                    message=f"{type(exc).__name__}: {exc}",
                    elapsed_seconds=sum(float(value) for value in phase_seconds.values()),
                )
                conn.commit()
                committed = True
            except Exception:  # noqa: BLE001
                conn.rollback()
        print(f"[error] {task.item_id}: {type(exc).__name__}: {exc}", flush=True)
    finally:
        phase_seconds["db_write"] = time.perf_counter() - db_started
        phase_seconds["total"] = sum(float(value) for value in phase_seconds.values())
        record_phase_observation(summary, source=task.source, item_id=task.item_id, phase_seconds=phase_seconds)
        if committed and raw_full_path and os.path.exists(wp(raw_full_path)) and (
            fetch.status == "ok" or not args.retain_failed_raw
        ):
            deleted = safe_delete_file(raw_full_path, day_staging)
            increment(summary, "raw_bytes_deleted", deleted)
            increment(summary, "raw_files_deleted")
            increment_source(summary, fetch.source, "raw_bytes_deleted", deleted)
            increment_source(summary, fetch.source, "raw_files_deleted")
            observe_storage(summary, task_base_dir, staging_root)


def process_model_or_himawari_task(
    conn: psycopg.Connection[Any],
    *,
    args: argparse.Namespace,
    run_id: str,
    task: public.FetchTask,
    experiment_dir: Path,
    day_staging: Path,
    summary: dict[str, Any],
) -> None:
    issue_key = task.item_id
    if args.skip_existing_complete and source_issue_has_features(conn, issue_key):
        increment(summary, "skipped_existing")
        increment_source(summary, task.source, "skipped_existing")
        return
    raw_full_path: Path | None = None
    fetch: public.FetchResult | None = None
    normalization_status = "not_run"
    started = time.perf_counter()
    try:
        fetch, metadata = fetch_task_to_path(
            task,
            experiment_dir,
            timeout=args.download_timeout_seconds,
            max_attempts=args.max_attempts,
        )
        raw_full_path = experiment_dir / fetch.path if fetch.path else None
        if raw_full_path and os.path.exists(wp(raw_full_path)):
            raw_object_bytes = os.path.getsize(wp(raw_full_path))
            set_max(summary, "max_raw_object_bytes", raw_object_bytes)
            set_source_max(summary, fetch.source, "max_raw_object_bytes", raw_object_bytes)
            staging_bytes_after_fetch = observe_storage(summary, experiment_dir, experiment_dir / "staging")
            set_source_max(summary, fetch.source, "max_staging_bytes_after_fetch", staging_bytes_after_fetch)
        increment(summary, "source_issues_touched")
        increment_source(summary, fetch.source, "source_issues_touched")
        if fetch.status == "ok":
            increment(summary, "fetch_ok")
            increment_source(summary, fetch.source, "fetch_ok")
        else:
            increment(summary, "fetch_failed")
            increment_source(summary, fetch.source, "fetch_failed")

        issue_record = fetch_to_issue_record(
            fetch,
            metadata,
            normalized_status="pending" if fetch.status == "ok" else "not_run",
            retain_failed_raw=args.retain_failed_raw,
        )
        upsert_source_issue(conn, run_id=run_id, experiment_id=args.experiment_id, **issue_record)

        normalization: dict[str, Any] | None = None
        station_count = 0
        area_count = 0
        if fetch.status == "ok":
            if fetch.kind == "model_grib":
                normalization = normalize_model_result_full(experiment_dir, asdict(fetch))
                station_count, area_count = insert_model_normalization(
                    conn,
                    run_id=run_id,
                    issue_key=issue_key,
                    result=normalization,
                )
            elif fetch.kind == "himawari_hsd":
                normalization = normalize_himawari_result_full(experiment_dir, fetch)
                station_count, area_count = insert_himawari_normalization(
                    conn,
                    run_id=run_id,
                    issue_key=issue_key,
                    result=normalization,
                )
            normalization_status = str(normalization.get("status") if normalization else "not_run")
            if normalization_status == "ok":
                increment(summary, "normalize_ok")
                increment_source(summary, fetch.source, "normalize_ok")
            else:
                increment(summary, "normalize_failed")
                increment_source(summary, fetch.source, "normalize_failed")
            increment(summary, "station_features_upserted", station_count)
            increment(summary, "area_features_upserted", area_count)
            increment_source(summary, fetch.source, "station_features_upserted", station_count)
            increment_source(summary, fetch.source, "area_features_upserted", area_count)

        final_issue_record = fetch_to_issue_record(
            fetch,
            metadata,
            normalized_status=normalization_status,
            retain_failed_raw=args.retain_failed_raw,
        )
        upsert_source_issue(conn, run_id=run_id, experiment_id=args.experiment_id, **final_issue_record)
        insert_ingest_event(
            conn,
            run_id=run_id,
            issue_key=issue_key,
            source=fetch.source,
            event_type="source_issue_processed",
            status=f"{fetch.status}/{normalization_status}",
            message=fetch.error,
            elapsed_seconds=time.perf_counter() - started,
            metadata=metadata,
        )
        conn.commit()
    except Exception as exc:  # noqa: BLE001
        conn.rollback()
        increment(summary, "task_errors")
        increment_source(summary, task.source, "task_errors")
        available = to_db_ts(task.availability_proxy_utc)
        if available is not None:
            upsert_source_issue(
                conn,
                run_id=run_id,
                issue_key=issue_key,
                source=task.source,
                product=task.kind,
                issue_day_utc=to_db_date(task.issue_day_utc or (task.observed_at_utc or "")[:10]),
                issued_at_utc=to_db_ts(task.issued_at_utc),
                observed_at_utc=to_db_ts(task.observed_at_utc),
                valid_at_utc=to_db_ts(task.valid_at_utc or task.observed_at_utc),
                availability_proxy_utc=available,
                availability_proxy_method=task.availability_proxy_method,
                status="error",
                source_url=task.url,
                raw_sha256=fetch.sha256 if fetch else None,
                raw_bytes=fetch.bytes if fetch else None,
                raw_retention_policy="raw_not_retained_fetch_failed",
                item_id=task.item_id,
                cycle_hour=task.cycle_hour,
                lead_hour=task.lead_hour,
                band=task.band,
                segment=task.segment,
                retrieved_at_utc=utc_now(),
                normalized_status="error",
                error=f"{type(exc).__name__}: {exc}",
                metadata={},
                experiment_id=args.experiment_id,
            )
            insert_ingest_event(
                conn,
                run_id=run_id,
                issue_key=issue_key,
                source=task.source,
                event_type="source_issue_error",
                status="error",
                message=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=time.perf_counter() - started,
            )
            conn.commit()
        print(f"[error] {task.item_id}: {type(exc).__name__}: {exc}", flush=True)
    finally:
        if raw_full_path and os.path.exists(wp(raw_full_path)) and (
            fetch is None or fetch.status == "ok" or not args.retain_failed_raw
        ):
            deleted = safe_delete_file(raw_full_path, day_staging)
            increment(summary, "raw_bytes_deleted", deleted)
            increment(summary, "raw_files_deleted")
            if fetch:
                increment_source(summary, fetch.source, "raw_bytes_deleted", deleted)
                increment_source(summary, fetch.source, "raw_files_deleted")
            observe_storage(summary, experiment_dir, experiment_dir / "staging")


def skip_existing_static_tasks(
    conn: psycopg.Connection[Any],
    *,
    args: argparse.Namespace,
    tasks: list[public.FetchTask],
    summary: dict[str, Any],
) -> list[public.FetchTask]:
    runnable: list[public.FetchTask] = []
    for task in tasks:
        if args.skip_existing_complete and source_issue_has_features(conn, task.item_id):
            increment(summary, "skipped_existing")
            increment_source(summary, task.source, "skipped_existing")
            continue
        runnable.append(task)
    return runnable


def process_optimized_model_tasks(
    conn: psycopg.Connection[Any],
    *,
    args: argparse.Namespace,
    run_id: str,
    tasks: list[public.FetchTask],
    task_base_dir: Path,
    day_staging: Path,
    staging_root: Path,
    summary: dict[str, Any],
    max_staging_bytes: int,
    stop_free_bytes: int,
) -> None:
    if not tasks:
        return
    pending = list(tasks)
    fetch_futures: dict[Any, public.FetchTask] = {}
    norm_futures: dict[Any, dict[str, Any]] = {}
    completed = 0
    max_norm_backlog = max(1, args.model_normalize_workers * 4)

    def submit_fetches(executor: ThreadPoolExecutor) -> None:
        while pending and len(fetch_futures) < args.model_fetch_workers and len(norm_futures) < max_norm_backlog:
            check_disk_limits(
                task_base_dir,
                staging_root,
                max_staging_bytes=max_staging_bytes,
                stop_free_bytes=stop_free_bytes,
            )
            task = pending.pop(0)
            future = executor.submit(
                fetch_static_worker,
                task_to_payload(task),
                str(task_base_dir),
                args.download_timeout_seconds,
                args.max_attempts,
                True,
                args.model_range_workers,
                args.model_range_coalesce_gap_bytes,
            )
            fetch_futures[future] = task

    with (
        ThreadPoolExecutor(max_workers=args.model_fetch_workers) as fetch_executor,
        ProcessPoolExecutor(max_workers=args.model_normalize_workers) as normalize_executor,
    ):
        submit_fetches(fetch_executor)
        while pending or fetch_futures or norm_futures:
            active = set(fetch_futures) | set(norm_futures)
            if not active:
                submit_fetches(fetch_executor)
                continue
            done, _pending = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                if future in fetch_futures:
                    task = fetch_futures.pop(future)
                    try:
                        fetched = future.result()
                    except Exception as exc:  # noqa: BLE001
                        fetched = {
                            "task": task_to_payload(task),
                            "fetch": error_fetch_from_task(task, f"{type(exc).__name__}: {exc}"),
                            "metadata": {
                                "provider_mode": "optimized_model_fetch_worker",
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                            "raw_full_path": None,
                            "phase_seconds": {"fetch": 0.0},
                        }
                    fetch = public.FetchResult(**fetched["fetch"])
                    if fetch.status == "ok":
                        norm_future = normalize_executor.submit(normalize_model_worker, str(task_base_dir), fetched["fetch"])
                        norm_futures[norm_future] = fetched
                    else:
                        persist_static_task_result(
                            conn,
                            args=args,
                            run_id=run_id,
                            task=task,
                            fetch_dict=fetched["fetch"],
                            metadata=fetched.get("metadata") or {},
                            normalization=None,
                            raw_full_path=Path(fetched["raw_full_path"]) if fetched.get("raw_full_path") else None,
                            task_base_dir=task_base_dir,
                            day_staging=day_staging,
                            staging_root=staging_root,
                            summary=summary,
                            phase_seconds=fetched.get("phase_seconds") or {"fetch": 0.0},
                        )
                        completed += 1
                else:
                    fetched = norm_futures.pop(future)
                    task = task_from_payload(fetched["task"])
                    phase_seconds = dict(fetched.get("phase_seconds") or {})
                    try:
                        normalized = future.result()
                        normalization = normalized.get("normalization")
                        phase_seconds.update(normalized.get("phase_seconds") or {})
                    except Exception as exc:  # noqa: BLE001
                        phase_seconds["normalize"] = 0.0
                        normalization = {
                            "status": "error",
                            "fetch": fetched["fetch"],
                            "station_row": {
                                "source": task.source,
                                "item_id": task.item_id,
                                "availability_proxy_utc": task.availability_proxy_utc,
                                "normalization_error": f"{type(exc).__name__}: {exc}",
                            },
                            "bbox_rows": [],
                        }
                    persist_static_task_result(
                        conn,
                        args=args,
                        run_id=run_id,
                        task=task,
                        fetch_dict=fetched["fetch"],
                        metadata=fetched.get("metadata") or {},
                        normalization=normalization,
                        raw_full_path=Path(fetched["raw_full_path"]) if fetched.get("raw_full_path") else None,
                        task_base_dir=task_base_dir,
                        day_staging=day_staging,
                        staging_root=staging_root,
                        summary=summary,
                        phase_seconds=phase_seconds,
                    )
                    completed += 1
                if completed and completed % args.progress_every == 0:
                    print(
                        f"[progress] optimized-model tasks={completed}/{len(tasks)} "
                        f"fetch_ok={summary.get('fetch_ok', 0)} errors={summary.get('task_errors', 0)}",
                        flush=True,
                    )
            submit_fetches(fetch_executor)


def process_optimized_himawari_tasks(
    conn: psycopg.Connection[Any],
    *,
    args: argparse.Namespace,
    run_id: str,
    tasks: list[public.FetchTask],
    task_base_dir: Path,
    day_staging: Path,
    staging_root: Path,
    summary: dict[str, Any],
    max_staging_bytes: int,
    stop_free_bytes: int,
) -> None:
    if not tasks:
        return
    pending = list(tasks)
    futures: dict[Any, public.FetchTask] = {}
    completed = 0
    max_inflight = max(1, args.himawari_workers * 2)

    def submit(executor: ThreadPoolExecutor) -> None:
        while pending and len(futures) < max_inflight:
            check_disk_limits(
                task_base_dir,
                staging_root,
                max_staging_bytes=max_staging_bytes,
                stop_free_bytes=stop_free_bytes,
            )
            task = pending.pop(0)
            futures[
                executor.submit(
                    fetch_normalize_himawari_worker,
                    task_to_payload(task),
                    str(task_base_dir),
                    args.download_timeout_seconds,
                    args.max_attempts,
                )
            ] = task

    with ThreadPoolExecutor(max_workers=args.himawari_workers) as executor:
        submit(executor)
        while pending or futures:
            done, _pending = wait(set(futures), return_when=FIRST_COMPLETED)
            for future in done:
                task = futures.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = {
                        "task": task_to_payload(task),
                        "fetch": error_fetch_from_task(task, f"{type(exc).__name__}: {exc}"),
                        "metadata": {"provider_mode": "optimized_himawari_worker", "error": f"{type(exc).__name__}: {exc}"},
                        "raw_full_path": None,
                        "phase_seconds": {"fetch": 0.0, "normalize": 0.0},
                        "normalization": None,
                    }
                persist_static_task_result(
                    conn,
                    args=args,
                    run_id=run_id,
                    task=task,
                    fetch_dict=result["fetch"],
                    metadata=result.get("metadata") or {},
                    normalization=result.get("normalization"),
                    raw_full_path=Path(result["raw_full_path"]) if result.get("raw_full_path") else None,
                    task_base_dir=task_base_dir,
                    day_staging=day_staging,
                    staging_root=staging_root,
                    summary=summary,
                    phase_seconds=result.get("phase_seconds") or {"fetch": 0.0, "normalize": 0.0},
                )
                completed += 1
                if completed and completed % args.progress_every == 0:
                    print(
                        f"[progress] optimized-himawari tasks={completed}/{len(tasks)} "
                        f"fetch_ok={summary.get('fetch_ok', 0)} errors={summary.get('task_errors', 0)}",
                        flush=True,
                    )
            submit(executor)


def process_optimized_static_tasks(
    conn: psycopg.Connection[Any],
    *,
    args: argparse.Namespace,
    run_id: str,
    tasks: list[public.FetchTask],
    task_base_dir: Path,
    day_staging: Path,
    staging_root: Path,
    summary: dict[str, Any],
    max_staging_bytes: int,
    stop_free_bytes: int,
) -> None:
    tasks = skip_existing_static_tasks(conn, args=args, tasks=tasks, summary=summary)
    model_tasks = [task for task in tasks if task.kind == "model_grib"]
    himawari_tasks = [task for task in tasks if task.kind == "himawari_hsd"]
    process_optimized_model_tasks(
        conn,
        args=args,
        run_id=run_id,
        tasks=model_tasks,
        task_base_dir=task_base_dir,
        day_staging=day_staging,
        staging_root=staging_root,
        summary=summary,
        max_staging_bytes=max_staging_bytes,
        stop_free_bytes=stop_free_bytes,
    )
    process_optimized_himawari_tasks(
        conn,
        args=args,
        run_id=run_id,
        tasks=himawari_tasks,
        task_base_dir=task_base_dir,
        day_staging=day_staging,
        staging_root=staging_root,
        summary=summary,
        max_staging_bytes=max_staging_bytes,
        stop_free_bytes=stop_free_bytes,
    )


def process_radar_day(
    conn: psycopg.Connection[Any],
    *,
    args: argparse.Namespace,
    run_id: str,
    day: date,
    summary: dict[str, Any],
) -> None:
    manifest, logs = fetch_envf_manifest_for_day(day, args.download_timeout_seconds, args.max_attempts)
    if args.max_radar_frames:
        manifest = manifest[: args.max_radar_frames]
    write_json(args.experiment_dir / "logs" / f"radar_manifest_chunks_{day.isoformat()}.json", logs)
    increment(summary, "radar_manifest_frames", len(manifest))
    increment_source(summary, "envf_hkust_hko_radar", "manifest_frames", len(manifest))
    for index, record in enumerate(manifest, start=1):
        issue_key = issue_key_from_radar(record)
        if args.skip_existing_complete and source_issue_has_features(conn, issue_key):
            increment(summary, "skipped_existing")
            increment_source(summary, "envf_hkust_hko_radar", "skipped_existing")
            continue
        started = time.perf_counter()
        row = fetch_radar_frame(
            record,
            timeout=args.download_timeout_seconds,
            max_attempts=args.max_attempts,
        )
        try:
            station_count, area_count = upsert_radar_issue_and_features(
                conn,
                run_id=run_id,
                experiment_id=args.experiment_id,
                record=row,
            )
            insert_ingest_event(
                conn,
                run_id=run_id,
                issue_key=issue_key,
                source=str(row.get("source")),
                event_type="radar_frame_processed",
                status=str(row.get("image_fetch_status")),
                message=row.get("image_fetch_error"),
                elapsed_seconds=time.perf_counter() - started,
                metadata={"attempts": row.get("image_fetch_attempts")},
            )
            conn.commit()
            increment(summary, "source_issues_touched")
            increment(summary, "station_features_upserted", station_count)
            increment(summary, "area_features_upserted", area_count)
            increment_source(summary, "envf_hkust_hko_radar", "source_issues_touched")
            increment_source(summary, "envf_hkust_hko_radar", "station_features_upserted", station_count)
            increment_source(summary, "envf_hkust_hko_radar", "area_features_upserted", area_count)
            if row.get("image_fetch_status") == "ok":
                increment(summary, "fetch_ok")
                increment(summary, "normalize_ok")
                increment_source(summary, "envf_hkust_hko_radar", "fetch_ok")
                increment_source(summary, "envf_hkust_hko_radar", "normalize_ok")
            else:
                increment(summary, "fetch_failed")
                increment(summary, "normalize_failed")
                increment_source(summary, "envf_hkust_hko_radar", "fetch_failed")
                increment_source(summary, "envf_hkust_hko_radar", "normalize_failed")
        except Exception as exc:  # noqa: BLE001
            conn.rollback()
            increment(summary, "task_errors")
            increment_source(summary, "envf_hkust_hko_radar", "task_errors")
            print(f"[error] radar {issue_key}: {type(exc).__name__}: {exc}", flush=True)
        if index % args.progress_every == 0:
            print(f"[progress] radar day={day} {index}/{len(manifest)}", flush=True)


def run_backfill(args: argparse.Namespace) -> dict[str, Any]:
    experiment_dir = args.experiment_dir
    args.experiment_dir = experiment_dir
    ensure_dir(experiment_dir)
    run_id = (
        f"{args.experiment_id}_{args.start_date:%Y%m%d}_{args.end_date:%Y%m%d}_"
        f"{utc_now():%Y%m%dT%H%M%SZ}_{os.getpid()}"
    )
    initialize_experiment_docs(experiment_dir, args, run_id)
    write_status(experiment_dir, "RUNNING" if not args.dry_run else "DRY_RUN", {"run_id": run_id})

    days = date_span(args.start_date, args.end_date)
    expected = task_counts_for_days(days, args.sources, args.cycles, args.leads)
    summary: dict[str, Any] = {
        "run_id": run_id,
        "experiment_id": args.experiment_id,
        "status": "dry_run" if args.dry_run else "running",
        "execution_mode": args.execution_mode,
        "start_date": args.start_date.isoformat(),
        "end_date": args.end_date.isoformat(),
        "sources": sorted(args.sources),
        "cycles": args.cycles,
        "leads": args.leads,
        "expected": expected,
        "started_at_utc": iso(utc_now()),
        "by_source": {},
    }
    write_json(experiment_dir / "metadata" / "expected_inventory.json", expected)
    if args.dry_run:
        write_results(experiment_dir, {**summary, "elapsed_seconds": 0.0})
        write_status(experiment_dir, "DRY_RUN_COMPLETE", summary)
        return summary

    db_url = args.database_url or os.environ.get("HKG_TMAX_DATABASE_URL") or os.environ.get("HKG_TMAX_DB_DSN")
    if not db_url:
        raise RuntimeError("Set HKG_TMAX_DATABASE_URL, HKG_TMAX_DB_DSN, or pass --database-url.")

    started = time.perf_counter()
    max_staging_bytes = int(args.max_staging_gb * 1024**3)
    stop_free_bytes = int(args.stop_free_gb * 1024**3)
    if args.staging_root is not None:
        task_base_dir = args.staging_root / run_id
    elif args.execution_mode == "optimized":
        task_base_dir = REPO_ROOT / "_weather_backfill_staging" / run_id
    else:
        task_base_dir = experiment_dir
    staging_root = task_base_dir / "staging"
    summary["task_base_dir"] = str(task_base_dir)
    summary["staging_root"] = str(staging_root)
    ensure_dir(task_base_dir)
    ensure_dir(staging_root)

    with psycopg.connect(db_url) as conn:
        apply_schema(conn)
        begin_ingest_run(
            conn,
            run_id=run_id,
            experiment_id=args.experiment_id,
            started_at=utc_now(),
            start_date=args.start_date,
            end_date=args.end_date,
            sources=args.sources,
            config={
                "cycles": args.cycles,
                "leads": args.leads,
                "max_staging_gb": args.max_staging_gb,
                "stop_free_gb": args.stop_free_gb,
                "retain_failed_raw": args.retain_failed_raw,
                "skip_existing_complete": args.skip_existing_complete,
                "experiment_id": args.experiment_id,
                "execution_mode": args.execution_mode,
                "model_fetch_workers": args.model_fetch_workers,
                "model_range_workers": args.model_range_workers,
                "model_normalize_workers": args.model_normalize_workers,
                "himawari_workers": args.himawari_workers,
                "model_range_coalesce_gap_bytes": args.model_range_coalesce_gap_bytes,
                "staging_root": str(staging_root),
            },
        )
        resource_sampler = ResourceSampler(staging_root) if args.cpu_telemetry else None
        if resource_sampler is not None:
            resource_sampler.__enter__()
        try:
            for day_index, day in enumerate(days, start=1):
                day_started = time.perf_counter()
                day_staging = staging_root / day.isoformat()
                ensure_dir(day_staging)
                print(f"[day] {day} ({day_index}/{len(days)})", flush=True)
                tasks = build_tasks_for_day(day, args.sources, args.cycles, args.leads, task_base_dir)
                if args.max_static_tasks:
                    tasks = tasks[: args.max_static_tasks]
                if args.execution_mode == "optimized":
                    check_disk_limits(
                        task_base_dir,
                        staging_root,
                        max_staging_bytes=max_staging_bytes,
                        stop_free_bytes=stop_free_bytes,
                    )
                    process_optimized_static_tasks(
                        conn,
                        args=args,
                        run_id=run_id,
                        tasks=tasks,
                        task_base_dir=task_base_dir,
                        day_staging=day_staging,
                        staging_root=staging_root,
                        max_staging_bytes=max_staging_bytes,
                        stop_free_bytes=stop_free_bytes,
                        summary=summary,
                    )
                else:
                    for index, task in enumerate(tasks, start=1):
                        staging_bytes = check_disk_limits(
                            task_base_dir,
                            staging_root,
                            max_staging_bytes=max_staging_bytes,
                            stop_free_bytes=stop_free_bytes,
                        )
                        set_max(summary, "max_staging_bytes", staging_bytes)
                        set_min(summary, "min_free_disk_bytes", drive_free_bytes(task_base_dir))
                        process_model_or_himawari_task(
                            conn,
                            args=args,
                            run_id=run_id,
                            task=task,
                            experiment_dir=task_base_dir,
                            day_staging=day_staging,
                            summary=summary,
                        )
                        if index % args.progress_every == 0:
                            print(
                                f"[progress] day={day} tasks={index}/{len(tasks)} "
                                f"fetch_ok={summary.get('fetch_ok', 0)} errors={summary.get('task_errors', 0)}",
                                flush=True,
                            )
                if "radar" in args.sources:
                    process_radar_day(conn, args=args, run_id=run_id, day=day, summary=summary)
                if os.path.exists(wp(day_staging)) and (not args.retain_failed_raw or file_size(day_staging) == 0):
                    safe_remove_staging_dir(day_staging, staging_root)
                leftover = file_size(day_staging)
                if leftover:
                    summary.setdefault("leftover_staging_by_day", {})[day.isoformat()] = leftover
                day_elapsed = time.perf_counter() - day_started
                summary.setdefault("day_elapsed_seconds", {})[day.isoformat()] = day_elapsed
                write_json(experiment_dir / "logs" / "live_summary.json", summary)
                print(f"[day-complete] {day} elapsed={day_elapsed:.1f}s", flush=True)

            summary["elapsed_seconds"] = time.perf_counter() - started
            summary["completed_at_utc"] = iso(utc_now())
            summary["final_staging_bytes"] = file_size(staging_root)
            if resource_sampler is not None:
                resource_sampler.__exit__(None, None, None)
                summary["resource_telemetry"] = resource_sampler.summary()
                resource_sampler = None
            summarize_phase_observations(summary)
            summary["status"] = "complete_with_failures" if summary.get("fetch_failed") or summary.get("task_errors") else "complete"
            finish_ingest_run(conn, run_id=run_id, status=str(summary["status"]), summary=summary)
            write_results(experiment_dir, summary)
            write_status(experiment_dir, str(summary["status"]).upper(), summary)
            return summary
        except Exception as exc:  # noqa: BLE001
            summary["elapsed_seconds"] = time.perf_counter() - started
            summary["status"] = "failed"
            summary["error"] = f"{type(exc).__name__}: {exc}"
            summary["final_staging_bytes"] = file_size(staging_root)
            if resource_sampler is not None:
                resource_sampler.__exit__(None, None, None)
                summary["resource_telemetry"] = resource_sampler.summary()
                resource_sampler = None
            summarize_phase_observations(summary)
            finish_ingest_run(conn, run_id=run_id, status="failed", summary=summary, error=summary["error"])
            write_results(experiment_dir, summary)
            write_status(experiment_dir, "FAILED", summary)
            raise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=parse_date, default=date(2026, 6, 25))
    parser.add_argument("--end-date", type=parse_date, default=date(2026, 7, 7))
    parser.add_argument(
        "--sources",
        type=normalize_sources,
        default=normalize_sources("gfs,gefs_control,himawari_b13_s0510,radar"),
    )
    parser.add_argument("--cycles", type=parse_csv_ints, default=[0, 6, 12, 18])
    parser.add_argument("--leads", type=parse_leads, default=list(range(0, 49, 3)))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--execution-mode", choices=["serial", "optimized"], default="serial")
    parser.add_argument("--model-fetch-workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--model-range-workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--model-normalize-workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--himawari-workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--model-range-coalesce-gap-bytes", type=int, default=0)
    parser.add_argument("--cpu-telemetry", action="store_true", default=False)
    parser.add_argument("--staging-root", type=Path, default=None)
    parser.add_argument("--max-staging-gb", type=float, default=4.0)
    parser.add_argument("--stop-free-gb", type=float, default=50.0)
    parser.add_argument("--retain-failed-raw", action="store_true", default=False)
    parser.add_argument("--skip-existing-complete", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--download-timeout-seconds", type=int, default=180)
    parser.add_argument("--max-attempts", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--max-static-tasks", type=int, default=100)
    parser.add_argument("--max-radar-frames", type=int, default=24)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.execute:
        print("DRY RUN: no provider or database calls made; pass --execute after reviewing budgets.")
        return 2
    if args.end_date < args.start_date or (args.end_date - args.start_date).days + 1 > 31:
        parser.error("execution date range must be ordered and no more than 31 days")
    if args.max_static_tasks < 1 or args.max_radar_frames < 1:
        parser.error("request budgets must be positive")
    summary = run_backfill(args)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
