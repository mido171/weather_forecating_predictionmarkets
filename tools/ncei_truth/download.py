from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode

import requests

NCEI_BASE_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
DATASET = "daily-summaries"
DATATYPE = "TMAX"
UNITS = "standard"
FMT = "json"
INCLUDE_ATTRIBUTES = "true"


@dataclass(frozen=True)
class WindowRequest:
    station_id: str
    station_usw: str
    start_date: date
    end_date: date


@dataclass(frozen=True)
class SnapshotResult:
    station_id: str
    station_usw: str
    start_date: date
    end_date: date
    url: str
    response_path: Path
    headers_path: Path
    retrieved_at_utc: str
    http_status: int
    body_sha256: str
    byte_count: int
    skipped_existing: bool


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_url(*, station_usw: str, start_date: date, end_date: date) -> str:
    params = {
        "dataset": DATASET,
        "stations": station_usw,
        "dataTypes": DATATYPE,
        "startDate": start_date.isoformat(),
        "endDate": end_date.isoformat(),
        "units": UNITS,
        "format": FMT,
        "includeAttributes": INCLUDE_ATTRIBUTES,
    }
    return f"{NCEI_BASE_URL}?{urlencode(params)}"


def iter_year_windows(start_date: date, end_date: date) -> Iterable[tuple[date, date]]:
    if end_date < start_date:
        raise ValueError("end_date must be >= start_date")
    for year in range(start_date.year, end_date.year + 1):
        y_start = date(year, 1, 1)
        y_end = date(year, 12, 31)
        s = max(start_date, y_start)
        e = min(end_date, y_end)
        if e >= s:
            yield s, e


def _window_dir(raw_root: Path, station_usw: str, start_date: date, end_date: date) -> Path:
    return (
        raw_root
        / "ncei_ads"
        / "daily-summaries"
        / station_usw
        / f"{start_date.isoformat()}_{end_date.isoformat()}"
    )


def _read_existing_snapshot(window_dir: Path, logger: logging.Logger) -> SnapshotResult | None:
    response_path = window_dir / "response.json"
    headers_path = window_dir / "headers.json"
    retrieved_path = window_dir / "retrieved_at_utc.txt"
    status_path = window_dir / "http_status.txt"
    sha_path = window_dir / "sha256.txt"
    request_path = window_dir / "request_url.txt"
    if not (response_path.exists() and headers_path.exists() and retrieved_path.exists() and status_path.exists() and sha_path.exists() and request_path.exists()):
        return None
    try:
        retrieved_at_utc = retrieved_path.read_text(encoding="utf-8").strip()
        http_status = int(status_path.read_text(encoding="utf-8").strip())
        body_sha256 = sha_path.read_text(encoding="utf-8").strip()
        byte_count = response_path.stat().st_size
        url = request_path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.warning("SNAPSHOT_REUSE_PARSE_FAILED dir=%s err=%s", window_dir, exc)
        return None

    parts = window_dir.name.split("_")
    if len(parts) != 2:
        return None
    s, e = date.fromisoformat(parts[0]), date.fromisoformat(parts[1])
    station_usw = window_dir.parent.name
    return SnapshotResult(
        station_id="",
        station_usw=station_usw,
        start_date=s,
        end_date=e,
        url=url,
        response_path=response_path,
        headers_path=headers_path,
        retrieved_at_utc=retrieved_at_utc,
        http_status=http_status,
        body_sha256=body_sha256,
        byte_count=byte_count,
        skipped_existing=True,
    )


def fetch_snapshot(
    *,
    req: WindowRequest,
    raw_root: Path,
    session: requests.Session,
    logger: logging.Logger,
    timeout_seconds: int = 120,
    skip_if_snapshot_exists: bool = True,
) -> SnapshotResult:
    window_dir = _window_dir(raw_root, req.station_usw, req.start_date, req.end_date)
    window_dir.mkdir(parents=True, exist_ok=True)
    existing = _read_existing_snapshot(window_dir, logger=logger) if skip_if_snapshot_exists else None
    if existing is not None:
        logger.info(
            "NCEI_FETCH_SKIP station=%s usw=%s range=%s..%s bytes=%d",
            req.station_id,
            req.station_usw,
            req.start_date,
            req.end_date,
            existing.byte_count,
        )
        return SnapshotResult(
            station_id=req.station_id,
            station_usw=req.station_usw,
            start_date=req.start_date,
            end_date=req.end_date,
            url=existing.url,
            response_path=existing.response_path,
            headers_path=existing.headers_path,
            retrieved_at_utc=existing.retrieved_at_utc,
            http_status=existing.http_status,
            body_sha256=existing.body_sha256,
            byte_count=existing.byte_count,
            skipped_existing=True,
        )

    url = build_url(station_usw=req.station_usw, start_date=req.start_date, end_date=req.end_date)
    logger.info("NCEI_FETCH_START station=%s usw=%s range=%s..%s", req.station_id, req.station_usw, req.start_date, req.end_date)
    retrieved_at_utc = _utc_now_iso()
    resp = session.get(url, timeout=timeout_seconds)
    body_bytes = resp.content
    sha256 = hashlib.sha256(body_bytes).hexdigest()
    status = int(resp.status_code)

    response_path = window_dir / "response.json"
    headers_path = window_dir / "headers.json"
    (window_dir / "request_url.txt").write_text(url, encoding="utf-8")
    response_path.write_bytes(body_bytes)
    headers_path.write_text(json.dumps(dict(resp.headers), indent=2, sort_keys=True), encoding="utf-8")
    (window_dir / "retrieved_at_utc.txt").write_text(retrieved_at_utc, encoding="utf-8")
    (window_dir / "sha256.txt").write_text(sha256, encoding="utf-8")
    (window_dir / "http_status.txt").write_text(str(status), encoding="utf-8")
    logger.info(
        "NCEI_FETCH_DONE station=%s usw=%s range=%s..%s status=%d bytes=%d sha256=%s",
        req.station_id,
        req.station_usw,
        req.start_date,
        req.end_date,
        status,
        len(body_bytes),
        sha256,
    )
    return SnapshotResult(
        station_id=req.station_id,
        station_usw=req.station_usw,
        start_date=req.start_date,
        end_date=req.end_date,
        url=url,
        response_path=response_path,
        headers_path=headers_path,
        retrieved_at_utc=retrieved_at_utc,
        http_status=status,
        body_sha256=sha256,
        byte_count=len(body_bytes),
        skipped_existing=False,
    )


def append_manifest_row(manifest_csv_path: Path, snapshot: SnapshotResult) -> None:
    manifest_csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "dataset",
        "station_id",
        "station_usw",
        "startDate",
        "endDate",
        "url",
        "retrieved_at_utc",
        "http_status",
        "bytes",
        "sha256",
        "skipped_existing",
    ]
    write_header = not manifest_csv_path.exists()
    with manifest_csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "dataset": DATASET,
                "station_id": snapshot.station_id,
                "station_usw": snapshot.station_usw,
                "startDate": snapshot.start_date.isoformat(),
                "endDate": snapshot.end_date.isoformat(),
                "url": snapshot.url,
                "retrieved_at_utc": snapshot.retrieved_at_utc,
                "http_status": snapshot.http_status,
                "bytes": snapshot.byte_count,
                "sha256": snapshot.body_sha256,
                "skipped_existing": int(snapshot.skipped_existing),
            }
        )


def run_download(
    *,
    station_map: dict[str, str],
    start_date: date,
    end_date: date,
    raw_root: Path,
    manifest_csv_path: Path,
    logger: logging.Logger,
    skip_if_snapshot_exists: bool = True,
) -> list[SnapshotResult]:
    session = requests.Session()
    session.headers.update({"User-Agent": "ncei-truth-downloader/1.0"})
    snapshots: list[SnapshotResult] = []
    for station_id, station_usw in station_map.items():
        for s, e in iter_year_windows(start_date, end_date):
            req = WindowRequest(
                station_id=station_id.strip().upper(),
                station_usw=station_usw.strip().upper(),
                start_date=s,
                end_date=e,
            )
            snap = fetch_snapshot(
                req=req,
                raw_root=raw_root,
                session=session,
                logger=logger,
                skip_if_snapshot_exists=skip_if_snapshot_exists,
            )
            append_manifest_row(manifest_csv_path, snap)
            snapshots.append(snap)
    return snapshots

