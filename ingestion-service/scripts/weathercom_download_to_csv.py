#!/usr/bin/env python3
"""
Download WeatherCom/Wunderground historical observations to CSV files (no DB writes).

Outputs:
  - Raw snapshot evidence per request window
  - Detailed manifest CSV
  - Window-level normalized 30m CSVs
  - Final merged normalized 30m CSV
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode

import pandas as pd
import requests

UTC = timezone.utc
DATE_FMT = "%Y-%m-%d"
API_DATE_FMT = "%Y%m%d"
DEFAULT_BASE_URL = "https://api.weather.com"


@dataclass(frozen=True)
class WindowTask:
    start_date: datetime
    end_date: datetime

    @property
    def start_str(self) -> str:
        return self.start_date.strftime(DATE_FMT)

    @property
    def end_str(self) -> str:
        return self.end_date.strftime(DATE_FMT)

    @property
    def start_api(self) -> str:
        return self.start_date.strftime(API_DATE_FMT)

    @property
    def end_api(self) -> str:
        return self.end_date.strftime(API_DATE_FMT)

    @property
    def key(self) -> str:
        return f"{self.start_str}_{self.end_str}"


@dataclass(frozen=True)
class WindowResult:
    task: WindowTask
    status_code: int
    bytes_len: int
    sha256: str
    retrieved_at_utc: str
    attempts: int
    observations_count: int
    window_csv_path: Optional[Path]
    raw_dir: Path
    request_url: str
    error: str
    skipped_existing: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WeatherCom historical downloader to CSV (no DB).")
    p.add_argument("--location-id", default="KNYC:9:US")
    p.add_argument("--start-date", default="1973-01-01")
    p.add_argument("--end-date", default="2026-12-31")
    p.add_argument("--units", default="e")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--api-key", default=os.environ.get("WEATHERCOM_API_KEY", ""))
    p.add_argument("--chunk-days", type=int, default=31)
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--max-retries", type=int, default=6)
    p.add_argument("--timeout-seconds", type=int, default=60)
    p.add_argument("--out-root", default=r"D:\Ahmed\data\kalshi\weathercom_knyc")
    p.add_argument(
        "--final-csv",
        default=r"D:\Ahmed\data\kalshi\KNYC_observations_30m_wunderground_1973_2026.csv",
    )
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    return p.parse_args()


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def build_windows(start_date: datetime, end_date: datetime, chunk_days: int) -> list[WindowTask]:
    if chunk_days < 1:
        raise ValueError("chunk_days must be >= 1")
    tasks: list[WindowTask] = []
    cur = start_date
    while cur <= end_date:
        w_end = min(end_date, cur + timedelta(days=chunk_days - 1))
        tasks.append(WindowTask(start_date=cur, end_date=w_end))
        cur = w_end + timedelta(days=1)
    return tasks


def safe_location_id(location_id: str) -> str:
    return location_id.replace(":", "_")


def build_request_url(base_url: str, location_id: str, api_key: str, units: str, task: WindowTask) -> str:
    base = base_url.rstrip("/")
    path = f"/v1/location/{location_id}/observations/historical.json"
    params = {
        "apiKey": api_key,
        "units": units,
        "startDate": task.start_api,
        "endDate": task.end_api,
    }
    return f"{base}{path}?{urlencode(params)}"


def redact_api_key_in_url(url: str) -> str:
    return re.sub(r"(apiKey=)[^&]+", r"\1REDACTED", str(url))


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("weathercom_csv_downloader")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(threadName)s | %(message)s")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def response_error_message(resp: requests.Response) -> str:
    body = resp.text.strip()
    if len(body) > 600:
        body = body[:600] + "..."
    return f"HTTP {resp.status_code} body={body}"


def parse_retry_after_seconds(resp: requests.Response) -> Optional[float]:
    v = resp.headers.get("Retry-After")
    if not v:
        return None
    try:
        return max(0.0, float(v))
    except Exception:
        return None


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def to_wdir_cardinal(wdir: Any) -> str:
    if wdir is None:
        return ""
    try:
        deg = float(wdir) % 360.0
    except Exception:
        return ""
    names = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = int((deg + 11.25) // 22.5) % 16
    return names[idx]


def normalize_to_30m_rows(location_id: str, payload: dict[str, Any]) -> pd.DataFrame:
    obs = payload.get("observations")
    if not isinstance(obs, list) or not obs:
        return pd.DataFrame(
            columns=[
                "request_location_id",
                "valid_time_utc",
                "temp",
                "dew_pt",
                "rh",
                "pressure",
                "vis",
                "wspd",
                "wdir",
                "gust",
                "precip_hrly",
                "clds",
                "wx_phrase",
                "uv_index",
                "uv_desc",
                "wdir_cardinal",
            ]
        )

    rows: list[dict[str, Any]] = []
    for item in obs:
        if not isinstance(item, dict):
            continue
        valid_gmt = item.get("valid_time_gmt")
        if valid_gmt is None:
            continue
        try:
            dt = datetime.fromtimestamp(int(valid_gmt), tz=UTC)
        except Exception:
            continue
        dt_30m = dt.replace(minute=(dt.minute // 30) * 30, second=0, microsecond=0)
        wdir = item.get("wdir")
        wdir_card = item.get("wdir_cardinal") or to_wdir_cardinal(wdir)
        rows.append(
            {
                "request_location_id": location_id,
                "valid_time_utc": dt_30m.strftime("%Y-%m-%d %H:%M:%S"),
                "temp": item.get("temp"),
                "dew_pt": item.get("dewPt"),
                "rh": item.get("rh"),
                "pressure": item.get("pressure"),
                "vis": item.get("vis"),
                "wspd": item.get("wspd"),
                "wdir": wdir,
                "gust": item.get("gust"),
                "precip_hrly": item.get("precip_hrly"),
                "clds": item.get("clds"),
                "wx_phrase": item.get("wx_phrase"),
                "uv_index": item.get("uv_index"),
                "uv_desc": item.get("uv_desc"),
                "wdir_cardinal": wdir_card,
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values("valid_time_utc").drop_duplicates(["valid_time_utc"], keep="last")
    return df


def fetch_window(
    *,
    session: requests.Session,
    logger: logging.Logger,
    base_url: str,
    api_key: str,
    location_id: str,
    units: str,
    out_root: Path,
    task: WindowTask,
    timeout_seconds: int,
    max_retries: int,
    skip_existing: bool,
) -> WindowResult:
    loc_safe = safe_location_id(location_id)
    raw_dir = out_root / "raw" / loc_safe / task.key
    win_csv_dir = out_root / "window_csv"
    window_csv_path = win_csv_dir / f"{loc_safe}_{task.key}.csv"
    request_url = build_request_url(base_url, location_id, api_key, units, task)
    request_url_redacted = redact_api_key_in_url(request_url)

    response_json_path = raw_dir / "response.json"
    existing_status_path = raw_dir / "http_status.txt"
    existing_retrieved_path = raw_dir / "retrieved_at_utc.txt"
    existing_sha_path = raw_dir / "sha256.txt"

    if skip_existing and response_json_path.exists() and existing_status_path.exists():
        status_val = existing_status_path.read_text(encoding="utf-8").strip()
        if status_val == "200":
            body_bytes = response_json_path.read_bytes()
            sha = hashlib.sha256(body_bytes).hexdigest()
            retrieved = existing_retrieved_path.read_text(encoding="utf-8").strip() if existing_retrieved_path.exists() else ""
            if existing_sha_path.exists() and existing_sha_path.read_text(encoding="utf-8").strip() != sha:
                logger.warning("SHA mismatch for existing raw window=%s; using recomputed sha.", task.key)
            try:
                payload = json.loads(body_bytes.decode("utf-8", errors="replace"))
            except Exception as exc:
                return WindowResult(
                    task=task,
                    status_code=0,
                    bytes_len=len(body_bytes),
                    sha256=sha,
                    retrieved_at_utc=retrieved,
                    attempts=0,
                    observations_count=0,
                    window_csv_path=None,
                    raw_dir=raw_dir,
                    request_url=request_url_redacted,
                    error=f"existing_json_parse_error: {exc}",
                    skipped_existing=True,
                )
            df = normalize_to_30m_rows(location_id, payload)
            win_csv_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(window_csv_path, index=False)
            logger.info(
                "SKIP_EXISTING window=%s status=200 bytes=%d obs=%d csv_rows=%d",
                task.key,
                len(body_bytes),
                len(payload.get("observations", [])) if isinstance(payload.get("observations"), list) else 0,
                len(df),
            )
            return WindowResult(
                task=task,
                status_code=200,
                bytes_len=len(body_bytes),
                sha256=sha,
                retrieved_at_utc=retrieved,
                attempts=0,
                observations_count=len(payload.get("observations", [])) if isinstance(payload.get("observations"), list) else 0,
                window_csv_path=window_csv_path,
                raw_dir=raw_dir,
                request_url=request_url_redacted,
                error="",
                skipped_existing=True,
            )

    retries = max(0, max_retries)
    for attempt in range(1, retries + 2):
        t0 = time.time()
        logger.info("REQUEST_START window=%s attempt=%d/%d url=%s", task.key, attempt, retries + 1, request_url_redacted)
        err = ""
        status_code = 0
        body_bytes = b""
        headers: dict[str, str] = {}
        try:
            resp = session.get(request_url, timeout=timeout_seconds)
            status_code = int(resp.status_code)
            headers = dict(resp.headers)
            body_bytes = resp.content
            elapsed = time.time() - t0
            logger.info(
                "REQUEST_DONE window=%s attempt=%d status=%d elapsed_s=%.3f bytes=%d",
                task.key,
                attempt,
                status_code,
                elapsed,
                len(body_bytes),
            )
            retryable = status_code in {429, 500, 502, 503, 504}
            if status_code == 200:
                retrieved = utc_now_iso()
                sha = hashlib.sha256(body_bytes).hexdigest()
                raw_dir.mkdir(parents=True, exist_ok=True)
                response_json_path.write_bytes(body_bytes)
                write_text(raw_dir / "request_url.txt", request_url_redacted + "\n")
                write_text(raw_dir / "headers.json", json.dumps(headers, indent=2, ensure_ascii=True) + "\n")
                write_text(raw_dir / "retrieved_at_utc.txt", retrieved + "\n")
                write_text(raw_dir / "sha256.txt", sha + "\n")
                write_text(raw_dir / "http_status.txt", "200\n")

                try:
                    payload = json.loads(body_bytes.decode("utf-8", errors="replace"))
                except Exception as exc:
                    return WindowResult(
                        task=task,
                        status_code=200,
                        bytes_len=len(body_bytes),
                        sha256=sha,
                        retrieved_at_utc=retrieved,
                        attempts=attempt,
                        observations_count=0,
                        window_csv_path=None,
                        raw_dir=raw_dir,
                        request_url=request_url_redacted,
                        error=f"json_parse_error: {exc}",
                        skipped_existing=False,
                    )
                obs_count = len(payload.get("observations", [])) if isinstance(payload.get("observations"), list) else 0
                df = normalize_to_30m_rows(location_id, payload)
                win_csv_dir.mkdir(parents=True, exist_ok=True)
                df.to_csv(window_csv_path, index=False)
                logger.info(
                    "WINDOW_NORMALIZED window=%s obs_count=%d csv_rows=%d csv_path=%s",
                    task.key,
                    obs_count,
                    len(df),
                    window_csv_path,
                )
                return WindowResult(
                    task=task,
                    status_code=200,
                    bytes_len=len(body_bytes),
                    sha256=sha,
                    retrieved_at_utc=retrieved,
                    attempts=attempt,
                    observations_count=obs_count,
                    window_csv_path=window_csv_path,
                    raw_dir=raw_dir,
                    request_url=request_url_redacted,
                    error="",
                    skipped_existing=False,
                )

            err = response_error_message(resp)
            if retryable and attempt <= retries:
                retry_after = parse_retry_after_seconds(resp)
                sleep_s = retry_after if retry_after is not None else min(60.0, 1.5 * (2 ** (attempt - 1)))
                logger.warning(
                    "REQUEST_RETRY window=%s attempt=%d status=%d sleep_s=%.2f error=%s",
                    task.key,
                    attempt,
                    status_code,
                    sleep_s,
                    err,
                )
                time.sleep(sleep_s)
                continue
            break
        except Exception as exc:
            elapsed = time.time() - t0
            err = f"exception: {exc}"
            logger.warning(
                "REQUEST_EXCEPTION window=%s attempt=%d elapsed_s=%.3f error=%s",
                task.key,
                attempt,
                elapsed,
                err,
            )
            if attempt <= retries:
                sleep_s = min(60.0, 1.5 * (2 ** (attempt - 1)))
                logger.warning("REQUEST_RETRY window=%s attempt=%d sleep_s=%.2f", task.key, attempt, sleep_s)
                time.sleep(sleep_s)
                continue
            status_code = 0
            body_bytes = b""
            headers = {}
            break

    retrieved = utc_now_iso()
    sha = hashlib.sha256(body_bytes).hexdigest() if body_bytes else ""
    raw_dir.mkdir(parents=True, exist_ok=True)
    if body_bytes:
        response_json_path.write_bytes(body_bytes)
    write_text(raw_dir / "request_url.txt", request_url_redacted + "\n")
    write_text(raw_dir / "headers.json", json.dumps(headers, indent=2, ensure_ascii=True) + "\n")
    write_text(raw_dir / "retrieved_at_utc.txt", retrieved + "\n")
    write_text(raw_dir / "sha256.txt", sha + "\n")
    write_text(raw_dir / "http_status.txt", f"{status_code}\n")
    logger.error("WINDOW_FAILED window=%s status=%d error=%s", task.key, status_code, err)
    return WindowResult(
        task=task,
        status_code=status_code,
        bytes_len=len(body_bytes),
        sha256=sha,
        retrieved_at_utc=retrieved,
        attempts=retries + 1,
        observations_count=0,
        window_csv_path=None,
        raw_dir=raw_dir,
        request_url=request_url_redacted,
        error=err,
        skipped_existing=False,
    )


def merge_windows_to_final_csv(window_paths: list[Path], final_csv_path: Path, logger: logging.Logger) -> int:
    if not window_paths:
        final_csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(final_csv_path, index=False)
        return 0

    frames: list[pd.DataFrame] = []
    for p in window_paths:
        try:
            df = pd.read_csv(p)
            if not df.empty:
                frames.append(df)
        except Exception as exc:
            logger.warning("MERGE_READ_FAIL path=%s error=%s", p, exc)
    if not frames:
        final_csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(final_csv_path, index=False)
        return 0

    all_df = pd.concat(frames, ignore_index=True)
    all_df["valid_time_utc"] = pd.to_datetime(all_df["valid_time_utc"], errors="coerce")
    all_df = all_df[all_df["valid_time_utc"].notna()].copy()
    all_df = all_df.sort_values(["request_location_id", "valid_time_utc"]).drop_duplicates(
        ["request_location_id", "valid_time_utc"], keep="last"
    )
    all_df["valid_time_utc"] = all_df["valid_time_utc"].dt.strftime("%Y-%m-%d %H:%M:%S")
    final_csv_path.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(final_csv_path, index=False)
    return len(all_df)


def write_manifest(manifest_path: Path, rows: list[WindowResult]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "location_id",
                "start_date",
                "end_date",
                "request_url",
                "retrieved_at_utc",
                "status_code",
                "bytes",
                "sha256",
                "attempts",
                "observations_count",
                "window_csv_path",
                "raw_dir",
                "skipped_existing",
                "error",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.request_url.split("/v1/location/")[1].split("/observations")[0],
                    r.task.start_str,
                    r.task.end_str,
                    r.request_url,
                    r.retrieved_at_utc,
                    r.status_code,
                    r.bytes_len,
                    r.sha256,
                    r.attempts,
                    r.observations_count,
                    str(r.window_csv_path) if r.window_csv_path else "",
                    str(r.raw_dir),
                    int(r.skipped_existing),
                    r.error,
                ]
            )


def main() -> int:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Missing API key. Pass --api-key or set WEATHERCOM_API_KEY.")

    start_date = datetime.strptime(args.start_date, DATE_FMT).replace(tzinfo=UTC)
    end_date = datetime.strptime(args.end_date, DATE_FMT).replace(tzinfo=UTC)
    if end_date < start_date:
        raise SystemExit("--end-date must be >= --start-date")

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(args.out_root).resolve()
    run_root = out_root / "runs" / run_id
    log_path = run_root / "logs" / f"weathercom_download_{run_id}.log"
    logger = setup_logger(log_path)

    logger.info("RUN_START run_id=%s", run_id)
    logger.info(
        "CONFIG location_id=%s start=%s end=%s chunk_days=%d workers=%d retries=%d timeout_s=%d out_root=%s final_csv=%s skip_existing=%s",
        args.location_id,
        args.start_date,
        args.end_date,
        args.chunk_days,
        args.max_workers,
        args.max_retries,
        args.timeout_seconds,
        out_root,
        args.final_csv,
        args.skip_existing,
    )

    tasks = build_windows(start_date, end_date, int(args.chunk_days))
    total = len(tasks)
    logger.info("WINDOWS_BUILT total=%d", total)

    session = requests.Session()
    results: list[WindowResult] = []
    lock = threading.Lock()
    completed = 0
    success = 0
    failed = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as ex:
        fut_map = {
            ex.submit(
                fetch_window,
                session=session,
                logger=logger,
                base_url=args.base_url,
                api_key=args.api_key,
                location_id=args.location_id,
                units=args.units,
                out_root=run_root,
                task=task,
                timeout_seconds=int(args.timeout_seconds),
                max_retries=int(args.max_retries),
                skip_existing=bool(args.skip_existing),
            ): task
            for task in tasks
        }
        for fut in as_completed(fut_map):
            task = fut_map[fut]
            try:
                res = fut.result()
            except Exception as exc:
                logger.exception("FUTURE_FAIL window=%s error=%s", task.key, exc)
                res = WindowResult(
                    task=task,
                    status_code=0,
                    bytes_len=0,
                    sha256="",
                    retrieved_at_utc=utc_now_iso(),
                    attempts=0,
                    observations_count=0,
                    window_csv_path=None,
                    raw_dir=run_root / "raw" / safe_location_id(args.location_id) / task.key,
                    request_url=build_request_url(args.base_url, args.location_id, args.api_key, args.units, task),
                    error=f"future_exception: {exc}",
                    skipped_existing=False,
                )
            with lock:
                results.append(res)
                completed += 1
                if res.status_code == 200:
                    success += 1
                else:
                    failed += 1
                if res.skipped_existing:
                    skipped += 1
                if completed % 25 == 0 or completed == total:
                    logger.info(
                        "PROGRESS completed=%d/%d success=%d failed=%d skipped_existing=%d",
                        completed,
                        total,
                        success,
                        failed,
                        skipped,
                    )

    manifest_path = run_root / "manifest" / f"manifest_{run_id}.csv"
    write_manifest(manifest_path, sorted(results, key=lambda x: (x.task.start_date, x.task.end_date)))
    logger.info("MANIFEST_WRITTEN path=%s rows=%d", manifest_path, len(results))

    window_paths = [r.window_csv_path for r in results if r.window_csv_path and r.status_code == 200]
    final_csv = Path(args.final_csv).resolve()
    final_rows = merge_windows_to_final_csv(window_paths, final_csv, logger)
    logger.info("FINAL_CSV_WRITTEN path=%s rows=%d", final_csv, final_rows)

    summary = {
        "run_id": run_id,
        "location_id": args.location_id,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "chunk_days": int(args.chunk_days),
        "max_workers": int(args.max_workers),
        "max_retries": int(args.max_retries),
        "total_windows": total,
        "success_windows": success,
        "failed_windows": failed,
        "skipped_existing_windows": skipped,
        "manifest_path": str(manifest_path),
        "final_csv_path": str(final_csv),
        "final_csv_rows": final_rows,
        "log_path": str(log_path),
        "finished_at_utc": utc_now_iso(),
    }
    summary_path = run_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("RUN_DONE summary=%s", summary_path)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
