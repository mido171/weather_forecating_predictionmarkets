#!/usr/bin/env python3
"""
High-throughput IEM MOS archive downloader for KNYC.

This script fetches historical MOS runs (GFS/NAM) from IEM in year chunks,
stores raw response snapshots, and exports a normalized wide CSV suitable for
runtime-slice experiments.

Default output root:
  D:\Ahmed\data\kalshi\training_data
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests


IEM_BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"
KNOWN_VARS = [
    "tmp",
    "dpt",
    "cld",
    "sky",
    "wdr",
    "wsp",
    "gst",
    "p06",
    "p12",
    "t06",
    "t12",
    "q06",
    "q12",
    "n_x",
    "n_n",
    "cig",
    "vis",
]

CANONICAL_COLUMNS = [
    "station_id",
    "model",
    "year",
    "runtime_utc",
    "forecast_time_utc",
    "retrieved_at_utc",
    "response_sha256",
    *KNOWN_VARS,
    *[f"{k}_raw" for k in KNOWN_VARS],
]


@dataclass(frozen=True)
class Task:
    model: str
    year: int
    station: str
    out_dir: Path

    @property
    def start_utc(self) -> str:
        return f"{self.year}-01-01T00:00Z"

    @property
    def end_utc(self) -> str:
        return f"{self.year}-12-31T23:59Z"

    @property
    def stem(self) -> str:
        return f"{self.station}_{self.model}_{self.year}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def to_utc_iso_from_epoch_ms(value: Optional[object]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        ms = int(value)
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    text = str(value).strip()
    if not text:
        return None
    # Epoch millis in string form.
    if text.isdigit():
        ms = int(text)
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    # ISO form from IEM often looks like "2025-01-01T00:00:00.000" (naive UTC).
    try:
        norm = text.replace("Z", "")
        dt = datetime.fromisoformat(norm)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    except Exception:
        return None


def parse_numeric(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"M", "T", "NA", "NAN"}:
        return None
    # Some MOS values can appear as "12/34"; keep first token.
    if "/" in text:
        text = text.split("/", 1)[0].strip()
    try:
        return float(text)
    except Exception:
        return None


def request_json_with_retry(
    session: requests.Session,
    params: Dict[str, str],
    max_retries: int,
    timeout_sec: int,
    backoff_base_sec: float,
) -> bytes:
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(IEM_BASE_URL, params=params, timeout=timeout_sec)
            if resp.status_code >= 500:
                raise RuntimeError(f"HTTP {resp.status_code}")
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")
            payload = resp.content
            if not payload:
                raise RuntimeError("empty payload")
            return payload
        except Exception as exc:
            last_err = exc
            sleep_s = backoff_base_sec * (2 ** (attempt - 1))
            logging.warning(
                "Request failed attempt=%d/%d model=%s station=%s sts=%s ets=%s err=%s sleep=%.2fs",
                attempt,
                max_retries,
                params.get("model"),
                params.get("station"),
                params.get("sts"),
                params.get("ets"),
                exc,
                sleep_s,
            )
            if attempt < max_retries:
                time.sleep(sleep_s)
    raise RuntimeError(f"request failed after retries: {last_err}")


def flatten_payload(
    payload_json: List[dict],
    station: str,
    model: str,
    year: int,
    retrieved_at_utc: str,
    response_sha256: str,
) -> pd.DataFrame:
    rows: List[dict] = []
    for item in payload_json:
        runtime_utc = to_utc_iso_from_epoch_ms(item.get("runtime"))
        ftime_utc = to_utc_iso_from_epoch_ms(item.get("ftime"))
        row = {
            "station_id": station,
            "model": model,
            "year": year,
            "runtime_utc": runtime_utc,
            "forecast_time_utc": ftime_utc,
            "retrieved_at_utc": retrieved_at_utc,
            "response_sha256": response_sha256,
        }
        for k in KNOWN_VARS:
            row[k] = parse_numeric(item.get(k))
            raw_key = f"{k}_raw"
            value = item.get(k)
            row[raw_key] = None if value is None else str(value)
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    df = pd.DataFrame(rows)
    return df.reindex(columns=CANONICAL_COLUMNS)


def run_task(task: Task, args: argparse.Namespace) -> Tuple[Task, int, int, str]:
    t0 = time.perf_counter()
    params = {
        "station": task.station,
        "model": task.model,
        "sts": task.start_utc,
        "ets": task.end_utc,
        "format": "json",
    }
    yearly_dir = task.out_dir / "mos_yearly"
    raw_dir = task.out_dir / "mos_raw"
    meta_dir = task.out_dir / "mos_meta"
    yearly_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    yearly_path = yearly_dir / f"{task.stem}.csv.gz"
    raw_path = raw_dir / f"{task.stem}.json.gz"
    meta_path = meta_dir / f"{task.stem}.json"

    if args.resume and yearly_path.exists() and meta_path.exists():
        try:
            df = pd.read_csv(yearly_path, nrows=5)
            ncols = len(df.columns)
            # Get row count quickly.
            row_count = sum(1 for _ in gzip.open(yearly_path, "rt", encoding="utf-8")) - 1
            logging.info(
                "SKIP existing model=%s year=%d rows=%d cols=%d file=%s",
                task.model,
                task.year,
                row_count,
                ncols,
                yearly_path,
            )
            return task, row_count, ncols, "skipped"
        except Exception:
            logging.warning("Resume check failed for %s, re-downloading", yearly_path)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "weather-forecasting-predictionmarkets MOS KNYC downloader",
            "Accept": "application/json",
        }
    )

    logging.info(
        "START model=%s year=%d station=%s sts=%s ets=%s",
        task.model,
        task.year,
        task.station,
        task.start_utc,
        task.end_utc,
    )
    raw_bytes = request_json_with_retry(
        session=session,
        params=params,
        max_retries=args.max_retries,
        timeout_sec=args.timeout_sec,
        backoff_base_sec=args.backoff_base_sec,
    )
    response_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    retrieved_at_utc = utc_now_iso()

    with gzip.open(raw_path, "wb") as f:
        f.write(raw_bytes)

    payload_json = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(payload_json, list):
        raise RuntimeError(f"Unexpected JSON type for model={task.model} year={task.year}")

    df = flatten_payload(
        payload_json=payload_json,
        station=task.station,
        model=task.model,
        year=task.year,
        retrieved_at_utc=retrieved_at_utc,
        response_sha256=response_sha256,
    )

    # Deduplicate at runtime+forecast_time for safety.
    if not df.empty:
        df = df.sort_values(["runtime_utc", "forecast_time_utc"]).drop_duplicates(
            subset=["runtime_utc", "forecast_time_utc"], keep="last"
        )

    df.to_csv(yearly_path, index=False, compression="gzip")

    runtime_hours = {}
    if not df.empty:
        dt = pd.to_datetime(df["runtime_utc"], utc=True, errors="coerce")
        runtime_hours = (
            dt.dt.hour.value_counts(dropna=True).sort_index().astype(int).to_dict()
        )
    meta = {
        "station_id": task.station,
        "model": task.model,
        "year": task.year,
        "request_params": params,
        "retrieved_at_utc": retrieved_at_utc,
        "response_sha256": response_sha256,
        "row_count": int(len(df)),
        "runtime_hour_counts": runtime_hours,
        "yearly_csv_gz": str(yearly_path),
        "raw_json_gz": str(raw_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - t0
    logging.info(
        "DONE model=%s year=%d rows=%d secs=%.2f yearly=%s raw=%s",
        task.model,
        task.year,
        len(df),
        elapsed,
        yearly_path,
        raw_path,
    )
    return task, int(len(df)), int(len(df.columns)), "downloaded"


def merge_yearly_files(out_dir: Path, station: str, start_year: int, end_year: int) -> Path:
    yearly_dir = out_dir / "mos_yearly"
    paths = sorted(yearly_dir.glob(f"{station}_*.csv.gz"))
    if not paths:
        raise RuntimeError("No yearly MOS files found to merge")

    merged_path = out_dir / f"{station}_mos_archive_{start_year}_{end_year}.csv.gz"
    total_rows = 0
    with gzip.open(merged_path, "wt", encoding="utf-8", newline="") as out_f:
        first = True
        for p in paths:
            df = pd.read_csv(p)
            df = df.reindex(columns=CANONICAL_COLUMNS)
            total_rows += len(df)
            df.to_csv(out_f, index=False, header=first)
            first = False
    logging.info("MERGED rows=%d files=%d into %s", total_rows, len(paths), merged_path)
    return merged_path


def build_manifest(
    out_dir: Path,
    results: List[Tuple[Task, int, int, str]],
    merged_path: Path,
    args: argparse.Namespace,
) -> Path:
    manifest_rows = []
    for task, row_count, col_count, status in results:
        manifest_rows.append(
            {
                "station_id": task.station,
                "model": task.model,
                "year": task.year,
                "status": status,
                "row_count": row_count,
                "column_count": col_count,
                "yearly_file": str(out_dir / "mos_yearly" / f"{task.stem}.csv.gz"),
                "raw_file": str(out_dir / "mos_raw" / f"{task.stem}.json.gz"),
                "meta_file": str(out_dir / "mos_meta" / f"{task.stem}.json"),
            }
        )
    manifest_df = pd.DataFrame(manifest_rows).sort_values(["model", "year"])
    manifest_path = out_dir / f"{args.station}_mos_download_manifest_{args.start_year}_{args.end_year}.csv"
    manifest_df.to_csv(manifest_path, index=False)

    run_meta = {
        "generated_at_utc": utc_now_iso(),
        "station_id": args.station,
        "models": args.models,
        "start_year": args.start_year,
        "end_year": args.end_year,
        "threads": args.threads,
        "timeout_sec": args.timeout_sec,
        "max_retries": args.max_retries,
        "merged_file": str(merged_path),
        "manifest_file": str(manifest_path),
        "total_rows": int(manifest_df["row_count"].sum()),
        "task_count": int(len(manifest_df)),
    }
    run_meta_path = out_dir / f"{args.station}_mos_download_run_meta_{args.start_year}_{args.end_year}.json"
    run_meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    logging.info("WROTE manifest=%s run_meta=%s", manifest_path, run_meta_path)
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download IEM MOS archive for KNYC with high concurrency.")
    parser.add_argument("--station", default="KNYC", help="Station id (default: KNYC)")
    parser.add_argument("--models", nargs="+", default=["GFS", "NAM"], help="Models to fetch (default: GFS NAM)")
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--backoff-base-sec", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        default=r"D:\Ahmed\data\kalshi\training_data",
        help="Output directory for MOS files",
    )
    parser.add_argument("--resume", action="store_true", help="Skip tasks with existing yearly output")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    station = args.station.strip().upper()
    models = [m.strip().upper() for m in args.models]
    years = list(range(args.start_year, args.end_year + 1))
    tasks = [Task(model=m, year=y, station=station, out_dir=out_dir) for m in models for y in years]

    logging.info(
        "MOS download start station=%s models=%s years=%d..%d threads=%d tasks=%d out_dir=%s",
        station,
        models,
        args.start_year,
        args.end_year,
        args.threads,
        len(tasks),
        out_dir,
    )

    results: List[Tuple[Task, int, int, str]] = []
    with cf.ThreadPoolExecutor(max_workers=args.threads, thread_name_prefix="mos-dl") as executor:
        future_map = {executor.submit(run_task, task, args): task for task in tasks}
        for fut in cf.as_completed(future_map):
            task = future_map[fut]
            try:
                result = fut.result()
                results.append(result)
            except Exception as exc:
                logging.exception("FAILED model=%s year=%d err=%s", task.model, task.year, exc)
                raise

    merged_path = merge_yearly_files(out_dir=out_dir, station=station, start_year=args.start_year, end_year=args.end_year)
    manifest_path = build_manifest(out_dir=out_dir, results=results, merged_path=merged_path, args=args)

    logging.info("MOS download complete merged=%s manifest=%s", merged_path, manifest_path)


if __name__ == "__main__":
    main()
