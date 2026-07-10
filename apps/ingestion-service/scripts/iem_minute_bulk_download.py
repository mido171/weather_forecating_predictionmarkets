#!/usr/bin/env python3
"""
Bulk downloader for IEM ASOS 1-minute tmpf data.

Designed to mirror the KMIA dataset structure:
  data/iem_minute_data/<STATION>/tmpf/UTC/yearly/<STATION>_tmpf_1min_UTC_<YEAR>.csv

This script runs multi-threaded and writes per-station manifests + logs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

VAR = "tmpf"
TZ = "UTC"
ENDPOINT = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py"
MAX_ATTEMPTS = 6
BASE_SLEEP = 3
INTER_TASK_SLEEP = 0.2

DEFAULT_YEAR_START = 2002
DEFAULT_YEAR_END = 2026

USER_AGENT = "weather-forecasting-predictionmarkets (IEM ASOS1min downloader)"


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _build_url(station: str, sts: str, ets: str, tz: str) -> str:
    params = {
        "station": station,
        "vars": VAR,
        "sts": sts,
        "ets": ets,
        "what": "download",
        "tz": tz,
        "delim": "comma",
    }
    return ENDPOINT + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)


def _validate_csv_header(first_line: str) -> bool:
    line = (first_line or "").strip()
    if not line:
        return False
    low = line.lower()
    if "<html" in low or "<!doctype" in low:
        return False
    return line.startswith("station,")


def _valid_column_index(header: str) -> int:
    cols = header.split(",")
    for i, c in enumerate(cols):
        if c.startswith("valid(") or c == "valid":
            return i
    return 1


def _parse_utc_ts(ts: str) -> dt.datetime | None:
    if not ts:
        return None
    ts = ts.strip()
    try:
        dt_obj = dt.datetime.strptime(ts, "%Y-%m-%d %H:%M")
        return dt_obj.replace(tzinfo=dt.timezone.utc)
    except Exception:
        return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _scan_csv(path: Path) -> Tuple[int, str, str]:
    rows = 0
    first_ts = ""
    last_ts = ""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
        idx = _valid_column_index(header)
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split(",")
            if len(parts) > idx:
                ts = parts[idx]
                if not first_ts:
                    first_ts = ts
                last_ts = ts
            rows += 1
    return rows, first_ts, last_ts


def _download_with_retry(url: str, out_path: Path, log) -> None:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            log(f"GET attempt {attempt}/{MAX_ATTEMPTS} -> {url}")
            tmp_path = out_path.with_suffix(out_path.suffix + ".partial")
            if tmp_path.exists():
                tmp_path.unlink()
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=600) as resp, tmp_path.open("wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            # Validate header
            with tmp_path.open("r", encoding="utf-8", errors="replace") as f:
                first_line = f.readline()
            if not _validate_csv_header(first_line):
                raise RuntimeError(f"Downloaded file failed validation: {first_line.strip()}")
            os.replace(tmp_path, out_path)
            log(f"OK: saved -> {out_path}")
            return
        except Exception as e:
            log(f"Download error: {e}")
            if attempt == MAX_ATTEMPTS:
                raise
            sleep_s = min(120, BASE_SLEEP * (2 ** (attempt - 1)))
            log(f"Sleeping {sleep_s} seconds before retry")
            time.sleep(sleep_s)


@dataclass
class StationState:
    station: str
    log_file: Path
    manifest_lines: List[List[str]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def log(self, msg: str) -> None:
        ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts}  {msg}"
        with self.lock:
            print(line)
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def add_manifest_row(self, row: List[str]) -> None:
        with self.lock:
            self.manifest_lines.append(row)


def _prepare_station_state(base_dir: Path, station: str) -> Tuple[StationState, Path, Path]:
    out_yearly = base_dir / station / VAR / TZ / "yearly"
    out_meta = base_dir / station / VAR / TZ / "meta"
    out_logs = base_dir / station / VAR / TZ / "logs"
    out_yearly.mkdir(parents=True, exist_ok=True)
    out_meta.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)
    log_file = out_logs / "download.log"
    state = StationState(station=station, log_file=log_file)
    return state, out_yearly, out_meta


def _should_download(year: int, out_file: Path, force: bool, now_utc: dt.datetime) -> bool:
    year_end = dt.datetime(year + 1, 1, 1, tzinfo=dt.timezone.utc)
    year_in_progress = year_end > now_utc + dt.timedelta(minutes=1)
    if force:
        return True
    if year_in_progress:
        return True
    if not out_file.exists():
        return True
    try:
        with out_file.open("r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline()
        return not _validate_csv_header(first_line)
    except Exception:
        return True


def _download_year_task(
    station: str,
    year: int,
    base_dir: Path,
    out_yearly: Path,
    state: StationState,
    force: bool,
    year_start: int,
    year_end: int,
) -> Tuple[str, int, List[str]]:
    now_utc = _utc_now()
    sts = f"{year}-01-01T00:00Z"
    ets = f"{year + 1}-01-01T00:00Z"
    url = _build_url(station, sts, ets, TZ)
    file_name = f"{station}_{VAR}_1min_{TZ}_{year}.csv"
    out_file = out_yearly / file_name
    rel_path = os.path.relpath(out_file, base_dir)

    status = "OK"
    notes = ""
    try:
        if _should_download(year, out_file, force, now_utc):
            state.log(f"Year {year}: downloading...")
            _download_with_retry(url, out_file, state.log)
        else:
            state.log(f"Year {year}: existing file valid; skipping download.")

        if not out_file.exists():
            raise RuntimeError("Output file missing after download/skip.")

        with out_file.open("r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline()
        if not _validate_csv_header(first_line):
            raise RuntimeError("Output file failed CSV validation.")

        size_bytes = out_file.stat().st_size
        sha = _sha256(out_file)
        rows, first_ts, last_ts = _scan_csv(out_file)

        year_end_utc = dt.datetime(year + 1, 1, 1, tzinfo=dt.timezone.utc)
        expected_last = year_end_utc - dt.timedelta(minutes=1)
        year_in_progress = year_end_utc > now_utc + dt.timedelta(minutes=1)

        if rows <= 0 or not first_ts or not last_ts:
            status = "ERROR"
            notes = "No data rows or missing timestamps."
        else:
            last_dt = _parse_utc_ts(last_ts)
            if last_dt is None:
                status = "WARN"
                notes = "Could not parse last_ts as UTC."
            elif year_in_progress:
                status = "PARTIAL"
                notes = "Year in progress; data will update."
                if last_dt < now_utc - dt.timedelta(days=3):
                    notes += " last_ts >3 days behind."
            elif last_dt < expected_last - dt.timedelta(days=2):
                status = "PARTIAL"
                notes = "Data ends earlier than expected."

        state.log(f"Year {year} done: rows={rows} last={last_ts} status={status}")

        row = [
            str(year),
            rel_path.replace("\\", "/"),
            str(size_bytes),
            sha,
            str(rows),
            first_ts,
            last_ts,
            status,
            notes,
        ]
        return station, year, row
    except Exception as e:
        status = "ERROR"
        notes = str(e)
        state.log(f"Year {year} failed: {notes}")
        row = [
            str(year),
            rel_path.replace("\\", "/"),
            "0",
            "",
            "0",
            "",
            "",
            status,
            notes,
        ]
        return station, year, row
    finally:
        time.sleep(INTER_TASK_SLEEP)


def _write_manifest(meta_dir: Path, station: str, year_start: int, year_end: int, rows: List[List[str]]) -> None:
    meta_dir.mkdir(parents=True, exist_ok=True)
    manifest = meta_dir / f"manifest_{station}_{VAR}_1min_{TZ}_{year_start}_{year_end}.csv"
    header = ["year", "file_path", "bytes", "sha256", "data_rows", "first_ts", "last_ts", "status", "notes"]
    tmp = manifest.with_suffix(manifest.suffix + ".partial")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
    os.replace(tmp, manifest)


def _discover_stations(data_root: Path) -> List[str]:
    stations = []
    if not data_root.exists():
        return stations
    for entry in data_root.iterdir():
        if entry.is_dir():
            stations.append(entry.name.upper())
    return sorted(stations)


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-threaded IEM 1-min tmpf downloader.")
    parser.add_argument("--data-root", default=None, help="Root data dir (default: ../data/iem_minute_data)")
    parser.add_argument("--stations", default=None, help="Comma-separated station list. Default: directories under data-root.")
    parser.add_argument("--year-start", type=int, default=DEFAULT_YEAR_START)
    parser.add_argument("--year-end", type=int, default=DEFAULT_YEAR_END)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    if args.data_root:
        data_root = Path(args.data_root).resolve()
    else:
        data_root = (script_dir.parent / ".." / "data" / "iem_minute_data").resolve()

    if args.stations:
        stations = [s.strip().upper() for s in args.stations.split(",") if s.strip()]
    else:
        stations = _discover_stations(data_root)

    if not stations:
        print(f"No stations found under {data_root}. Provide --stations.")
        return 1

    print(f"Data root: {data_root}")
    print(f"Stations: {','.join(stations)}")
    print(f"Years: {args.year_start}-{args.year_end}  Threads: {args.threads}  Force: {args.force}")

    station_states: Dict[str, StationState] = {}
    station_dirs: Dict[str, Tuple[Path, Path]] = {}
    for st in stations:
        state, out_yearly, out_meta = _prepare_station_state(data_root, st)
        station_states[st] = state
        station_dirs[st] = (out_yearly, out_meta)
        state.log(f"Starting station={st} years={args.year_start}-{args.year_end}")

    tasks = []
    for st in stations:
        out_yearly, _ = station_dirs[st]
        state = station_states[st]
        for year in range(args.year_start, args.year_end + 1):
            tasks.append((st, year, out_yearly, state))

    errors = 0
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=max(1, args.threads)) as ex:
        futures = []
        for st, year, out_yearly, state in tasks:
            futures.append(
                ex.submit(
                    _download_year_task,
                    st,
                    year,
                    data_root,
                    out_yearly,
                    state,
                    args.force,
                    args.year_start,
                    args.year_end,
                )
            )
        for fut in as_completed(futures):
            try:
                station, year, row = fut.result()
            except Exception as e:
                errors += 1
                print(f"Task failed: {e}")
                continue
            station_states[station].add_manifest_row(row)

    for st in stations:
        _, out_meta = station_dirs[st]
        rows = sorted(station_states[st].manifest_lines, key=lambda r: int(r[0]))
        _write_manifest(out_meta, st, args.year_start, args.year_end, rows)
        station_states[st].log(f"Finished station={st} manifest_rows={len(rows)}")

    if errors:
        print(f"Completed with {errors} task errors.")
        return 1

    print("All stations completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

