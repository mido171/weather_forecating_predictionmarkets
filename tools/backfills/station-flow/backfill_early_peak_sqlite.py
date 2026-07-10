from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import os
import sqlite3
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests


def repo_root() -> Path:
    cur = Path(__file__).resolve()
    for p in cur.parents:
        if (p / "pom.xml").exists() and (p / "ingestion-service").exists():
            return p
    raise FileNotFoundError("repo root not found")


ROOT = repo_root()


@dataclass(frozen=True)
class StationSpec:
    station_id: str
    wu_location_id: str
    nws_usw: str
    station_zoneid: str


STATION_SPECS: tuple[StationSpec, ...] = (
    StationSpec("KNYC", "KNYC:9:US", "USW00094728", "America/New_York"),
    StationSpec("KLGA", "KLGA:9:US", "USW00014732", "America/New_York"),
    StationSpec("KJFK", "KJFK:9:US", "USW00094789", "America/New_York"),
    StationSpec("KEWR", "KEWR:9:US", "USW00014734", "America/New_York"),
    StationSpec("KTEB", "KTEB:9:US", "USW00094741", "America/New_York"),
    StationSpec("KHPN", "KHPN:9:US", "USW00094745", "America/New_York"),
    StationSpec("KISP", "KISP:9:US", "USW00004781", "America/New_York"),
    StationSpec("KBDR", "KBDR:9:US", "USW00094702", "America/New_York"),
)

STATION_SPEC_BY_ID = {spec.station_id: spec for spec in STATION_SPECS}


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def today_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill per-station EarlyPeak SQLite stores.")
    p.add_argument(
        "--stations",
        default=",".join(spec.station_id for spec in STATION_SPECS),
        help="Comma-separated station ids to backfill.",
    )
    p.add_argument("--start-date", default="1973-01-01")
    p.add_argument("--end-date", default=today_utc())
    p.add_argument("--data-root", default=r"D:\Ahmed\data\sqlite\EarlyPeak")
    p.add_argument("--staging-root", default="")
    p.add_argument("--weathercom-api-key", default=os.environ.get("WEATHERCOM_API_KEY", ""))
    p.add_argument("--wu-chunk-days", type=int, default=31)
    p.add_argument("--wu-max-workers", type=int, default=16)
    p.add_argument("--wu-max-retries", type=int, default=6)
    p.add_argument("--wu-timeout-seconds", type=int, default=60)
    p.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    p.set_defaults(skip_existing=True)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def init_logger(level_name: str) -> logging.Logger:
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("early_peak_backfill")


def selected_station_specs(raw: str) -> list[StationSpec]:
    station_ids = [item.strip().upper() for item in str(raw).split(",") if item.strip()]
    out: list[StationSpec] = []
    for station_id in station_ids:
        if station_id not in STATION_SPEC_BY_ID:
            raise ValueError(f"Unknown station id: {station_id}")
        out.append(STATION_SPEC_BY_ID[station_id])
    return out


def load_weathercom_module():
    mod_path = ROOT / "ingestion-service" / "scripts" / "weathercom_download_to_csv.py"
    spec = importlib.util.spec_from_file_location("weathercom_download_to_csv", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {mod_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def extract_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("command stdout was empty")
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"stdout did not contain JSON: {raw[-500:]}")
    return json.loads(raw[start : end + 1])


def run_json_command(cmd: list[str], logger: logging.Logger) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed rc="
            f"{proc.returncode}: {' '.join(cmd)}\n"
            f"stdout_tail={(proc.stdout or '')[-3000:]}\n"
            f"stderr_tail={(proc.stderr or '')[-3000:]}"
        )
    logger.info("CMD_OK %s", " ".join(cmd))
    return extract_json_object(proc.stdout)


def write_ncei_config(path: Path, station_specs: list[StationSpec]) -> None:
    lines = ["station_map:"]
    for spec in station_specs:
        lines.append(f"  {spec.station_id}: {spec.nws_usw}")
    lines.extend(
        [
            "",
            "defaults:",
            "  dataset: daily-summaries",
            "  datatype: TMAX",
            "  units: standard",
            "  format: json",
            "  include_attributes: true",
            "  year_chunking: true",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fetch_ncei_truth(
    *,
    station_specs: list[StationSpec],
    start_date: str,
    end_date: str,
    ncei_root: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    cfg_path = ncei_root / "config.yaml"
    write_ncei_config(cfg_path, station_specs)
    cmd = [
        "python",
        str(ROOT / "tools" / "ncei_truth" / "run.py"),
        "--config",
        str(cfg_path),
        "--stations",
        ",".join(spec.station_id for spec in station_specs),
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--root-dir",
        str(ncei_root),
        "--log-level",
        "INFO",
    ]
    return run_json_command(cmd, logger)


def write_manifest_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fetch_wu_history(
    *,
    weathercom: Any,
    station_spec: StationSpec,
    start_date: str,
    end_date: str,
    api_key: str,
    out_root: Path,
    chunk_days: int,
    max_workers: int,
    max_retries: int,
    timeout_seconds: int,
    skip_existing: bool,
    logger: logging.Logger,
) -> dict[str, Any]:
    if not api_key:
        raise ValueError("WEATHERCOM_API_KEY is required for WU backfill")

    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "logs" / "weathercom_download.log"
    run_logger = weathercom.setup_logger(log_path)
    run_logger.setLevel(logger.level)
    run_logger.propagate = False
    run_logger.info(
        "WU_STATION_RUN_START station=%s location_id=%s range=%s..%s",
        station_spec.station_id,
        station_spec.wu_location_id,
        start_date,
        end_date,
    )

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=weathercom.UTC)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=weathercom.UTC)
    tasks = weathercom.build_windows(start_dt, end_dt, int(chunk_days))
    session = requests.Session()
    results: list[Any] = []
    completed = 0
    success = 0
    failed = 0
    skipped = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        fut_map = {
            executor.submit(
                weathercom.fetch_window,
                session=session,
                logger=run_logger,
                base_url=weathercom.DEFAULT_BASE_URL,
                api_key=api_key,
                location_id=station_spec.wu_location_id,
                units="e",
                out_root=out_root,
                task=task,
                timeout_seconds=int(timeout_seconds),
                max_retries=int(max_retries),
                skip_existing=bool(skip_existing),
            ): task
            for task in tasks
        }
        for fut in as_completed(fut_map):
            task = fut_map[fut]
            try:
                res = fut.result()
            except Exception as exc:
                run_logger.exception("WU_FUTURE_FAIL station=%s window=%s error=%s", station_spec.station_id, task.key, exc)
                raise
            with lock:
                results.append(res)
                completed += 1
                if res.status_code == 200:
                    success += 1
                else:
                    failed += 1
                if res.skipped_existing:
                    skipped += 1
                if completed % 25 == 0 or completed == len(tasks):
                    run_logger.info(
                        "WU_PROGRESS station=%s completed=%d/%d success=%d failed=%d skipped_existing=%d",
                        station_spec.station_id,
                        completed,
                        len(tasks),
                        success,
                        failed,
                        skipped,
                    )
    session.close()

    manifest_rows: list[dict[str, Any]] = []
    for res in sorted(results, key=lambda item: (item.task.start_date, item.task.end_date)):
        manifest_rows.append(
            {
                "location_id": station_spec.wu_location_id,
                "start_date": res.task.start_str,
                "end_date": res.task.end_str,
                "request_url": res.request_url,
                "retrieved_at_utc": res.retrieved_at_utc,
                "status_code": int(res.status_code),
                "bytes": int(res.bytes_len),
                "sha256": str(res.sha256),
                "attempts": int(res.attempts),
                "observations_count": int(res.observations_count),
                "window_csv_path": str(res.window_csv_path) if res.window_csv_path else "",
                "raw_dir": str(res.raw_dir),
                "skipped_existing": int(bool(res.skipped_existing)),
                "error": str(res.error or ""),
            }
        )
    manifest_path = out_root / "manifest" / "manifest.csv"
    write_manifest_rows(manifest_path, manifest_rows)

    window_paths = [Path(row["window_csv_path"]) for row in manifest_rows if row["status_code"] == 200 and row["window_csv_path"]]
    final_csv = out_root / "exports" / f"{station_spec.station_id}_wu_observations_30m_{start_date[:4]}_{end_date.replace('-', '')}.csv"
    final_rows = weathercom.merge_windows_to_final_csv(window_paths, final_csv, run_logger)
    summary = {
        "station_id": station_spec.station_id,
        "wu_location_id": station_spec.wu_location_id,
        "start_date": start_date,
        "end_date": end_date,
        "chunk_days": int(chunk_days),
        "max_workers": int(max_workers),
        "max_retries": int(max_retries),
        "timeout_seconds": int(timeout_seconds),
        "total_windows": int(len(tasks)),
        "success_windows": int(success),
        "failed_windows": int(failed),
        "skipped_existing_windows": int(skipped),
        "manifest_path": str(manifest_path),
        "final_csv_path": str(final_csv),
        "final_csv_rows": int(final_rows),
        "log_path": str(log_path),
        "finished_at_utc": now_utc_iso(),
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    run_logger.info("WU_STATION_RUN_DONE station=%s summary=%s", station_spec.station_id, summary_path)
    return summary


def init_early_peak_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=DELETE;
        PRAGMA synchronous=NORMAL;
        PRAGMA temp_store=MEMORY;

        CREATE TABLE IF NOT EXISTS station_meta(
            meta_key TEXT PRIMARY KEY,
            meta_value_json TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS nws_settled_tmax_daily(
            station_id TEXT NOT NULL,
            station_usw TEXT NOT NULL,
            target_date_local TEXT NOT NULL,
            settled_tmax INTEGER,
            truth_source TEXT,
            source_record_id TEXT,
            retrieved_at_utc TEXT,
            PRIMARY KEY(station_id, target_date_local)
        );

        CREATE TABLE IF NOT EXISTS nws_fetch_manifest(
            station_id TEXT NOT NULL,
            station_usw TEXT NOT NULL,
            window_start_date TEXT NOT NULL,
            window_end_date TEXT NOT NULL,
            request_url TEXT,
            response_path TEXT,
            headers_path TEXT,
            retrieved_at_utc TEXT,
            http_status INTEGER,
            body_sha256 TEXT,
            byte_count INTEGER,
            inserted_at_utc TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS nws_qa_report(
            station_id TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            rows_count INTEGER,
            duplicate_station_date_rows INTEGER,
            missing_dates_count INTEGER,
            qa_json TEXT,
            qa_markdown_path TEXT,
            inserted_at_utc TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS wu_observations_30m(
            station_id TEXT NOT NULL,
            request_location_id TEXT NOT NULL,
            valid_time_utc TEXT NOT NULL,
            valid_time_local TEXT,
            target_date_local TEXT,
            cutoff_minutes_local INTEGER,
            temp REAL,
            dew_pt REAL,
            rh REAL,
            pressure REAL,
            vis REAL,
            wspd REAL,
            wdir REAL,
            gust REAL,
            precip_hrly REAL,
            clds TEXT,
            wx_phrase TEXT,
            uv_index REAL,
            uv_desc TEXT,
            wdir_cardinal TEXT,
            PRIMARY KEY(request_location_id, valid_time_utc)
        );

        CREATE TABLE IF NOT EXISTS wu_fetch_manifest(
            station_id TEXT NOT NULL,
            location_id TEXT NOT NULL,
            window_start_date TEXT NOT NULL,
            window_end_date TEXT NOT NULL,
            request_url TEXT,
            retrieved_at_utc TEXT,
            status_code INTEGER,
            bytes INTEGER,
            sha256 TEXT,
            attempts INTEGER,
            observations_count INTEGER,
            window_csv_path TEXT,
            raw_dir TEXT,
            skipped_existing INTEGER,
            error TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_nws_settled_tmax_daily_date
        ON nws_settled_tmax_daily(target_date_local);

        CREATE INDEX IF NOT EXISTS idx_wu_observations_30m_date
        ON wu_observations_30m(target_date_local, cutoff_minutes_local);
        """
    )


def insert_station_meta(conn: sqlite3.Connection, meta_key: str, payload: Any) -> None:
    conn.execute(
        """
        INSERT INTO station_meta(meta_key, meta_value_json, updated_at_utc)
        VALUES(?,?,?)
        ON CONFLICT(meta_key) DO UPDATE SET
          meta_value_json=excluded.meta_value_json,
          updated_at_utc=excluded.updated_at_utc
        """,
        (meta_key, json.dumps(payload, sort_keys=True), now_utc_iso()),
    )


def ingest_nws_to_db(
    *,
    conn: sqlite3.Connection,
    station_spec: StationSpec,
    ncei_root: Path,
    ncei_summary: dict[str, Any],
) -> dict[str, Any]:
    conn.execute("DELETE FROM nws_settled_tmax_daily")
    conn.execute("DELETE FROM nws_fetch_manifest")
    conn.execute("DELETE FROM nws_qa_report")

    canonical = pd.read_csv(ncei_summary["canonical_csv"])
    canonical["station_id"] = canonical["station_id"].astype(str).str.upper()
    station_df = canonical[canonical["station_id"] == station_spec.station_id].copy()
    station_df["target_date_local"] = pd.to_datetime(station_df["target_date_local"], errors="coerce").dt.date.astype(str)
    station_df["tmax_f"] = pd.to_numeric(station_df["tmax_f"], errors="coerce")
    station_df = station_df[station_df["target_date_local"].notna() & station_df["tmax_f"].notna()].copy()
    settlement_rows = [
        (
            station_spec.station_id,
            station_spec.nws_usw,
            str(row.target_date_local),
            int(round(float(row.tmax_f))),
            str(row.truth_source),
            str(row.source_record_id),
            str(row.retrieved_at_utc),
        )
        for row in station_df.itertuples()
    ]
    conn.executemany(
        """
        INSERT INTO nws_settled_tmax_daily(
            station_id, station_usw, target_date_local, settled_tmax, truth_source, source_record_id, retrieved_at_utc
        ) VALUES(?,?,?,?,?,?,?)
        """,
        settlement_rows,
    )

    raw_dir = ncei_root / "raw" / "ncei_ads" / "daily-summaries" / station_spec.nws_usw
    manifest_rows: list[tuple[Any, ...]] = []
    if raw_dir.exists():
        for window_dir in sorted(raw_dir.glob("*_*")):
            if not window_dir.is_dir():
                continue
            try:
                start_part, end_part = window_dir.name.split("_", 1)
            except ValueError:
                continue
            response_path = window_dir / "response.json"
            manifest_rows.append(
                (
                    station_spec.station_id,
                    station_spec.nws_usw,
                    start_part,
                    end_part,
                    (window_dir / "request_url.txt").read_text(encoding="utf-8").strip()
                    if (window_dir / "request_url.txt").exists()
                    else "",
                    str(response_path),
                    str(window_dir / "headers.json"),
                    (window_dir / "retrieved_at_utc.txt").read_text(encoding="utf-8").strip()
                    if (window_dir / "retrieved_at_utc.txt").exists()
                    else "",
                    int((window_dir / "http_status.txt").read_text(encoding="utf-8").strip() or "0")
                    if (window_dir / "http_status.txt").exists()
                    else 0,
                    (window_dir / "sha256.txt").read_text(encoding="utf-8").strip()
                    if (window_dir / "sha256.txt").exists()
                    else "",
                    int(response_path.stat().st_size) if response_path.exists() else 0,
                    now_utc_iso(),
                )
            )
    conn.executemany(
        """
        INSERT INTO nws_fetch_manifest(
            station_id, station_usw, window_start_date, window_end_date, request_url, response_path, headers_path,
            retrieved_at_utc, http_status, body_sha256, byte_count, inserted_at_utc
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        manifest_rows,
    )

    qa_summary = ncei_summary.get(f"qa_{station_spec.station_id}", {}) or {}
    qa_json = {}
    if qa_summary.get("json") and Path(qa_summary["json"]).exists():
        qa_json = json.loads(Path(qa_summary["json"]).read_text(encoding="utf-8"))
    conn.execute(
        """
        INSERT INTO nws_qa_report(
            station_id, start_date, end_date, rows_count, duplicate_station_date_rows, missing_dates_count,
            qa_json, qa_markdown_path, inserted_at_utc
        ) VALUES(?,?,?,?,?,?,?,?,?)
        """,
        (
            station_spec.station_id,
            str(ncei_summary.get("start_date", "")),
            str(ncei_summary.get("end_date", "")),
            int(qa_summary.get("rows", 0) or 0),
            int(qa_summary.get("duplicate_station_date_rows", 0) or 0),
            int(qa_summary.get("missing_dates_count", 0) or 0),
            json.dumps(qa_json, sort_keys=True),
            str(qa_summary.get("md", "")),
            now_utc_iso(),
        ),
    )
    insert_station_meta(conn, "nws_run_summary", ncei_summary)
    return {
        "nws_settled_rows": int(len(settlement_rows)),
        "nws_manifest_rows": int(len(manifest_rows)),
        "nws_min_date": str(station_df["target_date_local"].min()) if not station_df.empty else "",
        "nws_max_date": str(station_df["target_date_local"].max()) if not station_df.empty else "",
    }


def ingest_wu_to_db(
    *,
    conn: sqlite3.Connection,
    station_spec: StationSpec,
    wu_summary: dict[str, Any],
) -> dict[str, Any]:
    conn.execute("DELETE FROM wu_observations_30m")
    conn.execute("DELETE FROM wu_fetch_manifest")

    manifest = pd.read_csv(wu_summary["manifest_path"])
    manifest["station_id"] = station_spec.station_id
    manifest = manifest[
        [
            "station_id",
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
    ].copy()
    manifest.columns = [
        "station_id",
        "location_id",
        "window_start_date",
        "window_end_date",
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
    manifest.to_sql("wu_fetch_manifest", conn, if_exists="append", index=False)

    tz = ZoneInfo(station_spec.station_zoneid)
    total_rows = 0
    min_utc = None
    max_utc = None
    min_local_date = None
    max_local_date = None
    for chunk in pd.read_csv(wu_summary["final_csv_path"], chunksize=200000):
        chunk["station_id"] = station_spec.station_id
        ts_utc = pd.to_datetime(chunk["valid_time_utc"], errors="coerce", utc=True)
        chunk = chunk[ts_utc.notna()].copy()
        if chunk.empty:
            continue
        ts_utc = pd.to_datetime(chunk["valid_time_utc"], errors="coerce", utc=True)
        ts_local = ts_utc.dt.tz_convert(tz)
        chunk["valid_time_local"] = ts_local.dt.strftime("%Y-%m-%d %H:%M:%S")
        chunk["target_date_local"] = ts_local.dt.strftime("%Y-%m-%d")
        chunk["cutoff_minutes_local"] = ts_local.dt.hour * 60 + ts_local.dt.minute
        chunk = chunk[
            [
                "station_id",
                "request_location_id",
                "valid_time_utc",
                "valid_time_local",
                "target_date_local",
                "cutoff_minutes_local",
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
        ].copy()
        chunk.to_sql("wu_observations_30m", conn, if_exists="append", index=False)
        total_rows += int(len(chunk))
        cur_min_utc = chunk["valid_time_utc"].min()
        cur_max_utc = chunk["valid_time_utc"].max()
        cur_min_date = chunk["target_date_local"].min()
        cur_max_date = chunk["target_date_local"].max()
        min_utc = cur_min_utc if min_utc is None or cur_min_utc < min_utc else min_utc
        max_utc = cur_max_utc if max_utc is None or cur_max_utc > max_utc else max_utc
        min_local_date = cur_min_date if min_local_date is None or cur_min_date < min_local_date else min_local_date
        max_local_date = cur_max_date if max_local_date is None or cur_max_date > max_local_date else max_local_date
    insert_station_meta(conn, "wu_run_summary", wu_summary)
    return {
        "wu_rows": int(total_rows),
        "wu_manifest_rows": int(len(manifest)),
        "wu_min_valid_time_utc": str(min_utc or ""),
        "wu_max_valid_time_utc": str(max_utc or ""),
        "wu_min_target_date_local": str(min_local_date or ""),
        "wu_max_target_date_local": str(max_local_date or ""),
    }


def validate_db(db_path: Path) -> dict[str, Any]:
    with sqlite3.connect(str(db_path)) as conn:
        nws_count, nws_min_date, nws_max_date = conn.execute(
            "SELECT COUNT(*), MIN(target_date_local), MAX(target_date_local) FROM nws_settled_tmax_daily"
        ).fetchone()
        wu_count, wu_min_utc, wu_max_utc, wu_min_date, wu_max_date = conn.execute(
            "SELECT COUNT(*), MIN(valid_time_utc), MAX(valid_time_utc), MIN(target_date_local), MAX(target_date_local) FROM wu_observations_30m"
        ).fetchone()
        failed_windows = conn.execute(
            "SELECT COUNT(*) FROM wu_fetch_manifest WHERE COALESCE(status_code, 0) <> 200"
        ).fetchone()[0]
    return {
        "nws_rows": int(nws_count or 0),
        "nws_min_date": str(nws_min_date or ""),
        "nws_max_date": str(nws_max_date or ""),
        "wu_rows": int(wu_count or 0),
        "wu_min_valid_time_utc": str(wu_min_utc or ""),
        "wu_max_valid_time_utc": str(wu_max_utc or ""),
        "wu_min_target_date_local": str(wu_min_date or ""),
        "wu_max_target_date_local": str(wu_max_date or ""),
        "wu_failed_windows": int(failed_windows or 0),
    }


def build_station_sqlite(
    *,
    station_spec: StationSpec,
    station_dir: Path,
    ncei_root: Path,
    ncei_summary: dict[str, Any],
    wu_summary: dict[str, Any],
    start_date: str,
    end_date: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    station_dir.mkdir(parents=True, exist_ok=True)
    db_name = f"{station_spec.station_id}_early_peak_{start_date[:4]}_{end_date.replace('-', '')}.sqlite"
    db_path = station_dir / db_name
    if db_path.exists():
        db_path.unlink()
    try:
        with sqlite3.connect(str(db_path)) as conn:
            init_early_peak_db(conn)
            insert_station_meta(
                conn,
                "station_spec",
                {
                    "station_id": station_spec.station_id,
                    "wu_location_id": station_spec.wu_location_id,
                    "nws_usw": station_spec.nws_usw,
                    "station_zoneid": station_spec.station_zoneid,
                    "start_date": start_date,
                    "end_date": end_date,
                },
            )
            nws_stats = ingest_nws_to_db(
                conn=conn,
                station_spec=station_spec,
                ncei_root=ncei_root,
                ncei_summary=ncei_summary,
            )
            wu_stats = ingest_wu_to_db(
                conn=conn,
                station_spec=station_spec,
                wu_summary=wu_summary,
            )
            combined_stats = {
                "station_id": station_spec.station_id,
                "db_path": str(db_path),
                **nws_stats,
                **wu_stats,
            }
            insert_station_meta(conn, "build_stats", combined_stats)
            conn.commit()
    except Exception:
        if db_path.exists():
            db_path.unlink()
        raise
    validation = validate_db(db_path)
    logger.info(
        "STATION_DB_READY station=%s path=%s nws_rows=%d wu_rows=%d",
        station_spec.station_id,
        db_path,
        validation["nws_rows"],
        validation["wu_rows"],
    )
    return {"station_id": station_spec.station_id, "db_path": str(db_path), **validation}


def main() -> int:
    args = parse_args()
    logger = init_logger(args.log_level)
    station_specs = selected_station_specs(args.stations)
    data_root = Path(args.data_root).resolve()
    staging_root = Path(args.staging_root).resolve() if args.staging_root else (data_root.parent / "EarlyPeak_staging").resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)

    weathercom = load_weathercom_module()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logger.info(
        "EARLYPEAK_RUN_START run_id=%s stations=%s range=%s..%s data_root=%s staging_root=%s",
        run_id,
        ",".join(spec.station_id for spec in station_specs),
        args.start_date,
        args.end_date,
        data_root,
        staging_root,
    )

    ncei_root = staging_root / "ncei_truth"
    ncei_summary = fetch_ncei_truth(
        station_specs=station_specs,
        start_date=args.start_date,
        end_date=args.end_date,
        ncei_root=ncei_root,
        logger=logger,
    )

    station_results: list[dict[str, Any]] = []
    for station_spec in station_specs:
        logger.info("STATION_START station=%s", station_spec.station_id)
        wu_root = staging_root / "weathercom" / station_spec.station_id
        wu_summary = fetch_wu_history(
            weathercom=weathercom,
            station_spec=station_spec,
            start_date=args.start_date,
            end_date=args.end_date,
            api_key=args.weathercom_api_key,
            out_root=wu_root,
            chunk_days=args.wu_chunk_days,
            max_workers=args.wu_max_workers,
            max_retries=args.wu_max_retries,
            timeout_seconds=args.wu_timeout_seconds,
            skip_existing=args.skip_existing,
            logger=logger,
        )
        station_dir = data_root / station_spec.station_id
        result = build_station_sqlite(
            station_spec=station_spec,
            station_dir=station_dir,
            ncei_root=ncei_root,
            ncei_summary=ncei_summary,
            wu_summary=wu_summary,
            start_date=args.start_date,
            end_date=args.end_date,
            logger=logger,
        )
        station_results.append(result)

    summary = {
        "run_id": run_id,
        "stations": [spec.station_id for spec in station_specs],
        "start_date": args.start_date,
        "end_date": args.end_date,
        "data_root": str(data_root),
        "staging_root": str(staging_root),
        "station_results": station_results,
        "generated_at_utc": now_utc_iso(),
    }
    summary_path = staging_root / f"early_peak_backfill_summary_{run_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("EARLYPEAK_RUN_DONE summary=%s", summary_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
