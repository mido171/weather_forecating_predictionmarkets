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
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd


def repo_root() -> Path:
    cur = Path(__file__).resolve()
    for parent in cur.parents:
        if (parent / "pom.xml").is_file() and (
            parent / "apps" / "ingestion-service"
        ).is_dir():
            return parent
    raise FileNotFoundError("Could not locate repo root")


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NCEI_DIR = ROOT / "tools" / "backfills" / "ncei_truth"
if str(NCEI_DIR) not in sys.path:
    sys.path.insert(0, str(NCEI_DIR))

import download as ncei_download  # type: ignore  # noqa: E402
import normalize as ncei_normalize  # type: ignore  # noqa: E402
import qa as ncei_qa  # type: ignore  # noqa: E402

from tools.backfills.pooled_strategy.iem_registry import (  # noqa: E402
    ResolvedStationMetadata,
    build_iem_airport_registry,
    build_station_crosswalk_rows,
    resolve_station_metadata,
)
from tools.backfills.pooled_strategy.sqlite_store import (  # noqa: E402
    begin_ingest_run,
    connect_station_db,
    finish_ingest_run,
    log_ingest_event,
    now_utc,
    upsert_source_status,
    upsert_station_registry,
)
from tools.backfills.pooled_strategy.station_universe import (  # noqa: E402
    StationSeed,
    get_station_seeds,
)


TRUTH_TABLES = ["nws_raw_snapshots", "nws_truth_canonical", "nws_truth_enriched", "nws_qa_reports", "nws_run_meta"]
MOS_TABLES = ["mos_raw_payloads", "mos_hourly_values", "mos_download_manifest", "mos_run_meta"]
WU_TABLES = ["wu_fetch_manifest", "wu_observations_30m", "wu_run_meta"]
KALSHI_TABLES = ["kalshi_download_manifest", "kalshi_minute_prices", "kalshi_run_meta"]


def _load_module(module_name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


MOS_MODULE = _load_module(
    "pooled_strategy_iem_mos_downloader",
    ROOT / "apps" / "ingestion-service" / "scripts" / "download_iem_mos_kncy_archive.py",
)
WU_BACKFILL_MODULE = _load_module(
    "pooled_strategy_wu_backfill",
    ROOT / "tools" / "backfills" / "station_flow" / "backfill_early_peak_sqlite.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 pooled-strategy station backfill into per-station SQLite stores.")
    parser.add_argument("--stations", default="all", help="Station scope: all, active, reserve, or comma-separated station ids.")
    parser.add_argument("--data-root", default=r"D:\Ahmed\data\sqlite\pooled_strategy")
    parser.add_argument("--training-data-root", default=r"D:\Ahmed\data\kalshi\training_data")
    parser.add_argument("--existing-nws-root", default=r"D:\Ahmed\data\sqlite\NWS")
    parser.add_argument("--existing-mos-root", default=r"D:\Ahmed\data\sqlite\MOS")
    parser.add_argument("--existing-wu-root", default=r"D:\Ahmed\data\sqlite\EarlyPeak")
    parser.add_argument("--existing-kalshi-root", default=r"D:\Ahmed\data\kalshi\kalshi_history")
    parser.add_argument("--truth-start-date", default="2000-06-01")
    parser.add_argument("--truth-end-date", default="2026-12-31")
    parser.add_argument("--mos-start-year", type=int, default=2000)
    parser.add_argument("--mos-end-year", type=int, default=2026)
    parser.add_argument("--wu-start-date", default="1973-01-01")
    parser.add_argument("--wu-end-date", default=datetime.now(timezone.utc).date().isoformat())
    parser.add_argument("--weathercom-api-key", default=os.environ.get("WEATHERCOM_API_KEY", ""))
    parser.add_argument("--kalshi-start-date", default="2024-10-01")
    parser.add_argument("--kalshi-end-date", default=datetime.now(timezone.utc).date().isoformat())
    parser.add_argument("--mos-threads", type=int, default=2)
    parser.add_argument("--mos-timeout-sec", type=int, default=120)
    parser.add_argument("--mos-max-retries", type=int, default=6)
    parser.add_argument("--mos-backoff-base-sec", type=float, default=1.0)
    parser.add_argument("--wu-chunk-days", type=int, default=31)
    parser.add_argument("--wu-max-workers", type=int, default=2)
    parser.add_argument("--wu-max-retries", type=int, default=6)
    parser.add_argument("--wu-timeout-seconds", type=int, default=60)
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--refresh-registry-cache", action="store_true")
    parser.add_argument("--download-missing-kalshi", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def json_safe(payload: Any) -> Any:
    return json.loads(json.dumps(payload, default=str))


def configure_logger(run_root: Path, level_name: str) -> logging.Logger:
    logger = logging.getLogger("pooled_strategy_stage1")
    logger.setLevel(getattr(logging, level_name.upper()))
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    run_root.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(run_root / "stage1_backfill.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return bool(row)


def latest_file(root: Path, patterns: list[str]) -> Path | None:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(root.glob(pattern))
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def ranked_file(candidates: list[Path], *, ranker: Callable[[Path], tuple[Any, ...]]) -> Path | None:
    if not candidates:
        return None
    return sorted(candidates, key=ranker, reverse=True)[0]


def clear_station_tables(conn: sqlite3.Connection, station_id: str, tables: list[str]) -> None:
    for table in tables:
        conn.execute(f"DELETE FROM {table} WHERE station_id=?", (station_id,))


def count_station_rows(conn: sqlite3.Connection, table: str, station_id: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table} WHERE station_id=?", (station_id,)).fetchone()[0] or 0)


def export_truth_from_db(conn: sqlite3.Connection, station_id: str, out_path: Path) -> dict[str, Any]:
    df = pd.read_sql_query(
        "SELECT station_id, target_date_local AS date, tmax_f AS settled_tmax FROM nws_truth_canonical WHERE station_id=? ORDER BY target_date_local",
        conn,
        params=(station_id,),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return {"path": str(out_path), "rows": int(len(df))}


def export_mos_from_db(conn: sqlite3.Connection, station_id: str, out_path: Path) -> dict[str, Any]:
    df = pd.read_sql_query(
        "SELECT * FROM mos_hourly_values WHERE station_id=? ORDER BY runtime_utc, forecast_time_utc",
        conn,
        params=(station_id,),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, compression="gzip")
    model_counts = {str(key): int(value) for key, value in df["model"].astype(str).value_counts().to_dict().items()} if not df.empty else {}
    return {"path": str(out_path), "rows": int(len(df)), "model_counts": model_counts}


def run_subprocess(command: list[str], logger: logging.Logger, *, cwd: Path | None = None) -> dict[str, Any]:
    logger.info("CMD_START %s", " ".join(command))
    proc = subprocess.run(
        command,
        cwd=str(cwd or ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    stdout_tail = (proc.stdout or "")[-4000:]
    stderr_tail = (proc.stderr or "")[-4000:]
    if stdout_tail:
        logger.info("CMD_STDOUT_TAIL %s", stdout_tail)
    if stderr_tail:
        logger.info("CMD_STDERR_TAIL %s", stderr_tail)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed rc={proc.returncode}: {' '.join(command)}\nstdout_tail={stdout_tail}\nstderr_tail={stderr_tail}"
        )
    logger.info("CMD_DONE rc=%d %s", proc.returncode, " ".join(command))
    return {
        "command": command,
        "returncode": proc.returncode,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


def import_nws_from_existing_sqlite(
    target_conn: sqlite3.Connection,
    *,
    target_station_id: str,
    station_usw: str,
    source_db_path: Path,
) -> dict[str, Any]:
    clear_station_tables(target_conn, target_station_id, TRUTH_TABLES)
    source_conn = sqlite3.connect(str(source_db_path))
    source_conn.row_factory = sqlite3.Row
    try:
        if table_exists(source_conn, "nws_truth_canonical"):
            for chunk in pd.read_sql_query("SELECT * FROM nws_truth_canonical", source_conn, chunksize=50000):
                chunk["station_id"] = target_station_id
                chunk["station_usw"] = station_usw
                chunk.to_sql("nws_truth_canonical", target_conn, if_exists="append", index=False)
            if table_exists(source_conn, "nws_raw_snapshots"):
                df = pd.read_sql_query("SELECT * FROM nws_raw_snapshots", source_conn)
                if not df.empty:
                    df["station_id"] = target_station_id
                    df["station_usw"] = station_usw
                    df.to_sql("nws_raw_snapshots", target_conn, if_exists="append", index=False)
            if table_exists(source_conn, "nws_truth_enriched"):
                for chunk in pd.read_sql_query("SELECT * FROM nws_truth_enriched", source_conn, chunksize=50000):
                    chunk["station_id"] = target_station_id
                    chunk["station_usw"] = station_usw
                    chunk.to_sql("nws_truth_enriched", target_conn, if_exists="append", index=False)
            if table_exists(source_conn, "nws_qa_reports"):
                df = pd.read_sql_query("SELECT * FROM nws_qa_reports", source_conn)
                if not df.empty:
                    df["station_id"] = target_station_id
                    df.to_sql("nws_qa_reports", target_conn, if_exists="append", index=False)
            if table_exists(source_conn, "nws_run_meta"):
                df = pd.read_sql_query("SELECT * FROM nws_run_meta", source_conn)
                if not df.empty:
                    df["station_id"] = target_station_id
                    df.to_sql("nws_run_meta", target_conn, if_exists="append", index=False)
        elif table_exists(source_conn, "nws_settled_tmax_daily"):
            df = pd.read_sql_query("SELECT * FROM nws_settled_tmax_daily", source_conn)
            if not df.empty:
                canonical = pd.DataFrame(
                    {
                        "station_id": target_station_id,
                        "station_usw": station_usw,
                        "target_date_local": df["target_date_local"],
                        "tmax_f": df["settled_tmax"],
                        "truth_source": df.get("truth_source", pd.Series(["EARLYPEAK_SQLITE_LOCAL"] * len(df))),
                        "source_record_id": df.get("source_record_id", pd.Series([""] * len(df))),
                        "retrieved_at_utc": df.get("retrieved_at_utc", pd.Series([""] * len(df))),
                    }
                )
                canonical.to_sql("nws_truth_canonical", target_conn, if_exists="append", index=False)
            if table_exists(source_conn, "nws_fetch_manifest"):
                manifest = pd.read_sql_query("SELECT * FROM nws_fetch_manifest", source_conn)
                if not manifest.empty:
                    manifest["station_id"] = target_station_id
                    manifest["station_usw"] = station_usw
                    manifest.to_sql("nws_raw_snapshots", target_conn, if_exists="append", index=False)
            if table_exists(source_conn, "nws_qa_report"):
                qa_df = pd.read_sql_query("SELECT * FROM nws_qa_report", source_conn)
                if not qa_df.empty:
                    qa_df["station_id"] = target_station_id
                    qa_df.rename(columns={"qa_markdown_path": "qa_md_path"}, inplace=True)
                    qa_df.to_sql("nws_qa_reports", target_conn, if_exists="append", index=False)
            if table_exists(source_conn, "station_meta"):
                meta_df = pd.read_sql_query("SELECT * FROM station_meta", source_conn)
                if not meta_df.empty:
                    meta_df["station_id"] = target_station_id
                    meta_df = meta_df[["station_id", "meta_key", "meta_value_json", "updated_at_utc"]]
                    meta_df.to_sql("nws_run_meta", target_conn, if_exists="append", index=False)
        else:
            raise ValueError(f"No recognized NWS tables in {source_db_path}")
        rows = count_station_rows(target_conn, "nws_truth_canonical", target_station_id)
        return {"mode": "import_existing_sqlite", "source_db_path": str(source_db_path), "rows": rows}
    finally:
        source_conn.close()


def import_nws_from_truth_csv(
    target_conn: sqlite3.Connection,
    *,
    target_station_id: str,
    station_usw: str,
    truth_csv_path: Path,
) -> dict[str, Any]:
    clear_station_tables(target_conn, target_station_id, TRUTH_TABLES)
    df = pd.read_csv(truth_csv_path)
    if {"date", "settled_tmax"}.issubset(df.columns):
        canonical = pd.DataFrame(
            {
                "station_id": target_station_id,
                "station_usw": station_usw,
                "target_date_local": pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str),
                "tmax_f": pd.to_numeric(df["settled_tmax"], errors="coerce"),
                "truth_source": "LOCAL_TRUTH_CSV",
                "source_record_id": df.get("source_record_id", pd.Series([""] * len(df))),
                "retrieved_at_utc": df.get("retrieved_at_utc", pd.Series([""] * len(df))),
            }
        )
    else:
        canonical = pd.DataFrame(
            {
                "station_id": target_station_id,
                "station_usw": station_usw,
                "target_date_local": pd.to_datetime(df["target_date_local"], errors="coerce").dt.date.astype(str),
                "tmax_f": pd.to_numeric(df["tmax_f"], errors="coerce"),
                "truth_source": df.get("truth_source", pd.Series(["LOCAL_TRUTH_CSV"] * len(df))),
                "source_record_id": df.get("source_record_id", pd.Series([""] * len(df))),
                "retrieved_at_utc": df.get("retrieved_at_utc", pd.Series([""] * len(df))),
            }
        )
    canonical = canonical.dropna(subset=["target_date_local", "tmax_f"]).copy()
    canonical["tmax_f"] = canonical["tmax_f"].round().astype(int)
    canonical.to_sql("nws_truth_canonical", target_conn, if_exists="append", index=False)
    target_conn.execute(
        "INSERT INTO nws_run_meta(station_id, meta_key, meta_value_json, updated_at_utc) VALUES(?,?,?,?)",
        (target_station_id, "local_truth_csv_import", json.dumps({"source_path": str(truth_csv_path)}, sort_keys=True), now_utc()),
    )
    return {"mode": "import_truth_csv", "source_path": str(truth_csv_path), "rows": int(len(canonical))}


def download_and_ingest_nws(
    target_conn: sqlite3.Connection,
    *,
    target_station_id: str,
    station_usw: str,
    start_date: str,
    end_date: str,
    staging_root: Path,
    logger: logging.Logger,
    resume: bool,
) -> dict[str, Any]:
    clear_station_tables(target_conn, target_station_id, TRUTH_TABLES)
    start_date_obj = datetime.fromisoformat(start_date).date()
    end_date_obj = datetime.fromisoformat(end_date).date()
    raw_root = staging_root / "raw"
    manifest_csv = staging_root / "manifests" / "manifest.csv"
    snapshots = ncei_download.run_download(
        station_map={target_station_id: station_usw},
        start_date=start_date_obj,
        end_date=end_date_obj,
        raw_root=raw_root,
        manifest_csv_path=manifest_csv,
        logger=logger,
        skip_if_snapshot_exists=resume,
    )
    rows = ncei_normalize.normalize_snapshots_to_rows(snapshots=snapshots, logger=logger)
    canonical_csv = staging_root / "canonical" / f"{target_station_id}_ncei_truth.csv"
    enriched_csv = staging_root / "canonical" / f"{target_station_id}_ncei_truth_enriched.csv"
    canonical_df = ncei_normalize.write_canonical_csv(rows, canonical_csv)
    enriched_df = ncei_normalize.write_enriched_rows(rows, enriched_csv)
    report = ncei_qa.build_station_qa_report(
        station_id=target_station_id,
        station_df=enriched_df[enriched_df["station_id"].astype(str).str.upper() == target_station_id],
        start_date=start_date_obj,
        end_date=end_date_obj,
    )
    qa_json = staging_root / "reports" / f"qa_{target_station_id}.json"
    qa_md = staging_root / "reports" / f"qa_{target_station_id}.md"
    qa_json.parent.mkdir(parents=True, exist_ok=True)
    qa_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    ncei_qa.write_station_qa_markdown(report, qa_md)

    raw_rows = [
        {
            "station_id": target_station_id,
            "station_usw": station_usw,
            "window_start_date": snap.start_date.isoformat(),
            "window_end_date": snap.end_date.isoformat(),
            "request_url": snap.url,
            "response_path": str(snap.response_path),
            "headers_path": str(snap.headers_path),
            "retrieved_at_utc": snap.retrieved_at_utc,
            "http_status": snap.http_status,
            "body_sha256": snap.body_sha256,
            "byte_count": snap.byte_count,
            "inserted_at_utc": now_utc(),
        }
        for snap in snapshots
    ]
    if raw_rows:
        pd.DataFrame(raw_rows).to_sql("nws_raw_snapshots", target_conn, if_exists="append", index=False)
    canonical_df["station_id"] = target_station_id
    canonical_df["station_usw"] = station_usw
    canonical_df.to_sql("nws_truth_canonical", target_conn, if_exists="append", index=False)
    enriched_df["station_id"] = target_station_id
    enriched_df["station_usw"] = station_usw
    enriched_df.to_sql("nws_truth_enriched", target_conn, if_exists="append", index=False)
    pd.DataFrame(
        [
            {
                "station_id": target_station_id,
                "start_date": start_date,
                "end_date": end_date,
                "rows_count": int(report.get("rows", 0) or 0),
                "duplicate_station_date_rows": int(report.get("duplicate_station_date_rows", 0) or 0),
                "missing_dates_count": int(len(report.get("missing_dates", []))),
                "qa_json": json.dumps(report, sort_keys=True),
                "qa_md_path": str(qa_md),
                "inserted_at_utc": now_utc(),
            }
        ]
    ).to_sql("nws_qa_reports", target_conn, if_exists="append", index=False)
    pd.DataFrame(
        [
            {
                "station_id": target_station_id,
                "meta_key": "ncei_truth_run",
                "meta_value_json": json.dumps(
                    {
                        "manifest_csv": str(manifest_csv),
                        "canonical_csv": str(canonical_csv),
                        "enriched_csv": str(enriched_csv),
                        "qa_json": str(qa_json),
                    },
                    sort_keys=True,
                ),
                "updated_at_utc": now_utc(),
            }
        ]
    ).to_sql("nws_run_meta", target_conn, if_exists="append", index=False)
    return {
        "mode": "download_ncei_truth",
        "snapshots": len(snapshots),
        "rows": int(len(canonical_df)),
        "manifest_csv": str(manifest_csv),
        "canonical_csv": str(canonical_csv),
        "qa_json": str(qa_json),
    }


def import_mos_from_existing_sqlite(
    target_conn: sqlite3.Connection,
    *,
    target_station_id: str,
    source_db_path: Path,
) -> dict[str, Any]:
    clear_station_tables(target_conn, target_station_id, MOS_TABLES)
    source_conn = sqlite3.connect(str(source_db_path))
    source_conn.row_factory = sqlite3.Row
    try:
        for table_name in ("mos_raw_payloads", "mos_hourly_values", "mos_download_manifest", "mos_run_meta"):
            if not table_exists(source_conn, table_name):
                continue
            if table_name == "mos_hourly_values":
                iterator = pd.read_sql_query(f"SELECT * FROM {table_name}", source_conn, chunksize=50000)
                for chunk in iterator:
                    if chunk.empty:
                        continue
                    chunk["station_id"] = target_station_id
                    chunk.to_sql(table_name, target_conn, if_exists="append", index=False)
                continue
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", source_conn)
            if df.empty:
                continue
            df["station_id"] = target_station_id
            df.to_sql(table_name, target_conn, if_exists="append", index=False)
        rows = count_station_rows(target_conn, "mos_hourly_values", target_station_id)
        return {"mode": "import_existing_sqlite", "source_db_path": str(source_db_path), "rows": rows}
    finally:
        source_conn.close()


def import_mos_from_download_dir(
    target_conn: sqlite3.Connection,
    *,
    target_station_id: str,
    download_root: Path,
    merged_csv_path: Path,
    manifest_csv_path: Path,
    run_meta_path: Path | None,
) -> dict[str, Any]:
    clear_station_tables(target_conn, target_station_id, MOS_TABLES)
    meta_dir = download_root / "mos_meta"
    meta_rows: list[dict[str, Any]] = []
    for meta_path in sorted(meta_dir.glob("*.json")):
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        meta_rows.append(
            {
                "station_id": target_station_id,
                "model": str(payload.get("model", "")).upper(),
                "year": int(payload.get("year", 0) or 0),
                "request_params_json": json.dumps(payload.get("request_params", {}), sort_keys=True),
                "retrieved_at_utc": str(payload.get("retrieved_at_utc", "")),
                "response_sha256": str(payload.get("response_sha256", "")),
                "row_count": int(payload.get("row_count", 0) or 0),
                "runtime_hour_counts_json": json.dumps(payload.get("runtime_hour_counts", {}), sort_keys=True),
                "yearly_csv_gz": str(payload.get("yearly_csv_gz", "")),
                "raw_json_gz": str(payload.get("raw_json_gz", "")),
                "meta_file": str(meta_path),
            }
        )
    if meta_rows:
        pd.DataFrame(meta_rows).to_sql("mos_raw_payloads", target_conn, if_exists="append", index=False)

    total_rows = 0
    for chunk in pd.read_csv(merged_csv_path, chunksize=200000, low_memory=False):
        if chunk.empty:
            continue
        chunk["station_id"] = target_station_id
        chunk.to_sql("mos_hourly_values", target_conn, if_exists="append", index=False)
        total_rows += int(len(chunk))

    manifest_df = pd.read_csv(manifest_csv_path)
    if not manifest_df.empty:
        manifest_df["station_id"] = target_station_id
        manifest_df.to_sql("mos_download_manifest", target_conn, if_exists="append", index=False)

    if run_meta_path is not None and run_meta_path.exists():
        payload = json.loads(run_meta_path.read_text(encoding="utf-8"))
        target_conn.execute(
            """
            INSERT INTO mos_run_meta(station_id, meta_key, meta_value_json, updated_at_utc)
            VALUES(?,?,?,?)
            ON CONFLICT(station_id, meta_key) DO UPDATE SET
              meta_value_json=excluded.meta_value_json,
              updated_at_utc=excluded.updated_at_utc
            """,
            (target_station_id, "mos_download_run_meta", json.dumps(payload, sort_keys=True), now_utc()),
        )
    return {
        "mode": "import_download_dir",
        "download_root": str(download_root),
        "rows": total_rows,
        "manifest_path": str(manifest_csv_path),
        "merged_csv_path": str(merged_csv_path),
    }


def import_mos_from_archive_csv(
    target_conn: sqlite3.Connection,
    *,
    target_station_id: str,
    archive_csv_path: Path,
) -> dict[str, Any]:
    clear_station_tables(target_conn, target_station_id, MOS_TABLES)
    column_order = list(MOS_MODULE.CANONICAL_COLUMNS)
    grouped_counts: dict[tuple[str, int], int] = {}
    total_rows = 0
    for chunk in pd.read_csv(archive_csv_path, chunksize=200000, low_memory=False):
        if chunk.empty:
            continue
        for col in column_order:
            if col not in chunk.columns:
                chunk[col] = None
        chunk = chunk[column_order].copy()
        chunk["station_id"] = target_station_id
        if chunk["year"].isna().all():
            runtime_year = pd.to_datetime(chunk["runtime_utc"], errors="coerce", utc=True).dt.year
            chunk["year"] = runtime_year
        chunk["year"] = pd.to_numeric(chunk["year"], errors="coerce").fillna(0).astype(int)
        chunk.to_sql("mos_hourly_values", target_conn, if_exists="append", index=False)
        total_rows += int(len(chunk))
        for (model, year), subdf in chunk.groupby(["model", "year"], dropna=False):
            grouped_counts[(str(model).upper(), int(year))] = grouped_counts.get((str(model).upper(), int(year)), 0) + int(len(subdf))

    manifest_rows = [
        {
            "station_id": target_station_id,
            "model": model,
            "year": year,
            "status": "import_archive_csv",
            "row_count": row_count,
            "column_count": len(column_order),
            "yearly_file": str(archive_csv_path),
            "raw_file": "",
            "meta_file": "",
        }
        for (model, year), row_count in sorted(grouped_counts.items())
    ]
    if manifest_rows:
        pd.DataFrame(manifest_rows).to_sql("mos_download_manifest", target_conn, if_exists="append", index=False)
        raw_rows = [
            {
                "station_id": row["station_id"],
                "model": row["model"],
                "year": row["year"],
                "request_params_json": json.dumps({"import_mode": "archive_csv", "source_path": str(archive_csv_path)}, sort_keys=True),
                "retrieved_at_utc": "",
                "response_sha256": "",
                "row_count": row["row_count"],
                "runtime_hour_counts_json": json.dumps({}, sort_keys=True),
                "yearly_csv_gz": str(archive_csv_path),
                "raw_json_gz": "",
                "meta_file": "",
            }
            for row in manifest_rows
        ]
        pd.DataFrame(raw_rows).to_sql("mos_raw_payloads", target_conn, if_exists="append", index=False)
    target_conn.execute(
        """
        INSERT INTO mos_run_meta(station_id, meta_key, meta_value_json, updated_at_utc)
        VALUES(?,?,?,?)
        ON CONFLICT(station_id, meta_key) DO UPDATE SET
          meta_value_json=excluded.meta_value_json,
          updated_at_utc=excluded.updated_at_utc
        """,
        (target_station_id, "archive_csv_import", json.dumps({"source_path": str(archive_csv_path)}, sort_keys=True), now_utc()),
    )
    return {"mode": "import_archive_csv", "source_path": str(archive_csv_path), "rows": total_rows}


def download_and_ingest_mos(
    target_conn: sqlite3.Connection,
    *,
    target_station_id: str,
    source_station_id: str,
    start_year: int,
    end_year: int,
    staging_root: Path,
    logger: logging.Logger,
    args: argparse.Namespace,
) -> dict[str, Any]:
    staging_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(
            ROOT
            / "apps"
            / "ingestion-service"
            / "scripts"
            / "download_iem_mos_kncy_archive.py"
        ),
        "--station",
        source_station_id,
        "--models",
        "GFS",
        "NAM",
        "--start-year",
        str(start_year),
        "--end-year",
        str(end_year),
        "--threads",
        str(max(1, int(args.mos_threads))),
        "--timeout-sec",
        str(int(args.mos_timeout_sec)),
        "--max-retries",
        str(int(args.mos_max_retries)),
        "--backoff-base-sec",
        str(float(args.mos_backoff_base_sec)),
        "--output-dir",
        str(staging_root),
        "--log-level",
        str(args.log_level).upper(),
    ]
    if args.resume:
        cmd.append("--resume")
    cmd_result = run_subprocess(cmd, logger, cwd=ROOT)
    merged_csv_path = staging_root / f"{source_station_id}_mos_archive_{start_year}_{end_year}.csv.gz"
    manifest_csv_path = staging_root / f"{source_station_id}_mos_download_manifest_{start_year}_{end_year}.csv"
    run_meta_path = staging_root / f"{source_station_id}_mos_download_run_meta_{start_year}_{end_year}.json"
    result = import_mos_from_download_dir(
        target_conn,
        target_station_id=target_station_id,
        download_root=staging_root,
        merged_csv_path=merged_csv_path,
        manifest_csv_path=manifest_csv_path,
        run_meta_path=run_meta_path if run_meta_path.exists() else None,
    )
    result["command"] = cmd_result
    result["source_station_id"] = source_station_id
    return result


def import_wu_from_existing_sqlite(
    target_conn: sqlite3.Connection,
    *,
    target_station_id: str,
    source_db_path: Path,
) -> dict[str, Any]:
    clear_station_tables(target_conn, target_station_id, WU_TABLES)
    source_conn = sqlite3.connect(str(source_db_path))
    source_conn.row_factory = sqlite3.Row
    try:
        if table_exists(source_conn, "wu_fetch_manifest"):
            df = pd.read_sql_query("SELECT * FROM wu_fetch_manifest", source_conn)
            if not df.empty:
                df["station_id"] = target_station_id
                df.to_sql("wu_fetch_manifest", target_conn, if_exists="append", index=False)
        if table_exists(source_conn, "wu_observations_30m"):
            for chunk in pd.read_sql_query("SELECT * FROM wu_observations_30m", source_conn, chunksize=200000):
                if chunk.empty:
                    continue
                chunk["station_id"] = target_station_id
                chunk.to_sql("wu_observations_30m", target_conn, if_exists="append", index=False)
        if table_exists(source_conn, "station_meta"):
            df = pd.read_sql_query("SELECT * FROM station_meta", source_conn)
            if not df.empty:
                df["station_id"] = target_station_id
                df = df[["station_id", "meta_key", "meta_value_json", "updated_at_utc"]]
                df.to_sql("wu_run_meta", target_conn, if_exists="append", index=False)
        rows = count_station_rows(target_conn, "wu_observations_30m", target_station_id)
        return {"mode": "import_existing_sqlite", "source_db_path": str(source_db_path), "rows": rows}
    finally:
        source_conn.close()


def download_and_ingest_wu(
    target_conn: sqlite3.Connection,
    *,
    target_station_id: str,
    station_zoneid: str,
    wu_location_id: str,
    start_date: str,
    end_date: str,
    staging_root: Path,
    weathercom_api_key: str,
    logger: logging.Logger,
    args: argparse.Namespace,
) -> dict[str, Any]:
    clear_station_tables(target_conn, target_station_id, WU_TABLES)
    weathercom = WU_BACKFILL_MODULE.load_weathercom_module()
    station_spec = SimpleNamespace(station_id=target_station_id, wu_location_id=wu_location_id)
    summary = WU_BACKFILL_MODULE.fetch_wu_history(
        weathercom=weathercom,
        station_spec=station_spec,
        start_date=start_date,
        end_date=end_date,
        api_key=weathercom_api_key,
        out_root=staging_root,
        chunk_days=int(args.wu_chunk_days),
        max_workers=int(args.wu_max_workers),
        max_retries=int(args.wu_max_retries),
        timeout_seconds=int(args.wu_timeout_seconds),
        skip_existing=bool(args.resume),
        logger=logger,
    )

    manifest = pd.read_csv(summary["manifest_path"])
    if not manifest.empty:
        manifest["station_id"] = target_station_id
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
        manifest.to_sql("wu_fetch_manifest", target_conn, if_exists="append", index=False)

    tz = ZoneInfo(station_zoneid)
    total_rows = 0
    final_csv_path = Path(summary["final_csv_path"])
    if final_csv_path.exists() and final_csv_path.stat().st_size > 0:
        for chunk in pd.read_csv(final_csv_path, chunksize=200000):
            if chunk.empty:
                continue
            chunk["station_id"] = target_station_id
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
            chunk.to_sql("wu_observations_30m", target_conn, if_exists="append", index=False)
            total_rows += int(len(chunk))

    target_conn.execute(
        """
        INSERT INTO wu_run_meta(station_id, meta_key, meta_value_json, updated_at_utc)
        VALUES(?,?,?,?)
        ON CONFLICT(station_id, meta_key) DO UPDATE SET
          meta_value_json=excluded.meta_value_json,
          updated_at_utc=excluded.updated_at_utc
        """,
        (target_station_id, "weathercom_run_summary", json.dumps(summary, sort_keys=True), now_utc()),
    )
    return {"mode": "download_weathercom", "rows": total_rows, "summary": summary}


def normalize_price(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def import_kalshi_from_root(
    target_conn: sqlite3.Connection,
    *,
    target_station_id: str,
    kalshi_root: Path,
) -> dict[str, Any]:
    clear_station_tables(target_conn, target_station_id, KALSHI_TABLES)
    manifest_path = kalshi_root / "manifest.json"
    manifest_payload: dict[str, Any] = {}
    if manifest_path.exists():
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    prefix = str(manifest_payload.get("file_prefix") or target_station_id).upper()

    csv_by_date: dict[str, Path] = {}
    for csv_path in sorted(kalshi_root.glob("*.csv")):
        stem = csv_path.stem
        if "_" not in stem:
            continue
        maybe_date = stem.rsplit("_", 1)[-1]
        if len(maybe_date) == 8 and maybe_date.isdigit():
            csv_by_date[f"{maybe_date[:4]}-{maybe_date[4:6]}-{maybe_date[6:8]}"] = csv_path

    manifest_rows: list[dict[str, Any]] = []
    minute_rows: list[tuple[str, str, str, str, float | None, str | None, str]] = []
    dates_payload = manifest_payload.get("dates") or {}
    all_dates = sorted(set(dates_payload.keys()) | set(csv_by_date.keys()))
    for target_date_local in all_dates:
        payload = dates_payload.get(target_date_local) or {}
        csv_path = csv_by_date.get(target_date_local) or kalshi_root / f"{prefix}_{target_date_local.replace('-', '')}.csv"
        csv_exists = csv_path.exists()
        market_tickers = payload.get("market_tickers") or []
        header_labels: list[str] = []
        rows_written_for_date = 0
        if csv_exists:
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, [])
                header_labels = header[1:]
                column_market_map: dict[str, str | None] = {}
                for idx, label in enumerate(header_labels):
                    column_market_map[label] = str(market_tickers[idx]) if idx < len(market_tickers) else None
                for row in reader:
                    if not row:
                        continue
                    timestamp_utc = str(row[0]).strip()
                    for idx, bucket_label in enumerate(header_labels, start=1):
                        if idx >= len(row):
                            continue
                        price = normalize_price(row[idx])
                        if price is None:
                            continue
                        rows_written_for_date += 1
                        minute_rows.append(
                            (
                                target_station_id,
                                target_date_local,
                                timestamp_utc,
                                bucket_label,
                                price,
                                column_market_map.get(bucket_label),
                                str(csv_path),
                            )
                        )
        manifest_rows.append(
            {
                "station_id": target_station_id,
                "target_date_local": target_date_local,
                "event_ticker": str(payload.get("event_ticker", "")),
                "market_tickers_json": json.dumps(market_tickers, sort_keys=True),
                "start_time_utc": payload.get("start_time"),
                "end_time_utc": payload.get("end_time"),
                "start_ts": payload.get("start_ts"),
                "end_ts": payload.get("end_ts"),
                "rows_written": int(payload.get("rows_written", 0) or 0) if payload else rows_written_for_date,
                "errors_json": json.dumps(payload.get("errors", []), sort_keys=True),
                "csv_path": str(csv_path) if csv_exists else "",
                "downloaded_at_utc": str(manifest_payload.get("generated_at_utc") or now_utc()),
            }
        )

    if manifest_rows:
        pd.DataFrame(manifest_rows).to_sql("kalshi_download_manifest", target_conn, if_exists="append", index=False)
    if minute_rows:
        target_conn.executemany(
            """
            INSERT INTO kalshi_minute_prices(
              station_id, target_date_local, timestamp_utc, bucket_label, yes_price, market_ticker, source_csv_path
            ) VALUES(?,?,?,?,?,?,?)
            """,
            minute_rows,
        )
    target_conn.execute(
        """
        INSERT INTO kalshi_run_meta(station_id, meta_key, meta_value_json, updated_at_utc)
        VALUES(?,?,?,?)
        ON CONFLICT(station_id, meta_key) DO UPDATE SET
          meta_value_json=excluded.meta_value_json,
          updated_at_utc=excluded.updated_at_utc
        """,
        (
            target_station_id,
            "kalshi_manifest",
            json.dumps({"manifest_path": str(manifest_path), "kalshi_root": str(kalshi_root)}, sort_keys=True),
            now_utc(),
        ),
    )
    return {
        "mode": "import_kalshi_root",
        "kalshi_root": str(kalshi_root),
        "manifest_rows": len(manifest_rows),
        "minute_rows": len(minute_rows),
    }


def maybe_download_kalshi(
    *,
    series_ticker: str,
    start_date: str,
    end_date: str,
    staging_root: Path,
    logger: logging.Logger,
    resume: bool,
) -> Path:
    staging_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(
            ROOT
            / "apps"
            / "ingestion-service"
            / "scripts"
            / "kalshi_download_temperature_minute.py"
        ),
        "--series",
        series_ticker,
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--out-dir",
        str(staging_root),
    ]
    cmd.append("--skip-existing" if resume else "--no-skip-existing")
    run_subprocess(cmd, logger, cwd=ROOT)
    return staging_root


def find_existing_truth_sqlite(existing_nws_root: Path, station_ids: list[str]) -> Path | None:
    for station_id in station_ids:
        station_root = existing_nws_root / station_id
        path = latest_file(station_root, ["*.sqlite"]) if station_root.exists() else None
        if path is not None:
            return path
    return None


def find_existing_truth_csv(training_data_root: Path, station_ids: list[str]) -> Path | None:
    root = training_data_root / "02_truth"
    if not root.exists():
        return None
    for station_id in station_ids:
        candidates: list[Path] = []
        for pattern in [
            f"{station_id}_settled_tmax_*.csv",
            f"{station_id}_settled_tmax.csv",
            f"{station_id}_settled_tmax_source_*.csv",
        ]:
            candidates.extend(root.glob(pattern))
        def ranker(path: Path) -> tuple[int, int, float]:
            name = path.name.lower()
            if "2000_2026" in name:
                priority = 5
            elif "2002_2026" in name:
                priority = 4
            elif "source_nws" in name:
                priority = 3
            elif "refresh" in name:
                priority = 1
            else:
                priority = 2
            return (priority, int(path.stat().st_size), float(path.stat().st_mtime))
        path = ranked_file(candidates, ranker=ranker)
        if path is not None:
            return path
    return None


def find_existing_mos_sqlite(existing_mos_root: Path, station_ids: list[str]) -> Path | None:
    for station_id in station_ids:
        station_root = existing_mos_root / station_id
        path = latest_file(station_root, ["*.sqlite"]) if station_root.exists() else None
        if path is not None:
            return path
    return None


def find_existing_mos_csv(training_data_root: Path, station_ids: list[str]) -> Path | None:
    root = training_data_root / "04_mos" / "archive_merged"
    if not root.exists():
        return None
    for station_id in station_ids:
        path = latest_file(root, [f"{station_id}_mos_archive_*.csv.gz", f"{station_id}_mos_archive_*.csv"])
        if path is not None:
            return path
    return None


def find_existing_wu_sqlite(existing_wu_root: Path, station_ids: list[str]) -> Path | None:
    for station_id in station_ids:
        station_root = existing_wu_root / station_id
        path = latest_file(station_root, ["*_early_peak_*.sqlite", "*.sqlite"]) if station_root.exists() else None
        if path is not None:
            return path
    return None


def find_existing_kalshi_root(existing_kalshi_root: Path, series_ticker: str, start_date: str, end_date: str) -> Path | None:
    exact = existing_kalshi_root / f"{series_ticker.lower()}_{start_date.replace('-', '_')}_to_{end_date.replace('-', '_')}"
    if exact.exists():
        return exact
    candidates = sorted(existing_kalshi_root.glob(f"{series_ticker.lower()}*"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def run_source_step(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    source_name: str,
    logger: logging.Logger,
    handler: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    started_at = now_utc()
    upsert_source_status(conn, run_id=run_id, source_name=source_name, status="RUNNING", started_at_utc=started_at, finished_at_utc=None)
    log_ingest_event(conn, run_id=run_id, level="INFO", component=source_name, message=f"{source_name} step started")
    conn.commit()
    try:
        result = json_safe(handler() or {})
    except Exception as exc:
        detail = {"error": str(exc)}
        finished_at = now_utc()
        upsert_source_status(
            conn,
            run_id=run_id,
            source_name=source_name,
            status="FAILED",
            started_at_utc=started_at,
            finished_at_utc=finished_at,
            detail=detail,
        )
        log_ingest_event(conn, run_id=run_id, level="ERROR", component=source_name, message=f"{source_name} step failed", detail=detail)
        conn.commit()
        logger.exception("SOURCE_STEP_FAILED source=%s error=%s", source_name, exc)
        return {"status": "FAILED", **detail}

    status = str(result.pop("status", "SUCCESS")).upper()
    finished_at = now_utc()
    upsert_source_status(
        conn,
        run_id=run_id,
        source_name=source_name,
        status=status,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        detail=result,
    )
    log_ingest_event(
        conn,
        run_id=run_id,
        level="INFO" if status in {"SUCCESS", "SKIPPED", "BLOCKED"} else "WARNING",
        component=source_name,
        message=f"{source_name} step completed",
        detail={"status": status, **result},
    )
    conn.commit()
    logger.info("SOURCE_STEP_DONE source=%s status=%s detail=%s", source_name, status, json.dumps(result, sort_keys=True))
    return {"status": status, **result}


def build_station_db(
    *,
    seed: StationSeed,
    metadata: ResolvedStationMetadata,
    args: argparse.Namespace,
    run_id: str,
    run_root: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    target_station_id = seed.station_id
    lookup_station_id = seed.metadata_lookup_station_id or seed.station_id
    source_station_candidates = [target_station_id]
    if lookup_station_id not in source_station_candidates:
        source_station_candidates.append(lookup_station_id)

    station_root = Path(args.data_root) / target_station_id
    station_root.mkdir(parents=True, exist_ok=True)
    db_path = station_root / f"{target_station_id}_pooled_strategy.sqlite"
    summary_history_dir = station_root / "_summaries"
    summary_history_dir.mkdir(parents=True, exist_ok=True)
    station_run_root = run_root / "stations" / target_station_id
    station_run_root.mkdir(parents=True, exist_ok=True)

    logger.info(
        "STATION_START station=%s lookup_station=%s timezone=%s kalshi_series=%s",
        target_station_id,
        lookup_station_id,
        metadata.station_zoneid,
        seed.kalshi_series or "",
    )

    conn = connect_station_db(db_path)
    try:
        begin_ingest_run(conn, run_id=run_id, stage_name="stage1_backfill")
        upsert_station_registry(conn, seed=seed, lookup_station_id=lookup_station_id, metadata=metadata)
        conn.commit()

        station_summary: dict[str, Any] = {
            "run_id": run_id,
            "station_id": target_station_id,
            "lookup_station_id": lookup_station_id,
            "db_path": str(db_path),
            "timezone": metadata.station_zoneid,
            "wu_location_id": metadata.wu_location_id,
            "nws_usw": metadata.nws_usw,
            "kalshi_series": seed.kalshi_series,
            "sources": {},
        }

        def nws_handler() -> dict[str, Any]:
            existing_db = find_existing_truth_sqlite(Path(args.existing_nws_root), source_station_candidates)
            if existing_db is not None:
                return import_nws_from_existing_sqlite(
                    conn,
                    target_station_id=target_station_id,
                    station_usw=metadata.nws_usw or "",
                    source_db_path=existing_db,
                )
            truth_csv = find_existing_truth_csv(Path(args.training_data_root), source_station_candidates)
            if truth_csv is not None:
                return import_nws_from_truth_csv(
                    conn,
                    target_station_id=target_station_id,
                    station_usw=metadata.nws_usw or "",
                    truth_csv_path=truth_csv,
                )
            if not metadata.nws_usw:
                return {"status": "BLOCKED", "reason": "missing_nws_usw_in_registry"}
            return download_and_ingest_nws(
                conn,
                target_station_id=target_station_id,
                station_usw=metadata.nws_usw,
                start_date=args.truth_start_date,
                end_date=args.truth_end_date,
                staging_root=station_run_root / "nws",
                logger=logger,
                resume=bool(args.resume),
            )

        def mos_handler() -> dict[str, Any]:
            existing_db = find_existing_mos_sqlite(Path(args.existing_mos_root), source_station_candidates)
            if existing_db is not None:
                return import_mos_from_existing_sqlite(
                    conn,
                    target_station_id=target_station_id,
                    source_db_path=existing_db,
                )
            archive_csv = find_existing_mos_csv(Path(args.training_data_root), source_station_candidates)
            if archive_csv is not None:
                return import_mos_from_archive_csv(
                    conn,
                    target_station_id=target_station_id,
                    archive_csv_path=archive_csv,
                )
            return download_and_ingest_mos(
                conn,
                target_station_id=target_station_id,
                source_station_id=lookup_station_id,
                start_year=int(args.mos_start_year),
                end_year=int(args.mos_end_year),
                staging_root=station_run_root / "mos",
                logger=logger,
                args=args,
            )

        def wu_handler() -> dict[str, Any]:
            existing_db = find_existing_wu_sqlite(Path(args.existing_wu_root), source_station_candidates)
            if existing_db is not None:
                return import_wu_from_existing_sqlite(
                    conn,
                    target_station_id=target_station_id,
                    source_db_path=existing_db,
                )
            if not args.weathercom_api_key:
                return {"status": "BLOCKED", "reason": "missing_weathercom_api_key"}
            return download_and_ingest_wu(
                conn,
                target_station_id=target_station_id,
                station_zoneid=metadata.station_zoneid,
                wu_location_id=metadata.wu_location_id,
                start_date=args.wu_start_date,
                end_date=args.wu_end_date,
                staging_root=station_run_root / "weathercom",
                weathercom_api_key=args.weathercom_api_key,
                logger=logger,
                args=args,
            )

        def kalshi_handler() -> dict[str, Any]:
            if not seed.kalshi_series:
                return {"status": "SKIPPED", "reason": "no_kalshi_series_for_station"}
            existing_root = find_existing_kalshi_root(
                Path(args.existing_kalshi_root),
                seed.kalshi_series,
                args.kalshi_start_date,
                args.kalshi_end_date,
            )
            if existing_root is not None:
                return import_kalshi_from_root(conn, target_station_id=target_station_id, kalshi_root=existing_root)
            if not args.download_missing_kalshi:
                return {"status": "SKIPPED", "reason": "kalshi_root_missing_and_download_disabled"}
            downloaded_root = maybe_download_kalshi(
                series_ticker=seed.kalshi_series,
                start_date=args.kalshi_start_date,
                end_date=args.kalshi_end_date,
                staging_root=station_run_root / "kalshi",
                logger=logger,
                resume=bool(args.resume),
            )
            return import_kalshi_from_root(conn, target_station_id=target_station_id, kalshi_root=downloaded_root)

        station_summary["sources"]["nws"] = run_source_step(conn, run_id=run_id, source_name="NWS", logger=logger, handler=nws_handler)
        station_summary["sources"]["mos"] = run_source_step(conn, run_id=run_id, source_name="MOS", logger=logger, handler=mos_handler)
        station_summary["sources"]["wu"] = run_source_step(conn, run_id=run_id, source_name="WU", logger=logger, handler=wu_handler)
        station_summary["sources"]["kalshi"] = run_source_step(conn, run_id=run_id, source_name="KALSHI", logger=logger, handler=kalshi_handler)

        counts = {
            "nws_truth_rows": count_station_rows(conn, "nws_truth_canonical", target_station_id),
            "mos_hourly_rows": count_station_rows(conn, "mos_hourly_values", target_station_id),
            "wu_rows": count_station_rows(conn, "wu_observations_30m", target_station_id),
            "kalshi_rows": count_station_rows(conn, "kalshi_minute_prices", target_station_id),
        }
        station_summary["counts"] = counts

        exports: dict[str, Any] = {}
        if counts["nws_truth_rows"] > 0:
            exports["truth_csv"] = export_truth_from_db(
                conn,
                target_station_id,
                Path(args.training_data_root) / "02_truth" / f"{target_station_id}_settled_tmax_{args.truth_start_date[:4]}_{args.truth_end_date[:4]}.csv",
            )
        if counts["mos_hourly_rows"] > 0:
            exports["mos_archive_csv_gz"] = export_mos_from_db(
                conn,
                target_station_id,
                Path(args.training_data_root)
                / "04_mos"
                / "archive_merged"
                / f"{target_station_id}_mos_archive_{args.mos_start_year}_{args.mos_end_year}.csv.gz",
            )
        station_summary["exports"] = exports

        source_statuses = [str(item.get("status", "")) for item in station_summary["sources"].values()]
        if any(status == "FAILED" for status in source_statuses):
            final_status = "PARTIAL_FAILURE"
        elif any(status == "BLOCKED" for status in source_statuses):
            final_status = "PARTIAL_SUCCESS"
        else:
            final_status = "SUCCESS"
        station_summary["final_status"] = final_status
        station_summary["generated_at_utc"] = now_utc()

        finish_ingest_run(conn, run_id=run_id, status=final_status, summary=station_summary)
        conn.commit()

        latest_summary_path = station_root / "stage1_latest_summary.json"
        history_summary_path = summary_history_dir / f"stage1_summary_{run_id}.json"
        latest_summary_path.write_text(json.dumps(json_safe(station_summary), indent=2, sort_keys=True), encoding="utf-8")
        history_summary_path.write_text(json.dumps(json_safe(station_summary), indent=2, sort_keys=True), encoding="utf-8")
        return json_safe(station_summary)
    finally:
        conn.close()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = data_root / "_runs" / run_id
    logger = configure_logger(run_root, args.log_level)

    seeds = get_station_seeds(args.stations)
    logger.info(
        "STAGE1_RUN_START run_id=%s stations=%d scope=%s data_root=%s training_data_root=%s weathercom_key_present=%s",
        run_id,
        len(seeds),
        args.stations,
        data_root,
        args.training_data_root,
        bool(args.weathercom_api_key),
    )

    registry_cache_dir = data_root / "_registry_cache"
    registry = build_iem_airport_registry(
        cache_dir=registry_cache_dir,
        refresh=bool(args.refresh_registry_cache),
        logger=logger,
    )

    registry_dir = data_root / "_registry"
    all_rows = build_station_crosswalk_rows(get_station_seeds("all"), registry)
    active_rows = build_station_crosswalk_rows(get_station_seeds("active"), registry)
    reserve_rows = build_station_crosswalk_rows(get_station_seeds("reserve"), registry)
    selected_rows = build_station_crosswalk_rows(seeds, registry)
    write_csv(registry_dir / "station_crosswalk_all.csv", all_rows)
    write_csv(registry_dir / "station_crosswalk_active.csv", active_rows)
    write_csv(registry_dir / "station_crosswalk_reserve.csv", reserve_rows)
    write_csv(registry_dir / f"station_crosswalk_selected_{run_id}.csv", selected_rows)

    station_results: list[dict[str, Any]] = []
    failed_station_builds: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds, start=1):
        lookup_station_id = seed.metadata_lookup_station_id or seed.station_id
        try:
            metadata = resolve_station_metadata(lookup_station_id, registry)
            result = build_station_db(
                seed=seed,
                metadata=metadata,
                args=args,
                run_id=run_id,
                run_root=run_root,
                logger=logger,
            )
            station_results.append(result)
            logger.info(
                "STATION_DONE index=%d/%d station=%s final_status=%s",
                index,
                len(seeds),
                seed.station_id,
                result.get("final_status", ""),
            )
        except Exception as exc:
            logger.exception("STATION_BUILD_FAILED station=%s error=%s", seed.station_id, exc)
            failed_station_builds.append({"station_id": seed.station_id, "error": str(exc)})

    status_counts: dict[str, int] = {}
    for item in station_results:
        status = str(item.get("final_status", "UNKNOWN"))
        status_counts[status] = status_counts.get(status, 0) + 1
    if failed_station_builds:
        status_counts["STATION_BUILD_FAILED"] = len(failed_station_builds)

    summary = {
        "run_id": run_id,
        "generated_at_utc": now_utc(),
        "scope": args.stations,
        "selected_station_count": len(seeds),
        "status_counts": status_counts,
        "data_root": str(data_root),
        "training_data_root": str(Path(args.training_data_root).resolve()),
        "existing_nws_root": str(Path(args.existing_nws_root).resolve()),
        "existing_mos_root": str(Path(args.existing_mos_root).resolve()),
        "existing_wu_root": str(Path(args.existing_wu_root).resolve()),
        "existing_kalshi_root": str(Path(args.existing_kalshi_root).resolve()),
        "weathercom_key_present": bool(args.weathercom_api_key),
        "station_results": station_results,
        "failed_station_builds": failed_station_builds,
        "registry_files": {
            "all": str(registry_dir / "station_crosswalk_all.csv"),
            "active": str(registry_dir / "station_crosswalk_active.csv"),
            "reserve": str(registry_dir / "station_crosswalk_reserve.csv"),
            "selected": str(registry_dir / f"station_crosswalk_selected_{run_id}.csv"),
        },
    }
    summary_path = run_root / "stage1_summary.json"
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
    logger.info("STAGE1_RUN_DONE summary=%s", summary_path)
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True))
    return 0 if not failed_station_builds else 1


if __name__ == "__main__":
    raise SystemExit(main())
