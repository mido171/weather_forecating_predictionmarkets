from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import sqlite3
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def repo_root() -> Path:
    cur = Path(__file__).resolve()
    for p in cur.parents:
        if (p / "pom.xml").exists() and (p / "ingestion-service").exists():
            return p
    raise FileNotFoundError("repo root not found")


ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.live import mos_blend12_bundle  # noqa: E402
from tools.station_flow.station_metadata import resolve_station_metadata  # noqa: E402


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run(cmd: List[str], log: logging.Logger) -> Dict[str, Any]:
    out = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, check=False)
    so = (out.stdout or "")[-3000:]
    se = (out.stderr or "")[-3000:]
    if out.returncode != 0:
        raise RuntimeError(f"rc={out.returncode}: {' '.join(cmd)}\nstdout={so}\nstderr={se}")
    log.info("CMD_OK %s", " ".join(cmd))
    return {"cmd": cmd, "returncode": out.returncode, "stdout_tail": so, "stderr_tail": se}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Station full flow (SQLite-first)")
    p.add_argument("--station-id", required=True)
    p.add_argument("--nws-code", required=True)
    p.add_argument("--data-root", default=r"D:\Ahmed\data")
    p.add_argument("--mos-station-id", default=None)
    p.add_argument("--station-zoneid", default=None)
    p.add_argument("--truth-start-date", default="2002-01-01")
    p.add_argument("--truth-end-date", default="2026-12-31")
    p.add_argument("--mos-start-year", type=int, default=2002)
    p.add_argument("--mos-end-year", type=int, default=2026)
    p.add_argument("--kalshi-start-date", default="2024-10-01")
    p.add_argument("--kalshi-end-date", default=datetime.now(timezone.utc).date().isoformat())
    p.add_argument("--dev-start", default="2022-01-01")
    p.add_argument("--dev-end", default="2023-12-31")
    p.add_argument("--test-start", default="2024-01-01")
    p.add_argument("--test-end", default="2025-12-31")
    p.add_argument("--target-date", default=None)
    p.add_argument("--backtest-mode", choices=["single", "cojoined"], default="single")
    p.add_argument("--run-cojoined", action="store_true")
    p.add_argument("--cojoined-stations", default="")
    p.add_argument("--resume", dest="resume", action="store_true")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.set_defaults(resume=True)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def init_nws_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS nws_raw_snapshots(station_id TEXT,station_usw TEXT,window_start_date TEXT,window_end_date TEXT,request_url TEXT,response_path TEXT,headers_path TEXT,retrieved_at_utc TEXT,http_status INTEGER,body_sha256 TEXT,byte_count INTEGER,inserted_at_utc TEXT);
        CREATE TABLE IF NOT EXISTS nws_truth_canonical(station_id TEXT NOT NULL,station_usw TEXT NOT NULL,target_date_local TEXT NOT NULL,tmax_f INTEGER,truth_source TEXT,source_record_id TEXT,retrieved_at_utc TEXT,PRIMARY KEY(station_id,target_date_local));
        CREATE TABLE IF NOT EXISTS nws_truth_enriched(station_id TEXT,station_usw TEXT,target_date_local TEXT,tmax_f INTEGER,truth_source TEXT,source_record_id TEXT,retrieved_at_utc TEXT,attribute_measurement_flag TEXT,attribute_quality_flag TEXT,attribute_source_flag TEXT,attribute_obs_time_hhmm TEXT,attribute_raw TEXT,source_station_field TEXT);
        CREATE TABLE IF NOT EXISTS nws_qa_reports(station_id TEXT,start_date TEXT,end_date TEXT,rows_count INTEGER,duplicate_station_date_rows INTEGER,missing_dates_count INTEGER,qa_json TEXT,qa_md_path TEXT,inserted_at_utc TEXT);
        CREATE TABLE IF NOT EXISTS nws_run_meta(station_id TEXT NOT NULL,meta_key TEXT NOT NULL,meta_value_json TEXT NOT NULL,updated_at_utc TEXT NOT NULL,PRIMARY KEY(station_id,meta_key));
        """
    )


def init_mos_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS mos_raw_payloads(station_id TEXT,model TEXT,year INTEGER,request_params_json TEXT,retrieved_at_utc TEXT,response_sha256 TEXT,row_count INTEGER,runtime_hour_counts_json TEXT,yearly_csv_gz TEXT,raw_json_gz TEXT,meta_file TEXT,PRIMARY KEY(station_id,model,year));
        CREATE TABLE IF NOT EXISTS mos_hourly_values(station_id TEXT,model TEXT,year INTEGER,runtime_utc TEXT,forecast_time_utc TEXT,retrieved_at_utc TEXT,response_sha256 TEXT,tmp REAL,dpt REAL,cld REAL,sky REAL,wdr REAL,wsp REAL,gst REAL,p06 REAL,p12 REAL,t06 REAL,t12 REAL,q06 REAL,q12 REAL,n_x REAL,n_n REAL,cig REAL,vis REAL,tmp_raw TEXT,dpt_raw TEXT,cld_raw TEXT,sky_raw TEXT,wdr_raw TEXT,wsp_raw TEXT,gst_raw TEXT,p06_raw TEXT,p12_raw TEXT,t06_raw TEXT,t12_raw TEXT,q06_raw TEXT,q12_raw TEXT,n_x_raw TEXT,n_n_raw TEXT,cig_raw TEXT,vis_raw TEXT);
        CREATE TABLE IF NOT EXISTS mos_download_manifest(station_id TEXT,model TEXT,year INTEGER,status TEXT,row_count INTEGER,column_count INTEGER,yearly_file TEXT,raw_file TEXT,meta_file TEXT);
        CREATE TABLE IF NOT EXISTS mos_run_meta(station_id TEXT NOT NULL,meta_key TEXT NOT NULL,meta_value_json TEXT NOT NULL,updated_at_utc TEXT NOT NULL,PRIMARY KEY(station_id,meta_key));
        """
    )


def ingest_nws(station: str, nws_code: str, root: Path, summary: Dict[str, Any], db: Path) -> Dict[str, Any]:
    can = pd.read_csv(summary["canonical_csv"])
    enr = pd.read_csv(summary["enriched_csv"])
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db)) as conn:
        init_nws_db(conn)
        for t in ["nws_raw_snapshots", "nws_truth_canonical", "nws_truth_enriched", "nws_qa_reports"]:
            conn.execute(f"DELETE FROM {t} WHERE station_id=?", (station,))
        raws = []
        rr = root / "raw" / "ncei_ads" / "daily-summaries" / nws_code.upper()
        for d in sorted(rr.glob("*_*")):
            if not d.is_dir():
                continue
            a, b = d.name.split("_", 1)
            r = d / "response.json"
            raws.append(
                (
                    station,
                    nws_code.upper(),
                    a,
                    b,
                    (d / "request_url.txt").read_text(encoding="utf-8").strip() if (d / "request_url.txt").exists() else "",
                    str(r),
                    str(d / "headers.json"),
                    (d / "retrieved_at_utc.txt").read_text(encoding="utf-8").strip()
                    if (d / "retrieved_at_utc.txt").exists()
                    else "",
                    int((d / "http_status.txt").read_text(encoding="utf-8").strip() or "0")
                    if (d / "http_status.txt").exists()
                    else 0,
                    (d / "sha256.txt").read_text(encoding="utf-8").strip() if (d / "sha256.txt").exists() else "",
                    int(r.stat().st_size) if r.exists() else 0,
                    now_utc(),
                )
            )
        conn.executemany("INSERT INTO nws_raw_snapshots VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", raws)
        can["station_id"] = can["station_id"].astype(str).str.upper()
        can = can[can["station_id"] == station].copy()
        can["target_date_local"] = pd.to_datetime(can["target_date_local"], errors="coerce").dt.date.astype(str)
        can["tmax_f"] = pd.to_numeric(can["tmax_f"], errors="coerce")
        rows = [
            (
                station,
                nws_code.upper(),
                str(r.target_date_local),
                int(r.tmax_f),
                str(r.truth_source),
                str(r.source_record_id),
                str(r.retrieved_at_utc),
            )
            for r in can.itertuples()
            if pd.notna(r.tmax_f)
        ]
        conn.executemany(
            "INSERT INTO nws_truth_canonical VALUES(?,?,?,?,?,?,?) ON CONFLICT(station_id,target_date_local) DO UPDATE SET station_usw=excluded.station_usw,tmax_f=excluded.tmax_f,truth_source=excluded.truth_source,source_record_id=excluded.source_record_id,retrieved_at_utc=excluded.retrieved_at_utc",
            rows,
        )
        enr["station_id"] = enr["station_id"].astype(str).str.upper()
        enr = enr[enr["station_id"] == station].copy()
        enr["target_date_local"] = pd.to_datetime(enr["target_date_local"], errors="coerce").dt.date.astype(str)
        vals = [
            (
                station,
                str(getattr(r, "station_usw", nws_code)).upper(),
                str(r.target_date_local),
                int(r.tmax_f) if pd.notna(r.tmax_f) else None,
                str(getattr(r, "truth_source", "")),
                str(getattr(r, "source_record_id", "")),
                str(getattr(r, "retrieved_at_utc", "")),
                str(getattr(r, "attribute_measurement_flag", "")),
                str(getattr(r, "attribute_quality_flag", "")),
                str(getattr(r, "attribute_source_flag", "")),
                str(getattr(r, "attribute_obs_time_hhmm", "")),
                str(getattr(r, "attribute_raw", "")),
                str(getattr(r, "source_station_field", "")),
            )
            for r in enr.itertuples()
        ]
        conn.executemany("INSERT INTO nws_truth_enriched VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", vals)
        qa = summary.get(f"qa_{station}", {}) or {}
        qj = {}
        if qa.get("json") and Path(qa["json"]).exists():
            qj = json.loads(Path(qa["json"]).read_text(encoding="utf-8"))
        conn.execute(
            "INSERT INTO nws_qa_reports VALUES(?,?,?,?,?,?,?,?,?)",
            (
                station,
                str(summary.get("start_date", "")),
                str(summary.get("end_date", "")),
                int(qa.get("rows", 0) or 0),
                int(qa.get("duplicate_station_date_rows", 0) or 0),
                int(qa.get("missing_dates_count", 0) or 0),
                json.dumps(qj),
                str(qa.get("md", "")),
                now_utc(),
            ),
        )
        conn.execute(
            "INSERT INTO nws_run_meta VALUES(?,?,?,?) ON CONFLICT(station_id,meta_key) DO UPDATE SET meta_value_json=excluded.meta_value_json,updated_at_utc=excluded.updated_at_utc",
            (station, "latest_summary", json.dumps(summary), now_utc()),
        )
        conn.commit()
    return {"db_path": str(db), "raw_rows": len(raws), "canonical_rows": len(rows), "enriched_rows": len(vals)}


def export_truth(db: Path, station: str, out: Path) -> Dict[str, Any]:
    with sqlite3.connect(str(db)) as conn:
        df = pd.read_sql_query(
            "SELECT station_id,target_date_local as date,tmax_f as settled_tmax FROM nws_truth_canonical WHERE station_id=? ORDER BY target_date_local",
            conn,
            params=(station,),
        )
    if df.empty:
        raise ValueError("truth export empty")
    df["settled_tmax"] = pd.to_numeric(df["settled_tmax"], errors="coerce").round().astype("Int64")
    df = df.dropna(subset=["date", "settled_tmax"]).drop_duplicates(["station_id", "date"], keep="last")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return {"path": str(out), "rows": int(len(df)), "sha256": sha256(out)}


def ingest_mos(station: str, dl: Path, merged: Path, manifest: Path, run_meta: Path, db: Path) -> Dict[str, Any]:
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db)) as conn:
        init_mos_db(conn)
        for t in ["mos_raw_payloads", "mos_hourly_values", "mos_download_manifest"]:
            conn.execute(f"DELETE FROM {t} WHERE station_id=?", (station,))
        meta_rows = []
        for p in sorted((dl / "mos_meta").glob("*.json")):
            j = json.loads(p.read_text(encoding="utf-8"))
            meta_rows.append(
                (
                    station,
                    str(j.get("model", "")).upper(),
                    int(j.get("year", 0) or 0),
                    json.dumps(j.get("request_params", {}), sort_keys=True),
                    str(j.get("retrieved_at_utc", "")),
                    str(j.get("response_sha256", "")),
                    int(j.get("row_count", 0) or 0),
                    json.dumps(j.get("runtime_hour_counts", {}), sort_keys=True),
                    str(j.get("yearly_csv_gz", "")),
                    str(j.get("raw_json_gz", "")),
                    str(p),
                )
            )
        conn.executemany(
            "INSERT INTO mos_raw_payloads VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(station_id,model,year) DO UPDATE SET request_params_json=excluded.request_params_json,retrieved_at_utc=excluded.retrieved_at_utc,response_sha256=excluded.response_sha256,row_count=excluded.row_count,runtime_hour_counts_json=excluded.runtime_hour_counts_json,yearly_csv_gz=excluded.yearly_csv_gz,raw_json_gz=excluded.raw_json_gz,meta_file=excluded.meta_file",
            meta_rows,
        )
        total = 0
        for c in pd.read_csv(merged, chunksize=200000):
            c["station_id"] = station
            c.to_sql("mos_hourly_values", conn, if_exists="append", index=False)
            total += len(c)
        man = pd.read_csv(manifest)
        if "station_id" not in man.columns:
            man["station_id"] = station
        man.to_sql("mos_download_manifest", conn, if_exists="append", index=False)
        meta = json.loads(run_meta.read_text(encoding="utf-8"))
        conn.execute(
            "INSERT INTO mos_run_meta VALUES(?,?,?,?) ON CONFLICT(station_id,meta_key) DO UPDATE SET meta_value_json=excluded.meta_value_json,updated_at_utc=excluded.updated_at_utc",
            (station, "latest_run_meta", json.dumps(meta), now_utc()),
        )
        conn.commit()
    return {
        "db_path": str(db),
        "raw_payload_rows": len(meta_rows),
        "hourly_rows": int(total),
        "manifest_rows": int(len(man)),
    }


def export_mos(db: Path, station: str, out: Path) -> Dict[str, Any]:
    with sqlite3.connect(str(db)) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM mos_hourly_values WHERE station_id=? ORDER BY runtime_utc,forecast_time_utc",
            conn,
            params=(station,),
        )
    if df.empty:
        raise ValueError("mos export empty")
    models = {str(k).upper(): int(v) for k, v in df["model"].astype(str).str.upper().value_counts().to_dict().items()}
    if models.get("GFS", 0) <= 0 or models.get("NAM", 0) <= 0:
        raise ValueError(f"missing GFS/NAM in export: {models}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False)
    return {"path": str(out), "rows": int(len(df)), "sha256": sha256(out), "model_counts": models}


def kalshi_dir(data_root: Path, series: str, start: str, end: str) -> Path:
    return data_root / "kalshi" / "kalshi_history" / f"{series.lower()}_{start.replace('-', '_')}_to_{end.replace('-', '_')}"


def expected_runtime(target_date: str) -> str:
    d = date.fromisoformat(target_date)
    rt = datetime(d.year, d.month, d.day, 12, 0, tzinfo=timezone.utc) - pd.Timedelta(days=1)
    return pd.Timestamp(rt).isoformat().replace("+00:00", "Z")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger("station_flow")

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = data_root / "runs" / "station_full_flow" / str(args.station_id).strip().lower() / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    resolved = resolve_station_metadata(
        station_id_series=args.station_id,
        mos_station_id_override=args.mos_station_id,
        station_zoneid_override=args.station_zoneid,
    )
    station = resolved.station_id

    commands: List[Dict[str, Any]] = []
    write_json(
        run_root / "resolved_inputs.json",
        {
            "generated_at_utc": now_utc(),
            "inputs": vars(args),
            "resolved_station": {
                "series_ticker": resolved.series_ticker,
                "station_id": resolved.station_id,
                "station_zoneid": resolved.station_zoneid,
                "file_prefix": resolved.file_prefix,
                "settlement_url": resolved.settlement_url,
                "issuedby": resolved.issuedby,
                "source": resolved.source,
            },
        },
    )

    ncei_root = run_root / "phase1" / "nws"
    cfg = ncei_root / "config.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        "\n".join(
            [
                "station_map:",
                f"  {station}: {str(args.nws_code).upper()}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    truth_csv = data_root / "kalshi" / "training_data" / "02_truth" / f"{station}_settled_tmax_2002_2026.csv"
    cmd_nws = [
        sys.executable,
        str(ROOT / "tools" / "ncei_truth" / "run.py"),
        "--config",
        str(cfg),
        "--stations",
        station,
        "--start-date",
        args.truth_start_date,
        "--end-date",
        args.truth_end_date,
        "--root-dir",
        str(ncei_root),
        "--write-simple-settlement-csv-path",
        str(truth_csv),
        "--simple-settlement-station-id",
        station,
        "--log-level",
        args.log_level,
    ]
    commands.append(run(cmd_nws, log))
    summary_path = sorted((ncei_root / "reports").glob("run_summary_*.json"), key=lambda p: p.stat().st_mtime)[-1]
    ncei_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    nws_db = data_root / "sqlite" / "NWS" / station / f"{station}_nws_truth_{args.truth_start_date[:4]}_{args.truth_end_date[:4]}.sqlite"
    nws_ing = ingest_nws(station, str(args.nws_code), ncei_root, ncei_summary, nws_db)
    truth_exp = export_truth(nws_db, station, truth_csv)

    mos_dl = data_root / "sqlite" / "MOS" / station / "raw_iem" / f"{station}_{args.mos_start_year}_{args.mos_end_year}"
    mos_dl.mkdir(parents=True, exist_ok=True)
    cmd_mos = [
        sys.executable,
        str(ROOT / "ingestion-service" / "scripts" / "download_iem_mos_kncy_archive.py"),
        "--station",
        station,
        "--models",
        "GFS",
        "NAM",
        "--start-year",
        str(args.mos_start_year),
        "--end-year",
        str(args.mos_end_year),
        "--output-dir",
        str(mos_dl),
        "--log-level",
        args.log_level,
    ]
    if args.resume:
        cmd_mos.append("--resume")
    commands.append(run(cmd_mos, log))

    mos_merged = mos_dl / f"{station}_mos_archive_{args.mos_start_year}_{args.mos_end_year}.csv.gz"
    mos_manifest = mos_dl / f"{station}_mos_download_manifest_{args.mos_start_year}_{args.mos_end_year}.csv"
    mos_meta = mos_dl / f"{station}_mos_download_run_meta_{args.mos_start_year}_{args.mos_end_year}.json"
    mos_db = data_root / "sqlite" / "MOS" / station / f"{station}_mos_{args.mos_start_year}_{args.mos_end_year}.sqlite"
    mos_ing = ingest_mos(station, mos_dl, mos_merged, mos_manifest, mos_meta, mos_db)
    mos_csv = data_root / "kalshi" / "training_data" / "04_mos" / "archive_merged" / f"{station}_mos_archive_{args.mos_start_year}_{args.mos_end_year}.csv.gz"
    mos_exp = export_mos(mos_db, station, mos_csv)

    kalshi_root = kalshi_dir(data_root, resolved.series_ticker, args.kalshi_start_date, args.kalshi_end_date)
    kalshi_root.mkdir(parents=True, exist_ok=True)
    cmd_kalshi = [
        sys.executable,
        str(ROOT / "ingestion-service" / "scripts" / "kalshi_download_temperature_minute.py"),
        "--series",
        resolved.series_ticker,
        "--file-prefix",
        resolved.file_prefix,
        "--start-date",
        args.kalshi_start_date,
        "--end-date",
        args.kalshi_end_date,
        "--out-dir",
        str(kalshi_root),
    ]
    cmd_kalshi.append("--skip-existing" if args.resume else "--no-skip-existing")
    commands.append(run(cmd_kalshi, log))

    exp_root = data_root / "kalshi" / "Experiments" / f"MOS_{station}"
    cmd_train = [
        sys.executable,
        str(ROOT / "ml" / "run_knyc_mos_first_plan.py"),
        "--station-id",
        station,
        "--station-zoneid",
        resolved.station_zoneid,
        "--mos-csv",
        str(mos_csv),
        "--truth-csv",
        str(truth_csv),
        "--out-root",
        str(exp_root),
        "--dev-start",
        args.dev_start,
        "--dev-end",
        args.dev_end,
        "--test-start",
        args.test_start,
        "--test-end",
        args.test_end,
    ]
    commands.append(run(cmd_train, log))

    bundle = exp_root / "03_blends" / "blend_12" / f"live_model_bundle_v2_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
    bundle_manifest = mos_blend12_bundle.train_and_write_bundle(
        station_id=station,
        station_zoneid=resolved.station_zoneid,
        mos_archive_path=mos_csv,
        truth_csv_path=truth_csv,
        bundle_dir=bundle,
        logger=log,
    )

    smoke_date = str(args.target_date or args.kalshi_start_date)
    parity_out = run_root / "phase3_parity_smoke"
    cmd_parity = [
        sys.executable,
        str(ROOT / "tools" / "live" / "mos_quantile_live_inference.py"),
        "--target-date",
        smoke_date,
        "--station-id",
        station,
        "--station-zoneid",
        resolved.station_zoneid,
        "--series",
        resolved.series_ticker,
        "--file-prefix",
        resolved.file_prefix,
        "--bundle-dir",
        str(bundle),
        "--mos-archive",
        str(mos_csv),
        "--truth-csv",
        str(truth_csv),
        "--market-root",
        str(kalshi_root),
        "--out-dir",
        str(parity_out),
        "--no-auto-download-market",
        "--log-level",
        args.log_level,
    ]
    commands.append(run(cmd_parity, log))
    parity_report = json.loads((parity_out / "inference_report.json").read_text(encoding="utf-8"))
    blk = (parity_report.get("inference_by_station") or {}).get(station) or parity_report.get(f"inference_{station.lower()}")
    if not blk:
        raise ValueError("parity missing station block")
    if str(blk.get("runtime_utc")) != expected_runtime(smoke_date):
        raise ValueError("parity runtime mismatch")
    if not bool((parity_report.get("leakage_proof") or {}).get("passes_all_guardrails", False)):
        raise ValueError("parity guardrail failure")

    live_report = None
    if args.target_date:
        live_out = data_root / "live" / "mos_quantile_live_inference" / f"{run_id}_target_{str(args.target_date).replace('-', '')}_{station.lower()}"
        cmd_live = cmd_parity.copy()
        i = cmd_live.index("--target-date")
        cmd_live[i + 1] = str(args.target_date)
        j = cmd_live.index("--out-dir")
        cmd_live[j + 1] = str(live_out)
        commands.append(run(cmd_live, log))
        live_report = str(live_out / "inference_report.json")

    mode = "cojoined" if args.run_cojoined else args.backtest_mode
    stations = [station]
    extra_prefix: Dict[str, str] = {}
    extra_series: Dict[str, str] = {}
    if mode == "cojoined":
        extras = [x.strip().upper() for x in str(args.cojoined_stations).split(",") if x.strip()]
        for x in extras:
            sid = x
            if x.startswith("KX"):
                try:
                    m = resolve_station_metadata(station_id_series=x)
                    sid = m.station_id
                    extra_prefix[sid] = m.file_prefix
                    extra_series[sid] = m.series_ticker
                except Exception:
                    sid = x
            if sid not in stations:
                stations.append(sid)

    pred_dev = {station: str(exp_root / "03_blends" / "blend_12" / "dev_predictions.parquet")}
    pred_test = {station: str(exp_root / "03_blends" / "blend_12" / "test_predictions.parquet")}
    truth_map = {station: str(truth_csv)}
    kalshi_map = {station: str(kalshi_root)}
    prefix_map = {station: resolved.file_prefix}
    for s in stations:
        if s == station:
            continue
        pr = data_root / "kalshi" / "Experiments" / f"MOS_{s}" / "03_blends" / "blend_12"
        if not pr.exists():
            pr = data_root / "kalshi" / "Experiments" / "MOS" / "03_blends" / "blend_12"
        pred_dev[s] = str(pr / "dev_predictions.parquet")
        pred_test[s] = str(pr / "test_predictions.parquet")
        t1 = data_root / "kalshi" / "training_data" / "02_truth" / f"{s}_settled_tmax_2002_2026.csv"
        t2 = data_root / "kalshi" / "training_data" / "02_truth" / f"{s}_settled_tmax.csv"
        truth_map[s] = str(t1 if t1.exists() else t2)
        kd = []
        if s in extra_series:
            kd = sorted(
                (data_root / "kalshi" / "kalshi_history").glob(f"{extra_series[s].lower()}_*"),
                key=lambda p: p.stat().st_mtime,
            )
        if not kd:
            kd = sorted((data_root / "kalshi" / "kalshi_history").glob(f"*{s.lower()}*"), key=lambda p: p.stat().st_mtime)
        if not kd:
            raise FileNotFoundError(f"kalshi root not found for {s}")
        kalshi_map[s] = str(kd[-1])
        prefix_map[s] = extra_prefix.get(s, s)

    bt_start = max(date.fromisoformat(args.kalshi_start_date), date.fromisoformat(args.test_start)).isoformat()
    bt_end = min(date.fromisoformat(args.kalshi_end_date), date.fromisoformat(args.test_end)).isoformat()
    bt_out = data_root / "kalshi" / "Experiments" / f"MOS_{station}" / "05_backtest"
    out_prefix = f"{mode}_blend12_{'_'.join([s.lower() for s in stations])}_tminus1_1200z_openplus30m_ev0p18_win67_risk6p5_cap500"
    cmd_bt = [
        sys.executable,
        str(ROOT / "backtesting" / "mos_blend12_knyc_kmia_cojoined_audit.py"),
        "--mode",
        mode,
        "--stations",
        ",".join(stations),
        "--prediction-source",
        "parquet",
        "--pred-dev-by-station-json",
        json.dumps(pred_dev),
        "--pred-test-by-station-json",
        json.dumps(pred_test),
        "--truth-csv-by-station-json",
        json.dumps(truth_map),
        "--kalshi-root-by-station-json",
        json.dumps(kalshi_map),
        "--file-prefix-by-station-json",
        json.dumps(prefix_map),
        "--start-date",
        bt_start,
        "--end-date",
        bt_end,
        "--entry-hour-z",
        "12",
        "--entry-minute-z",
        "0",
        "--min-entry-minutes-after-open",
        "30",
        "--ev-min",
        "0.18",
        "--win-min",
        "0.67",
        "--risk-fraction",
        "0.065",
        "--stake-cap-usd",
        "500",
        "--out-dir",
        str(bt_out),
        "--out-prefix",
        out_prefix,
        "--table-out",
        str(data_root / "kalshi" / "plots" / f"{out_prefix}_stockholm_table.csv"),
    ]
    commands.append(run(cmd_bt, log))

    summary_json = bt_out / f"summary_{out_prefix}.json"
    sanity_json = bt_out / f"sanity_{out_prefix}.json"
    manifest = {
        "schema": "station_full_flow_run_manifest_v1",
        "generated_at_utc": now_utc(),
        "run_root": str(run_root),
        "inputs": vars(args),
        "resolved_station": {
            "series_ticker": resolved.series_ticker,
            "station_id": resolved.station_id,
            "station_zoneid": resolved.station_zoneid,
            "file_prefix": resolved.file_prefix,
        },
        "phase_outputs": {
            "nws_sqlite": nws_ing,
            "mos_sqlite": mos_ing,
            "truth_export": truth_exp,
            "mos_export": mos_exp,
            "kalshi_root": str(kalshi_root),
            "bundle_dir": str(bundle),
            "bundle_manifest": bundle_manifest,
            "bundle_manifest_sha256": sha256(bundle / "manifest.json"),
            "parity_report": str(parity_out / "inference_report.json"),
            "live_inference_report": live_report,
            "backtest_summary_json": str(summary_json),
            "backtest_sanity_json": str(sanity_json),
        },
        "commands": commands,
    }
    manifest_path = run_root / "run_manifest.json"
    write_json(manifest_path, manifest)
    write_json(
        run_root / "run_notes.json",
        {
            "run_manifest": str(manifest_path),
            "station_id": station,
            "bundle_dir": str(bundle),
            "backtest_summary_json": str(summary_json),
            "backtest_sanity_json": str(sanity_json),
        },
    )
    print(
        json.dumps(
            {
                "run_manifest": str(manifest_path),
                "truth_export": truth_exp,
                "mos_export": mos_exp,
                "bundle_dir": str(bundle),
                "backtest_summary_json": str(summary_json),
                "backtest_sanity_json": str(sanity_json),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
