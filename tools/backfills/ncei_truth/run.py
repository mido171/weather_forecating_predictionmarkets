from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from download import run_download
from normalize import normalize_snapshots_to_rows, write_canonical_csv, write_enriched_rows, write_klga_training_truth
from qa import build_station_qa_report, write_station_qa_markdown


def _timestamp_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_date(v: str) -> date:
    return date.fromisoformat(v)


def _load_station_map_from_yaml(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    # Minimal parser for the fixed config structure:
    # station_map:
    #   KNYC: USW00094728
    #   KLGA: USW00014732
    out: dict[str, str] = {}
    in_map = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            continue
        if re.match(r"^\s*station_map\s*:\s*$", line):
            in_map = True
            continue
        if in_map:
            if re.match(r"^\S", line):
                # New top-level key -> stop map parse.
                break
            m = re.match(r"^\s+([A-Za-z0-9_:-]+)\s*:\s*([A-Za-z0-9_:-]+)\s*$", line)
            if m:
                out[m.group(1).strip().upper()] = m.group(2).strip().upper()
    if not out:
        raise ValueError(f"Could not parse station_map from {path}")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download and canonicalize NCEI daily-summaries TMAX truth with provenance snapshots."
    )
    p.add_argument("--config", default=str(THIS_DIR / "config.yaml"))
    p.add_argument("--stations", default="KNYC,KLGA", help="Comma-separated station ids from config station_map.")
    p.add_argument("--start-date", default="1973-01-01")
    p.add_argument("--end-date", default="2026-12-31")
    p.add_argument(
        "--root-dir",
        default=str((THIS_DIR.parents[2] / "data" / "truth_tmax").resolve()),
        help="Base output root for canonical/raw/manifests/reports.",
    )
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--no-skip-existing", action="store_true", help="Force re-fetch even if raw snapshot exists.")
    p.add_argument(
        "--write-klga-training-truth-path",
        default="",
        help="Optional output CSV path for KLGA training truth contract (request_location_id,target_date_local,max_temp_f,station_zoneid).",
    )
    p.add_argument(
        "--write-simple-settlement-csv-path",
        default="",
        help="Optional output CSV path with exactly: station_id,date,settled_tmax.",
    )
    p.add_argument(
        "--simple-settlement-station-id",
        default="",
        help="Optional station id for simple settlement CSV (defaults to single --stations entry when unambiguous).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("ncei_truth")

    cfg_path = Path(args.config).resolve()
    station_map_all = _load_station_map_from_yaml(cfg_path)
    selected = [x.strip().upper() for x in str(args.stations).split(",") if x.strip()]
    station_map: dict[str, str] = {}
    for sid in selected:
        if sid not in station_map_all:
            raise ValueError(f"Station {sid} not found in config station_map.")
        station_map[sid] = station_map_all[sid]
    start_date = _parse_date(args.start_date)
    end_date = _parse_date(args.end_date)
    if end_date < start_date:
        raise ValueError("end-date must be >= start-date")

    root = Path(args.root_dir).resolve()
    raw_root = root / "raw"
    canonical_root = root / "canonical"
    reports_root = root / "reports"
    manifest_root = root / "manifests"
    run_id = _timestamp_id()

    logger.info(
        "NCEI_TRUTH_RUN_START run_id=%s stations=%s range=%s..%s root=%s",
        run_id,
        ",".join(station_map.keys()),
        start_date,
        end_date,
        root,
    )
    manifest_csv = manifest_root / "manifest.csv"
    snapshots = run_download(
        station_map=station_map,
        start_date=start_date,
        end_date=end_date,
        raw_root=raw_root,
        manifest_csv_path=manifest_csv,
        logger=logger,
        skip_if_snapshot_exists=(not bool(args.no_skip_existing)),
    )

    rows = normalize_snapshots_to_rows(snapshots=snapshots, logger=logger)
    canonical_csv = canonical_root / f"ncei_tmax_truth_{run_id}.csv"
    enriched_csv = canonical_root / f"ncei_tmax_truth_enriched_{run_id}.csv"
    canonical_df = write_canonical_csv(rows, canonical_csv)
    enriched_df = write_enriched_rows(rows, enriched_csv)

    summary = {
        "run_id": run_id,
        "stations": station_map,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "manifest_csv": str(manifest_csv),
        "canonical_csv": str(canonical_csv),
        "enriched_csv": str(enriched_csv),
        "snapshot_count": len(snapshots),
        "canonical_rows": int(len(canonical_df)),
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    for station_id in station_map.keys():
        station_df = enriched_df[enriched_df["station_id"].astype(str).str.upper() == station_id].copy()
        report = build_station_qa_report(
            station_id=station_id,
            station_df=station_df,
            start_date=start_date,
            end_date=end_date,
        )
        qa_json = reports_root / f"qa_{station_id}_{start_date.isoformat()}_{end_date.isoformat()}_{run_id}.json"
        qa_md = reports_root / f"qa_{station_id}_{start_date.isoformat()}_{end_date.isoformat()}_{run_id}.md"
        qa_json.parent.mkdir(parents=True, exist_ok=True)
        qa_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        write_station_qa_markdown(report, qa_md)
        summary[f"qa_{station_id}"] = {
            "json": str(qa_json),
            "md": str(qa_md),
            "rows": int(report.get("rows", 0)),
            "duplicate_station_date_rows": int(report.get("duplicate_station_date_rows", 0)),
            "missing_dates_count": int(len(report.get("missing_dates", []))),
        }

    if args.write_klga_training_truth_path:
        out_path = Path(args.write_klga_training_truth_path).resolve()
        klga_df = write_klga_training_truth(canonical_df=canonical_df, station_id="KLGA", out_csv_path=out_path)
        summary["klga_training_truth_csv"] = str(out_path)
        summary["klga_training_truth_rows"] = int(len(klga_df))

    if args.write_simple_settlement_csv_path:
        simple_station = str(args.simple_settlement_station_id).strip().upper()
        if not simple_station:
            if len(station_map) == 1:
                simple_station = next(iter(station_map.keys()))
            else:
                raise ValueError(
                    "--simple-settlement-station-id is required when --stations includes multiple entries."
                )
        out_path = Path(args.write_simple_settlement_csv_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        simple_df = canonical_df[canonical_df["station_id"].astype(str).str.upper() == simple_station].copy()
        simple_df = simple_df[["station_id", "target_date_local", "tmax_f"]].rename(
            columns={
                "target_date_local": "date",
                "tmax_f": "settled_tmax",
            }
        )
        simple_df["date"] = pd.to_datetime(simple_df["date"], errors="coerce").dt.date.astype(str)
        simple_df["settled_tmax"] = pd.to_numeric(simple_df["settled_tmax"], errors="coerce").round().astype("Int64")
        simple_df = simple_df[simple_df["date"].notna() & simple_df["settled_tmax"].notna()].copy()
        simple_df = simple_df.sort_values(["station_id", "date"]).drop_duplicates(
            subset=["station_id", "date"], keep="last"
        )
        simple_df.to_csv(out_path, index=False, encoding="utf-8", columns=["station_id", "date", "settled_tmax"])
        logger.info(
            "NCEI_SIMPLE_SETTLEMENT_WRITTEN station=%s path=%s rows=%d",
            simple_station,
            out_path,
            len(simple_df),
        )
        summary["simple_settlement_csv"] = str(out_path)
        summary["simple_settlement_station_id"] = simple_station
        summary["simple_settlement_rows"] = int(len(simple_df))

    summary_path = reports_root / f"run_summary_{run_id}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("NCEI_TRUTH_RUN_DONE run_id=%s summary=%s", run_id, summary_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
