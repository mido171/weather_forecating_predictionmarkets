from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

from weather_ml import experiment_results_db


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Persist time feature sweeps and live calibration payloads to results DB."
    )
    parser.add_argument(
        "--db-name",
        default=experiment_results_db.default_db_name(),
        help="Database name to create/use for experiment results.",
    )
    parser.add_argument(
        "--db-url",
        help="Optional SQLAlchemy DB URL (overrides --db-name).",
    )
    parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        help="Path to time_feature_sweep.json (repeatable).",
    )
    parser.add_argument(
        "--sweep-kind",
        action="append",
        default=[],
        help="Optional sweep kind label matching --sweep order.",
    )
    parser.add_argument(
        "--station-id",
        help="Optional station id override for sweeps.",
    )
    parser.add_argument(
        "--live-calibration",
        action="append",
        default=[],
        help="Format: STATION:YYYYMMDD:SWEEP_ID:EXPERIMENT_ID (repeatable).",
    )
    parser.add_argument(
        "--calibration-json",
        action="append",
        default=[],
        help="Format: PATH:SWEEP_ID:EXPERIMENT_ID (repeatable).",
    )
    parser.add_argument(
        "--calibration-source",
        default="live_emos_w45",
        help="Calibration source label.",
    )
    parser.add_argument(
        "--live-script",
        default="tools/live/run_kmia_live.py",
        help="Path to live pipeline script.",
    )
    parser.add_argument(
        "--verbose-live",
        action="store_true",
        help="Enable verbose logging for live pipeline runs.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.db_url:
        db_url = args.db_url
    else:
        experiment_results_db.ensure_database(args.db_name)
        db_url = experiment_results_db.default_mysql_url(args.db_name)

    engine = experiment_results_db.create_db_engine(db_url)
    experiment_results_db.ensure_tables(engine)

    sweeps = [Path(path) for path in args.sweep]
    sweep_kinds = _resolve_sweep_kinds(sweeps, args.sweep_kind)
    for sweep_path, sweep_kind in zip(sweeps, sweep_kinds):
        experiment_results_db.persist_sweep(
            engine,
            sweep_path,
            sweep_kind=sweep_kind,
            station_id=args.station_id,
        )

    for entry in args.calibration_json:
        path_str, sweep_id, experiment_id = _parse_calibration_json(entry)
        payload = json.loads(Path(path_str).read_text(encoding="utf-8"))
        experiment_results_db.persist_calibration(
            engine,
            payload,
            sweep_id=sweep_id,
            experiment_id=experiment_id,
            calibration_source=args.calibration_source,
        )
        print(
            json.dumps(
                {
                    "station_id": payload.get("station_id"),
                    "asof_utc": payload.get("asof_utc"),
                    "target_date_local": payload.get("target_date_local"),
                    "mu_hat_f": payload.get("mu_hat_f"),
                    "sigma_hat_f": payload.get("sigma_hat_f"),
                    "sigma_emos_f": payload.get("sigma_emos_f"),
                    "rolling_bias_45": payload.get("rolling_bias_45"),
                    "rolling_rmse_45": payload.get("rolling_rmse_45"),
                },
                indent=2,
            )
        )

    for entry in args.live_calibration:
        station_id, target_date, sweep_id, experiment_id = _parse_live_calibration(entry)
        payload = _run_live_pipeline(
            args.live_script,
            station_id,
            target_date,
            verbose=args.verbose_live,
        )
        experiment_results_db.persist_calibration(
            engine,
            payload,
            sweep_id=sweep_id,
            experiment_id=experiment_id,
            calibration_source=args.calibration_source,
        )
        print(
            json.dumps(
                {
                    "station_id": payload.get("station_id"),
                    "asof_utc": payload.get("asof_utc"),
                    "target_date_local": payload.get("target_date_local"),
                    "mu_hat_f": payload.get("mu_hat_f"),
                    "sigma_hat_f": payload.get("sigma_hat_f"),
                    "sigma_emos_f": payload.get("sigma_emos_f"),
                    "rolling_bias_45": payload.get("rolling_bias_45"),
                    "rolling_rmse_45": payload.get("rolling_rmse_45"),
                },
                indent=2,
            )
        )

    return 0


def _resolve_sweep_kinds(sweeps: list[Path], kinds: list[str]) -> list[str]:
    resolved = []
    for idx, sweep_path in enumerate(sweeps):
        if idx < len(kinds):
            resolved.append(kinds[idx])
            continue
        if "time_feature_sweep_trees" in str(sweep_path).lower():
            resolved.append("time_feature_sweep_trees")
        else:
            resolved.append("time_feature_sweep")
    return resolved


def _parse_live_calibration(entry: str) -> tuple[str, str, str, str]:
    parts = entry.split(":")
    if len(parts) != 4:
        raise ValueError(
            "live-calibration must be STATION:YYYYMMDD:SWEEP_ID:EXPERIMENT_ID"
        )
    station_id, target_date_raw, sweep_id, experiment_id = parts
    target_date = _normalize_date(target_date_raw)
    return station_id, target_date, sweep_id, experiment_id


def _parse_calibration_json(entry: str) -> tuple[str, str, str]:
    parts = entry.split(":")
    if len(parts) != 3:
        raise ValueError("calibration-json must be PATH:SWEEP_ID:EXPERIMENT_ID")
    path_str, sweep_id, experiment_id = parts
    return path_str, sweep_id, experiment_id


def _normalize_date(value: str) -> str:
    if re.match(r"^\\d{8}$", value):
        return datetime.strptime(value, "%Y%m%d").strftime("%Y-%m-%d")
    return value


def _run_live_pipeline(
    script_path: str,
    station_id: str,
    target_date: str,
    *,
    verbose: bool,
) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    extra_paths = [str(repo_root), str(repo_root / "ml" / "src")]
    env["PYTHONPATH"] = ";".join([*extra_paths, pythonpath]) if pythonpath else ";".join(extra_paths)

    cmd = [
        "python",
        script_path,
        "--station",
        station_id,
        "--target-date",
        target_date,
    ]
    if verbose:
        cmd.append("--verbose")

    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"Live pipeline failed for {station_id}: {output}")

    json_text = _extract_first_json(output)
    return json.loads(json_text)


def _extract_first_json(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in live pipeline output.")
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_string = False
            continue
        if ch == "\"":
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    raise ValueError("Incomplete JSON object in live pipeline output.")


if __name__ == "__main__":
    raise SystemExit(main())
