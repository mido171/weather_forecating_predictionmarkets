"""Sweep runner for TFS2 experiments (DB-backed)."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
import threading

import numpy as np
from weather_ml import experiment_results_db

from . import data
from . import db as tfs2_db
from . import experiments
from .config import DEFAULT_OUTPUT_ROOT, DEFAULT_SPLIT

LOGGER = logging.getLogger(__name__)

SLOW_EXPERIMENT_IDS = {
    "CatBoost-MAEWithBustMOS",
    "XGB-QuantileTrioMeanReconstruction",
    "CorrectedForecastLibrary-StackRidge",
    "LocalLinearAnalogCalibration-LLR",
}
DEFAULT_HEARTBEAT_SECONDS = 60


def _default_sweep_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _setup_logging(log_file: Path | None, *, verbose: bool) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
    )


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (date, datetime)):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _rank_experiments(results: list[dict], key: str) -> list[dict]:
    items = []
    for exp in results:
        metrics = exp.get("metrics", {})
        value = metrics.get("test", {}).get(key)
        items.append((value, exp.get("experiment_id"), exp))
    items = [i for i in items if i[0] is not None]
    return [
        {"experiment_id": exp_id, key: value}
        for value, exp_id, _ in sorted(items, key=lambda x: x[0])
    ]


def _run_with_heartbeat(spec: experiments.ExperimentSpec, ctx: experiments.ExperimentContext, heartbeat_seconds: int) -> dict:
    stop = threading.Event()
    start = time.time()

    def _heartbeat() -> None:
        while not stop.wait(heartbeat_seconds):
            elapsed = time.time() - start
            LOGGER.warning("EXPERIMENT_HEARTBEAT %s elapsed=%.1fs", spec.experiment_id, elapsed)

    thread = threading.Thread(target=_heartbeat, daemon=True)
    thread.start()
    try:
        return spec.runner(ctx)
    finally:
        stop.set()


def run_sweep(
    *,
    station_id: str,
    start_date: date,
    end_date: date,
    sweep_id: str | None,
    output_root: Path | None,
    db_url: str | None,
    persist_db: bool,
    skip_experiment_ids: set[str] | None = None,
    heartbeat_seconds: int = DEFAULT_HEARTBEAT_SECONDS,
) -> Path:
    sweep_id = sweep_id or _default_sweep_id()
    sweep_root = (output_root or DEFAULT_OUTPUT_ROOT) / sweep_id
    sweep_root.mkdir(parents=True, exist_ok=True)

    LOGGER.info("SWEEP_START sweep_id=%s station=%s", sweep_id, station_id)
    LOGGER.info("SWEEP_ROOT %s", sweep_root)
    if skip_experiment_ids:
        LOGGER.info("SWEEP_SKIP_EXPERIMENTS %s", sorted(skip_experiment_ids))

    engine = data.create_engine_from_url(db_url)
    bundle = data.build_dataset(engine, station_id, start_date, end_date)
    LOGGER.info("DATASET_ROWS %d", len(bundle.df))
    LOGGER.info("DATASET_RANGE %s to %s", start_date, end_date)

    ctx = experiments.build_context(bundle.df, run_root=sweep_root)
    LOGGER.info(
        "BASELINE_A val_mae=%.4f test_mae=%.4f",
        ctx.baseline_a.metrics["validation"]["mae"],
        ctx.baseline_a.metrics["test"]["mae"],
    )
    LOGGER.info(
        "BASELINE_B val_mae=%.4f test_mae=%.4f",
        ctx.baseline_b.metrics["validation"]["mae"],
        ctx.baseline_b.metrics["test"]["mae"],
    )

    experiment_specs = experiments.build_experiments()
    results = []
    start_all = time.time()
    for idx, spec in enumerate(experiment_specs, start=1):
        if skip_experiment_ids and spec.experiment_id in skip_experiment_ids:
            LOGGER.warning("EXPERIMENT_SKIPPED %s", spec.experiment_id)
            results.append(
                {
                    "experiment_id": spec.experiment_id,
                    "name": spec.name,
                    "description": spec.description,
                    "skipped": True,
                    "skip_reason": "marked_slow",
                }
            )
            continue
        LOGGER.info("EXPERIMENT_START %s (%d/%d)", spec.experiment_id, idx, len(experiment_specs))
        t0 = time.time()
        try:
            result = _run_with_heartbeat(spec, ctx, heartbeat_seconds)
            results.append(result)
            LOGGER.info(
                "EXPERIMENT_END %s duration=%.1fs val_mae=%.4f test_mae=%.4f",
                spec.experiment_id,
                time.time() - t0,
                result.get("metrics", {}).get("validation", {}).get("mae", float("nan")),
                result.get("metrics", {}).get("test", {}).get("mae", float("nan")),
            )
        except Exception as exc:
            LOGGER.exception("EXPERIMENT_FAILED %s: %s", spec.experiment_id, exc)
            results.append(
                {
                    "experiment_id": spec.experiment_id,
                    "name": spec.name,
                    "description": spec.description,
                    "error": str(exc),
                }
            )

    duration = time.time() - start_all
    LOGGER.info("SWEEP_FINISH duration=%.1fs", duration)

    results_path = sweep_root / "tfs2_results.ndjson"
    with results_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "sweep_id": sweep_id,
        "sweep_kind": "time_feature_sweep_v2",
        "station": station_id,
        "created_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "dataset_ref": asdict(bundle.dataset_ref),
        "split_ref": ctx.split_ref,
        "baseline_a": ctx.baseline_a.metrics,
        "baseline_b": ctx.baseline_b.metrics,
        "experiments": results,
        "rankings": {
            "by_test_mae": _rank_experiments(results, "mae"),
        },
    }
    summary = _sanitize_for_json(summary)
    summary_path = sweep_root / "time_feature_sweep_v2.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8"
    )
    LOGGER.info("SWEEP_SUMMARY %s", summary_path)

    if persist_db:
        try:
            experiment_results_db.ensure_database(experiment_results_db.default_db_name())
            exp_engine = experiment_results_db.create_db_engine(
                experiment_results_db.default_mysql_url()
            )
            experiment_results_db.persist_sweep(
                exp_engine,
                summary_path,
                sweep_kind="time_feature_sweep_v2",
                station_id=station_id,
            )
            LOGGER.info("DB_PERSIST_OK experiment_results_db")
        except Exception:
            LOGGER.exception("DB_PERSIST_FAILED experiment_results_db")
        try:
            db_engine = tfs2_db.create_db_engine(db_url)
            payload = tfs2_db.load_sweep_summary(summary_path)
            tfs2_db.upsert_model_experiments(db_engine, payload, sweep_id=sweep_id)
            LOGGER.info("DB_PERSIST_OK model_experiment")
        except Exception:
            LOGGER.exception("DB_PERSIST_FAILED model_experiment")

    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TFS2 experiment sweep (DB-backed).")
    parser.add_argument("--station", default="KMIA")
    parser.add_argument("--start-date", default=str(DEFAULT_SPLIT.train_start))
    parser.add_argument("--end-date", default=str(DEFAULT_SPLIT.test_end))
    parser.add_argument("--sweep-id", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--db-url", default=None)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--skip-db", action="store_true")
    parser.add_argument("--include-slow", action="store_true")
    parser.add_argument("--heartbeat-seconds", type=int, default=DEFAULT_HEARTBEAT_SECONDS)
    parser.add_argument("--no-verbose", dest="verbose", action="store_false")
    parser.set_defaults(verbose=True)
    args = parser.parse_args()

    sweep_id = args.sweep_id or _default_sweep_id()
    output_root = Path(args.output_root) if args.output_root else DEFAULT_OUTPUT_ROOT
    log_file = Path(args.log_file) if args.log_file else (output_root / sweep_id / "tfs2_sweep.log")
    _setup_logging(log_file, verbose=args.verbose)
    skip_experiment_ids = set()
    if not args.include_slow:
        skip_experiment_ids |= SLOW_EXPERIMENT_IDS
    summary_path = run_sweep(
        station_id=args.station,
        start_date=date.fromisoformat(args.start_date),
        end_date=date.fromisoformat(args.end_date),
        sweep_id=sweep_id,
        output_root=output_root,
        db_url=args.db_url,
        persist_db=not args.skip_db,
        skip_experiment_ids=skip_experiment_ids,
        heartbeat_seconds=args.heartbeat_seconds,
    )
    LOGGER.info("SUMMARY_PATH %s", summary_path)


if __name__ == "__main__":
    main()
