"""Sweep runner for the 30-experiment pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from weather_ml import artifacts

from . import data, experiments, models, analogs
from .config import DEFAULT_GRIB_CSV, DEFAULT_MOS_CSV, DEFAULT_OUTPUT_ROOT

LOGGER = logging.getLogger(__name__)

HEAVY_EXPERIMENTS = {
    "MOSResidualizedContextSignals",
    "BustProbabilityGatedShrinkage",
    "ForwardChainingStackedEnsemble",
    "TwoStageResidualStackWithBiasFeatures",
    "MultiSnapshotAggregationAndEnsemble",
    "GuidanceSpacekNNAnalogPrediction",
    "ResidualAnalogCorrectionOnBaselineB",
    "PrototypeAnalogsKMedoidsWithForwardUpdate",
    "LocalLinearKNNRidgeForecaster",
}


def _setup_logging(level: str, log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=handlers,
    )


def _default_sweep_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run exp30 experiment sweep.")
    parser.add_argument("--grib-csv", default=str(DEFAULT_GRIB_CSV))
    parser.add_argument("--mos-csv", default=str(DEFAULT_MOS_CSV))
    parser.add_argument("--station", default="KMIA")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--sweep-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing NDJSON results.",
    )
    parser.add_argument(
        "--skip-heavy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip known heavy/lagging experiments.",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        help="Experiment id to skip (can be repeated).",
    )
    parser.add_argument("--max-exp-mins", type=float, default=10.0)
    parser.add_argument("--max-baseline-mins", type=float, default=20.0)
    parser.add_argument("--max-search-mins", type=float, default=None)
    parser.add_argument("--max-analog-mins", type=float, default=None)
    parser.add_argument("--trial-scale", type=float, default=1.0)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument(
        "--baseline-candidates",
        default="lgbm,xgb,catboost",
        help="Comma-separated baseline model families to evaluate.",
    )
    parser.add_argument(
        "--log-every-trial",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--log-every-seed",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--log-every-rows", type=int, default=200)
    return parser


def _load_existing_results(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    existing = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            exp_id = payload.get("experiment_id")
            if exp_id:
                existing[exp_id] = payload
        except json.JSONDecodeError:
            continue
    return existing


def _append_result(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sweep_id = args.sweep_id or _default_sweep_id()
    run_root = Path(args.output_root) / sweep_id
    run_root.mkdir(parents=True, exist_ok=True)
    log_file = Path(args.log_file) if args.log_file else run_root / "exp30_sweep.log"
    _setup_logging(args.log_level, log_file)

    seeds_override = [int(s) for s in args.seeds.split(",") if s.strip()]
    baseline_candidates = [c.strip() for c in args.baseline_candidates.split(",") if c.strip()]
    if args.skip_heavy and "catboost" in baseline_candidates:
        baseline_candidates = [c for c in baseline_candidates if c != "catboost"]
        LOGGER.warning("Skipping catboost baseline candidate due to skip-heavy.")
    experiments.set_runtime_config(
        seeds_override=seeds_override,
        baseline_candidates=baseline_candidates,
    )
    baseline_search_seconds = float(args.max_baseline_mins) * 60.0
    max_search_seconds = (
        float(args.max_search_mins) * 60.0
        if args.max_search_mins is not None
        else float(args.max_exp_mins) * 60.0 * 0.8
    )
    models.configure_runtime(
        trials_scale=args.trial_scale,
        max_search_seconds=baseline_search_seconds,
        log_every_trial=args.log_every_trial,
        log_every_seed=args.log_every_seed,
    )
    analog_budget = (
        float(args.max_analog_mins) * 60.0
        if args.max_analog_mins is not None
        else float(args.max_exp_mins) * 60.0 * 0.8
    )
    analogs.configure_runtime(
        max_seconds=analog_budget,
        log_every_rows=args.log_every_rows,
    )

    grib_df = data.load_gribstream_csv(Path(args.grib_csv))
    mos_df = data.load_mos_csv(Path(args.mos_csv))
    merged = data.merge_grib_mos(grib_df, mos_df)
    filtered = data.filter_station(merged, args.station)
    if filtered.empty:
        LOGGER.warning(
            "Station filter %s produced no rows; falling back to unfiltered data.",
            args.station,
        )
        merged = merged
        station_used = None
    else:
        merged = filtered
        station_used = args.station

    split = data.split_dataset(merged)
    train_mask = merged.index.isin(split.train_df.index)
    val_mask = merged.index.isin(split.val_df.index)
    test_mask = merged.index.isin(split.test_df.index)

    rng = np.random.default_rng(args.seed)

    LOGGER.info("SWEEP_START sweep_id=%s", sweep_id)
    ctx = experiments.build_context(
        merged,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        run_root=run_root,
        split_ref=split.split_ref,
        rng=rng,
    )
    # Reset search budget for per-experiment runs.
    models.configure_runtime(
        trials_scale=args.trial_scale,
        max_search_seconds=max_search_seconds,
        log_every_trial=args.log_every_trial,
        log_every_seed=args.log_every_seed,
    )

    LOGGER.info("Baseline B model: %s", ctx.baseline_b.model_name)
    experiment_specs = experiments.build_experiments()

    results = []
    results_path = run_root / "exp30_results.ndjson"
    existing = _load_existing_results(results_path) if args.resume else {}
    if existing:
        LOGGER.info("Resuming from %d existing results.", len(existing))
    skip_ids = set(args.skip or [])
    if args.skip_heavy:
        skip_ids |= HEAVY_EXPERIMENTS
    total_specs = len(experiment_specs)
    start_sweep = time.time()
    for spec in experiment_specs:
        if spec.experiment_id in existing:
            LOGGER.info("Skipping %s (already completed).", spec.experiment_id)
            results.append(existing[spec.experiment_id])
            continue
        if spec.experiment_id in skip_ids:
            LOGGER.warning("Skipping %s (lagging/heavy).", spec.experiment_id)
            payload = {
                "experiment_id": spec.experiment_id,
                "name": spec.name,
                "description": spec.description,
                "status": "skipped",
                "reason": "heavy_or_user_skip",
            }
            results.append(payload)
            _append_result(results_path, payload)
            continue
        LOGGER.info(
            "RUN_START %s (%d/%d)",
            spec.experiment_id,
            len(results) + 1,
            total_specs,
        )
        exp_start = time.time()
        status = "ok"
        error = None
        try:
            result = spec.runner(ctx)
        except Exception as exc:
            status = "failed"
            error = str(exc)
            result = {
                "experiment_id": spec.experiment_id,
                "name": spec.name,
                "description": spec.description,
            }
        duration = time.time() - exp_start
        result["status"] = status
        result["duration_seconds"] = duration
        if error:
            result["error"] = error
        if duration > float(args.max_exp_mins) * 60.0:
            result["status"] = "over_budget"
        results.append(result)
        _append_result(results_path, result)
        LOGGER.info(
            "RUN_END %s status=%s duration=%.1fs elapsed_total=%.1fs",
            spec.experiment_id,
            result["status"],
            duration,
            time.time() - start_sweep,
        )

    leaderboard_val = sorted(
        [
            {
                "experiment_id": r["experiment_id"],
                "val_mae": r["metrics"]["validation"].get("mae"),
            }
            for r in results
        ],
        key=lambda r: (r["val_mae"] if r["val_mae"] is not None else 1e9),
    )
    leaderboard_test = sorted(
        [
            {
                "experiment_id": r["experiment_id"],
                "test_mae": r["metrics"]["test"].get("mae"),
            }
            for r in results
        ],
        key=lambda r: (r["test_mae"] if r["test_mae"] is not None else 1e9),
    )

    summary = {
        "sweep_id": sweep_id,
        "created_utc": artifacts.utc_now_iso(),
        "dataset_ref": {
            "grib_csv": str(Path(args.grib_csv).resolve()),
            "mos_csv": str(Path(args.mos_csv).resolve()),
            "grib_hash": artifacts.sha256_file(Path(args.grib_csv)),
            "mos_hash": artifacts.sha256_file(Path(args.mos_csv)),
        },
        "station": station_used or "ALL",
        "split_ref": split.split_ref,
        "runtime": {
            "seeds": seeds_override,
            "trial_scale": args.trial_scale,
            "max_exp_minutes": args.max_exp_mins,
            "max_baseline_minutes": args.max_baseline_mins,
            "max_search_minutes": max_search_seconds / 60.0,
            "max_analog_minutes": analog_budget / 60.0,
            "baseline_candidates": baseline_candidates,
            "skip_heavy": args.skip_heavy,
            "skip_ids": sorted(skip_ids),
        },
        "baseline_b": {
            "model": ctx.baseline_b.model_name,
            "params": ctx.baseline_b.params,
            "metrics": ctx.baseline_b.metrics,
        },
        "leaderboard_val_mae": leaderboard_val,
        "leaderboard_test_mae": leaderboard_test,
        "experiments": results,
    }

    summary_path = run_root / "exp30_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    global_path = run_root / "exp30_global_results.json"
    global_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    LOGGER.info("Sweep complete. Output: %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
