from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from weather_ml.klga_daily_tmax_dist.config import PipelineConfig, default_output_root
from weather_ml.klga_daily_tmax_dist.pipeline import run_training_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="KLGA leakage-paranoid same-day Tmax distribution pipeline (LGBM + analog kNN)."
    )
    p.add_argument(
        "--mysql-url",
        default=None,
        help="Optional SQLAlchemy MySQL URL. If omitted, MYSQL_* env vars are used.",
    )
    p.add_argument(
        "--output-root",
        default=str(default_output_root()),
        help="Output root directory (default: artifacts/same_day_res_poly).",
    )
    p.add_argument(
        "--force-rebuild-dataset",
        action="store_true",
        help="Rebuild feature store even if artifacts/same_day_res_poly/feature_store exists.",
    )
    p.add_argument(
        "--skip-analog-blend",
        action="store_true",
        help="Disable analog kNN and blending; train/evaluate/export LGBM peak+delta only.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    p.add_argument(
        "--log-every-rows",
        type=int,
        default=2000,
        help="Progress logging interval by row count for long loops.",
    )
    p.add_argument(
        "--log-every-seconds",
        type=float,
        default=20.0,
        help="Progress logging interval by elapsed seconds for long loops.",
    )
    p.add_argument(
        "--peak-train-log-period",
        type=int,
        default=50,
        help="LightGBM iteration log period for peak model training.",
    )
    p.add_argument(
        "--delta-train-log-period",
        type=int,
        default=25,
        help="LightGBM iteration log period for delta model training.",
    )
    p.add_argument(
        "--train-log-every-seconds",
        type=float,
        default=10.0,
        help="Time-based LightGBM progress log interval during training.",
    )
    p.add_argument(
        "--train-heartbeat-seconds",
        type=float,
        default=10.0,
        help="Heartbeat interval while LightGBM is fitting (alive/ETA logs).",
    )
    p.add_argument(
        "--delta-objective",
        choices=["multiclass", "multiclassova"],
        default=None,
        help="Override delta objective.",
    )
    p.add_argument(
        "--disable-delta-class-weights",
        action="store_true",
        help="Disable delta class-weighting.",
    )
    p.add_argument(
        "--enable-delta-cutoff-weights",
        action="store_true",
        help="Enable optional late-cutoff weighting for delta training.",
    )
    p.add_argument(
        "--delta-cutoff-weight-alpha",
        type=float,
        default=None,
        help="Alpha for optional delta cutoff weighting.",
    )
    p.add_argument(
        "--include-feels-like",
        action="store_true",
        help="Include cleaned feels_like features if available.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    cfg_kwargs = {
        "output_root": Path(args.output_root).resolve(),
        "include_feels_like": bool(args.include_feels_like),
        "delta_use_class_weights": (not bool(args.disable_delta_class_weights)),
        "delta_use_cutoff_weights": bool(args.enable_delta_cutoff_weights),
    }
    if args.delta_objective is not None:
        cfg_kwargs["delta_objective"] = str(args.delta_objective)
    if args.delta_cutoff_weight_alpha is not None:
        cfg_kwargs["delta_cutoff_weight_alpha"] = float(args.delta_cutoff_weight_alpha)
    cfg = PipelineConfig(**cfg_kwargs)
    result = run_training_pipeline(
        cfg=cfg,
        mysql_url=args.mysql_url,
        force_rebuild_dataset=bool(args.force_rebuild_dataset),
        enable_analog=(not bool(args.skip_analog_blend)),
        log_level=args.log_level,
        log_every_rows=int(args.log_every_rows),
        log_every_seconds=float(args.log_every_seconds),
        peak_train_log_period=int(args.peak_train_log_period),
        delta_train_log_period=int(args.delta_train_log_period),
        train_log_every_seconds=float(args.train_log_every_seconds),
        train_heartbeat_seconds=float(args.train_heartbeat_seconds),
    )
    summary = {
        "run_dir": str(result.run_dir),
        "metrics_path": str(result.metrics_path),
        "combined_blended_val": result.metrics.get("combined_blended", {}).get("val", {}),
        "combined_blended_test": result.metrics.get("combined_blended", {}).get("test", {}),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
