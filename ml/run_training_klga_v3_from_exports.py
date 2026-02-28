from __future__ import annotations

import argparse
import importlib
import json
import logging
import subprocess
import sys
from datetime import date
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

REQUIRED_IMPORTS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "joblib": "joblib",
    "sklearn": "scikit-learn",
    "pyarrow": "pyarrow",
    "lightgbm": "lightgbm",
    "scipy": "scipy",
    "hmmlearn": "hmmlearn",
}


def _parse_date(v: str) -> date:
    return date.fromisoformat(v)


def _install_package(pkg: str, logger: logging.Logger) -> None:
    logger.info("DEP_INSTALL_START package=%s", pkg)
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
    logger.info("DEP_INSTALL_DONE package=%s", pkg)


def ensure_dependencies(*, logger: logging.Logger) -> None:
    logger.info("DEP_CHECK_START total=%d", len(REQUIRED_IMPORTS))
    for import_name, package_name in REQUIRED_IMPORTS.items():
        try:
            importlib.import_module(import_name)
            logger.info("DEP_OK import=%s package=%s", import_name, package_name)
        except Exception:
            logger.warning("DEP_MISSING import=%s package=%s", import_name, package_name)
            _install_package(package_name, logger)
            importlib.import_module(import_name)
            logger.info("DEP_OK_AFTER_INSTALL import=%s package=%s", import_name, package_name)
    logger.info("DEP_CHECK_DONE")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train KLGA V3 (regime600) from exported CSVs (no DB).")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--train-start", default="1992-01-01")
    p.add_argument("--train-end", default="2021-12-31")
    p.add_argument("--val-start", default="2022-01-01")
    p.add_argument("--val-end", default="2023-12-31")
    p.add_argument("--test-start", default="2024-01-01")
    p.add_argument("--test-end", default="2025-12-31")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--log-every-rows", type=int, default=2000)
    p.add_argument("--log-every-seconds", type=float, default=20.0)
    p.add_argument("--peak-train-log-period", type=int, default=50)
    p.add_argument("--delta-train-log-period", type=int, default=25)
    p.add_argument("--train-log-every-seconds", type=float, default=10.0)
    p.add_argument("--train-heartbeat-seconds", type=float, default=10.0)
    p.add_argument("--feature-budget-max", type=int, default=600)
    p.add_argument("--disable-hmm-features", action="store_true")
    p.add_argument("--disable-analog-blend", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger("training_klga_v3_exports_runner")

    ensure_dependencies(logger=logger)

    from weather_ml.klga_daily_tmax_dist.config import SplitConfig
    from weather_ml.training.lgbm_klga_v3_from_exports import LGBMV3TrainingConfig, run_lgbm_v3_training_from_exports

    split = SplitConfig(
        train_start=_parse_date(args.train_start),
        train_end=_parse_date(args.train_end),
        val_start=_parse_date(args.val_start),
        val_end=_parse_date(args.val_end),
        test_start=_parse_date(args.test_start),
        test_end=_parse_date(args.test_end),
    )

    out_root = Path(args.output_root).resolve() / "same_day_res_poly_v3"
    cfg = LGBMV3TrainingConfig(
        data_dir=Path(args.data_dir).resolve(),
        output_root=out_root,
        split=split,
        log_every_rows=int(args.log_every_rows),
        log_every_seconds=float(args.log_every_seconds),
        peak_train_log_period=int(args.peak_train_log_period),
        delta_train_log_period=int(args.delta_train_log_period),
        train_log_every_seconds=float(args.train_log_every_seconds),
        train_heartbeat_seconds=float(args.train_heartbeat_seconds),
        feature_budget_max=int(args.feature_budget_max),
        enable_hmm_features=(not bool(args.disable_hmm_features)),
        enable_analog_blend=(not bool(args.disable_analog_blend)),
    )

    logger.info("RUN_START backend=lgbm_v3 data_dir=%s output_root=%s", cfg.data_dir, cfg.output_root)
    result = run_lgbm_v3_training_from_exports(cfg=cfg, logger=logger)

    summary = {
        "backend": "lgbm_v3",
        "run_dir": str(result.run_dir),
        "metrics_path": str(result.metrics_path),
        "delta_val_multi_logloss_temp": result.metrics.get("delta", {}).get("val", {}).get("multi_logloss_temp"),
        "delta_test_multi_logloss_temp": result.metrics.get("delta", {}).get("test", {}).get("multi_logloss_temp"),
        "combined_val_nll": result.metrics.get("combined_blended", {}).get("val", {}).get("nll"),
        "combined_test_nll": result.metrics.get("combined_blended", {}).get("test", {}).get("nll"),
    }
    logger.info("RUN_DONE backend=lgbm_v3 run_dir=%s", summary.get("run_dir"))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
