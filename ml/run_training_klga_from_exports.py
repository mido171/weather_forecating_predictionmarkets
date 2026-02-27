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

REQUIRED_IMPORTS_BASE: dict[str, str] = {
    "numpy": "numpy",
    "pandas": "pandas",
    "joblib": "joblib",
    "sklearn": "scikit-learn",
    "pyarrow": "pyarrow",
}

REQUIRED_IMPORTS_LGBM: dict[str, str] = {
    "lightgbm": "lightgbm",
    "scipy": "scipy",
}

REQUIRED_IMPORTS_TABM: dict[str, str] = {
    "torch": "torch",
    "tabm": "tabm",
    "rtdl_num_embeddings": "rtdl_num_embeddings",
}


def _parse_date(v: str) -> date:
    return date.fromisoformat(v)


def _install_package(pkg: str, logger: logging.Logger) -> None:
    logger.info("DEP_INSTALL_START package=%s", pkg)
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
    logger.info("DEP_INSTALL_DONE package=%s", pkg)


def ensure_dependencies(*, backend: str, logger: logging.Logger) -> None:
    req = dict(REQUIRED_IMPORTS_BASE)
    if backend == "lgbm":
        req.update(REQUIRED_IMPORTS_LGBM)
    elif backend == "tabm":
        req.update(REQUIRED_IMPORTS_TABM)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    logger.info("DEP_CHECK_START backend=%s total=%d", backend, len(req))
    for import_name, package_name in req.items():
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
    p = argparse.ArgumentParser(
        description=(
            "Train KLGA peak/delta models from exported CSV folder only (no DB), "
            "with selectable backend (lgbm or tabm)."
        )
    )
    p.add_argument(
        "--model-backend",
        choices=["lgbm", "tabm"],
        default="lgbm",
        help="Training backend to run from exports.",
    )
    p.add_argument(
        "--data-dir",
        required=True,
        help=(
            "Absolute path to export folder containing "
            "daily_max_truth_klga.csv, observations_30m_required_columns.csv, station_universe.csv"
        ),
    )
    p.add_argument(
        "--output-root",
        default=None,
        help=(
            "Output root for run artifacts. Default: "
            "<data-dir>/training_results_<backend>."
        ),
    )
    p.add_argument("--train-start", default="1992-01-01", help="Train split start date.")
    p.add_argument("--train-end", default="2021-12-31", help="Train split end date.")
    p.add_argument("--val-start", default="2022-01-01", help="Validation split start date.")
    p.add_argument("--val-end", default="2023-12-31", help="Validation split end date.")
    p.add_argument("--test-start", default="2024-01-01", help="Test split start date.")
    p.add_argument("--test-end", default="2025-12-31", help="Test split end date.")

    p.add_argument("--log-level", default="INFO")
    p.add_argument("--log-every-rows", type=int, default=2000)
    p.add_argument("--log-every-seconds", type=float, default=20.0)

    # LGBM knobs.
    p.add_argument("--include-feels-like", action="store_true")
    p.add_argument(
        "--delta-objective",
        choices=["multiclass", "multiclassova"],
        default="multiclass",
    )
    p.add_argument("--disable-delta-class-weights", action="store_true")
    p.add_argument("--enable-delta-cutoff-weights", action="store_true")
    p.add_argument("--delta-cutoff-weight-alpha", type=float, default=1.0)
    p.add_argument("--peak-train-log-period", type=int, default=50)
    p.add_argument("--delta-train-log-period", type=int, default=25)
    p.add_argument("--train-log-every-seconds", type=float, default=10.0)
    p.add_argument("--train-heartbeat-seconds", type=float, default=10.0)

    # TabM knobs.
    p.add_argument("--max-epochs-peak", type=int, default=8)
    p.add_argument("--max-epochs-delta", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--device", default="auto", help="auto|cpu|cuda")
    p.add_argument("--tabm-arch-type", default="tabm", choices=["tabm", "tabm-mini", "tabm-packed"])
    p.add_argument("--tabm-k", type=int, default=32)
    p.add_argument("--tabm-n-blocks", type=int, default=3)
    p.add_argument("--tabm-d-block", type=int, default=256)
    p.add_argument("--tabm-dropout", type=float, default=0.2)
    p.add_argument("--tabm-start-scaling-init", default="random-signs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every-batches", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("training_klga_exports_runner")
    ensure_dependencies(backend=args.model_backend, logger=logger)

    from weather_ml.klga_daily_tmax_dist.config import SplitConfig
    from weather_ml.training.tabm_klga_from_exports import (
        LGBMExportsTrainingConfig,
        TabMTrainingConfig,
        run_lgbm_training_from_exports,
        run_tabm_training_from_exports,
    )

    data_dir = Path(args.data_dir).resolve()
    if args.output_root:
        output_root = Path(args.output_root).resolve()
    else:
        output_root = data_dir / f"training_results_{args.model_backend}"

    split = SplitConfig(
        train_start=_parse_date(args.train_start),
        train_end=_parse_date(args.train_end),
        val_start=_parse_date(args.val_start),
        val_end=_parse_date(args.val_end),
        test_start=_parse_date(args.test_start),
        test_end=_parse_date(args.test_end),
    )

    logger.info("RUN_START backend=%s data_dir=%s output_root=%s", args.model_backend, data_dir, output_root)
    if args.model_backend == "lgbm":
        cfg = LGBMExportsTrainingConfig(
            data_dir=data_dir,
            output_root=output_root,
            split=split,
            include_feels_like=bool(args.include_feels_like),
            delta_objective=str(args.delta_objective),
            delta_use_class_weights=(not bool(args.disable_delta_class_weights)),
            delta_use_cutoff_weights=bool(args.enable_delta_cutoff_weights),
            delta_cutoff_weight_alpha=float(args.delta_cutoff_weight_alpha),
            log_every_rows=int(args.log_every_rows),
            log_every_seconds=float(args.log_every_seconds),
            peak_train_log_period=int(args.peak_train_log_period),
            delta_train_log_period=int(args.delta_train_log_period),
            train_log_every_seconds=float(args.train_log_every_seconds),
            train_heartbeat_seconds=float(args.train_heartbeat_seconds),
        )
        result = run_lgbm_training_from_exports(cfg=cfg, logger=logger)
        summary = {
            "backend": "lgbm",
            "run_dir": str(result.run_dir),
            "metrics_path": str(result.metrics_path),
            "delta_val_multi_logloss_temp": result.metrics.get("delta", {}).get("val", {}).get("multi_logloss_temp"),
            "delta_test_multi_logloss_temp": result.metrics.get("delta", {}).get("test", {}).get("multi_logloss_temp"),
            "combined_val_nll": result.metrics.get("combined_lgbm", {}).get("val", {}).get("nll"),
            "combined_test_nll": result.metrics.get("combined_lgbm", {}).get("test", {}).get("nll"),
        }
    else:
        scaling_init = None if str(args.tabm_start_scaling_init).lower() == "none" else args.tabm_start_scaling_init
        cfg = TabMTrainingConfig(
            data_dir=data_dir,
            output_root=output_root,
            split=split,
            max_epochs_peak=int(args.max_epochs_peak),
            max_epochs_delta=int(args.max_epochs_delta),
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            patience=int(args.patience),
            device=str(args.device),
            tabm_arch_type=str(args.tabm_arch_type),
            tabm_k=int(args.tabm_k),
            tabm_n_blocks=int(args.tabm_n_blocks),
            tabm_d_block=int(args.tabm_d_block),
            tabm_dropout=float(args.tabm_dropout),
            tabm_start_scaling_init=scaling_init,
            seed=int(args.seed),
            log_every_batches=int(args.log_every_batches),
            log_every_rows=int(args.log_every_rows),
            log_every_seconds=float(args.log_every_seconds),
        )
        result = run_tabm_training_from_exports(cfg=cfg, logger=logger)
        summary = {
            "backend": "tabm",
            "run_dir": str(result.run_dir),
            "metrics_path": str(result.metrics_path),
            "delta_val_multi_logloss_temp": result.metrics.get("delta", {}).get("val", {}).get("multi_logloss_temp"),
            "delta_test_multi_logloss_temp": result.metrics.get("delta", {}).get("test", {}).get("multi_logloss_temp"),
            "combined_val_nll": result.metrics.get("combined", {}).get("val", {}).get("nll"),
            "combined_test_nll": result.metrics.get("combined", {}).get("test", {}).get("nll"),
        }

    logger.info("RUN_DONE backend=%s run_dir=%s", args.model_backend, summary.get("run_dir"))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

