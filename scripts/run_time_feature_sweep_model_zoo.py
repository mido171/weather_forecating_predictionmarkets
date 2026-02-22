from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from weather_ml import time_feature_sweep


MODEL_CONFIGS = [
    {
        "name": "lgbm",
        "label": "lgbm_31_20_lr05_300",
        "params": {
            "num_leaves": [31],
            "min_data_in_leaf": [20],
            "feature_fraction": [0.8],
            "bagging_fraction": [0.8],
            "lambda_l1": [0.0],
            "lambda_l2": [0.0],
            "learning_rate": [0.05],
            "n_estimators": [300],
        },
    },
    {
        "name": "lgbm",
        "label": "lgbm_63_20_lr05_300",
        "params": {
            "num_leaves": [63],
            "min_data_in_leaf": [20],
            "feature_fraction": [0.8],
            "bagging_fraction": [0.8],
            "lambda_l1": [0.0],
            "lambda_l2": [0.0],
            "learning_rate": [0.05],
            "n_estimators": [300],
        },
    },
    {
        "name": "lgbm",
        "label": "lgbm_63_50_lr05_500",
        "params": {
            "num_leaves": [63],
            "min_data_in_leaf": [50],
            "feature_fraction": [0.9],
            "bagging_fraction": [0.9],
            "lambda_l1": [0.1],
            "lambda_l2": [0.1],
            "learning_rate": [0.05],
            "n_estimators": [500],
        },
    },
    {
        "name": "lgbm",
        "label": "lgbm_127_50_lr10_300",
        "params": {
            "num_leaves": [127],
            "min_data_in_leaf": [50],
            "feature_fraction": [0.8],
            "bagging_fraction": [0.8],
            "lambda_l1": [0.0],
            "lambda_l2": [0.1],
            "learning_rate": [0.1],
            "n_estimators": [300],
        },
    },
    {
        "name": "lgbm",
        "label": "lgbm_31_100_lr05_500",
        "params": {
            "num_leaves": [31],
            "min_data_in_leaf": [100],
            "feature_fraction": [0.9],
            "bagging_fraction": [0.9],
            "lambda_l1": [0.1],
            "lambda_l2": [0.1],
            "learning_rate": [0.05],
            "n_estimators": [500],
        },
    },
    {
        "name": "lgbm",
        "label": "lgbm_63_100_lr10_500",
        "params": {
            "num_leaves": [63],
            "min_data_in_leaf": [100],
            "feature_fraction": [0.8],
            "bagging_fraction": [0.9],
            "lambda_l1": [0.0],
            "lambda_l2": [0.1],
            "learning_rate": [0.1],
            "n_estimators": [500],
        },
    },
    {
        "name": "xgb",
        "label": "xgb_d4_lr05_300",
        "params": {
            "max_depth": [4],
            "learning_rate": [0.05],
            "n_estimators": [300],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "min_child_weight": [1],
            "reg_lambda": [1.0],
        },
    },
    {
        "name": "xgb",
        "label": "xgb_d6_lr05_300",
        "params": {
            "max_depth": [6],
            "learning_rate": [0.05],
            "n_estimators": [300],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "min_child_weight": [1],
            "reg_lambda": [1.0],
        },
    },
    {
        "name": "xgb",
        "label": "xgb_d6_lr10_300",
        "params": {
            "max_depth": [6],
            "learning_rate": [0.1],
            "n_estimators": [300],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "min_child_weight": [5],
            "reg_lambda": [1.0],
        },
    },
    {
        "name": "xgb",
        "label": "xgb_d8_lr05_500",
        "params": {
            "max_depth": [8],
            "learning_rate": [0.05],
            "n_estimators": [500],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "min_child_weight": [1],
            "reg_lambda": [1.0],
        },
    },
    {
        "name": "xgb",
        "label": "xgb_d4_lr10_500",
        "params": {
            "max_depth": [4],
            "learning_rate": [0.1],
            "n_estimators": [500],
            "subsample": [1.0],
            "colsample_bytree": [0.8],
            "min_child_weight": [1],
            "reg_lambda": [3.0],
        },
    },
    {
        "name": "xgb",
        "label": "xgb_d6_lr10_500",
        "params": {
            "max_depth": [6],
            "learning_rate": [0.1],
            "n_estimators": [500],
            "subsample": [0.9],
            "colsample_bytree": [1.0],
            "min_child_weight": [5],
            "reg_lambda": [3.0],
        },
    },
    {
        "name": "catboost",
        "label": "cat_d6_lr05_500",
        "params": {
            "depth": [6],
            "learning_rate": [0.05],
            "iterations": [500],
            "l2_leaf_reg": [3],
        },
    },
    {
        "name": "catboost",
        "label": "cat_d8_lr05_800",
        "params": {
            "depth": [8],
            "learning_rate": [0.05],
            "iterations": [800],
            "l2_leaf_reg": [3],
        },
    },
    {
        "name": "catboost",
        "label": "cat_d8_lr10_500",
        "params": {
            "depth": [8],
            "learning_rate": [0.1],
            "iterations": [500],
            "l2_leaf_reg": [10],
        },
    },
    {
        "name": "catboost",
        "label": "cat_d10_lr05_800",
        "params": {
            "depth": [10],
            "learning_rate": [0.05],
            "iterations": [800],
            "l2_leaf_reg": [10],
        },
    },
    {
        "name": "gbr",
        "label": "gbr_d3_lr05_200",
        "params": {
            "max_depth": [3],
            "learning_rate": [0.05],
            "n_estimators": [200],
        },
    },
    {
        "name": "gbr",
        "label": "gbr_d3_lr05_400",
        "params": {
            "max_depth": [3],
            "learning_rate": [0.05],
            "n_estimators": [400],
        },
    },
    {
        "name": "gbr",
        "label": "gbr_d4_lr10_200",
        "params": {
            "max_depth": [4],
            "learning_rate": [0.1],
            "n_estimators": [200],
        },
    },
    {
        "name": "gbr",
        "label": "gbr_d4_lr10_400",
        "params": {
            "max_depth": [4],
            "learning_rate": [0.1],
            "n_estimators": [400],
        },
    },
    {
        "name": "random_forest",
        "label": "rf_200_d8_leaf1_sqrt",
        "params": {
            "n_estimators": [200],
            "max_depth": [8],
            "min_samples_leaf": [1],
            "max_features": ["sqrt"],
        },
    },
    {
        "name": "random_forest",
        "label": "rf_500_d16_leaf1_08",
        "params": {
            "n_estimators": [500],
            "max_depth": [16],
            "min_samples_leaf": [1],
            "max_features": [0.8],
        },
    },
    {
        "name": "random_forest",
        "label": "rf_500_dNone_leaf1_sqrt",
        "params": {
            "n_estimators": [500],
            "max_depth": [None],
            "min_samples_leaf": [1],
            "max_features": ["sqrt"],
        },
    },
    {
        "name": "random_forest",
        "label": "rf_300_d16_leaf3_08",
        "params": {
            "n_estimators": [300],
            "max_depth": [16],
            "min_samples_leaf": [3],
            "max_features": [0.8],
        },
    },
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run time feature sweep across a model zoo."
    )
    parser.add_argument("--config", required=True, help="Base YAML config path.")
    parser.add_argument(
        "--experiment-ids",
        nargs="*",
        help="Optional list of experiment ids to run.",
    )
    parser.add_argument(
        "--output-root",
        help="Optional output root for sweep results.",
    )
    parser.add_argument(
        "--truth-lag",
        type=int,
        default=2,
        help="Truth lag in days (default: 2).",
    )
    parser.add_argument(
        "--allow-tuning",
        action="store_true",
        help="Enable hyperparameter tuning.",
    )
    return parser


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_raw_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_config(path: Path, payload: dict) -> None:
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def main() -> int:
    args = build_parser().parse_args()
    repo_root = _resolve_repo_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    raw_config = _load_raw_config(config_path)

    artifacts_root = Path(raw_config.get("artifacts", {}).get("root_dir", "artifacts"))
    if not artifacts_root.is_absolute():
        artifacts_root = repo_root / artifacts_root

    if args.experiment_ids:
        experiment_ids = list(args.experiment_ids)
    else:
        experiment_ids = [f"EX{idx:02d}" for idx in range(61, 111)]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = (
        Path(args.output_root)
        if args.output_root
        else artifacts_root / "time_feature_sweep_model_zoo" / timestamp
    )
    output_root.mkdir(parents=True, exist_ok=True)

    aggregated = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "experiment_ids": experiment_ids,
        "model_configs": MODEL_CONFIGS,
        "runs": [],
    }

    for model_cfg in MODEL_CONFIGS:
        model_name = model_cfg["name"]
        label = model_cfg["label"]
        params = model_cfg["params"]

        model_root = output_root / label
        model_root.mkdir(parents=True, exist_ok=True)

        model_config = copy.deepcopy(raw_config)
        model_config.setdefault("models", {}).setdefault("mean", {})["primary"] = model_name
        model_config["models"]["mean"]["candidates"] = [model_name]
        model_config["models"]["mean"]["param_grid"] = {model_name: params}

        config_out = model_root / f"config_{label}.yaml"
        _write_config(config_out, model_config)

        argv = [
            "--config",
            str(config_out),
            "--sweep-id",
            f"{timestamp}_{label}",
            "--sweep-root",
            str(model_root),
            "--truth-lag",
            str(args.truth_lag),
            "--experiment-ids",
            *experiment_ids,
        ]
        if args.allow_tuning:
            argv.append("--allow-tuning")

        time_feature_sweep.main(argv)

        sweep_path = model_root / "time_feature_sweep.json"
        sweep_payload = json.loads(sweep_path.read_text(encoding="utf-8"))
        aggregated["runs"].append(
            {
                "model": model_name,
                "label": label,
                "params": params,
                "sweep_root": str(model_root),
                "sweep_path": str(sweep_path),
                "leaderboard_test_mae": sweep_payload.get("leaderboard_test_mae", []),
                "leaderboard_val_mae": sweep_payload.get("leaderboard_val_mae", []),
                "experiments": sweep_payload.get("experiments", []),
            }
        )

    aggregate_path = output_root / "model_zoo_results.json"
    aggregate_path.write_text(
        json.dumps(aggregated, indent=2, sort_keys=False),
        encoding="utf-8",
    )
    print(f"Aggregated results written to: {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
