from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from weather_ml import time_feature_sweep


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run time feature sweep with multiple tree models and aggregate results."
    )
    parser.add_argument("--config", required=True, help="Base YAML config path.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=["xgb", "catboost", "random_forest"],
        help="Tree model names to run (default: xgb catboost random_forest).",
    )
    parser.add_argument(
        "--experiment-ids",
        nargs="*",
        help="Optional list of experiment ids to run. Defaults to latest sweep ids.",
    )
    parser.add_argument(
        "--source-sweep",
        help="Optional sweep root to copy experiment ids from.",
    )
    parser.add_argument(
        "--output-root",
        help="Optional output root for tree sweeps.",
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


def _load_latest_sweep(root: Path) -> Path:
    sweep_paths = sorted(
        root.glob("*/time_feature_sweep.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not sweep_paths:
        raise FileNotFoundError(f"No sweep JSON found under {root}")
    return sweep_paths[0]


def _load_experiment_ids(source_path: Path) -> list[str]:
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    baseline_id = str(payload.get("baseline_experiment_id", "BASE"))
    ids = []
    for entry in payload.get("experiments", []):
        exp_id = str(entry.get("experiment_id"))
        if exp_id and exp_id != baseline_id:
            ids.append(exp_id)
    if not ids:
        raise ValueError(f"No experiments found in {source_path}")
    return ids


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
        source = Path(args.source_sweep) if args.source_sweep else artifacts_root / "time_feature_sweep"
        if source.is_dir():
            direct = source / "time_feature_sweep.json"
            source_sweep_path = direct if direct.exists() else _load_latest_sweep(source)
        else:
            source_sweep_path = source
        experiment_ids = _load_experiment_ids(source_sweep_path)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = (
        Path(args.output_root)
        if args.output_root
        else artifacts_root / "time_feature_sweep_trees" / timestamp
    )
    output_root.mkdir(parents=True, exist_ok=True)

    aggregated = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "experiment_ids": experiment_ids,
        "models": args.models,
        "runs": [],
    }

    for model_name in args.models:
        model_root = output_root / model_name
        model_root.mkdir(parents=True, exist_ok=True)

        model_config = dict(raw_config)
        model_config.setdefault("models", {}).setdefault("mean", {})["primary"] = model_name

        config_out = model_root / f"config_{model_name}.yaml"
        _write_config(config_out, model_config)

        argv = [
            "--config",
            str(config_out),
            "--sweep-id",
            f"{timestamp}_{model_name}",
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
                "sweep_root": str(model_root),
                "sweep_path": str(sweep_path),
                "leaderboard_test_mae": sweep_payload.get("leaderboard_test_mae", []),
                "leaderboard_val_mae": sweep_payload.get("leaderboard_val_mae", []),
                "experiments": sweep_payload.get("experiments", []),
            }
        )

    aggregate_path = output_root / "tree_sweep_results.json"
    aggregate_path.write_text(
        json.dumps(aggregated, indent=2, sort_keys=False),
        encoding="utf-8",
    )
    print(f"Aggregated results written to: {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
