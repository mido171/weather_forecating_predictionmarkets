"""Integration tests for feature strategy sweep."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

from weather_ml import feature_sweep


def test_feature_sweep_runs(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    rows = []
    base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for idx in range(6):
        date_val = base_date + timedelta(days=idx)
        target_date_local = date_val.date().isoformat()
        asof_utc = (date_val - timedelta(days=1)).replace(
            hour=12, minute=0, second=0, microsecond=0
        )
        base = 40.0 + idx
        rows.append(
            {
                "station_id": "KNYC",
                "target_date_local": target_date_local,
                "asof_utc": asof_utc.isoformat().replace("+00:00", "Z"),
                "nbm_tmax_f": base,
                "gfs_tmax_f": base + 1.0,
                "gefsatmosmean_tmax_f": base + 2.0,
                "nam_tmax_f": base + 1.5,
                "hrrr_tmax_f": base + 0.5,
                "rap_tmax_f": base + 1.2,
                "gefsatmos_tmp_spread_f": 2.0,
                "actual_tmax_f": base + 1.0,
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    config = {
        "data": {"csv_path": str(csv_path), "dataset_schema_version": 1},
        "validation": {
            "strict_schema": True,
            "forecast_min_f": -80.0,
            "forecast_max_f": 140.0,
            "spread_min_f": 0.0,
            "require_asof_not_after_target": True,
        },
        "split": {
            "train_start": "2024-01-01",
            "train_end": "2024-01-03",
            "test_start": "2024-01-05",
            "test_end": "2024-01-06",
            "gap_dates": [],
            "validation": {"enabled": True, "val_start": "2024-01-04", "val_end": "2024-01-04"},
            "cv": {"enabled": False, "n_splits": 2, "gap_days": 0},
        },
        "features": {
            "base_features": [
                "nbm_tmax_f",
                "gfs_tmax_f",
                "gefsatmosmean_tmax_f",
                "nam_tmax_f",
                "hrrr_tmax_f",
                "rap_tmax_f",
                "gefsatmos_tmp_spread_f",
            ],
            "derived": {
                "ensemble_stats": False,
                "pairwise_deltas": False,
                "model_vs_ens_deltas": False,
                "calendar": False,
                "station_onehot": False,
                "climatology": {"enabled": False},
            },
        },
        "models": {
            "mean": {"candidates": ["ridge"], "primary": "ridge", "param_grid": {}},
            "sigma": {"method": "two_stage", "primary": "ridge", "param_grid": {}},
        },
        "artifacts": {"root_dir": str(tmp_path / "artifacts"), "run_id": None, "overwrite": True},
        "seeds": {"global_seed": 7, "force_single_thread": True},
        "distribution": {"support_min_f": -30, "support_max_f": 130},
        "calibration": {"enabled": False, "method": "isotonic", "bins_to_calibrate": []},
        "postprocess": {"global_normal_calibration": {"enabled": False}},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    sweep_root = tmp_path / "sweep"
    feature_sweep.main(
        [
            "--config",
            str(config_path),
            "--sweep-root",
            str(sweep_root),
            "--strategy-ids",
            "S01",
            "S02",
        ]
    )

    assert (sweep_root / "feature_strategy_sweep.json").exists()
    assert (sweep_root / "S01" / "metrics.json").exists()
    assert (sweep_root / "S02" / "metrics.json").exists()
