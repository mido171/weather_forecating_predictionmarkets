"""Tests for global normal calibration."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import json
import yaml

from weather_ml import global_normal_calibration
from weather_ml import train


def test_compute_residual_stats_ddof() -> None:
    residuals = np.array([1.0, -1.0, 2.0, -2.0])
    stats = global_normal_calibration.compute_residual_stats(residuals, ddof=1)
    assert stats["n"] == 4
    assert stats["bias_mean_error_f"] == 0.0
    expected_sigma = float(np.std(residuals, ddof=1))
    assert np.isclose(stats["sigma_std_error_f"], expected_sigma)
    expected_mae = float(np.mean(np.abs(residuals)))
    assert np.isclose(stats["mae_f"], expected_mae)
    expected_rmse = float(np.sqrt(np.mean(residuals**2)))
    assert np.isclose(stats["rmse_f"], expected_rmse)


def test_compute_residual_stats_ddof_too_large() -> None:
    residuals = np.array([1.0])
    try:
        global_normal_calibration.compute_residual_stats(residuals, ddof=1)
    except ValueError as exc:
        assert "Not enough calibration rows" in str(exc)
    else:
        raise AssertionError("Expected ValueError for ddof=1 with n=1")


def test_integration_global_calibration(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    artifacts_root = tmp_path / "artifacts"

    rows = []
    base_date = datetime(2024, 12, 1, tzinfo=timezone.utc)
    dates = [base_date + timedelta(days=i) for i in range(10)]  # train
    dates += [base_date + timedelta(days=10)]  # test
    dates += [datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=i) for i in range(3)]  # calibration

    for idx, date_val in enumerate(dates):
        target_date_local = date_val.date().isoformat()
        asof_utc = (date_val - timedelta(days=1)).replace(
            hour=12, minute=0, second=0, microsecond=0
        )
        base = 40.0 + idx
        row = {
            "station_id": "KNYC",
            "target_date_local": target_date_local,
            "asof_utc": asof_utc.isoformat().replace("+00:00", "Z"),
            "gfs_tmax_f": base,
            "nam_tmax_f": base + 1.0,
            "gefsatmosmean_tmax_f": base + 2.0,
            "rap_tmax_f": base + 1.5,
            "hrrr_tmax_f": base + 0.5,
            "nbm_tmax_f": base + 1.2,
            "gefsatmos_tmp_spread_f": 2.0,
            "actual_tmax_f": base + 1.0,
        }
        rows.append(row)

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
            "train_start": "2024-12-01",
            "train_end": "2024-12-10",
            "test_start": "2024-12-11",
            "test_end": "2024-12-11",
            "gap_dates": [],
            "validation": {"enabled": False},
            "cv": {"enabled": True, "n_splits": 2, "gap_days": 0},
        },
        "features": {
            "base_features": [
                "gfs_tmax_f",
                "nam_tmax_f",
                "gefsatmosmean_tmax_f",
                "rap_tmax_f",
                "hrrr_tmax_f",
                "nbm_tmax_f",
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
        "artifacts": {"root_dir": str(artifacts_root), "run_id": None, "overwrite": True},
        "seeds": {"global_seed": 7, "force_single_thread": True},
        "distribution": {"support_min_f": -30, "support_max_f": 130},
        "calibration": {"enabled": False, "method": "isotonic", "bins_to_calibrate": []},
        "postprocess": {
            "global_normal_calibration": {
                "enabled": True,
                "cal_start": "2025-01-01",
                "cal_end": "2025-01-03",
                "ddof": 1,
                "station_scope": "ALL",
            },
            "baseline_median_calibration": {
                "enabled": True,
                "cal_start": "2024-12-01",
                "cal_end": "2025-01-03",
                "ddof": 1,
                "station_scope": "ALL",
            },
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    run_id = "integration_run"
    train.main(["--config", str(config_path), "--run-id", run_id])

    run_dir = artifacts_root / "runs" / run_id
    calibration_json = run_dir / "global_normal_calibration.json"
    residuals_csv = run_dir / "global_normal_residuals.csv"
    baseline_json = run_dir / "baseline_median_normal_calibration.json"
    baseline_csv = run_dir / "baseline_median_normal_residuals.csv"

    assert calibration_json.exists()
    assert residuals_csv.exists()
    assert baseline_json.exists()
    assert baseline_csv.exists()

    payload = json.loads(calibration_json.read_text(encoding="utf-8"))
    required_keys = {
        "method",
        "error_definition",
        "ddof",
        "n",
        "bias_mean_error_f",
        "sigma_std_error_f",
        "mae_f",
        "rmse_f",
        "residual_quantiles_f",
        "calibration_window",
        "station_scope",
        "model_ref",
        "dataset_ref",
        "created_utc",
    }
    assert required_keys.issubset(payload.keys())
    assert payload["n"] == 3
    assert payload["calibration_window"]["start"] == "2025-01-01"

    baseline_payload = json.loads(baseline_json.read_text(encoding="utf-8"))
    baseline_required = {
        "method",
        "error_definition",
        "ddof",
        "n",
        "bias_mean_error_f",
        "sigma_std_error_f",
        "mae_f",
        "rmse_f",
        "residual_quantiles_f",
        "calibration_window",
        "station_scope",
        "forecast_columns",
        "dataset_ref",
        "created_utc",
    }
    assert baseline_required.issubset(baseline_payload.keys())
