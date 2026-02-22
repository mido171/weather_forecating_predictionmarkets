"""Tests for RS-MoE mean model components."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from weather_ml import rs_moe
from weather_ml import time_feature_sweep


def test_temperature_scaling_softmax_sums_to_one_and_is_stable() -> None:
    logits = np.array(
        [
            [1000.0, 1001.0, 999.0],
            [-1000.0, -1001.0, -999.0],
        ],
        dtype=float,
    )
    probs = rs_moe.softmax_temperature(logits, temperature=2.0)
    assert probs.shape == (2, 3)
    assert np.all(np.isfinite(probs))
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-12)


def test_mixture_identity_exact() -> None:
    p = np.array([[0.2, 0.3, 0.5], [0.0, 1.0, 0.0]], dtype=float)
    mu_cool = np.array([10.0, 5.0], dtype=float)
    mu_normal = np.array([20.0, 6.0], dtype=float)
    mu_warm = np.array([30.0, 7.0], dtype=float)
    mu_hat = p[:, 0] * mu_cool + p[:, 1] * mu_normal + p[:, 2] * mu_warm
    rs_moe.assert_mixture_identity(
        p=p,
        mu_cool=mu_cool,
        mu_normal=mu_normal,
        mu_warm=mu_warm,
        mu_hat=mu_hat,
        atol=1e-12,
    )


def test_oof_fold_assignment_has_no_leakage_and_marks_burnin() -> None:
    n = 20
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    train_df = pd.DataFrame(
        {
            "station_id": ["KMIA"] * n,
            "target_date_local": dates,
            "asof_utc": pd.to_datetime(dates, utc=True),
            "actual_tmax_f": np.linspace(70.0, 75.0, n),
        }
    )
    X = np.random.default_rng(0).normal(size=(n, 4))
    y_regime = np.array([0, 1, 2, 1] * 5, dtype=int)[:n]

    class DummyGate:
        def fit(self, X_fit, y_fit) -> None:  # noqa: ANN001
            return None

        def predict(self, X_pred, prediction_type=None):  # noqa: ANN001
            _ = prediction_type
            return np.zeros((len(X_pred), 3), dtype=float)

    builder = rs_moe.OOFGateBuilder(
        rs_moe.OofGatingConfig(
            enabled=True,
            method="expanding_time_blocks",
            n_folds=4,
            burnin_fraction=0.2,
            min_rows_per_fold=1,
            weight_floor=0.02,
            random_seed=12345,
        )
    )
    result = builder.build_oof_logits(
        train_df=train_df,
        X_train=X,
        y_regime=y_regime,
        build_gate_model=lambda: DummyGate(),
    )

    burnin_size = result.burnin_size
    assert burnin_size == int(np.floor(n * 0.2))
    assert result.oof_is_model_based.shape == (n,)
    assert not result.oof_is_model_based[result.order[:burnin_size]].any()
    assert result.oof_is_model_based[result.order[burnin_size:]].all()

    burnin = result.order[:burnin_size]
    prev = burnin
    for block in result.fold_blocks:
        train_idx = np.concatenate([prev]) if len(prev) else np.array([], dtype=int)
        assert len(np.intersect1d(train_idx, block)) == 0
        prev = np.concatenate([prev, block])


def test_rs_moe_integration_smoke(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    n_total = 60
    dates = pd.date_range("2021-01-01", periods=n_total, freq="D")
    station_id = ["KMIA"] * n_total
    asof_utc = pd.to_datetime(dates, utc=True)

    base = 75.0 + np.linspace(0.0, 1.0, n_total)
    forecasts = {
        "gfs_tmax_f": base + rng.normal(0, 0.5, n_total),
        "nam_tmax_f": base + rng.normal(0, 0.5, n_total),
        "gefsatmosmean_tmax_f": base + rng.normal(0, 0.5, n_total),
        "rap_tmax_f": base + rng.normal(0, 0.5, n_total),
        "hrrr_tmax_f": base + rng.normal(0, 0.5, n_total),
        "nbm_tmax_f": base + rng.normal(0, 0.5, n_total),
    }
    ens_mean = np.mean(np.column_stack(list(forecasts.values())), axis=1)

    # Force all three regimes to appear, including in burn-in.
    resid = np.zeros(n_total, dtype=float)
    resid[0::10] = -10.0
    resid[1::10] = 10.0
    y_true = ens_mean + resid

    df = pd.DataFrame(
        {
            "station_id": station_id,
            "target_date_local": dates,
            "asof_utc": asof_utc,
            "gefsatmos_tmp_spread_f": np.full(n_total, 5.0),
            "actual_tmax_f": y_true,
            **forecasts,
        }
    )

    csv_path = tmp_path / "toy.csv"
    df.to_csv(csv_path, index=False)

    sweep_root = tmp_path / "sweep"
    sweep_root.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "data": {"csv_path": str(csv_path), "dataset_schema_version": 1},
        "validation": {"strict_schema": True},
        "split": {
            "train_start": "2021-01-01",
            "train_end": "2021-02-20",
            "test_start": "2021-02-21",
            "test_end": "2021-03-01",
            "gap_dates": [],
            "validation": {"enabled": True, "val_start": "2021-02-10", "val_end": "2021-02-14"},
            "cv": {"enabled": False, "n_splits": 2, "gap_days": 0},
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
            "calendar": True,
        },
        "models": {"mean": {"primary": "ridge", "param_grid": {}}, "sigma": {"method": "two_stage"}},
        "artifacts": {"root_dir": str(tmp_path / "artifacts"), "overwrite": True},
        "seeds": {"global_seed": 123},
        "distribution": {"support_min_f": -30, "support_max_f": 130},
        "calibration": {"enabled": False, "bins_to_calibrate": []},
        "rs_moe": {
            "oof_gating": {
                "enabled": True,
                "method": "expanding_time_blocks",
                "n_folds": 2,
                "burnin_fraction": 0.5,
                "min_rows_per_fold": 1,
                "weight_floor": 0.02,
                "random_seed": 12345,
            },
            "gate_model": {
                "library": "catboost",
                "params": {
                    "loss_function": "MultiClass",
                    "iterations": 30,
                    "depth": 2,
                    "learning_rate": 0.1,
                    "l2_leaf_reg": 3.0,
                    "random_seed": 12345,
                    "allow_writing_files": False,
                    "verbose": False,
                },
            },
            "gate_calibration": {
                "method": "temperature_scaling",
                "temperature_init": 1.0,
                "temperature_bounds": [0.5, 10.0],
                "optimizer": "lbfgs",
                "max_iter": 50,
                "tol": 1e-7,
            },
            "experts": {
                "library": "xgboost",
                "objective_variant": "absoluteerror",
                "absoluteerror_params": {
                    "objective": "reg:absoluteerror",
                    "n_estimators": 30,
                    "learning_rate": 0.1,
                    "max_depth": 2,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "min_child_weight": 1.0,
                    "reg_lambda": 1.0,
                    "reg_alpha": 0.0,
                    "tree_method": "hist",
                    "random_state": 12345,
                },
            },
        },
    }

    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    argv = [
        "--config",
        str(config_path),
        "--sweep-id",
        "pytest_rs_moe",
        "--sweep-root",
        str(sweep_root),
        "--truth-lag",
        "2",
        "--bootstrap-samples",
        "10",
        "--experiment-ids",
        "EX201",
    ]
    assert time_feature_sweep.main(argv) == 0

    run_dir = sweep_root / "EX201"
    required = [
        run_dir / "report.md",
        run_dir / "metrics.json",
        run_dir / "experiment_meta.json",
        run_dir / "config_resolved.yaml",
        run_dir / "dataset_id.txt",
        run_dir / "hashes.json",
        run_dir / "predictions_test.parquet",
        run_dir / "gate_model.cbm",
        run_dir / "expert_cool_model.joblib",
        run_dir / "expert_normal_model.joblib",
        run_dir / "expert_warm_model.joblib",
        run_dir / "gate_calibration.json",
        run_dir / "oof_gate_logits_train.parquet",
        run_dir / "oof_gate_probs_train.parquet",
    ]
    for path in required:
        assert path.exists()

    preds = pd.read_parquet(run_dir / "predictions_test.parquet")
    for col in [
        "mu_hat_f",
        "p_cool",
        "p_normal",
        "p_warm",
        "mu_cool",
        "mu_normal",
        "mu_warm",
        "gate_temperature",
        "model_type",
    ]:
        assert col in preds.columns

    mu_recomputed = (
        preds["p_cool"] * preds["mu_cool"]
        + preds["p_normal"] * preds["mu_normal"]
        + preds["p_warm"] * preds["mu_warm"]
    )
    max_abs = float(np.max(np.abs(mu_recomputed.to_numpy() - preds["mu_hat_f"].to_numpy())))
    assert max_abs <= 1e-10

    hashes = json.loads((run_dir / "hashes.json").read_text(encoding="utf-8"))
    assert str(run_dir / "gate_model.cbm") in hashes
