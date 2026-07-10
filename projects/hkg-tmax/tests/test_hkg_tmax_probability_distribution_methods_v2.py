from __future__ import annotations

import numpy as np
import pandas as pd

from hkg_tmax_probability.bucket_rules import bucket_boundaries_for_cdf, bucket_index, bucket_key
from hkg_tmax_probability.distribution_methods_v2 import (
    distribution_v2_predictor_columns,
    emos_predict,
    predict_distribution_methods_v2,
    quantile_cdf_gb_predict,
    threshold_cdf_gb_predict,
    two_piece_normal_bucket_probs,
    two_piece_normal_cdf,
)
from hkg_tmax_probability.leaderboard_v2 import apply_v2_champion_gates
from hkg_tmax_probability.leakage_audit import audit_modeling_table
from hkg_tmax_probability.models import predict_base_methods
from hkg_tmax_probability.validation import SplitWindow, train_validation_frames


def _config() -> dict:
    return {
        "models": {
            "B2_month_residual_pmf": {"shrink_alpha": 3.0},
            "B3_forecast_level_residual_pmf": {"shrink_alpha": 3.0},
            "B4_hierarchical_residual_pmf": {"alpha_grid": {"month_alpha": [2.0], "cell_alpha": [2.0]}},
            "B5_kernel_analog": {
                "nearest_neighbors": 25,
                "weight_floor": 1.0e-8,
                "default_bandwidths": {
                    "forecast_max_c": 1.0,
                    "forecast_range_c": 1.5,
                    "forecast_max_revision_c": 1.0,
                    "month_circular": 2.0,
                },
            },
            "mos": {"ridge_alphas": [0.1], "student_t_df": [5]},
            "C1_multinomial_ridge": {"c_values": [0.1]},
            "C2_ordinal_cdf_logistic": {"C": 0.1},
            "E1_normal_emos": {
                "mean_ridge_alphas": [0.1],
                "scale_ridge_alphas": [0.1],
                "sigma_floor_grid": [0.35],
                "sigma_cap": 4.0,
            },
            "E2_student_t_emos": {
                "mean_ridge_alphas": [0.1],
                "scale_ridge_alphas": [0.1],
                "sigma_floor_grid": [0.35],
                "sigma_cap": 4.0,
                "student_t_df": [3, 5],
            },
            "E3_two_piece_normal_emos": {
                "mean_ridge_alphas": [0.1],
                "scale_ridge_alphas": [0.1],
                "sigma_floor_grid": [0.35],
                "sigma_cap": 4.0,
            },
            "G1_gamlss_tree_location_scale": {
                "n_estimators": 5,
                "max_depth": 1,
                "learning_rate": 0.05,
                "sigma_floor": 0.35,
                "sigma_cap": 4.0,
                "random_state": 11,
            },
            "Q1_quantile_cdf_gb": {
                "quantiles": [0.1, 0.5, 0.9],
                "n_estimators": 5,
                "max_depth": 1,
                "learning_rate": 0.05,
                "random_state": 11,
            },
            "Q2_threshold_cdf_gb": {
                "n_estimators": 5,
                "max_depth": 1,
                "learning_rate": 0.05,
                "random_state": 11,
            },
            "T1_time_decay_b4": {
                "half_life_years": [6.0],
                "month_alpha": [2.0],
                "cell_alpha": [2.0],
            },
            "H1_b4_challenger_linear_pool": {
                "pool_candidates": ["E1_normal_emos", "T1_time_decay_b4"],
                "weight_grid": [0.0, 0.1],
            },
        },
        "stacking": {"l2_to_b4": 0.02},
    }


def _synthetic_frame(rows: int = 220, start: str = "2001-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=rows, freq="D")
    i = np.arange(rows, dtype=float)
    forecast = 29.4 + 2.2 * np.sin(i / 23.0) + 0.7 * np.cos(i / 11.0)
    residual = 0.35 * np.sin(i / 7.0) + 0.20 * np.cos(i / 5.0)
    target = np.round(forecast + residual, 1)
    forecast_1dp = np.round(forecast, 1)
    forecast_min = forecast_1dp - 4.0 - 0.3 * np.sin(i / 13.0)
    frame = pd.DataFrame(
        {
            "target_date": dates.date.astype(str),
            "cutoff_profile": "t_minus_1_2359_hkt",
            "cutoff_hkt": "23:59",
            "cutoff_at_utc": pd.to_datetime(dates) - pd.Timedelta(minutes=1),
            "issue_at_utc": pd.to_datetime(dates) - pd.Timedelta(hours=4),
            "is_primary_cutoff": True,
            "split_label": "synthetic",
            "bucket_key": [bucket_key(value) for value in target],
            "bucket_index": [bucket_index(value) for value in target],
            "target_tmax_c": target,
            "forecast_max_c": forecast_1dp,
            "forecast_min_c": forecast_min,
            "forecast_range_c": forecast_1dp - forecast_min,
            "forecast_midpoint_c": (forecast_1dp + forecast_min) / 2.0,
            "forecast_max_tenths": np.rint(forecast_1dp * 10).astype(int),
            "residual_c": target - forecast_1dp,
            "residual_tenths": np.rint((target - forecast_1dp) * 10).astype(int),
            "official_max_round": np.rint(forecast_1dp).astype(int),
            "official_max_bin": pd.cut(forecast_1dp, bins=[-np.inf, 28, 30, 32, np.inf], labels=["low", "mid", "high", "very_high"]),
            "issue_hour_hkt": 20 + (np.arange(rows) % 4),
            "revision_count": 2 + (np.arange(rows) % 3),
            "forecast_max_revision_c": 0.1 * np.sin(i / 3.0),
            "forecast_max_path_width_c": 0.4 + 0.1 * np.cos(i / 4.0),
            "forecast_max_std_path": 0.15 + 0.03 * np.sin(i / 6.0),
            "target_month": pd.to_datetime(dates).month,
            "target_dayofyear": pd.to_datetime(dates).dayofyear,
            "season": "synthetic",
            "revision_direction": "flat",
            "row_identity": [f"row-{idx:04d}" for idx in range(rows)],
            "target_table": "label_core",
        }
    )
    return frame


def test_decimal_bucket_boundaries_for_v2_contract() -> None:
    assert bucket_key("31.9") == "31"
    assert bucket_key("32.0") == "32"
    assert bucket_key("24.9") == "24_or_below"
    assert bucket_key("34.0") == "34_or_higher"


def test_normal_emos_probability_rows_sum_and_scale_is_bounded() -> None:
    frame = _synthetic_frame()
    train, validation = frame.iloc[:180].copy(), frame.iloc[180:210].copy()
    fit = emos_predict(train, validation, _config(), "E1_normal_emos", family="normal")
    assert fit.probabilities.shape == (30, 11)
    assert np.allclose(fit.probabilities.sum(axis=1), 1.0)
    assert (fit.params["scale_c"] >= 0.35).all()
    assert (fit.params["scale_c"] <= 4.0).all()


def test_student_t_emos_df_selection_is_deterministic() -> None:
    frame = _synthetic_frame()
    train, validation = frame.iloc[:180].copy(), frame.iloc[180:210].copy()
    first = emos_predict(train, validation, _config(), "E2_student_t_emos", family="student_t")
    second = emos_predict(train, validation, _config(), "E2_student_t_emos", family="student_t")
    assert first.details["student_t_df"] == second.details["student_t_df"]
    assert np.allclose(first.probabilities, second.probabilities)


def test_two_piece_normal_cdf_is_valid_and_monotone() -> None:
    location = np.array([31.2, 32.1])
    left = np.array([0.6, 0.8])
    right = np.array([1.0, 0.7])
    cdf = np.vstack([two_piece_normal_cdf(np.full(2, edge), location, left, right) for edge in bucket_boundaries_for_cdf()]).T
    assert np.all((cdf >= 0.0) & (cdf <= 1.0))
    assert np.all(np.diff(cdf, axis=1) >= -1.0e-12)
    probs = two_piece_normal_bucket_probs(location, left, right)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all(probs >= 0.0)


def test_quantile_and_threshold_cdf_outputs_are_monotone_after_projection() -> None:
    frame = _synthetic_frame()
    train, validation = frame.iloc[:180].copy(), frame.iloc[180:205].copy()
    for fit in [
        quantile_cdf_gb_predict(train, validation, _config()),
        threshold_cdf_gb_predict(train, validation, _config()),
    ]:
        assert np.allclose(fit.probabilities.sum(axis=1), 1.0)
        cdf = np.cumsum(fit.probabilities, axis=1)
        assert np.all(np.diff(cdf, axis=1) >= -1.0e-12)


def test_v2_predictor_surface_has_no_target_or_raw_audit_leakage() -> None:
    predictors = distribution_v2_predictor_columns()
    forbidden = ["target_tmax", "bucket_key", "bucket_index", "raw_audit", "first_publication", "sealed"]
    assert not [column for column in predictors if any(fragment in column.lower() for fragment in forbidden)]
    audit = audit_modeling_table(_synthetic_frame(30), predictor_columns=predictors)
    assert audit["status"] == "pass"


def test_fold_presealed_sealed_governance_prevents_tuning_leakage() -> None:
    frame = _synthetic_frame(80, start="2023-11-15")
    frame.loc[pd.to_datetime(frame["target_date"]) >= pd.Timestamp("2024-01-01"), "target_table"] = "sealed_confirmation"
    window = SplitWindow("sealed_2024_2026_05", "2023-12-31", "2024-01-01", "2024-01-31", sealed=True)
    train, validation = train_validation_frames(frame, window, primary_only=True)
    assert not train.empty
    assert not validation.empty
    assert (train["target_table"] != "sealed_confirmation").all()
    assert (validation["target_table"] == "sealed_confirmation").all()


def test_v1_b4_probabilities_remain_unchanged_when_v2_runs() -> None:
    frame = _synthetic_frame()
    train, validation = frame.iloc[:180].copy(), frame.iloc[180:205].copy()
    config = _config()
    base_before = predict_base_methods(train, validation, config)["B4_hierarchical_residual_pmf"].probabilities
    base_outputs = predict_base_methods(train, validation, config)
    v2_outputs, _, _ = predict_distribution_methods_v2(train, validation, config, base_outputs=base_outputs)
    base_after = base_outputs["B4_hierarchical_residual_pmf"].probabilities
    assert "B4_hierarchical_residual_pmf" not in v2_outputs
    assert np.allclose(base_before, base_after)


def test_scoreboard_champion_logic_keeps_b4_when_gains_are_below_thresholds() -> None:
    overall = pd.DataFrame(
        [
            {"method": "B4_hierarchical_residual_pmf", "family": "residual_pmf", "rps": 0.0400, "nll": 1.000, "brier": 0.040},
            {"method": "E1_normal_emos", "family": "emos", "rps": 0.0398, "nll": 1.001, "brier": 0.040},
        ]
    )
    fold14 = pd.DataFrame(
        [
            {"method": "B4_hierarchical_residual_pmf", "rps": 0.0400},
            {"method": "E1_normal_emos", "rps": 0.0397},
        ]
    )
    presealed = pd.DataFrame(
        [
            {"method": "B4_hierarchical_residual_pmf", "rps": 0.0400},
            {"method": "E1_normal_emos", "rps": 0.0399},
        ]
    )
    scored = apply_v2_champion_gates(
        overall,
        fold14,
        presealed,
        {
            "complex_vs_b4_fold14_min_rps_gain": 0.015,
            "complex_vs_b4_presealed_min_rps_gain": 0.010,
            "nll_worse_than_b4_max": 0.005,
            "brier_worse_than_b4_max": 0.002,
        },
    )
    champion = scored[scored["champion_flag"]]["method"].iloc[0]
    assert champion == "B4_hierarchical_residual_pmf"
    assert scored.loc[scored["method"] == "E1_normal_emos", "gates"].iloc[0].startswith("fail:")
