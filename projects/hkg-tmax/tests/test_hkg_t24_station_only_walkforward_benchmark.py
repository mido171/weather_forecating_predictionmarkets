from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_hkg_t24_station_only_walkforward_benchmark import (
    BenchmarkSpec,
    fold_definitions,
    leakage_audit,
    select_feature_columns,
    validate_fold_definitions,
)


def test_fold_definitions_are_chronological_and_long_enough() -> None:
    folds = fold_definitions()

    validate_fold_definitions(folds)

    assert [fold[0] for fold in folds] == [
        "fold_2000_2005",
        "fold_2006_2011",
        "fold_2012_2017",
        "fold_2018_2023",
    ]
    assert folds[0][3] == pd.Timestamp("1999-12-31")
    assert folds[-1][2] == pd.Timestamp("2023-12-31")


def test_validate_fold_definitions_rejects_training_overlap() -> None:
    folds = [("bad", pd.Timestamp("2000-01-01"), pd.Timestamp("2005-12-31"), pd.Timestamp("2000-01-01"))]

    with pytest.raises(ValueError, match="leaks training into test window"):
        validate_fold_definitions(folds)


def test_select_feature_columns_uses_train_only_correlation_ranking() -> None:
    train = pd.DataFrame(
        {
            "target_anomaly_vs_past_doy_c": [0.0, 1.0, 2.0, 3.0] * 1000,
            "feature_train_signal": [0.0, 1.0, 2.0, 3.0] * 1000,
            "feature_weak": [1.0, 0.0, 1.0, 0.0] * 1000,
            "doy_sin": [0.1] * 4000,
            "doy_cos": [0.2] * 4000,
            "available_feature_fraction": [1.0] * 4000,
        }
    )
    catalog = pd.DataFrame(
        [
            {
                "feature_id": "feature_train_signal",
                "source_family": "station_attribute",
                "raw_feature_name": "feature_train_signal",
            },
            {
                "feature_id": "feature_weak",
                "source_family": "station_attribute",
                "raw_feature_name": "feature_weak",
            },
        ]
    )

    columns, selected = select_feature_columns(
        train,
        catalog,
        BenchmarkSpec("test", "ridge_fold_selected", "top_train_corr", top_k=1, include_calendar=False),
    )

    assert columns == ["feature_train_signal"]
    assert selected["feature"].tolist() == ["feature_train_signal"]


def test_select_feature_columns_rejects_forbidden_model_columns() -> None:
    train = pd.DataFrame(
        {
            "target_anomaly_vs_past_doy_c": [0.0, 1.0, 2.0, 3.0] * 1000,
            "official_error_proxy": [0.0, 1.0, 2.0, 3.0] * 1000,
        }
    )
    catalog = pd.DataFrame(
        [
            {
                "feature_id": "official_error_proxy",
                "source_family": "station_attribute",
                "raw_feature_name": "official_error_proxy",
            }
        ]
    )

    with pytest.raises(ValueError, match="Forbidden model feature columns"):
        select_feature_columns(
            train,
            catalog,
            BenchmarkSpec("test", "ridge_station_only", "all_station", include_calendar=False),
        )


def test_leakage_audit_requires_pre_confirmation_predictions() -> None:
    predictions = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2023-12-31", "2024-01-01"]),
            "fold_id": ["a", "a"],
        }
    )
    leakage_rows = pd.DataFrame(
        {
            "train_end_before_test_start": [True],
            "selected_features_fit_inside_fold": [True],
            "scaler_imputer_fit_inside_fold": [True],
        }
    )

    audit = leakage_audit(predictions, leakage_rows)

    assert not bool(audit[audit["check_id"].eq("no_confirmation_predictions")].iloc[0]["passed"])
