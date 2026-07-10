from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_r10_latent_spatial_modes import (
    fold_local_pca_scores,
    long_report,
    station_offset_columns,
)


def test_station_offset_columns_are_sorted_and_specific() -> None:
    features = pd.DataFrame(
        {
            "station_offset_hko_minus_sha_tin": [1.0],
            "temp_network_spread_c": [2.0],
            "station_offset_hko_minus_chek_lap_kok": [0.5],
        }
    )

    assert station_offset_columns(features) == [
        "station_offset_hko_minus_chek_lap_kok",
        "station_offset_hko_minus_sha_tin",
    ]


def test_fold_local_pca_scores_fit_training_window_only() -> None:
    train = pd.DataFrame(
        {
            "station_offset_hko_minus_a": [0.0, 1.0, 2.0, 3.0],
            "station_offset_hko_minus_b": [0.0, 2.0, 4.0, 6.0],
            "station_offset_hko_minus_constant": [7.0, 7.0, 7.0, 7.0],
        }
    )
    test = pd.DataFrame(
        {
            "station_offset_hko_minus_a": [4.0, 5.0],
            "station_offset_hko_minus_b": [8.0, 10.0],
            "station_offset_hko_minus_constant": [7.0, 7.0],
        }
    )

    train_scores, test_scores, loadings, variance = fold_local_pca_scores(
        train,
        test,
        [
            "station_offset_hko_minus_a",
            "station_offset_hko_minus_b",
            "station_offset_hko_minus_constant",
        ],
        2,
        fold_id="fold_1",
        model_id="r10_test",
    )

    assert list(train_scores.columns) == ["pc1", "pc2", "pca_reconstruction_error"]
    assert list(test_scores.columns) == ["pc1", "pc2", "pca_reconstruction_error"]
    assert len(train_scores) == 4
    assert len(test_scores) == 2
    assert {row["feature"] for row in loadings} == {
        "station_offset_hko_minus_a",
        "station_offset_hko_minus_b",
    }
    assert "pc1_explained_variance_ratio" in variance


def test_r10_long_report_exceeds_required_experiment_narrative_length() -> None:
    report = long_report(
        {
            "champion": {
                "model_id": "r10_baseline_temp_calendar",
                "n": 911,
                "mae": 1.4723,
                "rmse": 1.8861,
                "bias": 0.0298,
                "crps_normal": 1.0512,
            },
            "oof_feasibility": {
                "status": "BLOCKED",
                "reason": "synthetic four-year OOF blocker",
            },
            "feature_min": "2020-07-02",
            "feature_max": "2023-12-31",
            "prediction_min": "2021-07-01",
            "prediction_max": "2023-12-31",
        }
    )

    assert len(report) >= 7500
    assert "fold-local" in report
    assert "station-coordinate" in report
