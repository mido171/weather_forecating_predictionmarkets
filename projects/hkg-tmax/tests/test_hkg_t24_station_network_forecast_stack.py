from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_hkg_t24_station_network_forecast_stack import (
    StackSpec,
    estimate_family_prior_mae,
    past_only_stack_predictions,
    prediction_from_estimates,
)


def minimal_stack_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-02"]),
            "forecast_source_family": ["press", "press", "rss"],
            "target_tmax_c": [10.0, 100.0, 100.0],
            "official_raw": [10.0, 0.0, 0.0],
            "anchor_0038_c": [10.0, 0.0, 0.0],
            "hard_0039_best_c": [0.0, 100.0, 100.0],
            "meta_case": ["a", "a", "a"],
        }
    )


def test_estimate_family_prior_mae_uses_prior_mask_only() -> None:
    values = np.array([10.0, 100.0])
    target = np.array([10.0, 0.0])
    feature_arrays = {"meta_case": np.array(["a", "a"])}

    count, mae = estimate_family_prior_mae(
        values=values,
        target=target,
        base_prior=np.array([True, False]),
        feature_arrays=feature_arrays,
        feature_names=("meta_case",),
        row_index=1,
        min_global_history=1,
        min_bucket_history=1,
    )

    assert count == 2
    assert mae == 0.0


def test_past_only_stack_predictions_excludes_same_date_other_source() -> None:
    predictions = past_only_stack_predictions(
        minimal_stack_frame(),
        StackSpec(
            feature_set="case",
            feature_names=("meta_case",),
            mode="best",
            same_source=False,
            family_group="core",
            family_names=("anchor_0038_c", "hard_0039_best_c"),
            min_global_history=1,
            min_bucket_history=1,
        ),
    )

    assert predictions.loc[2, "selected_family"] == "anchor_0038_c"
    assert predictions.loc[2, "candidate_prediction_c"] == 0.0


def test_past_only_stack_predictions_same_source_isolates_history() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"]),
            "forecast_source_family": ["press", "rss", "rss"],
            "target_tmax_c": [10.0, 10.0, 10.0],
            "official_raw": [10.0, 0.0, 0.0],
            "anchor_0038_c": [10.0, 0.0, 0.0],
            "hard_0039_best_c": [0.0, 10.0, 10.0],
            "meta_case": ["a", "a", "a"],
        }
    )

    predictions = past_only_stack_predictions(
        frame,
        StackSpec(
            feature_set="case",
            feature_names=("meta_case",),
            mode="best",
            same_source=True,
            family_group="core",
            family_names=("anchor_0038_c", "hard_0039_best_c"),
            min_global_history=1,
            min_bucket_history=1,
        ),
    )

    assert predictions.loc[2, "selected_family"] == "hard_0039_best_c"
    assert predictions.loc[2, "candidate_prediction_c"] == 10.0


def test_prediction_from_estimates_positive_lift_falls_back_to_anchor_without_lift() -> None:
    prediction, family, count, selected_mae, anchor_mae, weight = prediction_from_estimates(
        estimates={"anchor_0038_c": (10, 1.0), "hard_0039_best_c": (10, 1.2)},
        family_values={
            "anchor_0038_c": np.array([20.0]),
            "hard_0039_best_c": np.array([30.0]),
        },
        row_index=0,
        mode="positive_lift",
    )

    assert prediction == 20.0
    assert family == "anchor_0038_c_fallback"
    assert count == 0
    assert np.isnan(selected_mae)
    assert anchor_mae == 1.0
    assert weight == 0.0


def test_prediction_from_estimates_anchor_lift_blend_uses_prior_lift_weight() -> None:
    prediction, family, count, selected_mae, anchor_mae, weight = prediction_from_estimates(
        estimates={"anchor_0038_c": (100, 1.0), "hard_0039_best_c": (100, 0.8)},
        family_values={
            "anchor_0038_c": np.array([20.0]),
            "hard_0039_best_c": np.array([30.0]),
        },
        row_index=0,
        mode="anchor_lift_blend",
    )

    assert family == "anchor_lift_blend:hard_0039_best_c"
    assert count == 100
    assert selected_mae == 0.8
    assert anchor_mae == 1.0
    assert 0.20 < weight <= 0.85
    assert 20.0 < prediction < 30.0
