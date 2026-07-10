from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_hkg_t24_cluster_centroid_soft_gating import (
    ClusterCentroidModel,
    ClusterCentroidSpec,
    build_specs,
    correction_from_model,
    past_only_cluster_centroid_predictions,
    quarter_start,
)


def test_quarter_start_returns_first_day_of_calendar_quarter() -> None:
    assert quarter_start(pd.Timestamp("2020-01-31")) == pd.Timestamp("2020-01-01")
    assert quarter_start(pd.Timestamp("2020-04-01")) == pd.Timestamp("2020-04-01")
    assert quarter_start(pd.Timestamp("2020-12-31")) == pd.Timestamp("2020-10-01")


def test_past_only_cluster_centroid_predictions_excludes_current_quarter_label() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-04-02"]),
            "forecast_source_family": ["press", "press"],
            "primary_regime": ["default", "default"],
            "target_tmax_c": [12.0, 100.0],
            "official_raw": [10.0, 10.0],
            "prediction_0018_c": [10.0, 10.0],
            "feature": [1.0, 1.0],
        }
    )
    spec = ClusterCentroidSpec(
        anchor_col="prediction_0018_c",
        mode="failure_neighbor",
        same_source=False,
        failure_quantile=0.5,
        n_clusters=1,
        k_neighbors=1,
        min_history=1,
        min_failure_rows=1,
        shrinkage=0.0,
        correction_clip_c=100.0,
        gate_distance_quantile=1.0,
    )

    predictions, _models = past_only_cluster_centroid_predictions(frame, spec, ("feature",))

    assert bool(predictions.loc[1, "do_no_harm_gate_passed"]) is True
    assert predictions.loc[1, "candidate_prediction_c"] == 12.0


def test_past_only_cluster_centroid_predictions_same_source_isolates_history() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-04-02"]),
            "forecast_source_family": ["press", "rss", "rss"],
            "primary_regime": ["default", "default", "default"],
            "target_tmax_c": [10.0, 30.0, 10.0],
            "official_raw": [10.0, 10.0, 10.0],
            "prediction_0018_c": [10.0, 10.0, 10.0],
            "feature": [1.0, 1.0, 1.0],
        }
    )
    spec = ClusterCentroidSpec(
        anchor_col="prediction_0018_c",
        mode="failure_neighbor",
        same_source=True,
        failure_quantile=0.5,
        n_clusters=1,
        k_neighbors=1,
        min_history=1,
        min_failure_rows=1,
        shrinkage=0.0,
        correction_clip_c=100.0,
        gate_distance_quantile=1.0,
    )

    predictions, _models = past_only_cluster_centroid_predictions(frame, spec, ("feature",))

    assert bool(predictions.loc[2, "do_no_harm_gate_passed"]) is True
    assert predictions.loc[2, "candidate_prediction_c"] == 30.0


def test_correction_from_model_gate_blocks_distant_current_row() -> None:
    model = ClusterCentroidModel(
        features=("feature",),
        means=np.array([0.0]),
        stds=np.array([1.0]),
        centroids=np.array([[0.0]]),
        cluster_residual_means=np.array([5.0]),
        cluster_rows=np.array([2]),
        failure_scaled=np.array([[0.0], [0.1]]),
        failure_residuals=np.array([5.0, 5.0]),
        failure_labels=np.array([0, 0]),
        failure_dates=np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"),
        gate_distance=1.0,
        distance_scale=1.0,
        failure_threshold_c=1.0,
        prior_rows=2,
        failure_rows=2,
    )
    spec = ClusterCentroidSpec(
        anchor_col="prediction_0018_c",
        mode="centroid_mean",
        same_source=False,
        failure_quantile=0.5,
        n_clusters=1,
        k_neighbors=1,
    )

    result = correction_from_model(model, np.array([10.0]), spec=spec)

    assert result.gate_passed is False
    assert result.correction == 0.0


def test_build_specs_includes_expected_modes_and_anchor_columns() -> None:
    specs = build_specs()

    assert {spec.mode for spec in specs} == {"centroid_mean", "failure_neighbor"}
    assert {spec.anchor_col for spec in specs} == {"prediction_0018_c", "prediction_0026_c"}
    assert len(specs) == 32
