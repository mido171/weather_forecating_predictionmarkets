from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_hkg_t24_station_network_smooth_local_residual_models import (
    SmoothStationSpec,
    build_specs,
    past_only_smooth_predictions,
    smooth_residual_correction,
)


def minimal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-02"]),
            "forecast_source_family": ["press", "press", "rss"],
            "target_tmax_c": [10.0, 100.0, 0.0],
            "official_raw": [0.0, 0.0, 0.0],
            "anchor_0038_c": [0.0, 0.0, 0.0],
            "smooth_feature": [1.0, 1.0, 1.0],
            "meta_text_signal_state": ["sunny", "sunny", "sunny"],
        }
    )


def test_smooth_residual_correction_suppresses_do_no_harm_failure() -> None:
    correction, neighbors, _distance, local_anchor_mae, local_corrected_mae, gate = smooth_residual_correction(
        prior_features=np.array([[1.0], [2.0], [3.0]]),
        prior_residuals=np.array([1.0, -1.0, 1.0]),
        prior_age_days=np.array([1.0, 2.0, 3.0]),
        current_features=np.array([2.0]),
        k_neighbors=3,
        half_life_days=None,
        shrinkage=0.0,
        correction_clip_c=10.0,
        min_local_mae_improvement_c=10.0,
    )

    assert correction == 0.0
    assert neighbors == 3
    assert local_corrected_mae > local_anchor_mae - 10.0
    assert gate is False


def test_past_only_smooth_predictions_excludes_same_date_rows() -> None:
    predictions = past_only_smooth_predictions(
        minimal_frame(),
        SmoothStationSpec(
            rank=1,
            feature="smooth_feature",
            feature_label="f01_smooth_feature",
            extra_features=(),
            state_cols=(),
            same_source=False,
            k_neighbors=1,
            half_life_days=None,
            min_history=1,
            min_match_rows=1,
            shrinkage=0.0,
            correction_clip_c=200.0,
        ),
    )

    assert predictions.loc[2, "residual_correction_c"] == 10.0
    assert predictions.loc[2, "candidate_prediction_c"] == 10.0


def test_past_only_smooth_predictions_same_source_isolates_history() -> None:
    predictions = past_only_smooth_predictions(
        minimal_frame(),
        SmoothStationSpec(
            rank=1,
            feature="smooth_feature",
            feature_label="f01_smooth_feature",
            extra_features=(),
            state_cols=(),
            same_source=True,
            k_neighbors=1,
            half_life_days=None,
            min_history=1,
            min_match_rows=1,
            shrinkage=0.0,
            correction_clip_c=200.0,
        ),
    )

    assert predictions.loc[2, "residual_correction_c"] == 0.0
    assert predictions.loc[2, "prior_rows"] == 0


def test_past_only_smooth_predictions_state_gate_requires_matching_prior_state() -> None:
    frame = minimal_frame()
    frame.loc[0, "meta_text_signal_state"] = "rain"

    predictions = past_only_smooth_predictions(
        frame,
        SmoothStationSpec(
            rank=1,
            feature="smooth_feature",
            feature_label="f01_smooth_feature",
            extra_features=(),
            state_cols=("meta_text_signal_state",),
            same_source=False,
            k_neighbors=1,
            half_life_days=None,
            min_history=1,
            min_match_rows=1,
            shrinkage=0.0,
            correction_clip_c=200.0,
        ),
    )

    assert predictions.loc[2, "residual_correction_c"] == 0.0
    assert predictions.loc[2, "prior_rows"] == 0


def test_build_specs_deduplicates_missing_optional_states_and_extras() -> None:
    frame = pd.DataFrame({"smooth_feature": [1.0, 2.0], "meta_text_signal_state": ["sunny", "rain"]})
    feature_catalog = pd.DataFrame(
        [
            {
                "rank": 1,
                "feature": "smooth_feature",
                "feature_label": "f01_smooth_feature",
                "family": "test",
                "interaction_priority": 1.0,
            }
        ]
    )

    specs = build_specs(frame, feature_catalog)
    keys = {
        (spec.feature, spec.extra_features, spec.state_cols, spec.same_source, spec.k_neighbors, spec.half_life_days)
        for spec in specs
    }

    assert len(specs) == len(keys)
    assert {spec.state_cols for spec in specs} == {(), ("meta_text_signal_state",)}
