from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_stack_trust_meta_features import (
    bucket_binary,
    bucket_numeric,
    family_prior_estimates,
    past_only_meta_trust_predictions,
)


def test_bucket_numeric_uses_fixed_thresholds_and_missing_bucket() -> None:
    buckets = bucket_numeric(pd.Series([-2.0, -0.5, 0.0, 2.5, None]), (-1.0, 0.0, 2.0))

    assert buckets.to_list() == ["<= -1", "(-1, 0]", "(-1, 0]", "> 2", "missing"]


def test_bucket_binary_maps_half_or_more_to_yes() -> None:
    buckets = bucket_binary(pd.Series([0.0, 0.49, 0.5, 1.0, None]))

    assert buckets.to_list() == ["no", "no", "yes", "yes", "missing"]


def test_family_prior_estimates_excludes_current_label() -> None:
    target = pd.Series([10.0, 100.0]).to_numpy()
    family_values = {
        "official_raw": pd.Series([10.0, 10.0]).to_numpy(),
        "family_0033_smooth": pd.Series([20.0, 100.0]).to_numpy(),
    }
    feature_arrays = {"meta_case": pd.Series(["a", "a"]).to_numpy()}
    estimates = family_prior_estimates(
        family_values=family_values,
        target=target,
        base_prior=pd.Series([True, False]).to_numpy(),
        feature_arrays=feature_arrays,
        feature_names=("meta_case",),
        row_index=1,
        min_bucket_history=1,
        min_global_history=1,
    )

    assert estimates["official_raw"][1] == 0.0
    assert estimates["family_0033_smooth"][1] == 10.0


def test_past_only_meta_trust_predictions_excludes_current_label() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "forecast_source_family": ["press", "press"],
            "target_tmax_c": [10.0, 100.0],
            "official_raw": [10.0, 10.0],
            "family_0033_smooth": [20.0, 100.0],
            "family_0034_centroid": [10.0, 10.0],
            "family_0035_revision": [20.0, 100.0],
            "meta_case": ["a", "a"],
        }
    )

    predictions = past_only_meta_trust_predictions(
        frame,
        feature_names=("meta_case",),
        mode="best",
        same_source=False,
        min_bucket_history=1,
        min_global_history=1,
    )

    assert predictions.loc[1, "selected_family"] == "family_0034_centroid"
    assert predictions.loc[1, "expert_prediction_c"] == 10.0


def test_past_only_meta_trust_predictions_excludes_same_date_other_source() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-02"]),
            "forecast_source_family": ["press", "press", "rss"],
            "target_tmax_c": [10.0, 100.0, 100.0],
            "official_raw": [10.0, 10.0, 10.0],
            "family_0033_smooth": [20.0, 100.0, 100.0],
            "family_0034_centroid": [10.0, 10.0, 10.0],
            "family_0035_revision": [20.0, 100.0, 100.0],
            "meta_case": ["a", "a", "a"],
        }
    )

    predictions = past_only_meta_trust_predictions(
        frame,
        feature_names=("meta_case",),
        mode="best",
        same_source=False,
        min_bucket_history=1,
        min_global_history=1,
    )

    assert predictions.loc[2, "selected_family"] == "family_0034_centroid"
    assert predictions.loc[2, "expert_prediction_c"] == 10.0


def test_past_only_meta_trust_predictions_same_source_isolates_history() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"]),
            "forecast_source_family": ["press", "rss", "rss"],
            "target_tmax_c": [10.0, 10.0, 10.0],
            "official_raw": [10.0, 10.0, 10.0],
            "family_0033_smooth": [20.0, 10.0, 10.0],
            "family_0034_centroid": [10.0, 20.0, 20.0],
            "family_0035_revision": [20.0, 20.0, 20.0],
            "meta_case": ["a", "a", "a"],
        }
    )

    predictions = past_only_meta_trust_predictions(
        frame,
        feature_names=("meta_case",),
        mode="best",
        same_source=True,
        min_bucket_history=1,
        min_global_history=1,
    )

    assert predictions.loc[2, "selected_family"] == "family_0033_smooth"
    assert predictions.loc[2, "expert_prediction_c"] == 10.0
