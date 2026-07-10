from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_hkg_t24_multi_signal_local_residual_lab import (
    FeatureSet,
    LocalResidualSpec,
    available_features,
    build_specs,
    past_only_local_predictions,
)
from scripts.run_hkg_t24_official_residual_source_text_range_dynamics import (
    add_source_phase_features,
)


def test_available_features_omits_missing_low_coverage_and_constant_columns() -> None:
    frame = pd.DataFrame(
        {
            "good": [float(value) for value in range(5)],
            "constant": [1.0] * 5,
            "sparse": [1.0, None, None, None, None],
        }
    )

    out = available_features(frame, ("good", "constant", "sparse", "missing"), min_non_null=3)

    assert out == ("good",)


def test_build_specs_creates_prior_only_source_and_phase_variants() -> None:
    feature_sets = {"demo": FeatureSet("demo", {"a": ("f1", "f2"), "b": ("f3",)})}

    specs = build_specs(feature_sets)

    assert len(specs) == 4
    assert {spec.same_source for spec in specs} == {False, True}
    assert {spec.phase_conditioned for spec in specs} == {False, True}
    assert all(spec.features == ("f1", "f2", "f3") for spec in specs)


def test_past_only_local_predictions_excludes_current_target_date_label() -> None:
    frame = add_source_phase_features(
        pd.DataFrame(
            {
                "target_date": pd.date_range("2020-01-01", periods=5, freq="D"),
                "forecast_source_family": ["rss"] * 5,
                "target_tmax_c": [10.0, 10.0, 100.0, 10.0, 10.0],
                "forecast_max_c": [10.0] * 5,
                "feature_a": [0.0, 1.0, 100.0, 1.0, 1.0],
                "feature_b": [0.0, 1.0, 100.0, 1.0, 1.0],
            }
        )
    )
    spec = LocalResidualSpec(
        feature_set="leak_guard",
        features=("feature_a", "feature_b"),
        k_neighbors=1,
        same_source=False,
        phase_conditioned=False,
        shrinkage=0.0,
        min_history=2,
    )

    out = past_only_local_predictions(frame, spec)

    assert out.loc[2, "past_rows_used"] == 1
    assert out.loc[2, "residual_correction_c"] == pytest.approx(0.0)
    assert out.loc[2, "candidate_prediction_c"] == pytest.approx(10.0)


def test_past_only_local_predictions_same_source_and_phase_isolate_history() -> None:
    frame = add_source_phase_features(
        pd.DataFrame(
            {
                "target_date": pd.date_range("2020-01-01", periods=6, freq="D"),
                "forecast_source_family": ["rss", "press", "rss", "press", "rss", "press"],
                "target_tmax_c": [30.0, 11.0, 30.0, 11.0, 30.0, 11.0],
                "forecast_max_c": [10.0] * 6,
                "feature_a": [0.0, 100.0, 1.0, 101.0, 1.0, 101.0],
                "feature_b": [0.0, 100.0, 1.0, 101.0, 1.0, 101.0],
            }
        )
    )
    spec = LocalResidualSpec(
        feature_set="same_source_phase",
        features=("feature_a", "feature_b"),
        k_neighbors=1,
        same_source=True,
        phase_conditioned=True,
        shrinkage=0.0,
        correction_clip_c=25.0,
        min_history=2,
    )

    out = past_only_local_predictions(frame, spec)

    assert out.loc[4, "forecast_source_family"] == "rss"
    assert out.loc[4, "past_rows_used"] == 1
    assert out.loc[4, "residual_correction_c"] == pytest.approx(20.0)
    assert out.loc[4, "candidate_prediction_c"] == pytest.approx(30.0)
