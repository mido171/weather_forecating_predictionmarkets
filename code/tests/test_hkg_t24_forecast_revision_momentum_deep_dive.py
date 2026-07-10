from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_forecast_revision_momentum_deep_dive import (
    RevisionMomentumSpec,
    add_text_source_change_features,
    build_revision_specs,
    past_only_bias_feature,
    past_only_revision_predictions,
)
from scripts.run_hkg_t24_residual_failure_cluster_discovery import ArchetypeCondition


def test_past_only_bias_feature_excludes_current_label() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "forecast_source_family": ["press", "press"],
            "target_tmax_c": [12.0, 100.0],
            "prediction_0018_c": [10.0, 10.0],
        }
    )

    bias = past_only_bias_feature(
        frame,
        anchor_col="prediction_0018_c",
        same_source=False,
        lookback_rows=10,
        min_history=1,
    )

    assert pd.isna(bias.iloc[0])
    assert bias.iloc[1] == 2.0


def test_past_only_bias_feature_same_source_isolates_history() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "forecast_source_family": ["press", "rss", "rss"],
            "target_tmax_c": [10.0, 30.0, 10.0],
            "prediction_0018_c": [10.0, 10.0, 10.0],
        }
    )

    bias = past_only_bias_feature(
        frame,
        anchor_col="prediction_0018_c",
        same_source=True,
        lookback_rows=10,
        min_history=1,
    )

    assert bias.iloc[2] == 20.0


def test_add_text_source_change_features_respects_source_family() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "forecast_source_family": ["press", "rss", "press"],
            "text_hot": [0.0, 1.0, 1.0],
        }
    )

    out = add_text_source_change_features(frame)

    assert pd.isna(out.loc[1, "text_hot_change_1_source"])
    assert out.loc[2, "text_hot_lag1_source"] == 0.0
    assert out.loc[2, "text_hot_turned_on_source"] == 1.0


def test_past_only_revision_predictions_excludes_current_label() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "forecast_source_family": ["press", "press"],
            "primary_regime": ["default", "default"],
            "target_tmax_c": [12.0, 100.0],
            "official_raw": [10.0, 10.0],
            "prediction_0018_c": [10.0, 10.0],
            "forecast_max_change_1_source_c": [1.0, 2.0],
        }
    )
    spec = RevisionMomentumSpec(
        family_name="guard",
        anchor_col="prediction_0018_c",
        conditions=(ArchetypeCondition("forecast_max_change_1_source_c", "high", 0.50),),
        features=("forecast_max_change_1_source_c",),
        k_neighbors=1,
        same_source=False,
        half_life_days=None,
        min_history=1,
        min_match_rows=1,
        shrinkage=0.0,
        correction_clip_c=100.0,
    )

    predictions = past_only_revision_predictions(frame, spec)

    assert bool(predictions.loc[1, "do_no_harm_gate_passed"]) is True
    assert predictions.loc[1, "candidate_prediction_c"] == 12.0


def test_past_only_revision_predictions_same_source_isolates_history() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "forecast_source_family": ["press", "rss", "rss"],
            "primary_regime": ["default", "default", "default"],
            "target_tmax_c": [10.0, 30.0, 10.0],
            "official_raw": [10.0, 10.0, 10.0],
            "prediction_0018_c": [10.0, 10.0, 10.0],
            "forecast_max_change_1_source_c": [1.0, 1.0, 1.0],
        }
    )
    spec = RevisionMomentumSpec(
        family_name="same_source",
        anchor_col="prediction_0018_c",
        conditions=(ArchetypeCondition("forecast_max_change_1_source_c", "flag"),),
        features=("forecast_max_change_1_source_c",),
        k_neighbors=1,
        same_source=True,
        half_life_days=None,
        min_history=1,
        min_match_rows=1,
        shrinkage=0.0,
        correction_clip_c=100.0,
    )

    predictions = past_only_revision_predictions(frame, spec)

    assert bool(predictions.loc[2, "do_no_harm_gate_passed"]) is True
    assert predictions.loc[2, "candidate_prediction_c"] == 30.0


def test_build_revision_specs_filters_missing_families() -> None:
    rows = 350
    frame = pd.DataFrame(
        {
            "prediction_0018_c": [25.0] * rows,
            "prediction_0026_c": [25.0] * rows,
            "forecast_max_c": [25 + (index % 8) for index in range(rows)],
            "forecast_min_c": [20 + (index % 5) for index in range(rows)],
            "forecast_range_c": [5.0] * rows,
            "forecast_max_change_1_source_c": [index % 20 for index in range(rows)],
            "forecast_max_prior7_std_source_c": [index % 7 for index in range(rows)],
            "forecast_max_vs_prior7_mean_source_c": [index % 17 for index in range(rows)],
            "forecast_min_change_1_source_c": [index % 6 for index in range(rows)],
            "forecast_range_change_1_source_c": [0.0] * rows,
            "forecast_midpoint_change_1_source_c": [index % 9 for index in range(rows)],
            "issue_to_cutoff_change_1_source_c": [index % 11 for index in range(rows)],
            "month": [(index % 12) + 1 for index in range(rows)],
            "monsoon_phase_code": [index % 4 for index in range(rows)],
            "isd_pressure_mean_hpa_change_1d": [0.0] * rows,
            "pressure_plane_slope_magnitude_hpa_per_deg": [0.0] * rows,
            "isd_wind_speed_mean_mps": [index % 5 for index in range(rows)],
            "isd_wind_speed_max_mps": [index % 6 for index in range(rows)],
        }
    )

    specs = build_revision_specs(frame)

    assert specs
    assert {spec.family_name for spec in specs} == {"jump_core"}
