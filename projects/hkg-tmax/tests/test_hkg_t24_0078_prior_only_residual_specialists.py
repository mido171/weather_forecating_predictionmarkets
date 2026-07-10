from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_0078_prior_only_residual_specialists import (
    FeaturePredicate,
    SpecialistSpec,
    apply_specialist_spec,
    predicate_is_active,
    specialist_specs,
)


def make_spec(**overrides: object) -> SpecialistSpec:
    values = {
        "candidate_id": "test",
        "predicate": FeaturePredicate(
            feature="signal",
            direction="high",
            threshold=1.0,
            family="test_family",
            label="signal_high",
        ),
        "context_mode": "feature",
        "min_history": 1,
        "halflife_rows": 100.0,
        "support_shrink": 0.0,
        "min_prior_lift_c": -1.0,
        "correction_cap_c": 5.0,
    }
    values.update(overrides)
    return SpecialistSpec(**values)  # type: ignore[arg-type]


def test_predicate_is_active_handles_high_low_and_missing() -> None:
    high = FeaturePredicate("x", "high", 2.0, "test", "x_high")
    low = FeaturePredicate("x", "low", 2.0, "test", "x_low")

    assert predicate_is_active(pd.Series({"x": 2.5}), high)
    assert not predicate_is_active(pd.Series({"x": 1.5}), high)
    assert predicate_is_active(pd.Series({"x": 1.5}), low)
    assert not predicate_is_active(pd.Series({"x": None}), low)


def test_specialist_specs_do_not_use_current_target_columns() -> None:
    specs = specialist_specs({"target_tmax_c_current", "target_roll90_mean_lag7_c", "isd_dewpoint_midday_minus_temp_c"})

    assert specs
    assert all(not spec.predicate.feature.startswith("target_tmax_c") for spec in specs)


def test_specialist_does_not_use_current_row_for_own_correction() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2021-01-01", "2021-01-02"]),
            "current_target_tmax_c": [25.0, 25.0],
            "m0075_prediction_c": [20.0, 20.0],
            "fold_id": ["fold_a", "fold_a"],
            "forecast_source_family": ["rss_archive", "rss_archive"],
            "season": ["DJF", "DJF"],
            "month": [1, 1],
            "signal": [2.0, 2.0],
        }
    )

    predictions = apply_specialist_spec(frame, make_spec())

    assert predictions.loc[0, "specialist_correction_c"] == 0.0
    assert predictions.loc[1, "specialist_correction_c"] > 0.0


def test_inactive_predicate_does_not_update_state() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2021-01-01", "2021-01-02"]),
            "current_target_tmax_c": [25.0, 25.0],
            "m0075_prediction_c": [20.0, 20.0],
            "fold_id": ["fold_a", "fold_a"],
            "forecast_source_family": ["rss_archive", "rss_archive"],
            "season": ["DJF", "DJF"],
            "month": [1, 1],
            "signal": [0.0, 2.0],
        }
    )

    predictions = apply_specialist_spec(frame, make_spec())

    assert predictions.loc[0, "specialist_correction_c"] == 0.0
    assert predictions.loc[1, "specialist_correction_c"] == 0.0
