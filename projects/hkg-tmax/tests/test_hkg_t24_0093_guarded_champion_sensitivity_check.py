from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_0093_guarded_champion_sensitivity_check import (
    SensitivitySpec,
    apply_sensitivity_spec,
    guard_subsets,
    make_sensitivity_specs,
)


def test_guard_subsets_include_none_singletons_and_all() -> None:
    variants = guard_subsets("season_JJA;season_SON")

    assert variants == [
        ("no_guard", ()),
        ("guard_season-JJA", ("season_JJA",)),
        ("guard_season-SON", ("season_SON",)),
        ("guard_all_0092_failed", ("season_JJA", "season_SON")),
    ]


def test_make_sensitivity_specs_crosses_guard_history_shrink_and_cap() -> None:
    champion = pd.Series(
        {
            "feature": "isd_morning_to_midday_temp_rise_c",
            "context_mode": "source_season_feature",
            "failed_slices_guarded": "season_JJA;season_SON",
        }
    )

    specs = make_sensitivity_specs(champion)

    assert len(specs) == 4 * 4 * 2 * 4
    assert any(spec.min_history == 180 for spec in specs)
    assert any(spec.correction_cap_c == 0.35 for spec in specs)
    assert any(spec.guard_variant == "guard_all_0092_failed" for spec in specs)


def test_apply_sensitivity_spec_uses_base_prediction_on_guarded_slice() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-06-01"]),
            "forecast_source_family": ["press_archive", "press_archive", "press_archive"],
            "season": ["DJF", "DJF", "JJA"],
            "frame_segment": ["current_0081_frame"] * 3,
            "era_bucket": ["a", "a", "a"],
            "target_tmax_c": [18.0, 19.0, 20.0],
            "forecast_max_c": [20.0, 20.0, 20.0],
            "candidate_prediction_c": [20.0, 21.0, 22.0],
            "base_residual_c": [2.0, 2.0, 2.0],
            "feature_x__bucket": [0.0, 0.0, 0.0],
        }
    )
    spec = SensitivitySpec(
        candidate_id="test",
        feature="feature_x",
        context_mode="feature",
        min_history=1,
        shrink_rows=0.0,
        correction_cap_c=10.0,
        guard_variant="guard_jja",
        guard_slices=("season_JJA",),
    )

    prediction, diagnostics = apply_sensitivity_spec(frame, spec)

    assert prediction.tolist() == [20.0, 19.0, 22.0]
    assert diagnostics["guard_slices"].iloc[0] == "season_JJA"


def test_apply_sensitivity_spec_prediction_only_matches_diagnostic_path() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-06-01"]),
            "forecast_source_family": ["press_archive"] * 4,
            "season": ["DJF", "DJF", "DJF", "JJA"],
            "frame_segment": ["current_0081_frame"] * 4,
            "era_bucket": ["a"] * 4,
            "target_tmax_c": [18.0, 19.0, 18.5, 20.0],
            "forecast_max_c": [20.0, 20.0, 20.0, 20.0],
            "candidate_prediction_c": [20.0, 21.0, 21.0, 22.0],
            "base_residual_c": [2.0, 2.0, 2.5, 2.0],
            "feature_x__bucket": [0.0, 0.0, 0.0, 0.0],
        }
    )
    spec = SensitivitySpec(
        candidate_id="test",
        feature="feature_x",
        context_mode="feature",
        min_history=1,
        shrink_rows=0.0,
        correction_cap_c=10.0,
        guard_variant="no_guard",
        guard_slices=(),
    )

    diagnostic_prediction, diagnostics = apply_sensitivity_spec(frame, spec)
    light_prediction, light_diagnostics = apply_sensitivity_spec(frame, spec, include_diagnostics=False)

    assert light_prediction.tolist() == diagnostic_prediction.tolist()
    assert diagnostics.empty is False
    assert light_diagnostics.empty
