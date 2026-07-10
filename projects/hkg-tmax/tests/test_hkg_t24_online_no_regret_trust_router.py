from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_online_no_regret_trust_router import (
    TrustRouterSpec,
    apply_trust_router_spec,
    trust_router_specs,
)


def make_spec(**overrides: object) -> TrustRouterSpec:
    values = {
        "candidate_id": "test",
        "context_set": "behavior",
        "selection_mode": "no_regret_0075_gate",
        "min_history": 1,
        "halflife_rows": 100.0,
        "min_edge_c": 0.0,
        "fallback_model": "m0075",
    }
    values.update(overrides)
    return TrustRouterSpec(**values)  # type: ignore[arg-type]


def router_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2021-01-01", "2021-01-02"]),
            "target_tmax_c": [0.0, 0.0],
            "fold_id": ["fold_a", "fold_a"],
            "forecast_source_family": ["rss_archive", "rss_archive"],
            "season": ["DJF", "DJF"],
            "month": [1, 1],
            "signeddiff_bucket": ["neutral", "neutral"],
            "forecast_range_bucket": ["range_le_3c", "range_le_3c"],
            "weather_bucket": ["weather_other", "weather_other"],
            "active_count_bucket": ["station_stack_inactive", "station_stack_inactive"],
            "m0069_prediction_c": [0.0, 0.0],
            "m0074_prediction_c": [5.0, 5.0],
            "m0075_prediction_c": [10.0, 10.0],
        }
    )


def test_trust_router_specs_are_unique_and_bounded() -> None:
    specs = trust_router_specs()
    ids = [spec.candidate_id for spec in specs]

    assert len(specs) == 54
    assert len(ids) == len(set(ids))
    assert {spec.selection_mode for spec in specs} == {"best_model", "inverse_mae_blend", "no_regret_0075_gate"}


def test_no_regret_router_does_not_use_current_row_for_its_own_choice() -> None:
    predictions = apply_trust_router_spec(router_frame(), make_spec())

    assert predictions.loc[0, "selected_model"] == "m0075"
    assert predictions.loc[0, "candidate_prediction_c"] == 10.0
    assert predictions.loc[1, "selected_model"] == "m0069"
    assert predictions.loc[1, "candidate_prediction_c"] == 0.0


def test_inverse_mae_blend_stays_between_available_model_predictions_after_prior_exists() -> None:
    predictions = apply_trust_router_spec(router_frame(), make_spec(selection_mode="inverse_mae_blend"))

    assert predictions.loc[0, "selected_model"] == "m0075"
    assert 0.0 <= predictions.loc[1, "candidate_prediction_c"] <= 10.0
    assert predictions.loc[1, "selected_model"] == "blend_inverse_mae"
