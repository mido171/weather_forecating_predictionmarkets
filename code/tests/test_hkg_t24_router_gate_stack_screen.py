from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_router_gate_stack_screen import (
    StackCandidate,
    build_candidate_catalog,
    combine_fixed_gate_residual,
    combine_prior_predictions,
    top_gate_candidate_ids,
)


def stack_candidate(
    *,
    mode: str = "prior_best",
    same_source: bool = False,
    min_history: int = 2,
    gate_scale: float = 0.0,
) -> StackCandidate:
    return StackCandidate(
        candidate_id=f"case_{mode}",
        router_candidate_id="router_a",
        gate_candidate_id="gate_a",
        mode=mode,  # type: ignore[arg-type]
        same_source=same_source,
        min_history=min_history,
        gate_scale=gate_scale,
    )


def base_pair_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-03"]),
            "forecast_source_family": ["press", "press", "press", "press"],
            "target_tmax_c": [10.0, 10.0, 10.0, 10.0],
            "official_raw": [8.0, 8.0, 8.0, 8.0],
            "anchor_0038_c": [8.0, 8.0, 8.0, 8.0],
            "router_prediction_c": [8.0, 8.0, 8.0, 8.0],
            "gate_prediction_c": [10.0, 10.0, 10.0, 10.0],
            "residual_correction_c": [2.0, 2.0, 2.0, 2.0],
        }
    )


def test_build_candidate_catalog_has_unique_ids() -> None:
    catalog = build_candidate_catalog(["router_one", "router_two"], ["gate_one", "gate_two"])

    assert catalog["candidate_id"].is_unique
    assert {"prior_best", "prior_inverse_mae", "prior_positive_lift", "fixed_gate_residual"}.issubset(
        set(catalog["mode"])
    )


def test_top_gate_candidate_ids_ranks_only_available_prediction_ids() -> None:
    scoreboard = pd.DataFrame(
        {
            "candidate_id": ["missing_best", "available_second"],
            "full_mae": [0.1, 0.2],
            "late_mae": [0.1, 0.2],
            "full_rmse": [0.1, 0.2],
            "late_rmse": [0.1, 0.2],
        }
    )
    predictions = pd.DataFrame({"candidate_id": ["available_second"]})

    assert top_gate_candidate_ids(scoreboard, predictions) == ["available_second"]


def test_prior_best_excludes_same_date_rows() -> None:
    out = combine_prior_predictions(base_pair_frame(), stack_candidate())

    assert out.loc[0, "selected_family"] == "router_fallback"
    assert out.loc[1, "selected_family"] == "router_fallback"
    assert out.loc[1, "selected_prior_count"] == 0
    assert out.loc[2, "selected_family"] == "gate"
    assert out.loc[2, "candidate_prediction_c"] == 10.0


def test_same_source_prior_mode_isolates_history() -> None:
    frame = base_pair_frame()
    frame.loc[:, "forecast_source_family"] = ["press", "press", "rss", "rss"]

    all_prior = combine_prior_predictions(frame, stack_candidate(same_source=False))
    same_source = combine_prior_predictions(frame, stack_candidate(same_source=True, min_history=1))

    assert all_prior.loc[2, "selected_family"] == "gate"
    assert same_source.loc[2, "selected_family"] == "router_fallback"
    assert same_source.loc[3, "selected_family"] == "gate"


def test_fixed_gate_residual_adds_scaled_gate_correction() -> None:
    out = combine_fixed_gate_residual(base_pair_frame(), stack_candidate(mode="fixed_gate_residual", gate_scale=0.5))

    assert out["candidate_prediction_c"].tolist() == [9.0, 9.0, 9.0, 9.0]
    assert out["gate_weight"].tolist() == [0.5, 0.5, 0.5, 0.5]
