from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_hkg_t24_era_source_aware_fusion_model import (
    EraSourceFusionSpec,
    apply_prior_mixture_spec,
    apply_source_fixed_spec,
    group_key,
    leakage_audit,
    select_prior_weight,
    source_base_weight,
)
from scripts.run_hkg_t24_prior_calibrated_fusion_screen import WEIGHT_GRID


def make_spec(**overrides: object) -> EraSourceFusionSpec:
    values = {
        "candidate_id": "test",
        "mode": "source_fixed_weight",
        "candidate_class": "diagnostic_source_map",
        "primary_group_mode": "source",
        "secondary_group_mode": "global",
        "min_history": 0,
        "fallback_weight": 0.15,
        "fallback_mode": "source_map",
        "temperature_c": 0.0,
        "primary_alpha": 0.0,
        "secondary_alpha": 0.0,
        "global_alpha": 0.0,
        "press_weight": 0.28,
        "rss_weight": 0.15,
        "tilt_mode": "none",
        "tilt_step": 0.0,
        "cap_low": 0.0,
        "cap_high": 0.50,
    }
    values.update(overrides)
    return EraSourceFusionSpec(**values)  # type: ignore[arg-type]


def base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "target_tmax_c": [24.0, 30.0],
            "official_family_prediction_c": [20.0, 20.0],
            "station_family_prediction_c": [24.0, 30.0],
            "forecast_source_family": ["press_archive", "rss_archive"],
            "family_disagreement_c": [4.0, 10.0],
            "abs_family_disagreement_c": [4.0, 10.0],
            "active_member_count": [3, 0],
            "fold_id": ["fold_1", "fold_2"],
        }
    )


def test_source_base_weight_uses_family_specific_weights() -> None:
    spec = make_spec(press_weight=0.30, rss_weight=0.12)
    frame = base_frame()

    assert source_base_weight(frame.iloc[0], spec) == 0.30
    assert source_base_weight(frame.iloc[1], spec) == 0.12


def test_group_key_supports_source_signeddiff_active() -> None:
    key = group_key(base_frame().iloc[0], "source_signeddiff_active")

    assert key == "press_archive|station_warmer_ge_1c|station_stack_three_plus_members"


def test_source_fixed_tilt_is_clipped() -> None:
    spec = make_spec(
        tilt_mode="active_more",
        tilt_step=0.40,
        press_weight=0.30,
        rss_weight=0.15,
        cap_high=0.50,
    )

    out = apply_source_fixed_spec(base_frame(), spec)

    assert out["station_weight"].iloc[0] == 0.50
    assert out["station_weight"].iloc[1] == 0.0


def test_prior_mixture_selector_excludes_current_row() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "target_tmax_c": [24.0, 30.0],
            "official_family_prediction_c": [20.0, 20.0],
            "station_family_prediction_c": [24.0, 30.0],
            "forecast_source_family": ["rss_archive", "rss_archive"],
            "family_disagreement_c": [4.0, 10.0],
            "abs_family_disagreement_c": [4.0, 10.0],
            "active_member_count": [1, 1],
            "fold_id": ["fold", "fold"],
        }
    )
    spec = make_spec(
        mode="prior_best_weight",
        candidate_class="causal_prior_selector",
        primary_group_mode="global",
        secondary_group_mode="global",
        min_history=1,
        fallback_weight=0.0,
        fallback_mode="constant",
        primary_alpha=1.0,
        secondary_alpha=0.0,
        global_alpha=0.0,
    )

    out = apply_prior_mixture_spec(frame, spec)

    assert out["station_weight"].iloc[0] == 0.0
    assert out["station_weight"].iloc[1] == max(WEIGHT_GRID)
    assert out["primary_prior_count"].iloc[1] == 1


def test_select_prior_weight_softmax_stays_inside_bounds() -> None:
    abs_sums = np.linspace(10.0, 1.0, len(WEIGHT_GRID))

    weight = select_prior_weight(
        abs_sums=abs_sums,
        count=10,
        mode="prior_soft_weight",
        min_history=1,
        fallback_weight=0.15,
        temperature_c=0.05,
    )

    assert 0.0 <= weight <= max(WEIGHT_GRID)


def test_leakage_audit_keeps_diagnostic_maps_non_deployable() -> None:
    frame = base_frame()
    scoreboard = pd.DataFrame(
        {
            "candidate_class": ["diagnostic_source_map", "causal_prior_selector"],
            "deployable_gate_passed": [False, True],
            "mae": [0.9, 0.8],
        }
    )

    audit = leakage_audit(frame, scoreboard, {"best_mae": 0.81})

    diagnostic_check = audit[audit["check_id"].eq("diagnostic_source_maps_not_marked_deployable")]
    assert diagnostic_check["passed"].iloc[0]
