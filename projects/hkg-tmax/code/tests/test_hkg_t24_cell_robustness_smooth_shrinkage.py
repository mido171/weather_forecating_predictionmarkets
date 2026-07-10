from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_hkg_t24_cell_robustness_smooth_shrinkage import (
    SmoothShrinkageSpec,
    combine_smooth_decisions,
    ensure_calendar_columns,
    leakage_audit,
    select_smooth_decision,
    shrink_factor,
)
from scripts.run_hkg_t24_nonlinear_local_residual_fusion_lab import DELTA_GRID


def make_spec(**overrides: object) -> SmoothShrinkageSpec:
    values = {
        "candidate_id": "test",
        "mode": "causal_smooth_shrinkage",
        "candidate_class": "causal_smooth_shrinkage",
        "group_modes": ("source_signeddiff_range",),
        "min_history": 3,
        "min_prior_lift_c": 0.05,
        "support_shrink": 10.0,
        "lift_scale_c": 0.05,
        "base_shrink": 1.0,
        "max_abs_delta": 0.12,
        "combine_mode": "best_lift",
        "diagnostic_top_n": 0,
        "diagnostic_max_active_delta_mae": 0.0,
    }
    values.update(overrides)
    return SmoothShrinkageSpec(**values)  # type: ignore[arg-type]


def test_shrink_factor_is_bounded_and_monotonic() -> None:
    small = shrink_factor(count=10, prior_lift_c=0.02, support_shrink=100.0, lift_scale_c=0.05)
    large = shrink_factor(count=100, prior_lift_c=0.10, support_shrink=100.0, lift_scale_c=0.05)

    assert 0.0 < small < large < 1.0
    assert shrink_factor(count=0, prior_lift_c=1.0, support_shrink=100.0, lift_scale_c=0.05) == 0.0


def test_ensure_calendar_columns_derives_month_and_season() -> None:
    frame = pd.DataFrame({"target_date": pd.to_datetime(["2020-01-15", "2020-07-01"])})

    out = ensure_calendar_columns(frame)

    assert out["month"].astype(int).tolist() == [1, 7]
    assert out["season"].tolist() == ["DJF", "JJA"]


def test_select_smooth_decision_shrinks_best_delta() -> None:
    spec = make_spec(min_history=3, min_prior_lift_c=0.01, max_abs_delta=0.12)
    abs_sums = np.ones(len(DELTA_GRID), dtype=float) * 10.0
    zero_index = int(np.argmin(np.abs(np.array(DELTA_GRID))))
    best_index = list(DELTA_GRID).index(-0.18)
    abs_sums[zero_index] = 9.0
    abs_sums[best_index] = 6.0

    decision = select_smooth_decision(
        group_mode="g",
        group_key="k",
        count=3,
        abs_sums=abs_sums,
        spec=spec,
    )

    assert decision is not None
    assert decision.raw_delta == -0.18
    assert -0.12 <= decision.shrunk_delta < 0.0
    assert abs(decision.shrunk_delta) < abs(decision.raw_delta)


def test_select_smooth_decision_requires_min_history_and_lift() -> None:
    spec = make_spec(min_history=5, min_prior_lift_c=0.50)
    abs_sums = np.ones(len(DELTA_GRID), dtype=float)

    assert (
        select_smooth_decision(
            group_mode="g",
            group_key="k",
            count=4,
            abs_sums=abs_sums,
            spec=spec,
        )
        is None
    )


def test_combine_smooth_decisions_agreement_blocks_mixed_signs() -> None:
    spec_neg = make_spec(min_prior_lift_c=0.0)
    spec_agree = make_spec(combine_mode="agreement_mean")
    neg_sums = np.array([1.0 if value == -0.18 else 2.0 for value in DELTA_GRID])
    pos_sums = np.array([1.0 if value == 0.18 else 2.0 for value in DELTA_GRID])
    neg = select_smooth_decision(group_mode="g", group_key="a", count=10, abs_sums=neg_sums, spec=spec_neg)
    pos = select_smooth_decision(group_mode="g", group_key="b", count=10, abs_sums=pos_sums, spec=spec_neg)

    assert neg is not None
    assert pos is not None
    assert combine_smooth_decisions([neg, pos], spec_agree) == 0.0


def test_leakage_audit_keeps_diagnostic_atlas_non_deployable() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "base_0069_prediction_c": [20.0, 21.0],
        }
    )
    scoreboard = pd.DataFrame(
        {
            "candidate_class": ["diagnostic_smooth_atlas", "causal_smooth_shrinkage"],
            "deployable_gate_passed": [False, True],
            "delta_mae_vs_0069": [-0.01, -0.01],
            "fold_delta_max_vs_0069": [0.0, 0.0],
            "late_delta_mae_vs_0069": [0.0, 0.0],
        }
    )

    audit = leakage_audit(frame, scoreboard)

    assert audit["passed"].astype(bool).all()
