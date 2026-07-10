from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_hkg_t24_official_anchor_expert_blend_screen import past_only_expert_blend


def test_past_only_best_expert_uses_prior_error_only() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "forecast_source_family": ["rss"] * 5,
            "target_tmax_c": [10.0, 10.0, 10.0, 100.0, 10.0],
            "official_raw": [10.0, 10.0, 10.0, 10.0, 10.0],
            "expert_a": [10.0, 10.0, 10.0, 100.0, 100.0],
            "expert_b": [20.0, 20.0, 20.0, 20.0, 10.0],
        }
    )

    out = past_only_expert_blend(
        frame,
        experts=["official_raw", "expert_a", "expert_b"],
        mode="best",
        same_source=False,
        min_history=2,
    )

    assert out.loc[0, "selected_expert"] == "official_raw_fallback"
    assert out.loc[2, "selected_expert"] in {"expert_a", "official_raw"}
    assert out.loc[3, "selected_expert"] == "expert_a"
    assert out.loc[3, "expert_prediction_c"] == pytest.approx(100.0)
    assert out.loc[4, "selected_expert"] == "expert_a"


def test_past_only_inverse_mae_blend_falls_back_without_history() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.date_range("2020-01-01", periods=2, freq="D"),
            "forecast_source_family": ["rss", "rss"],
            "target_tmax_c": [10.0, 10.0],
            "official_raw": [11.0, 11.0],
            "expert_a": [10.0, 10.0],
        }
    )

    out = past_only_expert_blend(
        frame,
        experts=["official_raw", "expert_a"],
        mode="inverse_mae",
        same_source=True,
        min_history=3,
    )

    assert out["selected_expert"].to_list() == ["official_raw_fallback", "official_raw_fallback"]
    assert out["expert_prediction_c"].to_list() == [11.0, 11.0]
