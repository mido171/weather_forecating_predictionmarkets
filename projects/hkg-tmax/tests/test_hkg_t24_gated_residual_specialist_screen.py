from __future__ import annotations

import math

import pandas as pd

from scripts.run_hkg_t24_gated_residual_specialist_screen import (
    context_state,
    parse_edges,
    past_only_residual_correction,
)


def test_parse_edges_accepts_json_pair_only() -> None:
    assert parse_edges("[1.5, 3.0]") == (1.5, 3.0)
    assert parse_edges("[3.0, 1.5]") is None
    assert parse_edges("not-json") is None


def test_context_state_adds_season_without_changing_base() -> None:
    frame = pd.DataFrame({"season": [0, 1], "month": [1, 4]})
    base = pd.Series(["low|high", "mid|low"])
    assert context_state(base, frame, "base").tolist() == ["low|high", "mid|low"]
    assert context_state(base, frame, "season").tolist() == ["low|high|season=0", "mid|low|season=1"]
    assert context_state(base, frame, "month").tolist() == ["low|high|month=1", "mid|low|month=4"]


def test_past_only_residual_correction_excludes_same_date_rows() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-03"]),
            "forecast_source_family": ["press", "press", "press", "press"],
            "residual_to_add_c": [10.0, 10.0, 2.0, 2.0],
        }
    )
    state = pd.Series(["a", "a", "a", "a"])
    correction, cell_rows, global_rows = past_only_residual_correction(
        frame,
        state=state,
        same_source=False,
        min_history=2,
        shrink_k=0.0,
        correction_cap_c=99.0,
    )
    assert correction.tolist()[0:2] == [0.0, 0.0]
    assert cell_rows.tolist()[0:2] == [0, 0]
    assert global_rows.tolist()[0:2] == [0, 0]
    assert correction.iloc[2] == 10.0
    assert math.isclose(correction.iloc[3], 22.0 / 3.0)


def test_past_only_residual_correction_same_source_isolates_history() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "forecast_source_family": ["press", "rss", "rss"],
            "residual_to_add_c": [10.0, 2.0, 4.0],
        }
    )
    state = pd.Series(["a", "a", "a"])
    correction, _cell_rows, global_rows = past_only_residual_correction(
        frame,
        state=state,
        same_source=True,
        min_history=1,
        shrink_k=0.0,
        correction_cap_c=99.0,
    )
    assert correction.iloc[0] == 0.0
    assert correction.iloc[1] == 0.0
    assert global_rows.iloc[1] == 0
    assert correction.iloc[2] == 2.0
