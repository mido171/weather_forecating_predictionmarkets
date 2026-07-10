from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_hkg_t24_0088_prior_gated_specialist_stack import (
    StackSpec,
    apply_stack,
    context_key,
)


def make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-02"]),
            "forecast_source_family": ["press_archive", "press_archive", "press_archive"],
            "season": ["DJF", "DJF", "DJF"],
            "frame_segment": ["current_0081_frame", "current_0081_frame", "current_0081_frame"],
            "era_bucket": ["press_2000_2001", "press_2000_2001", "press_2000_2001"],
            "target_tmax_c": [10.0, 10.0, 10.0],
        }
    )


def test_context_key_can_include_source_and_frame() -> None:
    assert context_key(make_frame().iloc[0], "source_frame") == ("press_archive", "current_0081_frame")


def test_prior_gated_stack_uses_prior_dates_not_same_date_errors() -> None:
    frame = make_frame()
    candidate_ids = ["official_raw", "0086_base", "0087_best"]
    matrix = np.array(
        [
            [10.0, 20.0, 30.0],
            [99.0, 20.0, 10.0],
            [100.0, 20.0, 10.0],
        ]
    )
    spec = StackSpec(
        candidate_id="test",
        mode="selector",
        context_mode="global",
        min_history=1,
        blend_top_k=None,
    )

    prediction, selected = apply_stack(frame, spec, candidate_ids, matrix)

    assert prediction.tolist() == [20.0, 99.0, 100.0]
    assert selected == ["0086_base", "official_raw", "official_raw"]
