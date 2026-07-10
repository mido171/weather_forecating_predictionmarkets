from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_hkg_t24_0086_guarded_long_history_residual_specialists import (
    SpecialistSpec,
    apply_specialist,
    assign_bucket,
)


def test_assign_bucket_uses_right_sided_thresholds() -> None:
    values = pd.Series([0.0, 1.0, 2.0, 3.0, np.nan])
    thresholds = np.array([1.0, 2.0])

    buckets = assign_bucket(values, thresholds)

    assert buckets.tolist()[:4] == [0.0, 1.0, 2.0, 2.0]
    assert np.isnan(buckets.iloc[4])


def make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-02"]),
            "forecast_source_family": ["press_archive", "press_archive", "press_archive"],
            "season": ["DJF", "DJF", "DJF"],
            "frame_segment": ["current_0081_frame", "current_0081_frame", "current_0081_frame"],
            "era_bucket": ["press_2000_2001", "press_2000_2001", "press_2000_2001"],
            "target_tmax_c": [18.0, 19.0, 20.0],
            "forecast_max_c": [20.0, 20.0, 30.0],
            "candidate_prediction_c": [20.0, 21.0, 22.0],
            "base_residual_c": [2.0, 2.0, 2.0],
            "feature_x__bucket": [0.0, 0.0, 0.0],
        }
    )


def test_specialist_uses_prior_dates_not_same_date_residuals() -> None:
    spec = SpecialistSpec(
        candidate_id="test",
        feature="feature_x",
        context_mode="feature",
        min_history=1,
        shrink_rows=0.0,
        correction_cap_c=10.0,
    )

    prediction, diagnostics = apply_specialist(make_frame(), spec)

    assert prediction.tolist() == [20.0, 19.0, 20.0]
    assert diagnostics["prior_rows"].tolist() == [0, 1, 1]
    assert diagnostics["specialist_active"].tolist() == [False, True, True]
