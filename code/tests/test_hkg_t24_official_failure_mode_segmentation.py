from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_hkg_t24_official_failure_mode_segmentation import (
    assign_failure_flags,
    quantile_bucket,
    standardized_mean_diff,
)


def test_assign_failure_flags_labels_directional_tails() -> None:
    frame = pd.DataFrame(
        {
            "forecast_max_c": [10.0, 10.0, 10.0, 10.0, 10.0],
            "target_tmax_c": [15.0, 12.0, 10.0, 8.0, 5.0],
        }
    )
    frame["official_error_c"] = frame["forecast_max_c"] - frame["target_tmax_c"]
    frame["official_abs_error_c"] = frame["official_error_c"].abs()

    out = assign_failure_flags(frame)

    assert out.loc[0, "failure_mode"] == "severe_underprediction"
    assert out.loc[4, "failure_mode"] == "severe_overprediction"
    assert out.loc[2, "failure_mode"] == "routine"


def test_standardized_mean_diff_uses_flag_minus_rest() -> None:
    n, mean_diff, median_diff, std_diff = standardized_mean_diff(
        pd.Series([1.0, 2.0, 10.0, 12.0]),
        pd.Series([False, False, True, True]),
    )

    assert n == 4
    assert mean_diff == pytest.approx(9.5)
    assert median_diff == pytest.approx(9.5)
    assert std_diff > 0


def test_quantile_bucket_marks_insufficient_data() -> None:
    buckets = quantile_bucket(pd.Series([1.0, 2.0, 3.0]), 5)

    assert buckets.to_list() == ["insufficient", "insufficient", "insufficient"]
