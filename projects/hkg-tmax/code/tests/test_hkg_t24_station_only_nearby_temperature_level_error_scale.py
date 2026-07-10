from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_station_only_nearby_temperature_level_error_scale import (
    TEMP_HIGH,
    TEMP_LOW,
    TEMP_MID,
    TempScaleSpec,
    bucket_by_edges,
    compute_temp_scale,
)


def test_bucket_by_edges_labels_low_mid_high() -> None:
    buckets = bucket_by_edges(pd.Series([10.0, 20.0, 30.0, None]), low=15.0, high=25.0)

    assert buckets.tolist() == [TEMP_LOW, TEMP_MID, TEMP_HIGH, "nearby_temp_missing"]


def test_temp_scale_excludes_current_row_from_mean_and_scale() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "nearby_temp_bucket": [TEMP_HIGH, TEMP_HIGH, TEMP_HIGH],
            "residual_to_add_c": [1.0, 3.0, 90.0],
            "reference_expected_abs_c": [1.0, 1.0, 1.0],
        }
    )
    spec = TempScaleSpec(
        "test",
        ("nearby_temp_bucket",),
        TEMP_HIGH,
        min_prior_rows=1,
        mean_shrinkage=0.0,
        scale_shrinkage=0.0,
        mean_cap_c=100.0,
        min_sigma_multiplier=0.5,
        max_sigma_multiplier=10.0,
    )

    corrections, multipliers, prior_rows, raw_means, raw_multipliers = compute_temp_scale(frame, spec)

    assert corrections.tolist() == [0.0, 1.0, 2.0]
    assert multipliers.tolist() == [1.0, 1.0, 2.0]
    assert prior_rows.tolist() == [0, 1, 2]
    assert raw_means[0] != raw_means[0]
    assert raw_multipliers[0] != raw_multipliers[0]
