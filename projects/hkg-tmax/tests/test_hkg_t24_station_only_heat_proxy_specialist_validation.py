from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_station_only_heat_proxy_specialist_validation import (
    DJF_MONTHS,
    ProxySpec,
    compute_prior_proxy_correction,
    mask_for_proxy,
)


def test_mask_for_proxy_uses_month_and_declared_feature_buckets() -> None:
    frame = pd.DataFrame(
        {
            "month": [1, 1, 7],
            "station_temp_level_bucket": ["mid", "high", "mid"],
            "diagnostic_heat_bucket_pre2000_target": ["high", "mid", "mid"],
        }
    )
    spec = ProxySpec(
        "winter_mid",
        "winter_mid_heat_proxy",
        DJF_MONTHS,
        (("station_temp_level_bucket", "mid"),),
        min_prior_rows=1,
        shrinkage=0.0,
        cap_c=2.0,
    )

    mask = mask_for_proxy(frame, spec)

    assert mask.tolist() == [True, False, False]


def test_prior_proxy_correction_excludes_current_active_row() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "reference_residual_to_add_c": [1.0, 3.0, 90.0],
        }
    )
    active = pd.Series([True, True, True])
    spec = ProxySpec(
        "test",
        "winter_mid_heat_proxy",
        DJF_MONTHS,
        (("station_temp_level_bucket", "mid"),),
        min_prior_rows=1,
        shrinkage=0.0,
        cap_c=100.0,
    )

    corrections, prior_rows, raw_means = compute_prior_proxy_correction(frame, active, spec)

    assert corrections.tolist() == [0.0, 1.0, 2.0]
    assert prior_rows.tolist() == [0, 1, 2]
    assert raw_means[0] != raw_means[0]
