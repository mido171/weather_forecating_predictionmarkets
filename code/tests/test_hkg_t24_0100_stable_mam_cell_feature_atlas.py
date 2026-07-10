from __future__ import annotations

import math

import pandas as pd

from scripts.run_hkg_t24_0100_stable_mam_cell_feature_atlas import (
    atlas_family,
    diagnostic_score,
    feature_timestamp_status,
    quantile_response_spread,
    standardized_mean_diff,
)


def test_atlas_family_promotes_daily_sea_temperature_to_marine_proxy() -> None:
    assert atlas_family("daily_waglan_island_sea_temperature_lag7") == "marine_proxy"
    assert atlas_family("daily_north_point_sea_temperature_am_lag7_roll7") == "marine_proxy"
    assert atlas_family("daily_hong_kong_observatory_mean_cloud_amount_lag7") == "hko_daily_climate"
    assert atlas_family("isd_station_air_temperature_c_450110_99999") == "isd_station_network"
    assert atlas_family("volatility_mad_14_lag7_c") == "target_memory"


def test_feature_timestamp_status_allows_only_proven_future_inputs() -> None:
    target_status = feature_timestamp_status("target_roll120_mean_lag7_c", "target_memory")
    trajectory_status = feature_timestamp_status("trajectory_7_30_slope_c_per_day", "target_memory")
    station_status = feature_timestamp_status("isd_station_air_temperature_c_450110_99999", "isd_station_network")
    upper_air_status = feature_timestamp_status("ua_theta_925hpa_k", "upper_air")
    marine_status = feature_timestamp_status("daily_waglan_island_sea_temperature_lag7", "marine_proxy")

    assert target_status["allowed_for_future_walkforward"] is True
    assert trajectory_status["allowed_for_future_walkforward"] is True
    assert station_status["allowed_for_future_walkforward"] is True
    assert upper_air_status["allowed_for_future_walkforward"] is False
    assert upper_air_status["timestamp_audit_status"] == "timestamp_audit_required"
    assert marine_status["allowed_for_future_walkforward"] is False
    assert marine_status["timestamp_audit_status"] == "publication_lag_audit_required"


def test_standardized_mean_diff_uses_pooled_variance() -> None:
    group_n, ref_n, diff = standardized_mean_diff(
        pd.Series([3.0, 4.0, 5.0, 6.0]),
        pd.Series([1.0, 2.0, 3.0, 4.0]),
        min_rows=4,
    )

    assert group_n == 4
    assert ref_n == 4
    assert diff > 1.0


def test_standardized_mean_diff_returns_nan_for_tiny_or_flat_groups() -> None:
    _, _, tiny = standardized_mean_diff(pd.Series([1.0]), pd.Series([2.0]), min_rows=2)
    _, _, flat = standardized_mean_diff(pd.Series([1.0, 1.0]), pd.Series([1.0, 1.0]), min_rows=2)

    assert math.isnan(tiny)
    assert math.isnan(flat)


def test_quantile_response_spread_detects_response_separation() -> None:
    feature = pd.Series(range(12))
    response = pd.Series([0.0] * 4 + [1.0] * 4 + [3.0] * 4)

    spread = quantile_response_spread(feature, response, q=3, min_bucket_rows=4)

    assert spread["spread_rows"] == 12
    assert spread["spread_bucket_count"] == 3
    assert spread["spread_c"] == 3.0


def test_quantile_response_spread_requires_enough_rows() -> None:
    spread = quantile_response_spread(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]), q=3, min_bucket_rows=2)

    assert spread["spread_bucket_count"] == 0
    assert math.isnan(spread["spread_c"])


def test_diagnostic_score_rewards_signal_and_row_support() -> None:
    strong = diagnostic_score(
        corr_agreement_base_error=0.5,
        corr_agreement_base_abs_error=-0.4,
        corr_agreement_abs_improvement=0.3,
        corr_active_base_error=0.2,
        base_error_spread_c=1.0,
        base_abs_error_spread_c=0.8,
        improvement_spread_c=0.4,
        agreement_vs_other_std_diff=0.5,
        agreement_rows=60,
        full_history_rows=365 * 39,
    )
    weak = diagnostic_score(
        corr_agreement_base_error=0.5,
        corr_agreement_base_abs_error=-0.4,
        corr_agreement_abs_improvement=0.3,
        corr_active_base_error=0.2,
        base_error_spread_c=1.0,
        base_abs_error_spread_c=0.8,
        improvement_spread_c=0.4,
        agreement_vs_other_std_diff=0.5,
        agreement_rows=6,
        full_history_rows=365,
    )

    assert strong > weak
    assert strong > 0.0
