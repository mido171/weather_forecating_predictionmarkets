from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_hkg_t24_beastmode_signal_discovery import (
    apply_half_life_bias,
    hkt_cutoff_utc_for_target_dates,
    require_no_confirmation_dates,
    rolling_inverse_mae_blend,
    sanitize_temperature_forecasts,
    select_latest_pre_cutoff_forecast,
)


def test_hkt_cutoff_utc_for_target_dates_uses_tminus1_1500_hkt() -> None:
    cutoff = hkt_cutoff_utc_for_target_dates(pd.Series([pd.Timestamp("2023-08-10")]))

    assert str(cutoff.iloc[0]) == "2023-08-09 07:00:00+00:00"


def test_select_latest_pre_cutoff_forecast_rejects_late_issue() -> None:
    frame = pd.DataFrame(
        {
            "forecast_date": ["2023-08-10", "2023-08-10", "2023-08-10"],
            "available_at_hkt": [
                "2023-08-09T06:00:00+08:00",
                "2023-08-09T14:59:00+08:00",
                "2023-08-09T15:01:00+08:00",
            ],
            "forecast_min_temperature_c": [27, 28, 29],
            "forecast_max_temperature_c": [32, 33, 40],
        }
    )

    selected = select_latest_pre_cutoff_forecast(
        frame,
        target_col="forecast_date",
        issue_col="available_at_hkt",
        max_col="forecast_max_temperature_c",
        min_col="forecast_min_temperature_c",
        source_name="unit",
    )

    assert len(selected) == 1
    assert float(selected.iloc[0]["forecast_max_c"]) == 33.0


def test_sanitize_temperature_forecasts_filters_bad_max_but_allows_missing_min() -> None:
    frame = pd.DataFrame(
        {
            "forecast_date": ["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
            "available_at_hkt": [
                "2023-01-01T12:00:00+08:00",
                "2023-01-02T12:00:00+08:00",
                "2023-01-03T12:00:00+08:00",
                "2023-01-04T12:00:00+08:00",
            ],
            "forecast_min_temperature_c": [None, 31, 20, 20],
            "forecast_max_temperature_c": [25, 30, None, 80],
        }
    )

    clean = sanitize_temperature_forecasts(
        frame,
        target_col="forecast_date",
        issue_col="available_at_hkt",
        max_col="forecast_max_temperature_c",
        min_col="forecast_min_temperature_c",
        source_name="unit",
    )

    assert clean["target_date"].dt.strftime("%Y-%m-%d").to_list() == ["2023-01-02"]
    assert pd.isna(clean.iloc[0]["forecast_min_c"])


def test_apply_half_life_bias_uses_prior_rows_only() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "forecast_max_c": [20.0, 20.0, 20.0],
            "target_tmax_c": [22.0, 22.0, 20.0],
        }
    )

    correction = apply_half_life_bias(frame, half_life_days=30, min_history=1)

    assert correction.iloc[0] == 0.0
    assert correction.iloc[1] == pytest.approx(2.0)
    assert correction.iloc[2] == pytest.approx(2.0)


def test_require_no_confirmation_dates_blocks_2024_labels() -> None:
    require_no_confirmation_dates([pd.Timestamp("2023-12-31")], context="unit")

    with pytest.raises(RuntimeError, match="confirmation dates"):
        require_no_confirmation_dates([pd.Timestamp("2024-01-01")], context="unit")


def test_rolling_inverse_mae_blend_uses_prior_window_only() -> None:
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "left": [10.0, 10.0, 10.0],
            "right": [20.0, 20.0, 20.0],
            "target_tmax_c": [10.0, 10.0, 20.0],
        }
    )

    blended = rolling_inverse_mae_blend(
        frame,
        left_col="left",
        right_col="right",
        window_days=180,
        min_history=1,
    )

    assert blended.iloc[0] == pytest.approx(15.0)
    assert blended.iloc[1] == pytest.approx(10.0)
    assert blended.iloc[2] == pytest.approx(10.0)
