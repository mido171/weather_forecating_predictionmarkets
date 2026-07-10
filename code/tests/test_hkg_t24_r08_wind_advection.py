from __future__ import annotations

import pytest

from scripts.run_hkg_t24_r08_wind_advection import (
    compass_to_degrees,
    long_report,
    wind_uv_from_direction,
    zip_entry_in_sampling_window,
)


def test_compass_to_degrees_parses_cardinal_and_variable_values() -> None:
    assert compass_to_degrees("East") == 90.0
    assert compass_to_degrees("Southwest") == 225.0
    assert compass_to_degrees("Variable") is None


def test_wind_uv_from_direction_uses_meteorological_from_direction() -> None:
    u_east, v_east = wind_uv_from_direction(10.0, 90.0)
    u_north, v_north = wind_uv_from_direction(10.0, 0.0)

    assert u_east == pytest.approx(-10.0)
    assert v_east == pytest.approx(0.0, abs=1e-9)
    assert u_north == pytest.approx(0.0, abs=1e-9)
    assert v_north == pytest.approx(-10.0)


def test_wind_zip_entry_filter_keeps_operational_snapshot_windows() -> None:
    assert zip_entry_in_sampling_window("20230714-1447-latest_10min_wind.csv")
    assert zip_entry_in_sampling_window("20230714-0249-latest_10min_wind.csv")
    assert not zip_entry_in_sampling_window("20230714-1517-latest_10min_wind.csv")


def test_r08_long_report_exceeds_required_experiment_narrative_length() -> None:
    report = long_report(
        {
            "champion": {
                "model_id": "r08_baseline_temp_calendar",
                "n": 911,
                "mae": 1.4723,
                "rmse": 1.8861,
                "bias": 0.0298,
                "crps_normal": 1.0512,
            },
            "oof_feasibility": {
                "status": "BLOCKED",
                "reason": "synthetic four-year OOF blocker",
            },
            "feature_min": "2020-07-02",
            "feature_max": "2023-12-31",
            "prediction_min": "2021-07-01",
            "prediction_max": "2023-12-31",
            "wind_observation_min": "2021-12-29 11:40:00+08:00",
            "wind_observation_max": "2026-06-18 14:50:00+08:00",
        }
    )

    assert len(report) >= 7500
    assert "month-permuted" in report
