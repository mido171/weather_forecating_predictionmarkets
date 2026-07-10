from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_r09_station_temperature_gradient import (
    aggregate_temperature,
    long_report,
    station_slug,
    zip_entry_in_sampling_window,
)


def test_station_slug_is_stable_for_station_offset_columns() -> None:
    assert station_slug("Tate's Cairn") == "tate_s_cairn"
    assert station_slug("Tsuen Wan Shing Mun Valley") == "tsuen_wan_shing_mun_valley"


def test_temperature_zip_entry_filter_keeps_cutoff_relevant_snapshots() -> None:
    assert zip_entry_in_sampling_window("20230714-1447-latest_1min_temperature.csv")
    assert zip_entry_in_sampling_window("20230714-0248-latest_1min_temperature.csv")
    assert not zip_entry_in_sampling_window("20230714-1517-latest_1min_temperature.csv")


def test_aggregate_temperature_computes_group_contrasts() -> None:
    rows = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2023-07-15"] * 5),
            "station": ["HK Observatory", "Chek Lap Kok", "Ta Kwu Ling", "Sha Tin", "Cheung Chau"],
            "temperature_c": [30.0, 29.0, 33.0, 32.0, 28.0],
        }
    )

    out = aggregate_temperature(rows, "temp_network")

    assert out.loc[0, "temp_network_station_count"] == 5
    assert out.loc[0, "temp_network_spread_c"] == 5.0
    assert out.loc[0, "temp_network_hko_minus_median_c"] == 0.0
    assert out.loc[0, "temp_network_inland_minus_coastal_c"] > 0


def test_r09_long_report_exceeds_required_experiment_narrative_length() -> None:
    report = long_report(
        {
            "champion": {
                "model_id": "r09_baseline_temp_calendar",
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
            "temperature_observation_min": "2020-06-30 11:40:00+08:00",
            "temperature_observation_max": "2026-06-18 14:50:00+08:00",
        }
    )

    assert len(report) >= 7500
    assert "plane-fit" in report
