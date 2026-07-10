from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_r11_dynamic_upwind_station_selection import (
    hko_registry_has_geometry,
)
from scripts.run_hkg_t24_r11_dynamic_upwind_station_selection import (
    long_report as r11_long_report,
)
from scripts.run_hkg_t24_r12_solar_radiation import (
    active_cols,
    build_cutoff_solar_features,
)
from scripts.run_hkg_t24_r12_solar_radiation import (
    long_report as r12_long_report,
)
from scripts.run_hkg_t24_r13_to_r30_precondition_gates import (
    GATE_SPECS,
    readiness_rows,
)
from scripts.run_hkg_t24_r13_to_r30_precondition_gates import (
    long_report as gate_long_report,
)


def test_r11_hko_geometry_requires_coordinate_columns() -> None:
    no_geometry = pd.DataFrame({"network": ["HKO"], "station_name": ["HK Observatory"]})
    assert not hko_registry_has_geometry(no_geometry)

    with_geometry = pd.DataFrame(
        {
            "network": ["HKO"],
            "station_name": ["HK Observatory"],
            "latitude": [22.3],
            "longitude": [114.17],
            "elevation_m": [32.0],
        }
    )
    assert hko_registry_has_geometry(with_geometry)


def test_r12_cutoff_solar_features_respect_available_at() -> None:
    observations = pd.DataFrame(
        {
            "family": ["latest_1min_solar", "latest_1min_solar"],
            "local_date": pd.to_datetime(["2023-07-14", "2023-07-14"]),
            "available_at_hkt": pd.to_datetime(["2023-07-14 15:00", "2023-07-14 15:10"]).tz_localize(
                "Asia/Hong_Kong"
            ),
            "observed_at_hkt": pd.to_datetime(["2023-07-14 14:40", "2023-07-14 14:50"]).tz_localize(
                "Asia/Hong_Kong"
            ),
            "variable": ["global_solar_wm2", "global_solar_wm2"],
            "value": [500.0, 900.0],
        }
    )

    features = build_cutoff_solar_features(observations, pd.Series(pd.to_datetime(["2023-07-15"])))

    assert features.loc[0, "r12_global_solar_wm2_count"] == 1
    assert features.loc[0, "r12_global_solar_wm2_last"] == 500.0


def test_r12_active_cols_require_minimum_training_support() -> None:
    train = pd.DataFrame(
        {
            "well_supported": list(range(35)),
            "too_sparse": [1.0] * 10 + [None] * 25,
            "constant": [3.0] * 35,
        }
    )

    assert active_cols(train, ["well_supported", "too_sparse", "constant"]) == ["well_supported"]


def test_r11_and_r12_long_reports_exceed_required_narrative_length() -> None:
    readiness = pd.DataFrame(
        {
            "requirement": ["canonical HKO station latitude/longitude/elevation for all dynamic candidates"],
            "status": ["blocked"],
            "evidence": ["synthetic evidence"],
            "disposition": ["do not fake geometry"],
        }
    )
    r11_report = r11_long_report(
        {
            "sources": {
                "r08_wind_sampled": {
                    "rows": 1,
                    "station_count": 1,
                    "first_time": "2021-12-29",
                    "last_time": "2023-12-31",
                },
                "r09_temperature_sampled": {
                    "rows": 1,
                    "station_count": 1,
                    "first_time": "2020-06-30",
                    "last_time": "2023-12-31",
                },
            }
        },
        readiness,
    )
    r12_report = r12_long_report(
        {
            "champion": {
                "model_id": "r12_baseline_temp_calendar",
                "n": 911,
                "mae": 1.47,
                "rmse": 1.88,
                "bias": 0.03,
                "crps_normal": 1.05,
            },
            "baseline": {
                "n": 911,
                "mae": 1.47,
                "crps_normal": 1.05,
            },
            "oof_feasibility": {
                "status": "BLOCKED",
                "reason": "synthetic four-year OOF blocker",
            },
            "feature_min": "2020-07-02",
            "feature_max": "2023-12-31",
            "prediction_min": "2021-07-01",
            "prediction_max": "2023-12-31",
            "solar_min": "2021-06-29",
            "solar_max": "2023-12-31",
        }
    )

    assert len(r11_report) >= 7500
    assert len(r12_report) >= 7500


def test_r13_to_r30_gate_specs_cover_remaining_research_ids_and_narratives() -> None:
    assert [spec.research_id for spec in GATE_SPECS] == [f"HKG-T24-R{number:02d}" for number in range(13, 31)]

    for spec in GATE_SPECS:
        report = gate_long_report(spec, readiness_rows(spec))
        assert len(report) >= 7500
        assert "validation" in report.lower()
        assert "locked-test" in report.lower()
