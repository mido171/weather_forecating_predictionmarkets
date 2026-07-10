from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.run_hkg_t24_r05_thermal_memory import build_memory_features, feature_sets, long_report


def _write_minimal_r04_feature_matrix(data_root: Path) -> None:
    source = data_root / "gold" / "hkg_t24" / "r04_thermal_trajectory" / "r04_feature_matrix.parquet"
    source.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(10):
        rows.append(
            {
                "target_date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=idx),
                "target_tmax_c": 24.0 + idx,
                "doy_sin": 0.1,
                "doy_cos": 0.9,
                "day_length_hours": 11.0,
                "noon_solar_elevation_deg": 45.0,
                "hko_latest_temp_c": 20.0 + idx,
                "hko_temp_range_so_far_c": 3.0 + idx / 10,
                "hko_temp_std_so_far_c": 0.5 + idx / 100,
                "hko_temp_change_180m_to_latest_c": 0.2,
                "hko_temp_change_360m_to_latest_c": 0.4,
                "hko_trailing_nonwarming_minutes": 30.0,
            }
        )
    pd.DataFrame(rows).to_parquet(source, index=False)


def test_build_memory_features_uses_backward_lags_and_no_future_dates(tmp_path: Path) -> None:
    _write_minimal_r04_feature_matrix(tmp_path)

    features, output_path = build_memory_features(tmp_path)

    assert output_path.exists()
    assert str(features["target_date"].min().date()) == "2023-01-07"
    assert str(features["target_date"].max().date()) == "2023-01-10"
    first = features.iloc[0]
    assert first["lag1_cutoff_temp_c"] == 26.0
    assert first["lag7_cutoff_temp_c"] == 20.0
    assert (features["target_date"] < pd.Timestamp("2024-01-01")).all()


def test_feature_sets_exclude_target_columns(tmp_path: Path) -> None:
    _write_minimal_r04_feature_matrix(tmp_path)
    features, _ = build_memory_features(tmp_path)

    sets = feature_sets(features)

    assert "target_tmax_c" not in sets["r05_memory_lags_1_7"]
    assert "target_date" not in sets["r05_memory_lags_1_7"]
    assert "lag7_cutoff_temp_c" in sets["r05_memory_lags_1_7"]


def test_long_report_exceeds_required_experiment_narrative_length() -> None:
    report = long_report(
        {
            "champion": {
                "model_id": "r05_baseline_lag1_cutoff_temp_calendar",
                "n": 911,
                "mae": 1.5373,
                "rmse": 1.9642,
                "bias": 0.2004,
                "crps_normal": 1.0962,
            },
            "oof_feasibility": {
                "status": "BLOCKED",
                "reason": "synthetic strict four-year OOF blocker",
            },
            "feature_min": "2020-07-08",
            "feature_max": "2023-12-31",
            "prediction_min": "2021-07-01",
            "prediction_max": "2023-12-31",
        }
    )

    assert len(report) >= 7500
    assert "2020-07-08" in report
    assert "2023-12-31" in report
