from __future__ import annotations

import pandas as pd

from scripts.run_hkg_t24_r14_r17_robust_long_history import (
    ANALYSIS_END,
    MIN_HISTORY_YEARS,
    build_isd_features,
    clean_numeric_series,
    experiment_frame,
    long_report,
    make_experiment_specs,
    robust_fold_definitions,
)


def test_clean_numeric_series_removes_igra_sentinel_values() -> None:
    cleaned = clean_numeric_series(pd.Series([27.0, -888.8, -8888.0, 99.0]), lower=-90.0, upper=60.0)

    assert cleaned.iloc[0] == 27.0
    assert pd.isna(cleaned.iloc[1])
    assert pd.isna(cleaned.iloc[2])
    assert pd.isna(cleaned.iloc[3])


def test_robust_folds_are_four_or_five_years_and_stop_before_validation_2024() -> None:
    folds = robust_fold_definitions()

    assert folds[0][0] == "fold_1965_1969"
    assert folds[-1][2] == ANALYSIS_END
    for _, test_start, test_end, train_end in folds:
        assert train_end < test_start
        assert test_end <= ANALYSIS_END
        days = (test_end.date() - test_start.date()).days + 1
        assert days >= int(4 * 365.25)


def test_build_isd_features_uses_cutoff_fields_not_full_day_minmax() -> None:
    source = pd.DataFrame(
        {
            "station_id": ["450050-99999", "450110-99999"],
            "local_date": ["1980-01-01", "1980-01-01"],
            "obs_count": [3, 4],
            "latest_before_1500_hkt": ["1980-01-01T14:00:00+08:00", "1980-01-01T15:00:00+08:00"],
            "air_temperature_c_latest_before_1500": [20.0, 22.0],
            "dew_point_c_latest_before_1500": [15.0, 16.0],
            "sea_level_pressure_hpa_latest_before_1500": [1012.0, 1010.0],
            "wind_direction_deg_latest_before_1500": [90.0, 180.0],
            "wind_speed_mps_latest_before_1500": [3.0, 5.0],
            "daily_air_temperature_min_c": [-99.0, -99.0],
            "daily_air_temperature_max_c": [99.0, 99.0],
            "availability_tier": ["PROXY_WITH_LIMITATIONS", "PROXY_WITH_LIMITATIONS"],
            "operational_input_allowed": [False, False],
        }
    )

    features = build_isd_features(source, station_count=1)

    assert features.loc[0, "target_date"] == pd.Timestamp("1980-01-02")
    assert features.loc[0, "isd_air_temp_mean_c"] == 21.0
    assert "daily_air_temperature_max_c" not in features.columns
    assert "daily_air_temperature_min_c" not in features.columns


def test_r16_required_prefix_is_tuple_not_string() -> None:
    features = pd.DataFrame(
        {
            "doy_sin": [0.0],
            "doy_cos": [1.0],
            "target_tminus2_tmax_c": [25.0],
            "igra_temperature_c_850hpa": [18.0],
            "isd_air_temp_mean_c": [22.0],
        }
    )
    specs = {spec.research_id: spec for spec in make_experiment_specs(features)}

    assert specs["HKG-T24-R16"].required_non_null_prefixes == ("isd_",)


def test_multi_source_experiment_frame_requires_every_prefix_group() -> None:
    dates = pd.date_range("1950-01-01", "1990-12-31", freq="D")
    features = pd.DataFrame(
        {
            "target_date": dates,
            "target_tmax_c": 25.0,
            "doy_sin": 0.0,
            "doy_cos": 1.0,
            "target_tminus2_tmax_c": 25.0,
            "igra_temperature_c_850hpa": 18.0,
            "isd_air_temp_mean_c": 22.0,
        }
    )
    features.loc[0, "igra_temperature_c_850hpa"] = None
    features.loc[1, "isd_air_temp_mean_c"] = None
    spec = {item.research_id: item for item in make_experiment_specs(features)}["HKG-T24-R15"]

    filtered = experiment_frame(features, spec)

    assert filtered["target_date"].min() == pd.Timestamp("1950-01-03")


def test_robust_long_report_exceeds_required_narrative_length() -> None:
    features = pd.DataFrame(
        {
            "doy_sin": [0.0],
            "doy_cos": [1.0],
            "target_tminus2_tmax_c": [25.0],
            "igra_temperature_c_850hpa": [18.0],
            "isd_air_temp_mean_c": [22.0],
        }
    )
    spec = make_experiment_specs(features)[0]
    payload = {
        "feature_min": "1949-06-04",
        "feature_max": "2023-12-31",
        "input_history_years": 74.6,
        "prediction_min": "1965-01-01",
        "prediction_max": "2023-12-31",
        "feature_rows": 27000,
        "feature_columns": 120,
        "oof_feasibility": {
            "status": "PASS",
            "reason": "synthetic robust OOF span passes",
        },
        "baseline": {
            "model_id": "r14_lag_calendar_baseline",
            "n": 1000,
            "mae": 1.4,
            "rmse": 1.8,
            "bias": 0.0,
            "crps_normal": 1.0,
        },
        "champion": {
            "model_id": "r14_upper_air_core",
            "n": 1000,
            "mae": 1.3,
            "rmse": 1.7,
            "bias": -0.01,
            "crps_normal": 0.95,
            "mae_improvement_vs_baseline": 0.1,
        },
    }
    scoreboard = pd.DataFrame(
        [
            {
                "model_id": "r14_lag_calendar_baseline",
                "n": 1000,
                "first_date": "1965-01-01",
                "last_date": "2023-12-31",
                "mae": 1.4,
                "rmse": 1.8,
                "bias": 0.0,
                "crps_normal": 1.0,
                "coverage_80": 0.8,
                "coverage_90": 0.9,
            }
        ]
    )
    fold_scores = pd.DataFrame(
        [
            {
                "fold_id": "fold_1965_1969",
                "model_id": "r14_upper_air_core",
                "n": 1000,
                "mae": 1.3,
                "baseline_mae": 1.4,
                "mae_improvement_vs_baseline": 0.1,
                "crps_improvement_vs_baseline": 0.05,
            }
        ]
    )

    report = long_report(spec, payload, scoreboard, fold_scores)

    assert len(report) >= 7500
    assert str(int(MIN_HISTORY_YEARS)) in report
    assert "Validation 2024 is not read" in report
