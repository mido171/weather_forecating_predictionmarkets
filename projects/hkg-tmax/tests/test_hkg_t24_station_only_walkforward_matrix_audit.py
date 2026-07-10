from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_hkg_t24_station_only_walkforward_matrix_audit import (
    assert_no_forbidden_feature_columns,
    build_component_catalog,
    build_deployable_matrix,
    normalize_component_text,
    parse_component_text,
)


def test_parse_component_text_station_trajectory() -> None:
    parsed = parse_component_text(
        "450110-99999 air_temperature_c_latest_before_1500__current_minus_rolling_mean_14d"
    )

    assert parsed.source_family == "station_trajectory"
    assert parsed.station_id == "450110-99999"
    assert parsed.source_attribute == "air_temperature_c_latest_before_1500"
    assert parsed.raw_feature_name == "air_temperature_c_latest_before_1500__current_minus_rolling_mean_14d"


def test_parse_component_text_station_attribute() -> None:
    parsed = parse_component_text("592870-99999 air_temperature_c_latest_before_1500")

    assert parsed.source_family == "station_attribute"
    assert parsed.station_id == "592870-99999"
    assert parsed.source_attribute == "air_temperature_c_latest_before_1500"
    assert parsed.raw_feature_name == "air_temperature_c_latest_before_1500"


def test_parse_component_text_pair_spread() -> None:
    parsed = parse_component_text(
        "590960-99999 minus 596730-99999 sea_level_pressure_hpa_latest_before_1500"
    )

    assert parsed.source_family == "station_pair_spread"
    assert parsed.station_a == "590960-99999"
    assert parsed.station_b == "596730-99999"
    assert parsed.station_ids == "590960-99999,596730-99999"
    assert parsed.raw_feature_name == "sea_level_pressure_hpa_latest_before_1500"


def test_normalize_component_text_recovers_single_station_from_candidate_name() -> None:
    candidate = {
        "candidate_name": "450070-99999 air_temperature_c_latest_before_1500__current_minus_rolling_mean_14d"
    }

    out = normalize_component_text(
        "air_temperature_c_latest_before_1500__current_minus_rolling_mean_14d",
        candidate,
    )

    assert out == "450070-99999 air_temperature_c_latest_before_1500__current_minus_rolling_mean_14d"


def test_build_component_catalog_dedupes_repeated_components() -> None:
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "a",
                "candidate_type": "station_attribute",
                "candidate_name": "same component a",
                "deployable_feature_text": "592870-99999 air_temperature_c_latest_before_1500",
                "primary_score": 1.0,
                "official_error_score": 0.1,
                "audit_priority_score": 10.0,
            },
            {
                "candidate_id": "b",
                "candidate_type": "station_attribute",
                "candidate_name": "same component b",
                "deployable_feature_text": "592870-99999 air_temperature_c_latest_before_1500",
                "primary_score": 0.5,
                "official_error_score": 0.2,
                "audit_priority_score": 8.0,
            },
        ]
    )

    catalog, mapping = build_component_catalog(candidates)

    assert len(catalog) == 1
    assert set(mapping["candidate_id"]) == {"a", "b"}
    assert mapping[mapping["component_order"].eq(1)]["component_feature_id"].nunique() == 1
    assert len(mapping[mapping["component_order"].eq(0)]) == 2


def test_build_deployable_matrix_excludes_labels_and_adds_cutoff_rule() -> None:
    feature_frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2001-01-02", "2001-01-03"]),
            "target_tmax_c": [20.0, 21.0],
            "past_doy_count": [40, 40],
            "past_doy_mean_tmax_c": [19.0, 19.5],
            "target_anomaly_vs_past_doy_c": [1.0, 1.5],
            "station_feature": [5.0, 6.0],
        }
    )

    out = build_deployable_matrix(feature_frame)

    assert out.columns.tolist() == [
        "target_date",
        "station_feature",
        "source_local_date_rule",
        "source_cutoff_hkt",
    ]
    assert out["source_local_date_rule"].tolist() == ["target_date_minus_1", "target_date_minus_1"]
    assert out["source_cutoff_hkt"].tolist() == ["2001-01-01 15:00:00+08:00", "2001-01-02 15:00:00+08:00"]


def test_assert_no_forbidden_feature_columns_rejects_leakage_like_names() -> None:
    with pytest.raises(ValueError, match="Forbidden leakage-like feature columns"):
        assert_no_forbidden_feature_columns(["station_feature", "official_error_bucket"])
