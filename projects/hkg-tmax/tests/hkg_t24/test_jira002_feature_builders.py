from __future__ import annotations

import math
from datetime import date, timedelta

import pytest
from hkg_t24.features.calendar import calendar_feature_map
from hkg_t24.features.feature_dictionary import (
    ordered_feature_names,
    strict_feature_definitions,
    validate_feature_names,
)
from hkg_t24.features.nwp_daily import (
    build_gefs_ensemble_features,
    joule_m2_to_mj_m2,
    kelvin_to_c,
    meter_precip_to_mm,
    pa_to_hpa,
    threshold_feature_key,
)
from hkg_t24.features.official_anchor import (
    OfficialForecastRow,
    eligible_official_rows,
    official_feature_map,
)
from hkg_t24.features.official_text import psr_numeric_proxy
from hkg_t24.features.target_memory import (
    assert_target_year_index_matches_calendar,
    build_target_memory_features,
)


def _labels(count: int = 420) -> list[tuple[date, float]]:
    start = date(2020, 1, 1)
    return [(start + timedelta(days=offset), float(20 + (offset % 20))) for offset in range(count)]


def test_target_memory_full_formula_set_has_missing_indicators_and_no_lag1() -> None:
    features = build_target_memory_features(_labels())
    row = features[date(2021, 2, 23)]

    assert row["target__lag2_tmax_c"] == 37.0
    assert row["target__roll7_mean_lag2_c"] == pytest.approx(sum([31, 32, 33, 34, 35, 36, 37]) / 7)
    assert row["target__range7_lag2_c"] == 6.0
    assert row["target__lag2_tmax_c__is_missing"] == 0
    assert row["target__lag365_tmax_c__is_missing"] == 0
    assert "target__year_index__is_missing" not in row
    assert all("target__lag1_" not in name for name in row)


def test_target_memory_uses_pre_start_long_history_with_t2_boundary() -> None:
    labels = [(date(1999, 1, 1) + timedelta(days=offset), float(offset)) for offset in range(370)]
    features = build_target_memory_features(labels)
    row = features[date(2000, 1, 1)]

    assert row["target__lag365_tmax_c"] == 0.0
    assert row["target__lag2_tmax_c"] == 363.0
    assert row["target__lag365_tmax_c__is_missing"] == 0
    assert "target__lag1_tmax_c" not in row


def test_target_memory_selected_dates_still_use_pre_start_history() -> None:
    labels = [(date(1999, 1, 1) + timedelta(days=offset), float(offset)) for offset in range(370)]
    features = build_target_memory_features(labels, selected_dates={date(2000, 1, 1)})

    assert set(features) == {date(2000, 1, 1)}
    row = features[date(2000, 1, 1)]
    assert row["target__lag365_tmax_c"] == 0.0
    assert row["target__lag2_tmax_c"] == 363.0


def test_target_year_index_must_match_calendar_year_index() -> None:
    calendar = calendar_feature_map(date(2022, 6, 1))
    assert_target_year_index_matches_calendar({"target__year_index": 22}, calendar)
    with pytest.raises(ValueError, match="target__year_index"):
        assert_target_year_index_matches_calendar({"target__year_index": 21}, calendar)


def test_official_text_psr_mapping() -> None:
    assert psr_numeric_proxy("65%", None) == pytest.approx(0.65)
    assert psr_numeric_proxy("medium high", None) == pytest.approx(0.65)
    assert psr_numeric_proxy(None, "Fine and sunny") == pytest.approx(0.15)
    assert psr_numeric_proxy(None, "Thunderstorm and heavy rain") == pytest.approx(0.70)


def test_official_anchor_revision_features_use_latest_eligible_row() -> None:
    from datetime import UTC, datetime

    freeze = datetime(2023, 5, 31, 7, 0, tzinfo=UTC)
    features = official_feature_map(
        [
            OfficialForecastRow(
                issue_at_utc=datetime(2023, 5, 30, 7, 0, tzinfo=UTC),
                forecast_min_c=25.0,
                forecast_max_c=30.0,
                forecast_text="Fine and very hot with isolated showers",
            ),
            OfficialForecastRow(
                issue_at_utc=datetime(2023, 5, 31, 6, 0, tzinfo=UTC),
                forecast_min_c=26.0,
                forecast_max_c=32.0,
                forecast_text="Very hot with showers",
            ),
        ],
        operational_freeze_utc=freeze,
    )

    assert features["official__forecast_max_c"] == 32.0
    assert features["official__revision_count"] == 2
    assert features["official__revision_max_delta_c"] == 2.0
    assert features["official__text_very_hot_flag"] == 1
    assert features["official__text_showers_flag"] == 1


def test_official_anchor_rejects_post_freeze_rows_for_strict_h24n() -> None:
    from datetime import UTC, datetime

    freeze = datetime(2023, 5, 31, 7, 0, tzinfo=UTC)
    rows = [
        OfficialForecastRow(
            issue_at_utc=datetime(2023, 5, 31, 7, 1, tzinfo=UTC),
            forecast_min_c=26.0,
            forecast_max_c=32.0,
            forecast_text="Very hot",
        )
    ]

    assert eligible_official_rows(rows, operational_freeze_utc=freeze) == []
    features = official_feature_map(rows, operational_freeze_utc=freeze)
    assert features["official__forecast_max_c"] is None


def test_official_anchor_accepts_pre_freeze_tmax_only_rows() -> None:
    from datetime import UTC, datetime

    freeze = datetime(2023, 5, 31, 7, 0, tzinfo=UTC)
    features = official_feature_map(
        [
            OfficialForecastRow(
                issue_at_utc=datetime(2023, 5, 31, 6, 59, tzinfo=UTC),
                forecast_min_c=None,
                forecast_max_c=32.0,
                forecast_text="Very hot",
                row_quality_status="usable_local_tmax_only",
            )
        ],
        operational_freeze_utc=freeze,
    )

    assert features["official__forecast_max_c"] == 32.0
    assert features["official__forecast_min_c"] is None
    assert features["official__forecast_range_c"] is None


def test_nwp_unit_conversions_and_threshold_keys() -> None:
    assert kelvin_to_c(303.15) == pytest.approx(30.0)
    assert pa_to_hpa(101325.0) == pytest.approx(1013.25)
    assert meter_precip_to_mm(0.012) == pytest.approx(12.0)
    assert joule_m2_to_mj_m2(2_500_000.0) == pytest.approx(2.5)
    assert threshold_feature_key(30.0) == "prob_ge_30_0"
    assert threshold_feature_key(40.0) == "prob_ge_40_0"
    gefs = build_gefs_ensemble_features([29.0, 30.0, 31.0, 32.0, 40.0])
    assert gefs["gefsens__center__prob_ge_30_0"] == pytest.approx(0.8)
    assert gefs["gefsens__center__tmax_spread_p90_p10_c"] is not None


def test_strict_feature_dictionary_rejects_forbidden_prefixes_and_orders_missing_indicators() -> None:
    strict_names = [definition.feature_name for definition in strict_feature_definitions()]
    assert all(
        name.startswith(
            ("calendar__", "official__", "target__", "online__", "gfs__", "gefsmean__", "gefsens__", "router__")
        )
        for name in strict_names
    )
    with pytest.raises(ValueError, match="Strict feature names"):
        validate_feature_names("strict", ("station__network__tmax_mean_c",))
    with pytest.raises(ValueError, match="Forbidden finalized target-memory"):
        validate_feature_names("strict", ("target__lag1_tmax_c",))
    assert ordered_feature_names(["b__x__is_missing", "a__x", "b__x"]) == (
        "a__x",
        "b__x",
        "b__x__is_missing",
    )


def test_calendar_cyclical_features_are_stable() -> None:
    features = calendar_feature_map(date(2024, 3, 1))
    assert features["calendar__is_mam"] == 1
    assert features["calendar__year_index"] == 24
    radius = math.hypot(features["calendar__month_sin1"], features["calendar__month_cos1"])
    assert radius == pytest.approx(1.0)
