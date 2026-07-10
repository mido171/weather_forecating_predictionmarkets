from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from klga_tmax.evaluation.gribstream_bias_backtest import (
    ForecastRow,
    half_life_weighted_bias,
    score_models,
    select_raw_tmax,
    summarize_models,
    ModelCoverage,
)


def _forecast_row(
    target_date: date,
    raw: float,
    actual: float,
    *,
    label_available_at_utc: datetime | None = None,
) -> ForecastRow:
    return ForecastRow(
        model_id="gfs",
        target_date=target_date,
        cutoff_utc=datetime(2024, 1, target_date.day, 12, 45, tzinfo=timezone.utc),
        raw_tmax_f=raw,
        settled_wu_tmax_f=actual,
        label_available_at_utc=label_available_at_utc
        or datetime(2024, 1, target_date.day, 5, 0, tzinfo=timezone.utc),
        label_source_record_id="label",
        label_revision_number=1,
        max_source_run_time_utc=datetime(2024, 1, target_date.day, 6, 0, tzinfo=timezone.utc),
        max_source_available_at_utc=datetime(2024, 1, target_date.day, 7, 0, tzinfo=timezone.utc),
        source_feature_names=("grib_gfs_klga_core_member_0_tmax_proxy_f",),
        raw_method="direct",
    )


def test_select_raw_tmax_prefers_direct_tmax_proxy() -> None:
    selected = select_raw_tmax(
        model_id="gfs",
        features={
            "grib_gfs_klga_core_member_0_tmax_proxy_f": 82.0,
            "grib_gfs_klga_core_valid_18z_tmp_2m_f": 78.0,
        },
    )
    assert selected == (
        82.0,
        "direct:grib_gfs_klga_core_member_0_tmax_proxy_f",
        ("grib_gfs_klga_core_member_0_tmax_proxy_f",),
    )


def test_select_raw_tmax_uses_synoptic_max_for_two_time_models() -> None:
    selected = select_raw_tmax(
        model_id="ifsoper",
        features={
            "grib_ifsoper_klga_core_valid_18z_tmp_2m_f": 79.0,
            "grib_ifsoper_klga_core_valid_00z_nextday_tmp_2m_f": 76.5,
        },
    )
    assert selected is not None
    value, method, feature_names = selected
    assert value == pytest.approx(79.0)
    assert method == "synoptic:max_18z_00z_temperature_proxy"
    assert feature_names == (
        "grib_ifsoper_klga_core_valid_18z_tmp_2m_f",
        "grib_ifsoper_klga_core_valid_00z_nextday_tmp_2m_f",
    )


def test_half_life_weighted_bias_uses_raw_minus_actual_errors() -> None:
    prior = [
        _forecast_row(date(2024, 1, 5), raw=80.0, actual=78.0),
        _forecast_row(date(2024, 1, 9), raw=70.0, actual=72.0),
    ]
    bias = half_life_weighted_bias(
        current_target_date=date(2024, 1, 10),
        prior_rows=prior,
        half_life_days=5.0,
    )
    old_weight = 0.5 ** (5 / 5)
    new_weight = 0.5 ** (1 / 5)
    expected = (old_weight * 2.0 + new_weight * -2.0) / (old_weight + new_weight)
    assert bias == pytest.approx(expected)


def test_score_models_excludes_t_and_t_minus_1_labels_and_future_unavailable_labels() -> None:
    rows = [
        _forecast_row(date(2024, 1, 7), raw=70.0, actual=68.0),
        _forecast_row(date(2024, 1, 8), raw=71.0, actual=70.0),
        _forecast_row(
            date(2024, 1, 9),
            raw=72.0,
            actual=69.0,
            label_available_at_utc=datetime(2024, 1, 10, 13, 0, tzinfo=timezone.utc),
        ),
        _forecast_row(date(2024, 1, 10), raw=73.0, actual=72.0),
    ]

    scored = score_models(
        rows,
        lookback_days=45,
        half_life_days=15.0,
        label_lag_days=2,
    )["gfs"]

    row_for_jan_10 = next(row for row in scored if row.target_date == date(2024, 1, 10))
    assert row_for_jan_10.prior_error_count == 2
    assert row_for_jan_10.oldest_prior_target_date == date(2024, 1, 7)
    assert row_for_jan_10.newest_prior_target_date == date(2024, 1, 8)


def test_summarize_models_compares_baseline_and_corrected_same_rows() -> None:
    rows = [
        _forecast_row(date(2024, 1, 1), raw=80.0, actual=78.0),
        _forecast_row(date(2024, 1, 3), raw=79.0, actual=78.0),
        _forecast_row(date(2024, 1, 5), raw=78.0, actual=78.0),
    ]
    scored = score_models(rows, lookback_days=45, half_life_days=15.0, label_lag_days=2)
    coverage = [
        ModelCoverage(
            model_id="gfs",
            first_target_date=date(2024, 1, 1),
            last_target_date=date(2024, 1, 5),
            target_days=3,
            buffered_target_days=3,
            feature_rows=3,
            buffered_feature_rows=3,
            included_by_coverage=True,
        )
    ]
    summary = summarize_models(coverage, scored, min_test_days=1)[0]
    assert summary.scored_days == 2
    assert summary.baseline_mae_f is not None
    assert summary.corrected_mae_f is not None
