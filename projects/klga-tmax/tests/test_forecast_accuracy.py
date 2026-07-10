from __future__ import annotations

from datetime import date, datetime, timezone
from uuid import uuid4

import pytest

from klga_tmax.evaluation.accuracy import (
    ForecastEvaluationError,
    _score_row,
    aggregate_scores,
)
from klga_tmax.models.pmf import gaussian_pmf


def _base_row(observed: int = 72, expected: float = 72.4) -> dict[str, object]:
    return {
        "target_date": date(2026, 6, 20),
        "cutoff_id": "T_MINUS_1_STOCKHOLM_1915",
        "final_prediction_id": uuid4(),
        "calibrated_prediction_id": None,
        "settled_wu_tmax_f": observed,
        "expected_tmax_f": expected,
        "median_tmax_f": 72,
        "mode_tmax_f": 72,
        "prediction_interval_low_f": 69,
        "prediction_interval_high_f": 75,
        "pmf_json": gaussian_pmf(expected, 2.5),
        "label_source_record_id": uuid4(),
        "label_revision_number": 1,
        "label_available_at_utc": datetime(2026, 6, 21, 4, 10, tzinfo=timezone.utc),
    }


def test_daily_forecast_score_uses_settled_wu_label() -> None:
    score = _score_row(_base_row())
    assert score["settled_wu_tmax_f"] == 72
    assert score["absolute_error_f"] == pytest.approx(0.4)
    assert score["signed_error_f"] == pytest.approx(0.4)
    assert score["within_1f"] is True
    assert score["within_2f"] is True
    assert score["prediction_interval_hit"] is True
    assert score["leakage_checked"] is True
    assert score["diagnostics_json"]["label_source"] == "public.wunderground_daily_tmax"


def test_aggregate_forecast_accuracy_metrics() -> None:
    scores = [
        _score_row(_base_row(observed=72, expected=72.4)),
        _score_row(_base_row(observed=75, expected=73.0)),
    ]
    metrics = aggregate_scores(scores)
    assert metrics["row_count"] == 2
    assert metrics["mae_f"] == pytest.approx(1.2)
    assert metrics["bias_f"] == pytest.approx(-0.8)
    assert metrics["within_1f_hit_rate"] == pytest.approx(0.5)
    assert metrics["within_2f_hit_rate"] == pytest.approx(1.0)


def test_settled_wu_label_must_be_inside_temp_grid() -> None:
    with pytest.raises(ForecastEvaluationError, match="outside TEMP_GRID_F"):
        _score_row(_base_row(observed=116, expected=115.0))
