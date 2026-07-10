"""Slice score helpers for Jira003 system validation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date

from hkg_t24.models.final_formula import SystemPrediction
from hkg_t24.validation.metrics import ForecastMetricSummary, forecast_metrics


def monthly_system_metrics(
    predictions: Sequence[SystemPrediction],
    labels_by_date: Mapping[date, float],
) -> dict[int, ForecastMetricSummary]:
    grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for prediction in predictions:
        if prediction.final_point_tmax_c is None or prediction.target_date_hkt not in labels_by_date:
            continue
        grouped[prediction.target_date_hkt.month].append(
            (float(prediction.final_point_tmax_c), labels_by_date[prediction.target_date_hkt])
        )
    return {month: forecast_metrics(pairs) for month, pairs in grouped.items() if pairs}
