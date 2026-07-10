"""Ablation-matrix helpers for Jira003 system reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from hkg_t24.models.final_formula import SystemPrediction
from hkg_t24.validation.metrics import ForecastMetricSummary, forecast_metrics


@dataclass(frozen=True)
class AblationRow:
    candidate_id: str
    baseline_id: str
    metrics: ForecastMetricSummary
    baseline_metrics: ForecastMetricSummary
    delta_mae_c: float


def final_vs_pre_distribution_ablation(
    predictions: Sequence[SystemPrediction],
    labels_by_date: Mapping[date, float],
) -> AblationRow | None:
    final_pairs: list[tuple[float, float]] = []
    pre_pairs: list[tuple[float, float]] = []
    for prediction in predictions:
        label = labels_by_date.get(prediction.target_date_hkt)
        if label is None or prediction.final_point_tmax_c is None or prediction.final_pre_distribution_c is None:
            continue
        final_pairs.append((float(prediction.final_point_tmax_c), label))
        pre_pairs.append((float(prediction.final_pre_distribution_c), label))
    if not final_pairs or not pre_pairs:
        return None
    final_metrics = forecast_metrics(final_pairs)
    pre_metrics = forecast_metrics(pre_pairs)
    return AblationRow(
        candidate_id="final_distribution_p50",
        baseline_id="pre_distribution_point",
        metrics=final_metrics,
        baseline_metrics=pre_metrics,
        delta_mae_c=final_metrics.mae_c - pre_metrics.mae_c,
    )
