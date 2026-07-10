"""Expected absolute-error models for Jira003 expert routers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from statistics import median

from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore[import-untyped]

from hkg_t24.features.matrix_builder import FeatureMatrixRow
from hkg_t24.models.experts import ExpertPrediction
from hkg_t24.validation.metrics import clip

MIN_EXPECTED_ERROR_C = 0.20
MAX_EXPECTED_ERROR_C = 3.00


@dataclass(frozen=True)
class ExpectedErrorInput:
    expert_id: str
    target_date_hkt: date
    prediction_tmax_c: float
    label_tmax_c: float
    context: tuple[float, ...]


class ExpertExpectedErrorModel:
    """Per-expert expected-error model with a deterministic median fallback."""

    def __init__(self, expert_id: str, *, feature_names: Sequence[str]) -> None:
        self.expert_id = expert_id
        self.feature_names = tuple(feature_names)
        self._fallback_error_c = 1.0
        self._model: HistGradientBoostingRegressor | None = None

    @property
    def model_available(self) -> bool:
        return self._model is not None

    @property
    def fallback_error_c(self) -> float:
        return self._fallback_error_c

    def fit(self, rows: Sequence[ExpectedErrorInput]) -> None:
        if not rows:
            self._fallback_error_c = 1.0
            self._model = None
            return
        errors = [abs(row.label_tmax_c - row.prediction_tmax_c) for row in rows]
        self._fallback_error_c = clip(float(median(errors)), MIN_EXPECTED_ERROR_C, MAX_EXPECTED_ERROR_C)
        if len(rows) < 30:
            self._model = None
            return
        x = [list(row.context) for row in rows]
        y = errors
        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=100,
            max_leaf_nodes=7,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=20260626,
        )
        model.fit(x, y)
        self._model = model

    def predict(self, context: Sequence[float]) -> float:
        if self._model is None:
            return self._fallback_error_c
        raw = float(self._model.predict([list(context)])[0])
        return clip(raw, MIN_EXPECTED_ERROR_C, MAX_EXPECTED_ERROR_C)


def context_feature_names(rows: Sequence[FeatureMatrixRow]) -> tuple[str, ...]:
    """Pick numeric cutoff-safe context features available in the supplied matrix rows."""
    names = sorted(
        {
            name
            for row in rows
            for name, value in row.features.items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        }
    )
    return tuple(names)


def context_vector(row: FeatureMatrixRow, feature_names: Sequence[str]) -> tuple[float, ...]:
    values: list[float] = []
    for feature_name in feature_names:
        raw = row.features.get(feature_name)
        values.append(float(raw) if isinstance(raw, int | float) and not isinstance(raw, bool) else 0.0)
    return tuple(values)


def fit_expected_error_model(
    *,
    expert_id: str,
    feature_rows_by_date: Mapping[date, FeatureMatrixRow],
    predictions: Sequence[ExpertPrediction],
    labels_by_date: Mapping[date, float],
    feature_names: Sequence[str],
    cutoff_before: date | None,
) -> ExpertExpectedErrorModel:
    """Fit an expert error model using only rows before the supplied validation cutoff."""
    training_rows: list[ExpectedErrorInput] = []
    for prediction in predictions:
        if prediction.expert_id != expert_id or prediction.prediction_tmax_c is None:
            continue
        if prediction.target_date_hkt not in labels_by_date:
            continue
        if cutoff_before is not None and prediction.target_date_hkt >= cutoff_before:
            continue
        feature_row = feature_rows_by_date.get(prediction.target_date_hkt)
        if feature_row is None:
            continue
        training_rows.append(
            ExpectedErrorInput(
                expert_id=expert_id,
                target_date_hkt=prediction.target_date_hkt,
                prediction_tmax_c=float(prediction.prediction_tmax_c),
                label_tmax_c=float(labels_by_date[prediction.target_date_hkt]),
                context=context_vector(feature_row, feature_names),
            )
        )
    model = ExpertExpectedErrorModel(expert_id, feature_names=feature_names)
    model.fit(training_rows)
    return model
