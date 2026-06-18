import math

import pytest

from hkg_tmax.metrics import (
    MetricError,
    bias,
    crps_ensemble,
    mae,
    multiclass_brier,
    multiclass_log_loss,
    rmse,
)


def test_point_metrics() -> None:
    actual = [30.0, 32.0]
    predicted = [31.0, 31.0]
    assert bias(actual, predicted) == 0.0
    assert mae(actual, predicted) == 1.0
    assert rmse(actual, predicted) == 1.0


def test_perfect_probabilities() -> None:
    targets = [0, 1]
    probabilities = [[1.0, 0.0], [0.0, 1.0]]
    assert multiclass_log_loss(targets, probabilities) == 0.0
    assert multiclass_brier(targets, probabilities) == 0.0


def test_invalid_probability_sum_rejected() -> None:
    with pytest.raises(MetricError):
        multiclass_log_loss([0], [[0.6, 0.5]])


def test_crps_perfect_ensemble() -> None:
    assert crps_ensemble(31.0, [31.0, 31.0]) == 0.0
    assert math.isclose(crps_ensemble(0.0, [-1.0, 1.0]), 0.5)
