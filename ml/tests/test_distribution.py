"""Tests for distribution helpers."""

from __future__ import annotations

import numpy as np

from weather_ml import distribution


def test_normal_pmf_sums_to_one() -> None:
    pmf = distribution.normal_integer_pmf(
        mu=70.0, sigma=5.0, support_min=-30, support_max=130
    )
    assert np.isclose(pmf.sum(), 1.0, atol=1e-6)


def test_bins_from_pmf() -> None:
    pmf = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
    bins = distribution.bins_from_pmf(
        pmf,
        support_min=0,
        bin_specs=[
            {"name": "lt_2", "type": "threshold", "lt": 2},
            {"name": "ge_3", "type": "threshold", "ge": 3},
            {"name": "range_1_3", "type": "range", "start": 1, "end": 3},
        ],
    )
    assert np.isclose(bins["lt_2"], 0.3)
    assert np.isclose(bins["ge_3"], 0.4)
    assert np.isclose(bins["range_1_3"], 0.7)
