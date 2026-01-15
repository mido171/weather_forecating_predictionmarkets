"""Distribution helpers."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def normal_integer_pmf(
    mu: float,
    sigma: float,
    *,
    support_min: int,
    support_max: int,
) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    values = np.arange(support_min, support_max + 1)
    upper = (values + 0.5 - mu) / sigma
    lower = (values - 0.5 - mu) / sigma
    cdf_upper = _normal_cdf(upper)
    cdf_lower = _normal_cdf(lower)
    pmf = cdf_upper - cdf_lower
    total = pmf.sum()
    if total > 0:
        pmf = pmf / total
    return pmf


def bins_from_pmf(
    pmf: np.ndarray,
    *,
    support_min: int,
    bin_specs: Iterable[dict],
) -> dict[str, float]:
    probs: dict[str, float] = {}
    for spec in bin_specs:
        name = spec.get("name")
        if not name:
            continue
        if spec.get("type") == "threshold":
            if "lt" in spec:
                cutoff = int(spec["lt"])
                upper_index = cutoff - support_min
                probs[name] = float(pmf[:upper_index].sum())
            elif "ge" in spec:
                cutoff = int(spec["ge"])
                lower_index = cutoff - support_min
                probs[name] = float(pmf[lower_index:].sum())
        elif spec.get("type") == "range":
            start = int(spec["start"])
            end = int(spec["end"])
            start_idx = max(start - support_min, 0)
            end_idx = min(end - support_min + 1, len(pmf))
            probs[name] = float(pmf[start_idx:end_idx].sum())
    return probs


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))
