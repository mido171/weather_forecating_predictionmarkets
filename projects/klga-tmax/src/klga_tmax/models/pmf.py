from __future__ import annotations

from dataclasses import dataclass
import math

from klga_tmax.constants import PMF_SUM_TOLERANCE, TEMP_GRID_F


class PmfError(ValueError):
    pass


@dataclass(frozen=True)
class PmfSummary:
    pmf: dict[str, float]
    expected_tmax_f: float
    median_tmax_f: int
    mode_tmax_f: int
    prediction_interval_low_f: int
    prediction_interval_high_f: int
    uncertainty_f: float
    entropy: float


def gaussian_pmf(mean_f: float, sigma_f: float) -> dict[str, float]:
    sigma = max(float(sigma_f), 0.75)
    weights = {
        str(temp): math.exp(-0.5 * ((temp - mean_f) / sigma) ** 2)
        for temp in TEMP_GRID_F
    }
    return normalize_pmf(weights)


def normalize_pmf(weights: dict[str, float]) -> dict[str, float]:
    clipped = {str(temp): max(0.0, float(weights.get(str(temp), 0.0))) for temp in TEMP_GRID_F}
    total = sum(clipped.values())
    if total <= 0.0:
        raise PmfError("PMF weights must contain positive mass")
    pmf = {temp: value / total for temp, value in clipped.items()}
    validate_pmf(pmf)
    return pmf


def validate_pmf(pmf: dict[str, float]) -> None:
    missing = [temp for temp in TEMP_GRID_F if str(temp) not in pmf]
    if missing:
        raise PmfError(f"PMF missing temperatures: {missing[:5]}")
    total = sum(float(pmf[str(temp)]) for temp in TEMP_GRID_F)
    if abs(total - 1.0) > PMF_SUM_TOLERANCE:
        raise PmfError(f"PMF sums to {total}, not 1")
    negative = [temp for temp in TEMP_GRID_F if float(pmf[str(temp)]) < 0.0]
    if negative:
        raise PmfError(f"PMF has negative probabilities at: {negative[:5]}")


def summarize_pmf(pmf: dict[str, float], *, interval_mass: float = 0.80) -> PmfSummary:
    validate_pmf(pmf)
    expected = sum(temp * float(pmf[str(temp)]) for temp in TEMP_GRID_F)
    mode = max(TEMP_GRID_F, key=lambda temp: float(pmf[str(temp)]))
    cumulative = 0.0
    median = TEMP_GRID_F[-1]
    low_threshold = (1.0 - interval_mass) / 2.0
    high_threshold = 1.0 - low_threshold
    low = TEMP_GRID_F[0]
    high = TEMP_GRID_F[-1]
    low_set = False
    for temp in TEMP_GRID_F:
        cumulative += float(pmf[str(temp)])
        if not low_set and cumulative >= low_threshold:
            low = temp
            low_set = True
        if cumulative >= 0.5 and median == TEMP_GRID_F[-1]:
            median = temp
        if cumulative >= high_threshold:
            high = temp
            break
    variance = sum(((temp - expected) ** 2) * float(pmf[str(temp)]) for temp in TEMP_GRID_F)
    entropy = -sum(
        prob * math.log(prob)
        for prob in (float(pmf[str(temp)]) for temp in TEMP_GRID_F)
        if prob > 0.0
    )
    return PmfSummary(
        pmf=pmf,
        expected_tmax_f=expected,
        median_tmax_f=median,
        mode_tmax_f=mode,
        prediction_interval_low_f=low,
        prediction_interval_high_f=high,
        uncertainty_f=math.sqrt(max(variance, 0.0)),
        entropy=entropy,
    )


def shift_pmf(pmf: dict[str, float], shift_f: float) -> dict[str, float]:
    shifted: dict[str, float] = {str(temp): 0.0 for temp in TEMP_GRID_F}
    for temp in TEMP_GRID_F:
        shifted_temp = int(round(temp + shift_f))
        bounded = min(max(shifted_temp, TEMP_GRID_F[0]), TEMP_GRID_F[-1])
        shifted[str(bounded)] += float(pmf[str(temp)])
    return normalize_pmf(shifted)
