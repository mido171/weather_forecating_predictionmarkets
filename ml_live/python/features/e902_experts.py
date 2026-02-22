from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def fit_linear_calibration(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """
    Fit y ≈ a + b x using OLS. Returns (a, b, sigma_resid).
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        return float("nan"), float("nan"), float("nan")
    x = x[mask].astype(float)
    y = y[mask].astype(float)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    var = float(np.mean((x - x_mean) ** 2))
    if var <= 0.0:
        b = 0.0
    else:
        cov = float(np.mean((x - x_mean) * (y - y_mean)))
        b = cov / var
    a = y_mean - b * x_mean
    resid = y - (a + b * x)
    sigma = float(np.sqrt(np.mean(resid**2)))
    return a, b, sigma


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, float]:
    """
    Ridge regression with intercept; returns (coef, intercept).
    """
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if np.sum(mask) < X.shape[1] + 2:
        return np.full(X.shape[1], np.nan, dtype=float), float("nan")
    X = X[mask].astype(float)
    y = y[mask].astype(float)
    n, p = X.shape
    # Add intercept column (not regularized)
    X1 = np.hstack([np.ones((n, 1)), X])
    ridge = np.eye(p + 1) * alpha
    ridge[0, 0] = 0.0
    try:
        beta = np.linalg.solve(X1.T @ X1 + ridge, X1.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(X1.T @ X1 + ridge, X1.T @ y, rcond=None)[0]
    intercept = float(beta[0])
    coef = beta[1:]
    return coef.astype(float), intercept


def fit_ar1(residuals: np.ndarray) -> float:
    mask = np.isfinite(residuals)
    residuals = residuals[mask].astype(float)
    if residuals.size < 2:
        return float("nan")
    x = residuals[:-1]
    y = residuals[1:]
    denom = float(np.sum(x**2))
    if denom <= 0.0:
        return 0.0
    phi = float(np.sum(x * y) / denom)
    return phi


def mixture_mean(weights: np.ndarray, mus: np.ndarray) -> float:
    weights = _normalize_weights(weights)
    if weights is None:
        return float("nan")
    return float(np.sum(weights * mus))


def mixture_variance(weights: np.ndarray, mus: np.ndarray, sigmas: np.ndarray) -> float:
    weights = _normalize_weights(weights)
    if weights is None:
        return float("nan")
    mean = float(np.sum(weights * mus))
    second = float(np.sum(weights * (sigmas**2 + mus**2)))
    return second - mean**2


def mixture_cdf(x: float, weights: np.ndarray, mus: np.ndarray, sigmas: np.ndarray) -> float:
    weights = _normalize_weights(weights)
    if weights is None:
        return float("nan")
    z = (x - mus) / sigmas
    cdfs = 0.5 * (1.0 + _erf(z / math.sqrt(2.0)))
    return float(np.sum(weights * cdfs))


def mixture_quantile(
    q: float, weights: np.ndarray, mus: np.ndarray, sigmas: np.ndarray
) -> float:
    weights = _normalize_weights(weights)
    if weights is None:
        return float("nan")
    if not (0.0 < q < 1.0):
        raise ValueError("q must be in (0,1)")
    lo = float(np.nanmin(mus - 6.0 * sigmas))
    hi = float(np.nanmax(mus + 6.0 * sigmas))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        cdf = mixture_cdf(mid, weights, mus, sigmas)
        if not np.isfinite(cdf):
            return float("nan")
        if cdf < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def weights_entropy(weights: np.ndarray) -> float:
    weights = _normalize_weights(weights)
    if weights is None:
        return float("nan")
    eps = 1e-12
    return float(-np.sum(weights * np.log(weights + eps)))


def weights_eff_n(weights: np.ndarray) -> float:
    weights = _normalize_weights(weights)
    if weights is None:
        return float("nan")
    denom = float(np.sum(weights**2))
    if denom <= 0.0:
        return float("nan")
    return 1.0 / denom


def _normalize_weights(weights: np.ndarray) -> np.ndarray | None:
    weights = np.asarray(weights, dtype=float)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    s = float(np.sum(weights))
    if s <= 0.0:
        return None
    return weights / s


def _erf(x: np.ndarray) -> np.ndarray:
    # Vectorized error function using math.erf if numpy lacks erf
    try:
        return np.erf(x)  # type: ignore[attr-defined]
    except Exception:
        return np.vectorize(math.erf)(x)

