from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class EmosResult:
    c: float
    d: float
    sigma_emos: float
    rolling_bias: float
    rolling_rmse: float


def fit_emos_w45(
    history: np.ndarray,
    mu_hat: np.ndarray,
    sigma_hat: np.ndarray,
    sigma_floor: float = 0.5,
) -> tuple[float, float]:
    def nll(params: np.ndarray) -> float:
        c, d = params
        sigma2 = c + d * np.square(sigma_hat)
        sigma2 = np.maximum(sigma2, sigma_floor**2)
        return float(0.5 * np.sum(np.log(2.0 * math.pi * sigma2) + np.square(history - mu_hat) / sigma2))

    bounds = [(0.0, 100.0), (0.0, 100.0)]
    result = minimize(nll, np.array([0.5, 0.5]), bounds=bounds, method="L-BFGS-B")
    c, d = result.x
    return float(max(c, 0.0)), float(max(d, 0.0))


def apply_emos_sigma(sigma_hat: float, c: float, d: float, sigma_floor: float) -> float:
    sigma2 = c + d * (sigma_hat**2)
    sigma2 = max(sigma2, sigma_floor**2)
    return float(math.sqrt(sigma2))


def calibrate(
    history_df,
    target_sigma_hat: float,
    sigma_floor: float = 0.5,
) -> EmosResult:
    history = history_df["actual_tmax_f"].to_numpy(dtype=float)
    mu_hat = history_df["mu_hat_f"].to_numpy(dtype=float)
    sigma_hat = history_df["sigma_hat_f"].to_numpy(dtype=float)
    c, d = fit_emos_w45(history, mu_hat, sigma_hat, sigma_floor=sigma_floor)
    sigma_emos = apply_emos_sigma(target_sigma_hat, c, d, sigma_floor)
    residuals = history - mu_hat
    rolling_bias = float(np.mean(residuals))
    rolling_rmse = float(np.sqrt(np.mean(np.square(residuals))))
    return EmosResult(c=c, d=d, sigma_emos=sigma_emos, rolling_bias=rolling_bias, rolling_rmse=rolling_rmse)
