"""Utility helpers for MOS dataset generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd


def sha256_hex(data: str | bytes) -> str:
    payload = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(payload).hexdigest()


def canonical_json(obj: Any) -> str:
    if is_dataclass(obj):
        obj = asdict(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def hash_dict(obj: Any) -> str:
    return sha256_hex(canonical_json(obj))


def normalize_sql(sql: str) -> str:
    return " ".join(str(sql).split())


def to_date(value: Any) -> date:
    return pd.to_datetime(value).date()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def date_to_int(value: date) -> int:
    return int(value.strftime("%Y%m%d"))


def doy_diff(doy_a: int, doy_b: int) -> int:
    diff = abs(int(doy_a) - int(doy_b))
    return min(diff, 365 - diff)


def safe_div(numer: float, denom: float, default: float = np.nan) -> float:
    if denom == 0 or np.isnan(denom):
        return default
    return numer / denom


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return np.nan
    mask = ~(np.isnan(values) | np.isnan(weights))
    values = values[mask]
    weights = weights[mask]
    if values.size == 0:
        return np.nan
    w_sum = np.sum(weights)
    if w_sum <= 0:
        return np.nan
    return float(np.sum(values * weights) / w_sum)


def weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return np.nan
    mask = ~(np.isnan(values) | np.isnan(weights))
    values = values[mask]
    weights = weights[mask]
    if values.size == 0:
        return np.nan
    w_sum = np.sum(weights)
    if w_sum <= 0:
        return np.nan
    mean = np.sum(values * weights) / w_sum
    var = np.sum(weights * (values - mean) ** 2) / w_sum
    return float(np.sqrt(var))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: Iterable[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    quantiles = np.asarray(list(quantiles), dtype=float)
    if values.size == 0:
        return np.full_like(quantiles, np.nan, dtype=float)
    mask = ~(np.isnan(values) | np.isnan(weights))
    values = values[mask]
    weights = weights[mask]
    if values.size == 0:
        return np.full_like(quantiles, np.nan, dtype=float)
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    w_sum = np.sum(weights)
    if w_sum <= 0:
        return np.full_like(quantiles, np.nan, dtype=float)
    cum = np.cumsum(weights) / w_sum
    return np.interp(quantiles, cum, values, left=values[0], right=values[-1])


def weighted_entropy(weights: np.ndarray, eps: float = 1e-12) -> float:
    weights = np.asarray(weights, dtype=float)
    if weights.size == 0:
        return np.nan
    w_sum = np.sum(weights)
    if w_sum <= 0:
        return np.nan
    w = weights / w_sum
    return float(-np.sum(w * np.log(w + eps)))


def gini_from_weights(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    if weights.size == 0:
        return np.nan
    w_sum = np.sum(weights)
    if w_sum <= 0:
        return np.nan
    w = np.sort(weights / w_sum)
    n = w.size
    cum = np.cumsum(w)
    return float(1.0 - 2.0 * np.sum(cum) / n)


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    if weights.size == 0:
        return np.nan
    w_sum = np.sum(weights)
    if w_sum <= 0:
        return np.nan
    denom = np.sum(weights**2)
    if denom <= 0:
        return np.nan
    return float((w_sum**2) / denom)


def rolling_linear_regression(x: np.ndarray, y: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x_valid = np.where(mask, x, 0.0)
    y_valid = np.where(mask, y, 0.0)
    ones = mask.astype(float)

    sx = pd.Series(x_valid).rolling(window, min_periods=2).sum().to_numpy()
    sy = pd.Series(y_valid).rolling(window, min_periods=2).sum().to_numpy()
    sxx = pd.Series(x_valid**2).rolling(window, min_periods=2).sum().to_numpy()
    syy = pd.Series(y_valid**2).rolling(window, min_periods=2).sum().to_numpy()
    sxy = pd.Series(x_valid * y_valid).rolling(window, min_periods=2).sum().to_numpy()
    n = pd.Series(ones).rolling(window, min_periods=2).sum().to_numpy()

    b = np.full_like(sx, np.nan, dtype=float)
    a = np.full_like(sx, np.nan, dtype=float)
    r2 = np.full_like(sx, np.nan, dtype=float)

    valid = n >= 2
    denom = sxx - (sx**2) / n
    denom_ok = valid & (denom != 0)
    b[denom_ok] = (sxy - (sx * sy) / n)[denom_ok] / denom[denom_ok]
    a[denom_ok] = (sy / n)[denom_ok] - b[denom_ok] * (sx / n)[denom_ok]

    # R2 computation
    tss = syy - (sy**2) / n
    ssr = b * (sxy - (sx * sy) / n)
    r2_ok = denom_ok & (tss != 0)
    r2[r2_ok] = ssr[r2_ok] / tss[r2_ok]

    return a, b, r2
