"""Proper scoring and calibration diagnostics for ordered Tmax buckets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from hkg_tmax_probability.bucket_rules import (
    BUCKET_KEYS,
    PROBABILITY_COLUMNS,
    bucket_midpoints,
    normalize_probability_matrix,
)


@dataclass(frozen=True)
class ScoreSummary:
    rows: int
    rps: float
    nll: float
    brier: float
    crps: float
    ece: float
    mce: float
    entropy: float


def one_hot(indices: Iterable[int], width: int = len(BUCKET_KEYS)) -> np.ndarray:
    values = np.asarray(list(indices), dtype=int)
    out = np.zeros((len(values), width), dtype=float)
    out[np.arange(len(values)), values] = 1.0
    return out


def ranked_probability_score(probs: np.ndarray, true_indices: Iterable[int], normalized: bool = True) -> np.ndarray:
    matrix = normalize_probability_matrix(probs)
    truth = one_hot(true_indices, matrix.shape[1])
    diffs = np.cumsum(matrix, axis=1) - np.cumsum(truth, axis=1)
    scores = np.sum(diffs[:, :-1] ** 2, axis=1)
    if normalized:
        scores = scores / (matrix.shape[1] - 1)
    return scores


def multiclass_log_loss(probs: np.ndarray, true_indices: Iterable[int], floor: float = 1e-12) -> np.ndarray:
    matrix = normalize_probability_matrix(probs, floor=floor)
    idx = np.asarray(list(true_indices), dtype=int)
    return -np.log(np.clip(matrix[np.arange(len(idx)), idx], floor, 1.0))


def multiclass_brier(probs: np.ndarray, true_indices: Iterable[int]) -> np.ndarray:
    matrix = normalize_probability_matrix(probs)
    truth = one_hot(true_indices, matrix.shape[1])
    return np.mean((matrix - truth) ** 2, axis=1)


def crps_bucket_proxy(probs: np.ndarray, true_indices: Iterable[int]) -> np.ndarray:
    """CRPS approximation using bucket midpoint support."""
    matrix = normalize_probability_matrix(probs)
    mids = bucket_midpoints()
    idx = np.asarray(list(true_indices), dtype=int)
    observed = mids[idx]
    expected_abs_error = np.sum(matrix * np.abs(mids[None, :] - observed[:, None]), axis=1)
    pairwise_abs = np.abs(mids[:, None] - mids[None, :])
    expected_pairwise = np.einsum("ij,jk,ik->i", matrix, pairwise_abs, matrix)
    return expected_abs_error - 0.5 * expected_pairwise


def entropy(probs: np.ndarray) -> np.ndarray:
    matrix = normalize_probability_matrix(probs)
    return -np.sum(matrix * np.log(matrix), axis=1)


def calibration_errors(probs: np.ndarray, true_indices: Iterable[int], n_bins: int = 10) -> tuple[float, float, pd.DataFrame]:
    matrix = normalize_probability_matrix(probs)
    truth = np.asarray(list(true_indices), dtype=int)
    confidence = matrix.max(axis=1)
    predicted = matrix.argmax(axis=1)
    correct = (predicted == truth).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float | int]] = []
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        mask = (confidence >= lo) & ((confidence < hi) if i < n_bins - 1 else (confidence <= hi))
        count = int(mask.sum())
        if count == 0:
            rows.append({"bin": i, "lo": lo, "hi": hi, "count": 0, "confidence": np.nan, "accuracy": np.nan, "abs_gap": np.nan})
            continue
        conf = float(confidence[mask].mean())
        acc = float(correct[mask].mean())
        gap = abs(conf - acc)
        ece += gap * count / len(confidence)
        mce = max(mce, gap)
        rows.append({"bin": i, "lo": lo, "hi": hi, "count": count, "confidence": conf, "accuracy": acc, "abs_gap": gap})
    return float(ece), float(mce), pd.DataFrame(rows)


def summarize_scores(probs: np.ndarray, true_indices: Iterable[int]) -> ScoreSummary:
    y = list(true_indices)
    matrix = normalize_probability_matrix(probs)
    ece, mce, _ = calibration_errors(matrix, y)
    return ScoreSummary(
        rows=len(y),
        rps=float(np.mean(ranked_probability_score(matrix, y))),
        nll=float(np.mean(multiclass_log_loss(matrix, y))),
        brier=float(np.mean(multiclass_brier(matrix, y))),
        crps=float(np.mean(crps_bucket_proxy(matrix, y))),
        ece=ece,
        mce=mce,
        entropy=float(np.mean(entropy(matrix))),
    )


def probability_matrix_from_frame(frame: pd.DataFrame, prefix: str = "p_") -> np.ndarray:
    columns = [f"{prefix}{bucket}" for bucket in BUCKET_KEYS]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing probability columns: {missing}")
    return normalize_probability_matrix(frame[columns].to_numpy(dtype=float))


def score_frame(frame: pd.DataFrame, bucket_index_column: str = "bucket_index", prefix: str = "p_") -> ScoreSummary:
    return summarize_scores(probability_matrix_from_frame(frame, prefix=prefix), frame[bucket_index_column].astype(int).to_numpy())


def per_bucket_brier(probs: np.ndarray, true_indices: Iterable[int]) -> pd.DataFrame:
    matrix = normalize_probability_matrix(probs)
    truth = one_hot(true_indices, matrix.shape[1])
    rows = []
    for idx, bucket in enumerate(BUCKET_KEYS):
        rows.append({"bucket": bucket, "one_vs_rest_brier": float(np.mean((matrix[:, idx] - truth[:, idx]) ** 2))})
    return pd.DataFrame(rows)
