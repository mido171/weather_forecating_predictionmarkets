from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .train_quantiles import repair_quantile_crossings


INTERVALS = {
    "50": (0.25, 0.75, 0.50),
    "80": (0.10, 0.90, 0.80),
    "90": (0.05, 0.95, 0.90),
    "95": (0.025, 0.975, 0.95),
    "98": (0.01, 0.99, 0.98),
}


@dataclass
class ConformalState:
    window: int
    min_warmup: int
    score_buffers: dict[str, deque]
    median_resid_buffer: deque


def _score(y: float, ql: float, qh: float) -> float:
    return float(max(ql - y, y - qh, 0.0))


def _split_conformal_threshold(scores: list[float], coverage: float) -> float:
    if not scores:
        return 0.0
    arr = np.array(scores, dtype=float)
    n = len(arr)
    q = min(0.999999, max(0.0, np.ceil((n + 1) * coverage) / n))
    return float(np.quantile(arr, q, method="higher" if hasattr(np, "quantile") else "linear"))


def _interp_quantile(pred_row: pd.Series, quantiles: list[float], tau: float) -> float:
    xs = np.array(quantiles, dtype=float)
    ys = pred_row[[f"q_{q:.3f}" for q in quantiles]].to_numpy(dtype=float)
    return float(np.interp(tau, xs, ys))


def init_conformal_state(window: int, min_warmup: int) -> ConformalState:
    return ConformalState(
        window=window,
        min_warmup=min_warmup,
        score_buffers={k: deque(maxlen=window) for k in INTERVALS.keys()},
        median_resid_buffer=deque(maxlen=window),
    )


def seed_conformal_state(
    state: ConformalState,
    rows: pd.DataFrame,
    pred_q: pd.DataFrame,
    quantiles: list[float],
) -> None:
    if rows.empty:
        return
    merged = rows.join(pred_q)
    merged = merged.sort_values("valid_time_utc")
    for _, r in merged.iterrows():
        y = float(r["y_tmax"])
        for key, (lo, hi, _cov) in INTERVALS.items():
            ql = _interp_quantile(r, quantiles, lo)
            qh = _interp_quantile(r, quantiles, hi)
            state.score_buffers[key].append(_score(y, ql, qh))
        q50 = _interp_quantile(r, quantiles, 0.50)
        state.median_resid_buffer.append(float(y - q50))


def _correction_map(thresholds: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    taus = np.array([0.01, 0.025, 0.05, 0.10, 0.25], dtype=float)
    corrs = np.array([
        thresholds.get("98", 0.0),
        thresholds.get("95", 0.0),
        thresholds.get("90", 0.0),
        thresholds.get("80", 0.0),
        thresholds.get("50", 0.0),
    ], dtype=float)
    return taus, corrs


def apply_rolling_conformal(
    rows: pd.DataFrame,
    blend_q: pd.DataFrame,
    quantiles: list[float],
    state: ConformalState,
    update_state: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if rows.empty:
        return blend_q.copy(), pd.DataFrame()

    merged = rows.join(blend_q)
    merged = merged.sort_values("valid_time_utc")

    corrected_rows = []
    diag_rows = []

    for idx, r in merged.iterrows():
        thresholds: dict[str, float] = {}
        hist_len = min(len(v) for v in state.score_buffers.values()) if state.score_buffers else 0
        warmup = hist_len < state.min_warmup

        for key, (_lo, _hi, cov) in INTERVALS.items():
            thresholds[key] = _split_conformal_threshold(list(state.score_buffers[key]), cov)

        med_bias = float(np.median(np.array(state.median_resid_buffer, dtype=float))) if len(state.median_resid_buffer) > 0 else 0.0

        corr_taus, corr_vals = _correction_map(thresholds)
        row_pred = {f"q_{q:.3f}": float(r[f"q_{q:.3f}"]) for q in quantiles}

        corrected = {}
        for q in quantiles:
            key = f"q_{q:.3f}"
            base = row_pred[key]
            if abs(q - 0.5) < 1e-12:
                corrected[key] = base + med_bias
            elif q < 0.5:
                c = float(np.interp(q, corr_taus, corr_vals))
                corrected[key] = base - c
            else:
                mirror = 1.0 - q
                c = float(np.interp(mirror, corr_taus, corr_vals))
                corrected[key] = base + c

        corrected_df = pd.DataFrame([corrected])
        corrected_df, repaired = repair_quantile_crossings(corrected_df, quantiles)
        corrected = corrected_df.iloc[0].to_dict()

        out_row = {"index": idx}
        out_row.update(corrected)
        out_row["conformal_warmup"] = bool(warmup)
        out_row["conformal_hist_len"] = int(hist_len)
        corrected_rows.append(out_row)

        diag_rows.append(
            {
                "index": idx,
                "target_date_local": str(r["target_date_local"]),
                "valid_time_utc": str(r["valid_time_utc"]),
                "conformal_warmup": bool(warmup),
                "hist_len": int(hist_len),
                "median_bias": med_bias,
                "threshold_50": thresholds.get("50", 0.0),
                "threshold_80": thresholds.get("80", 0.0),
                "threshold_90": thresholds.get("90", 0.0),
                "threshold_95": thresholds.get("95", 0.0),
                "threshold_98": thresholds.get("98", 0.0),
                "crossing_repaired": int(repaired),
            }
        )

        if update_state:
            y = float(r["y_tmax"])
            for key, (lo, hi, _cov) in INTERVALS.items():
                ql = _interp_quantile(r, quantiles, lo)
                qh = _interp_quantile(r, quantiles, hi)
                state.score_buffers[key].append(_score(y, ql, qh))
            q50 = _interp_quantile(r, quantiles, 0.5)
            state.median_resid_buffer.append(float(y - q50))

    corr_df = pd.DataFrame(corrected_rows).set_index("index").sort_index()
    diag_df = pd.DataFrame(diag_rows).set_index("index").sort_index()
    return corr_df, diag_df
