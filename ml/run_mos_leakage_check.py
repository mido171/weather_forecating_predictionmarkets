"""Sanity/leakage checks for MOS feature store."""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_feature_store(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    df["asof_date_local"] = pd.to_datetime(df["asof_date_local"]).dt.date
    return df.sort_values("target_date_local").reset_index(drop=True)


def expected_lag(series: pd.Series, dates: pd.Series, lag_days: int) -> np.ndarray:
    mapping = dict(zip(dates, series))
    return np.array([mapping.get(d - timedelta(days=lag_days), np.nan) for d in dates])


def rolling_mean(series: pd.Series, window: int) -> np.ndarray:
    return series.shift(1).rolling(window=window, min_periods=1).mean().to_numpy()


def ewma_bias(err: np.ndarray, alpha: float) -> np.ndarray:
    out = np.full_like(err, np.nan, dtype=float)
    prev = 0.0
    has_prev = False
    for i, val in enumerate(err):
        if not np.isnan(val):
            prev = (1 - alpha) * prev + alpha * val if has_prev else val
            has_prev = True
        out[i] = prev if has_prev else np.nan
    return pd.Series(out).shift(1).to_numpy()


def half_life_alpha(days: float) -> float:
    return 1.0 - np.exp(np.log(0.5) / days)


def compare_arrays(actual: np.ndarray, expected: np.ndarray, tol: float = 1e-6) -> dict[str, Any]:
    mask = np.isfinite(actual) & np.isfinite(expected)
    if not mask.any():
        return {"count": 0, "max_abs_diff": None, "mismatch_rate": None}
    diffs = np.abs(actual[mask] - expected[mask])
    mismatch = diffs > tol
    return {
        "count": int(mask.sum()),
        "max_abs_diff": float(np.max(diffs)),
        "mismatch_rate": float(np.mean(mismatch)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Leakage sanity checks on MOS feature store.")
    parser.add_argument("--feature-store", required=True, help="Path to feature_store.csv")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--obs-cutoff-lag-days", type=int, default=0)
    args = parser.parse_args()

    df = load_feature_store(Path(args.feature_store))
    y = pd.to_numeric(df["y_actual_tmax_f"], errors="coerce")
    dates = df["target_date_local"]

    # Date alignment
    expected_asof = dates.apply(lambda d: d - timedelta(days=1))
    asof_ok = (df["asof_date_local"] == expected_asof)

    # Lag checks
    lag1 = expected_lag(y, dates, 1 + args.obs_cutoff_lag_days - 0)
    lag2 = expected_lag(y, dates, 2 + args.obs_cutoff_lag_days - 0)

    # Rolling means (based on shifted series)
    roll7 = rolling_mean(y, 7)
    roll30 = rolling_mean(y, 30)

    # EWMA bias check (gfs/nam)
    alpha_15 = half_life_alpha(15.0)
    err_gfs = y.to_numpy(dtype=float) - pd.to_numeric(df.get("feat_tmp_max_gfs_b0"), errors="coerce").to_numpy(dtype=float)
    err_nam = y.to_numpy(dtype=float) - pd.to_numeric(df.get("feat_tmp_max_nam_b0"), errors="coerce").to_numpy(dtype=float)
    ewma_gfs = ewma_bias(err_gfs, alpha_15)
    ewma_nam = ewma_bias(err_nam, alpha_15)

    results = {
        "rows": int(len(df)),
        "asof_alignment": {
            "mismatch_count": int((~asof_ok).sum()),
            "mismatch_rate": float((~asof_ok).mean()),
        },
        "lag_checks": {
            "feat_tmax_lag1": compare_arrays(
                pd.to_numeric(df.get("feat_tmax_lag1"), errors="coerce").to_numpy(), lag1
            ),
            "feat_tmax_lag2": compare_arrays(
                pd.to_numeric(df.get("feat_tmax_lag2"), errors="coerce").to_numpy(), lag2
            ),
        },
        "rolling_checks": {
            "feat_tmax_roll7_mean": compare_arrays(
                pd.to_numeric(df.get("feat_tmax_roll7_mean"), errors="coerce").to_numpy(), roll7
            ),
            "feat_tmax_roll30_mean": compare_arrays(
                pd.to_numeric(df.get("feat_tmax_roll30_mean"), errors="coerce").to_numpy(), roll30
            ),
        },
        "bias_checks": {
            "feat_bias_ewma_15d_gfs": compare_arrays(
                pd.to_numeric(df.get("feat_bias_ewma_15d_gfs"), errors="coerce").to_numpy(), ewma_gfs
            ),
            "feat_bias_ewma_15d_nam": compare_arrays(
                pd.to_numeric(df.get("feat_bias_ewma_15d_nam"), errors="coerce").to_numpy(), ewma_nam
            ),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
