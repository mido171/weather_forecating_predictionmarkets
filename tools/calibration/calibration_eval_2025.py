#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_INPUT_ROOT = Path(
    r"C:\Users\ahmad\Desktop\generalFiles\git\weather-forecasting-predictionmarkets"
    r"\weather_forecating_predictionmarkets\artifacts\time_feature_sweep\kmia"
)
DEFAULT_YEAR = 2025
DEFAULT_TRUTH_LAG = 2
DEFAULT_WINDOWS = [30, 45, 60, 90]
DEFAULT_SIGMA_MINS = [0.5, 0.75, 1.0]
DEFAULT_THRESHOLDS = list(range(80, 101))

DATE_CANDIDATES = ["target_date_local", "target_date", "date", "ds"]
TRUTH_CANDIDATES = [
    "y",
    "y_true",
    "actual",
    "target",
    "target_tmax_f",
    "actual_tmax_f",
]
MU_CANDIDATES = [
    "mu_raw",
    "mu_hat_f",
    "mu_hat",
    "mu",
    "yhat",
    "y_pred",
    "pred",
    "tmax_pred",
    "tmax_hat",
]
UNCERTAINTY_CANDIDATES = [
    "sigma_hat_f",
    "sigma_hat",
    "sigma_model",
    "ens_spread",
    "gefs_spread",
    "model_std",
    "ens_std",
    "spread",
    "tmp_spread_f",
    "pred_sigma",
]
MODEL_TOKENS = [
    "nbm",
    "hrrr",
    "rap",
    "gfs",
    "nam",
    "gefs",
    "gefsatmos",
    "gefsatmosmean",
    "tmax",
    "n_x_max",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate calibration for KMIA experiments (2025)."
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default=str(DEFAULT_INPUT_ROOT),
        help="Root folder with KMIA time_feature_sweep artifacts.",
    )
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument("--truth-lag", type=int, default=DEFAULT_TRUTH_LAG)
    parser.add_argument("--windows", nargs="*", type=int, default=DEFAULT_WINDOWS)
    parser.add_argument(
        "--sigma-min", nargs="*", type=float, default=DEFAULT_SIGMA_MINS
    )
    parser.add_argument("--max-experiments", type=int, default=None)
    parser.add_argument("--only-experiments", type=str, default=None)
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Default: <input-root>/calibration_eval_2025",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def resolve_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_map = {col.lower(): col for col in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def normalize_date(series: pd.Series) -> pd.Series:
    dt_series = pd.to_datetime(series, errors="coerce")
    if dt_series.dt.tz is not None:
        dt_series = dt_series.dt.tz_convert(None)
    return dt_series.dt.normalize()


def detect_uncertainty_column(df: pd.DataFrame) -> Optional[str]:
    return resolve_column(df, UNCERTAINTY_CANDIDATES)


def detect_model_columns(
    df: pd.DataFrame, exclude: Iterable[str]
) -> List[str]:
    exclude_lower = {col.lower() for col in exclude if col}
    cols: List[str] = []
    for col in df.columns:
        low = col.lower()
        if low in exclude_lower:
            continue
        if any(token in low for token in MODEL_TOKENS):
            cols.append(col)
    return cols


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported prediction file type: {path}")


def find_sweep_json(input_root: Path) -> Optional[Path]:
    candidates = list(input_root.glob("time_feature_sweep*.json"))
    if not candidates:
        candidates = list(input_root.rglob("time_feature_sweep*.json"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def resolve_run_dir(run_dir_raw: Optional[str], input_root: Path) -> Optional[Path]:
    if not run_dir_raw:
        return None
    run_dir = Path(run_dir_raw)
    if run_dir.exists():
        return run_dir
    parts = list(run_dir.parts)
    if "time_feature_sweep" in parts and input_root.name not in parts:
        tfs_index = parts.index("time_feature_sweep")
        parts.insert(tfs_index + 1, input_root.name)
        candidate = Path(*parts)
        if candidate.exists():
            return candidate
    if not run_dir.is_absolute():
        candidate = input_root / run_dir
        if candidate.exists():
            return candidate
    tail = run_dir.name
    matches = [p for p in input_root.rglob(tail) if p.is_dir()]
    if len(matches) == 1:
        return matches[0]
    return None


def find_prediction_file(run_dir: Path) -> Optional[Path]:
    patterns = [
        "predictions*.csv",
        "preds*.csv",
        "forecast*.csv",
        "predictions*.parquet",
        "preds*.parquet",
        "forecast*.parquet",
    ]
    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(run_dir.glob(pattern))
    if not candidates:
        for pattern in patterns:
            candidates.extend(run_dir.rglob(pattern))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime)
    for cand in candidates:
        if "test" in cand.name.lower():
            return cand
    return candidates[-1]


def find_dataset_root(run_dir: Path) -> Optional[Path]:
    for parent in run_dir.parents:
        candidate = parent / "datasets"
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def resolve_dataset_path(
    datasets_root: Path, dataset_id: str
) -> Optional[Path]:
    metadata_path = datasets_root / "metadata.json"
    if metadata_path.exists():
        metadata = load_json(metadata_path)
        entry = metadata.get(dataset_id) or metadata.get(dataset_id.strip())
        if entry and "prefix" in entry:
            candidate = datasets_root / entry["prefix"] / "data.parquet"
            if candidate.exists():
                return candidate
    candidates = list(datasets_root.glob(f"{dataset_id}*/data.parquet"))
    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime)
        return candidates[-1]
    # Fallback: dataset directories may be stored under a short hash prefix.
    prefix_matches = []
    for child in datasets_root.iterdir():
        if not child.is_dir():
            continue
        if dataset_id.startswith(child.name):
            data_path = child / "data.parquet"
            if data_path.exists():
                prefix_matches.append(data_path)
    if prefix_matches:
        prefix_matches.sort(key=lambda p: p.stat().st_mtime)
        return prefix_matches[-1]
    return None


def load_dataset_for_experiment(run_dir: Path) -> Optional[pd.DataFrame]:
    dataset_id_path = run_dir / "dataset_id.txt"
    if not dataset_id_path.exists():
        return None
    dataset_id = dataset_id_path.read_text(encoding="utf-8").strip()
    datasets_root = find_dataset_root(run_dir)
    if not datasets_root:
        return None
    data_path = resolve_dataset_path(datasets_root, dataset_id)
    if not data_path:
        return None
    df = pd.read_parquet(data_path)
    df.attrs["source_path"] = str(data_path)
    return df


def choose_thresholds(y_vals: np.ndarray) -> List[int]:
    if y_vals.size == 0:
        return DEFAULT_THRESHOLDS
    p05 = float(np.nanpercentile(y_vals, 5))
    p95 = float(np.nanpercentile(y_vals, 95))
    if not np.isfinite(p05) or not np.isfinite(p95):
        return DEFAULT_THRESHOLDS
    low = int(np.floor(p05 - 5))
    high = int(np.ceil(p95 + 5))
    if high <= low:
        return DEFAULT_THRESHOLDS
    return list(range(low, high + 1))


def compute_crps(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = (y - mu) / sigma
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    return sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / np.sqrt(np.pi))


def compute_nll(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 0.5 * np.log(2 * np.pi * sigma**2) + ((y - mu) ** 2) / (2 * sigma**2)


def load_optional_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return load_json(path)


def load_optional_csv(path: Path) -> Optional[List[dict]]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def build_run_summary(run_dir: Path) -> dict:
    run_config = load_optional_json(run_dir / "run_config.json") or {}
    best_overall = load_optional_json(run_dir / "best_overall.json")
    leaderboard = load_optional_csv(run_dir / "leaderboard_experiments.csv") or []
    experiments = []
    experiments_root = run_dir / "experiments"
    if experiments_root.exists():
        for exp_dir in sorted(experiments_root.iterdir()):
            if not exp_dir.is_dir():
                continue
            exp_payload = {
                "experiment_id": exp_dir.name,
                "experiment_meta": load_optional_json(exp_dir / "experiment_meta.json"),
                "best_calibrator": load_optional_json(exp_dir / "best_calibrator.json"),
                "metrics_2025": load_optional_json(exp_dir / "metrics_2025.json"),
                "calibrator_search": load_optional_csv(
                    exp_dir / "calibrator_search.csv"
                ),
                "daily_2025": load_optional_csv(exp_dir / "daily_2025.csv"),
                "plots_dir": str(exp_dir / "plots"),
            }
            experiments.append(exp_payload)
    return {
        "generated_at": datetime.now().isoformat(),
        "run_dir": str(run_dir),
        "run_config": run_config,
        "leaderboard": leaderboard,
        "best_overall": best_overall,
        "experiments": experiments,
    }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_calibrator_grid(
    windows: List[int], sigma_mins: List[float], emos_enabled: bool
) -> List[dict]:
    grid: List[dict] = []
    for window in windows:
        for sigma_min in sigma_mins:
            grid.append(
                {
                    "calibrator_id": f"A_w{window}_s{sigma_min}",
                    "family": "A",
                    "window": window,
                    "sigma_min": sigma_min,
                }
            )
    if emos_enabled:
        for window in windows:
            for mean_mapping in [0, 1]:
                for use_g in [0, 1]:
                    grid.append(
                        {
                            "calibrator_id": (
                                f"B_w{window}_m{mean_mapping}_g{use_g}"
                            ),
                            "family": "B",
                            "window": window,
                            "mean_mapping": bool(mean_mapping),
                            "use_g": bool(use_g),
                        }
                    )
    return grid


def summarize_metrics(daily: pd.DataFrame, thresholds: List[int]) -> dict:
    scored = daily[daily["insufficient_history"] == 0].copy()
    if scored.empty:
        return {}
    y = scored["y_true"].to_numpy(dtype=float)
    mu = scored["mu_calib"].to_numpy(dtype=float)
    sigma = scored["sigma"].to_numpy(dtype=float)
    metrics = {
        "mean_crps": float(np.nanmean(scored["crps"])),
        "mean_nll": float(np.nanmean(scored["nll"])),
        "mae_mu": float(np.nanmean(np.abs(mu - y))),
        "rmse_mu": float(np.sqrt(np.nanmean((mu - y) ** 2))),
        "bias_mu": float(np.nanmean(mu - y)),
        "sharpness_mean_sigma": float(np.nanmean(sigma)),
        "sharpness_median_sigma": float(np.nanmedian(sigma)),
        "sharpness_p90_sigma": float(np.nanpercentile(sigma, 90)),
    }
    coverage = {}
    for level in [0.5, 0.8, 0.9, 0.95]:
        z = norm.ppf((1 + level) / 2)
        lower = mu - z * sigma
        upper = mu + z * sigma
        cov = float(np.mean((y >= lower) & (y <= upper)))
        coverage[f"{int(level*100)}"] = cov
    metrics["coverage"] = coverage
    metrics["coverage_error_90"] = float(coverage["90"] - 0.9)
    brier = {}
    for k in thresholds:
        p_col = f"p_ge_{k}"
        o_col = f"o_ge_{k}"
        if p_col not in scored or o_col not in scored:
            continue
        p = scored[p_col].to_numpy(dtype=float)
        o = scored[o_col].to_numpy(dtype=float)
        brier[str(k)] = float(np.mean((p - o) ** 2))
    metrics["brier_per_k"] = brier
    if brier:
        metrics["brier_mean"] = float(np.mean(list(brier.values())))
    return metrics


def choose_best_calibrator(rows: List[dict]) -> Optional[dict]:
    if not rows:
        return None
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            r.get("mean_crps", np.inf),
            r.get("mean_nll", np.inf),
            abs(r.get("cov90_err", np.inf)),
        ),
    )
    return rows_sorted[0]


def fit_emos_params(
    mu_w: np.ndarray,
    y_w: np.ndarray,
    u_w: np.ndarray,
    s_hat: float,
    mean_mapping: bool,
    use_g: bool,
    sigma_min: float,
) -> Optional[np.ndarray]:
    if mu_w.size < 20:
        return None
    if not np.all(np.isfinite(u_w)):
        return None
    if mean_mapping:
        if use_g:
            x0 = np.array([0.0, 1.0, 1.0, 0.5, 0.1], dtype=float)
            bounds = [(None, None), (None, None), (0, None), (0, None), (0, None)]
        else:
            x0 = np.array([0.0, 1.0, 1.0, 0.5], dtype=float)
            bounds = [(None, None), (None, None), (0, None), (0, None)]
    else:
        if use_g:
            x0 = np.array([1.0, 0.5, 0.1], dtype=float)
            bounds = [(0, None), (0, None), (0, None)]
        else:
            x0 = np.array([1.0, 0.5], dtype=float)
            bounds = [(0, None), (0, None)]

    def nll(params: np.ndarray) -> float:
        if mean_mapping:
            if use_g:
                a, b, c, d, g = params
            else:
                a, b, c, d = params
                g = 0.0
            mu = a + b * mu_w
        else:
            if use_g:
                c, d, g = params
            else:
                c, d = params
                g = 0.0
            mu = mu_w
        sigma2 = c + d * (u_w**2) + g * (s_hat**2)
        sigma2 = np.maximum(sigma2, sigma_min**2)
        return float(np.mean(compute_nll(mu, np.sqrt(sigma2), y_w)))

    result = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
    if not result.success:
        return None
    return result.x


def calibrate_family_a(
    df: pd.DataFrame,
    year: int,
    truth_lag: int,
    window: int,
    sigma_min: float,
) -> pd.DataFrame:
    data = df.sort_values("date").reset_index(drop=True)
    dates = data["date"].to_numpy(dtype="datetime64[D]")
    mu_raw = data["mu_raw"].to_numpy(dtype=float)
    y_true = data["y_true"].to_numpy(dtype=float)
    resid = y_true - mu_raw
    valid_resid = np.isfinite(resid)
    resid_dates = dates[valid_resid]
    resid_values = resid[valid_resid]
    results = []
    for idx, current_date in enumerate(dates):
        if current_date.astype("datetime64[Y]").astype(int) + 1970 != year:
            continue
        mu_val = mu_raw[idx]
        y_val = y_true[idx]
        if not np.isfinite(mu_val) or not np.isfinite(y_val):
            continue
        end_date = current_date - np.timedelta64(truth_lag, "D")
        start_date = current_date - np.timedelta64(
            window + truth_lag - 1, "D"
        )
        start_idx = resid_dates.searchsorted(start_date, side="left")
        end_idx = resid_dates.searchsorted(end_date, side="right")
        if end_idx > start_idx:
            max_hist_date = resid_dates[start_idx:end_idx].max()
            if max_hist_date > end_date:
                raise AssertionError(
                    "Leakage check failed: history uses future dates."
                )
        window_resid = resid_values[start_idx:end_idx]
        insufficient = 0
        if window_resid.size < 20:
            insufficient = 1
            results.append(
                {
                    "date": pd.Timestamp(current_date),
                    "y_true": y_val,
                    "mu_raw": mu_val,
                    "mu_calib": np.nan,
                    "sigma": np.nan,
                    "z": np.nan,
                    "pit": np.nan,
                    "crps": np.nan,
                    "nll": np.nan,
                    "insufficient_history": insufficient,
                }
            )
            continue
        bias = float(np.mean(window_resid))
        sigma = float(np.std(window_resid, ddof=1))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = sigma_min
        sigma = max(sigma, sigma_min)
        mu_calib = mu_val + bias
        z = (y_val - mu_calib) / sigma
        pit = float(norm.cdf(z))
        crps = float(
            compute_crps(
                np.array([mu_calib]), np.array([sigma]), np.array([y_val])
            )[0]
        )
        nll = float(
            compute_nll(
                np.array([mu_calib]), np.array([sigma]), np.array([y_val])
            )[0]
        )
        results.append(
            {
                "date": pd.Timestamp(current_date),
                "y_true": y_val,
                "mu_raw": mu_val,
                "mu_calib": mu_calib,
                "sigma": sigma,
                "z": z,
                "pit": pit,
                "crps": crps,
                "nll": nll,
                "insufficient_history": insufficient,
            }
        )
    return pd.DataFrame(results)


def calibrate_family_b(
    df: pd.DataFrame,
    year: int,
    truth_lag: int,
    window: int,
    sigma_min: float,
    mean_mapping: bool,
    use_g: bool,
) -> pd.DataFrame:
    data = df.sort_values("date").reset_index(drop=True)
    dates = data["date"].to_numpy(dtype="datetime64[D]")
    mu_raw = data["mu_raw"].to_numpy(dtype=float)
    y_true = data["y_true"].to_numpy(dtype=float)
    u_vals = data["u"].to_numpy(dtype=float)
    resid = y_true - mu_raw
    valid_resid = np.isfinite(resid)
    resid_dates = dates[valid_resid]
    resid_values = resid[valid_resid]
    results = []
    for idx, current_date in enumerate(dates):
        if current_date.astype("datetime64[Y]").astype(int) + 1970 != year:
            continue
        mu_val = mu_raw[idx]
        y_val = y_true[idx]
        u_val = u_vals[idx]
        if not np.isfinite(mu_val) or not np.isfinite(y_val) or not np.isfinite(u_val):
            continue
        end_date = current_date - np.timedelta64(truth_lag, "D")
        start_date = current_date - np.timedelta64(
            window + truth_lag - 1, "D"
        )
        start_idx = dates.searchsorted(start_date, side="left")
        end_idx = dates.searchsorted(end_date, side="right")
        if end_idx > start_idx:
            max_hist_date = dates[start_idx:end_idx].max()
            if max_hist_date > end_date:
                raise AssertionError(
                    "Leakage check failed: history uses future dates."
                )
        mu_w = mu_raw[start_idx:end_idx]
        y_w = y_true[start_idx:end_idx]
        u_w = u_vals[start_idx:end_idx]
        mask = np.isfinite(mu_w) & np.isfinite(y_w) & np.isfinite(u_w)
        mu_w = mu_w[mask]
        y_w = y_w[mask]
        u_w = u_w[mask]
        insufficient = 0
        if mu_w.size < 20:
            insufficient = 1
            results.append(
                {
                    "date": pd.Timestamp(current_date),
                    "y_true": y_val,
                    "mu_raw": mu_val,
                    "mu_calib": np.nan,
                    "sigma": np.nan,
                    "z": np.nan,
                    "pit": np.nan,
                    "crps": np.nan,
                    "nll": np.nan,
                    "insufficient_history": insufficient,
                }
            )
            continue
        s_hat = float(np.std(y_w - mu_w, ddof=1))
        params = fit_emos_params(
            mu_w, y_w, u_w, s_hat, mean_mapping, use_g, sigma_min
        )
        if params is None:
            results.append(
                {
                    "date": pd.Timestamp(current_date),
                    "y_true": y_val,
                    "mu_raw": mu_val,
                    "mu_calib": np.nan,
                    "sigma": np.nan,
                    "z": np.nan,
                    "pit": np.nan,
                    "crps": np.nan,
                    "nll": np.nan,
                    "insufficient_history": 1,
                }
            )
            continue
        if mean_mapping:
            if use_g:
                a, b, c, d, g = params
            else:
                a, b, c, d = params
                g = 0.0
            mu_calib = a + b * mu_val
        else:
            if use_g:
                c, d, g = params
            else:
                c, d = params
                g = 0.0
            mu_calib = mu_val
        sigma2 = c + d * (u_val**2) + g * (s_hat**2)
        sigma2 = max(sigma2, sigma_min**2)
        sigma = float(np.sqrt(sigma2))
        z = (y_val - mu_calib) / sigma
        pit = float(norm.cdf(z))
        crps = float(
            compute_crps(
                np.array([mu_calib]), np.array([sigma]), np.array([y_val])
            )[0]
        )
        nll = float(
            compute_nll(
                np.array([mu_calib]), np.array([sigma]), np.array([y_val])
            )[0]
        )
        results.append(
            {
                "date": pd.Timestamp(current_date),
                "y_true": y_val,
                "mu_raw": mu_val,
                "mu_calib": mu_calib,
                "sigma": sigma,
                "z": z,
                "pit": pit,
                "crps": crps,
                "nll": nll,
                "insufficient_history": insufficient,
            }
        )
    return pd.DataFrame(results)

def add_threshold_columns(daily: pd.DataFrame, thresholds: List[int]) -> pd.DataFrame:
    if daily.empty:
        return daily
    mu = daily["mu_calib"].to_numpy(dtype=float)
    sigma = daily["sigma"].to_numpy(dtype=float)
    y = daily["y_true"].to_numpy(dtype=float)
    for k in thresholds:
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (k - mu) / sigma
            p = 1.0 - norm.cdf(z)
        daily[f"p_ge_{k}"] = p
        daily[f"o_ge_{k}"] = (y >= k).astype(int)
    return daily


def plot_pit_hist(daily: pd.DataFrame, path: Path) -> None:
    scored = daily[daily["insufficient_history"] == 0]
    pits = scored["pit"].dropna()
    if pits.empty:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(pits, bins=20, range=(0, 1), edgecolor="black")
    plt.xlabel("PIT")
    plt.ylabel("Count")
    plt.title("PIT Histogram")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_sigma_timeseries(daily: pd.DataFrame, path: Path) -> None:
    scored = daily[daily["insufficient_history"] == 0]
    if scored.empty:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(scored["date"], scored["sigma"], color="tab:blue")
    plt.xlabel("Date")
    plt.ylabel("Sigma")
    plt.title("Sigma Time Series")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_crps_timeseries(daily: pd.DataFrame, path: Path) -> None:
    scored = daily[daily["insufficient_history"] == 0]
    if scored.empty:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(scored["date"], scored["crps"], color="tab:green")
    plt.xlabel("Date")
    plt.ylabel("CRPS")
    plt.title("CRPS Time Series")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_coverage_bars(metrics: dict, path: Path) -> None:
    coverage = metrics.get("coverage", {})
    if not coverage:
        return
    levels = sorted(coverage.keys(), key=lambda v: int(v))
    actual = [coverage[level] for level in levels]
    nominal = [int(level) / 100 for level in levels]
    x = np.arange(len(levels))
    width = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, nominal, width, label="Nominal")
    plt.bar(x + width / 2, actual, width, label="Actual")
    plt.xticks(x, levels)
    plt.ylim(0, 1)
    plt.xlabel("Interval (%)")
    plt.ylabel("Coverage")
    plt.title("Coverage vs Nominal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_reliability(daily: pd.DataFrame, k: int, path: Path) -> None:
    p_col = f"p_ge_{k}"
    o_col = f"o_ge_{k}"
    if p_col not in daily or o_col not in daily:
        return
    p = daily[p_col].to_numpy(dtype=float)
    o = daily[o_col].to_numpy(dtype=float)
    mask = np.isfinite(p) & np.isfinite(o)
    if mask.sum() == 0:
        return
    p = p[mask]
    o = o[mask]
    bins = np.linspace(0.0, 1.0, 11)
    bin_ids = np.digitize(p, bins) - 1
    bin_ids = np.clip(bin_ids, 0, len(bins) - 2)
    pred_means = []
    obs_means = []
    for b in range(len(bins) - 1):
        b_mask = bin_ids == b
        if b_mask.sum() == 0:
            continue
        pred_means.append(float(np.mean(p[b_mask])))
        obs_means.append(float(np.mean(o[b_mask])))
    if not pred_means:
        return
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.plot(pred_means, obs_means, marker="o", color="tab:orange")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(f"Reliability (K={k})")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def select_reliability_thresholds(
    thresholds: List[int], count: int = 5
) -> List[int]:
    if not thresholds:
        return []
    if len(thresholds) <= count:
        return thresholds
    idx = np.linspace(0, len(thresholds) - 1, count)
    selected = [thresholds[int(round(i))] for i in idx]
    return sorted(set(selected))

def prepare_experiment_data(
    run_dir: Path, pred_path: Path
) -> Tuple[pd.DataFrame, dict]:
    pred_df = load_table(pred_path)
    meta = {
        "run_dir": str(run_dir),
        "prediction_path": str(pred_path),
        "prediction_rows": int(len(pred_df)),
        "prediction_columns": list(pred_df.columns),
    }
    date_col = resolve_column(pred_df, DATE_CANDIDATES)
    mu_col = resolve_column(pred_df, MU_CANDIDATES)
    truth_col = resolve_column(pred_df, TRUTH_CANDIDATES)
    u_col = detect_uncertainty_column(pred_df)
    meta["date_column"] = date_col
    meta["mu_column"] = mu_col
    meta["truth_column"] = truth_col
    meta["uncertainty_column"] = u_col
    if not date_col or not mu_col:
        raise ValueError("Missing date or mu column in predictions.")

    pred_df = pred_df.copy()
    pred_df["date"] = normalize_date(pred_df[date_col])
    pred_df["mu_raw"] = pd.to_numeric(pred_df[mu_col], errors="coerce")
    if truth_col:
        pred_df["y_true"] = pd.to_numeric(pred_df[truth_col], errors="coerce")
    if u_col:
        pred_df["u"] = pd.to_numeric(pred_df[u_col], errors="coerce")

    model_cols = []
    if "u" not in pred_df.columns:
        model_cols = detect_model_columns(
            pred_df, exclude=[date_col, mu_col, truth_col]
        )
        if len(model_cols) >= 2:
            pred_df[model_cols] = pred_df[model_cols].apply(
                pd.to_numeric, errors="coerce"
            )
            pred_df["u"] = pred_df[model_cols].std(axis=1)
    meta["model_columns_pred"] = model_cols

    need_truth = "y_true" not in pred_df.columns
    need_u = "u" not in pred_df.columns
    dataset_df = None
    if need_truth or need_u:
        dataset_df = load_dataset_for_experiment(run_dir)
    meta["dataset_path"] = (
        dataset_df.attrs.get("source_path") if dataset_df is not None else None
    )
    if dataset_df is not None:
        ds = dataset_df.copy()
        ds_date_col = resolve_column(ds, DATE_CANDIDATES)
        ds_truth_col = resolve_column(ds, TRUTH_CANDIDATES)
        if ds_date_col:
            ds["date"] = normalize_date(ds[ds_date_col])
        if "station_id" in ds.columns:
            if "station_id" in pred_df.columns:
                station = pred_df["station_id"].dropna().unique()
                if len(station) == 1:
                    ds = ds[ds["station_id"] == station[0]]
            elif ds["station_id"].nunique() == 1:
                station = ds["station_id"].unique()[0]
                ds = ds[ds["station_id"] == station]
        if need_truth and ds_truth_col:
            ds["y_true"] = pd.to_numeric(ds[ds_truth_col], errors="coerce")
        ds_model_cols = []
        ds_u_col = detect_uncertainty_column(ds)
        if need_u and ds_u_col:
            ds["u"] = pd.to_numeric(ds[ds_u_col], errors="coerce")
        elif need_u:
            ds_model_cols = detect_model_columns(
                ds, exclude=[ds_date_col, ds_truth_col, "y_true", "station_id"]
            )
        meta["model_columns_dataset"] = ds_model_cols
        keep_cols = ["date"]
        if "y_true" in ds.columns:
            keep_cols.append("y_true")
        if "u" in ds.columns:
            keep_cols.append("u")
        keep_cols += ds_model_cols
        ds_subset = ds[keep_cols].copy()
        pred_df = pred_df.merge(ds_subset, on="date", how="left")
        if need_u and "u" in ds_subset.columns:
            pred_df["u"] = pd.to_numeric(pred_df["u"], errors="coerce")
        elif need_u and ds_model_cols:
            pred_df[ds_model_cols] = pred_df[ds_model_cols].apply(
                pd.to_numeric, errors="coerce"
            )
            pred_df["u"] = pred_df[ds_model_cols].std(axis=1)

    if "y_true" not in pred_df.columns:
        raise ValueError("Missing y_true column after dataset merge.")
    if "u" not in pred_df.columns:
        pred_df["u"] = np.nan

    df = pred_df[["date", "mu_raw", "y_true", "u"]].copy()
    df = df.dropna(subset=["date"])
    df = df.groupby("date", as_index=False).mean(numeric_only=True)
    df = df.sort_values("date").reset_index(drop=True)
    return df, meta


def discover_experiments(
    input_root: Path, sweep_json: Optional[Path]
) -> List[dict]:
    experiments: List[dict] = []
    seen = set()
    if sweep_json:
        payload = load_json(sweep_json)
        exp_list = payload.get("experiments") or payload.get("runs") or []
        base_dir = sweep_json.parent
        for exp in exp_list:
            exp_id = exp.get("experiment_id") or exp.get("id")
            exp_name = exp.get("description") or exp.get("name")
            run_dir = resolve_run_dir(
                exp.get("output_dir")
                or exp.get("output_path")
                or exp.get("run_dir"),
                input_root,
            )
            if run_dir is None and exp_id:
                candidate = base_dir / exp_id
                if candidate.exists():
                    run_dir = candidate
            pred_path = None
            if run_dir is not None:
                pred_path = find_prediction_file(run_dir)
            artifact_hashes = exp.get("artifact_hashes") or {}
            if pred_path is None:
                for key in artifact_hashes.keys():
                    if any(tag in key for tag in ["predictions", "preds", "forecast"]):
                        cand = Path(key)
                        if cand.exists():
                            pred_path = cand
                            run_dir = cand.parent
                            break
            if run_dir is None:
                continue
            metrics_path = None
            candidate = run_dir / "metrics.json"
            if candidate.exists():
                metrics_path = candidate
            else:
                candidate = run_dir / "result.json"
                if candidate.exists():
                    metrics_path = candidate
            key = (str(run_dir), exp_id)
            if key in seen:
                continue
            seen.add(key)
            experiments.append(
                {
                    "experiment_id": exp_id or run_dir.name,
                    "experiment_name": exp_name,
                    "run_dir": run_dir,
                    "prediction_path": pred_path,
                    "metrics_path": metrics_path,
                    "sweep_json": sweep_json,
                }
            )
    if experiments:
        return experiments

    patterns = [
        "predictions*.csv",
        "preds*.csv",
        "forecast*.csv",
        "predictions*.parquet",
        "preds*.parquet",
        "forecast*.parquet",
    ]
    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(input_root.rglob(pattern))
    for pred_path in candidates:
        run_dir = pred_path.parent
        exp_id = run_dir.name
        key = (str(run_dir), exp_id)
        if key in seen:
            continue
        seen.add(key)
        metrics_path = None
        candidate = run_dir / "metrics.json"
        if candidate.exists():
            metrics_path = candidate
        experiments.append(
            {
                "experiment_id": exp_id,
                "experiment_name": None,
                "run_dir": run_dir,
                "prediction_path": pred_path,
                "metrics_path": metrics_path,
                "sweep_json": sweep_json,
            }
        )
    return experiments

def run_experiment(
    exp_meta: dict,
    year: int,
    truth_lag: int,
    windows: List[int],
    sigma_mins: List[float],
    thresholds: List[int],
    output_root: Path,
    log_fn,
    verbose: bool,
) -> Optional[dict]:
    exp_id = exp_meta.get("experiment_id") or exp_meta["run_dir"].name
    exp_name = exp_meta.get("experiment_name")
    pred_path = exp_meta.get("prediction_path")
    if pred_path is None or not Path(pred_path).exists():
        log_fn(f"[SKIP] {exp_id}: missing prediction file")
        return None
    try:
        df, meta = prepare_experiment_data(exp_meta["run_dir"], Path(pred_path))
    except Exception as exc:
        log_fn(f"[SKIP] {exp_id}: {exc}")
        return None

    if df.empty:
        log_fn(f"[SKIP] {exp_id}: no data")
        return None
    year_mask = df["date"].dt.year == year
    if year_mask.sum() == 0:
        log_fn(f"[SKIP] {exp_id}: no rows for {year}")
        return None

    u_available = np.isfinite(df["u"]).sum() >= 20
    grid = build_calibrator_grid(windows, sigma_mins, u_available)
    if verbose:
        log_fn(
            f"[INFO] {exp_id}: calibrators={len(grid)} u_available={u_available}"
        )

    rows: List[dict] = []
    daily_by_id: dict = {}
    metrics_by_id: dict = {}
    for cal in grid:
        cal_id = cal["calibrator_id"]
        log_fn(f"[CAL] {exp_id} {cal_id}")
        if cal["family"] == "A":
            daily = calibrate_family_a(
                df,
                year=year,
                truth_lag=truth_lag,
                window=cal["window"],
                sigma_min=cal["sigma_min"],
            )
        else:
            daily = calibrate_family_b(
                df,
                year=year,
                truth_lag=truth_lag,
                window=cal["window"],
                sigma_min=min(sigma_mins),
                mean_mapping=cal.get("mean_mapping", False),
                use_g=cal.get("use_g", False),
            )
        if daily.empty:
            continue
        daily = add_threshold_columns(daily, thresholds)
        metrics = summarize_metrics(daily, thresholds)
        if not metrics:
            continue
        cov90 = metrics.get("coverage", {}).get("90", np.nan)
        cov90_err = metrics.get("coverage_error_90", np.nan)
        mean_sigma = metrics.get("sharpness_mean_sigma", np.nan)
        row = {
            "experiment_id": exp_id,
            "calibrator_id": cal_id,
            "family": cal["family"],
            "window": cal.get("window"),
            "sigma_min": cal.get("sigma_min"),
            "mean_mapping": cal.get("mean_mapping"),
            "use_g": cal.get("use_g"),
            "mean_crps": metrics.get("mean_crps"),
            "mean_nll": metrics.get("mean_nll"),
            "mae_mu": metrics.get("mae_mu"),
            "rmse_mu": metrics.get("rmse_mu"),
            "bias_mu": metrics.get("bias_mu"),
            "mean_sigma": mean_sigma,
            "cov90": cov90,
            "cov90_err": cov90_err,
            "brier_mean": metrics.get("brier_mean"),
        }
        rows.append(row)
        daily_by_id[cal_id] = daily
        metrics_by_id[cal_id] = metrics

    best = choose_best_calibrator(rows)
    if best is None:
        log_fn(f"[SKIP] {exp_id}: no valid calibrators")
        return None
    best_id = best["calibrator_id"]
    best_daily = daily_by_id[best_id]
    best_metrics = metrics_by_id[best_id]

    exp_dir = output_root / "experiments" / exp_id
    ensure_dir(exp_dir)
    ensure_dir(exp_dir / "plots")

    meta_payload = {
        "experiment_id": exp_id,
        "experiment_name": exp_name,
        "run_dir": str(exp_meta["run_dir"]),
        "prediction_path": str(pred_path),
        "metrics_path": str(exp_meta.get("metrics_path"))
        if exp_meta.get("metrics_path")
        else None,
        "sweep_json": str(exp_meta.get("sweep_json"))
        if exp_meta.get("sweep_json")
        else None,
        "source_metadata": meta,
    }
    dump_json(exp_dir / "experiment_meta.json", meta_payload)

    pd.DataFrame(rows).to_csv(exp_dir / "calibrator_search.csv", index=False)
    dump_json(
        exp_dir / "best_calibrator.json",
        {"best": best, "metrics": best_metrics},
    )
    best_daily.to_csv(exp_dir / "daily_2025.csv", index=False)
    dump_json(exp_dir / "metrics_2025.json", best_metrics)

    plot_pit_hist(best_daily, exp_dir / "plots" / "pit_hist.png")
    plot_sigma_timeseries(
        best_daily, exp_dir / "plots" / "sigma_timeseries.png"
    )
    plot_crps_timeseries(
        best_daily, exp_dir / "plots" / "crps_timeseries.png"
    )
    plot_coverage_bars(
        best_metrics, exp_dir / "plots" / "coverage_bars.png"
    )
    for k in select_reliability_thresholds(thresholds, count=5):
        plot_reliability(
            best_daily, k, exp_dir / "plots" / f"reliability_K_{k}.png"
        )

    return {
        "experiment_id": exp_id,
        "experiment_name": exp_name,
        "best_calibrator_id": best_id,
        "mean_crps": best_metrics.get("mean_crps"),
        "mean_nll": best_metrics.get("mean_nll"),
        "mae_mu": best_metrics.get("mae_mu"),
        "rmse_mu": best_metrics.get("rmse_mu"),
        "mean_sigma": best_metrics.get("sharpness_mean_sigma"),
        "cov90": best_metrics.get("coverage", {}).get("90"),
        "cov90_err": best_metrics.get("coverage_error_90"),
        "brier_mean": best_metrics.get("brier_mean"),
        "daily_path": str(exp_dir / "daily_2025.csv"),
        "metrics_path": str(exp_dir / "metrics_2025.json"),
        "best_calibrator_path": str(exp_dir / "best_calibrator.json"),
    }

def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    if not input_root.exists():
        print(f"Input root not found: {input_root}")
        return 1

    sweep_json = find_sweep_json(input_root)
    experiments = discover_experiments(input_root, sweep_json)
    if args.only_experiments:
        allowed = {
            item.strip()
            for item in args.only_experiments.split(",")
            if item.strip()
        }
        experiments = [
            exp for exp in experiments if exp.get("experiment_id") in allowed
        ]
    if args.max_experiments:
        experiments = experiments[: args.max_experiments]

    if not experiments:
        print("No experiments discovered.")
        return 1

    if args.dry_run:
        print(f"Discovered {len(experiments)} experiments")
        for exp in experiments:
            exp_id = exp.get("experiment_id")
            print(
                f"- {exp_id} run_dir={exp['run_dir']} pred={exp.get('prediction_path')}"
            )
            pred_path = exp.get("prediction_path")
            if pred_path is None or not Path(pred_path).exists():
                print("  missing prediction file")
                continue
            try:
                _, meta = prepare_experiment_data(
                    exp["run_dir"], Path(pred_path)
                )
                print(
                    "  columns: date={date} mu={mu} truth={truth} u={u}".format(
                        date=meta.get("date_column"),
                        mu=meta.get("mu_column"),
                        truth=meta.get("truth_column"),
                        u=meta.get("uncertainty_column"),
                    )
                )
                print(
                    f"  model_cols_pred={meta.get('model_columns_pred')} model_cols_ds={meta.get('model_columns_dataset')}"
                )
            except Exception as exc:
                print(f"  error: {exc}")
        return 0

    output_root = (
        Path(args.output_root)
        if args.output_root
        else input_root / "calibration_eval_2025"
    )
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = output_root / run_id
    ensure_dir(run_dir)
    log_path = run_dir / "run.log"

    def log_fn(message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp} {message}"
        print(line)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log_fn(f"Calibration run start: {run_id}")

    thresholds = DEFAULT_THRESHOLDS
    for exp in experiments:
        pred_path = exp.get("prediction_path")
        if pred_path is None or not Path(pred_path).exists():
            continue
        try:
            df, _ = prepare_experiment_data(exp["run_dir"], Path(pred_path))
        except Exception:
            continue
        y_vals = df.loc[df["date"].dt.year == args.year, "y_true"].to_numpy(
            dtype=float
        )
        y_vals = y_vals[np.isfinite(y_vals)]
        if y_vals.size > 0:
            thresholds = choose_thresholds(y_vals)
            break
    log_fn(f"Thresholds: {thresholds[:5]}..{thresholds[-5:]}")

    run_config = {
        "input_root": str(input_root),
        "sweep_json_used": str(sweep_json) if sweep_json else None,
        "truth_lag_L": args.truth_lag,
        "year_evaluated": args.year,
        "threshold_set": thresholds,
        "windows": args.windows,
        "sigma_min": args.sigma_min,
        "calibrator_grid": build_calibrator_grid(
            args.windows, args.sigma_min, emos_enabled=True
        ),
    }
    dump_json(run_dir / "run_config.json", run_config)

    leaderboard_rows: List[dict] = []
    best_overall: Optional[dict] = None
    for exp in experiments:
        log_fn(f"[EXP] {exp.get('experiment_id')}")
        result = run_experiment(
            exp,
            year=args.year,
            truth_lag=args.truth_lag,
            windows=args.windows,
            sigma_mins=args.sigma_min,
            thresholds=thresholds,
            output_root=run_dir,
            log_fn=log_fn,
            verbose=args.verbose,
        )
        if result is None:
            continue
        leaderboard_rows.append(result)
        if best_overall is None or (
            result.get("mean_crps") is not None
            and result.get("mean_crps") < best_overall.get("mean_crps", np.inf)
        ):
            best_overall = result

    if not leaderboard_rows:
        log_fn("No experiments produced results.")
        return 1

    leaderboard_df = pd.DataFrame(leaderboard_rows)
    leaderboard_df = leaderboard_df.sort_values("mean_crps")
    columns = [
        "experiment_id",
        "experiment_name",
        "best_calibrator_id",
        "mean_crps",
        "mean_nll",
        "mae_mu",
        "rmse_mu",
        "mean_sigma",
        "cov90",
        "cov90_err",
        "brier_mean",
    ]
    for col in columns:
        if col not in leaderboard_df.columns:
            leaderboard_df[col] = np.nan
    leaderboard_df = leaderboard_df[columns]
    leaderboard_df.to_csv(run_dir / "leaderboard_experiments.csv", index=False)

    if best_overall:
        best_payload = {
            "experiment_id": best_overall.get("experiment_id"),
            "calibrator_id": best_overall.get("best_calibrator_id"),
            "metrics": {
                "mean_crps": best_overall.get("mean_crps"),
                "mean_nll": best_overall.get("mean_nll"),
                "mae_mu": best_overall.get("mae_mu"),
                "rmse_mu": best_overall.get("rmse_mu"),
                "mean_sigma": best_overall.get("mean_sigma"),
                "cov90": best_overall.get("cov90"),
                "cov90_err": best_overall.get("cov90_err"),
                "brier_mean": best_overall.get("brier_mean"),
            },
            "paths": {
                "daily": best_overall.get("daily_path"),
                "metrics": best_overall.get("metrics_path"),
                "best_calibrator": best_overall.get("best_calibrator_path"),
            },
        }
        dump_json(run_dir / "best_overall.json", best_payload)

    summary_payload = build_run_summary(run_dir)
    dump_json(run_dir / "run_summary.json", summary_payload)

    log_fn("Calibration run complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
