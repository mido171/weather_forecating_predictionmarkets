"""Sigma chooser backtest (global vs regime-binned sigma)."""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import norm
except ImportError:  # pragma: no cover
    norm = None

from weather_ml import artifacts
from weather_ml import dataset
from weather_ml import models_mean
from weather_ml import utils_seed
from weather_ml import validate
from weather_ml import splits
from weather_ml import time_feature_library as tfl
from weather_ml import time_feature_sweep as tfs

LOGGER = logging.getLogger(__name__)

DEFAULT_VALIDATION = {
    "strict_schema": False,
    "forecast_min_f": -80.0,
    "forecast_max_f": 140.0,
    "spread_min_f": 0.0,
    "require_asof_not_after_target": True,
}

DEFAULT_BINNING = {
    "p_low": 1.0 / 3.0,
    "p_high": 2.0 / 3.0,
    "min_samples_per_bin": 60,
    "shrinkage": True,
}

DEFAULT_SCORING = {
    "interval_levels": [0.5, 0.8, 0.9, 0.95],
    "reliability_bins": 10,
    "threshold_min_offset": -5,
    "threshold_max_offset": 5,
    "bootstrap_samples": 1000,
}

DEFAULT_DETERMINISM = {
    "seed": 1337,
    "threads": 1,
}


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _parse_date(value: str, field_name: str) -> date:
    if not value:
        raise ValueError(f"Missing required date field: {field_name}")
    return datetime.strptime(value, "%Y-%m-%d").date()


def _parse_optional_date(value: str | None) -> date | None:
    if value in (None, "", "null"):
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(repo_root: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _default_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short = artifacts.sha256_bytes(ts.encode("utf-8"))[:8]
    return f"{ts}_{short}"


def _ensure_norm() -> None:
    if norm is None:
        raise ImportError("scipy is required for this experiment (scipy.stats.norm).")


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config: dict[str, Any] = dict(payload)
    config.setdefault("dataset_schema_version", 1)
    config.setdefault("station_filter", [])
    config.setdefault("asof_policy", {"mode": "latest"})
    config.setdefault("feature_set_name", "BASE")
    config.setdefault("truth_lag", 2)
    config.setdefault("r_feature", None)
    config.setdefault("binning", {})
    config.setdefault("scoring", {})
    config.setdefault("determinism", {})
    config.setdefault("validation", {})
    config.setdefault("output_root", "experiments/results/sigma_chooser_backtest")
    config.setdefault("overwrite", False)

    mean_model = config.get("mean_model") or {}
    mean_model.setdefault("name", "lgbm")
    mean_model.setdefault("params", {})
    config["mean_model"] = mean_model

    validation = DEFAULT_VALIDATION.copy()
    validation.update(config.get("validation") or {})
    config["validation"] = validation

    binning = DEFAULT_BINNING.copy()
    binning.update(config.get("binning") or {})
    config["binning"] = binning

    scoring = DEFAULT_SCORING.copy()
    scoring.update(config.get("scoring") or {})
    config["scoring"] = scoring

    determinism = DEFAULT_DETERMINISM.copy()
    determinism.update(config.get("determinism") or {})
    config["determinism"] = determinism

    return config

def _apply_station_filter(df: pd.DataFrame, station_filter: list[str]) -> pd.DataFrame:
    if not station_filter:
        return df
    return df[df["station_id"].isin(station_filter)].copy()


def _apply_asof_policy(df: pd.DataFrame, policy: dict[str, Any]) -> pd.DataFrame:
    mode = str(policy.get("mode", "latest")).lower()
    if mode in {"latest", "max"}:
        return (
            df.sort_values("asof_utc")
            .groupby(["station_id", "target_date_local"], as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )
    if mode in {"earliest", "min"}:
        return (
            df.sort_values("asof_utc")
            .groupby(["station_id", "target_date_local"], as_index=False)
            .head(1)
            .reset_index(drop=True)
        )
    if mode in {"latest_before", "fixed"}:
        time_str = policy.get("asof_time_utc")
        if not time_str:
            raise ValueError("asof_policy requires asof_time_utc for fixed/latest_before")
        cutoff = datetime.strptime(time_str, "%H:%M").time()

        def _pick_group(group: pd.DataFrame) -> pd.DataFrame:
            group = group.sort_values("asof_utc")
            asof_time = group["asof_utc"].dt.time
            if mode == "fixed":
                exact = group[asof_time == cutoff]
                if not exact.empty:
                    return exact.tail(1)
            before = group[asof_time <= cutoff]
            if not before.empty:
                return before.tail(1)
            LOGGER.warning(
                "No rows at/before cutoff for %s %s; falling back to latest asof.",
                group["station_id"].iloc[0],
                group["target_date_local"].iloc[0],
            )
            return group.tail(1)

        picked = (
            df.groupby(["station_id", "target_date_local"], group_keys=False)
            .apply(_pick_group)
            .reset_index(drop=True)
        )
        return picked
    raise ValueError(f"Unknown asof_policy mode: {mode}")


def _validate_dataset(df: pd.DataFrame, validation_cfg: dict[str, Any]) -> None:
    required_cols = [
        "station_id",
        "target_date_local",
        "asof_utc",
        "actual_tmax_f",
        *tfs.MODEL_COLS,
    ]
    if "gefsatmos_tmp_spread_f" in df.columns:
        required_cols.append("gefsatmos_tmp_spread_f")
    rules = validate.ValidationRules(
        required_columns=required_cols,
        allowed_columns=required_cols if validation_cfg.get("strict_schema") else None,
        forecast_min_f=float(validation_cfg.get("forecast_min_f")),
        forecast_max_f=float(validation_cfg.get("forecast_max_f")),
        spread_min_f=float(validation_cfg.get("spread_min_f")),
        require_asof_not_after_target=bool(
            validation_cfg.get("require_asof_not_after_target")
        ),
    )
    validate.run_all_validations(df, rules)


def _split_data(
    df: pd.DataFrame,
    *,
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date,
    val_start: date | None,
    val_end: date | None,
    validation_enabled: bool,
) -> splits.SplitResult:
    return splits.filter_date_ranges(
        df,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        gap_dates=[],
        val_start=val_start,
        val_end=val_end,
        validation_enabled=validation_enabled,
    )


def _find_experiment(feature_set_name: str) -> tfs.ExperimentDefinition | None:
    name = feature_set_name.upper()
    if name == "BASE":
        return None
    for exp in tfs._build_experiments():
        if exp.experiment_id.upper() == name:
            return exp
    raise ValueError(f"Unknown feature_set_name: {feature_set_name}")


def _build_feature_frame(
    df: pd.DataFrame,
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_set_name: str,
    truth_lag: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], tfs.DerivedFeatureSet]:
    df = tfl.prepare_frame(df)
    df = tfs._add_base_columns(df)
    base_cols = tfs._base_feature_columns(df)
    experiment = _find_experiment(feature_set_name)
    if experiment is None:
        derived = tfs.DerivedFeatureSet(
            features=pd.DataFrame(index=df.index),
            formulas=[],
            train_fitted=[],
        )
    else:
        context = tfs.ExperimentContext(
            df=df,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            group_key=df["station_id"],
            truth_lag=truth_lag,
            seed=seed,
        )
        derived = experiment.build_features(context)
    feature_df = pd.concat([df[base_cols].astype(float), derived.features], axis=1)
    feature_columns = base_cols + list(derived.features.columns)
    return df, feature_df, feature_columns, derived


def _select_r_feature(df: pd.DataFrame, requested: str | None) -> str:
    if requested and requested in df.columns:
        return requested
    if "gefsatmos_tmp_spread_f" in df.columns:
        return "gefsatmos_tmp_spread_f"
    if "ens_std" in df.columns:
        return "ens_std"
    raise ValueError("No valid r_feature available in dataset.")

def _fit_model(
    *,
    model_name: str,
    params: dict[str, Any],
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
):
    model = models_mean.get_mean_model(model_name, seed=seed)
    if params:
        model.set_params(**params)
    model.fit(X_train, y_train)
    return model


def _build_oof_predictions(
    *,
    df: pd.DataFrame,
    features: pd.DataFrame,
    feature_columns: list[str],
    train_mask: pd.Series,
    model_name: str,
    params: dict[str, Any],
    seed: int,
) -> pd.Series:
    years = sorted(df.loc[train_mask, "target_date_local"].dt.year.unique())
    oof = pd.Series(index=df.index, dtype=float)
    for fold_year in years:
        train_years = [year for year in years if year < fold_year]
        if not train_years:
            continue
        fold_mask = train_mask & (df["target_date_local"].dt.year == fold_year)
        if not fold_mask.any():
            continue
        train_fold_mask = train_mask & df["target_date_local"].dt.year.isin(train_years)
        X_train = features.loc[train_fold_mask, feature_columns].to_numpy(dtype=float)
        y_train = df.loc[train_fold_mask, "actual_tmax_f"].to_numpy(dtype=float)
        if len(y_train) == 0:
            continue
        model = _fit_model(
            model_name=model_name,
            params=params,
            seed=seed,
            X_train=X_train,
            y_train=y_train,
        )
        X_fold = features.loc[fold_mask, feature_columns].to_numpy(dtype=float)
        oof.loc[fold_mask] = model.predict(X_fold)
        LOGGER.info("OOF fold %s: trained on %s rows, predicted %s rows.", fold_year, len(y_train), len(X_fold))
    return oof


def _compute_residual_stats(residuals: np.ndarray, ddof: int) -> dict[str, Any]:
    residuals = np.asarray(residuals, dtype=float)
    n = int(residuals.size)
    if n <= ddof:
        raise ValueError(f"Not enough residuals for ddof={ddof}: n={n}")
    bias = float(np.mean(residuals))
    sigma = float(np.std(residuals, ddof=ddof))
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    quantiles = np.quantile(residuals, [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99])
    return {
        "n": n,
        "bias_mean_error_f": bias,
        "sigma_std_error_f": sigma,
        "mae_f": mae,
        "rmse_f": rmse,
        "residual_quantiles_f": {
            "p01": float(quantiles[0]),
            "p05": float(quantiles[1]),
            "p10": float(quantiles[2]),
            "p50": float(quantiles[3]),
            "p90": float(quantiles[4]),
            "p95": float(quantiles[5]),
            "p99": float(quantiles[6]),
        },
    }


def _assign_bins(values: np.ndarray, t1: float, t2: float) -> np.ndarray:
    bins = np.where(values <= t1, "LOW", np.where(values <= t2, "MID", "HIGH"))
    return bins.astype("U4")


def _shrink_stat(
    *,
    raw: float,
    global_value: float,
    n: int,
    min_samples: int,
) -> float:
    if n <= 0:
        return global_value
    if n >= min_samples:
        return raw
    weight = n / float(min_samples)
    return float(global_value * (1.0 - weight) + raw * weight)


def _bin_stats(
    residuals: np.ndarray,
    *,
    ddof: int,
    global_bias: float,
    global_sigma: float,
    min_samples: int,
    shrinkage: bool,
) -> dict[str, Any]:
    n = int(residuals.size)
    if n <= ddof:
        return {
            "n": n,
            "bias_mean_error_f": float(global_bias),
            "sigma_std_error_f": float(global_sigma),
            "shrunk": True,
        }
    bias = float(np.mean(residuals))
    sigma = float(np.std(residuals, ddof=ddof))
    if shrinkage:
        bias = _shrink_stat(
            raw=bias,
            global_value=global_bias,
            n=n,
            min_samples=min_samples,
        )
        sigma = _shrink_stat(
            raw=sigma,
            global_value=global_sigma,
            n=n,
            min_samples=min_samples,
        )
    return {
        "n": n,
        "bias_mean_error_f": float(bias),
        "sigma_std_error_f": float(sigma),
        "shrunk": bool(shrinkage and n < min_samples),
    }

def _normal_crps(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    sigma_floor: float,
) -> np.ndarray:
    _ensure_norm()
    sigma = np.maximum(sigma, sigma_floor)
    z = (y - mu) / sigma
    pdf = norm.pdf(z)
    cdf = norm.cdf(z)
    return sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))


def _normal_nll(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    sigma_floor: float,
) -> np.ndarray:
    sigma = np.maximum(sigma, sigma_floor)
    z = (y - mu) / sigma
    return 0.5 * np.log(2.0 * math.pi * sigma**2) + 0.5 * z**2


def _interval_stats(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    levels: list[float],
    sigma_floor: float,
) -> dict[str, dict[str, float]]:
    _ensure_norm()
    sigma = np.maximum(sigma, sigma_floor)
    stats: dict[str, dict[str, float]] = {}
    for level in levels:
        z = float(norm.ppf((1.0 + level) / 2.0))
        lower = mu - z * sigma
        upper = mu + z * sigma
        coverage = float(np.mean((y >= lower) & (y <= upper)))
        width = float(np.mean(upper - lower))
        stats[str(level)] = {
            "coverage": coverage,
            "avg_width": width,
        }
    return stats


def _threshold_scores(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    thresholds: np.ndarray,
    sigma_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _ensure_norm()
    sigma = np.maximum(sigma, sigma_floor)
    z = (thresholds[None, :] - mu[:, None]) / sigma[:, None]
    prob = 1.0 - norm.cdf(z)
    actual = (y[:, None] >= thresholds[None, :]).astype(float)
    brier = np.mean((prob - actual) ** 2, axis=0)
    return prob, actual, brier


def _reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    bins: int,
) -> pd.DataFrame:
    y_prob = np.clip(y_prob, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.digitize(y_prob, edges, right=True) - 1
    records = []
    for idx in range(bins):
        mask = bin_ids == idx
        if not mask.any():
            records.append({"bin": idx, "count": 0, "pred": float("nan"), "obs": float("nan")})
            continue
        records.append(
            {
                "bin": idx,
                "count": int(mask.sum()),
                "pred": float(np.mean(y_prob[mask])),
                "obs": float(np.mean(y_true[mask])),
            }
        )
    return pd.DataFrame.from_records(records)


def _summarize_distribution(
    *,
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    thresholds: np.ndarray,
    levels: list[float],
    sigma_floor: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    crps = _normal_crps(y, mu, sigma, sigma_floor=sigma_floor)
    nll = _normal_nll(y, mu, sigma, sigma_floor=sigma_floor)
    interval = _interval_stats(y, mu, sigma, levels=levels, sigma_floor=sigma_floor)
    prob, actual, brier = _threshold_scores(
        y, mu, sigma, thresholds=thresholds, sigma_floor=sigma_floor
    )
    brier_by_threshold = {
        str(int(thresholds[idx])): float(score) for idx, score in enumerate(brier)
    }
    summary = {
        "crps_mean": float(np.mean(crps)),
        "crps_median": float(np.median(crps)),
        "nll_mean": float(np.mean(nll)),
        "intervals": interval,
        "brier_mean": float(np.mean(brier)),
        "brier_by_threshold": brier_by_threshold,
    }
    return summary, crps, nll, prob


def _bootstrap_crps_delta(
    *,
    crps_global: np.ndarray,
    crps_chooser: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, float]:
    if samples <= 0:
        return {"mean": 0.0, "p025": 0.0, "p975": 0.0, "samples": 0}
    rng = np.random.default_rng(seed)
    n = len(crps_global)
    deltas = np.zeros(samples, dtype=float)
    for idx in range(samples):
        sample_idx = rng.integers(0, n, size=n)
        deltas[idx] = float(np.mean(crps_global[sample_idx] - crps_chooser[sample_idx]))
    return {
        "mean": float(np.mean(deltas)),
        "p025": float(np.percentile(deltas, 2.5)),
        "p975": float(np.percentile(deltas, 97.5)),
        "samples": int(samples),
    }


def _plot_coverage(
    path: Path,
    *,
    levels: list[float],
    global_stats: dict[str, dict[str, float]],
    chooser_stats: dict[str, dict[str, float]],
) -> None:
    x = np.array(levels)
    g = np.array([global_stats[str(level)]["coverage"] for level in levels])
    c = np.array([chooser_stats[str(level)]["coverage"] for level in levels])
    plt.figure(figsize=(6, 4))
    plt.plot(x, x, "k--", label="ideal")
    plt.plot(x, g, marker="o", label="global")
    plt.plot(x, c, marker="o", label="chooser")
    plt.xlabel("Nominal coverage")
    plt.ylabel("Empirical coverage")
    plt.title("Coverage vs nominal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_sharpness(
    path: Path,
    *,
    levels: list[float],
    global_stats: dict[str, dict[str, float]],
    chooser_stats: dict[str, dict[str, float]],
) -> None:
    x = np.array(levels)
    g = np.array([global_stats[str(level)]["avg_width"] for level in levels])
    c = np.array([chooser_stats[str(level)]["avg_width"] for level in levels])
    plt.figure(figsize=(6, 4))
    plt.plot(x, g, marker="o", label="global")
    plt.plot(x, c, marker="o", label="chooser")
    plt.xlabel("Nominal coverage")
    plt.ylabel("Average interval width")
    plt.title("Sharpness")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_reliability(
    path: Path,
    *,
    global_curve: pd.DataFrame,
    chooser_curve: pd.DataFrame,
) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot([0, 1], [0, 1], "k--", label="ideal")
    plt.plot(global_curve["pred"], global_curve["obs"], marker="o", label="global")
    plt.plot(chooser_curve["pred"], chooser_curve["obs"], marker="o", label="chooser")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Reliability curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_crps_timeseries(
    path: Path,
    *,
    dates: pd.Series,
    crps_global: np.ndarray,
    crps_chooser: np.ndarray,
) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(dates, crps_global, label="global", alpha=0.7)
    plt.plot(dates, crps_chooser, label="chooser", alpha=0.7)
    plt.xlabel("Date")
    plt.ylabel("CRPS")
    plt.title("Daily CRPS")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_pit_histogram(
    path: Path,
    *,
    pits_global: np.ndarray,
    pits_chooser: np.ndarray,
) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(pits_global, bins=20, alpha=0.5, label="global")
    plt.hist(pits_chooser, bins=20, alpha=0.5, label="chooser")
    plt.xlabel("PIT value")
    plt.ylabel("Count")
    plt.title("PIT histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def _breakdown_table(
    df: pd.DataFrame,
    *,
    group_col: str,
    mu_global: np.ndarray,
    sigma_global: np.ndarray,
    mu_chooser: np.ndarray,
    sigma_chooser: np.ndarray,
    thresholds: np.ndarray,
    levels: list[float],
    sigma_floor: float,
) -> pd.DataFrame:
    records = []
    for value, idx in df.groupby(group_col).groups.items():
        y = df.loc[idx, "actual_tmax_f"].to_numpy(dtype=float)
        g_summary, _, _, _ = _summarize_distribution(
            y=y,
            mu=mu_global[idx],
            sigma=sigma_global[idx],
            thresholds=thresholds,
            levels=levels,
            sigma_floor=sigma_floor,
        )
        c_summary, _, _, _ = _summarize_distribution(
            y=y,
            mu=mu_chooser[idx],
            sigma=sigma_chooser[idx],
            thresholds=thresholds,
            levels=levels,
            sigma_floor=sigma_floor,
        )
        record = {
            group_col: value,
            "n": int(len(y)),
            "crps_global": g_summary["crps_mean"],
            "crps_chooser": c_summary["crps_mean"],
            "nll_global": g_summary["nll_mean"],
            "nll_chooser": c_summary["nll_mean"],
            "brier_global": g_summary["brier_mean"],
            "brier_chooser": c_summary["brier_mean"],
        }
        for level in levels:
            key = str(level)
            record[f"coverage_global_{key}"] = g_summary["intervals"][key]["coverage"]
            record[f"coverage_chooser_{key}"] = c_summary["intervals"][key]["coverage"]
        records.append(record)
    return pd.DataFrame.from_records(records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sigma chooser backtest")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument("--run-id", help="Optional run id override")
    parser.add_argument("--output-root", help="Override output root directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    args = build_parser().parse_args(argv)
    config_path = Path(args.config)
    config = _load_config(config_path)
    repo_root = _resolve_repo_root()

    dataset_path = _resolve_path(repo_root, config["dataset_path"])
    output_root = _resolve_path(repo_root, args.output_root or config["output_root"])
    run_id = args.run_id or config.get("run_id") or _default_run_id()
    run_dir = output_root / run_id

    if run_dir.exists() and not config.get("overwrite"):
        raise FileExistsError(f"Run dir already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    utils_seed.set_global_determinism(
        int(config["determinism"]["seed"]),
        single_thread=True,
    )
    _ensure_norm()

    df = dataset.load_csv(dataset_path)
    _validate_dataset(df, config["validation"])

    df = _apply_asof_policy(df, config.get("asof_policy", {}))
    df = _apply_station_filter(df, list(config.get("station_filter") or []))

    if df.empty:
        raise ValueError("Dataset empty after station filter / asof policy.")

    df = tfl.prepare_frame(df)
    df = tfs._add_base_columns(df)

    train_start = _parse_date(config["train_start"], "train_start")
    train_end = _parse_date(config["train_end"], "train_end")
    test_start = _parse_date(config["test_start"], "test_start")
    test_end = _parse_date(config["test_end"], "test_end")
    val_start = _parse_optional_date(config.get("validation", {}).get("val_start"))
    val_end = _parse_optional_date(config.get("validation", {}).get("val_end"))
    val_enabled = bool(config.get("validation", {}).get("enabled", False))

    split = _split_data(
        df,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        val_start=val_start,
        val_end=val_end,
        validation_enabled=val_enabled,
    )

    train_df = split.train_df
    val_df = split.val_df
    test_df = split.test_df
    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split is empty.")

    df_base, feature_df, feature_columns, derived = _build_feature_frame(
        df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_set_name=str(config["feature_set_name"]),
        truth_lag=int(config["truth_lag"]),
        seed=int(config["determinism"]["seed"]),
    )

    train_df = df_base.loc[train_df.index]
    val_df = df_base.loc[val_df.index]
    test_df = df_base.loc[test_df.index]

    train_full_idx = train_df.index.union(val_df.index)
    train_full_df = df_base.loc[train_full_idx]

    imputed, impute_meta = tfs._impute_features(feature_df, train_full_idx)

    X_train_full = imputed.loc[train_full_idx, feature_columns].to_numpy(dtype=float)
    y_train_full = train_full_df["actual_tmax_f"].to_numpy(dtype=float)

    model_name = str(config["mean_model"]["name"])
    model_params = dict(config["mean_model"].get("params") or {})
    seed = int(config["determinism"]["seed"])

    model_full = _fit_model(
        model_name=model_name,
        params=model_params,
        seed=seed,
        X_train=X_train_full,
        y_train=y_train_full,
    )

    mu_train = model_full.predict(
        imputed.loc[train_df.index, feature_columns].to_numpy(dtype=float)
    )
    mu_val = (
        model_full.predict(
            imputed.loc[val_df.index, feature_columns].to_numpy(dtype=float)
        )
        if not val_df.empty
        else np.array([])
    )

    train_metrics = {
        "mae": float(np.mean(np.abs(mu_train - train_df["actual_tmax_f"]))),
        "rmse": float(np.sqrt(np.mean((mu_train - train_df["actual_tmax_f"]) ** 2))),
        "bias": float(np.mean(mu_train - train_df["actual_tmax_f"])),
        "n": int(len(mu_train)),
    }
    val_metrics = None
    if not val_df.empty:
        val_metrics = {
            "mae": float(np.mean(np.abs(mu_val - val_df["actual_tmax_f"]))),
            "rmse": float(np.sqrt(np.mean((mu_val - val_df["actual_tmax_f"]) ** 2))),
            "bias": float(np.mean(mu_val - val_df["actual_tmax_f"])),
            "n": int(len(mu_val)),
        }

    train_mask = df_base.index.isin(train_full_idx)
    oof = _build_oof_predictions(
        df=df_base,
        features=imputed,
        feature_columns=feature_columns,
        train_mask=train_mask,
        model_name=model_name,
        params=model_params,
        seed=seed,
    )
    cal_mask = train_mask & oof.notna()
    cal_df = df_base.loc[cal_mask].copy()
    if cal_df.empty:
        raise ValueError("No OOF predictions available for calibration.")
    mu_oof = oof.loc[cal_mask].to_numpy(dtype=float)
    residuals = cal_df["actual_tmax_f"].to_numpy(dtype=float) - mu_oof

    ddof = int(config.get("ddof", 1))
    residual_stats = _compute_residual_stats(residuals, ddof=ddof)
    global_bias = residual_stats["bias_mean_error_f"]
    global_sigma = residual_stats["sigma_std_error_f"]

    r_feature = _select_r_feature(df_base, config.get("r_feature"))
    r_values_cal = cal_df[r_feature].to_numpy(dtype=float)
    if np.isnan(r_values_cal).any():
        med = float(np.nanmedian(r_values_cal))
        r_values_cal = np.nan_to_num(r_values_cal, nan=med)
        LOGGER.warning("Imputed NaN r_feature values in calibration to %s", med)
    p_low = float(config["binning"]["p_low"])
    p_high = float(config["binning"]["p_high"])
    t1 = float(np.quantile(r_values_cal, p_low))
    t2 = float(np.quantile(r_values_cal, p_high))
    if not t1 < t2:
        raise ValueError("Invalid r_feature quantile thresholds.")
    bins_cal = _assign_bins(r_values_cal, t1, t2)

    min_samples = int(config["binning"]["min_samples_per_bin"])
    shrinkage = bool(config["binning"]["shrinkage"])
    bin_stats = {}
    for label in ["LOW", "MID", "HIGH"]:
        mask = bins_cal == label
        stats = _bin_stats(
            residuals[mask],
            ddof=ddof,
            global_bias=global_bias,
            global_sigma=global_sigma,
            min_samples=min_samples,
            shrinkage=shrinkage,
        )
        bin_stats[label] = stats

    sigma_floor = float(config.get("sigma_floor", 0.25))

    X_test = imputed.loc[test_df.index, feature_columns].to_numpy(dtype=float)
    mu_test = model_full.predict(X_test)
    y_test = test_df["actual_tmax_f"].to_numpy(dtype=float)

    r_values_test = test_df[r_feature].to_numpy(dtype=float)
    if np.isnan(r_values_test).any():
        med = float(np.nanmedian(r_values_cal))
        r_values_test = np.nan_to_num(r_values_test, nan=med)
        LOGGER.warning("Imputed NaN r_feature values in test to %s", med)
    bins_test = _assign_bins(r_values_test, t1, t2)

    mu_adj_global = mu_test + global_bias
    sigma_global = np.full_like(mu_test, global_sigma, dtype=float)

    bias_by_bin = np.array([bin_stats[label]["bias_mean_error_f"] for label in bins_test])
    sigma_by_bin = np.array([bin_stats[label]["sigma_std_error_f"] for label in bins_test])
    mu_adj_chooser = mu_test + bias_by_bin
    sigma_chooser = sigma_by_bin

    thresholds = np.arange(
        int(np.floor(y_test.min()) + config["scoring"]["threshold_min_offset"]),
        int(np.ceil(y_test.max()) + config["scoring"]["threshold_max_offset"]) + 1,
    )

    global_summary, crps_global, nll_global, prob_global = _summarize_distribution(
        y=y_test,
        mu=mu_adj_global,
        sigma=sigma_global,
        thresholds=thresholds,
        levels=list(config["scoring"]["interval_levels"]),
        sigma_floor=sigma_floor,
    )
    chooser_summary, crps_chooser, nll_chooser, prob_chooser = _summarize_distribution(
        y=y_test,
        mu=mu_adj_chooser,
        sigma=sigma_chooser,
        thresholds=thresholds,
        levels=list(config["scoring"]["interval_levels"]),
        sigma_floor=sigma_floor,
    )

    bootstrap = _bootstrap_crps_delta(
        crps_global=crps_global,
        crps_chooser=crps_chooser,
        samples=int(config["scoring"]["bootstrap_samples"]),
        seed=seed,
    )

    prob_global_flat = prob_global.ravel()
    prob_chooser_flat = prob_chooser.ravel()
    actual_flat = (y_test[:, None] >= thresholds[None, :]).astype(float).ravel()
    reliability_global = _reliability_curve(
        actual_flat, prob_global_flat, bins=int(config["scoring"]["reliability_bins"])
    )
    reliability_chooser = _reliability_curve(
        actual_flat, prob_chooser_flat, bins=int(config["scoring"]["reliability_bins"])
    )

    scores_summary = {
        "point": {
            "mae": float(np.mean(np.abs(mu_test - y_test))),
            "rmse": float(np.sqrt(np.mean((mu_test - y_test) ** 2))),
            "bias": float(np.mean(mu_test - y_test)),
        },
        "global": global_summary,
        "chooser": chooser_summary,
        "delta": {
            "crps_mean": float(global_summary["crps_mean"] - chooser_summary["crps_mean"]),
            "nll_mean": float(global_summary["nll_mean"] - chooser_summary["nll_mean"]),
            "brier_mean": float(global_summary["brier_mean"] - chooser_summary["brier_mean"]),
        },
        "crps_delta_bootstrap": bootstrap,
    }

    predictions = test_df[["station_id", "target_date_local", "asof_utc"]].copy()
    predictions["mu_hat_f"] = mu_test
    predictions["actual_tmax_f"] = y_test
    predictions["r_value"] = r_values_test
    predictions["bin_assigned"] = bins_test
    predictions["mu_adj_global"] = mu_adj_global
    predictions["sigma_global"] = sigma_global
    predictions["mu_adj_chooser"] = mu_adj_chooser
    predictions["sigma_chooser"] = sigma_chooser
    predictions["residual_global"] = y_test - mu_adj_global
    predictions["residual_chooser"] = y_test - mu_adj_chooser
    predictions = predictions.reset_index(drop=True)

    breakdown_bin = _breakdown_table(
        predictions,
        group_col="bin_assigned",
        mu_global=mu_adj_global,
        sigma_global=sigma_global,
        mu_chooser=mu_adj_chooser,
        sigma_chooser=sigma_chooser,
        thresholds=thresholds,
        levels=list(config["scoring"]["interval_levels"]),
        sigma_floor=sigma_floor,
    )

    predictions["month"] = pd.to_datetime(predictions["target_date_local"]).dt.month
    breakdown_month = _breakdown_table(
        predictions,
        group_col="month",
        mu_global=mu_adj_global,
        sigma_global=sigma_global,
        mu_chooser=mu_adj_chooser,
        sigma_chooser=sigma_chooser,
        thresholds=thresholds,
        levels=list(config["scoring"]["interval_levels"]),
        sigma_floor=sigma_floor,
    )

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    _plot_coverage(
        plots_dir / "coverage_comparison.png",
        levels=list(config["scoring"]["interval_levels"]),
        global_stats=global_summary["intervals"],
        chooser_stats=chooser_summary["intervals"],
    )
    _plot_sharpness(
        plots_dir / "sharpness_comparison.png",
        levels=list(config["scoring"]["interval_levels"]),
        global_stats=global_summary["intervals"],
        chooser_stats=chooser_summary["intervals"],
    )
    _plot_reliability(
        plots_dir / "reliability_thresholds.png",
        global_curve=reliability_global,
        chooser_curve=reliability_chooser,
    )
    _plot_crps_timeseries(
        plots_dir / "crps_timeseries.png",
        dates=predictions["target_date_local"],
        crps_global=crps_global,
        crps_chooser=crps_chooser,
    )
    pits_global = norm.cdf((y_test - mu_adj_global) / np.maximum(sigma_global, sigma_floor))
    pits_chooser = norm.cdf((y_test - mu_adj_chooser) / np.maximum(sigma_chooser, sigma_floor))
    _plot_pit_histogram(
        plots_dir / "pit_histogram.png",
        pits_global=pits_global,
        pits_chooser=pits_chooser,
    )

    mean_dir = run_dir / "mean_model_artifacts"
    mean_dir.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(model_full, mean_dir / "mean_model.joblib")
    (mean_dir / "feature_list.json").write_text(
        json.dumps(feature_columns, indent=2, sort_keys=True), encoding="utf-8"
    )
    (mean_dir / "feature_state.json").write_text(
        json.dumps(impute_meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    (mean_dir / "train_metrics.json").write_text(
        json.dumps(train_metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    if val_metrics:
        (mean_dir / "val_metrics.json").write_text(
            json.dumps(val_metrics, indent=2, sort_keys=True), encoding="utf-8"
        )

    calibration_residuals = cal_df[["station_id", "target_date_local", "asof_utc"]].copy()
    calibration_residuals["mu_hat_oof"] = mu_oof
    calibration_residuals["actual_tmax_f"] = cal_df["actual_tmax_f"].to_numpy(dtype=float)
    calibration_residuals["residual_f"] = residuals
    calibration_residuals["r_value"] = r_values_cal

    dataset_hash = artifacts.sha256_file(dataset_path)
    dataset_id = artifacts.compute_dataset_id(
        dataset_path,
        int(config["dataset_schema_version"]),
        {"feature_set_name": config["feature_set_name"], "feature_columns": feature_columns},
    )
    model_hash = artifacts.sha256_file(mean_dir / "mean_model.joblib")

    calibration_global = {
        "method": "global_normal_residual",
        "error_definition": "e = actual_tmax_f - mu_hat_oof",
        "ddof": ddof,
        "n": residual_stats["n"],
        "bias_mean_error_f": residual_stats["bias_mean_error_f"],
        "sigma_std_error_f": residual_stats["sigma_std_error_f"],
        "mae_f": residual_stats["mae_f"],
        "rmse_f": residual_stats["rmse_f"],
        "residual_quantiles_f": residual_stats["residual_quantiles_f"],
        "calibration_window": {
            "start": train_start.isoformat(),
            "end": train_end.isoformat(),
        },
        "station_scope": {
            "mode": "LIST" if config.get("station_filter") else "ALL",
            "stations": list(config.get("station_filter") or []),
        },
        "model_ref": {
            "model_path": str(mean_dir / "mean_model.joblib"),
            "model_hash": model_hash,
            "model_name": model_name,
            "model_params": model_params,
        },
        "dataset_ref": {
            "dataset_id": dataset_id,
            "dataset_hash": dataset_hash,
            "rows_used": residual_stats["n"],
        },
        "created_utc": artifacts.utc_now_iso(),
    }

    calibration_sigma = {
        "method": "sigma_chooser",
        "r_feature": r_feature,
        "r_thresholds": {
            "p_low": p_low,
            "p_high": p_high,
            "t_low": t1,
            "t_high": t2,
        },
        "bins": bin_stats,
        "global_fallback": {
            "bias_mean_error_f": global_bias,
            "sigma_std_error_f": global_sigma,
        },
        "min_samples_per_bin": min_samples,
        "shrinkage": shrinkage,
        "calibration_window": {
            "start": train_start.isoformat(),
            "end": train_end.isoformat(),
        },
        "created_utc": artifacts.utc_now_iso(),
    }

    data_meta = {
        "rows": int(len(df_base)),
        "stations": sorted(df_base["station_id"].dropna().unique().tolist()),
        "min_date": str(df_base["target_date_local"].min()),
        "max_date": str(df_base["target_date_local"].max()),
        "rows_by_year": df_base["target_date_local"].dt.year.value_counts().to_dict(),
        "missingness": df_base.isna().sum().to_dict(),
    }

    config_resolved = {
        "run_id": run_id,
        "dataset_path": str(dataset_path),
        "dataset_hash": dataset_hash,
        "dataset_id": dataset_id,
        "feature_set_name": config["feature_set_name"],
        "feature_columns": feature_columns,
        "r_feature": r_feature,
        "mean_model": {"name": model_name, "params": model_params},
        "determinism": config["determinism"],
        "train_start": train_start.isoformat(),
        "train_end": train_end.isoformat(),
        "test_start": test_start.isoformat(),
        "test_end": test_end.isoformat(),
        "binning": config["binning"],
        "scoring": config["scoring"],
        "truth_lag": int(config["truth_lag"]),
        "sigma_floor": sigma_floor,
        "config_source": str(config_path),
        "derived_meta": {
            "formulas": list(derived.formulas),
            "train_fitted": list(derived.train_fitted),
            "imputation": impute_meta,
        },
    }

    (run_dir / "config_resolved.json").write_text(
        json.dumps(config_resolved, indent=2, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "data_snapshot_meta.json").write_text(
        json.dumps(data_meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    calibration_residuals.to_csv(run_dir / "calibration_residuals.csv", index=False)
    (run_dir / "calibration_global.json").write_text(
        json.dumps(calibration_global, indent=2, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "calibration_sigma_chooser.json").write_text(
        json.dumps(calibration_sigma, indent=2, sort_keys=True), encoding="utf-8"
    )
    predictions.to_csv(run_dir / "predictions_2025.csv", index=False)
    (run_dir / "scores_summary.json").write_text(
        json.dumps(scores_summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    breakdown_bin.to_csv(run_dir / "scores_breakdown_by_bin.csv", index=False)
    breakdown_month.to_csv(run_dir / "scores_breakdown_by_month.csv", index=False)

    hash_paths = [
        run_dir / "config_resolved.json",
        run_dir / "data_snapshot_meta.json",
        run_dir / "calibration_residuals.csv",
        run_dir / "calibration_global.json",
        run_dir / "calibration_sigma_chooser.json",
        run_dir / "predictions_2025.csv",
        run_dir / "scores_summary.json",
        run_dir / "scores_breakdown_by_bin.csv",
        run_dir / "scores_breakdown_by_month.csv",
        mean_dir / "mean_model.joblib",
        mean_dir / "feature_list.json",
        mean_dir / "feature_state.json",
        mean_dir / "train_metrics.json",
    ]
    if val_metrics:
        hash_paths.append(mean_dir / "val_metrics.json")
    hash_paths.extend(plots_dir.glob("*.png"))
    artifacts.write_hash_manifest(hash_paths, run_dir / "hashes.json")

    LOGGER.info("Sigma chooser backtest complete. Output: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
