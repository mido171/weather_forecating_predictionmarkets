"""Training + evaluation for KMIA Kalshi Tmax pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from . import metrics


@dataclass(frozen=True)
class KalshiTrainConfig:
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    val_start: date | None = None
    val_end: date | None = None
    n_splits: int = 5
    gap_days: int = 2
    recency_lambda: float = 0.0
    lgbm_params: dict[str, Any] | None = None
    quantile_params: dict[str, Any] | None = None
    quantiles: list[float] | None = None
    probability_events: list[dict[str, Any]] | None = None
    stacking: bool = True


def load_train_config(path: str | Path) -> KalshiTrainConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return _parse_train_config(raw)


def _parse_train_config(raw: dict[str, Any]) -> KalshiTrainConfig:
    def parse_date(value: str | None) -> date | None:
        if value is None:
            return None
        return pd.to_datetime(value).date()

    return KalshiTrainConfig(
        train_start=parse_date(raw["train_start"]),
        train_end=parse_date(raw["train_end"]),
        test_start=parse_date(raw["test_start"]),
        test_end=parse_date(raw["test_end"]),
        val_start=parse_date(raw.get("val_start")),
        val_end=parse_date(raw.get("val_end")),
        n_splits=int(raw.get("n_splits", 5)),
        gap_days=int(raw.get("gap_days", 2)),
        recency_lambda=float(raw.get("recency_lambda", 0.0)),
        lgbm_params=raw.get("lgbm_params", {}),
        quantile_params=raw.get("quantile_params", {}),
        quantiles=list(raw.get("quantiles", [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])),
        probability_events=list(raw.get("probability_events", [])),
        stacking=bool(raw.get("stacking", True)),
    )


def train_and_evaluate(
    df: pd.DataFrame,
    cfg: KalshiTrainConfig,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    df = df[df["y_actual_tmax_f"].notna()].copy()

    train_df = _split_df(df, cfg.train_start, cfg.train_end)
    val_df = _split_df(df, cfg.val_start, cfg.val_end) if cfg.val_start and cfg.val_end else pd.DataFrame()
    test_df = _split_df(df, cfg.test_start, cfg.test_end)

    if train_df.empty:
        raise ValueError("Training split is empty.")
    if test_df.empty:
        raise ValueError("Test split is empty.")

    feature_cols = _select_feature_columns(df)
    baseline_train, baseline_name = _baseline_series(train_df)
    baseline_val, _ = _baseline_series(val_df) if not val_df.empty else (pd.Series(dtype=float), "")
    baseline_test, _ = _baseline_series(test_df)

    y_train = train_df["y_actual_tmax_f"].to_numpy(dtype=float)
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    train_df, X_train, y_train, baseline_train = _filter_valid_split(
        train_df, X_train, y_train, baseline_train, split_name="train"
    )

    if not val_df.empty:
        y_val = val_df["y_actual_tmax_f"].to_numpy(dtype=float)
        X_val = val_df[feature_cols].to_numpy(dtype=float)
        val_df, X_val, y_val, baseline_val = _filter_valid_split(
            val_df, X_val, y_val, baseline_val, split_name="val"
        )
    else:
        y_val = np.array([], dtype=float)
        X_val = np.zeros((0, len(feature_cols)), dtype=float)

    y_test = test_df["y_actual_tmax_f"].to_numpy(dtype=float)
    X_test = test_df[feature_cols].to_numpy(dtype=float)
    test_df, X_test, y_test, baseline_test = _filter_valid_split(
        test_df, X_test, y_test, baseline_test, split_name="test"
    )

    weights = _recency_weights(train_df, cfg) if cfg.recency_lambda > 0 else None

    lgbm_model = _fit_lgbm_residual_model(
        X_train,
        y_train,
        baseline_train.to_numpy(dtype=float),
        params=cfg.lgbm_params or {},
        sample_weight=weights,
    )
    resid_pred_train = lgbm_model.predict(X_train)
    resid_pred_test = lgbm_model.predict(X_test)
    pred_train = baseline_train.to_numpy(dtype=float) + resid_pred_train
    pred_test = baseline_test.to_numpy(dtype=float) + resid_pred_test
    if len(X_val) > 0:
        resid_pred_val = lgbm_model.predict(X_val)
        pred_val = baseline_val.to_numpy(dtype=float) + resid_pred_val
    else:
        pred_val = np.array([], dtype=float)

    model_point_metrics = {
        "train": metrics.regression_metrics(y_train, pred_train),
        "test": metrics.regression_metrics(y_test, pred_test),
    }
    if pred_val.size > 0:
        model_point_metrics["val"] = metrics.regression_metrics(y_val, pred_val)

    baseline_point_metrics = {
        "train": metrics.regression_metrics(y_train, baseline_train.to_numpy(dtype=float)),
        "test": metrics.regression_metrics(y_test, baseline_test.to_numpy(dtype=float)),
    }
    if len(y_val) > 0:
        baseline_point_metrics["val"] = metrics.regression_metrics(
            y_val, baseline_val.to_numpy(dtype=float)
        )

    skill_test = 1.0 - (model_point_metrics["test"]["mae"] / baseline_point_metrics["test"]["mae"])
    skill_val = None
    if "val" in model_point_metrics and "val" in baseline_point_metrics:
        skill_val = 1.0 - (model_point_metrics["val"]["mae"] / baseline_point_metrics["val"]["mae"])

    quantile_preds_by_split = _fit_quantile_models_multi(
        X_train=X_train,
        y_train=y_train,
        baseline_train=baseline_train.to_numpy(dtype=float),
        eval_splits={
            "test": (X_test, baseline_test.to_numpy(dtype=float)),
            "val": (X_val, baseline_val.to_numpy(dtype=float)) if len(X_val) > 0 else None,
        },
        params=cfg.quantile_params or {},
        quantiles=cfg.quantiles or [],
        sample_weight=weights,
    )

    model_quantile_metrics_by_split = {}
    model_event_metrics_by_split = {}
    for split_name, q_preds in quantile_preds_by_split.items():
        y_split = y_test if split_name == "test" else y_val
        model_quantile_metrics_by_split[split_name] = _quantile_metrics(
            y_split, q_preds, quantiles=cfg.quantiles or []
        )
        model_event_metrics_by_split[split_name] = _probabilistic_event_metrics(
            y_split,
            q_preds,
            quantiles=cfg.quantiles or [],
            event_specs=cfg.probability_events or [],
        )

    baseline_prob = _baseline_probabilistic_suite(
        y_train=y_train,
        baseline_train=baseline_train.to_numpy(dtype=float),
        y_by_split={
            "test": y_test,
            "val": y_val if len(y_val) > 0 else None,
        },
        baseline_by_split={
            "test": baseline_test.to_numpy(dtype=float),
            "val": baseline_val.to_numpy(dtype=float) if len(baseline_val) > 0 else None,
        },
        quantiles=cfg.quantiles or [],
        event_specs=cfg.probability_events or [],
    )

    diff = _diff_summary(
        baseline_point=baseline_point_metrics,
        model_point=model_point_metrics,
        baseline_prob=baseline_prob,
        model_prob={
            "quantiles": model_quantile_metrics_by_split,
            "events": model_event_metrics_by_split,
        },
    )

    stack_metrics = None
    stack_predictions = None
    if cfg.stacking:
        stack_predictions, stack_metrics = _stacking_evaluate(
            train_df,
            test_df,
            feature_cols=feature_cols,
            baseline_name=baseline_name,
            baseline_train=baseline_train,
            baseline_test=baseline_test,
            lgbm_params=cfg.lgbm_params or {},
            n_splits=cfg.n_splits,
            gap_days=cfg.gap_days,
        )

    payload = {
        "splits": {
            "train": {"start": str(cfg.train_start), "end": str(cfg.train_end), "rows": int(len(train_df))},
            "val": {"start": str(cfg.val_start), "end": str(cfg.val_end), "rows": int(len(val_df))},
            "test": {"start": str(cfg.test_start), "end": str(cfg.test_end), "rows": int(len(test_df))},
        },
        "feature_count": len(feature_cols),
        "baseline_name": baseline_name,
        "metrics": {
            "baseline": {
                "point": baseline_point_metrics,
                "probabilistic": baseline_prob,
            },
            "model": {
                "point": model_point_metrics,
                "quantiles": model_quantile_metrics_by_split,
                "probabilistic_events": model_event_metrics_by_split,
            },
            "diff_model_minus_baseline": diff,
            "skill_score_test": skill_test,
            "skill_score_val": skill_val,
            "stacking": stack_metrics,
        },
    }

    _write_json(output_dir / "metrics.json", payload)
    _write_markdown(output_dir / "metrics.md", payload)
    _write_predictions(
        output_dir / "predictions_test.csv",
        test_df,
        y_test,
        baseline_test.to_numpy(dtype=float),
        pred_test,
        stack_predictions,
        quantile_preds_by_split.get("test", {}),
        cfg.quantiles or [],
    )
    return payload


def _select_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "y_actual_tmax_f",
        "station_id",
        "station_zoneid",
        "asof_date_local",
        "target_date_local",
        "asof_utc",
        "feature_version",
        "config_hash",
        "knn_config_hash",
        "sql_extract_hash_mos",
        "sql_extract_hash_truth",
        "raw_payload_hash_ref_agg",
        "mos_retrieved_at_utc_max",
        "mos_runtime_utc_max",
    }
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in exclude]


def _baseline_series(df: pd.DataFrame) -> tuple[pd.Series, str]:
    if "base_tmax_blend" in df.columns:
        baseline = pd.to_numeric(df["base_tmax_blend"], errors="coerce")
        bias_cols = ["bias_blend_mean_30", "bias_blend_mean_60", "bias_blend_mean_90"]
        for col in bias_cols:
            if col in df.columns:
                return baseline + pd.to_numeric(df[col], errors="coerce"), f"base_tmax_blend+{col}"
        return baseline, "base_tmax_blend"
    for col in ["mos_gfs_n_x_max", "mos_nam_n_x_max", "mos_gfs_tmp_max", "mos_nam_tmp_max"]:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce"), col
    raise ValueError("No baseline column found (expected base_tmax_blend or mos_* tmp/n_x).")


def _filter_valid_split(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    baseline: pd.Series,
    *,
    split_name: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.Series]:
    baseline_np = baseline.to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(baseline_np)
    if not np.any(mask):
        raise ValueError(f"No valid rows in split={split_name} after baseline filtering.")
    df_out = df.iloc[mask].reset_index(drop=True)
    X_out = X[mask]
    y_out = y[mask]
    baseline_out = baseline.iloc[mask].reset_index(drop=True)
    return df_out, X_out, y_out, baseline_out


def _recency_weights(df: pd.DataFrame, cfg: KalshiTrainConfig) -> np.ndarray:
    ages = (pd.to_datetime(cfg.train_end) - pd.to_datetime(df["target_date_local"])).dt.days
    return np.exp(-cfg.recency_lambda * ages.to_numpy(dtype=float))


def _fit_lgbm_residual_model(
    X: np.ndarray,
    y: np.ndarray,
    baseline: np.ndarray,
    *,
    params: dict[str, Any],
    sample_weight: np.ndarray | None,
):
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("lightgbm is required for Kalshi Tmax training.") from exc
    target = y - baseline
    model = lgb.LGBMRegressor(objective="mae", **params)
    model.fit(X, target, sample_weight=sample_weight)
    return model


def _fit_quantile_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    baseline_train: np.ndarray,
    X_test: np.ndarray,
    baseline_test: np.ndarray,
    *,
    params: dict[str, Any],
    quantiles: list[float],
    sample_weight: np.ndarray | None,
) -> dict[str, np.ndarray]:
    if not quantiles:
        return {}
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("lightgbm is required for quantile models.") from exc
    preds: dict[str, np.ndarray] = {}
    target = y_train - baseline_train
    for q in quantiles:
        model = lgb.LGBMRegressor(objective="quantile", alpha=q, **params)
        model.fit(X_train, target, sample_weight=sample_weight)
        q_pred = model.predict(X_test) + baseline_test
        preds[f"q{int(q*100):02d}"] = q_pred
    # Enforce monotonicity by sorting per row.
    if preds:
        keys = sorted(preds.keys())
        stack = np.vstack([preds[k] for k in keys])
        stack_sorted = np.sort(stack, axis=0)
        for idx, key in enumerate(keys):
            preds[key] = stack_sorted[idx, :]
    return preds


def _fit_quantile_models_multi(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    baseline_train: np.ndarray,
    eval_splits: dict[str, tuple[np.ndarray, np.ndarray] | None],
    params: dict[str, Any],
    quantiles: list[float],
    sample_weight: np.ndarray | None,
) -> dict[str, dict[str, np.ndarray]]:
    if not quantiles:
        return {}
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("lightgbm is required for quantile models.") from exc

    target = y_train - baseline_train
    split_preds: dict[str, dict[str, np.ndarray]] = {
        name: {} for name, payload in eval_splits.items() if payload is not None
    }
    for q in sorted(quantiles):
        model = lgb.LGBMRegressor(objective="quantile", alpha=q, **(params or {}))
        model.fit(X_train, target, sample_weight=sample_weight)
        key = f"q{int(q*100):02d}"
        for split_name, payload in eval_splits.items():
            if payload is None:
                continue
            X_eval, baseline_eval = payload
            split_preds[split_name][key] = model.predict(X_eval) + baseline_eval

    # Enforce monotonicity per row per split.
    keys = [f"q{int(q*100):02d}" for q in sorted(quantiles)]
    for split_name, preds in split_preds.items():
        if not preds:
            continue
        stack = np.vstack([preds[k] for k in keys])
        stack_sorted = np.sort(stack, axis=0)
        for idx, key in enumerate(keys):
            preds[key] = stack_sorted[idx, :]
    return split_preds


def _quantile_metrics(
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    *,
    quantiles: list[float],
) -> dict[str, Any]:
    if not preds:
        return {}
    results = {"pinball": {}, "coverage": {}, "interval_width": {}}
    for q in quantiles:
        key = f"q{int(q*100):02d}"
        if key not in preds:
            continue
        results["pinball"][key] = float(_pinball_loss(y_true, preds[key], q))
    # Coverage for central intervals if present
    for low, high, name in [(0.1, 0.9, "p80"), (0.05, 0.95, "p90"), (0.25, 0.75, "p50")]:
        low_key = f"q{int(low*100):02d}"
        high_key = f"q{int(high*100):02d}"
        if low_key in preds and high_key in preds:
            within = (y_true >= preds[low_key]) & (y_true <= preds[high_key])
            results["coverage"][name] = float(np.mean(within))
            results["interval_width"][name] = float(np.nanmean(preds[high_key] - preds[low_key]))
    if results["pinball"]:
        results["crps_approx"] = float(2.0 * np.mean(list(results["pinball"].values())))
    return results


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def _probabilistic_event_metrics(
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    *,
    quantiles: list[float],
    event_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    if not preds or not event_specs or not quantiles:
        return {}
    quantiles_sorted = sorted(quantiles)
    q_keys = [f"q{int(q*100):02d}" for q in quantiles_sorted]
    if any(key not in preds for key in q_keys):
        return {}
    q_values = np.vstack([preds[key] for key in q_keys])
    results = {"brier": {}, "log_loss": {}}
    for spec in event_specs:
        name = spec.get("name") or _event_name(spec)
        probs = _event_probability_from_quantiles(q_values, quantiles_sorted, spec)
        y_event = metrics.event_indicator(y_true, spec).astype(int)
        results["brier"][name] = float(metrics.brier_score(y_event, probs))
        results["log_loss"][name] = float(_log_loss_binary(y_event, probs))
    return results


def _event_probability_from_quantiles(
    q_values: np.ndarray,
    quantiles: list[float],
    spec: dict[str, Any],
) -> np.ndarray:
    if spec.get("type") == "threshold":
        if "lt" in spec:
            return _cdf_from_quantiles(q_values, quantiles, float(spec["lt"]))
        if "ge" in spec:
            return 1.0 - _cdf_from_quantiles(q_values, quantiles, float(spec["ge"]))
    if spec.get("type") == "range":
        start = float(spec["start"])
        end = float(spec["end"])
        return _cdf_from_quantiles(q_values, quantiles, end) - _cdf_from_quantiles(q_values, quantiles, start)
    raise ValueError(f"Unsupported event spec: {spec}")


def _cdf_from_quantiles(q_values: np.ndarray, quantiles: list[float], threshold: float) -> np.ndarray:
    # q_values shape: (n_quantiles, n_samples)
    qs = np.array(quantiles, dtype=float)
    values = q_values
    n_q, n_samples = values.shape
    cdf = np.zeros(n_samples, dtype=float)
    for i in range(n_samples):
        yq = values[:, i]
        if np.all(np.isnan(yq)):
            cdf[i] = np.nan
            continue
        # Ensure sorted by quantile
        order = np.argsort(qs)
        qs_sorted = qs[order]
        y_sorted = yq[order]
        if threshold <= y_sorted[0]:
            # Extrapolate below first quantile
            if n_q >= 2:
                slope = (qs_sorted[1] - qs_sorted[0]) / max(y_sorted[1] - y_sorted[0], 1e-6)
                cdf_val = qs_sorted[0] + slope * (threshold - y_sorted[0])
            else:
                cdf_val = qs_sorted[0]
            cdf[i] = float(np.clip(cdf_val, 0.0, 1.0))
            continue
        if threshold >= y_sorted[-1]:
            if n_q >= 2:
                slope = (qs_sorted[-1] - qs_sorted[-2]) / max(y_sorted[-1] - y_sorted[-2], 1e-6)
                cdf_val = qs_sorted[-1] + slope * (threshold - y_sorted[-1])
            else:
                cdf_val = qs_sorted[-1]
            cdf[i] = float(np.clip(cdf_val, 0.0, 1.0))
            continue
        idx = np.searchsorted(y_sorted, threshold, side="right")
        y_low = y_sorted[idx - 1]
        y_high = y_sorted[idx]
        q_low = qs_sorted[idx - 1]
        q_high = qs_sorted[idx]
        if y_high == y_low:
            cdf[i] = float(q_high)
        else:
            frac = (threshold - y_low) / (y_high - y_low)
            cdf[i] = float(q_low + frac * (q_high - q_low))
    return cdf


def _log_loss_binary(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    p = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def _event_name(spec: dict[str, Any]) -> str:
    if spec.get("type") == "threshold":
        if "lt" in spec:
            return f"lt_{spec['lt']}"
        if "ge" in spec:
            return f"ge_{spec['ge']}"
    if spec.get("type") == "range":
        return f"range_{spec['start']}_{spec['end']}"
    return "event"


def _baseline_probabilistic_suite(
    *,
    y_train: np.ndarray,
    baseline_train: np.ndarray,
    y_by_split: dict[str, np.ndarray | None],
    baseline_by_split: dict[str, np.ndarray | None],
    quantiles: list[float],
    event_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    if not quantiles and not event_specs:
        return {}
    sigma = _robust_sigma(y_train - baseline_train)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanstd(y_train - baseline_train))
    results: dict[str, Any] = {
        "method": "normal_fixed_sigma_from_train_residuals",
        "sigma": float(sigma),
    }
    if quantiles:
        q_preds_by_split: dict[str, dict[str, np.ndarray]] = {}
        for split_name, mu in baseline_by_split.items():
            if mu is None or len(mu) == 0:
                continue
            q_preds_by_split[split_name] = _normal_quantiles(mu, sigma, quantiles)
        q_metrics_by_split = {}
        for split_name, preds in q_preds_by_split.items():
            y_true = y_by_split.get(split_name)
            if y_true is None or len(y_true) == 0:
                continue
            q_metrics_by_split[split_name] = _quantile_metrics(
                y_true, preds, quantiles=quantiles
            )
        results["quantiles"] = q_metrics_by_split

    if event_specs:
        event_metrics_by_split = {}
        for split_name, mu in baseline_by_split.items():
            y_true = y_by_split.get(split_name)
            if mu is None or y_true is None or len(mu) == 0:
                continue
            event_metrics_by_split[split_name] = _event_metrics_normal(
                y_true=y_true, mu=mu, sigma=sigma, event_specs=event_specs
            )
        results["probabilistic_events"] = event_metrics_by_split
    return results


def _robust_sigma(residuals: np.ndarray) -> float:
    residuals = residuals.astype(float)
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size == 0:
        return float("nan")
    mad = float(np.median(np.abs(residuals - np.median(residuals))))
    return 1.4826 * mad


def _normal_quantiles(mu: np.ndarray, sigma: float, quantiles: list[float]) -> dict[str, np.ndarray]:
    try:
        from scipy.stats import norm
    except ImportError as exc:
        raise ImportError("scipy is required for baseline normal quantiles.") from exc
    out: dict[str, np.ndarray] = {}
    for q in sorted(quantiles):
        key = f"q{int(q*100):02d}"
        out[key] = mu + sigma * norm.ppf(q)
    return out


def _event_metrics_normal(
    *,
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: float,
    event_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        from scipy.stats import norm
    except ImportError as exc:
        raise ImportError("scipy is required for baseline normal event metrics.") from exc
    results = {"brier": {}, "log_loss": {}}
    for spec in event_specs:
        name = spec.get("name") or _event_name(spec)
        if spec.get("type") == "threshold":
            if "lt" in spec:
                probs = norm.cdf(float(spec["lt"]), loc=mu, scale=sigma)
            elif "ge" in spec:
                probs = 1.0 - norm.cdf(float(spec["ge"]), loc=mu, scale=sigma)
            else:
                continue
        elif spec.get("type") == "range":
            start = float(spec["start"])
            end = float(spec["end"])
            probs = norm.cdf(end, loc=mu, scale=sigma) - norm.cdf(start, loc=mu, scale=sigma)
        else:
            continue
        probs = np.asarray(probs, dtype=float)
        y_event = metrics.event_indicator(y_true, spec).astype(int)
        results["brier"][name] = float(metrics.brier_score(y_event, probs))
        results["log_loss"][name] = float(_log_loss_binary(y_event, probs))
    return results


def _diff_summary(
    *,
    baseline_point: dict[str, dict[str, float]],
    model_point: dict[str, dict[str, float]],
    baseline_prob: dict[str, Any],
    model_prob: dict[str, Any],
) -> dict[str, Any]:
    diff: dict[str, Any] = {"point": {}, "brier": {}, "log_loss": {}}
    for split in ["train", "val", "test"]:
        if split in baseline_point and split in model_point:
            diff["point"][split] = {
                "mae": float(model_point[split]["mae"] - baseline_point[split]["mae"]),
                "rmse": float(model_point[split]["rmse"] - baseline_point[split]["rmse"]),
                "bias": float(model_point[split]["bias"] - baseline_point[split]["bias"]),
            }

    base_events = (baseline_prob.get("probabilistic_events") or {})
    model_events = (model_prob.get("events") or {})
    for split in ["val", "test"]:
        be = base_events.get(split) or {}
        me = model_events.get(split) or {}
        for metric_name, key in [("brier", "brier"), ("log_loss", "log_loss")]:
            if metric_name not in be or metric_name not in me:
                continue
            base_map = be.get(metric_name) or {}
            model_map = me.get(metric_name) or {}
            out = {}
            for event, base_val in base_map.items():
                if event in model_map:
                    out[event] = float(model_map[event] - base_val)
            if out:
                diff[key][split] = out
    return diff


def _stacking_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    baseline_name: str,
    baseline_train: pd.Series,
    baseline_test: pd.Series,
    lgbm_params: dict[str, Any],
    n_splits: int,
    gap_days: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["y_actual_tmax_f"].to_numpy(dtype=float)
    dates = pd.to_datetime(train_df["target_date_local"]).to_numpy()

    oof_pred = np.full(len(train_df), np.nan, dtype=float)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, val_idx in tscv.split(X_train):
        if gap_days > 0:
            val_start = dates[val_idx[0]]
            gap_cutoff = val_start - np.timedelta64(gap_days, "D")
            train_idx = train_idx[dates[train_idx] <= gap_cutoff]
        if len(train_idx) == 0:
            continue
        model = _fit_lgbm_residual_model(
            X_train[train_idx],
            y_train[train_idx],
            baseline_train.to_numpy(dtype=float)[train_idx],
            params=lgbm_params,
            sample_weight=None,
        )
        resid_pred = model.predict(X_train[val_idx])
        oof_pred[val_idx] = baseline_train.to_numpy(dtype=float)[val_idx] + resid_pred

    valid_mask = np.isfinite(oof_pred)
    if not np.any(valid_mask):
        return np.full(len(test_df), np.nan, dtype=float), {"error": "No OOF predictions for stacking."}

    base_knn = train_df.get("knn_v0_analog_mu")
    meta_features = pd.DataFrame(
        {
            "pred_lgbm": oof_pred,
            "pred_baseline": baseline_train.to_numpy(dtype=float),
            "pred_knn": pd.to_numeric(base_knn, errors="coerce") if base_knn is not None else np.nan,
        }
    )
    meta_train = meta_features[valid_mask].to_numpy(dtype=float)
    col_medians = np.nanmedian(meta_train, axis=0)
    meta_train = np.where(np.isnan(meta_train), col_medians, meta_train)
    meta_y = y_train[valid_mask]

    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_train, meta_y)

    X_test = test_df[feature_cols].to_numpy(dtype=float)
    lgbm_full = _fit_lgbm_residual_model(
        X_train,
        y_train,
        baseline_train.to_numpy(dtype=float),
        params=lgbm_params,
        sample_weight=None,
    )
    resid_test = lgbm_full.predict(X_test)
    pred_lgbm_test = baseline_test.to_numpy(dtype=float) + resid_test
    base_knn_test = test_df.get("knn_v0_analog_mu")
    meta_test = np.column_stack(
        [
            pred_lgbm_test,
            baseline_test.to_numpy(dtype=float),
            pd.to_numeric(base_knn_test, errors="coerce") if base_knn_test is not None else np.full(len(test_df), np.nan),
        ]
    )
    meta_test = np.where(np.isnan(meta_test), col_medians, meta_test)
    pred_stack = meta_model.predict(meta_test)
    stack_metrics = metrics.regression_metrics(test_df["y_actual_tmax_f"].to_numpy(dtype=float), pred_stack)
    return pred_stack, {"test": stack_metrics, "baseline": baseline_name}


def _write_predictions(
    path: Path,
    df: pd.DataFrame,
    y_true: np.ndarray,
    baseline: np.ndarray,
    pred: np.ndarray,
    pred_stack: np.ndarray | None,
    quantile_preds: dict[str, np.ndarray],
    quantiles: list[float],
) -> None:
    out = df[["station_id", "target_date_local"]].copy()
    out["y_true"] = y_true
    out["pred_baseline"] = baseline
    out["pred_point"] = pred
    if pred_stack is not None:
        out["pred_stack"] = pred_stack
    for key, values in quantile_preds.items():
        out[key] = values
    out.to_csv(path, index=False)


def _split_df(df: pd.DataFrame, start: date | None, end: date | None) -> pd.DataFrame:
    if start is None or end is None:
        return pd.DataFrame()
    mask = (df["target_date_local"] >= start) & (df["target_date_local"] <= end)
    return df.loc[mask].copy()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# KMIA Kalshi Tmax Metrics", ""]
    lines.append("## Splits")
    for name, split in payload.get("splits", {}).items():
        lines.append(f"- {name}: {split.get('start')} → {split.get('end')} ({split.get('rows')} rows)")
    lines.append("")
    metrics_payload = payload.get("metrics", {}) or {}

    baseline_point = ((metrics_payload.get("baseline") or {}).get("point") or {})
    model_point = ((metrics_payload.get("model") or {}).get("point") or {})
    diff_point = ((metrics_payload.get("diff_model_minus_baseline") or {}).get("point") or {})

    lines.append("## Point Forecast")
    for split in ["train", "val", "test"]:
        if split not in baseline_point or split not in model_point:
            continue
        b = baseline_point[split]
        m = model_point[split]
        d = diff_point.get(split, {})
        lines.append(
            f"- {split}: baseline_mae={b['mae']:.4f} model_mae={m['mae']:.4f} delta_mae={d.get('mae', m['mae']-b['mae']):+.4f}"
        )
        lines.append(
            f"- {split}: baseline_rmse={b['rmse']:.4f} model_rmse={m['rmse']:.4f} delta_rmse={d.get('rmse', m['rmse']-b['rmse']):+.4f}"
        )
        lines.append(
            f"- {split}: baseline_bias={b['bias']:.4f} model_bias={m['bias']:.4f} delta_bias={d.get('bias', m['bias']-b['bias']):+.4f}"
        )

    if "skill_score_test" in metrics_payload:
        lines.append(f"- skill_score_test={float(metrics_payload['skill_score_test']):+.4f}")
    if metrics_payload.get("skill_score_val") is not None:
        lines.append(f"- skill_score_val={float(metrics_payload['skill_score_val']):+.4f}")

    baseline_prob = ((metrics_payload.get("baseline") or {}).get("probabilistic") or {})
    model_quantiles = ((metrics_payload.get("model") or {}).get("quantiles") or {})
    model_events = ((metrics_payload.get("model") or {}).get("probabilistic_events") or {})

    base_quantiles = baseline_prob.get("quantiles") or {}
    base_events = baseline_prob.get("probabilistic_events") or {}
    diff_events = metrics_payload.get("diff_model_minus_baseline") or {}

    for split in ["val", "test"]:
        mq = model_quantiles.get(split)
        bq = base_quantiles.get(split)
        if mq or bq:
            lines.append("")
            lines.append(f"## Quantiles ({split})")
            if mq:
                lines.append(f"- model_crps_approx={mq.get('crps_approx')}")
                for key, value in (mq.get("coverage") or {}).items():
                    lines.append(f"- model_coverage_{key}={value:.3f}")
                for key, value in (mq.get("interval_width") or {}).items():
                    lines.append(f"- model_width_{key}={value:.3f}")
            if bq:
                lines.append(f"- baseline_sigma={baseline_prob.get('sigma')}")
                lines.append(f"- baseline_crps_approx={bq.get('crps_approx')}")
                for key, value in (bq.get("coverage") or {}).items():
                    lines.append(f"- baseline_coverage_{key}={value:.3f}")
                for key, value in (bq.get("interval_width") or {}).items():
                    lines.append(f"- baseline_width_{key}={value:.3f}")

        me = model_events.get(split) or {}
        be = base_events.get(split) or {}
        if me or be:
            lines.append("")
            lines.append(f"## Probabilistic Events ({split})")
            brier_diff = ((diff_events.get("brier") or {}).get(split) or {})
            logloss_diff = ((diff_events.get("log_loss") or {}).get(split) or {})
            for event, model_val in (me.get("brier") or {}).items():
                base_val = (be.get("brier") or {}).get(event)
                if base_val is None:
                    continue
                delta = brier_diff.get(event, model_val - base_val)
                lines.append(f"- brier {event}: baseline={base_val:.4f} model={model_val:.4f} delta={delta:+.4f}")
            for event, model_val in (me.get("log_loss") or {}).items():
                base_val = (be.get("log_loss") or {}).get(event)
                if base_val is None:
                    continue
                delta = logloss_diff.get(event, model_val - base_val)
                lines.append(f"- logloss {event}: baseline={base_val:.4f} model={model_val:.4f} delta={delta:+.4f}")
    path.write_text("\n".join(lines), encoding="utf-8")
