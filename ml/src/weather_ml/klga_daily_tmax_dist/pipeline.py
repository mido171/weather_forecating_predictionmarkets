from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import logging
import time

import json
import numpy as np
import pandas as pd

from .analog_knn import (
    AnalogLibrary,
    AnalogPosterior,
    AnalogStandardizer,
    blend_posteriors,
    build_analog_library,
    calibrate_blend_bounds,
    fit_analog_standardizer,
    predict_knn_posterior,
)
from .config import PipelineConfig
from .infer import build_delta_pmf, delta_pmf_to_tmax_pmf
from .logging_utils import format_duration
from .make_dataset import build_feature_store
from .train_delta import predict_delta_conditional, train_delta_model
from .train_peak import predict_peak_probability, train_peak_model


def _timestamp_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _configure_run_logger(
    *,
    run_dir: Path,
    log_level: str | int = "INFO",
) -> tuple[logging.Logger, Path]:
    logger = logging.getLogger("weather_ml.klga_daily_tmax_dist")
    if isinstance(log_level, str):
        level = getattr(logging, log_level.upper(), logging.INFO)
    else:
        level = int(log_level)
    logger.setLevel(level)

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(stream_handler)

    run_log_path = run_dir / "run.log"
    has_run_file = False
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            try:
                if Path(h.baseFilename).resolve() == run_log_path.resolve():
                    has_run_file = True
                    h.setLevel(level)
            except Exception:
                continue
    if not has_run_file:
        file_handler = logging.FileHandler(run_log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger, run_log_path


def _split_masks(df: pd.DataFrame, cfg: PipelineConfig) -> dict[str, np.ndarray]:
    dates = pd.to_datetime(df["target_date_local"]).dt.date
    s = cfg.split
    train = (dates >= s.train_start) & (dates <= s.train_end)
    val = (dates >= s.val_start) & (dates <= s.val_end)
    test = (dates >= s.test_start) & (dates <= s.test_end)

    if np.any(train & val) or np.any(train & test) or np.any(val & test):
        raise AssertionError("Split overlap detected.")
    covered = train | val | test
    if not np.all(covered):
        outside = sorted(set(dates[~covered]))
        raise AssertionError(
            f"Split guard failed: {len(outside)} dates outside configured ranges, first={outside[:5]}"
        )
    return {"train": train.to_numpy(), "val": val.to_numpy(), "test": test.to_numpy()}


def _add_climo_features(df: pd.DataFrame, train_mask: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    train_df = out.loc[train_mask].copy()
    if train_df.empty:
        raise ValueError("Train split is empty; cannot compute climo lookup.")
    if train_df["delta"].isna().all():
        raise ValueError("Train split has no delta labels; cannot compute climo lookup.")

    key_cols = ["doy", "cutoff_minutes"]
    lookup = (
        train_df.groupby(key_cols, as_index=False)["delta"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "climo_rem_delta_mean", "std": "climo_rem_delta_std"})
    )
    lookup["climo_rem_delta_std"] = lookup["climo_rem_delta_std"].fillna(0.0)

    cutoff_fallback = (
        train_df.groupby("cutoff_minutes", as_index=False)["delta"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(
            columns={
                "mean": "climo_cutoff_fallback_mean",
                "std": "climo_cutoff_fallback_std",
            }
        )
    )
    cutoff_fallback["climo_cutoff_fallback_std"] = cutoff_fallback["climo_cutoff_fallback_std"].fillna(0.0)

    out = out.merge(lookup, on=key_cols, how="left")
    out = out.merge(cutoff_fallback, on="cutoff_minutes", how="left")
    out["climo_rem_delta_mean"] = out["climo_rem_delta_mean"].fillna(out["climo_cutoff_fallback_mean"])
    out["climo_rem_delta_std"] = out["climo_rem_delta_std"].fillna(out["climo_cutoff_fallback_std"])
    out = out.drop(columns=["climo_cutoff_fallback_mean", "climo_cutoff_fallback_std"])

    meta = {
        "lookup_rows": int(len(lookup)),
        "train_rows_used": int(len(train_df)),
    }
    return out, meta


def _model_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "target_date_local",
        "cutoff_local",
        "cutoff_utc",
        "midnight_utc",
        "max_valid_time_used_utc",
        "tmax_truth",
        "delta",
        "peak",
    }
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _fit_imputer(df: pd.DataFrame, *, feature_cols: list[str], train_mask: np.ndarray) -> dict[str, float]:
    med = {}
    train_df = df.loc[train_mask, feature_cols]
    for c in feature_cols:
        vals = pd.to_numeric(train_df[c], errors="coerce")
        arr = vals.to_numpy(dtype=float)
        if np.all(~np.isfinite(arr)):
            m = 0.0
        else:
            m = float(np.nanmedian(arr))
        if not np.isfinite(m):
            m = 0.0
        med[c] = m
    return med


def _apply_imputer(df: pd.DataFrame, *, feature_cols: list[str], medians: dict[str, float]) -> np.ndarray:
    x = df.loc[:, feature_cols].to_numpy(dtype=float)
    for i, c in enumerate(feature_cols):
        m = medians[c]
        col = x[:, i]
        bad = ~np.isfinite(col)
        if np.any(bad):
            col[bad] = m
            x[:, i] = col
    return x


def _recency_weights(df: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    d = pd.to_datetime(df.loc[mask, "target_date_local"]).dt.date
    if d.empty:
        return np.array([], dtype=float)
    d0 = min(d)
    d1 = max(d)
    span = max((d1 - d0).days, 1)
    rel = np.array([(x - d0).days / span for x in d], dtype=float)
    return 1.0 + 0.5 * rel


def _multi_logloss(y_true: np.ndarray, probs: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    p_true = np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1.0)
    return float(-np.mean(np.log(p_true)))


def _expand_class_probs(
    *,
    probs: np.ndarray,
    class_labels: np.ndarray,
    total_classes: int,
) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    class_labels = np.asarray(class_labels, dtype=int)
    if probs.shape[1] == total_classes and np.array_equal(class_labels, np.arange(total_classes)):
        return probs
    out = np.zeros((probs.shape[0], total_classes), dtype=float)
    valid = (class_labels >= 0) & (class_labels < total_classes)
    out[:, class_labels[valid]] = probs[:, valid]
    row_sum = np.sum(out, axis=1, keepdims=True)
    good = row_sum.squeeze() > 0
    out[good] = out[good] / row_sum[good]
    if np.any(~good):
        out[~good] = 1.0 / total_classes
    return out


def _pmf_prob_true(
    *,
    tmax_truth: float,
    tmax_sofar: float,
    delta_pmf: np.ndarray,
) -> float:
    if not np.isfinite(tmax_truth) or not np.isfinite(tmax_sofar):
        return np.nan
    base = int(np.round(float(tmax_sofar)))
    truth = int(np.round(float(tmax_truth)))
    delta_true = max(truth - base, 0)
    if delta_true >= len(delta_pmf):
        return float(delta_pmf[-1])
    return float(delta_pmf[delta_true])


def _evaluate_distribution_rows(
    *,
    df: pd.DataFrame,
    row_indices: np.ndarray,
    p_peak: np.ndarray,
    p_delta_cond: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame]:
    rows = []
    n = len(row_indices)
    for i, row_idx in enumerate(row_indices):
        rec = df.iloc[int(row_idx)]
        pmf = build_delta_pmf(
            p_peak=float(p_peak[i]),
            p_delta_cond=p_delta_cond[i],
        )
        prob_true = _pmf_prob_true(
            tmax_truth=float(rec["tmax_truth"]),
            tmax_sofar=float(rec["tmax_sofar"]),
            delta_pmf=pmf,
        )
        if not np.isfinite(prob_true):
            continue
        nll = float(-np.log(max(prob_true, 1e-12)))
        pred_delta = int(np.argmax(pmf))
        pred_tmax = int(np.round(float(rec["tmax_sofar"]))) + pred_delta
        truth = int(np.round(float(rec["tmax_truth"])))
        rows.append(
            {
                "row_index": int(row_idx),
                "target_date_local": str(rec["target_date_local"]),
                "cutoff_minutes": int(rec["cutoff_minutes"]),
                "tmax_truth": truth,
                "tmax_sofar_round": int(np.round(float(rec["tmax_sofar"]))),
                "pred_tmax_top1": pred_tmax,
                "top1_hit": 1.0 if pred_tmax == truth else 0.0,
                "prob_true_tmax": prob_true,
                "nll_true_tmax": nll,
                "p_peak": float(p_peak[i]),
            }
        )

    detail = pd.DataFrame(rows)
    if detail.empty:
        return {"n_rows": 0.0, "nll": np.nan, "top1_accuracy": np.nan}, detail
    metrics = {
        "n_rows": float(len(detail)),
        "nll": float(detail["nll_true_tmax"].mean()),
        "top1_accuracy": float(detail["top1_hit"].mean()),
    }
    return metrics, detail


def _cutoff_metrics(
    *,
    df_detail: pd.DataFrame,
) -> pd.DataFrame:
    if df_detail.empty:
        return pd.DataFrame(columns=["cutoff_minutes", "n_rows", "nll", "top1_accuracy"])
    g = (
        df_detail.groupby("cutoff_minutes", as_index=False)
        .agg(
            n_rows=("row_index", "count"),
            nll=("nll_true_tmax", "mean"),
            top1_accuracy=("top1_hit", "mean"),
        )
        .sort_values("cutoff_minutes")
        .reset_index(drop=True)
    )
    return g


def _temperature_bucket_calibration(
    *,
    df: pd.DataFrame,
    row_indices: np.ndarray,
    p_peak: np.ndarray,
    p_delta_cond: np.ndarray,
) -> pd.DataFrame:
    if len(row_indices) == 0:
        return pd.DataFrame(columns=["temp_bucket", "count", "pred_mean", "empirical"])
    truth = pd.to_numeric(df.iloc[row_indices]["tmax_truth"], errors="coerce")
    finite_truth = truth[np.isfinite(truth)]
    if finite_truth.empty:
        return pd.DataFrame(columns=["temp_bucket", "count", "pred_mean", "empirical"])
    tmin = int(finite_truth.min())
    tmax = int(finite_truth.max())
    temps = list(range(tmin, tmax + 1))

    pred_sums = {t: 0.0 for t in temps}
    obs_sums = {t: 0.0 for t in temps}
    counts = {t: 0 for t in temps}
    for i, row_idx in enumerate(row_indices):
        rec = df.iloc[int(row_idx)]
        tmax_sofar = float(rec["tmax_sofar"])
        tmax_truth = float(rec["tmax_truth"])
        if not np.isfinite(tmax_sofar) or not np.isfinite(tmax_truth):
            continue
        pmf = build_delta_pmf(
            p_peak=float(p_peak[i]),
            p_delta_cond=p_delta_cond[i],
        )
        tmax_pmf = delta_pmf_to_tmax_pmf(
            tmax_sofar=tmax_sofar,
            delta_pmf=pmf,
        )
        truth_temp = int(np.round(tmax_truth))
        for t in temps:
            pred_sums[t] += float(tmax_pmf.get(t, 0.0))
            obs_sums[t] += 1.0 if truth_temp == t else 0.0
            counts[t] += 1

    rows = []
    for t in temps:
        c = counts[t]
        if c <= 0:
            continue
        rows.append(
            {
                "temp_bucket": int(t),
                "count": int(c),
                "pred_mean": float(pred_sums[t] / c),
                "empirical": float(obs_sums[t] / c),
                "abs_gap": float(abs((pred_sums[t] / c) - (obs_sums[t] / c))),
            }
        )
    return pd.DataFrame(rows).sort_values("temp_bucket").reset_index(drop=True)


def _build_full_delta_arrays(
    *,
    full_len: int,
    class_count: int,
    row_indices: np.ndarray,
    p_peak: np.ndarray,
    p_delta: np.ndarray,
    q_score: np.ndarray | None = None,
    w_lgbm: np.ndarray | None = None,
) -> pd.DataFrame:
    out = pd.DataFrame({"row_index": np.arange(full_len, dtype=int)})
    out["p_peak"] = np.nan
    out.loc[row_indices, "p_peak"] = p_peak
    for k in range(class_count):
        out[f"p_delta_class_{k + 1}"] = np.nan
        out.loc[row_indices, f"p_delta_class_{k + 1}"] = p_delta[:, k]
    if q_score is not None:
        out["analog_q_score"] = np.nan
        out.loc[row_indices, "analog_q_score"] = q_score
    if w_lgbm is not None:
        out["blend_w_lgbm"] = np.nan
        out.loc[row_indices, "blend_w_lgbm"] = w_lgbm
    return out


def _nanmean_or_nan(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


@dataclass(frozen=True)
class PipelineRunResult:
    run_dir: Path
    metrics_path: Path
    metrics: dict[str, Any]


def run_training_pipeline(
    *,
    cfg: PipelineConfig,
    mysql_url: str | None = None,
    force_rebuild_dataset: bool = False,
    enable_analog: bool = True,
    log_level: str | int = "INFO",
    log_every_rows: int = 2000,
    log_every_seconds: float = 20.0,
    peak_train_log_period: int = 50,
    delta_train_log_period: int = 25,
    train_log_every_seconds: float = 10.0,
    train_heartbeat_seconds: float = 10.0,
) -> PipelineRunResult:
    run_dir = _ensure_dir(cfg.output_root / _timestamp_id())
    _ensure_dir(run_dir / "models")
    _ensure_dir(run_dir / "reports")
    _ensure_dir(run_dir / "predictions")
    logger, run_log_path = _configure_run_logger(run_dir=run_dir, log_level=log_level)
    pipeline_start = time.perf_counter()
    stage_total = 12 if enable_analog else 9
    stage_idx = 0

    def stage_start(name: str, *, details: str = "") -> tuple[int, float]:
        nonlocal stage_idx
        stage_idx += 1
        pct = 100.0 * (stage_idx - 1) / stage_total
        logger.info("STAGE_START [%d/%d %.1f%%] %s %s", stage_idx, stage_total, pct, name, details)
        return stage_idx, time.perf_counter()

    def stage_end(idx: int, name: str, t0: float, *, details: str = "") -> None:
        pct = 100.0 * idx / stage_total
        logger.info(
            "STAGE_END   [%d/%d %.1f%%] %s elapsed=%s %s",
            idx,
            stage_total,
            pct,
            name,
            format_duration(time.perf_counter() - t0),
            details,
        )

    logger.info(
        (
            "PIPELINE_START run_dir=%s run_log=%s force_rebuild_dataset=%s "
            "progress_rows=%d progress_seconds=%.1f peak_log_period=%d delta_log_period=%d "
            "train_log_seconds=%.1f train_heartbeat_seconds=%.1f enable_analog=%s"
        ),
        run_dir,
        run_log_path,
        force_rebuild_dataset,
        log_every_rows,
        float(log_every_seconds),
        peak_train_log_period,
        delta_train_log_period,
        float(train_log_every_seconds),
        float(train_heartbeat_seconds),
        bool(enable_analog),
    )

    feature_store_path = cfg.output_root / "feature_store" / "klga_feature_store.parquet"
    sidx, st0 = stage_start("build_feature_store", details=f"path={feature_store_path}")
    if force_rebuild_dataset or (not feature_store_path.exists()):
        ds = build_feature_store(
            cfg=cfg,
            mysql_url=mysql_url,
            output_root=cfg.output_root,
            logger=logger,
            log_every_rows=log_every_rows,
            log_every_seconds=log_every_seconds,
        )
        feature_store_path = ds.feature_store_path
        stage_end(
            sidx,
            "build_feature_store",
            st0,
            details=f"rows={ds.rows} dates={ds.dates} created_indexes={ds.created_indexes}",
        )
    else:
        logger.info("FEATURE_STORE_REUSE path=%s", feature_store_path)
        stage_end(sidx, "build_feature_store", st0, details="reused_existing_feature_store")

    sidx, st0 = stage_start("load_feature_store")
    df = pd.read_parquet(feature_store_path)
    if df.empty:
        raise ValueError("Feature store is empty.")
    stage_end(
        sidx,
        "load_feature_store",
        st0,
        details=f"rows={len(df)} cols={df.shape[1]} date_min={df['target_date_local'].min()} date_max={df['target_date_local'].max()}",
    )

    sidx, st0 = stage_start("prepare_splits_and_features")
    split_masks = _split_masks(df, cfg)
    df, climo_meta = _add_climo_features(df, split_masks["train"])
    feature_cols = _model_feature_columns(df)

    medians = _fit_imputer(df, feature_cols=feature_cols, train_mask=split_masks["train"])
    x_all = _apply_imputer(df, feature_cols=feature_cols, medians=medians)

    peak_series = pd.to_numeric(df["peak"], errors="coerce")
    delta_series = pd.to_numeric(df["delta"], errors="coerce")
    peak_label_mask = np.isfinite(peak_series.to_numpy(dtype=float))
    delta_label_mask = np.isfinite(delta_series.to_numpy(dtype=float))

    y_peak_all = np.full(len(df), -1, dtype=int)
    y_delta_all = np.full(len(df), -1, dtype=int)
    if np.any(peak_label_mask):
        y_peak_all[peak_label_mask] = (
            peak_series.loc[peak_label_mask].round().astype(int).to_numpy()
        )
    if np.any(delta_label_mask):
        y_delta_all[delta_label_mask] = (
            delta_series.loc[delta_label_mask].round().astype(int).to_numpy()
        )

    peak_train_mask = split_masks["train"] & peak_label_mask
    peak_val_mask = split_masks["val"] & peak_label_mask
    peak_test_mask = split_masks["test"] & peak_label_mask
    train_idx = np.where(peak_train_mask)[0]
    val_idx = np.where(peak_val_mask)[0]
    test_idx = np.where(peak_test_mask)[0]
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Peak train/validation/test splits are empty after label filtering.")

    train_weights = _recency_weights(df, peak_train_mask)
    stage_end(
        sidx,
        "prepare_splits_and_features",
        st0,
        details=(
            f"feature_count={len(feature_cols)} "
            f"train_rows={len(train_idx)} val_rows={len(val_idx)} test_rows={len(test_idx)} "
            f"peak_labels={int(np.sum(peak_label_mask))} delta_labels={int(np.sum(delta_label_mask))}"
        ),
    )

    sidx, st0 = stage_start("train_peak_model")
    peak_result = train_peak_model(
        X_train=x_all[train_idx],
        y_train=y_peak_all[train_idx],
        X_val=x_all[val_idx],
        y_val=y_peak_all[val_idx],
        sample_weight_train=train_weights,
        logger=logger,
        log_period=peak_train_log_period,
        log_every_seconds=train_log_every_seconds,
        heartbeat_seconds=train_heartbeat_seconds,
        stage_label="PEAK_TRAIN",
    )
    stage_end(
        sidx,
        "train_peak_model",
        st0,
        details=(
            f"val_logloss_cal={peak_result.val_metrics.get('logloss_cal')} "
            f"val_brier_cal={peak_result.val_metrics.get('brier_cal')}"
        ),
    )
    # Checkpoint peak artifacts early so they exist even if later stages fail.
    import joblib
    models_dir = run_dir / "models"
    peak_result.model.booster_.save_model(str(models_dir / "peak_model.txt"))
    joblib.dump(peak_result.isotonic, models_dir / "peak_isotonic.pkl")

    sidx, st0 = stage_start("predict_peak_probabilities")
    p_peak_raw_all = np.zeros(len(df), dtype=float)
    p_peak_cal_all = np.zeros(len(df), dtype=float)
    for idx_name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        logger.info("PEAK_PREDICT split=%s rows=%d", idx_name, len(idx))
        raw, cal = predict_peak_probability(
            model=peak_result.model,
            isotonic=peak_result.isotonic,
            X=x_all[idx],
        )
        p_peak_raw_all[idx] = raw
        p_peak_cal_all[idx] = cal
    stage_end(sidx, "predict_peak_probabilities", st0, details="done")

    # Delta model trains only on non-peak rows.
    k_delta = cfg.delta_class_max
    delta_train_mask = (
        split_masks["train"]
        & peak_label_mask
        & delta_label_mask
        & (y_peak_all == 0)
        & (y_delta_all >= 1)
    )
    delta_val_mask = (
        split_masks["val"]
        & peak_label_mask
        & delta_label_mask
        & (y_peak_all == 0)
        & (y_delta_all >= 1)
    )
    if not np.any(delta_train_mask):
        raise ValueError("No train rows available for delta model.")
    if not np.any(delta_val_mask):
        raise ValueError("No validation rows available for delta model.")

    delta_train_idx = np.where(delta_train_mask)[0]
    delta_val_idx = np.where(delta_val_mask)[0]
    y_delta_class_train = np.clip(y_delta_all[delta_train_idx], 1, k_delta) - 1
    y_delta_class_val = np.clip(y_delta_all[delta_val_idx], 1, k_delta) - 1
    train_weight_full = np.zeros(len(df), dtype=float)
    train_weight_full[train_idx] = train_weights
    delta_train_weights = train_weight_full[delta_train_idx]

    sidx, st0 = stage_start(
        "train_delta_model",
        details=f"delta_train_rows={len(delta_train_idx)} delta_val_rows={len(delta_val_idx)}",
    )
    delta_result = train_delta_model(
        X_train=x_all[delta_train_idx],
        y_train=y_delta_class_train,
        X_val=x_all[delta_val_idx],
        y_val=y_delta_class_val,
        num_classes=k_delta,
        sample_weight_train=delta_train_weights,
        logger=logger,
        log_period=delta_train_log_period,
        log_every_seconds=train_log_every_seconds,
        heartbeat_seconds=train_heartbeat_seconds,
        stage_label="DELTA_TRAIN",
    )
    stage_end(
        sidx,
        "train_delta_model",
        st0,
        details=f"val_multi_logloss_temp={delta_result.val_metrics.get('multi_logloss_temp')}",
    )
    # Checkpoint delta artifacts early so they exist even if later stages fail.
    delta_result.model.booster_.save_model(str(models_dir / "delta_model.txt"))
    (models_dir / "delta_temperature_T.json").write_text(
        json.dumps({"temperature": float(delta_result.temperature)}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    sidx, st0 = stage_start("predict_delta_conditionals")
    logits_all, p_delta_raw_all, p_delta_temp_all = predict_delta_conditional(
        model=delta_result.model,
        temperature=delta_result.temperature,
        X=x_all,
    )
    model_classes = np.asarray(delta_result.model.classes_, dtype=int)
    p_delta_raw_all = _expand_class_probs(
        probs=p_delta_raw_all,
        class_labels=model_classes,
        total_classes=k_delta,
    )
    p_delta_temp_all = _expand_class_probs(
        probs=p_delta_temp_all,
        class_labels=model_classes,
        total_classes=k_delta,
    )
    stage_end(sidx, "predict_delta_conditionals", st0, details=f"class_count={k_delta}")

    analog_std: AnalogStandardizer | None = None
    analog_val: AnalogPosterior | None = None
    analog_test: AnalogPosterior | None = None
    best_k: int | None = None
    best_val_nll: float = float("nan")
    q_low: float | None = None
    q_high: float | None = None
    val_w: np.ndarray | None = None
    test_w: np.ndarray | None = None

    if enable_analog:
        # Analog library and validation-time K selection.
        sidx, st0 = stage_start("build_analog_library")
        analog_cols = [c for c in cfg.analog_feature_columns if c in df.columns]
        missing_analog = sorted(set(cfg.analog_feature_columns).difference(analog_cols))
        if missing_analog:
            raise ValueError(f"Missing required analog features: {missing_analog}")

        analog_std = fit_analog_standardizer(
            df=df,
            feature_columns=analog_cols,
            train_mask=split_masks["train"],
            feature_weights=cfg.analog_feature_weights,
        )
        analog_lib = build_analog_library(
            df=df,
            standardizer=analog_std,
            delta_class_max=k_delta,
        )
        stage_end(
            sidx,
            "build_analog_library",
            st0,
            details=f"analog_feature_count={len(analog_cols)} library_rows={len(df)}",
        )

        sidx, st0 = stage_start("analog_k_selection")
        best_k = cfg.analog_default_k
        best_val_nll = np.inf
        analog_val_cache: dict[int, AnalogPosterior] = {}
        for k in cfg.analog_k_grid:
            logger.info("ANALOG_K_START k=%d val_rows=%d", k, len(val_idx))
            post = predict_knn_posterior(
                library=analog_lib,
                standardizer=analog_std,
                query_indices=val_idx,
                k=int(k),
                delta_class_max=k_delta,
                season_window_doy=cfg.analog_season_window_doy,
                min_pool=cfg.analog_min_pool,
                min_non_peak=cfg.analog_min_non_peak,
                logger=logger,
                log_every_rows=log_every_rows,
                log_every_seconds=log_every_seconds,
                log_label=f"ANALOG_VAL_K{k}",
            )
            analog_val_cache[int(k)] = post
            mask_ok = np.isfinite(post.p_peak) & np.isfinite(np.sum(post.p_delta_cond, axis=1))
            if not np.any(mask_ok):
                continue
            m, _ = _evaluate_distribution_rows(
                df=df,
                row_indices=val_idx[mask_ok],
                p_peak=post.p_peak[mask_ok],
                p_delta_cond=post.p_delta_cond[mask_ok],
            )
            nll = float(m.get("nll", np.inf))
            logger.info(
                "ANALOG_K_DONE k=%d valid_rows=%d val_nll=%s",
                k,
                int(np.sum(mask_ok)),
                nll,
            )
            if np.isfinite(nll) and nll < best_val_nll:
                best_val_nll = nll
                best_k = int(k)

        analog_val = analog_val_cache.get(best_k)
        if analog_val is None:
            analog_val = predict_knn_posterior(
                library=analog_lib,
                standardizer=analog_std,
                query_indices=val_idx,
                k=best_k,
                delta_class_max=k_delta,
                season_window_doy=cfg.analog_season_window_doy,
                min_pool=cfg.analog_min_pool,
                min_non_peak=cfg.analog_min_non_peak,
            )
        analog_test = predict_knn_posterior(
            library=analog_lib,
            standardizer=analog_std,
            query_indices=test_idx,
            k=best_k,
            delta_class_max=k_delta,
            season_window_doy=cfg.analog_season_window_doy,
            min_pool=cfg.analog_min_pool,
            min_non_peak=cfg.analog_min_non_peak,
            logger=logger,
            log_every_rows=log_every_rows,
            log_every_seconds=log_every_seconds,
            log_label=f"ANALOG_TEST_K{best_k}",
        )
        stage_end(
            sidx,
            "analog_k_selection",
            st0,
            details=f"best_k={best_k} best_val_nll={best_val_nll}",
        )

        sidx, st0 = stage_start("blend_posteriors")
        q_low, q_high = calibrate_blend_bounds(analog_val.q_score)
        val_peak_blend, val_delta_blend, val_w = blend_posteriors(
            p_peak_lgbm=p_peak_cal_all[val_idx],
            p_delta_lgbm=p_delta_temp_all[val_idx],
            p_peak_knn=analog_val.p_peak,
            p_delta_knn=analog_val.p_delta_cond,
            q_score=analog_val.q_score,
            q_low=q_low,
            q_high=q_high,
        )
        test_peak_blend, test_delta_blend, test_w = blend_posteriors(
            p_peak_lgbm=p_peak_cal_all[test_idx],
            p_delta_lgbm=p_delta_temp_all[test_idx],
            p_peak_knn=analog_test.p_peak,
            p_delta_knn=analog_test.p_delta_cond,
            q_score=analog_test.q_score,
            q_low=q_low,
            q_high=q_high,
        )
        stage_end(sidx, "blend_posteriors", st0, details=f"q_low={q_low:.4f} q_high={q_high:.4f}")
    else:
        logger.info("ANALOG_DISABLED using_lgbm_only_posteriors_for_evaluation_and_artifacts")
        val_peak_blend = np.asarray(p_peak_cal_all[val_idx], dtype=float)
        test_peak_blend = np.asarray(p_peak_cal_all[test_idx], dtype=float)
        val_delta_blend = np.asarray(p_delta_temp_all[val_idx], dtype=float)
        test_delta_blend = np.asarray(p_delta_temp_all[test_idx], dtype=float)

    sidx, st0 = stage_start("evaluate_metrics")
    # Peak metrics.
    from sklearn.metrics import brier_score_loss, log_loss

    peak_metrics = {
        "val": {
            "logloss_raw": float(log_loss(y_peak_all[val_idx], np.clip(p_peak_raw_all[val_idx], 1e-6, 1.0 - 1e-6))),
            "logloss_cal": float(log_loss(y_peak_all[val_idx], np.clip(p_peak_cal_all[val_idx], 1e-6, 1.0 - 1e-6))),
            "brier_raw": float(brier_score_loss(y_peak_all[val_idx], p_peak_raw_all[val_idx])),
            "brier_cal": float(brier_score_loss(y_peak_all[val_idx], p_peak_cal_all[val_idx])),
            "brier_blend": float(brier_score_loss(y_peak_all[val_idx], val_peak_blend)),
        },
        "test": {
            "logloss_raw": float(log_loss(y_peak_all[test_idx], np.clip(p_peak_raw_all[test_idx], 1e-6, 1.0 - 1e-6))),
            "logloss_cal": float(log_loss(y_peak_all[test_idx], np.clip(p_peak_cal_all[test_idx], 1e-6, 1.0 - 1e-6))),
            "brier_raw": float(brier_score_loss(y_peak_all[test_idx], p_peak_raw_all[test_idx])),
            "brier_cal": float(brier_score_loss(y_peak_all[test_idx], p_peak_cal_all[test_idx])),
            "brier_blend": float(brier_score_loss(y_peak_all[test_idx], test_peak_blend)),
        },
    }

    # Delta metrics.
    val_logits = logits_all[delta_val_idx]
    val_probs_raw = p_delta_raw_all[delta_val_idx]
    val_probs_temp = p_delta_temp_all[delta_val_idx]
    delta_metrics = {
        "val": {
            "multi_logloss_raw": _multi_logloss(y_delta_class_val, val_probs_raw),
            "multi_logloss_temp": _multi_logloss(y_delta_class_val, val_probs_temp),
        },
        "temperature": float(delta_result.temperature),
    }

    # Combined distribution metrics.
    lgbm_val_metrics, lgbm_val_detail = _evaluate_distribution_rows(
        df=df,
        row_indices=val_idx,
        p_peak=p_peak_cal_all[val_idx],
        p_delta_cond=p_delta_temp_all[val_idx],
    )
    lgbm_test_metrics, lgbm_test_detail = _evaluate_distribution_rows(
        df=df,
        row_indices=test_idx,
        p_peak=p_peak_cal_all[test_idx],
        p_delta_cond=p_delta_temp_all[test_idx],
    )
    blend_val_metrics, blend_val_detail = _evaluate_distribution_rows(
        df=df,
        row_indices=val_idx,
        p_peak=val_peak_blend,
        p_delta_cond=val_delta_blend,
    )
    blend_test_metrics, blend_test_detail = _evaluate_distribution_rows(
        df=df,
        row_indices=test_idx,
        p_peak=test_peak_blend,
        p_delta_cond=test_delta_blend,
    )

    cutoff_val = _cutoff_metrics(df_detail=blend_val_detail)
    cutoff_test = _cutoff_metrics(df_detail=blend_test_detail)
    cutoff_val.to_csv(run_dir / "reports" / "cutoff_metrics_val.csv", index=False)
    cutoff_test.to_csv(run_dir / "reports" / "cutoff_metrics_test.csv", index=False)

    calib_val = _temperature_bucket_calibration(
        df=df,
        row_indices=val_idx,
        p_peak=val_peak_blend,
        p_delta_cond=val_delta_blend,
    )
    calib_test = _temperature_bucket_calibration(
        df=df,
        row_indices=test_idx,
        p_peak=test_peak_blend,
        p_delta_cond=test_delta_blend,
    )
    calib_val.to_csv(run_dir / "reports" / "bucket_calibration_val.csv", index=False)
    calib_test.to_csv(run_dir / "reports" / "bucket_calibration_test.csv", index=False)
    stage_end(
        sidx,
        "evaluate_metrics",
        st0,
        details=(
            f"blend_val_nll={blend_val_metrics.get('nll')} "
            f"blend_test_nll={blend_test_metrics.get('nll')}"
        ),
    )

    sidx, st0 = stage_start("write_artifacts")
    # Persist prediction tables.
    pred_val = _build_full_delta_arrays(
        full_len=len(df),
        class_count=k_delta,
        row_indices=val_idx,
        p_peak=val_peak_blend,
        p_delta=val_delta_blend,
        q_score=(analog_val.q_score if analog_val is not None else None),
        w_lgbm=val_w,
    )
    pred_test = _build_full_delta_arrays(
        full_len=len(df),
        class_count=k_delta,
        row_indices=test_idx,
        p_peak=test_peak_blend,
        p_delta=test_delta_blend,
        q_score=(analog_test.q_score if analog_test is not None else None),
        w_lgbm=test_w,
    )
    pred_val.to_parquet(run_dir / "predictions" / "predictions_val.parquet", index=False)
    pred_test.to_parquet(run_dir / "predictions" / "predictions_test.parquet", index=False)
    blend_val_detail.to_parquet(run_dir / "predictions" / "distribution_eval_val.parquet", index=False)
    blend_test_detail.to_parquet(run_dir / "predictions" / "distribution_eval_test.parquet", index=False)

    # Save models and metadata.
    models_dir = run_dir / "models"
    peak_model_path = models_dir / "peak_model.txt"
    delta_model_path = models_dir / "delta_model.txt"
    peak_result.model.booster_.save_model(str(peak_model_path))
    delta_result.model.booster_.save_model(str(delta_model_path))

    joblib.dump(peak_result.isotonic, models_dir / "peak_isotonic.pkl")
    (models_dir / "delta_temperature_T.json").write_text(
        json.dumps({"temperature": float(delta_result.temperature)}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    (run_dir / "feature_list.json").write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
    (run_dir / "imputer_values.json").write_text(
        json.dumps(medians, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if analog_std is not None and enable_analog:
        analog_payload: dict[str, Any] = {
            "enabled": True,
            "feature_columns": list(analog_std.feature_columns),
            "mean": analog_std.mean.tolist(),
            "std": analog_std.std.tolist(),
            "weight": analog_std.weight.tolist(),
            "selected_k": int(best_k) if best_k is not None else None,
            "q_low": float(q_low) if q_low is not None else None,
            "q_high": float(q_high) if q_high is not None else None,
        }
    else:
        analog_payload = {
            "enabled": False,
            "reason": "skip_analog_blend",
        }
    (run_dir / "analog_standardizer.json").write_text(
        json.dumps(analog_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # Date range snapshots.
    (run_dir / "train_date_range.txt").write_text(
        f"{cfg.split.train_start} -> {cfg.split.train_end}\n",
        encoding="utf-8",
    )
    (run_dir / "val_date_range.txt").write_text(
        f"{cfg.split.val_start} -> {cfg.split.val_end}\n",
        encoding="utf-8",
    )
    (run_dir / "test_date_range.txt").write_text(
        f"{cfg.split.test_start} -> {cfg.split.test_end}\n",
        encoding="utf-8",
    )

    metrics: dict[str, Any] = {
        "run_id": run_dir.name,
        "feature_store_path": str(feature_store_path),
        "rows_total": int(len(df)),
        "split_rows": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "climo_lookup_meta": climo_meta,
        "peak": peak_metrics,
        "delta": delta_metrics,
        "combined_lgbm": {
            "val": lgbm_val_metrics,
            "test": lgbm_test_metrics,
        },
        "combined_blended": {
            "val": blend_val_metrics,
            "test": blend_test_metrics,
        },
        "analog": (
            {
                "enabled": True,
                "selected_k": int(best_k) if best_k is not None else None,
                "candidate_k_grid": list(cfg.analog_k_grid),
                "selected_k_val_nll": float(best_val_nll if np.isfinite(best_val_nll) else np.nan),
                "q_low": float(q_low) if q_low is not None else None,
                "q_high": float(q_high) if q_high is not None else None,
                "val_mean_q_score": _nanmean_or_nan(analog_val.q_score) if analog_val is not None else np.nan,
                "test_mean_q_score": _nanmean_or_nan(analog_test.q_score) if analog_test is not None else np.nan,
                "val_mean_candidates": _nanmean_or_nan(analog_val.candidate_count) if analog_val is not None else np.nan,
                "test_mean_candidates": _nanmean_or_nan(analog_test.candidate_count) if analog_test is not None else np.nan,
                "val_mean_non_peak_neighbors": _nanmean_or_nan(analog_val.non_peak_count) if analog_val is not None else np.nan,
                "test_mean_non_peak_neighbors": _nanmean_or_nan(analog_test.non_peak_count) if analog_test is not None else np.nan,
            }
            if enable_analog
            else {
                "enabled": False,
                "reason": "skip_analog_blend",
            }
        ),
    }

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    report_lines = [
        "# KLGA Same-Day Tmax Distribution Run",
        "",
        f"- run_id: {run_dir.name}",
        f"- feature_store_path: {feature_store_path}",
        f"- rows_total: {len(df)}",
        f"- split_rows: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}",
        "",
        "## Combined Distribution Metrics (Blended)",
        f"- val_nll: {blend_val_metrics.get('nll')}",
        f"- val_top1_accuracy: {blend_val_metrics.get('top1_accuracy')}",
        f"- test_nll: {blend_test_metrics.get('nll')}",
        f"- test_top1_accuracy: {blend_test_metrics.get('top1_accuracy')}",
        "",
        "## Peak Metrics",
        f"- val_logloss_cal: {peak_metrics['val']['logloss_cal']}",
        f"- val_brier_cal: {peak_metrics['val']['brier_cal']}",
        f"- test_logloss_cal: {peak_metrics['test']['logloss_cal']}",
        f"- test_brier_cal: {peak_metrics['test']['brier_cal']}",
        "",
        "## Delta Metrics",
        f"- val_multi_logloss_temp: {delta_metrics['val']['multi_logloss_temp']}",
        f"- temperature: {delta_result.temperature}",
        "",
        "## Analog Blend",
        f"- enabled: {bool(enable_analog)}",
    ]
    if enable_analog:
        report_lines.extend(
            [
                f"- selected_k: {best_k}",
                f"- q_low: {q_low}",
                f"- q_high: {q_high}",
            ]
        )
    else:
        report_lines.append("- skipped: true")
    (run_dir / "metrics.md").write_text("\n".join(report_lines), encoding="utf-8")

    # Save config snapshot.
    cfg_payload = {
        "target_station_id": cfg.target_station_id,
        "neighbor_station_ids": list(cfg.neighbor_station_ids),
        "enable_analog": bool(enable_analog),
        "cutoff_start": f"{cfg.cutoff_start_hour:02d}:{cfg.cutoff_start_minute:02d}",
        "cutoff_end": f"{cfg.cutoff_end_hour:02d}:{cfg.cutoff_end_minute:02d}",
        "cutoff_step_minutes": int(cfg.cutoff_step_minutes),
        "windows_minutes": list(cfg.windows_minutes),
        "delta_class_max": int(cfg.delta_class_max),
        "split": {
            "train_start": str(cfg.split.train_start),
            "train_end": str(cfg.split.train_end),
            "val_start": str(cfg.split.val_start),
            "val_end": str(cfg.split.val_end),
            "test_start": str(cfg.split.test_start),
            "test_end": str(cfg.split.test_end),
        },
        "analog_feature_columns": list(cfg.analog_feature_columns),
        "analog_feature_weights": cfg.analog_feature_weights,
    }
    (run_dir / "config.json").write_text(json.dumps(cfg_payload, indent=2, sort_keys=True), encoding="utf-8")
    stage_end(
        sidx,
        "write_artifacts",
        st0,
        details=f"metrics_path={metrics_path}",
    )
    logger.info(
        "PIPELINE_DONE elapsed=%s run_dir=%s",
        format_duration(time.perf_counter() - pipeline_start),
        run_dir,
    )

    return PipelineRunResult(run_dir=run_dir, metrics_path=metrics_path, metrics=metrics)
