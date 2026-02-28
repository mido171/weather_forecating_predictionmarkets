from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import logging
import threading
import time

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss
from .logging_utils import format_duration


@dataclass(frozen=True)
class PeakTrainResult:
    model: Any
    isotonic: IsotonicRegression
    val_metrics: dict[str, Any]
    class_balance: dict[str, float]


def _binary_reliability(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> dict[str, list[float]]:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    centers: list[float] = []
    pred_mean: list[float] = []
    obs_rate: list[float] = []
    counts: list[float] = []
    for i in range(bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        centers.append(float((lo + hi) * 0.5))
        pred_mean.append(float(np.mean(y_prob[mask])))
        obs_rate.append(float(np.mean(y_true[mask])))
        counts.append(float(np.sum(mask)))
    return {
        "bin_center": centers,
        "pred_mean": pred_mean,
        "obs_rate": obs_rate,
        "count": counts,
    }


def _lgbm_progress_callback(
    *,
    logger: logging.Logger,
    stage_label: str,
    total_iterations: int,
    period: int,
    log_every_seconds: float,
    state: dict[str, Any],
    lock: threading.Lock,
):
    start = time.perf_counter()
    last_emit = start

    def _callback(env) -> None:
        nonlocal last_emit
        it = int(env.iteration) + 1
        now = time.perf_counter()
        should_log_period = period > 0 and (it % period == 0)
        should_log_time = (now - last_emit) >= max(float(log_every_seconds), 0.1)
        should_log_final = it >= total_iterations

        eval_parts: list[str] = []
        for item in (env.evaluation_result_list or []):
            if len(item) >= 3:
                data_name = item[0]
                metric_name = item[1]
                metric_val = item[2]
                eval_parts.append(f"{data_name}.{metric_name}={metric_val:.6f}")
        eval_txt = " ".join(eval_parts)

        with lock:
            state["last_iter"] = it
            state["last_eval"] = eval_txt
            state["last_update"] = now

        if not should_log_period and not should_log_time and not should_log_final:
            return
        elapsed = now - start
        rate = it / elapsed if elapsed > 0 else 0.0
        remain = max(total_iterations - it, 0)
        eta_s = (remain / rate) if rate > 0 else float("inf")
        eta_txt = format_duration(eta_s) if eta_s < float("inf") else "?:??"
        pct = 100.0 * it / max(total_iterations, 1)
        logger.info(
            "%s iter=%d/%d pct=%.2f%% elapsed=%s eta=%s %s",
            stage_label,
            it,
            total_iterations,
            pct,
            format_duration(elapsed),
            eta_txt,
            eval_txt,
        )
        last_emit = now

    _callback.order = 10
    _callback.before_iteration = False
    return _callback


def _start_training_heartbeat(
    *,
    logger: logging.Logger,
    stage_label: str,
    total_iterations: int,
    heartbeat_seconds: float,
    state: dict[str, Any],
    lock: threading.Lock,
) -> tuple[threading.Event, threading.Thread]:
    stop = threading.Event()
    interval = max(float(heartbeat_seconds), 1.0)
    start = time.perf_counter()

    def _run() -> None:
        while not stop.wait(interval):
            now = time.perf_counter()
            with lock:
                it = int(state.get("last_iter", 0))
                last_eval = str(state.get("last_eval", ""))
                last_update = float(state.get("last_update", start))
            elapsed = now - start
            pct = 100.0 * it / max(total_iterations, 1)
            rate = it / elapsed if elapsed > 0 else 0.0
            remain = max(total_iterations - it, 0)
            eta_s = (remain / rate) if rate > 0 else float("inf")
            eta_txt = format_duration(eta_s) if eta_s < float("inf") else "?:??"
            since_update = max(now - last_update, 0.0)
            logger.info(
                "%s heartbeat iter=%d/%d pct=%.2f%% elapsed=%s eta=%s idle=%s %s",
                stage_label,
                it,
                total_iterations,
                pct,
                format_duration(elapsed),
                eta_txt,
                format_duration(since_update),
                last_eval,
            )

    thread = threading.Thread(target=_run, name=f"{stage_label}_heartbeat", daemon=True)
    thread.start()
    return stop, thread


def train_peak_model(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_weight_train: np.ndarray | None = None,
    categorical_feature: list[int] | tuple[int, ...] | None = None,
    params_override: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
    log_period: int = 50,
    log_every_seconds: float = 10.0,
    heartbeat_seconds: float = 10.0,
    stage_label: str = "PEAK_TRAIN",
) -> PeakTrainResult:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("lightgbm is required for peak training.") from exc

    active_logger = logger or logging.getLogger(__name__)
    y_train = np.asarray(y_train, dtype=int)
    y_val = np.asarray(y_val, dtype=int)
    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 64,
        "learning_rate": 0.03,
        "n_estimators": 5000,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }
    if params_override:
        params.update(params_override)

    model = lgb.LGBMClassifier(**params)
    total_iters = int(params.get("n_estimators", 0) or 0)
    active_logger.info(
        "%s setup train_rows=%d val_rows=%d total_iters=%d",
        stage_label,
        X_train.shape[0],
        X_val.shape[0],
        total_iters,
    )
    callbacks = [
        lgb.early_stopping(stopping_rounds=200, verbose=False),
    ]
    state_lock = threading.Lock()
    state: dict[str, Any] = {
        "last_iter": 0,
        "last_eval": "",
        "last_update": time.perf_counter(),
    }
    callbacks.append(
        _lgbm_progress_callback(
            logger=active_logger,
            stage_label=stage_label,
            total_iterations=max(total_iters, 1),
            period=max(int(log_period), 1),
            log_every_seconds=float(log_every_seconds),
            state=state,
            lock=state_lock,
        )
    )
    hb_stop, hb_thread = _start_training_heartbeat(
        logger=active_logger,
        stage_label=stage_label,
        total_iterations=max(total_iters, 1),
        heartbeat_seconds=float(heartbeat_seconds),
        state=state,
        lock=state_lock,
    )
    try:
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            categorical_feature=list(categorical_feature) if categorical_feature is not None else "auto",
            callbacks=callbacks,
        )
    finally:
        hb_stop.set()
        hb_thread.join(timeout=2.0)
    active_logger.info(
        "%s finished best_iteration=%s",
        stage_label,
        getattr(model, "best_iteration_", None),
    )

    p_val_raw = model.predict_proba(X_val)[:, 1]
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(p_val_raw, y_val)
    p_val_cal = np.clip(isotonic.predict(p_val_raw), 1e-6, 1.0 - 1e-6)

    val_metrics = {
        "logloss_raw": float(log_loss(y_val, np.clip(p_val_raw, 1e-6, 1.0 - 1e-6))),
        "logloss_cal": float(log_loss(y_val, p_val_cal)),
        "brier_raw": float(brier_score_loss(y_val, p_val_raw)),
        "brier_cal": float(brier_score_loss(y_val, p_val_cal)),
    }
    val_metrics["reliability_raw"] = _binary_reliability(y_val, p_val_raw)  # type: ignore[assignment]
    val_metrics["reliability_cal"] = _binary_reliability(y_val, p_val_cal)  # type: ignore[assignment]

    return PeakTrainResult(
        model=model,
        isotonic=isotonic,
        val_metrics=val_metrics,
        class_balance={
            "train_pos_rate": float(np.mean(y_train)),
            "train_pos_count": float(pos),
            "train_neg_count": float(neg),
        },
    )


def predict_peak_probability(
    *,
    model: Any,
    isotonic: IsotonicRegression,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p_raw = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    p_cal = np.asarray(isotonic.predict(p_raw), dtype=float)
    return np.clip(p_raw, 0.0, 1.0), np.clip(p_cal, 0.0, 1.0)
