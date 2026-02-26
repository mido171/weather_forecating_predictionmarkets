from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import logging
import threading
import time

import numpy as np
from .logging_utils import format_duration


@dataclass(frozen=True)
class DeltaTrainResult:
    model: Any
    temperature: float
    val_metrics: dict[str, float]
    class_counts: dict[str, float]


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    denom = np.sum(exp, axis=1, keepdims=True)
    return exp / np.maximum(denom, 1e-12)


def _multiclass_temperature_scale(logits: np.ndarray, y_true: np.ndarray) -> float:
    try:
        from scipy.optimize import minimize_scalar
    except ImportError as exc:
        raise ImportError("scipy is required for multiclass temperature scaling.") from exc

    y_true = np.asarray(y_true, dtype=int)
    logits = np.asarray(logits, dtype=float)

    def objective(log_temp: float) -> float:
        temp = float(np.exp(log_temp))
        probs = _softmax(logits / max(temp, 1e-6))
        p_true = np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1.0)
        return float(-np.mean(np.log(p_true)))

    res = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
    if not res.success:
        return 1.0
    return float(np.exp(res.x))


def _multi_logloss(y_true: np.ndarray, probs: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    p_true = np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1.0)
    return float(-np.mean(np.log(p_true)))


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


def train_delta_model(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    sample_weight_train: np.ndarray | None = None,
    params_override: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
    log_period: int = 25,
    log_every_seconds: float = 10.0,
    heartbeat_seconds: float = 10.0,
    stage_label: str = "DELTA_TRAIN",
) -> DeltaTrainResult:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("lightgbm is required for delta training.") from exc

    active_logger = logger or logging.getLogger(__name__)
    y_train = np.asarray(y_train, dtype=int)
    y_val = np.asarray(y_val, dtype=int)

    counts = np.bincount(y_train, minlength=num_classes).astype(float)
    class_weight = np.where(counts > 0, counts.sum() / np.maximum(counts, 1.0), 1.0)
    class_weight = class_weight / np.nanmean(class_weight)
    per_row_class_weight = class_weight[y_train]
    if sample_weight_train is None:
        combined_weight = per_row_class_weight
    else:
        combined_weight = np.asarray(sample_weight_train, dtype=float) * per_row_class_weight

    params = {
        "boosting_type": "gbdt",
        "objective": "multiclass",
        "num_class": int(num_classes),
        "metric": "multi_logloss",
        "num_leaves": 128,
        "learning_rate": 0.03,
        "n_estimators": 8000,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }
    if params_override:
        params.update(params_override)

    model = lgb.LGBMClassifier(**params)
    total_iters = int(params.get("n_estimators", 0) or 0)
    active_logger.info(
        "%s setup train_rows=%d val_rows=%d num_classes=%d total_iters=%d",
        stage_label,
        X_train.shape[0],
        X_val.shape[0],
        num_classes,
        total_iters,
    )
    callbacks = [
        lgb.early_stopping(stopping_rounds=300, verbose=False),
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
            sample_weight=combined_weight,
            eval_set=[(X_val, y_val)],
            eval_metric="multi_logloss",
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

    val_logits = np.asarray(model.predict(X_val, raw_score=True), dtype=float)
    val_probs_raw = _softmax(val_logits)
    temperature = _multiclass_temperature_scale(val_logits, y_val)
    val_probs_temp = _softmax(val_logits / max(temperature, 1e-6))

    val_metrics = {
        "multi_logloss_raw": _multi_logloss(y_val, val_probs_raw),
        "multi_logloss_temp": _multi_logloss(y_val, val_probs_temp),
        "temperature": float(temperature),
    }
    class_counts = {f"class_{k}": float(v) for k, v in enumerate(counts)}
    return DeltaTrainResult(
        model=model,
        temperature=float(temperature),
        val_metrics=val_metrics,
        class_counts=class_counts,
    )


def predict_delta_conditional(
    *,
    model: Any,
    temperature: float,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits = np.asarray(model.predict(X, raw_score=True), dtype=float)
    probs_raw = _softmax(logits)
    probs_temp = _softmax(logits / max(float(temperature), 1e-6))
    return logits, probs_raw, probs_temp
