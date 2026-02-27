from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
import copy
import json
import logging
import math
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

from weather_ml.klga_daily_tmax_dist.config import (
    OBS_ALLOWED_COLUMNS,
    OBS_OPTIONAL_COLUMNS,
    PipelineConfig,
    SplitConfig,
)
from weather_ml.klga_daily_tmax_dist.features import (
    build_daily_prior_frame,
    build_feature_rows,
    prepare_station_series,
)
from weather_ml.klga_daily_tmax_dist.logging_utils import format_duration
from weather_ml.klga_daily_tmax_dist.pipeline import (
    _add_climo_features,
    _apply_imputer,
    _build_full_delta_arrays,
    _cutoff_metrics,
    _delta_logloss_by_group,
    _expand_class_probs,
    _evaluate_distribution_rows,
    _fit_imputer,
    _model_feature_columns,
    _recency_weights,
    _temperature_bucket_calibration,
    _uv_quantile_labels,
)
from weather_ml.klga_daily_tmax_dist.timegrid import make_calendar_grid
from weather_ml.klga_daily_tmax_dist.train_delta import (
    _multi_logloss,
    _multiclass_temperature_scale,
    _softmax,
    predict_delta_conditional,
    train_delta_model,
)
from weather_ml.klga_daily_tmax_dist.train_peak import predict_peak_probability, train_peak_model


REQUIRED_EXPORT_FILES = (
    "daily_max_truth_klga.csv",
    "observations_30m_required_columns.csv",
    "station_universe.csv",
)


@dataclass(frozen=True)
class TabMTrainingConfig:
    data_dir: Path
    output_root: Path
    split: SplitConfig
    max_epochs_peak: int = 8
    max_epochs_delta: int = 8
    batch_size: int = 4096
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 3
    device: str = "auto"
    tabm_arch_type: str = "tabm"
    tabm_k: int = 32
    tabm_n_blocks: int = 3
    tabm_d_block: int = 256
    tabm_dropout: float = 0.2
    tabm_start_scaling_init: str | None = "random-signs"
    seed: int = 42
    log_every_batches: int = 50
    log_every_rows: int = 2000
    log_every_seconds: float = 20.0


@dataclass(frozen=True)
class TabMTrainingResult:
    run_dir: Path
    metrics_path: Path
    metrics: dict[str, Any]


@dataclass(frozen=True)
class LGBMExportsTrainingConfig:
    data_dir: Path
    output_root: Path
    split: SplitConfig
    include_feels_like: bool = False
    delta_objective: str = "multiclass"
    delta_use_class_weights: bool = True
    delta_use_cutoff_weights: bool = False
    delta_cutoff_weight_alpha: float = 1.0
    log_every_rows: int = 2000
    log_every_seconds: float = 20.0
    peak_train_log_period: int = 50
    delta_train_log_period: int = 25
    train_log_every_seconds: float = 10.0
    train_heartbeat_seconds: float = 10.0


@dataclass(frozen=True)
class LGBMExportsTrainingResult:
    run_dir: Path
    metrics_path: Path
    metrics: dict[str, Any]


def _timestamp_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _attach_run_file_handler(logger: logging.Logger, run_log_path: Path) -> None:
    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                if Path(handler.baseFilename).resolve() == run_log_path.resolve():
                    return
            except Exception:
                continue
    file_handler = logging.FileHandler(run_log_path, encoding="utf-8")
    file_handler.setLevel(logger.level or logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(file_handler)


def _write_df_with_csv_parquet(df: pd.DataFrame, out_no_ext: Path, logger: logging.Logger) -> None:
    csv_path = out_no_ext.with_suffix(".csv")
    parquet_path = out_no_ext.with_suffix(".parquet")
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as exc:
        logger.warning("WRITE_PARQUET_FAILED path=%s error=%s", parquet_path, exc)


def _resolve_device(device: str) -> torch.device:
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _set_seed(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _require_export_files(data_dir: Path) -> None:
    missing = [x for x in REQUIRED_EXPORT_FILES if not (data_dir / x).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required export files in {data_dir}: {missing}")


def _load_daily_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    needed = {"request_location_id", "target_date_local", "max_temp_f", "station_zoneid"}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"daily_max_truth_klga.csv missing columns: {sorted(missing)}")
    df["target_date_local"] = pd.to_datetime(df["target_date_local"], errors="coerce").dt.date
    df["max_temp_f"] = pd.to_numeric(df["max_temp_f"], errors="coerce").round().astype("Int64")
    return df.dropna(subset=["target_date_local"]).copy()


def _load_obs_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = set(OBS_ALLOWED_COLUMNS).difference(set(OBS_OPTIONAL_COLUMNS))
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"observations_30m_required_columns.csv missing columns: {sorted(missing)}")
    for optional in OBS_OPTIONAL_COLUMNS:
        if optional not in df.columns:
            df[optional] = np.nan
    df["valid_time_utc"] = pd.to_datetime(df["valid_time_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["valid_time_utc"]).copy()
    for c in ["temp", "dew_pt", "rh", "pressure", "vis", "wspd", "wdir", "gust", "precip_hrly", "uv_index", "feels_like"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _split_masks(df: pd.DataFrame, split: SplitConfig) -> dict[str, np.ndarray]:
    d = pd.to_datetime(df["target_date_local"]).dt.date
    return {
        "train": ((d >= split.train_start) & (d <= split.train_end)).to_numpy(),
        "val": ((d >= split.val_start) & (d <= split.val_end)).to_numpy(),
        "test": ((d >= split.test_start) & (d <= split.test_end)).to_numpy(),
    }


def _iter_batches(n: int, batch_size: int, rng: np.random.Generator):
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)
    for i in range(0, n, batch_size):
        yield idx[i : i + batch_size]


def _predict_peak_probs(model: Any, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    import torch

    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[i : i + batch_size]).to(device=device, dtype=torch.float32)
            logits = model(x_num=xb).squeeze(-1)
            p = torch.sigmoid(logits).mean(dim=1)
            out.append(p.detach().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.array([], dtype=np.float64)


def _predict_delta_logits_probs(
    model: Any, x: np.ndarray, device: torch.device, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    all_logits: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[i : i + batch_size]).to(device=device, dtype=torch.float32)
            logits = model(x_num=xb).mean(dim=1)
            probs = torch.softmax(logits, dim=-1)
            all_logits.append(logits.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())
    if not all_logits:
        return np.empty((0, 0), dtype=np.float64), np.empty((0, 0), dtype=np.float64)
    return np.concatenate(all_logits, axis=0), np.concatenate(all_probs, axis=0)


def _fit_peak(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TabMTrainingConfig,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[Any, IsotonicRegression, dict[str, Any]]:
    import tabm
    import torch
    import torch.nn.functional as F

    model = tabm.TabM.make(
        n_num_features=x_train.shape[1],
        cat_cardinalities=None,
        d_out=1,
        k=cfg.tabm_k,
        arch_type=cfg.tabm_arch_type,
        n_blocks=cfg.tabm_n_blocks,
        d_block=cfg.tabm_d_block,
        dropout=cfg.tabm_dropout,
        start_scaling_init=cfg.tabm_start_scaling_init,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed)

    best_epoch = 0
    best_val = math.inf
    best_state = copy.deepcopy(model.state_dict())
    patience_left = cfg.patience

    for epoch in range(1, cfg.max_epochs_peak + 1):
        epoch_start = time.perf_counter()
        model.train()
        n_batches = max(int(math.ceil(len(x_train) / max(cfg.batch_size, 1))), 1)
        active_logger_every = max(cfg.log_every_batches, 1)
        for bi, bidx in enumerate(_iter_batches(len(x_train), cfg.batch_size, rng), start=1):
            xb = torch.from_numpy(x_train[bidx]).to(device=device, dtype=torch.float32)
            yb = torch.from_numpy(y_train[bidx]).to(device=device, dtype=torch.float32)
            logits = model(x_num=xb).squeeze(-1)
            target = yb[:, None].expand_as(logits)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            if bi % active_logger_every == 0 or bi == n_batches:
                pct = 100.0 * float(bi) / float(n_batches)
                logger.info(
                    "TABM_PEAK_BATCH epoch=%d/%d batch=%d/%d (%.1f%%) loss=%.6f",
                    epoch,
                    cfg.max_epochs_peak,
                    bi,
                    n_batches,
                    pct,
                    float(loss.item()),
                )

        p_val_raw = _predict_peak_probs(model, x_val, device, cfg.batch_size)
        val_ll = float(log_loss(y_val, np.clip(p_val_raw, 1e-6, 1 - 1e-6)))
        logger.info(
            "TABM_PEAK_EPOCH epoch=%d/%d val_logloss=%.6f elapsed=%s",
            epoch,
            cfg.max_epochs_peak,
            val_ll,
            format_duration(time.perf_counter() - epoch_start),
        )
        if val_ll < best_val - 1e-6:
            best_val = val_ll
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    p_val_raw = _predict_peak_probs(model, x_val, device, cfg.batch_size)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val_raw, y_val)
    p_val_cal = np.clip(iso.predict(p_val_raw), 1e-6, 1 - 1e-6)
    return model, iso, {
        "best_epoch": int(best_epoch),
        "val_logloss_raw": float(log_loss(y_val, np.clip(p_val_raw, 1e-6, 1 - 1e-6))),
        "val_logloss_cal": float(log_loss(y_val, p_val_cal)),
        "val_brier_raw": float(brier_score_loss(y_val, p_val_raw)),
        "val_brier_cal": float(brier_score_loss(y_val, p_val_cal)),
    }


def _fit_delta(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    cfg: TabMTrainingConfig,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[Any, float, dict[str, Any]]:
    import tabm
    import torch
    import torch.nn.functional as F

    model = tabm.TabM.make(
        n_num_features=x_train.shape[1],
        cat_cardinalities=None,
        d_out=num_classes,
        k=cfg.tabm_k,
        arch_type=cfg.tabm_arch_type,
        n_blocks=cfg.tabm_n_blocks,
        d_block=cfg.tabm_d_block,
        dropout=cfg.tabm_dropout,
        start_scaling_init=cfg.tabm_start_scaling_init,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed + 1)

    best_epoch = 0
    best_val = math.inf
    best_state = copy.deepcopy(model.state_dict())
    patience_left = cfg.patience

    for epoch in range(1, cfg.max_epochs_delta + 1):
        epoch_start = time.perf_counter()
        model.train()
        n_batches = max(int(math.ceil(len(x_train) / max(cfg.batch_size, 1))), 1)
        active_logger_every = max(cfg.log_every_batches, 1)
        for bi, bidx in enumerate(_iter_batches(len(x_train), cfg.batch_size, rng), start=1):
            xb = torch.from_numpy(x_train[bidx]).to(device=device, dtype=torch.float32)
            yb = torch.from_numpy(y_train[bidx]).to(device=device, dtype=torch.long)
            logits_ens = model(x_num=xb)  # (B, K, C)
            k = logits_ens.shape[1]
            loss = F.cross_entropy(logits_ens.reshape(-1, num_classes), yb.repeat_interleave(k))
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            if bi % active_logger_every == 0 or bi == n_batches:
                pct = 100.0 * float(bi) / float(n_batches)
                logger.info(
                    "TABM_DELTA_BATCH epoch=%d/%d batch=%d/%d (%.1f%%) loss=%.6f",
                    epoch,
                    cfg.max_epochs_delta,
                    bi,
                    n_batches,
                    pct,
                    float(loss.item()),
                )

        v_logits, v_probs = _predict_delta_logits_probs(model, x_val, device, cfg.batch_size)
        v_ll = float(_multi_logloss(y_val, v_probs))
        logger.info(
            "TABM_DELTA_EPOCH epoch=%d/%d val_multi_logloss=%.6f elapsed=%s",
            epoch,
            cfg.max_epochs_delta,
            v_ll,
            format_duration(time.perf_counter() - epoch_start),
        )
        if v_ll < best_val - 1e-6:
            best_val = v_ll
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    val_logits, val_probs_raw = _predict_delta_logits_probs(model, x_val, device, cfg.batch_size)
    temperature = float(_multiclass_temperature_scale(val_logits, y_val))
    val_probs_temp = _softmax(val_logits / max(temperature, 1e-6))
    return model, temperature, {
        "best_epoch": int(best_epoch),
        "temperature": float(temperature),
        "val_multi_logloss_raw": float(_multi_logloss(y_val, val_probs_raw)),
        "val_multi_logloss_temp": float(_multi_logloss(y_val, val_probs_temp)),
    }


def run_tabm_training_from_exports(
    *,
    cfg: TabMTrainingConfig,
    logger: logging.Logger | None = None,
) -> TabMTrainingResult:
    import torch

    active_logger = logger or logging.getLogger(__name__)
    _set_seed(cfg.seed)
    run_dir = _ensure_dir(cfg.output_root / _timestamp_id())
    models_dir = _ensure_dir(run_dir / "models")
    reports_dir = _ensure_dir(run_dir / "reports")
    predictions_dir = _ensure_dir(run_dir / "predictions")
    _attach_run_file_handler(active_logger, run_dir / "run.log")
    pipeline_start = time.perf_counter()
    active_logger.info("TABM_EXPORT_RUN_START run_dir=%s data_dir=%s", run_dir, cfg.data_dir)

    stage_total = 9
    stage_idx = 0

    def stage_start(name: str, details: str = "") -> tuple[int, float]:
        nonlocal stage_idx
        stage_idx += 1
        pct = ((stage_idx - 1) / stage_total) * 100.0
        active_logger.info(
            "STAGE_START [%d/%d %.1f%%] %s %s",
            stage_idx,
            stage_total,
            pct,
            name,
            details,
        )
        return stage_idx, time.perf_counter()

    def stage_end(idx: int, name: str, st: float, details: str = "") -> None:
        pct = (idx / stage_total) * 100.0
        active_logger.info(
            "STAGE_END   [%d/%d %.1f%%] %s elapsed=%s %s",
            idx,
            stage_total,
            pct,
            name,
            format_duration(time.perf_counter() - st),
            details,
        )

    sidx, st0 = stage_start("validate_input_files")
    _require_export_files(cfg.data_dir)
    stage_end(sidx, "validate_input_files", st0)

    sidx, st0 = stage_start("load_raw_csvs")
    daily_df = _load_daily_csv(cfg.data_dir / "daily_max_truth_klga.csv")
    daily_df = daily_df[
        (daily_df["target_date_local"] >= cfg.split.train_start)
        & (daily_df["target_date_local"] <= cfg.split.test_end)
    ].copy()
    if daily_df.empty:
        raise ValueError("No daily rows in split horizon.")
    obs_df = _load_obs_csv(cfg.data_dir / "observations_30m_required_columns.csv")
    stage_end(
        sidx,
        "load_raw_csvs",
        st0,
        details=f"daily_rows={len(daily_df)} obs_rows={len(obs_df)}",
    )

    sidx, st0 = stage_start("build_feature_rows")
    p_cfg = PipelineConfig(split=cfg.split, output_root=cfg.output_root)
    calendar_df = make_calendar_grid(sorted(set(daily_df["target_date_local"])), tz=p_cfg.local_zone)
    start_obs_utc = pd.Timestamp(calendar_df["midnight_utc"].min()).tz_convert("UTC") - pd.Timedelta(hours=6)
    end_obs_utc = pd.Timestamp(calendar_df["cutoff_utc"].max()).tz_convert("UTC")
    obs_df = obs_df[(obs_df["valid_time_utc"] >= start_obs_utc) & (obs_df["valid_time_utc"] <= end_obs_utc)].copy()
    if obs_df.empty:
        raise ValueError("No observation rows after horizon filtering.")
    station_series = prepare_station_series(
        obs_df,
        station_ids=p_cfg.all_station_ids,
        include_feels_like=p_cfg.include_feels_like,
    )
    daily_prior_df = build_daily_prior_frame(daily_df)
    feat_df, audit = build_feature_rows(
        calendar_df=calendar_df,
        station_series=station_series,
        daily_truth_df=daily_df,
        daily_prior_df=daily_prior_df,
        cfg=p_cfg,
        logger=active_logger,
        log_every_rows=cfg.log_every_rows,
        log_every_seconds=cfg.log_every_seconds,
    )
    stage_end(sidx, "build_feature_rows", st0, details=f"feature_rows={len(feat_df)}")

    sidx, st0 = stage_start("prepare_training_matrices")
    masks = _split_masks(feat_df, cfg.split)
    feat_df, climo_meta = _add_climo_features(feat_df, masks["train"])
    feat_cols = _model_feature_columns(feat_df, p_cfg)
    medians = _fit_imputer(feat_df, feature_cols=feat_cols, train_mask=masks["train"])
    x_all = _apply_imputer(feat_df, feature_cols=feat_cols, medians=medians).astype(np.float32)

    peak_series = pd.to_numeric(feat_df["peak"], errors="coerce")
    delta_series = pd.to_numeric(feat_df["delta"], errors="coerce")
    peak_mask = np.isfinite(peak_series.to_numpy(dtype=float))
    delta_mask = np.isfinite(delta_series.to_numpy(dtype=float))
    y_peak_all = np.full(len(feat_df), -1, dtype=int)
    y_delta_all = np.full(len(feat_df), -1, dtype=int)
    y_peak_all[peak_mask] = peak_series.loc[peak_mask].round().astype(int).to_numpy()
    y_delta_all[delta_mask] = delta_series.loc[delta_mask].round().astype(int).to_numpy()

    peak_train_idx = np.where(masks["train"] & peak_mask)[0]
    peak_val_idx = np.where(masks["val"] & peak_mask)[0]
    peak_test_idx = np.where(masks["test"] & peak_mask)[0]
    if len(peak_train_idx) == 0 or len(peak_val_idx) == 0 or len(peak_test_idx) == 0:
        raise ValueError("Peak split rows are empty.")

    delta_class_max = PipelineConfig().delta_class_max
    delta_train_idx = np.where(masks["train"] & peak_mask & delta_mask & (y_peak_all == 0) & (y_delta_all >= 1))[0]
    delta_val_idx = np.where(masks["val"] & peak_mask & delta_mask & (y_peak_all == 0) & (y_delta_all >= 1))[0]
    delta_test_idx = np.where(masks["test"] & peak_mask & delta_mask & (y_peak_all == 0) & (y_delta_all >= 1))[0]
    if len(delta_train_idx) == 0 or len(delta_val_idx) == 0 or len(delta_test_idx) == 0:
        raise ValueError("Delta split rows are empty.")

    y_delta_train = np.clip(y_delta_all[delta_train_idx], 1, delta_class_max) - 1
    y_delta_val = np.clip(y_delta_all[delta_val_idx], 1, delta_class_max) - 1
    y_delta_test = np.clip(y_delta_all[delta_test_idx], 1, delta_class_max) - 1
    device = _resolve_device(cfg.device)
    active_logger.info("TABM_DEVICE %s", device)
    stage_end(
        sidx,
        "prepare_training_matrices",
        st0,
        details=(
            f"features={len(feat_cols)} peak_train={len(peak_train_idx)} "
            f"peak_val={len(peak_val_idx)} peak_test={len(peak_test_idx)} "
            f"delta_train={len(delta_train_idx)} delta_val={len(delta_val_idx)} delta_test={len(delta_test_idx)}"
        ),
    )

    sidx, st0 = stage_start("train_peak_model")
    peak_model, peak_iso, peak_train = _fit_peak(
        x_train=x_all[peak_train_idx],
        y_train=y_peak_all[peak_train_idx].astype(np.int64),
        x_val=x_all[peak_val_idx],
        y_val=y_peak_all[peak_val_idx].astype(np.int64),
        cfg=cfg,
        device=device,
        logger=active_logger,
    )
    stage_end(sidx, "train_peak_model", st0, details=f"best_epoch={peak_train.get('best_epoch')}")

    sidx, st0 = stage_start("train_delta_model")
    delta_model, delta_temp, delta_train = _fit_delta(
        x_train=x_all[delta_train_idx],
        y_train=y_delta_train.astype(np.int64),
        x_val=x_all[delta_val_idx],
        y_val=y_delta_val.astype(np.int64),
        num_classes=delta_class_max,
        cfg=cfg,
        device=device,
        logger=active_logger,
    )
    stage_end(
        sidx,
        "train_delta_model",
        st0,
        details=f"best_epoch={delta_train.get('best_epoch')} temperature={float(delta_temp):.6f}",
    )

    sidx, st0 = stage_start("predict_probabilities")
    p_peak_val_raw = _predict_peak_probs(peak_model, x_all[peak_val_idx], device, cfg.batch_size)
    p_peak_test_raw = _predict_peak_probs(peak_model, x_all[peak_test_idx], device, cfg.batch_size)
    p_peak_val_cal = np.clip(peak_iso.predict(p_peak_val_raw), 1e-6, 1 - 1e-6)
    p_peak_test_cal = np.clip(peak_iso.predict(p_peak_test_raw), 1e-6, 1 - 1e-6)

    d_val_logits, d_val_probs_raw = _predict_delta_logits_probs(delta_model, x_all[delta_val_idx], device, cfg.batch_size)
    d_test_logits, d_test_probs_raw = _predict_delta_logits_probs(
        delta_model, x_all[delta_test_idx], device, cfg.batch_size
    )
    d_val_probs_temp = _softmax(d_val_logits / max(delta_temp, 1e-6))
    d_test_probs_temp = _softmax(d_test_logits / max(delta_temp, 1e-6))

    p_delta_val_cond = np.zeros((len(peak_val_idx), delta_class_max), dtype=float)
    p_delta_test_cond = np.zeros((len(peak_test_idx), delta_class_max), dtype=float)
    val_pos = {idx: i for i, idx in enumerate(peak_val_idx)}
    test_pos = {idx: i for i, idx in enumerate(peak_test_idx)}
    for j, idx in enumerate(delta_val_idx):
        i = val_pos.get(int(idx))
        if i is not None:
            p_delta_val_cond[i, :] = d_val_probs_temp[j]
    for j, idx in enumerate(delta_test_idx):
        i = test_pos.get(int(idx))
        if i is not None:
            p_delta_test_cond[i, :] = d_test_probs_temp[j]
    stage_end(sidx, "predict_probabilities", st0)

    sidx, st0 = stage_start("evaluate_metrics")
    combined_val, combined_val_detail = _evaluate_distribution_rows(
        df=feat_df,
        row_indices=peak_val_idx,
        p_peak=p_peak_val_cal,
        p_delta_cond=p_delta_val_cond,
    )
    combined_test, combined_test_detail = _evaluate_distribution_rows(
        df=feat_df,
        row_indices=peak_test_idx,
        p_peak=p_peak_test_cal,
        p_delta_cond=p_delta_test_cond,
    )
    stage_end(
        sidx,
        "evaluate_metrics",
        st0,
        details=f"combined_val_nll={combined_val.get('nll')} combined_test_nll={combined_test.get('nll')}",
    )

    sidx, st0 = stage_start("write_prediction_and_report_files")
    pred_val = _build_full_delta_arrays(
        full_len=len(feat_df),
        class_count=delta_class_max,
        row_indices=peak_val_idx,
        p_peak=p_peak_val_cal,
        p_delta=p_delta_val_cond,
    )
    pred_test = _build_full_delta_arrays(
        full_len=len(feat_df),
        class_count=delta_class_max,
        row_indices=peak_test_idx,
        p_peak=p_peak_test_cal,
        p_delta=p_delta_test_cond,
    )
    _write_df_with_csv_parquet(pred_val, predictions_dir / "predictions_val", active_logger)
    _write_df_with_csv_parquet(pred_test, predictions_dir / "predictions_test", active_logger)
    _write_df_with_csv_parquet(combined_val_detail, predictions_dir / "distribution_eval_val", active_logger)
    _write_df_with_csv_parquet(combined_test_detail, predictions_dir / "distribution_eval_test", active_logger)

    cutoff_val = _cutoff_metrics(df_detail=combined_val_detail)
    cutoff_test = _cutoff_metrics(df_detail=combined_test_detail)
    cutoff_val.to_csv(reports_dir / "cutoff_metrics_val.csv", index=False)
    cutoff_test.to_csv(reports_dir / "cutoff_metrics_test.csv", index=False)

    calib_val = _temperature_bucket_calibration(
        df=feat_df,
        row_indices=peak_val_idx,
        p_peak=p_peak_val_cal,
        p_delta_cond=p_delta_val_cond,
    )
    calib_test = _temperature_bucket_calibration(
        df=feat_df,
        row_indices=peak_test_idx,
        p_peak=p_peak_test_cal,
        p_delta_cond=p_delta_test_cond,
    )
    calib_val.to_csv(reports_dir / "bucket_calibration_val.csv", index=False)
    calib_test.to_csv(reports_dir / "bucket_calibration_test.csv", index=False)
    stage_end(sidx, "write_prediction_and_report_files", st0)

    sidx, st0 = stage_start("write_models_and_metadata")
    torch.save({"state_dict": peak_model.state_dict()}, models_dir / "tabm_peak_model.pt")
    torch.save({"state_dict": delta_model.state_dict()}, models_dir / "tabm_delta_model.pt")
    joblib.dump(peak_iso, models_dir / "peak_isotonic.pkl")
    (models_dir / "delta_temperature_T.json").write_text(
        json.dumps({"temperature": float(delta_temp)}, indent=2),
        encoding="utf-8",
    )
    (run_dir / "feature_list.json").write_text(json.dumps(feat_cols, indent=2), encoding="utf-8")
    (run_dir / "imputer_values.json").write_text(json.dumps(medians, indent=2, sort_keys=True), encoding="utf-8")
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

    metrics = {
        "run_id": run_dir.name,
        "data_dir": str(cfg.data_dir),
        "rows_total": int(len(feat_df)),
        "split_rows_peak": {
            "train": int(len(peak_train_idx)),
            "val": int(len(peak_val_idx)),
            "test": int(len(peak_test_idx)),
        },
        "split_rows_delta": {
            "train": int(len(delta_train_idx)),
            "val": int(len(delta_val_idx)),
            "test": int(len(delta_test_idx)),
        },
        "audit": audit,
        "climo_lookup_meta": climo_meta,
        "peak": {
            "val": {
                "logloss_raw": float(log_loss(y_peak_all[peak_val_idx], np.clip(p_peak_val_raw, 1e-6, 1 - 1e-6))),
                "logloss_cal": float(log_loss(y_peak_all[peak_val_idx], p_peak_val_cal)),
                "brier_raw": float(brier_score_loss(y_peak_all[peak_val_idx], p_peak_val_raw)),
                "brier_cal": float(brier_score_loss(y_peak_all[peak_val_idx], p_peak_val_cal)),
            },
            "test": {
                "logloss_raw": float(log_loss(y_peak_all[peak_test_idx], np.clip(p_peak_test_raw, 1e-6, 1 - 1e-6))),
                "logloss_cal": float(log_loss(y_peak_all[peak_test_idx], p_peak_test_cal)),
                "brier_raw": float(brier_score_loss(y_peak_all[peak_test_idx], p_peak_test_raw)),
                "brier_cal": float(brier_score_loss(y_peak_all[peak_test_idx], p_peak_test_cal)),
            },
            "train_details": peak_train,
        },
        "delta": {
            "val": {
                "multi_logloss_raw": float(_multi_logloss(y_delta_val, d_val_probs_raw)),
                "multi_logloss_temp": float(_multi_logloss(y_delta_val, d_val_probs_temp)),
            },
            "test": {
                "multi_logloss_raw": float(_multi_logloss(y_delta_test, d_test_probs_raw)),
                "multi_logloss_temp": float(_multi_logloss(y_delta_test, d_test_probs_temp)),
            },
            "temperature": float(delta_temp),
            "train_details": delta_train,
        },
        "combined": {"val": combined_val, "test": combined_test},
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    cfg_payload = {
        "data_dir": str(cfg.data_dir),
        "output_root": str(cfg.output_root),
        "split": {
            "train_start": str(cfg.split.train_start),
            "train_end": str(cfg.split.train_end),
            "val_start": str(cfg.split.val_start),
            "val_end": str(cfg.split.val_end),
            "test_start": str(cfg.split.test_start),
            "test_end": str(cfg.split.test_end),
        },
        "tabm": {
            "arch_type": cfg.tabm_arch_type,
            "k": cfg.tabm_k,
            "n_blocks": cfg.tabm_n_blocks,
            "d_block": cfg.tabm_d_block,
            "dropout": cfg.tabm_dropout,
            "start_scaling_init": cfg.tabm_start_scaling_init,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "max_epochs_peak": cfg.max_epochs_peak,
            "max_epochs_delta": cfg.max_epochs_delta,
            "patience": cfg.patience,
            "device": str(device),
            "seed": cfg.seed,
        },
    }
    (run_dir / "config.json").write_text(json.dumps(cfg_payload, indent=2, sort_keys=True), encoding="utf-8")

    metrics_md = [
        "# KLGA TabM Peak/Delta Run",
        "",
        f"- run_id: {run_dir.name}",
        f"- data_dir: {cfg.data_dir}",
        f"- rows_total: {len(feat_df)}",
        f"- split_rows_peak: train={len(peak_train_idx)} val={len(peak_val_idx)} test={len(peak_test_idx)}",
        f"- split_rows_delta: train={len(delta_train_idx)} val={len(delta_val_idx)} test={len(delta_test_idx)}",
        "",
        "## Peak",
        f"- val_logloss_cal: {metrics['peak']['val']['logloss_cal']}",
        f"- test_logloss_cal: {metrics['peak']['test']['logloss_cal']}",
        f"- val_brier_cal: {metrics['peak']['val']['brier_cal']}",
        f"- test_brier_cal: {metrics['peak']['test']['brier_cal']}",
        "",
        "## Delta",
        f"- val_multi_logloss_temp: {metrics['delta']['val']['multi_logloss_temp']}",
        f"- test_multi_logloss_temp: {metrics['delta']['test']['multi_logloss_temp']}",
        f"- temperature: {metrics['delta']['temperature']}",
        "",
        "## Combined",
        f"- val_nll: {metrics['combined']['val'].get('nll')}",
        f"- test_nll: {metrics['combined']['test'].get('nll')}",
        f"- val_top1_accuracy: {metrics['combined']['val'].get('top1_accuracy')}",
        f"- test_top1_accuracy: {metrics['combined']['test'].get('top1_accuracy')}",
    ]
    (run_dir / "metrics.md").write_text("\n".join(metrics_md), encoding="utf-8")
    stage_end(sidx, "write_models_and_metadata", st0, details=f"metrics_path={metrics_path}")

    active_logger.info(
        "TABM_EXPORT_RUN_DONE elapsed=%s run_dir=%s",
        format_duration(time.perf_counter() - pipeline_start),
        run_dir,
    )
    return TabMTrainingResult(run_dir=run_dir, metrics_path=metrics_path, metrics=metrics)


def run_lgbm_training_from_exports(
    *,
    cfg: LGBMExportsTrainingConfig,
    logger: logging.Logger | None = None,
) -> LGBMExportsTrainingResult:
    active_logger = logger or logging.getLogger(__name__)
    run_dir = _ensure_dir(cfg.output_root / _timestamp_id())
    models_dir = _ensure_dir(run_dir / "models")
    reports_dir = _ensure_dir(run_dir / "reports")
    predictions_dir = _ensure_dir(run_dir / "predictions")
    _attach_run_file_handler(active_logger, run_dir / "run.log")
    pipeline_start = time.perf_counter()
    active_logger.info("LGBM_EXPORT_RUN_START run_dir=%s data_dir=%s", run_dir, cfg.data_dir)

    stage_total = 10
    stage_idx = 0

    def stage_start(name: str, details: str = "") -> tuple[int, float]:
        nonlocal stage_idx
        stage_idx += 1
        pct = ((stage_idx - 1) / stage_total) * 100.0
        active_logger.info(
            "STAGE_START [%d/%d %.1f%%] %s %s",
            stage_idx,
            stage_total,
            pct,
            name,
            details,
        )
        return stage_idx, time.perf_counter()

    def stage_end(idx: int, name: str, st: float, details: str = "") -> None:
        pct = (idx / stage_total) * 100.0
        active_logger.info(
            "STAGE_END   [%d/%d %.1f%%] %s elapsed=%s %s",
            idx,
            stage_total,
            pct,
            name,
            format_duration(time.perf_counter() - st),
            details,
        )

    sidx, st0 = stage_start("validate_input_files")
    _require_export_files(cfg.data_dir)
    stage_end(sidx, "validate_input_files", st0)

    sidx, st0 = stage_start("load_raw_csvs")
    daily_df = _load_daily_csv(cfg.data_dir / "daily_max_truth_klga.csv")
    daily_df = daily_df[
        (daily_df["target_date_local"] >= cfg.split.train_start)
        & (daily_df["target_date_local"] <= cfg.split.test_end)
    ].copy()
    if daily_df.empty:
        raise ValueError("No daily rows in split horizon.")
    obs_df = _load_obs_csv(cfg.data_dir / "observations_30m_required_columns.csv")
    stage_end(
        sidx,
        "load_raw_csvs",
        st0,
        details=f"daily_rows={len(daily_df)} obs_rows={len(obs_df)}",
    )

    sidx, st0 = stage_start("build_feature_rows")
    p_cfg = PipelineConfig(
        split=cfg.split,
        output_root=cfg.output_root,
        include_feels_like=cfg.include_feels_like,
        delta_objective=cfg.delta_objective,
        delta_use_class_weights=cfg.delta_use_class_weights,
        delta_use_cutoff_weights=cfg.delta_use_cutoff_weights,
        delta_cutoff_weight_alpha=cfg.delta_cutoff_weight_alpha,
    )
    calendar_df = make_calendar_grid(sorted(set(daily_df["target_date_local"])), tz=p_cfg.local_zone)
    start_obs_utc = pd.Timestamp(calendar_df["midnight_utc"].min()).tz_convert("UTC") - pd.Timedelta(hours=6)
    end_obs_utc = pd.Timestamp(calendar_df["cutoff_utc"].max()).tz_convert("UTC")
    obs_df = obs_df[(obs_df["valid_time_utc"] >= start_obs_utc) & (obs_df["valid_time_utc"] <= end_obs_utc)].copy()
    if obs_df.empty:
        raise ValueError("No observation rows after horizon filtering.")
    station_series = prepare_station_series(
        obs_df,
        station_ids=p_cfg.all_station_ids,
        include_feels_like=p_cfg.include_feels_like,
    )
    daily_prior_df = build_daily_prior_frame(daily_df)
    feat_df, audit = build_feature_rows(
        calendar_df=calendar_df,
        station_series=station_series,
        daily_truth_df=daily_df,
        daily_prior_df=daily_prior_df,
        cfg=p_cfg,
        logger=active_logger,
        log_every_rows=cfg.log_every_rows,
        log_every_seconds=cfg.log_every_seconds,
    )
    stage_end(sidx, "build_feature_rows", st0, details=f"feature_rows={len(feat_df)}")

    sidx, st0 = stage_start("prepare_training_matrices")
    masks = _split_masks(feat_df, cfg.split)
    state_lookup_path = models_dir / f"state_climo_lookup_{p_cfg.feature_contract_version}.json"
    feat_df, climo_meta = _add_climo_features(
        feat_df,
        masks["train"],
        cfg=p_cfg,
        state_lookup_path=state_lookup_path,
    )
    feat_cols = _model_feature_columns(feat_df, p_cfg)
    medians = _fit_imputer(feat_df, feature_cols=feat_cols, train_mask=masks["train"])
    x_all = _apply_imputer(feat_df, feature_cols=feat_cols, medians=medians)

    peak_series = pd.to_numeric(feat_df["peak"], errors="coerce")
    delta_series = pd.to_numeric(feat_df["delta"], errors="coerce")
    peak_mask = np.isfinite(peak_series.to_numpy(dtype=float))
    delta_mask = np.isfinite(delta_series.to_numpy(dtype=float))

    y_peak_all = np.full(len(feat_df), -1, dtype=int)
    y_delta_all = np.full(len(feat_df), -1, dtype=int)
    y_peak_all[peak_mask] = peak_series.loc[peak_mask].round().astype(int).to_numpy()
    y_delta_all[delta_mask] = delta_series.loc[delta_mask].round().astype(int).to_numpy()

    peak_train_mask = masks["train"] & peak_mask
    peak_val_mask = masks["val"] & peak_mask
    peak_test_mask = masks["test"] & peak_mask
    peak_train_idx = np.where(peak_train_mask)[0]
    peak_val_idx = np.where(peak_val_mask)[0]
    peak_test_idx = np.where(peak_test_mask)[0]
    if len(peak_train_idx) == 0 or len(peak_val_idx) == 0 or len(peak_test_idx) == 0:
        raise ValueError("Peak split rows are empty.")

    train_weights = _recency_weights(feat_df, peak_train_mask)

    delta_class_max = p_cfg.delta_class_max
    delta_train_idx = np.where(masks["train"] & peak_mask & delta_mask & (y_peak_all == 0) & (y_delta_all >= 1))[0]
    delta_val_idx = np.where(masks["val"] & peak_mask & delta_mask & (y_peak_all == 0) & (y_delta_all >= 1))[0]
    delta_test_idx = np.where(masks["test"] & peak_mask & delta_mask & (y_peak_all == 0) & (y_delta_all >= 1))[0]
    if len(delta_train_idx) == 0 or len(delta_val_idx) == 0:
        raise ValueError("Delta train/val split rows are empty.")

    y_delta_train = np.clip(y_delta_all[delta_train_idx], 1, delta_class_max) - 1
    y_delta_val = np.clip(y_delta_all[delta_val_idx], 1, delta_class_max) - 1
    y_delta_test = np.clip(y_delta_all[delta_test_idx], 1, delta_class_max) - 1
    stage_end(
        sidx,
        "prepare_training_matrices",
        st0,
        details=(
            f"features={len(feat_cols)} peak_train={len(peak_train_idx)} "
            f"peak_val={len(peak_val_idx)} peak_test={len(peak_test_idx)} "
            f"delta_train={len(delta_train_idx)} delta_val={len(delta_val_idx)} delta_test={len(delta_test_idx)}"
        ),
    )

    sidx, st0 = stage_start("train_peak_model")
    peak_result = train_peak_model(
        X_train=x_all[peak_train_idx],
        y_train=y_peak_all[peak_train_idx],
        X_val=x_all[peak_val_idx],
        y_val=y_peak_all[peak_val_idx],
        sample_weight_train=train_weights,
        logger=active_logger,
        log_period=cfg.peak_train_log_period,
        log_every_seconds=cfg.train_log_every_seconds,
        heartbeat_seconds=cfg.train_heartbeat_seconds,
        stage_label="PEAK_TRAIN_EXPORTS",
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

    sidx, st0 = stage_start("train_delta_model")
    train_weight_full = np.zeros(len(feat_df), dtype=float)
    train_weight_full[peak_train_idx] = train_weights
    delta_train_weights = train_weight_full[delta_train_idx]
    if cfg.delta_use_cutoff_weights and len(delta_train_idx) > 0:
        cutoff_vals = pd.to_numeric(
            feat_df.iloc[delta_train_idx]["cutoff_minutes"],
            errors="coerce",
        ).to_numpy(dtype=float)
        norm = (cutoff_vals - 240.0) / (1080.0 - 240.0)
        norm = np.clip(norm, 0.0, 1.0)
        cutoff_weight = 1.0 + float(cfg.delta_cutoff_weight_alpha) * norm
        delta_train_weights = delta_train_weights * cutoff_weight

    delta_result = train_delta_model(
        X_train=x_all[delta_train_idx],
        y_train=y_delta_train.astype(int),
        X_val=x_all[delta_val_idx],
        y_val=y_delta_val.astype(int),
        num_classes=delta_class_max,
        sample_weight_train=delta_train_weights,
        objective=cfg.delta_objective,
        use_class_weights=cfg.delta_use_class_weights,
        logger=active_logger,
        log_period=cfg.delta_train_log_period,
        log_every_seconds=cfg.train_log_every_seconds,
        heartbeat_seconds=cfg.train_heartbeat_seconds,
        stage_label="DELTA_TRAIN_EXPORTS",
    )
    stage_end(
        sidx,
        "train_delta_model",
        st0,
        details=f"val_multi_logloss_temp={delta_result.val_metrics.get('multi_logloss_temp')}",
    )

    sidx, st0 = stage_start("predict_probabilities")
    p_peak_val_raw, p_peak_val_cal = predict_peak_probability(
        model=peak_result.model,
        isotonic=peak_result.isotonic,
        X=x_all[peak_val_idx],
    )
    p_peak_test_raw, p_peak_test_cal = predict_peak_probability(
        model=peak_result.model,
        isotonic=peak_result.isotonic,
        X=x_all[peak_test_idx],
    )

    p_peak_raw_all = np.zeros(len(feat_df), dtype=float)
    p_peak_cal_all = np.zeros(len(feat_df), dtype=float)
    p_peak_raw_all[peak_val_idx] = p_peak_val_raw
    p_peak_raw_all[peak_test_idx] = p_peak_test_raw
    p_peak_cal_all[peak_val_idx] = p_peak_val_cal
    p_peak_cal_all[peak_test_idx] = p_peak_test_cal

    _, p_delta_raw_all, p_delta_temp_all = predict_delta_conditional(
        model=delta_result.model,
        temperature=delta_result.temperature,
        X=x_all,
    )
    model_classes = np.asarray(delta_result.model.classes_, dtype=int)
    p_delta_raw_all = _expand_class_probs(
        probs=p_delta_raw_all,
        class_labels=model_classes,
        total_classes=delta_class_max,
    )
    p_delta_temp_all = _expand_class_probs(
        probs=p_delta_temp_all,
        class_labels=model_classes,
        total_classes=delta_class_max,
    )
    stage_end(sidx, "predict_probabilities", st0)

    sidx, st0 = stage_start("evaluate_metrics")
    if len(delta_test_idx) > 0:
        delta_test_raw = float(_multi_logloss(y_delta_test, p_delta_raw_all[delta_test_idx]))
        delta_test_temp = float(_multi_logloss(y_delta_test, p_delta_temp_all[delta_test_idx]))
    else:
        delta_test_raw = float("nan")
        delta_test_temp = float("nan")

    combined_val, combined_val_detail = _evaluate_distribution_rows(
        df=feat_df,
        row_indices=peak_val_idx,
        p_peak=p_peak_cal_all[peak_val_idx],
        p_delta_cond=p_delta_temp_all[peak_val_idx],
    )
    combined_test, combined_test_detail = _evaluate_distribution_rows(
        df=feat_df,
        row_indices=peak_test_idx,
        p_peak=p_peak_cal_all[peak_test_idx],
        p_delta_cond=p_delta_temp_all[peak_test_idx],
    )

    cutoff_val = _cutoff_metrics(df_detail=combined_val_detail)
    cutoff_test = _cutoff_metrics(df_detail=combined_test_detail)
    cutoff_val.to_csv(reports_dir / "cutoff_metrics_val.csv", index=False)
    cutoff_test.to_csv(reports_dir / "cutoff_metrics_test.csv", index=False)

    calib_val = _temperature_bucket_calibration(
        df=feat_df,
        row_indices=peak_val_idx,
        p_peak=p_peak_cal_all[peak_val_idx],
        p_delta_cond=p_delta_temp_all[peak_val_idx],
    )
    calib_test = _temperature_bucket_calibration(
        df=feat_df,
        row_indices=peak_test_idx,
        p_peak=p_peak_cal_all[peak_test_idx],
        p_delta_cond=p_delta_temp_all[peak_test_idx],
    )
    calib_val.to_csv(reports_dir / "bucket_calibration_val.csv", index=False)
    calib_test.to_csv(reports_dir / "bucket_calibration_test.csv", index=False)

    cloud_parts: list[pd.DataFrame] = []
    uv_parts: list[pd.DataFrame] = []
    precip_parts: list[pd.DataFrame] = []
    for split_name, idx_arr, y_arr in [
        ("val", delta_val_idx, y_delta_val),
        ("test", delta_test_idx, y_delta_test),
    ]:
        if len(idx_arr) == 0:
            continue
        probs_arr = p_delta_temp_all[idx_arr]
        cloud_vals = (
            feat_df.iloc[idx_arr]["clds_norm_now"]
            if "clds_norm_now" in feat_df.columns
            else pd.Series(["UNK"] * len(idx_arr))
        )
        cloud_parts.append(
            _delta_logloss_by_group(
                y_true=y_arr,
                probs=probs_arr,
                group_values=cloud_vals,
                split_name=split_name,
                group_name="clds_norm_now",
            )
        )
        uv_vals = (
            _uv_quantile_labels(feat_df.iloc[idx_arr]["uv_index_now"], q=5)
            if "uv_index_now" in feat_df.columns
            else pd.Series(["MISSING"] * len(idx_arr))
        )
        uv_parts.append(
            _delta_logloss_by_group(
                y_true=y_arr,
                probs=probs_arr,
                group_values=uv_vals,
                split_name=split_name,
                group_name="uv_quantile",
            )
        )
        for precip_feature in ["any_precip_sofar", "precip_any_180", "wx_precip_any_180"]:
            if precip_feature not in feat_df.columns:
                continue
            precip_parts.append(
                _delta_logloss_by_group(
                    y_true=y_arr,
                    probs=probs_arr,
                    group_values=feat_df.iloc[idx_arr][precip_feature].astype(str),
                    split_name=split_name,
                    group_name=precip_feature,
                )
            )

    cloud_report = (
        pd.concat(cloud_parts, ignore_index=True)
        if cloud_parts
        else pd.DataFrame(columns=["split", "group_name", "group_value", "n_rows", "multi_logloss_temp"])
    )
    uv_report = (
        pd.concat(uv_parts, ignore_index=True)
        if uv_parts
        else pd.DataFrame(columns=["split", "group_name", "group_value", "n_rows", "multi_logloss_temp"])
    )
    precip_report = (
        pd.concat(precip_parts, ignore_index=True)
        if precip_parts
        else pd.DataFrame(columns=["split", "group_name", "group_value", "n_rows", "multi_logloss_temp"])
    )
    cloud_report.to_csv(reports_dir / "delta_logloss_by_cloud_regime.csv", index=False)
    uv_report.to_csv(reports_dir / "delta_logloss_by_uv_regime.csv", index=False)
    precip_report.to_csv(reports_dir / "delta_logloss_by_precip_regime.csv", index=False)

    peak_metrics = {
        "val": {
            "logloss_raw": float(log_loss(y_peak_all[peak_val_idx], np.clip(p_peak_val_raw, 1e-6, 1 - 1e-6))),
            "logloss_cal": float(log_loss(y_peak_all[peak_val_idx], np.clip(p_peak_val_cal, 1e-6, 1 - 1e-6))),
            "brier_raw": float(brier_score_loss(y_peak_all[peak_val_idx], p_peak_val_raw)),
            "brier_cal": float(brier_score_loss(y_peak_all[peak_val_idx], p_peak_val_cal)),
        },
        "test": {
            "logloss_raw": float(log_loss(y_peak_all[peak_test_idx], np.clip(p_peak_test_raw, 1e-6, 1 - 1e-6))),
            "logloss_cal": float(log_loss(y_peak_all[peak_test_idx], np.clip(p_peak_test_cal, 1e-6, 1 - 1e-6))),
            "brier_raw": float(brier_score_loss(y_peak_all[peak_test_idx], p_peak_test_raw)),
            "brier_cal": float(brier_score_loss(y_peak_all[peak_test_idx], p_peak_test_cal)),
        },
    }
    delta_metrics = {
        "val": {
            "multi_logloss_raw": float(_multi_logloss(y_delta_val, p_delta_raw_all[delta_val_idx])),
            "multi_logloss_temp": float(_multi_logloss(y_delta_val, p_delta_temp_all[delta_val_idx])),
        },
        "test": {
            "multi_logloss_raw": delta_test_raw,
            "multi_logloss_temp": delta_test_temp,
        },
        "temperature": float(delta_result.temperature),
    }
    stage_end(
        sidx,
        "evaluate_metrics",
        st0,
        details=f"combined_val_nll={combined_val.get('nll')} combined_test_nll={combined_test.get('nll')}",
    )

    sidx, st0 = stage_start("write_models_and_metadata")
    pred_val = _build_full_delta_arrays(
        full_len=len(feat_df),
        class_count=delta_class_max,
        row_indices=peak_val_idx,
        p_peak=p_peak_cal_all[peak_val_idx],
        p_delta=p_delta_temp_all[peak_val_idx],
    )
    pred_test = _build_full_delta_arrays(
        full_len=len(feat_df),
        class_count=delta_class_max,
        row_indices=peak_test_idx,
        p_peak=p_peak_cal_all[peak_test_idx],
        p_delta=p_delta_temp_all[peak_test_idx],
    )
    _write_df_with_csv_parquet(pred_val, predictions_dir / "predictions_val", active_logger)
    _write_df_with_csv_parquet(pred_test, predictions_dir / "predictions_test", active_logger)
    _write_df_with_csv_parquet(combined_val_detail, predictions_dir / "distribution_eval_val", active_logger)
    _write_df_with_csv_parquet(combined_test_detail, predictions_dir / "distribution_eval_test", active_logger)

    peak_result.model.booster_.save_model(str(models_dir / "peak_model.txt"))
    delta_result.model.booster_.save_model(str(models_dir / "delta_model.txt"))
    joblib.dump(peak_result.isotonic, models_dir / "peak_isotonic.pkl")
    (models_dir / "delta_temperature_T.json").write_text(
        json.dumps({"temperature": float(delta_result.temperature)}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    (run_dir / "feature_list.json").write_text(json.dumps(feat_cols, indent=2), encoding="utf-8")
    (run_dir / "imputer_values.json").write_text(json.dumps(medians, indent=2, sort_keys=True), encoding="utf-8")
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

    metrics = {
        "run_id": run_dir.name,
        "backend": "lgbm",
        "data_dir": str(cfg.data_dir),
        "feature_contract_version": p_cfg.feature_contract_version,
        "rows_total": int(len(feat_df)),
        "split_rows": {
            "train": int(len(peak_train_idx)),
            "val": int(len(peak_val_idx)),
            "test": int(len(peak_test_idx)),
        },
        "split_rows_delta": {
            "train": int(len(delta_train_idx)),
            "val": int(len(delta_val_idx)),
            "test": int(len(delta_test_idx)),
        },
        "audit": audit,
        "climo_lookup_meta": climo_meta,
        "peak": peak_metrics,
        "delta": delta_metrics,
        "combined_lgbm": {"val": combined_val, "test": combined_test},
        "combined_blended": {"val": combined_val, "test": combined_test},
        "analog": {"enabled": False, "reason": "exports_lgbm_backend_no_analog"},
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    cfg_payload = {
        "backend": "lgbm",
        "data_dir": str(cfg.data_dir),
        "output_root": str(cfg.output_root),
        "include_feels_like": bool(cfg.include_feels_like),
        "delta_objective": str(cfg.delta_objective),
        "delta_use_class_weights": bool(cfg.delta_use_class_weights),
        "delta_use_cutoff_weights": bool(cfg.delta_use_cutoff_weights),
        "delta_cutoff_weight_alpha": float(cfg.delta_cutoff_weight_alpha),
        "feature_contract_version": p_cfg.feature_contract_version,
        "split": {
            "train_start": str(cfg.split.train_start),
            "train_end": str(cfg.split.train_end),
            "val_start": str(cfg.split.val_start),
            "val_end": str(cfg.split.val_end),
            "test_start": str(cfg.split.test_start),
            "test_end": str(cfg.split.test_end),
        },
    }
    (run_dir / "config.json").write_text(json.dumps(cfg_payload, indent=2, sort_keys=True), encoding="utf-8")

    metrics_md = [
        "# KLGA LGBM Peak/Delta Run From Exports",
        "",
        f"- run_id: {run_dir.name}",
        f"- data_dir: {cfg.data_dir}",
        f"- rows_total: {len(feat_df)}",
        f"- split_rows_peak: train={len(peak_train_idx)} val={len(peak_val_idx)} test={len(peak_test_idx)}",
        f"- split_rows_delta: train={len(delta_train_idx)} val={len(delta_val_idx)} test={len(delta_test_idx)}",
        "",
        "## Peak",
        f"- val_logloss_cal: {peak_metrics['val']['logloss_cal']}",
        f"- test_logloss_cal: {peak_metrics['test']['logloss_cal']}",
        f"- val_brier_cal: {peak_metrics['val']['brier_cal']}",
        f"- test_brier_cal: {peak_metrics['test']['brier_cal']}",
        "",
        "## Delta",
        f"- val_multi_logloss_temp: {delta_metrics['val']['multi_logloss_temp']}",
        f"- test_multi_logloss_temp: {delta_metrics['test']['multi_logloss_temp']}",
        f"- temperature: {delta_metrics['temperature']}",
        "",
        "## Combined",
        f"- val_nll: {combined_val.get('nll')}",
        f"- test_nll: {combined_test.get('nll')}",
        f"- val_top1_accuracy: {combined_val.get('top1_accuracy')}",
        f"- test_top1_accuracy: {combined_test.get('top1_accuracy')}",
    ]
    (run_dir / "metrics.md").write_text("\n".join(metrics_md), encoding="utf-8")
    stage_end(sidx, "write_models_and_metadata", st0, details=f"metrics_path={metrics_path}")

    active_logger.info(
        "LGBM_EXPORT_RUN_DONE elapsed=%s run_dir=%s",
        format_duration(time.perf_counter() - pipeline_start),
        run_dir,
    )
    return LGBMExportsTrainingResult(run_dir=run_dir, metrics_path=metrics_path, metrics=metrics)
