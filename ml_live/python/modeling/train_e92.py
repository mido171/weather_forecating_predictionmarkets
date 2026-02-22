from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from ml_live.modeling import artifacts as artifact_utils
from ml_live.runtime.paths import models_dir


logger = logging.getLogger("ml_live.train")


@dataclass(frozen=True)
class TrainingConfig:
    station_id: str
    train_start: date
    train_end: date
    target_col: str = "actual_tmax_f"
    sigma_cap: float = 10.0
    sigma_floor: float = 0.5
    oof_folds: int = 5
    min_train_rows: int = 200


def load_dataset(path: Path) -> pd.DataFrame:
    logger.info("Loading dataset: %s", path)
    df = pd.read_parquet(path)
    if "target_date_local" not in df.columns:
        raise ValueError("Dataset missing target_date_local column")
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    return df


def filter_training_range(df: pd.DataFrame, cfg: TrainingConfig) -> pd.DataFrame:
    mask = (df["target_date_local"] >= cfg.train_start) & (df["target_date_local"] <= cfg.train_end)
    filtered = df.loc[mask].copy()
    if filtered.empty:
        raise ValueError("No training data after applying date range")
    return filtered


def _compute_ensmean_row(row: pd.Series) -> float:
    components = [
        "nbm_tmax_f",
        "hrrr_tmax_f",
        "rap_tmax_f",
        "gefsatmosmean_tmax_f",
        "gfs_n_x_max",
        "nam_n_x_max",
    ]
    values = [row.get(col) for col in components]
    values = [float(v) for v in values if v is not None and not pd.isna(v)]
    if not values:
        return float("nan")
    return float(np.mean(values))


def _bias_sources(feature_cols: list[str]) -> list[str]:
    sources: set[str] = set()
    for name in feature_cols:
        if name.startswith("bias_") and name.endswith("_rm60_l2"):
            sources.add(name[len("bias_") : -len("_rm60_l2")])
        elif name.endswith("_corr_rm60_l2"):
            sources.add(name[: -len("_corr_rm60_l2")])
    return sorted(sources)


def _ensure_derived_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    window_days: int = 60,
    lag_days: int = 2,
) -> pd.DataFrame:
    missing = [col for col in feature_cols if col not in df.columns]
    if not missing:
        return df
    df = df.copy()
    if "target_date_local" not in df.columns:
        raise ValueError("Dataset missing target_date_local column")
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date

    need_ensmean = any(
        name == "ensmean"
        or name.startswith("bias_ensmean")
        or name.startswith("ensmean_corr")
        for name in feature_cols
    )
    if need_ensmean and "ensmean" not in df.columns:
        df["ensmean"] = df.apply(_compute_ensmean_row, axis=1)

    sources = _bias_sources(feature_cols)
    if not sources:
        return df

    if target_col not in df.columns:
        raise ValueError(f"Dataset missing target column: {target_col}")

    df = df.sort_values("target_date_local").reset_index(drop=True)
    dates = df["target_date_local"].tolist()

    for source in sources:
        if source not in df.columns:
            raise ValueError(f"Dataset missing base feature for bias calculation: {source}")
        bias_col = f"bias_{source}_rm60_l2"
        corr_col = f"{source}_corr_rm60_l2"
        if bias_col in df.columns and corr_col in df.columns:
            continue
        bias_values: list[float] = []
        for target_date in dates:
            history_end = target_date - timedelta(days=lag_days)
            history_start = history_end - timedelta(days=window_days - 1)
            mask = (df["target_date_local"] >= history_start) & (df["target_date_local"] <= history_end)
            history = df.loc[mask]
            residuals = history[target_col] - history[source]
            residuals = residuals.dropna()
            if len(residuals) < window_days:
                bias_values.append(float("nan"))
                continue
            bias_values.append(float(residuals.mean()))
        if bias_col not in df.columns:
            df[bias_col] = bias_values
        if corr_col not in df.columns:
            df[corr_col] = df[source] + df[bias_col]

    return df


def select_features(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> pd.DataFrame:
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in dataset: {missing}")
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")
    return df


def drop_missing_rows(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> pd.DataFrame:
    before = len(df)
    filtered = df.dropna(subset=feature_cols + [target_col]).copy()
    dropped = before - len(filtered)
    if dropped:
        logger.info("Dropped %s rows with missing features/target", dropped)
    return filtered


def train_mu_model(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> LGBMRegressor:
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=64,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    model.fit(X, y)
    return model


def time_blocked_oof_predictions(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    folds: int,
    min_train_rows: int,
) -> pd.Series:
    dates = np.array(sorted(df["target_date_local"].unique()))
    if len(dates) < folds + 1:
        raise ValueError("Not enough unique dates for time-blocked CV")
    fold_sizes = np.full(folds, len(dates) // folds, dtype=int)
    fold_sizes[: len(dates) % folds] += 1
    indices = np.cumsum(fold_sizes)
    oof_preds = pd.Series(index=df.index, dtype=float)
    start = 0
    for fold_idx, end in enumerate(indices, start=1):
        val_dates = dates[start:end]
        train_dates = dates[:start]
        start = end
        if len(train_dates) == 0:
            continue
        train_df = df[df["target_date_local"].isin(train_dates)]
        val_df = df[df["target_date_local"].isin(val_dates)]
        if len(train_df) < min_train_rows:
            logger.warning("Skipping fold %s: not enough training rows (%s)", fold_idx, len(train_df))
            continue
        model = train_mu_model(train_df, feature_cols, target_col)
        preds = model.predict(val_df[feature_cols].to_numpy(dtype=float))
        oof_preds.loc[val_df.index] = preds
    return oof_preds


def train_sigma_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    oof_mu: pd.Series,
    sigma_cap: float,
    sigma_floor: float,
) -> tuple[LGBMRegressor, pd.Series]:
    residuals = df["actual_tmax_f"].to_numpy(dtype=float) - oof_mu.to_numpy(dtype=float)
    sigma_target = np.clip(np.abs(residuals), sigma_floor, sigma_cap)
    model = LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=64,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=52,
    )
    X = df[feature_cols].to_numpy(dtype=float)
    model.fit(X, sigma_target)
    return model, pd.Series(sigma_target, index=df.index)


def train_e92_models(
    cfg: TrainingConfig,
    dataset_path: Path | None = None,
    feature_list_path: Path | None = None,
) -> dict:
    e92_dir = artifact_utils.find_e92_run_dir()
    feature_path = feature_list_path or artifact_utils.resolve_feature_list(e92_dir)
    features = artifact_utils.load_feature_list(feature_path)
    dataset = dataset_path or artifact_utils.resolve_dataset_path(e92_dir)

    df = load_dataset(dataset)
    df = filter_training_range(df, cfg)
    df = _ensure_derived_features(df, features, cfg.target_col)
    df = select_features(df, features, cfg.target_col)
    df = drop_missing_rows(df, features, cfg.target_col)

    logger.info("Training rows=%s features=%s", len(df), len(features))
    mu_model = train_mu_model(df, features, cfg.target_col)

    oof_mu = time_blocked_oof_predictions(
        df, features, cfg.target_col, cfg.oof_folds, cfg.min_train_rows
    )
    oof_mask = oof_mu.notna()
    oof_df = df.loc[oof_mask].copy()
    oof_mu = oof_mu.loc[oof_mask]
    if len(oof_df) < cfg.min_train_rows:
        raise ValueError("Insufficient OOF predictions to train sigma model")

    sigma_model, sigma_target = train_sigma_model(
        oof_df, features, oof_mu, cfg.sigma_cap, cfg.sigma_floor
    )

    station_dir = models_dir(cfg.station_id)
    station_dir.mkdir(parents=True, exist_ok=True)
    mu_path = station_dir / "e92_mu_model.joblib"
    sigma_path = station_dir / "e92_sigma_model.joblib"
    feature_cols_path = station_dir / "e92_feature_columns.json"
    metadata_path = station_dir / "e92_training_metadata.json"
    sigma_meta_path = station_dir / "e92_sigma_metadata.json"

    joblib.dump(mu_model, mu_path)
    joblib.dump(sigma_model, sigma_path)
    feature_cols_path.write_text(json.dumps(features, indent=2), encoding="utf-8")

    metadata = {
        "station_id": cfg.station_id,
        "train_start": cfg.train_start.isoformat(),
        "train_end": cfg.train_end.isoformat(),
        "dataset_path": str(dataset),
        "feature_list_path": str(feature_path),
        "feature_hash": artifact_utils.feature_list_hash(features),
        "rows": int(len(df)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    sigma_meta = {
        "station_id": cfg.station_id,
        "sigma_target": "abs(residual)",
        "sigma_cap": cfg.sigma_cap,
        "sigma_floor": cfg.sigma_floor,
        "oof_rows": int(len(oof_df)),
        "oof_folds": cfg.oof_folds,
    }
    sigma_meta_path.write_text(json.dumps(sigma_meta, indent=2), encoding="utf-8")

    return {
        "mu_model_path": mu_path,
        "sigma_model_path": sigma_path,
        "feature_columns_path": feature_cols_path,
        "metadata_path": metadata_path,
        "sigma_metadata_path": sigma_meta_path,
    }
