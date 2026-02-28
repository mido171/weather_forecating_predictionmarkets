from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import logging

import joblib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureScalerState:
    enabled: bool
    scaler_name: str
    center: np.ndarray
    scale: np.ndarray
    train_rows_used: int
    n_features: int
    fitted_on_train_only: bool = True
    robust_quantile_range: tuple[float, float] | None = None

    def as_meta(self) -> dict[str, Any]:
        out = {
            "enabled": bool(self.enabled),
            "scaler_name": str(self.scaler_name),
            "train_rows_used": int(self.train_rows_used),
            "n_features": int(self.n_features),
            "fitted_on_train_only": bool(self.fitted_on_train_only),
        }
        if self.robust_quantile_range is not None:
            out["robust_quantile_range"] = [float(self.robust_quantile_range[0]), float(self.robust_quantile_range[1])]
        return out


def fit_imputer_medians_columnwise(
    *,
    df: pd.DataFrame,
    feature_cols: list[str],
    train_mask: np.ndarray,
) -> dict[str, float]:
    train_idx = np.where(np.asarray(train_mask, dtype=bool))[0]
    if train_idx.size == 0:
        raise ValueError("Cannot fit imputer medians with zero train rows.")
    med: dict[str, float] = {}
    for c in feature_cols:
        vals = pd.to_numeric(df[c].iloc[train_idx], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            med[c] = 0.0
            continue
        m = float(np.nanmedian(finite))
        med[c] = m if np.isfinite(m) else 0.0
    return med


def build_imputed_memmap(
    *,
    df: pd.DataFrame,
    feature_cols: list[str],
    medians: dict[str, float],
    out_path: Path,
    chunk_rows: int = 2048,
    dtype: np.dtype = np.float32,
    logger: logging.Logger | None = None,
) -> np.memmap:
    active_logger = logger or logging.getLogger(__name__)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = int(len(df))
    n_features = int(len(feature_cols))
    if n_rows <= 0 or n_features <= 0:
        raise ValueError(f"Invalid matrix dimensions n_rows={n_rows} n_features={n_features}")
    mvec = np.asarray([float(medians[c]) for c in feature_cols], dtype=np.float64)
    mm = np.memmap(out_path, mode="w+", dtype=dtype, shape=(n_rows, n_features))

    step = max(int(chunk_rows), 1)
    for i0 in range(0, n_rows, step):
        i1 = min(i0 + step, n_rows)
        block = (
            df.loc[i0:i1 - 1, feature_cols]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float64, copy=False)
        )
        bad = ~np.isfinite(block)
        if np.any(bad):
            block[bad] = mvec[np.where(bad)[1]]
        mm[i0:i1, :] = block.astype(dtype, copy=False)
    mm.flush()
    active_logger.info(
        "OOM_MEMMAP_BUILT path=%s rows=%d features=%d dtype=%s",
        out_path,
        n_rows,
        n_features,
        np.dtype(dtype).name,
    )
    return mm


def _fit_standard_scaler_from_memmap(
    *,
    x_all: np.memmap,
    train_idx: np.ndarray,
    chunk_rows: int = 4096,
) -> FeatureScalerState:
    n_features = int(x_all.shape[1])
    train_idx = np.asarray(train_idx, dtype=np.int64)
    if train_idx.size == 0:
        raise ValueError("Cannot fit standard scaler with zero train rows.")

    total = np.zeros(n_features, dtype=np.float64)
    total_sq = np.zeros(n_features, dtype=np.float64)
    n = 0
    step = max(int(chunk_rows), 1)
    for i0 in range(0, train_idx.size, step):
        idx = train_idx[i0 : i0 + step]
        block = np.asarray(x_all[idx, :], dtype=np.float64)
        total += np.sum(block, axis=0)
        total_sq += np.sum(np.square(block), axis=0)
        n += int(block.shape[0])
    mean = total / max(n, 1)
    var = (total_sq / max(n, 1)) - np.square(mean)
    var = np.where(np.isfinite(var), np.maximum(var, 1e-12), 1e-12)
    scale = np.sqrt(var, dtype=np.float64)
    scale = np.where(scale > 0.0, scale, 1.0)
    return FeatureScalerState(
        enabled=True,
        scaler_name="standard",
        center=mean.astype(np.float64, copy=False),
        scale=scale.astype(np.float64, copy=False),
        train_rows_used=int(train_idx.size),
        n_features=n_features,
    )


def _fit_robust_scaler_from_memmap(
    *,
    x_all: np.memmap,
    train_idx: np.ndarray,
    logger: logging.Logger | None = None,
) -> FeatureScalerState:
    active_logger = logger or logging.getLogger(__name__)
    n_features = int(x_all.shape[1])
    train_idx = np.asarray(train_idx, dtype=np.int64)
    if train_idx.size == 0:
        raise ValueError("Cannot fit robust scaler with zero train rows.")

    center = np.zeros(n_features, dtype=np.float64)
    scale = np.ones(n_features, dtype=np.float64)
    for j in range(n_features):
        vals = np.asarray(x_all[train_idx, j], dtype=np.float64)
        q25, q50, q75 = np.quantile(vals, [0.25, 0.5, 0.75])
        iqr = float(q75 - q25)
        center[j] = float(q50)
        scale[j] = iqr if np.isfinite(iqr) and iqr > 1e-12 else 1.0
        if (j + 1) % 500 == 0 or (j + 1) == n_features:
            active_logger.info(
                "ROBUST_SCALER_PROGRESS feature=%d/%d",
                j + 1,
                n_features,
            )
    return FeatureScalerState(
        enabled=True,
        scaler_name="robust",
        center=center,
        scale=scale,
        train_rows_used=int(train_idx.size),
        n_features=n_features,
        robust_quantile_range=(25.0, 75.0),
    )


def fit_scaler_state_from_memmap(
    *,
    x_all: np.memmap,
    train_idx: np.ndarray,
    scaler_name: str,
    chunk_rows: int = 4096,
    logger: logging.Logger | None = None,
) -> FeatureScalerState:
    key = str(scaler_name).strip().lower()
    n_features = int(x_all.shape[1])
    train_idx = np.asarray(train_idx, dtype=np.int64)
    if key in {"none", "off", "disabled"}:
        return FeatureScalerState(
            enabled=False,
            scaler_name="none",
            center=np.zeros(n_features, dtype=np.float64),
            scale=np.ones(n_features, dtype=np.float64),
            train_rows_used=int(train_idx.size),
            n_features=n_features,
        )
    if key == "standard":
        return _fit_standard_scaler_from_memmap(x_all=x_all, train_idx=train_idx, chunk_rows=chunk_rows)
    if key == "robust":
        return _fit_robust_scaler_from_memmap(x_all=x_all, train_idx=train_idx, logger=logger)
    raise ValueError(f"Unsupported feature scaler '{scaler_name}'. Expected one of: none, standard, robust.")


def apply_scaler_inplace_memmap(
    *,
    x_all: np.memmap,
    scaler_state: FeatureScalerState,
    chunk_rows: int = 2048,
) -> None:
    if not scaler_state.enabled:
        return
    center = scaler_state.center.astype(np.float32, copy=False)
    scale = scaler_state.scale.astype(np.float32, copy=False)
    n_rows = int(x_all.shape[0])
    step = max(int(chunk_rows), 1)
    for i0 in range(0, n_rows, step):
        i1 = min(i0 + step, n_rows)
        block = np.asarray(x_all[i0:i1, :], dtype=np.float32)
        block = (block - center[None, :]) / scale[None, :]
        x_all[i0:i1, :] = block
    x_all.flush()


def write_scaler_artifacts(
    *,
    scaler_state: FeatureScalerState,
    model_joblib_path: Path | None,
    meta_json_path: Path,
) -> None:
    meta_json_path.write_text(
        json.dumps(scaler_state.as_meta(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if model_joblib_path is not None:
        joblib.dump(
            {
                "scaler_name": scaler_state.scaler_name,
                "enabled": scaler_state.enabled,
                "center": scaler_state.center,
                "scale": scaler_state.scale,
                "fitted_on_train_only": scaler_state.fitted_on_train_only,
                "train_rows_used": scaler_state.train_rows_used,
                "n_features": scaler_state.n_features,
                "robust_quantile_range": scaler_state.robust_quantile_range,
            },
            model_joblib_path,
        )
