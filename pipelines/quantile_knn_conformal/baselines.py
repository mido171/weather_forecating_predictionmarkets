from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ClimoBaseline:
    by_doy_median: dict[int, float]
    global_median: float
    resid_quantiles: dict[float, float]


@dataclass
class PersistenceBaseline:
    rem_by_doy_cutoff: dict[tuple[int, int], float]
    rem_by_doy: dict[int, float]
    rem_global: float
    resid_quantiles: dict[float, float]


def _resid_quantiles(y: np.ndarray, yhat: np.ndarray, quantiles: list[float]) -> dict[float, float]:
    resid = y - yhat
    return {float(q): float(np.nanquantile(resid, q)) for q in quantiles}


def fit_climatology_baseline(train_df: pd.DataFrame, quantiles: list[float]) -> ClimoBaseline:
    med_by_doy = train_df.groupby("doy")["y_tmax"].median().to_dict()
    global_med = float(train_df["y_tmax"].median())
    yhat = train_df["doy"].map(med_by_doy).fillna(global_med).to_numpy(dtype=float)
    rq = _resid_quantiles(train_df["y_tmax"].to_numpy(dtype=float), yhat, quantiles)
    return ClimoBaseline(by_doy_median={int(k): float(v) for k, v in med_by_doy.items()}, global_median=global_med, resid_quantiles=rq)


def predict_climatology_quantiles(model: ClimoBaseline, df: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    base = df["doy"].map(model.by_doy_median).fillna(model.global_median).astype(float)
    out = pd.DataFrame(index=df.index)
    for q in quantiles:
        out[f"q_{q:.3f}"] = base + model.resid_quantiles.get(float(q), 0.0)
    return out


def fit_persistence_baseline(train_df: pd.DataFrame, quantiles: list[float]) -> PersistenceBaseline:
    floor = np.maximum(train_df["tmax_sofar"].to_numpy(dtype=float), train_df["temp"].to_numpy(dtype=float))
    rem = train_df["y_tmax"].to_numpy(dtype=float) - floor
    tmp = train_df[["doy", "cutoff_minutes"]].copy()
    tmp["rem"] = rem

    rem_by_dc = tmp.groupby(["doy", "cutoff_minutes"]) ["rem"].median().to_dict()
    rem_by_doy = tmp.groupby("doy")["rem"].median().to_dict()
    rem_global = float(np.nanmedian(rem))

    est = []
    for d, c, f in zip(train_df["doy"], train_df["cutoff_minutes"], floor, strict=False):
        key = (int(d), int(c))
        rr = rem_by_dc.get(key)
        if rr is None:
            rr = rem_by_doy.get(int(d), rem_global)
        est.append(float(f + rr))
    yhat = np.array(est, dtype=float)
    rq = _resid_quantiles(train_df["y_tmax"].to_numpy(dtype=float), yhat, quantiles)

    return PersistenceBaseline(
        rem_by_doy_cutoff={(int(k[0]), int(k[1])): float(v) for k, v in rem_by_dc.items()},
        rem_by_doy={int(k): float(v) for k, v in rem_by_doy.items()},
        rem_global=rem_global,
        resid_quantiles=rq,
    )


def predict_persistence_quantiles(model: PersistenceBaseline, df: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    floor = np.maximum(df["tmax_sofar"].to_numpy(dtype=float), df["temp"].to_numpy(dtype=float))
    base = np.empty(len(df), dtype=float)
    for i, (d, c, f) in enumerate(zip(df["doy"], df["cutoff_minutes"], floor, strict=False)):
        rr = model.rem_by_doy_cutoff.get((int(d), int(c)))
        if rr is None:
            rr = model.rem_by_doy.get(int(d), model.rem_global)
        base[i] = float(f + rr)

    out = pd.DataFrame(index=df.index)
    for q in quantiles:
        out[f"q_{q:.3f}"] = base + model.resid_quantiles.get(float(q), 0.0)
    return out
