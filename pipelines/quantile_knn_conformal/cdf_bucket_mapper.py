from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


@dataclass
class Bucket:
    name: str
    min_temp: int | None
    max_temp: int | None

    def contains(self, t: int) -> bool:
        lo_ok = True if self.min_temp is None else t >= self.min_temp
        hi_ok = True if self.max_temp is None else t <= self.max_temp
        return bool(lo_ok and hi_ok)


def default_buckets() -> list[Bucket]:
    return [
        Bucket("<=30F", None, 30),
        Bucket("31-40F", 31, 40),
        Bucket("41-50F", 41, 50),
        Bucket("51-60F", 51, 60),
        Bucket("61-70F", 61, 70),
        Bucket("71-80F", 71, 80),
        Bucket(">=81F", 81, None),
    ]


def load_buckets(path: str | None) -> list[Bucket]:
    if not path:
        return default_buckets()
    p = Path(path)
    if not p.exists():
        return default_buckets()
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    items = raw.get("buckets") if isinstance(raw, dict) else None
    if not isinstance(items, list) or not items:
        return default_buckets()
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        out.append(Bucket(name=str(it.get("name")), min_temp=it.get("min_temp"), max_temp=it.get("max_temp")))
    return out or default_buckets()


def _cdf_from_quantiles(qvals: np.ndarray, qlevels: np.ndarray):
    qs = np.array(qlevels, dtype=float)
    ys = np.array(qvals, dtype=float)
    sorter = np.argsort(ys)
    ys = ys[sorter]
    qs = qs[sorter]
    ys = np.concatenate(([0.0], ys, [120.0]))
    qs = np.concatenate(([0.0], qs, [1.0]))

    def cdf(x: np.ndarray) -> np.ndarray:
        return np.interp(x, ys, qs, left=0.0, right=1.0)

    return cdf


def quantile_rows_to_integer_pmf(
    pred_q: pd.DataFrame,
    quantiles: list[float],
    support_min: int = 0,
    support_max: int = 120,
) -> pd.DataFrame:
    qcols = [f"q_{q:.3f}" for q in quantiles]
    levels = np.array(quantiles, dtype=float)
    temps = np.arange(support_min, support_max + 1, dtype=int)

    rows = []
    for _, r in pred_q[qcols].iterrows():
        qvals = r.to_numpy(dtype=float)
        cdf = _cdf_from_quantiles(qvals, levels)
        pmf = cdf(temps + 0.5) - cdf(temps - 0.5)
        pmf = np.clip(pmf, 0.0, None)
        s = np.sum(pmf)
        if s <= 0:
            pmf = np.full_like(pmf, 1.0 / len(pmf), dtype=float)
        else:
            pmf = pmf / s
        row = {f"p_int_{t}": float(p) for t, p in zip(temps, pmf, strict=False)}
        row["top1_temp"] = int(temps[int(np.argmax(pmf))])
        row["top1_prob"] = float(np.max(pmf))
        top_idx = np.argsort(-pmf)[:5]
        row["top5_temps"] = ",".join(str(int(temps[i])) for i in top_idx)
        row["top5_probs"] = ",".join(f"{float(pmf[i]):.6f}" for i in top_idx)
        row["cdf_q10"] = float(np.interp(0.10, np.cumsum(pmf), temps))
        row["cdf_q50"] = float(np.interp(0.50, np.cumsum(pmf), temps))
        row["cdf_q90"] = float(np.interp(0.90, np.cumsum(pmf), temps))
        row["cdf_q95"] = float(np.interp(0.95, np.cumsum(pmf), temps))
        rows.append(row)

    return pd.DataFrame(rows, index=pred_q.index)


def integer_pmf_to_bucket_probs(pmf_df: pd.DataFrame, buckets: list[Bucket], support_min: int = 0, support_max: int = 120) -> pd.DataFrame:
    temps = np.arange(support_min, support_max + 1, dtype=int)
    pcols = [f"p_int_{t}" for t in temps]

    rows = []
    for _, r in pmf_df[pcols].iterrows():
        vals = r.to_numpy(dtype=float)
        row = {}
        for b in buckets:
            mask = np.array([b.contains(int(t)) for t in temps], dtype=bool)
            row[f"bucket_yes::{b.name}"] = float(np.sum(vals[mask]))
            row[f"bucket_no::{b.name}"] = float(1.0 - row[f"bucket_yes::{b.name}"])
        rows.append(row)
    return pd.DataFrame(rows, index=pmf_df.index)


def realized_bucket_outcomes(y_tmax: pd.Series, buckets: list[Bucket]) -> pd.DataFrame:
    rows = []
    for y in y_tmax.to_numpy(dtype=float):
        yi = int(round(y))
        row = {}
        for b in buckets:
            yes = 1.0 if b.contains(yi) else 0.0
            row[f"bucket_yes::{b.name}"] = yes
            row[f"bucket_no::{b.name}"] = 1.0 - yes
        rows.append(row)
    return pd.DataFrame(rows, index=y_tmax.index)
