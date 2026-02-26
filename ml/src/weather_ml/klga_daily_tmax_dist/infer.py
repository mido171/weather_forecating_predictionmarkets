from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BucketInterval:
    label: str
    lower: int | None
    upper: int | None
    kind: str


def build_delta_pmf(
    *,
    p_peak: float,
    p_delta_cond: np.ndarray,
) -> np.ndarray:
    p_peak = float(np.clip(p_peak, 0.0, 1.0))
    cond = np.asarray(p_delta_cond, dtype=float)
    if cond.ndim != 1:
        raise ValueError("p_delta_cond must be a 1D array.")
    if np.sum(cond) <= 0:
        cond = np.full_like(cond, 1.0 / len(cond))
    else:
        cond = cond / np.sum(cond)

    out = np.zeros(len(cond) + 1, dtype=float)
    out[0] = p_peak
    out[1:] = (1.0 - p_peak) * cond
    s = float(np.sum(out))
    if s > 0:
        out = out / s
    return out


def delta_pmf_to_tmax_pmf(
    *,
    tmax_sofar: float,
    delta_pmf: np.ndarray,
) -> dict[int, float]:
    base = int(np.round(float(tmax_sofar)))
    out: dict[int, float] = {}
    for k, p in enumerate(delta_pmf):
        temp = base + int(k)
        out[temp] = float(out.get(temp, 0.0) + p)
    s = sum(out.values())
    if s > 0:
        out = {k: float(v / s) for k, v in out.items()}
    return out


def parse_bucket_label(label: str) -> BucketInterval:
    src = label.strip()
    norm = src.lower().replace("\u00b0", "").replace("fahrenheit", "").replace("f", "")
    norm = " ".join(norm.split())

    if "or below" in norm:
        nums = re.findall(r"-?\d+", norm)
        if not nums:
            raise ValueError(f"Unable to parse bucket label: {label}")
        k = int(nums[0])
        return BucketInterval(label=label, lower=None, upper=k, kind="le")

    if "or higher" in norm or "or above" in norm:
        nums = re.findall(r"-?\d+", norm)
        if not nums:
            raise ValueError(f"Unable to parse bucket label: {label}")
        k = int(nums[0])
        return BucketInterval(label=label, lower=k, upper=None, kind="ge")

    m_dash = re.search(r"(-?\d+)\s*[-\u2013]\s*(-?\d+)", norm)
    if m_dash:
        lo = int(m_dash.group(1))
        hi = int(m_dash.group(2))
        if hi < lo:
            lo, hi = hi, lo
        return BucketInterval(label=label, lower=lo, upper=hi, kind="range")

    m_to = re.search(r"(-?\d+)\s*to\s*(-?\d+)", norm)
    if m_to:
        lo = int(m_to.group(1))
        hi = int(m_to.group(2))
        if hi < lo:
            lo, hi = hi, lo
        return BucketInterval(label=label, lower=lo, upper=hi, kind="range")

    m_exact = re.fullmatch(r"-?\d+", norm.strip())
    if m_exact:
        v = int(m_exact.group(0))
        return BucketInterval(label=label, lower=v, upper=v, kind="range")

    raise ValueError(f"Unable to parse bucket label: {label}")


def bucket_probability(
    *,
    tmax_pmf: dict[int, float],
    bucket: BucketInterval,
) -> float:
    p = 0.0
    for temp, prob in tmax_pmf.items():
        if bucket.kind == "range":
            if bucket.lower is None or bucket.upper is None:
                continue
            if bucket.lower <= temp <= bucket.upper:
                p += float(prob)
        elif bucket.kind == "le":
            if bucket.upper is not None and temp <= bucket.upper:
                p += float(prob)
        elif bucket.kind == "ge":
            if bucket.lower is not None and temp >= bucket.lower:
                p += float(prob)
    return float(p)


def compute_bucket_probabilities(
    *,
    tmax_pmf: dict[int, float],
    bucket_labels: Iterable[str],
) -> pd.DataFrame:
    rows = []
    for label in bucket_labels:
        interval = parse_bucket_label(label)
        rows.append(
            {
                "bucket_label": label,
                "p_bucket": bucket_probability(tmax_pmf=tmax_pmf, bucket=interval),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("p_bucket", ascending=False).reset_index(drop=True)
    return out


def write_tmax_pmf_json(path: Path, tmax_pmf: dict[int, float]) -> None:
    payload = {str(k): float(v) for k, v in sorted(tmax_pmf.items())}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
