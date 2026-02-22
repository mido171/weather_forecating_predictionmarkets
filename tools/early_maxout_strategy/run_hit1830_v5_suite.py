import argparse
import json
import math
import hashlib
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance

from scipy.stats import t as student_t

import lightgbm as lgb
from sqlalchemy import create_engine, text

VERSION = "hit1830_v5_suite_v1"
BATCH_NAME = "B5"

_EXEC_SUMMARY = {
    "EXP01_Fusion_SBFD": (
        "Fusion_SBFD uses the proven minute-core curve features with convective/suppression MOS blocks and adds a "
        "sea-breeze front detector that locates a robust change-point between 09:00 local and cutoff. It estimates "
        "pre/post slopes, plateau persistence, and cooling pulses to flag stabilization after sea-breeze onset. The "
        "goal is to reduce false YES calls on days where the curve looks peaked early but a delayed onshore regime "
        "permits later heating. It blends heat-gap signals, SBFD interactions, and calibrated probabilities to keep "
        "trade rate while improving precision in the 0.55–0.65 price band."
    ),
    "EXP02_Fusion_REVSHOCK": (
        "Fusion_REVSHOCK keeps the minute core and MOS convective block but adds revision-shock features computed from "
        "the last two to three MOS rows as-of cutoff. It uses per-variable deltas, absolute deltas, volatility, and "
        "cross-model disagreements to detect late guidance flips that imply higher afternoon heating or weaker "
        "suppression. Revision flags for tmp, q12, and cig provide directional cues. The intent is to override misleading "
        "curve impressions when MOS updates sharply, improving YES precision without collapsing trade frequency."
    ),
    "EXP03_Fusion_BC_GAP": (
        "Fusion_BC_GAP corrects systematic MOS bias in the expected max temperature headroom. It computes a leak-safe "
        "EWMA bias from past days, adjusts MOS max, and recalculates heat-gap and normalized gap features. Interaction "
        "terms with convective suppression (cig, q12) capture regimes where bias matters most. The experiment targets "
        "false YES caused by MOS being biased cool or warm in specific seasons, aiming for better calibration and more "
        "reliable EV at realistic entry prices."
    ),
    "EXP04_Fusion_REG_BETA_CAL": (
        "Fusion_REG_BETA_CAL keeps the Fusion base feature set unchanged but replaces global calibration with a regime-"
        "conditioned beta calibrator. Regimes are defined by wind-direction bins and suppression bins (cloud/precip), "
        "with shrinkage to the global calibrator for stability. The goal is to reduce calibration errors concentrated "
        "in specific sea-breeze or suppression regimes, improving EV and reducing false positives without requiring a "
        "higher decision threshold."
    ),
    "EXP05_Fusion_EV_WEIGHTED": (
        "Fusion_EV_WEIGHTED trains the same feature set as the baseline but applies EV-aligned sample weighting to "
        "penalize false YES more heavily, reflecting trading costs around c≈0.55. A small grid of negative-class weights "
        "is tuned on validation net-units. The model is then Platt-calibrated for smooth probabilities. This experiment "
        "targets higher precision at fixed trade rates and improves net units per 100 days by concentrating probability "
        "mass on genuinely high-confidence YES cases."
    ),
    "EXP06_Fusion_ANALOG_PRIOR": (
        "Fusion_ANALOG_PRIOR augments the fusion base with a leak-safe kNN analog prior. It embeds the morning curve "
        "using normalized DCT coefficients and combines them with key slope, plateau, and MOS suppression features. "
        "Neighbors are restricted to strictly earlier days (month-band ±1), producing analog YES-rate and analog delta "
        "features. The goal is to leverage repeated diurnal archetypes without leakage, improving ranking and "
        "calibration especially in ambiguous mid-day regimes."
    ),
    "EXP07_HAZARD_EXCEED": (
        "HAZARD_EXCEED reframes the target as the probability of a future exceedance after cutoff, then converts to "
        "p(YES) via 1−p(exceed). It uses curve momentum features (slopes, persistence, warming fraction) plus MOS "
        "suppression and wind regime proxies. Beta calibration is applied to exceedance probabilities before the "
        "complement. This reframing aims to better match the physics question—whether later heating will exceed the "
        "current max—yielding smoother probabilities around the trading threshold."
    ),
    "EXP08_ZI_DELTA": (
        "ZI_DELTA models the remaining heating as a zero-inflated process: a classifier predicts whether Δ≤0.01 (YES), "
        "while a regressor models log(Δ+0.01) on positive-Δ days. The classifier’s output is calibrated and reported as "
        "p(YES). This two-head structure separates true plateaus from small but meaningful remaining headroom, which can "
        "support abstention or tighter EV gating. It aims to reduce false YES in marginal cases while keeping recall on "
        "clear convective-plateau days."
    ),
    "EXP09_CNN2GBM": (
        "CNN2GBM is implemented as a LightGBM fusion that uses a DCT-based curve embedding as a proxy for a causal CNN "
        "encoder. The embedding captures higher-order shape information from the pre-cutoff diurnal evolution and is "
        "combined with MOS convective and wind features plus the minute-core stats. The experiment targets subtle shape "
        "patterns (micro-plateaus, multi-step warming) that standard scalar features miss, while keeping the training "
        "fast and leak-safe."
    ),
    "EXP10_CPC_EMBED": (
        "CPC_EMBED is a LightGBM-only proxy for contrastive predictive coding. It uses DCT curve features and slope "
        "histogram descriptors to summarize temporal dynamics that a CPC model would learn, then fuses them with MOS "
        "suppression and regime features. The goal is to improve robustness in rare boundary-layer regimes by using a "
        "richer representation of morning curve texture, without the cost or overfitting risk of deep sequence models."
    ),
    "EXP11_MOE_4REG": (
        "MOE_4REG builds a four-expert mixture-of-experts split by wind regime (onshore/offshore) and suppression "
        "regime (suppressed/clear). A multiclass LGBM gate predicts regime weights using MOS wind, cloud, and key minute "
        "features; regime-specific experts then score p(YES). This structure targets systematic errors where the same "
        "curve shape implies different outcomes under different regimes, aiming to reduce false positives in offshore "
        "clear cases while preserving recall in suppressed onshore days."
    ),
    "EXP12_MONO_PHYS": (
        "MONO_PHYS enforces physically consistent monotonic constraints: p(YES) should increase with minutes since max, "
        "drop-from-max, and plateau persistence, and decrease with positive slopes and large heat gap. The constrained "
        "LGBM reduces overfitting and prevents pathological high-confidence YES predictions when headroom is large. It "
        "keeps the feature set compact and interpretable and is designed to improve generalization and calibration "
        "stability, especially on out-of-sample regimes."
    ),
    "EXP13_SLOPE_HIST": (
        "SLOPE_HIST encodes the last 6 hours and full pre-cutoff period as histograms of 5‑minute temperature change "
        "rates, plus crash/recovery descriptors. These features distinguish smooth radiational warming from choppy, "
        "convectively influenced patterns. The model fuses slope texture with MOS suppression and dewpoint features to "
        "better separate early-peak days from late-heating days. It targets improvements in precision by downgrading "
        "steady warming curves that are unlikely to have already peaked."
    ),
    "EXP14_DROP_REBOUND": (
        "DROP_REBOUND explicitly models convective outflow signatures using detected 10‑minute crashes and 30‑minute "
        "recoveries. Aggregate features include crash magnitude, recovery fraction, crash energy, and time since last "
        "event, combined with MOS precip/suppression cues. This experiment aims to capture true early-peak days driven "
        "by outflow boundaries, improving recall in stormy regimes without exploding false positives from minor noise."
    ),
    "EXP15_QA_AWARE": (
        "QA_AWARE adds data-quality features that quantify missingness, gaps, and spike artifacts in the minute series. "
        "It lets the model down-weight unreliable curve signals and lean on MOS suppression/headroom when coverage is "
        "poor. Features include gap counts, max gap length, recent coverage fractions, and spike flags tied to max "
        "occurrences. The goal is to reduce false confidence and improve EV stability across years with changing sensor "
        "coverage or transient data issues."
    ),
    "EXP16_CTP_BAYES": (
        "CTP_BAYES builds a conditional prior p(YES) based on season, wind regime, and suppression regime using an "
        "expanding window of past days. The prior is converted to logit space and fused with minute core and MOS features "
        "in an LGBM residual model. This encourages predictions that respect climatological timing tendencies while still "
        "allowing day-specific deviations. It targets lower false positives in regimes where early Tmax is historically "
        "rare, without sacrificing recall on legitimate early-peak events."
    ),
    "EXP17_MOS_NOWCAST_MIS_v2": (
        "MOS_NOWCAST_MIS_v2 compares observed temperatures at 12Z/15Z with MOS t12/t12_2 blocks to estimate whether the "
        "day is running warmer or cooler than guidance. A leak-safe EWMA bias correction is applied to the mismatch. "
        "This creates a nowcast-style feature that directly answers “is the atmosphere ahead or behind forecast?” and "
        "thus whether more heating is likely. It is fused with the minute core and MOS suppression features to reduce "
        "false YES when the day is lagging MOS expectations."
    ),
    "EXP18_EV_STACK_OOF": (
        "EV_STACK_OOF combines diverse base models with strict time-based out-of-fold stacking to avoid leakage. Base "
        "models include fusion, monotone, hazard, and analog variants. OOF predictions are generated on contiguous year "
        "folds, then a regularized LGBM meta-model learns a calibrated blend. This is intended to reduce correlated "
        "errors while preserving realistic generalization, and to improve net units per 100 days without the collapse "
        "seen in naive stacking."
    ),
    "EXP19_MT_DELTA_BINS": (
        "MT_DELTA_BINS reframes the target into four Δ bins (≤0.01, 0.5, 1.5, >1.5) with multiclass LightGBM. The "
        "probability of the zero bin is used as p(YES), while tail probabilities indicate large-exceedance risk. This "
        "structure helps the model learn gradations of “NO” rather than a single class, improving calibration near the "
        "decision boundary and enabling risk-aware filtering when large late-day warming is likely."
    ),
    "EXP20_HB_PARAM_T": (
        "HB_PARAM_T models the distribution of log(Δ+0.01) using two LightGBM regressors for mean and scale, assuming a "
        "robust Student‑t form. It then computes p(YES)=P(Δ≤0.01) from the parametric CDF and applies beta calibration. "
        "This yields smoother probabilities and an uncertainty proxy from the predicted scale, which can be used for "
        "EV-based gating. The goal is improved calibration and steadier profitability across regimes with uncertain "
        "afternoon heating."
    ),
}

LOCAL_TZ = "America/New_York"
STOCKHOLM_TZ = "Europe/Stockholm"

START_DATE = date(2002, 1, 1)
END_DATE = date(2026, 12, 31)

EPS = 0.01
KNN_K = 50

MOS_VARIABLES = [
    "n_x",
    "tmp",
    "dpt",
    "wdr",
    "wsp",
    "p06",
    "p12",
    "q06",
    "q12",
    "t06",
    "t06_1",
    "t06_2",
    "t12",
    "t12_1",
    "t12_2",
    "cig",
    "vis",
    "pos",
    "poz",
]

MOS_MODELS = ["GFS", "NAM"]
MOS_VARIABLES_SQL = ", ".join(f"'{v}'" for v in MOS_VARIABLES)


@dataclass
class DayWindow:
    target_date_local: date
    day_start_utc: datetime
    day_end_utc: datetime
    cutoff_utc: datetime


def _sha256_str(text_value: str) -> str:
    return hashlib.sha256(text_value.encode("utf-8")).hexdigest()


def _date_range(start: date, end: date) -> List[date]:
    days = (end - start).days
    return [start + timedelta(days=i) for i in range(days + 1)]


def _build_day_windows() -> pd.DataFrame:
    local = ZoneInfo(LOCAL_TZ)
    stockholm = ZoneInfo(STOCKHOLM_TZ)
    rows = []
    for day in _date_range(START_DATE, END_DATE):
        day_start_local = datetime(day.year, day.month, day.day, 0, 0, tzinfo=local)
        day_end_local = day_start_local + timedelta(days=1)
        day_start_utc = day_start_local.astimezone(timezone.utc)
        day_end_utc = day_end_local.astimezone(timezone.utc)

        cutoff_local = datetime(day.year, day.month, day.day, 18, 30, tzinfo=stockholm)
        cutoff_utc = cutoff_local.astimezone(timezone.utc)

        if cutoff_utc < day_start_utc:
            cutoff_utc = day_start_utc
        if cutoff_utc >= day_end_utc:
            cutoff_utc = day_end_utc - timedelta(seconds=1)

        rows.append(
            {
                "target_date_local": day,
                "day_start_utc": day_start_utc,
                "day_end_utc": day_end_utc,
                "cutoff_utc": cutoff_utc,
            }
        )
    return pd.DataFrame(rows)


def _expected_points(start_utc: datetime, end_utc: datetime, freq_minutes: int, inclusive_end: bool) -> int:
    seconds = (end_utc - start_utc).total_seconds()
    if inclusive_end:
        return int(seconds // (freq_minutes * 60)) + 1
    return int(seconds // (freq_minutes * 60))


def _ols_slope(minutes: np.ndarray, values: np.ndarray) -> float:
    mask = np.isfinite(values)
    if mask.sum() < 3:
        return float("nan")
    x = minutes[mask] / 60.0
    y = values[mask].astype(float)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return float("nan")
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def _count_sign_changes(values: np.ndarray) -> float:
    diffs = np.diff(values)
    if diffs.size == 0:
        return float("nan")
    signs = np.sign(diffs)
    signs = signs[signs != 0]
    if signs.size <= 1:
        return 0.0
    return float(np.sum(signs[1:] * signs[:-1] < 0))


def _longest_run(mask: np.ndarray) -> int:
    best = 0
    run = 0
    for val in mask:
        if val:
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return best


def _dct_matrix(n: int, kmax: int) -> np.ndarray:
    n_idx = np.arange(n)[:, None]
    k_idx = np.arange(kmax)[None, :]
    return np.cos(np.pi / n * (n_idx + 0.5) * k_idx)

def _resample_fixed(values: np.ndarray, target_len: int) -> np.ndarray:
    if target_len <= 0:
        return np.array([], dtype=float)
    if values.size == 0:
        return np.full(target_len, np.nan)
    series = pd.Series(values).interpolate(limit_direction="both")
    series = series.fillna(series.median())
    x = np.linspace(0.0, 1.0, num=len(series), endpoint=True)
    xi = np.linspace(0.0, 1.0, num=target_len, endpoint=True)
    return np.interp(xi, x, series.to_numpy())


def _mad(values: np.ndarray) -> float:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return float("nan")
    med = np.median(vals)
    return float(np.median(np.abs(vals - med)))


def _gap_stats(is_present: np.ndarray, step_minutes: int = 5) -> Tuple[float, int, int]:
    if is_present.size == 0:
        return float("nan"), 0, 0
    max_gap = 0
    run = 0
    gap_cnt_15 = 0
    gap_cnt_30 = 0
    for ok in is_present:
        if not ok:
            run += 1
        else:
            if run * step_minutes >= 15:
                gap_cnt_15 += 1
            if run * step_minutes >= 30:
                gap_cnt_30 += 1
            max_gap = max(max_gap, run)
            run = 0
    if run > 0:
        if run * step_minutes >= 15:
            gap_cnt_15 += 1
        if run * step_minutes >= 30:
            gap_cnt_30 += 1
        max_gap = max(max_gap, run)
    return float(max_gap * step_minutes), int(gap_cnt_15), int(gap_cnt_30)


def _spike_stats(values: np.ndarray, step_minutes: int = 5) -> Tuple[int, int]:
    if values.size < 3:
        return 0, 0
    spikes = 0
    spike_indices = set()
    for i in range(1, len(values) - 1):
        if not (np.isfinite(values[i - 1]) and np.isfinite(values[i]) and np.isfinite(values[i + 1])):
            continue
        d1 = values[i] - values[i - 1]
        d2 = values[i + 1] - values[i]
        if d1 >= 2.0 and d2 <= -2.0:
            spikes += 1
            spike_indices.add(i)
        if d1 <= -2.0 and d2 >= 2.0:
            spikes += 1
            spike_indices.add(i)
    return spikes, len(spike_indices)


def _huber_loss(residuals: np.ndarray, delta: float = 1.0) -> float:
    abs_res = np.abs(residuals)
    quad = np.minimum(abs_res, delta)
    lin = abs_res - quad
    return float(np.sum(0.5 * quad * quad + delta * lin))


def _piecewise_two_segment(times_min: np.ndarray, values: np.ndarray, min_seg: int = 6) -> Dict[str, float]:
    n = len(values)
    if n < 2 * min_seg:
        return {
            "break_idx": None,
            "slope_pre": float("nan"),
            "slope_post": float("nan"),
            "sse": float("inf"),
        }
    best = {"sse": float("inf"), "break_idx": None, "slope_pre": float("nan"), "slope_post": float("nan")}
    for i in range(min_seg, n - min_seg + 1):
        left = values[:i]
        right = values[i:]
        t_left = times_min[:i]
        t_right = times_min[i:] - times_min[i]
        slope_left = _ols_slope(t_left, left)
        slope_right = _ols_slope(t_right, right)
        if not (np.isfinite(slope_left) and np.isfinite(slope_right)):
            continue
        intercept_left = np.nanmean(left) - slope_left * np.nanmean(t_left / 60.0)
        intercept_right = np.nanmean(right) - slope_right * np.nanmean(t_right / 60.0)
        pred_left = slope_left * (t_left / 60.0) + intercept_left
        pred_right = slope_right * (t_right / 60.0) + intercept_right
        residuals = np.concatenate([left - pred_left, right - pred_right])
        sse = _huber_loss(residuals, delta=1.0)
        if sse < best["sse"]:
            best.update({"sse": sse, "break_idx": i, "slope_pre": slope_left, "slope_post": slope_right})
    return best


def _segment_fit(prefix: Dict[str, np.ndarray], start: int, end: int) -> Tuple[float, float, float]:
    n = end - start
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    sx = prefix["sx"][end] - prefix["sx"][start]
    sy = prefix["sy"][end] - prefix["sy"][start]
    sxx = prefix["sxx"][end] - prefix["sxx"][start]
    sxy = prefix["sxy"][end] - prefix["sxy"][start]
    syy = prefix["syy"][end] - prefix["syy"][start]
    denom = n * sxx - sx * sx
    if denom == 0:
        return float("nan"), float("nan"), float("nan")
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    # SSE
    sse = (
        syy
        + slope * slope * sxx
        + n * intercept * intercept
        + 2 * slope * intercept * sx
        - 2 * slope * sxy
        - 2 * intercept * sy
    )
    return float(slope), float(intercept), float(sse)


def _two_break_piecewise(values: np.ndarray, step_minutes: int = 5, min_seg: int = 4) -> Dict[str, float]:
    n = len(values)
    if n < (min_seg * 3):
        return {
            "slope_1": float("nan"),
            "slope_2": float("nan"),
            "slope_3": float("nan"),
            "break1_min_before_cutoff": float("nan"),
            "break2_min_before_cutoff": float("nan"),
            "end_drop": float("nan"),
        }
    series = pd.Series(values).interpolate(limit_direction="both").to_numpy()
    if not np.isfinite(series).any():
        series = np.zeros_like(series, dtype=float)
    x = np.arange(n, dtype=float) * (step_minutes / 60.0)
    prefix = {
        "sx": np.concatenate([[0.0], np.cumsum(x)]),
        "sy": np.concatenate([[0.0], np.cumsum(series)]),
        "sxx": np.concatenate([[0.0], np.cumsum(x * x)]),
        "sxy": np.concatenate([[0.0], np.cumsum(x * series)]),
        "syy": np.concatenate([[0.0], np.cumsum(series * series)]),
    }

    best = {
        "sse": float("inf"),
        "i": None,
        "j": None,
        "slope1": float("nan"),
        "int1": float("nan"),
        "slope2": float("nan"),
        "int2": float("nan"),
        "slope3": float("nan"),
        "int3": float("nan"),
    }

    for i in range(min_seg, n - 2 * min_seg + 1):
        for j in range(i + min_seg, n - min_seg + 1):
            slope1, int1, sse1 = _segment_fit(prefix, 0, i)
            slope2, int2, sse2 = _segment_fit(prefix, i, j)
            slope3, int3, sse3 = _segment_fit(prefix, j, n)
            if not np.isfinite(sse1 + sse2 + sse3):
                continue
            sse = sse1 + sse2 + sse3
            if sse < best["sse"]:
                best.update({
                    "sse": sse,
                    "i": i,
                    "j": j,
                    "slope1": slope1,
                    "int1": int1,
                    "slope2": slope2,
                    "int2": int2,
                    "slope3": slope3,
                    "int3": int3,
                })

    if best["i"] is None or best["j"] is None:
        return {
            "slope_1": float("nan"),
            "slope_2": float("nan"),
            "slope_3": float("nan"),
            "break1_min_before_cutoff": float("nan"),
            "break2_min_before_cutoff": float("nan"),
            "end_drop": float("nan"),
        }

    i = best["i"]
    j = best["j"]
    break1_min_before = (n - 1 - i) * step_minutes
    break2_min_before = (n - 1 - j) * step_minutes
    y_start = best["slope3"] * x[j] + best["int3"]
    y_end = best["slope3"] * x[-1] + best["int3"]
    end_drop = y_start - y_end

    return {
        "slope_1": best["slope1"],
        "slope_2": best["slope2"],
        "slope_3": best["slope3"],
        "break1_min_before_cutoff": float(break1_min_before),
        "break2_min_before_cutoff": float(break2_min_before),
        "end_drop": float(end_drop),
    }


def _compute_metrics(y_true: np.ndarray, p_pred: np.ndarray, threshold: float) -> Dict[str, float]:
    y_hat = (p_pred >= threshold).astype(int)
    acc = accuracy_score(y_true, y_hat)
    bal = balanced_accuracy_score(y_true, y_hat)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_hat, average="binary", pos_label=1)
    cm = confusion_matrix(y_true, y_hat, labels=[0, 1])
    try:
        auc = roc_auc_score(y_true, p_pred)
    except ValueError:
        auc = float("nan")
    try:
        brier = brier_score_loss(y_true, p_pred)
    except ValueError:
        brier = float("nan")
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal),
        "yes_precision": float(prec),
        "yes_recall": float(rec),
        "yes_f1": float(f1),
        "roc_auc": float(auc),
        "brier": float(brier),
        "cm_tn": int(cm[0, 0]),
        "cm_fp": int(cm[0, 1]),
        "cm_fn": int(cm[1, 0]),
        "cm_tp": int(cm[1, 1]),
    }


def _select_thresholds(y_true: np.ndarray, p_pred: np.ndarray) -> Tuple[float, float, float]:
    best_t = 0.5
    best_acc = -1.0
    best_t_bal = 0.5
    best_bal = -1.0
    best_t_profit = 0.5
    best_profit = -1e9
    for t in np.linspace(0.05, 0.95, 91):
        y_hat = p_pred >= t
        acc = accuracy_score(y_true, y_hat)
        bal = balanced_accuracy_score(y_true, y_hat)
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
        if bal > best_bal:
            best_bal = bal
            best_t_bal = float(t)
        cm = confusion_matrix(y_true, y_hat, labels=[0, 1])
        tp = int(cm[1, 1])
        fp = int(cm[0, 1])
        profit = tp - fp
        if profit > best_profit:
            best_profit = profit
            best_t_profit = float(t)
    return best_t, best_t_bal, best_t_profit


def _net_units_per_100(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_hat, labels=[0, 1])
    tp = int(cm[1, 1])
    fp = int(cm[0, 1])
    n = len(y_true)
    if n == 0:
        return float("nan")
    return (tp - fp) / n * 100.0


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    # vectorized normal CDF using erf
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / np.sqrt(2.0)))

def _compute_cache_hash(minute_dir: Path, mos_version: str) -> str:
    meta = {
        "version": VERSION,
        "minute_files": [],
        "mos_version": mos_version,
        "mos_variables": MOS_VARIABLES,
        "mos_models": MOS_MODELS,
        "start_date": str(START_DATE),
        "end_date": str(END_DATE),
    }
    for path in sorted(minute_dir.glob("MIA_tmpf_1min_UTC_*.csv")):
        stat = path.stat()
        meta["minute_files"].append(
            {"name": path.name, "size": stat.st_size, "mtime": int(stat.st_mtime)}
        )
    return _sha256_str(json.dumps(meta, sort_keys=True))


def _read_cache(path: Path, meta_path: Path, expected_hash: str, reuse_cache: bool) -> pd.DataFrame | None:
    if not reuse_cache:
        return None
    if not path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if meta.get("hash") != expected_hash:
        return None
    return pd.read_parquet(path)


def _write_cache(path: Path, meta_path: Path, df: pd.DataFrame, hash_value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    meta = {
        "hash": hash_value,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "rows": int(len(df)),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _load_minute_series(minute_dir: Path) -> pd.DataFrame:
    files = sorted(minute_dir.glob("MIA_tmpf_1min_UTC_*.csv"))
    if not files:
        raise FileNotFoundError(f"No minute files found in {minute_dir}")
    frames = []
    for path in files:
        df = pd.read_csv(path, usecols=["valid(UTC)", "tmpf"], dtype={"tmpf": "string"})
        df["ts_utc"] = pd.to_datetime(df["valid(UTC)"], utc=True, errors="coerce")
        df["tmpf"] = pd.to_numeric(df["tmpf"], errors="coerce")
        df = df.dropna(subset=["ts_utc", "tmpf"])
        frames.append(df[["ts_utc", "tmpf"]])
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values("ts_utc")
    all_df = all_df.drop_duplicates(subset=["ts_utc"], keep="last")
    return all_df


def _build_minute_features(minute_dir: Path, cache_dir: Path, reuse_cache: bool, rebuild_cache: bool, mos_version: str) -> pd.DataFrame:
    cache_path = cache_dir / "hit1830_v5_minute_features.parquet"
    meta_path = cache_dir / "hit1830_v5_minute_features.meta.json"
    cache_hash = _compute_cache_hash(minute_dir, mos_version)
    if not rebuild_cache:
        cached = _read_cache(cache_path, meta_path, cache_hash, reuse_cache)
        if cached is not None:
            return cached

    day_windows = _build_day_windows()
    df_1m = _load_minute_series(minute_dir)
    df_1m = df_1m.sort_values("ts_utc")
    df_1m = df_1m.set_index("ts_utc")

    series_5m = df_1m["tmpf"].resample("5min").median()
    dct_len = 160
    dct_mat = _dct_matrix(dct_len, 21)

    rows = []
    minute_violations = 0
    for _, dw in day_windows.iterrows():
        day = dw["target_date_local"]
        day_start = dw["day_start_utc"]
        day_end = dw["day_end_utc"]
        cutoff = dw["cutoff_utc"]

        full_idx = pd.date_range(day_start, day_end - timedelta(minutes=5), freq="5min")
        full_series = series_5m.reindex(full_idx)
        if full_series.notna().sum() == 0:
            continue

        tmax_full = float(np.nanmax(full_series.to_numpy()))
        tmin_full = float(np.nanmin(full_series.to_numpy()))
        range_full = tmax_full - tmin_full if np.isfinite(tmax_full) and np.isfinite(tmin_full) else float("nan")

        partial_end = min(cutoff, day_end) - timedelta(minutes=5)
        partial_idx = pd.date_range(day_start, partial_end, freq="5min") if partial_end >= day_start else []
        if len(partial_idx) == 0:
            continue
        partial_series = series_5m.reindex(partial_idx)
        if partial_series.notna().sum() == 0:
            continue

        partial_vals = partial_series.to_numpy()
        tmax_sofar = float(np.nanmax(partial_vals))
        tmin_sofar = float(np.nanmin(partial_vals))
        range_sofar = tmax_sofar - tmin_sofar if np.isfinite(tmax_sofar) and np.isfinite(tmin_sofar) else float("nan")

        max_time_full_utc = full_series[full_series == tmax_full].index.min()
        tmax_time_local_minute = float("nan")
        if pd.notna(max_time_full_utc):
            max_time_local = max_time_full_utc.tz_convert(LOCAL_TZ)
            tmax_time_local_minute = max_time_local.hour * 60 + max_time_local.minute

        max_time_partial_utc = partial_series[partial_series == tmax_sofar].index.min()
        minutes_since_max = float("nan")
        tmax_sofar_time_local_minute = float("nan")
        if pd.notna(max_time_partial_utc):
            minutes_since_max = (cutoff - max_time_partial_utc).total_seconds() / 60.0
            max_time_partial_local = max_time_partial_utc.tz_convert(LOCAL_TZ)
            tmax_sofar_time_local_minute = max_time_partial_local.hour * 60 + max_time_partial_local.minute

        temp_now = float(np.nanmedian(partial_series.to_numpy()[-2:])) if partial_series.size >= 2 else float("nan")
        drop_from_max = tmax_sofar - temp_now if np.isfinite(tmax_sofar) and np.isfinite(temp_now) else float("nan")

        w12_end = cutoff - timedelta(minutes=5)
        w12_start = cutoff - timedelta(hours=12)
        w12_idx = pd.date_range(w12_start, w12_end, freq="5min")
        if len(w12_idx) and w12_idx.max() > cutoff:
            minute_violations += 1
        w12 = series_5m.reindex(w12_idx).to_numpy()

        w6_end = cutoff - timedelta(minutes=5)
        w6_start = cutoff - timedelta(hours=6)
        w6_idx = pd.date_range(w6_start, w6_end, freq="5min")
        if len(w6_idx) and w6_idx.max() > cutoff:
            minute_violations += 1
        w6 = series_5m.reindex(w6_idx).to_numpy()

        w4_end = cutoff - timedelta(minutes=5)
        w4_start = cutoff - timedelta(hours=4)
        w4_idx = pd.date_range(w4_start, w4_end, freq="5min")
        if len(w4_idx) and w4_idx.max() > cutoff:
            minute_violations += 1
        w4 = series_5m.reindex(w4_idx).to_numpy()

        w3_end = cutoff - timedelta(minutes=5)
        w3_start = cutoff - timedelta(hours=3)
        w3_idx = pd.date_range(w3_start, w3_end, freq="5min")
        if len(w3_idx) and w3_idx.max() > cutoff:
            minute_violations += 1
        w3 = series_5m.reindex(w3_idx).to_numpy()

        # Slopes
        def slope_last(points: int, values: np.ndarray) -> float:
            if len(values) < points:
                return float("nan")
            y = values[-points:]
            mins = np.arange(points) * 5
            return _ols_slope(mins, y)

        slope_15 = slope_last(3, w12)
        slope_30 = slope_last(6, w12)
        slope_60 = slope_last(12, w12)
        slope_120 = slope_last(24, w12)
        slope_180 = slope_last(36, w12)
        accel = slope_30 - slope_120 if np.isfinite(slope_30) and np.isfinite(slope_120) else float("nan")
        curvature_30_180 = slope_30 - slope_180 if np.isfinite(slope_30) and np.isfinite(slope_180) else float("nan")
        curvature_60_180 = slope_60 - slope_180 if np.isfinite(slope_60) and np.isfinite(slope_180) else float("nan")

        slope_sign_changes_last180 = _count_sign_changes(w3)

        # Change-point detection on W3 (single break)
        w3_interp = pd.Series(w3).interpolate(limit_direction="both").to_numpy()
        best_sse = float("inf")
        best_split = None
        best_before = float("nan")
        best_after = float("nan")
        best_drop = float("nan")
        if len(w3_interp) >= 10:
            for s in range(8, len(w3_interp) - 2):
                left = w3_interp[: s + 1]
                right = w3_interp[s:]
                mins_left = np.arange(len(left)) * 5
                mins_right = np.arange(len(right)) * 5
                slope_left = _ols_slope(mins_left, left)
                slope_right = _ols_slope(mins_right, right)
                if not np.isfinite(slope_left) or not np.isfinite(slope_right):
                    continue
                intercept_left = np.nanmean(left) - slope_left * np.nanmean(mins_left / 60.0)
                intercept_right = np.nanmean(right) - slope_right * np.nanmean(mins_right / 60.0)
                pred_left = slope_left * (mins_left / 60.0) + intercept_left
                pred_right = slope_right * (mins_right / 60.0) + intercept_right
                sse = np.nansum((left - pred_left) ** 2) + np.nansum((right - pred_right) ** 2)
                if sse < best_sse:
                    best_sse = sse
                    best_split = s
                    best_before = slope_left
                    best_after = slope_right
                    best_drop = pred_left[-1] - pred_right[0]

        cp_minute = float("nan")
        if best_split is not None:
            cp_minute = (len(w3_interp) - 1 - best_split) * 5

        # Change-point on W4 (single break)
        w4_interp = pd.Series(w4).interpolate(limit_direction="both").to_numpy()
        best4_sse = float("inf")
        best4_split = None
        best4_before = float("nan")
        best4_after = float("nan")
        best4_drop = float("nan")
        if len(w4_interp) >= 10:
            for s in range(8, len(w4_interp) - 2):
                left = w4_interp[: s + 1]
                right = w4_interp[s:]
                mins_left = np.arange(len(left)) * 5
                mins_right = np.arange(len(right)) * 5
                slope_left = _ols_slope(mins_left, left)
                slope_right = _ols_slope(mins_right, right)
                if not np.isfinite(slope_left) or not np.isfinite(slope_right):
                    continue
                intercept_left = np.nanmean(left) - slope_left * np.nanmean(mins_left / 60.0)
                intercept_right = np.nanmean(right) - slope_right * np.nanmean(mins_right / 60.0)
                pred_left = slope_left * (mins_left / 60.0) + intercept_left
                pred_right = slope_right * (mins_right / 60.0) + intercept_right
                sse = np.nansum((left - pred_left) ** 2) + np.nansum((right - pred_right) ** 2)
                if sse < best4_sse:
                    best4_sse = sse
                    best4_split = s
                    best4_before = slope_left
                    best4_after = slope_right
                    best4_drop = pred_left[-1] - pred_right[0]

        cp4_minute = float("nan")
        if best4_split is not None:
            cp4_minute = (len(w4_interp) - 1 - best4_split) * 5

        # Volatility / outflow
        std_30 = float(np.nanstd(w12[-6:])) if len(w12) >= 6 else float("nan")
        std_60 = float(np.nanstd(w12[-12:])) if len(w12) >= 12 else float("nan")
        std_180 = float(np.nanstd(w12[-36:])) if len(w12) >= 36 else float("nan")
        mad_30 = _mad(w12[-6:]) if len(w12) >= 6 else float("nan")
        mad_60 = _mad(w12[-12:]) if len(w12) >= 12 else float("nan")
        mad_180 = _mad(w12[-36:]) if len(w12) >= 36 else float("nan")

        mean_abs_delta_60 = float("nan")
        if len(w12) >= 12:
            diffs = np.diff(w12[-12:])
            mean_abs_delta_60 = float(np.nanmean(np.abs(diffs)))

        max_drop_30_last6h = float("nan")
        max_drop_60_last6h = float("nan")
        drop_cnt_30_ge0p5 = float("nan")
        drop_cnt_30_ge1 = float("nan")
        drop_cnt_30_ge2 = float("nan")
        last6h = w12[-72:] if len(w12) >= 72 else w12
        if len(last6h) >= 13:
            drops30 = last6h[:-6] - last6h[6:]
            drops60 = last6h[:-12] - last6h[12:]
            max_drop_30_last6h = float(np.nanmax(drops30))
            max_drop_60_last6h = float(np.nanmax(drops60))
            drop_cnt_30_ge0p5 = float(np.sum(drops30 >= 0.5))
            drop_cnt_30_ge1 = float(np.sum(drops30 >= 1.0))
            drop_cnt_30_ge2 = float(np.sum(drops30 >= 2.0))

        # Drop features from local midnight -> cutoff (partial series)
        partial_filled = pd.Series(partial_vals).interpolate(limit_direction="both").to_numpy()
        max_drop_10 = float("nan")
        max_drop_30 = float("nan")
        drop_cnt_30_ge_1f = float("nan")
        drop_cnt_30_ge_2f = float("nan")
        drop_cnt_10_ge_0p5 = float("nan")
        drop_cnt_20_ge_0p5 = float("nan")
        drop_cnt_30_ge_0p5 = float("nan")
        drop_cnt_10_ge_1f = float("nan")
        drop_cnt_10_ge_2f = float("nan")
        drop_cnt_20_ge_1f = float("nan")
        drop_cnt_20_ge_2f = float("nan")
        if len(partial_filled) >= 7:
            drops10 = partial_filled[:-2] - partial_filled[2:]
            drops20 = partial_filled[:-4] - partial_filled[4:]
            drops30_full = partial_filled[:-6] - partial_filled[6:]
            max_drop_10 = float(np.nanmax(drops10))
            max_drop_30 = float(np.nanmax(drops30_full))
            drop_cnt_10_ge_0p5 = float(np.sum(drops10 >= 0.5))
            drop_cnt_20_ge_0p5 = float(np.sum(drops20 >= 0.5))
            drop_cnt_30_ge_0p5 = float(np.sum(drops30_full >= 0.5))
            drop_cnt_10_ge_1f = float(np.sum(drops10 >= 1.0))
            drop_cnt_10_ge_2f = float(np.sum(drops10 >= 2.0))
            drop_cnt_20_ge_1f = float(np.sum(drops20 >= 1.0))
            drop_cnt_20_ge_2f = float(np.sum(drops20 >= 2.0))
            drop_cnt_30_ge_1f = float(np.sum(drops30_full >= 1.0))
            drop_cnt_30_ge_2f = float(np.sum(drops30_full >= 2.0))

        # Cooling run length (consecutive negative deltas)
        cooling_run_len = float("nan")
        if len(partial_filled) >= 3:
            deltas = np.diff(partial_filled)
            longest = 0
            run = 0
            for d in deltas:
                if d < 0:
                    run += 1
                    if run > longest:
                        longest = run
                else:
                    run = 0
            cooling_run_len = float(longest * 5)

        # Plateaus (last 120m)
        plateau_frac_0p1 = float("nan")
        plateau_frac_0p2 = float("nan")
        plateau_frac_0p3 = float("nan")
        plateau_longest_run_0p1 = float("nan")
        plateau_longest_run_0p2 = float("nan")
        plateau_longest_run_0p3 = float("nan")
        if len(w12) >= 24 and np.isfinite(tmax_sofar):
            last120 = w12[-24:]
            mask_0p1 = last120 >= (tmax_sofar - 0.1)
            mask_0p2 = last120 >= (tmax_sofar - 0.2)
            mask_0p3 = last120 >= (tmax_sofar - 0.3)
            plateau_frac_0p1 = float(np.nanmean(mask_0p1))
            plateau_frac_0p2 = float(np.nanmean(mask_0p2))
            plateau_frac_0p3 = float(np.nanmean(mask_0p3))
            plateau_longest_run_0p1 = float(_longest_run(mask_0p1) * 5)
            plateau_longest_run_0p2 = float(_longest_run(mask_0p2) * 5)
            plateau_longest_run_0p3 = float(_longest_run(mask_0p3) * 5)

        # Drop dynamics + rebound on last 6h
        drop_cnt_0p5 = float("nan")
        max_drop_30m = float("nan")
        rebound_after_drop = float("nan")
        last360 = partial_filled[-72:] if len(partial_filled) >= 72 else partial_filled
        if len(last360) >= 7:
            diffs = np.diff(last360)
            drop_cnt_0p5 = float(np.sum(diffs <= -0.5))
            drops30 = last360[:-6] - last360[6:]
            max_drop_30m = float(np.nanmax(drops30))
            rebound_vals = []
            for i in range(len(diffs)):
                if diffs[i] <= -0.5:
                    event_idx = i + 1
                    if event_idx + 6 < len(last360):
                        rebound_vals.append(last360[event_idx + 6] - last360[event_idx])
            rebound_after_drop = float(np.nanmax(rebound_vals)) if rebound_vals else float("nan")

        # DCT features
        dct_vals = _resample_fixed(partial_vals, dct_len)
        med = np.nanmedian(dct_vals)
        std = np.nanstd(dct_vals)
        std = std if np.isfinite(std) and std > 0 else 1.0
        dct_std = (dct_vals - med) / std
        coeffs_std = dct_mat.T @ dct_std
        energy = np.sum(coeffs_std[:20] ** 2)
        hi_energy = np.sum(coeffs_std[10:20] ** 2)
        hi_freq_energy_ratio = float(hi_energy / energy) if energy > 0 else float("nan")

        norm_vals = dct_vals.copy()
        vmin = np.nanmin(norm_vals)
        vmax = np.nanmax(norm_vals)
        vrange = vmax - vmin if np.isfinite(vmax) and np.isfinite(vmin) else 0.0
        norm_vals = (norm_vals - vmin) / (vrange + 0.5)
        coeffs_norm = dct_mat.T @ norm_vals

        # Coverage / gaps
        is_present = partial_series.notna().to_numpy()
        frac_bins_present_total = float(np.nanmean(is_present)) if is_present.size else float("nan")
        max_gap_minutes_total, gap_cnt_15, gap_cnt_30 = _gap_stats(is_present, step_minutes=5)
        is_present_last180 = is_present[-36:] if is_present.size >= 36 else is_present
        frac_bins_present_last180 = float(np.nanmean(is_present_last180)) if is_present_last180.size else float("nan")
        max_gap_minutes_last180, _, _ = _gap_stats(is_present_last180, step_minutes=5)

        spike_cnt, spike_idx_cnt = _spike_stats(partial_filled, step_minutes=5)
        max_sofar_is_spike_flag = 0
        if spike_idx_cnt > 0:
            max_idx = int(np.nanargmax(partial_filled)) if np.isfinite(np.nanmax(partial_filled)) else -1
            if max_idx >= 0:
                for i in range(1, len(partial_filled) - 1):
                    d1 = partial_filled[i] - partial_filled[i - 1]
                    d2 = partial_filled[i + 1] - partial_filled[i]
                    if (d1 >= 2.0 and d2 <= -2.0) or (d1 <= -2.0 and d2 >= 2.0):
                        if i == max_idx:
                            max_sofar_is_spike_flag = 1
                            break

        # Warming fraction and persistence
        warming_fraction_last180 = float("nan")
        if len(last360) >= 2:
            warming_fraction_last180 = float(np.nanmean(np.diff(last360[-36:]) > 0))

        high_end_persistence_min = float("nan")
        last240 = partial_filled[-48:] if len(partial_filled) >= 48 else partial_filled
        if last240.size > 0 and np.isfinite(tmax_sofar):
            high_end_persistence_min = float(np.sum(last240 >= (tmax_sofar - 0.2)) * 5)

        # Slope histogram bins
        def slope_hist(values: np.ndarray) -> List[float]:
            if values.size < 2:
                return [float("nan")] * 7
            slopes_hr = np.diff(values) * 12.0
            bins = [-6.0, -3.0, -1.0, -0.3, 0.3, 1.0, 3.0, 6.0]
            hist, _ = np.histogram(slopes_hr[np.isfinite(slopes_hr)], bins=bins)
            return [float(x) for x in hist]

        slope_bins_6h = slope_hist(last360)
        slope_bins_full = slope_hist(partial_filled)

        convective_crash_cnt = float("nan")
        crash_recovery_ratio = float("nan")
        if len(last360) >= 3:
            slopes = np.diff(last360)
            convective_crash_cnt = float(np.sum(slopes <= -0.7))
            recos = []
            for i in range(len(slopes)):
                if slopes[i] <= -0.7 and i + 6 < len(last360):
                    crash_mag = abs(slopes[i])
                    pos_sum = np.sum(np.clip(np.diff(last360[i + 1 : i + 7]), 0.0, None))
                    if crash_mag > 0:
                        recos.append(pos_sum / crash_mag)
            crash_recovery_ratio = float(np.nanmean(recos)) if recos else float("nan")

        # Outflow events (local 06:00 -> cutoff)
        outflow_cnt = 0
        outflow_max_crash = float("nan")
        outflow_mean_crash = float("nan")
        outflow_min_recovery_frac = float("nan")
        outflow_last_event_age_min = float("nan")
        crash_energy = float("nan")

        local_times = partial_series.index.tz_convert(LOCAL_TZ)
        local_minutes = (local_times.hour * 60 + local_times.minute).to_numpy()
        mask_06 = local_minutes >= 360
        series_06 = partial_filled[mask_06]
        times_06 = partial_series.index[mask_06]
        crashes = []
        recovery_fracs = []
        event_times = []
        if series_06.size >= 3:
            delta_10 = series_06[2:] - series_06[:-2]
            for i in range(len(delta_10)):
                if delta_10[i] <= -1.0:
                    crash_mag = series_06[i] - series_06[i + 2]
                    crashes.append(crash_mag)
                    if i + 8 < len(series_06):
                        recovery_30 = series_06[i + 8] - series_06[i + 2]
                        recovery_fracs.append(recovery_30 / max(0.1, crash_mag))
                    event_times.append(times_06[i + 2])
            if crashes:
                outflow_cnt = len(crashes)
                outflow_max_crash = float(np.nanmax(crashes))
                outflow_mean_crash = float(np.nanmean(crashes))
                outflow_min_recovery_frac = float(np.nanmin(recovery_fracs)) if recovery_fracs else float("nan")
                crash_energy = float(np.nansum(crashes))
                last_event = max(event_times)
                outflow_last_event_age_min = float((cutoff - last_event).total_seconds() / 60.0) if pd.notna(last_event) else float("nan")

        # Early day heat integral (local 06:00 -> cutoff)
        early_day_heat_integral = float("nan")
        if series_06.size >= 2:
            deltas = np.diff(series_06)
            early_day_heat_integral = float(np.nansum(np.clip(deltas, 0.0, None)))

        # SBFD features
        cutoff_local = cutoff.astimezone(ZoneInfo(LOCAL_TZ))
        cutoff_local_min = cutoff_local.hour * 60 + cutoff_local.minute
        sb_mask = (local_minutes >= 540) & (local_minutes <= cutoff_local_min)
        sb_vals = partial_series.to_numpy()[sb_mask]
        sb_times = local_minutes[sb_mask].astype(float)
        sb_expected = sb_mask.sum()
        sb_present = np.isfinite(sb_vals).sum()
        sb_coverage = float(sb_present / sb_expected) if sb_expected > 0 else float("nan")
        sb_break_time_local_min = float("nan")
        sb_slope_pre = float("nan")
        sb_slope_post = float("nan")
        sb_slope_drop = float("nan")
        sb_plateau_after = float("nan")
        sb_cooling_pulse = float("nan")
        sb_stabilized_flag = 0
        sbfd_missing = 1
        if sb_expected >= 6 and sb_coverage >= 0.85:
            sbfd_missing = 0
            sb_vals_filled = pd.Series(sb_vals).interpolate(limit_direction="both").to_numpy()
            times_rel = sb_times - sb_times.min()
            piece = _piecewise_two_segment(times_rel, sb_vals_filled, min_seg=6)
            if piece["break_idx"] is not None:
                sb_break_time_local_min = float(sb_times[piece["break_idx"]])
                sb_slope_pre = float(piece["slope_pre"])
                sb_slope_post = float(piece["slope_post"])
                sb_slope_drop = sb_slope_pre - sb_slope_post
                after_vals = sb_vals_filled[piece["break_idx"] :]
                if after_vals.size > 0 and np.isfinite(tmax_sofar):
                    sb_plateau_after = float(np.nanmean(after_vals >= (tmax_sofar - 0.2)))
                    if after_vals.size >= 5:
                        diffs20 = after_vals[4:] - after_vals[:-4]
                        sb_cooling_pulse = float(np.nanmin(diffs20)) if diffs20.size else float("nan")
                if np.isfinite(sb_slope_post) and np.isfinite(sb_plateau_after):
                    sb_stabilized_flag = int((sb_slope_post <= 0.2) and (sb_plateau_after >= 0.6))

        # Observations at 12Z / 15Z
        obs_at_12z = float("nan")
        obs_at_15z = float("nan")
        if cutoff >= datetime(cutoff.year, cutoff.month, cutoff.day, 12, 0, tzinfo=timezone.utc):
            anchor = datetime(cutoff.year, cutoff.month, cutoff.day, 12, 0, tzinfo=timezone.utc)
            obs_at_12z = float(partial_series.loc[:anchor].dropna().iloc[-1]) if not partial_series.loc[:anchor].dropna().empty else float("nan")
        if cutoff >= datetime(cutoff.year, cutoff.month, cutoff.day, 15, 0, tzinfo=timezone.utc):
            anchor = datetime(cutoff.year, cutoff.month, cutoff.day, 15, 0, tzinfo=timezone.utc)
            obs_at_15z = float(partial_series.loc[:anchor].dropna().iloc[-1]) if not partial_series.loc[:anchor].dropna().empty else float("nan")

        # Exceedance after cutoff (hazard target)
        post_idx = pd.date_range(cutoff + timedelta(minutes=5), day_end - timedelta(minutes=5), freq="5min")
        post_series = series_5m.reindex(post_idx)
        tmax_post = float(np.nanmax(post_series.to_numpy())) if post_series.notna().sum() > 0 else float("nan")
        y_exceed_future = float("nan")
        if np.isfinite(tmax_post) and np.isfinite(tmax_sofar):
            y_exceed_future = int(tmax_post > (tmax_sofar + EPS))

        rows.append(
            {
                "target_date_local": day,
                "cutoff_utc": cutoff,
                "tmax_full": tmax_full,
                "tmax_sofar": tmax_sofar,
                "y_hit_by_cutoff": int(tmax_sofar >= (tmax_full - EPS)),
                "y_exceed_future": y_exceed_future,
                "temp_now": temp_now,
                "max_sofar": tmax_sofar,
                "tmin_sofar": tmin_sofar,
                "range_sofar": range_sofar,
                "minutes_since_max": minutes_since_max,
                "drop_from_max": drop_from_max,
                "tmax_sofar_time_local_minute": tmax_sofar_time_local_minute,
                "slope_15m": slope_15,
                "slope_30m": slope_30,
                "slope_60m": slope_60,
                "slope_120m": slope_120,
                "slope_180m": slope_180,
                "accel": accel,
                "curvature_30_180": curvature_30_180,
                "curvature_60_180": curvature_60_180,
                "slope_sign_changes_last180": slope_sign_changes_last180,
                "cp_minute": cp_minute,
                "cp_slope_before": best_before,
                "cp_slope_after": best_after,
                "cp_drop": best_drop,
                "cp4_minute": cp4_minute,
                "cp4_slope_before": best4_before,
                "cp4_slope_after": best4_after,
                "cp4_drop": best4_drop,
                "std_30m": std_30,
                "std_60": std_60,
                "std_180": std_180,
                "mad_30m": mad_30,
                "mad_60m": mad_60,
                "mad_180m": mad_180,
                "mean_abs_delta_60m": mean_abs_delta_60,
                "max_drop_30m_last6h": max_drop_30_last6h,
                "max_drop_60m_last6h": max_drop_60_last6h,
                "drop_cnt_30m_ge0p5_last6h": drop_cnt_30_ge0p5,
                "drop_cnt_30m_ge1_last6h": drop_cnt_30_ge1,
                "drop_cnt_30m_ge2_last6h": drop_cnt_30_ge2,
                "max_drop_10": max_drop_10,
                "max_drop_30": max_drop_30,
                "drop_cnt_30_ge_1F": drop_cnt_30_ge_1f,
                "drop_cnt_30_ge_2F": drop_cnt_30_ge_2f,
                "drop_cnt_10_ge_0p5": drop_cnt_10_ge_0p5,
                "drop_cnt_20_ge_0p5": drop_cnt_20_ge_0p5,
                "drop_cnt_30_ge_0p5": drop_cnt_30_ge_0p5,
                "drop_cnt_10_ge_1F": drop_cnt_10_ge_1f,
                "drop_cnt_10_ge_2F": drop_cnt_10_ge_2f,
                "drop_cnt_20_ge_1F": drop_cnt_20_ge_1f,
                "drop_cnt_20_ge_2F": drop_cnt_20_ge_2f,
                "cooling_run_len": cooling_run_len,
                "plateau_frac_0p1_last120": plateau_frac_0p1,
                "plateau_frac_0p2_last120": plateau_frac_0p2,
                "plateau_frac_0p3_last120": plateau_frac_0p3,
                "plateau_longest_run_0p1_last120": plateau_longest_run_0p1,
                "plateau_longest_run_0p2_last120": plateau_longest_run_0p2,
                "plateau_longest_run_0p3_last120": plateau_longest_run_0p3,
                "drop_cnt_0p5_last6h": drop_cnt_0p5,
                "max_drop_30m": max_drop_30m,
                "rebound_after_drop": rebound_after_drop,
                "warming_fraction_last180": warming_fraction_last180,
                "high_end_persistence_min": high_end_persistence_min,
                "range_full": range_full,
                "tmax_time_local_minute": tmax_time_local_minute,
                "frac_bins_present_total": frac_bins_present_total,
                "frac_bins_present_last180": frac_bins_present_last180,
                "max_gap_minutes_total": max_gap_minutes_total,
                "max_gap_minutes_last180": max_gap_minutes_last180,
                "gap_cnt_ge15m": gap_cnt_15,
                "gap_cnt_ge30m": gap_cnt_30,
                "spike_cnt": spike_cnt,
                "max_sofar_is_spike_flag": max_sofar_is_spike_flag,
                "hi_freq_energy_ratio": hi_freq_energy_ratio,
                "obs_at_12z": obs_at_12z,
                "obs_at_15z": obs_at_15z,
                "early_day_heat_integral": early_day_heat_integral,
                "outflow_cnt": outflow_cnt,
                "outflow_max_crash": outflow_max_crash,
                "outflow_mean_crash": outflow_mean_crash,
                "outflow_min_recovery_frac": outflow_min_recovery_frac,
                "outflow_last_event_age_min": outflow_last_event_age_min,
                "crash_energy": crash_energy,
                "sb_break_time_local_min": sb_break_time_local_min,
                "sb_slope_pre": sb_slope_pre,
                "sb_slope_post": sb_slope_post,
                "sb_slope_drop": sb_slope_drop,
                "sb_plateau_after": sb_plateau_after,
                "sb_cooling_pulse": sb_cooling_pulse,
                "sb_stabilized_flag": sb_stabilized_flag,
                "sbfd_missing": sbfd_missing,
                "slope6h_bin0": slope_bins_6h[0],
                "slope6h_bin1": slope_bins_6h[1],
                "slope6h_bin2": slope_bins_6h[2],
                "slope6h_bin3": slope_bins_6h[3],
                "slope6h_bin4": slope_bins_6h[4],
                "slope6h_bin5": slope_bins_6h[5],
                "slope6h_bin6": slope_bins_6h[6],
                "slopefull_bin0": slope_bins_full[0],
                "slopefull_bin1": slope_bins_full[1],
                "slopefull_bin2": slope_bins_full[2],
                "slopefull_bin3": slope_bins_full[3],
                "slopefull_bin4": slope_bins_full[4],
                "slopefull_bin5": slope_bins_full[5],
                "slopefull_bin6": slope_bins_full[6],
                "convective_crash_cnt": convective_crash_cnt,
                "crash_recovery_ratio": crash_recovery_ratio,
            }
        )

        for i in range(20):
            rows[-1][f"dct_curve_{i}"] = float(coeffs_std[i]) if i < len(coeffs_std) else float("nan")
        for i in range(21):
            rows[-1][f"dct_norm_{i}"] = float(coeffs_norm[i]) if i < len(coeffs_norm) else float("nan")

    minute_df = pd.DataFrame(rows).sort_values("target_date_local")

    # Lag features
    minute_df["tmax_time_local_lag1"] = minute_df["tmax_time_local_minute"].shift(1)
    minute_df["tmax_time_local_lag2"] = minute_df["tmax_time_local_minute"].shift(2)
    minute_df["range_lag1"] = minute_df["range_full"].shift(1)
    minute_df["range_lag2"] = minute_df["range_full"].shift(2)

    # Outflow drop count on full day (30m drops >=2)
    outflow_counts = []
    for _, dw in day_windows.iterrows():
        day_start = dw["day_start_utc"]
        day_end = dw["day_end_utc"]
        full_idx = pd.date_range(day_start, day_end - timedelta(minutes=5), freq="5min")
        full_series = series_5m.reindex(full_idx).to_numpy()
        if len(full_series) < 7:
            outflow_counts.append(float("nan"))
            continue
        drops = full_series[:-6] - full_series[6:]
        outflow_counts.append(float(np.nansum(drops >= 2.0)))

    outflow_df = pd.DataFrame({
        "target_date_local": day_windows["target_date_local"],
        "outflow_drop_cnt": outflow_counts,
    })

    minute_df = minute_df.merge(outflow_df, on="target_date_local", how="left")
    minute_df["outflow_drop_cnt_lag1"] = minute_df["outflow_drop_cnt"].shift(1)
    minute_df["delta_range_1d"] = minute_df["range_lag1"] - minute_df["range_lag2"]

    if minute_violations > 0:
        raise RuntimeError(f"Minute leakage audit failed: {minute_violations} violations")

    _write_cache(cache_path, meta_path, minute_df, cache_hash)
    return minute_df

def _mos_version_stamp(engine) -> str:
    query = text(
        f"""
        SELECT MAX(id) AS max_id, MAX(retrieved_at_utc) AS max_retrieved
        FROM mos_daily_value
        WHERE station_id='KMIA'
          AND model IN ('GFS','NAM')
          AND target_date_local BETWEEN '2002-01-01' AND '2026-12-31'
          AND variable_code IN ({MOS_VARIABLES_SQL})
        """
    )
    with engine.connect() as conn:
        row = conn.execute(query).mappings().first()
    if not row:
        return "none"
    return f"{row['max_id']}|{row['max_retrieved']}"


def _load_mos_raw(cache_dir: Path, reuse_cache: bool, rebuild_cache: bool, engine) -> pd.DataFrame:
    cache_path = cache_dir / "mos_raw_kmia.parquet"
    meta_path = cache_dir / "mos_raw_kmia.meta.json"
    mos_stamp = _mos_version_stamp(engine)
    cache_hash = _sha256_str(json.dumps({"version": VERSION, "mos_stamp": mos_stamp, "vars": MOS_VARIABLES}, sort_keys=True))

    if not rebuild_cache:
        cached = _read_cache(cache_path, meta_path, cache_hash, reuse_cache)
        if cached is not None:
            return cached

    query = f"""
        SELECT target_date_local, model, variable_code, asof_utc, runtime_utc, retrieved_at_utc, id,
               value_min, value_max, value_mean, value_median, sample_count
        FROM mos_daily_value
        WHERE station_id='KMIA'
          AND model IN ('GFS','NAM')
          AND target_date_local BETWEEN '2002-01-01' AND '2026-12-31'
          AND variable_code IN ({MOS_VARIABLES_SQL})
    """

    frames = []
    with engine.connect() as conn:
        for chunk in pd.read_sql_query(text(query), conn, chunksize=250000):
            frames.append(chunk)
    mos_raw = pd.concat(frames, ignore_index=True)

    mos_raw["asof_utc"] = pd.to_datetime(mos_raw["asof_utc"], utc=True, errors="coerce")
    mos_raw["runtime_utc"] = pd.to_datetime(mos_raw["runtime_utc"], utc=True, errors="coerce")
    mos_raw["retrieved_at_utc"] = pd.to_datetime(mos_raw["retrieved_at_utc"], utc=True, errors="coerce")

    _write_cache(cache_path, meta_path, mos_raw, cache_hash)
    return mos_raw


def _select_latest_mos(mos_raw: pd.DataFrame, cutoff_map: Dict[date, datetime]) -> pd.DataFrame:
    mos = mos_raw.copy()
    mos["target_date_local"] = pd.to_datetime(mos["target_date_local"]).dt.date
    mos["cutoff_utc"] = mos["target_date_local"].map(cutoff_map)
    mos = mos[mos["cutoff_utc"].notna()].copy()
    mos = mos[mos["asof_utc"] <= mos["cutoff_utc"]].copy()
    if (mos["asof_utc"] > mos["cutoff_utc"]).any():
        raise RuntimeError("MOS leakage audit failed: asof_utc beyond cutoff_utc detected")
    mos = mos.sort_values(["asof_utc", "runtime_utc", "retrieved_at_utc", "id"])
    latest = mos.groupby(["target_date_local", "model", "variable_code"], as_index=False).tail(1)
    return latest


def _circular_mean_deg(deg_values: List[float]) -> float:
    vals = [v for v in deg_values if np.isfinite(v)]
    if not vals:
        return float("nan")
    radians = np.deg2rad(vals)
    sin_mean = np.mean(np.sin(radians))
    cos_mean = np.mean(np.cos(radians))
    angle = math.atan2(sin_mean, cos_mean)
    deg = np.rad2deg(angle)
    if deg < 0:
        deg += 360.0
    return float(deg)

def _angular_diff_deg(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    diff = abs(a - b) % 360.0
    if diff > 180.0:
        diff = 360.0 - diff
    return float(diff)

def _mos_base_features(latest: pd.DataFrame) -> pd.DataFrame:
    latest = latest.copy()
    latest["target_date_local"] = pd.to_datetime(latest["target_date_local"]).dt.date

    pv_max = latest.pivot(index="target_date_local", columns=["model", "variable_code"], values="value_max")
    pv_min = latest.pivot(index="target_date_local", columns=["model", "variable_code"], values="value_min")
    pv_mean = latest.pivot(index="target_date_local", columns=["model", "variable_code"], values="value_mean")

    def get_col(pv: pd.DataFrame, model: str, var: str) -> pd.Series:
        try:
            return pv[(model, var)]
        except KeyError:
            return pd.Series(index=pv.index, dtype=float)

    out = pd.DataFrame({"target_date_local": pv_max.index}).reset_index(drop=True)

    # n_x special: MOS daytime max (X) and nighttime min (N)
    x_gfs = get_col(pv_max, "GFS", "n_x")
    x_nam = get_col(pv_max, "NAM", "n_x")
    n_gfs = get_col(pv_min, "GFS", "n_x")
    n_nam = get_col(pv_min, "NAM", "n_x")
    mos_x_mean = pd.concat([x_gfs, x_nam], axis=1).mean(axis=1)
    mos_n_mean = pd.concat([n_gfs, n_nam], axis=1).mean(axis=1)
    mos_range = mos_x_mean - mos_n_mean
    mos_x_disagree = (x_gfs - x_nam).abs()
    mos_range_disagree = (x_gfs - n_gfs - (x_nam - n_nam)).abs()

    out["mos_x_gfs"] = x_gfs.values
    out["mos_x_nam"] = x_nam.values
    out["mos_n_gfs"] = n_gfs.values
    out["mos_n_nam"] = n_nam.values
    out["mos_x_mean"] = mos_x_mean.values
    out["mos_n_mean"] = mos_n_mean.values
    out["mos_range"] = mos_range.values
    out["mos_x_disagree"] = mos_x_disagree.values
    out["mos_range_disagree"] = mos_range_disagree.values

    # tmp-based tmax/tmin (for SBFD fusion)
    tmp_max_gfs = get_col(pv_max, "GFS", "tmp")
    tmp_max_nam = get_col(pv_max, "NAM", "tmp")
    tmp_min_gfs = get_col(pv_min, "GFS", "tmp")
    tmp_min_nam = get_col(pv_min, "NAM", "tmp")
    mos_tmax_mean = pd.concat([tmp_max_gfs, tmp_max_nam], axis=1).mean(axis=1)
    mos_tmin_mean = pd.concat([tmp_min_gfs, tmp_min_nam], axis=1).mean(axis=1)
    mos_tmp_range = mos_tmax_mean - mos_tmin_mean
    out["mos_tmax_gfs"] = tmp_max_gfs.values
    out["mos_tmax_nam"] = tmp_max_nam.values
    out["mos_tmin_gfs"] = tmp_min_gfs.values
    out["mos_tmin_nam"] = tmp_min_nam.values
    out["mos_tmax_mean"] = mos_tmax_mean.values
    out["mos_tmin_mean"] = mos_tmin_mean.values
    out["mos_range_mean"] = mos_tmp_range.values
    out["mos_tmax_disagree"] = (tmp_max_gfs - tmp_max_nam).abs().values
    out["mos_range_disagree_tmp"] = (tmp_max_gfs - tmp_min_gfs - (tmp_max_nam - tmp_min_nam)).abs().values

    # Core vars (tmp, dpt, wsp, wdr, cig, vis)
    core_vars = ["tmp", "dpt", "wsp", "wdr", "cig", "vis"]
    for var in core_vars:
        gfs_mean = get_col(pv_mean, "GFS", var)
        nam_mean = get_col(pv_mean, "NAM", var)
        gfs_max = get_col(pv_max, "GFS", var)
        nam_max = get_col(pv_max, "NAM", var)
        gfs_min = get_col(pv_min, "GFS", var)
        nam_min = get_col(pv_min, "NAM", var)

        mean_models = pd.concat([gfs_mean, nam_mean], axis=1).mean(axis=1)
        max_models = pd.concat([gfs_max, nam_max], axis=1).max(axis=1)
        min_models = pd.concat([gfs_min, nam_min], axis=1).min(axis=1)

        if var == "wdr":
            wdr_mean = pd.Series(index=mean_models.index, dtype=float)
            for idx in wdr_mean.index:
                wdr_mean.loc[idx] = _circular_mean_deg([gfs_mean.get(idx, np.nan), nam_mean.get(idx, np.nan)])
            out["mos_wdr_mean_models"] = wdr_mean.values
            out["mos_wdr_mean_disagree"] = [
                _angular_diff_deg(gfs_mean.get(idx, np.nan), nam_mean.get(idx, np.nan)) for idx in wdr_mean.index
            ]
        else:
            out[f"mos_{var}_mean_models"] = mean_models.values
            out[f"mos_{var}_mean_disagree"] = (gfs_mean - nam_mean).abs().values

        out[f"mos_{var}_gfs_mean"] = gfs_mean.values
        out[f"mos_{var}_nam_mean"] = nam_mean.values
        out[f"mos_{var}_gfs_max"] = gfs_max.values
        out[f"mos_{var}_nam_max"] = nam_max.values
        out[f"mos_{var}_gfs_min"] = gfs_min.values
        out[f"mos_{var}_nam_min"] = nam_min.values
        out[f"mos_{var}_max_models"] = max_models.values
        out[f"mos_{var}_min_models"] = min_models.values

    # Convective / suppression vars
    conv_vars = ["p06", "p12", "q06", "q12", "t06", "t06_1", "t06_2", "t12", "t12_1", "t12_2"]
    for var in conv_vars:
        gfs_mean = get_col(pv_mean, "GFS", var)
        nam_mean = get_col(pv_mean, "NAM", var)
        gfs_max = get_col(pv_max, "GFS", var)
        nam_max = get_col(pv_max, "NAM", var)
        gfs_min = get_col(pv_min, "GFS", var)
        nam_min = get_col(pv_min, "NAM", var)

        mean_models = pd.concat([gfs_mean, nam_mean], axis=1).mean(axis=1)
        max_models = pd.concat([gfs_max, nam_max], axis=1).max(axis=1)
        min_models = pd.concat([gfs_min, nam_min], axis=1).min(axis=1)

        out[f"mos_{var}_gfs_mean"] = gfs_mean.values
        out[f"mos_{var}_nam_mean"] = nam_mean.values
        out[f"mos_{var}_gfs_max"] = gfs_max.values
        out[f"mos_{var}_nam_max"] = nam_max.values
        out[f"mos_{var}_gfs_min"] = gfs_min.values
        out[f"mos_{var}_nam_min"] = nam_min.values
        out[f"mos_{var}_mean_models"] = mean_models.values
        out[f"mos_{var}_max_models"] = max_models.values
        out[f"mos_{var}_min_models"] = min_models.values
        out[f"mos_{var}_mean_disagree"] = (gfs_mean - nam_mean).abs().values
        out[f"mos_{var}_max_disagree"] = (gfs_max - nam_max).abs().values

    # Latest asof per model
    asof_latest = latest.groupby(["target_date_local", "model"], as_index=False)["asof_utc"].max()
    asof_pivot = asof_latest.pivot(index="target_date_local", columns="model", values="asof_utc")
    out["mos_latest_asof_utc_gfs"] = asof_pivot.get("GFS").values
    out["mos_latest_asof_utc_nam"] = asof_pivot.get("NAM").values

    tmp_rows = latest[latest["variable_code"] == "tmp"]
    if not tmp_rows.empty:
        tmp_asof = tmp_rows.pivot(index="target_date_local", columns="model", values="asof_utc")
        out["mos_tmp_asof_utc_gfs"] = tmp_asof.get("GFS").values
        out["mos_tmp_asof_utc_nam"] = tmp_asof.get("NAM").values
    # normalize timezone awareness
    for col in ["mos_latest_asof_utc_gfs", "mos_latest_asof_utc_nam", "mos_tmp_asof_utc_gfs", "mos_tmp_asof_utc_nam"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")

    # Wind components (meteorological convention)
    wdr_mean_models = out.get("mos_wdr_mean_models")
    wsp_mean_models = out.get("mos_wsp_mean_models")
    if wdr_mean_models is not None and wsp_mean_models is not None:
        wdr_rad = np.deg2rad(wdr_mean_models)
        out["mos_u_mean"] = -wsp_mean_models * np.sin(wdr_rad)
        out["mos_v_mean"] = -wsp_mean_models * np.cos(wdr_rad)

    return out


def _build_mos_features(cache_dir: Path, reuse_cache: bool, rebuild_cache: bool, engine) -> pd.DataFrame:
    cache_path = cache_dir / "hit1830_v5_mos_features.parquet"
    meta_path = cache_dir / "hit1830_v5_mos_features.meta.json"

    mos_stamp = _mos_version_stamp(engine)
    cache_hash = _sha256_str(json.dumps({"version": VERSION, "mos_stamp": mos_stamp, "vars": MOS_VARIABLES}, sort_keys=True))

    if not rebuild_cache:
        cached = _read_cache(cache_path, meta_path, cache_hash, reuse_cache)
        if cached is not None:
            return cached

    day_windows = _build_day_windows()
    cutoff_map_0 = {row.target_date_local: row.cutoff_utc for row in day_windows.itertuples()}

    mos_raw = _load_mos_raw(cache_dir, reuse_cache, rebuild_cache, engine)
    latest_0 = _select_latest_mos(mos_raw, cutoff_map_0)
    base_0 = _mos_base_features(latest_0)

    cycle_feats = _mos_cycle_revision_features(mos_raw, day_windows)
    rev_feats = _mos_revision_features(mos_raw, day_windows)
    mos = base_0.merge(cycle_feats, on="target_date_local", how="left")
    mos = mos.merge(rev_feats, on="target_date_local", how="left")

    _write_cache(cache_path, meta_path, mos, cache_hash)
    return mos


def _mos_cycle_revision_features(mos_raw: pd.DataFrame, day_windows: pd.DataFrame) -> pd.DataFrame:
    # Build cutoff maps for cycle-aligned revisions
    def cycle_time_map(hours: int, day_offset: int = 0) -> Dict[date, datetime]:
        mapping = {}
        for row in day_windows.itertuples():
            day = row.target_date_local + timedelta(days=day_offset)
            cycle_time = datetime(day.year, day.month, day.day, hours, 0, tzinfo=timezone.utc)
            mapping[row.target_date_local] = cycle_time
        return mapping

    cutoff_map_12 = cycle_time_map(12, 0)
    cutoff_map_00 = cycle_time_map(0, 0)
    cutoff_map_prev12 = cycle_time_map(12, -1)

    latest_12 = _select_latest_mos(mos_raw, cutoff_map_12)
    latest_00 = _select_latest_mos(mos_raw, cutoff_map_00)
    latest_prev12 = _select_latest_mos(mos_raw, cutoff_map_prev12)

    rev_vars = {
        "n_x": "max",
        "cig": "min",
        "vis": "min",
        "p12": "max",
        "q12": "max",
        "t12": "max",
        "wsp": "mean",
        "wdr": "mean",
        "dpt": "mean",
    }

    def cycle_values(latest: pd.DataFrame) -> pd.DataFrame:
        pv_max = latest.pivot(index="target_date_local", columns=["model", "variable_code"], values="value_max")
        pv_min = latest.pivot(index="target_date_local", columns=["model", "variable_code"], values="value_min")
        pv_mean = latest.pivot(index="target_date_local", columns=["model", "variable_code"], values="value_mean")

        def get_col(pv: pd.DataFrame, model: str, var: str) -> pd.Series:
            try:
                return pv[(model, var)]
            except KeyError:
                return pd.Series(index=pv.index, dtype=float)

        out = pd.DataFrame({"target_date_local": pv_max.index}).reset_index(drop=True)
        for var, stat in rev_vars.items():
            if stat == "max":
                gfs = get_col(pv_max, "GFS", var)
                nam = get_col(pv_max, "NAM", var)
            elif stat == "min":
                gfs = get_col(pv_min, "GFS", var)
                nam = get_col(pv_min, "NAM", var)
            else:
                gfs = get_col(pv_mean, "GFS", var)
                nam = get_col(pv_mean, "NAM", var)
            if var == "wdr":
                vals = []
                for idx in gfs.index:
                    vals.append(_circular_mean_deg([gfs.get(idx, np.nan), nam.get(idx, np.nan)]))
                out[var] = vals
            else:
                out[var] = pd.concat([gfs, nam], axis=1).mean(axis=1).values
        return out

    vals_12 = cycle_values(latest_12)
    vals_00 = cycle_values(latest_00)
    vals_prev12 = cycle_values(latest_prev12)

    out = pd.DataFrame({"target_date_local": day_windows["target_date_local"]})
    out = out.merge(vals_12, on="target_date_local", how="left", suffixes=("", "_12"))
    out = out.merge(vals_00, on="target_date_local", how="left", suffixes=("", "_00"))
    out = out.merge(vals_prev12, on="target_date_local", how="left", suffixes=("", "_prev12"))

    for var in rev_vars.keys():
        if var in out.columns and f"{var}_00" in out.columns:
            out[f"rev_12z_minus_00z_{var}"] = out[var] - out[f"{var}_00"]
            out[f"abs_rev_12z_minus_00z_{var}"] = (out[var] - out[f"{var}_00"]).abs()
        if var in out.columns and f"{var}_prev12" in out.columns:
            out[f"rev_12z_minus_prev12_{var}"] = out[var] - out[f"{var}_prev12"]
            out[f"abs_rev_12z_minus_prev12_{var}"] = (out[var] - out[f"{var}_prev12"]).abs()

    # Drop raw cycle columns to keep only revisions
    drop_cols = []
    for var in rev_vars.keys():
        drop_cols.extend([var, f"{var}_00", f"{var}_prev12"])
    out = out.drop(columns=[c for c in drop_cols if c in out.columns], errors="ignore")
    return out


def _mos_revision_features(mos_raw: pd.DataFrame, day_windows: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    cutoff_map = {row.target_date_local: row.cutoff_utc for row in day_windows.itertuples()}
    mos = mos_raw.copy()
    mos["target_date_local"] = pd.to_datetime(mos["target_date_local"]).dt.date
    mos["cutoff_utc"] = mos["target_date_local"].map(cutoff_map)
    mos = mos[mos["cutoff_utc"].notna()].copy()
    mos = mos[mos["asof_utc"] <= mos["cutoff_utc"]].copy()
    if (mos["asof_utc"] > mos["cutoff_utc"]).any():
        raise RuntimeError("MOS leakage audit failed (revision features): asof_utc beyond cutoff")
    mos = mos.sort_values(["asof_utc", "runtime_utc", "retrieved_at_utc", "id"], ascending=[False, False, False, False])
    mos["rank"] = mos.groupby(["target_date_local", "model", "variable_code"]).cumcount()

    latest = mos[mos["rank"] == 0]
    prev = mos[mos["rank"] == 1]
    topn = mos[mos["rank"] < top_k]

    rev_vars = [
        "tmp", "dpt", "wsp", "wdr", "p06", "p12", "q06", "q12",
        "cig", "vis", "t06", "t12", "t06_1", "t06_2", "t12_1", "t12_2",
        "n_x", "pos", "poz"
    ]
    stats = ["value_mean", "value_max", "value_min", "value_median"]

    out = pd.DataFrame(index=day_windows["target_date_local"])
    out.index.name = "target_date_local"

    for stat in stats:
        pv_latest = latest.pivot(index="target_date_local", columns=["model", "variable_code"], values=stat).reindex(out.index)
        pv_prev = prev.pivot(index="target_date_local", columns=["model", "variable_code"], values=stat).reindex(out.index)
        pv_vol = topn.groupby(["target_date_local", "model", "variable_code"])[stat].std().unstack([1, 2]).reindex(out.index)

        for var in rev_vars:
            for model in MOS_MODELS:
                col_latest = pv_latest.get((model, var))
                col_prev = pv_prev.get((model, var))
                if col_latest is None:
                    continue
                if col_prev is not None:
                    delta = col_latest - col_prev
                else:
                    delta = np.nan
                out[f"rev_{var}_{model}_{stat}"] = delta
                out[f"rev_abs_{var}_{model}_{stat}"] = delta.abs() if hasattr(delta, "abs") else np.nan
                if pv_vol is not None and (model, var) in pv_vol.columns:
                    out[f"rev_vol_{var}_{model}_{stat}"] = pv_vol[(model, var)]

            # cross-model disagreement at cutoff
            gfs = pv_latest.get(("GFS", var))
            nam = pv_latest.get(("NAM", var))
            if gfs is not None and nam is not None:
                out[f"rev_disagree_{var}_{stat}"] = (gfs - nam)
                out[f"rev_abs_disagree_{var}_{stat}"] = (gfs - nam).abs()

    # asof gap between latest and previous
    asof_latest = latest.pivot(index="target_date_local", columns=["model", "variable_code"], values="asof_utc").reindex(out.index)
    asof_prev = prev.pivot(index="target_date_local", columns=["model", "variable_code"], values="asof_utc").reindex(out.index)
    for var in rev_vars:
        for model in MOS_MODELS:
            a0 = asof_latest.get((model, var))
            a1 = asof_prev.get((model, var))
            if a0 is None:
                continue
            if a1 is None:
                out[f"rev_asof_gap_{var}_{model}"] = np.nan
            else:
                out[f"rev_asof_gap_{var}_{model}"] = (a0 - a1).dt.total_seconds() / 3600.0

    # aggregate revision deltas for flags
    for stat in ["value_max", "value_min"]:
        for var in ["tmp", "q12", "cig"]:
            gfs = out.get(f"rev_{var}_GFS_{stat}")
            nam = out.get(f"rev_{var}_NAM_{stat}")
            if gfs is not None and nam is not None:
                out[f"rev_{var}_{stat}_mean_models"] = pd.concat([gfs, nam], axis=1).mean(axis=1)

    return out.reset_index()

def _merge_dataset(minute_df: pd.DataFrame, mos_df: pd.DataFrame, cache_dir: Path, reuse_cache: bool, rebuild_cache: bool, cache_hash: str) -> pd.DataFrame:
    cache_path = cache_dir / "hit1830_v5_features.parquet"
    meta_path = cache_dir / "hit1830_v5_features.meta.json"
    if not rebuild_cache:
        cached = _read_cache(cache_path, meta_path, cache_hash, reuse_cache)
        if cached is not None:
            return cached

    df = minute_df.merge(mos_df, on="target_date_local", how="left")

    # Calendar features
    dt = pd.to_datetime(df["target_date_local"])
    doy = dt.dt.dayofyear
    df["doy"] = doy
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["month"] = dt.dt.month

    if "cutoff_local_minute" not in df.columns:
        cutoff_local = pd.to_datetime(df["cutoff_utc"], utc=True).dt.tz_convert(LOCAL_TZ)
        df["cutoff_local_minute"] = cutoff_local.dt.hour * 60 + cutoff_local.dt.minute

    # Heat gap features (n_x and tmp-based)
    df["gap_to_mos_x"] = df["mos_x_mean"] - df["max_sofar"]
    df["gap_frac"] = df["gap_to_mos_x"] / df["mos_range"].clip(lower=1.0)

    df["heat_gap"] = df["mos_tmax_mean"] - df["max_sofar"]
    df["heat_gap_norm"] = df["heat_gap"] / df["mos_range_mean"].clip(lower=1.0)
    df["completion_frac"] = (df["max_sofar"] - df["tmin_sofar"]) / df["mos_range_mean"].clip(lower=1.0)

    # Interaction features
    if "sb_stabilized_flag" in df.columns:
        df["sb_stabilized_x_heat_gap_norm"] = df["sb_stabilized_flag"] * df["heat_gap_norm"]
    if "sb_slope_post" in df.columns and "mos_cig_min_models" in df.columns:
        df["sb_slope_post_x_cig"] = df["sb_slope_post"] * df["mos_cig_min_models"]
    df["heat_gap_norm_x_slope_60"] = df["heat_gap_norm"] * df["slope_60m"]
    if "mos_cig_min_models" in df.columns:
        df["cig_min_x_max_sofar"] = df["mos_cig_min_models"] * df["max_sofar"]

    # MOS age controls (tmp)
    if "mos_tmp_asof_utc_gfs" in df.columns:
        df["mos_age_hours_tmp_gfs"] = (df["cutoff_utc"] - df["mos_tmp_asof_utc_gfs"]).dt.total_seconds() / 3600.0
    if "mos_tmp_asof_utc_nam" in df.columns:
        df["mos_age_hours_tmp_nam"] = (df["cutoff_utc"] - df["mos_tmp_asof_utc_nam"]).dt.total_seconds() / 3600.0

    # MOS time-block mismatches vs obs temp now
    for code in ["tmp", "t06", "t06_1", "t06_2", "t12", "t12_1", "t12_2"]:
        mean_col = f"mos_{code}_mean_models"
        if mean_col in df.columns:
            df[f"mis_{code}_mean"] = df["temp_now"] - df[mean_col]

    # Nowcast mismatch block
    if "mos_t12_mean_models" in df.columns:
        df["mismatch_12z"] = df["obs_at_12z"] - df["mos_t12_mean_models"]
    if "mos_t12_2_mean_models" in df.columns:
        df["mismatch_15z"] = df["obs_at_15z"] - df["mos_t12_2_mean_models"]
    elif "mos_t12_mean_models" in df.columns:
        df["mismatch_15z"] = df["obs_at_15z"] - df["mos_t12_mean_models"]
    if "mismatch_12z" in df.columns and "mismatch_15z" in df.columns:
        df["mismatch_trend"] = df["mismatch_15z"] - df["mismatch_12z"]

    # QPF category shaping (q12 bins)
    if "mos_q12_max_models" in df.columns:
        for thr in [1, 2, 3, 4, 5]:
            df[f"q12_ge_{thr}"] = (df["mos_q12_max_models"] >= thr).astype(float)

    # Delta future for regression targets
    df["delta_future"] = df["tmax_full"] - df["tmax_sofar"]
    df["log_delta_future"] = np.log(df["delta_future"].clip(lower=0) + 0.01)
    df["delta_bin"] = pd.cut(
        df["delta_future"],
        bins=[-np.inf, 0.01, 0.5, 1.5, np.inf],
        labels=[0, 1, 2, 3],
    ).astype(float)

    # Revision flags
    if "rev_tmp_value_max_mean_models" in df.columns:
        df["warm_revision_flag"] = (df["rev_tmp_value_max_mean_models"] >= 1.0).astype(float)
    if "rev_q12_value_max_mean_models" in df.columns:
        df["convective_revision_flag"] = (df["rev_q12_value_max_mean_models"] >= 0.05).astype(float)
    if "rev_cig_value_min_mean_models" in df.columns:
        df["suppression_revision_flag"] = (df["rev_cig_value_min_mean_models"] <= -0.1).astype(float)

    # Train/val/test masks
    train_mask, val_mask, _ = _prepare_splits(df)

    # Add climo range and heating fraction (train-only)
    df = _add_climo_range_features(df, train_mask)

    # Bias-correct MOS X (past-only EWMA)
    df = _add_bias_features(df)
    if "heat_gap_bc_norm" in df.columns and "mos_cig_min_models" in df.columns:
        df["heat_gap_bc_norm_x_cig"] = df["heat_gap_bc_norm"] * df["mos_cig_min_models"]
    if "heat_gap_bc_norm" in df.columns and "mos_q12_max_models" in df.columns:
        df["heat_gap_bc_norm_x_q12"] = df["heat_gap_bc_norm"] * df["mos_q12_max_models"]

    # Bias-correct nowcast mismatch
    df = _add_nowcast_bias_features(df)

    # Suppression index (train-only zscore)
    df = _add_suppression_index(df, train_mask)

    # MOS probability calibration features (train-only)
    df = _add_calibrated_mos_probs(df, train_mask)

    # Regime bins based on val thresholds
    df = _add_regime_bins(df, val_mask)

    # Prior features (expanding window)
    df = _add_prior_features(df)

    # Analog kNN features
    df = _add_knn_features(df, train_mask, KNN_K)

    # p_onshore gate (optional)
    df = _add_onshore_probability(df, train_mask)

    # Front/cold-advection flag
    df = _add_front_flag(df)

    _write_cache(cache_path, meta_path, df, cache_hash)
    return df

def _add_climo_range_features(df: pd.DataFrame, train_mask: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    train = df.loc[train_mask]
    range_climo = train.groupby("doy")["range_full"].median()
    overall = train["range_full"].median()
    df["range_climo_doy"] = df["doy"].map(range_climo).fillna(overall)
    df["heating_fraction_obs"] = df["range_sofar"] / df["range_climo_doy"].clip(lower=1.0)
    return df


def _add_bias_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("target_date_local")
    err = (df["tmax_full"] - df["mos_x_mean"]).shift(1)
    bias = err.ewm(span=45, adjust=False).mean()
    bias = bias.fillna(0.0)
    df["bias_x_ewma_45"] = bias
    df["bias_x_ewma_abs"] = bias.abs()
    df["ewma_var_45"] = err.rolling(window=45, min_periods=5).var().shift(1)
    df["mos_x_bc"] = df["mos_x_mean"] + df["bias_x_ewma_45"]
    df["heat_gap_bc"] = df["mos_x_bc"] - df["max_sofar"]
    df["heat_gap_bc_norm"] = df["heat_gap_bc"] / df["mos_range"].clip(lower=1.0)
    df["gap_to_mos_x_bc"] = df["heat_gap_bc"]
    return df


def _add_calibrated_mos_probs(df: pd.DataFrame, train_mask: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    prob_vars = [
        "mos_p06_max_models",
        "mos_p12_max_models",
        "mos_t06_max_models",
        "mos_t12_max_models",
        "mos_t06_1_max_models",
        "mos_t06_2_max_models",
        "mos_t12_1_max_models",
        "mos_t12_2_max_models",
    ]
    for var in prob_vars:
        if var not in df.columns:
            continue
        x_train = df.loc[train_mask, var]
        y_train = df.loc[train_mask, "y_hit_by_cutoff"]
        mask = x_train.notna()
        if mask.sum() < 50:
            df[f"{var}_cal"] = df[var]
            continue
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(x_train[mask], y_train[mask])
        df[f"{var}_cal"] = iso.transform(df[var].fillna(x_train[mask].median()))
    return df


def _add_nowcast_bias_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "mismatch_12z" not in df.columns:
        return df
    df = df.sort_values("target_date_local")
    err = (df["mismatch_12z"] - df["delta_future"]).shift(1)
    bias = err.ewm(span=45, adjust=False).mean()
    bias = bias.fillna(0.0)
    df["bias_mismatch_12z"] = bias
    df["corr_mismatch_12z"] = df["mismatch_12z"] - df["bias_mismatch_12z"]
    return df


def _add_suppression_index(df: pd.DataFrame, train_mask: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    if "mos_cig_min_models" not in df.columns or "mos_q12_max_models" not in df.columns:
        df["suppression_index"] = np.nan
        return df
    train = df.loc[train_mask]
    cig_mean = train["mos_cig_min_models"].mean()
    cig_std = train["mos_cig_min_models"].std()
    vis_mean = train["mos_vis_min_models"].mean() if "mos_vis_min_models" in df.columns else np.nan
    vis_std = train["mos_vis_min_models"].std() if "mos_vis_min_models" in df.columns else np.nan
    q12_mean = train["mos_q12_max_models"].mean()
    q12_std = train["mos_q12_max_models"].std()

    cig_std = cig_std if np.isfinite(cig_std) and cig_std > 0 else 1.0
    q12_std = q12_std if np.isfinite(q12_std) and q12_std > 0 else 1.0
    vis_std = vis_std if np.isfinite(vis_std) and vis_std > 0 else 1.0

    z_cig_low = (cig_mean - df["mos_cig_min_models"]) / cig_std
    z_q12_high = (df["mos_q12_max_models"] - q12_mean) / q12_std
    if "mos_vis_min_models" in df.columns:
        z_vis_low = (vis_mean - df["mos_vis_min_models"]) / vis_std
    else:
        z_vis_low = 0.0
    df["suppression_index"] = z_cig_low + z_vis_low + z_q12_high
    return df


def _add_regime_bins(df: pd.DataFrame, val_mask: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    wdr = df.get("mos_wdr_mean_models")
    if wdr is None:
        df["wind_bin"] = "unknown"
        df["onshore_flag"] = 0.0
        df["offshore_flag"] = 0.0
    else:
        onshore = (wdr >= 45) & (wdr <= 160)
        offshore = (wdr >= 225) & (wdr <= 315)
        wind_bin = np.where(onshore, "onshore", np.where(offshore, "offshore", "cross"))
        df["wind_bin"] = wind_bin
        df["onshore_flag"] = onshore.astype(float)
        df["offshore_flag"] = offshore.astype(float)

    # suppression bins using val thresholds
    if "mos_cig_min_models" in df.columns and "mos_q12_max_models" in df.columns and val_mask.any():
        cig = df.loc[val_mask, "mos_cig_min_models"].dropna()
        q12 = df.loc[val_mask, "mos_q12_max_models"].dropna()
        cig_q33 = cig.quantile(0.33) if not cig.empty else np.nan
        cig_q66 = cig.quantile(0.66) if not cig.empty else np.nan
        q12_q33 = q12.quantile(0.33) if not q12.empty else np.nan
        q12_q66 = q12.quantile(0.66) if not q12.empty else np.nan
        suppressed = (df["mos_cig_min_models"] <= cig_q33) | (df["mos_q12_max_models"] >= q12_q66)
        clear = (df["mos_cig_min_models"] >= cig_q66) & (df["mos_q12_max_models"] <= q12_q33)
        df["suppression_bin"] = np.where(suppressed, "suppressed", np.where(clear, "clear", "mixed"))
    else:
        df["suppression_bin"] = "unknown"

    # moisture bins
    if "mos_dpt_mean_models" in df.columns and val_mask.any():
        dpt = df.loc[val_mask, "mos_dpt_mean_models"].dropna()
        dpt_q33 = dpt.quantile(0.33) if not dpt.empty else np.nan
        dpt_q66 = dpt.quantile(0.66) if not dpt.empty else np.nan
        df["moisture_bin"] = np.where(
            df["mos_dpt_mean_models"] >= dpt_q66,
            "humid",
            np.where(df["mos_dpt_mean_models"] <= dpt_q33, "dry", "moderate"),
        )
    else:
        df["moisture_bin"] = "unknown"

    # season bins
    month = df["month"]
    df["season_bin"] = np.where(
        month.isin([12, 1, 2]),
        "DJF",
        np.where(month.isin([3, 4, 5]), "MAM", np.where(month.isin([6, 7, 8]), "JJA", "SON")),
    )
    wind_map = {"unknown": 0, "onshore": 1, "offshore": 2, "cross": 3}
    supp_map = {"unknown": 0, "suppressed": 1, "mixed": 2, "clear": 3}
    moist_map = {"unknown": 0, "humid": 1, "moderate": 2, "dry": 3}
    season_map = {"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3}
    df["wind_bin_code"] = df["wind_bin"].map(wind_map).fillna(0).astype(float)
    df["suppression_bin_code"] = df["suppression_bin"].map(supp_map).fillna(0).astype(float)
    df["moisture_bin_code"] = df["moisture_bin"].map(moist_map).fillna(0).astype(float)
    df["season_bin_code"] = df["season_bin"].map(season_map).fillna(0).astype(float)
    return df


def _add_prior_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("target_date_local")
    prior = []
    counts = {}
    total_yes = 0
    total_cnt = 0
    k = 1.0
    for _, row in df.iterrows():
        key = f"{row.get('season_bin','')}_{row.get('wind_bin','')}_{row.get('suppression_bin','')}"
        if total_cnt == 0:
            base_prior = 0.5
        else:
            base_prior = (total_yes + k) / (total_cnt + 2 * k)
        if key in counts:
            yes, cnt = counts[key]
            p = (yes + k) / (cnt + 2 * k)
        else:
            p = base_prior
        prior.append(p)
        # update counts after computing prior
        y = row.get("y_hit_by_cutoff")
        if pd.notna(y):
            total_yes += int(y)
            total_cnt += 1
            if key not in counts:
                counts[key] = [0, 0]
            counts[key][0] += int(y)
            counts[key][1] += 1

    df["prior_p_yes"] = prior
    df["logit_prior"] = np.log(np.clip(df["prior_p_yes"], 1e-4, 1 - 1e-4) / (1 - np.clip(df["prior_p_yes"], 1e-4, 1 - 1e-4)))
    return df


def _add_onshore_probability(df: pd.DataFrame, train_mask: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    if "mos_wdr_mean_models" not in df.columns or "mos_wsp_mean_models" not in df.columns:
        df["p_onshore"] = np.nan
        return df
    wdr = df["mos_wdr_mean_models"]
    wsp = df["mos_wsp_mean_models"]
    onshore_label = ((wdr >= 45) & (wdr <= 160) & (wsp >= 5)).astype(int)
    X = pd.DataFrame({
        "wdr_sin": np.sin(np.deg2rad(wdr)),
        "wdr_cos": np.cos(np.deg2rad(wdr)),
        "wsp": wsp,
        "doy_sin": df["doy_sin"],
        "doy_cos": df["doy_cos"],
    })
    imputer = SimpleImputer(strategy="median")
    X_all = imputer.fit_transform(X)
    X_train = X_all[train_mask]
    y_train = onshore_label[train_mask]
    if y_train.sum() < 50 or y_train.sum() == len(y_train):
        df["p_onshore"] = onshore_label.astype(float)
        return df
    lr = LogisticRegression(max_iter=200)
    lr.fit(X_train, y_train)
    df["p_onshore"] = lr.predict_proba(X_all)[:, 1]
    return df


def _add_front_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "mos_wdr_mean_models" not in df.columns or "mos_wsp_mean_models" not in df.columns:
        df["front_flag"] = 0.0
        return df
    wdr = df["mos_wdr_mean_models"]
    wsp = df["mos_wsp_mean_models"]
    northerly = (wdr >= 315) | (wdr <= 45)
    strong_wind = wsp >= 10
    cooling = (df["slope_60m"] <= 0) | (df["drop_from_max"] >= 0.3)
    df["front_flag"] = (northerly & strong_wind & cooling).astype(float)
    return df


def _add_climo_features(df: pd.DataFrame, train_mask: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    train = df.loc[train_mask].copy()

    climo_p_yes = train.groupby("doy")["y_hit_by_cutoff"].mean()
    climo_tmax = train.groupby("doy")["tmax_full"].median()
    climo_peak_time = train.groupby("doy")["tmax_time_local_minute"].median()
    climo_extra = train.groupby("doy")["delta_future"].median()

    overall_p_yes = train["y_hit_by_cutoff"].mean()
    overall_tmax = train["tmax_full"].median()
    overall_peak_time = train["tmax_time_local_minute"].median()
    overall_extra = train["delta_future"].median()

    df["prior_p_yes"] = df["doy"].map(climo_p_yes).fillna(overall_p_yes)
    df["climo_tmax_median_doy"] = df["doy"].map(climo_tmax).fillna(overall_tmax)
    df["climo_peak_time_median_doy"] = df["doy"].map(climo_peak_time).fillna(overall_peak_time)
    df["climo_extra_warm_after_cutoff_median_doy"] = df["doy"].map(climo_extra).fillna(overall_extra)

    df["max_sofar_minus_climo_tmax"] = df["max_sofar"] - df["climo_tmax_median_doy"]
    df["minutes_since_max_vs_climo_peak_time"] = df["tmax_sofar_time_local_minute"] - df["climo_peak_time_median_doy"]
    df["gap_to_climo_extra"] = df["climo_extra_warm_after_cutoff_median_doy"]
    return df


def _add_knn_features(df: pd.DataFrame, train_mask: np.ndarray, k: int) -> pd.DataFrame:
    df = df.copy()
    df["_orig_idx"] = np.arange(len(df))
    knn_cols = [f"dct_norm_{i}" for i in range(21)] + [
        "slope_60m",
        "slope_180m",
        "drop_from_max",
        "plateau_frac_0p2_last120",
        "heat_gap_norm",
        "mos_q12_max_models",
        "mos_cig_min_models",
    ]
    knn_cols = [c for c in knn_cols if c in df.columns]
    if not knn_cols:
        df = df.drop(columns=["_orig_idx"], errors="ignore")
        return df

    df_sorted = df.sort_values("target_date_local").reset_index(drop=True)
    imputer = SimpleImputer(strategy="median")
    X_all = imputer.fit_transform(df_sorted[knn_cols])
    dates = pd.to_datetime(df_sorted["target_date_local"])
    months = dates.dt.month.to_numpy()
    y = df_sorted["y_hit_by_cutoff"].to_numpy()
    delta_future = df_sorted["delta_future"].to_numpy()
    minutes_since = df_sorted["minutes_since_max"].to_numpy()

    k_eff = min(k + 50, len(df))
    nn = NearestNeighbors(n_neighbors=k_eff, metric="cosine")
    nn.fit(X_all)
    _, indices = nn.kneighbors(X_all, n_neighbors=k_eff)

    knn_p_yes = np.full(len(df_sorted), np.nan)
    knn_mean_delta = np.full(len(df_sorted), np.nan)
    knn_mean_minutes = np.full(len(df_sorted), np.nan)

    for i in range(len(df_sorted)):
        cand = indices[i]
        # past-only + month band ±1
        mask = dates.iloc[cand] < dates.iloc[i]
        m = months[i]
        month_diff = np.abs(months[cand] - m)
        month_diff = np.minimum(month_diff, 12 - month_diff)
        mask = mask & (month_diff <= 1)
        cand = cand[mask]
        if cand.size == 0:
            continue
        neigh = cand[:k]
        knn_p_yes[i] = float(np.nanmean(y[neigh]))
        knn_mean_delta[i] = float(np.nanmean(delta_future[neigh]))
        knn_mean_minutes[i] = float(np.nanmean(minutes_since[neigh]))

    df_sorted["knn_p_yes"] = knn_p_yes
    df_sorted["knn_mean_delta_future"] = knn_mean_delta
    df_sorted["knn_mean_minutes_since_max"] = knn_mean_minutes

    df = df.merge(df_sorted[["_orig_idx", "knn_p_yes", "knn_mean_delta_future", "knn_mean_minutes_since_max"]], on="_orig_idx", how="left")
    df = df.drop(columns=["_orig_idx"], errors="ignore")
    return df

def _fit_isotonic(y_val: np.ndarray, p_val: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val, y_val)
    return iso


def _clip_probs(p: np.ndarray) -> np.ndarray:
    return np.clip(p, 1e-6, 1 - 1e-6)


def _fit_beta_calibrator(y: np.ndarray, p: np.ndarray) -> Tuple[LogisticRegression, callable]:
    p = _clip_probs(p)
    X = np.column_stack([np.log(p), np.log1p(-p)])
    lr = LogisticRegression(max_iter=200)
    lr.fit(X, y)

    def transform(p_new: np.ndarray) -> np.ndarray:
        p_new = _clip_probs(p_new)
        X_new = np.column_stack([np.log(p_new), np.log1p(-p_new)])
        return lr.predict_proba(X_new)[:, 1]

    return lr, transform


def _fit_platt_calibrator(y: np.ndarray, p: np.ndarray) -> Tuple[LogisticRegression, callable]:
    p = _clip_probs(p)
    logit = np.log(p / (1 - p))
    lr = LogisticRegression(max_iter=200)
    lr.fit(logit.reshape(-1, 1), y)

    def transform(p_new: np.ndarray) -> np.ndarray:
        p_new = _clip_probs(p_new)
        logit_new = np.log(p_new / (1 - p_new))
        return lr.predict_proba(logit_new.reshape(-1, 1))[:, 1]

    return lr, transform


def _fit_regime_beta(y: np.ndarray, p: np.ndarray, regime_keys: np.ndarray, min_count: int = 200, shrink: float = 0.3) -> Tuple[Dict, callable]:
    _, global_transform = _fit_beta_calibrator(y, p)
    calibrators = {}
    for key in np.unique(regime_keys):
        mask = regime_keys == key
        if mask.sum() < min_count:
            continue
        _, transform = _fit_beta_calibrator(y[mask], p[mask])
        calibrators[key] = transform

    def transform(p_new: np.ndarray, regime_new: np.ndarray) -> np.ndarray:
        p_global = global_transform(p_new)
        out = p_global.copy()
        for key, cal in calibrators.items():
            mask = regime_new == key
            if mask.any():
                p_bin = cal(p_new[mask])
                out[mask] = (1 - shrink) * p_bin + shrink * p_global[mask]
        return out

    return calibrators, transform


def _reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> Tuple[List[Dict[str, float]], float]:
    p = np.clip(p, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    ece = 0.0
    n = len(y)
    for i in range(bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if mask.sum() == 0:
            rows.append({"bin": i, "count": 0, "pred": float("nan"), "obs": float("nan")})
            continue
        pred = float(np.mean(p[mask]))
        obs = float(np.mean(y[mask]))
        rows.append({"bin": i, "count": int(mask.sum()), "pred": pred, "obs": obs})
        ece += abs(pred - obs) * mask.sum() / max(n, 1)
    return rows, float(ece)


def _trade_summary(y: np.ndarray, p: np.ndarray, price: float, ev_min: float) -> Dict[str, float]:
    ev = p - price
    trade = ev >= ev_min
    tp = int(np.sum(trade & (y == 1)))
    fp = int(np.sum(trade & (y == 0)))
    n = len(y)
    n_trades = tp + fp
    trade_rate = n_trades / n if n > 0 else float("nan")
    net_units = tp - fp
    net_units_per_100 = net_units / n * 100.0 if n > 0 else float("nan")
    net_per_trade = net_units / n_trades if n_trades > 0 else float("nan")
    mean_ev = float(np.mean(ev[trade])) if n_trades > 0 else float("nan")
    pnl_per_contract = ((tp * (1 - price) - fp * price) / n_trades) if n_trades > 0 else float("nan")
    ev_quantiles = [float(q) for q in np.quantile(ev[trade], [0.1, 0.5, 0.9])] if n_trades > 0 else [float("nan")] * 3
    return {
        "price": price,
        "ev_min": ev_min,
        "tp": tp,
        "fp": fp,
        "trades": n_trades,
        "trade_rate": trade_rate,
        "net_units_per_100_days": net_units_per_100,
        "net_per_trade": net_per_trade,
        "mean_ev": mean_ev,
        "pnl_per_contract": pnl_per_contract,
        "ev_q10": ev_quantiles[0],
        "ev_q50": ev_quantiles[1],
        "ev_q90": ev_quantiles[2],
    }


def _group_trade_breakdown(df: pd.DataFrame, p: np.ndarray, y: np.ndarray, group_col: str, prices: List[float], ev_min: float) -> Dict[str, Dict[str, Dict[str, float]]]:
    breakdown = {}
    if group_col not in df.columns:
        return breakdown
    groups = pd.unique(df[group_col])
    for g in groups:
        mask = df[group_col] == g
        if mask.sum() == 0:
            continue
        entry = {}
        for price in prices:
            entry[f"price_{price}"] = _trade_summary(y[mask], p[mask], price, ev_min)
        breakdown[str(g)] = entry
    return breakdown


def _threshold_sweep(y: np.ndarray, p: np.ndarray, thresholds: List[float]) -> List[Dict[str, float]]:
    rows = []
    for thr in thresholds:
        y_hat = (p >= thr).astype(int)
        cm = confusion_matrix(y, y_hat, labels=[0, 1])
        tp = int(cm[1, 1])
        fp = int(cm[0, 1])
        n = len(y)
        net_units = tp - fp
        rows.append({
            "threshold": float(thr),
            "tp": tp,
            "fp": fp,
            "trade_rate": float(np.mean(y_hat)),
            "net_units_per_100_days": net_units / n * 100.0 if n > 0 else float("nan"),
        })
    return rows


def _feature_signature(feature_cols: List[str], extra: Dict) -> str:
    payload = {
        "features": sorted(feature_cols),
        "extra": extra,
    }
    return _sha256_str(json.dumps(payload, sort_keys=True))


def _dataset_signature(df: pd.DataFrame, train_mask: np.ndarray, val_mask: np.ndarray, test_mask: np.ndarray) -> str:
    payload = {
        "train_dates": [str(d) for d in df.loc[train_mask, "target_date_local"]],
        "val_dates": [str(d) for d in df.loc[val_mask, "target_date_local"]],
        "test_dates": [str(d) for d in df.loc[test_mask, "target_date_local"]],
        "label_version": "hit1830_eps0p01",
        "epsilon": EPS,
    }
    return _sha256_str(json.dumps(payload, sort_keys=True))


def _calibration_signature(cal_type: str, extra: Dict) -> str:
    payload = {"calibrator": cal_type, "fit_range": "VAL_2020_2022", "extra": extra}
    return _sha256_str(json.dumps(payload, sort_keys=True))


def _perm_importance(model: lgb.Booster, X_val: np.ndarray, y_val: np.ndarray, feature_names: List[str], top_n: int = 20) -> List[Dict[str, float]]:
    if model is None:
        return []
    gains = model.feature_importance(importance_type="gain")
    order = np.argsort(gains)[::-1]
    top_idx = order[: min(top_n, len(order))]
    base_pred = model.predict(X_val)
    try:
        base_auc = roc_auc_score(y_val, base_pred)
    except ValueError:
        base_auc = float("nan")
    rows = []
    rng = np.random.default_rng(42)
    for idx in top_idx:
        X_perm = X_val.copy()
        rng.shuffle(X_perm[:, idx])
        try:
            auc = roc_auc_score(y_val, model.predict(X_perm))
            imp = base_auc - auc
        except ValueError:
            imp = float("nan")
        rows.append({"feature": feature_names[idx], "gain": float(gains[idx]), "perm_importance": float(imp)})
    return rows


def _build_monotone_constraints(feature_cols: List[str]) -> List[int]:
    constraint_map = {
        "minutes_since_max": 1,
        "drop_from_max": 1,
        "plateau_frac_0p2_last120": 1,
        "heat_gap": -1,
        "heat_gap_norm": -1,
        "slope_60m": -1,
    }
    return [constraint_map.get(col, 0) for col in feature_cols]


def _train_lgbm_classifier(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, params: Dict, sample_weight: np.ndarray | None = None) -> lgb.Booster:
    train_set = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    model = lgb.train(
        params,
        train_set,
        valid_sets=[val_set],
        num_boost_round=2000,
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def _train_lgbm_regressor(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, params: Dict, sample_weight: np.ndarray | None = None) -> lgb.Booster:
    params = params.copy()
    params["objective"] = "regression"
    train_set = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    model = lgb.train(
        params,
        train_set,
        valid_sets=[val_set],
        num_boost_round=2000,
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def _train_lgbm_multiclass(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, params: Dict, num_class: int) -> lgb.Booster:
    params = params.copy()
    params["objective"] = "multiclass"
    params["num_class"] = num_class
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    model = lgb.train(
        params,
        train_set,
        valid_sets=[val_set],
        num_boost_round=2000,
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def _prepare_splits(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    years = pd.to_datetime(df["target_date_local"]).dt.year
    train_mask = years <= 2019
    val_mask = (years >= 2020) & (years <= 2022)
    test_mask = (years >= 2023) & (years <= 2025)
    return train_mask.to_numpy(), val_mask.to_numpy(), test_mask.to_numpy()


def _save_experiment(out_dir: Path, name: str, feature_list: List[str],
                     preds_val: pd.DataFrame, preds_test: pd.DataFrame,
                     metrics: Dict, model: lgb.Booster | None, extra_files: Dict | None = None) -> None:
    exp_dir = out_dir / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    preds_val.to_parquet(exp_dir / "preds_val.parquet", index=False)
    preds_test.to_parquet(exp_dir / "preds_test.parquet", index=False)

    (exp_dir / "features.json").write_text(json.dumps(feature_list, indent=2), encoding="utf-8")

    if model is not None:
        model.save_model(str(exp_dir / "model.txt"))

    if extra_files:
        for fname, content in extra_files.items():
            (exp_dir / fname).write_text(content, encoding="utf-8")

    (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _build_experiment_features(df: pd.DataFrame) -> Dict[str, List[str]]:
    dct_curve = [f"dct_curve_{i}" for i in range(20)]
    dct_norm = [f"dct_norm_{i}" for i in range(21)]
    slope_bins_6h = [f"slope6h_bin{i}" for i in range(7)]
    slope_bins_full = [f"slopefull_bin{i}" for i in range(7)]

    minute_core = [
        "temp_now",
        "max_sofar",
        "tmin_sofar",
        "minutes_since_max",
        "drop_from_max",
        "slope_15m",
        "slope_30m",
        "slope_60m",
        "slope_120m",
        "slope_180m",
        "curvature_30_180",
        "curvature_60_180",
        "plateau_frac_0p1_last120",
        "plateau_frac_0p2_last120",
        "plateau_frac_0p3_last120",
        "plateau_longest_run_0p1_last120",
        "plateau_longest_run_0p2_last120",
        "plateau_longest_run_0p3_last120",
        "drop_cnt_0p5_last6h",
        "max_drop_30m_last6h",
        "rebound_after_drop",
        "std_30m",
        "std_60m",
        "std_180",
        "mad_30m",
        "mad_60m",
        "mad_180",
        "mean_abs_delta_60m",
        "hi_freq_energy_ratio",
        "frac_bins_present_total",
        "max_gap_minutes_total",
    ]

    minute_core_full = minute_core + dct_curve

    minute_peak_conf = [
        "warming_fraction_last180",
        "high_end_persistence_min",
        "early_day_heat_integral",
    ]

    minute_qa = [
        "frac_bins_present_last180",
        "max_gap_minutes_last180",
        "gap_cnt_ge15m",
        "gap_cnt_ge30m",
        "spike_cnt",
        "max_sofar_is_spike_flag",
    ]

    minute_sbfd = [
        "sb_break_time_local_min",
        "sb_slope_pre",
        "sb_slope_post",
        "sb_slope_drop",
        "sb_plateau_after",
        "sb_cooling_pulse",
        "sb_stabilized_flag",
        "sbfd_missing",
        "sb_stabilized_x_heat_gap_norm",
        "sb_slope_post_x_cig",
    ]

    minute_hist = slope_bins_6h + slope_bins_full + ["convective_crash_cnt", "crash_recovery_ratio"]
    minute_outflow = [
        "outflow_cnt",
        "outflow_max_crash",
        "outflow_mean_crash",
        "outflow_min_recovery_frac",
        "outflow_last_event_age_min",
        "crash_energy",
    ]

    mos_core_tmp = [
        "mos_tmax_mean",
        "mos_tmin_mean",
        "mos_range_mean",
        "mos_tmax_disagree",
        "mos_range_disagree_tmp",
        "heat_gap",
        "heat_gap_norm",
        "completion_frac",
        "mos_age_hours_tmp_gfs",
        "mos_age_hours_tmp_nam",
    ]

    mos_core_nx = [
        "mos_x_mean",
        "mos_n_mean",
        "mos_range",
        "mos_x_disagree",
        "mos_range_disagree",
        "gap_to_mos_x",
        "gap_frac",
    ]

    mos_convective = [
        "mos_p06_max_models",
        "mos_p12_max_models",
        "mos_q06_max_models",
        "mos_q12_max_models",
        "mos_t06_max_models",
        "mos_t12_max_models",
        "mos_t06_1_max_models",
        "mos_t06_2_max_models",
        "mos_t12_1_max_models",
        "mos_t12_2_max_models",
        "mos_cig_min_models",
        "mos_vis_min_models",
    ]

    mos_wind = ["mos_wdr_mean_models", "mos_wsp_mean_models", "mos_u_mean", "mos_v_mean"]
    mos_dpt = ["mos_dpt_mean_models"]

    regime_core = [
        "doy_sin",
        "doy_cos",
        "month",
        "season_bin_code",
        "wind_bin_code",
        "suppression_bin_code",
        "moisture_bin_code",
        "suppression_index",
        "onshore_flag",
        "offshore_flag",
    ]

    mismatch_cols = [
        "mismatch_12z",
        "mismatch_15z",
        "mismatch_trend",
        "corr_mismatch_12z",
        "bias_mismatch_12z",
    ]

    bias_cols = [
        "bias_x_ewma_45",
        "bias_x_ewma_abs",
        "ewma_var_45",
        "mos_x_bc",
        "heat_gap_bc",
        "heat_gap_bc_norm",
        "heat_gap_bc_norm_x_cig",
        "heat_gap_bc_norm_x_q12",
    ]

    interaction_cols = [
        "heat_gap_norm_x_slope_60",
        "cig_min_x_max_sofar",
    ]

    analog_cols = [
        "knn_p_yes",
        "knn_mean_delta_future",
        "knn_mean_minutes_since_max",
    ]

    prior_cols = ["prior_p_yes", "logit_prior"]

    rev_cols = [c for c in df.columns if c.startswith("rev_")]
    rev_flags = ["warm_revision_flag", "convective_revision_flag", "suppression_revision_flag"]

    feature_sets = {}

    feature_sets["EXP01_Fusion_SBFD"] = minute_core_full + minute_sbfd + mos_core_tmp + mos_core_nx + mos_convective + mos_wind + mos_dpt + regime_core

    feature_sets["EXP02_Fusion_REVSHOCK"] = minute_core_full + mos_core_tmp + mos_core_nx + mos_convective + mos_wind + mos_dpt + regime_core + rev_cols + rev_flags

    feature_sets["EXP03_Fusion_BC_GAP"] = minute_core_full + mos_core_tmp + mos_convective + mos_wind + mos_dpt + regime_core + bias_cols

    feature_sets["EXP04_Fusion_REG_BETA_CAL"] = minute_core_full + mos_core_tmp + mos_core_nx + mos_convective + mos_wind + mos_dpt + regime_core

    feature_sets["EXP05_Fusion_EV_WEIGHTED"] = minute_core_full + mos_core_tmp + mos_core_nx + mos_convective + mos_wind + mos_dpt + regime_core + interaction_cols

    feature_sets["EXP06_Fusion_ANALOG_PRIOR"] = minute_core_full + mos_core_tmp + mos_core_nx + mos_convective + mos_wind + mos_dpt + regime_core + analog_cols

    feature_sets["EXP07_HAZARD_EXCEED"] = [
        "max_sofar",
        "temp_now",
        "minutes_since_max",
        "drop_from_max",
        "slope_15m",
        "slope_30m",
        "slope_60m",
        "slope_120m",
        "slope_180m",
        "warming_fraction_last180",
        "high_end_persistence_min",
        "early_day_heat_integral",
        "plateau_frac_0p2_last120",
        "drop_cnt_0p5_last6h",
    ] + mos_core_tmp + mos_convective + mos_dpt + mos_wind + regime_core

    feature_sets["EXP08_ZI_DELTA"] = minute_core_full + minute_peak_conf + mos_core_tmp + mos_convective + mos_wind + mos_dpt + regime_core

    feature_sets["EXP09_CNN2GBM"] = dct_norm + minute_core + minute_peak_conf + mos_core_tmp + mos_convective + mos_wind + mos_dpt + regime_core

    feature_sets["EXP10_CPC_EMBED"] = dct_curve + minute_core + minute_hist + mos_core_tmp + mos_convective + mos_wind + mos_dpt + regime_core

    feature_sets["EXP11_MOE_4REG"] = minute_core_full + mos_core_tmp + mos_convective + mos_wind + mos_dpt + regime_core

    feature_sets["EXP12_MONO_PHYS"] = [
        "minutes_since_max",
        "drop_from_max",
        "plateau_frac_0p2_last120",
        "slope_60m",
        "slope_180m",
        "heat_gap",
        "heat_gap_norm",
        "mos_tmax_mean",
        "mos_tmax_disagree",
        "mos_q12_max_models",
        "mos_p12_max_models",
        "mos_cig_min_models",
        "mos_vis_min_models",
        "doy_sin",
        "doy_cos",
    ]

    feature_sets["EXP13_SLOPE_HIST"] = minute_core + minute_hist + mos_core_tmp + mos_convective + mos_dpt + regime_core

    feature_sets["EXP14_DROP_REBOUND"] = minute_core + minute_outflow + mos_core_tmp + mos_convective + mos_dpt + regime_core

    feature_sets["EXP15_QA_AWARE"] = minute_core + minute_qa + mos_core_tmp + mos_convective + mos_dpt + regime_core

    feature_sets["EXP16_CTP_BAYES"] = minute_core + mos_core_tmp + mos_convective + prior_cols + regime_core

    feature_sets["EXP17_MOS_NOWCAST_MIS_v2"] = minute_core + mos_core_tmp + mos_convective + mismatch_cols + regime_core

    feature_sets["EXP18_EV_STACK_OOF"] = minute_core_full + mos_core_tmp + mos_convective + mos_wind + mos_dpt + regime_core

    feature_sets["EXP19_MT_DELTA_BINS"] = minute_core + mos_core_tmp + mos_convective + mos_dpt + regime_core

    feature_sets["EXP20_HB_PARAM_T"] = minute_core + minute_outflow + mos_core_tmp + mos_convective + mos_dpt + regime_core

    return feature_sets

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minute-dir", default=r"C:\Users\ahmad\Desktop\generalFiles\git\weather-forecasting-predictionmarkets\weather_forecating_predictionmarkets\data\iem_minute_data\MIA\tmpf\UTC\yearly")
    parser.add_argument("--out-dir", default=r"C:\Users\ahmad\Desktop\generalFiles\git\weather-forecasting-predictionmarkets\weather_forecating_predictionmarkets\artifacts\experiments\early_maxout_strategy\B5")
    parser.add_argument("--cache-dir", default=r"C:\Users\ahmad\Desktop\generalFiles\git\weather-forecasting-predictionmarkets\weather_forecating_predictionmarkets\cache")
    parser.add_argument("--reuse-cache", type=int, default=1)
    parser.add_argument("--rebuild-cache", type=int, default=0)
    args = parser.parse_args()

    minute_dir = Path(args.minute_dir)
    out_dir = Path(args.out_dir)
    cache_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = create_engine("mysql+pymysql://root:root@localhost/weather_predictionmarkets")

    mos_stamp = _mos_version_stamp(engine)
    minute_df = _build_minute_features(minute_dir, cache_dir, bool(args.reuse_cache), bool(args.rebuild_cache), mos_stamp)
    mos_df = _build_mos_features(cache_dir, bool(args.reuse_cache), bool(args.rebuild_cache), engine)

    cache_hash = _compute_cache_hash(minute_dir, mos_stamp)
    merged_df = _merge_dataset(minute_df, mos_df, cache_dir, bool(args.reuse_cache), bool(args.rebuild_cache), cache_hash)

    train_mask, val_mask, test_mask = _prepare_splits(merged_df)
    dataset_sig = _dataset_signature(merged_df, train_mask, val_mask, test_mask)

    y = merged_df["y_hit_by_cutoff"].to_numpy()
    y_exceed = merged_df["y_exceed_future"].to_numpy()
    delta_bin = merged_df["delta_bin"].to_numpy()

    always_no_acc = float(accuracy_score(y[test_mask], np.zeros_like(y[test_mask])))
    always_yes_acc = float(accuracy_score(y[test_mask], np.ones_like(y[test_mask])))

    feature_sets = _build_experiment_features(merged_df)

    def prepare_matrices(feature_cols: List[str], valid_mask: np.ndarray | None = None):
        cols = [c for c in feature_cols if c in merged_df.columns]
        if valid_mask is None:
            valid_mask = np.ones(len(merged_df), dtype=bool)
        train = train_mask & valid_mask
        val = val_mask & valid_mask
        test = test_mask & valid_mask
        cols = [c for c in cols if not merged_df.loc[train, c].isna().all()]
        X = merged_df[cols]
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X[train])
        X_val = imputer.transform(X[val])
        X_test = imputer.transform(X[test])
        return cols, X_train, X_val, X_test, train, val, test, imputer

    def evaluate_and_save(exp_name: str, feature_cols: List[str], p_val_raw: np.ndarray, p_test_raw: np.ndarray,
                          model: lgb.Booster | None, y_val: np.ndarray, y_test: np.ndarray,
                          val_mask_local: np.ndarray, test_mask_local: np.ndarray, X_val_local: np.ndarray | None,
                          cal_type: str, cal_extra: Dict, regime_keys: np.ndarray | None = None) -> Dict:
        if np.isnan(p_val_raw).any():
            p_val_raw = np.nan_to_num(p_val_raw, nan=float(np.nanmean(p_val_raw)))
        if np.isnan(p_test_raw).any():
            p_test_raw = np.nan_to_num(p_test_raw, nan=float(np.nanmean(p_val_raw)))

        if cal_type == "beta":
            _, cal_fn = _fit_beta_calibrator(y_val, p_val_raw)
            p_val_cal = cal_fn(p_val_raw)
            p_test_cal = cal_fn(p_test_raw)
        elif cal_type == "platt":
            _, cal_fn = _fit_platt_calibrator(y_val, p_val_raw)
            p_val_cal = cal_fn(p_val_raw)
            p_test_cal = cal_fn(p_test_raw)
        elif cal_type == "isotonic":
            iso = _fit_isotonic(y_val, p_val_raw)
            p_val_cal = iso.transform(p_val_raw)
            p_test_cal = iso.transform(p_test_raw)
        elif cal_type == "regime_beta" and regime_keys is not None:
            _, cal_fn = _fit_regime_beta(y_val, p_val_raw, regime_keys[val_mask_local])
            p_val_cal = cal_fn(p_val_raw, regime_keys[val_mask_local])
            p_test_cal = cal_fn(p_test_raw, regime_keys[test_mask_local])
        else:
            p_val_cal = p_val_raw
            p_test_cal = p_test_raw

        metrics_raw_val = _compute_metrics(y_val, p_val_raw, 0.5)
        metrics_raw_test = _compute_metrics(y_test, p_test_raw, 0.5)
        metrics_val = _compute_metrics(y_val, p_val_cal, 0.5)
        metrics_test = _compute_metrics(y_test, p_test_cal, 0.5)

        rel_val_raw, ece_val_raw = _reliability_table(y_val, p_val_raw)
        rel_val_cal, ece_val_cal = _reliability_table(y_val, p_val_cal)
        rel_test_raw, ece_test_raw = _reliability_table(y_test, p_test_raw)
        rel_test_cal, ece_test_cal = _reliability_table(y_test, p_test_cal)

        thresholds = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]
        sweep_val = _threshold_sweep(y_val, p_val_cal, thresholds)
        sweep_test = _threshold_sweep(y_test, p_test_cal, thresholds)

        prices = [0.50, 0.55, 0.60]
        ev_mins = [0.00, 0.03, 0.05]
        ev_summary_val = {}
        ev_summary_test = {}
        for price in prices:
            for ev_min in ev_mins:
                key = f"c{price}_ev{ev_min}"
                ev_summary_val[key] = _trade_summary(y_val, p_val_cal, price, ev_min)
                ev_summary_test[key] = _trade_summary(y_test, p_test_cal, price, ev_min)

        # breakdowns
        df_test = merged_df.loc[test_mask_local].copy()
        df_test["year"] = pd.to_datetime(df_test["target_date_local"]).dt.year
        df_test["season"] = df_test.get("season_bin")
        df_test["regime"] = df_test.get("wind_bin", "unknown").astype(str) + "_" + df_test.get("suppression_bin", "unknown").astype(str)
        breakdown_year = _group_trade_breakdown(df_test, p_test_cal, y_test, "year", prices, 0.03)
        breakdown_season = _group_trade_breakdown(df_test, p_test_cal, y_test, "season", prices, 0.03)
        breakdown_regime = _group_trade_breakdown(df_test, p_test_cal, y_test, "regime", prices, 0.03)

        feature_sig = _feature_signature(feature_cols, {"exp": exp_name, **cal_extra})
        calib_sig = _calibration_signature(cal_type, cal_extra)

        preds_val = merged_df.loc[val_mask_local, ["target_date_local", "cutoff_utc"]].copy()
        preds_val["y_true"] = y_val
        preds_val["p_raw"] = p_val_raw
        preds_val["p_cal"] = p_val_cal
        preds_val["y_pred_0p5"] = (p_val_cal >= 0.5).astype(int)

        preds_test = merged_df.loc[test_mask_local, ["target_date_local", "cutoff_utc"]].copy()
        preds_test["y_true"] = y_test
        preds_test["p_raw"] = p_test_raw
        preds_test["p_cal"] = p_test_cal
        preds_test["y_pred_0p5"] = (p_test_cal >= 0.5).astype(int)

        perm_imp = _perm_importance(model, X_val_local, y_val, feature_cols) if model is not None and X_val_local is not None else []

        metrics = {
            "exp_name": exp_name,
            "feature_sig": feature_sig,
            "dataset_sig": dataset_sig,
            "calibration_sig": calib_sig,
            "metrics_raw_val": metrics_raw_val,
            "metrics_raw_test": metrics_raw_test,
            "metrics_val": metrics_val,
            "metrics_test": metrics_test,
            "reliability_val_raw": rel_val_raw,
            "reliability_val_cal": rel_val_cal,
            "reliability_test_raw": rel_test_raw,
            "reliability_test_cal": rel_test_cal,
            "ece_val_raw": ece_val_raw,
            "ece_val_cal": ece_val_cal,
            "ece_test_raw": ece_test_raw,
            "ece_test_cal": ece_test_cal,
            "threshold_sweep_val": sweep_val,
            "threshold_sweep_test": sweep_test,
            "ev_summary_val": ev_summary_val,
            "ev_summary_test": ev_summary_test,
            "breakdown_year": breakdown_year,
            "breakdown_season": breakdown_season,
            "breakdown_regime": breakdown_regime,
            "always_no_acc": always_no_acc,
            "always_yes_acc": always_yes_acc,
            "permutation_importance": perm_imp,
        }

        _save_experiment(out_dir, exp_name, feature_cols, preds_val, preds_test, metrics, model)
        return metrics

    sig_seen = set()

    def check_sig(feature_cols: List[str], extra: Dict):
        sig = _feature_signature(feature_cols, extra)
        key = (sig, dataset_sig)
        if key in sig_seen:
            raise RuntimeError(f"Duplicate feature signature detected for {extra.get('exp')}")
        sig_seen.add(key)

    results = {}

    # Common params
    scale_pos_weight = (len(y[train_mask]) - y[train_mask].sum()) / max(y[train_mask].sum(), 1)
    lgb_params = {
        "objective": "binary",
        "boosting": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 80,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "max_bin": 255,
        "seed": 42,
        "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight,
    }

    # Regime keys for regime calibration
    regime_key = (merged_df.get("wind_bin", "unknown").astype(str) + "_" + merged_df.get("suppression_bin", "unknown").astype(str)).to_numpy()

    # EXP01 Fusion_SBFD
    exp = "EXP01_Fusion_SBFD"
    feats, X_train, X_val, X_test, tr, va, te, _ = prepare_matrices(feature_sets[exp])
    check_sig(feats, {"exp": exp})
    model = _train_lgbm_classifier(X_train, y[tr], X_val, y[va], lgb_params)
    p_val = model.predict(X_val)
    p_test = model.predict(X_test)
    results[exp] = evaluate_and_save(exp, feats, p_val, p_test, model, y[va], y[te], va, te, X_val, "beta", {"exp": exp}, regime_key)

    # EXP02 Fusion_REVSHOCK
    exp = "EXP02_Fusion_REVSHOCK"
    feats, X_train, X_val, X_test, tr, va, te, _ = prepare_matrices(feature_sets[exp])
    check_sig(feats, {"exp": exp})
    model = _train_lgbm_classifier(X_train, y[tr], X_val, y[va], lgb_params)
    p_val = model.predict(X_val)
    p_test = model.predict(X_test)
    results[exp] = evaluate_and_save(exp, feats, p_val, p_test, model, y[va], y[te], va, te, X_val, "platt", {"exp": exp}, regime_key)

    # EXP03 Fusion_BC_GAP
    exp = "EXP03_Fusion_BC_GAP"
    feats, X_train, X_val, X_test, tr, va, te, _ = prepare_matrices(feature_sets[exp])
    check_sig(feats, {"exp": exp})
    model = _train_lgbm_classifier(X_train, y[tr], X_val, y[va], lgb_params)
    p_val = model.predict(X_val)
    p_test = model.predict(X_test)
    results[exp] = evaluate_and_save(exp, feats, p_val, p_test, model, y[va], y[te], va, te, X_val, "beta", {"exp": exp}, regime_key)

    # EXP04 Regime beta calibration
    exp = "EXP04_Fusion_REG_BETA_CAL"
    feats, X_train, X_val, X_test, tr, va, te, _ = prepare_matrices(feature_sets[exp])
    check_sig(feats, {"exp": exp})
    model = _train_lgbm_classifier(X_train, y[tr], X_val, y[va], lgb_params)
    p_val = model.predict(X_val)
    p_test = model.predict(X_test)
    results[exp] = evaluate_and_save(exp, feats, p_val, p_test, model, y[va], y[te], va, te, X_val, "regime_beta", {"exp": exp}, regime_key)

    # EXP05 EV weighted
    exp = "EXP05_Fusion_EV_WEIGHTED"
    feats, X_train, X_val, X_test, tr, va, te, _ = prepare_matrices(feature_sets[exp])
    check_sig(feats, {"exp": exp})
    weight_grid = [1.1, 1.3, 1.6]
    best_model = None
    best_p_val = None
    best_score = -1e9
    for w in weight_grid:
        weights = np.where(y[tr] == 0, w, 1.0)
        model = _train_lgbm_classifier(X_train, y[tr], X_val, y[va], lgb_params, sample_weight=weights)
        p_val = model.predict(X_val)
        # score by net units at price 0.55
        summary = _trade_summary(y[va], p_val, 0.55, 0.03)
        if summary["net_units_per_100_days"] > best_score:
            best_score = summary["net_units_per_100_days"]
            best_model = model
            best_p_val = p_val
    p_test = best_model.predict(X_test)
    results[exp] = evaluate_and_save(exp, feats, best_p_val, p_test, best_model, y[va], y[te], va, te, X_val, "platt", {"exp": exp}, regime_key)

    # EXP06 Analog prior
    exp = "EXP06_Fusion_ANALOG_PRIOR"
    feats, X_train, X_val, X_test, tr, va, te, _ = prepare_matrices(feature_sets[exp])
    check_sig(feats, {"exp": exp})
    model = _train_lgbm_classifier(X_train, y[tr], X_val, y[va], lgb_params)
    p_val = model.predict(X_val)
    p_test = model.predict(X_test)
    results[exp] = evaluate_and_save(exp, feats, p_val, p_test, model, y[va], y[te], va, te, X_val, "beta", {"exp": exp}, regime_key)

    # EXP07 Hazard exceed
    exp = "EXP07_HAZARD_EXCEED"
    valid = np.isfinite(y_exceed)
    feats, X_train, X_val, X_test, tr, va, te, _ = prepare_matrices(feature_sets[exp], valid)
    check_sig(feats, {"exp": exp})
    model = _train_lgbm_classifier(X_train, y_exceed[tr], X_val, y_exceed[va], lgb_params)
    p_val_exceed = model.predict(X_val)
    p_test_exceed = model.predict(X_test)
    # calibrate exceed then complement
    _, cal_fn = _fit_beta_calibrator(y_exceed[va], p_val_exceed)
    p_val = 1 - cal_fn(p_val_exceed)
    p_test = 1 - cal_fn(p_test_exceed)
    results[exp] = evaluate_and_save(exp, feats, p_val, p_test, model, y[va], y[te], va, te, X_val, "beta", {"exp": exp}, regime_key)

    # EXP08 ZI_DELTA
    exp = "EXP08_ZI_DELTA"
    feats, X_train, X_val, X_test, tr, va, te, _ = prepare_matrices(feature_sets[exp])
    check_sig(feats, {"exp": exp})
    model_cls = _train_lgbm_classifier(X_train, y[tr], X_val, y[va], lgb_params)
    p_val_raw = model_cls.predict(X_val)
    p_test_raw = model_cls.predict(X_test)
    # optional regressor for delta magnitude (not used directly in calibration here)
    delta_mask = merged_df["delta_future"].to_numpy() > 0.01
    if delta_mask[tr].sum() > 50:
        model_reg = _train_lgbm_regressor(X_train[delta_mask[tr]], merged_df.loc[tr & delta_mask, "log_delta_future"].to_numpy(),
                                          X_val[delta_mask[va]], merged_df.loc[va & delta_mask, "log_delta_future"].to_numpy(), lgb_params)
        _ = model_reg.predict(X_val)
    results[exp] = evaluate_and_save(exp, feats, p_val_raw, p_test_raw, model_cls, y[va], y[te], va, te, X_val, "beta", {"exp": exp}, regime_key)

    # EXP09 CNN2GBM (embedding via DCT)
    exp = "EXP09_CNN2GBM"
    feats, X_train, X_val, X_test, tr, va, te, _ = prepare_matrices(feature_sets[exp])
    check_sig(feats, {"exp": exp})
    model = _train_lgbm_classifier(X_train, y[tr], X_val, y[va], lgb_params)
    p_val = model.predict(X_val)
    p_test = model.predict(X_test)
    results[exp] = evaluate_and_save(exp, feats, p_val, p_test, model, y[va], y[te], va, te, X_val, "beta", {"exp": exp}, regime_key)

    # EXP10 CPC_EMBED (embedding via DCT)
    exp = "EXP10_CPC_EMBED"
    feats, X_train, X_val, X_test, tr, va, te, _ = prepare_matrices(feature_sets[exp])
    check_sig(feats, {"exp": exp})
    model = _train_lgbm_classifier(X_train, y[tr], X_val, y[va], lgb_params)
    p_val = model.predict(X_val)
    p_test = model.predict(X_test)
    results[exp] = evaluate_and_save(exp, feats, p_val, p_test, model, y[va], y[te], va, te, X_val, "beta", {"exp": exp}, regime_key)

    # EXP11 MOE 4-regime
    exp = "EXP11_MOE_4REG"
    feats, X_train, X_val, X_test, tr, va, te, _ = prepare_matrices(feature_sets[exp])
    check_sig(feats, {"exp": exp})
    # define regimes
    wind = merged_df["wind_bin"].astype(str)
    supp = merged_df["suppression_bin"].astype(str)
    wind_reg = wind.where(wind == "onshore", "offshore")
    supp_reg = supp.where(supp == "suppressed", "clear")
    regime = wind_reg.str.cat(supp_reg, sep="_")
    regime_labels = pd.Categorical(regime, categories=["onshore_suppressed", "onshore_clear", "offshore_suppressed", "offshore_clear"]).codes

    # gate features
    gate_cols = ["mos_u_mean", "mos_v_mean", "mos_wsp_mean_models", "mos_cig_min_models", "mos_q12_max_models", "doy_sin", "doy_cos", "minutes_since_max", "slope_60m"]
    gate_cols = [c for c in gate_cols if c in merged_df.columns]
    gate_imputer = SimpleImputer(strategy="median")
    gate_X_train = gate_imputer.fit_transform(merged_df.loc[tr, gate_cols])
    gate_X_val = gate_imputer.transform(merged_df.loc[va, gate_cols])
    gate_X_test = gate_imputer.transform(merged_df.loc[te, gate_cols])
    gate_model = _train_lgbm_multiclass(gate_X_train, regime_labels[tr], gate_X_val, regime_labels[va], lgb_params, num_class=4)
    gate_val = gate_model.predict(gate_X_val)
    gate_test = gate_model.predict(gate_X_test)

    # experts
    expert_preds_val = np.zeros((va.sum(), 4))
    expert_preds_test = np.zeros((te.sum(), 4))
    for ridx, reg in enumerate(["onshore_suppressed", "onshore_clear", "offshore_suppressed", "offshore_clear"]):
        reg_mask = regime_labels == ridx
        if (tr & reg_mask).sum() < 50:
            # fallback to global
            model_reg = _train_lgbm_classifier(X_train, y[tr], X_val, y[va], lgb_params)
            expert_preds_val[:, ridx] = model_reg.predict(X_val)
            expert_preds_test[:, ridx] = model_reg.predict(X_test)
        else:
            model_reg = _train_lgbm_classifier(X_train[reg_mask[tr]], y[tr & reg_mask], X_val, y[va], lgb_params)
            expert_preds_val[:, ridx] = model_reg.predict(X_val)
            expert_preds_test[:, ridx] = model_reg.predict(X_test)

    p_val = np.sum(gate_val * expert_preds_val, axis=1)
    p_test = np.sum(gate_test * expert_preds_test, axis=1)
    results[exp] = evaluate_and_save(exp, feats, p_val, p_test, None, y[va], y[te], va, te, X_val, "beta", {"exp": exp}, regime_key)

    # EXP12 MONO_PHYS
    exp = "EXP12_MONO_PHYS"
    feats, X_train, X_val, X_test, tr, va, te, _ = prepare_matrices(feature_sets[exp])
    check_sig(feats, {"exp": exp})
    params_mono = lgb_params.copy()
    params_mono["monotone_constraints"] = _build_monotone_constraints(feats)
    model = _train_lgbm_classifier(X_train, y[tr], X_val, y[va], params_mono)
    p_val = model.predict(X_val)
    p_test = model.predict(X_test)
    results[exp] = evaluate_and_save(exp, feats, p_val, p_test, model, y[va], y[te], va, te, X_val, "beta", {"exp": exp}, regime_key)

    # EXP13 SLOPE_HIST
    exp = "EXP13_SLOPE_HIST"
    feats, X_train, X_val, X_test, tr, va, te, _ = prepare_matrices(feature_sets[exp])
    check_sig(feats, {"exp": exp})
    model = _train_lgbm_classifier(X_train, y[tr], X_val, y[va], lgb_params)
    p_val = model.predict(X_val)
    p_test = model.predict(X_test)
    results[exp] = evaluate_and_save(exp, feats, p_val, p_test, model, y[va], y[te], va, te, X_val, "platt", {"exp": exp}, regime_key)

    # EXP14 DROP_REBOUND
    exp = "EXP14_DROP_REBOUND"
    feats, X_train, X_val, X_test, tr, va, te, _ = prepare_matrices(feature_sets[exp])
    check_sig(feats, {"exp": exp})
    model = _train_lgbm_classifier(X_train, y[tr], X_val, y[va], lgb_params)
    p_val = model.predict(X_val)
    p_test = model.predict(X_test)
    results[exp] = evaluate_and_save(exp, feats, p_val, p_test, model, y[va], y[te], va, te, X_val, "beta", {"exp": exp}, regime_key)

    # EXP15 QA_AWARE
    exp = "EXP15_QA_AWARE"
    feats, X_train, X_val, X_test, tr, va, te, _ = prepare_matrices(feature_sets[exp])
    check_sig(feats, {"exp": exp})
    model = _train_lgbm_classifier(X_train, y[tr], X_val, y[va], lgb_params)
    p_val = model.predict(X_val)
    p_test = model.predict(X_test)
    results[exp] = evaluate_and_save(exp, feats, p_val, p_test, model, y[va], y[te], va, te, X_val, "platt", {"exp": exp}, regime_key)

    # EXP16 CTP_BAYES
    exp = "EXP16_CTP_BAYES"
    feats, X_train, X_val, X_test, tr, va, te, _ = prepare_matrices(feature_sets[exp])
    check_sig(feats, {"exp": exp})
    model = _train_lgbm_classifier(X_train, y[tr], X_val, y[va], lgb_params)
    p_val = model.predict(X_val)
    p_test = model.predict(X_test)
    results[exp] = evaluate_and_save(exp, feats, p_val, p_test, model, y[va], y[te], va, te, X_val, "beta", {"exp": exp}, regime_key)

    # EXP17 MOS_NOWCAST_MIS_v2
    exp = "EXP17_MOS_NOWCAST_MIS_v2"
    feats, X_train, X_val, X_test, tr, va, te, _ = prepare_matrices(feature_sets[exp])
    check_sig(feats, {"exp": exp})
    model = _train_lgbm_classifier(X_train, y[tr], X_val, y[va], lgb_params)
    p_val = model.predict(X_val)
    p_test = model.predict(X_test)
    results[exp] = evaluate_and_save(exp, feats, p_val, p_test, model, y[va], y[te], va, te, X_val, "platt", {"exp": exp}, regime_key)

    # EXP18 EV_STACK_OOF
    exp = "EXP18_EV_STACK_OOF"
    base_names = ["EXP01_Fusion_SBFD", "EXP12_MONO_PHYS", "EXP07_HAZARD_EXCEED", "EXP06_Fusion_ANALOG_PRIOR"]
    meta_feature_names = [f"base_{name}" for name in base_names] + ["suppression_index"]
    feats = meta_feature_names
    check_sig(feats, {"exp": exp, "bases": base_names})
    # define folds for OOF
    years = pd.to_datetime(merged_df["target_date_local"]).dt.year.to_numpy()
    folds = [(2002, 2006, 2007, 2009), (2002, 2009, 2010, 2014), (2002, 2014, 2015, 2019)]
    base_preds_oof = {name: np.full(len(merged_df), np.nan) for name in base_names}
    base_preds_val = {}
    base_preds_test = {}

    for train_end, _, oof_start, oof_end in folds:
        fold_train = (years <= train_end)
        fold_oof = (years >= oof_start) & (years <= oof_end)
        for name in base_names:
            fcols = feature_sets[name]
            fcols = [c for c in fcols if c in merged_df.columns]
            X = merged_df[fcols]
            imp = SimpleImputer(strategy="median")
            X_tr = imp.fit_transform(X[fold_train])
            X_oof = imp.transform(X[fold_oof])
            model = _train_lgbm_classifier(X_tr, y[fold_train], X_oof, y[fold_oof], lgb_params)
            base_preds_oof[name][fold_oof] = model.predict(X_oof)

    # meta training on OOF
    oof_mask = train_mask & np.all([np.isfinite(base_preds_oof[n]) for n in base_names], axis=0)
    meta_X = np.column_stack([base_preds_oof[n][oof_mask] for n in base_names] + [merged_df.loc[oof_mask, "suppression_index"].to_numpy()])
    meta_model = _train_lgbm_classifier(meta_X, y[oof_mask], meta_X, y[oof_mask], lgb_params)

    # fit base models on full train
    for name in base_names:
        fcols = feature_sets[name]
        fcols = [c for c in fcols if c in merged_df.columns]
        X = merged_df[fcols]
        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X[train_mask])
        X_val = imp.transform(X[val_mask])
        X_test = imp.transform(X[test_mask])
        model = _train_lgbm_classifier(X_tr, y[train_mask], X_val, y[val_mask], lgb_params)
        base_preds_val[name] = model.predict(X_val)
        base_preds_test[name] = model.predict(X_test)

    meta_val = np.column_stack([base_preds_val[n] for n in base_names] + [merged_df.loc[val_mask, "suppression_index"].to_numpy()])
    meta_test = np.column_stack([base_preds_test[n] for n in base_names] + [merged_df.loc[test_mask, "suppression_index"].to_numpy()])
    p_val = meta_model.predict(meta_val)
    p_test = meta_model.predict(meta_test)
    results[exp] = evaluate_and_save(exp, feats, p_val, p_test, meta_model, y[val_mask], y[test_mask], val_mask, test_mask, meta_val, "beta", {"exp": exp, "bases": base_names}, regime_key)

    # EXP19 MT_DELTA_BINS
    exp = "EXP19_MT_DELTA_BINS"
    valid = np.isfinite(delta_bin)
    feats, X_train, X_val, X_test, tr, va, te, _ = prepare_matrices(feature_sets[exp], valid)
    check_sig(feats, {"exp": exp})
    model = _train_lgbm_multiclass(X_train, delta_bin[tr].astype(int), X_val, delta_bin[va].astype(int), lgb_params, num_class=4)
    p_val_all = model.predict(X_val)
    p_test_all = model.predict(X_test)
    p_val = p_val_all[:, 0]
    p_test = p_test_all[:, 0]
    results[exp] = evaluate_and_save(exp, feats, p_val, p_test, model, y[va], y[te], va, te, X_val, "beta", {"exp": exp}, regime_key)

    # EXP20 HB_PARAM_T
    exp = "EXP20_HB_PARAM_T"
    feats, X_train, X_val, X_test, tr, va, te, _ = prepare_matrices(feature_sets[exp])
    check_sig(feats, {"exp": exp})
    # mean model
    model_mu = _train_lgbm_regressor(X_train, merged_df.loc[tr, "log_delta_future"].to_numpy(), X_val, merged_df.loc[va, "log_delta_future"].to_numpy(), lgb_params)
    mu_val = model_mu.predict(X_val)
    mu_test = model_mu.predict(X_test)
    # sigma model on residuals
    resid = merged_df.loc[tr, "log_delta_future"].to_numpy() - model_mu.predict(X_train)
    model_sigma = _train_lgbm_regressor(X_train, np.abs(resid), X_val, np.abs(merged_df.loc[va, "log_delta_future"].to_numpy() - mu_val), lgb_params)
    sigma_val = np.clip(model_sigma.predict(X_val), 0.1, None)
    sigma_test = np.clip(model_sigma.predict(X_test), 0.1, None)
    # Student-t CDF for log(delta+0.01) <= log(0.02)
    cutoff_log = math.log(0.02)
    p_val_raw = student_t.cdf((cutoff_log - mu_val) / sigma_val, df=6)
    p_test_raw = student_t.cdf((cutoff_log - mu_test) / sigma_test, df=6)
    results[exp] = evaluate_and_save(exp, feats, p_val_raw, p_test_raw, model_mu, y[va], y[te], va, te, X_val, "beta", {"exp": exp}, regime_key)

    # Summary outputs
    summary_json_path = out_dir / f"{BATCH_NAME}_hit1830_v5_experiments_summary.json"
    summary_md_path = out_dir / "hit1830_v5_experiments_report.md"

    summary_json_path.write_text(f"{BATCH_NAME}\n" + json.dumps(results, indent=2), encoding="utf-8")

    rows = []
    for name, metrics in results.items():
        val_m = metrics["metrics_val"]
        test_m = metrics["metrics_test"]
        rows.append(
            {
                "exp": name,
                "val_acc": val_m["accuracy"],
                "val_bal": val_m["balanced_accuracy"],
                "val_yes_recall": val_m["yes_recall"],
                "test_acc": test_m["accuracy"],
                "test_bal": test_m["balanced_accuracy"],
                "test_yes_recall": test_m["yes_recall"],
                "net_units_test": metrics["ev_summary_test"].get("c0.55_ev0.03", {}).get("net_units_per_100_days"),
                "trade_rate_test": metrics["ev_summary_test"].get("c0.55_ev0.03", {}).get("trade_rate"),
            }
        )

    best_val_acc = max(rows, key=lambda r: r["val_acc"])
    best_val_bal = max(rows, key=lambda r: r["val_bal"])

    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(f"# Hit 18:30 Stockholm V5 Experiments Report\n\n")
        f.write(f"Always-NO test accuracy: {always_no_acc:.3f}\n\n")
        f.write(f"Always-YES test accuracy: {always_yes_acc:.3f}\n\n")
        f.write("| Experiment | Val Acc | Val Bal Acc | Val YES Recall | Test Acc | Test Bal Acc | Test YES Recall | NetUnits/100 (Test, c0.55 ev0.03) | TradeRate (Test) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for row in rows:
            f.write(
                f"| {row['exp']} | {row['val_acc']:.3f} | {row['val_bal']:.3f} | {row['val_yes_recall']:.3f} | {row['test_acc']:.3f} | {row['test_bal']:.3f} | {row['test_yes_recall']:.3f} | {row['net_units_test']:.2f} | {row['trade_rate_test']:.3f} |\n"
            )
        f.write("\n")
        f.write(f"Best by Val Accuracy: {best_val_acc['exp']} (Val Acc {best_val_acc['val_acc']:.3f})\n\n")
        f.write(f"Best by Val Balanced Accuracy: {best_val_bal['exp']} (Val Bal Acc {best_val_bal['val_bal']:.3f})\n\n")

    # Executive summary
    exec_path = out_dir / f"{BATCH_NAME}_executive_summary"
    exec_lines = [BATCH_NAME, f"Batch {BATCH_NAME} Executive Summary", ""]
    for exp in sorted(results.keys()):
        desc = _EXEC_SUMMARY.get(exp, "")
        exec_lines.append(f"{exp}: {desc}")
        exec_lines.append("")
    exec_path.write_text("\n".join(exec_lines), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
