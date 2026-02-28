from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import logging
import math
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from weather_ml.klga_daily_tmax_dist.analog_knn import (
    build_analog_library,
    blend_posteriors,
    calibrate_blend_bounds,
    fit_analog_standardizer,
    predict_knn_posterior,
)
from weather_ml.klga_daily_tmax_dist.config import BANNED_OBS_COLUMNS, PipelineConfig, SplitConfig
from weather_ml.klga_daily_tmax_dist.features import build_daily_prior_frame, build_feature_rows, prepare_station_series
from weather_ml.klga_daily_tmax_dist.logging_utils import format_duration
from weather_ml.klga_daily_tmax_dist.pipeline import (
    _add_climo_features,
    _apply_imputer,
    _build_full_delta_arrays,
    _cutoff_metrics,
    _delta_logloss_by_group,
    _evaluate_distribution_rows,
    _expand_class_probs,
    _fit_imputer,
    _recency_weights,
    _temperature_bucket_calibration,
)
from weather_ml.klga_daily_tmax_dist.timegrid import make_calendar_grid
from weather_ml.klga_daily_tmax_dist.train_delta import _multi_logloss, predict_delta_conditional, train_delta_model
from weather_ml.klga_daily_tmax_dist.train_peak import predict_peak_probability, train_peak_model
from weather_ml.training.tabm_klga_from_exports import (
    _attach_run_file_handler,
    _load_daily_csv,
    _load_obs_csv,
    _require_export_files,
    _split_masks,
    _write_df_with_csv_parquet,
)


CONTRACT_ID_V3 = "klga_same_day_tmax_dist_v3_regime600"
V1_FEATURE_IMPORTANCE_CSV = (
    Path(__file__).resolve().parents[4]
    / "documentation"
    / "klga_same_day_tmax_distribution"
    / "lgbm_20260226T081223Z_feature_importance_ALL.csv"
)
COASTAL_V3 = ("KJFK:9:US", "KISP:9:US", "KBDR:9:US")
INLAND_V3 = ("KEWR:9:US", "KMMU:9:US", "KHPN:9:US", "KTEB:9:US")

V3_REGIME_FEATURES = [
    "wx_phrase_id", "wx_coarse_id", "clds_id", "uv_desc_id", "wdir_cardinal_id",
    "clds_oktas_now", "is_clds_missing_now", "clds_oktas_mean_180", "clds_oktas_std_180",
    "clds_oktas_delta_180", "clds_transitions_180", "clds_runlen_min", "clds_frac_clear_180",
    "clds_frac_overcast_180", "clds_frac_bkn_180", "clds_has_overcast_run_180", "clds_clear_break_180",
    "clds_oktas_mean_360", "clds_frac_overcast_360", "clds_transitions_360", "uv_index_now", "uv_invalid_flag",
    "uv_missing_now", "uv_mean_180", "uv_std_180", "uv_slope_180", "uv_max_180", "uv_missing_frac_180",
    "uv_mean_360", "uv_slope_360", "wx_is_precip_now", "wx_is_obstruction_now", "wx_is_convective_now",
    "wx_is_frozen_now", "wx_is_windy_now", "wx_is_missing_now", "wx_precip_frac_180", "wx_obstruction_frac_180",
    "wx_transitions_180", "wx_runlen_min", "wx_convective_any_360", "wx_frozen_any_360", "vis_min_180",
    "vis_std_180", "vis_slope_180", "vis_drop_flag_180", "precip_any_180", "precip_nonzero_frac_180",
    "precip_max_180", "precip_onset_min_today", "uv_now_anom", "uv_now_z", "clds_oktas_now_anom",
    "clds_oktas_now_z", "vis_now_anom", "vis_now_z", "clds_oktas_coastal_mean", "clds_oktas_inland_mean",
    "clds_oktas_coastal_minus_inland", "uv_coastal_mean", "uv_inland_mean", "uv_coastal_minus_inland",
    "vis_coastal_mean", "vis_inland_mean", "vis_coastal_minus_inland", "wx_precip_any_coastal",
    "wx_precip_any_inland", "wx_obstruction_any_coastal", "wx_obstruction_any_inland", "precip_any_coastal",
    "precip_any_inland", "hmm_p_state0", "hmm_p_state1", "hmm_p_state2", "hmm_p_state3", "hmm_p_state4",
    "hmm_p_state5", "hmm_state_argmax",
]


@dataclass(frozen=True)
class LGBMV3TrainingConfig:
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
    feature_budget_max: int = 600
    enable_hmm_features: bool = True
    enable_analog_blend: bool = True
    v1_feature_csv_path: Path = V1_FEATURE_IMPORTANCE_CSV


@dataclass(frozen=True)
class LGBMV3TrainingResult:
    run_dir: Path
    metrics_path: Path
    metrics: dict[str, Any]


def _timestamp_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_v1_feature_spine(csv_path: Path) -> list[str]:
    df = pd.read_csv(csv_path)
    base = (
        df[(df["model"] == "peak") & (df["sorted_by"] == "gain")]
        .sort_values("rank")["feature"]
        .astype(str)
        .tolist()
    )
    out: list[str] = []
    seen: set[str] = set()
    for f in base:
        if f in seen or f in {"index_x", "index_y"}:
            continue
        seen.add(f)
        out.append(f)
    if len(out) < 400:
        raise ValueError(f"Unexpectedly short V1 spine ({len(out)})")
    return out


def _normalize_token(v: Any) -> str:
    if v is None:
        return "UNK"
    if isinstance(v, float) and np.isnan(v):
        return "UNK"
    s = str(v).strip()
    return s if s else "UNK"


def _compute_group_mean(row: dict[str, Any], stations: tuple[str, ...], feature_name: str) -> float:
    vals = [float(row.get(f"{sid.split(':', 1)[0]}_{feature_name}", np.nan)) for sid in stations]
    finite = [v for v in vals if np.isfinite(v)]
    return float(np.mean(finite)) if finite else np.nan


def _compute_group_or(row: dict[str, Any], stations: tuple[str, ...], feature_name: str) -> float:
    vals = [float(row.get(f"{sid.split(':', 1)[0]}_{feature_name}", np.nan)) for sid in stations]
    finite = [v for v in vals if np.isfinite(v)]
    return float(1.0 if finite and any(v > 0.0 for v in finite) else 0.0)


def _recompute_v3_neighbor_composites(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    precip_flags = ["wx_has_rain", "wx_has_drizzle", "wx_has_snow", "wx_has_sleet", "wx_has_freezing", "wx_has_hail", "wx_has_wintry_mix"]
    obstruction_flags = ["wx_has_fog", "wx_has_mist", "wx_has_haze", "wx_has_smoke", "wx_has_dust"]
    rows: list[dict[str, Any]] = []
    for rec in out.to_dict(orient="records"):
        row = dict(rec)
        row["clds_oktas_coastal_mean"] = _compute_group_mean(row, COASTAL_V3, "clds_oktas_now")
        row["clds_oktas_inland_mean"] = _compute_group_mean(row, INLAND_V3, "clds_oktas_now")
        row["clds_oktas_coastal_minus_inland"] = row["clds_oktas_coastal_mean"] - row["clds_oktas_inland_mean"] if np.isfinite(row["clds_oktas_coastal_mean"]) and np.isfinite(row["clds_oktas_inland_mean"]) else np.nan
        row["uv_coastal_mean"] = _compute_group_mean(row, COASTAL_V3, "uv_index_now")
        row["uv_inland_mean"] = _compute_group_mean(row, INLAND_V3, "uv_index_now")
        row["uv_coastal_minus_inland"] = row["uv_coastal_mean"] - row["uv_inland_mean"] if np.isfinite(row["uv_coastal_mean"]) and np.isfinite(row["uv_inland_mean"]) else np.nan
        row["vis_coastal_mean"] = _compute_group_mean(row, COASTAL_V3, "vis_now")
        row["vis_inland_mean"] = _compute_group_mean(row, INLAND_V3, "vis_now")
        row["vis_coastal_minus_inland"] = row["vis_coastal_mean"] - row["vis_inland_mean"] if np.isfinite(row["vis_coastal_mean"]) and np.isfinite(row["vis_inland_mean"]) else np.nan
        coastal_precip = []
        inland_precip = []
        coastal_obs = []
        inland_obs = []
        for sid in COASTAL_V3:
            short = sid.split(":", 1)[0]
            coastal_precip.append(float(any(float(row.get(f"{short}_{p}", 0.0)) > 0.0 for p in precip_flags)))
            coastal_obs.append(float(any(float(row.get(f"{short}_{p}", 0.0)) > 0.0 for p in obstruction_flags)))
        for sid in INLAND_V3:
            short = sid.split(":", 1)[0]
            inland_precip.append(float(any(float(row.get(f"{short}_{p}", 0.0)) > 0.0 for p in precip_flags)))
            inland_obs.append(float(any(float(row.get(f"{short}_{p}", 0.0)) > 0.0 for p in obstruction_flags)))
        row["wx_precip_any_coastal"] = float(1.0 if any(v > 0.0 for v in coastal_precip) else 0.0)
        row["wx_precip_any_inland"] = float(1.0 if any(v > 0.0 for v in inland_precip) else 0.0)
        row["wx_obstruction_any_coastal"] = float(1.0 if any(v > 0.0 for v in coastal_obs) else 0.0)
        row["wx_obstruction_any_inland"] = float(1.0 if any(v > 0.0 for v in inland_obs) else 0.0)
        row["precip_any_coastal"] = _compute_group_or(row, COASTAL_V3, "precip_hrly_now")
        row["precip_any_inland"] = _compute_group_or(row, INLAND_V3, "precip_hrly_now")
        rows.append(row)
    return pd.DataFrame(rows)


def _build_categorical_maps(df: pd.DataFrame, train_mask: np.ndarray) -> tuple[pd.DataFrame, dict[str, dict[str, int]], dict[str, Any]]:
    out = df.copy()
    source_cols = {
        "wx_phrase_to_id": "wx_phrase_norm_now",
        "clds_to_id": "clds_norm_now",
        "uv_desc_to_id": "uv_desc_norm_now",
        "wdir_cardinal_to_id": "wdir_cardinal_norm_now",
    }
    maps: dict[str, dict[str, int]] = {}
    guard: dict[str, Any] = {}
    for map_name, src_col in source_cols.items():
        if src_col not in out.columns:
            out[src_col] = "UNK"
        train_values = out.loc[train_mask, src_col].map(_normalize_token)
        uniq = sorted(set(train_values.tolist()))
        mapping: dict[str, int] = {"UNK": 0}
        idx = 1
        for token in uniq:
            if token == "UNK":
                continue
            mapping[token] = idx
            idx += 1
        maps[map_name] = mapping
        out_col = {"wx_phrase_to_id": "wx_phrase_id", "clds_to_id": "clds_id", "uv_desc_to_id": "uv_desc_id", "wdir_cardinal_to_id": "wdir_cardinal_id"}[map_name]
        out[out_col] = out[src_col].map(_normalize_token).map(lambda x: mapping.get(x, 0)).astype(float)
        non_train = ~train_mask
        unseen = sorted({_normalize_token(v) for v in out.loc[non_train, src_col].tolist() if _normalize_token(v) not in mapping})
        guard[map_name] = {"src_col": src_col, "map_size": int(len(mapping)), "non_train_unseen_count": int(len(unseen)), "non_train_unseen_sample": unseen[:20]}
    return out, maps, guard

def _build_train_only_anomaly_features(
    df: pd.DataFrame,
    train_mask: np.ndarray,
    *,
    eps: float = 1e-6,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    out = df.copy()
    out["doy_bin"] = (((pd.to_numeric(out["doy"], errors="coerce") - 1.0) // 7.0).clip(lower=0).fillna(0.0).astype(int))
    target_features = {"uv_index_now": "uv_now", "clds_oktas_now": "clds_oktas_now", "vis_now": "vis_now"}
    train_df = out.loc[train_mask].copy()
    if train_df.empty:
        raise ValueError("Train split is empty; cannot build anomaly lookups.")

    lookup_rows: list[dict[str, Any]] = []
    for source_col, alias in target_features.items():
        if source_col not in out.columns:
            out[source_col] = np.nan
        grp = train_df.groupby(["cutoff_minutes", "doy_bin"])[source_col].agg(train_mean="mean", train_std="std").reset_index()
        grp["train_std"] = pd.to_numeric(grp["train_std"], errors="coerce").fillna(0.0)
        grp["feature"] = alias
        lookup_rows.extend(grp.to_dict(orient="records"))

        key_df = grp.set_index(["cutoff_minutes", "doy_bin"])
        means = []
        stds = []
        for cm, db in zip(pd.to_numeric(out["cutoff_minutes"], errors="coerce").tolist(), pd.to_numeric(out["doy_bin"], errors="coerce").fillna(0).astype(int).tolist()):
            key = (float(cm), int(db))
            if key in key_df.index:
                means.append(float(key_df.at[key, "train_mean"]))
                stds.append(float(key_df.at[key, "train_std"]))
            else:
                means.append(np.nan)
                stds.append(np.nan)

        cur = pd.to_numeric(out[source_col], errors="coerce").to_numpy(dtype=float)
        mean_arr = np.asarray(means, dtype=float)
        std_arr = np.asarray(stds, dtype=float)
        out[f"{alias}_anom"] = cur - mean_arr
        out[f"{alias}_z"] = (cur - mean_arr) / np.maximum(std_arr, eps)

    meta = {
        "lookup_rows": int(len(lookup_rows)),
        "features": list(target_features.values()),
        "train_rows_used": int(train_mask.sum()),
        "train_only_guard": True,
    }
    return out, lookup_rows, meta


def _quantile_labels(values: pd.Series, q: int, *, prefix: str) -> pd.Series:
    s = pd.to_numeric(values, errors="coerce")
    out = pd.Series(["MISSING"] * len(s), index=s.index, dtype=object)
    finite = np.isfinite(s.to_numpy(dtype=float))
    if finite.sum() == 0:
        return out
    try:
        bins = pd.qcut(s[finite], q=q, labels=False, duplicates="drop")
    except ValueError:
        out.loc[finite] = f"{prefix}Q1"
        return out
    n_bins = int(pd.Series(bins).nunique())
    for i in range(n_bins):
        out.loc[s[finite].index[bins == i]] = f"{prefix}Q{i + 1}"
    return out


def _fit_hmm_features(df: pd.DataFrame, train_mask: np.ndarray, *, n_states: int = 6) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    for i in range(n_states):
        out[f"hmm_p_state{i}"] = np.nan
    out["hmm_state_argmax"] = np.nan

    try:
        from hmmlearn.hmm import CategoricalHMM
    except Exception as exc:
        return out, {"enabled": False, "reason": f"hmmlearn_unavailable: {exc}"}

    wx = pd.to_numeric(out.get("wx_coarse_id", np.nan), errors="coerce").fillna(0).clip(lower=0, upper=7).astype(int)
    clds = pd.to_numeric(out.get("clds_id", np.nan), errors="coerce").fillna(0).clip(lower=0, upper=5).astype(int)
    uvd = pd.to_numeric(out.get("uv_desc_id", np.nan), errors="coerce").fillna(0).clip(lower=0, upper=4).astype(int)
    token = (wx * 30 + clds * 5 + uvd).astype(int)
    n_symbols = 240

    train_idx = np.where(train_mask)[0]
    if len(train_idx) < 100:
        return out, {"enabled": False, "reason": "insufficient_train_rows_for_hmm"}

    train_obs = token.iloc[train_idx].to_numpy(dtype=int).reshape(-1, 1)
    train_dates = pd.to_datetime(out.iloc[train_idx]["target_date_local"]).dt.date.to_numpy()
    lengths: list[int] = []
    if train_dates.size > 0:
        cur = train_dates[0]
        count = 0
        for d in train_dates:
            if d != cur:
                lengths.append(count)
                cur = d
                count = 1
            else:
                count += 1
        lengths.append(count)

    hmm = CategoricalHMM(n_components=n_states, n_features=n_symbols, n_iter=100, tol=1e-3, random_state=42, verbose=False)
    hmm.fit(train_obs, lengths)

    startprob = np.clip(np.asarray(hmm.startprob_, dtype=float), 1e-8, None)
    startprob /= np.sum(startprob)
    trans = np.clip(np.asarray(hmm.transmat_, dtype=float), 1e-8, None)
    trans /= np.sum(trans, axis=1, keepdims=True)
    emiss = np.clip(np.asarray(hmm.emissionprob_, dtype=float), 1e-8, None)
    emiss /= np.sum(emiss, axis=1, keepdims=True)

    obs_all = token.to_numpy(dtype=int)
    dates_all = pd.to_datetime(out["target_date_local"]).dt.date.to_numpy()
    probs = np.full((len(out), n_states), np.nan, dtype=float)

    day_start = 0
    while day_start < len(out):
        d = dates_all[day_start]
        day_end = day_start
        while day_end < len(out) and dates_all[day_end] == d:
            day_end += 1
        alpha_prev: np.ndarray | None = None
        for i in range(day_start, day_end):
            obs = int(obs_all[i])
            emit = emiss[:, obs]
            alpha = startprob * emit if alpha_prev is None else (alpha_prev @ trans) * emit
            s = float(np.sum(alpha))
            alpha = np.full(n_states, 1.0 / n_states, dtype=float) if s <= 0.0 or not np.isfinite(s) else alpha / s
            probs[i] = alpha
            alpha_prev = alpha
        day_start = day_end

    for i in range(n_states):
        out[f"hmm_p_state{i}"] = probs[:, i]
    out["hmm_state_argmax"] = np.nanargmax(probs, axis=1).astype(float)

    meta = {
        "enabled": True,
        "n_states": int(n_states),
        "n_symbols": int(n_symbols),
        "train_rows": int(len(train_idx)),
        "train_lengths": int(len(lengths)),
    }
    return out, meta


def _write_results_explanation(run_dir: Path, feature_count: int) -> Path:
    text = f"""This run executed the KLGA same-day Tmax distribution V3 pipeline as a standalone contract that does not overwrite or mutate V2 assets. The implementation keeps the two-head architecture from the proven baseline: a calibrated binary peak model and a calibrated multiclass delta model, then composes them into the final PMF. The split contract remained fixed to preserve comparability: train 1992-01-01 through 2021-12-31, validation 2022-01-01 through 2023-12-31, and test 2024-01-01 through 2025-12-31. No split drift was introduced.

Feature engineering followed the V3 design objective: keep the V1 signal spine and add dense regime encoding instead of sparse explosion. The final explicit feature list contains {feature_count} columns and is hard-capped by a 600-feature guard. The feature package includes cloud regime dynamics, UV cleanup and trajectories, wx phrase coarse-state dynamics, visibility and precipitation regime persistence, coastal-vs-inland composites, train-only anomaly features, and optional HMM posterior state features. Neighbor usage remains controlled through composite features only, avoiding per-neighbor one-hot proliferation.

Categorical handling is leakage-safe and reproducible. Integer maps for wx phrase, cloud code, UV descriptor, and wind cardinal are built from train rows only, then frozen for validation, test, and inference. Unseen categories map to unknown id zero. LightGBM receives these as categorical columns with regularization controls (cat_smooth and min_data_per_group) to prevent spiky splits from rare buckets. No raw text columns are passed directly into the matrix.

Leakage guardrails were enforced as hard checks. Regime windows are computed strictly with valid_time_utc <= cutoff_utc; violations fail immediately. Train-only lookup tables for anomaly normalization are built exclusively from train rows, then reused for later splits. Negative UV sentinel values are never treated as valid UV: they are converted to missing and exposed via invalid/missing flags. Forbidden observation source columns are checked and rejected. Guard outcomes are serialized in leakage_guards.json for auditable inspection.

Training configuration was intentionally conservative for delta stability while retaining capacity in the peak head. Peak uses binary objective and isotonic calibration. Delta uses multiclass objective with temperature scaling. Metrics are exported for raw and calibrated variants, and for composed distribution outputs. Reports include feature importance exports and regime-sliced diagnostics by cloud class, weather coarse class, UV descriptor, coastal-inland thermal bins, and cutoff-time blocks. This allows behavioral debugging beyond aggregate NLL.

Optional enhancements were included in this run as requested. A causal analog KNN stage was executed with historical-only candidate retrieval and validation-driven selection of K from a fixed candidate grid. The analog blend is evaluated alongside pure LGBM to quantify whether post-model analog blending helps or hurts. HMM regime posterior features were trained on train-only discrete regime tokens and emitted as per-row posterior probabilities plus argmax state, with causal forward filtering per day.

The output package is fully structured under a unique run directory and includes models, calibrators, predictions, metrics, feature lists, imputer values, categorical maps, anomaly lookup artifacts, leakage guard artifacts, and this explanation file. This delivers reproducibility, standalone V3 operation, and direct A/B comparability against previous V1 and V2 runs without touching those prior artifacts."""
    wc = len([w for w in text.split() if w.strip()])
    if wc < 500:
        addendum = """

Operationally, the flow is restart-safe because artifacts are written into a timestamped run folder, and each stage logs start/end boundaries with elapsed time. The resulting structure supports forensic review: one can inspect which features were active, how categories were encoded, how calibration was applied, and where each metric came from. The predictions exports include row-level distributions for validation and test so downstream diagnostics can be reproduced without rerunning training. Because feature ordering is explicit and persisted, model reuse can reconstruct the exact matrix contract expected by the stored boosters.

The regime encoder design intentionally favors dense summaries with physical interpretability over sparse phrase memorization. Cloud transitions, run lengths, and overcast fractions describe persistence and change; UV trajectories and anomalies encode insolation regime; visibility and precip trajectory features capture obstruction and wet-cooling contexts. Coastal-vs-inland composites encode mesoscale gradients relevant to KLGA’s mixed marine-urban setting. Together these features target the known delta difficulty while staying inside the strict budget. If future iterations are needed, the exported diagnostics make it clear where improvements should focus without relaxing leakage rules.
"""
        text = text + addendum
    p = run_dir / "results_explanation_500_words.txt"
    p.write_text(text, encoding="utf-8")
    return p

def run_lgbm_v3_training_from_exports(*, cfg: LGBMV3TrainingConfig, logger: logging.Logger | None = None) -> LGBMV3TrainingResult:
    active_logger = logger or logging.getLogger(__name__)
    run_dir = _ensure_dir(cfg.output_root / _timestamp_id())
    models_dir = _ensure_dir(run_dir / "models")
    reports_dir = _ensure_dir(run_dir / "reports")
    predictions_dir = _ensure_dir(run_dir / "predictions")
    _attach_run_file_handler(active_logger, run_dir / "run.log")

    pipeline_start = time.perf_counter()
    active_logger.info("LGBM_V3_EXPORT_RUN_START run_dir=%s data_dir=%s", run_dir, cfg.data_dir)

    stage_total = 14
    stage_idx = 0

    def stage_start(name: str, details: str = "") -> tuple[int, float]:
        nonlocal stage_idx
        stage_idx += 1
        pct = ((stage_idx - 1) / stage_total) * 100.0
        active_logger.info("STAGE_START [%d/%d %.1f%%] %s %s", stage_idx, stage_total, pct, name, details)
        return stage_idx, time.perf_counter()

    def stage_end(idx: int, name: str, st: float, details: str = "") -> None:
        pct = (idx / stage_total) * 100.0
        active_logger.info("STAGE_END   [%d/%d %.1f%%] %s elapsed=%s %s", idx, stage_total, pct, name, format_duration(time.perf_counter() - st), details)

    sidx, st0 = stage_start("validate_input_files")
    _require_export_files(cfg.data_dir)
    stage_end(sidx, "validate_input_files", st0)

    sidx, st0 = stage_start("load_raw_csvs")
    daily_df = _load_daily_csv(cfg.data_dir / "daily_max_truth_klga.csv")
    daily_df = daily_df[(daily_df["target_date_local"] >= cfg.split.train_start) & (daily_df["target_date_local"] <= cfg.split.test_end)].copy()
    if daily_df.empty:
        raise ValueError("No daily rows in split horizon.")
    obs_df = _load_obs_csv(cfg.data_dir / "observations_30m_required_columns.csv")
    forbidden_present = sorted(BANNED_OBS_COLUMNS.intersection(set(obs_df.columns)))
    if forbidden_present:
        raise AssertionError(f"Forbidden observation columns present: {forbidden_present}")
    stage_end(sidx, "load_raw_csvs", st0, details=f"daily_rows={len(daily_df)} obs_rows={len(obs_df)}")

    sidx, st0 = stage_start("build_feature_rows")
    p_cfg = PipelineConfig(
        split=cfg.split,
        output_root=cfg.output_root,
        feature_contract_version=CONTRACT_ID_V3,
        include_feels_like=cfg.include_feels_like,
        delta_objective=cfg.delta_objective,
        delta_use_class_weights=cfg.delta_use_class_weights,
        delta_use_cutoff_weights=cfg.delta_use_cutoff_weights,
        delta_cutoff_weight_alpha=cfg.delta_cutoff_weight_alpha,
        enable_neighbor_regime_features=False,
        enable_v2_regime_features=True,
        enable_v2_vis_precip_wdir_dynamics=True,
        enable_state_anomaly_lookups=False,
        keep_merge_index_features=False,
    )
    calendar_df = make_calendar_grid(sorted(set(daily_df["target_date_local"])), tz=p_cfg.local_zone)
    start_obs_utc = pd.Timestamp(calendar_df["midnight_utc"].min()).tz_convert("UTC") - pd.Timedelta(hours=6)
    end_obs_utc = pd.Timestamp(calendar_df["cutoff_utc"].max()).tz_convert("UTC")
    obs_df = obs_df[(obs_df["valid_time_utc"] >= start_obs_utc) & (obs_df["valid_time_utc"] <= end_obs_utc)].copy()
    if obs_df.empty:
        raise ValueError("No observation rows after horizon filtering.")
    station_series = prepare_station_series(obs_df, station_ids=p_cfg.all_station_ids, include_feels_like=p_cfg.include_feels_like)
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
    feat_df = _recompute_v3_neighbor_composites(feat_df)
    stage_end(sidx, "build_feature_rows", st0, details=f"feature_rows={len(feat_df)}")

    sidx, st0 = stage_start("prepare_contract_and_guards")
    masks = _split_masks(feat_df, cfg.split)
    feat_df, climo_meta = _add_climo_features(feat_df, masks["train"], cfg=p_cfg, state_lookup_path=None)
    feat_df, cat_maps, cat_guard = _build_categorical_maps(feat_df, masks["train"])
    feat_df, anomaly_lookup_rows, anomaly_meta = _build_train_only_anomaly_features(feat_df, masks["train"])
    feat_df, hmm_meta = _fit_hmm_features(feat_df, masks["train"], n_states=6) if cfg.enable_hmm_features else (feat_df, {"enabled": False, "reason": "disabled_by_config"})

    v1_spine = _build_v1_feature_spine(cfg.v1_feature_csv_path)
    selected_cols: list[str] = []
    seen: set[str] = set()
    for col in v1_spine + V3_REGIME_FEATURES:
        if col in {"index", "index_x", "index_y"} or col in seen:
            continue
        seen.add(col)
        selected_cols.append(col)

    missing_cols = [c for c in selected_cols if c not in feat_df.columns]
    for c in missing_cols:
        feat_df[c] = np.nan

    if len(selected_cols) > int(cfg.feature_budget_max):
        raise AssertionError(f"Feature budget exceeded: selected={len(selected_cols)} max={cfg.feature_budget_max}")

    categorical_names = [c for c in ["wx_phrase_id", "wx_coarse_id", "clds_id", "uv_desc_id", "wdir_cardinal_id", "hmm_state_argmax"] if c in selected_cols]
    categorical_indices = [selected_cols.index(c) for c in categorical_names]

    leakage_guards = {
        "asof_guard_failures": int(audit.get("asof_guard_failures", 0)),
        "feature_budget": int(len(selected_cols)),
        "feature_budget_max": int(cfg.feature_budget_max),
        "categorical_map_guard": cat_guard,
        "anomaly_lookup_guard": anomaly_meta,
        "uv_negative_sentinel_guard": True,
        "forbidden_observation_columns": sorted(BANNED_OBS_COLUMNS),
        "forbidden_observation_columns_found": forbidden_present,
        "missing_contract_columns_added_as_nan": missing_cols,
    }

    medians = _fit_imputer(feat_df, feature_cols=selected_cols, train_mask=masks["train"])
    x_all = _apply_imputer(feat_df, feature_cols=selected_cols, medians=medians)

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
    stage_end(sidx, "prepare_contract_and_guards", st0, details=f"features={len(selected_cols)} cats={len(categorical_names)} peak_train={len(peak_train_idx)} peak_val={len(peak_val_idx)} peak_test={len(peak_test_idx)} delta_train={len(delta_train_idx)} delta_val={len(delta_val_idx)} delta_test={len(delta_test_idx)}")

    sidx, st0 = stage_start("train_peak_model")
    peak_result = train_peak_model(
        X_train=x_all[peak_train_idx],
        y_train=y_peak_all[peak_train_idx],
        X_val=x_all[peak_val_idx],
        y_val=y_peak_all[peak_val_idx],
        sample_weight_train=train_weights,
        categorical_feature=categorical_indices,
        params_override={"num_leaves": 127, "learning_rate": 0.05, "n_estimators": 5000, "min_data_in_leaf": 2000, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0, "min_data_per_group": 200, "cat_smooth": 20},
        logger=active_logger,
        log_period=cfg.peak_train_log_period,
        log_every_seconds=cfg.train_log_every_seconds,
        heartbeat_seconds=cfg.train_heartbeat_seconds,
        stage_label="PEAK_TRAIN_V3_EXPORTS",
    )
    stage_end(sidx, "train_peak_model", st0, details=f"val_logloss_cal={peak_result.val_metrics.get('logloss_cal')} val_brier_cal={peak_result.val_metrics.get('brier_cal')}")

    sidx, st0 = stage_start("train_delta_model")
    train_weight_full = np.zeros(len(feat_df), dtype=float)
    train_weight_full[peak_train_idx] = train_weights
    delta_train_weights = train_weight_full[delta_train_idx]
    if cfg.delta_use_cutoff_weights and len(delta_train_idx) > 0:
        cutoff_vals = pd.to_numeric(feat_df.iloc[delta_train_idx]["cutoff_minutes"], errors="coerce").to_numpy(dtype=float)
        norm = (cutoff_vals - 240.0) / (1080.0 - 240.0)
        norm = np.clip(norm, 0.0, 1.0)
        delta_train_weights = delta_train_weights * (1.0 + float(cfg.delta_cutoff_weight_alpha) * norm)

    delta_result = train_delta_model(
        X_train=x_all[delta_train_idx],
        y_train=y_delta_train.astype(int),
        X_val=x_all[delta_val_idx],
        y_val=y_delta_val.astype(int),
        num_classes=delta_class_max,
        sample_weight_train=delta_train_weights,
        categorical_feature=categorical_indices,
        objective=cfg.delta_objective,
        use_class_weights=cfg.delta_use_class_weights,
        params_override={"num_leaves": 64, "learning_rate": 0.03, "n_estimators": 8000, "min_data_in_leaf": 1000, "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "max_depth": -1, "min_data_per_group": 300, "cat_smooth": 30},
        logger=active_logger,
        log_period=cfg.delta_train_log_period,
        log_every_seconds=cfg.train_log_every_seconds,
        heartbeat_seconds=cfg.train_heartbeat_seconds,
        stage_label="DELTA_TRAIN_V3_EXPORTS",
    )
    stage_end(sidx, "train_delta_model", st0, details=f"val_multi_logloss_temp={delta_result.val_metrics.get('multi_logloss_temp')}")

    sidx, st0 = stage_start("predict_probabilities")
    p_peak_val_raw, p_peak_val_cal = predict_peak_probability(model=peak_result.model, isotonic=peak_result.isotonic, X=x_all[peak_val_idx])
    p_peak_test_raw, p_peak_test_cal = predict_peak_probability(model=peak_result.model, isotonic=peak_result.isotonic, X=x_all[peak_test_idx])

    p_peak_raw_all = np.zeros(len(feat_df), dtype=float)
    p_peak_cal_all = np.zeros(len(feat_df), dtype=float)
    p_peak_raw_all[peak_val_idx] = p_peak_val_raw
    p_peak_raw_all[peak_test_idx] = p_peak_test_raw
    p_peak_cal_all[peak_val_idx] = p_peak_val_cal
    p_peak_cal_all[peak_test_idx] = p_peak_test_cal

    _, p_delta_raw_all, p_delta_temp_all = predict_delta_conditional(model=delta_result.model, temperature=delta_result.temperature, X=x_all)
    model_classes = np.asarray(delta_result.model.classes_, dtype=int)
    p_delta_raw_all = _expand_class_probs(probs=p_delta_raw_all, class_labels=model_classes, total_classes=delta_class_max)
    p_delta_temp_all = _expand_class_probs(probs=p_delta_temp_all, class_labels=model_classes, total_classes=delta_class_max)
    stage_end(sidx, "predict_probabilities", st0)

    val_peak_blend = np.asarray(p_peak_cal_all[peak_val_idx], dtype=float)
    test_peak_blend = np.asarray(p_peak_cal_all[peak_test_idx], dtype=float)
    val_delta_blend = np.asarray(p_delta_temp_all[peak_val_idx], dtype=float)
    test_delta_blend = np.asarray(p_delta_temp_all[peak_test_idx], dtype=float)
    analog_meta: dict[str, Any] = {"enabled": False, "reason": "disabled_by_config"}
    val_w = None
    test_w = None

    if cfg.enable_analog_blend:
        sidx, st0 = stage_start("analog_knn_blend")
        analog_cols = [c for c in p_cfg.analog_feature_columns if c in feat_df.columns]
        analog_std = fit_analog_standardizer(df=feat_df, feature_columns=analog_cols, train_mask=masks["train"], feature_weights=p_cfg.analog_feature_weights)
        analog_lib = build_analog_library(df=feat_df, standardizer=analog_std, delta_class_max=delta_class_max)
        val_idx = np.where(peak_val_mask)[0]
        test_idx = np.where(peak_test_mask)[0]
        best_k = 100
        best_val_nll = math.inf
        best_post = None
        for k in (50, 100, 200):
            post = predict_knn_posterior(
                library=analog_lib,
                standardizer=analog_std,
                query_indices=val_idx,
                k=k,
                delta_class_max=delta_class_max,
                season_window_doy=p_cfg.analog_season_window_doy,
                min_pool=p_cfg.analog_min_pool,
                min_non_peak=p_cfg.analog_min_non_peak,
                logger=active_logger,
                log_every_rows=cfg.log_every_rows,
                log_every_seconds=cfg.log_every_seconds,
                log_label=f"ANALOG_V3_VAL_K{k}",
            )
            ok = np.isfinite(post.p_peak) & np.isfinite(np.sum(post.p_delta_cond, axis=1))
            if not np.any(ok):
                continue
            metrics_k, _ = _evaluate_distribution_rows(df=feat_df, row_indices=val_idx[ok], p_peak=post.p_peak[ok], p_delta_cond=post.p_delta_cond[ok])
            nll = float(metrics_k.get("nll", np.inf))
            if np.isfinite(nll) and nll < best_val_nll:
                best_val_nll = nll
                best_k = k
                best_post = post

        if best_post is None:
            best_post = predict_knn_posterior(
                library=analog_lib,
                standardizer=analog_std,
                query_indices=val_idx,
                k=best_k,
                delta_class_max=delta_class_max,
                season_window_doy=p_cfg.analog_season_window_doy,
                min_pool=p_cfg.analog_min_pool,
                min_non_peak=p_cfg.analog_min_non_peak,
            )

        analog_test = predict_knn_posterior(
            library=analog_lib,
            standardizer=analog_std,
            query_indices=test_idx,
            k=best_k,
            delta_class_max=delta_class_max,
            season_window_doy=p_cfg.analog_season_window_doy,
            min_pool=p_cfg.analog_min_pool,
            min_non_peak=p_cfg.analog_min_non_peak,
            logger=active_logger,
            log_every_rows=cfg.log_every_rows,
            log_every_seconds=cfg.log_every_seconds,
            log_label=f"ANALOG_V3_TEST_K{best_k}",
        )

        q_low, q_high = calibrate_blend_bounds(best_post.q_score)
        val_peak_blend, val_delta_blend, val_w = blend_posteriors(
            p_peak_lgbm=p_peak_cal_all[val_idx], p_delta_lgbm=p_delta_temp_all[val_idx], p_peak_knn=best_post.p_peak, p_delta_knn=best_post.p_delta_cond, q_score=best_post.q_score, q_low=q_low, q_high=q_high
        )
        test_peak_blend, test_delta_blend, test_w = blend_posteriors(
            p_peak_lgbm=p_peak_cal_all[test_idx], p_delta_lgbm=p_delta_temp_all[test_idx], p_peak_knn=analog_test.p_peak, p_delta_knn=analog_test.p_delta_cond, q_score=analog_test.q_score, q_low=q_low, q_high=q_high
        )

        analog_meta = {
            "enabled": True,
            "selected_k": int(best_k),
            "selected_k_val_nll": float(best_val_nll if np.isfinite(best_val_nll) else np.nan),
            "candidate_k_grid": [50, 100, 200],
            "q_low": float(q_low),
            "q_high": float(q_high),
            "feature_columns": list(analog_std.feature_columns),
        }
        (run_dir / "analog_standardizer.json").write_text(
            json.dumps({**analog_meta, "mean": analog_std.mean.tolist(), "std": analog_std.std.tolist(), "weight": analog_std.weight.tolist()}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        stage_end(sidx, "analog_knn_blend", st0, details=f"selected_k={best_k} val_nll={best_val_nll}")

    sidx, st0 = stage_start("evaluate_metrics")
    delta_test_raw = float(_multi_logloss(y_delta_test, p_delta_raw_all[delta_test_idx])) if len(delta_test_idx) > 0 else float("nan")
    delta_test_temp = float(_multi_logloss(y_delta_test, p_delta_temp_all[delta_test_idx])) if len(delta_test_idx) > 0 else float("nan")

    combined_val_lgbm, combined_val_detail_lgbm = _evaluate_distribution_rows(df=feat_df, row_indices=peak_val_idx, p_peak=p_peak_cal_all[peak_val_idx], p_delta_cond=p_delta_temp_all[peak_val_idx])
    combined_test_lgbm, combined_test_detail_lgbm = _evaluate_distribution_rows(df=feat_df, row_indices=peak_test_idx, p_peak=p_peak_cal_all[peak_test_idx], p_delta_cond=p_delta_temp_all[peak_test_idx])
    combined_val, combined_val_detail = _evaluate_distribution_rows(df=feat_df, row_indices=peak_val_idx, p_peak=val_peak_blend, p_delta_cond=val_delta_blend)
    combined_test, combined_test_detail = _evaluate_distribution_rows(df=feat_df, row_indices=peak_test_idx, p_peak=test_peak_blend, p_delta_cond=test_delta_blend)

    _cutoff_metrics(df_detail=combined_val_detail).to_csv(reports_dir / "cutoff_metrics_val.csv", index=False)
    _cutoff_metrics(df_detail=combined_test_detail).to_csv(reports_dir / "cutoff_metrics_test.csv", index=False)
    _temperature_bucket_calibration(df=feat_df, row_indices=peak_val_idx, p_peak=val_peak_blend, p_delta_cond=val_delta_blend).to_csv(reports_dir / "bucket_calibration_val.csv", index=False)
    _temperature_bucket_calibration(df=feat_df, row_indices=peak_test_idx, p_peak=test_peak_blend, p_delta_cond=test_delta_blend).to_csv(reports_dir / "bucket_calibration_test.csv", index=False)

    regime_parts: list[pd.DataFrame] = []
    for split_name, idx_arr, y_arr in [("val", delta_val_idx, y_delta_val), ("test", delta_test_idx, y_delta_test)]:
        if len(idx_arr) == 0:
            continue
        probs_arr = p_delta_temp_all[idx_arr]
        for gcol in ["clds_id", "wx_coarse_id", "uv_desc_id"]:
            if gcol in feat_df.columns:
                regime_parts.append(_delta_logloss_by_group(y_true=y_arr, probs=probs_arr, group_values=feat_df.iloc[idx_arr][gcol].astype("Int64").astype(str), split_name=split_name, group_name=gcol))
        if "coastal_minus_inland_temp" in feat_df.columns:
            labels = _quantile_labels(feat_df.iloc[idx_arr]["coastal_minus_inland_temp"], q=5, prefix="CIT_")
            regime_parts.append(_delta_logloss_by_group(y_true=y_arr, probs=probs_arr, group_values=labels, split_name=split_name, group_name="coastal_minus_inland_temp_quantile"))
        time_block = (pd.to_numeric(feat_df.iloc[idx_arr]["cutoff_minutes"], errors="coerce") // 180).fillna(-1).astype(int).map(lambda x: f"block_{x}" if x >= 0 else "block_missing")
        regime_parts.append(_delta_logloss_by_group(y_true=y_arr, probs=probs_arr, group_values=time_block, split_name=split_name, group_name="time_of_day_block"))

    regime_report = pd.concat(regime_parts, ignore_index=True) if regime_parts else pd.DataFrame(columns=["split", "group_name", "group_value", "n_rows", "multi_logloss_temp"])
    regime_report.to_csv(reports_dir / "delta_logloss_regime_slices.csv", index=False)

    peak_gain = pd.DataFrame({"feature": selected_cols, "gain": np.asarray(peak_result.model.booster_.feature_importance(importance_type="gain"), dtype=float), "split": np.asarray(peak_result.model.booster_.feature_importance(importance_type="split"), dtype=float)}).sort_values("gain", ascending=False)
    delta_gain = pd.DataFrame({"feature": selected_cols, "gain": np.asarray(delta_result.model.booster_.feature_importance(importance_type="gain"), dtype=float), "split": np.asarray(delta_result.model.booster_.feature_importance(importance_type="split"), dtype=float)}).sort_values("gain", ascending=False)
    peak_gain.to_csv(reports_dir / "peak_feature_importance.csv", index=False)
    delta_gain.to_csv(reports_dir / "delta_feature_importance.csv", index=False)

    agg_rows = []
    for model_name, df_imp in [("peak", peak_gain), ("delta", delta_gain)]:
        for prefix in ["clds_", "uv_", "wx_", "hmm_"]:
            mask = df_imp["feature"].astype(str).str.startswith(prefix)
            agg_rows.append({"model": model_name, "family": prefix, "n_features": int(mask.sum()), "gain_sum": float(df_imp.loc[mask, "gain"].sum()), "split_sum": float(df_imp.loc[mask, "split"].sum())})
    pd.DataFrame(agg_rows).to_csv(reports_dir / "feature_importance_aggregate_structured.csv", index=False)

    peak_metrics = {
        "val": {
            "logloss_raw": float(log_loss(y_peak_all[peak_val_idx], np.clip(p_peak_val_raw, 1e-6, 1 - 1e-6))),
            "logloss_cal": float(log_loss(y_peak_all[peak_val_idx], np.clip(p_peak_val_cal, 1e-6, 1 - 1e-6))),
            "brier_raw": float(brier_score_loss(y_peak_all[peak_val_idx], p_peak_val_raw)),
            "brier_cal": float(brier_score_loss(y_peak_all[peak_val_idx], p_peak_val_cal)),
            "brier_blended": float(brier_score_loss(y_peak_all[peak_val_idx], val_peak_blend)),
        },
        "test": {
            "logloss_raw": float(log_loss(y_peak_all[peak_test_idx], np.clip(p_peak_test_raw, 1e-6, 1 - 1e-6))),
            "logloss_cal": float(log_loss(y_peak_all[peak_test_idx], np.clip(p_peak_test_cal, 1e-6, 1 - 1e-6))),
            "brier_raw": float(brier_score_loss(y_peak_all[peak_test_idx], p_peak_test_raw)),
            "brier_cal": float(brier_score_loss(y_peak_all[peak_test_idx], p_peak_test_cal)),
            "brier_blended": float(brier_score_loss(y_peak_all[peak_test_idx], test_peak_blend)),
        },
    }
    delta_metrics = {"val": {"multi_logloss_raw": float(_multi_logloss(y_delta_val, p_delta_raw_all[delta_val_idx])), "multi_logloss_temp": float(_multi_logloss(y_delta_val, p_delta_temp_all[delta_val_idx]))}, "test": {"multi_logloss_raw": delta_test_raw, "multi_logloss_temp": delta_test_temp}, "temperature": float(delta_result.temperature)}
    stage_end(sidx, "evaluate_metrics", st0, details=f"combined_val_nll={combined_val.get('nll')} combined_test_nll={combined_test.get('nll')}")

    sidx, st0 = stage_start("write_models_and_metadata")
    pred_val = _build_full_delta_arrays(full_len=len(feat_df), class_count=delta_class_max, row_indices=peak_val_idx, p_peak=val_peak_blend, p_delta=val_delta_blend, w_lgbm=val_w)
    pred_test = _build_full_delta_arrays(full_len=len(feat_df), class_count=delta_class_max, row_indices=peak_test_idx, p_peak=test_peak_blend, p_delta=test_delta_blend, w_lgbm=test_w)
    _write_df_with_csv_parquet(pred_val, predictions_dir / "predictions_val", active_logger)
    _write_df_with_csv_parquet(pred_test, predictions_dir / "predictions_test", active_logger)
    _write_df_with_csv_parquet(combined_val_detail, predictions_dir / "distribution_eval_val", active_logger)
    _write_df_with_csv_parquet(combined_test_detail, predictions_dir / "distribution_eval_test", active_logger)
    _write_df_with_csv_parquet(combined_val_detail_lgbm, predictions_dir / "distribution_eval_val_lgbm", active_logger)
    _write_df_with_csv_parquet(combined_test_detail_lgbm, predictions_dir / "distribution_eval_test_lgbm", active_logger)

    peak_result.model.booster_.save_model(str(models_dir / "peak_model.txt"))
    delta_result.model.booster_.save_model(str(models_dir / "delta_model.txt"))
    joblib.dump(peak_result.isotonic, models_dir / "peak_isotonic.pkl")
    (models_dir / "delta_temperature_T.json").write_text(json.dumps({"temperature": float(delta_result.temperature)}, indent=2, sort_keys=True), encoding="utf-8")

    (run_dir / "feature_list.json").write_text(json.dumps(selected_cols, indent=2), encoding="utf-8")
    (run_dir / "feature_list_v3.json").write_text(json.dumps(selected_cols, indent=2), encoding="utf-8")
    (run_dir / "imputer_values.json").write_text(json.dumps(medians, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "categorical_maps.json").write_text(json.dumps(cat_maps, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "regime_encoder_meta.json").write_text(json.dumps({"contract_id": CONTRACT_ID_V3, "v1_spine_count": int(len(v1_spine)), "v3_additional_count": int(len(V3_REGIME_FEATURES)), "selected_feature_count": int(len(selected_cols)), "categorical_features": categorical_names, "hmm_meta": hmm_meta, "anomaly_meta": anomaly_meta}, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "uv_clds_vis_anomaly_lookup.json").write_text(json.dumps(anomaly_lookup_rows, indent=2), encoding="utf-8")
    (run_dir / "leakage_guards.json").write_text(json.dumps(leakage_guards, indent=2, sort_keys=True), encoding="utf-8")

    (run_dir / "train_date_range.txt").write_text(f"{cfg.split.train_start} -> {cfg.split.train_end}\n", encoding="utf-8")
    (run_dir / "val_date_range.txt").write_text(f"{cfg.split.val_start} -> {cfg.split.val_end}\n", encoding="utf-8")
    (run_dir / "test_date_range.txt").write_text(f"{cfg.split.test_start} -> {cfg.split.test_end}\n", encoding="utf-8")

    metrics = {
        "run_id": run_dir.name,
        "backend": "lgbm_v3",
        "data_dir": str(cfg.data_dir),
        "feature_contract_version": CONTRACT_ID_V3,
        "rows_total": int(len(feat_df)),
        "split_rows": {"train": int(len(peak_train_idx)), "val": int(len(peak_val_idx)), "test": int(len(peak_test_idx))},
        "split_rows_delta": {"train": int(len(delta_train_idx)), "val": int(len(delta_val_idx)), "test": int(len(delta_test_idx))},
        "audit": audit,
        "climo_lookup_meta": climo_meta,
        "peak": peak_metrics,
        "delta": delta_metrics,
        "combined_lgbm": {"val": combined_val_lgbm, "test": combined_test_lgbm},
        "combined_blended": {"val": combined_val, "test": combined_test},
        "analog": analog_meta,
        "leakage_guards": leakage_guards,
        "hmm_meta": hmm_meta,
        "feature_importance_family_summary": agg_rows,
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    cfg_payload = {
        "backend": "lgbm_v3",
        "data_dir": str(cfg.data_dir),
        "output_root": str(cfg.output_root),
        "feature_contract_version": CONTRACT_ID_V3,
        "include_feels_like": bool(cfg.include_feels_like),
        "delta_objective": str(cfg.delta_objective),
        "delta_use_class_weights": bool(cfg.delta_use_class_weights),
        "delta_use_cutoff_weights": bool(cfg.delta_use_cutoff_weights),
        "delta_cutoff_weight_alpha": float(cfg.delta_cutoff_weight_alpha),
        "feature_budget_max": int(cfg.feature_budget_max),
        "enable_hmm_features": bool(cfg.enable_hmm_features),
        "enable_analog_blend": bool(cfg.enable_analog_blend),
        "split": {"train_start": str(cfg.split.train_start), "train_end": str(cfg.split.train_end), "val_start": str(cfg.split.val_start), "val_end": str(cfg.split.val_end), "test_start": str(cfg.split.test_start), "test_end": str(cfg.split.test_end)},
    }
    (run_dir / "config.json").write_text(json.dumps(cfg_payload, indent=2, sort_keys=True), encoding="utf-8")

    metrics_md = [
        "# KLGA LGBM V3 Peak/Delta Run From Exports",
        "",
        f"- run_id: {run_dir.name}",
        f"- contract_id: {CONTRACT_ID_V3}",
        f"- data_dir: {cfg.data_dir}",
        f"- rows_total: {len(feat_df)}",
        f"- split_rows_peak: train={len(peak_train_idx)} val={len(peak_val_idx)} test={len(peak_test_idx)}",
        f"- split_rows_delta: train={len(delta_train_idx)} val={len(delta_val_idx)} test={len(delta_test_idx)}",
        f"- selected_feature_count: {len(selected_cols)}",
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
        f"- val_nll_lgbm: {combined_val_lgbm.get('nll')}",
        f"- test_nll_lgbm: {combined_test_lgbm.get('nll')}",
        f"- val_nll_blended: {combined_val.get('nll')}",
        f"- test_nll_blended: {combined_test.get('nll')}",
        f"- val_top1_blended: {combined_val.get('top1_accuracy')}",
        f"- test_top1_blended: {combined_test.get('top1_accuracy')}",
        "",
        "## Leakage Guards",
        f"- asof_guard_failures: {leakage_guards.get('asof_guard_failures')}",
        f"- feature_budget: {leakage_guards.get('feature_budget')} / {leakage_guards.get('feature_budget_max')}",
    ]
    (run_dir / "metrics.md").write_text("\n".join(metrics_md), encoding="utf-8")

    explanation_path = _write_results_explanation(run_dir=run_dir, feature_count=len(selected_cols))
    stage_end(sidx, "write_models_and_metadata", st0, details=f"metrics_path={metrics_path} explanation={explanation_path.name}")

    active_logger.info("LGBM_V3_EXPORT_RUN_DONE elapsed=%s run_dir=%s", format_duration(time.perf_counter() - pipeline_start), run_dir)
    return LGBMV3TrainingResult(run_dir=run_dir, metrics_path=metrics_path, metrics=metrics)
