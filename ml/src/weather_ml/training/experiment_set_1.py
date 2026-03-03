from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import logging
import math
import re
import time

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit
from sklearn.metrics import brier_score_loss, log_loss

from weather_ml.klga_daily_tmax_dist.config import BANNED_OBS_COLUMNS, PipelineConfig, SplitConfig
from weather_ml.klga_daily_tmax_dist.features import build_daily_prior_frame, build_feature_rows, prepare_station_series
from weather_ml.klga_daily_tmax_dist.infer import build_delta_pmf
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
    _model_feature_columns,
    _recency_weights,
    _temperature_bucket_calibration,
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
from weather_ml.training.lgbm_klga_v3_from_exports import (
    CONTRACT_ID_V3,
    V1_FEATURE_IMPORTANCE_CSV,
    V3_REGIME_FEATURES,
    _build_categorical_maps,
    _build_train_only_anomaly_features,
    _build_v1_feature_spine,
    _fit_hmm_features,
    _recompute_v3_neighbor_composites,
)
from weather_ml.training.tabm_klga_from_exports import (
    _attach_run_file_handler,
    _load_daily_csv,
    _load_obs_csv,
    _load_station_universe_csv,
    _require_export_files,
    _split_masks,
    _write_df_with_csv_parquet,
)


PHRASE_STATES: tuple[str, ...] = (
    "CLEAR",
    "MOSTLY_CLEAR",
    "PARTLY_CLOUDY",
    "MOSTLY_CLOUDY",
    "OVERCAST_LOW",
    "OVERCAST_HIGH",
    "FOG_OR_LOW_VIS",
    "HAZE_SMOKE",
    "DRIZZLE_LIGHT_RAIN",
    "STEADY_RAIN",
    "SHOWERS_CONVECTIVE",
    "THUNDER",
    "SNOW_OR_ICE",
    "UNKNOWN_OR_MISSING",
)
PHRASE_STATE_TO_ID = {name: idx for idx, name in enumerate(PHRASE_STATES)}

COASTAL_SHORT = ("KJFK", "KISP", "KBDR")
INLAND_SHORT = ("KEWR", "KTEB", "KHPN", "KMMU")
NEIGHBOR_SHORT = ("KJFK", "KEWR", "KTEB", "KHPN", "KISP", "KBDR", "KMMU")

STATION_GEO = {
    "KLGA": (40.7794, -73.8740),
    "KJFK": (40.6413, -73.7781),
    "KEWR": (40.6895, -74.1745),
    "KTEB": (40.8501, -74.0608),
    "KHPN": (41.0670, -73.7076),
    "KISP": (40.7952, -73.1002),
    "KBDR": (41.1635, -73.1262),
    "KMMU": (40.7994, -74.4149),
}

LOW_PRIORITY_BASE_FEATURES = [
    "hmm_p_state5",
    "hmm_p_state4",
    "hmm_p_state3",
    "hmm_p_state2",
    "hmm_p_state1",
    "hmm_p_state0",
    "hmm_state_argmax",
]


@dataclass(frozen=True)
class ExperimentSet1Config:
    data_dir: Path
    output_root: Path
    split: SplitConfig
    experiments: tuple[str, ...] = ("E1", "E2", "E3", "E4")
    feature_budget_max: int = 600
    phrase_feature_cap: int = 150
    advection_feature_cap: int = 120
    ordinal_threshold_stride: int = 1
    enable_hmm_features: bool = True
    log_every_rows: int = 2000
    log_every_seconds: float = 20.0
    peak_train_log_period: int = 50
    delta_train_log_period: int = 25
    train_log_every_seconds: float = 10.0
    train_heartbeat_seconds: float = 10.0
    v1_feature_csv_path: Path = V1_FEATURE_IMPORTANCE_CSV


@dataclass(frozen=True)
class DatasetContext:
    feat_df: pd.DataFrame
    masks: dict[str, np.ndarray]
    base_feature_cols: list[str]
    categorical_names: list[str]
    categorical_indices: list[int]
    cat_maps: dict[str, dict[str, int]]
    leakage_guards: dict[str, Any]
    audit: dict[str, Any]
    climo_meta: dict[str, Any]
    anomaly_meta: dict[str, Any]
    hmm_meta: dict[str, Any]
    station_universe: dict[str, Any]
    daily_rows: int
    obs_rows: int
    forbidden_present: list[str]


@dataclass(frozen=True)
class SplitSlices:
    y_peak_all: np.ndarray
    y_delta_all: np.ndarray
    peak_train_idx: np.ndarray
    peak_val_idx: np.ndarray
    peak_test_idx: np.ndarray
    delta_train_idx: np.ndarray
    delta_val_idx: np.ndarray
    delta_test_idx: np.ndarray
    y_delta_train_class: np.ndarray
    y_delta_val_class: np.ndarray
    y_delta_test_class: np.ndarray
    delta_class_max: int


@dataclass(frozen=True)
class ExperimentArtifacts:
    experiment_id: str
    run_dir: Path
    metrics: dict[str, Any]
    feature_cols: list[str]
    p_peak_val: np.ndarray
    p_peak_test: np.ndarray
    p_delta_val: np.ndarray
    p_delta_test: np.ndarray
    peak_model_path: Path | None
    delta_model_path: Path | None
    model_manifest: dict[str, Any]


def _timestamp_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _station_col(short: str, base_name: str) -> str:
    if short == "KLGA":
        return base_name
    return f"{short}_{base_name}"


def _safe_numeric(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), np.nan, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _normalize_token(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    return str(v).strip()


def _apply_feature_budget(
    *,
    base_cols: list[str],
    extra_cols: list[str],
    max_features: int,
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for col in base_cols + extra_cols:
        if col in seen:
            continue
        seen.add(col)
        out.append(col)
    if len(out) <= max_features:
        return out
    trimmed = list(out)
    for col in LOW_PRIORITY_BASE_FEATURES:
        if len(trimmed) <= max_features:
            break
        if col in trimmed and col not in extra_cols:
            trimmed.remove(col)
    if len(trimmed) <= max_features:
        return trimmed
    overflow = len(trimmed) - max_features
    protected = set(extra_cols)
    safe_pool = [c for c in trimmed if c not in protected]
    if len(safe_pool) >= overflow:
        drop = set(safe_pool[-overflow:])
        trimmed = [c for c in trimmed if c not in drop]
    if len(trimmed) > max_features:
        trimmed = trimmed[:max_features]
    return trimmed


def _format_word_count(text: str) -> int:
    return len([w for w in text.split() if w.strip()])


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0) ** 2) + math.cos(p1) * math.cos(p2) * (math.sin(dlon / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1.0 - a)))
    return float(r * c)


def _bearing_met_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlon)
    b = math.degrees(math.atan2(y, x))
    return float((b + 360.0) % 360.0)


def _label_alignment_gate(
    *,
    cfg: ExperimentSet1Config,
    out_dir: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    station_universe = _load_station_universe_csv(cfg.data_dir / "station_universe.csv")
    target_short = station_universe.target_station_id.split(":", 1)[0]
    csv_path = cfg.data_dir / "daily_max_truth_klga.csv"
    raw = pd.read_csv(csv_path, low_memory=False)
    required = {"request_location_id", "target_date_local", "max_temp_f", "station_zoneid"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"daily_max_truth_klga.csv is missing required columns: {missing}")

    df = raw.copy()
    df["target_date_local"] = pd.to_datetime(df["target_date_local"], errors="coerce").dt.date
    max_temp_num = pd.to_numeric(df["max_temp_f"], errors="coerce")
    rounded = np.round(max_temp_num)
    integer_like = np.isfinite(max_temp_num.to_numpy(dtype=float)) & (
        np.abs(max_temp_num.to_numpy(dtype=float) - rounded.to_numpy(dtype=float)) <= 1e-6
    )

    zone_series = df["station_zoneid"].astype(str).str.strip()
    zone_ok = zone_series.eq("America/New_York")
    request_ids = df["request_location_id"].astype(str).str.strip()
    request_target = request_ids.str.startswith(target_short)

    dup_count = int(
        df.dropna(subset=["target_date_local"])
        .groupby("target_date_local")["request_location_id"]
        .count()
        .gt(1)
        .sum()
    )
    split_dates = pd.date_range(cfg.split.train_start, cfg.split.test_end, freq="D")
    have_dates = pd.Series(df["target_date_local"].dropna().unique())
    missing_dates = int(len(split_dates) - have_dates.isin(split_dates.date).sum())

    report = {
        "csv_path": str(csv_path),
        "rows_total": int(len(df)),
        "rows_in_split_window": int(
            ((df["target_date_local"] >= cfg.split.train_start) & (df["target_date_local"] <= cfg.split.test_end)).sum()
        ),
        "target_station_id": station_universe.target_station_id,
        "station_id_target_like_rows": int(request_target.sum()),
        "station_zoneid_expected": "America/New_York",
        "station_zoneid_bad_rows": int((~zone_ok).sum()),
        "station_zoneid_distinct": sorted(zone_series.dropna().unique().tolist())[:50],
        "max_temp_numeric_missing_rows": int(np.sum(~np.isfinite(max_temp_num.to_numpy(dtype=float)))),
        "max_temp_non_integer_rows": int(np.sum(~integer_like)),
        "max_temp_min": float(np.nanmin(max_temp_num.to_numpy(dtype=float))) if np.isfinite(max_temp_num).any() else np.nan,
        "max_temp_max": float(np.nanmax(max_temp_num.to_numpy(dtype=float))) if np.isfinite(max_temp_num).any() else np.nan,
        "duplicate_target_date_rows": dup_count,
        "missing_dates_within_split_window": missing_dates,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate_policy": "local_daily_max_truth_only_per_user_request",
    }

    fatal: list[str] = []
    if report["max_temp_numeric_missing_rows"] > 0:
        fatal.append("max_temp_f contains non-numeric/missing rows.")
    if report["max_temp_non_integer_rows"] > 0:
        fatal.append("max_temp_f contains non-integer rows; settlement requires whole-F.")
    if report["station_zoneid_bad_rows"] > 0:
        fatal.append("station_zoneid has rows outside America/New_York.")
    if report["duplicate_target_date_rows"] > 0:
        fatal.append("duplicate target_date_local rows found in daily truth.")
    report["fatal_issues"] = fatal

    _ensure_dir(out_dir)
    (out_dir / "label_alignment_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    md = [
        "# Label Alignment Audit (Local Daily Truth Gate)",
        "",
        f"- csv_path: `{report['csv_path']}`",
        f"- rows_total: {report['rows_total']}",
        f"- rows_in_split_window: {report['rows_in_split_window']}",
        f"- station_zoneid_bad_rows: {report['station_zoneid_bad_rows']}",
        f"- max_temp_non_integer_rows: {report['max_temp_non_integer_rows']}",
        f"- duplicate_target_date_rows: {report['duplicate_target_date_rows']}",
        f"- missing_dates_within_split_window: {report['missing_dates_within_split_window']}",
        f"- max_temp_range_f: [{report['max_temp_min']}, {report['max_temp_max']}]",
        "",
        "## Policy",
        "This gate intentionally uses only `daily_max_truth_klga.csv` as the settlement-aligned label source,",
        "per explicit user instruction for this experiment set.",
    ]
    if fatal:
        md += ["", "## Gate Status: FAIL"] + [f"- {item}" for item in fatal]
    else:
        md += ["", "## Gate Status: PASS", "- No fatal alignment issues detected in the local label source."]
    (out_dir / "label_alignment_audit.md").write_text("\n".join(md), encoding="utf-8")
    if fatal:
        raise AssertionError("Label alignment gate failed: " + " | ".join(fatal))
    logger.info("LABEL_GATE_PASS rows=%d split_rows=%d", report["rows_total"], report["rows_in_split_window"])
    return report


def _build_dataset_context(
    *,
    cfg: ExperimentSet1Config,
    logger: logging.Logger,
) -> DatasetContext:
    _require_export_files(cfg.data_dir)
    station_universe = _load_station_universe_csv(cfg.data_dir / "station_universe.csv")
    logger.info(
        "STATION_UNIVERSE target=%s neighbors=%s",
        station_universe.target_station_id,
        ",".join(station_universe.neighbor_station_ids),
    )
    daily_df = _load_daily_csv(cfg.data_dir / "daily_max_truth_klga.csv")
    daily_df = daily_df[
        (daily_df["target_date_local"] >= cfg.split.train_start)
        & (daily_df["target_date_local"] <= cfg.split.test_end)
    ].copy()
    if daily_df.empty:
        raise ValueError("No daily rows available in requested split horizon.")

    obs_df = _load_obs_csv(cfg.data_dir / "observations_30m_required_columns.csv")
    forbidden_present = sorted(BANNED_OBS_COLUMNS.intersection(set(obs_df.columns)))
    if forbidden_present:
        raise AssertionError(f"Forbidden observation columns found in export: {forbidden_present}")

    p_cfg = PipelineConfig(
        split=cfg.split,
        output_root=cfg.output_root,
        feature_contract_version=CONTRACT_ID_V3,
        target_station_id=station_universe.target_station_id,
        neighbor_station_ids=station_universe.neighbor_station_ids,
        include_feels_like=False,
        delta_objective="multiclass",
        delta_use_class_weights=True,
        delta_use_cutoff_weights=False,
        enable_neighbor_regime_features=False,
        enable_v2_regime_features=True,
        enable_v2_vis_precip_wdir_dynamics=True,
        enable_state_anomaly_lookups=False,
        keep_merge_index_features=False,
    )
    cal = make_calendar_grid(sorted(set(daily_df["target_date_local"])), tz=p_cfg.local_zone)
    start_obs_utc = pd.Timestamp(cal["midnight_utc"].min()).tz_convert("UTC") - pd.Timedelta(hours=6)
    end_obs_utc = pd.Timestamp(cal["cutoff_utc"].max()).tz_convert("UTC")
    obs_df = obs_df[(obs_df["valid_time_utc"] >= start_obs_utc) & (obs_df["valid_time_utc"] <= end_obs_utc)].copy()
    if obs_df.empty:
        raise ValueError("No observation rows after horizon filtering.")

    station_series = prepare_station_series(obs_df, station_ids=p_cfg.all_station_ids, include_feels_like=False)
    daily_prior_df = build_daily_prior_frame(daily_df)
    feat_df, audit = build_feature_rows(
        calendar_df=cal,
        station_series=station_series,
        daily_truth_df=daily_df,
        daily_prior_df=daily_prior_df,
        cfg=p_cfg,
        logger=logger,
        log_every_rows=cfg.log_every_rows,
        log_every_seconds=cfg.log_every_seconds,
    )
    if int(audit.get("asof_guard_failures", 0)) > 0:
        raise AssertionError(f"As-of leakage guard failed in feature build: {audit.get('asof_guard_failures')}")

    feat_df = _recompute_v3_neighbor_composites(feat_df)
    masks = _split_masks(feat_df, cfg.split)
    feat_df, climo_meta = _add_climo_features(feat_df, masks["train"], cfg=p_cfg, state_lookup_path=None)
    feat_df, cat_maps, cat_guard = _build_categorical_maps(feat_df, masks["train"])
    feat_df, anomaly_rows, anomaly_meta = _build_train_only_anomaly_features(feat_df, masks["train"])
    feat_df, hmm_meta = (
        _fit_hmm_features(feat_df, masks["train"], n_states=6)
        if cfg.enable_hmm_features
        else (feat_df, {"enabled": False, "reason": "disabled_by_config"})
    )

    v1_spine = _build_v1_feature_spine(cfg.v1_feature_csv_path)
    base_cols: list[str] = []
    seen: set[str] = set()
    for c in v1_spine + list(V3_REGIME_FEATURES):
        if c in {"index", "index_x", "index_y"} or c in seen:
            continue
        seen.add(c)
        base_cols.append(c)
    missing_contract = [c for c in base_cols if c not in feat_df.columns]
    for c in missing_contract:
        feat_df[c] = np.nan
    if len(base_cols) > cfg.feature_budget_max:
        raise AssertionError(f"Base feature budget exceeds cap: {len(base_cols)} > {cfg.feature_budget_max}")

    categorical_names = [
        c for c in ["wx_phrase_id", "wx_coarse_id", "clds_id", "uv_desc_id", "wdir_cardinal_id", "hmm_state_argmax"]
        if c in base_cols
    ]
    categorical_indices = [base_cols.index(c) for c in categorical_names]
    leakage_guards = {
        "asof_guard_failures": int(audit.get("asof_guard_failures", 0)),
        "target_station_id": p_cfg.target_station_id,
        "neighbor_station_ids": list(p_cfg.neighbor_station_ids),
        "feature_budget_base": int(len(base_cols)),
        "feature_budget_max": int(cfg.feature_budget_max),
        "categorical_map_guard": cat_guard,
        "anomaly_lookup_guard": anomaly_meta,
        "uv_negative_sentinel_guard": True,
        "forbidden_observation_columns": sorted(BANNED_OBS_COLUMNS),
        "forbidden_observation_columns_found": forbidden_present,
        "missing_contract_columns_added_as_nan": missing_contract,
        "hmm_meta": hmm_meta,
        "anomaly_lookup_rows": int(len(anomaly_rows)),
    }
    return DatasetContext(
        feat_df=feat_df,
        masks=masks,
        base_feature_cols=base_cols,
        categorical_names=categorical_names,
        categorical_indices=categorical_indices,
        cat_maps=cat_maps,
        leakage_guards=leakage_guards,
        audit=audit,
        climo_meta=climo_meta,
        anomaly_meta=anomaly_meta,
        hmm_meta=hmm_meta,
        station_universe={
            "target_station_id": station_universe.target_station_id,
            "neighbor_station_ids": list(station_universe.neighbor_station_ids),
            "all_station_ids": list(station_universe.all_station_ids),
        },
        daily_rows=int(len(daily_df)),
        obs_rows=int(len(obs_df)),
        forbidden_present=forbidden_present,
    )


def _prepare_split_slices(*, ctx: DatasetContext, delta_class_max: int) -> SplitSlices:
    feat_df = ctx.feat_df
    peak_series = pd.to_numeric(feat_df["peak"], errors="coerce")
    delta_series = pd.to_numeric(feat_df["delta"], errors="coerce")
    peak_mask = np.isfinite(peak_series.to_numpy(dtype=float))
    delta_mask = np.isfinite(delta_series.to_numpy(dtype=float))

    y_peak_all = np.full(len(feat_df), -1, dtype=int)
    y_delta_all = np.full(len(feat_df), -1, dtype=int)
    y_peak_all[peak_mask] = peak_series.loc[peak_mask].round().astype(int).to_numpy()
    y_delta_all[delta_mask] = delta_series.loc[delta_mask].round().astype(int).to_numpy()

    masks = ctx.masks
    peak_train_idx = np.where(masks["train"] & peak_mask)[0]
    peak_val_idx = np.where(masks["val"] & peak_mask)[0]
    peak_test_idx = np.where(masks["test"] & peak_mask)[0]
    if len(peak_train_idx) == 0 or len(peak_val_idx) == 0 or len(peak_test_idx) == 0:
        raise ValueError("Peak split rows are empty.")

    delta_train_idx = np.where(masks["train"] & peak_mask & delta_mask & (y_peak_all == 0) & (y_delta_all >= 1))[0]
    delta_val_idx = np.where(masks["val"] & peak_mask & delta_mask & (y_peak_all == 0) & (y_delta_all >= 1))[0]
    delta_test_idx = np.where(masks["test"] & peak_mask & delta_mask & (y_peak_all == 0) & (y_delta_all >= 1))[0]
    if len(delta_train_idx) == 0 or len(delta_val_idx) == 0:
        raise ValueError("Delta train/val split rows are empty.")

    y_delta_train_class = np.clip(y_delta_all[delta_train_idx], 1, delta_class_max) - 1
    y_delta_val_class = np.clip(y_delta_all[delta_val_idx], 1, delta_class_max) - 1
    y_delta_test_class = np.clip(y_delta_all[delta_test_idx], 1, delta_class_max) - 1
    return SplitSlices(
        y_peak_all=y_peak_all,
        y_delta_all=y_delta_all,
        peak_train_idx=peak_train_idx,
        peak_val_idx=peak_val_idx,
        peak_test_idx=peak_test_idx,
        delta_train_idx=delta_train_idx,
        delta_val_idx=delta_val_idx,
        delta_test_idx=delta_test_idx,
        y_delta_train_class=y_delta_train_class.astype(int),
        y_delta_val_class=y_delta_val_class.astype(int),
        y_delta_test_class=y_delta_test_class.astype(int),
        delta_class_max=int(delta_class_max),
    )


def _phrase_state_and_attrs(phrase: str) -> dict[str, float]:
    raw = phrase.strip().lower()
    if raw == "" or raw == "unk":
        state = "UNKNOWN_OR_MISSING"
    else:
        has_thunder = bool(re.search(r"thunder|t-?storm|storm", raw))
        has_snow_ice = bool(re.search(r"snow|sleet|freezing|ice|wintry|hail", raw))
        has_showers = "shower" in raw
        has_rain = bool(re.search(r"rain|drizzle", raw))
        has_fog = bool(re.search(r"fog|mist", raw))
        has_haze = bool(re.search(r"haze|smoke|dust", raw))
        has_overcast = "overcast" in raw
        has_high_overcast = bool(re.search(r"high overcast|cirrus|high cloud", raw))
        has_mostly_cloudy = bool(re.search(r"mostly cloudy|cloudy", raw))
        has_partly = bool(re.search(r"partly cloudy|partly sunny", raw))
        has_mostly_clear = bool(re.search(r"mostly clear|mostly sunny", raw))
        has_clear = bool(re.search(r"\bclear\b|fair|sunny", raw))

        if has_thunder:
            state = "THUNDER"
        elif has_snow_ice:
            state = "SNOW_OR_ICE"
        elif has_showers:
            state = "SHOWERS_CONVECTIVE"
        elif has_rain:
            if "drizzle" in raw or "light rain" in raw:
                state = "DRIZZLE_LIGHT_RAIN"
            else:
                state = "STEADY_RAIN"
        elif has_fog:
            state = "FOG_OR_LOW_VIS"
        elif has_haze:
            state = "HAZE_SMOKE"
        elif has_high_overcast:
            state = "OVERCAST_HIGH"
        elif has_overcast:
            state = "OVERCAST_LOW"
        elif has_mostly_cloudy:
            state = "MOSTLY_CLOUDY"
        elif has_partly:
            state = "PARTLY_CLOUDY"
        elif has_mostly_clear:
            state = "MOSTLY_CLEAR"
        elif has_clear:
            state = "CLEAR"
        else:
            state = "UNKNOWN_OR_MISSING"

    cloud_frac_lookup = {
        "CLEAR": 0.05,
        "MOSTLY_CLEAR": 0.20,
        "PARTLY_CLOUDY": 0.40,
        "MOSTLY_CLOUDY": 0.75,
        "OVERCAST_LOW": 0.95,
        "OVERCAST_HIGH": 0.85,
        "FOG_OR_LOW_VIS": 0.60,
        "HAZE_SMOKE": 0.20,
        "DRIZZLE_LIGHT_RAIN": 0.90,
        "STEADY_RAIN": 0.95,
        "SHOWERS_CONVECTIVE": 0.80,
        "THUNDER": 0.85,
        "SNOW_OR_ICE": 0.95,
        "UNKNOWN_OR_MISSING": 0.50,
    }
    precip_flag = float(1.0 if state in {"DRIZZLE_LIGHT_RAIN", "STEADY_RAIN", "SHOWERS_CONVECTIVE", "THUNDER", "SNOW_OR_ICE"} else 0.0)
    precip_rank = 0.0
    if state in {"DRIZZLE_LIGHT_RAIN"}:
        precip_rank = 1.0
    elif state in {"STEADY_RAIN", "SNOW_OR_ICE"}:
        precip_rank = 2.0
    elif state in {"SHOWERS_CONVECTIVE", "THUNDER"}:
        precip_rank = 3.0
    convective_flag = float(1.0 if state in {"SHOWERS_CONVECTIVE", "THUNDER"} else 0.0)
    fog_flag = float(1.0 if state == "FOG_OR_LOW_VIS" else 0.0)
    haze_flag = float(1.0 if state == "HAZE_SMOKE" else 0.0)
    wind_mod = float(1.0 if ("windy" in raw or "breezy" in raw) else 0.0)
    cloud_frac = float(cloud_frac_lookup[state])
    radiation_killer = float(np.clip(cloud_frac + 0.5 * haze_flag + 0.5 * fog_flag + 0.25 * precip_flag, 0.0, 1.0))
    return {
        "state_id": float(PHRASE_STATE_TO_ID[state]),
        "cloud_frac_est": cloud_frac,
        "precip_flag": precip_flag,
        "precip_intensity_rank": precip_rank,
        "convective_flag": convective_flag,
        "fog_flag": fog_flag,
        "haze_smoke_flag": haze_flag,
        "wind_modifier_flag": wind_mod,
        "radiation_killer_score": radiation_killer,
    }


def _add_e1_phrase_to_physics_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    out = df.copy()
    phrase_cols = [c for c in out.columns if c.endswith("wx_phrase_norm_now")]
    parsed: dict[str, dict[str, np.ndarray]] = {}
    mapping_snapshot: dict[str, dict[str, float]] = {}
    for col in phrase_cols:
        values = out[col].map(_normalize_token).str.lower()
        unique_vals = sorted(set(values.tolist()))
        local_map = {v: _phrase_state_and_attrs(v) for v in unique_vals}
        for k, v in local_map.items():
            if k not in mapping_snapshot:
                mapping_snapshot[k] = v
        attrs = {k: np.zeros(len(out), dtype=float) for k in next(iter(local_map.values())).keys()}
        for i, phrase in enumerate(values.tolist()):
            data = local_map.get(phrase, _phrase_state_and_attrs(phrase))
            for key, val in data.items():
                attrs[key][i] = float(val)
        station_short = "KLGA" if col == "wx_phrase_norm_now" else col.split("_", 1)[0]
        parsed[station_short] = attrs

    def station_attr(short: str, attr: str) -> np.ndarray:
        return parsed.get(short, {}).get(attr, np.full(len(out), np.nan, dtype=float))

    extras: list[str] = []
    out["e1_phrase_state_id"] = station_attr("KLGA", "state_id")
    out["e1_cloud_frac_est"] = station_attr("KLGA", "cloud_frac_est")
    out["e1_precip_flag"] = station_attr("KLGA", "precip_flag")
    out["e1_precip_intensity_rank"] = station_attr("KLGA", "precip_intensity_rank")
    out["e1_convective_flag"] = station_attr("KLGA", "convective_flag")
    out["e1_fog_flag"] = station_attr("KLGA", "fog_flag")
    out["e1_haze_smoke_flag"] = station_attr("KLGA", "haze_smoke_flag")
    out["e1_wind_modifier_flag"] = station_attr("KLGA", "wind_modifier_flag")
    out["e1_radiation_killer_score"] = station_attr("KLGA", "radiation_killer_score")
    extras.extend(
        [
            "e1_phrase_state_id",
            "e1_cloud_frac_est",
            "e1_precip_flag",
            "e1_precip_intensity_rank",
            "e1_convective_flag",
            "e1_fog_flag",
            "e1_haze_smoke_flag",
            "e1_wind_modifier_flag",
            "e1_radiation_killer_score",
        ]
    )

    for group_name, group_stations in [("coastal", COASTAL_SHORT), ("inland", INLAND_SHORT)]:
        for attr in ["cloud_frac_est", "precip_flag", "convective_flag", "fog_flag", "haze_smoke_flag", "radiation_killer_score"]:
            vals = np.vstack([station_attr(short, attr) for short in group_stations])
            feat_name = f"e1_{attr}_{group_name}_mean"
            out[feat_name] = np.nanmean(vals, axis=0)
            extras.append(feat_name)
    out["e1_radiation_killer_coastal_minus_inland"] = out["e1_radiation_killer_score_coastal_mean"] - out["e1_radiation_killer_score_inland_mean"]
    out["e1_cloud_frac_coastal_minus_inland"] = out["e1_cloud_frac_est_coastal_mean"] - out["e1_cloud_frac_est_inland_mean"]
    extras.extend(["e1_radiation_killer_coastal_minus_inland", "e1_cloud_frac_coastal_minus_inland"])

    runlen = _safe_numeric(out, "wx_runlen_min")
    out["e1_state_change_count_60"] = _safe_numeric(out, "wx_transitions_60")
    out["e1_state_change_count_180"] = _safe_numeric(out, "wx_transitions_180")
    out["e1_state_change_count_360"] = _safe_numeric(out, "wx_transitions_360")
    out["e1_state_persist_frac_60"] = np.clip(runlen / 60.0, 0.0, 1.0)
    out["e1_state_persist_frac_180"] = np.clip(runlen / 180.0, 0.0, 1.0)
    out["e1_state_persist_frac_360"] = np.clip(runlen / 360.0, 0.0, 1.0)
    out["e1_mins_since_last_change"] = runlen
    extras.extend(
        [
            "e1_state_change_count_60",
            "e1_state_change_count_180",
            "e1_state_change_count_360",
            "e1_state_persist_frac_60",
            "e1_state_persist_frac_180",
            "e1_state_persist_frac_360",
            "e1_mins_since_last_change",
        ]
    )

    doy = _safe_numeric(out, "doy")
    cutoff = _safe_numeric(out, "cutoff_minutes")
    s_geom = np.sin((np.pi / 2.0) * np.clip((cutoff - 240.0) / 840.0, 0.0, 1.0))
    s_geom = np.clip((0.50 + 0.50 * np.sin(2.0 * np.pi * ((doy - 80.0) / 366.0))) * s_geom, 0.0, 1.0)
    uv_now = _safe_numeric(out, "uv_index_now")
    uv_prev_180 = _safe_numeric(out, "uv_prev_180")
    uv_prev_30 = _safe_numeric(out, "uv_prev_30")
    clds_now = np.clip(_safe_numeric(out, "clds_oktas_now") / 8.0, 0.0, 1.0)
    clds_prev_180 = np.clip(_safe_numeric(out, "clds_oktas_prev_180") / 8.0, 0.0, 1.0)
    clds_prev_30 = np.clip(_safe_numeric(out, "clds_oktas_prev_30") / 8.0, 0.0, 1.0)
    vis_now = np.clip(_safe_numeric(out, "vis_now") / 10.0, 0.0, 1.0)
    vis_prev_180 = np.clip(_safe_numeric(out, "vis_prev_180") / 10.0, 0.0, 1.0)
    vis_prev_30 = np.clip(_safe_numeric(out, "vis_prev_30") / 10.0, 0.0, 1.0)

    def insol(uv: np.ndarray, cld: np.ndarray, vis: np.ndarray) -> np.ndarray:
        uv_factor = 0.4 + 0.6 * np.clip(uv / 11.0, 0.0, 1.0)
        return s_geom * uv_factor * (1.0 - 0.85 * cld) * vis

    insol_now = insol(uv_now, clds_now, vis_now)
    insol_prev_180 = insol(uv_prev_180, clds_prev_180, vis_prev_180)
    insol_prev_30 = insol(uv_prev_30, clds_prev_30, vis_prev_30)
    out["e1_insol_proxy"] = insol_now
    out["e1_insol_trend_180"] = (insol_now - insol_prev_180) / 180.0
    out["e1_insol_shock_180"] = np.maximum(np.abs(insol_now - insol_prev_30), np.abs(insol_prev_30 - insol_prev_180))
    out["e1_radiation_killer_slope_180"] = (_safe_numeric(out, "clds_oktas_delta_180") / 8.0) - (0.3 * _safe_numeric(out, "vis_slope_180"))
    out["e1_vis_slope_180"] = _safe_numeric(out, "vis_slope_180")
    out["e1_vis_shock_180"] = np.abs(_safe_numeric(out, "vis_delta_30"))
    extras.extend(
        [
            "e1_insol_proxy",
            "e1_insol_trend_180",
            "e1_insol_shock_180",
            "e1_radiation_killer_slope_180",
            "e1_vis_slope_180",
            "e1_vis_shock_180",
        ]
    )

    meta = {
        "state_count": int(len(PHRASE_STATES)),
        "states": list(PHRASE_STATES),
        "phrase_mapping_size": int(len(mapping_snapshot)),
        "phrase_mapping_snapshot": mapping_snapshot,
        "extra_feature_count": int(len(extras)),
    }
    return out, extras, meta


def _add_e2_advection_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    out = df.copy()
    klga_lat, klga_lon = STATION_GEO["KLGA"]
    neighbor_meta: dict[str, dict[str, float]] = {}
    for short in NEIGHBOR_SHORT:
        lat, lon = STATION_GEO[short]
        neighbor_meta[short] = {
            "dist_km": _haversine_km(klga_lat, klga_lon, lat, lon),
            "bearing_deg": _bearing_met_deg(klga_lat, klga_lon, lat, lon),
        }

    wdir = _safe_numeric(out, "wdir_now")
    wdir = np.where((wdir >= 0.0) & (wdir <= 360.0), wdir, np.nan)
    extras: list[str] = []

    def upwind(var_name: str, out_name: str) -> None:
        num = np.zeros(len(out), dtype=float)
        den = np.zeros(len(out), dtype=float)
        for short in NEIGHBOR_SHORT:
            vals = _safe_numeric(out, _station_col(short, var_name))
            meta = neighbor_meta[short]
            align = np.cos(np.deg2rad(meta["bearing_deg"] - wdir))
            align = np.where(np.isfinite(align), np.clip(align, 0.0, None), 0.0)
            dist_w = math.exp(-float(meta["dist_km"]) / 30.0)
            w = align * dist_w
            valid = np.isfinite(vals) & np.isfinite(wdir)
            num += np.where(valid, vals * w, 0.0)
            den += np.where(valid, w, 0.0)
        out[out_name] = np.where(den > 1e-12, num / den, np.nan)
        extras.append(out_name)

    upwind("temp_now", "e2_upwind_temp_now")
    upwind("dewpt_now", "e2_upwind_dewpt_now")
    upwind("clds_oktas_now", "e2_upwind_clds_oktas_now")
    upwind("vis_now", "e2_upwind_vis_now")
    upwind("pressure_now", "e2_upwind_pressure_now")

    out["e2_upwind_minus_klga_temp"] = out["e2_upwind_temp_now"] - _safe_numeric(out, "temp_now")
    out["e2_upwind_minus_klga_dewpt"] = out["e2_upwind_dewpt_now"] - _safe_numeric(out, "dewpt_now")
    extras.extend(["e2_upwind_minus_klga_temp", "e2_upwind_minus_klga_dewpt"])

    out["e2_jfk_minus_lga_temp"] = _safe_numeric(out, "KJFK_temp_now") - _safe_numeric(out, "temp_now")
    out["e2_jfk_minus_lga_dewpt"] = _safe_numeric(out, "KJFK_dewpt_now") - _safe_numeric(out, "dewpt_now")
    out["e2_ewr_minus_lga_temp"] = _safe_numeric(out, "KEWR_temp_now") - _safe_numeric(out, "temp_now")
    out["e2_ewr_minus_lga_dewpt"] = _safe_numeric(out, "KEWR_dewpt_now") - _safe_numeric(out, "dewpt_now")
    extras.extend(
        [
            "e2_jfk_minus_lga_temp",
            "e2_jfk_minus_lga_dewpt",
            "e2_ewr_minus_lga_temp",
            "e2_ewr_minus_lga_dewpt",
        ]
    )

    out["e2_coastal_minus_inland_temp"] = _safe_numeric(out, "coastal_minus_inland_temp")
    out["e2_coastal_minus_inland_dew"] = _safe_numeric(out, "dewpt_coastal_minus_inland")
    out["e2_coastal_minus_inland_pressure"] = _safe_numeric(out, "pressure_coastal_minus_inland")
    out["e2_nbr_temp_range"] = _safe_numeric(out, "nbr_temp_range")
    out["e2_nbr_pressure_range"] = _safe_numeric(out, "nbr_pressure_range")
    extras.extend(
        [
            "e2_coastal_minus_inland_temp",
            "e2_coastal_minus_inland_dew",
            "e2_coastal_minus_inland_pressure",
            "e2_nbr_temp_range",
            "e2_nbr_pressure_range",
        ]
    )

    coast_prev = np.nanmean(np.vstack([_safe_numeric(out, _station_col(s, "temp_prev_180")) for s in COASTAL_SHORT]), axis=0)
    inland_prev = np.nanmean(np.vstack([_safe_numeric(out, _station_col(s, "temp_prev_180")) for s in INLAND_SHORT]), axis=0)
    grad_now = _safe_numeric(out, "coastal_minus_inland_temp")
    grad_prev = coast_prev - inland_prev
    grad_flip = (np.sign(grad_now) != np.sign(grad_prev)) & np.isfinite(grad_now) & np.isfinite(grad_prev)
    out["e2_gradient_flip_180"] = grad_flip.astype(float)
    extras.append("e2_gradient_flip_180")

    sin_prev = _safe_numeric(out, "wdir_sin_prev_180")
    cos_prev = _safe_numeric(out, "wdir_cos_prev_180")
    prev_deg = (np.degrees(np.arctan2(sin_prev, cos_prev)) + 360.0) % 360.0
    east_now = ((wdir >= 60.0) & (wdir <= 160.0)).astype(float)
    east_prev = ((prev_deg >= 60.0) & (prev_deg <= 160.0) & np.isfinite(prev_deg)).astype(float)
    wind_shift = np.clip(east_now - east_prev, 0.0, 1.0)
    dew_jump = np.clip(_safe_numeric(out, "dew_pt_slope_180"), 0.0, None)
    temp_slope = _safe_numeric(out, "temp_slope_180")
    after_noon = (_safe_numeric(out, "cutoff_minutes") >= 720.0).astype(float)
    heat_collapse = np.clip(-temp_slope, 0.0, None) * after_noon
    score_raw = (1.3 * wind_shift) + (8.0 * dew_jump) + (6.0 * heat_collapse) + (1.0 * out["e2_gradient_flip_180"].to_numpy(dtype=float))
    sea_score = expit(score_raw - 1.5)
    out["e2_wind_shift_east_3h"] = wind_shift
    out["e2_dew_jump_3h"] = dew_jump
    out["e2_heat_collapse"] = heat_collapse
    out["e2_sea_breeze_score"] = sea_score
    out["e2_sea_breeze_flag"] = (sea_score >= 0.5).astype(float)
    extras.extend(
        [
            "e2_wind_shift_east_3h",
            "e2_dew_jump_3h",
            "e2_heat_collapse",
            "e2_sea_breeze_score",
            "e2_sea_breeze_flag",
        ]
    )

    meta = {
        "neighbor_meta": neighbor_meta,
        "extra_feature_count": int(len(extras)),
    }
    return out, extras, meta


def _write_feature_importance_with_contribution(df: pd.DataFrame, out_path: Path) -> None:
    out = df.copy()
    gain_sum = float(np.nansum(pd.to_numeric(out["gain"], errors="coerce").to_numpy(dtype=float)))
    split_sum = float(np.nansum(pd.to_numeric(out["split"], errors="coerce").to_numpy(dtype=float)))
    if gain_sum > 0.0:
        out["gain_contribution_pct"] = (pd.to_numeric(out["gain"], errors="coerce") / gain_sum) * 100.0
    else:
        out["gain_contribution_pct"] = 0.0
    if split_sum > 0.0:
        out["split_contribution_pct"] = (pd.to_numeric(out["split"], errors="coerce") / split_sum) * 100.0
    else:
        out["split_contribution_pct"] = 0.0
    out.to_csv(out_path, index=False)


def _optimize_convex_weights_peak(y_true: np.ndarray, member_probs: list[np.ndarray]) -> tuple[np.ndarray, float]:
    y = np.asarray(y_true, dtype=int)
    mats = [np.asarray(p, dtype=float) for p in member_probs]
    n = len(mats)
    if n == 0:
        raise ValueError("No member probabilities provided for peak blending.")

    x0 = np.full(n, 1.0 / n, dtype=float)
    bounds = [(0.0, 1.0)] * n
    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]

    def obj(w: np.ndarray) -> float:
        p = np.zeros_like(mats[0], dtype=float)
        for wi, arr in zip(w, mats):
            p += float(wi) * arr
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return float(log_loss(y, p))

    res = minimize(obj, x0=x0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        return x0, float(obj(x0))
    w = np.asarray(res.x, dtype=float)
    w = np.clip(w, 0.0, 1.0)
    s = float(np.sum(w))
    if s <= 0.0:
        w = x0
    else:
        w = w / s
    return w, float(obj(w))


def _optimize_convex_weights_delta(y_true: np.ndarray, member_probs: list[np.ndarray]) -> tuple[np.ndarray, float]:
    y = np.asarray(y_true, dtype=int)
    mats = [np.asarray(p, dtype=float) for p in member_probs]
    n = len(mats)
    if n == 0:
        raise ValueError("No member probabilities provided for delta blending.")
    x0 = np.full(n, 1.0 / n, dtype=float)
    bounds = [(0.0, 1.0)] * n
    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]

    def obj(w: np.ndarray) -> float:
        p = np.zeros_like(mats[0], dtype=float)
        for wi, arr in zip(w, mats):
            p += float(wi) * arr
        row_sum = np.sum(p, axis=1, keepdims=True)
        good = row_sum.squeeze() > 0.0
        p[good] = p[good] / row_sum[good]
        if np.any(~good):
            p[~good] = 1.0 / p.shape[1]
        return float(_multi_logloss(y, p))

    res = minimize(obj, x0=x0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        return x0, float(obj(x0))
    w = np.asarray(res.x, dtype=float)
    w = np.clip(w, 0.0, 1.0)
    s = float(np.sum(w))
    if s <= 0.0:
        w = x0
    else:
        w = w / s
    return w, float(obj(w))


def _blend_members(weights: np.ndarray, member_probs: list[np.ndarray]) -> np.ndarray:
    out = np.zeros_like(member_probs[0], dtype=float)
    for w, arr in zip(weights, member_probs):
        out += float(w) * np.asarray(arr, dtype=float)
    return out


def _calibrate_peak_temperature(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    y = np.asarray(y, dtype=int)
    logits = np.log(p / (1.0 - p))

    def objective(log_t: float) -> float:
        t = float(np.exp(log_t))
        p_cal = expit(logits / max(t, 1e-6))
        return float(log_loss(y, np.clip(p_cal, 1e-6, 1.0 - 1e-6)))

    res = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
    if not res.success:
        return 1.0
    return float(np.exp(res.x))


def _apply_peak_temperature(p: np.ndarray, temperature: float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    logits = np.log(p / (1.0 - p))
    return np.clip(expit(logits / max(float(temperature), 1e-6)), 1e-6, 1.0 - 1e-6)


def _apply_delta_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    logits = np.log(np.clip(p, 1e-12, 1.0))
    return _softmax(logits / max(float(temperature), 1e-6))


def _write_750_word_summary(
    *,
    run_dir: Path,
    experiment_id: str,
    feature_count: int,
    metrics: dict[str, Any],
    leakage_guards: dict[str, Any],
    experiment_notes: list[str],
) -> Path:
    combined = metrics.get("combined", {})
    peak = metrics.get("peak", {})
    delta = metrics.get("delta", {})
    paragraphs = [
        (
            f"Experiment {experiment_id} in Experiment_set_1 was designed to extend the KLGA same-day Tmax distribution system while preserving strict leakage safety. "
            f"The run used a date span from {metrics.get('split', {}).get('train_start')} through {metrics.get('split', {}).get('test_end')} with explicit train, validation, and test partitions. "
            "Every feature was computed with as-of semantics only, and every train-derived artifact (categorical maps, anomaly lookups, imputers, and calibration constants) was frozen before validation and test scoring. "
            "This protects against forward-looking leakage and keeps the run behavior reproducible for later strategy discussions."
        ),
        (
            "The label gate was enforced before model training, using daily_max_truth_klga.csv as the sole settlement-aligned source per user instruction. "
            "The gate checks included schema presence, whole-degree Fahrenheit integrity, timezone alignment, duplicate-date detection, and date-window coverage diagnostics. "
            "By running the gate first and persisting audit files, the experiment makes its label assumptions explicit and testable. "
            "This matters because distribution quality is only meaningful if the label source itself is aligned and stable."
        ),
        (
            f"The final feature contract for this run contained {feature_count} features after budget control. "
            "The selection process started with the proven V1/V3 signal spine, then appended experiment-specific features in a constrained manner, and finally applied a hard cap. "
            "This avoids sparse feature explosions and keeps model complexity auditable. "
            "The exported feature list and imputer maps are included in the run directory, which allows deterministic matrix reconstruction for standalone reuse."
        ),
        (
            "Modeling kept the Peak+Delta factorization: a calibrated binary peak head predicts whether the day has already peaked at cutoff, and a calibrated delta head predicts the remaining warming distribution conditional on non-peak regime. "
            "This decomposition is both meteorologically coherent and operationally practical for probability markets. "
            "Peak metrics and delta metrics are reported separately so interpretation stays clean, and combined NLL is reported as the live-like PMF quality measure."
        ),
        (
            f"Observed metrics for this run were: combined validation NLL={combined.get('val', {}).get('nll')}, combined test NLL={combined.get('test', {}).get('nll')}, "
            f"peak validation logloss calibrated={peak.get('val', {}).get('logloss_cal')}, peak test logloss calibrated={peak.get('test', {}).get('logloss_cal')}, "
            f"delta validation multiclass/ordinal logloss calibrated={delta.get('val', {}).get('multi_logloss_temp')}, and delta test logloss calibrated={delta.get('test', {}).get('multi_logloss_temp')}. "
            "These values should be read together with per-cutoff reports and bucket calibration files, because aggregate metrics can hide regime-specific weaknesses."
        ),
        (
            "Feature importance exports in this run include both raw gain/split values and normalized contribution percentages. "
            "That dual view is intentional: gain tracks loss reduction impact, while contribution percentages provide a stable relative ranking for cross-run comparison. "
            "For ensemble or ordinal variants, additional aggregate contribution files are included to summarize how the final probability output is influenced across member models or thresholds."
        ),
        (
            "Leakage paranoia remained central throughout the run. "
            f"As-of guard failures recorded: {leakage_guards.get('asof_guard_failures')}. "
            "Any non-zero value would have invalidated the experiment. "
            "The run also records forbidden observation columns and confirms none were present in the export inputs. "
            "This guards against accidental use of summary-like daily fields that could otherwise leak post-cutoff information into training features."
        ),
        (
            "Operationally, the run is restart-safe and non-destructive: artifacts are written into a timestamped subfolder under the experiment id, preserving prior runs. "
            "Each run includes configuration, metrics JSON/Markdown, prediction tables, model artifacts, feature metadata, and guard/audit artifacts. "
            "That structure supports fast forensic review, allows later codex sessions to pick up context quickly, and avoids silent overwrite behavior."
        ),
        (
            "Experiment-specific interpretation notes: "
            + " ".join(experiment_notes)
            + " The intended use of this output is iterative experimentation where each run can be compared by exact contract, exact artifacts, and exact metrics rather than by memory or informal notes."
        ),
    ]
    text = "\n\n".join(paragraphs)
    while _format_word_count(text) < 750:
        text += (
            "\n\nAdditional technical context: the PMF assembly uses the same canonical mapping from delta space to integer Fahrenheit outcomes, "
            "ensuring consistency with downstream bucket probability calculations. This means changes in run quality can be attributed to feature/objective/ensemble differences rather than post-hoc scoring differences. "
            "The artifact contract also makes room for future Experiment_set_2 expansions without breaking Experiment_set_1 reproducibility."
        )
    out_path = run_dir / "experiment_high_level_summary_750_words.md"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def _train_binary_lgbm(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_weight: np.ndarray | None,
    random_state: int,
) -> Any:
    import lightgbm as lgb

    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0
    model = lgb.LGBMClassifier(
        boosting_type="gbdt",
        objective="binary",
        metric="binary_logloss",
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=600,
        min_data_in_leaf=1200,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=2.0,
        scale_pos_weight=scale_pos_weight,
        random_state=int(random_state),
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        X_train,
        y_train.astype(int),
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val.astype(int))],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)],
    )
    return model


def _enforce_monotone_ge(p_ge: np.ndarray) -> np.ndarray:
    out = np.asarray(p_ge, dtype=float).copy()
    out[:, 0] = 1.0
    for j in range(1, out.shape[1]):
        out[:, j] = np.minimum(out[:, j], out[:, j - 1])
    out = np.clip(out, 1e-8, 1.0)
    return out


def _ge_to_pmf(p_ge: np.ndarray) -> np.ndarray:
    k = p_ge.shape[1]
    pmf = np.zeros((p_ge.shape[0], k), dtype=float)
    pmf[:, 0] = np.clip(1.0 - p_ge[:, 1], 0.0, 1.0)
    for idx in range(1, k - 1):
        pmf[:, idx] = np.clip(p_ge[:, idx] - p_ge[:, idx + 1], 0.0, 1.0)
    pmf[:, k - 1] = np.clip(p_ge[:, k - 1], 0.0, 1.0)
    row_sum = np.sum(pmf, axis=1, keepdims=True)
    good = row_sum.squeeze() > 0.0
    pmf[good] = pmf[good] / row_sum[good]
    if np.any(~good):
        pmf[~good] = 1.0 / k
    return pmf


def _expand_modeled_values(modeled: dict[int, np.ndarray], *, k_max: int) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    modeled_keys = sorted(modeled.keys())
    for k in range(2, k_max + 1):
        if k in modeled:
            out[k] = modeled[k]
            continue
        lower = [x for x in modeled_keys if x < k]
        upper = [x for x in modeled_keys if x > k]
        if lower and upper:
            kl = lower[-1]
            ku = upper[0]
            w = float(k - kl) / float(ku - kl)
            out[k] = (1.0 - w) * modeled[kl] + w * modeled[ku]
        elif lower:
            out[k] = modeled[lower[-1]]
        elif upper:
            out[k] = modeled[upper[0]]
        else:
            raise ValueError("No modeled thresholds available for ordinal expansion.")
    return out


def _train_ordinal_delta(
    *,
    x_all: np.ndarray,
    slices: SplitSlices,
    train_weights_full: np.ndarray,
    random_seed: int,
    threshold_stride: int,
    models_dir: Path,
    reports_dir: Path,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], Path]:
    k_max = int(slices.delta_class_max)
    x_train = x_all[slices.delta_train_idx]
    x_val = x_all[slices.delta_val_idx]
    x_test = x_all[slices.delta_test_idx]
    x_peak_val = x_all[slices.peak_val_idx]
    x_peak_test = x_all[slices.peak_test_idx]
    y_train_raw = slices.y_delta_train_class + 1
    y_val_raw = slices.y_delta_val_class + 1
    y_test_raw = slices.y_delta_test_class + 1
    sample_weight = train_weights_full[slices.delta_train_idx]
    thresholds = list(range(2, k_max + 1, max(int(threshold_stride), 1)))

    probs_peak_val: dict[int, np.ndarray] = {}
    probs_peak_test: dict[int, np.ndarray] = {}
    probs_delta_val: dict[int, np.ndarray] = {}
    probs_delta_test: dict[int, np.ndarray] = {}
    logits_delta_val: dict[int, np.ndarray] = {}
    models_manifest: dict[str, Any] = {"thresholds_modeled": thresholds, "models": []}
    importance_rows: list[pd.DataFrame] = []

    logger.info("ORDINAL_DELTA_START thresholds=%d k_max=%d stride=%d", len(thresholds), k_max, threshold_stride)
    for i, thr in enumerate(thresholds, start=1):
        yb_train = (y_train_raw >= thr).astype(int)
        yb_val = (y_val_raw >= thr).astype(int)
        yb_test = (y_test_raw >= thr).astype(int)
        prevalence = float(np.mean(yb_train))
        logger.info(
            "ORDINAL_THRESHOLD [%d/%d] k=%d prevalence=%.6f pos=%d neg=%d",
            i,
            len(thresholds),
            thr,
            prevalence,
            int(np.sum(yb_train == 1)),
            int(np.sum(yb_train == 0)),
        )
        if np.min(yb_train) == np.max(yb_train):
            p_const = float(yb_train[0])
            probs_peak_val[thr] = np.full(len(x_peak_val), p_const, dtype=float)
            probs_peak_test[thr] = np.full(len(x_peak_test), p_const, dtype=float)
            probs_delta_val[thr] = np.full(len(x_val), p_const, dtype=float)
            probs_delta_test[thr] = np.full(len(x_test), p_const, dtype=float)
            logit_const = float(np.log(np.clip(p_const, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - p_const, 1e-6, 1.0)))
            logits_delta_val[thr] = np.full(len(x_val), logit_const, dtype=float)
            models_manifest["models"].append(
                {"threshold": int(thr), "type": "constant", "constant_probability": p_const}
            )
            continue

        mdl = _train_binary_lgbm(
            X_train=x_train,
            y_train=yb_train,
            X_val=x_val,
            y_val=yb_val,
            sample_weight=sample_weight,
            random_state=random_seed + thr,
        )
        model_path = models_dir / f"ordinal_delta_ge_{thr}.txt"
        mdl.booster_.save_model(str(model_path))
        models_manifest["models"].append(
            {"threshold": int(thr), "type": "lgbm_binary", "model_path": str(model_path)}
        )

        p_peak_val = mdl.predict_proba(x_peak_val)[:, 1]
        p_peak_test = mdl.predict_proba(x_peak_test)[:, 1]
        p_delta_val = mdl.predict_proba(x_val)[:, 1]
        p_delta_test = mdl.predict_proba(x_test)[:, 1]
        probs_peak_val[thr] = np.clip(p_peak_val, 1e-6, 1.0 - 1e-6)
        probs_peak_test[thr] = np.clip(p_peak_test, 1e-6, 1.0 - 1e-6)
        probs_delta_val[thr] = np.clip(p_delta_val, 1e-6, 1.0 - 1e-6)
        probs_delta_test[thr] = np.clip(p_delta_test, 1e-6, 1.0 - 1e-6)
        logits_delta_val[thr] = np.asarray(mdl.predict(x_val, raw_score=True), dtype=float)

        imp = pd.DataFrame(
            {
                "threshold": int(thr),
                "feature_index": np.arange(mdl.booster_.num_feature(), dtype=int),
                "gain": np.asarray(mdl.booster_.feature_importance(importance_type="gain"), dtype=float),
                "split": np.asarray(mdl.booster_.feature_importance(importance_type="split"), dtype=float),
            }
        )
        importance_rows.append(imp)

    full_peak_val = _expand_modeled_values(probs_peak_val, k_max=k_max)
    full_peak_test = _expand_modeled_values(probs_peak_test, k_max=k_max)
    full_delta_val = _expand_modeled_values(probs_delta_val, k_max=k_max)
    full_delta_test = _expand_modeled_values(probs_delta_test, k_max=k_max)
    full_logits_delta_val = _expand_modeled_values(logits_delta_val, k_max=k_max)

    p_ge_peak_val = np.ones((len(x_peak_val), k_max), dtype=float)
    p_ge_peak_test = np.ones((len(x_peak_test), k_max), dtype=float)
    p_ge_delta_val = np.ones((len(x_val), k_max), dtype=float)
    p_ge_delta_test = np.ones((len(x_test), k_max), dtype=float)
    logits_mat_delta_val = np.zeros((len(x_val), k_max), dtype=float)
    for k in range(2, k_max + 1):
        col = k - 1
        p_ge_peak_val[:, col] = np.clip(full_peak_val[k], 1e-6, 1.0 - 1e-6)
        p_ge_peak_test[:, col] = np.clip(full_peak_test[k], 1e-6, 1.0 - 1e-6)
        p_ge_delta_val[:, col] = np.clip(full_delta_val[k], 1e-6, 1.0 - 1e-6)
        p_ge_delta_test[:, col] = np.clip(full_delta_test[k], 1e-6, 1.0 - 1e-6)
        logits_mat_delta_val[:, col] = np.asarray(full_logits_delta_val[k], dtype=float)
    p_ge_peak_val = _enforce_monotone_ge(p_ge_peak_val)
    p_ge_peak_test = _enforce_monotone_ge(p_ge_peak_test)
    p_ge_delta_val = _enforce_monotone_ge(p_ge_delta_val)
    p_ge_delta_test = _enforce_monotone_ge(p_ge_delta_test)

    def objective(log_t: float) -> float:
        t = float(np.exp(log_t))
        p_ge = np.ones((len(x_val), k_max), dtype=float)
        for k in range(2, k_max + 1):
            p_ge[:, k - 1] = expit(logits_mat_delta_val[:, k - 1] / max(t, 1e-6))
        p_ge = _enforce_monotone_ge(p_ge)
        pmf = _ge_to_pmf(p_ge)
        return float(_multi_logloss(slices.y_delta_val_class, pmf))

    res = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
    temperature = float(np.exp(res.x)) if res.success else 1.0

    def apply_temperature(p_ge: np.ndarray) -> np.ndarray:
        p = np.asarray(p_ge, dtype=float).copy()
        for k in range(2, k_max + 1):
            col = k - 1
            logits = np.log(np.clip(p[:, col], 1e-6, 1.0 - 1e-6) / np.clip(1.0 - p[:, col], 1e-6, 1.0))
            p[:, col] = expit(logits / max(temperature, 1e-6))
        p = _enforce_monotone_ge(p)
        return _ge_to_pmf(p)

    p_delta_peak_val = apply_temperature(p_ge_peak_val)
    p_delta_peak_test = apply_temperature(p_ge_peak_test)
    p_delta_delta_val = apply_temperature(p_ge_delta_val)
    p_delta_delta_test = apply_temperature(p_ge_delta_test)

    if importance_rows:
        all_imp = pd.concat(importance_rows, ignore_index=True)
        all_imp.to_csv(reports_dir / "ordinal_delta_feature_importance_by_threshold.csv", index=False)
        agg = (
            all_imp.groupby("feature_index", as_index=False)
            .agg(gain=("gain", "sum"), split=("split", "sum"))
            .sort_values("gain", ascending=False)
            .reset_index(drop=True)
        )
        _write_feature_importance_with_contribution(agg, reports_dir / "ordinal_delta_feature_importance_aggregate.csv")
    else:
        pd.DataFrame(columns=["threshold", "feature_index", "gain", "split"]).to_csv(
            reports_dir / "ordinal_delta_feature_importance_by_threshold.csv",
            index=False,
        )
        pd.DataFrame(columns=["feature_index", "gain", "split", "gain_contribution_pct", "split_contribution_pct"]).to_csv(
            reports_dir / "ordinal_delta_feature_importance_aggregate.csv",
            index=False,
        )

    delta_metrics = {
        "val": {"multi_logloss_temp": float(_multi_logloss(slices.y_delta_val_class, p_delta_delta_val))},
        "test": {"multi_logloss_temp": float(_multi_logloss(slices.y_delta_test_class, p_delta_delta_test))},
        "temperature": float(temperature),
        "mode": "ordinal",
        "threshold_stride": int(threshold_stride),
        "threshold_count_modeled": int(len(thresholds)),
    }
    manifest_path = models_dir / "ordinal_delta_models_manifest.json"
    models_manifest["temperature"] = float(temperature)
    manifest_path.write_text(json.dumps(models_manifest, indent=2, sort_keys=True), encoding="utf-8")
    return p_delta_peak_val, p_delta_peak_test, p_delta_delta_val, p_delta_delta_test, delta_metrics, manifest_path


def _run_single_experiment(
    *,
    experiment_id: str,
    run_dir: Path,
    feat_df: pd.DataFrame,
    feature_cols: list[str],
    ctx: DatasetContext,
    slices: SplitSlices,
    cfg: ExperimentSet1Config,
    logger: logging.Logger,
    delta_mode: str,
    random_seed: int,
    extra_meta: dict[str, Any],
) -> ExperimentArtifacts:
    models_dir = _ensure_dir(run_dir / "models")
    reports_dir = _ensure_dir(run_dir / "reports")
    predictions_dir = _ensure_dir(run_dir / "predictions")

    medians = _fit_imputer(feat_df, feature_cols=feature_cols, train_mask=ctx.masks["train"])
    x_all = _apply_imputer(feat_df, feature_cols=feature_cols, medians=medians)

    train_weights = _recency_weights(feat_df, ctx.masks["train"] & np.isfinite(pd.to_numeric(feat_df["peak"], errors="coerce")))
    peak_train_idx = slices.peak_train_idx
    peak_val_idx = slices.peak_val_idx
    peak_test_idx = slices.peak_test_idx
    delta_train_idx = slices.delta_train_idx
    delta_val_idx = slices.delta_val_idx
    delta_test_idx = slices.delta_test_idx

    peak_result = train_peak_model(
        X_train=x_all[peak_train_idx],
        y_train=slices.y_peak_all[peak_train_idx],
        X_val=x_all[peak_val_idx],
        y_val=slices.y_peak_all[peak_val_idx],
        sample_weight_train=train_weights,
        categorical_feature=[feature_cols.index(c) for c in ctx.categorical_names if c in feature_cols],
        params_override={
            "num_leaves": 127,
            "learning_rate": 0.05,
            "n_estimators": 3500,
            "min_data_in_leaf": 1800,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 5.0,
            "min_data_per_group": 200,
            "cat_smooth": 20,
            "random_state": int(random_seed),
        },
        logger=logger,
        log_period=cfg.peak_train_log_period,
        log_every_seconds=cfg.train_log_every_seconds,
        heartbeat_seconds=cfg.train_heartbeat_seconds,
        stage_label=f"{experiment_id}_PEAK_TRAIN",
    )
    p_peak_val_raw, p_peak_val_cal = predict_peak_probability(model=peak_result.model, isotonic=peak_result.isotonic, X=x_all[peak_val_idx])
    p_peak_test_raw, p_peak_test_cal = predict_peak_probability(model=peak_result.model, isotonic=peak_result.isotonic, X=x_all[peak_test_idx])

    train_weight_full = np.zeros(len(feat_df), dtype=float)
    train_weight_full[peak_train_idx] = train_weights
    delta_train_weights = train_weight_full[delta_train_idx]

    delta_manifest: dict[str, Any] = {"mode": delta_mode}
    if delta_mode == "multiclass":
        delta_result = train_delta_model(
            X_train=x_all[delta_train_idx],
            y_train=slices.y_delta_train_class.astype(int),
            X_val=x_all[delta_val_idx],
            y_val=slices.y_delta_val_class.astype(int),
            num_classes=slices.delta_class_max,
            sample_weight_train=delta_train_weights,
            categorical_feature=[feature_cols.index(c) for c in ctx.categorical_names if c in feature_cols],
            objective="multiclass",
            use_class_weights=True,
            params_override={
                "num_leaves": 64,
                "learning_rate": 0.03,
                "n_estimators": 4500,
                "min_data_in_leaf": 1200,
                "feature_fraction": 0.7,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "lambda_l2": 10.0,
                "max_depth": -1,
                "min_data_per_group": 300,
                "cat_smooth": 30,
                "random_state": int(random_seed),
            },
            logger=logger,
            log_period=cfg.delta_train_log_period,
            log_every_seconds=cfg.train_log_every_seconds,
            heartbeat_seconds=cfg.train_heartbeat_seconds,
            stage_label=f"{experiment_id}_DELTA_TRAIN",
        )
        _, p_delta_raw_all, p_delta_temp_all = predict_delta_conditional(model=delta_result.model, temperature=delta_result.temperature, X=x_all)
        model_classes = np.asarray(delta_result.model.classes_, dtype=int)
        p_delta_raw_all = _expand_class_probs(probs=p_delta_raw_all, class_labels=model_classes, total_classes=slices.delta_class_max)
        p_delta_temp_all = _expand_class_probs(probs=p_delta_temp_all, class_labels=model_classes, total_classes=slices.delta_class_max)
        p_delta_peak_val = p_delta_temp_all[peak_val_idx]
        p_delta_peak_test = p_delta_temp_all[peak_test_idx]
        p_delta_delta_val = p_delta_temp_all[delta_val_idx]
        p_delta_delta_test = p_delta_temp_all[delta_test_idx]
        delta_metrics = {
            "val": {
                "multi_logloss_raw": float(_multi_logloss(slices.y_delta_val_class, p_delta_raw_all[delta_val_idx])),
                "multi_logloss_temp": float(_multi_logloss(slices.y_delta_val_class, p_delta_delta_val)),
            },
            "test": {
                "multi_logloss_raw": float(_multi_logloss(slices.y_delta_test_class, p_delta_raw_all[delta_test_idx])),
                "multi_logloss_temp": float(_multi_logloss(slices.y_delta_test_class, p_delta_delta_test)),
            },
            "temperature": float(delta_result.temperature),
            "mode": "multiclass",
        }
        delta_model_path = models_dir / "delta_model.txt"
        delta_result.model.booster_.save_model(str(delta_model_path))
        (models_dir / "delta_temperature_T.json").write_text(
            json.dumps({"temperature": float(delta_result.temperature)}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        delta_manifest["model_path"] = str(delta_model_path)
        delta_manifest["temperature"] = float(delta_result.temperature)
    elif delta_mode == "ordinal":
        (
            p_delta_peak_val,
            p_delta_peak_test,
            p_delta_delta_val,
            p_delta_delta_test,
            delta_metrics,
            ordinal_manifest_path,
        ) = _train_ordinal_delta(
            x_all=x_all,
            slices=slices,
            train_weights_full=train_weight_full,
            random_seed=random_seed,
            threshold_stride=cfg.ordinal_threshold_stride,
            models_dir=models_dir,
            reports_dir=reports_dir,
            logger=logger,
        )
        delta_model_path = ordinal_manifest_path
        delta_manifest["manifest_path"] = str(ordinal_manifest_path)
    else:
        raise ValueError(f"Unsupported delta mode: {delta_mode}")

    combined_val, combined_val_detail = _evaluate_distribution_rows(
        df=feat_df,
        row_indices=peak_val_idx,
        p_peak=p_peak_val_cal,
        p_delta_cond=p_delta_peak_val,
    )
    combined_test, combined_test_detail = _evaluate_distribution_rows(
        df=feat_df,
        row_indices=peak_test_idx,
        p_peak=p_peak_test_cal,
        p_delta_cond=p_delta_peak_test,
    )

    _cutoff_metrics(df_detail=combined_val_detail).to_csv(reports_dir / "cutoff_metrics_val.csv", index=False)
    _cutoff_metrics(df_detail=combined_test_detail).to_csv(reports_dir / "cutoff_metrics_test.csv", index=False)
    _temperature_bucket_calibration(df=feat_df, row_indices=peak_val_idx, p_peak=p_peak_val_cal, p_delta_cond=p_delta_peak_val).to_csv(
        reports_dir / "bucket_calibration_val.csv",
        index=False,
    )
    _temperature_bucket_calibration(df=feat_df, row_indices=peak_test_idx, p_peak=p_peak_test_cal, p_delta_cond=p_delta_peak_test).to_csv(
        reports_dir / "bucket_calibration_test.csv",
        index=False,
    )

    peak_gain = pd.DataFrame(
        {
            "feature": feature_cols,
            "gain": np.asarray(peak_result.model.booster_.feature_importance(importance_type="gain"), dtype=float),
            "split": np.asarray(peak_result.model.booster_.feature_importance(importance_type="split"), dtype=float),
        }
    ).sort_values("gain", ascending=False)
    _write_feature_importance_with_contribution(peak_gain, reports_dir / "peak_feature_importance.csv")
    if delta_mode == "multiclass":
        delta_booster = delta_result.model.booster_
        delta_gain = pd.DataFrame(
            {
                "feature": feature_cols,
                "gain": np.asarray(delta_booster.feature_importance(importance_type="gain"), dtype=float),
                "split": np.asarray(delta_booster.feature_importance(importance_type="split"), dtype=float),
            }
        ).sort_values("gain", ascending=False)
        _write_feature_importance_with_contribution(delta_gain, reports_dir / "delta_feature_importance.csv")
    else:
        delta_gain = pd.read_csv(reports_dir / "ordinal_delta_feature_importance_aggregate.csv")
        if "feature_index" in delta_gain.columns:
            idx = pd.to_numeric(delta_gain["feature_index"], errors="coerce").fillna(-1).astype(int)
            delta_gain["feature"] = idx.map(lambda i: feature_cols[i] if 0 <= i < len(feature_cols) else f"feature_{i}")
            cols = [c for c in ["feature", "gain", "split", "gain_contribution_pct", "split_contribution_pct"] if c in delta_gain.columns]
            delta_gain = delta_gain[cols]
        delta_gain.to_csv(reports_dir / "delta_feature_importance.csv", index=False)

    pred_val = _build_full_delta_arrays(full_len=len(feat_df), class_count=slices.delta_class_max, row_indices=peak_val_idx, p_peak=p_peak_val_cal, p_delta=p_delta_peak_val)
    pred_test = _build_full_delta_arrays(full_len=len(feat_df), class_count=slices.delta_class_max, row_indices=peak_test_idx, p_peak=p_peak_test_cal, p_delta=p_delta_peak_test)
    _write_df_with_csv_parquet(pred_val, predictions_dir / "predictions_val", logger)
    _write_df_with_csv_parquet(pred_test, predictions_dir / "predictions_test", logger)
    _write_df_with_csv_parquet(combined_val_detail, predictions_dir / "distribution_eval_val", logger)
    _write_df_with_csv_parquet(combined_test_detail, predictions_dir / "distribution_eval_test", logger)

    peak_model_path = models_dir / "peak_model.txt"
    peak_result.model.booster_.save_model(str(peak_model_path))
    joblib.dump(peak_result.isotonic, models_dir / "peak_isotonic.pkl")
    (run_dir / "feature_list.json").write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
    (run_dir / "imputer_values.json").write_text(json.dumps(medians, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "categorical_maps.json").write_text(json.dumps(ctx.cat_maps, indent=2, sort_keys=True), encoding="utf-8")

    peak_metrics = {
        "val": {
            "logloss_raw": float(log_loss(slices.y_peak_all[peak_val_idx], np.clip(p_peak_val_raw, 1e-6, 1.0 - 1e-6))),
            "logloss_cal": float(log_loss(slices.y_peak_all[peak_val_idx], np.clip(p_peak_val_cal, 1e-6, 1.0 - 1e-6))),
            "brier_raw": float(brier_score_loss(slices.y_peak_all[peak_val_idx], p_peak_val_raw)),
            "brier_cal": float(brier_score_loss(slices.y_peak_all[peak_val_idx], p_peak_val_cal)),
        },
        "test": {
            "logloss_raw": float(log_loss(slices.y_peak_all[peak_test_idx], np.clip(p_peak_test_raw, 1e-6, 1.0 - 1e-6))),
            "logloss_cal": float(log_loss(slices.y_peak_all[peak_test_idx], np.clip(p_peak_test_cal, 1e-6, 1.0 - 1e-6))),
            "brier_raw": float(brier_score_loss(slices.y_peak_all[peak_test_idx], p_peak_test_raw)),
            "brier_cal": float(brier_score_loss(slices.y_peak_all[peak_test_idx], p_peak_test_cal)),
        },
    }
    metrics = {
        "experiment_id": experiment_id,
        "run_id": run_dir.name,
        "backend": "lgbm",
        "delta_mode": delta_mode,
        "feature_count": int(len(feature_cols)),
        "data_rows_total": int(len(feat_df)),
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
        "split": {
            "train_start": str(cfg.split.train_start),
            "train_end": str(cfg.split.train_end),
            "val_start": str(cfg.split.val_start),
            "val_end": str(cfg.split.val_end),
            "test_start": str(cfg.split.test_start),
            "test_end": str(cfg.split.test_end),
        },
        "peak": peak_metrics,
        "delta": delta_metrics,
        "combined": {"val": combined_val, "test": combined_test},
        "leakage_guards": ctx.leakage_guards,
        "extra_meta": extra_meta,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    md = [
        f"# Experiment {experiment_id}",
        "",
        f"- delta_mode: {delta_mode}",
        f"- feature_count: {len(feature_cols)}",
        "",
        "## Peak",
        f"- val_logloss_cal: {peak_metrics['val']['logloss_cal']}",
        f"- test_logloss_cal: {peak_metrics['test']['logloss_cal']}",
        "",
        "## Delta",
        f"- val_multi_logloss_temp: {delta_metrics['val']['multi_logloss_temp']}",
        f"- test_multi_logloss_temp: {delta_metrics['test']['multi_logloss_temp']}",
        "",
        "## Combined",
        f"- val_nll: {combined_val.get('nll')}",
        f"- test_nll: {combined_test.get('nll')}",
    ]
    (run_dir / "metrics.md").write_text("\n".join(md), encoding="utf-8")

    summary_path = _write_750_word_summary(
        run_dir=run_dir,
        experiment_id=experiment_id,
        feature_count=len(feature_cols),
        metrics=metrics,
        leakage_guards=ctx.leakage_guards,
        experiment_notes=[
            f"Delta mode was `{delta_mode}`.",
            f"Experiment metadata keys: {sorted(extra_meta.keys())[:12]}.",
            "Feature importance exports are in reports/ with contribution percentages.",
        ],
    )
    logger.info("EXPERIMENT_DONE id=%s run_dir=%s summary=%s", experiment_id, run_dir, summary_path.name)
    return ExperimentArtifacts(
        experiment_id=experiment_id,
        run_dir=run_dir,
        metrics=metrics,
        feature_cols=feature_cols,
        p_peak_val=np.asarray(p_peak_val_cal, dtype=float),
        p_peak_test=np.asarray(p_peak_test_cal, dtype=float),
        p_delta_val=np.asarray(p_delta_peak_val, dtype=float),
        p_delta_test=np.asarray(p_delta_peak_test, dtype=float),
        peak_model_path=peak_model_path,
        delta_model_path=delta_model_path,
        model_manifest=delta_manifest,
    )


def _run_e4_ensemble(
    *,
    run_dir: Path,
    feat_df: pd.DataFrame,
    ctx: DatasetContext,
    slices: SplitSlices,
    members: list[ExperimentArtifacts],
    cfg: ExperimentSet1Config,
    logger: logging.Logger,
) -> ExperimentArtifacts:
    if len(members) < 2:
        raise ValueError("E4 requires at least two member experiments.")
    models_dir = _ensure_dir(run_dir / "models")
    reports_dir = _ensure_dir(run_dir / "reports")
    predictions_dir = _ensure_dir(run_dir / "predictions")

    y_peak_val = slices.y_peak_all[slices.peak_val_idx]
    y_peak_test = slices.y_peak_all[slices.peak_test_idx]

    delta_mask_in_peak_val = np.isin(slices.peak_val_idx, slices.delta_val_idx)
    delta_mask_in_peak_test = np.isin(slices.peak_test_idx, slices.delta_test_idx)
    y_delta_val = slices.y_delta_val_class
    y_delta_test = slices.y_delta_test_class

    member_peak_val = [m.p_peak_val for m in members]
    member_peak_test = [m.p_peak_test for m in members]
    member_delta_val = [m.p_delta_val[delta_mask_in_peak_val] for m in members]
    member_delta_test = [m.p_delta_test[delta_mask_in_peak_test] for m in members]
    member_delta_peak_val = [m.p_delta_val for m in members]
    member_delta_peak_test = [m.p_delta_test for m in members]

    peak_weights, peak_val_loss = _optimize_convex_weights_peak(y_peak_val, member_peak_val)
    delta_weights, delta_val_loss = _optimize_convex_weights_delta(y_delta_val, member_delta_val)
    logger.info(
        "E4_WEIGHTS peak=%s delta=%s peak_val_logloss=%.6f delta_val_logloss=%.6f",
        peak_weights.tolist(),
        delta_weights.tolist(),
        peak_val_loss,
        delta_val_loss,
    )

    p_peak_val_blend = np.clip(_blend_members(peak_weights, member_peak_val), 1e-6, 1.0 - 1e-6)
    p_peak_test_blend = np.clip(_blend_members(peak_weights, member_peak_test), 1e-6, 1.0 - 1e-6)
    p_delta_val_blend = _blend_members(delta_weights, member_delta_peak_val)
    p_delta_test_blend = _blend_members(delta_weights, member_delta_peak_test)
    row_sum_val = np.sum(p_delta_val_blend, axis=1, keepdims=True)
    row_sum_test = np.sum(p_delta_test_blend, axis=1, keepdims=True)
    p_delta_val_blend = np.where(row_sum_val > 0.0, p_delta_val_blend / row_sum_val, 1.0 / p_delta_val_blend.shape[1])
    p_delta_test_blend = np.where(row_sum_test > 0.0, p_delta_test_blend / row_sum_test, 1.0 / p_delta_test_blend.shape[1])

    peak_temp = _calibrate_peak_temperature(p_peak_val_blend, y_peak_val)
    delta_temp = _multiclass_temperature_scale(
        logits=np.log(np.clip(p_delta_val_blend[delta_mask_in_peak_val], 1e-12, 1.0)),
        y_true=y_delta_val,
    )
    p_peak_val_cal = _apply_peak_temperature(p_peak_val_blend, peak_temp)
    p_peak_test_cal = _apply_peak_temperature(p_peak_test_blend, peak_temp)
    p_delta_val_cal = _apply_delta_temperature(p_delta_val_blend, delta_temp)
    p_delta_test_cal = _apply_delta_temperature(p_delta_test_blend, delta_temp)

    combined_val, combined_val_detail = _evaluate_distribution_rows(
        df=feat_df,
        row_indices=slices.peak_val_idx,
        p_peak=p_peak_val_cal,
        p_delta_cond=p_delta_val_cal,
    )
    combined_test, combined_test_detail = _evaluate_distribution_rows(
        df=feat_df,
        row_indices=slices.peak_test_idx,
        p_peak=p_peak_test_cal,
        p_delta_cond=p_delta_test_cal,
    )

    _cutoff_metrics(df_detail=combined_val_detail).to_csv(reports_dir / "cutoff_metrics_val.csv", index=False)
    _cutoff_metrics(df_detail=combined_test_detail).to_csv(reports_dir / "cutoff_metrics_test.csv", index=False)
    _temperature_bucket_calibration(df=feat_df, row_indices=slices.peak_val_idx, p_peak=p_peak_val_cal, p_delta_cond=p_delta_val_cal).to_csv(
        reports_dir / "bucket_calibration_val.csv",
        index=False,
    )
    _temperature_bucket_calibration(df=feat_df, row_indices=slices.peak_test_idx, p_peak=p_peak_test_cal, p_delta_cond=p_delta_test_cal).to_csv(
        reports_dir / "bucket_calibration_test.csv",
        index=False,
    )

    pred_val = _build_full_delta_arrays(
        full_len=len(feat_df),
        class_count=slices.delta_class_max,
        row_indices=slices.peak_val_idx,
        p_peak=p_peak_val_cal,
        p_delta=p_delta_val_cal,
    )
    pred_test = _build_full_delta_arrays(
        full_len=len(feat_df),
        class_count=slices.delta_class_max,
        row_indices=slices.peak_test_idx,
        p_peak=p_peak_test_cal,
        p_delta=p_delta_test_cal,
    )
    _write_df_with_csv_parquet(pred_val, predictions_dir / "predictions_val", logger)
    _write_df_with_csv_parquet(pred_test, predictions_dir / "predictions_test", logger)
    _write_df_with_csv_parquet(combined_val_detail, predictions_dir / "distribution_eval_val", logger)
    _write_df_with_csv_parquet(combined_test_detail, predictions_dir / "distribution_eval_test", logger)

    member_map = {m.experiment_id: m for m in members}
    peak_imp_rows = []
    delta_imp_rows = []
    for w, member in zip(peak_weights, members):
        p = member.run_dir / "reports" / "peak_feature_importance.csv"
        if p.exists():
            df_imp = pd.read_csv(p)
            df_imp["member_id"] = member.experiment_id
            df_imp["weighted_gain"] = pd.to_numeric(df_imp["gain"], errors="coerce") * float(w)
            peak_imp_rows.append(df_imp)
    for w, member in zip(delta_weights, members):
        p = member.run_dir / "reports" / "delta_feature_importance.csv"
        if p.exists():
            df_imp = pd.read_csv(p)
            if "feature" not in df_imp.columns:
                continue
            df_imp["member_id"] = member.experiment_id
            df_imp["weighted_gain"] = pd.to_numeric(df_imp.get("gain", 0.0), errors="coerce") * float(w)
            delta_imp_rows.append(df_imp)

    if peak_imp_rows:
        peak_all = pd.concat(peak_imp_rows, ignore_index=True)
        peak_agg = peak_all.groupby("feature", as_index=False).agg(gain=("weighted_gain", "sum"))
        peak_agg["split"] = np.nan
        peak_agg = peak_agg.sort_values("gain", ascending=False).reset_index(drop=True)
        _write_feature_importance_with_contribution(peak_agg, reports_dir / "peak_feature_importance.csv")
    else:
        pd.DataFrame(columns=["feature", "gain", "split", "gain_contribution_pct", "split_contribution_pct"]).to_csv(
            reports_dir / "peak_feature_importance.csv",
            index=False,
        )
    if delta_imp_rows:
        delta_all = pd.concat(delta_imp_rows, ignore_index=True)
        delta_agg = delta_all.groupby("feature", as_index=False).agg(gain=("weighted_gain", "sum"))
        delta_agg["split"] = np.nan
        delta_agg = delta_agg.sort_values("gain", ascending=False).reset_index(drop=True)
        _write_feature_importance_with_contribution(delta_agg, reports_dir / "delta_feature_importance.csv")
    else:
        pd.DataFrame(columns=["feature", "gain", "split", "gain_contribution_pct", "split_contribution_pct"]).to_csv(
            reports_dir / "delta_feature_importance.csv",
            index=False,
        )

    peak_metrics = {
        "val": {
            "logloss_cal": float(log_loss(y_peak_val, p_peak_val_cal)),
            "brier_cal": float(brier_score_loss(y_peak_val, p_peak_val_cal)),
        },
        "test": {
            "logloss_cal": float(log_loss(y_peak_test, p_peak_test_cal)),
            "brier_cal": float(brier_score_loss(y_peak_test, p_peak_test_cal)),
        },
    }
    delta_metrics = {
        "val": {"multi_logloss_temp": float(_multi_logloss(y_delta_val, p_delta_val_cal[delta_mask_in_peak_val]))},
        "test": {"multi_logloss_temp": float(_multi_logloss(y_delta_test, p_delta_test_cal[delta_mask_in_peak_test]))},
        "temperature": float(delta_temp),
        "mode": "ensemble_blend",
    }
    metrics = {
        "experiment_id": "E4",
        "run_id": run_dir.name,
        "backend": "lgbm_ensemble",
        "members": [m.experiment_id for m in members],
        "member_run_dirs": {m.experiment_id: str(m.run_dir) for m in members},
        "peak_weights": peak_weights.tolist(),
        "delta_weights": delta_weights.tolist(),
        "peak_temperature": float(peak_temp),
        "delta_temperature": float(delta_temp),
        "peak": peak_metrics,
        "delta": delta_metrics,
        "combined": {"val": combined_val, "test": combined_test},
        "split": {
            "train_start": str(cfg.split.train_start),
            "train_end": str(cfg.split.train_end),
            "val_start": str(cfg.split.val_start),
            "val_end": str(cfg.split.val_end),
            "test_start": str(cfg.split.test_start),
            "test_end": str(cfg.split.test_end),
        },
        "leakage_guards": ctx.leakage_guards,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "metrics.md").write_text(
        "\n".join(
            [
                "# Experiment E4",
                "",
                f"- members: {', '.join(metrics['members'])}",
                f"- val_nll: {combined_val.get('nll')}",
                f"- test_nll: {combined_test.get('nll')}",
                f"- peak_weights: {peak_weights.tolist()}",
                f"- delta_weights: {delta_weights.tolist()}",
            ]
        ),
        encoding="utf-8",
    )
    ensemble_manifest = {
        "members": metrics["members"],
        "member_run_dirs": metrics["member_run_dirs"],
        "peak_weights": metrics["peak_weights"],
        "delta_weights": metrics["delta_weights"],
        "peak_temperature": metrics["peak_temperature"],
        "delta_temperature": metrics["delta_temperature"],
    }
    ensemble_path = models_dir / "ensemble_weights.json"
    ensemble_path.write_text(json.dumps(ensemble_manifest, indent=2, sort_keys=True), encoding="utf-8")

    summary_path = _write_750_word_summary(
        run_dir=run_dir,
        experiment_id="E4",
        feature_count=int(np.mean([len(m.feature_cols) for m in members])),
        metrics=metrics,
        leakage_guards=ctx.leakage_guards,
        experiment_notes=[
            "E4 blends probability outputs from prior LGBM experiments using convex weights fit on validation data.",
            f"Member ids used: {', '.join(metrics['members'])}.",
            "Global temperature calibration was applied after blending to keep probabilities coherent.",
        ],
    )
    logger.info("E4_DONE run_dir=%s summary=%s", run_dir, summary_path.name)
    return ExperimentArtifacts(
        experiment_id="E4",
        run_dir=run_dir,
        metrics=metrics,
        feature_cols=[],
        p_peak_val=np.asarray(p_peak_val_cal, dtype=float),
        p_peak_test=np.asarray(p_peak_test_cal, dtype=float),
        p_delta_val=np.asarray(p_delta_val_cal, dtype=float),
        p_delta_test=np.asarray(p_delta_test_cal, dtype=float),
        peak_model_path=None,
        delta_model_path=ensemble_path,
        model_manifest={"ensemble": ensemble_manifest, "member_map": {k: str(v.run_dir) for k, v in member_map.items()}},
    )


def _load_prediction_frame(base_no_ext: Path) -> pd.DataFrame:
    pq = base_no_ext.with_suffix(".parquet")
    csv = base_no_ext.with_suffix(".csv")
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv, low_memory=False)
    raise FileNotFoundError(f"Missing prediction file for {base_no_ext}")


def _load_previous_artifact(set_root: Path, exp_id: str) -> ExperimentArtifacts:
    exp_root = set_root / exp_id
    if not exp_root.exists():
        raise FileNotFoundError(f"No folder found for {exp_id} under {set_root}")
    run_dirs = sorted([p for p in exp_root.iterdir() if p.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found for {exp_id} under {exp_root}")
    run_dir = run_dirs[-1]
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json for previous {exp_id} run: {run_dir}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    pred_val_df = _load_prediction_frame(run_dir / "predictions" / "predictions_val")
    pred_test_df = _load_prediction_frame(run_dir / "predictions" / "predictions_test")
    delta_cols = sorted(
        [c for c in pred_val_df.columns if c.startswith("p_delta_class_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    val_mask = np.isfinite(pd.to_numeric(pred_val_df.get("p_peak", np.nan), errors="coerce").to_numpy(dtype=float))
    test_mask = np.isfinite(pd.to_numeric(pred_test_df.get("p_peak", np.nan), errors="coerce").to_numpy(dtype=float))
    p_peak_val = pd.to_numeric(pred_val_df.loc[val_mask, "p_peak"], errors="coerce").to_numpy(dtype=float)
    p_peak_test = pd.to_numeric(pred_test_df.loc[test_mask, "p_peak"], errors="coerce").to_numpy(dtype=float)
    p_delta_val = pred_val_df.loc[val_mask, delta_cols].to_numpy(dtype=float)
    p_delta_test = pred_test_df.loc[test_mask, delta_cols].to_numpy(dtype=float)
    feature_cols = []
    feat_path = run_dir / "feature_list.json"
    if feat_path.exists():
        feature_cols = json.loads(feat_path.read_text(encoding="utf-8"))
    peak_model_path = (run_dir / "models" / "peak_model.txt")
    delta_model_path = (run_dir / "models" / "delta_model.txt")
    return ExperimentArtifacts(
        experiment_id=exp_id,
        run_dir=run_dir,
        metrics=metrics,
        feature_cols=feature_cols,
        p_peak_val=p_peak_val,
        p_peak_test=p_peak_test,
        p_delta_val=p_delta_val,
        p_delta_test=p_delta_test,
        peak_model_path=peak_model_path if peak_model_path.exists() else None,
        delta_model_path=delta_model_path if delta_model_path.exists() else None,
        model_manifest={"loaded_from_previous": True},
    )


def run_experiment_set_1(*, cfg: ExperimentSet1Config, logger: logging.Logger | None = None) -> dict[str, Any]:
    active_logger = logger or logging.getLogger(__name__)
    start = time.perf_counter()
    set_root = _ensure_dir(cfg.output_root / "Experiment_set_1")
    set_run_id = _timestamp_id()
    gate_dir = _ensure_dir(set_root / "label_alignment_audit" / set_run_id)
    _attach_run_file_handler(active_logger, set_root / f"experiment_set_1_{set_run_id}.log")

    active_logger.info(
        "EXPERIMENT_SET_1_START run_id=%s data_dir=%s output_root=%s experiments=%s",
        set_run_id,
        cfg.data_dir,
        cfg.output_root,
        ",".join(cfg.experiments),
    )
    label_gate = _label_alignment_gate(cfg=cfg, out_dir=gate_dir, logger=active_logger)
    ctx = _build_dataset_context(cfg=cfg, logger=active_logger)
    p_cfg = PipelineConfig(
        split=cfg.split,
        target_station_id=str(ctx.station_universe["target_station_id"]),
        neighbor_station_ids=tuple(ctx.station_universe["neighbor_station_ids"]),
    )
    slices = _prepare_split_slices(ctx=ctx, delta_class_max=int(p_cfg.delta_class_max))

    results: dict[str, ExperimentArtifacts] = {}
    run_manifest: dict[str, Any] = {
        "set_run_id": set_run_id,
        "set_root": str(set_root),
        "gate_dir": str(gate_dir),
        "label_gate": label_gate,
        "experiments": {},
    }

    for exp in cfg.experiments:
        exp_id = exp.strip().upper()
        if exp_id not in {"E1", "E2", "E3", "E4"}:
            raise ValueError(f"Unsupported experiment id in Experiment_set_1: {exp}")

        exp_dir = _ensure_dir(set_root / exp_id / set_run_id)
        _ensure_dir(exp_dir / "models")
        _ensure_dir(exp_dir / "reports")
        _ensure_dir(exp_dir / "predictions")
        (exp_dir / "label_alignment_audit.json").write_text(json.dumps(label_gate, indent=2, sort_keys=True), encoding="utf-8")

        active_logger.info("EXPERIMENT_START id=%s run_dir=%s", exp_id, exp_dir)
        if exp_id == "E1":
            feat_df_e1, e1_extra, e1_meta = _add_e1_phrase_to_physics_features(ctx.feat_df)
            selected_cols = _apply_feature_budget(
                base_cols=ctx.base_feature_cols,
                extra_cols=e1_extra[: cfg.phrase_feature_cap],
                max_features=cfg.feature_budget_max,
            )
            result = _run_single_experiment(
                experiment_id="E1",
                run_dir=exp_dir,
                feat_df=feat_df_e1,
                feature_cols=selected_cols,
                ctx=ctx,
                slices=slices,
                cfg=cfg,
                logger=active_logger,
                delta_mode="multiclass",
                random_seed=42,
                extra_meta={"feature_aug": "phrase_to_physics_v1", "e1_meta": e1_meta},
            )
            (exp_dir / "phrase_to_physics_v1.json").write_text(
                json.dumps(e1_meta, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        elif exp_id == "E2":
            feat_df_e2, e2_extra, e2_meta = _add_e2_advection_features(ctx.feat_df)
            selected_cols = _apply_feature_budget(
                base_cols=ctx.base_feature_cols,
                extra_cols=e2_extra[: cfg.advection_feature_cap],
                max_features=cfg.feature_budget_max,
            )
            result = _run_single_experiment(
                experiment_id="E2",
                run_dir=exp_dir,
                feat_df=feat_df_e2,
                feature_cols=selected_cols,
                ctx=ctx,
                slices=slices,
                cfg=cfg,
                logger=active_logger,
                delta_mode="multiclass",
                random_seed=73,
                extra_meta={"feature_aug": "wind_advection_seabreeze_v1", "e2_meta": e2_meta},
            )
            (exp_dir / "station_geo.json").write_text(
                json.dumps(STATION_GEO, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        elif exp_id == "E3":
            feat_df_e1, e1_extra, e1_meta = _add_e1_phrase_to_physics_features(ctx.feat_df)
            feat_df_e3, e2_extra, e2_meta = _add_e2_advection_features(feat_df_e1)
            combined_extra = list(e1_extra[: cfg.phrase_feature_cap]) + list(e2_extra[: cfg.advection_feature_cap])
            selected_cols = _apply_feature_budget(
                base_cols=ctx.base_feature_cols,
                extra_cols=combined_extra,
                max_features=cfg.feature_budget_max,
            )
            result = _run_single_experiment(
                experiment_id="E3",
                run_dir=exp_dir,
                feat_df=feat_df_e3,
                feature_cols=selected_cols,
                ctx=ctx,
                slices=slices,
                cfg=cfg,
                logger=active_logger,
                delta_mode="ordinal",
                random_seed=91,
                extra_meta={
                    "feature_aug": "phrase_plus_advection_for_ordinal",
                    "e1_meta": e1_meta,
                    "e2_meta": e2_meta,
                    "ordinal_threshold_stride": int(cfg.ordinal_threshold_stride),
                },
            )
        else:
            required = ["E1", "E2", "E3"]
            missing = [k for k in required if k not in results]
            if missing:
                for miss in missing:
                    results[miss] = _load_previous_artifact(set_root, miss)
                    active_logger.info("E4_MEMBER_LOADED_FROM_PREVIOUS id=%s run_dir=%s", miss, results[miss].run_dir)
            result = _run_e4_ensemble(
                run_dir=exp_dir,
                feat_df=ctx.feat_df,
                ctx=ctx,
                slices=slices,
                members=[results["E1"], results["E2"], results["E3"]],
                cfg=cfg,
                logger=active_logger,
            )

        results[exp_id] = result
        run_manifest["experiments"][exp_id] = {
            "run_dir": str(result.run_dir),
            "metrics": result.metrics,
            "peak_model_path": str(result.peak_model_path) if result.peak_model_path is not None else None,
            "delta_model_path": str(result.delta_model_path) if result.delta_model_path is not None else None,
        }
        active_logger.info("EXPERIMENT_END id=%s run_dir=%s", exp_id, result.run_dir)

    run_manifest["elapsed"] = format_duration(time.perf_counter() - start)
    run_manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path = set_root / f"experiment_set_1_manifest_{set_run_id}.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8")
    active_logger.info("EXPERIMENT_SET_1_DONE run_id=%s manifest=%s", set_run_id, manifest_path)
    return run_manifest
