from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hkg_t24_0085_long_history_feature_station_residual_bridge import (
    FEATURE_MATRIX_PATH,
    load_feature_matrix,
    numeric_feature_columns,
)
from scripts.run_hkg_t24_0098_source_submonth_stable_cell_specialist import load_inputs as load_0098_inputs
from scripts.run_hkg_t24_beastmode_signal_discovery import (
    CONFIRMATION_START,
    RESEARCH_ROOT,
    markdown_table,
    require_no_confirmation_dates,
    write_csv,
    write_json,
    write_text,
)
from scripts.run_hkg_t24_long_history_cross_family_interaction_atlas import (
    classify_feature_family,
    station_ids_in_feature,
    update_markdown_section,
)

FOLDER_NAME = "0100_stable_mam_cell_feature_atlas"
INPUT_0099_SUMMARY_PATH = RESEARCH_ROOT / "0099_mam_cell_policy_sensitivity" / "artifacts" / "summary.json"
INPUT_0099_TOP_PATH = RESEARCH_ROOT / "0099_mam_cell_policy_sensitivity" / "artifacts" / "top_predictions.csv"
INPUT_0099_DIAGNOSTICS_PATH = (
    RESEARCH_ROOT / "0099_mam_cell_policy_sensitivity" / "artifacts" / "best_gate_diagnostics.csv"
)
MIN_LONG_HISTORY_ROWS = 365 * 39
MIN_AGREEMENT_CORR_ROWS = 25
MIN_ACTIVE_CORR_ROWS = 20
MIN_BUCKET_ROWS = 8
TOP_FEATURES_PER_FAMILY = 25
BOOLEAN_TRUE = {"1", "true", "t", "yes", "y"}
EXTRA_NON_FEATURE_EXACT = {
    "past_doy_count",
    "past_doy_mean_tmax_c",
    "target_anomaly_vs_past_doy_c",
}
EXTRA_NON_FEATURE_TOKENS = (
    "official_error",
    "official_abs_error",
    "candidate_error",
    "prediction_error",
    "settlement",
)
MARINE_TOKENS = (
    "sea_temperature",
    "sea_temp",
    "waglan_island",
    "north_point",
)


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin(BOOLEAN_TRUE)


def finite_pair(x: pd.Series, y: pd.Series) -> pd.DataFrame:
    pair = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1)
    return pair.replace([np.inf, -np.inf], np.nan).dropna()


def safe_corr(x: pd.Series, y: pd.Series, *, min_rows: int) -> tuple[int, float]:
    pair = finite_pair(x, y)
    if len(pair) < min_rows:
        return int(len(pair)), math.nan
    if pair.iloc[:, 0].nunique(dropna=True) < 2 or pair.iloc[:, 1].nunique(dropna=True) < 2:
        return int(len(pair)), math.nan
    return int(len(pair)), float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))


def standardized_mean_diff(x_group: pd.Series, x_reference: pd.Series, *, min_rows: int = 20) -> tuple[int, int, float]:
    group = pd.to_numeric(x_group, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    reference = pd.to_numeric(x_reference, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(group) < min_rows or len(reference) < min_rows:
        return int(len(group)), int(len(reference)), math.nan
    group_var = float(group.var(ddof=1))
    reference_var = float(reference.var(ddof=1))
    pooled = math.sqrt((group_var + reference_var) / 2.0)
    if not math.isfinite(pooled) or pooled <= 0.0:
        return int(len(group)), int(len(reference)), math.nan
    return int(len(group)), int(len(reference)), float((group.mean() - reference.mean()) / pooled)


def quantile_response_spread(
    feature: pd.Series,
    response: pd.Series,
    *,
    q: int = 3,
    min_bucket_rows: int = MIN_BUCKET_ROWS,
) -> dict[str, object]:
    pair = finite_pair(feature, response)
    if len(pair) < q * min_bucket_rows or pair.iloc[:, 0].nunique(dropna=True) < q:
        return {
            "spread_rows": int(len(pair)),
            "spread_bucket_count": 0,
            "spread_min_mean": math.nan,
            "spread_max_mean": math.nan,
            "spread_c": math.nan,
        }
    try:
        buckets = pd.qcut(pair.iloc[:, 0], q=q, duplicates="drop")
    except ValueError:
        return {
            "spread_rows": int(len(pair)),
            "spread_bucket_count": 0,
            "spread_min_mean": math.nan,
            "spread_max_mean": math.nan,
            "spread_c": math.nan,
        }
    stats = pair.iloc[:, 1].groupby(buckets, observed=True).agg(["count", "mean"]).reset_index(drop=True)
    stats = stats[stats["count"].ge(min_bucket_rows)].copy()
    if len(stats) < 2:
        return {
            "spread_rows": int(len(pair)),
            "spread_bucket_count": int(len(stats)),
            "spread_min_mean": math.nan,
            "spread_max_mean": math.nan,
            "spread_c": math.nan,
        }
    low = float(stats["mean"].min())
    high = float(stats["mean"].max())
    return {
        "spread_rows": int(len(pair)),
        "spread_bucket_count": int(len(stats)),
        "spread_min_mean": low,
        "spread_max_mean": high,
        "spread_c": float(high - low),
    }


def atlas_family(feature_name: str) -> str:
    base_family = classify_feature_family(feature_name)
    lowered = feature_name.lower()
    if lowered.startswith("volatility_"):
        return "target_memory"
    if base_family == "hko_daily_climate" and any(token in lowered for token in MARINE_TOKENS):
        return "marine_proxy"
    return base_family


def feature_timestamp_status(feature_name: str, family: str) -> dict[str, object]:
    lowered = feature_name.lower()
    if family == "calendar_climatology":
        return {
            "timestamp_audit_status": "eligible_calendar_known_pre_cutoff",
            "allowed_for_future_walkforward": True,
            "cutoff_rule": "calendar fields are known before any forecast cutoff",
            "required_proof_before_model": "",
            "blocker": "",
        }
    if family == "target_memory" and lowered.startswith(
        (
            "target_lag",
            "target_roll",
            "target_spell",
            "target_reversal",
            "target_entropy",
            "target_abs_change",
            "trajectory_",
            "volatility_",
        )
    ):
        return {
            "timestamp_audit_status": "eligible_lagged_target_memory",
            "allowed_for_future_walkforward": True,
            "cutoff_rule": "uses lagged target history only; current target_tmax_c is excluded",
            "required_proof_before_model": "recompute from historical target rows available before the cutoff",
            "blocker": "",
        }
    if family == "isd_station_network":
        return {
            "timestamp_audit_status": "eligible_proven_pre_cutoff_station",
            "allowed_for_future_walkforward": True,
            "cutoff_rule": "uses station observations summarized before T-1 15:00 HKT",
            "required_proof_before_model": "recompute inside each fold from the cutoff station summary only",
            "blocker": "",
        }
    if family == "upper_air":
        return {
            "timestamp_audit_status": "timestamp_audit_required",
            "allowed_for_future_walkforward": False,
            "cutoff_rule": "must prove sounding issue or available-at time is before T-1 15:00 HKT",
            "required_proof_before_model": "join upper-air rows to provider issue/retrieval metadata before promotion",
            "blocker": "upper-air valid time alone is not enough for point-in-time eligibility",
        }
    if family in {"marine_proxy", "hko_daily_climate"}:
        return {
            "timestamp_audit_status": "publication_lag_audit_required",
            "allowed_for_future_walkforward": False,
            "cutoff_rule": "lagged official daily values need publication-time proof before cutoff",
            "required_proof_before_model": "attach daily-climate publication timestamp proof row by row",
            "blocker": "official daily climate publication lag is not fully proven",
        }
    return {
        "timestamp_audit_status": "needs_manual_timestamp_review",
        "allowed_for_future_walkforward": False,
        "cutoff_rule": "unknown",
        "required_proof_before_model": "write a source-specific issue/available-at contract",
        "blocker": "missing source-specific timestamp rule",
    }


def is_allowed_feature_column(feature_name: str, frame: pd.DataFrame) -> bool:
    if feature_name in EXTRA_NON_FEATURE_EXACT:
        return False
    lowered = feature_name.lower()
    if any(token in lowered for token in EXTRA_NON_FEATURE_TOKENS):
        return False
    return pd.api.types.is_numeric_dtype(frame[feature_name])


def feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = []
    for column in numeric_feature_columns(frame):
        if is_allowed_feature_column(column, frame):
            columns.append(column)
    return columns


def capped_abs(value: float, *, divisor: float, cap: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(abs(value) / divisor, cap)


def diagnostic_score(
    *,
    corr_agreement_base_error: float,
    corr_agreement_base_abs_error: float,
    corr_agreement_abs_improvement: float,
    corr_active_base_error: float,
    base_error_spread_c: float,
    base_abs_error_spread_c: float,
    improvement_spread_c: float,
    agreement_vs_other_std_diff: float,
    agreement_rows: int,
    full_history_rows: int,
) -> float:
    signal = (
        2.0 * capped_abs(corr_agreement_base_error, divisor=1.0, cap=1.0)
        + 1.5 * capped_abs(corr_agreement_base_abs_error, divisor=1.0, cap=1.0)
        + 1.5 * capped_abs(corr_agreement_abs_improvement, divisor=1.0, cap=1.0)
        + 1.0 * capped_abs(corr_active_base_error, divisor=1.0, cap=1.0)
        + 1.0 * capped_abs(base_error_spread_c, divisor=1.0, cap=2.0)
        + 0.8 * capped_abs(base_abs_error_spread_c, divisor=0.8, cap=2.0)
        + 0.8 * capped_abs(improvement_spread_c, divisor=0.4, cap=2.0)
        + 0.5 * capped_abs(agreement_vs_other_std_diff, divisor=1.0, cap=2.0)
    )
    agreement_factor = min(1.0, max(0.0, agreement_rows / 60.0))
    history_factor = min(1.0, max(0.0, full_history_rows / MIN_LONG_HISTORY_ROWS))
    return float(signal * agreement_factor * history_factor)


def load_0099_artifacts() -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    missing = [
        path
        for path in (INPUT_0099_SUMMARY_PATH, INPUT_0099_TOP_PATH, INPUT_0099_DIAGNOSTICS_PATH)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"0100 requires 0099 artifacts first: {missing}")
    summary = json.loads(INPUT_0099_SUMMARY_PATH.read_text(encoding="utf-8"))
    top = pd.read_csv(INPUT_0099_TOP_PATH)
    diagnostics = pd.read_csv(INPUT_0099_DIAGNOSTICS_PATH)
    for frame, context in ((top, "0100 0099 top predictions"), (diagnostics, "0100 0099 diagnostics")):
        frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
        frame.dropna(subset=["target_date"], inplace=True)
        frame.sort_values("target_date", inplace=True)
        require_no_confirmation_dates(frame["target_date"], context=context)
    return summary, top.reset_index(drop=True), diagnostics.reset_index(drop=True)


def build_eval_frame(features: pd.DataFrame, top_0099: pd.DataFrame, diagnostics_0099: pd.DataFrame) -> pd.DataFrame:
    frame_0098 = load_0098_inputs()[0].copy()
    frame_0098["target_date"] = pd.to_datetime(frame_0098["target_date"], errors="coerce").dt.normalize()
    frame_0098 = frame_0098[frame_0098["target_date"].notna() & (frame_0098["target_date"] < CONFIRMATION_START)].copy()
    require_no_confirmation_dates(frame_0098["target_date"], context="0100 0098 working frame")
    base = frame_0098[
        [
            "target_date",
            "forecast_source_family",
            "target_tmax_c",
            "forecast_max_c",
            "candidate_prediction_c",
            "season",
            "frame_segment",
            "era_bucket",
        ]
    ].rename(columns={"candidate_prediction_c": "base_0094_prediction_c"})
    top = top_0099[["target_date", "candidate_id", "candidate_prediction_c", "candidate_error_c"]].rename(
        columns={
            "candidate_id": "best_0099_candidate_id",
            "candidate_prediction_c": "best_0099_prediction_c",
            "candidate_error_c": "best_0099_error_c",
        }
    )
    diagnostics = diagnostics_0099[
        [
            "target_date",
            "pair_bucket_label",
            "mam_submonth",
            "gate_active_row",
            "prior_rows",
            "prior_mean_residual_c",
            "prior_direction",
            "bucket_stable_allowed",
            "source_submonth_stable_allowed",
            "cell_policy_allowed",
            "specialist_active",
            "specialist_correction_c",
        ]
    ].copy()
    eval_frame = base.merge(top, on="target_date", how="inner").merge(diagnostics, on="target_date", how="inner")
    eval_frame = eval_frame.merge(features, on="target_date", how="inner", suffixes=("", "_feature_matrix"))
    if "target_tmax_c_feature_matrix" in eval_frame.columns:
        eval_frame.drop(columns=["target_tmax_c_feature_matrix"], inplace=True)
    for column in ("target_tmax_c", "forecast_max_c", "base_0094_prediction_c", "best_0099_prediction_c"):
        eval_frame[column] = pd.to_numeric(eval_frame[column], errors="coerce")
    eval_frame = eval_frame[
        eval_frame[["target_tmax_c", "forecast_max_c", "base_0094_prediction_c", "best_0099_prediction_c"]]
        .notna()
        .all(axis=1)
    ].copy()
    eval_frame["base_0094_error_c"] = eval_frame["base_0094_prediction_c"] - eval_frame["target_tmax_c"]
    eval_frame["base_0094_abs_error_c"] = eval_frame["base_0094_error_c"].abs()
    eval_frame["raw_forecast_error_c"] = eval_frame["forecast_max_c"] - eval_frame["target_tmax_c"]
    eval_frame["best_0099_error_c"] = eval_frame["best_0099_prediction_c"] - eval_frame["target_tmax_c"]
    eval_frame["best_0099_abs_error_c"] = eval_frame["best_0099_error_c"].abs()
    eval_frame["abs_improvement_0099_vs_0094_c"] = (
        eval_frame["base_0094_abs_error_c"] - eval_frame["best_0099_abs_error_c"]
    )
    eval_frame["agreement_row"] = bool_series(eval_frame["gate_active_row"]) & bool_series(eval_frame["cell_policy_allowed"])
    eval_frame["specialist_active_row"] = bool_series(eval_frame["specialist_active"])
    eval_frame["other_mam_gate_row"] = bool_series(eval_frame["gate_active_row"]) & ~eval_frame["agreement_row"]
    require_no_confirmation_dates(eval_frame["target_date"], context="0100 eval frame")
    return eval_frame.sort_values("target_date").reset_index(drop=True)


def summarize_feature(feature: str, full_features: pd.DataFrame, eval_frame: pd.DataFrame) -> dict[str, object]:
    family = atlas_family(feature)
    full_values = pd.to_numeric(full_features[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
    full_valid_mask = full_values.notna()
    full_dates = full_features.loc[full_valid_mask, "target_date"]
    agreement_mask = eval_frame["agreement_row"].astype(bool)
    active_mask = eval_frame["specialist_active_row"].astype(bool)
    other_mam_mask = eval_frame["other_mam_gate_row"].astype(bool)
    values = pd.to_numeric(eval_frame[feature], errors="coerce")

    corr_agree_error_n, corr_agree_error = safe_corr(
        values[agreement_mask],
        eval_frame.loc[agreement_mask, "base_0094_error_c"],
        min_rows=MIN_AGREEMENT_CORR_ROWS,
    )
    corr_agree_abs_n, corr_agree_abs = safe_corr(
        values[agreement_mask],
        eval_frame.loc[agreement_mask, "base_0094_abs_error_c"],
        min_rows=MIN_AGREEMENT_CORR_ROWS,
    )
    corr_agree_improve_n, corr_agree_improve = safe_corr(
        values[agreement_mask],
        eval_frame.loc[agreement_mask, "abs_improvement_0099_vs_0094_c"],
        min_rows=MIN_AGREEMENT_CORR_ROWS,
    )
    corr_active_error_n, corr_active_error = safe_corr(
        values[active_mask],
        eval_frame.loc[active_mask, "base_0094_error_c"],
        min_rows=MIN_ACTIVE_CORR_ROWS,
    )
    group_n, reference_n, std_diff = standardized_mean_diff(values[agreement_mask], values[other_mam_mask])
    error_spread = quantile_response_spread(values[agreement_mask], eval_frame.loc[agreement_mask, "base_0094_error_c"])
    abs_error_spread = quantile_response_spread(
        values[agreement_mask],
        eval_frame.loc[agreement_mask, "base_0094_abs_error_c"],
    )
    improvement_spread = quantile_response_spread(
        values[agreement_mask],
        eval_frame.loc[agreement_mask, "abs_improvement_0099_vs_0094_c"],
    )
    timestamp = feature_timestamp_status(feature, family)
    full_history_rows = int(full_valid_mask.sum())
    agreement_rows = int(values[agreement_mask].replace([np.inf, -np.inf], np.nan).notna().sum())
    active_rows = int(values[active_mask].replace([np.inf, -np.inf], np.nan).notna().sum())
    score = diagnostic_score(
        corr_agreement_base_error=corr_agree_error,
        corr_agreement_base_abs_error=corr_agree_abs,
        corr_agreement_abs_improvement=corr_agree_improve,
        corr_active_base_error=corr_active_error,
        base_error_spread_c=float(error_spread["spread_c"]),
        base_abs_error_spread_c=float(abs_error_spread["spread_c"]),
        improvement_spread_c=float(improvement_spread["spread_c"]),
        agreement_vs_other_std_diff=std_diff,
        agreement_rows=agreement_rows,
        full_history_rows=full_history_rows,
    )
    return {
        "feature": feature,
        "family": family,
        "station_ids": station_ids_in_feature(feature),
        "full_history_rows": full_history_rows,
        "first_non_null_date": "" if full_dates.empty else full_dates.min().date().isoformat(),
        "last_non_null_date": "" if full_dates.empty else full_dates.max().date().isoformat(),
        "long_history_pass": full_history_rows >= MIN_LONG_HISTORY_ROWS,
        "agreement_non_null_rows": agreement_rows,
        "specialist_active_non_null_rows": active_rows,
        "agreement_mean": float(values[agreement_mask].mean()) if agreement_rows else math.nan,
        "other_mam_gate_mean": float(values[other_mam_mask].mean()) if reference_n else math.nan,
        "agreement_vs_other_group_rows": group_n,
        "agreement_vs_other_reference_rows": reference_n,
        "agreement_vs_other_std_diff": std_diff,
        "corr_agreement_base_error_n": corr_agree_error_n,
        "corr_agreement_base_error": corr_agree_error,
        "corr_agreement_base_abs_error_n": corr_agree_abs_n,
        "corr_agreement_base_abs_error": corr_agree_abs,
        "corr_agreement_abs_improvement_n": corr_agree_improve_n,
        "corr_agreement_abs_improvement": corr_agree_improve,
        "corr_active_base_error_n": corr_active_error_n,
        "corr_active_base_error": corr_active_error,
        "base_error_spread_rows": error_spread["spread_rows"],
        "base_error_spread_bucket_count": error_spread["spread_bucket_count"],
        "base_error_spread_c": error_spread["spread_c"],
        "base_abs_error_spread_c": abs_error_spread["spread_c"],
        "improvement_spread_c": improvement_spread["spread_c"],
        "diagnostic_score": score,
        **timestamp,
    }


def summarize_family(atlas: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if atlas.empty:
        return pd.DataFrame()
    for family, group in atlas.groupby("family", observed=True):
        sorted_group = group.sort_values(["diagnostic_score", "full_history_rows"], ascending=[False, False])
        top = sorted_group.iloc[0]
        allowed = group["allowed_for_future_walkforward"].astype(bool)
        rows.append(
            {
                "family": family,
                "feature_count": int(len(group)),
                "allowed_feature_count": int(allowed.sum()),
                "long_history_feature_count": int(group["long_history_pass"].astype(bool).sum()),
                "top_feature": top["feature"],
                "top_diagnostic_score": float(top["diagnostic_score"]),
                "top_timestamp_status": top["timestamp_audit_status"],
                "top_corr_agreement_base_error": top["corr_agreement_base_error"],
                "top_base_error_spread_c": top["base_error_spread_c"],
            }
        )
    return pd.DataFrame(rows).sort_values(["top_diagnostic_score", "feature_count"], ascending=[False, False])


def agreement_rows(eval_frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "target_date",
        "forecast_source_family",
        "season",
        "frame_segment",
        "era_bucket",
        "pair_bucket_label",
        "mam_submonth",
        "prior_rows",
        "prior_direction",
        "prior_mean_residual_c",
        "specialist_correction_c",
        "target_tmax_c",
        "forecast_max_c",
        "base_0094_prediction_c",
        "base_0094_error_c",
        "best_0099_prediction_c",
        "best_0099_error_c",
        "abs_improvement_0099_vs_0094_c",
        "specialist_active_row",
    ]
    return eval_frame.loc[eval_frame["agreement_row"].astype(bool), columns].copy()


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame]:
    summary_0099, top_0099, diagnostics_0099 = load_0099_artifacts()
    features = load_feature_matrix()
    eval_frame = build_eval_frame(features, top_0099, diagnostics_0099)
    columns = feature_columns(features)
    rows = [summarize_feature(column, features, eval_frame) for column in columns]
    atlas = pd.DataFrame(rows)
    atlas = atlas[atlas["long_history_pass"].astype(bool)].copy()
    atlas = atlas.sort_values(["diagnostic_score", "full_history_rows"], ascending=[False, False]).reset_index(drop=True)
    family_summary = summarize_family(atlas)
    agreement = agreement_rows(eval_frame)
    dates = pd.to_datetime(eval_frame["target_date"], errors="coerce")
    summary = {
        "generated_at_utc": now_utc(),
        "folder": FOLDER_NAME,
        "input_0099_candidate": summary_0099["best_0099_candidate"],
        "input_0099_mae": float(summary_0099["best_0099_mae"]),
        "input_0099_rmse": float(summary_0099["best_0099_rmse"]),
        "feature_matrix_path": str(FEATURE_MATRIX_PATH),
        "rows": int(len(eval_frame)),
        "first_target_date": dates.min().date().isoformat(),
        "last_target_date": dates.max().date().isoformat(),
        "feature_columns_scanned": int(len(columns)),
        "long_history_feature_columns_ranked": int(len(atlas)),
        "min_long_history_rows": int(MIN_LONG_HISTORY_ROWS),
        "agreement_rows": int(eval_frame["agreement_row"].sum()),
        "specialist_active_rows": int(eval_frame["specialist_active_row"].sum()),
        "other_mam_gate_rows": int(eval_frame["other_mam_gate_row"].sum()),
        "allowed_for_future_walkforward_features": int(atlas["allowed_for_future_walkforward"].astype(bool).sum()),
        "timestamp_blocked_or_review_features": int((~atlas["allowed_for_future_walkforward"].astype(bool)).sum()),
        "family_count": int(family_summary["family"].nunique()) if not family_summary.empty else 0,
        "top_feature": "" if atlas.empty else str(atlas.iloc[0]["feature"]),
        "top_feature_family": "" if atlas.empty else str(atlas.iloc[0]["family"]),
        "top_feature_score": math.nan if atlas.empty else float(atlas.iloc[0]["diagnostic_score"]),
        "top_future_allowed_feature": "",
        "top_future_allowed_feature_family": "",
        "top_future_allowed_score": math.nan,
        "confirmation_start": CONFIRMATION_START.date().isoformat(),
        "uses_2024_plus_rows": False,
        "status": "stable_mam_cell_feature_atlas_complete",
        "next_recommended_task": (
            "Run 0101_stable_mam_cell_feature_specialists: convert the top 0100 future-allowed station and "
            "lagged-target features into fold-local specialist gates, while keeping upper-air and daily marine "
            "features diagnostic-only until timestamp/publication proof is attached."
        ),
    }
    allowed = atlas[atlas["allowed_for_future_walkforward"].astype(bool)].copy()
    if not allowed.empty:
        best_allowed = allowed.iloc[0]
        summary["top_future_allowed_feature"] = str(best_allowed["feature"])
        summary["top_future_allowed_feature_family"] = str(best_allowed["family"])
        summary["top_future_allowed_score"] = float(best_allowed["diagnostic_score"])
    require_no_confirmation_dates(eval_frame["target_date"], context="0100 outputs")
    return atlas, family_summary, agreement, summary, eval_frame


def build_readme(
    *,
    summary: dict[str, object],
    atlas: pd.DataFrame,
    family_summary: pd.DataFrame,
    agreement: pd.DataFrame,
) -> str:
    display_cols = [
        "feature",
        "family",
        "diagnostic_score",
        "allowed_for_future_walkforward",
        "timestamp_audit_status",
        "agreement_non_null_rows",
        "corr_agreement_base_error",
        "corr_agreement_base_abs_error",
        "corr_agreement_abs_improvement",
        "base_error_spread_c",
        "base_abs_error_spread_c",
        "improvement_spread_c",
        "agreement_vs_other_std_diff",
        "first_non_null_date",
        "last_non_null_date",
        "station_ids",
    ]
    top_allowed = atlas[atlas["allowed_for_future_walkforward"].astype(bool)].head(25)
    diagnostic_blocked = atlas[~atlas["allowed_for_future_walkforward"].astype(bool)].head(25)
    return f"""# 0100 Stable MAM Cell Feature Atlas

Generated: `{summary['generated_at_utc']}`

## Purpose

`0099` produced the current pre-2024 champion by applying a tiny residual correction only when a MAM bucket-direction cell and a source/submonth/direction cell agreed. That correction improved MAE only slightly, so `0100` does not promote a new model. It asks a narrower question:

Which long-history features best explain the rows where that stable MAM cell exists, and which of those features are safe enough to test next?

The scan covers numeric long-history features from the existing feature matrix, including target memory, regional station network features, upper-air features, HKO daily climate proxies, and marine proxies. It keeps 2024+ rows sealed.

## Main Result

| Field | Value |
|---|---|
| Status | `{summary['status']}` |
| Input 0099 candidate | `{summary['input_0099_candidate']}` |
| Input 0099 MAE | `{summary['input_0099_mae']}` |
| Input 0099 RMSE | `{summary['input_0099_rmse']}` |
| Date range scored | `{summary['first_target_date']}` to `{summary['last_target_date']}` |
| Feature columns scanned | `{summary['feature_columns_scanned']}` |
| Long-history feature columns ranked | `{summary['long_history_feature_columns_ranked']}` |
| Minimum long-history rows | `{summary['min_long_history_rows']}` |
| Stable agreement rows | `{summary['agreement_rows']}` |
| Specialist active rows | `{summary['specialist_active_rows']}` |
| Other MAM gate rows | `{summary['other_mam_gate_rows']}` |
| Future-allowed ranked features | `{summary['allowed_for_future_walkforward_features']}` |
| Timestamp-blocked/review features | `{summary['timestamp_blocked_or_review_features']}` |
| Top feature | `{summary['top_feature']}` |
| Top feature family | `{summary['top_feature_family']}` |
| Top future-allowed feature | `{summary['top_future_allowed_feature']}` |
| Top future-allowed family | `{summary['top_future_allowed_feature_family']}` |
| 2024+ rows used | `{summary['uses_2024_plus_rows']}` |

## How To Read This

The atlas is a feature-discovery map, not a deployed predictor. A high diagnostic score means the feature separates residual behavior inside the stable MAM rows, especially by correlation with the 0094 base error, absolute error, and whether the 0099 correction helped. It is useful for choosing the next fold-local specialist tests.

The `allowed_for_future_walkforward` column is the key leakage guard. Station features and lagged target-memory features can move to fold-local testing. Upper-air and official daily/marine features remain diagnostic-only until their issue, available-at, or publication timestamps are proven before the forecast cutoff.

## Family Summary

{markdown_table(family_summary, max_rows=30)}

## Top Ranked Features

{markdown_table(atlas[display_cols].head(35), max_rows=35)}

## Top Future-Allowed Features

{markdown_table(top_allowed[display_cols], max_rows=25)}

## Top Diagnostic-Only Or Timestamp-Blocked Features

{markdown_table(diagnostic_blocked[display_cols], max_rows=25)}

## Stable Agreement Row Sample

{markdown_table(agreement.head(40), max_rows=40)}

## Leakage Controls

All scored target dates are before `{summary['confirmation_start']}`. The script imports 0099 diagnostics and predictions, merges them to the long-history feature matrix by `target_date`, and then rejects any 2024+ row before writing outputs. It excludes target outcome columns, official/forecast columns, candidate error columns, and known diagnostic/outcome tokens from the feature scan.

No model is fit in this experiment. No current-row target is used as a feature. Feature promotion is explicitly blocked unless its timestamp status allows future walk-forward use.

## Artifacts

- `artifacts/feature_atlas.csv`: full ranked long-history feature atlas.
- `artifacts/family_summary.csv`: feature family counts and best rows.
- `artifacts/agreement_rows.csv`: the stable MAM rows being explained.
- `artifacts/top_station.csv`: top ISD/station network features.
- `artifacts/top_target_memory.csv`: top lagged target-memory features.
- `artifacts/top_upper_air.csv`: diagnostic-only upper-air features.
- `artifacts/top_marine.csv`: diagnostic-only marine proxy features.
- `artifacts/top_hko_daily.csv`: diagnostic-only official daily climate features.
- `artifacts/summary.json`: machine-readable summary.

## Exact Next Recommended Task

{summary['next_recommended_task']}
"""


def update_milestones(summary: dict[str, object], atlas: pd.DataFrame, family_summary: pd.DataFrame) -> None:
    display_cols = [
        "feature",
        "family",
        "diagnostic_score",
        "allowed_for_future_walkforward",
        "timestamp_audit_status",
        "corr_agreement_base_error",
        "base_error_spread_c",
        "agreement_non_null_rows",
    ]
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_0100_stable_mam_cell_feature_atlas.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Current evidence | Status |
|---|---|---|
| Input champion | `{summary['input_0099_candidate']}` | From 0099 |
| Input MAE/RMSE | `{summary['input_0099_mae']}` / `{summary['input_0099_rmse']}` | Pre-2024 only |
| Stable agreement rows | `{summary['agreement_rows']}` | Explained by atlas |
| Specialist active rows | `{summary['specialist_active_rows']}` | Explained by atlas |
| Long-history features ranked | `{summary['long_history_feature_columns_ranked']}` | >= `{summary['min_long_history_rows']}` non-null rows |
| Future-allowed features | `{summary['allowed_for_future_walkforward_features']}` | Candidate next tests |
| Timestamp-blocked/review features | `{summary['timestamp_blocked_or_review_features']}` | Diagnostic only |
| Top feature | `{summary['top_feature']}` | `{summary['top_feature_family']}` |
| Top future-allowed feature | `{summary['top_future_allowed_feature']}` | `{summary['top_future_allowed_feature_family']}` |
| Leakage | `0` 2024+ rows | PASS |

Family summary:

{markdown_table(family_summary, max_rows=20)}

Top ranked features:

{markdown_table(atlas[display_cols].head(15), max_rows=15)}
"""
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="0100 Stable MAM Cell Feature Atlas",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="Exact Next Recommended Codex Task",
        section=str(summary["next_recommended_task"]),
    )


def run() -> dict[str, object]:
    atlas, family_summary, agreement, summary, _eval_frame = build_outputs()
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    write_csv(artifacts / "feature_atlas.csv", atlas)
    write_csv(artifacts / "family_summary.csv", family_summary)
    write_csv(artifacts / "agreement_rows.csv", agreement)
    write_csv(artifacts / "top_station.csv", atlas[atlas["family"].eq("isd_station_network")].head(TOP_FEATURES_PER_FAMILY))
    write_csv(artifacts / "top_target_memory.csv", atlas[atlas["family"].eq("target_memory")].head(TOP_FEATURES_PER_FAMILY))
    write_csv(artifacts / "top_upper_air.csv", atlas[atlas["family"].eq("upper_air")].head(TOP_FEATURES_PER_FAMILY))
    write_csv(artifacts / "top_marine.csv", atlas[atlas["family"].eq("marine_proxy")].head(TOP_FEATURES_PER_FAMILY))
    write_csv(artifacts / "top_hko_daily.csv", atlas[atlas["family"].eq("hko_daily_climate")].head(TOP_FEATURES_PER_FAMILY))
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "stable_mam_cell_feature_atlas_manifest.json", summary)
    write_text(folder / "README.md", build_readme(summary=summary, atlas=atlas, family_summary=family_summary, agreement=agreement))
    update_milestones(summary, atlas, family_summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-summary", action="store_true", help="Print JSON summary after writing artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run()
    if args.print_summary:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
