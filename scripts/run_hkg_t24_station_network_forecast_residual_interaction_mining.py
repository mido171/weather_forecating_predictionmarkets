from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hkg_t24_beastmode_signal_discovery import (  # noqa: E402
    CONFIRMATION_START,
    RESEARCH_ROOT,
    allowed_signal_column,
    feature_family,
    markdown_table,
    require_no_confirmation_dates,
    score_prediction_frame,
    write_csv,
    write_json,
    write_text,
)
from scripts.run_hkg_t24_forecast_history_state_deepening import (  # noqa: E402
    FOLDER_NAME as FORECAST_HISTORY_FOLDER,
)
from scripts.run_hkg_t24_forecast_history_state_deepening import (  # noqa: E402
    add_forecast_history_state_features,
)
from scripts.run_hkg_t24_forecast_revision_momentum_deep_dive import (  # noqa: E402
    add_revision_features,
    build_failure_frame,
)

FOLDER_NAME = "0039_station_network_residuals"
ANCHOR_CANDIDATE_ID = "trust_history_forecast_vs_prior7_bin_best_same_source"
ANCHOR_PATH = (
    RESEARCH_ROOT
    / FORECAST_HISTORY_FOLDER
    / "artifacts"
    / "top_forecast_history_predictions.csv"
)
DISCOVERY_END = pd.Timestamp("2017-12-31")
LATE_EVAL_START = pd.Timestamp("2018-01-01")
MIN_SCAN_ROWS = 500
MIN_DISCOVERY_ROWS = 500
TOP_DISCOVERY_FEATURES = 30
MIN_BUCKET_HISTORY = 160
MIN_CELL_HISTORY = 45
SHRINKAGE = 90.0
CORRECTION_CLIP_C = 2.0
STATION_RE = re.compile(r"^isd_station_(?P<metric>.+)_(?P<station>\d{5,6}_\d{5})$")


@dataclass(frozen=True)
class InteractionSpec:
    feature: str
    state_cols: tuple[str, ...]
    same_source: bool
    min_cell_history: int = MIN_CELL_HISTORY
    shrinkage: float = SHRINKAGE
    correction_clip_c: float = CORRECTION_CLIP_C


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 140) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([math.nan] * len(frame), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def safe_corr(left: pd.Series, right: pd.Series, *, method: str = "pearson", min_rows: int = MIN_SCAN_ROWS) -> float:
    pair = pd.concat([pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")], axis=1).dropna()
    if len(pair) < min_rows or pair.iloc[:, 0].nunique() <= 2 or pair.iloc[:, 1].nunique() <= 2:
        return math.nan
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method))


def qtail_delta(feature: pd.Series, target: pd.Series, *, min_rows: int = MIN_SCAN_ROWS) -> float:
    pair = pd.DataFrame(
        {
            "feature": pd.to_numeric(feature, errors="coerce"),
            "target": pd.to_numeric(target, errors="coerce"),
        }
    ).dropna()
    if len(pair) < min_rows or pair["feature"].nunique() <= 2:
        return math.nan
    low = pair["feature"].quantile(0.10)
    high = pair["feature"].quantile(0.90)
    low_mean = pair.loc[pair["feature"] <= low, "target"].mean()
    high_mean = pair.loc[pair["feature"] >= high, "target"].mean()
    return float(high_mean - low_mean)


def parse_station_feature(column: str) -> tuple[str, str] | None:
    match = STATION_RE.match(column)
    if not match:
        return None
    return match.group("station"), match.group("metric")


def load_anchor_predictions() -> pd.DataFrame:
    if not ANCHOR_PATH.exists():
        raise FileNotFoundError(f"Missing 0038 anchor predictions: {ANCHOR_PATH}")
    predictions = pd.read_csv(
        ANCHOR_PATH,
        usecols=[
            "target_date",
            "forecast_source_family",
            "target_tmax_c",
            "official_raw",
            "expert_prediction_c",
            "candidate_id",
            "selected_family",
        ],
    )
    predictions = predictions[predictions["candidate_id"].astype(str).eq(ANCHOR_CANDIDATE_ID)].copy()
    if predictions.empty:
        raise RuntimeError(f"Anchor candidate not found in 0038 top predictions: {ANCHOR_CANDIDATE_ID}")
    predictions["target_date"] = pd.to_datetime(predictions["target_date"], errors="coerce").dt.normalize()
    require_no_confirmation_dates(predictions["target_date"], context="0039 anchor predictions")
    predictions = predictions.rename(columns={"expert_prediction_c": "anchor_0038_c"})
    return predictions.drop_duplicates(["target_date", "forecast_source_family"], keep="last")


def station_metric_groups(frame: pd.DataFrame) -> dict[str, dict[str, str]]:
    groups: dict[str, dict[str, str]] = {}
    for column in frame.columns:
        parsed = parse_station_feature(column)
        if parsed is None:
            continue
        station, metric = parsed
        values = numeric_column(frame, column)
        if int(values.notna().sum()) < MIN_SCAN_ROWS or values.nunique(dropna=True) <= 2:
            continue
        groups.setdefault(metric, {})[station] = column
    return groups


def add_station_network_derived_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = frame.copy()
    new_cols: dict[str, pd.Series] = {}
    rows: list[dict[str, object]] = []
    for metric, station_cols in station_metric_groups(out).items():
        if len(station_cols) < 3:
            continue
        safe_metric = slug(metric, limit=80)
        matrix = pd.DataFrame({station: numeric_column(out, col) for station, col in station_cols.items()})
        mean_col = f"derived_{safe_metric}_network_mean"
        spread_col = f"derived_{safe_metric}_network_spread"
        new_cols[mean_col] = matrix.mean(axis=1, skipna=True)
        new_cols[spread_col] = matrix.max(axis=1, skipna=True) - matrix.min(axis=1, skipna=True)
        rows.extend(
            [
                {"feature": mean_col, "type": "station_network_mean", "source_metric": metric, "source_stations": len(station_cols)},
                {"feature": spread_col, "type": "station_network_spread", "source_metric": metric, "source_stations": len(station_cols)},
            ]
        )
        for station, column in station_cols.items():
            dev_col = f"derived_{safe_metric}_{station}_minus_network_mean"
            new_cols[dev_col] = numeric_column(out, column) - new_cols[mean_col]
            rows.append(
                {
                    "feature": dev_col,
                    "type": "station_minus_network_mean",
                    "source_metric": metric,
                    "source_stations": station,
                }
            )
        for station_a, station_b in combinations(sorted(station_cols), 2):
            col_a = station_cols[station_a]
            col_b = station_cols[station_b]
            pair_col = f"derived_{safe_metric}_{station_a}_minus_{station_b}"
            new_cols[pair_col] = numeric_column(out, col_a) - numeric_column(out, col_b)
            rows.append(
                {
                    "feature": pair_col,
                    "type": "station_pair_delta",
                    "source_metric": metric,
                    "source_stations": f"{station_a},{station_b}",
                }
            )
    if new_cols:
        out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1).copy()
    return out, pd.DataFrame(rows)


def build_analysis_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_frame, _prior_systems = build_failure_frame()
    raw_frame = add_revision_features(raw_frame)
    raw_frame["target_date"] = pd.to_datetime(raw_frame["target_date"], errors="coerce").dt.normalize()
    require_no_confirmation_dates(raw_frame["target_date"], context="0039 raw frame")
    raw_frame, _state_catalog = add_forecast_history_state_features(raw_frame)
    raw_frame, derived_catalog = add_station_network_derived_features(raw_frame)
    anchor = load_anchor_predictions()
    frame = raw_frame.merge(
        anchor[
            [
                "target_date",
                "forecast_source_family",
                "anchor_0038_c",
                "selected_family",
            ]
        ],
        on=["target_date", "forecast_source_family"],
        how="inner",
        validate="one_to_one",
    )
    frame["official_raw"] = pd.to_numeric(frame["official_raw"], errors="coerce")
    frame["anchor_0038_c"] = pd.to_numeric(frame["anchor_0038_c"], errors="coerce")
    frame["anchor_residual_c"] = pd.to_numeric(frame["target_tmax_c"], errors="coerce") - frame["anchor_0038_c"]
    frame["anchor_abs_error_c"] = frame["anchor_residual_c"].abs()
    frame["official_residual_c"] = pd.to_numeric(frame["target_tmax_c"], errors="coerce") - frame["official_raw"]
    frame["official_abs_error_c"] = frame["official_residual_c"].abs()
    return frame.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True), derived_catalog


def station_network_feature_allowed(column: str) -> bool:
    if column in {
        "target_tmax_c",
        "anchor_0038_c",
        "anchor_residual_c",
        "anchor_abs_error_c",
        "official_residual_c",
        "official_abs_error_c",
        "official_raw",
        "forecast_max_c",
        "forecast_min_c",
    }:
        return False
    if column.startswith(("target_", "official_", "anchor_")):
        allowed_target_lag = column.startswith(("target_lag", "target_roll", "target_spell", "target_abs_change"))
        if not allowed_target_lag:
            return False
    if not allowed_signal_column(column):
        return False
    prefixes = (
        "isd_",
        "daily_",
        "derived_",
        "thermal_",
        "dew_",
        "slp_",
        "pressure_plane_",
        "abs_north_south",
        "clim_",
        "trajectory_",
        "spell_",
        "volatility_",
        "spectral_",
        "target_lag",
        "target_roll",
        "target_abs_change",
    )
    return column.startswith(prefixes)


def station_network_candidate_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        if not station_network_feature_allowed(column):
            continue
        values = numeric_column(frame, column)
        if int(values.notna().sum()) < MIN_SCAN_ROWS or values.nunique(dropna=True) <= 2:
            continue
        columns.append(column)
    return columns


def source_stability(frame: pd.DataFrame, feature: str) -> tuple[float, int]:
    corrs: list[float] = []
    for _source, group in frame.groupby("forecast_source_family", observed=True):
        corr = safe_corr(group[feature], group["anchor_residual_c"], min_rows=220)
        if np.isfinite(corr):
            corrs.append(corr)
    if not corrs:
        return math.nan, 0
    signs = {1 if corr > 0 else -1 for corr in corrs if corr != 0.0}
    return float(max(abs(corr) for corr in corrs)), int(len(signs) == 1)


def scan_station_network_information(frame: pd.DataFrame, *, discovery_only: bool) -> pd.DataFrame:
    work = frame[pd.to_datetime(frame["target_date"]) <= DISCOVERY_END].copy() if discovery_only else frame.copy()
    rows: list[dict[str, object]] = []
    for column in station_network_candidate_columns(work):
        values = numeric_column(work, column)
        n = int(values.notna().sum())
        if n < MIN_DISCOVERY_ROWS or values.nunique(dropna=True) <= 2:
            continue
        valid_dates = pd.to_datetime(work.loc[values.notna(), "target_date"], errors="coerce")
        max_source_corr, source_sign_consistent = source_stability(work, column)
        rows.append(
            {
                "feature": column,
                "family": feature_family(column),
                "n": n,
                "coverage": float(n / len(work)),
                "first_date": "" if valid_dates.empty else str(valid_dates.min().date()),
                "last_date": "" if valid_dates.empty else str(valid_dates.max().date()),
                "target_pearson": safe_corr(values, work["target_tmax_c"]),
                "target_spearman": safe_corr(values, work["target_tmax_c"], method="spearman"),
                "anchor_residual_pearson": safe_corr(values, work["anchor_residual_c"]),
                "anchor_residual_spearman": safe_corr(values, work["anchor_residual_c"], method="spearman"),
                "anchor_abs_error_pearson": safe_corr(values, work["anchor_abs_error_c"]),
                "anchor_abs_error_spearman": safe_corr(values, work["anchor_abs_error_c"], method="spearman"),
                "official_residual_pearson": safe_corr(values, work["official_residual_c"]),
                "target_q90_minus_q10_c": qtail_delta(values, work["target_tmax_c"]),
                "anchor_residual_q90_minus_q10_c": qtail_delta(values, work["anchor_residual_c"]),
                "anchor_abs_error_q90_minus_q10_c": qtail_delta(values, work["anchor_abs_error_c"]),
                "max_source_abs_anchor_residual_corr": max_source_corr,
                "source_corr_sign_consistent": source_sign_consistent,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["max_abs_anchor_residual_corr"] = out[["anchor_residual_pearson", "anchor_residual_spearman"]].abs().max(axis=1)
    out["max_abs_anchor_abs_error_corr"] = out[["anchor_abs_error_pearson", "anchor_abs_error_spearman"]].abs().max(axis=1)
    out["max_abs_target_corr"] = out[["target_pearson", "target_spearman"]].abs().max(axis=1)
    out["interaction_priority"] = (
        out["max_abs_anchor_residual_corr"].fillna(0.0) * 3.0
        + out["max_abs_anchor_abs_error_corr"].fillna(0.0) * 1.5
        + out["anchor_residual_q90_minus_q10_c"].abs().fillna(0.0) * 0.25
        + out["anchor_abs_error_q90_minus_q10_c"].abs().fillna(0.0) * 0.15
        + out["max_source_abs_anchor_residual_corr"].fillna(0.0) * 0.75
        + out["max_abs_target_corr"].fillna(0.0) * 0.15
    )
    return out.sort_values("interaction_priority", ascending=False).reset_index(drop=True)


def past_only_tercile_bucket(
    values: pd.Series,
    dates: pd.Series,
    *,
    min_history: int = MIN_BUCKET_HISTORY,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    normalized_dates = pd.to_datetime(dates, errors="coerce").dt.normalize()
    out = pd.Series(["missing"] * len(values), index=values.index, dtype="object")
    for current_date in sorted(normalized_dates.dropna().unique()):
        prior = numeric[(normalized_dates < current_date) & numeric.notna()]
        current_mask = normalized_dates.eq(current_date)
        if len(prior) < min_history or prior.nunique() < 3:
            continue
        low = float(prior.quantile(1.0 / 3.0))
        high = float(prior.quantile(2.0 / 3.0))
        current_values = numeric[current_mask]
        labels = np.where(current_values <= low, "low", np.where(current_values >= high, "high", "mid"))
        labels = np.where(current_values.notna(), labels, "missing")
        out.loc[current_mask] = labels
    return out


def selected_feature_catalog(discovery_scan: pd.DataFrame) -> pd.DataFrame:
    if discovery_scan.empty:
        return discovery_scan
    selected = discovery_scan.head(TOP_DISCOVERY_FEATURES).copy()
    selected["selection_window"] = f"<= {DISCOVERY_END.date()}"
    selected["selection_rule"] = "top pre-2018 station/network interaction_priority"
    return selected


def add_selected_feature_buckets(frame: pd.DataFrame, selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = frame.copy()
    rows: list[dict[str, object]] = []
    for rank, row in enumerate(selected.itertuples(index=False), start=1):
        feature = str(row.feature)
        bucket_col = f"bucket_{rank:02d}_{slug(feature, limit=95)}"
        out[bucket_col] = past_only_tercile_bucket(out[feature], out["target_date"])
        rows.append(
            {
                "rank": rank,
                "feature": feature,
                "bucket_col": bucket_col,
                "family": str(row.family),
                "interaction_priority": float(row.interaction_priority),
                "anchor_residual_spearman": float(row.anchor_residual_spearman)
                if np.isfinite(row.anchor_residual_spearman)
                else math.nan,
                "anchor_residual_q90_minus_q10_c": float(row.anchor_residual_q90_minus_q10_c)
                if np.isfinite(row.anchor_residual_q90_minus_q10_c)
                else math.nan,
            }
        )
    return out, pd.DataFrame(rows)


def state_columns_for_specs(frame: pd.DataFrame) -> tuple[tuple[str, ...], ...]:
    candidates = (
        (),
        ("meta_forecast_vs_prior7_bin",),
        ("meta_forecast_vs_prior7_sign",),
        ("meta_forecast_jump_sign",),
        ("meta_forecast_range_change_sign",),
        ("meta_forecast_history_state",),
        ("meta_revision_range_state",),
        ("meta_text_signal_state",),
        ("meta_month",),
    )
    out: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for cols in candidates:
        filtered = tuple(col for col in cols if col in frame.columns)
        if filtered in seen:
            continue
        seen.add(filtered)
        out.append(filtered)
    return tuple(out)


def build_interaction_specs(frame: pd.DataFrame, bucket_catalog: pd.DataFrame) -> list[InteractionSpec]:
    specs: list[InteractionSpec] = []
    for row in bucket_catalog.itertuples(index=False):
        bucket_col = str(row.bucket_col)
        for state_cols in state_columns_for_specs(frame):
            for same_source in (False, True):
                specs.append(InteractionSpec(feature=bucket_col, state_cols=state_cols, same_source=same_source))
    return specs


def candidate_id(spec: InteractionSpec) -> str:
    source = "same_source" if spec.same_source else "all_prior"
    states = "feature_only" if not spec.state_cols else "_".join(col.replace("meta_", "") for col in spec.state_cols)
    return slug(f"station_network_{spec.feature}_{states}_{source}")


def past_only_group_residual_predictions(frame: pd.DataFrame, spec: InteractionSpec) -> pd.DataFrame:
    ordered = frame.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True).copy()
    key_cols = [spec.feature, *spec.state_cols]
    if spec.same_source:
        key_cols.append("forecast_source_family")
    work_cols = list(
        dict.fromkeys(
            [
                "target_date",
                "forecast_source_family",
                "target_tmax_c",
                "official_raw",
                "anchor_0038_c",
                "anchor_residual_c",
                *key_cols,
            ]
        )
    )
    work = ordered[work_cols].copy()
    for col in key_cols:
        work[col] = work[col].fillna("missing").astype(str)
    valid = work["anchor_residual_c"].notna() & work["anchor_0038_c"].notna() & work["target_tmax_c"].notna()
    work["residual_for_sum"] = np.where(valid, pd.to_numeric(work["anchor_residual_c"], errors="coerce"), 0.0)
    work["residual_count"] = valid.astype(float)
    day = (
        work.groupby([*key_cols, "target_date"], observed=True, dropna=False)
        .agg(day_residual_sum=("residual_for_sum", "sum"), day_residual_count=("residual_count", "sum"))
        .reset_index()
        .sort_values([*key_cols, "target_date"])
    )
    grouped = day.groupby(key_cols, observed=True, dropna=False)
    day["prior_residual_sum"] = grouped["day_residual_sum"].cumsum() - day["day_residual_sum"]
    day["prior_residual_count"] = grouped["day_residual_count"].cumsum() - day["day_residual_count"]
    work = work.merge(
        day[[*key_cols, "target_date", "prior_residual_sum", "prior_residual_count"]],
        on=[*key_cols, "target_date"],
        how="left",
        validate="many_to_one",
    )
    prior_count = pd.to_numeric(work["prior_residual_count"], errors="coerce").fillna(0.0)
    prior_mean = np.divide(
        pd.to_numeric(work["prior_residual_sum"], errors="coerce").fillna(0.0),
        prior_count,
        out=np.zeros(len(work), dtype=float),
        where=prior_count.to_numpy(dtype=float) > 0,
    )
    shrink = prior_count / (prior_count + spec.shrinkage)
    correction = np.clip(prior_mean * shrink, -spec.correction_clip_c, spec.correction_clip_c)
    correction = np.where(prior_count >= spec.min_cell_history, correction, 0.0)
    anchor = pd.to_numeric(work["anchor_0038_c"], errors="coerce")

    out = ordered[["target_date", "forecast_source_family", "target_tmax_c", "official_raw", "anchor_0038_c"]].copy()
    out["candidate_prediction_c"] = anchor + correction
    out["residual_correction_c"] = correction
    out["prior_cell_rows"] = prior_count.astype(int)
    out["interaction_key"] = work[key_cols].astype(str).agg("|".join, axis=1)
    out["candidate_id"] = candidate_id(spec)
    out["feature_bucket_col"] = spec.feature
    out["state_cols"] = ",".join(spec.state_cols)
    out["same_source"] = spec.same_source
    return out


def score_candidate(predictions: pd.DataFrame, *, spec: InteractionSpec) -> dict[str, object]:
    score = score_prediction_frame(predictions.rename(columns={"candidate_prediction_c": "prediction"}), "prediction")
    anchor = score_prediction_frame(predictions.rename(columns={"anchor_0038_c": "prediction"}), "prediction")
    official = score_prediction_frame(predictions.rename(columns={"official_raw": "prediction"}), "prediction")
    late = predictions[pd.to_datetime(predictions["target_date"]) >= LATE_EVAL_START].copy()
    late_score = score_prediction_frame(late.rename(columns={"candidate_prediction_c": "prediction"}), "prediction")
    late_anchor = score_prediction_frame(late.rename(columns={"anchor_0038_c": "prediction"}), "prediction")
    corrected_rows = int((pd.to_numeric(predictions["residual_correction_c"], errors="coerce").abs() > 1e-12).sum())
    return {
        "candidate_id": candidate_id(spec),
        "feature_bucket_col": spec.feature,
        "state_cols": ",".join(spec.state_cols),
        "same_source": spec.same_source,
        **score,
        "anchor_same_rows_mae": anchor["mae"],
        "official_same_rows_mae": official["mae"],
        "delta_vs_anchor": float(score["mae"] - anchor["mae"]),
        "delta_vs_official": float(score["mae"] - official["mae"]),
        "late_eval_n": int(late_score["n"]),
        "late_eval_first_date": str(late_score["first_date"]),
        "late_eval_last_date": str(late_score["last_date"]),
        "late_eval_mae": float(late_score["mae"]),
        "late_eval_rmse": float(late_score["rmse"]),
        "late_eval_anchor_mae": float(late_anchor["mae"]),
        "late_eval_delta_vs_anchor": float(late_score["mae"] - late_anchor["mae"]),
        "corrected_rows": corrected_rows,
        "mean_abs_correction_c": float(pd.to_numeric(predictions["residual_correction_c"], errors="coerce").abs().mean()),
    }


def run_interaction_screen(frame: pd.DataFrame, specs: list[InteractionSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for spec in specs:
        predictions = past_only_group_residual_predictions(frame, spec)
        score_rows.append(score_candidate(predictions, spec=spec))
        prediction_frames.append(predictions)
    scoreboard = pd.DataFrame(score_rows)
    if not scoreboard.empty:
        scoreboard = scoreboard.sort_values(["late_eval_mae", "mae", "rmse"]).reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return scoreboard, predictions


def interaction_atlas(frame: pd.DataFrame, bucket_catalog: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    state_cols = [
        col
        for col in (
            "meta_forecast_vs_prior7_bin",
            "meta_forecast_jump_sign",
            "meta_forecast_history_state",
            "meta_revision_range_state",
        )
        if col in frame.columns
    ]
    for bucket_row in bucket_catalog.head(12).itertuples(index=False):
        bucket_col = str(bucket_row.bucket_col)
        for state_col in state_cols:
            grouped = frame.dropna(subset=[bucket_col, state_col, "anchor_residual_c"]).groupby(
                [state_col, bucket_col],
                observed=True,
                dropna=False,
            )
            for (state_value, bucket_value), group in grouped:
                if len(group) < 35 or str(bucket_value) == "missing":
                    continue
                rows.append(
                    {
                        "feature": str(bucket_row.feature),
                        "bucket_col": bucket_col,
                        "state_col": state_col,
                        "state_value": str(state_value),
                        "feature_bucket": str(bucket_value),
                        "rows": int(len(group)),
                        "first_date": str(pd.to_datetime(group["target_date"]).min().date()),
                        "last_date": str(pd.to_datetime(group["target_date"]).max().date()),
                        "anchor_residual_mean_c": float(group["anchor_residual_c"].mean()),
                        "anchor_abs_error_mean_c": float(group["anchor_abs_error_c"].mean()),
                        "official_abs_error_mean_c": float(group["official_abs_error_c"].mean()),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    global_abs = float(frame["anchor_abs_error_c"].mean())
    out["abs_error_lift_vs_global_c"] = out["anchor_abs_error_mean_c"] - global_abs
    return out.sort_values(["abs_error_lift_vs_global_c", "rows"], ascending=[False, False]).reset_index(drop=True)


def baseline_comparison(scoreboard: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for system, column in (("0038_anchor", "anchor_0038_c"), ("official_raw", "official_raw")):
        score = score_prediction_frame(frame.rename(columns={column: "prediction"}), "prediction")
        late = frame[pd.to_datetime(frame["target_date"]) >= LATE_EVAL_START].copy()
        late_score = score_prediction_frame(late.rename(columns={column: "prediction"}), "prediction")
        rows.append(
            {
                "system": system,
                "candidate_id": column,
                **score,
                "late_eval_mae": late_score["mae"],
                "late_eval_rmse": late_score["rmse"],
            }
        )
    if not scoreboard.empty:
        best = scoreboard.iloc[0]
        rows.append(
            {
                "system": "0039_best_station_network_interaction",
                "candidate_id": str(best["candidate_id"]),
                "n": int(best["n"]),
                "first_date": str(best["first_date"]),
                "last_date": str(best["last_date"]),
                "mae": float(best["mae"]),
                "rmse": float(best["rmse"]),
                "bias": float(best["bias"]),
                "median_abs_error": float(best["median_abs_error"]),
                "late_eval_mae": float(best["late_eval_mae"]),
                "late_eval_rmse": float(best["late_eval_rmse"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["late_eval_mae", "mae"]).reset_index(drop=True)


def write_readme(
    *,
    folder: Path,
    manifest: dict[str, object],
    discovery_scan: pd.DataFrame,
    bucket_catalog: pd.DataFrame,
    atlas: pd.DataFrame,
    scoreboard: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    best = scoreboard.iloc[0] if not scoreboard.empty else None
    best_text = "No scoreable station-network interaction candidate was produced."
    if best is not None:
        best_text = (
            f"Best late-eval candidate: `{best['candidate_id']}` with late-window MAE "
            f"`{best['late_eval_mae']:.4f}` versus anchor `{best['late_eval_anchor_mae']:.4f}` "
            f"(delta `{best['late_eval_delta_vs_anchor']:.4f}`), and full walk-forward MAE "
            f"`{best['mae']:.4f}`."
        )
    readme = f"""# Station-Network Forecast Residual Interaction Mining

Generated: `{manifest['generated_at_utc']}`

## Purpose

`0038` showed that richer forecast-history labels alone do not beat the simpler 0037 same-source forecast-vs-prior trust rule. This run asks the next question: do long-history station and network attributes explain the remaining residuals of the current best forecast-history anchor?

The screen combines the 2000-2023 forecast archive rows with lagged/deployable station-network features derived from the long weather history. Feature discovery is restricted to rows through `{DISCOVERY_END.date()}`. Candidate scoring is then reported both on the full walk-forward window and on a configured late `{LATE_EVAL_START.date()}` to `2023-12-31` evaluation window. Because the current stable scored forecast archive is non-contiguous, the actual scored late rows run from `{manifest['late_eval_first_target_date']}` to `{manifest['late_eval_last_target_date']}`.

## Data Window

Rows used: `{manifest['official_rows']}` scored forecast rows.

Date range: `{manifest['first_target_date']}` to `{manifest['last_target_date']}`.

Late evaluation rows: `{manifest['late_eval_rows']}`.

Actual late evaluation date range: `{manifest['late_eval_first_target_date']}` to `{manifest['late_eval_last_target_date']}`.

Source counts: `{manifest['source_counts']}`.

## Leakage Contract

- All scored rows are earlier than `{CONFIRMATION_START.date()}`.
- Discovery feature ranking uses only rows with `target_date <= {DISCOVERY_END.date()}`.
- Late-era metrics are reported separately on `{LATE_EVAL_START.date()}` through `2023-12-31`.
- Feature buckets are computed as past-only terciles: each target date is bucketed using only earlier target dates.
- Residual corrections use only prior target dates in the same bucket/state cell; same-date rows from another source are excluded by date-level cumulative shifting.
- Same-source variants add source family to the interaction key.
- The anchor is the existing 0038/0037 trust selector; this run does not touch 2024+ confirmation rows.

## Main Result

{best_text}

## Baseline Comparison

{markdown_table(comparison, max_rows=10)}

## Selected Feature Catalog

{markdown_table(bucket_catalog.head(30), max_rows=30)}

## Discovery Feature Scan

{markdown_table(discovery_scan.head(40), max_rows=40)}

## Strongest Interaction Atlas Cells

{markdown_table(atlas.head(50), max_rows=50)}

## Candidate Scoreboard

{markdown_table(scoreboard.head(60), max_rows=60)}

## Interpretation

This is a high-value diagnostic even if it does not promote a new champion. If a station-network correction improves late 2018-2023 but not the full window, it is a signal that the station network contains regime-specific residual information but needs stronger stabilization before promotion. If it fails late-eval, the next step should not be more simple bucket interactions; it should move toward smoother local residual models or a refreshed continuous forecast archive so the forecast-state side is less sparse.
"""
    write_text(folder / "README.md", readme)


def update_master_index(manifest: dict[str, object]) -> None:
    index_path = RESEARCH_ROOT / "README.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# HKG Tmax Data Analysis\n"
    marker = "\n## Station-Network Forecast Residual Interaction Mining\n"
    base = existing.split(marker)[0].rstrip()
    text = f"""{base}
{marker}
Generated: `{manifest['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_station_network_forecast_residual_interaction_mining.py`:

- `{FOLDER_NAME}`: pre-2018 station/network feature discovery, forecast-state interaction atlas, and strict prior-only residual correction candidates around the current 0038 anchor.

| Metric | Value |
|---|---:|
| Official rows | {manifest['official_rows']} |
| Discovery features scanned | {manifest['discovery_features_scanned']} |
| Selected features | {manifest['selected_features']} |
| Interaction candidates | {manifest['interaction_candidates']} |
| Best full MAE | {manifest['best_full_mae']} |
| Best late-eval MAE | {manifest['best_late_eval_mae']} |
| Best late-eval delta vs anchor | {manifest['best_late_eval_delta_vs_anchor']} |

Leakage contract: selected features are ranked on data through `{DISCOVERY_END.date()}`; buckets and corrections are past-only by target date; all scored rows are before `{CONFIRMATION_START.date()}`.
"""
    write_text(index_path, text)


def update_milestones(manifest: dict[str, object]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Milestones\n"
    section_marker = "\n## Station-Network Forecast Residual Interaction Mining\n"
    blockers_marker = "\n## Current Blockers And Gaps\n"
    next_marker = "\n## Exact Next Recommended Codex Task\n"
    if section_marker in existing:
        base, rest = existing.split(section_marker, 1)
        suffix = f"{blockers_marker}{rest.split(blockers_marker, 1)[1]}" if blockers_marker in rest else ""
    elif blockers_marker in existing:
        base, rest = existing.split(blockers_marker, 1)
        suffix = f"{blockers_marker}{rest}"
    else:
        base = existing.rstrip()
        suffix = ""

    section = f"""{base.rstrip()}
{section_marker}
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_station_network_forecast_residual_interaction_mining.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Rows / candidates | Strongest current finding | Status |
|---|---:|---|---|
| Station/network discovery | `{manifest['discovery_features_scanned']}` pre-2018 features scanned; `{manifest['selected_features']}` selected | Top features and their residual correlations are in `discovery_scan.csv` and `selected_features.csv` | Pre-2018 selection only |
| Forecast-state interaction screen | `{manifest['interaction_candidates']}` candidates | Best late-eval candidate `{manifest['best_candidate']}`: actual late rows `{manifest['late_eval_first_target_date']}` to `{manifest['late_eval_last_target_date']}`, late MAE `{manifest['best_late_eval_mae']}`, delta vs anchor `{manifest['best_late_eval_delta_vs_anchor']}`, full MAE `{manifest['best_full_mae']}` | Audited |
| Leakage guards | past-only feature buckets and date-shifted residual cells | Zero 2024+ scored rows; same-date rows excluded; feature selection uses only rows through `{DISCOVERY_END.date()}` | Guarded |

Interpretation: `0039` tests whether long-history station/network attributes add residual signal after the current best forecast-history anchor. A late-eval improvement is not enough for final promotion by itself, but it identifies station-network interaction families worth turning into smoother local residual models. A late-eval loss means simple station-network bucket corrections are too brittle and the next lift should focus on smoother specialists or the continuous forecast archive refresh.
"""
    if suffix:
        if next_marker in suffix:
            before_next, _after_next = suffix.split(next_marker, 1)
            next_task = f"""{next_marker}

If the 0039 late-eval screen produced a robust positive delta, implement `0040_station_network_smooth_local_residual_models`: convert the strongest pre-2018 station/network interaction families into smooth distance-weighted local residual specialists with explicit 2018-2023 evaluation and zero 2024+ scoring. If it did not, prioritize snapshotting or finishing the active 2005-2026 forecast-detail backfill and rerun the forecast-anchor stack on the expanded continuous archive.
"""
            suffix = before_next.rstrip() + "\n\n" + next_task
        section += suffix
    write_text(path, section)


def write_outputs(
    *,
    frame: pd.DataFrame,
    derived_catalog: pd.DataFrame,
    discovery_scan: pd.DataFrame,
    full_scan: pd.DataFrame,
    bucket_catalog: pd.DataFrame,
    atlas: pd.DataFrame,
    scoreboard: pd.DataFrame,
    predictions: pd.DataFrame,
) -> dict[str, object]:
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    comparison = baseline_comparison(scoreboard, frame)

    write_csv(artifacts / "derived_features.csv", derived_catalog)
    write_csv(artifacts / "discovery_scan.csv", discovery_scan)
    write_csv(artifacts / "full_scan.csv", full_scan)
    write_csv(artifacts / "selected_features.csv", bucket_catalog)
    write_csv(artifacts / "interaction_atlas.csv", atlas)
    write_csv(artifacts / "candidate_scoreboard.csv", scoreboard)
    write_csv(artifacts / "candidate_predictions.csv", predictions)
    write_csv(artifacts / "baseline_comparison.csv", comparison)
    top_ids = scoreboard.head(8)["candidate_id"].to_list() if not scoreboard.empty else []
    write_csv(
        artifacts / "top_candidate_predictions.csv",
        predictions[predictions["candidate_id"].isin(top_ids)].copy() if top_ids else predictions.head(0),
    )

    best = scoreboard.iloc[0] if not scoreboard.empty else None
    late_mask = pd.to_datetime(frame["target_date"]) >= LATE_EVAL_START
    late_frame = frame[late_mask].copy()
    late_rows = int(late_mask.sum())
    late_first = "" if late_frame.empty else str(pd.to_datetime(late_frame["target_date"]).min().date())
    late_last = "" if late_frame.empty else str(pd.to_datetime(late_frame["target_date"]).max().date())
    manifest = {
        "generated_at_utc": now_utc(),
        "folder": FOLDER_NAME,
        "official_rows": int(len(frame)),
        "first_target_date": str(frame["target_date"].min().date()),
        "last_target_date": str(frame["target_date"].max().date()),
        "late_eval_rows": late_rows,
        "late_eval_first_target_date": late_first,
        "late_eval_last_target_date": late_last,
        "source_counts": {str(k): int(v) for k, v in frame["forecast_source_family"].value_counts().to_dict().items()},
        "discovery_end": str(DISCOVERY_END.date()),
        "late_eval_start": str(LATE_EVAL_START.date()),
        "discovery_features_scanned": int(len(discovery_scan)),
        "selected_features": int(len(bucket_catalog)),
        "interaction_candidates": int(len(scoreboard)),
        "best_candidate": "" if best is None else str(best["candidate_id"]),
        "best_full_mae": None if best is None else float(best["mae"]),
        "best_full_delta_vs_anchor": None if best is None else float(best["delta_vs_anchor"]),
        "best_late_eval_mae": None if best is None else float(best["late_eval_mae"]),
        "best_late_eval_delta_vs_anchor": None if best is None else float(best["late_eval_delta_vs_anchor"]),
        "anchor_full_mae": float(comparison.loc[comparison["system"].eq("0038_anchor"), "mae"].iloc[0]),
        "anchor_late_eval_mae": float(comparison.loc[comparison["system"].eq("0038_anchor"), "late_eval_mae"].iloc[0]),
        "confirmation_start": str(CONFIRMATION_START.date()),
    }
    write_json(RESEARCH_ROOT / "station_network_residuals_manifest.json", manifest)
    write_readme(
        folder=folder,
        manifest=manifest,
        discovery_scan=discovery_scan,
        bucket_catalog=bucket_catalog,
        atlas=atlas,
        scoreboard=scoreboard,
        comparison=comparison,
    )
    update_master_index(manifest)
    update_milestones(manifest)
    return manifest


def run() -> dict[str, object]:
    frame, derived_catalog = build_analysis_frame()
    require_no_confirmation_dates(frame["target_date"], context="0039 analysis frame")
    discovery_scan = scan_station_network_information(frame, discovery_only=True)
    full_scan = scan_station_network_information(frame, discovery_only=False)
    selected = selected_feature_catalog(discovery_scan)
    frame, bucket_catalog = add_selected_feature_buckets(frame, selected)
    specs = build_interaction_specs(frame, bucket_catalog)
    atlas = interaction_atlas(frame, bucket_catalog)
    scoreboard, predictions = run_interaction_screen(frame, specs)
    require_no_confirmation_dates(predictions["target_date"], context="0039 candidate predictions")
    return write_outputs(
        frame=frame,
        derived_catalog=derived_catalog,
        discovery_scan=discovery_scan,
        full_scan=full_scan,
        bucket_catalog=bucket_catalog,
        atlas=atlas,
        scoreboard=scoreboard,
        predictions=predictions,
    )


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run HKG T24 station-network forecast residual interaction mining.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
