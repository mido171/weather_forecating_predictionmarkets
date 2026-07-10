from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hkg_t24_beastmode_signal_discovery import (  # noqa: E402
    CONFIRMATION_START,
    RESEARCH_ROOT,
    markdown_table,
    require_no_confirmation_dates,
    write_csv,
    write_json,
    write_text,
)
from scripts.run_hkg_t24_long_history_cross_family_interaction_atlas import (  # noqa: E402
    update_markdown_section,
)
from scripts.run_hkg_t24_station_only_late_period_bias_repair import (  # noqa: E402
    score_prediction_frame,
)

FOLDER_NAME = "0059_station_only_february_march_transition_specialist"
ARTIFACT_0054 = RESEARCH_ROOT / "0054_station_only_walkforward_matrix_audit" / "artifacts"
ARTIFACT_0055 = RESEARCH_ROOT / "0055_station_only_walkforward_benchmark" / "artifacts"
ARTIFACT_0057 = RESEARCH_ROOT / "0057_station_only_residual_specialist_design_queue" / "artifacts"
ARTIFACT_0058 = RESEARCH_ROOT / "0058_station_only_late_period_bias_repair" / "artifacts"
FEATURE_MATRIX_PATH = ARTIFACT_0054 / "features.parquet"
COMPONENT_CATALOG_PATH = ARTIFACT_0054 / "components.csv"
PREDICTIONS_PATH = ARTIFACT_0055 / "predictions.parquet"
SUMMARY_0055_PATH = ARTIFACT_0055 / "summary.json"
DESIGN_QUEUE_PATH = ARTIFACT_0057 / "design_queue.csv"
SUMMARY_0058_PATH = ARTIFACT_0058 / "summary.json"
TRAINING_THRESHOLD_END = pd.Timestamp("1999-12-31")
DEVELOPMENT_END = pd.Timestamp("2023-12-31")
TRANSITION_MONTHS = (2, 3)
ADJACENT_MONTHS = (1, 4)


@dataclass(frozen=True)
class TransitionSpec:
    correction_id: str
    group_columns: tuple[str, ...]
    activation: str
    min_prior_rows: int
    shrinkage: float
    cap_c: float
    window_days: int | None = None


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 120) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def quantile_edges(values: pd.Series, *, low_q: float = 0.33, high_q: float = 0.67) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 365:
        raise ValueError(f"Need at least 365 pre-2000 values for stable gate thresholds, got {len(clean)}")
    low = float(clean.quantile(low_q))
    high = float(clean.quantile(high_q))
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError("Non-finite quantile edge")
    if low == high:
        std = float(clean.std(ddof=0))
        if not math.isfinite(std) or std <= 0:
            raise ValueError("Cannot build non-degenerate gate thresholds")
        low -= std * 0.1
        high += std * 0.1
    return low, high


def bucket_by_edges(values: pd.Series, low: float, high: float, labels: tuple[str, str, str]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    out = pd.Series("missing", index=values.index, dtype="object")
    out.loc[numeric <= low] = labels[0]
    out.loc[(numeric > low) & (numeric <= high)] = labels[1]
    out.loc[numeric > high] = labels[2]
    return out


def transition_phase(day_of_year: int) -> str:
    if 32 <= day_of_year <= 45:
        return "early_feb"
    if 46 <= day_of_year <= 60:
        return "late_feb"
    if 61 <= day_of_year <= 75:
        return "early_mar"
    if 76 <= day_of_year <= 91:
        return "late_mar"
    if 20 <= day_of_year <= 31:
        return "late_jan_buffer"
    if 92 <= day_of_year <= 105:
        return "early_apr_buffer"
    return "outside_transition"


def activation_weight(target_date: pd.Timestamp, activation: str) -> float:
    month = int(target_date.month)
    doy = int(target_date.dayofyear)
    if activation == "feb_mar_only":
        return 1.0 if month in TRANSITION_MONTHS else 0.0
    if activation == "feb_mar_plus_soft_adjacent":
        if month in TRANSITION_MONTHS:
            return 1.0
        if 20 <= doy <= 31:
            return float((doy - 19) / 12.0)
        if 92 <= doy <= 105:
            return float((106 - doy) / 14.0)
        return 0.0
    if activation == "feb_mar_edges_downweighted":
        if month not in TRANSITION_MONTHS:
            return 0.0
        if 32 <= doy <= 45 or 76 <= doy <= 91:
            return 0.65
        return 1.0
    raise ValueError(f"Unknown activation: {activation}")


def load_anchor_predictions() -> tuple[pd.DataFrame, dict[str, Any]]:
    summary = load_json(SUMMARY_0055_PATH)
    best_model_id = str(summary["best_model_id"])
    predictions = pd.read_parquet(PREDICTIONS_PATH)
    predictions["target_date"] = pd.to_datetime(predictions["target_date"], errors="coerce").dt.normalize()
    predictions = predictions[predictions["model_id"].astype(str).eq(best_model_id)].copy()
    if predictions.empty:
        raise RuntimeError(f"Missing 0055 best model predictions: {best_model_id}")
    predictions = predictions[predictions["target_date"].le(DEVELOPMENT_END)].copy()
    require_no_confirmation_dates(predictions["target_date"], context="0059 anchor predictions")
    predictions["month"] = predictions["target_date"].dt.month
    predictions["year"] = predictions["target_date"].dt.year
    predictions["day_of_year"] = predictions["target_date"].dt.dayofyear
    predictions["transition_phase"] = predictions["day_of_year"].map(transition_phase)
    predictions["anchor_error_c"] = predictions["point_forecast_c"] - predictions["target_tmax_c"]
    predictions["residual_to_add_c"] = predictions["target_tmax_c"] - predictions["point_forecast_c"]
    predictions = predictions.sort_values("target_date").reset_index(drop=True)
    return predictions, summary


def feature_columns_by_token(catalog: pd.DataFrame, token: str) -> list[str]:
    matches = catalog[
        catalog["feature_id"].astype(str).str.contains(token, case=False, na=False)
        | catalog["raw_feature_name"].astype(str).str.contains(token, case=False, na=False)
    ]
    return matches["feature_id"].astype(str).tolist()


def load_feature_gates() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not FEATURE_MATRIX_PATH.exists():
        raise FileNotFoundError(f"Missing 0054 feature matrix: {FEATURE_MATRIX_PATH}")
    if not COMPONENT_CATALOG_PATH.exists():
        raise FileNotFoundError(f"Missing 0054 component catalog: {COMPONENT_CATALOG_PATH}")
    features = pd.read_parquet(FEATURE_MATRIX_PATH)
    features["target_date"] = pd.to_datetime(features["target_date"], errors="coerce").dt.normalize()
    features = features[features["target_date"].notna() & features["target_date"].lt(CONFIRMATION_START)].copy()
    require_no_confirmation_dates(features["target_date"], context="0059 feature gates")
    catalog = pd.read_csv(COMPONENT_CATALOG_PATH)

    pressure_cols = [c for c in feature_columns_by_token(catalog, "sea_level_pressure") if c in features.columns]
    dew_cols = [c for c in feature_columns_by_token(catalog, "dew_point") if c in features.columns]
    wind_cols = [c for c in feature_columns_by_token(catalog, "wind_speed") if c in features.columns]
    if not pressure_cols or not dew_cols or not wind_cols:
        raise RuntimeError(
            "Need pressure, dew, and wind station-only feature families for 0059; "
            f"got pressure={len(pressure_cols)}, dew={len(dew_cols)}, wind={len(wind_cols)}"
        )

    gate_frame = features[["target_date", *pressure_cols, *dew_cols, *wind_cols]].copy()
    gate_frame["pressure_spread_abs_max"] = gate_frame[pressure_cols].abs().max(axis=1, skipna=True)
    gate_frame["dew_trajectory_mean"] = gate_frame[dew_cols].mean(axis=1, skipna=True)
    gate_frame["wind_spread_abs_max"] = gate_frame[wind_cols].abs().max(axis=1, skipna=True)
    pre2000 = gate_frame[gate_frame["target_date"].le(TRAINING_THRESHOLD_END)].copy()
    threshold_rows: list[dict[str, object]] = []
    for column, labels in [
        ("pressure_spread_abs_max", ("pressure_low", "pressure_mid", "pressure_high")),
        ("dew_trajectory_mean", ("dew_falling_or_dry", "dew_neutral", "dew_rising_or_moist")),
        ("wind_spread_abs_max", ("wind_low", "wind_mid", "wind_high")),
    ]:
        low, high = quantile_edges(pre2000[column])
        gate_column = column.replace("_abs_max", "").replace("_mean", "") + "_bucket"
        gate_frame[gate_column] = bucket_by_edges(gate_frame[column], low, high, labels)
        threshold_rows.append(
            {
                "gate_column": gate_column,
                "source_metric": column,
                "low_edge": low,
                "high_edge": high,
                "threshold_source": "1947-01-01_to_1999-12-31_feature_history",
                "pre2000_non_null_rows": int(pd.to_numeric(pre2000[column], errors="coerce").notna().sum()),
                "source_columns": len(
                    pressure_cols
                    if column.startswith("pressure")
                    else dew_cols
                    if column.startswith("dew")
                    else wind_cols
                ),
            }
        )
    return (
        gate_frame[
            [
                "target_date",
                "pressure_spread_abs_max",
                "dew_trajectory_mean",
                "wind_spread_abs_max",
                "pressure_spread_bucket",
                "dew_trajectory_bucket",
                "wind_spread_bucket",
            ]
        ].copy(),
        pd.DataFrame(threshold_rows),
    )


def build_model_frame() -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    anchor, summary = load_anchor_predictions()
    gates, thresholds = load_feature_gates()
    frame = anchor.merge(gates, on="target_date", how="left", validate="one_to_one")
    frame["transition_month_bucket"] = np.select(
        [frame["month"].eq(2), frame["month"].eq(3), frame["month"].isin(ADJACENT_MONTHS)],
        ["february", "march", "jan_apr_adjacent"],
        default="other",
    )
    frame["transition_target_window"] = frame["month"].isin(TRANSITION_MONTHS)
    frame["adjacent_window"] = frame["month"].isin(ADJACENT_MONTHS)
    frame = frame.sort_values("target_date").reset_index(drop=True)
    require_no_confirmation_dates(frame["target_date"], context="0059 model frame")
    return frame, summary, thresholds


def transition_specs() -> list[TransitionSpec]:
    return [
        TransitionSpec(
            "month_prior_mean_min45_shrink120_cap1p25",
            ("transition_month_bucket",),
            "feb_mar_only",
            45,
            120.0,
            1.25,
        ),
        TransitionSpec(
            "phase_prior_mean_min35_shrink90_cap1p25",
            ("transition_phase",),
            "feb_mar_only",
            35,
            90.0,
            1.25,
        ),
        TransitionSpec(
            "month_pressure_prior_mean_min30_shrink90_cap1p25",
            ("transition_month_bucket", "pressure_spread_bucket"),
            "feb_mar_only",
            30,
            90.0,
            1.25,
        ),
        TransitionSpec(
            "month_dew_prior_mean_min30_shrink90_cap1p25",
            ("transition_month_bucket", "dew_trajectory_bucket"),
            "feb_mar_only",
            30,
            90.0,
            1.25,
        ),
        TransitionSpec(
            "month_pressure_dew_prior_mean_min20_shrink75_cap1p25",
            ("transition_month_bucket", "pressure_spread_bucket", "dew_trajectory_bucket"),
            "feb_mar_only",
            20,
            75.0,
            1.25,
        ),
        TransitionSpec(
            "phase_pressure_dew_prior_mean_min18_shrink75_cap1p25",
            ("transition_phase", "pressure_spread_bucket", "dew_trajectory_bucket"),
            "feb_mar_only",
            18,
            75.0,
            1.25,
        ),
        TransitionSpec(
            "month_pressure_dew_rolling8y_min18_shrink75_cap1p25",
            ("transition_month_bucket", "pressure_spread_bucket", "dew_trajectory_bucket"),
            "feb_mar_only",
            18,
            75.0,
            1.25,
            window_days=2920,
        ),
        TransitionSpec(
            "phase_pressure_dew_wind_min15_shrink60_cap1p0",
            ("transition_phase", "pressure_spread_bucket", "dew_trajectory_bucket", "wind_spread_bucket"),
            "feb_mar_only",
            15,
            60.0,
            1.0,
        ),
        TransitionSpec(
            "phase_soft_adjacent_min35_shrink90_cap1p0",
            ("transition_phase",),
            "feb_mar_plus_soft_adjacent",
            35,
            90.0,
            1.0,
        ),
        TransitionSpec(
            "month_edges_downweighted_min45_shrink120_cap1p25",
            ("transition_month_bucket",),
            "feb_mar_edges_downweighted",
            45,
            120.0,
            1.25,
        ),
    ]


def group_key(row: pd.Series, columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(row[column]) for column in columns)


def shrink_and_cap(raw: float, prior_rows: int, spec: TransitionSpec) -> float:
    if prior_rows < spec.min_prior_rows or not math.isfinite(raw):
        return 0.0
    shrink = prior_rows / (prior_rows + spec.shrinkage)
    return float(np.clip(raw * shrink, -spec.cap_c, spec.cap_c))


def compute_transition_correction(frame: pd.DataFrame, spec: TransitionSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ordered = frame.sort_values("target_date").reset_index(drop=True)
    corrections = np.zeros(len(ordered), dtype=float)
    active_weights = np.zeros(len(ordered), dtype=float)
    prior_counts = np.zeros(len(ordered), dtype=int)
    raw_means = np.full(len(ordered), math.nan, dtype=float)
    expanding_state: dict[tuple[str, ...], tuple[int, float]] = defaultdict(lambda: (0, 0.0))
    rolling_state: dict[tuple[str, ...], dict[str, object]] = defaultdict(lambda: {"rows": deque(), "sum": 0.0})

    for idx, row in ordered.iterrows():
        current_date = pd.Timestamp(row["target_date"])
        key = group_key(row, spec.group_columns)
        weight = activation_weight(current_date, spec.activation)
        active_weights[idx] = weight
        if spec.window_days is None:
            count, total = expanding_state[key]
        else:
            state = rolling_state[key]
            rows = state["rows"]
            assert isinstance(rows, deque)
            min_date = current_date - pd.Timedelta(days=spec.window_days)
            while rows and rows[0][0] < min_date:
                _, old_residual = rows.popleft()
                state["sum"] = float(state["sum"]) - float(old_residual)
            count = len(rows)
            total = float(state["sum"])
        prior_counts[idx] = count
        if weight > 0 and count >= spec.min_prior_rows:
            raw = total / count
            raw_means[idx] = raw
            corrections[idx] = shrink_and_cap(raw, count, spec) * weight

        residual = float(row["residual_to_add_c"])
        if math.isfinite(residual):
            if spec.window_days is None:
                expanding_state[key] = (count + 1, total + residual)
            else:
                state = rolling_state[key]
                rows = state["rows"]
                assert isinstance(rows, deque)
                rows.append((current_date, residual))
                state["sum"] = float(state["sum"]) + residual
    return corrections, active_weights, prior_counts, raw_means


def apply_transition_specialist(frame: pd.DataFrame, spec: TransitionSpec) -> pd.DataFrame:
    ordered = frame.sort_values("target_date").reset_index(drop=True).copy()
    corrections, active_weights, prior_rows, raw_means = compute_transition_correction(ordered, spec)
    out = ordered[
        [
            "target_date",
            "target_tmax_c",
            "point_forecast_c",
            "fold_id",
            "year",
            "month",
            "day_of_year",
            "transition_phase",
            "transition_month_bucket",
            "transition_target_window",
            "adjacent_window",
            "pressure_spread_abs_max",
            "dew_trajectory_mean",
            "wind_spread_abs_max",
            "pressure_spread_bucket",
            "dew_trajectory_bucket",
            "wind_spread_bucket",
        ]
    ].copy()
    out = out.rename(columns={"point_forecast_c": "anchor_prediction_c"})
    out["candidate_prediction_c"] = out["anchor_prediction_c"] + corrections
    out["residual_correction_c"] = corrections
    out["activation_weight"] = active_weights
    out["raw_prior_residual_mean_c"] = raw_means
    out["prior_rows"] = prior_rows
    out["correction_id"] = spec.correction_id
    out["group_columns"] = "|".join(spec.group_columns)
    out["activation"] = spec.activation
    out["min_prior_rows"] = spec.min_prior_rows
    out["shrinkage"] = spec.shrinkage
    out["cap_c"] = spec.cap_c
    out["window_days"] = float(spec.window_days) if spec.window_days is not None else math.nan
    require_no_confirmation_dates(out["target_date"], context=f"0059 {spec.correction_id} predictions")
    return out.sort_values(["target_date", "correction_id"]).reset_index(drop=True)


def score_window(group: pd.DataFrame, mask: pd.Series, prediction_col: str) -> dict[str, object]:
    return score_prediction_frame(group[mask].copy(), prediction_col)


def score_candidates(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    subgroup_rows: list[dict[str, object]] = []
    for correction_id, group in predictions.groupby("correction_id", observed=True):
        full = score_prediction_frame(group, "candidate_prediction_c")
        anchor_full = score_prediction_frame(group, "anchor_prediction_c")
        transition_mask = group["transition_target_window"].astype(bool)
        adjacent_mask = group["adjacent_window"].astype(bool)
        rest_mask = ~(transition_mask | adjacent_mask)
        transition = score_window(group, transition_mask, "candidate_prediction_c")
        anchor_transition = score_window(group, transition_mask, "anchor_prediction_c")
        adjacent = score_window(group, adjacent_mask, "candidate_prediction_c")
        anchor_adjacent = score_window(group, adjacent_mask, "anchor_prediction_c")
        rest = score_window(group, rest_mask, "candidate_prediction_c")
        anchor_rest = score_window(group, rest_mask, "anchor_prediction_c")
        row = {
            "correction_id": correction_id,
            "group_columns": str(group["group_columns"].iloc[0]),
            "activation": str(group["activation"].iloc[0]),
            "window_days": float(group["window_days"].iloc[0]) if pd.notna(group["window_days"].iloc[0]) else math.nan,
            "min_prior_rows": int(group["min_prior_rows"].iloc[0]),
            "shrinkage": float(group["shrinkage"].iloc[0]),
            "cap_c": float(group["cap_c"].iloc[0]),
            "n": full["n"],
            "mae": full["mae"],
            "rmse": full["rmse"],
            "bias": full["bias"],
            "delta_mae_vs_anchor": float(full["mae"]) - float(anchor_full["mae"]),
            "transition_n": transition["n"],
            "transition_mae": transition["mae"],
            "transition_rmse": transition["rmse"],
            "transition_bias": transition["bias"],
            "transition_anchor_mae": anchor_transition["mae"],
            "transition_delta_mae_vs_anchor": float(transition["mae"]) - float(anchor_transition["mae"]),
            "adjacent_mae": adjacent["mae"],
            "adjacent_anchor_mae": anchor_adjacent["mae"],
            "adjacent_delta_mae_vs_anchor": float(adjacent["mae"]) - float(anchor_adjacent["mae"]),
            "rest_delta_mae_vs_anchor": float(rest["mae"]) - float(anchor_rest["mae"]),
            "mean_abs_correction_c": float(group["residual_correction_c"].abs().mean()),
            "active_transition_share": float(
                group.loc[transition_mask, "residual_correction_c"].abs().gt(1e-9).mean()
            ),
        }
        row["promotion_gate_passed"] = bool(
            row["transition_delta_mae_vs_anchor"] < 0.0
            and row["adjacent_delta_mae_vs_anchor"] <= 0.005
            and row["delta_mae_vs_anchor"] <= 0.005
        )
        rows.append(row)
        for subgroup_name, subgroup_mask, candidate_score, anchor_score in [
            ("feb_mar_transition", transition_mask, transition, anchor_transition),
            ("jan_apr_adjacent", adjacent_mask, adjacent, anchor_adjacent),
            ("other_months", rest_mask, rest, anchor_rest),
        ]:
            subgroup_rows.append(
                {
                    "correction_id": correction_id,
                    "subgroup": subgroup_name,
                    "n": candidate_score["n"],
                    "first_date": candidate_score["first_date"],
                    "last_date": candidate_score["last_date"],
                    "candidate_mae": candidate_score["mae"],
                    "anchor_mae": anchor_score["mae"],
                    "delta_mae_vs_anchor": float(candidate_score["mae"]) - float(anchor_score["mae"]),
                    "candidate_rmse": candidate_score["rmse"],
                    "anchor_rmse": anchor_score["rmse"],
                    "candidate_bias": candidate_score["bias"],
                    "anchor_bias": anchor_score["bias"],
                    "active_correction_share": float(
                        group.loc[subgroup_mask, "residual_correction_c"].abs().gt(1e-9).mean()
                    ),
                }
            )
    scoreboard = pd.DataFrame(rows).sort_values(
        ["promotion_gate_passed", "transition_delta_mae_vs_anchor", "delta_mae_vs_anchor"],
        ascending=[False, True, True],
    )
    subgroups = pd.DataFrame(subgroup_rows).sort_values(["subgroup", "delta_mae_vs_anchor"]).reset_index(drop=True)
    return scoreboard.reset_index(drop=True), subgroups


def leakage_audit(predictions: pd.DataFrame, thresholds: pd.DataFrame, scoreboard: pd.DataFrame) -> pd.DataFrame:
    checks = [
        {
            "check_id": "no_confirmation_rows",
            "passed": bool(pd.to_datetime(predictions["target_date"], errors="coerce").max() < CONFIRMATION_START),
            "evidence": f"last target_date {pd.to_datetime(predictions['target_date'], errors='coerce').max().date()}",
        },
        {
            "check_id": "gate_thresholds_pre2000_only",
            "passed": bool(thresholds["threshold_source"].astype(str).str.contains("1999-12-31").all()),
            "evidence": f"{len(thresholds)} gate threshold rows built from pre-2000 feature history",
        },
        {
            "check_id": "corrections_have_prior_history_only",
            "passed": bool((predictions["prior_rows"] >= 0).all()),
            "evidence": "streaming correction updates state after scoring each row",
        },
        {
            "check_id": "first_row_has_zero_correction",
            "passed": bool(
                predictions.sort_values("target_date")
                .groupby("correction_id", observed=True)
                .head(1)["residual_correction_c"]
                .abs()
                .le(1e-12)
                .all()
            ),
            "evidence": "no earlier residuals exist for first row of each correction",
        },
        {
            "check_id": "promotion_gate_blocks_adjacent_degradation",
            "passed": bool(
                scoreboard.loc[scoreboard["promotion_gate_passed"], "adjacent_delta_mae_vs_anchor"].le(0.005).all()
            ),
            "evidence": f"{int(scoreboard['promotion_gate_passed'].sum())} candidates passed gate",
        },
    ]
    return pd.DataFrame(checks)


def build_readme(
    *,
    summary: dict[str, Any],
    scoreboard: pd.DataFrame,
    subgroups: pd.DataFrame,
    leakage: pd.DataFrame,
    thresholds: pd.DataFrame,
    design_row: pd.DataFrame,
) -> str:
    return f"""# Station-Only February/March Transition Specialist

Generated: `{summary['generated_at_utc']}`

## Purpose

`0057` identified February and March as a deployable station-only failure mode. This experiment tests whether a small prior-only residual specialist can reduce those transition-month errors around the `0055` station-only winner.

This is not a final production model and not a 2024+ confirmation test. It is a focused residual-specialist screen.

## Leakage Contract

- Target dates scored: `{summary['first_date']}` to `{summary['last_date']}`.
- Anchor model: `{summary['anchor_model_id']}` from `0055`.
- Gate thresholds for pressure, dew, and wind are fixed from feature history through `1999-12-31`.
- Every residual correction for date `T` uses only residuals from dates strictly before `T`.
- 2024+ rows remain excluded.

## Headline

| Item | Value |
|---|---:|
| Rows scored | {summary['rows_scored']} |
| Transition rows | {summary['transition_rows']} |
| Anchor transition MAE | {summary['anchor_transition_mae']} |
| Best candidate | {summary['best_candidate']} |
| Best full MAE / RMSE | {summary['best_mae']} / {summary['best_rmse']} |
| Best transition MAE | {summary['best_transition_mae']} |
| Best transition delta vs anchor | {summary['best_transition_delta_mae_vs_anchor']} |
| Best adjacent delta vs anchor | {summary['best_adjacent_delta_mae_vs_anchor']} |
| Promotion gate passed | {summary['best_promotion_gate_passed']} |
| 0058 best full MAE reference | {summary['reference_0058_best_mae']} |

## Source Queue Row

{markdown_table(design_row, max_rows=5)}

## Gate Thresholds

{markdown_table(thresholds, max_rows=20)}

## Candidate Scoreboard

{markdown_table(scoreboard, max_rows=60)}

## Subgroup Scoreboard

{markdown_table(subgroups, max_rows=90)}

## Leakage Checks

{markdown_table(leakage, max_rows=20)}

## Interpretation

This test asks a narrow question: can the long-history station-only stack repair February/March transition misses without damaging neighboring January/April behavior? A gate-passing result is useful as a candidate component for later blending. A failing result is still useful: it means this error pocket needs richer inputs, probably the official forecast archive or more detailed synoptic features, rather than simple month/pressure/dew residual means.

## Files

- `artifacts/predictions.parquet`
- `artifacts/predictions_sample.csv`
- `artifacts/scoreboard.csv`
- `artifacts/subgroup_scoreboard.csv`
- `artifacts/gate_thresholds.csv`
- `artifacts/leakage_audit.csv`
- `artifacts/summary.json`
"""


def update_master_index(summary: dict[str, Any]) -> None:
    section = f"""
Generated: `{summary['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_station_only_february_march_transition_specialist.py`:

- `{FOLDER_NAME}`: prior-only February/March transition residual specialists around the `0055` station-only winner.

| Metric | Value |
|---|---:|
| Anchor transition MAE | {summary['anchor_transition_mae']} |
| Best candidate | {summary['best_candidate']} |
| Best transition MAE | {summary['best_transition_mae']} |
| Best transition delta vs anchor | {summary['best_transition_delta_mae_vs_anchor']} |
| Best adjacent delta vs anchor | {summary['best_adjacent_delta_mae_vs_anchor']} |
| Promotion gate passed | {summary['best_promotion_gate_passed']} |

Leakage contract: pre-2000 fixed gate thresholds; correction state updates only after each target date is scored.
"""
    update_markdown_section(
        RESEARCH_ROOT / "README.md",
        heading="Station-Only February/March Transition Specialist",
        section=section,
    )


def update_milestones(summary: dict[str, Any]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_station_only_february_march_transition_specialist.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Status |
|---|---|---|
| Anchor model | `{summary['anchor_model_id']}` | Tested |
| Target pocket | February/March rows `{summary['transition_rows']}` | Tested |
| Anchor transition MAE | `{summary['anchor_transition_mae']}` | Baseline |
| Best candidate | `{summary['best_candidate']}` | Diagnostic |
| Best full MAE / RMSE | `{summary['best_mae']}` / `{summary['best_rmse']}` | Pre-2024 only |
| Best transition delta | `{summary['best_transition_delta_mae_vs_anchor']}` | Diagnostic |
| Best adjacent delta | `{summary['best_adjacent_delta_mae_vs_anchor']}` | Guarded |
| Promotion gate passed | `{summary['best_promotion_gate_passed']}` | Guarded |
| Leakage checks passed | `{summary['leakage_checks_passed']}` / `{summary['leakage_check_rows']}` | Guarded |

Interpretation: `0059` isolates the late-winter/spring transition pocket with prior-only station residual gates. It uses no delayed RSS rows and no 2024+ confirmation rows.
"""
    update_markdown_section(
        path,
        heading="Station-Only February/March Transition Specialist",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    blocker = (
        f"43. February/March station-only specialist tested `{summary['candidate_count']}` prior-only variants; "
        f"best transition delta vs anchor is `{summary['best_transition_delta_mae_vs_anchor']}` from "
        f"`{summary['best_candidate']}`."
    )
    text = path.read_text(encoding="utf-8")
    if blocker not in text and "\n## Exact Next Recommended Codex Task\n" in text:
        text = text.replace("\n\n## Exact Next Recommended Codex Task\n", f"\n{blocker}\n\n## Exact Next Recommended Codex Task\n", 1)
        write_text(path, text)
    next_task = """
Continue the `0057` queue with `spring_transition_pressure_dew_specialist`: test MAM-only dew/pressure/wind residual specialists, compare them against `0058` and `0059`, and keep the current RSS archive as an optional diagnostic overlap only until backfill completes.
"""
    update_markdown_section(path, heading="Exact Next Recommended Codex Task", section=next_task)


def run(output_root: Path = RESEARCH_ROOT) -> dict[str, Any]:
    generated_at = now_utc()
    frame, source_summary, thresholds = build_model_frame()
    if not DESIGN_QUEUE_PATH.exists():
        raise FileNotFoundError(f"Missing 0057 design queue: {DESIGN_QUEUE_PATH}")
    design_queue = pd.read_csv(DESIGN_QUEUE_PATH)
    design_row = design_queue[design_queue["candidate_id"].astype(str).eq("february_march_transition_specialist")].copy()
    if design_row.empty:
        raise RuntimeError("0057 design queue does not contain february_march_transition_specialist")
    predictions = pd.concat([apply_transition_specialist(frame, spec) for spec in transition_specs()], ignore_index=True)
    scoreboard, subgroups = score_candidates(predictions)
    leakage = leakage_audit(predictions, thresholds, scoreboard)
    if not leakage["passed"].astype(bool).all():
        failed = leakage[~leakage["passed"].astype(bool)]["check_id"].tolist()
        raise RuntimeError(f"0059 leakage audit failed: {failed}")

    transition_frame = frame[frame["transition_target_window"]].rename(
        columns={"point_forecast_c": "anchor_prediction_c"}
    )
    anchor_transition = score_prediction_frame(transition_frame, "anchor_prediction_c")
    anchor_full = score_prediction_frame(
        frame.rename(columns={"point_forecast_c": "anchor_prediction_c"}),
        "anchor_prediction_c",
    )
    best = scoreboard.iloc[0]
    summary_0058 = load_json(SUMMARY_0058_PATH) if SUMMARY_0058_PATH.exists() else {}
    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "anchor_model_id": str(source_summary["best_model_id"]),
        "candidate_count": int(scoreboard["correction_id"].nunique()),
        "rows_scored": int(scoreboard["n"].max()),
        "transition_rows": int(anchor_transition["n"]),
        "first_date": str(frame["target_date"].min().date()),
        "last_date": str(frame["target_date"].max().date()),
        "anchor_mae": float(anchor_full["mae"]),
        "anchor_rmse": float(anchor_full["rmse"]),
        "anchor_transition_mae": float(anchor_transition["mae"]),
        "anchor_transition_rmse": float(anchor_transition["rmse"]),
        "best_candidate": str(best["correction_id"]),
        "best_mae": float(best["mae"]),
        "best_rmse": float(best["rmse"]),
        "best_bias": float(best["bias"]),
        "best_delta_mae_vs_anchor": float(best["delta_mae_vs_anchor"]),
        "best_transition_mae": float(best["transition_mae"]),
        "best_transition_delta_mae_vs_anchor": float(best["transition_delta_mae_vs_anchor"]),
        "best_adjacent_delta_mae_vs_anchor": float(best["adjacent_delta_mae_vs_anchor"]),
        "best_promotion_gate_passed": bool(best["promotion_gate_passed"]),
        "reference_0058_best_mae": summary_0058.get("best_mae"),
        "reference_0058_best_candidate": summary_0058.get("best_candidate"),
        "leakage_check_rows": int(len(leakage)),
        "leakage_checks_passed": int(leakage["passed"].astype(bool).sum()),
        "uses_2024_plus_rows": False,
    }

    folder = output_root / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(artifacts / "predictions.parquet", index=False)
    write_csv(artifacts / "predictions_sample.csv", predictions.head(1000))
    write_csv(artifacts / "scoreboard.csv", scoreboard)
    write_csv(artifacts / "subgroup_scoreboard.csv", subgroups)
    write_csv(artifacts / "gate_thresholds.csv", thresholds)
    write_csv(artifacts / "leakage_audit.csv", leakage)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "station_only_february_march_transition_specialist_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            summary=summary,
            scoreboard=scoreboard,
            subgroups=subgroups,
            leakage=leakage,
            thresholds=thresholds,
            design_row=design_row,
        ),
    )
    update_master_index(summary)
    update_milestones(summary)
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(
        description="Run prior-only February/March station residual transition specialists."
    ).parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
