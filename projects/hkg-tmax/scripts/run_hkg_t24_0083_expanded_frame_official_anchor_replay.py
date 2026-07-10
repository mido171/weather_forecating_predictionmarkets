from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hkg_t24_beastmode_signal_discovery import (
    CONFIRMATION_START,
    DATASETS_ROOT,
    RESEARCH_ROOT,
    markdown_table,
    require_no_confirmation_dates,
    write_csv,
    write_json,
    write_text,
)
from scripts.run_hkg_t24_long_history_cross_family_interaction_atlas import update_markdown_section

FOLDER_NAME = "0083_expanded_frame_official_anchor_replay"
OFFICIAL_SCORED_PATH = (
    DATASETS_ROOT / "05_hko_historical_rss_forecasts" / "hko_official_t15_scored_pre2024.parquet"
)
CURRENT_0081_SUMMARY_PATH = RESEARCH_ROOT / "0081_rss_gate_stability_stress" / "artifacts" / "summary.json"
CURRENT_0081_PREDICTIONS_PATH = (
    RESEARCH_ROOT / "0081_rss_gate_stability_stress" / "artifacts" / "top_predictions.csv"
)
RAW_CANDIDATE_ID = "official_raw"
EPSILON_MAE = 0.05


@dataclass(frozen=True)
class OnlineBiasSpec:
    candidate_id: str
    group_mode: str
    group_cols: tuple[str, ...]
    half_life_days: float
    min_history: int
    shrink_rows: float
    correction_cap_c: float


@dataclass
class ResidualState:
    weighted_sum: float = 0.0
    weight: float = 0.0
    count: int = 0
    last_date: pd.Timestamp | None = None

    def decay_to(self, current_date: pd.Timestamp, half_life_days: float) -> None:
        if self.last_date is None:
            self.last_date = current_date
            return
        delta_days = int((current_date - self.last_date).days)
        if delta_days <= 0:
            return
        decay = math.exp(-delta_days / half_life_days)
        self.weighted_sum *= decay
        self.weight *= decay
        self.last_date = current_date

    def mean(self) -> float:
        if self.weight <= 0:
            return 0.0
        return self.weighted_sum / self.weight

    def update(self, error_c: float) -> None:
        self.weighted_sum += float(error_c)
        self.weight += 1.0
        self.count += 1


@dataclass
class SelectorState:
    sum_abs_errors: np.ndarray
    count: int = 0

    def update(self, errors: np.ndarray) -> None:
        self.sum_abs_errors += errors
        self.count += 1

    def prior_mae(self) -> np.ndarray:
        if self.count <= 0:
            return np.full_like(self.sum_abs_errors, np.inf, dtype=float)
        return self.sum_abs_errors / self.count


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def load_official_frame() -> pd.DataFrame:
    if not OFFICIAL_SCORED_PATH.exists():
        raise FileNotFoundError(f"Missing official scored export: {OFFICIAL_SCORED_PATH}")
    frame = pd.read_parquet(OFFICIAL_SCORED_PATH).copy()
    required = {"target_date", "target_tmax_c", "forecast_max_c", "forecast_source_family"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"0083 official export missing required columns: {sorted(missing)}")
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
    frame = frame[frame["target_date"].notna()].copy()
    frame = frame[frame["target_date"] < CONFIRMATION_START].copy()
    require_no_confirmation_dates(frame["target_date"], context="0083 official frame")
    if "season" not in frame.columns:
        frame["season"] = frame["month"].astype(int).map(season_from_month)
    frame["month"] = frame["month"].astype(int)
    frame["forecast_max_c"] = pd.to_numeric(frame["forecast_max_c"], errors="coerce")
    frame["target_tmax_c"] = pd.to_numeric(frame["target_tmax_c"], errors="coerce")
    frame = frame[frame["forecast_max_c"].notna() & frame["target_tmax_c"].notna()].copy()
    frame["official_error_c"] = frame["forecast_max_c"] - frame["target_tmax_c"]
    frame["row_index"] = np.arange(len(frame))
    return frame.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True)


def date_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(pd.Timestamp(value).date().isoformat())


def score_arrays(
    *,
    target: np.ndarray,
    prediction: np.ndarray,
    dates: pd.Series,
    prefix: str = "",
) -> dict[str, object]:
    mask = np.isfinite(target) & np.isfinite(prediction)
    if not mask.any():
        return {
            f"{prefix}n": 0,
            f"{prefix}first_date": "",
            f"{prefix}last_date": "",
            f"{prefix}mae": math.nan,
            f"{prefix}rmse": math.nan,
            f"{prefix}bias": math.nan,
            f"{prefix}median_abs_error": math.nan,
        }
    errors = prediction[mask] - target[mask]
    masked_dates = pd.to_datetime(dates[mask], errors="coerce")
    return {
        f"{prefix}n": int(mask.sum()),
        f"{prefix}first_date": date_text(masked_dates.min()),
        f"{prefix}last_date": date_text(masked_dates.max()),
        f"{prefix}mae": float(np.mean(np.abs(errors))),
        f"{prefix}rmse": float(np.sqrt(np.mean(errors**2))),
        f"{prefix}bias": float(np.mean(errors)),
        f"{prefix}median_abs_error": float(np.median(np.abs(errors))),
    }


def make_bias_specs() -> list[OnlineBiasSpec]:
    group_modes: dict[str, tuple[str, ...]] = {
        "global": (),
        "source": ("forecast_source_family",),
        "source_season": ("forecast_source_family", "season"),
        "source_month": ("forecast_source_family", "month"),
    }
    half_lives = [180.0, 730.0]
    min_histories = [30, 90]
    shrink_rows = [90.0]
    specs: list[OnlineBiasSpec] = []
    for mode, cols in group_modes.items():
        for half_life in half_lives:
            for min_history in min_histories:
                for shrink in shrink_rows:
                    candidate_id = (
                        f"online_bias_{mode}_h{int(half_life)}_m{min_history}_"
                        f"s{int(shrink)}_cap1p5"
                    )
                    specs.append(
                        OnlineBiasSpec(
                            candidate_id=candidate_id,
                            group_mode=mode,
                            group_cols=cols,
                            half_life_days=half_life,
                            min_history=min_history,
                            shrink_rows=shrink,
                            correction_cap_c=1.5,
                        )
                    )
    return specs


def group_key(row: pd.Series, cols: tuple[str, ...]) -> tuple[object, ...]:
    if not cols:
        return ("global",)
    return tuple(row[col] for col in cols)


def apply_online_bias(frame: pd.DataFrame, spec: OnlineBiasSpec) -> np.ndarray:
    predictions = frame["forecast_max_c"].to_numpy(dtype=float).copy()
    forecast = frame["forecast_max_c"].to_numpy(dtype=float)
    errors = frame["official_error_c"].to_numpy(dtype=float)
    states: dict[tuple[object, ...], ResidualState] = {}

    for current_date, date_group in frame.groupby("target_date", sort=True, observed=True):
        current_timestamp = pd.Timestamp(current_date)
        pending_updates: list[tuple[tuple[object, ...], float]] = []
        for idx, row in date_group.iterrows():
            key = group_key(row, spec.group_cols)
            state = states.setdefault(key, ResidualState())
            state.decay_to(current_timestamp, spec.half_life_days)
            correction = 0.0
            if state.count >= spec.min_history and state.weight > 0:
                shrink = state.count / (state.count + spec.shrink_rows) if spec.shrink_rows > 0 else 1.0
                correction = state.mean() * shrink
                correction = float(np.clip(correction, -spec.correction_cap_c, spec.correction_cap_c))
            predictions[int(idx)] = forecast[int(idx)] - correction
            pending_updates.append((key, errors[int(idx)]))
        for key, error_c in pending_updates:
            states[key].update(error_c)
    return predictions


def selector_key(row: pd.Series, mode: str) -> tuple[object, ...]:
    if mode == "global":
        return ("global",)
    if mode == "source":
        return (row["forecast_source_family"],)
    raise ValueError(f"Unsupported selector mode: {mode}")


def apply_past_performance_selector(
    frame: pd.DataFrame,
    *,
    candidate_ids: list[str],
    prediction_matrix: np.ndarray,
    mode: str,
    min_prior_rows: int,
    blend_top_k: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    target = frame["target_tmax_c"].to_numpy(dtype=float)
    raw_idx = candidate_ids.index(RAW_CANDIDATE_ID)
    out = prediction_matrix[:, raw_idx].copy()
    selected_ids: list[str] = [RAW_CANDIDATE_ID for _ in range(len(frame))]
    states: dict[tuple[object, ...], SelectorState] = {}

    for _, date_group in frame.groupby("target_date", sort=True, observed=True):
        pending_updates: list[tuple[tuple[object, ...], np.ndarray]] = []
        for idx, row in date_group.iterrows():
            key = selector_key(row, mode)
            state = states.setdefault(
                key,
                SelectorState(sum_abs_errors=np.zeros(len(candidate_ids), dtype=float)),
            )
            row_idx = int(idx)
            if state.count >= min_prior_rows:
                prior_mae = state.prior_mae()
                order = np.argsort(prior_mae)
                if blend_top_k is None:
                    selected_idx = int(order[0])
                    out[row_idx] = prediction_matrix[row_idx, selected_idx]
                    selected_ids[row_idx] = candidate_ids[selected_idx]
                else:
                    chosen = order[:blend_top_k]
                    weights = 1.0 / (prior_mae[chosen] + EPSILON_MAE)
                    weights = weights / weights.sum()
                    out[row_idx] = float(np.dot(prediction_matrix[row_idx, chosen], weights))
                    selected_ids[row_idx] = "blend:" + ";".join(candidate_ids[int(pos)] for pos in chosen)
            errors = np.abs(prediction_matrix[row_idx, :] - target[row_idx])
            pending_updates.append((key, errors))
        for key, errors in pending_updates:
            states[key].update(errors)
    return out, selected_ids


def frame_masks(frame: pd.DataFrame, current_dates: set[pd.Timestamp]) -> dict[str, np.ndarray]:
    dates = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
    old = dates.isin(current_dates).to_numpy(dtype=bool)
    source = frame["forecast_source_family"].astype(str)
    return {
        "old_frame_": old,
        "newly_available_": ~old,
        "press_": source.eq("press_archive").to_numpy(dtype=bool),
        "rss_": source.eq("rss_archive").to_numpy(dtype=bool),
    }


def score_candidate(
    frame: pd.DataFrame,
    *,
    candidate_id: str,
    candidate_class: str,
    prediction: np.ndarray,
    masks: dict[str, np.ndarray],
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    target = frame["target_tmax_c"].to_numpy(dtype=float)
    dates = pd.to_datetime(frame["target_date"], errors="coerce")
    row = {
        "candidate_id": candidate_id,
        "candidate_class": candidate_class,
        **score_arrays(target=target, prediction=prediction, dates=dates),
    }
    for prefix, mask in masks.items():
        row.update(
            score_arrays(
                target=target[mask],
                prediction=prediction[mask],
                dates=dates[mask],
                prefix=prefix,
            )
        )
    if extra:
        row.update(extra)
    return row


def load_current_0081_dates_and_score() -> tuple[set[pd.Timestamp], dict[str, object]]:
    if not CURRENT_0081_PREDICTIONS_PATH.exists() or not CURRENT_0081_SUMMARY_PATH.exists():
        return set(), {}
    summary = json.loads(CURRENT_0081_SUMMARY_PATH.read_text(encoding="utf-8"))
    best_id = str(summary.get("best_candidate", ""))
    predictions = pd.read_csv(CURRENT_0081_PREDICTIONS_PATH)
    predictions["target_date"] = pd.to_datetime(predictions["target_date"], errors="coerce").dt.normalize()
    current_dates = set(predictions["target_date"].dropna().drop_duplicates())
    if best_id and "candidate_id" in predictions.columns:
        predictions = predictions[predictions["candidate_id"].astype(str).eq(best_id)].copy()
    if predictions.empty:
        return current_dates, summary
    target = pd.to_numeric(predictions["current_target_tmax_c"], errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(predictions["candidate_prediction_c"], errors="coerce").to_numpy(dtype=float)
    dates = pd.to_datetime(predictions["target_date"], errors="coerce")
    score = score_arrays(target=target, prediction=pred, dates=dates)
    return current_dates, {
        "candidate_id": best_id,
        "source": "0081_current_champion_old_frame",
        **score,
    }


def build_predictions_and_scoreboard(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    current_dates, current_0081_score = load_current_0081_dates_and_score()
    masks = frame_masks(frame, current_dates)
    predictions: dict[str, np.ndarray] = {RAW_CANDIDATE_ID: frame["forecast_max_c"].to_numpy(dtype=float)}
    definitions: list[dict[str, object]] = [
        {
            "candidate_id": RAW_CANDIDATE_ID,
            "candidate_class": "official_raw",
            "group_mode": "",
            "half_life_days": "",
            "min_history": "",
            "shrink_rows": "",
            "correction_cap_c": "",
            "selector_mode": "",
            "selector_min_prior_rows": "",
            "blend_top_k": "",
        }
    ]
    rows = [
        score_candidate(
            frame,
            candidate_id=RAW_CANDIDATE_ID,
            candidate_class="official_raw",
            prediction=predictions[RAW_CANDIDATE_ID],
            masks=masks,
        )
    ]

    for spec in make_bias_specs():
        prediction = apply_online_bias(frame, spec)
        predictions[spec.candidate_id] = prediction
        definitions.append(
            {
                "candidate_id": spec.candidate_id,
                "candidate_class": "online_past_only_bias_correction",
                "group_mode": spec.group_mode,
                "half_life_days": spec.half_life_days,
                "min_history": spec.min_history,
                "shrink_rows": spec.shrink_rows,
                "correction_cap_c": spec.correction_cap_c,
                "selector_mode": "",
                "selector_min_prior_rows": "",
                "blend_top_k": "",
            }
        )
        rows.append(
            score_candidate(
                frame,
                candidate_id=spec.candidate_id,
                candidate_class="online_past_only_bias_correction",
                prediction=prediction,
                masks=masks,
                extra={
                    "group_mode": spec.group_mode,
                    "half_life_days": spec.half_life_days,
                    "min_history": spec.min_history,
                    "shrink_rows": spec.shrink_rows,
                    "correction_cap_c": spec.correction_cap_c,
                },
            )
        )

    base_candidate_ids = list(predictions)
    prediction_matrix = np.column_stack([predictions[candidate_id] for candidate_id in base_candidate_ids])
    selector_records: list[dict[str, object]] = []
    for mode in ("global", "source"):
        for min_prior_rows in (90, 365, 730):
            selector_id = f"prior_selector_{mode}_min{min_prior_rows}"
            selector_prediction, selected_ids = apply_past_performance_selector(
                frame,
                candidate_ids=base_candidate_ids,
                prediction_matrix=prediction_matrix,
                mode=mode,
                min_prior_rows=min_prior_rows,
            )
            predictions[selector_id] = selector_prediction
            definitions.append(
                {
                    "candidate_id": selector_id,
                    "candidate_class": "past_performance_selector",
                    "group_mode": "",
                    "half_life_days": "",
                    "min_history": "",
                    "shrink_rows": "",
                    "correction_cap_c": "",
                    "selector_mode": mode,
                    "selector_min_prior_rows": min_prior_rows,
                    "blend_top_k": "",
                }
            )
            rows.append(
                score_candidate(
                    frame,
                    candidate_id=selector_id,
                    candidate_class="past_performance_selector",
                    prediction=selector_prediction,
                    masks=masks,
                    extra={"selector_mode": mode, "selector_min_prior_rows": min_prior_rows},
                )
            )
            selector_records.append(
                {
                    "candidate_id": selector_id,
                    "selected_candidate_ids": selected_ids,
                }
            )

            blend_id = f"prior_blend_{mode}_top5_min{min_prior_rows}"
            blend_prediction, blend_ids = apply_past_performance_selector(
                frame,
                candidate_ids=base_candidate_ids,
                prediction_matrix=prediction_matrix,
                mode=mode,
                min_prior_rows=min_prior_rows,
                blend_top_k=5,
            )
            predictions[blend_id] = blend_prediction
            definitions.append(
                {
                    "candidate_id": blend_id,
                    "candidate_class": "past_performance_blend",
                    "group_mode": "",
                    "half_life_days": "",
                    "min_history": "",
                    "shrink_rows": "",
                    "correction_cap_c": "",
                    "selector_mode": mode,
                    "selector_min_prior_rows": min_prior_rows,
                    "blend_top_k": 5,
                }
            )
            rows.append(
                score_candidate(
                    frame,
                    candidate_id=blend_id,
                    candidate_class="past_performance_blend",
                    prediction=blend_prediction,
                    masks=masks,
                    extra={"selector_mode": mode, "selector_min_prior_rows": min_prior_rows, "blend_top_k": 5},
                )
            )
            selector_records.append(
                {
                    "candidate_id": blend_id,
                    "selected_candidate_ids": blend_ids,
                }
            )

    scoreboard = pd.DataFrame(rows).sort_values(["mae", "rmse"]).reset_index(drop=True)
    definitions_frame = pd.DataFrame(definitions)
    best_id = str(scoreboard.iloc[0]["candidate_id"])
    best_prediction = predictions[best_id]
    selected_ids_lookup = {
        record["candidate_id"]: record["selected_candidate_ids"] for record in selector_records
    }
    selected_ids = selected_ids_lookup.get(best_id, [best_id for _ in range(len(frame))])
    top_predictions = frame[
        [
            "target_date",
            "forecast_source_family",
            "target_tmax_c",
            "forecast_max_c",
            "official_error_c",
            "season",
            "month",
        ]
    ].copy()
    top_predictions["candidate_id"] = best_id
    top_predictions["candidate_prediction_c"] = best_prediction
    top_predictions["candidate_error_c"] = best_prediction - top_predictions["target_tmax_c"]
    top_predictions["selected_candidate_id"] = selected_ids
    top_predictions["frame_segment"] = np.where(
        pd.to_datetime(top_predictions["target_date"]).isin(current_dates),
        "current_0081_frame",
        "newly_available_official_frame",
    )
    comparison_rows = [
        {
            "source": "0083_official_raw_expanded_frame",
            "candidate_id": RAW_CANDIDATE_ID,
            **scoreboard[scoreboard["candidate_id"].eq(RAW_CANDIDATE_ID)].iloc[0].to_dict(),
        },
        {
            "source": "0083_best_expanded_frame",
            "candidate_id": best_id,
            **scoreboard.iloc[0].to_dict(),
        },
    ]
    if current_0081_score:
        comparison_rows.append(current_0081_score)
    comparison = pd.DataFrame(comparison_rows)
    return scoreboard, definitions_frame, top_predictions, comparison


def source_scoreboard(frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    target = predictions["target_tmax_c"].to_numpy(dtype=float)
    pred = predictions["candidate_prediction_c"].to_numpy(dtype=float)
    dates = pd.to_datetime(predictions["target_date"], errors="coerce")
    for source, group in predictions.groupby("forecast_source_family", observed=True):
        mask = np.asarray(predictions.index.isin(group.index), dtype=bool)
        rows.append(
            {
                "forecast_source_family": str(source),
                **score_arrays(target=target[mask], prediction=pred[mask], dates=dates[mask]),
            }
        )
    raw = frame["forecast_max_c"].to_numpy(dtype=float)
    for source, group in frame.groupby("forecast_source_family", observed=True):
        mask = np.asarray(frame.index.isin(group.index), dtype=bool)
        rows.append(
            {
                "forecast_source_family": f"{source}_official_raw",
                **score_arrays(
                    target=frame["target_tmax_c"].to_numpy(dtype=float)[mask],
                    prediction=raw[mask],
                    dates=pd.to_datetime(frame["target_date"], errors="coerce")[mask],
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("forecast_source_family").reset_index(drop=True)


def build_summary(
    *,
    generated_at: str,
    frame: pd.DataFrame,
    scoreboard: pd.DataFrame,
    comparison: pd.DataFrame,
) -> dict[str, object]:
    best = scoreboard.iloc[0].to_dict()
    raw = scoreboard[scoreboard["candidate_id"].eq(RAW_CANDIDATE_ID)].iloc[0].to_dict()
    dates = pd.to_datetime(frame["target_date"], errors="coerce")
    return {
        "generated_at_utc": generated_at,
        "folder": FOLDER_NAME,
        "official_scored_path": str(OFFICIAL_SCORED_PATH),
        "rows": int(len(frame)),
        "unique_target_days": int(dates.nunique()),
        "first_target_date": date_text(dates.min()),
        "last_target_date": date_text(dates.max()),
        "confirmation_start": CONFIRMATION_START.date().isoformat(),
        "uses_2024_plus_rows": False,
        "candidate_count": int(len(scoreboard)),
        "best_candidate": str(best["candidate_id"]),
        "best_candidate_class": str(best["candidate_class"]),
        "best_mae": float(best["mae"]),
        "best_rmse": float(best["rmse"]),
        "best_bias": float(best["bias"]),
        "official_raw_mae": float(raw["mae"]),
        "official_raw_rmse": float(raw["rmse"]),
        "best_delta_mae_vs_official_raw": float(best["mae"]) - float(raw["mae"]),
        "best_old_frame_mae": float(best["old_frame_mae"]),
        "best_newly_available_mae": float(best["newly_available_mae"]),
        "comparison": comparison.to_dict("records"),
        "status": "expanded_frame_baseline_not_champion_replacement",
        "next_recommended_task": (
            "Run 0084 to convert the best 0083 expanded-frame official-anchor correction into fold/source "
            "hardened specialists, then compare against 0081 on the old frame and against official raw on the "
            "newly available 2004-08-06 to 2011-09-13 press segment. Keep 2024+ sealed."
        ),
    }


def build_readme(
    *,
    generated_at: str,
    summary: dict[str, object],
    scoreboard: pd.DataFrame,
    source_scores: pd.DataFrame,
    comparison: pd.DataFrame,
) -> str:
    top = scoreboard.head(15)
    return f"""# 0083 Expanded-Frame Official-Anchor Replay

Generated: `{generated_at}`

## Purpose

`0082` proved that the current `0081` champion predictions are still locked to an older partial frame. This run does not pretend otherwise. Instead, it uses the refreshed `0044` official forecast export directly and asks: what can we score, leakage-free, on every currently available pre-2024 official forecast row?

The inputs are the current local official forecast rows only: press archive rows from `2000-01-02` to `2011-09-13`, plus RSS rows from `2021-04-14` to `2023-12-31`. No 2024+ confirmation rows are opened.

## Method

The run evaluates:

- raw HKO official forecast max temperature;
- past-only online residual/bias corrections by global, source, source-season, and source-month contexts;
- half-life memories of 180 and 730 days in this compact first-pass replay;
- support shrinkage and minimum-history gates;
- past-performance selectors and top-5 blends that choose only from candidate errors observed before the target date.

Every correction for a target date uses only earlier target dates. Rows sharing the same target date are predicted before that date's errors are added to memory, so there is no same-day leakage.

## Main Result

| Field | Value |
|---|---|
| Status | `{summary['status']}` |
| Rows | `{summary['rows']}` |
| Unique target days | `{summary['unique_target_days']}` |
| Date range | `{summary['first_target_date']}` to `{summary['last_target_date']}` |
| Candidate count | `{summary['candidate_count']}` |
| Best candidate | `{summary['best_candidate']}` |
| Best candidate class | `{summary['best_candidate_class']}` |
| Best MAE | `{summary['best_mae']}` |
| Best RMSE | `{summary['best_rmse']}` |
| Official raw MAE | `{summary['official_raw_mae']}` |
| Delta MAE vs official raw | `{summary['best_delta_mae_vs_official_raw']}` |
| Best old-frame MAE | `{summary['best_old_frame_mae']}` |
| Best newly-available-frame MAE | `{summary['best_newly_available_mae']}` |
| 2024+ rows used | `{summary['uses_2024_plus_rows']}` |

## Top Candidates

{markdown_table(top, max_rows=15)}

## Source Scores

{markdown_table(source_scores, max_rows=20)}

## Comparison Context

{markdown_table(comparison, max_rows=10)}

## Interpretation

This is a refreshed-frame baseline, not a new production champion. Its value is that it scores the newly available official forecast rows now, instead of waiting for the whole raw-detail backfill to finish. The result tells us how much simple past-only bias memory can extract from the expanded official archive.

If the best rule beats raw official forecast on the expanded frame but remains worse than the old `0081` score on the old partial frame, that is expected: `0081` includes station/model-family machinery that has not yet been replayed on the newly available press rows. The correct next step is to harden the best 0083 correction family and then combine it with the existing `0081` specialist logic only after the stale router dependency has been regenerated or replaced.

## Exact Next Recommended Task

{summary['next_recommended_task']}
"""


def update_milestones(summary: dict[str, object], scoreboard: pd.DataFrame) -> None:
    section = f"""
Generated by:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_0083_expanded_frame_official_anchor_replay.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Current evidence | Status |
|---|---|---|
| Expanded official frame | `{summary['rows']}` rows, `{summary['first_target_date']}` to `{summary['last_target_date']}` | Scored pre-2024 only |
| Best 0083 candidate | `{summary['best_candidate']}` | Refreshed-frame baseline |
| Best MAE / RMSE | `{summary['best_mae']}` / `{summary['best_rmse']}` | Compared to official raw, not a new champion |
| Official raw MAE / RMSE | `{summary['official_raw_mae']}` / `{summary['official_raw_rmse']}` | Baseline |
| Delta vs official raw | `{summary['best_delta_mae_vs_official_raw']}` | Past-only correction lift |
| Leakage | `0` 2024+ rows | PASS |

Top candidates:

{markdown_table(scoreboard.head(8), max_rows=8)}

Interpretation: `0083` uses the expanded current official forecast archive instead of the stale old champion frame. It is not a replacement for `0081`; it is the first leakage-safe refreshed-frame baseline while the forecast backfill continues.
"""
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="0083 Expanded-Frame Official-Anchor Replay",
        section=section,
        insert_before="\n## Current Blockers And Gaps\n",
    )
    update_markdown_section(
        REPO_ROOT / "MILESTONES.md",
        heading="Exact Next Recommended Codex Task",
        section=(
            "Implement `0084_expanded_frame_hardened_official_specialists`: harden the best 0083 "
            "official-anchor correction family by source, era, season, and frame segment; compare it against "
            "official raw on the newly available press rows and against `0081` on the old frame; keep 2024+ sealed."
        ),
    )


def run() -> dict[str, object]:
    generated_at = now_utc()
    frame = load_official_frame()
    scoreboard, definitions, top_predictions, comparison = build_predictions_and_scoreboard(frame)
    source_scores = source_scoreboard(frame, top_predictions)
    summary = build_summary(
        generated_at=generated_at,
        frame=frame,
        scoreboard=scoreboard,
        comparison=comparison,
    )

    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    write_csv(artifacts / "scoreboard.csv", scoreboard)
    write_csv(artifacts / "candidate_definitions.csv", definitions)
    write_csv(artifacts / "top_predictions.csv", top_predictions)
    write_csv(artifacts / "source_scoreboard.csv", source_scores)
    write_csv(artifacts / "comparison.csv", comparison)
    write_json(artifacts / "summary.json", summary)
    write_json(RESEARCH_ROOT / "expanded_frame_official_anchor_replay_manifest.json", summary)
    write_text(
        folder / "README.md",
        build_readme(
            generated_at=generated_at,
            summary=summary,
            scoreboard=scoreboard,
            source_scores=source_scores,
            comparison=comparison,
        ),
    )
    update_milestones(summary, scoreboard)
    require_no_confirmation_dates(top_predictions["target_date"], context="0083 top predictions")
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(
        description="Run expanded-frame leakage-safe official-anchor replay on current HKG Tmax forecast archive."
    ).parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
