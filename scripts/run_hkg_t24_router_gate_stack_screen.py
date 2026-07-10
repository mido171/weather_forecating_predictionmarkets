from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

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
    score_prediction_frame,
    write_csv,
    write_json,
    write_text,
)
from scripts.run_hkg_t24_station_network_forecast_residual_interaction_mining import (  # noqa: E402
    LATE_EVAL_START,
)

FOLDER_NAME = "0049_router_gate_stack_screen"
SCREEN_STAGE = "stage1_router_gate_prior_only_stack"
ROUTER_ARTIFACTS = RESEARCH_ROOT / "0042_trust_router_sensitivity" / "artifacts"
GATE_ARTIFACTS = RESEARCH_ROOT / "0048_gated_residual_specialist_screen" / "artifacts"
ROUTER_PREDICTIONS_PATH = ROUTER_ARTIFACTS / "top_sensitivity_predictions.csv"
ROUTER_SCOREBOARD_PATH = ROUTER_ARTIFACTS / "sensitivity_scoreboard.csv"
GATE_PREDICTIONS_PATH = GATE_ARTIFACTS / "sample_candidate_predictions.csv"
GATE_SCOREBOARD_PATH = GATE_ARTIFACTS / "candidate_scoreboard.csv"
TOP_ROUTER_CANDIDATES = 4
TOP_GATE_CANDIDATES = 6
MIN_HISTORY_OPTIONS = (30, 80)
PRIOR_MODES: tuple[str, ...] = ("prior_best", "prior_inverse_mae", "prior_positive_lift")
SAME_SOURCE_OPTIONS = (False, True)
FIXED_GATE_SCALES = (0.25, 0.50)


CombineMode = Literal["prior_best", "prior_inverse_mae", "prior_positive_lift", "fixed_gate_residual"]


@dataclass(frozen=True)
class StackCandidate:
    candidate_id: str
    router_candidate_id: str
    gate_candidate_id: str
    mode: CombineMode
    same_source: bool
    min_history: int
    gate_scale: float


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def slug(text: str, *, limit: int = 150) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:limit]


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return pd.read_csv(path)


def ordered_unique(values: list[str], *, limit: int) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
        if len(out) >= limit:
            break
    return out


def top_router_candidate_ids(scoreboard: pd.DataFrame, predictions: pd.DataFrame) -> list[str]:
    available = set(predictions["candidate_id"].astype(str))
    ranked: list[str] = []
    if not scoreboard.empty:
        ranked.extend(scoreboard.sort_values(["mae", "late_eval_mae", "rmse"])["candidate_id"].astype(str).head(3))
        ranked.extend(scoreboard.sort_values(["late_eval_mae", "mae", "rmse"])["candidate_id"].astype(str).head(3))
    ranked = [candidate_id for candidate_id in ranked if candidate_id in available]
    return ordered_unique(ranked, limit=TOP_ROUTER_CANDIDATES)


def top_gate_candidate_ids(scoreboard: pd.DataFrame, predictions: pd.DataFrame) -> list[str]:
    available = set(predictions["candidate_id"].astype(str))
    ranked: list[str] = []
    available_scores = scoreboard[scoreboard["candidate_id"].astype(str).isin(available)].copy()
    if not available_scores.empty:
        ranked.extend(available_scores.sort_values(["full_mae", "late_mae", "full_rmse"])["candidate_id"].astype(str).head(4))
        ranked.extend(available_scores.sort_values(["late_mae", "full_mae", "late_rmse"])["candidate_id"].astype(str).head(4))
    else:
        ranked.extend(predictions["candidate_id"].astype(str).drop_duplicates())
    return ordered_unique(ranked, limit=TOP_GATE_CANDIDATES)


def load_router_predictions() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    predictions = load_csv(ROUTER_PREDICTIONS_PATH)
    scoreboard = load_csv(ROUTER_SCOREBOARD_PATH)
    predictions["target_date"] = pd.to_datetime(predictions["target_date"], errors="coerce").dt.normalize()
    predictions = predictions[predictions["target_date"].notna()].copy()
    require_no_confirmation_dates(predictions["target_date"], context="0049 router predictions")
    selected_ids = top_router_candidate_ids(scoreboard, predictions)
    if not selected_ids:
        raise RuntimeError("No 0042 router candidates were available for 0049")
    keep = [
        "target_date",
        "forecast_source_family",
        "target_tmax_c",
        "official_raw",
        "anchor_0038_c",
        "candidate_prediction_c",
        "candidate_id",
    ]
    missing = set(keep).difference(predictions.columns)
    if missing:
        raise ValueError(f"0042 router predictions missing columns: {sorted(missing)}")
    predictions = predictions[predictions["candidate_id"].astype(str).isin(selected_ids)][keep].copy()
    return predictions, scoreboard, selected_ids


def load_gate_predictions() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    predictions = load_csv(GATE_PREDICTIONS_PATH)
    scoreboard = load_csv(GATE_SCOREBOARD_PATH)
    predictions["target_date"] = pd.to_datetime(predictions["target_date"], errors="coerce").dt.normalize()
    predictions = predictions[predictions["target_date"].notna()].copy()
    require_no_confirmation_dates(predictions["target_date"], context="0049 gate predictions")
    selected_ids = top_gate_candidate_ids(scoreboard, predictions)
    if not selected_ids:
        raise RuntimeError("No 0048 gate candidates were available for 0049")
    keep = [
        "target_date",
        "forecast_source_family",
        "target_tmax_c",
        "forecast_max_c",
        "candidate_prediction_c",
        "residual_correction_c",
        "candidate_id",
    ]
    missing = set(keep).difference(predictions.columns)
    if missing:
        raise ValueError(f"0048 gate predictions missing columns: {sorted(missing)}")
    predictions = predictions[predictions["candidate_id"].astype(str).isin(selected_ids)][keep].copy()
    return predictions, scoreboard, selected_ids


def stack_candidate_id(
    *,
    router_candidate_id: str,
    gate_candidate_id: str,
    mode: CombineMode,
    same_source: bool,
    min_history: int,
    gate_scale: float,
) -> str:
    source = "same_source" if same_source else "all_prior"
    scale_token = str(gate_scale).replace(".", "p")
    raw = (
        f"0049_{mode}_{source}_h{min_history}_s{scale_token}_"
        f"r_{router_candidate_id}_g_{gate_candidate_id}"
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"{slug(raw, limit=170)}_{digest}"


def build_candidate_catalog(router_ids: list[str], gate_ids: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for router_candidate_id in router_ids:
        for gate_candidate_id in gate_ids:
            for min_history in MIN_HISTORY_OPTIONS:
                for same_source in SAME_SOURCE_OPTIONS:
                    for mode in PRIOR_MODES:
                        candidate_id = stack_candidate_id(
                            router_candidate_id=router_candidate_id,
                            gate_candidate_id=gate_candidate_id,
                            mode=mode,  # type: ignore[arg-type]
                            same_source=same_source,
                            min_history=min_history,
                            gate_scale=0.0,
                        )
                        rows.append(
                            {
                                "candidate_id": candidate_id,
                                "router_candidate_id": router_candidate_id,
                                "gate_candidate_id": gate_candidate_id,
                                "mode": mode,
                                "same_source": same_source,
                                "min_history": min_history,
                                "gate_scale": 0.0,
                            }
                        )
            for gate_scale in FIXED_GATE_SCALES:
                candidate_id = stack_candidate_id(
                    router_candidate_id=router_candidate_id,
                    gate_candidate_id=gate_candidate_id,
                    mode="fixed_gate_residual",
                    same_source=False,
                    min_history=0,
                    gate_scale=gate_scale,
                )
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "router_candidate_id": router_candidate_id,
                        "gate_candidate_id": gate_candidate_id,
                        "mode": "fixed_gate_residual",
                        "same_source": False,
                        "min_history": 0,
                        "gate_scale": gate_scale,
                    }
                )
    catalog = pd.DataFrame(rows)
    if not catalog["candidate_id"].is_unique:
        raise RuntimeError("0049 candidate IDs are not unique")
    return catalog


def merge_pair_frame(router_predictions: pd.DataFrame, gate_predictions: pd.DataFrame, spec: StackCandidate) -> pd.DataFrame:
    router = router_predictions[router_predictions["candidate_id"].astype(str).eq(spec.router_candidate_id)].copy()
    gate = gate_predictions[gate_predictions["candidate_id"].astype(str).eq(spec.gate_candidate_id)].copy()
    router = router.rename(
        columns={
            "candidate_prediction_c": "router_prediction_c",
            "candidate_id": "router_candidate_id",
        }
    )
    gate = gate.rename(
        columns={
            "candidate_prediction_c": "gate_prediction_c",
            "candidate_id": "gate_candidate_id",
            "forecast_max_c": "gate_anchor_prediction_c",
        }
    )
    merged = router.merge(
        gate[
            [
                "target_date",
                "forecast_source_family",
                "target_tmax_c",
                "gate_anchor_prediction_c",
                "gate_prediction_c",
                "residual_correction_c",
                "gate_candidate_id",
            ]
        ],
        on=["target_date", "forecast_source_family", "target_tmax_c"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise RuntimeError(f"No overlapping rows for {spec.candidate_id}")
    return merged.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True)


def prior_mae(values: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[int, float]:
    valid = mask & np.isfinite(values) & np.isfinite(target)
    count = int(valid.sum())
    if count == 0:
        return 0, math.nan
    return count, float(np.abs(values[valid] - target[valid]).mean())


def finite_inverse_weights(maes: list[float]) -> np.ndarray:
    weights = np.array([0.0 if not np.isfinite(value) else 1.0 / max(value, 1e-6) for value in maes], dtype=float)
    if weights.sum() <= 0.0:
        return weights
    return weights / weights.sum()


def combine_prior_predictions(frame: pd.DataFrame, spec: StackCandidate) -> pd.DataFrame:
    ordered = frame.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True).copy()
    dates = pd.to_datetime(ordered["target_date"], errors="coerce").dt.normalize().to_numpy(dtype="datetime64[ns]")
    sources = ordered["forecast_source_family"].astype(str).to_numpy()
    target = pd.to_numeric(ordered["target_tmax_c"], errors="coerce").to_numpy(dtype=float)
    anchor = pd.to_numeric(ordered["anchor_0038_c"], errors="coerce").to_numpy(dtype=float)
    router = pd.to_numeric(ordered["router_prediction_c"], errors="coerce").to_numpy(dtype=float)
    gate = pd.to_numeric(ordered["gate_prediction_c"], errors="coerce").to_numpy(dtype=float)

    predictions: list[float] = []
    selected_family: list[str] = []
    prior_rows: list[int] = []
    router_prior_mae: list[float] = []
    gate_prior_mae: list[float] = []
    anchor_prior_mae: list[float] = []
    selected_weight: list[float] = []

    for index, target_date in enumerate(dates):
        prior_mask = dates < target_date
        if spec.same_source:
            prior_mask &= sources == sources[index]
        router_count, router_mae = prior_mae(router, target, prior_mask)
        gate_count, gate_mae = prior_mae(gate, target, prior_mask)
        anchor_count, anchor_mae = prior_mae(anchor, target, prior_mask)
        router_ok = router_count >= spec.min_history and np.isfinite(router_mae)
        gate_ok = gate_count >= spec.min_history and np.isfinite(gate_mae)
        anchor_ok = anchor_count >= spec.min_history and np.isfinite(anchor_mae)

        prediction = router[index]
        family = "router_fallback"
        weight = 0.0
        count = max(router_count, gate_count)

        if spec.mode == "prior_best":
            if router_ok and gate_ok:
                if gate_mae < router_mae:
                    prediction = gate[index]
                    family = "gate"
                    weight = 1.0
                else:
                    prediction = router[index]
                    family = "router"
                    weight = 0.0
            elif gate_ok:
                prediction = gate[index]
                family = "gate_only_eligible"
                weight = 1.0
            elif router_ok:
                prediction = router[index]
                family = "router_only_eligible"
        elif spec.mode == "prior_inverse_mae":
            if router_ok and gate_ok:
                weights = finite_inverse_weights([router_mae, gate_mae])
                prediction = float(weights[0] * router[index] + weights[1] * gate[index])
                family = "router_gate_inverse_mae"
                weight = float(weights[1])
            elif gate_ok:
                prediction = gate[index]
                family = "gate_only_eligible"
                weight = 1.0
            elif router_ok:
                prediction = router[index]
                family = "router_only_eligible"
        elif spec.mode == "prior_positive_lift":
            eligible_values: list[float] = []
            eligible_maes: list[float] = []
            eligible_names: list[str] = []
            if anchor_ok and router_ok and router_mae < anchor_mae:
                eligible_values.append(router[index])
                eligible_maes.append(router_mae)
                eligible_names.append("router")
            if anchor_ok and gate_ok and gate_mae < anchor_mae:
                eligible_values.append(gate[index])
                eligible_maes.append(gate_mae)
                eligible_names.append("gate")
            if eligible_values:
                weights = finite_inverse_weights(eligible_maes)
                prediction = float(np.dot(weights, np.array(eligible_values, dtype=float)))
                family = "positive_lift_" + "_".join(eligible_names)
                weight = float(weights[eligible_names.index("gate")]) if "gate" in eligible_names else 0.0

        predictions.append(float(prediction))
        selected_family.append(family)
        prior_rows.append(count)
        router_prior_mae.append(float(router_mae) if np.isfinite(router_mae) else math.nan)
        gate_prior_mae.append(float(gate_mae) if np.isfinite(gate_mae) else math.nan)
        anchor_prior_mae.append(float(anchor_mae) if np.isfinite(anchor_mae) else math.nan)
        selected_weight.append(weight)

    ordered["candidate_prediction_c"] = predictions
    ordered["selected_family"] = selected_family
    ordered["selected_prior_count"] = prior_rows
    ordered["router_prior_mae"] = router_prior_mae
    ordered["gate_prior_mae"] = gate_prior_mae
    ordered["anchor_prior_mae"] = anchor_prior_mae
    ordered["gate_weight"] = selected_weight
    return ordered


def combine_fixed_gate_residual(frame: pd.DataFrame, spec: StackCandidate) -> pd.DataFrame:
    out = frame.sort_values(["target_date", "forecast_source_family"]).reset_index(drop=True).copy()
    out["candidate_prediction_c"] = out["router_prediction_c"] + spec.gate_scale * out["residual_correction_c"]
    out["selected_family"] = f"router_plus_{spec.gate_scale:g}x_gate_residual"
    out["selected_prior_count"] = 0
    out["router_prior_mae"] = math.nan
    out["gate_prior_mae"] = math.nan
    out["anchor_prior_mae"] = math.nan
    out["gate_weight"] = spec.gate_scale
    return out


def combine_predictions(frame: pd.DataFrame, spec: StackCandidate) -> pd.DataFrame:
    if spec.mode == "fixed_gate_residual":
        out = combine_fixed_gate_residual(frame, spec)
    else:
        out = combine_prior_predictions(frame, spec)
    out["candidate_id"] = spec.candidate_id
    out["combine_mode"] = spec.mode
    out["same_source"] = spec.same_source
    out["min_history"] = spec.min_history
    out["gate_scale"] = spec.gate_scale
    out["router_candidate_id"] = spec.router_candidate_id
    out["gate_candidate_id"] = spec.gate_candidate_id
    keep = [
        "target_date",
        "forecast_source_family",
        "target_tmax_c",
        "official_raw",
        "anchor_0038_c",
        "router_prediction_c",
        "gate_prediction_c",
        "residual_correction_c",
        "candidate_prediction_c",
        "selected_family",
        "selected_prior_count",
        "router_prior_mae",
        "gate_prior_mae",
        "anchor_prior_mae",
        "gate_weight",
        "candidate_id",
        "combine_mode",
        "same_source",
        "min_history",
        "gate_scale",
        "router_candidate_id",
        "gate_candidate_id",
    ]
    return out[keep].copy()


def late_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[pd.to_datetime(frame["target_date"], errors="coerce") >= LATE_EVAL_START].copy()


def score_stack_candidate(predictions: pd.DataFrame, spec: StackCandidate) -> dict[str, object]:
    score = score_prediction_frame(predictions, "candidate_prediction_c")
    late_score = score_prediction_frame(late_frame(predictions), "candidate_prediction_c")
    router_score = score_prediction_frame(predictions.rename(columns={"router_prediction_c": "prediction"}), "prediction")
    gate_score = score_prediction_frame(predictions.rename(columns={"gate_prediction_c": "prediction"}), "prediction")
    anchor_score = score_prediction_frame(predictions.rename(columns={"anchor_0038_c": "prediction"}), "prediction")
    official_score = score_prediction_frame(predictions.rename(columns={"official_raw": "prediction"}), "prediction")
    late_router = score_prediction_frame(late_frame(predictions).rename(columns={"router_prediction_c": "prediction"}), "prediction")
    late_gate = score_prediction_frame(late_frame(predictions).rename(columns={"gate_prediction_c": "prediction"}), "prediction")
    late_anchor = score_prediction_frame(late_frame(predictions).rename(columns={"anchor_0038_c": "prediction"}), "prediction")
    return {
        "candidate_id": spec.candidate_id,
        "router_candidate_id": spec.router_candidate_id,
        "gate_candidate_id": spec.gate_candidate_id,
        "combine_mode": spec.mode,
        "same_source": spec.same_source,
        "min_history": spec.min_history,
        "gate_scale": spec.gate_scale,
        **score,
        "late_eval_n": late_score["n"],
        "late_eval_first_date": late_score["first_date"],
        "late_eval_last_date": late_score["last_date"],
        "late_eval_mae": late_score["mae"],
        "late_eval_rmse": late_score["rmse"],
        "router_same_rows_mae": router_score["mae"],
        "gate_same_rows_mae": gate_score["mae"],
        "anchor_same_rows_mae": anchor_score["mae"],
        "official_same_rows_mae": official_score["mae"],
        "delta_vs_router": float(score["mae"] - router_score["mae"]),
        "delta_vs_gate": float(score["mae"] - gate_score["mae"]),
        "delta_vs_anchor": float(score["mae"] - anchor_score["mae"]),
        "late_eval_delta_vs_router": float(late_score["mae"] - late_router["mae"]),
        "late_eval_delta_vs_gate": float(late_score["mae"] - late_gate["mae"]),
        "late_eval_delta_vs_anchor": float(late_score["mae"] - late_anchor["mae"]),
        "mean_gate_weight": float(pd.to_numeric(predictions["gate_weight"], errors="coerce").fillna(0.0).mean()),
        "non_router_rows": int(
            (~predictions["selected_family"].astype(str).str.contains("router_fallback|router_only_eligible|router$", regex=True)).sum()
        ),
    }


def specs_from_catalog(catalog: pd.DataFrame) -> list[StackCandidate]:
    specs: list[StackCandidate] = []
    for row in catalog.itertuples(index=False):
        specs.append(
            StackCandidate(
                candidate_id=str(row.candidate_id),
                router_candidate_id=str(row.router_candidate_id),
                gate_candidate_id=str(row.gate_candidate_id),
                mode=str(row.mode),  # type: ignore[arg-type]
                same_source=bool(row.same_source),
                min_history=int(row.min_history),
                gate_scale=float(row.gate_scale),
            )
        )
    return specs


def run_stack_screen(
    router_predictions: pd.DataFrame,
    gate_predictions: pd.DataFrame,
    catalog: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for spec in specs_from_catalog(catalog):
        pair_frame = merge_pair_frame(router_predictions, gate_predictions, spec)
        predictions = combine_predictions(pair_frame, spec)
        score_rows.append(score_stack_candidate(predictions, spec))
        prediction_frames.append(predictions)
    scoreboard = pd.DataFrame(score_rows)
    if not scoreboard.empty:
        scoreboard = scoreboard.sort_values(["late_eval_mae", "mae", "rmse"]).reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    require_no_confirmation_dates(predictions["target_date"], context="0049 stack predictions")
    return scoreboard, predictions


def input_coverage(
    router_predictions: pd.DataFrame,
    gate_predictions: pd.DataFrame,
    router_ids: list[str],
    gate_ids: list[str],
) -> pd.DataFrame:
    rows = []
    for name, frame, ids in (
        ("0042_router_top_predictions", router_predictions, router_ids),
        ("0048_gate_sample_predictions", gate_predictions, gate_ids),
    ):
        rows.append(
            {
                "source_artifact": name,
                "rows": int(len(frame)),
                "selected_candidate_count": len(ids),
                "first_target_date": str(frame["target_date"].min().date()),
                "last_target_date": str(frame["target_date"].max().date()),
                "selected_candidate_ids": ";".join(ids),
            }
        )
    overlap = router_predictions[["target_date", "forecast_source_family"]].drop_duplicates().merge(
        gate_predictions[["target_date", "forecast_source_family"]].drop_duplicates(),
        on=["target_date", "forecast_source_family"],
        how="inner",
    )
    rows.append(
        {
            "source_artifact": "intersection_rows",
            "rows": int(len(overlap)),
            "selected_candidate_count": 0,
            "first_target_date": str(overlap["target_date"].min().date()),
            "last_target_date": str(overlap["target_date"].max().date()),
            "selected_candidate_ids": "",
        }
    )
    return pd.DataFrame(rows)


def baseline_comparison(scoreboard: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    base = predictions.drop_duplicates(["target_date", "forecast_source_family"]).copy()
    rows: list[dict[str, object]] = []
    for system, col in (
        ("official_raw", "official_raw"),
        ("anchor_0038", "anchor_0038_c"),
        ("best_0042_router_same_rows", "router_prediction_c"),
        ("best_0048_gate_same_rows", "gate_prediction_c"),
    ):
        score = score_prediction_frame(base.rename(columns={col: "prediction"}), "prediction")
        late_score = score_prediction_frame(late_frame(base).rename(columns={col: "prediction"}), "prediction")
        rows.append(
            {
                "system": system,
                **score,
                "late_eval_mae": late_score["mae"],
                "late_eval_rmse": late_score["rmse"],
            }
        )
    if not scoreboard.empty:
        best_late = scoreboard.iloc[0]
        best_full = scoreboard.sort_values(["mae", "late_eval_mae", "rmse"]).iloc[0]
        for system, row in (
            ("0049_best_late_router_gate_stack", best_late),
            ("0049_best_full_router_gate_stack", best_full),
        ):
            rows.append(
                {
                    "system": system,
                    "n": int(row["n"]),
                    "first_date": row["first_date"],
                    "last_date": row["last_date"],
                    "mae": float(row["mae"]),
                    "rmse": float(row["rmse"]),
                    "bias": float(row["bias"]),
                    "median_abs_error": float(row["median_abs_error"]),
                    "late_eval_mae": float(row["late_eval_mae"]),
                    "late_eval_rmse": float(row["late_eval_rmse"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["late_eval_mae", "mae", "rmse"]).reset_index(drop=True)


def selection_counts(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    return (
        predictions.groupby(["candidate_id", "selected_family"], observed=True, dropna=False)
        .agg(rows=("target_date", "count"), mean_gate_weight=("gate_weight", "mean"))
        .reset_index()
        .sort_values(["candidate_id", "rows"], ascending=[True, False])
    )


def write_readme(
    *,
    folder: Path,
    manifest: dict[str, object],
    coverage: pd.DataFrame,
    catalog: pd.DataFrame,
    comparison: pd.DataFrame,
    scoreboard: pd.DataFrame,
    counts: pd.DataFrame,
) -> None:
    best_late = scoreboard.iloc[0] if not scoreboard.empty else None
    best_full = scoreboard.sort_values(["mae", "late_eval_mae", "rmse"]).iloc[0] if not scoreboard.empty else None
    if best_late is None or best_full is None:
        result_text = "No scoreable 0049 stack candidate was produced."
    else:
        result_text = (
            f"Best actual late-window 0049 candidate: `{best_late['candidate_id']}` with late MAE "
            f"`{best_late['late_eval_mae']:.6f}`, late RMSE `{best_late['late_eval_rmse']:.6f}`, and late delta "
            f"vs its same-row 0042 router `{best_late['late_eval_delta_vs_router']:.6f}`. "
            f"Best full-window candidate: `{best_full['candidate_id']}` with full MAE `{best_full['mae']:.6f}`, "
            f"full RMSE `{best_full['rmse']:.6f}`, and full delta vs its same-row 0042 router "
            f"`{best_full['delta_vs_router']:.6f}`."
        )
    readme = f"""# 0049 Router Gate Stack Screen

Generated: `{manifest['generated_at_utc']}`

Script:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_router_gate_stack_screen.py
```

## Purpose

This experiment asks a narrow but important question: does the new long-history `0048` gated residual specialist add useful information on top of the current `0042` trust-router family, or is it mostly duplicating signal already captured by the existing router? The answer matters because the current best system is still far from the target performance level. Adding more components only helps if they improve MAE under a rule that could have been applied at the time of the forecast.

`0049` therefore does not start predictive modelling, does not train a black-box model, and does not touch 2024+ confirmation rows. It consumes already-audited row-level predictions from `0042` and `0048`, aligns them on the same `(target_date, forecast_source_family)` rows, and tests bounded combination rules.

## Leakage Contract

- All rows are required to be before `{CONFIRMATION_START.date()}`.
- Prior-performance modes only use scored rows with `target_date < current target_date`.
- Same-date rows cannot influence each other.
- Same-source variants restrict prior evidence to the same `forecast_source_family`.
- Fixed residual-addition variants use a hard-coded gate residual scale and no target labels for the current row.
- The candidate grid is selected from already documented 0042/0048 artifacts; this is a diagnostic stack screen, not a final promoted production model.

## Inputs

{markdown_table(coverage, max_rows=20)}

The selected `0042` rows are router predictions from the trust-router sensitivity experiment. The selected `0048` rows are gate residual candidates derived from long-history station/upper-air/weather regimes. `0049` uses their intersection because the current official scored archive is still non-contiguous.

## Candidate Design

The screen crosses `{manifest['router_candidate_count']}` router candidates with `{manifest['gate_candidate_count']}` gate candidates. For each pair it tests:

- `prior_best`: choose whichever of the router or gate had lower prior MAE.
- `prior_inverse_mae`: blend router and gate by inverse prior MAE.
- `prior_positive_lift`: use only families that beat the anchor on prior rows.
- `fixed_gate_residual`: add a small fixed fraction of the 0048 residual correction onto the 0042 router prediction.

The prior modes are run with all-prior and same-source evidence, and with history thresholds `{MIN_HISTORY_OPTIONS}`. The fixed residual modes use scales `{FIXED_GATE_SCALES}`.

## Main Result

{result_text}

## Baseline Comparison

{markdown_table(comparison, max_rows=20)}

## Top Scoreboard

{markdown_table(scoreboard.head(40), max_rows=40)}

## Selection Counts

{markdown_table(counts.head(60), max_rows=60)}

## Candidate Catalog

{markdown_table(catalog.head(80), max_rows=80)}

## Interpretation

This is a useful pressure test of complementarity. If the best `0049` candidates beat the same-row `0042` router by more than noise, then the long-history gated signal is adding something the forecast-history router does not already know. If `0049` only ties or worsens the router, then the long-history gated specialist should remain a diagnostic feature source rather than be added to the live stack.

The current result must be read with the same structural warning as the previous router experiments: the official scored forecast frame is non-contiguous. The local intersection covers the available press and RSS scored rows, not a seamless 2000-2023 daily archive. That means small MAE movements are useful research evidence, but not enough to claim production-grade superiority. The forecast archive refresh remains the highest-value blocker for proving robustness.

## Files

- `artifacts/input_coverage.csv`
- `artifacts/candidate_catalog.csv`
- `artifacts/scoreboard.csv`
- `artifacts/baseline_comparison.csv`
- `artifacts/selection_counts.csv`
- `artifacts/top_predictions.csv`
- `artifacts/summary.json`
"""
    write_text(folder / "README.md", readme)


def update_master_index(manifest: dict[str, object]) -> None:
    index_path = RESEARCH_ROOT / "README.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# HKG Tmax Data Analysis\n"
    marker = "\n## Router Gate Stack Screen\n"
    base = existing.split(marker)[0].rstrip()
    text = f"""{base}
{marker}
Generated: `{manifest['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_router_gate_stack_screen.py`:

- `{FOLDER_NAME}`: bounded prior-only combination screen testing whether `0048` long-history gated residual specialists add information on top of the `0042` trust-router family.

| Metric | Value |
|---|---:|
| Intersection rows | {manifest['intersection_rows']} |
| Stack candidates | {manifest['stack_candidate_count']} |
| Best late MAE | {manifest['best_late_mae']} |
| Best late delta vs router | {manifest['best_late_delta_vs_router']} |
| Best full MAE | {manifest['best_full_mae']} |
| Best full delta vs router | {manifest['best_full_delta_vs_router']} |

Leakage contract: all prior-performance choices use only earlier target dates; all scored rows are before `{CONFIRMATION_START.date()}`.
"""
    write_text(index_path, text)


def update_milestones(manifest: dict[str, object]) -> None:
    path = REPO_ROOT / "MILESTONES.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Milestones\n"
    section_marker = "\n## Router Gate Stack Screen\n"
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
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_t24_router_gate_stack_screen.py
```

New folder: `research/data_analysis/{FOLDER_NAME}`.

| Area | Evidence | Status |
|---|---|---|
| Rows | `{manifest['intersection_rows']}` aligned 0042/0048 rows | Non-contiguous official frame |
| Candidates | `{manifest['stack_candidate_count']}` bounded router/gate stack candidates | Diagnostic |
| Best late | MAE `{manifest['best_late_mae']}`, delta vs same-row 0042 router `{manifest['best_late_delta_vs_router']}` | Audited |
| Best full | MAE `{manifest['best_full_mae']}`, delta vs same-row 0042 router `{manifest['best_full_delta_vs_router']}` | Audited |
| Leakage guard | strict prior-only choices, same-source variants, zero 2024+ rows | Guarded |

Interpretation: `0049` checks whether long-history gated residual signals add to the current trust-router benchmark. It is useful as complementarity evidence, but not production proof while the scored forecast archive remains non-contiguous.
"""
    blocker = (
        f"33. Router/gate stack screening produced best full MAE `{manifest['best_full_mae']}` and best actual "
        f"late-window MAE `{manifest['best_late_mae']}` on `{manifest['stack_candidate_count']}` bounded candidates. "
        "This is still diagnostic because the official scored forecast archive is non-contiguous and 2024+ confirmation "
        "remains locked."
    )
    if blockers_marker in suffix and blocker not in suffix:
        before_next, after_next = suffix.split(next_marker, 1) if next_marker in suffix else (suffix, "")
        before_next = before_next.rstrip() + f"\n{blocker}\n"
        next_task = f"""{next_marker}

Run the `0045` bounded backfill smoke command outside the current socket-blocked sandbox, starting with 2008-01-01 through 2008-01-07 and `--limit 100`; then rerun `0044` to verify raw detail promotion before launching the full 2008-2026 backfill.
"""
        suffix = before_next + "\n" + next_task if after_next else before_next
    section += suffix
    write_text(path, section)


def write_outputs(
    *,
    router_predictions: pd.DataFrame,
    gate_predictions: pd.DataFrame,
    router_ids: list[str],
    gate_ids: list[str],
    catalog: pd.DataFrame,
    scoreboard: pd.DataFrame,
    predictions: pd.DataFrame,
) -> dict[str, object]:
    folder = RESEARCH_ROOT / FOLDER_NAME
    artifacts = folder / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    coverage = input_coverage(router_predictions, gate_predictions, router_ids, gate_ids)
    comparison = baseline_comparison(scoreboard, predictions)
    counts = selection_counts(predictions)
    top_ids = scoreboard.head(10)["candidate_id"].astype(str).to_list() if not scoreboard.empty else []
    top_predictions = predictions[predictions["candidate_id"].astype(str).isin(top_ids)].copy()
    write_csv(artifacts / "input_coverage.csv", coverage)
    write_csv(artifacts / "candidate_catalog.csv", catalog)
    write_csv(artifacts / "scoreboard.csv", scoreboard)
    write_csv(artifacts / "baseline_comparison.csv", comparison)
    write_csv(artifacts / "selection_counts.csv", counts)
    write_csv(artifacts / "top_predictions.csv", top_predictions)

    best_late = scoreboard.iloc[0] if not scoreboard.empty else None
    best_full = scoreboard.sort_values(["mae", "late_eval_mae", "rmse"]).iloc[0] if not scoreboard.empty else None
    base_rows = predictions.drop_duplicates(["target_date", "forecast_source_family"]) if not predictions.empty else predictions
    late = late_frame(base_rows)
    manifest = {
        "generated_at_utc": now_utc(),
        "folder": FOLDER_NAME,
        "screen_stage": SCREEN_STAGE,
        "intersection_rows": int(len(base_rows)),
        "first_target_date": "" if base_rows.empty else str(base_rows["target_date"].min().date()),
        "last_target_date": "" if base_rows.empty else str(base_rows["target_date"].max().date()),
        "late_eval_start": str(LATE_EVAL_START.date()),
        "late_eval_first_target_date": "" if late.empty else str(late["target_date"].min().date()),
        "late_eval_last_target_date": "" if late.empty else str(late["target_date"].max().date()),
        "late_eval_rows": int(len(late)),
        "router_candidate_count": len(router_ids),
        "gate_candidate_count": len(gate_ids),
        "stack_candidate_count": int(len(scoreboard)),
        "best_late_candidate": "" if best_late is None else str(best_late["candidate_id"]),
        "best_late_mae": None if best_late is None else float(best_late["late_eval_mae"]),
        "best_late_rmse": None if best_late is None else float(best_late["late_eval_rmse"]),
        "best_late_delta_vs_router": None if best_late is None else float(best_late["late_eval_delta_vs_router"]),
        "best_full_candidate": "" if best_full is None else str(best_full["candidate_id"]),
        "best_full_mae": None if best_full is None else float(best_full["mae"]),
        "best_full_rmse": None if best_full is None else float(best_full["rmse"]),
        "best_full_delta_vs_router": None if best_full is None else float(best_full["delta_vs_router"]),
        "uses_2024_plus_rows": False,
    }
    write_json(artifacts / "summary.json", manifest)
    write_json(RESEARCH_ROOT / "router_gate_stack_screen_manifest.json", manifest)
    write_readme(
        folder=folder,
        manifest=manifest,
        coverage=coverage,
        catalog=catalog,
        comparison=comparison,
        scoreboard=scoreboard,
        counts=counts,
    )
    update_master_index(manifest)
    update_milestones(manifest)
    return manifest


def run() -> dict[str, object]:
    router_predictions, _router_scoreboard, router_ids = load_router_predictions()
    gate_predictions, _gate_scoreboard, gate_ids = load_gate_predictions()
    catalog = build_candidate_catalog(router_ids, gate_ids)
    scoreboard, predictions = run_stack_screen(router_predictions, gate_predictions, catalog)
    return write_outputs(
        router_predictions=router_predictions,
        gate_predictions=gate_predictions,
        router_ids=router_ids,
        gate_ids=gate_ids,
        catalog=catalog,
        scoreboard=scoreboard,
        predictions=predictions,
    )


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run HKG T24 router/gate prior-only stack screen.").parse_args()


def main() -> None:
    parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
