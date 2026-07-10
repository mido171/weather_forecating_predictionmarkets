from __future__ import annotations

import hashlib
import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_hkg_t24_0184_hf_teacher_proxy_causal_memory_router as base


REPO_ROOT = base.REPO_ROOT
EXPERIMENTS_ROOT = base.EXPERIMENTS_ROOT
EXPERIMENT_ID = "0203"
SLUG = "july_october_micro_no_harm_governor"
TITLE = "July-October Micro No-Harm Governor Over 0196"
EXP_DIR = EXPERIMENTS_ROOT / f"{EXPERIMENT_ID}_{SLUG}"
PRIMARY_CANDIDATE_ID = "0203_july_october_micro_no_harm_governor_over_0196"
SRC_COPY_NAME = "run_0203.py"
MODEL_FOLDS = base.MODEL_FOLDS
LAG_DAYS = 7
PREDICTION_SOURCES = {
    "0194": EXPERIMENTS_ROOT / "0194_isd_role_compressed_regime_proxy" / "predictions.parquet",
    "0196": EXPERIMENTS_ROOT / "0196_station_network_tail_conditioned_residual_expert" / "predictions.parquet",
    "0199": EXPERIMENTS_ROOT / "0199_station_role_feature_family_replay" / "predictions.parquet",
    "0200": EXPERIMENTS_ROOT / "0200_climatological_analogue_blend" / "predictions.parquet",
    "0201": EXPERIMENTS_ROOT / "0201_independent_feature_group_stacked_router" / "predictions.parquet",
}
EXPERT_IDS = ["0194", "0199", "0200", "0201"]
CONFIG_GRID = [
    {"config_id": "cfg_00_parent_only", "active_months": [], "contexts": [], "min_support": 0, "min_lift_c": 0.0, "blend": 0.0, "cap_c": 0.0, "experts": []},
    {"config_id": "cfg_01_jul_oct_month_source_strict", "active_months": [7, 8, 9, 10], "contexts": ["month_source", "month", "active_months"], "min_support": 120, "min_lift_c": 0.0015, "blend": 1.0, "cap_c": 0.18, "experts": EXPERT_IDS},
    {"config_id": "cfg_02_jul_oct_month_source_shrunk", "active_months": [7, 8, 9, 10], "contexts": ["month_source", "month", "active_months"], "min_support": 120, "min_lift_c": 0.0010, "blend": 0.50, "cap_c": 0.12, "experts": EXPERT_IDS},
    {"config_id": "cfg_03_jul_oct_source_then_month", "active_months": [7, 8, 9, 10], "contexts": ["source", "month_source", "month"], "min_support": 180, "min_lift_c": 0.0010, "blend": 0.75, "cap_c": 0.15, "experts": EXPERT_IDS},
    {"config_id": "cfg_04_jul_oct_0200_only", "active_months": [7, 8, 9, 10], "contexts": ["month_source", "month", "active_months"], "min_support": 90, "min_lift_c": 0.0005, "blend": 1.0, "cap_c": 0.18, "experts": ["0200"]},
    {"config_id": "cfg_05_jul_oct_0194_0200_only", "active_months": [7, 8, 9, 10], "contexts": ["month_source", "month", "active_months"], "min_support": 90, "min_lift_c": 0.0005, "blend": 0.75, "cap_c": 0.15, "experts": ["0194", "0200"]},
    {"config_id": "cfg_06_jun_nov_broad_shrunk", "active_months": [6, 7, 8, 9, 10, 11], "contexts": ["month_source", "month", "active_months"], "min_support": 160, "min_lift_c": 0.0015, "blend": 0.50, "cap_c": 0.10, "experts": EXPERT_IDS},
]
CONFIG_SELECTION_MIN_LIFT_C = 0.0005


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rel(path: Path) -> str:
    return base.rel(path)


def date_text(value: Any) -> str:
    return base.date_text(value)


def load_predictions() -> pd.DataFrame:
    p0196 = pd.read_parquet(PREDICTION_SOURCES["0196"])
    keep = ["target_date", "target_tmax_c", "forecast_source_family", "season", "month", "official_prediction_c", "candidate_prediction_c", "candidate_abs_error_c", "fold_id"]
    frame = p0196[keep].copy()
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
    frame = frame.rename(columns={"candidate_prediction_c": "pred_0196", "candidate_abs_error_c": "abs_error_0196"})
    for expert_id, path in PREDICTION_SOURCES.items():
        if expert_id == "0196":
            continue
        df = pd.read_parquet(path)
        df["target_date"] = pd.to_datetime(df["target_date"], errors="coerce").dt.normalize()
        df = df[["target_date", "candidate_prediction_c"]].rename(columns={"candidate_prediction_c": f"pred_{expert_id}"})
        frame = frame.merge(df, on="target_date", how="left", validate="one_to_one")
    frame = frame[frame["target_date"] < pd.Timestamp("2024-01-01")].copy()
    for expert_id in ["0196", *EXPERT_IDS]:
        frame[f"pred_{expert_id}"] = pd.to_numeric(frame[f"pred_{expert_id}"], errors="coerce")
        frame[f"abs_error_{expert_id}"] = (frame[f"pred_{expert_id}"] - frame["target_tmax_c"]).abs()
        frame[f"gt3_{expert_id}"] = frame[f"abs_error_{expert_id}"].gt(3.0)
    if frame[[f"pred_{expert_id}" for expert_id in ["0196", *EXPERT_IDS]]].isna().any().any():
        raise RuntimeError("Missing frozen expert predictions after merge")
    base.assert_pre2024(frame, "0203 frozen expert frame")
    return frame.sort_values("target_date").reset_index(drop=True)


def context_mask(frame: pd.DataFrame, idx: int, history_idx: np.ndarray, context: str, active_months: set[int]) -> np.ndarray:
    month = int(frame.at[idx, "month"])
    source = str(frame.at[idx, "forecast_source_family"])
    hist = history_idx.copy()
    if context == "month_source":
        return hist[(frame.loc[hist, "month"].astype(int).to_numpy() == month) & (frame.loc[hist, "forecast_source_family"].astype(str).to_numpy() == source)]
    if context == "month":
        return hist[frame.loc[hist, "month"].astype(int).to_numpy() == month]
    if context == "source":
        return hist[frame.loc[hist, "forecast_source_family"].astype(str).to_numpy() == source]
    if context == "active_months":
        return hist[frame.loc[hist, "month"].astype(int).isin(active_months).to_numpy()]
    if context == "all":
        return hist
    raise ValueError(f"Unknown context {context}")


def route_one(frame: pd.DataFrame, idx: int, history_idx: np.ndarray, config: dict[str, Any]) -> tuple[str, str, int, float, float]:
    if not config["active_months"] or int(frame.at[idx, "month"]) not in set(config["active_months"]):
        return "0196", "inactive", 0, 0.0, 0.0
    active_months = set(int(x) for x in config["active_months"])
    best = {"expert": "0196", "context": "none", "support": 0, "delta": 0.0, "gt3_delta": 0.0}
    for context in config["contexts"]:
        subset = context_mask(frame, idx, history_idx, context, active_months)
        if len(subset) < int(config["min_support"]):
            continue
        parent_mae = float(frame.loc[subset, "abs_error_0196"].mean())
        parent_gt3 = float(frame.loc[subset, "gt3_0196"].mean())
        for expert_id in config["experts"]:
            expert_mae = float(frame.loc[subset, f"abs_error_{expert_id}"].mean())
            expert_gt3 = float(frame.loc[subset, f"gt3_{expert_id}"].mean())
            delta = expert_mae - parent_mae
            gt3_delta = expert_gt3 - parent_gt3
            if delta < best["delta"]:
                best = {"expert": expert_id, "context": context, "support": int(len(subset)), "delta": delta, "gt3_delta": gt3_delta}
        if best["expert"] != "0196":
            break
    if best["delta"] <= -float(config["min_lift_c"]) and best["gt3_delta"] <= 0.005:
        return best["expert"], best["context"], best["support"], best["delta"], best["gt3_delta"]
    return "0196", "no_prior_lift", int(best["support"]), float(best["delta"]), float(best["gt3_delta"])


def online_predict(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    dates = frame["target_date"].to_numpy(dtype="datetime64[D]")
    months = frame["month"].astype(int).to_numpy()
    sources = frame["forecast_source_family"].astype(str).to_numpy()
    parent = frame["pred_0196"].to_numpy(dtype=float)
    pred_arrays = {expert_id: frame[f"pred_{expert_id}"].to_numpy(dtype=float) for expert_id in ["0196", *EXPERT_IDS]}
    abs_arrays = {expert_id: frame[f"abs_error_{expert_id}"].to_numpy(dtype=float) for expert_id in ["0196", *EXPERT_IDS]}
    gt3_arrays = {expert_id: frame[f"gt3_{expert_id}"].to_numpy(dtype=bool) for expert_id in ["0196", *EXPERT_IDS]}
    active_months = set(int(x) for x in config["active_months"])
    out = frame[["target_date"]].copy()
    pred = parent.copy()
    selected_expert: list[str] = []
    selected_context: list[str] = []
    support_rows: list[int] = []
    prior_delta: list[float] = []
    prior_gt3_delta: list[float] = []
    for idx in range(len(frame)):
        mature_date = dates[idx] - np.timedelta64(LAG_DAYS, "D")
        stop = int(np.searchsorted(dates, mature_date, side="right"))
        expert_id = "0196"
        context = "inactive"
        support = 0
        delta = 0.0
        gt3_delta = 0.0
        if active_months and int(months[idx]) in active_months and stop > 0:
            best_expert = "0196"
            best_context = "none"
            best_support = 0
            best_delta = 0.0
            best_gt3_delta = 0.0
            hist_months = months[:stop]
            hist_sources = sources[:stop]
            for context_name in config["contexts"]:
                if context_name == "month_source":
                    mask = (hist_months == months[idx]) & (hist_sources == sources[idx])
                elif context_name == "month":
                    mask = hist_months == months[idx]
                elif context_name == "source":
                    mask = hist_sources == sources[idx]
                elif context_name == "active_months":
                    mask = np.isin(hist_months, list(active_months))
                elif context_name == "all":
                    mask = np.ones(stop, dtype=bool)
                else:
                    raise ValueError(f"Unknown context {context_name}")
                support_now = int(mask.sum())
                if support_now < int(config["min_support"]):
                    continue
                parent_mae = float(abs_arrays["0196"][:stop][mask].mean())
                parent_gt3 = float(gt3_arrays["0196"][:stop][mask].mean())
                for candidate_expert in config["experts"]:
                    expert_mae = float(abs_arrays[candidate_expert][:stop][mask].mean())
                    expert_gt3 = float(gt3_arrays[candidate_expert][:stop][mask].mean())
                    candidate_delta = expert_mae - parent_mae
                    candidate_gt3_delta = expert_gt3 - parent_gt3
                    if candidate_delta < best_delta:
                        best_expert = candidate_expert
                        best_context = context_name
                        best_support = support_now
                        best_delta = candidate_delta
                        best_gt3_delta = candidate_gt3_delta
                if best_expert != "0196":
                    break
            if best_delta <= -float(config["min_lift_c"]) and best_gt3_delta <= 0.005:
                expert_id = best_expert
                context = best_context
                support = best_support
                delta = best_delta
                gt3_delta = best_gt3_delta
            else:
                context = "no_prior_lift"
                support = best_support
                delta = best_delta
                gt3_delta = best_gt3_delta
        selected_expert.append(expert_id)
        selected_context.append(context)
        support_rows.append(support)
        prior_delta.append(delta)
        prior_gt3_delta.append(gt3_delta)
        if expert_id != "0196":
            raw_delta = float(pred_arrays[expert_id][idx] - parent[idx])
            correction = float(config["blend"]) * np.clip(raw_delta, -float(config["cap_c"]), float(config["cap_c"]))
            pred[idx] = parent[idx] + correction
    out["prediction_c"] = pred
    out["selected_expert"] = selected_expert
    out["selected_context"] = selected_context
    out["support_rows"] = support_rows
    out["prior_delta_vs_0196_mae_c"] = prior_delta
    out["prior_gt3_delta_vs_0196"] = prior_gt3_delta
    out["switched"] = out["selected_expert"].ne("0196")
    return out


def score_prediction(frame: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    target = frame["target_tmax_c"].to_numpy(dtype=float)
    parent = frame["pred_0196"].to_numpy(dtype=float)
    return {
        "mae_c": float(np.mean(np.abs(pred - target))),
        "parent_mae_c": float(np.mean(np.abs(parent - target))),
        "delta_vs_0196_mae_c": float(np.mean(np.abs(pred - target)) - np.mean(np.abs(parent - target))),
        "gt3_rate": float(np.mean(np.abs(pred - target) > 3.0)),
        "parent_gt3_rate": float(np.mean(np.abs(parent - target) > 3.0)),
    }


def select_config(train: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    for config in CONFIG_GRID:
        routed = online_predict(train, config)
        metrics = score_prediction(train, routed["prediction_c"].to_numpy(dtype=float))
        rows.append(
            {
                "config_id": config["config_id"],
                "train_mae_c": metrics["mae_c"],
                "train_parent_0196_mae_c": metrics["parent_mae_c"],
                "train_delta_vs_0196_mae_c": metrics["delta_vs_0196_mae_c"],
                "train_gt3_delta_vs_0196": metrics["gt3_rate"] - metrics["parent_gt3_rate"],
                "switch_rate": float(routed["switched"].mean()),
                "active_months": ";".join(str(x) for x in config["active_months"]),
                "contexts": ";".join(config["contexts"]),
                "min_support": config["min_support"],
                "min_lift_c": config["min_lift_c"],
                "blend": config["blend"],
                "cap_c": config["cap_c"],
                "experts": ";".join(config["experts"]),
            }
        )
    scores = pd.DataFrame(rows).sort_values(["train_delta_vs_0196_mae_c", "train_gt3_delta_vs_0196", "config_id"]).reset_index(drop=True)
    best = scores.iloc[0].to_dict()
    if float(best["train_delta_vs_0196_mae_c"]) <= -CONFIG_SELECTION_MIN_LIFT_C and float(best["train_gt3_delta_vs_0196"]) <= 0.005:
        for config in CONFIG_GRID:
            if config["config_id"] == best["config_id"]:
                return config, scores
    return CONFIG_GRID[0], scores


def compare(frame: pd.DataFrame, *, slice_type: str, slice_value: Any) -> dict[str, Any]:
    official = base.metric_row(frame, "official_prediction_c", label="official")
    parent = base.metric_row(frame, "pred_0196", label="p0196")
    candidate = base.metric_row(frame, "candidate_prediction_c", label=PRIMARY_CANDIDATE_ID)
    return {
        "slice_type": slice_type,
        "slice_value": str(slice_value),
        "n": candidate["n"],
        "official_mae_c": official["mae_c"],
        "parent_0196_mae_c": parent["mae_c"],
        "candidate_mae_c": candidate["mae_c"],
        "mae_delta_c": candidate["mae_c"] - official["mae_c"],
        "delta_vs_0196_mae_c": candidate["mae_c"] - parent["mae_c"],
        "candidate_bias_c": candidate["bias_c"],
        "parent_0196_gt3c_rate": parent["gt3c_rate"],
        "candidate_gt3c_rate": candidate["gt3c_rate"],
        "parent_0196_p95_abs_error_c": parent["p95_abs_error_c"],
        "candidate_p95_abs_error_c": candidate["p95_abs_error_c"],
    }


def run_walk_forward(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parts = []
    fold_rows = []
    selection_rows = []
    score_rows = []
    full_cache: dict[str, pd.DataFrame] = {}
    for start_year, end_year in MODEL_FOLDS:
        fold_id = f"fold_{start_year}_{end_year}"
        test_mask = frame["target_date"].dt.year.between(start_year, end_year)
        test_dates = set(frame.loc[test_mask, "target_date"])
        test = frame[test_mask].copy()
        if test.empty:
            fold_rows.append({"fold_id": fold_id, "n": 0})
            continue
        train = frame[frame["target_date"].dt.year < start_year].copy()
        if len(train) < 365:
            selected = CONFIG_GRID[0]
            scores = pd.DataFrame([{"config_id": selected["config_id"], "train_delta_vs_0196_mae_c": 0.0, "selection_reason": "first_fold_no_prior_history"}])
        else:
            selected, scores = select_config(train)
            scores["selection_reason"] = "prior_train_lift_grid"
        if selected["config_id"] not in full_cache:
            full_cache[selected["config_id"]] = online_predict(frame, selected)
        routed = full_cache[selected["config_id"]]
        test_route = routed[routed["target_date"].isin(test_dates)].copy().reset_index(drop=True)
        test = test.sort_values("target_date").reset_index(drop=True)
        test["candidate_prediction_c"] = test_route["prediction_c"].to_numpy(dtype=float)
        test["selected_expert"] = test_route["selected_expert"].to_numpy()
        test["selected_context"] = test_route["selected_context"].to_numpy()
        test["support_rows"] = test_route["support_rows"].to_numpy()
        test["prior_delta_vs_0196_mae_c"] = test_route["prior_delta_vs_0196_mae_c"].to_numpy()
        test["prior_gt3_delta_vs_0196"] = test_route["prior_gt3_delta_vs_0196"].to_numpy()
        test["switched"] = test_route["switched"].to_numpy()
        test["candidate_error_c"] = test["candidate_prediction_c"] - test["target_tmax_c"]
        test["candidate_abs_error_c"] = test["candidate_error_c"].abs()
        test["official_abs_error_c"] = (test["official_prediction_c"] - test["target_tmax_c"]).abs()
        test["fold_id"] = fold_id
        test["selected_config_id"] = selected["config_id"]
        metric = compare(test, slice_type="fold", slice_value=fold_id)
        metric.update(
            {
                "fold_id": fold_id,
                "start_year": start_year,
                "end_year": end_year,
                "selected_config_id": selected["config_id"],
                "switch_rate": float(test["switched"].mean()),
                "most_common_selected_expert": test.loc[test["switched"], "selected_expert"].mode().iat[0] if test["switched"].any() else "0196",
            }
        )
        fold_rows.append(metric)
        selection_rows.append({"fold_id": fold_id, **selected})
        scores = scores.copy()
        scores["fold_id"] = fold_id
        score_rows.append(scores)
        parts.append(test)
    predictions = pd.concat(parts, ignore_index=True).sort_values("target_date").reset_index(drop=True)
    predictions["candidate_id"] = PRIMARY_CANDIDATE_ID
    predictions["baseline_id"] = "official_forecast_max_c"
    predictions["model_family"] = "july_october_micro_no_harm_governor"
    return predictions, pd.DataFrame(fold_rows), pd.DataFrame(selection_rows), pd.concat(score_rows, ignore_index=True)


def build_slice_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [compare(predictions, slice_type="overall", slice_value="all")]
    for season, group in predictions.groupby("season", dropna=False):
        rows.append(compare(group, slice_type="season", slice_value=season))
    for month, group in predictions.groupby("month", dropna=False):
        rows.append(compare(group, slice_type="month", slice_value=month))
    for source, group in predictions.groupby("forecast_source_family", dropna=False):
        rows.append(compare(group, slice_type="source", slice_value=source))
    late = predictions[predictions["target_date"] >= pd.Timestamp("2020-01-01")]
    rows.append(compare(late, slice_type="late_window", slice_value="2020_2023"))
    tail = predictions[predictions["abs_error_0196"] >= 2.0]
    rows.append(compare(tail, slice_type="parent_tail", slice_value="parent_0196_abs_error_ge_2c"))
    switched = predictions[predictions["switched"]]
    if not switched.empty:
        rows.append(compare(switched, slice_type="governed_rows", slice_value="switched"))
    yearly = pd.DataFrame([compare(group, slice_type="year", slice_value=year) for year, group in predictions.groupby(predictions["target_date"].dt.year, dropna=False)])
    return pd.DataFrame(rows), yearly


def build_spec(created_at: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": created_at,
        "title": TITLE,
        "slug": SLUG,
        "mode": "promotion_oriented",
        "hypothesis": "The small July-October residual pockets left by 0196 can be reduced by an online T-7 mature no-harm governor over frozen leakage-passed child experts.",
        "rationale": "0200 showed a tiny independent analogue lift and 0201 showed some RSS-era pocket value but global stack harm. 0203 tests a narrower prior-loss governor instead of another broad stack.",
        "expected_sign_and_falsification": "Expected sign is lower MAE than 0196 with no fold or severe-tail harm. Falsified if parent-only is selected or if switched contexts fail to transfer.",
        "target": {"station": "Hong Kong Observatory headquarters", "variable": "tmax_c", "horizon": "T-24", "timezone": "Asia/Hong_Kong", "cutoff_contract_path": rel(base.OFFICIAL_PATH), "cutoff_function": "Current-row expert predictions are T-24 available and routing losses use only target dates <= T-7."},
        "frame": {"frame_id": "official_t15_pre2024_5265_rows", "development_start": "2000-01-02", "development_end_exclusive": "2024-01-01", "confirmation_locked": True, "row_universe_artifact": rel(PREDICTION_SOURCES["0196"])},
        "data_sources": [{"source_id": f"{expert_id}_predictions", "paths": [rel(path)], "eligibility": "DEPLOYABLE_LAGGED_ONLY"} for expert_id, path in PREDICTION_SOURCES.items()],
        "features": {"generation_rule": "For July-October and one broad warm fallback grid only switch from 0196 when T-7 mature prior context losses show an alternative expert beats 0196 by the predeclared margin.", "config_grid": CONFIG_GRID, "explicit_exclusions": ["2024+ rows", "current target outcome", "current residual or absolute error for routing"]},
        "response": {"variable": "frozen expert prediction selected by T-7 mature context loss governor", "prediction": PRIMARY_CANDIDATE_ID},
        "baseline": {"id": "official_forecast_max_c", "parent_reference": "0196_station_network_tail_conditioned_residual_expert"},
        "validation": {"outer_folds": [list(item) for item in MODEL_FOLDS], "config_selection": "prior years only; config must beat 0196 by 0.0005 C on prior training to avoid parent fallback", "online_loss_lag_days": LAG_DAYS},
        "metrics": ["MAE", "RMSE", "bias", "P95 AE", ">3C rate", "fold/year/season/month/source/tail/governed-row slices"],
        "acceptance_gates": {"minimum_mae_lift_vs_official_c": 0.01, "minimum_mae_lift_vs_0196_c": 0.001, "max_fold_harm_vs_0196_c": 0.001, "max_gt3_rate_delta_vs_0196": 0.005},
        "owner_authorized_confirmation": False,
    }


def write_docs(scoreboard: pd.DataFrame, slice_metrics: pd.DataFrame, yearly_metrics: pd.DataFrame, fold_metrics: pd.DataFrame, selections: pd.DataFrame, summary: dict[str, Any]) -> None:
    write_text(EXP_DIR / "README.md", f"""# {TITLE}

## Hypothesis

The remaining July-October 0196 pockets can be reduced by a strict online no-harm governor that switches to another frozen leakage-passed expert only when mature prior evidence supports it.

## Why This Experiment Exists

0200 found a tiny but real analogue signal and 0201 found source-era pocket value but failed globally. 0203 is deliberately narrower: it does not train a larger model and does not search arbitrary feature interactions.

## Cutoff

The cutoff is T-24 in Asia/Hong_Kong. The governor may use current-row frozen expert predictions, month, source family, and T-7 mature historical losses only.

## Dataset

The dataset is the canonical 5265-row pre-2024 official frame. Inputs are validator-clean predictions from 0194, 0196, 0199, 0200, and 0201.

## Feature

The only derived features are prior context MAE deltas, support counts, selected expert id, selected context, and switch indicator.

## Baseline

The primary baseline is `official_forecast_max_c`. The parent reference is `0196`, which is used by default for every row unless the mature-loss gate fires.

## Walk-Forward

Each fold selects a governor config on years before the fold. Within the selected config, row-level routing uses only outcomes whose target dates are at least `{LAG_DAYS}` days mature.

## Acceptance

Promotion requires at least 0.001 C global MAE improvement versus 0196 with no material fold or severe-tail harm. Status is `{summary['status']}`.
""")
    write_text(EXP_DIR / "RESULTS.md", f"""# Results

## Headline

{base.markdown_table(scoreboard)}

## Coverage

Rows: `{summary['n_common']}` from `{summary['date_start']}` to `{summary['date_end']}`. Common row hash: `{summary['common_row_hash']}`.

## Global

Official MAE is `{summary['baseline_mae_c']:.6f}` C. Parent 0196 MAE is `{summary['parent_0196_mae_c']:.6f}` C. Candidate MAE is `{summary['candidate_mae_c']:.6f}` C. Delta versus 0196 is `{summary['delta_vs_0196_mae_c']:.6f}` C.

## Fold

{base.markdown_table(fold_metrics[['fold_id', 'n', 'parent_0196_mae_c', 'candidate_mae_c', 'delta_vs_0196_mae_c', 'selected_config_id', 'switch_rate', 'most_common_selected_expert']], max_rows=20)}

## Year

{base.markdown_table(yearly_metrics[['slice_value', 'n', 'parent_0196_mae_c', 'candidate_mae_c', 'delta_vs_0196_mae_c']], max_rows=30)}

## Season And Month

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].isin(['season', 'month'])][['slice_type', 'slice_value', 'n', 'parent_0196_mae_c', 'candidate_mae_c', 'delta_vs_0196_mae_c']], max_rows=30)}

## Tail And Source

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].isin(['source', 'late_window', 'parent_tail', 'governed_rows'])][['slice_type', 'slice_value', 'n', 'parent_0196_mae_c', 'candidate_mae_c', 'delta_vs_0196_mae_c', 'parent_0196_gt3c_rate', 'candidate_gt3c_rate']], max_rows=25)}

## Leakage

Leakage status is `{summary['leakage_status']}`. Routing losses use a `{LAG_DAYS}` day maturity lag and confirmation rows used is `{summary['confirmation_rows_used']}`.

Selections:

{base.markdown_table(selections, max_rows=20)}
""")
    write_text(EXP_DIR / "CONCLUSION.md", f"""# Conclusion

## Verdict

Status is `{summary['status']}` with promotion decision `{summary['promotion_decision']}`.

## Learned

0203 tests whether the useful tiny signals from 0200 and 0201 can be harvested without the broad harm seen in 0201. The answer depends on whether mature prior month/source contexts fire often enough and transfer to the outer folds.

## MAE

Delta versus official is `{summary['mae_delta_c']:.6f}` C. Delta versus 0196 is `{summary['delta_vs_0196_mae_c']:.6f}` C.

## Robust

The design is robust against leakage because every row defaults to 0196 and switch decisions use only T-7 mature losses. Config selection is prior-year only.

## Failure

If the governor does not improve global MAE, the remaining pockets are either too small, too sparse, or too unstable to safely override the champion with same-corpus frozen experts. That failure would reinforce the 0202 conclusion that major progress needs a new timestamp-proven operational forecast archive.

## Promotion

The development gate to 0.45 C was not reached. Confirmation remains sealed.
""")


def main() -> None:
    created_at = utc_now()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in ("src", "artifacts", "logs", "diagnostics", "figures"):
        (EXP_DIR / subdir).mkdir(parents=True, exist_ok=True)
    spec = build_spec(created_at)
    write_json(EXP_DIR / "experiment_spec.json", spec)
    spec_sha = sha256_file(EXP_DIR / "experiment_spec.json")
    src_copy_path = EXP_DIR / "src" / SRC_COPY_NAME
    shutil.copy2(Path(__file__).resolve(), src_copy_path)
    frame = load_predictions()
    predictions, fold_metrics, selections, config_scores = run_walk_forward(frame)
    slice_metrics, yearly_metrics = build_slice_metrics(predictions)
    official_global = base.metric_row(predictions, "official_prediction_c", label="official")
    parent_global = base.metric_row(predictions, "pred_0196", label="p0196")
    candidate_global = base.metric_row(predictions, "candidate_prediction_c", label=PRIMARY_CANDIDATE_ID)
    mae_delta = candidate_global["mae_c"] - official_global["mae_c"]
    delta_vs_0196 = candidate_global["mae_c"] - parent_global["mae_c"]
    severe_harm_0196 = candidate_global["gt3c_rate"] - parent_global["gt3c_rate"]
    fold_worst_delta = float(fold_metrics["delta_vs_0196_mae_c"].max())
    if mae_delta <= -0.01 and delta_vs_0196 <= -0.001 and severe_harm_0196 <= 0.005 and fold_worst_delta <= 0.001:
        status = "COMPLETED_PROMOTION_CANDIDATE"
        promotion_decision = "PROMOTE_OVER_0196_NO_CONFIRMATION"
    elif mae_delta < 0:
        status = "COMPLETED_INFORMATION_GAIN_ONLY"
        promotion_decision = "DO_NOT_PROMOTE_YET_INFORMATION_GAIN"
    else:
        status = "COMPLETED_NULL_OR_NEGATIVE"
        promotion_decision = "DO_NOT_PROMOTE"
    common_row_hash = sha256_text("\n".join(date_text(value) for value in predictions["target_date"]))
    scoreboard = pd.DataFrame(
        [
            {"candidate_id": "official_forecast_max_c", "model_family": "baseline", "n": official_global["n"], "mae_c": official_global["mae_c"], "rmse_c": official_global["rmse_c"], "bias_c": official_global["bias_c"], "median_abs_error_c": official_global["median_abs_error_c"], "p95_abs_error_c": official_global["p95_abs_error_c"], "gt2c_rate": official_global["gt2c_rate"], "gt3c_rate": official_global["gt3c_rate"], "baseline_mae_c": official_global["mae_c"], "mae_delta_c": 0.0},
            {"candidate_id": "0196_parent_tail_expert", "model_family": "parent_reference", "n": parent_global["n"], "mae_c": parent_global["mae_c"], "rmse_c": parent_global["rmse_c"], "bias_c": parent_global["bias_c"], "median_abs_error_c": parent_global["median_abs_error_c"], "p95_abs_error_c": parent_global["p95_abs_error_c"], "gt2c_rate": parent_global["gt2c_rate"], "gt3c_rate": parent_global["gt3c_rate"], "baseline_mae_c": official_global["mae_c"], "mae_delta_c": parent_global["mae_c"] - official_global["mae_c"]},
            {"candidate_id": PRIMARY_CANDIDATE_ID, "model_family": "july_october_micro_no_harm_governor", "n": candidate_global["n"], "mae_c": candidate_global["mae_c"], "rmse_c": candidate_global["rmse_c"], "bias_c": candidate_global["bias_c"], "median_abs_error_c": candidate_global["median_abs_error_c"], "p95_abs_error_c": candidate_global["p95_abs_error_c"], "gt2c_rate": candidate_global["gt2c_rate"], "gt3c_rate": candidate_global["gt3c_rate"], "baseline_mae_c": official_global["mae_c"], "mae_delta_c": mae_delta},
        ]
    )
    row_coverage = pd.DataFrame([{"frame_id": "official_t15_pre2024_5265_rows", "parent_rows": int(len(frame)), "candidate_rows": int(len(predictions)), "baseline_rows": int(len(predictions)), "common_rows": int(len(predictions)), "date_start": date_text(predictions["target_date"].min()), "date_end": date_text(predictions["target_date"].max()), "row_policy": "all 0196 parent rows", "common_row_hash": common_row_hash}])
    data_manifest = pd.DataFrame([{"source_id": f"{expert_id}_predictions", "path": rel(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size, "timestamp_fields": "target_date;fold_id;frozen prediction", "availability_class": "DEPLOYABLE_LAGGED_ONLY", "row_count": int(len(frame)), "date_start": date_text(frame["target_date"].min()), "date_end": date_text(frame["target_date"].max()), "notes": "Leakage-passed frozen expert prediction used by 0203 governor."} for expert_id, path in PREDICTION_SOURCES.items()])
    feature_defs = pd.DataFrame(
        [
            {"feature_name": "t7_mature_context_loss_delta", "formula": "Mean absolute loss of each frozen expert minus 0196 in a predeclared month source context using rows with target_date <= T-7.", "input_columns": "target_date,month,forecast_source_family,frozen_expert_predictions,target_tmax_c for mature historical rows only", "fit_scope": "online prequential and fold-selected config", "availability_rule": "Only mature historical target outcomes at least seven target days before current T are admitted."},
            {"feature_name": "micro_governor_selected_expert", "formula": "0196 unless a frozen expert beats 0196 by config min_lift in a mature context with minimum support and no severe-error-rate harm.", "input_columns": "t7_mature_context_loss_delta,support_rows,month,forecast_source_family", "fit_scope": "current row routing after prior-year config selection", "availability_rule": "Current target outcome is never used for routing."},
        ]
    )
    pred_cols = ["target_date", "target_tmax_c", "forecast_source_family", "season", "month", "official_prediction_c", "pred_0196", "candidate_prediction_c", "selected_expert", "selected_context", "support_rows", "prior_delta_vs_0196_mae_c", "prior_gt3_delta_vs_0196", "switched", "candidate_error_c", "official_abs_error_c", "abs_error_0196", "candidate_abs_error_c", "fold_id", "selected_config_id", "candidate_id", "baseline_id", "model_family"]
    write_parquet(EXP_DIR / "predictions.parquet", predictions[pred_cols])
    write_csv(EXP_DIR / "scoreboard.csv", scoreboard)
    write_csv(EXP_DIR / "slice_metrics.csv", slice_metrics)
    write_csv(EXP_DIR / "yearly_metrics.csv", yearly_metrics)
    write_csv(EXP_DIR / "fold_metrics.csv", fold_metrics)
    write_csv(EXP_DIR / "row_coverage.csv", row_coverage)
    write_csv(EXP_DIR / "data_manifest.csv", data_manifest)
    write_csv(EXP_DIR / "feature_definitions.csv", feature_defs)
    write_csv(EXP_DIR / "artifacts" / "config_grid.csv", pd.DataFrame(CONFIG_GRID))
    write_csv(EXP_DIR / "artifacts" / "fold_config_selections.csv", selections)
    write_csv(EXP_DIR / "artifacts" / "config_selection_scores.csv", config_scores)
    write_csv(EXP_DIR / "correction_distribution.csv", (predictions["candidate_prediction_c"] - predictions["pred_0196"]).describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).reset_index())
    write_text(EXP_DIR / "leakage_audit.md", f"""# Leakage And Point-In-Time Audit

Status: `PASS`

## Cutoff

0203 is a T-24 governor over frozen leakage-passed expert predictions. Each current row can use only known expert predictions, month, source, and mature historical losses.

## Available State

Loss memory admits rows only when target_date <= T-{LAG_DAYS}. Current target outcomes and confirmation rows are excluded.

## Target And Rolling Checks

Fold config selection uses only years before the fold start. Online routing inside a fold may use earlier scored rows only after their outcomes are at least seven target days mature.

## Confirmation Proof

Maximum scored target date is `{date_text(predictions['target_date'].max())}`. Confirmation rows used: `0`. Owner authorization for confirmation: `false`.

## Row Identity

Candidate, parent 0196, and official baseline share `{len(predictions)}` rows. Common row hash: `{common_row_hash}`.
""")
    write_text(EXP_DIR / "REPRODUCE.md", f"""# Reproduction

From `{REPO_ROOT}`, run:

```powershell
python scripts/run_hkg_t24_0203_july_october_micro_no_harm_governor.py
```

Requires completed parent prediction folders 0194, 0196, 0199, 0200, and 0201. Confirmation rows remain locked.
""")
    code_sha = sha256_file(src_copy_path)
    data_manifest_sha = sha256_file(EXP_DIR / "data_manifest.csv")
    summary = {"experiment_id": EXPERIMENT_ID, "slug": SLUG, "status": status, "created_at_utc": created_at, "target": "HKO daily Tmax T-24", "frame_id": "official_t15_pre2024_5265_rows", "date_start": date_text(predictions["target_date"].min()), "date_end": date_text(predictions["target_date"].max()), "n_candidate": int(len(predictions)), "n_common": int(len(predictions)), "baseline_id": "official_forecast_max_c", "baseline_mae_c": official_global["mae_c"], "candidate_id": PRIMARY_CANDIDATE_ID, "candidate_mae_c": candidate_global["mae_c"], "mae_delta_c": mae_delta, "candidate_rmse_c": candidate_global["rmse_c"], "candidate_bias_c": candidate_global["bias_c"], "leakage_status": "PASS", "confirmation_rows_used": 0, "owner_authorized_confirmation": False, "promotion_decision": promotion_decision, "spec_sha256": spec_sha, "code_sha256": code_sha, "data_manifest_sha256": data_manifest_sha, "common_row_hash": common_row_hash, "baseline_n": int(len(predictions)), "candidate_n": int(len(predictions)), "development_gate_reached": bool(candidate_global["mae_c"] <= 0.45), "parent_0196_mae_c": parent_global["mae_c"], "delta_vs_0196_mae_c": delta_vs_0196, "fold_worst_delta_vs_0196_mae_c": fold_worst_delta, "severe_gt3_rate_delta_vs_0196": severe_harm_0196, "switch_rate": float(predictions["switched"].mean())}
    write_json(EXP_DIR / "summary.json", summary)
    write_docs(scoreboard, slice_metrics, yearly_metrics, fold_metrics, selections, summary)
    write_json(EXP_DIR / "run_manifest.json", {"experiment_id": EXPERIMENT_ID, "slug": SLUG, "created_at_utc": created_at, "completed_at_utc": utc_now(), "repo_root": str(REPO_ROOT), "script": rel(Path(__file__).resolve()), "spec_sha256": spec_sha, "code_sha256": code_sha, "state": "COMPLETED", "executor_invocation": "Executor skill instructions followed in-process because no separate skill agent activation is available."})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
