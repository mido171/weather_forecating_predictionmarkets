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
EXPERIMENT_ID = "0198"
SLUG = "source_era_online_expert_router"
TITLE = "Source-Era Online Expert Router Over 0196"
EXP_DIR = EXPERIMENTS_ROOT / f"{EXPERIMENT_ID}_{SLUG}"
PRIMARY_CANDIDATE_ID = "0198_source_era_online_expert_router_over_0196"
P0196 = EXPERIMENTS_ROOT / "0196_station_network_tail_conditioned_residual_expert" / "predictions.parquet"
SRC_COPY_NAME = "run_0198.py"
MODEL_FOLDS = base.MODEL_FOLDS
LAG_DAYS = 7
EXPERTS = [
    ("official", "official_prediction_c"),
    ("p0190", "parent_0190_prediction_c"),
    ("p0194", "parent_0194_prediction_c"),
    ("p0196", "parent_0196_prediction_c"),
]

CONFIG_GRID = [
    {"router": "parent_only", "contexts": "none", "halflife": 90.0, "min_history": 0, "temperature_c": 0.05, "champion_advantage_c": 0.0, "min_parent_weight": 1.00, "hard_best": False},
    {"router": "source_soft", "contexts": "source", "halflife": 60.0, "min_history": 30, "temperature_c": 0.05, "champion_advantage_c": 0.003, "min_parent_weight": 0.35, "hard_best": False},
    {"router": "source_month_soft", "contexts": "source_month", "halflife": 90.0, "min_history": 35, "temperature_c": 0.04, "champion_advantage_c": 0.003, "min_parent_weight": 0.30, "hard_best": False},
    {"router": "source_month_hard", "contexts": "source_month", "halflife": 120.0, "min_history": 45, "temperature_c": 0.05, "champion_advantage_c": 0.004, "min_parent_weight": 0.00, "hard_best": True},
    {"router": "source_season_soft", "contexts": "source_season", "halflife": 90.0, "min_history": 40, "temperature_c": 0.04, "champion_advantage_c": 0.003, "min_parent_weight": 0.30, "hard_best": False},
    {"router": "source_era_soft", "contexts": "source_era", "halflife": 120.0, "min_history": 45, "temperature_c": 0.04, "champion_advantage_c": 0.004, "min_parent_weight": 0.30, "hard_best": False},
    {"router": "source_tail_soft", "contexts": "source_tail", "halflife": 90.0, "min_history": 35, "temperature_c": 0.04, "champion_advantage_c": 0.003, "min_parent_weight": 0.30, "hard_best": False},
    {"router": "all_context_soft", "contexts": "all", "halflife": 120.0, "min_history": 45, "temperature_c": 0.05, "champion_advantage_c": 0.004, "min_parent_weight": 0.35, "hard_best": False},
]


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


def era_for_year(year: int) -> str:
    if year <= 2004:
        return "era_2000_2004"
    if year <= 2009:
        return "era_2005_2009"
    if year <= 2014:
        return "era_2010_2014"
    if year <= 2019:
        return "era_2015_2019"
    return "era_2020_2023"


def load_frame() -> pd.DataFrame:
    frame = pd.read_parquet(P0196)
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
    frame = frame[frame["target_date"] < pd.Timestamp("2024-01-01")].copy()
    frame = frame.rename(
        columns={
            "candidate_prediction_c": "parent_0196_prediction_c",
            "candidate_error_c": "parent_0196_error_c",
            "candidate_abs_error_c": "parent_0196_abs_error_c",
            "tail_expert_correction_c": "parent_0196_tail_expert_correction_c",
        }
    )
    frame["parent_0196_error_c"] = frame["parent_0196_prediction_c"] - frame["target_tmax_c"]
    frame["parent_0196_abs_error_c"] = frame["parent_0196_error_c"].abs()
    frame["parent_0194_error_c"] = frame["parent_0194_prediction_c"] - frame["target_tmax_c"]
    frame["parent_0194_abs_error_c"] = frame["parent_0194_error_c"].abs()
    frame["parent_0190_error_c"] = frame["parent_0190_prediction_c"] - frame["target_tmax_c"]
    frame["parent_0190_abs_error_c"] = frame["parent_0190_error_c"].abs()
    frame["official_error_c_signed"] = frame["official_prediction_c"] - frame["target_tmax_c"]
    frame["official_abs_error_c"] = frame["official_error_c_signed"].abs()
    frame["month"] = pd.to_numeric(frame["month"], errors="coerce").fillna(frame["target_date"].dt.month).astype(int)
    frame["year"] = frame["target_date"].dt.year.astype(int)
    frame["era"] = frame["year"].map(era_for_year)
    frame["tail_abs_c"] = pd.to_numeric(frame.get("parent_0196_tail_expert_correction_c", 0.0), errors="coerce").abs()
    frame["tail_sign"] = np.where(frame.get("parent_0196_tail_expert_correction_c", 0.0) > 0.0, "pos", np.where(frame.get("parent_0196_tail_expert_correction_c", 0.0) < 0.0, "neg", "zero"))
    frame["tail_mag_bin"] = pd.cut(frame["tail_abs_c"], bins=[-0.001, 0.001, 0.05, 0.10, 0.151], labels=["zero", "small", "medium", "cap"]).astype(str)
    base.assert_pre2024(frame, "0198 parent 0196 frame")
    return frame.sort_values("target_date").reset_index(drop=True)


def context_keys(row: pd.Series, config: dict[str, Any]) -> list[str]:
    mode = str(config["contexts"])
    if mode == "none":
        return []
    source = str(row.get("forecast_source_family") or "source_unknown")
    month = f"month={int(row['month']):02d}"
    season = str(row.get("season") or "season_unknown")
    era = str(row.get("era") or "era_unknown")
    tail = f"tail={row.get('tail_sign')}|{row.get('tail_mag_bin')}"
    keys = ["global", f"source={source}"]
    if mode in {"source_month", "all"}:
        keys.append(f"source={source}|{month}")
    if mode in {"source_season", "all"}:
        keys.append(f"source={source}|season={season}")
    if mode in {"source_era", "all"}:
        keys.extend([f"source={source}|{era}", f"source={source}|{era}|season={season}"])
    if mode in {"source_tail", "all"}:
        keys.extend([f"source={source}|{tail}", tail])
    return keys


def update_state(state: dict[str, dict[str, dict[str, float]]], key: str, losses: dict[str, float], decay: float) -> None:
    rec = state.setdefault(key, {})
    for expert, loss in losses.items():
        erec = rec.setdefault(expert, {"count": 0.0, "weighted_sum": 0.0, "weight_sum": 0.0})
        erec["weighted_sum"] = erec["weighted_sum"] * decay + float(loss)
        erec["weight_sum"] = erec["weight_sum"] * decay + 1.0
        erec["count"] += 1.0


def context_losses(state: dict[str, dict[str, dict[str, float]]], keys: list[str], config: dict[str, Any]) -> tuple[dict[str, float], int]:
    min_history = float(config["min_history"])
    totals = {name: 0.0 for name, _ in EXPERTS}
    weights = {name: 0.0 for name, _ in EXPERTS}
    active = 0
    for key in keys:
        rec = state.get(key)
        if not rec:
            continue
        key_ready = False
        for expert, _ in EXPERTS:
            erec = rec.get(expert)
            if not erec or erec["count"] < min_history or erec["weight_sum"] <= 0:
                continue
            raw = erec["weighted_sum"] / erec["weight_sum"]
            shrink = erec["count"] / (erec["count"] + min_history)
            weight = math.sqrt(max(1.0, erec["count"]))
            totals[expert] += raw * shrink * weight
            weights[expert] += weight
            key_ready = True
        active += int(key_ready)
    losses = {}
    for expert, _ in EXPERTS:
        if weights[expert] > 0:
            losses[expert] = totals[expert] / weights[expert]
    return losses, active


def expert_weights(losses: dict[str, float], active: int, config: dict[str, Any]) -> dict[str, float]:
    if config["router"] == "parent_only" or not losses or active <= 0:
        return {name: (1.0 if name == "p0196" else 0.0) for name, _ in EXPERTS}
    adjusted = {}
    for expert, loss in losses.items():
        adjusted[expert] = float(loss) - (float(config["champion_advantage_c"]) if expert == "p0196" else 0.0)
    if config["hard_best"]:
        best = min(adjusted, key=adjusted.get)
        return {name: (1.0 if name == best else 0.0) for name, _ in EXPERTS}
    min_loss = min(adjusted.values())
    temp = max(0.005, float(config["temperature_c"]))
    raw = {expert: math.exp(-(loss - min_loss) / temp) for expert, loss in adjusted.items()}
    total = sum(raw.values())
    weights = {name: raw.get(name, 0.0) / total for name, _ in EXPERTS}
    floor = float(config["min_parent_weight"])
    if weights["p0196"] < floor:
        other_total = sum(v for k, v in weights.items() if k != "p0196")
        scale = (1.0 - floor) / other_total if other_total > 0 else 0.0
        weights = {k: (floor if k == "p0196" else v * scale) for k, v in weights.items()}
    return weights


def prequential(frame: pd.DataFrame, config: dict[str, Any], config_id: str) -> pd.DataFrame:
    ordered = frame.sort_values("target_date").reset_index(drop=True).copy()
    dates = pd.to_datetime(ordered["target_date"]).dt.normalize()
    key_lists = [context_keys(row, config) for _, row in ordered.iterrows()]
    decay = float(np.power(0.5, 1.0 / float(config["halflife"])))
    state: dict[str, dict[str, dict[str, float]]] = {}
    add_idx = 0
    preds = []
    active_counts = []
    weight_rows = {name: [] for name, _ in EXPERTS}
    for idx, current_date in enumerate(dates):
        mature_date = current_date - pd.Timedelta(days=LAG_DAYS)
        while add_idx < len(ordered) and dates.iloc[add_idx] <= mature_date:
            row = ordered.iloc[add_idx]
            losses = {name: abs(float(row[col]) - float(row["target_tmax_c"])) for name, col in EXPERTS}
            for key in key_lists[add_idx]:
                update_state(state, key, losses, decay)
            add_idx += 1
        losses, active = context_losses(state, key_lists[idx], config)
        weights = expert_weights(losses, active, config)
        pred = 0.0
        for expert, col in EXPERTS:
            pred += weights[expert] * float(ordered.iloc[idx][col])
            weight_rows[expert].append(weights[expert])
        preds.append(pred)
        active_counts.append(active)
    out = ordered.copy()
    out["config_id"] = config_id
    out["candidate_prediction_c"] = preds
    out["candidate_correction_c"] = out["candidate_prediction_c"] - out["official_prediction_c"]
    out["candidate_error_c"] = out["candidate_prediction_c"] - out["target_tmax_c"]
    out["candidate_abs_error_c"] = out["candidate_error_c"].abs()
    for expert in weight_rows:
        out[f"weight_{expert}"] = weight_rows[expert]
    out["active_router_context_count"] = active_counts
    return out


def compare(frame: pd.DataFrame, *, slice_type: str, slice_value: Any) -> dict[str, Any]:
    official = base.metric_row(frame, "official_prediction_c", label="official")
    p0190 = base.metric_row(frame, "parent_0190_prediction_c", label="p0190")
    p0194 = base.metric_row(frame, "parent_0194_prediction_c", label="p0194")
    p0196 = base.metric_row(frame, "parent_0196_prediction_c", label="p0196")
    candidate = base.metric_row(frame, "candidate_prediction_c", label=PRIMARY_CANDIDATE_ID)
    return {
        "slice_type": slice_type,
        "slice_value": str(slice_value),
        "n": candidate["n"],
        "official_mae_c": official["mae_c"],
        "parent_0190_mae_c": p0190["mae_c"],
        "parent_0194_mae_c": p0194["mae_c"],
        "parent_0196_mae_c": p0196["mae_c"],
        "candidate_mae_c": candidate["mae_c"],
        "mae_delta_c": candidate["mae_c"] - official["mae_c"],
        "delta_vs_0196_mae_c": candidate["mae_c"] - p0196["mae_c"],
        "candidate_bias_c": candidate["bias_c"],
        "parent_0196_gt3c_rate": p0196["gt3c_rate"],
        "candidate_gt3c_rate": candidate["gt3c_rate"],
        "parent_0196_p95_abs_error_c": p0196["p95_abs_error_c"],
        "candidate_p95_abs_error_c": candidate["p95_abs_error_c"],
    }


def run_walk_forward(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config_rows = []
    preds = {}
    for idx, config in enumerate(CONFIG_GRID):
        cid = f"cfg_{idx:02d}_{config['router']}"
        cfg = {"config_id": cid, **config}
        config_rows.append(cfg)
        preds[cid] = prequential(frame, cfg, cid)
    config_table = pd.DataFrame(config_rows)
    parts = []
    folds = []
    selections = []
    for start_year, end_year in MODEL_FOLDS:
        fold_id = f"fold_{start_year}_{end_year}"
        test_mask = frame["target_date"].dt.year.between(start_year, end_year)
        if not test_mask.any():
            folds.append({"fold_id": fold_id, "n": 0})
            continue
        if start_year == MODEL_FOLDS[0][0]:
            selected = config_table.iloc[0].to_dict()
            selected["train_mae_c"] = math.nan
            selected["train_delta_vs_0196_mae_c"] = 0.0
        else:
            rows = []
            for _, cfg in config_table.iterrows():
                pred = preds[cfg["config_id"]]
                train = pred[pred["target_date"].dt.year < start_year]
                cand_mae = float(train["candidate_abs_error_c"].mean())
                parent_mae = float(train["parent_0196_abs_error_c"].mean())
                rows.append({**cfg.to_dict(), "train_mae_c": cand_mae, "train_delta_vs_0196_mae_c": cand_mae - parent_mae})
            selected = pd.DataFrame(rows).sort_values(["train_mae_c", "train_delta_vs_0196_mae_c", "config_id"]).iloc[0].to_dict()
        pred = preds[selected["config_id"]][test_mask].copy()
        pred["fold_id"] = fold_id
        pred["selected_config_id"] = selected["config_id"]
        pred["selected_router"] = selected["router"]
        parts.append(pred)
        metric = compare(pred, slice_type="fold", slice_value=fold_id)
        metric.update(
            {
                "fold_id": fold_id,
                "start_year": start_year,
                "end_year": end_year,
                "selected_config_id": selected["config_id"],
                "selected_router": selected["router"],
                "selection_train_mae_c": selected.get("train_mae_c"),
                "selection_train_delta_vs_0196_mae_c": selected.get("train_delta_vs_0196_mae_c"),
                "mean_weight_official": float(pred["weight_official"].mean()),
                "mean_weight_p0190": float(pred["weight_p0190"].mean()),
                "mean_weight_p0194": float(pred["weight_p0194"].mean()),
                "mean_weight_p0196": float(pred["weight_p0196"].mean()),
                "mean_active_router_context_count": float(pred["active_router_context_count"].mean()),
            }
        )
        folds.append(metric)
        selections.append({"fold_id": fold_id, **selected})
    out = pd.concat(parts, ignore_index=True).sort_values("target_date").reset_index(drop=True)
    out["candidate_id"] = PRIMARY_CANDIDATE_ID
    out["baseline_id"] = "official_forecast_max_c"
    out["model_family"] = "source_era_online_expert_router"
    return out, pd.DataFrame(folds), pd.DataFrame(selections)


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
    parent_tail = predictions[predictions["parent_0196_abs_error_c"] >= 2.0]
    rows.append(compare(parent_tail, slice_type="parent_tail", slice_value="parent_0196_abs_error_ge_2c"))
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
        "hypothesis": "T-7 mature source, source-era, and source-tail prior-loss states can route between official, 0190, 0194, and 0196 to reduce MAE beyond the 0196 champion.",
        "rationale": "0196 improves press more than RSS and leaves RSS late-window tail-rate pressure. 0197 showed correction shrinkage alone is too weak; a multi-expert router can fall back to safer parent experts when source-context evidence supports it.",
        "expected_sign_and_falsification": "Expected sign is lower MAE than 0196 or reduced RSS/tail harm. Falsified if prior-loss routing selects parent-only or worsens 0196.",
        "novelty": {"prior_experiments": ["0194", "0196", "0197"], "difference": "Online expert routing across multiple frozen forecast experts by source-era context, rather than additive residual fitting or single-parent trust shrinkage.", "similarity_audit_path": "RESULTS.md#comparison-limitations"},
        "target": {"station": "Hong Kong Observatory headquarters", "variable": "tmax_c", "horizon": "T-24", "timezone": "Asia/Hong_Kong", "cutoff_contract_path": rel(base.OFFICIAL_PATH), "cutoff_function": "For target T, expert loss states admit only target dates <= T-7.", "daily_boundary_contract": "HKO local daily maximum temperature for target local date T."},
        "frame": {"frame_id": "official_t15_pre2024_5265_rows", "development_start": "2000-01-02", "development_end_exclusive": "2024-01-01", "confirmation_locked": True, "row_universe_artifact": rel(P0196)},
        "data_sources": [{"source_id": "0196_parent_predictions", "paths": [rel(P0196)], "attributes": ["official", "0190", "0194", "0196 out-of-fold predictions", "T-7 mature prior expert losses"], "eligibility": "DEPLOYABLE_LAGGED_ONLY", "availability_proof": "0196 validator passed; routing state uses only losses from target dates <= T-7."}],
        "stations": [{"station_id": "HKO", "role": "target and mature expert-loss state", "attributes": ["daily Tmax"]}],
        "features": {"generation_rule": "Prequential expert loss states by global/source/source-month/source-season/source-era/source-tail contexts; softmax or hard-best expert weights over official, 0190, 0194, and 0196.", "grid": CONFIG_GRID, "explicit_exclusions": ["2024+ rows", "current target outcome", "current target residual/loss"]},
        "response": {"variable": "expert absolute loss for mature state; target_tmax_c for scoring only", "prediction": "weighted average of frozen official/0190/0194/0196 predictions"},
        "baseline": {"id": "official_forecast_max_c", "definition": "Raw official forecast on identical rows; 0196 is the parent champion reference."},
        "validation": {"outer_folds": [list(item) for item in MODEL_FOLDS], "inner_selection": "Config selected by prior-year MAE only; current fold outcomes are not used for config choice.", "minimum_train_rows": 365},
        "metrics": ["MAE", "RMSE", "bias", "P95 AE", ">3C rate", "fold/year/season/month/source/tail slices", "expert weights", "delta_vs_0196"],
        "sample_rules": {"row_policy": "All 0196 parent rows.", "missing_policy": "No mature context defaults to 0196 parent."},
        "acceptance_gates": {"minimum_mae_lift_vs_official_c": 0.01, "minimum_mae_lift_vs_0196_c": 0.001, "max_fold_harm_vs_0196_c": 0.001, "no_parent_tail_harm": ">3C rate cannot exceed 0196 by more than 0.005."},
        "rejection_conditions": ["Any 2024+ row.", "Any expert loss update newer than T-7.", "Parent row mismatch."],
        "required_outputs": ["README.md", "RESULTS.md", "CONCLUSION.md", "scoreboard.csv", "slice_metrics.csv", "yearly_metrics.csv", "fold_metrics.csv", "predictions.parquet"],
        "owner_authorized_confirmation": False,
    }


def write_docs(scoreboard: pd.DataFrame, slice_metrics: pd.DataFrame, yearly_metrics: pd.DataFrame, fold_metrics: pd.DataFrame, selections: pd.DataFrame, summary: dict[str, Any]) -> None:
    write_text(EXP_DIR / "README.md", f"""# {TITLE}

## Experiment Identity And Status

Experiment `{EXPERIMENT_ID}` is `{summary['status']}`. It is a T-7 mature online expert router over the 0196 champion.

## One-Sentence Hypothesis

Source-era prior losses can decide when the frozen 0196 station-tail expert should be trusted versus safer parent experts.

## Why It Is Worth Doing

0196 is the current champion, but its RSS late-window gain is small and tail rate rose there. 0197 showed simple correction trust shrinkage does not help. This experiment tests a richer routing response using multiple frozen experts.

## Prior Evidence And Novelty

0194 and 0196 proved the station-network lane is valuable. 0198 does not refit a new station model; it learns online source-era expert trust using only mature prior losses.

## Target, Horizon, And Exact Cutoff

Target is HKO Tmax at `T-24`, timezone `Asia/Hong_Kong`. Expert-loss states for target T use only target dates <= T-7.

## Datasets, Stations, And Attributes

Input is the validator-clean 0196 prediction artifact, which includes official, 0190, 0194, and 0196 predictions on the same frame.

## Feature Definitions

Features are mature expert-loss states by source, month, season, era, and tail-correction contexts. Details are in `feature_definitions.csv`.

## Response And Baseline

The scored prediction is a weighted average of frozen expert predictions. Official raw forecast is the primary baseline; 0196 is the parent champion reference.

## Walk-Forward Design

Each fold selects one router config using only prior-year MAE. The parent-only config exactly reproduces 0196 and is always available.

## Acceptance And Rejection Criteria

Acceptance requires at least 0.001 C global MAE lift versus 0196 without fold or severe-tail harm.

## Reproduction Command

Run `python scripts/run_hkg_t24_0198_source_era_online_expert_router.py` from the repository root.
""")
    write_text(EXP_DIR / "RESULTS.md", f"""# Results

## Headline Result Table

{base.markdown_table(scoreboard)}

## Coverage And Row Identity

Rows: `{summary['n_common']}` from `{summary['date_start']}` to `{summary['date_end']}`. Common row hash: `{summary['common_row_hash']}`.

## Global Metrics

Official MAE is `{summary['baseline_mae_c']:.6f}` C. Parent 0196 MAE is `{summary['parent_0196_mae_c']:.6f}` C. Candidate MAE is `{summary['candidate_mae_c']:.6f}` C. Delta versus 0196 is `{summary['delta_vs_0196_mae_c']:.6f}` C.

## Fold Stability

{base.markdown_table(fold_metrics[['fold_id', 'n', 'parent_0196_mae_c', 'candidate_mae_c', 'delta_vs_0196_mae_c', 'selected_config_id', 'selected_router', 'mean_weight_official', 'mean_weight_p0190', 'mean_weight_p0194', 'mean_weight_p0196']], max_rows=20)}

## Yearly And Monthly Results

{base.markdown_table(yearly_metrics[['slice_value', 'n', 'parent_0196_mae_c', 'candidate_mae_c', 'delta_vs_0196_mae_c']], max_rows=30)}

Month metrics:

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].eq('month')][['slice_value', 'n', 'parent_0196_mae_c', 'candidate_mae_c', 'delta_vs_0196_mae_c']], max_rows=20)}

## Tail And Source Results

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].isin(['season', 'source', 'late_window', 'parent_tail'])][['slice_type', 'slice_value', 'n', 'parent_0196_mae_c', 'candidate_mae_c', 'delta_vs_0196_mae_c', 'parent_0196_gt3c_rate', 'candidate_gt3c_rate']], max_rows=25)}

## Data-Quality And Leakage Result

Leakage status is `{summary['leakage_status']}`. Expert-loss router state is T-7 mature.

## Comparison Limitations

This is a child router over frozen experts. If it does not beat 0196 globally, it should not replace the parent.

Selections:

{base.markdown_table(selections, max_rows=20)}
""")
    write_text(EXP_DIR / "CONCLUSION.md", f"""# Conclusion

## Verdict

Status is `{summary['status']}` with promotion decision `{summary['promotion_decision']}`.

## What Was Learned

0198 tested whether source-era online expert losses can improve the 0196 champion by routing among official, 0190, 0194, and 0196.

## Realized Point-MAE Change

Delta versus official is `{summary['mae_delta_c']:.6f}` C. Delta versus 0196 is `{summary['delta_vs_0196_mae_c']:.6f}` C.

## Robustness And Uncertainty

Robustness comes from using frozen expert predictions, defaulting to 0196 when mature evidence is absent, selecting configs only on prior years, and updating expert losses only after seven target days. Remaining uncertainty is adaptive search risk across many child experiments on the same development frame.

## Failure Diagnosis

If parent-only is selected or MAE does not improve, source-era loss routing is not strong enough after the 0196 station-tail expert. Future work should then focus on feature-family dissection or genuinely new safe data rather than more parent routing.

## Promotion Status

Confirmation remains locked. Development gate to 0.45 C was not reached.
""")


def main() -> None:
    created_at = utc_now()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in ("src", "artifacts", "logs", "diagnostics", "figures"):
        (EXP_DIR / subdir).mkdir(parents=True, exist_ok=True)
    spec = build_spec(created_at)
    write_json(EXP_DIR / "experiment_spec.json", spec)
    spec_sha = sha256_file(EXP_DIR / "experiment_spec.json")
    write_json(EXP_DIR / "run_manifest.json", {"experiment_id": EXPERIMENT_ID, "slug": SLUG, "created_at_utc": created_at, "spec_sha256": spec_sha, "state": "SPEC_WRITTEN_BEFORE_SCORING"})
    src_copy_path = EXP_DIR / "src" / SRC_COPY_NAME
    shutil.copy2(Path(__file__).resolve(), src_copy_path)
    frame = load_frame()
    predictions, fold_metrics, selections = run_walk_forward(frame)
    slice_metrics, yearly_metrics = build_slice_metrics(predictions)
    official_global = base.metric_row(predictions, "official_prediction_c", label="official")
    p0190_global = base.metric_row(predictions, "parent_0190_prediction_c", label="p0190")
    p0194_global = base.metric_row(predictions, "parent_0194_prediction_c", label="p0194")
    p0196_global = base.metric_row(predictions, "parent_0196_prediction_c", label="p0196")
    candidate_global = base.metric_row(predictions, "candidate_prediction_c", label=PRIMARY_CANDIDATE_ID)
    mae_delta = candidate_global["mae_c"] - official_global["mae_c"]
    delta_vs_0196 = candidate_global["mae_c"] - p0196_global["mae_c"]
    severe_harm_0196 = candidate_global["gt3c_rate"] - p0196_global["gt3c_rate"]
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
            {"candidate_id": "0190_parent", "model_family": "expert_reference", "n": p0190_global["n"], "mae_c": p0190_global["mae_c"], "rmse_c": p0190_global["rmse_c"], "bias_c": p0190_global["bias_c"], "median_abs_error_c": p0190_global["median_abs_error_c"], "p95_abs_error_c": p0190_global["p95_abs_error_c"], "gt2c_rate": p0190_global["gt2c_rate"], "gt3c_rate": p0190_global["gt3c_rate"], "baseline_mae_c": official_global["mae_c"], "mae_delta_c": p0190_global["mae_c"] - official_global["mae_c"]},
            {"candidate_id": "0194_parent", "model_family": "expert_reference", "n": p0194_global["n"], "mae_c": p0194_global["mae_c"], "rmse_c": p0194_global["rmse_c"], "bias_c": p0194_global["bias_c"], "median_abs_error_c": p0194_global["median_abs_error_c"], "p95_abs_error_c": p0194_global["p95_abs_error_c"], "gt2c_rate": p0194_global["gt2c_rate"], "gt3c_rate": p0194_global["gt3c_rate"], "baseline_mae_c": official_global["mae_c"], "mae_delta_c": p0194_global["mae_c"] - official_global["mae_c"]},
            {"candidate_id": "0196_parent_tail_expert", "model_family": "parent_reference", "n": p0196_global["n"], "mae_c": p0196_global["mae_c"], "rmse_c": p0196_global["rmse_c"], "bias_c": p0196_global["bias_c"], "median_abs_error_c": p0196_global["median_abs_error_c"], "p95_abs_error_c": p0196_global["p95_abs_error_c"], "gt2c_rate": p0196_global["gt2c_rate"], "gt3c_rate": p0196_global["gt3c_rate"], "baseline_mae_c": official_global["mae_c"], "mae_delta_c": p0196_global["mae_c"] - official_global["mae_c"]},
            {"candidate_id": PRIMARY_CANDIDATE_ID, "model_family": "source_era_online_expert_router", "n": candidate_global["n"], "mae_c": candidate_global["mae_c"], "rmse_c": candidate_global["rmse_c"], "bias_c": candidate_global["bias_c"], "median_abs_error_c": candidate_global["median_abs_error_c"], "p95_abs_error_c": candidate_global["p95_abs_error_c"], "gt2c_rate": candidate_global["gt2c_rate"], "gt3c_rate": candidate_global["gt3c_rate"], "baseline_mae_c": official_global["mae_c"], "mae_delta_c": mae_delta},
        ]
    )
    row_coverage = pd.DataFrame([{"frame_id": "official_t15_pre2024_5265_rows", "parent_rows": int(len(frame)), "candidate_rows": int(len(predictions)), "baseline_rows": int(len(predictions)), "common_rows": int(len(predictions)), "date_start": date_text(predictions["target_date"].min()), "date_end": date_text(predictions["target_date"].max()), "row_policy": "all 0196 parent rows", "common_row_hash": common_row_hash}])
    weight_distribution = predictions[["weight_official", "weight_p0190", "weight_p0194", "weight_p0196", "active_router_context_count"]].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).reset_index()
    data_manifest = pd.DataFrame([{"source_id": "0196_parent_predictions", "path": rel(P0196), "sha256": sha256_file(P0196), "size_bytes": P0196.stat().st_size, "row_count": int(len(frame)), "date_start": date_text(frame["target_date"].min()), "date_end": date_text(frame["target_date"].max()), "timestamp_fields": "target_date;frozen expert predictions;T-7 mature expert losses", "availability_class": "DEPLOYABLE_LAGGED_ONLY", "notes": "Validator-clean 0196 predictions; expert-loss state admits only rows <= T-7."}])
    feature_definitions = pd.DataFrame([{"feature_name": "t7_source_era_online_expert_losses", "role": "candidate_router", "formula": "EW absolute-loss states for official/0190/0194/0196 experts by predeclared source-era contexts; current row uses only target_date <= T-7 state.", "input_columns": "expert predictions,target_tmax_c,target_date,forecast_source_family,season,month,parent_0196_tail_expert_correction_c", "units": "degC absolute error and normalized weights", "lag": "7 target days minimum", "window": "predeclared EW half-life grid", "fit_scope": "fold-local config selection; prequential state", "availability_rule": "Current target outcome never updates its own router weights.", "missingness_policy": "No mature context gives default 0196 weight 1.0."}])
    pred_cols = ["target_date", "target_tmax_c", "forecast_source_family", "season", "month", "era", "official_prediction_c", "parent_0190_prediction_c", "parent_0194_prediction_c", "parent_0196_prediction_c", "candidate_prediction_c", "candidate_correction_c", "official_abs_error_c", "parent_0190_abs_error_c", "parent_0194_abs_error_c", "parent_0196_abs_error_c", "candidate_abs_error_c", "weight_official", "weight_p0190", "weight_p0194", "weight_p0196", "active_router_context_count", "fold_id", "selected_config_id", "selected_router", "candidate_id", "baseline_id", "model_family"]
    write_parquet(EXP_DIR / "predictions.parquet", predictions[pred_cols])
    write_csv(EXP_DIR / "scoreboard.csv", scoreboard)
    write_csv(EXP_DIR / "slice_metrics.csv", slice_metrics)
    write_csv(EXP_DIR / "yearly_metrics.csv", yearly_metrics)
    write_csv(EXP_DIR / "fold_metrics.csv", fold_metrics)
    write_csv(EXP_DIR / "row_coverage.csv", row_coverage)
    write_csv(EXP_DIR / "correction_distribution.csv", weight_distribution)
    write_csv(EXP_DIR / "data_manifest.csv", data_manifest)
    write_csv(EXP_DIR / "feature_definitions.csv", feature_definitions)
    write_csv(EXP_DIR / "artifacts" / "config_grid.csv", pd.DataFrame(CONFIG_GRID))
    write_csv(EXP_DIR / "artifacts" / "fold_config_selections.csv", selections)
    write_text(EXP_DIR / "leakage_audit.md", f"""# Leakage And Point-In-Time Audit

Status: `PASS`

## Cutoff

0198 consumes validator-clean 0196 parent predictions. For target T, router expert-loss contexts include only rows with target_date <= T-{LAG_DAYS}.

## Available State

Current-row source, season, month, era, tail-correction sign/magnitude, and all frozen expert predictions are known before the target outcome. The current target value and current expert losses are never used to decide the row's weights.

## Target And Rolling Checks

Fold config selection uses only years before the fold start. Online loss states can update with previous rows in the scored era only after their outcomes are at least seven target days old.

## Confirmation Proof

Maximum scored target date is `{date_text(predictions['target_date'].max())}`. Confirmation rows used: `0`. Owner authorization for confirmation: `false`.

## Row Identity

Candidate, all frozen experts, and official baseline share `{len(predictions)}` rows. Common row hash: `{common_row_hash}`.
""")
    write_text(EXP_DIR / "REPRODUCE.md", f"""# Reproduction

From `{REPO_ROOT}`, run:

```powershell
python scripts/run_hkg_t24_0198_source_era_online_expert_router.py
```

Requires completed parent predictions from 0196. Confirmation rows remain locked.
""")
    code_sha = sha256_file(src_copy_path)
    data_manifest_sha = sha256_file(EXP_DIR / "data_manifest.csv")
    summary = {"experiment_id": EXPERIMENT_ID, "slug": SLUG, "status": status, "created_at_utc": created_at, "target": "HKO daily Tmax T-24", "frame_id": "official_t15_pre2024_5265_rows", "date_start": date_text(predictions["target_date"].min()), "date_end": date_text(predictions["target_date"].max()), "n_candidate": int(len(predictions)), "n_common": int(len(predictions)), "baseline_id": "official_forecast_max_c", "baseline_mae_c": official_global["mae_c"], "candidate_id": PRIMARY_CANDIDATE_ID, "candidate_mae_c": candidate_global["mae_c"], "mae_delta_c": mae_delta, "candidate_rmse_c": candidate_global["rmse_c"], "candidate_bias_c": candidate_global["bias_c"], "leakage_status": "PASS", "confirmation_rows_used": 0, "owner_authorized_confirmation": False, "promotion_decision": promotion_decision, "spec_sha256": spec_sha, "code_sha256": code_sha, "data_manifest_sha256": data_manifest_sha, "common_row_hash": common_row_hash, "baseline_n": int(len(predictions)), "candidate_n": int(len(predictions)), "development_gate_reached": bool(candidate_global["mae_c"] <= 0.45), "parent_0190_mae_c": p0190_global["mae_c"], "parent_0194_mae_c": p0194_global["mae_c"], "parent_0196_mae_c": p0196_global["mae_c"], "delta_vs_0196_mae_c": delta_vs_0196, "fold_worst_delta_vs_0196_mae_c": fold_worst_delta, "severe_gt3_rate_delta_vs_0196": severe_harm_0196}
    write_json(EXP_DIR / "summary.json", summary)
    write_docs(scoreboard, slice_metrics, yearly_metrics, fold_metrics, selections, summary)
    write_json(EXP_DIR / "run_manifest.json", {"experiment_id": EXPERIMENT_ID, "slug": SLUG, "created_at_utc": created_at, "completed_at_utc": utc_now(), "repo_root": str(REPO_ROOT), "script": rel(Path(__file__).resolve()), "spec_sha256": spec_sha, "code_sha256": code_sha, "state": "COMPLETED"})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
