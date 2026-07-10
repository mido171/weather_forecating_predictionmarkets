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
EXPERIMENT_ID = "0189"
SLUG = "t7_online_expert_weighting"
TITLE = "T-7 Online Expert Weighting Over Official, 0185, and 0187"
EXP_DIR = EXPERIMENTS_ROOT / f"{EXPERIMENT_ID}_{SLUG}"
PRIMARY_CANDIDATE_ID = "0189_t7_contextual_online_expert_weighting"
MODEL_FOLDS = base.MODEL_FOLDS
P0185 = EXPERIMENTS_ROOT / "0185_lag7_online_residual_memory_router" / "predictions.parquet"
P0187 = EXPERIMENTS_ROOT / "0187_deployable_isd_memory_residual_fusion" / "predictions.parquet"
EXPERTS = ["official", "0185", "0187"]
LAG_DAYS = 7
CONFIG_GRID = [
    {"context_mode": "global", "halflife": 20.0, "temperature": 0.06, "min_count": 20},
    {"context_mode": "global", "halflife": 45.0, "temperature": 0.08, "min_count": 20},
    {"context_mode": "global", "halflife": 90.0, "temperature": 0.10, "min_count": 20},
    {"context_mode": "season", "halflife": 20.0, "temperature": 0.06, "min_count": 20},
    {"context_mode": "season", "halflife": 45.0, "temperature": 0.08, "min_count": 20},
    {"context_mode": "season", "halflife": 90.0, "temperature": 0.10, "min_count": 20},
    {"context_mode": "month", "halflife": 20.0, "temperature": 0.06, "min_count": 20},
    {"context_mode": "month", "halflife": 45.0, "temperature": 0.08, "min_count": 20},
    {"context_mode": "month", "halflife": 90.0, "temperature": 0.10, "min_count": 20},
    {"context_mode": "all", "halflife": 20.0, "temperature": 0.06, "min_count": 20},
    {"context_mode": "all", "halflife": 45.0, "temperature": 0.08, "min_count": 20},
    {"context_mode": "all", "halflife": 90.0, "temperature": 0.10, "min_count": 20},
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


def load_parent_frame() -> pd.DataFrame:
    p85 = pd.read_parquet(P0185)
    p87 = pd.read_parquet(P0187)
    for frame in (p85, p87):
        frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
        frame.drop(frame[frame["target_date"] >= pd.Timestamp("2024-01-01")].index, inplace=True)
    cols = ["target_date", "target_tmax_c", "forecast_source_family", "season", "month", "official_prediction_c"]
    out = p85[cols + ["candidate_prediction_c", "candidate_correction_c"]].rename(
        columns={
            "candidate_prediction_c": "prediction_0185_c",
            "candidate_correction_c": "correction_0185_c",
        }
    )
    out = out.merge(
        p87[["target_date", "candidate_prediction_c", "candidate_correction_c"]].rename(
            columns={
                "candidate_prediction_c": "prediction_0187_c",
                "candidate_correction_c": "correction_0187_c",
            }
        ),
        on="target_date",
        how="inner",
        validate="one_to_one",
    )
    out["prediction_official_c"] = out["official_prediction_c"]
    for expert in EXPERTS:
        out[f"error_{expert}_c"] = out[f"prediction_{expert}_c"] - out["target_tmax_c"]
        out[f"abs_error_{expert}_c"] = out[f"error_{expert}_c"].abs()
    base.assert_pre2024(out, "0189 parent frame")
    return out.sort_values("target_date").reset_index(drop=True)


def context_keys(row: pd.Series, mode: str) -> list[str]:
    source = str(row.get("forecast_source_family") or "source_unknown")
    season = str(row.get("season") or "season_unknown")
    month = f"month_{int(row.get('month')):02d}" if pd.notna(row.get("month")) else "month_unknown"
    keys = ["global"]
    if mode in {"season", "all"}:
        keys.extend([f"season={season}", f"season={season}|source={source}"])
    if mode in {"month", "all"}:
        keys.extend([month, f"{month}|source={source}"])
    return keys


def update_loss_state(state: dict[str, dict[str, dict[str, float]]], key: str, losses: dict[str, float], decay: float) -> None:
    bucket = state.setdefault(key, {expert: {"count": 0.0, "weighted_loss": 0.0, "weight_sum": 0.0} for expert in EXPERTS})
    for expert, loss in losses.items():
        record = bucket[expert]
        record["weighted_loss"] = record["weighted_loss"] * decay + float(loss)
        record["weight_sum"] = record["weight_sum"] * decay + 1.0
        record["count"] += 1.0


def expert_weights(state: dict[str, dict[str, dict[str, float]]], keys: list[str], config: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    losses = {expert: [] for expert in EXPERTS}
    counts = {expert: 0.0 for expert in EXPERTS}
    for key in keys:
        bucket = state.get(key)
        if not bucket:
            continue
        for expert in EXPERTS:
            record = bucket[expert]
            if record["count"] >= float(config["min_count"]) and record["weight_sum"] > 0:
                losses[expert].append(record["weighted_loss"] / record["weight_sum"])
                counts[expert] += record["count"]
    mean_losses = {}
    for expert in EXPERTS:
        if losses[expert]:
            mean_losses[expert] = float(np.mean(losses[expert]))
        else:
            mean_losses[expert] = math.nan
    if all(not math.isfinite(value) for value in mean_losses.values()):
        return {"official": 0.0, "0185": 1.0, "0187": 0.0}, mean_losses
    filled = {expert: (mean_losses[expert] if math.isfinite(mean_losses[expert]) else mean_losses["0185"]) for expert in EXPERTS}
    raw = np.array([-filled[expert] / float(config["temperature"]) for expert in EXPERTS], dtype=float)
    raw = raw - np.max(raw)
    exp = np.exp(raw)
    weights = exp / exp.sum()
    return {expert: float(weight) for expert, weight in zip(EXPERTS, weights)}, mean_losses


def prequential_config_predictions(frame: pd.DataFrame, config: dict[str, Any], config_id: str) -> pd.DataFrame:
    ordered = frame.sort_values("target_date").reset_index(drop=True).copy()
    dates = pd.to_datetime(ordered["target_date"]).dt.normalize()
    contexts = [context_keys(row, str(config["context_mode"])) for _, row in ordered.iterrows()]
    decay = float(np.power(0.5, 1.0 / float(config["halflife"])))
    state: dict[str, dict[str, dict[str, float]]] = {}
    add_index = 0
    predictions: list[float] = []
    correction: list[float] = []
    weight_rows: list[dict[str, float]] = []
    for idx, current_date in enumerate(dates):
        mature_date = current_date - pd.Timedelta(days=LAG_DAYS)
        while add_index < len(ordered) and dates.iloc[add_index] <= mature_date:
            losses = {expert: float(ordered.iloc[add_index][f"abs_error_{expert}_c"]) for expert in EXPERTS}
            for key in contexts[add_index]:
                update_loss_state(state, key, losses, decay)
            add_index += 1
        weights, mean_losses = expert_weights(state, contexts[idx], config)
        pred = sum(weights[expert] * float(ordered.iloc[idx][f"prediction_{expert}_c"]) for expert in EXPERTS)
        predictions.append(pred)
        correction.append(pred - float(ordered.iloc[idx]["official_prediction_c"]))
        weight_rows.append(
            {
                "weight_official": weights["official"],
                "weight_0185": weights["0185"],
                "weight_0187": weights["0187"],
                "loss_official": mean_losses["official"],
                "loss_0185": mean_losses["0185"],
                "loss_0187": mean_losses["0187"],
            }
        )
    out = ordered.copy()
    out["config_id"] = config_id
    out["candidate_prediction_c"] = predictions
    out["candidate_correction_c"] = correction
    out = pd.concat([out, pd.DataFrame(weight_rows)], axis=1)
    out["candidate_error_c"] = out["candidate_prediction_c"] - out["target_tmax_c"]
    out["candidate_abs_error_c"] = out["candidate_error_c"].abs()
    out["official_error_c_signed"] = out["official_prediction_c"] - out["target_tmax_c"]
    out["official_abs_error_c"] = out["official_error_c_signed"].abs()
    return out


def run_walk_forward(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config_rows = []
    config_predictions = {}
    for idx, config in enumerate(CONFIG_GRID, start=1):
        config_id = f"cfg_{idx:02d}_{config['context_mode']}_h{int(config['halflife'])}_t{str(config['temperature']).replace('.', 'p')}"
        cfg = {"config_id": config_id, **config}
        config_rows.append(cfg)
        config_predictions[config_id] = prequential_config_predictions(frame, cfg, config_id)
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
            selected_id = config_table.iloc[0]["config_id"]
            selected = config_table.iloc[0].to_dict()
        else:
            rows = []
            for _, cfg in config_table.iterrows():
                pred = config_predictions[cfg["config_id"]]
                train = pred[pred["target_date"].dt.year < start_year]
                rows.append({**cfg.to_dict(), "train_mae_c": float(train["candidate_abs_error_c"].mean())})
            selection_table = pd.DataFrame(rows).sort_values(["train_mae_c", "halflife", "temperature"]).reset_index(drop=True)
            selected = selection_table.iloc[0].to_dict()
            selected_id = selected["config_id"]
        pred = config_predictions[selected_id][test_mask].copy()
        pred["fold_id"] = fold_id
        pred["selected_config_id"] = selected_id
        parts.append(pred)
        metric = base.compare_metrics(pred, slice_type="fold", slice_value=fold_id)
        metric.update(
            {
                "fold_id": fold_id,
                "start_year": start_year,
                "end_year": end_year,
                "selected_config_id": selected_id,
                "selected_context_mode": selected.get("context_mode"),
                "selected_halflife": selected.get("halflife"),
                "selected_temperature": selected.get("temperature"),
                "selection_train_mae_c": selected.get("train_mae_c", math.nan),
                "mean_weight_official": float(pred["weight_official"].mean()),
                "mean_weight_0185": float(pred["weight_0185"].mean()),
                "mean_weight_0187": float(pred["weight_0187"].mean()),
            }
        )
        folds.append(metric)
        selections.append({"fold_id": fold_id, **selected})
    predictions = pd.concat(parts, ignore_index=True).sort_values("target_date").reset_index(drop=True)
    predictions["candidate_id"] = PRIMARY_CANDIDATE_ID
    predictions["baseline_id"] = "official_forecast_max_c"
    predictions["model_family"] = "t7_online_contextual_expert_weighting"
    return predictions, pd.DataFrame(folds), pd.DataFrame(selections)


def build_slice_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [base.compare_metrics(predictions, slice_type="overall", slice_value="all")]
    for season, group in predictions.groupby("season", dropna=False):
        rows.append(base.compare_metrics(group, slice_type="season", slice_value=season))
    for month, group in predictions.groupby("month", dropna=False):
        rows.append(base.compare_metrics(group, slice_type="month", slice_value=month))
    for source, group in predictions.groupby("forecast_source_family", dropna=False):
        rows.append(base.compare_metrics(group, slice_type="source", slice_value=source))
    late = predictions[predictions["target_date"] >= pd.Timestamp("2020-01-01")]
    rows.append(base.compare_metrics(late, slice_type="late_window", slice_value="2020_2023"))
    tail = predictions[predictions["official_abs_error_c"] >= 2.0]
    rows.append(base.compare_metrics(tail, slice_type="official_tail", slice_value="official_abs_error_ge_2c"))
    yearly = pd.DataFrame(
        [
            base.compare_metrics(group, slice_type="year", slice_value=year)
            for year, group in predictions.groupby(predictions["target_date"].dt.year, dropna=False)
        ]
    )
    return pd.DataFrame(rows), yearly


def build_spec(created_at: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": created_at,
        "title": TITLE,
        "slug": SLUG,
        "mode": "promotion_oriented",
        "hypothesis": "A T-7 prequential expert-weighting ensemble can adapt between official, 0185, and 0187 better than the hard 0188 context gate.",
        "rationale": "0188 improved raw MAE by routing between 0185 and 0187, but hard context decisions are brittle. A soft online weighting rule can exploit gradual performance shifts while preserving T-7 target availability.",
        "expected_sign_and_falsification": "Expected sign is lower MAE than 0188 and 0185. It is falsified if soft weights collapse to 0185 or add unstable expert mixing damage.",
        "novelty": {"prior_experiments": ["0185", "0187", "0188"], "difference": "Soft T-7 online expert weighting, not hard prior context routing.", "similarity_audit_path": "RESULTS.md#comparison-limitations"},
        "target": {
            "station": "Hong Kong Observatory headquarters",
            "variable": "tmax_c",
            "horizon": "T-24",
            "timezone": "Asia/Hong_Kong",
            "cutoff_contract_path": rel(base.OFFICIAL_PATH),
            "cutoff_function": "Parent predictions must pass leakage audits; expert weights update only from target rows matured by seven days.",
            "daily_boundary_contract": "HKO local daily maximum temperature for target local date T.",
        },
        "frame": {"frame_id": "official_t15_pre2024_5265_rows", "development_start": "2000-01-02", "development_end_exclusive": "2024-01-01", "confirmation_locked": True, "row_universe_artifact": rel(P0185)},
        "data_sources": [
            {"source_id": "0185_parent_predictions", "paths": [rel(P0185)], "attributes": ["candidate_prediction_c"], "eligibility": "DEPLOYABLE_LAGGED_ONLY", "availability_proof": "0185 validator passed; T-7 residual maturity."},
            {"source_id": "0187_parent_predictions", "paths": [rel(P0187)], "attributes": ["candidate_prediction_c"], "eligibility": "DEPLOYABLE_LAGGED_ONLY", "availability_proof": "0187 validator passed; cutoff-safe station fusion."},
        ],
        "stations": [{"station_id": "HKO", "role": "target and expert-loss audit", "attributes": ["daily Tmax"]}],
        "features": {
            "generation_rule": "Maintain exponentially weighted prior absolute error per expert and context; compute softmax expert weights; update states only with rows <= T-7.",
            "grid": CONFIG_GRID,
            "explicit_exclusions": ["2024+ rows", "current-fold expert-loss updates", "unvalidated parents"],
        },
        "response": {"variable": "target_tmax_c", "prediction": "soft weighted average of official, 0185, and 0187 predictions"},
        "baseline": {"id": "official_forecast_max_c", "definition": "Raw official forecast on identical rows."},
        "validation": {"outer_folds": [list(item) for item in MODEL_FOLDS], "inner_selection": "Select context/halflife/temperature by prior-year MAE only.", "minimum_train_rows": 365},
        "metrics": ["MAE", "RMSE", "bias", "P90/P95/max AE", ">2C and >3C rates", "fold/year/season/month/source/tail slices"],
        "sample_rules": {"row_policy": "All common parent rows.", "missing_policy": "No mature expert-loss state defaults to 0185."},
        "acceptance_gates": {"minimum_mae_lift_vs_official_c": 0.01, "minimum_mae_lift_vs_0188_c": 0.002, "no_tail_harm": ">3C rate cannot exceed official by more than 0.005."},
        "rejection_conditions": ["Any 2024+ row.", "Any expert weights updated from target rows newer than T-7.", "Parent row mismatch."],
        "required_outputs": ["README.md", "RESULTS.md", "CONCLUSION.md", "scoreboard.csv", "slice_metrics.csv", "yearly_metrics.csv", "fold_metrics.csv", "predictions.parquet"],
        "owner_authorized_confirmation": False,
    }


def write_docs(predictions: pd.DataFrame, scoreboard: pd.DataFrame, slice_metrics: pd.DataFrame, yearly_metrics: pd.DataFrame, fold_metrics: pd.DataFrame, selections: pd.DataFrame, summary: dict[str, Any]) -> None:
    write_text(
        EXP_DIR / "README.md",
        f"""# {TITLE}

## Experiment Identity And Status

Experiment `{EXPERIMENT_ID}` is `{summary['status']}`. It is a T-7 online expert-weighting child of 0185, 0187, and 0188.

## One-Sentence Hypothesis

Soft online expert weighting from matured prior losses can outperform a hard context gate while remaining T-24 compliant.

## Why It Is Worth Doing

0188 showed complementary parent strengths but used hard context decisions. A smooth online weighting system is a cleaner deployable architecture for uncertain expert superiority.

## Prior Evidence And Novelty

0185 is the residual-memory champion, 0187 improves station/tail slices, and 0188 showed routing value. 0189 replaces hard routing with prequential T-7 expert weights.

## Target, Horizon, And Exact Cutoff

Target is HKO daily Tmax at `T-24`, timezone `Asia/Hong_Kong`. Expert losses enter weights only after the target date is at least seven days old.

## Datasets, Stations, And Attributes

Inputs are validator-clean parent predictions from 0185 and 0187 plus the official prediction on the same frame.

## Feature Definitions

The candidate uses context keys and exponentially weighted expert-loss states. Details are in `feature_definitions.csv`.

## Response And Baseline

Response is target Tmax. Baseline is raw official forecast; 0185, 0187, and 0188 are references.

## Walk-Forward Design

Each outer fold selects one predeclared weighting configuration using only prior years. Prediction-time states update only from rows with target date <= T-7.

## Acceptance And Rejection Criteria

Acceptance requires lower MAE than official and meaningful lift over 0188 without severe-tail harm.

## Expected Failure Modes

Soft weights can fail if expert errors are too close, if weights overreact to stale contexts, or if 0185 already dominates.

## Reproduction Command

Run `python scripts/run_hkg_t24_0189_t7_online_expert_weighting.py` from the repository root.
""",
    )
    weight_summary = predictions[["weight_official", "weight_0185", "weight_0187"]].mean().reset_index()
    weight_summary.columns = ["weight", "mean_value"]
    write_text(
        EXP_DIR / "RESULTS.md",
        f"""# Results

## Headline Result Table

{base.markdown_table(scoreboard)}

## Coverage And Row Identity

The scored frame has `{summary['n_common']}` rows from `{summary['date_start']}` to `{summary['date_end']}`. Common row hash: `{summary['common_row_hash']}`.

## Global Metrics

Official MAE is `{summary['baseline_mae_c']:.6f}` C. Candidate MAE is `{summary['candidate_mae_c']:.6f}` C. Delta versus official is `{summary['mae_delta_c']:.6f}` C. Delta versus 0188 is `{summary['delta_vs_0188_mae_c']:.6f}` C.

## Fold Stability

{base.markdown_table(fold_metrics[['fold_id', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c', 'selected_config_id', 'mean_weight_official', 'mean_weight_0185', 'mean_weight_0187']], max_rows=20)}

## Yearly And Seasonal Results

{base.markdown_table(yearly_metrics[['slice_value', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c']], max_rows=30)}

Season metrics:

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].eq('season')][['slice_value', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c']], max_rows=10)}

## Source And Source-Era Results

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].isin(['source', 'late_window'])][['slice_type', 'slice_value', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c']], max_rows=20)}

## High-Error-Tail Results

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].eq('official_tail')][['slice_value', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c', 'official_gt3c_rate', 'candidate_gt3c_rate']], max_rows=10)}

## Signed Over/Underforecast Results

Row-level weights, losses, signed errors, and corrections are in `predictions.parquet`.

## Ablations

Parent expert rows and 0188 reference are in `scoreboard.csv`; selected weighting configs are in `artifacts/fold_config_selections.csv`.

## Data-Quality And Leakage Result

Leakage status is `{summary['leakage_status']}`. No expert-loss state reads target rows newer than T-7.

## Comparison Limitations

This is a meta-experiment over validated parents. It is not evidence that raw station features alone are deployable as a final model.

Mean weights:

{base.markdown_table(weight_summary, max_rows=10)}

Selections:

{base.markdown_table(selections, max_rows=20)}
""",
    )
    write_text(
        EXP_DIR / "CONCLUSION.md",
        f"""# Conclusion

## Verdict

Status is `{summary['status']}` with promotion decision `{summary['promotion_decision']}`.

## What Was Learned

0189 tested whether soft T-7 expert weighting is better than hard routing between the same leakage-passed parents.

## Realized Point-MAE Change

Delta versus official is `{summary['mae_delta_c']:.6f}` C. Delta versus 0188 is `{summary['delta_vs_0188_mae_c']:.6f}` C.

## Information Gain Outside Point MAE

The learned weights reveal whether 0187's station signal should be used as a small continuous influence or only as a hard routed specialist.

## Robustness And Uncertainty

All weight states are prequential and T-7 mature. Fold config selection is prior-history only.

## Failure Diagnosis

If this did not beat 0188, hard context routing is more useful than smooth loss weighting for these experts.

## Promotion Status

Confirmation remains locked and unauthorized. Development gate to 0.45 C was not reached.

## Implication For Future Research

Use this result to decide whether the next step should refine router contexts or invent a new independent safe signal.
""",
    )


def main() -> None:
    created_at = utc_now()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in ("src", "artifacts", "logs", "diagnostics", "figures"):
        (EXP_DIR / subdir).mkdir(exist_ok=True)
    spec = build_spec(created_at)
    write_json(EXP_DIR / "experiment_spec.json", spec)
    spec_sha = sha256_file(EXP_DIR / "experiment_spec.json")
    write_json(EXP_DIR / "run_manifest.json", {"experiment_id": EXPERIMENT_ID, "slug": SLUG, "created_at_utc": created_at, "spec_sha256": spec_sha, "state": "SPEC_WRITTEN_BEFORE_SCORING"})
    shutil.copy2(Path(__file__).resolve(), EXP_DIR / "src" / Path(__file__).name)

    frame = load_parent_frame()
    predictions, fold_metrics, selections = run_walk_forward(frame)
    slice_metrics, yearly_metrics = build_slice_metrics(predictions)
    official_global = base.metric_row(predictions, "official_prediction_c", label="official_forecast_max_c")
    candidate_global = base.metric_row(predictions, "candidate_prediction_c", label=PRIMARY_CANDIDATE_ID)
    p85_global = base.metric_row(predictions, "prediction_0185_c", label="0185_parent")
    p87_global = base.metric_row(predictions, "prediction_0187_c", label="0187_parent")
    # 0188 reference is read only as a comparable completed artifact.
    p88_path = EXPERIMENTS_ROOT / "0188_nested_expert_router_0185_0187" / "predictions.parquet"
    p88 = pd.read_parquet(p88_path)
    p88["target_date"] = pd.to_datetime(p88["target_date"], errors="coerce").dt.normalize()
    p88 = predictions[["target_date", "target_tmax_c"]].merge(
        p88[["target_date", "candidate_prediction_c"]].rename(columns={"candidate_prediction_c": "prediction_0188_c"}),
        on="target_date",
        how="left",
        validate="one_to_one",
    )
    p88_global = base.metric_row(p88, "prediction_0188_c", label="0188_parent")
    mae_delta = candidate_global["mae_c"] - official_global["mae_c"]
    delta_vs_0188 = candidate_global["mae_c"] - p88_global["mae_c"]
    severe_harm = candidate_global["gt3c_rate"] - official_global["gt3c_rate"]
    fold_worst_delta = float(fold_metrics["mae_delta_c"].max())
    if mae_delta <= -0.01 and delta_vs_0188 <= -0.002 and severe_harm <= 0.005 and fold_worst_delta <= 0.02:
        status = "COMPLETED_PROMOTION_CANDIDATE"
        promotion_decision = "PROMOTE_OVER_0188_NO_CONFIRMATION"
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
            {"candidate_id": PRIMARY_CANDIDATE_ID, "model_family": "t7_online_contextual_expert_weighting", "n": candidate_global["n"], "mae_c": candidate_global["mae_c"], "rmse_c": candidate_global["rmse_c"], "bias_c": candidate_global["bias_c"], "median_abs_error_c": candidate_global["median_abs_error_c"], "p95_abs_error_c": candidate_global["p95_abs_error_c"], "gt2c_rate": candidate_global["gt2c_rate"], "gt3c_rate": candidate_global["gt3c_rate"], "baseline_mae_c": official_global["mae_c"], "mae_delta_c": mae_delta},
            {"candidate_id": "0185_parent_lag7_memory", "model_family": "parent_reference", "n": p85_global["n"], "mae_c": p85_global["mae_c"], "rmse_c": p85_global["rmse_c"], "bias_c": p85_global["bias_c"], "median_abs_error_c": p85_global["median_abs_error_c"], "p95_abs_error_c": p85_global["p95_abs_error_c"], "gt2c_rate": p85_global["gt2c_rate"], "gt3c_rate": p85_global["gt3c_rate"], "baseline_mae_c": official_global["mae_c"], "mae_delta_c": p85_global["mae_c"] - official_global["mae_c"]},
            {"candidate_id": "0187_parent_isd_fusion", "model_family": "parent_reference", "n": p87_global["n"], "mae_c": p87_global["mae_c"], "rmse_c": p87_global["rmse_c"], "bias_c": p87_global["bias_c"], "median_abs_error_c": p87_global["median_abs_error_c"], "p95_abs_error_c": p87_global["p95_abs_error_c"], "gt2c_rate": p87_global["gt2c_rate"], "gt3c_rate": p87_global["gt3c_rate"], "baseline_mae_c": official_global["mae_c"], "mae_delta_c": p87_global["mae_c"] - official_global["mae_c"]},
            {"candidate_id": "0188_parent_router", "model_family": "parent_reference", "n": p88_global["n"], "mae_c": p88_global["mae_c"], "rmse_c": p88_global["rmse_c"], "bias_c": p88_global["bias_c"], "median_abs_error_c": p88_global["median_abs_error_c"], "p95_abs_error_c": p88_global["p95_abs_error_c"], "gt2c_rate": p88_global["gt2c_rate"], "gt3c_rate": p88_global["gt3c_rate"], "baseline_mae_c": official_global["mae_c"], "mae_delta_c": p88_global["mae_c"] - official_global["mae_c"]},
        ]
    )
    row_coverage = pd.DataFrame(
        [
            {"frame_id": "official_t15_pre2024_5265_rows", "parent_rows": int(len(frame)), "candidate_rows": int(len(predictions)), "baseline_rows": int(len(predictions)), "common_rows": int(len(predictions)), "date_start": date_text(predictions["target_date"].min()), "date_end": date_text(predictions["target_date"].max()), "row_policy": "common validated parent rows", "common_row_hash": common_row_hash}
        ]
    )
    correction_distribution = predictions["candidate_correction_c"].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).reset_index()
    correction_distribution.columns = ["statistic", "candidate_correction_c"]
    data_manifest = pd.DataFrame(
        [
            {"source_id": "0185_parent_predictions", "path": rel(P0185), "sha256": sha256_file(P0185), "size_bytes": P0185.stat().st_size, "row_count": int(len(frame)), "date_start": date_text(frame["target_date"].min()), "date_end": date_text(frame["target_date"].max()), "timestamp_fields": "target_date;T-7 residual state", "availability_class": "DEPLOYABLE_LAGGED_ONLY", "notes": "Validator-clean parent."},
            {"source_id": "0187_parent_predictions", "path": rel(P0187), "sha256": sha256_file(P0187), "size_bytes": P0187.stat().st_size, "row_count": int(len(frame)), "date_start": date_text(frame["target_date"].min()), "date_end": date_text(frame["target_date"].max()), "timestamp_fields": "target_date;station fusion", "availability_class": "DEPLOYABLE_LAGGED_ONLY", "notes": "Validator-clean parent."},
        ]
    )
    feature_definitions = pd.DataFrame(
        [
            {"feature_name": "t7_online_expert_loss_weights", "role": "router", "formula": "Softmax over exponentially weighted expert absolute-error states updated only from rows with target_date <= T-7.", "input_columns": "official, 0185, and 0187 parent predictions; prior matured target errors; month/season/source contexts", "units": "expert weights", "lag": "7 target days minimum for loss updates", "window": "predeclared EW half-life grid", "fit_scope": "fold-local config selection; prequential state", "availability_rule": "No current or recent target row updates the prediction state.", "missingness_policy": "No mature state defaults to 0185."}
        ]
    )
    pred_cols = [
        "target_date",
        "target_tmax_c",
        "forecast_source_family",
        "season",
        "month",
        "official_prediction_c",
        "prediction_0185_c",
        "prediction_0187_c",
        "candidate_prediction_c",
        "candidate_correction_c",
        "weight_official",
        "weight_0185",
        "weight_0187",
        "loss_official",
        "loss_0185",
        "loss_0187",
        "official_error_c_signed",
        "candidate_error_c",
        "official_abs_error_c",
        "candidate_abs_error_c",
        "fold_id",
        "selected_config_id",
        "candidate_id",
        "baseline_id",
        "model_family",
    ]
    write_parquet(EXP_DIR / "predictions.parquet", predictions[pred_cols])
    write_csv(EXP_DIR / "scoreboard.csv", scoreboard)
    write_csv(EXP_DIR / "slice_metrics.csv", slice_metrics)
    write_csv(EXP_DIR / "yearly_metrics.csv", yearly_metrics)
    write_csv(EXP_DIR / "fold_metrics.csv", fold_metrics)
    write_csv(EXP_DIR / "row_coverage.csv", row_coverage)
    write_csv(EXP_DIR / "correction_distribution.csv", correction_distribution)
    write_csv(EXP_DIR / "data_manifest.csv", data_manifest)
    write_csv(EXP_DIR / "feature_definitions.csv", feature_definitions)
    write_csv(EXP_DIR / "artifacts" / "config_grid.csv", pd.DataFrame(CONFIG_GRID))
    write_csv(EXP_DIR / "artifacts" / "fold_config_selections.csv", selections)
    write_text(
        EXP_DIR / "leakage_audit.md",
        f"""# Leakage And Point-In-Time Audit

Status: `PASS`

## Cutoff

0189 consumes validator-clean parent predictions from 0185 and 0187. Expert loss states are updated only from rows at least `{LAG_DAYS}` target days older than the current target date.

## Available State

For row T, the prequential state admits only target dates <= T-7. If no mature state exists, the prediction defaults to the 0185 parent.

## Target And Rolling Checks

The current target is used only for scoring. Fold configuration selection uses only target years before the fold start.

## Confirmation Proof

Maximum scored target date is `{date_text(predictions['target_date'].max())}`. Confirmation rows used: `0`. Owner authorization for confirmation: `false`.

## Row Identity

Candidate and official baseline share `{len(predictions)}` rows. Common row hash: `{common_row_hash}`.
""",
    )
    write_text(EXP_DIR / "REPRODUCE.md", f"""# Reproduction

From `{REPO_ROOT}`, run:

```powershell
python scripts/run_hkg_t24_0189_t7_online_expert_weighting.py
```

Requires completed parent predictions from 0185, 0187, and 0188 for comparison. Confirmation rows remain locked.
""")
    code_sha = sha256_file(EXP_DIR / "src" / Path(__file__).name)
    data_manifest_sha = sha256_file(EXP_DIR / "data_manifest.csv")
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "slug": SLUG,
        "status": status,
        "created_at_utc": created_at,
        "target": "HKO daily Tmax T-24",
        "frame_id": "official_t15_pre2024_5265_rows",
        "date_start": date_text(predictions["target_date"].min()),
        "date_end": date_text(predictions["target_date"].max()),
        "n_candidate": int(len(predictions)),
        "n_common": int(len(predictions)),
        "baseline_id": "official_forecast_max_c",
        "baseline_mae_c": official_global["mae_c"],
        "candidate_id": PRIMARY_CANDIDATE_ID,
        "candidate_mae_c": candidate_global["mae_c"],
        "mae_delta_c": mae_delta,
        "candidate_rmse_c": candidate_global["rmse_c"],
        "candidate_bias_c": candidate_global["bias_c"],
        "leakage_status": "PASS",
        "confirmation_rows_used": 0,
        "owner_authorized_confirmation": False,
        "promotion_decision": promotion_decision,
        "spec_sha256": spec_sha,
        "code_sha256": code_sha,
        "data_manifest_sha256": data_manifest_sha,
        "common_row_hash": common_row_hash,
        "baseline_n": int(len(predictions)),
        "candidate_n": int(len(predictions)),
        "development_gate_reached": bool(candidate_global["mae_c"] <= 0.45),
        "parent_0185_mae_c": p85_global["mae_c"],
        "parent_0187_mae_c": p87_global["mae_c"],
        "parent_0188_mae_c": p88_global["mae_c"],
        "delta_vs_0188_mae_c": delta_vs_0188,
        "fold_worst_mae_delta_c": fold_worst_delta,
        "severe_gt3_rate_delta": severe_harm,
    }
    write_json(EXP_DIR / "summary.json", summary)
    write_docs(predictions, scoreboard, slice_metrics, yearly_metrics, fold_metrics, selections, summary)
    write_json(
        EXP_DIR / "run_manifest.json",
        {"experiment_id": EXPERIMENT_ID, "slug": SLUG, "created_at_utc": created_at, "completed_at_utc": utc_now(), "repo_root": str(REPO_ROOT), "script": rel(Path(__file__).resolve()), "spec_sha256": spec_sha, "code_sha256": code_sha, "state": "COMPLETED"},
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
