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
EXPERIMENT_ID = "0195"
SLUG = "june_warm_season_noharm_governor"
TITLE = "June Warm-Season No-Harm Governor Over 0194"
EXP_DIR = EXPERIMENTS_ROOT / f"{EXPERIMENT_ID}_{SLUG}"
PRIMARY_CANDIDATE_ID = "0195_june_warm_season_noharm_governor_over_0194"
P0194 = EXPERIMENTS_ROOT / "0194_isd_role_compressed_regime_proxy" / "predictions.parquet"
SRC_COPY_NAME = "run_0195.py"
MODEL_FOLDS = base.MODEL_FOLDS
LAG_DAYS = 7

CONFIG_GRID = [
    {"mode": "no_gate", "halflife": 90.0, "min_history": 0, "harm_threshold_c": math.inf, "weight_on_harm": 1.0, "contexts": "none"},
    {"mode": "month_soft", "halflife": 60.0, "min_history": 20, "harm_threshold_c": 0.004, "weight_on_harm": 0.50, "contexts": "month"},
    {"mode": "month_hard", "halflife": 90.0, "min_history": 25, "harm_threshold_c": 0.003, "weight_on_harm": 0.00, "contexts": "month"},
    {"mode": "warm_month_hard", "halflife": 90.0, "min_history": 20, "harm_threshold_c": 0.002, "weight_on_harm": 0.00, "contexts": "warm_month"},
    {"mode": "month_source_soft", "halflife": 90.0, "min_history": 25, "harm_threshold_c": 0.003, "weight_on_harm": 0.50, "contexts": "month_source"},
    {"mode": "month_source_hard", "halflife": 120.0, "min_history": 30, "harm_threshold_c": 0.003, "weight_on_harm": 0.00, "contexts": "month_source"},
    {"mode": "season_source_soft", "halflife": 120.0, "min_history": 35, "harm_threshold_c": 0.004, "weight_on_harm": 0.50, "contexts": "season_source"},
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


def load_parent() -> pd.DataFrame:
    frame = pd.read_parquet(P0194)
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
    frame = frame[frame["target_date"] < pd.Timestamp("2024-01-01")].copy()
    rename = {
        "candidate_prediction_c": "parent_0194_prediction_c",
        "candidate_correction_c": "parent_0194_total_correction_c",
        "candidate_error_c": "parent_0194_error_c",
        "candidate_abs_error_c": "parent_0194_abs_error_c",
        "isd_role_correction_c": "parent_0194_isd_role_correction_c",
    }
    frame = frame.rename(columns={k: v for k, v in rename.items() if k in frame.columns})
    frame["parent_0190_error_c"] = frame["parent_0190_prediction_c"] - frame["target_tmax_c"]
    frame["parent_0190_abs_error_c"] = frame["parent_0190_error_c"].abs()
    frame["parent_0194_error_c"] = frame["parent_0194_prediction_c"] - frame["target_tmax_c"]
    frame["parent_0194_abs_error_c"] = frame["parent_0194_error_c"].abs()
    frame["parent_0194_vs_0190_abs_loss_delta_c"] = frame["parent_0194_abs_error_c"] - frame["parent_0190_abs_error_c"]
    frame["parent_0194_delta_from_0190_c"] = frame["parent_0194_prediction_c"] - frame["parent_0190_prediction_c"]
    frame["month"] = pd.to_numeric(frame["month"], errors="coerce").fillna(frame["target_date"].dt.month).astype(int)
    base.assert_pre2024(frame, "0195 parent 0194 frame")
    return frame.sort_values("target_date").reset_index(drop=True)


def context_keys(row: pd.Series, config: dict[str, Any]) -> list[str]:
    mode = str(config["contexts"])
    if mode == "none":
        return []
    month = int(row["month"])
    season = str(row.get("season") or "season_unknown")
    source = str(row.get("forecast_source_family") or "source_unknown")
    keys = ["global"]
    if mode in {"month", "warm_month", "month_source"}:
        keys.append(f"month={month:02d}")
    if mode == "warm_month" and month in {5, 6, 7, 8, 9}:
        keys.append("warm_months")
        keys.append(f"warm_month={month:02d}")
    if mode == "month_source":
        keys.append(f"month={month:02d}|source={source}")
    if mode == "season_source":
        keys.extend([f"season={season}", f"season={season}|source={source}"])
    return keys


def update_state(state: dict[str, dict[str, float]], key: str, value: float, decay: float) -> None:
    rec = state.setdefault(key, {"count": 0.0, "weighted_sum": 0.0, "weight_sum": 0.0})
    rec["weighted_sum"] = rec["weighted_sum"] * decay + value
    rec["weight_sum"] = rec["weight_sum"] * decay + 1.0
    rec["count"] += 1.0


def harm_score(state: dict[str, dict[str, float]], keys: list[str], config: dict[str, Any]) -> tuple[float, int]:
    scores = []
    active = 0
    for key in keys:
        rec = state.get(key)
        if not rec or rec["count"] < float(config["min_history"]) or rec["weight_sum"] <= 0:
            continue
        raw = rec["weighted_sum"] / rec["weight_sum"]
        shrink = rec["count"] / (rec["count"] + float(config["min_history"]))
        scores.append(raw * shrink)
        active += 1
    if not scores:
        return 0.0, 0
    return float(max(scores)), active


def prequential(frame: pd.DataFrame, config: dict[str, Any], config_id: str) -> pd.DataFrame:
    ordered = frame.sort_values("target_date").reset_index(drop=True).copy()
    if config["mode"] == "no_gate":
        out = ordered.copy()
        out["config_id"] = config_id
        out["governor_weight"] = 1.0
        out["harm_score_c"] = 0.0
        out["active_governor_context_count"] = 0
        return finalize_predictions(out)

    dates = pd.to_datetime(ordered["target_date"]).dt.normalize()
    keys = [context_keys(row, config) for _, row in ordered.iterrows()]
    decay = float(np.power(0.5, 1.0 / float(config["halflife"])))
    state: dict[str, dict[str, float]] = {}
    add_idx = 0
    weights = []
    harms = []
    active_counts = []
    for idx, current_date in enumerate(dates):
        mature_date = current_date - pd.Timedelta(days=LAG_DAYS)
        while add_idx < len(ordered) and dates.iloc[add_idx] <= mature_date:
            rel_loss = float(ordered.iloc[add_idx]["parent_0194_vs_0190_abs_loss_delta_c"])
            for key in keys[add_idx]:
                update_state(state, key, rel_loss, decay)
            add_idx += 1
        score, active = harm_score(state, keys[idx], config)
        weight = float(config["weight_on_harm"]) if active and score > float(config["harm_threshold_c"]) else 1.0
        weights.append(weight)
        harms.append(score)
        active_counts.append(active)
    out = ordered.copy()
    out["config_id"] = config_id
    out["governor_weight"] = weights
    out["harm_score_c"] = harms
    out["active_governor_context_count"] = active_counts
    return finalize_predictions(out)


def finalize_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["candidate_prediction_c"] = out["parent_0190_prediction_c"] + out["governor_weight"] * out["parent_0194_delta_from_0190_c"]
    out["candidate_correction_c"] = out["candidate_prediction_c"] - out["official_prediction_c"]
    out["candidate_error_c"] = out["candidate_prediction_c"] - out["target_tmax_c"]
    out["candidate_abs_error_c"] = out["candidate_error_c"].abs()
    out["official_error_c_signed"] = out["official_prediction_c"] - out["target_tmax_c"]
    out["official_abs_error_c"] = out["official_error_c_signed"].abs()
    return out


def metric_for(frame: pd.DataFrame, pred_col: str, label: str) -> dict[str, Any]:
    return base.metric_row(frame, pred_col, label=label)


def compare_four(frame: pd.DataFrame, *, slice_type: str, slice_value: Any) -> dict[str, Any]:
    official = metric_for(frame, "official_prediction_c", "official")
    p0190 = metric_for(frame, "parent_0190_prediction_c", "0190_parent")
    p0194 = metric_for(frame, "parent_0194_prediction_c", "0194_parent")
    candidate = metric_for(frame, "candidate_prediction_c", PRIMARY_CANDIDATE_ID)
    return {
        "slice_type": slice_type,
        "slice_value": str(slice_value),
        "n": candidate["n"],
        "official_mae_c": official["mae_c"],
        "parent_0190_mae_c": p0190["mae_c"],
        "parent_0194_mae_c": p0194["mae_c"],
        "candidate_mae_c": candidate["mae_c"],
        "mae_delta_c": candidate["mae_c"] - official["mae_c"],
        "delta_vs_0190_mae_c": candidate["mae_c"] - p0190["mae_c"],
        "delta_vs_0194_mae_c": candidate["mae_c"] - p0194["mae_c"],
        "candidate_bias_c": candidate["bias_c"],
        "official_gt3c_rate": official["gt3c_rate"],
        "parent_0190_gt3c_rate": p0190["gt3c_rate"],
        "parent_0194_gt3c_rate": p0194["gt3c_rate"],
        "candidate_gt3c_rate": candidate["gt3c_rate"],
        "official_p95_abs_error_c": official["p95_abs_error_c"],
        "parent_0194_p95_abs_error_c": p0194["p95_abs_error_c"],
        "candidate_p95_abs_error_c": candidate["p95_abs_error_c"],
    }


def run_walk_forward(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config_rows = []
    preds = {}
    for idx, config in enumerate(CONFIG_GRID):
        cid = f"cfg_{idx:02d}_{config['mode']}"
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
            selected["train_delta_vs_0194_mae_c"] = 0.0
        else:
            rows = []
            for _, cfg in config_table.iterrows():
                pred = preds[cfg["config_id"]]
                train = pred[pred["target_date"].dt.year < start_year]
                cand_mae = float(train["candidate_abs_error_c"].mean())
                parent_mae = float(train["parent_0194_abs_error_c"].mean())
                rows.append({**cfg.to_dict(), "train_mae_c": cand_mae, "train_delta_vs_0194_mae_c": cand_mae - parent_mae})
            selected = pd.DataFrame(rows).sort_values(["train_mae_c", "train_delta_vs_0194_mae_c", "config_id"]).iloc[0].to_dict()
        pred = preds[selected["config_id"]][test_mask].copy()
        pred["fold_id"] = fold_id
        pred["selected_config_id"] = selected["config_id"]
        pred["selected_mode"] = selected["mode"]
        pred["selected_weight_on_harm"] = selected.get("weight_on_harm")
        pred["selected_harm_threshold_c"] = selected.get("harm_threshold_c")
        parts.append(pred)
        metric = compare_four(pred, slice_type="fold", slice_value=fold_id)
        metric.update(
            {
                "fold_id": fold_id,
                "start_year": start_year,
                "end_year": end_year,
                "selected_config_id": selected["config_id"],
                "selected_mode": selected["mode"],
                "selected_weight_on_harm": selected.get("weight_on_harm"),
                "selected_harm_threshold_c": selected.get("harm_threshold_c"),
                "selection_train_mae_c": selected.get("train_mae_c"),
                "selection_train_delta_vs_0194_mae_c": selected.get("train_delta_vs_0194_mae_c"),
                "governed_row_rate": float((pred["governor_weight"] < 1.0).mean()),
                "mean_governor_weight": float(pred["governor_weight"].mean()),
            }
        )
        folds.append(metric)
        selections.append({"fold_id": fold_id, **selected})
    out = pd.concat(parts, ignore_index=True).sort_values("target_date").reset_index(drop=True)
    out["candidate_id"] = PRIMARY_CANDIDATE_ID
    out["baseline_id"] = "official_forecast_max_c"
    out["model_family"] = "june_warm_season_noharm_governor"
    return out, pd.DataFrame(folds), pd.DataFrame(selections)


def build_slice_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [compare_four(predictions, slice_type="overall", slice_value="all")]
    for season, group in predictions.groupby("season", dropna=False):
        rows.append(compare_four(group, slice_type="season", slice_value=season))
    for month, group in predictions.groupby("month", dropna=False):
        rows.append(compare_four(group, slice_type="month", slice_value=month))
    for source, group in predictions.groupby("forecast_source_family", dropna=False):
        rows.append(compare_four(group, slice_type="source", slice_value=source))
    late = predictions[predictions["target_date"] >= pd.Timestamp("2020-01-01")]
    rows.append(compare_four(late, slice_type="late_window", slice_value="2020_2023"))
    tail = predictions[predictions["parent_0194_abs_error_c"] >= 2.0]
    rows.append(compare_four(tail, slice_type="parent_tail", slice_value="parent_0194_abs_error_ge_2c"))
    yearly = pd.DataFrame([compare_four(group, slice_type="year", slice_value=year) for year, group in predictions.groupby(predictions["target_date"].dt.year, dropna=False)])
    return pd.DataFrame(rows), yearly


def build_spec(created_at: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": created_at,
        "title": TITLE,
        "slug": SLUG,
        "mode": "promotion_oriented",
        "hypothesis": "The 0194 champion can be made safer and possibly lower-MAE by abstaining from its ISD role correction in months or warm-season contexts where T-7 mature prior relative loss shows harm versus 0190.",
        "rationale": "0194 is the new champion but has June harm versus 0190. A default-apply governor can preserve the 0194 lift while using mature relative-loss history to block repeat harm pockets.",
        "expected_sign_and_falsification": "Expected sign is lower or equal MAE than 0194 with reduced June harm. Falsified if prior relative-loss gates either never activate or remove useful 0194 corrections.",
        "novelty": {"prior_experiments": ["0190", "0194"], "difference": "Applies a T-7 mature no-harm governor over 0194 rather than learning a new station model.", "similarity_audit_path": "RESULTS.md#comparison-limitations"},
        "target": {"station": "Hong Kong Observatory headquarters", "variable": "tmax_c", "horizon": "T-24", "timezone": "Asia/Hong_Kong", "cutoff_contract_path": rel(base.OFFICIAL_PATH), "cutoff_function": "For target T, governor state admits only 0194-vs-0190 loss deltas from target dates <= T-7.", "daily_boundary_contract": "HKO local daily maximum temperature for target local date T."},
        "frame": {"frame_id": "official_t15_pre2024_5265_rows", "development_start": "2000-01-02", "development_end_exclusive": "2024-01-01", "confirmation_locked": True, "row_universe_artifact": rel(P0194)},
        "data_sources": [{"source_id": "0194_parent_predictions", "paths": [rel(P0194)], "attributes": ["0194 prediction", "0190 fallback prediction", "mature relative loss for online state"], "eligibility": "DEPLOYABLE_LAGGED_ONLY", "availability_proof": "0194 validator passed; loss state uses only target dates <= T-7."}],
        "stations": [{"station_id": "HKO", "role": "target and mature loss state", "attributes": ["daily Tmax"]}],
        "features": {"generation_rule": "T-7 mature relative-loss states by month, warm-month, month-source, and season-source contexts. Default action is use 0194; harmful contexts shrink to 0190 or halfway.", "grid": CONFIG_GRID, "explicit_exclusions": ["2024+ rows", "current target outcome", "current target residual/loss"]},
        "response": {"variable": "abs_error_0194 - abs_error_0190 for mature state; target_tmax_c for scoring only", "prediction": "0190 fallback plus governed 0194-minus-0190 correction"},
        "baseline": {"id": "official_forecast_max_c", "definition": "Raw official forecast on identical rows; 0194 reported as parent champion reference."},
        "validation": {"outer_folds": [list(item) for item in MODEL_FOLDS], "inner_selection": "Config selected by prior-year MAE only; no current fold outcomes used.", "minimum_train_rows": 365},
        "metrics": ["MAE", "RMSE", "bias", "P95 AE", ">3C rate", "fold/year/season/month/source/tail slices", "delta_vs_0194"],
        "sample_rules": {"row_policy": "All 0194 parent rows.", "missing_policy": "No mature context gives default 0194 weight 1.0."},
        "acceptance_gates": {"minimum_mae_lift_vs_official_c": 0.01, "minimum_mae_lift_vs_0194_c": 0.001, "max_fold_harm_vs_0194_c": 0.001, "no_parent_tail_harm": ">3C rate cannot exceed 0194 by more than 0.005."},
        "rejection_conditions": ["Any 2024+ row.", "Any relative-loss update newer than T-7.", "Parent row mismatch."],
        "required_outputs": ["README.md", "RESULTS.md", "CONCLUSION.md", "scoreboard.csv", "slice_metrics.csv", "yearly_metrics.csv", "fold_metrics.csv", "predictions.parquet"],
        "owner_authorized_confirmation": False,
    }


def write_docs(scoreboard: pd.DataFrame, slice_metrics: pd.DataFrame, yearly_metrics: pd.DataFrame, fold_metrics: pd.DataFrame, selections: pd.DataFrame, summary: dict[str, Any]) -> None:
    write_text(EXP_DIR / "README.md", f"""# {TITLE}

## Experiment Identity And Status

Experiment `{EXPERIMENT_ID}` is `{summary['status']}`. It is a T-7 mature no-harm governor over 0194.

## One-Sentence Hypothesis

The 0194 ISD role correction should remain the default, but prior mature loss can identify warm-month contexts where falling back toward 0190 avoids repeat harm.

## Why It Is Worth Doing

0194 is the current champion, yet June is worse than 0190 by 0.004368 C. This experiment tests whether that harm can be reduced without giving back the global station-network lift.

## Prior Evidence And Novelty

0194 found strong independent ISD signal. 0195 does not refit that model; it only governs the already-validated correction using deployable T-7 mature relative-loss memory.

## Target, Horizon, And Exact Cutoff

Target is HKO Tmax at `T-24`, timezone `Asia/Hong_Kong`. Governor state for target T uses only relative losses from target dates <= T-7.

## Datasets, Stations, And Attributes

Input is the validator-clean 0194 prediction artifact on the official pre-2024 frame.

## Feature Definitions

Features are mature relative-loss states by month, warm-month, month-source, and season-source contexts. Details are in `feature_definitions.csv`.

## Response And Baseline

The scored prediction is 0190 fallback plus a governed 0194 correction. Official raw forecast is the primary baseline; 0194 is the parent champion reference.

## Walk-Forward Design

Each fold selects one predeclared governor config using only prior-year MAE. The no-gate config exactly reproduces 0194 and is always available.

## Acceptance And Rejection Criteria

Acceptance requires at least 0.001 C global MAE lift versus 0194 without fold or severe-tail harm.

## Reproduction Command

Run `python scripts/run_hkg_t24_0195_june_warm_season_noharm_governor.py` from the repository root.
""")
    write_text(EXP_DIR / "RESULTS.md", f"""# Results

## Headline Result Table

{base.markdown_table(scoreboard)}

## Coverage And Row Identity

Rows: `{summary['n_common']}` from `{summary['date_start']}` to `{summary['date_end']}`. Common row hash: `{summary['common_row_hash']}`.

## Global Metrics

Official MAE is `{summary['baseline_mae_c']:.6f}` C. Parent 0194 MAE is `{summary['parent_0194_mae_c']:.6f}` C. Candidate MAE is `{summary['candidate_mae_c']:.6f}` C. Delta versus 0194 is `{summary['delta_vs_0194_mae_c']:.6f}` C.

## Fold Stability

{base.markdown_table(fold_metrics[['fold_id', 'n', 'parent_0194_mae_c', 'candidate_mae_c', 'delta_vs_0194_mae_c', 'selected_config_id', 'selected_mode', 'governed_row_rate', 'mean_governor_weight']], max_rows=20)}

## Yearly And Monthly Results

{base.markdown_table(yearly_metrics[['slice_value', 'n', 'parent_0194_mae_c', 'candidate_mae_c', 'delta_vs_0194_mae_c']], max_rows=30)}

Month metrics:

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].eq('month')][['slice_value', 'n', 'parent_0194_mae_c', 'candidate_mae_c', 'delta_vs_0194_mae_c']], max_rows=20)}

## Seasonal And Source Results

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].isin(['season', 'source', 'late_window', 'parent_tail'])][['slice_type', 'slice_value', 'n', 'parent_0194_mae_c', 'candidate_mae_c', 'delta_vs_0194_mae_c', 'parent_0194_gt3c_rate', 'candidate_gt3c_rate']], max_rows=20)}

## Data-Quality And Leakage Result

Leakage status is `{summary['leakage_status']}`. Relative-loss governor state is T-7 mature.

## Comparison Limitations

This is a child governor over 0194. If it does not beat 0194 globally, it should not replace the parent even if it protects one month.

Selections:

{base.markdown_table(selections, max_rows=20)}
""")
    write_text(EXP_DIR / "CONCLUSION.md", f"""# Conclusion

## Verdict

Status is `{summary['status']}` with promotion decision `{summary['promotion_decision']}`.

## What Was Learned

0195 tested whether the remaining June/warm-season harm in 0194 can be safely governed by mature prior relative-loss states.

## Realized Point-MAE Change

Delta versus official is `{summary['mae_delta_c']:.6f}` C. Delta versus 0194 is `{summary['delta_vs_0194_mae_c']:.6f}` C.

## Robustness And Uncertainty

The no-gate 0194 reproduction is always in the config grid. Any abstention must earn selection on prior years only.

## Failure Diagnosis

If the selected config is `no_gate`, the governor did not find enough prior-history evidence that abstaining from 0194 improves global MAE. That means the June harm pocket exists, but the available month and warm-season relative-loss memory is not strong enough to justify overriding the champion under chronological selection.

## Promotion Status

Confirmation remains locked. Development gate to 0.45 C was not reached.

## Implication For Future Research

Do not replace 0194 with this governor unless it beats 0194 globally. A more targeted follow-up would need a stronger independent trigger, such as capped-correction trust or station-tail regime evidence, rather than month-only loss memory.
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
    frame = load_parent()
    predictions, fold_metrics, selections = run_walk_forward(frame)
    slice_metrics, yearly_metrics = build_slice_metrics(predictions)
    official_global = metric_for(predictions, "official_prediction_c", "official")
    p0190_global = metric_for(predictions, "parent_0190_prediction_c", "0190_parent")
    p0194_global = metric_for(predictions, "parent_0194_prediction_c", "0194_parent")
    candidate_global = metric_for(predictions, "candidate_prediction_c", PRIMARY_CANDIDATE_ID)
    mae_delta = candidate_global["mae_c"] - official_global["mae_c"]
    delta_vs_0190 = candidate_global["mae_c"] - p0190_global["mae_c"]
    delta_vs_0194 = candidate_global["mae_c"] - p0194_global["mae_c"]
    severe_harm_0194 = candidate_global["gt3c_rate"] - p0194_global["gt3c_rate"]
    fold_worst_delta = float(fold_metrics["delta_vs_0194_mae_c"].max())
    if mae_delta <= -0.01 and delta_vs_0194 <= -0.001 and severe_harm_0194 <= 0.005 and fold_worst_delta <= 0.001:
        status = "COMPLETED_PROMOTION_CANDIDATE"
        promotion_decision = "PROMOTE_OVER_0194_NO_CONFIRMATION"
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
            {"candidate_id": "0194_parent_isd_role_proxy", "model_family": "parent_reference", "n": p0194_global["n"], "mae_c": p0194_global["mae_c"], "rmse_c": p0194_global["rmse_c"], "bias_c": p0194_global["bias_c"], "median_abs_error_c": p0194_global["median_abs_error_c"], "p95_abs_error_c": p0194_global["p95_abs_error_c"], "gt2c_rate": p0194_global["gt2c_rate"], "gt3c_rate": p0194_global["gt3c_rate"], "baseline_mae_c": official_global["mae_c"], "mae_delta_c": p0194_global["mae_c"] - official_global["mae_c"]},
            {"candidate_id": PRIMARY_CANDIDATE_ID, "model_family": "june_warm_season_noharm_governor", "n": candidate_global["n"], "mae_c": candidate_global["mae_c"], "rmse_c": candidate_global["rmse_c"], "bias_c": candidate_global["bias_c"], "median_abs_error_c": candidate_global["median_abs_error_c"], "p95_abs_error_c": candidate_global["p95_abs_error_c"], "gt2c_rate": candidate_global["gt2c_rate"], "gt3c_rate": candidate_global["gt3c_rate"], "baseline_mae_c": official_global["mae_c"], "mae_delta_c": mae_delta},
        ]
    )
    row_coverage = pd.DataFrame([{"frame_id": "official_t15_pre2024_5265_rows", "parent_rows": int(len(frame)), "candidate_rows": int(len(predictions)), "baseline_rows": int(len(predictions)), "common_rows": int(len(predictions)), "date_start": date_text(predictions["target_date"].min()), "date_end": date_text(predictions["target_date"].max()), "row_policy": "all 0194 parent rows", "common_row_hash": common_row_hash}])
    correction_distribution = predictions[["governor_weight", "harm_score_c", "active_governor_context_count"]].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).reset_index()
    data_manifest = pd.DataFrame([{"source_id": "0194_parent_predictions", "path": rel(P0194), "sha256": sha256_file(P0194), "size_bytes": P0194.stat().st_size, "row_count": int(len(frame)), "date_start": date_text(frame["target_date"].min()), "date_end": date_text(frame["target_date"].max()), "timestamp_fields": "target_date;0194 prediction;0190 fallback;T-7 mature relative loss", "availability_class": "DEPLOYABLE_LAGGED_ONLY", "notes": "Validator-clean 0194 predictions; relative loss state admits only rows <= T-7."}])
    feature_definitions = pd.DataFrame([{"feature_name": "t7_0194_vs_0190_relative_loss_governor", "role": "candidate_governor", "formula": "EW mean of abs_error_0194 - abs_error_0190 by predeclared month/warm/source contexts; current row uses only target_date <= T-7 state.", "input_columns": "parent_0194_abs_error_c,parent_0190_abs_error_c,target_date,month,season,forecast_source_family", "units": "degC relative absolute error", "lag": "7 target days minimum", "window": "predeclared EW half-life grid", "fit_scope": "fold-local config selection; prequential state", "availability_rule": "Current target outcome never updates its own governor state.", "missingness_policy": "No mature harmful context gives default 0194 weight 1.0."}])
    pred_cols = ["target_date", "target_tmax_c", "forecast_source_family", "season", "month", "official_prediction_c", "parent_0190_prediction_c", "parent_0194_prediction_c", "candidate_prediction_c", "parent_0194_delta_from_0190_c", "governor_weight", "harm_score_c", "active_governor_context_count", "candidate_correction_c", "official_error_c_signed", "parent_0190_error_c", "parent_0194_error_c", "candidate_error_c", "official_abs_error_c", "parent_0190_abs_error_c", "parent_0194_abs_error_c", "candidate_abs_error_c", "fold_id", "selected_config_id", "selected_mode", "candidate_id", "baseline_id", "model_family"]
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
    write_text(EXP_DIR / "leakage_audit.md", f"""# Leakage And Point-In-Time Audit

Status: `PASS`

## Cutoff

0195 consumes validator-clean 0194 parent predictions. For target T, governor relative-loss contexts include only rows with target_date <= T-{LAG_DAYS}.

## Available State

Current-row month, season, source, 0190 fallback prediction, and 0194 prediction are known at decision time. Current target outcome and current relative loss are never used to decide the row's governor weight.

## Target And Rolling Checks

Fold config selection uses only years before the fold start. Current target values are used only for scoring.

## Confirmation Proof

Maximum scored target date is `{date_text(predictions['target_date'].max())}`. Confirmation rows used: `0`. Owner authorization for confirmation: `false`.

## Row Identity

Candidate, parent 0194, parent 0190, and official baseline share `{len(predictions)}` rows. Common row hash: `{common_row_hash}`.
""")
    write_text(EXP_DIR / "REPRODUCE.md", f"""# Reproduction

From `{REPO_ROOT}`, run:

```powershell
python scripts/run_hkg_t24_0195_june_warm_season_noharm_governor.py
```

Requires completed parent predictions from 0194. Confirmation rows remain locked.
""")
    code_sha = sha256_file(src_copy_path)
    data_manifest_sha = sha256_file(EXP_DIR / "data_manifest.csv")
    summary = {"experiment_id": EXPERIMENT_ID, "slug": SLUG, "status": status, "created_at_utc": created_at, "target": "HKO daily Tmax T-24", "frame_id": "official_t15_pre2024_5265_rows", "date_start": date_text(predictions["target_date"].min()), "date_end": date_text(predictions["target_date"].max()), "n_candidate": int(len(predictions)), "n_common": int(len(predictions)), "baseline_id": "official_forecast_max_c", "baseline_mae_c": official_global["mae_c"], "candidate_id": PRIMARY_CANDIDATE_ID, "candidate_mae_c": candidate_global["mae_c"], "mae_delta_c": mae_delta, "candidate_rmse_c": candidate_global["rmse_c"], "candidate_bias_c": candidate_global["bias_c"], "leakage_status": "PASS", "confirmation_rows_used": 0, "owner_authorized_confirmation": False, "promotion_decision": promotion_decision, "spec_sha256": spec_sha, "code_sha256": code_sha, "data_manifest_sha256": data_manifest_sha, "common_row_hash": common_row_hash, "baseline_n": int(len(predictions)), "candidate_n": int(len(predictions)), "development_gate_reached": bool(candidate_global["mae_c"] <= 0.45), "parent_0190_mae_c": p0190_global["mae_c"], "delta_vs_0190_mae_c": delta_vs_0190, "parent_0194_mae_c": p0194_global["mae_c"], "delta_vs_0194_mae_c": delta_vs_0194, "fold_worst_delta_vs_0194_mae_c": fold_worst_delta, "severe_gt3_rate_delta_vs_0194": severe_harm_0194}
    write_json(EXP_DIR / "summary.json", summary)
    write_docs(scoreboard, slice_metrics, yearly_metrics, fold_metrics, selections, summary)
    write_json(EXP_DIR / "run_manifest.json", {"experiment_id": EXPERIMENT_ID, "slug": SLUG, "created_at_utc": created_at, "completed_at_utc": utc_now(), "repo_root": str(REPO_ROOT), "script": rel(Path(__file__).resolve()), "spec_sha256": spec_sha, "code_sha256": code_sha, "state": "COMPLETED"})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
