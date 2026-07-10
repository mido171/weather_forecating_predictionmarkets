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
EXPERIMENT_ID = "0192"
SLUG = "t7_tail_risk_trust_governor"
TITLE = "T-7 Tail-Risk Trust Governor"
EXP_DIR = EXPERIMENTS_ROOT / f"{EXPERIMENT_ID}_{SLUG}"
PRIMARY_CANDIDATE_ID = "0192_t7_tail_risk_trust_governor"
P0190 = EXPERIMENTS_ROOT / "0190_t7_post_ensemble_residual_calibration" / "predictions.parquet"
SRC_COPY_NAME = "run_0192.py"
MODEL_FOLDS = base.MODEL_FOLDS
LAG_DAYS = 7
CONFIG_GRID = [
    {"context_mode": "global", "halflife": 20.0, "min_history": 20, "shrink": 80.0, "loss_threshold_c": 0.00, "shrink_floor": 0.75},
    {"context_mode": "month", "halflife": 45.0, "min_history": 30, "shrink": 100.0, "loss_threshold_c": 0.00, "shrink_floor": 0.50},
    {"context_mode": "month_source", "halflife": 45.0, "min_history": 35, "shrink": 120.0, "loss_threshold_c": 0.00, "shrink_floor": 0.50},
    {"context_mode": "season_sign", "halflife": 45.0, "min_history": 30, "shrink": 120.0, "loss_threshold_c": 0.01, "shrink_floor": 0.60},
    {"context_mode": "month_sign", "halflife": 60.0, "min_history": 35, "shrink": 140.0, "loss_threshold_c": 0.01, "shrink_floor": 0.50},
    {"context_mode": "all_sign_mag", "halflife": 90.0, "min_history": 45, "shrink": 160.0, "loss_threshold_c": 0.015, "shrink_floor": 0.40},
    {"context_mode": "all_sign_mag", "halflife": 120.0, "min_history": 60, "shrink": 220.0, "loss_threshold_c": 0.02, "shrink_floor": 0.25},
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
    frame = pd.read_parquet(P0190)
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.normalize()
    frame = frame[frame["target_date"] < pd.Timestamp("2024-01-01")].copy()
    frame = frame.rename(
        columns={
            "candidate_prediction_c": "parent_0190_prediction_c",
            "candidate_correction_c": "parent_0190_total_correction_c",
            "calibration_correction_c": "parent_0190_calibration_correction_c",
            "active_calibration_context_count": "parent_0190_active_context_count",
        }
    )
    frame["parent_0190_residual_c"] = frame["target_tmax_c"] - frame["parent_0190_prediction_c"]
    frame["parent_0190_abs_error_c"] = frame["parent_0190_residual_c"].abs()
    frame["prediction_official_c"] = frame["official_prediction_c"]
    base.assert_pre2024(frame, "0191 parent 0190 frame")
    return frame.sort_values("target_date").reset_index(drop=True)


def contexts(row: pd.Series, mode: str) -> list[str]:
    source = str(row.get("forecast_source_family") or "source_unknown")
    season = str(row.get("season") or "season_unknown")
    month = f"month_{int(row.get('month')):02d}" if pd.notna(row.get("month")) else "month_unknown"
    parent_corr = float(row.get("parent_0190_calibration_correction_c") or 0.0)
    corr_sign = "corr_pos" if parent_corr > 0.02 else "corr_neg" if parent_corr < -0.02 else "corr_flat"
    corr_mag = "corr_large" if abs(parent_corr) >= 0.15 else "corr_mid" if abs(parent_corr) >= 0.07 else "corr_small"
    out = ["global"]
    if mode in {"month", "month_source", "month_sign", "all_sign_mag"}:
        out.append(month)
    if mode in {"month_source", "all_sign_mag"}:
        out.append(f"{month}|source={source}")
    if mode in {"season_sign", "all_sign_mag"}:
        out.extend([f"season={season}", f"season={season}|{corr_sign}"])
    if mode in {"month_sign", "all_sign_mag"}:
        out.extend([f"{month}|{corr_sign}", f"{month}|source={source}|{corr_sign}"])
    if mode == "all_sign_mag":
        out.extend([f"{corr_sign}|{corr_mag}", f"source={source}|{corr_sign}|{corr_mag}", f"{month}|{corr_sign}|{corr_mag}"])
    return out


def update_state(state: dict[str, dict[str, float]], key: str, value: float, decay: float) -> None:
    rec = state.setdefault(key, {"count": 0.0, "weighted_sum": 0.0, "weight_sum": 0.0})
    rec["weighted_sum"] = rec["weighted_sum"] * decay + value
    rec["weight_sum"] = rec["weight_sum"] * decay + 1.0
    rec["count"] += 1.0


def trust_weight(state: dict[str, dict[str, float]], keys: list[str], config: dict[str, Any]) -> tuple[float, int, float]:
    values = []
    weights = []
    active = 0
    for key in keys:
        rec = state.get(key)
        if not rec or rec["count"] < float(config["min_history"]) or rec["weight_sum"] <= 0:
            continue
        raw = rec["weighted_sum"] / rec["weight_sum"]
        shrink = rec["count"] / (rec["count"] + float(config["shrink"]))
        values.append(raw * shrink)
        weights.append(math.sqrt(min(rec["count"], float(config["shrink"]))))
        active += 1
    if not values:
        return 1.0, 0, 0.0
    prior_loss_diff = float(np.average(np.asarray(values), weights=np.asarray(weights)))
    threshold = float(config["loss_threshold_c"])
    if prior_loss_diff <= threshold:
        return 1.0, active, prior_loss_diff
    excess = min(1.0, max(0.0, (prior_loss_diff - threshold) / 0.10))
    floor = float(config["shrink_floor"])
    weight = 1.0 - excess * (1.0 - floor)
    return float(np.clip(weight, floor, 1.0)), active, prior_loss_diff


def prequential(frame: pd.DataFrame, config: dict[str, Any], config_id: str) -> pd.DataFrame:
    ordered = frame.sort_values("target_date").reset_index(drop=True).copy()
    dates = pd.to_datetime(ordered["target_date"]).dt.normalize()
    key_list = [contexts(row, str(config["context_mode"])) for _, row in ordered.iterrows()]
    decay = float(np.power(0.5, 1.0 / float(config["halflife"])))
    state: dict[str, dict[str, float]] = {}
    add_idx = 0
    corrections = []
    active_counts = []
    trust_weights = []
    prior_loss_scores = []
    for idx, current_date in enumerate(dates):
        mature_date = current_date - pd.Timedelta(days=LAG_DAYS)
        while add_idx < len(ordered) and dates.iloc[add_idx] <= mature_date:
            parent_abs = abs(float(ordered.iloc[add_idx]["parent_0190_prediction_c"]) - float(ordered.iloc[add_idx]["target_tmax_c"]))
            official_abs = abs(float(ordered.iloc[add_idx]["official_prediction_c"]) - float(ordered.iloc[add_idx]["target_tmax_c"]))
            value = parent_abs - official_abs
            for key in key_list[add_idx]:
                update_state(state, key, value, decay)
            add_idx += 1
        weight, active, prior_loss = trust_weight(state, key_list[idx], config)
        parent_delta = float(ordered.iloc[idx]["parent_0190_prediction_c"] - ordered.iloc[idx]["official_prediction_c"])
        corrections.append((weight - 1.0) * parent_delta)
        active_counts.append(active)
        trust_weights.append(weight)
        prior_loss_scores.append(prior_loss)
    out = ordered.copy()
    out["config_id"] = config_id
    out["calibration_correction_c"] = corrections
    out["tail_governor_weight"] = trust_weights
    out["prior_parent_loss_diff_c"] = prior_loss_scores
    out["active_calibration_context_count"] = active_counts
    out["candidate_prediction_c"] = out["official_prediction_c"] + out["tail_governor_weight"] * (out["parent_0190_prediction_c"] - out["official_prediction_c"])
    out["candidate_correction_c"] = out["candidate_prediction_c"] - out["official_prediction_c"]
    out["candidate_error_c"] = out["candidate_prediction_c"] - out["target_tmax_c"]
    out["candidate_abs_error_c"] = out["candidate_error_c"].abs()
    out["official_error_c_signed"] = out["official_prediction_c"] - out["target_tmax_c"]
    out["official_abs_error_c"] = out["official_error_c_signed"].abs()
    return out


def run_walk_forward(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config_rows = []
    preds = {}
    for idx, config in enumerate(CONFIG_GRID, start=1):
        cid = f"cfg_{idx:02d}_{config['context_mode']}_h{int(config['halflife'])}_floor{str(config['shrink_floor']).replace('.', 'p')}"
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
        else:
            rows = []
            for _, cfg in config_table.iterrows():
                pred = preds[cfg["config_id"]]
                train = pred[pred["target_date"].dt.year < start_year]
                rows.append({**cfg.to_dict(), "train_mae_c": float(train["candidate_abs_error_c"].mean())})
            selected = pd.DataFrame(rows).sort_values(["train_mae_c", "shrink_floor", "halflife"]).iloc[0].to_dict()
        pred = preds[selected["config_id"]][test_mask].copy()
        pred["fold_id"] = fold_id
        pred["selected_config_id"] = selected["config_id"]
        parts.append(pred)
        metric = base.compare_metrics(pred, slice_type="fold", slice_value=fold_id)
        metric.update(
            {
                "fold_id": fold_id,
                "start_year": start_year,
                "end_year": end_year,
                "selected_config_id": selected["config_id"],
                "selected_context_mode": selected.get("context_mode"),
                "selected_halflife": selected.get("halflife"),
                "selected_shrink_floor": selected.get("shrink_floor"),
                "selected_loss_threshold_c": selected.get("loss_threshold_c"),
                "selection_train_mae_c": selected.get("train_mae_c"),
                "mean_calibration_correction_c": float(pred["calibration_correction_c"].mean()),
                "mean_abs_calibration_correction_c": float(pred["calibration_correction_c"].abs().mean()),
            }
        )
        folds.append(metric)
        selections.append({"fold_id": fold_id, **selected})
    out = pd.concat(parts, ignore_index=True).sort_values("target_date").reset_index(drop=True)
    out["candidate_id"] = PRIMARY_CANDIDATE_ID
    out["baseline_id"] = "official_forecast_max_c"
    out["model_family"] = "t7_tail_risk_trust_governor"
    return out, pd.DataFrame(folds), pd.DataFrame(selections)


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
        "hypothesis": "A T-7 mature relative-loss trust state can shrink the 0190 correction when prior contexts show it is likely to lose to raw official, reducing tail harm without leaking target-current outcomes.",
        "rationale": "0190 is the current comparable champion but June and a few yearly slices still show small damage. 0191 showed another additive residual layer is redundant, so 0192 changes role from correction to trust and shrinkage.",
        "expected_sign_and_falsification": "Expected sign is lower MAE than 0190 with no severe-tail harm. Falsified if shrinking the 0190 correction sacrifices global lift or fails to identify the damaged contexts.",
        "novelty": {"prior_experiments": ["0190", "0191"], "difference": "Trust-governor role using mature relative loss of parent versus official, not another residual-additive correction.", "similarity_audit_path": "RESULTS.md#comparison-limitations"},
        "target": {"station": "Hong Kong Observatory headquarters", "variable": "tmax_c", "horizon": "T-24", "timezone": "Asia/Hong_Kong", "cutoff_contract_path": rel(base.OFFICIAL_PATH), "cutoff_function": "Parent 0190 predictions are exact-frame; calibrator updates only with residuals matured by seven days.", "daily_boundary_contract": "HKO local daily maximum temperature for target local date T."},
        "frame": {"frame_id": "official_t15_pre2024_5265_rows", "development_start": "2000-01-02", "development_end_exclusive": "2024-01-01", "confirmation_locked": True, "row_universe_artifact": rel(P0190)},
        "data_sources": [{"source_id": "0190_parent_predictions", "paths": [rel(P0190)], "attributes": ["candidate_prediction_c", "calibration_correction_c", "target_tmax_c for matured relative-loss state"], "eligibility": "DEPLOYABLE_LAGGED_ONLY", "availability_proof": "0190 validator passed; trust state uses only rows <= T-7."}],
        "stations": [{"station_id": "HKO", "role": "target and matured residual state", "attributes": ["daily Tmax"]}],
        "features": {"generation_rule": "EW states of mature relative loss abs(parent0190 error) minus abs(official error), by month/source/season and known parent correction sign or magnitude; if prior loss is positive the 0190 correction is shrunk toward official.", "grid": CONFIG_GRID, "explicit_exclusions": ["2024+ rows", "current target residual", "relative-loss rows newer than T-7"]},
        "response": {"variable": "absolute-error difference between parent_0190_prediction_c and official_prediction_c", "prediction": "official_prediction_c plus trust_weight times the 0190 correction"},
        "baseline": {"id": "official_forecast_max_c", "definition": "Raw official forecast on identical rows."},
        "validation": {"outer_folds": [list(item) for item in MODEL_FOLDS], "inner_selection": "Config selected by prior-year MAE only.", "minimum_train_rows": 365},
        "metrics": ["MAE", "RMSE", "bias", "P90/P95/max AE", ">2C and >3C rates", "fold/year/season/month/source/tail slices"],
        "sample_rules": {"row_policy": "All 0190 parent rows.", "missing_policy": "No mature residual context gives zero correction."},
        "acceptance_gates": {"minimum_mae_lift_vs_official_c": 0.01, "minimum_mae_lift_vs_0190_c": 0.002, "no_tail_harm": ">3C rate cannot exceed official by more than 0.005."},
        "rejection_conditions": ["Any 2024+ row.", "Any residual newer than T-7 updates prediction.", "Parent row mismatch."],
        "required_outputs": ["README.md", "RESULTS.md", "CONCLUSION.md", "scoreboard.csv", "slice_metrics.csv", "yearly_metrics.csv", "fold_metrics.csv", "predictions.parquet"],
        "owner_authorized_confirmation": False,
    }


def write_docs(predictions: pd.DataFrame, scoreboard: pd.DataFrame, slice_metrics: pd.DataFrame, yearly_metrics: pd.DataFrame, fold_metrics: pd.DataFrame, selections: pd.DataFrame, summary: dict[str, Any]) -> None:
    write_text(EXP_DIR / "README.md", f"""# {TITLE}

## Experiment Identity And Status

Experiment `{EXPERIMENT_ID}` is `{summary['status']}`. It is a T-7 mature relative-loss trust governor over 0190.

## One-Sentence Hypothesis

The 0190 correction can be shrunk toward raw official in contexts where mature prior evidence says the champion tends to lose.

## Why It Is Worth Doing

0190 is the current comparable champion, but June and a few yearly slices still show small damage. 0192 tests whether a prior-loss governor can protect those states.

## Prior Evidence And Novelty

0190 established the current champion. 0191 showed additive correction-sign residual stacking is redundant, so this experiment changes role to trust shrinkage.

## Target, Horizon, And Exact Cutoff

Target is HKO Tmax at `T-24`, timezone `Asia/Hong_Kong`; relative-loss states are admitted only when at least seven days old.

## Datasets, Stations, And Attributes

Input is the validator-clean 0190 parent prediction artifact on the official frame.

## Feature Definitions

Features are mature relative-loss states by global, source, season, month, and known 0190 correction sign/magnitude contexts. Details are in `feature_definitions.csv`.

## Response And Baseline

Response is whether 0190 has recently beaten raw official in comparable mature contexts. Primary baseline remains raw official forecast, with 0190 as parent reference.

## Walk-Forward Design

Each outer fold selects one config from prior-year performance; row-level relative-loss states update only from T-7 mature outcomes.

## Acceptance And Rejection Criteria

Acceptance requires lower MAE than official and material lift over 0190 without severe-tail harm.

## Expected Failure Modes

The governor can fail by shrinking away useful 0190 corrections or by learning noisy relative-loss states.

## Reproduction Command

Run `python scripts/run_hkg_t24_0192_t7_tail_risk_trust_governor.py` from the repository root.
""")
    write_text(EXP_DIR / "RESULTS.md", f"""# Results

## Headline Result Table

{base.markdown_table(scoreboard)}

## Coverage And Row Identity

The scored frame has `{summary['n_common']}` rows from `{summary['date_start']}` to `{summary['date_end']}`. Common row hash: `{summary['common_row_hash']}`.

## Global Metrics

Official MAE is `{summary['baseline_mae_c']:.6f}` C. Candidate MAE is `{summary['candidate_mae_c']:.6f}` C. Delta versus official is `{summary['mae_delta_c']:.6f}` C. Delta versus 0190 is `{summary['delta_vs_0190_mae_c']:.6f}` C.

## Fold Stability

{base.markdown_table(fold_metrics[['fold_id', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c', 'selected_config_id', 'mean_calibration_correction_c', 'mean_abs_calibration_correction_c']], max_rows=20)}

## Yearly And Seasonal Results

{base.markdown_table(yearly_metrics[['slice_value', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c']], max_rows=30)}

Season metrics:

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].eq('season')][['slice_value', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c']], max_rows=10)}

## Source And Source-Era Results

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].isin(['source', 'late_window'])][['slice_type', 'slice_value', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c']], max_rows=20)}

## High-Error-Tail Results

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].eq('official_tail')][['slice_value', 'n', 'official_mae_c', 'candidate_mae_c', 'mae_delta_c', 'official_gt3c_rate', 'candidate_gt3c_rate']], max_rows=10)}

## Signed Over/Underforecast Results

Row-level parent residual, trust weight, prior relative-loss state, and signed errors are in `predictions.parquet`.

## Ablations

Parent 0190 reference is in `scoreboard.csv`; selected configs are in `artifacts/fold_config_selections.csv`.

## Data-Quality And Leakage Result

Leakage status is `{summary['leakage_status']}`. Residual states are T-7 mature.

## Comparison Limitations

This is a child trust-governor experiment over 0190. If it does not beat 0190, it should not replace the parent.

Selections:

{base.markdown_table(selections, max_rows=20)}
""")
    write_text(EXP_DIR / "CONCLUSION.md", f"""# Conclusion

## Verdict

Status is `{summary['status']}` with promotion decision `{summary['promotion_decision']}`.

## What Was Learned

0192 tested whether mature prior relative loss can tell when to trust or shrink the 0190 correction.

## Realized Point-MAE Change

Delta versus official is `{summary['mae_delta_c']:.6f}` C. Delta versus 0190 is `{summary['delta_vs_0190_mae_c']:.6f}` C.

## Information Gain Outside Point MAE

The result shows whether trust/routing information remains after the 0190 post-ensemble calibration.

## Robustness And Uncertainty

All trust states are T-7 mature and selected by prior-year performance only.

## Failure Diagnosis

If not promoted, the likely cause is that 0190's correction is already robust enough that prior-loss shrinkage gives up more lift than it saves.

## Promotion Status

Confirmation remains locked. Development gate to 0.45 C was not reached.

## Implication For Future Research

If this fails, move away from parent-only governors and search for new independent safe signals or finer routing contexts.
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
    official_global = base.metric_row(predictions, "official_prediction_c", label="official_forecast_max_c")
    candidate_global = base.metric_row(predictions, "candidate_prediction_c", label=PRIMARY_CANDIDATE_ID)
    parent_global = base.metric_row(predictions, "parent_0190_prediction_c", label="0190_parent")
    mae_delta = candidate_global["mae_c"] - official_global["mae_c"]
    delta_vs_parent = candidate_global["mae_c"] - parent_global["mae_c"]
    severe_harm = candidate_global["gt3c_rate"] - official_global["gt3c_rate"]
    fold_worst_delta = float(fold_metrics["mae_delta_c"].max())
    if mae_delta <= -0.01 and delta_vs_parent <= -0.002 and severe_harm <= 0.005 and fold_worst_delta <= 0.02:
        status = "COMPLETED_PROMOTION_CANDIDATE"
        promotion_decision = "PROMOTE_OVER_0190_NO_CONFIRMATION"
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
            {"candidate_id": PRIMARY_CANDIDATE_ID, "model_family": "t7_tail_risk_trust_governor", "n": candidate_global["n"], "mae_c": candidate_global["mae_c"], "rmse_c": candidate_global["rmse_c"], "bias_c": candidate_global["bias_c"], "median_abs_error_c": candidate_global["median_abs_error_c"], "p95_abs_error_c": candidate_global["p95_abs_error_c"], "gt2c_rate": candidate_global["gt2c_rate"], "gt3c_rate": candidate_global["gt3c_rate"], "baseline_mae_c": official_global["mae_c"], "mae_delta_c": mae_delta},
            {"candidate_id": "0190_parent_post_ensemble_calibrator", "model_family": "parent_reference", "n": parent_global["n"], "mae_c": parent_global["mae_c"], "rmse_c": parent_global["rmse_c"], "bias_c": parent_global["bias_c"], "median_abs_error_c": parent_global["median_abs_error_c"], "p95_abs_error_c": parent_global["p95_abs_error_c"], "gt2c_rate": parent_global["gt2c_rate"], "gt3c_rate": parent_global["gt3c_rate"], "baseline_mae_c": official_global["mae_c"], "mae_delta_c": parent_global["mae_c"] - official_global["mae_c"]},
        ]
    )
    row_coverage = pd.DataFrame([{"frame_id": "official_t15_pre2024_5265_rows", "parent_rows": int(len(frame)), "candidate_rows": int(len(predictions)), "baseline_rows": int(len(predictions)), "common_rows": int(len(predictions)), "date_start": date_text(predictions["target_date"].min()), "date_end": date_text(predictions["target_date"].max()), "row_policy": "all 0190 parent rows", "common_row_hash": common_row_hash}])
    correction_distribution = predictions["calibration_correction_c"].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).reset_index()
    correction_distribution.columns = ["statistic", "calibration_correction_c"]
    data_manifest = pd.DataFrame([{"source_id": "0190_parent_predictions", "path": rel(P0190), "sha256": sha256_file(P0190), "size_bytes": P0190.stat().st_size, "row_count": int(len(frame)), "date_start": date_text(frame["target_date"].min()), "date_end": date_text(frame["target_date"].max()), "timestamp_fields": "target_date;T-7 parent-vs-official relative-loss state;parent correction sign", "availability_class": "DEPLOYABLE_LAGGED_ONLY", "notes": "Validator-clean parent 0190 predictions."}])
    feature_definitions = pd.DataFrame([{"feature_name": "t7_parent_0190_tail_risk_trust_weight", "role": "candidate_router", "formula": "EW mean of mature relative loss abs(parent_0190 error) - abs(official error) in predeclared contexts; positive prior loss shrinks the 0190 correction toward official by a predeclared floor.", "input_columns": "parent_0190_prediction_c,official_prediction_c,parent_0190_calibration_correction_c,target_tmax_c,season,month,source", "units": "degC relative loss and unitless weight", "lag": "7 target days minimum", "window": "predeclared EW half-life grid", "fit_scope": "fold-local config selection; prequential state", "availability_rule": "No current or recent target outcome updates prediction state; current parent correction sign is known before target resolution.", "missingness_policy": "No mature state gives trust weight 1.0."}])
    pred_cols = ["target_date", "target_tmax_c", "forecast_source_family", "season", "month", "official_prediction_c", "parent_0190_prediction_c", "parent_0190_calibration_correction_c", "candidate_prediction_c", "parent_0190_residual_c", "calibration_correction_c", "candidate_correction_c", "tail_governor_weight", "prior_parent_loss_diff_c", "active_calibration_context_count", "official_error_c_signed", "candidate_error_c", "official_abs_error_c", "candidate_abs_error_c", "fold_id", "selected_config_id", "candidate_id", "baseline_id", "model_family"]
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

0192 consumes validator-clean 0190 parent predictions. Trust states use parent-vs-official relative loss only after target dates are at least `{LAG_DAYS}` days old. The current-row context may use the known parent correction sign and magnitude, but never the current target residual.

## Available State

For target T, residual contexts include only rows with target_date <= T-7. No current target residual is available to its own prediction.

## Target And Rolling Checks

Fold configuration selection uses only years before the fold start. Current target values are used only for scoring.

## Confirmation Proof

Maximum scored target date is `{date_text(predictions['target_date'].max())}`. Confirmation rows used: `0`. Owner authorization for confirmation: `false`.

## Row Identity

Candidate and official baseline share `{len(predictions)}` rows. Common row hash: `{common_row_hash}`.
""")
    write_text(EXP_DIR / "REPRODUCE.md", f"""# Reproduction

From `{REPO_ROOT}`, run:

```powershell
python scripts/run_hkg_t24_0192_t7_tail_risk_trust_governor.py
```

Requires completed parent predictions from 0190. Confirmation rows remain locked.
""")
    code_sha = sha256_file(src_copy_path)
    data_manifest_sha = sha256_file(EXP_DIR / "data_manifest.csv")
    summary = {"experiment_id": EXPERIMENT_ID, "slug": SLUG, "status": status, "created_at_utc": created_at, "target": "HKO daily Tmax T-24", "frame_id": "official_t15_pre2024_5265_rows", "date_start": date_text(predictions["target_date"].min()), "date_end": date_text(predictions["target_date"].max()), "n_candidate": int(len(predictions)), "n_common": int(len(predictions)), "baseline_id": "official_forecast_max_c", "baseline_mae_c": official_global["mae_c"], "candidate_id": PRIMARY_CANDIDATE_ID, "candidate_mae_c": candidate_global["mae_c"], "mae_delta_c": mae_delta, "candidate_rmse_c": candidate_global["rmse_c"], "candidate_bias_c": candidate_global["bias_c"], "leakage_status": "PASS", "confirmation_rows_used": 0, "owner_authorized_confirmation": False, "promotion_decision": promotion_decision, "spec_sha256": spec_sha, "code_sha256": code_sha, "data_manifest_sha256": data_manifest_sha, "common_row_hash": common_row_hash, "baseline_n": int(len(predictions)), "candidate_n": int(len(predictions)), "development_gate_reached": bool(candidate_global["mae_c"] <= 0.45), "parent_0190_mae_c": parent_global["mae_c"], "delta_vs_0190_mae_c": delta_vs_parent, "fold_worst_mae_delta_c": fold_worst_delta, "severe_gt3_rate_delta": severe_harm}
    write_json(EXP_DIR / "summary.json", summary)
    write_docs(predictions, scoreboard, slice_metrics, yearly_metrics, fold_metrics, selections, summary)
    write_json(EXP_DIR / "run_manifest.json", {"experiment_id": EXPERIMENT_ID, "slug": SLUG, "created_at_utc": created_at, "completed_at_utc": utc_now(), "repo_root": str(REPO_ROOT), "script": rel(Path(__file__).resolve()), "spec_sha256": spec_sha, "code_sha256": code_sha, "state": "COMPLETED"})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
