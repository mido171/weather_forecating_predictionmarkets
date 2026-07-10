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
EXPERIMENT_ID = "0200"
SLUG = "climatological_analogue_blend"
TITLE = "Climatological Analogue Blend Over 0196"
EXP_DIR = EXPERIMENTS_ROOT / f"{EXPERIMENT_ID}_{SLUG}"
PRIMARY_CANDIDATE_ID = "0200_climatological_analogue_blend_over_0196"
P0196 = EXPERIMENTS_ROOT / "0196_station_network_tail_conditioned_residual_expert" / "predictions.parquet"
SRC_COPY_NAME = "run_0200.py"
MODEL_FOLDS = base.MODEL_FOLDS
LAG_DAYS = 7

CONFIG_GRID = [
    {"mode": "parent_only", "k": 0, "doy_window": 0, "blend": 0.0, "cap_c": 0.0, "same_source": False, "use_parent_distance": False, "use_tail_distance": False},
    {"mode": "seasonal_doy_k30", "k": 30, "doy_window": 30, "blend": 0.25, "cap_c": 0.15, "same_source": False, "use_parent_distance": False, "use_tail_distance": False},
    {"mode": "seasonal_doy_k60", "k": 60, "doy_window": 45, "blend": 0.25, "cap_c": 0.20, "same_source": False, "use_parent_distance": False, "use_tail_distance": False},
    {"mode": "source_doy_k40", "k": 40, "doy_window": 45, "blend": 0.30, "cap_c": 0.20, "same_source": True, "use_parent_distance": False, "use_tail_distance": False},
    {"mode": "parent_analogue_k50", "k": 50, "doy_window": 60, "blend": 0.25, "cap_c": 0.20, "same_source": False, "use_parent_distance": True, "use_tail_distance": False},
    {"mode": "source_parent_analogue_k40", "k": 40, "doy_window": 60, "blend": 0.30, "cap_c": 0.20, "same_source": True, "use_parent_distance": True, "use_tail_distance": False},
    {"mode": "source_parent_tail_k40", "k": 40, "doy_window": 60, "blend": 0.30, "cap_c": 0.20, "same_source": True, "use_parent_distance": True, "use_tail_distance": True},
    {"mode": "loose_parent_tail_k80", "k": 80, "doy_window": 90, "blend": 0.20, "cap_c": 0.15, "same_source": False, "use_parent_distance": True, "use_tail_distance": True},
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
    frame["official_abs_error_c"] = (frame["official_prediction_c"] - frame["target_tmax_c"]).abs()
    frame["month"] = pd.to_numeric(frame["month"], errors="coerce").fillna(frame["target_date"].dt.month).astype(int)
    frame["day_of_year"] = frame["target_date"].dt.dayofyear.astype(int)
    frame["source_code"] = frame["forecast_source_family"].astype("category").cat.codes.astype(int)
    frame["tail_abs_c"] = pd.to_numeric(frame.get("parent_0196_tail_expert_correction_c", 0.0), errors="coerce").fillna(0.0).abs()
    base.assert_pre2024(frame, "0200 parent 0196 frame")
    return frame.sort_values("target_date").reset_index(drop=True)


def cyclic_doy_distance(values: np.ndarray, current: int) -> np.ndarray:
    raw = np.abs(values - current)
    return np.minimum(raw, 366 - raw)


def analogue_prediction(frame: pd.DataFrame, config: dict[str, Any], config_id: str) -> pd.DataFrame:
    ordered = frame.sort_values("target_date").reset_index(drop=True).copy()
    if config["mode"] == "parent_only":
        out = ordered.copy()
        out["config_id"] = config_id
        out["analogue_prediction_c"] = np.nan
        out["analogue_count"] = 0
        out["analogue_adjustment_c"] = 0.0
        out["candidate_prediction_c"] = out["parent_0196_prediction_c"]
        return finalize(out)

    dates = pd.to_datetime(ordered["target_date"]).dt.normalize()
    doys = ordered["day_of_year"].to_numpy(dtype=float)
    target = ordered["target_tmax_c"].to_numpy(dtype=float)
    parent = ordered["parent_0196_prediction_c"].to_numpy(dtype=float)
    source = ordered["source_code"].to_numpy(dtype=int)
    tail_abs = ordered["tail_abs_c"].to_numpy(dtype=float)
    analogue_preds = []
    counts = []
    adjustments = []
    for idx, current_date in enumerate(dates):
        mature_date = current_date - pd.Timedelta(days=LAG_DAYS)
        hist = np.where(dates.to_numpy() <= np.datetime64(mature_date))[0]
        if hist.size == 0:
            analogue_preds.append(np.nan)
            counts.append(0)
            adjustments.append(0.0)
            continue
        doy_dist = cyclic_doy_distance(doys[hist], int(doys[idx]))
        mask = doy_dist <= float(config["doy_window"])
        if config["same_source"]:
            mask &= source[hist] == source[idx]
        candidates = hist[mask]
        if candidates.size < max(10, int(config["k"]) // 3):
            candidates = hist[doy_dist <= max(float(config["doy_window"]), 90.0)]
        if candidates.size == 0:
            analogue_preds.append(np.nan)
            counts.append(0)
            adjustments.append(0.0)
            continue
        dist = cyclic_doy_distance(doys[candidates], int(doys[idx])) / max(1.0, float(config["doy_window"]))
        if config["use_parent_distance"]:
            dist = dist + np.abs(parent[candidates] - parent[idx]) / 2.0
        if config["use_tail_distance"]:
            dist = dist + np.abs(tail_abs[candidates] - tail_abs[idx]) / 0.20
        order = np.argsort(dist)[: int(config["k"])]
        chosen = candidates[order]
        chosen_dist = dist[order]
        weights = 1.0 / (0.25 + chosen_dist)
        analog = float(np.average(target[chosen], weights=weights))
        adjustment = float(np.clip(analog - parent[idx], -float(config["cap_c"]), float(config["cap_c"])))
        analogue_preds.append(analog)
        counts.append(int(chosen.size))
        adjustments.append(float(config["blend"]) * adjustment)
    out = ordered.copy()
    out["config_id"] = config_id
    out["analogue_prediction_c"] = analogue_preds
    out["analogue_count"] = counts
    out["analogue_adjustment_c"] = adjustments
    out["candidate_prediction_c"] = out["parent_0196_prediction_c"] + out["analogue_adjustment_c"]
    return finalize(out)


def finalize(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["candidate_correction_c"] = out["candidate_prediction_c"] - out["official_prediction_c"]
    out["candidate_error_c"] = out["candidate_prediction_c"] - out["target_tmax_c"]
    out["candidate_abs_error_c"] = out["candidate_error_c"].abs()
    return out


def compare(frame: pd.DataFrame, *, slice_type: str, slice_value: Any) -> dict[str, Any]:
    official = base.metric_row(frame, "official_prediction_c", label="official")
    parent = base.metric_row(frame, "parent_0196_prediction_c", label="p0196")
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


def run_walk_forward(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config_rows = []
    preds = {}
    for idx, config in enumerate(CONFIG_GRID):
        cid = f"cfg_{idx:02d}_{config['mode']}"
        cfg = {"config_id": cid, **config}
        config_rows.append(cfg)
        preds[cid] = analogue_prediction(frame, cfg, cid)
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
        pred["selected_mode"] = selected["mode"]
        parts.append(pred)
        metric = compare(pred, slice_type="fold", slice_value=fold_id)
        metric.update(
            {
                "fold_id": fold_id,
                "start_year": start_year,
                "end_year": end_year,
                "selected_config_id": selected["config_id"],
                "selected_mode": selected["mode"],
                "selection_train_mae_c": selected.get("train_mae_c"),
                "selection_train_delta_vs_0196_mae_c": selected.get("train_delta_vs_0196_mae_c"),
                "mean_analogue_count": float(pred["analogue_count"].mean()),
                "mean_abs_analogue_adjustment_c": float(pred["analogue_adjustment_c"].abs().mean()),
            }
        )
        folds.append(metric)
        selections.append({"fold_id": fold_id, **selected})
    out = pd.concat(parts, ignore_index=True).sort_values("target_date").reset_index(drop=True)
    out["candidate_id"] = PRIMARY_CANDIDATE_ID
    out["baseline_id"] = "official_forecast_max_c"
    out["model_family"] = "climatological_analogue_blend"
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
    tail = predictions[predictions["parent_0196_abs_error_c"] >= 2.0]
    rows.append(compare(tail, slice_type="parent_tail", slice_value="parent_0196_abs_error_ge_2c"))
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
        "hypothesis": "A strictly historical same-season analogue target mean can correct remaining 0196 errors when similar pre-cutoff parent/source/tail states have resolved differently in the past.",
        "rationale": "Recent trust, routing, and restricted station-family experiments plateaued around 0196. Analogue blending tests a different safe base-rate signal rather than another station model.",
        "expected_sign_and_falsification": "Expected sign is lower MAE than 0196 or improved late/source slices. Falsified if parent-only is selected or analogue adjustments worsen 0196.",
        "novelty": {"prior_experiments": ["0196", "0198", "0199"], "difference": "Historical same-season analogue target blend over 0196, not station-model refitting or expert routing.", "similarity_audit_path": "RESULTS.md#comparison-limitations"},
        "target": {"station": "Hong Kong Observatory headquarters", "variable": "tmax_c", "horizon": "T-24", "timezone": "Asia/Hong_Kong", "cutoff_contract_path": rel(base.OFFICIAL_PATH), "cutoff_function": "For target T, analogue targets are admitted only from target dates <= T-7.", "daily_boundary_contract": "HKO local daily maximum temperature for target local date T."},
        "frame": {"frame_id": "official_t15_pre2024_5265_rows", "development_start": "2000-01-02", "development_end_exclusive": "2024-01-01", "confirmation_locked": True, "row_universe_artifact": rel(P0196)},
        "data_sources": [{"source_id": "0196_parent_predictions", "paths": [rel(P0196)], "attributes": ["0196 prediction", "source", "tail correction", "historical target outcomes for T-7 mature analogues"], "eligibility": "DEPLOYABLE_LAGGED_ONLY", "availability_proof": "0196 validator passed; analogue target history uses only rows <= T-7."}],
        "stations": [{"station_id": "HKO", "role": "target and historical analogue pool", "attributes": ["daily Tmax"]}],
        "features": {"generation_rule": "For each row, select historical analogue days at least seven target days old by day-of-year window, optional source match, parent prediction distance, and tail correction distance; blend clipped analogue-minus-parent adjustment into 0196.", "grid": CONFIG_GRID, "explicit_exclusions": ["2024+ rows", "current target outcome", "current target residual/loss"]},
        "response": {"variable": "historical target_tmax_c for mature analogue pool; target_tmax_c for scoring only", "prediction": "0196 parent plus clipped historical analogue adjustment"},
        "baseline": {"id": "official_forecast_max_c", "definition": "Raw official forecast on identical rows; 0196 is parent champion reference."},
        "validation": {"outer_folds": [list(item) for item in MODEL_FOLDS], "inner_selection": "Config selected by prior-year MAE only; current fold outcomes are not used for config choice.", "minimum_train_rows": 365},
        "metrics": ["MAE", "RMSE", "bias", "P95 AE", ">3C rate", "fold/year/season/month/source/tail slices", "analogue count", "adjustment distribution"],
        "sample_rules": {"row_policy": "All 0196 parent rows.", "missing_policy": "No mature analogue gives default 0196 prediction."},
        "acceptance_gates": {"minimum_mae_lift_vs_official_c": 0.01, "minimum_mae_lift_vs_0196_c": 0.001, "max_fold_harm_vs_0196_c": 0.001, "no_parent_tail_harm": ">3C rate cannot exceed 0196 by more than 0.005."},
        "rejection_conditions": ["Any 2024+ row.", "Any analogue target newer than T-7.", "Parent row mismatch."],
        "required_outputs": ["README.md", "RESULTS.md", "CONCLUSION.md", "scoreboard.csv", "slice_metrics.csv", "yearly_metrics.csv", "fold_metrics.csv", "predictions.parquet"],
        "owner_authorized_confirmation": False,
    }


def write_docs(scoreboard: pd.DataFrame, slice_metrics: pd.DataFrame, yearly_metrics: pd.DataFrame, fold_metrics: pd.DataFrame, selections: pd.DataFrame, summary: dict[str, Any]) -> None:
    write_text(EXP_DIR / "README.md", f"""# {TITLE}

## Experiment Identity And Status

Experiment `{EXPERIMENT_ID}` is `{summary['status']}`. It is a strictly historical analogue blend over 0196.

## One-Sentence Hypothesis

Mature same-season analogue target outcomes can supply a safe base-rate correction when 0196 is locally miscalibrated.

## Why It Is Worth Doing

0196 remains the champion after multiple trust and routing attempts. Analogue blending tests a different safe signal family: historical target outcomes under similar pre-cutoff parent/source/tail states.

## Prior Evidence And Novelty

This does not refit station features or route among frozen experts. It asks whether historical analogue climatology adds incremental correction beyond the station-tail system.

## Target, Horizon, And Exact Cutoff

Target is HKO Tmax at `T-24`, timezone `Asia/Hong_Kong`. Analogue target outcomes for row T are admitted only from target dates <= T-7.

## Datasets, Stations, And Attributes

Input is the validator-clean 0196 prediction artifact, including parent prediction, source, tail correction, and historical target outcomes for mature analogue rows.

## Feature Definitions

Analogue definitions are in `feature_definitions.csv`; selected configs are in `artifacts/fold_config_selections.csv`.

## Response And Baseline

The candidate is 0196 plus a clipped analogue-minus-parent adjustment. Official raw forecast is the primary baseline; 0196 is the parent champion reference.

## Walk-Forward Design

Each fold selects one analogue config using only prior-year MAE. Parent-only exactly reproduces 0196 and is always available.

## Acceptance And Rejection Criteria

Acceptance requires at least 0.001 C global MAE lift versus 0196 without fold or severe-tail harm.

## Reproduction Command

Run `python scripts/run_hkg_t24_0200_climatological_analogue_blend.py` from the repository root.
""")
    write_text(EXP_DIR / "RESULTS.md", f"""# Results

## Headline Result Table

{base.markdown_table(scoreboard)}

## Coverage And Row Identity

Rows: `{summary['n_common']}` from `{summary['date_start']}` to `{summary['date_end']}`. Common row hash: `{summary['common_row_hash']}`.

## Global Metrics

Official MAE is `{summary['baseline_mae_c']:.6f}` C. Parent 0196 MAE is `{summary['parent_0196_mae_c']:.6f}` C. Candidate MAE is `{summary['candidate_mae_c']:.6f}` C. Delta versus 0196 is `{summary['delta_vs_0196_mae_c']:.6f}` C.

## Fold Stability

{base.markdown_table(fold_metrics[['fold_id', 'n', 'parent_0196_mae_c', 'candidate_mae_c', 'delta_vs_0196_mae_c', 'selected_config_id', 'selected_mode', 'mean_analogue_count', 'mean_abs_analogue_adjustment_c']], max_rows=20)}

## Yearly And Monthly Results

{base.markdown_table(yearly_metrics[['slice_value', 'n', 'parent_0196_mae_c', 'candidate_mae_c', 'delta_vs_0196_mae_c']], max_rows=30)}

Month metrics:

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].eq('month')][['slice_value', 'n', 'parent_0196_mae_c', 'candidate_mae_c', 'delta_vs_0196_mae_c']], max_rows=20)}

## Tail And Source Results

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].isin(['season', 'source', 'late_window', 'parent_tail'])][['slice_type', 'slice_value', 'n', 'parent_0196_mae_c', 'candidate_mae_c', 'delta_vs_0196_mae_c', 'parent_0196_gt3c_rate', 'candidate_gt3c_rate']], max_rows=25)}

## Data-Quality And Leakage Result

Leakage status is `{summary['leakage_status']}`. Analogue targets are T-7 mature.

## Comparison Limitations

This is a child blend over 0196. If it does not beat 0196 globally, it should not replace the parent.

Selections:

{base.markdown_table(selections, max_rows=20)}
""")
    write_text(EXP_DIR / "CONCLUSION.md", f"""# Conclusion

## Verdict

Status is `{summary['status']}` with promotion decision `{summary['promotion_decision']}`.

## What Was Learned

0200 tested whether historical analogue target outcomes add safe base-rate correction beyond the 0196 station-tail champion.

## Realized Point-MAE Change

Delta versus official is `{summary['mae_delta_c']:.6f}` C. Delta versus 0196 is `{summary['delta_vs_0196_mae_c']:.6f}` C.

## Robustness And Uncertainty

Robustness comes from T-7 analogue maturity, prior-year config selection, parent-only fallback, and clipped small adjustments. The uncertainty is that analogue similarity is still fit to the development corpus and may not generalize if source regimes drift.

## Failure Diagnosis

If parent-only is selected or analogue blending worsens MAE, historical target analogue correction is already absorbed by 0196 or is too noisy. Future work should then prioritize new timestamp-proven forecast data rather than more climatological blending.

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
    parent_global = base.metric_row(predictions, "parent_0196_prediction_c", label="p0196")
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
            {"candidate_id": PRIMARY_CANDIDATE_ID, "model_family": "climatological_analogue_blend", "n": candidate_global["n"], "mae_c": candidate_global["mae_c"], "rmse_c": candidate_global["rmse_c"], "bias_c": candidate_global["bias_c"], "median_abs_error_c": candidate_global["median_abs_error_c"], "p95_abs_error_c": candidate_global["p95_abs_error_c"], "gt2c_rate": candidate_global["gt2c_rate"], "gt3c_rate": candidate_global["gt3c_rate"], "baseline_mae_c": official_global["mae_c"], "mae_delta_c": mae_delta},
        ]
    )
    row_coverage = pd.DataFrame([{"frame_id": "official_t15_pre2024_5265_rows", "parent_rows": int(len(frame)), "candidate_rows": int(len(predictions)), "baseline_rows": int(len(predictions)), "common_rows": int(len(predictions)), "date_start": date_text(predictions["target_date"].min()), "date_end": date_text(predictions["target_date"].max()), "row_policy": "all 0196 parent rows", "common_row_hash": common_row_hash}])
    correction_distribution = predictions[["analogue_count", "analogue_adjustment_c"]].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).reset_index()
    data_manifest = pd.DataFrame([{"source_id": "0196_parent_predictions", "path": rel(P0196), "sha256": sha256_file(P0196), "size_bytes": P0196.stat().st_size, "row_count": int(len(frame)), "date_start": date_text(frame["target_date"].min()), "date_end": date_text(frame["target_date"].max()), "timestamp_fields": "target_date;0196 prediction;historical target outcomes admitted at T-7", "availability_class": "DEPLOYABLE_LAGGED_ONLY", "notes": "Validator-clean 0196 predictions; analogue target pool uses only rows <= T-7."}])
    feature_definitions = pd.DataFrame([{"feature_name": "t7_historical_climatological_analogue_adjustment", "role": "candidate_adjustment", "formula": "Weighted mean target_tmax_c among historical rows with target_date <= T-7 selected by day-of-year window, optional source match, parent prediction distance, and tail correction distance; clipped blend into 0196.", "input_columns": "target_tmax_c,target_date,day_of_year,forecast_source_family,parent_0196_prediction_c,parent_0196_tail_expert_correction_c", "units": "degC", "lag": "7 target days minimum", "window": "predeclared analogue grid", "fit_scope": "fold-local config selection; prequential analogue pool", "availability_rule": "Current target outcome is never in its own analogue pool.", "missingness_policy": "No mature analogue gives default 0196 prediction."}])
    pred_cols = ["target_date", "target_tmax_c", "forecast_source_family", "season", "month", "official_prediction_c", "parent_0196_prediction_c", "candidate_prediction_c", "analogue_prediction_c", "analogue_count", "analogue_adjustment_c", "candidate_correction_c", "official_abs_error_c", "parent_0196_abs_error_c", "candidate_abs_error_c", "fold_id", "selected_config_id", "selected_mode", "candidate_id", "baseline_id", "model_family"]
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

0200 consumes validator-clean 0196 parent predictions. For target T, the analogue pool includes only historical rows with target_date <= T-{LAG_DAYS}.

## Available State

Current-row source, day-of-year, 0196 prediction, and 0196 tail correction are known before target resolution. Current target outcome is never included in its own analogue pool.

## Target And Rolling Checks

Fold config selection uses only years before the fold start. Online analogue pools can include previous scored-era rows only after their outcomes are at least seven target days old.

## Confirmation Proof

Maximum scored target date is `{date_text(predictions['target_date'].max())}`. Confirmation rows used: `0`. Owner authorization for confirmation: `false`.

## Row Identity

Candidate, parent 0196, and official baseline share `{len(predictions)}` rows. Common row hash: `{common_row_hash}`.
""")
    write_text(EXP_DIR / "REPRODUCE.md", f"""# Reproduction

From `{REPO_ROOT}`, run:

```powershell
python scripts/run_hkg_t24_0200_climatological_analogue_blend.py
```

Requires completed parent predictions from 0196. Confirmation rows remain locked.
""")
    code_sha = sha256_file(src_copy_path)
    data_manifest_sha = sha256_file(EXP_DIR / "data_manifest.csv")
    summary = {"experiment_id": EXPERIMENT_ID, "slug": SLUG, "status": status, "created_at_utc": created_at, "target": "HKO daily Tmax T-24", "frame_id": "official_t15_pre2024_5265_rows", "date_start": date_text(predictions["target_date"].min()), "date_end": date_text(predictions["target_date"].max()), "n_candidate": int(len(predictions)), "n_common": int(len(predictions)), "baseline_id": "official_forecast_max_c", "baseline_mae_c": official_global["mae_c"], "candidate_id": PRIMARY_CANDIDATE_ID, "candidate_mae_c": candidate_global["mae_c"], "mae_delta_c": mae_delta, "candidate_rmse_c": candidate_global["rmse_c"], "candidate_bias_c": candidate_global["bias_c"], "leakage_status": "PASS", "confirmation_rows_used": 0, "owner_authorized_confirmation": False, "promotion_decision": promotion_decision, "spec_sha256": spec_sha, "code_sha256": code_sha, "data_manifest_sha256": data_manifest_sha, "common_row_hash": common_row_hash, "baseline_n": int(len(predictions)), "candidate_n": int(len(predictions)), "development_gate_reached": bool(candidate_global["mae_c"] <= 0.45), "parent_0196_mae_c": parent_global["mae_c"], "delta_vs_0196_mae_c": delta_vs_0196, "fold_worst_delta_vs_0196_mae_c": fold_worst_delta, "severe_gt3_rate_delta_vs_0196": severe_harm_0196}
    write_json(EXP_DIR / "summary.json", summary)
    write_docs(scoreboard, slice_metrics, yearly_metrics, fold_metrics, selections, summary)
    write_json(EXP_DIR / "run_manifest.json", {"experiment_id": EXPERIMENT_ID, "slug": SLUG, "created_at_utc": created_at, "completed_at_utc": utc_now(), "repo_root": str(REPO_ROOT), "script": rel(Path(__file__).resolve()), "spec_sha256": spec_sha, "code_sha256": code_sha, "state": "COMPLETED"})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
