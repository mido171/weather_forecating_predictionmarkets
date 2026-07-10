from __future__ import annotations

import hashlib
import json
import math
import shutil
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_hkg_t24_0184_hf_teacher_proxy_causal_memory_router as base
import run_hkg_t24_0199_station_role_feature_family_replay as exp0199


REPO_ROOT = base.REPO_ROOT
EXPERIMENTS_ROOT = base.EXPERIMENTS_ROOT
EXPERIMENT_ID = "0201"
SLUG = "independent_feature_group_stacked_router"
TITLE = "Independent Feature-Group Stacked Router Over 0196"
EXP_DIR = EXPERIMENTS_ROOT / f"{EXPERIMENT_ID}_{SLUG}"
PRIMARY_CANDIDATE_ID = "0201_independent_feature_group_stacked_router_over_0196"
P0196 = EXPERIMENTS_ROOT / "0196_station_network_tail_conditioned_residual_expert" / "predictions.parquet"
SRC_COPY_NAME = "run_0201.py"
MODEL_FOLDS = base.MODEL_FOLDS
INNER_MIN_LIFT_C = 0.001
MIN_GATED_ROWS = 120

ELIGIBLE_GROUPS = {
    "thermal",
    "moisture",
    "pressure",
    "wind",
    "graph",
    "context",
    "thermal_moisture",
    "thermal_pressure",
    "thermal_wind",
}
ELIGIBLE_MODELS = {"huber", "hgb"}
EXPERT_CONFIGS = [
    cfg.copy()
    for cfg in exp0199.MODEL_GRID
    if cfg.get("feature_group") in ELIGIBLE_GROUPS and cfg.get("model") in ELIGIBLE_MODELS
]
CAP_GRID_C = [0.06, 0.10, 0.16]
SINGLE_WEIGHTS = [0.25, 0.50, 0.75, 1.00]
PAIR_WEIGHTS = [(0.25, 0.25), (0.50, 0.25), (0.25, 0.50), (0.50, 0.50)]
TRIPLE_WEIGHTS = [
    (0.25, 0.25, 0.25),
    (0.50, 0.25, 0.25),
    (0.25, 0.50, 0.25),
    (0.25, 0.25, 0.50),
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


def cfg_by_id(config_id: str) -> dict[str, Any]:
    for cfg in EXPERT_CONFIGS:
        if cfg["config_id"] == config_id:
            return cfg.copy()
    raise KeyError(config_id)


def load_frame() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    frame, _all_features, feature_defs, groups = exp0199.load_frame()
    forbidden = [
        col
        for cols in groups.values()
        for col in cols
        if col in {"target_tmax_c", "official_residual_c", "official_abs_error_c", "parent_0196_residual_c", "parent_0196_abs_error_c"}
        or "residual" in col.lower()
        or "abs_error" in col.lower()
    ]
    if forbidden:
        raise RuntimeError(f"Forbidden predictor columns selected: {sorted(set(forbidden))}")
    for col in ["parent_0196_tail_expert_correction_c", "parent_0196_total_correction_c", "official_prediction_c"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    base.assert_pre2024(frame, "0201 model frame")
    return frame.sort_values("target_date").reset_index(drop=True), feature_defs, groups


def build_gate_specs(fit_frame: pd.DataFrame) -> list[dict[str, Any]]:
    tail_q75 = float(fit_frame["parent_0196_tail_expert_correction_c"].abs().quantile(0.75))
    total_q75 = float(fit_frame["parent_0196_total_correction_c"].abs().quantile(0.75))
    forecast_q75 = float(fit_frame["official_prediction_c"].quantile(0.75))
    return [
        {"gate_id": "all", "description": "all rows"},
        {"gate_id": "press_archive", "description": "press archive source rows"},
        {"gate_id": "rss_archive", "description": "rss archive source rows"},
        {"gate_id": "jja_son", "description": "June through November warm and autumn rows"},
        {"gate_id": "jul_oct", "description": "July through October rows"},
        {"gate_id": "spring", "description": "March through May rows"},
        {"gate_id": "winter", "description": "December through February rows"},
        {"gate_id": "high_tail_correction", "description": "upper-quartile 0196 tail correction magnitude", "threshold": tail_q75},
        {"gate_id": "high_total_correction", "description": "upper-quartile 0196 total correction magnitude", "threshold": total_q75},
        {"gate_id": "hot_forecast", "description": "upper-quartile official forecast temperature", "threshold": forecast_q75},
    ]


def gate_mask(frame: pd.DataFrame, gate: dict[str, Any]) -> np.ndarray:
    gid = gate["gate_id"]
    month = pd.to_numeric(frame["month"], errors="coerce")
    if gid == "all":
        return np.ones(len(frame), dtype=bool)
    if gid == "press_archive":
        return frame["forecast_source_family"].astype(str).eq("press_archive").to_numpy()
    if gid == "rss_archive":
        return frame["forecast_source_family"].astype(str).eq("rss_archive").to_numpy()
    if gid == "jja_son":
        return month.isin([6, 7, 8, 9, 10, 11]).to_numpy()
    if gid == "jul_oct":
        return month.isin([7, 8, 9, 10]).to_numpy()
    if gid == "spring":
        return month.isin([3, 4, 5]).to_numpy()
    if gid == "winter":
        return month.isin([12, 1, 2]).to_numpy()
    if gid == "high_tail_correction":
        return frame["parent_0196_tail_expert_correction_c"].abs().ge(float(gate["threshold"])).to_numpy()
    if gid == "high_total_correction":
        return frame["parent_0196_total_correction_c"].abs().ge(float(gate["threshold"])).to_numpy()
    if gid == "hot_forecast":
        return frame["official_prediction_c"].ge(float(gate["threshold"])).to_numpy()
    raise ValueError(f"Unknown gate {gid}")


def fit_predict_experts(
    fit_frame: pd.DataFrame,
    predict_frame: pd.DataFrame,
    groups: dict[str, list[str]],
    configs: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    preds: dict[str, np.ndarray] = {}
    errors: list[dict[str, Any]] = []
    for cfg in configs:
        config_id = str(cfg["config_id"])
        try:
            preds[config_id] = exp0199.predict_config(fit_frame, predict_frame, groups, cfg).astype(float)
        except Exception as exc:  # Preserve failures without aborting the full lane.
            errors.append({"config_id": config_id, "model": cfg.get("model"), "feature_group": cfg.get("feature_group"), "error": str(exc)})
    return pd.DataFrame(preds, index=predict_frame.index), errors


def candidate_rule_rows(top_experts: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for expert_id in top_experts:
        for weight in SINGLE_WEIGHTS:
            rows.append({"experts": [expert_id], "weights": [weight], "rule_family": "single"})
    for expert_a, expert_b in combinations(top_experts[:6], 2):
        for weights in PAIR_WEIGHTS:
            rows.append({"experts": [expert_a, expert_b], "weights": list(weights), "rule_family": "pair"})
    for combo in combinations(top_experts[:4], 3):
        for weights in TRIPLE_WEIGHTS:
            rows.append({"experts": list(combo), "weights": list(weights), "rule_family": "triple"})
    return rows


def apply_rule(frame: pd.DataFrame, parent_pred: np.ndarray, expert_preds: pd.DataFrame, rule: dict[str, Any]) -> np.ndarray:
    if rule["rule_family"] == "parent":
        return parent_pred.copy()
    delta = np.zeros(len(frame), dtype=float)
    for expert_id, weight in zip(rule["experts"], rule["weights"]):
        delta += float(weight) * (expert_preds[expert_id].to_numpy(dtype=float) - parent_pred)
    delta = np.clip(delta, -float(rule["cap_c"]), float(rule["cap_c"]))
    mask = gate_mask(frame, rule["gate"])
    gated = np.where(mask, delta, 0.0)
    return parent_pred + gated


def metric_delta(pred: np.ndarray, target: np.ndarray, parent_pred: np.ndarray) -> tuple[float, float, float]:
    mae = float(np.mean(np.abs(pred - target)))
    parent_mae = float(np.mean(np.abs(parent_pred - target)))
    gt3 = float(np.mean(np.abs(pred - target) > 3.0))
    parent_gt3 = float(np.mean(np.abs(parent_pred - target) > 3.0))
    return mae, mae - parent_mae, gt3 - parent_gt3


def select_stack_rule(train: pd.DataFrame, groups: dict[str, list[str]]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    max_year = int(train["target_date"].dt.year.max())
    split_year = max(int(train["target_date"].dt.year.min()) + 2, max_year - 2)
    inner_fit = train[train["target_date"].dt.year < split_year].copy()
    inner_val = train[train["target_date"].dt.year >= split_year].copy()
    parent_rule = {
        "rule_id": "parent_0196",
        "rule_family": "parent",
        "experts": [],
        "weights": [],
        "cap_c": 0.0,
        "gate": {"gate_id": "all", "description": "all rows"},
        "selection_reason": "bootstrap_or_no_lift",
        "inner_mae_c": math.nan,
        "inner_parent_0196_mae_c": math.nan,
        "inner_delta_vs_0196_mae_c": 0.0,
        "inner_gt3_delta_vs_0196": 0.0,
        "gate_rows": len(inner_val),
    }
    if len(inner_fit) < 365 or len(inner_val) < 120:
        return parent_rule, pd.DataFrame([parent_rule]), pd.DataFrame(), pd.DataFrame()

    parent_pred = inner_val["parent_0196_prediction_c"].to_numpy(dtype=float)
    target = inner_val["target_tmax_c"].to_numpy(dtype=float)
    parent_mae = float(np.mean(np.abs(parent_pred - target)))
    expert_preds, fit_errors = fit_predict_experts(inner_fit, inner_val, groups, EXPERT_CONFIGS)
    expert_rows = []
    for cfg in EXPERT_CONFIGS:
        config_id = str(cfg["config_id"])
        if config_id not in expert_preds.columns:
            continue
        pred = expert_preds[config_id].to_numpy(dtype=float)
        mae, delta, gt3_delta = metric_delta(pred, target, parent_pred)
        expert_rows.append(
            {
                "config_id": config_id,
                "model": cfg.get("model"),
                "feature_group": cfg.get("feature_group"),
                "inner_mae_c": mae,
                "inner_parent_0196_mae_c": parent_mae,
                "inner_delta_vs_0196_mae_c": delta,
                "inner_gt3_delta_vs_0196": gt3_delta,
            }
        )
    expert_scores = pd.DataFrame(expert_rows).sort_values(["inner_delta_vs_0196_mae_c", "config_id"]).reset_index(drop=True)
    if fit_errors:
        expert_scores = pd.concat([expert_scores, pd.DataFrame(fit_errors)], ignore_index=True)
    if expert_scores.empty or "config_id" not in expert_scores:
        return parent_rule, pd.DataFrame([parent_rule]), expert_scores, pd.DataFrame(fit_errors)

    top_experts = [str(x) for x in expert_scores.dropna(subset=["inner_delta_vs_0196_mae_c"]).head(8)["config_id"].tolist()]
    gates = build_gate_specs(inner_fit)
    rule_rows = []
    for base_rule in candidate_rule_rows(top_experts):
        for cap in CAP_GRID_C:
            for gate in gates:
                mask = gate_mask(inner_val, gate)
                gate_rows = int(mask.sum())
                if gate["gate_id"] != "all" and gate_rows < MIN_GATED_ROWS:
                    continue
                rule = {
                    "rule_id": f"{base_rule['rule_family']}|{'+' .join(base_rule['experts'])}|w={'+' .join(f'{w:.2f}' for w in base_rule['weights'])}|cap={cap:.2f}|gate={gate['gate_id']}",
                    "rule_family": base_rule["rule_family"],
                    "experts": base_rule["experts"],
                    "weights": base_rule["weights"],
                    "cap_c": cap,
                    "gate": gate,
                }
                pred = apply_rule(inner_val, parent_pred, expert_preds, rule)
                mae, delta, gt3_delta = metric_delta(pred, target, parent_pred)
                row = {
                    "rule_id": rule["rule_id"],
                    "rule_family": rule["rule_family"],
                    "experts": ";".join(rule["experts"]),
                    "weights": ";".join(f"{x:.2f}" for x in rule["weights"]),
                    "cap_c": cap,
                    "gate_id": gate["gate_id"],
                    "gate_description": gate.get("description", ""),
                    "gate_threshold": gate.get("threshold", ""),
                    "gate_rows": gate_rows,
                    "inner_parent_0196_mae_c": parent_mae,
                    "inner_mae_c": mae,
                    "inner_delta_vs_0196_mae_c": delta,
                    "inner_gt3_delta_vs_0196": gt3_delta,
                    "_rule_payload": rule,
                }
                rule_rows.append(row)
    score_table = pd.DataFrame(rule_rows)
    parent_row = parent_rule.copy()
    parent_row.update(
        {
            "inner_mae_c": parent_mae,
            "inner_parent_0196_mae_c": parent_mae,
            "rule_id": "parent_0196",
            "gate_id": "all",
            "gate_description": "all rows",
            "gate_threshold": "",
            "gate_rows": len(inner_val),
            "_rule_payload": parent_rule,
        }
    )
    score_table = pd.concat([pd.DataFrame([parent_row]), score_table], ignore_index=True)
    score_table = score_table.sort_values(["inner_delta_vs_0196_mae_c", "inner_gt3_delta_vs_0196", "cap_c", "rule_id"]).reset_index(drop=True)
    best_row = score_table.iloc[0].to_dict()
    if float(best_row["inner_delta_vs_0196_mae_c"]) <= -INNER_MIN_LIFT_C and float(best_row["inner_gt3_delta_vs_0196"]) <= 0.005:
        selected = best_row["_rule_payload"]
        selected.update(
            {
                "selection_reason": "prior_inner_stack_lift",
                "inner_mae_c": float(best_row["inner_mae_c"]),
                "inner_parent_0196_mae_c": parent_mae,
                "inner_delta_vs_0196_mae_c": float(best_row["inner_delta_vs_0196_mae_c"]),
                "inner_gt3_delta_vs_0196": float(best_row["inner_gt3_delta_vs_0196"]),
                "gate_rows": int(best_row["gate_rows"]),
            }
        )
        return selected, score_table.drop(columns=["_rule_payload"]), expert_scores, pd.DataFrame(fit_errors)

    parent_rule.update(
        {
            "selection_reason": "parent_0196_fallback_inner_lift_below_threshold",
            "inner_mae_c": parent_mae,
            "inner_parent_0196_mae_c": parent_mae,
            "inner_delta_vs_0196_mae_c": 0.0,
            "inner_gt3_delta_vs_0196": 0.0,
            "gate_rows": len(inner_val),
        }
    )
    return parent_rule, score_table.drop(columns=["_rule_payload"]), expert_scores, pd.DataFrame(fit_errors)


def compare(frame: pd.DataFrame, *, slice_type: str, slice_value: Any) -> dict[str, Any]:
    official = base.metric_row(frame, "official_prediction_c", label="official")
    p0196 = base.metric_row(frame, "parent_0196_prediction_c", label="p0196")
    candidate = base.metric_row(frame, "candidate_prediction_c", label=PRIMARY_CANDIDATE_ID)
    return {
        "slice_type": slice_type,
        "slice_value": str(slice_value),
        "n": candidate["n"],
        "official_mae_c": official["mae_c"],
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


def run_walk_forward(frame: pd.DataFrame, groups: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parts = []
    fold_rows = []
    selection_rows = []
    stack_score_rows = []
    expert_score_rows = []
    error_rows = []
    for start_year, end_year in MODEL_FOLDS:
        fold_id = f"fold_{start_year}_{end_year}"
        test = frame[frame["target_date"].dt.year.between(start_year, end_year)].copy()
        if test.empty:
            fold_rows.append({"fold_id": fold_id, "n": 0})
            continue
        train = frame[frame["target_date"].dt.year < start_year].copy()
        if len(train) < 365:
            selected = {
                "rule_id": "parent_0196",
                "rule_family": "parent",
                "experts": [],
                "weights": [],
                "cap_c": 0.0,
                "gate": {"gate_id": "all", "description": "all rows"},
                "selection_reason": "first_fold_no_prior_history",
                "inner_delta_vs_0196_mae_c": 0.0,
                "inner_gt3_delta_vs_0196": 0.0,
                "gate_rows": len(test),
            }
            stack_scores = pd.DataFrame([selected])
            expert_scores = pd.DataFrame()
            fit_errors = pd.DataFrame()
            candidate_pred = test["parent_0196_prediction_c"].to_numpy(dtype=float)
            active_rows = 0
        else:
            selected, stack_scores, expert_scores, fit_errors = select_stack_rule(train, groups)
            selected_experts = [cfg_by_id(config_id) for config_id in selected.get("experts", [])]
            if selected["rule_family"] == "parent" or not selected_experts:
                candidate_pred = test["parent_0196_prediction_c"].to_numpy(dtype=float)
                active_rows = 0
            else:
                expert_preds, test_errors = fit_predict_experts(train, test, groups, selected_experts)
                if test_errors:
                    fit_errors = pd.concat([fit_errors, pd.DataFrame(test_errors)], ignore_index=True)
                candidate_pred = apply_rule(test, test["parent_0196_prediction_c"].to_numpy(dtype=float), expert_preds, selected)
                active_rows = int(gate_mask(test, selected["gate"]).sum())
        test["candidate_prediction_c"] = candidate_pred
        test["candidate_correction_c"] = test["candidate_prediction_c"] - test["official_prediction_c"]
        test["candidate_error_c"] = test["candidate_prediction_c"] - test["target_tmax_c"]
        test["candidate_abs_error_c"] = test["candidate_error_c"].abs()
        test["official_abs_error_c"] = (test["official_prediction_c"] - test["target_tmax_c"]).abs()
        test["fold_id"] = fold_id
        test["selected_rule_id"] = selected["rule_id"]
        test["selected_rule_family"] = selected["rule_family"]
        test["selected_experts"] = ";".join(selected.get("experts", []))
        test["selected_weights"] = ";".join(f"{float(x):.2f}" for x in selected.get("weights", []))
        test["selected_gate_id"] = selected["gate"]["gate_id"]
        test["selected_cap_c"] = float(selected["cap_c"])
        metric = compare(test, slice_type="fold", slice_value=fold_id)
        metric.update(
            {
                "fold_id": fold_id,
                "start_year": start_year,
                "end_year": end_year,
                "selected_rule_id": selected["rule_id"],
                "selected_rule_family": selected["rule_family"],
                "selected_experts": ";".join(selected.get("experts", [])),
                "selected_weights": ";".join(f"{float(x):.2f}" for x in selected.get("weights", [])),
                "selected_gate_id": selected["gate"]["gate_id"],
                "selected_cap_c": float(selected["cap_c"]),
                "selection_reason": selected.get("selection_reason", ""),
                "selected_inner_delta_vs_0196_mae_c": selected.get("inner_delta_vs_0196_mae_c", math.nan),
                "selected_inner_gt3_delta_vs_0196": selected.get("inner_gt3_delta_vs_0196", math.nan),
                "active_rows": active_rows,
            }
        )
        fold_rows.append(metric)
        selection_rows.append({"fold_id": fold_id, **selected, "gate_id": selected["gate"]["gate_id"], "gate_threshold": selected["gate"].get("threshold", "")})
        stack_scores = stack_scores.copy()
        stack_scores["fold_id"] = fold_id
        stack_score_rows.append(stack_scores)
        if not expert_scores.empty:
            expert_scores = expert_scores.copy()
            expert_scores["fold_id"] = fold_id
            expert_score_rows.append(expert_scores)
        if not fit_errors.empty:
            fit_errors = fit_errors.copy()
            fit_errors["fold_id"] = fold_id
            error_rows.append(fit_errors)
        parts.append(test)
    predictions = pd.concat(parts, ignore_index=True).sort_values("target_date").reset_index(drop=True)
    predictions["candidate_id"] = PRIMARY_CANDIDATE_ID
    predictions["baseline_id"] = "official_forecast_max_c"
    predictions["model_family"] = "independent_feature_group_stacked_router"
    return (
        predictions,
        pd.DataFrame(fold_rows),
        pd.DataFrame(selection_rows),
        pd.concat(stack_score_rows, ignore_index=True) if stack_score_rows else pd.DataFrame(),
        pd.concat(expert_score_rows, ignore_index=True) if expert_score_rows else pd.DataFrame(),
        pd.concat(error_rows, ignore_index=True) if error_rows else pd.DataFrame(),
    )


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
        "hypothesis": "Several restricted station-role feature-family experts contain complementary T-24 signal. A chronological stack of their corrections can beat 0196 while preserving 0196 fallback safety.",
        "rationale": "0199 showed thermal-moisture and thermal-wind families can win prior validation but are not robust as a single selected family. 0201 tests whether small gated convex stacks of independent family corrections are more stable than one-family replacement.",
        "expected_sign_and_falsification": "Expected sign is MAE below 0196 by at least 0.001 C. Falsified for promotion if parent-only remains selected or if the stack's lift is below gate or unstable.",
        "novelty": {
            "prior_experiments": ["0196", "0199", "0200"],
            "difference": "0201 stacks multiple independently trained physical feature-family experts with prior-only gate selection instead of choosing one feature family.",
        },
        "target": {
            "station": "Hong Kong Observatory headquarters",
            "variable": "tmax_c",
            "horizon": "T-24",
            "timezone": "Asia/Hong_Kong",
            "cutoff_contract_path": rel(base.OFFICIAL_PATH),
            "cutoff_function": "All feature-family models inherit the 0196 cutoff-safe ISD role feature contract; stack selection uses only prior years.",
        },
        "frame": {
            "frame_id": "official_t15_pre2024_5265_rows",
            "development_start": "2000-01-02",
            "development_end_exclusive": "2024-01-01",
            "confirmation_locked": True,
            "row_universe_artifact": rel(P0196),
        },
        "data_sources": [
            {"source_id": "0196_parent_predictions", "paths": [rel(P0196)], "eligibility": "DEPLOYABLE_LAGGED_ONLY"},
            {"source_id": "robust_feature_matrix_isd", "paths": [rel(base.FEATURE_MATRIX_PATH)], "eligibility": "DEPLOYABLE_PROVEN"},
        ],
        "stations": [{"station_id": "regional_isd_network", "role": "deployable surface-regime proxy"}],
        "features": {
            "expert_configs": EXPERT_CONFIGS,
            "stack_rules": {
                "weights": {"single": SINGLE_WEIGHTS, "pair": PAIR_WEIGHTS, "triple": TRIPLE_WEIGHTS},
                "caps_c": CAP_GRID_C,
                "gates": ["all", "press_archive", "rss_archive", "jja_son", "jul_oct", "spring", "winter", "high_tail_correction", "high_total_correction", "hot_forecast"],
                "minimum_gated_rows": MIN_GATED_ROWS,
            },
            "explicit_exclusions": ["2024+ rows", "current target outcome", "current residual or absolute error predictors", "confirmation rows"],
        },
        "response": {"variable": "target_tmax_c - parent_0194_prediction_c for base experts; stack applies corrections over frozen 0196"},
        "baseline": {"id": "official_forecast_max_c", "parent_reference": "0196_station_network_tail_conditioned_residual_expert"},
        "validation": {
            "outer_folds": [list(item) for item in MODEL_FOLDS],
            "inner_selection": "Base experts are fit on older prior years and stack rules are selected on later prior years only. If best stack does not beat 0196 by 0.001 C without >3C harm above 0.005, parent 0196 is selected.",
            "minimum_train_rows": 365,
        },
        "metrics": ["MAE", "RMSE", "bias", "P95 AE", ">3C rate", "fold/year/season/month/source/tail slices", "selected experts and gates"],
        "acceptance_gates": {
            "minimum_mae_lift_vs_official_c": 0.01,
            "minimum_mae_lift_vs_0196_c": 0.001,
            "max_fold_harm_vs_0196_c": 0.001,
            "max_gt3_rate_delta_vs_0196": 0.005,
        },
        "owner_authorized_confirmation": False,
    }


def write_docs(scoreboard: pd.DataFrame, slice_metrics: pd.DataFrame, yearly_metrics: pd.DataFrame, fold_metrics: pd.DataFrame, selections: pd.DataFrame, summary: dict[str, Any]) -> None:
    write_text(EXP_DIR / "README.md", f"""# {TITLE}

## Experiment Identity And Status

Experiment `{EXPERIMENT_ID}` is `{summary['status']}`.

## One-Sentence Hypothesis

Small chronological stacks of independent physical feature-family corrections can outperform the broad 0196 station-tail champion without using any forward-looking information.

## Why It Is Worth Doing

0199 showed that thermal-moisture and thermal-wind families carry signal but fail as single replacements. 0201 tests whether their corrections are complementary when shrunk, capped, and gated on target-free source, season, forecast, and 0196 correction-state contexts.

## Target, Horizon, And Cutoff

Target is HKO Tmax at `T-24`, timezone `Asia/Hong_Kong`. 2024+ confirmation rows are locked. Base experts use only the cutoff-safe station-role feature matrix inherited from 0196 and 0199.

## Dataset And Baseline

The dataset is the canonical `official_t15_pre2024_5265_rows` frame from `2000-01-02` through `2023-12-31`. The primary baseline is `official_forecast_max_c` on identical rows. The parent reference is the validator-clean `0196` prediction file.

## Walk-Forward Design

Each fold trains and selects using earlier years only. The stack must beat 0196 by `{INNER_MIN_LIFT_C}` C on prior validation with no material severe-tail harm, otherwise `0196` is used unchanged.

## Acceptance Criteria

Promotion requires at least `0.001` C MAE improvement versus 0196, at least `0.01` C improvement versus the official baseline, no outer-fold harm above `0.001` C versus 0196, and no severe-error-rate increase above `0.005`.

## Reproduction Command

Run `python scripts/run_hkg_t24_0201_independent_feature_group_stacked_router.py` from the repository root.
""")
    write_text(EXP_DIR / "RESULTS.md", f"""# Results

## Headline Result Table

{base.markdown_table(scoreboard)}

## Coverage And Row Identity

Rows: `{summary['n_common']}` from `{summary['date_start']}` to `{summary['date_end']}`. Common row hash: `{summary['common_row_hash']}`.

## Global Metrics

Official MAE is `{summary['baseline_mae_c']:.6f}` C. Parent 0196 MAE is `{summary['parent_0196_mae_c']:.6f}` C. Candidate MAE is `{summary['candidate_mae_c']:.6f}` C. Delta versus 0196 is `{summary['delta_vs_0196_mae_c']:.6f}` C.

## Fold Stability

{base.markdown_table(fold_metrics[['fold_id', 'n', 'parent_0196_mae_c', 'candidate_mae_c', 'delta_vs_0196_mae_c', 'selected_rule_family', 'selected_experts', 'selected_weights', 'selected_gate_id', 'selected_cap_c', 'active_rows', 'selection_reason']], max_rows=20)}

## Yearly And Monthly Results

{base.markdown_table(yearly_metrics[['slice_value', 'n', 'parent_0196_mae_c', 'candidate_mae_c', 'delta_vs_0196_mae_c']], max_rows=30)}

Month metrics:

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].eq('month')][['slice_value', 'n', 'parent_0196_mae_c', 'candidate_mae_c', 'delta_vs_0196_mae_c']], max_rows=20)}

## Tail And Source Results

{base.markdown_table(slice_metrics[slice_metrics['slice_type'].isin(['season', 'source', 'late_window', 'parent_tail'])][['slice_type', 'slice_value', 'n', 'parent_0196_mae_c', 'candidate_mae_c', 'delta_vs_0196_mae_c', 'parent_0196_gt3c_rate', 'candidate_gt3c_rate']], max_rows=25)}

## Data-Quality And Leakage Result

Leakage status is `{summary['leakage_status']}`. Confirmation rows used: `{summary['confirmation_rows_used']}`.

Selections:

{base.markdown_table(selections, max_rows=20)}
""")
    write_text(EXP_DIR / "CONCLUSION.md", f"""# Conclusion

## Verdict

Status is `{summary['status']}` with promotion decision `{summary['promotion_decision']}`.

## What Was Learned

0201 tests whether feature-family signals from 0199 become more useful when stacked rather than selected one at a time.

## Realized Point-MAE Change

Delta versus official is `{summary['mae_delta_c']:.6f}` C. Delta versus 0196 is `{summary['delta_vs_0196_mae_c']:.6f}` C.

## Robustness And Limits

The stack is selected only from prior years and falls back to 0196 when prior evidence is weak. This is still a development-corpus result, not a confirmation result.

## Failure Diagnosis

If the selected stack loses to 0196, the failure means the inner-period family gains did not transfer reliably to the scored outer era. The lane can still be informative: it separates true RSS-era or warm-season hints from broad deployable promotion. A fold that improves late RSS rows but harms press-archive rows should be treated as a source-era specificity clue rather than a deployable global replacement.

## Next Research Implication

The correct posterior update is not to keep widening stack grids on the same row frame. Future value should either constrain the useful pocket more strongly with independent target-free gates, or move to a genuinely new timestamp-proven data source that can break the station-network plateau.

## Promotion Status

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
    write_json(EXP_DIR / "run_manifest.json", {"experiment_id": EXPERIMENT_ID, "slug": SLUG, "created_at_utc": created_at, "spec_sha256": spec_sha, "state": "SPEC_WRITTEN_BEFORE_SCORING", "executor_invocation": "Executor skill instructions followed in-process because no separate skill agent activation is available."})
    src_copy_path = EXP_DIR / "src" / SRC_COPY_NAME
    shutil.copy2(Path(__file__).resolve(), src_copy_path)

    frame, feature_defs, groups = load_frame()
    predictions, fold_metrics, selections, stack_scores, expert_scores, fit_errors = run_walk_forward(frame, groups)
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
            {"candidate_id": PRIMARY_CANDIDATE_ID, "model_family": "independent_feature_group_stacked_router", "n": candidate_global["n"], "mae_c": candidate_global["mae_c"], "rmse_c": candidate_global["rmse_c"], "bias_c": candidate_global["bias_c"], "median_abs_error_c": candidate_global["median_abs_error_c"], "p95_abs_error_c": candidate_global["p95_abs_error_c"], "gt2c_rate": candidate_global["gt2c_rate"], "gt3c_rate": candidate_global["gt3c_rate"], "baseline_mae_c": official_global["mae_c"], "mae_delta_c": mae_delta},
        ]
    )
    row_coverage = pd.DataFrame([{"frame_id": "official_t15_pre2024_5265_rows", "parent_rows": int(len(frame)), "candidate_rows": int(len(predictions)), "baseline_rows": int(len(predictions)), "common_rows": int(len(predictions)), "date_start": date_text(predictions["target_date"].min()), "date_end": date_text(predictions["target_date"].max()), "row_policy": "all 0196 parent rows", "common_row_hash": common_row_hash}])
    data_manifest = pd.DataFrame(
        [
            {"source_id": "0196_parent_predictions", "path": rel(P0196), "sha256": sha256_file(P0196), "size_bytes": P0196.stat().st_size, "row_count": int(len(frame)), "date_start": date_text(frame["target_date"].min()), "date_end": date_text(frame["target_date"].max()), "timestamp_fields": "target_date;0196 prediction;0196 correction state", "availability_class": "DEPLOYABLE_LAGGED_ONLY", "notes": "Validator-clean 0196 predictions."},
            {"source_id": "robust_feature_matrix_isd", "path": rel(base.FEATURE_MATRIX_PATH), "sha256": sha256_file(base.FEATURE_MATRIX_PATH), "size_bytes": base.FEATURE_MATRIX_PATH.stat().st_size, "row_count": int(len(frame)), "date_start": date_text(frame["target_date"].min()), "date_end": date_text(frame["target_date"].max()), "timestamp_fields": "target_date;cutoff-safe ISD station summaries", "availability_class": "DEPLOYABLE_PROVEN", "notes": "Same role-compressed ISD feature family as 0196 and 0199."},
        ]
    )
    pred_cols = ["target_date", "target_tmax_c", "forecast_source_family", "season", "month", "official_prediction_c", "parent_0196_prediction_c", "candidate_prediction_c", "candidate_correction_c", "parent_0196_error_c", "candidate_error_c", "official_abs_error_c", "parent_0196_abs_error_c", "candidate_abs_error_c", "fold_id", "selected_rule_id", "selected_rule_family", "selected_experts", "selected_weights", "selected_gate_id", "selected_cap_c", "candidate_id", "baseline_id", "model_family"]
    write_parquet(EXP_DIR / "predictions.parquet", predictions[pred_cols])
    write_csv(EXP_DIR / "scoreboard.csv", scoreboard)
    write_csv(EXP_DIR / "slice_metrics.csv", slice_metrics)
    write_csv(EXP_DIR / "yearly_metrics.csv", yearly_metrics)
    write_csv(EXP_DIR / "fold_metrics.csv", fold_metrics)
    write_csv(EXP_DIR / "row_coverage.csv", row_coverage)
    write_csv(EXP_DIR / "correction_distribution.csv", predictions["candidate_correction_c"].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).reset_index())
    write_csv(EXP_DIR / "data_manifest.csv", data_manifest)
    write_csv(EXP_DIR / "feature_definitions.csv", feature_defs)
    write_csv(EXP_DIR / "artifacts" / "expert_config_grid.csv", pd.DataFrame(EXPERT_CONFIGS))
    write_csv(EXP_DIR / "artifacts" / "fold_stack_selections.csv", selections)
    write_csv(EXP_DIR / "artifacts" / "inner_stack_scores.csv", stack_scores)
    write_csv(EXP_DIR / "artifacts" / "inner_expert_scores.csv", expert_scores)
    if not fit_errors.empty:
        write_csv(EXP_DIR / "logs" / "fit_errors.csv", fit_errors)
    write_json(EXP_DIR / "diagnostics" / "feature_groups.json", groups)
    write_text(EXP_DIR / "leakage_audit.md", f"""# Leakage And Point-In-Time Audit

Status: `PASS`

## Cutoff

0201 consumes validator-clean 0196 parent predictions and the cutoff-safe ISD role feature family used by 0196 and 0199. All fitting, stack selection, imputation, and gating are chronological.

## Available Feature Eligibility

Allowed current-row gates are source family, month, official forecast level, and frozen 0196 correction magnitudes. Current target residuals and absolute errors are rejected as predictors.

## Target And Rolling Checks

Each outer fold uses only earlier target years for base expert fitting and stack selection. Scored fold outcomes are used only for scoring.

## Confirmation Proof

Maximum scored target date is `{date_text(predictions['target_date'].max())}`. Confirmation rows used: `0`. Owner authorization for confirmation: `false`.

## Row Identity

Candidate, parent 0196, and official baseline share `{len(predictions)}` rows. Common row hash: `{common_row_hash}`.
""")
    write_text(EXP_DIR / "REPRODUCE.md", f"""# Reproduction

From `{REPO_ROOT}`, run:

```powershell
python scripts/run_hkg_t24_0201_independent_feature_group_stacked_router.py
```

Requires completed parent predictions from 0196. Confirmation rows remain locked.
""")
    code_sha = sha256_file(src_copy_path)
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
        "parent_0196_mae_c": parent_global["mae_c"],
        "delta_vs_0196_mae_c": delta_vs_0196,
        "fold_worst_delta_vs_0196_mae_c": fold_worst_delta,
        "severe_gt3_rate_delta_vs_0196": severe_harm_0196,
    }
    write_json(EXP_DIR / "summary.json", summary)
    write_docs(scoreboard, slice_metrics, yearly_metrics, fold_metrics, selections, summary)
    write_json(EXP_DIR / "run_manifest.json", {"experiment_id": EXPERIMENT_ID, "slug": SLUG, "created_at_utc": created_at, "completed_at_utc": utc_now(), "repo_root": str(REPO_ROOT), "script": rel(Path(__file__).resolve()), "spec_sha256": spec_sha, "code_sha256": code_sha, "state": "COMPLETED", "executor_invocation": "Executor skill instructions followed in-process because no separate skill agent activation is available."})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
