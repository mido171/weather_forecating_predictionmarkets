"""Run HKG Tmax probability distribution-method V2 benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from hkg_tmax_probability.bucket_rules import BUCKET_KEYS, PROBABILITY_COLUMNS
from hkg_tmax_probability.data_build import DEFAULT_DATABASE_URL, build_modeling_table, write_modeling_artifacts
from hkg_tmax_probability.distribution_methods_v2 import (
    distribution_v2_predictor_columns,
    method_details_frame,
    predict_distribution_methods_v2,
)
from hkg_tmax_probability.label_publication_audit import apply_first_publication_labels, run_label_publication_audit
from hkg_tmax_probability.leaderboard_v2 import apply_v2_champion_gates
from hkg_tmax_probability.leakage_audit import audit_modeling_table, write_leakage_audit
from hkg_tmax_probability.live_inference import write_live_inference_example
from hkg_tmax_probability.models import predict_all_methods
from hkg_tmax_probability.reporting import (
    bootstrap_deltas,
    grouped_scoreboard,
    probability_predictions_frame,
    score_predictions,
    write_diagnostics,
)
from hkg_tmax_probability.validation import split_windows_from_config, train_validation_frames

REPO_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_SELECTION_SPLITS = ["fold1", "fold2", "fold3", "fold4", "presealed_2022_2023"]
FOLD14_SPLITS = ["fold1", "fold2", "fold3", "fold4"]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _run_window(
    modeling: pd.DataFrame,
    config: dict[str, Any],
    window,
    cutoff_profile: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    frame = modeling.copy()
    if cutoff_profile is None:
        primary_only = True
        suffix = ""
    else:
        frame = frame[frame["cutoff_profile"] == cutoff_profile].copy()
        primary_only = False
        suffix = f"__{cutoff_profile}"
    train, validation = train_validation_frames(frame, window, primary_only=primary_only)
    split_label = window.name + suffix
    if train.empty or validation.empty:
        return (
            pd.DataFrame(),
            {"window": split_label, "skipped": True, "reason": "empty train or validation"},
            pd.DataFrame(),
            pd.DataFrame(),
        )

    base_outputs, base_details = predict_all_methods(train, validation, config)
    v2_outputs, v2_details, continuous_params = predict_distribution_methods_v2(train, validation, config, base_outputs=base_outputs)
    outputs = {**base_outputs, **v2_outputs}
    details = {**base_details, **v2_details}
    prediction_frames = [
        probability_predictions_frame(validation, method, output.family, output.probabilities, split_label)
        for method, output in outputs.items()
    ]
    predictions = pd.concat(prediction_frames, ignore_index=True)
    if not continuous_params.empty:
        continuous_params["validation_split"] = split_label

    stack_weights = []
    for method, detail in details.items():
        if method == "S1_conservative_simplex_stack":
            for base_method, weight in detail.get("weights", {}).items():
                stack_weights.append({"validation_split": split_label, "base_method": base_method, "weight": weight, "stack": method})
        if method == "H1_b4_challenger_linear_pool":
            stack_weights.append(
                {
                    "validation_split": split_label,
                    "base_method": "B4_hierarchical_residual_pmf",
                    "weight": detail.get("b4_weight", 1.0),
                    "stack": method,
                }
            )
            if detail.get("selected_challenger"):
                stack_weights.append(
                    {
                        "validation_split": split_label,
                        "base_method": detail.get("selected_challenger"),
                        "weight": detail.get("challenger_weight", 0.0),
                        "stack": method,
                    }
                )
    log = {"window": split_label, "method_details": details}
    return predictions, log, pd.DataFrame(stack_weights), continuous_params


def _score_subset(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    b4_score = score_predictions(frame[frame["method"] == "B4_hierarchical_residual_pmf"], split_name)
    b4_rps = float(b4_score["rps"].iloc[0]) if not b4_score.empty else None
    return score_predictions(frame, split_name, b4_rps=b4_rps)


def _aggregate_primary_leaderboard(
    primary_predictions: pd.DataFrame,
    config: dict[str, Any],
    leakage: dict[str, Any],
    row_gate: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = primary_predictions[primary_predictions["is_primary_cutoff"]].copy()
    fold14_frame = primary[primary["validation_split"].isin(FOLD14_SPLITS)].copy()
    presealed_frame = primary[primary["validation_split"] == "presealed_2022_2023"].copy()
    overall_frame = primary[primary["validation_split"].isin(PRIMARY_SELECTION_SPLITS)].copy()
    fold14 = _score_subset(fold14_frame, "fold1_4_primary")
    presealed = _score_subset(presealed_frame, "presealed_2022_2023_primary")
    overall = _score_subset(overall_frame, "fold1_4_plus_presealed_primary")
    leaderboard = apply_v2_champion_gates(
        overall,
        fold14,
        presealed,
        config["acceptance_gates"],
        leakage_pass=leakage.get("status") == "pass",
        row_identity_pass=row_gate.get("status") == "pass",
    )
    return leaderboard, fold14, presealed


def _with_high_bucket_group(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    high = {"31", "32", "33", "34_or_higher"}
    out["high_bucket_group"] = np.where(out["bucket_key"].isin(high), out["bucket_key"], "other")
    return out


def _write_scoreboards(output_dir: Path, predictions: pd.DataFrame, leaderboard: pd.DataFrame, config: dict[str, Any]) -> None:
    leaderboard.to_csv(output_dir / "scoreboard.csv", index=False)
    primary = predictions[predictions["is_primary_cutoff"]].copy()
    split_scores = []
    for split, group in primary.groupby("validation_split", sort=False):
        split_scores.append(_score_subset(group, str(split)))
    pd.concat(split_scores, ignore_index=True).to_csv(output_dir / "scoreboard_by_split.csv", index=False)
    grouped_scoreboard(primary, "target_month").to_csv(output_dir / "scoreboard_by_month.csv", index=False)
    grouped_scoreboard(primary[primary["target_month"] == 7], "target_month").to_csv(output_dir / "scoreboard_july.csv", index=False)
    grouped_scoreboard(primary, "season").to_csv(output_dir / "scoreboard_by_season.csv", index=False)
    grouped_scoreboard(primary, "official_max_bin").to_csv(output_dir / "scoreboard_by_official_max_bin.csv", index=False)
    grouped_scoreboard(primary, "issue_hour_hkt").to_csv(output_dir / "scoreboard_by_issue_hour.csv", index=False)
    grouped_scoreboard(primary, "revision_direction").to_csv(output_dir / "scoreboard_by_revision_direction.csv", index=False)
    grouped_scoreboard(predictions, "cutoff_profile").to_csv(output_dir / "scoreboard_by_cutoff.csv", index=False)
    grouped_scoreboard(_with_high_bucket_group(primary), "high_bucket_group").to_csv(output_dir / "scoreboard_by_high_bucket.csv", index=False)
    calibration_scores = []
    for split, group in primary[primary["family"] == "calibration"].groupby("validation_split", sort=False):
        calibration_scores.append(score_predictions(group, str(split)))
    if calibration_scores:
        pd.concat(calibration_scores, ignore_index=True).to_csv(output_dir / "calibration_layer_scoreboard.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "calibration_layer_scoreboard.csv", index=False)
    bootstrap_deltas(
        predictions,
        baseline_method="B4_hierarchical_residual_pmf",
        iterations=int(config.get("bootstrap", {}).get("iterations", 500)),
        seed=int(config.get("bootstrap", {}).get("seed", 20260706)),
    ).to_csv(output_dir / "proper_score_deltas_bootstrap.csv", index=False)


def _write_probability_artifacts(output_dir: Path, predictions: pd.DataFrame) -> None:
    predictions.to_parquet(output_dir / "per_fold_predictions.parquet", index=False)
    predictions[["target_date", "cutoff_profile", "validation_split", "method", *PROBABILITY_COLUMNS]].to_parquet(
        output_dir / "bucket_probabilities.parquet", index=False
    )
    long_probs = predictions.melt(
        id_vars=["target_date", "cutoff_profile", "validation_split", "method", "row_identity"],
        value_vars=list(PROBABILITY_COLUMNS),
        var_name="bucket_probability_column",
        value_name="probability",
    )
    long_probs["bucket_key"] = long_probs["bucket_probability_column"].str.replace("p_", "", regex=False)
    midpoint_map = {
        "24_or_below": 24.5,
        "25": 25.5,
        "26": 26.5,
        "27": 27.5,
        "28": 28.5,
        "29": 29.5,
        "30": 30.5,
        "31": 31.5,
        "32": 32.5,
        "33": 33.5,
        "34_or_higher": 34.5,
    }
    long_probs["representative_tmax_c"] = long_probs["bucket_key"].map(midpoint_map)
    long_probs.to_parquet(output_dir / "one_decimal_pmfs.parquet", index=False)


def _row_identity_gate(predictions: pd.DataFrame) -> dict[str, Any]:
    rows = []
    violations = 0
    for split, split_group in predictions.groupby("validation_split"):
        expected = None
        for method, group in split_group.groupby("method"):
            current = set(group["row_identity"])
            if expected is None:
                expected = current
            same = current == expected
            if not same:
                violations += 1
            rows.append({"validation_split": split, "method": method, "row_count": len(current), "identical_to_split_first": same})
    return {"status": "pass" if violations == 0 else "fail", "violations": violations, "by_method": rows}


def _first_publication_scoreboard(
    output_dir: Path,
    primary_predictions: pd.DataFrame,
    label_audit: pd.DataFrame,
    label_details: dict[str, Any],
    config: dict[str, Any],
    leakage: dict[str, Any],
    row_gate: dict[str, Any],
) -> None:
    if not label_details.get("scoreboard_required"):
        pd.DataFrame().to_csv(output_dir / "first_publication_scoreboard.csv", index=False)
        return
    merged = primary_predictions.merge(label_audit[["target_date", "first_publication_bucket_key"]], on="target_date", how="left")
    key_to_idx = {bucket: idx for idx, bucket in enumerate(BUCKET_KEYS)}
    merged["bucket_index"] = merged["first_publication_bucket_key"].fillna(merged["bucket_key"]).map(key_to_idx).astype(int)
    merged["bucket_key"] = merged["first_publication_bucket_key"].fillna(merged["bucket_key"])
    leaderboard, _, _ = _aggregate_primary_leaderboard(merged, config, leakage, row_gate)
    leaderboard.to_csv(output_dir / "first_publication_scoreboard.csv", index=False)


def _write_model_card(output_dir: Path, leaderboard: pd.DataFrame, leakage: dict[str, Any], label_audit: dict[str, Any], row_gate: dict[str, Any]) -> None:
    champion = leaderboard[leaderboard["champion_flag"]].iloc[0]
    top = leaderboard.head(12)[
        [
            "rank",
            "method",
            "family",
            "rps",
            "relative_rps_gain_vs_b4",
            "fold14_relative_rps_gain_vs_b4",
            "presealed_relative_rps_gain_vs_b4",
            "nll",
            "brier",
            "gates",
            "champion_flag",
        ]
    ]
    lines = [
        "# HKG Tmax Probability Distribution Methods V2 Model Card",
        "",
        f"Supreme method after V2 gates: `{champion['method']}`.",
        "",
        "Scope: weather probability distribution only. No market prices, EV, order books, Kelly sizing, PnL, market-implied blending, or trade recommendations are used or emitted.",
        "",
        "Target: HKO Daily Extract one-decimal HKG daily maximum temperature bucket.",
        "Forecast surface: strict HKO Info.gov local forecast rows selected at the configured pre-target cutoffs.",
        "Primary cutoff: T-1 23:59 HKT. Sensitivity cutoffs: T-1 18:00 and T-1 21:00 HKT.",
        "",
        "Promotion rule: challengers must beat B4 by at least 1.5% RPS on folds 1-4 and 1.0% RPS on the 2022-2023 presealed holdout, while not worsening NLL by more than 0.005 or Brier by more than 0.002.",
        "",
        f"Champion normalized RPS: {champion['rps']:.6f}",
        f"Champion NLL: {champion['nll']:.6f}",
        f"Champion Brier: {champion['brier']:.6f}",
        f"Champion ECE: {champion['ece']:.6f}",
        f"Champion gates: `{champion['gates']}`",
        "",
        f"Leakage audit: `{leakage.get('status')}` with total violations `{leakage.get('total_violations')}`.",
        f"Row-identity gate: `{row_gate.get('status')}` with violations `{row_gate.get('violations')}`.",
        f"Label first-publication audit: `{label_audit.get('status')}`, bucket changes `{label_audit.get('bucket_changes')}`.",
        "",
        "## Top Leaderboard Rows",
        "",
        _markdown_table(top),
        "",
        "## Methods Benchmarked",
        "",
        "- V1 baselines and champion family: B0-B6, P1/P2, C1/C2, K0-K2, S1.",
        "- V2 challengers: E1 normal EMOS, E2 Student-t EMOS, E3 two-piece normal EMOS, G1 tree location-scale, Q1 quantile CDF gradient boosting, Q2 threshold CDF gradient boosting, T1 time-decay B4, H1 conservative B4-plus-challenger pool.",
    ]
    (output_dir / "final_probability_model_card.md").write_text("\n".join(lines), encoding="utf-8")


def _write_supreme_summary(output_dir: Path, leaderboard: pd.DataFrame, fold14: pd.DataFrame, presealed: pd.DataFrame) -> None:
    champion = leaderboard[leaderboard["champion_flag"]].iloc[0]
    raw_best = leaderboard.iloc[0]
    lines = [
        "# HKG Tmax Probability Engine V2 Supreme Method Summary",
        "",
        f"Supreme method: `{champion['method']}`.",
        f"Raw lowest-RPS method: `{raw_best['method']}`.",
        "",
        "The supreme method is chosen by proper scoring rules plus the predeclared V2 promotion gates. A challenger can have an attractive raw score and still fail promotion if its fold 1-4 gain, presealed gain, NLL, Brier, leakage, or row-identity contract does not clear the gate.",
        "",
        "## Supreme Row",
        "",
        _markdown_table(leaderboard[leaderboard["champion_flag"]]),
        "",
        "## Raw Leaderboard Top 20",
        "",
        _markdown_table(leaderboard.head(20)),
        "",
        "## Fold 1-4 Scoreboard",
        "",
        _markdown_table(fold14.sort_values("rps").head(20)),
        "",
        "## Presealed 2022-2023 Scoreboard",
        "",
        _markdown_table(presealed.sort_values("rps").head(20)),
        "",
        "Interpretation rule: B4 remains the default champion unless a challenger clears all promotion gates. This prevents choosing a more complex probability engine from a marginal, unstable, or poorly calibrated score difference.",
    ]
    (output_dir / "supreme_method_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.6f}")
        else:
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else str(value))
    columns = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in display.iterrows():
        values = [str(row[column]).replace("|", "\\|") for column in display.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_manifest(output_dir: Path, config_path: Path, artifact_names: list[str]) -> None:
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        git_commit = "unknown"
    artifacts = []
    for name in sorted(artifact_names):
        path = output_dir / name
        if not path.exists() or path.is_dir():
            continue
        artifacts.append({"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
    manifest = {
        "generated_by": "scripts/run_hkg_tmax_probability_distribution_methods_v2.py",
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit,
        "config_path": str(config_path),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    (output_dir / "reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "experiments" / "hkg_tmax" / "probability_distribution_methods_v2.yaml")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "experiments" / "hkg_tmax_probability_distribution_methods_v2" / "results",
    )
    parser.add_argument("--database-url", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    database_url = args.database_url or config.get("database_url") or DEFAULT_DATABASE_URL

    modeling, selected, eligible, row_audit = build_modeling_table(config, database_url=database_url)
    write_modeling_artifacts(output_dir, modeling, selected, eligible, row_audit)
    leakage = audit_modeling_table(modeling, predictor_columns=distribution_v2_predictor_columns())
    write_leakage_audit(output_dir, leakage)
    label_audit, label_details = run_label_publication_audit(modeling, database_url=database_url, output_dir=output_dir)
    apply_first_publication_labels(modeling, label_audit).to_parquet(output_dir / "modeling_table_with_first_publication_labels.parquet", index=False)

    windows = split_windows_from_config(config)
    all_predictions = []
    selection_logs = []
    stack_weights = []
    continuous_params = []
    for window in windows:
        predictions, log, weights, params = _run_window(modeling, config, window)
        if not predictions.empty:
            all_predictions.append(predictions)
        selection_logs.append(log)
        if not weights.empty:
            stack_weights.append(weights)
        if not params.empty:
            continuous_params.append(params)

    primary_predictions = pd.concat(all_predictions, ignore_index=True)

    sensitivity_predictions = []
    for cutoff in modeling["cutoff_profile"].drop_duplicates().tolist():
        if cutoff == "t_minus_1_2359_hkt":
            continue
        for window in windows:
            predictions, log, weights, params = _run_window(modeling, config, window, cutoff_profile=cutoff)
            if not predictions.empty:
                sensitivity_predictions.append(predictions)
            selection_logs.append(log)
            if not weights.empty:
                stack_weights.append(weights)
            if not params.empty:
                continuous_params.append(params)

    predictions = pd.concat([primary_predictions, *sensitivity_predictions], ignore_index=True)
    row_gate = _row_identity_gate(primary_predictions)
    (output_dir / "row_identity_gate.json").write_text(json.dumps(row_gate, indent=2, default=str), encoding="utf-8")

    leaderboard, fold14, presealed = _aggregate_primary_leaderboard(primary_predictions, config, leakage, row_gate)
    _write_scoreboards(output_dir, predictions, leaderboard, config)
    _write_probability_artifacts(output_dir, predictions)
    write_diagnostics(output_dir, predictions)

    if stack_weights:
        pd.concat(stack_weights, ignore_index=True).to_csv(output_dir / "stack_weights.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "stack_weights.csv", index=False)

    (output_dir / "method_selection_log.json").write_text(json.dumps(selection_logs, indent=2, default=str), encoding="utf-8")
    method_details_frame(selection_logs).to_parquet(output_dir / "method_details.parquet", index=False)
    if continuous_params:
        pd.concat(continuous_params, ignore_index=True).to_parquet(output_dir / "continuous_distribution_params.parquet", index=False)
    else:
        pd.DataFrame().to_parquet(output_dir / "continuous_distribution_params.parquet", index=False)
    _first_publication_scoreboard(output_dir, primary_predictions, label_audit, label_details, config, leakage, row_gate)

    champion_method = str(leaderboard[leaderboard["champion_flag"]]["method"].iloc[0])
    write_live_inference_example(output_dir, primary_predictions, champion_method)
    _write_model_card(output_dir, leaderboard, leakage, label_details, row_gate)
    _write_supreme_summary(output_dir, leaderboard, fold14, presealed)
    artifact_names = [path.name for path in output_dir.iterdir()]
    _write_manifest(output_dir, args.config.resolve(), artifact_names)

    print("HKG Tmax probability distribution methods V2 complete")
    print(f"Output: {output_dir}")
    print(leaderboard.head(25).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
