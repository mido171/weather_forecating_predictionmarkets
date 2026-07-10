"""Run HKG Tmax probability bucket calibration V1 benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from hkg_tmax_probability.bucket_rules import BUCKET_KEYS, PROBABILITY_COLUMNS
from hkg_tmax_probability.data_build import (
    DEFAULT_DATABASE_URL,
    build_modeling_table,
    write_modeling_artifacts,
)
from hkg_tmax_probability.label_publication_audit import (
    apply_first_publication_labels,
    run_label_publication_audit,
)
from hkg_tmax_probability.leakage_audit import audit_modeling_table, write_leakage_audit
from hkg_tmax_probability.live_inference import write_live_inference_example
from hkg_tmax_probability.models import predict_all_methods
from hkg_tmax_probability.reporting import (
    add_leaderboard_rank_and_gates,
    bootstrap_deltas,
    grouped_scoreboard,
    probability_predictions_frame,
    score_predictions,
    write_diagnostics,
    write_manifest,
    write_model_summary,
)
from hkg_tmax_probability.validation import split_windows_from_config, train_validation_frames

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _run_window(
    modeling: pd.DataFrame,
    config: dict[str, Any],
    window,
    cutoff_profile: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    frame = modeling.copy()
    if cutoff_profile is None:
        primary_only = True
        suffix = ""
    else:
        frame = frame[frame["cutoff_profile"] == cutoff_profile].copy()
        primary_only = False
        suffix = f"__{cutoff_profile}"
    train, validation = train_validation_frames(frame, window, primary_only=primary_only)
    if train.empty or validation.empty:
        return pd.DataFrame(), {"window": window.name, "skipped": True, "reason": "empty train or validation"}, pd.DataFrame()
    outputs, details = predict_all_methods(train, validation, config)
    prediction_frames = [
        probability_predictions_frame(validation, method, output.family, output.probabilities, window.name + suffix)
        for method, output in outputs.items()
    ]
    predictions = pd.concat(prediction_frames, ignore_index=True)
    stack_weights = []
    for method, detail in details.items():
        if method == "S1_conservative_simplex_stack":
            for base_method, weight in detail.get("weights", {}).items():
                stack_weights.append({"validation_split": window.name + suffix, "base_method": base_method, "weight": weight})
    log = {"window": window.name + suffix, "method_details": details}
    return predictions, log, pd.DataFrame(stack_weights)


def _aggregate_primary_leaderboard(predictions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    include = predictions[
        predictions["validation_split"].isin(["fold1", "fold2", "fold3", "fold4", "presealed_2022_2023"])
        & predictions["is_primary_cutoff"]
    ].copy()
    b4_score = score_predictions(include[include["method"] == "B4_hierarchical_residual_pmf"], "fold1_4_plus_presealed_primary")
    b4_rps = float(b4_score["rps"].iloc[0]) if not b4_score.empty else None
    scored = score_predictions(include, "fold1_4_plus_presealed_primary", b4_rps=b4_rps)
    return add_leaderboard_rank_and_gates(scored, config["acceptance_gates"])


def _write_scoreboards(output_dir: Path, predictions: pd.DataFrame, leaderboard: pd.DataFrame, config: dict[str, Any]) -> None:
    leaderboard.to_csv(output_dir / "scoreboard.csv", index=False)
    split_scores = []
    for split, group in predictions[predictions["is_primary_cutoff"]].groupby("validation_split", sort=False):
        b4 = score_predictions(group[group["method"] == "B4_hierarchical_residual_pmf"], split)
        b4_rps = float(b4["rps"].iloc[0]) if not b4.empty else None
        split_scores.append(score_predictions(group, split, b4_rps=b4_rps))
    pd.concat(split_scores, ignore_index=True).to_csv(output_dir / "scoreboard_by_split.csv", index=False)
    grouped_scoreboard(predictions[predictions["is_primary_cutoff"]], "target_month").to_csv(output_dir / "scoreboard_by_month.csv", index=False)
    grouped_scoreboard(predictions[(predictions["is_primary_cutoff"]) & (predictions["target_month"] == 7)], "target_month").to_csv(
        output_dir / "scoreboard_july.csv", index=False
    )
    grouped_scoreboard(predictions[predictions["is_primary_cutoff"]], "season").to_csv(output_dir / "scoreboard_by_season.csv", index=False)
    grouped_scoreboard(predictions[predictions["is_primary_cutoff"]], "official_max_bin").to_csv(
        output_dir / "scoreboard_by_official_max_bin.csv", index=False
    )
    grouped_scoreboard(predictions[predictions["is_primary_cutoff"]], "issue_hour_hkt").to_csv(
        output_dir / "scoreboard_by_issue_hour.csv", index=False
    )
    grouped_scoreboard(predictions[predictions["is_primary_cutoff"]], "revision_direction").to_csv(
        output_dir / "scoreboard_by_revision_direction.csv", index=False
    )
    grouped_scoreboard(predictions, "cutoff_profile").to_csv(output_dir / "scoreboard_by_cutoff.csv", index=False)
    calibration_scores = []
    for split, group in predictions[predictions["family"] == "calibration"].groupby("validation_split", sort=False):
        calibration_scores.append(score_predictions(group, str(split)))
    if calibration_scores:
        pd.concat(calibration_scores, ignore_index=True).to_csv(output_dir / "calibration_layer_scoreboard.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "calibration_layer_scoreboard.csv", index=False)
    bootstrap_deltas(
        predictions,
        baseline_method="B4_hierarchical_residual_pmf",
        iterations=int(config.get("bootstrap", {}).get("iterations", 500)),
        seed=int(config.get("bootstrap", {}).get("seed", 20260705)),
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


def _first_publication_scoreboard(output_dir: Path, predictions: pd.DataFrame, label_audit: pd.DataFrame, details: dict[str, Any]) -> None:
    if not details.get("scoreboard_required"):
        pd.DataFrame().to_csv(output_dir / "first_publication_scoreboard.csv", index=False)
        return
    merged = predictions.merge(
        label_audit[["target_date", "first_publication_bucket_key"]], on="target_date", how="left"
    )
    key_to_idx = {bucket: idx for idx, bucket in enumerate(BUCKET_KEYS)}
    merged["bucket_index"] = merged["first_publication_bucket_key"].fillna(merged["bucket_key"]).map(key_to_idx).astype(int)
    merged["bucket_key"] = merged["first_publication_bucket_key"].fillna(merged["bucket_key"])
    leaderboard = _aggregate_primary_leaderboard(merged, {"acceptance_gates": {"nll_worse_than_b4_max": 0.005, "brier_worse_than_b4_max": 0.002}})
    leaderboard.to_csv(output_dir / "first_publication_scoreboard.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "experiments" / "hkg_tmax" / "probability_bucket_v1.yaml")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "experiments"
        / "campaigns"
        / "probability"
        / "buckets-v1"
        / "results",
    )
    parser.add_argument("--database-url", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    database_url = args.database_url or config.get("database_url") or DEFAULT_DATABASE_URL

    modeling, selected, eligible, row_audit = build_modeling_table(config, database_url=database_url)
    write_modeling_artifacts(output_dir, modeling, selected, eligible, row_audit)
    leakage = audit_modeling_table(modeling)
    write_leakage_audit(output_dir, leakage)
    label_audit, label_details = run_label_publication_audit(modeling, database_url=database_url, output_dir=output_dir)
    apply_first_publication_labels(modeling, label_audit).to_parquet(output_dir / "modeling_table_with_first_publication_labels.parquet", index=False)

    windows = split_windows_from_config(config)
    all_predictions = []
    selection_logs = []
    stack_weights = []
    for window in windows:
        predictions, log, weights = _run_window(modeling, config, window)
        if not predictions.empty:
            all_predictions.append(predictions)
        selection_logs.append(log)
        if not weights.empty:
            stack_weights.append(weights)

    primary_predictions = pd.concat(all_predictions, ignore_index=True)

    # Cutoff sensitivity: run the same modeling suite per cutoff profile.  These
    # rows are used for cutoff scoreboards only; the primary leaderboard above is
    # still strictly T-1 23:59 HKT.
    sensitivity_predictions = []
    for cutoff in modeling["cutoff_profile"].drop_duplicates().tolist():
        if cutoff == "t_minus_1_2359_hkt":
            continue
        for window in windows:
            predictions, log, weights = _run_window(modeling, config, window, cutoff_profile=cutoff)
            if not predictions.empty:
                sensitivity_predictions.append(predictions)
            selection_logs.append(log)
            if not weights.empty:
                stack_weights.append(weights)
    predictions = pd.concat([primary_predictions, *sensitivity_predictions], ignore_index=True)

    leaderboard = _aggregate_primary_leaderboard(primary_predictions, config)
    _write_scoreboards(output_dir, predictions, leaderboard, config)
    _write_probability_artifacts(output_dir, predictions)
    write_diagnostics(output_dir, predictions)
    if stack_weights:
        pd.concat(stack_weights, ignore_index=True).to_csv(output_dir / "stack_weights.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "stack_weights.csv", index=False)

    row_gate = _row_identity_gate(primary_predictions)
    (output_dir / "row_identity_gate.json").write_text(json.dumps(row_gate, indent=2, default=str), encoding="utf-8")
    (output_dir / "model_selection_log.json").write_text(json.dumps(selection_logs, indent=2, default=str), encoding="utf-8")
    method_details = []
    for log in selection_logs:
        for method, details in log.get("method_details", {}).items():
            method_details.append({"validation_split": log["window"], "method": method, "details_json": json.dumps(details, default=str)})
    pd.DataFrame(method_details).to_parquet(output_dir / "distribution_params.parquet", index=False)
    _first_publication_scoreboard(output_dir, primary_predictions, label_audit, label_details)

    champion_method = str(leaderboard[leaderboard["champion_flag"]]["method"].iloc[0])
    write_live_inference_example(output_dir, primary_predictions, champion_method)
    write_model_summary(
        output_dir,
        leaderboard,
        leakage,
        label_details,
        config.get("acceptance_gates", {}),
    )
    artifact_names = [path.name for path in output_dir.iterdir()]
    write_manifest(output_dir, args.config.resolve(), artifact_names)

    print("HKG Tmax probability bucket V1 complete")
    print(f"Output: {output_dir}")
    print(leaderboard.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
