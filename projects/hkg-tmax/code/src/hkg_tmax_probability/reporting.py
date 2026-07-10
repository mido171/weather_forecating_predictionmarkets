"""Artifact reporting for the HKG Tmax probability bucket experiment."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hkg_tmax_probability.bucket_rules import BUCKET_KEYS, PROBABILITY_COLUMNS, normalize_probability_matrix
from hkg_tmax_probability.scoring import calibration_errors, per_bucket_brier, summarize_scores


def probability_predictions_frame(base: pd.DataFrame, method: str, family: str, probs: np.ndarray, split: str) -> pd.DataFrame:
    matrix = normalize_probability_matrix(probs)
    out = base[
        [
            "target_date",
            "cutoff_profile",
            "cutoff_hkt",
            "is_primary_cutoff",
            "split_label",
            "bucket_key",
            "bucket_index",
            "target_tmax_c",
            "forecast_max_c",
            "forecast_min_c",
            "official_max_bin",
            "issue_hour_hkt",
            "revision_direction",
            "target_month",
            "season",
            "row_identity",
        ]
    ].copy()
    out["method"] = method
    out["family"] = family
    out["validation_split"] = split
    for idx, column in enumerate(PROBABILITY_COLUMNS):
        out[column] = matrix[:, idx]
    out["predicted_bucket_index"] = matrix.argmax(axis=1)
    out["predicted_bucket_key"] = out["predicted_bucket_index"].map(lambda value: BUCKET_KEYS[int(value)])
    out["max_probability"] = matrix.max(axis=1)
    return out


def score_predictions(predictions: pd.DataFrame, split_name: str, b4_rps: float | None = None) -> pd.DataFrame:
    rows = []
    for (method, family), group in predictions.groupby(["method", "family"], sort=False):
        probs = group[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
        summary = summarize_scores(probs, group["bucket_index"].to_numpy(dtype=int))
        delta = np.nan if b4_rps is None else summary.rps - b4_rps
        gain = np.nan if b4_rps is None or b4_rps == 0 else (b4_rps - summary.rps) / b4_rps
        rows.append(
            {
                "method": method,
                "family": family,
                "split": split_name,
                "row_count": summary.rows,
                "rps": summary.rps,
                "rps_delta_vs_b4": delta,
                "relative_rps_gain_vs_b4": gain,
                "nll": summary.nll,
                "brier": summary.brier,
                "crps": summary.crps,
                "ece": summary.ece,
                "mce": summary.mce,
                "entropy": summary.entropy,
            }
        )
    return pd.DataFrame(rows).sort_values("rps").reset_index(drop=True)


def add_leaderboard_rank_and_gates(scoreboard: pd.DataFrame, gates: dict[str, Any]) -> pd.DataFrame:
    out = scoreboard.sort_values("rps", ascending=True).reset_index(drop=True).copy()
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    b4 = out[out["method"] == "B4_hierarchical_residual_pmf"]
    b4_nll = float(b4["nll"].iloc[0]) if not b4.empty else np.nan
    b4_brier = float(b4["brier"].iloc[0]) if not b4.empty else np.nan
    gate_labels = []
    for _, row in out.iterrows():
        failures = []
        if not np.isnan(b4_nll) and row["nll"] > b4_nll + float(gates.get("nll_worse_than_b4_max", 0.005)):
            failures.append("nll")
        if not np.isnan(b4_brier) and row["brier"] > b4_brier + float(gates.get("brier_worse_than_b4_max", 0.002)):
            failures.append("brier")
        gate_labels.append("pass" if not failures else "fail:" + ",".join(failures))
    out["gates"] = gate_labels
    best_pass = out[out["gates"] == "pass"]
    champion = best_pass["method"].iloc[0] if not best_pass.empty else out["method"].iloc[0]
    if not b4.empty:
        simplicity_threshold = float(
            max(
                gates.get("complex_vs_b4_presealed_min_rps_gain", 0.010),
                gates.get("complex_vs_b4_fold14_min_rps_gain", 0.015),
            )
        )
        complex_pass = best_pass[
            (best_pass["method"] != "B4_hierarchical_residual_pmf")
            & (best_pass["relative_rps_gain_vs_b4"].astype(float) >= simplicity_threshold)
        ]
        champion = complex_pass["method"].iloc[0] if not complex_pass.empty else "B4_hierarchical_residual_pmf"
    out["champion_flag"] = out["method"].eq(champion)
    return out


def grouped_scoreboard(predictions: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for (group_value, method, family), group in predictions.groupby([group_column, "method", "family"], dropna=False, sort=False):
        if len(group) < 5:
            continue
        summary = summarize_scores(group[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float), group["bucket_index"].to_numpy(dtype=int))
        rows.append(
            {
                group_column: group_value,
                "method": method,
                "family": family,
                "row_count": summary.rows,
                "rps": summary.rps,
                "nll": summary.nll,
                "brier": summary.brier,
                "crps": summary.crps,
                "ece": summary.ece,
                "mce": summary.mce,
                "entropy": summary.entropy,
            }
        )
    return pd.DataFrame(rows).sort_values([group_column, "rps"]).reset_index(drop=True)


def write_diagnostics(output_dir: Path, predictions: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    reliability_rows = []
    bucket_brier_rows = []
    pit_rows = []
    interval_rows = []
    sharpness_rows = []
    for (split, method), group in predictions.groupby(["validation_split", "method"], sort=False):
        probs = group[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
        y = group["bucket_index"].to_numpy(dtype=int)
        _, _, rel = calibration_errors(probs, y)
        rel["validation_split"] = split
        rel["method"] = method
        reliability_rows.append(rel)
        brier = per_bucket_brier(probs, y)
        brier["validation_split"] = split
        brier["method"] = method
        bucket_brier_rows.append(brier)
        cdf = np.cumsum(normalize_probability_matrix(probs), axis=1)
        pit = np.array([cdf[i, max(y[i] - 1, 0)] if y[i] > 0 else 0.0 for i in range(len(y))]) + 0.5 * probs[np.arange(len(y)), y]
        pit_rows.append(pd.DataFrame({"validation_split": split, "method": method, "pit": pit}))
        lower90 = (cdf >= 0.05).argmax(axis=1)
        upper90 = (cdf >= 0.95).argmax(axis=1)
        covered90 = (y >= lower90) & (y <= upper90)
        interval_rows.append(
            pd.DataFrame(
                {
                    "validation_split": [split],
                    "method": [method],
                    "coverage90": [float(covered90.mean())],
                    "mean_width90_buckets": [float(np.mean(upper90 - lower90 + 1))],
                }
            )
        )
        sharpness_rows.append(
            pd.DataFrame(
                {
                    "validation_split": [split],
                    "method": [method],
                    "mean_max_probability": [float(probs.max(axis=1).mean())],
                    "mean_entropy": [float((-probs * np.log(np.clip(probs, 1e-12, 1))).sum(axis=1).mean())],
                }
            )
        )
    pd.concat(reliability_rows, ignore_index=True).to_csv(output_dir / "reliability_bins.csv", index=False)
    pd.concat(bucket_brier_rows, ignore_index=True).to_csv(output_dir / "one_vs_rest_brier_by_bucket.csv", index=False)
    pd.concat(pit_rows, ignore_index=True).to_csv(output_dir / "pit_values.csv", index=False)
    pd.concat(interval_rows, ignore_index=True).to_csv(output_dir / "interval_coverage.csv", index=False)
    pd.concat(sharpness_rows, ignore_index=True).to_csv(output_dir / "sharpness_diagnostics.csv", index=False)


def bootstrap_deltas(predictions: pd.DataFrame, baseline_method: str, iterations: int, seed: int) -> pd.DataFrame:
    primary = predictions[predictions["is_primary_cutoff"]].copy()
    methods = sorted(primary["method"].unique())
    rows = []
    rng = np.random.default_rng(seed)
    for split, split_group in primary.groupby("validation_split", sort=False):
        pivotable = split_group[["row_identity", "bucket_index", "method", *PROBABILITY_COLUMNS]].copy()
        identities = sorted(pivotable["row_identity"].unique())
        baseline = pivotable[pivotable["method"] == baseline_method].set_index("row_identity")
        if baseline.empty:
            continue
        method_groups = {method: pivotable[pivotable["method"] == method].set_index("row_identity") for method in methods}
        for method, method_group in method_groups.items():
            common = [rid for rid in identities if rid in baseline.index and rid in method_group.index]
            if not common:
                continue
            base_probs = baseline.loc[common, list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
            method_probs = method_group.loc[common, list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
            y = baseline.loc[common, "bucket_index"].to_numpy(dtype=int)
            base_scores = np.asarray(__import__("hkg_tmax_probability.scoring", fromlist=["ranked_probability_score"]).ranked_probability_score(base_probs, y))
            method_scores = np.asarray(__import__("hkg_tmax_probability.scoring", fromlist=["ranked_probability_score"]).ranked_probability_score(method_probs, y))
            deltas = []
            for _ in range(iterations):
                sample = rng.integers(0, len(common), size=len(common))
                deltas.append(float(method_scores[sample].mean() - base_scores[sample].mean()))
            rows.append(
                {
                    "split": split,
                    "method": method,
                    "baseline_method": baseline_method,
                    "row_count": len(common),
                    "mean_delta_rps": float(np.mean(deltas)),
                    "p05_delta_rps": float(np.quantile(deltas, 0.05)),
                    "p50_delta_rps": float(np.quantile(deltas, 0.50)),
                    "p95_delta_rps": float(np.quantile(deltas, 0.95)),
                }
            )
    return pd.DataFrame(rows)


def write_model_card(output_dir: Path, leaderboard: pd.DataFrame, leakage: dict[str, Any], label_audit: dict[str, Any]) -> None:
    champion = leaderboard[leaderboard["champion_flag"]].iloc[0]
    lines = [
        "# HKG Tmax Probability Bucket V1 Model Card",
        "",
        f"Champion: `{champion['method']}`",
        "",
        "Scope: weather probability distribution only. No market prices, EV, order books, Kelly sizing, PnL, or trade recommendations are used or emitted.",
        "",
        "Primary target: HKO Daily Extract one-decimal HKG daily maximum temperature bucket.",
        "",
        f"Primary normalized RPS: {champion['rps']:.6f}",
        f"NLL: {champion['nll']:.6f}",
        f"Brier: {champion['brier']:.6f}",
        f"ECE: {champion['ece']:.6f}",
        "",
        f"Leakage audit status: `{leakage.get('status')}` with total violations `{leakage.get('total_violations')}`.",
        f"Label first-publication audit: `{label_audit.get('status')}`, bucket changes `{label_audit.get('bucket_changes')}`.",
        "",
        "Selection rule: leaderboard sorted by normalized RPS ascending; methods must also pass no-worse NLL/Brier gates versus B4.",
    ]
    (output_dir / "final_probability_model_card.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(output_dir: Path, config_path: Path, artifact_names: list[str]) -> None:
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=output_dir.parents[2], text=True).strip()
    except Exception:
        git_commit = "unknown"
    artifacts = []
    for name in sorted(artifact_names):
        path = output_dir / name
        if not path.exists() or path.is_dir():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        artifacts.append({"path": name, "sha256": digest, "bytes": path.stat().st_size})
    manifest = {
        "generated_by": "scripts/run_hkg_tmax_probability_bucket_v1.py",
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit,
        "config_path": str(config_path),
        "artifacts": artifacts,
    }
    (output_dir / "reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
