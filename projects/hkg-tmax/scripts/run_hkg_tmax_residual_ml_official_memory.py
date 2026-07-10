from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from hkg_tmax.data.official_residual_memory_features import build_residual_memory_features
from hkg_tmax.evaluation.official_residual_memory_runner import (
    run_official_residual_memory_experiment,
)
from hkg_tmax.evaluation.reporting import (
    artifact_manifest,
    feature_missingness_report,
    markdown_table,
    source_eligibility_audit,
    write_csv,
    write_json,
    write_parquet,
    write_text,
)
from hkg_tmax.features.leakage_guards import next_round_leakage_audit_payload
from hkg_tmax.features.pruned_feature_policy import (
    feature_policy_report,
    validate_pruned_features,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "experiments" / "hkg_tmax" / "residual_ml_official_memory.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "experiments" / "hkg_tmax" / "0003_official_residual_memory_20260706" / "results"
DEFAULT_COMPAT_OUTPUT = REPO_ROOT / "experiments" / "hkg_tmax_residual_ml_official_memory" / "results"
DEFAULT_DATABASE_URL = "postgresql://postgres:root@127.0.0.1:5432/hkg_tmax_research"


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def log(message: str) -> None:
    print(f"[hkg_tmax_official_memory] {utc_now()} {message}", flush=True)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def read_previous_artifacts(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrices = [
        pd.read_parquet(results_dir / "feature_matrix_trainval.parquet"),
        pd.read_parquet(results_dir / "feature_matrix_presealed_holdout.parquet"),
        pd.read_parquet(results_dir / "feature_matrix_sealed_confirmation.parquet"),
    ]
    matrix = pd.concat(matrices, ignore_index=True)
    matrix["target_date"] = pd.to_datetime(matrix["target_date"], errors="coerce").dt.normalize()
    predictions = pd.read_parquet(results_dir / "prediction_rows.parquet")
    predictions["target_date"] = pd.to_datetime(predictions["target_date"], errors="coerce").dt.normalize()
    lineage = pd.DataFrame(json.loads((results_dir / "feature_lineage.json").read_text(encoding="utf-8")))
    source_eligibility = pd.read_csv(results_dir / "source_eligibility_audit.csv")
    if "target_date" in source_eligibility:
        source_eligibility["target_date"] = pd.to_datetime(source_eligibility["target_date"], errors="coerce").dt.normalize()
    return matrix, predictions, lineage, source_eligibility


def compact_score_summary(scoreboard: pd.DataFrame, primary_cutoff: str) -> dict[str, Any]:
    primary = scoreboard[scoreboard["cutoff_profile"].eq(primary_cutoff)].copy()

    def row(model_id: str) -> dict[str, Any] | None:
        hit = primary[primary["model_id"].eq(model_id)]
        if hit.empty:
            return None
        first = hit.iloc[0]
        return {
            "mae": float(first["mae"]),
            "rmse": float(first["rmse"]),
            "p90_absolute_error": float(first["p90_absolute_error"]),
            "n_scored": int(first["n_scored"]),
        }

    return {
        "A0_raw_official": row("A0_raw_official"),
        "A7_final_residual_ensemble": row("A7_final_residual_ensemble"),
        "D1_official_residual_memory_shrinkage": row("D1_official_residual_memory_shrinkage"),
        "D2_A3_plus_residual_memory_lgbm": row("D2_A3_plus_residual_memory_lgbm"),
        "D3_pruned_full_plus_residual_memory_lgbm": row("D3_pruned_full_plus_residual_memory_lgbm"),
        "D4_residual_memory_constrained_stack": row("D4_residual_memory_constrained_stack"),
        "D5_conservative_A7_plus_memory_blend": row("D5_conservative_A7_plus_memory_blend"),
    }


def model_card(summary: dict[str, Any], scoreboard: pd.DataFrame) -> str:
    primary = scoreboard[scoreboard["cutoff_profile"].eq(summary["primary_cutoff_profile"])].copy()
    keep = primary[
        primary["model_id"].isin(
            [
                "A0_raw_official",
                "A7_final_residual_ensemble",
                "D0_A7_reproduction",
                "D1_official_residual_memory_shrinkage",
                "D2_A3_plus_residual_memory_lgbm",
                "D3_pruned_full_plus_residual_memory_lgbm",
                "D4_residual_memory_constrained_stack",
                "D5_conservative_A7_plus_memory_blend",
            ]
        )
    ].sort_values("mae")
    table = markdown_table(keep, max_rows=20)
    promotion = summary.get("promotion", {})
    return f"""# HKG Tmax Official Residual Memory Model Card

Experiment: `hkg_tmax_0003_official_residual_memory_20260706`.

Primary question: whether lag-safe official forecast residual memory can improve the existing A7 residual-ML research candidate enough to justify promotion over the raw official forecast.

Scope: point forecasting only. No probability buckets, Polymarket prices, EV, sizing, PnL, or trading features were used.

Primary target: HKO Daily Extract Absolute Daily Max in deg. C.

Primary anchor: latest strict Info.gov local forecast maximum before `T-1 23:59 HKT`.

Residual definition: `actual_tmax_c - selected_official_forecast_max_c`.

Residual-memory rule: for target date `T`, the newest allowed residual source date is `T-2`. Lag-1 residuals are disabled.

## Decision

Decision: `{promotion.get("decision")}`.

Reason: `{promotion.get("reason")}`.

Leakage audit: `{summary.get("leakage_audit_status")}`.

Row-identity gate: `{summary.get("row_identity_gate", {}).get("status")}`.

Publication-safety audit: `{summary.get("residual_memory_publication_safety_status")}`.

## Primary Score Rows

{table}

## Interpretation

`D5_conservative_A7_plus_memory_blend` is the only promotable candidate. It can promote only if it clears the predeclared development, presealed, no-harm, row-identity, leakage, and sealed report-only reversal gates. If the gate result is `no_promote`, deployment remains raw official forecast while A7 remains a research reference.
"""


def summary_payload(
    *,
    config: dict[str, Any],
    result: Any,
    leakage: dict[str, Any],
    residual_memory_safety: dict[str, Any],
    feature_count: int,
    memory_feature_count: int,
    output_dir: Path,
) -> dict[str, Any]:
    primary_cutoff = config.get("primary_cutoff_profile", "tminus1_2359")
    return {
        "generated_at_utc": utc_now(),
        "experiment_id": "hkg_tmax_0003_official_residual_memory_20260706",
        "primary_cutoff_profile": primary_cutoff,
        "scope": "point_forecast_only_no_trading_no_probability",
        "feature_count_before_memory": int(feature_count),
        "residual_memory_feature_count": int(memory_feature_count),
        "primary_score_summary": compact_score_summary(result.scoreboards["scoreboard"], primary_cutoff),
        "promotion": result.promotion,
        "leakage_audit_status": leakage.get("status"),
        "residual_memory_publication_safety_status": residual_memory_safety.get("status"),
        "row_identity_gate": result.row_identity_gate,
        "output_dir": str(output_dir),
        "config": config,
    }


def write_experiment_docs(exp_dir: Path, *, config_path: Path, summary: dict[str, Any]) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    try:
        config_display = str(config_path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        config_display = str(config_path)
    write_text(
        exp_dir / "README.md",
        f"""# 0003 Official Residual Memory

Status: `{summary.get("promotion", {}).get("decision")}`.

Purpose: test whether lag-safe memory of prior official forecast residuals improves HKG Tmax point forecasts beyond the A7 residual-ML research candidate.

Runner: `scripts/run_hkg_tmax_residual_ml_official_memory.py`

Config: `{config_display}`

Results: `results/`
""",
    )
    write_text(
        exp_dir / "HYPOTHESIS.md",
        """# Hypothesis

Recent lag-safe official forecast residuals contain short-term forecaster, station, and regime bias that is not fully captured by official max, forecast revisions, HKO hourly state, station gradients, calendar, or target climatology.
""",
    )
    write_text(
        exp_dir / "PROTOCOL.md",
        """# Protocol

- Primary cutoff: `T-1 23:59 HKT`.
- Sensitivity cutoffs: `T-1 21:00 HKT`, `T-1 18:00 HKT`.
- Fold 1-4 are used for model and hyperparameter selection.
- 2022-2023 is presealed holdout after candidate freeze.
- 2024-2026-05 is sealed confirmation and report-only.
- Residual-memory predictors use same-cutoff official residuals from `T-2` or older.
- Lag-1 residuals, target-date residuals, raw audit payloads, helped/worsened labels, raw error bins, and sealed labels are excluded from predictors.
""",
    )
    write_text(
        exp_dir / "ASOF_CONTRACT.md",
        """# As-Of Contract

For prediction target date `T` at cutoff `c`, a residual-memory source row for prior date `d` is eligible only when `d <= T-2` and the prior official anchor for `d` was selected using the same cutoff profile `c`.

The selected target-day official anchor itself must be the latest eligible Info.gov local forecast row with issue time at or before the cutoff.
""",
    )
    write_text(
        exp_dir / "DATA_MANIFEST.yaml",
        """sources:
  official_forecasts: public.hko_historical_forecasts_2000_2026
  label_core: label_core.hko_daily_tmax
  sealed_confirmation: sealed_confirmation.hko_daily_tmax
  target_history: feature_safe.hko_target_history_pre2024
  hourly_readings: public.hko_info_gov_hourly_readings_1998_2026
input_artifacts:
  previous_results: experiments/hkg_tmax_residual_ml_strategy/results
""",
    )
    write_text(
        exp_dir / "REPRODUCE.md",
        f"""# Reproduce

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_hkg_tmax_residual_ml_official_memory.py --config {config_path} --output-dir {exp_dir / "results"}
```
""",
    )
    write_text(
        exp_dir / "STATUS.yaml",
        f"""status: {summary.get("promotion", {}).get("decision")}
primary_conclusion: "{summary.get("promotion", {}).get("reason")}"
leakage: {summary.get("leakage_audit_status")}
row_identity: {summary.get("row_identity_gate", {}).get("status")}
reproducible: true
""",
    )


def run(config_path: Path, output_dir: Path, compat_output_dir: Path | None, database_url: str | None = None) -> dict[str, Any]:
    del database_url  # This runner reuses already audited PostgreSQL-backed artifacts.
    config = load_config(config_path)
    seed = int(config.get("seed", 20260706))
    previous_dir = REPO_ROOT / config.get("input_artifacts", {}).get("previous_results_dir", "experiments/hkg_tmax_residual_ml_strategy/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"loading previous A7 artifacts from {previous_dir}")
    matrix, previous_predictions, lineage, source_eligibility = read_previous_artifacts(previous_dir)
    cutoff_profiles = list(config.get("cutoff_profiles", ["tminus1_2359", "tminus1_2100", "tminus1_1800"]))
    memory_config = config.get("residual_memory", {})
    min_counts_cfg = memory_config.get("min_counts", {})
    min_counts = {
        7: int(min_counts_cfg.get("roll7", 4)),
        14: int(min_counts_cfg.get("roll14", 7)),
        30: int(min_counts_cfg.get("roll30", 15)),
        60: int(min_counts_cfg.get("roll60", 30)),
    }
    log("building lag-safe official residual-memory features")
    memory_result = build_residual_memory_features(
        matrix,
        cutoff_profiles=sorted(matrix["cutoff_profile"].dropna().astype(str).unique().tolist()),
        min_lag_days=int(memory_config.get("min_lag_days", 2)),
        min_counts=min_counts,
        large_threshold_c=float(memory_config.get("large_residual_threshold_c", 1.5)),
        lag1_enabled=bool(memory_config.get("lag1_enabled", False)),
    )
    matrix = memory_result.frame
    lineage = pd.concat([lineage, pd.DataFrame(memory_result.lineage_rows)], ignore_index=True)
    policy = validate_pruned_features(
        matrix,
        max_raw_features=int(config.get("feature_policy", {}).get("max_raw_features", 90)),
    )
    feature_names = policy.feature_names
    if len(feature_names) + len(memory_result.feature_names) > int(config.get("feature_policy", {}).get("max_features_with_residual_memory", 130)):
        raise ValueError("Pruned feature set plus residual-memory block exceeds configured feature cap")
    log(f"selected pruned features={len(feature_names)} residual_memory_features={len(memory_result.feature_names)}")
    result = run_official_residual_memory_experiment(
        matrix,
        previous_predictions,
        cutoff_profiles=cutoff_profiles,
        feature_names=feature_names,
        memory_feature_names=memory_result.feature_names,
        config=config,
        seed=seed,
    )
    leakage = next_round_leakage_audit_payload(
        matrix,
        lineage,
        feature_names=[*feature_names, *memory_result.feature_names],
        router_thresholds={"sealed_rows_used_for_selection": False, "selection_stage": "rolling_validation_fold1_to_fold4"},
        router_predictions=pd.DataFrame(),
    )
    if memory_result.publication_safety_audit.get("status") != "pass":
        leakage["status"] = "fail"
        leakage["residual_memory_publication_safety_audit"] = memory_result.publication_safety_audit
    summary = summary_payload(
        config=config,
        result=result,
        leakage=leakage,
        residual_memory_safety=memory_result.publication_safety_audit,
        feature_count=len(feature_names),
        memory_feature_count=len(memory_result.feature_names),
        output_dir=output_dir,
    )
    log("writing artifacts")
    write_json(output_dir / "summary.json", summary)
    write_text(output_dir / "model_card.md", model_card(summary, result.scoreboards["scoreboard"]))
    for name, frame in result.scoreboards.items():
        write_csv(output_dir / f"{name}.csv", frame)
    write_csv(output_dir / "residual_memory_feature_audit.csv", memory_result.feature_audit)
    write_json(output_dir / "residual_memory_publication_safety_audit.json", memory_result.publication_safety_audit)
    write_json(output_dir / "leakage_audit.json", leakage)
    write_json(
        output_dir / "row_count_audit.json",
        {
            "matrix_rows": int(len(matrix)),
            "previous_prediction_rows": int(len(previous_predictions)),
            "combined_prediction_rows": int(len(result.predictions)),
            "candidate_rows": int(len(result.candidate_rows)),
            "rows_by_cutoff": matrix.groupby("cutoff_profile").size().astype(int).to_dict(),
            "prediction_rows_by_model": result.predictions.groupby("model_id").size().astype(int).to_dict(),
        },
    )
    write_json(output_dir / "row_identity_gate.json", result.row_identity_gate)
    write_json(output_dir / "feature_lineage.json", lineage.to_dict(orient="records"))
    write_json(output_dir / "model_selection_log.json", result.model_selection_log)
    write_json(output_dir / "ensemble_weights.json", result.ensemble_weights)
    write_csv(output_dir / "feature_missingness_report.csv", feature_missingness_report(matrix, [*feature_names, *memory_result.feature_names]))
    write_csv(output_dir / "feature_policy_report.csv", feature_policy_report(matrix, max_raw_features=int(config.get("feature_policy", {}).get("max_raw_features", 90))))
    write_csv(output_dir / "feature_importance_lgbm.csv", result.feature_importance)
    write_csv(output_dir / "source_eligibility_audit.csv", source_eligibility_audit(matrix))
    write_csv(output_dir / "previous_source_eligibility_audit.csv", source_eligibility)
    write_csv(output_dir / "prediction_rows.csv", result.predictions)
    write_parquet(output_dir / "prediction_rows.parquet", result.predictions)
    write_parquet(output_dir / "prediction_rows_candidates.parquet", result.candidate_rows)
    write_csv(output_dir / "prediction_rows_candidates.csv", result.candidate_rows)
    write_csv(output_dir / "artifact_manifest.csv", artifact_manifest(output_dir))
    write_experiment_docs(output_dir.parent, config_path=config_path, summary=summary)
    if compat_output_dir is not None:
        compat_output_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(output_dir, compat_output_dir, dirs_exist_ok=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HKG Tmax official residual-memory point-forecast experiment")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--compat-output-dir", default=str(DEFAULT_COMPAT_OUTPUT))
    parser.add_argument("--no-compat-copy", action="store_true")
    parser.add_argument(
        "--database-url",
        default=os.environ.get("HKG_TMAX_DATABASE_URL") or os.environ.get("DATABASE_URL") or DEFAULT_DATABASE_URL,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(
        config_path=Path(args.config),
        output_dir=Path(args.output_dir),
        compat_output_dir=None if args.no_compat_copy else Path(args.compat_output_dir),
        database_url=args.database_url,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
