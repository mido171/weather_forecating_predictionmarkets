from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
import yaml

from hkg_tmax.data.forecast_anchor import (
    CUTOFF_PROFILES,
    add_forecast_features,
    build_cutoff_frame,
    load_strict_forecasts,
    load_targets,
)
from hkg_tmax.data.hourly_readings_features import build_hourly_features, load_hourly_readings
from hkg_tmax.data.target_history_features import add_target_history_features, load_target_history
from hkg_tmax.evaluation.ablation_runner import run_ablation_pipeline
from hkg_tmax.evaluation.reporting import (
    artifact_manifest,
    feature_missingness_report,
    final_model_card,
    residual_diagnostics,
    row_count_audit,
    source_eligibility_audit,
    write_csv,
    write_json,
    write_parquet,
    write_text,
)
from hkg_tmax.features.feature_registry import FeatureRegistry
from hkg_tmax.features.leakage_guards import leakage_audit_payload


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "experiments" / "hkg_tmax" / "residual_ml_strategy.yaml"
DEFAULT_DATABASE_URL = "postgresql://postgres:root@127.0.0.1:5432/hkg_tmax_research"


def log(message: str) -> None:
    print(f"[hkg_tmax_residual_ml] {utc_now()} {message}", flush=True)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def redacted_url(url: str) -> str:
    if "://" not in url or "@" not in url:
        return url
    scheme, rest = url.split("://", maxsplit=1)
    userinfo, host = rest.split("@", maxsplit=1)
    user = userinfo.split(":", maxsplit=1)[0]
    return f"{scheme}://{user}:***@{host}"


def configured_profiles(names: list[str] | None) -> tuple[Any, ...]:
    if not names:
        return CUTOFF_PROFILES
    allowed = {profile.name: profile for profile in CUTOFF_PROFILES}
    return tuple(allowed[name] for name in names)


def flatten_features(families: dict[str, list[str]]) -> list[str]:
    out: list[str] = []
    for names in families.values():
        for name in names:
            if name not in out:
                out.append(name)
    return out


def build_matrix(
    *,
    database_url: str,
    config: dict[str, Any],
    cutoff_profiles: tuple[Any, ...],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any], FeatureRegistry]:
    date_params = {
        "start_date": config["dates"]["start_date"],
        "presealed_end_date": config["dates"]["presealed_end_date"],
        "sealed_start_date": config["dates"]["sealed_start_date"],
        "sealed_end_date": config["dates"]["sealed_end_date"],
    }
    registry = FeatureRegistry()
    with psycopg.connect(database_url) as connection:
        log("loading targets")
        targets = load_targets(connection, date_params)
        log(f"loaded targets rows={len(targets)}")
        log("loading strict Info.gov forecasts")
        forecasts = load_strict_forecasts(connection, date_params)
        log(f"loaded strict forecasts rows={len(forecasts)}")
        log("building cutoff frame and forecast features")
        base = build_cutoff_frame(targets, cutoff_profiles)
        matrix = add_forecast_features(base, forecasts, registry)
        log(f"forecast feature matrix rows={len(matrix)}")
        hourly_params = {
            "hourly_start_utc": (pd.to_datetime(matrix["cutoff_at_utc"], utc=True).min() - pd.Timedelta(hours=36)).to_pydatetime(),
            "hourly_end_utc": pd.to_datetime(matrix["cutoff_at_utc"], utc=True).max().to_pydatetime(),
        }
        log("loading Info.gov hourly readings")
        hourly = load_hourly_readings(connection, hourly_params)
        log(f"loaded hourly rows={len(hourly)}")
        log("building hourly and station-network features")
        matrix, hourly_reports = build_hourly_features(matrix, hourly, registry)
        log("loading lag2+ target history")
        history = load_target_history(connection)
        log(f"loaded target history rows={len(history)}")
        log("building target-history features")
        matrix, target_history_audit = add_target_history_features(matrix, history, registry)
        log(f"completed feature matrix rows={len(matrix)}")
    reports = dict(hourly_reports)
    reports["target_history_feature_audit"] = target_history_audit
    source_frames = {
        "targets": targets,
        "forecasts": forecasts,
        "hourly": hourly,
        "history": history,
    }
    metadata = {
        "target_rows": int(len(targets)),
        "forecast_rows": int(len(forecasts)),
        "hourly_rows": int(len(hourly)),
        "target_history_rows": int(len(history)),
    }
    return matrix, reports, {"source_frames": source_frames, "metadata": metadata}, registry


def write_outputs(
    *,
    output_dir: Path,
    matrix: pd.DataFrame,
    reports: dict[str, pd.DataFrame],
    source_context: dict[str, Any],
    registry: FeatureRegistry,
    leakage: dict[str, Any],
    run_result: dict[str, Any],
    config: dict[str, Any],
    database_url: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_names = flatten_features(registry.families)
    predictions = run_result["predictions"]
    scoreboards = run_result["scoreboards"]
    model_selection = run_result["model_selection"]
    schema = registry.schema_frame(matrix)
    lineage = registry.lineage_frame()
    trainval = matrix[matrix["target_date"].le(pd.Timestamp("2021-12-31"))].copy()
    presealed = matrix[matrix["target_date"].between(pd.Timestamp("2022-01-01"), pd.Timestamp("2023-12-31"))].copy()
    sealed = matrix[matrix["target_date"].ge(pd.Timestamp("2024-01-01"))].copy()
    write_text(
        output_dir / "README.md",
        "# HKG Tmax Residual ML Strategy Results\n\n"
        "Generated by `scripts/run_hkg_tmax_residual_ml_strategy.py`.\n\n"
        "This experiment is a leakage-audited research run, not a production trading release.\n",
    )
    write_json(
        output_dir / "row_count_audit.json",
        row_count_audit(
            targets=source_context["source_frames"]["targets"],
            forecasts=source_context["source_frames"]["forecasts"],
            hourly=source_context["source_frames"]["hourly"],
            matrix=matrix,
            predictions=predictions,
        ),
    )
    write_json(output_dir / "leakage_audit.json", leakage)
    write_csv(output_dir / "source_eligibility_audit.csv", source_eligibility_audit(matrix))
    write_parquet(output_dir / "feature_matrix_trainval.parquet", trainval)
    write_parquet(output_dir / "feature_matrix_presealed_holdout.parquet", presealed)
    write_parquet(output_dir / "feature_matrix_sealed_confirmation.parquet", sealed)
    write_json(output_dir / "feature_matrix_schema.json", schema.to_dict(orient="records"))
    write_json(output_dir / "feature_lineage.json", lineage.to_dict(orient="records"))
    write_csv(output_dir / "feature_missingness_report.csv", feature_missingness_report(matrix, feature_names))
    for name, frame in reports.items():
        filename = f"{name}.csv"
        write_csv(output_dir / filename, frame if frame is not None else pd.DataFrame())
    forecast_dist = (
        matrix[matrix["forecast_selector_status"].eq("selected")]
        .groupby(["cutoff_profile", "eligible_forecast_count"])
        .size()
        .reset_index(name="rows")
    )
    write_csv(output_dir / "forecast_revision_count_distribution.csv", forecast_dist)
    for filename in (
        "scoreboard.csv",
        "scoreboard_by_split.csv",
        "scoreboard_by_month.csv",
        "scoreboard_by_regime.csv",
        "ablation_scoreboard.csv",
        "cutoff_sensitivity_scoreboard.csv",
    ):
        write_csv(output_dir / filename, scoreboards.get(filename.removesuffix(".csv"), pd.DataFrame()))
    write_parquet(output_dir / "prediction_rows.parquet", predictions)
    write_csv(output_dir / "prediction_rows.csv", predictions)
    write_json(output_dir / "model_selection_log.json", model_selection)
    write_json(
        output_dir / "ensemble_weights.json",
        {cutoff: payload.get("ensemble", {}) for cutoff, payload in model_selection.get("cutoffs", {}).items()},
    )
    importance = run_result["feature_importance"]
    write_csv(output_dir / "feature_importance_lgbm.csv", importance[importance["model_id"].astype(str).str.contains("lgbm", case=False, na=False)])
    write_csv(output_dir / "feature_importance_catboost.csv", importance[importance["model_id"].eq("M3_catboost_residual")])
    write_csv(output_dir / "linear_coefficients.csv", importance[importance["model_id"].eq("M4_huber_residual")])
    write_json(
        output_dir / "clipping_and_preprocessing_audit.json",
        {
            "residual_prediction_hard_cap_c": [-3.0, 3.0],
            "linear_numeric_imputation": "training-fold median",
            "tree_missing_values": "LightGBM/CatBoost native or encoded missing flags; categorical MISSING token",
            "sealed_rows_used_for_preprocessing": False,
        },
    )
    catboost_status = "fit"
    if importance[importance["model_id"].eq("M3_catboost_residual")].empty:
        catboost_status = "no_importance_rows"
    write_text(
        output_dir / "residual_error_diagnostics.md",
        residual_diagnostics(predictions, model_selection.get("promotion", {})),
    )
    write_text(
        output_dir / "final_model_card.md",
        final_model_card(
            promotion=model_selection.get("promotion", {}),
            scoreboard=scoreboards["scoreboard"],
            leakage_status=leakage["status"],
            feature_count=int(len(schema)),
            catboost_status=catboost_status,
        ),
    )
    summary = {
        "generated_at_utc": utc_now(),
        "database_url_redacted": redacted_url(database_url),
        "config": config,
        "source_counts": source_context["metadata"],
        "feature_rows": int(len(matrix)),
        "feature_count": int(len(schema)),
        "prediction_rows": int(len(predictions)),
        "leakage_status": leakage["status"],
        "promotion": model_selection.get("promotion", {}),
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "artifact_manifest.csv", artifact_manifest(output_dir))
    return summary


def run(config_path: Path, database_url: str, output_dir: Path, cutoff_profile_names: list[str] | None) -> dict[str, Any]:
    config = load_config(config_path)
    profiles = configured_profiles(cutoff_profile_names or config.get("cutoff_profiles"))
    matrix, reports, source_context, registry = build_matrix(
        database_url=database_url,
        config=config,
        cutoff_profiles=profiles,
    )
    lineage = registry.lineage_frame()
    log("running leakage audit")
    leakage = leakage_audit_payload(matrix, lineage)
    if leakage["status"] != "pass" and not config.get("allow_leakage_failures", False):
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "leakage_audit.json", leakage)
        raise SystemExit(f"Leakage audit failed with {leakage['total_violations']} violations")
    log("running ablation/model pipeline")
    run_result = run_ablation_pipeline(
        matrix,
        registry.families,
        cutoff_profiles=[profile.name for profile in profiles],
        seed=int(config.get("seed", 20260705)),
    )
    log("writing artifacts")
    return write_outputs(
        output_dir=output_dir,
        matrix=matrix,
        reports=reports,
        source_context=source_context,
        registry=registry,
        leakage=leakage,
        run_result=run_result,
        config=config,
        database_url=database_url,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HKG Tmax residual ML strategy")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--database-url",
        default=os.environ.get("HKG_TMAX_DATABASE_URL") or os.environ.get("DATABASE_URL") or DEFAULT_DATABASE_URL,
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "experiments" / "hkg_tmax_residual_ml_strategy" / "results"),
    )
    parser.add_argument("--cutoff-profile", default=None)
    parser.add_argument("--cutoff-profiles", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = None
    if args.cutoff_profile:
        profiles = [args.cutoff_profile]
    elif args.cutoff_profiles:
        profiles = [item.strip() for item in args.cutoff_profiles.split(",") if item.strip()]
    summary = run(
        config_path=Path(args.config),
        database_url=args.database_url,
        output_dir=Path(args.output_dir),
        cutoff_profile_names=profiles,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
