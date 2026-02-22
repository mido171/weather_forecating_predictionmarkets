"""Build MOS-based dataset with leakage-safe features and KNN analog blocks."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from .mos_bias_features import compute_bias_features
from .mos_calendar import add_calendar_features, build_calendar
from .mos_config import KnnViewConfig, MosDatasetConfig, load_config
from .mos_constants import DEFAULT_VARIABLES
from .mos_db import create_engine_from_config, load_db_config, load_db_config_from_env
from .mos_knn import compute_knn_features, prepare_distance_data
from .mos_knn_derived import add_knn_consistency_zscores, add_knn_cross_view_features
from .mos_mos_features import (
    add_cross_model_features,
    add_missing_flags,
    add_shape_features,
    add_revision_features,
    build_mos_pivots,
    build_bucket_pivots,
    compute_baseline_medians,
    compute_update_counts,
    fetch_mos_rows,
    select_latest_mos,
)
from .mos_obs_features import compute_obs_features, fetch_truth_rows
from .iem_minute_features import compute_iem_minute_features
from .mos_utils import hash_dict, sha256_hex, utc_now


LOGGER = logging.getLogger(__name__)


def compute_feature_order(df: pd.DataFrame) -> list[str]:
    id_cols = [
        "station_id",
        "station_zoneid",
        "asof_date_local",
        "target_date_local",
        "asof_utc",
        "feature_version",
        "config_hash",
        "knn_config_hash",
        "sql_extract_hash_mos",
        "sql_extract_hash_truth",
        "raw_payload_hash_ref_agg",
        "mos_retrieved_at_utc_max",
        "mos_runtime_utc_max",
    ]
    target = ["y_actual_tmax_f"]
    cal_cols = sorted([c for c in df.columns if c.startswith("cal_")])
    obs_cols = sorted([c for c in df.columns if c.startswith("obs_")])
    mos_cols = sorted(
        [
            c
            for c in df.columns
            if c.startswith("mos_")
            and not c.startswith("mos_shape_")
            and not c.startswith("mos_xmodel_")
        ]
    )
    mos_shape_cols = sorted([c for c in df.columns if c.startswith("mos_shape_")])
    mos_xmodel_cols = sorted(
        [c for c in df.columns if c.startswith("mos_xmodel_") or c.startswith("base_")]
    )
    bias_cols = sorted([c for c in df.columns if c.startswith("bias_")])
    iem_cols = sorted([c for c in df.columns if c.startswith("iem_")])
    knn_cols = sorted(
        [c for c in df.columns if c.startswith("knn_") and not c.startswith("knn_v0_nn")]
    )
    knn_slots = sorted([c for c in df.columns if c.startswith("knn_v0_nn")])
    int_cols = sorted([c for c in df.columns if c.startswith("int_")])

    ordered = (
        id_cols
        + target
        + cal_cols
        + obs_cols
        + mos_cols
        + mos_shape_cols
        + mos_xmodel_cols
        + bias_cols
        + iem_cols
        + knn_cols
        + knn_slots
        + int_cols
    )
    return [c for c in ordered if c in df.columns]


def build_dataset(cfg: MosDatasetConfig, engine) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = cfg.normalized()
    if not cfg.variables:
        cfg = MosDatasetConfig(**{**cfg.__dict__, "variables": DEFAULT_VARIABLES}).normalized()

    run_started = utc_now()

    cal_df = build_calendar(cfg)
    start_target = cal_df["target_date_local"].min()
    end_target = cal_df["target_date_local"].max()

    mos_raw, sql_hash_mos = fetch_mos_rows(engine, cfg, start_target, end_target)
    latest = select_latest_mos(mos_raw, cal_df, cfg)
    mos_pivot = build_mos_pivots(latest)
    if mos_pivot.empty:
        mos_pivot = cal_df[["target_date_local"]].copy()
    count_median, window_median = compute_baseline_medians(latest, cfg)
    mos_features = add_missing_flags(mos_pivot, cfg, count_median, window_median)
    mos_features = add_shape_features(mos_features, cfg)

    df = cal_df.merge(mos_features, on="target_date_local", how="left")
    bucket_hours = cfg.asof_buckets_hours or []
    if bucket_hours:
        bucket_pivots = build_bucket_pivots(mos_raw, cal_df, cfg, bucket_hours)
        for pivot in bucket_pivots.values():
            if pivot.empty:
                continue
            df = df.merge(pivot, on="target_date_local", how="left")
        df = add_revision_features(df, cfg, bucket_hours)
        update_counts = compute_update_counts(mos_raw, cal_df, cfg)
        if not update_counts.empty:
            df = df.merge(update_counts, on="target_date_local", how="left")
    df = add_cross_model_features(df, cfg)
    df = add_calendar_features(df, cfg)
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    df["asof_date_local"] = pd.to_datetime(df["asof_date_local"]).dt.date

    audit = latest.groupby("target_date_local").agg(
        mos_retrieved_at_utc_max=("retrieved_at_utc", "max"),
        mos_runtime_utc_max=("runtime_utc", "max"),
        raw_payload_hash_ref_list=("raw_payload_hash_ref", lambda x: sorted(set(x))),
    )
    audit["raw_payload_hash_ref_agg"] = audit["raw_payload_hash_ref_list"].apply(
        lambda vals: sha256_hex("|".join(vals)) if vals else sha256_hex("")
    )
    audit = audit.drop(columns=["raw_payload_hash_ref_list"]).reset_index()
    df = df.merge(audit, on="target_date_local", how="left")

    max_window = max(cfg.obs_windows_days or [0])
    truth_start = min(cfg.build_start_asof, cfg.baseline_start or cfg.build_start_asof)
    truth_start = truth_start - timedelta(days=max_window + cfg.obs_cutoff_lag_days + 2)
    truth_end = cfg.end_asof + timedelta(days=1)
    truth, sql_hash_truth = fetch_truth_rows(engine, cfg, truth_start, truth_end)
    if not truth.empty:
        truth = truth.rename(columns={"date_local": "target_date_local"})
    df = df.merge(
        truth[["target_date_local", "tmax_f"]].rename(columns={"tmax_f": "y_actual_tmax_f"}),
        on="target_date_local",
        how="left",
    )

    obs_features = compute_obs_features(cal_df, truth.rename(columns={"target_date_local": "date_local"}), cfg)
    if not obs_features.empty:
        df = df.merge(obs_features, on=["target_date_local", "asof_date_local"], how="left")

    minute_features = compute_iem_minute_features(cal_df, cfg, truth.rename(columns={"target_date_local": "date_local"}))
    if not minute_features.empty:
        df = df.merge(minute_features, on="target_date_local", how="left")

    df = df.sort_values("asof_date_local").reset_index(drop=True)

    knn_meta = {"skipped": False}
    if not cfg.skip_knn:
        distance_start = cfg.distance_calibration_start or cfg.build_start_asof
        distance_end = cfg.distance_calibration_end or cfg.output_start_asof - timedelta(days=1)
        calib_mask = (
            pd.to_datetime(df["asof_date_local"]) >= pd.to_datetime(distance_start)
        ) & (pd.to_datetime(df["asof_date_local"]) <= pd.to_datetime(distance_end))

        distance_features = cfg.distance_features or []
        missing = [col for col in distance_features if col not in df.columns]
        if missing:
            raise ValueError(f"Missing distance features: {missing}")

        distance_data = prepare_distance_data(
            df,
            distance_features,
            calib_mask.to_numpy(),
            cfg.missing_penalty,
            cfg.distance_feature_weights or {},
        )

        knn_views = cfg.knn_views or []
        if not knn_views:
            knn_views = [
                KnnViewConfig("v0", "full", "l2"),
                KnnViewConfig("v1", "season45", "l2"),
                KnnViewConfig("v2", "full", "cosine"),
                KnnViewConfig("v3", "season45", "cosine"),
                KnnViewConfig("v4", "full", "rank"),
                KnnViewConfig("v5", "season45", "rank"),
            ]

        thresholds = cfg.thresholds or list(range(50, 101, 2)) + [45, 105]
        consistency_features = [
            "base_tmax_blend",
            "base_tmax_abs_spread",
            "mos_gfs_n_x_max",
            "mos_nam_n_x_max",
            "mos_xmodel_blend_tmp_mean",
            "mos_xmodel_blend_dpt_mean",
            "mos_xmodel_blend_wsp_mean",
            "mos_xmodel_blend_vis_mean",
            "mos_xmodel_blend_cig_median",
            "mos_xmodel_blend_p12_max",
            "obs_tmax_last",
            "obs_tmax_vs_mean_30",
        ]

        knn_df, knn_meta = compute_knn_features(
            df,
            distance_data,
            knn_views,
            cfg.k,
            thresholds,
            cfg.tau_fixed or [0.8, 1.0, 1.6],
            45,
            cfg.obs_cutoff_lag_days,
            consistency_features,
            "base_tmax_blend",
            "y_actual_tmax_f",
        )
        df = pd.concat([df.reset_index(drop=True), knn_df.reset_index(drop=True)], axis=1)

        # Second-order KNN features: view disagreement and consistency z-scores.
        view_names = [v.name for v in knn_views]
        df = add_knn_cross_view_features(df, view_names=view_names)
        df = add_knn_consistency_zscores(df, view_names=view_names, consistency_features=consistency_features)
    else:
        knn_meta = {"skipped": True}
        knn_views = []
        distance_features = []
        thresholds = []

    # Bias features need to run after KNN so we can optionally compute KNN-forecast bias history.
    bias_features = compute_bias_features(df, cfg)
    if not bias_features.empty:
        df = df.merge(bias_features, on="target_date_local", how="left")

    df["int_hot_flag_90"] = (df["base_tmax_blend"] >= 90).astype(float)
    df["int_hot_flag_95"] = (df["base_tmax_blend"] >= 95).astype(float)
    df["int_spread_x_season_sin"] = df["base_tmax_abs_spread"] * df["cal_d_doy_sin"]
    df["int_spread_x_season_cos"] = df["base_tmax_abs_spread"] * df["cal_d_doy_cos"]
    knn_ess = df.get("knn_v0_ess_ratio", pd.Series(np.nan, index=df.index))
    knn_p50 = df.get("knn_v0_dist_p50", pd.Series(np.nan, index=df.index))
    knn_std = df.get("knn_v0_analog_std", pd.Series(np.nan, index=df.index))
    knn_p90 = df.get("knn_v0_analog_p_ge_90", pd.Series(np.nan, index=df.index))
    df["int_knn_reliability"] = knn_ess / (knn_p50 + 1e-6)
    df["int_analog_std_x_base"] = knn_std * df["base_tmax_blend"]
    df["int_bias60_x_base"] = df.get("bias_blend_mean_60") * df["base_tmax_blend"]
    df["int_recent_trend14_x_base"] = df.get("obs_tmax_slope_14") * df["base_tmax_blend"]
    df["int_prob_ge_90_x_spread"] = knn_p90 * df["base_tmax_abs_spread"]

    df["asof_date_local"] = pd.to_datetime(df["asof_date_local"]).dt.date
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date

    config_hash = hash_dict(cfg.to_canonical_dict())
    knn_config_hash = hash_dict(
        {
            "k": cfg.k,
            "knn_views": [asdict(v) for v in knn_views],
            "distance_features": distance_features,
            "distance_feature_weights": cfg.distance_feature_weights or {},
            "thresholds": thresholds,
            "missing_penalty": cfg.missing_penalty,
            "tau_fixed": cfg.tau_fixed or [0.8, 1.0, 1.6],
            "skip_knn": cfg.skip_knn,
        }
    )
    df["feature_version"] = cfg.feature_version
    df["config_hash"] = config_hash
    df["knn_config_hash"] = knn_config_hash
    df["sql_extract_hash_mos"] = sql_hash_mos
    df["sql_extract_hash_truth"] = sql_hash_truth

    df = df[(df["asof_date_local"] >= cfg.output_start_asof) & (df["asof_date_local"] <= cfg.end_asof)].copy()

    feature_list = compute_feature_order(df)
    df = df[feature_list]

    run_finished = utc_now()

    metadata = {
        "run_started_at_utc": run_started.isoformat(),
        "run_finished_at_utc": run_finished.isoformat(),
        "feature_version": cfg.feature_version,
        "config_hash": config_hash,
        "config": cfg.to_canonical_dict(),
        "knn_config_hash": knn_config_hash,
        "sql_extract_hash_mos": sql_hash_mos,
        "sql_extract_hash_truth": sql_hash_truth,
        "git_commit": _git_commit_hash(),
        "row_counts": {
            "total_days": int(len(cal_df)),
            "output_days": int(len(df)),
            "dropped_days": int(len(cal_df) - len(df)),
        },
        "missingness": _missingness_report(df),
        "knn_failures": knn_meta,
        "target_distribution_by_year": _target_distribution(df),
        "sample_hashes": {
            "first_5": df["raw_payload_hash_ref_agg"].head(5).tolist(),
            "last_5": df["raw_payload_hash_ref_agg"].tail(5).tolist(),
        },
        "environment": _environment_report(),
    }

    return df, metadata


def _missingness_report(df: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for prefix in ["cal_", "obs_", "mos_", "mos_shape_", "mos_xmodel_", "bias_", "knn_", "int_"]:
        cols = [c for c in df.columns if c.startswith(prefix)]
        if not cols:
            continue
        report[prefix.rstrip("_")] = float(df[cols].isna().mean().mean())
    mos_missing = {
        col.replace("_count", ""): float(df[col].isna().mean())
        for col in df.columns
        if col.startswith("mos_") and col.endswith("_count")
    }
    report["mos_missing_by_model_var"] = mos_missing
    return report


def _target_distribution(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    dist: dict[str, dict[str, float]] = {}
    if "y_actual_tmax_f" not in df.columns:
        return dist
    df_year = df.copy()
    df_year["year"] = pd.to_datetime(df_year["target_date_local"]).dt.year
    for year, group in df_year.groupby("year"):
        dist[str(year)] = {
            "mean": float(group["y_actual_tmax_f"].mean()),
            "std": float(group["y_actual_tmax_f"].std()),
            "count": int(group["y_actual_tmax_f"].notna().sum()),
        }
    return dist


def _environment_report() -> dict[str, Any]:
    import numpy
    import pandas
    import sklearn
    import sys

    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "sklearn": sklearn.__version__,
    }


def _git_commit_hash() -> str | None:
    import subprocess

    try:
        repo_root = Path(__file__).resolve().parents[3]
        value = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root)
        return value.decode("utf-8").strip()
    except Exception:
        return None


def write_outputs(df: pd.DataFrame, metadata: dict[str, Any], cfg: MosDatasetConfig) -> Path:
    output_root = Path(cfg.output_root)
    output_dir = output_root / "kmia_tmax_nextday" / cfg.feature_version / metadata["config_hash"]
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_list_path = output_dir / "feature_list.json"
    feature_list_path.write_text(json.dumps(list(df.columns), indent=2), encoding="utf-8")

    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if cfg.output_partition_yearly:
        df = df.copy()
        df["year"] = pd.to_datetime(df["asof_date_local"]).dt.year
        for year, group in df.groupby("year"):
            out_path = output_dir / f"features_{int(year)}.csv"
            tmp_path = out_path.with_suffix(".csv.tmp")
            group.drop(columns=["year"]).to_csv(tmp_path, index=False, na_rep="")
            tmp_path.replace(out_path)
    else:
        out_path = output_dir / "features.csv"
        tmp_path = out_path.with_suffix(".csv.tmp")
        df.to_csv(tmp_path, index=False, na_rep="")
        tmp_path.replace(out_path)

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MOS daily dataset for KMIA Tmax next-day.")
    parser.add_argument("--config", required=True, help="Path to dataset config JSON")
    parser.add_argument("--db-config", help="Path to DB config JSON")
    parser.add_argument("--db-env", action="store_true", help="Load DB config from env vars")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    if cfg.variables is None or not cfg.variables:
        cfg = cfg.normalized()
        cfg = MosDatasetConfig(**{**cfg.__dict__, "variables": DEFAULT_VARIABLES}).normalized()

    if args.db_env:
        db_cfg = load_db_config_from_env()
    elif args.db_config:
        db_cfg = load_db_config(args.db_config)
    else:
        raise ValueError("Provide --db-config or --db-env for database credentials.")

    engine = create_engine_from_config(db_cfg)
    df, metadata = build_dataset(cfg, engine)
    output_dir = write_outputs(df, metadata, cfg)
    LOGGER.info("Wrote dataset to %s", output_dir)


if __name__ == "__main__":
    main()
