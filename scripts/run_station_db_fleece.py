from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text

from ml_live.db.mysql import MysqlConfig, MySqlStore
from ml_live.features.e92_features import ENSEMBLE_COMPONENTS, build_feature_vector
from ml_live.runtime.clock import asof_from_target_date
from ml_live.runtime.paths import artifacts_root, models_dir


def _parse_date(value: str) -> date:
    cleaned = value.strip()
    if len(cleaned) == 8 and cleaned.isdigit():
        return datetime.strptime(cleaned, "%Y%m%d").date()
    return datetime.strptime(cleaned, "%Y-%m-%d").date()


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else float("nan")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if len(y_true) else float("nan")


def _bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred - y_true)) if len(y_true) else float("nan")


def _quantiles(series: pd.Series, qs: list[float]) -> dict[str, float]:
    if series.empty:
        return {str(q): float("nan") for q in qs}
    return {str(q): float(series.quantile(q)) for q in qs}


@dataclass(frozen=True)
class TrainStats:
    mins: dict[str, float]
    maxs: dict[str, float]
    quantiles: dict[str, dict[str, float]]


def _mysql_config_from_env() -> MysqlConfig:
    return MysqlConfig(
        host=os.environ.get("MYSQL_HOST", "localhost"),
        port=int(os.environ.get("MYSQL_PORT", "3306")),
        database=os.environ.get("MYSQL_DB", "weather_predictionmarkets"),
        user=os.environ.get("MYSQL_USER", "root"),
        password=os.environ.get("MYSQL_PASSWORD", "root"),
    )


def _load_station_features(store: MySqlStore, station_id: str, start: date, end: date) -> pd.DataFrame:
    sql = """
        SELECT *
        FROM live_features_daily
        WHERE station_id = :station_id
          AND target_date_local BETWEEN :start_date AND :end_date
        ORDER BY target_date_local ASC
    """
    params = {"station_id": station_id, "start_date": start, "end_date": end}
    df = pd.read_sql(text(sql), store.engine, params=params)
    if df.empty:
        return df
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    # Stored as UTC but without tz info.
    df["asof_utc"] = pd.to_datetime(df["asof_utc"], utc=True)
    return df


def _load_station_truth(store: MySqlStore, station_id: str, start: date, end: date) -> pd.DataFrame:
    df = store.fetch_cli_truth_history(station_id, start, end)
    if df.empty:
        return df
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    return df


def _infer_available_date_range(
    store: MySqlStore,
    station_id: str,
) -> tuple[date | None, date | None]:
    sql = """
        SELECT MIN(target_date_local) AS min_d, MAX(target_date_local) AS max_d
        FROM live_features_daily
        WHERE station_id = :station_id
    """
    row = pd.read_sql(text(sql), store.engine, params={"station_id": station_id})
    if row.empty or row.iloc[0]["min_d"] is None:
        return None, None
    min_d = pd.to_datetime(row.iloc[0]["min_d"]).date()
    max_d = pd.to_datetime(row.iloc[0]["max_d"]).date()
    return min_d, max_d


def _load_train_stats(
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> TrainStats:
    qs = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0]
    mins: dict[str, float] = {}
    maxs: dict[str, float] = {}
    quantiles: dict[str, dict[str, float]] = {}
    for col in feature_cols:
        if col not in train_df.columns:
            continue
        series = pd.to_numeric(train_df[col], errors="coerce").dropna()
        mins[col] = float(series.min()) if not series.empty else float("nan")
        maxs[col] = float(series.max()) if not series.empty else float("nan")
        quantiles[col] = _quantiles(series, qs)
    return TrainStats(mins=mins, maxs=maxs, quantiles=quantiles)


def _pct_rank(series: pd.Series, value: float) -> float:
    """Return percentile rank in [0,1] using <= comparison; NaN if series empty."""
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty or pd.isna(value):
        return float("nan")
    return float((series <= value).mean())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fleece station DB history: feature distributions, OOD vs training, and model behavior."
    )
    parser.add_argument("--station", default="KMIA", help="Station ID (default: KMIA).")
    parser.add_argument("--start-date", default="2020-01-01", help="Start date (YYYY-MM-DD or YYYYMMDD).")
    parser.add_argument("--end-date", default=None, help="End date (YYYY-MM-DD or YYYYMMDD).")
    parser.add_argument("--window-days", type=int, default=60, help="Rolling window size for bias features.")
    parser.add_argument("--truth-lag-days", type=int, default=2, help="Truth lag days for bias features.")
    parser.add_argument("--output-dir", default=None, help="Output directory under artifacts/diagnostics.")
    args = parser.parse_args()

    station_id = args.station.strip().upper()
    requested_start = _parse_date(args.start_date)
    requested_end = _parse_date(args.end_date) if args.end_date else None

    store = MySqlStore(_mysql_config_from_env())

    avail_min, avail_max = _infer_available_date_range(store, station_id)
    if avail_min is None or avail_max is None:
        raise SystemExit(f"No live_features_daily history found for station={station_id}")

    end_date = min(requested_end, avail_max) if requested_end else avail_max
    # We can only compute feature-based diagnostics where we have live_features_daily rows.
    start_date = max(requested_start, avail_min)

    # Load full features/truth needed for rolling windows.
    history_margin = timedelta(days=args.window_days + args.truth_lag_days + 5)
    feature_load_start = max(avail_min, start_date - history_margin)
    truth_load_start = start_date - history_margin

    base_features = _load_station_features(store, station_id, feature_load_start, end_date)
    if base_features.empty:
        raise SystemExit(f"No live_features_daily rows found for station={station_id} in range")

    truth = _load_station_truth(store, station_id, truth_load_start, end_date)

    station_models = models_dir(station_id)
    feature_cols_path = station_models / "e92_feature_columns.json"
    mu_model_path = station_models / "e92_mu_model.joblib"
    sigma_model_path = station_models / "e92_sigma_model.joblib"
    training_meta_path = station_models / "e92_training_metadata.json"

    feature_cols = json.loads(feature_cols_path.read_text(encoding="utf-8"))
    mu_model = joblib.load(mu_model_path)
    sigma_model = joblib.load(sigma_model_path)

    training_meta = json.loads(training_meta_path.read_text(encoding="utf-8"))
    train_df = pd.read_parquet(training_meta["dataset_path"])
    train_df = train_df[train_df["station_id"].str.upper() == station_id]
    train_stats = _load_train_stats(train_df, feature_cols)
    train_start = _parse_date(str(training_meta.get("train_start")))
    train_end = _parse_date(str(training_meta.get("train_end")))

    # Restrict to requested analysis window (but keep earlier rows in base_features for bias windows).
    base_features_win = base_features[
        (base_features["target_date_local"] >= start_date) & (base_features["target_date_local"] <= end_date)
    ].copy()
    base_features_all = base_features.copy()

    # Sanity: expected asof check (should be 12Z D-1).
    expected_asof = base_features_win["target_date_local"].apply(asof_from_target_date)
    asof_mismatch = base_features_win[base_features_win["asof_utc"] != expected_asof]
    asof_mismatch_rows = [
        {
            "target_date_local": str(r["target_date_local"]),
            "asof_utc": str(pd.to_datetime(r["asof_utc"], utc=True)),
            "expected_asof_utc": str(asof_from_target_date(r["target_date_local"])),
        }
        for _, r in asof_mismatch.iterrows()
    ]

    required_base_cols = set(ENSEMBLE_COMPONENTS + ["gefsatmos_tmp_spread_f"])
    missing_cols = sorted([c for c in required_base_cols if c not in base_features_all.columns])
    if missing_cols:
        raise SystemExit(f"Missing required base columns in live_features_daily: {missing_cols}")

    # Compute ensmean for every row (used for summaries / OOD checks).
    base_features_all["ensmean"] = base_features_all.apply(
        lambda r: float(np.mean([r[c] for c in ENSEMBLE_COMPONENTS])), axis=1
    )
    base_features_win["ensmean"] = base_features_win.apply(
        lambda r: float(np.mean([r[c] for c in ENSEMBLE_COMPONENTS])), axis=1
    )

    # Base-feature OOD counts vs training (even for days we may not score).
    base_ood_counts: dict[str, int] = {}
    for col in sorted(required_base_cols.union({"ensmean"})):
        if col not in train_df.columns:
            # ensmean isn't stored in the training parquet for this run; we compute it below.
            if col != "ensmean":
                continue
        if col == "ensmean":
            train_series = train_df.apply(lambda r: float(np.mean([r[c] for c in ENSEMBLE_COMPONENTS])), axis=1)
            train_min = float(train_series.min())
            train_max = float(train_series.max())
            live_series = base_features_win["ensmean"]
        else:
            train_min = float(train_df[col].min())
            train_max = float(train_df[col].max())
            live_series = pd.to_numeric(base_features_win[col], errors="coerce")
        base_ood_counts[col] = int(((live_series < train_min) | (live_series > train_max)).sum())

    # Predict across full window where possible, and compute full-feature OOD rates.
    rows: list[dict[str, Any]] = []
    feature_errors: list[dict[str, Any]] = []
    scored = 0

    # Use only rows with complete base columns for feature construction.
    base_complete = base_features_all.dropna(subset=list(required_base_cols)).copy()
    truth_complete = truth.dropna(subset=["actual_tmax_f"]).copy()

    # Precompute for percentile ranks.
    train_feature_series: dict[str, pd.Series] = {}
    for col in feature_cols:
        if col in train_df.columns:
            train_feature_series[col] = pd.to_numeric(train_df[col], errors="coerce")

    # ensmean stats in training (computed).
    train_ensmean = train_df.apply(lambda r: float(np.mean([r[c] for c in ENSEMBLE_COMPONENTS])), axis=1)
    train_ensmean_min = float(train_ensmean.min())
    train_ensmean_max = float(train_ensmean.max())

    for _, day_row in base_features_win.iterrows():
        target = day_row["target_date_local"]
        # Only score if truth is available (historical), and we have base features.
        truth_row = truth_complete[truth_complete["target_date_local"] == target]
        actual = float(truth_row.iloc[0]["actual_tmax_f"]) if not truth_row.empty else None

        base_row = day_row.to_dict()
        ensmean = float(day_row["ensmean"])

        # Full feature vector + prediction uses 60d window + truth lag.
        try:
            feature_df = build_feature_vector(
                feature_cols,
                base_row,
                base_complete,
                truth_complete,
                target,
                truth_lag_days=args.truth_lag_days,
                window_days=args.window_days,
            )
            mu_hat = float(mu_model.predict(feature_df.to_numpy(dtype=float))[0])
            sigma_hat = float(max(0.5, sigma_model.predict(feature_df.to_numpy(dtype=float))[0]))
        except Exception as exc:  # noqa: BLE001 - diagnostics script
            feature_errors.append({"target_date_local": str(target), "error": str(exc)})
            continue

        feature_dict = feature_df.iloc[0].to_dict()

        # Feature OOD vs training distribution.
        ood_below = 0
        ood_above = 0
        ood_features: list[str] = []
        for col in feature_cols:
            val = float(feature_dict[col])
            train_min = train_stats.mins.get(col, float("nan"))
            train_max = train_stats.maxs.get(col, float("nan"))
            if not np.isfinite(train_min) or not np.isfinite(train_max):
                continue
            if val < train_min:
                ood_below += 1
                ood_features.append(f"{col}<min")
            elif val > train_max:
                ood_above += 1
                ood_features.append(f"{col}>max")

        abs_err = abs(mu_hat - actual) if actual is not None else None

        row_out: dict[str, Any] = {
            "station_id": station_id,
            "target_date_local": target,
            "asof_utc": day_row["asof_utc"],
            "actual_tmax_f": actual,
            "mu_hat_f": mu_hat,
            "sigma_hat_f": sigma_hat,
            "ensmean": ensmean,
            "mu_minus_ensmean": mu_hat - ensmean,
            "abs_error": abs_err,
            "error": (mu_hat - actual) if actual is not None else None,
            "gefs_spread_f": float(day_row.get("gefsatmos_tmp_spread_f")),
            "ood_feature_below_count": ood_below,
            "ood_feature_above_count": ood_above,
            "ood_feature_total": ood_below + ood_above,
            "ood_features": "|".join(ood_features),
            # Base features (for quick inspection).
            "nbm_tmax_f": float(day_row.get("nbm_tmax_f")),
            "hrrr_tmax_f": float(day_row.get("hrrr_tmax_f")),
            "rap_tmax_f": float(day_row.get("rap_tmax_f")),
            "gefsatmosmean_tmax_f": float(day_row.get("gefsatmosmean_tmax_f")),
            "gfs_n_x_max": float(day_row.get("gfs_n_x_max")),
            "nam_n_x_max": float(day_row.get("nam_n_x_max")),
        }

        # Percentile ranks for key diagnostics.
        row_out["train_pct_ensmean"] = _pct_rank(train_ensmean, ensmean)
        for col in [
            "nbm_tmax_f",
            "hrrr_tmax_f",
            "gefsatmosmean_tmax_f",
            "gfs_n_x_max",
            "nam_n_x_max",
            "gefsatmos_tmp_spread_f",
        ]:
            if col in train_df.columns:
                row_out[f"train_pct_{col}"] = _pct_rank(train_df[col], float(day_row.get(col)))

        rows.append(row_out)
        if actual is not None:
            scored += 1

    df_out = pd.DataFrame(rows)

    # Aggregate feature OOD counts across scored days.
    feature_ood_summary: list[dict[str, Any]] = []
    if not df_out.empty:
        # Expand "ood_features" for per-feature counts.
        exploded = (
            df_out[["target_date_local", "ood_features"]]
            .assign(ood_features=df_out["ood_features"].fillna("").astype(str))
        )
        tokens: list[str] = []
        for s in exploded["ood_features"].tolist():
            if not s:
                continue
            tokens.extend([t for t in s.split("|") if t])
        counts = pd.Series(tokens).value_counts()
        for key, count in counts.items():
            feature_ood_summary.append({"feature_flag": key, "count": int(count)})

    # Save outputs.
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = artifacts_root() / "diagnostics" / f"{station_id.lower()}_db_fleece_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV for manual inspection.
    if not df_out.empty:
        df_csv = df_out.copy()
        df_csv["target_date_local"] = df_csv["target_date_local"].astype(str)
        df_csv["asof_utc"] = pd.to_datetime(df_csv["asof_utc"], utc=True).astype(str)
        df_csv.to_csv(out_dir / "per_day.csv", index=False)

    (out_dir / "feature_ood_flags_counts.json").write_text(
        json.dumps(feature_ood_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # Summary.
    y_true = df_out["actual_tmax_f"].dropna().to_numpy(dtype=float) if not df_out.empty else np.array([])
    y_pred = df_out.loc[df_out["actual_tmax_f"].notna(), "mu_hat_f"].to_numpy(dtype=float) if not df_out.empty else np.array([])
    def _metrics_for(sub: pd.DataFrame) -> dict[str, float]:
        sub = sub[sub["actual_tmax_f"].notna()].copy()
        if sub.empty:
            return {"mae": float("nan"), "rmse": float("nan"), "bias": float("nan")}
        y_t = sub["actual_tmax_f"].to_numpy(dtype=float)
        y_p = sub["mu_hat_f"].to_numpy(dtype=float)
        return {"mae": _mae(y_t, y_p), "rmse": _rmse(y_t, y_p), "bias": _bias(y_t, y_p)}

    metrics_all = _metrics_for(df_out) if not df_out.empty else {"mae": float("nan"), "rmse": float("nan"), "bias": float("nan")}
    metrics_train = _metrics_for(
        df_out[
            (pd.to_datetime(df_out["target_date_local"]).dt.date >= train_start)
            & (pd.to_datetime(df_out["target_date_local"]).dt.date <= train_end)
        ]
    ) if not df_out.empty else {"mae": float("nan"), "rmse": float("nan"), "bias": float("nan")}
    metrics_post = _metrics_for(
        df_out[pd.to_datetime(df_out["target_date_local"]).dt.date > train_end]
    ) if not df_out.empty else {"mae": float("nan"), "rmse": float("nan"), "bias": float("nan")}

    # Last-120 window ending at the last day with truth available (for apples-to-apples).
    truth_max = None
    if not truth_complete.empty:
        truth_max = max(truth_complete["target_date_local"])
    metrics_last120 = {"mae": float("nan"), "rmse": float("nan"), "bias": float("nan")}
    if truth_max is not None:
        last120_start = truth_max - timedelta(days=119)
        metrics_last120 = _metrics_for(
            df_out[
                (pd.to_datetime(df_out["target_date_local"]).dt.date >= last120_start)
                & (pd.to_datetime(df_out["target_date_local"]).dt.date <= truth_max)
            ]
        )

    summary = {
        "station_id": station_id,
        "requested_start_date": str(requested_start),
        "requested_end_date": str(requested_end) if requested_end else None,
        "features_available_min_date": str(avail_min),
        "features_available_max_date": str(avail_max),
        "analysis_start_date": str(start_date),
        "analysis_end_date": str(end_date),
        "window_days": int(args.window_days),
        "truth_lag_days": int(args.truth_lag_days),
        "n_feature_days": int(len(base_features_win)),
        "n_scored_days": int(len(y_true)),
        "train_period": {"start": str(train_start), "end": str(train_end)},
        "metrics": {
            "all_scored": metrics_all,
            "train_period_scored": metrics_train,
            "post_train_scored": metrics_post,
            "last120_scored": metrics_last120,
            "last_truth_date": str(truth_max) if truth_max is not None else None,
        },
        "asof_mismatch_rows": asof_mismatch_rows,
        "train_feature_mins": train_stats.mins,
        "train_feature_maxs": train_stats.maxs,
        "train_ensmean_min": train_ensmean_min,
        "train_ensmean_max": train_ensmean_max,
        "base_feature_ood_counts_vs_training": base_ood_counts,
        "feature_vector_errors": feature_errors[:50],
        "feature_vector_error_count": len(feature_errors),
    }

    # Top outliers.
    if not df_out.empty and "abs_error" in df_out.columns:
        top_abs = (
            df_out[df_out["abs_error"].notna()]
            .sort_values("abs_error", ascending=False)
            .head(20)[
                [
                    "target_date_local",
                    "actual_tmax_f",
                    "mu_hat_f",
                    "ensmean",
                    "mu_minus_ensmean",
                    "abs_error",
                    "gefs_spread_f",
                    "ood_feature_total",
                    "ood_features",
                ]
            ]
            .to_dict(orient="records")
        )
        summary["top_abs_error"] = top_abs

    if not df_out.empty and "mu_minus_ensmean" in df_out.columns:
        top_mu_minus_ens = (
            df_out.sort_values("mu_minus_ensmean", ascending=False)
            .head(20)[
                [
                    "target_date_local",
                    "mu_hat_f",
                    "ensmean",
                    "mu_minus_ensmean",
                    "gefs_spread_f",
                    "ood_feature_total",
                    "ood_features",
                ]
            ]
            .to_dict(orient="records")
        )
        summary["top_mu_minus_ensmean"] = top_mu_minus_ens

    if not df_out.empty and "ood_feature_total" in df_out.columns:
        top_ood = (
            df_out.sort_values("ood_feature_total", ascending=False)
            .head(20)[
                [
                    "target_date_local",
                    "mu_hat_f",
                    "ensmean",
                    "gefs_spread_f",
                    "ood_feature_total",
                    "ood_features",
                ]
            ]
            .to_dict(orient="records")
        )
        summary["top_ood_days"] = top_ood

    def _jsonable(obj: Any) -> Any:
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_jsonable(v) for v in obj]
        return obj

    (out_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # Human-readable note.
    note_lines = [
        f"Station: {station_id}",
        f"Available live_features_daily: {avail_min} .. {avail_max}",
        f"Analysis window: {start_date} .. {end_date}",
        f"Scored days (truth available): {len(y_true)}",
        (
            "MAE(all)={:.3f} MAE(post-train)={:.3f} MAE(last120)={:.3f}".format(
                summary["metrics"]["all_scored"]["mae"],
                summary["metrics"]["post_train_scored"]["mae"],
                summary["metrics"]["last120_scored"]["mae"],
            )
        ),
        "",
        "Files:",
        "  - per_day.csv (per-day features/predictions/outliers)",
        "  - summary.json (rollups + top outliers)",
        "  - feature_ood_flags_counts.json (counts of feature<min/feature>max flags)",
    ]
    (out_dir / "README.txt").write_text("\n".join(note_lines) + "\n", encoding="utf-8")

    print(f"Wrote diagnostics to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
