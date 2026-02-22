#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import os
from pathlib import Path
from typing import Any
import sys


# Allow running this script directly without needing PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
from scipy.stats import kstest, norm
from sqlalchemy import text

from ml_live.calibration.emos_w45 import calibrate
from ml_live.db.mysql import MysqlConfig, MySqlStore
from ml_live.features.e92_features import build_feature_vector
from ml_live.runtime.clock import asof_from_target_date, parse_target_date
from ml_live.runtime.paths import artifacts_root, models_dir


UTC = timezone.utc


@dataclass(frozen=True)
class EvalConfig:
    station_id: str
    eval_start: date
    eval_end: date
    truth_lag_days: int = 2
    bias_window_days: int = 60
    emos_window_days: int = 45
    sigma_floor: float = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate calibration for the KMIA live pipeline using DB core tables."
    )
    parser.add_argument("--station", default="KMIA", help="Station ID (default: KMIA).")
    parser.add_argument(
        "--start-date",
        default="2025-01-01",
        help="Target date local start (YYYYMMDD or YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        default="2026-01-25",
        help="Target date local end (YYYYMMDD or YYYY-MM-DD).",
    )
    parser.add_argument(
        "--sigma-floor",
        type=float,
        default=0.5,
        help="Minimum sigma for sigma_hat and EMOS (default: 0.5).",
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Comma-separated integer thresholds for Brier scores (default: auto from truth range).",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output folder. Default: artifacts/live_pipeline_calibration_eval/<run_id>",
    )
    return parser.parse_args()


def _date_range(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError(f"end_date < start_date: {start}..{end}")
    days: list[date] = []
    cursor = start
    while cursor <= end:
        days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _parse_thresholds(value: str | None) -> list[int]:
    if not value:
        return []
    out: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def _choose_thresholds_from_truth(y: pd.Series) -> list[int]:
    y_vals = pd.to_numeric(y, errors="coerce").dropna().to_numpy(dtype=float)
    if y_vals.size == 0:
        return list(range(80, 101))
    p05 = float(np.nanpercentile(y_vals, 5))
    p95 = float(np.nanpercentile(y_vals, 95))
    if not np.isfinite(p05) or not np.isfinite(p95):
        return list(range(80, 101))
    low = int(np.floor(p05 - 5))
    high = int(np.ceil(p95 + 5))
    if high <= low:
        return list(range(80, 101))
    # Keep a reasonable range for Miami; avoid extreme tails from bad data.
    low = max(low, 40)
    high = min(high, 110)
    return list(range(low, high + 1))


def _compute_crps(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = (y - mu) / sigma
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    return sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / np.sqrt(np.pi))


def _compute_nll(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 0.5 * np.log(2 * np.pi * sigma**2) + ((y - mu) ** 2) / (2 * sigma**2)


def _load_truth_cli_with_station_fallback(
    store: MySqlStore,
    station_id: str,
    start_date: date,
    end_date: date,
) -> tuple[pd.DataFrame, list[date]]:
    cli = store.fetch_cli_truth_history(station_id, start_date, end_date).copy()
    cli["target_date_local"] = pd.to_datetime(cli["target_date_local"]).dt.date
    if "actual_tmax_f" in cli.columns:
        cli["actual_tmax_f"] = pd.to_numeric(cli["actual_tmax_f"], errors="coerce")

    required = set(_date_range(start_date, end_date))
    present = set(cli.loc[cli["actual_tmax_f"].notna(), "target_date_local"])
    missing = sorted(required - present)
    if not missing:
        return cli, []

    # Fill gaps from station_daily_truth for dates that exist there (it ends at 2025-12-31).
    sql = """
        SELECT station_id, date_local AS target_date_local, tmax_f AS actual_tmax_f
        FROM station_daily_truth
        WHERE station_id = :station_id
          AND date_local BETWEEN :start_date AND :end_date
    """
    params = {"station_id": station_id, "start_date": start_date, "end_date": end_date}
    station_truth = pd.read_sql(text(sql), store.engine, params=params)
    if not station_truth.empty:
        station_truth["target_date_local"] = pd.to_datetime(station_truth["target_date_local"]).dt.date
        station_truth["actual_tmax_f"] = pd.to_numeric(station_truth["actual_tmax_f"], errors="coerce")
        station_truth = station_truth[station_truth["actual_tmax_f"].notna()]

    if station_truth.empty:
        return cli, missing

    fill_map = {
        row["target_date_local"]: float(row["actual_tmax_f"])
        for _, row in station_truth.iterrows()
    }
    filled: list[dict[str, Any]] = []
    for day in missing:
        value = fill_map.get(day)
        if value is None:
            continue
        filled.append(
            {
                "station_id": station_id,
                "target_date_local": day,
                "actual_tmax_f": float(value),
            }
        )
    if not filled:
        return cli, missing

    combined = pd.concat([cli, pd.DataFrame(filled)], ignore_index=True)
    combined = combined.drop_duplicates(subset=["station_id", "target_date_local"], keep="last")
    combined = combined.sort_values("target_date_local")
    filled_days = sorted({row["target_date_local"] for row in filled})
    return combined, filled_days


def _load_core_features(
    store: MySqlStore,
    station_id: str,
    start_date: date,
    end_date: date,
    asof_utc_max: datetime,
) -> tuple[pd.DataFrame, list[date]]:
    grib = store.fetch_gribstream_daily_feature_history(station_id, start_date, end_date, asof_utc_max)
    mos = store.fetch_mos_n_x_history(station_id, start_date, end_date, asof_utc_max)
    if mos.empty:
        raise ValueError("MOS history is empty; cannot evaluate pipeline")
    mos = mos.copy()
    mos["target_date_local"] = pd.to_datetime(mos["target_date_local"]).dt.date
    mos["asof_utc"] = pd.to_datetime(mos["asof_utc"], utc=True)

    if grib.empty:
        raise ValueError("GribStream daily history is empty; cannot evaluate pipeline")

    features = grib.merge(mos, on=["station_id", "target_date_local", "asof_utc"], how="left")

    # MOS core history can be incomplete for isolated dates. Fall back to live_features_daily if present.
    required_mos_cols = ["gfs_n_x_max", "nam_n_x_max"]
    if any(col not in features.columns for col in required_mos_cols):
        raise ValueError(f"Missing MOS columns in merged features: {required_mos_cols}")
    missing_mos = features[required_mos_cols].isna().any(axis=1)
    fallback_days: list[date] = []
    if missing_mos.any():
        fallback_days = sorted(set(features.loc[missing_mos, "target_date_local"]))
        sql = """
            SELECT station_id, target_date_local, asof_utc, gfs_n_x_max, nam_n_x_max
            FROM live_features_daily
            WHERE station_id = :station_id
              AND target_date_local BETWEEN :start_date AND :end_date
        """
        params = {"station_id": station_id, "start_date": start_date, "end_date": end_date}
        live = pd.read_sql(text(sql), store.engine, params=params)
        if not live.empty:
            live["target_date_local"] = pd.to_datetime(live["target_date_local"]).dt.date
            live["asof_utc"] = pd.to_datetime(live["asof_utc"], utc=True)
            live = live.drop_duplicates(subset=["station_id", "target_date_local", "asof_utc"])
            features = features.merge(
                live,
                on=["station_id", "target_date_local", "asof_utc"],
                how="left",
                suffixes=("", "_live"),
            )
            for col in required_mos_cols:
                live_col = f"{col}_live"
                if live_col not in features.columns:
                    continue
                features[col] = features[col].where(features[col].notna(), features[live_col])
            drop_cols = [f"{col}_live" for col in required_mos_cols if f"{col}_live" in features.columns]
            if drop_cols:
                features = features.drop(columns=drop_cols)
    return features, fallback_days


def _build_predictions(
    cfg: EvalConfig,
    feature_cols: list[str],
    mu_model,
    sigma_model,
    features: pd.DataFrame,
    truth: pd.DataFrame,
    pred_start: date,
) -> pd.DataFrame:
    features = features.copy()
    truth = truth.copy()
    features["target_date_local"] = pd.to_datetime(features["target_date_local"]).dt.date
    truth["target_date_local"] = pd.to_datetime(truth["target_date_local"]).dt.date

    idx = features.set_index(["target_date_local", "asof_utc"])
    feature_rows: dict[tuple[date, pd.Timestamp], dict[str, Any]] = idx.to_dict(orient="index")

    preds: list[dict[str, Any]] = []
    for target_date in _date_range(pred_start, cfg.eval_end):
        expected_asof = pd.Timestamp(asof_from_target_date(target_date))
        key = (target_date, expected_asof)
        base_row = feature_rows.get(key)
        if base_row is None:
            raise ValueError(f"Missing base features for target_date_local={target_date} asof={expected_asof}")

        feature_df = build_feature_vector(
            feature_cols,
            base_row,
            features,
            truth,
            target_date,
            truth_lag_days=cfg.truth_lag_days,
            window_days=cfg.bias_window_days,
            allow_partial_history=False,
        )
        mu_hat_f = float(mu_model.predict(feature_df.to_numpy(dtype=float))[0])
        sigma_hat_f = float(max(cfg.sigma_floor, sigma_model.predict(feature_df.to_numpy(dtype=float))[0]))
        preds.append(
            {
                "station_id": cfg.station_id,
                "target_date_local": target_date,
                "asof_utc": expected_asof.to_pydatetime(),
                "mu_hat_f": mu_hat_f,
                "sigma_hat_f": sigma_hat_f,
            }
        )

    df = pd.DataFrame(preds)
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    df["asof_utc"] = pd.to_datetime(df["asof_utc"], utc=True)
    return df


def _score_distribution(
    daily: pd.DataFrame,
    thresholds: list[int],
    mu_col: str,
    sigma_col: str,
    prefix: str,
) -> tuple[pd.DataFrame, dict]:
    scored = daily.copy()
    y = scored["actual_tmax_f"].to_numpy(dtype=float)
    mu = scored[mu_col].to_numpy(dtype=float)
    sigma = scored[sigma_col].to_numpy(dtype=float)
    sigma = np.maximum(sigma, 1e-9)

    z = (y - mu) / sigma
    pit = norm.cdf(z)
    crps = _compute_crps(mu, sigma, y)
    nll = _compute_nll(mu, sigma, y)

    scored[f"{prefix}_z"] = z
    scored[f"{prefix}_pit"] = pit
    scored[f"{prefix}_crps"] = crps
    scored[f"{prefix}_nll"] = nll

    # Coverage.
    coverage: dict[str, float] = {}
    for level in [0.5, 0.8, 0.9, 0.95]:
        z_alpha = float(norm.ppf((1 + level) / 2))
        lower = mu - z_alpha * sigma
        upper = mu + z_alpha * sigma
        coverage[str(int(level * 100))] = float(np.mean((y >= lower) & (y <= upper)))

    pits = pd.Series(pit).dropna()
    if len(pits) >= 20:
        ks = kstest(pits.to_numpy(dtype=float), "uniform")
        pit_ks = {"statistic": float(ks.statistic), "pvalue": float(ks.pvalue)}
    else:
        pit_ks = {"statistic": None, "pvalue": None}

    # Brier per threshold (event Tmax >= k).
    brier: dict[str, float] = {}
    for k in thresholds:
        p = 1.0 - norm.cdf((k - mu) / sigma)
        o = (y >= float(k)).astype(float)
        brier[str(k)] = float(np.mean((p - o) ** 2))

    metrics = {
        "mean_crps": float(np.nanmean(crps)),
        "mean_nll": float(np.nanmean(nll)),
        "sharpness_mean_sigma": float(np.nanmean(sigma)),
        "sharpness_median_sigma": float(np.nanmedian(sigma)),
        "sharpness_p90_sigma": float(np.nanpercentile(sigma, 90)),
        "coverage": coverage,
        "coverage_error_90": float(coverage.get("90", float("nan")) - 0.9),
        "pit_ks": pit_ks,
        "brier_per_threshold": brier,
        "brier_mean": float(np.mean(list(brier.values()))) if brier else None,
    }
    return scored, metrics


def _mu_metrics(daily: pd.DataFrame, mu_col: str) -> dict:
    y = daily["actual_tmax_f"].to_numpy(dtype=float)
    mu = daily[mu_col].to_numpy(dtype=float)
    residual = mu - y
    return {
        "mae": float(np.nanmean(np.abs(residual))),
        "rmse": float(np.sqrt(np.nanmean(np.square(residual)))),
        "bias": float(np.nanmean(residual)),
    }


def main() -> int:
    args = parse_args()
    station_id = (args.station or "KMIA").strip().upper()
    eval_start = parse_target_date(args.start_date)
    eval_end = parse_target_date(args.end_date)

    cfg = EvalConfig(
        station_id=station_id,
        eval_start=eval_start,
        eval_end=eval_end,
        sigma_floor=float(args.sigma_floor),
    )

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_root = (
        Path(args.output_root)
        if args.output_root
        else (artifacts_root() / "live_pipeline_calibration_eval" / run_id)
    )
    out_root.mkdir(parents=True, exist_ok=True)

    # Construct MySQL config from environment, mirroring tools/live/run_kmia_live.py defaults.
    mysql_cfg = MysqlConfig(
        host=str(os.getenv("MYSQL_HOST") or "localhost"),
        port=int(os.getenv("MYSQL_PORT") or 3306),
        database=str(os.getenv("MYSQL_DB") or "weather_predictionmarkets"),
        user=str(os.getenv("MYSQL_USER") or "root"),
        password=str(os.getenv("MYSQL_PASSWORD") or ""),
    )
    store = MySqlStore(mysql_cfg)

    station_models_dir = models_dir(station_id)
    feature_cols_path = station_models_dir / "e92_feature_columns.json"
    mu_model_path = station_models_dir / "e92_mu_model.joblib"
    sigma_model_path = station_models_dir / "e92_sigma_model.joblib"
    if not feature_cols_path.exists():
        raise FileNotFoundError(f"Missing feature columns file: {feature_cols_path}")
    feature_cols = json.loads(feature_cols_path.read_text(encoding="utf-8"))
    mu_model = joblib.load(mu_model_path)
    sigma_model = joblib.load(sigma_model_path)

    # Prediction start must include enough history to fit EMOS for eval_start.
    pred_start = cfg.eval_start - timedelta(days=(cfg.emos_window_days + cfg.truth_lag_days - 1))
    hist_start = pred_start - timedelta(days=(cfg.bias_window_days + cfg.truth_lag_days - 1))

    asof_utc_max = asof_from_target_date(cfg.eval_end)
    truth_all, truth_filled_days = _load_truth_cli_with_station_fallback(
        store, station_id, hist_start, cfg.eval_end
    )
    features_all, mos_fallback_days = _load_core_features(
        store, station_id, hist_start, cfg.eval_end, asof_utc_max
    )

    # Build predictions (mu_hat_f, sigma_hat_f) for pred_start..eval_end.
    preds_all = _build_predictions(
        cfg=cfg,
        feature_cols=feature_cols,
        mu_model=mu_model,
        sigma_model=sigma_model,
        features=features_all,
        truth=truth_all,
        pred_start=pred_start,
    )

    # Score only eval_start..eval_end.
    truth_eval = truth_all[
        (truth_all["target_date_local"] >= cfg.eval_start)
        & (truth_all["target_date_local"] <= cfg.eval_end)
    ].copy()
    preds_eval = preds_all[
        (preds_all["target_date_local"] >= cfg.eval_start)
        & (preds_all["target_date_local"] <= cfg.eval_end)
    ].copy()

    daily = preds_eval.merge(
        truth_eval[["station_id", "target_date_local", "actual_tmax_f"]],
        on=["station_id", "target_date_local"],
        how="inner",
    )
    expected_days = len(_date_range(cfg.eval_start, cfg.eval_end))
    if len(daily) != expected_days:
        have = set(daily["target_date_local"])
        missing_days = [
            d.isoformat()
            for d in _date_range(cfg.eval_start, cfg.eval_end)
            if d not in have
        ]
        raise ValueError(
            f"Incomplete eval set rows={len(daily)} expected={expected_days}. "
            f"Missing days={missing_days[:10]}{'...' if len(missing_days) > 10 else ''}"
        )

    # Compute EMOS sigma for each eval day (online, using the prior 45 days ending at D-2).
    pred_map = preds_all.set_index("target_date_local")
    truth_map = truth_all.set_index("target_date_local")

    emos_rows: list[dict[str, Any]] = []
    for target_date in daily["target_date_local"].tolist():
        emos_end = target_date - timedelta(days=cfg.truth_lag_days)
        emos_start = emos_end - timedelta(days=(cfg.emos_window_days - 1))
        hist_dates = _date_range(emos_start, emos_end)
        hist_pred = pred_map.loc[hist_dates][["mu_hat_f", "sigma_hat_f"]].reset_index()
        hist_truth = truth_map.loc[hist_dates][["actual_tmax_f"]].reset_index()
        emos_hist = hist_pred.merge(hist_truth, on="target_date_local", how="inner")
        emos_hist = emos_hist.dropna(subset=["mu_hat_f", "sigma_hat_f", "actual_tmax_f"])
        if len(emos_hist) < cfg.emos_window_days:
            raise ValueError(
                f"Insufficient EMOS history rows={len(emos_hist)} need={cfg.emos_window_days} "
                f"for target_date_local={target_date}"
            )
        target_sigma_hat = float(pred_map.loc[target_date]["sigma_hat_f"])
        emos_result = calibrate(emos_hist, target_sigma_hat, sigma_floor=cfg.sigma_floor)
        emos_rows.append(
            {
                "target_date_local": target_date,
                "sigma_emos_f": float(emos_result.sigma_emos),
                "emos_c": float(emos_result.c),
                "emos_d": float(emos_result.d),
                "rolling_bias_45": float(emos_result.rolling_bias),
                "rolling_rmse_45": float(emos_result.rolling_rmse),
            }
        )

    emos_df = pd.DataFrame(emos_rows)
    daily = daily.merge(emos_df, on="target_date_local", how="left")
    daily["final_sigma_f"] = daily["sigma_emos_f"].where(daily["sigma_emos_f"].notna(), daily["sigma_hat_f"])
    daily["residual_f"] = daily["actual_tmax_f"] - daily["mu_hat_f"]

    thresholds = _parse_thresholds(args.thresholds)
    if not thresholds:
        thresholds = _choose_thresholds_from_truth(daily["actual_tmax_f"])

    scored, metrics_hat = _score_distribution(
        daily, thresholds, mu_col="mu_hat_f", sigma_col="sigma_hat_f", prefix="hat"
    )
    scored, metrics_final = _score_distribution(
        scored, thresholds, mu_col="mu_hat_f", sigma_col="final_sigma_f", prefix="final"
    )

    mu_metrics = _mu_metrics(scored, mu_col="mu_hat_f")

    # Write outputs.
    daily_csv = out_root / "daily_scores.csv"
    scored_out = scored.copy()
    scored_out["target_date_local"] = scored_out["target_date_local"].astype(str)
    scored_out["asof_utc"] = pd.to_datetime(scored_out["asof_utc"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    scored_out.to_csv(daily_csv, index=False)

    brier_path = out_root / "brier_by_threshold.csv"
    brier_rows = []
    for k in thresholds:
        brier_rows.append(
            {
                "threshold": int(k),
                "brier_hat": metrics_hat["brier_per_threshold"].get(str(k)),
                "brier_final": metrics_final["brier_per_threshold"].get(str(k)),
            }
        )
    pd.DataFrame(brier_rows).to_csv(brier_path, index=False)

    report = {
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "station_id": station_id,
        "eval_start": cfg.eval_start.isoformat(),
        "eval_end": cfg.eval_end.isoformat(),
        "asof_policy": "asof_utc = target_date_local - 1 day @ 12:00Z",
        "config": {
            "truth_lag_days": cfg.truth_lag_days,
            "bias_window_days": cfg.bias_window_days,
            "emos_window_days": cfg.emos_window_days,
            "sigma_floor": cfg.sigma_floor,
        },
        "models": {
            "feature_columns_path": str(feature_cols_path),
            "mu_model_path": str(mu_model_path),
            "sigma_model_path": str(sigma_model_path),
        },
        "truth": {
            "primary": "cli_daily (IEM CLI)",
            "fallback": "station_daily_truth for missing CLI dates",
            "fallback_days": [d.isoformat() for d in truth_filled_days],
        },
        "features": {
            "mos_fallback": "live_features_daily for missing MOS core values",
            "mos_fallback_days": [d.isoformat() for d in mos_fallback_days],
        },
        "counts": {
            "n_days": int(len(scored)),
            "n_thresholds": int(len(thresholds)),
        },
        "mu_metrics": mu_metrics,
        "dist_metrics_sigma_hat": metrics_hat,
        "dist_metrics_final_sigma": metrics_final,
        "files": {
            "daily_scores_csv": str(daily_csv),
            "brier_by_threshold_csv": str(brier_path),
        },
    }

    report_path = out_root / "calibration_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    # Single-file bundle for easy sharing / handoff.
    bundle_path = out_root / "calibration_results_full.json"
    daily_json = scored_out.replace({np.nan: None}).to_dict(orient="records")
    brier_json = pd.DataFrame(brier_rows).replace({np.nan: None}).to_dict(orient="records")
    bundle = {
        "schema_version": 1,
        "bundle_generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "report": report,
        "daily_scores": daily_json,
        "brier_by_threshold": brier_json,
    }
    bundle_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    print(str(report_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
