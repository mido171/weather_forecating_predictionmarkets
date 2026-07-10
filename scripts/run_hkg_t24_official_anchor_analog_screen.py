from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hkg_t24_beastmode_signal_discovery import (  # noqa: E402
    CONFIRMATION_START,
    RESEARCH_ROOT,
    build_analysis_frame,
    load_champion_predictions,
    load_features,
    markdown_table,
    require_no_confirmation_dates,
    score_prediction_frame,
    write_csv,
    write_json,
    write_text,
)
from scripts.run_hkg_t24_forecast_anchor_forensics import load_official_forecasts  # noqa: E402

MIN_HISTORY = 180
MIN_ANALOGS = 20

FEATURE_SETS: dict[str, list[str]] = {
    "surface_moisture_rise": [
        "isd_air_temp_mean_c_change_1d",
        "isd_morning_to_midday_temp_rise_c",
        "isd_temp_dewpoint_spread_mean_c",
        "isd_dewpoint_midday_minus_temp_c",
    ],
    "upper_surface_ceiling": [
        "ua_layer_1000_925_ceiling_minus_isd_temp_c",
        "ua_layer_925_850_ceiling_minus_isd_temp_c",
        "ua_warm_layer_depth_count",
        "igra_dd_925hpa_c",
    ],
    "upper_thermal_moisture": [
        "ua_theta_e_1000_850_mean_k",
        "ua_mse_1000_850_mean_kj_kg",
        "ua_theta_850hpa_k",
        "ua_dewpoint_1000hpa_c",
    ],
    "target_volatility_memory": [
        "target_roll14_std_lag7_c",
        "target_volatility_forecastability_score_lag7",
        "target_abs_change_7_14_c",
        "spectral_abs_change_energy_30_lag7_c",
    ],
    "station_dewpoint_network": [
        "isd_station_dew_point_c_595010_99999",
        "isd_station_dew_point_c_450070_99999",
        "isd_station_dew_point_c_450110_99999",
        "isd_dew_point_mean_c_change_1d",
    ],
}


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def score_row(
    frame: pd.DataFrame,
    *,
    prediction_col: str,
    candidate_id: str,
    feature_set: str,
    detail: str,
) -> dict[str, object]:
    scored = frame.dropna(subset=[prediction_col, "target_tmax_c", "forecast_max_c"]).copy()
    if scored.empty:
        return {
            "candidate_id": candidate_id,
            "feature_set": feature_set,
            "detail": detail,
            "n": 0,
            "first_date": "",
            "last_date": "",
            "mae": math.nan,
            "rmse": math.nan,
            "bias": math.nan,
            "official_same_rows_mae": math.nan,
            "delta_vs_official_same_rows": math.nan,
        }
    candidate = score_prediction_frame(scored, prediction_col)
    official = score_prediction_frame(scored, "forecast_max_c")
    return {
        "candidate_id": candidate_id,
        "feature_set": feature_set,
        "detail": detail,
        **candidate,
        "official_same_rows_mae": official["mae"],
        "official_same_rows_rmse": official["rmse"],
        "delta_vs_official_same_rows": float(candidate["mae"] - official["mae"]),
    }


def weighted_mean(values: np.ndarray, distances: np.ndarray, *, weighted: bool) -> float:
    if not weighted:
        return float(np.mean(values))
    weights = 1.0 / np.maximum(distances, 1e-6)
    return float(np.average(values, weights=weights))


def analog_correction_prediction(
    frame: pd.DataFrame,
    *,
    features: list[str],
    k: int,
    pool_mode: str,
    season_conditioned: bool,
    weighted: bool,
    min_history: int = MIN_HISTORY,
    min_analogs: int = MIN_ANALOGS,
) -> pd.DataFrame:
    ordered = frame.sort_values("target_date").reset_index(drop=True).copy()
    feature_matrix = ordered[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    residual = (
        pd.to_numeric(ordered["target_tmax_c"], errors="coerce")
        - pd.to_numeric(ordered["forecast_max_c"], errors="coerce")
    ).to_numpy(dtype=float)
    official = pd.to_numeric(ordered["forecast_max_c"], errors="coerce").to_numpy(dtype=float)
    sources = ordered["forecast_source_family"].astype(str).to_numpy()
    seasons = ordered["season"].astype(str).to_numpy()
    predictions: list[float] = []
    corrections: list[float] = []
    rows_used: list[int] = []
    mean_distances: list[float] = []

    for index in range(len(ordered)):
        current = feature_matrix[index]
        if not np.isfinite(current).all():
            predictions.append(math.nan)
            corrections.append(math.nan)
            rows_used.append(0)
            mean_distances.append(math.nan)
            continue

        prior_features = feature_matrix[:index]
        prior_residual = residual[:index]
        prior_sources = sources[:index]
        prior_seasons = seasons[:index]
        valid = np.isfinite(prior_residual) & np.isfinite(prior_features).all(axis=1)
        if pool_mode == "same_source":
            valid &= prior_sources == sources[index]
        prior_features = prior_features[valid]
        prior_residual = prior_residual[valid]
        prior_seasons = prior_seasons[valid]
        if len(prior_features) < min_history:
            predictions.append(math.nan)
            corrections.append(math.nan)
            rows_used.append(int(len(prior_features)))
            mean_distances.append(math.nan)
            continue

        if season_conditioned:
            season_mask = prior_seasons == seasons[index]
            if int(season_mask.sum()) >= min_history:
                prior_features = prior_features[season_mask]
                prior_residual = prior_residual[season_mask]

        med = np.nanmedian(prior_features, axis=0)
        scale = np.nanmedian(np.abs(prior_features - med), axis=0)
        scale = np.where(scale < 1e-6, np.nanstd(prior_features, axis=0), scale)
        scale = np.where(scale < 1e-6, 1.0, scale)
        z_prior = (prior_features - med) / scale
        z_current = (current - med) / scale
        distances = np.sqrt(np.mean(np.square(z_prior - z_current), axis=1))
        order = np.argsort(distances)
        take = order[: min(k, len(order))]
        if len(take) < min_analogs:
            predictions.append(math.nan)
            corrections.append(math.nan)
            rows_used.append(int(len(take)))
            mean_distances.append(math.nan)
            continue
        correction = weighted_mean(prior_residual[take], distances[take], weighted=weighted)
        corrections.append(correction)
        predictions.append(float(official[index] + correction))
        rows_used.append(int(len(take)))
        mean_distances.append(float(np.mean(distances[take])))

    out = ordered.copy()
    out["candidate_prediction_c"] = predictions
    out["analog_correction_c"] = corrections
    out["analog_rows_used"] = rows_used
    out["analog_mean_distance"] = mean_distances
    return out


def build_official_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    features = load_features()
    champion = load_champion_predictions()
    frame = build_analysis_frame(features, champion)
    require_no_confirmation_dates(frame["target_date"], context="official analog feature frame")
    official = load_official_forecasts(frame)
    require_no_confirmation_dates(official["target_date"], context="official analog official rows")
    return frame, official


def run_screen(frame: pd.DataFrame, official: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for feature_set, requested_features in FEATURE_SETS.items():
        features = [feature for feature in requested_features if feature in frame.columns]
        if len(features) < 2:
            continue
        enriched = official.merge(frame[["target_date", *features]], on="target_date", how="left")
        for k in (50,):
            for pool_mode in ("same_source", "pooled_prior"):
                for season_conditioned in (False, True):
                    for weighted in (False, True):
                        candidate_id = (
                            f"{feature_set}_k{k}_{pool_mode}_season{int(season_conditioned)}"
                            f"_weighted{int(weighted)}"
                        )
                        pred = analog_correction_prediction(
                            enriched,
                            features=features,
                            k=k,
                            pool_mode=pool_mode,
                            season_conditioned=season_conditioned,
                            weighted=weighted,
                        )
                        pred["candidate_id"] = candidate_id
                        pred["feature_set"] = feature_set
                        pred["features"] = ",".join(features)
                        pred["k"] = k
                        pred["pool_mode"] = pool_mode
                        pred["season_conditioned"] = season_conditioned
                        pred["distance_weighted"] = weighted
                        rows.append(
                            score_row(
                                pred,
                                prediction_col="candidate_prediction_c",
                                candidate_id=candidate_id,
                                feature_set=feature_set,
                                detail=(
                                    f"features={','.join(features)}; k={k}; pool={pool_mode}; "
                                    f"season={season_conditioned}; weighted={weighted}"
                                ),
                            )
                        )
                        predictions.append(pred)
    scoreboard = pd.DataFrame(rows).sort_values(
        ["delta_vs_official_same_rows", "mae"],
        na_position="last",
    )
    all_predictions = pd.concat(predictions, ignore_index=True, sort=False) if predictions else pd.DataFrame()
    return scoreboard.reset_index(drop=True), all_predictions


def write_outputs(scoreboard: pd.DataFrame, predictions: pd.DataFrame, official: pd.DataFrame) -> None:
    folder = RESEARCH_ROOT / "0017_past_only_official_residual_analog_screen"
    artifacts = folder / "artifacts"
    write_csv(artifacts / "scoreboard.csv", scoreboard)
    write_csv(artifacts / "all_candidate_predictions.csv", predictions)

    source_coverage = (
        official.groupby("forecast_source_family", observed=True)
        .agg(
            rows=("target_date", "count"),
            first_date=("target_date", "min"),
            last_date=("target_date", "max"),
            raw_mae=("official_abs_error_c", "mean"),
            raw_bias=("official_error_c", "mean"),
        )
        .reset_index()
    )
    source_coverage["first_date"] = source_coverage["first_date"].dt.strftime("%Y-%m-%d")
    source_coverage["last_date"] = source_coverage["last_date"].dt.strftime("%Y-%m-%d")
    write_csv(artifacts / "source_coverage.csv", source_coverage)

    best = scoreboard.head(1).to_dict("records")
    best_text = "No scoreable analog candidate was produced."
    if best:
        row = best[0]
        best_text = (
            f"Best analog candidate: `{row['candidate_id']}` with MAE `{row['mae']:.4f}` "
            f"versus same-row official MAE `{row['official_same_rows_mae']:.4f}` "
            f"(delta `{row['delta_vs_official_same_rows']:.4f}`) over `{int(row['n'])}` rows."
        )

    text = f"""# Past-Only Official Residual Analog Screen

Generated: `{now_utc()}`

## What Was Tested

This insight tests whether nearest historical analogs can correct the official HKO forecast residual. For each target date, the feature scaler, nearest-neighbor pool, and residual correction are built only from earlier official-forecast rows.

Feature-set candidates:

{markdown_table(pd.DataFrame([{'feature_set': key, 'features': ', '.join(value)} for key, value in FEATURE_SETS.items()]), max_rows=20)}

## Leakage Contract

- Official forecasts are selected no later than `T-1 15:00 HKT`.
- Target labels from `{CONFIRMATION_START.date()}` onward are blocked.
- Analog scalers use only prior rows for the scored target date.
- Analog residuals use only prior target outcomes.
- This is a screen, not a promoted model.

## Source Coverage

{markdown_table(source_coverage, max_rows=10)}

## Main Result

{best_text}

## Top Analog Candidates

{markdown_table(scoreboard.head(30), max_rows=30)}

## Interpretation

Analog correction is a stricter test than a static correlation table. If it produces only small gains, then the available partial official forecast archive is not yet enough to learn a strong correction rule. The next meaningful unlock remains continuous 2000-2023 forecast vintages and then a fold-local regime/expert blend.
"""
    write_text(folder / "README.md", text)


def update_master_index(manifest: dict[str, object]) -> None:
    index_path = RESEARCH_ROOT / "README.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# HKG Tmax Data Analysis\n"
    marker = "\n## Official Residual Analog Screen\n"
    base = existing.split(marker)[0].rstrip()
    text = f"""{base}
{marker}
Generated: `{manifest['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_official_anchor_analog_screen.py`:

- `0017_past_only_official_residual_analog_screen`: nearest-historical-analog official residual corrections using prior rows only.

| Metric | Value |
|---|---:|
| Official rows | {manifest['official_rows']} |
| Candidate rows scored | {manifest['candidate_rows']} |
| Prediction rows written | {manifest['prediction_rows']} |
| Best candidate MAE | {manifest['best_mae']} |
| Best delta vs same-row official | {manifest['best_delta_vs_official']} |

Leakage contract: analog distances, scaling, neighbor selection, and residual corrections use only rows earlier than the scored target date.
"""
    write_text(index_path, text)


def run() -> dict[str, object]:
    frame, official = build_official_frame()
    scoreboard, predictions = run_screen(frame, official)
    write_outputs(scoreboard, predictions, official)
    best = scoreboard.iloc[0] if not scoreboard.empty else None
    manifest = {
        "generated_at_utc": now_utc(),
        "confirmation_start": str(CONFIRMATION_START.date()),
        "official_rows": int(len(official)),
        "candidate_rows": int(len(scoreboard)),
        "prediction_rows": int(len(predictions)),
        "first_target_date": str(official["target_date"].min().date()),
        "last_target_date": str(official["target_date"].max().date()),
        "best_candidate_id": "" if best is None else str(best["candidate_id"]),
        "best_mae": None if best is None else float(best["mae"]),
        "best_delta_vs_official": None if best is None else float(best["delta_vs_official_same_rows"]),
        "folder": "0017_past_only_official_residual_analog_screen",
    }
    write_json(RESEARCH_ROOT / "official_residual_analog_screen_manifest.json", manifest)
    update_master_index(manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run past-only official residual analog screen.").parse_args()


def main() -> None:
    parse_args()
    manifest = run()
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
