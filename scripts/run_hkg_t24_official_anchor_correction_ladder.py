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

FEATURE_CONDITIONING_PATH = (
    RESEARCH_ROOT
    / "0013_official_forecast_feature_conditioning"
    / "artifacts"
    / "feature_corr.csv"
)
MAX_FEATURES = 20
MIN_HISTORY = 120
MIN_GROUP = 40


def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def prediction_score_row(
    frame: pd.DataFrame,
    *,
    prediction_col: str,
    candidate_id: str,
    mechanism: str,
    detail: str,
) -> dict[str, object]:
    scored = frame.dropna(subset=[prediction_col, "target_tmax_c", "target_date"]).copy()
    if scored.empty:
        return {
            "candidate_id": candidate_id,
            "mechanism": mechanism,
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
        "mechanism": mechanism,
        "detail": detail,
        **candidate,
        "official_same_rows_mae": official["mae"],
        "official_same_rows_rmse": official["rmse"],
        "delta_vs_official_same_rows": float(candidate["mae"] - official["mae"]),
    }


def mean_correction_prediction(
    frame: pd.DataFrame,
    *,
    group_cols: list[str],
    fallback_group_cols: list[list[str]],
    min_history: int = MIN_HISTORY,
    min_group: int = MIN_GROUP,
) -> pd.DataFrame:
    ordered = frame.sort_values("target_date").reset_index(drop=True).copy()
    residual = (
        pd.to_numeric(ordered["target_tmax_c"], errors="coerce")
        - pd.to_numeric(ordered["forecast_max_c"], errors="coerce")
    ).to_numpy(dtype=float)
    official = pd.to_numeric(ordered["forecast_max_c"], errors="coerce").to_numpy(dtype=float)
    corrections: list[float] = []
    rows_used: list[int] = []

    for index, _row in ordered.iterrows():
        prior = ordered.iloc[:index].copy()
        valid = np.isfinite(residual[:index])
        prior = prior.loc[valid].copy()
        prior_residual = residual[:index][valid]
        if len(prior) < min_history:
            corrections.append(math.nan)
            rows_used.append(int(len(prior)))
            continue

        candidate_masks: list[pd.Series] = []
        all_groups = [group_cols, *fallback_group_cols, []]
        for cols in all_groups:
            if not cols:
                candidate_masks.append(pd.Series(True, index=prior.index))
                continue
            mask = pd.Series(True, index=prior.index)
            for col in cols:
                mask &= prior[col].eq(ordered.at[index, col])
            candidate_masks.append(mask)

        chosen = candidate_masks[-1]
        for mask in candidate_masks:
            if int(mask.sum()) >= min_group:
                chosen = mask
                break
        correction = float(np.mean(prior_residual[chosen.to_numpy()]))
        corrections.append(correction)
        rows_used.append(int(chosen.sum()))

    out = ordered.copy()
    out["past_correction_c"] = corrections
    out["past_rows_used"] = rows_used
    out["candidate_prediction_c"] = official + np.array(corrections, dtype=float)
    return out


def bucket_correction_prediction(
    frame: pd.DataFrame,
    *,
    feature: str,
    bins: int,
    pool_mode: str,
    season_conditioned: bool,
    min_history: int = MIN_HISTORY,
    min_group: int = MIN_GROUP,
) -> pd.DataFrame:
    ordered = frame.sort_values("target_date").reset_index(drop=True).copy()
    feature_values = pd.to_numeric(ordered[feature], errors="coerce").to_numpy(dtype=float)
    residual = (
        pd.to_numeric(ordered["target_tmax_c"], errors="coerce")
        - pd.to_numeric(ordered["forecast_max_c"], errors="coerce")
    ).to_numpy(dtype=float)
    official = pd.to_numeric(ordered["forecast_max_c"], errors="coerce").to_numpy(dtype=float)
    sources = ordered["forecast_source_family"].astype(str).to_numpy()
    seasons = ordered["season"].astype(str).to_numpy()
    corrections: list[float] = []
    rows_used: list[int] = []

    for index, current_value in enumerate(feature_values):
        if not np.isfinite(current_value):
            corrections.append(math.nan)
            rows_used.append(0)
            continue

        prior_values = feature_values[:index]
        prior_residual = residual[:index]
        prior_sources = sources[:index]
        prior_seasons = seasons[:index]
        valid = np.isfinite(prior_values) & np.isfinite(prior_residual)
        if pool_mode == "same_source":
            valid &= prior_sources == sources[index]
        prior_values = prior_values[valid]
        prior_residual = prior_residual[valid]
        prior_seasons = prior_seasons[valid]
        if len(prior_values) < min_history or len(np.unique(prior_values)) < bins:
            corrections.append(math.nan)
            rows_used.append(int(len(prior_values)))
            continue

        quantiles = np.unique(np.quantile(prior_values, np.linspace(0, 1, bins + 1)))
        if len(quantiles) <= 2:
            bucket_mask = prior_values == current_value
        else:
            bucket_index = int(np.searchsorted(quantiles[1:-1], current_value, side="right"))
            low = quantiles[bucket_index]
            high = quantiles[bucket_index + 1]
            if bucket_index == 0:
                bucket_mask = prior_values <= high
            elif bucket_index == len(quantiles) - 2:
                bucket_mask = prior_values >= low
            else:
                bucket_mask = (prior_values >= low) & (prior_values <= high)

        chosen = bucket_mask.copy()
        if season_conditioned:
            season_mask = chosen & (prior_seasons == seasons[index])
            if int(season_mask.sum()) >= min_group:
                chosen = season_mask
        if int(chosen.sum()) < min_group:
            chosen = bucket_mask
        if int(chosen.sum()) < min_group:
            chosen = np.ones(len(prior_values), dtype=bool)

        correction = float(np.mean(prior_residual[chosen]))
        corrections.append(correction)
        rows_used.append(int(chosen.sum()))

    out = ordered.copy()
    out["candidate_feature"] = feature
    out["past_correction_c"] = corrections
    out["past_rows_used"] = rows_used
    out["candidate_prediction_c"] = official + np.array(corrections, dtype=float)
    return out


def load_candidate_features(frame: pd.DataFrame) -> list[str]:
    if FEATURE_CONDITIONING_PATH.exists():
        ranked = pd.read_csv(FEATURE_CONDITIONING_PATH)
        features = [
            feature
            for feature in ranked["feature"].dropna().astype(str).to_list()
            if feature in frame.columns
        ]
    else:
        features = []
    unique: list[str] = []
    for feature in features:
        if feature not in unique:
            unique.append(feature)
        if len(unique) >= MAX_FEATURES:
            break
    return unique


def build_official_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    features = load_features()
    champion = load_champion_predictions()
    frame = build_analysis_frame(features, champion)
    require_no_confirmation_dates(frame["target_date"], context="official anchor ladder feature frame")
    official = load_official_forecasts(frame)
    require_no_confirmation_dates(official["target_date"], context="official anchor ladder official rows")
    return frame, official


def run_ladder(frame: pd.DataFrame, official: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_rows: list[dict[str, object]] = []
    prediction_tables: list[pd.DataFrame] = []

    group_specs = [
        (
            "source_global",
            ["forecast_source_family"],
            [[]],
            "past mean target-minus-official residual for same source",
        ),
        (
            "source_season",
            ["forecast_source_family", "season"],
            [["forecast_source_family"], []],
            "past same-source same-season residual with fallback to source/global",
        ),
        (
            "source_month",
            ["forecast_source_family", "month"],
            [["forecast_source_family", "season"], ["forecast_source_family"], []],
            "past same-source same-month residual with season/source/global fallback",
        ),
    ]
    for candidate_id, group_cols, fallback_cols, detail in group_specs:
        predictions = mean_correction_prediction(
            official,
            group_cols=group_cols,
            fallback_group_cols=fallback_cols,
        )
        predictions["candidate_id"] = candidate_id
        predictions["mechanism"] = "past_group_mean"
        candidate_rows.append(
            prediction_score_row(
                predictions,
                prediction_col="candidate_prediction_c",
                candidate_id=candidate_id,
                mechanism="past_group_mean",
                detail=detail,
            )
        )
        prediction_tables.append(predictions)

    candidate_features = load_candidate_features(frame)
    enriched = official.merge(
        frame[["target_date", *candidate_features]],
        on="target_date",
        how="left",
    )
    for feature in candidate_features:
        for bins in (3, 5):
            for pool_mode in ("same_source", "pooled_prior"):
                for season_conditioned in (False, True):
                    candidate_id = f"{pool_mode}_{feature}_q{bins}_season_{int(season_conditioned)}"
                    predictions = bucket_correction_prediction(
                        enriched,
                        feature=feature,
                        bins=bins,
                        pool_mode=pool_mode,
                        season_conditioned=season_conditioned,
                    )
                    predictions["candidate_id"] = candidate_id
                    predictions["mechanism"] = "past_feature_bucket"
                    predictions["pool_mode"] = pool_mode
                    predictions["bins"] = bins
                    predictions["season_conditioned"] = season_conditioned
                    candidate_rows.append(
                        prediction_score_row(
                            predictions,
                            prediction_col="candidate_prediction_c",
                            candidate_id=candidate_id,
                            mechanism="past_feature_bucket",
                            detail=f"{feature}; bins={bins}; pool={pool_mode}; season={season_conditioned}",
                        )
                    )
                    prediction_tables.append(predictions)

    scoreboard = pd.DataFrame(candidate_rows).sort_values(
        ["delta_vs_official_same_rows", "mae"],
        na_position="last",
    )
    all_predictions = pd.concat(prediction_tables, ignore_index=True, sort=False)
    return scoreboard.reset_index(drop=True), all_predictions


def write_outputs(scoreboard: pd.DataFrame, predictions: pd.DataFrame, official: pd.DataFrame) -> None:
    folder = RESEARCH_ROOT / "0016_past_only_official_anchor_correction_ladder"
    artifacts = folder / "artifacts"
    write_csv(artifacts / "scoreboard.csv", scoreboard)
    write_csv(artifacts / "all_candidate_predictions.csv", predictions)
    write_csv(artifacts / "official_rows_used.csv", official)

    best = scoreboard.head(1).to_dict("records")
    best_text = "No scoreable candidate was produced."
    if best:
        row = best[0]
        best_text = (
            f"Best candidate: `{row['candidate_id']}` with MAE `{row['mae']:.4f}` "
            f"versus same-row official MAE `{row['official_same_rows_mae']:.4f}` "
            f"(delta `{row['delta_vs_official_same_rows']:.4f}`) over `{int(row['n'])}` rows."
        )

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

    text = f"""# Past-Only Official Anchor Correction Ladder

Generated: `{now_utc()}`

## What Was Tested

This insight tests whether the official HKO forecast can be corrected with strictly past-only residual information. Every target date is processed in chronological order. The correction for a row is computed only from earlier rows, never from the current row or any later row.

Candidate families:

- source-level expanding mean bias;
- source + season/month expanding mean bias;
- feature-bucket residual correction using top official-error features from `0013_official_forecast_feature_conditioning`;
- same-source and pooled-prior variants.

## Leakage Contract

- Official forecasts are already selected by latest issue no later than `T-1 15:00 HKT`.
- Target labels from `{CONFIRMATION_START.date()}` onward are blocked.
- Bucket thresholds are fitted from prior rows only for each target date.
- This is not a promoted model. It is a correction screen for the next formal experiment.

## Source Coverage

{markdown_table(source_coverage, max_rows=10)}

## Main Result

{best_text}

## Top Candidates

{markdown_table(scoreboard.head(30), max_rows=30)}

## Interpretation

If this ladder shows only small improvements, the next step is not arbitrary feature expansion. The next step is stronger regime identification, better official forecast vintage continuity, and fold-local weighting between official, station-network, and upper-air experts.
"""
    write_text(folder / "README.md", text)


def update_master_index(manifest: dict[str, object]) -> None:
    index_path = RESEARCH_ROOT / "README.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# HKG Tmax Data Analysis\n"
    marker = "\n## Official-Anchor Correction Ladder\n"
    base = existing.split(marker)[0].rstrip()
    text = f"""{base}
{marker}
Generated: `{manifest['generated_at_utc']}`

Additional folder created by `scripts/run_hkg_t24_official_anchor_correction_ladder.py`:

- `0016_past_only_official_anchor_correction_ladder`: source/season/month/feature-bucket corrections around the official forecast anchor, all fitted from prior rows only.

Key counts:

| Metric | Value |
|---|---:|
| Official rows | {manifest['official_rows']} |
| Candidate rows scored | {manifest['candidate_rows']} |
| Prediction rows written | {manifest['prediction_rows']} |
| Best candidate MAE | {manifest['best_mae']} |
| Best delta vs same-row official | {manifest['best_delta_vs_official']} |

Leakage contract: all correction estimates use only rows earlier than the scored target date, and confirmation labels from `{CONFIRMATION_START.date()}` onward remain blocked.
"""
    write_text(index_path, text)


def run() -> dict[str, object]:
    frame, official = build_official_frame()
    scoreboard, predictions = run_ladder(frame, official)
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
        "folder": "0016_past_only_official_anchor_correction_ladder",
    }
    write_json(RESEARCH_ROOT / "official_anchor_correction_ladder_manifest.json", manifest)
    update_master_index(manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Run past-only official anchor correction ladder.").parse_args()


def main() -> None:
    parse_args()
    manifest = run()
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
